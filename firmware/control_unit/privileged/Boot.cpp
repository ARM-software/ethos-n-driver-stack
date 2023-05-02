//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "Cmsis.hpp"
#include "HardwareHelpers.hpp"

#include <common/FirmwareApi.hpp>
#include <common/TaskSvc.hpp>
#include <unprivileged/Task.hpp>

#include <scylla_addr_fields.h>
#include <scylla_regs.h>

#include <cstdint>

using namespace ethosn::control_unit;

// Include space for both privileged and unprivileged stacks in the firmware binary,
// so that the kernel module and TF-A don't need to allocate these separately.
// These are placed in specially aligned locations by the scatter file.
// Note that for the dual core carveout case, because we don't compile with
// position-independent code, these will always refer to the first core's stack
// even when running on the second core!
// This is fine because the privileged stack actually used is the one provided in the
// vector table at boot time and is filled in correctly by the kernel (different for
// each core), and the unprivileged stack is set up based on this (see bottomOfTaskStack
// initialisation).
// We avoid position-independent code because it makes the compiled firmware code
// more complicated and slower, and the compiler generates code which offsets
// global variables with function addresses and this causes problems when
// resetting the NPU as the same addresses get offset again. We also don't need
// it because we use the same code for both cores, and just duplicate the stacks
// and vector tables.
__attribute__((used, section("UNPRIV_STACK"))) char g_UnprivilegedStack[0x40000];
__attribute__((used, section("PRIV_STACK"))) char g_PrivilegedStack[0x40000];

extern "C" __attribute__((noreturn)) void main(void);

namespace ethosn
{
namespace control_unit
{

extern void EnableMpu(size_t mailboxSize, size_t commandStreamSize);    // Defined in Mpu.cpp
extern void SvcHandler(unsigned number, unsigned* args);
extern void PopulateTaskConfig(ethosn::control_unit::TaskConfig* config);

}    // namespace control_unit
}    // namespace ethosn

#if CONTROL_UNIT_DEBUG_MONITOR
extern "C" void mriExceptionHandler(void);
#endif

extern "C" {
__attribute__((naked, used)) void __user_setup_stackheap()
{
    __ASM(
        // Stack stored in SP
        // Leave the stack pointer as it was

        // Heap pointer in r0 and size in r2
        // There is no heap so set to zero
        "mov  r0, #0;"
        "mov  r2, #0;"

        // Return
        "bx   lr;");
}

__attribute__((interrupt("IRQ"), naked)) static void SvcTopHandler()
{
    __ASM volatile(
        // Use LR to determine the caller's stack
        // MSP: 9 (1001) AND 4 (100) = 0, Z == 1, EQ
        // PSP: D (1101) AND 4 (100) = 4, Z == 0, NE
        "tst    lr, #4;"
        "ite    eq;"
        "mrseq  r1, msp;"
        "mrsne  r1, psp;"
        // Traverse the stack to get the return address
        "ldr    r0, [r1, #24];"
        // Traverse back two bytes from the return address to get the SVC
        // instruction and read the lowest byte to get the SVC number
        "ldrb   r0, [r0, #-2];"
        // r0: SVC number
        // r1: Stack pointer to SVC arguments
        // Call C SVC handler
        "push   {lr};"
        "bl     %[SvcHandler];"
        "pop    {lr};"
        "bx     lr;" ::[SvcHandler] "i"(&SvcHandler));
}

__attribute__((interrupt("IRQ"), naked)) static void PendSvHandler()
{
    __ASM volatile(
        // Use LR to determine the caller's stack and privileges
        // MSP: 9 (1001) AND 4 (100) = 0, Z == 1, EQ
        // PSP: D (1101) AND 4 (100) = 4, Z == 0, NE
        // Store current task's context to its stack
        "tst    lr, #4;"
        "ite    eq;"
        "mrseq  r0, msp;"
        "mrsne  r0, psp;"
        "mov    r2, r0;"
        "stmdb  r0!, {r4-r11};"
        "ite    eq;"
        "msreq  msp, r0;"
        "msrne  psp, r0;"

        // Load next task's context from its stack
        "ite    eq;"
        "mrseq  r0, psp;"
        "mrsne  r0, msp;"
        "ldmfd  r0!, {r4-r11};"
        "ite    eq;"
        "msreq  psp, r0;"
        "msrne  msp, r0;"

        // Only R4-R11 is loaded here because as specified by AAPCS, the
        // hardware will load the rest of the registers from the task's stack
        // when leaving the PendSV handler.

        // Update privileges
        // Determined by the first bit in the control register:
        // 0: privileged
        // 1: unprivileged
        "mrs    r1, control;"
        "ite    eq;"
        "orreq  r1, r1, 0x1;"
        "bicne  r1, r1, 0x1;"
        "msr    control, r1;"

        // Update LR to return to the correct thread mode
        "ite    eq;"
        "moveq  lr,0xFFFFFFFD;"
        "movne  lr,0xFFFFFFF9;"

        // Get the SVC number if the call came from the unprivileged task
        // (See the SvcTopHandler for how the SVC number is extracted)
        "itte     ne;"
        "ldrne    r1, [r2, #24];"
        "ldrbne   r1, [r1, #-2];"
        "moveq    r1, #0;"

        // If this was a task switch from the unprivileged task, take the task
        // message address and set it as return value in the privileged task.
        "cmp      r1, %[svc_num];"
        "itt      eq;"
        "ldreq    r2, [r2];"
        "streq    r2, [r0];"

        // Branch to new task
        "bx     lr;" ::[svc_num] "i"(ethosn::control_unit::TASK_SVC_TASK_SWITCH));
}
}

namespace
{
__attribute__((noreturn)) void Hang()
{
    // Raise error interrupt to inform host
    dl1_setirq_ext_r setReg;
    setReg.set_err(1);
    *reinterpret_cast<volatile uint32_t*>(TOP_REG(DL1_RP, DL1_SETIRQ_EXT)) = setReg.word;

    while (true)
    {
        // Without the following line, armclang may optimize away the infinite loop
        // because it'd be without side effects and thus undefined behaviour.
        __ASM volatile("");
    }
}

__attribute__((interrupt("IRQ"), noreturn, used)) void HangIrq()
{
    Hang();
}

// Must be naked to avoid pushing stuff onto the stack in its preamble, which affects us
// extracting useful data from the stack for debugging (IrqContext).
__attribute__((interrupt("IRQ"), noreturn, used, naked)) void FaultIrq()
{
    __ASM volatile(
        // Check which stack (MSP vs PSP) was being used when the exception occured, by checking the value of LR.
        // The hardware will set this to some magic values (see EXC_RETURN).
        // We need to know this so we know which stack to get the PC from.
        // MSP: 9 (1001) AND 4 (100) = 0, Z == 1, EQ
        // PSP: D (1101) AND 4 (100) = 4, Z == 0, NE
        "tst    lr, #4;"
        "ite    eq;"
        "mrseq  r1, msp;"
        "mrsne  r1, psp;"
        // Traverse the stack to get the return address
        "ldr    r0, [r1, #24];"
        // We've put the PC value into r0, and we now call the FaultIrqImpl function, which takes its
        // first (and only) argument in r0.
        "B FaultIrqImpl;");
}

extern "C" __attribute__((interrupt("IRQ"), noreturn, used)) void FaultIrqImpl(uint32_t pcFromIrqContext)
{
    using namespace ethosn::control_unit;

    // Fill in a "dump" struct which we send to the kernel via the GP registers,
    // so that it can print out some useful debugging information.
    // We build up the struct locally here rather than writing directly into the GPs,
    // as we're only allowed to write entire 32-bit words to the GP registers and all the bitfield
    // mangling doesn't honour that.
    ethosn_firmware_dump dump = {};

    // Set a "magic" number so that the kernel knows that we have filled in a dump struct
    // (sometimes when the kernel does a GP dump they might not have been filled in via this code).
    dump.magic = ETHOSN_FIRMWARE_DUMP_MAGIC;

    // IPSR non-reserved bits
    IPSR_Type ipsr;
    ipsr.w = __get_xPSR();
    // Store just the lowest 5 bits. We only have one additional interrupt, for a total of 17, so 5 bits is sufficient
    dump.ISR = ipsr.b.ISR & 0b11111;

    // CFSR non-reserved bits
    uint32_t cfsr = Fault::GetConfigurableFaultStatusRegister();

    dump.CFSR_MMFSR_MMARVALID = (cfsr & SCB_CFSR_MMARVALID_Msk) ? 1 : 0;
    dump.CFSR_MMFSR_MSTKERR   = (cfsr & SCB_CFSR_MSTKERR_Msk) ? 1 : 0;
    dump.CFSR_MMFSR_MUNSTKERR = (cfsr & SCB_CFSR_MUNSTKERR_Msk) ? 1 : 0;
    dump.CFSR_MMFSR_DACCVIOL  = (cfsr & SCB_CFSR_DACCVIOL_Msk) ? 1 : 0;
    dump.CFSR_MMFSR_IACCVIOL  = (cfsr & SCB_CFSR_IACCVIOL_Msk) ? 1 : 0;

    dump.CFSR_BFSR_BFARVALID   = (cfsr & SCB_CFSR_BFARVALID_Msk) ? 1 : 0;
    dump.CFSR_BFSR_STKERR      = (cfsr & SCB_CFSR_STKERR_Msk) ? 1 : 0;
    dump.CFSR_BFSR_UNSTKERR    = (cfsr & SCB_CFSR_UNSTKERR_Msk) ? 1 : 0;
    dump.CFSR_BFSR_IMPRECISERR = (cfsr & SCB_CFSR_IMPRECISERR_Msk) ? 1 : 0;
    dump.CFSR_BFSR_PRECISERR   = (cfsr & SCB_CFSR_PRECISERR_Msk) ? 1 : 0;
    dump.CFSR_BFSR_IBUSERR     = (cfsr & SCB_CFSR_IBUSERR_Msk) ? 1 : 0;

    dump.CFSR_UFSR_DIVBYZERO  = (cfsr & SCB_CFSR_DIVBYZERO_Msk) ? 1 : 0;
    dump.CFSR_UFSR_UNALIGNED  = (cfsr & SCB_CFSR_UNALIGNED_Msk) ? 1 : 0;
    dump.CFSR_UFSR_NOCP       = (cfsr & SCB_CFSR_NOCP_Msk) ? 1 : 0;
    dump.CFSR_UFSR_INVPC      = (cfsr & SCB_CFSR_INVPC_Msk) ? 1 : 0;
    dump.CFSR_UFSR_INVSTATE   = (cfsr & SCB_CFSR_INVSTATE_Msk) ? 1 : 0;
    dump.CFSR_UFSR_UNDEFINSTR = (cfsr & SCB_CFSR_UNDEFINSTR_Msk) ? 1 : 0;

    // HFSR non-reserved bits
    uint32_t hfsr     = Fault::GetHardFaultStatusRegister();
    dump.HFSR_FORCED  = (hfsr & SCB_HFSR_FORCED_Msk) ? 1 : 0;
    dump.HFSR_VECTTBL = (hfsr & SCB_HFSR_VECTTBL_Msk) ? 1 : 0;

    // MMFAR
    dump.MMFAR = Fault::GetMemManageFaultAddressRegister();

    // BFAR
    dump.BFAR = Fault::GetBusFaultAddressRegister();

    // TOP_ERR_CAUSE non-reserved bits
    top_err_cause_r top_err_cause;
    top_err_cause.word = *reinterpret_cast<volatile uint32_t*>(TOP_REG(GLOBAL_RP, GLOBAL_TOP_ERR_CAUSE));
    dump.TOP_ERR_CAUSE_ENGINE_RAM_CORRECTABLE_ERR     = top_err_cause.bits.engine_ram_correctable_err;
    dump.TOP_ERR_CAUSE_ENGINE_RAM_UNCORRECTABLE_ERR   = top_err_cause.bits.engine_ram_uncorrectable_err;
    dump.TOP_ERR_CAUSE_TOP_TOLERABLE_RAM_ERR          = top_err_cause.bits.top_tolerable_ram_err;
    dump.TOP_ERR_CAUSE_TOP_RECOVERABLE_RAM_ERR        = top_err_cause.bits.top_recoverable_ram_err;
    dump.TOP_ERR_CAUSE_MCU_LOCKUP_ERR                 = top_err_cause.bits.mcu_lockup_err;
    dump.TOP_ERR_CAUSE_MCU_INSTR_ERR                  = top_err_cause.bits.mcu_instr_err;
    dump.TOP_ERR_CAUSE_MCU_DATA_READ_ERR              = top_err_cause.bits.mcu_data_read_err;
    dump.TOP_ERR_CAUSE_MCU_DATA_WRITE_ERR             = top_err_cause.bits.mcu_data_write_err;
    dump.TOP_ERR_CAUSE_DMA_READ_ERR                   = top_err_cause.bits.dma_read_err;
    dump.TOP_ERR_CAUSE_DMA_WRITE_ERR                  = top_err_cause.bits.dma_write_err;
    dump.TOP_ERR_CAUSE_STASH_TRANSLATION_ERR          = top_err_cause.bits.stash_translation_err;
    dump.TOP_ERR_CAUSE_DMA_QUEUE_PROGRAMMING_ERR      = top_err_cause.bits.dma_queue_programming_err;
    dump.TOP_ERR_CAUSE_PWRCTLR_ACTIVE_PROGRAMMING_ERR = top_err_cause.bits.pwrctlr_active_programming_err;
    dump.TOP_ERR_CAUSE_STASH_TRANS_PROGRAMMING_ERR    = top_err_cause.bits.stash_trans_programming_err;
    dump.TOP_ERR_CAUSE_TSU_EVENT_OVERFLOW_ERR         = top_err_cause.bits.tsu_event_overflow_err;
    dump.TOP_ERR_CAUSE_STRIPE_PROGRAMMING_ERR         = top_err_cause.bits.stripe_programming_err;
    dump.TOP_ERR_CAUSE_STRIPE_WRITE_WHILE_BUSY_ERR    = top_err_cause.bits.stripe_write_while_busy_err;
    dump.TOP_ERR_CAUSE_BLOCK_PROGRAMMING_ERR          = top_err_cause.bits.block_programming_err;
    dump.TOP_ERR_CAUSE_BLOCK_WRITE_WHILE_BUSY_ERR     = top_err_cause.bits.block_write_while_busy_err;
    dump.TOP_ERR_CAUSE_SHADOW_ERR                     = top_err_cause.bits.shadow_err;
    dump.TOP_ERR_CAUSE_ENGINE_FUNC_ERR                = top_err_cause.bits.engine_func_err;

    // TOP_ERR_ADDRESS
    top_err_address_r top_err_address;
    top_err_address.word         = *reinterpret_cast<volatile uint32_t*>(TOP_REG(GLOBAL_RP, GLOBAL_TOP_ERR_ADDRESS));
    dump.TOP_ERR_ADDRESS_ADDRESS = top_err_address.bits.address;
    dump.TOP_ERR_ADDRESS_BANK    = top_err_address.bits.bank;
    dump.TOP_ERR_ADDRESS_NCU_MCU_ICACHE_TAG   = top_err_address.bits.ncu_mcu_icache_tag;
    dump.TOP_ERR_ADDRESS_NCU_MCU_ICACHE_DATA  = top_err_address.bits.ncu_mcu_icache_data;
    dump.TOP_ERR_ADDRESS_NCU_MCU_DCACHE_TAG   = top_err_address.bits.ncu_mcu_dcache_tag;
    dump.TOP_ERR_ADDRESS_NCU_MCU_DCACHE_DATA  = top_err_address.bits.ncu_mcu_dcache_data;
    dump.TOP_ERR_ADDRESS_DFC_ROB              = top_err_address.bits.dfc_rob;
    dump.TOP_ERR_ADDRESS_DFC_COMPRESSOR_SIM   = top_err_address.bits.dfc_compressor_sim;
    dump.TOP_ERR_ADDRESS_DFC_COMPRESSOR_REM   = top_err_address.bits.dfc_compressor_rem;
    dump.TOP_ERR_ADDRESS_DFC_COMPRESSOR_UNARY = top_err_address.bits.dfc_compressor_unary;
    dump.TOP_ERR_ADDRESS_DFC_DECOMPRESSOR     = top_err_address.bits.dfc_decompressor;
    dump.TOP_ERR_ADDRESS_ERR_MULTI            = top_err_address.bits.err_multi;
    dump.TOP_ERR_ADDRESS_ERR_UNCORRECTED      = top_err_address.bits.err_uncorrected;

    // Each CE may have separate errors, but we don't have space to dump them all.
    // We just dump the details for the first CE that had an error.
    dl2_unit_count_r unit_count;
    unit_count.word       = *reinterpret_cast<volatile uint32_t*>(TOP_REG(DL2_RP, DL2_UNIT_COUNT));
    const uint32_t numCes = unit_count.get_quad_count() * unit_count.get_engines_per_quad();
    for (uint32_t ce = 0; ce < numCes; ++ce)
    {
        ce_err_cause_r ce_err_cause;
        ce_err_cause.word = *reinterpret_cast<volatile uint32_t*>(CE_REG(ce, CE_RP, CE_CE_ERR_CAUSE));
        if (ce_err_cause.word != 0)
        {
            if (dump.cesWithError == 0)
            {
                // This is the first CE with an error - fill in the details

                // CE_ERR_CAUSE non-reserved bits
                dump.CE_ERR_CAUSE_ENGINE_RAM_CORRECTABLE_ERR   = ce_err_cause.bits.engine_ram_correctable_err;
                dump.CE_ERR_CAUSE_ENGINE_RAM_UNCORRECTABLE_ERR = ce_err_cause.bits.engine_ram_uncorrectable_err;
                dump.CE_ERR_CAUSE_MCU_LOCKUP_ERR               = ce_err_cause.bits.mcu_lockup_err;
                dump.CE_ERR_CAUSE_MCU_INSTR_ERR                = ce_err_cause.bits.mcu_instr_err;
                dump.CE_ERR_CAUSE_MCU_DATA_READ_ERR            = ce_err_cause.bits.mcu_data_read_err;
                dump.CE_ERR_CAUSE_MCU_DATA_WRITE_ERR           = ce_err_cause.bits.mcu_data_write_err;
                dump.CE_ERR_CAUSE_UDMA_LOAD_ERR                = ce_err_cause.bits.udma_load_err;
                dump.CE_ERR_CAUSE_UDMA_STORE_ERR               = ce_err_cause.bits.udma_store_err;
                dump.CE_ERR_CAUSE_MCU_ILLEGAL_COPROC_ERR       = ce_err_cause.bits.mcu_illegal_coproc_err;
                dump.CE_ERR_CAUSE_UDMA_COLLISION_ERR           = ce_err_cause.bits.udma_collision_err;
                dump.CE_ERR_CAUSE_RF_RD_COLLISION_ERR          = ce_err_cause.bits.rf_rd_collision_err;
                dump.CE_ERR_CAUSE_RF_WR_COLLISION_ERR          = ce_err_cause.bits.rf_wr_collision_err;
                dump.CE_ERR_CAUSE_VE_DIV_0_ERR                 = ce_err_cause.bits.ve_div_0_err;
                dump.CE_ERR_CAUSE_PLE_LANE_ERR                 = ce_err_cause.bits.ple_lane_err;

                // CE_ERR_ADDRESS non-reserved bits
                ce_err_address_r ce_err_address;
                ce_err_address.word = *reinterpret_cast<volatile uint32_t*>(CE_REG(ce, CE_RP, CE_CE_ERR_ADDRESS));
                dump.CE_ERR_ADDRESS_ADDRESS         = ce_err_address.bits.address;
                dump.CE_ERR_ADDRESS_BANK            = ce_err_address.bits.bank;
                dump.CE_ERR_ADDRESS_DFC_EMC0        = ce_err_address.bits.dfc_emc0;
                dump.CE_ERR_ADDRESS_DFC_EMC1        = ce_err_address.bits.dfc_emc1;
                dump.CE_ERR_ADDRESS_DFC_EMC2        = ce_err_address.bits.dfc_emc2;
                dump.CE_ERR_ADDRESS_DFC_EMC3        = ce_err_address.bits.dfc_emc3;
                dump.CE_ERR_ADDRESS_MCE_OFM0        = ce_err_address.bits.mce_ofm0;
                dump.CE_ERR_ADDRESS_MCE_OFM1        = ce_err_address.bits.mce_ofm1;
                dump.CE_ERR_ADDRESS_MCE_OFM2        = ce_err_address.bits.mce_ofm2;
                dump.CE_ERR_ADDRESS_MCE_OFM3        = ce_err_address.bits.mce_ofm3;
                dump.CE_ERR_ADDRESS_PLE_INPUT0      = ce_err_address.bits.ple_input0;
                dump.CE_ERR_ADDRESS_PLE_INPUT1      = ce_err_address.bits.ple_input1;
                dump.CE_ERR_ADDRESS_PLE_INPUT2      = ce_err_address.bits.ple_input2;
                dump.CE_ERR_ADDRESS_PLE_INPUT3      = ce_err_address.bits.ple_input3;
                dump.CE_ERR_ADDRESS_PLE_OUTPUT      = ce_err_address.bits.ple_output;
                dump.CE_ERR_ADDRESS_PLE_MCU         = ce_err_address.bits.ple_mcu;
                dump.CE_ERR_ADDRESS_ERR_MULTI       = ce_err_address.bits.err_multi;
                dump.CE_ERR_ADDRESS_ERR_UNCORRECTED = ce_err_address.bits.err_uncorrected;
            }

            dump.cesWithError |= (1 << ce);
        }
    }

    // Program Counter which the NCU might have put onto the stack for us (depending on what the error is)
    dump.pc = pcFromIrqContext;

    // Copy dump struct to GPs, one word at a time
    static_assert(sizeof(ethosn_firmware_dump) == 32,
                  "ethosn_firmware_dump struct must fit GP regs exactly otherwise the loop below might be wrong");
    for (int w = 0; w < 8; ++w)
    {
        *reinterpret_cast<volatile uint32_t*>(TOP_REG(DL1_RP, DL1_GP0 + w * (DL1_GP1 - DL1_GP0))) =
            reinterpret_cast<uint32_t*>(&dump)[w];
    }

    // Raise an error interrupt to the host CPU
    Hang();
}

struct TaskContextFrame
{
    // Software context
    uint32_t r11;
    uint32_t r10;
    uint32_t r9;
    uint32_t r8;
    uint32_t r7;
    uint32_t r6;
    uint32_t r5;
    uint32_t r4;
    // Hardware context
    uint32_t r0;
    uint32_t r1;
    uint32_t r2;
    uint32_t r3;
    uint32_t r12;
    uint32_t lr;
    uint32_t pc;
    uint32_t psr;
};

#define STR(x) #x
#define RESET_REG(n) __ASM volatile("MOV " STR(r##n) ", #0" : : : STR(r##n))

__attribute__((section("BOOT"), noreturn, used)) void __start()
{
    using namespace ethosn::control_unit;

    // Initialise registers r0-r12 and LR(=r14)
    // They must have a valid value before being potentially pushed to stack by
    // C calling convention or by context saving in exception handling
    RESET_REG(0);
    RESET_REG(1);
    RESET_REG(2);
    RESET_REG(3);
    RESET_REG(4);
    RESET_REG(5);
    RESET_REG(6);
    RESET_REG(7);
    RESET_REG(8);
    RESET_REG(9);
    RESET_REG(10);
    RESET_REG(11);
    RESET_REG(12);
    RESET_REG(14);

    // The SVCall and PendSV interrupts are given the same priority so they can't preempt each
    // other. The SVCall and PendSV interrupts are also given the lowest priority (highest value) so
    // they are always handled last to avoid needing critical regions where interrupts needs to be
    // turned off.
    Interrupts::SetSVCallPriority(0xFFU);
    Interrupts::SetPendSVPriority(0xFFU);

    // Enable interrupt #0, which is element 16 in the vector table and is configured to trigger
    // when there is an error in the hardware (see SYSCTLR1).
    NVIC_EnableIRQ(IRQn_Type::Irq0);

    // Setup and prepare the unprivileged task stack (PSP)
    // Note that this calculation needs to perform an offset based on the stack pointer
    // value from the vector table, which will be different to the g_UnprivilegedStack address
    // for the second core in a dual-core carveout setup, because we don't compile with
    // position-independent code.
    uint32_t bottomOfPrivilegedStackActual = reinterpret_cast<volatile uint32_t*>(SCB->VTOR)[0];
    uint32_t bottomOfPrivilegedStackCompiled =
        reinterpret_cast<uint32_t>(&g_PrivilegedStack[0]) + sizeof(g_PrivilegedStack);
    uint32_t offset               = bottomOfPrivilegedStackActual - bottomOfPrivilegedStackCompiled;
    char* bottomOfTaskStack       = &g_UnprivilegedStack[0] + sizeof(g_UnprivilegedStack) + offset;
    struct TaskConfig* taskConfig = reinterpret_cast<struct TaskConfig*>(bottomOfTaskStack - sizeof(TaskConfig));
    PopulateTaskConfig(taskConfig);
    TaskContextFrame* taskContextFrame =
        reinterpret_cast<TaskContextFrame*>(reinterpret_cast<uint8_t*>(taskConfig) - sizeof(TaskContextFrame));
    // Initialize software context
    taskContextFrame->r11 = 0;
    taskContextFrame->r10 = 0;
    taskContextFrame->r9  = 0;
    taskContextFrame->r8  = 0;
    taskContextFrame->r7  = 0;
    taskContextFrame->r6  = 0;
    taskContextFrame->r5  = 0;
    taskContextFrame->r4  = 0;
    // Initialize hardware context
    taskContextFrame->r0  = reinterpret_cast<uint32_t>(taskConfig);
    taskContextFrame->r1  = 0;
    taskContextFrame->r2  = 0;
    taskContextFrame->r3  = 0;
    taskContextFrame->r12 = 0;
    taskContextFrame->lr  = 0x0;    // Important for GDB to detect bottom of callstack
    taskContextFrame->pc  = reinterpret_cast<uint32_t>(&Task);
    taskContextFrame->psr = 0x01000000;
    // Set PSP stack pointer
    __ASM volatile("msr psp, %0;" : : "r"(taskContextFrame));

    // Caches
    Cache::IEnable();
    Cache::DEnable();

    // The mailbox and command stream region sizes are configurable by the kernel module,
    // and communicated to us via GP registers.
    const uint32_t mailboxSize       = *reinterpret_cast<volatile uint32_t*>(TOP_REG(DL1_RP, GP_MAILBOX_SIZE));
    const uint32_t commandStreamSize = *reinterpret_cast<volatile uint32_t*>(TOP_REG(DL1_RP, GP_COMMAND_STREAM_SIZE));
    EnableMpu(mailboxSize, commandStreamSize);

    // Call into MainHardware.cpp.
    // Note that we don't call __main() (which is the libcxx startup sequence) because this performs
    // a bunch of initialisation which we don't need or want because it tries to zero out some memory which is
    // read-only and will already be zeroed in the firmware binary.
    main();
}

}    // namespace

using ExecFuncPtr = void (*)();

#if CONTROL_UNIT_DEBUG_MONITOR
constexpr ExecFuncPtr FaultHandler = mriExceptionHandler;
constexpr ExecFuncPtr DebugHandler = mriExceptionHandler;
#else
constexpr ExecFuncPtr FaultHandler = &FaultIrq;
constexpr ExecFuncPtr DebugHandler = &HangIrq;
#endif

__attribute__((used, section("VECTOR_TABLE"))) ExecFuncPtr g_VectorTable[17] = {
    // Note that for dual core carveout, the initial stack pointer here is only valid for the first core.
    // The second core has this value overwritten by the kernel module before booting the firmware.
    reinterpret_cast<ExecFuncPtr>(&g_PrivilegedStack[0] + sizeof(g_PrivilegedStack)),    // Initial stack pointer
    &__start,                                                                            // Initial program counter
    &HangIrq,                                                                            // NMIException
    FaultHandler,                                                                        // HardFaultException
    FaultHandler,                                                                        // MemManageException
    FaultHandler,                                                                        // BusFaultException
    FaultHandler,                                                                        // UsageFaultException
    0,                                                                                   // Reserved
    0,                                                                                   // Reserved
    0,                                                                                   // Reserved
    0,                                                                                   // Reserved
    &SvcTopHandler,                                                                      // SVCHandler
    DebugHandler,                                                                        // DebugMonitor
    0,                                                                                   // Reserved
    &PendSvHandler,                                                                      // PendSV
    &HangIrq,                                                                            // SysTickHandler
    // The NCU MCU has a single interrupt, which we configure to trigger when there is a hardware error (see SYSCTLR1
    FaultHandler,    // First interrupt
};
