//
// Copyright © 2018-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#include "include/ethosn_ple/hw.h"
#include "include/ethosn_ple/utils.h"

#include <ncu_ple_interface_def.h>

#pragma clang section bss = "BOOT_BSS" data = "BOOT_DATA" rodata = "BOOT_RODATA" text = "BOOT_TEXT"

using ExecFuncPtr = void (*)();

// Exception context for the MPU
struct IrqContext
{
    uint32_t r0;
    uint32_t r1;
    uint32_t r2;
    uint32_t r3;
    uint32_t r12;
    uint32_t lr;
    uint32_t pc;
    uint32_t xPsr;
};

namespace
{
__STATIC_FORCEINLINE uint32_t GetPcInIrqContext()
{
    uint32_t pc = 0;
    __ASM volatile("LDR %[pc], [sp,%[OFFSET]]" : [pc] "=r"(pc) : [OFFSET] "I"(offsetof(IrqContext, pc)));
    return pc;
}

template <typename T>
__STATIC_FORCEINLINE void SetPcInIrqContext(const T pc)
{
    static_assert(sizeof(T) == sizeof(uint32_t), "");
    static_assert(alignof(T) == alignof(uint32_t), "");

    __ASM volatile("STR %[pc], [sp,%[OFFSET]]" : : [pc] "r"(pc), [OFFSET] "I"(offsetof(IrqContext, pc)));
}

template <unsigned I>
__STATIC_FORCEINLINE void ResetReg()
{
    __ASM volatile("MOV r%c0, #0"
                   : /* No outputs. */
                   : "n"(I)
                   : "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r8", "r9", "r10", "r11", "r12");
}

STATIC_LOOP_FN_WRAPPER(ResetRegFn, ResetReg);
}    // namespace

__attribute__((used, section("STACK"))) char g_Stack[STACK_SIZE];

extern "C" {
__NO_RETURN void main();

__NO_RETURN void __start()
{
    using namespace static_loop;

    // Initialise registers r0-r12
    // They must have a valid value before being potentially pushed to stack by
    // C calling convention or by context saving in exception handling.
    // The link register (LR/R14) is initialized in the call to main().
    For<Range<0, 13>>::Invoke(ResetRegFn{});

    __set_MSPLIM(reinterpret_cast<uint32_t>(&g_Stack));

    // Set CPACR bits 15-0 to enable CP0-CP7, and clear bits 20-23 to disable CP10-CP11
    SCB->CPACR = 0xFFFF;

    // Enable hard, bus, mem and usage fault detection in SHCSR, bits 16-18.
    // Enable stkof, bf, div_0_trp, unalign_trp and usersetm bits in CCR
    SCB->SHCSR =
        _VAL2FLD(SCB_SHCSR_USGFAULTENA, 1) | _VAL2FLD(SCB_SHCSR_BUSFAULTENA, 1) | _VAL2FLD(SCB_SHCSR_MEMFAULTENA, 1);

    SCB->CCR = _VAL2FLD(SCB_CCR_USERSETMPEND, 1) | _VAL2FLD(SCB_CCR_UNALIGN_TRP, 1) | _VAL2FLD(SCB_CCR_DIV_0_TRP, 1) |
               _VAL2FLD(SCB_CCR_BFHFNMIGN, 0) | _VAL2FLD(SCB_CCR_STKOFHFNMIGN, 0);

    main();
}

__NO_RETURN void __reset()
{
    // Even though SP is initialised automatically on first boot, we need to reset it manually on subsequent
    // resets through NmiHandler().
    // The bottom of the stack (which grows up) is at the end of SRAM
    __ASM volatile("MOV sp, %0" : : "r"(SRAM_SIZE));

    // We may be running a new kernel now, which has a different stack size
    __set_MSPLIM(reinterpret_cast<uint32_t>(&g_Stack));

    main();
}

void NmiHandler()
{
    SetPcInIrqContext(&__reset);
}

__NO_RETURN void FaultIrq()
{
    ncu_ple_interface::PleMsg::FaultInfo faultInfo;

    faultInfo.cfsr  = SCB->CFSR;
    faultInfo.pc    = GetPcInIrqContext();
    faultInfo.shcsr = SCB->SHCSR;

    // Clear bits in MMFSR, BFSR, UFSR
    // Clear bits in HFSR
    // Reset SHCSR
    SCB->CFSR = faultInfo.cfsr;
    // cppcheck-suppress selfAssignment
    SCB->HFSR = SCB->HFSR;

    auto& msg = *reinterpret_cast<volatile ncu_ple_interface::PleMsg*>(PLE_REG(CE_RP, CE_PLE_SCRATCH0));

    WriteToRegisters(&msg.type, ncu_ple_interface::PleMsg::FaultInfo::type);
    WriteToRegisters(&msg.faultInfo, faultInfo);

    __SEV();

    Hang();
}

__NO_RETURN void HangIrq()
{
    Hang();
}

const ExecFuncPtr g_InitVtor[] __attribute__((section("VECTOR_TABLE"), used)) = {
    reinterpret_cast<ExecFuncPtr>(
        SRAM_SIZE),    // Initial SP is at the bottom of the stack (which grows up) is at the end of SRAM
    &__start,          // Initial PC, set to entry point
    &NmiHandler,       // NMIException
    &FaultIrq,         // HardFaultException
    &FaultIrq,         // MemManageException
    &FaultIrq,         // BusFaultException
    &FaultIrq,         // UsageFaultException
    0,                 // Reserved
    0,                 // Reserved
    0,                 // Reserved
    0,                 // Reserved
    &HangIrq,          // SVCHandler
    &HangIrq,          // DebugMonitor
    0,                 // Reserved
    &HangIrq,          // PendSVC
    &HangIrq,          // SysTickHandler

    /* Configurable interrupts start here...*/

    &HangIrq,    // Irq0Handler
};
}
