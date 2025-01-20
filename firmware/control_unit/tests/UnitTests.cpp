//
// Copyright © 2018-2025 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "ComparisonUtils.hpp"

#include "../unprivileged/Firmware.hpp"

#include <common/FirmwareApi.hpp>
#include <common/Optimize.hpp>
#include <model/LoggingHal.hpp>
#include <model/ModelHal.hpp>
#include <model/UscriptHal.hpp>
#include <ncu_ple_interface_def.h>

#include <catch.hpp>

#include <ethosn_command_stream/CommandStreamBuilder.hpp>
#include <ethosn_command_stream/PleKernelIds.hpp>
#include <scylla_addr_fields.h>
#include <scylla_regs.h>

#include <chrono>
#include <cstring>
#include <iostream>

using namespace ethosn;
using namespace control_unit;

constexpr uint32_t tsuEventMaskRef = 0x000002ac;

namespace
{

std::string GetArchName()
{
    std::stringstream name;

    name << NPU_ARCH_VERSION_MAJOR << '.' << NPU_ARCH_VERSION_MINOR << '.' << NPU_ARCH_VERSION_PATCH;

    return name.str();
}

}    // namespace

/**
 * @brief   UnitTest to write a register and read back the same register using the Model backend
 */
TEST_CASE("ModelHal_RegReadWrite")
{
    ModelHal model;
    Firmware<ModelHal> fw(model, 0);

    uint32_t regCheck = 0;
    model.WriteReg(TOP_REG(TSU_RP, TSU_TSU_CONTROL), 12);
    regCheck = model.ReadReg(TOP_REG(TSU_RP, TSU_TSU_CONTROL));

    REQUIRE(regCheck == 12);
}

/**
 * @brief   UnitTest of WaitForEvent which sets the mask register and spawns a thread which Waits for the event.
 *          The mask register gets unset after 1 second from the main thread and the WFE in the second thread should
 *          return
 */
TEST_CASE("ModelHal_WaitForEvent")
{
    ModelHal model;
    Firmware<ModelHal> fw(model, 0);

    // Enable all CE units
    ce_enables_r ceEnables;
    ceEnables.set_ce_enable(1);
    ceEnables.set_mce_enable(1);
    ceEnables.set_mac_enable(255);
    model.WriteReg(TOP_REG(CE_RP, CE_CE_ENABLES), ceEnables.word);

    // Wait for an event - should timeout as no event has been generated
    // Temporarily disable asserts otherwise this will fire when testing a deliberate hang
#if defined(CONTROL_UNIT_ASSERTS)
    auto assertCallbackBackup                     = ethosn::control_unit::utils::g_AssertCallback;
    ethosn::control_unit::utils::g_AssertCallback = nullptr;
#endif
    {
        INFO("WaitForEvent has returned prematurely");
        auto startTime = std::chrono::high_resolution_clock::now();
        model.WaitForEventsWithTimeout(1000);
        REQUIRE((std::chrono::high_resolution_clock::now() - startTime) > std::chrono::milliseconds(500));
    }

    // Mask everything
    tsu_event_msk_r maskReg(0);
    model.WriteReg(TOP_REG(TSU_RP, TSU_TSU_EVENT_MSK), maskReg.word);

    // Set proper EMCs count
    dma_emcs_r dmaEngines;
    dmaEngines.set_emcs((1U << model.NumEmcs()) - 1);
    model.WriteReg(TOP_REG(DMA_RP, DMA_DMA_EMCS), dmaEngines.word);

    // Do something that will trigger an event. In this case a DMA.
    dma_rd_cmd_r rdCmd;
    rdCmd.set_format(dma_format_read_t::BROADCAST);
    model.WriteReg(TOP_REG(DMA_RP, DMA_DMA_RD_CMD), rdCmd.word);

    // Wait for it - it should either timeout as everything is masked or it may have returned spuriously.
    // Please note that TSU_EVENT register should be updated regardless of the mask.
    {
        INFO("Event has not been delivered");
        model.WaitForEventsWithTimeout(1000);
        REQUIRE(model.ReadReg(TOP_REG(TSU_RP, TSU_TSU_EVENT)) == 0x80);
    }

    // Unmask the event
    maskReg.set_dma_done_mask(event_mask_t::ENABLED);
    model.WriteReg(TOP_REG(TSU_RP, TSU_TSU_EVENT_MSK), maskReg.word);

    // Create another fake event
    model.WriteReg(TOP_REG(DMA_RP, DMA_DMA_RD_CMD), rdCmd.word);

    // Wait for it - it should return immediately as the event is no longer masked and the event should be visible.
    {
        INFO("WaitForEvent has not finished within the timeout or the event has not been triggered");
        model.WaitForEventsWithTimeout(1000);
        REQUIRE(tsu_event_r(model.ReadReg(TOP_REG(TSU_RP, TSU_TSU_EVENT))).get_dma_done() == event_t::TRIGGERED);
    }

#if defined(CONTROL_UNIT_ASSERTS)
    ethosn::control_unit::utils::g_AssertCallback = assertCallbackBackup;
#endif
}

/**
 * @brief   UnitTest to dump useful registers
 */
TEST_CASE("LogUsefulRegisters")
{
    LoggingHal::Options options;
    options.m_EthosNVariant = LoggingHal::Options::EthosNVariant::N78_1TOPS_2PLE_RATIO;
    LoggingHal loggingHal(options);
    Firmware<LoggingHal> fw(loggingHal, 0);

    control_unit::utils::LogUsefulRegisters(loggingHal);

    static const std::vector<LoggingHal::Entry> golden{
        { LoggingHal::Entry::ReadReg, TOP_REG(DL2_RP, DL2_PWRCTLR), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(DMA_RP, DMA_DMA_CHANNELS), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(DMA_RP, DMA_DMA_COMP_CONFIG0), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(DMA_RP, DMA_DMA_EMCS), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(DMA_RP, DMA_DMA_RD_CMD), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(DMA_RP, DMA_DMA_STRIDE0), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(DMA_RP, DMA_DMA_STRIDE1), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(DMA_RP, DMA_DMA_TOTAL_BYTES), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(DMA_RP, DMA_DMA_WR_CMD), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(DMA_RP, DMA_DRAM_ADDR_H), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(DMA_RP, DMA_DRAM_ADDR_L), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(DMA_RP, DMA_SRAM_ADDR), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(GLOBAL_RP, GLOBAL_BLOCK_BANK_CONFIG), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(GLOBAL_RP, GLOBAL_PLE_MCEIF_CONFIG), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(GLOBAL_RP, GLOBAL_STRIPE_BANK_CONFIG), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(GLOBAL_RP, GLOBAL_STRIPE_BANK_CONTROL), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(PMU_RP, PMU_PMCNTENCLR), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(PMU_RP, PMU_PMCR), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(PMU_RP, PMU_PMINTENCLR), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(PMU_RP, PMU_PMOVSCLR), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(STRIPE_RP, CE_STRIPE_ACTIVATION_CONFIG), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(STRIPE_RP, CE_STRIPE_CE_CONTROL), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(STRIPE_RP, CE_STRIPE_DEPTHWISE_CONTROL), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(STRIPE_RP, CE_STRIPE_FILTER), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(STRIPE_RP, CE_STRIPE_IFM_BOTTOM_SLOTS), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(STRIPE_RP, CE_STRIPE_IFM_CONFIG1), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(STRIPE_RP, CE_STRIPE_IFM_CONFIG2_IG0), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(STRIPE_RP, CE_STRIPE_IFM_DEFAULT_SLOT_SIZE), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(STRIPE_RP, CE_STRIPE_IFM_MID_SLOTS), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(STRIPE_RP, CE_STRIPE_IFM_PAD0_IG0), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(STRIPE_RP, CE_STRIPE_IFM_PAD1_IG0), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(STRIPE_RP, CE_STRIPE_IFM_PAD2_IG0), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(STRIPE_RP, CE_STRIPE_IFM_PAD3_IG0), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(STRIPE_RP, CE_STRIPE_IFM_ROW_STRIDE), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(STRIPE_RP, CE_STRIPE_IFM_SLOT_BASE_ADDRESS_IG0), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(STRIPE_RP, CE_STRIPE_IFM_SLOT_PAD_CONFIG), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(STRIPE_RP, CE_STRIPE_IFM_SLOT_STRIDE), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(STRIPE_RP, CE_STRIPE_IFM_TOP_SLOTS), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(STRIPE_RP, CE_STRIPE_IFM_ZERO_POINT), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(STRIPE_RP, CE_STRIPE_MUL_ENABLE_OG0), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(STRIPE_RP, CE_STRIPE_OFM_CONFIG), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(STRIPE_RP, CE_STRIPE_OFM_STRIPE_SIZE), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(STRIPE_RP, CE_STRIPE_STRIPE_BLOCK_CONFIG), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(STRIPE_RP, CE_STRIPE_VP_CONTROL), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(STRIPE_RP, CE_STRIPE_WEIGHT_BASE_ADDR_OG0), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(STRIPE_RP, CE_STRIPE_WIDE_KERNEL_CONTROL), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(STRIPE_RP, CE_STRIPE_WIDE_KERNEL_OFFSET), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(TSU_RP, TSU_TSU_CONTROL), 0x0 },
        { LoggingHal::Entry::ReadReg, TOP_REG(TSU_RP, TSU_TSU_EVENT_MSK), 0x0 },
        // Check only the first engine
        { LoggingHal::Entry::ReadReg, CE_REG(0, CE_RP, CE_CE_ENABLES), 0x0 },
        { LoggingHal::Entry::ReadReg, CE_REG(0, CE_RP, CE_PLE_CONTROL_0), 0x1 },
        { LoggingHal::Entry::ReadReg, CE_REG(0, CE_RP, CE_PLE_CONTROL_1), 0x0 },
        { LoggingHal::Entry::ReadReg, CE_REG(0, CE_RP, CE_PLE_SCRATCH5), 0x0 },
        { LoggingHal::Entry::ReadReg, CE_REG(0, CE_RP, CE_PLE_SCRATCH7), 0x0 },
        { LoggingHal::Entry::ReadReg, CE_REG(0, CE_RP, CE_PLE_SETIRQ), 0x0 },
        { LoggingHal::Entry::ReadReg, CE_REG(0, CE_RP, CE_PLE_UDMA_LOAD_COMMAND), 0x0 },
        { LoggingHal::Entry::ReadReg, CE_REG(0, CE_RP, CE_PLE_UDMA_LOAD_PARAMETERS), 0x0 },
    };

    RequireLoggingHalEntriesContainsInOrder(golden, loggingHal.GetEntries());
}

/**
 * @brief   UnitTest the generated uScript file using the Uscript proxy, using the Model backend
 */
TEST_CASE("UscriptHal_ModelHal_Ufile")
{
    std::string uName = "uscript_ufile.txt";
    ModelHal model;
    UscriptHal<ModelHal> proxy(model, uName.c_str(), true);

    proxy.WriteReg(TOP_REG(TSU_RP, TSU_TSU_CONTROL), 12);

    std::string expected = "ARCH " + GetArchName() +
                           "\n"
                           "PRODUCT N78" +
                           "\n"
                           "RESET\n"
                           "WRITEREG TSU.TSU_CONTROL 0000000c\n";
    // Binary mode is required to avoid inserting \r characters on Windows
    std::ifstream uFile(uName, std::ios::binary);
    std::ostringstream actualStream;
    actualStream << uFile.rdbuf();
    std::string actual = actualStream.str();

    REQUIRE(actual == expected);
}

TEST_CASE("UscriptHal_LoadMem")
{
    std::string uName = "uscript_ufile.txt";
    ModelHal model;
    UscriptHal<ModelHal> uscript(model, uName.c_str(), true);

    uscript.RecordDramLoad(0x12345678, "hello.hex");

    std::string expected = "ARCH " + GetArchName() +
                           "\n"
                           "PRODUCT N78" +
                           "\n"
                           "RESET\n"
                           "LOAD_MEM hello.hex 12345678\n";
    // Binary mode is required to avoid inserting \r characters on Windows
    std::ifstream uFile(uName, std::ios::binary);
    std::ostringstream actualStream;
    actualStream << uFile.rdbuf();
    std::string actual = actualStream.str();

    REQUIRE(actual == expected);
}

TEST_CASE("UscriptHal_DumpMem")
{
    std::string uName = "uscript_ufile.txt";
    ModelHal model;
    UscriptHal<ModelHal> uscript(model, uName.c_str(), true);

    uscript.DumpDram("hello.hex", 0x1000, 0x100);

    std::string expected = "ARCH " + GetArchName() +
                           "\n"
                           "PRODUCT N78" +
                           "\n"
                           "RESET\n"
                           "DUMP_MEM 0000000000001000 0000000000001100 > hello.hex\n";
    // Binary mode is required to avoid inserting \r characters on Windows
    std::ifstream uFile(uName, std::ios::binary);
    std::ostringstream actualStream;
    actualStream << uFile.rdbuf();
    std::string actual = actualStream.str();

    REQUIRE(actual == expected);
}

namespace
{
std::unique_ptr<std::vector<uint32_t>> CreateInferenceData(const std::initializer_list<ethosn_buffer_desc>& bufInfos,
                                                           const std::vector<uint32_t>& commandStreamData)
{
    using namespace command_stream;

    std::unique_ptr<std::vector<uint32_t>> inferenceDataPtr = std::make_unique<std::vector<uint32_t>>();
    std::vector<uint32_t>& inferenceData                    = *inferenceDataPtr;

    uint32_t numBuffers = static_cast<uint32_t>(bufInfos.size()) + 1;    // Plus 1 for command stream
    ethosn_buffer_array bufferArray;
    bufferArray.num_buffers = numBuffers;
    EmplaceBack<ethosn_buffer_array>(inferenceData, bufferArray);

    // Write buffer info for command stream. For now set offset and size to zero, we will correct them later.
    EmplaceBack<ethosn_buffer_desc>(inferenceData, { 0u, 0u, ETHOSN_BUFFER_CMD_FW });
    for (auto&& bInfo : bufInfos)
    {
        EmplaceBack<ethosn_buffer_desc>(inferenceData, std::move(bInfo));
    }

    const uint32_t headSize = static_cast<uint32_t>(inferenceData.size() * sizeof(inferenceData[0]));

    // Append the command stream data.
    uint32_t roundedCommandStreamSize = control_unit::utils::DivRoundUp(
        static_cast<uint32_t>(commandStreamData.size() * sizeof(uint32_t)), Pow2(sizeof(uint32_t)));

    inferenceData.resize(inferenceData.size() + roundedCommandStreamSize);

    auto cmdStreamAddrInHeader = reinterpret_cast<char*>(inferenceData.data()) + headSize;
    auto cmdStreamSizeBytes    = commandStreamData.size() * sizeof(commandStreamData[0]);
    memcpy(cmdStreamAddrInHeader, commandStreamData.data(), cmdStreamSizeBytes);

    // Update the buffer table entry for the command stream.^
    ethosn_buffer_desc& cmdStreamBufferInfo = *reinterpret_cast<ethosn_buffer_desc*>(
        reinterpret_cast<char*>(inferenceData.data()) + sizeof(ethosn_buffer_array));

    cmdStreamBufferInfo.address =
        reinterpret_cast<ethosn_address_t>(reinterpret_cast<char*>(inferenceData.data()) + headSize);
    cmdStreamBufferInfo.size = static_cast<uint32_t>(commandStreamData.size() * sizeof(commandStreamData[0]));

    return inferenceDataPtr;
}

}    // namespace

/**
 * @brief   UnitTest of ple MCU sev event. Test case setup inference to avoid assert when running firmware,
            but the actual result from convolution is not of interests.
            Set ce_status mcu_txev register to mimic ple sev events has happened,
            so firmware can use it to run the code in waitForEvent().

            Test case initialize non zeros values in scratch 5-7 registers representing ple has run into fault handler,
            test case check if firmware has asserted at the correct function.
*/

TEST_CASE("ModelHal_WaitForSevEvent")
{
    using namespace ethosn::command_stream;

    std::vector<Agent> agents;
    std::vector<CommandVariant> dmaRdCommands;
    std::vector<CommandVariant> dmaWrCommands;
    std::vector<CommandVariant> mceCommands;
    std::vector<CommandVariant> pleCommands;

    IfmS ifmS1             = {};
    ifmS1.bufferId         = 1;
    ifmS1.DMA_COMP_CONFIG0 = 0x0;
    ifmS1.DMA_STRIDE1      = 0x0;
    Agent ifmSAgent1{ ifmS1 };
    agents.push_back(ifmSAgent1);

    IfmS ifmS2             = {};
    ifmS2.bufferId         = 1;
    ifmS2.DMA_COMP_CONFIG0 = 0x0;
    ifmS2.DMA_STRIDE1      = 0x0;
    Agent ifmSAgent2{ ifmS2 };
    agents.push_back(ifmSAgent2);

    PleL pleL        = {};
    pleL.pleKernelId = PleKernelId::V4442_ADDITION_bw16_bh16_bm1_u8;
    Agent pleLAgent{ pleL };
    agents.push_back(pleLAgent);

    PleS pleS              = {};
    pleS.inputMode         = PleInputMode::SRAM_TWO_INPUTS;
    pleS.pleKernelId       = PleKernelId::V4442_ADDITION_bw16_bh16_bm1_u8;
    pleS.pleKernelSramAddr = 0x0;
    Agent pleSAgent{ pleS };
    agents.push_back(pleSAgent);

    OfmS ofmS             = {};
    ofmS.bufferId         = 2;
    ofmS.DMA_COMP_CONFIG0 = 0x0;
    ofmS.DMA_STRIDE1      = 0x0;
    Agent ofmSAgent{ ofmS };
    agents.push_back(ofmSAgent);

    DmaCommand loadPle;
    loadPle.type            = CommandType::LoadPleCodeIntoSram;
    loadPle.agentId         = 2;
    loadPle.m_DramOffset    = 0x0;
    loadPle.SRAM_ADDR       = 0x0;
    loadPle.DMA_SRAM_STRIDE = 0x0;
    loadPle.DMA_STRIDE0     = 0x0;
    loadPle.DMA_STRIDE2     = 0x0;
    loadPle.DMA_STRIDE3     = 0x0;
    loadPle.DMA_CHANNELS    = 0x0;
    loadPle.DMA_EMCS        = 0x3;
    loadPle.DMA_TOTAL_BYTES = 0x0;
    loadPle.DMA_CMD         = 0x28;
    dmaRdCommands.push_back(CommandVariant(loadPle));

    DmaCommand loadIfm1;
    loadIfm1.type            = CommandType::LoadIfmStripe;
    loadIfm1.agentId         = 0;
    loadIfm1.m_DramOffset    = 0x0;
    loadIfm1.SRAM_ADDR       = 0x100;
    loadIfm1.DMA_SRAM_STRIDE = 0x0;
    loadIfm1.DMA_STRIDE0     = 0x0;
    loadIfm1.DMA_STRIDE2     = 0x0;
    loadIfm1.DMA_STRIDE3     = 0x0;
    loadIfm1.DMA_CHANNELS    = 0xf;
    loadIfm1.DMA_EMCS        = 0xff;
    loadIfm1.DMA_TOTAL_BYTES = 0x23ff;
    loadIfm1.DMA_CMD         = 0x11;
    dmaRdCommands.push_back(CommandVariant(loadIfm1));

    DmaCommand loadIfm2;
    loadIfm2.type            = CommandType::LoadIfmStripe;
    loadIfm2.agentId         = 1;
    loadIfm2.m_DramOffset    = 0x0;
    loadIfm2.SRAM_ADDR       = 0x148;
    loadIfm2.DMA_SRAM_STRIDE = 0x0;
    loadIfm2.DMA_STRIDE0     = 0x0;
    loadIfm2.DMA_STRIDE2     = 0x0;
    loadIfm2.DMA_STRIDE3     = 0x0;
    loadIfm2.DMA_CHANNELS    = 0xf;
    loadIfm2.DMA_EMCS        = 0xff;
    loadIfm2.DMA_TOTAL_BYTES = 0x23ff;
    loadIfm2.DMA_CMD         = 0x12;
    dmaRdCommands.push_back(CommandVariant(loadIfm2));

    WaitForCounterCommand waitCmd;
    waitCmd.type         = CommandType::WaitForCounter;
    waitCmd.counterName  = CounterName::PleStripe;
    waitCmd.counterValue = 1;
    dmaWrCommands.push_back(CommandVariant(waitCmd));

    DmaCommand storeOfm;
    storeOfm.type            = CommandType::StoreOfmStripe;
    storeOfm.agentId         = 4;
    storeOfm.m_DramOffset    = 0x0;
    storeOfm.SRAM_ADDR       = 0x190;
    storeOfm.DMA_SRAM_STRIDE = 0x0;
    storeOfm.DMA_STRIDE0     = 0x0;
    storeOfm.DMA_STRIDE2     = 0x0;
    storeOfm.DMA_STRIDE3     = 0x0;
    storeOfm.DMA_CHANNELS    = 0xf;
    storeOfm.DMA_EMCS        = 0xff;
    storeOfm.DMA_TOTAL_BYTES = 0x23ff;
    storeOfm.DMA_CMD         = 0x14;
    dmaWrCommands.push_back(CommandVariant(storeOfm));

    WaitForCounterCommand pleWaitCmd3;
    pleWaitCmd3.type         = CommandType::WaitForCounter;
    pleWaitCmd3.counterName  = CounterName::DmaRd;
    pleWaitCmd3.counterValue = 3;
    pleCommands.push_back(CommandVariant(pleWaitCmd3));

    LoadPleCodeIntoPleSramCommand pleLoadCmd;
    pleLoadCmd.type    = CommandType::LoadPleCodeIntoPleSram;
    pleLoadCmd.agentId = 3;
    pleCommands.push_back(CommandVariant(pleLoadCmd));

    WaitForCounterCommand pleWaitCmd4;
    pleWaitCmd4.type         = CommandType::WaitForCounter;
    pleWaitCmd4.counterName  = CounterName::PleCodeLoadedIntoPleSram;
    pleWaitCmd4.counterValue = 1;
    pleCommands.push_back(CommandVariant(pleWaitCmd4));

    // Logging HAL doesn't simulate the PLE running, leaving an error
    // message here for the firmware to pick up on the next spin
    StartPleStripeCommand startPle;
    startPle.type       = CommandType::StartPleStripe;
    startPle.agentId    = 3;
    startPle.SCRATCH[0] = 0x0;
    startPle.SCRATCH[1] = 0x0;
    startPle.SCRATCH[2] = 0x0;
    startPle.SCRATCH[3] = 0x7;
    startPle.SCRATCH[4] = 0x0;
    startPle.SCRATCH[5] = 0x0;
    startPle.SCRATCH[6] = 0x0;
    startPle.SCRATCH[7] = 0x0;
    pleCommands.push_back(CommandVariant(startPle));

    std::vector<uint32_t> cmdStream =
        BuildCommandStream(agents, dmaRdCommands, dmaWrCommands, mceCommands, pleCommands);

    const uint32_t inputDramAddr  = 0x60100000U;
    const uint32_t outputDramAddr = 0x60C00000U;

    std::unique_ptr<std::vector<uint32_t>> inferenceData = CreateInferenceData(
        {
            { inputDramAddr, 1 * 24 * 24 * 16, ETHOSN_BUFFER_INPUT },
            { outputDramAddr, 1 * 24 * 24 * 16, ETHOSN_BUFFER_OUTPUT },
        },
        cmdStream);

    Inference inference(reinterpret_cast<ethosn_address_t>(inferenceData->data()));

    LoggingHal::Options options;
    options.m_EthosNVariant                      = LoggingHal::Options::EthosNVariant::N78_4TOPS_4PLE_RATIO;
    options.m_PleWaitsForGlobalStripeBankControl = false;
    LoggingHal loggingHal(options);
    Firmware<LoggingHal> fw(loggingHal, 0);

    REQUIRE(!fw.RunInference(inference).success);
}

TEST_CASE("HalBase_ClearSram")
{
    LoggingHal loggingHal(LoggingHal::Options{});
    loggingHal.ClearSram();
    std::vector<LoggingHal::Entry> entries = loggingHal.GetEntries();

    static const std::vector<LoggingHal::Entry> golden{
        { LoggingHal::Entry::WriteReg, CE_REG(0, CE_RP, CE_CE_INST), 0x1 },
        { LoggingHal::Entry::WriteReg, CE_REG(1, CE_RP, CE_CE_INST), 0x1 },
        { LoggingHal::Entry::WriteReg, CE_REG(2, CE_RP, CE_CE_INST), 0x1 },
        { LoggingHal::Entry::WriteReg, CE_REG(3, CE_RP, CE_CE_INST), 0x1 },
        { LoggingHal::Entry::WriteReg, CE_REG(4, CE_RP, CE_CE_INST), 0x1 },
        { LoggingHal::Entry::WriteReg, CE_REG(5, CE_RP, CE_CE_INST), 0x1 },
        { LoggingHal::Entry::WriteReg, CE_REG(6, CE_RP, CE_CE_INST), 0x1 },
        { LoggingHal::Entry::WriteReg, CE_REG(7, CE_RP, CE_CE_INST), 0x1 },
    };

    RequireLoggingHalEntriesContainsInOrder(golden, entries);
}
