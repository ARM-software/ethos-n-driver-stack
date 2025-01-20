//
// Copyright © 2021-2025 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../Profiling.hpp"
#include "Types.hpp"
#include "ncu_ple_interface_def.h"
#include <common/Inference.hpp>
#include <common/Utils.hpp>

#include <ethosn_command_stream/PleKernelIds.hpp>

#include <cinttypes>

namespace ethosn
{
namespace control_unit
{
// Defined in PleKernelBinaries.hpp
extern const std::pair<uint32_t, uint32_t> g_PleKernelOffsetsAndSizes[];
}    // namespace control_unit
}    // namespace ethosn

namespace ethosn::control_unit
{

enum class CeEnables : uint8_t
{
    AllDisabled = 0,
    OneEnabled  = 1,
    TwoEnabled  = 2,
    // ...
    AllEnabledForPleOnly = 254,
    Unknown              = 255,
};

struct CompletedTsuEvents
{
    bool pleStripeDone            = false;
    bool pleCodeLoadedIntoPleSram = false;
    bool pleError                 = false;
};

/// This class groups methods to program HW registers for the different type of agents
template <typename Hal>
class HwAbstraction
{
public:
    HwAbstraction(BufferTable buffertable,
                  const uint64_t pleKernelDataAddr,
                  Hal& hal,
                  profiling::ProfilingData<Hal>& profiling);

    uint32_t GetNumCmdsInDmaRdQueue();
    uint32_t GetNumCmdsInMceQueue();
    bool IsPleBusy();
    uint32_t GetNumCmdsInDmaWrQueue();

    /// The HandleXYZ functions return a  profiling entry ID for the event that the Handle method started,
    /// usually a HW event such as a DMA transaction.
    /// The calling code is responsible for ending this event when the corresponding
    /// HW event is complete.
    /// @{
    profiling::ProfilingOnly<uint8_t> HandleDmaRdCmdIfm(const IfmS& agentData, const DmaCommand& cmd);
    profiling::ProfilingOnly<uint8_t> HandleDmaWrCmdOfm(const OfmS& agentData, const DmaCommand& cmd);
    profiling::ProfilingOnly<uint8_t> HandleDmaRdCmdWeights(const WgtS& agentData, const DmaCommand& cmd);
    profiling::ProfilingOnly<uint8_t> HandleDmaRdCmdPleCode(const PleL& agentData, const DmaCommand& cmd);

    void HandleWriteMceStripeRegs(const MceS& agentData, const ProgramMceStripeCommand& cmd);
    profiling::ProfilingOnly<uint8_t> HandleStartMceStripeBank(const MceS& agentData, const StartMceStripeCommand& cmd);
    profiling::ProfilingOnly<uint8_t> HandlePleStripeCmd(const PleS& agentData, const StartPleStripeCommand& cmd);
    /// @}

    bool TrySetCeEnables(CeEnables numEnabledCes);

    CompletedTsuEvents UpdateTsuEvents();
    void LoadPleCodeIntoPleSram(uint32_t agentId, const PleS& agentData);
    void RestartPle();
    void ConfigMcePle(const MceS& agentData);

    void WaitForEvents();

    decltype(auto) GetLogger()
    {
        return m_Hal.m_Logger;
    }

    bool HasErrors()
    {
        return m_IsPleError;
    }

    profiling::ProfilingData<Hal>& GetProfiling()
    {
        return m_Profiling;
    }

    void RecordProfilingCounters();

    /// Checks if everything in the hardware is idle and any outstanding tasks are finished.
    /// At the end of an inference, this should return true.
    bool IsFinished();

    void EnableDebug()
    {
        m_Hal.EnableDebug();
    }
    void DisableDebug()
    {
        m_Hal.DisableDebug();
    }
    void StoreDebugGpRegister(uint32_t gpNum, uint32_t value);

private:
    Hal& m_Hal;
    const BufferTable m_BufferTable;
    bool m_IsPleBusy{ false };
    bool m_IsPleError{ false };
    uint64_t m_PleKernelDataAddr;

    /// Stores which CEs have been enabled via the CE_CE_ENABLES registers. It is simpler to store
    /// this ourselves rather than read from the registers, because there are 8 registers
    /// (one per CE) to read.
    /// Because these registers are not banked like most of the other CE registers, we can't pre-program
    /// these and instead must set them just before kicking off a stripe. We must also avoid changing
    /// these while other stripes are running as the changes take effect immediately.
    /// Initialize to a value that ensure the CE-enabled flags are configured for the first
    /// stripe.
    CeEnables m_NumCesEnabled = CeEnables::Unknown;

    profiling::ProfilingData<Hal>& m_Profiling;
    profiling::ProfilingOnly<uint32_t> m_DmaRdNumTransactions       = 0;
    profiling::ProfilingOnly<uint32_t> m_DmaRdTotalBytesTransferred = 0;
    profiling::ProfilingOnly<uint32_t> m_DmaWrNumTransactions       = 0;
    profiling::ProfilingOnly<uint32_t> m_DmaWrTotalBytesTransferred = 0;
    profiling::ProfilingOnly<uint8_t> m_ProfilingUdmaEntryId;
};

/// Class template argument deduction guideline that allows for instantiation code to cleanly choose
/// whether the Hal should be captured by value or by reference.
///
/// Examples:
///     Hal hal;
///
///     HwAbstraction hw{ bufferTable, hal, ... };        // hal will be captured by reference.
///                                                       // Note: decltype(hw) = HwAbstraction<Hal&, ...>
///
///     HwAbstraction hw{ bufferTable, Hal{ hal }, ... }; // hal will be captured by value.
///                                                       // Note: decltype(hw) = HwAbstraction<Hal, ...>
///
///     HwAbstraction hw{ bufferTable, Hal{}, ... };      // An internal Hal will be copy-initialized
///                                                       // from the expression Hal{}.
///                                                       // Note: decltype(hw) = HwAbstraction<Hal, ...>
///
template <typename Hal>
HwAbstraction(BufferTable, uint64_t, Hal &&)->HwAbstraction<Hal>;

//************************************** IMPLEMENTATION **************************************

template <typename Hal>
uint32_t HwAbstraction<Hal>::GetNumCmdsInDmaRdQueue()

{
    constexpr uint32_t hwQueueSize = 4;
    const dma_status_r st          = m_Hal.ReadReg(TOP_REG(DMA_RP, DMA_DMA_STATUS));
    return hwQueueSize - st.get_rd_cmdq_free();
}

template <typename Hal>
uint32_t HwAbstraction<Hal>::GetNumCmdsInMceQueue()
{
    const stripe_bank_status_r st = m_Hal.ReadReg(TOP_REG(TSU_RP, TSU_STRIPE_BANK_STATUS));

    uint32_t numBanksBusy = 0;
    numBanksBusy += st.get_bank0_status() != bank_status_t::IDLE;
    numBanksBusy += st.get_bank1_status() != bank_status_t::IDLE;

    return numBanksBusy;
}

template <typename Hal>
bool HwAbstraction<Hal>::IsPleBusy()
{
    return m_IsPleBusy;
}

template <typename Hal>
uint32_t HwAbstraction<Hal>::GetNumCmdsInDmaWrQueue()
{
    constexpr uint32_t hwQueueSize = 4;
    const dma_status_r st          = m_Hal.ReadReg(TOP_REG(DMA_RP, DMA_DMA_STATUS));
    return hwQueueSize - st.get_wr_cmdq_free();
}

// Return false on error
template <typename Hal>
CompletedTsuEvents HwAbstraction<Hal>::UpdateTsuEvents()
{
    const tsu_event_r tsuEventReg{ m_Hal.ReadReg(TOP_REG(TSU_RP, TSU_TSU_EVENT)) };
    CompletedTsuEvents result = { false, false, false };

    if (tsuEventReg.get_udma_or_clear_done() == event_t::TRIGGERED)
    {
        // Record the end of the UDMA event, which was started in LoadPleCodeIntoPleSram.
        m_Profiling.RecordEnd(m_ProfilingUdmaEntryId);

        // Now that the new PLE code is ready, tell the PLE to run it.
        RestartPle();

        result.pleCodeLoadedIntoPleSram = true;
    }
    if (tsuEventReg.get_ple_stripe_done() == event_t::TRIGGERED)
    {
        m_IsPleBusy          = false;
        m_IsPleError         = false;
        result.pleStripeDone = true;
        // Mask the PLE from receiving any events. At this point the PLE should be sleeping and waiting to be told
        // to process another stripe, or to be reloaded with new code. We don't want the PLE to be woken
        // up until we have told it to process another stripe, which would otherwise happen because it may
        // receive a BLOCK_DONE signal from the MCE. If the PLE were to wake up as we were loading new code into
        // it, it could start executing random code!
        utils::DisablePleMcuEvents(m_Hal);

        // Read the scratch registers that the PLE should have written, containing the number
        // of blocks that the PLE processed in the stripe it just finished.
        // Use this to decrement our counter, so that we know when it is safe to reconfigure the MCEIF
        constexpr uint32_t ceIdx = 0;    // We use CE 0 for consistency with the corresponding MCE calculation

        ncu_ple_interface::PleMsg::Type msgType =
            static_cast<ncu_ple_interface::PleMsg::Type>(m_Hal.ReadReg(CE_REG(ceIdx, CE_RP, CE_PLE_SCRATCH0)));
        if (msgType != ncu_ple_interface::PleMsg::StripeDone::type)
        {
            // Assume non-StripeDone messages are errors
            m_Hal.m_Logger.Error("Ple[%d] Error: PleMsg is not STRIPE_DONE", ceIdx);
            m_IsPleError    = true;
            result.pleError = true;
        }
    }

    // Sample profiling counters - this part of the code is called quite frequently so it is a good place to do this.
    RecordProfilingCounters();

    return result;
}

template <typename Hal>
HwAbstraction<Hal>::HwAbstraction(const BufferTable buffertable,
                                  const uint64_t pleKernelDataAddr,
                                  Hal& hal,
                                  profiling::ProfilingData<Hal>& profiling)
    : m_Hal{ hal }
    , m_BufferTable{ buffertable }
    , m_PleKernelDataAddr{ pleKernelDataAddr }
    , m_Profiling(profiling)
{
    ASSERT_MSG(IsFinished(), "Must be constructed when HW is idle");
}

/// This initial implementation only needs to implement limited support:
/// - Stripes don't need to pack boundary data in a single slot.
template <typename Hal>
profiling::ProfilingOnly<uint8_t> HwAbstraction<Hal>::HandleDmaRdCmdIfm(const IfmS& agentData, const DmaCommand& cmd)
{
    using namespace command_stream;

    m_Hal.m_Logger.Debug("Execute %s", ToString(cmd).GetCString());

    uint8_t profilingSetupEntryId = m_Profiling.RecordStart(TimelineEventType::DmaReadSetup);

    ethosn_address_t dramAddr = m_BufferTable[agentData.bufferId].address + cmd.m_DramOffset;
    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DRAM_ADDR_L), static_cast<uint32_t>(dramAddr));
    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DRAM_ADDR_H), static_cast<uint32_t>(dramAddr >> 32));

    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DMA_COMP_CONFIG0), agentData.DMA_COMP_CONFIG0);
    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DMA_STRIDE1), agentData.DMA_STRIDE1);

    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_SRAM_ADDR), cmd.SRAM_ADDR);
    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DMA_SRAM_STRIDE), cmd.DMA_SRAM_STRIDE);
    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DMA_STRIDE0), cmd.DMA_STRIDE0);
    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DMA_STRIDE2), cmd.DMA_STRIDE2);
    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DMA_STRIDE3), cmd.DMA_STRIDE3);
    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DMA_CHANNELS), cmd.DMA_CHANNELS);

    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DMA_EMCS), cmd.DMA_EMCS);
    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DMA_TOTAL_BYTES), cmd.DMA_TOTAL_BYTES);

    // The stream type field in the cmd register is set here in the firmware, as this controls
    // access to external memory (e.g. NSAIDs) and so is more of a security concern, so we don't want
    // the host system's userspace to be able to change this.
    dma_rd_cmd_r rdCmd = cmd.DMA_CMD;

    dma_stream_type_t streamType = dma_stream_type_t::STREAM_6;    // Default stream type
    switch (m_BufferTable[agentData.bufferId].type)
    {
        case ETHOSN_BUFFER_INPUT:
        {
            streamType = dma_stream_type_t::STREAM_6;
            break;
        }
        case ETHOSN_BUFFER_INTERMEDIATE:
        {
            streamType = dma_stream_type_t::STREAM_7;
            break;
        }
        case ETHOSN_BUFFER_CONSTANT:
        {
            streamType = dma_stream_type_t::STREAM_5;
            break;
        }
        default:
        {
            FATAL_MSG("Invalid buffer type for input: %u", m_BufferTable[agentData.bufferId].type);
            break;
        }
    }
    rdCmd.set_stream_type(streamType);

    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DMA_RD_CMD), rdCmd.word);

    m_Profiling.RecordEnd(profilingSetupEntryId);

    profiling::ProfilingOnly<uint8_t> profilingEntryId = m_Profiling.RecordStart(TimelineEventType::DmaRead);

    ++m_DmaRdNumTransactions;
    m_DmaRdTotalBytesTransferred += cmd.DMA_TOTAL_BYTES + 1;

    return profilingEntryId;
}

/// This initial implementation only needs to implement limited support:
/// - Stripes don't need to pack boundary data in a single slot.
template <typename Hal>
profiling::ProfilingOnly<uint8_t> HwAbstraction<Hal>::HandleDmaWrCmdOfm(const OfmS& agentData, const DmaCommand& cmd)
{
    using namespace command_stream;

    m_Hal.m_Logger.Debug("Execute %s", ToString(cmd).GetCString());

    uint8_t profilingSetupEntryId = m_Profiling.RecordStart(TimelineEventType::DmaWriteSetup);

    ethosn_address_t dramAddr = m_BufferTable[agentData.bufferId].address + cmd.m_DramOffset;
    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DRAM_ADDR_L), static_cast<uint32_t>(dramAddr));
    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DRAM_ADDR_H), static_cast<uint32_t>(dramAddr >> 32));

    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DMA_COMP_CONFIG0), agentData.DMA_COMP_CONFIG0);
    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DMA_STRIDE1), agentData.DMA_STRIDE1);

    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_SRAM_ADDR), cmd.SRAM_ADDR);
    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DMA_SRAM_STRIDE), cmd.DMA_SRAM_STRIDE);
    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DMA_STRIDE0), cmd.DMA_STRIDE0);
    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DMA_STRIDE2), cmd.DMA_STRIDE2);
    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DMA_STRIDE3), cmd.DMA_STRIDE3);
    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DMA_CHANNELS), cmd.DMA_CHANNELS);

    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DMA_EMCS), cmd.DMA_EMCS);
    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DMA_TOTAL_BYTES), cmd.DMA_TOTAL_BYTES);

    // The stream type field in the cmd register is set here in the firmware, as this controls
    // access to external memory (e.g. NSAIDs) and so is more of a security concern, so we don't want
    // the host system's userspace to be able to change this.
    dma_wr_cmd_r wrCmd = cmd.DMA_CMD;

    dma_stream_type_t streamType = dma_stream_type_t::STREAM_8;    // Default stream type
    switch (m_BufferTable[agentData.bufferId].type)
    {
        case ETHOSN_BUFFER_OUTPUT:
        {
            streamType = dma_stream_type_t::STREAM_8;
            break;
        }
        case ETHOSN_BUFFER_INTERMEDIATE:
        {
            streamType = dma_stream_type_t::STREAM_7;
            break;
        }
        default:
        {
            FATAL_MSG("Invalid buffer type for output: %u", m_BufferTable[agentData.bufferId].type);
            break;
        }
    }
    // Setting stream
    wrCmd.set_stream_type(streamType);

    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DMA_WR_CMD), wrCmd.word);

    m_Profiling.RecordEnd(profilingSetupEntryId);

    profiling::ProfilingOnly<uint8_t> profilingEntryId = m_Profiling.RecordStart(TimelineEventType::DmaWrite);

    ++m_DmaWrNumTransactions;
    m_DmaWrTotalBytesTransferred += cmd.DMA_TOTAL_BYTES + 1;

    return profilingEntryId;
}

template <typename Hal>
profiling::ProfilingOnly<uint8_t> HwAbstraction<Hal>::HandleDmaRdCmdWeights(const WgtS& agentData,
                                                                            const DmaCommand& cmd)
{
    m_Hal.m_Logger.Debug("Execute %s", ToString(cmd).GetCString());

    using namespace command_stream;

    FATAL_COND_MSG(m_BufferTable[agentData.bufferId].type == ETHOSN_BUFFER_CONSTANT,
                   "Invalid buffer type for weights: %u", m_BufferTable[agentData.bufferId].type);

    uint8_t profilingSetupEntryId = m_Profiling.RecordStart(TimelineEventType::DmaReadSetup);

    // write DMA registers
    {
        const ethosn_address_t weightsBufferAddr = m_BufferTable[agentData.bufferId].address;
        const ethosn_address_t dramAddr          = weightsBufferAddr + cmd.m_DramOffset;
        m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DRAM_ADDR_L), static_cast<uint32_t>(dramAddr));
        m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DRAM_ADDR_H), static_cast<uint32_t>(dramAddr >> 32));
    }
    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_SRAM_ADDR), cmd.SRAM_ADDR);
    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DMA_EMCS), cmd.DMA_EMCS);
    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DMA_TOTAL_BYTES), cmd.DMA_TOTAL_BYTES);

    // The stream type field in the cmd register is set here in the firmware, as this controls
    // access to external memory (e.g. NSAIDs) and so is more of a security concern, so we don't want
    // the host system's userspace to be able to change this.
    dma_rd_cmd_r rdCmd = cmd.DMA_CMD;
    rdCmd.set_stream_type(dma_stream_type_t::STREAM_5);
    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DMA_RD_CMD), rdCmd.word);

    m_Profiling.RecordEnd(profilingSetupEntryId);

    profiling::ProfilingOnly<uint8_t> profilingEntryId = m_Profiling.RecordStart(TimelineEventType::DmaRead);

    ++m_DmaRdNumTransactions;
    m_DmaRdTotalBytesTransferred += cmd.DMA_TOTAL_BYTES + 1;

    return profilingEntryId;
}

template <typename Hal>
profiling::ProfilingOnly<uint8_t> HwAbstraction<Hal>::HandleDmaRdCmdPleCode(const PleL& agentData,
                                                                            const DmaCommand& cmd)
{
    m_Hal.m_Logger.Debug("Execute %s", ToString(cmd).GetCString());

    uint8_t profilingSetupEntryId = m_Profiling.RecordStart(TimelineEventType::DmaReadSetup);

    const uint32_t pleKernelId = static_cast<uint32_t>(agentData.pleKernelId);

    // DRAM address
    // write DMA registers
    {
        const ethosn_address_t dramAddr = m_PleKernelDataAddr + g_PleKernelOffsetsAndSizes[pleKernelId].first;
        m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DRAM_ADDR_L), static_cast<uint32_t>(dramAddr));
        m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DRAM_ADDR_H), static_cast<uint32_t>(dramAddr >> 32));
    }
    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_SRAM_ADDR), cmd.SRAM_ADDR);
    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DMA_EMCS), cmd.DMA_EMCS);

    // n/a for BRODCAST format: DMA_DMA_CHANNELS, DMA_DMA_STRIDEx
    const uint32_t totalBytes = g_PleKernelOffsetsAndSizes[pleKernelId].second;
    {
        dma_total_bytes_r tot;
        tot.set_total_bytes(totalBytes);
        m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DMA_TOTAL_BYTES), tot.word);
    }

    // The last write should be to DMA_DMA_RD_CMD, which will push the command to the HW queue

    // The stream type field in the cmd register is set here in the firmware, as this controls
    // access to external memory (e.g. NSAIDs) and so is more of a security concern, so we don't want
    // the host system's userspace to be able to change this.
    dma_rd_cmd_r rdCmd = cmd.DMA_CMD;
    rdCmd.set_stream_type(dma_stream_type_t::STREAM_4);
    m_Hal.WriteReg(TOP_REG(DMA_RP, DMA_DMA_RD_CMD), rdCmd.word);

    m_Profiling.RecordEnd(profilingSetupEntryId);

    profiling::ProfilingOnly<uint8_t> profilingEntryId = m_Profiling.RecordStart(TimelineEventType::DmaRead);

    ++m_DmaRdNumTransactions;
    m_DmaRdTotalBytesTransferred += totalBytes;

    return profilingEntryId;
}

/// This initial implementation has limited support:
/// Not wide kernel, padding 0, filter 1x1, not fully connected,
/// slots only in mid/center, not winograd, direct mode, not upsample, any data type U8.
template <typename Hal>
void HwAbstraction<Hal>::HandleWriteMceStripeRegs(const MceS& agentData, const ProgramMceStripeCommand& cmd)
{
    m_Hal.m_Logger.Debug("Execute %s", ToString(cmd).GetCString());

    using namespace command_stream;

    uint8_t profilingSetupEntryId = m_Profiling.RecordStart(TimelineEventType::MceStripeSetup);

    const bool isDepthwise = agentData.mceOpMode == MceOperation::DEPTHWISE_CONVOLUTION;
    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_ACTIVATION_CONFIG), agentData.ACTIVATION_CONFIG);
    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_WIDE_KERNEL_CONTROL), agentData.WIDE_KERNEL_CONTROL);
    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_FILTER), agentData.FILTER);
    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_IFM_ZERO_POINT), agentData.IFM_ZERO_POINT);
    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_IFM_DEFAULT_SLOT_SIZE), agentData.IFM_DEFAULT_SLOT_SIZE);
    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_IFM_SLOT_STRIDE), agentData.IFM_SLOT_STRIDE);
    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_STRIPE_BLOCK_CONFIG), agentData.STRIPE_BLOCK_CONFIG);
    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_DEPTHWISE_CONTROL), agentData.DEPTHWISE_CONTROL);
    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_IFM_SLOT_BASE_ADDRESS_IG0), agentData.IFM_SLOT_BASE_ADDRESS);
    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_IFM_SLOT_BASE_ADDRESS_IG1), agentData.IFM_SLOT_BASE_ADDRESS);
    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_IFM_SLOT_BASE_ADDRESS_IG2), agentData.IFM_SLOT_BASE_ADDRESS);
    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_IFM_SLOT_BASE_ADDRESS_IG3), agentData.IFM_SLOT_BASE_ADDRESS);

    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_CE_CONTROL), cmd.CE_CONTROL);

    uint32_t engines = m_Hal.NumCes().GetValue();

    // config Mul enable in OGs
    {
        if (isDepthwise)
        {
            // Different per-CE
            for (uint32_t ce = 0U; ce < engines; ++ce)
            {
                m_Hal.WriteReg(CE_REG(ce, STRIPE_RP, CE_STRIPE_MUL_ENABLE_OG0), cmd.MUL_ENABLE[ce][0]);
                m_Hal.WriteReg(CE_REG(ce, STRIPE_RP, CE_STRIPE_MUL_ENABLE_OG1), cmd.MUL_ENABLE[ce][1]);
                m_Hal.WriteReg(CE_REG(ce, STRIPE_RP, CE_STRIPE_MUL_ENABLE_OG2), cmd.MUL_ENABLE[ce][2]);
                m_Hal.WriteReg(CE_REG(ce, STRIPE_RP, CE_STRIPE_MUL_ENABLE_OG3), cmd.MUL_ENABLE[ce][3]);
            }
        }
        else
        {
            // Same for all CEs
            m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_MUL_ENABLE_OG0), cmd.MUL_ENABLE[0][0]);
            m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_MUL_ENABLE_OG1), cmd.MUL_ENABLE[0][1]);
            m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_MUL_ENABLE_OG2), cmd.MUL_ENABLE[0][2]);
            m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_MUL_ENABLE_OG3), cmd.MUL_ENABLE[0][3]);
        }
    }

    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_IFM_ROW_STRIDE), cmd.IFM_ROW_STRIDE);
    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_IFM_CONFIG1), cmd.IFM_CONFIG1);

    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_IFM_PAD0_IG0), cmd.IFM_PAD[0][0]);
    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_IFM_PAD0_IG1), cmd.IFM_PAD[0][1]);
    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_IFM_PAD0_IG2), cmd.IFM_PAD[0][2]);
    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_IFM_PAD0_IG3), cmd.IFM_PAD[0][3]);

    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_IFM_PAD1_IG0), cmd.IFM_PAD[1][0]);
    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_IFM_PAD1_IG1), cmd.IFM_PAD[1][1]);
    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_IFM_PAD1_IG2), cmd.IFM_PAD[1][2]);
    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_IFM_PAD1_IG3), cmd.IFM_PAD[1][3]);

    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_IFM_PAD2_IG0), cmd.IFM_PAD[2][0]);
    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_IFM_PAD2_IG1), cmd.IFM_PAD[2][1]);
    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_IFM_PAD2_IG2), cmd.IFM_PAD[2][2]);
    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_IFM_PAD2_IG3), cmd.IFM_PAD[2][3]);

    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_IFM_PAD3_IG0), cmd.IFM_PAD[3][0]);
    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_IFM_PAD3_IG1), cmd.IFM_PAD[3][1]);
    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_IFM_PAD3_IG2), cmd.IFM_PAD[3][2]);
    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_IFM_PAD3_IG3), cmd.IFM_PAD[3][3]);

    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_WIDE_KERNEL_OFFSET), cmd.WIDE_KERNEL_OFFSET);

    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_IFM_TOP_SLOTS), cmd.IFM_TOP_SLOTS);
    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_IFM_MID_SLOTS), cmd.IFM_MID_SLOTS);
    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_IFM_BOTTOM_SLOTS), cmd.IFM_BOTTOM_SLOTS);
    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_IFM_SLOT_PAD_CONFIG), cmd.IFM_SLOT_PAD_CONFIG);

    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_OFM_STRIPE_SIZE), cmd.OFM_STRIPE_SIZE);
    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_OFM_CONFIG), cmd.OFM_CONFIG);

    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_WEIGHT_BASE_ADDR_OG0), cmd.WEIGHT_BASE_ADDR[0]);
    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_WEIGHT_BASE_ADDR_OG1), cmd.WEIGHT_BASE_ADDR[1]);
    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_WEIGHT_BASE_ADDR_OG2), cmd.WEIGHT_BASE_ADDR[2]);
    m_Hal.WriteReg(TOP_REG(STRIPE_RP, CE_STRIPE_WEIGHT_BASE_ADDR_OG3), cmd.WEIGHT_BASE_ADDR[3]);

    for (uint32_t ce = 0; ce < engines; ++ce)
    {
        m_Hal.WriteReg(CE_REG(ce, STRIPE_RP, CE_STRIPE_IFM_CONFIG2_IG0), cmd.IFM_CONFIG2[ce][0]);
        m_Hal.WriteReg(CE_REG(ce, STRIPE_RP, CE_STRIPE_IFM_CONFIG2_IG1), cmd.IFM_CONFIG2[ce][1]);
        m_Hal.WriteReg(CE_REG(ce, STRIPE_RP, CE_STRIPE_IFM_CONFIG2_IG2), cmd.IFM_CONFIG2[ce][2]);
        m_Hal.WriteReg(CE_REG(ce, STRIPE_RP, CE_STRIPE_IFM_CONFIG2_IG3), cmd.IFM_CONFIG2[ce][3]);
    }

    m_Profiling.RecordEnd(profilingSetupEntryId);
}

template <typename Hal>
profiling::ProfilingOnly<uint8_t> HwAbstraction<Hal>::HandlePleStripeCmd(const PleS& agentData,
                                                                         const StartPleStripeCommand& cmd)
{
    m_Hal.m_Logger.Debug("Execute %s", ToString(cmd).GetCString());

    ASSERT_MSG(!m_IsPleBusy, "Can't start a new PLE stripe while it is already processing one");

    using namespace command_stream;

    uint8_t profilingSetupEntryId = m_Profiling.RecordStart(TimelineEventType::PleStripeSetup);

    const bool isSram =
        agentData.inputMode == PleInputMode::SRAM_ONE_INPUT || agentData.inputMode == PleInputMode::SRAM_TWO_INPUTS;
    if (isSram)
    {
        ASSERT_MSG(m_NumCesEnabled == CeEnables::AllEnabledForPleOnly, "CE enables not set correctly");
    }

    m_Hal.WriteReg(TOP_REG(CE_RP, CE_PLE_SCRATCH0), cmd.SCRATCH[0]);
    m_Hal.WriteReg(TOP_REG(CE_RP, CE_PLE_SCRATCH1), cmd.SCRATCH[1]);
    m_Hal.WriteReg(TOP_REG(CE_RP, CE_PLE_SCRATCH2), cmd.SCRATCH[2]);
    m_Hal.WriteReg(TOP_REG(CE_RP, CE_PLE_SCRATCH3), cmd.SCRATCH[3]);
    m_Hal.WriteReg(TOP_REG(CE_RP, CE_PLE_SCRATCH4), cmd.SCRATCH[4]);
    m_Hal.WriteReg(TOP_REG(CE_RP, CE_PLE_SCRATCH5), cmd.SCRATCH[5]);
    m_Hal.WriteReg(TOP_REG(CE_RP, CE_PLE_SCRATCH6), cmd.SCRATCH[6]);
    m_Hal.WriteReg(TOP_REG(CE_RP, CE_PLE_SCRATCH7), cmd.SCRATCH[7]);

    // After programming the PLE_SCRATCH registers, the firmware needs to send an event to the PLE's.
    // Before we send the event though, we need to un-mask the event so that the PLE actually receives it.
    utils::EnablePleMcuEvents(m_Hal);
    {
        ple_setirq_r pleSetIrq;
        pleSetIrq.set_event(1);
        m_Hal.WriteReg(TOP_REG(CE_RP, CE_PLE_SETIRQ), pleSetIrq.word);
    }

    m_Profiling.RecordEnd(profilingSetupEntryId);

    profiling::ProfilingOnly<uint8_t> profilingEntryId = m_Profiling.RecordStart(TimelineEventType::PleStripe);

    m_IsPleBusy = true;

    return profilingEntryId;
}

template <typename Hal>
void HwAbstraction<Hal>::RestartPle()
{
    m_Hal.m_Logger.Debug("Execute RestartPle");
    ASSERT_MSG(!m_IsPleBusy, "Can't restart the PLE whilst it is still processing a stripe");

    const bool plesAreInReset = ple_control_0_r(m_Hal.ReadReg(CE_REG(0, CE_RP, CE_PLE_CONTROL_0))).get_cpuwait() == 1U;

    // Restart the PLEs to start running code.
    // They will run until they are blocked on waiting for the first block_done from the MCE.
    if (plesAreInReset)
    {
        // This is the default status of the PLE-MCU when coming out of reset
        ple_control_0_r pleControl0;
        pleControl0.set_cpuwait(0);
        m_Hal.WriteReg(TOP_REG(CE_RP, CE_PLE_CONTROL_0), pleControl0.word);
    }
    else
    {
        // PLE is running. Assert NMI to make it jump to the reset vector
        ple_setirq_r pleSetIrq;
        pleSetIrq.set_nmi(1);
        m_Hal.WriteReg(TOP_REG(CE_RP, CE_PLE_SETIRQ), pleSetIrq.word);
    }
}

template <typename Hal>
void HwAbstraction<Hal>::LoadPleCodeIntoPleSram(uint32_t agentId, const PleS& agentData)
{
    m_Hal.m_Logger.Debug("Execute LoadPleCodeIntoPleSram{ .agentId = %u }", agentId);

    ASSERT_MSG(!m_IsPleBusy, "Can't load a new kernel if the PLE is still processing a stripe");

    const uint32_t pleKernelId = static_cast<uint32_t>(agentData.pleKernelId);

    ple_udma_load_parameters_r udma_param;
    udma_param.set_emc(udma_emc_choice_t::EMC_0);
    uint32_t sizeBytes = g_PleKernelOffsetsAndSizes[pleKernelId].second;
    ASSERT_MSG(sizeBytes % m_Hal.NumBytesPerBeat() == 0, "PLE kernel size must be multiple of 16");
    udma_param.set_length(sizeBytes / m_Hal.NumBytesPerBeat());
    udma_param.set_ple(udma_ple_choice_t::MCU_MEM);
    m_Hal.WriteReg(TOP_REG(CE_RP, CE_PLE_UDMA_LOAD_PARAMETERS), udma_param.word);

    ple_udma_load_command_r udma_cmd;
    udma_cmd.set_emc_addr(agentData.pleKernelSramAddr);
    udma_cmd.set_ple_addr(0);
    m_Hal.WriteReg(TOP_REG(CE_RP, CE_PLE_UDMA_LOAD_COMMAND), udma_cmd.word);

    m_ProfilingUdmaEntryId = m_Profiling.RecordStart(TimelineEventType::Udma);
}

template <typename Hal>
void HwAbstraction<Hal>::WaitForEvents()
{
    m_Hal.m_Logger.Debug("WFE");

    dl2_int_status_r intStatus;
    do
    {
        uint8_t wfeEventId = m_Profiling.RecordStart(TimelineEventType::Wfe);
        m_Hal.WaitForEvents();
        m_Profiling.RecordEnd(wfeEventId);

        // Check why we were woken up and go back to sleep if it wasn't
        // an event we are interested in. There are several reasons why we could have been
        // woken from WFE that we are not interested in, including a spurious wakeup
        // and also a return from SVC instruction due to logging.
        intStatus.word = m_Hal.ReadReg(TOP_REG(DL2_RP, DL2_INT_STATUS));
#if defined(CONTROL_UNIT_MODEL)
        // The model doesn't correctly simulate the DL2_INT_STATUS register, so we assume
        // something interesting happened.
        intStatus.set_tsu_evnt(1);
#endif
    } while (intStatus.word == 0);
}

/// Config GLOBAL.PLE_MCEIF_CONFIG register
template <typename Hal>
void HwAbstraction<Hal>::ConfigMcePle(const MceS& agentData)
{
    m_Hal.m_Logger.Debug("Execute ConfigMcePle");

    m_Hal.WriteReg(TOP_REG(GLOBAL_RP, GLOBAL_PLE_MCEIF_CONFIG), agentData.PLE_MCEIF_CONFIG);
}

/// Start Mce stripe
template <typename Hal>
profiling::ProfilingOnly<uint8_t> HwAbstraction<Hal>::HandleStartMceStripeBank(const MceS& agentData,
                                                                               const StartMceStripeCommand& cmd)
{
    ETHOSN_UNUSED(agentData);

    ASSERT_MSG(m_NumCesEnabled == static_cast<CeEnables>(cmd.CE_ENABLES), "CE enables not configured correctly");

    m_Hal.m_Logger.Debug("Execute %s", ToString(cmd).GetCString());

    profiling::ProfilingOnly<uint8_t> profilingEntryId = m_Profiling.RecordStart(TimelineEventType::MceStripe);

    stripe_bank_control_r stripeBankControl;
    stripeBankControl.set_start(1);
    m_Hal.WriteReg(TOP_REG(GLOBAL_RP, GLOBAL_STRIPE_BANK_CONTROL), stripeBankControl.word);

    return profilingEntryId;
}

template <typename Hal>
inline bool HwAbstraction<Hal>::TrySetCeEnables(CeEnables ceEnables)
{
    ASSERT(ceEnables != CeEnables::Unknown);
    if (m_NumCesEnabled == ceEnables)
    {
        return true;    // Already configured as requested, nothing to do
    }

    // We must avoid changing the registers while other stripes are running, as the CE_CE_ENABLES registers
    // are not banked and so changes will take effect immediately. Note that it's fine to change the register
    // if just the PLE is running, as we're only actually enabling/disabling the MAC units, which have no effect
    // on the PLE. Therefore we only check if the MCE is running or not. It would actually be wrong to check the
    // PLE as well here, as it could result in a deadlock if the PLE was just started and waiting for the MCE
    // to start, but the MCE could never start because it was waiting to set the CE_CE_ENABLES registers.
    if (GetNumCmdsInMceQueue() > 0)
    {
        return false;
    }

    // ce_enables_r for CEs with active OGs
    ce_enables_r ceEnablesActive;
    ceEnablesActive.set_ce_enable(1);
    ceEnablesActive.set_mce_enable(1);
    ceEnablesActive.set_mac_enable(0XFFU);

    // ce_enables_r for CEs with inactive OGs
    ce_enables_r ceEnablesInactive;
    ceEnablesInactive.set_ce_enable(1);
    ceEnablesInactive.set_mce_enable(1);
    ceEnablesInactive.set_mac_enable(1);

    // We need to set the register for each CE, but in the simple cases of all enabled or all
    // disabled, we can optimise this by using a single broadcast write instead
    if (ceEnables == CeEnables::AllDisabled)
    {
        m_Hal.WriteReg(TOP_REG(CE_RP, CE_CE_ENABLES), ceEnablesInactive.word);
    }
    else if (ceEnables == static_cast<CeEnables>(static_cast<uint32_t>(m_Hal.NumCes())))
    {
        m_Hal.WriteReg(TOP_REG(CE_RP, CE_CE_ENABLES), ceEnablesActive.word);
    }
    else if (ceEnables == CeEnables::AllEnabledForPleOnly)
    {
        ce_enables_r ceEnablesPleOnly{};
        ceEnablesPleOnly.set_ce_enable(1);
        m_Hal.WriteReg(TOP_REG(CE_RP, CE_CE_ENABLES), ceEnablesPleOnly.word);
    }
    else
    {
        // Disable all initially
        m_Hal.WriteReg(TOP_REG(CE_RP, CE_CE_ENABLES), ceEnablesInactive.word);

        // Enable the CEs that actually need to process data
        for (uint32_t ce = 0; ce < static_cast<uint32_t>(ceEnables); ++ce)
        {
            m_Hal.WriteReg(CE_REG(ce, CE_RP, CE_CE_ENABLES), ceEnablesActive.word);
        }
    }

    m_NumCesEnabled = ceEnables;

    return true;
}

template <typename Hal>
void HwAbstraction<Hal>::RecordProfilingCounters()
{
    m_Profiling.RecordCounter(FirmwareCounterName::DwtSleepCycleCount, profiling::GetDwtSleepCycleCount());
    m_Profiling.RecordCounter(FirmwareCounterName::DmaNumReads, m_DmaRdNumTransactions);
    m_Profiling.RecordCounter(FirmwareCounterName::DmaNumWrites, m_DmaWrNumTransactions);
    m_Profiling.RecordCounter(FirmwareCounterName::DmaReadBytes, m_DmaRdTotalBytesTransferred);
    m_Profiling.RecordCounter(FirmwareCounterName::DmaWriteBytes, m_DmaWrTotalBytesTransferred);
    m_Profiling.RecordHwCounters();
}

template <typename Hal>
bool HwAbstraction<Hal>::IsFinished()
{
    return GetNumCmdsInDmaRdQueue() == 0 && GetNumCmdsInMceQueue() == 0 && !IsPleBusy() &&
           GetNumCmdsInDmaWrQueue() == 0;
}

template <typename Hal>
void HwAbstraction<Hal>::StoreDebugGpRegister(uint32_t gpNum, uint32_t value)
{
    ASSERT(gpNum <= 6);    // GP 7 is used for the mailbox address, which needs to be preserved
    uint32_t reg = DL1_GP0 + (DL1_GP1 - DL1_GP0) * gpNum;
    m_Hal.WriteReg(TOP_REG(DL1_RP, reg), value);
}

}    // namespace ethosn::control_unit
