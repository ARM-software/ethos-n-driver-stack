//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../MacroUtils.hpp"
#include "PleKernelId.hpp"

#include <ethosn_utils/SmallVector.hpp>
#include <ethosn_utils/Span.hpp>

#include <array>
#include <cstdint>

namespace ethosn
{
namespace command_stream
{
namespace cascading
{

/// Ifm Streamer data, which is the same for every stripe of this agent.
struct IfmS
{
    /// Identifies which DRAM buffer in the buffer table is copied from.
    uint16_t bufferId;

    /// Register values for the DMA, which are set the same for every stripe of this agent.
    /// @{
    uint32_t DMA_COMP_CONFIG0;
    uint32_t DMA_STRIDE1;
    uint32_t DMA_STRIDE2;
    /// @}
};

/// Output Streamer data, which is the same for every stripe of this agent.
struct OfmS
{
    /// Identifies which DRAM buffer in the buffer table is copied to.
    uint16_t bufferId;

    /// Register values for the DMA, which are set the same for every stripe of this agent.
    /// @{
    uint32_t DMA_COMP_CONFIG0;
    uint32_t DMA_STRIDE1;
    uint32_t DMA_STRIDE2;
    /// @}
};

/// Weight Streamer data, which is the same for every stripe
struct WgtS
{
    /// Identifies which DRAM buffer in the buffer table is copied from.
    uint16_t bufferId;
};

/// The type of MCE operation this is (regular convolution/depthwise/fully connected)
enum class MceOperation : uint8_t
{
    CONVOLUTION,
    DEPTHWISE_CONVOLUTION,
    FULLY_CONNECTED,
};

/// Mce Scheduler data, which is the same for every stripe
struct MceS
{
    /// The type of MCE operation this is (regular convolution/depthwise/fully connected)
    MceOperation mceOpMode;
    /// Which PLE kernel will be used to process the output of the MCE.
    PleKernelId pleKernelId;

    /// Register values for the MCE, which are set the same for every stripe of this agent.
    /// @{
    uint32_t ACTIVATION_CONFIG;
    uint32_t WIDE_KERNEL_CONTROL;
    uint32_t FILTER;
    uint32_t IFM_ZERO_POINT;
    uint32_t IFM_DEFAULT_SLOT_SIZE;
    uint32_t IFM_SLOT_STRIDE;
    uint32_t STRIPE_BLOCK_CONFIG;
    uint32_t DEPTHWISE_CONTROL;
    uint32_t IFM_SLOT_BASE_ADDRESS;
    /// @}

    /// Register value for the MCEIF, which is the same for every stripe of this agent.
    uint32_t PLE_MCEIF_CONFIG;
};

/// PLE Loader data, which is the same for every stripe
struct PleL
{
    /// ID of the kernel that should be loaded into SRAM.
    PleKernelId pleKernelId;
};

/// MCE operation by fused PLE, or only PLE
enum class PleInputMode : uint8_t
{
    /// Input from MCE, all OGs are active (CONVOLUTION or fully connected)
    MCE_ALL_OGS,
    /// Input from MCE, only one OG is active (DEPTHWISE_CONVOLUTION)
    MCE_ONE_OG,
    /// MCE is inactive, read input data from SRAM
    SRAM_ONE_INPUT,
    SRAM_TWO_INPUTS,
};

/// PLE Scheduler data, which is the same for every stripe
struct PleS
{
    /// Source of input data to PLE
    PleInputMode inputMode;
    /// ID of the kernel that should be loaded into and executed on the PLE.
    PleKernelId pleKernelId;
    /// PLE kernel location in SRAM
    uint32_t pleKernelSramAddr;
};

/// Enum tag for agent data
enum class AgentType : uint32_t
{
    IFM_STREAMER,
    WGT_STREAMER,
    MCE_SCHEDULER,
    PLE_LOADER,
    PLE_SCHEDULER,
    OFM_STREAMER,
};

/// Immutable tagged union of agent data that can only be constructed from the concrete agent data type.
/// The corresponding constructor overload will set the enum tag accordingly. Note that constructors are
/// intentionally not explicit because implicit conversion is desirable for cleaner code.
struct AgentData
{
    const AgentType type;

    union
    {
        const IfmS ifm;
        const WgtS wgt;
        const MceS mce;
        const PleL pleL;
        const PleS pleS;
        const OfmS ofm;
    };

    constexpr AgentData(const AgentData& other) = default;

    AgentData& operator=(const AgentData& other)
    {
        return *new (this) AgentData{ other };
    }

    constexpr AgentData(const IfmS& data)
        : type{ AgentType::IFM_STREAMER }
        , ifm{ data }
    {}

    constexpr AgentData(const WgtS& data)
        : type{ AgentType::WGT_STREAMER }
        , wgt{ data }
    {}

    constexpr AgentData(const MceS& data)
        : type{ AgentType::MCE_SCHEDULER }
        , mce{ data }
    {}

    constexpr AgentData(const PleL& data)
        : type{ AgentType::PLE_LOADER }
        , pleL{ data }
    {}

    constexpr AgentData(const PleS& data)
        : type{ AgentType::PLE_SCHEDULER }
        , pleS{ data }
    {}

    constexpr AgentData(const OfmS& data)
        : type{ AgentType::OFM_STREAMER }
        , ofm{ data }
    {}
};

/// Contains both common data (common to all types of agent) and tagged data (specific for
/// an agent type) for an agent
struct Agent
{
    // Total number of stripes for this Agent including reloads (if any)
    uint16_t numStripesTotal;
    /// Agent-type-specific data
    AgentData data;
};

// "Extra data" can be associated with Commands.
// This can be different for each stripe in an agent, as opposed to data in the Agent types (e.g. IfmS)
// which is the same across all stripes.

/// Extra data associated with LoadIfmStripe, LoadWgtStripe, LoadPleCode and StoreOfmStripe Commands,
/// which is different for every stripe
struct DmaExtraData
{
    /// Offset in bytes into the DRAM buffer to start the DMA.
    uint32_t m_DramOffset;

    /// Register values for the DMA, which are set differently for each stripe of the agent.
    /// @{
    uint32_t SRAM_ADDR;
    uint32_t DMA_SRAM_STRIDE;
    uint32_t DMA_STRIDE0;
    uint32_t DMA_STRIDE3;
    uint32_t DMA_CHANNELS;

    uint32_t DMA_EMCS;
    uint32_t DMA_TOTAL_BYTES;
    uint32_t DMA_CMD;
    /// @}

    /// Some stripes require multiple DMA commands (each called a 'chunk').
    /// This field indicates if this is the last chunk for the stripe, otherwise further commands
    /// need to be completed before the stripe is complete.
    uint8_t m_IsLastChunk;
};

/// Extra data associated with ProgramMceStripe Commands,
/// which is different for every stripe
struct ProgramMceExtraData
{
    /// Register values for the MCE, which are set differently for each stripe of the agent.
    /// @{
    uint32_t CE_CONTROL;
    std::array<std::array<uint32_t, 4>, 8> MUL_ENABLE;    // Indexed by CE then OG
    uint32_t IFM_ROW_STRIDE;
    uint32_t IFM_CONFIG1;
    std::array<std::array<uint32_t, 4>, 4> IFM_PAD;    // Indexed by subfilter number then IG
    uint32_t WIDE_KERNEL_OFFSET;
    uint32_t IFM_TOP_SLOTS;
    uint32_t IFM_MID_SLOTS;
    uint32_t IFM_BOTTOM_SLOTS;
    uint32_t IFM_SLOT_PAD_CONFIG;
    uint32_t OFM_STRIPE_SIZE;
    uint32_t OFM_CONFIG;
    std::array<uint32_t, 4> WEIGHT_BASE_ADDR;              // Indexed by OG
    std::array<std::array<uint32_t, 4>, 8> IFM_CONFIG2;    // Indexed by CE then IG
    /// @}

    /// How many blocks will this MCE command send to the PLE.
    uint32_t m_NumBlocksProgrammedForMce;
};

/// Extra data associated with StartMceStripe Commands,
/// which is different for every stripe
struct StartMceExtraData
{
    /// Register value.
    uint32_t CE_ENABLES;
};

/// Extra data associated with StartPleStripe Commands,
/// which is different for every stripe
struct StartPleExtraData
{
    /// Register values.
    std::array<uint32_t, 8> SCRATCH;
};

enum class CommandType : uint32_t
{
    WaitForAgent,
    LoadIfmStripe,
    LoadWgtStripe,
    ProgramMceStripe,
    StartMceStripe,
    LoadPleCode,
    StartPleStripe,
    StoreOfmStripe,
};

/// Generic command stored which is stored in four lists for the firmware to execute.
struct Command
{
    CommandType type;
    uint32_t agentId;
    uint32_t stripeId;
    /// Some types of command have extra associated data, which is stored in a different array
    /// in the command stream.
    /// This offset (in bytes) is from the start of this Command struct to the start of that struct.
    /// The type of the extra data depends on the type of this Command.
    /// Some Commands don't have any extra data, in which case this would be set to zero.
    uint32_t extraDataOffset;

    /// Helpers to get access to any extra data. No type checking is performed (see above)!
    /// @{
    const DmaExtraData& GetDmaExtraData() const
    {
        return *reinterpret_cast<const DmaExtraData*>(reinterpret_cast<const char*>(this) + extraDataOffset);
    }
    const ProgramMceExtraData& GetProgramMceExtraData() const
    {
        return *reinterpret_cast<const ProgramMceExtraData*>(reinterpret_cast<const char*>(this) + extraDataOffset);
    }
    const StartMceExtraData& GetStartMceExtraData() const
    {
        return *reinterpret_cast<const StartMceExtraData*>(reinterpret_cast<const char*>(this) + extraDataOffset);
    }
    const StartPleExtraData& GetStartPleExtraData() const
    {
        return *reinterpret_cast<const StartPleExtraData*>(reinterpret_cast<const char*>(this) + extraDataOffset);
    }
    /// @}
};

}    // namespace cascading
}    // namespace command_stream
}    // namespace ethosn
