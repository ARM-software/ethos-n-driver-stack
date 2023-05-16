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
#include <cstddef>
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
struct Agent
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

    constexpr Agent(const Agent& other) = default;

    Agent& operator=(const Agent& other)
    {
        return *new (this) Agent{ other };
    }

    constexpr Agent(const IfmS& data)
        : type{ AgentType::IFM_STREAMER }
        , ifm{ data }
    {}

    constexpr Agent(const WgtS& data)
        : type{ AgentType::WGT_STREAMER }
        , wgt{ data }
    {}

    constexpr Agent(const MceS& data)
        : type{ AgentType::MCE_SCHEDULER }
        , mce{ data }
    {}

    constexpr Agent(const PleL& data)
        : type{ AgentType::PLE_LOADER }
        , pleL{ data }
    {}

    constexpr Agent(const PleS& data)
        : type{ AgentType::PLE_SCHEDULER }
        , pleS{ data }
    {}

    constexpr Agent(const OfmS& data)
        : type{ AgentType::OFM_STREAMER }
        , ofm{ data }
    {}
};

enum class CommandType : uint32_t
{
    WaitForCounter,
    LoadIfmStripe,
    LoadWgtStripe,
    ProgramMceStripe,
    StartMceStripe,
    LoadPleCode,
    StartPleStripe,
    StoreOfmStripe,
};

/// Base command type. The four lists of commands for the firmware to execute are
/// contiguously stored lists of structs which are derived from this type.
/// The first field (`type`) identifies which kind of Command it is.
/// Note that this means the size of each Command in a list could be different.
struct Command
{
    CommandType type;

    /// Commands will always be a sub-type of this one.
    /// This gets the size of the actual command struct, based on the `type` field.
    size_t GetSize() const;
};

enum class CounterName : uint32_t
{
    DmaRd,
    DmaWr,
    MceStripe,
    PleStripe,
};

/// Data for CommandType::WaitForCounter, which describes waiting for a progress
/// counter to reach a certain value.
struct WaitForCounterCommand : public Command
{
    CounterName counterName;
    uint32_t counterValue;
};

/// Data for CommandType::LoadIfmStripe, LoadWgtStripe, LoadPleCode and StoreOfmStripe,
/// which describes transferring some data between Dram and Sram.
struct DmaCommand : public Command
{
    uint32_t agentId;

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
};

/// Data for CommandType::ProgramMceStripe,
/// which describes setting up MCE registers.
struct ProgramMceStripeCommand : public Command
{
    uint32_t agentId;

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

/// Data for CommandType::StartMceStripe,
/// which describes kicking off an MCE stripe.
struct StartMceStripeCommand : public Command
{
    uint32_t agentId;
    /// Register value.
    uint32_t CE_ENABLES;
};

/// Data for CommandType::StartPleStripe,
/// which describes kicking off a PLE stripe.
struct StartPleStripeCommand : public Command
{
    uint32_t agentId;
    /// Register values.
    std::array<uint32_t, 8> SCRATCH;
};

inline size_t Command::GetSize() const
{
    switch (type)
    {
        case CommandType::WaitForCounter:
            return sizeof(WaitForCounterCommand);
        case CommandType::LoadIfmStripe:
            return sizeof(DmaCommand);
        case CommandType::LoadWgtStripe:
            return sizeof(DmaCommand);
        case CommandType::ProgramMceStripe:
            return sizeof(ProgramMceStripeCommand);
        case CommandType::StartMceStripe:
            return sizeof(StartMceStripeCommand);
        case CommandType::LoadPleCode:
            return sizeof(DmaCommand);
        case CommandType::StartPleStripe:
            return sizeof(StartPleStripeCommand);
        case CommandType::StoreOfmStripe:
            return sizeof(DmaCommand);
        default:
            return 0;
    }
}

}    // namespace cascading
}    // namespace command_stream
}    // namespace ethosn
