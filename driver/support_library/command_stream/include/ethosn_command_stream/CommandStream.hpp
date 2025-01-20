//
// Copyright © 2018-2025 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "PleKernelIds.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

#define ETHOSN_COMMAND_STREAM_VERSION_MAJOR 8
#define ETHOSN_COMMAND_STREAM_VERSION_MINOR 0
#define ETHOSN_COMMAND_STREAM_VERSION_PATCH 0

namespace ethosn
{
namespace command_stream
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

/// Tagged union of agent data that can only be constructed from the concrete agent data type.
/// The corresponding constructor overload will set the enum tag accordingly. Note that constructors are
/// intentionally not explicit because implicit conversion is desirable for cleaner code.
struct Agent
{
    AgentType type;

    union
    {
        IfmS ifm;
        WgtS wgt;
        MceS mce;
        PleL pleL;
        PleS pleS;
        OfmS ofm;
    };

    Agent(const Agent& other) = default;

    Agent(const IfmS& data)
    {
        // Clear unused padding due to union, to avoid uninitalised data in command stream
        memset(static_cast<void*>(this), 0, sizeof(*this));
        type = AgentType::IFM_STREAMER;
        ifm  = data;
    }

    Agent(const WgtS& data)
    {
        // Clear unused padding due to union, to avoid uninitalised data in command stream
        memset(static_cast<void*>(this), 0, sizeof(*this));
        type = AgentType::WGT_STREAMER;
        wgt  = data;
    }

    Agent(const MceS& data)
    {
        // Clear unused padding due to union, to avoid uninitalised data in command stream
        memset(static_cast<void*>(this), 0, sizeof(*this));
        type = AgentType::MCE_SCHEDULER;
        mce  = data;
    }

    Agent(const PleL& data)
    {
        // Clear unused padding due to union, to avoid uninitalised data in command stream
        memset(static_cast<void*>(this), 0, sizeof(*this));
        type = AgentType::PLE_LOADER;
        pleL = data;
    }

    Agent(const PleS& data)
    {
        // Clear unused padding due to union, to avoid uninitalised data in command stream
        memset(static_cast<void*>(this), 0, sizeof(*this));
        type = AgentType::PLE_SCHEDULER;
        pleS = data;
    }

    Agent(const OfmS& data)
    {
        // Clear unused padding due to union, to avoid uninitalised data in command stream
        memset(static_cast<void*>(this), 0, sizeof(*this));
        type = AgentType::OFM_STREAMER;
        ofm  = data;
    }
};

enum class CommandType : uint32_t
{
    WaitForCounter,
    LoadIfmStripe,
    LoadWgtStripe,
    ProgramMceStripe,
    ConfigMceif,
    StartMceStripe,
    LoadPleCodeIntoSram,
    LoadPleCodeIntoPleSram,
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
    Mceif,
    MceStripe,
    PleCodeLoadedIntoPleSram,
    PleStripe,
};

/// Data for CommandType::WaitForCounter, which describes waiting for a progress
/// counter to reach a certain value.
struct WaitForCounterCommand : public Command
{
    CounterName counterName;
    uint32_t counterValue;
};

/// Data for CommandType::LoadIfmStripe, LoadWgtStripe, LoadPleCodeIntoSram and StoreOfmStripe,
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
    uint32_t DMA_STRIDE2;    // Might differ per-stripe for NCHW
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

/// Data for CommandType::ConfigMceif,
/// which describes configuring the MCEIF ready for an MCE stripe.
struct ConfigMceifCommand : public Command
{
    uint32_t agentId;
};

/// Data for CommandType::StartMceStripe,
/// which describes kicking off an MCE stripe.
struct StartMceStripeCommand : public Command
{
    uint32_t agentId;
    /// Register value.
    uint32_t CE_ENABLES;
};

/// Data for CommandType::LoadPleCodeIntoPleSram,
/// which describes UDMA'ing the PLE code from SRAM into Ple SRAM.
struct LoadPleCodeIntoPleSramCommand : public Command
{
    uint32_t agentId;
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
        case CommandType::ConfigMceif:
            return sizeof(ConfigMceifCommand);
        case CommandType::StartMceStripe:
            return sizeof(StartMceStripeCommand);
        case CommandType::LoadPleCodeIntoSram:
            return sizeof(DmaCommand);
        case CommandType::LoadPleCodeIntoPleSram:
            return sizeof(LoadPleCodeIntoPleSramCommand);
        case CommandType::StartPleStripe:
            return sizeof(StartPleStripeCommand);
        case CommandType::StoreOfmStripe:
            return sizeof(DmaCommand);
        default:
            return 0;
    }
}

struct CommandStream
{
    /// Total size (in bytes) of all the data in this CommandStream. This includes the size of this struct,
    /// plus the data which follows it (array of Agents and lists of mixed-type Commands).
    uint32_t TotalSize;

    /// Offset (in bytes) from the start of this struct to the array of agents.
    uint32_t AgentsOffset;
    uint32_t NumAgents;

    /// Offset (in bytes) from the start of this struct to the DMA read commands.
    uint32_t DmaRdCommandsOffset;
    uint32_t NumDmaRdCommands;

    /// Offset (in bytes) from the start of this struct to the DMA write commands.
    uint32_t DmaWrCommandsOffset;
    uint32_t NumDmaWrCommands;

    /// Offset (in bytes) from the start of this struct to the MCE commands.
    uint32_t MceCommandsOffset;
    uint32_t NumMceCommands;

    /// Offset (in bytes) from the start of this struct to the PLE commands.
    uint32_t PleCommandsOffset;
    uint32_t NumPleCommands;

    // Following this struct there will be an array of Agent then four
    // lists of mixed-type Commands.
    // The above fields describe this layout, and the below methods provide easy access to them.

    const Agent* GetAgentsArray() const
    {
        const char* basePtr = reinterpret_cast<const char*>(this);
        return reinterpret_cast<const Agent*>(basePtr + AgentsOffset);
    }
    const Command* GetDmaRdCommandsBegin() const
    {
        const char* basePtr = reinterpret_cast<const char*>(this);
        return reinterpret_cast<const Command*>(basePtr + DmaRdCommandsOffset);
    }
    const Command* GetDmaWrCommandsBegin() const
    {
        const char* basePtr = reinterpret_cast<const char*>(this);
        return reinterpret_cast<const Command*>(basePtr + DmaWrCommandsOffset);
    }
    const Command* GetMceCommandsBegin() const
    {
        const char* basePtr = reinterpret_cast<const char*>(this);
        return reinterpret_cast<const Command*>(basePtr + MceCommandsOffset);
    }
    const Command* GetPleCommandsBegin() const
    {
        const char* basePtr = reinterpret_cast<const char*>(this);
        return reinterpret_cast<const Command*>(basePtr + PleCommandsOffset);
    }
    const char* GetEndAddress() const
    {
        const char* basePtr = reinterpret_cast<const char*>(this);
        return basePtr + TotalSize;
    }
};

class CommandStreamParser
{
public:
    CommandStreamParser(const void* rawBegin, const void* rawEnd)
        : m_VersionMajor(0)
        , m_VersionMinor(0)
        , m_VersionPatch(0)
        , m_Data(nullptr)
    {
        const uint32_t* rawBeginU32 = reinterpret_cast<const uint32_t*>(rawBegin);
        const uint32_t* rawEndU32   = reinterpret_cast<const uint32_t*>(rawEnd);

        constexpr ptrdiff_t versionHeaderSizeWords = 4;
        if (rawEndU32 - rawBeginU32 < versionHeaderSizeWords)
        {
            return;
        }

        const uint32_t fourcc             = rawBeginU32[0];
        constexpr uint32_t expectedFourcc = static_cast<uint32_t>('E') | (static_cast<uint32_t>('N') << 8) |
                                            (static_cast<uint32_t>('C') << 16) | (static_cast<uint32_t>('S') << 24);
        if (fourcc != expectedFourcc)
        {
            return;
        }

        m_VersionMajor = rawBeginU32[1];
        m_VersionMinor = rawBeginU32[2];
        m_VersionPatch = rawBeginU32[3];
        if (m_VersionMajor != ETHOSN_COMMAND_STREAM_VERSION_MAJOR ||
            m_VersionMinor != ETHOSN_COMMAND_STREAM_VERSION_MINOR ||
            m_VersionPatch != ETHOSN_COMMAND_STREAM_VERSION_PATCH)
        {
            return;
        }

        m_Data = reinterpret_cast<const CommandStream*>(rawBeginU32 + versionHeaderSizeWords);
    }

    bool IsValid() const
    {
        return m_Data != nullptr;
    }

    const CommandStream* GetData() const
    {
        return m_Data;
    }

    uint32_t GetVersionMajor() const
    {
        return m_VersionMajor;
    }
    uint32_t GetVersionMinor() const
    {
        return m_VersionMinor;
    }
    uint32_t GetVersionPatch() const
    {
        return m_VersionPatch;
    }

private:
    uint32_t m_VersionMajor;
    uint32_t m_VersionMinor;
    uint32_t m_VersionPatch;
    const CommandStream* m_Data;
};

}    // namespace command_stream
}    // namespace ethosn
