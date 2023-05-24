//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "DmaRegisters.hpp"
#include "MceRegisters.hpp"
#include "PleRegisters.hpp"

#include "../include/ethosn_support_library/Optional.hpp"
#include "ethosn_command_stream/CommandStreamBuffer.hpp"

namespace ethosn
{
namespace support_library
{

struct DebuggingContext;

/// Used to represent a ratio in the number of stripes of this/other agent
/// that are needed by other/this agent
struct DependencyRatio
{
    uint16_t other;
    uint16_t self;
};

/// Used to represent a dependency between this agent and some other agent
struct Dependency
{
    /// Relative position of the other agent wrt the agent that owns this Dependency object.
    /// We can use unsigned type because it always references another agent, down the sequence
    /// for schedule and write-after-read dependencies, and up the sequence for read-after-write
    /// dependencies. The sign is implicit in that way. Using unsigned for extra range.
    uint8_t relativeAgentId;
    /// In the presence of reloads, the number of stripes in self/other in each reload.
    DependencyRatio outerRatio;
    /// Ratio between stripe counters. E.g. two Ifm Streamer stripes might be needed for each
    /// stripe of the consumer Mce Scheduler
    DependencyRatio innerRatio;
    /// Extra number of stripes that are needed. E.g. 3x3 conv:
    ///    IfmS stripes  MceS stripes
    ///            +        *
    ///            |        |
    ///            +        | +
    ///            |        | |
    ///            +        * *
    ///            |        | |
    ///            +        + | +
    ///            |          | |
    ///            +          * *
    ///            |          | |
    ///            +          + |  <- innerRatio[IfmS] = 1 / 2
    ///            |            |
    ///            +            *
    ///            |            |  <- boundary = 1
    ///            +            +
    int8_t boundary;
};

/// Contains dependency info for an agent
struct AgentDependencyInfo
{
    /// Array of read-after-write dependencies.
    std::vector<Dependency> readDependencies;
    /// Array of write-after-read dependencies related to a tile size. The agent should pause progress before
    /// overwriting a slot in the tile until the existing data is no longer needed by any reader agent.
    std::vector<Dependency> writeDependencies;
};

/// This is the support library's intermediate representation of an agent, which contains more details than
/// the final command stream representation.
struct AgentDesc
{
    uint16_t numStripesTotal;

    command_stream::cascading::AgentType type;

    union
    {
        IfmSDesc ifm;
        WgtSDesc wgt;
        MceSDesc mce;
        PleLDesc pleL;
        PleSDesc pleS;
        OfmSDesc ofm;
    };

    explicit AgentDesc(uint16_t numStripesTotal, const IfmSDesc& data)
        : numStripesTotal(numStripesTotal)
        , type{ command_stream::cascading::AgentType::IFM_STREAMER }
        , ifm{ data }
    {}

    explicit AgentDesc(uint16_t numStripesTotal, const WgtSDesc& data)
        : numStripesTotal(numStripesTotal)
        , type{ command_stream::cascading::AgentType::WGT_STREAMER }
        , wgt{ data }
    {}

    explicit AgentDesc(uint16_t numStripesTotal, const MceSDesc& data)
        : numStripesTotal(numStripesTotal)
        , type{ command_stream::cascading::AgentType::MCE_SCHEDULER }
        , mce{ data }
    {}

    explicit AgentDesc(uint16_t numStripesTotal, const PleLDesc& data)
        : numStripesTotal(numStripesTotal)
        , type{ command_stream::cascading::AgentType::PLE_LOADER }
        , pleL{ data }
    {}

    explicit AgentDesc(uint16_t numStripesTotal, const PleSDesc& data)
        : numStripesTotal(numStripesTotal)
        , type{ command_stream::cascading::AgentType::PLE_SCHEDULER }
        , pleS{ data }
    {}

    explicit AgentDesc(uint16_t numStripesTotal, const OfmSDesc& data)
        : numStripesTotal(numStripesTotal)
        , type{ command_stream::cascading::AgentType::OFM_STREAMER }
        , ofm{ data }
    {}
};

struct AgentDescAndDeps
{
    AgentDesc agent;
    AgentDependencyInfo deps;
};

/// Stores a value for each of the firmware counters.
struct Counters
{
    uint32_t m_DmaRd                    = 0;
    uint32_t m_DmaWr                    = 0;
    uint32_t m_Mceif                    = 0;
    uint32_t m_MceStripe                = 0;
    uint32_t m_PleCodeLoadedIntoPleSram = 0;
    uint32_t m_PleStripe                = 0;

    uint32_t Get(ethosn::command_stream::cascading::CounterName counterName) const;
    void Set(ethosn::command_stream::cascading::CounterName counterName, uint32_t value);

    static Counters Max(const Counters& a, const Counters& b);
};

/// Logic for converting a list of agents with dependency information into four
/// lists of commands (Dma read, Dma write, Mce and Ple) to be executed by the firmware.
class Scheduler
{
public:
    Scheduler(const std::vector<AgentDescAndDeps>& agents,
              const HardwareCapabilities& capabilities,
              const DebuggingContext& debuggingContext);

    void Schedule();

    const std::vector<ethosn::command_stream::CommandVariant>& GetDmaRdCommands() const;
    const std::vector<ethosn::command_stream::CommandVariant>& GetDmaWrCommands() const;
    const std::vector<ethosn::command_stream::CommandVariant>& GetMceCommands() const;
    const std::vector<ethosn::command_stream::CommandVariant>& GetPleCommands() const;

private:
    /// Adding a WaitForCounter will often mean implicitly waiting for other counter values too.
    /// A trivial example of this would be that waiting for DmaRd=2 also means waiting for DmaRd=1,
    /// but there are more complicated examples like waiting for MceStripe=2 where the MCE queue
    /// waits for DmaRd=1 before kicking off stripe number 2, means that you are implicitly waiting
    /// for DmaRd=1 as well.
    /// This object stores these dependencies/implications, and allows us to omit some WaitForCounters
    /// which we can guarantee will always be met.
    class CounterImplications
    {
    public:
        /// Gets the minimum value of each counter which we can guarantee will have been reached,
        /// when the given counter reaches the given value.
        Counters Get(ethosn::command_stream::cascading::CounterName counterName, uint32_t value) const;

        /// Records that when the given counter reaches the given value, the other counters will have
        /// at least the values given.
        void Update(ethosn::command_stream::cascading::CounterName counterName, uint32_t value, Counters counters);

    private:
        /// Internal storage - for each counter name and value pair, the value that we can guarantee the
        /// other counters will have reached.
        std::map<std::pair<ethosn::command_stream::cascading::CounterName, uint32_t>, Counters> m_Map;
    };

    /// Wraps a list of Commands along with storage of which counter values were last waited
    /// on. This allows us to avoid inserting redundant WaitForCounter commands on
    /// counters which we can guarantee will have already passed that value.
    class CommandQueue
    {
    public:
        CommandQueue(const CounterImplications& counterImplications)
            : m_CounterImplications(counterImplications)
        {}

        void Push(const ethosn::command_stream::CommandVariant& c);
        const std::vector<ethosn::command_stream::CommandVariant>& GetCommands() const;

        const Counters& GetLastCounterValuesWaitedFor() const
        {
            return m_LastCounterValuesWaitedFor;
        }

    private:
        std::vector<ethosn::command_stream::CommandVariant> m_Commands;

        /// Reference to data shared between the different queues, for eliminating redundant WaitForCounters.
        const CounterImplications& m_CounterImplications;

        /// The maximum value of each firmware counter which we know has been reached by the time we get
        /// to the current point in this command queue.
        Counters m_LastCounterValuesWaitedFor;
    };

    /// Schedules the next stripe for the given agent.
    /// Also advances the progress for the given agent.
    void ScheduleOneStripe(const uint32_t agentId);

    void InsertWriteDependencies(const AgentDependencyInfo& agent,
                                 const uint32_t agentId,
                                 const uint32_t stripeId,
                                 const uint16_t tileSize,
                                 CommandQueue& commands);
    void InsertReadDependencies(const AgentDependencyInfo& agent,
                                const uint32_t agentId,
                                const uint32_t stripeId,
                                const utils::Optional<command_stream::cascading::AgentType> agentTypeToIgnore,
                                CommandQueue& commands);

    void ScheduleIfmStreamerStripe(const uint32_t agentId, uint32_t stripeId);
    void ScheduleWgtStreamerStripe(const uint32_t agentId, uint32_t stripeId);
    void ScheduleMceSchedulerStripe(const uint32_t agentId, uint32_t stripeId);
    void SchedulePleLoaderStripe(const uint32_t agentId, uint32_t stripeId);
    void SchedulePleSchedulerStripe(const uint32_t agentId, uint32_t stripeId);
    void ScheduleOfmStreamerStripe(const uint32_t agentId, uint32_t stripeId);

    CommandQueue& GetQueueForAgentType(command_stream::cascading::AgentType agentType);
    void PushWaitForCounterCommand(command_stream::cascading::AgentType otherAgentType,
                                   uint32_t otherAgentId,
                                   uint32_t otherStripeId,
                                   CommandQueue& commands);

    const DebuggingContext& m_DebuggingContext;

    /// The list of agents that this Scheduler will process.
    const std::vector<AgentDescAndDeps>& m_Agents;

    /// Keeps track of the next stripe that needs to be scheduled for each agent.
    std::vector<uint32_t> m_AgentProgress;

    CommandQueue m_DmaRdCommands;
    CommandQueue m_DmaWrCommands;
    CommandQueue m_MceCommands;
    CommandQueue m_PleCommands;

    uint32_t m_NextRdDmaCmdId = 0;
    uint32_t m_NextWrDmaCmdId = 4;

    /// Map from agent ID and stripe ID to the value that a firmware counter will have
    /// when that stripe is finished.
    /// @{
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> m_DmaRdCounters;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> m_DmaWrCounters;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> m_MceStripeCounters;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> m_PleStripeCounters;
    /// @}

    /// The value that each of the firmware counters will have after the stripes that have already been scheduled
    /// have finished.
    Counters m_Counters;

    /// Adding a WaitForCounter on a particular counter value will often mean implicitly waiting for
    /// other counter values too. This stores those dependencies, and allows us to omit some WaitForCounters
    /// which we can guarantee will always be met.
    CounterImplications m_CounterImplications;

    command_stream::cascading::PleKernelId m_MceifConfiguration  = command_stream::cascading::PleKernelId::NOT_FOUND;
    command_stream::cascading::PleKernelId m_LastLoadedPleKernel = command_stream::cascading::PleKernelId::NOT_FOUND;

    const HardwareCapabilities& m_Capabilities;
};

}    // namespace support_library
}    // namespace ethosn
