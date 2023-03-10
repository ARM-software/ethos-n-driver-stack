//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "Scheduler.hpp"

#include "../Utils.hpp"
#include "DmaRegisters.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>

using namespace ethosn::command_stream::cascading;

namespace ethosn
{
namespace support_library
{

namespace
{

/// Returns the stripe id of the agent down the sequence that first needs
/// stripe x of the current agent based on the dependency info.
constexpr int GetFirstReaderStripeId(const Dependency& dep, const uint32_t x)
{
    if (x == 0)
    {
        return 0;
    }

    const int outer = dep.outerRatio.other * (x / dep.outerRatio.self);

    int inner = (x % dep.outerRatio.self) - dep.boundary;
    inner     = dep.innerRatio.other * (inner / dep.innerRatio.self);
    inner     = std::min(std::max(inner, 0), dep.outerRatio.other - 1);

    return outer + inner;
}

/// Returns the largest stripe id of the producer agent up the sequence that needs
/// to be completed before stripe x of the current agent can start.
constexpr int GetLargestNeededStripeId(const Dependency& dep, const uint32_t x)
{
    const int outer = dep.outerRatio.other * (x / dep.outerRatio.self);

    int inner = x % dep.outerRatio.self;
    inner     = dep.innerRatio.other * (inner / dep.innerRatio.self);
    inner     = inner + dep.innerRatio.other - 1 + dep.boundary;
    inner     = std::min(std::max(inner, 0), dep.outerRatio.other - 1);

    return outer + inner;
}

/// Returns the stripe id of the agent down the sequence that last uses
/// stripe x of the current agent.
constexpr int GetLastReaderStripeId(const Dependency& dep, const uint32_t x)
{
    const int outer = dep.outerRatio.other * (x / dep.outerRatio.self);

    int inner = (x % dep.outerRatio.self) + dep.boundary;
    inner     = dep.innerRatio.other * (inner / dep.innerRatio.self);
    inner     = inner + dep.innerRatio.other - 1;
    inner     = std::min(std::max(inner, 0), dep.outerRatio.other - 1);

    return outer + inner;
}

/// Returns the stripe id of the agent down the sequence that last uses
/// stripe (x - tileSize) of the current agent.
inline int GetLastReaderOfEvictedStripeId(const Dependency& dep, const uint32_t x, const uint32_t tileSize)
{
    assert(x >= tileSize);
    return GetLastReaderStripeId(dep, x - tileSize);
}

}    // namespace

void Scheduler::InsertWriteDependencies(const AgentDependencyInfo& agent,
                                        const uint32_t agentId,
                                        const uint32_t stripeId,
                                        const uint16_t tileSize,
                                        std::vector<Command>& commands)
{
    if (stripeId < tileSize)
    {
        return;
    }
    for (const auto& writeDependency : agent.writeDependencies)
    {
        const int stripeToWaitFor = GetLastReaderOfEvictedStripeId(writeDependency, stripeId, tileSize);
        if (stripeToWaitFor >= 0)
        {
            // When stripeId == tileSize, it is needed to insert the wait because this is the first stripe that
            // overwrite something in the tile.
            if ((stripeId == tileSize) ||
                (stripeToWaitFor != GetLastReaderOfEvictedStripeId(writeDependency, stripeId - 1, tileSize)))
            {
                commands.push_back(Command{ CommandType::WaitForAgent, agentId + writeDependency.relativeAgentId,
                                            static_cast<uint32_t>(stripeToWaitFor), 0 });
            }
        }
    }
}

void Scheduler::InsertReadDependencies(const AgentDependencyInfo& agent,
                                       const uint32_t agentId,
                                       const uint32_t stripeId,
                                       const utils::Optional<AgentType> agentTypeToIgnore,
                                       std::vector<Command>& commands)
{
    for (const auto& readDependency : agent.readDependencies)
    {
        const uint32_t otherAgentId    = agentId - readDependency.relativeAgentId;
        const AgentType otherAgentType = m_Agents[otherAgentId].agent.type;

        if (utils::Optional<AgentType>{ otherAgentType } != agentTypeToIgnore)
        {
            const int stripeToWaitFor = GetLargestNeededStripeId(readDependency, stripeId);
            if (stripeToWaitFor >= 0)
            {
                // When the very first stripe is scheduled (i.e. stripeId == 0), it is needed to wait
                // that the read dependency is met before programing the stripe.
                if ((stripeId == 0) || (stripeToWaitFor != GetLargestNeededStripeId(readDependency, stripeId - 1)))
                {
                    commands.push_back(
                        Command{ CommandType::WaitForAgent, otherAgentId, static_cast<uint32_t>(stripeToWaitFor), 0 });
                }
            }
        }
    }
}

void Scheduler::ScheduleIfmStreamerStripe(const uint32_t agentId, uint32_t stripeId)
{
    const AgentDescAndDeps& agentAndDeps = m_Agents[agentId];
    assert(agentAndDeps.agent.type == AgentType::IFM_STREAMER);

    g_Logger.Verbose("Schedule IfmStreamerStripe { .agentId = %u, .stripeId = %d }", agentId, stripeId);

    const uint16_t tileSize = agentAndDeps.agent.ifm.fmData.tile.numSlots;
    InsertWriteDependencies(agentAndDeps.deps, agentId, stripeId, tileSize, m_DmaRdCommands);
    InsertReadDependencies(agentAndDeps.deps, agentId, stripeId, {}, m_DmaRdCommands);

    const uint32_t numChunks = CalculateNumChunks(agentAndDeps.agent.ifm, stripeId);
    for (uint32_t i = 0; i < numChunks; ++i)
    {
        m_DmaRdCommands.push_back(Command{ CommandType::LoadIfmStripe, agentId, stripeId, 0 });
    }
}

void Scheduler::ScheduleWgtStreamerStripe(const uint32_t agentId, uint32_t stripeId)
{
    const AgentDescAndDeps& agentAndDeps = m_Agents[agentId];
    assert(agentAndDeps.agent.type == AgentType::WGT_STREAMER);

    g_Logger.Verbose("Schedule WgtStreamerStripe { .agentId = %u, .stripeId = %d }", agentId, stripeId);

    const uint16_t tileSize = agentAndDeps.agent.wgt.tile.numSlots;
    InsertWriteDependencies(agentAndDeps.deps, agentId, stripeId, tileSize, m_DmaRdCommands);
    InsertReadDependencies(agentAndDeps.deps, agentId, stripeId, {}, m_DmaRdCommands);

    m_DmaRdCommands.push_back(Command{ CommandType::LoadWgtStripe, agentId, stripeId, 0 });
}

void Scheduler::ScheduleMceSchedulerStripe(const uint32_t agentId, uint32_t stripeId)
{
    const AgentDescAndDeps& agentAndDeps = m_Agents[agentId];
    assert(agentAndDeps.agent.type == AgentType::MCE_SCHEDULER);
    assert(agentAndDeps.deps.writeDependencies.size() == 0);

    g_Logger.Verbose("Schedule MceSchedulerStripe { .agentId = %u, .stripeId = %d }", agentId, stripeId);

    m_MceCommands.push_back(Command{ CommandType::ProgramMceStripe, agentId, stripeId, 0 });

    InsertReadDependencies(agentAndDeps.deps, agentId, stripeId, {}, m_MceCommands);

    m_MceCommands.push_back(Command{ CommandType::StartMceStripe, agentId, stripeId, 0 });
}

void Scheduler::SchedulePleLoaderStripe(const uint32_t agentId, uint32_t stripeId)
{
    const AgentDescAndDeps& agentAndDeps = m_Agents[agentId];
    assert(agentAndDeps.agent.type == AgentType::PLE_LOADER);

    g_Logger.Verbose("Schedule PleLoaderStripe { .agentId = %u, .stripeId = %d }", agentId, stripeId);

    constexpr uint16_t tileSize = 1;    // There isn't a tile for PleLoaderStripes
    InsertWriteDependencies(agentAndDeps.deps, agentId, stripeId, tileSize, m_DmaRdCommands);

    InsertReadDependencies(agentAndDeps.deps, agentId, stripeId, {}, m_DmaRdCommands);

    m_DmaRdCommands.push_back(Command{ CommandType::LoadPleCode, agentId, stripeId, 0 });
}

void Scheduler::SchedulePleSchedulerStripe(const uint32_t agentId, uint32_t stripeId)
{
    const AgentDescAndDeps& agentAndDeps = m_Agents[agentId];
    assert(agentAndDeps.agent.type == AgentType::PLE_SCHEDULER);
    assert(agentAndDeps.deps.readDependencies.size() > 0);

    g_Logger.Verbose("Schedule PleSchedulerStripe { .agentId = %u, .stripeId = %d }", agentId, stripeId);

    const uint16_t tileSize = agentAndDeps.agent.pleS.ofmTile.numSlots;
    InsertWriteDependencies(agentAndDeps.deps, agentId, stripeId, tileSize, m_PleCommands);

    // Read dependencies on the MCE are ignored, as the hardware manages MCE-PLE dependencies
    // automatically using the BUFFER_FREED signal and block counters. The read dependency
    // is still needed for the IsStripeNeeded logic to work, but we don't need to insert
    // the corresponding wait commands in the queue (and in fact, doing so would be wrong
    // because it would cause the MCE to deadlock waiting for the PLE, which wouldn't have
    // started yet).
    const auto agentTypeToIgnore = AgentType::MCE_SCHEDULER;
    InsertReadDependencies(agentAndDeps.deps, agentId, stripeId, agentTypeToIgnore, m_PleCommands);

    m_PleCommands.push_back(Command{ CommandType::StartPleStripe, agentId, stripeId, 0 });
}

void Scheduler::ScheduleOfmStreamerStripe(const uint32_t agentId, uint32_t stripeId)
{
    const AgentDescAndDeps& agentAndDeps = m_Agents[agentId];
    assert(agentAndDeps.agent.type == AgentType::OFM_STREAMER);
    assert(agentAndDeps.deps.writeDependencies.size() == 0);

    g_Logger.Verbose("Schedule OfmStreamerStripe { .agentId = %u, .stripeId = %d }", agentId, stripeId);

    InsertReadDependencies(agentAndDeps.deps, agentId, stripeId, {}, m_DmaWrCommands);

    const uint32_t numChunks = CalculateNumChunks(agentAndDeps.agent.ofm, stripeId);
    for (uint32_t i = 0; i < numChunks; ++i)
    {
        m_DmaWrCommands.push_back(Command{ CommandType::StoreOfmStripe, agentId, stripeId, 0 });
    }
}

Scheduler::Scheduler(const std::vector<AgentDescAndDeps>& agents)
    : m_Agents{ agents }
    , m_AgentProgress(agents.size(), 0)
    , m_BaseAgentId(0)
{}

const std::vector<Command>& Scheduler::GetDmaRdCommands() const
{
    return m_DmaRdCommands;
}

const std::vector<Command>& Scheduler::GetDmaWrCommands() const
{
    return m_DmaWrCommands;
}

const std::vector<Command>& Scheduler::GetMceCommands() const
{
    return m_MceCommands;
}

const std::vector<Command>& Scheduler::GetPleCommands() const
{
    return m_PleCommands;
}

bool Scheduler::Finished() const
{
    return m_BaseAgentId >= m_Agents.size();
}

void Scheduler::Schedule()
{
    using namespace command_stream::cascading;

    // Points to the agent that we will next attempt to schedule stripes from in the loop body
    uint32_t currentAgentId = 0;
    while (!Finished())
    {
        LogProgress();

        if (currentAgentId >= m_Agents.size())
        {
            // Reached the end of the command stream - go back to the last non-completed agent (on the next loop iteration).
            currentAgentId = m_BaseAgentId;
            continue;
        }

        const AgentDescAndDeps& agentAndDeps = m_Agents[currentAgentId];
        const uint32_t stripeId              = m_AgentProgress[currentAgentId];
        if (stripeId == agentAndDeps.agent.numStripesTotal)
        {
            // This agent is finished (fully scheduled), try the next (on the next loop iteration)
            // Advance the base agent ID appropriately
            if (m_BaseAgentId == currentAgentId)
            {
                ++m_BaseAgentId;
            }
            ++currentAgentId;
            continue;
        }

        // We don't bother looking ahead forever in the command stream to find agents that need scheduling,
        // as in practice we will only be scheduling a few agents in advance at any one time.
        // Therefore we stop once we get too far ahead.
        // A simple check is that we stop once we find the first agent which is waiting for stripes from other agents
        // to be scheduled (IsStripeReady), but this would stop too soon as it could then miss agents later in a strategy 0
        // cascade that need scheduling, which are blocking the earlier agents from proceeding (and then we would deadlock).
        // Therefore we keep looking ahead if this agent has already started, and only stop once we hit an agent that
        // hasn't started and can't start.
        if (!IsStripeReady(currentAgentId) && stripeId == 0)
        {
            // Go back to the last non-completed agent (on the next loop iteration).
            currentAgentId = m_BaseAgentId;
            continue;
        }

        // Schedule as many stripes as possible from the current agent.
        SpinAgent(currentAgentId);

        // Successfully scheduled everything that can be for this agent - move to the next (for the next loop iteration)
        // If this agent is now finished (fully scheduled), advance the base agent ID appropriately.
        // Note that we can't rely on the above code to advance the base agent ID, as it might never get called
        // if we never rewind, for example in a long sequence of single-stripe agents.
        if (m_BaseAgentId == currentAgentId &&
            m_AgentProgress[currentAgentId] == m_Agents[currentAgentId].agent.numStripesTotal)
        {
            ++m_BaseAgentId;
        }
        ++currentAgentId;
    }
}

void Scheduler::SpinAgent(uint32_t agentId)
{
    while (IsStripeReady(agentId) && IsStripeNeeded(agentId))
    {
        Schedule(agentId);
    }
}

void Scheduler::Schedule(const uint32_t agentId)
{
    using namespace command_stream::cascading;

    const AgentDescAndDeps& agentAndDeps = m_Agents[agentId];
    const uint32_t stripeId              = m_AgentProgress[agentId];

    switch (agentAndDeps.agent.type)
    {
        case AgentType::IFM_STREAMER:
        {
            ScheduleIfmStreamerStripe(agentId, stripeId);
            break;
        }
        case AgentType::WGT_STREAMER:
        {
            ScheduleWgtStreamerStripe(agentId, stripeId);
            break;
        }
        case AgentType::MCE_SCHEDULER:
        {
            ScheduleMceSchedulerStripe(agentId, stripeId);
            break;
        }
        case AgentType::PLE_LOADER:
        {
            SchedulePleLoaderStripe(agentId, stripeId);
            break;
        }
        case AgentType::PLE_SCHEDULER:
        {
            SchedulePleSchedulerStripe(agentId, stripeId);
            break;
        }
        case AgentType::OFM_STREAMER:
        {
            ScheduleOfmStreamerStripe(agentId, stripeId);
            break;
        }
        default:
        {
            assert(false && "Unknown agent type");
            break;
        }
    }

    m_AgentProgress[agentId] = stripeId + 1;
}

bool Scheduler::IsStripeReady(const uint32_t agentId, const uint32_t distanceThreshold) const
{
    for (const auto& dep : m_Agents[agentId].deps.readDependencies)
    {
        if (dep.relativeAgentId > distanceThreshold)
        {
            const uint32_t otherId = agentId - dep.relativeAgentId;

            const uint32_t stripeId      = m_AgentProgress[agentId];
            const uint32_t otherStripeId = m_AgentProgress[otherId];

            const bool isReady = static_cast<int>(otherStripeId) > GetLargestNeededStripeId(dep, stripeId);

            if (!isReady)
            {
                return false;
            }
        }
    }
    return true;
}

bool Scheduler::IsStripeNeeded(const uint32_t agentId) const
{
    const uint32_t stripeId = m_AgentProgress[agentId];

    if (stripeId >= m_Agents[agentId].agent.numStripesTotal)
    {
        // This agent has finished, so no more of its stripes can be needed.
        return false;
    }

    bool hasDependency = false;
    for (const auto& dep : m_Agents[agentId].deps.scheduleDependencies)
    {
        hasDependency = true;

        const uint32_t otherId       = agentId + dep.relativeAgentId;
        const uint32_t otherStripeId = m_AgentProgress[otherId];

        const bool isNeeded = IsStripeReady(otherId, dep.relativeAgentId) &&
                              (GetFirstReaderStripeId(dep, stripeId) <= static_cast<int>(otherStripeId));

        if (isNeeded)
        {
            return true;
        }
    }
    return !hasDependency;
}

void Scheduler::LogProgress() const
{
    // Unfortunately we can't rely on the compiler to optimise out all the logging code below,
    // because access to m_AgentProgress has side-effects.
    if (g_LogCompileTimeMaxSeverity >= ethosn::utils::log::Severity::Debug)
    {
        std::string logMsg;
        logMsg += "Scheduler: " + std::to_string(m_BaseAgentId) + "/" + std::to_string(m_Agents.size()) + " complete. ";
        if (!Finished())
        {
            logMsg += "In progress: ";
            for (uint32_t agentId = m_BaseAgentId; agentId < m_Agents.size() && agentId < m_BaseAgentId + 10; ++agentId)
            {
                logMsg += "[" + std::to_string(agentId) + "] = " + std::to_string(m_AgentProgress[agentId]) + "/" +
                          std::to_string(static_cast<int>(m_Agents[agentId].agent.numStripesTotal)) + ", ";
            }
        }
        g_Logger.Verbose("%s", logMsg.c_str());
    }
}

}    // namespace support_library
}    // namespace ethosn
