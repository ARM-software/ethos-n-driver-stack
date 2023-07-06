//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "Scheduler.hpp"

#include "../DebuggingContext.hpp"
#include "../Utils.hpp"
#include "DmaRegisters.hpp"

#include <ethosn_utils/Strings.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <vector>

using namespace ethosn::command_stream::cascading;
using CommandVariant = ethosn::command_stream::CommandVariant;

namespace ethosn
{
namespace support_library
{

namespace
{

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
    if (x < tileSize)
    {
        return -1;
    }
    return GetLastReaderStripeId(dep, x - tileSize);
}

void DumpDependency(std::ofstream& f, const Dependency& d, const char* type)
{
    f << "    <" << type << ">\n";
    f << "      <OTHER_AGENT_ID>" << d.otherAgentId << "</OTHER_AGENT_ID>\n";
    f << "      <OUTER_RATIO><OTHER>" << d.outerRatio.other << "</OTHER><SELF>" << d.outerRatio.self
      << "</SELF></OUTER_RATIO>\n";
    f << "      <INNER_RATIO><OTHER>" << d.innerRatio.other << "</OTHER><SELF>" << d.innerRatio.self
      << "</SELF></INNER_RATIO>\n";
    f << "      <BOUNDARY>" << static_cast<uint32_t>(d.boundary) << "</BOUNDARY>\n";
    f << "      <WRITES_TO_TILE>" << static_cast<uint32_t>(d.writesToTile) << "</WRITES_TO_TILE>\n";
    f << "      <USE_FOR_SCHEDULING>" << static_cast<uint32_t>(d.useForScheduling) << "</USE_FOR_SCHEDULING>\n";
    f << "      <USE_FOR_COMMAND_STREAM>" << static_cast<uint32_t>(d.useForCommandStream)
      << "</USE_FOR_COMMAND_STREAM>\n";
    f << "    </" << type << ">\n";
}

void DumpDependencies(std::ofstream& f, const std::vector<AgentDescAndDeps>& agents)
{
    f << "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n";
    f << "<STREAM><CASCADE>\n";
    f << "<NUM_AGENTS>" << agents.size() << "</NUM_AGENTS>\n";
    for (size_t a = 0; a < agents.size(); ++a)
    {
        f << "  <AGENT> <!-- Agent " << a << " -->\n";

        switch (agents[a].agent.type)
        {
            case AgentType::IFM_STREAMER:
                f << "    <IFM_STREAMER>\n";
                f << "      <TILE><NUM_SLOTS>" << agents[a].agent.ifm.fmData.tile.numSlots << "</NUM_SLOTS></TILE>\n";
                f << "    </IFM_STREAMER>\n";
                break;
            case AgentType::MCE_SCHEDULER:
                f << "    <MCE_SCHEDULER>\n";
                f << "    </MCE_SCHEDULER>\n";
                break;
            case AgentType::OFM_STREAMER:
                f << "    <OFM_STREAMER>\n";
                f << "    </OFM_STREAMER>\n";
                break;
            case AgentType::PLE_LOADER:
                f << "    <PLE_LOADER>\n";
                f << "    </PLE_LOADER>\n";
                break;
            case AgentType::PLE_SCHEDULER:
                f << "    <PLE_SCHEDULER>\n";
                f << "      <OFM_TILE><NUM_SLOTS>" << agents[a].agent.pleS.ofmTile.numSlots
                  << "</NUM_SLOTS></OFM_TILE>\n";
                f << "    </PLE_SCHEDULER>\n";
                break;
            case AgentType::WGT_STREAMER:
                f << "    <WGT_STREAMER>\n";
                f << "      <TILE><NUM_SLOTS>" << agents[a].agent.wgt.tile.numSlots << "</NUM_SLOTS></TILE>\n";
                f << "    </WGT_STREAMER>\n";
                break;
            default:
                assert(false);
                break;
        }

        f << "    <NUM_STRIPES_TOTAL>" << agents[a].agent.numStripesTotal << "</NUM_STRIPES_TOTAL>\n";
        for (Dependency d : agents[a].deps)
        {
            DumpDependency(f, d, "DEPENDENCY");
        }
        f << "  </AGENT>\n";
    }
    f << "</CASCADE></STREAM>\n";
}

}    // namespace

uint32_t Counters::Get(CounterName counterName) const
{
    switch (counterName)
    {
        case CounterName::DmaRd:
            return m_DmaRd;
        case CounterName::DmaWr:
            return m_DmaWr;
        case CounterName::Mceif:
            return m_Mceif;
        case CounterName::MceStripe:
            return m_MceStripe;
        case CounterName::PleCodeLoadedIntoPleSram:
            return m_PleCodeLoadedIntoPleSram;
        case CounterName::PleStripe:
            return m_PleStripe;
        default:
            throw std::runtime_error("Invalid counter name");
    }
}

void Counters::Set(CounterName counterName, uint32_t value)
{
    switch (counterName)
    {
        case CounterName::DmaRd:
            m_DmaRd = value;
            break;
        case CounterName::DmaWr:
            m_DmaWr = value;
            break;
        case CounterName::Mceif:
            m_Mceif = value;
            break;
        case CounterName::MceStripe:
            m_MceStripe = value;
            break;
        case CounterName::PleCodeLoadedIntoPleSram:
            m_PleCodeLoadedIntoPleSram = value;
            break;
        case CounterName::PleStripe:
            m_PleStripe = value;
            break;
        default:
            throw std::runtime_error("Invalid counter name");
    }
}

Counters Counters::Max(const Counters& a, const Counters& b)
{
    Counters result;
    result.m_DmaRd                    = std::max(a.m_DmaRd, b.m_DmaRd);
    result.m_DmaWr                    = std::max(a.m_DmaWr, b.m_DmaWr);
    result.m_Mceif                    = std::max(a.m_Mceif, b.m_Mceif);
    result.m_MceStripe                = std::max(a.m_MceStripe, b.m_MceStripe);
    result.m_PleCodeLoadedIntoPleSram = std::max(a.m_PleCodeLoadedIntoPleSram, b.m_PleCodeLoadedIntoPleSram);
    result.m_PleStripe                = std::max(a.m_PleStripe, b.m_PleStripe);
    return result;
}

Counters Scheduler::CounterImplications::Get(ethosn::command_stream::cascading::CounterName counterName,
                                             uint32_t value) const
{
    auto it = m_Map.find(std::make_pair(counterName, value));
    if (it == m_Map.end())
    {
        // Due to the way we use CounterImplications, we should never query something that hasn't
        // been added already.
        throw InternalErrorException("Unexpected use of CounterImplications");
    }
    return it->second;
}

void Scheduler::CounterImplications::Update(ethosn::command_stream::cascading::CounterName counterName,
                                            uint32_t value,
                                            Counters counters)
{
    // The counter that we are recording implications for has a clear guaranteed value
    // (using max() here just to avoid overwriting a larger value, although this shouldn't happen)
    counters.Set(counterName, std::max(counters.Get(counterName), value));

    auto it = m_Map.find(std::make_pair(counterName, value));
    if (it == m_Map.end())
    {
        m_Map[std::make_pair(counterName, value)] = counters;
    }
    else
    {
        // If we already have information, update it
        it->second = Counters::Max(it->second, counters);
    }
}

void Scheduler::CommandQueue::Push(const CommandVariant& c)
{
    if (c.type == CommandType::WaitForCounter)
    {
        // Before we add a WaitForCounter command, check if we can optimise it out.
        // This results in smaller command stream which will be faster for the firmware to process,
        // and should have no effect on the correctness of the command stream.
        const WaitForCounterCommand& waitCommand = c.waitForCounter;

        // Skip adding this command if we know that this counter value will already have been reached
        // by the time we get to this point in the queue
        uint32_t alreadyWaitedFor = m_LastCounterValuesWaitedFor.Get(waitCommand.counterName);
        if (alreadyWaitedFor >= waitCommand.counterValue)
        {
            return;
        }

        // Waiting for this counter value might then implicitly be waiting for other counters,
        // which we remember, so that we might be able to skip later WaitForCounters.
        m_LastCounterValuesWaitedFor = Counters::Max(
            m_LastCounterValuesWaitedFor, m_CounterImplications.Get(waitCommand.counterName, waitCommand.counterValue));

        // If the most recent command in the queue was also a WaitForCounter, we may be able to merge this
        // with the new one instead of adding another, if the new one implies waiting for the existing one too
        if (!m_Commands.empty() && m_Commands.back().type == CommandType::WaitForCounter)
        {
            WaitForCounterCommand& lastCmd = m_Commands.back().waitForCounter;
            if (m_CounterImplications.Get(waitCommand.counterName, waitCommand.counterValue).Get(lastCmd.counterName) >=
                lastCmd.counterValue)
            {
                lastCmd = waitCommand;
                return;
            }
        }
    }
    m_Commands.push_back(c);
}

const std::vector<CommandVariant>& Scheduler::CommandQueue::GetCommands() const
{
    return m_Commands;
}

Scheduler::CommandQueue& Scheduler::GetQueueForAgentType(AgentType agentType)
{
    switch (agentType)
    {
        case AgentType::IFM_STREAMER:
            return m_DmaRdCommands;
        case AgentType::WGT_STREAMER:
            return m_DmaRdCommands;
        case AgentType::MCE_SCHEDULER:
            return m_MceCommands;
        case AgentType::PLE_LOADER:
            return m_DmaRdCommands;
        case AgentType::PLE_SCHEDULER:
            return m_PleCommands;
        case AgentType::OFM_STREAMER:
            return m_DmaWrCommands;
        default:
            throw InternalErrorException("Unknown agent type");
    }
}

void Scheduler::PushWaitForCounterCommand(AgentType otherAgentType,
                                          uint32_t otherAgentId,
                                          uint32_t otherStripeId,
                                          CommandQueue& commands)
{
    WaitForCounterCommand waitCommand;
    waitCommand.type = CommandType::WaitForCounter;
    switch (otherAgentType)
    {
        case AgentType::IFM_STREAMER:
            waitCommand.counterName  = CounterName::DmaRd;
            waitCommand.counterValue = m_DmaRdCounters.at(std::make_pair(otherAgentId, otherStripeId));
            break;
        case AgentType::WGT_STREAMER:
            waitCommand.counterName  = CounterName::DmaRd;
            waitCommand.counterValue = m_DmaRdCounters.at(std::make_pair(otherAgentId, otherStripeId));
            break;
        case AgentType::MCE_SCHEDULER:
            waitCommand.counterName  = CounterName::MceStripe;
            waitCommand.counterValue = m_MceStripeCounters.at(std::make_pair(otherAgentId, otherStripeId));
            break;
        case AgentType::PLE_LOADER:
            waitCommand.counterName  = CounterName::DmaRd;
            waitCommand.counterValue = m_DmaRdCounters.at(std::make_pair(otherAgentId, otherStripeId));
            break;
        case AgentType::PLE_SCHEDULER:
            waitCommand.counterName  = CounterName::PleStripe;
            waitCommand.counterValue = m_PleStripeCounters.at(std::make_pair(otherAgentId, otherStripeId));
            break;
        case AgentType::OFM_STREAMER:
            waitCommand.counterName  = CounterName::DmaWr;
            waitCommand.counterValue = m_DmaWrCounters.at(std::make_pair(otherAgentId, otherStripeId));
            break;
        default:
            throw InternalErrorException("Unknown agent type");
    }

    commands.Push(CommandVariant(waitCommand));
}

void Scheduler::AddWaitForCounterCommands(const std::vector<Dependency>& dependencies,
                                          const uint32_t agentId,
                                          const uint32_t stripeId,
                                          const uint16_t tileSize,
                                          CommandQueue& commands)
{
    for (const auto& dep : dependencies)
    {
        // Not all dependencies are to be used for the command stream (some are just for scheduling)
        if (!dep.useForCommandStream)
        {
            continue;
        }

        const uint32_t otherAgentId = dep.otherAgentId;

        const int stripeToWaitFor = dep.writesToTile ? GetLastReaderOfEvictedStripeId(dep, stripeId, tileSize)
                                                     : GetLargestNeededStripeId(dep, stripeId);
        if (stripeToWaitFor >= m_Agents[otherAgentId].agent.numStripesTotal)
        {
            throw InternalErrorException(
                (std::string("Stripe ID out of range in AddWaitForCounterCommands: ") +
                 std::to_string(stripeToWaitFor) + "/" + ToString(m_Agents[otherAgentId].agent.numStripesTotal) +
                 " for agent " + std::to_string(agentId) + " depending on agent " + std::to_string(otherAgentId))
                    .c_str());
        }
        if (stripeToWaitFor >= 0)
        {
            bool sameQueue = &GetQueueForAgentType(m_Agents[otherAgentId].agent.type) == &commands;
            // Don't add dependencies on earlier stripes in the same queue as the order enforces this anyway.
            if (!sameQueue)
            {
                PushWaitForCounterCommand(m_Agents[otherAgentId].agent.type, otherAgentId, stripeToWaitFor, commands);
            }
            else if (sameQueue && m_AgentProgress[otherAgentId] < static_cast<uint32_t>(stripeToWaitFor))
            {
                // Dependencies to later stripes in the same queue are always invalid
                // and indicate there is an issue in the dependencies.
                throw InternalErrorException(
                    (std::string(
                         "Invalid scheduling detected due to dependencies on later stripes in the same queue: agent ") +
                     std::to_string(agentId) + " has dependency on agent " + std::to_string(otherAgentId))
                        .c_str());
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
    AddWaitForCounterCommands(agentAndDeps.deps, agentId, stripeId, tileSize, m_DmaRdCommands);

    const uint32_t numChunks = CalculateNumChunks(agentAndDeps.agent.ifm, stripeId);
    for (uint32_t chunkId = 0; chunkId < numChunks; ++chunkId)
    {
        DmaCommand cmd = GenerateDmaCommandForLoadIfmStripe(m_Agents[agentId].agent.ifm, agentId, stripeId, chunkId,
                                                            m_Capabilities, m_NextRdDmaCmdId);
        m_DmaRdCommands.Push(CommandVariant(cmd));
        m_NextRdDmaCmdId = (m_NextRdDmaCmdId + 1) % 4;
    }

    m_Counters.m_DmaRd += numChunks;
    m_DmaRdCounters[std::make_pair(agentId, stripeId)] = m_Counters.m_DmaRd;

    // Update the shared counter implications so that other queues know that when they wait on this
    // new counter value, they are also implicitly waiting on anything else that this queue has waited
    // on too
    m_CounterImplications.Update(CounterName::DmaRd, m_Counters.m_DmaRd,
                                 m_DmaRdCommands.GetLastCounterValuesWaitedFor());
}

void Scheduler::ScheduleWgtStreamerStripe(const uint32_t agentId, uint32_t stripeId)
{
    const AgentDescAndDeps& agentAndDeps = m_Agents[agentId];
    assert(agentAndDeps.agent.type == AgentType::WGT_STREAMER);

    g_Logger.Verbose("Schedule WgtStreamerStripe { .agentId = %u, .stripeId = %d }", agentId, stripeId);

    const uint16_t tileSize = agentAndDeps.agent.wgt.tile.numSlots;
    AddWaitForCounterCommands(agentAndDeps.deps, agentId, stripeId, tileSize, m_DmaRdCommands);

    DmaCommand cmd = GenerateDmaCommandForLoadWgtStripe(m_Agents[agentId].agent.wgt, agentId, stripeId, m_Capabilities,
                                                        m_NextRdDmaCmdId);
    m_DmaRdCommands.Push(CommandVariant(cmd));
    m_NextRdDmaCmdId = (m_NextRdDmaCmdId + 1) % 4;

    m_Counters.m_DmaRd += 1;
    m_DmaRdCounters[std::make_pair(agentId, stripeId)] = m_Counters.m_DmaRd;

    // Update the shared counter implications so that other queues know that when they wait on this
    // new counter value, they are also implicitly waiting on anything else that this queue has waited
    // on too
    m_CounterImplications.Update(CounterName::DmaRd, m_Counters.m_DmaRd,
                                 m_DmaRdCommands.GetLastCounterValuesWaitedFor());
}

void Scheduler::ScheduleMceSchedulerStripe(const uint32_t agentId, uint32_t stripeId)
{
    const AgentDescAndDeps& agentAndDeps = m_Agents[agentId];
    assert(agentAndDeps.agent.type == AgentType::MCE_SCHEDULER);

    g_Logger.Verbose("Schedule MceSchedulerStripe { .agentId = %u, .stripeId = %d }", agentId, stripeId);

    auto cmd = GenerateProgramMceStripeCommand(agentAndDeps.agent.mce, agentId, stripeId, m_Capabilities);
    m_MceCommands.Push(CommandVariant(cmd));

    AddWaitForCounterCommands(agentAndDeps.deps, agentId, stripeId, 0, m_MceCommands);

    // Reconfigure the MCEIF if necessary. This will be if this is the first MCE stripe in the whole inference,
    // or if the MCEIF configuration was changed due to a different PLE kernel being loaded.
    bool pleKernelChanged =
        m_Counters.m_PleStripe > 0 &&
        m_Agents[m_PleCommands.GetCommands().back().startPleStripe.agentId].agent.pleS.pleKernelId !=
            m_Agents[agentId].agent.mce.pleKernelId &&
        // Note this extra condition is needed because for strategy 1 cascading, we schedule all Mce
        // stripes before the Ple, and we don't want to reconfigure MCEIF for every stripe.
        m_MceifConfiguration != m_Agents[agentId].agent.mce.pleKernelId;

    if (pleKernelChanged || m_Counters.m_Mceif == 0)
    {
        // If the PLE kernel has changed then the MCEIF will need reconfiguring, but we first need to wait
        // for the PLE to "catch up".
        // Otherwise the following PLE command could reset the MCEIF after we've set it
        // (e.g. if it's a standalone PLE) but before we've finished using it.
        if (pleKernelChanged)
        {
            WaitForCounterCommand waitCommand;
            waitCommand.type         = CommandType::WaitForCounter;
            waitCommand.counterName  = CounterName::PleStripe;
            waitCommand.counterValue = m_Counters.m_PleStripe;
            m_MceCommands.Push(CommandVariant(waitCommand));
        }

        ConfigMceifCommand mceifCommand;
        mceifCommand.type    = CommandType::ConfigMceif;
        mceifCommand.agentId = agentId;
        m_MceCommands.Push(CommandVariant(mceifCommand));

        m_Counters.m_Mceif += 1;
        m_MceifConfiguration = m_Agents[agentId].agent.mce.pleKernelId;

        // Update the shared counter implications so that other queues know that when they wait on this
        // new counter value, they are also implicitly waiting on other counters too
        m_CounterImplications.Update(CounterName::Mceif, m_Counters.m_Mceif,
                                     m_MceCommands.GetLastCounterValuesWaitedFor());
    }

    auto cmd2 = GenerateStartMceStripeCommand(agentAndDeps.agent.mce, agentId, stripeId, m_Capabilities);
    m_MceCommands.Push(CommandVariant(cmd2));

    m_Counters.m_MceStripe += 1;
    m_MceStripeCounters[std::make_pair(agentId, stripeId)] = m_Counters.m_MceStripe;

    // Update the shared counter implications so that other queues know that when they wait on this
    // new counter value, they are also implicitly waiting on anything else that this queue has waited
    // on too
    m_CounterImplications.Update(CounterName::MceStripe, m_Counters.m_MceStripe,
                                 m_MceCommands.GetLastCounterValuesWaitedFor());
}

void Scheduler::SchedulePleLoaderStripe(const uint32_t agentId, uint32_t stripeId)
{
    const AgentDescAndDeps& agentAndDeps = m_Agents[agentId];
    assert(agentAndDeps.agent.type == AgentType::PLE_LOADER);

    g_Logger.Verbose("Schedule PleLoaderStripe { .agentId = %u, .stripeId = %d }", agentId, stripeId);

    constexpr uint16_t tileSize = 1;    // There isn't a tile for PleLoaderStripes
    AddWaitForCounterCommands(agentAndDeps.deps, agentId, stripeId, tileSize, m_DmaRdCommands);

    auto cmd =
        GenerateDmaCommandForLoadPleCode(m_Agents[agentId].agent.pleL, agentId, m_Capabilities, m_NextRdDmaCmdId);
    m_DmaRdCommands.Push(CommandVariant(cmd));
    m_NextRdDmaCmdId = (m_NextRdDmaCmdId + 1) % 4;

    m_Counters.m_DmaRd += 1;
    m_DmaRdCounters[std::make_pair(agentId, stripeId)] = m_Counters.m_DmaRd;

    // Update the shared counter implications so that other queues know that when they wait on this
    // new counter value, they are also implicitly waiting on anything else that this queue has waited
    // on too
    m_CounterImplications.Update(CounterName::DmaRd, m_Counters.m_DmaRd,
                                 m_DmaRdCommands.GetLastCounterValuesWaitedFor());
}

void Scheduler::SchedulePleSchedulerStripe(const uint32_t agentId, uint32_t stripeId)
{
    const AgentDescAndDeps& agentAndDeps = m_Agents[agentId];
    assert(agentAndDeps.agent.type == AgentType::PLE_SCHEDULER);

    g_Logger.Verbose("Schedule PleSchedulerStripe { .agentId = %u, .stripeId = %d }", agentId, stripeId);

    const uint16_t tileSize = agentAndDeps.agent.pleS.ofmTile.numSlots;
    AddWaitForCounterCommands(agentAndDeps.deps, agentId, stripeId, tileSize, m_PleCommands);

    // Load new PLE code if necessary
    if (m_LastLoadedPleKernel != agentAndDeps.agent.pleS.pleKernelId)
    {
        LoadPleCodeIntoPleSramCommand loadCommand;
        loadCommand.type    = CommandType::LoadPleCodeIntoPleSram;
        loadCommand.agentId = agentId;
        m_PleCommands.Push(CommandVariant(loadCommand));

        m_LastLoadedPleKernel = agentAndDeps.agent.pleS.pleKernelId;
        m_Counters.m_PleCodeLoadedIntoPleSram += 1;

        // Update the shared counter implications so that other queues know that when they wait on this
        // new counter value, they are also implicitly waiting on other counters too
        m_CounterImplications.Update(CounterName::PleCodeLoadedIntoPleSram, m_Counters.m_PleCodeLoadedIntoPleSram,
                                     m_PleCommands.GetLastCounterValuesWaitedFor());

        WaitForCounterCommand waitCommand;
        waitCommand.type         = CommandType::WaitForCounter;
        waitCommand.counterName  = CounterName::PleCodeLoadedIntoPleSram;
        waitCommand.counterValue = m_Counters.m_PleCodeLoadedIntoPleSram;
        m_PleCommands.Push(CommandVariant(waitCommand));

        // Loading a new kernel invalidates the MCEIF configuration, as the PLE will be reset and therefore
        // forget its position in the PLE input SRAM buffer ring buffer. Clearing this will force the
        // MCE stripe to reconfigure it appropriately.
        m_MceifConfiguration = PleKernelId::NOT_FOUND;
    }

    // Wait for MCEIF to have been configured if necessary
    // If this PLE kernel takes input from the MCE, we need to wait until the MCEIF has been
    // reconfigured for this kernel. This is handled by the Mce command queue and so we add a WaitForCounter
    // on the MCEIF counter, based on the most recent value.
    const bool isSram = agentAndDeps.agent.pleS.inputMode == PleInputMode::SRAM_ONE_INPUT ||
                        agentAndDeps.agent.pleS.inputMode == PleInputMode::SRAM_TWO_INPUTS;
    if (!isSram)
    {
        if (m_MceifConfiguration == PleKernelId::NOT_FOUND)
        {
            WaitForCounterCommand waitCommand;
            waitCommand.type         = CommandType::WaitForCounter;
            waitCommand.counterName  = CounterName::Mceif;
            waitCommand.counterValue = m_Counters.m_Mceif;
            m_PleCommands.Push(CommandVariant(waitCommand));
        }
    }

    auto cmd = GenerateStartPleStripeCommand(m_Agents[agentId].agent.pleS, agentId, stripeId);
    m_PleCommands.Push(CommandVariant(cmd));

    m_Counters.m_PleStripe += 1;
    m_PleStripeCounters[std::make_pair(agentId, stripeId)] = m_Counters.m_PleStripe;

    // Update the shared counter implications so that other queues know that when they wait on this
    // new counter value, they are also implicitly waiting on anything else that this queue has waited
    // on too
    m_CounterImplications.Update(CounterName::PleStripe, m_Counters.m_PleStripe,
                                 m_PleCommands.GetLastCounterValuesWaitedFor());
}

void Scheduler::ScheduleOfmStreamerStripe(const uint32_t agentId, uint32_t stripeId)
{
    const AgentDescAndDeps& agentAndDeps = m_Agents[agentId];
    assert(agentAndDeps.agent.type == AgentType::OFM_STREAMER);

    g_Logger.Verbose("Schedule OfmStreamerStripe { .agentId = %u, .stripeId = %d }", agentId, stripeId);

    AddWaitForCounterCommands(agentAndDeps.deps, agentId, stripeId, {}, m_DmaWrCommands);

    const uint32_t numChunks = CalculateNumChunks(agentAndDeps.agent.ofm, stripeId);
    for (uint32_t chunkId = 0; chunkId < numChunks; ++chunkId)
    {
        auto cmd = (GenerateDmaCommandForStoreOfmStripe(m_Agents[agentId].agent.ofm, agentId, stripeId, chunkId,
                                                        m_Capabilities, m_NextWrDmaCmdId));
        m_DmaWrCommands.Push(CommandVariant(cmd));
        m_NextWrDmaCmdId = 4 + ((m_NextWrDmaCmdId + 1) % 4);
    }

    m_Counters.m_DmaWr += numChunks;
    m_DmaWrCounters[std::make_pair(agentId, stripeId)] = m_Counters.m_DmaWr;

    // Update the shared counter implications so that other queues know that when they wait on this
    // new counter value, they are also implicitly waiting on anything else that this queue has waited
    // on too
    m_CounterImplications.Update(CounterName::DmaWr, m_Counters.m_DmaWr,
                                 m_DmaWrCommands.GetLastCounterValuesWaitedFor());
}

Scheduler::Scheduler(const std::vector<AgentDescAndDeps>& agents,
                     const HardwareCapabilities& capabilities,
                     const DebuggingContext& debuggingContext)
    : m_DebuggingContext(debuggingContext)
    , m_Agents{ agents }
    , m_AgentProgress(agents.size(), 0)
    , m_DmaRdCommands(m_CounterImplications)
    , m_DmaWrCommands(m_CounterImplications)
    , m_MceCommands(m_CounterImplications)
    , m_PleCommands(m_CounterImplications)
    , m_Capabilities(capabilities)
{}

const std::vector<CommandVariant>& Scheduler::GetDmaRdCommands() const
{
    return m_DmaRdCommands.GetCommands();
}

const std::vector<CommandVariant>& Scheduler::GetDmaWrCommands() const
{
    return m_DmaWrCommands.GetCommands();
}

const std::vector<CommandVariant>& Scheduler::GetMceCommands() const
{
    return m_MceCommands.GetCommands();
}

const std::vector<CommandVariant>& Scheduler::GetPleCommands() const
{
    return m_PleCommands.GetCommands();
}

void Scheduler::ScheduleOneStripe(const uint32_t agentId)
{
    using namespace command_stream::cascading;

    const AgentDescAndDeps& agentAndDeps = m_Agents[agentId];
    const uint32_t stripeId              = m_AgentProgress[agentId];

    if (stripeId >= m_Agents[agentId].agent.numStripesTotal)
    {
        throw InternalErrorException(
            (std::string("Stripe ID out of range in ScheduleOneStripe: ") + std::to_string(stripeId) + "/" +
             ToString(m_Agents[agentId].agent.numStripesTotal) + " for agent " + std::to_string(agentId))
                .c_str());
    }

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

void Scheduler::Schedule()
{
    using namespace command_stream::cascading;

    // For debugging the scheduling dependencies, dump out some of the intermediate command stream representation
    if (m_DebuggingContext.m_DebugInfo.m_DumpDebugFiles >= CompilationOptions::DebugLevel::Medium)
    {
        std::ofstream f(m_DebuggingContext.GetAbsolutePathOutputFileName("ScheduleDependencies.xml"));
        DumpDependencies(f, m_Agents);
    }

    struct Context
    {
        uint32_t agentId;
    };

    auto EvaluateDependencies = [&](uint32_t agentId, std::vector<Context>& stack) {
        for (const auto& dep : m_Agents[agentId].deps)
        {
            // Not all dependencies are used for scheduling (some are just for the command stream)
            if (!dep.useForScheduling)
            {
                continue;
            }

            uint16_t tileSize;
            switch (m_Agents[agentId].agent.type)
            {
                case AgentType::IFM_STREAMER:
                    tileSize = m_Agents[agentId].agent.ifm.fmData.tile.numSlots;
                    break;
                case AgentType::WGT_STREAMER:
                    tileSize = m_Agents[agentId].agent.wgt.tile.numSlots;
                    break;
                case AgentType::PLE_SCHEDULER:
                    tileSize = m_Agents[agentId].agent.pleS.ofmTile.numSlots;
                    break;
                default:
                    tileSize = 1;
                    break;
            }

            const int largestNeededStripeId =
                dep.writesToTile ? GetLastReaderOfEvictedStripeId(dep, m_AgentProgress[agentId], tileSize)
                                 : GetLargestNeededStripeId(dep, m_AgentProgress[agentId]);

            if (static_cast<int>(m_AgentProgress[dep.otherAgentId]) <= largestNeededStripeId)
            {
                stack.push_back(Context{ dep.otherAgentId });
                return true;
            }
        }
        return false;
    };

    for (uint32_t a = 0; a < m_Agents.size(); ++a)
    {
        if (m_Agents[a].agent.type == AgentType::OFM_STREAMER)
        {
            // Note that we use a while loop and check m_AgentProgress, as we may end up scheduling
            // stripes further ahead too
            while (m_AgentProgress[a] < m_Agents[a].agent.numStripesTotal)
            {
                // Store the stripes we want to schedule on a stack
                std::vector<Context> stack;

                Context context{ a };
                stack.push_back(context);

                while (!stack.empty())
                {
                    if (stack.size() > m_Agents.size())
                    {
                        throw InternalErrorException(
                            ("Dependency cycle detected with agent IDs: " +
                             ethosn::utils::Join(" -> ", stack, [](Context x) { return std::to_string(x.agentId); }))
                                .c_str());
                    }

                    Context current = stack.back();

                    bool hasDependencies = EvaluateDependencies(current.agentId, stack);
                    if (hasDependencies)
                    {
                        continue;
                    }

                    ScheduleOneStripe(current.agentId);
                    stack.pop_back();
                }
            }
        }
    }

    // Verify that all stripes from all agents have been scheduled.
    // If not, then some dependencies are probably wrong
    for (size_t a = 0; a < m_Agents.size(); ++a)
    {
        if (m_AgentProgress[a] != m_Agents[a].agent.numStripesTotal)
        {
            throw InternalErrorException(
                (std::string("Agent ") + std::to_string(a) + " has not had all its stripes scheduled: " +
                 std::to_string(m_AgentProgress[a]) + " / " + ToString(m_Agents[a].agent.numStripesTotal))
                    .c_str());
        }
    }
}

}    // namespace support_library
}    // namespace ethosn
