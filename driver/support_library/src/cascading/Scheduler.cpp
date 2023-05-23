//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "Scheduler.hpp"

#include "../DebuggingContext.hpp"
#include "../Utils.hpp"
#include "DmaRegisters.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <stack>

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
    assert(x >= tileSize);
    return GetLastReaderStripeId(dep, x - tileSize);
}

void DumpDependency(std::ofstream& f, const Dependency& d, const char* type)
{
    f << "    <" << type << ">\n";
    f << "      <RELATIVE_AGENT_ID>" << static_cast<uint32_t>(d.relativeAgentId) << "</RELATIVE_AGENT_ID>\n";
    f << "      <OUTER_RATIO><OTHER>" << d.outerRatio.other << "</OTHER><SELF>" << d.outerRatio.self
      << "</SELF></OUTER_RATIO>\n";
    f << "      <INNER_RATIO><OTHER>" << d.innerRatio.other << "</OTHER><SELF>" << d.innerRatio.self
      << "</SELF></INNER_RATIO>\n";
    f << "      <BOUNDARY>" << static_cast<uint32_t>(d.boundary) << "</BOUNDARY>\n";
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
        for (Dependency d : agents[a].deps.readDependencies)
        {
            DumpDependency(f, d, "READ_DEPENDENCY");
        }
        for (Dependency d : agents[a].deps.writeDependencies)
        {
            DumpDependency(f, d, "WRITE_DEPENDENCY");
        }
        f << "  </AGENT>\n";
    }
    f << "</CASCADE></STREAM>\n";
}

}    // namespace

void Scheduler::CommandQueue::Push(const CommandVariant& c)
{
    if (c.type == CommandType::WaitForCounter)
    {
        const WaitForCounterCommand& waitCommand = c.waitForCounter;
        // Skip adding this command if we've already waited for this counter (or a later one)
        auto lastStripeWaitedForIt = m_LastValueWaitedForCounterName.find(waitCommand.counterName);
        if (lastStripeWaitedForIt != m_LastValueWaitedForCounterName.end() &&
            lastStripeWaitedForIt->second >= waitCommand.counterValue)
        {
            return;
        }
        // Remember that we've now waited for this counter value, so that future waits might be skippable.
        m_LastValueWaitedForCounterName[waitCommand.counterName] = waitCommand.counterValue;
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

void Scheduler::InsertWriteDependencies(const AgentDependencyInfo& agent,
                                        const uint32_t agentId,
                                        const uint32_t stripeId,
                                        const uint16_t tileSize,
                                        CommandQueue& commands)
{
    if (stripeId < tileSize)
    {
        return;
    }
    for (const auto& writeDependency : agent.writeDependencies)
    {
        const uint32_t otherAgentId = agentId + writeDependency.relativeAgentId;

        const int stripeToWaitFor = GetLastReaderOfEvictedStripeId(writeDependency, stripeId, tileSize);
        if (stripeToWaitFor >= m_Agents[otherAgentId].agent.numStripesTotal)
        {
            throw InternalErrorException(
                (std::string("Stripe ID out of range in InsertWriteDependencies: ") + std::to_string(stripeToWaitFor) +
                 "/" + ToString(m_Agents[otherAgentId].agent.numStripesTotal) + " for agent " +
                 std::to_string(agentId) + " depending on agent " + std::to_string(otherAgentId))
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
                     std::to_string(agentId) + " has write dependency on agent " + std::to_string(otherAgentId))
                        .c_str());
            }
        }
    }
}

void Scheduler::InsertReadDependencies(const AgentDependencyInfo& agent,
                                       const uint32_t agentId,
                                       const uint32_t stripeId,
                                       const utils::Optional<AgentType> agentTypeToIgnore,
                                       CommandQueue& commands)
{
    for (const auto& readDependency : agent.readDependencies)
    {
        const uint32_t otherAgentId    = agentId - readDependency.relativeAgentId;
        const AgentType otherAgentType = m_Agents[otherAgentId].agent.type;

        if (utils::Optional<AgentType>{ otherAgentType } != agentTypeToIgnore)
        {
            const int stripeToWaitFor = GetLargestNeededStripeId(readDependency, stripeId);
            if (stripeToWaitFor >= m_Agents[otherAgentId].agent.numStripesTotal)
            {
                throw InternalErrorException(
                    (std::string("Stripe ID out of range in InsertReadDependencies: ") +
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
                    PushWaitForCounterCommand(m_Agents[otherAgentId].agent.type, otherAgentId, stripeToWaitFor,
                                              commands);
                }
                else if (sameQueue && m_AgentProgress[otherAgentId] < static_cast<uint32_t>(stripeToWaitFor))
                {
                    // Dependencies to later stripes in the same queue are always invalid
                    // and indicate there is an issue in the dependencies.
                    throw InternalErrorException((std::string("Invalid scheduling detected due to dependencies on "
                                                              "later stripes in the same queue: agent ") +
                                                  std::to_string(agentId) + " has read dependency on agent " +
                                                  std::to_string(otherAgentId))
                                                     .c_str());
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
    for (uint32_t chunkId = 0; chunkId < numChunks; ++chunkId)
    {
        DmaCommand cmd = GenerateDmaCommandForLoadIfmStripe(m_Agents[agentId].agent.ifm, agentId, stripeId, chunkId,
                                                            m_Capabilities, m_NextRdDmaCmdId);
        m_DmaRdCommands.Push(CommandVariant(cmd));
        m_NextRdDmaCmdId = (m_NextRdDmaCmdId + 1) % 4;
    }

    m_DmaRdCounter += numChunks;
    m_DmaRdCounters[std::make_pair(agentId, stripeId)] = m_DmaRdCounter;
}

void Scheduler::ScheduleWgtStreamerStripe(const uint32_t agentId, uint32_t stripeId)
{
    const AgentDescAndDeps& agentAndDeps = m_Agents[agentId];
    assert(agentAndDeps.agent.type == AgentType::WGT_STREAMER);

    g_Logger.Verbose("Schedule WgtStreamerStripe { .agentId = %u, .stripeId = %d }", agentId, stripeId);

    const uint16_t tileSize = agentAndDeps.agent.wgt.tile.numSlots;
    InsertWriteDependencies(agentAndDeps.deps, agentId, stripeId, tileSize, m_DmaRdCommands);
    InsertReadDependencies(agentAndDeps.deps, agentId, stripeId, {}, m_DmaRdCommands);

    DmaCommand cmd = GenerateDmaCommandForLoadWgtStripe(m_Agents[agentId].agent.wgt, agentId, stripeId, m_Capabilities,
                                                        m_NextRdDmaCmdId);
    m_DmaRdCommands.Push(CommandVariant(cmd));
    m_NextRdDmaCmdId = (m_NextRdDmaCmdId + 1) % 4;

    m_DmaRdCounter += 1;
    m_DmaRdCounters[std::make_pair(agentId, stripeId)] = m_DmaRdCounter;
}

void Scheduler::ScheduleMceSchedulerStripe(const uint32_t agentId, uint32_t stripeId)
{
    const AgentDescAndDeps& agentAndDeps = m_Agents[agentId];
    assert(agentAndDeps.agent.type == AgentType::MCE_SCHEDULER);
    assert(agentAndDeps.deps.writeDependencies.size() == 0);

    g_Logger.Verbose("Schedule MceSchedulerStripe { .agentId = %u, .stripeId = %d }", agentId, stripeId);

    auto cmd = GenerateProgramMceStripeCommand(agentAndDeps.agent.mce, agentId, stripeId, m_Capabilities);
    m_MceCommands.Push(CommandVariant(cmd));

    InsertReadDependencies(agentAndDeps.deps, agentId, stripeId, {}, m_MceCommands);

    // Reconfigure the MCEIF if necessary. This will be if this is the first MCE stripe in the whole inference,
    // or if the MCEIF configuration was changed due to a different PLE kernel being loaded.
    bool pleKernelChanged =
        m_PleStripeCounter > 0 &&
        m_Agents[m_PleCommands.GetCommands().back().startPleStripe.agentId].agent.pleS.pleKernelId !=
            m_Agents[agentId].agent.mce.pleKernelId &&
        // Note this extra condition is needed because for strategy 1 cascading, we schedule all Mce
        // stripes before the Ple, and we don't want to reconfigure MCEIF for every stripe.
        m_MceifConfiguration != m_Agents[agentId].agent.mce.pleKernelId;

    if (pleKernelChanged || m_MceifCounter == 0)
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
            waitCommand.counterValue = m_PleStripeCounter;
            m_MceCommands.Push(CommandVariant(waitCommand));
        }

        ConfigMceifCommand mceifCommand;
        mceifCommand.type    = CommandType::ConfigMceif;
        mceifCommand.agentId = agentId;
        m_MceCommands.Push(CommandVariant(mceifCommand));

        m_MceifCounter += 1;
        m_MceifConfiguration = m_Agents[agentId].agent.mce.pleKernelId;
    }

    auto cmd2 = GenerateStartMceStripeCommand(agentAndDeps.agent.mce, agentId, stripeId, m_Capabilities);
    m_MceCommands.Push(CommandVariant(cmd2));

    m_MceStripeCounter += 1;
    m_MceStripeCounters[std::make_pair(agentId, stripeId)] = m_MceStripeCounter;
}

void Scheduler::SchedulePleLoaderStripe(const uint32_t agentId, uint32_t stripeId)
{
    const AgentDescAndDeps& agentAndDeps = m_Agents[agentId];
    assert(agentAndDeps.agent.type == AgentType::PLE_LOADER);

    g_Logger.Verbose("Schedule PleLoaderStripe { .agentId = %u, .stripeId = %d }", agentId, stripeId);

    constexpr uint16_t tileSize = 1;    // There isn't a tile for PleLoaderStripes
    InsertWriteDependencies(agentAndDeps.deps, agentId, stripeId, tileSize, m_DmaRdCommands);

    InsertReadDependencies(agentAndDeps.deps, agentId, stripeId, {}, m_DmaRdCommands);

    auto cmd =
        GenerateDmaCommandForLoadPleCode(m_Agents[agentId].agent.pleL, agentId, m_Capabilities, m_NextRdDmaCmdId);
    m_DmaRdCommands.Push(CommandVariant(cmd));
    m_NextRdDmaCmdId = (m_NextRdDmaCmdId + 1) % 4;

    m_DmaRdCounter += 1;
    m_DmaRdCounters[std::make_pair(agentId, stripeId)] = m_DmaRdCounter;
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

    // Load new PLE code if necessary
    if (m_LastLoadedPleKernel != agentAndDeps.agent.pleS.pleKernelId)
    {
        LoadPleCodeIntoPleSramCommand loadCommand;
        loadCommand.type    = CommandType::LoadPleCodeIntoPleSram;
        loadCommand.agentId = agentId;
        m_PleCommands.Push(CommandVariant(loadCommand));

        m_LastLoadedPleKernel = agentAndDeps.agent.pleS.pleKernelId;
        m_PleCodeLoadedIntoPleSramCounter += 1;

        WaitForCounterCommand waitCommand;
        waitCommand.type         = CommandType::WaitForCounter;
        waitCommand.counterName  = CounterName::PleCodeLoadedIntoPleSram;
        waitCommand.counterValue = m_PleCodeLoadedIntoPleSramCounter;
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
            waitCommand.counterValue = m_MceifCounter;
            m_PleCommands.Push(CommandVariant(waitCommand));
        }
    }

    auto cmd = GenerateStartPleStripeCommand(m_Agents[agentId].agent.pleS, agentId, stripeId);
    m_PleCommands.Push(CommandVariant(cmd));

    m_PleStripeCounter += 1;
    m_PleStripeCounters[std::make_pair(agentId, stripeId)] = m_PleStripeCounter;
}

void Scheduler::ScheduleOfmStreamerStripe(const uint32_t agentId, uint32_t stripeId)
{
    const AgentDescAndDeps& agentAndDeps = m_Agents[agentId];
    assert(agentAndDeps.agent.type == AgentType::OFM_STREAMER);
    assert(agentAndDeps.deps.writeDependencies.size() == 0);

    g_Logger.Verbose("Schedule OfmStreamerStripe { .agentId = %u, .stripeId = %d }", agentId, stripeId);

    InsertReadDependencies(agentAndDeps.deps, agentId, stripeId, {}, m_DmaWrCommands);

    const uint32_t numChunks = CalculateNumChunks(agentAndDeps.agent.ofm, stripeId);
    for (uint32_t chunkId = 0; chunkId < numChunks; ++chunkId)
    {
        auto cmd = (GenerateDmaCommandForStoreOfmStripe(m_Agents[agentId].agent.ofm, agentId, stripeId, chunkId,
                                                        m_Capabilities, m_NextWrDmaCmdId));
        m_DmaWrCommands.Push(CommandVariant(cmd));
        m_NextWrDmaCmdId = 4 + ((m_NextWrDmaCmdId + 1) % 4);
    }

    m_DmaWrCounter += numChunks;
    m_DmaWrCounters[std::make_pair(agentId, stripeId)] = m_DmaWrCounter;
}

Scheduler::Scheduler(const std::vector<AgentDescAndDeps>& agents,
                     const HardwareCapabilities& capabilities,
                     const DebuggingContext& debuggingContext)
    : m_DebuggingContext(debuggingContext)
    , m_Agents{ agents }
    , m_AgentProgress(agents.size(), 0)
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
        int agentId;
    };

    auto EvaluateReadDependencies = [&](int agentId, std::stack<Context>& stack) {
        for (const auto& dep : m_Agents[agentId].deps.readDependencies)
        {
            const int otherAgentId = agentId - dep.relativeAgentId;

            const int largestNeededStripeId = GetLargestNeededStripeId(dep, m_AgentProgress[agentId]);
            if (static_cast<int>(m_AgentProgress[otherAgentId]) <= largestNeededStripeId)
            {
                stack.push(Context{ otherAgentId });
                return true;
            }
        }
        return false;
    };

    auto EvaluateWriteDependencies = [&](int agentId, std::stack<Context>& stack) {
        for (const auto& dep : m_Agents[agentId].deps.writeDependencies)
        {
            const int otherAgentId = agentId + dep.relativeAgentId;

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

            if (m_AgentProgress[agentId] < tileSize)
            {
                continue;
            }

            const int stripeToWaitFor = GetLastReaderOfEvictedStripeId(dep, m_AgentProgress[agentId], tileSize);
            if (static_cast<int>(m_AgentProgress[otherAgentId]) <= stripeToWaitFor)
            {
                stack.push(Context{ otherAgentId });
                return true;
            }
        }
        return false;
    };

    for (int a = 0; a < static_cast<int>(m_Agents.size()); ++a)
    {
        if (m_Agents[a].agent.type == AgentType::OFM_STREAMER)
        {
            // Note that we use a while loop and check m_AgentProgress, as we may end up scheduling
            // stripes further ahead too
            while (static_cast<int>(m_AgentProgress[a]) < m_Agents[a].agent.numStripesTotal)
            {
                // Store the stripes we want to schedule on a stack
                std::stack<Context> stack;

                Context context{ a };
                stack.push(context);

                while (!stack.empty())
                {
                    if (stack.size() > m_Agents.size())
                    {
                        throw InternalErrorException("Dependency cycle detected");
                    }

                    Context current = stack.top();

                    bool hasReadDependencies = EvaluateReadDependencies(current.agentId, stack);
                    if (hasReadDependencies)
                    {
                        continue;
                    }
                    // Also need to look at write dependencies, speficically for the case with two ofm streamers
                    // at the end of a section, so that we schedule them interleaved ratehr than all of one then all of the other
                    bool hasWriteDependencies = EvaluateWriteDependencies(current.agentId, stack);
                    if (hasWriteDependencies)
                    {
                        continue;
                    }
                    ScheduleOneStripe(current.agentId);
                    stack.pop();
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
