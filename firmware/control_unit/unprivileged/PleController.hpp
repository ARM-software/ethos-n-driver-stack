//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "../Profiling.hpp"
#include "CommandList.hpp"
#include "HwAbstraction.hpp"
#include <common/Log.hpp>

#include <ethosn_command_stream/PleKernelIds.hpp>

namespace ethosn::control_unit
{

class PleController
{
public:
    PleController(const Command* commandsBegin, uint32_t numCommands, const char* endOfCmdStream);

    template <typename Ctrl>
    bool HandleCommands(Ctrl& ctrl);

    template <typename Ctrl>
    void UpdateProgress(Ctrl& ctrl, bool pleStripeDone, bool pleCodeLoadedIntoPleSram);

    template <typename Ctrl>
    LoggingString GetStateString(Ctrl& ctrl, uint32_t origNumCommands) const;

    bool IsDone() const;

private:
    static constexpr uint32_t m_PrefetchSize = 4 * g_CacheLineSize;    // Found experimentally

    template <typename Ctrl>
    bool HandleCommand(Ctrl& ctrl, const Command& cmd);

    using PleKernelId = ethosn::command_stream::PleKernelId;

    CommandList m_CmdQueue;
    profiling::ProfilingOnly<uint8_t> m_InProgressProfilingEntryId;
};

inline PleController::PleController(const Command* commandsBegin, uint32_t numCommands, const char* endOfCmdStream)
    : m_CmdQueue(commandsBegin, numCommands, m_PrefetchSize, endOfCmdStream)
{}

template <typename Ctrl>
bool PleController::HandleCommands(Ctrl& ctrl)
{
    bool madeProgress = false;
    while (!m_CmdQueue.IsEmpty())
    {
        if (!HandleCommand(ctrl, m_CmdQueue.GetFirst()))
        {
            break;
        }
        m_CmdQueue.RemoveFirst();
        m_CmdQueue.Prefetch();
        madeProgress = true;
    }

    return madeProgress;
}

template <typename Ctrl>
inline void PleController::UpdateProgress(Ctrl& ctrl, bool pleStripeDone, bool pleCodeLoadedIntoPleSram)
{
    if (pleStripeDone)
    {
        ctrl.hwAbstraction.GetLogger().Debug("Ple stripe completed");
        ctrl.pleStripeCounter += 1;
        ctrl.hwAbstraction.GetProfiling().RecordEnd(m_InProgressProfilingEntryId);
    }
    if (pleCodeLoadedIntoPleSram)
    {
        ctrl.hwAbstraction.GetLogger().Debug("Ple code loaded into PLE sram");
        ctrl.pleCodeLoadedIntoPleSramCounter += 1;
    }
}

template <typename Ctrl>
bool PleController::HandleCommand(Ctrl& ctrl, const Command& cmd)
{
    using namespace command_stream;

    if (cmd.type == CommandType::WaitForCounter)
    {
        return ResolveWaitForCounterCommand(static_cast<const WaitForCounterCommand&>(cmd), ctrl);
    }
    else if (cmd.type == CommandType::LoadPleCodeIntoPleSram)
    {
        if (ctrl.hwAbstraction.IsPleBusy())
        {
            return false;
        }

        const LoadPleCodeIntoPleSramCommand& pleCommand = static_cast<const LoadPleCodeIntoPleSramCommand&>(cmd);
        uint32_t agentId                                = pleCommand.agentId;
        const PleS& agentData                           = ctrl.agents[agentId].pleS;

        ctrl.hwAbstraction.LoadPleCodeIntoPleSram(agentId, agentData);
        return true;
    }
    else
    {
        ASSERT(cmd.type == CommandType::StartPleStripe);

        if (ctrl.hwAbstraction.IsPleBusy())
        {
            return false;
        }

        const StartPleStripeCommand& pleCommand = static_cast<const StartPleStripeCommand&>(cmd);
        uint32_t agentId                        = pleCommand.agentId;
        const PleS& agentData                   = ctrl.agents[agentId].pleS;

        const bool isSram =
            agentData.inputMode == PleInputMode::SRAM_ONE_INPUT || agentData.inputMode == PleInputMode::SRAM_TWO_INPUTS;
        if (isSram)
        {
            // CE-enable flags are not banked like the other CE registers we set, so cannot be written in advance.
            // We may not be able to set them yet if other stripes are still running, in which case we will wait and
            // try again later
            if (!ctrl.hwAbstraction.TrySetCeEnables(CeEnables::AllEnabledForPleOnly))
            {
                return false;
            }
        }

        profiling::ProfilingOnly<uint8_t> profilingEntryId =
            ctrl.hwAbstraction.HandlePleStripeCmd(agentData, pleCommand);

        m_InProgressProfilingEntryId = profilingEntryId;

        return true;
    }
}

template <typename Ctrl>
LoggingString PleController::GetStateString(Ctrl& ctrl, uint32_t origNumCommands) const
{
    LoggingString result;
    result.AppendFormat("Ple: Stripe counter = %u, PLE code loaded into PLE sram counter = %u, %s, Commands = %s",
                        ctrl.pleStripeCounter, ctrl.pleCodeLoadedIntoPleSramCounter,
                        ctrl.hwAbstraction.IsPleBusy() ? "Busy" : "Idle",
                        CommandListToString(m_CmdQueue, origNumCommands).GetCString());
    return result;
}

inline bool PleController::IsDone() const
{
    return m_CmdQueue.IsEmpty();
}

}    // namespace ethosn::control_unit
