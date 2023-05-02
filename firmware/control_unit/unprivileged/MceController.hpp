//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "../Profiling.hpp"
#include "CommandList.hpp"
#include "PleController.hpp"
#include <common/Log.hpp>

#include <common/Utils.hpp>

#include <variant>

namespace ethosn::control_unit
{

class MceController
{
public:
    MceController(const Command* commandsBegin, uint32_t numCommands);

    template <typename Ctrl>
    bool HandleCommands(Ctrl& ctrl);

    template <typename Ctrl>
    void UpdateProgress(Ctrl& ctrl);

    template <typename Ctrl>
    LoggingString GetStateString(Ctrl& ctrl, uint32_t origNumCommands) const;

    bool IsDone() const;

private:
    template <typename Ctrl>
    bool HandleCommand(Ctrl& ctrl, const Command& cmd);

    template <typename Ctrl>
    bool HandleWriteMceStripeRegs(Ctrl& ctrl, const Command& cmd);

    template <typename Ctrl>
    bool HandleConfigMceif(Ctrl& ctrl, const Command& cmd);

    template <typename Ctrl>
    bool HandleStartMceStripeBank(Ctrl& ctrl, const Command& cmd);

    CommandList m_CmdQueue;
    uint32_t m_NumCommandsInProgress;

    profiling::ProfilingOnly<uint32_t> m_ProfilingWrappingCounter = 0;
    profiling::ProfilingOnly<uint8_t> m_InProgressProfilingEntryIds[2];
};

inline MceController::MceController(const Command* commandsBegin, uint32_t numCommands)
    : m_CmdQueue(commandsBegin, numCommands)
    , m_NumCommandsInProgress(0)
{}

template <typename Ctrl>
bool MceController::HandleCommands(Ctrl& ctrl)
{
    bool madeProgress = false;
    while (!m_CmdQueue.IsEmpty())
    {
        if (!HandleCommand(ctrl, m_CmdQueue.GetFirst()))
        {
            break;
        }
        m_CmdQueue.RemoveFirst();
        madeProgress = true;
    }

    return madeProgress;
}

template <typename Ctrl>
inline void MceController::UpdateProgress(Ctrl& ctrl)
{
    const uint32_t numCmdsInHwQueue = ctrl.hwAbstraction.GetNumCmdsInMceQueue();
    const uint32_t numCompletedJobs = m_NumCommandsInProgress - numCmdsInHwQueue;
    if (numCompletedJobs > 0)
    {
        ctrl.hwAbstraction.GetLogger().Debug("%u Mce stripe(s) completed", numCompletedJobs);
        m_NumCommandsInProgress -= numCompletedJobs;

        for (uint32_t i = 0; i < numCompletedJobs; ++i)
        {
            ctrl.hwAbstraction.GetProfiling().RecordEnd(m_InProgressProfilingEntryIds[(ctrl.mceStripeCounter + i) % 2]);
        }

        ctrl.mceStripeCounter += numCompletedJobs;
    }
}

template <typename Ctrl>
bool MceController::HandleCommand(Ctrl& ctrl, const Command& cmd)
{
    using namespace command_stream;

    if (cmd.type == CommandType::WaitForCounter)
    {
        return ResolveWaitForCounterCommand(static_cast<const WaitForCounterCommand&>(cmd), ctrl);
    }
    else if (cmd.type == CommandType::ProgramMceStripe)
    {
        return HandleWriteMceStripeRegs(ctrl, cmd);
    }
    else if (cmd.type == CommandType::ConfigMceif)
    {
        return HandleConfigMceif(ctrl, cmd);
    }
    else if (cmd.type == CommandType::StartMceStripe)
    {
        return HandleStartMceStripeBank(ctrl, cmd);
    }
    else
    {
        ASSERT_MSG(false, "Unexpected command type");
        return false;
    }
}

template <typename Ctrl>
bool MceController::HandleWriteMceStripeRegs(Ctrl& ctrl, const Command& cmd)
{
    if (m_NumCommandsInProgress == 2)
    {
        return false;
    }

    const ProgramMceStripeCommand& programMceCommand = static_cast<const ProgramMceStripeCommand&>(cmd);
    uint32_t agentId                                 = programMceCommand.agentId;

    ctrl.hwAbstraction.HandleWriteMceStripeRegs(ctrl.agents[agentId].mce, programMceCommand);

    return true;
}

template <typename Ctrl>
bool MceController::HandleConfigMceif(Ctrl& ctrl, const Command& cmd)
{
    ctrl.hwAbstraction.ConfigMcePle(ctrl.agents[static_cast<const ConfigMceifCommand&>(cmd).agentId].mce);
    ctrl.mceifCounter += 1;
    return true;
}

template <typename Ctrl>
bool MceController::HandleStartMceStripeBank(Ctrl& ctrl, const Command& cmd)
{
    ASSERT(m_NumCommandsInProgress < 2);

    const StartMceStripeCommand& startMceCommand = static_cast<const StartMceStripeCommand&>(cmd);
    uint32_t agentId                             = startMceCommand.agentId;

    const MceS& agentData = ctrl.agents[agentId].mce;

    // CE-enable flags are not banked like the other CE registers we set, so cannot be written in advance.
    // We may not be able to set them yet if other stripes are still running, in which case we will wait and
    // try again later
    if (!ctrl.hwAbstraction.TrySetCeEnables(static_cast<CeEnables>(startMceCommand.CE_ENABLES)))
    {
        return false;
    }

    profiling::ProfilingOnly<uint8_t> profilingEntryId =
        ctrl.hwAbstraction.HandleStartMceStripeBank(agentData, startMceCommand);

    m_InProgressProfilingEntryIds[m_ProfilingWrappingCounter] = profilingEntryId;
    m_ProfilingWrappingCounter                                = (m_ProfilingWrappingCounter + 1) % 2;

    m_NumCommandsInProgress += 1;

    return true;
}

template <typename Ctrl>
LoggingString MceController::GetStateString(Ctrl& ctrl, uint32_t origNumCommands) const
{
    LoggingString result;
    result.AppendFormat("Mce: Stripe counter = %u, Mceif counter = %u, In-progress = %u, Commands = %s",
                        ctrl.mceStripeCounter, ctrl.mceifCounter, m_NumCommandsInProgress,
                        CommandListToString(m_CmdQueue, origNumCommands).GetCString());
    return result;
}

inline bool MceController::IsDone() const
{
    return m_CmdQueue.IsEmpty() && m_NumCommandsInProgress == 0;
}

}    // namespace ethosn::control_unit
