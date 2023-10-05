//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "../Profiling.hpp"
#include "CommandList.hpp"
#include <common/Log.hpp>

#include <variant>

namespace ethosn::control_unit
{

class DmaWrController
{
public:
    DmaWrController(const Command* commandsBegin, uint32_t numCommands, const char* endOfCmdStream);

    template <typename Ctrl>
    bool HandleCommands(Ctrl& ctrl);

    template <typename Ctrl>
    void UpdateProgress(Ctrl& ctrl);

    template <typename Ctrl>
    LoggingString GetStateString(Ctrl& ctrl, uint32_t origNumCommands) const;

    bool IsDone() const;

private:
    static constexpr uint32_t m_PrefetchSize = 4 * g_CacheLineSize;    // Found experimentally

    template <typename Ctrl>
    bool HandleCommand(Ctrl& ctrl, const Command& cmd);

    CommandList m_CmdQueue;
    uint32_t m_NumCommandsInProgress;

    profiling::ProfilingOnly<uint32_t> m_ProfilingWrappingCounter = 0;
    profiling::ProfilingOnly<uint8_t> m_InProgressProfilingEntryIds[4];
};

inline DmaWrController::DmaWrController(const Command* commandsBegin, uint32_t numCommands, const char* endOfCmdStream)
    : m_CmdQueue(commandsBegin, numCommands, m_PrefetchSize, endOfCmdStream)
    , m_NumCommandsInProgress(0)
{}

template <typename Ctrl>
bool DmaWrController::HandleCommands(Ctrl& ctrl)
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
inline void DmaWrController::UpdateProgress(Ctrl& ctrl)
{
    const uint32_t numCmdsInHwQueue = ctrl.hwAbstraction.GetNumCmdsInDmaWrQueue();
    const uint32_t numCompletedJobs = m_NumCommandsInProgress - numCmdsInHwQueue;
    if (numCompletedJobs > 0)
    {
        ctrl.hwAbstraction.GetLogger().Debug("%u DmaWr command(s) completed", numCompletedJobs);
        m_NumCommandsInProgress -= numCompletedJobs;

        for (uint32_t i = 0; i < numCompletedJobs; ++i)
        {
            ctrl.hwAbstraction.GetProfiling().RecordEnd(m_InProgressProfilingEntryIds[(ctrl.dmaWrCounter + i) % 4]);
        }

        ctrl.dmaWrCounter += numCompletedJobs;
    }
}

template <typename Ctrl>
bool DmaWrController::HandleCommand(Ctrl& ctrl, const Command& cmd)
{
    using namespace command_stream;

    if (cmd.type == CommandType::WaitForCounter)
    {
        return ResolveWaitForCounterCommand(static_cast<const WaitForCounterCommand&>(cmd), ctrl);
    }
    ASSERT(cmd.type == CommandType::StoreOfmStripe);

    const DmaCommand& dmaCommand = static_cast<const DmaCommand&>(cmd);
    uint32_t agentId             = dmaCommand.agentId;

    // If the HW queue has no space for any more commands, then we can't do anything.
    if (m_NumCommandsInProgress == 4)
    {
        return false;
    }

    const Agent& agent = ctrl.agents[agentId];

    profiling::ProfilingOnly<uint8_t> profilingEntryId = ctrl.hwAbstraction.HandleDmaWrCmdOfm(agent.ofm, dmaCommand);
    ETHOSN_UNUSED(profilingEntryId);

    m_NumCommandsInProgress += 1;

    m_InProgressProfilingEntryIds[m_ProfilingWrappingCounter] = profilingEntryId;
    m_ProfilingWrappingCounter                                = (m_ProfilingWrappingCounter + 1) % 4;

    return true;
}

template <typename Ctrl>
LoggingString DmaWrController::GetStateString(Ctrl& ctrl, uint32_t origNumCommands) const
{
    LoggingString result;
    result.AppendFormat("DmaWr: Counter = %u, In-progress = %u, Commands = %s", ctrl.dmaWrCounter,
                        m_NumCommandsInProgress, CommandListToString(m_CmdQueue, origNumCommands).GetCString());
    return result;
}

inline bool DmaWrController::IsDone() const
{
    return m_CmdQueue.IsEmpty() && m_NumCommandsInProgress == 0;
}

}    // namespace ethosn::control_unit
