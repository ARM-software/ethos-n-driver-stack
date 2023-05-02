//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "include/unprivileged/Task.hpp"

#include "Firmware.hpp"

#include <common/Log.hpp>
#include <common/TaskSvc.hpp>
#include <common/hals/HardwareHal.hpp>

#include <cinttypes>

namespace ethosn
{
namespace control_unit
{
// Defined in PleKernelBinaries.hpp
extern const uint8_t g_PleKernelBinaries[];

namespace
{

// Context switch to the privileged task and pass the given message to it
void WaitForTaskMessage(TaskMessage* message)
{
    register TaskMessage* msg asm("r0") = message;
    asm volatile("svc %[svc_num]" ::"r"(msg), [svc_num] "i"(TASK_SVC_TASK_SWITCH));
}

uint64_t ApplyAddrExtend(const uint32_t addr, const uint32_t addrExtend)
{
    constexpr uint32_t shift = 29;
    constexpr uint32_t mask  = (1U << shift) - 1U;

    return (uint64_t{ addrExtend } << shift) | uint64_t{ addr & mask };
}
}    // namespace

extern "C" void Task(const TaskConfig* config)
{
    LoggerType logger({ LogSink });
    logger.SetMaxSeverity(config->logSeverity);
    logger.Debug("Unprivileged task started");

    const uint32_t pleDataAddr    = reinterpret_cast<uint32_t>(&g_PleKernelBinaries[0]);
    const uint64_t pleDataAddrU64 = ApplyAddrExtend(pleDataAddr, config->pleAddrExtend);

    HardwareHal hardware(logger);
    Firmware<HardwareHal> fw(hardware, pleDataAddrU64);

    TaskMessage message;
    message.status = TaskMessageStatus::OK;

    Firmware<HardwareHal>::InferenceResult latestInferenceResult;

    while (true)
    {
        WaitForTaskMessage(&message);
        logger.Debug("Got task message: 0x%02x", static_cast<uint32_t>(message.type));

        switch (message.type)
        {
            case TaskMessageType::CAPABILITIES:
            {
                auto capPair                   = fw.GetCapabilities();
                message.data.capabilities.data = capPair.first;
                message.data.capabilities.size = capPair.second;
                message.status                 = TaskMessageStatus::OK;
                break;
            }
            case TaskMessageType::INFERENCE:
            {
                Inference inference(message.data.inference.bufferArray);
                latestInferenceResult = fw.RunInference(inference);
                message.status = latestInferenceResult.success ? TaskMessageStatus::OK : TaskMessageStatus::FAILED;
                message.data.inference.cycleCount = latestInferenceResult.cycleCount;
                break;
            }
            case TaskMessageType::PROFILING_ENABLE:
            {
                fw.ResetAndEnableProfiling(message.data.profilingConfig.config);
                message.status = TaskMessageStatus::OK;
                break;
            }
            case TaskMessageType::PROFILING_DISABLE:
            {
                fw.StopProfiling();
                message.status = TaskMessageStatus::OK;
                break;
            }
            case TaskMessageType::POST_INFERENCE_CLEANUP:
            {
                // Even when profiling is disabled we still report some limited stats.
                logger.Info("Total inference cycle count: %" PRIu64, latestInferenceResult.cycleCount);
#if defined(CONTROL_UNIT_PROFILING)
                logger.Info("%u profiling entries written.", latestInferenceResult.numProfilingEntries.nonOverflow);
                if (latestInferenceResult.numProfilingEntries.overflow > 0)
                {
                    size_t numEntriesRequired = latestInferenceResult.numProfilingEntries.nonOverflow +
                                                latestInferenceResult.numProfilingEntries.overflow;
                    size_t numBytesRequired =
                        numEntriesRequired * sizeof(ethosn_profiling_entry) + sizeof(ethosn_profiling_buffer);
                    logger.Warning("PROFILING BUFFER IS FULL. Overflowed by %u entries. Consider increasing the size "
                                   "to at least %u entries, i.e. %u bytes.",
                                   latestInferenceResult.numProfilingEntries.overflow, numEntriesRequired,
                                   numBytesRequired);
                }
#endif
                message.status = TaskMessageStatus::OK;
                break;
            }
            default:
            {
                logger.Error("Unknown task message type: 0x%02x", static_cast<uint32_t>(message.type));
                message.status = TaskMessageStatus::FAILED;
                break;
            }
        }
    }
}

}    // namespace control_unit
}    // namespace ethosn
