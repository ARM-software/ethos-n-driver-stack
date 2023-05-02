//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common/FirmwareApi.hpp>

#include <cstddef>
#include <cstdint>

namespace ethosn
{
namespace control_unit
{

// SVC function for logging a message
//
// @param r0    Log severity
// @param r1    Address to the string to log
constexpr uint32_t TASK_SVC_LOG_MESSAGE = 0x0U;

// SVC function to get the DWT's sleep cycle counter
// @return      uint32_t value
constexpr uint32_t TASK_SVC_GET_DWT_SLEEP_CYCLE_COUNT = 0x1U;

// SVC function to clean and invalidate the data cache
constexpr uint32_t TASK_SVC_DCACHE_CLEAN_INVALIDATE = 0x2U;

// SVC function for switching between the privileged and unprivileged task
//
// From unprivileged to privileged task:
// When calling this SVC function, r0 should be populated with an address to a TaskMessage struct that can be used by the
// privileged task. The struct will be used by the privileged task to request actions to be performed by the
// unprivileged task and to check the success of those actions.
//
// @param r0    Address to TaskMessage struct
//
// From privileged to unprivileged task:
// When calling this SVC function, the TaskMessage struct at the address given by the unprivileged task in a previous call
// should be populated with the action that should be performed by the unprivileged task. An exception to this is when
// calling the SVC function for the first time as no address have been given by the unprivileged task at that time.
//
// @return      Address to TaskMessage struct
//
constexpr uint32_t TASK_SVC_TASK_SWITCH = 0xFFU;

enum class TaskMessageStatus : uint32_t
{
    OK,
    FAILED
};

enum class TaskMessageType : uint32_t
{
    CAPABILITIES,

    INFERENCE,

    PROFILING_ENABLE,
    PROFILING_DISABLE,

    POST_INFERENCE_CLEANUP,
};

struct InferenceData
{
    uint64_t bufferArray;    ///< Passed from MainHardware to non-privileged task.
    uint64_t cycleCount;     ///< Passed back from non-privileged task to MainHardware.
};

struct CapabilitiesData
{
    const char* data;
    size_t size;
};

struct ProfilingConfigData
{
    /// Set by privileged, read by unprivileged, to provide the profiling configuration.
    ethosn_firmware_profiling_configuration config;
};

static_assert(ETHOSN_PROFILING_MAX_HW_COUNTERS <= 6U, "Only up to 6 hardware counters are supported");

struct TaskMessage
{
    TaskMessageType type;
    TaskMessageStatus status;
    union
    {
        CapabilitiesData capabilities;
        InferenceData inference;
        ProfilingConfigData profilingConfig;
    } data;
};

}    // namespace control_unit
}    // namespace ethosn
