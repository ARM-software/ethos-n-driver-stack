//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

// This file implements some of internal profiling functions by forwarding requests to the kernel module.
// These functions are declared in ProfilingInternal.hpp.

#include "ProfilingInternal.hpp"
#include "Utils.hpp"

#include <uapi/ethosn.h>

#include <ethosn_utils/Strings.hpp>

#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

namespace ethosn
{
namespace driver_library
{
namespace profiling
{

int g_FirmwareBufferFd = 0;
// Clock frequency expressed in MHz (it is provided by the kernel module).
int g_ClockFrequencyMhz                       = 0;
std::string g_FirmwareProfilingOffsetFilename = ethosn::utils::ReplaceAll(
    std::string(FIRMWARE_PROFILING_NODE), "firmware_profiling", "wall_clock_time_at_firmware_zero");

bool ConfigureKernelDriver(Configuration config, const std::string& device)
{
    if (config.m_NumHardwareCounters > 6)
    {
        g_Logger.Warning("More than 6 hardware counters specified, only the first 6 will be used.");
        return false;
    }
    int ethosnFd = open(device.c_str(), O_RDONLY);
    if (ethosnFd < 0)
    {
        throw std::runtime_error(std::string("Unable to open " + device + ": ") + strerror(errno));
    }

    ethosn_profiling_config kernelConfig;
    kernelConfig.enable_profiling     = config.m_EnableProfiling;
    kernelConfig.firmware_buffer_size = config.m_FirmwareBufferSize;
    kernelConfig.num_hw_counters      = config.m_NumHardwareCounters;
    for (uint32_t i = 0; i < kernelConfig.num_hw_counters; ++i)
    {
        kernelConfig.hw_counters[i] = ConvertHwCountersToKernel(config.m_HardwareCounters[i]);
    }
    int result          = ioctl(ethosnFd, ETHOSN_IOCTL_CONFIGURE_PROFILING, &kernelConfig);
    g_ClockFrequencyMhz = ioctl(ethosnFd, ETHOSN_IOCTL_GET_CLOCK_FREQUENCY);
    close(ethosnFd);

    if (result != 0)
    {
        return false;
    }

    if (g_ClockFrequencyMhz <= 0)
    {
        g_ClockFrequencyMhz = 0;
        return false;
    }

    // Close firmware profiling buffer file if it was open before
    if (g_FirmwareBufferFd > 0)
    {
        close(g_FirmwareBufferFd);
    }
    // Re-open if profiling is now enabled
    if (kernelConfig.enable_profiling)
    {
        g_FirmwareBufferFd = open(FIRMWARE_PROFILING_NODE, O_RDONLY);
    }
    else
    {
        g_FirmwareBufferFd = 0;
    }

    return true;
}

uint64_t GetKernelDriverCounterValue(PollCounterName counter, const std::string& device)
{
    int ethosnFd = open(device.c_str(), O_RDONLY);
    if (ethosnFd < 0)
    {
        throw std::runtime_error(std::string("Unable to open " + device + ": ") + strerror(errno));
    }

    ethosn_poll_counter_name kernelCounterName;
    switch (counter)
    {
        case PollCounterName::KernelDriverNumMailboxMessagesSent:
            kernelCounterName = ETHOSN_POLL_COUNTER_NAME_MAILBOX_MESSAGES_SENT;
            break;
        case PollCounterName::KernelDriverNumMailboxMessagesReceived:
            kernelCounterName = ETHOSN_POLL_COUNTER_NAME_MAILBOX_MESSAGES_RECEIVED;
            break;
        case PollCounterName::KernelDriverNumRuntimePowerSuspend:
            kernelCounterName = ETHOSN_POLL_COUNTER_NAME_RPM_SUSPEND;
            break;
        case PollCounterName::KernelDriverNumRuntimePowerResume:
            kernelCounterName = ETHOSN_POLL_COUNTER_NAME_RPM_RESUME;
            break;
        case PollCounterName::KernelDriverNumPowerSuspend:
            kernelCounterName = ETHOSN_POLL_COUNTER_NAME_PM_SUSPEND;
            break;
        case PollCounterName::KernelDriverNumPowerResume:
            kernelCounterName = ETHOSN_POLL_COUNTER_NAME_PM_RESUME;
            break;
        default:
            ETHOSN_FAIL_MSG("Invalid counter");
    }

    int result = ioctl(ethosnFd, ETHOSN_IOCTL_GET_COUNTER_VALUE, &kernelCounterName);

    close(ethosnFd);

    if (result < 0)
    {
        throw std::runtime_error(std::string("Unable to retrieve counter value: ") + strerror(errno));
    }

    return result;
}

// Append all firmware profiling entry to the global profiling.
bool AppendKernelDriverEntries()
{
    if (g_FirmwareBufferFd <= 0)
    {
        return false;
    }

    // Read the firmware timestamp offset from the kernel module
    uint64_t profilingTimestampOffset = 0;
    {
        std::ifstream f(g_FirmwareProfilingOffsetFilename);
        f >> profilingTimestampOffset;
    }

    // Read entries from the buffer until we catch up
    std::map<uint8_t, profiling::ProfilingEntry> inProgressTimelineEvents;
    uint64_t mostRecentCorrectedKernelTimestamp = 0;
    std::array<ethosn_profiling_entry, 64> readBuffer;
    while (true)
    {
        int result = static_cast<int>(read(g_FirmwareBufferFd, readBuffer.begin(), sizeof(readBuffer)));
        if (result < 0)
        {
            return false;
        }
        else if (result == 0)
        {
            break;
        }
        else
        {
            size_t numEntriesRead = static_cast<size_t>(result) / sizeof(ethosn_profiling_entry);
            for (size_t i = 0; i < numEntriesRead; ++i)
            {
                std::pair<bool, profiling::ProfilingEntry> successAndEntry =
                    ConvertProfilingEntry(readBuffer[i], inProgressTimelineEvents, mostRecentCorrectedKernelTimestamp,
                                          g_ClockFrequencyMhz, profilingTimestampOffset);
                // Not all firmware profiling entries yield an entry we expose from the driver library
                if (successAndEntry.first)
                {
                    g_ProfilingEntries.push_back(successAndEntry.second);
                }
            }
        }
    }

    return true;
}

}    // namespace profiling
}    // namespace driver_library
}    // namespace ethosn
