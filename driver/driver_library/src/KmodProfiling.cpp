//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

// This file implements some of internal profiling functions by forwarding requests to the kernel module.
// These functions are declared in ProfilingInternal.hpp.

#include "ProfilingInternal.hpp"
#include "Utils.hpp"

#include <uapi/ethosn.h>

#include <fcntl.h>
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

namespace
{

struct TimeSync
{
    TimeSync()
        : m_Valid(false)
        , m_Delta(0)
    {}

    // True if the time delta is a valid value
    bool m_Valid;
    // Time delta between the host reference clock and the accelerator clock.
    // We are disregarding any clock drift.
    int64_t m_Delta;
};

}    // namespace

int g_FirmwareBufferFd = 0;
// Clock frequency expressed in MHz (it is provided by the kernel module).
int g_ClockFrequencyMhz = 0;

bool ConfigureKernelDriver(Configuration config, const std::string& device)
{
    if (config.m_NumHardwareCounters > 6)
    {
        std::cerr << "Warning more than 6 hardware counters specified, only the first 6 will be used.\n";
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
        throw std::runtime_error(std::string("Unable to retrieve counter value. errno: ") + strerror(errno));
    }

    return result;
}

namespace
{

using EntryId = decltype(ethosn_profiling_entry::id);

// Get time delta between firmware profiling and global profiling
TimeSync GetTimeDelta(const std::vector<ProfilingEntry>& entries)
{
    using namespace std::chrono;
    static_assert(sizeof(uint64_t) == sizeof(time_point<high_resolution_clock>), "Timestamp size does not match");

    TimeSync retVal;
    time_point<high_resolution_clock> timestamp = {};
    uint64_t sync                               = 0;
    uint8_t index                               = 0;
    EntryId lastTimelineEventId                 = 0;

    auto ResetSearchIndex = [&]() -> void {
        sync                = 0;
        index               = 0;
        lastTimelineEventId = 0;
    };

    auto HandleTimelineEventInstant = [&](const ProfilingEntry& entry) -> void {
        if (entry.m_MetadataCategory == ProfilingEntry::MetadataCategory::FirmwareTimeSync)
        {
            DataUnion temp = {};
            temp.m_Raw     = static_cast<uint32_t>(entry.m_MetadataValue);
            if (index == 0)
            {
                timestamp           = entry.m_Timestamp;
                lastTimelineEventId = static_cast<EntryId>(entry.m_Id);
            }
            else
            {
                ++lastTimelineEventId;
                if (entry.m_Id != lastTimelineEventId)
                {
                    // The Ids of the four events containing part of the host CPU timestamp must be consecutive.
                    ResetSearchIndex();
                    return;
                }
            }

            for (uint8_t i = 0; i < 2; ++i)
            {
                // The ETHOSN_MESSAGE_TIME_SYNC message is sent by the host CPU to the accelerator CPU.
                // It contains the timestamp taken by the host CPU using the reference monotonic clock.
                // This value is 64 bits long and it does not fit in the current data field of the
                // ethosn_profiling_entry which is 32 bits long (note that the firmware uses 8 bits of
                // the data field for the category).
                // This value is split in two bytes chunks and spread across four events of type
                // TIMELINE_EVENT_INSTANT.
                sync |= static_cast<uint64_t>(temp.m_TimeSyncFields.m_TimeSyncData[i])
                        << (8 * (sizeof(uint64_t) - 1 - index));
                ++index;
            }

            if (index == sizeof(uint64_t))
            {
                // Time sync fully retrieved from 4 consecutive messages
                // The timestamp of the message (i.e. PMU cycle count) is stored as the number of nanoseconds since the
                // high_resolution_clock epoch. Convert this to actual nanoseconds based on the clock frequency
                // before calculating the difference with the host CPU (also measured in nanoseconds).
                retVal.m_Delta = sync - (1000 / g_ClockFrequencyMhz) * timestamp.time_since_epoch().count();
                retVal.m_Valid = true;
                ResetSearchIndex();
                return;
            }
        }
    };

    // Update the global profiling entries
    for (const ProfilingEntry& entry : entries)
    {
        switch (entry.m_Type)
        {
            case ProfilingEntry::Type::TimelineEventInstant:
                HandleTimelineEventInstant(entry);
                break;
            default:
                break;
        }
    }

    return retVal;
}

}    // namespace

int64_t g_ProfilingDelta = 0;

// Append all firmware profiling entry to the global profiling.
bool AppendKernelDriverEntries()
{
    if (g_FirmwareBufferFd <= 0)
    {
        return false;
    }
    std::vector<ProfilingEntry> entries;

    // Read entries from the buffer until we catch up
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
                entries.push_back(ConvertProfilingEntry(readBuffer[i]));
            }
        }
    }

    TimeSync timeDelta = GetTimeDelta(entries);
    if (timeDelta.m_Valid)
    {
        g_ProfilingDelta = timeDelta.m_Delta;
    }

    // Sync up firmware entries using g_ProfilingDelta and update global profiling entries with
    // correct timestamps
    for (ProfilingEntry& entry : entries)
    {
        // The timestamp of the message (i.e. PMU cycle count) is stored as the number of nanoseconds since the
        // high_resolution_clock epoch. Convert this to actual nanoseconds based on the clock frequency
        // before calculating the difference with the host CPU (also measured in nanoseconds).
        entry.m_Timestamp = std::chrono::time_point<std::chrono::high_resolution_clock>(std::chrono::nanoseconds(
            (1000 / g_ClockFrequencyMhz) * entry.m_Timestamp.time_since_epoch().count() + g_ProfilingDelta));

        g_ProfilingEntries.push_back(entry);
    }

    return true;
}

}    // namespace profiling
}    // namespace driver_library
}    // namespace ethosn
