//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
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

ethosn_profiling_hw_counter_types ConvertHwCountersToKernel(HardwareCounters counter)
{
    switch (counter)
    {
        case HardwareCounters::FirmwareBusAccessRdTransfers:
        {
            return ethosn_profiling_hw_counter_types::BUS_ACCESS_RD_TRANSFERS;
        }
        case HardwareCounters::FirmwareBusRdCompleteTransfers:
        {
            return ethosn_profiling_hw_counter_types::BUS_RD_COMPLETE_TRANSFERS;
        }
        case HardwareCounters::FirmwareBusReadBeats:
        {
            return ethosn_profiling_hw_counter_types::BUS_READ_BEATS;
        }
        case HardwareCounters::FirmwareBusReadTxfrStallCycles:
        {
            return ethosn_profiling_hw_counter_types::BUS_READ_TXFR_STALL_CYCLES;
        }
        case HardwareCounters::FirmwareBusAccessWrTransfers:
        {
            return ethosn_profiling_hw_counter_types::BUS_ACCESS_WR_TRANSFERS;
        }
        case HardwareCounters::FirmwareBusWrCompleteTransfers:
        {
            return ethosn_profiling_hw_counter_types::BUS_WR_COMPLETE_TRANSFERS;
        }
        case HardwareCounters::FirmwareBusWriteBeats:
        {
            return ethosn_profiling_hw_counter_types::BUS_WRITE_BEATS;
        }
        case HardwareCounters::FirmwareBusWriteTxfrStallCycles:
        {
            return ethosn_profiling_hw_counter_types::BUS_WRITE_TXFR_STALL_CYCLES;
        }
        case HardwareCounters::FirmwareBusWriteStallCycles:
        {
            return ethosn_profiling_hw_counter_types::BUS_WRITE_STALL_CYCLES;
        }
        case HardwareCounters::FirmwareBusErrorCount:
        {
            return ethosn_profiling_hw_counter_types::BUS_ERROR_COUNT;
        }
        case HardwareCounters::FirmwareNcuMcuIcacheMiss:
        {
            return ethosn_profiling_hw_counter_types::NCU_MCU_ICACHE_MISS;
        }
        case HardwareCounters::FirmwareNcuMcuDcacheMiss:
        {
            return ethosn_profiling_hw_counter_types::NCU_MCU_DCACHE_MISS;
        }
        case HardwareCounters::FirmwareNcuMcuBusReadBeats:
        {
            return ethosn_profiling_hw_counter_types::NCU_MCU_BUS_READ_BEATS;
        }
        case HardwareCounters::FirmwareNcuMcuBusWriteBeats:
        {
            return ethosn_profiling_hw_counter_types::NCU_MCU_BUS_WRITE_BEATS;
        }
        default:
        {
            assert(!"ethosn_profiling_hw_counter_types not insync with HardwareCounters");
            return ethosn_profiling_hw_counter_types::NCU_MCU_BUS_WRITE_BEATS;
        }
    }
}

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
int g_ClockFrequency = 0;

bool ConfigureKernelDriver(Configuration config)
{
    if (config.m_NumHardwareCounters > 6)
    {
        std::cerr << "Warning more than 6 hardware counters specified, only the first 6 will be used.\n";
        return false;
    }
    int ethosnFd = open(STRINGIZE_VALUE_OF(DEVICE_NODE), O_RDONLY);
    if (ethosnFd < 0)
    {
        throw std::runtime_error(std::string("Unable to open ") + std::string(STRINGIZE_VALUE_OF(DEVICE_NODE)) +
                                 std::string(": ") + strerror(errno));
    }

    ethosn_profiling_config kernelConfig;
    kernelConfig.enable_profiling     = config.m_EnableProfiling;
    kernelConfig.firmware_buffer_size = config.m_FirmwareBufferSize;
    kernelConfig.num_hw_counters      = config.m_NumHardwareCounters;
    for (uint32_t i = 0; i < kernelConfig.num_hw_counters; ++i)
    {
        kernelConfig.hw_counters[i] = ConvertHwCountersToKernel(config.m_HardwareCounters[i]);
    }
    int result       = ioctl(ethosnFd, ETHOSN_IOCTL_CONFIGURE_PROFILING, &kernelConfig);
    g_ClockFrequency = ioctl(ethosnFd, ETHOSN_IOCTL_GET_CLOCK_FREQUENCY);
    close(ethosnFd);

    if (result != 0)
    {
        return false;
    }

    if (g_ClockFrequency <= 0)
    {
        g_ClockFrequency = 0;
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
        g_FirmwareBufferFd = open(STRINGIZE_VALUE_OF(FIRMWARE_PROFILING_NODE), O_RDONLY);
    }
    else
    {
        g_FirmwareBufferFd = 0;
    }

    return true;
}

uint64_t GetKernelDriverCounterValue(PollCounterName counter)
{
    int ethosnFd = open(STRINGIZE_VALUE_OF(DEVICE_NODE), O_RDONLY);
    if (ethosnFd < 0)
    {
        throw std::runtime_error(std::string("Unable to open ") + std::string(STRINGIZE_VALUE_OF(DEVICE_NODE)) +
                                 std::string(": ") + strerror(errno));
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
        default:
            assert(!"Invalid counter");
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

using EntryId   = decltype(ethosn_profiling_entry::id);
using EntryData = decltype(ethosn_profiling_entry::data);

uint64_t GetIdForCounterValue(EntryId id)
{
    uint64_t retVal;
    // Convert ID (which is this case is the counter name)
    switch (static_cast<FirmwareCounterName>(id))
    {
        case FirmwareCounterName::DwtSleepCycleCount:
            retVal = static_cast<uint64_t>(CollatedCounterName::FirmwareDwtSleepCycleCount);
            break;
        case FirmwareCounterName::EventQueueSize:
            retVal = static_cast<uint64_t>(CollatedCounterName::FirmwareEventQueueSize);
            break;
        case FirmwareCounterName::DmaNumReads:
            retVal = static_cast<uint64_t>(CollatedCounterName::FirmwareDmaNumReads);
            break;
        case FirmwareCounterName::DmaNumWrites:
            retVal = static_cast<uint64_t>(CollatedCounterName::FirmwareDmaNumWrites);
            break;
        case FirmwareCounterName::DmaReadBytes:
            retVal = static_cast<uint64_t>(CollatedCounterName::FirmwareDmaReadBytes);
            break;
        case FirmwareCounterName::DmaWriteBytes:
            retVal = static_cast<uint64_t>(CollatedCounterName::FirmwareDmaWriteBytes);
            break;
        case FirmwareCounterName::BusAccessRdTransfers:
            retVal = static_cast<uint64_t>(CollatedCounterName::FirmwareBusAccessRdTransfers);
            break;
        case FirmwareCounterName::BusRdCompleteTransfers:
            retVal = static_cast<uint64_t>(CollatedCounterName::FirmwareBusRdCompleteTransfers);
            break;
        case FirmwareCounterName::BusReadBeats:
            retVal = static_cast<uint64_t>(CollatedCounterName::FirmwareBusReadBeats);
            break;
        case FirmwareCounterName::BusReadTxfrStallCycles:
            retVal = static_cast<uint64_t>(CollatedCounterName::FirmwareBusReadTxfrStallCycles);
            break;
        case FirmwareCounterName::BusAccessWrTransfers:
            retVal = static_cast<uint64_t>(CollatedCounterName::FirmwareBusAccessWrTransfers);
            break;
        case FirmwareCounterName::BusWrCompleteTransfers:
            retVal = static_cast<uint64_t>(CollatedCounterName::FirmwareBusWrCompleteTransfers);
            break;
        case FirmwareCounterName::BusWriteBeats:
            retVal = static_cast<uint64_t>(CollatedCounterName::FirmwareBusWriteBeats);
            break;
        case FirmwareCounterName::BusWriteTxfrStallCycles:
            retVal = static_cast<uint64_t>(CollatedCounterName::FirmwareBusWriteTxfrStallCycles);
            break;
        case FirmwareCounterName::BusWriteStallCycles:
            retVal = static_cast<uint64_t>(CollatedCounterName::FirmwareBusWriteStallCycles);
            break;
        case FirmwareCounterName::BusErrorCount:
            retVal = static_cast<uint64_t>(CollatedCounterName::FirmwareBusErrorCount);
            break;
        case FirmwareCounterName::NcuMcuIcacheMiss:
            retVal = static_cast<uint64_t>(CollatedCounterName::FirmwareNcuMcuIcacheMiss);
            break;
        case FirmwareCounterName::NcuMcuDcacheMiss:
            retVal = static_cast<uint64_t>(CollatedCounterName::FirmwareNcuMcuDcacheMiss);
            break;
        case FirmwareCounterName::NcuMcuBusReadBeats:
            retVal = static_cast<uint64_t>(CollatedCounterName::FirmwareNcuMcuBusReadBeats);
            break;
        case FirmwareCounterName::NcuMcuBusWriteBeats:
            retVal = static_cast<uint64_t>(CollatedCounterName::FirmwareNcuMcuBusWriteBeats);
            break;
        default:
            assert(false);
            break;
    }
    return retVal;
}

EntryDataCategory GetFirmwareCategory(const EntryData data)
{
    DataUnion temp = {};
    temp.m_Raw     = data;
    return temp.m_Category;
}

ProfilingEntry::MetadataCategory ConvertCategoryEntry(const EntryDataCategory category)
{
    ProfilingEntry::MetadataCategory retVal;
    switch (category)
    {
        case EntryDataCategory::WfeSleeping:
            retVal = ProfilingEntry::MetadataCategory::FirmwareWfeSleeping;
            break;
        case EntryDataCategory::Inference:
            retVal = ProfilingEntry::MetadataCategory::FirmwareInference;
            break;
        case EntryDataCategory::Command:
            retVal = ProfilingEntry::MetadataCategory::FirmwareCommand;
            break;
        case EntryDataCategory::Dma:
            retVal = ProfilingEntry::MetadataCategory::FirmwareDma;
            break;
        case EntryDataCategory::Tsu:
            retVal = ProfilingEntry::MetadataCategory::FirmwareTsu;
            break;
        case EntryDataCategory::MceStripeSetup:
            retVal = ProfilingEntry::MetadataCategory::FirmwareMceStripeSetup;
            break;
        case EntryDataCategory::PleStripeSetup:
            retVal = ProfilingEntry::MetadataCategory::FirmwarePleStripeSetup;
            break;
        case EntryDataCategory::Label:
            retVal = ProfilingEntry::MetadataCategory::FirmwareLabel;
            break;
        case EntryDataCategory::DmaSetup:
            retVal = ProfilingEntry::MetadataCategory::FirmwareDmaSetup;
            break;
        case EntryDataCategory::GetCompleteCommand:
            retVal = ProfilingEntry::MetadataCategory::FirmwareGetCompleteCommand;
            break;
        case EntryDataCategory::ScheduleNextCommand:
            retVal = ProfilingEntry::MetadataCategory::FirmwareScheduleNextCommand;
            break;
        case EntryDataCategory::WfeChecking:
            retVal = ProfilingEntry::MetadataCategory::FirmwareWfeChecking;
            break;
        case EntryDataCategory::TimeSync:
            retVal = ProfilingEntry::MetadataCategory::FirmwareTimeSync;
            break;
        default:
            assert(false);
            break;
    }
    return retVal;
}

// Converts a profiling entry reported by the kernel into the Driver Library's public ProfilingEntry representation.
ProfilingEntry ConvertProfilingEntry(const ethosn_profiling_entry& kernelEntry)
{
    ProfilingEntry result;
    // Clock frequency is expressed in MHz.
    result.m_Timestamp = std::chrono::time_point<std::chrono::high_resolution_clock>(
        std::chrono::nanoseconds((1000 / g_ClockFrequency) * kernelEntry.timestamp));
    switch (kernelEntry.type)
    {
        case ethosn_profiling_entry_type::COUNTER_VALUE:
            result.m_Id               = GetIdForCounterValue(kernelEntry.id);
            result.m_Type             = ProfilingEntry::Type::CounterSample;
            result.m_MetadataCategory = ProfilingEntry::MetadataCategory::CounterValue;
            result.m_MetadataValue    = kernelEntry.data;
            break;
        case ethosn_profiling_entry_type::TIMELINE_EVENT_START:
            result.m_Id               = static_cast<uint64_t>(kernelEntry.id);
            result.m_Type             = ProfilingEntry::Type::TimelineEventStart;
            result.m_MetadataCategory = ConvertCategoryEntry(GetFirmwareCategory(kernelEntry.data));
            result.m_MetadataValue    = kernelEntry.data;
            break;
        case ethosn_profiling_entry_type::TIMELINE_EVENT_END:
            result.m_Id               = static_cast<uint64_t>(kernelEntry.id);
            result.m_Type             = ProfilingEntry::Type::TimelineEventEnd;
            result.m_MetadataCategory = ConvertCategoryEntry(GetFirmwareCategory(kernelEntry.data));
            result.m_MetadataValue    = kernelEntry.data;
            break;
        case ethosn_profiling_entry_type::TIMELINE_EVENT_INSTANT:
            result.m_Id               = static_cast<uint64_t>(kernelEntry.id);
            result.m_Type             = ProfilingEntry::Type::TimelineEventInstant;
            result.m_MetadataCategory = ConvertCategoryEntry(GetFirmwareCategory(kernelEntry.data));
            result.m_MetadataValue    = kernelEntry.data;
            break;
        default:
            assert(false);
            break;
    }
    return result;
}

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
                retVal.m_Delta = sync - timestamp.time_since_epoch().count();
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
        entry.m_Timestamp = std::chrono::time_point<std::chrono::high_resolution_clock>(
            std::chrono::nanoseconds(entry.m_Timestamp.time_since_epoch().count() + g_ProfilingDelta));
        g_ProfilingEntries.push_back(entry);
    }

    return true;
}

}    // namespace profiling
}    // namespace driver_library
}    // namespace ethosn
