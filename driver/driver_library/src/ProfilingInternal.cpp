//
// Copyright © 2019-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "ProfilingInternal.hpp"

#include <ethosn_utils/Macros.hpp>

#include "Utils.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace ethosn
{
namespace driver_library
{
namespace profiling
{

uint64_t GetNextTimelineEventId()
{
    g_NextTimelineEventId = g_NextTimelineEventId + 1;
    return g_NextTimelineEventId;
}

bool ApplyConfiguration(Configuration config, const std::string& device)
{
    bool hasKernelConfigureSucceeded = ConfigureKernelDriver(config, device);

    if (hasKernelConfigureSucceeded && g_CurrentConfiguration.m_EnableProfiling && !config.m_EnableProfiling)
    {
        g_ProfilingEntries.clear();
        g_BufferToLifetimeEventId.clear();
        g_InferenceToLifetimeEventId.clear();
        g_NextTimelineEventId = 0;
    }

    return hasKernelConfigureSucceeded;
}

namespace
{
std::vector<std::string> Split(std::string s, char delim)
{
    std::stringstream ss(s);
    std::string token;
    std::vector<std::string> results;
    while (std::getline(ss, token, delim))
    {
        results.push_back(token);
    }
    return results;
}
}    // namespace

Configuration GetConfigFromString(const char* str)
{
    if (!str)
    {
        return Configuration();
    }
    Configuration config     = {};
    config.m_EnableProfiling = true;
    for (auto option : Split(str, ' '))
    {
        auto optionPair         = Split(option, '=');
        std::string optionName  = optionPair.size() >= 1 ? optionPair[0] : "";
        std::string optionValue = optionPair.size() >= 2 ? optionPair[1] : "";
        if (optionName == "dumpFile")
        {
            g_DumpFile = optionValue;
        }
        else if (optionName == "firmwareBufferSize")
        {
            config.m_FirmwareBufferSize = static_cast<uint32_t>(std::stoul(optionValue));
        }
        else if (optionName == "hwCounters")
        {
            auto hwCounters = Split(optionValue, ',');
            if (hwCounters.size() > 6)
            {
                g_Logger.Error("There can only be at most 6 hardware counters");
                continue;
            }
            for (auto counter : hwCounters)
            {
                if (counter == "busAccessRdTransfers")
                {
                    config.m_HardwareCounters[config.m_NumHardwareCounters++] =
                        HardwareCounters::FirmwareBusAccessRdTransfers;
                }
                else if (counter == "busRdCompleteTransfers")
                {
                    config.m_HardwareCounters[config.m_NumHardwareCounters++] =
                        HardwareCounters::FirmwareBusRdCompleteTransfers;
                }
                else if (counter == "busReadBeats")
                {
                    config.m_HardwareCounters[config.m_NumHardwareCounters++] = HardwareCounters::FirmwareBusReadBeats;
                }
                else if (counter == "busReadTxfrStallCycles")
                {
                    config.m_HardwareCounters[config.m_NumHardwareCounters++] =
                        HardwareCounters::FirmwareBusReadTxfrStallCycles;
                }
                else if (counter == "busAccessWrTransfers")
                {
                    config.m_HardwareCounters[config.m_NumHardwareCounters++] =
                        HardwareCounters::FirmwareBusAccessWrTransfers;
                }
                else if (counter == "busWrCompleteTransfers")
                {
                    config.m_HardwareCounters[config.m_NumHardwareCounters++] =
                        HardwareCounters::FirmwareBusWrCompleteTransfers;
                }
                else if (counter == "busWriteBeats")
                {
                    config.m_HardwareCounters[config.m_NumHardwareCounters++] = HardwareCounters::FirmwareBusWriteBeats;
                }
                else if (counter == "busWriteTxfrStallCycles")
                {
                    config.m_HardwareCounters[config.m_NumHardwareCounters++] =
                        HardwareCounters::FirmwareBusWriteTxfrStallCycles;
                }
                else if (counter == "busWriteStallCycles")
                {
                    config.m_HardwareCounters[config.m_NumHardwareCounters++] =
                        HardwareCounters::FirmwareBusWriteStallCycles;
                }
                else if (counter == "busErrorCount")
                {
                    config.m_HardwareCounters[config.m_NumHardwareCounters++] = HardwareCounters::FirmwareBusErrorCount;
                }
                else if (counter == "ncuMcuIcacheMiss")
                {
                    config.m_HardwareCounters[config.m_NumHardwareCounters++] =
                        HardwareCounters::FirmwareNcuMcuIcacheMiss;
                }
                else if (counter == "ncuMcuDcacheMiss")
                {
                    config.m_HardwareCounters[config.m_NumHardwareCounters++] =
                        HardwareCounters::FirmwareNcuMcuDcacheMiss;
                }
                else if (counter == "ncuMcuBusReadBeats")
                {
                    config.m_HardwareCounters[config.m_NumHardwareCounters++] =
                        HardwareCounters::FirmwareNcuMcuBusReadBeats;
                }
                else if (counter == "ncuMcuBusWriteBeats")
                {
                    config.m_HardwareCounters[config.m_NumHardwareCounters++] =
                        HardwareCounters::FirmwareNcuMcuBusWriteBeats;
                }
            }
        }
    }
    return config;
}

// In scenarios with multiple devices it is a known limitation that
// profiling configuration for devices other than the default one may
// be out of sync.
Configuration GetDefaultConfiguration()
{
    const char* profilingConfigEnv = std::getenv("ETHOSN_DRIVER_LIBRARY_PROFILING_CONFIG");
    if (!profilingConfigEnv)
    {
        return Configuration();
    }
    auto config = GetConfigFromString(profilingConfigEnv);

    if (!ApplyConfiguration(config, DEVICE_NODE))
    {
        return Configuration();
    }
    return config;
}

std::string g_DumpFile               = "";
Configuration g_CurrentConfiguration = GetDefaultConfiguration();

std::vector<ProfilingEntry> g_ProfilingEntries              = {};
std::map<Buffer*, uint64_t> g_BufferToLifetimeEventId       = {};
std::map<Inference*, uint64_t> g_InferenceToLifetimeEventId = {};
uint64_t g_NextTimelineEventId                              = 0;

bool Configure(Configuration config, const std::string& device)
{
    bool isConfigurationApplied = ApplyConfiguration(config, device);
    if (isConfigurationApplied)
    {
        g_CurrentConfiguration = config;
    }
    return isConfigurationApplied;
}

bool Configure(Configuration config)
{
    return Configure(config, DEVICE_NODE);
}

std::vector<ProfilingEntry> ReportNewProfilingData()
{
    std::vector<ProfilingEntry> res(std::move(g_ProfilingEntries));

    g_ProfilingEntries.clear();
    return res;
}

uint64_t GetCounterValue(PollCounterName counter, const std::string& device)
{
    if (!g_CurrentConfiguration.m_EnableProfiling)
    {
        return 0;
    }

    switch (counter)
    {
        case PollCounterName::DriverLibraryNumLiveBuffers:
            return g_BufferToLifetimeEventId.size();
        case PollCounterName::DriverLibraryNumLiveInferences:
            return g_InferenceToLifetimeEventId.size();
        case PollCounterName::KernelDriverNumMailboxMessagesSent:    // Deliberate fallthrough
        case PollCounterName::KernelDriverNumMailboxMessagesReceived:
        case PollCounterName::KernelDriverNumRuntimePowerSuspend:
        case PollCounterName::KernelDriverNumRuntimePowerResume:
        case PollCounterName::KernelDriverNumPowerSuspend:
        case PollCounterName::KernelDriverNumPowerResume:
            return GetKernelDriverCounterValue(counter, device);
        default:
            ETHOSN_FAIL_MSG("Invalid counter");
            return 0;
    }
}

uint64_t GetCounterValue(PollCounterName counter)
{
    return GetCounterValue(counter, DEVICE_NODE);
}

const char* EntryTypeToCString(ProfilingEntry::Type type)
{
    switch (type)
    {
        case ProfilingEntry::Type::TimelineEventStart:
            return "TimelineEventStart";
        case ProfilingEntry::Type::TimelineEventEnd:
            return "TimelineEventEnd";
        case ProfilingEntry::Type::TimelineEventInstant:
            return "TimelineEventInstant";
        case ProfilingEntry::Type::CounterSample:
            return "CounterSample";
        default:
            return nullptr;
    }
}

const char* CollatedCounterNameToCString(CollatedCounterName counterName)
{
    switch (counterName)
    {
        case CollatedCounterName::FirmwareDwtSleepCycleCount:
            return "FirmwareDwtSleepCycleCount";
        case CollatedCounterName::FirmwareEventQueueSize:
            return "FirmwareEventQueueSize";
        case CollatedCounterName::FirmwareDmaNumReads:
            return "FirmwareDmaNumReads";
        case CollatedCounterName::FirmwareDmaNumWrites:
            return "FirmwareDmaNumWrites";
        case CollatedCounterName::FirmwareDmaReadBytes:
            return "FirmwareDmaReadBytes";
        case CollatedCounterName::FirmwareDmaWriteBytes:
            return "FirmwareDmaWriteBytes";
        case CollatedCounterName::FirmwareBusAccessRdTransfers:
            return "FirmwareBusAccessRdTransfers";
        case CollatedCounterName::FirmwareBusRdCompleteTransfers:
            return "FirmwareBusRdCompleteTransfers";
        case CollatedCounterName::FirmwareBusReadBeats:
            return "FirmwareBusReadBeats";
        case CollatedCounterName::FirmwareBusReadTxfrStallCycles:
            return "FirmwareBusReadTxfrStallCycles";
        case CollatedCounterName::FirmwareBusAccessWrTransfers:
            return "FirmwareBusAccessWrTransfers";
        case CollatedCounterName::FirmwareBusWrCompleteTransfers:
            return "FirmwareBusWrCompleteTransfers";
        case CollatedCounterName::FirmwareBusWriteBeats:
            return "FirmwareBusWriteBeats";
        case CollatedCounterName::FirmwareBusWriteTxfrStallCycles:
            return "FirmwareBusWriteTxfrStallCycles";
        case CollatedCounterName::FirmwareBusWriteStallCycles:
            return "FirmwareBusWriteStallCycles";
        case CollatedCounterName::FirmwareBusErrorCount:
            return "FirmwareBusErrorCount";
        case CollatedCounterName::FirmwareNcuMcuIcacheMiss:
            return "FirmwareNcuMcuIcacheMiss";
        case CollatedCounterName::FirmwareNcuMcuDcacheMiss:
            return "FirmwareNcuMcuDcacheMiss";
        case CollatedCounterName::FirmwareNcuMcuBusReadBeats:
            return "FirmwareNcuMcuBusReadBeats";
        case CollatedCounterName::FirmwareNcuMcuBusWriteBeats:
            return "FirmwareNcuMcuBusWriteBeats";
        default:
            return nullptr;
    }
}

const char* PollCounterNameToCString(PollCounterName counterName)
{
    switch (counterName)
    {
        case PollCounterName::DriverLibraryNumLiveBuffers:
            return "DriverLibraryNumLiveBuffers";
        case PollCounterName::DriverLibraryNumLiveInferences:
            return "DriverLibraryNumLiveInferences";
        case PollCounterName::KernelDriverNumMailboxMessagesSent:
            return "KernelDriverNumMailboxMessagesSent";
        case PollCounterName::KernelDriverNumMailboxMessagesReceived:
            return "KernelDriverNumMailboxMessagesReceived";
        case PollCounterName::KernelDriverNumRuntimePowerSuspend:
            return "KernelDriverNumRuntimePowerSuspend";
        case PollCounterName::KernelDriverNumRuntimePowerResume:
            return "KernelDriverNumRuntimePowerResume";
        case PollCounterName::KernelDriverNumPowerSuspend:
            return "KernelDriverNumPowerSuspend";
        case PollCounterName::KernelDriverNumPowerResume:
            return "KernelDriverNumPowerResume";
        default:
            return nullptr;
    }
}

const char* MetadataCategoryToCString(ProfilingEntry::MetadataCategory category)
{
    switch (category)
    {
        case ProfilingEntry::MetadataCategory::FirmwareInference:
            return "FirmwareInference";
        case ProfilingEntry::MetadataCategory::FirmwareUpdateProgress:
            return "FirmwareUpdateProgress";
        case ProfilingEntry::MetadataCategory::FirmwareWfe:
            return "FirmwareWfe";
        case ProfilingEntry::MetadataCategory::FirmwareDmaReadSetup:
            return "FirmwareDmaReadSetup";
        case ProfilingEntry::MetadataCategory::FirmwareDmaRead:
            return "FirmwareDmaRead";
        case ProfilingEntry::MetadataCategory::FirmwareDmaWriteSetup:
            return "FirmwareDmaWriteSetup";
        case ProfilingEntry::MetadataCategory::FirmwareDmaWrite:
            return "FirmwareDmaWrite";
        case ProfilingEntry::MetadataCategory::FirmwareMceStripeSetup:
            return "FirmwareMceStripeSetup";
        case ProfilingEntry::MetadataCategory::FirmwareMceStripe:
            return "FirmwareMceStripe";
        case ProfilingEntry::MetadataCategory::FirmwarePleStripeSetup:
            return "FirmwarePleStripeSetup";
        case ProfilingEntry::MetadataCategory::FirmwarePleStripe:
            return "FirmwarePleStripe";
        case ProfilingEntry::MetadataCategory::FirmwareUdma:
            return "FirmwareUdma";
        case ProfilingEntry::MetadataCategory::FirmwareLabel:
            return "FirmwareLabel";
        case ProfilingEntry::MetadataCategory::InferenceLifetime:
            return "InferenceLifetime";
        case ProfilingEntry::MetadataCategory::BufferLifetime:
            return "BufferLifetime";
        case ProfilingEntry::MetadataCategory::CounterValue:
            return "CounterValue";
        default:
            return nullptr;
    }
}

ethosn_profiling_hw_counter_types ConvertHwCountersToKernel(HardwareCounters counter)
{
    switch (counter)
    {
        case HardwareCounters::FirmwareBusAccessRdTransfers:
            return ethosn_profiling_hw_counter_types::BUS_ACCESS_RD_TRANSFERS;
        case HardwareCounters::FirmwareBusRdCompleteTransfers:
            return ethosn_profiling_hw_counter_types::BUS_RD_COMPLETE_TRANSFERS;
        case HardwareCounters::FirmwareBusReadBeats:
            return ethosn_profiling_hw_counter_types::BUS_READ_BEATS;
        case HardwareCounters::FirmwareBusReadTxfrStallCycles:
            return ethosn_profiling_hw_counter_types::BUS_READ_TXFR_STALL_CYCLES;
        case HardwareCounters::FirmwareBusAccessWrTransfers:
            return ethosn_profiling_hw_counter_types::BUS_ACCESS_WR_TRANSFERS;
        case HardwareCounters::FirmwareBusWrCompleteTransfers:
            return ethosn_profiling_hw_counter_types::BUS_WR_COMPLETE_TRANSFERS;
        case HardwareCounters::FirmwareBusWriteBeats:
            return ethosn_profiling_hw_counter_types::BUS_WRITE_BEATS;
        case HardwareCounters::FirmwareBusWriteTxfrStallCycles:
            return ethosn_profiling_hw_counter_types::BUS_WRITE_TXFR_STALL_CYCLES;
        case HardwareCounters::FirmwareBusWriteStallCycles:
            return ethosn_profiling_hw_counter_types::BUS_WRITE_STALL_CYCLES;
        case HardwareCounters::FirmwareBusErrorCount:
            return ethosn_profiling_hw_counter_types::BUS_ERROR_COUNT;
        case HardwareCounters::FirmwareNcuMcuIcacheMiss:
            return ethosn_profiling_hw_counter_types::NCU_MCU_ICACHE_MISS;
        case HardwareCounters::FirmwareNcuMcuDcacheMiss:
            return ethosn_profiling_hw_counter_types::NCU_MCU_DCACHE_MISS;
        case HardwareCounters::FirmwareNcuMcuBusReadBeats:
            return ethosn_profiling_hw_counter_types::NCU_MCU_BUS_READ_BEATS;
        case HardwareCounters::FirmwareNcuMcuBusWriteBeats:
            return ethosn_profiling_hw_counter_types::NCU_MCU_BUS_WRITE_BEATS;
        default:
            throw std::runtime_error("ethosn_profiling_hw_counter_types not in sync with HardwareCounters");
    }
}

using EntryId   = decltype(ethosn_profiling_entry::id);
using EntryData = decltype(ethosn_profiling_entry::data);

uint64_t GetIdForCounterValue(EntryId id)
{
    // Convert ID (which is this case is the counter name)
    switch (static_cast<FirmwareCounterName>(id))
    {
        case FirmwareCounterName::DwtSleepCycleCount:
            return static_cast<uint64_t>(CollatedCounterName::FirmwareDwtSleepCycleCount);
        case FirmwareCounterName::EventQueueSize:
            return static_cast<uint64_t>(CollatedCounterName::FirmwareEventQueueSize);
        case FirmwareCounterName::DmaNumReads:
            return static_cast<uint64_t>(CollatedCounterName::FirmwareDmaNumReads);
        case FirmwareCounterName::DmaNumWrites:
            return static_cast<uint64_t>(CollatedCounterName::FirmwareDmaNumWrites);
        case FirmwareCounterName::DmaReadBytes:
            return static_cast<uint64_t>(CollatedCounterName::FirmwareDmaReadBytes);
        case FirmwareCounterName::DmaWriteBytes:
            return static_cast<uint64_t>(CollatedCounterName::FirmwareDmaWriteBytes);
        case FirmwareCounterName::BusAccessRdTransfers:
            return static_cast<uint64_t>(CollatedCounterName::FirmwareBusAccessRdTransfers);
        case FirmwareCounterName::BusRdCompleteTransfers:
            return static_cast<uint64_t>(CollatedCounterName::FirmwareBusRdCompleteTransfers);
        case FirmwareCounterName::BusReadBeats:
            return static_cast<uint64_t>(CollatedCounterName::FirmwareBusReadBeats);
        case FirmwareCounterName::BusReadTxfrStallCycles:
            return static_cast<uint64_t>(CollatedCounterName::FirmwareBusReadTxfrStallCycles);
        case FirmwareCounterName::BusAccessWrTransfers:
            return static_cast<uint64_t>(CollatedCounterName::FirmwareBusAccessWrTransfers);
        case FirmwareCounterName::BusWrCompleteTransfers:
            return static_cast<uint64_t>(CollatedCounterName::FirmwareBusWrCompleteTransfers);
        case FirmwareCounterName::BusWriteBeats:
            return static_cast<uint64_t>(CollatedCounterName::FirmwareBusWriteBeats);
        case FirmwareCounterName::BusWriteTxfrStallCycles:
            return static_cast<uint64_t>(CollatedCounterName::FirmwareBusWriteTxfrStallCycles);
        case FirmwareCounterName::BusWriteStallCycles:
            return static_cast<uint64_t>(CollatedCounterName::FirmwareBusWriteStallCycles);
        case FirmwareCounterName::BusErrorCount:
            return static_cast<uint64_t>(CollatedCounterName::FirmwareBusErrorCount);
        case FirmwareCounterName::NcuMcuIcacheMiss:
            return static_cast<uint64_t>(CollatedCounterName::FirmwareNcuMcuIcacheMiss);
        case FirmwareCounterName::NcuMcuDcacheMiss:
            return static_cast<uint64_t>(CollatedCounterName::FirmwareNcuMcuDcacheMiss);
        case FirmwareCounterName::NcuMcuBusReadBeats:
            return static_cast<uint64_t>(CollatedCounterName::FirmwareNcuMcuBusReadBeats);
        case FirmwareCounterName::NcuMcuBusWriteBeats:
            return static_cast<uint64_t>(CollatedCounterName::FirmwareNcuMcuBusWriteBeats);
        default:
            throw std::runtime_error("Unknown counter with ID " + std::to_string(id));
    }
}

ProfilingEntry::MetadataCategory ConvertTimelineEventToMetadataCategory(TimelineEventType timelineEventType)
{
    switch (timelineEventType)
    {
        case TimelineEventType::Inference:
            return ProfilingEntry::MetadataCategory::FirmwareInference;
        case TimelineEventType::UpdateProgress:
            return ProfilingEntry::MetadataCategory::FirmwareUpdateProgress;
        case TimelineEventType::Wfe:
            return ProfilingEntry::MetadataCategory::FirmwareWfe;
        case TimelineEventType::DmaReadSetup:
            return ProfilingEntry::MetadataCategory::FirmwareDmaReadSetup;
        case TimelineEventType::DmaRead:
            return ProfilingEntry::MetadataCategory::FirmwareDmaRead;
        case TimelineEventType::DmaWriteSetup:
            return ProfilingEntry::MetadataCategory::FirmwareDmaWriteSetup;
        case TimelineEventType::DmaWrite:
            return ProfilingEntry::MetadataCategory::FirmwareDmaWrite;
        case TimelineEventType::MceStripeSetup:
            return ProfilingEntry::MetadataCategory::FirmwareMceStripeSetup;
        case TimelineEventType::MceStripe:
            return ProfilingEntry::MetadataCategory::FirmwareMceStripe;
        case TimelineEventType::PleStripeSetup:
            return ProfilingEntry::MetadataCategory::FirmwarePleStripeSetup;
        case TimelineEventType::PleStripe:
            return ProfilingEntry::MetadataCategory::FirmwarePleStripe;
        case TimelineEventType::Udma:
            return ProfilingEntry::MetadataCategory::FirmwareUdma;
        case TimelineEventType::Label:
            return ProfilingEntry::MetadataCategory::FirmwareLabel;
        default:
            throw std::runtime_error("Unknown timeline event type with value " +
                                     std::to_string(static_cast<uint32_t>(timelineEventType)));
    }
}

// Converts a profiling entry reported by the kernel into the Driver Library's public ProfilingEntry representation.
std::pair<bool, ProfilingEntry> ConvertProfilingEntry(const ethosn_profiling_entry& kernelEntry,
                                                      std::map<uint8_t, ProfilingEntry>& inProgressTimelineEvents,
                                                      uint64_t& mostRecentCorrectedKernelTimestamp,
                                                      int clockFrequencyMhz,
                                                      uint64_t nanosecondOffset)
{
    ProfilingEntry result = {};

    TimelineEntryDataUnion dataUnion;
    dataUnion.m_Raw = kernelEntry.data;

    // Convert the timestamp reported from the kernel/firmware, into a wall clock time to report
    // in the public API. This needs to account for the clock frequency and offset of the timestamps
    // from the firmware (they measure in clock cycles, not seconds) and also potential wraparound
    // of the 32-bit timestamp field.

    if (kernelEntry.type == ethosn_profiling_entry_type::TIMELINE_EVENT_INSTANT &&
        static_cast<TimelineEventType>(dataUnion.m_Type) == TimelineEventType::TimestampFull)
    {
        // If we were given a full timestamp field then we don't need to account for wraparound.
        // These are sent at the start of an inference to make sure we don't miss any time between
        // configuring profiling and the start of an inference.
        mostRecentCorrectedKernelTimestamp =
            static_cast<uint64_t>(kernelEntry.timestamp) |
            (static_cast<uint64_t>(dataUnion.m_TimestampFullFields.m_TimestampUpperBits) << 32U);
        // We don't actually convert the rest of this entry as it has no further use now that
        // we have updated the timestamp to use for converting future entries
        return { false, {} };
    }

    // Account for timestamp overflow, assuming that at most a single overflow occured.
    // This should be sufficient for entries during an inference because they will be quite close together.
    // For larger gaps though we may incorrectly "skip" time, which is why the firmware sends a TimestampFull
    // entry (see above) at the start of an inference.
    const uint32_t diff =
        static_cast<uint32_t>(kernelEntry.timestamp - mostRecentCorrectedKernelTimestamp % UINT32_MAX);
    uint64_t overflowCorrectedKernelTimestamp = mostRecentCorrectedKernelTimestamp + diff;

    // Remember this corrected timestamp for the next entry we convert, so that we can correctly
    // correct that timestamp too.
    mostRecentCorrectedKernelTimestamp = overflowCorrectedKernelTimestamp;

    // Now we account for the different clock frequency and offset
    result.m_Timestamp = std::chrono::time_point<std::chrono::high_resolution_clock>(
        std::chrono::nanoseconds((1000 / clockFrequencyMhz) * overflowCorrectedKernelTimestamp + nanosecondOffset));

    switch (kernelEntry.type)
    {
        case ethosn_profiling_entry_type::COUNTER_VALUE:
            result.m_Id               = GetIdForCounterValue(kernelEntry.id);
            result.m_Type             = ProfilingEntry::Type::CounterSample;
            result.m_MetadataCategory = ProfilingEntry::MetadataCategory::CounterValue;
            result.m_MetadataValue    = kernelEntry.data;
            break;
        case ethosn_profiling_entry_type::TIMELINE_EVENT_START:
        {
            // Rather than using the ID from the kernel entry which is only short and will re-use values,
            // assign a new unique ID to make later processing simpler.
            result.m_Id   = GetNextTimelineEventId();
            result.m_Type = ProfilingEntry::Type::TimelineEventStart;
            result.m_MetadataCategory =
                ConvertTimelineEventToMetadataCategory(static_cast<TimelineEventType>(dataUnion.m_Type));

            // Remember that this event is in flight, so we can match it up with the end
            // event and assign the same ID to it
            inProgressTimelineEvents[kernelEntry.id] = result;
        }
        break;
        case ethosn_profiling_entry_type::TIMELINE_EVENT_END:
        {
            // Find the corresponding start event, so that we can use the same ID (we re-map the IDs)
            auto startEntryIt = inProgressTimelineEvents.find(kernelEntry.id);
            if (startEntryIt == inProgressTimelineEvents.end())
            {
                g_Logger.Warning("Profiling TIMELINE_EVENT_END entry has no corresponding start event - skipping");
                return { false, {} };
            }

            result.m_Id   = startEntryIt->second.m_Id;
            result.m_Type = ProfilingEntry::Type::TimelineEventEnd;
            // Also copy the metadata from the start event for convenience
            // (the end event from the firmware won't have anything here)
            result.m_MetadataCategory = startEntryIt->second.m_MetadataCategory;
            result.m_MetadataValue    = startEntryIt->second.m_MetadataValue;

            inProgressTimelineEvents.erase(startEntryIt);
        }
        break;
        case ethosn_profiling_entry_type::TIMELINE_EVENT_INSTANT:
        {
            // The ID from the kernel entry won't be set as it isn't needed,
            // so we assign a new unique ID to make further processing simpler.
            result.m_Id   = GetNextTimelineEventId();
            result.m_Type = ProfilingEntry::Type::TimelineEventInstant;
            result.m_MetadataCategory =
                ConvertTimelineEventToMetadataCategory(static_cast<TimelineEventType>(dataUnion.m_Type));
            if (result.m_MetadataCategory == ProfilingEntry::MetadataCategory::FirmwareLabel)
            {
                // Convert the label and store into driver library metadata field.
                // It can be decoded from the public API using GetFirmwareLabel
                result.m_MetadataValue = dataUnion.m_LabelFields.m_Char1 | dataUnion.m_LabelFields.m_Char2 << 8 |
                                         dataUnion.m_LabelFields.m_Char3 << 16;
            }
        }
        break;
        default:
            throw std::runtime_error(std::string("Invalid profiling entry type from kernel"));
            break;
    }
    return { true, result };
}

}    // namespace profiling
}    // namespace driver_library
}    // namespace ethosn
