//
// Copyright © 2019-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "ProfilingInternal.hpp"

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

uint64_t GetNextTimeLineEventId()
{
    g_NextTimelineEventId = std::max(g_DriverLibraryEventIdBase, g_NextTimelineEventId + 1);
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
        g_NextTimelineEventId = g_DriverLibraryEventIdBase;
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
                std::cerr << "There can only be at most 6 hardware counters\n";
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
uint64_t g_NextTimelineEventId                              = g_DriverLibraryEventIdBase;

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
            assert(!"Invalid counter");
            return 0;
    }
}

uint64_t GetCounterValue(PollCounterName counter)
{
    return GetCounterValue(counter, DEVICE_NODE);
}

const char* MetadataCategoryToCString(ProfilingEntry::MetadataCategory category)
{
    switch (category)
    {
        case ProfilingEntry::MetadataCategory::FirmwareWfeSleeping:
            return "FirmwareWfeSleeping";
        case ProfilingEntry::MetadataCategory::FirmwareInference:
            return "FirmwareInference";
        case ProfilingEntry::MetadataCategory::FirmwareCommand:
            return "FirmwareCommand";
        case ProfilingEntry::MetadataCategory::FirmwareDma:
            return "FirmwareDma";
        case ProfilingEntry::MetadataCategory::FirmwareTsu:
            return "FirmwareTsu";
        case ProfilingEntry::MetadataCategory::FirmwareMceStripeSetup:
            return "FirmwareMceStripeSetup";
        case ProfilingEntry::MetadataCategory::FirmwarePleStripeSetup:
            return "FirmwarePleStripeSetup";
        case ProfilingEntry::MetadataCategory::FirmwareLabel:
            return "FirmwareLabel";
        case ProfilingEntry::MetadataCategory::FirmwareDmaSetup:
            return "FirmwareDmaSetup";
        case ProfilingEntry::MetadataCategory::FirmwareGetCompleteCommand:
            return "FirmwareGetCompleteCommand";
        case ProfilingEntry::MetadataCategory::FirmwareScheduleNextCommand:
            return "FirmwareScheduleNextCommand";
        case ProfilingEntry::MetadataCategory::FirmwareWfeChecking:
            return "FirmwareWfeChecking";
        case ProfilingEntry::MetadataCategory::FirmwareTimeSync:
            return "FirmwareTimeSync";
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

const char* MetadataTypeToCString(ProfilingEntry::Type type)
{
    switch (type)
    {
        case ProfilingEntry::Type::TimelineEventStart:
            return "Start";
        case ProfilingEntry::Type::TimelineEventEnd:
            return "End";
        case ProfilingEntry::Type::TimelineEventInstant:
            return "Instant";
        case ProfilingEntry::Type::CounterSample:
            return "Counter";
        default:
            return nullptr;
    }
}

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
            assert(!"ethosn_profiling_hw_counter_types not in sync with HardwareCounters");
            return ethosn_profiling_hw_counter_types::NCU_MCU_BUS_WRITE_BEATS;
        }
    }
}

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
            // Set the return value so we don't get errors when asserts are disabled
            retVal = static_cast<uint64_t>(CollatedCounterName::NumValues);
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
        case EntryDataCategory::Agent:
            retVal = ProfilingEntry::MetadataCategory::FirmwareAgent;
            break;
        case EntryDataCategory::AgentStripe:
            retVal = ProfilingEntry::MetadataCategory::FirmwareAgentStripe;
            break;
        default:
            // Set the return value so we don't get errors when asserts are disabled
            retVal = ProfilingEntry::MetadataCategory::FirmwareWfeSleeping;
            assert(false);
            break;
    }
    return retVal;
}

// Converts a profiling entry reported by the kernel into the Driver Library's public ProfilingEntry representation.
ProfilingEntry ConvertProfilingEntry(const ethosn_profiling_entry& kernelEntry)
{
    ProfilingEntry result;
    // Assume for now that the kernel timestamps are in nanoseconds since the high_resolution_clock epoch.
    // This is correct for entries from the model-based firmware, but is wrong (and will be fixed up later) for entries
    // from the hardware-based firmware.
    result.m_Timestamp =
        std::chrono::time_point<std::chrono::high_resolution_clock>(std::chrono::nanoseconds(kernelEntry.timestamp));

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
            throw std::runtime_error(std::string("Invalid profiling entry type from kernel"));
            break;
    }
    return result;
}

}    // namespace profiling
}    // namespace driver_library
}    // namespace ethosn
