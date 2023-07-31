//
// Copyright © 2019-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "../include/ethosn_driver_library/Profiling.hpp"

#include <uapi/ethosn_shared.h>

#include <chrono>
#include <map>
#include <string>
#include <vector>

namespace ethosn
{
namespace driver_library
{

class Buffer;
class Inference;

namespace profiling
{

namespace metadata
{

/// These functions encode the metadata values.
/// They should match the decoding functions in ProfilingMetadataImpl.hpp.

inline uint64_t CreateCounterValue(uint64_t counterValue)
{
    // The counter value is stored verbatim in the metadata value
    return counterValue;
}

}    // namespace metadata

Configuration GetConfigFromString(const char* str);

extern Configuration g_CurrentConfiguration;
extern std::vector<ProfilingEntry> g_ProfilingEntries;
extern uint64_t g_NextTimelineEventId;

extern std::map<Buffer*, uint64_t> g_BufferToLifetimeEventId;
extern std::map<Inference*, uint64_t> g_InferenceToLifetimeEventId;

/// If set, automatically dump profiling entries and counters to this file after each inference.
/// Set by the environment variable parsed in GetDefaultConfiguration().
extern std::string g_DumpFile;

/// ProfilingInternal functions
/// @{
uint64_t GetNextTimelineEventId();
template <typename T>
void RecordLifetimeEvent(T* object,
                         std::map<T*, uint64_t>& objectToLifetimeEventId,
                         profiling::ProfilingEntry::Type type,
                         profiling::ProfilingEntry::MetadataCategory category)
{
    using namespace std::chrono;
    using namespace profiling;
    time_point<high_resolution_clock> timestamp = high_resolution_clock::now();
    ProfilingEntry entry;
    entry.m_Timestamp = timestamp;
    entry.m_Type      = type;
    uint64_t id;
    if (type == ProfilingEntry::Type::TimelineEventStart)
    {
        id                              = g_NextTimelineEventId;
        objectToLifetimeEventId[object] = id;
        GetNextTimelineEventId();
    }
    else
    {
        assert(type == ProfilingEntry::Type::TimelineEventEnd);
        auto it = objectToLifetimeEventId.find(object);
        if (it == objectToLifetimeEventId.end())
        {
            // If the profiling was enabled after creating this object then no event should be registered.
            return;
        }
        id = it->second;
        objectToLifetimeEventId.erase(it);
    }
    entry.m_Id               = id;
    entry.m_MetadataCategory = category;
    entry.m_MetadataValue    = 0;
    g_ProfilingEntries.push_back(entry);
}
/// @}

/// Implemented by the backend (model, kernel module etc.)
/// in either KmodProfiling.cpp or NullKmodProfiling.cpp.
/// @{
bool ConfigureKernelDriver(Configuration config, const std::string& device);
uint64_t GetKernelDriverCounterValue(PollCounterName counter, const std::string& device);
/// Append all entries reported by the kernel driver to the given vector.
bool AppendKernelDriverEntries();
/// @}

ethosn_profiling_hw_counter_types ConvertHwCountersToKernel(HardwareCounters counter);

std::pair<bool, ProfilingEntry> ConvertProfilingEntry(const ethosn_profiling_entry& kernelEntry,
                                                      std::map<uint8_t, ProfilingEntry>& inProgressTimelineEvents,
                                                      uint64_t& mostRecentCorrectedKernelTimestamp,
                                                      int clockFrequencyMhz,
                                                      uint64_t nanosecondOffset);

}    // namespace profiling
}    // namespace driver_library
}    // namespace ethosn
