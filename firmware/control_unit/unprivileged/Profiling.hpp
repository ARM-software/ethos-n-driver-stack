//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Pmu.hpp"

#include <common/Containers.hpp>
#include <common/TaskSvc.hpp>

#include <uapi/ethosn_shared.h>

#include <chrono>
#include <cstdint>

namespace ethosn
{
namespace control_unit
{
namespace profiling
{

/// Define empty implementations for profiling types when profiling is disabled.
/// This means client code can be written independently of whether profiling is enabled.
/// @{
template <typename TMasqueradeAs>
struct EmptyStructMasqueradingAsNumericType
{
    constexpr EmptyStructMasqueradingAsNumericType()
    {}
    constexpr EmptyStructMasqueradingAsNumericType(TMasqueradeAs)
    {}

    operator TMasqueradeAs() const
    {
        return TMasqueradeAs();
    }

    EmptyStructMasqueradingAsNumericType& operator+=(const EmptyStructMasqueradingAsNumericType&)
    {
        return *this;
    }
    EmptyStructMasqueradingAsNumericType& operator++()
    {
        return *this;
    }
};

/// Type-wrapper to use when you want a variable only for profiling builds.
/// In profiling builds this maps directly to the given type, but on non-
/// profiling builds it maps to an empty struct for less overhead.
template <typename T>
using ProfilingOnly =
#if defined(CONTROL_UNIT_PROFILING)
    T;
#else
    EmptyStructMasqueradingAsNumericType<T>;
#endif

/// Store of all profiling data collected so far.
template <typename Hal>
class ProfilingDataImpl
{
public:
    ProfilingDataImpl(Pmu<Hal>& pmu);

    bool IsEnabled() const;

    void Reset();
    void Reset(const ethosn_firmware_profiling_configuration& config);

    /// Marks the beginning of a period where profiling will stop writing new entries once it
    /// loops around and catches up with itself. This is used to prevent it overwriting entries
    /// for the same inference, before the driver library has read them.
    void BeginInference();

    struct NumEntriesWritten
    {
        size_t nonOverflow;
        size_t overflow;
    };
    NumEntriesWritten EndInference();

    void Record(ethosn_profiling_entry entry);

    void UpdateWritePointer();

    /// Records the full PMU timestamp value in a profiling entry.
    /// Normal entries only contain a 32-bit timestamp to save space, which means that it can
    /// overflow and we may not be able to reconstruct the full timestamp in the driver library.
    /// By sending a message containing the full timestamp it helps the driver library to
    /// reconstruct the full timestamps without any missing time.
    void RecordTimestampFull();

    /// Records the start of a new profiling event.
    /// Returns the ID of the event to be passed to RecordEnd when you want to record the end.
    uint8_t RecordStart(TimelineEventType event);

    /// Records the end part of a profiling event with the given ID and type.
    /// The event ID provided should be the one returned by the corresponding RecordStart.
    void RecordEnd(uint8_t id);

    /// Records an instantaneous profiling event.
    void RecordInstant(TimelineEventType event);

    /// Records a custom label (up to 3 chars)
    void RecordLabel(const char* label);

    /// Records a sample at the current time for the given counter with the given value.
    void RecordCounter(FirmwareCounterName counterName, uint32_t counterValue);

    /// Records all configured Hardware counters.
    void RecordHwCounters();

private:
    // Handles some of the annoying bitmasking to avoid duplication between the Record functions.
    ethosn_profiling_entry MakeEntry(ethosn_profiling_entry_type type, uint8_t id, uint32_t data);

    uint8_t GetFirstFreeEntryId();
    void MarkEntryIdAsFree(uint8_t id);

    Pmu<Hal>& m_Pmu;

    ethosn_firmware_profiling_configuration m_Config;

    /// Stores all the profiling entries encountered so far.
    ethosn_profiling_buffer* m_Buffer;
    uint32_t m_WriteIndex;
    uint32_t m_NumEntriesThisInference;
    /// Stores if we have overflowed the number of events we can store before a Reset/EndInference,
    /// and by how much.
    uint32_t m_NumEntriesThisInferenceOverflow;
    size_t m_BufferEntriesCapacity;

    /// Entry IDs are re-used once a start/end pair has finished.
    /// This is a bitfield which keeps track of which IDs are available for use.
    uint32_t m_FreeEntryIds;
};

/// Empty implementation to avoid call sites from having to check CONTROL_UNIT_PROFILING.
template <typename Hal>
struct NullProfilingData
{
    NullProfilingData(Pmu<Hal>&)
    {}

    void Reset()
    {}
    void Reset(const ethosn_firmware_profiling_configuration&)
    {}

    bool IsEnabled()
    {
        return false;
    }

    void BeginInference()
    {}

    EmptyStructMasqueradingAsNumericType<typename profiling::ProfilingDataImpl<Hal>::NumEntriesWritten> EndInference()
    {
        return {};
    }

    void RecordTimestampFull()
    {}

    uint8_t RecordStart(TimelineEventType)
    {
        return 0;
    }

    void RecordEnd(uint8_t)
    {}

    void RecordInstant(TimelineEventType)
    {}

    void RecordLabel(const char*)
    {}

    void RecordCounter(FirmwareCounterName, uint32_t)
    {}

    void UpdateWritePointer()
    {}

    void RecordHwCounters()
    {}
};

template <typename Hal>
using ProfilingData =
#if defined(CONTROL_UNIT_PROFILING)
    ProfilingDataImpl<Hal>;
#else
    NullProfilingData<Hal>;
#endif

uint32_t GetDwtSleepCycleCount();

}    // namespace profiling
}    // namespace control_unit
}    // namespace ethosn
