//
// Copyright © 2019-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "ProfilingMetadataImpl.hpp"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace ethosn
{
namespace driver_library
{
namespace profiling
{

/// A set of counters which only a maximum of 6 can be activated at once
enum class HardwareCounters
{
    FirmwareBusAccessRdTransfers,
    FirmwareBusRdCompleteTransfers,
    FirmwareBusReadBeats,
    FirmwareBusReadTxfrStallCycles,
    FirmwareBusAccessWrTransfers,
    FirmwareBusWrCompleteTransfers,
    FirmwareBusWriteBeats,
    FirmwareBusWriteTxfrStallCycles,
    FirmwareBusWriteStallCycles,
    FirmwareBusErrorCount,
    FirmwareNcuMcuIcacheMiss,
    FirmwareNcuMcuDcacheMiss,
    FirmwareNcuMcuBusReadBeats,
    FirmwareNcuMcuBusWriteBeats,
    NumValues,
};

/// Global profiling options which can be passed to Configure(...).
struct Configuration
{
    bool m_EnableProfiling         = false;
    uint32_t m_FirmwareBufferSize  = 0;
    uint32_t m_NumHardwareCounters = 0;
    HardwareCounters m_HardwareCounters[6];
};

/// Re-configures the profiling options for the ethosn driver stack based on the given Configuration object.
bool Configure(Configuration config);
bool Configure(Configuration config, const std::string& device);

/// All the counters that can be requested using Configure(...) and ScheduleInference(...)
/// and collected using ReportNewProfilingData(...).
/// These counters cannot be polled using GetCounterValue().
enum class CollatedCounterName
{
    FirmwareDwtSleepCycleCount,
    FirmwareEventQueueSize,
    FirmwareDmaNumReads,
    FirmwareDmaNumWrites,
    FirmwareDmaReadBytes,
    FirmwareDmaWriteBytes,
    FirmwareBusAccessRdTransfers,
    FirmwareBusRdCompleteTransfers,
    FirmwareBusReadBeats,
    FirmwareBusReadTxfrStallCycles,
    FirmwareBusAccessWrTransfers,
    FirmwareBusWrCompleteTransfers,
    FirmwareBusWriteBeats,
    FirmwareBusWriteTxfrStallCycles,
    FirmwareBusWriteStallCycles,
    FirmwareBusErrorCount,
    FirmwareNcuMcuIcacheMiss,
    FirmwareNcuMcuDcacheMiss,
    FirmwareNcuMcuBusReadBeats,
    FirmwareNcuMcuBusWriteBeats,
    /// The number of counter types in this enum.
    NumValues,
};

/// All the counters that can be polled using GetCounterValue(...).
/// Note that this does not include any counters that are collated and retrieved
/// later (e.g. those from the Control Unit) as these counters cannot be polled directly.
/// See the CollatedCounterName enum for details of these.
enum class PollCounterName
{
    /// The number of currently live instances of the Buffer class.
    DriverLibraryNumLiveBuffers = static_cast<int>(CollatedCounterName::NumValues),
    /// The number of currently live instances of the Inference class.
    DriverLibraryNumLiveInferences,

    /// The number of mailbox messages sent by the kernel driver.
    KernelDriverNumMailboxMessagesSent,
    /// The number of mailbox messages received by the kernel driver.
    KernelDriverNumMailboxMessagesReceived,

    /// The number of times that device goes into runtime suspend state.
    KernelDriverNumRuntimePowerSuspend,
    /// The number of times that device goes into runtime resume state.
    KernelDriverNumRuntimePowerResume,

    /// The number of times that device goes into suspend state.
    KernelDriverNumPowerSuspend,
    /// The number of times that device goes into resume state.
    KernelDriverNumPowerResume,

    /// The number of counter types in this enum.
    NumValues,
};

/// Queries the current value of the given profiling counter.
/// If the appropriate profiling options for the requested counter have not been enabled via Configure(...)
/// then the result is undefined.
/// This function is thread-safe.
uint64_t GetCounterValue(PollCounterName counter);
uint64_t GetCounterValue(PollCounterName counter, const std::string& device);

/// A single entry in the vector returned by ReportNewProfilingData.
/// This can represent a timeline event or a counter sample.
/// It contains a timestamp, the type of event, the Id of the event and the metadata associated with the event
/// The metadata is stored as a tagged union with type m_MetadataCategory and the data is accessible through the accessor methods.
class ProfilingEntry
{
public:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_Timestamp;

    enum class Type
    {
        /// The start of an event with duration, for example the start of
        /// a DMA transaction. A corresponding event with type = End and the same id is expected.
        TimelineEventStart,
        /// The end of an event with duration, for example the end of a DMA
        /// transaction. A corresponding event with type = Start and the same id is expected.
        TimelineEventEnd,
        /// An instantaneous event, for example an interrupt being received.
        TimelineEventInstant,
        /// A sample of a counter.
        CounterSample
    };
    Type m_Type;

    /// For timeline entries, this uniquely identifies which event this entry refers to.
    /// Multiple entries can have the same id in order to group related timeline entries
    /// (e.g. a single timeline event can have both a TimelineEventStart entry and a TimelineEventEnd entry).
    /// For counter value entries, this uniquely identifies which of the counters this entry is reporting
    /// a value for, and corresponds to an enumerator of CollatedCounterName.
    uint64_t m_Id;

    /// Determines the type of data in the m_MetadataValue field.
    /// Applicable values of this enum depend on the m_Type field (e.g. the metadata type for CounterSample entries
    /// will always be CounterValue).
    /// @{
    enum class MetadataCategory
    {
        FirmwareInference,
        FirmwareUpdateProgress,
        FirmwareWfe,
        FirmwareDmaReadSetup,
        FirmwareDmaRead,
        FirmwareDmaWriteSetup,
        FirmwareDmaWrite,
        FirmwareMceStripeSetup,
        FirmwareMceStripe,
        FirmwarePleStripeSetup,
        FirmwarePleStripe,
        FirmwareUdma,
        FirmwareLabel,

        // Non-firmware related categories go here.
        InferenceLifetime,
        BufferLifetime,
        CounterValue,
    };
    MetadataCategory m_MetadataCategory;
    /// @}

    /// Additional data for this entry, the contents of which are determined by m_MetadataCategory and can be decoded
    /// via the below accessor methods.
    /// For CounterSample entries, this will contain the counter value itself (see GetCounterValue()).
    /// For timeline event entries, this may contain further details of what the timeline event represents,
    /// for example identifying a command number or stripe index.
    uint64_t m_MetadataValue;

    /// Functions to retrieve metadata. Only some of these will be applicable based on the metadata category
    /// (m_MetadataCategory).
    /// E.g. BufferLifetime doesn't have any metadata and GetCounterValue is only applicable with CounterValue
    /// Attempting to access metadata values which aren't applicable to the metadata category is undefined.
    /// @{
    uint64_t GetCounterValue() const
    {
        assert(m_MetadataCategory == MetadataCategory::CounterValue);
        return impl::GetCounterValue(m_MetadataValue);
    }
    std::string GetFirmwareLabel() const
    {
        assert(m_MetadataCategory == MetadataCategory::FirmwareLabel);
        return impl::GetFirmwareLabel(m_MetadataValue);
    }
    /// @}
};

std::vector<ProfilingEntry> ReportNewProfilingData();

const char* EntryTypeToCString(ProfilingEntry::Type type);
const char* CollatedCounterNameToCString(CollatedCounterName counterName);
const char* PollCounterNameToCString(PollCounterName counterName);
const char* MetadataCategoryToCString(ProfilingEntry::MetadataCategory category);

}    // namespace profiling
}    // namespace driver_library
}    // namespace ethosn
