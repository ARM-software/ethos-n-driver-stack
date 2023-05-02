//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "Profiling.hpp"

using namespace ethosn::control_unit::utils;

namespace ethosn
{
namespace control_unit
{
namespace profiling
{

namespace
{
FirmwareCounterName HwCounterTypeToCounterName(ethosn_profiling_hw_counter_types hwCounter)
{
    switch (hwCounter)
    {
        case ethosn_profiling_hw_counter_types::BUS_ACCESS_RD_TRANSFERS:
        {
            return FirmwareCounterName::BusAccessRdTransfers;
        }
        case ethosn_profiling_hw_counter_types::BUS_RD_COMPLETE_TRANSFERS:
        {
            return FirmwareCounterName::BusRdCompleteTransfers;
        }
        case ethosn_profiling_hw_counter_types::BUS_READ_BEATS:
        {
            return FirmwareCounterName::BusReadBeats;
        }
        case ethosn_profiling_hw_counter_types::BUS_READ_TXFR_STALL_CYCLES:
        {
            return FirmwareCounterName::BusReadTxfrStallCycles;
        }
        case ethosn_profiling_hw_counter_types::BUS_ACCESS_WR_TRANSFERS:
        {
            return FirmwareCounterName::BusAccessWrTransfers;
        }
        case ethosn_profiling_hw_counter_types::BUS_WR_COMPLETE_TRANSFERS:
        {
            return FirmwareCounterName::BusWrCompleteTransfers;
        }
        case ethosn_profiling_hw_counter_types::BUS_WRITE_BEATS:
        {
            return FirmwareCounterName::BusWriteBeats;
        }
        case ethosn_profiling_hw_counter_types::BUS_WRITE_TXFR_STALL_CYCLES:
        {
            return FirmwareCounterName::BusWriteTxfrStallCycles;
        }
        case ethosn_profiling_hw_counter_types::BUS_WRITE_STALL_CYCLES:
        {
            return FirmwareCounterName::BusWriteStallCycles;
        }
        case ethosn_profiling_hw_counter_types::BUS_ERROR_COUNT:
        {
            return FirmwareCounterName::BusErrorCount;
        }
        case ethosn_profiling_hw_counter_types::NCU_MCU_ICACHE_MISS:
        {
            return FirmwareCounterName::NcuMcuIcacheMiss;
        }
        case ethosn_profiling_hw_counter_types::NCU_MCU_DCACHE_MISS:
        {
            return FirmwareCounterName::NcuMcuDcacheMiss;
        }
        case ethosn_profiling_hw_counter_types::NCU_MCU_BUS_READ_BEATS:
        {
            return FirmwareCounterName::NcuMcuBusReadBeats;
        }
        case ethosn_profiling_hw_counter_types::NCU_MCU_BUS_WRITE_BEATS:
        {
            return FirmwareCounterName::NcuMcuBusWriteBeats;
        }
        default:
        {
            ASSERT_MSG(
                false,
                "HwCounterTypeToCounterName: Cannot map ethosn_profiling_hw_counter_types to FirmwareCounterName");
            return static_cast<FirmwareCounterName>(0);
        }
    }
}
}    // namespace

template <typename Hal>
ProfilingDataImpl<Hal>::ProfilingDataImpl(Pmu<Hal>& pmu)
    : m_Pmu(pmu)
{
    Reset();
}

template <typename Hal>
bool ProfilingDataImpl<Hal>::IsEnabled() const
{
    return m_Config.enable_profiling;
}

template <typename Hal>
void ProfilingDataImpl<Hal>::Reset()
{
    m_Config                          = {};
    m_WriteIndex                      = 0;
    m_Buffer                          = nullptr;
    m_BufferEntriesCapacity           = 0;
    m_FreeEntryIds                    = 0xFFFFFFFF;
    m_NumEntriesThisInference         = 0;
    m_NumEntriesThisInferenceOverflow = 0;
}

template <typename Hal>
void ProfilingDataImpl<Hal>::Reset(const ethosn_firmware_profiling_configuration& config)
{
    Reset();

    m_Config = config;
    m_Buffer = reinterpret_cast<ethosn_profiling_buffer*>(config.buffer_address);
    m_BufferEntriesCapacity =
        (config.buffer_size - offsetof(ethosn_profiling_buffer, entries)) / sizeof(m_Buffer->entries[0]);

    m_Pmu.Reset(config.num_hw_counters, &config.hw_counters[0]);
}

template <typename Hal>
void ProfilingDataImpl<Hal>::BeginInference()
{
    m_NumEntriesThisInference         = 0;
    m_NumEntriesThisInferenceOverflow = 0;
}

template <typename Hal>
typename ProfilingDataImpl<Hal>::NumEntriesWritten ProfilingDataImpl<Hal>::EndInference()
{
    NumEntriesWritten result = { m_NumEntriesThisInference, m_NumEntriesThisInferenceOverflow };
    // Reset counters so that we unblock future profiling events that happen outside of an inference.
    m_NumEntriesThisInference         = 0;
    m_NumEntriesThisInferenceOverflow = 0;
    return result;
}

template <typename Hal>
void ProfilingDataImpl<Hal>::Record(ethosn_profiling_entry entry)
{
    if (m_Buffer == nullptr)
    {
        return;
    }
    if (m_NumEntriesThisInference >= m_BufferEntriesCapacity - 1)
    {
        // We assume that the kernel couldn't possibly have read any of the entries we wrote during this inference yet,
        // so we stop writing any more to avoid overwriting earlier ones (the earlier ones are probably more interesting
        // and make it clearer that the buffer has gotten full, and includes the time-sync data).
        // This will be reported in a warning to the user once the inference is finished.
        ++m_NumEntriesThisInferenceOverflow;
        return;
    }

    // Write new entry at the current write pointer
    ethosn_profiling_entry& dest = m_Buffer->entries[m_WriteIndex];
    dest                         = entry;

    // Increment write pointer, wrapping around if we get to the end
    m_WriteIndex = (m_WriteIndex + 1) % static_cast<uint32_t>(m_BufferEntriesCapacity);

    ++m_NumEntriesThisInference;

#if defined CONTROL_UNIT_HARDWARE
    // NOTE: The cache management is done only at the end of the inference to
    // reduce performance degradation. At the end of the inference the
    // write pointer is updated (at the end of Firmware::RunInference)
    // and the privilege firmware must flush and invalidate the data cache anyway.
    // For debugging however it can be useful to flush immediately, to make sure
    // profiling data is readable by the host in case the firmware hangs/crashes
    // during the inference.
    constexpr bool immediateFlush = false;
    if (immediateFlush)
    {
        asm volatile("svc %[svc_num]" ::[svc_num] "i"(ethosn::control_unit::TASK_SVC_DCACHE_CLEAN_INVALIDATE));
    }
#else
    // However when not running on the real hardware, there is no downside to writing it immediately
    // and this avoids us from having to manually flush at the end.
    constexpr bool immediateFlush = true;
#endif

    if (immediateFlush)
    {
        UpdateWritePointer();
    }
}

template <typename Hal>
void ProfilingDataImpl<Hal>::UpdateWritePointer()
{
    if (m_Buffer == nullptr)
    {
        return;
    }

    // Update write pointer
    m_Buffer->firmware_write_index = m_WriteIndex;
}

template <typename Hal>
ethosn_profiling_entry ProfilingDataImpl<Hal>::MakeEntry(ethosn_profiling_entry_type type, uint8_t id, uint32_t data)
{
    ethosn_profiling_entry entry;
    // We only use the low 32-bits of the cycle count register as this is all we have space
    // for in the profiling struct, and it's quicker to read one register than two.
    // Overflow shouldn't be an issue because we send the full timestamp at the start of an inference
    // and 4 billion cycles is BIG network!
    entry.timestamp = m_Pmu.GetCycleCount32();
    entry.type      = ETHOSN_NUMERIC_CAST(type, uint32_t, 2);
    entry.id        = ETHOSN_NUMERIC_CAST(id, uint32_t, 5);
    entry.data      = ETHOSN_NUMERIC_CAST(data, uint32_t, 25);
    return entry;
}

template <typename Hal>
void ProfilingDataImpl<Hal>::RecordTimestampFull()
{
    ethosn_profiling_entry entry;
    // Low 32-bits go into the regular timestamp field
    uint64_t fullTimestamp           = m_Pmu.GetCycleCount64();
    entry.timestamp                  = static_cast<uint32_t>(fullTimestamp);
    entry.type                       = ethosn_profiling_entry_type::TIMELINE_EVENT_INSTANT;
    entry.id                         = 0;    // ID unused for instant entries
    TimelineEntryDataUnion dataUnion = { .m_Raw = 0 };
    dataUnion.m_Type                 = static_cast<uint32_t>(TimelineEventType::TimestampFull);
    // Upper 21 bits go into the payload
    dataUnion.m_TimestampFullFields.m_TimestampUpperBits = ETHOSN_NUMERIC_CAST(fullTimestamp >> 32, uint32_t, 21);
    entry.data                                           = ETHOSN_NUMERIC_CAST(dataUnion.m_Raw, uint32_t, 25);
    Record(entry);
}

template <typename Hal>
uint8_t ProfilingDataImpl<Hal>::RecordStart(TimelineEventType event)
{
    uint8_t id                       = GetFirstFreeEntryId();
    TimelineEntryDataUnion dataUnion = { .m_Raw = 0 };
    dataUnion.m_Type                 = ETHOSN_NUMERIC_CAST(event, uint32_t, 4);
    Record(MakeEntry(ethosn_profiling_entry_type::TIMELINE_EVENT_START, id, dataUnion.m_Raw));
    return id;
}

template <typename Hal>
void ProfilingDataImpl<Hal>::RecordEnd(uint8_t id)
{
    // Note we pass data=0 here, as all data should have been passed in the start event
    Record(MakeEntry(ethosn_profiling_entry_type::TIMELINE_EVENT_END, id, 0));

    // This ID can now be re-used for other events
    MarkEntryIdAsFree(id);
}

template <typename Hal>
void ProfilingDataImpl<Hal>::RecordInstant(TimelineEventType event)
{
    TimelineEntryDataUnion dataUnion = { .m_Raw = 0 };
    dataUnion.m_Type                 = ETHOSN_NUMERIC_CAST(event, uint32_t, 4);
    // ID unused for instant entries
    Record(MakeEntry(ethosn_profiling_entry_type::TIMELINE_EVENT_INSTANT, 0, dataUnion.m_Raw));
}

template <typename Hal>
void ProfilingDataImpl<Hal>::RecordLabel(const char* label)
{
    TimelineEntryDataUnion dataUnion = { .m_Raw = 0 };
    dataUnion.m_Type                 = static_cast<uint32_t>(TimelineEventType::Label);
    dataUnion.m_LabelFields.m_Char1  = 0;
    dataUnion.m_LabelFields.m_Char2  = 0;
    dataUnion.m_LabelFields.m_Char3  = 0;
    if (label[0] != 0)
    {
        dataUnion.m_LabelFields.m_Char1 = ETHOSN_NUMERIC_CAST(static_cast<uint8_t>(label[0]), uint32_t, 7);
        if (label[1] != 0)
        {
            dataUnion.m_LabelFields.m_Char2 = ETHOSN_NUMERIC_CAST(static_cast<uint8_t>(label[1]), uint32_t, 7);
            if (label[2] != 0)
            {
                dataUnion.m_LabelFields.m_Char3 = ETHOSN_NUMERIC_CAST(static_cast<uint8_t>(label[2]), uint32_t, 7);
            }
        }
    }

    // ID unused for instant entries
    Record(MakeEntry(ethosn_profiling_entry_type::TIMELINE_EVENT_INSTANT, 0, dataUnion.m_Raw));
}

template <typename Hal>
void ProfilingDataImpl<Hal>::RecordCounter(FirmwareCounterName counterName, uint32_t counterValue)
{
    // Mask the counter value as it may genuinely overflow and we don't want ETHOSN_NUMERIC_CAST to raise an error
    Record(MakeEntry(ethosn_profiling_entry_type::COUNTER_VALUE, static_cast<uint8_t>(counterName),
                     counterValue & ((1 << 25) - 1)));
}

template <typename Hal>
void ProfilingDataImpl<Hal>::RecordHwCounters()
{
    for (uint32_t i = 0; i < m_Config.num_hw_counters; ++i)
    {
        // Mask the counter value as it may genuinely overflow and we don't want ETHOSN_NUMERIC_CAST to raise an error
        Record(MakeEntry(ethosn_profiling_entry_type::COUNTER_VALUE,
                         static_cast<uint8_t>(HwCounterTypeToCounterName(m_Config.hw_counters[i])),
                         m_Pmu.ReadCounter(i) & ((1 << 25) - 1)));
    }
}

template <typename Hal>
uint8_t ProfilingDataImpl<Hal>::GetFirstFreeEntryId()
{
#if defined(__GNUC__)
    const uint8_t firstSet = static_cast<uint8_t>(__builtin_ffs(static_cast<int>(m_FreeEntryIds)));
#elif defined(_MSC_VER)
    // Emulate the GCC behaviour
    unsigned long index;
    const uint8_t firstSet = _BitScanForward(&index, m_FreeEntryIds) == 0 ? 0 : static_cast<uint8_t>(index) + 1U;
#endif
    // If there are no bits set then firstSet will be set to zero.
    // There's not much we can do in this case, so we just continue assuming everything is OK.
    const uint8_t freeId = static_cast<uint8_t>(firstSet - 1U);

    // Record this ID as used
    m_FreeEntryIds &= ~(1U << freeId);

    return freeId;
}

template <typename Hal>
void ProfilingDataImpl<Hal>::MarkEntryIdAsFree(uint8_t id)
{
    m_FreeEntryIds |= (1U << id);
}

uint32_t GetDwtSleepCycleCount()
{
#if defined(CONTROL_UNIT_HARDWARE) && defined(CONTROL_UNIT_PROFILING)
    register uint32_t count asm("r0");
    asm volatile("svc %[svc_num]" : "=r"(count) : [svc_num] "i"(TASK_SVC_GET_DWT_SLEEP_CYCLE_COUNT));
    return count;
#else
    return 0;
#endif
}

}    // namespace profiling
}    // namespace control_unit
}    // namespace ethosn

// Because we are defining template methods in this cpp file we need to explicitly instantiate all versions
// that we intend to use.
#if defined(CONTROL_UNIT_MODEL)
#include <model/ModelHal.hpp>
template class ethosn::control_unit::profiling::ProfilingDataImpl<ethosn::control_unit::ModelHal>;

#include <model/UscriptHal.hpp>
template class ethosn::control_unit::profiling::ProfilingDataImpl<
    ethosn::control_unit::UscriptHal<ethosn::control_unit::ModelHal>>;

#include <model/LoggingHal.hpp>
template class ethosn::control_unit::profiling::ProfilingDataImpl<ethosn::control_unit::LoggingHal>;
#endif

#if defined(CONTROL_UNIT_HARDWARE)
#include <common/hals/HardwareHal.hpp>
template class ethosn::control_unit::profiling::ProfilingDataImpl<ethosn::control_unit::HardwareHal>;
#endif
