/*
 *
 * (C) COPYRIGHT 2020 ARM Limited. All rights reserved.
 *
 * This program is free software and is provided to you under the terms of the
 * GNU General Public License version 2 as published by the Free Software
 * Foundation, and any use by you of this program is subject to the terms
 * of such GNU licence.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, you can access it online at
 * http://www.gnu.org/licenses/gpl-2.0.html.
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 */

/* This header defines interfaces that are shared between driver library,
 * kernel module and firmware.
 */

#ifndef _ETHOSN_SHARED_H_
#define _ETHOSN_SHARED_H_

#ifdef __KERNEL__
#include <linux/types.h>
#else
#include <stdint.h>
#endif

/**
 * enum ethosn_profiling_entry_type - Equivalent to the Driver Library's
 *                                 ProfilingEntry::Type enum.
 */
enum ethosn_profiling_entry_type {
	TIMELINE_EVENT_START,
	TIMELINE_EVENT_END,
	TIMELINE_EVENT_INSTANT,
	COUNTER_VALUE
};

/**
 * struct ethosn_profiling_entry - Equivalent to the Driver Library's
 *				ProfilingEntry struct, with some minor
 *				differences.
 * @timestamp:	Clock cycles as defined by the PMU.
 * @type:	@see ethosn_profiling_entry_type.
 * @id:		See driver_library::ProfilingEntry::m_Id.
 * @data:	Generic data associated with this entry, combining
 *              driver_library::ProfilingEntry::m_MetadataCategory and
 *              driver_library::ProfilingEntry::m_MetadataValue.
 *
 * This struct is designed to be as lightweight as possible, because we will be
 * creating and storing lots of these and we want the profiling overhead to be
 * as small as possible.
 */
struct ethosn_profiling_entry {
	uint64_t timestamp;
	uint16_t type;
	uint16_t id;
	uint32_t data;
};

/**
 * enum ethosn_profiling_hw_counter_types - Equivalent to the Driver Library's
 *                                       HardwareCounters enum.
 */
enum ethosn_profiling_hw_counter_types {
	BUS_ACCESS_RD_TRANSFERS,
	BUS_RD_COMPLETE_TRANSFERS,
	BUS_READ_BEATS,
	BUS_READ_TXFR_STALL_CYCLES,
	BUS_ACCESS_WR_TRANSFERS,
	BUS_WR_COMPLETE_TRANSFERS,
	BUS_WRITE_BEATS,
	BUS_WRITE_TXFR_STALL_CYCLES,
	BUS_WRITE_STALL_CYCLES,
	BUS_ERROR_COUNT,
	NCU_MCU_ICACHE_MISS,
	NCU_MCU_DCACHE_MISS,
	NCU_MCU_BUS_READ_BEATS,
	NCU_MCU_BUS_WRITE_BEATS
};

#if defined(__cplusplus)
enum class FirmwareCounterName {
	DwtSleepCycleCount,
	EventQueueSize,
	DmaNumReads,
	DmaNumWrites,
	DmaReadBytes,
	DmaWriteBytes,
	BusAccessRdTransfers,
	BusRdCompleteTransfers,
	BusReadBeats,
	BusReadTxfrStallCycles,
	BusAccessWrTransfers,
	BusWrCompleteTransfers,
	BusWriteBeats,
	BusWriteTxfrStallCycles,
	BusWriteStallCycles,
	BusErrorCount,
	NcuMcuIcacheMiss,
	NcuMcuDcacheMiss,
	NcuMcuBusReadBeats,
	NcuMcuBusWriteBeats,
};

using EntryData = decltype(ethosn_profiling_entry::data);

/* Note that the order of these categories matters for the python parser that */
/* generates the json file. New categories need to be added at the bottom of */
/* the list. */
enum class EntryDataCategory: uint8_t {
	WfeSleeping,
	Inference,
	Command,
	Dma,
	Tsu,
	MceStripeSetup,
	PleStripeSetup,
	Label,
	DmaSetup,
	GetCompleteCommand,
	ScheduleNextCommand,
	WfeChecking,
	TimeSync,
};

/* Describes the encoding of the "Data" field. */
union DataUnion {
	EntryData m_Raw; /* Raw access to the full Data value. */
	struct {
		/* Identifies the category for this entry. The layout of the
		 * rest of the data is category-specific.
		 */
		EntryDataCategory m_Category;
		union {
			struct {
				uint8_t m_Chars[3];
			} m_LabelFields;
			struct {
				uint8_t m_CommandIdx : 8;
			} m_CommandFields;
			struct {
				uint8_t m_CommandIdx : 8;
				uint8_t m_DmaCategory : 5;
				uint8_t m_DmaHardwareId : 3;
				uint8_t m_StripeIdx : 8;
			} m_DmaFields;
			struct {
				uint8_t m_CommandIdx : 8;
				uint8_t m_StripeIdx : 8;
				uint8_t m_BankId : 1;
			} m_TsuFields;
			struct {
				uint8_t m_CommandIdx : 8;
				uint8_t m_StripeIdx : 8;
			} m_MceStripeSetupFields;
			struct {
				uint8_t m_CommandIdx : 8;
				uint8_t m_StripeIdx : 8;
			} m_PleStripeSetupFields;
			struct {
				uint8_t m_CommandIdx : 8;
				uint8_t m_StripeIdx : 8;
				uint8_t m_DmaCategory : 3;
			} m_DmaStripeSetupFields;
			struct {
				uint8_t m_CommandIdx : 8;
			} m_CompleteCommandsFields;
			struct {
				uint8_t m_CommandIdx : 8;
			} m_ScheduleCommandsFields;
			struct {
				uint8_t m_TimeSyncData[3];
			} m_TimeSyncFields;
		};
	};
};

static_assert(sizeof(DataUnion) == sizeof(EntryData),
	      "Union/struct packing is incorrect");

#endif /* defined(__cplusplus) */
#endif /* _ETHOSN_SHARED_H_ */
