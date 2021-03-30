/*
 *
 * (C) COPYRIGHT 2018-2021 Arm Limited.
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

#ifndef _ETHOSN_FIRMWARE_H_
#define _ETHOSN_FIRMWARE_H_

/* This file defines structs using zero-length arrays which are not available
 * in the C++ standard. The compilers we use do however support them as
 * extensions to the standard, so we can disable the warnings they produce.
 * Disabling of warnings is compiler-specific.
 */
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4200)
#endif

#include "uapi/ethosn_shared.h"

#ifdef __KERNEL__
#include <linux/types.h>
#else
#include <stdint.h>
#endif

/**
 * Version information
 *
 * This is common for the fat binary (ethosn.bin) and the individual
 * firmware binaries (sub-components of the fat binary).
 */
#define ETHOSN_FIRMWARE_VERSION_MAJOR 1
#define ETHOSN_FIRMWARE_VERSION_MINOR 0
#define ETHOSN_FIRMWARE_VERSION_PATCH 0

/** Max length of a cache line. Used to separate host and Ethos-N data. */
#define ETHOSN_CACHE_LINE_SIZE 128

/* Stack size */
#define ETHOSN_STACK_SIZE                  0x40000

#pragma pack(push, 1)

/******************************************************************************
 * Mailbox
 ******************************************************************************/

/**
 * Pointer to memory that will be accessed directly by the MCU should be 32 bit.
 *
 * Pointer to memory that will be copied with help of DMA may be up to 49 bits.
 */
typedef uint64_t ethosn_address_t;

/**
 * struct ethosn_queue - Dynamic size queue designed to be written from
 *                    one CPU and read from another.
 * @var capacity:	Size in bytes of the queue.
 * @var read:		Read index in bytes in the Data array.
 *                  The "reading" CPU advances this as it reads data from the
 *                  queue.
 *                  It should never read past the 'write' offset as that means
 *                  the end of the queue has been reached and it should wait
 *                  for the writing end to write some more data.
 * @var write:		Write index in bytes in the Data array.
 *                  The "writing" CPU advances this as it writes data into the
 *                  queue.
 *                  It should never write past the 'read' pointer as that means
 *                  the queue has become full and it should wait for the reading
 *                  end to catch up.
 * @var data:		Data array.
 *
 *
 * The queue is empty if-and-only-if: read == write.
 * Note that the queue can never be completely full, as that would be
 * indistinguishable from being empty!
 */
struct ethosn_queue {
	union {
		struct {
			uint32_t capacity;
			uint32_t read;
		};

		/* Padding added to avoid caching issues between non-coherent
		 * CPUs.
		 */
		uint8_t pad_0[ETHOSN_CACHE_LINE_SIZE];
	};
	union {
		uint32_t write;

		/* Padding added to avoid caching issues between non-coherent
		 * CPUs.
		 */
		uint8_t pad_1[ETHOSN_CACHE_LINE_SIZE];
	};
	uint8_t data[];
};

/**
 * Checks if data of the given size could ever fit in the queue, even
 * if it was completely empty. This is a simple check against the capacity,
 * but is wrapped in a function as the comparison is slightly different
 * to what you might naively expect.
 */
static inline bool ethosn_queue_can_ever_fit(const struct ethosn_queue *queue,
					     uint32_t size)
{

	/* Note we disallow the queue from ever becoming
	 * completely full, as that would be indistinguishable from being
	 * completely empty!
	 */
	return size < queue->capacity;
}

/**
 * Gets the current size of the given queue, i.e. how many bytes are available
 * to read.
 */
static inline uint32_t ethosn_queue_get_size(const struct ethosn_queue *queue)
{
	const uint32_t mask = queue->capacity - 1;

	return (queue->write - queue->read) & mask;
}

/**
 * Gets the amount of free space in the given queue, i.e. how many bytes can be
 * written.
 */
static inline uint32_t ethosn_queue_get_free_space(
	const struct ethosn_queue *queue)
{

	/* Note we subtract one to prevent the queue from ever becoming
	 * completely full, as that would be indistinguishable from being
	 * completely empty!
	 */
	return queue->capacity - ethosn_queue_get_size(queue) - 1;
}

/**
 * Skips the given number of bytes from the queue. This is equivalent to reading
 * those bytes and discarding them.
 * Returns false if there is not enough data in the queue to skip.
 */
static inline bool ethosn_queue_skip(struct ethosn_queue *queue,
				     uint8_t size)
{
	const uint32_t mask = queue->capacity - 1;

	/* Check that there is enough data for us to read */
	if (size > ethosn_queue_get_size(queue))
		return false;

	queue->read = (queue->read + size) & mask;

	return true;
}

/**
 * Reads the given number of bytes from the queue.
 * The caller is required to commit the read_pending pointer
 * to queue.read when they have ensured that it can be advanced.
 * Returns false if there is not enough data in the queue to read.
 */
static inline bool ethosn_queue_read(const struct ethosn_queue *queue,
				     uint8_t *dst,
				     uint32_t size,
				     uint32_t *read_pending)
{
	const uint32_t mask = queue->capacity - 1;
	uint32_t read = queue->read;
	uint32_t i;

	/* Check that there is enough data for us to read */
	if (size > ethosn_queue_get_size(queue))
		return false;

	for (i = 0; i < size; ++i) {
		dst[i] = queue->data[read];
		read = (read + 1) & mask;
	}

	*read_pending = read;

	return true;
}

/**
 * Writes the given buffer of bytes to the queue.
 * The caller is required to commit the write_pending pointer
 * to queue.write when they have ensured that the payload is
 * readable (e.g. flushed) by the "reading" CPU.
 * Returns false if there is not enough free space in the queue.
 */
static inline bool ethosn_queue_write(struct ethosn_queue *queue,
				      const uint8_t *src,
				      uint32_t size,
				      uint32_t *write_pending)
{
	const uint32_t mask = queue->capacity - 1;
	uint32_t write;
	uint32_t i;

	/* Check if there is enough space for our data */
	if (ethosn_queue_get_free_space(queue) < size)
		return false;

	write = *write_pending;

	for (i = 0; i < size; ++i) {
		queue->data[write] = src[i];
		write = (write + 1) & mask;
	}

	*write_pending = write;

	return true;
}

/**
 * struct ethosn_mailbox - Mailbox structure
 * @var request:	Pointer to message queue going from host to Ethos-N .
 * @var response:	Pointer to message queue going from Ethos-N to host.
 * @var severity:	Log severity level. @see ethosn_log_severity.
 *
 * This is the interface between the host and the Ethos-N .
 */
struct ethosn_mailbox {
	ethosn_address_t request;
	ethosn_address_t response;
	uint32_t         severity;
};

/******************************************************************************
 * Message types
 ******************************************************************************/

/**
 * Message types.
 */
enum ethosn_message_type {
	/* The order of the message type matters for the enriched RTL testing
	 * system. Inference request and response must not change to avoid
	 * problems.
	 */

	/* ethosn_message_inference_request */
	ETHOSN_MESSAGE_INFERENCE_REQUEST,
	/* ethosn_message_inference_response */
	ETHOSN_MESSAGE_INFERENCE_RESPONSE,

	/* ethosn_message_text */
	ETHOSN_MESSAGE_TEXT,

	/* ethosn_firmware_profiling_configuration */
	ETHOSN_MESSAGE_CONFIGURE_PROFILING,
	/* void */
	ETHOSN_MESSAGE_CONFIGURE_PROFILING_ACK,

	/* ethosn_message_stream_request */
	ETHOSN_MESSAGE_STREAM_REQUEST,

	/* ethosn_message_stream_response */
	ETHOSN_MESSAGE_STREAM_RESPONSE,

	/* ethosn_message_time_sync_request */
	ETHOSN_MESSAGE_TIME_SYNC,

	/* uint32_t */
	ETHOSN_MESSAGE_DELAY,

	/* void */
	ETHOSN_MESSAGE_MPU_ENABLE_REQUEST,
	ETHOSN_MESSAGE_MPU_ENABLE_RESPONSE,

	ETHOSN_MESSAGE_PING,
	ETHOSN_MESSAGE_PONG,

	ETHOSN_MESSAGE_FW_HW_CAPS_REQUEST,
	ETHOSN_MESSAGE_FW_HW_CAPS_RESPONSE,

	ETHOSN_MESSAGE_MAX
};

/**
 * struct ethosn_message_header - Message header
 * @var type:		Message type. @see ethosn_message_type.
 * @var length:		Length in bytes of Value array.
 *
 * Every message between host and Ethos-N should begin with a message header.
 * The type of the message determines if the header is followed by additional
 * payload data.
 */
struct ethosn_message_header {
	uint32_t type;
	uint32_t length;
	uint8_t  value[];
};

/******************************************************************************
 * Inference
 ******************************************************************************/

/**
 * struct ethosn_buffer_desc - Buffer descriptor
 * @var address:	Pointer to buffer.
 * @var size:		Size in bytes of buffer.
 */
struct ethosn_buffer_desc {
	ethosn_address_t address;
	uint32_t         size;
};

/**
 * struct ethosn_buffer_array - Dynamic size buffer array
 * @var num_buffers:	Number of buffers.
 * @var buffers:	Array of buffer descriptors.
 */
struct ethosn_buffer_array {
	uint32_t                  num_buffers;
	struct ethosn_buffer_desc buffers[];
};

/**
 * Inference status.
 */
enum ethosn_inference_status {
	ETHOSN_INFERENCE_STATUS_OK,
	ETHOSN_INFERENCE_STATUS_ERROR,
	ETHOSN_INFERENCE_STATUS_MAX
};

/**
 * struct ethosn_message_inference_request - Inference request message
 * @var user_argument:	User argument.
 * @var buffer_array:	Pointer to buffer array. @see ethosn_buffer_header.
 *
 * Following a ethosn_message_header.
 */
struct ethosn_message_inference_request {
	uint64_t         user_argument;
	ethosn_address_t buffer_array;
};

/**
 * struct ethosn_message_inference_response - Inference response message
 * @var user_argument:	User argument.
 * @var status:		Inference status.
 *
 * Following a ethosn_message_header.
 */
struct ethosn_message_inference_response {
	uint64_t user_argument;
	uint32_t status;
};

/******************************************************************************
 * Text message logging
 ******************************************************************************/

/**
 * Severity of log message.
 */
enum ethosn_log_severity {
	ETHOSN_LOG_PANIC,
	ETHOSN_LOG_ERROR,
	ETHOSN_LOG_WARNING,
	ETHOSN_LOG_INFO,
	ETHOSN_LOG_DEBUG,
	ETHOSN_LOG_VERBOSE
};

/**
 * struct ethosn_message_text - Text message
 * @var severity:	Severity of log message.
 *
 * Following a ethosn_message_type.
 */
struct ethosn_message_text {
	uint32_t severity;
	char     text[];
};

/******************************************************************************
 * Profiling
 ******************************************************************************/

/**
 * struct ethosn_firmware_profiling_configuration - Message payload sent to the
 *	firmware for a ETHOSN_MESSAGE_CONFIGURE_PROFILING message. Describes the
 *	profiling configuration that the firmware should set itself to.
 *
 * @buffer_address: Firmware-accessible address to a struct of type
 *                  ethosn_profiling_buffer which is where the firmware should
 *                  write its profiling data.
 */
struct ethosn_firmware_profiling_configuration {
	bool                                   enable_profiling;
	ethosn_address_t                       buffer_address;
	uint32_t                               buffer_size;
	uint32_t                               num_hw_counters;
	enum ethosn_profiling_hw_counter_types hw_counters[6];
};

/**
 * struct ethosn_profiling_buffer - Layout of the firmware's profiling buffer.
 *	This is a circular buffer which the firmware writes into and the kernel
 *      reads from. When the firmware reaches the end, it simply starts
 *	overwriting at the beginning again. There is no mechanism in place to
 *	prevent the firmware from overwriting data which the kernel has not yet
 *	read.
 *      This is intentional as we do not want to stall the firmware waiting for
 *      the kernel (especially as there may not be anyone reading the profiling
 *      data at the other side!).
 *
 * @firmware_write_index: Index into the entries array that the firmware
 *                        should write to next. This is updated by the firmware
 *			  and read-only for the kernel.
 * @entries: Payload of the buffer.
 */

struct ethosn_profiling_buffer {
	union {
		uint32_t firmware_write_index;

		/* Padding to ensure firmware_write_index and entries are on
		 * different cache lines, so flushing is independent.
		 */
		uint8_t padding[ETHOSN_CACHE_LINE_SIZE];
	};
	struct ethosn_profiling_entry entries[];
};

/**
 * struct ethosn_message_profiling_entries - Profiling entries message
 * @num_entries: Number of entries.
 * @entries:	 @see ethosn_profiling_entry.
 *
 * Following a ethosn_message_type.
 */
struct ethosn_message_profiling_entries {
	uint32_t                      num_entries;
	struct ethosn_profiling_entry entries[];
};

/**
 * struct ethosn_message_time_sync_request - Message sync request
 * @var timestamp:	Application processor timestamp
 *
 * Following a ethosn_message_type.
 */
struct ethosn_message_time_sync_request {
	uint64_t timestamp;
};

/******************************************************************************
 * Memory regions management
 ******************************************************************************/

/**
 * Streams identifier
 */
enum  ethosn_stream_id {
	ETHOSN_STREAM_FIRMWARE         = 0,
	ETHOSN_STREAM_WORKING_DATA     = 1,
	ETHOSN_STREAM_COMMAND_STREAM   = 2,
	ETHOSN_STREAM_DMA              = 3,
	ETHOSN_STREAM_DMA_INTERMEDIATE = 4,
};

/**
 * struct ethosn_message_stream_request - Memory stream message
 * @var stream_id:	Stream id
 * @var size:		Size of the stream memory region
 *
 * Following a ethosn_message_type.
 */
struct ethosn_message_stream_request {
	uint32_t stream_id;
	uint32_t size;
};

/**
 * Region setup status.
 */
enum ethosn_stream_status {
	ETHOSN_STREAM_STATUS_OK,
	ETHOSN_STREAM_STATUS_ERROR,
	ETHOSN_STREAM_STATUS_MAX
};

/**
 * struct ethosn_message_stream_response - Memory stream response message
 * @var stream_id:	Stream id
 * @var status:         Stream memory region size setup status.
 *
 * Following a ethosn_message_header.
 */
struct ethosn_message_stream_response {
	uint32_t stream_id;
	uint32_t status;
};

/*
 * Define the DL1_GP registers to be used for special purpose communication
 * between kernel driver and firmware
 */
#define GP_IRQ                          DL1_GP0
#define GP_MMUSID0                      DL1_GP1
#define GP_MAILBOX                      DL1_GP2
#define GP_STREAM1_ADDRESS_EXTEND       DL1_GP3
#define GP_STREAM2_ADDRESS_EXTEND       DL1_GP4

#pragma pack(pop)

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#endif /* _ETHOSN_FIRMWARE_H_ */
