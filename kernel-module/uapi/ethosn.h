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

/* This header defines the interface between the Ethos-N kernel module and
 * userspace.
 */

#ifndef _ETHOSN_H_
#define _ETHOSN_H_

#include "ethosn_shared.h"

#include <linux/ioctl.h>
#include <linux/types.h>

#ifndef __KERNEL__
#define __user
#define __packed __attribute__((__packed__))
#endif

/*****************************************************************************
 * Inference
 *****************************************************************************/

/*
 * Ethos-N Driver user space interface
 *
 * Example usage:
 *
 *      int dev_fd = open("/dev/ethosn0", O_RDWR);
 *
 *      struct ethosn_network_req network = {
 *          ...
 *      };
 *      int net_fd = ioctl(dev_fd, ETHOSN_IOCTL_REGISTER_NETWORK, &network);
 *
 *      struct ethosn_buffer_req buf_req;
 *
 *      buf_req.size  = 1024;
 *      buf_req.flags = MB_WRONLY | MB_ZERO;
 *      int input_fd = ioctl(dev_fd, ETHOSN_IOCTL_CREATE_BUFFER, &buf_req);
 *
 *      buf_req.size  = 512;
 *      buf_req.flags = MB_RDONLY | MB_ZERO;
 *      int output_fd = ioctl(dev_fd, ETHOSN_IOCTL_CREATE_BUFFER, &buf_req);
 *
 *      // dev_fd can be closed and existing handles remain valid
 *      close(dev_fd);
 *
 *      ...
 *
 *      // Use mmap to populate input buffers
 *      in_ptr = mmap(..., input_fd, ...);
 *      memcpy(in_ptr, src, size);
 *
 *      ...
 *
 *      // Use ioctl to schedule an inference
 *      const int inputs[] = { input_fd };
 *      const int outputs[] = { output_fd };
 *      const struct ethosn_inference_req sched_req = {
 *          .num_inputs = sizeof(inputs) / sizeof(inputs[0]),
 *          .input_fds = inputs,
 *          .num_outputs = sizeof(outputs) / sizeof(outputs[0]),
 *          .output_fds = outputs,
 *      };
 *      int sched_fd = ioctl(dev_fd, ETHOSN_IOCTL_SCHEDULE_INFERENCE,
 *                           &sched_req);
 *
 *      ...
 *
 *      // Use select/poll/epoll to wait for scheduled inference
 *      struct pollfd poll_fd = { .fd = sched_fd, .events = POLLIN };
 *      poll(&poll_fd, 1, -1);
 *
 *      ...
 *
 *      // Use read to read the status of inference execution
 *      int32_t status;
 *      ssize_t n = read(inference, &status, sizeof(status));
 *      if ((n != sizeof(status)) || (status != ETHOSN_INFERENCE_COMPLETED))
 *      {
 *          handle_error();
 *      }
 *
 *      ...
 *
 *      // Use mmap to read output buffers
 *      out_ptr = mmap(..., output_fd, ...);
 *      memcpy(dst, out_ptr, size);
 *
 *      ...
 *
 *      // Use close to release handles
 *      close(net_fd);
 *      close(input_fd);
 *      close(output_fd);
 *      close(sched_fd);
 */

struct ethosn_buffer_info {
	__u32 id;     /* <- id in command stream */
	__u32 offset; /* <- ignored for inputs/outputs */
	__u32 size;
};

struct ethosn_buffer_infos {
	__u32                           num;
	const struct ethosn_buffer_info __user *info;
};

struct ethosn_constant_data {
	__u32             size;
	const void __user *data;
};

struct ethosn_network_req {
	struct ethosn_buffer_infos  dma_buffers;
	struct ethosn_constant_data dma_data;

	struct ethosn_buffer_infos  cu_buffers;
	struct ethosn_constant_data cu_data;

	struct ethosn_buffer_infos  intermediate_buffers;
	__u32                       intermediate_data_size;

	struct ethosn_buffer_infos  input_buffers;
	struct ethosn_buffer_infos  output_buffers;
};

struct ethosn_inference_req {
	__u32            num_inputs;
	const int __user *input_fds;

	__u32            num_outputs;
	const int __user *output_fds;
};

struct ethosn_buffer_req {
	__u32 size;
	__u32 flags;
};

/*****************************************************************************
 * Capabilities
 *****************************************************************************/

/**
 * struct ethosn_fw_hw_capabilities - Information about the FW and HW
 * capabilities.
 * @data:	FW and HW capabilities data.
 * @size:	Size of data.
 */
struct ethosn_fw_hw_capabilities {
	void  *data;
	__u32 size;
};

/*****************************************************************************
 * Logging
 *****************************************************************************/

/**
 * Magic word "ANPU" that prefix all messages.
 *
 * Messages are stored in native byte order. The magic word can be used to
 * detect if the log has been stored in the same byte order as the application
 * unpacking the log is using.
 */
#define ETHOSN_LOG_MAGIC                   0x414e5055

/**
 * enum ethosn_log_type - Message type.
 */
enum ethosn_log_type {
	ETHOSN_LOG_TYPE_TEXT     = 0,
	ETHOSN_LOG_TYPE_UAPI     = 1,
	ETHOSN_LOG_TYPE_FIRMWARE = 2,
	ETHOSN_LOG_TYPE_MAX
};

/**
 * struct ethosn_log_timeval - Portable time value format.
 * @sec:		Seconds since 1970-01-01, Unix time epoch.
 * @nsec:		Nano seconds.
 */
struct ethosn_log_timeval {
	__u64 sec;
	__u64 nsec;
} __packed;

/**
 * struct ethosn_log_header - Common header for all messages stored in RAM
 * buffer.
 * @magic:		Magic word.
 * @length:		Length of message, excluding this header.
 * @type:		Message type.
 * @timestamp:		Time stamp.
 */
struct ethosn_log_header {
	__u32                     magic;
	__u32                     length;
	__u32                     type;
	struct ethosn_log_timeval timestamp;
} __packed;

/**
 * struct ethosn_log_uapi_header - Common header for all UAPI log messages.
 * @ioctl:		IOCTL number.
 */
struct ethosn_log_uapi_header {
	__u32 ioctl;
} __packed;

/**
 * struct ethosn_log_uapi_buffer_req - Log buffer request.
 * @request:		UAPI request.
 * @handle:		Handle identifier.
 * @fd:			User space file descriptor.
 */
struct ethosn_log_uapi_buffer_req {
	struct ethosn_buffer_req request;
	__u64                    handle;
	__u32                    fd;
} __packed;

/**
 * struct ethosn_log_uapi_network_req - Log network request.
 * @request:		UAPI request.
 * @handle:		Handle identifier.
 * @fd:			User space file descriptor.
 */
struct ethosn_log_uapi_network_req {
	struct ethosn_network_req request;
	__u64                     handle;
	__u32                     fd;
} __packed;

/**
 * struct ethosn_log_uapi_inference_req - Log inference request.
 * @request:		UAPI request.
 * @handle:		Handle identifier.
 * @network_handle:	Network handle the inference is connected to.
 * @fd:			User space file descriptor.
 */
struct ethosn_log_uapi_inference_req {
	struct ethosn_inference_req request;
	__u64                       handle;
	__u64                       network_handle;
	__u32                       fd;
} __packed;

/**
 * enum ethosn_log_firmware_direction - Direction of firmware message.
 * @ETHOSN_LOG_FIRMWARE_INPUT:	Host<-Firmware.
 * @ETHOSN_LOG_FIRMWARE_OUTPUT:	Host->Firmware.
 */
enum ethosn_log_firmware_direction {
	ETHOSN_LOG_FIRMWARE_INPUT,
	ETHOSN_LOG_FIRMWARE_OUTPUT,
	ETHOSN_LOG_FIRMWARE_MAX
};

/**
 * struct ethosn_profiling_config - Global profiling options which can be
 *      passed to ETHOSN_IOCTL_CONFIGURE_PROFILING.
 */
struct ethosn_profiling_config {
	bool                                   enable_profiling;
	__u32                                  firmware_buffer_size;
	__u32                                  num_hw_counters;
	enum ethosn_profiling_hw_counter_types hw_counters[6];
} __packed;

/**
 * enum ethosn_poll_counter_name - All the counters that can be queried using
 *      ETHOSN_IOCTL_GET_COUNTER_VALUE.
 */
enum ethosn_poll_counter_name {
	/* Mailbox messages */
	ETHOSN_POLL_COUNTER_NAME_MAILBOX_MESSAGES_SENT,
	ETHOSN_POLL_COUNTER_NAME_MAILBOX_MESSAGES_RECEIVED,

	/* Power management */
	ETHOSN_POLL_COUNTER_NAME_RPM_SUSPEND,
	ETHOSN_POLL_COUNTER_NAME_RPM_RESUME,
	ETHOSN_POLL_COUNTER_NAME_PM_SUSPEND,
	ETHOSN_POLL_COUNTER_NAME_PM_RESUME,
};

/**
 * struct ethosn_log_firmware_header - Firmware log header.
 * @inference:		Current running inference handle.
 * @direction:		Message direction.
 */
struct ethosn_log_firmware_header {
	__u64 inference;
	__u32 direction;
} __packed;

#define ETHOSN_IOCTL_BASE       0x01
#define ETHOSN_IO(nr)           _IO(ETHOSN_IOCTL_BASE, nr)
#define ETHOSN_IOR(nr, type)    _IOR(ETHOSN_IOCTL_BASE, nr, type)
#define ETHOSN_IOW(nr, type)    _IOW(ETHOSN_IOCTL_BASE, nr, type)
#define ETHOSN_IOWR(nr, type)   _IOWR(ETHOSN_IOCTL_BASE, nr, type)

#define ETHOSN_IOCTL_CREATE_BUFFER \
	ETHOSN_IOW(0x00, struct ethosn_buffer_req)
#define ETHOSN_IOCTL_REGISTER_NETWORK \
	ETHOSN_IOW(0x01, struct ethosn_network_req)
#define ETHOSN_IOCTL_SCHEDULE_INFERENCE	\
	ETHOSN_IOW(0x02, struct ethosn_inference_req)
#define ETHOSN_IOCTL_FW_HW_CAPABILITIES	\
	ETHOSN_IOR(0x03, void *)
#define ETHOSN_IOCTL_LOG_CLEAR \
	ETHOSN_IO(0x04)
#define ETHOSN_IOCTL_GET_COUNTER_VALUE \
	ETHOSN_IOW(0x05, enum ethosn_poll_counter_name)
#define ETHOSN_IOCTL_CONFIGURE_PROFILING \
	ETHOSN_IOW(0x06, struct ethosn_profiling_config)
#define ETHOSN_IOCTL_GET_CLOCK_FREQUENCY \
	ETHOSN_IOW(0x07, void *)
#define ETHOSN_IOCTL_PING \
	ETHOSN_IO(0x08)
#define ETHOSN_IOCTL_GET_INTERMEDIATE_BUFFER \
	ETHOSN_IO(0x09)
#define ETHOSN_IOCTL_GET_VERSION \
	ETHOSN_IO(0x0a)

/*
 * Results from reading an inference file descriptor.
 * Note these must be kept in-sync with the driver library's definitions.
 */
#define ETHOSN_INFERENCE_SCHEDULED   0
#define ETHOSN_INFERENCE_RUNNING     1
#define ETHOSN_INFERENCE_COMPLETED   2
#define ETHOSN_INFERENCE_ERROR       3

#define MB_RDONLY 00000000
#define MB_WRONLY 00000001
#define MB_RDWR   00000002
#define MB_ZERO   00000010

/* Version information */
#define ETHOSN_KERNEL_MODULE_VERSION_MAJOR 2
#define ETHOSN_KERNEL_MODULE_VERSION_MINOR 0
#define ETHOSN_KERNEL_MODULE_VERSION_PATCH 0

/**
 * struct ethosn_kernel_module_version - stores the kernel module's version info
 * @major: This corresponds to the major version.
 * @minor: This corresponds to the minor version.
 * @patch: This corresponds to the patch version.
 */
struct ethosn_kernel_module_version {
	uint32_t major;
	uint32_t minor;
	uint32_t patch;
};

#endif /* _ETHOSN_H_ */
