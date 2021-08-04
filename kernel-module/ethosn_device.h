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

#ifndef _ETHOSN_DEVICE_H_
#define _ETHOSN_DEVICE_H_

#include "scylla_addr_fields_public.h"
#include "scylla_regs_public.h"
#include "ethosn_dma.h"
#include "ethosn_firmware.h"
#include "uapi/ethosn.h"

#include <linux/atomic.h>
#include <linux/cdev.h>
#include <linux/debugfs.h>
#include <linux/delay.h>
#include <linux/io.h>
#include <linux/list.h>
#include <linux/mutex.h>
#include <linux/timer.h>
#include <linux/wait.h>

struct ethosn_inference;

struct ethosn_addr_map {
	u32              region;
	ethosn_address_t extension;
};

struct ethosn_inference_queue {
	struct mutex     inference_queue_mutex;
	struct list_head inference_queue;
};

/*
 * This enum contains different error condition that can be reported
 * by our driver.
 *
 * The decision of adding a new error into this list must be carefully
 * considered as this way of reporting this kind of error pollutes the
 * production code.
 */
enum ethosn_status_code {
	WRONG_CORE_SCHEDULE,
	CONCURRENT_INFERENCE_DETECTED,
	INFERENCE_SCHEDULED_ON_BUSY_CORE
};

struct ethosn_device {
	struct ethosn_core            **core;
	struct device                 *dev;
	struct cdev                   cdev;
	struct mutex                  mutex;
	int                           num_cores;
	struct ethosn_inference_queue queue;
	struct ethosn_dma_allocator   *allocator;
	uint32_t                      current_busy_cores;
	uint32_t                      status_mask;
};

struct ethosn_core {
	struct device               *dev;
	uint32_t                    core_id;
	struct dentry               *debug_dir;
	struct debugfs_regset32     debug_regset;

	void __iomem                *top_regs;
	int                         queue_size;

	struct ethosn_device        *parent;
	struct ethosn_dma_allocator *allocator;
	struct ethosn_addr_map      dma_map;
	struct ethosn_addr_map      firmware_map;
	struct ethosn_addr_map      work_data_map;
	struct ethosn_dma_info      *firmware;
	struct ethosn_dma_info      *firmware_stack;
	struct ethosn_dma_info      *firmware_vtable;
	struct ethosn_dma_info      *mailbox;
	struct ethosn_dma_info      *mailbox_request;
	struct ethosn_dma_info      *mailbox_response;
	void                        *mailbox_message;
	uint32_t                    num_pongs_received;
	bool                        firmware_running;

	/* Stores the response from the firmware containing capabilities data.
	 * This is allocated when the data is received from the firmware and
	 * copied into user space when requested via an ioctl.
	 */
	struct {
		void   *data;
		size_t size;
	}            fw_and_hw_caps;

	/* Information on memory regions */
	bool         ethosn_f_stream_configured;
	bool         ethosn_wd_stream_configured;
	bool         ethosn_cs_stream_configured;
	bool         ethosn_mpu_enabled;

	struct mutex mutex;

	/* Whether to tell the firmware to send level-sensitive interrupts
	 * in all cases. This is set based on the interrupt configuration in
	 * the .dts and used when booting the firmware.
	 */
	bool                    force_firmware_level_interrupts;
	struct workqueue_struct *irq_wq;
	struct work_struct      irq_work;
	atomic_t                irq_status;

	struct ethosn_inference *current_inference;

	/*
	 * This tells us if the device initialization has been completed.
	 * Set it to 1 before returning from ethon_device_init().
	 * Set it to 0 at the beginning of ethosn_device_deinit().
	 */
	atomic_t init_done;

	/* Ram log */
	struct {
		struct mutex      mutex;
		wait_queue_head_t wq;
		struct dentry     *dentry;
		size_t            size;
		uint8_t           *data;
		size_t            rpos;
		size_t            wpos;
	} ram_log;

	struct {
		struct ethosn_profiling_config config;
		uint32_t                       mailbox_messages_sent;
		uint32_t                       mailbox_messages_received;
		uint32_t                       rpm_suspend_count;
		uint32_t                       rpm_resume_count;
		uint32_t                       pm_suspend_count;
		uint32_t                       pm_resume_count;

		/* The buffer currently being written to by the firmware to
		 * record profiling entries.
		 * See also firmware_buffer_pending below.
		 */
		struct ethosn_dma_info *firmware_buffer;

		/* When a change to the profiling buffer is requested (e.g.
		 * turning it off or changing the size) we cannot free the old
		 * buffer immediately as the firmware may still be writing to
		 * it. We must keep the old buffer around until the firmware
		 * has acknowledged that it is using the new one.
		 * This buffer represents the new one which has been sent to the
		 * firmware and we are waiting for acknowledgement that it
		 * is being used.
		 */
		bool                   is_waiting_for_firmware_ack;
		struct ethosn_dma_info *firmware_buffer_pending;
	} profiling;
};

/**
 * ethosn_device_init() - Initialize the Ethos-N core.
 * @core:	Pointer to Ethos-N core.
 *
 * Return: 0 on success, else error code.
 */
int ethosn_device_init(struct ethosn_core *core);

/**
 * ethosn_device_deinit() - Deinitialize the Ethos-N core.
 * @core:	Pointer to Ethos-N core.
 */
void ethosn_device_deinit(struct ethosn_core *core);

/**
 * to_ethosn_addr() - Convert Linux address to Ethos-N address.
 * @linux_addr:		Linux address.
 * @addr_map:		Ethos-N region extensions info
 *
 *                  MCU                                       Linux
 *              - +------+  region_offset                   +-------+
 *              | | Code |  +-----------+  -                |       |
 *              | +------+  |           |  | region_extend  |       |
 *              | | SRAM |  |           |  v                |       |
 * region_addr  | +------+  |           |                   |       |
 *              | | Regs |  |           | linux_addr        |       |
 *              | +------+  |           +-----------------> |       |
 *              | | RAM0 |  |                               |       |
 *  ethosn_addr v +------+  |                               |       |
 *  ------------> | RAM1 | -+                               |       |
 *                +------+                                  |       |
 *                | Dev0 |                                  +-------+
 *                +------+
 *                | Dev1 |
 *                +------+
 *                | Bus  |
 *                +------+
 *
 * The MCU address space is divided into 8 regions. For region 'code', 'ram0'
 * and 'ram1' address extensions can be configured which are appended to the
 * region address.
 *
 * 'ethosn_addr' is a 32 bit MCU address. The upper 3 bits of the address decide
 * which region the address belongs to.
 *
 * 'region_offset' is the offset from the begin of the region.
 *
 * 'region_extend' is the address extension for a region.
 *
 * The linux address is calculated as:
 * region_mask = ((1 << 29) - 1);
 * region_offset = ethosn_addr & region_mask;
 * linux_addr = region_offset + region_extend;
 *
 * This function tries to invert the calculation and find the ethosn_addr.
 *
 * Return: Ethos-N address on success, else error code.
 */
resource_size_t to_ethosn_addr(const resource_size_t linux_addr,
			       const struct ethosn_addr_map *addr_map);

/**
 * ethosn_write_top_reg() - Write top register.
 * @core:	Pointer to Ethos-N core.
 * @page:	Register page.
 * @offset:	Register offset.
 * @value:	Value to be written.
 */
void ethosn_write_top_reg(struct ethosn_core *core,
			  const u32 page,
			  const u32 offset,
			  const u32 value);

/**
 * ethosn_read_top_reg() - Read top register.
 * @core:	Pointer to Ethos-N core.
 * @page:	Register page.
 * @offset:	Register offset.
 *
 * Return: Register value.
 */
u32 ethosn_read_top_reg(struct ethosn_core *core,
			const u32 page,
			const u32 offset);

/**
 * ethosn_smmu_available() -  Checks if SMMU is available.
 * @dev: Pointer to struct device.
 *
 * Return: 'true' if SMMU is available, otherwise 'false'
 */
bool ethosn_smmu_available(struct device *dev);

/**
 * ethosn_reset_and_start_ethosn() - Perform startup sequence for device
 * @core:	Pointer to Ethos-N core.
 *
 * Return: 0 on success, else error code.
 */
int ethosn_reset_and_start_ethosn(struct ethosn_core *core);

/**
 * ethosn_notify_firmware() - Trigger IRQ on Ethos-N .
 * @core:	Pointer to Ethos-N core.
 */
void ethosn_notify_firmware(struct ethosn_core *core);

/**
 * ethosn_reset() - Reset the Ethos-N .
 * @core:	Pointer to Ethos-N core.
 *
 * Return: 0 on success, else error code.
 */
int ethosn_reset(struct ethosn_core *core);

/**
 * ethosn_set_power_ctrl() - Configure power control.
 * @core:	Pointer to Ethos-N core.
 * @clk_on:	Request clock on if true.
 */
void ethosn_set_power_ctrl(struct ethosn_core *core,
			   bool clk_on);

/**
 * ethosn_set_mmu_stream_id() - Configure the mmu stream id0.
 * @core:-	Pointer to the Ethos-N core
 *
 * Return: Negative error code on error, zero otherwise
 */
int ethosn_set_mmu_stream_id(struct ethosn_core *core);

/**
 * ethosn_set_addr_ext() - Set address extension offset for stream.
 * @core:	Pointer to Ethos-N core.
 * @stream:	Which stream to update.
 * @offset:	Address offset.
 *
 * Return: 0 on success, else error code.
 */
int ethosn_set_addr_ext(struct ethosn_core *core,
			unsigned int stream,
			ethosn_address_t offset,
			struct ethosn_addr_map *addr_map);

/**
 * ethosn_dump_gps() - Dump all general purpose registers.
 * @core:	Pointer to Ethos-N core.
 *
 * Return: None.
 */
void ethosn_dump_gps(struct ethosn_core *core);

/**
 * ethosn_read_message() - Read message from Ethos-N mailbox.
 * @core:	Pointer to Ethos-N core.
 * @header:	Message header.
 * @data:	Pointer to data.
 * @length:	Max length in bytes of data buffer.
 *
 * Return: 0 on success, else error code.
 */
int ethosn_read_message(struct ethosn_core *core,
			struct ethosn_message_header *header,
			void *data,
			size_t length);

/**
 * ethosn_write_message() - Write message to Ethos-N mailbox.
 * @core:	Pointer to Ethos-N core.
 * @type:	Message type.
 * @data:	Pointer to data.
 * @length:	Length in bytes of data buffer.
 *
 * Return: 0 on success, else error code.
 */
int ethosn_write_message(struct ethosn_core *core,
			 enum ethosn_message_type type,
			 void *data,
			 size_t length);

/**
 * ethosn_send_version_request() - Send version request to Ethos-N .
 * @core:	Pointer to Ethos-N core.
 *
 * Return: 0 on success, else error code.
 */
int ethosn_send_version_request(struct ethosn_core *core);

/**
 * ethosn_send_fw_hw_capabilities_request() - Send FW & HW capabilities request.
 * @core:	Pointer to Ethos-N core.
 *
 * Return: 0 on success, else error code.
 */
int ethosn_send_fw_hw_capabilities_request(struct ethosn_core *core);

/**
 * ethosn_send_stash_request() - Send stash request if SMMU is available.
 * @core: Pointer to Ethos-N core.
 *
 * Return: 0 on succss, else error code
 */
int ethosn_send_stash_request(struct ethosn_core *core);

/**
 * ethosn_send_configure_profiling() - Send request to tell the firmware to
 *                                  enable/ disable profiling.
 * @core:	Pointer to Ethos-N core.
 *
 * Return: 0 on success, else error code.
 */
int ethosn_configure_firmware_profiling(struct ethosn_core *core,
					struct ethosn_profiling_config *
					new_config);

/**
 * ethosn_configure_firmware_profiling_ack() - Update internal state to
 *                                          account for the firmware having
 *                                          acknowledged a configure profiling
 *                                          request. Typically this would mean
 *                                          freeing any old buffer that is no
 *                                          longer being used.
 * @core:	Pointer to Ethos-N core.
 *
 * Return: 0 on success, else error code.
 */
int ethosn_configure_firmware_profiling_ack(struct ethosn_core *core);

/**
 * ethosn_send_time_sync() - Send sync timestamp to the firmware in order to
 *                           sync the firmware profiling data with the user
 *                           space profiling data.
 * @core:	Pointer to Ethos-N core.
 *
 * Return: 0 on success, else error code.
 */
int ethosn_send_time_sync(struct ethosn_core *core);

/**
 * ethosn_send_ping() - Send ping to Ethos-N .
 * @core:	Pointer to Ethos-N core.
 *
 * Return: 0 on success, else error code.
 */
int ethosn_send_ping(struct ethosn_core *core);

/**
 * ethosn_send_inference() - Send inference to Ethos-N .
 * @core:		Pointer to Ethos-N core.
 * @buffer_array:	DMA address to buffer array.
 * @user_arg:		User argument. Will be returned in interence response.
 *
 * Return: 0 on success, else error code.
 */
int ethosn_send_inference(struct ethosn_core *core,
			  dma_addr_t buffer_array,
			  uint64_t user_arg);

/**
 * ethosn_send_stream_request() - Send region request to Ethos-N .
 * @core:	Pointer to Ethos-N core.
 * @stream_id:	Stream identifier.
 *
 * Return: 0 on success, else error code.
 */
int ethosn_send_stream_request(struct ethosn_core *core,
			       enum ethosn_stream_id stream_id);

/**
 * ethosn_send_mpu_enable_request() - Send Mpu enable request to Ethos-N .
 * @core:		Pointer to Ethos-N core.
 *
 * Return: 0 on success, else error code.
 */
int ethosn_send_mpu_enable_request(struct ethosn_core *core);

/* ethosn_profiling_enabled() - Get status of the profiling enabled switch.
 *
 * Return: 'true' if profiling is enabled, otherwise 'false'.
 */
bool ethosn_profiling_enabled(void);

/* ethosn_stashing_enabled() - Get status of the stashing enabled switch.
 *
 * Return: 'true' if stashing is enabled, otherwise 'false'.
 */
bool ethosn_stashing_enabled(void);

/* ethosn_mailbox_empty() - Check if the mailbox is empty.
 *
 * Return: 'true' if mailbox is empty, otherwise 'false'.
 */
bool ethosn_mailbox_empty(struct ethosn_queue *queue);

/* ethosn_clock_frequency() - Get clock frequency in MHz.
 *
 * Return: clock frequency.
 */
int ethosn_clock_frequency(void);

/* ethosn_get_global_core_for_testing() - Exposes global access to the
 *                                        most-recently created Ethos-N core
 *                                        (in case of single core) or core0 (in
 *                                        case of multicore) for testing
 *                                        purposes.
 */
struct ethosn_core *ethosn_get_global_core_for_testing(void);

/* ethosn_get_global_device_for_testing() - Exposes global access to the
 *                                          Ethos-N parent device for testing
 *                                          purposes.
 */
struct ethosn_device *ethosn_get_global_device_for_testing(void);

#endif /* _ETHOSN_DEVICE_H_ */
