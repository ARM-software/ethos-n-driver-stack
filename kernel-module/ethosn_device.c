/*
 *
 * (C) COPYRIGHT 2018-2019 ARM Limited. All rights reserved.
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

#include "ethosn_device.h"

#include "ethosn_firmware.h"
#include "ethosn_log.h"

#include <linux/firmware.h>
#include <linux/iommu.h>
#include <linux/log2.h>
#include <linux/module.h>
#include <linux/of.h>
#include <linux/uaccess.h>
#include <linux/time.h>

/* Number of bits the MCU Vector Table address is shifted. */
#define SYSCTLR0_INITVTOR_SHIFT         7

/* Init vector table size */
#define ETHOSN_VTABLE_SIZE                 16

/* Firmware code size */
#define ETHOSN_CODE_SIZE                   0x40000

/* Timeout in us when resetting the Ethos-N */
#define ETHOSN_RESET_TIMEOUT_US            (10 * 1000 * 1000)
#define ETHOSN_RESET_WAIT_US               1

/* Regset32 entry */
#define REGSET32(r) { __stringify(r), \
		      TOP_REG(DL1_RP, DL1_ ## r) - TOP_REG(0, 0) }

#define NANOSECONDS_IN_A_SECOND         (1000 * 1000 * 1000)

static int severity = ETHOSN_LOG_INFO;
module_param(severity, int, 0660);

static int ethosn_queue_size = 65536;
module_param_named(queue_size, ethosn_queue_size, int, 0440);

static bool profiling_enabled;
module_param_named(profiling, profiling_enabled, bool, 0664);

/* Clock frequency expressed in MHz */
static int clock_frequency = 1000;
module_param_named(clock_frequency, clock_frequency, int, 0440);

/* Exposes global access to the most-recently created Ethos-N device for testing
 * purposes. See ethosn-tests module
 */
static struct ethosn_device *ethosn_global_device_for_testing;

static void __iomem *ethosn_top_reg_addr(void __iomem *const top_regs,
					 const u32 page,
					 const u32 offset)
{
	return (u8 __iomem *)top_regs + (TOP_REG(page, offset) - TOP_REG(0, 0));
}

resource_size_t to_ethosn_addr(const resource_size_t linux_addr,
			       const struct ethosn_addr_map *addr_map)
{
	const resource_size_t region_addr = addr_map->region;
	const resource_size_t region_extend = addr_map->extension;
	const resource_size_t region_size = 1 << REGION_SHIFT;
	const resource_size_t region_mask = region_size - 1;
	resource_size_t ethosn_addr;

	/* Verify that region addresses are a multiple of the region size. */
	if ((region_addr | region_extend) & region_mask)
		return -EFAULT;

	/*
	 * Verify that the Linux address lies between the region extend and the
	 * region size.
	 */
	if ((linux_addr < region_extend) ||
	    (linux_addr >= (region_extend + region_size)))
		return -EFAULT;

	/* Combine the region address with the region offset. */
	ethosn_addr = region_addr | (linux_addr & region_mask);

	return ethosn_addr;
}

/**
 * mailbox_init() - Initialize the mailbox structure.
 * @ethosn:	Ethos-N device.
 *
 * Return: 0 on success, else error code.
 */
static int mailbox_init(struct ethosn_device *ethosn)
{
	struct ethosn_mailbox *mailbox = ethosn->mailbox->cpu_addr;
	struct ethosn_queue *request = ethosn->mailbox_request->cpu_addr;
	struct ethosn_queue *response = ethosn->mailbox_response->cpu_addr;
	resource_size_t mailbox_addr;

	/* Clear memory */
	memset(mailbox, 0, ethosn->mailbox->size);
	memset(request, 0, ethosn->mailbox_request->size);
	memset(response, 0, ethosn->mailbox_response->size);

	/* Setup queue sizes */
	request->capacity = ethosn->mailbox_request->size -
			    sizeof(struct ethosn_queue);
	response->capacity = ethosn->mailbox_response->size -
			     sizeof(struct ethosn_queue);

	/* Set severity, and make sure it's in the range [PANIC, VERBOSE]. */
	mailbox->severity = max(min(severity,
				    ETHOSN_LOG_VERBOSE), ETHOSN_LOG_PANIC);

	/* Set Ethos-N addresses from mailbox to queues */
	mailbox->request = to_ethosn_addr(ethosn->mailbox_request->iova_addr,
					  &ethosn->work_data_map);
	if (IS_ERR_VALUE((unsigned long)mailbox->request))
		return -EFAULT;

	mailbox->response = to_ethosn_addr(ethosn->mailbox_response->iova_addr,
					   &ethosn->work_data_map);
	if (IS_ERR_VALUE((unsigned long)mailbox->response))
		return -EFAULT;

	/* Store mailbox address in GP2 */
	mailbox_addr = to_ethosn_addr(ethosn->mailbox->iova_addr,
				      &ethosn->work_data_map);
	if (IS_ERR_VALUE((unsigned long)mailbox_addr))
		return -EFAULT;

	/* Sync memory to device */
	ethosn_dma_sync_for_device(ethosn, ethosn->mailbox);
	ethosn_dma_sync_for_device(ethosn, ethosn->mailbox_request);
	ethosn_dma_sync_for_device(ethosn, ethosn->mailbox_response);

	/* Store mailbox CU address in GP2 */
	ethosn_write_top_reg(ethosn, DL1_RP, GP_MAILBOX, mailbox_addr);

	return 0;
}

/**
 * mailbox_alloc() - Allocate the mailbox.
 * @ethosn:	Pointer to Ethos-N device.
 *
 * Return: 0 on success, else error code.
 */
static int mailbox_alloc(struct ethosn_device *ethosn)
{
	ethosn->mailbox = ethosn_dma_alloc(ethosn,
					   sizeof(struct ethosn_mailbox),
					   ETHOSN_PROT_READ | ETHOSN_PROT_WRITE,
					   ETHOSN_STREAM_WORKING_DATA,
					   GFP_KERNEL);
	if (IS_ERR(ethosn->mailbox_request)) {
		dev_warn(ethosn->dev,
			 "Failed to allocate memory for mailbox");

		return PTR_ERR(ethosn->mailbox_request);
	}

	ethosn->mailbox_request =
		ethosn_dma_alloc(ethosn,
				 sizeof(struct ethosn_queue) +
				 ethosn->queue_size,
				 ETHOSN_PROT_READ | ETHOSN_PROT_WRITE,
				 ETHOSN_STREAM_WORKING_DATA,
				 GFP_KERNEL);
	if (IS_ERR(ethosn->mailbox_request)) {
		dev_warn(ethosn->dev,
			 "Failed to allocate memory for mailbox request queue");

		return PTR_ERR(ethosn->mailbox_request);
	}

	ethosn->mailbox_response =
		ethosn_dma_alloc(ethosn,
				 sizeof(struct ethosn_queue) +
				 ethosn->queue_size,
				 ETHOSN_PROT_READ | ETHOSN_PROT_WRITE,
				 ETHOSN_STREAM_WORKING_DATA,
				 GFP_KERNEL);
	if (IS_ERR(ethosn->mailbox_response)) {
		dev_warn(ethosn->dev,
			 "Failed to allocate memory for mailbox response queue");

		return PTR_ERR(ethosn->mailbox_response);
	}

	ethosn->mailbox_message =
		devm_kzalloc(ethosn->dev, ethosn->queue_size, GFP_KERNEL);
	if (!ethosn->mailbox_message)
		return -ENOMEM;

	ethosn->num_pongs_received = 0;

	return 0;
}

/**
 * mailbox_free() - Free the mailbox.
 * @ethosn:	Pointer to Ethos-N device.
 */
static void mailbox_free(struct ethosn_device *ethosn)
{
	ethosn_dma_free(ethosn, ETHOSN_STREAM_WORKING_DATA, ethosn->mailbox);
	ethosn_dma_free(ethosn, ETHOSN_STREAM_WORKING_DATA,
			ethosn->mailbox_request);
	ethosn_dma_free(ethosn, ETHOSN_STREAM_WORKING_DATA,
			ethosn->mailbox_response);
	devm_kfree(ethosn->dev, ethosn->mailbox_message);
}

/**
 * streams_init() - Initialize the stream memory regions.
 * @ethosn:	Ethos-N device.
 *
 * Return: 0 on success, else error code.
 */
static int streams_init(struct ethosn_device *ethosn)
{
	int ret;

	ret = ethosn_send_stream_request(ethosn, ETHOSN_STREAM_FIRMWARE);
	if (ret)
		return ret;

	ret = ethosn_send_stream_request(ethosn, ETHOSN_STREAM_WORKING_DATA);
	if (ret)
		return ret;

	ret = ethosn_send_stream_request(ethosn, ETHOSN_STREAM_COMMAND_STREAM);
	if (ret)
		return ret;

	ret = ethosn_send_mpu_enable_request(ethosn);
	if (ret)
		return ret;

	return 0;
}

void ethosn_write_top_reg(struct ethosn_device *ethosn,
			  const u32 page,
			  const u32 offset,
			  const u32 value)
{
	iowrite32(value, ethosn_top_reg_addr(ethosn->top_regs, page, offset));
}

/* Exported for use by ethosn-tests module * */
EXPORT_SYMBOL(ethosn_write_top_reg);

u32 ethosn_read_top_reg(struct ethosn_device *ethosn,
			const u32 page,
			const u32 offset)
{
	return ioread32(ethosn_top_reg_addr(ethosn->top_regs, page, offset));
}

/**
 * boot_firmware() - Boot firmware.
 * @ethosn:	Pointer to Ethos-N device.
 *
 * Return: 0 on success, else error code.
 */
static int boot_firmware(struct ethosn_device *ethosn)
{
	struct dl1_sysctlr0_r sysctlr0 = { .word = 0 };
	struct dl1_sysctlr1_r sysctlr1 = { .word = 0 };
	uint32_t *vtable = ethosn->firmware_vtable->cpu_addr;

	memset(vtable, 0, ethosn->firmware_vtable->size);

	/* Set vtable stack pointer */
	vtable[0] = to_ethosn_addr(ethosn->firmware_stack->iova_addr,
				   &ethosn->work_data_map);
	if (vtable[0] >= (uint32_t)-MAX_ERRNO)
		return (int)vtable[0];

	vtable[0] += ethosn->firmware_stack->size;

	/* Set vtable reset program counter */
	vtable[1] = to_ethosn_addr(ethosn->firmware->iova_addr,
				   &ethosn->firmware_map) + 1;
	if (vtable[1] >= (uint32_t)-MAX_ERRNO)
		return (int)vtable[1];

	ethosn_dma_sync_for_device(ethosn, ethosn->firmware_vtable);

	/* Enable events */
	sysctlr1.bits.mcu_setevnt = 1;
	sysctlr1.bits.mcu_gpevnt = 1;
	ethosn_write_top_reg(ethosn, DL1_RP, DL1_SYSCTLR1, sysctlr1.word);

	/* Set firmware init address and release CPU wait */
	sysctlr0.bits.cpuwait = 0;
	sysctlr0.bits.initvtor =
		to_ethosn_addr(ethosn->firmware_vtable->iova_addr,
			       &ethosn->firmware_map) >>
		SYSCTLR0_INITVTOR_SHIFT;
	ethosn_write_top_reg(ethosn, DL1_RP, DL1_SYSCTLR0, sysctlr0.word);

	return 0;
}

int ethosn_reset_and_start_ethosn(struct ethosn_device *ethosn)
{
	int timeout;
	int ret;

	dev_info(ethosn->dev, "Reset the ethosn\n");

	/* Reset the Ethos-N */
	ret = ethosn_reset(ethosn);
	if (ret)
		return ret;

	/* Enable clock */
	ethosn_set_power_ctrl(ethosn, true);

	/* Set MMU Stream id0 if iommu is present */
	if (iommu_present(ethosn->dev->bus)) {
		ret = ethosn_set_mmu_stream_id(ethosn);
		if (ret)
			return ret;
	}

	/* Configure address extension for stream 0, 1 and 2 */
	ret = ethosn_set_addr_ext(
		ethosn, ETHOSN_STREAM_FIRMWARE,
		ethosn_dma_get_addr_base(ethosn, ETHOSN_STREAM_FIRMWARE),
		&ethosn->firmware_map);
	if (ret)
		return ret;

	ret = ethosn_set_addr_ext(
		ethosn, ETHOSN_STREAM_WORKING_DATA,
		ethosn_dma_get_addr_base(ethosn, ETHOSN_STREAM_WORKING_DATA),
		&ethosn->work_data_map);
	if (ret)
		return ret;

	ret = ethosn_set_addr_ext(
		ethosn, ETHOSN_STREAM_COMMAND_STREAM,
		ethosn_dma_get_addr_base(ethosn, ETHOSN_STREAM_COMMAND_STREAM),
		&ethosn->dma_map);
	if (ret)
		return ret;

	if (ethosn->force_firmware_level_interrupts)
		ethosn_write_top_reg(ethosn, DL1_RP, GP_IRQ, 1);

	/* Initialize the mailbox */
	ret = mailbox_init(ethosn);
	if (ret)
		return ret;

	/* Boot the firmware */
	ret = boot_firmware(ethosn);
	if (ret)
		return ret;

	/* Init streams regions */
	ret = streams_init(ethosn);
	if (ret != 0)
		return ret;

	/* Ping firmware */
	ret = ethosn_send_ping(ethosn);
	if (ret != 0)
		return ret;

	/* Send FW and HW capabilities request */
	ret = ethosn_send_fw_hw_capabilities_request(ethosn);
	if (ret != 0)
		return ret;

	/* Set FW's profiling state. This is also set whenever profiling is
	 * enabled/disabled, but we need to do it on each reboot in case
	 * the firmware crashes, so that its profiling state is restored.
	 */
	ret = ethosn_configure_firmware_profiling(ethosn,
						  &ethosn->profiling.config);
	if (ret != 0)
		return ret;

	dev_info(ethosn->dev, "Waiting for Ethos-N\n");

	/* Wait for firmware to set GP2 to 0 which indicates that it has booted.
	 * Also wait for it to reply with the FW & HW caps message.
	 * This is necessary so that the user can't query us for the caps before
	 * they are ready.
	 * Also wait for the memory regions to be correctly setup. This is
	 * necessary to execute inferences.
	 */
	for (timeout = 0; timeout < ETHOSN_RESET_TIMEOUT_US;
	     timeout += ETHOSN_RESET_WAIT_US) {
		bool mem_ready = ethosn->ethosn_f_stream_configured &&
				 ethosn->ethosn_wd_stream_configured &&
				 ethosn->ethosn_cs_stream_configured &&
				 ethosn->ethosn_mpu_enabled;

		if (ethosn_read_top_reg(ethosn, DL1_RP, GP_MAILBOX) == 0 &&
		    ethosn->fw_and_hw_caps.size > 0 && mem_ready)
			break;

		udelay(ETHOSN_RESET_WAIT_US);
	}

	if (timeout >= ETHOSN_RESET_TIMEOUT_US) {
		dev_err(ethosn->dev, "Timeout while waiting for Ethos-N\n");

		return -ETIME;
	}

	ethosn->firmware_running = true;

	return 0;
}

void ethosn_notify_firmware(struct ethosn_device *ethosn)
{
	struct dl1_setirq_int_r irq = {
		.bits          = {
			.event = 1,
		}
	};

	ethosn_write_top_reg(ethosn, DL1_RP, DL1_SETIRQ_INT,
			     irq.word);
}

static int ethosn_hard_reset(struct ethosn_device *ethosn)
{
	struct dl1_sysctlr0_r sysctlr0 = { .word = 0 };
	unsigned int timeout;

	dev_info(ethosn->dev, "Hard reset the hardware.\n");

	/* Initiate hard reset */
	sysctlr0.bits.hard_rstreq = 1;
	ethosn_write_top_reg(ethosn, DL1_RP, DL1_SYSCTLR0, sysctlr0.word);

	/* Wait for hard reset to complete */
	for (timeout = 0; timeout < ETHOSN_RESET_TIMEOUT_US;
	     timeout += ETHOSN_RESET_WAIT_US) {
		sysctlr0.word =
			ethosn_read_top_reg(ethosn, DL1_RP, DL1_SYSCTLR0);

		if (sysctlr0.bits.hard_rstreq == 0)
			break;

		udelay(ETHOSN_RESET_WAIT_US);
	}

	if (timeout >= ETHOSN_RESET_TIMEOUT_US) {
		dev_err(ethosn->dev, "Failed to hard reset the hardware.\n");

		return -EFAULT;
	}

	return 0;
}

static int ethosn_soft_reset(struct ethosn_device *ethosn)
{
	struct dl1_sysctlr0_r sysctlr0 = { .word = 0 };
	unsigned int timeout;

	dev_info(ethosn->dev, "Soft reset the hardware.\n");

	/* Soft reset, block new AXI requests */
	sysctlr0.bits.soft_rstreq = 3;
	ethosn_write_top_reg(ethosn, DL1_RP, DL1_SYSCTLR0, sysctlr0.word);

	/* Wait for reset to complete */
	for (timeout = 0; timeout < ETHOSN_RESET_TIMEOUT_US;
	     timeout += ETHOSN_RESET_WAIT_US) {
		sysctlr0.word =
			ethosn_read_top_reg(ethosn, DL1_RP, DL1_SYSCTLR0);

		if (sysctlr0.bits.soft_rstreq == 0)
			break;

		udelay(ETHOSN_RESET_WAIT_US);
	}

	if (timeout >= ETHOSN_RESET_TIMEOUT_US) {
		dev_warn(ethosn->dev,
			 "Failed to soft reset the hardware. sysctlr0=0x%08x\n",
			 sysctlr0.word);

		return -ETIME;
	}

	return 0;
}

int ethosn_reset(struct ethosn_device *ethosn)
{
	int ret = -EINVAL;

	ret = ethosn_soft_reset(ethosn);
	if (ret)
		ret = ethosn_hard_reset(ethosn);

	return ret;
}

void ethosn_set_power_ctrl(struct ethosn_device *ethosn,
			   bool clk_on)
{
	struct dl1_pwrctlr_r pwrctlr = { .word = 0 };

	pwrctlr.bits.active = clk_on;
	ethosn_write_top_reg(ethosn, DL1_RP, DL1_PWRCTLR, pwrctlr.word);
}

/**
 * ethosn_set_mmu_stream_id() - Configure the mmu stream id0.
 * @ethosn:- Pointer to the Ethos-N device
 *
 * Return: Negative error code on error, zero otherwise
 */
int ethosn_set_mmu_stream_id(struct ethosn_device *ethosn)
{
	struct iommu_fwspec *fwspec = ethosn->dev->iommu_fwspec;
	int ret = -EINVAL;
	unsigned int stream_id;
	static const int mmusid_0 = DL1_STREAM0_MMUSID;

	/*
	 * Currently, it is permitted to define only one stream id in the dts
	 * file. There is no advantage of defining multiple stream ids when
	 * the device uses all the streams at almost all the times.
	 */
	if (fwspec->num_ids > 1) {
		dev_err(ethosn->dev,
			"Support for multiple streams for a single device is not allowed\n");

		return ret;
	}

	stream_id = fwspec->ids[0];

	/*
	 * The value of stream id fetched from the dts is used to program the
	 * STREAM0_MMUSID register. The other stream id registers are programmed
	 * based on this value in the firmware.
	 */
	ethosn_write_top_reg(ethosn, DL1_RP, mmusid_0, stream_id);
	ethosn_write_top_reg(ethosn, DL1_RP, GP_MMUSID0, stream_id);

	return 0;
}

/**
 * ethosn_set_addr_ext() - Configure address extension for stream
 * @ethosn: The Ethos-N device
 * @stream: stream to configure (must be 0-2)
 * @offset: Offset to apply. Lower 29 bits will be ignored
 * @addr_map: Address map to be used later to convert to Ethos-N addresses
 *
 * Return: Negative error code on error, zero otherwise
 */
int ethosn_set_addr_ext(struct ethosn_device *ethosn,
			unsigned int stream,
			ethosn_address_t offset,
			struct ethosn_addr_map *addr_map)
{
	/*
	 * Program the STREAM0_ADDRESS_EXTEND register. And program the values
	 * for STREAM1_ADDRESS_EXTEND and STREAM2_ADDRESS_EXTEND registers in
	 * GP registers (at indices GP_STREAM1_ADDRESS_EXTEND,
	 * GP_STREAM2_ADDRESS_EXTEND) respectively. The firmware (during bootup)
	 * will read GP1 and GP2 and program STREAM1_ADDRESS_EXTEND and
	 * STREAM2_ADDRESS_EXTEND registers.
	 */
	static const int stream_to_page[] = {
		DL1_STREAM0_ADDRESS_EXTEND,
		GP_STREAM1_ADDRESS_EXTEND,
		GP_STREAM2_ADDRESS_EXTEND,
	};
	static const u32 stream_to_offset[] = {
		0,
		REGION_EXT_RAM0 << REGION_SHIFT,
		REGION_EXT_RAM1 << REGION_SHIFT,
	};
	struct dl1_stream0_address_extend_r ext = { .word = 0 };

	BUILD_BUG_ON(ARRAY_SIZE(stream_to_page) !=
		     ARRAY_SIZE(stream_to_offset));

	if (stream >= ARRAY_SIZE(stream_to_page)) {
		dev_err(ethosn->dev,
			"Illegal stream %u for address extension.\n",
			stream);

		return -EFAULT;
	}

	ext.bits.addrextend = offset >> REGION_SHIFT;

	ethosn_write_top_reg(ethosn, DL1_RP, stream_to_page[stream],
			     ext.word);

	if (addr_map) {
		addr_map->region = stream_to_offset[stream];
		addr_map->extension = offset & ~ETHOSN_REGION_MASK;
	}

	return 0;
}

static int get_gp_offset(struct ethosn_device *ethosn,
			 unsigned int index)
{
	static const int index_to_offset[] = {
		DL1_GP0,
		DL1_GP1,
		DL1_GP2,
		DL1_GP3,
		DL1_GP4,
		DL1_GP5,
		DL1_GP6,
		DL1_GP7
	};

	if (index >= ARRAY_SIZE(index_to_offset)) {
		dev_err(ethosn->dev,
			"Illegal index %u of general purpose register.\n",
			index);

		return -EFAULT;
	}

	return index_to_offset[index];
}

void ethosn_dump_gps(struct ethosn_device *ethosn)
{
	int offset;
	unsigned int i;

	for (i = 0; i < 8; i++) {
		offset = get_gp_offset(ethosn, i);
		if (offset < 0)
			break;

		dev_info(ethosn->dev,
			 "GP%u=0x%08x\n",
			 i, ethosn_read_top_reg(ethosn, DL1_RP, offset));
	}
}

/****************************************************************************
 * Mailbox
 ****************************************************************************/

/**
 * ethosn_read_message() - Read message from queue.
 * @queue:	Pointer to queue.
 * @header:	Pointer to message header.
 * @data:	Pointer to data buffer.
 * @length:	Maximum length of data buffer.
 *
 * Return: Number of messages read on success, else error code.
 */
int ethosn_read_message(struct ethosn_device *ethosn,
			struct ethosn_message_header *header,
			void *data,
			size_t length)
{
	struct ethosn_queue *queue = ethosn->mailbox_response->cpu_addr;
	bool ret;
	uint32_t read_pending;

	if (ethosn->mailbox_response->size <
	    (sizeof(*queue) + queue->capacity) ||
	    !is_power_of_2(queue->capacity)) {
		dev_err(ethosn->dev,
			"Illegal mailbox queue capacity. alloc_size=%zu, queue capacity=%u\n",
			ethosn->mailbox_request->size, queue->capacity);

		return -EFAULT;
	}

	ethosn_dma_sync_for_cpu(ethosn, ethosn->mailbox_response);

	ret = ethosn_queue_read(queue, (uint8_t *)header,
				sizeof(struct ethosn_message_header),
				&read_pending);
	if (!ret)
		return 0;

	/*
	 * It's possible that the "writing" side (e.g. CU firmware) has written
	 * the header but hasn't yet written the payload. In this case we give
	 * up and will try again once the "writing" side sends the interrupt to
	 * indicate it has finished writing the whole message.
	 */
	if ((ethosn_queue_get_size(queue) -
	     sizeof(struct ethosn_message_header)) <
	    header->length)
		return 0;

	queue->read = read_pending;

	dev_dbg(ethosn->dev,
		"Received message. type=%u, length=%u, read=%u, write=%u.\n",
		header->type, header->length, queue->read,
		queue->write);

	if (length < header->length) {
		dev_warn(ethosn->dev,
			 "Message too large to read. header.length=%u, length=%zu.\n",
			 header->length, length);

		ethosn_queue_skip(queue, header->length);

		return -ENOMEM;
	}

	ret = ethosn_queue_read(queue, data, header->length, &read_pending);
	if (!ret) {
		dev_err(ethosn->dev,
			"Failed to read message payload. size=%u, queue capacity=%u\n",
			header->length, queue->capacity);

		return -EFAULT;
	}

	queue->read = read_pending;

	ethosn_dma_sync_for_device(ethosn, ethosn->mailbox_response);

	ethosn_log_firmware(ethosn, ETHOSN_LOG_FIRMWARE_INPUT, header, data);
	if (ethosn->profiling.config.enable_profiling)
		++ethosn->profiling.mailbox_messages_received;

	return 1;
}

/**
 * ethosn_write_message() - Write message to queue.
 * @queue:	Pointer to queue.
 * @type:	Message type.
 * @data:	Pointer to data buffer.
 * @length:	Length of data buffer.
 *
 * Return: 0 on success, else error code.
 */
int ethosn_write_message(struct ethosn_device *ethosn,
			 enum ethosn_message_type type,
			 void *data,
			 size_t length)
{
	struct ethosn_queue *queue = ethosn->mailbox_request->cpu_addr;
	struct ethosn_message_header header = {
		.type   = type,
		.length = length
	};
	bool ret;
	uint32_t write_pending;

	if (ethosn->mailbox_response->size <
	    (sizeof(*queue) + queue->capacity) ||
	    !is_power_of_2(queue->capacity)) {
		dev_err(ethosn->dev,
			"Illegal mailbox queue capacity. alloc_size=%zu, queue capacity=%u\n",
			ethosn->mailbox_request->size, queue->capacity);

		return -EFAULT;
	}

	ethosn_dma_sync_for_cpu(ethosn, ethosn->mailbox_request);

	dev_dbg(ethosn->dev,
		"Write message. type=%u, length=%zu, read=%u, write=%u.\n",
		type, length, queue->read, queue->write);

	ret =
		ethosn_queue_write(queue, (uint8_t *)&header,
				   sizeof(struct ethosn_message_header),
				   &write_pending);
	if (!ret)
		return ret;

	/*
	 * Sync the payload before committing the updated write pointer so that
	 * the "reading" side (e.g. CU firmware) can't read invalid data.
	 */
	ethosn_dma_sync_for_device(ethosn, ethosn->mailbox_request);
	queue->write = write_pending;

	ret = ethosn_queue_write(queue, data, length, &write_pending);
	if (!ret)
		return ret;

	/*
	 * Sync the payload before committing the updated write pointer so that
	 * the "reading" side (e.g. CU firmware) can't read invalid data.
	 */
	ethosn_dma_sync_for_device(ethosn, ethosn->mailbox_request);
	queue->write = write_pending;
	/* Sync the write pointer */
	ethosn_dma_sync_for_device(ethosn, ethosn->mailbox_request);
	ethosn_notify_firmware(ethosn);

	ethosn_log_firmware(ethosn, ETHOSN_LOG_FIRMWARE_OUTPUT, &header, data);
	if (ethosn->profiling.config.enable_profiling)
		++ethosn->profiling.mailbox_messages_sent;

	return 0;
}

/* Exported for use by ethosn-tests module * */
EXPORT_SYMBOL(ethosn_write_message);

int ethosn_send_fw_hw_capabilities_request(struct ethosn_device *ethosn)
{
	dev_dbg(ethosn->dev, "-> FW & HW Capabilities\n");

	return ethosn_write_message(ethosn, ETHOSN_MESSAGE_FW_HW_CAPS_REQUEST,
				    NULL, 0);
}

/*
 * Note we do not use the profiling config in ethosn->profiling, because if we
 * are in the process of updating that, it may not yet have been committed.
 * Instead we take the arguments explicitly.
 */
static int ethosn_send_configure_profiling(struct ethosn_device *ethosn,
					   bool enable,
					   uint32_t num_hw_counters,
					   enum
					   ethosn_profiling_hw_counter_types *
					   hw_counters,
					   struct ethosn_dma_info *buffer)
{
	struct ethosn_firmware_profiling_configuration fw_new_config;
	int i;

	fw_new_config.enable_profiling = enable;

	if (!IS_ERR_OR_NULL(buffer)) {
		fw_new_config.buffer_size = buffer->size;
		fw_new_config.buffer_address =
			to_ethosn_addr(buffer->iova_addr,
				       &ethosn->work_data_map);
		if (IS_ERR_VALUE((unsigned long)fw_new_config.buffer_address)) {
			dev_err(ethosn->dev,
				"Error converting firmware profiling buffer to_ethosn_addr.\n");

			return -EFAULT;
		}

		fw_new_config.num_hw_counters = num_hw_counters;
		for (i = 0; i < num_hw_counters; ++i)
			fw_new_config.hw_counters[i] = hw_counters[i];
	} else {
		fw_new_config.buffer_address = 0;
		fw_new_config.buffer_size = 0;
	}

	dev_dbg(ethosn->dev,
		"-> ETHOSN_MESSAGE_CONFIGURE_PROFILING, enable_profiling=%d, buffer_address=0x%08llx, buffer_size=%d\n",
		fw_new_config.enable_profiling, fw_new_config.buffer_address,
		fw_new_config.buffer_size);

	return ethosn_write_message(ethosn, ETHOSN_MESSAGE_CONFIGURE_PROFILING,
				    &fw_new_config, sizeof(fw_new_config));
}

int ethosn_configure_firmware_profiling(struct ethosn_device *ethosn,
					struct ethosn_profiling_config *
					new_config)
{
	int ret;

	/* If we are already waiting for the firmware to acknowledge use of a
	 * new buffer then we cannot allocate another.
	 * We must wait for it to acknowledge first.
	 */
	if (ethosn->profiling.is_waiting_for_firmware_ack) {
		dev_err(ethosn->dev,
			"Already waiting for firmware to acknowledge new profiling config.\n");

		return -EINVAL;
	}

	/* Allocate new profiling buffer.
	 * Note we do not overwrite the existing buffer yet, as the firmware may
	 * still be using it
	 */
	if (new_config->enable_profiling &&
	    new_config->firmware_buffer_size > 0) {
		struct ethosn_profiling_buffer *buffer;

		ethosn->profiling.firmware_buffer_pending =
			ethosn_dma_alloc(ethosn,
					 new_config->firmware_buffer_size,
					 ETHOSN_PROT_READ | ETHOSN_PROT_WRITE,
					 ETHOSN_STREAM_WORKING_DATA,
					 GFP_KERNEL);
		if (IS_ERR(ethosn->profiling.firmware_buffer_pending)) {
			dev_err(ethosn->dev,
				"Error allocating firmware profiling buffer.\n");
			ret = PTR_ERR(
				ethosn->profiling.firmware_buffer_pending);

			return ret;
		}

		/* Initialize the firmware_write_index. */
		buffer =
			(struct ethosn_profiling_buffer *)
			ethosn->profiling.firmware_buffer_pending->cpu_addr;
		buffer->firmware_write_index = 0;
		ethosn_dma_sync_for_device(
			ethosn,
			ethosn->profiling.firmware_buffer_pending);
	} else {
		ethosn->profiling.firmware_buffer_pending = NULL;
	}

	ethosn->profiling.is_waiting_for_firmware_ack = true;

	ret = ethosn_send_configure_profiling(
		ethosn, new_config->enable_profiling,
		new_config->num_hw_counters,
		new_config->hw_counters,
		ethosn->profiling.firmware_buffer_pending);
	if (ret != 0) {
		dev_err(ethosn->dev,
			"ethosn_send_configure_profiling failed.\n");
		ethosn_dma_free(ethosn, ETHOSN_STREAM_WORKING_DATA,
				ethosn->profiling.firmware_buffer_pending);
		ethosn->profiling.firmware_buffer_pending = NULL;

		return ret;
	}

	return 0;
}

int ethosn_configure_firmware_profiling_ack(struct ethosn_device *ethosn)
{
	if (!ethosn->profiling.is_waiting_for_firmware_ack) {
		dev_err(ethosn->dev,
			"Unexpected configure profiling ack from firmware.\n");

		return -EINVAL;
	}

	/* We can now free the old buffer (if any), as we know the firmware is
	 * no longer writing to it
	 */
	ethosn_dma_free(ethosn, ETHOSN_STREAM_WORKING_DATA,
			ethosn->profiling.firmware_buffer);

	/* What used to be the pending buffer is now the proper one. */
	ethosn->profiling.firmware_buffer =
		ethosn->profiling.firmware_buffer_pending;
	ethosn->profiling.firmware_buffer_pending = NULL;
	ethosn->profiling.is_waiting_for_firmware_ack = false;

	return 0;
}

int ethosn_send_time_sync(struct ethosn_device *ethosn)
{
	struct ethosn_message_time_sync_request request;
	struct timespec res;

	getnstimeofday(&res);

	dev_dbg(ethosn->dev, "-> Time Sync\n");

	request.timestamp = res.tv_sec * NANOSECONDS_IN_A_SECOND + res.tv_nsec;

	return ethosn_write_message(ethosn, ETHOSN_MESSAGE_TIME_SYNC, &request,
				    sizeof(request));
}

int ethosn_send_ping(struct ethosn_device *ethosn)
{
	dev_dbg(ethosn->dev, "-> Ping\n");

	return ethosn_write_message(ethosn, ETHOSN_MESSAGE_PING, NULL, 0);
}

int ethosn_send_inference(struct ethosn_device *ethosn,
			  dma_addr_t buffer_array,
			  uint64_t user_arg)
{
	struct ethosn_message_inference_request request;

	request.buffer_array = to_ethosn_addr(buffer_array, &ethosn->dma_map);
	request.user_argument = user_arg;

	dev_dbg(ethosn->dev,
		"-> Inference. buffer_array=0x%08llx, user_args=0x%llx\n",
		request.buffer_array, request.user_argument);

	return ethosn_write_message(ethosn, ETHOSN_MESSAGE_INFERENCE_REQUEST,
				    &request,
				    sizeof(request));
}

int ethosn_send_stream_request(struct ethosn_device *ethosn,
			       enum ethosn_stream_id stream_id)
{
	struct ethosn_message_stream_request request;

	request.stream_id = stream_id;
	request.size = ethosn_dma_get_addr_size(ethosn, stream_id);
	if (request.size == 0)
		return -EFAULT;

	dev_dbg(ethosn->dev,
		"-> Stream=%u. size=0x%x", request.stream_id,
		request.size);

	return ethosn_write_message(ethosn, ETHOSN_MESSAGE_STREAM_REQUEST,
				    &request,
				    sizeof(request));
}

int ethosn_send_mpu_enable_request(struct ethosn_device *ethosn)
{
	dev_dbg(ethosn->dev,
		"-> Mpu enable.");

	return ethosn_write_message(ethosn, ETHOSN_MESSAGE_MPU_ENABLE_REQUEST,
				    NULL, 0);
}

/****************************************************************************
 * Firmware
 ****************************************************************************/

/**
 * firmware_load - Load firmware binary with given name.
 * @ethosn:		Pointer to Ethos-N device.
 * @firmware_name:	Name of firmware binary.
 *
 * Return: 0 on success, else error code.
 */
static int firmware_load(struct ethosn_device *ethosn,
			 const char *firmware_name)
{
	const struct firmware *fw;
	size_t size;
	int ret;

	/* Request firmware binary */
	ret = request_firmware(&fw, firmware_name, ethosn->dev);
	if (ret)
		return ret;

	/* Make sure code size is at least 256 KB */
	size = max_t(size_t, ETHOSN_CODE_SIZE, fw->size);

	/* Allocate memory for firmware code */
	ethosn->firmware =
		ethosn_dma_alloc(ethosn, size,
				 ETHOSN_PROT_READ | ETHOSN_PROT_WRITE,
				 ETHOSN_STREAM_FIRMWARE, GFP_KERNEL);
	if (IS_ERR(ethosn->firmware)) {
		ret = PTR_ERR(ethosn->firmware);
		goto release_fw;
	}

	memcpy(ethosn->firmware->cpu_addr, fw->data, fw->size);
	ethosn_dma_sync_for_device(ethosn, ethosn->firmware);

	/* Allocate stack */
	ethosn->firmware_stack =
		ethosn_dma_alloc(ethosn, ETHOSN_STACK_SIZE,
				 ETHOSN_PROT_READ | ETHOSN_PROT_WRITE,
				 ETHOSN_STREAM_WORKING_DATA,
				 GFP_KERNEL);
	if (IS_ERR(ethosn->firmware_stack)) {
		ret = PTR_ERR(ethosn->firmware_stack);
		goto free_firmware;
	}

	ethosn_dma_sync_for_device(ethosn, ethosn->firmware_stack);

	/* Allocate vtable */
	ethosn->firmware_vtable =
		ethosn_dma_alloc(ethosn, ETHOSN_VTABLE_SIZE * sizeof(uint32_t),
				 ETHOSN_PROT_READ | ETHOSN_PROT_WRITE,
				 ETHOSN_STREAM_FIRMWARE,
				 GFP_KERNEL);
	if (IS_ERR(ethosn->firmware_vtable)) {
		ret = PTR_ERR(ethosn->firmware_vtable);
		goto free_stack;
	}

	release_firmware(fw);

	return 0;

free_stack:
	ethosn_dma_free(ethosn, ETHOSN_STREAM_WORKING_DATA,
			ethosn->firmware_stack);
free_firmware:
	ethosn_dma_free(ethosn, ETHOSN_STREAM_FIRMWARE, ethosn->firmware);
release_fw:
	release_firmware(fw);

	return ret;
}

/**
 * firmware_init - Allocate and initialize firmware.
 * @ethosn:		Pointer to Ethos-N device.
 *
 * Try to load firmware binaries in given order.
 *
 * Return: 0 on success, else error code.
 */
static int firmware_init(struct ethosn_device *ethosn)
{
	static const char *const firmware_names[] = {
		"ethosn.bin"
	};
	int i;
	int ret;

	for (i = 0; i < ARRAY_SIZE(firmware_names); i++) {
		ret = firmware_load(ethosn, firmware_names[i]);
		if (!ret)
			break;
	}

	if (ret) {
		dev_err(ethosn->dev, "No firmware found.\n");

		return ret;
	}

	return 0;
}

/**
 * firmware_deinit - Free firmware resources.
 * @ethosn:		Pointer to Ethos-N device.
 */
static void firmware_deinit(struct ethosn_device *ethosn)
{
	ethosn_dma_free(ethosn, ETHOSN_STREAM_FIRMWARE, ethosn->firmware);
	ethosn_dma_free(ethosn, ETHOSN_STREAM_WORKING_DATA,
			ethosn->firmware_stack);
	ethosn_dma_free(ethosn, ETHOSN_STREAM_FIRMWARE,
			ethosn->firmware_vtable);
}

/****************************************************************************
 * Debugfs
 ****************************************************************************/

/**
 * mailbox_fops_read - Mailbox read file operation.
 * @file:		File handle.
 * @buf_user:		User space buffer.
 * @count:		Size of user space buffer.
 * @position:		Current file position.
 *
 * Return: Number of bytes read, else error code.
 */
static ssize_t mailbox_fops_read(struct file *file,
				 char __user *buf_user,
				 size_t count,
				 loff_t *position)
{
	struct ethosn_device *ethosn = file->f_inode->i_private;
	char buf[200];
	size_t n = 0;
	int ret;

	ret = mutex_lock_interruptible(&ethosn->mutex);
	if (ret)
		return ret;

	if (ethosn->mailbox_request) {
		struct ethosn_queue *queue = ethosn->mailbox_request->cpu_addr;

		ethosn_dma_sync_for_cpu(ethosn, ethosn->mailbox_request);

		n += scnprintf(&buf[n], sizeof(buf) - n,
			       "Request queue : %llx\n",
			       ethosn->mailbox_request->iova_addr);
		n += scnprintf(&buf[n], sizeof(buf) - n, "    capacity  : %u\n",
			       queue->capacity);
		n += scnprintf(&buf[n], sizeof(buf) - n, "    read      : %u\n",
			       queue->read);
		n += scnprintf(&buf[n], sizeof(buf) - n, "    write     : %u\n",
			       queue->write);
	}

	if (ethosn->mailbox_response) {
		struct ethosn_queue *queue = ethosn->mailbox_response->cpu_addr;

		ethosn_dma_sync_for_cpu(ethosn, ethosn->mailbox_response);

		n += scnprintf(&buf[n], sizeof(buf) - n,
			       "Response queue: %llx\n",
			       ethosn->mailbox_response->iova_addr);
		n += scnprintf(&buf[n], sizeof(buf) - n, "    capacity  : %u\n",
			       queue->capacity);
		n += scnprintf(&buf[n], sizeof(buf) - n, "    read      : %u\n",
			       queue->read);
		n += scnprintf(&buf[n], sizeof(buf) - n, "    write     : %u\n",
			       queue->write);
	}

	if (ethosn->mailbox) {
		struct ethosn_mailbox *mailbox = ethosn->mailbox->cpu_addr;

		ethosn_dma_sync_for_cpu(ethosn, ethosn->mailbox);

		n += scnprintf(&buf[n], sizeof(buf) - n, "Severity      : %u\n",
			       mailbox->severity);
	}

	mutex_unlock(&ethosn->mutex);

	return simple_read_from_buffer(buf_user, count, position, buf, n);
}

/**
 * firmware_profiling_read - Called when a userspace process reads the
 *			     firmware_profiling debugfs entry,
 *                           to retrieve profiling entries.
 *
 * The kernel maintains the user's fd offset as normal and this function
 * handles mapping that offset into the circular buffer.
 * It is not possible for the fd read offset to "overtake" the firmware's
 * write pointer (the function prevents it) - this means userspace
 * can never read into uninitialised data or read older entries that it has
 * already seen.
 * When the fd offset reaches the size of the buffer, it will keep increasing
 * beyond the size, but read operations will interpret this as modulo the
 * buffer size. There is no mechanism in place to prevent the firmware
 * write pointer from overtaking any of the userspace fd offsets
 * (which is deliberate - we don't want to stall the firmware based on any
 * user-space processes not reading profiling data fast enough).
 * This means that it is possible for a process reading from the fd to observe
 * a "skip" in the data if it is not reading it fast enough.
 *
 * @file:		File handle.
 * @buf_user:		User space buffer.
 * @count:		Size of user space buffer.
 * @position:		Current file position.
 *
 * Return: Number of bytes read, else error code.
 */
static ssize_t firmware_profiling_read(struct file *file,
				       char __user *buf_user,
				       size_t count,
				       loff_t *position)
{
	struct ethosn_device *ethosn = file->f_inode->i_private;
	ssize_t ret;
	ssize_t num_bytes_read;
	size_t buffer_entries_offset;
	size_t buffer_entries_count;
	size_t buffer_entries_size_bytes;
	loff_t read_buffer_offset;
	struct ethosn_profiling_buffer *buffer;
	uint32_t firmware_write_offset;

	/* Make sure the profiling buffer isn't deallocated underneath us */
	ret = mutex_lock_interruptible(&ethosn->mutex);
	if (ret != 0)
		return ret;

	/* Report error if profiling is not enabled (i.e. no profiling buffer
	 * allocated)
	 */
	if (IS_ERR_OR_NULL(ethosn->profiling.firmware_buffer)) {
		ret = -EINVAL;
		goto cleanup;
	}

	/* Calculate size etc. of the buffer. */
	buffer =
		(struct ethosn_profiling_buffer *)
		ethosn->profiling.firmware_buffer->cpu_addr;

	buffer_entries_offset =
		offsetof(struct ethosn_profiling_buffer, entries);
	buffer_entries_count =
		(ethosn->profiling.config.firmware_buffer_size -
		 buffer_entries_offset) /
		sizeof(struct ethosn_profiling_entry);
	buffer_entries_size_bytes = buffer_entries_count *
				    sizeof(struct ethosn_profiling_entry);

	/* Convert from file offset to position in the buffer.
	 * This accounts for the fact that the buffer is circular so the file
	 * offset may be larger than the actual buffer size.
	 */
	read_buffer_offset = *position % buffer_entries_size_bytes;

	/* Copy firmware_write_index as the firmware may write to this in the
	 * background.
	 */
	firmware_write_offset = buffer->firmware_write_index *
				sizeof(struct ethosn_profiling_entry);

	if (read_buffer_offset < firmware_write_offset) {
		/* Firmware has written data further down the buffer, but not
		 * enough to wrap around.
		 */
		num_bytes_read = simple_read_from_buffer(buf_user, count,
							 &read_buffer_offset,
							 buffer->entries,
							 firmware_write_offset);
	} else if (read_buffer_offset > firmware_write_offset) {
		/* Firmware has written data further down the buffer and then
		 * wrapped around.
		 * First read the remaining data at the bottom of the buffer,
		 * all the way to the end.
		 */
		num_bytes_read = simple_read_from_buffer(
			buf_user, count, &read_buffer_offset,
			buffer->entries, buffer_entries_size_bytes);

		/* Then, if the user buffer has any space left, continue
		 * reading data from the top of the buffer.
		 */
		if (num_bytes_read > 0 && num_bytes_read < count) {
			read_buffer_offset = 0;
			num_bytes_read += simple_read_from_buffer(
				buf_user + num_bytes_read,
				count - num_bytes_read, &read_buffer_offset,
				buffer->entries, firmware_write_offset);
		}
	} else {
		/* No more data available (or the firmware has written so much
		 * that it has wrapped around to exactly where it was)
		 */
		num_bytes_read = 0;
	}

	ret = num_bytes_read;

	/* Update user's file offset */
	if (num_bytes_read > 0)
		*position += num_bytes_read;

cleanup:
	mutex_unlock(&ethosn->mutex);

	return ret;
}

static void dfs_deinit(struct ethosn_device *ethosn)
{
	debugfs_remove_recursive(ethosn->debug_dir);
}

static void dfs_init(struct ethosn_device *ethosn)
{
	static const struct debugfs_reg32 regs[] = {
		REGSET32(SYSCTLR0),
		REGSET32(SYSCTLR1),
		REGSET32(PWRCTLR),
		REGSET32(CLRIRQ_EXT),
		REGSET32(SETIRQ_INT),
		REGSET32(IRQ_STATUS),
		REGSET32(GP0),
		REGSET32(GP1),
		REGSET32(GP2),
		REGSET32(GP3),
		REGSET32(GP4),
		REGSET32(GP5),
		REGSET32(GP6),
		REGSET32(GP7),
		REGSET32(STREAM0_ADDRESS_EXTEND),
		REGSET32(NPU_ID),
		REGSET32(UNIT_COUNT),
		REGSET32(MCE_FEATURES),
		REGSET32(DFC_FEATURES),
		REGSET32(PLE_FEATURES),
		REGSET32(WD_FEATURES),
		REGSET32(ECOID)
	};
	static const struct file_operations mailbox_fops = {
		.owner = THIS_MODULE,
		.read  = &mailbox_fops_read
	};
	static const struct file_operations firmware_profiling_fops = {
		.owner = THIS_MODULE,
		.read  = &firmware_profiling_read
	};
	char name[16];

	/* Create debugfs directory */
	snprintf(name, sizeof(name), "ethosn%u", ethosn->dev->id);
	ethosn->debug_dir = debugfs_create_dir(name, NULL);
	if (IS_ERR_OR_NULL(ethosn->debug_dir))
		return;

	/* Register map */
	ethosn->debug_regset.regs = regs;
	ethosn->debug_regset.nregs = ARRAY_SIZE(regs);
	ethosn->debug_regset.base = ethosn->top_regs;
	debugfs_create_regset32("registers", 0400, ethosn->debug_dir,
				&ethosn->debug_regset);

	/* Mailbox */
	debugfs_create_file("mailbox", 0400, ethosn->debug_dir, ethosn,
			    &mailbox_fops);

	/* Expose the firmware's profiling stream to user-space as a file. */
	debugfs_create_file("firmware_profiling", 0400, ethosn->debug_dir,
			    ethosn,
			    &firmware_profiling_fops);
}

/****************************************************************************
 * Device setup
 ****************************************************************************/

int ethosn_device_init(struct ethosn_device *ethosn)
{
	int ret;

	/* Round up queue size to next power of 2 */
	ethosn->queue_size = roundup_pow_of_two(ethosn_queue_size);

	/* Initialize debugfs */
	dfs_init(ethosn);

	/* Initialize log */
	ret = ethosn_log_init(ethosn);
	if (ret)
		goto remove_debufs;

	/* Load the firmware */
	ret = firmware_init(ethosn);
	if (ret)
		goto deinit_log;

	/* Allocate the mailbox structure */
	ret = mailbox_alloc(ethosn);
	if (ret)
		goto deinit_firmware;

	ethosn_global_device_for_testing = ethosn;

	/* Completed the device initialization */
	atomic_set(&ethosn->init_done, 1);

	return 0;

deinit_firmware:
	firmware_deinit(ethosn);

deinit_log:
	ethosn_log_deinit(ethosn);

remove_debufs:
	dfs_deinit(ethosn);

	return ret;
}

void ethosn_device_deinit(struct ethosn_device *ethosn)
{
	int ret;

	ret = mutex_lock_interruptible(&ethosn->mutex);
	if (ret)
		return;

	/* Started the device de-initialization */
	atomic_set(&ethosn->init_done, 0);

	ethosn_global_device_for_testing = NULL;

	ethosn_hard_reset(ethosn);
	firmware_deinit(ethosn);
	mailbox_free(ethosn);
	ethosn_log_deinit(ethosn);
	dfs_deinit(ethosn);
	mutex_unlock(&ethosn->mutex);
	if (ethosn->fw_and_hw_caps.data)
		devm_kfree(ethosn->dev, ethosn->fw_and_hw_caps.data);

	if (!IS_ERR_OR_NULL(ethosn->profiling.firmware_buffer))
		ethosn_dma_free(ethosn, ETHOSN_STREAM_WORKING_DATA,
				ethosn->profiling.firmware_buffer);

	if (!IS_ERR_OR_NULL(ethosn->profiling.firmware_buffer_pending))
		ethosn_dma_free(ethosn, ETHOSN_STREAM_WORKING_DATA,
				ethosn->profiling.firmware_buffer_pending);
}

bool ethosn_profiling_enabled(void)
{
	return profiling_enabled;
}

bool ethosn_mailbox_empty(struct ethosn_queue *queue)
{
	return (queue->read == queue->write);
}

int ethosn_clock_frequency(void)
{
	return clock_frequency;
}

/* uncrustify-off */
struct ethosn_device *ethosn_get_global_device_for_testing(void)
{
	return ethosn_global_device_for_testing;
}

/* Exported for use by ethosn-tests module */
EXPORT_SYMBOL(ethosn_get_global_device_for_testing);
/* uncrustify-on */
