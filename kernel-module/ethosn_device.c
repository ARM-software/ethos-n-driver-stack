/*
 *
 * (C) COPYRIGHT 2018-2022 Arm Limited.
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

#include "ethosn_backport.h"
#include "ethosn_firmware.h"
#include "ethosn_smc.h"

#include <linux/firmware.h>
#include <linux/iommu.h>
#include <linux/log2.h>
#include <linux/module.h>
#include <linux/of.h>
#include <linux/of_reserved_mem.h>
#include <linux/uaccess.h>
#include <linux/time.h>

/* Number of bits the MCU Vector Table address is shifted. */
#define SYSCTLR0_INITVTOR_SHIFT         7

/* Init vector table size */
#define ETHOSN_VTABLE_SIZE                 16

/* Firmware code size */
#define ETHOSN_CODE_SIZE                   0x40000

/* Timeout in us when resetting the Ethos-N */
#define ETHOSN_RESET_TIMEOUT_US         (10 * 1000 * 1000)
#define ETHOSN_RESET_WAIT_US            1

/* Regset32 entry */
#define REGSET32(r) { __stringify(r), \
		      TOP_REG(DL1_RP, DL1_ ## r) - TOP_REG(0, 0) }

static int severity = ETHOSN_LOG_INFO;
module_param(severity, int, 0660);

static int ethosn_queue_size = 65536;
module_param_named(queue_size, ethosn_queue_size, int, 0440);

static bool profiling_enabled;
module_param_named(profiling, profiling_enabled, bool, 0664);

/* Clock frequency expressed in MHz */
static int clock_frequency = 1000;
module_param_named(clock_frequency, clock_frequency, int, 0440);

static bool stashing_enabled = true;
module_param_named(stashing, stashing_enabled, bool, 0440);

/* Exposes global access to the most-recently created Ethos-N core for testing
 * purposes.
 */
static struct ethosn_core *ethosn_global_core_for_testing;

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

bool ethosn_smmu_available(struct device *dev)
{
	int len;
	bool has_smmu = false;
	bool is_parent = of_get_available_child_count(dev->of_node) > 0;
	struct device_node *node;

	/* iommus property is only available in the children
	 * nodes (i.e. ethosn-core)
	 */
	if (is_parent)
		node = of_get_next_available_child(dev->of_node, NULL);
	else
		node = dev->of_node;

	has_smmu = !IS_ERR_OR_NULL(of_find_property(node, "iommus", &len));

	if (is_parent)
		of_node_put(node);

	return has_smmu;
}

/* Exported for use by test module */
EXPORT_SYMBOL(ethosn_smmu_available);

/**
 * ethosn_mailbox_init() - Initialize the mailbox structure.
 * @core:	Pointer to Ethos-N core.
 *
 * Return: 0 on success, else error code.
 */
static int ethosn_mailbox_init(struct ethosn_core *core)
{
	struct ethosn_mailbox *mailbox = core->mailbox->cpu_addr;
	struct ethosn_queue *request = core->mailbox_request->cpu_addr;
	struct ethosn_queue *response = core->mailbox_response->cpu_addr;
	resource_size_t mailbox_addr;

	/* Clear memory */
	memset(mailbox, 0, core->mailbox->size);
	memset(request, 0, core->mailbox_request->size);
	memset(response, 0, core->mailbox_response->size);

	/* Setup queue sizes */
	request->capacity = core->mailbox_request->size -
			    sizeof(struct ethosn_queue);
	response->capacity = core->mailbox_response->size -
			     sizeof(struct ethosn_queue);

	/* Set severity, and make sure it's in the range [PANIC, VERBOSE]. */
	mailbox->severity = max(min(severity,
				    ETHOSN_LOG_VERBOSE), ETHOSN_LOG_PANIC);

	/* Set Ethos-N addresses from mailbox to queues */
	mailbox->request = to_ethosn_addr(core->mailbox_request->iova_addr,
					  &core->work_data_map);
	if (IS_ERR_VALUE((unsigned long)mailbox->request))
		return -EFAULT;

	mailbox->response = to_ethosn_addr(core->mailbox_response->iova_addr,
					   &core->work_data_map);
	if (IS_ERR_VALUE((unsigned long)mailbox->response))
		return -EFAULT;

	/* Store mailbox address in GP2 */
	mailbox_addr = to_ethosn_addr(core->mailbox->iova_addr,
				      &core->work_data_map);
	if (IS_ERR_VALUE((unsigned long)mailbox_addr))
		return -EFAULT;

	/* Sync memory to device */
	ethosn_dma_sync_for_device(core->allocator, core->mailbox);
	ethosn_dma_sync_for_device(core->allocator, core->mailbox_request);
	ethosn_dma_sync_for_device(core->allocator, core->mailbox_response);

	/* Store mailbox CU address in GP2 */
	ethosn_write_top_reg(core, DL1_RP, GP_MAILBOX, mailbox_addr);

	return 0;
}

/**
 * mailbox_alloc() - Allocate the mailbox.
 * @core:	Pointer to Ethos-N core.
 *
 * Return: 0 on success, else error code.
 */
static int mailbox_alloc(struct ethosn_core *core)
{
	struct ethosn_dma_allocator *allocator = core->allocator;
	int ret = -ENOMEM;

	core->mailbox =
		ethosn_dma_alloc_and_map(
			allocator,
			sizeof(struct ethosn_mailbox),
			ETHOSN_PROT_READ | ETHOSN_PROT_WRITE,
			ETHOSN_STREAM_WORKING_DATA,
			GFP_KERNEL,
			"mailbox-header");
	if (IS_ERR_OR_NULL(core->mailbox)) {
		dev_warn(core->dev,
			 "Failed to allocate memory for mailbox");
		goto err_exit;
	}

	core->mailbox_request =
		ethosn_dma_alloc_and_map(
			allocator,
			sizeof(struct ethosn_queue) +
			core->queue_size,
			ETHOSN_PROT_READ | ETHOSN_PROT_WRITE,
			ETHOSN_STREAM_WORKING_DATA,
			GFP_KERNEL,
			"mailbox-request");
	if (IS_ERR_OR_NULL(core->mailbox_request)) {
		dev_warn(core->dev,
			 "Failed to allocate memory for mailbox request queue");
		goto err_free_mailbox;
	}

	core->mailbox_response =
		ethosn_dma_alloc_and_map(allocator,
					 sizeof(struct ethosn_queue) +
					 core->queue_size,
					 ETHOSN_PROT_READ | ETHOSN_PROT_WRITE,
					 ETHOSN_STREAM_WORKING_DATA,
					 GFP_KERNEL,
					 "mailbox-response");
	if (IS_ERR_OR_NULL(core->mailbox_response)) {
		dev_warn(core->dev,
			 "Failed to allocate memory for mailbox response queue");
		goto err_free_mailbox_request;
	}

	core->mailbox_message =
		devm_kzalloc(core->parent->dev, core->queue_size,
			     GFP_KERNEL);
	if (!core->mailbox_message)
		goto err_free_mailbox_response;

	core->num_pongs_received = 0;

	return 0;

err_free_mailbox_response:
	ethosn_dma_unmap_and_free(allocator, core->mailbox_response,
				  ETHOSN_STREAM_WORKING_DATA);
err_free_mailbox_request:
	ethosn_dma_unmap_and_free(allocator, core->mailbox_request,
				  ETHOSN_STREAM_WORKING_DATA);
err_free_mailbox:
	ethosn_dma_unmap_and_free(allocator, core->mailbox,
				  ETHOSN_STREAM_WORKING_DATA);
err_exit:

	return ret;
}

/**
 * ethosn_mailbox_free() - Free the mailbox.
 * @core:	Pointer to Ethos-N core.
 */
static void ethosn_mailbox_free(struct ethosn_core *core)
{
	ethosn_dma_unmap_and_free(core->allocator, core->mailbox,
				  ETHOSN_STREAM_WORKING_DATA);
	core->mailbox = NULL;

	ethosn_dma_unmap_and_free(core->allocator, core->mailbox_request,
				  ETHOSN_STREAM_WORKING_DATA);
	core->mailbox_request = NULL;

	ethosn_dma_unmap_and_free(core->allocator, core->mailbox_response,
				  ETHOSN_STREAM_WORKING_DATA);
	core->mailbox_response = NULL;

	if (core->mailbox_message) {
		devm_kfree(core->parent->dev, core->mailbox_message);
		core->mailbox_message = NULL;
	}
}

void ethosn_write_top_reg(struct ethosn_core *core,
			  const u32 page,
			  const u32 offset,
			  const u32 value)
{
	iowrite32(value, ethosn_top_reg_addr(core->top_regs, page, offset));
}

/* Exported for use by test module * */
EXPORT_SYMBOL(ethosn_write_top_reg);

u32 ethosn_read_top_reg(struct ethosn_core *core,
			const u32 page,
			const u32 offset)
{
	return ioread32(ethosn_top_reg_addr(core->top_regs, page, offset));
}

/* Exported for use by test module */
EXPORT_SYMBOL(ethosn_read_top_reg);

static int ethosn_task_stack_init(struct ethosn_core *core)
{
	u32 stack_addr = to_ethosn_addr(core->firmware_stack_task->iova_addr,
					&core->work_data_map);

	if (IS_ERR_VALUE((unsigned long)stack_addr))
		return -EFAULT;

	stack_addr += core->firmware_stack_task->size;

	ethosn_write_top_reg(core, DL1_RP, GP_TASK_STACK, stack_addr);

	return 0;
}

/**
 * ethosn_boot_firmware() - Boot firmware.
 * @core:	Pointer to Ethos-N core.
 *
 * Return: 0 on success, else error code.
 */
static int ethosn_boot_firmware(struct ethosn_core *core)
{
	struct dl1_sysctlr0_r sysctlr0 = { .word = 0 };
	struct dl1_sysctlr1_r sysctlr1 = { .word = 0 };
	uint32_t *vtable = core->firmware_vtable->cpu_addr;

	memset(vtable, 0, core->firmware_vtable->size);

	/* Set vtable stack pointer */
	vtable[0] = to_ethosn_addr(core->firmware_stack_main->iova_addr,
				   &core->work_data_map);
	if (vtable[0] >= (uint32_t)-MAX_ERRNO)
		return (int)vtable[0];

	vtable[0] += core->firmware_stack_main->size;

	/* Set vtable reset program counter */
	vtable[1] = to_ethosn_addr(core->firmware->iova_addr,
				   &core->firmware_map) + 1;
	if (vtable[1] >= (uint32_t)-MAX_ERRNO)
		return (int)vtable[1];

	ethosn_dma_sync_for_device(core->allocator, core->firmware_vtable);

	/* Enable events */
	sysctlr1.bits.mcu_setevnt = 1;
	sysctlr1.bits.mcu_gpevnt = 1;
	ethosn_write_top_reg(core, DL1_RP, DL1_SYSCTLR1, sysctlr1.word);

	/* Set firmware init address and release CPU wait */
	sysctlr0.bits.cpuwait = 0;
	sysctlr0.bits.initvtor =
		to_ethosn_addr(core->firmware_vtable->iova_addr,
			       &core->firmware_map) >>
		SYSCTLR0_INITVTOR_SHIFT;
	ethosn_write_top_reg(core, DL1_RP, DL1_SYSCTLR0, sysctlr0.word);

	return 0;
}

void ethosn_notify_firmware(struct ethosn_core *core)
{
	struct dl1_setirq_int_r irq = {
		.bits          = {
			.event = 1,
		}
	};

	ethosn_write_top_reg(core, DL1_RP, DL1_SETIRQ_INT,
			     irq.word);
}

static int ethosn_hard_reset(struct ethosn_core *core)
{
#ifdef ETHOSN_NS
	struct dl1_sysctlr0_r sysctlr0 = { .word = 0 };
	unsigned int timeout;

	dev_info(core->dev, "Hard reset the hardware.\n");

	/* Initiate hard reset */
	sysctlr0.bits.hard_rstreq = 1;
	ethosn_write_top_reg(core, DL1_RP, DL1_SYSCTLR0, sysctlr0.word);

	/* Wait for hard reset to complete */
	for (timeout = 0; timeout < ETHOSN_RESET_TIMEOUT_US;
	     timeout += ETHOSN_RESET_WAIT_US) {
		sysctlr0.word =
			ethosn_read_top_reg(core, DL1_RP, DL1_SYSCTLR0);

		if (sysctlr0.bits.hard_rstreq == 0)
			break;

		udelay(ETHOSN_RESET_WAIT_US);
	}

	if (timeout >= ETHOSN_RESET_TIMEOUT_US) {
		dev_err(core->dev, "Failed to hard reset the hardware.\n");

		return -EFAULT;
	}

	return 0;

#else

	/*
	 * Access to DL1 registers is blocked in secure mode so reset is done
	 * with a SMC call. The call will block until the reset is done or
	 * timeout.
	 */
	return ethosn_smc_core_reset(core->dev, core->phys_addr, 1);
#endif
}

static int ethosn_soft_reset(struct ethosn_core *core)
{
#ifdef ETHOSN_NS
	struct dl1_sysctlr0_r sysctlr0 = { .word = 0 };
	unsigned int timeout;

	dev_info(core->dev, "Soft reset the hardware.\n");

	/* Soft reset, block new AXI requests */
	sysctlr0.bits.soft_rstreq = 3;
	ethosn_write_top_reg(core, DL1_RP, DL1_SYSCTLR0, sysctlr0.word);

	/* Wait for reset to complete */
	for (timeout = 0; timeout < ETHOSN_RESET_TIMEOUT_US;
	     timeout += ETHOSN_RESET_WAIT_US) {
		sysctlr0.word =
			ethosn_read_top_reg(core, DL1_RP, DL1_SYSCTLR0);

		if (sysctlr0.bits.soft_rstreq == 0)
			break;

		udelay(ETHOSN_RESET_WAIT_US);
	}

	if (timeout >= ETHOSN_RESET_TIMEOUT_US) {
		dev_warn(core->dev,
			 "Failed to soft reset the hardware. sysctlr0=0x%08x\n",
			 sysctlr0.word);

		return -ETIME;
	}

#else

	/*
	 * Access to DL1 registers is blocked in secure mode so reset is done
	 * with a SMC call. The call will block until the reset is done or
	 * timeout.
	 */
	if (ethosn_smc_core_reset(core->dev, core->phys_addr, 0))
		return -ETIME;

#endif

	return 0;
}

int ethosn_reset(struct ethosn_core *core)
{
	int ret = -EINVAL;

	ret = ethosn_soft_reset(core);
	if (ret)
		ret = ethosn_hard_reset(core);

	return ret;
}

void ethosn_set_power_ctrl(struct ethosn_core *core,
			   bool clk_on)
{
	struct dl1_pwrctlr_r pwrctlr = { .word = 0 };

	pwrctlr.bits.active = clk_on;
	ethosn_write_top_reg(core, DL1_RP, DL1_PWRCTLR, pwrctlr.word);
}

/**
 * ethosn_set_mmu_stream_id() - Configure the mmu stream id0.
 * @core:	Pointer to Ethos-N core
 *
 * Return: Negative error code on error, zero otherwise
 */
int ethosn_set_mmu_stream_id(struct ethosn_core *core)
{
	struct iommu_fwspec *fwspec = dev_iommu_fwspec_get(core->dev);
	int ret = -EINVAL;
	unsigned int stream_id;
	static const int mmusid_0 = DL1_STREAM0_MMUSID;

	/*
	 * Currently, it is permitted to define only one stream id in the dts
	 * file. There is no advantage of defining multiple stream ids when
	 * the device uses all the streams at almost all the times.
	 */
	if (fwspec->num_ids > 1) {
		dev_err(core->dev,
			"Support for multiple streams for a single device is not allowed\n");

		return ret;
	}

	stream_id = fwspec->ids[0];

	/*
	 * The value of stream id fetched from the dts is used to program the
	 * STREAM0_MMUSID register. The other stream id registers are programmed
	 * based on this value in the firmware.
	 */
	ethosn_write_top_reg(core, DL1_RP, mmusid_0, stream_id);
	ethosn_write_top_reg(core, DL1_RP, GP_MMUSID0, stream_id);

	return 0;
}

/**
 * ethosn_set_addr_ext() - Configure address extension for stream
 * @core:       Pointer to Ethos-N core
 * @stream: stream to configure (must be 0-2)
 * @offset: Offset to apply. Lower 29 bits will be ignored
 * @addr_map: Address map to be used later to convert to Ethos-N addresses
 *
 * Return: Negative error code on error, zero otherwise
 */
int ethosn_set_addr_ext(struct ethosn_core *core,
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
		dev_err(core->dev,
			"Illegal stream %u for address extension.\n",
			stream);

		return -EFAULT;
	}

	ext.bits.addrextend = offset >> REGION_SHIFT;

	ethosn_write_top_reg(core, DL1_RP, stream_to_page[stream],
			     ext.word);

	if (addr_map) {
		addr_map->region = stream_to_offset[stream];
		addr_map->extension = offset & ~ETHOSN_REGION_MASK;
	}

	return 0;
}

static int get_gp_offset(struct ethosn_core *core,
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
		dev_err(core->dev,
			"Illegal index %u of general purpose register.\n",
			index);

		return -EFAULT;
	}

	return index_to_offset[index];
}

void ethosn_dump_gps(struct ethosn_core *core)
{
	int offset;
	unsigned int i;

	for (i = 0; i < 8; i++) {
		offset = get_gp_offset(core, i);
		if (offset < 0)
			break;

		dev_info(core->dev,
			 "GP%u=0x%08x\n",
			 i, ethosn_read_top_reg(core, DL1_RP, offset));
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
int ethosn_read_message(struct ethosn_core *core,
			struct ethosn_message_header *header,
			void *data,
			size_t length)
{
	struct ethosn_queue *queue = core->mailbox_response->cpu_addr;
	bool ret;
	uint32_t read_pending;

	if (core->mailbox_response->size <
	    (sizeof(*queue) + queue->capacity) ||
	    !is_power_of_2(queue->capacity)) {
		dev_err(core->dev,
			"Illegal mailbox queue capacity. alloc_size=%zu, queue capacity=%u\n",
			core->mailbox_request->size, queue->capacity);

		return -EFAULT;
	}

	ethosn_dma_sync_for_cpu(core->allocator, core->mailbox_response);

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

	dev_dbg(core->dev,
		"Received message. type=%u, length=%u, read=%u, write=%u.\n",
		header->type, header->length, queue->read,
		queue->write);

	if (length < header->length) {
		dev_warn(core->dev,
			 "Message too large to read. header.length=%u, length=%zu.\n",
			 header->length, length);

		ethosn_queue_skip(queue, header->length);

		return -ENOMEM;
	}

	ret = ethosn_queue_read(queue, data, header->length, &read_pending);
	if (!ret) {
		dev_err(core->dev,
			"Failed to read message payload. size=%u, queue capacity=%u\n",
			header->length, queue->capacity);

		return -EFAULT;
	}

	queue->read = read_pending;

	ethosn_dma_sync_for_device(core->allocator, core->mailbox_response);

	if (core->profiling.config.enable_profiling)
		++core->profiling.mailbox_messages_received;

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
int ethosn_write_message(struct ethosn_core *core,
			 enum ethosn_message_type type,
			 void *data,
			 size_t length)
{
	struct ethosn_queue *queue = core->mailbox_request->cpu_addr;
	struct ethosn_message_header header = {
		.type   = type,
		.length = length
	};
	bool ret;
	uint32_t write_pending;

	if (core->mailbox_response->size <
	    (sizeof(*queue) + queue->capacity) ||
	    !is_power_of_2(queue->capacity)) {
		dev_err(core->dev,
			"Illegal mailbox queue capacity. alloc_size=%zu, queue capacity=%u\n",
			core->mailbox_request->size, queue->capacity);

		return -EFAULT;
	}

	ethosn_dma_sync_for_cpu(core->allocator, core->mailbox_request);

	dev_dbg(core->dev,
		"Write message. type=%u, length=%zu, read=%u, write=%u.\n",
		type, length, queue->read, queue->write);

	write_pending = queue->write;

	ret =
		ethosn_queue_write(queue, (uint8_t *)&header,
				   sizeof(struct ethosn_message_header),
				   &write_pending);
	if (!ret)
		return ret;

	ret = ethosn_queue_write(queue, data, length, &write_pending);
	if (!ret)
		return ret;

	/*
	 * Sync the payload before committing the updated write pointer so that
	 * the "reading" side (e.g. CU firmware) can't read invalid data.
	 */
	ethosn_dma_sync_for_device(core->allocator, core->mailbox_request);

	/*
	 * Update the write pointer after all the data has been written.
	 */
	queue->write = write_pending;

	/* Sync the write pointer */
	ethosn_dma_sync_for_device(core->allocator, core->mailbox_request);
	ethosn_notify_firmware(core);

	if (core->profiling.config.enable_profiling)
		++core->profiling.mailbox_messages_sent;

	return 0;
}

/* Exported for use by test module * */
EXPORT_SYMBOL(ethosn_write_message);

int ethosn_send_fw_hw_capabilities_request(struct ethosn_core *core)
{
	/* If it's a firmware reboot (i.e. capabilities have been
	 * already received once) don't request caps again
	 */
	if (core->fw_and_hw_caps.size > 0U)
		return 0;

	dev_dbg(core->dev, "-> FW & HW Capabilities\n");

	return ethosn_write_message(core, ETHOSN_MESSAGE_FW_HW_CAPS_REQUEST,
				    NULL, 0);
}

/*
 * Note we do not use the profiling config in ethosn->profiling, because if we
 * are in the process of updating that, it may not yet have been committed.
 * Instead we take the arguments explicitly.
 */
static int ethosn_send_configure_profiling(struct ethosn_core *core,
					   bool enable,
					   uint32_t num_hw_counters,
					   enum
					   ethosn_profiling_hw_counter_types *
					   hw_counters,
					   struct ethosn_dma_info *buffer)
{
	struct ethosn_firmware_profiling_configuration fw_new_config;
	int i;

	if (num_hw_counters > ETHOSN_PROFILING_MAX_HW_COUNTERS) {
		dev_err(core->dev,
			"Invalid number of hardware profiling counters\n");

		return -EINVAL;
	}

	fw_new_config.enable_profiling = enable;

	if (!IS_ERR_OR_NULL(buffer)) {
		fw_new_config.buffer_size = buffer->size;
		fw_new_config.buffer_address =
			to_ethosn_addr(buffer->iova_addr,
				       &core->work_data_map);
		if (IS_ERR_VALUE((unsigned long)fw_new_config.buffer_address)) {
			dev_err(core->dev,
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

	dev_dbg(core->dev,
		"-> ETHOSN_MESSAGE_CONFIGURE_PROFILING, enable_profiling=%d, buffer_address=0x%08llx, buffer_size=%d\n",
		fw_new_config.enable_profiling, fw_new_config.buffer_address,
		fw_new_config.buffer_size);

	return ethosn_write_message(core, ETHOSN_MESSAGE_CONFIGURE_PROFILING,
				    &fw_new_config, sizeof(fw_new_config));
}

int ethosn_configure_firmware_profiling(struct ethosn_core *core,
					struct ethosn_profiling_config *
					new_config)
{
	int ret = -ENOMEM;

	/* If we are already waiting for the firmware to acknowledge use of a
	 * new buffer then we cannot allocate another.
	 * We must wait for it to acknowledge first.
	 */
	if (core->profiling.is_waiting_for_firmware_ack) {
		dev_err(core->dev,
			"Already waiting for firmware to acknowledge new profiling config.\n");

		ret = -EINVAL;
		goto ret;
	}

	/* Allocate new profiling buffer.
	 * Note we do not overwrite the existing buffer yet, as the firmware may
	 * still be using it
	 */
	if (new_config->enable_profiling &&
	    new_config->firmware_buffer_size > 0) {
		struct ethosn_profiling_buffer *buffer;

		core->profiling.firmware_buffer_pending =
			ethosn_dma_alloc_and_map(
				core->allocator,
				new_config->firmware_buffer_size,
				ETHOSN_PROT_READ | ETHOSN_PROT_WRITE,
				ETHOSN_STREAM_WORKING_DATA,
				GFP_KERNEL,
				"profiling-firmware-buffer");
		if (IS_ERR(core->profiling.firmware_buffer_pending)) {
			dev_err(core->dev,
				"Error allocating firmware profiling buffer.\n");

			goto ret;
		}

		/* Initialize the firmware_write_index. */
		buffer =
			(struct ethosn_profiling_buffer *)
			core->profiling.firmware_buffer_pending->cpu_addr;
		buffer->firmware_write_index = 0;
		ethosn_dma_sync_for_device(
			core->allocator,
			core->profiling.firmware_buffer_pending);
	} else {
		core->profiling.firmware_buffer_pending = NULL;
	}

	core->profiling.is_waiting_for_firmware_ack = true;

	ret = ethosn_send_configure_profiling(
		core, new_config->enable_profiling,
		new_config->num_hw_counters,
		new_config->hw_counters,
		core->profiling.firmware_buffer_pending);
	if (ret != 0) {
		dev_err(core->dev, "ethosn_send_configure_profiling failed.\n");

		goto free_buf;
	}

	return 0;

free_buf:
	ethosn_dma_unmap_and_free(
		core->allocator,
		core->profiling.firmware_buffer_pending,
		ETHOSN_STREAM_WORKING_DATA);
	core->profiling.firmware_buffer_pending = NULL;
ret:

	return ret;
}

int ethosn_configure_firmware_profiling_ack(struct ethosn_core *core)
{
	if (!core->profiling.is_waiting_for_firmware_ack) {
		dev_err(core->dev,
			"Unexpected configure profiling ack from firmware.\n");

		return -EINVAL;
	}

	/* We can now free the old buffer (if any), as we know the firmware is
	 * no longer writing to it
	 */
	ethosn_dma_unmap_and_free(core->allocator,
				  core->profiling.firmware_buffer,
				  ETHOSN_STREAM_WORKING_DATA);

	/* What used to be the pending buffer is now the proper one. */
	core->profiling.firmware_buffer =
		core->profiling.firmware_buffer_pending;
	core->profiling.firmware_buffer_pending = NULL;
	core->profiling.is_waiting_for_firmware_ack = false;

	return 0;
}

int ethosn_send_time_sync(struct ethosn_core *core)
{
	struct ethosn_message_time_sync_request request;

	dev_dbg(core->dev, "-> Time Sync\n");

	request.timestamp = ktime_get_real_ns();

	return ethosn_write_message(core, ETHOSN_MESSAGE_TIME_SYNC, &request,
				    sizeof(request));
}

int ethosn_send_ping(struct ethosn_core *core)
{
	dev_dbg(core->dev, "-> Ping\n");

	return ethosn_write_message(core, ETHOSN_MESSAGE_PING, NULL, 0);
}

int ethosn_send_inference(struct ethosn_core *core,
			  dma_addr_t buffer_array,
			  uint64_t user_arg)
{
	struct ethosn_message_inference_request request;

	request.buffer_array = to_ethosn_addr(buffer_array, &core->dma_map);
	request.user_argument = user_arg;

	dev_dbg(core->dev,
		"-> Inference. buffer_array=0x%08llx, user_args=0x%llx\n",
		request.buffer_array, request.user_argument);

	return ethosn_write_message(core, ETHOSN_MESSAGE_INFERENCE_REQUEST,
				    &request,
				    sizeof(request));
}

/**
 * ethosn_send_region_request() - Send memory region request to device.
 * @core:	Pointer to core device.
 * @region_id:	Memory region identifier.
 *
 * Return: 0 on success, else error code.
 */
static int ethosn_send_region_request(struct ethosn_core *core,
				      enum ethosn_region_id region_id)
{
	struct ethosn_message_region_request request = { 0 };

	switch (region_id) {
	case ETHOSN_REGION_FIRMWARE:
		request.addr = to_ethosn_addr(
			ethosn_dma_get_addr_base(core->allocator,
						 ETHOSN_STREAM_FIRMWARE),
			&core->firmware_map);

		request.size = ethosn_dma_get_addr_size(core->allocator,
							ETHOSN_STREAM_FIRMWARE);
		break;
	case ETHOSN_REGION_WORKING_DATA_MAIN:
		request.addr = to_ethosn_addr(
			ethosn_dma_get_addr_base(core->allocator,
						 ETHOSN_STREAM_WORKING_DATA),
			&core->work_data_map);

		request.size = ethosn_dma_get_addr_size(
			core->allocator,
			ETHOSN_STREAM_WORKING_DATA);
		break;
	case ETHOSN_REGION_WORKING_DATA_TASK:
		request.addr = to_ethosn_addr(
			core->firmware_stack_task->iova_addr,
			&core->work_data_map);

		request.size = core->firmware_stack_task->size;
		break;
	case ETHOSN_REGION_COMMAND_STREAM:
		request.addr = to_ethosn_addr(
			ethosn_dma_get_addr_base(core->allocator,
						 ETHOSN_STREAM_COMMAND_STREAM),
			&core->dma_map);

		request.size = ethosn_dma_get_addr_size(
			core->allocator,
			ETHOSN_STREAM_COMMAND_STREAM);
		break;
	default:
		dev_err(core->dev, "Unknown memory region ID: %u", region_id);

		return -EFAULT;
	}

	if (request.size == 0)
		return -EFAULT;

	request.id = region_id;

	dev_dbg(core->dev, "-> Region=%u, addr=0x%x, size=0x%x\n",
		request.id, request.addr, request.size);

	return ethosn_write_message(core, ETHOSN_MESSAGE_REGION_REQUEST,
				    &request,
				    sizeof(request));
}

/**
 * ethosn_send_mpu_enable_request() - Send Mpu enable request to device.
 * @core:		Pointer to core device.
 *
 * Return: 0 on success, else error code.
 */
static int ethosn_send_mpu_enable_request(struct ethosn_core *core)
{
	dev_dbg(core->dev,
		"-> Mpu enable.");

	return ethosn_write_message(core, ETHOSN_MESSAGE_MPU_ENABLE_REQUEST,
				    NULL, 0);
}

int ethosn_send_stash_request(struct ethosn_core *core)
{
	if (!ethosn_stashing_enabled())
		return 0;

	if (ethosn_smmu_available(core->dev)) {
		dev_dbg(core->dev, "-> SMMU Available");

		return ethosn_write_message(core, ETHOSN_MESSAGE_STASH_REQUEST,
					    NULL, 0);
	} else {
		dev_dbg(core->dev, "-> SMMU Not Available");

		return 0;
	}
}

/****************************************************************************
 * Firmware
 ****************************************************************************/

/* Big FW binary structure */
struct ethosn_big_fw {
	uint32_t fw_ver_major;
	uint32_t fw_ver_minor;
	uint32_t fw_ver_patch;
	uint32_t fw_cnt;
	struct ethosn_big_fw_desc {
		uint32_t arch_min;
		uint32_t arch_max;
		uint32_t offset;
		uint32_t size;
	} desc[];
} __packed;

static struct ethosn_big_fw_desc *find_big_fw_desc(struct ethosn_core *core,
						   struct ethosn_big_fw *big_fw)
{
	struct dl1_npu_id_r npu_id;
	int i = big_fw->fw_cnt;
	uint32_t arch;

	npu_id.word = ethosn_read_top_reg(core, DL1_RP, DL1_NPU_ID);
	arch = npu_id.bits.arch_major << 24 |
	       npu_id.bits.arch_minor << 16 |
	       npu_id.bits.arch_rev;

	dev_dbg(core->dev,
		"NPU reported version %u.%u.%u. FWs in BIG FW: %u. FW version in BIG FW: %u.%u.%u\n",
		npu_id.bits.arch_major,
		npu_id.bits.arch_minor,
		npu_id.bits.arch_rev,
		big_fw->fw_cnt,
		big_fw->fw_ver_major,
		big_fw->fw_ver_minor,
		big_fw->fw_ver_patch);

	while (i--) {
		if (big_fw->desc[i].arch_min <= arch &&
		    arch <= big_fw->desc[i].arch_max)
			return &big_fw->desc[i];

		dev_dbg(core->dev, "Skip FW min=0x%08x, max=0x%08x\n",
			big_fw->desc[i].arch_min,
			big_fw->desc[i].arch_max);
	}

	dev_err(core->dev, "Cannot find compatible FW in BIG FW.\n");

	return ERR_PTR(-EINVAL);
}

static int verify_firmware(struct ethosn_core *core,
			   struct ethosn_big_fw *big_fw)
{
	if (big_fw->fw_ver_major != ETHOSN_FIRMWARE_VERSION_MAJOR) {
		dev_err(core->dev,
			"Wrong firmware version. Version %u.x.x is required.\n",
			ETHOSN_FIRMWARE_VERSION_MAJOR);

		return -EINVAL;
	}

	return 0;
}

/**
 * firmware_load - Load firmware binary with given name.
 * @core:		Pointer to Ethos-N core.
 * @firmware_name:	Name of firmware binary.
 *
 * Return: 0 on success, else error code.
 */
static int firmware_load(struct ethosn_core *core,
			 const char *firmware_name)
{
	const struct firmware *fw;
	struct ethosn_big_fw *big_fw;
	struct ethosn_big_fw_desc *big_fw_desc;
	size_t size;
	int ret = -ENOMEM;

	/* Request firmware binary */
	ret = request_firmware(&fw, firmware_name, core->parent->dev);
	if (ret)
		return ret;

	big_fw = (struct ethosn_big_fw *)fw->data;

	/* Find a FW binary for this NPU */
	big_fw_desc = find_big_fw_desc(core, big_fw);
	if (IS_ERR(big_fw_desc))
		return -EINVAL;

	/* Check FW binary version compatibility */
	ret = verify_firmware(core, big_fw);
	if (ret)
		return ret;

	dev_dbg(core->dev,
		"Found FW. arch_min=0x%08x, arch_max=0x%08x, offset=0x%08x, size=0x%08x",
		big_fw_desc->arch_min,
		big_fw_desc->arch_max,
		big_fw_desc->offset,
		big_fw_desc->size);
	/* Make sure code size is at least 256 KB */
	size = max_t(size_t, ETHOSN_CODE_SIZE, big_fw_desc->size);

	/* Allocate memory for firmware code */
	if (!core->firmware)
		core->firmware =
			ethosn_dma_alloc_and_map(core->allocator, size,
						 ETHOSN_PROT_READ |
						 ETHOSN_PROT_WRITE,
						 ETHOSN_STREAM_FIRMWARE,
						 GFP_KERNEL,
						 "firmware-code");

	if (IS_ERR_OR_NULL(core->firmware)) {
		ret = -ENOMEM;
		goto release_fw;
	}

	memcpy(core->firmware->cpu_addr, fw->data + big_fw_desc->offset,
	       big_fw_desc->size);
	ethosn_dma_sync_for_device(core->allocator, core->firmware);

	/* Allocate task stack */
	if (!core->firmware_stack_task)
		core->firmware_stack_task =
			ethosn_dma_alloc_and_map(core->allocator,
						 ETHOSN_STACK_SIZE,
						 ETHOSN_PROT_READ |
						 ETHOSN_PROT_WRITE,
						 ETHOSN_STREAM_WORKING_DATA,
						 GFP_KERNEL,
						 "firmware-stack-task");

	if (IS_ERR_OR_NULL(core->firmware_stack_task)) {
		ret = -ENOMEM;
		goto free_firmware;
	}

	/* Allocate main stack */
	if (!core->firmware_stack_main)
		core->firmware_stack_main =
			ethosn_dma_alloc_and_map(core->allocator,
						 ETHOSN_STACK_SIZE,
						 ETHOSN_PROT_READ |
						 ETHOSN_PROT_WRITE,
						 ETHOSN_STREAM_WORKING_DATA,
						 GFP_KERNEL,
						 "firmware-stack-main");

	if (IS_ERR_OR_NULL(core->firmware_stack_main)) {
		ret = -ENOMEM;
		goto free_stack_task;
	}

	/* Allocate vtable */
	if (!core->firmware_vtable)
		core->firmware_vtable =
			ethosn_dma_alloc_and_map(core->allocator,
						 ETHOSN_VTABLE_SIZE *
						 sizeof(uint32_t),
						 ETHOSN_PROT_READ |
						 ETHOSN_PROT_WRITE,
						 ETHOSN_STREAM_FIRMWARE,
						 GFP_KERNEL,
						 "firmware-vtable");

	if (IS_ERR_OR_NULL(core->firmware_vtable)) {
		ret = -ENOMEM;
		goto free_stack_main;
	}

	release_firmware(fw);

	return 0;

free_stack_main:
	ethosn_dma_unmap_and_free(core->allocator, core->firmware_stack_main,
				  ETHOSN_STREAM_WORKING_DATA);
free_stack_task:
	ethosn_dma_unmap_and_free(core->allocator, core->firmware_stack_task,
				  ETHOSN_STREAM_WORKING_DATA);
free_firmware:
	ethosn_dma_unmap_and_free(core->allocator, core->firmware,
				  ETHOSN_STREAM_FIRMWARE);
release_fw:
	release_firmware(fw);

	return ret;
}

/**
 * firmware_init - Allocate and initialize firmware.
 * @core:		Pointer to Ethos-N core.
 *
 * Try to load firmware binaries in given order.
 *
 * Return: 0 on success, else error code.
 */
static int firmware_init(struct ethosn_core *core)
{
	static const char *const firmware_names[] = {
		"ethosn.bin"
	};
	int i;
	int ret;

	for (i = 0; i < ARRAY_SIZE(firmware_names); ++i) {
		ret = firmware_load(core, firmware_names[i]);
		if (!ret)
			break;
	}

	if (ret) {
		dev_err(core->dev, "No firmware found.\n");

		return ret;
	}

	return 0;
}

/**
 * ethosn_regions_init() - Initialize the memory regions.
 * @core:	Pointer to Ethos-N core.
 *
 * Return: 0 on success, else error code.
 */
static int ethosn_regions_init(struct ethosn_core *core)
{
	int ret;

	ret = ethosn_send_region_request(core, ETHOSN_REGION_FIRMWARE);
	if (ret)
		return ret;

	ret = ethosn_send_region_request(core, ETHOSN_REGION_WORKING_DATA_MAIN);
	if (ret)
		return ret;

	ret = ethosn_send_region_request(core, ETHOSN_REGION_WORKING_DATA_TASK);
	if (ret)
		return ret;

	ret = ethosn_send_region_request(core, ETHOSN_REGION_COMMAND_STREAM);
	if (ret)
		return ret;

	ret = ethosn_send_mpu_enable_request(core);
	if (ret)
		return ret;

	return 0;
}

int ethosn_reset_and_start_ethosn(struct ethosn_core *core)
{
	int timeout;
	int ret;

	dev_info(core->dev, "Reset core device\n");

	/* Firmware is not running */
	core->firmware_running = false;

	/* Clear any outstanding configuration */
	if (core->profiling.is_waiting_for_firmware_ack) {
		ret = ethosn_configure_firmware_profiling_ack(core);
		if (ret)
			return ret;
	}

	/* Load the firmware */
	ret = firmware_init(core);
	if (ret)
		return ret;

	/* Reset the Ethos-N core */
	ret = ethosn_reset(core);
	if (ret)
		return ret;

	/* Set MMU Stream id0 if iommu is present */
	if (ethosn_smmu_available(core->dev)) {
		ret = ethosn_set_mmu_stream_id(core);
		if (ret)
			return ret;
	}

	/* Configure address extension for stream 0, 1 and 2 */
	ret = ethosn_set_addr_ext(
		core, ETHOSN_STREAM_FIRMWARE,
		ethosn_dma_get_addr_base(core->allocator,
					 ETHOSN_STREAM_FIRMWARE),
		&core->firmware_map);
	if (ret)
		return ret;

	ret = ethosn_set_addr_ext(
		core, ETHOSN_STREAM_WORKING_DATA,
		ethosn_dma_get_addr_base(core->allocator,
					 ETHOSN_STREAM_WORKING_DATA),
		&core->work_data_map);
	if (ret)
		return ret;

	ret = ethosn_set_addr_ext(
		core, ETHOSN_STREAM_COMMAND_STREAM,
		ethosn_dma_get_addr_base(core->allocator,
					 ETHOSN_STREAM_COMMAND_STREAM),
		&core->dma_map);
	if (ret)
		return ret;

	if (core->force_firmware_level_interrupts)
		ethosn_write_top_reg(core, DL1_RP, GP_IRQ, 1);

	/* Initialize the mailbox */
	ret = ethosn_mailbox_init(core);
	if (ret)
		return ret;

	/* Initialize the firmware task stack */
	ret = ethosn_task_stack_init(core);
	if (ret)
		return ret;

	/* Boot the firmware */
	ret = ethosn_boot_firmware(core);
	if (ret)
		return ret;

	dev_info(core->dev, "Waiting for core device\n");

	/* Wait for firmware to set GP_MAILBOX to 0 which indicates that it has
	 * booted
	 */
	for (timeout = 0; timeout < ETHOSN_RESET_TIMEOUT_US;
	     timeout += ETHOSN_RESET_WAIT_US) {
		if (ethosn_read_top_reg(core, DL1_RP, GP_MAILBOX) == 0)
			break;

		udelay(ETHOSN_RESET_WAIT_US);
	}

	if (timeout >= ETHOSN_RESET_TIMEOUT_US) {
		dev_err(core->dev, "Timeout while waiting for core device\n");

		return -ETIME;
	}

	/* Firmware is now up and running */
	core->firmware_running = true;

	/* Init memory regions */
	ret = ethosn_regions_init(core);
	if (ret != 0)
		return ret;

	/* Ping firmware */
	ret = ethosn_send_ping(core);
	if (ret != 0)
		return ret;

	/* Enable stashing */
	ret = ethosn_send_stash_request(core);
	if (ret != 0)
		return ret;

	/* Send FW and HW capabilities request */
	ret = ethosn_send_fw_hw_capabilities_request(core);
	if (ret != 0)
		return ret;

	/* Set FW's profiling state. This is also set whenever profiling is
	 * enabled/disabled, but we need to do it on each reboot in case
	 * the firmware crashes, so that its profiling state is restored.
	 */
	ret = ethosn_configure_firmware_profiling(core,
						  &core->profiling.config);
	if (ret != 0)
		return ret;

	return 0;
}

/**
 * ethosn_firmware_deinit - Free firmware resources.
 * @core:		Pointer to Ethos-N core.
 */
static void ethosn_firmware_deinit(struct ethosn_core *core)
{
	ethosn_dma_unmap_and_free(core->allocator, core->firmware,
				  ETHOSN_STREAM_FIRMWARE);
	core->firmware = NULL;

	ethosn_dma_unmap_and_free(core->allocator, core->firmware_stack_main,
				  ETHOSN_STREAM_WORKING_DATA);
	core->firmware_stack_main = NULL;

	ethosn_dma_unmap_and_free(core->allocator, core->firmware_stack_task,
				  ETHOSN_STREAM_WORKING_DATA);
	core->firmware_stack_task = NULL;

	ethosn_dma_unmap_and_free(core->allocator, core->firmware_vtable,
				  ETHOSN_STREAM_FIRMWARE);
	core->firmware_vtable = NULL;
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
	struct ethosn_core *core = file->f_inode->i_private;
	char buf[200];
	size_t n = 0;
	int ret;

	ret = mutex_lock_interruptible(&core->mutex);
	if (ret)
		return ret;

	if (core->mailbox_request) {
		struct ethosn_queue *queue = core->mailbox_request->cpu_addr;

		ethosn_dma_sync_for_cpu(core->allocator, core->mailbox_request);

		n += scnprintf(&buf[n], sizeof(buf) - n,
			       "Request queue : %llx\n",
			       core->mailbox_request->iova_addr);
		n += scnprintf(&buf[n], sizeof(buf) - n, "    capacity  : %u\n",
			       queue->capacity);
		n += scnprintf(&buf[n], sizeof(buf) - n, "    read      : %u\n",
			       queue->read);
		n += scnprintf(&buf[n], sizeof(buf) - n, "    write     : %u\n",
			       queue->write);
	}

	if (core->mailbox_response) {
		struct ethosn_queue *queue = core->mailbox_response->cpu_addr;

		ethosn_dma_sync_for_cpu(core->allocator,
					core->mailbox_response);

		n += scnprintf(&buf[n], sizeof(buf) - n,
			       "Response queue: %llx\n",
			       core->mailbox_response->iova_addr);
		n += scnprintf(&buf[n], sizeof(buf) - n, "    capacity  : %u\n",
			       queue->capacity);
		n += scnprintf(&buf[n], sizeof(buf) - n, "    read      : %u\n",
			       queue->read);
		n += scnprintf(&buf[n], sizeof(buf) - n, "    write     : %u\n",
			       queue->write);
	}

	if (core->mailbox) {
		struct ethosn_mailbox *mailbox = core->mailbox->cpu_addr;

		ethosn_dma_sync_for_cpu(core->allocator, core->mailbox);

		n += scnprintf(&buf[n], sizeof(buf) - n, "Severity      : %u\n",
			       mailbox->severity);
	}

	mutex_unlock(&core->mutex);

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
	struct ethosn_core *core = file->f_inode->i_private;
	ssize_t ret;
	ssize_t num_bytes_read;
	size_t buffer_entries_offset;
	size_t buffer_entries_count;
	size_t buffer_entries_size_bytes;
	loff_t read_buffer_offset;
	struct ethosn_profiling_buffer *buffer;
	uint32_t firmware_write_offset;

	/* Make sure the profiling buffer isn't deallocated underneath us */
	ret = mutex_lock_interruptible(&core->mutex);
	if (ret != 0)
		return ret;

	/* Report error if profiling is not enabled (i.e. no profiling buffer
	 * allocated)
	 */
	if (IS_ERR_OR_NULL(core->profiling.firmware_buffer)) {
		ret = -EINVAL;
		goto cleanup;
	}

	/* Calculate size etc. of the buffer. */
	buffer =
		(struct ethosn_profiling_buffer *)
		core->profiling.firmware_buffer->cpu_addr;

	buffer_entries_offset =
		offsetof(struct ethosn_profiling_buffer, entries);
	buffer_entries_count =
		(core->profiling.config.firmware_buffer_size -
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
	mutex_unlock(&core->mutex);

	return ret;
}

static void dfs_deinit(struct ethosn_core *core)
{
	debugfs_remove_recursive(core->debug_dir);
	core->debug_dir = NULL;
}

static void dfs_init(struct ethosn_core *core)
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
	snprintf(name, sizeof(name), "core%u", core->core_id);
	core->debug_dir = debugfs_create_dir(name, core->parent->debug_dir);
	if (IS_ERR_OR_NULL(core->debug_dir))
		return;

	/* Register map */
	core->debug_regset.regs = regs;
	core->debug_regset.nregs = ARRAY_SIZE(regs);
	core->debug_regset.base = core->top_regs;
	debugfs_create_regset32("registers", 0400, core->debug_dir,
				&core->debug_regset);

	/* Mailbox */
	debugfs_create_file("mailbox", 0400, core->debug_dir, core,
			    &mailbox_fops);

	/* Expose the firmware's profiling stream to user-space as a file. */
	debugfs_create_file("firmware_profiling", 0400, core->debug_dir,
			    core,
			    &firmware_profiling_fops);
}

/****************************************************************************
 * Device setup
 ****************************************************************************/

int ethosn_device_init(struct ethosn_core *core)
{
	int ret;

	/* Round up queue size to next power of 2 */
	core->queue_size = roundup_pow_of_two(ethosn_queue_size);

	/* Initialize debugfs */
	dfs_init(core);

	/* Load the firmware */
	ret = firmware_init(core);
	if (ret)
		goto remove_debufs;

	/* Allocate the mailbox structure */
	ret = mailbox_alloc(core);
	if (ret)
		goto deinit_firmware;

	/* For multi-npu, we test only the first NPU */
	if (ethosn_global_core_for_testing == NULL)
		ethosn_global_core_for_testing = core;

	/* Completed the device initialization */
	atomic_set(&core->init_done, 1);

	return 0;

deinit_firmware:
	ethosn_firmware_deinit(core);

remove_debufs:
	dfs_deinit(core);

	return ret;
}

void ethosn_device_deinit(struct ethosn_core *core)
{
	int ret;

	/* Verify that the core is initialized */
	if (atomic_read(&core->init_done) == 0)
		return;

	ret = mutex_lock_interruptible(&core->mutex);
	if (ret)
		return;

	/* Started the device de-initialization */
	atomic_set(&core->init_done, 0);

	ethosn_global_core_for_testing = NULL;

	ethosn_hard_reset(core);
	ethosn_firmware_deinit(core);
	ethosn_mailbox_free(core);
	dfs_deinit(core);
	mutex_unlock(&core->mutex);
	if (core->fw_and_hw_caps.data) {
		devm_kfree(core->parent->dev, core->fw_and_hw_caps.data);
		core->fw_and_hw_caps.data = NULL;
	}

	if (!IS_ERR_OR_NULL(core->profiling.firmware_buffer)) {
		ethosn_dma_unmap_and_free(
			core->allocator,
			core->profiling.firmware_buffer,
			ETHOSN_STREAM_WORKING_DATA);
		core->profiling.firmware_buffer = NULL;
	}

	if (!IS_ERR_OR_NULL(core->profiling.firmware_buffer_pending)) {
		ethosn_dma_unmap_and_free(
			core->allocator,
			core->profiling.firmware_buffer_pending,
			ETHOSN_STREAM_WORKING_DATA);
		core->profiling.firmware_buffer_pending = NULL;
	}
}

static void ethosn_release_reserved_mem(void *const dev)
{
	of_reserved_mem_device_release((struct device *)dev);
}

int ethosn_init_reserved_mem(struct device *const dev)
{
	int ret;

	ret = of_reserved_mem_device_init(dev);
	if (ret)
		return ret;

	return devm_add_action_or_reset(dev, ethosn_release_reserved_mem,
					dev);
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

bool ethosn_stashing_enabled(void)
{
	return stashing_enabled;
}

/* Exported for use by test module */
EXPORT_SYMBOL(ethosn_stashing_enabled);

struct ethosn_core *ethosn_get_global_core_for_testing(void)
{
	return ethosn_global_core_for_testing;
}

/* Exported for use by test module */
EXPORT_SYMBOL(ethosn_get_global_core_for_testing);
