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

#include "scylla_addr_fields_public.h"
#include "scylla_regs_public.h"
#include "ethosn_buffer.h"
#include "ethosn_device.h"
#include "ethosn_firmware.h"
#include "ethosn_network.h"
#include "ethosn_core.h"
#include "ethosn_smc.h"
#include "uapi/ethosn.h"

#include <linux/atomic.h>
#include <linux/fs.h>
#include <linux/idr.h>
#include <linux/interrupt.h>
#include <linux/io.h>
#include <linux/module.h>
#include <linux/of.h>
#include <linux/of_address.h>
#include <linux/of_platform.h>
#include <linux/pci.h>
#include <linux/platform_device.h>
#include <linux/slab.h>
#include <linux/string.h>
#include <linux/types.h>
#include <linux/uaccess.h>
#include <linux/version.h>
#include <linux/iommu.h>
#include <linux/pm_runtime.h>

#define ETHOSN_DRIVER_NAME    "ethosn"
#define ETHOSN_STR(s) #s
#define ETHOSN_DRIVER_VERSION_STR(major, minor, patch) \
	ETHOSN_STR(major) "." ETHOSN_STR(minor) "." ETHOSN_STR(patch)
#define ETHOSN_DRIVER_VERSION ETHOSN_DRIVER_VERSION_STR( \
		ETHOSN_KERNEL_MODULE_VERSION_MAJOR,	 \
		ETHOSN_KERNEL_MODULE_VERSION_MINOR,	 \
		ETHOSN_KERNEL_MODULE_VERSION_PATCH)

#define ETHOSN_MAX_DEVICES (1U << MINORBITS)

#define ETHOSN_PCI_VENDOR 0x13b5
#define ETHOSN_PCI_DEVICE 0x0001

#define ETHOSN_SMMU_MAX_ADDR_BITS 49

#define TOP_REG_SIZE \
	(TOP_REG(REGPAGE_MASK, REGOFFSET_MASK) - TOP_REG(0, 0) + 1)

/* Timeout in us when pinging the Ethos-N and waiting for a pong. */
#define ETHOSN_PING_TIMEOUT_US (10 * 1000 * 1000)
#define ETHOSN_PING_WAIT_US 1

#define ETHOSN_MAX_NUM_IRQS 3

static int ethosn_major;
static struct ethosn_device *ethosn_global_device_for_testing;
static DEFINE_IDA(ethosn_ida);

/* Ethos-N class infrastructure */
static struct class ethosn_class = {
	.name = ETHOSN_DRIVER_NAME,
};

static void __iomem *ethosn_map_iomem(const struct ethosn_core *const core,
				      const struct resource *const res,
				      const resource_size_t size)
{
	const resource_size_t rsize = !res ? 0 : resource_size(res);
	void __iomem *ptr;
	char *full_res_name;

	dev_dbg(core->dev,
		"Mapping resource. name=%s, start=%llx, size=%llu\n",
		res->name, res->start, size);

	/* Check resource size */
	if (rsize < size) {
		dev_err(core->dev,
			"'%s' resource not found or not big enough: %llu < %llu\n",
			res->name, rsize, size);

		return IOMEM_ERR_PTR(-EINVAL);
	}

	full_res_name = devm_kasprintf(core->parent->dev, GFP_KERNEL,
				       "%s : %s",
				       of_node_full_name(
					       core->parent->dev->of_node),
				       res->name);
	if (!full_res_name)
		return IOMEM_ERR_PTR(-ENOMEM);

	/* Reserve address space */
	if (!devm_request_mem_region(core->parent->dev, res->start, size,
				     full_res_name)) {
		dev_err(core->dev,
			"can't request region for resource %pR\n",
			res);

		return IOMEM_ERR_PTR(-EBUSY);
	}

	/* Map address space */
	ptr = devm_ioremap(core->parent->dev, res->start, size);
	if (IS_ERR(ptr))
		dev_err(core->dev,
			"failed to map '%s': start=%llu size=%llu\n",
			res->name, res->start, size);

	return ptr;
}

/**
 * rtrim() - Trim characters from end of string.
 *
 * Remove any character found in trim from the end of str.
 *
 * Return: Pointer to str.
 */
static char *rtrim(char *str,
		   const char *trim)
{
	char *end = str + strlen(str);

	/*
	 * Loop from end to begin of string. Break the loop if end char does not
	 * match any char in trim.
	 */
	while (end-- > str)
		if (!strchr(trim, *end))
			break;

	end[1] = '\0';

	return str;
}

/**
 * reset_profiling_counters() - Resets all profiling counters
 *
 */
static void reset_profiling_counters(struct ethosn_core *core)
{
	core->profiling.mailbox_messages_sent = 0;
	core->profiling.mailbox_messages_received = 0;
	core->profiling.rpm_suspend_count = 0;
	core->profiling.rpm_resume_count = 0;
	core->profiling.pm_suspend_count = 0;
	core->profiling.pm_resume_count = 0;
}

static void update_busy_core(struct ethosn_core *core)
{
	struct ethosn_device *ethosn = core->parent;

	uint32_t core_id = core->core_id;
	uint32_t core_mask = (1 << core_id);

	if ((ethosn->current_busy_cores & core_mask) == 0) {
		dev_err(core->dev,
			"Scheduler has scheduled an inference on the wrong core");
		ethosn->status_mask |= (1 << WRONG_CORE_SCHEDULE);
	} else {
		ethosn->current_busy_cores &= ~(1 << core->core_id);
	}

	/* If after clearing our core id, the current_busy_cores
	 * isn't zero, it means that another inference is executing
	 * concurrently.
	 */
	if (ethosn->current_busy_cores != 0) {
		dev_info(ethosn->dev, "Concurrent inferences detected");
		ethosn->status_mask |=
			(1 << CONCURRENT_INFERENCE_DETECTED);
	}
}

static int handle_message(struct ethosn_core *core)
{
	struct ethosn_message_header header;
	int ret;

	/* Read message from queue. Reserve one byte for end of string. */
	ret = ethosn_read_message(core, &header, core->mailbox_message,
				  core->queue_size - 1);
	if (ret <= 0)
		return ret;

	dev_dbg(core->dev, "Message. type=%u, length=%u\n",
		header.type, header.length);

	switch (header.type) {
	case ETHOSN_MESSAGE_REGION_RESPONSE: {
		struct ethosn_message_region_response *rsp =
			core->mailbox_message;

		dev_dbg(core->dev, "<- Region=%u. status=%u\n", rsp->id,
			rsp->status);

		break;
	}
	case ETHOSN_MESSAGE_MPU_ENABLE_RESPONSE: {
		dev_dbg(core->dev, "<- Mpu enabled\n");
		break;
	}
	case ETHOSN_MESSAGE_FW_HW_CAPS_RESPONSE: {
		dev_dbg(core->dev, "<- FW & HW Capabilities\n");

		/* Free previous memory (if any) */
		if (core->fw_and_hw_caps.data)
			devm_kfree(core->parent->dev,
				   core->fw_and_hw_caps.data);

		/* Allocate new memory */
		core->fw_and_hw_caps.data = devm_kzalloc(core->parent->dev,
							 header.length,
							 GFP_KERNEL);
		if (!core->fw_and_hw_caps.data)
			return -ENOMEM;

		/* Copy data returned from firmware into our storage, so that it
		 * can be passed to user-space when queried
		 */
		memcpy(core->fw_and_hw_caps.data, core->mailbox_message,
		       header.length);
		core->fw_and_hw_caps.size = header.length;
		break;
	}
	case ETHOSN_MESSAGE_INFERENCE_RESPONSE: {
		struct ethosn_message_inference_response *rsp =
			core->mailbox_message;
		struct ethosn_inference *inference = (void *)rsp->user_argument;
		int status;

		dev_dbg(core->dev,
			"<- Inference. user_arg=0x%llx, status=%u\n",
			rsp->user_argument, rsp->status);

		status = rsp->status == ETHOSN_INFERENCE_STATUS_OK ?
			 ETHOSN_INFERENCE_COMPLETED : ETHOSN_INFERENCE_ERROR;

		update_busy_core(core);
		ethosn_network_poll(core, inference, status);
		break;
	}
	case ETHOSN_MESSAGE_PONG: {
		++core->num_pongs_received;
		dev_dbg(core->dev, "<- Pong\n");
		break;
	}
	case ETHOSN_MESSAGE_TEXT: {
		struct ethosn_message_text *text = core->mailbox_message;
		char *eos = (char *)core->mailbox_message + header.length;

		/* Null terminate str. One byte has been reserved for this. */
		*eos = '\0';

		dev_info(core->dev, "<- Text. text=\"%s\"\n",
			 rtrim(text->text, "\n"));
		break;
	}
	case ETHOSN_MESSAGE_CONFIGURE_PROFILING_ACK: {
		dev_dbg(core->dev,
			"<- ETHOSN_MESSAGE_CONFIGURE_PROFILING_ACK\n");
		ethosn_configure_firmware_profiling_ack(core);
		break;
	}
	default: {
		dev_warn(core->dev,
			 "Unsupported message type. Type=%u, Length=%u, ret=%d.\n",
			 header.type, header.length, ret);
		break;
	}
	}

	return 1;
}

/**
 * ethosn_irq_bottom() - IRQ bottom handler
 * @work:	Work structure part of Ethos-N core
 *
 * Execute bottom half of interrupt in process context.
 */
static void ethosn_irq_bottom(struct work_struct *work)
{
	struct ethosn_core *core = container_of(work, struct ethosn_core,
						irq_work);
	struct dl1_irq_status_r status;
	int ret;

	ret = mutex_lock_interruptible(&core->mutex);
	if (ret)
		return;

	if (atomic_read(&core->init_done) == 0)
		goto end;

	/* Read and clear the IRQ status bits. */
	status.word = atomic_xchg(&core->irq_status, 0);

	dev_dbg(core->dev,
		"Irq bottom, word=0x%08x, err=%u, debug=%u, job=%u core_id=%u\n",
		status.word, status.bits.setirq_err, status.bits.setirq_dbg,
		status.bits.setirq_job, core->core_id);

	/* Handle mailbox messages. Note we do this before checking for errors
	 * so that we get as much debugging
	 * information from the firmware as possible before resetting it.
	 */
	do {
		ret = handle_message(core);
	} while (ret > 0);

	/* Inference failed. Reset firmware. */
	if (status.bits.setirq_err ||
	    status.bits.tol_err || status.bits.func_err ||
	    status.bits.rec_err || status.bits.unrec_err) {
		/* Failure may happen before the firmware is deemed running. */
		ethosn_dump_gps(core);

		dev_warn(core->dev,
			 "Reset core due to error interrupt. irq_status=0x%08x\n",
			 status.word);

		if (core->firmware_running) {
			(void)ethosn_reset_and_start_ethosn(core);
			ethosn_network_poll(core, core->current_inference,
					    ETHOSN_INFERENCE_ERROR);
		}
	}

end:

	mutex_unlock(&core->mutex);
}

/**
 * ethosn_irq_top() - IRQ top handler
 * @irq:	IRQ number.
 * @dev:	User argument, Ethos-N core.
 *
 * Handle IRQ in interrupt context. Clear the interrupt and trigger bottom
 * half to handle the rest of the interrupt.
 */
static irqreturn_t ethosn_irq_top(const int irq,
				  void *dev)
{
	struct ethosn_core *const core = dev;
	struct dl1_irq_status_r status;
	struct dl1_clrirq_ext_r clear = { .word = 0 };

	status.word = ethosn_read_top_reg(core, DL1_RP, DL1_IRQ_STATUS);

	/* Save the IRQ status for the bottom half. */
	atomic_or(status.word, &core->irq_status);

	/* Job bit is currently not correctly set by hardware. */
	clear.bits.err = status.bits.setirq_err;
	clear.bits.debug = status.bits.setirq_dbg;
	clear.bits.job = status.bits.setirq_job;

	/* This was not meant for us */
	if (status.word == 0)
		return IRQ_NONE;

	/* Clear interrupt. */
	ethosn_write_top_reg(core, DL1_RP, DL1_CLRIRQ_EXT,
			     clear.word);

	/* Defer to work queue. */
	queue_work(core->irq_wq, &core->irq_work);

	return IRQ_HANDLED;
}

/**
 * ethosn_init_interrupt() - Register IRQ handlers
 * @core: Ethos-N core
 * @irq_numbers: List of IRQ numbers that we should listen to.
 * @irq_flags: List of the IRQF_TRIGGER_ flags to use for each corresponding irq
 *             number in the irq_numbers array.
 * @num_irqs: Number of valid entries in the irq_numbers and irq_flags arrays.
 *
 * Return:
 * * 0 - Success
 * * Negative error code
 */
static int ethosn_init_interrupt(struct ethosn_core *const core,
				 const int irq_numbers[ETHOSN_MAX_NUM_IRQS],
				 const unsigned long
				 irq_flags[ETHOSN_MAX_NUM_IRQS],
				 unsigned int num_irqs)
{
	int ret;
	int irq_idx;

	/* Create a work queue to handle the IRQs that come in. We do only a
	 * minimal amount of work in the IRQ handler itself ("ethosn_irq_top")
	 * and
	 * defer the rest of the work to this work queue ("ethosn_irq_bottom").
	 * Note we must do this before registering any handlers, as the handler
	 * callback uses this work queue.
	 */
	core->irq_wq = create_singlethread_workqueue("ethosn_workqueue");
	if (!core->irq_wq) {
		dev_err(core->dev, "Failed to create work queue\n");

		return -EINVAL;
	}

	INIT_WORK(&core->irq_work, ethosn_irq_bottom);

	/* Register an IRQ handler for each number requested.
	 * We use the same handler for each of these as we check the type of
	 * interrupt using the Ethos-N's IRQ status register, and so don't need
	 * to
	 * differentiate based on the IRQ number.
	 */
	for (irq_idx = 0; irq_idx < num_irqs; ++irq_idx) {
		const int irq_num = irq_numbers[irq_idx];
		const unsigned long this_irq_flags = irq_flags[irq_idx];

		dev_dbg(core->dev, "Requesting IRQ %d with flags 0x%lx\n",
			irq_num, this_irq_flags);

		ret = devm_request_irq(core->parent->dev, irq_num,
				       &ethosn_irq_top,
				       this_irq_flags, ETHOSN_DRIVER_NAME,
				       core);
		if (ret) {
			dev_err(core->dev, "Failed to request IRQ %d\n",
				irq_num);

			return ret;
		}
	}

	return 0;
}

static ssize_t num_cores_show(struct device *dev,
			      struct device_attribute *attr,
			      char *buf)
{
	struct ethosn_device *ethosn = dev_get_drvdata(dev);

	return scnprintf(buf, PAGE_SIZE, "%d\n", ethosn->num_cores);
}

static ssize_t status_mask_show(struct device *dev,
				struct device_attribute *attr,
				char *buf)
{
	struct ethosn_device *ethosn = dev_get_drvdata(dev);

	return scnprintf(buf, PAGE_SIZE, "%#x\n",
			 ethosn->status_mask);
}

static const DEVICE_ATTR_RO(num_cores);
static const DEVICE_ATTR_RO(status_mask);

static const struct attribute *attrs[] = {
	&dev_attr_num_cores.attr,
	&dev_attr_status_mask.attr,
	NULL
};

/**
 * ethosn_open() - Open the Ethos-N core node.
 *
 * Open the device node and stored the Ethos-N core handle in the file private
 * data.
 */
static int ethosn_open(struct inode *inode,
		       struct file *file)
{
	file->private_data =
		container_of(inode->i_cdev, struct ethosn_device, cdev);

	return nonseekable_open(inode, file);
}

static void print_buffer_info(struct ethosn_device *ethosn,
			      const char *prefix,
			      u32 ninfos,
			      const struct ethosn_buffer_info __user *infos)
{
	char buf[200];
	size_t n = 0;
	u32 i;
	const char *delim = "";

	n += scnprintf(&buf[n], sizeof(buf) - n, "    %s: ", prefix);

	for (i = 0; i < ninfos; ++i) {
		struct ethosn_buffer_info info;

		if (copy_from_user(&info, &infos[i], sizeof(info)))
			break;

		n += scnprintf(&buf[n], sizeof(buf) - n, "%s{%u, %u, %u}",
			       delim, info.id, info.offset, info.size);

		delim = ", ";
	}

	dev_dbg(ethosn->dev, "%s\n", buf);
}

/**
 * ethosn_ioctl() - Take commands from user space
 * @filep:	File struct.
 * @cmd:	See IOCTL in ethosn.h.
 * @arg:	Command argument.
 *
 * Return: 0 on success, else error code.
 */
static long ethosn_ioctl(struct file *const filep,
			 unsigned int cmd,
			 unsigned long arg)
{
	struct ethosn_device *ethosn = filep->private_data;
	void __user *const udata = (void __user *)arg;
	int ret = 0;

	switch (cmd) {
	case ETHOSN_IOCTL_GET_VERSION: {
		struct ethosn_kernel_module_version act_version = {
			.major = ETHOSN_KERNEL_MODULE_VERSION_MAJOR,
			.minor = ETHOSN_KERNEL_MODULE_VERSION_MINOR,
			.patch = ETHOSN_KERNEL_MODULE_VERSION_PATCH,
		};

		if (copy_to_user(udata, &act_version, sizeof(act_version))) {
			ret = -EFAULT;
			break;
		}

		break;
	}
	case ETHOSN_IOCTL_CREATE_BUFFER: {
		struct ethosn_buffer_req buf_req;

		if (copy_from_user(&buf_req, udata, sizeof(buf_req))) {
			ret = -EFAULT;
			break;
		}

		ret = mutex_lock_interruptible(&ethosn->mutex);
		if (ret)
			break;

		dev_dbg(ethosn->dev,
			"IOCTL: Create buffer. size=%u, flags=0x%x\n",
			buf_req.size, buf_req.flags);

		ret = ethosn_buffer_register(ethosn, &buf_req);

		dev_dbg(ethosn->dev,
			"IOCTL: Created buffer. fd=%d\n", ret);

		mutex_unlock(&ethosn->mutex);

		break;
	}
	case ETHOSN_IOCTL_REGISTER_NETWORK: {
		struct ethosn_network_req net_req;

		if (copy_from_user(&net_req, udata, sizeof(net_req))) {
			ret = -EFAULT;
			break;
		}

		ret = mutex_lock_interruptible(&ethosn->mutex);
		if (ret)
			break;

		dev_dbg(ethosn->dev,
			"IOCTL: Register network. num_dma=%u, num_cu=%u, num_inputs=%u, num_outputs=%u\n",
			net_req.dma_buffers.num,
			net_req.cu_buffers.num,
			net_req.input_buffers.num,
			net_req.output_buffers.num);

		print_buffer_info(ethosn, "dma", net_req.dma_buffers.num,
				  net_req.dma_buffers.info);
		print_buffer_info(ethosn, "cu", net_req.cu_buffers.num,
				  net_req.cu_buffers.info);
		print_buffer_info(ethosn, "intermediate",
				  net_req.intermediate_buffers.num,
				  net_req.intermediate_buffers.info);
		print_buffer_info(ethosn, "input", net_req.input_buffers.num,
				  net_req.input_buffers.info);
		print_buffer_info(ethosn, "output", net_req.output_buffers.num,
				  net_req.output_buffers.info);

		ret = ethosn_network_register(ethosn, &net_req);

		dev_dbg(ethosn->dev,
			"IOCTL: Registered network. fd=%d\n", ret);

		mutex_unlock(&ethosn->mutex);

		break;
	}
	case ETHOSN_IOCTL_FW_HW_CAPABILITIES: {
		/* In the case of multicore, we get the hardware capabilities
		 * for core[0]. As both the cores are of the same variant,
		 * so it should just be fine.
		 */
		struct ethosn_core *core = ethosn->core[0];

		ret = mutex_lock_interruptible(&core->mutex);
		if (ret)
			break;

		/* If the user provided a NULL pointer then simply return the
		 * size of the data.
		 * If they provided a valid pointer then copy the data to them
		 */
		if (!udata) {
			ret = core->fw_and_hw_caps.size;
		} else {
			if (copy_to_user(udata, core->fw_and_hw_caps.data,
					 core->fw_and_hw_caps.size)) {
				dev_warn(core->dev,
					 "Failed to copy firmware and hardware capabilities to user.\n");
				ret = -EFAULT;
			} else {
				ret = 0;
			}
		}

		/* It may happen that users ask for capabilities before the
		 * firmware has responded, in that case a fault is reported
		 */
		if (core->fw_and_hw_caps.size == 0)
			ret = -EAGAIN;

		mutex_unlock(&core->mutex);

		break;
	}
	case ETHOSN_IOCTL_CONFIGURE_PROFILING: {
		struct ethosn_core *core = ethosn->core[0];
		struct ethosn_profiling_config new_config;

		if (!ethosn_profiling_enabled()) {
			ret = -EACCES;
			dev_err(core->dev, "Profiling: access denied\n");
			break;
		}

		pm_runtime_get_sync(core->dev);

		ret = mutex_lock_interruptible(&core->mutex);
		if (ret)
			goto configure_profiling_put;

		if (copy_from_user(&new_config, udata, sizeof(new_config))) {
			ret = -EFAULT;
			goto configure_profiling_mutex;
		}

		dev_dbg(core->dev,
			"IOCTL: Configure profiling. enable_profiling=%u, firmware_buffer_size=%u num_hw_counters=%d\n",
			new_config.enable_profiling,
			new_config.firmware_buffer_size,
			new_config.num_hw_counters);

		/* Forward new state to the firmware */
		ret = ethosn_configure_firmware_profiling(core, &new_config);

		if (ret != 0)
			goto configure_profiling_mutex;

		if (core->profiling.config.enable_profiling &&
		    !new_config.enable_profiling)
			reset_profiling_counters(core);

		core->profiling.config = new_config;

configure_profiling_mutex:
		mutex_unlock(&core->mutex);
configure_profiling_put:
		pm_runtime_mark_last_busy(core->dev);
		pm_runtime_put(core->dev);

		break;
	}
	case ETHOSN_IOCTL_GET_COUNTER_VALUE: {
		struct ethosn_core *core = ethosn->core[0];
		enum ethosn_poll_counter_name counter_name;

		ret = mutex_lock_interruptible(&core->mutex);
		if (ret)
			break;

		if (!core->profiling.config.enable_profiling) {
			ret = -ENODATA;
			dev_err(core->dev, "Profiling counter: no data\n");
			goto get_counter_value_end;
		}

		if (copy_from_user(&counter_name, udata,
				   sizeof(counter_name))) {
			ret = -EFAULT;
			dev_err(core->dev,
				"Profiling counter: error in copy_from_user\n");
			goto get_counter_value_end;
		}

		switch (counter_name) {
		case ETHOSN_POLL_COUNTER_NAME_MAILBOX_MESSAGES_SENT:
			ret = core->profiling.mailbox_messages_sent;
			break;
		case ETHOSN_POLL_COUNTER_NAME_MAILBOX_MESSAGES_RECEIVED:
			ret = core->profiling.mailbox_messages_received;
			break;
		case ETHOSN_POLL_COUNTER_NAME_RPM_SUSPEND:
			ret = core->profiling.rpm_suspend_count;
			break;
		case ETHOSN_POLL_COUNTER_NAME_RPM_RESUME:
			ret = core->profiling.rpm_resume_count;
			break;
		case ETHOSN_POLL_COUNTER_NAME_PM_SUSPEND:
			ret = core->profiling.pm_suspend_count;
			break;
		case ETHOSN_POLL_COUNTER_NAME_PM_RESUME:
			ret = core->profiling.pm_resume_count;
			break;
		default:
			ret = -EINVAL;
			dev_err(core->dev,
				"Profiling counter: invalid counter_name\n");
			break;
		}

get_counter_value_end:
		mutex_unlock(&core->mutex);

		break;
	}
	case ETHOSN_IOCTL_GET_CLOCK_FREQUENCY: {
		struct ethosn_core *core = ethosn->core[0];

		ret = mutex_lock_interruptible(&core->mutex);
		if (ret)
			break;

		dev_dbg(core->dev,
			"IOCTL: Get clock frequency\n");

		ret = ethosn_clock_frequency();

		mutex_unlock(&core->mutex);

		break;
	}
	case ETHOSN_IOCTL_PING: {
		struct ethosn_core *core = ethosn->core[0];
		uint32_t num_pongs_before = core->num_pongs_received;
		int timeout;

		pm_runtime_get_sync(core->dev);

		/* Send a ping */
		ret = mutex_lock_interruptible(&core->mutex);
		if (ret)
			goto ping_put;

		ret = ethosn_send_ping(core);

		mutex_unlock(&core->mutex);

		if (ret != 0)
			goto ping_put;

		/* Wait for a pong to come back, with a timeout. */
		for (timeout = 0; timeout < ETHOSN_PING_TIMEOUT_US;
		     timeout += ETHOSN_PING_WAIT_US) {
			if (core->num_pongs_received > num_pongs_before)
				break;

			udelay(ETHOSN_PING_WAIT_US);
		}

		if (timeout >= ETHOSN_PING_TIMEOUT_US) {
			dev_err(core->dev,
				"Timeout while waiting for device to pong\n");
			ret = -ETIME;
			goto ping_put;
		}

		ret = 0;
ping_put:
		pm_runtime_mark_last_busy(core->dev);
		pm_runtime_put(core->dev);

		break;
	}
	default: {
		ret = -EINVAL;
		break;
	}
	}

	return ret;
}

static void ethosn_device_release(void *const opaque)
{
	struct ethosn_device *const ethosn = opaque;
	struct cdev *const cdev = &ethosn->cdev;
	int i = 0;

	while (i < ethosn->num_cores) {
		ethosn_set_power_ctrl(ethosn->core[i], false);
		if (ethosn->core[i]->irq_wq)
			destroy_workqueue(ethosn->core[i]->irq_wq);

		++i;
	}

	sysfs_remove_files(&ethosn->dev->kobj, attrs);
	debugfs_remove_recursive(ethosn->debug_dir);

	device_destroy(&ethosn_class, cdev->dev);
	cdev_del(cdev);
	ida_simple_remove(&ethosn_ida, MINOR(cdev->dev));
}

static int ethosn_device_create(struct ethosn_device *ethosn,
				int id)
{
	static const struct file_operations ethosn_fops = {
		.owner          = THIS_MODULE,
		.open           = &ethosn_open,
		.unlocked_ioctl = &ethosn_ioctl,
#ifdef CONFIG_COMPAT
		.compat_ioctl   = &ethosn_ioctl,
#endif
	};

	struct device *sysdev;
	dev_t devt;
	int ret;

	devt = MKDEV(ethosn_major, id);

	cdev_init(&ethosn->cdev, &ethosn_fops);
	ethosn->cdev.owner = THIS_MODULE;

	ret = cdev_add(&ethosn->cdev, devt, 1);
	if (ret) {
		dev_err(ethosn->dev, "unable to add character device\n");

		return ret;
	}

	sysdev = device_create(&ethosn_class, ethosn->dev, devt, ethosn,
			       "ethosn%d", id);
	if (IS_ERR(sysdev)) {
		dev_err(ethosn->dev, "device register failed\n");
		ret = PTR_ERR(sysdev);
		goto err_remove_chardev;
	}

	ret = sysfs_create_files(&ethosn->dev->kobj, attrs);
	if (ret)
		goto destroy_device;

	return devm_add_action_or_reset(ethosn->dev,
					ethosn_device_release,
					ethosn);

destroy_device:
	device_destroy(&ethosn_class, ethosn->cdev.dev);
err_remove_chardev:
	cdev_del(&ethosn->cdev);

	return ret;
}

/**
 * ethosn_driver_probe() - Do common probing functionality
 * @core:	ethosn core struct
 * @top_regs:	Register memory resource
 * @irq_numbers: List of IRQ numbers that we should listen to.
 * @irq_flags: List of the IRQF_TRIGGER_ flags to use for each corresponding irq
 *             number in the irq_numbers array.
 * @num_irqs: Number of valid entries in the irq_numbers and irq_flags arrays.
 * @force_firmware_level_interrupts: Whether to tell the firmware to send
 *                                   level-sensitive interrupts in all cases.
 */
static int ethosn_driver_probe(struct ethosn_core *core,
			       const struct resource *const top_regs,
			       const int irq_numbers[ETHOSN_MAX_NUM_IRQS],
			       const unsigned long
			       irq_flags[ETHOSN_MAX_NUM_IRQS],
			       unsigned int num_irqs,
			       bool force_firmware_level_interrupts)
{
	struct ethosn_profiling_config config = {};
	const phys_addr_t core_addr = top_regs->start;
	int ret = ethosn_smc_version_check(core->dev);

#ifdef ETHOSN_NS

	/*
	 * If the SiP service is available verify the NPU's
	 * secure status. If not, assume it's non-secure.
	 */
	ret = !ret ? ethosn_smc_is_secure(core->dev, core_addr) : 0;
	if (ret) {
		if (ret == 1) {
			dev_err(core->dev,
				"Device in secure mode, non-secure kernel not supported.\n");
			ret = -EPERM;
		}

		return ret;
	}

#else
	if (ret) {
		dev_err(core->dev,
			"SiP service required for secure kernel.\n");

		return -EPERM;
	}

#endif

	mutex_init(&core->mutex);

	core->phys_addr = core_addr;
	core->top_regs = ethosn_map_iomem(core, top_regs, TOP_REG_SIZE);
	if (IS_ERR(core->top_regs))
		return PTR_ERR(core->top_regs);

	ret = ethosn_init_interrupt(core, irq_numbers, irq_flags, num_irqs);
	if (ret)
		return ret;

	/* Remember that we need to tell the firmware to use level interrupts.
	 * We can't do this now because the Ethos-N hasn't been turned on yet.
	 */
	core->force_firmware_level_interrupts =
		force_firmware_level_interrupts;

	core->allocator = ethosn_dma_allocator_create(core->dev);
	if (IS_ERR_OR_NULL(core->allocator))
		return PTR_ERR(core->allocator);

	/* Default to profiling disabled */
	config.enable_profiling = false;
	core->profiling.config = config;
	reset_profiling_counters(core);

	core->profiling.is_waiting_for_firmware_ack = false;
	core->profiling.firmware_buffer = NULL;
	core->profiling.firmware_buffer_pending = NULL;

	ret = ethosn_device_init(core);
	if (ret)
		goto destroy_allocator;

	ret = ethosn_reset_and_start_ethosn(core);
	if (ret)
		goto device_deinit;

	pm_runtime_mark_last_busy(core->dev);
	pm_runtime_put_autosuspend(core->dev);

	dev_info(core->dev, "Ethos-N is running\n");

	return 0;

device_deinit:
	ethosn_device_deinit(core);
destroy_allocator:
	ethosn_dma_allocator_destroy(&core->allocator);

	return ret;
}

/**
 * ethosn_pdev_num_cores() - Get the number of cores
 * @pdev: Platform device
 * Return: count of cores
 */
static unsigned int ethosn_pdev_num_cores(struct platform_device *pdev)
{
	return of_get_available_child_count(pdev->dev.of_node);
}

/*****************************************************************************
 * Platform device
 *****************************************************************************/

/**
 * ethosn_pdev_enum_interrupts() - Do platform specific interrupt enumeration
 * @pdev: Platform device
 * @irq_numbers: List of IRQ numbers that we should listen to.
 * @irq_flags: List of the IRQF_TRIGGER_ flags to use for each corresponding irq
 *             number in the irq_numbers array.
 * @force_firmware_level_interrupts: Whether to tell the firmware to send
 *                                   level-sensitive interrupts in all cases.
 * Return:
 * * Number of valid entries in the irq_numbers and irq_flags arrays.
 * * -EINVAL - Invalid number of IRQ in dtb
 */
static int ethosn_pdev_enum_interrupts(struct platform_device *pdev,
				       int *irq_numbers,
				       unsigned long *irq_flags,
				       bool *force_firmware_level_interrupts)
{
	int num_irqs = 0;
	int irq_count = platform_irq_count(pdev);
	int irq_idx;

	if (irq_count > ETHOSN_MAX_NUM_IRQS) {
		dev_err(&pdev->dev, "Invalid number of IRQs %d > %d", irq_count,
			ETHOSN_MAX_NUM_IRQS);

		return -EINVAL;
	}

	/* Get details of all the IRQs defined in the device tree.
	 * Depending on the system configuration there may be just one
	 * or several.
	 * It is also possible that more than one IRQ is combined onto
	 * the same line.
	 */
	for (irq_idx = 0;
	     irq_idx < irq_count;
	     ++irq_idx) {
		int irq_idx_existing;
		int irq_number;
		struct resource *resource =
			platform_get_resource(pdev, IORESOURCE_IRQ,
					      irq_idx);
		if (!resource) {
			dev_err(&pdev->dev,
				"platform_get_resource failed for IRQ index %d.\n",
				irq_idx);

			return -EINVAL;
		}

		irq_number = platform_get_irq(pdev, irq_idx);
		if (irq_number < 0) {
			dev_err(&pdev->dev,
				"platform_get_irq failed for IRQ index %d.\n",
				irq_idx);

			return -EINVAL;
		}

		/* Check if we have already seen an IRQ with the same
		 * number. (i.e. a line that is shared for more than
		 * one interrupt).
		 */
		for (irq_idx_existing = 0; irq_idx_existing < num_irqs;
		     ++irq_idx_existing)
			if (irq_numbers[irq_idx_existing] == irq_number)
				break;

		if (irq_idx_existing == num_irqs) {
			/* Not a shared line.
			 * Store the irq number and flags for use by
			 * ethosn_driver_probe. The flags (i.e. edge
			 * or level) depends on the interrupt name,
			 * as different Ethos-N interrupts use
			 * different types
			 */
			irq_numbers[num_irqs] = irq_number;
			if (strcmp(resource->name, "job") == 0) {
				/* Spec defines JOB interrupt to be
				 * EDGE
				 */
				irq_flags[num_irqs] =
					IRQF_SHARED |
					IRQF_TRIGGER_RISING;
			} else if (strcmp(resource->name, "err") == 0) {
				/* Spec defines ERR interrupt to be
				 * LEVEL
				 */
				irq_flags[num_irqs] = IRQF_SHARED |
						      IRQF_TRIGGER_HIGH;
			} else if (strcmp(resource->name, "debug") ==
				   0) {
				/* Spec defines DEBUG interrupt to be
				 * EDGE
				 */
				irq_flags[num_irqs] =
					IRQF_SHARED |
					IRQF_TRIGGER_RISING;
			} else {
				dev_err(&pdev->dev,
					"Unknown interrupt name '%s'.\n",
					resource->name);

				return -EINVAL;
			}

			++num_irqs;
		} else {
			/* If the line is shared, then this must be a
			 * level-based interrupt, so modify any
			 * previously set flags.
			 * We must also tell the firmware to send level-
			 * sensitive interrupts in all cases, so they
			 * can be safely ORed.
			 */
			irq_flags[irq_idx_existing] =
				IRQF_SHARED | IRQF_TRIGGER_HIGH;
			*force_firmware_level_interrupts = true;
		}
	}

	return num_irqs;
}

/**
 * ethosn_pdev_remove() - Do platform specific remove
 * @pdev: Platform device
 * Return:
 * * 0 - OK
 */
static int ethosn_pdev_remove(struct platform_device *pdev)
{
	struct ethosn_device *ethosn =
		dev_get_drvdata(&pdev->dev);

	/* Force depopulating children */
	of_platform_depopulate(&pdev->dev);

	ethosn_dma_allocator_destroy(&ethosn->allocator);

	return 0;
}

/**
 * ethosn_pdev_probe() - Do platform specific probing
 * @pdev: Platform device
 * Return:
 * * 0 - OK
 * * -EINVAL - Invalid argument
 */
static int ethosn_pdev_probe(struct platform_device *pdev)
{
	int ret = -ENOMEM;
	int num_irqs;
	int irq_numbers[ETHOSN_MAX_NUM_IRQS];
	unsigned long irq_flags[ETHOSN_MAX_NUM_IRQS];
	bool force_firmware_level_interrupts = false;
	unsigned int num_of_npus = 0;
	int resource_idx = 0;
	struct ethosn_device *ethosn = NULL;
	int platform_id = -1;
	char name[16];

	dma_set_mask_and_coherent(&pdev->dev,
				  DMA_BIT_MASK(ETHOSN_SMMU_MAX_ADDR_BITS));

	num_of_npus = ethosn_pdev_num_cores(pdev);

	if (num_of_npus == 0) {
		dev_info(&pdev->dev, "Failed to probe any NPU\n");

		return -EINVAL;
	}

	platform_id = ida_simple_get(&ethosn_ida, 0,
				     ETHOSN_MAX_DEVICES,
				     GFP_KERNEL);
	if (platform_id < 0)
		return platform_id;

	/* We need to allocate the parent device (ie struct
	 * ethosn_parent_device) only for the first time.
	 */
	dev_dbg(&pdev->dev, "Probing Ethos-N device id %u with %u core%s\n",
		platform_id, num_of_npus, num_of_npus > 1 ? "s" : "");

	ethosn = devm_kzalloc(&pdev->dev, sizeof(*ethosn),
			      GFP_KERNEL);
	if (!ethosn)
		goto err_early_exit;

	ethosn_global_device_for_testing = ethosn;

	ethosn->parent_id = platform_id;
	ethosn->dev = &pdev->dev;

	ethosn->current_busy_cores = 0;
	ethosn->status_mask = 0;

	snprintf(name, sizeof(name), "ethosn%u", ethosn->parent_id);
	ethosn->debug_dir = debugfs_create_dir(name, NULL);

	/* Create a top level allocator for parent device
	 */
	ethosn->allocator = ethosn_dma_allocator_create(ethosn->dev);
	if (IS_ERR_OR_NULL(ethosn->allocator))
		goto err_free_ethosn;

	INIT_LIST_HEAD(&ethosn->queue.inference_queue);

	/* Allocate space for num_of_npus ethosn cores */
	ethosn->core = devm_kzalloc(&pdev->dev,
				    (sizeof(struct ethosn_core *) *
				     num_of_npus),
				    GFP_KERNEL);

	if (!ethosn->core)
		goto err_destroy_allocator;

	dev_set_drvdata(&pdev->dev, ethosn);

	/* We need to populate child platform devices once parent
	 * device has been allocated and passed as device driver data
	 */
	dev_dbg(&pdev->dev, "Populating children\n");

	ret = of_platform_default_populate(pdev->dev.of_node, NULL, &pdev->dev);
	if (ret) {
		dev_err(&pdev->dev, "Failed to populate child devices\n");

		goto err_free_core_list;
	}

	/*
	 * Child device probing errors are not propagated back to the populate
	 * call so verify that the expected number of cores were setup
	 */
	if (ethosn->num_cores != num_of_npus) {
		dev_err(&pdev->dev, "Failed to populate all child devices\n");

		goto err_depopulate_device;
	}

	dev_dbg(&pdev->dev, "Populated %d children\n", ethosn->num_cores);

	mutex_init(&ethosn->mutex);

	mutex_init(&ethosn->queue.inference_queue_mutex);

	/* Currently we assume that the reserved memory is
	 * common to all the NPUs
	 */
	dev_dbg(&pdev->dev, "Init reserved mem\n");

	ret = ethosn_init_reserved_mem(&pdev->dev);
	if (ret)
		dev_dbg(&pdev->dev,
			"Reserved mem not present or init failed\n");

	if (!ethosn_smmu_available(&pdev->dev))
		if (ret)
			goto err_depopulate_device;

	/* Enumerate irqs */
	num_irqs = ethosn_pdev_enum_interrupts(
		pdev,
		irq_numbers,
		irq_flags,
		&force_firmware_level_interrupts);

	if (num_irqs < 0) {
		ret = num_irqs;
		goto err_depopulate_device;
	}

	/* At this point, all child device have been populated.
	 * Now the ethosn cores can be properly probed.
	 */
	for (resource_idx = 0; resource_idx < ethosn->num_cores;
	     ++resource_idx) {
		struct resource *top_regs = platform_get_resource(
			pdev,
			IORESOURCE_MEM,
			resource_idx);

		if (!ethosn->core[resource_idx]->dev) {
			dev_err(&pdev->dev,
				"NULL ethosn-core device reference");

			ret = -EINVAL;
			goto err_depopulate_device;
		}

		ret = ethosn_driver_probe(ethosn->core[resource_idx], top_regs,
					  irq_numbers, irq_flags, num_irqs,
					  force_firmware_level_interrupts);
		if (ret)
			goto err_depopulate_device;
	}

	ret = ethosn_device_create(ethosn, platform_id);
	if (ret)
		goto err_depopulate_device;

	return 0;

err_depopulate_device:
	ethosn_pdev_remove(pdev);
err_free_core_list:
	devm_kfree(&pdev->dev, ethosn->core);
err_destroy_allocator:
	ethosn_dma_allocator_destroy(&ethosn->allocator);
err_free_ethosn:
	devm_kfree(&pdev->dev, ethosn);
err_early_exit:
	ida_simple_remove(&ethosn_ida, platform_id);

	return ret;
}

struct ethosn_device *ethosn_get_global_device_for_testing(void)
{
	return ethosn_global_device_for_testing;
}

/* Exported for use by test module */
EXPORT_SYMBOL(ethosn_get_global_device_for_testing);

static const struct of_device_id ethosn_pdev_match[] = {
	{ .compatible = ETHOSN_DRIVER_NAME },
	{ /* Sentinel */ },
};

MODULE_DEVICE_TABLE(of, ethosn_pdev_match);

static struct platform_driver ethosn_pdev_driver = {
	.probe                  = &ethosn_pdev_probe,
	.remove                 = &ethosn_pdev_remove,
	.driver                 = {
		.name           = ETHOSN_DRIVER_NAME,
		.owner          = THIS_MODULE,
		.of_match_table = of_match_ptr(ethosn_pdev_match),
	},
};

/*****************************************************************************
 * PCI device
 *****************************************************************************/

/**
 * ethosn_pci_probe() - Do PCI specific probing
 * @pdev: Platform device
 * Return:
 * * 0 - OK
 * * -EINVAL - Invalid argument
 */
static int ethosn_pci_probe(struct pci_dev *pdev,
			    const struct pci_device_id *id)
{
	/* The PCI driver does not seem to use the dts file and so we cannot
	 * query for the IRQ setup. We only use PCI for the qemu environment
	 * so we hardcode the interrupts here.
	 * As the PCI driver does not parse the dts, so it is assumed to work
	 * for single core NPU only.
	 */
	int irq_numbers[ETHOSN_MAX_NUM_IRQS] = { pdev->irq };
	unsigned long irq_flags[ETHOSN_MAX_NUM_IRQS] = {
		IRQF_SHARED | IRQF_TRIGGER_HIGH
	};
	int num_irqs = 1;
	struct ethosn_device *ethosn;

	dma_set_mask_and_coherent(&pdev->dev, ETHOSN_REGION_MASK);

	ethosn = devm_kzalloc(&pdev->dev, sizeof(*ethosn), GFP_KERNEL);
	if (!ethosn)
		return -ENOMEM;

	/* Allocating for the parent device. We assume for a single core NPU
	 * only.
	 */
	ethosn->dev = &pdev->dev;
	dev_set_drvdata(&pdev->dev, ethosn);
	ethosn->num_cores = 1;

	/* Allocating the child device (ie struct ethosn_core)
	 */
	ethosn->core[0] = devm_kzalloc(&pdev->dev, sizeof(struct ethosn_core),
				       GFP_KERNEL);
	if (!ethosn->core[0])
		return -ENOMEM;

	return ethosn_driver_probe(ethosn->core[0],
				   &pdev->resource[0],
				   irq_numbers, irq_flags, num_irqs, true);
}

static struct pci_device_id ethosn_pci_device_id[] = {
	{ PCI_DEVICE(ETHOSN_PCI_VENDOR,
		     ETHOSN_PCI_DEVICE) },
	{ 0, }
};

MODULE_DEVICE_TABLE(pci, ethosn_pci_device_id);

static struct pci_driver ethosn_pci_driver = {
	.name     = ETHOSN_DRIVER_NAME,
	.id_table = ethosn_pci_device_id,
	.probe    = &ethosn_pci_probe
};

/*****************************************************************************
 * Module initialization and destruction
 *****************************************************************************/

static int ethosn_major_init(void)
{
	dev_t devt;
	int ret;

	ret = alloc_chrdev_region(&devt, 0, ETHOSN_MAX_DEVICES,
				  ETHOSN_DRIVER_NAME);
	if (ret)
		return ret;

	ethosn_major = MAJOR(devt);

	return 0;
}

static void ethosn_major_cleanup(void)
{
	unregister_chrdev_region(MKDEV(ethosn_major, 0), ETHOSN_MAX_DEVICES);
}

static int ethosn_class_init(void)
{
	int ret;

	/* This is the first time in here, set everything up properly */
	ret = ethosn_major_init();
	if (ret)
		return ret;

	ret = class_register(&ethosn_class);
	if (ret) {
		pr_err("class_register failed for device\n");
		goto cleanup_ethosn;
	}

	ret = pci_register_driver(&ethosn_pci_driver);
	if (ret != 0) {
		pr_err("Failed to register PCI driver.\n");
		goto unregister_class;
	}

	return 0;

unregister_class:
	class_unregister(&ethosn_class);

cleanup_ethosn:
	ethosn_major_cleanup();

	return ret;
}

static void ethosn_class_release(void)
{
	pci_unregister_driver(&ethosn_pci_driver);
	class_unregister(&ethosn_class);
	ethosn_major_cleanup();
}

static int __init ethosn_init(void)
{
	int ret = ethosn_class_init();

	if (ret)
		return ret;

	ret = ethosn_core_platform_driver_register();

	if (ret)
		return ret;

	return platform_driver_register(&ethosn_pdev_driver);
}

static void __exit ethosn_exit(void)
{
	platform_driver_unregister(&ethosn_pdev_driver);
	ethosn_core_platform_driver_unregister();
	ethosn_class_release();
}

module_init(ethosn_init)
module_exit(ethosn_exit)
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Arm Limited");
MODULE_DESCRIPTION("Arm Ethos-N Driver");
MODULE_VERSION(ETHOSN_DRIVER_VERSION);
