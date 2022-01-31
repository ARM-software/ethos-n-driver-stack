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

#include "ethosn_network.h"

#include "ethosn_backport.h"
#include "ethosn_buffer.h"
#include "ethosn_debug.h"
#include "ethosn_device.h"
#include "ethosn_dma.h"
#include "ethosn_firmware.h"
#include "uapi/ethosn.h"

#include <linux/anon_inodes.h>
#include <linux/atomic.h>
#include <linux/device.h>
#include <linux/file.h>
#include <linux/fs.h>
#include <linux/kref.h>
#include <linux/list.h>
#include <linux/module.h>
#include <linux/pm_runtime.h>
#include <linux/poll.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/version.h>
#include <linux/wait.h>

#include <stdbool.h>

#define ETHOSN_INFERENCE_ABORTED   -1

#define DEVICE_POLL_JIFFIES (1 * HZ / 1000)

#define MAX_PENDING ((int)-1)

struct ethosn_network {
	/* This is the ethosn device on which the memory for constant_dma_data,
	 * constant_cu_data, inference_data and intermediate_data was
	 * allocated. The memory is allocated/mapped on all the cores.
	 */
	struct ethosn_device      *ethosn;

	struct ethosn_dma_info    *constant_dma_data;
	struct ethosn_dma_info    *constant_cu_data;
	struct ethosn_dma_info    **inference_data;
	struct ethosn_dma_info    **intermediate_data;

	u32                       num_intermediates;
	struct ethosn_buffer_info *intermediates;

	u32                       num_inputs;
	struct ethosn_buffer_info *inputs;

	u32                       num_outputs;
	struct ethosn_buffer_info *outputs;

	/* file pointer used for ref-counting */
	struct file               *file;
};

static struct device *net_to_dev(const struct ethosn_network *const net)
{
	return net->ethosn->dev;
}

static struct device *ifr_to_dev(const struct ethosn_inference *const ifr)
{
	return net_to_dev(ifr->network);
}

static struct ethosn_buffer_array *get_inference_header(
	const struct ethosn_network *const network,
	uint32_t core_id)
{
	return network->inference_data[core_id]->cpu_addr;
}

static int set_binding(struct ethosn_network *network,
		       uint32_t core_id,
		       struct ethosn_buffer_info *buf_info,
		       ethosn_address_t container_start,
		       ethosn_address_t container_size,
		       bool check_in_container)
{
	ethosn_address_t buf_start = container_start + buf_info->offset;
	ethosn_address_t buf_end = buf_start + buf_info->size;
	ethosn_address_t container_end = container_start + container_size;
	struct ethosn_buffer_array *buffers =
		get_inference_header(network, core_id);

	if (buf_start > buf_end) {
		dev_err(net_to_dev(network),
			"Overflow in inference binding: %llu > %llu\n",
			buf_start, buf_end);

		return -EINVAL;
	}

	if (check_in_container && (buf_end > container_end)) {
		dev_err(net_to_dev(
				network),
			"Inference binding outside of container: { %u, %u } > { 0, %llu }\n",
			buf_info->offset, buf_info->offset + buf_info->size,
			container_size);

		return -EINVAL;
	}

	buffers->buffers[buf_info->id].address = buf_start;
	buffers->buffers[buf_info->id].size = buf_info->size;

	return 0;
}

static int update_bindings(struct ethosn_network *network,
			   uint32_t core_id,
			   u32 num_buffer_infos,
			   struct ethosn_buffer_info *buffer_infos,
			   ethosn_address_t container_start,
			   ethosn_address_t container_size,
			   bool check_duplicates,
			   bool check_in_container)
{
	u32 i;
	ethosn_address_t min_buf_start = container_size;
	ethosn_address_t max_buf_end = 0;
	struct ethosn_buffer_array *buffers =
		get_inference_header(network, core_id);

	for (i = 0; i < num_buffer_infos; ++i) {
		struct ethosn_buffer_info *const buf_info =
			&buffer_infos[i];
		ethosn_address_t buf_start = buf_info->offset;
		ethosn_address_t buf_end = buf_start + buf_info->size;
		int ret;

		if (buf_info->id >= buffers->num_buffers) {
			dev_err(net_to_dev(network),
				"Invalid inference binding id: %u >= %u\n",
				buf_info->id, buffers->num_buffers);

			return -EINVAL;
		}

		if (check_duplicates &&
		    (buffers->buffers[buf_info->id].size != 0)) {
			dev_err(net_to_dev(network),
				"Duplicate inference binding id: %u\n",
				buf_info->id);

			return -EINVAL;
		}

		ret = set_binding(network,
				  core_id,
				  buf_info,
				  container_start,
				  container_size,
				  check_in_container);
		if (ret)
			return ret;

		if (buf_start < min_buf_start)
			min_buf_start = buf_start;

		if (buf_end > max_buf_end)
			max_buf_end = buf_end;
	}

	if (check_in_container &&
	    ((min_buf_start > 0) || (max_buf_end < container_size)))
		/* Buffers have alignment requirements and this below
		 * is only an indication
		 */
		dev_dbg(net_to_dev(network),
			"Unused buffer data { %llu, %llu } <> { 0, %llu }\n",
			min_buf_start, max_buf_end, container_size);

	return 0;
}

static void get_network(struct ethosn_network *network)
{
	get_file(network->file);
}

static void put_network(struct ethosn_network *network)
{
	fput(network->file);
}

static void free_buffers(const u32 n,
			 struct ethosn_buffer **bufs)
{
	u32 i;

	if (IS_ERR_OR_NULL(bufs))
		return;

	for (i = 0; i < n; ++i)
		put_ethosn_buffer(bufs[i]);

	kfree(bufs);
}

static void inference_kref_release(struct kref *kref)
{
	struct ethosn_inference *const inference =
		container_of(kref, struct ethosn_inference, kref);

	struct ethosn_network *const network = inference->network;

	dev_dbg(ifr_to_dev(inference),
		"Released inference. handle=0x%pK\n", inference);

	free_buffers(network->num_inputs, inference->inputs);
	free_buffers(network->num_outputs, inference->outputs);

	put_network(network);

	kfree(inference);
}

static void get_inference(struct ethosn_inference *inference)
{
	kref_get(&inference->kref);
}

int ethosn_put_inference(struct ethosn_inference *inference)
{
	return kref_put(&inference->kref, &inference_kref_release);
}

static struct ethosn_buffer **read_buffer_fds(struct ethosn_network *network,
					      u32 n,
					      const int __user *fds,
					      struct ethosn_buffer_info *infos)
{
	struct ethosn_buffer **bufs;
	int error;
	u32 i;

	bufs = kcalloc(n, sizeof(*bufs), GFP_KERNEL);
	if (!bufs)
		return ERR_PTR(-ENOMEM);

	for (i = 0; i < n;) {
		u32 buf_size = infos[i].size;
		struct ethosn_buffer *buf;
		int fd;

		if (copy_from_user(&fd, fds + i, sizeof(fd))) {
			error = -EFAULT;
			goto err_free_bufs;
		}

		buf = ethosn_buffer_get(fd);
		if (IS_ERR(buf)) {
			error = PTR_ERR(buf);
			dev_err(net_to_dev(
					network),
				"ethosn_buffer_get returned an error: %d\n",
				error);
			goto err_free_bufs;
		}

		if (!buf) {
			error = -EFAULT;
			dev_err(net_to_dev(
					network),
				"ethosn_buffer_get returned an empty buffer\n"
				);
			goto err_free_bufs;
		}

		bufs[i] = buf;

		++i;

		if (buf->ethosn->dev != net_to_dev(network)) {
			dev_err(net_to_dev(
					network),
				"device buffer 0x%pK belongs to a different dev\n",
				buf);
			error = -EINVAL;
			goto err_free_bufs;
		}

		if (buf->dma_info->size < buf_size) {
			dev_err(net_to_dev(
					network),
				"Network size does not match buffer size. handle=0x%pK, buf_size=%zu, network_size=%u, fd=%d\n",
				buf, buf->dma_info->size, buf_size, fd);
			error = -EINVAL;
			goto err_free_bufs;
		}
	}

	return bufs;

err_free_bufs:
	free_buffers(i, bufs);

	return ERR_PTR(error);
}

/**
 * ethosn_schedule_inference() - Send an inference to Ethos-N
 *
 * If an inference isn't already running, send it to Ethos-N for execution.
 * Return:
 * * 0 - OK
 * * Negative error code
 */
int ethosn_schedule_inference(struct ethosn_inference *inference)
{
	struct ethosn_network *network = inference->network;
	struct ethosn_core *core = inference->core;
	struct ethosn_device *ethosn = core->parent;
	uint32_t core_id = core->core_id;
	struct device *core_dev = core->dev;
	u32 i;
	int ret;

	if (inference->status == ETHOSN_INFERENCE_RUNNING) {
		dev_err(core_dev, "Core %d got an inference while busy",
			core_id);
		ethosn->status_mask |=
			(1 << INFERENCE_SCHEDULED_ON_BUSY_CORE);
	}

	if (inference->status != ETHOSN_INFERENCE_SCHEDULED)
		return 0;

	inference->status = ETHOSN_INFERENCE_RUNNING;

	for (i = 0; i < network->num_inputs; ++i) {
		struct ethosn_dma_info *dma_info =
			inference->inputs[i]->dma_info;

		ret = update_bindings(network,
				      core_id,
				      1,
				      &network->inputs[i],
				      dma_info->iova_addr,
				      dma_info->size,
				      false,
				      true);
		if (ret)
			goto out_inference_error;
	}

	for (i = 0; i < network->num_outputs; ++i) {
		struct ethosn_dma_info *dma_info =
			inference->outputs[i]->dma_info;

		ret = update_bindings(network,
				      core_id,
				      1,
				      &network->outputs[i],
				      dma_info->iova_addr,
				      dma_info->size,
				      false,
				      true);
		if (ret)
			goto out_inference_error;
	}

	ret = update_bindings(network,
			      core_id,
			      network->num_intermediates,
			      network->intermediates,
			      network->intermediate_data[core_id] == NULL ? 0 :
			      network->intermediate_data[core_id]->iova_addr,
			      network->intermediate_data[core_id] == NULL ? 0 :
			      network->intermediate_data[core_id]->size,
			      false,
			      true);

	if (ret)
		goto out_inference_error;

	if (ethosn_mailbox_empty(core->mailbox_request->cpu_addr) &&
	    core->profiling.config.enable_profiling) {
		/* Send sync message */
		ret = ethosn_send_time_sync(core);
		if (ret)
			goto out_inference_error;
	}

	/* kick off execution */
	dev_dbg(core_dev, "Starting execution of inference");
	ethosn_dma_sync_for_device(core->allocator,
				   network->inference_data[core_id]);
	core->current_inference = inference;

	pm_runtime_get_sync(core->dev);

	/* send the inference to the core assigned to it */
	ret = ethosn_send_inference(core,
				    network->inference_data[core_id]->iova_addr,
				    (ptrdiff_t)inference);
	if (ret) {
		core->current_inference = NULL;
		pm_runtime_mark_last_busy(core->dev);
		pm_runtime_put(core->dev);

		goto out_inference_error;
	}

	get_inference(inference);
	ethosn->current_busy_cores |= (1 << core_id);
	dev_dbg(core_dev, "Scheduled inference 0x%pK on core_id = %u\n",
		inference,
		core_id);

	return 0;

out_inference_error:
	dev_err(core_dev,
		"Error scheduling inference 0x%pK: %d on core_id = %u\n",
		inference, ret, core_id);
	inference->status = ETHOSN_INFERENCE_ERROR;

	return ret;
}

/**
 * ethosn_schedule_queued_inference() - Schedule a queue inference.
 * @core:	Ethos-N core.
 *
 * Pop the inference queue until either the queue is empty or an inference has
 * been successfully scheduled.
 */
void ethosn_schedule_queued_inference(struct ethosn_core *core)
{
	struct ethosn_inference *inference = NULL;
	struct ethosn_device *ethosn = core->parent;
	int ret = 0;

	/* This will be invoked from the irq handlers of multiple npus.
	 * The inference queue needs to be protected against concurrent
	 * operation.
	 */
	ret = mutex_lock_interruptible(
		&ethosn->queue.inference_queue_mutex);
	if (ret)
		return;

	if (!list_empty(&ethosn->queue.inference_queue)) {
		inference = list_first_entry(&ethosn->queue.inference_queue,
					     typeof(*inference),
					     queue_node);
		if (inference != NULL)
			list_del(&inference->queue_node);
		else
			dev_dbg(ethosn->dev,
				"Inference is NULL\n");
	}

	mutex_unlock(&ethosn->queue.inference_queue_mutex);

	if (inference != NULL) {
		/* Schedule the inference on a particular core */
		inference->core = core;

		(void)ethosn_schedule_inference(inference);
	}
}

/**
 * inference_create() - Create and schedule an inference job
 * @network: Inference network
 * @ifr_req: Inference description
 * @inference_ptr: Output resulting inference struct
 *
 * Return: Valid pointer on success, else error pointer.
 */
static
struct ethosn_inference *inference_create(struct ethosn_network *network,
					  struct ethosn_inference_req *ifr_req)
{
	struct ethosn_inference *inference;
	int ret;

	if ((ifr_req->num_inputs != network->num_inputs) ||
	    (ifr_req->num_outputs != network->num_outputs)) {
		dev_err(network->ethosn->dev,
			"Input/output mismatch: %d != %d or %d != %d",
			ifr_req->num_inputs, network->num_inputs,
			ifr_req->num_outputs, network->num_outputs);

		return ERR_PTR(-EINVAL);
	}

	inference = kzalloc(sizeof(*inference), GFP_KERNEL);
	if (!inference)
		return ERR_PTR(-ENOMEM);

	get_network(network);

	inference->network = network;
	inference->status = ETHOSN_INFERENCE_SCHEDULED;
	init_waitqueue_head(&inference->poll_wqh);
	kref_init(&inference->kref);

	inference->inputs = read_buffer_fds(network,
					    ifr_req->num_inputs,
					    ifr_req->input_fds,
					    network->inputs);
	if (IS_ERR(inference->inputs)) {
		ret = PTR_ERR(inference->inputs);
		goto err_put_inference;
	}

	inference->outputs = read_buffer_fds(network,
					     ifr_req->num_outputs,
					     ifr_req->output_fds,
					     network->outputs);
	if (IS_ERR(inference->outputs)) {
		ret = PTR_ERR(inference->outputs);
		goto err_put_inference;
	}

	return inference;

err_put_inference:
	ethosn_put_inference(inference);

	return ERR_PTR(ret);
}

static int inference_release(struct inode *inode,
			     struct file *filep)
{
	struct ethosn_inference *inference = filep->private_data;

	/*
	 * Note we don't use mutex_lock_interruptible here as we need to make
	 * sure we release the network so we don't leak resources.
	 * This would prevent the kernel module from being unloaded
	 * when requested.
	 */

	/*
	 * Check inference status before locking the mutex since it might not
	 * be necessary.
	 */
	if (inference->status == ETHOSN_INFERENCE_SCHEDULED) {
		/*
		 * Use the same mutex that is used for adding
		 * inference to the list.
		 */
		struct ethosn_device *ethosn = inference->network->ethosn;

		mutex_lock(
			&ethosn->queue.inference_queue_mutex);

		/* Inference might be running or completed by now */
		if (inference->status == ETHOSN_INFERENCE_SCHEDULED)
			list_del(&inference->queue_node);

		mutex_unlock(
			&ethosn->queue.inference_queue_mutex);
	}

	/*
	 * Check inference status before locking the mutex since it might not
	 * be necessary or even possible i.e. core is assigned only for running
	 * inferences.
	 */
	if (inference->status == ETHOSN_INFERENCE_RUNNING) {
		struct ethosn_core *core = inference->core;

		mutex_lock(&core->mutex);

		/* Inference might be completed by now */
		if (inference->status == ETHOSN_INFERENCE_RUNNING) {
			dev_warn(core->dev,
				 "Reset Ethos-N due to error inference abort. handle=0x%pK\n",
				 inference);

			(void)ethosn_reset_and_start_ethosn(core);
			ethosn_network_poll(core, inference,
					    ETHOSN_INFERENCE_STATUS_ERROR);
		}

		mutex_unlock(&core->mutex);
	}

	wake_up_poll(&inference->poll_wqh, EPOLLHUP);

	ethosn_put_inference(inference);

	return 0;
}

static __poll_t inference_poll(struct file *file,
			       poll_table *wait)
{
	struct ethosn_inference *inference = file->private_data;
	__poll_t ret = 0;

	poll_wait(file, &inference->poll_wqh, wait);

	if (inference->status < ETHOSN_INFERENCE_SCHEDULED)
		ret = EPOLLERR;

	else if (inference->status > ETHOSN_INFERENCE_RUNNING)
		ret = EPOLLIN;

	return ret;
}

static ssize_t inference_read(struct file *file,
			      char __user *buf,
			      size_t count,
			      loff_t *ppos)
{
	struct ethosn_inference *inference = file->private_data;

	if (WARN_ON((inference->status < ETHOSN_INFERENCE_SCHEDULED) ||
		    (inference->status > ETHOSN_INFERENCE_ERROR)))
		return -EINVAL;

	if (count != sizeof(inference->status))
		return -EINVAL;

	return put_user(inference->status,
			(int32_t __user *)buf) ? -EFAULT :
	       sizeof(inference->status);
}

/**
 * ethosn_inference_register() - Create an inference job
 *
 * Return: File descriptor on success, else error code.
 */
static int ethosn_inference_register(struct ethosn_network *network,
				     struct ethosn_inference_req *req)
{
	static const struct file_operations inference_fops = {
		.owner   = THIS_MODULE,
		.release = &inference_release,
		.poll    = &inference_poll,
		.read    = &inference_read,
	};
	int i = 0;
	bool found = false;
	struct ethosn_device *ethosn = network->ethosn;
	struct ethosn_core *core = ethosn->core[0];
	struct ethosn_inference *inference;
	int ret_fd, ret;

	inference = inference_create(network, req);
	if (IS_ERR(inference))
		return PTR_ERR(inference);

	ret_fd = anon_inode_getfd("ethosn-inference",
				  &inference_fops,
				  inference,
				  O_RDONLY | O_CLOEXEC);

	if (ret_fd < 0) {
		ethosn_put_inference(inference);

		return ret_fd;
	}

	dev_dbg(ifr_to_dev(inference),
		"Registered inference. handle=0x%pK\n", inference);

	ret = mutex_lock_interruptible(
		&ethosn->queue.inference_queue_mutex);
	/* Return the file descriptor */
	if (ret) {
		/* Queue node hasn't been added to the list.
		 * Make sure that nothing is removed from the list on release
		 */
		inference->status = ETHOSN_INFERENCE_ERROR;
		goto end;
	}

	/* Queue and schedule inference. */
	list_add_tail(&inference->queue_node, &ethosn->queue.inference_queue);

	mutex_unlock(&ethosn->queue.inference_queue_mutex);

	for (i = 0; i < ethosn->num_cores && !found; ++i) {
		/* Check the status of the core
		 */
		core = ethosn->core[i];

		ret = mutex_lock_interruptible(&core->mutex);

		/* Return the file descriptor */
		if (ret)
			goto end;

		if (core->current_inference == NULL) {
			found = true;
			ethosn_schedule_queued_inference(core);
		}

		mutex_unlock(&core->mutex);
	}

	if (!found)
		dev_dbg(ethosn->dev,
			"Could not find any free core. Total cores = %d\n",
			ethosn->num_cores);

end:

	return ret_fd;
}

/**
 * network_ioctl() - Take network command from user space
 * @filep: File struct
 * @cmd: User command
 * * ETHOSN_IOCTL_SCHEDULE_INFERENCE
 *
 * Return:
 * * Inference file descriptor on success
 * * Negative error code on failure
 */
static long network_ioctl(struct file *filep,
			  unsigned int cmd,
			  unsigned long arg)
{
	struct ethosn_network *network = filep->private_data;
	const void __user *udata = (void __user *)arg;
	int ret;
	u64 time;

	time = ktime_get_ns();

	switch (cmd) {
	case ETHOSN_IOCTL_SCHEDULE_INFERENCE: {
		struct ethosn_inference_req infer_req;

		if (copy_from_user(&infer_req, udata, sizeof(infer_req))) {
			ret = -EFAULT;
			break;
		}

		ret = ethosn_inference_register(network, &infer_req);

		dev_dbg(net_to_dev(
				network), "SCHEDULE_INFERENCE: time %llu",
			time);

		break;
	}
	case ETHOSN_IOCTL_GET_INTERMEDIATE_BUFFER: {
		if (network->ethosn->num_cores > 1)
			dev_warn(net_to_dev(
					 network),
				 "Intermediate buffer for multi-core system: core 0 will be returned.");

		ethosn_dma_sync_for_cpu(network->ethosn->core[0]->allocator,
					network->intermediate_data[0]);

		ret = ethosn_get_dma_view_fd(network->ethosn,
					     network->intermediate_data[0]);
		break;
	}
	default: {
		ret = -EINVAL;
	}
	}

	return ret;
}

static int init_bindings(struct ethosn_network *network,
			 uint32_t core_id,
			 u32 num_binfos,
			 const struct ethosn_buffer_info __user *binfos_user,
			 ethosn_address_t container_start,
			 ethosn_address_t container_size,
			 bool check_in_container,
			 struct ethosn_buffer_info **binfos_save)
{
	struct ethosn_buffer_info *binfos;
	size_t binfos_size;
	int ret;

	binfos_size = num_binfos * sizeof(*binfos);
	binfos = kmalloc(binfos_size, GFP_KERNEL);

	if (!binfos)
		return -ENOMEM;

	if (copy_from_user(binfos, binfos_user, binfos_size)) {
		dev_err(net_to_dev(network), "Error reading binfos\n");
		ret = -EFAULT;
		goto out_free_binfos;
	}

	ret = update_bindings(network,
			      core_id,
			      num_binfos,
			      binfos,
			      container_start,
			      container_size,
			      true,
			      check_in_container);

	if (ret || !binfos_save)
		goto out_free_binfos;

	*binfos_save = binfos;

	return ret;

out_free_binfos:
	kfree(binfos);

	return ret;
}

static int init_inference_data(struct ethosn_network *network,
			       struct ethosn_core *core,
			       u32 num_bindings,
			       struct ethosn_network_req *net_req,
			       uint32_t core_id)
{
	u32 i;
	int ret;
	struct ethosn_buffer_array *buffers =
		get_inference_header(network, core_id);
	struct ethosn_device *ethosn = network->ethosn;

	buffers->num_buffers = num_bindings;

	for (i = 0; i < num_bindings; ++i)
		memset(&buffers->buffers[i], 0, sizeof(buffers->buffers[i]));

	ethosn_dma_sync_for_device(ethosn->allocator,
				   network->constant_dma_data);
	ret = init_bindings(network,
			    core_id,
			    net_req->dma_buffers.num,
			    net_req->dma_buffers.info,
			    network->constant_dma_data->iova_addr,
			    net_req->dma_data.size,
			    true,
			    NULL);
	if (ret)
		return ret;

	ethosn_dma_sync_for_device(ethosn->allocator,
				   network->constant_cu_data);
	ret = init_bindings(network,
			    core_id,
			    net_req->cu_buffers.num,
			    net_req->cu_buffers.info,
			    to_ethosn_addr(network->constant_cu_data->iova_addr,
					   &core->dma_map),
			    net_req->cu_data.size,
			    true,
			    NULL);
	if (ret)
		return ret;

	ret = init_bindings(network,
			    core_id,
			    net_req->intermediate_buffers.num,
			    net_req->intermediate_buffers.info,
			    0,
			    0,
			    false,
			    &network->intermediates);
	if (ret)
		return ret;

	network->num_intermediates = net_req->intermediate_buffers.num;

	ret = init_bindings(network,
			    core_id,
			    net_req->input_buffers.num,
			    net_req->input_buffers.info,
			    0,
			    0,
			    false,
			    &network->inputs);
	if (ret)
		return ret;

	network->num_inputs = net_req->input_buffers.num;

	for (i = 0; i < network->num_inputs; ++i) {
		if (network->inputs[i].offset != 0)
			dev_warn(net_to_dev(network),
				 "Ignored input offset %u\n",
				 network->inputs[i].offset);

		network->inputs[i].offset = 0;
	}

	ret = init_bindings(network,
			    core_id,
			    net_req->output_buffers.num,
			    net_req->output_buffers.info,
			    0,
			    0,
			    false,
			    &network->outputs);
	if (ret)
		return ret;

	network->num_outputs = net_req->output_buffers.num;

	for (i = 0; i < network->num_outputs; ++i) {
		if (network->outputs[i].offset != 0)
			dev_warn(net_to_dev(network),
				 "Ignored output offset %u\n",
				 network->outputs[i].offset);

		network->outputs[i].offset = 0;
	}

	for (i = 0; i < num_bindings; ++i)
		if (buffers->buffers[i].size == 0) {
			dev_err(net_to_dev(network),
				"Missing inference binding id\n");

			return -EINVAL;
		}

	return 0;
}

static int alloc_init_inference_data(struct ethosn_network *network,
				     struct ethosn_network_req *req)
{
	u32 num_bindings;
	size_t size;
	int ret = -ENOMEM;
	int i = 0;
	int num_cores = network->ethosn->num_cores;

	struct ethosn_core *core = network->ethosn->core[0];

	num_bindings = req->cu_buffers.num;
	num_bindings += req->dma_buffers.num;
	num_bindings += req->intermediate_buffers.num;
	num_bindings += req->input_buffers.num;
	num_bindings += req->output_buffers.num;

	size = sizeof(struct ethosn_buffer_array) + num_bindings *
	       sizeof(struct ethosn_buffer_desc);

	/*
	 * The inference data (which is ethosn_buffer_array) needs to be
	 * allocated per core. The reason being each core will have a
	 * unique entry for the "intermediate data" inside the
	 * ethosn_buffer_array.
	 */
	network->inference_data = kzalloc(
		(sizeof(*(network->inference_data)) * num_cores),
		GFP_KERNEL);
	if (!network->inference_data)
		return ret;

	/*
	 * Each core needs it own intermediate data. It reads/writes to this
	 * data during the execution of an inference.
	 */
	network->intermediate_data = kzalloc(
		(sizeof(*(network->intermediate_data)) * num_cores),
		GFP_KERNEL);
	if (!network->intermediate_data)
		return ret;

	for (i = 0; i < num_cores; i++) {
		core = network->ethosn->core[i];
		ret = -ENOMEM;

		network->inference_data[i] =
			ethosn_dma_alloc_and_map(core->allocator,
						 size, ETHOSN_PROT_READ,
						 ETHOSN_STREAM_COMMAND_STREAM,
						 GFP_KERNEL,
						 "network-inference-data");
		if (IS_ERR_OR_NULL(network->inference_data[i]))
			return ret;

		network->intermediate_data[i] =
			ethosn_dma_alloc_and_map(
				core->allocator,
				req->intermediate_data_size,
				ETHOSN_PROT_READ | ETHOSN_PROT_WRITE,
				ETHOSN_STREAM_DMA,
				GFP_KERNEL,
				"network-intermediate-data");

		if (IS_ERR_OR_NULL(network->intermediate_data[i]))
			return ret;

		ret = init_inference_data(network, core, num_bindings, req, i);

		if (ret)
			return ret;
	}

	return ret;
}

static void free_network(struct ethosn_network *network)
{
	int i = 0;

	struct ethosn_device *ethosn = network->ethosn;

	dev_dbg(net_to_dev(network),
		"Released network. handle=0x%pK\n", network);

	for (i = 0; i < ethosn->num_cores; i++) {
		struct ethosn_core *core = ethosn->core[i];

		/* Unmap virtual addresses from core */
		ethosn_dma_unmap(core->allocator,
				 network->constant_dma_data,
				 ETHOSN_STREAM_DMA);
		ethosn_dma_unmap(core->allocator,
				 network->constant_cu_data,
				 ETHOSN_STREAM_COMMAND_STREAM);

		/* Free allocated dma from core */
		if (network->intermediate_data)
			ethosn_dma_unmap_and_free(
				core->allocator,
				network->intermediate_data[i],
				ETHOSN_STREAM_DMA);

		if (network->inference_data)
			ethosn_dma_unmap_and_free(
				core->allocator,
				network->inference_data[i],
				ETHOSN_STREAM_COMMAND_STREAM);
	}

	/* Free allocated dma from top level device */
	ethosn_dma_free(ethosn->allocator, network->constant_dma_data);
	ethosn_dma_free(ethosn->allocator, network->constant_cu_data);

	kfree(network->intermediate_data);
	kfree(network->inference_data);
	kfree(network->intermediates);
	kfree(network->inputs);
	kfree(network->outputs);

	put_device(net_to_dev(network));

	kfree(network);
}

/**
 * create_network() - Create a new network
 * @ethosn:     Ethos-N device
 * @net_rq:     Network description
 *
 * Return: Network pointer on success, else error code.
 */
static struct ethosn_network *create_network(struct ethosn_device *ethosn,
					     struct ethosn_network_req *net_req)
{
	/* Note:- We register network on ethosn.
	 * For carveout :- We allocate constant data. inference data
	 *                 and intermediate data on top level device.
	 *                 All the cores can access the same buffer as
	 *                 the complete carveout memory is shared.
	 * For smmu :- The constant data is allocated on parent and
	 *             mapped on all cores. Inference data and
	 *             intermediate data are allocated and mapped
	 *             on all cores.
	 */
	struct ethosn_network *network;
	int ret = -ENOMEM;
	int i;

	network = kzalloc(sizeof(*network), GFP_KERNEL);
	if (!network)
		return ERR_PTR(-ENOMEM);

	network->ethosn = ethosn;

	/* Increment ref-count on device. Not sure why this is necessary,
	 * but it needs to be before any potential failures so that when we
	 * decrement the ref-count in free_network we can rely on it having been
	 * previously incremented.
	 */
	get_device(ethosn->dev);

	network->constant_dma_data = ethosn_dma_alloc(ethosn->allocator,
						      net_req->dma_data.size,
						      GFP_KERNEL,
						      "network-constant-dma-data");

	if (IS_ERR_OR_NULL(network->constant_dma_data))
		goto err_free_network;

	for (i = 0; i < ethosn->num_cores; ++i) {
		ret = ethosn_dma_map(ethosn->core[i]->allocator,
				     network->constant_dma_data,
				     ETHOSN_PROT_READ,
				     ETHOSN_STREAM_DMA);
		if (ret)
			goto err_free_network;
	}

	if (copy_from_user(network->constant_dma_data->cpu_addr,
			   net_req->dma_data.data,
			   net_req->dma_data.size)) {
		dev_err(ethosn->dev,
			"Error reading constant dma data\n");
		goto err_free_network;
	}

	network->constant_cu_data =
		ethosn_dma_alloc(ethosn->allocator,
				 net_req->cu_data.size,
				 GFP_KERNEL,
				 "network-constant-cu-data");
	if (IS_ERR_OR_NULL(network->constant_cu_data))
		goto err_free_network;

	for (i = 0; i < ethosn->num_cores; ++i) {
		ret = ethosn_dma_map(ethosn->core[i]->allocator,
				     network->constant_cu_data,
				     ETHOSN_PROT_READ,
				     ETHOSN_STREAM_COMMAND_STREAM);
		if (ret)
			goto err_free_network;
	}

	if (copy_from_user(network->constant_cu_data->cpu_addr,
			   net_req->cu_data.data,
			   net_req->cu_data.size)) {
		dev_err(ethosn->dev,
			"Error reading constant cu data\n");
		goto err_free_network;
	}

	ret = alloc_init_inference_data(network, net_req);
	if (ret)
		goto err_free_network;

	return network;

err_free_network:
	free_network(network);

	return ERR_PTR(ret);
}

static int network_release(struct inode *inode,
			   struct file *filep)
{
	struct ethosn_network *network = filep->private_data;
	struct ethosn_device *ethosn = network->ethosn;

	/* Note we don't use mutex_lock_interruptible here as we need to make
	 * sure we release the network so we don't leak resources.
	 * This would prevent the kernel module from being unloaded
	 * when requested.
	 */
	mutex_lock(&ethosn->mutex);

	free_network(network);

	mutex_unlock(&ethosn->mutex);

	return 0;
}

/**
 * ethosn_network_register() - Create a network
 * @ethosn:	Ethos-N device
 * @net_req:	Network description
 *
 * Return: FD on success, else error code
 */
int ethosn_network_register(struct ethosn_device *ethosn,
			    struct ethosn_network_req *net_req)
{
	static const struct file_operations network_fops = {
		.owner          = THIS_MODULE,
		.release        = &network_release,
		.unlocked_ioctl = &network_ioctl,
#ifdef CONFIG_COMPAT
		.compat_ioctl   = &network_ioctl,
#endif
	};

	struct ethosn_network *network;
	int fd;

	network = create_network(ethosn, net_req);
	if (IS_ERR(network))
		return PTR_ERR(network);

	fd = anon_inode_getfd("ethosn-network",
			      &network_fops,
			      network,
			      O_RDONLY | O_CLOEXEC);
	if (fd < 0) {
		free_network(network);

		return fd;
	}

	network->file = fget(fd);
	fput(network->file);

	dev_dbg(ethosn->dev,
		"Registered network. handle=0x%pK\n", network);

	return fd;
}

void ethosn_network_poll(struct ethosn_core *core,
			 struct ethosn_inference *inference,
			 int status)
{
	if (inference) {
		inference->status = status;

		wake_up_poll(&inference->poll_wqh, EPOLLIN);

		dev_dbg(core->dev,
			"END_INFERENCE: inference 0x%pK time %llu on core_id = %u",
			inference, ktime_get_ns(), core->core_id);

		ethosn_put_inference(inference);
		pm_runtime_mark_last_busy(core->dev);
		pm_runtime_put(core->dev);
	}

	/* Reset current running inference. */
	core->current_inference = NULL;

	/* Schedule next queued inference. */
	ethosn_schedule_queued_inference(core);
}
