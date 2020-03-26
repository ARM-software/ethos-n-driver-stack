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

#include "ethosn_network.h"

#include "ethosn_buffer.h"
#include "ethosn_device.h"
#include "ethosn_dma.h"
#include "ethosn_firmware.h"
#include "ethosn_log.h"
#include "uapi/ethosn.h"

#include <linux/anon_inodes.h>
#include <linux/atomic.h>
#include <linux/device.h>
#include <linux/file.h>
#include <linux/fs.h>
#include <linux/kref.h>
#include <linux/list.h>
#include <linux/module.h>
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
	struct ethosn_device      *ethosn;

	struct ethosn_dma_info    *constant_dma_data;
	struct ethosn_dma_info    *constant_cu_data;
	struct ethosn_dma_info    *inference_data;
	struct ethosn_dma_info    *intermediate_data;

	u32                       num_intermediates;
	struct ethosn_buffer_info *intermediates;

	u32                       num_inputs;
	struct ethosn_buffer_info *inputs;

	u32                       num_outputs;
	struct ethosn_buffer_info *outputs;

	/* file pointer used for ref-counting */
	struct file               *file;
};

struct ethosn_inference {
	struct ethosn_network *network;

	struct list_head      queue_node;

	struct ethosn_buffer  **inputs;
	struct ethosn_buffer  **outputs;

	u32                   status;

	wait_queue_head_t     poll_wqh;

	/* Reference counting */
	struct kref           kref;
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
	const struct ethosn_network *const network)
{
	return network->inference_data->cpu_addr;
}

static int set_binding(struct ethosn_network *network,
		       struct ethosn_buffer_info *buf_info,
		       ethosn_address_t container_start,
		       ethosn_address_t container_size,
		       bool check_in_container)
{
	ethosn_address_t buf_start = container_start + buf_info->offset;
	ethosn_address_t buf_end = buf_start + buf_info->size;
	ethosn_address_t container_end = container_start + container_size;
	struct ethosn_buffer_array *buffers = get_inference_header(network);

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
	struct ethosn_buffer_array *buffers = get_inference_header(network);

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
		dev_warn(net_to_dev(network),
			 "Unused buffer data! { %llu, %llu } <> { 0, %llu }\n",
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

	put_network(network);

	free_buffers(network->num_inputs, inference->inputs);
	free_buffers(network->num_outputs, inference->outputs);

	kfree(inference);
}

static void get_inference(struct ethosn_inference *inference)
{
	kref_get(&inference->kref);
}

static int put_inference(struct ethosn_inference *inference)
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
			goto err_free_bufs;
		}

		bufs[i] = buf;

		++i;

		if (buf->ethosn->dev != net_to_dev(network)) {
			dev_err(net_to_dev(
					network),
				"ethosn buffer 0x%pK belongs to a different dev\n",
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
 * schedule_inference() - Send an inference to Ethos-N
 *
 * If an inference isn't already running, send it to Ethos-N for execution.
 * Return:
 * * 0 - OK
 * * Negative error code
 */
static int schedule_inference(struct ethosn_inference *inference)
{
	struct ethosn_network *network = inference->network;
	struct ethosn_device *ethosn = network->ethosn;
	struct device *dev = ethosn->dev;
	u32 i;
	int ret;

	if (inference->status != ETHOSN_INFERENCE_SCHEDULED)
		return 0;

	inference->status = ETHOSN_INFERENCE_RUNNING;

	for (i = 0; i < network->num_inputs; ++i) {
		struct ethosn_dma_info *dma_info =
			inference->inputs[i]->dma_info;
		ethosn_dma_sync_for_device(ethosn, dma_info);
		ret = update_bindings(network,
				      1,
				      &network->inputs[i],
				      dma_info->iova_addr,
				      dma_info->size,
				      false,
				      true);
		if (WARN_ON(ret))
			goto out_inference_error;
	}

	for (i = 0; i < network->num_outputs; ++i) {
		struct ethosn_dma_info *dma_info =
			inference->outputs[i]->dma_info;

		ethosn_dma_sync_for_device(ethosn, dma_info);
		ret = update_bindings(network,
				      1,
				      &network->outputs[i],
				      dma_info->iova_addr,
				      dma_info->size,
				      false,
				      true);
		if (WARN_ON(ret))
			goto out_inference_error;
	}

	if (ethosn_mailbox_empty(ethosn->mailbox_request->cpu_addr) &&
	    ethosn->profiling.config.enable_profiling) {
		/* Send sync message */
		ret = ethosn_send_time_sync(ethosn);
		if (ret)
			return ret;
	}

	/* kick off execution */
	dev_dbg(dev, "Starting execution of inference");
	ethosn_dma_sync_for_device(ethosn, network->inference_data);
	ethosn->current_inference = inference;
	ret = ethosn_send_inference(network->ethosn,
				    network->inference_data->iova_addr,
				    (ptrdiff_t)inference);
	if (ret) {
		ethosn->current_inference = NULL;

		return ret;
	}

	get_inference(inference);
	dev_dbg(dev, "Scheduled inference 0x%pK\n", inference);

	return 0;

out_inference_error:
	dev_err(dev, "Error scheduling inference 0x%pK: %d\n", inference, ret);
	inference->status = ETHOSN_INFERENCE_ERROR;

	return ret;
}

/**
 * schedule_queued_inference() - Schedule a queue inference.
 * @ethosn:	Ethos-N device.
 *
 * Pop the inference queue until either the queue is empty or an inference has
 * been successfully scheduled.
 */
static void schedule_queued_inference(struct ethosn_device *ethosn)
{
	struct ethosn_inference *inference;

	while (!list_empty(&ethosn->inference_queue)) {
		inference = list_first_entry(&ethosn->inference_queue,
					     typeof(*inference), queue_node);

		/* Return if an inference is currently running. */
		if (ethosn->current_inference)
			return;

		list_del(&inference->queue_node);
		(void)schedule_inference(inference);
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
	    (ifr_req->num_outputs != network->num_outputs))
		return ERR_PTR(-EINVAL);

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
	put_inference(inference);

	return ERR_PTR(ret);
}

static int inference_release(struct inode *inode,
			     struct file *filep)
{
	struct ethosn_inference *inference = filep->private_data;
	struct ethosn_network *network = inference->network;
	struct ethosn_device *ethosn = network->ethosn;

	/* Note we don't use mutex_lock_interruptible here as we need to make
	 * sure we release the network so we don't leak resources.
	 * This would prevent the kernel module from being unloaded
	 * when requested.
	 */
	mutex_lock(&ethosn->mutex);

	if (inference->status == ETHOSN_INFERENCE_SCHEDULED)
		list_del(&inference->queue_node);

	if (inference->status == ETHOSN_INFERENCE_RUNNING) {
		dev_warn(ethosn->dev,
			 "Reset Ethos-N due to error inference abort. handle=0x%pK\n",
			 inference);

		(void)ethosn_reset_and_start_ethosn(ethosn);
		ethosn_network_poll(ethosn, inference,
				    ETHOSN_INFERENCE_STATUS_ERROR);
	}

	wake_up_poll(&inference->poll_wqh, POLLHUP);

	put_inference(inference);

	mutex_unlock(&ethosn->mutex);

	return 0;
}

static unsigned int inference_poll(struct file *file,
				   poll_table *wait)
{
	struct ethosn_inference *inference = file->private_data;

	poll_wait(file, &inference->poll_wqh, wait);

	if (inference->status < ETHOSN_INFERENCE_SCHEDULED)
		return POLLERR;

	if (inference->status > ETHOSN_INFERENCE_RUNNING)
		return POLLIN;

	return 0;
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
	struct ethosn_device *ethosn = network->ethosn;
	struct ethosn_inference *inference;
	struct ethosn_log_uapi_inference_req log;
	int ret;

	inference = inference_create(network, req);
	if (IS_ERR(inference))
		return PTR_ERR(inference);

	ret = anon_inode_getfd("ethosn-inference",
			       &inference_fops,
			       inference,
			       O_RDONLY | O_CLOEXEC);

	if (ret < 0) {
		put_inference(inference);

		return ret;
	}

	dev_dbg(ifr_to_dev(inference),
		"Registered inference. handle=0x%pK\n", inference);

	log.request = *req;
	log.handle = (ptrdiff_t)inference;
	log.network_handle = (ptrdiff_t)network;
	log.fd = ret;
	ethosn_log_uapi(ethosn, ETHOSN_IOCTL_SCHEDULE_INFERENCE, &log,
			sizeof(log));

	/* Queue and schedule inference. */
	list_add_tail(&inference->queue_node, &ethosn->inference_queue);
	schedule_queued_inference(ethosn);

	return ret;
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
	struct ethosn_device *ethosn = network->ethosn;
	const void __user *udata = (void __user *)arg;
	int ret;

	ret = mutex_lock_interruptible(&ethosn->mutex);
	if (ret)
		return ret;

	switch (cmd) {
	case ETHOSN_IOCTL_SCHEDULE_INFERENCE: {
		struct ethosn_inference_req infer_req;

		if (copy_from_user(&infer_req, udata, sizeof(infer_req))) {
			ret = -EFAULT;
			break;
		}

		ret = ethosn_inference_register(network, &infer_req);
		break;
	}
	default: {
		ret = -EINVAL;
	}
	}

	mutex_unlock(&ethosn->mutex);

	return ret;
}

static int init_bindings(struct ethosn_network *network,
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

		return -EFAULT;
	}

	ret = update_bindings(network,
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
			       u32 num_bindings,
			       struct ethosn_network_req *net_req)
{
	u32 i;
	int ret;
	struct ethosn_buffer_array *buffers = get_inference_header(network);

	buffers->num_buffers = num_bindings;

	for (i = 0; i < num_bindings; ++i)
		memset(&buffers->buffers[i], 0, sizeof(buffers->buffers[i]));

	ethosn_dma_sync_for_device(network->ethosn, network->constant_dma_data);
	ret = init_bindings(network,
			    net_req->dma_buffers.num,
			    net_req->dma_buffers.info,
			    network->constant_dma_data->iova_addr,
			    net_req->dma_data.size,
			    true,
			    NULL);
	if (ret)
		return ret;

	ethosn_dma_sync_for_device(network->ethosn, network->constant_cu_data);
	ret = init_bindings(network,
			    net_req->cu_buffers.num,
			    net_req->cu_buffers.info,
			    to_ethosn_addr(network->constant_cu_data->iova_addr,
					   &network->ethosn->dma_map),
			    net_req->cu_data.size,
			    true,
			    NULL);
	if (ret)
		return ret;

	ethosn_dma_sync_for_device(network->ethosn, network->intermediate_data);
	ret = init_bindings(network,
			    net_req->intermediate_buffers.num,
			    net_req->intermediate_buffers.info,
			    network->intermediate_data == NULL ? 0 :
			    network->intermediate_data->iova_addr,
			    network->intermediate_data == NULL ? 0 :
			    network->intermediate_data->size,
			    true,
			    &network->intermediates);
	if (ret)
		return ret;

	network->num_intermediates = net_req->intermediate_buffers.num;

	ret = init_bindings(network,
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

	num_bindings = req->cu_buffers.num;
	num_bindings += req->dma_buffers.num;
	num_bindings += req->intermediate_buffers.num;
	num_bindings += req->input_buffers.num;
	num_bindings += req->output_buffers.num;

	size = sizeof(struct ethosn_buffer_array) + num_bindings *
	       sizeof(struct ethosn_buffer_desc);

	network->inference_data = ethosn_dma_alloc(network->ethosn,
						   size, ETHOSN_PROT_READ,
						   ETHOSN_STREAM_COMMAND_STREAM,
						   GFP_KERNEL);
	if (IS_ERR_OR_NULL(network->inference_data))
		return ret;

	network->intermediate_data =
		ethosn_dma_alloc(network->ethosn,
				 req->intermediate_data_size,
				 ETHOSN_PROT_READ | ETHOSN_PROT_WRITE,
				 ETHOSN_STREAM_DMA,
				 GFP_KERNEL);
	if (IS_ERR(network->intermediate_data))
		return ret;

	return init_inference_data(network, num_bindings, req);
}

static void free_network(struct ethosn_network *network)
{
	dev_dbg(net_to_dev(network),
		"Released network. handle=0x%pK\n", network);

	ethosn_dma_free(network->ethosn, ETHOSN_STREAM_COMMAND_STREAM,
			network->inference_data);
	ethosn_dma_free(network->ethosn, ETHOSN_STREAM_COMMAND_STREAM,
			network->constant_cu_data);
	ethosn_dma_free(network->ethosn, ETHOSN_STREAM_DMA,
			network->constant_dma_data);
	ethosn_dma_free(network->ethosn, ETHOSN_STREAM_DMA,
			network->intermediate_data);

	kfree(network->intermediates);
	kfree(network->inputs);
	kfree(network->outputs);

	put_device(net_to_dev(network));

	kfree(network);
}

/**
 * create_network() - Create a new network
 * @ethosn:        Ethos-N device
 * @net_rq:     Network description
 *
 * Return: Network pointer on success, else error code.
 */
static struct ethosn_network *create_network(struct ethosn_device *ethosn,
					     struct ethosn_network_req *net_req)
{
	struct ethosn_network *network;
	int ret = -ENOMEM;

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

	network->constant_dma_data = ethosn_dma_alloc(network->ethosn,
						      net_req->dma_data.size,
						      ETHOSN_PROT_READ,
						      ETHOSN_STREAM_DMA,
						      GFP_KERNEL);
	if (IS_ERR_OR_NULL(network->constant_dma_data))
		goto err_free_network;

	if (copy_from_user(network->constant_dma_data->cpu_addr,
			   net_req->dma_data.data,
			   net_req->dma_data.size)) {
		dev_err(ethosn->dev,
			"Error reading constant dma data\n");
		goto err_free_network;
	}

	network->constant_cu_data =
		ethosn_dma_alloc(network->ethosn,
				 net_req->cu_data.size,
				 ETHOSN_PROT_READ,
				 ETHOSN_STREAM_COMMAND_STREAM,
				 GFP_KERNEL);
	if (IS_ERR_OR_NULL(network->constant_cu_data))
		goto err_free_network;

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
	struct ethosn_log_uapi_network_req log;
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

	log.request = *net_req;
	log.handle = (ptrdiff_t)network;
	log.fd = fd;
	ethosn_log_uapi(ethosn, ETHOSN_IOCTL_REGISTER_NETWORK, &log,
			sizeof(log));

	return fd;
}

void ethosn_network_poll(struct ethosn_device *ethosn,
			 struct ethosn_inference *inference,
			 int status)
{
	if (inference) {
		int i;

		inference->status = status;

		for (i = 0; i < inference->network->num_outputs; ++i)
			ethosn_dma_sync_for_cpu(
				ethosn,
				inference->outputs[i]->dma_info);

		wake_up_poll(&inference->poll_wqh, POLLIN);
		put_inference(inference);
	}

	/* Reset current running inference. */
	ethosn->current_inference = NULL;

	/* Schedule next queued inference. */
	schedule_queued_inference(ethosn);
}
