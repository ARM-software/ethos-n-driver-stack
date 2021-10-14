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

#include "ethosn_buffer.h"

#include "ethosn_device.h"
#include "ethosn_log.h"

#include <linux/anon_inodes.h>
#include <linux/device.h>
#include <linux/file.h>
#include <linux/slab.h>
#include <linux/types.h>

#if (MB_RDONLY != O_RDONLY) ||	   \
	(MB_WRONLY != O_WRONLY) || \
	(MB_RDWR != O_RDWR)
#error "MB_ flags are not correctly defined"
#endif

static int ethosn_buffer_release(struct inode *inode,
				 struct file *file);
static int ethosn_buffer_mmap(struct file *file,
			      struct vm_area_struct *vma);
static loff_t ethosn_buffer_llseek(struct file *file,
				   loff_t offset,
				   int whence);
static long ethosn_buffer_ioctl(struct file *file,
				unsigned int cmd,
				unsigned long arg);
static int ethosn_dma_view_release(struct inode *const inode,
				   struct file *const file);

static const struct file_operations ethosn_buffer_fops = {
	.release        = &ethosn_buffer_release,
	.mmap           = &ethosn_buffer_mmap,
	.llseek         = &ethosn_buffer_llseek,
	.unlocked_ioctl = &ethosn_buffer_ioctl,
#ifdef CONFIG_COMPAT
	.compat_ioctl   = &ethosn_buffer_ioctl,
#endif
};

static bool is_ethosn_buffer_file(const struct file *const file)
{
	return file->f_op == &ethosn_buffer_fops;
}

/* Note we share the same mmap and llseek as a regular buffer, but
 * we need a different release implementation as we aren't freeing
 * any underlying storage.
 */
static const struct file_operations ethosn_dma_view_fops = {
	.release = &ethosn_dma_view_release,
	.mmap    = &ethosn_buffer_mmap,
	.llseek  = &ethosn_buffer_llseek,
};

static bool is_ethosn_dma_view_file(const struct file *const file)
{
	return file->f_op == &ethosn_dma_view_fops;
}

static int ethosn_dma_view_release(struct inode *const inode,
				   struct file *const file)
{
	struct ethosn_buffer *buf = file->private_data;
	struct ethosn_device *ethosn = buf->ethosn;

	if (WARN_ON(!is_ethosn_dma_view_file(file)))
		return -EBADF;

	dev_dbg(ethosn->dev, "Release DMA view. handle=0x%pK\n", buf);

	put_device(ethosn->dev);

	kfree(buf);

	return 0;
}

static void buffer_unmap_and_free_dma(struct ethosn_buffer *buf,
				      int num_cores)
{
	struct ethosn_device *ethosn = buf->ethosn;
	int i;

	/* Unmap iova per core through core allocator */
	for (i = 0; i < num_cores; ++i)
		ethosn_dma_unmap(
			ethosn->core[i]->allocator,
			buf->dma_info,
			ETHOSN_STREAM_DMA);

	ethosn_dma_free(ethosn->allocator, buf->dma_info);
}

static int ethosn_buffer_release(struct inode *const inode,
				 struct file *const file)
{
	struct ethosn_buffer *buf = file->private_data;
	struct ethosn_device *ethosn = buf->ethosn;
	int ret;

	if (WARN_ON(!is_ethosn_buffer_file(file)))
		return -EBADF;

	ret = mutex_lock_interruptible(&ethosn->mutex);
	if (ret)
		return ret;

	dev_dbg(buf->ethosn->dev, "Release buffer. handle=0x%pK\n", buf);

	buffer_unmap_and_free_dma(buf, ethosn->num_cores);

	put_device(buf->ethosn->dev);

	kfree(buf);

	mutex_unlock(&ethosn->mutex);

	return 0;
}

static int ethosn_buffer_mmap(struct file *const file,
			      struct vm_area_struct *const vma)
{
	struct ethosn_buffer *buf;
	struct ethosn_dma_allocator *allocator;

	if (WARN_ON(!is_ethosn_buffer_file(file) &&
		    !is_ethosn_dma_view_file(file)))
		return -EBADF;

	buf = file->private_data;
	allocator = buf->ethosn->allocator;

	return ethosn_dma_mmap(allocator, vma, buf->dma_info);
}

static loff_t ethosn_buffer_llseek(struct file *const file,
				   const loff_t offset,
				   const int whence)
{
	struct ethosn_buffer *buf;

	if (WARN_ON(!is_ethosn_buffer_file(file) &&
		    !is_ethosn_dma_view_file(file)))
		return -EBADF;

	buf = file->private_data;

	/* only support discovering the end of the buffer,
	 * but also allow SEEK_SET to maintain the idiomatic
	 * SEEK_END(0), SEEK_CUR(0) pattern
	 */

	if (offset != 0)
		return -EINVAL;

	if (whence == SEEK_END)
		return buf->dma_info->size;
	else if (whence == SEEK_SET)
		return 0;
	else
		return -EINVAL;
}

static long ethosn_buffer_ioctl(struct file *file,
				unsigned int cmd,
				unsigned long arg)
{
	struct ethosn_buffer *buf = file->private_data;
	struct ethosn_device *ethosn = buf->ethosn;
	int ret = 0;

	switch (cmd) {
	case ETHOSN_IOCTL_SYNC_FOR_CPU: {
		dev_dbg(ethosn->dev, "ETHOSN_IOCTL_SYNC_FOR_CPU\n");

		ethosn_dma_sync_for_cpu(ethosn->allocator, buf->dma_info);

		break;
	}
	case ETHOSN_IOCTL_SYNC_FOR_DEVICE: {
		dev_dbg(ethosn->dev, "ETHOSN_IOCTL_SYNC_FOR_DEVICE\n");

		ethosn_dma_sync_for_device(ethosn->allocator, buf->dma_info);
		break;
	}
	default: {
		ret = -EINVAL;
	}
	}

	return ret;
}

/**
 * ethosn_buffer_register() - Register a new Ethos-N buffer
 * @ethosn: [in]     pointer to Ethos-N device
 * @buf_req: [in]  buffer size and flags
 *
 * Return:
 * * File descriptor for the new Ethos-N buffer on success
 * * Negative error code on failure
 */
int ethosn_buffer_register(struct ethosn_device *ethosn,
			   struct ethosn_buffer_req *buf_req)
{
	struct ethosn_buffer *buf;
	struct ethosn_log_uapi_buffer_req log;
	int fd;
	int ret = -ENOMEM;
	int i;

	buf = kzalloc(sizeof(*buf), GFP_KERNEL);
	if (!buf)
		return -ENOMEM;

	dev_dbg(ethosn->dev, "Create buffer. handle=0x%pK, size=%u\n",
		buf, buf_req->size);

	/* Note:- We create buffers on ethosn.
	 * For carveout :- Both the cores can access the same buffer as
	 *                 the complete carveout memory is shared.
	 * For smmu :- The buffer is allocated on core[0]. And it should
	 *             be remapped to both the cores. The remapping part
	 *             is yet to be done.
	 */
	buf->ethosn = ethosn;

	buf->dma_info =
		ethosn_dma_alloc(ethosn->allocator, buf_req->size, GFP_KERNEL);
	if (IS_ERR_OR_NULL(buf->dma_info))
		goto err_kfree;

	/* Map iova per core through core allocator */
	for (i = 0; i < ethosn->num_cores; ++i) {
		int ret = ethosn_dma_map(
			ethosn->core[i]->allocator,
			buf->dma_info,
			ETHOSN_PROT_READ | ETHOSN_PROT_WRITE,
			ETHOSN_STREAM_DMA);

		if (ret < 0)
			goto err_dma_free;
	}

	ret = anon_inode_getfd("ethosn-buffer",
			       &ethosn_buffer_fops,
			       buf,
			       (buf_req->flags & O_ACCMODE) | O_CLOEXEC);
	if (ret < 0)
		goto err_dma_free;

	fd = ret;

	buf->file = fget(fd);
	buf->file->f_mode |= FMODE_LSEEK;

	fput(buf->file);

	get_device(ethosn->dev);

	if (buf_req->flags & MB_ZERO) {
		memset(buf->dma_info->cpu_addr, 0, buf->dma_info->size);
		ethosn_dma_sync_for_device(ethosn->allocator, buf->dma_info);
		dev_dbg(ethosn->dev, "Zeroed device buffer 0x%pK\n", buf);
	}

	log.request = *buf_req;
	log.handle = (ptrdiff_t)buf;
	log.fd = fd;

	/* FIXME :- This needs to be invoked with ethosn
	 */
	ethosn_log_uapi(ethosn->core[0], ETHOSN_IOCTL_CREATE_BUFFER, &log,
			sizeof(log));

	return fd;

err_dma_free:
	buffer_unmap_and_free_dma(buf, i);
err_kfree:
	kfree(buf);

	return ret;
}

/**
 * ethosn_buffer_get() - Returns the ethosn_buffer structure related to an fd
 * @fd: [in]    fd associated with the ethosn_buffer to be returned
 *
 * Return:
 * * On success, returns the ethosn_buffer structure associated with an fd;
 *   uses file's refcounting done by fget to increase refcount.
 * * ERR_PTR otherwise.
 */
struct ethosn_buffer *ethosn_buffer_get(int fd)
{
	struct file *file;

	file = fget(fd);

	if (!file)
		return ERR_PTR(-EBADF);

	if (!is_ethosn_buffer_file(file)) {
		fput(file);

		return ERR_PTR(-EINVAL);
	}

	return file->private_data;
}

/**
 * put_ethosn_buffer() - Decreases refcount of the buffer
 * @buf: [in]    Ethos-N buffer to reduce refcount of
 *
 * Uses file's refcounting done implicitly by fput()
 */
void put_ethosn_buffer(struct ethosn_buffer *buf)
{
	if (WARN_ON(!buf))
		return;

	if (WARN_ON(!buf->file))
		return;

	if (WARN_ON(!is_ethosn_buffer_file(buf->file)))
		return;

	fput(buf->file);
}

/**
 * ethosn_get_dma_view_fops() - Get dma view file operations.
 *
 * Return: File operations reference.
 */
const struct file_operations *ethosn_get_dma_view_fops(void)
{
	return &ethosn_dma_view_fops;
}
