/*
 *
 * (C) COPYRIGHT 2018-2019 Arm Limited. All rights reserved.
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

#include "ethosn_log.h"

#include "uapi/ethosn.h"

#include <linux/debugfs.h>
#include <linux/fs.h>
#include <linux/module.h>
#include <linux/poll.h>
#include <linux/uio.h>

static int write_vec(struct ethosn_core *core,
		     struct kvec *vec,
		     int count)
{
	size_t length;
	size_t pos;
	int i;
	int ret;

	/* Calculate the total length of the output. */
	for (i = 0, length = 0; i < count; ++i)
		length += vec[i].iov_len;

	/* Round up to next 32-bit boundary. */
	length = roundup(length, 4);

	if (length > core->ram_log.size)
		return -EINVAL;

	ret = mutex_lock_interruptible(&core->ram_log.mutex);
	if (ret)
		return ret;

	pos = core->ram_log.wpos & (core->ram_log.size - 1);

	/* Loop over scatter input. */
	for (i = 0; i < count; ++i) {
		const char *buf = vec[i].iov_base;
		size_t len = vec[i].iov_len;

		/* Copy log message to output buffer. */
		while (len > 0) {
			size_t n = min(len, core->ram_log.size - pos);

			memcpy(&core->ram_log.data[pos], buf, n);

			len -= n;
			buf += n;
			pos = (pos + n) & (core->ram_log.size - 1);
		}
	}

	/* Update write_pos. Length has already been 4 byte aligned */
	core->ram_log.wpos += length;

	mutex_unlock(&core->ram_log.mutex);

	wake_up_interruptible(&core->ram_log.wq);

	return 0;
}

static ssize_t read_buf(struct ethosn_core *core,
			char __user *buf,
			size_t count,
			loff_t *position)
{
	size_t wpos = core->ram_log.wpos;
	ssize_t n = 0;

	/* Make sure position is not beyond end of file. */
	if (*position > wpos)
		return -EINVAL;

	/* If position is more than BUFFER_SIZE bytes behind, then fast forward
	 * to current position minus BUFFER_SIZE.
	 */
	if ((wpos - *position) > core->ram_log.size)
		*position = wpos - core->ram_log.size;

	/* Copy data to user space. */
	while ((n < count) && (*position < wpos)) {
		size_t offset;
		size_t length;

		/* Offset in circular buffer. */
		offset = *position & (core->ram_log.size - 1);

		/* Available number of bytes. */
		length = min((size_t)(wpos - *position), count - n);

		/* Make sure length does not go beyond end of circular buffer.
		 */
		length = min(length, core->ram_log.size - offset);

		/* Copy data from kernel- to user space. */
		length -= copy_to_user(&buf[n], &core->ram_log.data[offset],
				       length);

		/* No bytes were copied. Return error. */
		if (length == 0)
			return -EINVAL;

		*position += length;
		n += length;
	}

	return n;
}

static ssize_t fops_read(struct file *file,
			 char __user *buf,
			 size_t count,
			 loff_t *position)
{
	struct ethosn_core *core = file->private_data;
	ssize_t n;
	int ret;

	if (*position == core->ram_log.wpos && file->f_flags & O_NONBLOCK)
		return -EAGAIN;

	ret = mutex_lock_interruptible(&core->ram_log.mutex);
	if (ret)
		return ret;

	n = read_buf(core, buf, count, position);

	mutex_unlock(&core->ram_log.mutex);

	return n;
}

static unsigned int fops_poll(struct file *file,
			      poll_table *wait)
{
	struct ethosn_core *core = file->private_data;
	unsigned int mask = 0;

	poll_wait(file, &core->ram_log.wq, wait);

	if (file->f_pos < core->ram_log.wpos)
		mask |= POLLIN | POLLRDNORM;
	else if (file->f_pos > core->ram_log.wpos)
		mask |= POLLERR;

	return mask;
}

static long fops_ioctl(struct file *file,
		       unsigned int cmd,
		       unsigned long arg)
{
	struct ethosn_core *core = file->private_data;

	switch (cmd) {
	case ETHOSN_IOCTL_LOG_CLEAR:
		core->ram_log.rpos = core->ram_log.wpos;
		file->f_pos = core->ram_log.rpos;
		break;
	default:

		return -EINVAL;
	}

	return 0;
}

static int fops_open(struct inode *inode,
		     struct file *file)
{
	struct ethosn_core *core = inode->i_private;

	file->private_data = inode->i_private;
	file->f_pos = core->ram_log.rpos;

	return 0;
}

int ethosn_log_init(struct ethosn_core *core)
{
	static const struct file_operations fops = {
		.owner          = THIS_MODULE,
		.open           = &fops_open,
		.poll           = &fops_poll,
		.read           = &fops_read,
		.unlocked_ioctl = &fops_ioctl
	};

	mutex_init(&core->ram_log.mutex);
	init_waitqueue_head(&core->ram_log.wq);

	core->ram_log.size = core->queue_size;
	core->ram_log.data = devm_kzalloc(core->parent->dev, core->ram_log.size,
					  GFP_KERNEL);
	if (!core->ram_log.data)
		return -ENOMEM;

	/* Create debugfs file handle */
	if (!IS_ERR_OR_NULL(core->debug_dir)) {
		core->ram_log.dentry = debugfs_create_file(
			"log", 0400, core->debug_dir, core, &fops);
		if (IS_ERR(core->ram_log.dentry))
			dev_warn(core->dev,
				 "Failed to create log debugfs file.\n");
	}

	return 0;
}

void ethosn_log_deinit(struct ethosn_core *core)
{
	debugfs_remove(core->ram_log.dentry);
	core->ram_log.dentry = NULL;
	if (core->ram_log.data) {
		devm_kfree(core->parent->dev, core->ram_log.data);
		core->ram_log.data = NULL;
	}
}

int ethosn_log_text(struct ethosn_core *core,
		    const char *msg)
{
	struct ethosn_log_header header;
	struct timespec64 timespec;
	struct kvec vec[2];

	if (IS_ERR_OR_NULL(core->ram_log.dentry))
		return 0;

	ktime_get_ts64(&timespec);

	header.magic = ETHOSN_LOG_MAGIC;
	header.type = ETHOSN_LOG_TYPE_TEXT;
	header.length = strlen(msg);
	header.timestamp.sec = timespec.tv_sec;
	header.timestamp.nsec = timespec.tv_nsec;

	vec[0].iov_base = &header;
	vec[0].iov_len = sizeof(header);

	vec[1].iov_base = (void *)msg;
	vec[1].iov_len = header.length;

	return write_vec(core, vec, 2);
}

int ethosn_log_uapi(struct ethosn_core *core,
		    uint32_t ioctl,
		    void *data,
		    size_t length)
{
	struct ethosn_log_header header;
	struct ethosn_log_uapi_header uapi;
	struct timespec64 timespec;
	struct kvec vec[3];

	if (IS_ERR_OR_NULL(core->ram_log.dentry))
		return 0;

	ktime_get_ts64(&timespec);

	header.magic = ETHOSN_LOG_MAGIC;
	header.type = ETHOSN_LOG_TYPE_UAPI;
	header.timestamp.sec = timespec.tv_sec;
	header.timestamp.nsec = timespec.tv_nsec;

	uapi.ioctl = ioctl;

	vec[0].iov_base = &header;
	vec[0].iov_len = sizeof(header);

	vec[1].iov_base = &uapi;
	vec[1].iov_len = sizeof(uapi);

	vec[2].iov_base = data;
	vec[2].iov_len = length;

	header.length = vec[1].iov_len + vec[2].iov_len;

	return write_vec(core, vec, 3);
}

int ethosn_log_firmware(struct ethosn_core *core,
			enum ethosn_log_firmware_direction direction,
			struct ethosn_message_header *hdr,
			void *data)
{
	struct ethosn_log_header header;
	struct ethosn_log_firmware_header firmware;
	struct timespec64 timespec;
	struct kvec vec[4];

	if (IS_ERR_OR_NULL(core->ram_log.dentry))
		return 0;

	ktime_get_ts64(&timespec);

	header.magic = ETHOSN_LOG_MAGIC;
	header.type = ETHOSN_LOG_TYPE_FIRMWARE;
	header.timestamp.sec = timespec.tv_sec;
	header.timestamp.nsec = timespec.tv_nsec;

	firmware.inference = (ptrdiff_t)core->current_inference;
	firmware.direction = direction;

	vec[0].iov_base = &header;
	vec[0].iov_len = sizeof(header);

	vec[1].iov_base = &firmware;
	vec[1].iov_len = sizeof(firmware);

	vec[2].iov_base = hdr;
	vec[2].iov_len = sizeof(*hdr);

	vec[3].iov_base = data;
	vec[3].iov_len = hdr->length;

	header.length = vec[1].iov_len + vec[2].iov_len + vec[3].iov_len;

	return write_vec(core, vec, 4);
}
