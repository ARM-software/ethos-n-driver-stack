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

#include "ethosn_dma.h"

#include "ethosn_device.h"
#include "ethosn_dma_carveout.h"
#include "ethosn_dma_iommu.h"

#include <linux/iommu.h>

static const struct ethosn_dma_allocator_ops *get_ops(
	struct ethosn_dma_allocator *allocator)
{
	if (allocator)
		return allocator->ops;
	else
		return NULL;
}

struct ethosn_dma_allocator *ethosn_dma_allocator_create(struct device *dev)
{
	struct ethosn_dma_allocator *allocator;

	if (ethosn_smmu_available(dev))
		allocator = ethosn_dma_iommu_allocator_create(dev);
	else
		allocator = ethosn_dma_carveout_allocator_create(dev);

	return allocator;
}

void ethosn_dma_allocator_destroy(struct ethosn_dma_allocator **allocator)
{
	const struct ethosn_dma_allocator_ops *ops = allocator ?
						     get_ops(*allocator) : NULL;

	if (!ops)
		return;

	ops->destroy(*allocator);
	*allocator = NULL;
}

struct ethosn_dma_info *ethosn_dma_alloc(struct ethosn_dma_allocator *allocator,
					 const size_t size,
					 gfp_t gfp,
					 const char *debug_tag)
{
	const struct ethosn_dma_allocator_ops *ops = get_ops(allocator);
	struct ethosn_dma_info *dma_info = NULL;

	if (!ops)
		goto exit;

	dma_info = ops->alloc(allocator, size, gfp);

	if (IS_ERR_OR_NULL(dma_info)) {
		dev_err(allocator->dev,
			"failed to dma_alloc %zu bytes for %s\n",
			size, debug_tag == NULL ? "(unknown)" : debug_tag);
		goto exit;
	}

	dev_dbg(allocator->dev,
		"DMA alloc for %s. handle=0x%pK, cpu_addr=0x%pK, size=%zu\n",
		debug_tag == NULL ? "(unknown)" : debug_tag,
		dma_info, dma_info->cpu_addr, size);

	/* Zero the memory. This ensures the previous contents of the
	 * memory doesn't affect us (if the same physical memory
	 * is re-used). This means we get deterministic results in
	 * cases where parts of an intermediate buffer are read
	 * before being written.
	 */
	memset(dma_info->cpu_addr, 0, size);
	ops->sync_for_device(allocator, dma_info);

exit:

	return dma_info;
}

int ethosn_dma_map(struct ethosn_dma_allocator *allocator,
		   struct ethosn_dma_info *dma_info,
		   int prot,
		   enum ethosn_stream_id stream_id)
{
	const struct ethosn_dma_allocator_ops *ops = get_ops(allocator);
	int ret = -EINVAL;

	if (!ops)
		goto exit;

	if (!ops->map)
		goto exit;

	if (IS_ERR_OR_NULL(dma_info))
		goto exit;

	ret = ops->map(allocator, dma_info, prot, stream_id);

	if (ret)
		dev_err(allocator->dev, "failed mapping dma on stream %d\n",
			stream_id);
	else
		dev_dbg(allocator->dev,
			"DMA mapped. handle=0x%pK, iova=0x%llx, prot=0x%x, stream=%u\n",
			dma_info, dma_info->iova_addr, prot, stream_id);

exit:

	return ret;
}

void ethosn_dma_unmap(struct ethosn_dma_allocator *allocator,
		      struct ethosn_dma_info *const dma_info,
		      enum ethosn_stream_id stream_id)
{
	const struct ethosn_dma_allocator_ops *ops = get_ops(allocator);

	if (!ops)
		return;

	if (!ops->unmap)
		return;

	if (IS_ERR_OR_NULL(dma_info))
		return;

	ops->unmap(allocator, dma_info, stream_id);
}

void ethosn_dma_free(struct ethosn_dma_allocator *allocator,
		     struct ethosn_dma_info *const dma_info)
{
	const struct ethosn_dma_allocator_ops *ops = get_ops(allocator);

	if (!ops)
		return;

	if (IS_ERR_OR_NULL(dma_info))
		return;

	ops->free(allocator, dma_info);
}

struct ethosn_dma_info *ethosn_dma_alloc_and_map(
	struct ethosn_dma_allocator *allocator,
	const size_t size,
	int prot,
	enum ethosn_stream_id stream_id,
	gfp_t gfp,
	const char *debug_tag)
{
	struct ethosn_dma_info *dma_info = NULL;
	int ret;

	dma_info = ethosn_dma_alloc(allocator, size, gfp, debug_tag);
	if (IS_ERR_OR_NULL(dma_info))
		goto exit;

	ret = ethosn_dma_map(allocator, dma_info, prot, stream_id);
	if (ret < 0) {
		dev_err(allocator->dev, "failed to map stream %u\n", stream_id);
		goto exit_free_dma_info;
	}

exit:

	return dma_info;

exit_free_dma_info:
	ethosn_dma_free(allocator, dma_info);

	return NULL;
}

void ethosn_dma_unmap_and_free(struct ethosn_dma_allocator *allocator,
			       struct ethosn_dma_info *const dma_info,
			       enum ethosn_stream_id stream_id)
{
	ethosn_dma_unmap(allocator, dma_info, stream_id);
	ethosn_dma_free(allocator, dma_info);
}

struct ethosn_dma_info *ethosn_dma_import(
	struct ethosn_dma_allocator *allocator,
	int fd,
	size_t size)
{
	const struct ethosn_dma_allocator_ops *ops = get_ops(allocator);
	struct ethosn_dma_info *dma_info = NULL;

	if (!ops)
		goto exit;

	dma_info = ops->import(allocator, fd, size);

	if (IS_ERR_OR_NULL(dma_info)) {
		dev_err(allocator->dev, "failed to dma_import %zu bytes\n",
			dma_info->size);
		goto exit;
	}

	dev_dbg(allocator->dev,
		"DMA import. handle=0x%pK, cpu_addr=0x%pK, size=%zu\n",
		dma_info, dma_info->cpu_addr, dma_info->size);

exit:

	return dma_info;
}

void ethosn_dma_release(struct ethosn_dma_allocator *allocator,
			struct ethosn_dma_info *const dma_info)
{
	const struct ethosn_dma_allocator_ops *ops = get_ops(allocator);

	if (!ops)
		return;

	if (IS_ERR_OR_NULL(dma_info))
		return;

	ops->release(allocator, dma_info);
}

int ethosn_dma_mmap(struct ethosn_dma_allocator *allocator,
		    struct vm_area_struct *const vma,
		    const struct ethosn_dma_info *const dma_info)
{
	const struct ethosn_dma_allocator_ops *ops = get_ops(allocator);
	int ret = -EINVAL;

	if (!ops)
		goto exit;

	if (!ops->mmap)
		goto exit;

	return ops->mmap(allocator, vma, dma_info);

exit:

	return ret;
}

resource_size_t ethosn_dma_get_addr_size(struct ethosn_dma_allocator *allocator,
					 enum ethosn_stream_id stream_id)
{
	const struct ethosn_dma_allocator_ops *ops = get_ops(allocator);
	int ret = -EINVAL;

	if (!ops)
		goto exit;

	if (!ops->get_addr_size)
		goto exit;

	return ops->get_addr_size(allocator, stream_id);

exit:

	return ret;
}

dma_addr_t ethosn_dma_get_addr_base(struct ethosn_dma_allocator *allocator,
				    enum ethosn_stream_id stream_id)
{
	const struct ethosn_dma_allocator_ops *ops = get_ops(allocator);
	int ret = -EINVAL;

	if (!ops)
		goto exit;

	if (!ops->get_addr_base)
		goto exit;

	return ops->get_addr_base(allocator, stream_id);

exit:

	return ret;
}

void ethosn_dma_sync_for_device(struct ethosn_dma_allocator *allocator,
				struct ethosn_dma_info *dma_info)
{
	const struct ethosn_dma_allocator_ops *ops = get_ops(allocator);

	if (!ops)
		return;

	if (!ops->sync_for_device)
		return;

	if (IS_ERR_OR_NULL(dma_info))
		return;

	ops->sync_for_device(allocator, dma_info);
}

void ethosn_dma_sync_for_cpu(struct ethosn_dma_allocator *allocator,
			     struct ethosn_dma_info *dma_info)
{
	const struct ethosn_dma_allocator_ops *ops = get_ops(allocator);

	if (!ops)
		return;

	if (!ops->sync_for_cpu)
		return;

	if (IS_ERR_OR_NULL(dma_info))
		return;

	ops->sync_for_cpu(allocator, dma_info);
}

/* Exported for use by test module */
EXPORT_SYMBOL(ethosn_dma_sync_for_cpu);
