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

#include "ethosn_dma.h"

#include "ethosn_device.h"
#include "ethosn_dma_carveout.h"
#include "ethosn_dma_iommu.h"

#include <linux/iommu.h>

struct ethosn_dma_allocator *ethosn_dma_allocator_create(
	struct ethosn_device *npu)
{
	struct ethosn_dma_allocator *allocator;

	if (iommu_present(npu->dev->bus))
		allocator = ethosn_dma_iommu_allocator_create(npu);
	else
		allocator = ethosn_dma_carveout_allocator_create(npu);

	return allocator;
}

void ethosn_dma_allocator_destroy(struct ethosn_device *npu,
				  struct ethosn_dma_allocator *allocator)
{
	allocator->destroy(npu, allocator);
}

struct ethosn_dma_info *ethosn_dma_alloc(struct ethosn_device *npu,
					 const size_t size,
					 int prot,
					 enum ethosn_stream_id stream_id,
					 gfp_t gfp)
{
	struct ethosn_dma_allocator *allocator = npu->allocator;
	struct ethosn_dma_info *dma_info = NULL;

	dma_info = allocator->alloc(npu, allocator, size, prot, stream_id, gfp);

	if (IS_ERR_OR_NULL(dma_info))
		dev_dbg(npu->dev, "failed to dma_alloc %zu bytes\n",
			size);
	else
		dev_dbg(npu->dev,
			"DMA alloc. handle=0x%pK, cpu_addr=0x%pK, iova=0x%llx, size=%zu prot=0x%x\n",
			dma_info, dma_info->cpu_addr,
			dma_info->iova_addr, size, prot);

	return dma_info;
}

void ethosn_dma_free(struct ethosn_device *npu,
		     enum ethosn_stream_id stream_id,
		     struct ethosn_dma_info *const dma_info)
{
	struct ethosn_dma_allocator *allocator = npu->allocator;

	if (!IS_ERR_OR_NULL(dma_info)) {
		dev_dbg(npu->dev, "DMA free. handle=0x%pK\n", dma_info);
		allocator->free(npu, allocator, stream_id, dma_info);
	}
}

int ethosn_dma_mmap(struct ethosn_device *npu,
		    struct vm_area_struct *const vma,
		    const struct ethosn_dma_info *const dma_info)
{
	struct ethosn_dma_allocator *allocator = npu->allocator;
	int ret;

	ret = allocator->mmap(npu, vma, dma_info);

	return ret;
}

resource_size_t ethosn_dma_get_addr_size(struct ethosn_device *npu,
					 enum ethosn_stream_id stream_id)
{
	struct ethosn_dma_allocator *allocator = npu->allocator;

	return allocator->get_addr_size(npu, stream_id);
}

dma_addr_t ethosn_dma_get_addr_base(struct ethosn_device *npu,
				    enum ethosn_stream_id stream_id)
{
	struct ethosn_dma_allocator *allocator = npu->allocator;

	return allocator->get_addr_base(npu, stream_id);
}

void ethosn_dma_sync_for_device(struct ethosn_device *npu,
				struct ethosn_dma_info *dma_info)
{
	struct ethosn_dma_allocator *allocator = npu->allocator;

	if (!IS_ERR_OR_NULL(dma_info))
		allocator->sync_for_device(npu, dma_info);
}

void ethosn_dma_sync_for_cpu(struct ethosn_device *npu,
			     struct ethosn_dma_info *dma_info)
{
	struct ethosn_dma_allocator *allocator = npu->allocator;

	if (!IS_ERR_OR_NULL(dma_info))
		allocator->sync_for_cpu(npu, dma_info);
}
