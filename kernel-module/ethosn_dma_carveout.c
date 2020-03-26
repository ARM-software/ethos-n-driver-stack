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

#include "ethosn_dma_carveout.h"

#include "ethosn_device.h"

#include <linux/dma-mapping.h>
#include <linux/iommu.h>
#include <linux/of_address.h>

static struct ethosn_dma_info *carveout_alloc(struct ethosn_device *ethosn,
					      struct ethosn_dma_allocator
					      *_allocator,
					      const size_t size,
					      int prot,
					      enum ethosn_stream_id stream_id,
					      gfp_t gfp)
{
	struct ethosn_dma_info *dma_info;
	void *cpu_addr = NULL;
	dma_addr_t dma_addr = 0;

	/* FIXME:- We cannot allocate addresses at different 512MB offsets */
	/* for the different streams. */
	stream_id = 0;
	/* Protection cannot be enforced. */
	prot = 0;
	dma_info = devm_kzalloc(ethosn->dev, sizeof(struct ethosn_dma_info),
				GFP_KERNEL);
	if (!dma_info)
		return ERR_PTR(-ENOMEM);

	if (size) {
		cpu_addr = dma_alloc_wc(ethosn->dev, size, &dma_addr, gfp);
		if (!cpu_addr) {
			dev_dbg(ethosn->dev, "failed to dma_alloc %zu bytes\n",
				size);
			devm_kfree(ethosn->dev, dma_info);

			return ERR_PTR(-ENOMEM);
		}
	}

	*dma_info = (struct ethosn_dma_info) {
		.size = size,
		.cpu_addr = cpu_addr,
		.iova_addr = dma_addr,
	};

	return dma_info;
}

static void carveout_free(struct ethosn_device *const ethosn,
			  struct ethosn_dma_allocator *allocator,
			  enum ethosn_stream_id stream_id,
			  struct ethosn_dma_info *dma_info)
{
	const dma_addr_t dma_addr = dma_info->iova_addr;

	/* FIXME:- We cannot allocate addresses at different 512MB offsets */
	/* for the different streams. */
	stream_id = 0;
	if (dma_info->size)
		dma_free_wc(ethosn->dev, dma_info->size, dma_info->cpu_addr,
			    dma_addr);

	memset(dma_info, 0, sizeof(struct ethosn_dma_info));
	devm_kfree(ethosn->dev, dma_info);
}

static void carveout_sync_for_device(struct ethosn_device *ethosn,
				     struct ethosn_dma_info *dma_info)
{}

static void carveout_sync_for_cpu(struct ethosn_device *ethosn,
				  struct ethosn_dma_info *dma_info)
{}

static int carveout_mmap(struct ethosn_device *ethosn,
			 struct vm_area_struct *const vma,
			 const struct ethosn_dma_info *const dma_info)
{
	const size_t size = dma_info->size;
	void *const cpu_addr = dma_info->cpu_addr;
	const dma_addr_t dma_addr = dma_info->iova_addr;

	const dma_addr_t mmap_addr =
		((dma_addr >> PAGE_SHIFT) + vma->vm_pgoff) << PAGE_SHIFT;

	int ret;

	ret = dma_mmap_wc(ethosn->dev, vma, cpu_addr, dma_addr, size);

	if (ret)
		dev_warn(ethosn->dev,
			 "Failed to DMA map buffer. handle=0x%pK, addr=0x%llx, size=%lu\n",
			 dma_info, mmap_addr, vma->vm_end - vma->vm_start);
	else
		dev_dbg(ethosn->dev,
			"DMA map. handle=0x%pK, addr=0x%llx, start=0x%lx, size=%lu\n",
			dma_info, mmap_addr, vma->vm_start,
			vma->vm_end - vma->vm_start);

	return ret;
}

static dma_addr_t carveout_get_addr_base(struct ethosn_device *ethosn,
					 enum ethosn_stream_id stream_id)
{
	struct device_node *res_mem;
	struct resource r;

	res_mem =
		of_parse_phandle(ethosn->dev->of_node, "memory-region", 0);
	if (!res_mem)
		return 0;

	if (of_address_to_resource(res_mem, 0, &r))
		return 0;
	else
		return r.start;
}

static resource_size_t carveout_get_addr_size(struct ethosn_device *ethosn,
					      enum ethosn_stream_id stream_id)
{
	struct device_node *res_mem;
	struct resource r;

	res_mem =
		of_parse_phandle(ethosn->dev->of_node, "memory-region", 0);
	if (!res_mem)
		return 0;

	if (of_address_to_resource(res_mem, 0, &r))
		return 0;
	else
		return resource_size(&r);
}

static void carveout_allocator_destroy(struct ethosn_device *ethosn,
				       struct ethosn_dma_allocator *allocator)
{
	memset(allocator, 0, sizeof(struct ethosn_dma_allocator));
	devm_kfree(ethosn->dev, allocator);
}

struct ethosn_dma_allocator *ethosn_dma_carveout_allocator_create(
	struct ethosn_device *ethosn)
{
	struct ethosn_dma_allocator *allocator;

	allocator = devm_kzalloc(ethosn->dev,
				 sizeof(struct ethosn_dma_allocator),
				 GFP_KERNEL);

	if (allocator)
		*allocator = (struct ethosn_dma_allocator) {
			.destroy = carveout_allocator_destroy,
			.alloc = carveout_alloc,
			.free = carveout_free,
			.sync_for_device = carveout_sync_for_device,
			.sync_for_cpu = carveout_sync_for_cpu,
			.mmap = carveout_mmap,
			.get_addr_base = carveout_get_addr_base,
			.get_addr_size = carveout_get_addr_size,
		};
	else
		allocator = ERR_PTR(-ENOMEM);

	return allocator;
}
