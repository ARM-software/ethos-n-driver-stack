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

struct ethosn_allocator_internal {
	struct ethosn_dma_allocator allocator;

	struct device_node          *res_mem;
};

static struct ethosn_dma_info *carveout_alloc(
	struct ethosn_dma_allocator *allocator,
	const size_t size,
	gfp_t gfp)
{
	struct ethosn_dma_info *dma_info;
	void *cpu_addr = NULL;
	dma_addr_t dma_addr = 0;

	/* FIXME:- We cannot allocate addresses at different 512MB offsets */
	/* for the different streams. */
	dma_info = devm_kzalloc(allocator->dev,
				sizeof(struct ethosn_dma_info),
				GFP_KERNEL);
	if (!dma_info)
		return ERR_PTR(-ENOMEM);

	if (size) {
		cpu_addr =
			dma_alloc_wc(allocator->dev, size, &dma_addr, gfp);
		if (!cpu_addr) {
			dev_dbg(allocator->dev,
				"failed to dma_alloc %zu bytes\n",
				size);
			devm_kfree(allocator->dev, dma_info);

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

static int carveout_map(struct ethosn_dma_allocator *allocator,
			struct ethosn_dma_info *dma_info,
			int prot,
			enum ethosn_stream_id stream_id)
{
	return 0;
}

static void carveout_unmap(struct ethosn_dma_allocator *allocator,
			   struct ethosn_dma_info *dma_info,
			   enum ethosn_stream_id stream_id)
{}

static void carveout_free(struct ethosn_dma_allocator *allocator,
			  struct ethosn_dma_info *dma_info)
{
	const dma_addr_t dma_addr = dma_info->iova_addr;

	/* FIXME:- We cannot allocate addresses at different 512MB offsets */
	/* for the different streams. */
	if (dma_info->size)
		dma_free_wc(allocator->dev, dma_info->size,
			    dma_info->cpu_addr,
			    dma_addr);

	memset(dma_info, 0, sizeof(struct ethosn_dma_info));
	devm_kfree(allocator->dev, dma_info);
}

static void carveout_sync_for_device(struct ethosn_dma_allocator *allocator,
				     struct ethosn_dma_info *dma_info)
{}

static void carveout_sync_for_cpu(struct ethosn_dma_allocator *allocator,
				  struct ethosn_dma_info *dma_info)
{}

static int carveout_mmap(struct ethosn_dma_allocator *allocator,
			 struct vm_area_struct *const vma,
			 const struct ethosn_dma_info *const dma_info)
{
	const size_t size = dma_info->size;
	void *const cpu_addr = dma_info->cpu_addr;
	const dma_addr_t dma_addr = dma_info->iova_addr;

	const dma_addr_t mmap_addr =
		((dma_addr >> PAGE_SHIFT) + vma->vm_pgoff) << PAGE_SHIFT;

	int ret;

	ret = dma_mmap_wc(allocator->dev, vma, cpu_addr, dma_addr, size);

	if (ret)
		dev_warn(allocator->dev,
			 "Failed to DMA map buffer. handle=0x%pK, addr=0x%llx, size=%lu\n",
			 dma_info, mmap_addr, vma->vm_end - vma->vm_start);
	else
		dev_dbg(allocator->dev,
			"DMA map. handle=0x%pK, addr=0x%llx, start=0x%lx, size=%lu\n",
			dma_info, mmap_addr, vma->vm_start,
			vma->vm_end - vma->vm_start);

	return ret;
}

static dma_addr_t carveout_get_addr_base(
	struct ethosn_dma_allocator *_allocator,
	enum ethosn_stream_id stream_id)
{
	struct ethosn_allocator_internal *allocator =
		container_of(_allocator, typeof(*allocator), allocator);
	struct resource r;

	if (!allocator->res_mem)
		return 0;

	if (of_address_to_resource(allocator->res_mem, 0, &r))
		return 0;
	else
		return r.start;
}

static resource_size_t carveout_get_addr_size(
	struct ethosn_dma_allocator *_allocator,
	enum ethosn_stream_id stream_id)
{
	struct ethosn_allocator_internal *allocator =
		container_of(_allocator, typeof(*allocator), allocator);
	struct resource r;

	if (!allocator->res_mem)
		return 0;

	if (of_address_to_resource(allocator->res_mem, 0, &r))
		return 0;
	else
		return resource_size(&r);
}

static void carveout_allocator_destroy(struct ethosn_dma_allocator *allocator)
{
	struct device *dev = allocator->dev;

	memset(allocator, 0, sizeof(struct ethosn_dma_allocator));
	devm_kfree(dev, allocator);
}

struct ethosn_dma_allocator *ethosn_dma_carveout_allocator_create(
	struct device *dev)
{
	static struct ethosn_dma_allocator_ops ops = {
		.destroy         = carveout_allocator_destroy,
		.alloc           = carveout_alloc,
		.map             = carveout_map,
		.unmap           = carveout_unmap,
		.free            = carveout_free,
		.sync_for_device = carveout_sync_for_device,
		.sync_for_cpu    = carveout_sync_for_cpu,
		.mmap            = carveout_mmap,
		.get_addr_base   = carveout_get_addr_base,
		.get_addr_size   = carveout_get_addr_size,
	};
	struct ethosn_allocator_internal *allocator;
	struct device_node *res_mem;

	/* Iterrates backwards device tree to find a memory-region phandle */
	do {
		res_mem = of_parse_phandle(dev->of_node, "memory-region", 0);
		if (res_mem)
			break;

		/* TODO: Check if parent is null in case of reaching root
		 * Maybe check against dev->bus->dev_root to make sure root node
		 * is reached
		 */
		dev = dev->parent;
	} while (dev);

	if (!res_mem)
		return ERR_PTR(-EINVAL);

	allocator = devm_kzalloc(dev,
				 sizeof(struct ethosn_allocator_internal),
				 GFP_KERNEL);

	if (!allocator)
		return ERR_PTR(-ENOMEM);

	allocator->res_mem = res_mem;
	allocator->allocator.dev = dev;
	allocator->allocator.ops = &ops;

	return &allocator->allocator;
}
