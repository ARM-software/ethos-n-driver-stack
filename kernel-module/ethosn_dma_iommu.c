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

#include "ethosn_dma_iommu.h"

#include "ethosn_device.h"

#include <linux/iommu.h>
#include <linux/iova.h>
#include <linux/kernel.h>
#include <linux/version.h>
#include <linux/vmalloc.h>

#define IOMMU_FIRMWARE_ADDR_BASE        0x80000000
#define IOMMU_WORKING_DATA_ADDR_BASE    0xA0000000

/**
 * IOMMU_COMMAND_STREAM includes the region for command stream and all other
 * constant cu data (ie weights metadata and binding table).
 */
#define IOMMU_COMMAND_STREAM_ADDR_BASE  0xC0000000
#define IOMMU_DMA_ADDR_BASE             0xE0000000
#define IOMMU_ADDR_SIZE 0x20000000

struct ethosn_iommu_stream {
	void        *bitmap;
	dma_addr_t  addr_base;
	size_t      bits;
	struct page *page;
	spinlock_t  lock;
};

struct ethosn_iommu_domain {
	struct iommu_domain        *iommu_domain;
	struct ethosn_iommu_stream stream_firmware;
	struct ethosn_iommu_stream stream_working_data;
	struct ethosn_iommu_stream stream_command_stream;
	struct ethosn_iommu_stream stream_dma;
};

struct ethosn_allocator_internal {
	struct ethosn_dma_allocator allocator;
	/* Allocator private members */
	struct ethosn_iommu_domain  ethosn_iommu_domain;
};

struct ethosn_dma_info_internal {
	struct ethosn_dma_info info;
	/* Allocator private members */
	dma_addr_t             *dma_addr;
	struct page            **pages;
};

static struct ethosn_iommu_stream *iommu_get_stream(
	struct ethosn_iommu_domain *domain,
	enum ethosn_stream_id stream_id)
{
	struct ethosn_iommu_stream *stream = NULL;

	switch (stream_id) {
	case ETHOSN_STREAM_FIRMWARE:
		stream = &domain->stream_firmware;
		break;

	case ETHOSN_STREAM_WORKING_DATA:
		stream = &domain->stream_working_data;
		break;

	case ETHOSN_STREAM_COMMAND_STREAM:
		stream = &domain->stream_command_stream;
		break;

	case ETHOSN_STREAM_DMA:
		stream = &domain->stream_dma;
		break;

	default:
		break;
	}

	return stream;
}

static dma_addr_t iommu_get_addr_base(struct ethosn_device *ethosn,
				      enum ethosn_stream_id stream_id)
{
	dma_addr_t addr = 0;

	switch (stream_id) {
	case ETHOSN_STREAM_FIRMWARE:
		addr = IOMMU_FIRMWARE_ADDR_BASE;
		break;

	case ETHOSN_STREAM_WORKING_DATA:
		addr = IOMMU_WORKING_DATA_ADDR_BASE;
		break;

	case ETHOSN_STREAM_COMMAND_STREAM:
		addr = IOMMU_COMMAND_STREAM_ADDR_BASE;
		break;

	case ETHOSN_STREAM_DMA:
		addr = IOMMU_DMA_ADDR_BASE;
		break;

	default:
		break;
	}

	return addr;
}

static resource_size_t iommu_get_addr_size(struct ethosn_device *ethosn,
					   enum ethosn_stream_id stream_id)
{
	return IOMMU_ADDR_SIZE;
}

static dma_addr_t iommu_alloc_iova(struct ethosn_iommu_domain *domain,
				   enum ethosn_stream_id stream_id,
				   const size_t size)
{
	unsigned long start = 0;
	int nr_pages = DIV_ROUND_UP(size, PAGE_SIZE);
	unsigned long flags;
	struct ethosn_iommu_stream *stream = NULL;

	stream = iommu_get_stream(domain, stream_id);
	if (!stream)
		goto early_exit;

	spin_lock_irqsave(&stream->lock, flags);

	start = bitmap_find_next_zero_area(stream->bitmap, stream->bits, 0,
					   nr_pages, 0);
	if (start > stream->bits) {
		stream = NULL;
		goto ret;
	}

	bitmap_set(stream->bitmap, start, nr_pages);
ret:
	spin_unlock_irqrestore(&stream->lock, flags);
early_exit:
	if (stream)
		return stream->addr_base + start * PAGE_SIZE;
	else
		return 0;
}

static void iommu_free_iova(struct ethosn_iommu_domain *domain,
			    dma_addr_t start,
			    enum ethosn_stream_id stream_id,
			    size_t size)
{
	unsigned long flags;
	struct ethosn_iommu_stream *stream = NULL;
	int nr_pages = DIV_ROUND_UP(size, PAGE_SIZE);

	stream = iommu_get_stream(domain, stream_id);
	if (!stream)
		return;

	spin_lock_irqsave(&stream->lock, flags);

	bitmap_clear(stream->bitmap,
		     (start - stream->addr_base) / PAGE_SIZE,
		     nr_pages);

	spin_unlock_irqrestore(&stream->lock, flags);
}

static struct ethosn_dma_info *iommu_alloc(struct ethosn_device *ethosn,
					   struct ethosn_dma_allocator
					   *_allocator,
					   const size_t size,
					   int prot,
					   enum ethosn_stream_id stream_id,
					   gfp_t gfp)
{
	struct ethosn_allocator_internal *allocator =
		container_of(_allocator, typeof(*allocator), allocator);
	struct ethosn_iommu_domain *domain = &allocator->ethosn_iommu_domain;
	struct ethosn_iommu_stream *stream =
		iommu_get_stream(domain, stream_id);
	struct page **pages = NULL;
	struct ethosn_dma_info_internal *dma_info;
	void *cpu_addr = NULL;
	dma_addr_t *dma_addr = NULL, start_addr = 0;

	int nr_pages = DIV_ROUND_UP(size, PAGE_SIZE);
	int i, err, iommu_prot = 0;

	if ((prot & ETHOSN_PROT_READ) == ETHOSN_PROT_READ)
		iommu_prot |= IOMMU_READ;

	if ((prot & ETHOSN_PROT_WRITE) == ETHOSN_PROT_WRITE)
		iommu_prot |= IOMMU_WRITE;

	if (!stream)
		goto early_exit;

	dma_info =
		devm_kzalloc(ethosn->dev,
			     sizeof(struct ethosn_dma_info_internal),
			     GFP_KERNEL);
	if (!dma_info)
		goto early_exit;

	if (!size)
		goto ret;

	pages = (struct page **)
		devm_kzalloc(ethosn->dev,
			     sizeof(struct page *) * nr_pages,
			     GFP_KERNEL);
	if (!pages)
		goto free_dma_info;

	dma_addr = (dma_addr_t *)
		   devm_kzalloc(ethosn->dev,
				sizeof(dma_addr_t) * nr_pages,
				GFP_KERNEL);
	if (!dma_addr)
		goto free_pages_list;

	start_addr =
		iommu_alloc_iova(domain,
				 stream_id, size);
	if (!start_addr)
		goto free_dma_addr;

	dev_dbg(ethosn->dev,
		"%s: mapping %lu bytes starting at 0x%llX prot 0x%x\n",
		__func__,
		size, start_addr, iommu_prot);

	for (i = 0; i < nr_pages; ++i) {
		pages[i] = alloc_page(gfp);
		if (!pages[i])
			goto unmap_pages;

		if (stream->page)
			iommu_unmap(
				domain->iommu_domain,
				start_addr + i * PAGE_SIZE,
				PAGE_SIZE);

		err = iommu_map(
			domain->iommu_domain,
			start_addr + i * PAGE_SIZE,
			page_to_phys(pages[i]),
			PAGE_SIZE,
			iommu_prot);

		if (err) {
			dev_err(ethosn->dev,
				"failed to iommu map iova 0x%llX pa 0x%llX size %lu\n",
				start_addr + i * PAGE_SIZE,
				page_to_phys(pages[i]), PAGE_SIZE);
			goto unmap_pages;
		}

		dma_addr[i] = dma_map_page(ethosn->dev, pages[i], 0,
					   PAGE_SIZE,
					   DMA_BIDIRECTIONAL);

		if (dma_mapping_error(ethosn->dev, dma_addr[i])) {
			dev_err(ethosn->dev,
				"failed to dma map pa 0x%llX\n",
				page_to_phys(pages[i]));
			goto unmap_pages;
		}
	}

	cpu_addr = vmap(pages, nr_pages, 0, PAGE_KERNEL);
	if (!cpu_addr)
		goto unmap_pages;

ret:
	*dma_info = (struct ethosn_dma_info_internal) {
		.info = (struct ethosn_dma_info) {
			.size = size,
			.cpu_addr = cpu_addr,
			.iova_addr = start_addr,
		},
		.dma_addr = dma_addr,
		.pages = pages,
	};

	return &dma_info->info;
unmap_pages:
	iommu_free_iova(domain, start_addr, stream_id,
			size);
	for (i = 0; i < nr_pages; ++i)
		if (pages[i]) {
			dma_unmap_page(ethosn->dev, dma_addr[i],
				       PAGE_SIZE, DMA_BIDIRECTIONAL);
			iommu_unmap(domain->iommu_domain,
				    start_addr + i * PAGE_SIZE, PAGE_SIZE);
			__free_page(pages[i]);

			if (stream->page)
				iommu_map(
					domain->iommu_domain,
					start_addr + i * PAGE_SIZE,
					page_to_phys(stream->page),
					PAGE_SIZE,
					IOMMU_READ);
		}

free_dma_addr:
	devm_kfree(ethosn->dev, dma_addr);
free_pages_list:
	devm_kfree(ethosn->dev, pages);
free_dma_info:
	devm_kfree(ethosn->dev, dma_info);
early_exit:

	return ERR_PTR(-ENOMEM);
}

static void iommu_free(struct ethosn_device *const ethosn,
		       struct ethosn_dma_allocator *_allocator,
		       enum ethosn_stream_id stream_id,
		       struct ethosn_dma_info *const _dma_info)
{
	struct ethosn_dma_info_internal *dma_info =
		container_of(_dma_info, typeof(*dma_info), info);
	struct ethosn_allocator_internal *allocator =
		container_of(_allocator, typeof(*allocator), allocator);
	struct ethosn_iommu_domain *domain = &allocator->ethosn_iommu_domain;
	struct ethosn_iommu_stream *stream =
		iommu_get_stream(domain, stream_id);

	int nr_pages = DIV_ROUND_UP(dma_info->info.size, PAGE_SIZE);
	int i;

	if (!stream)
		return;

	if (dma_info->info.size) {
		vunmap(dma_info->info.cpu_addr);

		for (i = 0; i < nr_pages; ++i) {
			iommu_unmap(domain->iommu_domain,
				    dma_info->info.iova_addr + i * PAGE_SIZE,
				    PAGE_SIZE);
			dma_unmap_page(ethosn->dev, dma_info->dma_addr[i],
				       PAGE_SIZE, DMA_BIDIRECTIONAL);
			__free_page(dma_info->pages[i]);

			if (stream->page)
				iommu_map(
					domain->iommu_domain,
					dma_info->info.iova_addr +
					i * PAGE_SIZE,
					page_to_phys(stream->page),
					PAGE_SIZE,
					IOMMU_READ);
		}

		iommu_free_iova(domain,
				dma_info->info.iova_addr, stream_id,
				dma_info->info.size);
		devm_kfree(ethosn->dev, dma_info->dma_addr);
		devm_kfree(ethosn->dev, dma_info->pages);
	}

	memset(dma_info, 0, sizeof(*dma_info));
	devm_kfree(ethosn->dev, dma_info);
}

static void iommu_sync_for_device(struct ethosn_device *ethosn,
				  struct ethosn_dma_info *_dma_info)
{
	struct ethosn_dma_info_internal *dma_info =
		container_of(_dma_info, typeof(*dma_info), info);

	int nr_pages = DIV_ROUND_UP(dma_info->info.size, PAGE_SIZE);
	int i;

	for (i = 0; i < nr_pages; ++i)
		dma_sync_single_for_device(ethosn->dev,
					   dma_info->dma_addr[i], PAGE_SIZE,
					   DMA_TO_DEVICE);
}

static void iommu_sync_for_cpu(struct ethosn_device *ethosn,
			       struct ethosn_dma_info *_dma_info)
{
	struct ethosn_dma_info_internal *dma_info =
		container_of(_dma_info, typeof(*dma_info), info);

	int nr_pages = DIV_ROUND_UP(dma_info->info.size, PAGE_SIZE);
	int i;

	for (i = 0; i < nr_pages; ++i)
		dma_sync_single_for_cpu(ethosn->dev,
					dma_info->dma_addr[i],
					PAGE_SIZE, DMA_FROM_DEVICE);
}

static int iommu_mmap(struct ethosn_device *ethosn,
		      struct vm_area_struct *const vma,
		      const struct ethosn_dma_info *const _dma_info)
{
	struct ethosn_dma_info_internal *dma_info =
		container_of(_dma_info, typeof(*dma_info), info);
	int nr_pages = DIV_ROUND_UP(dma_info->info.size, PAGE_SIZE);
	int i;

	for (i = 0; i < nr_pages; ++i) {
		unsigned long addr = vma->vm_start + i * PAGE_SIZE;
		unsigned long pfn = page_to_pfn(dma_info->pages[i]);
		unsigned long size = PAGE_SIZE;

		if (remap_pfn_range(vma, addr, pfn, size, vma->vm_page_prot))
			return -EAGAIN;
	}

	return 0;
}

static int iommu_stream_init(struct ethosn_device *ethosn,
			     struct ethosn_iommu_domain *domain,
			     enum ethosn_stream_id stream_id,
			     size_t bitmap_size)
{
	struct ethosn_iommu_stream *stream =
		iommu_get_stream(domain, stream_id);
	int nr_pages = DIV_ROUND_UP(IOMMU_ADDR_SIZE, PAGE_SIZE);
	int i, k, err;

	dev_dbg(ethosn->dev,
		"%s: stream_id %u\n", __func__, stream_id);

	stream->bitmap =
		devm_kzalloc(ethosn->dev, bitmap_size, GFP_KERNEL);
	if (!stream->bitmap)
		return -ENOMEM;

	stream->addr_base = iommu_get_addr_base(ethosn, stream_id);
	stream->bits = bitmap_size * BITS_PER_BYTE;
	spin_lock_init(&stream->lock);

	if (stream_id == ETHOSN_STREAM_DMA)
		return 0;

	stream->page = alloc_page(GFP_KERNEL);

	if (!stream->page)
		goto free_bitmap;

	/*
	 * Map all the virtual space to a single physical page to be
	 * protected against speculative accesses.
	 */
	for (i = 0; i < nr_pages; ++i) {
		err = iommu_map(
			domain->iommu_domain,
			stream->addr_base + i * PAGE_SIZE,
			page_to_phys(stream->page),
			PAGE_SIZE,
			IOMMU_READ);

		if (err) {
			dev_err(ethosn->dev,
				"failed to iommu map iova 0x%llX pa 0x%llX size %lu\n",
				stream->addr_base + i * PAGE_SIZE,
				page_to_phys(stream->page), PAGE_SIZE);
			goto unmap_page;
		}
	}

	return 0;
unmap_page:
	for (k = 0; k < i; ++k)
		iommu_unmap(domain->iommu_domain,
			    stream->addr_base + k * PAGE_SIZE, PAGE_SIZE);

	__free_page(stream->page);
free_bitmap:
	devm_kfree(ethosn->dev, stream->bitmap);

	return -ENOMEM;
}

static void iommu_stream_deinit(struct ethosn_device *ethosn,
				struct ethosn_iommu_domain *domain,
				enum ethosn_stream_id stream_id)
{
	struct ethosn_iommu_stream *stream =
		iommu_get_stream(domain, stream_id);
	int nr_pages = DIV_ROUND_UP(IOMMU_ADDR_SIZE, PAGE_SIZE);
	int i;

	dev_dbg(ethosn->dev,
		"%s: stream_id %u\n", __func__, stream_id);

	if (!stream)
		return;

	devm_kfree(ethosn->dev,
		   stream->bitmap);

	if (!stream->page)
		return;

	/* Unmap all the virtual space (see iommu_stream_init). */
	for (i = 0; i < nr_pages; ++i)
		iommu_unmap(domain->iommu_domain,
			    stream->addr_base + i * PAGE_SIZE,
			    PAGE_SIZE);

	__free_page(stream->page);
}

static void iommu_allocator_destroy(struct ethosn_device *ethosn,
				    struct ethosn_dma_allocator *_allocator)
{
	struct ethosn_allocator_internal *allocator =
		container_of(_allocator, typeof(*allocator), allocator);

	iommu_stream_deinit(ethosn, &allocator->ethosn_iommu_domain,
			    ETHOSN_STREAM_FIRMWARE);
	iommu_stream_deinit(ethosn, &allocator->ethosn_iommu_domain,
			    ETHOSN_STREAM_WORKING_DATA);
	iommu_stream_deinit(ethosn, &allocator->ethosn_iommu_domain,
			    ETHOSN_STREAM_COMMAND_STREAM);
	iommu_stream_deinit(ethosn, &allocator->ethosn_iommu_domain,
			    ETHOSN_STREAM_DMA);
	devm_kfree(ethosn->dev, allocator);
}

struct ethosn_dma_allocator *ethosn_dma_iommu_allocator_create(
	struct ethosn_device *ethosn)
{
	struct ethosn_allocator_internal *allocator;
	struct ethosn_iommu_domain *domain;
	size_t bitmap_size;
	int ret;

	allocator = devm_kzalloc(ethosn->dev,
				 sizeof(struct ethosn_allocator_internal),
				 GFP_KERNEL);
	if (!allocator)
		return ERR_PTR(-ENOMEM);

	domain = &allocator->ethosn_iommu_domain;
	domain->iommu_domain = iommu_get_domain_for_dev(ethosn->dev);

	bitmap_size = BITS_TO_LONGS(IOMMU_ADDR_SIZE >> PAGE_SHIFT) *
		      sizeof(unsigned long);

	ret = iommu_stream_init(ethosn, domain, ETHOSN_STREAM_FIRMWARE,
				bitmap_size);
	if (ret)
		goto err_stream_firmware;

	ret = iommu_stream_init(ethosn, domain, ETHOSN_STREAM_WORKING_DATA,
				bitmap_size);
	if (ret)
		goto err_stream_working_data;

	ret = iommu_stream_init(ethosn, domain, ETHOSN_STREAM_COMMAND_STREAM,
				bitmap_size);
	if (ret)
		goto err_stream_command_stream;

	ret = iommu_stream_init(ethosn, domain, ETHOSN_STREAM_DMA,
				bitmap_size);
	if (ret)
		goto err_stream_dma;

	allocator->allocator = (struct ethosn_dma_allocator) {
		.destroy = iommu_allocator_destroy,
		.alloc = iommu_alloc,
		.free = iommu_free,
		.sync_for_device = iommu_sync_for_device,
		.sync_for_cpu = iommu_sync_for_cpu,
		.mmap = iommu_mmap,
		.get_addr_base = iommu_get_addr_base,
		.get_addr_size = iommu_get_addr_size,
	};

	return &allocator->allocator;

err_stream_dma:
	iommu_stream_deinit(ethosn, domain,
			    ETHOSN_STREAM_COMMAND_STREAM);
err_stream_command_stream:
	iommu_stream_deinit(ethosn, domain,
			    ETHOSN_STREAM_WORKING_DATA);
err_stream_working_data:
	iommu_stream_deinit(ethosn, domain,
			    ETHOSN_STREAM_FIRMWARE);
err_stream_firmware:
	devm_kfree(ethosn->dev, allocator);

	return ERR_PTR(-ENOMEM);
}
