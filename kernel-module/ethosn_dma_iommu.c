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

#include "ethosn_dma_iommu.h"

#include "ethosn_backport.h"
#include "ethosn_device.h"

#include <linux/dma-buf.h>
#include <linux/iommu.h>
#include <linux/iova.h>
#include <linux/kernel.h>
#include <linux/scatterlist.h>
#include <linux/slab.h>
#include <linux/version.h>
#include <linux/vmalloc.h>

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
	struct ethosn_iommu_stream stream_weight_data;
	struct ethosn_iommu_stream stream_buffer_data;
	struct ethosn_iommu_stream stream_intermediate_data;
};

struct ethosn_allocator_internal {
	struct ethosn_dma_sub_allocator allocator;
	/* Allocator private members */
	struct ethosn_iommu_domain      ethosn_iommu_domain;
};

struct dma_buf_internal {
	struct dma_buf            *dmabuf;
	int                       fd;
	/* Scatter-gather table of the imported buffer. */
	struct sg_table           *sgt;
	/* dma-buf attachment of the imported buffer. */
	struct dma_buf_attachment *attachment;
};

struct ethosn_dma_info_internal {
	struct ethosn_dma_info  info;
	/* Allocator private members */
	dma_addr_t              *dma_addr;
	struct page             **pages;
	struct dma_buf_internal *dma_buf_internal;
	struct scatterlist      **scatterlist;
};

static phys_addr_t ethosn_page_to_phys(int index,
				       struct ethosn_dma_info_internal
				       *dma_info)
{
	phys_addr_t ret = 0;

	if (dma_info->dma_buf_internal)
		ret = sg_phys(dma_info->scatterlist[index]);
	else
		ret = page_to_phys(dma_info->pages[index]);

	return ret;
}

static size_t ethosn_page_size(unsigned int start,
			       unsigned int end,
			       struct ethosn_dma_info_internal *dma_info)
{
	size_t size = 0;
	unsigned int i;

	if (dma_info->dma_buf_internal)
		for (i = start; i < end; i++)
			size += sg_dma_len(dma_info->scatterlist[i]);

	else
		size = (end - start) * PAGE_SIZE;

	return size;
}

static int ethosn_nr_sg_objects(struct ethosn_dma_info_internal *dma_info)
{
	int nr_pages = 0;

	if (dma_info->dma_buf_internal)
		nr_pages = dma_info->dma_buf_internal->sgt->nents;
	else
		nr_pages = DIV_ROUND_UP(dma_info->info.size, PAGE_SIZE);

	return nr_pages;
}

static int ethosn_nr_pages(struct ethosn_dma_info_internal *dma_info)
{

	/* info.size is calculated on import and can be used for scatterlists
	 * also.
	 */
	return DIV_ROUND_UP(dma_info->info.size, PAGE_SIZE);
}

static struct ethosn_iommu_stream *iommu_get_stream(
	struct ethosn_iommu_domain *domain,
	enum ethosn_stream_type stream_type)
{
	struct ethosn_iommu_stream *stream = NULL;

	switch (stream_type) {
	case ETHOSN_STREAM_FIRMWARE:
	case ETHOSN_STREAM_PLE_CODE:
		stream = &domain->stream_firmware;
		break;

	case ETHOSN_STREAM_WORKING_DATA:
	case ETHOSN_STREAM_DEBUG:
		stream = &domain->stream_working_data;
		break;

	case ETHOSN_STREAM_COMMAND_STREAM:
		stream = &domain->stream_command_stream;
		break;

	case ETHOSN_STREAM_WEIGHT_DATA:
		stream = &domain->stream_weight_data;
		break;

	case ETHOSN_STREAM_IO_BUFFER:
		stream = &domain->stream_buffer_data;
		break;

	case ETHOSN_STREAM_INTERMEDIATE_BUFFER:
		stream = &domain->stream_intermediate_data;
		break;

	default:
		break;
	}

	return stream;
}

static dma_addr_t iommu_get_addr_base(
	struct ethosn_dma_sub_allocator *allocator,
	enum ethosn_stream_type stream_type)
{
	dma_addr_t addr = 0;

	switch (stream_type) {
	case ETHOSN_STREAM_FIRMWARE:
		addr = IOMMU_FIRMWARE_ADDR_BASE;
		break;

	case ETHOSN_STREAM_WORKING_DATA:
		addr = IOMMU_WORKING_DATA_ADDR_BASE;
		break;

	case ETHOSN_STREAM_COMMAND_STREAM:
		addr = IOMMU_COMMAND_STREAM_ADDR_BASE;
		break;

	case ETHOSN_STREAM_PLE_CODE:
		addr = IOMMU_FIRMWARE_ADDR_BASE;
		break;

	case ETHOSN_STREAM_WEIGHT_DATA:
		addr = IOMMU_WEIGHT_DATA_ADDR_BASE;
		break;

	case ETHOSN_STREAM_IO_BUFFER:
		addr = IOMMU_BUFFER_ADDR_BASE;
		break;

	case ETHOSN_STREAM_INTERMEDIATE_BUFFER:
		addr = IOMMU_INTERMEDIATE_BUFFER_ADDR_BASE;
		break;

	default:
		break;
	}

	return addr;
}

static resource_size_t iommu_get_addr_size(
	struct ethosn_dma_sub_allocator *allocator,
	enum ethosn_stream_type stream_type)
{
	return IOMMU_ADDR_SIZE;
}

static dma_addr_t iommu_alloc_iova(struct device *dev,
				   struct ethosn_dma_info_internal *dma,
				   struct ethosn_iommu_stream *stream)
{
	unsigned long start = 0;
	unsigned long flags;
	int ret;
	int nr_pages = ethosn_nr_pages(dma);
	dma_addr_t iova = 0;

	spin_lock_irqsave(&stream->lock, flags);

	ret = ethosn_bitmap_find_next_zero_area(dev,
						&stream->bitmap, &stream->bits,
						nr_pages, &start);
	if (ret)
		goto ret;

	bitmap_set(stream->bitmap, start, nr_pages);

	iova = stream->addr_base + PAGE_SIZE * start;

ret:
	spin_unlock_irqrestore(&stream->lock, flags);

	return iova;
}

static void iommu_free_iova(dma_addr_t start,
			    struct ethosn_iommu_stream *stream,
			    int nr_pages)
{
	unsigned long flags;

	if (!stream)
		return;

	spin_lock_irqsave(&stream->lock, flags);

	bitmap_clear(stream->bitmap,
		     (start - stream->addr_base) / PAGE_SIZE,
		     nr_pages);

	spin_unlock_irqrestore(&stream->lock, flags);
}

static void iommu_free_pages(struct ethosn_dma_sub_allocator *allocator,
			     dma_addr_t dma_addr[],
			     struct page *pages[],
			     int nr_pages)
{
	int i;

	for (i = 0; i < nr_pages; ++i)
		if (dma_addr[i] && pages[i])
			dma_free_pages(allocator->dev, PAGE_SIZE, pages[i],
				       dma_addr[i], DMA_BIDIRECTIONAL);
}

static struct ethosn_dma_info *iommu_alloc(
	struct ethosn_dma_sub_allocator *allocator,
	const size_t size,
	gfp_t gfp)
{
	struct page **pages = NULL;
	struct ethosn_dma_info_internal *dma_info;
	void *cpu_addr = NULL;
	dma_addr_t *dma_addr = NULL;
	int nr_pages = DIV_ROUND_UP(size, PAGE_SIZE);
	int i;

	dma_info =
		devm_kzalloc(allocator->dev,
			     sizeof(struct ethosn_dma_info_internal),
			     GFP_KERNEL);
	if (!dma_info)
		goto early_exit;

	if (!size)
		goto ret;

	pages = (struct page **)
		devm_kzalloc(allocator->dev,
			     sizeof(struct page *) * nr_pages,
			     GFP_KERNEL);
	if (!pages)
		goto free_dma_info;

	dma_addr = (dma_addr_t *)
		   devm_kzalloc(allocator->dev,
				sizeof(dma_addr_t) * nr_pages,
				GFP_KERNEL);
	if (!dma_addr)
		goto free_pages_list;

	for (i = 0; i < nr_pages; ++i) {
		pages[i] = dma_alloc_pages(allocator->dev, PAGE_SIZE,
					   &dma_addr[i], DMA_BIDIRECTIONAL,
					   gfp);

		if (!pages[i])
			goto free_pages;

		if (dma_mapping_error(allocator->dev, dma_addr[i])) {
			dev_err(allocator->dev,
				"failed to dma map pa 0x%llX\n",
				page_to_phys(pages[i]));
			__free_page(pages[i]);
			goto free_pages;
		}
	}

	cpu_addr = vmap(pages, nr_pages, 0, PAGE_KERNEL);
	if (!cpu_addr)
		goto free_pages;

	dev_dbg(allocator->dev,
		"Allocated DMA. handle=%pK allocator->dev = %pK",
		dma_info, allocator->dev);

ret:
	*dma_info = (struct ethosn_dma_info_internal) {
		.info = (struct ethosn_dma_info) {
			.size = size,
			.cpu_addr = cpu_addr,
			.iova_addr = 0
		},
		.dma_addr = dma_addr,
		.pages = pages
	};

	return &dma_info->info;

free_pages:
	iommu_free_pages(allocator, dma_addr, pages, i);
free_pages_list:
	devm_kfree(allocator->dev, pages);
free_dma_info:
	devm_kfree(allocator->dev, dma_info);
early_exit:

	return ERR_PTR(-ENOMEM);
}

static void iommu_unmap_iova_pages(struct ethosn_dma_info_internal *dma_info,
				   struct iommu_domain *domain,
				   struct ethosn_iommu_stream *stream,
				   int nr_pages)
{
	int i;

	for (i = 0; i < nr_pages; ++i) {
		unsigned long iova_addr =
			dma_info->info.iova_addr + ethosn_page_size(0, i,
								    dma_info);

		if (dma_info->pages[i]) {
			/* TODO: Should handle error here */
			iommu_unmap(domain, iova_addr,
				    ethosn_page_size(i, i + 1, dma_info));

			if (stream->page)
				iommu_map(
					domain,
					iova_addr,
					page_to_phys(stream->page),
					PAGE_SIZE,
					IOMMU_READ);
		}
	}
}

static int iommu_iova_map(struct ethosn_dma_sub_allocator *allocator,
			  struct ethosn_dma_info *_dma_info,
			  int prot)
{
	struct ethosn_allocator_internal *allocator_private =
		container_of(allocator, typeof(*allocator_private), allocator);
	struct ethosn_iommu_domain *domain =
		&allocator_private->ethosn_iommu_domain;
	struct ethosn_iommu_stream *stream =
		iommu_get_stream(domain, _dma_info->stream_type);
	struct ethosn_dma_info_internal *dma_info =
		container_of(_dma_info, typeof(*dma_info), info);
	int nr_scatter_pages = ethosn_nr_sg_objects(dma_info);
	dma_addr_t start_addr = 0;
	int i, err, iommu_prot = 0;

	if (!dma_info->info.size)
		goto ret;

	if (!stream)
		goto early_exit;

	if (!dma_info->pages)
		goto early_exit;

	start_addr = iommu_alloc_iova(allocator_private->allocator.dev,
				      dma_info, stream);
	if (!start_addr)
		goto early_exit;

	if ((prot & ETHOSN_PROT_READ) == ETHOSN_PROT_READ)
		iommu_prot |= IOMMU_READ;

	if ((prot & ETHOSN_PROT_WRITE) == ETHOSN_PROT_WRITE)
		iommu_prot |= IOMMU_WRITE;

	if ((dma_info->info.iova_addr) &&
	    (dma_info->info.iova_addr != start_addr)) {
		dev_err(allocator->dev,
			"Invalid iova: 0x%llX != 0x%llX\n",
			dma_info->info.iova_addr, start_addr);
		goto free_iova;
	}

	dma_info->info.iova_addr = start_addr;

	dev_dbg(allocator->dev,
		"%s: mapping %lu bytes starting at 0x%llX prot 0x%x\n",
		__func__, dma_info->info.size, start_addr, iommu_prot);

	for (i = 0; i < nr_scatter_pages; ++i) {
		if (stream->page)
			iommu_unmap(
				domain->iommu_domain,
				start_addr + ethosn_page_size(0, i, dma_info),
				PAGE_SIZE);

		err = iommu_map(
			domain->iommu_domain,
			start_addr + ethosn_page_size(0, i, dma_info),
			ethosn_page_to_phys(i, dma_info),
			ethosn_page_size(i, i + 1, dma_info),
			iommu_prot);

		if (err) {
			dev_err(
				allocator->dev,
				"failed to iommu map iova 0x%llX pa 0x%llX size %lu\n",
				start_addr + ethosn_page_size(0, i, dma_info),
				ethosn_page_to_phys(i, dma_info),
				ethosn_page_size(i, i + 1, dma_info));
			goto unmap_pages;
		}
	}

ret:

	return 0;

unmap_pages:
	/* remap the current i-th page if it needs to */
	if (stream->page)
		iommu_map(
			domain->iommu_domain,
			start_addr + ethosn_page_size(0, i, dma_info),
			page_to_phys(stream->page),
			PAGE_SIZE,
			IOMMU_READ);

	/* Unmap only the actual number of pages mapped i.e. i */
	iommu_unmap_iova_pages(dma_info, domain->iommu_domain, stream, i);

free_iova:

	/* iommu_alloc_iova allocs the total number of pages,
	 * so it needs to free all of iovas irrespectively of
	 * how many have been actually mapped.
	 * Use start_addr since dma_info isn't updated in the
	 * case of error.
	 */
	iommu_free_iova(start_addr, stream, ethosn_nr_pages(dma_info));

early_exit:

	return -ENOMEM;
}

static struct ethosn_dma_info *iommu_import(
	struct ethosn_dma_sub_allocator *allocator,
	int fd,
	size_t size)
{
	struct page **pages = NULL;
	struct ethosn_dma_info_internal *dma_info;
	dma_addr_t *dma_addr = NULL;
	int i = 0;
	size_t scatterlist_size = 0;
	struct dma_buf_internal *dma_buf_internal = NULL;
	struct scatterlist **sctrlst = NULL;
	struct scatterlist *scatterlist = NULL;
	struct scatterlist *tmp_scatterlist = NULL;
	struct device *parent_device;

	dma_info =
		devm_kzalloc(allocator->dev,
			     sizeof(struct ethosn_dma_info_internal),
			     GFP_KERNEL);
	if (!dma_info) {
		dev_err(allocator->dev,
			"iommu_import: devm_kzalloc for dma_info failed");
		goto early_exit;
	}

	dma_buf_internal =
		devm_kzalloc(allocator->dev,
			     sizeof(struct dma_buf_internal),
			     GFP_KERNEL);
	if (!dma_buf_internal) {
		dev_err(allocator->dev,
			"iommu_import: devm_kzalloc for dma_buf_internal failed");

		goto free_dma_info;
	}

	dma_buf_internal->fd = fd;

	dma_buf_internal->dmabuf = dma_buf_get(fd);
	if (IS_ERR(dma_buf_internal->dmabuf)) {
		dev_err(allocator->dev, "iommu_import: dma_buf_get failed");
		goto free_buf_internal;
	}

	/* We can't pass the allocator device to dma_buf_attach, which leads to
	 * the linux dma framework attempting to map the buffer using the iommu
	 * mentioned in the dts for the allocator, which is not what we want
	 * because we are handling the mapping ourselves. This was leading to
	 * two mappings occurring which was leading to crashes and corrupted
	 * data.
	 *
	 * Instead we pass the ethosn_core device, which does not have an
	 * associated iommu, so the linux dma framework does a "direct" mapping,
	 * which doesn't seem to cause any problems.
	 */
	parent_device = allocator->dev->parent->parent;
	dma_buf_internal->attachment = dma_buf_attach(dma_buf_internal->dmabuf,
						      parent_device);
	if (IS_ERR(dma_buf_internal->attachment)) {
		dev_err(allocator->dev, "iommu_import: dma_buf_attach failed");
		goto fail_put;
	}

	dma_buf_internal->sgt = ethosn_dma_buf_map_attachment(
		dma_buf_internal->attachment);
	if (IS_ERR(dma_buf_internal->sgt)) {
		dev_err(allocator->dev,
			"iommu_import: ethosn_dma_buf_map_attachment failed");
		goto fail_detach;
	}

	sctrlst = (struct scatterlist **)
		  devm_kzalloc(allocator->dev,
			       sizeof(struct scatterlist *) *
			       dma_buf_internal->sgt->nents,
			       GFP_KERNEL);
	if (!sctrlst) {
		dev_err(allocator->dev,
			"iommu_import: devm_kzalloc for sctrlst failed");
		goto fail_unmap_attachment;
	}

	pages = (struct page **)
		devm_kzalloc(allocator->dev,
			     sizeof(struct page *) *
			     dma_buf_internal->sgt->nents,
			     GFP_KERNEL);
	if (!pages) {
		dev_err(allocator->dev,
			"iommu_import: devm_kzalloc for pages failed");
		goto free_scatterlist;
	}

	dma_addr = (dma_addr_t *)
		   devm_kzalloc(
		allocator->dev,
		sizeof(dma_addr_t) * dma_buf_internal->sgt->nents,
		GFP_KERNEL);

	if (!dma_addr) {
		dev_err(allocator->dev,
			"iommu_import: devm_kzalloc for dma_addr failed");
		goto free_pages_list;
	}

	/* Note:
	 * we copy the content of sg_table and scatterlist structs into the
	 * ethosn_dma_info_internal and ethosn_dma_info ones so that we can
	 * reuse most of the API functions we already made for
	 * ETHOSN_CREATE_BUFFER ioctl.
	 */
	scatterlist = dma_buf_internal->sgt->sgl;

	for_each_sg(scatterlist, tmp_scatterlist, dma_buf_internal->sgt->nents,
		    i) {
		if (tmp_scatterlist->offset != 0) {
			dev_err(
				allocator->dev,
				"iommu_import: failed to iommu import scatterlist offset is not zero, we only support zero");
			goto free_dma_address;
		}

		pages[i] = sg_page(tmp_scatterlist);
		dma_addr[i] = sg_dma_address(tmp_scatterlist);
		scatterlist_size += sg_dma_len(tmp_scatterlist);
		sctrlst[i] = tmp_scatterlist;
	}

	if (scatterlist_size < size) {
		dev_err(allocator->dev,
			"iommu_import: Provided buffer size does not match scatterlist");
		goto free_dma_address;
	}

	dev_dbg(allocator->dev,
		"iommu_import: Imported shared DMA buffer. handle=%pK",
		dma_info);

	*dma_info = (struct ethosn_dma_info_internal) {
		.info = (struct ethosn_dma_info) {
			.size = scatterlist_size,
			.cpu_addr = NULL,
			.iova_addr = 0
		},
		.dma_addr = dma_addr,
		.pages = pages,
		.dma_buf_internal = dma_buf_internal,
		.scatterlist = sctrlst
	};

	return &dma_info->info;

free_dma_address:
	devm_kfree(allocator->dev, dma_addr);
free_pages_list:
	devm_kfree(allocator->dev, pages);
free_scatterlist:
	devm_kfree(allocator->dev, sctrlst);
fail_unmap_attachment:
	dma_buf_unmap_attachment(dma_buf_internal->attachment,
				 dma_buf_internal->sgt, DMA_BIDIRECTIONAL);
fail_detach:
	dma_buf_detach(dma_buf_internal->dmabuf,
		       dma_buf_internal->attachment);
fail_put:
	dma_buf_put(dma_buf_internal->dmabuf);
free_buf_internal:
	memset(dma_buf_internal, 0, sizeof(*dma_buf_internal));
	devm_kfree(allocator->dev, dma_buf_internal);
free_dma_info:
	memset(dma_info, 0, sizeof(*dma_info));
	devm_kfree(allocator->dev, dma_info);
early_exit:

	return ERR_PTR(-ENOMEM);
}

static void iommu_release(struct ethosn_dma_sub_allocator *allocator,
			  struct ethosn_dma_info **_dma_info)
{
	struct ethosn_dma_info_internal *dma_info =
		container_of(*_dma_info, typeof(*dma_info), info);
	struct dma_buf_internal *dma_buf_internal = dma_info->dma_buf_internal;

	if (dma_info->info.size) {
		devm_kfree(allocator->dev, dma_info->dma_addr);
		devm_kfree(allocator->dev, dma_info->pages);
		devm_kfree(allocator->dev, dma_info->scatterlist);
	}

	dma_buf_unmap_attachment(dma_buf_internal->attachment,
				 dma_buf_internal->sgt, DMA_BIDIRECTIONAL);
	dma_buf_detach(dma_buf_internal->dmabuf,
		       dma_buf_internal->attachment);
	dma_buf_put(dma_buf_internal->dmabuf);

	memset(dma_buf_internal, 0, sizeof(*dma_buf_internal));
	devm_kfree(allocator->dev, dma_buf_internal);
	memset(dma_info, 0, sizeof(*dma_info));
	devm_kfree(allocator->dev, dma_info);

	/* Clear the caller's pointer, so they aren't left with it dangling */
	*_dma_info = (struct ethosn_dma_info *)NULL;
}

static void iommu_iova_unmap(struct ethosn_dma_sub_allocator *allocator,
			     struct ethosn_dma_info *const _dma_info)
{
	struct ethosn_dma_info_internal *dma_info =
		container_of(_dma_info, typeof(*dma_info), info);
	struct ethosn_allocator_internal *allocator_private =
		container_of(allocator, typeof(*allocator_private), allocator);
	struct ethosn_iommu_domain *domain =
		&allocator_private->ethosn_iommu_domain;
	struct ethosn_iommu_stream *stream =
		iommu_get_stream(domain, _dma_info->stream_type);

	if (!stream)
		return;

	if (dma_info->info.size) {
		int nr_scatter_pages = ethosn_nr_sg_objects(dma_info);
		int nr_pages = ethosn_nr_pages(dma_info);

		iommu_unmap_iova_pages(dma_info, domain->iommu_domain, stream,
				       nr_scatter_pages);

		iommu_free_iova(dma_info->info.iova_addr, stream, nr_pages);
	}
}

static void iommu_free(struct ethosn_dma_sub_allocator *allocator,
		       struct ethosn_dma_info **_dma_info)
{
	struct ethosn_dma_info_internal *dma_info =
		container_of(*_dma_info, typeof(*dma_info), info);
	int nr_pages = DIV_ROUND_UP((*_dma_info)->size, PAGE_SIZE);

	vunmap(dma_info->info.cpu_addr);

	if (dma_info->info.size) {
		/* Only free the pages if we allocated them,
		 * not the case for dmabuf import?
		 */
		if (!dma_info->dma_buf_internal)
			iommu_free_pages(allocator, dma_info->dma_addr,
					 dma_info->pages, nr_pages);

		devm_kfree(allocator->dev, dma_info->dma_addr);
		devm_kfree(allocator->dev, dma_info->pages);
	}

	memset(dma_info, 0, sizeof(*dma_info));
	devm_kfree(allocator->dev, dma_info);

	/* Clear the caller's pointer, so they aren't left with it dangling */
	*_dma_info = (struct ethosn_dma_info *)NULL;
}

static void iommu_sync_for_device(struct ethosn_dma_sub_allocator *allocator,
				  struct ethosn_dma_info *_dma_info)
{
	struct ethosn_dma_info_internal *dma_info =
		container_of(_dma_info, typeof(*dma_info), info);
	int nr_pages = ethosn_nr_sg_objects(dma_info);
	int i;

	if (dma_info->dma_buf_internal)
		dma_buf_end_cpu_access(dma_info->dma_buf_internal->dmabuf,
				       DMA_TO_DEVICE);
	else
		for (i = 0; i < nr_pages; ++i)
			dma_sync_single_for_device(allocator->dev,
						   dma_info->dma_addr[i],
						   ethosn_page_size(i, i + 1,
								    dma_info),
						   DMA_TO_DEVICE);
}

static void iommu_sync_for_cpu(struct ethosn_dma_sub_allocator *allocator,
			       struct ethosn_dma_info *_dma_info)
{
	struct ethosn_dma_info_internal *dma_info =
		container_of(_dma_info, typeof(*dma_info), info);
	int nr_scatter_pages = ethosn_nr_sg_objects(dma_info);
	int i;

	if (dma_info->dma_buf_internal)
		dma_buf_begin_cpu_access(dma_info->dma_buf_internal->dmabuf,
					 DMA_FROM_DEVICE);
	else
		for (i = 0; i < nr_scatter_pages; ++i)
			dma_sync_single_for_cpu(allocator->dev,
						dma_info->dma_addr[i],
						ethosn_page_size(i, i + 1,
								 dma_info),
						DMA_FROM_DEVICE);
}

static int iommu_mmap(struct ethosn_dma_sub_allocator *allocator,
		      struct vm_area_struct *const vma,
		      const struct ethosn_dma_info *const _dma_info)
{
	struct ethosn_dma_info_internal *dma_info =
		container_of(_dma_info, typeof(*dma_info), info);
	int nr_scatter_pages = ethosn_nr_sg_objects(dma_info);
	int i;

	for (i = 0; i < nr_scatter_pages; ++i) {
		unsigned long addr = vma->vm_start + ethosn_page_size(0, i,
								      dma_info);
		unsigned long pfn = page_to_pfn(dma_info->pages[i]);
		unsigned long size = ethosn_page_size(i, i + 1, dma_info);

		if (remap_pfn_range(vma, addr, pfn, size, vma->vm_page_prot))
			return -EAGAIN;
	}

	return 0;
}

static int iommu_stream_init(struct ethosn_allocator_internal *allocator,
			     enum ethosn_stream_type stream_type,
			     size_t bitmap_size)
{
	struct ethosn_iommu_domain *domain = &allocator->ethosn_iommu_domain;
	struct ethosn_iommu_stream *stream =
		iommu_get_stream(domain, stream_type);
	int nr_pages = DIV_ROUND_UP(IOMMU_ADDR_SIZE, PAGE_SIZE);
	int i, k, err;

	dev_dbg(allocator->allocator.dev,
		"%s: stream_type %u\n", __func__, stream_type);

	stream->bitmap =
		devm_kzalloc(allocator->allocator.dev, bitmap_size, GFP_KERNEL);
	if (!stream->bitmap)
		return -ENOMEM;

	stream->addr_base =
		iommu_get_addr_base(&allocator->allocator, stream_type);
	stream->bits = bitmap_size * BITS_PER_BYTE;
	spin_lock_init(&stream->lock);

	if (stream_type > ETHOSN_STREAM_COMMAND_STREAM)
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
			dev_err(allocator->allocator.dev,
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
	devm_kfree(allocator->allocator.dev, stream->bitmap);

	return -ENOMEM;
}

static void iommu_stream_deinit(struct ethosn_allocator_internal *allocator,
				enum ethosn_stream_type stream_type)
{
	struct ethosn_iommu_domain *domain = &allocator->ethosn_iommu_domain;
	struct ethosn_iommu_stream *stream =
		iommu_get_stream(domain, stream_type);
	int nr_pages = DIV_ROUND_UP(IOMMU_ADDR_SIZE, PAGE_SIZE);
	int i;

	dev_dbg(allocator->allocator.dev,
		"%s: stream_type %u\n", __func__, stream_type);

	if (!stream)
		return;

	/* Parent and children share the streams, make sure that it is not
	 * freed twice.
	 */
	if (stream->bitmap) {
		devm_kfree(allocator->allocator.dev,
			   stream->bitmap);
		stream->bitmap = NULL;
	}

	if (!stream->page)
		return;

	/* Unmap all the virtual space (see iommu_stream_init). */
	for (i = 0; i < nr_pages; ++i)
		iommu_unmap(domain->iommu_domain,
			    stream->addr_base + i * PAGE_SIZE,
			    PAGE_SIZE);

	__free_page(stream->page);
	stream->page = NULL;
}

static void iommu_allocator_destroy(struct ethosn_dma_sub_allocator *_allocator,
				    enum ethosn_stream_type stream_type)
{
	struct ethosn_allocator_internal *allocator;
	struct iommu_domain *domain;
	struct device *dev;

	if (!_allocator)
		return;

	allocator = container_of(_allocator, typeof(*allocator), allocator);
	domain = allocator->ethosn_iommu_domain.iommu_domain;
	dev = _allocator->dev;

	iommu_stream_deinit(allocator, stream_type);

	memset(allocator, 0, sizeof(struct ethosn_allocator_internal));
	devm_kfree(dev, allocator);

	ethosn_iommu_put_domain_for_dev(dev, domain);
}

struct ethosn_dma_sub_allocator *ethosn_dma_iommu_allocator_create(
	struct device *dev,
	enum ethosn_stream_type stream_type)
{
	static const struct ethosn_dma_allocator_ops ops = {
		.destroy         = iommu_allocator_destroy,
		.alloc           = iommu_alloc,
		.free            = iommu_free,
		.mmap            = iommu_mmap,
		.map             = iommu_iova_map,
		.import          = iommu_import,
		.release         = iommu_release,
		.unmap           = iommu_iova_unmap,
		.sync_for_device = iommu_sync_for_device,
		.sync_for_cpu    = iommu_sync_for_cpu,
		.get_addr_base   = iommu_get_addr_base,
		.get_addr_size   = iommu_get_addr_size
	};
	static const struct ethosn_dma_allocator_ops ops_no_iommu = {
		.destroy         = iommu_allocator_destroy,
		.alloc           = iommu_alloc,
		.free            = iommu_free,
		.mmap            = iommu_mmap,
		.import          = iommu_import,
		.release         = iommu_release,
		.sync_for_device = iommu_sync_for_device,
		.sync_for_cpu    = iommu_sync_for_cpu,
	};
	struct ethosn_allocator_internal *allocator;
	struct iommu_domain *domain = NULL;
	struct iommu_fwspec *fwspec = NULL;
	size_t bitmap_size;
	int ret;

	domain = ethosn_iommu_get_domain_for_dev(dev);

	allocator = devm_kzalloc(dev,
				 sizeof(struct ethosn_allocator_internal),
				 GFP_KERNEL);
	if (!allocator)
		return ERR_PTR(-ENOMEM);

	allocator->allocator.dev = dev;
	allocator->ethosn_iommu_domain.iommu_domain = domain;

	if (domain) {
		bitmap_size = BITS_TO_LONGS(IOMMU_ADDR_SIZE >> PAGE_SHIFT) *
			      sizeof(unsigned long);

		ret = iommu_stream_init(allocator, stream_type, bitmap_size);
		if (ret)
			goto err_stream;

		allocator->allocator.ops = &ops;
	} else {
		allocator->allocator.ops = &ops_no_iommu;
	}

	fwspec = dev_iommu_fwspec_get(allocator->allocator.dev);

	if (!fwspec || fwspec->num_ids != 1) {
		ret = -EINVAL;
		goto err_stream;
	}

	allocator->allocator.smmu_stream_id = fwspec->ids[0];

	dev_dbg(dev, "Created IOMMU DMA allocator. handle=%pK", allocator);

	return &allocator->allocator;

err_stream:
	devm_kfree(dev, allocator);

	return ERR_PTR(ret);
}
