/*
 *
 * (C) COPYRIGHT 2021 Arm Limited.
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

#include "ethosn_backport.h"

#if (KERNEL_VERSION(5, 10, 0) > LINUX_VERSION_CODE)
struct page *dma_alloc_pages(struct device *dev,
			     size_t size,
			     dma_addr_t *dma_handle,
			     enum dma_data_direction dir,
			     gfp_t gfp)
{
	struct page *page;

	if (size != PAGE_SIZE) {
		dev_dbg(dev,
			"Backport implementation only supports size equal to PAGE_SIZE=%lu\n",
			PAGE_SIZE);

		return NULL;
	}

	page = alloc_page(gfp);
	if (!page)
		return NULL;

	*dma_handle = dma_map_page(dev, page, 0,
				   size,
				   dir);

	return page;
}

void dma_free_pages(struct device *dev,
		    size_t size,
		    struct page *page,
		    dma_addr_t dma_handle,
		    enum dma_data_direction dir)
{
	if (dma_handle)
		dma_unmap_page(dev, dma_handle,
			       size, dir);

	if (page)
		__free_page(page);
}

#endif
