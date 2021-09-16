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

#ifndef _ETHOSN_BACKPORT_H_
#define _ETHOSN_BACKPORT_H_

#include <linux/dma-mapping.h>
#include <linux/fs.h>
#include <linux/iommu.h>
#include <linux/eventpoll.h>
#include <linux/version.h>

#if !defined(EPOLLERR)
#define EPOLLERR        POLLERR
#define EPOLLIN         POLLIN
#define EPOLLHUP        POLLHUP
#define EPOLLRDNORM     POLLRDNORM
#endif

#if (KERNEL_VERSION(4, 16, 0) > LINUX_VERSION_CODE)
typedef unsigned __bitwise __poll_t;
#endif

#if (KERNEL_VERSION(5, 0, 0) > LINUX_VERSION_CODE)
static inline struct iommu_fwspec *dev_iommu_fwspec_get(struct device *dev)
{
	return dev->iommu_fwspec;
}

#endif

#if (KERNEL_VERSION(5, 10, 0) > LINUX_VERSION_CODE)
struct page *dma_alloc_pages(struct device *dev,
			     size_t size,
			     dma_addr_t *dma_handle,
			     enum dma_data_direction dir,
			     gfp_t gfp);

void dma_free_pages(struct device *dev,
		    size_t size,
		    struct page *page,
		    dma_addr_t dma_handle,
		    enum dma_data_direction dir);
#endif
void ethosn_iommu_put_domain_for_dev(struct device *dev,
				     struct iommu_domain *domain);

struct iommu_domain *ethosn_iommu_get_domain_for_dev(struct device *dev);

int ethosn_bitmap_find_next_zero_area(struct device *dev,
				      void **bitmap,
				      size_t bits,
				      int nr_pages,
				      unsigned long *start);

#endif /* _ETHOSN_BACKPORT_H_ */
