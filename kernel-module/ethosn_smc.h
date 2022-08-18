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

#ifndef _ETHOSN_SMC_H_
#define _ETHOSN_SMC_H_

#include "ethosn_device.h"

#include <linux/arm-smccc.h>
#include <linux/types.h>

/**
 * ethosn_smc_version_check() - Check SiP service version compatibility
 * @device:	Pointer to the struct device on which to log the error if any.
 *
 * Checks that the Arm Ethos-N NPU SiP service is available and that it is
 * running a compatible version.
 *
 * Return: 0 on success, else error code.
 */
int ethosn_smc_version_check(const struct device *dev);

/**
 * ethosn_smc_is_secure() - Call SiP service to get the NPU's secure status
 * @device:	Pointer to the struct device on which to log the error if any.
 * @core_addr:	Address to Ethos-N core.
 *
 * Return: 0 if unsecure, 1 if secure or negative error code on failure.
 */
int ethosn_smc_is_secure(const struct device *dev,
			 phys_addr_t core_addr);

/**
 * ethosn_smc_core_reset() - Call SiP service to reset a NPU core
 * @device:		Pointer to the struct device on which to log the error
 *			if any.
 * @core_addr:		Address to Ethos-N core.
 * @asset_alloc_idx:	Index of the asset allocator to use.
 * @hard_reset:		Indicates if a hard or soft reset should be performed.
 *
 * Return: 0 on success, else error code.
 */
int ethosn_smc_core_reset(const struct device *dev,
			  phys_addr_t core_addr,
			  uint32_t asset_alloc_idx,
			  int hard_reset);

#endif /* _ETHOSN_SMC_H_ */
