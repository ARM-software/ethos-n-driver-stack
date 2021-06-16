/*
 *
 * (C) COPYRIGHT 2020-2021 Arm Limited.
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

#ifndef _ETHOSN_CORE_H_
#define _ETHOSN_CORE_H_

#include <linux/types.h>

#ifdef CONFIG_PM
#define ETHOSN_AUTOSUSPEND_DELAY_MS 500
#else
#define ETHOSN_AUTOSUSPEND_DELAY_MS 0
#endif  /* CONFIG_PM */

int ethosn_core_platform_driver_register(void);

void ethosn_core_platform_driver_unregister(void);

int ethosn_get_autosuspend_delay(void);

#endif /* _ETHOSN_CORE_H_ */
