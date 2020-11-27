/*
 *
 * (C) COPYRIGHT 2018-2019 Arm Limited. All rights reserved.
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

#ifndef _ETHOSN_LOG_H_
#define _ETHOSN_LOG_H_

#include "ethosn_device.h"

struct ethosn_core;

/**
 * ethosn_log_init - Initialize log object.
 * @core:	Ethos-N core.
 *
 * Return: 0 on success, else error code.
 */
int ethosn_log_init(struct ethosn_core *core);

/**
 * ethosn_log_deinit - Deinitialize log object.
 * @core:	Ethos-N core.
 */
void ethosn_log_deinit(struct ethosn_core *core);

/**
 * ethosn_log_text() - Write text message to log.
 * @core:	Ethos-N core.
 * @msg:	Message.
 *
 * Return: 0 on success, else error code.
 */
int ethosn_log_text(struct ethosn_core *core,
		    const char *msg);

/**
 * ethosn_log_uapi() - Write UAPI message to log.
 * @core:	Ethos-N core.
 * @ioctl:	IOCTL.
 * @data:	Pointer to data.
 * @length:	Length of data.
 *
 * Return: 0 on succes, else error code.
 */
int ethosn_log_uapi(struct ethosn_core *core,
		    uint32_t ioctl,
		    void *data,
		    size_t length);

/**
 * ethosn_log_firmware() - Write firmware message to log.
 * @core:	Ethos-N core.
 * @direction:	0=host->firmware, 1=host<-firmware.
 * @header:	Firmware interface message header.
 * @data:	Data following message header.
 *
 * Return: 0 on success, else error code.
 */
int ethosn_log_firmware(struct ethosn_core *core,
			enum ethosn_log_firmware_direction direction,
			struct ethosn_message_header *header,
			void *data);

#endif /* _ETHOSN_LOG_H_ */
