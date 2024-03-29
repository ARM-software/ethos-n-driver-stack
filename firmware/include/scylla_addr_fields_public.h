//
// Copyright © 2019 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

/* This header file is NOT automatically generated */

#pragma once

#define REGION_REGISTERS (0x4 >> 1)
#define REGION_EXT_RAM0  (0x6 >> 1)
#define REGION_EXT_RAM1  (0x8 >> 1)

#define REGION_SHIFT 29
#define REGION_MASK 0x3

#define BROADCAST_SHIFT 28
#define BROADCAST_MASK 0x1

#define CE_SHIFT 20
#define CE_MASK 0xFF

#define REGPAGE_SHIFT 16
#define REGPAGE_MASK 0xF

#define REGOFFSET_SHIFT 0
#define REGOFFSET_MASK 0xFFFF

/* Register page offsets */
#define DL1_RP            0x1

/* Compose a Scylla register address from the bit components */
#define SCYLLA_REG(broadcast, ce, page, offset)             \
	(((REGION_REGISTERS & REGION_MASK) << REGION_SHIFT) |  \
	(((broadcast) & BROADCAST_MASK) << BROADCAST_SHIFT) |  \
	(((ce) & CE_MASK) << CE_SHIFT)                      |  \
	(((page) & REGPAGE_MASK) << REGPAGE_SHIFT)          |  \
	(((offset) & REGOFFSET_MASK) << REGOFFSET_SHIFT))

#define TOP_REG(page, offset) SCYLLA_REG(1, 0, (page), (offset))
