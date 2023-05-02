//
// Copyright © 2018-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Sizes.hpp"
#include "hw.h"

// ====================================================
// Defines for block relationships
// ====================================================

namespace    // Internal linkage
{

// A block is NxM groups, depending on the block size chosen.
// The build system is expected to define BLOCK_WIDTH_IN_ELEMENTS, BLOCK_HEIGHT_IN_ELEMENTS and BLOCK_MULTIPLIER.
static_assert((BLOCK_WIDTH_IN_ELEMENTS % ELEMENTS_PER_GROUP_1D) == 0,
              "Block width must be a multiple of the group size");
static_assert((BLOCK_HEIGHT_IN_ELEMENTS % ELEMENTS_PER_GROUP_1D) == 0,
              "Block height must be a multiple of the group size");
constexpr unsigned int GROUPS_PER_BLOCK_X  = BLOCK_WIDTH_IN_ELEMENTS / ELEMENTS_PER_GROUP_1D;
constexpr unsigned int GROUPS_PER_BLOCK_Y  = BLOCK_HEIGHT_IN_ELEMENTS / ELEMENTS_PER_GROUP_1D;
constexpr unsigned int GROUPS_PER_BLOCK    = GROUPS_PER_BLOCK_X * GROUPS_PER_BLOCK_Y;
constexpr unsigned int PATCHES_PER_BLOCK_X = PATCHES_PER_GROUP_1D * GROUPS_PER_BLOCK_X;
constexpr unsigned int PATCHES_PER_BLOCK_Y = PATCHES_PER_GROUP_1D * GROUPS_PER_BLOCK_Y;
constexpr unsigned int REGISTERS_PER_BLOCK = GROUPS_PER_BLOCK * REGISTERS_PER_GROUP;
constexpr unsigned int WORDS_PER_BLOCK     = WORDS_PER_REGISTER * REGISTERS_PER_BLOCK;

constexpr unsigned k_BlockMultiplier = BLOCK_MULTIPLIER;

static_assert((k_BlockMultiplier == 1U) || (PATCHES_PER_BLOCK_Y == 2U),
              "Block multiplier can only be >1 if block height is equal to 8");

using BlockSize = sizes::BlockSize<PATCHES_PER_BLOCK_X * k_BlockMultiplier, PATCHES_PER_BLOCK_Y>;

}    // namespace
