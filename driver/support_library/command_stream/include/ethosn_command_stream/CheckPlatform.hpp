//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <climits>
#include <cstdint>

namespace ethosn
{
namespace check_platform
{

struct Empty
{};

static_assert(CHAR_BIT == 8, "");

static_assert(sizeof(Empty) == 1, "");
static_assert(alignof(Empty) == 1, "");

static_assert(sizeof(char) == 1, "");
static_assert(alignof(char) == 1, "");

static_assert(sizeof(unsigned char) == 1, "");
static_assert(alignof(unsigned char) == 1, "");

static_assert(sizeof(bool) == 1, "");
static_assert(alignof(bool) == 1, "");

static_assert(sizeof(int8_t) == 1, "");
static_assert(alignof(int8_t) == 1, "");

static_assert(sizeof(uint8_t) == 1, "");
static_assert(alignof(uint8_t) == 1, "");

static_assert(sizeof(int16_t) == 2, "");
static_assert(alignof(int16_t) == 2, "");

static_assert(sizeof(uint16_t) == 2, "");
static_assert(alignof(uint16_t) == 2, "");

static_assert(sizeof(int32_t) == 4, "");
static_assert(alignof(int32_t) == 4, "");

static_assert(sizeof(uint32_t) == 4, "");
static_assert(alignof(uint32_t) == 4, "");

}    // namespace check_platform
}    // namespace ethosn
