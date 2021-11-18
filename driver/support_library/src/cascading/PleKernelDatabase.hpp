//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Plan.hpp"

namespace ethosn
{
namespace support_library
{
using namespace command_stream;
namespace plelib
{
namespace impl
{

enum PleKernelIdBlockSize
{
    _8X8,
    _8X16,
    _8X32,
    _16X8,
    _16X16,
    _32X8,
    NUM_BLOCK_SIZES
};

enum PleKernelIdBlockMultiplier
{
    _1,
    _2,
    _4,
    NUM_BLOCK_MS
};

enum PleKernelIdDataType
{
    S8,
    U8,
    NUM_DATA_TYPES
};

constexpr uint32_t PleOpIndex(PleOperation op)
{
    return (static_cast<uint32_t>(op));
}

struct alignas(alignof(uint16_t)) PleBlkSizeKey
{
    // For performance reason, declare the field
    // in the order they are used in operator uint16_t()
    uint8_t blockHeight;
    uint8_t blockWidth;

    constexpr operator uint16_t() const
    {
        // Endian-independent code to convert this struct to a single uint16.
        // The order of those field is important.
        // See also the comment in the definition of 'class KernelId' in gen_plelib.py
        return static_cast<uint16_t>((static_cast<uint32_t>(blockHeight) << 0) |
                                     (static_cast<uint32_t>(blockWidth) << 8));
    }
};

using PleBlkSizeMap = std::map<PleBlkSizeKey, PleKernelIdBlockSize>;

using PleKernelDataTypeMap = std::map<bool, PleKernelIdDataType>;

using PlekernelBlkMulMap = std::map<uint32_t, PleKernelIdBlockMultiplier>;

struct PleKernelIdDatabase
{
    PleKernelId data[PleOpIndex(PleOperation::NUM_OPS)][NUM_DATA_TYPES][NUM_BLOCK_SIZES][NUM_BLOCK_MS];
};

}    // namespace impl

PleKernelId FindPleKernelIdFromDatabase(BlockConfig blockConfig,
                                        uint32_t stripeWidth,
                                        ethosn::command_stream::DataType outputDataType,
                                        PleOperation op);

}    // namespace plelib
}    // namespace support_library
}    // namespace ethosn
