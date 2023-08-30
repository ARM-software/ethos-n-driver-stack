//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ethosn_command_stream/CommandStream.hpp>

#include <cassert>
#include <cstdint>

namespace ethosn
{
namespace support_library
{

/// Slot info for data in SRAM
struct Tile
{
    uint32_t baseAddr;
    uint16_t numSlots;
    uint32_t slotSize;
};

struct TensorSize
{
    uint32_t height;
    uint32_t width;
    uint32_t channels;
};

/// Calculate SRAM address from SRAM Tile
inline uint32_t SramAddr(Tile tile, uint32_t stripeId)
{
    return static_cast<uint32_t>(tile.baseAddr + ((static_cast<uint32_t>(stripeId) % tile.numSlots) * tile.slotSize));
}

}    // namespace support_library
}    // namespace ethosn
