//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Part.hpp"
#include "Plan.hpp"
#include "StripeHelper.hpp"

namespace ethosn
{
namespace support_library
{
namespace impl
{

BufferFormat GetFormat(Location location);
BufferFormat GetBufferFormatFromCompilerDataFormat(const CompilerDataFormat& format);

struct TileSizeCalculation
{
    uint32_t slotSizeInBytes;
    uint32_t sizeInBytes;
    bool forbidFcafWide;
};

TileSizeCalculation CalculateTileSize(const HardwareCapabilities& caps,
                                      const TensorShape& inputTensorShape,
                                      const TensorShape& inputStripeShape,
                                      PackedBoundaryThickness packedBoundaryThickness,
                                      uint32_t numStripesInTile,
                                      bool couldSourceBeFcaf);

}    // namespace impl
}    // namespace support_library
}    // namespace ethosn
