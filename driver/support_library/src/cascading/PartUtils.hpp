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

CascadingBufferFormat GetFormat(Location location);
CascadingBufferFormat GetCascadingBufferFormatFromCompilerDataFormat(const CompilerDataFormat& format);

struct TileSizeCalculation
{
    uint32_t slotSizeInBytes;
    uint32_t sizeInBytes;
    bool forbidFcafWide;
};

TileSizeCalculation CalculateTileSize(const HardwareCapabilities& caps,
                                      const TensorShape& inputTensorShape,
                                      const TensorShape& inputStripeShape,
                                      command_stream::cascading::PackedBoundaryThickness packedBoundaryThickness,
                                      uint32_t numStripes,
                                      bool couldSourceBeFcaf);

}    // namespace impl
}    // namespace support_library
}    // namespace ethosn
