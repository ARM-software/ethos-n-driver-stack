//
// Copyright © 2021-2022 Arm Limited.
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
std::pair<uint32_t, uint32_t>
    CalculateTileSize(const HardwareCapabilities& caps,
                      const TensorShape& inputTensorShape,
                      const TensorShape& inputStripeShape,
                      command_stream::cascading::PackedBoundaryThickness packedBoundaryThickness,
                      uint32_t numStripes,
                      bool couldSourceBeFcaf);

}    // namespace impl
}    // namespace support_library
}    // namespace ethosn
