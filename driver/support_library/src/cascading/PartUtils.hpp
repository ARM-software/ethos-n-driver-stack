//
// Copyright © 2021 Arm Limited.
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
uint32_t CalculateBufferSize(const TensorShape& shape, CascadingBufferFormat f);
uint32_t CalculateSizeInBytes(const TensorShape& shape);
uint32_t CalculateTileSize(const HardwareCapabilities& caps,
                           const TensorShape& tensorShape,
                           const TensorShape& stripeShape,
                           uint32_t numStripes);
uint32_t CalculateTileSize(Node* node,
                           const HardwareCapabilities& caps,
                           const TensorShape& inputTensorShape,
                           const TensorShape& inputStripeShape,
                           const TensorShape& outputStripeShape,
                           uint32_t numStripes);

}    // namespace impl
}    // namespace support_library
}    // namespace ethosn
