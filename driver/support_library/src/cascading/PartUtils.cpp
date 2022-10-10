//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "PartUtils.hpp"

#include "../Utils.hpp"

using namespace ethosn::support_library::utils;

namespace ethosn
{
namespace support_library
{

namespace impl
{

CascadingBufferFormat GetFormat(Location location)
{
    switch (location)
    {
        case Location::Dram:
            return CascadingBufferFormat::NHWC;
        case Location::PleInputSram:
        case Location::Sram:
            return CascadingBufferFormat::NHWCB;
        case Location::VirtualSram:
            return CascadingBufferFormat::NHWC;
        default:
            throw NotSupportedException("Unknown location");
    }
}

CascadingBufferFormat GetCascadingBufferFormatFromCompilerDataFormat(const CompilerDataFormat& format)
{
    switch (format)
    {
        case (CompilerDataFormat::NHWC):
            return CascadingBufferFormat::NHWC;
        case (CompilerDataFormat::NCHW):
            return CascadingBufferFormat::NCHW;
        case (CompilerDataFormat::NHWCB):
            return CascadingBufferFormat::NHWCB;
        case (CompilerDataFormat::WEIGHT):
            return CascadingBufferFormat::WEIGHT;
        default:
        {
            std::string error = "In " + std::string(ETHOSN_FUNCTION_SIGNATURE) + ": value " +
                                std::to_string(static_cast<uint32_t>(format)) + " is not valid";
            throw NotSupportedException(error.c_str());
        }
    }
}

std::pair<uint32_t, uint32_t>
    CalculateTileSize(const HardwareCapabilities& caps,
                      const TensorShape& inputTensorShape,
                      const TensorShape& inputStripeShape,
                      command_stream::cascading::PackedBoundaryThickness packedBoundaryThickness,
                      uint32_t numStripes,
                      bool couldSourceBeFcaf)
{
    const TensorShape stripeShapeInclBoundary = {
        1,
        GetHeight(inputStripeShape) + packedBoundaryThickness.top + packedBoundaryThickness.bottom,
        GetWidth(inputStripeShape) + packedBoundaryThickness.left + packedBoundaryThickness.right,
        GetChannels(inputStripeShape),
    };
    const uint32_t slotSize      = TotalSizeBytes(stripeShapeInclBoundary);
    uint32_t inputFullStripeSize = slotSize * numStripes;

    // Input tile clamp is only allowed if the plan is not de-compressed from FCAF
    // or the stripe shape is not multiple of any type of FCAF cell.
    // Note if the HW always writes to SRAM in full FCAF cell size if the source FCAF compressed.
    const bool inputTileClamp =
        (!couldSourceBeFcaf ||
         (!IsCompressionFormatCompatibleWithStripeShape(CompilerDataCompressedFormat::FCAF_DEEP, inputStripeShape) &&
          !IsCompressionFormatCompatibleWithStripeShape(CompilerDataCompressedFormat::FCAF_WIDE, inputStripeShape))) &&
        !AnyPackedBoundaryData(packedBoundaryThickness);

    if (inputTileClamp)
    {
        const uint32_t inputTileSize = utils::MaxTileSize(inputTensorShape, caps);
        return { slotSize, std::min(inputTileSize, inputFullStripeSize) };
    }
    else
    {
        return { slotSize, inputFullStripeSize };
    }
}

}    // namespace impl
}    // namespace support_library
}    // namespace ethosn
