//
// Copyright © 2021-2023 Arm Limited.
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

TileSizeCalculation CalculateTileSize(const HardwareCapabilities& caps,
                                      const TensorShape& inputTensorShape,
                                      const TensorShape& inputStripeShape,
                                      PackedBoundaryThickness packedBoundaryThickness,
                                      uint32_t numStripes,
                                      bool couldSourceBeFcaf)
{
    TileSizeCalculation result = { 0, 0, false };

    // Calculate the size needed for each slot. This is based on the space needed for one stripe,
    // but might need additional space for packed boundary data, and rounding because of FCAF.
    // If the tile could be decompressed from FCAF then we need
    // to make sure we have full FCAF cells available as the HW always writes to SRAM in full FCAF
    // cell size if the source is FCAF compressed (only in width and height though, channels is fine).
    // This is fine as cell shapes in W/H are mostly 8, apart from FCAF_WIDE which has 16 width,
    // which is the problematic one here.
    assert(GetWidth(g_FcafDeepCellShape) == 8 && GetHeight(g_FcafDeepCellShape) == 8 &&
           GetHeight(g_FcafWideCellShape) == 8);

    const TensorShape stripeShapeInclBoundary = {
        1,
        GetHeight(inputStripeShape) + packedBoundaryThickness.top + packedBoundaryThickness.bottom,
        GetWidth(inputStripeShape) + packedBoundaryThickness.left + packedBoundaryThickness.right,
        GetChannels(inputStripeShape),
    };

    bool couldSourceBeFcafWide = couldSourceBeFcaf && !packedBoundaryThickness.AnyNonZero() &&
                                 IsCompressionFormatCompatibleWithStripeShape(CompilerDataCompressedFormat::FCAF_WIDE,
                                                                              inputStripeShape, inputTensorShape);

    TensorShape stripeShapeRoundedUpFcaf = stripeShapeInclBoundary;
    if (couldSourceBeFcafWide && GetWidth(stripeShapeInclBoundary) % GetWidth(g_FcafWideCellShape) != 0)
    {
        // Before we round it up, check if this would lead to significantly higher SRAM usage.
        // In some cases, it is better to avoid increasing the tile size and instead forbid FCAF_WIDE being
        // used for this buffer. FCAF_DEEP may still be usable, depending on the context.
        // We've chosen an arbitrary threshold of 10% for this.
        uint32_t newWidth =
            utils::RoundUpToNearestMultiple(GetWidth(stripeShapeInclBoundary), GetWidth(g_FcafWideCellShape));
        if (static_cast<float>(newWidth) / static_cast<float>(GetWidth(stripeShapeInclBoundary)) < 1.10f)
        {
            stripeShapeRoundedUpFcaf[2] = newWidth;
        }
        else
        {
            result.forbidFcafWide = true;
            // So that the optimisation below can take advantage of knowing that we won't use FCAF_WIDE
            couldSourceBeFcafWide = false;
        }
    }

    result.slotSizeInBytes = TotalSizeBytes(stripeShapeRoundedUpFcaf);
    result.sizeInBytes     = result.slotSizeInBytes * numStripes;

    // If the tensor doesn't have many stripes in it, then it's possible that we would allocate
    // more space in the tile than will actually be used (e.g. tensor is 65 high, stripes are 64 high,
    // numStripesInTile = 2). We therefore clamp the tile size to avoid allocating too much.
    // We also need to account for FCAF here as above.

    // If packed boundary data is used then we can't do this optimisation, because boundary data is
    // always laid out afterwards and assumes the full stripe shape.
    if (packedBoundaryThickness.AnyNonZero())
    {
        return result;
    }

    uint32_t widthMultiple  = GetWidth(g_BrickGroupShape);
    uint32_t heightMultiple = GetHeight(g_BrickGroupShape);
    if (couldSourceBeFcafWide)
    {
        widthMultiple = std::max(widthMultiple, GetWidth(g_FcafWideCellShape));
    }

    const uint32_t inputTileSize = utils::MaxTileSize(inputTensorShape, caps, widthMultiple, heightMultiple);
    result.sizeInBytes           = std::min(inputTileSize, result.sizeInBytes);
    return result;
}

}    // namespace impl
}    // namespace support_library
}    // namespace ethosn
