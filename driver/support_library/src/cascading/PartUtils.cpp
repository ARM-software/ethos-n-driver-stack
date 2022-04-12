//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../Utils.hpp"
#include "PartUtils.hpp"

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
            throw NotSupportedException("Unkwnown location");
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

uint32_t CalculateBufferSize(const TensorShape& shape, CascadingBufferFormat f)
{
    switch (f)
    {
        case CascadingBufferFormat::NHWCB:
            return utils::TotalSizeBytesNHWCB(shape);
        case CascadingBufferFormat::NHWC:
        case CascadingBufferFormat::WEIGHT:
            return utils::TotalSizeBytes(shape);
        default:
            assert(false);
            return 0;
    }
}

uint32_t CalculateSizeInBytes(const TensorShape& shape)
{
    return utils::TotalSizeBytesNHWCB(shape);
}

uint32_t CalculateTileSize(const HardwareCapabilities& caps,
                           const TensorShape& tensorShape,
                           const TensorShape& stripeShape,
                           uint32_t numStripes)
{
    // Restrict the tile max size to be the full tensor so we don't waste space when we have partial stripes
    const uint32_t inputFullStripeSize = numStripes * utils::TotalSizeBytesNHWCB(stripeShape);
    const uint32_t inputTileSize       = utils::MaxTileSize(tensorShape, caps);

    return std::min(inputTileSize, inputFullStripeSize);
}

uint32_t CalculateTileSize(Node* node,
                           const HardwareCapabilities& caps,
                           const TensorShape& inputTensorShape,
                           const TensorShape& inputStripeShape,
                           const TensorShape& outputStripeShape,
                           uint32_t numStripes)
{
    using namespace ethosn::support_library::utils;

    uint32_t inputFullStripeSize;

    if (IsObjectOfType<MceOperationNode>(node))
    {
        auto mceNode                    = GetObjectAs<MceOperationNode>(node);
        auto kernelHeight               = mceNode->GetWeightsInfo().m_Dimensions[0];
        auto padTop                     = mceNode->GetPadTop();
        const uint32_t brickGroupHeight = GetHeight(caps.GetBrickGroupShape());

        // Work out the tile sizes by deciding how many stripes we want in each tile
        const NeedBoundary needBoundaryY = ethosn::support_library::utils::GetBoundaryRequirements(
            padTop, GetHeight(inputTensorShape), GetHeight(inputStripeShape), GetHeight(outputStripeShape),
            kernelHeight);

        const bool isStreamingWidth = GetWidth(inputStripeShape) < GetWidth(inputTensorShape);

        const bool needsBoundarySlots = (needBoundaryY.m_Before || needBoundaryY.m_After) && (isStreamingWidth);
        const uint32_t inputStripeXZ  = GetWidth(inputStripeShape) * GetChannels(inputStripeShape);

        const uint32_t boundarySlotSize = needsBoundarySlots ? (brickGroupHeight * inputStripeXZ) : 0U;
        const uint32_t defaultSlotSize  = TotalSizeBytes(inputStripeShape);

        // We need the boundary slots both on the top and bottom of the stripe
        const uint32_t totalSlotSize = (2U * boundarySlotSize) + defaultSlotSize;

        inputFullStripeSize = totalSlotSize * numStripes;
    }
    else
    {
        // Restrict the tile max size to be the full tensor so we don't waste space when we have partial stripes
        inputFullStripeSize = numStripes * CalculateSizeInBytes(inputStripeShape);
    }
    const uint32_t inputTileSize = utils::MaxTileSize(inputTensorShape, caps);

    return std::min(inputTileSize, inputFullStripeSize);
}

}    // namespace impl
}    // namespace support_library
}    // namespace ethosn
