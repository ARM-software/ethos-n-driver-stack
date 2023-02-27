//
// Copyright © 2022-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Plan.hpp"
#include "SubmapFilter.hpp"
#include "Utils.hpp"

#define ETHOSN_ASSERT_MSG(cond, msg) assert(cond)
#include "ethosn_utils/NumericCast.hpp"

using namespace ethosn::command_stream::cascading;

namespace ethosn
{
namespace support_library
{
namespace cascading_compiler
{

namespace CommonUtils
{
inline void SetTileInfoForBuffer(const HardwareCapabilities& hwCap, Tile& tile, const SramBuffer* const buffer)
{
    assert(buffer->m_Format == CascadingBufferFormat::NHWCB || buffer->m_Format == CascadingBufferFormat::WEIGHT);

    tile.baseAddr = ethosn::utils::NumericCast<uint32_t>(buffer->m_Offset.value());
    tile.numSlots = ethosn::utils::NumericCast<uint16_t>(buffer->m_NumStripes);
    tile.slotSize =
        ethosn::utils::NumericCast<uint32_t>(utils::DivRoundUp(buffer->m_SlotSizeInBytes, hwCap.GetNumberOfSrams()));
}

inline uint16_t CalculateEdgeSize(uint32_t tensorSize, uint32_t defaultStripeSize)
{
    uint16_t edge = ethosn::utils::NumericCast<uint16_t>(tensorSize % defaultStripeSize);
    return edge != 0 ? edge : ethosn::utils::NumericCast<uint16_t>(defaultStripeSize);
}

}    // namespace CommonUtils

namespace StreamersUtils
{
inline void SetBufferDataType(FmSData& streamerData, const CascadingBufferFormat bufferFormat)
{
    switch (bufferFormat)
    {
        case CascadingBufferFormat::NHWC:
            streamerData.dataType = FmsDataType::NHWC;
            break;
        case CascadingBufferFormat::NHWCB:
            streamerData.dataType = FmsDataType::NHWCB;
            break;
        case CascadingBufferFormat::FCAF_DEEP:
            streamerData.dataType = FmsDataType::FCAF_DEEP;
            break;
        case CascadingBufferFormat::FCAF_WIDE:
            streamerData.dataType = FmsDataType::FCAF_WIDE;
            break;
        default:
            assert(false);
    }
}

inline void SetStripeHeightInfo(FmSData& streamerData, const TensorShape& tensorShape, const TensorShape& stripeShape)
{
    uint16_t tensorHeight = ethosn::utils::NumericCast<uint16_t>(utils::GetHeight(tensorShape));
    uint16_t stripeHeight = ethosn::utils::NumericCast<uint16_t>(utils::GetHeight(stripeShape));

    assert(stripeHeight != 0);

    streamerData.numStripes.height =
        ethosn::utils::NumericCast<uint16_t>(utils::GetNumStripesH(tensorShape, stripeShape));

    streamerData.dfltStripeSize.height = stripeHeight;

    streamerData.edgeStripeSize.height = CommonUtils::CalculateEdgeSize(tensorHeight, stripeHeight);
    if (streamerData.dataType != FmsDataType::NHWC)
    {
        // Note that we don't round up to the cell shape for FCAF, only the brick group shape.
        // This is because FCAF transfers work fine with partial cells, and we need to keep
        // this stripe shape consistent with the PLE's interepretation of stripe layout, which rounds
        // to the brick group shape only.
        streamerData.edgeStripeSize.height = ethosn::utils::NumericCast<uint16_t>(utils::RoundUpToNearestMultiple(
            static_cast<uint32_t>(streamerData.edgeStripeSize.height), utils::GetHeight(g_BrickGroupShape)));
    }
}

inline void SetStripeWidthInfo(FmSData& streamerData, const TensorShape& tensorShape, const TensorShape& stripeShape)
{
    uint16_t tensorWidth = ethosn::utils::NumericCast<uint16_t>(utils::GetWidth(tensorShape));
    uint16_t stripeWidth = ethosn::utils::NumericCast<uint16_t>(utils::GetWidth(stripeShape));

    assert(stripeWidth != 0);

    streamerData.numStripes.width =
        ethosn::utils::NumericCast<uint16_t>(utils::GetNumStripesW(tensorShape, stripeShape));

    streamerData.dfltStripeSize.width = stripeWidth;

    streamerData.edgeStripeSize.width = CommonUtils::CalculateEdgeSize(tensorWidth, stripeWidth);
    // Note that we don't round up to the cell shape for FCAF, only the brick group shape.
    // This is because FCAF transfers work fine with partial cells, and we need to keep
    // this stripe shape consistent with the PLE's interepretation of stripe layout, which rounds
    // to the brick group shape only.
    if (streamerData.dataType != FmsDataType::NHWC)
    {
        streamerData.edgeStripeSize.width = ethosn::utils::NumericCast<uint16_t>(utils::RoundUpToNearestMultiple(
            static_cast<uint32_t>(streamerData.edgeStripeSize.width), utils::GetWidth(g_BrickGroupShape)));
    }
}

inline void SetStripeChannelsInfo(FmSData& streamerData,
                                  const TensorShape& tensorShape,
                                  const TensorShape& stripeShape,
                                  const TensorShape& supertensorOffset,
                                  const TensorShape& supertensorShape)
{
    uint16_t tensorChannels = ethosn::utils::NumericCast<uint16_t>(utils::GetChannels(tensorShape));
    uint16_t stripeChannels = ethosn::utils::NumericCast<uint16_t>(utils::GetChannels(stripeShape));

    assert(stripeChannels != 0);

    streamerData.numStripes.channels =
        ethosn::utils::NumericCast<uint16_t>(utils::GetNumStripesC(tensorShape, stripeShape));

    streamerData.dfltStripeSize.channels = stripeChannels;

    streamerData.edgeStripeSize.channels = CommonUtils::CalculateEdgeSize(tensorChannels, stripeChannels);

    if (streamerData.dataType == FmsDataType::FCAF_DEEP || streamerData.dataType == FmsDataType::FCAF_WIDE)
    {
        // Because FCAF cells are compressed, they must be read and written with the DMA configured with the same
        // number of channels. For example, writing a cell with full-depth and then attempting to read only one
        // channel from it would be an error. Normally, partial depth cells would only occur at the edge of a tensor
        // and this is fine because it will be both read and written with the same partial depth. However in some cases,
        // for example Split or Concat with the padding channels optimisation, we will read or write partial depth cells
        // partway through a (super)tensor, and this could cause problems. To avoid this, we always round up the stripe
        // size to a full cell when we're partway through a (super)tensor.
        const uint32_t channelEnd             = utils::GetChannels(supertensorOffset) + utils::GetChannels(tensorShape);
        const bool isEndOfSupertensorChannels = channelEnd == utils::GetChannels(supertensorShape);
        if (!isEndOfSupertensorChannels)
        {
            const uint32_t cellDepth = streamerData.dataType == FmsDataType::FCAF_DEEP
                                           ? utils::GetChannels(g_FcafDeepCellShape)
                                           : utils::GetChannels(g_FcafWideCellShape);
            streamerData.edgeStripeSize.channels = ethosn::utils::NumericCast<uint16_t>(utils::RoundUpToNearestMultiple(
                static_cast<uint32_t>(streamerData.edgeStripeSize.channels), cellDepth));
        }
    }
}

inline void SetSuperTensorSizeInCells(FmSData& streamerData,
                                      const TensorShape& tensorShape,
                                      const CascadingBufferFormat bufferFormat)
{
    uint16_t cellWidth = 0;
    uint16_t cellDepth = 0;

    switch (bufferFormat)
    {
        case CascadingBufferFormat::NHWC:
            cellWidth = 1;
            cellDepth = 1;
            break;
        case CascadingBufferFormat::NHWCB:
            cellWidth = 8;
            cellDepth = 16;
            break;
        case CascadingBufferFormat::FCAF_DEEP:
            cellWidth = 8;
            cellDepth = 32;
            break;
        case CascadingBufferFormat::FCAF_WIDE:
            cellWidth = 16;
            cellDepth = 16;
            break;
        default:
            assert(false);
    }

    streamerData.supertensorSizeInCells.width =
        ethosn::utils::NumericCast<uint16_t>(utils::DivRoundUp(tensorShape[2], cellWidth));
    streamerData.supertensorSizeInCells.channels =
        ethosn::utils::NumericCast<uint16_t>(utils::DivRoundUp(tensorShape[3], cellDepth));
}

inline void SetStripeIdStrides(FmSData& streamerData, TraversalOrder traversalOrder)
{
    if (traversalOrder == TraversalOrder::Xyz)
    {
        // Width is one because we are traversing along X first.
        streamerData.stripeIdStrides.width = 1;
        // We get to the next row after we have done "width" number of stripes
        streamerData.stripeIdStrides.height = streamerData.numStripes.width;
        // We get to the next plane after we have done "width * height" number of stripes
        streamerData.stripeIdStrides.channels =
            ethosn::utils::NumericCast<uint16_t>(streamerData.numStripes.width * streamerData.numStripes.height);
    }
    else if (traversalOrder == TraversalOrder::Zxy)
    {
        // Channels is one because we are traversing along Z first.
        streamerData.stripeIdStrides.channels = 1U;
        // We get to the next column after we have done "channels" number of stripes
        streamerData.stripeIdStrides.width = streamerData.numStripes.channels;
        // We get to the next row after we have done "depth x width" number of stripes
        streamerData.stripeIdStrides.height =
            ethosn::utils::NumericCast<uint16_t>(streamerData.numStripes.width * streamerData.numStripes.channels);
    }
    else
    {
        assert(false);
    }
}

}    // namespace StreamersUtils

namespace MceSUtils
{

inline void
    SetMcesOfmHeightStripeInfo(MceS& mceSchedulerData, const TensorShape& ofmShape, const TensorShape& ofmStripeShape)
{
    uint16_t ofmHeight       = ethosn::utils::NumericCast<uint16_t>(utils::GetHeight(ofmShape));
    uint16_t ofmStripeHeight = ethosn::utils::NumericCast<uint16_t>(utils::GetHeight(ofmStripeShape));

    assert(ofmStripeHeight != 0);

    mceSchedulerData.numStripes.ofmHeight =
        ethosn::utils::NumericCast<uint16_t>(utils::GetNumStripesH(ofmShape, ofmStripeShape));

    mceSchedulerData.dfltStripeSize.ofmHeight = ofmStripeHeight;

    mceSchedulerData.edgeStripeSize.ofmHeight = CommonUtils::CalculateEdgeSize(ofmHeight, ofmStripeHeight);
}

inline void
    SetMcesOfmWidthStripeInfo(MceS& mceSchedulerData, const TensorShape& ofmShape, const TensorShape& ofmStripeShape)
{
    uint16_t ofmWidth       = ethosn::utils::NumericCast<uint16_t>(utils::GetWidth(ofmShape));
    uint16_t ofmStripeWidth = ethosn::utils::NumericCast<uint16_t>(utils::GetWidth(ofmStripeShape));

    assert(ofmStripeWidth != 0);

    mceSchedulerData.numStripes.ofmWidth =
        ethosn::utils::NumericCast<uint16_t>(utils::GetNumStripesW(ofmShape, ofmStripeShape));

    mceSchedulerData.dfltStripeSize.ofmWidth = ofmStripeWidth;

    mceSchedulerData.edgeStripeSize.ofmWidth = CommonUtils::CalculateEdgeSize(ofmWidth, ofmStripeWidth);
}

inline void
    SetMcesOfmChannelsStripeInfo(MceS& mceSchedulerData, const TensorShape& ofmShape, const TensorShape& ofmStripeShape)
{
    uint16_t ofmChannels       = ethosn::utils::NumericCast<uint16_t>(utils::GetChannels(ofmShape));
    uint16_t ofmStripeChannels = ethosn::utils::NumericCast<uint16_t>(utils::GetChannels(ofmStripeShape));

    assert(ofmStripeChannels != 0);

    mceSchedulerData.numStripes.ofmChannels =
        ethosn::utils::NumericCast<uint16_t>(utils::GetNumStripesC(ofmShape, ofmStripeShape));

    mceSchedulerData.dfltStripeSize.ofmChannels = ofmStripeChannels;

    mceSchedulerData.edgeStripeSize.ofmChannels = CommonUtils::CalculateEdgeSize(ofmChannels, ofmStripeChannels);
}

inline void
    SetMcesIfmChannelsStripeInfo(MceS& mceSchedulerData, const TensorShape& ifmShape, const TensorShape& ifmStripeShape)
{
    uint16_t ifmChannels       = ethosn::utils::NumericCast<uint16_t>(utils::GetChannels(ifmShape));
    uint16_t ifmStripeChannels = ethosn::utils::NumericCast<uint16_t>(utils::GetChannels(ifmStripeShape));

    assert(ifmStripeChannels != 0);

    mceSchedulerData.numStripes.ifmChannels =
        mceSchedulerData.mceOpMode == MceOperation::DEPTHWISE_CONVOLUTION
            ? 1
            : ethosn::utils::NumericCast<uint16_t>(utils::GetNumStripesC(ifmShape, ifmStripeShape));

    mceSchedulerData.dfltStripeSize.ifmChannels = ifmStripeChannels;

    mceSchedulerData.edgeStripeSize.ifmChannels = CommonUtils::CalculateEdgeSize(ifmChannels, ifmStripeChannels);
}

inline void SetStripeIdStrides(MceS& mceSchedulerData, TraversalOrder traversalOrder)
{
    // ifmChannels is one because we always go through the full IFM depth first (no matter the traversal order)
    mceSchedulerData.stripeIdStrides.ifmChannels = 1U;
    if (traversalOrder == TraversalOrder::Xyz)
    {
        // We move to the next column after we have done "ifmChannels" number of stripes
        mceSchedulerData.stripeIdStrides.ofmWidth = mceSchedulerData.numStripes.ifmChannels;
        // We move to the next row after we have done "ifmChannels * width" number of stripes
        mceSchedulerData.stripeIdStrides.ofmHeight = ethosn::utils::NumericCast<uint16_t>(
            mceSchedulerData.numStripes.ifmChannels * mceSchedulerData.numStripes.ofmWidth);
        // Finally, we move to the next OFM depth after we have done "ifmChannels * width * height" number of stripes
        mceSchedulerData.stripeIdStrides.ofmChannels = ethosn::utils::NumericCast<uint16_t>(
            mceSchedulerData.numStripes.ifmChannels * mceSchedulerData.numStripes.ofmWidth *
            mceSchedulerData.numStripes.ofmHeight);
    }
    else if (traversalOrder == TraversalOrder::Zxy)
    {
        // We move to the next OFM depth after we have done "ifmChannels" number of stripes
        mceSchedulerData.stripeIdStrides.ofmChannels = mceSchedulerData.numStripes.ifmChannels;
        // We move to the next column after we have done "ifmChannels * ofmChannels" number of stripes
        mceSchedulerData.stripeIdStrides.ofmWidth = ethosn::utils::NumericCast<uint16_t>(
            mceSchedulerData.numStripes.ifmChannels * mceSchedulerData.numStripes.ofmChannels);
        // Finally, we move to the row after we have done "ifmChannels * ofmChannels * width" number of stripes
        mceSchedulerData.stripeIdStrides.ofmHeight = ethosn::utils::NumericCast<uint16_t>(
            mceSchedulerData.numStripes.ifmChannels * mceSchedulerData.numStripes.ofmWidth *
            mceSchedulerData.numStripes.ifmChannels * mceSchedulerData.numStripes.ofmChannels);
    }
    else
    {
        assert(false);
    }
}

inline void setMcesOpMode(MceS& mceSchedulerData, command_stream::MceOperation operationMode)
{
    if (operationMode == command_stream::MceOperation::CONVOLUTION)
    {
        mceSchedulerData.mceOpMode = MceOperation::CONVOLUTION;
    }
    else if (operationMode == command_stream::MceOperation::DEPTHWISE_CONVOLUTION)
    {
        mceSchedulerData.mceOpMode = MceOperation::DEPTHWISE_CONVOLUTION;
    }
    else if (operationMode == command_stream::MceOperation::FULLY_CONNECTED)
    {
        mceSchedulerData.mceOpMode = MceOperation::FULLY_CONNECTED;
    }
    else
    {
        assert(false);
    }
}

inline void setMcesAlgorithm(MceS& mceSchedulerData, CompilerMceAlgorithm algorithm)
{
    if (algorithm == CompilerMceAlgorithm::Direct)
    {
        mceSchedulerData.algorithm = MceAlgorithm::DIRECT;
    }
    else if (algorithm == CompilerMceAlgorithm::Winograd)
    {
        mceSchedulerData.algorithm = MceAlgorithm::WINOGRAD;
    }
    else
    {
        assert(false);
    }
}

/// Sets the following properties of the MceS data:
///   * filterShape
///   * ifmDelta[Default/Edge/OneFromEdge]
///   * padding
/// The calculations account for the various supported values of:
///   * Filter shapes (e.g. 3x3)
///   * Different padding types (e.g. SAME, VALID)
///   * Striding (none or 2x2)
///   * Upscaling (none or 2x, with odd or even output sizes)
///   * Packed boundary data (none, either or both dimensions) - although this doesn't actually affect the result, thankfully
inline void SetMcesConvolutionData(MceS& mceS, const OpGraph& opGraph, MceOp* const mceOp, bool isWideFilter)
{
    using namespace ethosn::utils;

    const OpGraph::BufferList inputBuffers = opGraph.GetInputs(mceOp);
    const Buffer* inputBuffer              = inputBuffers[g_MceIfmBufferIndex];
    const Buffer* weightBuffer             = inputBuffers[g_MceWeightBufferIndex];
    const Buffer* outputBuffer             = opGraph.GetOutput(mceOp);

    const uint32_t outputWidth  = utils::GetWidth(outputBuffer->m_TensorShape);
    const uint32_t outputHeight = utils::GetHeight(outputBuffer->m_TensorShape);

    const uint32_t inputWidth  = utils::GetWidth(inputBuffer->m_TensorShape);
    const uint32_t inputHeight = utils::GetHeight(inputBuffer->m_TensorShape);

    const uint32_t filterWidth  = weightBuffer->m_TensorShape[1];
    const uint32_t filterHeight = weightBuffer->m_TensorShape[0];

    const uint32_t strideX = mceOp->m_Stride.m_X;
    const uint32_t strideY = mceOp->m_Stride.m_Y;

    const uint32_t padLeft = mceOp->m_PadLeft;
    const uint32_t padTop  = mceOp->m_PadTop;

    const bool isUpsample = mceS.upsampleType != UpsampleType::OFF;
    auto Upscale          = [isUpsample](uint32_t dim, UpsampleEdgeMode mode) {
        return isUpsample ? dim * 2 - (mode == UpsampleEdgeMode::DROP ? 1 : 0) : dim;
    };
    const uint32_t upscaledInputWidth  = Upscale(inputWidth, mceS.upsampleEdgeMode.col);
    const uint32_t upscaledInputHeight = Upscale(inputHeight, mceS.upsampleEdgeMode.row);

    mceS.isWideFilter = isWideFilter;

    // For strided convolution, previous agents have 'interleaved' the data to break it down into four submaps
    // per original IFM channel, and this agent will combine them back again. Therefore we have up to 4 sets of
    // filterShapes and ifmDeltas to calculate, one for each submap.
    const std::vector<SubmapFilter> filters =
        GetSubmapFilters(filterWidth, filterHeight, strideX, strideY, padLeft, padTop, weightBuffer->m_TensorShape);
    for (size_t s = 0; s < filters.size(); s++)
    {
        const SubmapFilter& subfilter = filters[s];

        // Figure out the size the IFM for this subfilter, accounting for upscaling and striding
        uint32_t submapIfmWidth  = 0;
        uint32_t submapIfmHeight = 0;
        if (isUpsample)
        {
            // Strided and upsampling aren't supported at the same time, so this logic deals only with one or the other, not both.
            assert(strideX == 1 && strideY == 1);
            submapIfmWidth  = upscaledInputWidth;
            submapIfmHeight = upscaledInputHeight;
        }
        else
        {
            const TensorShape submapIfmShape = subfilter.GetIfmSubmapShape(
                mceOp->m_uninterleavedInputShape.has_value() ? mceOp->m_uninterleavedInputShape.value()
                                                             : inputBuffer->m_TensorShape);
            submapIfmWidth  = utils::GetWidth(submapIfmShape);
            submapIfmHeight = utils::GetHeight(submapIfmShape);
        }

        // Set the filter shapes
        {
            // If stride is greater than the filter shape in any dimension, some submaps don't participate in the computation.
            // For those cases a kernel 1x1 with weights equal to zero is created in the support library.
            // For example, a 1x1 kernel with 2x2 stride: only one submap will actually be relevant.
            mceS.filterShape[s].height = NumericCast<uint8_t>(std::max(1u, subfilter.GetFilterY()));
            mceS.filterShape[s].width  = NumericCast<uint8_t>(std::max(1u, subfilter.GetFilterX()));

            // When using wide filter, the filter shape needs to be rounded up to be a whole multiple
            // of the base filter size (3). Except for the value 1, which doesn't round.
            if (isWideFilter)
            {
                if (mceS.filterShape[s].height > 1)
                {
                    mceS.filterShape[s].height = static_cast<uint8_t>(
                        utils::RoundUpToNearestMultiple(static_cast<uint32_t>(mceS.filterShape[s].height), 3));
                }
                if (mceS.filterShape[s].width > 1)
                {
                    mceS.filterShape[s].width = static_cast<uint8_t>(
                        utils::RoundUpToNearestMultiple(static_cast<uint32_t>(mceS.filterShape[s].width), 3));
                }
            }
        }

        // Set the padding on the top/left side.
        {
            // This is the offset between the OFM element being computed and the top-left corner of the window of IFM
            // elements that are multiplied by the filter. This may include real IFM data or padding (i.e. zeroes).
            // Coordinates are aligned with 0,0 being both the top left of the OFM stripe, and the top-left of the IFM centre slot.
            // For non-strided cases, this is simply the top/left padding of the overall convolution, but for striding there
            // is an additional offset as some submaps skip some IFM elements.
            mceS.padding[s].left = NumericCast<uint8_t>(subfilter.GetPadLeft(padLeft));
            mceS.padding[s].top  = NumericCast<uint8_t>(subfilter.GetPadTop(padTop));
        }

        // Set the IFM deltas
        {
            // The deltas are the amount of additional valid (not zero padding) IFM data outside the boundary of the OFM stripe
            // on the right/bottom edge.
            // Coordinates are aligned with 0,0 being both the top left of the OFM stripe, and the top-left of the IFM centre slot.
            // Some general notes:
            //    * For stripes that are towards the top-left of the tensor, there will be plenty of IFM data available to the
            //      bottom/right, so these values will be large
            //    * For stripes nearer the bottom-right, the amount of IFM data available will be less, as we are nearing the edge.
            //    * We don't need to set this value to the exact amount of available data (and in fact can't, because of firmware
            //      limitations); it only needs to be large enough to include the data that will actually be needed.
            //      For example, a 3x3 convolution with SAME padding will only need 1 pixel of additional IFM data outside the OFM
            //      boundary, so even if there is loads of IFM data available any value >= 1 will be correct.
            //    * When the IFM is upscaled, the delta values need to account for this (i.e. the geometry uses the upscaled IFM,
            //      not the original IFM).
            //    * When we have multiple submaps for striding, each submap may have a different IFM and filter size, so the deltas
            //      may be different.
            // Because the deltas vary between stripes of the OFM, we can't provide a single value in the command stream.
            // However we don't need a separate value for every stripe, because most of the deltas across the OFM will be the same,
            // and it's only towards the bottom/right edge that the deltas are different. For the currently supported cases,
            // we need three sets - one for the top-left OFM stripes ("default"), one for the second-to-last row/col of OFM stripes
            // ("oneFromEdge"), and one for the final row/col of OFM stripes ("edge").

            constexpr int8_t maxDelta = 15;    // The maximum value the firmware supports

            // The stripes towards the top/left have plenty of IFM data available to the bottom/right, so set these
            // deltas to a large enough value to cover all supported cases
            mceS.ifmDeltaDefault[s].width  = maxDelta;
            mceS.ifmDeltaDefault[s].height = maxDelta;

            // Because the OFM and IFM coordinate spaces are aligned, the last column/row of OFM stripes has additional
            // IFM data equal simply to the difference between the overall IFM and OFM tensor sizes
            mceS.ifmDeltaEdge[s].width =
                NumericCast<int8_t>(static_cast<int32_t>(submapIfmWidth) - static_cast<int32_t>(outputWidth));
            mceS.ifmDeltaEdge[s].height =
                NumericCast<int8_t>(static_cast<int32_t>(submapIfmHeight) - static_cast<int32_t>(outputHeight));

            // The OFM stripe row/col just before the last one will have more IFM data available, as there is one more
            // stripe of data there. That stripe will be an edge stripe. Note that depending on the streaming strategy,
            // that final stripe of data may actually be packed into the same stripe (packed boundary data), but this doesn't
            // affect the _amount_ of data available, and so this calculation is still correct.
            // This value could be quite large, so limit it to the firmware max supported value.
            mceS.ifmDeltaOneFromEdge[s].width = NumericCast<int8_t>(
                std::min<int32_t>(mceS.ifmDeltaEdge[s].width + mceS.edgeStripeSize.ofmWidth, maxDelta));
            mceS.ifmDeltaOneFromEdge[s].height = NumericCast<int8_t>(
                std::min<int32_t>(mceS.ifmDeltaEdge[s].height + mceS.edgeStripeSize.ofmHeight, maxDelta));
        }
    }

    // All four filterShapes must be set to the same even when there is only one that is used (firmware requirement)
    for (size_t s = filters.size(); s < 4; s++)
    {
        mceS.filterShape[s] = mceS.filterShape[0];
    }
}

}    // namespace MceSUtils

namespace PleSUtils
{

inline void SetPlesTileInfo(const HardwareCapabilities& hwCap, PleS& pleS, const SramBuffer* const outputBuffer)
{
    pleS.ofmTile.baseAddr = ethosn::utils::NumericCast<uint32_t>(outputBuffer->m_Offset.value());
    const uint32_t ratio = utils::DivRoundUp(utils::GetHeight(outputBuffer->m_StripeShape), pleS.dfltStripeSize.height);
    pleS.ofmTile.numSlots = ethosn::utils::NumericCast<uint16_t>(outputBuffer->m_NumStripes * ratio);
    pleS.ofmTile.slotSize = ethosn::utils::NumericCast<uint32_t>(
        utils::DivRoundUp(pleS.dfltStripeSize.width * pleS.dfltStripeSize.height * pleS.dfltStripeSize.channels,
                          hwCap.GetNumberOfSrams()));
}

inline void
    SetPlesHeightStripeInfo(PleS& pleSchedulerData, const TensorShape& ofmShape, const TensorShape& ofmStripeShape)
{
    uint16_t ofmHeight       = ethosn::utils::NumericCast<uint16_t>(utils::GetHeight(ofmShape));
    uint16_t ofmStripeHeight = ethosn::utils::NumericCast<uint16_t>(utils::GetHeight(ofmStripeShape));

    pleSchedulerData.dfltStripeSize.height = ofmStripeHeight;
    pleSchedulerData.numStripes.height =
        ethosn::utils::NumericCast<uint16_t>(utils::GetNumStripesH(ofmShape, ofmStripeShape));

    pleSchedulerData.edgeStripeSize.height = CommonUtils::CalculateEdgeSize(ofmHeight, ofmStripeHeight);
}

inline void
    SetPlesWidthStripeInfo(PleS& pleSchedulerData, const TensorShape& ofmShape, const TensorShape& ofmStripeShape)
{
    uint16_t ofmWidth       = ethosn::utils::NumericCast<uint16_t>(utils::GetWidth(ofmShape));
    uint16_t ofmStripeWidth = ethosn::utils::NumericCast<uint16_t>(utils::GetWidth(ofmStripeShape));

    pleSchedulerData.dfltStripeSize.width = ofmStripeWidth;
    pleSchedulerData.numStripes.width =
        ethosn::utils::NumericCast<uint16_t>(utils::GetNumStripesW(ofmShape, ofmStripeShape));

    pleSchedulerData.edgeStripeSize.width = CommonUtils::CalculateEdgeSize(ofmWidth, ofmStripeWidth);
}

inline void
    SetPlesChannelsStripeInfo(PleS& pleSchedulerData, const TensorShape& ofmShape, const TensorShape& ofmStripeShape)
{
    uint16_t ofmChannels       = ethosn::utils::NumericCast<uint16_t>(utils::GetChannels(ofmShape));
    uint16_t ofmStripeChannels = ethosn::utils::NumericCast<uint16_t>(utils::GetChannels(ofmStripeShape));

    pleSchedulerData.dfltStripeSize.channels = ofmStripeChannels;
    pleSchedulerData.numStripes.channels =
        ethosn::utils::NumericCast<uint16_t>(utils::GetNumStripesC(ofmShape, ofmStripeShape));

    pleSchedulerData.edgeStripeSize.channels = CommonUtils::CalculateEdgeSize(ofmChannels, ofmStripeChannels);
}

inline void SetStripeIdStrides(PleS& pleSchedulerData, SramBuffer* outputBuffer)
{
    // Note that this defines the order of stripes within the tensor, NOT the order of blocks within the stripe
    // (which is always XYZ).
    if (outputBuffer->m_Order == TraversalOrder::Xyz)
    {
        // Width is one because we are traversing along X first.
        pleSchedulerData.stripeIdStrides.width = 1;
        // We get to the next row after we have done "width" number of stripes
        pleSchedulerData.stripeIdStrides.height = pleSchedulerData.numStripes.width;
        // We get to the next plane after we have done "width x height" number of stripes
        pleSchedulerData.stripeIdStrides.channels = ethosn::utils::NumericCast<uint16_t>(
            pleSchedulerData.numStripes.width * pleSchedulerData.numStripes.height);
    }
    else if (outputBuffer->m_Order == TraversalOrder::Zxy)
    {
        // Channels is one because we are traversing along Z first.
        pleSchedulerData.stripeIdStrides.channels = 1U;
        // We get to the next column after we have done "channels" number of stripes
        pleSchedulerData.stripeIdStrides.width = pleSchedulerData.numStripes.channels;
        // We get to the next row after we have done "depth x width" number of stripes
        pleSchedulerData.stripeIdStrides.height = ethosn::utils::NumericCast<uint16_t>(
            pleSchedulerData.numStripes.width * pleSchedulerData.numStripes.channels);
    }
    else
    {
        assert(false);
    }
}

inline void SetFusedPleSInputMode(PleS& pleSchedulerData, MceOp* pleOpProducer)
{
    // Calculate input mode of Ple OP dependent on input buffer producer.
    switch (pleOpProducer->m_Op)
    {
        case command_stream::MceOperation::CONVOLUTION:
            pleSchedulerData.inputMode = PleInputMode::MCE_ALL_OGS;
            break;
        case command_stream::MceOperation::DEPTHWISE_CONVOLUTION:
            pleSchedulerData.inputMode = PleInputMode::MCE_ONE_OG;
            break;
        case command_stream::MceOperation::FULLY_CONNECTED:
            pleSchedulerData.inputMode = PleInputMode::MCE_ALL_OGS;
            break;
        default:
            assert(false);
    }
}

}    // namespace PleSUtils

namespace DependencyUtils
{

inline void CalculateInnerRatio(command_stream::cascading::Dependency& agentDependency)
{
    if (agentDependency.outerRatio.self > agentDependency.outerRatio.other)
    {
        if (agentDependency.innerRatio.self == 0)
        {
            agentDependency.innerRatio.self = ethosn::utils::NumericCast<uint16_t>(agentDependency.outerRatio.self /
                                                                                   agentDependency.outerRatio.other);
        }
    }
    else
    {
        if (agentDependency.innerRatio.other == 0)
        {
            agentDependency.innerRatio.other = ethosn::utils::NumericCast<uint16_t>(agentDependency.outerRatio.other /
                                                                                    agentDependency.outerRatio.self);
        }
    }
}

inline uint16_t CalculateGCD(uint16_t a, uint16_t b)
{
    if (a == 0)
    {
        return ethosn::utils::NumericCast<uint16_t>(b);
    }
    return CalculateGCD(static_cast<uint16_t>(b % a), a);
}

inline uint16_t FindGreatestCommonDenominator(uint16_t a, uint16_t b, uint16_t c)
{
    uint16_t gcdAB = CalculateGCD(a, b);
    if (c == 0)
    {
        return gcdAB;
    }
    else
    {
        return CalculateGCD(gcdAB, c);
    }
    return 1;
}

inline void CalculateRemainingAgentDependencies(command_stream::cascading::Dependency& agentDependency)
{
    uint8_t boundary = 0U;
    bool simplify    = false;
    if (agentDependency.outerRatio.self > agentDependency.outerRatio.other)
    {
        if (agentDependency.innerRatio.other == 0)
        {
            boundary = agentDependency.boundary = ethosn::utils::NumericCast<int8_t>(
                agentDependency.outerRatio.self - (agentDependency.innerRatio.self * agentDependency.outerRatio.other));
            agentDependency.innerRatio.other = 1;
            agentDependency.boundary         = boundary;
            simplify                         = true;
        }
    }
    else
    {
        if (agentDependency.innerRatio.self == 0)
        {
            boundary = ethosn::utils::NumericCast<int8_t>(
                agentDependency.outerRatio.other -
                (agentDependency.innerRatio.other * agentDependency.outerRatio.self));
            agentDependency.innerRatio.self = 1;
            agentDependency.boundary        = boundary;
            simplify                        = true;
        }
    }

    if (simplify)
    {
        uint16_t commonFactor = FindGreatestCommonDenominator(
            agentDependency.outerRatio.other, agentDependency.outerRatio.self, agentDependency.boundary);

        // Reduce dependency values by a common factor to produce equivalent but smaller outer ratios
        agentDependency.outerRatio.other =
            ethosn::utils::NumericCast<uint16_t>(agentDependency.outerRatio.other / commonFactor);
        agentDependency.outerRatio.self =
            ethosn::utils::NumericCast<uint16_t>(agentDependency.outerRatio.self / commonFactor);
        agentDependency.boundary = ethosn::utils::NumericCast<int8_t>(agentDependency.boundary / commonFactor);
    }
}

// Adds a new dependency to the first free slot of the given array of dependencies.
// A free slot is determined by the relativeAgentId being zero.
// If no free slots are available, this asserts.
template <size_t N>
inline void AddDependency(std::array<command_stream::cascading::Dependency, N>& deps,
                          const command_stream::cascading::Dependency& dep)
{
    assert(dep.relativeAgentId != 0);
    for (uint32_t i = 0; i < deps.size(); ++i)
    {
        if (deps[i].relativeAgentId == 0)
        {
            deps[i] = dep;
            return;
        }
    }
    assert(false);
}

int8_t CalculateMceSBoundary(const command_stream::cascading::MceS& mce)
{
    // MceS needs to wait for two IfmS stripes at the start of each outer ratio if neighbouring data
    // is needed. This is not applicable if all the boundary data is packed though.
    if (!(mce.isPackedBoundaryX && mce.isPackedBoundaryY))
    {
        uint8_t maxFilterWidth  = utils::Max<uint8_t>(mce.filterShape, [](const FilterShape& s) { return s.width; });
        uint8_t maxFilterHeight = utils::Max<uint8_t>(mce.filterShape, [](const FilterShape& s) { return s.height; });

        bool needsBoundaryBeforeX =
            mce.numStripes.ofmWidth > 1 && (maxFilterWidth >= 2 || mce.upsampleType != UpsampleType::OFF);
        bool needsBoundaryAfterX =
            mce.numStripes.ofmWidth > 1 && (maxFilterWidth >= 3 || mce.upsampleType != UpsampleType::OFF);
        bool needsBoundaryBeforeY =
            mce.numStripes.ofmHeight > 1 && (maxFilterHeight >= 2 || mce.upsampleType != UpsampleType::OFF);
        bool needsBoundaryAfterY =
            mce.numStripes.ofmHeight > 1 && (maxFilterHeight >= 3 || mce.upsampleType != UpsampleType::OFF);
        return needsBoundaryBeforeX || needsBoundaryAfterX || needsBoundaryBeforeY || needsBoundaryAfterY;
    }
    return 0;
}

void CalculateIfmSMceSOuterRatio(const command_stream::cascading::Agent& mce,
                                 const command_stream::cascading::Agent& ifm,
                                 uint16_t& outMceRatio,
                                 uint16_t& outIfmRatio)
{
    // Determine which dimension (if any) should correspond to the "outer ratio" in the dependency
    if (ifm.data.ifm.fmData.numStripes.channels > 1 && mce.data.mce.mceOpMode != MceOperation::DEPTHWISE_CONVOLUTION)
    {
        // IFM splitting => this is the outer dim
        outIfmRatio = ethosn::utils::NumericCast<uint16_t>(ifm.data.ifm.fmData.numStripes.channels);
        outMceRatio = ethosn::utils::NumericCast<uint16_t>(mce.data.mce.numStripes.ifmChannels);
    }
    else if (mce.data.mce.mceOpMode == MceOperation::DEPTHWISE_CONVOLUTION &&
             ifm.data.ifm.fmData.numStripes.height > 1 && ifm.data.ifm.fmData.numStripes.channels > 1)
    {
        // Depthwise with splitting height and channels => outer ratio is for each column
        outIfmRatio = ethosn::utils::NumericCast<uint16_t>(ifm.data.ifm.fmData.numStripes.height);
        outMceRatio = ethosn::utils::NumericCast<uint16_t>(mce.data.mce.numStripes.ofmHeight);
    }
    else if (mce.data.mce.mceOpMode == MceOperation::DEPTHWISE_CONVOLUTION &&
             ifm.data.ifm.fmData.numStripes.width > 1 && ifm.data.ifm.fmData.numStripes.channels > 1)
    {
        // Depthwise with splitting width and channels => outer ratio is for each row
        outIfmRatio = ethosn::utils::NumericCast<uint16_t>(ifm.data.ifm.fmData.numStripes.width);
        outMceRatio = ethosn::utils::NumericCast<uint16_t>(mce.data.mce.numStripes.ofmWidth);
    }
    else if (ifm.data.ifm.fmData.numStripes.height > 1 && ifm.data.ifm.fmData.numStripes.width > 1)
    {
        // Splitting width and height => outer ratio is for each row
        outIfmRatio = ethosn::utils::NumericCast<uint16_t>(
            ifm.data.ifm.fmData.numStripes.width *
            // Note we use the ifmChannels from the MceS, not the IfmS, so that this is correct for depthwise
            // (where IfmS might have multiple IFM stripes but MceS won't)
            mce.data.mce.numStripes.ifmChannels);
        outMceRatio = ethosn::utils::NumericCast<uint16_t>(mce.data.mce.numStripes.ofmWidth *
                                                           mce.data.mce.numStripes.ifmChannels);
    }
    else
    {
        // Outer ratio is not needed (set to max)
        outIfmRatio = ifm.info.numStripesTotal;
        outMceRatio = mce.info.numStripesTotal;
    }
}

}    // namespace DependencyUtils
}    // namespace cascading_compiler
}    // namespace support_library
}    // namespace ethosn
