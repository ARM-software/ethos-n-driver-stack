//
// Copyright © 2022 Arm Limited.
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
inline void SetTileInfoForBuffer(const HardwareCapabilities& hwCap, Tile& tile, const Buffer* const buffer)
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

inline uint32_t CalculateDramOffsetNHWCB(const TensorShape& tensorShape,
                                         uint32_t offsetY,
                                         uint32_t offsetX,
                                         uint32_t offsetC,
                                         const HardwareCapabilities& caps)
{
    using namespace utils;
    const TensorShape& brickGroupShape = caps.GetBrickGroupShape();
    const TensorShape& patchShape      = caps.GetPatchShape();
    const uint32_t brickGroupSize      = GetNumElements(brickGroupShape);
    const uint32_t brickGroupHeight    = GetHeight(brickGroupShape);
    const uint32_t brickGroupWidth     = GetWidth(brickGroupShape);
    const uint32_t brickGroupChannels  = GetChannels(brickGroupShape);
    const uint32_t patchSize           = GetNumElements(patchShape);
    const uint32_t patchHeight         = GetHeight(patchShape);
    const uint32_t patchWidth          = GetWidth(patchShape);

    uint32_t numBrickGroupDepth = utils::DivRoundUp(GetChannels(tensorShape), brickGroupChannels);
    uint32_t numBrickGroupWidth = utils::DivRoundUp(GetWidth(tensorShape), brickGroupWidth);

    uint32_t offsetBrickGroupX = offsetX / brickGroupWidth;
    uint32_t offsetBrickGroupY = offsetY / brickGroupHeight;
    uint32_t offsetBrickGroupC = offsetC / brickGroupChannels;
    uint32_t offsetChannels    = offsetC % brickGroupChannels;
    uint32_t offsetBrickGroups = offsetBrickGroupC + offsetBrickGroupX * numBrickGroupDepth +
                                 offsetBrickGroupY * numBrickGroupDepth * numBrickGroupWidth;
    uint32_t offsetWithinBrickGroupX         = offsetX % brickGroupWidth;
    uint32_t offsetWithinBrickGroupY         = offsetY % brickGroupHeight;
    uint32_t patchWithinBrickGroupX          = offsetWithinBrickGroupX / patchWidth;
    uint32_t patchWithinBrickGroupY          = offsetWithinBrickGroupY / patchHeight;
    const uint32_t brickGroupHeightInPatches = brickGroupHeight / patchHeight;
    uint32_t brickWithinBrickGroup  = patchWithinBrickGroupX * brickGroupHeightInPatches + patchWithinBrickGroupY;
    uint32_t offsetWithinBrickGroup = (brickWithinBrickGroup * brickGroupChannels + offsetChannels) * patchSize;

    uint32_t offsetBytes = brickGroupSize * offsetBrickGroups + offsetWithinBrickGroup;

    return offsetBytes;
}

inline uint32_t
    CalculateDramOffsetNHWC(const TensorShape& tensorShape, uint32_t offsetY, uint32_t offsetX, uint32_t offsetC)
{
    using namespace utils;
    return offsetC + offsetX * GetChannels(tensorShape) + offsetY * GetChannels(tensorShape) * GetWidth(tensorShape);
}

inline uint32_t GetDramOffset(const CascadingBufferFormat dataFormat,
                              const TensorShape& tensorSize,
                              const TensorShape& offset,
                              const HardwareCapabilities& caps)
{
    uint32_t offsetBytes = 0;

    switch (dataFormat)
    {
        case CascadingBufferFormat::NHWCB:
        {
            offsetBytes = CalculateDramOffsetNHWCB(tensorSize, offset[1], offset[2], offset[3], caps);
            break;
        }
        case CascadingBufferFormat::NHWC:
        case CascadingBufferFormat::NCHW:
        {
            offsetBytes = CalculateDramOffsetNHWC(tensorSize, offset[1], offset[2], offset[3]);
            break;
        }
        default:
        {
            assert(false);
        }
    };

    return offsetBytes;
}

inline std::vector<Op*> GetSortedOps(const OpGraph& opGraph)
{
    std::vector<Op*> targets;
    for (const auto& op : opGraph.GetOps())
    {
        auto outputBuf = opGraph.GetOutput(op);
        if (outputBuf != nullptr)
        {
            const auto& consumers = opGraph.GetConsumers(outputBuf);
            // If the op's output buffer doesn't have an output it is a leaf node
            if (consumers.size() == 0)
            {
                targets.push_back(op);
            }
        }
    }
    std::vector<Op*> sorted;
    // Define a function to get the incoming vertices for the topological sort
    auto GetIncomingOps = [&](Op* op) {
        std::vector<Op*> result;
        const OpGraph::BufferList& inputBuffers = opGraph.GetInputs(op);
        for (const auto& buf : inputBuffers)
        {
            Op* op = opGraph.GetProducer(buf);
            if (op != nullptr)
            {
                result.push_back(op);
            }
        }
        return result;
    };
    utils::GraphTopologicalSort<Op*, std::vector<Op*>>(targets, GetIncomingOps, sorted);

    return sorted;
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

inline void SetStripeHeightInfo(const HardwareCapabilities& hwCap,
                                FmSData& streamerData,
                                const TensorShape& tensorShape,
                                const TensorShape& stripeShape)
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
        streamerData.edgeStripeSize.height = ethosn::utils::NumericCast<uint16_t>(utils::RoundUpToNearestMultiple(
            static_cast<uint32_t>(streamerData.edgeStripeSize.height), utils::GetHeight(hwCap.GetBrickGroupShape())));
    }
}

inline void SetStripeWidthInfo(const HardwareCapabilities& hwCap,
                               FmSData& streamerData,
                               const TensorShape& tensorShape,
                               const TensorShape& stripeShape)
{
    uint16_t tensorWidth = ethosn::utils::NumericCast<uint16_t>(utils::GetWidth(tensorShape));
    uint16_t stripeWidth = ethosn::utils::NumericCast<uint16_t>(utils::GetWidth(stripeShape));

    assert(stripeWidth != 0);

    streamerData.numStripes.width =
        ethosn::utils::NumericCast<uint16_t>(utils::GetNumStripesW(tensorShape, stripeShape));

    streamerData.dfltStripeSize.width = stripeWidth;

    streamerData.edgeStripeSize.width = CommonUtils::CalculateEdgeSize(tensorWidth, stripeWidth);
    if (streamerData.dataType != FmsDataType::NHWC)
    {
        streamerData.edgeStripeSize.width = ethosn::utils::NumericCast<uint16_t>(utils::RoundUpToNearestMultiple(
            static_cast<uint32_t>(streamerData.edgeStripeSize.width), utils::GetWidth(hwCap.GetBrickGroupShape())));
    }
}

inline void SetStripeChannelsInfo(FmSData& streamerData, const TensorShape& tensorShape, const TensorShape& stripeShape)
{
    uint16_t tensorChannels = ethosn::utils::NumericCast<uint16_t>(utils::GetChannels(tensorShape));
    uint16_t stripeChannels = ethosn::utils::NumericCast<uint16_t>(utils::GetChannels(stripeShape));

    assert(stripeChannels != 0);

    streamerData.numStripes.channels =
        ethosn::utils::NumericCast<uint16_t>(utils::GetNumStripesC(tensorShape, stripeShape));

    streamerData.dfltStripeSize.channels = stripeChannels;

    streamerData.edgeStripeSize.channels = CommonUtils::CalculateEdgeSize(tensorChannels, stripeChannels);
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

inline void setMcesStridedConvolutionData(MceS& mceSchedulerData, const OpGraph& mergedOpGraph, MceOp* const ptrMceOp)
{

    // Get the input buffers to the Mce Op
    OpGraph::BufferList inputBuffers = mergedOpGraph.GetInputs(ptrMceOp);
    Buffer* inputBuffer              = inputBuffers[g_MceIfmBufferIndex];
    Buffer* weightBuffer             = inputBuffers[g_MceWeightBufferIndex];
    Buffer* outputBuffer             = mergedOpGraph.GetOutput(ptrMceOp);

    auto filters =
        GetSubmapFilters(weightBuffer->m_TensorShape[1], weightBuffer->m_TensorShape[0], ptrMceOp->m_Stride.m_X,
                         ptrMceOp->m_Stride.m_Y, ptrMceOp->m_PadLeft, ptrMceOp->m_PadTop, weightBuffer->m_TensorShape);

    // Set the filter shapes, padding information, and Ifm delta for each sub map
    for (uint8_t subMapIndex = 0; subMapIndex < 4; subMapIndex++)
    {
        // Set the filter shapes

        // If stride is greater than filterSize in any dimension, some submaps don't participate in the computation.
        // For those cases a kernel 1x1 with weights equal to zero is created in the support library.
        mceSchedulerData.filterShape[subMapIndex].height =
            ethosn::utils::NumericCast<uint8_t>(std::max(1u, filters[subMapIndex].GetFilterY()));
        mceSchedulerData.filterShape[subMapIndex].width =
            ethosn::utils::NumericCast<uint8_t>(std::max(1u, filters[subMapIndex].GetFilterX()));

        // Set the padding information for each sub map
        const uint32_t x        = subMapIndex % ptrMceOp->m_Stride.m_X;
        const uint32_t y        = subMapIndex / ptrMceOp->m_Stride.m_X;
        const uint32_t shiftedX = (x + ptrMceOp->m_PadLeft) % ptrMceOp->m_Stride.m_X;
        const uint32_t shiftedY = (y + ptrMceOp->m_PadTop) % ptrMceOp->m_Stride.m_Y;

        const uint32_t leftPad =
            utils::DivRoundUp(static_cast<uint32_t>(std::max(
                                  static_cast<int32_t>(ptrMceOp->m_PadLeft) - static_cast<int32_t>(shiftedX), 0)),
                              ptrMceOp->m_Stride.m_X);
        mceSchedulerData.padding[subMapIndex].left = ethosn::utils::NumericCast<uint8_t>(leftPad);

        const uint32_t topPad =
            utils::DivRoundUp(static_cast<uint32_t>(std::max(
                                  static_cast<int32_t>(ptrMceOp->m_PadTop) - static_cast<int32_t>(shiftedY), 0)),
                              ptrMceOp->m_Stride.m_Y);
        mceSchedulerData.padding[subMapIndex].top = ethosn::utils::NumericCast<uint8_t>(topPad);

        // Set the Ifm delta for each sub map
        assert(ptrMceOp->m_uninterleavedInputShape.has_value());

        uint32_t currSubmapInputWidth  = 0;
        uint32_t currSubmapInputHeight = 0;

        if (ptrMceOp->m_Stride.m_X > 1 || ptrMceOp->m_Stride.m_Y > 1)
        {
            /// This represents the post interleave input width for the specific submap index.
            currSubmapInputWidth = utils::DivRoundUp(
                static_cast<uint32_t>(
                    std::max(static_cast<int32_t>(utils::GetWidth(ptrMceOp->m_uninterleavedInputShape.value())) -
                                 static_cast<int32_t>(x),
                             0)),
                ptrMceOp->m_Stride.m_X);
            /// This represents the post interleave input height for the specific submap index.
            currSubmapInputHeight = utils::DivRoundUp(
                static_cast<uint32_t>(
                    std::max(static_cast<int32_t>(utils::GetHeight(ptrMceOp->m_uninterleavedInputShape.value())) -
                                 static_cast<int32_t>(y),
                             0)),
                ptrMceOp->m_Stride.m_Y);
        }

        // Ifm stripe width/height delta is the amount of valid data outside the ifm stripe on the right/bottom edge
        // that can be used to calculate the Ofm stripe.

        // Set the Ifm delta default
        {
            mceSchedulerData.ifmDeltaDefault[subMapIndex].height = ethosn::utils::NumericCast<int8_t>(
                (mceSchedulerData.filterShape[subMapIndex].height / 2) + inputBuffer->m_PackedBoundaryThickness.bottom);
            mceSchedulerData.ifmDeltaDefault[subMapIndex].width = ethosn::utils::NumericCast<int8_t>(
                (mceSchedulerData.filterShape[subMapIndex].width / 2) + inputBuffer->m_PackedBoundaryThickness.right);
        }

        // Set the Ifm delta edge
        // This is equal to the difference between the IFM and OFM
        // width/height when at the edges of the whole IFM.
        {
            const int32_t ifmStripeNeighboringDataRight =
                static_cast<int32_t>(currSubmapInputWidth) -
                static_cast<int32_t>(utils::GetWidth(outputBuffer->m_TensorShape));

            const int32_t ifmStripeNeighboringDataBottom =
                static_cast<int32_t>(currSubmapInputHeight) -
                static_cast<int32_t>(utils::GetHeight(outputBuffer->m_TensorShape));

            mceSchedulerData.ifmDeltaEdge[subMapIndex].height =
                ethosn::utils::NumericCast<int8_t>(ifmStripeNeighboringDataBottom);
            mceSchedulerData.ifmDeltaEdge[subMapIndex].width =
                ethosn::utils::NumericCast<int8_t>(ifmStripeNeighboringDataRight);
        }
    }
}
}    // namespace MceSUtils

namespace PleSUtils
{

inline void SetPlesTileInfo(const HardwareCapabilities& hwCap, PleS& pleS, const Buffer* const outputBuffer)
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

inline void SetStripeIdStrides(PleS& pleSchedulerData, Buffer* outputBuffer)
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
    return CalculateGCD(b % a, a);
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
            boundary = agentDependency.boundary = ethosn::utils::NumericCast<uint8_t>(
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
            boundary = ethosn::utils::NumericCast<uint8_t>(
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
        agentDependency.boundary = ethosn::utils::NumericCast<uint8_t>(agentDependency.boundary / commonFactor);
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

}    // namespace DependencyUtils
}    // namespace cascading_compiler
}    // namespace support_library
}    // namespace ethosn
