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

    tile.baseAddr = ethosn::utils::NumericCast<uint16_t>(buffer->m_Offset.value());
    tile.numSlots = ethosn::utils::NumericCast<uint16_t>(buffer->m_NumStripes);

    switch (buffer->m_Format)
    {
        // Intentional fallthrough
        case CascadingBufferFormat::NHWCB:
            tile.slotSize = ethosn::utils::NumericCast<uint16_t>(
                utils::DivRoundUp(utils::TotalSizeBytesNHWCB(buffer->m_StripeShape), hwCap.GetNumberOfSrams()));
            break;
        case CascadingBufferFormat::WEIGHT:
            tile.slotSize = ethosn::utils::NumericCast<uint16_t>(
                utils::DivRoundUp(buffer->m_SizeInBytes, (hwCap.GetNumberOfSrams() * buffer->m_NumStripes)));
            break;
        default:
            assert(false);
    }
}

uint32_t CalculateBufferSize(const TensorShape& shape, CascadingBufferFormat dataFormat)
{
    assert(dataFormat == CascadingBufferFormat::NHWC || dataFormat == CascadingBufferFormat::NCHW ||
           dataFormat == CascadingBufferFormat::NHWCB || dataFormat == CascadingBufferFormat::FCAF_WIDE ||
           dataFormat == CascadingBufferFormat::FCAF_DEEP);

    switch (dataFormat)
    {
        case CascadingBufferFormat::FCAF_DEEP:
            return ethosn::support_library::utils::TotalSizeBytesFCAFDeep(shape);
        case CascadingBufferFormat::FCAF_WIDE:
            return ethosn::support_library::utils::TotalSizeBytesFCAFWide(shape);
        case CascadingBufferFormat::NHWCB:
            return ethosn::support_library::utils::TotalSizeBytesNHWCB(shape);
        default:
            return ethosn::support_library::utils::TotalSizeBytes(shape);
    }
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

    streamerData.edgeStripeSize.height = stripeHeight;

    uint32_t remainingHeight = tensorHeight % stripeHeight;

    if (remainingHeight != 0)
    {
        streamerData.edgeStripeSize.height = ethosn::utils::NumericCast<uint16_t>(
            utils::RoundUpToNearestMultiple(remainingHeight, utils::GetHeight(hwCap.GetBrickGroupShape())));
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

    streamerData.edgeStripeSize.width = stripeWidth;

    uint32_t remainingWidth = tensorWidth % stripeWidth;

    if (remainingWidth != 0)
    {
        streamerData.edgeStripeSize.width = ethosn::utils::NumericCast<uint16_t>(
            utils::RoundUpToNearestMultiple(remainingWidth, utils::GetWidth(hwCap.GetBrickGroupShape())));
    }
}

inline void SetStripeChannelsInfo(const HardwareCapabilities& hwCap,
                                  FmSData& streamerData,
                                  const TensorShape& tensorShape,
                                  const TensorShape& stripeShape)
{
    uint16_t tensorChannels = ethosn::utils::NumericCast<uint16_t>(utils::GetChannels(tensorShape));
    uint16_t stripeChannels = ethosn::utils::NumericCast<uint16_t>(utils::GetChannels(stripeShape));

    assert(stripeChannels != 0);

    streamerData.numStripes.channels =
        ethosn::utils::NumericCast<uint16_t>(utils::GetNumStripesC(tensorShape, stripeShape));

    streamerData.dfltStripeSize.channels = stripeChannels;

    streamerData.edgeStripeSize.channels = stripeChannels;

    uint32_t remainingChannels = tensorChannels % stripeChannels;

    if (remainingChannels != 0)
    {
        streamerData.edgeStripeSize.channels = ethosn::utils::NumericCast<uint16_t>(
            utils::RoundUpToNearestMultiple(remainingChannels, utils::GetChannels(hwCap.GetBrickGroupShape())));
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
        streamerData.stripeIdStrides.height =
            ethosn::utils::NumericCast<uint16_t>(streamerData.numStripes.width * streamerData.numStripes.channels);
        streamerData.stripeIdStrides.width    = streamerData.numStripes.channels;
        streamerData.stripeIdStrides.channels = 1U;
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

    mceSchedulerData.edgeStripeSize.ofmHeight = ofmStripeHeight;

    uint16_t remainingHeight = ofmHeight % ofmStripeHeight;

    if (remainingHeight != 0)
    {
        mceSchedulerData.edgeStripeSize.ofmHeight = remainingHeight;
    }
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

    mceSchedulerData.edgeStripeSize.ofmWidth = ofmStripeWidth;

    uint16_t remainingWidth = ofmWidth % ofmStripeWidth;

    if (remainingWidth != 0)
    {
        mceSchedulerData.edgeStripeSize.ofmWidth = remainingWidth;
    }
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

    mceSchedulerData.edgeStripeSize.ofmChannels = ofmStripeChannels;

    uint16_t remainingChannels = ofmChannels % ofmStripeChannels;

    if (remainingChannels != 0)
    {
        mceSchedulerData.edgeStripeSize.ofmChannels = remainingChannels;
    }
}

inline void
    SetMcesIfmChannelsStripeInfo(MceS& mceSchedulerData, const TensorShape& ifmShape, const TensorShape& ifmStripeShape)
{
    uint16_t ifmChannels       = ethosn::utils::NumericCast<uint16_t>(utils::GetChannels(ifmShape));
    uint16_t ifmStripeChannels = ethosn::utils::NumericCast<uint16_t>(utils::GetChannels(ifmStripeShape));

    assert(ifmStripeChannels != 0);

    mceSchedulerData.numStripes.ifmChannels =
        ethosn::utils::NumericCast<uint16_t>(utils::GetNumStripesC(ifmShape, ifmStripeShape));

    mceSchedulerData.dfltStripeSize.ifmChannels = ifmStripeChannels;

    mceSchedulerData.edgeStripeSize.ifmChannels = ifmStripeChannels;

    uint16_t remainingChannels = ifmChannels % ifmStripeChannels;

    if (remainingChannels != 0)
    {
        mceSchedulerData.edgeStripeSize.ifmChannels = remainingChannels;
    }
}

inline void SetStripeIdStrides(MceS& mceSchedulerData, TraversalOrder traversalOrder)
{
    if (traversalOrder == TraversalOrder::Xyz)
    {
        mceSchedulerData.stripeIdStrides.ofmHeight = ethosn::utils::NumericCast<uint16_t>(
            mceSchedulerData.numStripes.ifmChannels * mceSchedulerData.numStripes.ofmWidth);
        mceSchedulerData.stripeIdStrides.ofmWidth    = mceSchedulerData.numStripes.ifmChannels;
        mceSchedulerData.stripeIdStrides.ofmChannels = ethosn::utils::NumericCast<uint16_t>(
            mceSchedulerData.numStripes.ifmChannels * mceSchedulerData.numStripes.ofmWidth *
            mceSchedulerData.numStripes.ofmHeight);
        mceSchedulerData.stripeIdStrides.ifmChannels = 1U;
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
    Buffer* weightBuffer             = inputBuffers[g_MceWeightBufferIndex];

    auto filters =
        GetSubmapFilters(weightBuffer->m_TensorShape[1], weightBuffer->m_TensorShape[0], ptrMceOp->m_Stride.m_X,
                         ptrMceOp->m_Stride.m_Y, ptrMceOp->m_PadLeft, ptrMceOp->m_PadTop, weightBuffer->m_TensorShape);

    // Set the filter shapes, padding information, and Ifm delta for each sub map
    for (uint8_t subMapIndex = 0; subMapIndex < 4; subMapIndex++)
    {
        // Set the filter shapes
        mceSchedulerData.filterShape[subMapIndex].height =
            ethosn::utils::NumericCast<uint8_t>(filters[subMapIndex].GetFilterY());
        mceSchedulerData.filterShape[subMapIndex].width =
            ethosn::utils::NumericCast<uint8_t>(filters[subMapIndex].GetFilterX());

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

        /// Ifm stripe width/height delta is the amount of valid data outside the ifm stripe on the right/bottom edge
        /// that can be used to calculate the Ofm stripe. This is equal to the difference between the IFM and OFM
        /// width/height when at the edges of the whole IFM.

        const int32_t ifmStripeNeighboringDataRight =
            static_cast<int32_t>(currSubmapInputWidth) -
            static_cast<int32_t>(utils::GetWidth(ptrMceOp->m_OutputStripeShape));

        const int32_t ifmStripeNeighboringDataBottom =
            static_cast<int32_t>(currSubmapInputHeight) -
            static_cast<int32_t>(utils::GetHeight(ptrMceOp->m_OutputStripeShape));

        // Set the Ifm delta default
        mceSchedulerData.ifmDeltaDefault[subMapIndex].height =
            ethosn::utils::NumericCast<int8_t>(ifmStripeNeighboringDataBottom);
        mceSchedulerData.ifmDeltaDefault[subMapIndex].width =
            ethosn::utils::NumericCast<int8_t>(ifmStripeNeighboringDataRight);

        // Set the Ifm delta edge
        mceSchedulerData.ifmDeltaEdge[subMapIndex].height =
            ethosn::utils::NumericCast<int8_t>(ifmStripeNeighboringDataBottom);
        mceSchedulerData.ifmDeltaEdge[subMapIndex].width =
            ethosn::utils::NumericCast<int8_t>(ifmStripeNeighboringDataRight);
    }
}
}    // namespace MceSUtils

namespace PleSUtils
{

inline void
    SetPlesHeightStripeInfo(PleS& pleSchedulerData, const TensorShape& ofmShape, const TensorShape& ofmStripeShape)
{
    uint16_t ofmHeight       = ethosn::utils::NumericCast<uint16_t>(utils::GetHeight(ofmShape));
    uint16_t ofmStripeHeight = ethosn::utils::NumericCast<uint16_t>(utils::GetHeight(ofmStripeShape));

    pleSchedulerData.dfltStripeSize.height = ofmStripeHeight;
    pleSchedulerData.numStripes.height =
        ethosn::utils::NumericCast<uint16_t>(utils::GetNumStripesH(ofmShape, ofmStripeShape));

    pleSchedulerData.edgeStripeSize.height = ofmStripeHeight;

    uint16_t remainingHeight = ofmHeight % ofmStripeHeight;
    if (remainingHeight != 0)
    {
        pleSchedulerData.edgeStripeSize.height = remainingHeight;
    }
}

inline void
    SetPlesWidthStripeInfo(PleS& pleSchedulerData, const TensorShape& ofmShape, const TensorShape& ofmStripeShape)
{
    uint16_t ofmWidth       = ethosn::utils::NumericCast<uint16_t>(utils::GetWidth(ofmShape));
    uint16_t ofmStripeWidth = ethosn::utils::NumericCast<uint16_t>(utils::GetWidth(ofmStripeShape));

    pleSchedulerData.dfltStripeSize.width = ofmStripeWidth;
    pleSchedulerData.numStripes.width =
        ethosn::utils::NumericCast<uint16_t>(utils::GetNumStripesW(ofmShape, ofmStripeShape));

    pleSchedulerData.edgeStripeSize.width = ofmStripeWidth;

    uint16_t remainingWidth = ofmWidth % ofmStripeWidth;
    if (remainingWidth != 0)
    {
        pleSchedulerData.edgeStripeSize.width = remainingWidth;
    }
}

inline void
    SetPlesChannelsStripeInfo(PleS& pleSchedulerData, const TensorShape& ofmShape, const TensorShape& ofmStripeShape)
{
    uint16_t ofmChannels       = ethosn::utils::NumericCast<uint16_t>(utils::GetChannels(ofmShape));
    uint16_t ofmStripeChannels = ethosn::utils::NumericCast<uint16_t>(utils::GetChannels(ofmStripeShape));

    pleSchedulerData.dfltStripeSize.channels = ofmStripeChannels;
    pleSchedulerData.numStripes.channels =
        ethosn::utils::NumericCast<uint16_t>(utils::GetNumStripesC(ofmShape, ofmStripeShape));

    pleSchedulerData.edgeStripeSize.channels = ofmStripeChannels;

    uint16_t remainingChannels = ofmChannels % ofmStripeChannels;
    if (remainingChannels != 0)
    {
        pleSchedulerData.edgeStripeSize.channels = remainingChannels;
    }
}

inline void SetStripeIdStrides(PleS& pleSchedulerData, Buffer* outputBuffer)
{
    if (outputBuffer->m_Order == TraversalOrder::Xyz)
    {
        pleSchedulerData.stripeIdStrides.height =
            ethosn::utils::NumericCast<uint16_t>(pleSchedulerData.numStripes.width);
        pleSchedulerData.stripeIdStrides.width    = 1;
        pleSchedulerData.stripeIdStrides.channels = ethosn::utils::NumericCast<uint16_t>(
            pleSchedulerData.numStripes.width * pleSchedulerData.numStripes.height);
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
        agentDependency.innerRatio.self =
            ethosn::utils::NumericCast<uint16_t>(agentDependency.outerRatio.self / agentDependency.outerRatio.other);
    }
    else
    {
        agentDependency.innerRatio.other =
            ethosn::utils::NumericCast<uint16_t>(agentDependency.outerRatio.other / agentDependency.outerRatio.self);
    }
}

inline uint8_t CalculateGCD(uint16_t a, uint16_t b)
{
    if (a == 0)
    {
        return ethosn::utils::NumericCast<uint8_t>(b);
    }
    return CalculateGCD(b % a, a);
}

inline uint8_t FindGreatestCommonDenominator(uint16_t a, uint16_t b, uint8_t c)
{
    uint8_t gcdAB = CalculateGCD(a, b);
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
    if (agentDependency.outerRatio.self > agentDependency.outerRatio.other)
    {
        boundary = agentDependency.boundary = ethosn::utils::NumericCast<uint8_t>(
            agentDependency.outerRatio.self - (agentDependency.innerRatio.self * agentDependency.outerRatio.other));
        agentDependency.innerRatio.other = 1;
    }
    else
    {
        boundary = ethosn::utils::NumericCast<uint8_t>(
            agentDependency.outerRatio.other - (agentDependency.innerRatio.other * agentDependency.outerRatio.self));
        agentDependency.innerRatio.self = 1;
    }

    agentDependency.boundary = boundary;
    uint8_t commonFactor     = FindGreatestCommonDenominator(agentDependency.outerRatio.other,
                                                         agentDependency.outerRatio.self, agentDependency.boundary);

    // Reduce dependency values by a common factor to produce equivalent but smaller outer ratios
    agentDependency.outerRatio.other =
        ethosn::utils::NumericCast<uint16_t>(agentDependency.outerRatio.other / commonFactor);
    agentDependency.outerRatio.self =
        ethosn::utils::NumericCast<uint16_t>(agentDependency.outerRatio.self / commonFactor);
    agentDependency.boundary = ethosn::utils::NumericCast<uint8_t>(agentDependency.boundary / commonFactor);
}

}    // namespace DependencyUtils
}    // namespace cascading_compiler
}    // namespace support_library
}    // namespace ethosn
