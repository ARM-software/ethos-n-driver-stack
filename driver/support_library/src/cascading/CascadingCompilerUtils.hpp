//
// Copyright © 2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Plan.hpp"
#include "Utils.hpp"

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

    tile.baseAddr = static_cast<uint16_t>(buffer->m_Offset.value());
    tile.numSlots = static_cast<uint16_t>(buffer->m_NumStripes);

    switch (buffer->m_Format)
    {
        case CascadingBufferFormat::NHWCB:
            tile.slotSize =
                static_cast<uint16_t>(utils::TotalSizeBytesNHWCB(buffer->m_StripeShape) / hwCap.GetNumberOfSrams());
            break;
        case CascadingBufferFormat::WEIGHT:
            tile.slotSize =
                static_cast<uint16_t>(buffer->m_SizeInBytes / (hwCap.GetNumberOfSrams() * buffer->m_NumStripes));
            break;
        default:
            assert(false);
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

inline void SetStripeHeightInfo(FmSData& streamerData, const TensorShape& tensorShape, const TensorShape& stripeShape)
{
    uint16_t tensorHeight = static_cast<uint16_t>(utils::GetHeight(tensorShape));
    uint16_t stripeHeight = static_cast<uint16_t>(utils::GetHeight(stripeShape));

    assert(stripeHeight != 0);

    streamerData.numStripes.height = static_cast<uint16_t>(utils::GetNumStripesH(tensorShape, stripeShape));

    streamerData.dfltStripeSize.height = stripeHeight;

    streamerData.edgeStripeSize.height = stripeHeight;

    uint16_t remainingHeight = tensorHeight % stripeHeight;

    if (remainingHeight != 0)
    {
        streamerData.edgeStripeSize.height = remainingHeight;
    }
}

inline void SetStripeWidthInfo(FmSData& streamerData, const TensorShape& tensorShape, const TensorShape& stripeShape)
{
    uint16_t tensorWidth = static_cast<uint16_t>(utils::GetWidth(tensorShape));
    uint16_t stripeWidth = static_cast<uint16_t>(utils::GetWidth(stripeShape));

    assert(stripeWidth != 0);

    streamerData.numStripes.width = static_cast<uint16_t>(utils::GetNumStripesW(tensorShape, stripeShape));

    streamerData.dfltStripeSize.width = stripeWidth;

    streamerData.edgeStripeSize.width = stripeWidth;

    uint16_t remainingWidth = tensorWidth % stripeWidth;

    if (remainingWidth != 0)
    {
        streamerData.edgeStripeSize.width = remainingWidth;
    }
}

inline void SetStripeChannelsInfo(FmSData& streamerData, const TensorShape& tensorShape, const TensorShape& stripeShape)
{
    uint16_t tensorChannels = static_cast<uint16_t>(utils::GetChannels(tensorShape));
    uint16_t stripeChannels = static_cast<uint16_t>(utils::GetChannels(stripeShape));

    assert(stripeChannels != 0);

    streamerData.numStripes.channels = static_cast<uint16_t>(utils::GetNumStripesC(tensorShape, stripeShape));

    streamerData.dfltStripeSize.channels = stripeChannels;

    streamerData.edgeStripeSize.channels = stripeChannels;

    uint16_t remainingChannels = tensorChannels % stripeChannels;

    if (remainingChannels != 0)
    {
        streamerData.edgeStripeSize.channels = remainingChannels;
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

    streamerData.supertensorSizeInCells.width    = static_cast<uint16_t>(utils::DivRoundUp(tensorShape[2], cellWidth));
    streamerData.supertensorSizeInCells.channels = static_cast<uint16_t>(utils::DivRoundUp(tensorShape[3], cellDepth));
}

inline void SetStripeIdStrides(FmSData& streamerData, TraversalOrder traversalOrder)
{
    if (traversalOrder == TraversalOrder::Xyz)
    {
        streamerData.stripeIdStrides.height =
            static_cast<uint16_t>(streamerData.numStripes.width * streamerData.numStripes.channels);
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
    uint16_t ofmHeight       = static_cast<uint16_t>(utils::GetHeight(ofmShape));
    uint16_t ofmStripeHeight = static_cast<uint16_t>(utils::GetHeight(ofmStripeShape));

    assert(ofmStripeHeight != 0);

    mceSchedulerData.numStripes.ofmHeight = static_cast<uint16_t>(utils::GetNumStripesH(ofmShape, ofmStripeShape));

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
    uint16_t ofmWidth       = static_cast<uint16_t>(utils::GetWidth(ofmShape));
    uint16_t ofmStripeWidth = static_cast<uint16_t>(utils::GetWidth(ofmStripeShape));

    assert(ofmStripeWidth != 0);

    mceSchedulerData.numStripes.ofmWidth = static_cast<uint16_t>(utils::GetNumStripesW(ofmShape, ofmStripeShape));

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
    uint16_t ofmChannels       = static_cast<uint16_t>(utils::GetChannels(ofmShape));
    uint16_t ofmStripeChannels = static_cast<uint16_t>(utils::GetChannels(ofmStripeShape));

    assert(ofmStripeChannels != 0);

    mceSchedulerData.numStripes.ofmChannels = static_cast<uint16_t>(utils::GetNumStripesC(ofmShape, ofmStripeShape));

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
    uint16_t ifmChannels       = static_cast<uint16_t>(utils::GetChannels(ifmShape));
    uint16_t ifmStripeChannels = static_cast<uint16_t>(utils::GetChannels(ifmStripeShape));

    assert(ifmStripeChannels != 0);

    mceSchedulerData.numStripes.ifmChannels = static_cast<uint16_t>(utils::GetNumStripesC(ifmShape, ifmStripeShape));

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
        mceSchedulerData.stripeIdStrides.ofmHeight =
            static_cast<uint16_t>(mceSchedulerData.numStripes.ifmChannels * mceSchedulerData.numStripes.ofmWidth);
        mceSchedulerData.stripeIdStrides.ofmWidth = mceSchedulerData.numStripes.ifmChannels;
        mceSchedulerData.stripeIdStrides.ofmChannels =
            static_cast<uint16_t>(mceSchedulerData.numStripes.ifmChannels * mceSchedulerData.numStripes.ofmWidth *
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
}    // namespace MceSUtils

namespace PleSUtils
{

inline void
    SetPlesHeightStripeInfo(PleS& pleSchedulerData, const TensorShape& ofmShape, const TensorShape& ofmStripeShape)
{
    uint16_t ofmHeight       = static_cast<uint16_t>(utils::GetHeight(ofmShape));
    uint16_t ofmStripeHeight = static_cast<uint16_t>(utils::GetHeight(ofmStripeShape));

    pleSchedulerData.dfltStripeSize.height = ofmStripeHeight;
    pleSchedulerData.numStripes.height     = static_cast<uint16_t>(utils::GetNumStripesH(ofmShape, ofmStripeShape));

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
    uint16_t ofmWidth       = static_cast<uint16_t>(utils::GetWidth(ofmShape));
    uint16_t ofmStripeWidth = static_cast<uint16_t>(utils::GetWidth(ofmStripeShape));

    pleSchedulerData.dfltStripeSize.width = ofmStripeWidth;
    pleSchedulerData.numStripes.width     = static_cast<uint16_t>(utils::GetNumStripesW(ofmShape, ofmStripeShape));

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
    uint16_t ofmChannels       = static_cast<uint16_t>(utils::GetChannels(ofmShape));
    uint16_t ofmStripeChannels = static_cast<uint16_t>(utils::GetChannels(ofmStripeShape));

    pleSchedulerData.dfltStripeSize.channels = ofmStripeChannels;
    pleSchedulerData.numStripes.channels     = static_cast<uint16_t>(utils::GetNumStripesC(ofmShape, ofmStripeShape));

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
        pleSchedulerData.stripeIdStrides.height = static_cast<uint16_t>(pleSchedulerData.numStripes.width);
        pleSchedulerData.stripeIdStrides.width  = 1;
        pleSchedulerData.stripeIdStrides.channels =
            static_cast<uint16_t>(pleSchedulerData.numStripes.width * pleSchedulerData.numStripes.height);
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
}    // namespace cascading_compiler
}    // namespace support_library
}    // namespace ethosn
