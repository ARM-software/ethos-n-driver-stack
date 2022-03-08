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
}    // namespace cascading_compiler
}    // namespace support_library
}    // namespace ethosn
