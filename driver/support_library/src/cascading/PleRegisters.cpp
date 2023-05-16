//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "PleRegisters.hpp"

using namespace ethosn::command_stream::cascading;

namespace ethosn
{
namespace support_library
{

namespace
{

namespace ncu_ple_interface
{

enum class Flags
{
    TOP    = 0x1,
    BOTTOM = 0x2,
    LEFT   = 0x4,
    RIGHT  = 0x8,
};

struct InputInfo
{
    uint16_t dfcAddr;
    int16_t zeroPoint;
    uint16_t multiplier;
    uint16_t shift;
};

struct OutputInfo
{
    uint16_t dfcAddr;
    int16_t zeroPoint;
};

enum class MceOp : uint16_t
{
    CONVOLUTION,
    DEPTHWISE_CONVOLUTION
};

struct alignas(uint32_t) StripeInfo
{
    uint32_t flags;
    std::array<InputInfo, 2> inputs;
    OutputInfo output;
    uint16_t stripeWidth;
    uint16_t stripeHeight;
    uint16_t stripeDepth;
    MceOp mceOp;
};

}    // namespace ncu_ple_interface

uint16_t PleDfcAddr(Tile tile, uint32_t stripeId)
{
    constexpr uint32_t numBytesPerBeat = 16;
    return static_cast<uint16_t>(SramAddr(tile, stripeId) / numBytesPerBeat);
}

}    // namespace

command_stream::cascading::StartPleStripeCommand
    GenerateStartPleStripeCommand(const PleSDesc& pleS, uint32_t agentId, uint32_t stripeId)
{
    StartPleStripeCommand result = {};
    result.type                  = CommandType::StartPleStripe;
    result.agentId               = agentId;
    result.stripeId              = stripeId;

    ncu_ple_interface::StripeInfo pleInfo = {};

    TensorSize stripeCoord;
    stripeCoord.width    = (static_cast<uint32_t>(stripeId) / pleS.stripeIdStrides.width) % pleS.numStripes.width;
    stripeCoord.height   = (static_cast<uint32_t>(stripeId) / pleS.stripeIdStrides.height) % pleS.numStripes.height;
    stripeCoord.channels = (static_cast<uint32_t>(stripeId) / pleS.stripeIdStrides.channels) % pleS.numStripes.channels;

    if (stripeCoord.height == 0)
    {
        pleInfo.flags |= static_cast<uint32_t>(ncu_ple_interface::Flags::TOP);
    }
    if (stripeCoord.height == (pleS.numStripes.height - 1U))
    {
        pleInfo.flags |= static_cast<uint32_t>(ncu_ple_interface::Flags::BOTTOM);
    }
    if (stripeCoord.width == 0)
    {
        pleInfo.flags |= static_cast<uint32_t>(ncu_ple_interface::Flags::LEFT);
    }
    if (stripeCoord.width == (pleS.numStripes.width - 1U))
    {
        pleInfo.flags |= static_cast<uint32_t>(ncu_ple_interface::Flags::RIGHT);
    }

    pleInfo.output.zeroPoint = pleS.ofmZeroPoint;

    TensorSize stripeSize;
    stripeSize.width =
        (stripeCoord.width == (pleS.numStripes.width - 1U)) ? pleS.edgeStripeSize.width : pleS.defaultStripeSize.width;
    stripeSize.height = (stripeCoord.height == (pleS.numStripes.height - 1U)) ? pleS.edgeStripeSize.height
                                                                              : pleS.defaultStripeSize.height;
    stripeSize.channels = (stripeCoord.channels == (pleS.numStripes.channels - 1U)) ? pleS.edgeStripeSize.channels
                                                                                    : pleS.defaultStripeSize.channels;

    pleInfo.stripeWidth  = static_cast<uint16_t>(stripeSize.width);
    pleInfo.stripeHeight = static_cast<uint16_t>(stripeSize.height);
    pleInfo.stripeDepth  = static_cast<uint16_t>(stripeSize.channels);

    pleInfo.output.dfcAddr = PleDfcAddr(pleS.ofmTile, stripeId);

    // specific work according to PLE input: either from SRAM or from MCE
    if (pleS.inputMode == PleInputMode::SRAM_ONE_INPUT || pleS.inputMode == PleInputMode::SRAM_TWO_INPUTS)
    {
        pleInfo.inputs[0].dfcAddr = PleDfcAddr(pleS.ifmTile0, stripeId);
        if (pleS.inputMode == PleInputMode::SRAM_TWO_INPUTS)
        {
            pleInfo.inputs[1].dfcAddr = PleDfcAddr(pleS.ifmTile1, stripeId);
        }
    }
    else
    {
        // PLE takes input from MCE
        static_assert(static_cast<ncu_ple_interface::MceOp>(PleInputMode::MCE_ALL_OGS) ==
                          ncu_ple_interface::MceOp::CONVOLUTION,
                      "");
        static_assert(static_cast<ncu_ple_interface::MceOp>(PleInputMode::MCE_ONE_OG) ==
                          ncu_ple_interface::MceOp::DEPTHWISE_CONVOLUTION,
                      "");
        pleInfo.mceOp = static_cast<ncu_ple_interface::MceOp>(pleS.inputMode);
    }

    pleInfo.inputs[0].zeroPoint = pleS.ifmInfo0.zeroPoint;
    pleInfo.inputs[1].zeroPoint = pleS.ifmInfo1.zeroPoint;

    pleInfo.inputs[0].multiplier = pleS.ifmInfo0.multiplier;
    pleInfo.inputs[1].multiplier = pleS.ifmInfo1.multiplier;

    pleInfo.inputs[0].shift = pleS.ifmInfo0.shift;
    pleInfo.inputs[1].shift = pleS.ifmInfo1.shift;

    // Write PLE struct to PLE scratch registers
    static_assert(sizeof(pleInfo) <= sizeof(result.SCRATCH), "StripeInfo must fit in scratch registers");
    memcpy(&result.SCRATCH[0], &pleInfo, sizeof(pleInfo));
    return result;
}

}    // namespace support_library
}    // namespace ethosn
