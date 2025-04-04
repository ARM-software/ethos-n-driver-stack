//
// Copyright © 2021-2024 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "PleRegisters.hpp"

#include "OpGraph.hpp"
#include "Utils.hpp"

#define ETHOSN_ASSERT_MSG(cond, msg) assert(cond)
#include <ethosn_utils/NumericCast.hpp>

using namespace ethosn::command_stream;
using namespace ethosn::utils;

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
    const uint32_t addr                = SramAddr(tile, stripeId);
    assert(addr % numBytesPerBeat == 0);
    return static_cast<uint16_t>(addr / numBytesPerBeat);
}

}    // namespace

command_stream::StartPleStripeCommand
    GenerateStartPleStripeCommand(const PleSDesc& pleS, uint32_t agentId, uint32_t stripeId)
{
    StartPleStripeCommand result = {};
    result.type                  = CommandType::StartPleStripe;
    result.agentId               = agentId;

    TensorSize stripeCoord;
    stripeCoord.width    = (static_cast<uint32_t>(stripeId) / pleS.stripeIdStrides.width) % pleS.numStripes.width;
    stripeCoord.height   = (static_cast<uint32_t>(stripeId) / pleS.stripeIdStrides.height) % pleS.numStripes.height;
    stripeCoord.channels = (static_cast<uint32_t>(stripeId) / pleS.stripeIdStrides.channels) % pleS.numStripes.channels;

    TensorSize stripeSize;
    stripeSize.width =
        (stripeCoord.width == (pleS.numStripes.width - 1U)) ? pleS.edgeStripeSize.width : pleS.defaultStripeSize.width;
    stripeSize.height = (stripeCoord.height == (pleS.numStripes.height - 1U)) ? pleS.edgeStripeSize.height
                                                                              : pleS.defaultStripeSize.height;
    stripeSize.channels = (stripeCoord.channels == (pleS.numStripes.channels - 1U)) ? pleS.edgeStripeSize.channels
                                                                                    : pleS.defaultStripeSize.channels;

    // Most PLE kernels use a common layout for the scratch registers, but some have their own layout
    if (pleS.m_PleOp->m_Op == PleOperation::MAXPOOL1D)
    {
        result.SCRATCH[0] = stripeSize.width;
        result.SCRATCH[1] = stripeSize.height;
        result.SCRATCH[2] = stripeSize.channels;
        // For valid padding cases, the input size can be larger than the output size in the direction
        // of the pooling, so we send this value separately.
        if (pleS.m_PleOp->m_SelectionIntParams.count("is_direction_x") > 0)
        {
            result.SCRATCH[3] = utils::GetWidth(pleS.m_InputBuffer0->m_TensorShape);
        }
        else if (pleS.m_PleOp->m_SelectionIntParams.count("is_direction_y") > 0)
        {
            result.SCRATCH[3] = utils::GetHeight(pleS.m_InputBuffer0->m_TensorShape);
        }
        result.SCRATCH[4] = SramAddr(pleS.ifmTile0, stripeId);
        result.SCRATCH[5] = SramAddr(pleS.ofmTile, stripeId);
        result.SCRATCH[6] = pleS.m_PleOp->m_RuntimeParams.at("pad_before");
        result.SCRATCH[7] = pleS.m_PleOp->m_RuntimeParams.at("pooling_size");
    }
    else if (pleS.m_PleOp->m_Op == PleOperation::MULTIPLICATION)
    {
        // We encode the stripe size with 16 bits
        // The stripe size should be smaller than this to fit in SRAM anyway so this is just a sanity check.
        assert(stripeSize.height < 0x0000ffff && stripeSize.width < 0x0000ffff && stripeSize.channels < 0x0000ffff);
        result.SCRATCH[0] = (stripeSize.width & 0x0000ffff);
        result.SCRATCH[0] |= ((stripeSize.height & 0x0000ffff) << 16);
        result.SCRATCH[1] = stripeSize.channels & 0x0000ffff;
        result.SCRATCH[2] = pleS.ofmZeroPoint & 0x0000ffff;
        result.SCRATCH[3] = pleS.m_PleOp->m_RuntimeParams.at("overall_multiplier") & 0x0000ffff;
        result.SCRATCH[3] |= (pleS.m_PleOp->m_RuntimeParams.at("overall_shift") << 16);
        result.SCRATCH[4] = pleS.m_PleOp->m_RuntimeParams.at("input0_zeropoint") & 0x0000ffff;
        result.SCRATCH[4] |= (pleS.m_PleOp->m_RuntimeParams.at("input1_zeropoint") << 16);
        result.SCRATCH[5] = SramAddr(pleS.ifmTile0, stripeId);
        result.SCRATCH[6] = SramAddr(pleS.ifmTile1, stripeId);
        result.SCRATCH[7] = SramAddr(pleS.ofmTile, stripeId);
    }
    else
    {
        ncu_ple_interface::StripeInfo pleInfo = {};

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

        pleInfo.stripeWidth  = static_cast<uint16_t>(stripeSize.width);
        pleInfo.stripeHeight = static_cast<uint16_t>(stripeSize.height);
        pleInfo.stripeDepth  = static_cast<uint16_t>(stripeSize.channels);

        pleInfo.output.dfcAddr = PleDfcAddr(pleS.ofmTile, stripeId);

        // For max pooling (odd), we may need to schedule an additional "zero size" stripe at the end so that
        // the PLE kernel can receive the final row of elements from the MCE and use this to complete the pooling
        // for the previous stripe. This messes up the SRAM addresses for PLE outputs, so we ignore these zero size
        // stripes for the purposes of SRAM offsets.
        if (pleS.edgeStripeSize.height == 0)
        {
            uint32_t adjustedStripeId = (stripeId / pleS.numStripes.height) * (pleS.numStripes.height - 1) +
                                        std::min(stripeId % pleS.numStripes.height, pleS.numStripes.height - 2);
            pleInfo.output.dfcAddr = PleDfcAddr(pleS.ofmTile, adjustedStripeId);
        }

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

        if (pleS.m_PleOp->m_RuntimeParams.count("input0_multiplier") > 0)
        {
            pleInfo.inputs[0].multiplier = NumericCast<uint16_t>(pleS.m_PleOp->m_RuntimeParams.at("input0_multiplier"));
        }
        if (pleS.m_PleOp->m_RuntimeParams.count("input0_shift") > 0)
        {
            pleInfo.inputs[0].shift = NumericCast<uint16_t>(pleS.m_PleOp->m_RuntimeParams.at("input0_shift"));
        }
        if (pleS.m_PleOp->m_RuntimeParams.count("input1_multiplier") > 0)
        {
            pleInfo.inputs[1].multiplier = NumericCast<uint16_t>(pleS.m_PleOp->m_RuntimeParams.at("input1_multiplier"));
        }
        if (pleS.m_PleOp->m_RuntimeParams.count("input1_shift") > 0)
        {
            pleInfo.inputs[1].shift = NumericCast<uint16_t>(pleS.m_PleOp->m_RuntimeParams.at("input1_shift"));
        }
        // Write PLE struct to PLE scratch registers
        static_assert(sizeof(pleInfo) <= sizeof(result.SCRATCH), "StripeInfo must fit in scratch registers");
        memcpy(&result.SCRATCH[0], &pleInfo, sizeof(pleInfo));
    }

    return result;
}

}    // namespace support_library
}    // namespace ethosn
