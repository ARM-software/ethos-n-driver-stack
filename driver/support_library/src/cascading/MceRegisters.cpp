//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "MceRegisters.hpp"

#include "../Utils.hpp"
#include "RegistersLayout.hpp"

using namespace ethosn::command_stream::cascading;
using namespace ethosn::support_library::utils;
using namespace ethosn::support_library::registers;

namespace ethosn
{
namespace support_library
{

MceUpsampleType ConvertResizeAlgorithmToCascadingCommand(const ResizeAlgorithm algorithm)
{
    if (algorithm == ResizeAlgorithm::BILINEAR)
    {
        return MceUpsampleType::BILINEAR;
    }
    else if (algorithm == ResizeAlgorithm::NEAREST_NEIGHBOUR)
    {
        return MceUpsampleType::NEAREST_NEIGHBOUR;
    }
    else
    {
        assert(false);
        return MceUpsampleType::OFF;
    }
}

namespace
{

wit_resampling_mode_t GetResamplingMode(const MceUpsampleType& upsampleType)
{
    switch (upsampleType)
    {
        case MceUpsampleType::TRANSPOSE:
        {
            return wit_resampling_mode_t::TRANSPOSE;
        }
        case MceUpsampleType::NEAREST_NEIGHBOUR:
        {
            return wit_resampling_mode_t::NEAREST_NEIGHBOR;
        }
        case MceUpsampleType::BILINEAR:
        {
            return wit_resampling_mode_t::BILINEAR;
        }
        case MceUpsampleType::OFF:
        {
            return wit_resampling_mode_t::NONE;
        }
        default:
        {
            assert(false);
            return wit_resampling_mode_t::NONE;
        }
    }
}

filter_mode_t GetFilterMode(const MceSDesc& cmd)
{
    switch (cmd.mceOpMode)
    {
        case command_stream::cascading::MceOperation::CONVOLUTION:
        {
            return filter_mode_t::FILTER_NXM;
        }
        case command_stream::cascading::MceOperation::DEPTHWISE_CONVOLUTION:
        {
            return filter_mode_t::DEPTHWISE_SEPARABLE;
        }
        case command_stream::cascading::MceOperation::FULLY_CONNECTED:
        {
            return filter_mode_t::VECTOR_PRODUCT;
        }
        default:
        {
            assert(false);
            return filter_mode_t::FILTER_NXM;
        }
    }
}

uint32_t GetNumIfmChannels(bool isFullyConnected, uint32_t dfltIfmStripeChannels, uint32_t currentIfmStripeChannels)
{
    // Weights encoder generates multiple of 1024 input channels for fully connected
    // and for that reason the input needs to be multiple of 8x8x16 (X Y Z) = 1024.
    // Also the weights encoder assumes that all the stripes have same size included
    // the edge ones.
    return isFullyConnected ? RoundUpToNearestMultiple(dfltIfmStripeChannels, 16U) : currentIfmStripeChannels;
}

bool IsStriding(const StrideXy& stride)
{
    return (stride.x > 1U || stride.y > 1U);
}

}    // namespace

command_stream::cascading::ProgramMceExtraData
    GenerateProgramMceExtraData(const MceSDesc& mceS, uint32_t stripeId, const HardwareCapabilities& caps)
{
    ProgramMceExtraData result = {};

    assert((((mceS.convStrideXy.x == 2) && (mceS.convStrideXy.y == 2)) ||
            ((mceS.convStrideXy.x == 1) && (mceS.convStrideXy.y == 1))));

    MceSWorkSize stripeCoord;
    stripeCoord.ofmWidth = (static_cast<uint32_t>(stripeId) / mceS.stripeIdStrides.ofmWidth) % mceS.numStripes.ofmWidth;
    stripeCoord.ofmHeight =
        (static_cast<uint32_t>(stripeId) / mceS.stripeIdStrides.ofmHeight) % mceS.numStripes.ofmHeight;
    stripeCoord.ifmChannels =
        (static_cast<uint32_t>(stripeId) / mceS.stripeIdStrides.ifmChannels) % mceS.numStripes.ifmChannels;
    stripeCoord.ofmChannels =
        (static_cast<uint32_t>(stripeId) / mceS.stripeIdStrides.ofmChannels) % mceS.numStripes.ofmChannels;

    const bool isDepthwise      = mceS.mceOpMode == command_stream::cascading::MceOperation::DEPTHWISE_CONVOLUTION;
    const bool isFullyConnected = mceS.mceOpMode == command_stream::cascading::MceOperation::FULLY_CONNECTED;

    MceSWorkSize isEdge;    // Are we at the right/bottom/back of the tensor
    isEdge.ofmWidth    = (stripeCoord.ofmWidth == (mceS.numStripes.ofmWidth - 1U));
    isEdge.ofmHeight   = (stripeCoord.ofmHeight == (mceS.numStripes.ofmHeight - 1U));
    isEdge.ifmChannels = (stripeCoord.ifmChannels == (mceS.numStripes.ifmChannels - 1U));
    isEdge.ofmChannels = (stripeCoord.ofmChannels == (mceS.numStripes.ofmChannels - 1U));
    if (isDepthwise)
    {
        // For depthwise, the number of IFM channels should always equal the number of OFM channels, but the numStripes
        // iteration can't represent this. Instead, we always have 1 in the ifmChannels dimension, so we have to override
        // the edge calculation here.
        assert(mceS.numStripes.ifmChannels == 1);
        isEdge.ifmChannels = isEdge.ofmChannels;
    }

    MceSWorkSize stripeSize;
    stripeSize.ofmWidth  = (isEdge.ofmWidth) ? mceS.edgeStripeSize.ofmWidth : mceS.defaultStripeSize.ofmWidth;
    stripeSize.ofmHeight = (isEdge.ofmHeight) ? mceS.edgeStripeSize.ofmHeight : mceS.defaultStripeSize.ofmHeight;
    stripeSize.ifmChannels =
        (isEdge.ifmChannels) ? mceS.edgeStripeSize.ifmChannels : mceS.defaultStripeSize.ifmChannels;
    stripeSize.ofmChannels =
        (isEdge.ofmChannels) ? mceS.edgeStripeSize.ofmChannels : mceS.defaultStripeSize.ofmChannels;

    // config CE_STRIP CE_CONTROL and ACTIVATION_CONFIG used when RELU is enabled
    const bool isMacAccOutDisabled = stripeCoord.ifmChannels != (mceS.numStripes.ifmChannels - 1U);
    {
        const int32_t reluMin = mceS.reluActiv.min;
        const int32_t reluMax = mceS.reluActiv.max;

        ce_control_r ceControl;

        ceControl.set_ifm_pad_n_active(mceS.convStrideXy.x * mceS.convStrideXy.y);
        ceControl.set_wide_mul_mode(wide_mul_mode_t::WEIGHT_8_IFM_8);
        ceControl.set_resampling_mode(GetResamplingMode(mceS.upsampleType));
        ceControl.set_horiz_reinterleave_enable(horiz_reinterleave_enable_t::DISABLE);
        ceControl.set_vert_reinterleave_enable(vert_reinterleave_enable_t::DISABLE);
        ceControl.set_upsample_2x_odd_height_enable(
            (mceS.upsampleEdgeMode.row == MceUpsampleEdgeMode::DROP && isEdge.ofmHeight)
                ? wit_upscale_odd_height_enable_t::ENABLE
                : wit_upscale_odd_height_enable_t::DISABLE);
        ceControl.set_upsample_2x_odd_width_enable(
            (mceS.upsampleEdgeMode.col == MceUpsampleEdgeMode::DROP && isEdge.ofmWidth)
                ? wit_upscale_odd_width_enable_t::ENABLE
                : wit_upscale_odd_width_enable_t::DISABLE);
        ceControl.set_wit_broadcast_mode(isDepthwise ? wit_broadcast_mode_t::LOCAL : wit_broadcast_mode_t::ALL);
        ceControl.set_signed_ifm_mode(mceS.isIfmSigned ? signed_ifm_mode_t::ENABLE : signed_ifm_mode_t::DISABLE);
        ceControl.set_winograd_enable(mceS.algorithm == MceAlgorithm::WINOGRAD);

        bool useRelu = false;
        if (mceS.isOfmSigned)
        {
            useRelu = reluMin > -128 || reluMax < 127;
        }
        else
        {
            useRelu = reluMin > 0 || reluMax < 255;
        }
        ceControl.set_relu_enable(static_cast<uint32_t>(useRelu));
        ceControl.set_ofm_bypass_enable(false);
        ceControl.set_mac_acc_clr_disable(stripeCoord.ifmChannels != 0);
        ceControl.set_mac_acc_out_dis(isMacAccOutDisabled);
        ceControl.set_output_ofm_data_type(mceS.isOfmSigned ? output_ofm_data_type_t::INT8
                                                            : output_ofm_data_type_t::UINT8);

        result.CE_CONTROL = ceControl.word;
    }

    // config Mul enable in OGs
    {
        const uint32_t numOgs = caps.GetOgsPerEngine();
        const uint32_t numCes = caps.GetNumberOfEngines();
        const uint32_t numIgs = caps.GetIgsPerEngine();

        if (isDepthwise)
        {
            assert(numCes <= result.MUL_ENABLE.size());

            for (uint32_t ce = 0U; ce < numCes; ++ce)
            {
                // Calculate how many OFMs the CE will generate
                const uint32_t numOfmsForCe = utils::DivRoundUp(std::max(stripeSize.ofmChannels, ce) - ce, numCes);

                // Calculate how many multipliers that will be needed to generate the OFMs
                // Only a subset of the multipliers are used if the CE has more IGs than OFMs to generate.
                const uint32_t numOgMulsToEnable = std::min<uint32_t>(numIgs, numOfmsForCe);
                assert(numOgMulsToEnable <= result.MUL_ENABLE[ce].size());

                // Enable the multipliers needed in the CE's OGs
                for (uint32_t og = 0U; og < numOgMulsToEnable; ++og)
                {
                    const uint32_t mulEnable  = 1U << ((og * numCes) + ce);
                    result.MUL_ENABLE[ce][og] = mulEnable;
                }

                // Disable the rest of the multipliers
                for (uint32_t og = numOgMulsToEnable; og < numOgs; ++og)
                {
                    result.MUL_ENABLE[ce][og] = 0U;
                }
            }
        }
        else
        {
            for (uint32_t og = 0U; og < numOgs; ++og)
            {
                constexpr uint32_t mulEnableAll = 0xFFFFFFFFU;
                for (uint32_t ce = 0U; ce < numCes; ++ce)
                {
                    result.MUL_ENABLE[ce][og] = mulEnableAll;
                }
            }
        }
    }

    {
        const uint32_t numOfSrams  = caps.GetNumberOfSrams();
        const uint32_t groupSizeX  = 8U;
        const uint32_t groupSizeY  = 8U;
        const uint32_t groupStride = utils::DivRoundUp(stripeSize.ifmChannels, numOfSrams) * groupSizeX * groupSizeY;

        {
            const uint32_t defaultNumGroupsX = utils::DivRoundUp(mceS.ifmStripeShapeDefault.width, groupSizeX);
            // Note that we don't use residual slots when packing boundary data in X dimension, so don't
            // need to account for that here.
            const uint32_t residualNumGroupsX = utils::DivRoundUp(mceS.ifmStripeShapeEdge.width, groupSizeX);

            ifm_row_stride_r ifmRowStride;

            ifmRowStride.set_ifm_default_row_stride(defaultNumGroupsX * groupStride);
            ifmRowStride.set_ifm_residual_row_stride(residualNumGroupsX * groupStride);

            result.IFM_ROW_STRIDE = ifmRowStride.word;
        }

        {
            ifm_config1_r ifmConfig1;

            ifmConfig1.set_ifm_group_stride(groupStride);
            ifmConfig1.set_num_ifm_global(
                GetNumIfmChannels(isFullyConnected, mceS.defaultStripeSize.ifmChannels, stripeSize.ifmChannels));

            result.IFM_CONFIG1 = ifmConfig1.word;
        }
    }

    IfmDelta delta[4];
    auto selectDelta = [&](uint32_t stripeIdx, uint32_t numStripes) -> const std::array<IfmDelta, 4>& {
        if (stripeIdx == (numStripes - 1U))
        {
            return mceS.ifmDeltaEdge;
        }
        else if (stripeIdx == (numStripes - 2U))
        {
            return mceS.ifmDeltaOneFromEdge;
        }
        else
        {
            return mceS.ifmDeltaDefault;
        }
    };
    const std::array<IfmDelta, 4>& deltaWidth  = selectDelta(stripeCoord.ofmWidth, mceS.numStripes.ofmWidth);
    const std::array<IfmDelta, 4>& deltaHeight = selectDelta(stripeCoord.ofmHeight, mceS.numStripes.ofmHeight);
    for (uint8_t i = 0; i < 4; i++)
    {
        delta[i].width  = deltaWidth[i].width;
        delta[i].height = deltaHeight[i].height;
    }

    if (mceS.isWideFilter == 0)
    {
        // config all IFM PADx IGx registers
        const uint32_t numIgs = caps.GetIgsPerEngine();
        assert(numIgs <= result.IFM_PAD[0].size());

        for (uint32_t ig = 0; ig < numIgs; ++ig)
        {
            ifm_pad0_ig0_r ifmPad0;
            ifmPad0.set_ifm_stripe_width_delta(delta[0].width);
            ifmPad0.set_ifm_stripe_height_delta(delta[0].height);
            ifmPad0.set_left_pad(mceS.padding[0].left);
            ifmPad0.set_top_pad(mceS.padding[0].top);
            result.IFM_PAD[0][ig] = ifmPad0.word;

            // In case of strided convolution, IFM_PAD1, IFM_PAD2 and IFM_PAD3 must be set as well
            if (IsStriding(mceS.convStrideXy))
            {
                ifm_pad0_ig0_r ifmPad1;
                ifmPad1.set_ifm_stripe_width_delta(delta[1].width);
                ifmPad1.set_ifm_stripe_height_delta(delta[1].height);
                ifmPad1.set_left_pad(mceS.padding[1].left);
                ifmPad1.set_top_pad(mceS.padding[1].top);
                result.IFM_PAD[1][ig] = ifmPad1.word;

                ifm_pad0_ig0_r ifmPad2;
                ifmPad2.set_ifm_stripe_width_delta(delta[2].width);
                ifmPad2.set_ifm_stripe_height_delta(delta[2].height);
                ifmPad2.set_left_pad(mceS.padding[2].left);
                ifmPad2.set_top_pad(mceS.padding[2].top);
                result.IFM_PAD[2][ig] = ifmPad2.word;

                ifm_pad0_ig0_r ifmPad3;
                ifmPad3.set_ifm_stripe_width_delta(delta[3].width);
                ifmPad3.set_ifm_stripe_height_delta(delta[3].height);
                ifmPad3.set_left_pad(mceS.padding[3].left);
                ifmPad3.set_top_pad(mceS.padding[3].top);
                result.IFM_PAD[3][ig] = ifmPad3.word;
            }
        }
    }
    else
    {
        // In wide kernel mode, the CE_STRIPE_WIDE_KERNEL_OFFSET register is used instead of CE_STRIPE_IFM_PAD0_IG0 etc.
        wide_kernel_offset_r wideKernelOffset;
        assert(delta[0].width >= 0 && delta[0].height >= 0 && "Deltas must be positive for wide kernel");
        wideKernelOffset.set_wide_delta_width(static_cast<uint32_t>(delta[0].width));
        wideKernelOffset.set_wide_delta_height(static_cast<uint32_t>(delta[0].height));

        wideKernelOffset.set_wide_filter_offset_w(mceS.padding[0].left);
        wideKernelOffset.set_wide_filter_offset_h(mceS.padding[0].top);

        result.WIDE_KERNEL_OFFSET = wideKernelOffset.word;
    }

    // configure IFM slots
    {
        // There are several different streaming strategies that result in different slot patterns:
        // In the following example diagrams, we assume a tile size of 4 and that the central stripe is stripe 0
        // An X means that the value of that slot is irrelevant because it will not be used.
        //    Single stripe ("strategy 3/1"):
        //        X X X
        //        X 0 X
        //        X X X
        //    Vertical streaming ("strategy 0"):
        //        X 3 X
        //        X 0 X
        //        X 1 X
        //    Horizontal streaming ("strategy 4"):
        //        X X X
        //        3 0 1
        //        X X X
        //    Horizontal and vertical streaming, with re-use of packed boundary data in the X direction ("strategy 6 XY").
        //       Note that the top data comes from the same slot as the mid data, but it's at the bottom of that slot.
        //       Note that the bottom slots are irrelevant because the bottom neighbouring data is included in the mid slot.
        //        3 0 1
        //        3 0 1
        //        X X X
        //    Horizontal and vertical streaming, with re-use of packed boundary data in the Y direction ("strategy 6 YX").
        //       Note that the left data comes from the same slot as the central data, but it's at the right of that slot.
        //       Note that the right slots are irrelevant because the right neighbouring data is included in the central slot.
        //        3 3 X
        //        0 0 X
        //        1 1 X
        //    Horizontal, vertical and IFM depth streaming ("strategy 7").
        //       Note there is no re-use of data between stripes, and all data is packed into a single slot.
        //       The left/top data comes from the same slot as the mid data, but it's at the bottom/right of that slot
        //       The right/bottom slots are irrelevant because the right/bottom neighbouring data is included in the central slot.
        //        0 0 X
        //        0 0 X
        //        X X X
        //
        // The top three cases can all be handled with a single pattern:
        //        X 3 X
        //        3 0 1
        //        X 1 X
        // The bottom three cases are each handled separately and determined by the isPackedBoundaryX/Y flags.
        // We use zero as the value for X, although this is arbitrary.
        //
        // See also the diagrams on DmaCmdState::Region.

        // We don't use residual slots when packing boundary data in X dimension, because this would complicate
        // the configuration and we wouldn't gain anything because we need to use multiple DMA transfers for IFM data
        // anyway.
        const uint32_t isResidualLeft = false;
        const uint32_t isResidualCenter =
            isEdge.ofmWidth && !mceS.isExtraIfmStripeAtRightEdge && !mceS.isPackedBoundaryX;
        const uint32_t isResidualRight =
            stripeCoord.ofmWidth + 1U - (mceS.isExtraIfmStripeAtRightEdge ? 1 : 0) >= mceS.numStripes.ofmWidth - 1U &&
            !mceS.isPackedBoundaryX;

        ifm_top_slots_r ifmTopSlots;
        ifmTopSlots.set_top_left_residual(isResidualLeft);
        ifmTopSlots.set_top_center_residual(isResidualCenter);
        ifmTopSlots.set_top_right_residual(isResidualRight);
        ifm_mid_slots_r ifmMidSlots;
        ifmMidSlots.set_mid_left_residual(isResidualLeft);
        ifmMidSlots.set_mid_center_residual(isResidualCenter);
        ifmMidSlots.set_mid_right_residual(isResidualRight);
        ifm_bottom_slots_r ifmBottomSlots;
        ifmBottomSlots.set_bottom_left_residual(isResidualLeft);
        ifmBottomSlots.set_bottom_center_residual(isResidualCenter);
        ifmBottomSlots.set_bottom_right_residual(isResidualRight);

        uint32_t slotId = static_cast<uint32_t>(stripeId);

        // For strategy 6, skip slots only containing boundary data
        if (mceS.isExtraIfmStripeAtRightEdge && !mceS.isPackedBoundaryX && mceS.isPackedBoundaryY)
        {
            // X first then Y, skip boundary only X slots
            slotId += stripeCoord.ofmHeight;
        }
        else if (mceS.isExtraIfmStripeAtBottomEdge && mceS.isPackedBoundaryX && !mceS.isPackedBoundaryY)
        {
            // Y first then X, skip boundary only Y slots
            slotId += stripeCoord.ofmWidth;
        }

        const uint32_t prev    = (slotId - 1) % mceS.ifmTile.numSlots;
        const uint32_t current = (slotId) % mceS.ifmTile.numSlots;
        const uint32_t next    = (slotId + 1) % mceS.ifmTile.numSlots;

        // Helper function to make the slot setting code below look more natural
        auto setSlots = [&](const std::array<uint32_t, 3>& top, const std::array<uint32_t, 3>& mid,
                            const std::array<uint32_t, 3>& bottom) {
            ifmTopSlots.set_top_left_slot(top[0]);
            ifmTopSlots.set_top_center_slot(top[1]);
            ifmTopSlots.set_top_right_slot(top[2]);

            ifmMidSlots.set_mid_left_slot(mid[0]);
            ifmMidSlots.set_mid_center_slot(mid[1]);
            ifmMidSlots.set_mid_right_slot(mid[2]);

            ifmBottomSlots.set_bottom_left_slot(bottom[0]);
            ifmBottomSlots.set_bottom_center_slot(bottom[1]);
            ifmBottomSlots.set_bottom_right_slot(bottom[2]);
        };

        if (!mceS.isPackedBoundaryX && !mceS.isPackedBoundaryY)
        {
            // Streaming in width or height only (or not at all)
            setSlots(
                // clang-format off
                { 0,         prev,       0    },
                { prev,      current,    next },
                { 0,         next,       0    }    // clang-format on
            );
        }
        else if (!mceS.isPackedBoundaryX && mceS.isPackedBoundaryY)
        {
            // Streaming width and height, X first ("strategy 6 XY")
            setSlots(
                // clang-format off
                { prev,      current,    next },
                { prev,      current,    next },
                { 0,         0,          0    }    // clang-format on
            );
        }
        else if (mceS.isPackedBoundaryX && !mceS.isPackedBoundaryY)
        {
            // Streaming width and height, Y first ("strategy 6 YX")
            setSlots(
                // clang-format off
                { prev,       prev,        0 },
                { current,    current,     0 },
                { next,       next,        0 }    // clang-format on
            );
        }
        else if (mceS.isPackedBoundaryX && mceS.isPackedBoundaryY)
        {
            // Streaming width, height and IFM depth ("strategy 7"). All data in one slot
            setSlots(
                // clang-format off
                { current,   current,  0 },
                { current,   current,  0 },
                { 0,         0,        0 }    // clang-format on
            );
        }

        result.IFM_TOP_SLOTS    = ifmTopSlots.word;
        result.IFM_MID_SLOTS    = ifmMidSlots.word;
        result.IFM_BOTTOM_SLOTS = ifmBottomSlots.word;

        ifm_slot_pad_config_r ifmSlotPad;
        // Slots on the top/left always contain valid data, except when we're on the first row/col of the OFM.
        // Slots on the right/bottom always contain valid data, except when we're on the last row/col of the OFM,
        // however even the last OFM row/col might have valid data to the right/bottom if the IFM has an extra stripe
        // compared to the OFM (a case that can occur with VALID padding).
        ifmSlotPad.set_top_data(stripeCoord.ofmHeight > 0 ? 1 : 0);
        ifmSlotPad.set_bottom_data(!isEdge.ofmHeight || mceS.isExtraIfmStripeAtBottomEdge ? 1 : 0);
        ifmSlotPad.set_left_data(stripeCoord.ofmWidth > 0 ? 1 : 0);
        ifmSlotPad.set_right_data(!isEdge.ofmWidth || mceS.isExtraIfmStripeAtRightEdge ? 1 : 0);
        result.IFM_SLOT_PAD_CONFIG = ifmSlotPad.word;
    }

    // config OFM stripe size
    {
        ofm_stripe_size_r ofmStripeSize;

        ofmStripeSize.set_ofm_stripe_width(stripeSize.ofmWidth);
        ofmStripeSize.set_ofm_stripe_height(stripeSize.ofmHeight);

        result.OFM_STRIPE_SIZE = ofmStripeSize.word;
    }

    // Number of OFM in current stripe being processed by all CEs
    {
        ofm_config_r ofmConfig;
        ofmConfig.set_num_ofm(stripeSize.ofmChannels);

        result.OFM_CONFIG = ofmConfig.word;
    }

    // config all WEIGHT_BASE_ADDR_OGx registers
    {
        // Weights SRAM offset also depends on the number of OFM per SRAM bank.
        // When ogs_per_emcs > 1, some ogs will take weight data from the same sram
        // eg: og0 and og2 will target same sram while og1 and og3 a different one
        const uint32_t numEmcs             = caps.GetNumberofSramsPerEngine();
        const uint32_t numOgs              = caps.GetOgsPerEngine();
        const uint32_t numOgsPerEmc        = numOgs / numEmcs;
        const uint32_t sramSpacePerOg      = mceS.wgtTile.slotSize / numOgsPerEmc;
        const uint32_t weightTileBaseAddr  = mceS.wgtTile.baseAddr;
        const uint32_t weightTileSize      = static_cast<uint32_t>(mceS.wgtTile.numSlots * mceS.wgtTile.slotSize);
        const uint32_t weightStripeSramIdx = mceS.numStripes.ifmChannels == 1
                                                 ? static_cast<uint32_t>(stripeCoord.ofmChannels)
                                                 : static_cast<uint32_t>(stripeId);

        assert(numOgs <= result.WEIGHT_BASE_ADDR.size());
        for (uint32_t og = 0; og < numOgs; ++og)
        {
            const uint32_t ogIdxWithinEmc = og / numEmcs;
            const uint32_t baseAddrOg = SramAddr(mceS.wgtTile, weightStripeSramIdx) + (ogIdxWithinEmc * sramSpacePerOg);

            assert(baseAddrOg <= (weightTileBaseAddr + weightTileSize) && "Weight base address out of tile.");
            ETHOSN_UNUSED(weightTileBaseAddr);
            ETHOSN_UNUSED(weightTileSize);

            weight_base_addr_og0_r weightBaseAddr;

            weightBaseAddr.set_address(baseAddrOg);

            result.WEIGHT_BASE_ADDR[og] = weightBaseAddr.word;
        }
    }

    // Set all CE regs to ifmGlobal initially
    {
        const uint32_t numCes = caps.GetNumberOfEngines();
        const uint32_t numIgs = caps.GetIgsPerEngine();
        {
            ifm_config2_ig0_r ifmConfig2;

            ifmConfig2.set_num_ifm_local(
                GetNumIfmChannels(isFullyConnected, mceS.defaultStripeSize.ifmChannels, stripeSize.ifmChannels));
            assert(numCes <= result.IFM_CONFIG2.size());
            assert(numIgs <= result.IFM_CONFIG2[0].size());
            for (uint32_t ig = 0; ig < numIgs; ++ig)
            {
                for (uint32_t ce = 0; ce < numCes; ++ce)
                {
                    result.IFM_CONFIG2[ce][ig] = ifmConfig2.word;
                }
            }
        }

        // For strided convolutions, ifmLocal needs special config
        if (mceS.convStrideXy.x * mceS.convStrideXy.y > 1U)
        {
            const uint32_t numIfmConsumedPerCe{ numCes * numIgs };

            // ifmGlobal is the number of ifm channels before submap decomposition times num submaps
            // and with extra channels to fill emcs in the last group of % IfmConsumedPerCe channels.
            // If ifmGlobal is not a multiple of numIfmConsumedPerCe, ifmLocal needs to be set differently
            // for igs >= ifmGlobal % numIfmConsumedPerCe.
            // Refer to "MCE specification" section "Usage of IFM parameters" for more details.
            const uint32_t residualIgThreshold = stripeSize.ifmChannels % numIfmConsumedPerCe;

            if (residualIgThreshold != 0U)
            {
                const uint32_t numChannelsPerGroup{ mceS.convStrideXy.x * mceS.convStrideXy.y * numIfmConsumedPerCe };

                const uint32_t ifmLocal = (stripeSize.ifmChannels / numChannelsPerGroup) * numChannelsPerGroup;

                ifm_config2_ig0_r ifmConfig2;

                ifmConfig2.set_num_ifm_local(ifmLocal);

                for (uint32_t ig = residualIgThreshold; ig < numIfmConsumedPerCe; ++ig)
                {
                    const uint32_t ce         = ig % numCes;
                    const uint32_t igWithinCe = ig / numCes;

                    result.IFM_CONFIG2[ce][igWithinCe] = ifmConfig2.word;
                }
            }
        }
    }

    // Record how many blocks we have programmed the MCE to produce. We can't increment
    // m_NumBlocksWaitingForPle yet, as we haven't actually kicked off this stripe yet,
    // however calculating it here and storing it is easier as we have all the relevant
    // variables.
    // Note that we calculate the number of blocks for CE 0 specifically (different CEs
    // may produce different numbers of blocks), as we read the corresponding value from PLE 0.
    if (!isMacAccOutDisabled)
    {
        result.m_NumBlocksProgrammedForMce = utils::DivRoundUp(stripeSize.ofmWidth, mceS.blockSize.width) *
                                             utils::DivRoundUp(stripeSize.ofmHeight, mceS.blockSize.height) *
                                             utils::DivRoundUp(stripeSize.ofmChannels, caps.GetNumberOfEngines());
    }

    return result;
}

command_stream::cascading::StartMceExtraData
    GenerateStartMceExtraData(const MceSDesc& mceS, uint32_t stripeId, const HardwareCapabilities& caps)
{
    StartMceExtraData result = {};

    const bool isFullyConnected = mceS.mceOpMode == command_stream::cascading::MceOperation::FULLY_CONNECTED;

    // For fully connected, assume batch size is 1 (number of mac units enabled scale with batches),
    // which is equivalent to disabling all CEs (see TrySetCeEnables)
    if (isFullyConnected)
    {
        result.CE_ENABLES = 0;
    }
    else
    {

        uint32_t ofmChannels;
        {
            const uint32_t stripeIdStrideZ    = mceS.stripeIdStrides.ofmChannels;
            const uint32_t numStripesZ        = mceS.numStripes.ofmChannels;
            const uint32_t defaultStripeSizeZ = mceS.defaultStripeSize.ofmChannels;
            const uint32_t edgeStripeSizeZ    = mceS.edgeStripeSize.ofmChannels;

            const uint32_t stripeCoordZ = (static_cast<uint32_t>(stripeId) / stripeIdStrideZ) % numStripesZ;

            ofmChannels = (stripeCoordZ == (numStripesZ - 1)) ? edgeStripeSizeZ : defaultStripeSizeZ;
        }

        // Enable as many CEs as there are OFM channels
        result.CE_ENABLES = std::min<uint32_t>(ofmChannels, caps.GetNumberOfEngines());
    }
    return result;
}

command_stream::cascading::MceS CreateMceS(const MceSDesc& mceSDesc)
{
    MceS mceS        = {};
    mceS.mceOpMode   = mceSDesc.mceOpMode;
    mceS.pleKernelId = mceSDesc.pleKernelId;

    // ACTIVATION_CONFIG
    {
        const int32_t reluMin = mceSDesc.reluActiv.min;
        const int32_t reluMax = mceSDesc.reluActiv.max;

        activation_config_r activationConfig;

        // Relu min and relu max values can be negative but arch header file stores the data in 16 bit unsigned format
        // and hence, both of these values have to be truncated.
        activationConfig.set_relu_min(static_cast<uint32_t>(reluMin) & ((1u << 16) - 1));
        activationConfig.set_relu_max(static_cast<uint32_t>(reluMax) & ((1u << 16) - 1));

        mceS.ACTIVATION_CONFIG = activationConfig.word;
    }

    // wide kernel enable/disable
    {
        wide_kernel_control_r wideKernelControl;
        if (mceSDesc.isWideFilter)
        {
            assert(((mceSDesc.algorithm == MceAlgorithm::WINOGRAD &&
                     (mceSDesc.filterShape[0].width > 3 || mceSDesc.filterShape[0].height > 3)) ||
                    (mceSDesc.algorithm == MceAlgorithm::DIRECT &&
                     (mceSDesc.filterShape[0].width > 7 || mceSDesc.filterShape[0].height > 7))) &&
                   "Wide kernel not supported for this filter shape and algorithm");
            assert((mceSDesc.filterShape[0].width == 1 || (mceSDesc.filterShape[0].width % 3) == 0) &&
                   "Wide kernel width invalid");
            assert((mceSDesc.filterShape[0].height == 1 || (mceSDesc.filterShape[0].height % 3) == 0) &&
                   "Wide kernel height invalid");
            wideKernelControl.set_wide_kernel_enable(true);
            wideKernelControl.set_wide_filter_width(mceSDesc.filterShape[0].width);
            wideKernelControl.set_wide_filter_height(mceSDesc.filterShape[0].height);
        }
        mceS.WIDE_KERNEL_CONTROL = wideKernelControl.word;
    }

    // config filter width and height
    {
        // Set the kernel filter mode
        filter_r filter;
        filter.set_filter_mode(GetFilterMode(mceSDesc));

        if (!mceSDesc.isWideFilter)
        {
            // Filter shape (e.g. 3x3). Note that all 4 filters must be set to the same even when
            // ifm_pad_n_active is set to 1, otherwise the HW raises a functional error.
            filter.set_filter0_width(mceSDesc.filterShape[0].width);
            filter.set_filter0_height(mceSDesc.filterShape[0].height);
            filter.set_filter1_width(mceSDesc.filterShape[1].width);
            filter.set_filter1_height(mceSDesc.filterShape[1].height);
            filter.set_filter2_width(mceSDesc.filterShape[2].width);
            filter.set_filter2_height(mceSDesc.filterShape[2].height);
            filter.set_filter3_width(mceSDesc.filterShape[3].width);
            filter.set_filter3_height(mceSDesc.filterShape[3].height);
        }
        // Write the stripe filter register after all of the bits have been set
        mceS.FILTER = filter.word;
    }

    {
        ifm_zero_point_r ifmZeroPoint;

        // Zero point value can be negative but arch header file stores the data in 8 bit unsigned format
        // and hence, zero point value has to be truncated.
        ifmZeroPoint.set_zero_point(static_cast<uint32_t>(mceSDesc.ifmZeroPoint) & ((1u << 8) - 1));

        mceS.IFM_ZERO_POINT = ifmZeroPoint.word;
    }

    {
        // Note that the word 'default' in this register refers to default vs boundary slots, rather
        // than default vs residual slots. We don't use boundary slots anyway.
        // It's therefore important to realize that this is used for width/height of residual slots too
        // (which is confusing, given the name!).
        ifm_default_slot_size_r ifmDefaultSlotSize;

        // Note that we always use the default IFM stripe shape, not the edge IFM stripe shape,
        // because neighbouring slots may be full stripes, and the stripe geometry calculations
        // therefore need to be done with the full stripe shape.
        ifmDefaultSlotSize.set_ifm_default_slot_width(mceSDesc.ifmStripeShapeDefault.width);
        ifmDefaultSlotSize.set_ifm_default_slot_height(mceSDesc.ifmStripeShapeDefault.height);

        mceS.IFM_DEFAULT_SLOT_SIZE = ifmDefaultSlotSize.word;
    }

    {
        ifm_slot_stride_r ifmSlotStride;

        ifmSlotStride.set_ifm_default_slot_stride(mceSDesc.ifmTile.slotSize);

        mceS.IFM_SLOT_STRIDE = ifmSlotStride.word;
    }

    {
        stripe_block_config_r stripeBlockConfig;

        stripeBlockConfig.set_ofm_default_block_width(mceSDesc.blockSize.width);
        stripeBlockConfig.set_ofm_default_block_height(mceSDesc.blockSize.height);
        // From architecture spec for field OFM_BYPASS_HALF_PATCH_OUTPUT_TYPE:
        //   0 (default)   Emit 2x4 half-patches. Use for N x M convolution or Winograd 3x1 convolution
        //   1             Emit 4x2 half-patches. Use for vector product or Winograd 3x3 or 1x3.
        // Set to 0 because we don't use OFM bypass
        stripeBlockConfig.set_ofm_bypass_half_patch_output_type(0U);
        stripeBlockConfig.set_mceif_shuffle_pattern(mceif_shuffle_pattern_t::FLIPPED_N);

        mceS.STRIPE_BLOCK_CONFIG = stripeBlockConfig.word;
    }

    {
        depthwise_control_r depthwiseControl;

        depthwiseControl.set_num_ifms_per_ofm(static_cast<uint32_t>(mceSDesc.convStrideXy.x * mceSDesc.convStrideXy.y));

        mceS.DEPTHWISE_CONTROL = depthwiseControl.word;
    }

    // configure IFM slot IGx base address
    {
        ifm_slot_base_address_ig0_r ifmSlotBaseAddress;
        ifmSlotBaseAddress.set_ifm_slot_base_addr(mceSDesc.ifmTile.baseAddr);
        mceS.IFM_SLOT_BASE_ADDRESS = ifmSlotBaseAddress.word;
    }

    // PLE_MCEIF_CONFIG
    {
        constexpr uint32_t bytesPerElement = 1U;

        constexpr uint32_t INRAM_SIZE  = 1024;
        const uint32_t mceifBufferSize = bytesPerElement * mceSDesc.blockSize.width * mceSDesc.blockSize.height;
        const uint32_t numBuffers      = INRAM_SIZE / mceifBufferSize;

        ple_mceif_config_r pleMceifConfig;
        pleMceifConfig.set_mceif_num_bufs(numBuffers);
        pleMceifConfig.set_mceif_buf_size(mceifBufferSize / 16);
        mceS.PLE_MCEIF_CONFIG = pleMceifConfig.word;
    }

    return mceS;
}

}    // namespace support_library
}    // namespace ethosn
