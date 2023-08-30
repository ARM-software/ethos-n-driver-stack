//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "TestUtils.hpp"

#include "../src/MceRegisters.hpp"
#include "../src/RegistersLayout.hpp"

#include <ethosn_command_stream/CommandStream.hpp>

#include <catch.hpp>

using namespace ethosn::support_library;
using namespace ethosn::support_library::registers;
using namespace ethosn::command_stream;

TEST_CASE("MceSDesc/Depthwise")
{
    // Check that registers are setup correctly for depthwise convolution and that the expected
    // number of OG multipliers in the CEs are enabled depending on the variant.

    // Configure the agent data. Note that most of this is irrelevant for this test - we only
    // care about the mceOpMode, convStrideXy and stripe OFM channels
    MceSDesc mceS                     = {};
    mceS.ifmTile.baseAddr             = 0x0U;
    mceS.ifmTile.numSlots             = 2U;
    mceS.ifmTile.slotSize             = 0x100U;
    mceS.wgtTile.baseAddr             = 0x1000U;
    mceS.wgtTile.numSlots             = 2U;
    mceS.wgtTile.slotSize             = 0x100U;
    mceS.blockSize.width              = 8U;
    mceS.blockSize.height             = 8U;
    mceS.defaultStripeSize            = { 16, 16, 16, 16 };
    mceS.edgeStripeSize               = { 16, 16, 16, 16 };
    mceS.numStripes.ofmHeight         = 1U;
    mceS.numStripes.ofmWidth          = 1U;
    mceS.numStripes.ofmChannels       = 1U;
    mceS.numStripes.ifmChannels       = 1U;
    mceS.stripeIdStrides.ofmHeight    = 1U;
    mceS.stripeIdStrides.ofmWidth     = 1U;
    mceS.stripeIdStrides.ofmChannels  = 1U;
    mceS.stripeIdStrides.ifmChannels  = 1U;
    mceS.convStrideXy.x               = 1;
    mceS.convStrideXy.y               = 1;
    mceS.ifmZeroPoint                 = 0;
    mceS.isIfmSigned                  = false;
    mceS.isOfmSigned                  = false;
    mceS.upsampleType                 = MceUpsampleType::OFF;
    mceS.upsampleEdgeMode             = { MceUpsampleEdgeMode::GENERATE, MceUpsampleEdgeMode::GENERATE };
    mceS.mceOpMode                    = MceOperation::DEPTHWISE_CONVOLUTION;
    mceS.algorithm                    = MceAlgorithm::DIRECT;
    mceS.isWideFilter                 = false;
    mceS.isExtraIfmStripeAtRightEdge  = false;
    mceS.isExtraIfmStripeAtBottomEdge = false;
    mceS.isPackedBoundaryX            = false;
    mceS.isPackedBoundaryY            = false;
    mceS.filterShape                  = { { { 1, 1 }, { 1, 1 }, { 1, 1 }, { 1, 1 } } };
    mceS.padding                      = { { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } } };
    mceS.ifmDeltaDefault              = { { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } } };
    mceS.ifmDeltaOneFromEdge          = { { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } } };
    mceS.ifmDeltaEdge                 = { { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } } };
    mceS.ifmStripeShapeDefault        = { 16, 16 };
    mceS.ifmStripeShapeEdge           = { 16, 16 };
    mceS.reluActiv.min                = 0;
    mceS.reluActiv.max                = 255;
    mceS.pleKernelId                  = PleKernelId::V8422_PASSTHROUGH_bw16_bh16_bm1;

    auto CheckEnabledMuls = [&](const HardwareCapabilities& caps, uint32_t ceIdx, uint32_t expectedMulsOg0,
                                uint32_t expectedMulsOg1, uint32_t expectedMulsOg2, uint32_t expectedMulsOg3) {
        MceS agent                   = CreateMceS(mceS);
        ProgramMceStripeCommand data = GenerateProgramMceStripeCommand(mceS, 0, 0, caps);

        ce_control_r ceControl = data.CE_CONTROL;
        CHECK(ceControl.get_wit_broadcast_mode() == wit_broadcast_mode_t::LOCAL);

        depthwise_control_r expDepthwiseControl;
        expDepthwiseControl.set_num_ifms_per_ofm(1U);
        CHECK(agent.DEPTHWISE_CONTROL == expDepthwiseControl.word);

        mul_enable_og0_r expectedMuls(expectedMulsOg0);
        CHECK(data.MUL_ENABLE[ceIdx][0] == expectedMuls.word);

        expectedMuls.set_mul_enable(expectedMulsOg1);
        CHECK(data.MUL_ENABLE[ceIdx][1] == expectedMuls.word);

        expectedMuls.set_mul_enable(expectedMulsOg2);
        CHECK(data.MUL_ENABLE[ceIdx][2] == expectedMuls.word);

        expectedMuls.set_mul_enable(expectedMulsOg3);
        CHECK(data.MUL_ENABLE[ceIdx][3] == expectedMuls.word);
    };

    // Each CE has 1 IG per 2 OG so only half of the OGs will be used in each CE
    const HardwareCapabilities variant2Tops = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_2TOPS_4PLE_RATIO);
    CheckEnabledMuls(variant2Tops, 0, 0x1, 0x10, 0x0, 0x0);
    CheckEnabledMuls(variant2Tops, 1, 0x2, 0x20, 0x0, 0x0);
    CheckEnabledMuls(variant2Tops, 2, 0x4, 0x40, 0x0, 0x0);
    CheckEnabledMuls(variant2Tops, 3, 0x8, 0x80, 0x0, 0x0);

    // Same ratio of IGs and OGs so all OGs can be used in the CEs
    const HardwareCapabilities variant4Tops = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
    CheckEnabledMuls(variant4Tops, 0, 0x1, 0x100, 0x0, 0x0);
    CheckEnabledMuls(variant4Tops, 1, 0x2, 0x200, 0x0, 0x0);
    CheckEnabledMuls(variant4Tops, 2, 0x4, 0x400, 0x0, 0x0);
    CheckEnabledMuls(variant4Tops, 3, 0x8, 0x800, 0x0, 0x0);
    CheckEnabledMuls(variant4Tops, 4, 0x10, 0x1000, 0x0, 0x0);
    CheckEnabledMuls(variant4Tops, 5, 0x20, 0x2000, 0x0, 0x0);
    CheckEnabledMuls(variant4Tops, 6, 0x40, 0x4000, 0x0, 0x0);
    CheckEnabledMuls(variant4Tops, 7, 0x80, 0x8000, 0x0, 0x0);
}

TEST_CASE("MceSDesc/WeightsAddress")
{
    // Check that the weights address is correctly calculated, for a case where we have multiple stripes
    // in the IFM and OFM dimensions.

    // Configure the agent data. Note that most of this is irrelevant for this test - we only
    // care about wgtTile, numStripes and stripeIdStrides.
    MceSDesc mceS                     = {};
    mceS.ifmTile.baseAddr             = 0x0U;
    mceS.ifmTile.numSlots             = 2U;
    mceS.ifmTile.slotSize             = 0x100U;
    mceS.wgtTile.baseAddr             = 0x1000U;
    mceS.wgtTile.numSlots             = 2U;
    mceS.wgtTile.slotSize             = 0x100U;
    mceS.blockSize.width              = 8U;
    mceS.blockSize.height             = 8U;
    mceS.defaultStripeSize            = { 16, 16, 16, 16 };
    mceS.edgeStripeSize               = { 16, 16, 16, 16 };
    mceS.numStripes.ofmHeight         = 1U;
    mceS.numStripes.ofmWidth          = 1U;
    mceS.numStripes.ofmChannels       = 2U;
    mceS.numStripes.ifmChannels       = 2U;
    mceS.stripeIdStrides.ofmHeight    = 2U;
    mceS.stripeIdStrides.ofmWidth     = 2U;
    mceS.stripeIdStrides.ofmChannels  = 1U;
    mceS.stripeIdStrides.ifmChannels  = 1U;
    mceS.convStrideXy.x               = 1;
    mceS.convStrideXy.y               = 1;
    mceS.ifmZeroPoint                 = 0;
    mceS.isIfmSigned                  = false;
    mceS.isOfmSigned                  = false;
    mceS.upsampleType                 = MceUpsampleType::OFF;
    mceS.upsampleEdgeMode             = { MceUpsampleEdgeMode::GENERATE, MceUpsampleEdgeMode::GENERATE };
    mceS.mceOpMode                    = MceOperation::CONVOLUTION;
    mceS.algorithm                    = MceAlgorithm::DIRECT;
    mceS.isWideFilter                 = false;
    mceS.isExtraIfmStripeAtRightEdge  = false;
    mceS.isExtraIfmStripeAtBottomEdge = false;
    mceS.isPackedBoundaryX            = false;
    mceS.isPackedBoundaryY            = false;
    mceS.filterShape                  = { { { 1, 1 }, { 1, 1 }, { 1, 1 }, { 1, 1 } } };
    mceS.padding                      = { { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } } };
    mceS.ifmDeltaDefault              = { { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } } };
    mceS.ifmDeltaOneFromEdge          = { { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } } };
    mceS.ifmDeltaEdge                 = { { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } } };
    mceS.ifmStripeShapeDefault        = { 16, 16 };
    mceS.ifmStripeShapeEdge           = { 16, 16 };
    mceS.reluActiv.min                = 0;
    mceS.reluActiv.max                = 255;
    mceS.pleKernelId                  = PleKernelId::V8422_PASSTHROUGH_bw16_bh16_bm1;

    auto CheckStripe = [&](uint32_t stripeId, uint32_t expectedAddressOg0, uint32_t expectedAddressOg1,
                           uint32_t expectedAddressOg2, uint32_t expectedAddressOg3) {
        const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_2TOPS_4PLE_RATIO);

        ProgramMceStripeCommand data = GenerateProgramMceStripeCommand(mceS, 0, stripeId, caps);

        weight_base_addr_og0_r expected;
        expected.set_address(expectedAddressOg0);
        CHECK(data.WEIGHT_BASE_ADDR[0] == expected.word);

        expected.set_address(expectedAddressOg1);
        CHECK(data.WEIGHT_BASE_ADDR[1] == expected.word);

        expected.set_address(expectedAddressOg2);
        CHECK(data.WEIGHT_BASE_ADDR[2] == expected.word);

        expected.set_address(expectedAddressOg3);
        CHECK(data.WEIGHT_BASE_ADDR[3] == expected.word);
    };

    // 1st stripe is at the start of the weight tile
    CheckStripe(0, 0x1000, 0x1000, 0x1080, 0x1080);
    // 2nd stripe advances along IFM dimension, is the 2nd weight stripe, and will be in the 2nd slot
    CheckStripe(1, 0x1100, 0x1100, 0x1180, 0x1180);
    // 3rd stripe is back to the start in the IFM dimension, but the second element in
    // the OFM dimension, is the 3rd weight stripe, and will be in the 1st slot
    CheckStripe(2, 0x1000, 0x1000, 0x1080, 0x1080);
    // 4th stripe advances along IFM dimension and is still the second element in
    // the OFM dimension, is the 4th weight stripe, and will be in the 2nd slot
    CheckStripe(3, 0x1100, 0x1100, 0x1180, 0x1180);
}

TEST_CASE("MceSDesc/Edge Stripe")
{
    // Check that the stripe shape is correctly set for edge stripes

    // Configure the agent data. Note that most of this is irrelevant for this test - we only
    // care about defaultStripeSize, edgeStripeSize, numStripes and stripeIdStrides.
    MceSDesc mceS                     = {};
    mceS.ifmTile.baseAddr             = 0x0U;
    mceS.ifmTile.numSlots             = 2U;
    mceS.ifmTile.slotSize             = 0x100U;
    mceS.wgtTile.baseAddr             = 0x1000U;
    mceS.wgtTile.numSlots             = 2U;
    mceS.wgtTile.slotSize             = 0x100U;
    mceS.blockSize.width              = 8U;
    mceS.blockSize.height             = 8U;
    mceS.defaultStripeSize            = { 16, 16, 16, 16 };    // H W O I
    mceS.edgeStripeSize               = { 1, 2, 3, 4 };        // H W O I
    mceS.numStripes.ofmHeight         = 5U;
    mceS.numStripes.ofmWidth          = 5U;
    mceS.numStripes.ofmChannels       = 5U;
    mceS.numStripes.ifmChannels       = 5U;
    mceS.stripeIdStrides.ofmHeight    = 125U;
    mceS.stripeIdStrides.ofmWidth     = 25U;
    mceS.stripeIdStrides.ofmChannels  = 5U;
    mceS.stripeIdStrides.ifmChannels  = 1U;
    mceS.convStrideXy.x               = 1;
    mceS.convStrideXy.y               = 1;
    mceS.ifmZeroPoint                 = 0;
    mceS.isIfmSigned                  = false;
    mceS.isOfmSigned                  = false;
    mceS.upsampleType                 = MceUpsampleType::OFF;
    mceS.upsampleEdgeMode             = { MceUpsampleEdgeMode::GENERATE, MceUpsampleEdgeMode::GENERATE };
    mceS.mceOpMode                    = MceOperation::CONVOLUTION;
    mceS.algorithm                    = MceAlgorithm::DIRECT;
    mceS.isWideFilter                 = false;
    mceS.isExtraIfmStripeAtRightEdge  = false;
    mceS.isExtraIfmStripeAtBottomEdge = false;
    mceS.isPackedBoundaryX            = false;
    mceS.isPackedBoundaryY            = false;
    mceS.filterShape                  = { { { 1, 1 }, { 1, 1 }, { 1, 1 }, { 1, 1 } } };
    mceS.padding                      = { { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } } };
    mceS.ifmDeltaDefault              = { { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } } };
    mceS.ifmDeltaOneFromEdge          = { { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } } };
    mceS.ifmDeltaEdge                 = { { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } } };
    mceS.ifmStripeShapeDefault        = { 16, 16 };
    mceS.ifmStripeShapeEdge           = { 2, 1 };
    mceS.reluActiv.min                = 0;
    mceS.reluActiv.max                = 255;
    mceS.pleKernelId                  = PleKernelId::V8422_PASSTHROUGH_bw16_bh16_bm1;

    auto CheckStripe = [&](uint32_t stripeId, uint32_t expectedOfmStripeHeight, uint32_t expectedOfmStripeWidth,
                           uint32_t expectedStripeOfm, uint32_t expectedStripeIfm) {
        const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_2TOPS_4PLE_RATIO);

        ProgramMceStripeCommand data = GenerateProgramMceStripeCommand(mceS, 0, stripeId, caps);

        ofm_stripe_size_r expectedOfmStripeSize;
        expectedOfmStripeSize.set_ofm_stripe_height(expectedOfmStripeHeight);
        expectedOfmStripeSize.set_ofm_stripe_width(expectedOfmStripeWidth);
        CHECK(data.OFM_STRIPE_SIZE == expectedOfmStripeSize.word);

        ofm_config_r expectedOfmConfig;
        expectedOfmConfig.set_num_ofm(expectedStripeOfm);
        CHECK(data.OFM_CONFIG == expectedOfmConfig.word);

        ifm_config1_r actualIfmConfig1(data.IFM_CONFIG1);
        CHECK(actualIfmConfig1.get_num_ifm_global() == expectedStripeIfm);
    };

    // Stripe 0 is at position [H W O I] = [0 0 0 0] and is therefore a full stripe in all dimensions
    CheckStripe(0, 16, 16, 16, 16);
    // Stripe 4 is at position [H W O I] = [0 0 0 4] and is therefore a partial stripe in the I dimension
    CheckStripe(4, 16, 16, 16, 4);
    // Stripe 20 is at position [H W O I] = [0 0 4 0] and is therefore a partial stripe in the O dimension
    CheckStripe(20, 16, 16, 3, 16);
    // Stripe 24 is at position [H W O I] = [0 0 4 4] and is therefore a partial stripe in the O and I dimensions
    CheckStripe(24, 16, 16, 3, 4);
    // Stripe 100 is at position [H W O I] = [0 4 0 0] and is therefore a partial stripe in the W dimension
    CheckStripe(100, 16, 2, 16, 16);
    // Stripe 104 is at position [H W O I] = [0 4 0 4] and is therefore a partial stripe in the I and W dimensions
    CheckStripe(104, 16, 2, 16, 4);
    // Stripe 120 is at position [H W O I] = [0 4 4 0] and is therefore a partial stripe in the O and W dimensions
    CheckStripe(120, 16, 2, 3, 16);
    // Stripe 124 is at position [H W O I] = [0 4 4 4] and is therefore a partial stripe in the O, I and W dimensions
    CheckStripe(124, 16, 2, 3, 4);
    // Stripe 500 is at position [H W O I] = [4 0 0 0] and is therefore a partial stripe in the H dimension
    CheckStripe(500, 1, 16, 16, 16);
    // Stripe 504 is at position [H W O I] = [4 0 0 4] and is therefore a partial stripe in the I and H dimensions
    CheckStripe(504, 1, 16, 16, 4);
    // Stripe 520 is at position [H W O I] = [4 0 4 0] and is therefore a partial stripe in the O and H dimensions
    CheckStripe(520, 1, 16, 3, 16);
    // Stripe 524 is at position [H W O I] = [4 0 4 4] and is therefore a partial stripe in the O, I and H dimensions
    CheckStripe(524, 1, 16, 3, 4);
    // Stripe 600 is at position [H W O I] = [4 4 0 0] and is therefore a partial stripe in the W and H dimensions
    CheckStripe(600, 1, 2, 16, 16);
    // Stripe 604 is at position [H W O I] = [4 4 0 4] and is therefore a partial stripe in the I, W and H dimensions
    CheckStripe(604, 1, 2, 16, 4);
    // Stripe 620 is at position [H W O I] = [4 4 4 0] and is therefore a partial stripe in the O, W and H dimensions
    CheckStripe(620, 1, 2, 3, 16);
    // Stripe 624 is at position [H W O I] = [4 4 4 4] and is therefore a partial stripe in all dimensions
    CheckStripe(624, 1, 2, 3, 4);
}

TEST_CASE("MceSDesc/Slots")
{
    // Check that the slot registers are correctly set

    // Configure the agent data. Note that most of this is irrelevant for this test - we mostly only
    // care about the numStripes and IFM tile size
    MceSDesc mceS                     = {};
    mceS.ifmTile.baseAddr             = 0x0U;
    mceS.ifmTile.numSlots             = 4U;
    mceS.ifmTile.slotSize             = 0x100U;
    mceS.wgtTile.baseAddr             = 0x1000U;
    mceS.wgtTile.numSlots             = 2U;
    mceS.wgtTile.slotSize             = 0x100U;
    mceS.blockSize.width              = 8U;
    mceS.blockSize.height             = 8U;
    mceS.defaultStripeSize            = { 16, 16, 16, 16 };    // H W O I
    mceS.edgeStripeSize               = { 1, 2, 3, 4 };        // H W O I
    mceS.numStripes.ofmHeight         = 5U;
    mceS.numStripes.ofmWidth          = 5U;
    mceS.numStripes.ofmChannels       = 1U;
    mceS.numStripes.ifmChannels       = 1U;
    mceS.stripeIdStrides.ofmHeight    = 5U;
    mceS.stripeIdStrides.ofmWidth     = 1U;
    mceS.stripeIdStrides.ofmChannels  = 1U;
    mceS.stripeIdStrides.ifmChannels  = 1U;
    mceS.convStrideXy.x               = 1;
    mceS.convStrideXy.y               = 1;
    mceS.ifmZeroPoint                 = 0;
    mceS.isIfmSigned                  = false;
    mceS.isOfmSigned                  = false;
    mceS.upsampleType                 = MceUpsampleType::OFF;
    mceS.upsampleEdgeMode             = { MceUpsampleEdgeMode::GENERATE, MceUpsampleEdgeMode::GENERATE };
    mceS.mceOpMode                    = MceOperation::CONVOLUTION;
    mceS.algorithm                    = MceAlgorithm::DIRECT;
    mceS.isWideFilter                 = false;
    mceS.isExtraIfmStripeAtRightEdge  = false;
    mceS.isExtraIfmStripeAtBottomEdge = false;
    mceS.isPackedBoundaryX            = false;
    mceS.isPackedBoundaryY            = false;
    mceS.filterShape                  = { { { 1, 1 }, { 1, 1 }, { 1, 1 }, { 1, 1 } } };
    mceS.padding                      = { { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } } };
    mceS.ifmDeltaDefault              = { { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } } };
    mceS.ifmDeltaOneFromEdge          = { { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } } };
    mceS.ifmDeltaEdge                 = { { { 0, 0 }, { 0, 0 }, { 0, 0 }, { 0, 0 } } };
    mceS.ifmStripeShapeDefault        = { 16, 16 };
    mceS.ifmStripeShapeEdge           = { 2, 1 };
    mceS.reluActiv.min                = 0;
    mceS.reluActiv.max                = 255;
    mceS.pleKernelId                  = PleKernelId::V8422_PASSTHROUGH_bw16_bh16_bm1;

    auto checkStripe =
        [&](uint32_t stripeId, const std::array<uint32_t, 3>& expectedTopSlots,
            const std::array<bool, 3>& expectedTopResidual, const std::array<uint32_t, 3>& expectedMidSlots,
            const std::array<bool, 3>& expectedMidResidual, const std::array<uint32_t, 3>& expectedBotSlots,
            const std::array<bool, 3>& expectedBotResidual) {
            INFO("Stripe " << stripeId);

            const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_2TOPS_4PLE_RATIO);

            ProgramMceStripeCommand data = GenerateProgramMceStripeCommand(mceS, 0, stripeId, caps);

            ifm_top_slots_r expectedTopSlotsReg;
            expectedTopSlotsReg.set_top_left_slot(expectedTopSlots[0]);
            expectedTopSlotsReg.set_top_center_slot(expectedTopSlots[1]);
            expectedTopSlotsReg.set_top_right_slot(expectedTopSlots[2]);
            expectedTopSlotsReg.set_top_left_residual(expectedTopResidual[0]);
            expectedTopSlotsReg.set_top_center_residual(expectedTopResidual[1]);
            expectedTopSlotsReg.set_top_right_residual(expectedTopResidual[2]);
            CHECK(data.IFM_TOP_SLOTS == expectedTopSlotsReg.word);

            ifm_mid_slots_r expectedMidSlotsReg;
            expectedMidSlotsReg.set_mid_left_slot(expectedMidSlots[0]);
            expectedMidSlotsReg.set_mid_center_slot(expectedMidSlots[1]);
            expectedMidSlotsReg.set_mid_right_slot(expectedMidSlots[2]);
            expectedMidSlotsReg.set_mid_left_residual(expectedMidResidual[0]);
            expectedMidSlotsReg.set_mid_center_residual(expectedMidResidual[1]);
            expectedMidSlotsReg.set_mid_right_residual(expectedMidResidual[2]);
            CHECK(data.IFM_MID_SLOTS == expectedMidSlotsReg.word);

            ifm_bottom_slots_r expectedBotSlotsReg;
            expectedBotSlotsReg.set_bottom_left_slot(expectedBotSlots[0]);
            expectedBotSlotsReg.set_bottom_center_slot(expectedBotSlots[1]);
            expectedBotSlotsReg.set_bottom_right_slot(expectedBotSlots[2]);
            expectedBotSlotsReg.set_bottom_left_residual(expectedBotResidual[0]);
            expectedBotSlotsReg.set_bottom_center_residual(expectedBotResidual[1]);
            expectedBotSlotsReg.set_bottom_right_residual(expectedBotResidual[2]);
            CHECK(data.IFM_BOTTOM_SLOTS == expectedBotSlotsReg.word);
        };

    SECTION("No packed boundary data")
    {
        // Stripe 0 uses IFM slot 0 as central, and neighbouring slots are +/-1 modulo the tile size of 4 (i.e. 1 and 3).
        // Even though some of these slots won't be used, we always set them to simplify the code.
        // The residual flags are all false as we are at the far left of the tensor.
        // clang-format off
        checkStripe(0,
            { 0, 3, 0 }, { false, false, false },
            { 3, 0, 1 }, { false, false, false },
            { 0, 1, 0 }, { false, false, false });
        // Stripe 1 uses IFM slot 1 as central, and neighbouring slots are again +/-1 (i.e. 0 and 2)
        checkStripe(1,
            { 0, 0, 0 }, { false, false, false },
            { 0, 1, 2 }, { false, false, false },
            { 0, 2, 0 }, { false, false, false });
        // Same again
        checkStripe(2,
            { 0, 1, 0 }, { false, false, false },
            { 1, 2, 3 }, { false, false, false },
            { 0, 3, 0 }, { false, false, false });
        // And again, this time the 'after' slots (right/bottom) wrap around to zero.
        // The residual flags for the right column are now true, as those slots are at the edge of the tensor.
        checkStripe(3,
            { 0, 2, 0 }, { false, false, true },
            { 2, 3, 0 }, { false, false, true },
            { 0, 0, 0 }, { false, false, true });
        // Central stripe has now wrapped around to zero, so this is identical to stripe 0
        // The residual flags for the centre column are now true, as those slots are at the edge of the tensor.
        checkStripe(4,
            { 0, 3, 0 }, { false, true, true },
            { 3, 0, 1 }, { false, true, true },
            { 0, 1, 0 }, { false, true, true });
        // clang-format on
    }
    SECTION("Packed boundary data X")
    {
        mceS.isPackedBoundaryX = true;
        // Stripe IDs go y first
        mceS.stripeIdStrides.ofmHeight   = 1U;
        mceS.stripeIdStrides.ofmWidth    = 5U;
        mceS.stripeIdStrides.ofmChannels = 1U;
        mceS.stripeIdStrides.ifmChannels = 1U;
        // Stripe 0 uses IFM slot 0 as central.
        // Neighbouring slots above and below are +/-1 modulo the tile size of 4 (i.e. 1 and 3).
        // Neighbouring data to the left and right is packed into the same slot,
        // but right data is included within the slot shape so the right slot is never used (and set to 0 arbitrarily)
        // The residual flags are all false as we are at the far left of the tensor.
        // clang-format off
        checkStripe(0,
            { 3, 3, 0 }, { false, false, false },
            { 0, 0, 0 }, { false, false, false },
            { 1, 1, 0 }, { false, false, false });
        // Stripe 1 uses IFM slot 1 as central, and neighbouring slots above/below are again +/-1 (i.e. 0 and 2)
        checkStripe(1,
            { 0, 0, 0 }, { false, false, false },
            { 1, 1, 0 }, { false, false, false },
            { 2, 2, 0 }, { false, false, false });
        // Same again
        checkStripe(2,
            { 1, 1, 0 }, { false, false, false },
            { 2, 2, 0 }, { false, false, false },
            { 3, 3, 0 }, { false, false, false });
        // And again, this time the 'after' slots (bottom) wrap around to zero.
        checkStripe(3,
            { 2, 2, 0 }, { false, false, false },
            { 3, 3, 0 }, { false, false, false },
            { 0, 0, 0 }, { false, false, false });
        // Central stripe has now wrapped around to zero, so this is identical to stripe 0
        checkStripe(4,
            { 3, 3, 0 }, { false, false, false },
            { 0, 0, 0 }, { false, false, false },
            { 1, 1, 0 }, { false, false, false });
        // Stripe 24 is at position (4, 4) and is the bottom-right stripe.
        // Normally we would set the residual flags for this, but we don't because we're using packed boundary data in X
        checkStripe(24,
            { 3, 3, 0 }, { false, false, false },
            { 0, 0, 0 }, { false, false, false },
            { 1, 1, 0 }, { false, false, false });
        // clang-format on
    }
    SECTION("Packed boundary data Y")
    {
        mceS.isPackedBoundaryY = true;
        // Stripe 0 uses IFM slot 0 as central.
        // Neighbouring slots to the left and right are +/-1 modulo the tile size of 4 (i.e. 1 and 3).
        // Neighbouring data to the top and bottom is packed into the same slot,
        // but bottom data is included within the slot shape so the bottom slot is never used (and set to 0 arbitrarily)
        // The residual flags are all false as we are at the far left of the tensor.
        // clang-format off
        checkStripe(0,
            { 3, 0, 1 }, { false, false, false },
            { 3, 0, 1 }, { false, false, false },
            { 0, 0, 0 }, { false, false, false });
        // Stripe 1 uses IFM slot 1 as central, and neighbouring slots left/right are again +/-1 (i.e. 0 and 2)
        checkStripe(1,
            { 0, 1, 2 }, { false, false, false },
            { 0, 1, 2 }, { false, false, false },
            { 0, 0, 0 }, { false, false, false });
        // Same again
        checkStripe(2,
            { 1, 2, 3 }, { false, false, false },
            { 1, 2, 3 }, { false, false, false },
            { 0, 0, 0 }, { false, false, false });
        // And again, this time the 'after' slots (right) wrap around to zero.
        // The residual flags for the right column are now true, as those slots are at the edge of the tensor.
        checkStripe(3,
            { 2, 3, 0 }, { false, false, true },
            { 2, 3, 0 }, { false, false, true },
            { 0, 0, 0 }, { false, false, true });
        // Central stripe has now wrapped around to zero, so this is identical to stripe 0
        // The residual flags for the centre column are now true, as those slots are at the edge of the tensor.
        checkStripe(4,
            { 3, 0, 1 }, { false, true, true },
            { 3, 0, 1 }, { false, true, true },
            { 0, 0, 0 }, { false, true, true });
        // clang-format on
    }
    SECTION("Packed boundary data X and Y")
    {
        mceS.isPackedBoundaryX = true;
        mceS.isPackedBoundaryY = true;
        // Stripe 0 uses IFM slot 0 as central.
        // Neighbouring slots to all sides are packed into the same slot,
        // but bottom/right data is included within the slot shape so the those slots are never used (and set to 0 arbitrarily)
        // The residual flags are all false as we are at the far left of the tensor.
        // clang-format off
        checkStripe(0,
            { 0, 0, 0 }, { false, false, false },
            { 0, 0, 0 }, { false, false, false },
            { 0, 0, 0 }, { false, false, false });
        // Stripe 1 uses IFM slot 1
        checkStripe(1,
            { 1, 1, 0 }, { false, false, false },
            { 1, 1, 0 }, { false, false, false },
            { 0, 0, 0 }, { false, false, false });
        // Same again
        checkStripe(2,
            { 2, 2, 0 }, { false, false, false },
            { 2, 2, 0 }, { false, false, false },
            { 0, 0, 0 }, { false, false, false });
        // And again
        // Normally we would set the residual flags for this, but we don't because we're using packed boundary data in X
        checkStripe(3,
            { 3, 3, 0 }, { false, false, false },
            { 3, 3, 0 }, { false, false, false },
            { 0, 0, 0 }, { false, false, false });
        // Central stripe has now wrapped around to zero, so this is identical to stripe 0
        // Normally we would set the residual flags for this, but we don't because we're using packed boundary data in X
        checkStripe(4,
            { 0, 0, 0 }, { false, false, false },
            { 0, 0, 0 }, { false, false, false },
            { 0, 0, 0 }, { false, false, false });
        // clang-format on
    }
}
