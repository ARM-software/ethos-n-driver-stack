//
// Copyright © 2021 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"
#include "../src/CapabilitiesInternal.hpp"
#include "../src/cascading/MceEstimationUtils.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>

using namespace ethosn::support_library;
using namespace ethosn::command_stream;
using namespace ethosn::support_library::utils;

#define GENERATE_VARIANT()                                                                                             \
    GENERATE(EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO, EthosNVariant::ETHOS_N78_1TOPS_4PLE_RATIO,                     \
             EthosNVariant::ETHOS_N78_2TOPS_2PLE_RATIO, EthosNVariant::ETHOS_N78_2TOPS_4PLE_RATIO,                     \
             EthosNVariant::ETHOS_N78_4TOPS_2PLE_RATIO, EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO,                     \
             EthosNVariant::ETHOS_N78_8TOPS_2PLE_RATIO)

namespace
{

constexpr uint32_t
    GetNumberOfInputChannels(uint32_t originalInputChannels, uint32_t numberOfEngines, const bool isStrided)
{
    uint32_t inputChannels = originalInputChannels;

    if (isStrided)
    {
        const uint32_t inputChannelsMultipleOfEngines = originalInputChannels % numberOfEngines;

        inputChannels = originalInputChannels * 4;
        if (inputChannelsMultipleOfEngines != 0)
        {
            inputChannels += 3 * (numberOfEngines - inputChannelsMultipleOfEngines);
        }
    }

    return inputChannels;
}

}    // namespace

TEST_CASE("MceStats DepthwiseConvolution", "[Estimation][Mce]")
{
    const uint32_t strideXY              = GENERATE(1, 2);
    const EthosNVariant variant          = GENERATE_VARIANT();
    const uint32_t originalInputChannels = GENERATE(3, 16, 19, 32);

    const HardwareCapabilities caps = GetEthosN78HwCapabilities(variant);
    const uint32_t numberOfEngines  = caps.GetNumberOfEngines();
    const uint32_t numTotIfm        = caps.GetIgsPerEngine() * numberOfEngines;
    const uint32_t halfPatchHeight  = caps.GetPatchShape()[1];
    const uint32_t halfPatchWidth   = utils::DivRoundUp(caps.GetPatchShape()[2], 2u);
    const uint32_t inputChannels    = GetNumberOfInputChannels(originalInputChannels, numberOfEngines, (strideXY != 1));

    const Stride stride(strideXY, strideXY);
    const TensorShape inputShape{ 1, 112, 112, inputChannels };
    const TensorShape outputShape{ 1, 112, 112, 32 };
    const TensorShape weightShape{ 3, 3, 32, 1 };

    MceStats stats = GetMceStats(caps, stride, MceOperation::DEPTHWISE_CONVOLUTION, CompilerMceAlgorithm::Direct,
                                 inputShape, outputShape, weightShape);

    uint32_t cycleCount = weightShape[0] * weightShape[1] * DivRoundUp(outputShape[1], halfPatchHeight) *
                          DivRoundUp(outputShape[2], halfPatchWidth) * DivRoundUp(outputShape[3], numTotIfm);
    // Depthwise convolution.
    INFO("variant: " << EthosNVariantAsString(variant) << ", stride: " << strideXY
                     << ", input channels: " << originalInputChannels);
    REQUIRE(stats.m_CycleCount == cycleCount);
}

TEST_CASE("MceStats Convolution", "[Estimation][Mce]")
{
    const uint32_t strideXY              = 1;
    const EthosNVariant variant          = GENERATE_VARIANT();
    const uint32_t originalInputChannels = GENERATE(3, 16, 19, 32, 256);

    const HardwareCapabilities caps = GetEthosN78HwCapabilities(variant);
    const uint32_t numberOfEngines  = caps.GetNumberOfEngines();
    const uint32_t numTotIfm        = caps.GetIgsPerEngine() * numberOfEngines;
    const uint32_t numTotOfm        = caps.GetOgsPerEngine() * numberOfEngines;
    const uint32_t halfPatchHeight  = caps.GetPatchShape()[1];
    const uint32_t halfPatchWidth   = utils::DivRoundUp(caps.GetPatchShape()[2], 2u);
    const uint32_t inputChannels    = GetNumberOfInputChannels(originalInputChannels, numberOfEngines, (strideXY != 1));

    const Stride stride(strideXY, strideXY);
    const TensorShape inputShape{ 1, 224, 224, inputChannels };
    const TensorShape outputShape{ 1, 448, 448, 64 };
    const TensorShape weightShape{ 3, 3, 3, 32 };

    MceStats stats = GetMceStats(caps, stride, MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct, inputShape,
                                 outputShape, weightShape);

    uint32_t cycleCount = weightShape[0] * weightShape[1] * DivRoundUp(inputShape[3], numTotIfm) *
                          DivRoundUp(outputShape[3], numTotOfm) * DivRoundUp(outputShape[1], halfPatchHeight) *
                          DivRoundUp(outputShape[2], halfPatchWidth);
    // Normal convolution.
    INFO("variant: " << EthosNVariantAsString(variant) << ", stride: " << strideXY
                     << ", input channels: " << originalInputChannels);
    REQUIRE(stats.m_CycleCount == cycleCount);
}

TEST_CASE("MceStats winograd", "[Estimation][Mce]")
{
    // Taking the following test parameters from the performance analysis of Inception V4 on 4TOPS, 4 PLE-RATIO, 1024KB:
    // Input: 17x17x128
    // Output: 17x17x128
    // Weights: 1x9x128x128
    // RTL cycle count : 17280

    const EthosNVariant variant     = EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO;
    const HardwareCapabilities caps = GetEthosN78HwCapabilities(variant);
    const Stride stride{ 1, 1 };
    const TensorShape inputShape{ 1, 17, 17, 128 };
    const TensorShape outputShape{ 1, 17, 17, 128 };
    const TensorShape weightShape{ 1, 9, 128, 128 };

    MceStats stats = GetMceStats(caps, stride, MceOperation::CONVOLUTION, CompilerMceAlgorithm::Winograd, inputShape,
                                 outputShape, weightShape);
    uint32_t cycleCount = 17280;

    REQUIRE(stats.m_CycleCount == cycleCount);
}
