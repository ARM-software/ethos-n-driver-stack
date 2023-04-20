//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"
#include "../include/ethosn_support_library/SupportQueries.hpp"
#include "../src/Compiler.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>
#include <ethosn_command_stream/CommandStreamBuffer.hpp>

using namespace ethosn::support_library;

TEST_CASE("FullyConnectedSupported")
{
    char reason[1024];

    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    SECTION("InputInfo is not UINT8_QUANTIZED.")
    {
        TensorInfo inputNotUint8Quant({ 1, 1, 1, 4096 }, DataType::INT32_QUANTIZED, DataFormat::NHWC,
                                      QuantizationInfo(0, 1.0f));
        REQUIRE(queries.IsFullyConnectedSupported(TensorInfo(), TensorInfo(), FullyConnectedInfo(), inputNotUint8Quant,
                                                  nullptr, reason, sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "UINT8_QUANTIZED"));
    }

    SECTION("Invalid input data format")
    {
        TensorInfo inputInvalidFormat({ 1, 1, 1, 4096 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO,
                                      QuantizationInfo(0, 1.0f));
        REQUIRE(queries.IsFullyConnectedSupported(TensorInfo(), TensorInfo(), FullyConnectedInfo(), inputInvalidFormat,
                                                  nullptr, reason, sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Only NHWC and NHWCB"));
    }

    SECTION("Invalid weights data type")
    {
        TensorInfo input({ 1, 1, 1, 4096 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo weightsInvalidType{ { 1, 1, 4096, 1000 }, DataType::INT32_QUANTIZED, DataFormat::HWIO, { 0, 1.0f } };
        TensorInfo bias;
        FullyConnectedInfo fcInfo;
        REQUIRE(queries.IsFullyConnectedSupported(bias, weightsInvalidType, fcInfo, input, nullptr, reason,
                                                  sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Weights for fully connected must be UINT8_QUANTIZED"));
    }

    SECTION("Invalid weights data format")
    {
        TensorInfo input({ 1, 1, 1, 4096 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo weightsInvalidFormat{
            { 1, 1, 4096, 1000 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, { 0, 1.0f }
        };
        TensorInfo bias;
        FullyConnectedInfo fcInfo;
        REQUIRE(queries.IsFullyConnectedSupported(bias, weightsInvalidFormat, fcInfo, input, nullptr, reason,
                                                  sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Weights for fully connected must be HWIO"));
    }

    SECTION("Weights invalid W")
    {
        TensorInfo input({ 1, 1, 1, 4096 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo weightsInvalidW{ { 1, 2, 4096, 1000 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, { 0, 1.0f } };
        TensorInfo bias;
        FullyConnectedInfo fcInfo;
        REQUIRE(queries.IsFullyConnectedSupported(bias, weightsInvalidW, fcInfo, input, nullptr, reason,
                                                  sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Weights tensor must have H and W set to 1 as these dimensions are not needed."));
    }

    SECTION("Weights invalid H")
    {
        TensorInfo input({ 1, 1, 1, 4096 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo weightsInvalidH{ { 2, 1, 4096, 1000 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, { 0, 1.0f } };
        TensorInfo bias;
        FullyConnectedInfo fcInfo;
        REQUIRE(queries.IsFullyConnectedSupported(bias, weightsInvalidH, fcInfo, input, nullptr, reason,
                                                  sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Weights tensor must have H and W set to 1 as these dimensions are not needed."));
    }

    SECTION("Weights invalid I")
    {
        TensorInfo input({ 1, 1, 1, 4096 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo weightsInvalidI{ { 1, 1, 4097, 1000 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, { 0, 1.0f } };
        TensorInfo bias;
        FullyConnectedInfo fcInfo;
        REQUIRE(queries.IsFullyConnectedSupported(bias, weightsInvalidI, fcInfo, input, nullptr, reason,
                                                  sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason,
                         "Weights tensor must have I dimension equal to the number of channels of the input tensor."));
    }

    SECTION("Invalid bias data type")
    {
        TensorInfo input({ 1, 1, 1, 4096 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo biasInvalidDataType{ { 1, 1, 1, 1000 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, { 0, 1.0f } };
        TensorInfo weights{ { 1, 1, 4096, 1000 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, { 0, 1.0f } };
        FullyConnectedInfo fcInfo;
        REQUIRE(queries.IsFullyConnectedSupported(biasInvalidDataType, weights, fcInfo, input, nullptr, reason,
                                                  sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Bias for fully connected must be INT32_QUANTIZED"));
    }

    SECTION("Invalid bias data format")
    {
        TensorInfo input({ 1, 1, 1, 4096 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo weights{ { 1, 1, 4096, 1000 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, { 0, 1.0f } };
        TensorInfo biasInvalidDataFormat{ { 1, 1, 1, 1000 }, DataType::INT32_QUANTIZED, DataFormat::HWIO, { 0, 1.0f } };
        FullyConnectedInfo fcInfo;
        REQUIRE(queries.IsFullyConnectedSupported(biasInvalidDataFormat, weights, fcInfo, input, nullptr, reason,
                                                  sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Bias for fully connected must be NHWC"));
    }

    SECTION("Invalid bias shape")
    {
        TensorInfo input({ 1, 1, 1, 4096 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo weights{ { 1, 1, 4096, 1000 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, { 0, 1.0f } };
        TensorInfo bias{ { 1, 2, 3, 4 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, 1.0f } };
        FullyConnectedInfo fcInfo;
        REQUIRE(queries.IsFullyConnectedSupported(bias, weights, fcInfo, input, nullptr, reason, sizeof(reason)) ==
                SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Invalid bias tensor dimensions"));
    }

    SECTION("Output info incorrect")
    {
        TensorInfo input({ 1, 1, 1, 4096 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo weights{ { 1, 1, 4096, 1000 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, { 0, 1.0f } };
        TensorInfo bias{ { 1, 1, 1, 1000 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, 1.0f } };
        FullyConnectedInfo fcInfo;
        TensorInfo output = TensorInfo({ 1, 2, 3, 4 });
        REQUIRE(queries.IsFullyConnectedSupported(bias, weights, fcInfo, input, &output, reason, sizeof(reason)) ==
                SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Provided outputInfo is incorrect"));
    }

    SECTION("Invalid zero point range")
    {
        TensorInfo weights{ { 1, 1, 4096, 1000 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, { -10, 1.0f } };
        TensorInfo input({ 1, 1, 1, 4096 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo bias{ { 1, 1, 1, 1000 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, 1.0f } };
        FullyConnectedInfo fcInfo;
        REQUIRE(queries.IsFullyConnectedSupported(bias, weights, fcInfo, input, nullptr, reason, sizeof(reason)) ==
                SupportedLevel::EstimateOnly);
        REQUIRE(Contains(reason, "Zero point out of range for weights info"));

        weights.m_QuantizationInfo.SetZeroPoint(0);
        input.m_QuantizationInfo.SetZeroPoint(-10);
        REQUIRE(queries.IsFullyConnectedSupported(bias, weights, fcInfo, input, nullptr, reason, sizeof(reason)) ==
                SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Zero point out of range for input info"));

        input.m_QuantizationInfo.SetZeroPoint(0);
        fcInfo.m_OutputQuantizationInfo.SetZeroPoint(-10);
        REQUIRE(queries.IsFullyConnectedSupported(bias, weights, fcInfo, input, nullptr, reason, sizeof(reason)) ==
                SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Zero point out of range for fullyConnectedInfo"));
    }

    SECTION("EstimateOnly for implicit reshape on input")
    {
        TensorInfo input({ 1, 8, 8, 5 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo weights{ { 1, 1, 320, 1000 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, { 0, 1.0f } };
        TensorInfo bias{ { 1, 1, 1, 1000 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, 1.0f } };
        FullyConnectedInfo fcInfo;
        REQUIRE(queries.IsFullyConnectedSupported(bias, weights, fcInfo, input, nullptr, reason, sizeof(reason)) ==
                SupportedLevel::EstimateOnly);
        REQUIRE(Contains(reason, "one dimensional"));
    }

    SECTION("Estimate only for bias quant scale mismatch")
    {
        TensorInfo weights{ { 1, 1, 4096, 1000 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, { 0, 1.0f } };
        TensorInfo input({ 1, 1, 1, 4096 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo bias{ { 1, 1, 1, 1000 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, 99.0f } };
        FullyConnectedInfo fcInfo;
        REQUIRE(queries.IsFullyConnectedSupported(bias, weights, fcInfo, input, nullptr, reason, sizeof(reason)) ==
                SupportedLevel::EstimateOnly);
        REQUIRE(Contains(reason, "Bias for fully connected must have quantization parameters"));
    }

    SECTION("Estimate only for overall multiplier out of range")
    {
        TensorInfo weights{ { 1, 1, 4096, 1000 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, { 0, 65540.0f } };
        TensorInfo input({ 1, 1, 1, 4096 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo bias{ { 1, 1, 1, 1000 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, 65540.0f } };
        FullyConnectedInfo fcInfo({ 0, 1.0f });
        REQUIRE(queries.IsFullyConnectedSupported(bias, weights, fcInfo, input, nullptr, reason, sizeof(reason)) ==
                SupportedLevel::EstimateOnly);
        REQUIRE(Contains(reason, "Overall scale"));
    }

    SECTION("Successful case")
    {
        const auto inputDataType = GENERATE(DataType::INT8_QUANTIZED, DataType::UINT8_QUANTIZED);

        float weightScale = 1.0f / (16 * 16 * 16 * 8);
        TensorInfo weights{ { 1, 1, 4096, 1000 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, { 0, weightScale } };
        TensorInfo input({ 1, 1, 1, 4096 }, inputDataType, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo bias{ { 1, 1, 1, 1000 },
                         DataType::INT32_QUANTIZED,
                         DataFormat::NHWC,
                         { 0, weightScale * input.m_QuantizationInfo.GetScale() } };
        TensorInfo output = TensorInfo();
        FullyConnectedInfo fcInfo;
        REQUIRE(queries.IsFullyConnectedSupported(bias, weights, fcInfo, input, &output) == SupportedLevel::Supported);
    }
}
