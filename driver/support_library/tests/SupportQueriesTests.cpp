//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"
#include "../include/ethosn_support_library/SupportQueries.hpp"
#include "../src/Network.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>

#include <cstring>
#include <iostream>
#include <unordered_set>

using namespace ethosn::support_library;

TEST_CASE("InputSupported", "[IsSupported]")
{
    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N57));

    TensorInfo info({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
    REQUIRE(queries.IsInputSupported(info) == SupportedLevel::Supported);
}

TEST_CASE("OutputSupported", "[IsSupported]")
{
    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N57));

    TensorInfo info({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
    REQUIRE(queries.IsOutputSupported(info, DataFormat::NHWC) == SupportedLevel::Supported);
}

TEST_CASE("OutputSupportedNHWCB", "[IsSupported]")
{
    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N57));

    TensorInfo info({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
    REQUIRE(queries.IsOutputSupported(info, DataFormat::NHWCB) == SupportedLevel::Supported);
}

SCENARIO("With QuantizationDim", "[IsSupported]")
{
    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N57));

    char reason[1024];
    QuantizationInfo quantInfo(0, 1.0f);
    quantInfo.SetQuantizationDim(3);

    GIVEN("An Input TensorInfo with QuantizationDim set")
    {
        TensorInfo inputInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, quantInfo);

        WHEN("Checking if supported as Input")
        {
            THEN("Input shall not be supported")
            {
                REQUIRE(queries.IsInputSupported(inputInfo, nullptr, reason, sizeof(reason)) ==
                        SupportedLevel::Unsupported);
                INFO(reason);
                REQUIRE(Contains(reason, "Quantization Dim should not be used on Input"));
            }
        }
    }
}

constexpr const uint32_t UNSUPPORTED_OUTPUT_DIM = 33;
constexpr const uint32_t UNSUPPORTED_WEIGHT_DIM = 64;
constexpr const uint32_t INPUT_DIM              = 32;
constexpr const uint32_t OUTPUT_DIM             = 32;
constexpr const uint32_t TOTAL_SRAM             = 2048;

#define CHECK_UNSUPPORTED_TENSOR_DEPTH_REASON(reason)                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        INFO(reason);                                                                                                  \
        REQUIRE(Contains(reason, "Tensor max depth cannot fit in SRAM"));                                              \
    } while (0)

#define CHECK_UNSUPPORTED(result, reason)                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        SupportedLevel level = result;                                                                                 \
        INFO(reason);                                                                                                  \
        CHECK(level == SupportedLevel::Unsupported);                                                                   \
    } while (0)

TEST_CASE("Unsupported Tensor Depth", "[IsSupported][TVM]")
{
    char reason[1024] = {
        0,
    };
    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N57, TOTAL_SRAM));
    TensorInfo inputInfo({ 1, 16, 16, UNSUPPORTED_OUTPUT_DIM }, DataType::UINT8_QUANTIZED, DataFormat::NHWC,
                         QuantizationInfo(0, 1.0f));
    std::vector<TensorInfo> outputs(2);

    SECTION("Input")
    {
        CHECK_UNSUPPORTED(queries.IsInputSupported(inputInfo, &outputs[0], reason, sizeof(reason)), reason);
        CHECK_UNSUPPORTED_TENSOR_DEPTH_REASON(reason);
    }
    SECTION("Output")
    {
        CHECK_UNSUPPORTED(queries.IsOutputSupported(inputInfo, DataFormat::NHWC, reason, sizeof(reason)), reason);
        CHECK_UNSUPPORTED_TENSOR_DEPTH_REASON(reason);
    }
    SECTION("Convolution")
    {
        // Generate 2 tests with invalid tensor depth:
        // - Unsupported caused by input
        // - Unsupported caused by weights
        // clang-format off
        auto inputShape = GENERATE(
            TensorShape{ 1, 16, 16, UNSUPPORTED_OUTPUT_DIM },
            TensorShape{ 1, 16, 16, OUTPUT_DIM }
        );
        // clang-format on

        TensorInfo convInputInfo(inputShape, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo biasInfo({ 1, 1, 1, UNSUPPORTED_WEIGHT_DIM }, DataType::INT32_QUANTIZED, DataFormat::NHWC,
                            QuantizationInfo(0, 0.9f));
        TensorInfo weightInfo({ 1, 1, INPUT_DIM, UNSUPPORTED_WEIGHT_DIM }, DataType::UINT8_QUANTIZED, DataFormat::HWIO,
                              QuantizationInfo(0, 0.9f));
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 1, 1 });

        CHECK_UNSUPPORTED(queries.IsConvolutionSupported(biasInfo, weightInfo, convInfo, convInputInfo, &outputs[0],
                                                         reason, sizeof(reason)),
                          reason);
        CHECK_UNSUPPORTED_TENSOR_DEPTH_REASON(reason);
    }
    SECTION("DepthwiseConvolution")
    {
        // Generate 2 tests with invalid tensor depth:
        // - Unsupported caused by input
        // - Unsupported caused by weights
        // clang-format off
        auto inputShape = GENERATE(
            TensorShape{ 1, 16, 16, UNSUPPORTED_OUTPUT_DIM },
            TensorShape{ 1, 16, 16, OUTPUT_DIM }
        );
        // clang-format on

        TensorInfo convInputInfo(inputShape, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo biasInfo({ 1, 1, 1, OUTPUT_DIM * 2 }, DataType::INT32_QUANTIZED, DataFormat::NHWC,
                            QuantizationInfo(0, 0.9f));
        TensorInfo weightInfo({ 1, 1, OUTPUT_DIM, 2 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM,
                              QuantizationInfo(0, 0.9f));
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 1, 1 });

        CHECK_UNSUPPORTED(queries.IsDepthwiseConvolutionSupported(biasInfo, weightInfo, convInfo, convInputInfo,
                                                                  &outputs[0], reason, sizeof(reason)),
                          reason);
        CHECK_UNSUPPORTED_TENSOR_DEPTH_REASON(reason);
    }
    SECTION("TransposeConvolution")
    {
        // Generate 2 tests with invalid tensor depth:
        // - Unsupported caused by input
        // - Unsupported caused by weights
        // clang-format off
        auto inputShape = GENERATE(
            TensorShape{ 1, 16, 16, UNSUPPORTED_OUTPUT_DIM },
            TensorShape{ 1, 16, 16, OUTPUT_DIM }
        );
        // clang-format on

        TensorInfo convInputInfo(inputShape, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo biasInfo({ 1, 1, 1, UNSUPPORTED_WEIGHT_DIM }, DataType::INT32_QUANTIZED, DataFormat::NHWC,
                            QuantizationInfo(0, 0.9f));
        TensorInfo weightInfo({ 1, 1, INPUT_DIM, UNSUPPORTED_WEIGHT_DIM }, DataType::UINT8_QUANTIZED, DataFormat::HWIO,
                              QuantizationInfo(0, 0.9f));
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 2, 2 });

        CHECK_UNSUPPORTED(queries.IsTransposeConvolutionSupported(biasInfo, weightInfo, convInfo, convInputInfo,
                                                                  &outputs[0], reason, sizeof(reason)),
                          reason);
        CHECK_UNSUPPORTED_TENSOR_DEPTH_REASON(reason);
    }
    SECTION("Concatenation")
    {
        // Generate 3 tests with invalid tensor depth:
        // - Unsupported caused by inputInfo1
        // - Unsupported caused by inputInfo2
        // - Unsupported caused by output
        // clang-format off
        const auto shapes = GENERATE(
            //                          Input 1                                Input 2
            std::vector<TensorShape>{ { 1, 16, 16, UNSUPPORTED_OUTPUT_DIM }, { 1, 16, 16, OUTPUT_DIM } },
            std::vector<TensorShape>{ { 1, 16, 16, OUTPUT_DIM },             { 1, 16, 16, UNSUPPORTED_OUTPUT_DIM } },
            std::vector<TensorShape>{ { 1, 16, 16, (OUTPUT_DIM / 2) + 1 },   { 1, 16, 16, OUTPUT_DIM / 2 } }
        );
        // clang-format on

        TensorInfo inputInfo1(shapes[0], DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo());
        TensorInfo inputInfo2(shapes[1], DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo());
        ConcatenationInfo concatInfo(3, QuantizationInfo());

        CHECK_UNSUPPORTED(queries.IsConcatenationSupported({ inputInfo1, inputInfo2 }, concatInfo, &outputs[0], reason,
                                                           sizeof(reason)),
                          reason);
        CHECK_UNSUPPORTED_TENSOR_DEPTH_REASON(reason);
    }
    SECTION("Split")
    {
        SplitInfo splitInfo(3, { INPUT_DIM / 2, INPUT_DIM / 2 });

        CHECK_UNSUPPORTED(queries.IsSplitSupported(inputInfo, splitInfo, &outputs, reason, sizeof(reason)), reason);
        CHECK_UNSUPPORTED_TENSOR_DEPTH_REASON(reason);
    }
    SECTION("Addition")
    {
        // Generate 2 tests with invalid tensor depth:
        // - Unsupported caused by inputInfo1
        // - Unsupported caused by inputInfo2
        // clang-format off
        const auto shapes = GENERATE(
            //                          Input 1                                Input 2
            std::vector<TensorShape>{ { 1, 16, 16, UNSUPPORTED_OUTPUT_DIM }, { 1, 16, 16, OUTPUT_DIM } },
            std::vector<TensorShape>{ { 1, 16, 16, OUTPUT_DIM },             { 1, 16, 16, UNSUPPORTED_OUTPUT_DIM } }
        );
        // clang-format on

        TensorInfo inputInfo1(shapes[0], DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo());
        TensorInfo inputInfo2(shapes[1], DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo());

        CHECK_UNSUPPORTED(queries.IsAdditionSupported(inputInfo1, inputInfo2, QuantizationInfo(), &outputs[0], reason,
                                                      sizeof(reason)),
                          reason);
        CHECK_UNSUPPORTED_TENSOR_DEPTH_REASON(reason);
    }
    SECTION("FullyConnected")
    {
        // Note:
        //   Cannot test the output tensor for now as we always have Width dimention = 1 for
        //   FullyConnected output. Wich bypasses the tensor depth check.
        // clang-format off
        auto inputShape = TensorShape{ 1, 2, 2, UNSUPPORTED_OUTPUT_DIM };
        // clang-format on

        TensorInfo fullConnInputInfo(inputShape, DataType::UINT8_QUANTIZED, DataFormat::NHWC,
                                     QuantizationInfo(0, 1.0f));
        TensorInfo biasInfo({ 1, 1, 1, UNSUPPORTED_WEIGHT_DIM }, DataType::INT32_QUANTIZED, DataFormat::NHWC,
                            QuantizationInfo(0, 0.9f));
        TensorInfo weightInfo({ 1, 1, inputShape[3], UNSUPPORTED_WEIGHT_DIM }, DataType::UINT8_QUANTIZED,
                              DataFormat::HWIO, QuantizationInfo(0, 0.9f));

        CHECK_UNSUPPORTED(queries.IsFullyConnectedSupported(biasInfo, weightInfo, FullyConnectedInfo(),
                                                            fullConnInputInfo, &outputs[0], reason, sizeof(reason)),
                          reason);
        CHECK_UNSUPPORTED_TENSOR_DEPTH_REASON(reason);
    }
    SECTION("Relu")
    {
        CHECK_UNSUPPORTED(queries.IsReluSupported(ReluInfo(), inputInfo, &outputs[0], reason, sizeof(reason)), reason);
        CHECK_UNSUPPORTED_TENSOR_DEPTH_REASON(reason);
    }
    SECTION("LeakyRelu")
    {
        CHECK_UNSUPPORTED(queries.IsLeakyReluSupported(LeakyReluInfo(), inputInfo, &outputs[0], reason, sizeof(reason)),
                          reason);
        CHECK_UNSUPPORTED_TENSOR_DEPTH_REASON(reason);
    }
    SECTION("Requantize")
    {
        CHECK_UNSUPPORTED(
            queries.IsRequantizeSupported(RequantizeInfo(), inputInfo, &outputs[0], reason, sizeof(reason)), reason);
        CHECK_UNSUPPORTED_TENSOR_DEPTH_REASON(reason);
    }
    SECTION("Sigmoid")
    {
        CHECK_UNSUPPORTED(queries.IsSigmoidSupported(inputInfo, &outputs[0], reason, sizeof(reason)), reason);
        CHECK_UNSUPPORTED_TENSOR_DEPTH_REASON(reason);
    }
    SECTION("Pooling")
    {
        PoolingInfo poolingInfo(2, 2, 2, 2, { 0, 0, 0, 0 }, PoolingType::MAX);

        CHECK_UNSUPPORTED(queries.IsPoolingSupported(poolingInfo, inputInfo, &outputs[0], reason, sizeof(reason)),
                          reason);
        CHECK_UNSUPPORTED_TENSOR_DEPTH_REASON(reason);
    }
    SECTION("MeanXy")
    {
        CHECK_UNSUPPORTED(queries.IsMeanXySupported(inputInfo, &outputs[0], reason, sizeof(reason)), reason);
        CHECK_UNSUPPORTED_TENSOR_DEPTH_REASON(reason);
    }
    SECTION("Reshape")
    {
        // Generate 2 tests with invalid tensor depth:
        // - Unsupported caused by input
        // - Unsupported caused by new dimensions
        // clang-format off
        const auto shapes = GENERATE(
            //                          Input                                          New shape
            std::vector<TensorShape>{ { 1, 16, 16, UNSUPPORTED_OUTPUT_DIM },         { 1, 16, UNSUPPORTED_OUTPUT_DIM, 16 } },
            std::vector<TensorShape>{ { 1, 16, UNSUPPORTED_OUTPUT_DIM, OUTPUT_DIM }, { 1, 16, OUTPUT_DIM, UNSUPPORTED_OUTPUT_DIM } }
        );
        // clang-format on

        TensorInfo reshapeInputInfo(shapes[0], DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));

        CHECK_UNSUPPORTED(queries.IsReshapeSupported(shapes[1], reshapeInputInfo, &outputs[0], reason, sizeof(reason)),
                          reason);
        CHECK_UNSUPPORTED_TENSOR_DEPTH_REASON(reason);
    }
    SECTION("DepthToSpace")
    {
        CHECK_UNSUPPORTED(
            queries.IsDepthToSpaceSupported(inputInfo, DepthToSpaceInfo(2), &outputs[0], reason, sizeof(reason)),
            reason);
        CHECK_UNSUPPORTED_TENSOR_DEPTH_REASON(reason);
    }
    SECTION("Resize")
    {
        ResizeInfo resizeInfo(ResizeAlgorithm::BILINEAR, 32, 32, QuantizationInfo());

        CHECK_UNSUPPORTED(queries.IsResizeSupported(resizeInfo, inputInfo, &outputs[0], reason, sizeof(reason)),
                          reason);
        CHECK_UNSUPPORTED_TENSOR_DEPTH_REASON(reason);
    }
    SECTION("Transpose")
    {
        // Generate 2 tests with invalid tensor depth:
        // - Unsupported caused by input
        // - Unsupported caused by output
        // clang-format off
        const auto shapes = GENERATE(
            //                          Input                                          TransposeInfo
            std::vector<TensorShape>{ { 1, 16, 16, UNSUPPORTED_OUTPUT_DIM },         { 0, 2, 1, 3 } },
            std::vector<TensorShape>{ { 1, 16, UNSUPPORTED_OUTPUT_DIM, OUTPUT_DIM }, { 0, 1, 3, 2 } }
        );
        // clang-format on

        TensorInfo transposeInputInfo(shapes[0], DataType::UINT8_QUANTIZED, DataFormat::NHWC,
                                      QuantizationInfo(0, 1.0f));
        CHECK_UNSUPPORTED(
            queries.IsTransposeSupported(shapes[1], transposeInputInfo, &outputs[0], reason, sizeof(reason)), reason);
        CHECK_UNSUPPORTED_TENSOR_DEPTH_REASON(reason);
    }
    SECTION("SpaceToDepth")
    {
        constexpr uint8_t blockSize = 2;
        // Generate 2 tests with invalid tensor depth:
        // - Unsupported caused by input
        // - Unsupported caused by output
        // clang-format off
        const auto shape = GENERATE_COPY(
            TensorShape{ 1, 16, 16, UNSUPPORTED_OUTPUT_DIM },
            TensorShape{ 1, blockSize * UNSUPPORTED_OUTPUT_DIM, blockSize * UNSUPPORTED_OUTPUT_DIM, OUTPUT_DIM }
        );
        // clang-format on

        SpaceToDepthInfo info(blockSize);
        TensorInfo spaceToDepthInputInfo(shape, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        CHECK_UNSUPPORTED(
            queries.IsSpaceToDepthSupported(spaceToDepthInputInfo, info, &outputs[0], reason, sizeof(reason)), reason);
        CHECK_UNSUPPORTED_TENSOR_DEPTH_REASON(reason);
    }
}
