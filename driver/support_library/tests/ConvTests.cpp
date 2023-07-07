//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/SupportQueries.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>

using namespace ethosn::support_library;

namespace
{

SupportedLevel IsConvolutionSupportedImpl(SupportQueries& queries,
                                          const uint32_t kernelSizeX,
                                          const uint32_t kernelSizeY,
                                          const uint32_t numChannels,
                                          const uint32_t strideX,
                                          const uint32_t strideY,
                                          const uint32_t height,
                                          const uint32_t width,
                                          const Padding& padding = {},
                                          const bool isDepthwise = false)
{
    ConvolutionInfo convInfo;
    convInfo.m_Stride.m_X = strideX;
    convInfo.m_Stride.m_Y = strideY;
    convInfo.m_Padding    = padding;
    convInfo.m_OutputQuantizationInfo.SetZeroPoint(0);
    convInfo.m_OutputQuantizationInfo.SetScale(1.1f);

    TensorInfo input;
    input.m_Dimensions       = { { 1, height, width, numChannels } };
    input.m_DataType         = DataType::UINT8_QUANTIZED;
    input.m_DataFormat       = DataFormat::NHWCB;
    input.m_QuantizationInfo = { 0, 1.f };

    TensorInfo weightsInfo;
    weightsInfo.m_Dimensions       = { { kernelSizeY, kernelSizeX, numChannels, isDepthwise ? 1 : numChannels } };
    weightsInfo.m_DataType         = DataType::UINT8_QUANTIZED;
    weightsInfo.m_DataFormat       = isDepthwise ? DataFormat::HWIM : DataFormat::HWIO;
    weightsInfo.m_QuantizationInfo = { 0, 1.f };

    TensorInfo bias;
    bias.m_Dimensions       = { { 1, 1, 1, numChannels } };
    bias.m_DataType         = DataType::INT32_QUANTIZED;
    bias.m_DataFormat       = DataFormat::NHWC;
    bias.m_QuantizationInfo = { 0, 1.f };

    TensorInfo output;
    output.m_Dimensions       = { {
        1,
        ((height + padding.m_Top + padding.m_Bottom - kernelSizeY) / convInfo.m_Stride.m_Y) + 1,
        ((width + padding.m_Left + padding.m_Right - kernelSizeX) / convInfo.m_Stride.m_X) + 1,
        numChannels,
    } };
    output.m_DataType         = DataType::UINT8_QUANTIZED;
    output.m_DataFormat       = DataFormat::NHWCB;
    output.m_QuantizationInfo = { 0, 1.1f };

    return isDepthwise ? queries.IsDepthwiseConvolutionSupported(bias, weightsInfo, convInfo, input, &output)
                       : queries.IsConvolutionSupported(bias, weightsInfo, convInfo, input, &output);
}

SupportedLevel IsDepthwiseConvolutionSupportedImpl(SupportQueries& queries,
                                                   const uint32_t kernelSizeX,
                                                   const uint32_t kernelSizeY,
                                                   const uint32_t numChannels,
                                                   const uint32_t strideX,
                                                   const uint32_t strideY,
                                                   const uint32_t height,
                                                   const uint32_t width,
                                                   const Padding& padding = {})
{
    return IsConvolutionSupportedImpl(queries, kernelSizeX, kernelSizeY, numChannels, strideX, strideY, height, width,
                                      padding, true);
}

}    // namespace

TEST_CASE("ConvolutionSupported")
{
    char reason[1024];

    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    SECTION("Invalid case - zero kernel size")
    {
        TensorInfo biasInfo({ 1, 1, 1, 1 }, DataType::INT32_QUANTIZED);
        TensorInfo weightsInfo({ 0, 0, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO);
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 1, 1 });
        TensorInfo inputInfo({ 1, 1, 1, 1 });
        REQUIRE(queries.IsConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                               sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Invalid kernel/stride parameters"));
    }

    SECTION("Invalid case - zero stride")
    {
        TensorInfo biasInfo({ 1, 1, 1, 1 }, DataType::INT32_QUANTIZED);
        TensorInfo weightsInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO);
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 0, 0 });
        TensorInfo inputInfo({ 1, 1, 1, 1 });
        REQUIRE(queries.IsConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                               sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Invalid kernel/stride parameters"));
    }

    SECTION("Invalid case - output tensor would be empty")
    {
        TensorInfo biasInfo({ 1, 1, 1, 1 }, DataType::INT32_QUANTIZED);
        TensorInfo weightsInfo({ 999, 999, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO);
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 1, 1 });
        TensorInfo inputInfo({ 1, 1, 1, 1 });
        REQUIRE(queries.IsConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                               sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Output tensor would be empty"));
    }

    SECTION("Unsupported conv input data types")
    {
        const auto inputDataType = DataType::INT32_QUANTIZED;

        TensorInfo biasInfo({ 1, 1, 1, 1 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, 1.f });
        TensorInfo weightsInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO);
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 1, 1 });
        TensorInfo inputInfo({ 1, 1, 1, 1 }, inputDataType, DataFormat::NHWCB, { 0, 0.5f });
        REQUIRE(queries.IsConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                               sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Input to conv must be UINT8_QUANTIZED or INT8_QUANTIZED"));
    }

    SECTION("Supported conv input data types")
    {
        const auto inputDataType = GENERATE(DataType::UINT8_QUANTIZED, DataType::INT8_QUANTIZED);

        TensorInfo biasInfo({ 1, 1, 1, 1 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, 0.5f });
        TensorInfo weightsInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO);
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 1, 1 });
        TensorInfo inputInfo({ 1, 1, 1, 1 }, inputDataType, DataFormat::NHWCB, { 0, 0.5f });
        auto isSupported =
            queries.IsConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason, sizeof(reason));
        INFO(reason);
        REQUIRE(!(isSupported == SupportedLevel::Unsupported));
    }

    SECTION("Supported conv per channel quantization")
    {
        const auto inputDataType = GENERATE(DataType::UINT8_QUANTIZED, DataType::INT8_QUANTIZED);

        TensorInfo biasInfo({ 1, 1, 1, 3 }, DataType::INT32_QUANTIZED, DataFormat::NHWC);
        biasInfo.m_QuantizationInfo.SetScales(QuantizationScales{ 0.1f, 0.2f, 0.3f });
        biasInfo.m_QuantizationInfo.SetZeroPoint(0);
        biasInfo.m_QuantizationInfo.SetQuantizationDim(3);
        TensorInfo weightsInfo({ 1, 1, 1, 3 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO);
        weightsInfo.m_QuantizationInfo.SetScales(QuantizationScales{ 0.1f, 0.2f, 0.3f });
        weightsInfo.m_QuantizationInfo.SetZeroPoint(0);
        weightsInfo.m_QuantizationInfo.SetQuantizationDim(3);
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 1, 1 });
        TensorInfo inputInfo({ 1, 1, 1, 1 }, inputDataType, DataFormat::NHWCB, { 0, 1.f });
        auto isSupported =
            queries.IsConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason, sizeof(reason));
        INFO(reason);
        REQUIRE(isSupported == SupportedLevel::Supported);
    }

    SECTION("Unsupported conv per channel quantization: bias scales incorrect")
    {
        const auto inputDataType = GENERATE(DataType::UINT8_QUANTIZED, DataType::INT8_QUANTIZED);

        TensorInfo biasInfo({ 1, 1, 1, 3 }, DataType::INT32_QUANTIZED, DataFormat::NHWC);
        biasInfo.m_QuantizationInfo.SetScales(QuantizationScales{ 0.1f, 0.2f, 0.3f });
        biasInfo.m_QuantizationInfo.SetZeroPoint(0);
        biasInfo.m_QuantizationInfo.SetQuantizationDim(3);
        TensorInfo weightsInfo({ 1, 1, 1, 3 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO);
        weightsInfo.m_QuantizationInfo.SetScales(QuantizationScales{ 0.1f, 0.2f, 0.3f });
        weightsInfo.m_QuantizationInfo.SetZeroPoint(0);
        weightsInfo.m_QuantizationInfo.SetQuantizationDim(3);
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 1, 1 });
        TensorInfo inputInfo({ 1, 1, 1, 1 }, inputDataType, DataFormat::NHWCB, { 0, 2.f });
        auto isSupported =
            queries.IsConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason, sizeof(reason));
        INFO(reason);
        REQUIRE(isSupported == SupportedLevel::EstimateOnly);
        REQUIRE(Contains(reason,
                         "Bias for conv must have quantization parameters with scale of input scale x weight scale"));
    }

    SECTION("Unsupported conv per channel quantization: unmatching scales sizes")
    {
        const auto inputDataType = GENERATE(DataType::UINT8_QUANTIZED, DataType::INT8_QUANTIZED);

        TensorInfo biasInfo({ 1, 1, 1, 3 }, DataType::INT32_QUANTIZED, DataFormat::NHWC);
        biasInfo.m_QuantizationInfo.SetScales(QuantizationScales{ 0.1f, 0.2f });
        biasInfo.m_QuantizationInfo.SetZeroPoint(0);
        biasInfo.m_QuantizationInfo.SetQuantizationDim(3);
        TensorInfo weightsInfo({ 1, 1, 1, 3 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO);
        weightsInfo.m_QuantizationInfo.SetScales(QuantizationScales{ 0.1f, 0.2f, 0.3f });
        weightsInfo.m_QuantizationInfo.SetZeroPoint(0);
        weightsInfo.m_QuantizationInfo.SetQuantizationDim(3);
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 1, 1 });
        TensorInfo inputInfo({ 1, 1, 1, 1 }, inputDataType, DataFormat::NHWCB, { 0, 2.f });
        auto isSupported =
            queries.IsConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason, sizeof(reason));
        INFO(reason);
        REQUIRE(isSupported == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Convolution: Biases must have quantization scales with same number of elements as "
                                 "the quantization dim. Expected: 3, got: 2."));
    }

    SECTION("Unsupported conv overall scale: negative")
    {
        const auto inputDataType = GENERATE(DataType::UINT8_QUANTIZED, DataType::INT8_QUANTIZED);

        TensorInfo biasInfo({ 1, 1, 1, 1 }, DataType::INT32_QUANTIZED, DataFormat::NHWC);
        biasInfo.m_QuantizationInfo.SetScales(QuantizationScales{ -2.3e-10f });
        biasInfo.m_QuantizationInfo.SetZeroPoint(0);
        biasInfo.m_QuantizationInfo.SetQuantizationDim(3);
        TensorInfo weightsInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO);
        weightsInfo.m_QuantizationInfo.SetScales(QuantizationScales{ -2.3e-10f });
        weightsInfo.m_QuantizationInfo.SetZeroPoint(0);
        weightsInfo.m_QuantizationInfo.SetQuantizationDim(3);
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 1, 1 }, { 0, 1.f });
        TensorInfo inputInfo({ 1, 1, 1, 1 }, inputDataType, DataFormat::NHWCB, { 0, 1.f });
        auto isSupported =
            queries.IsConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason, sizeof(reason));
        INFO(reason);
        REQUIRE(isSupported == SupportedLevel::EstimateOnly);
        REQUIRE(Contains(reason, "Overall scale (of the input * weights / output) should be in the range"));
    }

    SECTION("Supported conv overall scale: just fits")
    {
        const auto inputDataType = GENERATE(DataType::UINT8_QUANTIZED, DataType::INT8_QUANTIZED);

        TensorInfo biasInfo({ 1, 1, 1, 1 }, DataType::INT32_QUANTIZED, DataFormat::NHWC);
        biasInfo.m_QuantizationInfo.SetScales(QuantizationScales{ 2.33e-10f });
        biasInfo.m_QuantizationInfo.SetZeroPoint(0);
        biasInfo.m_QuantizationInfo.SetQuantizationDim(3);
        TensorInfo weightsInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO);
        weightsInfo.m_QuantizationInfo.SetScales(QuantizationScales{ 2.33e-10f });
        weightsInfo.m_QuantizationInfo.SetZeroPoint(0);
        weightsInfo.m_QuantizationInfo.SetQuantizationDim(3);
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 1, 1 }, { 0, 1.f });
        TensorInfo inputInfo({ 1, 1, 1, 1 }, inputDataType, DataFormat::NHWCB, { 0, 1.f });
        auto isSupported =
            queries.IsConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason, sizeof(reason));
        INFO(reason);
        REQUIRE(isSupported == SupportedLevel::Supported);
    }

    SECTION("Unsupported conv per channel quantization: unsupported axis")
    {
        const auto inputDataType = GENERATE(DataType::UINT8_QUANTIZED, DataType::INT8_QUANTIZED);

        TensorInfo biasInfo({ 1, 1, 1, 3 }, DataType::INT32_QUANTIZED, DataFormat::NHWC);
        biasInfo.m_QuantizationInfo.SetScales(QuantizationScales{ 0.1f, 0.2f, 0.3f });
        biasInfo.m_QuantizationInfo.SetZeroPoint(0);
        biasInfo.m_QuantizationInfo.SetQuantizationDim(3);

        TensorInfo weightsInfo({ 1, 1, 1, 3 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO);
        weightsInfo.m_QuantizationInfo.SetScales(QuantizationScales{ 0.1f, 0.2f, 0.3f });
        weightsInfo.m_QuantizationInfo.SetZeroPoint(0);
        weightsInfo.m_QuantizationInfo.SetQuantizationDim(3);

        SECTION("Invalid bias axis")
        {
            biasInfo.m_QuantizationInfo.SetQuantizationDim(0);
        }

        SECTION("Invalid weight axis")
        {
            weightsInfo.m_QuantizationInfo.SetQuantizationDim(0);
        }

        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 1, 1 });
        TensorInfo inputInfo({ 1, 1, 1, 1 }, inputDataType, DataFormat::NHWCB, { 0, 2.f });
        auto isSupported =
            queries.IsConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason, sizeof(reason));
        INFO(reason);
        REQUIRE(isSupported == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Per channel quantization axis must be 3"));
    }

    SECTION("Invalid zero point")
    {
        TensorInfo weightsInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, QuantizationInfo(0, 1.0f));
        TensorInfo inputInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo biasInfo({ 1, 1, 1, 1 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 1, 1 });

        SECTION("Invalid weight zero point")
        {
            weightsInfo.m_QuantizationInfo.SetZeroPoint(-10);
            REQUIRE(queries.IsConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                   sizeof(reason)) == SupportedLevel::Unsupported);
            INFO(reason);
            REQUIRE(Contains(reason, "Zero point out of range for weights info"));
        }

        SECTION("Invalid input zero point")
        {
            inputInfo.m_QuantizationInfo.SetZeroPoint(-10);
            REQUIRE(queries.IsConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                   sizeof(reason)) == SupportedLevel::Unsupported);
            INFO(reason);
            REQUIRE(Contains(reason, "Zero point out of range for input info"));
        }

        SECTION("Invalid convInfo zero point")
        {
            convInfo.m_OutputQuantizationInfo.SetZeroPoint(-10);
            REQUIRE(queries.IsConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                   sizeof(reason)) == SupportedLevel::Unsupported);
            INFO(reason);
            REQUIRE(Contains(reason, "Zero point out of range for convInfo"));
        }
    }

    SECTION("Check max padding")
    {
        TensorInfo weightsInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, QuantizationInfo(0, 1.0f));
        TensorInfo inputInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo biasInfo({ 1, 1, 1, 1 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        ConvolutionInfo convInfo_pad_max({ 7, 7, 7, 7 }, { 1, 1 });
        ConvolutionInfo convInfo_pad_too_big({ 8, 8, 8, 8 }, { 1, 1 });
        REQUIRE(queries.IsConvolutionSupported(biasInfo, weightsInfo, convInfo_pad_max, inputInfo, nullptr, reason,
                                               sizeof(reason)) == SupportedLevel::Supported);

        REQUIRE(queries.IsConvolutionSupported(biasInfo, weightsInfo, convInfo_pad_too_big, inputInfo, nullptr, reason,
                                               sizeof(reason)) == SupportedLevel::EstimateOnly);
        INFO(reason);
        REQUIRE(Contains(reason, "Unsupported padding"));
    }

    // A configuration we should never need to support but could potentially estimate
    REQUIRE(IsConvolutionSupportedImpl(queries, 5, 5, 1, 77, 99, 16, 16) == SupportedLevel::EstimateOnly);

    // 1x1/(1,1)
    REQUIRE(IsConvolutionSupportedImpl(queries, 1, 1, 16, 1, 1, 16, 16, { 0, 0, 0, 0 }) == SupportedLevel::Supported);
    REQUIRE(IsConvolutionSupportedImpl(queries, 1, 1, 1, 1, 1, 16, 16, { 0, 0, 0, 0 }) == SupportedLevel::Supported);
    REQUIRE(IsConvolutionSupportedImpl(queries, 1, 1, 1, 1, 1, 16, 16, { 1, 1, 0, 0 }) == SupportedLevel::Supported);

    // 1x1/(2,2)
    REQUIRE(IsConvolutionSupportedImpl(queries, 1, 1, 1, 2, 2, 16, 16, { 0, 0, 0, 0 }) == SupportedLevel::Supported);
    REQUIRE(IsConvolutionSupportedImpl(queries, 1, 1, 1, 2, 2, 16, 16, { 0, 0, 1, 1 }) == SupportedLevel::Supported);

    // 3x3/(1,1)
    REQUIRE(IsConvolutionSupportedImpl(queries, 3, 3, 1, 1, 1, 16, 16, { 0, 0, 0, 0 }) == SupportedLevel::Supported);
    REQUIRE(IsConvolutionSupportedImpl(queries, 3, 3, 1, 1, 1, 16, 16, { 1, 1, 1, 1 }) == SupportedLevel::Supported);
    REQUIRE(IsConvolutionSupportedImpl(queries, 3, 3, 1, 1, 1, 16, 16, { 0, 1, 0, 1 }) == SupportedLevel::Supported);

    // 3x3/(2,2)
    REQUIRE(IsConvolutionSupportedImpl(queries, 3, 3, 16, 2, 2, 16, 16, { 0, 0, 0, 0 }) == SupportedLevel::Supported);
    REQUIRE(IsConvolutionSupportedImpl(queries, 3, 3, 16, 2, 2, 16, 16, { 0, 1, 0, 1 }) == SupportedLevel::Supported);
    REQUIRE(IsConvolutionSupportedImpl(queries, 3, 3, 16, 2, 2, 16, 16, { 1, 0, 1, 0 }) == SupportedLevel::Supported);
    REQUIRE(IsConvolutionSupportedImpl(queries, 3, 3, 16, 2, 2, 16, 16, { 1, 1, 1, 1 }) == SupportedLevel::Supported);
    REQUIRE(IsConvolutionSupportedImpl(queries, 3, 3, 16, 2, 2, 15, 15, { 2, 1, 2, 1 }) == SupportedLevel::Supported);
    REQUIRE(IsConvolutionSupportedImpl(queries, 3, 3, 16, 2, 2, 15, 15, { 1, 2, 1, 2 }) == SupportedLevel::Supported);

    REQUIRE(IsConvolutionSupportedImpl(queries, 3, 3, 16, 2, 2, 15, 15, { 0, 0, 0, 0 }) == SupportedLevel::Supported);
    REQUIRE(IsConvolutionSupportedImpl(queries, 3, 3, 16, 2, 2, 15, 15, { 1, 1, 1, 1 }) == SupportedLevel::Supported);
    REQUIRE(IsConvolutionSupportedImpl(queries, 3, 3, 16, 2, 2, 15, 15, { 0, 1, 0, 1 }) == SupportedLevel::Supported);

    // 5x5/(1,1)
    REQUIRE(IsConvolutionSupportedImpl(queries, 5, 5, 1, 1, 1, 16, 16, { 0, 0, 0, 0 }) == SupportedLevel::Supported);
    REQUIRE(IsConvolutionSupportedImpl(queries, 5, 5, 1, 1, 1, 16, 16, { 2, 2, 2, 2 }) == SupportedLevel::Supported);
    REQUIRE(IsConvolutionSupportedImpl(queries, 5, 5, 1, 1, 1, 16, 16, { 1, 2, 1, 2 }) == SupportedLevel::Supported);

    // 7x7/(2,2)
    REQUIRE(IsConvolutionSupportedImpl(queries, 7, 7, 1, 2, 2, 16, 16, { 0, 0, 0, 0 }) == SupportedLevel::Supported);
    REQUIRE(IsConvolutionSupportedImpl(queries, 7, 7, 1, 2, 2, 16, 16, { 2, 3, 2, 3 }) == SupportedLevel::Supported);
    REQUIRE(IsConvolutionSupportedImpl(queries, 7, 7, 1, 2, 2, 16, 16, { 3, 3, 3, 3 }) == SupportedLevel::Supported);
    REQUIRE(IsConvolutionSupportedImpl(queries, 7, 7, 1, 2, 2, 16, 16, { 3, 4, 3, 4 }) == SupportedLevel::Supported);
    REQUIRE(IsConvolutionSupportedImpl(queries, 7, 7, 1, 2, 2, 16, 16, { 4, 3, 4, 3 }) == SupportedLevel::Supported);

    // 9x9/(2,2)
    REQUIRE(IsConvolutionSupportedImpl(queries, 9, 9, 1, 2, 2, 16, 16, { 0, 0, 0, 0 }) == SupportedLevel::EstimateOnly);

    // 1x3/(1,1)
    REQUIRE(IsConvolutionSupportedImpl(queries, 1, 3, 1, 1, 1, 16, 16, { 0, 0, 0, 0 }) == SupportedLevel::Supported);
    REQUIRE(IsConvolutionSupportedImpl(queries, 1, 3, 1, 1, 1, 16, 16, { 1, 1, 0, 0 }) == SupportedLevel::Supported);
    REQUIRE(IsConvolutionSupportedImpl(queries, 1, 3, 1, 1, 1, 16, 16, { 1, 1, 1, 1 }) == SupportedLevel::Supported);

    // 3x1/(1,1)
    REQUIRE(IsConvolutionSupportedImpl(queries, 3, 1, 1, 1, 1, 16, 16, { 0, 0, 0, 0 }) == SupportedLevel::Supported);
    REQUIRE(IsConvolutionSupportedImpl(queries, 3, 1, 1, 1, 1, 16, 16, { 0, 0, 1, 1 }) == SupportedLevel::Supported);
    REQUIRE(IsConvolutionSupportedImpl(queries, 3, 1, 1, 1, 1, 16, 16, { 1, 1, 1, 1 }) == SupportedLevel::Supported);

    // 1x7/(1,1)
    REQUIRE(IsConvolutionSupportedImpl(queries, 1, 7, 1, 1, 1, 16, 16, { 0, 0, 0, 0 }) == SupportedLevel::Supported);
    REQUIRE(IsConvolutionSupportedImpl(queries, 1, 7, 1, 1, 1, 16, 16, { 3, 3, 0, 0 }) == SupportedLevel::Supported);
    REQUIRE(IsConvolutionSupportedImpl(queries, 1, 7, 1, 1, 1, 16, 16, { 3, 3, 1, 1 }) == SupportedLevel::Supported);

    // 7x1/(1,1)
    REQUIRE(IsConvolutionSupportedImpl(queries, 7, 1, 1, 1, 1, 16, 16, { 0, 0, 0, 0 }) == SupportedLevel::Supported);
    REQUIRE(IsConvolutionSupportedImpl(queries, 7, 1, 1, 1, 1, 16, 16, { 0, 0, 3, 3 }) == SupportedLevel::Supported);
    REQUIRE(IsConvolutionSupportedImpl(queries, 7, 1, 1, 1, 1, 16, 16, { 1, 1, 3, 3 }) == SupportedLevel::Supported);

    // 9x9/(1,1)
    REQUIRE(IsConvolutionSupportedImpl(queries, 9, 9, 1, 1, 1, 16, 16, { 0, 0, 0, 0 }) == SupportedLevel::Supported);
    REQUIRE(IsConvolutionSupportedImpl(queries, 9, 9, 1, 1, 1, 16, 16, { 4, 4, 4, 4 }) == SupportedLevel::Supported);
    REQUIRE(IsConvolutionSupportedImpl(queries, 9, 9, 1, 1, 1, 16, 16, { 4, 1, 4, 4 }) == SupportedLevel::Supported);
}

TEST_CASE("DepthwiseConvolutionSupported")
{
    char reason[1024];

    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    SECTION("Invalid case - zero kernel size")
    {
        TensorInfo biasInfo({ 1, 1, 1, 1 }, DataType::INT32_QUANTIZED);
        TensorInfo weightsInfo({ 0, 0, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM);
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 1, 1 });
        TensorInfo inputInfo({ 1, 1, 1, 1 });
        REQUIRE(queries.IsDepthwiseConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                        sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Invalid kernel/stride parameters"));
    }

    SECTION("Invalid case - zero stride")
    {
        TensorInfo biasInfo({ 1, 1, 1, 1 }, DataType::INT32_QUANTIZED);
        TensorInfo weightsInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM);
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 0, 0 });
        TensorInfo inputInfo({ 1, 1, 1, 1 });
        REQUIRE(queries.IsDepthwiseConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                        sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Invalid kernel/stride parameters"));
    }

    SECTION("Invalid case - output tensor would be empty")
    {
        TensorInfo biasInfo({ 1, 1, 1, 1 }, DataType::INT32_QUANTIZED);
        TensorInfo weightsInfo({ 999, 999, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM);
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 1, 1 });
        TensorInfo inputInfo({ 1, 1, 1, 1 });
        REQUIRE(queries.IsDepthwiseConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                        sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Output tensor would be empty"));
    }

    SECTION("Channel multiplier > 1 is not supported with > 1 input channel")
    {
        TensorInfo inputInfo({ 1, 16, 16, 2 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC);
        ConvolutionInfo convInfo = ConvolutionInfo({ 0, 0, 0, 0 }, { 1, 1 }, { 0, 1.1f });
        TensorInfo biasInfo({ 1, 1, 1, 64 }, DataType::INT32_QUANTIZED, DataFormat::NHWC);
        TensorInfo weightsInfo({ 1, 1, 2, 32 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM);
        REQUIRE(queries.IsDepthwiseConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo) ==
                SupportedLevel::EstimateOnly);
    }

    SECTION("Unsupported depthwise conv input data types")
    {
        const auto inputDataType = DataType::INT32_QUANTIZED;

        TensorInfo biasInfo({ 1, 1, 1, 1 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, 1.f });
        TensorInfo weightsInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM);
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 1, 1 });
        TensorInfo inputInfo({ 1, 1, 1, 1 }, inputDataType, DataFormat::NHWCB, { 0, 0.5f });
        REQUIRE(queries.IsDepthwiseConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                        sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Input to depthwise conv must be UINT8_QUANTIZED or INT8_QUANTIZED"));
    }

    SECTION("Supported depthwise conv input data types")
    {
        const auto inputDataType = GENERATE(DataType::UINT8_QUANTIZED, DataType::INT8_QUANTIZED);

        TensorInfo biasInfo({ 1, 1, 1, 1 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, 1.f });
        TensorInfo weightsInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM);
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 1, 1 });
        TensorInfo inputInfo({ 1, 1, 1, 1 }, inputDataType, DataFormat::NHWCB, { 0, 0.5f });
        auto isSupported = queries.IsDepthwiseConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr,
                                                                   reason, sizeof(reason));
        INFO(reason);
        REQUIRE(!(isSupported == SupportedLevel::Unsupported));
    }

    // A configuration we should never need to support but could potentially estimate
    REQUIRE(IsDepthwiseConvolutionSupportedImpl(queries, 5, 5, 1, 77, 99, 16, 16) == SupportedLevel::EstimateOnly);

    // Supported configurations
    REQUIRE(IsDepthwiseConvolutionSupportedImpl(queries, 3, 3, 16, 1, 1, 16, 16, { 1, 1, 1, 1 }) ==
            SupportedLevel::Supported);
    REQUIRE(IsDepthwiseConvolutionSupportedImpl(queries, 3, 3, 1, 2, 2, 16, 16, { 0, 1, 0, 1 }) ==
            SupportedLevel::Supported);
    REQUIRE(IsDepthwiseConvolutionSupportedImpl(queries, 3, 3, 16, 2, 2, 15, 15, { 1, 1, 1, 1 }) ==
            SupportedLevel::Supported);
    REQUIRE(IsDepthwiseConvolutionSupportedImpl(queries, 3, 3, 16, 2, 2, 16, 16, { 1, 1, 1, 1 }) ==
            SupportedLevel::Supported);
    REQUIRE(IsDepthwiseConvolutionSupportedImpl(queries, 7, 7, 16, 2, 2, 16, 16, { 3, 3, 3, 3 }) ==
            SupportedLevel::Supported);

    SECTION("Channel multiplier > 1 is supported with 1 input channel")
    {
        TensorInfo inputInfo({ 1, 16, 16, 1 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC);
        ConvolutionInfo convInfo = ConvolutionInfo({ 0, 0, 0, 0 }, { 1, 1 }, { 0, 1.1f });
        TensorInfo biasInfo({ 1, 1, 1, 32 }, DataType::INT32_QUANTIZED, DataFormat::NHWC);
        TensorInfo weightsInfo({ 1, 1, 1, 32 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM);
        REQUIRE(queries.IsDepthwiseConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo) ==
                SupportedLevel::Supported);
    }

    SECTION("Supported per-input-channel weights quantization")
    {
        TensorInfo biasInfo({ 1, 1, 1, 3 }, DataType::INT32_QUANTIZED, DataFormat::NHWC);
        biasInfo.m_QuantizationInfo.SetScales(QuantizationScales{ 0.1f, 0.2f, 0.3f });
        biasInfo.m_QuantizationInfo.SetZeroPoint(0);
        biasInfo.m_QuantizationInfo.SetQuantizationDim(3);
        TensorInfo weightsInfo({ 1, 1, 3, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM);
        weightsInfo.m_QuantizationInfo.SetScales(QuantizationScales{ 0.1f, 0.2f, 0.3f });
        weightsInfo.m_QuantizationInfo.SetZeroPoint(0);
        weightsInfo.m_QuantizationInfo.SetQuantizationDim(2);
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 1, 1 });
        TensorInfo inputInfo({ 1, 1, 1, 3 }, DataType::UINT8_QUANTIZED, DataFormat::NHWCB, { 0, 1.f });
        CHECK(queries.IsDepthwiseConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                      sizeof(reason)) == SupportedLevel::Supported);
    }

    SECTION("Unsupported bias quantization dim")
    {
        TensorInfo biasInfo({ 1, 1, 1, 3 }, DataType::INT32_QUANTIZED, DataFormat::NHWC);
        biasInfo.m_QuantizationInfo.SetQuantizationDim(0);
        TensorInfo weightsInfo({ 1, 1, 3, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM);
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 1, 1 });
        TensorInfo inputInfo({ 1, 1, 1, 3 }, DataType::UINT8_QUANTIZED, DataFormat::NHWCB, { 0, 1.f });
        CHECK(queries.IsDepthwiseConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                      sizeof(reason)) == SupportedLevel::Unsupported);
        CHECK(Contains(reason, "Per channel quantization axis must be 3 for Biases"));
    }
    SECTION("Unsupported bias num scales (inconsistent with tensor dimension)")
    {
        TensorInfo biasInfo({ 1, 1, 1, 3 }, DataType::INT32_QUANTIZED, DataFormat::NHWC);
        biasInfo.m_QuantizationInfo.SetQuantizationDim(3);
        biasInfo.m_QuantizationInfo.SetScales(QuantizationScales{ 0.1f, 0.2f });    // There should be three of these
        TensorInfo weightsInfo({ 1, 1, 3, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM);
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 1, 1 });
        TensorInfo inputInfo({ 1, 1, 1, 3 }, DataType::UINT8_QUANTIZED, DataFormat::NHWCB, { 0, 1.f });
        CHECK(queries.IsDepthwiseConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                      sizeof(reason)) == SupportedLevel::Unsupported);
        CHECK(Contains(reason, "Biases must have quantization scales with same number of elements as the quantization "
                               "dim. Expected: 3, got: 2."));
    }

    SECTION("Unsupported weights quantization dim")
    {
        TensorInfo biasInfo({ 1, 1, 1, 3 }, DataType::INT32_QUANTIZED, DataFormat::NHWC);
        TensorInfo weightsInfo({ 1, 1, 3, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM);
        weightsInfo.m_QuantizationInfo.SetQuantizationDim(3);
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 1, 1 });
        TensorInfo inputInfo({ 1, 1, 1, 3 }, DataType::UINT8_QUANTIZED, DataFormat::NHWCB, { 0, 1.f });
        CHECK(queries.IsDepthwiseConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                      sizeof(reason)) == SupportedLevel::Unsupported);
        CHECK(Contains(reason, "Per channel quantization axis must be 2 for Weights"));
    }
    SECTION("Unsupported weights num scales")
    {
        TensorInfo biasInfo({ 1, 1, 1, 3 }, DataType::INT32_QUANTIZED, DataFormat::NHWC);
        TensorInfo weightsInfo({ 1, 1, 3, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM);
        weightsInfo.m_QuantizationInfo.SetScales(QuantizationScales{ 0.1f, 0.2f });    // There should be three of these
        weightsInfo.m_QuantizationInfo.SetQuantizationDim(2);
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 1, 1 });
        TensorInfo inputInfo({ 1, 1, 1, 3 }, DataType::UINT8_QUANTIZED, DataFormat::NHWCB, { 0, 1.f });
        CHECK(queries.IsDepthwiseConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                      sizeof(reason)) == SupportedLevel::Unsupported);
        CHECK(Contains(reason, "Weights must have quantization scales with same number of elements as the quantization "
                               "dim. Expected: 3, got: 2."));
    }

    SECTION("Unsupported input quantization dim")
    {
        TensorInfo biasInfo({ 1, 1, 1, 3 }, DataType::INT32_QUANTIZED, DataFormat::NHWC);
        TensorInfo weightsInfo({ 1, 1, 3, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM);
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 1, 1 });
        TensorInfo inputInfo({ 1, 1, 1, 3 }, DataType::UINT8_QUANTIZED, DataFormat::NHWCB, { 0, 1.f });
        inputInfo.m_QuantizationInfo.SetQuantizationDim(3);
        CHECK(queries.IsDepthwiseConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                      sizeof(reason)) == SupportedLevel::Unsupported);
        CHECK(Contains(reason, "Quantization Dim should not be used on Input"));
    }
    SECTION("Unsupported input num scales")
    {
        TensorInfo biasInfo({ 1, 1, 1, 3 }, DataType::INT32_QUANTIZED, DataFormat::NHWC);
        TensorInfo weightsInfo({ 1, 1, 3, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM);
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 1, 1 });
        TensorInfo inputInfo({ 1, 1, 1, 3 }, DataType::UINT8_QUANTIZED, DataFormat::NHWCB, { 0, 1.f });
        inputInfo.m_QuantizationInfo.SetScales(QuantizationScales{ 0.1f, 0.2f });
        CHECK(queries.IsDepthwiseConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                      sizeof(reason)) == SupportedLevel::Unsupported);
        CHECK(Contains(reason, "Input quantization scales must have a size of 1"));
    }

    SECTION("Unsupported output quantization dim")
    {
        TensorInfo biasInfo({ 1, 1, 1, 3 }, DataType::INT32_QUANTIZED, DataFormat::NHWC);
        TensorInfo weightsInfo({ 1, 1, 3, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM);
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 1, 1 });
        convInfo.m_OutputQuantizationInfo.SetQuantizationDim(3);
        TensorInfo inputInfo({ 1, 1, 1, 3 }, DataType::UINT8_QUANTIZED, DataFormat::NHWCB, { 0, 1.f });
        CHECK(queries.IsDepthwiseConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                      sizeof(reason)) == SupportedLevel::Unsupported);
        CHECK(Contains(reason, "Quantization Dim should not be used on Output"));
    }
    SECTION("Unsupported output num scales")
    {
        TensorInfo biasInfo({ 1, 1, 1, 3 }, DataType::INT32_QUANTIZED, DataFormat::NHWC);
        TensorInfo weightsInfo({ 1, 1, 3, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM);
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 1, 1 });
        convInfo.m_OutputQuantizationInfo.SetScales(QuantizationScales{ 0.1f, 0.2f });
        TensorInfo inputInfo({ 1, 1, 1, 3 }, DataType::UINT8_QUANTIZED, DataFormat::NHWCB, { 0, 1.f });
        CHECK(queries.IsDepthwiseConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                      sizeof(reason)) == SupportedLevel::Unsupported);
        CHECK(Contains(reason, "Output quantization scales must have a size of 1"));
    }

    SECTION("Unsupported - bias scales inconsistent with input * weights")
    {
        TensorInfo biasInfo({ 1, 1, 1, 3 }, DataType::INT32_QUANTIZED, DataFormat::NHWC);
        biasInfo.m_QuantizationInfo.SetScales(QuantizationScales{ 0.1f, 0.2f, 0.3f });
        biasInfo.m_QuantizationInfo.SetZeroPoint(0);
        biasInfo.m_QuantizationInfo.SetQuantizationDim(3);
        TensorInfo weightsInfo({ 1, 1, 3, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM);
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 1, 1 });
        TensorInfo inputInfo({ 1, 1, 1, 3 }, DataType::UINT8_QUANTIZED, DataFormat::NHWCB, { 0, 1.f });
        CHECK(queries.IsDepthwiseConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                      sizeof(reason)) == SupportedLevel::EstimateOnly);
        CHECK(Contains(reason, "Bias for depthwise conv must have quantization parameters with zero point of 0 and "
                               "scale of input scale x weight scale"));
    }

    SECTION("Unsupported overall scale on one channel")
    {
        TensorInfo biasInfo({ 1, 1, 1, 3 }, DataType::INT32_QUANTIZED, DataFormat::NHWC);
        biasInfo.m_QuantizationInfo.SetScales(QuantizationScales{ 0.1f, 0.2f, 65540.f });
        biasInfo.m_QuantizationInfo.SetZeroPoint(0);
        biasInfo.m_QuantizationInfo.SetQuantizationDim(3);
        TensorInfo weightsInfo({ 1, 1, 3, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM);
        weightsInfo.m_QuantizationInfo.SetScales(QuantizationScales{ 0.1f, 0.2f, 65540.f });
        weightsInfo.m_QuantizationInfo.SetZeroPoint(0);
        weightsInfo.m_QuantizationInfo.SetQuantizationDim(2);
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 1, 1 });
        TensorInfo inputInfo({ 1, 1, 1, 3 }, DataType::UINT8_QUANTIZED, DataFormat::NHWCB, { 0, 1.f });
        CHECK(queries.IsDepthwiseConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                      sizeof(reason)) == SupportedLevel::EstimateOnly);
        CHECK(Contains(reason, "Overall scale (of the input * weights / output) should be in the range"));
    }

    SECTION("Invalid zero point")
    {
        TensorInfo weightsInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, QuantizationInfo(0, 1.0f));
        TensorInfo inputInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo biasInfo({ 1, 1, 1, 1 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 1, 1 });

        SECTION("Invalid weights zero point")
        {
            weightsInfo.m_QuantizationInfo.SetZeroPoint(-10);
            REQUIRE(queries.IsConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                   sizeof(reason)) == SupportedLevel::Unsupported);
            INFO(reason);
            REQUIRE(Contains(reason, "Zero point out of range for weights info"));
        }

        SECTION("Invalid input zero point")
        {
            inputInfo.m_QuantizationInfo.SetZeroPoint(-10);
            REQUIRE(queries.IsConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                   sizeof(reason)) == SupportedLevel::Unsupported);
            INFO(reason);
            REQUIRE(Contains(reason, "Zero point out of range for input info"));
        }

        SECTION("Invalid output zero point")
        {
            convInfo.m_OutputQuantizationInfo.SetZeroPoint(-10);
            REQUIRE(queries.IsConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                   sizeof(reason)) == SupportedLevel::Unsupported);
            INFO(reason);
            REQUIRE(Contains(reason, "Zero point out of range for convInfo"));
        }
    }

    SECTION("Check max padding")
    {
        TensorInfo inputInfo({ 1, 16, 16, 1 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC);
        TensorInfo biasInfo({ 1, 1, 1, 32 }, DataType::INT32_QUANTIZED, DataFormat::NHWC);
        TensorInfo weightsInfo({ 1, 1, 1, 32 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM);
        ConvolutionInfo convInfo_pad_max     = ConvolutionInfo({ 7, 7, 7, 7 }, { 1, 1 }, { 0, 1.1f });
        ConvolutionInfo convInfo_pad_too_big = ConvolutionInfo({ 8, 8, 8, 8 }, { 1, 1 }, { 0, 1.1f });

        REQUIRE(queries.IsDepthwiseConvolutionSupported(biasInfo, weightsInfo, convInfo_pad_max, inputInfo, nullptr,
                                                        reason, sizeof(reason)) == SupportedLevel::Supported);

        REQUIRE(queries.IsDepthwiseConvolutionSupported(biasInfo, weightsInfo, convInfo_pad_too_big, inputInfo, nullptr,
                                                        reason, sizeof(reason)) == SupportedLevel::EstimateOnly);
        INFO(reason);
        REQUIRE(Contains(reason, "Unsupported padding"));
    }
}
