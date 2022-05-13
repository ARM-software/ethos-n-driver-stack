//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/SupportQueries.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>

using namespace ethosn::support_library;

TEST_CASE("TransposeConvSupported")
{
    char reason[1024];

    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Input incorrect data type
    {
        TensorInfo biasInfo;
        TensorInfo weightsInfo;
        ConvolutionInfo convInfo;
        TensorInfo inputInfo({ 1, 1, 1, 1 }, DataType::INT32_QUANTIZED);
        REQUIRE(queries.IsTransposeConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                        sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Input to transpose conv must be UINT8_QUANTIZED"));
    }

    // Input incorrect data format
    {
        TensorInfo biasInfo;
        TensorInfo weightsInfo;
        ConvolutionInfo convInfo;
        TensorInfo inputInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO);
        REQUIRE(queries.IsTransposeConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                        sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Input to transpose conv must be NHWC or NHWCB"));
    }

    // Weights incorrect data type
    {
        TensorInfo biasInfo;
        TensorInfo weightsInfo({ 1, 1, 1, 1 }, DataType::INT32_QUANTIZED);
        ConvolutionInfo convInfo;
        TensorInfo inputInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC);
        REQUIRE(queries.IsTransposeConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                        sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Weights for transpose conv must be UINT8_QUANTIZED"));
    }

    // Weights incorrect data format
    {
        TensorInfo biasInfo;
        TensorInfo weightsInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC);
        ConvolutionInfo convInfo;
        TensorInfo inputInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC);
        REQUIRE(queries.IsTransposeConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                        sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Weights for transpose conv must be HWIO"));
    }

    // Bias incorrect data type
    {
        TensorInfo biasInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO);
        TensorInfo weightsInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO);
        ConvolutionInfo convInfo;
        TensorInfo inputInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC);
        REQUIRE(queries.IsTransposeConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                        sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Bias for transpose conv must be INT32_QUANTIZED"));
    }

    // Bias incorrect data format
    {
        TensorInfo biasInfo({ 1, 1, 1, 1 }, DataType::INT32_QUANTIZED, DataFormat::HWIO);
        TensorInfo weightsInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO);
        ConvolutionInfo convInfo;
        TensorInfo inputInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC);
        REQUIRE(queries.IsTransposeConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                        sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Bias for transpose conv must be NHWC"));
    }

    // Bias dimensions
    {
        TensorInfo biasInfo({ 1, 1, 1, 1 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 8.0f));
        TensorInfo weightsInfo({ 1, 1, 1, 10 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, QuantizationInfo(0, 2.0f));
        ConvolutionInfo convInfo;
        TensorInfo inputInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 4.0f));
        REQUIRE(queries.IsTransposeConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                        sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Invalid bias tensor dimensions"));
    }

    // Weights dimensions
    {
        TensorInfo biasInfo({ 1, 1, 1, 10 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 8.0f));
        TensorInfo weightsInfo({ 1, 1, 1, 10 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, QuantizationInfo(0, 2.0f));
        ConvolutionInfo convInfo;
        TensorInfo inputInfo({ 1, 1, 1, 5 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 4.0f));
        REQUIRE(queries.IsTransposeConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                        sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Weights input channels dimension (I) must match Input channels dimension (C)"));
    }

    // Invalid (zero) kernel size
    {
        TensorInfo biasInfo({ 1, 1, 1, 10 }, DataType::INT32_QUANTIZED, DataFormat::NHWC);
        TensorInfo weightsInfo({ 0, 0, 5, 10 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO);
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 1, 1 });
        TensorInfo inputInfo({ 1, 1, 1, 5 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC);
        REQUIRE(queries.IsTransposeConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                        sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Invalid kernel/stride parameters"));
    }

    // Invalid (zero) stride
    {
        TensorInfo biasInfo({ 1, 1, 1, 10 }, DataType::INT32_QUANTIZED, DataFormat::NHWC);
        TensorInfo weightsInfo({ 1, 1, 5, 10 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO);
        ConvolutionInfo convInfo({ 0, 0, 0, 0 }, { 0, 0 });
        TensorInfo inputInfo({ 1, 1, 1, 5 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC);
        REQUIRE(queries.IsTransposeConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                        sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Invalid kernel/stride parameters"));
    }

    // Output would be zero size
    {
        TensorInfo biasInfo({ 1, 1, 1, 10 }, DataType::INT32_QUANTIZED, DataFormat::NHWC);
        TensorInfo weightsInfo({ 3, 3, 5, 10 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO);
        ConvolutionInfo convInfo({ 10, 10, 10, 10 }, { 1, 1 });
        TensorInfo inputInfo({ 1, 1, 1, 5 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC);
        REQUIRE(queries.IsTransposeConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                        sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Output tensor would be empty"));
    }

    // Output info incorrect
    {
        TensorInfo biasInfo({ 1, 1, 1, 10 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 8.0f));
        TensorInfo weightsInfo({ 3, 3, 5, 10 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, QuantizationInfo(0, 2.0f));
        ConvolutionInfo convInfo(Padding(0, 0, 0, 0), Stride(2, 2), QuantizationInfo(0, 8.1f));
        TensorInfo inputInfo({ 1, 1, 1, 5 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 4.0f));
        TensorInfo outputInfo({ 1, 2, 3, 4 });
        REQUIRE(queries.IsTransposeConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, &outputInfo, reason,
                                                        sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Provided outputInfo is incorrect"));
    }

    // Zero point outside of valid range
    {
        TensorInfo biasInfo({ 1, 1, 1, 1 }, DataType::INT32_QUANTIZED, DataFormat::NHWC);
        TensorInfo weightsInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO,
                               QuantizationInfo(-10, 1.0f));
        ConvolutionInfo convInfo;
        TensorInfo inputInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC);

        // Invalid weight zero point
        {
            REQUIRE(queries.IsTransposeConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                            sizeof(reason)) == SupportedLevel::EstimateOnly);
            REQUIRE(Contains(reason, "Zero point out of range for weights info"));
        }

        // Invalid input zero point
        {
            weightsInfo.m_QuantizationInfo.SetZeroPoint(0);
            inputInfo.m_QuantizationInfo.SetZeroPoint(-10);
            REQUIRE(queries.IsTransposeConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                            sizeof(reason)) == SupportedLevel::Unsupported);
            REQUIRE(Contains(reason, "Zero point out of range for input info"));
        }

        // Invalid convInfo zero point
        {
            inputInfo.m_QuantizationInfo.SetZeroPoint(0);
            convInfo.m_OutputQuantizationInfo.SetZeroPoint(-10);
            REQUIRE(queries.IsTransposeConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                            sizeof(reason)) == SupportedLevel::Unsupported);
            REQUIRE(Contains(reason, "Zero point out of range for convInfo"));
        }
    }

    // Bias quantization params
    {
        // Incorrect scale
        TensorInfo biasInfo({ 1, 1, 1, 1 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 7.0f));
        TensorInfo weightsInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, QuantizationInfo(0, 2.0f));
        ConvolutionInfo convInfo;
        TensorInfo inputInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 4.0f));
        REQUIRE(queries.IsTransposeConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                        sizeof(reason)) == SupportedLevel::EstimateOnly);
        REQUIRE(Contains(reason, "Bias for transpose conv"));

        // Incorrect zero point
        biasInfo.m_QuantizationInfo = QuantizationInfo(123, 8.0f);
        REQUIRE(queries.IsTransposeConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                        sizeof(reason)) == SupportedLevel::EstimateOnly);
        REQUIRE(Contains(reason, "Bias for transpose conv"));
    }

    // Invalid kernel sizes
    {
        TensorInfo biasInfo({ 1, 1, 1, 10 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 8.0f));
        TensorInfo weightsInfo({ 13, 14, 5, 10 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO,
                               QuantizationInfo(0, 2.0f));
        ConvolutionInfo convInfo;
        TensorInfo inputInfo({ 1, 1, 1, 5 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 4.0f));
        REQUIRE(queries.IsTransposeConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                        sizeof(reason)) == SupportedLevel::EstimateOnly);
        REQUIRE(Contains(reason, "Unsupported kernel size"));
    }

    // Invalid stride
    {
        TensorInfo biasInfo({ 1, 1, 1, 10 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 8.0f));
        TensorInfo weightsInfo({ 3, 3, 5, 10 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, QuantizationInfo(0, 2.0f));
        ConvolutionInfo convInfo(Padding(0, 0, 0, 0), Stride(1, 2), QuantizationInfo(0, 1.0f));
        TensorInfo inputInfo({ 1, 1, 1, 5 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 4.0f));
        REQUIRE(queries.IsTransposeConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                        sizeof(reason)) == SupportedLevel::EstimateOnly);
        REQUIRE(Contains(reason, "Unsupported stride. Stride X and Y must be equal to 2"));
    }

    // Unsupported padding
    {
        TensorInfo biasInfo({ 1, 1, 1, 10 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 8.0f));
        TensorInfo weightsInfo({ 3, 3, 5, 10 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, QuantizationInfo(0, 2.0f));
        ConvolutionInfo convInfo(Padding(1, 2, 3, 4), Stride(2, 2), QuantizationInfo(0, 1.0f));
        TensorInfo inputInfo({ 1, 10, 10, 5 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 4.0f));
        REQUIRE(queries.IsTransposeConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                        sizeof(reason)) == SupportedLevel::EstimateOnly);
        REQUIRE(Contains(reason, "Unsupported padding"));
    }

    // Valid padding with wide kernel is unsupported
    {
        TensorInfo biasInfo({ 1, 1, 1, 10 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 8.0f));
        TensorInfo weightsInfo({ 9, 9, 5, 10 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, QuantizationInfo(0, 2.0f));
        ConvolutionInfo convInfo(Padding(0, 0, 0, 0), Stride(2, 2), QuantizationInfo(0, 1.0f));
        TensorInfo inputInfo({ 1, 1, 1, 5 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 4.0f));
        REQUIRE(queries.IsTransposeConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                        sizeof(reason)) == SupportedLevel::EstimateOnly);
        REQUIRE(Contains(reason, "Padding must be SAME for kernel > 7x7."));
    }

    // Overall scale out of range
    {
        TensorInfo biasInfo({ 1, 1, 1, 10 }, DataType::INT32_QUANTIZED, DataFormat::NHWC,
                            QuantizationInfo(0, 262160.0f));
        TensorInfo weightsInfo({ 3, 3, 5, 10 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO,
                               QuantizationInfo(0, 65540.f));
        ConvolutionInfo convInfo(Padding(0, 0, 0, 0), Stride(2, 2), QuantizationInfo(0, 0.1f));
        TensorInfo inputInfo({ 1, 1, 1, 5 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 4.0f));
        REQUIRE(queries.IsTransposeConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                        sizeof(reason)) == SupportedLevel::EstimateOnly);
        REQUIRE(Contains(reason, "Overall scale"));
    }

    // Supported explicit padding
    {
        TensorInfo biasInfo({ 1, 1, 1, 10 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 8.0f));
        TensorInfo weightsInfo({ 3, 3, 5, 10 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, QuantizationInfo(0, 2.0f));
        ConvolutionInfo convInfo(Padding(1, 1, 1, 1), Stride(2, 2), QuantizationInfo(0, 8.1f));
        TensorInfo inputInfo({ 1, 10, 10, 5 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 4.0f));
        REQUIRE(queries.IsTransposeConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                        sizeof(reason)) == SupportedLevel::Supported);
    }

    // Supported explicit padding
    {
        TensorInfo biasInfo({ 1, 1, 1, 10 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 8.0f));
        TensorInfo weightsInfo({ 7, 7, 5, 10 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, QuantizationInfo(0, 2.0f));
        ConvolutionInfo convInfo(Padding(3, 3, 3, 3), Stride(2, 2), QuantizationInfo(0, 8.1f));
        TensorInfo inputInfo({ 1, 10, 10, 5 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 4.0f));
        REQUIRE(queries.IsTransposeConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                        sizeof(reason)) == SupportedLevel::Supported);
    }

    // Successful case, with outputInfo set to null
    {
        TensorInfo biasInfo({ 1, 1, 1, 10 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 8.0f));
        TensorInfo weightsInfo({ 3, 3, 5, 10 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, QuantizationInfo(0, 2.0f));
        ConvolutionInfo convInfo(Padding(0, 0, 0, 0), Stride(2, 2), QuantizationInfo(0, 8.1f));
        TensorInfo inputInfo({ 1, 1, 1, 5 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 4.0f));
        REQUIRE(queries.IsTransposeConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, nullptr, reason,
                                                        sizeof(reason)) == SupportedLevel::Supported);
    }

    // Successful cases, with outputInfo being filled in.
    // Tests that each of the padding types (same before, same after and valid) are accepted.
    {
        TensorInfo biasInfo({ 1, 1, 1, 10 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 8.0f));
        TensorInfo weightsInfo({ 2, 2, 5, 10 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, QuantizationInfo(0, 2.0f));
        ConvolutionInfo convInfo(Padding(), Stride(2, 2), QuantizationInfo(0, 8.1f));
        TensorInfo inputInfo({ 1, 2, 2, 5 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 4.0f));

        // Valid padding
        {
            convInfo.m_Padding = Padding(0, 0, 0, 0);
            TensorInfo outputInfo;
            REQUIRE(queries.IsTransposeConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, &outputInfo,
                                                            reason, sizeof(reason)) == SupportedLevel::Supported);
            REQUIRE(outputInfo == TensorInfo({ 1, 4, 4, 10 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC,
                                             QuantizationInfo(0, 8.1f)));
        }

        // Same padding (prefer before)
        {
            convInfo.m_Padding = Padding(1, 0, 1, 0);
            TensorInfo outputInfo;
            REQUIRE(queries.IsTransposeConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, &outputInfo,
                                                            reason, sizeof(reason)) == SupportedLevel::Supported);
            REQUIRE(outputInfo == TensorInfo({ 1, 3, 3, 10 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC,
                                             QuantizationInfo(0, 8.1f)));
        }

        // Same padding (prefer after)
        {
            convInfo.m_Padding = Padding(0, 1, 0, 1);
            TensorInfo outputInfo;
            REQUIRE(queries.IsTransposeConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo, &outputInfo,
                                                            reason, sizeof(reason)) == SupportedLevel::Supported);
            REQUIRE(outputInfo == TensorInfo({ 1, 3, 3, 10 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC,
                                             QuantizationInfo(0, 8.1f)));
        }
    }

    // Tests for transpose_conv per channel quantization
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
        ConvolutionInfo convInfo(Padding(0, 0, 0, 0), Stride(2, 2));
        TensorInfo inputInfo({ 1, 1, 1, 1 }, inputDataType, DataFormat::NHWCB, { 0, 1.f });

        // Supported case
        {
            auto isSupported = queries.IsTransposeConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo,
                                                                       nullptr, reason, sizeof(reason));
            INFO(reason);
            REQUIRE(isSupported == SupportedLevel::Supported);
        }

        // Unsupported due to incorrect bias scale
        {
            inputInfo.m_QuantizationInfo.SetScale(2.f);
            auto isSupported = queries.IsTransposeConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo,
                                                                       nullptr, reason, sizeof(reason));
            INFO(reason);
            REQUIRE(isSupported == SupportedLevel::EstimateOnly);
            REQUIRE(Contains(reason,
                             "Bias for transpose conv must have quantization parameters with zero point of 0 and "
                             "scale of input scale x weight scale"));
        }

        // Unsupported due to unmatching scales sizes
        {
            inputInfo.m_QuantizationInfo.SetScale(1.f);
            biasInfo.m_QuantizationInfo.SetScales(QuantizationScales{ 0.1f, 0.2f });
            auto isSupported = queries.IsTransposeConvolutionSupported(biasInfo, weightsInfo, convInfo, inputInfo,
                                                                       nullptr, reason, sizeof(reason));
            INFO(reason);
            REQUIRE(isSupported == SupportedLevel::Unsupported);
            REQUIRE(Contains(
                reason, "Transpose Convolution: Biases must have quantization scales with same number of elements as "
                        "the quantization dim. Expected: 3, got: 2."));
        }
    }
}
