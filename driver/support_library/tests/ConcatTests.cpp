//
// Copyright © 2018-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"
#include "../include/ethosn_support_library/SupportQueries.hpp"
#include "../src/Utils.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>
#include <ethosn_command_stream/CommandStreamBuffer.hpp>

#include <cmath>

using namespace ethosn::support_library;

TEST_CASE("ConcatenationSupported")
{
    char reason[1024];

    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    SECTION("No inputs")
    {
        REQUIRE(queries.IsConcatenationSupported({}, ConcatenationInfo(3, QuantizationInfo()), nullptr, reason,
                                                 sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Must have at least one input"));
    }

    SECTION("Incorrect input data format")
    {
        REQUIRE(queries.IsConcatenationSupported(
                    { TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC),
                      TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NCHW) },
                    ConcatenationInfo(3, QuantizationInfo()), nullptr, reason,
                    sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Input to concatenation must be NHWC or NHWCB"));
    }

    SECTION("Incorrect input data type")
    {
        REQUIRE(queries.IsConcatenationSupported(
                    { TensorInfo({ 1, 16, 16, 16 }, DataType::INT32_QUANTIZED, DataFormat::NHWC),
                      TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC) },
                    ConcatenationInfo(3, QuantizationInfo()), nullptr, reason,
                    sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Input tensors must be UINT8_QUANTIZED or INT8_QUANTIZED"));
    }

    SECTION("Invalid axis")
    {
        REQUIRE(queries.IsConcatenationSupported(
                    { TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC),
                      TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC) },
                    ConcatenationInfo(17, QuantizationInfo()), nullptr, reason,
                    sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Concatenation axis must refer to a valid dimension (0-3)"));
    }

    SECTION("Unsupported axis")
    {
        REQUIRE(queries.IsConcatenationSupported(
                    { TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC),
                      TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC) },
                    ConcatenationInfo(0, QuantizationInfo()), nullptr, reason,
                    sizeof(reason)) == SupportedLevel::EstimateOnly);
        REQUIRE(Contains(reason, "Concatenation cannot be performed along batch axis (axis 0)"));
    }

    SECTION("Incompatible dimensions (Height)")
    {
        REQUIRE(queries.IsConcatenationSupported(
                    { TensorInfo({ 1, 8, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC),
                      TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC) },
                    ConcatenationInfo(3, QuantizationInfo()), nullptr, reason,
                    sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(
            reason, "Input tensors must have the same size along all dimensions except the concatenation dimension"));
    }

    SECTION("Incompatible dimensions (Width)")
    {
        REQUIRE(queries.IsConcatenationSupported(
                    { TensorInfo({ 1, 16, 8, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC),
                      TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC) },
                    ConcatenationInfo(3, QuantizationInfo()), nullptr, reason,
                    sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(
            reason, "Input tensors must have the same size along all dimensions except the concatenation dimension"));
    }

    SECTION("Incompatible dimensions (Channels)")
    {
        REQUIRE(queries.IsConcatenationSupported(
                    { TensorInfo({ 1, 16, 16, 8 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC),
                      TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC) },
                    ConcatenationInfo(1, QuantizationInfo()), nullptr, reason,
                    sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(
            reason, "Input tensors must have the same size along all dimensions except the concatenation dimension"));
    }

    SECTION("Invalid output tensor info")
    {
        TensorInfo outputInfo({ 1, 16, 16, 31 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC);
        REQUIRE(queries.IsConcatenationSupported(
                    { TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC),
                      TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC) },
                    ConcatenationInfo(3, QuantizationInfo()), &outputInfo, reason,
                    sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Provided outputInfo is incorrect"));
    }

    SECTION("Output scale too small")
    {
        REQUIRE(queries.IsConcatenationSupported(
                    { TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC),
                      TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC) },
                    ConcatenationInfo(3, QuantizationInfo(0, 1. / 128.f)), nullptr, reason,
                    sizeof(reason)) == SupportedLevel::EstimateOnly);
        REQUIRE(Contains(reason, "Output scales must be bigger than input scale / 128"));
    }

    SECTION("Invalid zero point")
    {
        std::vector<TensorInfo> inputInfos = {
            TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC),
            TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC)
        };
        TensorInfo outputInfo({ 1, 16, 16, 32 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC,
                              QuantizationInfo(0, 1.0f));
        ConcatenationInfo ConcatInfo(3, QuantizationInfo(0, 1.0f));

        SECTION("Invalid input zero point")
        {
            inputInfos[0].m_QuantizationInfo.SetZeroPoint(-10);
            REQUIRE(queries.IsConcatenationSupported(inputInfos, ConcatInfo, &outputInfo, reason, sizeof(reason)) ==
                    SupportedLevel::Unsupported);
            INFO(reason);
            REQUIRE(Contains(reason, "Zero point out of range for at least one input info"));
        }

        SECTION("Invalid concatInfo zero point")
        {
            inputInfos[0].m_QuantizationInfo.SetZeroPoint(0);
            inputInfos[1].m_QuantizationInfo.SetZeroPoint(0);
            ConcatInfo.m_OutputQuantizationInfo.SetZeroPoint(-10);
            REQUIRE(queries.IsConcatenationSupported(inputInfos, ConcatInfo, &outputInfo, reason, sizeof(reason)) ==
                    SupportedLevel::Unsupported);
            INFO(reason);
            REQUIRE(Contains(reason, "Zero point out of range for concatInfo"));
        }
    }

    SECTION("Output scale just fits")
    {
        REQUIRE(queries.IsConcatenationSupported(
                    { TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC),
                      TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC) },
                    ConcatenationInfo(3, QuantizationInfo(0, 1.f / 127.99f)), nullptr, reason,
                    sizeof(reason)) == SupportedLevel::Supported);
    }

    SECTION("Successful case (output info provided)")
    {
        TensorInfo outputInfo({ 1, 16, 16, 32 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC);
        REQUIRE(queries.IsConcatenationSupported(
                    { TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC),
                      TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC) },
                    ConcatenationInfo(3, QuantizationInfo()), &outputInfo) == SupportedLevel::Supported);
    }

    SECTION("Successful case (output info provided)")
    {
        TensorInfo outputInfo({ 1, 16, 16, 32 }, DataType::INT8_QUANTIZED, DataFormat::NHWC);
        REQUIRE(queries.IsConcatenationSupported(
                    { TensorInfo({ 1, 16, 16, 16 }, DataType::INT8_QUANTIZED, DataFormat::NHWC),
                      TensorInfo({ 1, 16, 16, 16 }, DataType::INT8_QUANTIZED, DataFormat::NHWC) },
                    ConcatenationInfo(3, QuantizationInfo()), &outputInfo) == SupportedLevel::Supported);
    }

    SECTION("Successful case (output info filled in)")
    {
        TensorInfo outputInfo;
        REQUIRE(queries.IsConcatenationSupported(
                    { TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC),
                      TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC) },
                    ConcatenationInfo(3, QuantizationInfo()), &outputInfo) == SupportedLevel::Supported);
        REQUIRE(outputInfo == TensorInfo({ 1, 16, 16, 32 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC));
    }

    SECTION("Successful case (output info filled in)")
    {
        TensorInfo outputInfo;
        REQUIRE(queries.IsConcatenationSupported(
                    { TensorInfo({ 1, 16, 16, 16 }, DataType::INT8_QUANTIZED, DataFormat::NHWC),
                      TensorInfo({ 1, 16, 16, 16 }, DataType::INT8_QUANTIZED, DataFormat::NHWC) },
                    ConcatenationInfo(3, QuantizationInfo()), &outputInfo) == SupportedLevel::Supported);
        REQUIRE(outputInfo == TensorInfo({ 1, 16, 16, 32 }, DataType::INT8_QUANTIZED, DataFormat::NHWC));
    }
}
