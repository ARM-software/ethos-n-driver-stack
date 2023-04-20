//
// Copyright © 2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"
#include "../include/ethosn_support_library/SupportQueries.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>

using namespace ethosn::support_library;

TEST_CASE("TanhSupported")
{
    char reason[1024];
    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    SECTION("Supported configuration")
    {
        const auto inputDataType = GENERATE(DataType::INT8_QUANTIZED, DataType::UINT8_QUANTIZED);
        const QuantizationInfo outQuantization((inputDataType == DataType::INT8_QUANTIZED) ? 0 : 128, 1.0f / 128);
        TensorInfo input({ 1, 16, 16, 16 }, inputDataType, DataFormat::NHWC, QuantizationInfo(127, 1.0f));
        TensorInfo output({ 1, 16, 16, 16 }, inputDataType, DataFormat::NHWC, outQuantization);
        REQUIRE(queries.IsTanhSupported(input, &output) == SupportedLevel::Supported);
    }

    SECTION("OutputInfo nullptr")
    {
        const auto inputDataType = GENERATE(DataType::UINT8_QUANTIZED, DataType::UINT8_QUANTIZED);
        TensorInfo input({ 1, 16, 16, 16 }, inputDataType, DataFormat::NHWC, QuantizationInfo(127, 1.0f));
        REQUIRE(queries.IsTanhSupported(input, nullptr) == SupportedLevel::Supported);
    }

    SECTION("Output info filled in")
    {
        TensorInfo output;
        TensorInfo input({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(127, 1.0f));
        REQUIRE(queries.IsTanhSupported(input, &output) == SupportedLevel::Supported);
        REQUIRE(output == TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC,
                                     QuantizationInfo(128, 1.0f / 128)));
    }

    SECTION("Wrong quantization")
    {
        TensorInfo input({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(127, 1.0f));
        TensorInfo output({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        REQUIRE(queries.IsTanhSupported(input, &output) == SupportedLevel::Unsupported);
    }

    SECTION("Wrong size")
    {
        TensorInfo input({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(127, 1.0f));
        TensorInfo output({ 1, 8, 8, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC,
                          QuantizationInfo(0, 1.0f / 256));
        REQUIRE(queries.IsTanhSupported(input, &output) == SupportedLevel::Unsupported);
    }

    SECTION("Invalid zero point")
    {
        TensorInfo input({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(-10, 1.0f));
        REQUIRE(queries.IsTanhSupported(input, nullptr, reason, sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Zero point out of range for input info"));
    }
}
