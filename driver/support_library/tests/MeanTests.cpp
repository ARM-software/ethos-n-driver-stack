//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"
#include "../include/ethosn_support_library/SupportQueries.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>
#include <ethosn_command_stream/CommandStreamBuffer.hpp>

using namespace ethosn::support_library;

TEST_CASE("MeanXySupported", "[IsSupported]")
{
    char reason[1024];
    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    GIVEN("An Input TensorInfo with supported dimensions")
    {
        TensorInfo input({ 1, 7, 7, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));

        THEN("MeanXy shall be supported")
        {
            REQUIRE(queries.IsMeanXySupported(input, nullptr, reason, sizeof(reason)) == SupportedLevel::Supported);
        }
    }

    GIVEN("An Input TensorInfo with supported dimensions")
    {
        TensorInfo input({ 1, 8, 8, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));

        THEN("MeanXy shall be supported")
        {
            REQUIRE(queries.IsMeanXySupported(input, nullptr, reason, sizeof(reason)) == SupportedLevel::Supported);
        }
    }

    GIVEN("An Input TensorInfo with unsupported dimensions")
    {
        TensorInfo input({ 1, 6, 6, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));

        THEN("MeanXy shall not be supported")
        {
            REQUIRE(queries.IsMeanXySupported(input, nullptr, reason, sizeof(reason)) == SupportedLevel::Unsupported);
            INFO(reason);
            REQUIRE(Contains(reason, "MeanXy is supported for 7x7 and 8x8 as HeightxWidth only"));
        }
    }

    GIVEN("An Output TensorInfo with unsupported dimensions")
    {
        TensorInfo input({ 1, 7, 7, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo output({ 1, 7, 7, 1 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));

        THEN("MeanXy shall not be supported")
        {
            REQUIRE(queries.IsMeanXySupported(input, &output, reason, sizeof(reason)) == SupportedLevel::Unsupported);
            INFO(reason);
            REQUIRE(Contains(reason, "Provided outputInfo is incorrect"));
        }
    }

    GIVEN("An Output TensorInfo of size 0")
    {
        TensorInfo input({ 1, 7, 7, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo output({ 0, 0, 0, 0 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));

        THEN("Outputinfo should be {1, 1, 1, 16}")
        {
            TensorInfo expectedOutput({ 1, 1, 1, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC,
                                      QuantizationInfo(0, 1.0f));

            REQUIRE(queries.IsMeanXySupported(input, &output, reason, sizeof(reason)) == SupportedLevel::Supported);
            REQUIRE(output == expectedOutput);
        }
    }

    GIVEN("Input TensorInfo with out of range zero point")
    {
        TensorInfo input({ 1, 7, 7, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(-10, 1.0f));

        THEN("MeanXy shall not be supported")
        {
            REQUIRE(queries.IsMeanXySupported(input, nullptr, reason, sizeof(reason)) == SupportedLevel::Unsupported);
            REQUIRE(Contains(reason, "Zero point out of range for input info"));
        }
    }
}
