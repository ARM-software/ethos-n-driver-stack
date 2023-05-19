//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"
#include "../include/ethosn_support_library/SupportQueries.hpp"
#include "../src/Compiler.hpp"
#include "../src/Utils.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>
#include <ethosn_command_stream/CommandStreamBuffer.hpp>

using namespace ethosn::support_library;
using namespace ethosn::support_library::utils;

TEST_CASE("SplitSupported")
{
    char reason[1024];

    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Not enough splits
    REQUIRE(queries.IsSplitSupported(
                TensorInfo({ 1, 16, 16, 64 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)),
                SplitInfo(3, {}), nullptr, reason, sizeof(reason)) == SupportedLevel::Unsupported);
    REQUIRE(Contains(reason, "Must have at least 1 output"));

    // Unsupported datatype
    REQUIRE(queries.IsSplitSupported(
                TensorInfo({ 1, 16, 16, 64 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)),
                SplitInfo(3, { 32, 32 }), nullptr, reason, sizeof(reason)) == SupportedLevel::Unsupported);
    REQUIRE(Contains(reason, "Input tensor must be UINT8_QUANTIZED or INT8_QUANTIZED"));

    // Unsupported data format
    REQUIRE(queries.IsSplitSupported(
                TensorInfo({ 1, 16, 16, 64 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, QuantizationInfo(1, 2)),
                SplitInfo(3, { 32, 32 }), nullptr, reason, sizeof(reason)) == SupportedLevel::Unsupported);
    REQUIRE(Contains(reason, "Input tensor must be NHWC or NHWCB"));

    // Invalid axis
    REQUIRE(queries.IsSplitSupported(
                TensorInfo({ 1, 16, 16, 64 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)),
                SplitInfo(7, { 32, 32 }), nullptr, reason, sizeof(reason)) == SupportedLevel::Unsupported);
    REQUIRE(Contains(reason, "Axis must refer to a valid dimension"));

    // Invalid sum of sizes
    REQUIRE(queries.IsSplitSupported(
                TensorInfo({ 1, 16, 16, 64 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)),
                SplitInfo(3, { 32, 16 }), nullptr, reason, sizeof(reason)) == SupportedLevel::Unsupported);
    REQUIRE(Contains(reason, "Sizes must sum to the total size of the input tensor along the split axis"));

    // Invalid number of outputInfos provided
    {
        std::vector<TensorInfo> outputInfos(3);
        REQUIRE(queries.IsSplitSupported(
                    TensorInfo({ 1, 16, 16, 64 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)),
                    SplitInfo(3, { 32, 32 }), &outputInfos, reason, sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Provided outputInfos array has incorrect size"));
    }

    // Invalid outputInfo provided
    {
        std::vector<TensorInfo> outputInfos{
            TensorInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)),
            TensorInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)),
        };
        REQUIRE(queries.IsSplitSupported(
                    TensorInfo({ 1, 16, 16, 64 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)),
                    SplitInfo(3, { 32, 32 }), &outputInfos, reason, sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Provided outputInfo at index 0 is incorrect"));
    }

    // Unsupported axis
    REQUIRE(queries.IsSplitSupported(
                TensorInfo({ 1, 16, 16, 64 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)),
                SplitInfo(0, { 0, 1 }), nullptr, reason, sizeof(reason)) == SupportedLevel::Unsupported);
    REQUIRE(Contains(reason, "Split cannot be performed along batch axis"));

    // Zero point outside of valid range
    {
        REQUIRE(queries.IsSplitSupported(TensorInfo({ 1, 16, 16, 64 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC,
                                                    QuantizationInfo(-10, 2)),
                                         SplitInfo(3, { 30, 34 }), nullptr, reason,
                                         sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Zero point out of range for input info"));
    }

    // Successful case (output info provided)
    {
        std::vector<TensorInfo> outputInfos{
            TensorInfo({ 1, 16, 16, 32 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)),
            TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)),
            TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2))
        };
        REQUIRE(queries.IsSplitSupported(
                    TensorInfo({ 1, 16, 16, 64 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)),
                    SplitInfo(3, { 32, 16, 16 }), &outputInfos) == SupportedLevel::Supported);
    }

    // Successful case (output infos filled in)
    {
        std::vector<TensorInfo> outputInfos(3);
        REQUIRE(queries.IsSplitSupported(
                    TensorInfo({ 1, 16, 16, 64 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)),
                    SplitInfo(3, { 32, 16, 16 }), &outputInfos) == SupportedLevel::Supported);
        REQUIRE(outputInfos.size() == 3);
        REQUIRE(outputInfos[0] ==
                TensorInfo({ 1, 16, 16, 32 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)));
        REQUIRE(outputInfos[1] ==
                TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)));
        REQUIRE(outputInfos[2] ==
                TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(1, 2)));
    }
}
