//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/SupportQueries.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>

using namespace ethosn::support_library;

TEST_CASE("queries.IsDepthToSpaceSupported")
{
    char reason[1024];

    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    SECTION("Input incorrect data type")
    {
        TensorInfo inputInfo({ 1, 1, 1, 1 }, DataType::INT32_QUANTIZED);
        REQUIRE(queries.IsDepthToSpaceSupported(inputInfo, DepthToSpaceInfo(2), nullptr, reason, sizeof(reason)) ==
                SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Input to depth to space must be UINT8_QUANTIZED or INT8_QUANTIZED"));
    }

    SECTION("Input incorrect data format")
    {
        TensorInfo inputInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO);
        REQUIRE(queries.IsDepthToSpaceSupported(inputInfo, DepthToSpaceInfo(2), nullptr, reason, sizeof(reason)) ==
                SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "must be NHWC or NHWCB"));
    }

    SECTION("Input tensor size incompatible with block size")
    {
        TensorInfo inputInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC);
        REQUIRE(queries.IsDepthToSpaceSupported(inputInfo, DepthToSpaceInfo(2), nullptr, reason, sizeof(reason)) ==
                SupportedLevel::Unsupported);
        REQUIRE(
            Contains(reason, "Number of channels of input must be an exact multiple of the square of the block size"));
    }

    SECTION("Incorrect output info")
    {
        TensorInfo inputInfo({ 1, 1, 1, 4 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC);
        TensorInfo outputInfo({ 1, 2, 3, 4 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC);
        REQUIRE(queries.IsDepthToSpaceSupported(inputInfo, DepthToSpaceInfo(2), &outputInfo, reason, sizeof(reason)) ==
                SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Provided outputInfo is incorrect"));
    }

    SECTION("EstimateOnly block size")
    {
        TensorInfo inputInfo({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC);
        REQUIRE(queries.IsDepthToSpaceSupported(inputInfo, DepthToSpaceInfo(1), nullptr, reason, sizeof(reason)) ==
                SupportedLevel::EstimateOnly);
        REQUIRE(Contains(reason, "Only block size of 2 is supported"));
    }

    SECTION("Successful cases")
    {
        const auto inputDataType = GENERATE(DataType::INT8_QUANTIZED, DataType::UINT8_QUANTIZED);

        TensorInfo inputInfo({ 1, 1, 1, 4 }, inputDataType, DataFormat::NHWC);

        SECTION("Output info not provided")
        {
            REQUIRE(queries.IsDepthToSpaceSupported(inputInfo, DepthToSpaceInfo(2), nullptr, reason, sizeof(reason)) ==
                    SupportedLevel::Supported);
            INFO(reason);
        }

        SECTION("Output info filled in for us")
        {
            TensorInfo outputInfo;
            REQUIRE(queries.IsDepthToSpaceSupported(inputInfo, DepthToSpaceInfo(2), &outputInfo, reason,
                                                    sizeof(reason)) == SupportedLevel::Supported);
            INFO(reason);
            REQUIRE(outputInfo == TensorInfo({ 1, 2, 2, 1 }, inputDataType, DataFormat::NHWC));
        }

        SECTION("Output info provided")
        {
            TensorInfo outputInfo({ 1, 2, 2, 1 }, inputDataType, DataFormat::NHWC);
            REQUIRE(queries.IsDepthToSpaceSupported(inputInfo, DepthToSpaceInfo(2), &outputInfo, reason,
                                                    sizeof(reason)) == SupportedLevel::Supported);
            INFO(reason);
        }
    }
}
