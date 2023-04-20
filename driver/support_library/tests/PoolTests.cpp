//
// Copyright © 2018-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"
#include "../include/ethosn_support_library/SupportQueries.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>
#include <ethosn_command_stream/CommandStreamBuffer.hpp>

using namespace ethosn::support_library;

namespace
{

struct Xy
{
    uint32_t x;
    uint32_t y;
};

SupportedLevel IsPoolingSupportedImpl(SupportQueries& queries,
                                      const Xy& inSize,
                                      const Xy& kSize,
                                      const Xy& stride,
                                      const Padding& padding,
                                      const PoolingType poolingType)
{
    const PoolingInfo poolingInfo(kSize.x, kSize.y, stride.x, stride.y, padding, poolingType);

    const TensorInfo input({ 1, inSize.y, inSize.x, 16 });

    const uint32_t outputSizeY = ((inSize.y - kSize.y + padding.m_Top + padding.m_Bottom) / stride.y) + 1;
    const uint32_t outputSizeX = ((inSize.x - kSize.x + padding.m_Left + padding.m_Right) / stride.x) + 1;

    TensorInfo output({ 1, outputSizeY, outputSizeX, 16 });

    return queries.IsPoolingSupported(poolingInfo, input, &output);
}
}    // namespace

TEST_CASE("PoolingSupported")
{
    char reason[1024];

    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    const Padding noPad    = { 0, 0, 0, 0 };
    const Padding padAfter = { 0, 1, 0, 1 };
    const Padding padAll   = { 1, 1, 1, 1 };

    // Invalid pooling size
    {
        TensorInfo input({ 1, 10, 10, 16 });
        PoolingInfo poolingInfo(0, 0, 2, 2, { 0, 0, 0, 0 }, PoolingType::MAX);
        REQUIRE(queries.IsPoolingSupported(poolingInfo, input, nullptr, reason, sizeof(reason)) ==
                SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Invalid pooling size/stride"));
    }

    // Invalid pooling stride
    {
        TensorInfo input({ 1, 10, 10, 16 });
        PoolingInfo poolingInfo(2, 2, 0, 0, { 0, 0, 0, 0 }, PoolingType::MAX);
        REQUIRE(queries.IsPoolingSupported(poolingInfo, input, nullptr, reason, sizeof(reason)) ==
                SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Invalid pooling size/stride"));
    }

    // Incorrect output info
    {
        TensorInfo input({ 1, 10, 10, 16 });
        PoolingInfo poolingInfo(5, 5, 2, 2, { 0, 0, 0, 0 }, PoolingType::MAX);
        TensorInfo output({ 1, 2, 3, 4 });
        REQUIRE(queries.IsPoolingSupported(poolingInfo, input, &output, reason, sizeof(reason)) ==
                SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Provided outputInfo is incorrect"));
    }

    // Input and output XY cannot be fit into SRAM (Z split is possible)
    {
        TensorInfo input({ 1, 480, 33, 64 });
        PoolingInfo poolingInfo(3, 3, 1, 1, { 1, 1, 1, 1 }, PoolingType::AVG);
        REQUIRE(queries.IsPoolingSupported(poolingInfo, input, nullptr, reason, sizeof(reason)) ==
                SupportedLevel::EstimateOnly);
        REQUIRE(Contains(reason, "AVG pooling 3x3_1_1: maximum input width x height cannot fit into SRAM"));
    }

    // Input and output XY can be fit into SRAM (Z split is possible)
    {
        TensorInfo input({ 1, 480, 32, 64 });
        PoolingInfo poolingInfo(3, 3, 1, 1, { 1, 1, 1, 1 }, PoolingType::AVG);
        REQUIRE(queries.IsPoolingSupported(poolingInfo, input, nullptr, reason, sizeof(reason)) ==
                SupportedLevel::Supported);
    }

    // Input and output XY cannot be fit into SRAM (Z split is not possible)
    {
        TensorInfo input({ 1, 481, 64, 16 });
        PoolingInfo poolingInfo(3, 3, 1, 1, { 1, 1, 1, 1 }, PoolingType::AVG);
        REQUIRE(queries.IsPoolingSupported(poolingInfo, input, nullptr, reason, sizeof(reason)) ==
                SupportedLevel::EstimateOnly);
        REQUIRE(Contains(reason, "AVG pooling 3x3_1_1: maximum input width x height cannot fit into SRAM"));
    }

    // Input and output XY can be fit into SRAM (Z split is not possible)
    {
        TensorInfo input({ 1, 480, 64, 16 });
        PoolingInfo poolingInfo(3, 3, 1, 1, { 1, 1, 1, 1 }, PoolingType::AVG);
        REQUIRE(queries.IsPoolingSupported(poolingInfo, input, nullptr, reason, sizeof(reason)) ==
                SupportedLevel::Supported);
    }

    // Invalid zero point for input info
    {
        TensorInfo input({ 1, 16, 16, 32 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(-10, 1.0f));
        PoolingInfo poolingInfo(3, 3, 1, 1, { 1, 1, 1, 1 }, PoolingType::AVG);
        REQUIRE(queries.IsPoolingSupported(poolingInfo, input, nullptr, reason, sizeof(reason)) ==
                SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Zero point out of range for input info"));
    }

    // EstimateOnly
    REQUIRE(IsPoolingSupportedImpl(queries, { 16, 16 }, { 2, 2 }, { 1, 1 }, noPad, PoolingType::MAX) ==
            SupportedLevel::EstimateOnly);
    REQUIRE(IsPoolingSupportedImpl(queries, { 16, 16 }, { 3, 3 }, { 1, 1 }, noPad, PoolingType::MAX) ==
            SupportedLevel::EstimateOnly);
    REQUIRE(IsPoolingSupportedImpl(queries, { 16, 16 }, { 3, 3 }, { 1, 1 }, noPad, PoolingType::AVG) ==
            SupportedLevel::EstimateOnly);
    REQUIRE(IsPoolingSupportedImpl(queries, { 16, 16 }, { 5, 5 }, { 3, 3 }, noPad, PoolingType::AVG) ==
            SupportedLevel::EstimateOnly);
    REQUIRE(IsPoolingSupportedImpl(queries, { 16, 16 }, { 7, 7 }, { 1, 1 }, noPad, PoolingType::AVG) ==
            SupportedLevel::EstimateOnly);
    REQUIRE(IsPoolingSupportedImpl(queries, { 16, 16 }, { 7, 7 }, { 2, 2 }, noPad, PoolingType::AVG) ==
            SupportedLevel::EstimateOnly);
    REQUIRE(IsPoolingSupportedImpl(queries, { 16, 16 }, { 8, 8 }, { 1, 1 }, noPad, PoolingType::AVG) ==
            SupportedLevel::EstimateOnly);
    REQUIRE(IsPoolingSupportedImpl(queries, { 16, 16 }, { 8, 8 }, { 2, 2 }, noPad, PoolingType::AVG) ==
            SupportedLevel::EstimateOnly);

    // Supported
    REQUIRE(IsPoolingSupportedImpl(queries, { 16, 16 }, { 1, 1 }, { 2, 2 }, noPad, PoolingType::MAX) ==
            SupportedLevel::Supported);
    REQUIRE(IsPoolingSupportedImpl(queries, { 16, 16 }, { 2, 2 }, { 2, 2 }, noPad, PoolingType::MAX) ==
            SupportedLevel::Supported);
    REQUIRE(IsPoolingSupportedImpl(queries, { 17, 17 }, { 2, 2 }, { 2, 2 }, padAfter, PoolingType::MAX) ==
            SupportedLevel::Supported);
    REQUIRE(IsPoolingSupportedImpl(queries, { 17, 17 }, { 3, 3 }, { 2, 2 }, noPad, PoolingType::MAX) ==
            SupportedLevel::Supported);
    REQUIRE(IsPoolingSupportedImpl(queries, { 16, 16 }, { 3, 3 }, { 2, 2 }, padAfter, PoolingType::MAX) ==
            SupportedLevel::Supported);
    REQUIRE(IsPoolingSupportedImpl(queries, { 16, 16 }, { 3, 3 }, { 1, 1 }, padAll, PoolingType::AVG) ==
            SupportedLevel::Supported);
    REQUIRE(IsPoolingSupportedImpl(queries, { 7, 7 }, { 7, 7 }, { 1, 1 }, noPad,
                                   PoolingType::AVG) == SupportedLevel::Supported);    // mean cases
    REQUIRE(IsPoolingSupportedImpl(queries, { 8, 8 }, { 8, 8 }, { 1, 1 }, noPad,
                                   PoolingType::AVG) == SupportedLevel::Supported);    // mean cases
}
