//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"
#include "../include/ethosn_support_library/SupportQueries.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>

using namespace ethosn::support_library;

TEST_CASE("ReshapeSupported")
{
    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Supported configuration
    TensorInfo input({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
    TensorInfo output({ 1, 16, 8, 32 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
    REQUIRE(queries.IsReshapeSupported({ 1, 16, 8, 32 }, input, &output) == SupportedLevel::Supported);
    REQUIRE(input.m_Dimensions[0] * input.m_Dimensions[1] * input.m_Dimensions[2] * input.m_Dimensions[3] ==
            output.m_Dimensions[0] * output.m_Dimensions[1] * output.m_Dimensions[2] * output.m_Dimensions[3]);
}

TEST_CASE("ReshapeNotSupported")
{
    char reason[1024];
    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Not Supported configuration
    TensorInfo input({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
    TensorInfo output({ 1, 16, 1, 32 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
    REQUIRE(queries.IsReshapeSupported({ 1, 16, 1, 32 }, input, &output) == SupportedLevel::Unsupported);

    // Invalid zero point
    input.m_Dimensions = { 1, 16, 2, 16 };
    input.m_QuantizationInfo.SetZeroPoint(-10);
    REQUIRE(queries.IsReshapeSupported({ 1, 16, 1, 32 }, input, nullptr, reason, sizeof(reason)) ==
            SupportedLevel::Unsupported);
    REQUIRE(Contains(reason, "Zero point out of range for input info"));
}
