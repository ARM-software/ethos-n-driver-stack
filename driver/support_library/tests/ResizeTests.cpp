//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"
#include "../include/ethosn_support_library/SupportQueries.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>

using namespace ethosn::support_library;

TEST_CASE("ResizeSupported")
{
    const auto inputDataType = GENERATE(DataType::INT8_QUANTIZED, DataType::UINT8_QUANTIZED);

    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Supported configuration
    TensorInfo input({ 1, 16, 16, 16 }, inputDataType, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
    TensorInfo output({ 1, 32, 32, 16 }, inputDataType, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
    REQUIRE(queries.IsResizeSupported(ResizeInfo(ResizeAlgorithm::BILINEAR, 32, 32, QuantizationInfo(0, 1.0f)), input,
                                      &output) == SupportedLevel::Supported);

    // Test support string reporting with incorrect outputInfo.
    constexpr size_t reasonLength = 256;
    char reason[reasonLength + 1];
    REQUIRE(queries.IsResizeSupported(ResizeInfo(ResizeAlgorithm::BILINEAR, 31, 31, QuantizationInfo(0, 1.0f)), input,
                                      &output, reason, reasonLength) == SupportedLevel::Unsupported);
    REQUIRE(std::string(reason) == "Provided outputInfo is incorrect");

    // Test support string reporting with incorrect resizeInfo (m_NewHeight).
    REQUIRE(queries.IsResizeSupported(ResizeInfo(ResizeAlgorithm::BILINEAR, 30, 31, QuantizationInfo(0, 1.0f)), input,
                                      &output, reason, reasonLength) == SupportedLevel::Unsupported);
    REQUIRE(std::string(reason) == "Requested height isn't supported");

    // Test support string reporting with incorrect resizeInfo (m_NewWidth).
    REQUIRE(queries.IsResizeSupported(ResizeInfo(ResizeAlgorithm::BILINEAR, 31, 30, QuantizationInfo(0, 1.0f)), input,
                                      &output, reason, reasonLength) == SupportedLevel::Unsupported);
    REQUIRE(std::string(reason) == "Requested width isn't supported");

    // Test support string reporting with invalid zero point for input
    input.m_QuantizationInfo.SetZeroPoint(-129);
    REQUIRE(queries.IsResizeSupported(ResizeInfo(ResizeAlgorithm::BILINEAR, 32, 32, QuantizationInfo(0, 1.0f)), input,
                                      nullptr, reason, reasonLength) == SupportedLevel::Unsupported);
    REQUIRE(std::string(reason) == "Zero point out of range for input info");

    // Test support string reporting with invalid zero point for resizeInfo
    input.m_QuantizationInfo.SetZeroPoint(0);
    REQUIRE(queries.IsResizeSupported(ResizeInfo(ResizeAlgorithm::BILINEAR, 32, 32, QuantizationInfo(-129, 1.0f)),
                                      input, nullptr, reason, reasonLength) == SupportedLevel::Unsupported);
    REQUIRE(std::string(reason) == "Zero point out of range for resizeInfo");
}
