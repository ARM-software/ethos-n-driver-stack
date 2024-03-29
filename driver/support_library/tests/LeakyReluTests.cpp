//
// Copyright © 2018-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"
#include "../include/ethosn_support_library/SupportQueries.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>

using namespace ethosn::support_library;

TEST_CASE("LeakyReluSupported Supported")
{
    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Supported configuration
    TensorInfo input({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
    TensorInfo output({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
    REQUIRE(queries.IsLeakyReluSupported(LeakyReluInfo(0.1f, QuantizationInfo(0, 1.0f)), input, &output) ==
            SupportedLevel::Supported);
}

TEST_CASE("LeakyReluSupported EstimateOnly alpha >= 1")
{
    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Supported configuration
    TensorInfo input({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
    TensorInfo output({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
    REQUIRE(queries.IsLeakyReluSupported(LeakyReluInfo(1.f, QuantizationInfo(0, 1.0f)), input, &output) ==
            SupportedLevel::EstimateOnly);
}

TEST_CASE("LeakyReluSupported EstimateOnly negative alpha")
{
    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Supported configuration
    TensorInfo input({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
    TensorInfo output({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
    REQUIRE(queries.IsLeakyReluSupported(LeakyReluInfo(-1.f, QuantizationInfo(0, 1.0f)), input, &output) ==
            SupportedLevel::EstimateOnly);
}

TEST_CASE("LeakyReluSupported Unsupported zero point out of range")
{
    char reason[1024];
    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Check unsupported for zero point out of range in inputinfo
    TensorInfo input({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(-10, 1.0f));
    REQUIRE(queries.IsLeakyReluSupported(LeakyReluInfo(0.1f, QuantizationInfo(0, 1.0f)), input, nullptr, reason,
                                         sizeof(reason)) == SupportedLevel::Unsupported);
    REQUIRE(Contains(reason, "Zero point out of range for input info"));

    // Check unsupported for zero point out of range for outputinfo
    input.m_QuantizationInfo.SetZeroPoint(0);
    REQUIRE(queries.IsLeakyReluSupported(LeakyReluInfo(0.1f, QuantizationInfo(-10, 1.0f)), input, nullptr, reason,
                                         sizeof(reason)) == SupportedLevel::Unsupported);
    REQUIRE(Contains(reason, "Zero point out of range for leakyReluInfo"));
}

/// Tests that a network comprising a single leaky relu creates an mce operation beforehand.
TEST_CASE("LeakyRelu EstimateOnly")
{
    // Create the estimation network
    CompilationOptions options;
    std::shared_ptr<Network> network = CreateEstimationNetwork(GetRawDefaultCapabilities());
    std::shared_ptr<Operand> input   = AddInput(network, TensorInfo({ 1, 16, 16, 16 })).tensor;
    std::shared_ptr<Operand> leakyRelu =
        AddLeakyRelu(network, *input, LeakyReluInfo(0.1f, QuantizationInfo(0, 1.0f))).tensor;
    std::shared_ptr<Output> output = AddOutput(network, *leakyRelu).tensor;

    // Estimate it
    EstimationOptions estimationOptions{};
    estimationOptions.m_Current = true;

    std::vector<PassPerformanceData> perfData = EstimatePerformance(*network, options, estimationOptions).m_Stream;

    // Check that it has completed.
    REQUIRE(perfData.size() > 0U);
    // Check that it's a Mce plus Fused Ple operation.
    REQUIRE(perfData.at(0).m_Stats.m_Mce.m_CycleCount == 32U);
    REQUIRE(perfData.at(0).m_Stats.m_Ple.m_NumOfPatches == 16U);
}
