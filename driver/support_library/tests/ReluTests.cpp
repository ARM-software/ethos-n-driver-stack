//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"
#include "../include/ethosn_support_library/SupportQueries.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>
#include <ethosn_command_stream/CommandStreamBuffer.hpp>

using namespace ethosn::support_library;

TEST_CASE("ReluSupported")
{
    const auto inputDataType = GENERATE(DataType::INT8_QUANTIZED, DataType::UINT8_QUANTIZED);

    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Supported configuration
    TensorInfo input({ 1, 16, 16, 16 }, inputDataType, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
    TensorInfo output({ 1, 16, 16, 16 }, inputDataType, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
    REQUIRE(queries.IsReluSupported(ReluInfo(0, 255), input, &output) == SupportedLevel::Supported);

    // Test support string reporting with hi limit < lo limit, which would not
    // make sense to support
    constexpr size_t reasonLength = 256;
    char reason[reasonLength + 1];
    REQUIRE(queries.IsReluSupported(ReluInfo(0x42, 42), input, &output, reason, reasonLength) ==
            SupportedLevel::Unsupported);
    REQUIRE(std::string(reason) == "Relu has lower bound > upper bound");

    // Test for zero point check
    input.m_QuantizationInfo.SetZeroPoint(-129);
    REQUIRE(queries.IsReluSupported(ReluInfo(0, 255), input, nullptr, reason, reasonLength) ==
            SupportedLevel::Unsupported);
    REQUIRE(std::string(reason) == "Zero point out of range for input info");
}

/// Tests that a network comprising a single Relu creates an identity depthwise convolution beforehand.
TEST_CASE("SingleRelu")
{
    // Create the network
    CompilationOptions options;
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultCapabilities());
    std::shared_ptr<Operand> input   = AddInput(network, TensorInfo({ 1, 16, 16, 16 })).tensor;
    std::shared_ptr<Operand> relu    = AddRelu(network, *input, ReluInfo(10, 20)).tensor;
    std::shared_ptr<Output> output   = AddOutput(network, *relu).tensor;

    // Compile it
    std::vector<std::unique_ptr<CompiledNetwork>> compiledNetwork =
        ethosn::support_library::Compile(*network, CompilationOptions());

    // Extract all the conv commands
    using namespace ethosn::command_stream;
    CommandStream cmdStream = GetCommandStream(compiledNetwork[0].get());
    std::vector<McePle> convCmds;
    for (const auto& cmdHeader : cmdStream)
    {
        if (cmdHeader.m_Opcode() == Opcode::OPERATION_MCE_PLE)
        {
            convCmds.push_back(cmdHeader.GetCommand<Opcode::OPERATION_MCE_PLE>()->m_Data());
        }
    }

    // Check that the conv commands are as expected. There should be one which has a relu afterwards.
    REQUIRE(convCmds.size() == 1);
    REQUIRE((convCmds[0].m_MceData().m_ActivationMin() == 10 && convCmds[0].m_MceData().m_ActivationMax() == 20));
}
