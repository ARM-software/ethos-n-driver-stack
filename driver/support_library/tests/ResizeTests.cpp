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

    // Test support string reporting with incorrect resizeInfo with odd width, even height.
    REQUIRE(queries.IsResizeSupported(ResizeInfo(ResizeAlgorithm::BILINEAR, 32, 31, QuantizationInfo(0, 1.0f)), input,
                                      &output, reason, reasonLength) == SupportedLevel::Unsupported);
    REQUIRE(std::string(reason) == "Requested width and height must be both even or both odd");

    // Test support string reporting with incorrect resizeInfo with even width, odd height.
    REQUIRE(queries.IsResizeSupported(ResizeInfo(ResizeAlgorithm::BILINEAR, 31, 32, QuantizationInfo(0, 1.0f)), input,
                                      &output, reason, reasonLength) == SupportedLevel::Unsupported);
    REQUIRE(std::string(reason) == "Requested width and height must be both even or both odd");
}

/// Tests that a network comprising a resize is converted in an identity depthwise convolution with the correct upsample parameter.
TEST_CASE("Add Resize to a network")
{
    // Create the network
    CompilationOptions options       = GetDefaultCompilationOptions();
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultCapabilities());
    std::shared_ptr<Operand> input   = AddInput(network, TensorInfo({ 1, 16, 16, 16 })).tensor;
    std::shared_ptr<Operand> resize =
        AddResize(network, *input, ResizeInfo(ResizeAlgorithm::BILINEAR, 31, 31, QuantizationInfo(0, 1.0f))).tensor;
    std::shared_ptr<Output> output = AddOutput(network, *resize).tensor;

    // Compile it
    std::vector<std::unique_ptr<CompiledNetwork>> compiledNetwork =
        ethosn::support_library::Compile(*network, GetDefaultCompilationOptions());

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

    // Check that the conv command is as expected.
    REQUIRE(convCmds.size() == 1);
    REQUIRE(convCmds[0].m_MceData().m_OutputShape() == TensorShape{ 1, 31, 31, 16 });
    REQUIRE(convCmds[0].m_MceData().m_UpsampleType() == UpsampleType::BILINEAR);
}
