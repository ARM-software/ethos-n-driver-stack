//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"
#include "../include/ethosn_support_library/SupportQueries.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>
#include <ethosn_command_stream/CommandStreamBuffer.hpp>

using namespace ethosn::support_library;

TEST_CASE("ConstantSupported")
{
    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N57));

    TensorInfo info({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
    REQUIRE(queries.IsConstantSupported(info) == SupportedLevel::Supported);
}

TEST_CASE("Constant used as input to operation fails to compile")
{
    // Create the network
    CompilationOptions options       = GetDefaultCompilationOptions();
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultCapabilities());

    std::shared_ptr<Constant> constant =
        AddConstant(network, TensorInfo({ 1, 1, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, { 0, 1.f }),
                    std::vector<uint8_t>(16 * 16 * 16, 0).data())
            .tensor;
    auto constantOperand           = GetOperand(constant);
    std::shared_ptr<Output> output = AddOutput(network, *constantOperand).tensor;

    // Compile it
    std::vector<std::unique_ptr<CompiledNetwork>> compiledNetwork = ethosn::support_library::Compile(*network, options);

    REQUIRE(compiledNetwork.size() == 0);
}

/// Checks that the support_library compiles the network as expected
/// when an unconnected constant is added to the graph.
TEST_CASE("Constant unconnected")
{
    constexpr float scale = 0.5;
    TensorInfo inputInfo0{
        { { 1, 16, 16, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, scale },
    };
    TensorInfo inputInfo1{
        { { 1, 16, 16, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, scale },
    };

    TensorInfo constantInfo{
        { { 1, 1, 1, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, scale },
    };

    const std::vector<uint8_t> constData(constantInfo.m_Dimensions[0] * constantInfo.m_Dimensions[1] *
                                         constantInfo.m_Dimensions[2] * constantInfo.m_Dimensions[3]);

    CompilationOptions options       = GetDefaultCompilationOptions();
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultCapabilities());

    // Build up the network
    std::shared_ptr<Operand> input0    = AddInput(network, inputInfo0).tensor;
    std::shared_ptr<Operand> input1    = AddInput(network, inputInfo1).tensor;
    std::shared_ptr<Constant> constant = AddConstant(network, constantInfo, constData.data()).tensor;
    std::shared_ptr<Operand> addition  = AddAddition(network, *input0, *input1, inputInfo0.m_QuantizationInfo).tensor;
    std::shared_ptr<Output> output     = AddOutput(network, *addition).tensor;

    std::vector<std::unique_ptr<CompiledNetwork>> compiledNetwork =
        ethosn::support_library::Compile(*network, GetDefaultCompilationOptions());

    // Extract the PleOnly operations
    using namespace ethosn::command_stream;
    CommandStream cmdStream = GetCommandStream(compiledNetwork[0].get());
    std::vector<PleOnly> commands;
    for (const auto& cmdHeader : cmdStream)
    {
        if (cmdHeader.m_Opcode() == Opcode::OPERATION_PLE_ONLY)
        {
            commands.push_back(cmdHeader.GetCommand<Opcode::OPERATION_PLE_ONLY>()->m_Data());
        }
    }

    REQUIRE(commands.size() == 1);
    REQUIRE(commands[0].m_NumInputInfos() == 2u);
    REQUIRE(commands[0].m_InputInfo().m_TensorShape() == TensorShape{ 1, 16, 16, 16 });
    REQUIRE(commands[0].m_InputInfo2().m_TensorShape() == TensorShape{ 1, 16, 16, 16 });
    REQUIRE(commands[0].m_OutputInfo().m_TensorShape() == TensorShape{ 1, 16, 16, 16 });
}
