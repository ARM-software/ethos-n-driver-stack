//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>
#include <ethosn_command_stream/CommandStreamBuffer.hpp>

using namespace ethosn::support_library;

/// Tests that a network containing a strided conv on input layer
TEST_CASE("StridedConvInputLayer")
{
    // Create the network
    CompilationOptions options;
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultCapabilities());
    std::shared_ptr<Operand> input   = AddInput(network, TensorInfo({ 1, 32, 32, 3 })).tensor;

    std::shared_ptr<Constant> bias =
        AddConstant(network, TensorInfo({ 1, 1, 1, 16 }, DataType::INT32_QUANTIZED), std::vector<uint8_t>(16, 0).data())
            .tensor;

    std::shared_ptr<Constant> weights =
        AddConstant(network, TensorInfo({ 3, 3, 3, 16 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO),
                    std::vector<uint8_t>(3 * 3 * 16 * 16, 0).data())
            .tensor;

    // Add conv laye.tensorr
    std::shared_ptr<Operand> conv =
        AddConvolution(network, *input, *bias, *weights,
                       ConvolutionInfo(Padding(0, 0, 0, 0), Stride(2, 2), QuantizationInfo(0, 1.1f)))
            .tensor;

    std::shared_ptr<Output> output1 = AddOutput(network, *conv).tensor;

    // Compile it
    std::vector<std::unique_ptr<CompiledNetwork>> compiledNetwork =
        ethosn::support_library::Compile(*network, CompilationOptions());

    // Extract all the conv commands
    using namespace ethosn::command_stream;
    CommandStream cmdStream = GetCommandStream(compiledNetwork[0].get());
    std::vector<McePle> convCmds;
    std::vector<PleOnly> pleCmds;
    for (const auto& cmdHeader : cmdStream)
    {
        if (cmdHeader.m_Opcode() == Opcode::OPERATION_MCE_PLE)
        {
            convCmds.push_back(cmdHeader.GetCommand<Opcode::OPERATION_MCE_PLE>()->m_Data());
        }
        if (cmdHeader.m_Opcode() == Opcode::OPERATION_PLE_ONLY)
        {
            pleCmds.push_back(cmdHeader.GetCommand<Opcode::OPERATION_PLE_ONLY>()->m_Data());
        }
    }

    // Check that we have two commands
    REQUIRE(convCmds.size() == 2);
    REQUIRE(pleCmds.size() == 0);
    REQUIRE(convCmds[0].m_MceData().m_Stride().m_X() == 1);
    REQUIRE(convCmds[0].m_MceData().m_Stride().m_Y() == 1);
    REQUIRE(convCmds[1].m_MceData().m_Stride().m_X() == 2);
    REQUIRE(convCmds[1].m_MceData().m_Stride().m_Y() == 2);

    REQUIRE(convCmds[0].m_InputInfo().m_TensorShape()[2] == 32);
    REQUIRE(convCmds[0].m_InputInfo().m_TensorShape()[1] == 32);
    REQUIRE(convCmds[0].m_InputInfo().m_TensorShape()[3] == 3);
    REQUIRE(convCmds[0].m_OutputInfo().m_TensorShape()[2] == 16);
    REQUIRE(convCmds[0].m_OutputInfo().m_SupertensorShape()[2] == 16);
    REQUIRE(convCmds[0].m_OutputInfo().m_TensorShape()[1] == 16);
    REQUIRE(convCmds[0].m_OutputInfo().m_SupertensorShape()[1] == 16);
    REQUIRE(convCmds[0].m_OutputInfo().m_TensorShape()[3] == 51);
    REQUIRE(convCmds[0].m_OutputInfo().m_SupertensorShape()[3] == 51);

    // Input to conv2 should be in interleaved shape
    REQUIRE(convCmds[1].m_InputInfo().m_TensorShape()[2] == 16);
    REQUIRE(convCmds[1].m_InputInfo().m_TensorShape()[1] == 16);
    REQUIRE(convCmds[1].m_InputInfo().m_TensorShape()[3] == 51);
    REQUIRE(convCmds[1].m_OutputInfo().m_TensorShape()[2] == 15);
    REQUIRE(convCmds[1].m_OutputInfo().m_SupertensorShape()[2] == 15);
    REQUIRE(convCmds[1].m_OutputInfo().m_TensorShape()[1] == 15);
    REQUIRE(convCmds[1].m_OutputInfo().m_SupertensorShape()[1] == 15);
    REQUIRE(convCmds[1].m_OutputInfo().m_TensorShape()[3] == 16);
    REQUIRE(convCmds[1].m_OutputInfo().m_SupertensorShape()[3] == 16);
}

/// Tests that a network containing a strided conv on non-input layer
TEST_CASE("StridedConvNonInputLayer")
{
    // Create the network
    CompilationOptions options;
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultCapabilities());
    std::shared_ptr<Operand> input   = AddInput(network, TensorInfo({ 1, 32, 32, 16 })).tensor;

    std::shared_ptr<Constant> bias1 =
        AddConstant(network, TensorInfo({ 1, 1, 1, 16 }, DataType::INT32_QUANTIZED), std::vector<uint8_t>(16, 0).data())
            .tensor;

    std::shared_ptr<Constant> bias2 =
        AddConstant(network,
                    TensorInfo({ 1, 1, 1, 16 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.1f)),
                    std::vector<uint8_t>(16, 1).data())
            .tensor;

    std::shared_ptr<Constant> weights1 =
        AddConstant(network, TensorInfo({ 1, 1, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO),
                    std::vector<uint8_t>(1 * 16 * 16, 0).data())
            .tensor;
    std::shared_ptr<Constant> weights2 =
        AddConstant(network, TensorInfo({ 3, 3, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO),
                    std::vector<uint8_t>(3 * 3 * 16 * 16, 0).data())
            .tensor;

    // Add conv1 laye.tensorr
    std::shared_ptr<Operand> conv1 =
        AddConvolution(network, *input, *bias1, *weights1,
                       ConvolutionInfo(Padding(0, 0, 0, 0), Stride(1, 1), QuantizationInfo(0, 1.1f)))
            .tensor;

    // Add conv2 laye.tensorr
    std::shared_ptr<Operand> conv2 =
        AddConvolution(network, *conv1, *bias2, *weights2,
                       ConvolutionInfo(Padding(0, 0, 0, 0), Stride(2, 2), QuantizationInfo(0, 1.2f)))
            .tensor;

    std::shared_ptr<Output> output1 = AddOutput(network, *conv2).tensor;

    // Compile it
    std::vector<std::unique_ptr<CompiledNetwork>> compiledNetwork =
        ethosn::support_library::Compile(*network, CompilationOptions());

    // Extract all the conv commands
    using namespace ethosn::command_stream;
    CommandStream cmdStream = GetCommandStream(compiledNetwork[0].get());
    std::vector<McePle> convCmds;
    std::vector<PleOnly> pleCmds;
    for (const auto& cmdHeader : cmdStream)
    {
        if (cmdHeader.m_Opcode() == Opcode::OPERATION_MCE_PLE)
        {
            convCmds.push_back(cmdHeader.GetCommand<Opcode::OPERATION_MCE_PLE>()->m_Data());
        }
        if (cmdHeader.m_Opcode() == Opcode::OPERATION_PLE_ONLY)
        {
            pleCmds.push_back(cmdHeader.GetCommand<Opcode::OPERATION_PLE_ONLY>()->m_Data());
        }
    }

    // Check that we have two MCE commands
    REQUIRE(convCmds.size() == 2);
    REQUIRE(pleCmds.size() == 0);
    REQUIRE(convCmds[0].m_MceData().m_Stride().m_X() == 1);
    REQUIRE(convCmds[0].m_MceData().m_Stride().m_Y() == 1);
    REQUIRE(convCmds[1].m_MceData().m_Stride().m_X() == 2);
    REQUIRE(convCmds[1].m_MceData().m_Stride().m_Y() == 2);

    REQUIRE(convCmds[0].m_InputInfo().m_TensorShape()[2] == 32);
    REQUIRE(convCmds[0].m_InputInfo().m_TensorShape()[1] == 32);
    REQUIRE(convCmds[0].m_InputInfo().m_TensorShape()[3] == 16);
    REQUIRE(convCmds[0].m_OutputInfo().m_TensorShape()[2] == 16);
    REQUIRE(convCmds[0].m_OutputInfo().m_TensorShape()[1] == 16);
    REQUIRE(convCmds[0].m_OutputInfo().m_TensorShape()[3] == 64);

    // Input to conv2 should be in interleaved shape
    REQUIRE(convCmds[1].m_InputInfo().m_TensorShape()[2] == 16);
    REQUIRE(convCmds[1].m_InputInfo().m_TensorShape()[1] == 16);
    REQUIRE(convCmds[1].m_InputInfo().m_TensorShape()[3] == 64);
    REQUIRE(convCmds[1].m_OutputInfo().m_TensorShape()[2] == 15);
    REQUIRE(convCmds[1].m_OutputInfo().m_SupertensorShape()[2] == 15);
    REQUIRE(convCmds[1].m_OutputInfo().m_TensorShape()[1] == 15);
    REQUIRE(convCmds[1].m_OutputInfo().m_SupertensorShape()[1] == 15);
    REQUIRE(convCmds[1].m_OutputInfo().m_TensorShape()[3] == 16);
    REQUIRE(convCmds[1].m_OutputInfo().m_SupertensorShape()[3] == 16);
}
