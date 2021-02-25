//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>
#include <ethosn_command_stream/CommandStreamBuffer.hpp>

using namespace ethosn::support_library;

/// Tests that a simple branching has the inputs and output buffers correctly linked in the command stream.
TEST_CASE("SimpleBranch")
{
    // Create the network
    CompilationOptions options       = GetDefaultCompilationOptions();
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultCapabilities());
    std::shared_ptr<Operand> input   = AddInput(network, TensorInfo({ 1, 16, 16, 16 })).tensor;

    std::shared_ptr<Constant> bias =
        AddConstant(network, TensorInfo({ 1, 1, 1, 16 }, DataType::INT32_QUANTIZED), std::vector<uint8_t>(16, 0).data())
            .tensor;
    std::shared_ptr<Constant> weights =
        AddConstant(network, TensorInfo({ 1, 1, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO),
                    std::vector<uint8_t>(16 * 16 * 16, 0).data())
            .tensor;
    std::shared_ptr<Operand> conv1 =
        AddConvolution(network, *input, *bias, *weights,
                       ConvolutionInfo(Padding(0, 0, 0, 0), Stride(1, 1), QuantizationInfo(0, 1.1f)))
            .tensor;

    std::shared_ptr<Operand> conv2 =
        AddConvolution(network, *input, *bias, *weights,
                       ConvolutionInfo(Padding(0, 0, 0, 0), Stride(1, 1), QuantizationInfo(0, 1.1f)))
            .tensor;

    std::shared_ptr<Output> output1 = AddOutput(network, *conv1).tensor;
    std::shared_ptr<Output> output2 = AddOutput(network, *conv2).tensor;

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

    // Check that the conv commands are as expected. There should be two that share an input but have different outputs.
    REQUIRE(convCmds.size() == 2);
    REQUIRE(convCmds[0].m_InputInfo().m_DramBufferId() == convCmds[1].m_InputInfo().m_DramBufferId());
    REQUIRE(convCmds[0].m_OutputInfo().m_DramBufferId() != convCmds[1].m_OutputInfo().m_DramBufferId());
}

/// Tests that a network containing a conv followed by a branch with two relus works as expected.
TEST_CASE("ReluAfterBranch")
{
    // Create the network
    CompilationOptions options       = GetDefaultCompilationOptions();
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultCapabilities());
    std::shared_ptr<Operand> input   = AddInput(network, TensorInfo({ 1, 16, 16, 16 })).tensor;

    std::shared_ptr<Constant> bias =
        AddConstant(network, TensorInfo({ 1, 1, 1, 16 }, DataType::INT32_QUANTIZED), std::vector<uint8_t>(16, 0).data())
            .tensor;
    std::shared_ptr<Constant> weights =
        AddConstant(network, TensorInfo({ 1, 1, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO),
                    std::vector<uint8_t>(16 * 16 * 16, 0).data())
            .tensor;
    std::shared_ptr<Operand> conv =
        AddConvolution(network, *input, *bias, *weights,
                       ConvolutionInfo(Padding(0, 0, 0, 0), Stride(1, 1), QuantizationInfo(0, 1.1f)))
            .tensor;

    std::shared_ptr<Operand> relu1 = AddRelu(network, *conv, ReluInfo(10, 255)).tensor;
    std::shared_ptr<Operand> relu2 = AddRelu(network, *conv, ReluInfo(20, 255)).tensor;

    std::shared_ptr<Output> output1 = AddOutput(network, *relu1).tensor;
    std::shared_ptr<Output> output2 = AddOutput(network, *relu2).tensor;

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

    // Check that the conv commands are as expected. There should be 3, the main one and two with a relu afterwards.
    REQUIRE(convCmds.size() == 3);
    REQUIRE(convCmds[1].m_MceData().m_ActivationMin() == 20);
    REQUIRE(convCmds[1].m_InputInfo().m_DramBufferId() == convCmds[0].m_OutputInfo().m_DramBufferId());
    REQUIRE(convCmds[2].m_MceData().m_ActivationMin() == 10);
    REQUIRE(convCmds[2].m_InputInfo().m_DramBufferId() == convCmds[0].m_OutputInfo().m_DramBufferId());
}

/// Tests that the output of a branch can stay in Sram
TEST_CASE("Branch in Sram")
{
    // Create the network
    CompilationOptions options       = GetDefaultCompilationOptions();
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultCapabilities());
    std::shared_ptr<Operand> input =
        AddInput(network, TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWCB)).tensor;

    std::shared_ptr<Constant> bias1 =
        AddConstant(network, TensorInfo({ 1, 1, 1, 16 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, 1.0f }),
                    std::vector<uint8_t>(16, 0).data())
            .tensor;
    std::shared_ptr<Constant> bias2 =
        AddConstant(network, TensorInfo({ 1, 1, 1, 16 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, 1.1f }),
                    std::vector<uint8_t>(16, 0).data())
            .tensor;
    std::shared_ptr<Constant> weights =
        AddConstant(network, TensorInfo({ 1, 1, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, { 0, 1.f }),
                    std::vector<uint8_t>(16 * 16 * 16, 0).data())
            .tensor;
    std::shared_ptr<Operand> conv1 =
        AddConvolution(network, *input, *bias1, *weights,
                       ConvolutionInfo(Padding(0, 0, 0, 0), Stride(1, 1), QuantizationInfo(0, 1.1f)))
            .tensor;
    std::shared_ptr<Operand> conv2 =
        AddConvolution(network, *conv1, *bias2, *weights,
                       ConvolutionInfo(Padding(0, 0, 0, 0), Stride(1, 1), QuantizationInfo(0, 1.2f)))
            .tensor;
    std::shared_ptr<Operand> conv3 =
        AddConvolution(network, *conv1, *bias2, *weights,
                       ConvolutionInfo(Padding(0, 0, 0, 0), Stride(1, 1), QuantizationInfo(0, 1.2f)))
            .tensor;
    std::vector<Operand*> concatInputs = std::vector<Operand*>{ conv2.get(), conv3.get() };
    std::shared_ptr<Operand> concat =
        AddConcatenation(network, concatInputs, ConcatenationInfo(3, QuantizationInfo(0, 1.2f))).tensor;
    std::shared_ptr<Output> output1 = AddOutput(network, *concat).tensor;

    // Compile it
    std::vector<std::unique_ptr<CompiledNetwork>> compiledNetwork = ethosn::support_library::Compile(*network, options);

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
    REQUIRE(convCmds.size() == 3);
    REQUIRE(convCmds[0].m_InputInfo().m_DataLocation() == ethosn::command_stream::DataLocation::DRAM);
    REQUIRE(convCmds[0].m_OutputInfo().m_DataLocation() == ethosn::command_stream::DataLocation::SRAM);
    REQUIRE(convCmds[1].m_InputInfo().m_DataLocation() == ethosn::command_stream::DataLocation::SRAM);
    REQUIRE(convCmds[1].m_OutputInfo().m_DataLocation() == ethosn::command_stream::DataLocation::DRAM);
    REQUIRE(convCmds[2].m_InputInfo().m_DataLocation() == ethosn::command_stream::DataLocation::SRAM);
    REQUIRE(convCmds[2].m_OutputInfo().m_DataLocation() == ethosn::command_stream::DataLocation::DRAM);
}
