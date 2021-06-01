//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>
#include <ethosn_command_stream/CommandStreamBuffer.hpp>

using namespace ethosn::support_library;

/// Tests that a layer has NHWC for input and NHWCB for output
TEST_CASE("Test NHWC Input and NHWCB Output")
{
    // Create the network
    CompilationOptions options;
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultCapabilities());
    std::shared_ptr<Operand> inputConv1 =
        AddInput(network, TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC)).tensor;

    std::shared_ptr<Constant> biasConv1 =
        AddConstant(network, TensorInfo({ 1, 1, 1, 16 }, DataType::INT32_QUANTIZED), std::vector<uint8_t>(16, 0).data())
            .tensor;
    std::shared_ptr<Constant> weightsConv1 =
        AddConstant(network, TensorInfo({ 1, 1, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO),
                    std::vector<uint8_t>(16 * 16 * 16, 0).data())
            .tensor;
    std::shared_ptr<Operand> conv1 =
        AddConvolution(network, *inputConv1, *biasConv1, *weightsConv1,
                       ConvolutionInfo(Padding(0, 0, 0, 0), Stride(1, 1), QuantizationInfo(0, 1.1f)))
            .tensor;
    std::shared_ptr<Output> output = AddOutput(network, *conv1, DataFormat::NHWCB).tensor;

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

    // Check that we have NHWCB output
    REQUIRE(convCmds.size() == 1);
    REQUIRE(convCmds[0].m_InputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::NHWC);
    REQUIRE(convCmds[0].m_OutputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::NHWCB);
}

/// Tests a command stream comprising 2 convolutions which should produce compressed intermediate DRAM data.
TEST_CASE("NhwcbCompressed")
{
    // Create the network
    CompilationOptions options;
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultCapabilities());
    auto input                       = AddInput(network, TensorInfo({ 1, 1024, 32, 16 }));

    auto bias1 = AddConstant(network, TensorInfo({ 1, 1, 1, 16 }, DataType::INT32_QUANTIZED),
                             std::vector<uint8_t>(16, 0).data());

    auto bias2 = AddConstant(
        network, TensorInfo({ 1, 1, 1, 16 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.1f)),
        std::vector<uint8_t>(16, 1).data());

    auto weights1 = AddConstant(network, TensorInfo({ 3, 3, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO),
                                std::vector<uint8_t>(3 * 3 * 16 * 16, 0).data());
    auto weights2 = AddConstant(network, TensorInfo({ 3, 3, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO),
                                std::vector<uint8_t>(3 * 3 * 16 * 16, 0).data());

    // Add conv1 layer
    auto conv1 = AddConvolution(network, *input.tensor, *bias1.tensor, *weights1.tensor,
                                ConvolutionInfo(Padding(0, 0, 0, 0), Stride(1, 1), QuantizationInfo(0, 1.1f)));

    // Add conv2 layer
    auto conv2 = AddConvolution(network, *conv1.tensor, *bias2.tensor, *weights2.tensor,
                                ConvolutionInfo(Padding(0, 0, 0, 0), Stride(1, 1), QuantizationInfo(0, 1.2f)));

    auto output1 = AddOutput(network, *conv2.tensor);

    // Compile it
    options.m_EnableIntermediateCompression                       = true;
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

    // Check that the conv commands are as expected. Inputs and outputs to the network at NHWC
    // the intermediate layers are NHWCB_COMPRESSED.
    REQUIRE(convCmds.size() == 2);
    REQUIRE(convCmds[0].m_InputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::NHWC);
    REQUIRE(convCmds[0].m_OutputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::NHWCB_COMPRESSED);
    REQUIRE(convCmds[1].m_InputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::NHWCB_COMPRESSED);
    REQUIRE(convCmds[1].m_OutputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::NHWC);
}

TEST_CASE("FcafDeepCompressed")
{
    // Create the network
    CompilationOptions options;
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultEthosN78Capabilities());
    auto input                       = AddInput(network, TensorInfo({ 1, 1024, 32, 32 }));

    auto bias1 = AddConstant(network, TensorInfo({ 1, 1, 1, 32 }, DataType::INT32_QUANTIZED),
                             std::vector<uint8_t>(32, 0).data());

    auto bias2 = AddConstant(
        network, TensorInfo({ 1, 1, 1, 32 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.1f)),
        std::vector<uint8_t>(32, 1).data());

    auto weights1 = AddConstant(network, TensorInfo({ 1, 1, 32, 32 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO),
                                std::vector<uint8_t>(1 * 1 * 32 * 32, 0).data());
    auto weights2 = AddConstant(network, TensorInfo({ 1, 1, 32, 32 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO),
                                std::vector<uint8_t>(1 * 1 * 32 * 32, 0).data());

    // Add conv1 layer
    auto conv1 = AddConvolution(network, *input.tensor, *bias1.tensor, *weights1.tensor,
                                ConvolutionInfo(Padding(0, 0, 0, 0), Stride(1, 1), QuantizationInfo(0, 1.1f)));

    // Add conv2 layer
    auto conv2 = AddConvolution(network, *conv1.tensor, *bias2.tensor, *weights2.tensor,
                                ConvolutionInfo(Padding(0, 0, 0, 0), Stride(1, 1), QuantizationInfo(0, 1.2f)));

    auto output1 = AddOutput(network, *conv2.tensor);

    // Compile it
    options.m_EnableIntermediateCompression                       = true;
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

    // Check that the conv commands are as expected. Inputs and outputs to the network at NHWC
    // the intermediate layers are FCAF_DEEP.
    REQUIRE(convCmds.size() == 2);
    REQUIRE(convCmds[0].m_InputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::NHWC);
    REQUIRE(convCmds[0].m_OutputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::FCAF_DEEP);
    REQUIRE(convCmds[1].m_InputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::FCAF_DEEP);
    REQUIRE(convCmds[1].m_OutputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::NHWC);
}

TEST_CASE("FcafWideCompressed")
{
    // Create the network
    CompilationOptions options;
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultEthosN78Capabilities());
    auto input                       = AddInput(network, TensorInfo({ 1, 1024, 32, 16 }));

    auto bias1 = AddConstant(network, TensorInfo({ 1, 1, 1, 16 }, DataType::INT32_QUANTIZED),
                             std::vector<uint8_t>(16, 0).data());

    auto bias2 = AddConstant(
        network, TensorInfo({ 1, 1, 1, 16 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.1f)),
        std::vector<uint8_t>(16, 1).data());

    auto weights1 = AddConstant(network, TensorInfo({ 1, 1, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO),
                                std::vector<uint8_t>(1 * 1 * 16 * 16, 0).data());
    auto weights2 = AddConstant(network, TensorInfo({ 1, 1, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO),
                                std::vector<uint8_t>(1 * 1 * 16 * 16, 0).data());

    // Add conv1 layer
    auto conv1 = AddConvolution(network, *input.tensor, *bias1.tensor, *weights1.tensor,
                                ConvolutionInfo(Padding(0, 0, 0, 0), Stride(1, 1), QuantizationInfo(0, 1.1f)));

    // Add conv2 layer
    auto conv2 = AddConvolution(network, *conv1.tensor, *bias2.tensor, *weights2.tensor,
                                ConvolutionInfo(Padding(0, 0, 0, 0), Stride(1, 1), QuantizationInfo(0, 1.2f)));

    auto output1 = AddOutput(network, *conv2.tensor);

    // Compile it
    options.m_EnableIntermediateCompression                       = true;
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

    // Check that the conv commands are as expected. Inputs and outputs to the network at NHWC
    // the intermediate layers are FCAF_WIDE.
    REQUIRE(convCmds.size() == 2);
    REQUIRE(convCmds[0].m_InputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::NHWC);
    REQUIRE(convCmds[0].m_OutputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::FCAF_WIDE);
    REQUIRE(convCmds[1].m_InputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::FCAF_WIDE);
    REQUIRE(convCmds[1].m_OutputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::NHWC);
}

TEST_CASE("FcafDeepPartialCompressed")
{
    // Create the network
    CompilationOptions options;
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultEthosN78Capabilities());
    auto input                       = AddInput(network, TensorInfo({ 1, 1035, 28, 32 }));

    auto bias1 = AddConstant(network, TensorInfo({ 1, 1, 1, 32 }, DataType::INT32_QUANTIZED),
                             std::vector<uint8_t>(32, 0).data());

    auto bias2 = AddConstant(
        network, TensorInfo({ 1, 1, 1, 32 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.1f)),
        std::vector<uint8_t>(32, 1).data());

    auto weights1 = AddConstant(network, TensorInfo({ 1, 1, 32, 32 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO),
                                std::vector<uint8_t>(1 * 1 * 32 * 32, 0).data());
    auto weights2 = AddConstant(network, TensorInfo({ 1, 1, 32, 32 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO),
                                std::vector<uint8_t>(1 * 1 * 32 * 32, 0).data());

    // Add conv1 layer
    auto conv1 = AddConvolution(network, *input.tensor, *bias1.tensor, *weights1.tensor,
                                ConvolutionInfo(Padding(0, 0, 0, 0), Stride(1, 1), QuantizationInfo(0, 1.1f)));

    // Add conv2 layer
    auto conv2 = AddConvolution(network, *conv1.tensor, *bias2.tensor, *weights2.tensor,
                                ConvolutionInfo(Padding(0, 0, 0, 0), Stride(1, 1), QuantizationInfo(0, 1.2f)));

    auto output1 = AddOutput(network, *conv2.tensor);

    // Compile it
    options.m_EnableIntermediateCompression                       = true;
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

    // Check that the conv commands are as expected. Inputs and outputs to the network at NHWC
    // the intermediate layers are FCAF_DEEP.
    REQUIRE(convCmds.size() == 2);
    REQUIRE(convCmds[0].m_InputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::NHWC);
    REQUIRE(convCmds[0].m_OutputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::FCAF_DEEP);
    REQUIRE(convCmds[1].m_InputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::FCAF_DEEP);
    REQUIRE(convCmds[1].m_OutputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::NHWC);
}

TEST_CASE("FcafWidePartialCompressed")
{
    // Create the network
    CompilationOptions options;
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultEthosN78Capabilities());
    auto input                       = AddInput(network, TensorInfo({ 1, 1035, 28, 16 }));

    auto bias1 = AddConstant(network, TensorInfo({ 1, 1, 1, 16 }, DataType::INT32_QUANTIZED),
                             std::vector<uint8_t>(16, 0).data());

    auto bias2 = AddConstant(
        network, TensorInfo({ 1, 1, 1, 16 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.1f)),
        std::vector<uint8_t>(16, 1).data());

    auto weights1 = AddConstant(network, TensorInfo({ 1, 1, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO),
                                std::vector<uint8_t>(1 * 1 * 16 * 16, 0).data());
    auto weights2 = AddConstant(network, TensorInfo({ 1, 1, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO),
                                std::vector<uint8_t>(1 * 1 * 16 * 16, 0).data());

    // Add conv1 layer
    auto conv1 = AddConvolution(network, *input.tensor, *bias1.tensor, *weights1.tensor,
                                ConvolutionInfo(Padding(0, 0, 0, 0), Stride(1, 1), QuantizationInfo(0, 1.1f)));

    // Add conv2 layer
    auto conv2 = AddConvolution(network, *conv1.tensor, *bias2.tensor, *weights2.tensor,
                                ConvolutionInfo(Padding(0, 0, 0, 0), Stride(1, 1), QuantizationInfo(0, 1.2f)));

    auto output1 = AddOutput(network, *conv2.tensor);

    // Compile it
    options.m_EnableIntermediateCompression                       = true;
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

    // Check that the conv commands are as expected. Inputs and outputs to the network at NHWC
    // the intermediate layers are FCAF_WIDE.
    REQUIRE(convCmds.size() == 2);
    REQUIRE(convCmds[0].m_InputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::NHWC);
    REQUIRE(convCmds[0].m_OutputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::FCAF_WIDE);
    REQUIRE(convCmds[1].m_InputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::FCAF_WIDE);
    REQUIRE(convCmds[1].m_OutputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::NHWC);
}
