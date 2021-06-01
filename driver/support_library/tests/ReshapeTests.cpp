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
    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Not Supported configuration
    TensorInfo input({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
    TensorInfo output({ 1, 16, 1, 32 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
    REQUIRE(queries.IsReshapeSupported({ 1, 16, 1, 32 }, input, &output) == SupportedLevel::Unsupported);
}

/// Tests Single Reshape Command using SRAM to SRAM reshape
TEST_CASE("Test Single Reshape layer SRAM")
{
    const auto inputDataType    = GENERATE(DataType::INT8_QUANTIZED, DataType::UINT8_QUANTIZED);
    const auto expectedDataType = utils::GetCommandDataType(inputDataType);

    // Create the network
    CompilationOptions options;
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultCapabilities());
    std::shared_ptr<Operand> inputConv1 =
        AddInput(network, TensorInfo({ 1, 16, 16, 16 }, inputDataType, DataFormat::NHWCB)).tensor;

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

    std::shared_ptr<Operand> reshape = AddReshape(network, *conv1, { 1, 32, 8, 16 }).tensor;
    std::shared_ptr<Constant> biasConv2 =
        AddConstant(network,
                    TensorInfo({ 1, 1, 1, 16 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.1f)),
                    std::vector<uint8_t>(16, 0).data())
            .tensor;
    std::shared_ptr<Constant> weightsConv2 =
        AddConstant(network, TensorInfo({ 1, 1, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO),
                    std::vector<uint8_t>(16 * 16 * 16, 0).data())
            .tensor;
    std::shared_ptr<Operand> conv2 =
        AddConvolution(network, *reshape, *biasConv2, *weightsConv2,
                       ConvolutionInfo(Padding(0, 0, 0, 0), Stride(1, 1), QuantizationInfo(0, 1.2f)))
            .tensor;
    std::shared_ptr<Output> output1 = AddOutput(network, *conv2).tensor;

    // Compile it
    std::vector<std::unique_ptr<CompiledNetwork>> compiledNetwork =
        ethosn::support_library::Compile(*network, CompilationOptions());

    // Extract all the conv commands
    using namespace ethosn::command_stream;
    CommandStream cmdStream = GetCommandStream(compiledNetwork[0].get());
    std::vector<McePle> convCmds;
    std::vector<ethosn::command_stream::Convert> reshapeCmds;
    for (const auto& cmdHeader : cmdStream)
    {
        if (cmdHeader.m_Opcode() == Opcode::OPERATION_MCE_PLE)
        {
            convCmds.push_back(cmdHeader.GetCommand<Opcode::OPERATION_MCE_PLE>()->m_Data());
        }
        if (cmdHeader.m_Opcode() == Opcode::OPERATION_CONVERT)
        {
            reshapeCmds.push_back(cmdHeader.GetCommand<Opcode::OPERATION_CONVERT>()->m_Data());
        }
    }

    TensorShape conv1_output = { 1, 16, 16, 16 };
    TensorShape conv2_input  = { 1, 32, 8, 16 };
    TensorShape conv2_output = { 1, 32, 8, 16 };

    // Check that we have NHWC with actual
    REQUIRE(convCmds.size() == 2);
    REQUIRE(convCmds[0].m_OutputInfo().m_DataType() == expectedDataType);
    REQUIRE(convCmds[0].m_OutputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::NHWCB);
    REQUIRE(convCmds[0].m_OutputInfo().m_TensorShape() == conv1_output);
    REQUIRE(convCmds[0].m_OutputInfo().m_DataLocation() == ethosn::command_stream::DataLocation::SRAM);
    REQUIRE(convCmds[1].m_InputInfo().m_DataType() == expectedDataType);
    REQUIRE(convCmds[1].m_InputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::NHWCB);
    REQUIRE(convCmds[1].m_InputInfo().m_DataLocation() == ethosn::command_stream::DataLocation::SRAM);
    REQUIRE(convCmds[1].m_InputInfo().m_TensorShape() == conv2_input);
    REQUIRE(convCmds[1].m_OutputInfo().m_TensorShape() == conv2_output);

    // This also has a special reshape command
    REQUIRE(reshapeCmds.size() == 1);
    REQUIRE(reshapeCmds[0].m_InputInfo().m_DataType() == expectedDataType);
    REQUIRE(reshapeCmds[0].m_InputInfo().m_TensorShape() == conv1_output);
    REQUIRE(reshapeCmds[0].m_InputInfo().m_SramOffset() == convCmds[0].m_OutputInfo().m_SramOffset());
    REQUIRE(reshapeCmds[0].m_OutputInfo().m_DataType() == expectedDataType);
    REQUIRE(reshapeCmds[0].m_OutputInfo().m_TensorShape() == conv2_output);
    REQUIRE(reshapeCmds[0].m_OutputInfo().m_SramOffset() == 0x0);
}

/// Tests Multiple Reshape Commands following each other using SRAM to SRAM reshape
TEST_CASE("Test Multiple Reshape layers SRAM")
{
    // Create the network
    CompilationOptions options;
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultCapabilities());
    std::shared_ptr<Operand> inputConv1 =
        AddInput(network, TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWCB)).tensor;

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

    std::shared_ptr<Operand> reshape1 = AddReshape(network, *conv1, { 1, 16, 32, 8 }).tensor;
    std::shared_ptr<Operand> reshape2 = AddReshape(network, *reshape1, { 1, 32, 32, 4 }).tensor;
    std::shared_ptr<Operand> reshape3 = AddReshape(network, *reshape2, { 1, 32, 4, 32 }).tensor;
    std::shared_ptr<Operand> reshape4 = AddReshape(network, *reshape3, { 1, 32, 8, 16 }).tensor;
    std::shared_ptr<Constant> biasConv2 =
        AddConstant(network,
                    TensorInfo({ 1, 1, 1, 16 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.1f)),
                    std::vector<uint8_t>(16, 0).data())
            .tensor;
    std::shared_ptr<Constant> weightsConv2 =
        AddConstant(network, TensorInfo({ 1, 1, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO),
                    std::vector<uint8_t>(16 * 16 * 16, 0).data())
            .tensor;
    std::shared_ptr<Operand> conv2 =
        AddConvolution(network, *reshape4, *biasConv2, *weightsConv2,
                       ConvolutionInfo(Padding(0, 0, 0, 0), Stride(1, 1), QuantizationInfo(0, 1.2f)))
            .tensor;
    std::shared_ptr<Output> output1 = AddOutput(network, *conv2).tensor;

    // Compile it
    std::vector<std::unique_ptr<CompiledNetwork>> compiledNetwork =
        ethosn::support_library::Compile(*network, CompilationOptions());

    // Extract all the conv commands
    using namespace ethosn::command_stream;
    CommandStream cmdStream = GetCommandStream(compiledNetwork[0].get());
    std::vector<McePle> convCmds;
    std::vector<ethosn::command_stream::Convert> reshapeCmds;
    for (const auto& cmdHeader : cmdStream)
    {
        if (cmdHeader.m_Opcode() == Opcode::OPERATION_MCE_PLE)
        {
            convCmds.push_back(cmdHeader.GetCommand<Opcode::OPERATION_MCE_PLE>()->m_Data());
        }
        if (cmdHeader.m_Opcode() == Opcode::OPERATION_CONVERT)
        {
            reshapeCmds.push_back(cmdHeader.GetCommand<Opcode::OPERATION_CONVERT>()->m_Data());
        }
    }

    TensorShape conv1_output = { 1, 16, 16, 16 };
    TensorShape conv2_input  = { 1, 32, 8, 16 };
    TensorShape conv2_output = { 1, 32, 8, 16 };

    // Check that we have NHWC with actual
    REQUIRE(convCmds.size() == 2);
    REQUIRE(convCmds[0].m_OutputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::NHWCB);
    REQUIRE(convCmds[0].m_OutputInfo().m_DataLocation() == ethosn::command_stream::DataLocation::SRAM);
    REQUIRE(convCmds[0].m_OutputInfo().m_TensorShape() == conv1_output);
    REQUIRE(convCmds[1].m_InputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::NHWCB);
    REQUIRE(convCmds[1].m_InputInfo().m_DataLocation() == ethosn::command_stream::DataLocation::SRAM);
    REQUIRE(convCmds[1].m_InputInfo().m_TensorShape() == conv2_input);
    REQUIRE(convCmds[1].m_OutputInfo().m_TensorShape() == conv2_output);

    // This also has a special reshape command
    REQUIRE(reshapeCmds.size() == 1);
    REQUIRE(reshapeCmds[0].m_InputInfo().m_TensorShape() == conv1_output);
    REQUIRE(reshapeCmds[0].m_InputInfo().m_SramOffset() == convCmds[0].m_OutputInfo().m_SramOffset());
    REQUIRE(reshapeCmds[0].m_OutputInfo().m_TensorShape() == conv2_output);
    REQUIRE(reshapeCmds[0].m_OutputInfo().m_SramOffset() == 0x0);
}

/// Tests Single Reshape Command using SRAM to DRAM reshape
TEST_CASE("Test Single Reshape layer DRAM")
{
    // Create the network
    CompilationOptions options;
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultCapabilities());
    std::shared_ptr<Operand> inputConv1 =
        AddInput(network, TensorInfo({ 1, 256, 128, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWCB)).tensor;

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

    std::shared_ptr<Operand> reshape = AddReshape(network, *conv1, { 1, 128, 256, 16 }).tensor;
    std::shared_ptr<Constant> biasConv2 =
        AddConstant(network,
                    TensorInfo({ 1, 1, 1, 16 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.1f)),
                    std::vector<uint8_t>(16, 0).data())
            .tensor;
    std::shared_ptr<Constant> weightsConv2 =
        AddConstant(network, TensorInfo({ 1, 1, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO),
                    std::vector<uint8_t>(16 * 16 * 16, 0).data())
            .tensor;
    std::shared_ptr<Operand> conv2 =
        AddConvolution(network, *reshape, *biasConv2, *weightsConv2,
                       ConvolutionInfo(Padding(0, 0, 0, 0), Stride(1, 1), QuantizationInfo(0, 1.2f)))
            .tensor;
    std::shared_ptr<Output> output1 = AddOutput(network, *conv2).tensor;

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

    TensorShape conv1_output = { 1, 256, 128, 16 };
    TensorShape conv2_input  = { 1, 128, 256, 16 };
    TensorShape conv2_output = { 1, 128, 256, 16 };

    // Check that we have NHWC with actual
    REQUIRE(convCmds.size() == 2);
    REQUIRE(convCmds[0].m_OutputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::NHWC);
    REQUIRE(convCmds[0].m_OutputInfo().m_DataLocation() == ethosn::command_stream::DataLocation::DRAM);
    REQUIRE(convCmds[0].m_OutputInfo().m_TensorShape() == conv1_output);
    REQUIRE(convCmds[0].m_OutputInfo().m_SupertensorShape() == conv1_output);
    REQUIRE(convCmds[1].m_InputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::NHWC);
    REQUIRE(convCmds[1].m_InputInfo().m_DataLocation() == ethosn::command_stream::DataLocation::DRAM);
    REQUIRE(convCmds[1].m_InputInfo().m_TensorShape() == conv2_input);
    REQUIRE(convCmds[1].m_OutputInfo().m_TensorShape() == conv2_output);
    REQUIRE(convCmds[1].m_OutputInfo().m_SupertensorShape() == conv2_output);
}

/// Tests Multiple Reshape Commands following each other using SRAM to DRAM reshape
TEST_CASE("Test Multiple Reshape layers DRAM")
{
    // Create the network
    CompilationOptions options;
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultCapabilities());
    std::shared_ptr<Operand> inputConv1 =
        AddInput(network, TensorInfo({ 1, 256, 128, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWCB)).tensor;

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

    std::shared_ptr<Operand> reshape1 = AddReshape(network, *conv1, { 1, 256, 16, 128 }).tensor;
    std::shared_ptr<Operand> reshape2 = AddReshape(network, *reshape1, { 1, 4, 512, 256 }).tensor;
    std::shared_ptr<Operand> reshape3 = AddReshape(network, *reshape2, { 1, 256, 256, 8 }).tensor;
    std::shared_ptr<Operand> reshape4 = AddReshape(network, *reshape3, { 1, 128, 256, 16 }).tensor;
    std::shared_ptr<Constant> biasConv2 =
        AddConstant(network,
                    TensorInfo({ 1, 1, 1, 16 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.1f)),
                    std::vector<uint8_t>(16, 0).data())
            .tensor;
    std::shared_ptr<Constant> weightsConv2 =
        AddConstant(network, TensorInfo({ 1, 1, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO),
                    std::vector<uint8_t>(16 * 16 * 16, 0).data())
            .tensor;
    std::shared_ptr<Operand> conv2 =
        AddConvolution(network, *reshape4, *biasConv2, *weightsConv2,
                       ConvolutionInfo(Padding(0, 0, 0, 0), Stride(1, 1), QuantizationInfo(0, 1.2f)))
            .tensor;
    std::shared_ptr<Output> output1 = AddOutput(network, *conv2).tensor;

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

    TensorShape conv1_output = { 1, 256, 128, 16 };
    TensorShape conv2_input  = { 1, 128, 256, 16 };
    TensorShape conv2_output = { 1, 128, 256, 16 };

    // Check that we have NHWC with actual
    REQUIRE(convCmds.size() == 2);
    REQUIRE(convCmds[0].m_OutputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::NHWC);
    REQUIRE(convCmds[0].m_OutputInfo().m_DataLocation() == ethosn::command_stream::DataLocation::DRAM);
    REQUIRE(convCmds[0].m_OutputInfo().m_TensorShape() == conv1_output);
    REQUIRE(convCmds[0].m_OutputInfo().m_SupertensorShape() == conv1_output);
    REQUIRE(convCmds[1].m_InputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::NHWC);
    REQUIRE(convCmds[1].m_InputInfo().m_DataLocation() == ethosn::command_stream::DataLocation::DRAM);
    REQUIRE(convCmds[1].m_InputInfo().m_TensorShape() == conv2_input);
    REQUIRE(convCmds[1].m_OutputInfo().m_TensorShape() == conv2_output);
    REQUIRE(convCmds[1].m_OutputInfo().m_SupertensorShape() == conv2_output);
}

/// Tests reshape as last layer when using strategy 3
TEST_CASE("Test Reshape as last layer NHWC Strategy 3")
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

    std::shared_ptr<Operand> reshape = AddReshape(network, *conv1, { 1, 32, 8, 16 }).tensor;

    std::shared_ptr<Output> output = AddOutput(network, *reshape).tensor;

    // Compile it
    std::vector<std::unique_ptr<CompiledNetwork>> compiledNetwork =
        ethosn::support_library::Compile(*network, CompilationOptions());

    // Extract all the conv commands
    using namespace ethosn::command_stream;
    CommandStream cmdStream = GetCommandStream(compiledNetwork[0].get());
    std::vector<McePle> convCmd;
    std::vector<ethosn::command_stream::Convert> reshapeCmd;
    for (const auto& cmdHeader : cmdStream)
    {
        if (cmdHeader.m_Opcode() == Opcode::OPERATION_MCE_PLE)
        {
            convCmd.push_back(cmdHeader.GetCommand<Opcode::OPERATION_MCE_PLE>()->m_Data());
        }
        if (cmdHeader.m_Opcode() == Opcode::OPERATION_CONVERT)
        {
            reshapeCmd.push_back(cmdHeader.GetCommand<Opcode::OPERATION_CONVERT>()->m_Data());
        }
    }

    TensorShape conv1_output = { 1, 16, 16, 16 };

    // Check that we have NHWC with actual
    REQUIRE(convCmd.size() == 1);
    REQUIRE(convCmd[0].m_OutputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::NHWC);
    REQUIRE(convCmd[0].m_OutputInfo().m_TensorShape() == conv1_output);
    REQUIRE(convCmd[0].m_OutputInfo().m_SupertensorShape() == conv1_output);
    REQUIRE(convCmd[0].m_OutputInfo().m_DataLocation() == ethosn::command_stream::DataLocation::DRAM);

    // Strategy 3 triggers SRAM->SRAM reshape but since its last layer it
    // should write data directly to DRAM without any reshape
    // NHWC data in DRAM has always the same layout. Only the interpretation changes.
    // Hence no reshape commands should be present
    REQUIRE(reshapeCmd.size() == 0);
}

/// Test reshape as last layer and NHWCB
TEST_CASE("Test Reshape as last layer NHWCB")
{
    // Create the network
    CompilationOptions options;
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultCapabilities());
    std::shared_ptr<Operand> inputConv1 =
        AddInput(network, TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWCB)).tensor;

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
    std::shared_ptr<Operand> reshape = AddReshape(network, *conv1, { 1, 32, 8, 16 }).tensor;
    std::shared_ptr<Output> output   = AddOutput(network, *reshape, DataFormat::NHWCB).tensor;

    // Compile it
    std::vector<std::unique_ptr<CompiledNetwork>> compiledNetwork =
        ethosn::support_library::Compile(*network, CompilationOptions());

    // Extract all the conv commands
    using namespace ethosn::command_stream;
    CommandStream cmdStream = GetCommandStream(compiledNetwork[0].get());
    std::vector<McePle> convCmd;
    std::vector<ethosn::command_stream::Convert> reshapeCmd;
    for (const auto& cmdHeader : cmdStream)
    {
        if (cmdHeader.m_Opcode() == Opcode::OPERATION_MCE_PLE)
        {
            convCmd.push_back(cmdHeader.GetCommand<Opcode::OPERATION_MCE_PLE>()->m_Data());
        }
        if (cmdHeader.m_Opcode() == Opcode::OPERATION_CONVERT)
        {
            reshapeCmd.push_back(cmdHeader.GetCommand<Opcode::OPERATION_CONVERT>()->m_Data());
        }
    }

    TensorShape conv1_output = { 1, 16, 16, 16 };

    // Check that we have NHWC with actual
    REQUIRE(convCmd.size() == 1);
    REQUIRE(convCmd[0].m_OutputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::NHWC);
    REQUIRE(convCmd[0].m_OutputInfo().m_TensorShape() == conv1_output);
    REQUIRE(convCmd[0].m_OutputInfo().m_SupertensorShape() == conv1_output);
    REQUIRE(convCmd[0].m_OutputInfo().m_DataLocation() == ethosn::command_stream::DataLocation::DRAM);

    REQUIRE(reshapeCmd.size() == 1);
}

/// Test reshape as last layer NHWCB DRAM with Strategy 0
TEST_CASE("Test Reshape as last layer NHWCB DRAM with Strategy 0")
{
    // Create the network
    CompilationOptions options;
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultCapabilities());
    std::shared_ptr<Operand> inputConv1 =
        AddInput(network, TensorInfo({ 1, 256, 128, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWCB)).tensor;

    std::shared_ptr<Constant> biasConv1 =
        AddConstant(network, TensorInfo({ 1, 1, 1, 16 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, 1.0f }),
                    std::vector<uint8_t>(16, 0).data())
            .tensor;

    std::shared_ptr<Constant> weightsConv1 =
        AddConstant(network, TensorInfo({ 1, 1, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO),
                    std::vector<uint8_t>(16 * 16 * 16, 0).data())
            .tensor;

    std::shared_ptr<Operand> conv1 =
        AddConvolution(network, *inputConv1, *biasConv1, *weightsConv1,
                       ConvolutionInfo(Padding(0, 0, 0, 0), Stride(1, 1), QuantizationInfo(0, 1.1f)))
            .tensor;

    std::shared_ptr<Operand> reshape = AddReshape(network, *conv1, { 1, 128, 256, 16 }).tensor;
    std::shared_ptr<Output> output   = AddOutput(network, *reshape, DataFormat::NHWCB).tensor;

    // Compile it
    std::vector<std::unique_ptr<CompiledNetwork>> compiledNetwork =
        ethosn::support_library::Compile(*network, CompilationOptions());

    // Extract all the conv commands
    using namespace ethosn::command_stream;
    CommandStream cmdStream = GetCommandStream(compiledNetwork[0].get());
    std::vector<McePle> convCmds;
    std::vector<ethosn::command_stream::Convert> reshapeCmd;
    for (const auto& cmdHeader : cmdStream)
    {
        if (cmdHeader.m_Opcode() == Opcode::OPERATION_MCE_PLE)
        {
            convCmds.push_back(cmdHeader.GetCommand<Opcode::OPERATION_MCE_PLE>()->m_Data());
        }
        if (cmdHeader.m_Opcode() == Opcode::OPERATION_CONVERT)
        {
            reshapeCmd.push_back(cmdHeader.GetCommand<Opcode::OPERATION_CONVERT>()->m_Data());
        }
    }

    TensorShape conv_output = { 1, 256, 128, 16 };

    REQUIRE(convCmds.size() == 1);
    REQUIRE(convCmds[0].m_SramConfig().m_AllocationStrategy() ==
            ethosn::command_stream::SramAllocationStrategy::STRATEGY_0);
    REQUIRE(convCmds[0].m_InputInfo().m_DataLocation() == ethosn::command_stream::DataLocation::DRAM);
    REQUIRE(convCmds[0].m_OutputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::NHWC);
    REQUIRE(convCmds[0].m_OutputInfo().m_DataLocation() == ethosn::command_stream::DataLocation::DRAM);
    REQUIRE(convCmds[0].m_OutputInfo().m_TensorShape() == conv_output);
    REQUIRE(convCmds[0].m_OutputInfo().m_SupertensorShape() == conv_output);

    REQUIRE(reshapeCmd.size() == 1);
    REQUIRE(reshapeCmd[0].m_InputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::NHWC);
    REQUIRE(reshapeCmd[0].m_InputInfo().m_DataLocation() == ethosn::command_stream::DataLocation::DRAM);
    REQUIRE(reshapeCmd[0].m_OutputInfo().m_DataFormat() == ethosn::command_stream::DataFormat::NHWCB);
    REQUIRE(reshapeCmd[0].m_OutputInfo().m_DataLocation() == ethosn::command_stream::DataLocation::DRAM);
}
