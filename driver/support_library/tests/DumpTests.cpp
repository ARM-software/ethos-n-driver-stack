//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>
#include <ethosn_command_stream/CommandStreamBuffer.hpp>

using namespace ethosn::support_library;

/// Tests compiler option to omit dump commands.
TEST_CASE("DumpCmdDisabled")
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
    std::shared_ptr<Output> output1 = AddOutput(network, *conv).tensor;

    std::vector<std::unique_ptr<CompiledNetwork>> compiledNetwork =
        ethosn::support_library::Compile(*network, GetDefaultCompilationOptions());

    // Check that there are no dump commands in the stream
    using namespace ethosn::command_stream;
    CommandStream cmdStream = GetCommandStream(compiledNetwork[0].get());

    for (const auto& cmdHeader : cmdStream)
    {
        REQUIRE(cmdHeader.m_Opcode() != Opcode::DUMP_DRAM);
        REQUIRE(cmdHeader.m_Opcode() != Opcode::DUMP_SRAM);
    }
}

/// Tests default compiler option to include dump commands.
TEST_CASE("DumpCmdEnabled")
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

    std::shared_ptr<Output> output1 = AddOutput(network, *conv).tensor;

    options.m_DebugInfo.m_DumpRam = true;

    std::vector<std::unique_ptr<CompiledNetwork>> compiledNetwork = ethosn::support_library::Compile(*network, options);

    // Check that there are dump commands present in the stream
    using namespace ethosn::command_stream;
    CommandStream cmdStream = GetCommandStream(compiledNetwork[0].get());

    bool foundDumpCmd = false;
    for (const auto& cmdHeader : cmdStream)
    {
        foundDumpCmd |= cmdHeader.m_Opcode() == Opcode::DUMP_DRAM;
        foundDumpCmd |= cmdHeader.m_Opcode() == Opcode::DUMP_SRAM;
    }
    REQUIRE(foundDumpCmd);
}

TEST_CASE("initialSramDump")
{
    // Create the network
    CompilationOptions options       = GetDefaultCompilationOptions();
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultCapabilities());
    std::shared_ptr<Operand> input   = AddInput(network, TensorInfo({ 1, 16, 16, 16 })).tensor;
    std::shared_ptr<Operand> relu    = AddRelu(network, *input, ReluInfo(10, 250)).tensor;
    std::shared_ptr<Output> output   = AddOutput(network, *relu).tensor;

    // Set to dump sram at start of stream
    options.m_DebugInfo.m_InitialSramDump = true;

    // Compile it
    std::vector<std::unique_ptr<CompiledNetwork>> compiledNetwork = ethosn::support_library::Compile(*network, options);

    // Check command stream
    using namespace ethosn::command_stream;
    CommandStream cmdStream = GetCommandStream(compiledNetwork[0].get());

    // Check that the command stream starts with the SRAM dump
    bool intialDump       = false;
    const auto& cmdHeader = cmdStream.begin();
    if (cmdHeader->m_Opcode() == Opcode::DUMP_SRAM)
    {
        intialDump = true;
    }
    REQUIRE(intialDump == true);
}
