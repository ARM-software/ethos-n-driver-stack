//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"
#include "../src/cascading/InputPart.hpp"
#include "../src/cascading/McePart.hpp"
#include "../src/cascading/NetworkToGraphOfPartsConverter.hpp"
#include "../src/cascading/OutputPart.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>
#include <ethosn_command_stream/CommandStreamBuffer.hpp>

using namespace ethosn::support_library;

/// Tests compiler option to omit dump commands.
TEST_CASE("DumpCmdDisabled")
{
    CompilationOptions compOpt;
    compOpt.m_DebugInfo.m_DumpRam = false;

    TensorInfo inputInfo{
        { { 1, 16, 16, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    TensorInfo biasInfo0{
        { { 1, 1, 1, 16 } },
        DataType::INT32_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };
    const std::vector<uint8_t> biasData0(utils::TotalSizeBytes(biasInfo0));

    TensorInfo weightsInfo0{
        { { 1, 1, 16, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::HWIO,
        { 0, 1.f },
    };
    const std::vector<uint8_t> weightsData0(utils::TotalSizeBytes(weightsInfo0));

    ConvolutionInfo convInfo0{
        { 0, 0, 0, 0 },
        { 1, 1 },
        { 0, 1.f },
    };

    SplitInfo splitInfo1{ 1, { 9, 7 } };

    // Create the network
    // Input -> Conv -> Split -> Output
    //                        -> Output
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultCapabilities());

    std::shared_ptr<Operand> input = AddInput(network, inputInfo).tensor;

    std::shared_ptr<Constant> bias0    = AddConstant(network, biasInfo0, biasData0.data()).tensor;
    std::shared_ptr<Constant> weights0 = AddConstant(network, weightsInfo0, weightsData0.data()).tensor;
    std::shared_ptr<Operand> conv0     = AddConvolution(network, *input, *bias0, *weights0, convInfo0).tensor;

    std::vector<std::shared_ptr<Operand>> split1 = AddSplit(network, *conv0, splitInfo1).tensors;

    std::shared_ptr<Output> output1 = AddOutput(network, *split1[0]).tensor;
    std::shared_ptr<Output> output2 = AddOutput(network, *split1[1]).tensor;

    std::vector<std::unique_ptr<CompiledNetwork>> compiledNetwork = ethosn::support_library::Compile(*network, compOpt);

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
    CompilationOptions compOpt;
    compOpt.m_DebugInfo.m_DumpRam = true;

    TensorInfo inputInfo{
        { { 1, 16, 16, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    TensorInfo biasInfo0{
        { { 1, 1, 1, 16 } },
        DataType::INT32_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };
    const std::vector<uint8_t> biasData0(utils::TotalSizeBytes(biasInfo0));

    TensorInfo weightsInfo0{
        { { 1, 1, 16, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::HWIO,
        { 0, 1.f },
    };
    const std::vector<uint8_t> weightsData0(utils::TotalSizeBytes(weightsInfo0));

    ConvolutionInfo convInfo0{
        { 0, 0, 0, 0 },
        { 1, 1 },
        { 0, 1.f },
    };

    SplitInfo splitInfo1{ 1, { 9, 7 } };

    // Create the network
    // Input -> Conv -> Split -> Output
    //                        -> Output
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultCapabilities());

    std::shared_ptr<Operand> input = AddInput(network, inputInfo).tensor;

    std::shared_ptr<Constant> bias0    = AddConstant(network, biasInfo0, biasData0.data()).tensor;
    std::shared_ptr<Constant> weights0 = AddConstant(network, weightsInfo0, weightsData0.data()).tensor;
    std::shared_ptr<Operand> conv0     = AddConvolution(network, *input, *bias0, *weights0, convInfo0).tensor;

    std::vector<std::shared_ptr<Operand>> split1 = AddSplit(network, *conv0, splitInfo1).tensors;

    std::shared_ptr<Output> output1 = AddOutput(network, *split1[0]).tensor;
    std::shared_ptr<Output> output2 = AddOutput(network, *split1[1]).tensor;

    std::vector<std::unique_ptr<CompiledNetwork>> compiledNetwork = ethosn::support_library::Compile(*network, compOpt);

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
    CompilationOptions options;
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
