//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"
#include "../include/ethosn_support_library/SupportQueries.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>

using namespace ethosn::support_library;

TEST_CASE("TanhSupported")
{
    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    SECTION("Supported configuration")
    {
        const auto inputDataType = GENERATE(DataType::INT8_QUANTIZED, DataType::UINT8_QUANTIZED);
        const QuantizationInfo outQuantization((inputDataType == DataType::INT8_QUANTIZED) ? 0 : 128, 1.0f / 128);
        TensorInfo input({ 1, 16, 16, 16 }, inputDataType, DataFormat::NHWC, QuantizationInfo(127, 1.0f));
        TensorInfo output({ 1, 16, 16, 16 }, inputDataType, DataFormat::NHWC, outQuantization);
        REQUIRE(queries.IsTanhSupported(input, &output) == SupportedLevel::Supported);
    }

    SECTION("OutputInfo nullptr")
    {
        const auto inputDataType = GENERATE(DataType::UINT8_QUANTIZED, DataType::UINT8_QUANTIZED);
        TensorInfo input({ 1, 16, 16, 16 }, inputDataType, DataFormat::NHWC, QuantizationInfo(127, 1.0f));
        REQUIRE(queries.IsTanhSupported(input, nullptr) == SupportedLevel::Supported);
    }

    SECTION("Output info filled in")
    {
        TensorInfo output;
        TensorInfo input({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(127, 1.0f));
        REQUIRE(queries.IsTanhSupported(input, &output) == SupportedLevel::Supported);
        REQUIRE(output == TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC,
                                     QuantizationInfo(128, 1.0f / 128)));
    }

    SECTION("Wrong quantization")
    {
        TensorInfo input({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(127, 1.0f));
        TensorInfo output({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        REQUIRE(queries.IsTanhSupported(input, &output) == SupportedLevel::Unsupported);
    }

    SECTION("Wrong size")
    {
        TensorInfo input({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(127, 1.0f));
        TensorInfo output({ 1, 8, 8, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC,
                          QuantizationInfo(0, 1.0f / 256));
        REQUIRE(queries.IsTanhSupported(input, &output) == SupportedLevel::Unsupported);
    }
}

TEST_CASE("Add Tanh")
{
    // Create the network
    CompilationOptions options;
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultCapabilities());
    std::shared_ptr<Operand> input   = AddInput(network, TensorInfo({ 1, 16, 16, 16 })).tensor;
    std::shared_ptr<Operand> tanh    = AddTanh(network, *input).tensor;
    std::shared_ptr<Output> output   = AddOutput(network, *tanh).tensor;

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

    // Check that the conv commands are as expected. There should be one which has a tanh afterwards.
    REQUIRE(convCmds.size() == 1);
    REQUIRE((convCmds[0].m_MceData().m_ActivationMin() == 0 && convCmds[0].m_MceData().m_ActivationMax() == 44));
    REQUIRE(
        (convCmds[0].m_PleData().m_RescaleMultiplier0() == 47274 && convCmds[0].m_PleData().m_RescaleShift0() == 6));
}
