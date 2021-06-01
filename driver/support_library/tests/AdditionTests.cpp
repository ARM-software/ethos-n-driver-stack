//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"
#include "../include/ethosn_support_library/SupportQueries.hpp"
#include "TestUtils.hpp"
#include "Utils.hpp"

#include <catch.hpp>
#include <ethosn_command_stream/CommandStreamBuffer.hpp>

using namespace ethosn::support_library;

TEST_CASE("IsAdditionSupported")
{
    char reason[1024];

    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    SECTION("Unsupported cases")
    {
        QuantizationInfo outputQuantizationInfo;
        SECTION("Height not compatible")
        {
            TensorInfo input0 = TensorInfo({ 1, 2, 1, 4 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, { 0, 1.0f });
            TensorInfo input1 = TensorInfo({ 1, 3, 1, 4 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, { 0, 1.0f });
            REQUIRE(queries.IsAdditionSupported(input0, input1, outputQuantizationInfo, nullptr, reason,
                                                sizeof(reason)) == SupportedLevel::Unsupported);
            REQUIRE(Contains(reason, "Height must be either equal or one of the tensor's height must be 1"));
        }

        SECTION("Incorrect output info provided")
        {
            TensorInfo input0 = TensorInfo({ 1, 1, 1, 4 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, { 0, 1.0f });
            TensorInfo input1 = TensorInfo({ 1, 2, 3, 4 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, { 0, 1.0f });
            TensorInfo output({ 1, 1, 1, 4 });
            REQUIRE(queries.IsAdditionSupported(input0, input1, outputQuantizationInfo, &output, reason,
                                                sizeof(reason)) == SupportedLevel::Unsupported);
            REQUIRE(Contains(reason, "Provided outputInfo is incorrect"));
        }

        SECTION("Unsupported input data type")
        {
            TensorInfo input0 = TensorInfo({ 1, 1, 1, 4 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, 1.0f });
            TensorInfo input1 = TensorInfo({ 1, 2, 3, 4 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, 1.0f });
            TensorInfo output;
            REQUIRE(queries.IsAdditionSupported(input0, input1, outputQuantizationInfo, &output, reason,
                                                sizeof(reason)) == SupportedLevel::Unsupported);
            REQUIRE(Contains(reason, "Input to addition must be UINT8_QUANTIZED or INT8_QUANTIZED"));
        }

        SECTION("Mismatching input data types")
        {
            TensorInfo input0 = TensorInfo({ 1, 1, 1, 4 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, { 0, 1.0f });
            TensorInfo input1 = TensorInfo({ 1, 2, 3, 4 }, DataType::INT8_QUANTIZED, DataFormat::NHWC, { 0, 1.0f });
            TensorInfo output;
            REQUIRE(queries.IsAdditionSupported(input0, input1, outputQuantizationInfo, &output, reason,
                                                sizeof(reason)) == SupportedLevel::Unsupported);
            REQUIRE(Contains(reason, "Inputs to addition must have the same data type"));
        }
    }

    SECTION("EstimateOnly cases")
    {
        TensorInfo input0 = TensorInfo({ 1, 2, 3, 4 });
        TensorInfo output = TensorInfo({ 1, 2, 3, 4 });
        SECTION("Stretch width")
        {
            TensorInfo input1 = TensorInfo({ 1, 2, 1, 4 });
            QuantizationInfo outputQuantizationInfo;
            REQUIRE(queries.IsAdditionSupported(input0, input1, outputQuantizationInfo, &output, reason,
                                                sizeof(reason)) == SupportedLevel::EstimateOnly);
            REQUIRE(Contains(reason, "Cannot stretch along the requested dimensions"));
        }

        SECTION("Stretch channels")
        {
            TensorInfo input1 = TensorInfo({ 1, 2, 3, 1 });
            QuantizationInfo outputQuantizationInfo;
            REQUIRE(queries.IsAdditionSupported(input0, input1, outputQuantizationInfo, &output, reason,
                                                sizeof(reason)) == SupportedLevel::EstimateOnly);
            REQUIRE(Contains(reason, "Cannot stretch along the requested dimensions"));
        }
    }

    SECTION("Supported cases")
    {
        const auto inputDataType = GENERATE(DataType::UINT8_QUANTIZED, DataType::INT8_QUANTIZED);
        TensorInfo input0        = TensorInfo({ 1, 2, 3, 4 }, inputDataType, DataFormat::NHWC, { 2, 2.0f });
        TensorInfo input1        = TensorInfo({ 1, 2, 3, 4 }, inputDataType, DataFormat::NHWC, { 7, 7.0f });
        QuantizationInfo outputQuantizationInfo;

        SECTION("Output info not provided")
        {
            REQUIRE(queries.IsAdditionSupported(input0, input1, outputQuantizationInfo, nullptr) ==
                    SupportedLevel::Supported);
        }

        SECTION("Output info filled in for us")
        {
            TensorInfo outputInfo;
            REQUIRE(queries.IsAdditionSupported(input0, input1, outputQuantizationInfo, &outputInfo, reason,
                                                sizeof(reason)) == SupportedLevel::Supported);
            REQUIRE(outputInfo == TensorInfo({ 1, 2, 3, 4 }, inputDataType, DataFormat::NHWC, { 0, 1.0f }));
        }

        SECTION("Output info provided and correct")
        {
            TensorInfo outputInfo({ 1, 2, 3, 4 }, inputDataType, DataFormat::NHWC, { 0, 1.0f });
            REQUIRE(queries.IsAdditionSupported(input0, input1, outputQuantizationInfo, &outputInfo, reason,
                                                sizeof(reason)) == SupportedLevel::Supported);
        }

        SECTION("Output info provided but incorrect")
        {
            TensorInfo outputInfo({ 1, 2, 3, 4 }, inputDataType, DataFormat::NHWC, { 9, 9.0f });
            REQUIRE(queries.IsAdditionSupported(input0, input1, outputQuantizationInfo, &outputInfo, reason,
                                                sizeof(reason)) == SupportedLevel::Unsupported);
            REQUIRE(Contains(reason, "Provided outputInfo is incorrect"));
        }
    }
}

/// Checks the CompiledNetwork that the support_library produces for PLE Only Addition of 2 tensors is as expected
TEST_CASE("PleOnlyAddition2Tensors")
{
    const auto inputDataType         = GENERATE(DataType::INT8_QUANTIZED, DataType::UINT8_QUANTIZED);
    const auto expectedInputDataType = utils::GetCommandDataType(inputDataType);

    constexpr float inputScale = 0.5;
    TensorInfo inputInfo0{
        { { 1, 16, 16, 16 } },
        inputDataType,
        DataFormat::NHWC,
        { 0, inputScale },
    };
    TensorInfo inputInfo1{
        { { 1, 16, 16, 16 } },
        inputDataType,
        DataFormat::NHWC,
        { 0, inputScale },
    };

    CompilationOptions options       = GetDefaultCompilationOptions();
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultCapabilities());

    // Build up the network
    std::shared_ptr<Operand> input0   = AddInput(network, inputInfo0).tensor;
    std::shared_ptr<Operand> input1   = AddInput(network, inputInfo1).tensor;
    std::shared_ptr<Operand> addition = AddAddition(network, *input0, *input1, inputInfo0.m_QuantizationInfo).tensor;
    std::shared_ptr<Output> output    = AddOutput(network, *addition).tensor;

    std::vector<std::unique_ptr<CompiledNetwork>> compiledNetwork = ethosn::support_library::Compile(*network, options);

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
    REQUIRE(commands[0].m_InputInfo().m_DataType() == expectedInputDataType);
    REQUIRE(commands[0].m_InputInfo2().m_TensorShape() == TensorShape{ 1, 16, 16, 16 });
    REQUIRE(commands[0].m_InputInfo2().m_DataType() == expectedInputDataType);
    REQUIRE(commands[0].m_OutputInfo().m_TensorShape() == TensorShape{ 1, 16, 16, 16 });
    REQUIRE(commands[0].m_OutputInfo().m_DataType() == expectedInputDataType);
}

/// Checks that the support_library fails to build the network when the
/// addition input tensors shapes are not compatible.
TEST_CASE("PleOnlyAddition2Tensors fails to build the network")
{
    constexpr float inputScale = 0.5;
    TensorInfo inputInfo0{
        { { 1, 16, 16, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, inputScale },
    };
    TensorInfo inputInfo1{
        { { 1, 8, 8, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, inputScale },
    };

    CompilationOptions options       = GetDefaultCompilationOptions();
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultCapabilities());
    bool failed                      = false;

    // Build up the network
    std::shared_ptr<Operand> input0 = AddInput(network, inputInfo0).tensor;
    std::shared_ptr<Operand> input1 = AddInput(network, inputInfo1).tensor;
    try
    {
        std::shared_ptr<Operand> addition =
            AddAddition(network, *input0, *input1, inputInfo0.m_QuantizationInfo).tensor;
    }
    catch (const NotSupportedException&)
    {
        failed = true;
    }

    REQUIRE(failed);
}
