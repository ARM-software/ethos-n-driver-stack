//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"
#include "../include/ethosn_support_library/SupportQueries.hpp"
#include "../src/GraphNodes.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>
#include <ethosn_command_stream/CommandStreamBuffer.hpp>

using namespace ethosn;
using namespace ethosn::support_library;

TEST_CASE("Requantize Supported")
{
    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    TensorInfo input({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
    TensorInfo output({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
    REQUIRE(queries.IsRequantizeSupported(RequantizeInfo(QuantizationInfo(0, 1.0f)), input, &output) ==
            SupportedLevel::Supported);

    SECTION("Output Scale larger than minimum")
    {
        REQUIRE(queries.IsRequantizeSupported(RequantizeInfo(QuantizationInfo(0, 1.f / 127.99f)), input, nullptr) ==
                SupportedLevel::Supported);
    }
}

TEST_CASE("Requantize Unsupported")
{
    char reason[1024];
    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    SECTION("Incorrect output shape")
    {
        TensorInfo input({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        TensorInfo output({ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        REQUIRE(queries.IsRequantizeSupported(RequantizeInfo(QuantizationInfo(0, 1.0f)), input, &output, reason,
                                              sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Provided outputInfo is incorrect"));
    }

    SECTION("Invalid zero point")
    {
        TensorInfo input({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        REQUIRE(queries.IsRequantizeSupported(RequantizeInfo(QuantizationInfo(-10, 1.0f)), input, nullptr, reason,
                                              sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Zero point out of range"));
    }

    SECTION("Per channel quantization not supported")
    {
        TensorInfo input({ 1, 1, 1, 2 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        QuantizationInfo quantizationInfo(0, { 0.5f, 0.4f }, 3);
        REQUIRE(queries.IsRequantizeSupported(RequantizeInfo(quantizationInfo), input, nullptr, reason,
                                              sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Quantization Dim should not be used on Output"));
    }

    SECTION("Multiple output quantization scales in the output")
    {
        TensorInfo input({ 1, 1, 1, 2 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        QuantizationInfo quantizationInfo(0, 0.5);
        quantizationInfo.SetScales(std::vector<float>{ 0.5f, 0.4f });
        REQUIRE(queries.IsRequantizeSupported(RequantizeInfo(quantizationInfo), input, nullptr, reason,
                                              sizeof(reason)) == SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Output quantization scales must have a size of 1"));
    }
}

TEST_CASE("Requantize EstimateOnly")
{
    char reason[1024];
    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    SECTION("Output Scale smaller than minimum")
    {
        TensorInfo input({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        REQUIRE(queries.IsRequantizeSupported(RequantizeInfo(QuantizationInfo(0, 1.0f / 128.f)), input, nullptr, reason,
                                              sizeof(reason)) == SupportedLevel::EstimateOnly);
        REQUIRE(Contains(reason, "Output scale must be bigger than input scale / 128"));
    }
}

/// Tests that a network comprising a single Requantize creates an identity depthwise convolution beforehand.
TEST_CASE("Add Requantize to a network")
{
    SupportQueries queries(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Create the network
    CompilationOptions options       = GetDefaultCompilationOptions();
    std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultCapabilities());

    const TensorInfo inputInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC,
                               QuantizationInfo(33, 2.0f));
    std::shared_ptr<Operand> input = AddInput(network, inputInfo).tensor;
    // overallScale = inputScale * weightScale / outputScale  must be >= 0 and < 1,
    // which is overallScale = 2.0f * 0.5f (Identity) / 1.5f
    // if the Requantize quantization info is not taken into account the weights encoder will assert.
    std::shared_ptr<Operand> requantize =
        AddRequantize(network, *input, RequantizeInfo(QuantizationInfo(127, 1.5f))).tensor;
    std::shared_ptr<Output> output = AddOutput(network, *requantize).tensor;

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
    REQUIRE(convCmds[0].m_MceData().m_OutputShape() == TensorShape{ 1, 16, 16, 16 });
    // Check output zero point
    REQUIRE(convCmds[0].m_OutputInfo().m_ZeroPoint() == (127));
}

/// Tests that a network comprising a single Requantize creates an mce operation beforehand.
TEST_CASE("Single Requantize EstimateOnly")
{

    // Create the estimation network
    CompilationOptions options       = GetDefaultCompilationOptions();
    std::shared_ptr<Network> network = CreateEstimationNetwork(GetRawDefaultCapabilities());
    std::shared_ptr<Operand> input   = AddInput(network, TensorInfo({ 1, 16, 16, 16 })).tensor;
    std::shared_ptr<Operand> requantize =
        AddRequantize(network, *input, RequantizeInfo(QuantizationInfo(0, 1.0f))).tensor;
    std::shared_ptr<Output> output = AddOutput(network, *requantize).tensor;

    // Estimate it
    EstimationOptions estimationOptions{};
    estimationOptions.m_Current = true;

    std::vector<PassPerformanceData> perfData = EstimatePerformance(*network, options, estimationOptions).m_Stream;

    // Check that it has completed.
    REQUIRE(perfData.size() > 0U);
    // Check that it's a Mce plus Fused Ple operation.
    REQUIRE(perfData.at(0).m_Stats.m_Mce.m_CycleCount == 32U);
    REQUIRE(perfData.at(0).m_Stats.m_Ple.m_NumOfPatches == 16U);
}

/// Tests that a network with a requantization with an output scale less than half the input scale can compile
TEST_CASE("Requantize output scale less than half input scale")
{
    auto network = CreateNetwork(GetRawDefaultCapabilities());

    TensorInfo inputInfo{
        { { 1, 16, 16, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWCB,
        { 128, 0.0627451017f },
    };

    RequantizeInfo requantInfo({ 0, 0.03f });

    auto input      = AddInput(network, inputInfo).tensor;
    auto requantize = AddRequantize(network, *input, requantInfo).tensor;
    auto output     = AddOutput(network, *requantize).tensor;

    CompilationOptions compilationOptions = GetDefaultCompilationOptions();
    compilationOptions.m_StrictPrecision  = true;
    std::vector<std::unique_ptr<CompiledNetwork>> compiledNetwork =
        ethosn::support_library::Compile(*network, compilationOptions);

    REQUIRE(compiledNetwork.size() == 1);
}

TEST_CASE("RequantizeNode::Apply UINT8")
{
    GIVEN("A RequantizeNode designed to requantize from [-1, 1] to [-0.5, 3.5]")
    {
        QuantizationInfo inputQuantInfo(128, 2 / 255.0f);
        QuantizationInfo outputQuantInfo(32, 4 / 255.0f);
        RequantizeNode r(0, TensorShape{ 1, 1, 1, 1 }, DataType::UINT8_QUANTIZED, outputQuantInfo,
                         CompilerDataFormat::NHWC, {});

        AND_GIVEN("MceData with relu bounds of [-0.75, 0.5] in the original quant space")
        {
            command_stream::MceData mceData;
            mceData.m_ActivationMin() = 32;
            mceData.m_ActivationMax() = 192;

            WHEN("Telling the RequantizeNode to modify the MceData")
            {
                r.Apply(mceData, inputQuantInfo);

                THEN("The MceData's relu bounds is modified to represent the same bounds in the new quant space")
                {
                    // Note we can't represent the lower bound of -0.75 in the new space, so it is clamped
                    REQUIRE(mceData.m_ActivationMin() == 0);
                    REQUIRE(mceData.m_ActivationMax() == 64);
                }
            }
        }
    }
}

TEST_CASE("RequantizeNode::Apply INT8")
{
    GIVEN("A RequantizeNode designed to requantize from [-1, 1] to [-0.5, 3.5]")
    {
        QuantizationInfo inputQuantInfo(0, 2 / 255.0f);
        QuantizationInfo outputQuantInfo(-96, 4 / 255.0f);
        RequantizeNode r(0, TensorShape{ 1, 1, 1, 1 }, DataType::INT8_QUANTIZED, outputQuantInfo,
                         CompilerDataFormat::NHWC, {});

        AND_GIVEN("MceData with relu bounds of [-0.75, 0.5] in the original quant space")
        {
            command_stream::MceData mceData;
            mceData.m_ActivationMin() = -96;
            mceData.m_ActivationMax() = 64;

            WHEN("Telling the RequantizeNode to modify the MceData")
            {
                r.Apply(mceData, inputQuantInfo);

                THEN("The MceData's relu bounds is modified to represent the same bounds in the new quant space")
                {
                    // Note we can't represent the lower bound of -0.75 in the new space, so it is clamped
                    REQUIRE(mceData.m_ActivationMin() == -128);
                    REQUIRE(mceData.m_ActivationMax() == -64);
                }
            }
        }
    }
}
