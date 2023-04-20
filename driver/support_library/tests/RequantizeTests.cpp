//
// Copyright © 2018-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"
#include "../include/ethosn_support_library/SupportQueries.hpp"
#include "../src/GraphNodes.hpp"
#include "../src/cascading/InputPart.hpp"
#include "../src/cascading/McePart.hpp"
#include "../src/cascading/NetworkToGraphOfPartsConverter.hpp"
#include "../src/cascading/OutputPart.hpp"
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

    SECTION("Requantize with different input/output valid type")
    {
        RequantizeInfo requantizeInfo   = RequantizeInfo(QuantizationInfo(-10, 1.0f));
        requantizeInfo.m_OutputDataType = DataType::INT8_QUANTIZED;
        REQUIRE(queries.IsRequantizeSupported(RequantizeInfo(requantizeInfo), input, nullptr) ==
                SupportedLevel::Supported);
    }

    SECTION("Successful case (output info with INT8_QUANTIZED type is supported and filled in)")
    {
        TensorInfo outputInfo;
        RequantizeInfo requantizeInfo   = RequantizeInfo(QuantizationInfo(0, 1.0f));
        requantizeInfo.m_OutputDataType = DataType::INT8_QUANTIZED;
        REQUIRE(queries.IsRequantizeSupported(RequantizeInfo(requantizeInfo), input, &outputInfo) ==
                SupportedLevel::Supported);
        REQUIRE(outputInfo ==
                TensorInfo({ 1, 16, 16, 16 }, DataType::INT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f)));
    }

    SECTION("Successful case (output info with UINT8_QUANTIZED type is supported and filled in)")
    {
        TensorInfo input({ 1, 16, 16, 16 }, DataType::INT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(-128, 1.0f));
        TensorInfo outputInfo;
        RequantizeInfo requantizeInfo   = RequantizeInfo(QuantizationInfo(0, 1.0f));
        requantizeInfo.m_OutputDataType = DataType::UINT8_QUANTIZED;
        REQUIRE(queries.IsRequantizeSupported(RequantizeInfo(requantizeInfo), input, &outputInfo) ==
                SupportedLevel::Supported);
        REQUIRE(outputInfo ==
                TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f)));
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

    SECTION("Invalid zero point for inputInfo")
    {
        TensorInfo input({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(-10, 1.0f));
        RequantizeInfo requantizeInfo = RequantizeInfo(QuantizationInfo(0, 1.0f));
        REQUIRE(queries.IsRequantizeSupported(requantizeInfo, input, nullptr, reason, sizeof(reason)) ==
                SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Zero point out of range for input info"));
    }

    SECTION("Invalid zero point for RequantizeInfo")
    {
        TensorInfo input({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        RequantizeInfo requantizeInfo   = RequantizeInfo(QuantizationInfo(-129, 1.0f));
        requantizeInfo.m_OutputDataType = DataType::INT8_QUANTIZED;
        REQUIRE(queries.IsRequantizeSupported(requantizeInfo, input, nullptr, reason, sizeof(reason)) ==
                SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Zero point out of range for requantizeInfo"));
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

    SECTION("Requantize with different input/output invalid type")
    {
        TensorInfo input({ 1, 1, 1, 2 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(0, 1.0f));
        RequantizeInfo requantizeInfo   = RequantizeInfo(QuantizationInfo(0, 1.0f));
        requantizeInfo.m_OutputDataType = DataType::INT32_QUANTIZED;
        REQUIRE(queries.IsRequantizeSupported(RequantizeInfo(requantizeInfo), input, nullptr, reason, sizeof(reason)) ==
                SupportedLevel::Unsupported);
    }

    SECTION("Requantize with incorrect outputInfo")
    {
        TensorInfo input({ 1, 16, 16, 16 }, DataType::INT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(-128, 1.0f));
        TensorInfo output({ 1, 16, 16, 16 }, DataType::INT8_QUANTIZED, DataFormat::NHWC, QuantizationInfo(-128, 1.0f));
        RequantizeInfo requantizeInfo   = RequantizeInfo(QuantizationInfo(0, 1.0f));
        requantizeInfo.m_OutputDataType = DataType::UINT8_QUANTIZED;
        REQUIRE(queries.IsRequantizeSupported(RequantizeInfo(requantizeInfo), input, &output, reason, sizeof(reason)) ==
                SupportedLevel::Unsupported);
        REQUIRE(Contains(reason, "Provided outputInfo is incorrect"));
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

    CompilationOptions compilationOptions{};
    compilationOptions.m_StrictPrecision = true;
    std::vector<std::unique_ptr<CompiledNetwork>> compiledNetwork =
        ethosn::support_library::Compile(*network, compilationOptions);

    REQUIRE(compiledNetwork.size() == 1);
}

// Tests that a network with a Requantization with a different input/output data type can compile
TEST_CASE("Compile a network with Requantize layer with different input/output types")
{
    const DataType inputType = GENERATE(DataType::UINT8_QUANTIZED, DataType::INT8_QUANTIZED);
    const DataType outputType =
        (inputType == DataType::UINT8_QUANTIZED) ? DataType::INT8_QUANTIZED : DataType::UINT8_QUANTIZED;

    auto network = CreateNetwork(GetRawDefaultCapabilities());

    TensorInfo inputInfo{
        { { 1, 16, 16, 16 } },
        inputType,
        DataFormat::NHWCB,
        { 127, 0.0627451017f },
    };

    RequantizeInfo requantInfo({ 0, 0.03f });
    requantInfo.m_OutputDataType = outputType;

    auto input      = AddInput(network, inputInfo).tensor;
    auto requantize = AddRequantize(network, *input, requantInfo).tensor;
    auto output     = AddOutput(network, *requantize).tensor;

    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    REQUIRE(graph.GetNumParts() == 3);

    // Part 0: Input
    REQUIRE(graph.GetPartInputs(0).size() == 0);
    REQUIRE(graph.GetPartOutputs(0).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 0, 0 }).has_value() == false);

    const InputPart* inputPart0 = dynamic_cast<const InputPart*>(&graph.GetPart(0));
    REQUIRE(inputPart0 != nullptr);

    Plans plansInputPart0 =
        inputPart0->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    CHECK(plansInputPart0.size() == 1);

    Buffer* bufferOutputPart0 = plansInputPart0[0].GetOutputBuffer(PartOutputSlot{ inputPart0->GetPartId(), 0 });
    REQUIRE(bufferOutputPart0 != nullptr);
    if (bufferOutputPart0)
    {
        REQUIRE(bufferOutputPart0->m_TensorShape == TensorShape{ 1, 16, 16, 16 });
        REQUIRE(bufferOutputPart0->m_DataType == inputType);
    }

    // Part 1: DEPTHWISE_CONVOLUTION on mce
    REQUIRE(graph.GetPartInputs(1).size() == 1);
    REQUIRE(graph.GetPartOutputs(1).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 1, 0 }).value().m_PartId == 0);

    const McePart* part = dynamic_cast<const McePart*>(&graph.GetPart(1));
    REQUIRE(part != nullptr);
    auto operation = part->GetMceOperation();
    REQUIRE(operation.has_value());
    // Identity McePart is executed as depthwise convolution
    REQUIRE(operation.value() == ethosn::command_stream::MceOperation::DEPTHWISE_CONVOLUTION);

    // Part 2: Output
    REQUIRE(dynamic_cast<const OutputPart*>(&graph.GetPart(2)) != nullptr);
    REQUIRE(graph.GetPartInputs(2).size() == 1);
    REQUIRE(graph.GetPartOutputs(2).size() == 0);
    REQUIRE(graph.GetConnectedOutputSlot({ 2, 0 }).value().m_PartId == 1);
    REQUIRE(graph.GetConnectedInputSlots({ 2, 0 }).size() == 0);

    const OutputPart* outputPart2 = dynamic_cast<const OutputPart*>(&graph.GetPart(2));
    REQUIRE(outputPart2 != nullptr);

    Plans plansOutputPart2 =
        outputPart2->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    CHECK(plansOutputPart2.size() == 1);

    Buffer* bufferInputPart2 = plansOutputPart2[0].GetInputBuffer(PartInputSlot{ outputPart2->GetPartId(), 0 });
    REQUIRE(bufferInputPart2 != nullptr);
    if (bufferInputPart2)
    {
        REQUIRE(bufferInputPart2->m_TensorShape == TensorShape{ 1, 16, 16, 16 });
        REQUIRE(bufferInputPart2->m_DataType == outputType);
    }
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
