//
// Copyright © 2018-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"
#include "../include/ethosn_support_library/SupportQueries.hpp"
#include "../src/ThreadPool.hpp"
#include "../src/cascading/EstimateOnlyPart.hpp"
#include "../src/cascading/InputPart.hpp"
#include "../src/cascading/NetworkToGraphOfPartsConverter.hpp"
#include "../src/cascading/OutputPart.hpp"
#include "../src/cascading/Part.hpp"
#include "../src/cascading/StandalonePlePart.hpp"
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

        SECTION("Invalid zero point range")
        {
            TensorInfo input0 = TensorInfo({ 1, 1, 1, 4 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, { 0, 1.0f });
            TensorInfo input1 = TensorInfo({ 1, 1, 1, 4 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, { -10, 1.0f });
            REQUIRE(queries.IsAdditionSupported(input0, input1, outputQuantizationInfo, nullptr, reason,
                                                sizeof(reason)) == SupportedLevel::Unsupported);
            INFO(reason);
            REQUIRE(Contains(reason, "Zero point out of range for input1 info"));

            input1.m_QuantizationInfo.SetZeroPoint(0);
            REQUIRE(queries.IsAdditionSupported(input0, input1, QuantizationInfo(-10, 1.0f), nullptr, reason,
                                                sizeof(reason)) == SupportedLevel::Unsupported);
            INFO(reason);
            REQUIRE(Contains(reason, "Zero point out of range for outputQuantizationInfo"));
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
    const DataType inputType   = GENERATE(DataType::UINT8_QUANTIZED, DataType::INT8_QUANTIZED);
    constexpr float inputScale = 0.5;

    TensorInfo inputInfo0{
        { { 1, 16, 16, 16 } },
        inputType,
        DataFormat::NHWC,
        { 0, inputScale },
    };
    TensorInfo inputInfo1{
        { { 1, 16, 16, 16 } },
        inputType,
        DataFormat::NHWC,
        { 0, inputScale },
    };

    CompilationOptions options{};

    const std::shared_ptr<Network> network = CreateNetwork(GetRawDefaultCapabilities());

    // Build up the network
    std::shared_ptr<Operand> input0   = AddInput(network, inputInfo0).tensor;
    std::shared_ptr<Operand> input1   = AddInput(network, inputInfo1).tensor;
    std::shared_ptr<Operand> addition = AddAddition(network, *input0, *input1, inputInfo0.m_QuantizationInfo).tensor;
    std::shared_ptr<Output> output    = AddOutput(network, *addition).tensor;

    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    ThreadPool threadPool(0);
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext,
                                                                  threadPool);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    // Check for each Part:
    //  * Whether the type of the generated Part is correct
    //  * Whether the PleOperation command stream is correct for Operations using StandalonePlePart
    //  * The number of Input/Output slots
    //  * Whether PartInputSlots connect to PartOutputSlots of the correct Part
    //  * For the last Part, check that there are no connections to any following PartInputSlots
    REQUIRE(graph.GetNumParts() == 4);

    // Part 0: Input
    REQUIRE(graph.GetPartInputs(0).size() == 0);
    REQUIRE(graph.GetPartOutputs(0).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 0, 0 }).has_value() == false);

    const InputPart* inputPart0 = dynamic_cast<const InputPart*>(&graph.GetPart(0));
    REQUIRE(inputPart0 != nullptr);

    Plans plansInputPart0 =
        inputPart0->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, { nullptr }, 1);
    CHECK(plansInputPart0.size() == 1);

    Buffer* bufferOutputPart0 = plansInputPart0[0].GetOutputBuffer(PartOutputSlot{ inputPart0->GetPartId(), 0 });
    REQUIRE(bufferOutputPart0 != nullptr);
    if (bufferOutputPart0)
    {
        REQUIRE(bufferOutputPart0->m_TensorShape == TensorShape{ 1, 16, 16, 16 });
        REQUIRE(bufferOutputPart0->m_DataType == inputType);
    }

    // Part 1: Input
    REQUIRE(graph.GetPartInputs(1).size() == 0);
    REQUIRE(graph.GetPartOutputs(1).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 1, 0 }).has_value() == false);

    const InputPart* inputPart1 = dynamic_cast<const InputPart*>(&graph.GetPart(1));
    REQUIRE(inputPart1 != nullptr);

    Plans plansInputPart1 =
        inputPart1->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, { nullptr }, 1);
    CHECK(plansInputPart1.size() == 1);

    Buffer* bufferOutputPart1 = plansInputPart1[0].GetOutputBuffer(PartOutputSlot{ inputPart1->GetPartId(), 0 });
    REQUIRE(bufferOutputPart1 != nullptr);
    if (bufferOutputPart1)
    {
        REQUIRE(bufferOutputPart1->m_TensorShape == TensorShape{ 1, 16, 16, 16 });
        REQUIRE(bufferOutputPart1->m_DataType == inputType);
    }

    // Part 2: Add with ple
    REQUIRE(graph.GetPartInputs(2).size() == 2);
    REQUIRE(graph.GetPartOutputs(2).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 2, 0 }).value().m_PartId == 0);
    REQUIRE(graph.GetConnectedOutputSlot({ 2, 1 }).value().m_PartId == 1);

    const StandalonePlePart* additionPlePart = dynamic_cast<const StandalonePlePart*>(&graph.GetPart(2));
    REQUIRE(additionPlePart != nullptr);
    Plans additionPlans =
        additionPlePart->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, { nullptr }, 1);
    Op* maybePleOpAdditionPlans = additionPlans[0].m_OpGraph.GetOp(0);
    REQUIRE(IsPleOp(maybePleOpAdditionPlans));
    PleOp* pleOpAdditionPlans = static_cast<PleOp*>(maybePleOpAdditionPlans);
    REQUIRE(pleOpAdditionPlans->m_Op == ethosn::command_stream::PleOperation::ADDITION);

    // Part 3: Output
    REQUIRE(graph.GetPartInputs(3).size() == 1);
    REQUIRE(graph.GetPartOutputs(3).size() == 0);
    REQUIRE(graph.GetConnectedOutputSlot({ 3, 0 }).value().m_PartId == 2);
    REQUIRE(graph.GetConnectedInputSlots({ 3, 0 }).size() == 0);

    const OutputPart* outputPart3 = dynamic_cast<const OutputPart*>(&graph.GetPart(3));
    REQUIRE(outputPart3 != nullptr);

    Plans plansOutputPart3 =
        outputPart3->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, { nullptr }, 1);
    CHECK(plansOutputPart3.size() == 1);

    Buffer* bufferInputPart3 = plansOutputPart3[0].GetInputBuffer(PartInputSlot{ outputPart3->GetPartId(), 0 });
    REQUIRE(bufferInputPart3 != nullptr);
    if (bufferInputPart3)
    {
        REQUIRE(bufferInputPart3->m_TensorShape == TensorShape{ 1, 16, 16, 16 });
        REQUIRE(bufferInputPart3->m_DataType == inputType);
    }
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

    CompilationOptions options;
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
