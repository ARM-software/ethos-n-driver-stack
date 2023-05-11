//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"
#include "../src/DebuggingContext.hpp"
#include "../src/Network.hpp"
#include "../src/Utils.hpp"
#include "../src/cascading/ConcatPart.hpp"
#include "../src/cascading/ConstantPart.hpp"
#include "../src/cascading/EstimateOnlyPart.hpp"
#include "../src/cascading/FullyConnectedPart.hpp"
#include "../src/cascading/FusedPlePart.hpp"
#include "../src/cascading/InputPart.hpp"
#include "../src/cascading/McePart.hpp"
#include "../src/cascading/NetworkToGraphOfPartsConverter.hpp"
#include "../src/cascading/OutputPart.hpp"
#include "../src/cascading/Part.hpp"
#include "../src/cascading/ReshapePart.hpp"
#include "../src/cascading/SplitPart.hpp"
#include "../src/cascading/StandalonePlePart.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>

#include <fstream>
#include <iostream>
#include <unordered_set>

using namespace ethosn::support_library;

TEST_CASE("CreateIdentityMcePartWithPaddedOutputChannels")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    auto compare = [&](const std::vector<std::pair<uint32_t, uint32_t>>& padAmounts,
                       const std::vector<uint8_t>& expectedWeights) {
        std::unique_ptr<McePart> paddingPart = CreateIdentityMcePartWithPaddedOutputChannels(
            0, TensorShape{ 1, 1, 1, 5 }, QuantizationInfo(), QuantizationInfo(), 0, DataType::UINT8_QUANTIZED,
            DataType::UINT8_QUANTIZED, estOpt, compOpt, caps, padAmounts, debuggingContext);

        CHECK(paddingPart->GetWeightsData() == expectedWeights);
    };

    // clang-format off
    compare({}, { // No padding - identity matrix
        2, 0, 0, 0, 0,
        0, 2, 0, 0, 0,
        0, 0, 2, 0, 0,
        0, 0, 0, 2, 0,
        0, 0, 0, 0, 2,
    });
    compare({ { 0, 2 }, { 2, 3 } }, {
        0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 2, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 2, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
        });
    // clang-format on
}

TEST_CASE("CreateIdentityMcePartWithRemovedInputChannels")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    auto compare = [&](const std::vector<std::pair<uint32_t, uint32_t>>& removeAmounts,
                       const std::vector<uint8_t>& expectedWeights) {
        std::unique_ptr<McePart> paddingPart = CreateIdentityMcePartWithRemovedInputChannels(
            0, TensorShape{ 1, 1, 1, 5 }, QuantizationInfo(), QuantizationInfo(), 0, DataType::UINT8_QUANTIZED,
            DataType::UINT8_QUANTIZED, estOpt, compOpt, caps, removeAmounts, debuggingContext);

        // Get the weights
        CHECK(paddingPart->GetWeightsData() == expectedWeights);
    };

    // clang-format off
    compare({}, { // No removing - identity matrix
        2, 0, 0, 0, 0,
        0, 2, 0, 0, 0,
        0, 0, 2, 0, 0,
        0, 0, 0, 2, 0,
        0, 0, 0, 0, 2,
        });
    compare({ { 0, 1 }, { 3, 2 } }, {
        0, 0,
        2, 0,
        0, 2,
        0, 0,
        0, 0,
        });
    // clang-format on
}

// Manually creates a Network of Operands and Operations and converts it to a GraphOfParts using the NetworkToGraphOfPartsConverter().
// The topology is chosen to test Networks of supported Part types such as:
//      * Input Part
//      * Mce Part
//      * Pooling Part (Max 2x2_2_2 variation))
//      * Reshape Part
//      * Output Part
TEST_CASE("NetworkToGraphOfPartsConverterTest")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{
        { { 1, 128, 128, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    TensorInfo biasInfo{
        { { 1, 1, 1, 16 } },
        DataType::INT32_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    TensorInfo bias2Info{
        { { 1, 1, 1, 16 } },
        DataType::INT32_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.1f },
    };

    TensorInfo weightsInfo{
        { { 3, 3, 16, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::HWIO,
        { 0, 1.f },
    };

    ConvolutionInfo convInfo{
        { 0, 0, 0, 0 },
        { 1, 1 },
        { 0, 1.1f },
    };

    ConvolutionInfo conv2Info{
        { 0, 0, 0, 0 },
        { 2, 2 },
        { 0, 1.2f },
    };

    constexpr PoolingInfo poolingInfo{ 2, 2, 2, 2, { 0, 0, 0, 0 }, PoolingType::MAX };
    constexpr TensorShape reshapeInfo{ { 1, 126, 126, 16 } };
    const std::vector<uint8_t> biasData(utils::TotalSizeBytes(biasInfo));
    const std::vector<uint8_t> bias2Data(utils::TotalSizeBytes(bias2Info));
    const std::vector<uint8_t> weightsData(utils::TotalSizeBytes(weightsInfo));

    const std::shared_ptr<Network> network =
        CreateNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Network topology:
    // { Input, Constant, Constant } -> Convolution -> Reshape -> Pooling -> Convolution -> Output

    std::shared_ptr<Operand> input       = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Constant> bias       = AddConstant(network, biasInfo, biasData.data()).tensor;
    std::shared_ptr<Constant> bias2      = AddConstant(network, bias2Info, bias2Data.data()).tensor;
    std::shared_ptr<Constant> weights    = AddConstant(network, weightsInfo, weightsData.data()).tensor;
    std::shared_ptr<Operand> conv        = AddConvolution(network, *input, *bias, *weights, convInfo).tensor;
    std::shared_ptr<Operand> reshape     = AddReshape(network, *conv, reshapeInfo).tensor;
    std::shared_ptr<Operand> pooling     = AddPooling(network, *reshape, poolingInfo).tensor;
    std::shared_ptr<Operand> convStrided = AddConvolution(network, *pooling, *bias2, *weights, conv2Info).tensor;
    std::shared_ptr<Output> output       = AddOutput(network, *convStrided).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest Output.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::High);
    }

    // Check for each Part:
    //  * Whether the type of the generated Part is correct
    //  * The number of Input/Output slots
    //  * Whether PartInputSlots connect to PartOutputSlots of the preceding Part
    //  * For the last Part, check that there are no connections to any following PartInputSlots
    REQUIRE(graph.GetNumParts() == 7);

    REQUIRE(dynamic_cast<const InputPart*>(&graph.GetPart(0)) != nullptr);
    REQUIRE(graph.GetPartInputs(0).size() == 0);
    REQUIRE(graph.GetPartOutputs(0).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 0, 0 }).has_value() == false);

    REQUIRE(dynamic_cast<const McePart*>(&graph.GetPart(1)) != nullptr);
    REQUIRE(graph.GetPartInputs(1).size() == 1);
    REQUIRE(graph.GetPartOutputs(1).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 1, 0 }).value().m_PartId == 0);

    REQUIRE(dynamic_cast<const ReshapePart*>(&graph.GetPart(2)) != nullptr);
    REQUIRE(graph.GetPartInputs(2).size() == 1);
    REQUIRE(graph.GetPartOutputs(2).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 2, 0 }).value().m_PartId == 1);

    REQUIRE(dynamic_cast<const FusedPlePart*>(&graph.GetPart(3)) != nullptr);
    REQUIRE(graph.GetPartInputs(3).size() == 1);
    REQUIRE(graph.GetPartOutputs(3).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 3, 0 }).value().m_PartId == 2);

    REQUIRE(dynamic_cast<const FusedPlePart*>(&graph.GetPart(4)) != nullptr);
    REQUIRE(graph.GetPartInputs(4).size() == 1);
    REQUIRE(graph.GetPartOutputs(4).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 4, 0 }).value().m_PartId == 3);

    REQUIRE(dynamic_cast<const McePart*>(&graph.GetPart(5)) != nullptr);
    REQUIRE(graph.GetPartInputs(5).size() == 1);
    REQUIRE(graph.GetPartOutputs(5).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 5, 0 }).value().m_PartId == 4);

    REQUIRE(dynamic_cast<const OutputPart*>(&graph.GetPart(6)) != nullptr);
    REQUIRE(graph.GetPartInputs(6).size() == 1);
    REQUIRE(graph.GetPartOutputs(6).size() == 0);
    REQUIRE(graph.GetConnectedOutputSlot({ 6, 0 }).value().m_PartId == 5);
    REQUIRE(graph.GetConnectedInputSlots({ 6, 0 }).size() == 0);
}

// Manually creates a Network of Operands and Operations and converts it to a GraphOfParts using the
// NetworkToGraphOfPartsConverter().
TEST_CASE("NetworkToGraphOfPartsConverterTest Requantize Same Quantization")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{
        { { 1, 128, 128, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    const std::shared_ptr<Network> network =
        CreateNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    std::shared_ptr<Operand> input = AddInput(network, inputInfo).tensor;

    // Add the remaining Operations for this Unit Test
    std::shared_ptr<Operand> requantize = AddRequantize(network, *input, RequantizeInfo({ 0, 1.f })).tensor;
    std::shared_ptr<Output> output      = AddOutput(network, *requantize).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest Requantize.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest Requantize Output.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::High);
    }

    // Check for each Part:
    //  * Whether the type of the generated Part is correct
    //  * The number of Input/Output slots
    //  * Whether PartInputSlots connect to PartOutputSlots of the correct Part
    //  * For the last Part, check that there are no connections to any following PartInputSlots
    REQUIRE(graph.GetNumParts() == 2);

    REQUIRE(dynamic_cast<const InputPart*>(&graph.GetPart(0)) != nullptr);
    REQUIRE(graph.GetPartInputs(0).size() == 0);
    REQUIRE(graph.GetPartOutputs(0).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 0, 0 }).has_value() == false);

    REQUIRE(dynamic_cast<const OutputPart*>(&graph.GetPart(1)) != nullptr);
    REQUIRE(graph.GetPartInputs(1).size() == 1);
    REQUIRE(graph.GetPartOutputs(1).size() == 0);
    REQUIRE(graph.GetConnectedOutputSlot({ 1, 0 }).value().m_PartId == 0);
    REQUIRE(graph.GetConnectedInputSlots({ 1, 0 }).size() == 0);
}

// Manually creates a Network of Operands and Operations and converts it to a GraphOfParts using the
// NetworkToGraphOfPartsConverter().
TEST_CASE("NetworkToGraphOfPartsConverterTest Requantize")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{
        { { 1, 128, 128, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    const std::shared_ptr<Network> network =
        CreateNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    std::shared_ptr<Operand> input = AddInput(network, inputInfo).tensor;

    // Add the remaining Operations for this Unit Test
    std::shared_ptr<Operand> requantize = AddRequantize(network, *input, RequantizeInfo({ 1, 1.2f })).tensor;
    std::shared_ptr<Output> output      = AddOutput(network, *requantize).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest Requantize.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest Requantize Output.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::High);
    }

    // Check for each Part:
    //  * Whether the type of the generated Part is correct
    //  * The number of Input/Output slots
    //  * Whether PartInputSlots connect to PartOutputSlots of the correct Part
    //  * For the last Part, check that there are no connections to any following PartInputSlots
    REQUIRE(graph.GetNumParts() == 3);

    REQUIRE(dynamic_cast<const InputPart*>(&graph.GetPart(0)) != nullptr);
    REQUIRE(graph.GetPartInputs(0).size() == 0);
    REQUIRE(graph.GetPartOutputs(0).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 0, 0 }).has_value() == false);

    const McePart* part = dynamic_cast<const McePart*>(&graph.GetPart(1));
    REQUIRE(part != nullptr);
    REQUIRE(graph.GetPartInputs(1).size() == 1);
    REQUIRE(graph.GetPartOutputs(1).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 1, 0 }).value().m_PartId == 0);
    auto operation = part->GetMceOperation();
    REQUIRE(operation.has_value());
    // Identity McePart is executed as depthwise convolution
    REQUIRE(operation.value() == ethosn::command_stream::MceOperation::DEPTHWISE_CONVOLUTION);

    REQUIRE(dynamic_cast<const OutputPart*>(&graph.GetPart(2)) != nullptr);
    REQUIRE(graph.GetPartInputs(2).size() == 1);
    REQUIRE(graph.GetPartOutputs(2).size() == 0);
    REQUIRE(graph.GetConnectedOutputSlot({ 2, 0 }).value().m_PartId == 1);
    REQUIRE(graph.GetConnectedInputSlots({ 2, 0 }).size() == 0);
}

TEST_CASE("NetworkToGraphOfPartsConverter Requantize EstimateOnly")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{ { { 1, 1, 1, 16 } }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, { 0, 2.f } };

    // Output scale must be bigger than input scale / 128, so this will return EstimateOnly
    // when IsRequantizeSupported is called.
    RequantizeInfo requantizeInfo({ 1, 0.01f });

    const std::shared_ptr<Network> network =
        CreateEstimationNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));
    std::shared_ptr<Operand> input      = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Operand> requantize = AddRequantize(network, *input, requantizeInfo).tensor;
    std::shared_ptr<Output> output      = AddOutput(network, *requantize).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest Requantize EstimateOnly.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest Requantize EstimateOnly Output.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::High);
    }

    // InputPart, McePart, OutputPart
    REQUIRE(graph.GetNumParts() == 3);

    // We check only the EstimateOnlyPart that we expect to be created - the Input and Output part and connections
    // between the Parts are covered by NetworkToGraphOfPartsConverterTest
    const EstimateOnlyPart* estimateOnlyPart = dynamic_cast<const EstimateOnlyPart*>(&graph.GetPart(1));
    REQUIRE(estimateOnlyPart != nullptr);
    auto plans = estimateOnlyPart->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    REQUIRE(plans[0].GetInputBuffer(PartInputSlot{ estimateOnlyPart->GetPartId(), 0 })->m_TensorShape ==
            TensorShape{ 1, 1, 1, 16 });
    REQUIRE(plans[0].GetOutputBuffer(PartOutputSlot{ estimateOnlyPart->GetPartId(), 0 })->m_TensorShape ==
            TensorShape{ 1, 1, 1, 16 });
    Op* maybeEstimateOnlyOp = plans[0].m_OpGraph.GetOp(0);
    REQUIRE(IsEstimateOnlyOp(maybeEstimateOnlyOp));
    EstimateOnlyOp* estimateOnlyOp = static_cast<EstimateOnlyOp*>(maybeEstimateOnlyOp);
    CHECK(estimateOnlyOp->m_ReasonForEstimateOnly.find("Output scale must be bigger than input scale / 128") !=
          std::string::npos);
}

// Manually creates a Network of Operands and Operations and converts it to a GraphOfParts using the
// NetworkToGraphOfPartsConverter().
// The topology is chosen to test Networks of supported Part types such as:
//      * Concat Part
TEST_CASE("NetworkToGraphOfPartsConverterTest Concat")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{
        { { 1, 128, 128, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    TensorInfo input2Info{
        { { 1, 128, 128, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.2f },
    };

    TensorInfo input3Info{
        { { 1, 128, 128, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 1, 1.2f },
    };

    const std::shared_ptr<Network> network =
        CreateNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Network topology:
    /*
       { Input3 } \
       { Input2 }  -> Concatenation -> Output
       { Input  } /
    */

    // Add 2x Inputs with different quantization information from the Concatenation.
    // This will trigger the creation of 2x MceParts added to the respective Inputs of the ConcatPart.
    std::vector<Operand*> layers;
    std::shared_ptr<Operand> input = AddInput(network, inputInfo).tensor;
    layers.push_back(input.get());
    std::shared_ptr<Operand> input2 = AddInput(network, input2Info).tensor;
    layers.push_back(input2.get());

    // Add a third Input with the same quantization information as the Concatenation.
    // This will test whether the Concatenation Visitor function connects all generated Parts (ConcatPart, McePart(s)) correctly.
    std::shared_ptr<Operand> input3 = AddInput(network, input3Info).tensor;
    layers.push_back(input3.get());

    // Add the remaining Operations for this Unit Test
    std::shared_ptr<Operand> concat = AddConcatenation(network, layers, ConcatenationInfo(3, { 1, 1.2f })).tensor;
    std::shared_ptr<Output> output  = AddOutput(network, *concat).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest Concat.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest Concat Output.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::High);
    }

    // Check for each Part:
    //  * Whether the type of the generated Part is correct
    //  * The number of Input/Output slots
    //  * Whether PartInputSlots connect to PartOutputSlots of the correct Part
    //  * For the last Part, check that there are no connections to any following PartInputSlots
    REQUIRE(graph.GetNumParts() == 7);

    REQUIRE(dynamic_cast<const InputPart*>(&graph.GetPart(0)) != nullptr);
    REQUIRE(graph.GetPartInputs(0).size() == 0);
    REQUIRE(graph.GetPartOutputs(0).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 0, 0 }).has_value() == false);

    REQUIRE(dynamic_cast<const InputPart*>(&graph.GetPart(1)) != nullptr);
    REQUIRE(graph.GetPartInputs(1).size() == 0);
    REQUIRE(graph.GetPartOutputs(1).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 1, 0 }).has_value() == false);

    REQUIRE(dynamic_cast<const InputPart*>(&graph.GetPart(2)) != nullptr);
    REQUIRE(graph.GetPartInputs(2).size() == 0);
    REQUIRE(graph.GetPartOutputs(2).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 2, 0 }).has_value() == false);

    REQUIRE(dynamic_cast<const McePart*>(&graph.GetPart(3)) != nullptr);
    REQUIRE(graph.GetPartInputs(3).size() == 1);
    REQUIRE(graph.GetPartOutputs(3).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 3, 0 }).value().m_PartId == 0);

    REQUIRE(dynamic_cast<const McePart*>(&graph.GetPart(4)) != nullptr);
    REQUIRE(graph.GetPartInputs(4).size() == 1);
    REQUIRE(graph.GetPartOutputs(4).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 4, 0 }).value().m_PartId == 1);

    REQUIRE(dynamic_cast<const ConcatPart*>(&graph.GetPart(5)) != nullptr);
    REQUIRE(graph.GetPartInputs(5).size() == 3);
    REQUIRE(graph.GetPartOutputs(5).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 5, 0 }).value().m_PartId == 3);
    REQUIRE(graph.GetConnectedOutputSlot({ 5, 1 }).value().m_PartId == 4);
    REQUIRE(graph.GetConnectedOutputSlot({ 5, 2 }).value().m_PartId == 2);

    REQUIRE(dynamic_cast<const OutputPart*>(&graph.GetPart(6)) != nullptr);
    REQUIRE(graph.GetPartInputs(6).size() == 1);
    REQUIRE(graph.GetPartOutputs(6).size() == 0);
    REQUIRE(graph.GetConnectedOutputSlot({ 6, 0 }).value().m_PartId == 5);
    REQUIRE(graph.GetConnectedInputSlots({ 6, 0 }).size() == 0);
}

// Test the Network to graph of parts converter with a concat operation that must use NHWC
TEST_CASE("NetworkToGraphOfPartsConverterTest Concat NHWC")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{
        { { 1, 1, 1, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    // Have a concat axis not a multiple of the brick group shape
    TensorInfo input2Info{
        { { 1, 1, 5, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    const std::shared_ptr<Network> network =
        CreateNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Network topology:
    /*
       { Input2 } \
                  -> Concatenation -> Output
       { Input  } /
    */

    std::vector<Operand*> layers;
    std::shared_ptr<Operand> input = AddInput(network, inputInfo).tensor;
    layers.push_back(input.get());
    std::shared_ptr<Operand> input2 = AddInput(network, input2Info).tensor;
    layers.push_back(input2.get());

    std::shared_ptr<Operand> concat = AddConcatenation(network, layers, ConcatenationInfo(2, { 0, 1.f })).tensor;
    std::shared_ptr<Output> output  = AddOutput(network, *concat).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest Concat.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest Concat Output.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::High);
    }

    REQUIRE(dynamic_cast<const ConcatPart*>(&graph.GetPart(2)) != nullptr);
    REQUIRE(graph.GetPartInputs(2).size() == 2);
    REQUIRE(graph.GetPartOutputs(2).size() == 1);

    // The plans generated from this concat part should have NHWC input and output buffers.
    Plans plans =
        graph.GetPart(2).GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{ 16u, 16u }, nullptr, 0);
    for (auto&& plan : plans)
    {
        auto& inputMappings = plan.m_InputMappings;
        for (auto&& inputMapping : inputMappings)
        {
            REQUIRE(inputMapping.first->m_Format == CascadingBufferFormat::NHWC);
        }
        auto& outputMappings = plan.m_OutputMappings;
        for (auto&& outputMapping : outputMappings)
        {
            REQUIRE(outputMapping.first->m_Format == CascadingBufferFormat::NHWC);
        }
    }
}

TEST_CASE("NetworkToGraphOfPartsConverter Concat Padding")
{
    // Confirms that padding channels are added as expected.

    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo input1Info{
        { { 1, 16, 16, 1 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.0f },
    };

    TensorInfo input2Info{
        { { 1, 16, 16, 1 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.0f },
    };

    const std::shared_ptr<Network> network =
        std::make_shared<Network>(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO), false, true);

    // Network topology:
    /*
       { Input3 } \
       { Input2 }  -> Concatenation -> Output
       { Input  } /
    */

    std::vector<Operand*> layers;
    std::shared_ptr<Operand> input = AddInput(network, input1Info).tensor;
    layers.push_back(input.get());
    std::shared_ptr<Operand> input2 = AddInput(network, input2Info).tensor;
    layers.push_back(input2.get());

    std::shared_ptr<Operand> concat = AddConcatenation(network, layers, ConcatenationInfo(3, { 0, 1.0f })).tensor;
    std::shared_ptr<Output> output  = AddOutput(network, *concat).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverter Concat Padding.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverter Concat Padding Output.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::High);
    }

    // InputPart, InputPart, ConcatPart, McePart (to remove padding channels), OutputPart
    REQUIRE(graph.GetNumParts() == 5);

    const ConcatPart* concatPart = dynamic_cast<const ConcatPart*>(&graph.GetPart(2));
    REQUIRE(concatPart != nullptr);
    CHECK(graph.GetConnectedOutputSlot({ 2, 0 }).value().m_PartId == 0);
    CHECK(graph.GetConnectedOutputSlot({ 2, 1 }).value().m_PartId == 1);
    CHECK(graph.GetConnectedInputSlots({ 2, 0 }) == std::vector<PartInputSlot>{ { 3, 0 } });
    // Check the concat offsets
    CHECK(utils::GetChannels(concatPart->GetOutputTensorShape()) == 32);
    CHECK(concatPart->GetOffsets() == std::vector<uint32_t>{ 0, 16 });

    const McePart* mcePart = dynamic_cast<const McePart*>(&graph.GetPart(3));
    REQUIRE(mcePart != nullptr);
    CHECK(graph.GetConnectedOutputSlot({ 3, 0 }).value().m_PartId == 2);
    CHECK(graph.GetConnectedInputSlots({ 3, 0 }) == std::vector<PartInputSlot>{ { 4, 0 } });
    // Check that padding channels have been added
    CHECK(utils::GetChannels(mcePart->GetInputTensorShape()) == 32);
    // clang-format off
    CHECK(mcePart->GetWeightsData() == std::vector<uint8_t>{
        2, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 2,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
    });
    // clang-format on
}

TEST_CASE("NetworkToGraphOfPartsConverterTest Concat EstimateOnly")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    // Concatenation with the output scale too small relative to the input scale is EstimateOnly
    TensorInfo input1Info{
        { 1, 16, 16, 24 },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.0f },
    };

    TensorInfo input2Info{
        { 1, 16, 16, 24 },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.0f },
    };

    ConcatenationInfo concatenationInfo{ 3, { 0, 0.0001f } };

    const std::shared_ptr<Network> network =
        CreateEstimationNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Network topology:
    /*
       { Input2 }  -> Concatenation -> Output
       { Input1 } /
    */

    // Add 2x Inputs from the Concatenation.
    std::vector<Operand*> layers;
    std::shared_ptr<Operand> input1 = AddInput(network, input1Info).tensor;
    layers.push_back(input1.get());
    std::shared_ptr<Operand> input2 = AddInput(network, input1Info).tensor;
    layers.push_back(input2.get());

    // Add the remaining Operations for this Unit Test
    std::shared_ptr<Operand> concat = AddConcatenation(network, layers, concatenationInfo).tensor;
    std::shared_ptr<Output> output  = AddOutput(network, *concat).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest Concat EstimateOnly.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest Concat EstimateOnly Output.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::High);
    }

    // InputPart1, InputPart2, ConcatPart, OutputPart
    REQUIRE(graph.GetNumParts() == 4);

    // We check only the EstimateOnlyPart that we expect to be created - the Input and Output part and connections
    // between the Parts are covered by NetworkToGraphOfPartsConverterTest
    const EstimateOnlyPart* estimateOnlyPart = dynamic_cast<const EstimateOnlyPart*>(&graph.GetPart(2));
    REQUIRE(estimateOnlyPart != nullptr);
    auto plans = estimateOnlyPart->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    REQUIRE(plans[0].GetInputBuffer(PartInputSlot{ estimateOnlyPart->GetPartId(), 0 })->m_TensorShape ==
            TensorShape{ 1, 16, 16, 24 });
    REQUIRE(plans[0].GetOutputBuffer(PartOutputSlot{ estimateOnlyPart->GetPartId(), 0 })->m_TensorShape ==
            TensorShape{ 1, 16, 16, 48 });
    Op* maybeEstimateOnlyOp = plans[0].m_OpGraph.GetOp(0);
    REQUIRE(IsEstimateOnlyOp(maybeEstimateOnlyOp));
    EstimateOnlyOp* estimateOnlyOp = static_cast<EstimateOnlyOp*>(maybeEstimateOnlyOp);
    CHECK(estimateOnlyOp->m_ReasonForEstimateOnly.find("Output scales must be bigger than input scale / 128") !=
          std::string::npos);
}

// Manually creates a Network of Operands and Operations and converts it to a GraphOfParts using the NetworkToGraphOfPartsConverter().
// The topology is chosen to test Networks of supported Part types such as:
//      * MeanXy Part (7x7, 8x8 variations)
//      * Pooling Part (MeanXy_7x7, MeanXy_8x8 variations)
TEST_CASE("NetworkToGraphOfPartsConverterTest MeanXy")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo_7x7{
        { { 1, 7, 7, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    TensorInfo inputInfo_8x8{
        { { 1, 8, 8, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    // Add MeanXy info in the form of PoolingInfo, for use with Pooling Visitor.
    // Both options for strides 1,1 and 2,2 are tested
    constexpr PoolingInfo poolingInfo_7x7{ 7, 7, 1, 1, { 0, 0, 0, 0 }, PoolingType::AVG };
    constexpr PoolingInfo poolingInfo_8x8{ 8, 8, 2, 2, { 0, 0, 0, 0 }, PoolingType::AVG };
    const std::shared_ptr<Network> network =
        CreateNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Network topology:
    /*
                     /> Pooling (MeanXy_8x8) -> OutputPool_8x8
       { Input_8x8 } -> MeanXy_8x8 -> Output_8x8
                     /> Pooling (MeanXy_7x7) -> OutputPool_7x7
       { Input_7x7 } -> MeanXy_7x7 -> Output_7x7
    */

    std::shared_ptr<Operand> input_7x7      = AddInput(network, inputInfo_7x7).tensor;
    std::shared_ptr<Operand> meanxy_7x7     = AddMeanXy(network, *input_7x7).tensor;
    std::shared_ptr<Output> output_7x7      = AddOutput(network, *meanxy_7x7).tensor;
    std::shared_ptr<Operand> meanxyPool_7x7 = AddPooling(network, *input_7x7, poolingInfo_7x7).tensor;
    std::shared_ptr<Output> outputPool_7x7  = AddOutput(network, *meanxyPool_7x7).tensor;
    std::shared_ptr<Operand> input_8x8      = AddInput(network, inputInfo_8x8).tensor;
    std::shared_ptr<Operand> meanxy_8x8     = AddMeanXy(network, *input_8x8).tensor;
    std::shared_ptr<Output> output_8x8      = AddOutput(network, *meanxy_8x8).tensor;
    std::shared_ptr<Operand> meanxyPool_8x8 = AddPooling(network, *input_8x8, poolingInfo_8x8).tensor;
    std::shared_ptr<Output> outputPool_8x8  = AddOutput(network, *meanxyPool_8x8).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest MeanXy.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest MeanXy Output.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::High);
    }

    // Check for each Part:
    //  * Whether the type of the generated Part is correct
    //  * Whether the PleOperation command stream is correct for Operations using FusedPleParts (e.g. MeanXy_7x7, MeanXy_8x8 ...)
    //  * The number of Input/Output slots
    //  * Whether PartInputSlots connect to PartOutputSlots of the correct Part
    //  * For the last Part, check that there are no connections to any following PartInputSlots
    REQUIRE(graph.GetNumParts() == 10);

    // MeanXy_7x7
    // Checks on Parts generated from MeanXy Visitor.
    REQUIRE(dynamic_cast<const InputPart*>(&graph.GetPart(0)) != nullptr);
    REQUIRE(graph.GetPartInputs(0).size() == 0);
    REQUIRE(graph.GetPartOutputs(0).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 0, 0 }).has_value() == false);

    const FusedPlePart* meanxyPlePart_7x7 = dynamic_cast<const FusedPlePart*>(&graph.GetPart(1));
    REQUIRE(meanxyPlePart_7x7 != nullptr);
    auto meanxyPlans_7x7 =
        meanxyPlePart_7x7->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    Op* maybePleOpMeanXy7x7 = meanxyPlans_7x7[0].m_OpGraph.GetOp(2);
    REQUIRE(IsPleOp(maybePleOpMeanXy7x7));
    PleOp* pleOpMeanXy7x7 = static_cast<PleOp*>(maybePleOpMeanXy7x7);
    REQUIRE(pleOpMeanXy7x7->m_Op == ethosn::command_stream::PleOperation::MEAN_XY_7X7);
    REQUIRE(graph.GetPartInputs(1).size() == 1);
    REQUIRE(graph.GetPartOutputs(1).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 1, 0 }).value().m_PartId == 0);

    REQUIRE(dynamic_cast<const OutputPart*>(&graph.GetPart(2)) != nullptr);
    REQUIRE(graph.GetPartInputs(2).size() == 1);
    REQUIRE(graph.GetPartOutputs(2).size() == 0);
    REQUIRE(graph.GetConnectedOutputSlot({ 2, 0 }).value().m_PartId == 1);
    REQUIRE(graph.GetConnectedInputSlots({ 2, 0 }).size() == 0);

    // Checks on Parts generated from Pooling Visitor.
    const FusedPlePart* meanxyPoolPlePart_7x7 = dynamic_cast<const FusedPlePart*>(&graph.GetPart(3));
    REQUIRE(meanxyPoolPlePart_7x7 != nullptr);
    auto meanxyPoolPlans_7x7 =
        meanxyPoolPlePart_7x7->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    Op* maybePleOpMeanXyPoolPlans7x7 = meanxyPoolPlans_7x7[0].m_OpGraph.GetOp(2);
    REQUIRE(IsPleOp(maybePleOpMeanXyPoolPlans7x7));
    PleOp* pleOpMeanXyPool7x7 = static_cast<PleOp*>(maybePleOpMeanXyPoolPlans7x7);
    REQUIRE(pleOpMeanXyPool7x7->m_Op == ethosn::command_stream::PleOperation::MEAN_XY_7X7);
    REQUIRE(graph.GetPartInputs(3).size() == 1);
    REQUIRE(graph.GetPartOutputs(3).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 3, 0 }).value().m_PartId == 0);

    REQUIRE(dynamic_cast<const OutputPart*>(&graph.GetPart(4)) != nullptr);
    REQUIRE(graph.GetPartInputs(4).size() == 1);
    REQUIRE(graph.GetPartOutputs(4).size() == 0);
    REQUIRE(graph.GetConnectedOutputSlot({ 4, 0 }).value().m_PartId == 3);
    REQUIRE(graph.GetConnectedInputSlots({ 4, 0 }).size() == 0);

    // MeanXy_8x8
    // Checks on Parts generated from MeanXy Visitor.
    REQUIRE(dynamic_cast<const InputPart*>(&graph.GetPart(5)) != nullptr);
    REQUIRE(graph.GetPartInputs(5).size() == 0);
    REQUIRE(graph.GetPartOutputs(5).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 5, 0 }).has_value() == false);

    const FusedPlePart* meanxyPlePart_8x8 = dynamic_cast<const FusedPlePart*>(&graph.GetPart(6));
    REQUIRE(meanxyPlePart_8x8 != nullptr);
    auto meanxyPlans_8x8 =
        meanxyPlePart_8x8->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    Op* maybePleOpMeanXyPlans8x8 = meanxyPlans_8x8[0].m_OpGraph.GetOp(2);
    REQUIRE(IsPleOp(maybePleOpMeanXyPlans8x8));
    PleOp* pleOpMeanXyPlans8x8 = static_cast<PleOp*>(maybePleOpMeanXyPlans8x8);
    REQUIRE(pleOpMeanXyPlans8x8->m_Op == ethosn::command_stream::PleOperation::MEAN_XY_8X8);
    REQUIRE(graph.GetPartInputs(6).size() == 1);
    REQUIRE(graph.GetPartOutputs(6).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 6, 0 }).value().m_PartId == 5);

    REQUIRE(dynamic_cast<const OutputPart*>(&graph.GetPart(7)) != nullptr);
    REQUIRE(graph.GetPartInputs(7).size() == 1);
    REQUIRE(graph.GetPartOutputs(7).size() == 0);
    REQUIRE(graph.GetConnectedOutputSlot({ 7, 0 }).value().m_PartId == 6);
    REQUIRE(graph.GetConnectedInputSlots({ 7, 0 }).size() == 0);

    // Checks on Parts generated from Pooling Visitor.
    const FusedPlePart* meanxyPoolPlePart_8x8 = dynamic_cast<const FusedPlePart*>(&graph.GetPart(8));
    REQUIRE(meanxyPoolPlePart_8x8 != nullptr);
    auto meanxyPoolPlans_8x8 =
        meanxyPoolPlePart_8x8->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    Op* maybePleOpMeanXyPoolPlans8x8 = meanxyPoolPlans_8x8[0].m_OpGraph.GetOp(2);
    REQUIRE(IsPleOp(maybePleOpMeanXyPoolPlans8x8));
    PleOp* pleOpMeanXyPoolPlans8x8 = static_cast<PleOp*>(maybePleOpMeanXyPoolPlans8x8);
    REQUIRE(pleOpMeanXyPoolPlans8x8->m_Op == ethosn::command_stream::PleOperation::MEAN_XY_8X8);
    REQUIRE(graph.GetPartInputs(8).size() == 1);
    REQUIRE(graph.GetPartOutputs(8).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 8, 0 }).value().m_PartId == 5);

    REQUIRE(dynamic_cast<const OutputPart*>(&graph.GetPart(9)) != nullptr);
    REQUIRE(graph.GetPartInputs(9).size() == 1);
    REQUIRE(graph.GetPartOutputs(9).size() == 0);
    REQUIRE(graph.GetConnectedOutputSlot({ 9, 0 }).value().m_PartId == 8);
    REQUIRE(graph.GetConnectedInputSlots({ 9, 0 }).size() == 0);
}

// Manually creates a Network of Operands and Operations and converts it to a GraphOfParts using the NetworkToGraphOfPartsConverter().
// The topology is chosen to test Networks of supported Part types such as:
//      * LeakyRelu Part
//      * Sigmoid Part
//      * Tanh Part
TEST_CASE("NetworkToGraphOfPartsConverterTest LeakyRelu Sigmoid Tanh")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{
        { { 1, 7, 7, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    LeakyReluInfo leakyreluInfo{
        0.1f,
        { 0, 1.f },
    };

    const std::shared_ptr<Network> network =
        CreateNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Network topology:
    /*
                 /-> LeakyRelu -> Output3
       { Input } - > Sigmoid -> Output2
                 \-> Tanh -> Output
    */

    std::shared_ptr<Operand> input     = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Operand> tanh      = AddTanh(network, *input).tensor;
    std::shared_ptr<Output> output     = AddOutput(network, *tanh).tensor;
    std::shared_ptr<Operand> sigmoid   = AddSigmoid(network, *input).tensor;
    std::shared_ptr<Output> output2    = AddOutput(network, *sigmoid).tensor;
    std::shared_ptr<Operand> leakyrelu = AddLeakyRelu(network, *input, leakyreluInfo).tensor;
    std::shared_ptr<Output> output3    = AddOutput(network, *leakyrelu).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest LeakyRelu Sigmoid Tanh.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest LeakyRelu Sigmoid Tanh Output.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::High);
    }

    // Check for each Part:
    //  * Whether the type of the generated Part is correct
    //  * Whether the PleOperation command stream is correct for Operations using FusedPleParts (e.g. LeakyRelu, Sigmoid, Tanh ...)
    //  * The number of Input/Output slots
    //  * Whether PartInputSlots connect to PartOutputSlots of the correct Part
    //  * For the last Part, check that there are no connections to any following PartInputSlots
    REQUIRE(graph.GetNumParts() == 7);

    REQUIRE(dynamic_cast<const InputPart*>(&graph.GetPart(0)) != nullptr);
    REQUIRE(graph.GetPartInputs(0).size() == 0);
    REQUIRE(graph.GetPartOutputs(0).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 0, 0 }).has_value() == false);

    const FusedPlePart* tanhPlePart = dynamic_cast<const FusedPlePart*>(&graph.GetPart(1));
    REQUIRE(tanhPlePart != nullptr);
    auto tanhPlans = tanhPlePart->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    Op* maybePleOpTanhPlans = tanhPlans[0].m_OpGraph.GetOp(2);
    REQUIRE(IsPleOp(maybePleOpTanhPlans));
    PleOp* pleOpTanhPlans = static_cast<PleOp*>(maybePleOpTanhPlans);
    REQUIRE(pleOpTanhPlans->m_Op == ethosn::command_stream::PleOperation::SIGMOID);
    REQUIRE(graph.GetPartInputs(1).size() == 1);
    REQUIRE(graph.GetPartOutputs(1).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 1, 0 }).value().m_PartId == 0);

    REQUIRE(dynamic_cast<const OutputPart*>(&graph.GetPart(2)) != nullptr);
    REQUIRE(graph.GetPartInputs(2).size() == 1);
    REQUIRE(graph.GetPartOutputs(2).size() == 0);
    REQUIRE(graph.GetConnectedOutputSlot({ 2, 0 }).value().m_PartId == 1);
    REQUIRE(graph.GetConnectedInputSlots({ 2, 0 }).size() == 0);

    const FusedPlePart* sigmoidPlePart = dynamic_cast<const FusedPlePart*>(&graph.GetPart(3));
    REQUIRE(sigmoidPlePart != nullptr);
    auto sigmoidPlans =
        sigmoidPlePart->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    Op* maybePleOpSigmoidPlans = sigmoidPlans[0].m_OpGraph.GetOp(2);
    REQUIRE(IsPleOp(maybePleOpSigmoidPlans));
    PleOp* pleOpSigmoidPlans = static_cast<PleOp*>(maybePleOpSigmoidPlans);
    REQUIRE(pleOpSigmoidPlans->m_Op == ethosn::command_stream::PleOperation::SIGMOID);
    REQUIRE(graph.GetPartInputs(3).size() == 1);
    REQUIRE(graph.GetPartOutputs(3).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 3, 0 }).value().m_PartId == 0);

    REQUIRE(dynamic_cast<const OutputPart*>(&graph.GetPart(4)) != nullptr);
    REQUIRE(graph.GetPartInputs(4).size() == 1);
    REQUIRE(graph.GetPartOutputs(4).size() == 0);
    REQUIRE(graph.GetConnectedOutputSlot({ 4, 0 }).value().m_PartId == 3);
    REQUIRE(graph.GetConnectedInputSlots({ 4, 0 }).size() == 0);

    const FusedPlePart* leakyreluPlePart = dynamic_cast<const FusedPlePart*>(&graph.GetPart(5));
    REQUIRE(leakyreluPlePart != nullptr);
    auto leakyreluPlans =
        leakyreluPlePart->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    Op* maybePleOpLeakyReluPlans = leakyreluPlans[0].m_OpGraph.GetOp(2);
    REQUIRE(IsPleOp(maybePleOpLeakyReluPlans));
    PleOp* pleOpLeakyReluPlans = static_cast<PleOp*>(maybePleOpLeakyReluPlans);
    REQUIRE(pleOpLeakyReluPlans->m_Op == ethosn::command_stream::PleOperation::LEAKY_RELU);
    REQUIRE(graph.GetPartInputs(5).size() == 1);
    REQUIRE(graph.GetPartOutputs(5).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 5, 0 }).value().m_PartId == 0);

    REQUIRE(dynamic_cast<const OutputPart*>(&graph.GetPart(6)) != nullptr);
    REQUIRE(graph.GetPartInputs(6).size() == 1);
    REQUIRE(graph.GetPartOutputs(6).size() == 0);
    REQUIRE(graph.GetConnectedOutputSlot({ 6, 0 }).value().m_PartId == 5);
    REQUIRE(graph.GetConnectedInputSlots({ 6, 0 }).size() == 0);
}

TEST_CASE("NetworkToGraphOfPartsConverter LeakyRelu EstimateOnly")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{
        { { 1, 16, 16, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    // Leaky relu alpha must be less than 1 and greater than 0,
    // so this will return EstimateOnly when IsLeakyReluSupported is called.
    LeakyReluInfo leakyreluInfo{
        1.5f,
        { 0, 1.f },
    };

    const std::shared_ptr<Network> network =
        CreateEstimationNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Network topology:
    // Input -> Relu -> Output
    std::shared_ptr<Operand> input     = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Operand> leakyrelu = AddLeakyRelu(network, *input, leakyreluInfo).tensor;
    std::shared_ptr<Output> output     = AddOutput(network, *leakyrelu).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTestsLeakyReluEstimateOnly.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTestsLeakyReluEstimateOnlyOutput.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::Low);
    }

    // InputPart, McePart, OutputPart
    REQUIRE(graph.GetNumParts() == 3);

    // We check only the EstimateOnlyPart that we expect to be created - the Input and Output part and connections
    // between the Parts are covered by NetworkToGraphOfPartsConverterTest
    const EstimateOnlyPart* estimateOnlyPart = dynamic_cast<const EstimateOnlyPart*>(&graph.GetPart(1));
    REQUIRE(estimateOnlyPart != nullptr);
    auto plans = estimateOnlyPart->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    REQUIRE(plans[0].GetInputBuffer(PartInputSlot{ estimateOnlyPart->GetPartId(), 0 })->m_TensorShape ==
            TensorShape{ 1, 16, 16, 16 });
    REQUIRE(plans[0].GetOutputBuffer(PartOutputSlot{ estimateOnlyPart->GetPartId(), 0 })->m_TensorShape ==
            TensorShape{ 1, 16, 16, 16 });
    Op* maybeEstimateOnlyOp = plans[0].m_OpGraph.GetOp(0);
    REQUIRE(IsEstimateOnlyOp(maybeEstimateOnlyOp));
    EstimateOnlyOp* estimateOnlyOp = static_cast<EstimateOnlyOp*>(maybeEstimateOnlyOp);
    CHECK(estimateOnlyOp->m_ReasonForEstimateOnly.find("Leaky relu alpha must be less than 1 and greater than 0") !=
          std::string::npos);
}

// Manually creates a Network of Operands and Operations and converts it to a GraphOfParts using the NetworkToGraphOfPartsConverter().
// The topology is chosen to test Networks of supported Part types such as:
//      * Pooling Part (MaxPool 3x3_2_2_even/odd variations)
TEST_CASE("NetworkToGraphOfPartsConverterTest MaxPool_3X3_2_2")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{
        { { 1, 32, 32, 1 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    TensorInfo input2Info{
        { { 1, 33, 33, 1 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    constexpr PoolingInfo poolingInfo{ 3, 3, 2, 2, { 0, 1, 0, 1 }, PoolingType::MAX };
    constexpr PoolingInfo pooling2Info{ 3, 3, 2, 2, { 0, 0, 0, 0 }, PoolingType::MAX };

    const std::shared_ptr<Network> network =
        CreateNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Network topology:
    /*
       { Input2 } -> MaxPool_3x3_2_2_odd -> Output2
       { Input } -> MaxPool_3x3_2_2_even -> Output
    */

    std::shared_ptr<Operand> input       = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Operand> maxpoolEven = AddPooling(network, *input, poolingInfo).tensor;
    std::shared_ptr<Output> output       = AddOutput(network, *maxpoolEven).tensor;
    std::shared_ptr<Operand> input2      = AddInput(network, input2Info).tensor;
    std::shared_ptr<Operand> maxpoolOdd  = AddPooling(network, *input2, pooling2Info).tensor;
    std::shared_ptr<Output> output2      = AddOutput(network, *maxpoolOdd).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest MaxPool_3x3_2_2.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest MaxPool_3x3_2_2 Output.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::High);
    }

    // Check for each Part:
    //  * Whether the type of the generated Part is correct
    //  * Whether the PleOperation command stream is correct for Operations using FusedPleParts (e.g. MaxPool_3x3_2_2_even/odd ...)
    //  * The number of Input/Output slots
    //  * Whether PartInputSlots connect to PartOutputSlots of the correct Part
    //  * For the last Part, check that there are no connections to any following PartInputSlots
    REQUIRE(graph.GetNumParts() == 6);

    REQUIRE(dynamic_cast<const InputPart*>(&graph.GetPart(0)) != nullptr);
    REQUIRE(graph.GetPartInputs(0).size() == 0);
    REQUIRE(graph.GetPartOutputs(0).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 0, 0 }).has_value() == false);

    const FusedPlePart* maxpoolEvenPart = dynamic_cast<const FusedPlePart*>(&graph.GetPart(1));
    REQUIRE(maxpoolEvenPart != nullptr);
    auto maxpoolEvenPlans =
        maxpoolEvenPart->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    Op* maybePleOpMaxPoolEvenPlans = maxpoolEvenPlans[0].m_OpGraph.GetOp(2);
    REQUIRE(IsPleOp(maybePleOpMaxPoolEvenPlans));
    PleOp* pleOpMaxPoolEvenPlans = static_cast<PleOp*>(maybePleOpMaxPoolEvenPlans);
    REQUIRE(pleOpMaxPoolEvenPlans->m_Op == ethosn::command_stream::PleOperation::MAXPOOL_3X3_2_2_EVEN);
    REQUIRE(graph.GetPartInputs(1).size() == 1);
    REQUIRE(graph.GetPartOutputs(1).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 1, 0 }).value().m_PartId == 0);

    REQUIRE(dynamic_cast<const OutputPart*>(&graph.GetPart(2)) != nullptr);
    REQUIRE(graph.GetPartInputs(2).size() == 1);
    REQUIRE(graph.GetPartOutputs(2).size() == 0);
    REQUIRE(graph.GetConnectedOutputSlot({ 2, 0 }).value().m_PartId == 1);
    REQUIRE(graph.GetConnectedInputSlots({ 2, 0 }).size() == 0);

    REQUIRE(dynamic_cast<const InputPart*>(&graph.GetPart(3)) != nullptr);
    REQUIRE(graph.GetPartInputs(3).size() == 0);
    REQUIRE(graph.GetPartOutputs(3).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 3, 0 }).has_value() == false);

    const FusedPlePart* maxpoolOddPart = dynamic_cast<const FusedPlePart*>(&graph.GetPart(4));
    REQUIRE(maxpoolOddPart != nullptr);
    auto maxpoolOddPlans =
        maxpoolOddPart->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    Op* maybePleOpMaxPoolOddPlans = maxpoolOddPlans[0].m_OpGraph.GetOp(2);
    REQUIRE(IsPleOp(maybePleOpMaxPoolOddPlans));
    PleOp* pleOpMaxPoolOddPlans = static_cast<PleOp*>(maybePleOpMaxPoolOddPlans);
    REQUIRE(pleOpMaxPoolOddPlans->m_Op == ethosn::command_stream::PleOperation::MAXPOOL_3X3_2_2_ODD);
    REQUIRE(graph.GetPartInputs(4).size() == 1);
    REQUIRE(graph.GetPartOutputs(4).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 4, 0 }).value().m_PartId == 3);

    REQUIRE(dynamic_cast<const OutputPart*>(&graph.GetPart(5)) != nullptr);
    REQUIRE(graph.GetPartInputs(5).size() == 1);
    REQUIRE(graph.GetPartOutputs(5).size() == 0);
    REQUIRE(graph.GetConnectedOutputSlot({ 5, 0 }).value().m_PartId == 4);
    REQUIRE(graph.GetConnectedInputSlots({ 5, 0 }).size() == 0);
}

TEST_CASE("NetworkToGraphOfPartsConverter FullyConnected")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{
        { { 1, 1, 1, 4096 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    TensorInfo biasInfo{
        { { 1, 1, 1, 1024 } },
        DataType::INT32_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    TensorInfo weightsInfo{
        { { 1, 1, 4096, 1024 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::HWIO,
        { 0, 1.f },
    };

    FullyConnectedInfo fcInfo{
        { 0, 1.1f },
    };

    const std::vector<uint8_t> biasData(utils::TotalSizeBytes(biasInfo));
    const std::vector<uint8_t> weightsData(utils::TotalSizeBytes(weightsInfo));

    const std::shared_ptr<Network> network =
        CreateNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Network topology:
    // Input -> FullyConnected -> Output
    std::shared_ptr<Operand> input    = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Constant> bias    = AddConstant(network, biasInfo, biasData.data()).tensor;
    std::shared_ptr<Constant> weights = AddConstant(network, weightsInfo, weightsData.data()).tensor;
    std::shared_ptr<Operand> conv     = AddFullyConnected(network, *input, *bias, *weights, fcInfo).tensor;
    std::shared_ptr<Output> output    = AddOutput(network, *conv).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTests.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTests_Output.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::Low);
    }

    // InputPart, McePart, OutputPart
    REQUIRE(graph.GetNumParts() == 3);

    // McePart has a fully connected part in it
    const FullyConnectedPart* part = dynamic_cast<const FullyConnectedPart*>(&graph.GetPart(1));
    REQUIRE(part != nullptr);

    auto plans      = part->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    Op* maybeDmaOp0 = plans[0].m_OpGraph.GetOp(0);
    REQUIRE(typeid(*maybeDmaOp0) == typeid(DmaOp));
    DmaOp* dmaOp0 = static_cast<DmaOp*>(maybeDmaOp0);
    REQUIRE(dmaOp0->m_TransferFormat == CascadingBufferFormat::NHWCB);
    Op* maybeDmaOp1 = plans[0].m_OpGraph.GetOp(1);
    REQUIRE(typeid(*maybeDmaOp1) == typeid(DmaOp));
    DmaOp* dmaOp1 = static_cast<DmaOp*>(maybeDmaOp1);
    REQUIRE(dmaOp1->m_TransferFormat == CascadingBufferFormat::WEIGHT);
    Op* maybeMceOp = plans[0].m_OpGraph.GetOp(2);
    REQUIRE(IsMceOp(maybeMceOp));
    MceOp* mceOp = static_cast<MceOp*>(maybeMceOp);
    REQUIRE(mceOp->m_Op == ethosn::command_stream::MceOperation::FULLY_CONNECTED);
}

TEST_CASE("NetworkToGraphOfPartsConverter FullyConnected EstimateOnly")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    // Input to fully connected is expected to be one dimensional
    // using the channels dimension, so this will return EstimateOnly.
    TensorInfo inputInfo{
        { 1, 1, 2, 256 },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    TensorInfo biasInfo{
        { 1, 1, 1, 64 },
        DataType::INT32_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    TensorInfo weightsInfo{
        { 1, 1, 512, 64 },
        DataType::UINT8_QUANTIZED,
        DataFormat::HWIO,
        { 0, 1.f },
    };

    FullyConnectedInfo fcInfo{
        { 0, 1.1f },
    };

    const std::vector<uint8_t> biasData(utils::TotalSizeBytes(biasInfo));
    const std::vector<uint8_t> weightsData(utils::TotalSizeBytes(weightsInfo));

    const std::shared_ptr<Network> network =
        CreateEstimationNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Network topology:
    // Input -> FullyConnected -> Output
    std::shared_ptr<Operand> input    = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Constant> bias    = AddConstant(network, biasInfo, biasData.data()).tensor;
    std::shared_ptr<Constant> weights = AddConstant(network, weightsInfo, weightsData.data()).tensor;
    std::shared_ptr<Operand> conv     = AddFullyConnected(network, *input, *bias, *weights, fcInfo).tensor;
    std::shared_ptr<Output> output    = AddOutput(network, *conv).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterFullyConnectedEstimateOnlyTests.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterFullyConnectedEstimateOnlyTests_Output.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::Low);
    }

    // InputPart, McePart, OutputPart
    REQUIRE(graph.GetNumParts() == 3);

    // We check only the EstimateOnlyPart that we expect to be created - the Input and Output part and connections
    // between the Parts are covered by NetworkToGraphOfPartsConverterTest
    const EstimateOnlyPart* estimateOnlyPart = dynamic_cast<const EstimateOnlyPart*>(&graph.GetPart(1));
    REQUIRE(estimateOnlyPart != nullptr);
    auto plans = estimateOnlyPart->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    REQUIRE(plans[0].GetInputBuffer(PartInputSlot{ estimateOnlyPart->GetPartId(), 0 })->m_TensorShape ==
            TensorShape{ 1, 1, 2, 256 });
    REQUIRE(plans[0].GetOutputBuffer(PartOutputSlot{ estimateOnlyPart->GetPartId(), 0 })->m_TensorShape ==
            TensorShape{ 1, 1, 1, 64 });
    Op* maybeEstimateOnlyOp = plans[0].m_OpGraph.GetOp(0);
    REQUIRE(IsEstimateOnlyOp(maybeEstimateOnlyOp));
    EstimateOnlyOp* estimateOnlyOp = static_cast<EstimateOnlyOp*>(maybeEstimateOnlyOp);
    CHECK(estimateOnlyOp->m_ReasonForEstimateOnly.find(
              "Input to fully connected is expected to be one dimensional using the channels dimension.") !=
          std::string::npos);
}

TEST_CASE("NetworkToGraphOfPartsConverter Basic Depthwise")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{
        { { 1, 64, 64, 64 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    TensorInfo biasInfo{
        { { 1, 1, 1, 64 } },
        DataType::INT32_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    TensorInfo weightsInfo{
        { { 3, 3, 64, 1 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::HWIM,
        { 0, 1.f },
    };

    ConvolutionInfo convInfo{
        { 0, 0, 0, 0 },
        { 1, 1 },
        { 0, 1.1f },
    };

    const std::vector<uint8_t> biasData(utils::TotalSizeBytes(biasInfo));
    const std::vector<uint8_t> weightsData(utils::TotalSizeBytes(weightsInfo));

    const std::shared_ptr<Network> network =
        CreateNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Network topology:
    // Input -> Convolution -> Output
    std::shared_ptr<Operand> input    = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Constant> bias    = AddConstant(network, biasInfo, biasData.data()).tensor;
    std::shared_ptr<Constant> weights = AddConstant(network, weightsInfo, weightsData.data()).tensor;
    std::shared_ptr<Operand> conv     = AddDepthwiseConvolution(network, *input, *bias, *weights, convInfo).tensor;
    std::shared_ptr<Output> output    = AddOutput(network, *conv).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTests.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTests_Output.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::Low);
    }

    // InputPart, McePart, OutputPart
    REQUIRE(graph.GetNumParts() == 3);

    // McePart has a depthwise convolution in it
    const McePart* part = dynamic_cast<const McePart*>(&graph.GetPart(1));
    REQUIRE(part != nullptr);
    auto operation = part->GetMceOperation();
    REQUIRE(operation.has_value());
    REQUIRE(operation.value() == ethosn::command_stream::MceOperation::DEPTHWISE_CONVOLUTION);
}

TEST_CASE("NetworkToGraphOfPartsConverter Strided Depthwise")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{
        { { 1, 64, 64, 64 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    TensorInfo biasInfo{
        { { 1, 1, 1, 64 } },
        DataType::INT32_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    TensorInfo weightsInfo{
        { { 3, 3, 64, 1 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::HWIM,
        { 0, 1.f },
    };

    ConvolutionInfo convInfo{
        { 0, 0, 0, 0 },
        { 2, 2 },
        { 0, 1.1f },
    };

    const std::vector<uint8_t> biasData(utils::TotalSizeBytes(biasInfo));
    const std::vector<uint8_t> weightsData(utils::TotalSizeBytes(weightsInfo));

    const std::shared_ptr<Network> network =
        CreateNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Network topology:
    // Input -> Strided Depthwise Convolution -> Output
    std::shared_ptr<Operand> input    = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Constant> bias    = AddConstant(network, biasInfo, biasData.data()).tensor;
    std::shared_ptr<Constant> weights = AddConstant(network, weightsInfo, weightsData.data()).tensor;
    std::shared_ptr<Operand> conv     = AddDepthwiseConvolution(network, *input, *bias, *weights, convInfo).tensor;
    std::shared_ptr<Output> output    = AddOutput(network, *conv).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTests.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTests_Output.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::Low);
    }

    // InputPart, FusedPlePart, McePart, OutputPart
    REQUIRE(graph.GetNumParts() == 4);

    // McePart has a depthwise convolution in it
    const FusedPlePart* plePart = dynamic_cast<const FusedPlePart*>(&graph.GetPart(1));
    const McePart* mcePart      = dynamic_cast<const McePart*>(&graph.GetPart(2));
    REQUIRE(plePart != nullptr);
    REQUIRE(mcePart != nullptr);
    auto operation = mcePart->GetMceOperation();
    REQUIRE(operation.has_value());
    REQUIRE(operation.value() == ethosn::command_stream::MceOperation::DEPTHWISE_CONVOLUTION);

    {
        auto plans     = plePart->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
        Op* maybePleOp = plans[0].m_OpGraph.GetOp(2);
        REQUIRE(IsPleOp(maybePleOp));
        PleOp* pleOp = static_cast<PleOp*>(maybePleOp);
        REQUIRE(pleOp->m_Op == ethosn::command_stream::PleOperation::INTERLEAVE_2X2_2_2);
    }
    {
        auto plans     = mcePart->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
        Op* maybeMceOp = plans[0].m_OpGraph.GetOp(1);
        REQUIRE(IsMceOp(maybeMceOp));
        MceOp* mceOp = static_cast<MceOp*>(maybeMceOp);
        REQUIRE(mceOp->m_Stride == Stride{ 2, 2 });
    }
}

TEST_CASE("NetworkToGraphOfPartsConverter Multichannel Depthwise")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{
        { { 1, 64, 64, 1 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    TensorInfo biasInfo{
        { { 1, 1, 1, 4 } },
        DataType::INT32_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    TensorInfo weightsInfo{
        { { 3, 3, 1, 4 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::HWIM,
        { 0, 1.f },
    };

    ConvolutionInfo convInfo{
        { 0, 0, 0, 0 },
        { 1, 1 },
        { 0, 1.1f },
    };

    const std::vector<uint8_t> biasData(utils::TotalSizeBytes(biasInfo));
    const std::vector<uint8_t> weightsData(utils::TotalSizeBytes(weightsInfo));

    const std::shared_ptr<Network> network =
        CreateNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Network topology:
    // Input -> Multichannel Depthwise Convolution -> Output
    std::shared_ptr<Operand> input    = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Constant> bias    = AddConstant(network, biasInfo, biasData.data()).tensor;
    std::shared_ptr<Constant> weights = AddConstant(network, weightsInfo, weightsData.data()).tensor;
    std::shared_ptr<Operand> conv     = AddDepthwiseConvolution(network, *input, *bias, *weights, convInfo).tensor;
    std::shared_ptr<Output> output    = AddOutput(network, *conv).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTests.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTests_Output.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::Low);
    }

    // InputPart, McePart, OutputPart
    REQUIRE(graph.GetNumParts() == 3);

    // McePart has a 2D convolution in it
    const McePart* mcePart = dynamic_cast<const McePart*>(&graph.GetPart(1));
    REQUIRE(mcePart != nullptr);
    auto operation = mcePart->GetMceOperation();
    REQUIRE(operation.has_value());
    // Depthwise with channel multiplier > 1 is supported only for number of input channels = 1, which is equivalent to
    // normal convolution and should be executed as such
    REQUIRE(operation.value() == ethosn::command_stream::MceOperation::CONVOLUTION);
}

TEST_CASE("NetworkToGraphOfPartsConverter Depthwise EstimateOnly")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{ { 1, 1, 1, 3 }, DataType::UINT8_QUANTIZED, DataFormat::NHWCB, { 0, 1.f } };

    TensorInfo weightsInfo{ { 1, 1, 3, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM };

    // Bias for depthwise conv must have quantization parameters with zero point of 0 and
    // scale of input scale x weight scale, so this will return EstimateOnly.
    TensorInfo biasInfo{ { 1, 1, 1, 3 }, DataType::INT32_QUANTIZED, DataFormat::NHWC };
    biasInfo.m_QuantizationInfo.SetScales(QuantizationScales{ 0.1f, 0.2f, 0.3f });
    biasInfo.m_QuantizationInfo.SetZeroPoint(0);
    biasInfo.m_QuantizationInfo.SetQuantizationDim(3);

    ConvolutionInfo convInfo{ { 0, 0, 0, 0 }, { 1, 1 } };

    const std::vector<uint8_t> biasData(utils::TotalSizeBytes(biasInfo));
    const std::vector<uint8_t> weightsData(utils::TotalSizeBytes(weightsInfo));

    const std::shared_ptr<Network> network =
        CreateEstimationNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Network topology:
    // Input -> Convolution -> Output
    std::shared_ptr<Operand> input    = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Constant> bias    = AddConstant(network, biasInfo, biasData.data()).tensor;
    std::shared_ptr<Constant> weights = AddConstant(network, weightsInfo, weightsData.data()).tensor;
    std::shared_ptr<Operand> conv     = AddDepthwiseConvolution(network, *input, *bias, *weights, convInfo).tensor;
    std::shared_ptr<Output> output    = AddOutput(network, *conv).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTests EstimateOnly.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTests_Output.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::Low);
    }

    // InputPart, McePart, OutputPart
    REQUIRE(graph.GetNumParts() == 3);

    // We check only the EstimateOnlyPart that we expect to be created - the Input and Output part and connections
    // between the Parts are covered by NetworkToGraphOfPartsConverterTest
    const EstimateOnlyPart* estimateOnlyPart = dynamic_cast<const EstimateOnlyPart*>(&graph.GetPart(1));
    REQUIRE(estimateOnlyPart != nullptr);
    auto plans = estimateOnlyPart->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    REQUIRE(plans[0].GetInputBuffer(PartInputSlot{ estimateOnlyPart->GetPartId(), 0 })->m_TensorShape ==
            TensorShape{ 1, 1, 1, 3 });
    REQUIRE(plans[0].GetOutputBuffer(PartOutputSlot{ estimateOnlyPart->GetPartId(), 0 })->m_TensorShape ==
            TensorShape{ 1, 1, 1, 3 });
    Op* maybeEstimateOnlyOp = plans[0].m_OpGraph.GetOp(0);
    REQUIRE(IsEstimateOnlyOp(maybeEstimateOnlyOp));
    EstimateOnlyOp* estimateOnlyOp = static_cast<EstimateOnlyOp*>(maybeEstimateOnlyOp);
    CHECK(estimateOnlyOp->m_ReasonForEstimateOnly.find("Bias for depthwise conv must have quantization parameters with "
                                                       "zero point of 0 and scale of input scale x weight scale") !=
          std::string::npos);
}

TEST_CASE("NetworkToGraphOfPartsConverterTest AVGPOOL_3X3_1_1_UDMA")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{
        { { 1, 16, 16, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    const PoolingInfo poolingInfo = PoolingInfo{ 3, 3, 1, 1, { 1, 1, 1, 1 }, PoolingType::AVG };

    const std::shared_ptr<Network> network =
        CreateNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    std::shared_ptr<Operand> input   = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Operand> avgPool = AddPooling(network, *input, poolingInfo).tensor;
    std::shared_ptr<Output> output   = AddOutput(network, *avgPool).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest AVGPOOL_3X3_1_1_UDMA.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest AVGPOOL_3X3_1_1_UDMA Output.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::High);
    }

    // Check for each Part:
    //  * Whether the type of the generated Part is correct
    //  * Whether the PleOperation command stream is correct for Operations using StandalonePlePart
    //  * The number of Input/Output slots
    //  * Whether PartInputSlots connect to PartOutputSlots of the correct Part
    //  * For the last Part, check that there are no connections to any following PartInputSlots
    REQUIRE(graph.GetNumParts() == 3);

    REQUIRE(dynamic_cast<const InputPart*>(&graph.GetPart(0)) != nullptr);
    REQUIRE(graph.GetPartInputs(0).size() == 0);
    REQUIRE(graph.GetPartOutputs(0).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 0, 0 }).has_value() == false);

    const StandalonePlePart* avePoolPlePart = dynamic_cast<const StandalonePlePart*>(&graph.GetPart(1));
    REQUIRE(avePoolPlePart != nullptr);
    auto avePoolPlans =
        avePoolPlePart->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    Op* maybePleOpAvePoolPlans = avePoolPlans[0].m_OpGraph.GetOp(0);
    REQUIRE(IsPleOp(maybePleOpAvePoolPlans));
    PleOp* pleOpAvePoolPlans = static_cast<PleOp*>(maybePleOpAvePoolPlans);
    REQUIRE(pleOpAvePoolPlans->m_Op == ethosn::command_stream::PleOperation::AVGPOOL_3X3_1_1_UDMA);
    REQUIRE(graph.GetPartInputs(1).size() == 1);
    REQUIRE(graph.GetPartOutputs(1).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 1, 0 }).value().m_PartId == 0);

    REQUIRE(dynamic_cast<const OutputPart*>(&graph.GetPart(2)) != nullptr);
    REQUIRE(graph.GetPartInputs(2).size() == 1);
    REQUIRE(graph.GetPartOutputs(2).size() == 0);
    REQUIRE(graph.GetConnectedOutputSlot({ 2, 0 }).value().m_PartId == 1);
    REQUIRE(graph.GetConnectedInputSlots({ 2, 0 }).size() == 0);
}

TEST_CASE("NetworkToGraphOfPartsConverterTest AVGPOOL_3X3_1_1_UDMA EstimateOnly")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{
        { 1, 16, 16, 16 },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    // The poolingSizeY must be 3 here for AVG pooling, so this will return EstimateOnly.
    const PoolingInfo poolingInfo = PoolingInfo{ 3, 2, 1, 1, { 1, 1, 1, 1 }, PoolingType::AVG };

    const std::shared_ptr<Network> network =
        CreateEstimationNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    std::shared_ptr<Operand> input   = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Operand> avgPool = AddPooling(network, *input, poolingInfo).tensor;
    std::shared_ptr<Output> output   = AddOutput(network, *avgPool).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest AVGPOOL_3X3_1_1_UDMA.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest AVGPOOL_3X3_1_1_UDMA Output.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::High);
    }

    // InputPart, McePart, OutputPart
    REQUIRE(graph.GetNumParts() == 3);

    // We check only the EstimateOnlyPart that we expect to be created - the Input and Output part and connections
    // between the Parts are covered by NetworkToGraphOfPartsConverterTest
    const EstimateOnlyPart* estimateOnlyPart = dynamic_cast<const EstimateOnlyPart*>(&graph.GetPart(1));
    REQUIRE(estimateOnlyPart != nullptr);
    auto plans = estimateOnlyPart->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    REQUIRE(plans[0].GetInputBuffer(PartInputSlot{ estimateOnlyPart->GetPartId(), 0 })->m_TensorShape ==
            TensorShape{ 1, 16, 16, 16 });
    REQUIRE(plans[0].GetOutputBuffer(PartOutputSlot{ estimateOnlyPart->GetPartId(), 0 })->m_TensorShape ==
            TensorShape{ 1, 17, 16, 16 });
    Op* maybeEstimateOnlyOp = plans[0].m_OpGraph.GetOp(0);
    REQUIRE(IsEstimateOnlyOp(maybeEstimateOnlyOp));
    EstimateOnlyOp* estimateOnlyOp = static_cast<EstimateOnlyOp*>(maybeEstimateOnlyOp);
    CHECK(estimateOnlyOp->m_ReasonForEstimateOnly.find("Unsupported configuration in AVG pooling") !=
          std::string::npos);
}

TEST_CASE("NetworkToGraphOfPartsConverterTest ADDITION")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo1{
        { { 1, 16, 16, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    TensorInfo inputInfo2{
        { { 1, 16, 16, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    const std::shared_ptr<Network> network =
        CreateNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    std::shared_ptr<Operand> input1   = AddInput(network, inputInfo1).tensor;
    std::shared_ptr<Operand> input2   = AddInput(network, inputInfo2).tensor;
    std::shared_ptr<Operand> addition = AddAddition(network, *input1, *input2, QuantizationInfo(0, 1.f)).tensor;
    std::shared_ptr<Output> output    = AddOutput(network, *addition).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest ADDITION.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest ADDITION.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::High);
    }

    // Check for each Part:
    //  * Whether the type of the generated Part is correct
    //  * Whether the PleOperation command stream is correct for Operations using StandalonePlePart
    //  * The number of Input/Output slots
    //  * Whether PartInputSlots connect to PartOutputSlots of the correct Part
    //  * For the last Part, check that there are no connections to any following PartInputSlots
    REQUIRE(graph.GetNumParts() == 4);

    REQUIRE(dynamic_cast<const InputPart*>(&graph.GetPart(0)) != nullptr);
    REQUIRE(graph.GetPartInputs(0).size() == 0);
    REQUIRE(graph.GetPartOutputs(0).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 0, 0 }).has_value() == false);

    REQUIRE(dynamic_cast<const InputPart*>(&graph.GetPart(1)) != nullptr);
    REQUIRE(graph.GetPartInputs(1).size() == 0);
    REQUIRE(graph.GetPartOutputs(1).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 1, 0 }).has_value() == false);

    const StandalonePlePart* additionPlePart = dynamic_cast<const StandalonePlePart*>(&graph.GetPart(2));
    REQUIRE(additionPlePart != nullptr);
    auto additionPlans =
        additionPlePart->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    Op* maybePleOpAdditionPlans = additionPlans[0].m_OpGraph.GetOp(0);
    REQUIRE(IsPleOp(maybePleOpAdditionPlans));
    PleOp* pleOpAdditionPlans = static_cast<PleOp*>(maybePleOpAdditionPlans);
    REQUIRE(pleOpAdditionPlans->m_Op == ethosn::command_stream::PleOperation::ADDITION);
    REQUIRE(graph.GetPartInputs(2).size() == 2);
    REQUIRE(graph.GetPartOutputs(2).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 2, 0 }).value().m_PartId == 0);
    REQUIRE(graph.GetConnectedOutputSlot({ 2, 1 }).value().m_PartId == 1);

    REQUIRE(dynamic_cast<const OutputPart*>(&graph.GetPart(3)) != nullptr);
    REQUIRE(graph.GetPartInputs(3).size() == 1);
    REQUIRE(graph.GetPartOutputs(3).size() == 0);
    REQUIRE(graph.GetConnectedOutputSlot({ 3, 0 }).value().m_PartId == 2);
    REQUIRE(graph.GetConnectedInputSlots({ 3, 0 }).size() == 0);
}

TEST_CASE("NetworkToGraphOfPartsConverterTest ADDITION_RESCALE")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo1{
        { { 1, 16, 16, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    TensorInfo inputInfo2{
        { { 1, 16, 16, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    const std::shared_ptr<Network> network =
        CreateNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    std::shared_ptr<Operand> input1   = AddInput(network, inputInfo1).tensor;
    std::shared_ptr<Operand> input2   = AddInput(network, inputInfo2).tensor;
    std::shared_ptr<Operand> addition = AddAddition(network, *input1, *input2, QuantizationInfo(0, 1.1f)).tensor;
    std::shared_ptr<Output> output    = AddOutput(network, *addition).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest ADDITION_RESCALE.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest ADDITION_RESCALE.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::High);
    }

    // Check for each Part:
    //  * Whether the type of the generated Part is correct
    //  * Whether the PleOperation command stream is correct for Operations using StandalonePlePart
    //  * The number of Input/Output slots
    //  * Whether PartInputSlots connect to PartOutputSlots of the correct Part
    //  * For the last Part, check that there are no connections to any following PartInputSlots
    REQUIRE(graph.GetNumParts() == 4);

    REQUIRE(dynamic_cast<const InputPart*>(&graph.GetPart(0)) != nullptr);
    REQUIRE(graph.GetPartInputs(0).size() == 0);
    REQUIRE(graph.GetPartOutputs(0).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 0, 0 }).has_value() == false);

    REQUIRE(dynamic_cast<const InputPart*>(&graph.GetPart(1)) != nullptr);
    REQUIRE(graph.GetPartInputs(1).size() == 0);
    REQUIRE(graph.GetPartOutputs(1).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 1, 0 }).has_value() == false);

    const StandalonePlePart* additionPlePart = dynamic_cast<const StandalonePlePart*>(&graph.GetPart(2));
    REQUIRE(additionPlePart != nullptr);
    auto additionPlans =
        additionPlePart->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    Op* maybePleOpAdditionPlans = additionPlans[0].m_OpGraph.GetOp(0);
    REQUIRE(IsPleOp(maybePleOpAdditionPlans));
    PleOp* pleOpAdditionPlans = static_cast<PleOp*>(maybePleOpAdditionPlans);
    REQUIRE(pleOpAdditionPlans->m_Op == ethosn::command_stream::PleOperation::ADDITION_RESCALE);
    REQUIRE(graph.GetPartInputs(2).size() == 2);
    REQUIRE(graph.GetPartOutputs(2).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 2, 0 }).value().m_PartId == 0);
    REQUIRE(graph.GetConnectedOutputSlot({ 2, 1 }).value().m_PartId == 1);

    REQUIRE(dynamic_cast<const OutputPart*>(&graph.GetPart(3)) != nullptr);
    REQUIRE(graph.GetPartInputs(3).size() == 1);
    REQUIRE(graph.GetPartOutputs(3).size() == 0);
    REQUIRE(graph.GetConnectedOutputSlot({ 3, 0 }).value().m_PartId == 2);
    REQUIRE(graph.GetConnectedInputSlots({ 3, 0 }).size() == 0);
}

TEST_CASE("NetworkToGraphOfPartsConverterTest ADDITION EstimateOnly")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo1{
        { 1, 16, 16, 16 },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    // Stretching of dimensions isn't supported,
    // so stretcing the channels will return EstimateOnly.
    TensorInfo inputInfo2{
        { 1, 16, 16, 1 },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    const std::shared_ptr<Network> network =
        CreateEstimationNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    std::shared_ptr<Operand> input1   = AddInput(network, inputInfo1).tensor;
    std::shared_ptr<Operand> input2   = AddInput(network, inputInfo2).tensor;
    std::shared_ptr<Operand> addition = AddAddition(network, *input1, *input2, QuantizationInfo(0, 1.f)).tensor;
    std::shared_ptr<Output> output    = AddOutput(network, *addition).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest ADDITION EstimateOnly.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest ADDITION EstimateOnly.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::High);
    }

    // InputPart1, InputPart2, McePart, OutputPart
    REQUIRE(graph.GetNumParts() == 4);

    // We check only the EstimateOnlyPart that we expect to be created - the Input and Output part and connections
    // between the Parts are covered by NetworkToGraphOfPartsConverterTest
    const EstimateOnlyPart* estimateOnlyPart = dynamic_cast<const EstimateOnlyPart*>(&graph.GetPart(2));
    REQUIRE(estimateOnlyPart != nullptr);
    auto plans = estimateOnlyPart->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    REQUIRE(plans[0].GetInputBuffer(PartInputSlot{ estimateOnlyPart->GetPartId(), 0 })->m_TensorShape ==
            TensorShape{ 1, 16, 16, 16 });
    REQUIRE(plans[0].GetOutputBuffer(PartOutputSlot{ estimateOnlyPart->GetPartId(), 0 })->m_TensorShape ==
            TensorShape{ 1, 16, 16, 16 });
    Op* maybeEstimateOnlyOp = plans[0].m_OpGraph.GetOp(0);
    REQUIRE(IsEstimateOnlyOp(maybeEstimateOnlyOp));
    EstimateOnlyOp* estimateOnlyOp = static_cast<EstimateOnlyOp*>(maybeEstimateOnlyOp);
    CHECK(estimateOnlyOp->m_ReasonForEstimateOnly.find("Cannot stretch along the requested dimensions.") !=
          std::string::npos);
}

// Manually creates a Network of Operands and Operations and converts it to a GraphOfParts using the NetworkToGraphOfPartsConverter.
// The topology is chosen to test that the Resize operation is correctly converted to an McePart.
TEST_CASE("NetworkToGraphOfPartsConverter Resize")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{
        { { 1, 16, 16, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    const std::shared_ptr<Network> network =
        CreateNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));
    std::shared_ptr<Operand> input = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Operand> resize =
        AddResize(network, *input, ResizeInfo(ResizeAlgorithm::BILINEAR, 32, 32, QuantizationInfo(0, 1.0f))).tensor;
    std::shared_ptr<Output> output = AddOutput(network, *resize).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest Resize.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest Resize Output.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::High);
    }

    // InputPart, McePart, OutputPart
    REQUIRE(graph.GetNumParts() == 3);

    // We check only the McePart that we expect to be created - the Input and Output part and connections
    // between the Parts are covered by NetworkToGraphOfPartsConverterTest
    const McePart* mcePart = dynamic_cast<const McePart*>(&graph.GetPart(1));
    REQUIRE(mcePart != nullptr);
    auto plans     = mcePart->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    Op* maybeMceOp = plans[0].m_OpGraph.GetOp(1);
    REQUIRE(IsMceOp(maybeMceOp));
    MceOp* mceOp = static_cast<MceOp*>(maybeMceOp);
    CHECK(mceOp->m_UpscaleFactor == 2);
    CHECK(mceOp->m_UpsampleType == MceUpsampleType::BILINEAR);
}

TEST_CASE("NetworkToGraphOfPartsConverter Relu")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{
        { { 1, 16, 16, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    ReluInfo reluInfo(100, 200);

    const std::shared_ptr<Network> network =
        CreateNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Network topology:
    // Input -> Relu -> Output
    std::shared_ptr<Operand> input = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Operand> relu  = AddRelu(network, *input, reluInfo).tensor;
    std::shared_ptr<Output> output = AddOutput(network, *relu).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTestsRelu.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTests_ReluOutput.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::Low);
    }

    // InputPart, McePart, OutputPart
    REQUIRE(graph.GetNumParts() == 3);

    const McePart* part = dynamic_cast<const McePart*>(&graph.GetPart(1));
    REQUIRE(part != nullptr);

    auto plans     = part->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    Op* maybeMceOp = plans[0].m_OpGraph.GetOp(1);
    REQUIRE(IsMceOp(maybeMceOp));
    MceOp* mceOp = static_cast<MceOp*>(maybeMceOp);
    // Ensure the lower and upper bound on the mce op is correct.
    REQUIRE(mceOp->m_LowerBound == 100);
    REQUIRE(mceOp->m_UpperBound == 200);
}

TEST_CASE("NetworkToGraphOfPartsConverter Conv Relu")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{
        { { 1, 16, 16, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    TensorInfo biasInfo{
        { { 1, 1, 1, 16 } },
        DataType::INT32_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    TensorInfo weightsInfo{
        { { 1, 1, 16, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::HWIO,
        { 0, 1.f },
    };

    ConvolutionInfo convInfo{
        { 0, 0, 0, 0 },
        { 1, 1 },
        { 0, 1.1f },
    };

    const std::vector<uint8_t> biasData(utils::TotalSizeBytes(biasInfo));
    const std::vector<uint8_t> weightsData(utils::TotalSizeBytes(weightsInfo));

    ReluInfo reluInfo(100, 200);

    const std::shared_ptr<Network> network =
        CreateNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Network topology:
    // Input -> Conv -> Relu -> Output
    std::shared_ptr<Operand> input    = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Constant> bias    = AddConstant(network, biasInfo, biasData.data()).tensor;
    std::shared_ptr<Constant> weights = AddConstant(network, weightsInfo, weightsData.data()).tensor;
    std::shared_ptr<Operand> conv     = AddConvolution(network, *input, *bias, *weights, convInfo).tensor;
    auto relu                         = AddRelu(network, *conv, reluInfo);
    std::shared_ptr<Output> output    = AddOutput(network, *relu.tensor).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTestsConvRelu.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTests_ConvReluOutput.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::Low);
    }

    // InputPart, McePart, OutputPart
    REQUIRE(graph.GetNumParts() == 3);

    const McePart* part = dynamic_cast<const McePart*>(&graph.GetPart(1));
    REQUIRE(part != nullptr);

    auto plans     = part->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    Op* maybeMceOp = plans[0].m_OpGraph.GetOp(1);
    REQUIRE(IsMceOp(maybeMceOp));
    MceOp* mceOp = static_cast<MceOp*>(maybeMceOp);
    // Ensure the lower and upper bound on the mce op is correct.
    REQUIRE(mceOp->m_LowerBound == 100);
    REQUIRE(mceOp->m_UpperBound == 200);
    REQUIRE(mceOp->m_OperationIds.count(relu.operationId) == 1);
}

// Checks that two consecutive relus get merged together into the preceding conv.
TEST_CASE("NetworkToGraphOfPartsConverter Conv Relu Relu")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{
        { { 1, 16, 16, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    TensorInfo biasInfo{
        { { 1, 1, 1, 16 } },
        DataType::INT32_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    TensorInfo weightsInfo{
        { { 1, 1, 16, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::HWIO,
        { 0, 1.f },
    };

    ConvolutionInfo convInfo{
        { 0, 0, 0, 0 },
        { 1, 1 },
        { 0, 1.1f },
    };

    const std::vector<uint8_t> biasData(utils::TotalSizeBytes(biasInfo));
    const std::vector<uint8_t> weightsData(utils::TotalSizeBytes(weightsInfo));

    ReluInfo reluInfo1(100, 200);
    ReluInfo reluInfo2(150, 220);

    const std::shared_ptr<Network> network =
        CreateNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Network topology:
    // Input -> Conv -> Relu -> Relu -> Output
    std::shared_ptr<Operand> input    = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Constant> bias    = AddConstant(network, biasInfo, biasData.data()).tensor;
    std::shared_ptr<Constant> weights = AddConstant(network, weightsInfo, weightsData.data()).tensor;
    std::shared_ptr<Operand> conv     = AddConvolution(network, *input, *bias, *weights, convInfo).tensor;
    auto relu1                        = AddRelu(network, *conv, reluInfo1);
    auto relu2                        = AddRelu(network, *relu1.tensor, reluInfo2);
    std::shared_ptr<Output> output    = AddOutput(network, *relu2.tensor).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTestsConvReluRelu.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTests_ConvReluReluOutput.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::Low);
    }

    // InputPart, McePart, OutputPart
    REQUIRE(graph.GetNumParts() == 3);

    const McePart* part = dynamic_cast<const McePart*>(&graph.GetPart(1));
    REQUIRE(part != nullptr);

    auto plans     = part->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    Op* maybeMceOp = plans[0].m_OpGraph.GetOp(1);
    REQUIRE(IsMceOp(maybeMceOp));
    MceOp* mceOp = static_cast<MceOp*>(maybeMceOp);
    // Ensure the lower and upper bound on the mce op is correct.
    REQUIRE(mceOp->m_LowerBound == 150);
    REQUIRE(mceOp->m_UpperBound == 200);
    REQUIRE(mceOp->m_OperationIds.count(relu1.operationId) == 1);
    REQUIRE(mceOp->m_OperationIds.count(relu2.operationId) == 1);
}

// Checks that a relu isn't merged into the preceding conv, if the conv has another consumer
TEST_CASE("NetworkToGraphOfPartsConverter Conv Relu Branch")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{
        { { 1, 16, 16, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    TensorInfo biasInfo{
        { { 1, 1, 1, 16 } },
        DataType::INT32_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    TensorInfo weightsInfo{
        { { 1, 1, 16, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::HWIO,
        { 0, 1.f },
    };

    ConvolutionInfo convInfo{
        { 0, 0, 0, 0 },
        { 1, 1 },
        { 0, 1.1f },
    };

    const std::vector<uint8_t> biasData(utils::TotalSizeBytes(biasInfo));
    const std::vector<uint8_t> weightsData(utils::TotalSizeBytes(weightsInfo));

    ReluInfo reluInfo1(100, 200);

    const std::shared_ptr<Network> network =
        CreateNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Network topology:
    // Input -> Conv -> Relu -> Output
    //              \-> Output
    std::shared_ptr<Operand> input    = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Constant> bias    = AddConstant(network, biasInfo, biasData.data()).tensor;
    std::shared_ptr<Constant> weights = AddConstant(network, weightsInfo, weightsData.data()).tensor;
    std::shared_ptr<Operand> conv     = AddConvolution(network, *input, *bias, *weights, convInfo).tensor;
    auto relu1                        = AddRelu(network, *conv, reluInfo1);
    std::shared_ptr<Output> output1   = AddOutput(network, *conv).tensor;
    std::shared_ptr<Output> output2   = AddOutput(network, *relu1.tensor).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTestsConvReluBranch.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTests_ConvReluBranchOutput.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::Low);
    }

    // InputPart, McePart, OutputPart, McePart, OutputPart
    REQUIRE(graph.GetNumParts() == 5);

    {
        const McePart* part = dynamic_cast<const McePart*>(&graph.GetPart(1));
        REQUIRE(part != nullptr);

        auto plans     = part->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
        Op* maybeMceOp = plans[0].m_OpGraph.GetOp(1);
        REQUIRE(IsMceOp(maybeMceOp));
        MceOp* mceOp = static_cast<MceOp*>(maybeMceOp);
        // Ensure the lower and upper bound on the mce op is correct.
        REQUIRE(mceOp->m_LowerBound == 0);
        REQUIRE(mceOp->m_UpperBound == 255);
        REQUIRE(mceOp->m_OperationIds.count(relu1.operationId) == 0);
    }

    {
        const McePart* part = dynamic_cast<const McePart*>(&graph.GetPart(3));
        REQUIRE(part != nullptr);

        auto plans     = part->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
        Op* maybeMceOp = plans[0].m_OpGraph.GetOp(1);
        REQUIRE(IsMceOp(maybeMceOp));
        MceOp* mceOp = static_cast<MceOp*>(maybeMceOp);
        // Ensure the lower and upper bound on the mce op is correct.
        REQUIRE(mceOp->m_LowerBound == 100);
        REQUIRE(mceOp->m_UpperBound == 200);
        REQUIRE(mceOp->m_OperationIds.count(relu1.operationId) == 1);
    }
}

TEST_CASE("NetworkToGraphOfPartsConverter Relu Conv")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{
        { { 1, 16, 16, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    TensorInfo biasInfo{
        { { 1, 1, 1, 16 } },
        DataType::INT32_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    TensorInfo weightsInfo{
        { { 1, 1, 16, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::HWIO,
        { 0, 1.f },
    };

    ConvolutionInfo convInfo{
        { 0, 0, 0, 0 },
        { 1, 1 },
        { 0, 1.1f },
    };

    const std::vector<uint8_t> biasData(utils::TotalSizeBytes(biasInfo));
    const std::vector<uint8_t> weightsData(utils::TotalSizeBytes(weightsInfo));

    ReluInfo reluInfo(100, 200);

    const std::shared_ptr<Network> network =
        CreateNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Network topology:
    // Input -> Relu -> Conv -> Output
    std::shared_ptr<Operand> input    = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Operand> relu     = AddRelu(network, *input, reluInfo).tensor;
    std::shared_ptr<Constant> bias    = AddConstant(network, biasInfo, biasData.data()).tensor;
    std::shared_ptr<Constant> weights = AddConstant(network, weightsInfo, weightsData.data()).tensor;
    std::shared_ptr<Operand> conv     = AddConvolution(network, *relu, *bias, *weights, convInfo).tensor;
    std::shared_ptr<Output> output    = AddOutput(network, *conv).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTestsConvRelu.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTests_ConvReluOutput.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::Low);
    }

    // InputPart, McePart, McePart, OutputPart
    REQUIRE(graph.GetNumParts() == 4);
    {
        const McePart* part = dynamic_cast<const McePart*>(&graph.GetPart(1));
        REQUIRE(part != nullptr);

        auto plans     = part->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
        Op* maybeMceOp = plans[0].m_OpGraph.GetOp(1);
        REQUIRE(IsMceOp(maybeMceOp));
        MceOp* mceOp = static_cast<MceOp*>(maybeMceOp);
        // Ensure the lower and upper bound on the identity mce op for the relu is correct.
        REQUIRE(mceOp->m_LowerBound == 100);
        REQUIRE(mceOp->m_UpperBound == 200);
    }

    {
        const McePart* part = dynamic_cast<const McePart*>(&graph.GetPart(2));
        REQUIRE(part != nullptr);

        auto plans     = part->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
        Op* maybeMceOp = plans[0].m_OpGraph.GetOp(1);
        REQUIRE(IsMceOp(maybeMceOp));
        MceOp* mceOp = static_cast<MceOp*>(maybeMceOp);
        // Ensure the lower and upper bound on convolution hasn't changed.
        REQUIRE(mceOp->m_LowerBound == 0);
        REQUIRE(mceOp->m_UpperBound == 255);
    }
}

TEST_CASE("NetworkToGraphOfPartsConverter Const as Input EstimateOnly")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{ { 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, { 0, 1.f } };

    TensorInfo biasInfo{ { 1, 1, 1, 16 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, 1.f } };

    TensorInfo weightsInfo{ { 1, 1, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, { 0, 1.f } };

    // Stride X and Y must be equal and in { 1, 2 }, so this will return EstimateOnly.
    ConvolutionInfo convInfo{ { 0, 0, 0, 0 }, { 1, 2 }, { 0, 1.1f } };

    const std::vector<uint8_t> inputData(utils::TotalSizeBytes(inputInfo));
    const std::vector<uint8_t> biasData(utils::TotalSizeBytes(biasInfo));
    const std::vector<uint8_t> weightsData(utils::TotalSizeBytes(weightsInfo));

    const std::shared_ptr<Network> network =
        CreateEstimationNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Network topology:
    // Const -> Conv -> Output
    std::shared_ptr<Constant> inputC  = AddConstant(network, inputInfo, inputData.data()).tensor;
    std::shared_ptr<Constant> bias    = AddConstant(network, biasInfo, biasData.data()).tensor;
    std::shared_ptr<Constant> weights = AddConstant(network, weightsInfo, weightsData.data()).tensor;
    std::shared_ptr<Operand> conv     = AddConvolution(network, *GetOperand(inputC), *bias, *weights, convInfo).tensor;
    std::shared_ptr<Output> output    = AddOutput(network, *conv).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTestsConvEstimateOnly.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTests_ConvEstimateOnlyOutput.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::Low);
    }

    // ConstPart, EstimateOnlyPart, OutputPart
    REQUIRE(graph.GetNumParts() == 3);
    //Confirm that constant is indeed the first part and in place of input
    const ConstantPart* constPart = dynamic_cast<const ConstantPart*>(&graph.GetPart(0));
    REQUIRE(constPart != nullptr);
    // Check the EstimateOnlyPart and that it's created properly
    const EstimateOnlyPart* estimateOnlyPart = dynamic_cast<const EstimateOnlyPart*>(&graph.GetPart(1));
    REQUIRE(estimateOnlyPart != nullptr);
    auto plans = estimateOnlyPart->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    REQUIRE(plans[0].GetInputBuffer(PartInputSlot{ estimateOnlyPart->GetPartId(), 0 })->m_TensorShape ==
            TensorShape{ 1, 16, 16, 16 });
    REQUIRE(plans[0].GetOutputBuffer(PartOutputSlot{ estimateOnlyPart->GetPartId(), 0 })->m_TensorShape ==
            TensorShape{ 1, 8, 16, 16 });

    Op* maybeEstimateOnlyOp = plans[0].m_OpGraph.GetOp(0);
    REQUIRE(IsEstimateOnlyOp(maybeEstimateOnlyOp));
    EstimateOnlyOp* estimateOnlyOp = static_cast<EstimateOnlyOp*>(maybeEstimateOnlyOp);
    CHECK(estimateOnlyOp->m_ReasonForEstimateOnly.find(
              "Unsupported stride. Stride X and Y must be equal and in { 1, 2 }") != std::string::npos);
}

TEST_CASE("NetworkToGraphOfPartsConverter Conv EstimateOnly")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{ { 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, { 0, 1.f } };

    TensorInfo biasInfo{ { 1, 1, 1, 16 }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, 1.f } };

    TensorInfo weightsInfo{ { 1, 1, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, { 0, 1.f } };

    // Stride X and Y must be equal and in { 1, 2 }, so this will return EstimateOnly.
    ConvolutionInfo convInfo{ { 0, 0, 0, 0 }, { 1, 2 }, { 0, 1.1f } };

    const std::vector<uint8_t> biasData(utils::TotalSizeBytes(biasInfo));
    const std::vector<uint8_t> weightsData(utils::TotalSizeBytes(weightsInfo));

    const std::shared_ptr<Network> network =
        CreateEstimationNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Network topology:
    // Input -> Conv -> Output
    std::shared_ptr<Operand> input    = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Constant> bias    = AddConstant(network, biasInfo, biasData.data()).tensor;
    std::shared_ptr<Constant> weights = AddConstant(network, weightsInfo, weightsData.data()).tensor;
    std::shared_ptr<Operand> conv     = AddConvolution(network, *input, *bias, *weights, convInfo).tensor;
    std::shared_ptr<Output> output    = AddOutput(network, *conv).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTestsConvEstimateOnly.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTests_ConvEstimateOnlyOutput.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::Low);
    }

    // InputPart, McePart, OutputPart
    REQUIRE(graph.GetNumParts() == 3);

    // We check only the EstimateOnlyPart that we expect to be created - the Input and Output part and connections
    // between the Parts are covered by NetworkToGraphOfPartsConverterTest
    const EstimateOnlyPart* estimateOnlyPart = dynamic_cast<const EstimateOnlyPart*>(&graph.GetPart(1));
    REQUIRE(estimateOnlyPart != nullptr);
    auto plans = estimateOnlyPart->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    REQUIRE(plans[0].GetInputBuffer(PartInputSlot{ estimateOnlyPart->GetPartId(), 0 })->m_TensorShape ==
            TensorShape{ 1, 16, 16, 16 });
    REQUIRE(plans[0].GetOutputBuffer(PartOutputSlot{ estimateOnlyPart->GetPartId(), 0 })->m_TensorShape ==
            TensorShape{ 1, 8, 16, 16 });
    Op* maybeEstimateOnlyOp = plans[0].m_OpGraph.GetOp(0);
    REQUIRE(IsEstimateOnlyOp(maybeEstimateOnlyOp));
    EstimateOnlyOp* estimateOnlyOp = static_cast<EstimateOnlyOp*>(maybeEstimateOnlyOp);
    CHECK(estimateOnlyOp->m_ReasonForEstimateOnly.find(
              "Unsupported stride. Stride X and Y must be equal and in { 1, 2 }") != std::string::npos);
}

// Manually creates a Network of Operands and Operations and converts it to a GraphOfParts using the NetworkToGraphOfPartsConverter.
// The topology is chosen to test that the TransposeConvolution operation with a small kernel is correctly
// converted to an McePart using upscale.
TEST_CASE("NetworkToGraphOfPartsConverter TransposeConvolution")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{ { { 1, 16, 16, 16 } }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, { 0, 1.f } };
    TensorInfo biasInfo{ { { 1, 1, 1, 4 } }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, 1.f } };
    TensorInfo weightsInfo{ { { 3, 3, 16, 4 } }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, { 0, 1.f } };
    ConvolutionInfo convInfo{ { 0, 0, 0, 0 }, { 2, 2 }, { 0, 1.1f } };

    const std::vector<uint8_t> biasData(utils::TotalSizeBytes(biasInfo));
    const std::vector<uint8_t> weightsData(utils::TotalSizeBytes(weightsInfo));

    const std::shared_ptr<Network> network =
        CreateNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));
    std::shared_ptr<Operand> input    = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Constant> bias    = AddConstant(network, biasInfo, biasData.data()).tensor;
    std::shared_ptr<Constant> weights = AddConstant(network, weightsInfo, weightsData.data()).tensor;
    std::shared_ptr<Operand> tconv    = AddTransposeConvolution(network, *input, *bias, *weights, convInfo).tensor;
    std::shared_ptr<Output> output    = AddOutput(network, *tconv).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest TransposeConvolution.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest TransposeConvolution Output.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::High);
    }

    // InputPart, McePart, OutputPart
    REQUIRE(graph.GetNumParts() == 3);

    // We check only the McePart that we expect to be created - the Input and Output part and connections
    // between the Parts are covered by NetworkToGraphOfPartsConverterTest
    const McePart* mcePart = dynamic_cast<const McePart*>(&graph.GetPart(1));
    REQUIRE(mcePart != nullptr);
    auto plans     = mcePart->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    Op* maybeMceOp = plans[0].m_OpGraph.GetOp(1);
    REQUIRE(IsMceOp(maybeMceOp));
    MceOp* mceOp = static_cast<MceOp*>(maybeMceOp);
    CHECK(mceOp->m_UpscaleFactor == 2);
    CHECK(mceOp->m_UpsampleType == MceUpsampleType::TRANSPOSE);
    CHECK(mceOp->m_PadTop == 2);
    CHECK(mceOp->m_PadLeft == 2);
    CHECK(mceOp->m_Stride == Stride{ 1, 1 });
    CHECK(mceOp->m_Op == ethosn::command_stream::MceOperation::CONVOLUTION);
}

// Manually creates a Network of Operands and Operations and converts it to a GraphOfParts using the NetworkToGraphOfPartsConverter.
// The topology is chosen to test that the TransposeConvolution operation with a large kernel is correctly
// converted to two MceParts, with the first using an upscale.
TEST_CASE("NetworkToGraphOfPartsConverter TransposeConvolution Large Weights")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{ { { 1, 16, 16, 16 } }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, { 0, 1.f } };
    TensorInfo biasInfo{ { { 1, 1, 1, 4 } }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, 1.f } };
    TensorInfo weightsInfo{ { { 9, 9, 16, 4 } }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, { 0, 1.f } };
    ConvolutionInfo convInfo{ { 4, 4, 4, 4 }, { 2, 2 }, { 0, 1.1f } };

    const std::vector<int32_t> biasData(utils::TotalSizeBytes(biasInfo));
    const std::vector<uint8_t> weightsData(utils::TotalSizeBytes(weightsInfo));

    const std::shared_ptr<Network> network =
        CreateNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));
    std::shared_ptr<Operand> input    = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Constant> bias    = AddConstant(network, biasInfo, biasData.data()).tensor;
    std::shared_ptr<Constant> weights = AddConstant(network, weightsInfo, weightsData.data()).tensor;
    std::shared_ptr<Operand> tconv    = AddTransposeConvolution(network, *input, *bias, *weights, convInfo).tensor;
    std::shared_ptr<Output> output    = AddOutput(network, *tconv).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest TransposeConvolution Large Weights.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest TransposeConvolution Large Weights Output.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::High);
    }

    // InputPart, McePart, McePart, OutputPart
    REQUIRE(graph.GetNumParts() == 4);

    // We check only the MceParts that we expect to be created - the Input and Output part and connections
    // between the Parts are covered by NetworkToGraphOfPartsConverterTest
    const McePart* mcePart1 = dynamic_cast<const McePart*>(&graph.GetPart(1));
    REQUIRE(mcePart1 != nullptr);
    auto plans1     = mcePart1->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    Op* maybeMceOp1 = plans1[0].m_OpGraph.GetOp(1);
    REQUIRE(IsMceOp(maybeMceOp1));
    MceOp* mceOp1 = static_cast<MceOp*>(maybeMceOp1);
    CHECK(mceOp1->m_UpscaleFactor == 2);
    CHECK(mceOp1->m_UpsampleType == MceUpsampleType::TRANSPOSE);
    CHECK(mceOp1->m_PadTop == 0);
    CHECK(mceOp1->m_PadLeft == 0);
    CHECK(mceOp1->m_Stride == Stride{ 1, 1 });
    CHECK(mceOp1->m_Op == ethosn::command_stream::MceOperation::DEPTHWISE_CONVOLUTION);

    const McePart* mcePart2 = dynamic_cast<const McePart*>(&graph.GetPart(2));
    REQUIRE(mcePart2 != nullptr);
    auto plans2     = mcePart2->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    Op* maybeMceOp2 = plans2[0].m_OpGraph.GetOp(1);
    REQUIRE(IsMceOp(maybeMceOp2));
    MceOp* mceOp2 = static_cast<MceOp*>(maybeMceOp2);
    CHECK(mceOp2->m_UpscaleFactor == 1);
    CHECK(mceOp2->m_UpsampleType == MceUpsampleType::OFF);
    CHECK(mceOp2->m_PadTop == 4);
    CHECK(mceOp2->m_PadLeft == 4);
    CHECK(mceOp2->m_Stride == Stride{ 1, 1 });
    CHECK(mceOp2->m_Op == ethosn::command_stream::MceOperation::CONVOLUTION);
}

// Manually creates a Network of Operands and Operations and converts it to a GraphOfParts using the NetworkToGraphOfPartsConverter.
// The topology is chosen to test that the TransposeConvolution operation with an estimate-only configuration
// is converted to an EstimateOnlyPart
TEST_CASE("NetworkToGraphOfPartsConverter TransposeConvolution EstimateOnly")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{ { { 1, 16, 16, 16 } }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, { 0, 1.f } };
    TensorInfo biasInfo{ { { 1, 1, 1, 4 } }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, 1.f } };
    TensorInfo weightsInfo{ { { 9, 9, 16, 4 } }, DataType::UINT8_QUANTIZED, DataFormat::HWIO, { 0, 1.f } };
    // Stride 3,3 is estimate-only
    ConvolutionInfo convInfo{ { 4, 4, 4, 4 }, { 3, 3 }, { 0, 1.1f } };

    const std::vector<int32_t> biasData(utils::TotalSizeBytes(biasInfo));
    const std::vector<uint8_t> weightsData(utils::TotalSizeBytes(weightsInfo));

    const std::shared_ptr<Network> network =
        CreateEstimationNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));
    std::shared_ptr<Operand> input    = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Constant> bias    = AddConstant(network, biasInfo, biasData.data()).tensor;
    std::shared_ptr<Constant> weights = AddConstant(network, weightsInfo, weightsData.data()).tensor;
    std::shared_ptr<Operand> tconv    = AddTransposeConvolution(network, *input, *bias, *weights, convInfo).tensor;
    std::shared_ptr<Output> output    = AddOutput(network, *tconv).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest TransposeConvolution EstimateOnly.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest TransposeConvolution EstimateOnly Output.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::High);
    }

    // InputPart, EstimateOnlyPart, OutputPart
    REQUIRE(graph.GetNumParts() == 3);

    // We check only the EstimateOnlyPart that we expect to be created - the Input and Output part and connections
    // between the Parts are covered by NetworkToGraphOfPartsConverterTest
    const EstimateOnlyPart* estimateOnlyPart = dynamic_cast<const EstimateOnlyPart*>(&graph.GetPart(1));
    REQUIRE(estimateOnlyPart != nullptr);
    auto plans = estimateOnlyPart->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    REQUIRE(plans[0].GetInputBuffer(PartInputSlot{ estimateOnlyPart->GetPartId(), 0 })->m_TensorShape ==
            TensorShape{ 1, 16, 16, 16 });
    REQUIRE(plans[0].GetOutputBuffer(PartOutputSlot{ estimateOnlyPart->GetPartId(), 0 })->m_TensorShape ==
            TensorShape{ 1, 46, 46, 4 });
    Op* maybeEstimateOnlyOp = plans[0].m_OpGraph.GetOp(0);
    REQUIRE(IsEstimateOnlyOp(maybeEstimateOnlyOp));
    EstimateOnlyOp* estimateOnlyOp = static_cast<EstimateOnlyOp*>(maybeEstimateOnlyOp);
    CHECK(estimateOnlyOp->m_ReasonForEstimateOnly.find("Unsupported stride") != std::string::npos);
}

TEST_CASE("NetworkToGraphOfPartsConverter Reinterpret Quantization")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{
        { { 1, 16, 16, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 0.9f },
    };

    TensorInfo biasInfo{
        { { 1, 1, 1, 16 } },
        DataType::INT32_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    TensorInfo weightsInfo{
        { { 1, 1, 16, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::HWIO,
        { 0, 1.f },
    };

    ConvolutionInfo convInfo{
        { 0, 0, 0, 0 },
        { 1, 1 },
        { 0, 1.1f },
    };

    const std::vector<uint8_t> biasData(utils::TotalSizeBytes(biasInfo));
    const std::vector<uint8_t> weightsData(utils::TotalSizeBytes(weightsInfo));

    const std::shared_ptr<Network> network =
        CreateNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Network topology:
    // Input -> ReinterpretQuant -> Conv -> Output
    std::shared_ptr<Operand> input = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Operand> reinterpretQuant =
        AddReinterpretQuantization(network, *input, QuantizationInfo(0, 1.f)).tensor;
    std::shared_ptr<Constant> bias    = AddConstant(network, biasInfo, biasData.data()).tensor;
    std::shared_ptr<Constant> weights = AddConstant(network, weightsInfo, weightsData.data()).tensor;
    std::shared_ptr<Operand> conv     = AddConvolution(network, *reinterpretQuant, *bias, *weights, convInfo).tensor;
    std::shared_ptr<Output> output    = AddOutput(network, *conv).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTestsReinterpretQuantization.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTests_ReinterpretQuantizationOutput.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::Low);
    }

    // InputPart, McePart, OutputPart
    REQUIRE(graph.GetNumParts() == 3);

    {
        const McePart* part = dynamic_cast<const McePart*>(&graph.GetPart(1));
        REQUIRE(part != nullptr);

        auto plans = part->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
        REQUIRE(plans[0].m_OpGraph.GetBuffers()[0]->m_QuantizationInfo == QuantizationInfo(0, 1.f));
    }
}

TEST_CASE("NetworkToGraphOfPartsConverter Split")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{
        { { 1, 16, 16, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 0.9f },
    };

    SplitInfo splitInfo{ 1, { 9, 7 } };

    const std::shared_ptr<Network> network =
        CreateNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Network topology:
    // Input -> Split -> Output
    //                -> Output
    std::shared_ptr<Operand> input              = AddInput(network, inputInfo).tensor;
    std::vector<std::shared_ptr<Operand>> split = AddSplit(network, *input, splitInfo).tensors;
    std::shared_ptr<Output> output0             = AddOutput(network, *split[0]).tensor;
    std::shared_ptr<Output> output1             = AddOutput(network, *split[1]).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTestsSplit.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTests_SplitOutput.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::High);
    }

    // InputPart, SplitPart, OutputPart, OutputPart
    REQUIRE(graph.GetNumParts() == 4);

    REQUIRE(dynamic_cast<const SplitPart*>(&graph.GetPart(1)) != nullptr);
    REQUIRE(graph.GetPartInputs(1).size() == 1);
    REQUIRE(graph.GetPartOutputs(1).size() == 2);
    REQUIRE(graph.GetConnectedOutputSlot({ 1, 0 }).value().m_PartId == 0);
    REQUIRE(graph.GetConnectedInputSlots({ 1, 0 }).size() == 1);
    REQUIRE(graph.GetConnectedInputSlots({ 1, 1 }).size() == 1);

    REQUIRE(dynamic_cast<const OutputPart*>(&graph.GetPart(2)) != nullptr);
    REQUIRE(graph.GetPartInputs(2).size() == 1);
    REQUIRE(graph.GetPartOutputs(2).size() == 0);
    REQUIRE(graph.GetConnectedOutputSlot({ 2, 0 }).value().m_PartId == 1);
    REQUIRE(graph.GetConnectedInputSlots({ 2, 0 }).size() == 0);

    REQUIRE(dynamic_cast<const OutputPart*>(&graph.GetPart(3)) != nullptr);
    REQUIRE(graph.GetPartInputs(3).size() == 1);
    REQUIRE(graph.GetPartOutputs(3).size() == 0);
    REQUIRE(graph.GetConnectedOutputSlot({ 3, 0 }).value().m_PartId == 1);
    REQUIRE(graph.GetConnectedInputSlots({ 3, 0 }).size() == 0);
}

TEST_CASE("NetworkToGraphOfPartsConverter Split Padding")
{
    // Confirms that padding channels are added as expected.

    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{
        { { 1, 16, 16, 2 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 0.9f },
    };

    SplitInfo splitInfo{ 3, { 1, 1 } };

    const std::shared_ptr<Network> network =
        std::make_shared<Network>(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO), false, true);

    // Network topology:
    // Input -> Split -> Output
    //                -> Output
    std::shared_ptr<Operand> input              = AddInput(network, inputInfo).tensor;
    std::vector<std::shared_ptr<Operand>> split = AddSplit(network, *input, splitInfo).tensors;
    std::shared_ptr<Output> output0             = AddOutput(network, *split[0]).tensor;
    std::shared_ptr<Output> output1             = AddOutput(network, *split[1]).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverter Split Padding.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverter Split Padding Output.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::High);
    }

    // InputPart, McePart (to add padding channels), SplitPart, OutputPart, OutputPart
    REQUIRE(graph.GetNumParts() == 5);

    const McePart* mcePart = dynamic_cast<const McePart*>(&graph.GetPart(1));
    REQUIRE(mcePart != nullptr);
    CHECK(graph.GetConnectedOutputSlot({ 1, 0 }).value().m_PartId == 0);
    CHECK(graph.GetConnectedInputSlots({ 1, 0 }) == std::vector<PartInputSlot>{ { 2, 0 } });
    // Check that padding channels have been added
    CHECK(utils::GetChannels(mcePart->GetOutputTensorShape()) == 32);
    // clang-format off
    CHECK(mcePart->GetWeightsData() == std::vector<uint8_t>{
        2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    });
    // clang-format on

    const SplitPart* splitPart = dynamic_cast<const SplitPart*>(&graph.GetPart(2));
    REQUIRE(splitPart != nullptr);
    CHECK(graph.GetConnectedOutputSlot({ 2, 0 }).value().m_PartId == 1);
    CHECK(graph.GetConnectedInputSlots({ 2, 0 }) == std::vector<PartInputSlot>{ { 3, 0 } });
    CHECK(graph.GetConnectedInputSlots({ 2, 1 }) == std::vector<PartInputSlot>{ { 4, 0 } });
    // Check the split offsets
    CHECK(utils::GetChannels(splitPart->GetInputTensorShape()) == 32);
    CHECK(splitPart->GetOffsets() == std::vector<uint32_t>{ 0, 16 });
}

TEST_CASE("NetworkToGraphOfPartsConverter Transpose")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{
        { { 1, 32, 16, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 0.9f },
    };

    TransposeInfo transposeInfo{ { 0, 2, 1, 3 } };

    const std::shared_ptr<Network> network =
        CreateEstimationNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Network topology:
    // Input -> Transpose -> Output
    std::shared_ptr<Operand> input     = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Operand> transpose = AddTranspose(network, *input, transposeInfo).tensor;
    std::shared_ptr<Output> output     = AddOutput(network, *transpose).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTestsTranspose.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTests_TransposeOutput.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::Low);
    }

    // InputPart, EstimateOnlyPart, OutputPart
    REQUIRE(graph.GetNumParts() == 3);

    {
        const EstimateOnlyPart* part = dynamic_cast<const EstimateOnlyPart*>(&graph.GetPart(1));
        REQUIRE(part != nullptr);
    }
}

TEST_CASE("NetworkToGraphOfPartsConverter SpaceToDepth")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{
        { { 1, 32, 16, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 0.9f },
    };

    SpaceToDepthInfo spaceToDepthInfo(2);
    const std::shared_ptr<Network> network =
        CreateEstimationNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Network topology:
    // Input -> SpaceToDepth -> Output
    std::shared_ptr<Operand> input     = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Operand> transpose = AddSpaceToDepth(network, *input, spaceToDepthInfo).tensor;
    std::shared_ptr<Output> output     = AddOutput(network, *transpose).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTestsSpaceToDepth.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTests_SpaceToDepthOutput.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::Low);
    }

    // InputPart, EstimateOnlyPart, OutputPart
    REQUIRE(graph.GetNumParts() == 3);

    {
        const EstimateOnlyPart* part = dynamic_cast<const EstimateOnlyPart*>(&graph.GetPart(1));
        REQUIRE(part != nullptr);
    }
}

TEST_CASE("NetworkToGraphOfPartsConverterTest Downsample_2x2")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{
        { { 1, 32, 32, 1 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    constexpr PoolingInfo poolingInfo{ 1, 1, 2, 2, { 0, 0, 0, 0 }, PoolingType::MAX };

    const std::shared_ptr<Network> network =
        CreateNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Network topology:
    /*
       { Input } -> MaxPool_1x1_2_2 -> Output
    */

    std::shared_ptr<Operand> input      = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Operand> downsample = AddPooling(network, *input, poolingInfo).tensor;
    std::shared_ptr<Output> output      = AddOutput(network, *downsample).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest DownSample_2x2.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest DownSample_2x2_Output.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::High);
    }

    // Check for each Part:
    //  * Whether the type of the generated Part is correct
    //  * Whether the PleOperation command stream is correct for Operations using FusedPleParts (DOWNSAMPLE_2X2)
    //  * The number of Input/Output slots
    //  * Whether PartInputSlots connect to PartOutputSlots of the correct Part
    //  * For the last Part, check that there are no connections to any following PartInputSlots
    REQUIRE(graph.GetNumParts() == 3);

    REQUIRE(dynamic_cast<const InputPart*>(&graph.GetPart(0)) != nullptr);
    REQUIRE(graph.GetPartInputs(0).size() == 0);
    REQUIRE(graph.GetPartOutputs(0).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 0, 0 }).has_value() == false);

    const FusedPlePart* downsamplePart = dynamic_cast<const FusedPlePart*>(&graph.GetPart(1));
    REQUIRE(downsamplePart != nullptr);
    auto downsamplePlans =
        downsamplePart->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    Op* maybePleOpDownsamplePlans = downsamplePlans[0].m_OpGraph.GetOp(2);
    REQUIRE(IsPleOp(maybePleOpDownsamplePlans));
    PleOp* pleOpDownsamplePlans = static_cast<PleOp*>(maybePleOpDownsamplePlans);
    REQUIRE(pleOpDownsamplePlans->m_Op == ethosn::command_stream::PleOperation::DOWNSAMPLE_2X2);
    REQUIRE(graph.GetPartInputs(1).size() == 1);
    REQUIRE(graph.GetPartOutputs(1).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 1, 0 }).value().m_PartId == 0);

    REQUIRE(dynamic_cast<const OutputPart*>(&graph.GetPart(2)) != nullptr);
    REQUIRE(graph.GetPartInputs(2).size() == 1);
    REQUIRE(graph.GetPartOutputs(2).size() == 0);
    REQUIRE(graph.GetConnectedOutputSlot({ 2, 0 }).value().m_PartId == 1);
    REQUIRE(graph.GetConnectedInputSlots({ 2, 0 }).size() == 0);
}

// Manually creates a Network of Operands and Operations and converts it to a GraphOfParts using the
// NetworkToGraphOfPartsConverter.
// The topology is chosen to test that the DepthToSpace operation is correctly converted to an
// McePart.
TEST_CASE("NetworkToGraphOfPartsConverter DepthToSpace")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{ { { 1, 1, 1, 4 } }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, { 0, 1.f } };
    DepthToSpaceInfo depthToSpaceInfo{ 2 };

    const std::shared_ptr<Network> network =
        CreateNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));
    std::shared_ptr<Operand> input        = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Operand> depthtospace = AddDepthToSpace(network, *input, depthToSpaceInfo).tensor;
    std::shared_ptr<Output> output        = AddOutput(network, *depthtospace).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest DepthToSpace.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest DeppthToSpace Output.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::High);
    }

    // InputPart, McePart, OutputPart
    REQUIRE(graph.GetNumParts() == 3);

    // We check only the McePart that we expect to be created - the Input and Output part and
    // connections
    // between the Parts are covered by NetworkToGraphOfPartsConverterTest
    const McePart* mcePart = dynamic_cast<const McePart*>(&graph.GetPart(1));
    REQUIRE(mcePart != nullptr);
    auto plans     = mcePart->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    Op* maybeMceOp = plans[0].m_OpGraph.GetOp(1);
    REQUIRE(IsMceOp(maybeMceOp));
    MceOp* mceOp = static_cast<MceOp*>(maybeMceOp);
    CHECK(mceOp->m_UpscaleFactor == 2);
    CHECK(mceOp->m_UpsampleType == MceUpsampleType::TRANSPOSE);
    CHECK(mceOp->m_PadTop == 1);
    CHECK(mceOp->m_PadLeft == 1);
    CHECK(mceOp->m_Stride == Stride{ 1, 1 });
    CHECK(mceOp->m_Op == ethosn::command_stream::MceOperation::CONVOLUTION);
}

TEST_CASE("NetworkToGraphOfPartsConverter DepthToSpace EstimateOnly")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{ { { 1, 1, 1, 16 } }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, { 0, 1.f } };

    // Only block size of 2 is supported, so this will return EstimateOnly
    // when IsDepthToSpaceSupported is called.
    DepthToSpaceInfo depthToSpaceInfo{ 1 };

    const std::shared_ptr<Network> network =
        CreateEstimationNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));
    std::shared_ptr<Operand> input        = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Operand> depthtospace = AddDepthToSpace(network, *input, depthToSpaceInfo).tensor;
    std::shared_ptr<Output> output        = AddOutput(network, *depthtospace).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest DepthToSpace EstimateOnly.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest DepthToSpace EstimateOnly Output.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::High);
    }

    // InputPart, McePart, OutputPart
    REQUIRE(graph.GetNumParts() == 3);

    // We check only the EstimateOnlyPart that we expect to be created - the Input and Output part and connections
    // between the Parts are covered by NetworkToGraphOfPartsConverterTest
    const EstimateOnlyPart* estimateOnlyPart = dynamic_cast<const EstimateOnlyPart*>(&graph.GetPart(1));
    REQUIRE(estimateOnlyPart != nullptr);
    auto plans = estimateOnlyPart->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    REQUIRE(plans[0].GetInputBuffer(PartInputSlot{ estimateOnlyPart->GetPartId(), 0 })->m_TensorShape ==
            TensorShape{ 1, 1, 1, 16 });
    REQUIRE(plans[0].GetOutputBuffer(PartOutputSlot{ estimateOnlyPart->GetPartId(), 0 })->m_TensorShape ==
            TensorShape{ 1, 1, 1, 16 });
    Op* maybeEstimateOnlyOp = plans[0].m_OpGraph.GetOp(0);
    REQUIRE(IsEstimateOnlyOp(maybeEstimateOnlyOp));
    EstimateOnlyOp* estimateOnlyOp = static_cast<EstimateOnlyOp*>(maybeEstimateOnlyOp);
    CHECK(estimateOnlyOp->m_ReasonForEstimateOnly.find("Only block size of 2 is supported") != std::string::npos);
}

TEST_CASE("NetworkToGraphOfPartsConverter EstimateOnly")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;

    TensorInfo inputInfo{ { 1, 1, 1, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC, { 0, 1.f } };

    std::string reasonForEstimateOnly = "EstimateOnly operation added.";
    EstimateOnlyInfo estimateOnlyInfo({ inputInfo }, reasonForEstimateOnly);

    const std::shared_ptr<Network> network =
        CreateEstimationNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    TensorAndId<Operand> input = AddInput(network, inputInfo);
    TensorsAndId estimateOnly = AddEstimateOnly(network, std::vector<Operand*>{ input.tensor.get() }, estimateOnlyInfo);
    TensorAndId<Output> output = AddOutput(network, *estimateOnly.tensors[0], DataFormat::NHWCB);

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest EstimateOnly.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(*network, caps, estOpt, compOpt, debuggingContext);
    GraphOfParts graph = networkToGraphOfPartsConverter.ReleaseGraphOfParts();
    graph.SortAndCompact();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest EstimateOnly Output.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::Low);
    }

    REQUIRE(graph.GetNumParts() == 3);

    const EstimateOnlyPart* estimateOnlyPart = dynamic_cast<const EstimateOnlyPart*>(&graph.GetPart(1));
    REQUIRE(estimateOnlyPart != nullptr);
    auto plans = estimateOnlyPart->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    REQUIRE(plans[0].GetInputBuffer(PartInputSlot{ estimateOnlyPart->GetPartId(), 0 })->m_TensorShape ==
            TensorShape{ 1, 1, 1, 16 });
    REQUIRE(plans[0].GetOutputBuffer(PartOutputSlot{ estimateOnlyPart->GetPartId(), 0 })->m_TensorShape ==
            TensorShape{ 1, 1, 1, 16 });
    Op* maybeEstimateOnlyOp = plans[0].m_OpGraph.GetOp(0);
    REQUIRE(IsEstimateOnlyOp(maybeEstimateOnlyOp));
    EstimateOnlyOp* estimateOnlyOp = static_cast<EstimateOnlyOp*>(maybeEstimateOnlyOp);
    CHECK(estimateOnlyOp->m_ReasonForEstimateOnly.find("EstimateOnly operation added.") != std::string::npos);
}
