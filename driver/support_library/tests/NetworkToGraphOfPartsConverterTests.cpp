//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"
#include "../src/DebuggingContext.hpp"
#include "../src/Network.hpp"
#include "../src/Utils.hpp"
#include "../src/cascading/NetworkToGraphOfPartsConverter.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>

#include <fstream>
#include <iostream>
#include <unordered_set>

using namespace ethosn::support_library;

// Manually creates a Network of Operands and Operations and converts it to a GraphOfParts using the NetworkToGraphOfPartsConverter().
// The topology is chosen to test Networks of supported Part types such as:
//      * Input Part
//      * Mce Part
//      * Pooling Part (MAX))
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

    NetworkToGraphOfPartsConverter m_NetworkToGraphOfPartsConverter(*network, caps, estOpt, compOpt);
    GraphOfParts graph = m_NetworkToGraphOfPartsConverter.ReleaseGraphOfParts();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest Output.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::Low);
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

// Manually creates a Network of Operands and Operations and converts it to a GraphOfParts using the NetworkToGraphOfPartsConverter().
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

    NetworkToGraphOfPartsConverter m_NetworkToGraphOfPartsConverter(*network, caps, estOpt, compOpt);
    GraphOfParts graph = m_NetworkToGraphOfPartsConverter.ReleaseGraphOfParts();

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

// Manually creates a Network of Operands and Operations and converts it to a GraphOfParts using the NetworkToGraphOfPartsConverter().
// The topology is chosen to test Networks of supported Part types such as:
//      * MeanXy Part (7x7, 8x8 variations)
TEST_CASE("NetworkToGraphOfPartsConverterTest MeanXy")
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

    TensorInfo input2Info{
        { { 1, 8, 8, 16 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::NHWC,
        { 0, 1.f },
    };

    const std::shared_ptr<Network> network =
        CreateNetwork(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

    // Network topology:
    /* 
       { Input2 } -> MeanXy_8x8 -> Output2
       { Input } -> MeanXy_7x7 -> Output
    */

    std::shared_ptr<Operand> input   = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Operand> input2  = AddInput(network, input2Info).tensor;
    std::shared_ptr<Operand> meanxy  = AddMeanXy(network, *input).tensor;
    std::shared_ptr<Operand> meanxy2 = AddMeanXy(network, *input2).tensor;
    std::shared_ptr<Output> output   = AddOutput(network, *meanxy).tensor;
    std::shared_ptr<Output> output2  = AddOutput(network, *meanxy2).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest MeanXy.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    NetworkToGraphOfPartsConverter m_NetworkToGraphOfPartsConverter(*network, caps, estOpt, compOpt);
    GraphOfParts graph = m_NetworkToGraphOfPartsConverter.ReleaseGraphOfParts();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest MeanXy Output.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::High);
    }

    // Check for each Part:
    //  * Whether the type of the generated Part is correct
    //  * Whether the PleOperation command stream is correct for Operations using FusedPleParts (e.g. MeanXy_7x7, MeanXy_8x8...)
    //  * The number of Input/Output slots
    //  * Whether PartInputSlots connect to PartOutputSlots of the correct Part
    //  * For the last Part, check that there are no connections to any following PartInputSlots
    REQUIRE(graph.GetNumParts() == 6);

    REQUIRE(dynamic_cast<const InputPart*>(&graph.GetPart(0)) != nullptr);
    REQUIRE(graph.GetPartInputs(0).size() == 0);
    REQUIRE(graph.GetPartOutputs(0).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 0, 0 }).has_value() == false);

    const FusedPlePart* meanxyPlePart = dynamic_cast<const FusedPlePart*>(&graph.GetPart(1));
    REQUIRE(meanxyPlePart != nullptr);
    auto meanxyPlans = meanxyPlePart->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    REQUIRE(dynamic_cast<PleOp*>(meanxyPlans[0].m_OpGraph.GetOp(2))->m_Op ==
            ethosn::command_stream::PleOperation::MEAN_XY_7X7);
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

    const FusedPlePart* meanxy2PlePart = dynamic_cast<const FusedPlePart*>(&graph.GetPart(4));
    REQUIRE(meanxy2PlePart != nullptr);
    auto meanxy2Plans =
        meanxy2PlePart->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    REQUIRE(dynamic_cast<PleOp*>(meanxy2Plans[0].m_OpGraph.GetOp(2))->m_Op ==
            ethosn::command_stream::PleOperation::MEAN_XY_8X8);
    REQUIRE(graph.GetPartInputs(4).size() == 1);
    REQUIRE(graph.GetPartOutputs(4).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 4, 0 }).value().m_PartId == 3);

    REQUIRE(dynamic_cast<const OutputPart*>(&graph.GetPart(5)) != nullptr);
    REQUIRE(graph.GetPartInputs(5).size() == 1);
    REQUIRE(graph.GetPartOutputs(5).size() == 0);
    REQUIRE(graph.GetConnectedOutputSlot({ 5, 0 }).value().m_PartId == 4);
    REQUIRE(graph.GetConnectedInputSlots({ 5, 0 }).size() == 0);
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
                 /-> LeakyRelu -> Output2
       { Input } - > Sigmoid -> Output1
                 \-> Tanh -> Output
    */

    std::shared_ptr<Operand> input     = AddInput(network, inputInfo).tensor;
    std::shared_ptr<Operand> leakyrelu = AddLeakyRelu(network, *input, leakyreluInfo).tensor;
    std::shared_ptr<Operand> sigmoid   = AddSigmoid(network, *input).tensor;
    std::shared_ptr<Operand> tanh      = AddTanh(network, *input).tensor;
    std::shared_ptr<Output> output     = AddOutput(network, *leakyrelu).tensor;
    std::shared_ptr<Output> output2    = AddOutput(network, *sigmoid).tensor;
    std::shared_ptr<Output> output3    = AddOutput(network, *tanh).tensor;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest LeakyRelu Sigmoid Tanh.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    NetworkToGraphOfPartsConverter m_NetworkToGraphOfPartsConverter(*network, caps, estOpt, compOpt);
    GraphOfParts graph = m_NetworkToGraphOfPartsConverter.ReleaseGraphOfParts();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTest LeakyRelu Sigmoid Tanh Output.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::High);
    }

    // Check for each Part:
    //  * Whether the type of the generated Part is correct
    //  * Whether the PleOperation command stream is correct for Operations using FusedPleParts (e.g. LeakyRelu, Sigmoid, Tanh...)
    //  * The number of Input/Output slots
    //  * Whether PartInputSlots connect to PartOutputSlots of the correct Part
    //  * For the last Part, check that there are no connections to any following PartInputSlots
    REQUIRE(graph.GetNumParts() == 7);

    REQUIRE(dynamic_cast<const InputPart*>(&graph.GetPart(0)) != nullptr);
    REQUIRE(graph.GetPartInputs(0).size() == 0);
    REQUIRE(graph.GetPartOutputs(0).size() == 3);
    REQUIRE(graph.GetConnectedOutputSlot({ 0, 0 }).has_value() == false);

    const FusedPlePart* tanhPlePart = dynamic_cast<const FusedPlePart*>(&graph.GetPart(1));
    REQUIRE(tanhPlePart != nullptr);
    auto tanhPlans = tanhPlePart->GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 1);
    REQUIRE(dynamic_cast<PleOp*>(tanhPlans[0].m_OpGraph.GetOp(2))->m_Op ==
            ethosn::command_stream::PleOperation::SIGMOID);
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
    REQUIRE(dynamic_cast<PleOp*>(sigmoidPlans[0].m_OpGraph.GetOp(2))->m_Op ==
            ethosn::command_stream::PleOperation::SIGMOID);
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
    REQUIRE(dynamic_cast<PleOp*>(leakyreluPlans[0].m_OpGraph.GetOp(2))->m_Op ==
            ethosn::command_stream::PleOperation::LEAKY_RELU);
    REQUIRE(graph.GetPartInputs(5).size() == 1);
    REQUIRE(graph.GetPartOutputs(5).size() == 1);
    REQUIRE(graph.GetConnectedOutputSlot({ 5, 0 }).value().m_PartId == 0);

    REQUIRE(dynamic_cast<const OutputPart*>(&graph.GetPart(6)) != nullptr);
    REQUIRE(graph.GetPartInputs(6).size() == 1);
    REQUIRE(graph.GetPartOutputs(6).size() == 0);
    REQUIRE(graph.GetConnectedOutputSlot({ 6, 0 }).value().m_PartId == 5);
    REQUIRE(graph.GetConnectedInputSlots({ 6, 0 }).size() == 0);
}
