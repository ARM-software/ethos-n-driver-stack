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
TEST_CASE("NetworkToGraphOfPartsConverterTests")
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
    std::shared_ptr<Operand> input = AddInput(network, inputInfo).tensor;

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
        std::ofstream stream("NetworkToGraphOfPartsConverterTests.dot");
        SaveNetworkToDot(*network, stream, DetailLevel::High);
    }

    NetworkToGraphOfPartsConverter m_NetworkToGraphOfPartsConverter(*network, caps, estOpt, compOpt);
    GraphOfParts graph = m_NetworkToGraphOfPartsConverter.ReleaseGraphOfParts();

    bool dumpGraphOfPartsToFile = false;
    if (dumpGraphOfPartsToFile)
    {
        std::ofstream stream("NetworkToGraphOfPartsConverterTests_Output.dot");
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
