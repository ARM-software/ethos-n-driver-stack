//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "GraphNodes.hpp"
#include "TestUtils.hpp"
#include "cascading/CombinerDFS.hpp"
#include "cascading/ConcatPart.hpp"
#include "cascading/Estimation.hpp"
#include "cascading/FusedPlePart.hpp"
#include "cascading/InputPart.hpp"
#include "cascading/McePart.hpp"
#include "cascading/OutputPart.hpp"
#include "cascading/Plan.hpp"
#include "cascading/ReshapePart.hpp"
#include "cascading/StandalonePlePart.hpp"
#include "cascading/Visualisation.hpp"

#include <catch.hpp>
#include <fstream>

using namespace ethosn;
using namespace ethosn::support_library;
namespace sl    = ethosn::support_library;
namespace utils = ethosn::support_library::utils;
using namespace ethosn::command_stream;

/// Checks SaveNetworkToDot produces the expected output, focusing on the overall network topology (connections
/// between operations) rather than on the details given for each individual operation.
TEST_CASE("SaveNetworkToDot Network Topology", "[Visualisation]")
{
    // Build an arbitrary network, making sure to demonstrate multiple inputs, multiple outputs and multiple consumers.
    Network network(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO), true);

    Input& input = network.AddInput(sl::TensorInfo({ 1, 16, 16, 32 }));
    network.AddOutput(input.GetOutput(0), sl::DataFormat::NHWCB);
    Split& split          = network.AddSplit(input.GetOutput(0), SplitInfo(3, { 16, 16 }));
    Concatenation& concat = network.AddConcatenation({ &split.GetOutput(0), &split.GetOutput(1) },
                                                     ConcatenationInfo(3, QuantizationInfo()));
    network.AddOutput(concat.GetOutput(0), sl::DataFormat::NHWCB);

    // For easier debugging of this test (and so that you can see the pretty graph!), dump to a file
    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("SaveNetworkToDot Network Topology.dot");
        SaveNetworkToDot(network, stream, DetailLevel::Low);
    }

    // Save to a string and check against expected result
    std::stringstream stream;
    SaveNetworkToDot(network, stream, DetailLevel::Low);

    std::string expected =
        R"(digraph SupportLibraryGraph
{
Operation0[label = "0: Input\n", shape = oval]
Operand0_0[label = "Operand\n", shape = box]
Operation0 -> Operand0_0
Operation1[label = "1: Output\n", shape = oval]
Operand0_0 -> Operation1
Operation2[label = "2: Split\n", shape = oval]
Operand0_0 -> Operation2
Operand2_0[label = "Operand\n", shape = box]
Operation2 -> Operand2_0[ label="Output 0"]
Operand2_1[label = "Operand\n", shape = box]
Operation2 -> Operand2_1[ label="Output 1"]
Operation3[label = "3: Concatenation\n", shape = oval]
Operand2_0 -> Operation3[ label="Input 0"]
Operand2_1 -> Operation3[ label="Input 1"]
Operand3_0[label = "Operand\n", shape = box]
Operation3 -> Operand3_0
Operation4[label = "4: Output\n", shape = oval]
Operand3_0 -> Operation4
}
)";

    REQUIRE(stream.str() == expected);
}

/// Checks SaveNetworkToDot produces the expected output, focusing on the details given for each individual operation/
/// operand rather than the overall graph topology (connections between operations and operands).
TEST_CASE("SaveNetworkToDot Details", "[Visualisation]")
{
    // Build a simple network of operations, to check the details are printed correctly for each one.
    Network network(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO));

    Input& input   = network.AddInput(sl::TensorInfo({ 1, 16, 16, 32 }));
    Constant& bias = network.AddConstant(
        sl::TensorInfo({ 1, 1, 1, 32 }, sl::DataType::INT32_QUANTIZED, sl::DataFormat::NHWC, QuantizationInfo(0, 0.5f)),
        std::vector<int32_t>(32, 0).data());
    Constant& weightsConv = network.AddConstant(sl::TensorInfo({ 3, 3, 32, 32 }, sl::DataType::UINT8_QUANTIZED,
                                                               sl::DataFormat::HWIO, QuantizationInfo(0, 0.5f)),
                                                std::vector<int32_t>(3 * 3 * 32 * 32, 0).data());
    network.AddConvolution(input.GetOutput(0), bias, weightsConv, ConvolutionInfo());
    Constant& weightsDepthwise = network.AddConstant(
        sl::TensorInfo({ 3, 3, 32, 1 }, sl::DataType::UINT8_QUANTIZED, sl::DataFormat::HWIM, QuantizationInfo(0, 0.5f)),
        std::vector<int32_t>(3 * 3 * 32 * 1, 0).data());
    network.AddDepthwiseConvolution(input.GetOutput(0), bias, weightsDepthwise, ConvolutionInfo());
    network.AddTransposeConvolution(input.GetOutput(0), bias, weightsConv, ConvolutionInfo({}, { 2, 2 }));

    Input& inputFc      = network.AddInput(sl::TensorInfo({ 1, 1, 1, 32 }));
    Constant& weightsFc = network.AddConstant(sl::TensorInfo({ 1, 1, 32, 32 }, sl::DataType::UINT8_QUANTIZED,
                                                             sl::DataFormat::HWIO, QuantizationInfo(0, 0.5f)),
                                              std::vector<int32_t>(1 * 1 * 32 * 32, 0).data());
    network.AddFullyConnected(inputFc.GetOutput(0), bias, weightsFc, FullyConnectedInfo());

    // For easier debugging of this test (and so that you can see the pretty graph!), dump to a file
    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("SaveNetworkToDot Details.dot");
        SaveNetworkToDot(network, stream, DetailLevel::High);
    }

    // Save to a string and check against expected result
    std::stringstream stream;
    SaveNetworkToDot(network, stream, DetailLevel::High);

    std::string expected =
        R"(digraph SupportLibraryGraph
{
Operation0[label = "0: Input\n", shape = oval]
Operand0_0[label = "Operand\nShape = [1, 16, 16, 32]\nFormat = NHWC\nType = UINT8_QUANTIZED\nQuant. info = ZeroPoint = 0, Scale = 1.000000\n", shape = box]
Operation0 -> Operand0_0
Operation1[label = "1: Constant\n", shape = oval]
Operand1_0[label = "Operand\nShape = [1, 1, 1, 32]\nFormat = NHWC\nType = INT32_QUANTIZED\nQuant. info = ZeroPoint = 0, Scale = 0.500000\n", shape = box]
Operation1 -> Operand1_0
Operation2[label = "2: Constant\n", shape = oval]
Operand2_0[label = "Operand\nShape = [3, 3, 32, 32]\nFormat = HWIO\nType = UINT8_QUANTIZED\nQuant. info = ZeroPoint = 0, Scale = 0.500000\n", shape = box]
Operation2 -> Operand2_0
Operation3[label = "3: Convolution\nWeights: 2\nBias: 1\n", shape = oval]
Operand0_0 -> Operation3
Operand3_0[label = "Operand\nShape = [1, 14, 14, 32]\nFormat = NHWC\nType = UINT8_QUANTIZED\nQuant. info = ZeroPoint = 0, Scale = 1.000000\n", shape = box]
Operation3 -> Operand3_0
Operation4[label = "4: Constant\n", shape = oval]
Operand4_0[label = "Operand\nShape = [3, 3, 32, 1]\nFormat = HWIM\nType = UINT8_QUANTIZED\nQuant. info = ZeroPoint = 0, Scale = 0.500000\n", shape = box]
Operation4 -> Operand4_0
Operation5[label = "5: DepthwiseConvolution\nWeights: 4\nBias: 1\n", shape = oval]
Operand0_0 -> Operation5
Operand5_0[label = "Operand\nShape = [1, 14, 14, 32]\nFormat = NHWC\nType = UINT8_QUANTIZED\nQuant. info = ZeroPoint = 0, Scale = 1.000000\n", shape = box]
Operation5 -> Operand5_0
Operation6[label = "6: TransposeConvolution\nWeights: 2\nBias: 1\n", shape = oval]
Operand0_0 -> Operation6
Operand6_0[label = "Operand\nShape = [1, 33, 33, 32]\nFormat = NHWC\nType = UINT8_QUANTIZED\nQuant. info = ZeroPoint = 0, Scale = 1.000000\n", shape = box]
Operation6 -> Operand6_0
Operation7[label = "7: Input\n", shape = oval]
Operand7_0[label = "Operand\nShape = [1, 1, 1, 32]\nFormat = NHWC\nType = UINT8_QUANTIZED\nQuant. info = ZeroPoint = 0, Scale = 1.000000\n", shape = box]
Operation7 -> Operand7_0
Operation8[label = "8: Constant\n", shape = oval]
Operand8_0[label = "Operand\nShape = [1, 1, 32, 32]\nFormat = HWIO\nType = UINT8_QUANTIZED\nQuant. info = ZeroPoint = 0, Scale = 0.500000\n", shape = box]
Operation8 -> Operand8_0
Operation9[label = "9: FullyConnected\nWeights: 8\nBias: 1\n", shape = oval]
Operand7_0 -> Operation9
Operand9_0[label = "Operand\nShape = [1, 1, 1, 32]\nFormat = NHWC\nType = UINT8_QUANTIZED\nQuant. info = ZeroPoint = 0, Scale = 1.000000\n", shape = box]
Operation9 -> Operand9_0
}
)";

    REQUIRE(stream.str() == expected);
}

/// Checks SaveOpGraphToDot produces the expected output, focusing on the overall graph topology (connections
/// between nodes) rather than on the details given for each individual node.
TEST_CASE("SaveOpGraphToDot Graph Topology", "[Visualisation]")
{
    // Build an arbitrary graph, making sure to demonstrate multiple inputs and multiple consumers.
    // This is a rough approximation of what a Plan for convolution might look like:
    //
    //  Ifm (Dram)     -> Dma -> Ifm (Sram)     - \                                            /-> Consumer 1
    //                                             ->  Mce -> Ofm (Sram) -> Dma -> Ofm (Dram)
    //  Weights (Dram) -> Dma -> Weights (Sram) - /                                            \-> Consumer 2
    //
    OpGraph graph;

    Buffer dramIfm;
    dramIfm.m_DebugTag = "Dram Ifm";
    DmaOp dmaIfm;
    dmaIfm.m_DebugTag = "Dma Ifm";
    Buffer sramIfm;
    sramIfm.m_DebugTag = "Sram Ifm";

    Buffer dramWeights;
    dramWeights.m_DebugTag = "Dram Weights";
    DmaOp dmaWeights;
    dmaWeights.m_DebugTag = "Dma Weights";
    Buffer sramWeights;
    sramWeights.m_DebugTag = "Sram Weights";

    MceOp mce;
    mce.m_DebugTag = "Mce";

    Buffer sramOfm;
    sramOfm.m_DebugTag = "Sram Ofm";
    DmaOp dmaOfm;
    dmaOfm.m_DebugTag = "Dma Ofm";
    Buffer dramOfm;
    dramOfm.m_DebugTag = "Dram Ofm";

    MceOp consumer1;
    consumer1.m_DebugTag = "Consumer 1";
    MceOp consumer2;
    consumer2.m_DebugTag = "Consumer 2";

    graph.AddBuffer(&dramIfm);
    graph.AddOp(&dmaIfm);
    graph.AddBuffer(&sramIfm);
    graph.AddBuffer(&dramWeights);
    graph.AddOp(&dmaWeights);
    graph.AddBuffer(&sramWeights);
    graph.AddOp(&mce);
    graph.AddBuffer(&sramOfm);
    graph.AddOp(&dmaOfm);
    graph.AddBuffer(&dramOfm);
    graph.AddOp(&consumer1);
    graph.AddOp(&consumer2);

    graph.AddConsumer(&dramIfm, &dmaIfm, 0);
    graph.SetProducer(&sramIfm, &dmaIfm);
    graph.AddConsumer(&sramIfm, &mce, 0);
    graph.AddConsumer(&dramWeights, &dmaWeights, 0);
    graph.SetProducer(&sramWeights, &dmaWeights);
    graph.AddConsumer(&sramWeights, &mce, 1);
    graph.SetProducer(&sramOfm, &mce);
    graph.AddConsumer(&sramOfm, &dmaOfm, 0);
    graph.SetProducer(&dramOfm, &dmaOfm);
    graph.AddConsumer(&dramOfm, &consumer1, 0);
    graph.AddConsumer(&dramOfm, &consumer2, 0);

    // For easier debugging of this test (and so that you can see the pretty graph!), dump to a file
    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("SaveOpGraphToDot Graph Topology.dot");
        SaveOpGraphToDot(graph, stream, DetailLevel::Low);
    }

    // Save to a string and check against expected result
    std::stringstream stream;
    SaveOpGraphToDot(graph, stream, DetailLevel::Low);

    std::string expected =
        R"(digraph SupportLibraryGraph
{
Dma_Ifm[label = "Dma Ifm", shape = oval, color = darkgoldenrod]
Dma_Weights[label = "Dma Weights", shape = oval, color = darkgoldenrod]
Mce[label = "Mce", shape = oval]
Dma_Ofm[label = "Dma Ofm", shape = oval, color = darkgoldenrod]
Consumer_1[label = "Consumer 1", shape = oval]
Consumer_2[label = "Consumer 2", shape = oval]
Dram_Ifm[label = "Dram Ifm", shape = box, color = brown]
Sram_Ifm[label = "Sram Ifm", shape = box, color = brown]
Dram_Weights[label = "Dram Weights", shape = box, color = brown]
Sram_Weights[label = "Sram Weights", shape = box, color = brown]
Sram_Ofm[label = "Sram Ofm", shape = box, color = brown]
Dram_Ofm[label = "Dram Ofm", shape = box, color = brown]
Dram_Ifm -> Dma_Ifm
Dma_Ifm -> Sram_Ifm
Sram_Ifm -> Mce[ label="Input 0"]
Dram_Weights -> Dma_Weights
Dma_Weights -> Sram_Weights
Sram_Weights -> Mce[ label="Input 1"]
Mce -> Sram_Ofm
Sram_Ofm -> Dma_Ofm
Dma_Ofm -> Dram_Ofm
Dram_Ofm -> Consumer_1
Dram_Ofm -> Consumer_2
{ rank = "same"; Mce; Sram_Weights; Dma_Weights; Dram_Weights; }
}
)";

    REQUIRE(stream.str() == expected);
}

/// Checks SaveOpGraphToDot produces the expected output, focusing on the details given for each individual node
/// rather than the overall graph topology (connections between nodes).
TEST_CASE("SaveOpGraphToDot Node Details", "[Visualisation]")
{
    // Build a simple graph of disconnected nodes, to check the details are printed correctly for each one.
    OpGraph graph;

    Buffer buffer1(Lifetime::Cascade, Location::PleInputSram, CascadingBufferFormat::WEIGHT, { 1, 2, 3, 4 },
                   { 5, 6, 7, 8 }, TraversalOrder::Zxy, 1234, QuantizationInfo(10, 0.1f));
    buffer1.m_NumStripes = 9;
    buffer1.m_DebugTag   = "Buffer1";
    graph.AddBuffer(&buffer1);

    MceOp mce(Lifetime::Atomic, MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct, { 3u, 4u }, { 1, 2, 3, 4 },
              { 5, 6, 7, 8 }, { 9, 10, 11, 12 }, TraversalOrder::Zxy, Stride(10, 20), 30, 40, 100, 200);
    mce.m_DebugTag = "Mce";
    graph.AddOp(&mce);

    DmaOp dma(Lifetime::Cascade);
    dma.m_DebugTag = "Dma";
    graph.AddOp(&dma);

    PleOp ple(Lifetime::Atomic, PleOperation::ADDITION, { 16u, 16u }, 2, { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } },
              { 9, 10, 11, 12 }, ethosn::command_stream::DataType::U8);
    ple.m_DebugTag = "Ple";
    graph.AddOp(&ple);

    // For easier debugging of this test (and so that you can see the pretty graph!), dump to a file
    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("SaveOpGraphToDot Node Details.dot");
        SaveOpGraphToDot(graph, stream, DetailLevel::High);
    }

    // Save to a string and check against expected result
    std::stringstream stream;
    SaveOpGraphToDot(graph, stream, DetailLevel::High);

    std::string expected =
        R"(digraph SupportLibraryGraph
{
Mce[label = "Mce\nIdx in OpGraph: 0\nLifetime = Atomic\nMceOp\nOp = CONVOLUTION\nAlgo = DIRECT\nBlock Config = 3x4\nInput Stripe Shape = [1, 2, 3, 4]\nOutput Stripe Shape = [5, 6, 7, 8]\nWeights Stripe Shape = [9, 10, 11, 12]\nOrder = Zxy\nStride = 10, 20\nPad L/T = 30, 40\nLower/Upper Bound = 100, 200\nOperation Ids = []\n", shape = oval]
Dma[label = "Dma\nIdx in OpGraph: 1\nLifetime = Cascade\nDmaOp\nOperation Ids = []\n", shape = oval, color = darkgoldenrod]
Ple[label = "Ple\nIdx in OpGraph: 2\nLifetime = Atomic\nPleOp\nOp = ADDITION\nBlock Config = 16x16\nNum Inputs = 2\nInput Stripe Shapes = [[1, 2, 3, 4], [5, 6, 7, 8]]\nOutput Stripe Shape = [9, 10, 11, 12]\nOutput Data type = U8\nPle kernel Id = ADDITION_16X16_1\nOperation Ids = []\n", shape = oval]
Buffer1[label = "Buffer1\nLifetime = Cascade\nLocation = PleInputSram\nFormat = WEIGHT\nQuant. Info = ZeroPoint = 10, Scale = 0.100000\nTensor shape = [1, 2, 3, 4]\nStripe shape = [5, 6, 7, 8]\nNum. Stripes = 9\nOrder = Zxy\nSize in bytes = 1234\n", shape = box]
}
)";
    REQUIRE(stream.str() == expected);
}

/// Checks SaveEstimatedOpGraphToDot produces the expected output.
/// We test only the low detail version, because the implementation of SaveEstimatedOpGraphToDot shares a lot of the
/// same code that is tested elsewhere, so we are only really interested in testing the grouping into passes and the
/// display of the pass performance stats.
TEST_CASE("SaveEstimatedOpGraphToDot", "[Visualisation]")
{
    // Build a simple graph with two cascaded PleOps, which we then create a fake EstimatedOpGraph struct to describe.
    // Include a EstimateOnlyOp at the end which we will exclude from the EstimatedOpGraph, to test the case where some Ops
    // aren't in a Pass.
    OpGraph graph;

    Buffer inputBuffer(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, { 1, 2, 3, 4 }, { 5, 6, 7, 8 },
                       TraversalOrder::Xyz, 1, QuantizationInfo(0, 1.0f));
    inputBuffer.m_DebugTag = "InputBuffer";
    graph.AddBuffer(&inputBuffer);

    PleOp ple1(Lifetime::Atomic, PleOperation::ADDITION, { 16u, 16u }, 2, { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } },
               { 9, 10, 11, 12 }, ethosn::command_stream::DataType::U8);
    ple1.m_DebugTag = "Ple1";
    graph.AddOp(&ple1);

    Buffer intermediateBuffer(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, { 1, 2, 3, 4 },
                              { 5, 6, 7, 8 }, TraversalOrder::Xyz, 1, QuantizationInfo(0, 1.0f));
    intermediateBuffer.m_DebugTag = "IntermediateBuffer";
    graph.AddBuffer(&intermediateBuffer);

    PleOp ple2(Lifetime::Atomic, PleOperation::ADDITION, { 16u, 16u }, 2, { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } },
               { 9, 10, 11, 12 }, ethosn::command_stream::DataType::U8);
    ple2.m_DebugTag = "Ple2";
    graph.AddOp(&ple2);

    Buffer outputBuffer(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, { 1, 2, 3, 4 }, { 5, 6, 7, 8 },
                        TraversalOrder::Xyz, 1, QuantizationInfo(0, 1.0f));
    outputBuffer.m_DebugTag = "OutputBuffer";
    graph.AddBuffer(&outputBuffer);

    EstimateOnlyOp dma("No reason");
    dma.m_DebugTag = "EstimateOnly";
    graph.AddOp(&dma);

    graph.AddConsumer(&inputBuffer, &ple1, 0);
    graph.SetProducer(&intermediateBuffer, &ple1);
    graph.AddConsumer(&intermediateBuffer, &ple2, 0);
    graph.SetProducer(&outputBuffer, &ple2);
    graph.AddConsumer(&outputBuffer, &dma, 0);

    // Create EstimatedOpGraph describing this graph being partitioned into two Passes that have been estimated,
    // with some dummy figures
    EstimatedOpGraph estimatedOpGraph;
    PassPerformanceData pass1;
    pass1.m_Stats.m_Ple.m_NumOfPatches = 10;
    estimatedOpGraph.m_PerfData.m_Stream.push_back(pass1);
    PassPerformanceData pass2;
    pass2.m_Stats.m_Ple.m_NumOfPatches = 20;
    estimatedOpGraph.m_PerfData.m_Stream.push_back(pass2);
    estimatedOpGraph.m_OpToPass[&ple1] = 0;
    estimatedOpGraph.m_OpToPass[&ple2] = 1;

    // For easier debugging of this test (and so that you can see the pretty graph!), dump to a file
    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("SaveEstimatedOpGraphToDot.dot");
        SaveEstimatedOpGraphToDot(graph, estimatedOpGraph, stream, DetailLevel::Low);
    }

    // Save to a string and check against expected result
    std::stringstream stream;
    SaveEstimatedOpGraphToDot(graph, estimatedOpGraph, stream, DetailLevel::Low);

    std::string expected =
        R"(digraph SupportLibraryGraph
{
subgraph clusterPass0
{
label="Pass0"
labeljust=l
fontsize = 56
Ple1[label = "Ple1", shape = oval]
InputBuffer[label = "InputBuffer", shape = box, color = blue]
Pass0_Perf[label = "{\l    \"OperationIds\": [ ],\l    \"ParentIds\": [],\l    \"Input\":\l    {\l        \"DramParallelBytes\": 0,\l        \"DramNonParallelBytes\": 0,\l        \"SramBytes\": 0,\l        \"NumCentralStripes\": 0,\l        \"NumBoundaryStripes\": 0,\l        \"NumReloads\": 0\l    },\l    \"Output\":\l    {\l        \"DramParallelBytes\": 0,\l        \"DramNonParallelBytes\": 0,\l        \"SramBytes\": 0,\l        \"NumCentralStripes\": 0,\l        \"NumBoundaryStripes\": 0,\l        \"NumReloads\": 0\l    },\l    \"Weights\":\l    {\l        \"DramParallelBytes\": 0,\l        \"DramNonParallelBytes\": 0,\l        \"SramBytes\": 0,\l        \"NumCentralStripes\": 0,\l        \"NumBoundaryStripes\": 0,\l        \"NumReloads\": 0,\l        \"CompressionSavings\": 0\l    },\l    \"Mce\":\l    {\l        \"Operations\": 0,\l        \"CycleCount\": 0\l    },\l    \"Ple\":\l    {\l        \"NumOfPatches\": 10,\l        \"Operation\": 0\l    }\l}\l", shape = note]
}
subgraph clusterPass1
{
label="Pass1"
labeljust=l
fontsize = 56
Ple2[label = "Ple2", shape = oval]
Pass1_Perf[label = "{\l    \"OperationIds\": [ ],\l    \"ParentIds\": [],\l    \"Input\":\l    {\l        \"DramParallelBytes\": 0,\l        \"DramNonParallelBytes\": 0,\l        \"SramBytes\": 0,\l        \"NumCentralStripes\": 0,\l        \"NumBoundaryStripes\": 0,\l        \"NumReloads\": 0\l    },\l    \"Output\":\l    {\l        \"DramParallelBytes\": 0,\l        \"DramNonParallelBytes\": 0,\l        \"SramBytes\": 0,\l        \"NumCentralStripes\": 0,\l        \"NumBoundaryStripes\": 0,\l        \"NumReloads\": 0\l    },\l    \"Weights\":\l    {\l        \"DramParallelBytes\": 0,\l        \"DramNonParallelBytes\": 0,\l        \"SramBytes\": 0,\l        \"NumCentralStripes\": 0,\l        \"NumBoundaryStripes\": 0,\l        \"NumReloads\": 0,\l        \"CompressionSavings\": 0\l    },\l    \"Mce\":\l    {\l        \"Operations\": 0,\l        \"CycleCount\": 0\l    },\l    \"Ple\":\l    {\l        \"NumOfPatches\": 20,\l        \"Operation\": 0\l    }\l}\l", shape = note]
}
EstimateOnly[label = "EstimateOnly", shape = oval]
IntermediateBuffer[label = "IntermediateBuffer", shape = box, color = blue]
OutputBuffer[label = "OutputBuffer", shape = box, color = blue]
InputBuffer -> Ple1
Ple1 -> IntermediateBuffer
IntermediateBuffer -> Ple2
Ple2 -> OutputBuffer
OutputBuffer -> EstimateOnly
}
)";

    REQUIRE(stream.str() == expected);
}

/// Checks SaveGraphOfPartsToDot produces the expected output, focusing on the overall graph topology (connections
/// between nodes and parts) rather than on the details given for each individual Part.
TEST_CASE("SaveGraphOfPartsToDot Graph Topology", "[Visualisation]")
{
    // Build an arbitrary graph, making sure to demonstrate multiple inputs and multiple consumers.
    //
    /// I1 \       / M ------- O1
    ///     \     /          /
    ///      --- S --------D ---- O2
    /// I2 /                    /
    ///                       I3
    ///
    DebuggableObject::ms_IdCounter = 0;    // Reset counter so we get deterministic results

    GraphOfParts graph;

    auto& parts = graph.m_Parts;

    auto i1             = std::make_unique<MockPart>(graph.GeneratePartId());
    auto i2             = std::make_unique<MockPart>(graph.GeneratePartId());
    auto s              = std::make_unique<MockPart>(graph.GeneratePartId());
    auto m              = std::make_unique<MockPart>(graph.GeneratePartId());
    auto d              = std::make_unique<MockPart>(graph.GeneratePartId());
    auto o1             = std::make_unique<MockPart>(graph.GeneratePartId());
    auto o2             = std::make_unique<MockPart>(graph.GeneratePartId());
    auto i3             = std::make_unique<MockPart>(graph.GeneratePartId());
    const BasePart& pi1 = *i1;
    const BasePart& pi2 = *i2;
    const BasePart& ps  = *s;
    const BasePart& pm  = *m;
    const BasePart& pd  = *d;
    const BasePart& po1 = *o1;
    const BasePart& po2 = *o2;
    const BasePart& pi3 = *i3;
    parts.push_back(std::move(i1));
    parts.push_back(std::move(i2));
    parts.push_back(std::move(s));
    parts.push_back(std::move(m));
    parts.push_back(std::move(d));
    parts.push_back(std::move(o1));
    parts.push_back(std::move(o2));
    parts.push_back(std::move(i3));

    PartOutputSlot i1Output  = { pi1.GetPartId(), 0 };
    PartOutputSlot i2Output  = { pi2.GetPartId(), 0 };
    PartInputSlot sInput0    = { ps.GetPartId(), 0 };
    PartInputSlot sInput1    = { ps.GetPartId(), 1 };
    PartOutputSlot sOutput0  = { ps.GetPartId(), 0 };
    PartOutputSlot sOutput1  = { ps.GetPartId(), 1 };
    PartInputSlot mInput0    = { pm.GetPartId(), 0 };
    PartOutputSlot mOutput0  = { pm.GetPartId(), 0 };
    PartInputSlot dInput0    = { pd.GetPartId(), 0 };
    PartOutputSlot dOutput0  = { pd.GetPartId(), 0 };
    PartOutputSlot dOutput1  = { pd.GetPartId(), 1 };
    PartInputSlot o1Input0   = { po1.GetPartId(), 0 };
    PartInputSlot o1Input1   = { po1.GetPartId(), 1 };
    PartInputSlot o2Input0   = { po2.GetPartId(), 0 };
    PartInputSlot o2Input1   = { po2.GetPartId(), 1 };
    PartOutputSlot i3Output0 = { pi3.GetPartId(), 0 };

    graph.m_Connections[sInput0]  = i1Output;
    graph.m_Connections[sInput1]  = i2Output;
    graph.m_Connections[dInput0]  = sOutput0;
    graph.m_Connections[mInput0]  = sOutput1;
    graph.m_Connections[o1Input0] = mOutput0;
    graph.m_Connections[o1Input1] = dOutput0;
    graph.m_Connections[o2Input0] = dOutput1;
    graph.m_Connections[o2Input1] = i3Output0;

    DebuggableObject::ms_IdCounter = 0;    // Reset counter so we get deterministic results

    const CompilationOptions compOpt;

    // For easier debugging of this test (and so that you can see the pretty graph!), dump to a file
    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("GraphOfParts Graph Topology.dot");
        SaveGraphOfPartsToDot(graph, stream, DetailLevel::Low);
    }

    // Save to a string and check against expected result
    std::stringstream stream;
    SaveGraphOfPartsToDot(graph, stream, DetailLevel::Low);

    std::string expected =
        R"(digraph SupportLibraryGraph
{
BasePart_0[label = "BasePart 0"]
BasePart_1[label = "BasePart 1"]
BasePart_2[label = "BasePart 2"]
BasePart_3[label = "BasePart 3"]
BasePart_4[label = "BasePart 4"]
BasePart_5[label = "BasePart 5"]
BasePart_6[label = "BasePart 6"]
BasePart_7[label = "BasePart 7"]
BasePart_0 -> BasePart_2[ headlabel="Slot 0"]
BasePart_1 -> BasePart_2[ headlabel="Slot 1"]
BasePart_2 -> BasePart_3[ taillabel="Slot 1"]
BasePart_2 -> BasePart_4[ taillabel="Slot 0"]
BasePart_3 -> BasePart_5[ headlabel="Slot 0"]
BasePart_4 -> BasePart_5[ taillabel="Slot 0"][ headlabel="Slot 1"]
BasePart_4 -> BasePart_6[ taillabel="Slot 1"][ headlabel="Slot 0"]
BasePart_7 -> BasePart_6[ headlabel="Slot 1"]
}
)";

    REQUIRE(stream.str() == expected);
}

/// Checks SaveGraphOfPartsToDot produces the expected output, focusing on the details given for each individual Part
/// rather than the overall graph topology (connections between parts).
TEST_CASE("SaveGraphOfPartsToDot Part Details", "[Visualisation]")
{
    const std::set<uint32_t> correspondingOperationIds;
    const EstimationOptions estOpt;
    const CompilationOptions compOpt;
    const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);

    // Build a simple graph of disconnected parts, to check the details are printed correctly for each one.
    GraphOfParts parts;
    DebuggableObject::ms_IdCounter = 0;    // Reset counter so we get deterministic results

    // FusedPlePart
    auto fusedPlePart = std::make_unique<FusedPlePart>(
        1, TensorShape{ 1, 2, 3, 4 }, TensorShape{ 5, 6, 7, 8 }, QuantizationInfo(9, 10.0f),
        QuantizationInfo(11, 12.0f), PleOperation::DOWNSAMPLE_2X2, support_library::utils::ShapeMultiplier{ 1, 2, 3 },
        estOpt, compOpt, caps, std::set<uint32_t>{ 13, 14, 15 }, ethosn::command_stream::DataType::U8);
    parts.m_Parts.push_back(std::move(fusedPlePart));

    // McePart
    McePart::ConstructionParams params(estOpt, compOpt, caps);
    params.m_Id                     = 5;
    params.m_InputTensorShape       = TensorShape{ 1, 2, 3, 4 };
    params.m_OutputTensorShape      = TensorShape{ 5, 6, 7, 8 };
    params.m_InputQuantizationInfo  = QuantizationInfo(9, 10.0f);
    params.m_OutputQuantizationInfo = QuantizationInfo(11, 12.0f);
    params.m_WeightsInfo            = sl::TensorInfo(TensorShape{ 9, 10, 11, 12 }, sl::DataType::UINT8_QUANTIZED,
                                          sl::DataFormat::NHWC, QuantizationInfo(11, 12.0f));
    params.m_WeightsData            = std::vector<uint8_t>();
    params.m_BiasInfo               = sl::TensorInfo(TensorShape{ 19, 110, 111, 112 }, sl::DataType::UINT8_QUANTIZED,
                                       sl::DataFormat::NHWC, QuantizationInfo(111, 112.0f));
    params.m_BiasData               = std::vector<int32_t>{};
    params.m_Stride                 = Stride{ 2, 2 };
    params.m_PadTop                 = 1;
    params.m_PadLeft                = 3;
    params.m_Op                     = MceOperation::DEPTHWISE_CONVOLUTION;
    params.m_OperationIds           = std::set<uint32_t>{ 13, 14, 15 };
    params.m_DataType               = command_stream::DataType::U8;
    params.m_UpscaleFactor          = 3;
    params.m_UpsampleType           = command_stream::UpsampleType::NEAREST_NEIGHBOUR;
    auto mcePart                    = std::make_unique<McePart>(std::move(params));
    parts.m_Parts.push_back(std::move(mcePart));

    // ConcatPart
    auto concatPart = std::make_unique<ConcatPart>(
        2, std::vector<sl::TensorInfo>{ TensorShape{ 1, 2, 3, 4 } }, ConcatenationInfo(3, QuantizationInfo(9, 10.0f)),
        CompilerDataFormat::NHWCB, std::set<uint32_t>{ 13, 14, 15 }, estOpt, compOpt, caps);
    parts.m_Parts.push_back(std::move(concatPart));

    // InputPart
    auto inputPart =
        std::make_unique<InputPart>(3, TensorShape{ 1, 2, 3, 4 }, CompilerDataFormat::NHWCB, QuantizationInfo(9, 10.0f),
                                    std::set<uint32_t>{ 13, 14, 15 }, estOpt, compOpt, caps);

    parts.m_Parts.push_back(std::move(inputPart));

    // OutputPart
    auto outputPart = std::make_unique<OutputPart>(5, TensorShape{ 1, 2, 3, 4 }, CompilerDataFormat::NHWCB,
                                                   QuantizationInfo(9, 10.0f), std::set<uint32_t>{ 13, 14, 15 }, estOpt,
                                                   compOpt, caps);
    parts.m_Parts.push_back(std::move(outputPart));

    // ReshapePart
    auto reshapePart = std::make_unique<ReshapePart>(8, TensorShape{ 1, 2, 3, 4 }, TensorShape{ 5, 6, 7, 8 },
                                                     CompilerDataFormat::NHWCB, QuantizationInfo(9, 10.0f),
                                                     std::set<uint32_t>{ 13, 14, 15 }, estOpt, compOpt, caps);
    parts.m_Parts.push_back(std::move(reshapePart));

    // Standalone PLE part
    auto standalonePlePart = std::make_unique<StandalonePlePart>(
        9, std::vector<TensorShape>{ TensorShape{ 1, 2, 3, 4 }, TensorShape{ 1, 2, 3, 4 } }, TensorShape{ 1, 2, 3, 4 },
        std::vector<QuantizationInfo>{ QuantizationInfo(9, 10.0f), QuantizationInfo(9, 10.0f) },
        QuantizationInfo(9, 10.0f), ethosn::command_stream::PleOperation::ADDITION, estOpt, compOpt, caps,
        std::set<uint32_t>{ 1 }, ethosn::command_stream::DataType::U8);
    parts.m_Parts.push_back(std::move(standalonePlePart));

    // For easier debugging of this test (and so that you can see the pretty graph!), dump to a file
    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("GraphOfParts Part Details.dot");
        SaveGraphOfPartsToDot(parts, stream, DetailLevel::High);
    }

    // Save to a string and check against expected result
    std::stringstream stream;
    SaveGraphOfPartsToDot(parts, stream, DetailLevel::High);

    std::string expected =
        R"(digraph SupportLibraryGraph
{
BasePart_0[label = "FusedPlePart: BasePart 0\nPartId = 1\nCompilerDataFormat = NONE\nCorrespondingOperationIds = [13, 14, 15]\nInputTensorShape = [1, 2, 3, 4]\nOutputTensorShape = [5, 6, 7, 8]\nInputQuantizationInfo = ZeroPoint = 9, Scale = 10.000000\nOutputQuantizationInfo = ZeroPoint = 11, Scale = 12.000000\nKernelOperation = DOWNSAMPLE_2X2\nShapeMultiplier = [1/1, 2/1, 3/1]\nStripeGenerator.MceInputTensorShape = [1, 2, 3, 4]\nStripeGenerator.MceOutputTensorShape = [1, 2, 3, 4]\nStripeGenerator.PleOutputTensorShape = [5, 6, 7, 8]\nStripeGenerator.KernelHeight = 1\nStripeGenerator.KernelWidth = 1\nStripeGenerator.Stride = 1, 1\nStripeGenerator.UpscaleFactor = 1\nStripeGenerator.Operation = DEPTHWISE_CONVOLUTION\nStripeGenerator.MceShapeMultiplier = [1/1, 1/1, 1/1]\nStripeGenerator.PleShapeMultiplier = [1/1, 2/1, 3/1]\n"]
BasePart_1[label = "McePart: BasePart 1\nPartId = 5\nCompilerDataFormat = NONE\nCorrespondingOperationIds = [13, 14, 15]\nInputTensorShape = [1, 2, 3, 4]\nOutputTensorShape = [5, 6, 7, 8]\nInputQuantizationInfo = ZeroPoint = 9, Scale = 10.000000\nOutputQuantizationInfo = ZeroPoint = 11, Scale = 12.000000\nWeightsInfo = ([9, 10, 11, 12], UINT8_QUANTIZED, NHWC, ZeroPoint = 11, Scale = 12.000000)\nBiasInfo = ([19, 110, 111, 112], UINT8_QUANTIZED, NHWC, ZeroPoint = 111, Scale = 112.000000)\nStride = 2, 2\nUpscaleFactor = 3\nUpsampleType = NEAREST_NEIGHBOUR\nPadTop = 1\nPadLeft = 3\nOperation = DEPTHWISE_CONVOLUTION\nStripeGenerator.MceInputTensorShape = [1, 2, 3, 4]\nStripeGenerator.MceOutputTensorShape = [5, 6, 7, 8]\nStripeGenerator.PleOutputTensorShape = [5, 6, 7, 8]\nStripeGenerator.KernelHeight = 9\nStripeGenerator.KernelWidth = 10\nStripeGenerator.Stride = 2, 2\nStripeGenerator.UpscaleFactor = 3\nStripeGenerator.Operation = DEPTHWISE_CONVOLUTION\nStripeGenerator.MceShapeMultiplier = [3/1, 3/1, 1/1]\nStripeGenerator.PleShapeMultiplier = [1/1, 1/1, 1/1]\n"]
BasePart_2[label = "ConcatPart: BasePart 2\nPartId = 2\nCompilerDataFormat = NHWCB\nCorrespondingOperationIds = [13, 14, 15]\nInputTensorsInfo = [([1, 2, 3, 4], UINT8_QUANTIZED, NHWC, ZeroPoint = 0, Scale = 1.000000)]\nConcatInfo.Axis = 3\nConcatInfo.OutputQuantInfo = ZeroPoint = 9, Scale = 10.000000\n"]
BasePart_3[label = "InputPart: BasePart 3\nPartId = 3\nCompilerDataFormat = NHWCB\nCorrespondingOperationIds = [13, 14, 15]\nOutputTensorShape = [1, 2, 3, 4]\nOutputQuantizationInfo = ZeroPoint = 9, Scale = 10.000000\n"]
BasePart_4[label = "OutputPart: BasePart 4\nPartId = 5\nCompilerDataFormat = NHWCB\nCorrespondingOperationIds = [13, 14, 15]\nInputTensorShape = [1, 2, 3, 4]\nInputQuantizationInfo = ZeroPoint = 9, Scale = 10.000000\n"]
BasePart_5[label = "ReshapePart: BasePart 5\nPartId = 8\nCompilerDataFormat = NHWCB\nCorrespondingOperationIds = [13, 14, 15]\nInputTensorShape = [1, 2, 3, 4]\nOutputTensorShape = [5, 6, 7, 8]\nOutputQuantizationInfo = ZeroPoint = 9, Scale = 10.000000\n"]
BasePart_6[label = "StandalonePlePart: BasePart 6\nPartId = 9\nCompilerDataFormat = NONE\nCorrespondingOperationIds = [1]\nInputTensorShape = [[1, 2, 3, 4], [1, 2, 3, 4]]\nOutputTensorShape = [1, 2, 3, 4]\nInputQuantizationInfo = [ZeroPoint = 9, Scale = 10.000000, ZeroPoint = 9, Scale = 10.000000]\nOutputQuantizationInfo = ZeroPoint = 9, Scale = 10.000000\n"]
}
)";

    REQUIRE(stream.str() == expected);
}

/// Checks SavePlansToDot produces the expected output, focusing on the overall graph topology (connections
/// between nodes and parts) rather than on the details given for each individual node.
/// Details of each node are covered by the "SaveOpGraphToDot Node Details" test
TEST_CASE("SavePlansToDot Graph Topology", "[Visualisation]")
{
    DebuggableObject::ms_IdCounter = 0;    // Reset counter so we get deterministic results

    // Create simple graph
    Graph graph;
    NameOnlyNode* nodeA = graph.CreateAndAddNode<NameOnlyNode>("a");
    NameOnlyNode* nodeB = graph.CreateAndAddNode<NameOnlyNode>("b");
    graph.Connect(nodeA, nodeB);

    // Generate two plans for the node. These plans are not realistic at all.
    PartOutputSlot planAOutputSlot = PartOutputSlot{ 0, 0 };
    OwnedOpGraph planAOpGraph;
    planAOpGraph.AddBuffer(std::make_unique<Buffer>());
    Plan planA(PartInputMapping{}, PartOutputMapping{ { planAOpGraph.GetBuffers()[0], planAOutputSlot } });
    planA.m_OpGraph = std::move(planAOpGraph);

    OwnedOpGraph planBOpGraph;
    PartInputSlot planBInputSlot   = PartInputSlot{ 1, 0 };
    PartOutputSlot planBOutputSlot = PartOutputSlot{ 1, 0 };
    planBOpGraph.AddBuffer(std::make_unique<Buffer>());
    planBOpGraph.AddOp(std::make_unique<DmaOp>());
    planBOpGraph.AddBuffer(std::make_unique<Buffer>());
    planBOpGraph.AddConsumer(planBOpGraph.GetBuffers()[0], planBOpGraph.GetOps()[0], 0);
    planBOpGraph.SetProducer(planBOpGraph.GetBuffers()[1], planBOpGraph.GetOps()[0]);
    Plan planB(PartInputMapping{ { planBOpGraph.GetBuffers()[0], planBInputSlot } },
               PartOutputMapping{ { planBOpGraph.GetBuffers()[1], planBOutputSlot } });
    planB.m_OpGraph = std::move(planBOpGraph);

    const CompilationOptions compOpt;

    Plans plans;
    plans.push_back(std::move(planA));
    plans.push_back(std::move(planB));

    // For easier debugging of this test (and so that you can see the pretty graph!), dump to a file
    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("SavePlansToDot Graph Topology.dot");
        SavePlansToDot(plans, stream, DetailLevel::Low);
    }

    // Save to a string and check against expected result
    std::stringstream stream;
    SavePlansToDot(plans, stream, DetailLevel::Low);

    std::string expected =
        R"(digraph SupportLibraryGraph
{
subgraph clusterPlan_1
{
label="Plan 1"
labeljust=l
Buffer_0[label = "Buffer 0", shape = box, color = brown]
OutputLabelBuffer_0[label = "Output Slot 0", shape = box]
Buffer_0 -> OutputLabelBuffer_0[dir = back, arrowtail = box]
}
subgraph clusterPlan_5
{
label="Plan 5"
labeljust=l
DmaOp_3[label = "DmaOp 3", shape = oval, color = darkgoldenrod]
Buffer_2[label = "Buffer 2", shape = box, color = brown]
Buffer_4[label = "Buffer 4", shape = box, color = brown]
Buffer_2 -> DmaOp_3
DmaOp_3 -> Buffer_4
InputLabelBuffer_2[label = "Input Slot 0", shape = box]
InputLabelBuffer_2 -> Buffer_2[arrowhead = box]
OutputLabelBuffer_4[label = "Output Slot 0", shape = box]
Buffer_4 -> OutputLabelBuffer_4[dir = back, arrowtail = box]
}
}
)";

    REQUIRE(stream.str() == expected);
}

/// Checks SaveCombinationToDot produces the expected output, focusing on the overall graph topology (connections
/// between nodes, parts and glues) rather than on the details given for each individual node.
/// Details of each node are covered by other tests.
///
/// The topology of the Combination is chosen to test cases including:
///   * Plans without any inputs (A)
///   * Plans without any outputs (F, G)
///   * Two plans being connected via a glue (A -> BC)
///   * Two plans being connected without a glue (BC -> DE)
///   * A part having two plans using its output, each with a different glue (DE -> F/G)
///   * Two plans being connected by two different glues (for two different connections) (DE -> G)
///   * A chain of plans containing just a single buffer each, each of which "reinterprets" its input to output (B -> C)
///
///  ( A ) -> g -> ( B ) -> ( C ) -> ( D ) ---> g -> ( F )
///                               \  (   ) \'
///                                | (   )  \-> g -> (   )
///                                | (   )           ( G )
///                                \-( E ) -->  g -> (   )
TEST_CASE("SaveCombinationToDot Graph Topology", "[Visualisation]")
{
    DebuggableObject::ms_IdCounter = 0;    // Reset counter so we get deterministic results

    GraphOfParts graph;
    auto& parts       = graph.m_Parts;
    auto& connections = graph.m_Connections;

    auto pA         = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB         = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pC         = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pDE        = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pF         = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pG         = std::make_unique<MockPart>(graph.GeneratePartId());
    PartId partAId  = pA->GetPartId();
    PartId partBId  = pB->GetPartId();
    PartId partCId  = pC->GetPartId();
    PartId partDEId = pDE->GetPartId();
    PartId partFId  = pF->GetPartId();
    PartId partGId  = pG->GetPartId();
    parts.push_back(std::move(pA));
    parts.push_back(std::move(pB));
    parts.push_back(std::move(pC));
    parts.push_back(std::move(pDE));
    parts.push_back(std::move(pF));
    parts.push_back(std::move(pG));

    PartOutputSlot partAOutputSlot0 = { partAId, 0 };

    PartInputSlot partBInputSlot0   = { partBId, 0 };
    PartOutputSlot partBOutputSlot0 = { partBId, 0 };

    PartInputSlot partCInputSlot0   = { partCId, 0 };
    PartOutputSlot partCOutputSlot0 = { partCId, 0 };

    PartInputSlot partDEInputSlot0   = { partDEId, 0 };
    PartInputSlot partDEInputSlot1   = { partDEId, 1 };
    PartOutputSlot partDEOutputSlot0 = { partDEId, 0 };
    PartOutputSlot partDEOutputSlot1 = { partDEId, 1 };

    PartInputSlot partFInputSlot0 = { partFId, 0 };

    PartInputSlot partGInputSlot0 = { partGId, 0 };
    PartInputSlot partGInputSlot1 = { partGId, 1 };

    connections[partBInputSlot0]  = partAOutputSlot0;
    connections[partCInputSlot0]  = partBOutputSlot0;
    connections[partDEInputSlot0] = partCOutputSlot0;
    connections[partDEInputSlot1] = partCOutputSlot0;
    connections[partFInputSlot0]  = partDEOutputSlot0;
    connections[partGInputSlot0]  = partDEOutputSlot0;
    connections[partGInputSlot1]  = partDEOutputSlot1;

    Plan planA;
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 0, QuantizationInfo()));
    planA.m_OpGraph.GetBuffers().back()->m_DebugTag = "InputDram";
    planA.m_OutputMappings                          = { { planA.m_OpGraph.GetBuffers()[0], partAOutputSlot0 } };

    // Glue between A and B
    Glue glueA_BC;
    glueA_BC.m_Graph.AddOp(std::make_unique<DmaOp>());
    glueA_BC.m_Graph.GetOps()[0]->m_DebugTag = "InputDma";
    glueA_BC.m_InputSlot                     = { glueA_BC.m_Graph.GetOps()[0], 0 };
    glueA_BC.m_Output.push_back(glueA_BC.m_Graph.GetOps()[0]);

    // Part consisting of node B
    Plan planB;
    planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planB.m_OpGraph.GetBuffers().back()->m_DebugTag = "InputSram1";
    planB.m_InputMappings                           = { { planB.m_OpGraph.GetBuffers()[0], partBInputSlot0 } };
    planB.m_OutputMappings                          = { { planB.m_OpGraph.GetBuffers()[0], partBOutputSlot0 } };

    // Part consisting of node C
    Plan planC;
    planC.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planC.m_OpGraph.GetBuffers().back()->m_DebugTag = "InputSram2";
    planC.m_InputMappings                           = { { planC.m_OpGraph.GetBuffers()[0], partCInputSlot0 } };
    planC.m_OutputMappings                          = { { planC.m_OpGraph.GetBuffers()[0], partCOutputSlot0 } };

    // Part consisting of nodes D and E
    Plan planDE;
    planDE.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                        TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                        TraversalOrder::Xyz, 4, QuantizationInfo()));
    planDE.m_OpGraph.GetBuffers().back()->m_DebugTag = "IntermediateSramInput1";
    planDE.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                        TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                        TraversalOrder::Xyz, 0, QuantizationInfo()));
    planDE.m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputSram1";
    planDE.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                        TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                        TraversalOrder::Xyz, 4, QuantizationInfo()));
    planDE.m_OpGraph.GetBuffers().back()->m_DebugTag = "IntermediateSramInput2";
    planDE.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                        TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                        TraversalOrder::Xyz, 0, QuantizationInfo()));
    planDE.m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputSram2";
    planDE.m_InputMappings                           = { { planDE.m_OpGraph.GetBuffers()[0], partDEInputSlot0 },
                               { planDE.m_OpGraph.GetBuffers()[2], partDEInputSlot1 } };
    planDE.m_OutputMappings                          = { { planDE.m_OpGraph.GetBuffers()[1], partDEOutputSlot0 },
                                { planDE.m_OpGraph.GetBuffers()[3], partDEOutputSlot1 } };
    planDE.m_OpGraph.AddOp(
        std::make_unique<MceOp>(Lifetime::Atomic, MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
                                BlockConfig{ 16u, 16u }, TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                TensorShape{ 1, 1, 1, 16 }, TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));
    planDE.m_OpGraph.GetOps()[0]->m_DebugTag = "Mce2";
    planDE.m_OpGraph.AddConsumer(planDE.m_OpGraph.GetBuffers()[0], planDE.m_OpGraph.GetOps()[0], 0);
    planDE.m_OpGraph.AddConsumer(planDE.m_OpGraph.GetBuffers()[2], planDE.m_OpGraph.GetOps()[0], 1);
    planDE.m_OpGraph.SetProducer(planDE.m_OpGraph.GetBuffers()[1], planDE.m_OpGraph.GetOps()[0]);
    planDE.m_OpGraph.SetProducer(planDE.m_OpGraph.GetBuffers()[3], planDE.m_OpGraph.GetOps()[0]);

    // Glue between D and F
    Glue glueD_F;
    glueD_F.m_Graph.AddOp(std::make_unique<DmaOp>());
    glueD_F.m_Graph.GetOps()[0]->m_DebugTag = "OutputDma1";
    glueD_F.m_InputSlot                     = { glueD_F.m_Graph.GetOps()[0], 0 };
    glueD_F.m_Output.push_back(glueD_F.m_Graph.GetOps()[0]);

    // Glue between D and G
    Glue glueD_G;
    glueD_G.m_Graph.AddOp(std::make_unique<DmaOp>());
    glueD_G.m_Graph.GetOps()[0]->m_DebugTag = "OutputDma2";
    glueD_G.m_InputSlot                     = { glueD_G.m_Graph.GetOps()[0], 0 };
    glueD_G.m_Output.push_back(glueD_G.m_Graph.GetOps()[0]);

    // Glue between E and G
    Glue glueE_G;
    glueE_G.m_Graph.AddOp(std::make_unique<DmaOp>());
    glueE_G.m_Graph.GetOps()[0]->m_DebugTag = "OutputDma3";
    glueE_G.m_InputSlot                     = { glueE_G.m_Graph.GetOps()[0], 0 };
    glueE_G.m_Output.push_back(glueE_G.m_Graph.GetOps()[0]);

    // Part consisting of node F
    Plan planF;
    planF.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 0, QuantizationInfo()));
    planF.m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputDram1";
    planF.m_InputMappings                           = { { planF.m_OpGraph.GetBuffers()[0], partFInputSlot0 } };

    // Part consisting of node G
    Plan planG;
    planG.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 0, QuantizationInfo()));
    planG.m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputDram2";
    planG.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 0, QuantizationInfo()));
    planG.m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputDram3";
    planG.m_InputMappings                           = { { planG.m_OpGraph.GetBuffers()[0], partGInputSlot0 },
                              { planG.m_OpGraph.GetBuffers()[1], partGInputSlot1 } };

    // Create Combination with all the plans and glues
    Combination comb;

    Elem elemA  = { std::make_shared<Plan>(std::move(planA)), { { partBInputSlot0, { &glueA_BC, true } } } };
    Elem elemB  = { std::make_shared<Plan>(std::move(planB)), {} };
    Elem elemC  = { std::make_shared<Plan>(std::move(planC)), {} };
    Elem elemDE = { std::make_shared<Plan>(std::move(planDE)),
                    { { partFInputSlot0, { &glueD_F, true } },
                      { partGInputSlot0, { &glueD_G, true } },
                      { partGInputSlot1, { &glueE_G, true } } } };
    Elem elemF  = { std::make_shared<Plan>(std::move(planF)), {} };
    Elem elemG  = { std::make_shared<Plan>(std::move(planG)), {} };
    comb.m_Elems.insert(std::make_pair(0, elemA));
    comb.m_PartIdsInOrder.push_back(0);
    comb.m_Elems.insert(std::make_pair(1, elemB));
    comb.m_PartIdsInOrder.push_back(1);
    comb.m_Elems.insert(std::make_pair(2, elemC));
    comb.m_PartIdsInOrder.push_back(2);
    comb.m_Elems.insert(std::make_pair(3, elemDE));
    comb.m_PartIdsInOrder.push_back(3);
    comb.m_Elems.insert(std::make_pair(4, elemF));
    comb.m_PartIdsInOrder.push_back(4);
    comb.m_Elems.insert(std::make_pair(5, elemG));
    comb.m_PartIdsInOrder.push_back(5);

    // For easier debugging of this test (and so that you can see the pretty graph!), dump to a file
    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("SaveCombinationToDot Graph Topology.dot");
        SaveCombinationToDot(comb, graph, stream, DetailLevel::Low);
    }

    // Save to a string and check against expected result
    std::stringstream stream;
    SaveCombinationToDot(comb, graph, stream, DetailLevel::Low);

    std::string expected =
        R"(digraph SupportLibraryGraph
{
subgraph clusterPlan_6
{
label="Plan 6"
labeljust=l
InputDram[label = "InputDram", shape = box, color = brown]
}
subgraph clusterPlan_6_Glue_0
{
label="Plan 6 Glue 0"
labeljust=l
InputDma[label = "InputDma", shape = oval, color = darkgoldenrod]
}
InputDram -> InputDma
subgraph clusterPlan_9
{
label="Plan 9"
labeljust=l
InputSram1[label = "InputSram1", shape = box, color = blue]
}
InputDma -> InputSram1
subgraph clusterPlan_11
{
label="Plan 11"
labeljust=l
InputSram2[label = "InputSram2", shape = box, color = blue]
}
InputSram1 -> InputSram2
subgraph clusterPlan_13
{
label="Plan 13"
labeljust=l
Mce2[label = "Mce2", shape = oval]
IntermediateSramInput1[label = "IntermediateSramInput1", shape = box, color = blue]
OutputSram1[label = "OutputSram1", shape = box, color = blue]
IntermediateSramInput2[label = "IntermediateSramInput2", shape = box, color = blue]
OutputSram2[label = "OutputSram2", shape = box, color = blue]
IntermediateSramInput1 -> Mce2[ label="Input 0"]
Mce2 -> OutputSram1
IntermediateSramInput2 -> Mce2[ label="Input 1"]
Mce2 -> OutputSram2
{ rank = "same"; Mce2; IntermediateSramInput2; }
}
InputSram2 -> IntermediateSramInput1
InputSram2 -> IntermediateSramInput2
subgraph clusterPlan_13_Glue_0
{
label="Plan 13 Glue 0"
labeljust=l
OutputDma1[label = "OutputDma1", shape = oval, color = darkgoldenrod]
}
OutputSram1 -> OutputDma1
subgraph clusterPlan_13_Glue_1
{
label="Plan 13 Glue 1"
labeljust=l
OutputDma2[label = "OutputDma2", shape = oval, color = darkgoldenrod]
}
OutputSram1 -> OutputDma2
subgraph clusterPlan_13_Glue_2
{
label="Plan 13 Glue 2"
labeljust=l
OutputDma3[label = "OutputDma3", shape = oval, color = darkgoldenrod]
}
OutputSram2 -> OutputDma3
subgraph clusterPlan_22
{
label="Plan 22"
labeljust=l
OutputDram1[label = "OutputDram1", shape = box, color = brown]
}
OutputDma1 -> OutputDram1
subgraph clusterPlan_24
{
label="Plan 24"
labeljust=l
OutputDram2[label = "OutputDram2", shape = box, color = brown]
OutputDram3[label = "OutputDram3", shape = box, color = brown]
}
OutputDma2 -> OutputDram2
OutputDma3 -> OutputDram3
}
)";

    REQUIRE(stream.str() == expected);
}

// Create graph:
//
//
//   - - > C
//  |
//  A - -> B
//  |
//   -- >  D
//
//  AB -- SRAM to SRAM
//  AC -- SRAM to SRAM
//  AD -- SRAM to DRAM
TEST_CASE("SaveCombinationBranchToDot", "[Visualisation]")
{
    DebuggableObject::ms_IdCounter = 0;    // Reset counter so we get deterministic results

    GraphOfParts graph;
    auto& parts       = graph.m_Parts;
    auto& connections = graph.m_Connections;

    auto pA = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pC = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pD = std::make_unique<MockPart>(graph.GeneratePartId());

    BasePart& partA = *pA;
    BasePart& partB = *pB;
    BasePart& partC = *pC;
    BasePart& partD = *pD;

    parts.push_back(std::move(pA));
    parts.push_back(std::move(pB));
    parts.push_back(std::move(pC));
    parts.push_back(std::move(pD));

    PartOutputSlot partAOutputSlot = { partA.GetPartId(), 0 };

    PartInputSlot partBInputSlot = { partB.GetPartId(), 0 };
    PartInputSlot partCInputSlot = { partC.GetPartId(), 0 };
    PartInputSlot partDInputSlot = { partD.GetPartId(), 0 };

    connections[partBInputSlot] = { partAOutputSlot };
    connections[partCInputSlot] = { partAOutputSlot };
    connections[partDInputSlot] = { partAOutputSlot };

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(&compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Plan planA;
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planA.m_OutputMappings = { { planA.m_OpGraph.GetBuffers()[0], partAOutputSlot } };

    Plan planB;
    planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planB.m_InputMappings = { { planB.m_OpGraph.GetBuffers()[0], partBInputSlot } };

    Plan planC;
    planC.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planC.m_InputMappings = { { planC.m_OpGraph.GetBuffers()[0], partCInputSlot } };

    Plan planD;
    planD.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planD.m_InputMappings = { { planD.m_OpGraph.GetBuffers()[0], partDInputSlot } };

    Combination combA(partA, std::move(planA), 0, graph);
    Combination combB(partB, std::move(planB), 1, graph);
    Combination combC(partC, std::move(planC), 2, graph);
    Combination combD(partD, std::move(planD), 3, graph);

    // Merge the combinations
    Combination comb = combB + combD + combC + combA;

    REQUIRE(combA.m_PartIdsInOrder[0] == 0);
    REQUIRE(combA.m_HeadOrderRank == 0);
    REQUIRE(combB.m_PartIdsInOrder[0] == 1);
    REQUIRE(combB.m_HeadOrderRank == 1);
    REQUIRE(combC.m_PartIdsInOrder[0] == 2);
    REQUIRE(combC.m_HeadOrderRank == 2);
    REQUIRE(combD.m_PartIdsInOrder[0] == 3);
    REQUIRE(combD.m_HeadOrderRank == 3);
    REQUIRE(comb.m_PartIdsInOrder[0] == 0);
    REQUIRE(comb.m_HeadOrderRank == 0);

    Combiner combiner(graph, hwCaps, estOpt, debuggingContext);

    std::vector<PartConnection> destPartEdge;

    // Part B and the edge that connects to its source Part A
    PartConnection edgeA2B = graph.GetConnectionsBetween(partA.GetPartId(), partB.GetPartId()).at(0);
    destPartEdge.push_back(edgeA2B);
    // Part C and the edge that connects to its source Part A
    PartConnection edgeA2C = graph.GetConnectionsBetween(partA.GetPartId(), partC.GetPartId()).at(0);
    destPartEdge.push_back(edgeA2C);
    // Part D and the edge that connects to its source Part A
    PartConnection edgeA2D = graph.GetConnectionsBetween(partA.GetPartId(), partD.GetPartId()).at(0);
    destPartEdge.push_back(edgeA2D);

    Combination combGlued = combiner.GluePartToCombinationSrcToDests(partA, comb, destPartEdge);

    // One glue shared by A-B, A-C (SRAM - SRAM) and A-D (SRAM - DRAM)
    // The glue has (1) 1 x input DMA (2) DRAM buffer (3) 2 x ouput DMA
    REQUIRE(combGlued.m_Elems.size() == 4);

    // For easier debugging of this test (and so that you can see the pretty graph!), dump to a file
    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("SaveCombinationToDot Graph Topology.dot");
        SaveCombinationToDot(combGlued, graph, stream, DetailLevel::Low);
    }

    // Save to a string and check against expected result
    std::stringstream stream;
    SaveCombinationToDot(combGlued, graph, stream, DetailLevel::Low);

    std::string expected =
        R"(digraph SupportLibraryGraph
{
subgraph clusterPlan_4
{
label="Plan 4"
labeljust=l
Buffer_5[label = "Buffer 5", shape = box, color = blue]
}
subgraph clusterPlan_4_Glue_0
{
label="Plan 4 Glue 0"
labeljust=l
DmaOp_13[label = "DmaOp 13", shape = oval, color = darkgoldenrod]
DmaOp_14[label = "DmaOp 14", shape = oval, color = darkgoldenrod]
DmaOp_15[label = "DmaOp 15", shape = oval, color = darkgoldenrod]
Buffer_12[label = "Buffer 12", shape = box, color = brown]
DmaOp_13 -> Buffer_12
Buffer_12 -> DmaOp_14
Buffer_12 -> DmaOp_15
}
Buffer_5 -> DmaOp_13
subgraph clusterPlan_6
{
label="Plan 6"
labeljust=l
Buffer_7[label = "Buffer 7", shape = box, color = blue]
}
DmaOp_14 -> Buffer_7
subgraph clusterPlan_10
{
label="Plan 10"
labeljust=l
Buffer_11[label = "Buffer 11", shape = box, color = brown]
}
Buffer_12 -> Buffer_11
subgraph clusterPlan_8
{
label="Plan 8"
labeljust=l
Buffer_9[label = "Buffer 9", shape = box, color = blue]
}
DmaOp_15 -> Buffer_9
}
)";

    REQUIRE(stream.str() == expected);
}
