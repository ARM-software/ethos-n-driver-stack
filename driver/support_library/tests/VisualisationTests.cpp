//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "GraphNodes.hpp"
#include "TestUtils.hpp"
#include "cascading/CombinerDFS.hpp"
#include "cascading/Estimation.hpp"
#include "cascading/Part.hpp"
#include "cascading/Plan.hpp"
#include "cascading/Visualisation.hpp"

#include <catch.hpp>
#include <fstream>

using namespace ethosn::support_library;
namespace sl = ethosn::support_library;
using namespace ethosn::command_stream;

/// Checks SaveNetworkToDot produces the expected output, focusing on the overall network topology (connections
/// between operations) rather than on the details given for each individual operation.
TEST_CASE("SaveNetworkToDot Network Topology", "[Visualisation]")
{
    // Build an arbitrary network, making sure to demonstrate multiple inputs, multiple outputs and multiple consumers.
    Network network(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO));

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
Operation1[label = "1: Output\n", shape = oval]
Operand0_0 -> Operation1
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
Operation6[label = "6: TransposeConvolution\nWeights: 2\nBias: 1\n", shape = oval]
Operand0_0 -> Operation6
Operand6_0[label = "Operand\nShape = [1, 33, 33, 32]\nFormat = NHWC\nType = UINT8_QUANTIZED\nQuant. info = ZeroPoint = 0, Scale = 1.000000\n", shape = box]
Operation6 -> Operand6_0
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
              { 5, 6, 7, 8 }, { 9, 10, 11, 12 }, TraversalOrder::Zxy, Stride(10, 20), 30, 40);
    mce.m_DebugTag = "Mce";
    graph.AddOp(&mce);

    DmaOp dma(Lifetime::Cascade, Location::Sram);
    dma.m_DebugTag = "Dma";
    graph.AddOp(&dma);

    PleOp ple(Lifetime::Atomic, PleOperation::ADDITION, { 3u, 4u }, 2, { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } },
              { 9, 10, 11, 12 });
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
Mce[label = "Mce\nLifetime = Atomic\nMceOp\nOp = CONVOLUTION\nAlgo = DIRECT\nBlock Config = 3x4\nInput Stripe Shape = [1, 2, 3, 4]\nOutput Stripe Shape = [5, 6, 7, 8]\nWeights Stripe Shape = [9, 10, 11, 12]\nOrder = Zxy\nStride = 10, 20\nPad L/T = 30, 40\nOperation Ids = []\n", shape = oval]
Dma[label = "Dma\nLifetime = Cascade\nDmaOp\nLocation = Sram\nOperation Ids = []\n", shape = oval, color = darkgoldenrod]
Ple[label = "Ple\nLifetime = Atomic\nPleOp\nOp = ADDITION\nBlock Config = 3x4\nNum Inputs = 2\nInput Stripe Shapes = [[1, 2, 3, 4], [5, 6, 7, 8]]\nOutput Stripe Shape = [9, 10, 11, 12]\nOperation Ids = []\n", shape = oval]
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
    // Include a DmaOp at the end which we will exclude from the EstimatedOpGraph, to test the case where some Ops
    // haven't been estimated.
    OpGraph graph;

    Buffer inputBuffer(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, { 1, 2, 3, 4 }, { 5, 6, 7, 8 },
                       TraversalOrder::Xyz, 1, QuantizationInfo(0, 1.0f));
    inputBuffer.m_DebugTag = "InputBuffer";
    graph.AddBuffer(&inputBuffer);

    PleOp ple1(Lifetime::Atomic, PleOperation::ADDITION, { 3u, 4u }, 2, { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } },
               { 9, 10, 11, 12 });
    ple1.m_DebugTag = "Ple1";
    graph.AddOp(&ple1);

    Buffer intermediateBuffer(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, { 1, 2, 3, 4 },
                              { 5, 6, 7, 8 }, TraversalOrder::Xyz, 1, QuantizationInfo(0, 1.0f));
    intermediateBuffer.m_DebugTag = "IntermediateBuffer";
    graph.AddBuffer(&intermediateBuffer);

    PleOp ple2(Lifetime::Atomic, PleOperation::ADDITION, { 3u, 4u }, 2, { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } },
               { 9, 10, 11, 12 });
    ple2.m_DebugTag = "Ple2";
    graph.AddOp(&ple2);

    PleOp ple(Lifetime::Atomic, PleOperation::ADDITION, { 3u, 4u }, 2, { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } },
              { 9, 10, 11, 12 });
    ple.m_DebugTag = "Ple";
    graph.AddOp(&ple);

    Buffer outputBuffer(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, { 1, 2, 3, 4 }, { 5, 6, 7, 8 },
                        TraversalOrder::Xyz, 1, QuantizationInfo(0, 1.0f));
    outputBuffer.m_DebugTag = "OutputBuffer";
    graph.AddBuffer(&outputBuffer);

    DmaOp dma(Lifetime::Atomic, Location::Dram);
    dma.m_DebugTag = "Dma";
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
    estimatedOpGraph.m_UnestimatedOps  = { &dma };

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
labelloc = "t"
label="1 Op(s) were unestimated!!"
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
Dma[label = "Dma", shape = oval, color = darkgoldenrod]
IntermediateBuffer[label = "IntermediateBuffer", shape = box, color = blue]
OutputBuffer[label = "OutputBuffer", shape = box, color = blue]
InputBuffer -> Ple1
Ple1 -> IntermediateBuffer
IntermediateBuffer -> Ple2
Ple2 -> OutputBuffer
OutputBuffer -> Dma
}
)";

    REQUIRE(stream.str() == expected);
}

/// Simple Node type for tests.
/// Includes a friendly name and ignores shape, quantisation info etc. so that tests
/// can focus on graph topology.
class NameOnlyNode : public Node
{
public:
    NameOnlyNode(NodeId id, std::string name)
        : Node(id,
               TensorShape(),
               sl::DataType::UINT8_QUANTIZED,
               QuantizationInfo(),
               CompilerDataFormat::NONE,
               std::set<uint32_t>{ 0 })
        , m_Name(name)
    {}

    bool IsPrepared() override
    {
        return false;
    }

    std::string m_Name;
};

/// Checks SaveGraphToDot produces the expected output, focusing on the overall graph topology (connections
/// between nodes and parts) rather than on the details given for each individual node.
TEST_CASE("SaveGraphToDot Graph Topology", "[Visualisation]")
{
    // Build an arbitrary graph, making sure to demonstrate multiple inputs and multiple consumers.
    //
    /// I1 \       / M ------- O1
    ///     \     /          /
    ///      --- S --------D ---- O2
    /// I2 /                    /
    ///                       I3
    DebuggableObject::ms_IdCounter = 0;    // Reset counter so we get deterministic results

    Graph g;
    NameOnlyNode* i1 = g.CreateAndAddNode<NameOnlyNode>("I1");
    NameOnlyNode* i2 = g.CreateAndAddNode<NameOnlyNode>("I2");
    NameOnlyNode* s  = g.CreateAndAddNode<NameOnlyNode>("S");
    NameOnlyNode* m  = g.CreateAndAddNode<NameOnlyNode>("M");
    NameOnlyNode* d  = g.CreateAndAddNode<NameOnlyNode>("D");
    NameOnlyNode* o1 = g.CreateAndAddNode<NameOnlyNode>("O1");
    NameOnlyNode* o2 = g.CreateAndAddNode<NameOnlyNode>("O2");
    NameOnlyNode* i3 = g.CreateAndAddNode<NameOnlyNode>("I3");

    g.Connect(i1, s, 0);
    g.Connect(i2, s, 1);
    g.Connect(s, m);
    g.Connect(m, o1, 0);
    g.Connect(s, d);
    g.Connect(d, o1, 1);
    g.Connect(d, o2, 0);
    g.Connect(i3, o2, 1);

    const EstimationOptions estOpt;
    const CompilationOptions compOpt;
    // Assign some nodes into Parts. Note we don't assign all nodes to a part, so we can test that works correctly.
    auto part1        = std::make_unique<Part>(0, estOpt, compOpt,
                                        GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));
    part1->m_SubGraph = { i1, i2 };
    auto part2        = std::make_unique<Part>(1, estOpt, compOpt,
                                        GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));
    part2->m_SubGraph = { m, o1, d };
    GraphOfParts parts;
    parts.m_Parts.push_back(std::move(part1));
    parts.m_Parts.push_back(std::move(part2));

    // For easier debugging of this test (and so that you can see the pretty graph!), dump to a file
    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("SaveGraphToDot Graph Topology.dot");
        SaveGraphToDot(g, &parts, stream, DetailLevel::Low);
    }

    // Save to a string and check against expected result
    std::stringstream stream;
    SaveGraphToDot(g, &parts, stream, DetailLevel::Low);

    std::string expected =
        R"(digraph SupportLibraryGraph
{
subgraph clusterPart_0
{
label="Part 0"
labeljust=l
0[label = "Node 0\n", shape = oval]
1[label = "Node 1\n", shape = oval]
}
subgraph clusterPart_1
{
label="Part 1"
labeljust=l
3[label = "Node 3\n", shape = oval]
5[label = "Node 5\n", shape = oval]
4[label = "Node 4\n", shape = oval]
}
2[label = "Node 2\n", shape = oval]
6[label = "Node 6\n", shape = oval]
7[label = "Node 7\n", shape = oval]
0 -> 2[ label="Input 0"]
1 -> 2[ label="Input 1"]
2 -> 3
3 -> 5[ label="Input 0"]
2 -> 4
4 -> 5[ label="Input 1"]
4 -> 6[ label="Input 0"]
7 -> 6[ label="Input 1"]
}
)";

    REQUIRE(stream.str() == expected);
}

/// Checks SaveGraphToDot produces the expected output, focusing on the details given for each individual node
/// rather than the overall graph topology (connections between nodes).
TEST_CASE("SaveGraphToDot Node Details", "[Visualisation]")
{
    const EstimationOptions estOpt;
    const CompilationOptions compOpt;
    const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);

    // Build a simple graph of disconnected nodes, to check the details are printed correctly for each one.
    DebuggableObject::ms_IdCounter = 0;    // Reset counter so we get deterministic results

    Graph g;
    InputNode* i        = g.CreateAndAddNode<InputNode>(TensorShape{ 1, 2, 3, 4 }, std::set<uint32_t>{ 1 });
    MceOperationNode* m = g.CreateAndAddNode<MceOperationNode>(
        TensorShape(), TensorShape{ 5, 6, 7, 8 }, sl::DataType::UINT8_QUANTIZED, QuantizationInfo(),
        ethosn::support_library::TensorInfo(), std::vector<uint8_t>(), ethosn::support_library::TensorInfo(),
        std::vector<int32_t>(), Stride(), 0, 0, ethosn::command_stream::MceOperation::FULLY_CONNECTED,
        CompilerDataFormat::NHWCB, std::set<uint32_t>{ 2 });

    // Arbitrarily Put all nodes into one part
    auto part1        = std::make_unique<Part>(0, estOpt, compOpt, caps);
    part1->m_SubGraph = { i, m };
    GraphOfParts parts;
    parts.m_Parts.push_back(std::move(part1));

    // For easier debugging of this test (and so that you can see the pretty graph!), dump to a file
    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("SaveGraphToDot Node Details.dot");
        SaveGraphToDot(g, &parts, stream, DetailLevel::High);
    }

    // Save to a string and check against expected result
    std::stringstream stream;
    SaveGraphToDot(g, &parts, stream, DetailLevel::High);

    std::string expected =
        R"(digraph SupportLibraryGraph
{
subgraph clusterPart_0
{
label="Part 0"
labeljust=l
0[label = "Node 0\nInputNode\nCorrespondingOperationIds: 1\nShape = [1, 2, 3, 4]\nFormat = NHWC\nCompressedFormat = NONE\n", shape = oval]
1[label = "Node 1\nMceOperationNode\nFULLY_CONNECTED\nCorrespondingOperationIds: 2\nShape = [5, 6, 7, 8]\nFormat = NHWCB\nCompressedFormat = NONE\n", shape = oval]
}
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
    OwnedOpGraph planAOpGraph;
    planAOpGraph.AddBuffer(std::make_unique<Buffer>());
    auto planA =
        std::make_unique<Plan>(Plan::InputMapping{}, Plan::OutputMapping{ { planAOpGraph.GetBuffers()[0], nodeB } });
    planA->m_OpGraph = std::move(planAOpGraph);

    OwnedOpGraph planBOpGraph;
    planBOpGraph.AddBuffer(std::make_unique<Buffer>());
    planBOpGraph.AddOp(std::make_unique<DmaOp>());
    planBOpGraph.AddBuffer(std::make_unique<Buffer>());
    planBOpGraph.AddConsumer(planBOpGraph.GetBuffers()[0], planBOpGraph.GetOps()[0], 0);
    planBOpGraph.SetProducer(planBOpGraph.GetBuffers()[1], planBOpGraph.GetOps()[0]);
    auto planB = std::make_unique<Plan>(Plan::InputMapping{ { planBOpGraph.GetBuffers()[0], nodeB->GetInput(0) } },
                                        Plan::OutputMapping{ { planBOpGraph.GetBuffers()[1], nodeB } });
    planB->m_OpGraph = std::move(planBOpGraph);

    const EstimationOptions estOpt;
    const CompilationOptions compOpt;
    const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
    Part part(0, estOpt, compOpt, caps);
    part.m_SubGraph.push_back(nodeA);
    part.m_SubGraph.push_back(nodeB);
    part.m_Plans.push_back(std::move(planA));
    part.m_Plans.push_back(std::move(planB));

    // For easier debugging of this test (and so that you can see the pretty graph!), dump to a file
    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("SavePlansToDot Graph Topology.dot");
        SavePlansToDot(part, stream, DetailLevel::Low);
    }

    // Save to a string and check against expected result
    std::stringstream stream;
    SavePlansToDot(part, stream, DetailLevel::Low);

    std::string expected =
        R"(digraph SupportLibraryGraph
{
subgraph clusterPlan_1
{
label="Plan 1"
labeljust=l
Buffer_0[label = "Buffer 0", shape = box, color = brown]
OutputLabelBuffer_0[label = "Output from Node 1
", shape = box]
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
InputLabelBuffer_2[label = "Input from Node 0
", shape = box]
InputLabelBuffer_2 -> Buffer_2[arrowhead = box]
OutputLabelBuffer_4[label = "Output from Node 1
", shape = box]
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
///   * Plans without any inputs
///   * Plans without any outputs
///   * Two plans being connected via a glue
///   * Two plans being connected without a glue
///   * A plan having two plans using its output, each with a different glue.
///   * Two plans being connected by two different glues (for two different connections)
///
///  ( A ) -> g -> ( BC ) -> ( D ) ---> g -> ( F )
///                       \  (   ) \'
///                        | (   )  \-> g -> (   )
///                        | (   )           ( G )
///                        \-( E ) -->  g -> (   )
TEST_CASE("SaveCombinationToDot Graph Topology", "[Visualisation]")
{
    DebuggableObject::ms_IdCounter = 0;    // Reset counter so we get deterministic results

    Graph graph;
    NameOnlyNode* nodeA = graph.CreateAndAddNode<NameOnlyNode>("a");
    NameOnlyNode* nodeB = graph.CreateAndAddNode<NameOnlyNode>("b");
    NameOnlyNode* nodeC = graph.CreateAndAddNode<NameOnlyNode>("c");
    NameOnlyNode* nodeD = graph.CreateAndAddNode<NameOnlyNode>("d");
    NameOnlyNode* nodeE = graph.CreateAndAddNode<NameOnlyNode>("e");
    NameOnlyNode* nodeF = graph.CreateAndAddNode<NameOnlyNode>("f");
    NameOnlyNode* nodeG = graph.CreateAndAddNode<NameOnlyNode>("g");

    graph.Connect(nodeA, nodeB, 0);
    graph.Connect(nodeB, nodeC, 0);
    graph.Connect(nodeC, nodeD, 0);
    graph.Connect(nodeC, nodeE, 0);
    graph.Connect(nodeD, nodeF, 0);
    graph.Connect(nodeD, nodeG, 0);
    graph.Connect(nodeE, nodeG, 1);

    GraphOfParts parts;

    const EstimationOptions estOpt;
    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
    PartId partId                     = 0;

    // Part consisting of node A
    parts.m_Parts.push_back(std::make_unique<Part>(partId, estOpt, compOpt, hwCaps));
    ++partId;
    parts.m_Parts.back()->m_SubGraph.push_back(nodeA);
    std::unique_ptr<Plan> planA = std::make_unique<Plan>();
    planA->m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB,
                                                        TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                        TraversalOrder::Xyz, 0, QuantizationInfo()));
    planA->m_OpGraph.GetBuffers().back()->m_DebugTag = "InputDram";
    planA->m_OutputMappings                          = { { planA->m_OpGraph.GetBuffers()[0], nodeA } };
    parts.m_Parts.back()->m_Plans.push_back(std::move(planA));

    // Glue between A and BC
    Glue glueA_BC;
    glueA_BC.m_Graph.AddOp(std::make_unique<DmaOp>());
    glueA_BC.m_Graph.GetOps()[0]->m_DebugTag = "InputDma";
    glueA_BC.m_InputSlot                     = { glueA_BC.m_Graph.GetOps()[0], 0 };
    glueA_BC.m_Output                        = glueA_BC.m_Graph.GetOps()[0];

    // Part consisting of nodes B and C
    parts.m_Parts.push_back(std::make_unique<Part>(partId, estOpt, compOpt, hwCaps));
    ++partId;
    parts.m_Parts.back()->m_SubGraph.push_back(nodeB);
    parts.m_Parts.back()->m_SubGraph.push_back(nodeC);
    std::unique_ptr<Plan> planBC = std::make_unique<Plan>();
    planBC->m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                         TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                         TraversalOrder::Xyz, 4, QuantizationInfo()));
    planBC->m_OpGraph.GetBuffers().back()->m_DebugTag = "InputSram";
    planBC->m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                         TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                         TraversalOrder::Xyz, 0, QuantizationInfo()));
    planBC->m_OpGraph.GetBuffers().back()->m_DebugTag = "IntermediateSramOutput";
    planBC->m_InputMappings                           = { { planBC->m_OpGraph.GetBuffers()[0], nodeB->GetInput(0) } };
    planBC->m_OutputMappings                          = { { planBC->m_OpGraph.GetBuffers()[1], nodeC } };
    planBC->m_OpGraph.AddOp(std::make_unique<MceOp>(Lifetime::Atomic, MceOperation::CONVOLUTION,
                                                    CompilerMceAlgorithm::Direct, BlockConfig{ 16u, 16u },
                                                    TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                    TensorShape{ 1, 1, 1, 16 }, TraversalOrder::Xyz, Stride(), 0, 0));
    planBC->m_OpGraph.GetOps()[0]->m_DebugTag = "Mce1";
    planBC->m_OpGraph.AddConsumer(planBC->m_OpGraph.GetBuffers()[0], planBC->m_OpGraph.GetOps()[0], 0);
    planBC->m_OpGraph.SetProducer(planBC->m_OpGraph.GetBuffers()[1], planBC->m_OpGraph.GetOps()[0]);
    parts.m_Parts.back()->m_Plans.push_back(std::move(planBC));

    // Part consisting of nodes D and E
    parts.m_Parts.push_back(std::make_unique<Part>(partId, estOpt, compOpt, hwCaps));
    ++partId;
    parts.m_Parts.back()->m_SubGraph.push_back(nodeD);
    parts.m_Parts.back()->m_SubGraph.push_back(nodeE);
    std::unique_ptr<Plan> planDE = std::make_unique<Plan>();
    planDE->m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                         TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                         TraversalOrder::Xyz, 4, QuantizationInfo()));
    planDE->m_OpGraph.GetBuffers().back()->m_DebugTag = "IntermediateSramInput1";
    planDE->m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                         TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                         TraversalOrder::Xyz, 0, QuantizationInfo()));
    planDE->m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputSram1";
    planDE->m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                         TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                         TraversalOrder::Xyz, 4, QuantizationInfo()));
    planDE->m_OpGraph.GetBuffers().back()->m_DebugTag = "IntermediateSramInput2";
    planDE->m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                         TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                         TraversalOrder::Xyz, 0, QuantizationInfo()));
    planDE->m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputSram2";
    planDE->m_InputMappings                           = { { planDE->m_OpGraph.GetBuffers()[0], nodeD->GetInput(0) },
                                { planDE->m_OpGraph.GetBuffers()[2], nodeE->GetInput(0) } };
    planDE->m_OutputMappings                          = { { planDE->m_OpGraph.GetBuffers()[1], nodeD },
                                 { planDE->m_OpGraph.GetBuffers()[3], nodeE } };
    planDE->m_OpGraph.AddOp(std::make_unique<MceOp>(Lifetime::Atomic, MceOperation::CONVOLUTION,
                                                    CompilerMceAlgorithm::Direct, BlockConfig{ 16u, 16u },
                                                    TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                    TensorShape{ 1, 1, 1, 16 }, TraversalOrder::Xyz, Stride(), 0, 0));
    planDE->m_OpGraph.GetOps()[0]->m_DebugTag = "Mce2";
    planDE->m_OpGraph.AddConsumer(planDE->m_OpGraph.GetBuffers()[0], planDE->m_OpGraph.GetOps()[0], 0);
    planDE->m_OpGraph.AddConsumer(planDE->m_OpGraph.GetBuffers()[2], planDE->m_OpGraph.GetOps()[0], 1);
    planDE->m_OpGraph.SetProducer(planDE->m_OpGraph.GetBuffers()[1], planDE->m_OpGraph.GetOps()[0]);
    planDE->m_OpGraph.SetProducer(planDE->m_OpGraph.GetBuffers()[3], planDE->m_OpGraph.GetOps()[0]);
    parts.m_Parts.back()->m_Plans.push_back(std::move(planDE));

    // Glue between D and F
    Glue glueD_F;
    glueD_F.m_Graph.AddOp(std::make_unique<DmaOp>());
    glueD_F.m_Graph.GetOps()[0]->m_DebugTag = "OutputDma1";
    glueD_F.m_InputSlot                     = { glueD_F.m_Graph.GetOps()[0], 0 };
    glueD_F.m_Output                        = glueD_F.m_Graph.GetOps()[0];

    // Glue between D and G
    Glue glueD_G;
    glueD_G.m_Graph.AddOp(std::make_unique<DmaOp>());
    glueD_G.m_Graph.GetOps()[0]->m_DebugTag = "OutputDma2";
    glueD_G.m_InputSlot                     = { glueD_G.m_Graph.GetOps()[0], 0 };
    glueD_G.m_Output                        = glueD_G.m_Graph.GetOps()[0];

    // Glue between E and G
    Glue glueE_G;
    glueE_G.m_Graph.AddOp(std::make_unique<DmaOp>());
    glueE_G.m_Graph.GetOps()[0]->m_DebugTag = "OutputDma3";
    glueE_G.m_InputSlot                     = { glueE_G.m_Graph.GetOps()[0], 0 };
    glueE_G.m_Output                        = glueE_G.m_Graph.GetOps()[0];

    // Part consisting of node F
    parts.m_Parts.push_back(std::make_unique<Part>(partId, estOpt, compOpt, hwCaps));
    ++partId;
    parts.m_Parts.back()->m_SubGraph.push_back(nodeF);
    std::unique_ptr<Plan> planF = std::make_unique<Plan>();
    planF->m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB,
                                                        TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                        TraversalOrder::Xyz, 0, QuantizationInfo()));
    planF->m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputDram1";
    planF->m_InputMappings                           = { { planF->m_OpGraph.GetBuffers()[0], nodeF->GetInput(0) } };
    parts.m_Parts.back()->m_Plans.push_back(std::move(planF));

    // Part consisting of node G
    parts.m_Parts.push_back(std::make_unique<Part>(partId, estOpt, compOpt, hwCaps));
    parts.m_Parts.back()->m_SubGraph.push_back(nodeG);
    std::unique_ptr<Plan> planG = std::make_unique<Plan>();
    planG->m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB,
                                                        TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                        TraversalOrder::Xyz, 0, QuantizationInfo()));
    planG->m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputDram2";
    planG->m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB,
                                                        TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                        TraversalOrder::Xyz, 0, QuantizationInfo()));
    planG->m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputDram3";
    planG->m_InputMappings                           = { { planG->m_OpGraph.GetBuffers()[0], nodeG->GetInput(0) },
                               { planG->m_OpGraph.GetBuffers()[1], nodeG->GetInput(1) } };
    parts.m_Parts.back()->m_Plans.push_back(std::move(planG));

    // Create Combination with all the plans and glues
    Combination comb;
    Elem elemA  = { 0, { { nodeB->GetInput(0), { &glueA_BC } } } };
    Elem elemBC = { 0, {} };
    Elem elemDE = { 0,
                    { { nodeF->GetInput(0), { &glueD_F } },
                      { nodeG->GetInput(0), { &glueD_G } },
                      { nodeG->GetInput(1), { &glueE_G } } } };
    Elem elemF  = { 0, {} };
    Elem elemG  = { 0, {} };
    comb.m_Elems.insert(std::make_pair(0, elemA));
    comb.m_Elems.insert(std::make_pair(1, elemBC));
    comb.m_Elems.insert(std::make_pair(2, elemDE));
    comb.m_Elems.insert(std::make_pair(3, elemF));
    comb.m_Elems.insert(std::make_pair(4, elemG));

    // For easier debugging of this test (and so that you can see the pretty graph!), dump to a file
    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("SaveCombinationToDot Graph Topology.dot");
        SaveCombinationToDot(comb, parts, stream, DetailLevel::Low);
    }

    // Save to a string and check against expected result
    std::stringstream stream;
    SaveCombinationToDot(comb, parts, stream, DetailLevel::Low);

    std::string expected =
        R"(digraph SupportLibraryGraph
{
subgraph clusterPlan_1
{
label="Plan 1"
labeljust=l
InputDram[label = "InputDram", shape = box, color = brown]
}
subgraph clusterPlan_1_Glue_0
{
label="Plan 1 Glue 0"
labeljust=l
InputDma[label = "InputDma", shape = oval, color = darkgoldenrod]
}
InputDram -> InputDma
subgraph clusterPlan_5
{
label="Plan 5"
labeljust=l
Mce1[label = "Mce1", shape = oval]
InputSram[label = "InputSram", shape = box, color = blue]
IntermediateSramOutput[label = "IntermediateSramOutput", shape = box, color = blue]
InputSram -> Mce1
Mce1 -> IntermediateSramOutput
}
InputDma -> InputSram
subgraph clusterPlan_10
{
label="Plan 10"
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
IntermediateSramOutput -> IntermediateSramInput1
IntermediateSramOutput -> IntermediateSramInput2
subgraph clusterPlan_10_Glue_0
{
label="Plan 10 Glue 0"
labeljust=l
OutputDma1[label = "OutputDma1", shape = oval, color = darkgoldenrod]
}
OutputSram1 -> OutputDma1
subgraph clusterPlan_10_Glue_1
{
label="Plan 10 Glue 1"
labeljust=l
OutputDma2[label = "OutputDma2", shape = oval, color = darkgoldenrod]
}
OutputSram1 -> OutputDma2
subgraph clusterPlan_10_Glue_2
{
label="Plan 10 Glue 2"
labeljust=l
OutputDma3[label = "OutputDma3", shape = oval, color = darkgoldenrod]
}
OutputSram2 -> OutputDma3
subgraph clusterPlan_20
{
label="Plan 20"
labeljust=l
OutputDram1[label = "OutputDram1", shape = box, color = brown]
}
OutputDma1 -> OutputDram1
subgraph clusterPlan_23
{
label="Plan 23"
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
