//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "Compiler.hpp"
#include "GraphNodes.hpp"
#include "TestUtils.hpp"
#include "cascading/CascadingCommandStreamGenerator.hpp"
#include "cascading/CombinerDFS.hpp"
#include "cascading/ConcatPart.hpp"
#include "cascading/ConstantPart.hpp"
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
namespace sl       = ethosn::support_library;
namespace utils    = ethosn::support_library::utils;
using BlockConfig  = ethosn::command_stream::BlockConfig;
using MceOperation = ethosn::command_stream::MceOperation;
using PleOperation = ethosn::command_stream::PleOperation;

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
    // This is a rough approximation of what a Plan for convolution might look like, with some added
    // bits to test multiple consumers and producers
    //                                                                                Dma
    //                                                                                 |
    //  Ifm (Dram)     -> Dma -> Ifm (Sram)     - \                                    v       /-> Consumer 1
    //                                             ->  Mce -> Ofm (Sram) -> Dma -> Ofm (Dram)
    //  Weights (Dram) -> Dma -> Weights (Sram) - /                                            \-> Consumer 2
    //
    OpGraph graph;

    Buffer dramIfm;
    dramIfm.m_DebugTag = "Dram Ifm";
    DmaOp dmaIfm(CascadingBufferFormat::NHWCB);
    dmaIfm.m_DebugTag = "Dma Ifm";
    Buffer sramIfm;
    sramIfm.m_DebugTag = "Sram Ifm";

    Buffer dramWeights;
    dramWeights.m_DebugTag = "Dram Weights";
    DmaOp dmaWeights(CascadingBufferFormat::WEIGHT);
    dmaWeights.m_DebugTag = "Dma Weights";
    Buffer sramWeights;
    sramWeights.m_DebugTag = "Sram Weights";
    sramWeights.m_Format   = CascadingBufferFormat::WEIGHT;

    MceOp mce;
    mce.m_DebugTag = "Mce";

    Buffer sramOfm;
    sramOfm.m_DebugTag = "Sram Ofm";
    DmaOp dmaOfm(CascadingBufferFormat::NHWCB);
    dmaOfm.m_DebugTag = "Dma Ofm";
    DmaOp dmaExtra(CascadingBufferFormat::NHWCB);
    dmaExtra.m_DebugTag = "Dma Extra";
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
    graph.AddOp(&dmaExtra);
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
    graph.AddProducer(&dramOfm, &dmaExtra);
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
Dma_Extra[label = "Dma Extra", shape = oval, color = darkgoldenrod]
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
Dma_Extra -> Dram_Ofm
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

    Buffer buffer1(Location::PleInputSram, CascadingBufferFormat::WEIGHT, { 1, 2, 3, 4 }, { 5, 6, 7, 8 },
                   TraversalOrder::Zxy, 1234, QuantizationInfo(10, 0.1f));
    buffer1.m_DataType   = DataType::INT32_QUANTIZED;
    buffer1.m_NumStripes = 9;
    buffer1.m_DebugTag   = "Buffer1";
    buffer1.m_Offset     = 0;
    buffer1.m_BufferType = BufferType::Intermediate;
    graph.AddBuffer(&buffer1);

    MceOp mce(MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct, { 3u, 4u }, { 1, 2, 3, 4 }, { 5, 6, 7, 8 },
              { 9, 10, 11, 12 }, TraversalOrder::Zxy, Stride(10, 20), 30, 40, 100, 200);
    mce.m_DebugTag      = "Mce";
    mce.m_UpscaleFactor = 2;
    mce.m_UpsampleType  = command_stream::cascading::UpsampleType::NEAREST_NEIGHBOUR;
    graph.AddOp(&mce);

    DmaOp dma(CascadingBufferFormat::NHWCB);
    dma.m_DebugTag = "Dma";
    graph.AddOp(&dma);

    PleOp ple(PleOperation::ADDITION, { 16u, 16u }, 2, { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }, { 9, 10, 11, 12 },
              DataType::UINT8_QUANTIZED, true);
    ple.m_DebugTag         = "Ple";
    ple.m_Offset           = 0;
    ple.m_Input0Multiplier = 10;
    ple.m_Input0Shift      = 11;
    ple.m_Input1Multiplier = 12;
    ple.m_Input1Shift      = 13;
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
Mce[label = "Mce\nIdx in OpGraph: 0\nMceOp\nOp = CONVOLUTION\nAlgo = DIRECT\nBlock Config = 3x4\nInput Stripe Shape = [1, 2, 3, 4]\nOutput Stripe Shape = [5, 6, 7, 8]\nWeights Stripe Shape = [9, 10, 11, 12]\nOrder = Zxy\nStride = 10, 20\nPad L/T = 30, 40\nUpscaleFactor = 2\nUpsampleType = NEAREST_NEIGHBOUR\nLower/Upper Bound = 100, 200\nOperation Ids = []\n", shape = oval]
Dma[label = "Dma\nIdx in OpGraph: 1\nDmaOp\nOperation Ids = []\nTransfer Format = NHWCB\nOffset = [0, 0, 0, 0]\n", shape = oval, color = darkgoldenrod]
Ple[label = "Ple\nIdx in OpGraph: 2\nPleOp\nOp = ADDITION\nBlock Config = 16x16\nNum Inputs = 2\nInput Stripe Shapes = [[1, 2, 3, 4], [5, 6, 7, 8]]\nOutput Stripe Shape = [9, 10, 11, 12]\nPle kernel Id = ADDITION_16X16_1\nKernel Load = 1\nOffset = 0 (0x0)\nOperation Ids = []\nInput0Multiplier = 10\nInput0Shift = 11\nInput1Multiplier = 12\nInput1Shift = 13\n", shape = oval]
Buffer1[label = "Buffer1\nLocation = PleInputSram\nFormat = WEIGHT\nData Type = INT32_QUANTIZED\nQuant. Info = ZeroPoint = 10, Scale = 0.100000\nTensor shape = [1, 2, 3, 4]\nStripe shape = [5, 6, 7, 8]\nNum. Stripes = 9\nOrder = Zxy\nOffset = 0 (0x0)\nSize in bytes = 1234 (0x4D2)\nSlot size in bytes = 0 (0x0)\nType = Intermediate\nPacked boundary thickness = { L: 0, T: 0, R: 0, B: 0}\nNum loads = 1\n", shape = box]
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

    Buffer inputBuffer(Location::Sram, CascadingBufferFormat::NHWCB, { 1, 2, 3, 4 }, { 5, 6, 7, 8 },
                       TraversalOrder::Xyz, 1, QuantizationInfo(0, 1.0f));
    inputBuffer.m_DebugTag = "InputBuffer";
    graph.AddBuffer(&inputBuffer);

    PleOp ple1(PleOperation::ADDITION, { 16u, 16u }, 2, { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }, { 9, 10, 11, 12 },
               DataType::UINT8_QUANTIZED, true);
    ple1.m_DebugTag = "Ple1";
    graph.AddOp(&ple1);

    Buffer intermediateBuffer(Location::Sram, CascadingBufferFormat::NHWCB, { 1, 2, 3, 4 }, { 5, 6, 7, 8 },
                              TraversalOrder::Xyz, 1, QuantizationInfo(0, 1.0f));
    intermediateBuffer.m_DebugTag = "IntermediateBuffer";
    graph.AddBuffer(&intermediateBuffer);

    PleOp ple2(PleOperation::ADDITION, { 16u, 16u }, 2, { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }, { 9, 10, 11, 12 },
               DataType::UINT8_QUANTIZED, true);
    ple2.m_DebugTag = "Ple2";
    graph.AddOp(&ple2);

    Buffer outputBuffer(Location::Sram, CascadingBufferFormat::NHWCB, { 1, 2, 3, 4 }, { 5, 6, 7, 8 },
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
    estimatedOpGraph.m_Metric = 57.2;
    PassPerformanceData pass1;
    pass1.m_Stats.m_Ple.m_NumOfPatches = 10;
    estimatedOpGraph.m_PerfData.m_Stream.push_back(pass1);
    PassPerformanceData pass2;
    pass2.m_Stats.m_Ple.m_NumOfPatches = 20;
    estimatedOpGraph.m_PerfData.m_Stream.push_back(pass2);
    estimatedOpGraph.m_OpToPass[&ple1] = 0;
    estimatedOpGraph.m_OpToPass[&ple2] = 1;

    std::map<uint32_t, std::string> extraPassDetails  = { { 0, "Extra details for pass 0!" } };
    std::map<Op*, std::string> extraOpDetails         = { { &ple1, "Extra details for Ple1!" } };
    std::map<Buffer*, std::string> extraBufferDetails = { { &inputBuffer, "Extra details for InputBuffer!" } };

    // For easier debugging of this test (and so that you can see the pretty graph!), dump to a file
    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("SaveEstimatedOpGraphToDot.dot");
        SaveEstimatedOpGraphToDot(graph, estimatedOpGraph, stream, DetailLevel::Low, extraPassDetails, extraOpDetails,
                                  extraBufferDetails);
    }

    // Save to a string and check against expected result
    std::stringstream stream;
    SaveEstimatedOpGraphToDot(graph, estimatedOpGraph, stream, DetailLevel::Low, extraPassDetails, extraOpDetails,
                              extraBufferDetails);

    std::string expected =
        R"(digraph SupportLibraryGraph
{
labelloc="t";
label="Total metric = 57.2";
subgraph clusterPass0
{
label="Pass0\nExtra details for pass 0!\n"
labeljust=l
fontsize = 56
Ple1[label = "Ple1", shape = oval]
InputBuffer[label = "InputBuffer", shape = box, color = blue]
Pass0_Perf[label = "Metric = 0\l\l{\l    \"OperationIds\": [ ],\l    \"ParentIds\": [],\l    \"Input\":\l    {\l        \"DramParallelBytes\": 0,\l        \"DramNonParallelBytes\": 0,\l        \"SramBytes\": 0,\l        \"NumCentralStripes\": 0,\l        \"NumBoundaryStripes\": 0,\l        \"NumReloads\": 0\l    },\l    \"Output\":\l    {\l        \"DramParallelBytes\": 0,\l        \"DramNonParallelBytes\": 0,\l        \"SramBytes\": 0,\l        \"NumCentralStripes\": 0,\l        \"NumBoundaryStripes\": 0,\l        \"NumReloads\": 0\l    },\l    \"Weights\":\l    {\l        \"DramParallelBytes\": 0,\l        \"DramNonParallelBytes\": 0,\l        \"SramBytes\": 0,\l        \"NumCentralStripes\": 0,\l        \"NumBoundaryStripes\": 0,\l        \"NumReloads\": 0,\l        \"CompressionSavings\": 0\l    },\l    \"Mce\":\l    {\l        \"Operations\": 0,\l        \"CycleCount\": 0\l    },\l    \"Ple\":\l    {\l        \"NumOfPatches\": 10,\l        \"Operation\": 0\l    }\l}\l", shape = note]
}
subgraph clusterPass1
{
label="Pass1"
labeljust=l
fontsize = 56
Ple2[label = "Ple2", shape = oval]
Pass1_Perf[label = "Metric = 0\l\l{\l    \"OperationIds\": [ ],\l    \"ParentIds\": [],\l    \"Input\":\l    {\l        \"DramParallelBytes\": 0,\l        \"DramNonParallelBytes\": 0,\l        \"SramBytes\": 0,\l        \"NumCentralStripes\": 0,\l        \"NumBoundaryStripes\": 0,\l        \"NumReloads\": 0\l    },\l    \"Output\":\l    {\l        \"DramParallelBytes\": 0,\l        \"DramNonParallelBytes\": 0,\l        \"SramBytes\": 0,\l        \"NumCentralStripes\": 0,\l        \"NumBoundaryStripes\": 0,\l        \"NumReloads\": 0\l    },\l    \"Weights\":\l    {\l        \"DramParallelBytes\": 0,\l        \"DramNonParallelBytes\": 0,\l        \"SramBytes\": 0,\l        \"NumCentralStripes\": 0,\l        \"NumBoundaryStripes\": 0,\l        \"NumReloads\": 0,\l        \"CompressionSavings\": 0\l    },\l    \"Mce\":\l    {\l        \"Operations\": 0,\l        \"CycleCount\": 0\l    },\l    \"Ple\":\l    {\l        \"NumOfPatches\": 20,\l        \"Operation\": 0\l    }\l}\l", shape = note]
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

    // Because we only test with Low detail, we don't see the extra details added for the Op/Buffer (extraOpDetails/extraBufferDetails).
    // Do a smaller follow-up test to check just this:
    std::stringstream stream2;
    SaveEstimatedOpGraphToDot(graph, estimatedOpGraph, stream2, DetailLevel::High, extraPassDetails, extraOpDetails,
                              extraBufferDetails);
    REQUIRE(stream2.str().find("Extra details for Ple1!") != std::string::npos);
    REQUIRE(stream2.str().find("Extra details for InputBuffer!") != std::string::npos);
}

/// Checks SaveCompiledOpGraphToDot produces the expected output.
/// We only test some small details of the output, because the implementation of SaveCompiledOpGraphToDot shares a lot of the
/// same code that is tested above in SaveEstimatedOpGraphToDot, so we are only really interested in testing the
/// agent IDs marked on each Pass and Op, and buffer IDs.
TEST_CASE("SaveCompiledOpGraphToDot", "[Visualisation]")
{
    // Build a very simple graph with two Ops in a Pass, which we then create a fake CompiledOpGraph struct to describe.
    OpGraph graph;

    PleOp ple1(PleOperation::ADDITION, { 16u, 16u }, 2, { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }, { 9, 10, 11, 12 },
               DataType::UINT8_QUANTIZED, true);
    ple1.m_DebugTag = "Ple1";
    graph.AddOp(&ple1);

    PleOp ple2(PleOperation::ADDITION, { 16u, 16u }, 2, { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }, { 9, 10, 11, 12 },
               DataType::UINT8_QUANTIZED, true);
    ple2.m_DebugTag = "Ple2";
    graph.AddOp(&ple2);

    Buffer buffer;
    buffer.m_DebugTag = "Buffer";
    graph.AddBuffer(&buffer);

    // Create CompiledOpGraph describing this graph
    cascading_compiler::CompiledOpGraph compiledOpGraph;
    PassPerformanceData pass1;
    compiledOpGraph.m_EstimatedOpGraph.m_PerfData.m_Stream.push_back(pass1);
    compiledOpGraph.m_EstimatedOpGraph.m_OpToPass[&ple1] = 0;
    compiledOpGraph.m_EstimatedOpGraph.m_OpToPass[&ple2] = 0;
    compiledOpGraph.m_OpToAgentIdMapping                 = { { &ple1, 4 }, { &ple2, 5 } };
    compiledOpGraph.m_BufferIds                          = { { &buffer, 9 } };

    // For easier debugging of this test (and so that you can see the pretty graph!), dump to a file
    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("SaveCompiledOpGraphToDot.dot");
        SaveCompiledOpGraphToDot(graph, compiledOpGraph, stream, DetailLevel::High);
    }

    // Save to a string and check against expected result
    std::stringstream stream;
    SaveCompiledOpGraphToDot(graph, compiledOpGraph, stream, DetailLevel::High);

    CHECK(stream.str().find("Agent IDs: 4 - 5") != std::string::npos);
    CHECK(stream.str().find("Agent ID: 4") != std::string::npos);
    CHECK(stream.str().find("Agent ID: 5") != std::string::npos);
    CHECK(stream.str().find("Buffer ID: 9") != std::string::npos);
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
MockPart_0[label = "MockPart 0"]
MockPart_1[label = "MockPart 1"]
MockPart_2[label = "MockPart 2"]
MockPart_3[label = "MockPart 3"]
MockPart_4[label = "MockPart 4"]
MockPart_5[label = "MockPart 5"]
MockPart_6[label = "MockPart 6"]
MockPart_7[label = "MockPart 7"]
MockPart_0 -> MockPart_2[ headlabel="Slot 0"]
MockPart_1 -> MockPart_2[ headlabel="Slot 1"]
MockPart_2 -> MockPart_3[ taillabel="Slot 1"]
MockPart_2 -> MockPart_4[ taillabel="Slot 0"]
MockPart_3 -> MockPart_5[ headlabel="Slot 0"]
MockPart_4 -> MockPart_5[ taillabel="Slot 0"][ headlabel="Slot 1"]
MockPart_4 -> MockPart_6[ taillabel="Slot 1"][ headlabel="Slot 0"]
MockPart_7 -> MockPart_6[ headlabel="Slot 1"]
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
        estOpt, compOpt, caps, std::set<uint32_t>{ 13, 14, 15 }, DataType::UINT8_QUANTIZED, DataType::UINT8_QUANTIZED);
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
    params.m_InputDataType          = DataType::UINT8_QUANTIZED;
    params.m_OutputDataType         = DataType::UINT8_QUANTIZED;
    params.m_UpscaleFactor          = 3;
    params.m_UpsampleType           = command_stream::cascading::UpsampleType::NEAREST_NEIGHBOUR;
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
                                    DataType::UINT8_QUANTIZED, std::set<uint32_t>{ 13, 14, 15 }, estOpt, compOpt, caps);

    parts.m_Parts.push_back(std::move(inputPart));

    // OutputPart
    auto outputPart = std::make_unique<OutputPart>(5, TensorShape{ 1, 2, 3, 4 }, CompilerDataFormat::NHWCB,
                                                   QuantizationInfo(9, 10.0f), DataType::UINT8_QUANTIZED,
                                                   std::set<uint32_t>{ 13, 14, 15 }, 0, estOpt, compOpt, caps);
    parts.m_Parts.push_back(std::move(outputPart));

    // ReshapePart
    auto reshapePart = std::make_unique<ReshapePart>(
        8, TensorShape{ 1, 2, 3, 4 }, TensorShape{ 5, 6, 7, 8 }, CompilerDataFormat::NHWCB, QuantizationInfo(9, 10.0f),
        DataType::UINT8_QUANTIZED, std::set<uint32_t>{ 13, 14, 15 }, estOpt, compOpt, caps);
    parts.m_Parts.push_back(std::move(reshapePart));

    // Standalone PLE part
    auto standalonePlePart = std::make_unique<StandalonePlePart>(
        9, std::vector<TensorShape>{ TensorShape{ 1, 2, 3, 4 }, TensorShape{ 1, 2, 3, 4 } }, TensorShape{ 1, 2, 3, 4 },
        std::vector<QuantizationInfo>{ QuantizationInfo(9, 10.0f), QuantizationInfo(9, 10.0f) },
        QuantizationInfo(9, 10.0f), ethosn::command_stream::PleOperation::ADDITION, estOpt, compOpt, caps,
        std::set<uint32_t>{ 1 }, DataType::UINT8_QUANTIZED);
    parts.m_Parts.push_back(std::move(standalonePlePart));

    // ConstantPart
    auto constantPart = std::make_unique<ConstantPart>(10, TensorShape{ 1, 2, 3, 4 }, CompilerDataFormat::NHWCB,
                                                       QuantizationInfo(9, 10.0f), DataType::UINT8_QUANTIZED,
                                                       std::set<uint32_t>{ 7 }, estOpt, compOpt, caps);
    parts.m_Parts.push_back(std::move(constantPart));

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
FusedPlePart_1[label = "FusedPlePart 1\nCompilerDataFormat = NONE\nCorrespondingOperationIds = [13, 14, 15]\nInputTensorShape = [1, 2, 3, 4]\nOutputTensorShape = [5, 6, 7, 8]\nInputQuantizationInfo = ZeroPoint = 9, Scale = 10.000000\nOutputQuantizationInfo = ZeroPoint = 11, Scale = 12.000000\nInputDataType = UINT8_QUANTIZED\nOutputDataType = UINT8_QUANTIZED\nKernelOperation = DOWNSAMPLE_2X2\nShapeMultiplier = [1/1, 2/1, 3/1]\nStripeGenerator.MceInputTensorShape = [1, 2, 3, 4]\nStripeGenerator.MceOutputTensorShape = [1, 2, 3, 4]\nStripeGenerator.PleOutputTensorShape = [5, 6, 7, 8]\nStripeGenerator.KernelHeight = 1\nStripeGenerator.KernelWidth = 1\nStripeGenerator.UpscaleFactor = 1\nStripeGenerator.Operation = DEPTHWISE_CONVOLUTION\nStripeGenerator.MceShapeMultiplier = [1/1, 1/1, 1/1]\nStripeGenerator.PleShapeMultiplier = [1/1, 2/1, 3/1]\n"]
McePart_5[label = "McePart 5\nCompilerDataFormat = NONE\nCorrespondingOperationIds = [13, 14, 15]\nInputTensorShape = [1, 2, 3, 4]\nOutputTensorShape = [5, 6, 7, 8]\nInputQuantizationInfo = ZeroPoint = 9, Scale = 10.000000\nOutputQuantizationInfo = ZeroPoint = 11, Scale = 12.000000\nInputDataType = UINT8_QUANTIZED\nOutputDataType = UINT8_QUANTIZED\nWeightsInfo = ([9, 10, 11, 12], UINT8_QUANTIZED, NHWC, ZeroPoint = 11, Scale = 12.000000)\nBiasInfo = ([19, 110, 111, 112], UINT8_QUANTIZED, NHWC, ZeroPoint = 111, Scale = 112.000000)\nStride = 2, 2\nUpscaleFactor = 3\nUpsampleType = NEAREST_NEIGHBOUR\nPadTop = 1\nPadLeft = 3\nOperation = DEPTHWISE_CONVOLUTION\nStripeGenerator.MceInputTensorShape = [1, 2, 3, 4]\nStripeGenerator.MceOutputTensorShape = [5, 6, 7, 8]\nStripeGenerator.PleOutputTensorShape = [5, 6, 7, 8]\nStripeGenerator.KernelHeight = 9\nStripeGenerator.KernelWidth = 10\nStripeGenerator.UpscaleFactor = 3\nStripeGenerator.Operation = DEPTHWISE_CONVOLUTION\nStripeGenerator.MceShapeMultiplier = [3/1, 3/1, 1/1]\nStripeGenerator.PleShapeMultiplier = [1/1, 1/1, 1/1]\n"]
ConcatPart_2[label = "ConcatPart 2\nCompilerDataFormat = NHWCB\nCorrespondingOperationIds = [13, 14, 15]\nInputTensorsInfo = [([1, 2, 3, 4], UINT8_QUANTIZED, NHWC, ZeroPoint = 0, Scale = 1.000000)]\nConcatInfo.Axis = 3\nConcatInfo.OutputQuantInfo = ZeroPoint = 9, Scale = 10.000000\n"]
InputPart_3[label = "InputPart 3\nCompilerDataFormat = NHWCB\nCorrespondingOperationIds = [13, 14, 15]\nOutputTensorShape = [1, 2, 3, 4]\nOutputQuantizationInfo = ZeroPoint = 9, Scale = 10.000000\nOutputDataType = UINT8_QUANTIZED\n"]
OutputPart_5[label = "OutputPart 5\nCompilerDataFormat = NHWCB\nCorrespondingOperationIds = [13, 14, 15]\nInputTensorShape = [1, 2, 3, 4]\nInputQuantizationInfo = ZeroPoint = 9, Scale = 10.000000\nInputDataType = UINT8_QUANTIZED\n"]
ReshapePart_8[label = "ReshapePart 8\nCompilerDataFormat = NHWCB\nCorrespondingOperationIds = [13, 14, 15]\nInputTensorShape = [1, 2, 3, 4]\nOutputTensorShape = [5, 6, 7, 8]\nOutputQuantizationInfo = ZeroPoint = 9, Scale = 10.000000\nDataType = UINT8_QUANTIZED\n"]
StandalonePlePart_9[label = "StandalonePlePart 9\nCompilerDataFormat = NONE\nCorrespondingOperationIds = [1]\nInputTensorShape = [[1, 2, 3, 4], [1, 2, 3, 4]]\nOutputTensorShape = [1, 2, 3, 4]\nInputQuantizationInfo = [ZeroPoint = 9, Scale = 10.000000, ZeroPoint = 9, Scale = 10.000000]\nOutputQuantizationInfo = ZeroPoint = 9, Scale = 10.000000\n"]
ConstantPart_10[label = "ConstantPart 10\nCompilerDataFormat = NHWCB\nCorrespondingOperationIds = [7]\nOutputTensorShape = [1, 2, 3, 4]\nOutputQuantizationInfo = ZeroPoint = 9, Scale = 10.000000\nOutputDataType = UINT8_QUANTIZED\n"]
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
    planBOpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
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
///   * A replacement buffer in the ending glue (F)
///
///  ( A ) -> g -> ( B ) -> ( C ) -> ( D ) ---> g -> ( F ) -> g
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

    PartInputSlot partFInputSlot0   = { partFId, 0 };
    PartOutputSlot partFOutputSlot0 = { partFId, 0 };

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
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 0, QuantizationInfo()));
    planA.m_OpGraph.GetBuffers().back()->m_DebugTag = "InputDram";
    planA.m_OutputMappings                          = { { planA.m_OpGraph.GetBuffers()[0], partAOutputSlot0 } };

    // Part consisting of node B
    Plan planB;
    planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planB.m_OpGraph.GetBuffers().back()->m_DebugTag = "InputSram1";
    planB.m_InputMappings                           = { { planB.m_OpGraph.GetBuffers()[0], partBInputSlot0 } };
    planB.m_OutputMappings                          = { { planB.m_OpGraph.GetBuffers()[0], partBOutputSlot0 } };

    // Part consisting of node C
    Plan planC;
    planC.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planC.m_OpGraph.GetBuffers().back()->m_DebugTag = "InputSram2";
    planC.m_InputMappings                           = { { planC.m_OpGraph.GetBuffers()[0], partCInputSlot0 } };
    planC.m_OutputMappings                          = { { planC.m_OpGraph.GetBuffers()[0], partCOutputSlot0 } };

    // Part consisting of nodes D and E
    Plan planDE;
    planDE.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                        TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                        TraversalOrder::Xyz, 4, QuantizationInfo()));
    planDE.m_OpGraph.GetBuffers().back()->m_DebugTag = "IntermediateSramInput1";
    planDE.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                        TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                        TraversalOrder::Xyz, 0, QuantizationInfo()));
    planDE.m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputSram1";
    planDE.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                        TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                        TraversalOrder::Xyz, 4, QuantizationInfo()));
    planDE.m_OpGraph.GetBuffers().back()->m_DebugTag = "IntermediateSramInput2";
    planDE.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                        TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                        TraversalOrder::Xyz, 0, QuantizationInfo()));
    planDE.m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputSram2";
    planDE.m_InputMappings                           = { { planDE.m_OpGraph.GetBuffers()[0], partDEInputSlot0 },
                               { planDE.m_OpGraph.GetBuffers()[2], partDEInputSlot1 } };
    planDE.m_OutputMappings                          = { { planDE.m_OpGraph.GetBuffers()[1], partDEOutputSlot0 },
                                { planDE.m_OpGraph.GetBuffers()[3], partDEOutputSlot1 } };
    planDE.m_OpGraph.AddOp(std::make_unique<MceOp>(
        MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct, BlockConfig{ 16u, 16u }, TensorShape{ 1, 17, 16, 16 },
        TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 1, 1, 16 }, TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));
    planDE.m_OpGraph.GetOps()[0]->m_DebugTag = "Mce2";
    planDE.m_OpGraph.AddConsumer(planDE.m_OpGraph.GetBuffers()[0], planDE.m_OpGraph.GetOps()[0], 0);
    planDE.m_OpGraph.AddConsumer(planDE.m_OpGraph.GetBuffers()[2], planDE.m_OpGraph.GetOps()[0], 1);
    planDE.m_OpGraph.SetProducer(planDE.m_OpGraph.GetBuffers()[1], planDE.m_OpGraph.GetOps()[0]);
    planDE.m_OpGraph.SetProducer(planDE.m_OpGraph.GetBuffers()[3], planDE.m_OpGraph.GetOps()[0]);

    // Part consisting of node F
    Plan planF;
    planF.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 0, QuantizationInfo()));
    planF.m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputDram1";
    planF.m_InputMappings                           = { { planF.m_OpGraph.GetBuffers()[0], partFInputSlot0 } };

    // Part consisting of node G
    Plan planG;
    planG.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 0, QuantizationInfo()));
    planG.m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputDram2";
    planG.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 0, QuantizationInfo()));
    planG.m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputDram3";
    planG.m_InputMappings                           = { { planG.m_OpGraph.GetBuffers()[0], partGInputSlot0 },
                              { planG.m_OpGraph.GetBuffers()[1], partGInputSlot1 } };

    // The end glueing of A is empty. But the starting glue of B has the connections.
    auto endingGlueA = std::make_shared<EndingGlue>();

    auto startingGlueB = std::make_shared<StartingGlue>();
    startingGlueB->m_Graph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
    startingGlueB->m_Graph.GetOps()[0]->m_DebugTag = "InputDma";
    startingGlueB->m_ExternalConnections.m_BuffersToOps.insert(
        { planA.m_OpGraph.GetBuffers().back(), startingGlueB->m_Graph.GetOps()[0] });
    startingGlueB->m_ExternalConnections.m_OpsToBuffers.insert(
        { startingGlueB->m_Graph.GetOps()[0], planB.m_OpGraph.GetBuffers()[0] });

    auto endingGlueB = std::make_shared<EndingGlue>();

    auto startingGlueC = std::make_shared<StartingGlue>();
    startingGlueC->m_ExternalConnections.m_ReplacementBuffers.insert(
        { planC.m_OpGraph.GetBuffers()[0], planB.m_OpGraph.GetBuffers()[0] });

    auto endingGlueC = std::make_shared<EndingGlue>();

    auto startingGlueDE = std::make_shared<StartingGlue>();
    startingGlueDE->m_ExternalConnections.m_ReplacementBuffers.insert(
        { planDE.m_OpGraph.GetBuffers()[0], planC.m_OpGraph.GetBuffers()[0] });
    startingGlueDE->m_ExternalConnections.m_ReplacementBuffers.insert(
        { planDE.m_OpGraph.GetBuffers()[2], planC.m_OpGraph.GetBuffers()[0] });

    auto endingGlueD = std::make_shared<EndingGlue>();
    endingGlueD->m_Graph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
    endingGlueD->m_Graph.GetOps()[0]->m_DebugTag = "OutputDma1";
    endingGlueD->m_ExternalConnections.m_BuffersToOps.insert(
        { planDE.m_OpGraph.GetBuffers()[1], endingGlueD->m_Graph.GetOps()[0] });
    endingGlueD->m_Graph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
    endingGlueD->m_Graph.GetOps()[0]->m_DebugTag = "OutputDma2";
    endingGlueD->m_ExternalConnections.m_BuffersToOps.insert(
        { planDE.m_OpGraph.GetBuffers()[3], endingGlueD->m_Graph.GetOps()[1] });
    auto startingGlueF = std::make_shared<StartingGlue>();
    startingGlueF->m_ExternalConnections.m_OpsToBuffers.insert(
        { endingGlueD->m_Graph.GetOps()[0], planF.m_OpGraph.GetBuffers().back() });

    auto startingGluefromDtoG = std::make_shared<StartingGlue>();
    startingGluefromDtoG->m_ExternalConnections.m_OpsToBuffers.insert(
        { endingGlueD->m_Graph.GetOps()[1], planG.m_OpGraph.GetBuffers()[0] });

    auto endingGlueE = std::make_shared<EndingGlue>();
    endingGlueE->m_Graph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
    endingGlueE->m_Graph.GetOps()[0]->m_DebugTag = "OutputDma3";
    endingGlueE->m_ExternalConnections.m_BuffersToOps.insert(
        { planDE.m_OpGraph.GetBuffers()[1], endingGlueE->m_Graph.GetOps()[0] });
    auto startingGluefromEtoG = std::make_shared<StartingGlue>();
    startingGluefromEtoG->m_ExternalConnections.m_OpsToBuffers.insert(
        { endingGlueE->m_Graph.GetOps()[0], planG.m_OpGraph.GetBuffers()[1] });

    auto endingGlueF = std::make_shared<EndingGlue>();
    endingGlueF->m_Graph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                            TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                            TraversalOrder::Xyz, 0, QuantizationInfo()));
    endingGlueF->m_Graph.GetBuffers()[0]->m_DebugTag = "ReplacementBuffer";
    endingGlueF->m_ExternalConnections.m_ReplacementBuffers.insert(
        { planF.m_OpGraph.GetBuffers()[0], endingGlueF->m_Graph.GetBuffers()[0] });

    Combination comb;

    Elem elemA;
    elemA.m_Plan        = std::make_shared<Plan>(std::move(planA));
    elemA.m_EndingGlues = { { partAOutputSlot0, endingGlueA } };

    Elem elemB;
    elemB.m_Plan          = std::make_shared<Plan>(std::move(planB));
    elemB.m_StartingGlues = { { partBInputSlot0, startingGlueB } };
    elemB.m_EndingGlues   = { { partBOutputSlot0, endingGlueB } };

    Elem elemC;
    elemC.m_Plan          = std::make_shared<Plan>(std::move(planC));
    elemC.m_StartingGlues = { { partCInputSlot0, startingGlueC } };
    elemC.m_EndingGlues   = { { partCOutputSlot0, endingGlueC } };

    Elem elemDE;
    elemDE.m_Plan          = std::make_shared<Plan>(std::move(planDE));
    elemDE.m_StartingGlues = { { partDEInputSlot0, startingGlueDE } };
    elemDE.m_EndingGlues   = { { partDEOutputSlot0, endingGlueD }, { partDEOutputSlot1, endingGlueE } };

    Elem elemF;
    elemF.m_Plan          = std::make_shared<Plan>(std::move(planF));
    elemF.m_StartingGlues = { { partFInputSlot0, startingGlueF } };
    elemF.m_EndingGlues   = { { partFOutputSlot0, endingGlueF } };

    Elem elemG;
    elemG.m_Plan          = std::make_shared<Plan>(std::move(planG));
    elemG.m_StartingGlues = { { partGInputSlot0, startingGluefromDtoG }, { partGInputSlot1, startingGluefromEtoG } };

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
        SaveCombinationToDot(comb, stream, DetailLevel::Low);
    }

    // Save to a string and check against expected result
    std::stringstream stream;
    SaveCombinationToDot(comb, stream, DetailLevel::Low);

    std::string expected =
        R"(digraph SupportLibraryGraph
{
subgraph clusterPlan_6
{
label="Part 0: Plan 6"
labeljust=l
InputDram[label = "InputDram", shape = box, color = brown]
}
subgraph clusterPart_0_Plan_6_Ending_Glue
{
label="Part 0 Plan 6 Ending Glue"
labeljust=l
}
subgraph clusterPlan_8
{
label="Part 1: Plan 8"
labeljust=l
InputSram1[label = "InputSram1", shape = box, color = blue]
}
subgraph clusterPart_1_Plan_8_Starting_Glue
{
label="Part 1 Plan 8 Starting Glue"
labeljust=l
InputDma[label = "InputDma", shape = oval, color = darkgoldenrod]
}
InputDram -> InputDma
InputDma -> InputSram1
subgraph clusterPart_1_Plan_8_Ending_Glue
{
label="Part 1 Plan 8 Ending Glue"
labeljust=l
}
subgraph clusterPlan_10
{
label="Part 2: Plan 10"
labeljust=l
InputSram2[label = "InputSram2", shape = box, color = blue]
}
subgraph clusterPart_2_Plan_10_Starting_Glue
{
label="Part 2 Plan 10 Starting Glue"
labeljust=l
}
InputSram1 -> InputSram2[style = dashed, label="Replaced by", dir="back"]
subgraph clusterPart_2_Plan_10_Ending_Glue
{
label="Part 2 Plan 10 Ending Glue"
labeljust=l
}
subgraph clusterPlan_12
{
label="Part 3: Plan 12"
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
}
subgraph clusterPart_3_Plan_12_Starting_Glue
{
label="Part 3 Plan 12 Starting Glue"
labeljust=l
}
InputSram2 -> IntermediateSramInput1[style = dashed, label="Replaced by", dir="back"]
InputSram2 -> IntermediateSramInput2[style = dashed, label="Replaced by", dir="back"]
subgraph clusterPart_3_Plan_12_Ending_Glue
{
label="Part 3 Plan 12 Ending Glue"
labeljust=l
OutputDma2[label = "OutputDma2", shape = oval, color = darkgoldenrod]
DmaOp_25[label = "DmaOp 25", shape = oval, color = darkgoldenrod]
}
OutputSram1 -> OutputDma2
OutputSram2 -> DmaOp_25
subgraph clusterPart_3_Plan_12_Ending_Glue
{
label="Part 3 Plan 12 Ending Glue"
labeljust=l
OutputDma3[label = "OutputDma3", shape = oval, color = darkgoldenrod]
}
OutputSram1 -> OutputDma3
subgraph clusterPlan_18
{
label="Part 4: Plan 18"
labeljust=l
OutputDram1[label = "OutputDram1", shape = box, color = brown]
}
subgraph clusterPart_4_Plan_18_Starting_Glue
{
label="Part 4 Plan 18 Starting Glue"
labeljust=l
}
OutputDma2 -> OutputDram1
subgraph clusterPart_4_Plan_18_Ending_Glue
{
label="Part 4 Plan 18 Ending Glue"
labeljust=l
ReplacementBuffer[label = "ReplacementBuffer", shape = box, color = brown]
}
OutputDram1 -> ReplacementBuffer[style = dashed, label="Replaced by"]
subgraph clusterPlan_20
{
label="Part 5: Plan 20"
labeljust=l
OutputDram2[label = "OutputDram2", shape = box, color = brown]
OutputDram3[label = "OutputDram3", shape = box, color = brown]
}
subgraph clusterPart_5_Plan_20_Starting_Glue
{
label="Part 5 Plan 20 Starting Glue"
labeljust=l
}
DmaOp_25 -> OutputDram2
subgraph clusterPart_5_Plan_20_Starting_Glue
{
label="Part 5 Plan 20 Starting Glue"
labeljust=l
}
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
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Plan planA;
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planA.m_OutputMappings = { { planA.m_OpGraph.GetBuffers()[0], partAOutputSlot } };

    Plan planB;
    planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planB.m_InputMappings = { { planB.m_OpGraph.GetBuffers()[0], partBInputSlot } };

    Plan planC;
    planC.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planC.m_InputMappings = { { planC.m_OpGraph.GetBuffers()[0], partCInputSlot } };

    Plan planD;
    planD.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planD.m_InputMappings = { { planD.m_OpGraph.GetBuffers()[0], partDInputSlot } };

    Combination combA(partA, std::move(planA), 0);
    Combination combB(partB, std::move(planB), 1);
    Combination combC(partC, std::move(planC), 2);
    Combination combD(partD, std::move(planD), 3);

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

    Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);

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
        std::ofstream stream("SaveCombinationBranchToDot.dot");
        SaveCombinationToDot(combGlued, stream, DetailLevel::Low);
    }

    // Save to a string and check against expected result
    std::stringstream stream;
    SaveCombinationToDot(combGlued, stream, DetailLevel::Low);

    std::string expected =
        R"(digraph SupportLibraryGraph
{
subgraph clusterPlan_4
{
label="Part 0: Plan 4"
labeljust=l
Buffer_5[label = "Buffer 5", shape = box, color = blue]
}
subgraph clusterPart_0_Plan_4_Ending_Glue
{
label="Part 0 Plan 4 Ending Glue"
labeljust=l
DmaOp_12[label = "DmaOp 12", shape = oval, color = darkgoldenrod]
DmaOp_14[label = "DmaOp 14", shape = oval, color = darkgoldenrod]
Buffer_13[label = "Buffer 13", shape = box, color = brown]
DmaOp_14 -> Buffer_13
}
Buffer_5 -> DmaOp_12
Buffer_5 -> DmaOp_14
subgraph clusterPlan_6
{
label="Part 1: Plan 6"
labeljust=l
Buffer_7[label = "Buffer 7", shape = box, color = blue]
}
subgraph clusterPart_1_Plan_6_Starting_Glue
{
label="Part 1 Plan 6 Starting Glue"
labeljust=l
DmaOp_15[label = "DmaOp 15", shape = oval, color = darkgoldenrod]
}
Buffer_13 -> DmaOp_15
DmaOp_15 -> Buffer_7
subgraph clusterPlan_10
{
label="Part 3: Plan 10"
labeljust=l
Buffer_11[label = "Buffer 11", shape = box, color = brown]
}
subgraph clusterPart_3_Plan_10_Starting_Glue
{
label="Part 3 Plan 10 Starting Glue"
labeljust=l
}
DmaOp_12 -> Buffer_11
subgraph clusterPlan_8
{
label="Part 2: Plan 8"
labeljust=l
Buffer_9[label = "Buffer 9", shape = box, color = blue]
}
subgraph clusterPart_2_Plan_8_Starting_Glue
{
label="Part 2 Plan 8 Starting Glue"
labeljust=l
DmaOp_16[label = "DmaOp 16", shape = oval, color = darkgoldenrod]
}
Buffer_13 -> DmaOp_16
DmaOp_16 -> Buffer_9
}
)";
    std::string output = stream.str();
    REQUIRE(output == expected);
}
