//
// Copyright © 2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/cascading/CascadingCompiler.hpp"
#include "../src/cascading/CombinerDFS.hpp"
#include "../src/cascading/PartUtils.hpp"
#include "../src/cascading/StripeHelper.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>
#include <fstream>

using namespace ethosn::support_library;
using namespace ethosn::support_library::impl;
using namespace ethosn::command_stream;
using namespace ethosn::command_stream::cascading;
using namespace cascading_compiler;
using PleKernelId = ethosn::command_stream::cascading::PleKernelId;

//////////////////////////////////////////////////////////////////////////////////////////////
// Agent Data Tests
//////////////////////////////////////////////////////////////////////////////////////////////

// IfmStreamer Agent Data Test
TEST_CASE("IfmStreamer Agent Data Test", "[CascadingCompiler]")
{}

// WeightStreamer Agent Data Test
TEST_CASE("WeightStreamer Agent Data Test", "[CascadingCompiler]")
{}

// MceScheduler Agent Data Test
TEST_CASE("MceScheduler Agent Data Test", "[CascadingCompiler]")
{
    GraphOfParts graph;
    auto& parts       = graph.m_Parts;
    auto& connections = graph.m_Connections;

    auto inputDramPart  = std::make_unique<MockPart>(graph.GeneratePartId());
    auto inputSramPart  = std::make_unique<MockPart>(graph.GeneratePartId());
    auto weightDramPart = std::make_unique<MockPart>(graph.GeneratePartId());
    auto weightSramPart = std::make_unique<MockPart>(graph.GeneratePartId());
    auto mcePlePart     = std::make_unique<MockPart>(graph.GeneratePartId());
    auto outputDramPart = std::make_unique<MockPart>(graph.GeneratePartId());

    PartId inputDramPartId  = inputDramPart->GetPartId();
    PartId inputSramPartId  = inputSramPart->GetPartId();
    PartId weightDramPartId = weightDramPart->GetPartId();
    PartId weightSramPartId = weightSramPart->GetPartId();
    PartId mcePlePartId     = mcePlePart->GetPartId();
    PartId outputDramPartId = outputDramPart->GetPartId();

    parts.push_back(std::move(inputDramPart));
    parts.push_back(std::move(inputSramPart));
    parts.push_back(std::move(weightDramPart));
    parts.push_back(std::move(weightSramPart));
    parts.push_back(std::move(mcePlePart));
    parts.push_back(std::move(outputDramPart));

    PartOutputSlot inputDramPartOutputSlot0  = { inputDramPartId, 0 };
    PartOutputSlot weightDramPartOutputSlot0 = { weightDramPartId, 0 };

    PartInputSlot inputSramPartInputSlot0   = { inputSramPartId, 0 };
    PartOutputSlot inputSramPartOutputSlot0 = { inputSramPartId, 0 };

    PartInputSlot weightSramPartInputSlot0   = { weightSramPartId, 0 };
    PartOutputSlot weightSramPartOutputSlot0 = { weightSramPartId, 0 };

    PartInputSlot mcePlePartInputSlot0   = { mcePlePartId, 0 };
    PartInputSlot mcePlePartInputSlot1   = { mcePlePartId, 1 };
    PartOutputSlot mcePlePartOutputSlot0 = { mcePlePartId, 0 };

    PartInputSlot outputDramPartInputSlot0 = { outputDramPartId, 0 };

    connections[inputSramPartInputSlot0]  = inputDramPartOutputSlot0;
    connections[weightSramPartInputSlot0] = weightDramPartOutputSlot0;
    connections[mcePlePartInputSlot0]     = inputSramPartOutputSlot0;
    connections[mcePlePartInputSlot1]     = weightSramPartOutputSlot0;
    connections[outputDramPartInputSlot0] = mcePlePartOutputSlot0;

    const std::set<uint32_t> operationIds = { 0 };
    ethosn::support_library::impl::NumMemoryStripes numMemoryStripes;

    // Plan inputDramPlan
    Plan inputDramPlan;
    inputDramPlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                               TensorShape{ 1, 160, 160, 3 }, TensorShape{ 0, 0, 0, 0 },
                                                               TraversalOrder::Xyz, 0, QuantizationInfo()));
    inputDramPlan.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Input;
    inputDramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "InputDramBuffer";
    inputDramPlan.m_OutputMappings = { { inputDramPlan.m_OpGraph.GetBuffers()[0], inputDramPartOutputSlot0 } };

    // Glue glueInputDram_InputSram
    Glue glueInputDram_InputSram;
    glueInputDram_InputSram.m_Graph.AddOp(std::make_unique<DmaOp>());
    glueInputDram_InputSram.m_Graph.GetOps()[0]->m_DebugTag = "InputDmaOp";
    glueInputDram_InputSram.m_InputSlot                     = { glueInputDram_InputSram.m_Graph.GetOps()[0], 0 };
    glueInputDram_InputSram.m_Output.push_back(glueInputDram_InputSram.m_Graph.GetOps()[0]);

    // Plan inputSramPlan
    Plan inputSramPlan;
    inputSramPlan.m_OpGraph.AddBuffer(
        std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 160, 160, 3 },
                                 TensorShape{ 1, 8, 8, 16 }, TraversalOrder::Xyz, 4, QuantizationInfo()));
    inputSramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "InputSramBuffer";
    inputSramPlan.m_OpGraph.GetBuffers().back()->m_Offset     = 0x0000F0F0;
    inputSramPlan.m_OpGraph.GetBuffers().back()->m_NumStripes = 4;
    inputSramPlan.m_InputMappings  = { { inputSramPlan.m_OpGraph.GetBuffers()[0], inputSramPartInputSlot0 } };
    inputSramPlan.m_OutputMappings = { { inputSramPlan.m_OpGraph.GetBuffers()[0], inputSramPartOutputSlot0 } };

    Buffer* ptrInputBuffer   = inputSramPlan.m_OpGraph.GetBuffers().back();
    uint32_t inputStripeSize = CalculateBufferSize(ptrInputBuffer->m_StripeShape, ptrInputBuffer->m_Format);
    int32_t inputZeroPoint   = ptrInputBuffer->m_QuantizationInfo.GetZeroPoint();

    // Plan weightDramPlan
    Plan weightDramPlan;
    weightDramPlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                                TensorShape{ 1, 1, 3, 1 }, TensorShape{ 0, 0, 0, 0 },
                                                                TraversalOrder::Xyz, 0, QuantizationInfo()));
    weightDramPlan.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Input;
    weightDramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "WeightDramBuffer";
    weightDramPlan.m_OutputMappings = { { weightDramPlan.m_OpGraph.GetBuffers()[0], weightDramPartOutputSlot0 } };

    // Glue glueWeightDram_WeightSram
    Glue glueWeightDram_WeightSram;
    glueWeightDram_WeightSram.m_Graph.AddOp(std::make_unique<DmaOp>());
    glueWeightDram_WeightSram.m_Graph.GetOps()[0]->m_DebugTag = "WeightDmaOp";
    glueWeightDram_WeightSram.m_InputSlot                     = { glueWeightDram_WeightSram.m_Graph.GetOps()[0], 0 };
    glueWeightDram_WeightSram.m_Output.push_back(glueWeightDram_WeightSram.m_Graph.GetOps()[0]);

    // Plan weightSramPlan
    Plan weightSramPlan;
    weightSramPlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                                TensorShape{ 1, 1, 3, 1 }, TensorShape{ 1, 1, 16, 1 },
                                                                TraversalOrder::Xyz, 4, QuantizationInfo()));
    weightSramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "WeightSramBuffer";
    weightSramPlan.m_OpGraph.GetBuffers().back()->m_Offset     = 0x00000F0F;
    weightSramPlan.m_OpGraph.GetBuffers().back()->m_NumStripes = 3;
    weightSramPlan.m_InputMappings  = { { weightSramPlan.m_OpGraph.GetBuffers()[0], weightSramPartInputSlot0 } };
    weightSramPlan.m_OutputMappings = { { weightSramPlan.m_OpGraph.GetBuffers()[0], weightSramPartOutputSlot0 } };

    Buffer* ptrWeightBuffer   = weightSramPlan.m_OpGraph.GetBuffers().back();
    uint32_t weightStripeSize = CalculateBufferSize(ptrWeightBuffer->m_StripeShape, ptrWeightBuffer->m_Format);
    uint8_t kernelHeight      = static_cast<uint8_t>(ptrWeightBuffer->m_TensorShape[1]);
    uint8_t kernelWidth       = static_cast<uint8_t>(ptrWeightBuffer->m_TensorShape[2]);

    // Plan mcePlePlan
    Plan mcePlePlan;
    mcePlePlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                            TensorShape{ 1, 160, 160, 3 }, TensorShape{ 1, 8, 8, 16 },
                                                            TraversalOrder::Xyz, 4, QuantizationInfo()));
    mcePlePlan.m_OpGraph.GetBuffers().back()->m_DebugTag = "IntermediateInputSramBuffer";
    mcePlePlan.m_OpGraph.GetBuffers().back()->m_Offset   = 0;
    mcePlePlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                            TensorShape{ 1, 1, 3, 1 }, TensorShape{ 1, 1, 16, 1 },
                                                            TraversalOrder::Xyz, 4, QuantizationInfo()));
    mcePlePlan.m_OpGraph.GetBuffers().back()->m_DebugTag = "IntermediateWeightSramBuffer";
    mcePlePlan.m_OpGraph.GetBuffers().back()->m_Offset   = 0;
    mcePlePlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::PleInputSram, CascadingBufferFormat::NHWCB,
                                                            TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                            TraversalOrder::Xyz, 0, QuantizationInfo()));
    mcePlePlan.m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputPleInputSramBuffer";
    mcePlePlan.m_OpGraph.GetBuffers().back()->m_Offset   = 0;

    mcePlePlan.m_OpGraph.AddOp(std::make_unique<MceOp>(
        Lifetime::Cascade, ethosn::command_stream::MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
        BlockConfig{ 16u, 16u }, TensorShape{ 1, 8, 8, 16 }, TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 1, 16, 1 },
        TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));
    mcePlePlan.m_OpGraph.GetOps()[0]->m_DebugTag = "MceOp";

    mcePlePlan.m_OpGraph.AddConsumer(mcePlePlan.m_OpGraph.GetBuffers()[0], mcePlePlan.m_OpGraph.GetOps()[0], 0);
    mcePlePlan.m_OpGraph.AddConsumer(mcePlePlan.m_OpGraph.GetBuffers()[1], mcePlePlan.m_OpGraph.GetOps()[0], 1);
    mcePlePlan.m_OpGraph.SetProducer(mcePlePlan.m_OpGraph.GetBuffers()[2], mcePlePlan.m_OpGraph.GetOps()[0]);

    int8_t ifmDeltaHeight = static_cast<int8_t>(inputSramPlan.m_OpGraph.GetBuffers()[0]->m_TensorShape[1] -
                                                mcePlePlan.m_OpGraph.GetBuffers()[2]->m_TensorShape[1]);
    int8_t ifmDeltaWidth  = static_cast<int8_t>(inputSramPlan.m_OpGraph.GetBuffers()[0]->m_TensorShape[2] -
                                               mcePlePlan.m_OpGraph.GetBuffers()[2]->m_TensorShape[2]);

    // Adding a passthrough PLE kernel to the plan
    // The PleKernelId is expected to be PASSTHROUGH_8x8_1
    auto pleOp =
        std::make_unique<PleOp>(Lifetime::Cascade, ethosn::command_stream::PleOperation::PASSTHROUGH,
                                BlockConfig{ 8u, 8u }, 1, std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 8 } },
                                TensorShape{ 1, 4, 4, 32 }, ethosn::command_stream::DataType::U8, true);
    pleOp.get()->m_Offset     = 0x0000FFFF;
    numMemoryStripes.m_Output = 1;
    auto outBufferAndPleOp    = AddPleToOpGraph(mcePlePlan.m_OpGraph, Lifetime::Cascade, TraversalOrder::Xyz,
                                             TensorShape{ 1, 8, 8, 32 }, numMemoryStripes, std::move(pleOp),
                                             TensorShape{ 1, 80, 80, 24 }, QuantizationInfo(), operationIds);
    mcePlePlan.m_OpGraph.GetBuffers().back()->m_Offset = 0;
    mcePlePlan.m_OpGraph.AddConsumer(mcePlePlan.m_OpGraph.GetBuffers()[2], mcePlePlan.m_OpGraph.GetOps()[1], 0);

    mcePlePlan.m_InputMappings  = { { mcePlePlan.m_OpGraph.GetBuffers()[0], mcePlePartInputSlot0 },
                                   { mcePlePlan.m_OpGraph.GetBuffers()[1], mcePlePartInputSlot1 } };
    mcePlePlan.m_OutputMappings = { { mcePlePlan.m_OpGraph.GetBuffers()[3], mcePlePartOutputSlot0 } };

    // Glue glueOutputSram_OutputDram
    Glue glueOutputSram_OutputDram;
    glueOutputSram_OutputDram.m_Graph.AddOp(std::make_unique<DmaOp>());
    glueOutputSram_OutputDram.m_Graph.GetOps()[0]->m_DebugTag = "OutputDmaOp";
    glueOutputSram_OutputDram.m_InputSlot                     = { glueOutputSram_OutputDram.m_Graph.GetOps()[0], 0 };
    glueOutputSram_OutputDram.m_Output.push_back(glueOutputSram_OutputDram.m_Graph.GetOps()[0]);

    // Plan outputDramPlan
    Plan outputDramPlan;
    outputDramPlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                                TensorShape{ 1, 80, 80, 24 }, TensorShape{ 0, 0, 0, 0 },
                                                                TraversalOrder::Xyz, 0, QuantizationInfo()));
    outputDramPlan.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Output;
    outputDramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "OutputDramBuffer";
    outputDramPlan.m_InputMappings = { { outputDramPlan.m_OpGraph.GetBuffers()[0], outputDramPartInputSlot0 } };

    // Create Combination with all the plans and glues
    Combination comb;

    Elem elemInputDram  = { std::make_shared<Plan>(std::move(inputDramPlan)),
                           { { inputSramPartInputSlot0, { &glueInputDram_InputSram, true } } } };
    Elem elemInputSram  = { std::make_shared<Plan>(std::move(inputSramPlan)), {} };
    Elem elemWeightDram = { std::make_shared<Plan>(std::move(weightDramPlan)),
                            { { weightSramPartInputSlot0, { &glueWeightDram_WeightSram, true } } } };
    Elem elemWeightSram = { std::make_shared<Plan>(std::move(weightSramPlan)), {} };
    Elem elemMcePle     = { std::make_shared<Plan>(std::move(mcePlePlan)),
                        { { outputDramPartInputSlot0, { &glueOutputSram_OutputDram, true } } } };
    Elem elemOutputDram = { std::make_shared<Plan>(std::move(outputDramPlan)), {} };

    comb.m_Elems.insert(std::make_pair(0, elemInputDram));
    comb.m_PartIdsInOrder.push_back(0);
    comb.m_Elems.insert(std::make_pair(1, elemInputSram));
    comb.m_PartIdsInOrder.push_back(1);
    comb.m_Elems.insert(std::make_pair(2, elemWeightDram));
    comb.m_PartIdsInOrder.push_back(2);
    comb.m_Elems.insert(std::make_pair(3, elemWeightSram));
    comb.m_PartIdsInOrder.push_back(3);
    comb.m_Elems.insert(std::make_pair(4, elemMcePle));
    comb.m_PartIdsInOrder.push_back(4);
    comb.m_Elems.insert(std::make_pair(5, elemOutputDram));
    comb.m_PartIdsInOrder.push_back(5);

    bool dumpInputGraphToFile = false;
    if (dumpInputGraphToFile)
    {
        std::ofstream stream("CascadingCompiler MceSchedulerAgent Input.dot");
        SaveCombinationToDot(comb, graph, stream, DetailLevel::High);
    }

    // Call function under test
    OpGraph mergedOpGraph = GetOpGraphForCombination(comb, graph);

    bool dumpOutputGraphToFile = false;
    if (dumpOutputGraphToFile)
    {
        std::ofstream stream("CascadingCompiler MceSchedulerAgent Output.dot");
        SaveOpGraphToDot(mergedOpGraph, stream, DetailLevel::High);
    }

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    CascadingCompiler cascadingCompiler(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = cascadingCompiler.Compile();

    std::vector<Agent> commandStream = cascadingCompiler.GetCommandStreamOfAgents();

    const Agent& mceSAgent = commandStream[3];
    const MceS& mceSData   = mceSAgent.data.mce;

    REQUIRE(mceSData.ifmTile.baseAddr == 0x0000F0F0);
    REQUIRE(mceSData.ifmTile.numSlots == 4);
    REQUIRE(mceSData.ifmTile.slotSize == inputStripeSize);

    REQUIRE(mceSData.wgtTile.baseAddr == 0x00000F0F);
    REQUIRE(mceSData.wgtTile.numSlots == 3);
    REQUIRE(mceSData.wgtTile.slotSize == weightStripeSize);

    REQUIRE(mceSData.blockSize.width == 16);
    REQUIRE(mceSData.blockSize.height == 16);

    REQUIRE(mceSData.dfltStripeSize.ofmHeight == 8);
    REQUIRE(mceSData.dfltStripeSize.ofmWidth == 8);
    REQUIRE(mceSData.dfltStripeSize.ofmChannels == 8);
    REQUIRE(mceSData.dfltStripeSize.ifmChannels == 16);

    REQUIRE(mceSData.edgeStripeSize.ofmHeight == 1);
    REQUIRE(mceSData.edgeStripeSize.ofmWidth == 8);
    REQUIRE(mceSData.edgeStripeSize.ofmChannels == 8);
    REQUIRE(mceSData.edgeStripeSize.ifmChannels == 3);

    REQUIRE(mceSData.numStripes.ofmHeight == 3);
    REQUIRE(mceSData.numStripes.ofmWidth == 2);
    REQUIRE(mceSData.numStripes.ofmChannels == 2);
    REQUIRE(mceSData.numStripes.ifmChannels == 1);

    REQUIRE(mceSData.stripeIdStrides.ofmHeight == 2);
    REQUIRE(mceSData.stripeIdStrides.ofmWidth == 1);
    REQUIRE(mceSData.stripeIdStrides.ofmChannels == 6);
    REQUIRE(mceSData.stripeIdStrides.ifmChannels == 1);

    REQUIRE(mceSData.convStrideXy.x == 1);
    REQUIRE(mceSData.convStrideXy.y == 1);

    REQUIRE(mceSData.ifmZeroPoint == inputZeroPoint);
    REQUIRE(mceSData.mceOpMode == cascading::MceOperation::CONVOLUTION);
    REQUIRE(mceSData.algorithm == cascading::MceAlgorithm::DIRECT);

    REQUIRE(mceSData.filterShape.height == kernelHeight);
    REQUIRE(mceSData.filterShape.width == kernelWidth);

    REQUIRE(mceSData.padding.left == 0);
    REQUIRE(mceSData.padding.top == 0);

    REQUIRE(mceSData.ifmDelta.height == ifmDeltaHeight);
    REQUIRE(mceSData.ifmDelta.width == ifmDeltaWidth);

    REQUIRE(mceSData.reluActiv.max == 255);
    REQUIRE(mceSData.reluActiv.min == 0);

    REQUIRE(mceSData.pleKernelId == cascading::PleKernelId::PASSTHROUGH_8X8_1);

    ETHOSN_UNUSED(outBufferAndPleOp);
}

// PleLoader Agent Data Test
TEST_CASE("PleLoader Agent Data Test", "[CascadingCompiler]")
{
    GraphOfParts graph;
    auto& parts       = graph.m_Parts;
    auto& connections = graph.m_Connections;

    auto inputDramPart  = std::make_unique<MockPart>(graph.GeneratePartId());
    auto inputSramPart  = std::make_unique<MockPart>(graph.GeneratePartId());
    auto weightDramPart = std::make_unique<MockPart>(graph.GeneratePartId());
    auto weightSramPart = std::make_unique<MockPart>(graph.GeneratePartId());
    auto mcePlePart     = std::make_unique<MockPart>(graph.GeneratePartId());
    auto outputDramPart = std::make_unique<MockPart>(graph.GeneratePartId());

    PartId inputDramPartId  = inputDramPart->GetPartId();
    PartId inputSramPartId  = inputSramPart->GetPartId();
    PartId weightDramPartId = weightDramPart->GetPartId();
    PartId weightSramPartId = weightSramPart->GetPartId();
    PartId mcePlePartId     = mcePlePart->GetPartId();
    PartId outputDramPartId = outputDramPart->GetPartId();

    parts.push_back(std::move(inputDramPart));
    parts.push_back(std::move(inputSramPart));
    parts.push_back(std::move(weightDramPart));
    parts.push_back(std::move(weightSramPart));
    parts.push_back(std::move(mcePlePart));
    parts.push_back(std::move(outputDramPart));

    PartOutputSlot inputDramPartOutputSlot0  = { inputDramPartId, 0 };
    PartOutputSlot weightDramPartOutputSlot0 = { weightDramPartId, 0 };

    PartInputSlot inputSramPartInputSlot0   = { inputSramPartId, 0 };
    PartOutputSlot inputSramPartOutputSlot0 = { inputSramPartId, 0 };

    PartInputSlot weightSramPartInputSlot0   = { weightSramPartId, 0 };
    PartOutputSlot weightSramPartOutputSlot0 = { weightSramPartId, 0 };

    PartInputSlot mcePlePartInputSlot0   = { mcePlePartId, 0 };
    PartInputSlot mcePlePartInputSlot1   = { mcePlePartId, 1 };
    PartOutputSlot mcePlePartOutputSlot0 = { mcePlePartId, 0 };

    PartInputSlot outputDramPartInputSlot0 = { outputDramPartId, 0 };

    connections[inputSramPartInputSlot0]  = inputDramPartOutputSlot0;
    connections[weightSramPartInputSlot0] = weightDramPartOutputSlot0;
    connections[mcePlePartInputSlot0]     = inputSramPartOutputSlot0;
    connections[mcePlePartInputSlot1]     = weightSramPartOutputSlot0;
    connections[outputDramPartInputSlot0] = mcePlePartOutputSlot0;

    const std::set<uint32_t> operationIds = { 0 };
    ethosn::support_library::impl::NumMemoryStripes numMemoryStripes;

    // Plan inputDramPlan
    Plan inputDramPlan;
    inputDramPlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                               TensorShape{ 1, 160, 160, 3 }, TensorShape{ 0, 0, 0, 0 },
                                                               TraversalOrder::Xyz, 0, QuantizationInfo()));
    inputDramPlan.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Input;
    inputDramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "InputDramBuffer";
    inputDramPlan.m_OutputMappings = { { inputDramPlan.m_OpGraph.GetBuffers()[0], inputDramPartOutputSlot0 } };

    // Glue glueInputDram_InputSram
    Glue glueInputDram_InputSram;
    glueInputDram_InputSram.m_Graph.AddOp(std::make_unique<DmaOp>());
    glueInputDram_InputSram.m_Graph.GetOps()[0]->m_DebugTag = "InputDmaOp";
    glueInputDram_InputSram.m_InputSlot                     = { glueInputDram_InputSram.m_Graph.GetOps()[0], 0 };
    glueInputDram_InputSram.m_Output.push_back(glueInputDram_InputSram.m_Graph.GetOps()[0]);

    // Plan inputSramPlan
    Plan inputSramPlan;
    inputSramPlan.m_OpGraph.AddBuffer(
        std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 160, 160, 3 },
                                 TensorShape{ 1, 8, 8, 16 }, TraversalOrder::Xyz, 4, QuantizationInfo()));
    inputSramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "InputSramBuffer";
    inputSramPlan.m_OpGraph.GetBuffers().back()->m_Offset     = 0x0000F0F0;
    inputSramPlan.m_OpGraph.GetBuffers().back()->m_NumStripes = 4;
    inputSramPlan.m_InputMappings  = { { inputSramPlan.m_OpGraph.GetBuffers()[0], inputSramPartInputSlot0 } };
    inputSramPlan.m_OutputMappings = { { inputSramPlan.m_OpGraph.GetBuffers()[0], inputSramPartOutputSlot0 } };

    // Plan weightDramPlan
    Plan weightDramPlan;
    weightDramPlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                                TensorShape{ 1, 1, 3, 1 }, TensorShape{ 0, 0, 0, 0 },
                                                                TraversalOrder::Xyz, 0, QuantizationInfo()));
    weightDramPlan.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Input;
    weightDramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "WeightDramBuffer";
    weightDramPlan.m_OutputMappings = { { weightDramPlan.m_OpGraph.GetBuffers()[0], weightDramPartOutputSlot0 } };

    // Glue glueWeightDram_WeightSram
    Glue glueWeightDram_WeightSram;
    glueWeightDram_WeightSram.m_Graph.AddOp(std::make_unique<DmaOp>());
    glueWeightDram_WeightSram.m_Graph.GetOps()[0]->m_DebugTag = "WeightDmaOp";
    glueWeightDram_WeightSram.m_InputSlot                     = { glueWeightDram_WeightSram.m_Graph.GetOps()[0], 0 };
    glueWeightDram_WeightSram.m_Output.push_back(glueWeightDram_WeightSram.m_Graph.GetOps()[0]);

    // Plan weightSramPlan
    Plan weightSramPlan;
    weightSramPlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                                TensorShape{ 1, 1, 3, 1 }, TensorShape{ 1, 1, 16, 1 },
                                                                TraversalOrder::Xyz, 4, QuantizationInfo()));
    weightSramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "WeightSramBuffer";
    weightSramPlan.m_OpGraph.GetBuffers().back()->m_Offset     = 0x00000F0F;
    weightSramPlan.m_OpGraph.GetBuffers().back()->m_NumStripes = 3;
    weightSramPlan.m_InputMappings  = { { weightSramPlan.m_OpGraph.GetBuffers()[0], weightSramPartInputSlot0 } };
    weightSramPlan.m_OutputMappings = { { weightSramPlan.m_OpGraph.GetBuffers()[0], weightSramPartOutputSlot0 } };

    // Plan mcePlePlan
    Plan mcePlePlan;
    mcePlePlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                            TensorShape{ 1, 160, 160, 3 }, TensorShape{ 1, 8, 8, 16 },
                                                            TraversalOrder::Xyz, 4, QuantizationInfo()));
    mcePlePlan.m_OpGraph.GetBuffers().back()->m_DebugTag = "IntermediateInputSramBuffer";
    mcePlePlan.m_OpGraph.GetBuffers().back()->m_Offset   = 0;
    mcePlePlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                            TensorShape{ 1, 1, 3, 1 }, TensorShape{ 1, 1, 16, 1 },
                                                            TraversalOrder::Xyz, 4, QuantizationInfo()));
    mcePlePlan.m_OpGraph.GetBuffers().back()->m_DebugTag = "IntermediateWeightSramBuffer";
    mcePlePlan.m_OpGraph.GetBuffers().back()->m_Offset   = 0;
    mcePlePlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::PleInputSram, CascadingBufferFormat::NHWCB,
                                                            TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                            TraversalOrder::Xyz, 0, QuantizationInfo()));
    mcePlePlan.m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputPleInputSramBuffer";
    mcePlePlan.m_OpGraph.GetBuffers().back()->m_Offset   = 0;

    mcePlePlan.m_OpGraph.AddOp(std::make_unique<MceOp>(
        Lifetime::Cascade, ethosn::command_stream::MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
        BlockConfig{ 16u, 16u }, TensorShape{ 1, 8, 8, 16 }, TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 1, 16, 1 },
        TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));
    mcePlePlan.m_OpGraph.GetOps()[0]->m_DebugTag = "MceOp";

    mcePlePlan.m_OpGraph.AddConsumer(mcePlePlan.m_OpGraph.GetBuffers()[0], mcePlePlan.m_OpGraph.GetOps()[0], 0);
    mcePlePlan.m_OpGraph.AddConsumer(mcePlePlan.m_OpGraph.GetBuffers()[1], mcePlePlan.m_OpGraph.GetOps()[0], 1);
    mcePlePlan.m_OpGraph.SetProducer(mcePlePlan.m_OpGraph.GetBuffers()[2], mcePlePlan.m_OpGraph.GetOps()[0]);

    // Adding a passthrough PLE kernel to the plan
    // The PleKernelId is expected to be PASSTHROUGH_8x8_1
    auto pleOp =
        std::make_unique<PleOp>(Lifetime::Cascade, ethosn::command_stream::PleOperation::PASSTHROUGH,
                                BlockConfig{ 8u, 8u }, 1, std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 8 } },
                                TensorShape{ 1, 4, 4, 32 }, ethosn::command_stream::DataType::U8, true);
    pleOp.get()->m_Offset     = 0x0000FFFF;
    numMemoryStripes.m_Output = 1;
    auto outBufferAndPleOp    = AddPleToOpGraph(mcePlePlan.m_OpGraph, Lifetime::Cascade, TraversalOrder::Xyz,
                                             TensorShape{ 1, 8, 8, 32 }, numMemoryStripes, std::move(pleOp),
                                             TensorShape{ 1, 80, 80, 24 }, QuantizationInfo(), operationIds);
    mcePlePlan.m_OpGraph.GetBuffers().back()->m_Offset = 0;
    mcePlePlan.m_OpGraph.AddConsumer(mcePlePlan.m_OpGraph.GetBuffers()[2], mcePlePlan.m_OpGraph.GetOps()[1], 0);

    mcePlePlan.m_InputMappings  = { { mcePlePlan.m_OpGraph.GetBuffers()[0], mcePlePartInputSlot0 },
                                   { mcePlePlan.m_OpGraph.GetBuffers()[1], mcePlePartInputSlot1 } };
    mcePlePlan.m_OutputMappings = { { mcePlePlan.m_OpGraph.GetBuffers()[3], mcePlePartOutputSlot0 } };

    // Glue glueOutputSram_OutputDram
    Glue glueOutputSram_OutputDram;
    glueOutputSram_OutputDram.m_Graph.AddOp(std::make_unique<DmaOp>());
    glueOutputSram_OutputDram.m_Graph.GetOps()[0]->m_DebugTag = "OutputDmaOp";
    glueOutputSram_OutputDram.m_InputSlot                     = { glueOutputSram_OutputDram.m_Graph.GetOps()[0], 0 };
    glueOutputSram_OutputDram.m_Output.push_back(glueOutputSram_OutputDram.m_Graph.GetOps()[0]);

    // Plan outputDramPlan
    Plan outputDramPlan;
    outputDramPlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                                TensorShape{ 1, 80, 80, 24 }, TensorShape{ 0, 0, 0, 0 },
                                                                TraversalOrder::Xyz, 0, QuantizationInfo()));
    outputDramPlan.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Output;
    outputDramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "OutputDramBuffer";
    outputDramPlan.m_InputMappings = { { outputDramPlan.m_OpGraph.GetBuffers()[0], outputDramPartInputSlot0 } };

    // Create Combination with all the plans and glues
    Combination comb;

    Elem elemInputDram  = { std::make_shared<Plan>(std::move(inputDramPlan)),
                           { { inputSramPartInputSlot0, { &glueInputDram_InputSram, true } } } };
    Elem elemInputSram  = { std::make_shared<Plan>(std::move(inputSramPlan)), {} };
    Elem elemWeightDram = { std::make_shared<Plan>(std::move(weightDramPlan)),
                            { { weightSramPartInputSlot0, { &glueWeightDram_WeightSram, true } } } };
    Elem elemWeightSram = { std::make_shared<Plan>(std::move(weightSramPlan)), {} };
    Elem elemMcePle     = { std::make_shared<Plan>(std::move(mcePlePlan)),
                        { { outputDramPartInputSlot0, { &glueOutputSram_OutputDram, true } } } };
    Elem elemOutputDram = { std::make_shared<Plan>(std::move(outputDramPlan)), {} };

    comb.m_Elems.insert(std::make_pair(0, elemInputDram));
    comb.m_PartIdsInOrder.push_back(0);
    comb.m_Elems.insert(std::make_pair(1, elemInputSram));
    comb.m_PartIdsInOrder.push_back(1);
    comb.m_Elems.insert(std::make_pair(2, elemWeightDram));
    comb.m_PartIdsInOrder.push_back(2);
    comb.m_Elems.insert(std::make_pair(3, elemWeightSram));
    comb.m_PartIdsInOrder.push_back(3);
    comb.m_Elems.insert(std::make_pair(4, elemMcePle));
    comb.m_PartIdsInOrder.push_back(4);
    comb.m_Elems.insert(std::make_pair(5, elemOutputDram));
    comb.m_PartIdsInOrder.push_back(5);

    bool dumpInputGraphToFile = false;
    if (dumpInputGraphToFile)
    {
        std::ofstream stream("CascadingCompiler MceSchedulerAgent Input.dot");
        SaveCombinationToDot(comb, graph, stream, DetailLevel::High);
    }

    // Call function under test
    OpGraph mergedOpGraph = GetOpGraphForCombination(comb, graph);

    bool dumpOutputGraphToFile = false;
    if (dumpOutputGraphToFile)
    {
        std::ofstream stream("CascadingCompiler MceSchedulerAgent Output.dot");
        SaveOpGraphToDot(mergedOpGraph, stream, DetailLevel::High);
    }

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    CascadingCompiler cascadingCompiler(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = cascadingCompiler.Compile();

    std::vector<Agent> commandStream = cascadingCompiler.GetCommandStreamOfAgents();

    const Agent& pleLAgent = commandStream[2];
    const PleL& pleLData   = pleLAgent.data.pleL;

    REQUIRE(pleLData.sramAddr == 0x0000FFFF);
    REQUIRE(pleLData.pleKernelId == cascading::PleKernelId::PASSTHROUGH_8X8_1);

    ETHOSN_UNUSED(outBufferAndPleOp);
}

// PleScheduler Agent Data Test
TEST_CASE("PleScheduler Agent Data Test", "[CascadingCompiler]")
{}

// OfmStreamer Agent Data Test
TEST_CASE("OfmStreamer Agent Data Test", "[CascadingCompiler]")
{}

//////////////////////////////////////////////////////////////////////////////////////////////
// Read After Write Dependency Tests
//////////////////////////////////////////////////////////////////////////////////////////////

// IfmStreamer Agent - Read After Write Dependency Test
TEST_CASE("IfmStreamer-OfmStreamer ReadAfterWriteDependency Test", "[CascadingCompiler]")
{}

// MceScheduler Agent - Read After Write Dependency Test
TEST_CASE("MceScheduler-IfmStreamer ReadAfterWriteDependency Test", "[CascadingCompiler]")
{}

// MceScheduler Agent - Read After Write Dependency Test
TEST_CASE("MceScheduler-WeightStreamer ReadAfterWriteDependency Test", "[CascadingCompiler]")
{}

// PleScheduler Agent - Read After Write Dependency Test
TEST_CASE("PleScheduler-IfmStreamer ReadAfterWriteDependency Test", "[CascadingCompiler]")
{}

// PleScheduler Agent - Read After Write Dependency Test
TEST_CASE("PleScheduler-MceScheduler ReadAfterWriteDependency Test", "[CascadingCompiler]")
{}

// PleScheduler Agent - Read After Write Dependency Test
TEST_CASE("PleScheduler-PleLoader ReadAfterWriteDependency Test", "[CascadingCompiler]")
{}

// OfmStreamer Agent - Read After Write Dependency Test
TEST_CASE("OfmStreamer-IfmStreamer ReadAfterWriteDependency Test", "[CascadingCompiler]")
{}

// OfmStreamer Agent - Read After Write Dependency Test
TEST_CASE("OfmStreamer-PleScheduler ReadAfterWriteDependency Test", "[CascadingCompiler]")
{}

//////////////////////////////////////////////////////////////////////////////////////////////
// Write After Read Dependency Tests
//////////////////////////////////////////////////////////////////////////////////////////////

// IfmStreamer Agent - Write After Read Dependency Test
TEST_CASE("IfmStreamer-MceScheduler WriteAfterReadDependency Test", "[CascadingCompiler]")
{}

// IfmStreamer Agent - Write After Read Dependency Test
TEST_CASE("IfmStreamer-PleScheduler WriteAfterReadDependency Test", "[CascadingCompiler]")
{}

// IfmStreamer Agent - Write After Read Dependency Test
TEST_CASE("IfmStreamer-OfmStreamer WriteAfterReadDependency Test", "[CascadingCompiler]")
{}

// WeightStreamer Agent - Write After Read Dependency Test
TEST_CASE("WeightStreamer-MceScheduler WriteAfterReadDependency Test", "[CascadingCompiler]")
{}

// MceScheduler Agent - Write After Read Dependency Test
TEST_CASE("MceScheduler-PleScheduler WriteAfterReadDependency Test", "[CascadingCompiler]")
{}

// PleScheduler Agent - Write After Read Dependency Test
TEST_CASE("PleScheduler-OfmStreamer WriteAfterReadDependency Test", "[CascadingCompiler]")
{}

//////////////////////////////////////////////////////////////////////////////////////////////
// Schedule Time Dependency Tests
//////////////////////////////////////////////////////////////////////////////////////////////

// IfmStreamer Agent - Schedule Time Dependency Test
TEST_CASE("IfmStreamer-MceScheduler ScheduleTimeDependency Test", "[CascadingCompiler]")
{}

// IfmStreamer Agent - Schedule Time Dependency Test
TEST_CASE("IfmStreamer-PleScheduler ScheduleTimeDependency Test", "[CascadingCompiler]")
{}

// IfmStreamer Agent - Schedule Time Dependency Test
TEST_CASE("IfmStreamer-OfmStreamer ScheduleTimeDependency Test", "[CascadingCompiler]")
{}

// WeightStreamer Agent - Schedule Time Dependency Test
TEST_CASE("WeightStreamer-MceScheduler ScheduleTimeDependency Test", "[CascadingCompiler]")
{}

// MceScheduler Agent
TEST_CASE("MceScheduler-PleScheduler ScheduleTimeDependency Test", "[CascadingCompiler]")
{}

// PleLoader Agent - Schedule Time Dependency Test
TEST_CASE("PLELoader-MceScheduler ScheduleTimeDependency Test", "[CascadingCompiler]")
{
    GraphOfParts graph;
    auto& parts       = graph.m_Parts;
    auto& connections = graph.m_Connections;

    auto inputDramPart  = std::make_unique<MockPart>(graph.GeneratePartId());
    auto inputSramPart  = std::make_unique<MockPart>(graph.GeneratePartId());
    auto weightDramPart = std::make_unique<MockPart>(graph.GeneratePartId());
    auto weightSramPart = std::make_unique<MockPart>(graph.GeneratePartId());
    auto mcePlePart     = std::make_unique<MockPart>(graph.GeneratePartId());
    auto outputDramPart = std::make_unique<MockPart>(graph.GeneratePartId());

    PartId inputDramPartId  = inputDramPart->GetPartId();
    PartId inputSramPartId  = inputSramPart->GetPartId();
    PartId weightDramPartId = weightDramPart->GetPartId();
    PartId weightSramPartId = weightSramPart->GetPartId();
    PartId mcePlePartId     = mcePlePart->GetPartId();
    PartId outputDramPartId = outputDramPart->GetPartId();

    parts.push_back(std::move(inputDramPart));
    parts.push_back(std::move(inputSramPart));
    parts.push_back(std::move(weightDramPart));
    parts.push_back(std::move(weightSramPart));
    parts.push_back(std::move(mcePlePart));
    parts.push_back(std::move(outputDramPart));

    PartOutputSlot inputDramPartOutputSlot0  = { inputDramPartId, 0 };
    PartOutputSlot weightDramPartOutputSlot0 = { weightDramPartId, 0 };

    PartInputSlot inputSramPartInputSlot0   = { inputSramPartId, 0 };
    PartOutputSlot inputSramPartOutputSlot0 = { inputSramPartId, 0 };

    PartInputSlot weightSramPartInputSlot0   = { weightSramPartId, 0 };
    PartOutputSlot weightSramPartOutputSlot0 = { weightSramPartId, 0 };

    PartInputSlot mcePlePartInputSlot0   = { mcePlePartId, 0 };
    PartInputSlot mcePlePartInputSlot1   = { mcePlePartId, 1 };
    PartOutputSlot mcePlePartOutputSlot0 = { mcePlePartId, 0 };

    PartInputSlot outputDramPartInputSlot0 = { outputDramPartId, 0 };

    connections[inputSramPartInputSlot0]  = inputDramPartOutputSlot0;
    connections[weightSramPartInputSlot0] = weightDramPartOutputSlot0;
    connections[mcePlePartInputSlot0]     = inputSramPartOutputSlot0;
    connections[mcePlePartInputSlot1]     = weightSramPartOutputSlot0;
    connections[outputDramPartInputSlot0] = mcePlePartOutputSlot0;

    const std::set<uint32_t> operationIds = { 0 };
    ethosn::support_library::impl::NumMemoryStripes numMemoryStripes;

    // Plan inputDramPlan
    Plan inputDramPlan;
    inputDramPlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                               TensorShape{ 1, 160, 160, 3 }, TensorShape{ 0, 0, 0, 0 },
                                                               TraversalOrder::Xyz, 0, QuantizationInfo()));
    inputDramPlan.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Input;
    inputDramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "InputDramBuffer";
    inputDramPlan.m_OutputMappings = { { inputDramPlan.m_OpGraph.GetBuffers()[0], inputDramPartOutputSlot0 } };

    // Glue glueInputDram_InputSram
    Glue glueInputDram_InputSram;
    glueInputDram_InputSram.m_Graph.AddOp(std::make_unique<DmaOp>());
    glueInputDram_InputSram.m_Graph.GetOps()[0]->m_DebugTag = "InputDmaOp";
    glueInputDram_InputSram.m_InputSlot                     = { glueInputDram_InputSram.m_Graph.GetOps()[0], 0 };
    glueInputDram_InputSram.m_Output.push_back(glueInputDram_InputSram.m_Graph.GetOps()[0]);

    // Plan inputSramPlan
    Plan inputSramPlan;
    inputSramPlan.m_OpGraph.AddBuffer(
        std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 160, 160, 3 },
                                 TensorShape{ 1, 8, 8, 16 }, TraversalOrder::Xyz, 4, QuantizationInfo()));
    inputSramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "InputSramBuffer";
    inputSramPlan.m_OpGraph.GetBuffers().back()->m_Offset     = 0x0000F0F0;
    inputSramPlan.m_OpGraph.GetBuffers().back()->m_NumStripes = 4;
    inputSramPlan.m_InputMappings  = { { inputSramPlan.m_OpGraph.GetBuffers()[0], inputSramPartInputSlot0 } };
    inputSramPlan.m_OutputMappings = { { inputSramPlan.m_OpGraph.GetBuffers()[0], inputSramPartOutputSlot0 } };

    // Plan weightDramPlan
    Plan weightDramPlan;
    weightDramPlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                                TensorShape{ 1, 1, 3, 1 }, TensorShape{ 0, 0, 0, 0 },
                                                                TraversalOrder::Xyz, 0, QuantizationInfo()));
    weightDramPlan.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Input;
    weightDramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "WeightDramBuffer";
    weightDramPlan.m_OutputMappings = { { weightDramPlan.m_OpGraph.GetBuffers()[0], weightDramPartOutputSlot0 } };

    // Glue glueWeightDram_WeightSram
    Glue glueWeightDram_WeightSram;
    glueWeightDram_WeightSram.m_Graph.AddOp(std::make_unique<DmaOp>());
    glueWeightDram_WeightSram.m_Graph.GetOps()[0]->m_DebugTag = "WeightDmaOp";
    glueWeightDram_WeightSram.m_InputSlot                     = { glueWeightDram_WeightSram.m_Graph.GetOps()[0], 0 };
    glueWeightDram_WeightSram.m_Output.push_back(glueWeightDram_WeightSram.m_Graph.GetOps()[0]);

    // Plan weightSramPlan
    Plan weightSramPlan;
    weightSramPlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                                TensorShape{ 1, 1, 3, 1 }, TensorShape{ 1, 1, 16, 1 },
                                                                TraversalOrder::Xyz, 4, QuantizationInfo()));
    weightSramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "WeightSramBuffer";
    weightSramPlan.m_OpGraph.GetBuffers().back()->m_Offset     = 0x00000F0F;
    weightSramPlan.m_OpGraph.GetBuffers().back()->m_NumStripes = 3;
    weightSramPlan.m_InputMappings  = { { weightSramPlan.m_OpGraph.GetBuffers()[0], weightSramPartInputSlot0 } };
    weightSramPlan.m_OutputMappings = { { weightSramPlan.m_OpGraph.GetBuffers()[0], weightSramPartOutputSlot0 } };

    // Plan mcePlePlan
    Plan mcePlePlan;
    mcePlePlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                            TensorShape{ 1, 160, 160, 3 }, TensorShape{ 1, 8, 8, 16 },
                                                            TraversalOrder::Xyz, 4, QuantizationInfo()));
    mcePlePlan.m_OpGraph.GetBuffers().back()->m_DebugTag = "IntermediateInputSramBuffer";
    mcePlePlan.m_OpGraph.GetBuffers().back()->m_Offset   = 0;
    mcePlePlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                            TensorShape{ 1, 1, 3, 1 }, TensorShape{ 1, 1, 16, 1 },
                                                            TraversalOrder::Xyz, 4, QuantizationInfo()));
    mcePlePlan.m_OpGraph.GetBuffers().back()->m_DebugTag = "IntermediateWeightSramBuffer";
    mcePlePlan.m_OpGraph.GetBuffers().back()->m_Offset   = 0;
    mcePlePlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::PleInputSram, CascadingBufferFormat::NHWCB,
                                                            TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                            TraversalOrder::Xyz, 0, QuantizationInfo()));
    mcePlePlan.m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputPleInputSramBuffer";
    mcePlePlan.m_OpGraph.GetBuffers().back()->m_Offset   = 0;

    mcePlePlan.m_OpGraph.AddOp(std::make_unique<MceOp>(
        Lifetime::Cascade, ethosn::command_stream::MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
        BlockConfig{ 16u, 16u }, TensorShape{ 1, 8, 8, 16 }, TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 1, 16, 1 },
        TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));
    mcePlePlan.m_OpGraph.GetOps()[0]->m_DebugTag = "MceOp";

    mcePlePlan.m_OpGraph.AddConsumer(mcePlePlan.m_OpGraph.GetBuffers()[0], mcePlePlan.m_OpGraph.GetOps()[0], 0);
    mcePlePlan.m_OpGraph.AddConsumer(mcePlePlan.m_OpGraph.GetBuffers()[1], mcePlePlan.m_OpGraph.GetOps()[0], 1);
    mcePlePlan.m_OpGraph.SetProducer(mcePlePlan.m_OpGraph.GetBuffers()[2], mcePlePlan.m_OpGraph.GetOps()[0]);

    // Adding a passthrough PLE kernel to the plan
    // The PleKernelId is expected to be PASSTHROUGH_8x8_1
    auto pleOp =
        std::make_unique<PleOp>(Lifetime::Cascade, ethosn::command_stream::PleOperation::PASSTHROUGH,
                                BlockConfig{ 8u, 8u }, 1, std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 8 } },
                                TensorShape{ 1, 4, 4, 32 }, ethosn::command_stream::DataType::U8, true);
    pleOp.get()->m_Offset     = 0x0000FFFF;
    numMemoryStripes.m_Output = 1;
    auto outBufferAndPleOp    = AddPleToOpGraph(mcePlePlan.m_OpGraph, Lifetime::Cascade, TraversalOrder::Xyz,
                                             TensorShape{ 1, 8, 8, 32 }, numMemoryStripes, std::move(pleOp),
                                             TensorShape{ 1, 80, 80, 24 }, QuantizationInfo(), operationIds);
    mcePlePlan.m_OpGraph.GetBuffers().back()->m_Offset = 0;
    mcePlePlan.m_OpGraph.AddConsumer(mcePlePlan.m_OpGraph.GetBuffers()[2], mcePlePlan.m_OpGraph.GetOps()[1], 0);

    mcePlePlan.m_InputMappings  = { { mcePlePlan.m_OpGraph.GetBuffers()[0], mcePlePartInputSlot0 },
                                   { mcePlePlan.m_OpGraph.GetBuffers()[1], mcePlePartInputSlot1 } };
    mcePlePlan.m_OutputMappings = { { mcePlePlan.m_OpGraph.GetBuffers()[3], mcePlePartOutputSlot0 } };

    // Glue glueOutputSram_OutputDram
    Glue glueOutputSram_OutputDram;
    glueOutputSram_OutputDram.m_Graph.AddOp(std::make_unique<DmaOp>());
    glueOutputSram_OutputDram.m_Graph.GetOps()[0]->m_DebugTag = "OutputDmaOp";
    glueOutputSram_OutputDram.m_InputSlot                     = { glueOutputSram_OutputDram.m_Graph.GetOps()[0], 0 };
    glueOutputSram_OutputDram.m_Output.push_back(glueOutputSram_OutputDram.m_Graph.GetOps()[0]);

    // Plan outputDramPlan
    Plan outputDramPlan;
    outputDramPlan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                                TensorShape{ 1, 80, 80, 24 }, TensorShape{ 0, 0, 0, 0 },
                                                                TraversalOrder::Xyz, 0, QuantizationInfo()));
    outputDramPlan.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Output;
    outputDramPlan.m_OpGraph.GetBuffers().back()->m_DebugTag   = "OutputDramBuffer";
    outputDramPlan.m_InputMappings = { { outputDramPlan.m_OpGraph.GetBuffers()[0], outputDramPartInputSlot0 } };

    // Create Combination with all the plans and glues
    Combination comb;

    Elem elemInputDram  = { std::make_shared<Plan>(std::move(inputDramPlan)),
                           { { inputSramPartInputSlot0, { &glueInputDram_InputSram, true } } } };
    Elem elemInputSram  = { std::make_shared<Plan>(std::move(inputSramPlan)), {} };
    Elem elemWeightDram = { std::make_shared<Plan>(std::move(weightDramPlan)),
                            { { weightSramPartInputSlot0, { &glueWeightDram_WeightSram, true } } } };
    Elem elemWeightSram = { std::make_shared<Plan>(std::move(weightSramPlan)), {} };
    Elem elemMcePle     = { std::make_shared<Plan>(std::move(mcePlePlan)),
                        { { outputDramPartInputSlot0, { &glueOutputSram_OutputDram, true } } } };
    Elem elemOutputDram = { std::make_shared<Plan>(std::move(outputDramPlan)), {} };

    comb.m_Elems.insert(std::make_pair(0, elemInputDram));
    comb.m_PartIdsInOrder.push_back(0);
    comb.m_Elems.insert(std::make_pair(1, elemInputSram));
    comb.m_PartIdsInOrder.push_back(1);
    comb.m_Elems.insert(std::make_pair(2, elemWeightDram));
    comb.m_PartIdsInOrder.push_back(2);
    comb.m_Elems.insert(std::make_pair(3, elemWeightSram));
    comb.m_PartIdsInOrder.push_back(3);
    comb.m_Elems.insert(std::make_pair(4, elemMcePle));
    comb.m_PartIdsInOrder.push_back(4);
    comb.m_Elems.insert(std::make_pair(5, elemOutputDram));
    comb.m_PartIdsInOrder.push_back(5);

    bool dumpInputGraphToFile = false;
    if (dumpInputGraphToFile)
    {
        std::ofstream stream("CascadingCompiler MceSchedulerAgent Input.dot");
        SaveCombinationToDot(comb, graph, stream, DetailLevel::High);
    }

    // Call function under test
    OpGraph mergedOpGraph = GetOpGraphForCombination(comb, graph);

    bool dumpOutputGraphToFile = false;
    if (dumpOutputGraphToFile)
    {
        std::ofstream stream("CascadingCompiler MceSchedulerAgent Output.dot");
        SaveOpGraphToDot(mergedOpGraph, stream, DetailLevel::High);
    }

    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    CascadingCompiler cascadingCompiler(mergedOpGraph, operationIds, hwCaps, compOpt);
    std::unique_ptr<CompiledNetwork> compiledNetwork = cascadingCompiler.Compile();

    std::vector<Agent> commandStream = cascadingCompiler.GetCommandStreamOfAgents();

    const Agent& pleLAgent                   = commandStream[2];
    const Agent& mceSAgent                   = commandStream[3];
    const Dependency& pleLScheduleDependency = pleLAgent.info.scheduleDependencies.at(0);

    uint32_t numberOfMceStripes = mceSAgent.data.mce.numStripes.ofmHeight * mceSAgent.data.mce.numStripes.ofmWidth;
    REQUIRE(pleLScheduleDependency.relativeAgentId == 1);
    REQUIRE(pleLScheduleDependency.outerRatio.other == numberOfMceStripes);
    REQUIRE(pleLScheduleDependency.outerRatio.self == 1);
    REQUIRE(pleLScheduleDependency.innerRatio.other == numberOfMceStripes);
    REQUIRE(pleLScheduleDependency.innerRatio.self == 1);
    REQUIRE(pleLScheduleDependency.boundary == 0);

    ETHOSN_UNUSED(outBufferAndPleOp);
}

// PleLoader Agent - Schedule Time Dependency Test
TEST_CASE("PLELoader-PleScheduler ScheduleTimeDependency Test", "[CascadingCompiler]")
{}

// PleScheduler Agent - Schedule Time Dependency Test
TEST_CASE("PleScheduler-OfmStreamer ScheduleTimeDependency Test", "[CascadingCompiler]")
{}

// OfmStreamer Agent - Schedule Time Dependency Test
TEST_CASE("OfmStreamer-IfmStreamer ScheduleTimeDependency Test", "[CascadingCompiler]")
{}
