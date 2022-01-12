//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/DebuggingContext.hpp"
#include "../src/GraphNodes.hpp"
#include "../src/SramAllocator.hpp"
#include "../src/cascading/Cascading.hpp"
#include "../src/cascading/CombinerDFS.hpp"
#include "../src/cascading/StripeHelper.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>

#include <fstream>

using namespace ethosn::support_library;
using namespace ethosn::command_stream;
using PleKernelId = ethosn::command_stream::cascading::PleKernelId;

TEST_CASE("IsPartSiso", "[CombinerDFS]")
{
    // Create graph:
    //
    //          D
    //          |
    //  A - B - C
    //          |
    //          E
    //

    GraphOfParts graph;
    auto& parts       = graph.m_Parts;
    auto& connections = graph.m_Connections;

    auto pA = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pC = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pD = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pE = std::make_unique<MockPart>(graph.GeneratePartId());

    BasePart& partA = *pA;
    BasePart& partB = *pB;
    BasePart& partC = *pC;
    BasePart& partD = *pD;
    BasePart& partE = *pE;

    PartId partAId = pA->GetPartId();
    PartId partBId = pB->GetPartId();
    PartId partCId = pC->GetPartId();
    PartId partDId = pD->GetPartId();
    PartId partEId = pE->GetPartId();
    parts.push_back(std::move(pA));
    parts.push_back(std::move(pB));
    parts.push_back(std::move(pC));
    parts.push_back(std::move(pD));
    parts.push_back(std::move(pE));

    PartOutputSlot partAOutputSlot0 = { partAId, 0 };

    PartInputSlot partBInputSlot0   = { partBId, 0 };
    PartOutputSlot partBOutputSlot0 = { partBId, 0 };

    PartInputSlot partCInputSlot0   = { partCId, 0 };
    PartOutputSlot partCOutputSlot0 = { partCId, 0 };

    PartInputSlot partDInputSlot0 = { partDId, 0 };

    PartInputSlot partEInputSlot0 = { partEId, 0 };

    connections[partBInputSlot0] = partAOutputSlot0;
    connections[partCInputSlot0] = partBOutputSlot0;
    connections[partDInputSlot0] = partCOutputSlot0;
    connections[partEInputSlot0] = partCOutputSlot0;

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(&compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Combiner combiner(graph, hwCaps, estOpt, debuggingContext);

    REQUIRE(combiner.IsPartSiso(partA) == false);
    REQUIRE(combiner.IsPartSiso(partB) == true);
    REQUIRE(combiner.IsPartSiso(partC) == false);
    REQUIRE(combiner.IsPartSiso(partD) == false);
    REQUIRE(combiner.IsPartSiso(partE) == false);
}

TEST_CASE("IsPartSimo", "[CombinerDFS]")
{
    // Create graph:
    //
    //          D
    //          |
    //  A - B - C
    //          |
    //          E
    //

    GraphOfParts graph;
    auto& parts       = graph.m_Parts;
    auto& connections = graph.m_Connections;

    auto pA = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pC = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pD = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pE = std::make_unique<MockPart>(graph.GeneratePartId());

    BasePart& partA = *pA;
    BasePart& partB = *pB;
    BasePart& partC = *pC;
    BasePart& partD = *pD;
    BasePart& partE = *pE;

    PartId partAId = pA->GetPartId();
    PartId partBId = pB->GetPartId();
    PartId partCId = pC->GetPartId();
    PartId partDId = pD->GetPartId();
    PartId partEId = pE->GetPartId();
    parts.push_back(std::move(pA));
    parts.push_back(std::move(pB));
    parts.push_back(std::move(pC));
    parts.push_back(std::move(pD));
    parts.push_back(std::move(pE));

    PartOutputSlot partAOutputSlot0 = { partAId, 0 };

    PartInputSlot partBInputSlot0   = { partBId, 0 };
    PartOutputSlot partBOutputSlot0 = { partBId, 0 };

    PartInputSlot partCInputSlot0   = { partCId, 0 };
    PartOutputSlot partCOutputSlot0 = { partCId, 0 };

    PartInputSlot partDInputSlot0 = { partDId, 0 };

    PartInputSlot partEInputSlot0 = { partEId, 0 };

    connections[partBInputSlot0] = partAOutputSlot0;
    connections[partCInputSlot0] = partBOutputSlot0;
    connections[partDInputSlot0] = partCOutputSlot0;
    connections[partEInputSlot0] = partCOutputSlot0;

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(&compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Combiner combiner(graph, hwCaps, estOpt, debuggingContext);

    REQUIRE(combiner.IsPartSimo(partA) == false);
    REQUIRE(combiner.IsPartSimo(partB) == false);
    REQUIRE(combiner.IsPartSimo(partC) == true);
    REQUIRE(combiner.IsPartSimo(partD) == false);
    REQUIRE(combiner.IsPartSimo(partE) == false);
}

TEST_CASE("IsPartMiso", "[CombinerDFS]")
{
    // Create graph:
    //
    //  A
    //  |
    //  C - D
    //  |
    //  B
    //
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

    PartId partAId = pA->GetPartId();
    PartId partBId = pB->GetPartId();
    PartId partCId = pC->GetPartId();
    PartId partDId = pD->GetPartId();
    parts.push_back(std::move(pA));
    parts.push_back(std::move(pB));
    parts.push_back(std::move(pC));
    parts.push_back(std::move(pD));

    PartOutputSlot partAOutputSlot0 = { partAId, 0 };

    PartOutputSlot partBOutputSlot0 = { partBId, 0 };

    PartInputSlot partCInputSlot0   = { partCId, 0 };
    PartInputSlot partCInputSlot1   = { partCId, 1 };
    PartOutputSlot partCOutputSlot0 = { partCId, 0 };

    PartInputSlot partDInputSlot0 = { partDId, 0 };

    connections[partCInputSlot0] = partAOutputSlot0;
    connections[partCInputSlot1] = partBOutputSlot0;
    connections[partDInputSlot0] = partCOutputSlot0;

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(&compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Combiner combiner(graph, hwCaps, estOpt, debuggingContext);

    REQUIRE(combiner.IsPartMiso(partA) == false);
    REQUIRE(combiner.IsPartMiso(partB) == false);
    REQUIRE(combiner.IsPartMiso(partC) == true);
    REQUIRE(combiner.IsPartMiso(partD) == false);
}

TEST_CASE("IsPartMimo", "[CombinerDFS]")
{
    // Create graph:
    //
    //  A    E
    //  |    |
    //   - - C - D
    //       |
    //       B
    //
    GraphOfParts graph;
    auto& parts       = graph.m_Parts;
    auto& connections = graph.m_Connections;

    auto pA = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pC = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pD = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pE = std::make_unique<MockPart>(graph.GeneratePartId());

    BasePart& partA = *pA;
    BasePart& partB = *pB;
    BasePart& partC = *pC;
    BasePart& partD = *pD;
    BasePart& partE = *pE;

    PartId partAId = pA->GetPartId();
    PartId partBId = pB->GetPartId();
    PartId partCId = pC->GetPartId();
    PartId partDId = pD->GetPartId();
    PartId partEId = pE->GetPartId();
    parts.push_back(std::move(pA));
    parts.push_back(std::move(pB));
    parts.push_back(std::move(pC));
    parts.push_back(std::move(pD));
    parts.push_back(std::move(pE));

    PartOutputSlot partAOutputSlot0 = { partAId, 0 };

    PartOutputSlot partBOutputSlot0 = { partBId, 0 };

    PartInputSlot partCInputSlot0   = { partCId, 0 };
    PartInputSlot partCInputSlot1   = { partCId, 1 };
    PartOutputSlot partCOutputSlot0 = { partCId, 0 };
    PartOutputSlot partCOutputSlot1 = { partCId, 1 };

    PartInputSlot partDInputSlot0 = { partDId, 0 };

    PartInputSlot partEInputSlot0 = { partEId, 0 };

    connections[partCInputSlot0] = partAOutputSlot0;
    connections[partCInputSlot1] = partBOutputSlot0;
    connections[partDInputSlot0] = partCOutputSlot0;
    connections[partEInputSlot0] = partCOutputSlot1;

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(&compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Combiner combiner(graph, hwCaps, estOpt, debuggingContext);

    REQUIRE(combiner.IsPartMimo(partA) == false);
    REQUIRE(combiner.IsPartMimo(partB) == false);
    REQUIRE(combiner.IsPartMimo(partC) == true);
    REQUIRE(combiner.IsPartMimo(partD) == false);
    REQUIRE(combiner.IsPartMimo(partE) == false);
}

TEST_CASE("IsPartInput and IsPartOutput", "[CombinerDFS]")
{
    // Create graph:
    //
    //  A    E
    //  |    |
    //   - - C - D
    //       |
    //       B
    //
    GraphOfParts graph;
    auto& parts       = graph.m_Parts;
    auto& connections = graph.m_Connections;

    auto pA = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pC = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pD = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pE = std::make_unique<MockPart>(graph.GeneratePartId());

    BasePart& partA = *pA;
    BasePart& partB = *pB;
    BasePart& partC = *pC;
    BasePart& partD = *pD;
    BasePart& partE = *pE;

    PartId partAId = pA->GetPartId();
    PartId partBId = pB->GetPartId();
    PartId partCId = pC->GetPartId();
    PartId partDId = pD->GetPartId();
    PartId partEId = pE->GetPartId();
    parts.push_back(std::move(pA));
    parts.push_back(std::move(pB));
    parts.push_back(std::move(pC));
    parts.push_back(std::move(pD));
    parts.push_back(std::move(pE));

    PartOutputSlot partAOutputSlot0 = { partAId, 0 };

    PartOutputSlot partBOutputSlot0 = { partBId, 0 };

    PartInputSlot partCInputSlot0   = { partCId, 0 };
    PartInputSlot partCInputSlot1   = { partCId, 1 };
    PartOutputSlot partCOutputSlot0 = { partCId, 0 };
    PartOutputSlot partCOutputSlot1 = { partCId, 1 };

    PartInputSlot partDInputSlot0 = { partDId, 0 };

    PartInputSlot partEInputSlot0 = { partEId, 0 };

    connections[partCInputSlot0] = partAOutputSlot0;
    connections[partCInputSlot1] = partBOutputSlot0;
    connections[partDInputSlot0] = partCOutputSlot0;
    connections[partEInputSlot0] = partCOutputSlot1;

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(&compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Combiner combiner(graph, hwCaps, estOpt, debuggingContext);

    REQUIRE(combiner.IsPartInput(partA) == true);
    REQUIRE(combiner.IsPartOutput(partA) == false);

    REQUIRE(combiner.IsPartInput(partB) == true);
    REQUIRE(combiner.IsPartOutput(partB) == false);

    REQUIRE(combiner.IsPartInput(partC) == false);
    REQUIRE(combiner.IsPartOutput(partC) == false);

    REQUIRE(combiner.IsPartInput(partD) == false);
    REQUIRE(combiner.IsPartOutput(partD) == true);

    REQUIRE(combiner.IsPartInput(partE) == false);
    REQUIRE(combiner.IsPartOutput(partE) == true);
}

TEST_CASE("IsPartSo and IsPartMo", "[CombinerDFS]")
{
    // Create graph:
    //
    //  A    E
    //  |    |
    //   - - C - D
    //       |
    //       B - F
    //
    GraphOfParts graph;
    auto& parts       = graph.m_Parts;
    auto& connections = graph.m_Connections;

    auto pA = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pC = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pD = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pE = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pF = std::make_unique<MockPart>(graph.GeneratePartId());

    BasePart& partA = *pA;
    BasePart& partB = *pB;
    BasePart& partC = *pC;
    BasePart& partD = *pD;
    BasePart& partE = *pE;
    BasePart& partF = *pF;

    PartId partAId = pA->GetPartId();
    PartId partBId = pB->GetPartId();
    PartId partCId = pC->GetPartId();
    PartId partDId = pD->GetPartId();
    PartId partEId = pE->GetPartId();
    PartId partFId = pF->GetPartId();
    parts.push_back(std::move(pA));
    parts.push_back(std::move(pB));
    parts.push_back(std::move(pC));
    parts.push_back(std::move(pD));
    parts.push_back(std::move(pE));
    parts.push_back(std::move(pF));

    PartOutputSlot partAOutputSlot0 = { partAId, 0 };

    PartOutputSlot partBOutputSlot0 = { partBId, 0 };

    PartInputSlot partCInputSlot0   = { partCId, 0 };
    PartInputSlot partCInputSlot1   = { partCId, 1 };
    PartOutputSlot partCOutputSlot0 = { partCId, 0 };
    PartOutputSlot partCOutputSlot1 = { partCId, 1 };

    PartInputSlot partDInputSlot0 = { partDId, 0 };

    PartInputSlot partEInputSlot0 = { partEId, 0 };

    PartInputSlot partFInputSlot0 = { partFId, 0 };

    connections[partCInputSlot0] = partAOutputSlot0;
    connections[partCInputSlot1] = partBOutputSlot0;
    connections[partDInputSlot0] = partCOutputSlot0;
    connections[partEInputSlot0] = partCOutputSlot1;
    connections[partFInputSlot0] = partBOutputSlot0;

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(&compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Combiner combiner(graph, hwCaps, estOpt, debuggingContext);

    REQUIRE(combiner.IsPartSo(partA) == true);
    REQUIRE(combiner.IsPartMo(partA) == false);

    REQUIRE(combiner.IsPartSo(partB) == false);
    REQUIRE(combiner.IsPartMo(partB) == true);

    REQUIRE(combiner.IsPartSo(partC) == false);
    REQUIRE(combiner.IsPartMo(partC) == true);

    REQUIRE(combiner.IsPartSo(partD) == false);
    REQUIRE(combiner.IsPartMo(partD) == false);

    REQUIRE(combiner.IsPartSo(partE) == false);
    REQUIRE(combiner.IsPartMo(partE) == false);

    REQUIRE(combiner.IsPartSo(partF) == false);
    REQUIRE(combiner.IsPartMo(partF) == false);
}

// Manually creates a partial combination starting and ending in Sram and converts it to an OpGraph using the GetOpGraphForCombination.
// The topology is chosen to test cases including:
//      * Partial combinations starting and ending in Sram
//      * Glue containing input and output DmaOps, e.g. DmaOp -> DramBuffer -> DmaOp
// ( A ) -> g -> ( B ) -> g -> ( C )
TEST_CASE("GetOpGraphForDfsCombinationPartialSram", "[CombinerDFS]")
{
    GraphOfParts graph;
    auto& parts       = graph.m_Parts;
    auto& connections = graph.m_Connections;

    auto pA        = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB        = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pC        = std::make_unique<MockPart>(graph.GeneratePartId());
    PartId partAId = pA->GetPartId();
    PartId partBId = pB->GetPartId();
    PartId partCId = pC->GetPartId();
    parts.push_back(std::move(pA));
    parts.push_back(std::move(pB));
    parts.push_back(std::move(pC));

    PartOutputSlot partAOutputSlot0 = { partAId, 0 };

    PartInputSlot partBInputSlot0   = { partBId, 0 };
    PartOutputSlot partBOutputSlot0 = { partBId, 0 };

    PartInputSlot partCInputSlot0 = { partCId, 0 };

    connections[partBInputSlot0] = partAOutputSlot0;
    connections[partCInputSlot0] = partBOutputSlot0;

    // Plan A
    Plan planA;
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 0, QuantizationInfo()));
    planA.m_OpGraph.GetBuffers().back()->m_DebugTag = "InputSramA";
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 0, QuantizationInfo()));
    planA.m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputSramA";
    planA.m_OutputMappings                          = { { planA.m_OpGraph.GetBuffers()[1], partAOutputSlot0 } };
    planA.m_OpGraph.AddOp(
        std::make_unique<MceOp>(Lifetime::Atomic, MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
                                BlockConfig{ 16u, 16u }, TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                TensorShape{ 1, 1, 1, 16 }, TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));
    planA.m_OpGraph.GetOps()[0]->m_DebugTag = "MceA";
    planA.m_OpGraph.AddConsumer(planA.m_OpGraph.GetBuffers()[0], planA.m_OpGraph.GetOps()[0], 0);
    planA.m_OpGraph.SetProducer(planA.m_OpGraph.GetBuffers()[1], planA.m_OpGraph.GetOps()[0]);

    // Glue between A and B
    Glue glueA_B;
    glueA_B.m_Graph.AddOp(std::make_unique<DmaOp>());
    glueA_B.m_Graph.GetOps()[0]->m_DebugTag = "InputDma";
    glueA_B.m_Graph.AddOp(std::make_unique<DmaOp>());
    glueA_B.m_Graph.GetOps()[1]->m_DebugTag = "OutputDma";
    glueA_B.m_InputSlot                     = { glueA_B.m_Graph.GetOps()[0], 0 };
    glueA_B.m_Output.push_back(glueA_B.m_Graph.GetOps()[1]);
    glueA_B.m_OutDmaOffset = 1;
    glueA_B.m_Graph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 0, QuantizationInfo()));
    glueA_B.m_Graph.GetBuffers().back()->m_DebugTag = "DramBuffer";
    glueA_B.m_Graph.AddConsumer(glueA_B.m_Graph.GetBuffers()[0], glueA_B.m_Graph.GetOps()[1], 0);
    glueA_B.m_Graph.SetProducer(glueA_B.m_Graph.GetBuffers()[0], glueA_B.m_Graph.GetOps()[0]);

    // Plan B
    Plan planB;
    planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planB.m_OpGraph.GetBuffers().back()->m_DebugTag = "InputSramB";
    planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planB.m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputSramB";
    planB.m_InputMappings                           = { { planB.m_OpGraph.GetBuffers()[0], partBInputSlot0 } };
    planB.m_OutputMappings                          = { { planB.m_OpGraph.GetBuffers()[1], partBOutputSlot0 } };
    planB.m_OpGraph.AddOp(
        std::make_unique<MceOp>(Lifetime::Atomic, MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
                                BlockConfig{ 16u, 16u }, TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                TensorShape{ 1, 1, 1, 16 }, TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));
    planB.m_OpGraph.GetOps()[0]->m_DebugTag = "MceB";
    planB.m_OpGraph.AddConsumer(planB.m_OpGraph.GetBuffers()[0], planB.m_OpGraph.GetOps()[0], 0);
    planB.m_OpGraph.SetProducer(planB.m_OpGraph.GetBuffers()[1], planB.m_OpGraph.GetOps()[0]);

    // Glue between B and C
    Glue glueB_C;
    glueB_C.m_Graph.AddOp(std::make_unique<DmaOp>());
    glueB_C.m_Graph.GetOps()[0]->m_DebugTag = "InputDmaC";
    glueB_C.m_InputSlot                     = { glueB_C.m_Graph.GetOps()[0], 0 };
    glueB_C.m_Output.push_back(glueB_C.m_Graph.GetOps()[0]);

    // Plan C
    Plan planC;
    planC.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planC.m_OpGraph.GetBuffers().back()->m_DebugTag = "InputSramC";
    planC.m_InputMappings                           = { { planC.m_OpGraph.GetBuffers()[0], partCInputSlot0 } };

    // Create Combination with all the plans and glues
    Combination comb;

    Elem elemA = { std::make_shared<Plan>(std::move(planA)), { { partBInputSlot0, { &glueA_B, true } } } };
    Elem elemB = { std::make_shared<Plan>(std::move(planB)), { { partCInputSlot0, { &glueB_C, true } } } };
    Elem elemC = { std::make_shared<Plan>(std::move(planC)), {} };

    comb.m_Elems.insert(std::make_pair(0, elemA));
    comb.m_PartIdsInOrder.push_back(0);
    comb.m_Elems.insert(std::make_pair(1, elemB));
    comb.m_PartIdsInOrder.push_back(1);
    comb.m_Elems.insert(std::make_pair(2, elemC));
    comb.m_PartIdsInOrder.push_back(2);

    bool dumpInputGraphToFile = false;
    if (dumpInputGraphToFile)
    {
        std::ofstream stream("GetOpGraphForDfsCombinationPartialSram Input.dot");
        SaveCombinationToDot(comb, graph, stream, DetailLevel::High);
    }

    // Call function under test
    OpGraph combOpGraph = GetOpGraphForCombination(comb, graph);

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("GetOpGraphForDfsCombinationPartialSram Output.dot");
        SaveOpGraphToDot(combOpGraph, stream, DetailLevel::High);
    }

    REQUIRE(combOpGraph.GetBuffers().size() == 6);
    REQUIRE(combOpGraph.GetBuffers()[0]->m_DebugTag == "InputSramA");
    REQUIRE(combOpGraph.GetBuffers()[1]->m_DebugTag == "OutputSramA");
    REQUIRE(combOpGraph.GetBuffers()[2]->m_DebugTag == "DramBuffer");
    REQUIRE(combOpGraph.GetBuffers()[3]->m_DebugTag == "InputSramB");
    REQUIRE(combOpGraph.GetBuffers()[4]->m_DebugTag == "OutputSramB");
    REQUIRE(combOpGraph.GetBuffers()[5]->m_DebugTag == "InputSramC");

    REQUIRE(combOpGraph.GetOps().size() == 5);
    REQUIRE(combOpGraph.GetOps()[0]->m_DebugTag == "MceA");
    REQUIRE(combOpGraph.GetOps()[1]->m_DebugTag == "InputDma");
    REQUIRE(combOpGraph.GetOps()[2]->m_DebugTag == "OutputDma");
    REQUIRE(combOpGraph.GetOps()[3]->m_DebugTag == "MceB");
    REQUIRE(combOpGraph.GetOps()[4]->m_DebugTag == "InputDmaC");

    REQUIRE(combOpGraph.GetProducer(combOpGraph.GetBuffers()[0]) == nullptr);
    REQUIRE(combOpGraph.GetProducer(combOpGraph.GetBuffers()[1])->m_DebugTag == "MceA");
    REQUIRE(combOpGraph.GetProducer(combOpGraph.GetBuffers()[2])->m_DebugTag == "InputDma");
    REQUIRE(combOpGraph.GetProducer(combOpGraph.GetBuffers()[3])->m_DebugTag == "OutputDma");
    REQUIRE(combOpGraph.GetProducer(combOpGraph.GetBuffers()[4])->m_DebugTag == "MceB");
    REQUIRE(combOpGraph.GetProducer(combOpGraph.GetBuffers()[5])->m_DebugTag == "InputDmaC");

    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[0]).size() == 1);
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[0])[0].first->m_DebugTag == "MceA");
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[0])[0].second == 0);

    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[1]).size() == 1);
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[1])[0].first->m_DebugTag == "InputDma");
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[1])[0].second == 0);

    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[2]).size() == 1);
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[2])[0].first->m_DebugTag == "OutputDma");
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[2])[0].second == 0);

    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[3]).size() == 1);
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[3])[0].first->m_DebugTag == "MceB");
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[3])[0].second == 0);

    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[4]).size() == 1);
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[4])[0].first->m_DebugTag == "InputDmaC");
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[4])[0].second == 0);

    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[5]).size() == 0);
}

// Manually creates a partial combination starting and ending in Sram and converts it to an OpGraph using the GetOpGraphForCombination.
// The topology is chosen to test cases including:
//      * Partial combinations starting and ending in Sram
//      * Glue containing multiple output DmaOps and direct connections to Dram buffers
// ( A ) -> g --> ( B )
//            \   (   )
//            |   (   )
//            \-> ( C )
//            \-> ( D )
TEST_CASE("GetOpGraphForDfsCombinationPartialDram", "[CombinerDFS]")
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

    // Plan A
    Plan planA;
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planA.m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputSramA";
    planA.m_OutputMappings                          = { { planA.m_OpGraph.GetBuffers()[0], partAOutputSlot } };

    // Plan B
    Plan planB;
    planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planB.m_OpGraph.GetBuffers().back()->m_DebugTag = "InputSramB";
    planB.m_InputMappings                           = { { planB.m_OpGraph.GetBuffers()[0], partBInputSlot } };

    // Plan C
    Plan planC;
    planC.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planC.m_OpGraph.GetBuffers().back()->m_DebugTag = "InputSramC";
    planC.m_InputMappings                           = { { planC.m_OpGraph.GetBuffers()[0], partCInputSlot } };

    // Plan D
    Plan planD;
    planD.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planD.m_OpGraph.GetBuffers().back()->m_DebugTag = "InputDramD";
    planD.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planD.m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputSramD";
    planD.m_InputMappings                           = { { planD.m_OpGraph.GetBuffers()[0], partDInputSlot } };
    planD.m_OpGraph.AddOp(std::make_unique<DmaOp>());
    planD.m_OpGraph.GetOps()[0]->m_DebugTag = "DmaToSramD";
    planD.m_OpGraph.AddConsumer(planD.m_OpGraph.GetBuffers()[0], planD.m_OpGraph.GetOps()[0], 0);
    planD.m_OpGraph.SetProducer(planD.m_OpGraph.GetBuffers()[1], planD.m_OpGraph.GetOps()[0]);

    // Create Combination with all the plans and glues
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

    // Access Glue Buffer and Ops and set an appropriate name for debugging purposes
    auto elemIt                                                = combGlued.m_Elems.find(partA.GetPartId());
    auto elemAB                                                = elemIt->second.m_Glues.find(edgeA2B.m_Destination);
    elemAB->second.m_Glue->m_Graph.GetBuffers()[0]->m_DebugTag = "DramBuffer";
    elemAB->second.m_Glue->m_Graph.GetOps()[0]->m_DebugTag     = "InputDmaGlue";
    elemAB->second.m_Glue->m_Graph.GetOps()[1]->m_DebugTag     = "OutputDmaB";
    elemAB->second.m_Glue->m_Graph.GetOps()[2]->m_DebugTag     = "OutputDmaC";

    bool dumpInputGraphToFile = false;
    if (dumpInputGraphToFile)
    {
        std::ofstream stream("GetOpGraphForDfsCombinationPartialDram Input.dot");
        SaveCombinationToDot(combGlued, graph, stream, DetailLevel::High);
    }

    // Call function under test
    OpGraph combOpGraph = GetOpGraphForCombination(combGlued, graph);

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("GetOpGraphForDfsCombinationPartialDram Output.dot");
        SaveOpGraphToDot(combOpGraph, stream, DetailLevel::High);
    }

    REQUIRE(combOpGraph.GetBuffers().size() == 5);
    REQUIRE(combOpGraph.GetBuffers()[0]->m_DebugTag == "OutputSramA");
    REQUIRE(combOpGraph.GetBuffers()[1]->m_DebugTag == "DramBuffer");
    REQUIRE(combOpGraph.GetBuffers()[2]->m_DebugTag == "InputSramB");
    REQUIRE(combOpGraph.GetBuffers()[3]->m_DebugTag == "OutputSramD");
    REQUIRE(combOpGraph.GetBuffers()[4]->m_DebugTag == "InputSramC");

    REQUIRE(combOpGraph.GetOps().size() == 4);
    REQUIRE(combOpGraph.GetOps()[0]->m_DebugTag == "InputDmaGlue");
    REQUIRE(combOpGraph.GetOps()[1]->m_DebugTag == "OutputDmaB");
    REQUIRE(combOpGraph.GetOps()[2]->m_DebugTag == "DmaToSramD");
    REQUIRE(combOpGraph.GetOps()[3]->m_DebugTag == "OutputDmaC");

    REQUIRE(combOpGraph.GetProducer(combOpGraph.GetBuffers()[0]) == nullptr);
    REQUIRE(combOpGraph.GetProducer(combOpGraph.GetBuffers()[1])->m_DebugTag == "InputDmaGlue");
    REQUIRE(combOpGraph.GetProducer(combOpGraph.GetBuffers()[2])->m_DebugTag == "OutputDmaB");
    REQUIRE(combOpGraph.GetProducer(combOpGraph.GetBuffers()[3])->m_DebugTag == "DmaToSramD");
    REQUIRE(combOpGraph.GetProducer(combOpGraph.GetBuffers()[4])->m_DebugTag == "OutputDmaC");

    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[0]).size() == 1);
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[0])[0].first->m_DebugTag == "InputDmaGlue");
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[0])[0].second == 0);

    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[1]).size() == 3);
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[1])[0].first->m_DebugTag == "OutputDmaB");
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[1])[0].second == 0);
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[1])[1].first->m_DebugTag == "DmaToSramD");
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[1])[1].second == 0);
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[1])[2].first->m_DebugTag == "OutputDmaC");
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[1])[2].second == 0);

    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[2]).size() == 0);

    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[3]).size() == 0);

    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[4]).size() == 0);
}

/// Manually creates a Combination and then converts it to an OpGraph using GetOpGraphForCombination, and checking
/// the resulting graph structure is correct.
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
TEST_CASE("GetOpGraphForDfsCombination", "[CombinerDFS]")
{
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

    // Call function under test
    OpGraph combOpGraph = GetOpGraphForCombination(comb, graph);

    // For easier debugging of this test (and so that you can see the pretty graph!), dump the output to a file
    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("GetOpGraphForCombination Output.dot");
        SaveOpGraphToDot(combOpGraph, stream, DetailLevel::High);
    }

    // Check the resulting OpGraph is correct
    REQUIRE(combOpGraph.GetBuffers().size() == 7);
    REQUIRE(combOpGraph.GetBuffers()[0]->m_DebugTag == "InputDram");
    REQUIRE(combOpGraph.GetBuffers()[1]->m_DebugTag == "InputSram1");
    REQUIRE(combOpGraph.GetBuffers()[2]->m_DebugTag == "OutputSram1");
    REQUIRE(combOpGraph.GetBuffers()[3]->m_DebugTag == "OutputSram2");
    REQUIRE(combOpGraph.GetBuffers()[4]->m_DebugTag == "OutputDram1");
    REQUIRE(combOpGraph.GetBuffers()[5]->m_DebugTag == "OutputDram2");
    REQUIRE(combOpGraph.GetBuffers()[6]->m_DebugTag == "OutputDram3");

    REQUIRE(combOpGraph.GetOps().size() == 5);
    REQUIRE(combOpGraph.GetOps()[0]->m_DebugTag == "InputDma");
    REQUIRE(combOpGraph.GetOps()[1]->m_DebugTag == "Mce2");
    REQUIRE(combOpGraph.GetOps()[2]->m_DebugTag == "OutputDma1");
    REQUIRE(combOpGraph.GetOps()[3]->m_DebugTag == "OutputDma2");
    REQUIRE(combOpGraph.GetOps()[4]->m_DebugTag == "OutputDma3");

    REQUIRE(combOpGraph.GetProducer(combOpGraph.GetBuffers()[0]) == nullptr);
    REQUIRE(combOpGraph.GetProducer(combOpGraph.GetBuffers()[1])->m_DebugTag == "InputDma");
    REQUIRE(combOpGraph.GetProducer(combOpGraph.GetBuffers()[2])->m_DebugTag == "Mce2");
    REQUIRE(combOpGraph.GetProducer(combOpGraph.GetBuffers()[3])->m_DebugTag == "Mce2");
    REQUIRE(combOpGraph.GetProducer(combOpGraph.GetBuffers()[4])->m_DebugTag == "OutputDma1");
    REQUIRE(combOpGraph.GetProducer(combOpGraph.GetBuffers()[5])->m_DebugTag == "OutputDma2");
    REQUIRE(combOpGraph.GetProducer(combOpGraph.GetBuffers()[6])->m_DebugTag == "OutputDma3");

    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[0]).size() == 1);
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[0])[0].first->m_DebugTag == "InputDma");
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[0])[0].second == 0);

    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[1]).size() == 2);
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[1])[0].first->m_DebugTag == "Mce2");
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[1])[0].second == 0);
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[1])[1].first->m_DebugTag == "Mce2");
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[1])[1].second == 1);

    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[2]).size() == 2);
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[2])[0].first->m_DebugTag == "OutputDma1");
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[2])[0].second == 0);
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[2])[1].first->m_DebugTag == "OutputDma2");
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[2])[1].second == 0);

    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[3]).size() == 1);
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[3])[0].first->m_DebugTag == "OutputDma3");
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[3])[0].second == 0);

    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[4]).size() == 0);

    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[5]).size() == 0);

    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[6]).size() == 0);
}

TEST_CASE("Combination operator+", "[CombinerDFS]")
{
    // Create graph:
    //
    //  A - B - C
    //
    GraphOfParts graph;

    auto& parts = graph.m_Parts;

    auto pA               = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB               = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pC               = std::make_unique<MockPart>(graph.GeneratePartId());
    const BasePart& partA = *pA;
    const BasePart& partB = *pB;
    const BasePart& partC = *pC;
    parts.push_back(std::move(pA));
    parts.push_back(std::move(pB));
    parts.push_back(std::move(pC));

    Plan planA;
    Plan planB;
    int planBId = planB.m_DebugId;
    Plan planC;

    Combination combA(partA, std::move(planA), 0, graph);
    Combination combB(partB, std::move(planB), 1, graph);
    Combination combC(partC, std::move(planC), 2, graph);

    REQUIRE(combA.m_Elems.size() == 1);
    REQUIRE(combB.m_Elems.size() == 1);
    REQUIRE(combC.m_Elems.size() == 1);

    Combination comb;
    REQUIRE(comb.m_Elems.size() == 0);

    comb = combA + combB + combC;
    REQUIRE(comb.m_Elems.size() == 3);
    // All parts are in the final combination
    for (size_t i = 0; i < graph.m_Parts.size(); ++i)
    {
        BasePart& part = *graph.m_Parts[i];
        REQUIRE(comb.m_Elems.find(part.GetPartId()) != comb.m_Elems.end());
    }

    // Nothing changes if combA is added again
    comb = comb + combA;
    REQUIRE(comb.m_Elems.size() == 3);

    // There is no glue
    for (size_t i = 0; i < graph.m_Parts.size(); ++i)
    {
        BasePart& part = *graph.m_Parts[i];
        for (auto& glueIt : comb.m_Elems.at(part.GetPartId()).m_Glues)
        {
            REQUIRE(glueIt.second.m_Glue == nullptr);
        }
    }

    // Simple glue between B and C
    Glue glueB_C;
    glueB_C.m_Graph.AddOp(std::make_unique<DmaOp>());
    glueB_C.m_Graph.GetOps()[0]->m_DebugTag = "DmaBC";
    glueB_C.m_InputSlot                     = { glueB_C.m_Graph.GetOps()[0], 0 };
    glueB_C.m_Output.push_back(glueB_C.m_Graph.GetOps()[0]);

    PartInputSlot partCInputSlot = { partC.GetPartId(), 0 };

    Combination combBGlue(partB, &partCInputSlot, &glueB_C, graph);

    comb = comb + combBGlue;
    // Number of elemnts didn't change
    REQUIRE(comb.m_Elems.size() == 3);
    // Glue has been added
    REQUIRE(comb.m_Elems.at(partB.GetPartId()).m_Glues.size() == 1);
    const Glue* glueTest = comb.m_Elems.at(partB.GetPartId()).m_Glues.at(partCInputSlot).m_Glue;
    // It has the correct tag
    REQUIRE(glueTest->m_Graph.GetOps()[0]->m_DebugTag == "DmaBC");
    REQUIRE(comb.m_Elems.at(partB.GetPartId()).m_Plan->m_DebugId == planBId);
}

TEST_CASE("FindBestCombinationForPart cache", "[CombinerDFS]")
{
    // Create graph:
    //
    //  A - B - C
    //
    GraphOfParts graph;

    auto& parts = graph.m_Parts;

    auto pA               = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB               = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pC               = std::make_unique<MockPart>(graph.GeneratePartId());
    const BasePart& partA = *pA;
    const BasePart& partB = *pB;
    const BasePart& partC = *pC;
    parts.push_back(std::move(pA));
    parts.push_back(std::move(pB));
    parts.push_back(std::move(pC));

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(&compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    class MockCombiner : public Combiner
    {
        using Combiner::Combiner;

    public:
        Combination FindBestCombinationForPartImpl(const BasePart&) override
        {
            m_NumFindBestCombinationForPartImplCalled++;
            return Combination{};
        }

        uint64_t m_NumFindBestCombinationForPartImplCalled = 0;
    };

    MockCombiner combiner(graph, hwCaps, estOpt, debuggingContext);

    // Map is empty
    Combination comb = combiner.FindBestCombinationForPart(partA);
    // Map has partA
    REQUIRE(combiner.m_NumFindBestCombinationForPartImplCalled == 1);
    comb = combiner.FindBestCombinationForPart(partA);
    // Map has still only partA
    REQUIRE(combiner.m_NumFindBestCombinationForPartImplCalled == 1);
    comb = combiner.FindBestCombinationForPart(partB);
    // Map has partB
    REQUIRE(combiner.m_NumFindBestCombinationForPartImplCalled == 2);
    comb = combiner.FindBestCombinationForPart(partB);
    // Map has still only partA and partB
    REQUIRE(combiner.m_NumFindBestCombinationForPartImplCalled == 2);
    comb = combiner.FindBestCombinationForPart(partC);
    // Map has partC
    REQUIRE(combiner.m_NumFindBestCombinationForPartImplCalled == 3);
    comb = combiner.FindBestCombinationForPart(partC);
    // Map has still only partA, partB and partC
    REQUIRE(combiner.m_NumFindBestCombinationForPartImplCalled == 3);
}

TEST_CASE("ArePlansCompatible", "[CombinerDFS]")
{
    GraphOfParts gOfParts;
    auto& parts       = gOfParts.m_Parts;
    auto& connections = gOfParts.m_Connections;

    auto pA               = std::make_unique<MockPart>(gOfParts.GeneratePartId());
    auto pB               = std::make_unique<MockPart>(gOfParts.GeneratePartId());
    const BasePart& partA = *pA;
    const BasePart& partB = *pB;
    parts.push_back(std::move(pA));
    parts.push_back(std::move(pB));

    PartOutputSlot planAOutputSlot = { partA.GetPartId(), 0 };
    PartInputSlot planBInputSlot   = { partB.GetPartId(), 0 };
    PartOutputSlot planBOutputSlot = { partB.GetPartId(), 0 };

    connections[planBInputSlot] = planAOutputSlot;

    const EstimationOptions estOpt;
    const CompilationOptions compOpt;
    const DebuggingContext debuggingContext(&compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);

    // Part consisting of node A
    Plan planA;
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planA.m_OpGraph.GetBuffers().back()->m_DebugTag = "InputDram";
    planA.m_OutputMappings                          = { { planA.m_OpGraph.GetBuffers()[0], planAOutputSlot } };

    // Part consisting of node B
    Plan planB;
    planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planB.m_OpGraph.GetBuffers().back()->m_DebugTag = "InputSram1";
    planB.m_InputMappings                           = { { planB.m_OpGraph.GetBuffers()[0], planBInputSlot } };
    planB.m_OutputMappings                          = { { planB.m_OpGraph.GetBuffers()[0], planBOutputSlot } };

    Combiner combiner(gOfParts, hwCaps, estOpt, debuggingContext);

    REQUIRE(combiner.ArePlansCompatible(planA, planB, PartConnection{ planBInputSlot, planAOutputSlot }) == true);
}

TEST_CASE("GluePartToCombination", "[CombinerDFS]")
{
    // Create graph:
    //
    //        B
    //  A     |
    //  |     v
    //  |     1
    //   -->0 D 2<- - C
    //
    GraphOfParts graph;

    auto& parts       = graph.m_Parts;
    auto& connections = graph.m_Connections;

    auto pA               = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB               = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pC               = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pD               = std::make_unique<MockPart>(graph.GeneratePartId());
    const BasePart& partA = *pA;
    const BasePart& partB = *pB;
    const BasePart& partC = *pC;
    const BasePart& partD = *pD;
    parts.push_back(std::move(pA));
    parts.push_back(std::move(pB));
    parts.push_back(std::move(pC));
    parts.push_back(std::move(pD));

    PartOutputSlot partAOutputSlot = { partA.GetPartId(), 0 };
    PartOutputSlot partBOutputSlot = { partB.GetPartId(), 0 };
    PartOutputSlot partCOutputSlot = { partC.GetPartId(), 0 };
    PartInputSlot partDInputSlot0  = { partD.GetPartId(), 0 };
    PartInputSlot partDInputSlot1  = { partD.GetPartId(), 1 };
    PartInputSlot partDInputSlot2  = { partD.GetPartId(), 2 };

    connections[partDInputSlot0] = partAOutputSlot;
    connections[partDInputSlot1] = partBOutputSlot;
    connections[partDInputSlot2] = partCOutputSlot;

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
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 16, 16 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planB.m_OutputMappings = { { planB.m_OpGraph.GetBuffers()[0], partBOutputSlot } };

    Plan planC;
    planC.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 16, 16 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planC.m_OutputMappings = { { planC.m_OpGraph.GetBuffers()[0], partCOutputSlot } };

    Plan planD;
    planD.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 16, 16, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planD.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 16, 48 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planD.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 32, 16, 48 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));

    planD.m_InputMappings = { { planD.m_OpGraph.GetBuffers()[0], partDInputSlot0 },
                              { planD.m_OpGraph.GetBuffers()[1], partDInputSlot1 },
                              { planD.m_OpGraph.GetBuffers()[2], partDInputSlot2 } };

    Combination combA(partA, std::move(planA), 0, graph);
    Combination combB(partB, std::move(planB), 1, graph);
    Combination combC(partC, std::move(planC), 2, graph);
    Combination combD(partD, std::move(planD), 3, graph);

    // Merge the combinations
    Combination comb = combA + combB + combC + combD;

    // There is no glue
    for (PartId i = 0; i < graph.m_Parts.size(); ++i)
    {
        const BasePart& part = graph.GetPart(i);
        for (auto& glueIt : comb.m_Elems.at(part.GetPartId()).m_Glues)
        {
            REQUIRE(glueIt.second.m_Glue == nullptr);
        }
    }

    Combiner combiner(graph, hwCaps, estOpt, debuggingContext);

    const auto& sources = graph.GetSourceConnections(partD.GetPartId());

    Combination combGlued = combiner.GluePartToCombinationDestToSrcs(partD, comb, sources);

    REQUIRE(combGlued.m_Elems.size() == 4);
    // There is a glue for each input part
    for (auto elem : combGlued.m_Elems)
    {
        if (elem.first == partD.GetPartId())
        {
            continue;
        }
        CHECK(!elem.second.m_Glues.empty());
    }

    // A and B have glue and the buffer in Dram is in the expected format
    auto elemIt = combGlued.m_Elems.find(partA.GetPartId());
    REQUIRE(elemIt != combGlued.m_Elems.end());
    REQUIRE(elemIt->second.m_Glues.begin()->second.m_Glue->m_Graph.GetBuffers().at(0)->m_Location == Location::Dram);
    REQUIRE(elemIt->second.m_Glues.begin()->second.m_Glue->m_Graph.GetBuffers().at(0)->m_Format ==
            CascadingBufferFormat::FCAF_DEEP);
    elemIt = combGlued.m_Elems.find(partB.GetPartId());
    REQUIRE(elemIt != combGlued.m_Elems.end());
    REQUIRE(elemIt->second.m_Glues.begin()->second.m_Glue->m_Graph.GetBuffers().at(0)->m_Location == Location::Dram);
    REQUIRE(elemIt->second.m_Glues.begin()->second.m_Glue->m_Graph.GetBuffers().at(0)->m_Format ==
            CascadingBufferFormat::FCAF_WIDE);
}

TEST_CASE("CombinerSortTest1", "[CombinerDFS]")
{
    // Create graph:
    //
    //
    //   A- -> B -> C -- > D
    //     \    \           \         |
    //      \    \           H---->I
    //       \    \        / |____ K
    //        E -> F ---> G
    //             ^
    //             |
    //             J

    GraphOfParts graph;
    auto& parts       = graph.m_Parts;
    auto& connections = graph.m_Connections;

    auto pA = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pC = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pD = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pE = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pF = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pG = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pH = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pI = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pJ = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pK = std::make_unique<MockPart>(graph.GeneratePartId());

    BasePart& partA = *pA;
    BasePart& partB = *pB;
    BasePart& partC = *pC;
    BasePart& partD = *pD;
    BasePart& partE = *pE;
    BasePart& partF = *pF;
    BasePart& partG = *pG;
    BasePart& partH = *pH;
    BasePart& partI = *pI;
    BasePart& partJ = *pJ;
    BasePart& partK = *pK;

    parts.push_back(std::move(pA));
    parts.push_back(std::move(pB));
    parts.push_back(std::move(pC));
    parts.push_back(std::move(pD));
    parts.push_back(std::move(pE));
    parts.push_back(std::move(pF));
    parts.push_back(std::move(pG));
    parts.push_back(std::move(pH));
    parts.push_back(std::move(pI));
    parts.push_back(std::move(pJ));
    parts.push_back(std::move(pK));

    PartOutputSlot partAOutputSlot0 = { partA.GetPartId(), 0 };
    PartOutputSlot partAOutputSlot1 = { partA.GetPartId(), 1 };
    PartOutputSlot partBOutputSlot0 = { partB.GetPartId(), 0 };
    PartOutputSlot partCOutputSlot  = { partC.GetPartId(), 0 };
    PartOutputSlot partDOutputSlot  = { partD.GetPartId(), 0 };
    PartOutputSlot partEOutputSlot  = { partE.GetPartId(), 0 };
    PartOutputSlot partFOutputSlot  = { partF.GetPartId(), 0 };
    PartOutputSlot partGOutputSlot  = { partG.GetPartId(), 0 };
    PartOutputSlot partHOutputSlot0 = { partH.GetPartId(), 0 };
    PartOutputSlot partHOutputSlot1 = { partH.GetPartId(), 1 };
    PartOutputSlot partJOutputSlot  = { partJ.GetPartId(), 0 };

    PartInputSlot partBInputSlot  = { partB.GetPartId(), 0 };
    PartInputSlot partCInputSlot  = { partC.GetPartId(), 0 };
    PartInputSlot partDInputSlot  = { partD.GetPartId(), 0 };
    PartInputSlot partEInputSlot  = { partE.GetPartId(), 0 };
    PartInputSlot partFInputSlot0 = { partF.GetPartId(), 0 };
    PartInputSlot partFInputSlot1 = { partF.GetPartId(), 1 };
    PartInputSlot partFInputSlot2 = { partF.GetPartId(), 2 };
    PartInputSlot partGInputSlot  = { partG.GetPartId(), 0 };
    PartInputSlot partHInputSlot0 = { partH.GetPartId(), 0 };
    PartInputSlot partHInputSlot1 = { partH.GetPartId(), 1 };
    PartInputSlot partIInputSlot  = { partI.GetPartId(), 0 };
    PartInputSlot partKInputSlot  = { partK.GetPartId(), 0 };

    connections[partBInputSlot]  = { partAOutputSlot0 };
    connections[partCInputSlot]  = { partBOutputSlot0 };
    connections[partDInputSlot]  = { partCOutputSlot };
    connections[partFInputSlot0] = { partBOutputSlot0 };
    connections[partEInputSlot]  = { partAOutputSlot1 };
    connections[partFInputSlot1] = { partEOutputSlot };
    connections[partGInputSlot]  = { partFOutputSlot };
    connections[partFInputSlot2] = { partJOutputSlot };
    connections[partHInputSlot0] = { partDOutputSlot };
    connections[partHInputSlot1] = { partGOutputSlot };
    connections[partIInputSlot]  = { partHOutputSlot0 };
    connections[partKInputSlot]  = { partHOutputSlot1 };

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(&compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Combiner combiner(graph, hwCaps, estOpt, debuggingContext);

    bool isSorted = combiner.TopologicalSortParts();

    // After sorting , the expected is A, B, C, D, E, J, F, G, H, I, K
    REQUIRE(isSorted == true);
    REQUIRE(combiner.GetNextPart(&partA) == &partB);
    REQUIRE(combiner.GetNextPart(&partB) == &partC);
    REQUIRE(combiner.GetNextPart(&partC) == &partD);
    REQUIRE(combiner.GetNextPart(&partD) == &partE);
    REQUIRE(combiner.GetNextPart(&partE) == &partJ);
    REQUIRE(combiner.GetNextPart(&partJ) == &partF);
    REQUIRE(combiner.GetNextPart(&partF) == &partG);
    REQUIRE(combiner.GetNextPart(&partG) == &partH);
    REQUIRE(combiner.GetNextPart(&partH) == &partI);
    REQUIRE(combiner.GetNextPart(&partI) == &partK);
    REQUIRE(combiner.GetNextPart(&partK) == nullptr);
}

TEST_CASE("CombinerSortTest2", "[CombinerDFS]")
{
    // Create graph:
    //
    //
    //   A- -> B - -> C ---
    //                     |
    //                     G
    //                     |
    //   D- -> E - -> F ---
    GraphOfParts graph;
    auto& parts       = graph.m_Parts;
    auto& connections = graph.m_Connections;

    auto pA = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pC = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pD = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pE = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pF = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pG = std::make_unique<MockPart>(graph.GeneratePartId());

    BasePart& partA = *pA;
    BasePart& partB = *pB;
    BasePart& partC = *pC;
    BasePart& partD = *pD;
    BasePart& partE = *pE;
    BasePart& partF = *pF;
    BasePart& partG = *pG;

    parts.push_back(std::move(pA));
    parts.push_back(std::move(pB));
    parts.push_back(std::move(pC));
    parts.push_back(std::move(pD));
    parts.push_back(std::move(pE));
    parts.push_back(std::move(pF));
    parts.push_back(std::move(pG));

    PartOutputSlot partAOutputSlot = { partA.GetPartId(), 0 };
    PartOutputSlot partBOutputSlot = { partB.GetPartId(), 0 };
    PartOutputSlot partCOutputSlot = { partC.GetPartId(), 0 };
    PartOutputSlot partDOutputSlot = { partD.GetPartId(), 0 };
    PartOutputSlot partEOutputSlot = { partE.GetPartId(), 0 };
    PartOutputSlot partFOutputSlot = { partF.GetPartId(), 0 };

    PartInputSlot partBInputSlot  = { partB.GetPartId(), 0 };
    PartInputSlot partCInputSlot  = { partC.GetPartId(), 0 };
    PartInputSlot partEInputSlot  = { partE.GetPartId(), 0 };
    PartInputSlot partFInputSlot  = { partF.GetPartId(), 0 };
    PartInputSlot partGInputSlot0 = { partG.GetPartId(), 0 };
    PartInputSlot partGInputSlot1 = { partG.GetPartId(), 1 };

    connections[partBInputSlot]  = { partAOutputSlot };
    connections[partCInputSlot]  = { partBOutputSlot };
    connections[partEInputSlot]  = { partDOutputSlot };
    connections[partFInputSlot]  = { partEOutputSlot };
    connections[partGInputSlot0] = { partCOutputSlot };
    connections[partGInputSlot1] = { partFOutputSlot };

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(&compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Combiner combiner(graph, hwCaps, estOpt, debuggingContext);

    bool isSorted = combiner.TopologicalSortParts();

    // After sorting , the expected is A, B, C, D, E, F, G
    REQUIRE(isSorted == true);
    REQUIRE(combiner.GetNextPart(&partA) == &partB);
    REQUIRE(combiner.GetNextPart(&partB) == &partC);
    REQUIRE(combiner.GetNextPart(&partC) == &partD);
    REQUIRE(combiner.GetNextPart(&partD) == &partE);
    REQUIRE(combiner.GetNextPart(&partE) == &partF);
    REQUIRE(combiner.GetNextPart(&partF) == &partG);
    REQUIRE(combiner.GetNextPart(&partG) == nullptr);
}

TEST_CASE("GetCombPartsInOrder", "[CombinerDFS]")
{
    // Create graph:
    //
    //   A -> B -> C -> D -> E
    //
    //
    GraphOfParts graph;
    auto& parts       = graph.m_Parts;
    auto& connections = graph.m_Connections;

    auto pA = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pC = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pD = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pE = std::make_unique<MockPart>(graph.GeneratePartId());

    BasePart& partA = *pA;
    BasePart& partB = *pB;
    BasePart& partC = *pC;
    BasePart& partD = *pD;
    BasePart& partE = *pE;

    parts.push_back(std::move(pA));
    parts.push_back(std::move(pB));
    parts.push_back(std::move(pC));
    parts.push_back(std::move(pD));
    parts.push_back(std::move(pE));

    PartOutputSlot partAOutputSlot = { partA.GetPartId(), 0 };
    PartOutputSlot partBOutputSlot = { partB.GetPartId(), 0 };
    PartOutputSlot partCOutputSlot = { partC.GetPartId(), 0 };
    PartOutputSlot partDOutputSlot = { partD.GetPartId(), 0 };

    PartInputSlot partBInputSlot = { partB.GetPartId(), 0 };
    PartInputSlot partCInputSlot = { partC.GetPartId(), 0 };
    PartInputSlot partDInputSlot = { partD.GetPartId(), 0 };
    PartInputSlot partEInputSlot = { partE.GetPartId(), 0 };

    connections[partBInputSlot] = { partAOutputSlot };
    connections[partCInputSlot] = { partBOutputSlot };
    connections[partDInputSlot] = { partCOutputSlot };
    connections[partEInputSlot] = { partDOutputSlot };

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(&compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    GraphOfParts gOfParts;

    Plan planA;
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planA.m_OutputMappings = { { planA.m_OpGraph.GetBuffers()[0], partAOutputSlot } };

    Plan planB;
    planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 16, 16 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planB.m_InputMappings  = { { planB.m_OpGraph.GetBuffers()[0], partBInputSlot } };
    planB.m_OutputMappings = { { planB.m_OpGraph.GetBuffers()[0], partBOutputSlot } };

    Plan planC;
    planC.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 16, 16 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planC.m_InputMappings  = { { planC.m_OpGraph.GetBuffers()[0], partCInputSlot } };
    planC.m_OutputMappings = { { planC.m_OpGraph.GetBuffers()[0], partCOutputSlot } };

    Plan planD;
    planD.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 16, 16, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planD.m_InputMappings  = { { planD.m_OpGraph.GetBuffers()[0], partDInputSlot } };
    planD.m_OutputMappings = { { planD.m_OpGraph.GetBuffers()[0], partDOutputSlot } };

    Plan planE;
    planE.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 16, 16, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planE.m_InputMappings = { { planD.m_OpGraph.GetBuffers()[0], partEInputSlot } };

    Combiner combiner(gOfParts, hwCaps, estOpt, debuggingContext);

    bool isSorted = combiner.TopologicalSortParts();
    REQUIRE(isSorted == true);

    Combination combA(partA, std::move(planA), 0, graph);
    Combination combB(partB, std::move(planB), 1, graph);
    Combination combC(partC, std::move(planC), 2, graph);
    Combination combD(partD, std::move(planD), 3, graph);
    Combination combE(partE, std::move(planE), 4, graph);

    Combination comb = combD + combE;
    {
        REQUIRE(comb.m_HeadOrderRank == 3);
        std::vector<PartId> expectedList = { 3, 4 };
        REQUIRE(comb.m_PartIdsInOrder == expectedList);
    }

    {
        // Adding combinations is commutative.
        comb = combE + combD;
        REQUIRE(comb.m_HeadOrderRank == 3);
        std::vector<PartId> expectedList = { 3, 4 };
        REQUIRE(comb.m_PartIdsInOrder == expectedList);
    }

    {
        comb = combC + comb;
        REQUIRE(comb.m_HeadOrderRank == 2);
        std::vector<PartId> expectedList = { 2, 3, 4 };
        REQUIRE(comb.m_PartIdsInOrder == expectedList);
    }

    {
        comb = combB + comb;
        REQUIRE(comb.m_HeadOrderRank == 1);
        std::vector<PartId> expectedList = { 1, 2, 3, 4 };
        REQUIRE(comb.m_PartIdsInOrder == expectedList);
    }

    {
        comb = combA + comb;
        REQUIRE(comb.m_HeadOrderRank == 0);
        std::vector<PartId> expectedList = { 0, 1, 2, 3, 4 };
        REQUIRE(comb.m_PartIdsInOrder == expectedList);
    }
}

TEST_CASE("GluePartToCombinationBranch0", "[CombinerDFS]")
{
    // Create graph:
    //
    //
    //   - - > C
    //  |
    //  A - -> B
    //

    GraphOfParts graph;
    auto& parts       = graph.m_Parts;
    auto& connections = graph.m_Connections;

    auto pA = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pC = std::make_unique<MockPart>(graph.GeneratePartId());

    BasePart& partA = *pA;
    BasePart& partB = *pB;
    BasePart& partC = *pC;

    PartId partAId = pA->GetPartId();
    PartId partBId = pB->GetPartId();
    PartId partCId = pC->GetPartId();

    parts.push_back(std::move(pA));
    parts.push_back(std::move(pB));
    parts.push_back(std::move(pC));

    PartOutputSlot partAOutputSlot = { partA.GetPartId(), 0 };

    PartInputSlot partBInputSlot = { partB.GetPartId(), 0 };
    PartInputSlot partCInputSlot = { partC.GetPartId(), 0 };

    connections[partBInputSlot] = { partAOutputSlot };
    connections[partCInputSlot] = { partAOutputSlot };

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

    Combination combA(partA, std::move(planA), 0, graph);
    Combination combB(partB, std::move(planB), 1, graph);
    Combination combC(partC, std::move(planC), 2, graph);

    // Merge the combinations
    Combination comb = combA + combB + combC;

    REQUIRE(combA.m_PartIdsInOrder[0] == 0);
    REQUIRE(combA.m_HeadOrderRank == 0);
    REQUIRE(combB.m_PartIdsInOrder[0] == 1);
    REQUIRE(combB.m_HeadOrderRank == 1);
    REQUIRE(combC.m_PartIdsInOrder[0] == 2);
    REQUIRE(combC.m_HeadOrderRank == 2);
    REQUIRE(comb.m_PartIdsInOrder[0] == 0);
    REQUIRE(comb.m_HeadOrderRank == 0);

    // There is no glue
    for (PartId i = 0; i < graph.m_Parts.size(); ++i)
    {
        const BasePart& part = graph.GetPart(i);
        for (auto& glueIt : comb.m_Elems.at(part.GetPartId()).m_Glues)
        {
            REQUIRE(glueIt.second.m_Glue == nullptr);
        }
    }

    Combiner combiner(graph, hwCaps, estOpt, debuggingContext);

    std::vector<PartConnection> destPartEdge;

    // Part B and the edge that connects to its source Part A
    PartConnection edgeA2B = graph.GetConnectionsBetween(partAId, partBId).at(0);
    destPartEdge.push_back(edgeA2B);
    // Part C and the edge that connects to its source Part A
    PartConnection edgeA2C = graph.GetConnectionsBetween(partAId, partCId).at(0);
    destPartEdge.push_back(edgeA2C);

    Combination combGlued = combiner.GluePartToCombinationSrcToDests(partA, comb, destPartEdge);

    REQUIRE(combGlued.m_PartIdsInOrder[0] == 0);
    REQUIRE(combGlued.m_HeadOrderRank == 0);

    // One glue shared by A-B, A-C
    // The glue has (1) 1 x input DMA (2) DRAM buffer (3) 2 x ouput DMA

    REQUIRE(combGlued.m_Elems.size() == 3);

    // Elem Part A's glue should have two elements
    // (*edgeAB, *glue) (*edgeAC, *glue)
    auto elemIt = combGlued.m_Elems.find(partAId);
    REQUIRE(elemIt != combGlued.m_Elems.end());
    REQUIRE(elemIt->second.m_Glues.size() == 2);

    auto elemAB = elemIt->second.m_Glues.find(edgeA2B.m_Destination);
    REQUIRE(elemAB != elemIt->second.m_Glues.end());
    auto elemAC = elemIt->second.m_Glues.find(edgeA2C.m_Destination);
    REQUIRE(elemAC != elemIt->second.m_Glues.end());
    REQUIRE(elemAB->second.m_Glue == elemAC->second.m_Glue);
    REQUIRE(elemAB->second.m_Glue->m_Graph.GetBuffers().at(0)->m_Location == Location::Dram);
    REQUIRE(elemAB->second.m_Glue->m_Graph.GetBuffers().at(0)->m_Format == CascadingBufferFormat::FCAF_DEEP);
    REQUIRE(elemAB->second.m_Glue->m_Graph.GetOps().size() == 3);
}

TEST_CASE("GluePartToCombinationBranch1", "[CombinerDFS]")
{
    // Create graph:
    //
    //
    //   - - > C
    //  |
    //  A - -> B
    //  |
    //   -- >  D
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

    PartId partAId = pA->GetPartId();
    PartId partBId = pB->GetPartId();
    PartId partCId = pC->GetPartId();
    PartId partDId = pD->GetPartId();

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

    // There is no glue
    for (PartId i = 0; i < graph.m_Parts.size(); ++i)
    {
        const BasePart& part = graph.GetPart(i);
        for (auto& glueIt : comb.m_Elems.at(part.GetPartId()).m_Glues)
        {
            REQUIRE(glueIt.second.m_Glue == nullptr);
        }
    }

    Combiner combiner(graph, hwCaps, estOpt, debuggingContext);

    std::vector<PartConnection> destPartEdge;

    // Part B and the edge that connects to its source Part A
    PartConnection edgeA2B = graph.GetConnectionsBetween(partAId, partBId).at(0);
    destPartEdge.push_back(edgeA2B);
    // Part C and the edge that connects to its source Part A
    PartConnection edgeA2C = graph.GetConnectionsBetween(partAId, partCId).at(0);
    destPartEdge.push_back(edgeA2C);
    // Part D and the edge that connects to its source Part A
    PartConnection edgeA2D = graph.GetConnectionsBetween(partAId, partDId).at(0);
    destPartEdge.push_back(edgeA2D);

    Combination combGlued = combiner.GluePartToCombinationSrcToDests(partA, comb, destPartEdge);

    // One glue shared by A-B, A-C (SRAM - SRAM) and A-D (SRAM - DRAM)
    // The glue has (1) 1 x input DMA (2) DRAM buffer (3) 2 x ouput DMA
    REQUIRE(combGlued.m_Elems.size() == 4);

    // Elem Part A's glue should have three elements
    // (*edgeAB, *glue0) (*edgeAC, *glue0) (*edgeAD, *glue1)
    auto elemIt = combGlued.m_Elems.find(partAId);
    REQUIRE(elemIt != combGlued.m_Elems.end());
    REQUIRE(elemIt->second.m_Glues.size() == 3);

    auto elemAB = elemIt->second.m_Glues.find(edgeA2B.m_Destination);
    REQUIRE(elemAB != elemIt->second.m_Glues.end());
    auto elemAC = elemIt->second.m_Glues.find(edgeA2C.m_Destination);
    REQUIRE(elemAC != elemIt->second.m_Glues.end());
    auto elemAD = elemIt->second.m_Glues.find(edgeA2D.m_Destination);
    REQUIRE(elemAD != elemIt->second.m_Glues.end());

    REQUIRE(elemAB->second.m_Glue == elemAC->second.m_Glue);
    REQUIRE(elemAB->second.m_Glue == elemAD->second.m_Glue);
    REQUIRE(elemAB->second.m_OutDma == true);
    REQUIRE(elemAC->second.m_OutDma == true);
    REQUIRE(elemAD->second.m_OutDma == false);
    REQUIRE(elemAB->second.m_Glue->m_Graph.GetBuffers().at(0)->m_Location == Location::Dram);
    REQUIRE(elemAB->second.m_Glue->m_Graph.GetBuffers().at(0)->m_Format == CascadingBufferFormat::NHWCB);
    REQUIRE(elemAB->second.m_Glue->m_Graph.GetOps().size() == 3);
}

TEST_CASE("IsPlanInputGlueable", "[CombinerDFS]")
{
    GraphOfParts gOfParts;
    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(&compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Plan planA;
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 16, 16, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(
        Lifetime::Atomic, Location::VirtualSram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 64, 64, 64 },
        TensorShape{ 1, 8, 16, 48 }, TraversalOrder::Xyz, 4, QuantizationInfo()));
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 32, 16, 48 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));

    planA.m_InputMappings = { { planA.m_OpGraph.GetBuffers()[0], {} },
                              { planA.m_OpGraph.GetBuffers()[1], {} },
                              { planA.m_OpGraph.GetBuffers()[2], {} } };

    Combiner combiner(gOfParts, hwCaps, estOpt, debuggingContext);

    REQUIRE(combiner.IsPlanInputGlueable(planA) == false);

    Plan planB;
    planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 16, 16, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 16, 48 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 32, 16, 48 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));

    planB.m_InputMappings = { { planB.m_OpGraph.GetBuffers()[0], {} },
                              { planB.m_OpGraph.GetBuffers()[1], {} },
                              { planB.m_OpGraph.GetBuffers()[2], {} } };

    REQUIRE(combiner.IsPlanInputGlueable(planB) == true);
}

TEST_CASE("ArePlansAllowedToMerge", "[CombinerDFS]")
{
    // Create graph:
    //
    //  C --> A - - > B
    //
    GraphOfParts graph;

    auto& parts       = graph.m_Parts;
    auto& connections = graph.m_Connections;

    auto pA               = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB               = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pC               = std::make_unique<MockPart>(graph.GeneratePartId());
    const BasePart& partA = *pA;
    const BasePart& partB = *pB;
    const BasePart& partC = *pC;
    parts.push_back(std::move(pA));
    parts.push_back(std::move(pB));
    parts.push_back(std::move(pC));

    PartOutputSlot partAOutputSlot = { partA.GetPartId(), 0 };
    PartOutputSlot partCOutputSlot = { partC.GetPartId(), 0 };
    PartInputSlot partAInputSlot   = { partA.GetPartId(), 0 };
    PartInputSlot partBInputSlot   = { partB.GetPartId(), 0 };

    connections[partAInputSlot] = partCOutputSlot;
    connections[partBInputSlot] = partAOutputSlot;

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(&compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Plan planA;
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));

    planA.m_OpGraph.AddOp(
        std::make_unique<MceOp>(Lifetime::Atomic, MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
                                BlockConfig{ 16u, 16u }, TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 64, 64, 64 },
                                TensorShape{ 1, 1, 1, 64 }, TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));

    planA.m_OpGraph.SetProducer(planA.m_OpGraph.GetBuffers()[1], planA.m_OpGraph.GetOps()[0]);
    planA.m_InputMappings  = { { planA.m_OpGraph.GetBuffers()[0], partAInputSlot } };
    planA.m_OutputMappings = { { planA.m_OpGraph.GetBuffers()[1], partAOutputSlot } };

    Plan planB;
    planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 16, 16 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planB.m_OpGraph.AddOp(
        std::make_unique<MceOp>(Lifetime::Atomic, MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
                                BlockConfig{ 16u, 16u }, TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 64, 64, 64 },
                                TensorShape{ 1, 1, 1, 64 }, TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));

    planB.m_OpGraph.AddOp(
        std::make_unique<MceOp>(Lifetime::Atomic, MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
                                BlockConfig{ 16u, 16u }, TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 64, 64, 64 },
                                TensorShape{ 1, 1, 1, 64 }, TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));

    planB.m_OpGraph.AddConsumer(planB.m_OpGraph.GetBuffers()[0], planB.m_OpGraph.GetOps()[0], 0);
    planB.m_OpGraph.AddConsumer(planB.m_OpGraph.GetBuffers()[0], planB.m_OpGraph.GetOps()[1], 0);
    planB.m_InputMappings = { { planB.m_OpGraph.GetBuffers()[0], partBInputSlot } };

    Combiner combiner(graph, hwCaps, estOpt, debuggingContext);

    REQUIRE(combiner.ArePlansAllowedToMerge(planA, planB, PartConnection{ partBInputSlot, partAOutputSlot }) == true);

    // Create a new plan with a different Block Config i.e. 8x32
    Plan planBdiffBlockConfig;
    planBdiffBlockConfig.m_OpGraph.AddBuffer(std::make_unique<Buffer>(
        Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 64, 64, 64 },
        TensorShape{ 1, 8, 16, 16 }, TraversalOrder::Xyz, 4, QuantizationInfo()));
    planBdiffBlockConfig.m_OpGraph.AddOp(
        std::make_unique<MceOp>(Lifetime::Atomic, MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
                                BlockConfig{ 16u, 16u }, TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 64, 64, 64 },
                                TensorShape{ 1, 1, 1, 64 }, TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));

    planBdiffBlockConfig.m_OpGraph.AddOp(
        std::make_unique<MceOp>(Lifetime::Atomic, MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
                                BlockConfig{ 8u, 32u }, TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 64, 64, 64 },
                                TensorShape{ 1, 1, 1, 64 }, TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));

    planBdiffBlockConfig.m_OpGraph.AddConsumer(planBdiffBlockConfig.m_OpGraph.GetBuffers()[0],
                                               planBdiffBlockConfig.m_OpGraph.GetOps()[0], 0);
    planBdiffBlockConfig.m_OpGraph.AddConsumer(planBdiffBlockConfig.m_OpGraph.GetBuffers()[0],
                                               planBdiffBlockConfig.m_OpGraph.GetOps()[1], 0);
    planBdiffBlockConfig.m_InputMappings = { { planBdiffBlockConfig.m_OpGraph.GetBuffers()[0], partBInputSlot } };

    // They cannot be merged
    REQUIRE(combiner.ArePlansAllowedToMerge(planA, planBdiffBlockConfig,
                                            PartConnection{ partBInputSlot, partAOutputSlot }) == false);

    // Create a new plan with a different streaming strategy
    Plan planBdiffStrategy;
    planBdiffStrategy.m_OpGraph.AddBuffer(std::make_unique<Buffer>(
        Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 64, 64, 64 },
        TensorShape{ 1, 8, 16, 64 }, TraversalOrder::Xyz, 4, QuantizationInfo()));

    planBdiffStrategy.m_InputMappings = { { planBdiffStrategy.m_OpGraph.GetBuffers()[0], partBInputSlot } };

    // Consumer plan is streaming full depth while producer plan is not
    REQUIRE(combiner.ArePlansAllowedToMerge(planA, planBdiffStrategy,
                                            PartConnection{ partBInputSlot, partAOutputSlot }) == true);
    REQUIRE(combiner.ArePlansStreamingStrategiesCompatible(planA, planBdiffStrategy,
                                                           PartConnection{ partBInputSlot, partAOutputSlot }) == false);
}

TEST_CASE("IsPlanAllocated", "[CombinerDFS]")
{
    GraphOfParts graph;

    auto& parts = graph.m_Parts;

    auto pA = std::make_unique<MockPart>(graph.GeneratePartId());

    const BasePart& partA = *pA;
    parts.push_back(std::move(pA));

    PartOutputSlot partAOutputSlot = { partA.GetPartId(), 0 };
    PartInputSlot partAInputSlot   = { partA.GetPartId(), 0 };

    const uint32_t ifmSize = 524288;
    const uint32_t ofmSize = 65536;

    Plan planA;
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Cascade, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 32, 16, 1024 }, TensorShape{ 1, 32, 16, 1024 },
                                                       TraversalOrder::Xyz, ifmSize, QuantizationInfo()));
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 32, 16, 1024 }, TensorShape{ 1, 4, 16, 1024 },
                                                       TraversalOrder::Xyz, ofmSize, QuantizationInfo()));

    planA.m_OpGraph.AddOp(
        std::make_unique<MceOp>(Lifetime::Cascade, MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
                                BlockConfig{ 8u, 8u }, TensorShape{ 1, 32, 16, 1024 }, TensorShape{ 1, 32, 16, 1024 },
                                TensorShape{ 1, 32, 16, 1024 }, TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));
    planA.m_OpGraph.SetProducer(planA.m_OpGraph.GetBuffers()[1], planA.m_OpGraph.GetOps()[0]);
    planA.m_InputMappings  = { { planA.m_OpGraph.GetBuffers()[0], partAInputSlot } };
    planA.m_OutputMappings = { { planA.m_OpGraph.GetBuffers()[1], partAOutputSlot } };

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(&compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    ethosn::support_library::impl::NumMemoryStripes numMemoryStripes;
    const std::set<uint32_t> operationIds = { 0 };

    Combiner combiner(graph, hwCaps, estOpt, debuggingContext);

    SramAllocator alloc(hwCaps.GetTotalSramSize() / hwCaps.GetNumberOfSrams());
    PleOperations pleOps = {};

    // SRAM has space for ofmSize
    alloc.Allocate(0, (hwCaps.GetTotalSramSize() - ofmSize) / hwCaps.GetNumberOfSrams(), AllocationPreference::Start);

    // SRAM has enough space for ofm and the plan does not have a PLE kernel
    SramAllocator alloc1 = alloc;
    REQUIRE(combiner.IsPlanAllocated(alloc1, planA, pleOps) == true);
    REQUIRE(pleOps.size() == 0);

    // Adding a passthrough PLE kernel to the plan
    // The PleKernelId is expected to be PASSTHROUGH_8x8_2
    auto op =
        std::make_unique<PleOp>(Lifetime::Cascade, ethosn::command_stream::PleOperation::PASSTHROUGH,
                                BlockConfig{ 8u, 8u }, 1, std::vector<TensorShape>{ TensorShape{ 1, 32, 16, 1024 } },
                                TensorShape{ 1, 32, 16, 1024 }, ethosn::command_stream::DataType::U8);

    numMemoryStripes.m_Output = 0;
    auto outBufferAndPleOp    = AddPleToOpGraph(planA.m_OpGraph, Lifetime::Cascade, TraversalOrder::Xyz,
                                             TensorShape{ 1, 32, 16, 1024 }, numMemoryStripes, std::move(op),
                                             TensorShape{ 1, 32, 16, 1024 }, QuantizationInfo(), operationIds);

    // With a PLE kernel, the plan can no longer be fit into SRAM
    SramAllocator alloc2 = alloc;
    REQUIRE(combiner.IsPlanAllocated(alloc2, planA, pleOps) == false);
    REQUIRE(pleOps.size() == 0);

    SramAllocator alloc3 = alloc;

    // PLE kernel used previously has different block height
    // The plan is NOT expected to be fit into SRAM
    PleKernelId pleKernel1 = PleKernelId::PASSTHROUGH_8X16_1;
    PleOperations pleOps1  = { pleKernel1 };
    REQUIRE(combiner.IsPlanAllocated(alloc3, planA, pleOps1) == false);
    REQUIRE(pleOps1.size() == 1);

    // PLE kernel passthrough is already used previously in the same
    // section, then the plan is expected to be fit into SRAM
    PleKernelId pleKernel2 = PleKernelId::PASSTHROUGH_8X8_2;
    PleOperations pleOps2  = { pleKernel2 };
    REQUIRE(combiner.IsPlanAllocated(alloc3, planA, pleOps2) == true);
    REQUIRE(pleOps2.size() == 1);

    // SRAM has space for ofm and ple kernel
    SramAllocator alloc4(hwCaps.GetTotalSramSize() / hwCaps.GetNumberOfSrams());
    alloc4.Allocate(0, (hwCaps.GetTotalSramSize() - ofmSize - hwCaps.GetMaxPleSize()) / hwCaps.GetNumberOfSrams(),
                    AllocationPreference::Start);
    REQUIRE(pleOps.size() == 0);
    REQUIRE(combiner.IsPlanAllocated(alloc4, planA, pleOps) == true);
    REQUIRE(pleOps.size() == 1);
    REQUIRE(pleOps.at(0) == PleKernelId::PASSTHROUGH_8X8_2);

    ETHOSN_UNUSED(outBufferAndPleOp);
}
