// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/DebuggingContext.hpp"
#include "../src/GraphNodes.hpp"
#include "../src/SramAllocator.hpp"
#include "../src/cascading/Cascading.hpp"
#include "../src/cascading/CombinerDFS.hpp"
#include "../src/cascading/ConcatPart.hpp"
#include "../src/cascading/McePart.hpp"
#include "../src/cascading/StripeHelper.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>

#include <fstream>

using namespace ethosn::support_library;
using PleKernelId  = ethosn::command_stream::cascading::PleKernelId;
using BlockConfig  = ethosn::command_stream::BlockConfig;
using MceOperation = ethosn::command_stream::MceOperation;
using PleOperation = ethosn::command_stream::PleOperation;

// These Mock classes are used locally to create a test framework for double-buffering logic.
class WeightPart : public MockPart
{
public:
    WeightPart(PartId id,
               uint32_t* numPlansCounter,
               std::vector<uint32_t>* numWeightBuffers,
               std::function<bool(CascadeType, PartId)> filter,
               bool hasInput,
               bool hasOutput)
        : MockPart(id, hasInput, hasOutput)
        , m_NumPlansCounter(numPlansCounter)
        , m_NumWeightBuffers(numWeightBuffers)
        , m_Filter(filter)
    {}

    virtual Plans GetPlans(CascadeType cascadeType,
                           ethosn::command_stream::BlockConfig,
                           Buffer*,
                           uint32_t numWeightBuffers) const override
    {
        if (!m_Filter(cascadeType, m_PartId))
        {
            return Plans();
        }
        Plans plans;

        PartInputMapping inputMappings;
        PartOutputMapping outputMappings;

        OwnedOpGraph opGraph;

        if (m_HasInput)
        {
            opGraph.AddBuffer(
                std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB, TraversalOrder::Xyz));
            Buffer* buffer             = opGraph.GetBuffers().back();
            buffer->m_TensorShape      = { 1, 16, 16, 16 };
            buffer->m_StripeShape      = { 1, 16, 16, 16 };
            buffer->m_SizeInBytes      = 16 * 16 * 16;
            buffer->m_QuantizationInfo = { 0, 1.f };

            inputMappings[buffer] = PartInputSlot{ m_PartId, 0 };
        }

        if (m_HasOutput)
        {
            opGraph.AddBuffer(
                std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB, TraversalOrder::Xyz));
            Buffer* buffer             = opGraph.GetBuffers().back();
            buffer->m_TensorShape      = { 1, 16, 16, 16 };
            buffer->m_StripeShape      = { 1, 16, 16, 16 };
            buffer->m_SizeInBytes      = 16 * 16 * 16;
            buffer->m_QuantizationInfo = { 0, 1.f };

            outputMappings[buffer] = PartOutputSlot{ m_PartId, 0 };
        }

        if (m_HasInput && m_HasOutput)
        {
            opGraph.AddOp(std::make_unique<PleOp>(ethosn::command_stream::PleOperation::PASSTHROUGH,
                                                  BlockConfig{ 8u, 8u }, 1,
                                                  std::vector<TensorShape>{ TensorShape{ 1, 16, 16, 16 } },
                                                  TensorShape{ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, true));

            opGraph.AddConsumer(opGraph.GetBuffers().front(), opGraph.GetOps()[0], 0);
            opGraph.SetProducer(opGraph.GetBuffers().back(), opGraph.GetOps()[0]);
        }

        Plan plan(std::move(inputMappings), std::move(outputMappings));
        plan.m_OpGraph = std::move(opGraph);
        plans.push_back(std::move(plan));

        (*m_NumPlansCounter)++;
        m_NumWeightBuffers->push_back(numWeightBuffers);

        return plans;
    }

    virtual utils::Optional<ethosn::command_stream::MceOperation> GetMceOperation() const override
    {
        return {};
    }

    bool CanDoubleBufferWeights() const override
    {
        return true;
    }

    uint32_t* m_NumPlansCounter;
    std::vector<uint32_t>* m_NumWeightBuffers;
    // Function instance used to store the filter lambda function.
    std::function<bool(CascadeType, PartId)> m_Filter;
};

class NoWeightPart : public WeightPart
{
public:
    NoWeightPart(PartId id,
                 uint32_t* numPlansCounter,
                 std::vector<uint32_t>* weightBuffers,
                 std::function<bool(CascadeType, PartId)> filter,
                 bool hasInput,
                 bool hasOutput)
        : WeightPart(id, numPlansCounter, weightBuffers, filter, hasInput, hasOutput)
    {}

    bool CanDoubleBufferWeights() const override
    {
        return false;
    }
};

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
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);

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
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);

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
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);

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
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);

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
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);

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
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);

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

// Manually creates a 3-part network consisting of MockParts without weights, to test the double buffering logic of the Combiner.
// The topology is chosen to test cases including:
//      * StandalonePle kernels without weights.
//      * Start/Continue/EndSection logic only.
TEST_CASE("DoubleBufferingTestVariant_PleKernelsOnly", "[CombinerDFS]")
{
    // Create graph:
    //
    //  Input - A (Ple) - B (Ple) - C (Ple)
    //

    GraphOfParts graph;
    auto& parts              = graph.m_Parts;
    auto& connections        = graph.m_Connections;
    uint32_t numPlansCounter = 0;
    std::array<std::vector<uint32_t>, 3> planWeightBuffers;
    // Filter lambda function used to force the Combiner in generating specific Plans for specific Parts.
    auto filter = [](auto cascadeType, auto partId) {
        return ((partId == 0 && cascadeType == CascadeType::Beginning) ||
                (partId == 1 && cascadeType == CascadeType::Middle) ||
                (partId == 2 && cascadeType == CascadeType::End));
    };

    auto pA = std::make_unique<NoWeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[0], filter,
                                             false, true);
    auto pB = std::make_unique<NoWeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[1], filter,
                                             true, true);
    auto pC = std::make_unique<NoWeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[2], filter,
                                             true, false);

    BasePart& partA = *pA;

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

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);
    combiner.TopologicalSortParts();
    Combination comb = combiner.FindBestCombinationForPart(partA);

    // The network consists of mocked Ple kernels without weights, hence no double buffering should be taking place.
    // Number of plans: 1 weightStripes * (1 PlePlan/weightStripe + 1 PlePlan/weightStripe + 1 PlePlan/weightStripe) == 3.
    // Number of weightStripes per plan per part: e.g. PlePart generates 1x plans, with 1 weightStripes.
    REQUIRE(numPlansCounter == 3);
    REQUIRE(planWeightBuffers[0] == std::vector<uint32_t>{ 1 });
    REQUIRE(planWeightBuffers[1] == std::vector<uint32_t>{ 1 });
    REQUIRE(planWeightBuffers[2] == std::vector<uint32_t>{ 1 });
}

// Manually creates a 3-part network consisting of MockParts with and without weights, to test the double buffering logic of the Combiner.
// The topology is chosen to test cases including:
//      * Mce and StandalonePle kernels with and without weights.
//      * SinglePartSection logic only.
TEST_CASE("DoubleBufferingTestVariant_SinglePartSection", "[CombinerDFS]")
{
    // Create graph:
    //
    //  Input - A (Ple) - B (Mce) - C (Ple)
    //

    GraphOfParts graph;
    auto& parts              = graph.m_Parts;
    auto& connections        = graph.m_Connections;
    uint32_t numPlansCounter = 0;
    std::array<std::vector<uint32_t>, 3> planWeightBuffers;
    // Filter lambda function used to force the Combiner in generating specific Plans for specific Parts.
    auto filter = [](auto cascadeType, auto partId) {
        return ((partId == 1 && cascadeType == CascadeType::Lonely) ||
                (partId == 2 && cascadeType == CascadeType::Lonely) ||
                (partId == 3 && cascadeType == CascadeType::Lonely));
    };

    auto pInput = std::make_unique<MockPart>(graph.GeneratePartId(), false, true);
    auto pA = std::make_unique<NoWeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[0], filter,
                                             true, true);
    auto pB = std::make_unique<WeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[1], filter,
                                           true, true);
    auto pC = std::make_unique<NoWeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[2], filter,
                                             true, false);

    BasePart& partA = *pA;

    PartId partInputId = pInput->GetPartId();
    PartId partAId     = pA->GetPartId();
    PartId partBId     = pB->GetPartId();
    PartId partCId     = pC->GetPartId();
    parts.push_back(std::move(pInput));
    parts.push_back(std::move(pA));
    parts.push_back(std::move(pB));
    parts.push_back(std::move(pC));

    PartOutputSlot partInputOutputSlot0 = { partInputId, 0 };

    PartInputSlot partAInputSlot0   = { partAId, 0 };
    PartOutputSlot partAOutputSlot0 = { partAId, 0 };

    PartInputSlot partBInputSlot0   = { partBId, 0 };
    PartOutputSlot partBOutputSlot0 = { partBId, 0 };

    PartInputSlot partCInputSlot0 = { partCId, 0 };

    connections[partAInputSlot0] = partInputOutputSlot0;
    connections[partBInputSlot0] = partAOutputSlot0;
    connections[partCInputSlot0] = partBOutputSlot0;

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);
    combiner.TopologicalSortParts();
    Combination comb = combiner.FindBestCombinationForPart(partA);

    // The network consists of mocked Mce and Ple kernels with and without weights, hence only the first Mce should be double buffered.
    // Number of plans: 1 weightStripes * 1 PlePlan/weightStripe + 2 weightStripes * 1 McePlan/weightStripe + 1 weightStripe * 1 PlePlan/weightStripe == 4.
    // Number of weightStripes per plan per part: PlePart generates 1x plans, with 1 weightStripes.
    //                                            McePart generates 2x plans, with 1 and 2 weightStripes respectively.
    //                                            PlePart generates 1x plans, with 1 weightStripes, since only SinglePartSection is called.
    REQUIRE(numPlansCounter == 4);
    REQUIRE(planWeightBuffers[0] == std::vector<uint32_t>{ 1 });
    REQUIRE(planWeightBuffers[1] == std::vector<uint32_t>{ 1, 2 });
    REQUIRE(planWeightBuffers[2] == std::vector<uint32_t>{ 1 });
}

// Manually creates a 3-part network consisting of MockParts with and without weights, to test the double buffering logic of the Combiner.
// The topology is chosen to test cases including:
//      * Mce and StandalonePle kernels with and without weights.
//      * Start/Continue/EndSection logic only.
TEST_CASE("DoubleBufferingTestVariant_McePleMce", "[CombinerDFS]")
{
    // Create graph:
    //
    //  Input - A (Mce) - B (Ple) - C (Mce)
    //

    GraphOfParts graph;
    auto& parts              = graph.m_Parts;
    auto& connections        = graph.m_Connections;
    uint32_t numPlansCounter = 0;
    std::array<std::vector<uint32_t>, 3> planWeightBuffers;
    // Filter lambda function used to force the Combiner in generating specific Plans for specific Parts.
    auto filter = [](auto cascadeType, auto partId) {
        return ((partId == 1 && cascadeType == CascadeType::Beginning) ||
                (partId == 2 && cascadeType == CascadeType::Middle) ||
                (partId == 3 && cascadeType == CascadeType::End));
    };

    auto pInput = std::make_unique<MockPart>(graph.GeneratePartId(), false, true);
    auto pA     = std::make_unique<WeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[0], filter,
                                           true, true);
    auto pB = std::make_unique<NoWeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[1], filter,
                                             true, true);
    auto pC = std::make_unique<WeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[2], filter,
                                           true, true);
    auto pOutput = std::make_unique<MockPart>(graph.GeneratePartId(), true, false);

    BasePart& partA = *pA;

    PartId partInputId  = pInput->GetPartId();
    PartId partAId      = pA->GetPartId();
    PartId partBId      = pB->GetPartId();
    PartId partCId      = pC->GetPartId();
    PartId partOutputId = pOutput->GetPartId();
    parts.push_back(std::move(pInput));
    parts.push_back(std::move(pA));
    parts.push_back(std::move(pB));
    parts.push_back(std::move(pC));
    parts.push_back(std::move(pOutput));

    PartOutputSlot partInputOutputSlot0 = { partInputId, 0 };

    PartInputSlot partAInputSlot0   = { partAId, 0 };
    PartOutputSlot partAOutputSlot0 = { partAId, 0 };

    PartInputSlot partBInputSlot0   = { partBId, 0 };
    PartOutputSlot partBOutputSlot0 = { partBId, 0 };

    PartInputSlot partCInputSlot0   = { partCId, 0 };
    PartOutputSlot partCOutputSlot0 = { partCId, 0 };

    PartInputSlot partOutputInputSlot0 = { partOutputId, 0 };

    connections[partAInputSlot0]      = partInputOutputSlot0;
    connections[partBInputSlot0]      = partAOutputSlot0;
    connections[partCInputSlot0]      = partBOutputSlot0;
    connections[partOutputInputSlot0] = partCOutputSlot0;

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);
    combiner.TopologicalSortParts();
    Combination comb = combiner.FindBestCombinationForPart(partA);

    // The network consists of mocked Mce and Ple kernels with and without weights, hence only the first Mce should be double buffered.
    // Number of plans: 2 weightStripes * (1 McePlan/weightStripe + 1 PlePlan/weightStripe + 1 PlePlan/weightStripe) == 6.
    // Number of weightStripes per plan per part: e.g. McePart generates 2x plans, with 1 and 2 weightStripes respectively.
    REQUIRE(numPlansCounter == 6);
    REQUIRE(planWeightBuffers[0] == std::vector<uint32_t>{ 1, 2 });
    REQUIRE(planWeightBuffers[1] == std::vector<uint32_t>{ 1, 2 });
    REQUIRE(planWeightBuffers[2] == std::vector<uint32_t>{ 1, 2 });
}

// Manually creates a 3-part network consisting of MockParts with and without weights, to test the double buffering logic of the Combiner.
// The topology is chosen to test cases including:
//      * Mce and StandalonePle kernels with and without weights.
//      * Start/Continue/EndSection logic only.
TEST_CASE("DoubleBufferingTestVariant_PleMceMce", "[CombinerDFS]")
{
    // Create graph:
    //
    //  Input - A (Ple) - B (Mce) - C (Mce)
    //

    GraphOfParts graph;
    auto& parts              = graph.m_Parts;
    auto& connections        = graph.m_Connections;
    uint32_t numPlansCounter = 0;
    std::array<std::vector<uint32_t>, 3> planWeightBuffers;
    // Filter lambda function used to force the Combiner in generating specific Plans for specific Parts.
    auto filter = [](auto cascadeType, auto partId) {
        return ((partId == 1 && cascadeType == CascadeType::Beginning) ||
                (partId == 2 && cascadeType == CascadeType::Middle) ||
                (partId == 3 && cascadeType == CascadeType::End));
    };

    auto pInput = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pA = std::make_unique<NoWeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[0], filter,
                                             true, true);
    auto pB = std::make_unique<WeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[1], filter,
                                           true, true);
    auto pC = std::make_unique<WeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[2], filter,
                                           true, true);
    auto pOutput = std::make_unique<MockPart>(graph.GeneratePartId(), true, false);

    BasePart& partA = *pA;

    PartId partInputId  = pInput->GetPartId();
    PartId partAId      = pA->GetPartId();
    PartId partBId      = pB->GetPartId();
    PartId partCId      = pC->GetPartId();
    PartId partOutputId = pOutput->GetPartId();
    parts.push_back(std::move(pInput));
    parts.push_back(std::move(pA));
    parts.push_back(std::move(pB));
    parts.push_back(std::move(pC));
    parts.push_back(std::move(pOutput));

    PartOutputSlot partInputOutputSlot0 = { partInputId, 0 };

    PartInputSlot partAInputSlot0   = { partAId, 0 };
    PartOutputSlot partAOutputSlot0 = { partAId, 0 };

    PartInputSlot partBInputSlot0   = { partBId, 0 };
    PartOutputSlot partBOutputSlot0 = { partBId, 0 };

    PartInputSlot partCInputSlot0   = { partCId, 0 };
    PartOutputSlot partCOutputSlot0 = { partCId, 0 };

    PartInputSlot partOutputInputSlot0 = { partOutputId, 0 };

    connections[partAInputSlot0]      = partInputOutputSlot0;
    connections[partBInputSlot0]      = partAOutputSlot0;
    connections[partCInputSlot0]      = partBOutputSlot0;
    connections[partOutputInputSlot0] = partCOutputSlot0;

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);
    combiner.TopologicalSortParts();
    Combination comb = combiner.FindBestCombinationForPart(partA);

    // The network consists of mocked Mce and Ple kernels with and without weights, hence only the first Mce should be double buffered.
    // Number of plans: 1 weightStripe * 1 PlePlan/weightStripe + 2 weightStripes * (1 McePlan/weightStripe + 1 McePlan/weightStripe) == 5.
    // Number of weightStripes per plan per part: e.g. PlePart generates 1x plans, with 1 weightStripe.
    //                                            e.g. McePart generates 2x plans, with 1 and 2 weightStripes respectively.
    REQUIRE(numPlansCounter == 5);
    REQUIRE(planWeightBuffers[0] == std::vector<uint32_t>{ 1 });
    REQUIRE(planWeightBuffers[1] == std::vector<uint32_t>{ 1, 2 });
    REQUIRE(planWeightBuffers[2] == std::vector<uint32_t>{ 1, 2 });
}

// Manually creates a 4-part network consisting of MockParts with and without weights, to test the double buffering logic of the Combiner.
// The topology is chosen to test cases including:
//      * Mce and StandalonePle kernels with and without weights.
//      * Start/Continue/EndSection logic only.
TEST_CASE("DoubleBufferingTestVariant_PleMceMcePle", "[CombinerDFS]")
{
    // Create graph:
    //
    //  Input - A (Ple) - B (Mce) - C (Mce) - D (Ple)
    //

    GraphOfParts graph;
    auto& parts              = graph.m_Parts;
    auto& connections        = graph.m_Connections;
    uint32_t numPlansCounter = 0;
    std::array<std::vector<uint32_t>, 4> planWeightBuffers;
    // Filter lambda function used to force the Combiner in generating specific Plans for specific Parts.
    auto filter = [](auto cascadeType, auto partId) {
        return ((partId == 1 && cascadeType == CascadeType::Beginning) ||
                (partId == 2 && cascadeType == CascadeType::Middle) ||
                (partId == 3 && cascadeType == CascadeType::Middle) ||
                (partId == 4 && cascadeType == CascadeType::End));
    };

    auto pInput = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pA = std::make_unique<NoWeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[0], filter,
                                             true, true);
    auto pB = std::make_unique<WeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[1], filter,
                                           true, true);
    auto pC = std::make_unique<WeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[2], filter,
                                           true, true);
    auto pD = std::make_unique<NoWeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[3], filter,
                                             true, true);
    auto pOutput = std::make_unique<MockPart>(graph.GeneratePartId(), true, false);

    BasePart& partA = *pA;

    PartId partInputId  = pInput->GetPartId();
    PartId partAId      = pA->GetPartId();
    PartId partBId      = pB->GetPartId();
    PartId partCId      = pC->GetPartId();
    PartId partDId      = pD->GetPartId();
    PartId partOutputId = pOutput->GetPartId();
    parts.push_back(std::move(pInput));
    parts.push_back(std::move(pA));
    parts.push_back(std::move(pB));
    parts.push_back(std::move(pC));
    parts.push_back(std::move(pD));
    parts.push_back(std::move(pOutput));

    PartOutputSlot partInputOutputSlot0 = { partInputId, 0 };

    PartInputSlot partAInputSlot0   = { partAId, 0 };
    PartOutputSlot partAOutputSlot0 = { partAId, 0 };

    PartInputSlot partBInputSlot0   = { partBId, 0 };
    PartOutputSlot partBOutputSlot0 = { partBId, 0 };

    PartInputSlot partCInputSlot0   = { partCId, 0 };
    PartOutputSlot partCOutputSlot0 = { partCId, 0 };

    PartInputSlot partDInputSlot0   = { partDId, 0 };
    PartOutputSlot partDOutputSlot0 = { partDId, 0 };

    PartInputSlot partOutputInputSlot0 = { partOutputId, 0 };

    connections[partAInputSlot0]      = partInputOutputSlot0;
    connections[partBInputSlot0]      = partAOutputSlot0;
    connections[partCInputSlot0]      = partBOutputSlot0;
    connections[partDInputSlot0]      = partCOutputSlot0;
    connections[partOutputInputSlot0] = partDOutputSlot0;

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);
    combiner.TopologicalSortParts();
    Combination comb = combiner.FindBestCombinationForPart(partA);

    // The network consists of mocked Mce and Ple kernels with and without weights, hence only the first Mce should be double buffered.
    // Number of plans: 1 weightStripe * 1 PlePlan/weightStripe + 2 weightStripes * (1 McePlan/weightStripe + 1 McePlan/weightStripe + 1 PlePlan/weightStripe) == 7.
    // Number of weightStripes per plan per part: e.g. PlePart generates 1x plans, with 1 weightStripe.
    //                                            e.g. McePart generates 2x plans, with 1 and 2 weightStripes respectively.
    REQUIRE(numPlansCounter == 7);
    REQUIRE(planWeightBuffers[0] == std::vector<uint32_t>{ 1 });
    REQUIRE(planWeightBuffers[1] == std::vector<uint32_t>{ 1, 2 });
    REQUIRE(planWeightBuffers[2] == std::vector<uint32_t>{ 1, 2 });
    REQUIRE(planWeightBuffers[3] == std::vector<uint32_t>{ 1, 2 });
}

// Manually creates a plan using Ops with Atomic Lifetimes to test the SramAllocator logic.
// The topology is chosen to test cases including:
//      * Sram Buffers which are consumed by Ops with Atomic Lifetime.
//      * PleInputSram Buffer which is consumed by a Ple Op
// I -> Mce -> PleInputSram -> Ple -> O
//   /
//  W
TEST_CASE("BufferDeallocationTest_AtomicOps", "[CombinerDFS]")
{
    GraphOfParts graph;
    auto& parts = graph.m_Parts;

    auto pA = std::make_unique<MockPart>(graph.GeneratePartId());

    PartId partAId = pA->GetPartId();
    parts.push_back(std::move(pA));

    PartInputSlot partAInputSlot0   = { partAId, 0 };
    PartOutputSlot partAOutputSlot0 = { partAId, 0 };

    const uint32_t ifmSize = 1 * 16 * 16 * 16;
    const uint32_t ofmSize = 1 * 16 * 16 * 16;

    // Plan A
    Plan planA;
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 16, 16, 16 }, TensorShape{ 1, 16, 16, 16 },
                                                       TraversalOrder::Xyz, ifmSize, QuantizationInfo()));
    planA.m_OpGraph.GetBuffers().back()->m_DebugTag = "InputSram";
    size_t inputBufferIndex                         = planA.m_OpGraph.GetBuffers().size() - 1;

    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::PleInputSram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 16, 16, 16 }, TensorShape{ 1, 16, 16, 16 },
                                                       TraversalOrder::Xyz, ifmSize, QuantizationInfo()));
    planA.m_OpGraph.GetBuffers().back()->m_DebugTag = "PleInputSram";
    size_t pleInputSramIndex                        = planA.m_OpGraph.GetBuffers().size() - 1;

    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 1, 1, 16 }, TensorShape{ 1, 1, 1, 16 },
                                                       TraversalOrder::Xyz, (uint32_t)16, QuantizationInfo()));
    planA.m_OpGraph.GetBuffers().back()->m_DebugTag = "MceWeightsSram";
    size_t mceWeightsBufferIndex                    = planA.m_OpGraph.GetBuffers().size() - 1;

    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 16, 16, 16 }, TensorShape{ 1, 16, 16, 16 },
                                                       TraversalOrder::Xyz, ofmSize, QuantizationInfo()));
    planA.m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputSram";
    size_t outputBufferIndex                        = planA.m_OpGraph.GetBuffers().size() - 1;

    planA.m_OpGraph.AddOp(std::make_unique<MceOp>(
        MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct, BlockConfig{ 16u, 16u }, TensorShape{ 1, 16, 16, 16 },
        TensorShape{ 1, 16, 16, 16 }, TensorShape{ 1, 1, 1, 16 }, TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));
    planA.m_OpGraph.GetOps()[0]->m_DebugTag = "MceOp";
    size_t mceOpIndex                       = planA.m_OpGraph.GetOps().size() - 1;
    planA.m_OpGraph.AddOp(std::make_unique<PleOp>(ethosn::command_stream::PleOperation::PASSTHROUGH,
                                                  BlockConfig{ 8u, 8u }, 1,
                                                  std::vector<TensorShape>{ TensorShape{ 1, 16, 16, 16 } },
                                                  TensorShape{ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, true));
    size_t pleOpIndex                       = planA.m_OpGraph.GetOps().size() - 1;
    planA.m_OpGraph.GetOps()[1]->m_DebugTag = "PleOp";
    planA.m_OpGraph.AddConsumer(planA.m_OpGraph.GetBuffers()[inputBufferIndex], planA.m_OpGraph.GetOps()[mceOpIndex],
                                0);
    planA.m_OpGraph.SetProducer(planA.m_OpGraph.GetBuffers()[pleInputSramIndex], planA.m_OpGraph.GetOps()[mceOpIndex]);
    planA.m_OpGraph.AddConsumer(planA.m_OpGraph.GetBuffers()[pleInputSramIndex], planA.m_OpGraph.GetOps()[pleOpIndex],
                                0);
    planA.m_OpGraph.AddConsumer(planA.m_OpGraph.GetBuffers()[mceWeightsBufferIndex],
                                planA.m_OpGraph.GetOps()[mceOpIndex], 1);
    planA.m_OpGraph.SetProducer(planA.m_OpGraph.GetBuffers()[outputBufferIndex], planA.m_OpGraph.GetOps()[pleOpIndex]);
    planA.m_InputMappings  = { { planA.m_OpGraph.GetBuffers()[inputBufferIndex], partAInputSlot0 } };
    planA.m_OutputMappings = { { planA.m_OpGraph.GetBuffers()[outputBufferIndex], partAOutputSlot0 } };

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);
    combiner.TopologicalSortParts();

    SectionContext context = { SramAllocator(hwCaps.GetTotalSramSize() / hwCaps.GetNumberOfSrams()), {}, {} };

    // Check that no buffers are allocated before calling IsPlanAllocated().
    REQUIRE(context.alloc.GetAllocationSize() == 0);
    REQUIRE(combiner.IsPlanAllocated(context, planA, nullptr, StatsType::StartSection) == true);
    // Check that all 4 buffers (Input, Mce Weights, Ple Code, Output) have been allocated.
    REQUIRE(context.alloc.GetAllocationSize() == 4);
    // Check that 2 buffers (Mce Weights, Input) have been deallocated.
    combiner.DeallocateUnusedBuffers(*planA.m_OpGraph.GetBuffers()[outputBufferIndex], context);
    REQUIRE(context.alloc.GetAllocationSize() == 2);
    // Check that it is only the Input and Mce Weights buffers that have been deallocated.
    REQUIRE(context.alloc.TryFree(0, planA.m_OpGraph.GetBuffers()[inputBufferIndex]->m_Offset.value()) == false);
    REQUIRE(context.alloc.TryFree(0, planA.m_OpGraph.GetBuffers()[mceWeightsBufferIndex]->m_Offset.value()) == false);
}

// Manually creates a plan using Ops with Atomic Lifetimes to test the SramAllocator logic.
// The topology is chosen to test cases including:
//      * Sram Buffers which are consumed by Ops with Atomic Lifetime.
//      * Sram Buffers which are consumed by Ops with Cascade Lifetime.
//      * PleInputSram Buffer which is consumed by a Ple Op
// I -> Mce -> PleInputSram -> Ple -> O
//   /
//  W
TEST_CASE("BufferDeallocationTest_CascadeOps", "[CombinerDFS]")
{
    GraphOfParts graph;
    auto& parts = graph.m_Parts;

    auto pA = std::make_unique<MockPart>(graph.GeneratePartId());

    PartId partAId = pA->GetPartId();
    parts.push_back(std::move(pA));

    PartInputSlot partAInputSlot0   = { partAId, 0 };
    PartOutputSlot partAOutputSlot0 = { partAId, 0 };

    const uint32_t ifmSize = 1 * 16 * 16 * 16;
    const uint32_t ofmSize = 1 * 16 * 16 * 16;

    // Plan A
    Plan planA;
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 16, 16, 16 }, TensorShape{ 1, 16, 16, 16 },
                                                       TraversalOrder::Xyz, ifmSize, QuantizationInfo()));
    planA.m_OpGraph.GetBuffers().back()->m_DebugTag = "InputSram";
    size_t inputBufferIndex                         = planA.m_OpGraph.GetBuffers().size() - 1;

    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::PleInputSram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 16, 16, 16 }, TensorShape{ 1, 16, 16, 16 },
                                                       TraversalOrder::Xyz, ifmSize, QuantizationInfo()));
    planA.m_OpGraph.GetBuffers().back()->m_DebugTag = "PleInputSram";
    size_t pleInputSramIndex                        = planA.m_OpGraph.GetBuffers().size() - 1;

    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 1, 1, 16 }, TensorShape{ 1, 1, 1, 16 },
                                                       TraversalOrder::Xyz, (uint32_t)16, QuantizationInfo()));
    planA.m_OpGraph.GetBuffers().back()->m_DebugTag = "MceWeightsSram";
    size_t mceWeightsBufferIndex                    = planA.m_OpGraph.GetBuffers().size() - 1;

    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 32, 16, 16 }, TensorShape{ 1, 16, 16, 16 },
                                                       TraversalOrder::Xyz, ofmSize, QuantizationInfo()));
    planA.m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputSram";
    size_t outputBufferIndex                        = planA.m_OpGraph.GetBuffers().size() - 1;

    planA.m_OpGraph.AddOp(std::make_unique<MceOp>(
        MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct, BlockConfig{ 16u, 16u }, TensorShape{ 1, 16, 16, 16 },
        TensorShape{ 1, 16, 16, 16 }, TensorShape{ 1, 1, 1, 16 }, TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));
    planA.m_OpGraph.GetOps()[0]->m_DebugTag = "MceOp";
    size_t mceOpIndex                       = planA.m_OpGraph.GetOps().size() - 1;
    planA.m_OpGraph.AddOp(std::make_unique<PleOp>(ethosn::command_stream::PleOperation::PASSTHROUGH,
                                                  BlockConfig{ 8u, 8u }, 1,
                                                  std::vector<TensorShape>{ TensorShape{ 1, 16, 16, 16 } },
                                                  TensorShape{ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, true));
    size_t pleOpIndex                       = planA.m_OpGraph.GetOps().size() - 1;
    planA.m_OpGraph.GetOps()[1]->m_DebugTag = "PleOp";
    planA.m_OpGraph.AddConsumer(planA.m_OpGraph.GetBuffers()[inputBufferIndex], planA.m_OpGraph.GetOps()[mceOpIndex],
                                0);
    planA.m_OpGraph.SetProducer(planA.m_OpGraph.GetBuffers()[pleInputSramIndex], planA.m_OpGraph.GetOps()[mceOpIndex]);
    planA.m_OpGraph.AddConsumer(planA.m_OpGraph.GetBuffers()[pleInputSramIndex], planA.m_OpGraph.GetOps()[pleOpIndex],
                                0);
    planA.m_OpGraph.AddConsumer(planA.m_OpGraph.GetBuffers()[mceWeightsBufferIndex],
                                planA.m_OpGraph.GetOps()[mceOpIndex], 1);
    planA.m_OpGraph.SetProducer(planA.m_OpGraph.GetBuffers()[outputBufferIndex], planA.m_OpGraph.GetOps()[pleOpIndex]);
    planA.m_InputMappings  = { { planA.m_OpGraph.GetBuffers()[inputBufferIndex], partAInputSlot0 } };
    planA.m_OutputMappings = { { planA.m_OpGraph.GetBuffers()[outputBufferIndex], partAOutputSlot0 } };

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);
    combiner.TopologicalSortParts();

    SectionContext context = { SramAllocator(hwCaps.GetTotalSramSize() / hwCaps.GetNumberOfSrams()), {}, {} };

    // Check that no buffers are allocated before calling IsPlanAllocated().
    REQUIRE(context.alloc.GetAllocationSize() == 0);
    REQUIRE(combiner.IsPlanAllocated(context, planA, nullptr, StatsType::StartSection) == true);
    // Check that all 4 buffers (Input, Mce Weights, Ple Code, Output) have been allocated.
    REQUIRE(context.alloc.GetAllocationSize() == 4);
    // Check that none of the buffers have been deallocated.
    combiner.DeallocateUnusedBuffers(*planA.m_OpGraph.GetBuffers()[outputBufferIndex], context);
    REQUIRE(context.alloc.GetAllocationSize() == 4);
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
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 0, QuantizationInfo()));
    planA.m_OpGraph.GetBuffers().back()->m_DebugTag = "InputSramA";
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 0, QuantizationInfo()));
    planA.m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputSramA";
    planA.m_OutputMappings                          = { { planA.m_OpGraph.GetBuffers()[1], partAOutputSlot0 } };
    planA.m_OpGraph.AddOp(std::make_unique<MceOp>(
        MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct, BlockConfig{ 16u, 16u }, TensorShape{ 1, 17, 16, 16 },
        TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 1, 1, 16 }, TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));
    planA.m_OpGraph.GetOps()[0]->m_DebugTag = "MceA";
    planA.m_OpGraph.AddConsumer(planA.m_OpGraph.GetBuffers()[0], planA.m_OpGraph.GetOps()[0], 0);
    planA.m_OpGraph.SetProducer(planA.m_OpGraph.GetBuffers()[1], planA.m_OpGraph.GetOps()[0]);

    // Plan B
    Plan planB;
    planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planB.m_OpGraph.GetBuffers().back()->m_DebugTag = "InputSramB";
    planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planB.m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputSramB";
    planB.m_InputMappings                           = { { planB.m_OpGraph.GetBuffers()[0], partBInputSlot0 } };
    planB.m_OutputMappings                          = { { planB.m_OpGraph.GetBuffers()[1], partBOutputSlot0 } };
    planB.m_OpGraph.AddOp(std::make_unique<MceOp>(
        MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct, BlockConfig{ 16u, 16u }, TensorShape{ 1, 17, 16, 16 },
        TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 1, 1, 16 }, TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));
    planB.m_OpGraph.GetOps()[0]->m_DebugTag = "MceB";
    planB.m_OpGraph.AddConsumer(planB.m_OpGraph.GetBuffers()[0], planB.m_OpGraph.GetOps()[0], 0);
    planB.m_OpGraph.SetProducer(planB.m_OpGraph.GetBuffers()[1], planB.m_OpGraph.GetOps()[0]);

    // Plan C
    Plan planC;
    planC.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planC.m_OpGraph.GetBuffers().back()->m_DebugTag = "InputSramC";
    planC.m_InputMappings                           = { { planC.m_OpGraph.GetBuffers()[0], partCInputSlot0 } };

    auto endingGlueA = std::make_shared<EndingGlue>();
    endingGlueA->m_Graph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
    endingGlueA->m_Graph.GetOps()[0]->m_DebugTag = "InputDma";
    endingGlueA->m_Graph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                            TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                            TraversalOrder::Xyz, 0, QuantizationInfo()));
    endingGlueA->m_Graph.GetBuffers().back()->m_BufferType = BufferType::Intermediate;
    endingGlueA->m_Graph.GetBuffers().back()->m_DebugTag   = "DramBuffer";
    endingGlueA->m_Graph.SetProducer(endingGlueA->m_Graph.GetBuffers()[0], endingGlueA->m_Graph.GetOps()[0]);
    endingGlueA->m_ExternalConnections.m_BuffersToOps.insert(
        { planA.m_OpGraph.GetBuffers()[1], endingGlueA->m_Graph.GetOp(0) });

    auto startingGlueB = std::make_shared<StartingGlue>();
    startingGlueB->m_Graph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
    startingGlueB->m_Graph.GetOps()[0]->m_DebugTag = "OutputDma";
    startingGlueB->m_ExternalConnections.m_BuffersToOps.insert(
        { endingGlueA->m_Graph.GetBuffers().back(), startingGlueB->m_Graph.GetOp(0) });
    startingGlueB->m_ExternalConnections.m_OpsToBuffers.insert(
        { startingGlueB->m_Graph.GetOps()[0], planB.m_OpGraph.GetBuffers().front() });

    auto endingGlueB = std::make_shared<EndingGlue>();

    auto startingGlueC = std::make_shared<StartingGlue>();
    startingGlueC->m_Graph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
    startingGlueC->m_Graph.GetOps()[0]->m_DebugTag = "InputDmaC";
    startingGlueC->m_ExternalConnections.m_BuffersToOps.insert(
        { planB.m_OpGraph.GetBuffers().back(), startingGlueC->m_Graph.GetOp(0) });
    startingGlueC->m_ExternalConnections.m_OpsToBuffers.insert(
        { startingGlueC->m_Graph.GetOp(0), planC.m_OpGraph.GetBuffers().front() });

    // Create Combination with all the plans and glues
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
        SaveCombinationToDot(comb, stream, DetailLevel::High);
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

// Manually creating a MISO test case with three parts
//         A   B
//          \ /
//           C
// Both A and B's output buffers's location are in SRAM
// C's input buffer is DRAM
// This test is to validate the order of the operations
// is as expected in the Op graph.
TEST_CASE("GetOpGraphForDfsMISOSramsToDrams", "[CombinerDFS]")
{
    DebuggableObject::ms_IdCounter = 0;    // Reset counter so we get deterministic results

    GraphOfParts graph;
    auto& parts       = graph.m_Parts;
    auto& connections = graph.m_Connections;

    auto pA = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pC = std::make_unique<MockPart>(graph.GeneratePartId());

    BasePart& partA = *pA;
    BasePart& partB = *pB;
    BasePart& partC = *pC;

    parts.push_back(std::move(pA));
    parts.push_back(std::move(pB));
    parts.push_back(std::move(pC));

    PartOutputSlot partAOutputSlot = { partA.GetPartId(), 0 };
    PartOutputSlot partBOutputSlot = { partB.GetPartId(), 0 };

    PartInputSlot partCInputSlot0 = { partC.GetPartId(), 0 };
    PartInputSlot partCInputSlot1 = { partC.GetPartId(), 1 };

    connections[partCInputSlot0] = { partAOutputSlot };
    connections[partCInputSlot1] = { partBOutputSlot };

    const CompilationOptions compOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);

    // Plan A
    Plan planA;
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planA.m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputSramA";
    planA.m_OutputMappings                          = { { planA.m_OpGraph.GetBuffers()[0], partAOutputSlot } };
    auto dummyOpA                                   = std::make_unique<DummyOp>();
    dummyOpA->m_DebugTag                            = "DummyA";
    planA.m_OpGraph.AddOp(std::move(dummyOpA));

    // Plan B
    Plan planB;
    planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planB.m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputSramB";
    planB.m_OutputMappings                          = { { planB.m_OpGraph.GetBuffers()[0], partBOutputSlot } };
    auto dummyOpB                                   = std::make_unique<DummyOp>();
    dummyOpB->m_DebugTag                            = "DummyB";
    planB.m_OpGraph.AddOp(std::move(dummyOpB));

    // Glue between A and C
    auto endingGlueA = std::make_shared<EndingGlue>();
    endingGlueA->m_Graph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
    endingGlueA->m_Graph.GetOps()[0]->m_DebugTag = "GlueAC_Dma";
    endingGlueA->m_ExternalConnections.m_BuffersToOps.insert(
        { planA.m_OpGraph.GetBuffers().back(), endingGlueA->m_Graph.GetOps().back() });

    auto endingGlueB = std::make_shared<EndingGlue>();
    endingGlueB->m_Graph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
    endingGlueB->m_Graph.GetOps()[0]->m_DebugTag = "GlueBC_Dma";
    endingGlueB->m_ExternalConnections.m_BuffersToOps.insert(
        { planB.m_OpGraph.GetBuffers().back(), endingGlueB->m_Graph.GetOps().back() });

    // Plan C
    Plan planC;
    planC.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planC.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Intermediate;
    planC.m_OpGraph.GetBuffers().back()->m_DebugTag   = "Input0DramC";
    planC.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planC.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Intermediate;
    planC.m_OpGraph.GetBuffers().back()->m_DebugTag   = "Input1DramC";
    planC.m_InputMappings                             = { { planC.m_OpGraph.GetBuffers()[0], partCInputSlot0 },
                              { planC.m_OpGraph.GetBuffers()[1], partCInputSlot1 } };
    auto dummyOpC                                     = std::make_unique<DummyOp>();
    dummyOpC->m_DebugTag                              = "DummyC";
    planC.m_OpGraph.AddOp(std::move(dummyOpC));

    auto startingGlueC_A = std::make_shared<StartingGlue>();
    startingGlueC_A->m_ExternalConnections.m_OpsToBuffers.insert(
        { endingGlueA->m_Graph.GetOps().back(), planC.m_OpGraph.GetBuffers()[0] });

    auto startingGlueC_B = std::make_shared<StartingGlue>();
    startingGlueC_A->m_ExternalConnections.m_OpsToBuffers.insert(
        { endingGlueB->m_Graph.GetOps().back(), planC.m_OpGraph.GetBuffers()[1] });

    // Create Combination with all the plans and glues
    Combination comb;

    Elem elemA;
    elemA.m_Plan        = std::make_shared<Plan>(std::move(planA));
    elemA.m_EndingGlues = { { partAOutputSlot, endingGlueA } };

    Elem elemB;
    elemB.m_Plan        = std::make_shared<Plan>(std::move(planB));
    elemB.m_EndingGlues = { { partBOutputSlot, endingGlueB } };

    Elem elemC;
    elemC.m_Plan          = std::make_shared<Plan>(std::move(planC));
    elemC.m_StartingGlues = { { partCInputSlot0, startingGlueC_A }, { partCInputSlot1, startingGlueC_B } };

    comb.m_Elems.insert(std::make_pair(0, elemA));
    comb.m_PartIdsInOrder.push_back(0);
    comb.m_Elems.insert(std::make_pair(1, elemB));
    comb.m_PartIdsInOrder.push_back(1);
    comb.m_Elems.insert(std::make_pair(2, elemC));
    comb.m_PartIdsInOrder.push_back(2);

    bool dumpInputGraphToFile = false;
    if (dumpInputGraphToFile)
    {
        std::ofstream stream("GetOpGraphForDfsMISOSramsToDrams.dot");
        SaveCombinationToDot(comb, stream, DetailLevel::High);
    }

    // Call function under test
    OpGraph combOpGraph = GetOpGraphForCombination(comb, graph);

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("GetOpGraphForDfsCombinationMergedBuffer Output.dot");
        SaveOpGraphToDot(combOpGraph, stream, DetailLevel::High);
    }

    // Output buffers of the sources A, B are both in SRAM
    // Therefore, the input DMA of the glue must be inserted
    // right after the op of its source plan
    // The correct order of should be:
    // (1) opA (2) glueAC_DMA (3) op B (4) glueBC_DMA (5) op C
    REQUIRE(combOpGraph.GetBuffers().size() == 4);
    REQUIRE(combOpGraph.GetBuffers()[0]->m_DebugTag == "OutputSramA");
    REQUIRE(combOpGraph.GetBuffers()[1]->m_DebugTag == "OutputSramB");
    REQUIRE(combOpGraph.GetBuffers()[2]->m_DebugTag == "Input0DramC");
    REQUIRE(combOpGraph.GetBuffers()[3]->m_DebugTag == "Input1DramC");
    REQUIRE(combOpGraph.GetOps().size() == 5);
    REQUIRE(combOpGraph.GetOps()[0]->m_DebugTag == "DummyA");
    REQUIRE(combOpGraph.GetOps()[1]->m_DebugTag == "GlueAC_Dma");
    REQUIRE(combOpGraph.GetOps()[2]->m_DebugTag == "DummyB");
    REQUIRE(combOpGraph.GetOps()[3]->m_DebugTag == "GlueBC_Dma");
    REQUIRE(combOpGraph.GetOps()[4]->m_DebugTag == "DummyC");

    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[0]).size() == 1);
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[0])[0].first->m_DebugTag == "GlueAC_Dma");
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[1]).size() == 1);
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[1])[0].first->m_DebugTag == "GlueBC_Dma");
    REQUIRE(combOpGraph.GetProducer(combOpGraph.GetBuffers()[2])->m_DebugTag == "GlueAC_Dma");
    REQUIRE(combOpGraph.GetProducer(combOpGraph.GetBuffers()[3])->m_DebugTag == "GlueBC_Dma");
}

// Manually creating a MISO test case with three parts
//         A   B
//          \ /
//           C
// Both A and B's output buffers's location are in DRAM
// C's input buffer is SRAM
// This test is to validate the order of the operations
// is as expected in the Op graph.
TEST_CASE("GetOpGraphForDfsMISODramsToSrams", "[CombinerDFS]")
{
    DebuggableObject::ms_IdCounter = 0;    // Reset counter so we get deterministic results

    GraphOfParts graph;
    auto& parts       = graph.m_Parts;
    auto& connections = graph.m_Connections;

    auto pA = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pC = std::make_unique<MockPart>(graph.GeneratePartId());

    BasePart& partA = *pA;
    BasePart& partB = *pB;
    BasePart& partC = *pC;

    parts.push_back(std::move(pA));
    parts.push_back(std::move(pB));
    parts.push_back(std::move(pC));

    PartOutputSlot partAOutputSlot = { partA.GetPartId(), 0 };
    PartOutputSlot partBOutputSlot = { partB.GetPartId(), 0 };

    PartInputSlot partCInputSlot0 = { partC.GetPartId(), 0 };
    PartInputSlot partCInputSlot1 = { partC.GetPartId(), 1 };

    connections[partCInputSlot0] = { partAOutputSlot };
    connections[partCInputSlot1] = { partBOutputSlot };

    const CompilationOptions compOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);

    // Plan A
    Plan planA;
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planA.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Intermediate;
    planA.m_OpGraph.GetBuffers().back()->m_DebugTag   = "OutputSramA";
    planA.m_OutputMappings                            = { { planA.m_OpGraph.GetBuffers()[0], partAOutputSlot } };
    auto dummyOpA                                     = std::make_unique<DummyOp>();
    dummyOpA->m_DebugTag                              = "DummyA";
    planA.m_OpGraph.AddOp(std::move(dummyOpA));
    planA.m_OpGraph.SetProducer(planA.m_OpGraph.GetBuffers().back(), planA.m_OpGraph.GetOps().back());

    // Plan B
    Plan planB;
    planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planB.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Intermediate;
    planB.m_OpGraph.GetBuffers().back()->m_DebugTag   = "OutputSramB";
    planB.m_OutputMappings                            = { { planB.m_OpGraph.GetBuffers()[0], partBOutputSlot } };
    auto dummyOpB                                     = std::make_unique<DummyOp>();
    dummyOpB->m_DebugTag                              = "DummyB";
    planB.m_OpGraph.AddOp(std::move(dummyOpB));
    planB.m_OpGraph.SetProducer(planB.m_OpGraph.GetBuffers().back(), planB.m_OpGraph.GetOps().back());

    // Plan C
    Plan planC;
    planC.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planC.m_OpGraph.GetBuffers().back()->m_DebugTag = "Input0DramC";
    planC.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planC.m_OpGraph.GetBuffers().back()->m_DebugTag = "Input1DramC";
    planC.m_InputMappings                           = { { planC.m_OpGraph.GetBuffers()[0], partCInputSlot0 },
                              { planC.m_OpGraph.GetBuffers()[1], partCInputSlot1 } };
    auto dummyOpC                                   = std::make_unique<DummyOp>();
    dummyOpC->m_DebugTag                            = "DummyC";
    planC.m_OpGraph.AddOp(std::move(dummyOpC));

    auto endingGlueA = std::make_shared<EndingGlue>();
    auto endingGlueB = std::make_shared<EndingGlue>();

    auto startingGlueCA = std::make_shared<StartingGlue>();
    startingGlueCA->m_Graph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
    startingGlueCA->m_Graph.GetOps()[0]->m_DebugTag = "GlueAC_Dma";
    startingGlueCA->m_ExternalConnections.m_BuffersToOps.insert(
        { planA.m_OpGraph.GetBuffers().back(), startingGlueCA->m_Graph.GetOp(0) });
    startingGlueCA->m_ExternalConnections.m_OpsToBuffers.insert(
        { startingGlueCA->m_Graph.GetOp(0), planC.m_OpGraph.GetBuffers().front() });

    auto startingGlueCB = std::make_shared<StartingGlue>();
    startingGlueCB->m_Graph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
    startingGlueCB->m_Graph.GetOps()[0]->m_DebugTag = "GlueBC_Dma";
    startingGlueCB->m_ExternalConnections.m_BuffersToOps.insert(
        { planB.m_OpGraph.GetBuffers().back(), startingGlueCB->m_Graph.GetOp(0) });
    startingGlueCB->m_ExternalConnections.m_OpsToBuffers.insert(
        { startingGlueCB->m_Graph.GetOp(0), planC.m_OpGraph.GetBuffers().back() });

    // Create Combination with all the plans and glues
    Combination comb;

    Elem elemA;
    elemA.m_Plan        = std::make_shared<Plan>(std::move(planA));
    elemA.m_EndingGlues = { { partAOutputSlot, endingGlueA } };

    Elem elemB;
    elemB.m_Plan        = std::make_shared<Plan>(std::move(planB));
    elemB.m_EndingGlues = { { partBOutputSlot, endingGlueB } };

    Elem elemC;
    elemC.m_Plan          = std::make_shared<Plan>(std::move(planC));
    elemC.m_StartingGlues = { { partCInputSlot0, startingGlueCA }, { partCInputSlot1, startingGlueCB } };

    comb.m_Elems.insert(std::make_pair(0, elemA));
    comb.m_PartIdsInOrder.push_back(0);
    comb.m_Elems.insert(std::make_pair(1, elemB));
    comb.m_PartIdsInOrder.push_back(1);
    comb.m_Elems.insert(std::make_pair(2, elemC));
    comb.m_PartIdsInOrder.push_back(2);

    bool dumpInputGraphToFile = false;
    if (dumpInputGraphToFile)
    {
        std::ofstream stream("GetOpGraphForDfsMISOSramsToDrams.dot");
        SaveCombinationToDot(comb, stream, DetailLevel::High);
    }

    // Call function under test
    OpGraph combOpGraph = GetOpGraphForCombination(comb, graph);

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("GetOpGraphForDfsCombinationMergedBuffer Output.dot");
        SaveOpGraphToDot(combOpGraph, stream, DetailLevel::High);
    }

    // Output buffers of the sources A, B are both in SRAM
    // Therefore, the input DMA of the glue must be inserted
    // right before the op of its destination plan
    REQUIRE(combOpGraph.GetBuffers().size() == 4);
    REQUIRE(combOpGraph.GetBuffers()[0]->m_DebugTag == "OutputSramA");
    REQUIRE(combOpGraph.GetBuffers()[1]->m_DebugTag == "OutputSramB");
    REQUIRE(combOpGraph.GetBuffers()[2]->m_DebugTag == "Input0DramC");
    REQUIRE(combOpGraph.GetBuffers()[3]->m_DebugTag == "Input1DramC");
    REQUIRE(combOpGraph.GetOps().size() == 5);
    REQUIRE(combOpGraph.GetOps()[0]->m_DebugTag == "DummyA");
    REQUIRE(combOpGraph.GetOps()[1]->m_DebugTag == "DummyB");
    REQUIRE(combOpGraph.GetOps()[2]->m_DebugTag == "GlueAC_Dma");
    REQUIRE(combOpGraph.GetOps()[3]->m_DebugTag == "GlueBC_Dma");
    REQUIRE(combOpGraph.GetOps()[4]->m_DebugTag == "DummyC");

    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[0]).size() == 1);
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[0])[0].first->m_DebugTag == "GlueAC_Dma");
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[1]).size() == 1);
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[1])[0].first->m_DebugTag == "GlueBC_Dma");
    REQUIRE(combOpGraph.GetProducer(combOpGraph.GetBuffers()[2])->m_DebugTag == "GlueAC_Dma");
    REQUIRE(combOpGraph.GetProducer(combOpGraph.GetBuffers()[3])->m_DebugTag == "GlueBC_Dma");
}

// Manually creates a partial combination starting and ending in Sram.
// The topology is chosen to test cases including:
//      * Partial combinations starting and ending in Sram
//      * Glue containing multiple output DmaOps and direct connections to Dram buffers
// ( A ) -> g --> ( B )
//            \   (   )
//            |   (   )
//            \-> ( C )
//            \-> ( D - E )
//
//  D's input buffer is DRAM, but it can share glue with B, C because
//  it is not an output part.
TEST_CASE("Add shared glue between Dram and Sram", "[CombinerDFS]")
{
    DebuggableObject::ms_IdCounter = 0;    // Reset counter so we get deterministic results

    GraphOfParts graph;
    auto& parts       = graph.m_Parts;
    auto& connections = graph.m_Connections;

    auto pA = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pC = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pD = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pE = std::make_unique<MockPart>(graph.GeneratePartId());

    PartId partAId = pA->GetPartId();
    PartId partBId = pB->GetPartId();
    PartId partCId = pC->GetPartId();
    PartId partDId = pD->GetPartId();

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
    PartOutputSlot partDOutputSlot = { partD.GetPartId(), 0 };

    PartInputSlot partBInputSlot = { partB.GetPartId(), 0 };
    PartInputSlot partCInputSlot = { partC.GetPartId(), 0 };
    PartInputSlot partDInputSlot = { partD.GetPartId(), 0 };
    PartInputSlot partEInputSlot = { partE.GetPartId(), 0 };

    connections[partBInputSlot] = { partAOutputSlot };
    connections[partCInputSlot] = { partAOutputSlot };
    connections[partDInputSlot] = { partAOutputSlot };
    connections[partEInputSlot] = { partDOutputSlot };

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    // Plan A
    Plan planA;
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planA.m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputSramA";
    planA.m_OutputMappings                          = { { planA.m_OpGraph.GetBuffers()[0], partAOutputSlot } };

    // Plan B
    Plan planB;
    planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planB.m_OpGraph.GetBuffers().back()->m_DebugTag = "InputSramB";
    planB.m_InputMappings                           = { { planB.m_OpGraph.GetBuffers()[0], partBInputSlot } };

    // Plan C
    Plan planC;
    planC.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planC.m_OpGraph.GetBuffers().back()->m_DebugTag = "InputSramC";
    planC.m_InputMappings                           = { { planC.m_OpGraph.GetBuffers()[0], partCInputSlot } };

    // Plan D
    Plan planD;
    planD.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planD.m_OpGraph.GetBuffers().back()->m_DebugTag   = "InputDramD";
    planD.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Intermediate;
    planD.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planD.m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputSramD";
    planD.m_InputMappings                           = { { planD.m_OpGraph.GetBuffers()[0], partDInputSlot } };
    planD.m_OpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
    planD.m_OpGraph.GetOps()[0]->m_DebugTag = "DmaToSramD";
    planD.m_OpGraph.AddConsumer(planD.m_OpGraph.GetBuffers()[0], planD.m_OpGraph.GetOps()[0], 0);
    planD.m_OpGraph.SetProducer(planD.m_OpGraph.GetBuffers()[1], planD.m_OpGraph.GetOps()[0]);

    // Create Combination with all the plans and glues
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

    bool dumpInputGraphToFile = false;
    if (dumpInputGraphToFile)
    {
        std::ofstream stream("Add shared glue between dram and sram Input.dot");
        SaveCombinationToDot(combGlued, stream, DetailLevel::High);
    }

    // The output should be the ending glue of part A contains a Dram buffer, This Dram buffer is used for part D's input buffer
    // And Part B and C's sram buffers are dma'd from that merged buffer.
    auto elemA              = combGlued.m_Elems.find(partAId);
    EndingGlue* endingGlueA = elemA->second.m_EndingGlues.find(partAOutputSlot)->second.get();

    auto elemB                  = combGlued.m_Elems.find(partBId);
    StartingGlue* startingGlueB = elemB->second.m_StartingGlues.find(partBInputSlot)->second.get();
    REQUIRE(
        startingGlueB->m_ExternalConnections.m_BuffersToOps.find(endingGlueA->m_Graph.GetBuffers().back())->second ==
        startingGlueB->m_Graph.GetOp(0));

    auto elemC                  = combGlued.m_Elems.find(partCId);
    StartingGlue* startingGlueC = elemC->second.m_StartingGlues.find(partCInputSlot)->second.get();
    REQUIRE(
        startingGlueC->m_ExternalConnections.m_BuffersToOps.find(endingGlueA->m_Graph.GetBuffers().back())->second ==
        startingGlueC->m_Graph.GetOp(0));

    auto elemD                  = combGlued.m_Elems.find(partDId);
    StartingGlue* startingGlueD = elemD->second.m_StartingGlues.find(partDInputSlot)->second.get();
    REQUIRE(startingGlueD->m_ExternalConnections.m_ReplacementBuffers
                .find(elemD->second.m_Plan->m_OpGraph.GetBuffers().front())
                ->second == endingGlueA->m_Graph.GetBuffers().back());
}

// Manually creates a partial combination starting in Dram with NHWC and going into sram.
// Glue will be generated which adds a conversion to NHWCB through sram
TEST_CASE("GetOpGraphCombinationDramSramConversion", "[CombinerDFS]")
{
    DebuggableObject::ms_IdCounter = 0;    // Reset counter so we get deterministic results

    GraphOfParts graph;
    auto& parts       = graph.m_Parts;
    auto& connections = graph.m_Connections;

    auto pA = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB = std::make_unique<MockPart>(graph.GeneratePartId());

    BasePart& partA = *pA;
    BasePart& partB = *pB;

    parts.push_back(std::move(pA));
    parts.push_back(std::move(pB));

    PartOutputSlot partAOutputSlot = { partA.GetPartId(), 0 };
    PartInputSlot partBInputSlot   = { partB.GetPartId(), 0 };

    connections[partBInputSlot] = { partAOutputSlot };

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Plan planA;
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWC,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 0, 0, 0, 0 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planA.m_OpGraph.GetBuffers().back()->m_DebugTag = "StartingDramBuffer";
    planA.m_OutputMappings                          = { { planA.m_OpGraph.GetBuffers()[0], partAOutputSlot } };
    Buffer* startingBuffer                          = planA.m_OpGraph.GetBuffers().back();

    // Plan B
    Plan planB;
    planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 16 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planB.m_OpGraph.GetBuffers().back()->m_DebugTag = "FinalSramBuffer";
    planB.m_InputMappings                           = { { planB.m_OpGraph.GetBuffers()[0], partBInputSlot } };
    Buffer* finalBuffer                             = planB.m_OpGraph.GetBuffers().back();

    // Create Combination with all the plans and glues
    Combination combA(partA, std::move(planA), 0);
    Combination combB(partB, std::move(planB), 1);

    // Merge the combinations
    Combination comb = combA + combB;

    REQUIRE(combA.m_PartIdsInOrder[0] == 0);
    REQUIRE(combA.m_HeadOrderRank == 0);
    REQUIRE(combB.m_PartIdsInOrder[0] == 1);
    REQUIRE(combB.m_HeadOrderRank == 1);
    REQUIRE(comb.m_PartIdsInOrder[0] == 0);
    REQUIRE(comb.m_HeadOrderRank == 0);

    Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);

    std::vector<PartConnection> destPartEdge;

    // Part B and the edge that connects to its source Part A
    PartConnection edgeA2B = graph.GetConnectionsBetween(partA.GetPartId(), partB.GetPartId()).at(0);
    destPartEdge.push_back(edgeA2B);

    Combination combGlued = combiner.GluePartToCombinationSrcToDests(partA, comb, destPartEdge);

    // Access Glue Buffer and Ops and set an appropriate name for debugging purposes
    //auto elemAB = elemIt->second.m_Glues.find(edgeA2B.m_Destination);
    auto endingGlueA   = combGlued.m_Elems.find(partA.GetPartId())->second.m_EndingGlues[partAOutputSlot];
    auto startingGlueB = combGlued.m_Elems.find(partB.GetPartId())->second.m_StartingGlues[partBInputSlot];
    REQUIRE(endingGlueA->m_Graph.GetBuffers().size() == 0);
    REQUIRE(startingGlueB->m_Graph.GetBuffers().size() == 2);
    REQUIRE(startingGlueB->m_Graph.GetBuffers()[0]->m_Location == Location::Sram);
    REQUIRE(startingGlueB->m_Graph.GetBuffers()[0]->m_Format == CascadingBufferFormat::NHWCB);
    REQUIRE(startingGlueB->m_Graph.GetBuffers()[1]->m_Location == Location::Dram);
    REQUIRE(startingGlueB->m_Graph.GetBuffers()[1]->m_Format == CascadingBufferFormat::NHWCB);

    bool dumpInputGraphToFile = false;
    if (dumpInputGraphToFile)
    {
        std::ofstream stream("GetOpGraphCombinationDramSramConversion Input.dot");
        SaveCombinationToDot(combGlued, stream, DetailLevel::High);
    }

    // Call function under test
    OpGraph combOpGraph = GetOpGraphForCombination(combGlued, graph);

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("GetOpGraphCombinationDramSramConversion Output.dot");
        SaveOpGraphToDot(combOpGraph, stream, DetailLevel::High);
    }

    REQUIRE(combOpGraph.GetBuffers().size() == 4);
    REQUIRE(combOpGraph.GetBuffers()[0] == startingBuffer);
    REQUIRE(combOpGraph.GetBuffers()[1] == startingGlueB->m_Graph.GetBuffers()[0]);
    REQUIRE(combOpGraph.GetBuffers()[2] == startingGlueB->m_Graph.GetBuffers()[1]);
    REQUIRE(combOpGraph.GetBuffers()[3] == finalBuffer);

    REQUIRE(combOpGraph.GetOps().size() == 3);
    REQUIRE(combOpGraph.GetOps() == startingGlueB->m_Graph.GetOps());

    REQUIRE(combOpGraph.GetProducer(combOpGraph.GetBuffers()[0]) == nullptr);
    REQUIRE(combOpGraph.GetProducer(combOpGraph.GetBuffers()[1]) == combOpGraph.GetOps()[0]);
    REQUIRE(combOpGraph.GetProducer(combOpGraph.GetBuffers()[2]) == combOpGraph.GetOps()[1]);
    REQUIRE(combOpGraph.GetProducer(combOpGraph.GetBuffers()[3]) == combOpGraph.GetOps()[2]);

    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[0]) ==
            OpGraph::ConsumersList{ { combOpGraph.GetOps()[0], 0 } });
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[1]) ==
            OpGraph::ConsumersList{ { combOpGraph.GetOps()[1], 0 } });
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[2]) ==
            OpGraph::ConsumersList{ { combOpGraph.GetOps()[2], 0 } });
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[3]) == OpGraph::ConsumersList{});
}

// Manually creates a partial combination with two dram buffers being merged
TEST_CASE("GetOpGraphCombinationDramDramMerge", "[CombinerDFS]")
{
    DebuggableObject::ms_IdCounter = 0;    // Reset counter so we get deterministic results

    GraphOfParts graph;
    auto& parts       = graph.m_Parts;
    auto& connections = graph.m_Connections;

    auto pA = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB = std::make_unique<MockPart>(graph.GeneratePartId());

    BasePart& partA = *pA;
    BasePart& partB = *pB;

    parts.push_back(std::move(pA));
    parts.push_back(std::move(pB));

    PartOutputSlot partAOutputSlot = { partA.GetPartId(), 0 };
    PartInputSlot partBInputSlot   = { partB.GetPartId(), 0 };

    connections[partBInputSlot] = { partAOutputSlot };

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Plan planA;
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWC,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 0, 0, 0, 0 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planA.m_OpGraph.GetBuffers().back()->m_DebugTag = "StartingDramBuffer";
    planA.m_OutputMappings                          = { { planA.m_OpGraph.GetBuffers()[0], partAOutputSlot } };
    Buffer* startingBuffer                          = planA.m_OpGraph.GetBuffers().back();

    // Plan B
    Plan planB;
    planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWC,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 0, 0, 0, 0 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planB.m_OpGraph.GetBuffers().back()->m_DebugTag = "FinalDramBuffer";
    planB.m_InputMappings                           = { { planB.m_OpGraph.GetBuffers()[0], partBInputSlot } };
    Buffer* endingBuffer                            = planB.m_OpGraph.GetBuffers().back();

    // Create Combination with all the plans and glues
    Combination combA(partA, std::move(planA), 0);
    Combination combB(partB, std::move(planB), 1);

    // Merge the combinations
    Combination comb = combA + combB;

    REQUIRE(combA.m_PartIdsInOrder[0] == 0);
    REQUIRE(combA.m_HeadOrderRank == 0);
    REQUIRE(combB.m_PartIdsInOrder[0] == 1);
    REQUIRE(combB.m_HeadOrderRank == 1);
    REQUIRE(comb.m_PartIdsInOrder[0] == 0);
    REQUIRE(comb.m_HeadOrderRank == 0);

    Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);

    std::vector<PartConnection> destPartEdge;

    // Part B and the edge that connects to its source Part A
    PartConnection edgeA2B = graph.GetConnectionsBetween(partA.GetPartId(), partB.GetPartId()).at(0);
    destPartEdge.push_back(edgeA2B);

    Combination combGlued = combiner.GluePartToCombinationSrcToDests(partA, comb, destPartEdge);

    bool dumpInputGraphToFile = false;
    if (dumpInputGraphToFile)
    {
        std::ofstream stream("GetOpGraphCombinationDramDramMerge Input.dot");
        SaveCombinationToDot(combGlued, stream, DetailLevel::High);
    }

    // Call function under test
    OpGraph combOpGraph = GetOpGraphForCombination(combGlued, graph);

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("GetOpGraphCombinationDramDramMerge Output.dot");
        SaveOpGraphToDot(combOpGraph, stream, DetailLevel::High);
    }

    REQUIRE(combOpGraph.GetBuffers().size() == 1);
    // The buffer should be a new merged buffer
    REQUIRE((combOpGraph.GetBuffers()[0] != startingBuffer && combOpGraph.GetBuffers()[0] != endingBuffer &&
             combOpGraph.GetBuffers()[0]->m_DebugTag.find("Merged") != std::string::npos));

    REQUIRE(combOpGraph.GetOps().size() == 0);
}

TEST_CASE("GetOpGraphForDfsCombinationMergedBuffer", "[CombinerDFS]")
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
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 0, QuantizationInfo()));
    planA.m_OpGraph.GetBuffers().back()->m_DebugTag = "PlanA_Buffer0";
    planA.m_OutputMappings                          = { { planA.m_OpGraph.GetBuffers()[0], partAOutputSlot0 } };

    // Plan B
    Plan planB;
    planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planB.m_OpGraph.GetBuffers().back()->m_DebugTag = "PlanB_Buffer0";
    planB.m_InputMappings                           = { { planB.m_OpGraph.GetBuffers()[0], partBInputSlot0 } };
    planB.m_OutputMappings                          = { { planB.m_OpGraph.GetBuffers()[0], partBOutputSlot0 } };

    // Plan C
    Plan planC;
    planC.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planC.m_OpGraph.GetBuffers().back()->m_DebugTag = "PlanC_Buffer0";
    planC.m_InputMappings                           = { { planC.m_OpGraph.GetBuffers()[0], partCInputSlot0 } };

    auto endingGlueA = std::make_shared<EndingGlue>();

    auto startingGlueB = std::make_shared<StartingGlue>();
    startingGlueB->m_ExternalConnections.m_ReplacementBuffers.insert(
        { planB.m_OpGraph.GetBuffers().front(), planA.m_OpGraph.GetBuffers().back() });

    auto endingGlueB = std::make_shared<EndingGlue>();

    auto startingGlueC = std::make_shared<StartingGlue>();
    startingGlueC->m_Graph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
    startingGlueC->m_Graph.GetOps()[0]->m_DebugTag = "GlueBC_Dma";
    startingGlueC->m_ExternalConnections.m_BuffersToOps.insert(
        { planB.m_OpGraph.GetBuffers().back(), startingGlueC->m_Graph.GetOp(0) });
    startingGlueC->m_ExternalConnections.m_OpsToBuffers.insert(
        { startingGlueC->m_Graph.GetOp(0), planC.m_OpGraph.GetBuffers().front() });

    // Create Combination with all the plans and glues
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
        SaveCombinationToDot(comb, stream, DetailLevel::High);
    }

    // Call function under test
    OpGraph combOpGraph = GetOpGraphForCombination(comb, graph);

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("GetOpGraphForDfsCombinationMergedBuffer Output.dot");
        SaveOpGraphToDot(combOpGraph, stream, DetailLevel::High);
    }

    // PlanA and plan B both have one buffer and no operation.
    // Their buffers are merged and only planA_Buffer0 is expected
    // to be added to the combOpGraph

    REQUIRE(combOpGraph.GetBuffers().size() == 2);
    REQUIRE(combOpGraph.GetBuffers()[0]->m_DebugTag == "PlanA_Buffer0");
    REQUIRE(combOpGraph.GetBuffers()[1]->m_DebugTag == "PlanC_Buffer0");
}

/// Manually creates a Combination and then converts it to an OpGraph using GetOpGraphForCombination, and checking
/// the resulting graph structure is correct.
/// The topology of the Combination is chosen to test cases including:
///   * Plans without any inputs (A)
///   * Plans without any outputs (F, G)
///   * Two plans being connected via a glue (A -> B)
///   * Two plans being connected without a glue (B -> DE)
///   * A part having two plans using its output, each with a different glue (DE -> F/G)
///   * Two plans being connected by two different glues (for two different connections) (DE -> G)
///   * A replacement buffer in the ending glue (A)
///
///  ( A ) -> g -> ( B ) -> ( D ) ---> g -> ( F )
///                       \  (   ) \'
///                        | (   )  \-> g -> (   )
///                        | (   )           ( G )
///                        \-( E ) -->  g -> (   )
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

    PartInputSlot partDEInputSlot0   = { partDEId, 0 };
    PartInputSlot partDEInputSlot1   = { partDEId, 1 };
    PartOutputSlot partDEOutputSlot0 = { partDEId, 0 };
    PartOutputSlot partDEOutputSlot1 = { partDEId, 1 };

    PartInputSlot partFInputSlot0 = { partFId, 0 };

    PartInputSlot partGInputSlot0 = { partGId, 0 };
    PartInputSlot partGInputSlot1 = { partGId, 1 };

    connections[partBInputSlot0]  = partAOutputSlot0;
    connections[partDEInputSlot0] = partBOutputSlot0;
    connections[partDEInputSlot1] = partBOutputSlot0;
    connections[partFInputSlot0]  = partDEOutputSlot0;
    connections[partGInputSlot0]  = partDEOutputSlot0;
    connections[partGInputSlot1]  = partDEOutputSlot1;

    Plan planA;
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 0, QuantizationInfo()));
    planA.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Input;
    planA.m_OpGraph.GetBuffers().back()->m_DebugTag   = "InputDram";
    planA.m_OutputMappings                            = { { planA.m_OpGraph.GetBuffers()[0], partAOutputSlot0 } };

    // Part consisting of node B
    Plan planB;
    planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planB.m_OpGraph.GetBuffers().back()->m_DebugTag = "InputSram1";
    planB.m_InputMappings                           = { { planB.m_OpGraph.GetBuffers()[0], partBInputSlot0 } };
    planB.m_OutputMappings                          = { { planB.m_OpGraph.GetBuffers()[0], partBOutputSlot0 } };

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
    planF.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Output;
    planF.m_OpGraph.GetBuffers().back()->m_DebugTag   = "OutputDram1";
    planF.m_InputMappings                             = { { planF.m_OpGraph.GetBuffers()[0], partFInputSlot0 } };

    // Part consisting of node G
    Plan planG;
    planG.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 0, QuantizationInfo()));
    planG.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Output;
    planG.m_OpGraph.GetBuffers().back()->m_DebugTag   = "OutputDram2";
    planG.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 0, QuantizationInfo()));
    planG.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Output;
    planG.m_OpGraph.GetBuffers().back()->m_DebugTag   = "OutputDram3";
    planG.m_InputMappings                             = { { planG.m_OpGraph.GetBuffers()[0], partGInputSlot0 },
                              { planG.m_OpGraph.GetBuffers()[1], partGInputSlot1 } };

    // The end glueing of A is empty. But the starting glue of B has the connections.
    auto startingGlueA = std::make_shared<StartingGlue>();
    auto endingGlueA   = std::make_shared<EndingGlue>();
    endingGlueA->m_Graph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                            TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                            TraversalOrder::Xyz, 0, QuantizationInfo()));
    endingGlueA->m_Graph.GetBuffers()[0]->m_DebugTag = "ReplacementBuffer";
    endingGlueA->m_ExternalConnections.m_ReplacementBuffers.insert(
        { planA.m_OpGraph.GetBuffers()[0], endingGlueA->m_Graph.GetBuffers()[0] });

    auto startingGlueB = std::make_shared<StartingGlue>();
    startingGlueB->m_Graph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
    startingGlueB->m_Graph.GetOps()[0]->m_DebugTag = "InputDma";
    startingGlueB->m_ExternalConnections.m_BuffersToOps.insert(
        { endingGlueA->m_Graph.GetBuffers().back(), startingGlueB->m_Graph.GetOps()[0] });
    startingGlueB->m_ExternalConnections.m_OpsToBuffers.insert(
        { startingGlueB->m_Graph.GetOps()[0], planB.m_OpGraph.GetBuffers()[0] });

    auto endingGlueB = std::make_shared<EndingGlue>();

    auto startingGlueDE0 = std::make_shared<StartingGlue>();
    startingGlueDE0->m_ExternalConnections.m_ReplacementBuffers.insert(
        { planDE.m_OpGraph.GetBuffers()[0], planB.m_OpGraph.GetBuffers()[0] });

    auto startingGlueDE1 = std::make_shared<StartingGlue>();
    startingGlueDE1->m_ExternalConnections.m_ReplacementBuffers.insert(
        { planDE.m_OpGraph.GetBuffers()[2], planB.m_OpGraph.GetBuffers()[0] });

    auto endingGlueD = std::make_shared<EndingGlue>();
    endingGlueD->m_Graph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
    endingGlueD->m_Graph.GetOps()[0]->m_DebugTag = "OutputDma1";
    endingGlueD->m_ExternalConnections.m_BuffersToOps.insert(
        { planDE.m_OpGraph.GetBuffers()[1], endingGlueD->m_Graph.GetOps()[0] });
    endingGlueD->m_Graph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
    endingGlueD->m_Graph.GetOps()[1]->m_DebugTag = "OutputDma2";
    endingGlueD->m_ExternalConnections.m_BuffersToOps.insert(
        { planDE.m_OpGraph.GetBuffers()[1], endingGlueD->m_Graph.GetOps()[1] });
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
        { planDE.m_OpGraph.GetBuffers()[3], endingGlueE->m_Graph.GetOps()[0] });
    auto startingGluefromEtoG = std::make_shared<StartingGlue>();
    startingGluefromEtoG->m_ExternalConnections.m_OpsToBuffers.insert(
        { endingGlueE->m_Graph.GetOps()[0], planG.m_OpGraph.GetBuffers()[1] });

    auto endingGlueF = std::make_shared<EndingGlue>();
    auto endingGlueG = std::make_shared<EndingGlue>();

    // Create Combination with all the plans and glues
    Combination comb;

    Elem elemA;
    elemA.m_Plan        = std::make_shared<Plan>(std::move(planA));
    elemA.m_EndingGlues = { { partAOutputSlot0, endingGlueA } };

    Elem elemB;
    elemB.m_Plan          = std::make_shared<Plan>(std::move(planB));
    elemB.m_StartingGlues = { { partBInputSlot0, startingGlueB } };
    elemB.m_EndingGlues   = { { partBOutputSlot0, endingGlueB } };

    Elem elemDE;
    elemDE.m_Plan          = std::make_shared<Plan>(std::move(planDE));
    elemDE.m_StartingGlues = { { partDEInputSlot0, startingGlueDE0 }, { partDEInputSlot1, startingGlueDE1 } };
    elemDE.m_EndingGlues   = { { partDEOutputSlot0, endingGlueD }, { partDEOutputSlot1, endingGlueE } };

    Elem elemF;
    elemF.m_Plan          = std::make_shared<Plan>(std::move(planF));
    elemF.m_StartingGlues = { { partFInputSlot0, startingGlueF } };

    Elem elemG;
    elemG.m_Plan          = std::make_shared<Plan>(std::move(planG));
    elemG.m_StartingGlues = { { partGInputSlot0, startingGluefromDtoG }, { partGInputSlot1, startingGluefromEtoG } };

    comb.m_Elems.insert(std::make_pair(partAId, elemA));
    comb.m_PartIdsInOrder.push_back(partAId);
    comb.m_Elems.insert(std::make_pair(partBId, elemB));
    comb.m_PartIdsInOrder.push_back(partBId);
    comb.m_Elems.insert(std::make_pair(partDEId, elemDE));
    comb.m_PartIdsInOrder.push_back(partDEId);
    comb.m_Elems.insert(std::make_pair(partFId, elemF));
    comb.m_PartIdsInOrder.push_back(partFId);
    comb.m_Elems.insert(std::make_pair(partGId, elemG));
    comb.m_PartIdsInOrder.push_back(partGId);

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("GetOpGraphForCombination Input.dot");
        SaveCombinationToDot(comb, stream, DetailLevel::High);
    }

    // Call function under test
    OpGraph combOpGraph = GetOpGraphForCombination(comb, graph);

    // For easier debugging of this test (and so that you can see the pretty graph!), dump the output to a file
    if (dumpToFile)
    {
        std::ofstream stream("GetOpGraphForCombination Output.dot");
        SaveOpGraphToDot(combOpGraph, stream, DetailLevel::High);
    }

    // Check the resulting OpGraph is correct
    REQUIRE(combOpGraph.GetBuffers().size() == 7);
    REQUIRE(combOpGraph.GetBuffers()[0]->m_DebugTag == "ReplacementBuffer");
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
    Plan planC;

    Combination combA(partA, std::move(planA), 0);
    Combination combB(partB, std::move(planB), 1);
    Combination combC(partC, std::move(planC), 2);

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
}

TEST_CASE("Combination AddGlue", "[CombinerDFS]")
{
    // Create graph:
    //
    //  A - B
    //
    GraphOfParts graph;

    auto& parts = graph.m_Parts;

    auto pA               = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB               = std::make_unique<MockPart>(graph.GeneratePartId());
    const BasePart& partA = *pA;
    const BasePart& partB = *pB;
    PartId partAId        = partA.GetPartId();
    PartId partBId        = partB.GetPartId();
    parts.push_back(std::move(pA));
    parts.push_back(std::move(pB));

    PartOutputSlot partAOutputSlot0 = { partAId, 0 };

    PartInputSlot partBInputSlot0 = { partBId, 0 };

    Plan planA;
    Plan planB;

    Combination combA(partA, std::move(planA), 0);
    Combination combB(partB, std::move(planB), 1);

    REQUIRE(combA.m_Elems.size() == 1);
    REQUIRE(combB.m_Elems.size() == 1);

    EndingGlue endingGlueA;
    combA.AddEndingGlue(std::move(endingGlueA), partAOutputSlot0);
    REQUIRE(combA.m_Elems.find(partAOutputSlot0.m_PartId)->second.m_EndingGlues.size() == 1);

    StartingGlue startingGlueB;
    combB.SetStartingGlue(std::move(startingGlueB), partBInputSlot0);
    REQUIRE(combB.m_Elems.find(partBInputSlot0.m_PartId)->second.m_StartingGlues.size() == 1);

    Combination comb = combA + combB;
    // Adding combinations adds their glues.
    REQUIRE(comb.m_Elems.find(partAOutputSlot0.m_PartId)->second.m_EndingGlues.size() == 1);
    REQUIRE(comb.m_Elems.find(partBInputSlot0.m_PartId)->second.m_StartingGlues.size() == 1);
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
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
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

    MockCombiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);

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
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);

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
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);

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
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    GraphOfParts gOfParts;

    Plan planA;
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planA.m_OutputMappings = { { planA.m_OpGraph.GetBuffers()[0], partAOutputSlot } };

    Plan planB;
    planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 16, 16 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planB.m_InputMappings  = { { planB.m_OpGraph.GetBuffers()[0], partBInputSlot } };
    planB.m_OutputMappings = { { planB.m_OpGraph.GetBuffers()[0], partBOutputSlot } };

    Plan planC;
    planC.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 16, 16 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planC.m_InputMappings  = { { planC.m_OpGraph.GetBuffers()[0], partCInputSlot } };
    planC.m_OutputMappings = { { planC.m_OpGraph.GetBuffers()[0], partCOutputSlot } };

    Plan planD;
    planD.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 16, 16, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planD.m_InputMappings  = { { planD.m_OpGraph.GetBuffers()[0], partDInputSlot } };
    planD.m_OutputMappings = { { planD.m_OpGraph.GetBuffers()[0], partDOutputSlot } };

    Plan planE;
    planE.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 16, 16, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planE.m_InputMappings = { { planD.m_OpGraph.GetBuffers()[0], partEInputSlot } };

    Combiner combiner(gOfParts, hwCaps, compOpt, estOpt, debuggingContext);

    bool isSorted = combiner.TopologicalSortParts();
    REQUIRE(isSorted == true);

    Combination combA(partA, std::move(planA), 0);
    Combination combB(partB, std::move(planB), 1);
    Combination combC(partC, std::move(planC), 2);
    Combination combD(partD, std::move(planD), 3);
    Combination combE(partE, std::move(planE), 4);

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

    Combination combA(partA, std::move(planA), 0);
    Combination combB(partB, std::move(planB), 1);
    Combination combC(partC, std::move(planC), 2);

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
        REQUIRE(comb.m_Elems.at(part.GetPartId()).m_EndingGlues.size() == 0);
        REQUIRE(comb.m_Elems.at(part.GetPartId()).m_StartingGlues.size() == 0);
    }

    Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);

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

    OpGraph opGraph = GetOpGraphForCombination(combGlued, graph);

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("GluePartToCombinationBranch0.dot");
        SaveOpGraphToDot(opGraph, stream, DetailLevel::High);
    }

    // Part A should have 1 ending glue containing a dma and a dram buffer
    {
        auto elemIt = combGlued.m_Elems.find(partAId);
        REQUIRE(elemIt != combGlued.m_Elems.end());
        REQUIRE(elemIt->second.m_StartingGlues.size() == 0);
        REQUIRE(elemIt->second.m_EndingGlues.size() == 1);
        REQUIRE(elemIt->second.m_EndingGlues.begin()->second->m_Graph.GetBuffers().size() == 1);
        REQUIRE(elemIt->second.m_EndingGlues.begin()->second->m_Graph.GetBuffers()[0]->m_Location == Location::Dram);
        REQUIRE(elemIt->second.m_EndingGlues.begin()->second->m_Graph.GetBuffers()[0]->m_Format ==
                CascadingBufferFormat::FCAF_DEEP);
        REQUIRE(elemIt->second.m_EndingGlues.begin()->second->m_Graph.GetOps().size() == 1);
    }
    // Part B and C should have 1 starting glue containing just 1 dma op each.
    {
        auto elemIt = combGlued.m_Elems.find(partBId);
        REQUIRE(elemIt != combGlued.m_Elems.end());
        REQUIRE(elemIt->second.m_StartingGlues.size() == 1);
        REQUIRE(elemIt->second.m_EndingGlues.size() == 0);
        REQUIRE(elemIt->second.m_StartingGlues.begin()->second->m_Graph.GetBuffers().size() == 0);
        REQUIRE(elemIt->second.m_StartingGlues.begin()->second->m_Graph.GetOps().size() == 1);
    }
    {
        auto elemIt = combGlued.m_Elems.find(partCId);
        REQUIRE(elemIt != combGlued.m_Elems.end());
        REQUIRE(elemIt->second.m_StartingGlues.size() == 1);
        REQUIRE(elemIt->second.m_EndingGlues.size() == 0);
        REQUIRE(elemIt->second.m_StartingGlues.begin()->second->m_Graph.GetBuffers().size() == 0);
        REQUIRE(elemIt->second.m_StartingGlues.begin()->second->m_Graph.GetOps().size() == 1);
    }
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
    //
    //  D is an output node on DRAM and cannot share
    //  glue with B, C
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
    auto dummyOpD         = std::make_unique<DummyOp>();
    dummyOpD->m_DebugTag  = "DummyD";
    planD.m_OpGraph.AddOp(std::move(dummyOpD));
    planD.m_OpGraph.AddConsumer(planD.m_OpGraph.GetBuffers().back(), planD.m_OpGraph.GetOps().back(), 0);

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

    // There is no glue
    for (PartId i = 0; i < graph.m_Parts.size(); ++i)
    {
        const BasePart& part = graph.GetPart(i);
        REQUIRE(comb.m_Elems.at(part.GetPartId()).m_EndingGlues.size() == 0);
        REQUIRE(comb.m_Elems.at(part.GetPartId()).m_StartingGlues.size() == 0);
    }

    Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);

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

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("GluePartToCombinationBranch1.dot");
        SaveCombinationToDot(combGlued, stream, DetailLevel::High);
    }

    REQUIRE(combGlued.m_Elems.size() == 4);

    // Elem Part A's glue should have three elements
    auto elemA = combGlued.m_Elems.find(partAId);
    REQUIRE(elemA != combGlued.m_Elems.end());
    REQUIRE(elemA->second.m_EndingGlues.size() == 1);

    auto endingGlueA = elemA->second.m_EndingGlues[partAOutputSlot];
    REQUIRE(endingGlueA->m_Graph.GetBuffers().size() == 1);
    REQUIRE(endingGlueA->m_Graph.GetOps().size() == 2);
    REQUIRE(endingGlueA->m_Graph.GetProducer(endingGlueA->m_Graph.GetBuffers().front()) ==
            endingGlueA->m_Graph.GetOp(1));
    const auto& planABuffers = combGlued.m_Elems.find(partAId)->second.m_Plan->m_OpGraph.GetBuffers();
    REQUIRE(endingGlueA->m_ExternalConnections.m_BuffersToOps.find(planABuffers.back())->second ==
            endingGlueA->m_Graph.GetOp(0));

    auto elemB               = combGlued.m_Elems.find(partBId);
    const auto& planBBuffers = elemB->second.m_Plan->m_OpGraph.GetBuffers();
    auto startingGlueB       = elemB->second.m_StartingGlues[partBInputSlot];
    REQUIRE(startingGlueB->m_Graph.GetBuffers().size() == 0);
    REQUIRE(startingGlueB->m_Graph.GetOps().size() == 1);
    REQUIRE(
        startingGlueB->m_ExternalConnections.m_BuffersToOps.find(endingGlueA->m_Graph.GetBuffers().back())->second ==
        startingGlueB->m_Graph.GetOp(0));
    REQUIRE(startingGlueB->m_ExternalConnections.m_OpsToBuffers.find(startingGlueB->m_Graph.GetOp(0))->second ==
            planBBuffers.front());

    auto elemC               = combGlued.m_Elems.find(partCId);
    const auto& planCBuffers = elemC->second.m_Plan->m_OpGraph.GetBuffers();
    auto startingGlueC       = elemC->second.m_StartingGlues[partCInputSlot];
    REQUIRE(startingGlueC->m_Graph.GetBuffers().size() == 0);
    REQUIRE(startingGlueC->m_Graph.GetOps().size() == 1);
    REQUIRE(
        startingGlueC->m_ExternalConnections.m_BuffersToOps.find(endingGlueA->m_Graph.GetBuffers().back())->second ==
        startingGlueC->m_Graph.GetOp(0));
    REQUIRE(startingGlueC->m_ExternalConnections.m_OpsToBuffers.find(startingGlueC->m_Graph.GetOp(0))->second ==
            planCBuffers.front());

    auto elemD               = combGlued.m_Elems.find(partDId);
    const auto& planDBuffers = elemD->second.m_Plan->m_OpGraph.GetBuffers();
    auto startingGlueD       = elemD->second.m_StartingGlues[partDInputSlot];
    REQUIRE(startingGlueD->m_ExternalConnections.m_OpsToBuffers.find(endingGlueA->m_Graph.GetOp(0))->second ==
            planDBuffers.back());
}

TEST_CASE("GluePartToCombinationBranch2", "[CombinerDFS]")
{
    // Create graph:
    //
    //
    //   - - > C
    //  |
    //  A - -> B
    //  |
    //   -- >  D --> E
    //
    //  D is not an output node, but its buffer  in DRAM is in NHWC format
    //  and cannot share glue with B, C
    //  Note that E is merely a "dummy" part to make D an non-output
    //  part.
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

    parts.push_back(std::move(pA));
    parts.push_back(std::move(pB));
    parts.push_back(std::move(pC));
    parts.push_back(std::move(pD));
    parts.push_back(std::move(pE));

    PartOutputSlot partAOutputSlot = { partA.GetPartId(), 0 };
    PartOutputSlot partDOutputSlot = { partD.GetPartId(), 0 };

    PartInputSlot partBInputSlot = { partB.GetPartId(), 0 };
    PartInputSlot partCInputSlot = { partC.GetPartId(), 0 };
    PartInputSlot partDInputSlot = { partD.GetPartId(), 0 };
    PartInputSlot partEInputSlot = { partE.GetPartId(), 0 };

    connections[partBInputSlot] = { partAOutputSlot };
    connections[partCInputSlot] = { partAOutputSlot };
    connections[partDInputSlot] = { partAOutputSlot };
    connections[partEInputSlot] = { partDOutputSlot };

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
    planD.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWC,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planD.m_InputMappings = { { planD.m_OpGraph.GetBuffers()[0], partDInputSlot } };

    Plan planE;
    planE.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planE.m_InputMappings = { { planE.m_OpGraph.GetBuffers()[0], partDInputSlot } };

    Combination combA(partA, std::move(planA), 0);
    Combination combB(partB, std::move(planB), 1);
    Combination combC(partC, std::move(planC), 2);
    Combination combD(partD, std::move(planD), 3);
    Combination combE(partE, std::move(planE), 4);

    // Merge the combinations
    Combination comb = combE + combB + combD + combC + combA;

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
        REQUIRE(comb.m_Elems.at(part.GetPartId()).m_EndingGlues.size() == 0);
        REQUIRE(comb.m_Elems.at(part.GetPartId()).m_StartingGlues.size() == 0);
    }

    Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);

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

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("GluePartToCombinationBranch2.dot");
        SaveCombinationToDot(combGlued, stream, DetailLevel::High);
    }

    // One glue shared by A-B, A-C (SRAM - SRAM)
    // The glue has (1) 1 x input DMA (2) DRAM buffer (3) 2 x ouput DMA.
    // A-D uses a separate glue (SRAM - DRAM) that has one DMA.
    // Note part D's input buffer in DRAM is NHWC so that it
    // cannot share glue with others.
    REQUIRE(combGlued.m_Elems.size() == 5);

    auto elemA              = combGlued.m_Elems.find(partAId);
    EndingGlue* endingGlueA = elemA->second.m_EndingGlues.find(partAOutputSlot)->second.get();
    OpGraph& opGraphA       = elemA->second.m_Plan->m_OpGraph;
    REQUIRE(endingGlueA->m_Graph.GetOps().size() == 4);
    REQUIRE(endingGlueA->m_Graph.GetBuffers().size() == 3);
    REQUIRE(endingGlueA->m_ExternalConnections.m_BuffersToOps ==
            std::multimap<Buffer*, Op*>{ { opGraphA.GetBuffers()[0], endingGlueA->m_Graph.GetOp(0) },
                                         { opGraphA.GetBuffers()[0], endingGlueA->m_Graph.GetOps().back() } });

    auto elemB                  = combGlued.m_Elems.find(partBId);
    StartingGlue* startingGlueB = elemB->second.m_StartingGlues.find(partBInputSlot)->second.get();
    OpGraph& opGraphB           = elemB->second.m_Plan->m_OpGraph;
    REQUIRE(startingGlueB->m_Graph.GetOps().size() == 1);
    REQUIRE(startingGlueB->m_Graph.GetBuffers().size() == 0);
    REQUIRE(startingGlueB->m_ExternalConnections.m_BuffersToOps.find(endingGlueA->m_Graph.GetBuffers()[2])->second ==
            startingGlueB->m_Graph.GetOp(0));
    REQUIRE(startingGlueB->m_ExternalConnections.m_OpsToBuffers.find(startingGlueB->m_Graph.GetOp(0))->second ==
            opGraphB.GetBuffers()[0]);

    auto elemC                  = combGlued.m_Elems.find(partCId);
    StartingGlue* startingGlueC = elemC->second.m_StartingGlues.find(partCInputSlot)->second.get();
    OpGraph& opGraphC           = elemC->second.m_Plan->m_OpGraph;
    REQUIRE(startingGlueC->m_Graph.GetOps().size() == 1);
    REQUIRE(startingGlueC->m_Graph.GetBuffers().size() == 0);
    REQUIRE(startingGlueC->m_ExternalConnections.m_BuffersToOps.find(endingGlueA->m_Graph.GetBuffers()[2])->second ==
            startingGlueC->m_Graph.GetOp(0));
    REQUIRE(startingGlueC->m_ExternalConnections.m_OpsToBuffers.find(startingGlueC->m_Graph.GetOp(0))->second ==
            opGraphC.GetBuffers()[0]);

    auto elemD                  = combGlued.m_Elems.find(partDId);
    StartingGlue* startingGlueD = elemD->second.m_StartingGlues.find(partDInputSlot)->second.get();
    OpGraph& opGraphD           = elemD->second.m_Plan->m_OpGraph;
    REQUIRE(startingGlueD->m_Graph.GetOps().size() == 0);
    REQUIRE(startingGlueD->m_Graph.GetBuffers().size() == 0);
    REQUIRE(startingGlueD->m_ExternalConnections.m_OpsToBuffers.find(endingGlueA->m_Graph.GetOp(2))->second ==
            opGraphD.GetBuffers()[0]);
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

    auto mockBuffer =
        std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 32, 16, 1024 },
                                 TensorShape{ 1, 32, 16, 1024 }, TraversalOrder::Xyz, ifmSize, QuantizationInfo());
    mockBuffer.get()->m_Offset = 0;

    Plan planA;
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 32, 16, 1024 }, TensorShape{ 1, 32, 16, 1024 },
                                                       TraversalOrder::Xyz, ifmSize, QuantizationInfo()));
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 32, 16, 1024 }, TensorShape{ 1, 4, 16, 1024 },
                                                       TraversalOrder::Xyz, ofmSize, QuantizationInfo()));

    planA.m_OpGraph.AddOp(std::make_unique<MceOp>(
        MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct, BlockConfig{ 8u, 8u }, TensorShape{ 1, 32, 16, 1024 },
        TensorShape{ 1, 4, 16, 1024 }, TensorShape{ 1, 32, 16, 1024 }, TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));
    planA.m_OpGraph.SetProducer(planA.m_OpGraph.GetBuffers()[1], planA.m_OpGraph.GetOps()[0]);
    planA.m_InputMappings  = { { planA.m_OpGraph.GetBuffers()[0], partAInputSlot } };
    planA.m_OutputMappings = { { planA.m_OpGraph.GetBuffers()[1], partAOutputSlot } };

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    ethosn::support_library::impl::NumMemoryStripes numMemoryStripes;
    const std::set<uint32_t> operationIds = { 0 };

    Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);

    SectionContext context = { SramAllocator(hwCaps.GetTotalSramSize() / hwCaps.GetNumberOfSrams()), {}, {} };

    // SRAM has enough space for ofm and the plan does not have a PLE kernel
    SectionContext context1 = context;
    REQUIRE(combiner.IsPlanAllocated(context1, planA, mockBuffer.get(), StatsType::ContinueSection) == true);
    REQUIRE(context1.pleOps.size() == 0);

    // Adding a passthrough PLE kernel to the plan
    // The PleKernelId is expected to be PASSTHROUGH_8x8_2
    auto op = std::make_unique<PleOp>(ethosn::command_stream::PleOperation::PASSTHROUGH, BlockConfig{ 8u, 8u }, 1,
                                      std::vector<TensorShape>{ TensorShape{ 1, 4, 16, 1024 } },
                                      TensorShape{ 1, 4, 16, 1024 }, DataType::UINT8_QUANTIZED, true);

    numMemoryStripes.m_Output = 1;
    auto outBufferAndPleOp =
        AddPleToOpGraph(planA.m_OpGraph, TensorShape{ 1, 4, 16, 1024 }, numMemoryStripes, std::move(op),
                        TensorShape{ 1, 4, 16, 1024 }, QuantizationInfo(), DataType::UINT8_QUANTIZED, operationIds);

    Op* maybePleOp   = planA.m_OpGraph.GetOp(1);
    const bool isPle = IsPleOp(maybePleOp);
    REQUIRE(isPle);

    PleOp* actualPleOp = static_cast<PleOp*>(maybePleOp);

    // With a PLE kernel, the plan can still fit into SRAM and there is a need to Load the Kernel
    SectionContext context2 = context;
    REQUIRE(combiner.IsPlanAllocated(context2, planA, mockBuffer.get(), StatsType::ContinueSection) == true);
    REQUIRE(context2.pleOps.size() == 1);
    REQUIRE(actualPleOp->m_LoadKernel == true);

    // PLE kernel used previously has different block height
    // The plan is expected to be fit into SRAM and there is a need to Load the Kernel
    SectionContext context3 = context;
    PleKernelId pleKernel1  = PleKernelId::PASSTHROUGH_8X16_1;
    context3.pleOps         = { { pleKernel1, 0 } };
    REQUIRE(combiner.IsPlanAllocated(context3, planA, mockBuffer.get(), StatsType::ContinueSection) == true);
    REQUIRE(context3.pleOps.size() == 2);
    REQUIRE(actualPleOp->m_LoadKernel == true);

    // PLE kernel passthrough is already used previously in the same
    // section, the plan is expected to be fit into SRAM and no need to Load the Kernel
    SectionContext context4 = context;
    PleKernelId pleKernel2  = PleKernelId::PASSTHROUGH_8X8_2;
    context4.pleOps         = { { pleKernel2, 0 } };
    REQUIRE(combiner.IsPlanAllocated(context4, planA, mockBuffer.get(), StatsType::ContinueSection) == true);
    REQUIRE(context4.pleOps.size() == 1);
    REQUIRE(actualPleOp->m_LoadKernel == false);

    SectionContext context5 = context;
    // Allocate memory where the plan and the allocated memory exceeds the SRAM Size
    uint32_t planSize          = ofmSize + planA.m_OpGraph.GetBuffers()[2]->m_SizeInBytes + hwCaps.GetMaxPleSize();
    uint32_t remainingSramSize = hwCaps.GetTotalSramSize() - planSize;
    context5.alloc.Allocate(0, ((remainingSramSize + hwCaps.GetNumberOfSrams()) / hwCaps.GetNumberOfSrams()),
                            AllocationPreference::Start);
    REQUIRE(combiner.IsPlanAllocated(context5, planA, mockBuffer.get(), StatsType::ContinueSection) == false);
    REQUIRE(context5.pleOps.size() == 0);
    REQUIRE(actualPleOp->m_LoadKernel == true);

    ETHOSN_UNUSED(outBufferAndPleOp);
}

TEST_CASE("SramAllocationForSinglePartSection", "[CombinerDFS]")
{
    GIVEN("A Graph of one part where its corresponding plan fits into a single section")
    {
        GraphOfParts graph;

        auto& parts = graph.m_Parts;

        auto pA = std::make_unique<MockPart>(graph.GeneratePartId());

        const BasePart& partA = *pA;

        parts.push_back(std::move(pA));

        PartInputSlot partAInputSlot   = { partA.GetPartId(), 0 };
        PartOutputSlot partAOutputSlot = { partA.GetPartId(), 0 };

        ethosn::support_library::impl::NumMemoryStripes numMemoryStripes;
        const std::set<uint32_t> operationIds = { 0 };

        const CompilationOptions compOpt;
        const EstimationOptions estOpt;
        const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
        const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

        Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);
        SectionContext context     = { SramAllocator(hwCaps.GetTotalSramSize() / hwCaps.GetNumberOfSrams()), {}, {} };
        uint32_t currentSramOffset = 0;

        Plan planA;

        const uint32_t inputBufferSize  = 512;
        const uint32_t outputBufferSize = 512;

        planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                           TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 8, 8, 8 },
                                                           TraversalOrder::Xyz, inputBufferSize, QuantizationInfo()));
        planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::PleInputSram, CascadingBufferFormat::NHWCB,
                                                           TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 8, 8, 8 },
                                                           TraversalOrder::Xyz, outputBufferSize, QuantizationInfo()));

        planA.m_OpGraph.AddOp(std::make_unique<MceOp>(
            MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct, BlockConfig{ 8u, 8u }, TensorShape{ 1, 8, 8, 8 },
            TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 8, 8, 8 }, TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));

        planA.m_InputMappings  = { { planA.m_OpGraph.GetBuffers()[0], partAInputSlot } };
        planA.m_OutputMappings = { { planA.m_OpGraph.GetBuffers()[1], partAOutputSlot } };

        WHEN("Lonely section with a plan that has no Ple Op")
        {
            REQUIRE(planA.m_OpGraph.GetBuffers()[0]->m_Offset.has_value() == false);
            REQUIRE(planA.m_OpGraph.GetBuffers()[1]->m_Offset.has_value() == false);
            REQUIRE(combiner.IsPlanAllocated(context, planA, nullptr, StatsType::SinglePartSection) == true);
            REQUIRE(planA.m_OpGraph.GetBuffers()[0]->m_Offset.has_value() == true);
            REQUIRE(planA.m_OpGraph.GetBuffers()[0]->m_Offset.value() == currentSramOffset);
            currentSramOffset = planA.m_OpGraph.GetBuffers()[0]->m_SizeInBytes / hwCaps.GetNumberOfSrams();
            REQUIRE(planA.m_OpGraph.GetBuffers()[1]->m_Offset.has_value() == false);
            REQUIRE(context.pleOps.size() == 0);
        }

        WHEN("Lonely section with a plan that has Ple Op")
        {
            // Adding a passthrough PLE kernel to the plan
            // The PleKernelId is expected to be PASSTHROUGH_8x8_2
            auto op = std::make_unique<PleOp>(ethosn::command_stream::PleOperation::PASSTHROUGH, BlockConfig{ 8u, 8u },
                                              1, std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 8 } },
                                              TensorShape{ 1, 8, 8, 8 }, DataType::UINT8_QUANTIZED, true);
            numMemoryStripes.m_Output = 1;
            auto outBufferAndPleOp =
                AddPleToOpGraph(planA.m_OpGraph, TensorShape{ 1, 8, 8, 8 }, numMemoryStripes, std::move(op),
                                TensorShape{ 1, 8, 8, 8 }, QuantizationInfo(), DataType::UINT8_QUANTIZED, operationIds);

            Op* maybePleOp   = planA.m_OpGraph.GetOp(1);
            const bool isPle = IsPleOp(maybePleOp);
            REQUIRE(isPle);

            PleOp* actualPleOp = static_cast<PleOp*>(maybePleOp);

            REQUIRE(actualPleOp->m_Offset.has_value() == false);

            REQUIRE(planA.m_OpGraph.GetBuffers()[0]->m_Offset.has_value() == false);
            REQUIRE(planA.m_OpGraph.GetBuffers()[1]->m_Offset.has_value() == false);
            REQUIRE(planA.m_OpGraph.GetBuffers()[2]->m_Offset.has_value() == false);
            REQUIRE(combiner.IsPlanAllocated(context, planA, nullptr, StatsType::SinglePartSection) == true);

            REQUIRE(actualPleOp->m_Offset.has_value() == true);
            REQUIRE(actualPleOp->m_Offset.value() == currentSramOffset);

            currentSramOffset += hwCaps.GetMaxPleSize();
            REQUIRE(planA.m_OpGraph.GetBuffers()[0]->m_Offset.has_value() == true);
            REQUIRE(planA.m_OpGraph.GetBuffers()[0]->m_Offset.value() == currentSramOffset);
            currentSramOffset += planA.m_OpGraph.GetBuffers()[0]->m_SizeInBytes / hwCaps.GetNumberOfSrams();
            // Note that Buffer 1 is the output from MceOp where its location is in PleInputSRAM not SRAM
            REQUIRE(planA.m_OpGraph.GetBuffers()[1]->m_Offset.has_value() == false);
            REQUIRE(planA.m_OpGraph.GetBuffers()[2]->m_Offset.has_value() == true);
            REQUIRE(planA.m_OpGraph.GetBuffers()[2]->m_Offset.value() == currentSramOffset);
            currentSramOffset += planA.m_OpGraph.GetBuffers()[2]->m_SizeInBytes / hwCaps.GetNumberOfSrams();
            REQUIRE(context.pleOps.size() == 1);

            ETHOSN_UNUSED(outBufferAndPleOp);
        }
    }
}

TEST_CASE("SramAllocationForMultiplePartSection", "[CombinerDFS]")
{
    GIVEN("A Graph of three parts where their corresponding plans fits into a single section")
    {
        GraphOfParts graph;

        auto& parts = graph.m_Parts;

        auto pA = std::make_unique<MockPart>(graph.GeneratePartId());
        auto pB = std::make_unique<MockPart>(graph.GeneratePartId());
        auto pC = std::make_unique<MockPart>(graph.GeneratePartId());

        const BasePart& partA = *pA;
        const BasePart& partB = *pB;
        const BasePart& partC = *pC;

        parts.push_back(std::move(pA));
        parts.push_back(std::move(pB));
        parts.push_back(std::move(pC));

        PartInputSlot partAInputSlot   = { partA.GetPartId(), 0 };
        PartOutputSlot partAOutputSlot = { partA.GetPartId(), 0 };
        PartInputSlot partBInputSlot   = { partB.GetPartId(), 0 };
        PartOutputSlot partBOutputSlot = { partB.GetPartId(), 0 };
        PartInputSlot partCInputSlot   = { partC.GetPartId(), 0 };
        PartOutputSlot partCOutputSlot = { partC.GetPartId(), 0 };

        ethosn::support_library::impl::NumMemoryStripes numMemoryStripes;
        const std::set<uint32_t> operationIds = { 0 };

        const CompilationOptions compOpt;
        const EstimationOptions estOpt;
        const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
        const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

        Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);
        SectionContext context     = { SramAllocator(hwCaps.GetTotalSramSize() / hwCaps.GetNumberOfSrams()), {}, {} };
        uint32_t currentSramOffset = 0;

        Plan planA;

        const uint32_t inputBufferSize  = 512;
        const uint32_t outputBufferSize = 512;

        planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                           TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 8, 8, 8 },
                                                           TraversalOrder::Xyz, inputBufferSize, QuantizationInfo()));
        planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::PleInputSram, CascadingBufferFormat::NHWCB,
                                                           TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 8, 8, 8 },
                                                           TraversalOrder::Xyz, outputBufferSize, QuantizationInfo()));

        planA.m_OpGraph.AddOp(std::make_unique<MceOp>(
            MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct, BlockConfig{ 8u, 8u }, TensorShape{ 1, 8, 8, 8 },
            TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 8, 8, 8 }, TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));

        planA.m_InputMappings  = { { planA.m_OpGraph.GetBuffers()[0], partAInputSlot } };
        planA.m_OutputMappings = { { planA.m_OpGraph.GetBuffers()[1], partAOutputSlot } };

        WHEN("Starting the section with the first plan that has no Ple Op")
        {
            REQUIRE(planA.m_OpGraph.GetBuffers()[0]->m_Offset.has_value() == false);
            REQUIRE(planA.m_OpGraph.GetBuffers()[1]->m_Offset.has_value() == false);
            REQUIRE(combiner.IsPlanAllocated(context, planA, nullptr, StatsType::StartSection) == true);
            REQUIRE(planA.m_OpGraph.GetBuffers()[0]->m_Offset.has_value() == true);
            REQUIRE(planA.m_OpGraph.GetBuffers()[0]->m_Offset.value() == currentSramOffset);
            currentSramOffset = planA.m_OpGraph.GetBuffers()[0]->m_SizeInBytes / hwCaps.GetNumberOfSrams();
            REQUIRE(planA.m_OpGraph.GetBuffers()[1]->m_Offset.has_value() == false);
            REQUIRE(context.pleOps.size() == 0);
        }

        WHEN("Starting the section with the first plan that has Ple Op")
        {
            // Adding a passthrough PLE kernel to the plan
            // The PleKernelId is expected to be PASSTHROUGH_8x8_2
            auto op = std::make_unique<PleOp>(ethosn::command_stream::PleOperation::PASSTHROUGH, BlockConfig{ 8u, 8u },
                                              1, std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 8 } },
                                              TensorShape{ 1, 8, 8, 8 }, DataType::UINT8_QUANTIZED, true);
            numMemoryStripes.m_Output = 1;
            auto outBufferAndPleOp =
                AddPleToOpGraph(planA.m_OpGraph, TensorShape{ 1, 8, 8, 8 }, numMemoryStripes, std::move(op),
                                TensorShape{ 1, 8, 8, 8 }, QuantizationInfo(), DataType::UINT8_QUANTIZED, operationIds);

            Op* maybePleOp   = planA.m_OpGraph.GetOp(1);
            const bool isPle = IsPleOp(maybePleOp);
            REQUIRE(isPle);

            PleOp* actualPleOp = static_cast<PleOp*>(maybePleOp);

            REQUIRE(actualPleOp->m_Offset.has_value() == false);

            REQUIRE(planA.m_OpGraph.GetBuffers()[0]->m_Offset.has_value() == false);
            REQUIRE(planA.m_OpGraph.GetBuffers()[1]->m_Offset.has_value() == false);
            REQUIRE(planA.m_OpGraph.GetBuffers()[2]->m_Offset.has_value() == false);
            REQUIRE(combiner.IsPlanAllocated(context, planA, nullptr, StatsType::StartSection) == true);

            REQUIRE(actualPleOp->m_Offset.has_value() == true);
            REQUIRE(actualPleOp->m_Offset.value() == currentSramOffset);

            currentSramOffset += hwCaps.GetMaxPleSize();
            REQUIRE(planA.m_OpGraph.GetBuffers()[0]->m_Offset.has_value() == true);
            REQUIRE(planA.m_OpGraph.GetBuffers()[0]->m_Offset.value() == currentSramOffset);
            currentSramOffset += planA.m_OpGraph.GetBuffers()[0]->m_SizeInBytes / hwCaps.GetNumberOfSrams();
            // Note that Buffer 1 is the output from MceOp where its location is in PleInputSRAM not SRAM
            REQUIRE(planA.m_OpGraph.GetBuffers()[1]->m_Offset.has_value() == false);
            REQUIRE(planA.m_OpGraph.GetBuffers()[2]->m_Offset.has_value() == true);
            REQUIRE(planA.m_OpGraph.GetBuffers()[2]->m_Offset.value() == currentSramOffset);
            currentSramOffset += planA.m_OpGraph.GetBuffers()[2]->m_SizeInBytes / hwCaps.GetNumberOfSrams();
            REQUIRE(context.pleOps.size() == 1);

            Plan planB;

            const uint32_t InputBufferSize  = 512;
            const uint32_t OutputBufferSize = 512;

            planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(
                Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 8, 8, 8 },
                TraversalOrder::Xyz, InputBufferSize, QuantizationInfo()));

            planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(
                Location::PleInputSram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 8, 8, 8 },
                TensorShape{ 1, 8, 8, 8 }, TraversalOrder::Xyz, OutputBufferSize, QuantizationInfo()));

            planB.m_OpGraph.AddOp(std::make_unique<MceOp>(MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
                                                          BlockConfig{ 8u, 8u }, TensorShape{ 1, 8, 8, 8 },
                                                          TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 8, 8, 8 },
                                                          TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));

            planB.m_InputMappings  = { { planB.m_OpGraph.GetBuffers()[0], partBInputSlot } };
            planB.m_OutputMappings = { { planB.m_OpGraph.GetBuffers()[1], partBOutputSlot } };

            WHEN("Continuing the section with the second plan that has no Ple Op")
            {
                REQUIRE(planB.m_OpGraph.GetBuffers()[0]->m_Offset.has_value() == false);
                REQUIRE(planB.m_OpGraph.GetBuffers()[1]->m_Offset.has_value() == false);
                REQUIRE(combiner.IsPlanAllocated(context, planB, planA.m_OpGraph.GetBuffers()[2],
                                                 StatsType::ContinueSection) == true);
                REQUIRE(planB.m_OpGraph.GetBuffers()[0]->m_Offset.has_value() == true);
                REQUIRE(planB.m_OpGraph.GetBuffers()[0]->m_Offset.value() ==
                        planA.m_OpGraph.GetBuffers()[2]->m_Offset.value());
                REQUIRE(planB.m_OpGraph.GetBuffers()[1]->m_Offset.has_value() == false);
            }

            WHEN("Continuing the section with the second plan that has already loaded Ple Op")
            {
                // Adding a passthrough PLE kernel to the plan
                // The PleKernelId is expected to be PASSTHROUGH_8x8_2
                auto op =
                    std::make_unique<PleOp>(ethosn::command_stream::PleOperation::PASSTHROUGH, BlockConfig{ 8u, 8u }, 1,
                                            std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 8 } },
                                            TensorShape{ 1, 8, 8, 8 }, DataType::UINT8_QUANTIZED, true);
                numMemoryStripes.m_Output = 1;
                auto outBufferAndPleOp = AddPleToOpGraph(planB.m_OpGraph, TensorShape{ 1, 8, 8, 8 }, numMemoryStripes,
                                                         std::move(op), TensorShape{ 1, 8, 8, 8 }, QuantizationInfo(),
                                                         DataType::UINT8_QUANTIZED, operationIds);

                Op* maybePleOpA   = planA.m_OpGraph.GetOp(1);
                const bool isPleA = IsPleOp(maybePleOpA);
                REQUIRE(isPleA);

                PleOp* actualPleOpA = static_cast<PleOp*>(maybePleOpA);

                Op* maybePleOpB   = planB.m_OpGraph.GetOp(1);
                const bool isPleB = IsPleOp(maybePleOpB);
                REQUIRE(isPleB);

                PleOp* actualPleOpB = static_cast<PleOp*>(maybePleOpB);

                REQUIRE(actualPleOpB->m_LoadKernel == true);
                REQUIRE(combiner.IsPlanAllocated(context, planB, planA.m_OpGraph.GetBuffers()[2],
                                                 StatsType::ContinueSection) == true);
                REQUIRE(context.pleOps.size() == 1);

                REQUIRE(actualPleOpB->m_LoadKernel == false);
                REQUIRE(actualPleOpB->m_Offset == actualPleOpA->m_Offset);

                REQUIRE(planB.m_OpGraph.GetBuffers()[2]->m_Offset.value() == currentSramOffset);
                currentSramOffset += planB.m_OpGraph.GetBuffers()[2]->m_SizeInBytes / hwCaps.GetNumberOfSrams();

                ETHOSN_UNUSED(outBufferAndPleOp);
            }

            WHEN("Continuing the section with the second plan that has Ple Op not already loaded")
            {
                // Adding a passthrough PLE kernel to the plan
                auto op =
                    std::make_unique<PleOp>(ethosn::command_stream::PleOperation::PASSTHROUGH, BlockConfig{ 16u, 16u },
                                            1, std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 8 } },
                                            TensorShape{ 1, 8, 8, 8 }, DataType::UINT8_QUANTIZED, true);

                numMemoryStripes.m_Output = 1;

                auto outBufferAndPleOp = AddPleToOpGraph(planB.m_OpGraph, TensorShape{ 1, 8, 8, 8 }, numMemoryStripes,
                                                         std::move(op), TensorShape{ 1, 8, 8, 8 }, QuantizationInfo(),
                                                         DataType::UINT8_QUANTIZED, operationIds);

                Op* maybePleOpA   = planA.m_OpGraph.GetOp(1);
                const bool isPleA = IsPleOp(maybePleOpA);
                REQUIRE(isPleA);

                Op* maybePleOpB   = planB.m_OpGraph.GetOp(1);
                const bool isPleB = IsPleOp(maybePleOpB);
                REQUIRE(isPleB);

                PleOp* actualPleOpB = static_cast<PleOp*>(maybePleOpB);

                REQUIRE(actualPleOpB->m_LoadKernel == true);

                REQUIRE(combiner.IsPlanAllocated(context, planB, planA.m_OpGraph.GetBuffers()[2],
                                                 StatsType::ContinueSection) == true);
                REQUIRE(context.pleOps.size() == 2);

                REQUIRE(actualPleOpB->m_LoadKernel == true);
                REQUIRE(actualPleOpB->m_Offset == currentSramOffset);

                currentSramOffset += hwCaps.GetMaxPleSize();
                REQUIRE(planB.m_OpGraph.GetBuffers()[2]->m_Offset.value() == currentSramOffset);
                currentSramOffset += planB.m_OpGraph.GetBuffers()[2]->m_SizeInBytes / hwCaps.GetNumberOfSrams();

                Plan planC;

                const uint32_t InputBufferSize  = 512;
                const uint32_t OutputBufferSize = 512;

                planC.m_OpGraph.AddBuffer(std::make_unique<Buffer>(
                    Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 8, 8, 8 },
                    TraversalOrder::Xyz, InputBufferSize, QuantizationInfo()));

                planC.m_OpGraph.AddBuffer(std::make_unique<Buffer>(
                    Location::PleInputSram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 8, 8, 8 },
                    TensorShape{ 1, 8, 8, 8 }, TraversalOrder::Xyz, OutputBufferSize, QuantizationInfo()));

                planC.m_OpGraph.AddOp(std::make_unique<MceOp>(MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
                                                              BlockConfig{ 8u, 8u }, TensorShape{ 1, 8, 8, 8 },
                                                              TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 8, 8, 8 },
                                                              TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));

                planC.m_InputMappings  = { { planC.m_OpGraph.GetBuffers()[0], partCInputSlot } };
                planC.m_OutputMappings = { { planC.m_OpGraph.GetBuffers()[1], partCOutputSlot } };

                WHEN("Ending the section with the third plan that has no Ple Op")
                {
                    REQUIRE(planC.m_OpGraph.GetBuffers()[0]->m_Offset.has_value() == false);
                    REQUIRE(planC.m_OpGraph.GetBuffers()[1]->m_Offset.has_value() == false);
                    REQUIRE(combiner.IsPlanAllocated(context, planC, planB.m_OpGraph.GetBuffers()[2],
                                                     StatsType::EndSection) == true);
                    REQUIRE(planC.m_OpGraph.GetBuffers()[0]->m_Offset.has_value() == true);
                    REQUIRE(planC.m_OpGraph.GetBuffers()[0]->m_Offset.value() ==
                            planB.m_OpGraph.GetBuffers()[2]->m_Offset.value());
                    REQUIRE(planC.m_OpGraph.GetBuffers()[1]->m_Offset.has_value() == false);
                }

                WHEN("Ending the section with the third plan that has already loaded Ple Op")
                {
                    // Adding a passthrough PLE kernel to the plan
                    auto op = std::make_unique<PleOp>(ethosn::command_stream::PleOperation::PASSTHROUGH,
                                                      BlockConfig{ 16u, 16u }, 1,
                                                      std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 8 } },
                                                      TensorShape{ 1, 8, 8, 8 }, DataType::UINT8_QUANTIZED, true);
                    numMemoryStripes.m_Output = 1;
                    auto outBufferAndPleOp    = AddPleToOpGraph(
                        planC.m_OpGraph, TensorShape{ 1, 8, 8, 8 }, numMemoryStripes, std::move(op),
                        TensorShape{ 1, 8, 8, 8 }, QuantizationInfo(), DataType::UINT8_QUANTIZED, operationIds);

                    Op* maybePleOpA   = planA.m_OpGraph.GetOp(1);
                    const bool isPleA = IsPleOp(maybePleOpA);
                    REQUIRE(isPleA);

                    Op* maybePleOpB   = planB.m_OpGraph.GetOp(1);
                    const bool isPleB = IsPleOp(maybePleOpB);
                    REQUIRE(isPleB);

                    PleOp* actualPleOpB = static_cast<PleOp*>(maybePleOpB);

                    Op* maybePleOpC   = planC.m_OpGraph.GetOp(1);
                    const bool isPleC = IsPleOp(maybePleOpC);
                    REQUIRE(isPleC);

                    PleOp* actualPleOpC = static_cast<PleOp*>(maybePleOpC);

                    REQUIRE(actualPleOpC->m_LoadKernel == true);

                    REQUIRE(combiner.IsPlanAllocated(context, planC, planB.m_OpGraph.GetBuffers()[2],
                                                     StatsType::EndSection) == true);
                    REQUIRE(context.pleOps.size() == 2);

                    REQUIRE(actualPleOpC->m_LoadKernel == false);

                    REQUIRE(actualPleOpC->m_Offset == actualPleOpB->m_Offset);

                    REQUIRE(planC.m_OpGraph.GetBuffers()[2]->m_Offset.value() == currentSramOffset);
                    currentSramOffset += planC.m_OpGraph.GetBuffers()[2]->m_SizeInBytes / hwCaps.GetNumberOfSrams();

                    ETHOSN_UNUSED(outBufferAndPleOp);
                }

                WHEN("Ending the section with the third plan that has Ple Op not already loaded")
                {
                    // Adding a passthrough PLE kernel to the plan
                    auto op = std::make_unique<PleOp>(ethosn::command_stream::PleOperation::PASSTHROUGH,
                                                      BlockConfig{ 8u, 32u }, 1,
                                                      std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 8 } },
                                                      TensorShape{ 1, 8, 8, 8 }, DataType::UINT8_QUANTIZED, true);

                    numMemoryStripes.m_Output = 1;

                    auto outBufferAndPleOp = AddPleToOpGraph(
                        planC.m_OpGraph, TensorShape{ 1, 8, 8, 8 }, numMemoryStripes, std::move(op),
                        TensorShape{ 1, 8, 8, 8 }, QuantizationInfo(), DataType::UINT8_QUANTIZED, operationIds);

                    Op* maybePleOpB   = planB.m_OpGraph.GetOp(1);
                    const bool isPleB = IsPleOp(maybePleOpB);
                    REQUIRE(isPleB);

                    PleOp* actualPleOpB = static_cast<PleOp*>(maybePleOpB);

                    Op* maybePleOpC   = planC.m_OpGraph.GetOp(1);
                    const bool isPleC = IsPleOp(maybePleOpC);
                    REQUIRE(isPleC);

                    PleOp* actualPleOpC = static_cast<PleOp*>(maybePleOpC);

                    REQUIRE(actualPleOpC->m_LoadKernel == true);

                    REQUIRE(combiner.IsPlanAllocated(context, planC, planB.m_OpGraph.GetBuffers()[2],
                                                     StatsType::EndSection) == true);
                    REQUIRE(context.pleOps.size() == 3);

                    REQUIRE(actualPleOpB->m_LoadKernel == true);

                    REQUIRE(actualPleOpC->m_Offset == currentSramOffset);

                    currentSramOffset += hwCaps.GetMaxPleSize();
                    REQUIRE(planC.m_OpGraph.GetBuffers()[2]->m_Offset.value() == currentSramOffset);
                    currentSramOffset += planC.m_OpGraph.GetBuffers()[2]->m_SizeInBytes / hwCaps.GetNumberOfSrams();

                    ETHOSN_UNUSED(outBufferAndPleOp);
                }

                ETHOSN_UNUSED(outBufferAndPleOp);
            }

            ETHOSN_UNUSED(outBufferAndPleOp);
        }
    }
}

TEST_CASE("ArePlansAllowedToMerge IdentityParts", "[CombinerDFS]")
{
    // Create graph:
    //
    //  --> A - - > B
    //
    GraphOfParts graph;
    auto& parts       = graph.m_Parts;
    auto& connections = graph.m_Connections;

    auto pA = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB = std::make_unique<MockPart>(graph.GeneratePartId());

    PartId partAId = pA->GetPartId();
    PartId partBId = pB->GetPartId();

    parts.push_back(std::move(pA));
    parts.push_back(std::move(pB));

    PartOutputSlot partAOutputSlot0 = { partAId, 0 };

    PartInputSlot partBInputSlot0 = { partBId, 0 };

    connections[partBInputSlot0] = partAOutputSlot0;

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Plan planA;
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 0, QuantizationInfo()));
    MceOp mceOp(MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct, BlockConfig{ 16u, 16u },
                TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 1, 1, 64 },
                TraversalOrder::Xyz, Stride(), 0, 0, 0, 255);
    planA.m_OpGraph.AddOp(std::make_unique<MceOp>(std::move(mceOp)));
    planA.m_OpGraph.SetProducer(planA.m_OpGraph.GetBuffers()[0], planA.m_OpGraph.GetOps()[0]);
    planA.m_OutputMappings = { { planA.m_OpGraph.GetBuffers()[0], partAOutputSlot0 } };

    Plan planB;
    planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    PleOp pleOp(PleOperation::PASSTHROUGH, BlockConfig{ 16u, 16u }, 1U, { TensorShape{ 1, 64, 64, 64 } },
                TensorShape{ 1, 64, 64, 64 }, DataType::UINT8_QUANTIZED, true);
    planB.m_OpGraph.AddOp(std::make_unique<PleOp>(std::move(pleOp)));
    planB.m_OpGraph.AddConsumer(planB.m_OpGraph.GetBuffers()[0], planB.m_OpGraph.GetOps()[0], 0);
    planB.m_InputMappings = { { planB.m_OpGraph.GetBuffers()[0], partBInputSlot0 } };

    Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);

    REQUIRE(combiner.ArePlansAllowedToMerge(planA, planB) == true);

    planA.m_HasIdentityPle = true;

    REQUIRE(combiner.ArePlansAllowedToMerge(planA, planB) == true);

    planB.m_HasIdentityMce = true;

    REQUIRE(combiner.ArePlansAllowedToMerge(planA, planB) == false);
}

TEST_CASE("IsSectionSizeSupported", "[CombinerDFS]")
{
    uint32_t totalAgentsRef = 0;

    GraphOfParts graph;
    auto& parts = graph.m_Parts;

    size_t mceOpIndex;
    size_t pleOpIndex;
    size_t dmaOpIndex;

    // Create 3 identical plans, each of the topology:
    //     Input - Mce - PleInputSram - Ple - Output
    //             /
    //     Dma - Weights
    PartInputSlot partsInputSlot0[3];
    PartOutputSlot partsOutputSlot0[3];
    Plan plans[3];
    int i = 0;
    for (Plan& plan : plans)
    {
        auto part     = std::make_unique<MockPart>(graph.GeneratePartId());
        PartId partId = part->GetPartId();
        parts.push_back(std::move(part));

        partsInputSlot0[i]  = { partId, 0 };
        partsOutputSlot0[i] = { partId, 0 };

        plan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                          TensorShape{ 1, 32, 16, 16 }, TensorShape{ 1, 16, 16, 16 },
                                                          TraversalOrder::Xyz, 4, QuantizationInfo()));
        plan.m_OpGraph.GetBuffers().back()->m_DebugTag = "InputSram";
        size_t inputBufferIndex                        = plan.m_OpGraph.GetBuffers().size() - 1;

        plan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::PleInputSram, CascadingBufferFormat::NHWCB,
                                                          TensorShape{ 1, 32, 16, 16 }, TensorShape{ 1, 16, 16, 16 },
                                                          TraversalOrder::Xyz, 4, QuantizationInfo()));
        plan.m_OpGraph.GetBuffers().back()->m_DebugTag = "PleInputSram";
        size_t pleInputSramIndex                       = plan.m_OpGraph.GetBuffers().size() - 1;

        plan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::WEIGHT,
                                                          TensorShape{ 1, 32, 16, 16 }, TensorShape{ 1, 16, 16, 16 },
                                                          TraversalOrder::Xyz, 4, QuantizationInfo()));
        plan.m_OpGraph.GetBuffers().back()->m_DebugTag = "MceWeightsSram";
        size_t mceWeightsBufferIndex                   = plan.m_OpGraph.GetBuffers().size() - 1;

        plan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB,
                                                          TensorShape{ 1, 32, 16, 16 }, TensorShape{ 1, 16, 16, 16 },
                                                          TraversalOrder::Xyz, 4, QuantizationInfo()));
        plan.m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputSram";
        size_t outputBufferIndex                       = plan.m_OpGraph.GetBuffers().size() - 1;

        plan.m_OpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
        dmaOpIndex                                      = plan.m_OpGraph.GetOps().size() - 1;
        plan.m_OpGraph.GetOps()[dmaOpIndex]->m_DebugTag = "DmaOp";
        // GetNumberOfAgents() returns the number of agents needed execute the Op.
        // According to current implementation of the function, it returns 2 if a
        // Ple Op's Kernel needs to be loaded from Dram. In every other case, it
        // returns 1. The test below assumes this implementation of the function.
        totalAgentsRef += plan.m_OpGraph.GetOps()[dmaOpIndex]->GetNumberOfAgents();

        // All Ops with have Lifetime::Cascade
        plan.m_OpGraph.AddOp(std::make_unique<MceOp>());
        mceOpIndex                                      = plan.m_OpGraph.GetOps().size() - 1;
        plan.m_OpGraph.GetOps()[mceOpIndex]->m_DebugTag = "MceOp";
        totalAgentsRef += plan.m_OpGraph.GetOps()[mceOpIndex]->GetNumberOfAgents();

        plan.m_OpGraph.AddOp(std::make_unique<PleOp>());
        pleOpIndex                                      = plan.m_OpGraph.GetOps().size() - 1;
        plan.m_OpGraph.GetOps()[pleOpIndex]->m_DebugTag = "PleOp";
        totalAgentsRef += plan.m_OpGraph.GetOps()[pleOpIndex]->GetNumberOfAgents();

        plan.m_OpGraph.AddConsumer(plan.m_OpGraph.GetBuffers()[inputBufferIndex], plan.m_OpGraph.GetOps()[mceOpIndex],
                                   0);
        plan.m_OpGraph.SetProducer(plan.m_OpGraph.GetBuffers()[pleInputSramIndex], plan.m_OpGraph.GetOps()[mceOpIndex]);
        plan.m_OpGraph.AddConsumer(plan.m_OpGraph.GetBuffers()[pleInputSramIndex], plan.m_OpGraph.GetOps()[pleOpIndex],
                                   0);
        plan.m_OpGraph.AddConsumer(plan.m_OpGraph.GetBuffers()[mceWeightsBufferIndex],
                                   plan.m_OpGraph.GetOps()[mceOpIndex], 1);
        plan.m_OpGraph.SetProducer(plan.m_OpGraph.GetBuffers()[outputBufferIndex], plan.m_OpGraph.GetOps()[pleOpIndex]);
        plan.m_OpGraph.SetProducer(plan.m_OpGraph.GetBuffers()[mceWeightsBufferIndex],
                                   plan.m_OpGraph.GetOps()[dmaOpIndex]);
        plan.m_InputMappings  = { { plan.m_OpGraph.GetBuffers()[inputBufferIndex], partsInputSlot0[i] } };
        plan.m_OutputMappings = { { plan.m_OpGraph.GetBuffers()[outputBufferIndex], partsOutputSlot0[i] } };

        ++i;
    }

    // Account for Dma Ops in glue logic
    totalAgentsRef += 2;

    uint32_t totalAgents = 0;

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);

    GIVEN("Three plans that can be combined into a single section")
    {
        WHEN("Window size is greater than the total number of agents")
        {
            const HardwareCapabilities hwCaps = GetHwCapabilitiesWithFwOverrides(
                EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO, {}, totalAgentsRef + 1, {}, {});
            Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);

            REQUIRE(combiner.IsSectionSizeSupported(StatsType::StartSection, plans[0], totalAgents) == true);
            REQUIRE(combiner.IsSectionSizeSupported(StatsType::ContinueSection, plans[1], totalAgents) == true);
            REQUIRE(combiner.IsSectionSizeSupported(StatsType::EndSection, plans[2], totalAgents) == true);
        }
        WHEN("Window size is equal to the total number of agents")
        {
            const HardwareCapabilities hwCaps =
                GetHwCapabilitiesWithFwOverrides(EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO, {}, totalAgentsRef, {}, {});
            Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);

            REQUIRE(combiner.IsSectionSizeSupported(StatsType::StartSection, plans[0], totalAgents) == true);
            REQUIRE(combiner.IsSectionSizeSupported(StatsType::ContinueSection, plans[1], totalAgents) == true);
            REQUIRE(combiner.IsSectionSizeSupported(StatsType::EndSection, plans[2], totalAgents) == true);
            REQUIRE(totalAgents == totalAgentsRef);
        }
        WHEN("Window size is smaller than the total number of agents if all Ops were Cascade")
        {
            const HardwareCapabilities hwCaps = GetHwCapabilitiesWithFwOverrides(
                EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO, {}, totalAgentsRef - 1, {}, {});
            Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);

            REQUIRE(combiner.IsSectionSizeSupported(StatsType::StartSection, plans[0], totalAgents) == true);
            WHEN("All Ops are Cascade")
            {
                uint32_t totalAgentsComb1 = totalAgents;
                REQUIRE(combiner.IsSectionSizeSupported(StatsType::ContinueSection, plans[1], totalAgentsComb1) ==
                        true);
                REQUIRE(combiner.IsSectionSizeSupported(StatsType::EndSection, plans[2], totalAgentsComb1) == false);
                REQUIRE(totalAgentsComb1 == 14);

                uint32_t totalAgentsComb2 = totalAgents;
                REQUIRE(combiner.IsSectionSizeSupported(StatsType::EndSection, plans[1], totalAgentsComb2) == true);
                REQUIRE(totalAgentsComb2 == 10);
            }
            WHEN("The Output of the Ple Op the full tensor")
            {
                plans[1].m_OpGraph.GetBuffers()[3]->m_StripeShape = plans[1].m_OpGraph.GetBuffers()[3]->m_TensorShape;
                REQUIRE(combiner.IsSectionSizeSupported(StatsType::ContinueSection, plans[1], totalAgents) == true);
                REQUIRE(combiner.IsSectionSizeSupported(StatsType::EndSection, plans[2], totalAgents) == true);
            }
            WHEN("The Mce Op in the third plan is Atomic")
            {
                plans[2].m_OpGraph.GetBuffers()[3]->m_StripeShape = plans[2].m_OpGraph.GetBuffers()[3]->m_TensorShape;
                REQUIRE(combiner.IsSectionSizeSupported(StatsType::ContinueSection, plans[1], totalAgents) == true);
                REQUIRE(combiner.IsSectionSizeSupported(StatsType::EndSection, plans[2], totalAgents) == true);
            }
            WHEN("The weight loader Dma Op in the second plan is Atomic")
            {
                plans[1].m_OpGraph.GetBuffers()[2]->m_StripeShape = plans[1].m_OpGraph.GetBuffers()[2]->m_TensorShape;
                REQUIRE(combiner.IsSectionSizeSupported(StatsType::ContinueSection, plans[1], totalAgents) == true);
                REQUIRE(combiner.IsSectionSizeSupported(StatsType::EndSection, plans[2], totalAgents) == false);
            }
        }
        WHEN("Window size is smaller than the total number of agents so that no plan fits")
        {
            const HardwareCapabilities hwCaps =
                GetHwCapabilitiesWithFwOverrides(EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO, {}, 3, {}, {});
            Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);

            REQUIRE(combiner.IsSectionSizeSupported(StatsType::StartSection, plans[0], totalAgents) == false);
            REQUIRE(combiner.IsSectionSizeSupported(StatsType::ContinueSection, plans[1], totalAgents) == false);
            REQUIRE(combiner.IsSectionSizeSupported(StatsType::EndSection, plans[2], totalAgents) == false);
        }
    }
    GIVEN("A single part section")
    {
        WHEN("Window size can accomodate the plan")
        {
            const HardwareCapabilities hwCaps =
                GetHwCapabilitiesWithFwOverrides(EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO, {}, 16, {}, {});
            Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);

            REQUIRE(combiner.IsSectionSizeSupported(StatsType::SinglePartSection, plans[0], totalAgents) == true);
        }
        WHEN("Window size is smaller than the plan")
        {
            const HardwareCapabilities hwCaps =
                GetHwCapabilitiesWithFwOverrides(EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO, {}, 2, {}, {});
            Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);

            REQUIRE(combiner.IsSectionSizeSupported(StatsType::SinglePartSection, plans[0], totalAgents) == false);
        }
        WHEN("Window size is only 4 but all Ops are Atomic")
        {
            const HardwareCapabilities hwCaps =
                GetHwCapabilitiesWithFwOverrides(EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO, {}, 4, {}, {});
            Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);
            for (Buffer* b : plans[0].m_OpGraph.GetBuffers())
            {
                b->m_StripeShape = b->m_TensorShape;
            }
            REQUIRE(combiner.IsSectionSizeSupported(StatsType::SinglePartSection, plans[0], totalAgents) == true);
        }
        WHEN("The plan contains a Concat Part")
        {
            const PartId partId = 1;

            std::vector<ethosn::support_library::TensorInfo> inputTensorsInfo;
            ethosn::support_library::TensorInfo inputTensorInfo1;
            ethosn::support_library::TensorInfo inputTensorInfo2;
            CompilerDataFormat compilerDataFormat;

            inputTensorInfo1.m_Dimensions = { 1, 16, 16, 16 };
            inputTensorInfo1.m_DataType   = ethosn::support_library::DataType::INT8_QUANTIZED;
            inputTensorInfo1.m_DataFormat = ethosn::support_library::DataFormat::NHWC;

            inputTensorInfo2.m_Dimensions = { 1, 16, 16, 16 };
            inputTensorInfo2.m_DataType   = ethosn::support_library::DataType::INT8_QUANTIZED;
            inputTensorInfo2.m_DataFormat = ethosn::support_library::DataFormat::NHWC;

            compilerDataFormat = CompilerDataFormat::NHWC;

            inputTensorsInfo.push_back(inputTensorInfo1);
            inputTensorsInfo.push_back(inputTensorInfo2);

            QuantizationInfo quantizationInfo(0, 1.0f);
            ConcatenationInfo concatInfo(1, quantizationInfo);

            const std::set<uint32_t> operationIds = { 1 };
            const EstimationOptions estOpt;
            const CompilationOptions compOpt;

            const HardwareCapabilities hwCaps =
                GetHwCapabilitiesWithFwOverrides(EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO, {}, 64, {}, {});

            ConcatPart concatPart(partId, inputTensorsInfo, concatInfo, compilerDataFormat, operationIds, estOpt,
                                  compOpt, hwCaps);

            Plans concatPlans =
                concatPart.GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 0);

            Combiner combiner(graph, hwCaps, compOpt, estOpt, debuggingContext);

            REQUIRE(combiner.IsSectionSizeSupported(StatsType::SinglePartSection, concatPlans[0], totalAgents) == true);

            // The number of agents in a Concat Part must be equal to twice the number of its inputs. As the number of
            // inputs in this case is two, the number of agents must be 4. However by the end of the plan,
            // the data is back in DRAM so the final tally is zero.
            REQUIRE(totalAgents == 0);
        }
    }
}
