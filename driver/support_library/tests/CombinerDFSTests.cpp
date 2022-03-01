// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/DebuggingContext.hpp"
#include "../src/GraphNodes.hpp"
#include "../src/SramAllocator.hpp"
#include "../src/cascading/Cascading.hpp"
#include "../src/cascading/CombinerDFS.hpp"
#include "../src/cascading/McePart.hpp"
#include "../src/cascading/StripeHelper.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>

#include <fstream>

using namespace ethosn::support_library;
using namespace ethosn::command_stream;
using PleKernelId = ethosn::command_stream::cascading::PleKernelId;

// These Mock classes are used locally to create a test framework for double-buffering logic.
class WeightPart : public MockPart
{
public:
    WeightPart(PartId id,
               uint32_t* numPlansCounter,
               std::vector<uint32_t>* numWeightBuffers,
               std::function<bool(CascadeType, PartId)> filter)
        : MockPart(id)
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

        opGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                   TraversalOrder::Xyz));
        Buffer* buffer             = opGraph.GetBuffers()[0];
        buffer->m_TensorShape      = { 1, 16, 16, 16 };
        buffer->m_StripeShape      = { 1, 16, 16, 16 };
        buffer->m_SizeInBytes      = 16 * 16 * 16;
        buffer->m_QuantizationInfo = { 0, 1.f };

        inputMappings[buffer]  = PartInputSlot{ m_PartId, 0 };
        outputMappings[buffer] = PartOutputSlot{ m_PartId, 0 };

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
                 std::function<bool(CascadeType, PartId)> filter)
        : WeightPart(id, numPlansCounter, weightBuffers, filter)
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
        return ((partId == 1 && cascadeType == CascadeType::Beginning) ||
                (partId == 2 && cascadeType == CascadeType::Middle) ||
                (partId == 3 && cascadeType == CascadeType::End));
    };

    auto pInput = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pA = std::make_unique<NoWeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[0], filter);
    auto pB = std::make_unique<NoWeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[1], filter);
    auto pC = std::make_unique<NoWeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[2], filter);

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
    const DebuggingContext debuggingContext(&compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Combiner combiner(graph, hwCaps, estOpt, debuggingContext);
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

    auto pInput = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pA = std::make_unique<NoWeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[0], filter);
    auto pB = std::make_unique<WeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[1], filter);
    auto pC = std::make_unique<NoWeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[2], filter);

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
    const DebuggingContext debuggingContext(&compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Combiner combiner(graph, hwCaps, estOpt, debuggingContext);
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

    auto pInput = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pA     = std::make_unique<WeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[0], filter);
    auto pB = std::make_unique<NoWeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[1], filter);
    auto pC = std::make_unique<WeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[2], filter);

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
    const DebuggingContext debuggingContext(&compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Combiner combiner(graph, hwCaps, estOpt, debuggingContext);
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
    auto pA = std::make_unique<NoWeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[0], filter);
    auto pB = std::make_unique<WeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[1], filter);
    auto pC = std::make_unique<WeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[2], filter);

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
    const DebuggingContext debuggingContext(&compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Combiner combiner(graph, hwCaps, estOpt, debuggingContext);
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
    auto pA = std::make_unique<NoWeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[0], filter);
    auto pB = std::make_unique<WeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[1], filter);
    auto pC = std::make_unique<WeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[2], filter);
    auto pD = std::make_unique<NoWeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[3], filter);

    BasePart& partA = *pA;

    PartId partInputId = pInput->GetPartId();
    PartId partAId     = pA->GetPartId();
    PartId partBId     = pB->GetPartId();
    PartId partCId     = pC->GetPartId();
    PartId partDId     = pD->GetPartId();
    parts.push_back(std::move(pInput));
    parts.push_back(std::move(pA));
    parts.push_back(std::move(pB));
    parts.push_back(std::move(pC));
    parts.push_back(std::move(pD));

    PartOutputSlot partInputOutputSlot0 = { partInputId, 0 };

    PartInputSlot partAInputSlot0   = { partAId, 0 };
    PartOutputSlot partAOutputSlot0 = { partAId, 0 };

    PartInputSlot partBInputSlot0   = { partBId, 0 };
    PartOutputSlot partBOutputSlot0 = { partBId, 0 };

    PartInputSlot partCInputSlot0   = { partCId, 0 };
    PartOutputSlot partCOutputSlot0 = { partCId, 0 };

    PartInputSlot partDInputSlot0 = { partDId, 0 };

    connections[partAInputSlot0] = partInputOutputSlot0;
    connections[partBInputSlot0] = partAOutputSlot0;
    connections[partCInputSlot0] = partBOutputSlot0;
    connections[partDInputSlot0] = partCOutputSlot0;

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(&compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Combiner combiner(graph, hwCaps, estOpt, debuggingContext);
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
    glueA_B.m_Graph.GetBuffers().back()->m_BufferType = BufferType::Intermediate;
    glueA_B.m_Graph.GetBuffers().back()->m_DebugTag   = "DramBuffer";
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
    const DebuggingContext debuggingContext(&compOpt.m_DebugInfo);

    // Plan A
    Plan planA;
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planA.m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputSramA";
    planA.m_OutputMappings                          = { { planA.m_OpGraph.GetBuffers()[0], partAOutputSlot } };
    auto dummyOpA                                   = std::make_unique<DummyOp>();
    dummyOpA->m_DebugTag                            = "DummyA";
    planA.m_OpGraph.AddOp(std::move(dummyOpA));

    // Plan B
    Plan planB;
    planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planB.m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputSramB";
    planB.m_OutputMappings                          = { { planB.m_OpGraph.GetBuffers()[0], partBOutputSlot } };
    auto dummyOpB                                   = std::make_unique<DummyOp>();
    dummyOpB->m_DebugTag                            = "DummyB";
    planB.m_OpGraph.AddOp(std::move(dummyOpB));

    // Glue between A and C
    Glue glueA_C;
    glueA_C.m_Graph.AddOp(std::make_unique<DmaOp>());
    glueA_C.m_Graph.GetOps()[0]->m_DebugTag = "GlueAC_Dma";
    glueA_C.m_InputSlot                     = { glueA_C.m_Graph.GetOps()[0], 0 };
    glueA_C.m_Output.push_back(glueA_C.m_Graph.GetOps()[0]);

    // Glue between B and C
    Glue glueB_C;
    glueB_C.m_Graph.AddOp(std::make_unique<DmaOp>());
    glueB_C.m_Graph.GetOps()[0]->m_DebugTag = "GlueBC_Dma";
    glueB_C.m_InputSlot                     = { glueB_C.m_Graph.GetOps()[0], 0 };
    glueB_C.m_Output.push_back(glueB_C.m_Graph.GetOps()[0]);

    // Plan C
    Plan planC;
    planC.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planC.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Intermediate;
    planC.m_OpGraph.GetBuffers().back()->m_DebugTag   = "Input0DramC";
    planC.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planC.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Intermediate;
    planC.m_OpGraph.GetBuffers().back()->m_DebugTag   = "Input1DramC";
    planC.m_InputMappings                             = { { planC.m_OpGraph.GetBuffers()[0], partCInputSlot0 },
                              { planC.m_OpGraph.GetBuffers()[1], partCInputSlot1 } };
    auto dummyOpC                                     = std::make_unique<DummyOp>();
    dummyOpC->m_DebugTag                              = "DummyC";
    planC.m_OpGraph.AddOp(std::move(dummyOpC));

    // Create Combination with all the plans and glues
    Combination comb;

    Elem elemA = { std::make_shared<Plan>(std::move(planA)), { { partCInputSlot0, { &glueA_C, true } } } };
    Elem elemB = { std::make_shared<Plan>(std::move(planB)), { { partCInputSlot1, { &glueB_C, true } } } };
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
        std::ofstream stream("GetOpGraphForDfsMISOSramsToDrams.dot");
        SaveCombinationToDot(comb, graph, stream, DetailLevel::High);
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
    const DebuggingContext debuggingContext(&compOpt.m_DebugInfo);

    // Plan A
    Plan planA;
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB,
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
    planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planB.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Intermediate;
    planB.m_OpGraph.GetBuffers().back()->m_DebugTag   = "OutputSramB";
    planB.m_OutputMappings                            = { { planB.m_OpGraph.GetBuffers()[0], partBOutputSlot } };
    auto dummyOpB                                     = std::make_unique<DummyOp>();
    dummyOpB->m_DebugTag                              = "DummyB";
    planB.m_OpGraph.AddOp(std::move(dummyOpB));
    planB.m_OpGraph.SetProducer(planB.m_OpGraph.GetBuffers().back(), planB.m_OpGraph.GetOps().back());

    // Glue between A and C
    Glue glueA_C;
    glueA_C.m_Graph.AddOp(std::make_unique<DmaOp>());
    glueA_C.m_Graph.GetOps()[0]->m_DebugTag = "GlueAC_Dma";
    glueA_C.m_InputSlot                     = { glueA_C.m_Graph.GetOps()[0], 0 };
    glueA_C.m_Output.push_back(glueA_C.m_Graph.GetOps()[0]);

    // Glue between B and C
    Glue glueB_C;
    glueB_C.m_Graph.AddOp(std::make_unique<DmaOp>());
    glueB_C.m_Graph.GetOps()[0]->m_DebugTag = "GlueBC_Dma";
    glueB_C.m_InputSlot                     = { glueB_C.m_Graph.GetOps()[0], 0 };
    glueB_C.m_Output.push_back(glueB_C.m_Graph.GetOps()[0]);

    // Plan C
    Plan planC;
    planC.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planC.m_OpGraph.GetBuffers().back()->m_DebugTag = "Input0DramC";
    planC.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planC.m_OpGraph.GetBuffers().back()->m_DebugTag = "Input1DramC";
    planC.m_InputMappings                           = { { planC.m_OpGraph.GetBuffers()[0], partCInputSlot0 },
                              { planC.m_OpGraph.GetBuffers()[1], partCInputSlot1 } };
    auto dummyOpC                                   = std::make_unique<DummyOp>();
    dummyOpC->m_DebugTag                            = "DummyC";
    planC.m_OpGraph.AddOp(std::move(dummyOpC));

    // Create Combination with all the plans and glues
    Combination comb;

    Elem elemA = { std::make_shared<Plan>(std::move(planA)), { { partCInputSlot0, { &glueA_C, true } } } };
    Elem elemB = { std::make_shared<Plan>(std::move(planB)), { { partCInputSlot1, { &glueB_C, true } } } };
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
        std::ofstream stream("GetOpGraphForDfsMISOSramsToDrams.dot");
        SaveCombinationToDot(comb, graph, stream, DetailLevel::High);
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
    // The correct order of should be:
    // (1) opA (2) op B  (3) glueAC_DMA (4) glueBC_DMA (5) op C    REQUIRE(combOpGraph.GetBuffers().size() == 4);
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
    planD.m_OpGraph.GetBuffers().back()->m_DebugTag   = "InputDramD";
    planD.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Intermediate;
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
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 0, QuantizationInfo()));
    planA.m_OpGraph.GetBuffers().back()->m_DebugTag = "PlanA_Buffer0";
    planA.m_OutputMappings                          = { { planA.m_OpGraph.GetBuffers()[0], partAOutputSlot0 } };

    // Plan B
    Plan planB;
    planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planB.m_OpGraph.GetBuffers().back()->m_DebugTag = "PlanB_Buffer0";
    planB.m_InputMappings                           = { { planB.m_OpGraph.GetBuffers()[0], partBInputSlot0 } };
    planB.m_OutputMappings                          = { { planB.m_OpGraph.GetBuffers()[0], partBOutputSlot0 } };

    // Glue between B and C
    Glue glueB_C;
    glueB_C.m_Graph.AddOp(std::make_unique<DmaOp>());
    glueB_C.m_Graph.GetOps()[0]->m_DebugTag = "GlueBC_Dma";
    glueB_C.m_InputSlot                     = { glueB_C.m_Graph.GetOps()[0], 0 };
    glueB_C.m_Output.push_back(glueB_C.m_Graph.GetOps()[0]);

    // Plan C
    Plan planC;
    planC.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    planC.m_OpGraph.GetBuffers().back()->m_DebugTag = "PlanC_Buffer0";
    planC.m_InputMappings                           = { { planC.m_OpGraph.GetBuffers()[0], partCInputSlot0 } };

    // Create Combination with all the plans and glues
    Combination comb;

    Elem elemA = { std::make_shared<Plan>(std::move(planA)), {} };
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
    planA.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Input;
    planA.m_OpGraph.GetBuffers().back()->m_DebugTag   = "InputDram";
    planA.m_OutputMappings                            = { { planA.m_OpGraph.GetBuffers()[0], partAOutputSlot0 } };

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
    planF.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Output;
    planF.m_OpGraph.GetBuffers().back()->m_DebugTag   = "OutputDram1";
    planF.m_InputMappings                             = { { planF.m_OpGraph.GetBuffers()[0], partFInputSlot0 } };

    // Part consisting of node G
    Plan planG;
    planG.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 0, QuantizationInfo()));
    planG.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Output;
    planG.m_OpGraph.GetBuffers().back()->m_DebugTag   = "OutputDram2";
    planG.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 0, QuantizationInfo()));
    planG.m_OpGraph.GetBuffers().back()->m_BufferType = BufferType::Output;
    planG.m_OpGraph.GetBuffers().back()->m_DebugTag   = "OutputDram3";
    planG.m_InputMappings                             = { { planG.m_OpGraph.GetBuffers()[0], partGInputSlot0 },
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

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("GetOpGraphForCombination Input.dot");
        SaveCombinationToDot(comb, graph, stream, DetailLevel::High);
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

    auto mockBuffer = std::make_unique<Buffer>(Lifetime::Cascade, Location::Sram, CascadingBufferFormat::NHWCB,
                                               TensorShape{ 1, 32, 16, 1024 }, TensorShape{ 1, 32, 16, 1024 },
                                               TraversalOrder::Xyz, ifmSize, QuantizationInfo());
    mockBuffer.get()->m_Offset = 0;

    Plan planA;
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Cascade, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 32, 16, 1024 }, TensorShape{ 1, 32, 16, 1024 },
                                                       TraversalOrder::Xyz, ifmSize, QuantizationInfo()));
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 32, 16, 1024 }, TensorShape{ 1, 4, 16, 1024 },
                                                       TraversalOrder::Xyz, ofmSize, QuantizationInfo()));

    planA.m_OpGraph.AddOp(
        std::make_unique<MceOp>(Lifetime::Cascade, MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
                                BlockConfig{ 8u, 8u }, TensorShape{ 1, 32, 16, 1024 }, TensorShape{ 1, 4, 16, 1024 },
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

    // SRAM has enough space for ofm and the plan does not have a PLE kernel
    SramAllocator alloc1 = alloc;
    REQUIRE(combiner.IsPlanAllocated(alloc1, planA, pleOps, mockBuffer.get(), StatsType::ContinueSection) == true);
    REQUIRE(pleOps.size() == 0);

    // Adding a passthrough PLE kernel to the plan
    // The PleKernelId is expected to be PASSTHROUGH_8x8_2
    auto op =
        std::make_unique<PleOp>(Lifetime::Cascade, ethosn::command_stream::PleOperation::PASSTHROUGH,
                                BlockConfig{ 8u, 8u }, 1, std::vector<TensorShape>{ TensorShape{ 1, 4, 16, 1024 } },
                                TensorShape{ 1, 4, 16, 1024 }, ethosn::command_stream::DataType::U8, true);

    numMemoryStripes.m_Output = 1;
    auto outBufferAndPleOp    = AddPleToOpGraph(planA.m_OpGraph, Lifetime::Cascade, TraversalOrder::Xyz,
                                             TensorShape{ 1, 4, 16, 1024 }, numMemoryStripes, std::move(op),
                                             TensorShape{ 1, 4, 16, 1024 }, QuantizationInfo(), operationIds);

    PleOp* ptrPleOp = dynamic_cast<PleOp*>(planA.m_OpGraph.GetOp(1));
    if (ptrPleOp == nullptr)
    {
        REQUIRE(false);
    }

    // With a PLE kernel, the plan can still fit into SRAM and there is a need to Load the Kernel
    SramAllocator alloc2 = alloc;
    REQUIRE(combiner.IsPlanAllocated(alloc2, planA, pleOps, mockBuffer.get(), StatsType::ContinueSection) == true);
    REQUIRE(pleOps.size() == 1);
    if (ptrPleOp != nullptr)
    {
        REQUIRE(ptrPleOp->m_LoadKernel == true);
    }

    // PLE kernel used previously has different block height
    // The plan is expected to be fit into SRAM and there is a need to Load the Kernel
    SramAllocator alloc3   = alloc;
    PleKernelId pleKernel1 = PleKernelId::PASSTHROUGH_8X16_1;
    PleOperations pleOps1  = { { pleKernel1, 0 } };
    REQUIRE(combiner.IsPlanAllocated(alloc3, planA, pleOps1, mockBuffer.get(), StatsType::ContinueSection) == true);
    REQUIRE(pleOps1.size() == 2);
    if (ptrPleOp != nullptr)
    {
        REQUIRE(ptrPleOp->m_LoadKernel == true);
    }

    // PLE kernel passthrough is already used previously in the same
    // section, the plan is expected to be fit into SRAM and no need to Load the Kernel
    SramAllocator alloc4   = alloc;
    PleKernelId pleKernel2 = PleKernelId::PASSTHROUGH_8X8_2;
    PleOperations pleOps2  = { { pleKernel2, 0 } };
    REQUIRE(combiner.IsPlanAllocated(alloc4, planA, pleOps2, mockBuffer.get(), StatsType::ContinueSection) == true);
    REQUIRE(pleOps2.size() == 1);
    if (ptrPleOp != nullptr)
    {
        REQUIRE(ptrPleOp->m_LoadKernel == false);
    }

    SramAllocator alloc5 = alloc;
    // Allocate memory where the plan and the allocated memory exceeds the SRAM Size
    uint32_t planSize          = ofmSize + planA.m_OpGraph.GetBuffers()[2]->m_SizeInBytes + hwCaps.GetMaxPleSize();
    uint32_t remainingSramSize = hwCaps.GetTotalSramSize() - planSize;
    alloc5.Allocate(0, ((remainingSramSize + hwCaps.GetNumberOfSrams()) / hwCaps.GetNumberOfSrams()),
                    AllocationPreference::Start);
    PleOperations pleOps3 = {};
    REQUIRE(combiner.IsPlanAllocated(alloc5, planA, pleOps3, mockBuffer.get(), StatsType::ContinueSection) == false);
    REQUIRE(pleOps3.size() == 0);
    if (ptrPleOp != nullptr)
    {
        REQUIRE(ptrPleOp->m_LoadKernel == true);
    }

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
        const DebuggingContext debuggingContext(&compOpt.m_DebugInfo);
        const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

        Combiner combiner(graph, hwCaps, estOpt, debuggingContext);
        SramAllocator alloc(hwCaps.GetTotalSramSize() / hwCaps.GetNumberOfSrams());
        uint32_t currentSramOffset = 0;
        PleOperations pleOps       = {};

        Plan planA;

        const uint32_t inputBufferSize  = 512;
        const uint32_t outputBufferSize = 512;

        planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(
            Lifetime::Cascade, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 8, 8, 8 },
            TensorShape{ 1, 8, 8, 8 }, TraversalOrder::Xyz, inputBufferSize, QuantizationInfo()));
        planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(
            Lifetime::Cascade, Location::PleInputSram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 8, 8, 8 },
            TensorShape{ 1, 8, 8, 8 }, TraversalOrder::Xyz, outputBufferSize, QuantizationInfo()));

        planA.m_OpGraph.AddOp(
            std::make_unique<MceOp>(Lifetime::Cascade, MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
                                    BlockConfig{ 8u, 8u }, TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 8, 8, 8 },
                                    TensorShape{ 1, 8, 8, 8 }, TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));

        planA.m_InputMappings  = { { planA.m_OpGraph.GetBuffers()[0], partAInputSlot } };
        planA.m_OutputMappings = { { planA.m_OpGraph.GetBuffers()[1], partAOutputSlot } };

        WHEN("Lonely section with a plan that has no Ple Op")
        {
            REQUIRE(planA.m_OpGraph.GetBuffers()[0]->m_Offset.has_value() == false);
            REQUIRE(planA.m_OpGraph.GetBuffers()[1]->m_Offset.has_value() == false);
            REQUIRE(combiner.IsPlanAllocated(alloc, planA, pleOps, nullptr, StatsType::SinglePartSection) == true);
            REQUIRE(planA.m_OpGraph.GetBuffers()[0]->m_Offset.has_value() == true);
            REQUIRE(planA.m_OpGraph.GetBuffers()[0]->m_Offset.value() == currentSramOffset);
            currentSramOffset = planA.m_OpGraph.GetBuffers()[0]->m_SizeInBytes / hwCaps.GetNumberOfSrams();
            REQUIRE(planA.m_OpGraph.GetBuffers()[1]->m_Offset.has_value() == false);
            REQUIRE(pleOps.size() == 0);
        }

        WHEN("Lonely section with a plan that has Ple Op")
        {
            // Adding a passthrough PLE kernel to the plan
            // The PleKernelId is expected to be PASSTHROUGH_8x8_2
            auto op =
                std::make_unique<PleOp>(Lifetime::Cascade, ethosn::command_stream::PleOperation::PASSTHROUGH,
                                        BlockConfig{ 8u, 8u }, 1, std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 8 } },
                                        TensorShape{ 1, 8, 8, 8 }, ethosn::command_stream::DataType::U8, true);
            numMemoryStripes.m_Output = 1;
            auto outBufferAndPleOp    = AddPleToOpGraph(planA.m_OpGraph, Lifetime::Cascade, TraversalOrder::Xyz,
                                                     TensorShape{ 1, 8, 8, 8 }, numMemoryStripes, std::move(op),
                                                     TensorShape{ 1, 8, 8, 8 }, QuantizationInfo(), operationIds);

            PleOp* ptrPleOp = dynamic_cast<PleOp*>(planA.m_OpGraph.GetOp(1));
            if (ptrPleOp == nullptr)
            {
                REQUIRE(false);
            }

            if (ptrPleOp != nullptr)
            {
                REQUIRE(ptrPleOp->m_Offset.has_value() == false);
            }

            REQUIRE(planA.m_OpGraph.GetBuffers()[0]->m_Offset.has_value() == false);
            REQUIRE(planA.m_OpGraph.GetBuffers()[1]->m_Offset.has_value() == false);
            REQUIRE(planA.m_OpGraph.GetBuffers()[2]->m_Offset.has_value() == false);
            REQUIRE(combiner.IsPlanAllocated(alloc, planA, pleOps, nullptr, StatsType::SinglePartSection) == true);

            if (ptrPleOp != nullptr)
            {
                REQUIRE(ptrPleOp->m_Offset.has_value() == true);
                REQUIRE(ptrPleOp->m_Offset.value() == currentSramOffset);
            }

            currentSramOffset += hwCaps.GetMaxPleSize() / hwCaps.GetNumberOfSrams();
            REQUIRE(planA.m_OpGraph.GetBuffers()[0]->m_Offset.has_value() == true);
            REQUIRE(planA.m_OpGraph.GetBuffers()[0]->m_Offset.value() == currentSramOffset);
            currentSramOffset += planA.m_OpGraph.GetBuffers()[0]->m_SizeInBytes / hwCaps.GetNumberOfSrams();
            // Note that Buffer 1 is the output from MceOp where its location is in PleInputSRAM not SRAM
            REQUIRE(planA.m_OpGraph.GetBuffers()[1]->m_Offset.has_value() == false);
            REQUIRE(planA.m_OpGraph.GetBuffers()[2]->m_Offset.has_value() == true);
            REQUIRE(planA.m_OpGraph.GetBuffers()[2]->m_Offset.value() == currentSramOffset);
            currentSramOffset += planA.m_OpGraph.GetBuffers()[2]->m_SizeInBytes / hwCaps.GetNumberOfSrams();
            REQUIRE(pleOps.size() == 1);

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
        const DebuggingContext debuggingContext(&compOpt.m_DebugInfo);
        const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

        Combiner combiner(graph, hwCaps, estOpt, debuggingContext);
        SramAllocator alloc(hwCaps.GetTotalSramSize() / hwCaps.GetNumberOfSrams());
        uint32_t currentSramOffset = 0;
        PleOperations pleOps       = {};

        Plan planA;

        const uint32_t inputBufferSize  = 512;
        const uint32_t outputBufferSize = 512;

        planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(
            Lifetime::Cascade, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 8, 8, 8 },
            TensorShape{ 1, 8, 8, 8 }, TraversalOrder::Xyz, inputBufferSize, QuantizationInfo()));
        planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(
            Lifetime::Cascade, Location::PleInputSram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 8, 8, 8 },
            TensorShape{ 1, 8, 8, 8 }, TraversalOrder::Xyz, outputBufferSize, QuantizationInfo()));

        planA.m_OpGraph.AddOp(
            std::make_unique<MceOp>(Lifetime::Cascade, MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
                                    BlockConfig{ 8u, 8u }, TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 8, 8, 8 },
                                    TensorShape{ 1, 8, 8, 8 }, TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));

        planA.m_InputMappings  = { { planA.m_OpGraph.GetBuffers()[0], partAInputSlot } };
        planA.m_OutputMappings = { { planA.m_OpGraph.GetBuffers()[1], partAOutputSlot } };

        WHEN("Starting the section with the first plan that has no Ple Op")
        {
            REQUIRE(planA.m_OpGraph.GetBuffers()[0]->m_Offset.has_value() == false);
            REQUIRE(planA.m_OpGraph.GetBuffers()[1]->m_Offset.has_value() == false);
            REQUIRE(combiner.IsPlanAllocated(alloc, planA, pleOps, nullptr, StatsType::StartSection) == true);
            REQUIRE(planA.m_OpGraph.GetBuffers()[0]->m_Offset.has_value() == true);
            REQUIRE(planA.m_OpGraph.GetBuffers()[0]->m_Offset.value() == currentSramOffset);
            currentSramOffset = planA.m_OpGraph.GetBuffers()[0]->m_SizeInBytes / hwCaps.GetNumberOfSrams();
            REQUIRE(planA.m_OpGraph.GetBuffers()[1]->m_Offset.has_value() == false);
            REQUIRE(pleOps.size() == 0);
        }

        WHEN("Starting the section with the first plan that has Ple Op")
        {
            // Adding a passthrough PLE kernel to the plan
            // The PleKernelId is expected to be PASSTHROUGH_8x8_2
            auto op =
                std::make_unique<PleOp>(Lifetime::Cascade, ethosn::command_stream::PleOperation::PASSTHROUGH,
                                        BlockConfig{ 8u, 8u }, 1, std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 8 } },
                                        TensorShape{ 1, 8, 8, 8 }, ethosn::command_stream::DataType::U8, true);
            numMemoryStripes.m_Output = 1;
            auto outBufferAndPleOp    = AddPleToOpGraph(planA.m_OpGraph, Lifetime::Cascade, TraversalOrder::Xyz,
                                                     TensorShape{ 1, 8, 8, 8 }, numMemoryStripes, std::move(op),
                                                     TensorShape{ 1, 8, 8, 8 }, QuantizationInfo(), operationIds);

            PleOp* ptrPleOp = dynamic_cast<PleOp*>(planA.m_OpGraph.GetOp(1));
            if (ptrPleOp == nullptr)
            {
                REQUIRE(false);
            }

            if (ptrPleOp != nullptr)
            {
                REQUIRE(ptrPleOp->m_Offset.has_value() == false);
            }

            REQUIRE(planA.m_OpGraph.GetBuffers()[0]->m_Offset.has_value() == false);
            REQUIRE(planA.m_OpGraph.GetBuffers()[1]->m_Offset.has_value() == false);
            REQUIRE(planA.m_OpGraph.GetBuffers()[2]->m_Offset.has_value() == false);
            REQUIRE(combiner.IsPlanAllocated(alloc, planA, pleOps, nullptr, StatsType::StartSection) == true);

            if (ptrPleOp != nullptr)
            {
                REQUIRE(ptrPleOp->m_Offset.has_value() == true);
                REQUIRE(ptrPleOp->m_Offset.value() == currentSramOffset);
            }

            currentSramOffset += hwCaps.GetMaxPleSize() / hwCaps.GetNumberOfSrams();
            REQUIRE(planA.m_OpGraph.GetBuffers()[0]->m_Offset.has_value() == true);
            REQUIRE(planA.m_OpGraph.GetBuffers()[0]->m_Offset.value() == currentSramOffset);
            currentSramOffset += planA.m_OpGraph.GetBuffers()[0]->m_SizeInBytes / hwCaps.GetNumberOfSrams();
            // Note that Buffer 1 is the output from MceOp where its location is in PleInputSRAM not SRAM
            REQUIRE(planA.m_OpGraph.GetBuffers()[1]->m_Offset.has_value() == false);
            REQUIRE(planA.m_OpGraph.GetBuffers()[2]->m_Offset.has_value() == true);
            REQUIRE(planA.m_OpGraph.GetBuffers()[2]->m_Offset.value() == currentSramOffset);
            currentSramOffset += planA.m_OpGraph.GetBuffers()[2]->m_SizeInBytes / hwCaps.GetNumberOfSrams();
            REQUIRE(pleOps.size() == 1);

            Plan planB;

            const uint32_t InputBufferSize  = 512;
            const uint32_t OutputBufferSize = 512;

            planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(
                Lifetime::Cascade, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 8, 8, 8 },
                TensorShape{ 1, 8, 8, 8 }, TraversalOrder::Xyz, InputBufferSize, QuantizationInfo()));

            planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(
                Lifetime::Cascade, Location::PleInputSram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 8, 8, 8 },
                TensorShape{ 1, 8, 8, 8 }, TraversalOrder::Xyz, OutputBufferSize, QuantizationInfo()));

            planB.m_OpGraph.AddOp(
                std::make_unique<MceOp>(Lifetime::Cascade, MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
                                        BlockConfig{ 8u, 8u }, TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 8, 8, 8 },
                                        TensorShape{ 1, 8, 8, 8 }, TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));

            planB.m_InputMappings  = { { planB.m_OpGraph.GetBuffers()[0], partBInputSlot } };
            planB.m_OutputMappings = { { planB.m_OpGraph.GetBuffers()[1], partBOutputSlot } };

            WHEN("Continuing the section with the second plan that has no Ple Op")
            {
                REQUIRE(planB.m_OpGraph.GetBuffers()[0]->m_Offset.has_value() == false);
                REQUIRE(planB.m_OpGraph.GetBuffers()[1]->m_Offset.has_value() == false);
                REQUIRE(combiner.IsPlanAllocated(alloc, planB, pleOps, planA.m_OpGraph.GetBuffers()[2],
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
                auto op = std::make_unique<PleOp>(
                    Lifetime::Cascade, ethosn::command_stream::PleOperation::PASSTHROUGH, BlockConfig{ 8u, 8u }, 1,
                    std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 8 } }, TensorShape{ 1, 8, 8, 8 },
                    ethosn::command_stream::DataType::U8, true);
                numMemoryStripes.m_Output = 1;
                auto outBufferAndPleOp    = AddPleToOpGraph(planB.m_OpGraph, Lifetime::Cascade, TraversalOrder::Xyz,
                                                         TensorShape{ 1, 8, 8, 8 }, numMemoryStripes, std::move(op),
                                                         TensorShape{ 1, 8, 8, 8 }, QuantizationInfo(), operationIds);

                PleOp* ptrPleOpA = dynamic_cast<PleOp*>(planA.m_OpGraph.GetOp(1));
                PleOp* ptrPleOpB = dynamic_cast<PleOp*>(planB.m_OpGraph.GetOp(1));
                if (ptrPleOpA == nullptr || ptrPleOpB == nullptr)
                {
                    REQUIRE(false);
                }

                REQUIRE(ptrPleOpB->m_LoadKernel == true);
                REQUIRE(combiner.IsPlanAllocated(alloc, planB, pleOps, planA.m_OpGraph.GetBuffers()[2],
                                                 StatsType::ContinueSection) == true);
                REQUIRE(pleOps.size() == 1);

                if (ptrPleOpA != nullptr && ptrPleOpB != nullptr)
                {
                    REQUIRE(ptrPleOpB->m_LoadKernel == false);
                    REQUIRE(ptrPleOpB->m_Offset == ptrPleOpA->m_Offset);
                }

                REQUIRE(planB.m_OpGraph.GetBuffers()[2]->m_Offset.value() == currentSramOffset);
                currentSramOffset += planB.m_OpGraph.GetBuffers()[2]->m_SizeInBytes / hwCaps.GetNumberOfSrams();

                ETHOSN_UNUSED(outBufferAndPleOp);
            }

            WHEN("Continuing the section with the second plan that has Ple Op not already loaded")
            {
                // Adding a passthrough PLE kernel to the plan
                auto op = std::make_unique<PleOp>(
                    Lifetime::Cascade, ethosn::command_stream::PleOperation::PASSTHROUGH, BlockConfig{ 16u, 16u }, 1,
                    std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 8 } }, TensorShape{ 1, 8, 8, 8 },
                    ethosn::command_stream::DataType::U8, true);

                numMemoryStripes.m_Output = 1;

                auto outBufferAndPleOp = AddPleToOpGraph(planB.m_OpGraph, Lifetime::Cascade, TraversalOrder::Xyz,
                                                         TensorShape{ 1, 8, 8, 8 }, numMemoryStripes, std::move(op),
                                                         TensorShape{ 1, 8, 8, 8 }, QuantizationInfo(), operationIds);

                PleOp* ptrPleOpA = dynamic_cast<PleOp*>(planA.m_OpGraph.GetOp(1));
                PleOp* ptrPleOpB = dynamic_cast<PleOp*>(planB.m_OpGraph.GetOp(1));
                if (ptrPleOpA == nullptr || ptrPleOpB == nullptr)
                {
                    REQUIRE(false);
                }

                if (ptrPleOpB != nullptr)
                {
                    REQUIRE(ptrPleOpB->m_LoadKernel == true);
                }

                REQUIRE(combiner.IsPlanAllocated(alloc, planB, pleOps, planA.m_OpGraph.GetBuffers()[2],
                                                 StatsType::ContinueSection) == true);
                REQUIRE(pleOps.size() == 2);

                if (ptrPleOpB != nullptr)
                {
                    REQUIRE(ptrPleOpB->m_LoadKernel == true);
                    REQUIRE(ptrPleOpB->m_Offset == currentSramOffset);
                }

                currentSramOffset += hwCaps.GetMaxPleSize() / hwCaps.GetNumberOfSrams();
                REQUIRE(planB.m_OpGraph.GetBuffers()[2]->m_Offset.value() == currentSramOffset);
                currentSramOffset += planB.m_OpGraph.GetBuffers()[2]->m_SizeInBytes / hwCaps.GetNumberOfSrams();

                Plan planC;

                const uint32_t InputBufferSize  = 512;
                const uint32_t OutputBufferSize = 512;

                planC.m_OpGraph.AddBuffer(std::make_unique<Buffer>(
                    Lifetime::Cascade, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 8, 8, 8 },
                    TensorShape{ 1, 8, 8, 8 }, TraversalOrder::Xyz, InputBufferSize, QuantizationInfo()));

                planC.m_OpGraph.AddBuffer(std::make_unique<Buffer>(
                    Lifetime::Cascade, Location::PleInputSram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 8, 8, 8 },
                    TensorShape{ 1, 8, 8, 8 }, TraversalOrder::Xyz, OutputBufferSize, QuantizationInfo()));

                planC.m_OpGraph.AddOp(
                    std::make_unique<MceOp>(Lifetime::Cascade, MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
                                            BlockConfig{ 8u, 8u }, TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 8, 8, 8 },
                                            TensorShape{ 1, 8, 8, 8 }, TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));

                planC.m_InputMappings  = { { planC.m_OpGraph.GetBuffers()[0], partCInputSlot } };
                planC.m_OutputMappings = { { planC.m_OpGraph.GetBuffers()[1], partCOutputSlot } };

                WHEN("Ending the section with the third plan that has no Ple Op")
                {
                    REQUIRE(planC.m_OpGraph.GetBuffers()[0]->m_Offset.has_value() == false);
                    REQUIRE(planC.m_OpGraph.GetBuffers()[1]->m_Offset.has_value() == false);
                    REQUIRE(combiner.IsPlanAllocated(alloc, planC, pleOps, planB.m_OpGraph.GetBuffers()[2],
                                                     StatsType::EndSection) == true);
                    REQUIRE(planC.m_OpGraph.GetBuffers()[0]->m_Offset.has_value() == true);
                    REQUIRE(planC.m_OpGraph.GetBuffers()[0]->m_Offset.value() ==
                            planB.m_OpGraph.GetBuffers()[2]->m_Offset.value());
                    REQUIRE(planC.m_OpGraph.GetBuffers()[1]->m_Offset.has_value() == false);
                }

                WHEN("Ending the section with the third plan that has already loaded Ple Op")
                {
                    // Adding a passthrough PLE kernel to the plan
                    auto op = std::make_unique<PleOp>(
                        Lifetime::Cascade, ethosn::command_stream::PleOperation::PASSTHROUGH, BlockConfig{ 16u, 16u },
                        1, std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 8 } }, TensorShape{ 1, 8, 8, 8 },
                        ethosn::command_stream::DataType::U8, true);
                    numMemoryStripes.m_Output = 1;
                    auto outBufferAndPleOp    = AddPleToOpGraph(
                        planC.m_OpGraph, Lifetime::Cascade, TraversalOrder::Xyz, TensorShape{ 1, 8, 8, 8 },
                        numMemoryStripes, std::move(op), TensorShape{ 1, 8, 8, 8 }, QuantizationInfo(), operationIds);

                    PleOp* ptrPleOpA = dynamic_cast<PleOp*>(planA.m_OpGraph.GetOp(1));
                    PleOp* ptrPleOpB = dynamic_cast<PleOp*>(planB.m_OpGraph.GetOp(1));
                    PleOp* ptrPleOpC = dynamic_cast<PleOp*>(planC.m_OpGraph.GetOp(1));
                    if (ptrPleOpA == nullptr || ptrPleOpB == nullptr || ptrPleOpC == nullptr)
                    {
                        REQUIRE(false);
                    }

                    if (ptrPleOpC != nullptr)
                    {
                        REQUIRE(ptrPleOpC->m_LoadKernel == true);
                    }

                    REQUIRE(combiner.IsPlanAllocated(alloc, planC, pleOps, planB.m_OpGraph.GetBuffers()[2],
                                                     StatsType::EndSection) == true);
                    REQUIRE(pleOps.size() == 2);

                    if (ptrPleOpC != nullptr)
                    {
                        REQUIRE(ptrPleOpC->m_LoadKernel == false);
                    }

                    if (ptrPleOpC != nullptr && ptrPleOpB != nullptr)
                    {
                        REQUIRE(ptrPleOpC->m_Offset == ptrPleOpB->m_Offset);
                    }

                    REQUIRE(planC.m_OpGraph.GetBuffers()[2]->m_Offset.value() == currentSramOffset);
                    currentSramOffset += planC.m_OpGraph.GetBuffers()[2]->m_SizeInBytes / hwCaps.GetNumberOfSrams();

                    ETHOSN_UNUSED(outBufferAndPleOp);
                }

                WHEN("Ending the section with the third plan that has Ple Op not already loaded")
                {
                    // Adding a passthrough PLE kernel to the plan
                    auto op = std::make_unique<PleOp>(
                        Lifetime::Cascade, ethosn::command_stream::PleOperation::PASSTHROUGH, BlockConfig{ 8u, 32u }, 1,
                        std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 8 } }, TensorShape{ 1, 8, 8, 8 },
                        ethosn::command_stream::DataType::U8, true);

                    numMemoryStripes.m_Output = 1;

                    auto outBufferAndPleOp = AddPleToOpGraph(
                        planC.m_OpGraph, Lifetime::Cascade, TraversalOrder::Xyz, TensorShape{ 1, 8, 8, 8 },
                        numMemoryStripes, std::move(op), TensorShape{ 1, 8, 8, 8 }, QuantizationInfo(), operationIds);

                    PleOp* ptrPleOpB = dynamic_cast<PleOp*>(planB.m_OpGraph.GetOp(1));
                    PleOp* ptrPleOpC = dynamic_cast<PleOp*>(planC.m_OpGraph.GetOp(1));
                    if (ptrPleOpB == nullptr || ptrPleOpC == nullptr)
                    {
                        REQUIRE(false);
                    }

                    if (ptrPleOpC != nullptr)
                    {
                        REQUIRE(ptrPleOpC->m_LoadKernel == true);
                    }

                    REQUIRE(combiner.IsPlanAllocated(alloc, planC, pleOps, planB.m_OpGraph.GetBuffers()[2],
                                                     StatsType::EndSection) == true);
                    REQUIRE(pleOps.size() == 3);

                    if (ptrPleOpB != nullptr)
                    {
                        REQUIRE(ptrPleOpB->m_LoadKernel == true);
                    }

                    if (ptrPleOpC != nullptr)
                    {
                        REQUIRE(ptrPleOpC->m_Offset == currentSramOffset);
                    }

                    currentSramOffset += hwCaps.GetMaxPleSize() / hwCaps.GetNumberOfSrams();
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
    const DebuggingContext debuggingContext(&compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Plan planA;
    planA.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                       TraversalOrder::Xyz, 0, QuantizationInfo()));
    MceOp mceOp(Lifetime::Atomic, MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct, BlockConfig{ 16u, 16u },
                TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 1, 1, 64 },
                TraversalOrder::Xyz, Stride(), 0, 0, 0, 255);
    planA.m_OpGraph.AddOp(std::make_unique<MceOp>(std::move(mceOp)));
    planA.m_OpGraph.SetProducer(planA.m_OpGraph.GetBuffers()[0], planA.m_OpGraph.GetOps()[0]);
    planA.m_OutputMappings = { { planA.m_OpGraph.GetBuffers()[0], partAOutputSlot0 } };

    Plan planB;
    planB.m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                       TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                       TraversalOrder::Xyz, 4, QuantizationInfo()));
    PleOp pleOp(Lifetime::Atomic, PleOperation::PASSTHROUGH, BlockConfig{ 16u, 16u }, 1U,
                { TensorShape{ 1, 64, 64, 64 } }, TensorShape{ 1, 64, 64, 64 }, ethosn::command_stream::DataType::U8,
                true);
    planB.m_OpGraph.AddOp(std::make_unique<PleOp>(std::move(pleOp)));
    planB.m_OpGraph.AddConsumer(planB.m_OpGraph.GetBuffers()[0], planB.m_OpGraph.GetOps()[0], 0);
    planB.m_InputMappings = { { planB.m_OpGraph.GetBuffers()[0], partBInputSlot0 } };

    Combiner combiner(graph, hwCaps, estOpt, debuggingContext);

    REQUIRE(combiner.ArePlansAllowedToMerge(planA, planB, PartConnection{ partBInputSlot0, partAOutputSlot0 }) == true);

    planA.m_HasIdentityPle = true;

    REQUIRE(combiner.ArePlansAllowedToMerge(planA, planB, PartConnection{ partBInputSlot0, partAOutputSlot0 }) == true);

    planB.m_HasIdentityMce = true;

    REQUIRE(combiner.ArePlansAllowedToMerge(planA, planB, PartConnection{ partBInputSlot0, partAOutputSlot0 }) ==
            false);
}
