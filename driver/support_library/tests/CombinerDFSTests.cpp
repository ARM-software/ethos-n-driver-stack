// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/DebuggingContext.hpp"
#include "../src/SramAllocator.hpp"
#include "../src/ThreadPool.hpp"
#include "../src/cascading/Cascading.hpp"
#include "../src/cascading/CombinerDFS.hpp"
#include "../src/cascading/ConcatPart.hpp"
#include "../src/cascading/McePart.hpp"
#include "../src/cascading/StripeHelper.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>

#include <atomic>
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
               std::atomic<uint32_t>* numPlansCounter,
               std::vector<uint32_t>* numWeightBuffers,
               const std::function<bool(CascadeType, PartId)>& filter,
               bool hasInput,
               bool hasOutput)
        : MockPart(id, hasInput, hasOutput)
        , m_NumPlansCounter(numPlansCounter)
        , m_NumWeightBuffers(numWeightBuffers)
        , m_Filter(filter)
    {}

    Plans GetPlans(CascadeType cascadeType,
                   ethosn::command_stream::BlockConfig,
                   const std::vector<Buffer*>&,
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
            SramBuffer* buffer         = opGraph.AddBuffer(std::make_unique<SramBuffer>());
            buffer->m_Format           = CascadingBufferFormat::NHWCB;
            buffer->m_Order            = TraversalOrder::Xyz;
            buffer->m_TensorShape      = { 1, 16, 16, 16 };
            buffer->m_StripeShape      = { 1, 16, 16, 16 };
            buffer->m_SizeInBytes      = 16 * 16 * 16;
            buffer->m_QuantizationInfo = { 0, 1.f };

            inputMappings[buffer] = PartInputSlot{ m_PartId, 0 };
        }

        if (m_HasOutput)
        {
            SramBuffer* buffer         = opGraph.AddBuffer(std::make_unique<SramBuffer>());
            buffer->m_Format           = CascadingBufferFormat::NHWCB;
            buffer->m_Order            = TraversalOrder::Xyz;
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

    std::atomic<uint32_t>* m_NumPlansCounter;
    std::vector<uint32_t>* m_NumWeightBuffers;
    // Function instance used to store the filter lambda function.
    std::function<bool(CascadeType, PartId)> m_Filter;
};

class NoWeightPart : public WeightPart
{
public:
    NoWeightPart(PartId id,
                 std::atomic<uint32_t>* numPlansCounter,
                 std::vector<uint32_t>* weightBuffers,
                 const std::function<bool(CascadeType, PartId)>& filter,
                 bool hasInput,
                 bool hasOutput)
        : WeightPart(id, numPlansCounter, weightBuffers, filter, hasInput, hasOutput)
    {}

    bool CanDoubleBufferWeights() const override
    {
        return false;
    }
};

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
    std::atomic<uint32_t> numPlansCounter(0);
    std::array<std::vector<uint32_t>, 3> planWeightBuffers;
    // Filter lambda function used to force the CombinerTest in generating specific Plans for specific Parts.
    auto filter = [](auto cascadeType, auto partId) {
        return ((partId == 1 && cascadeType == CascadeType::Beginning) ||
                (partId == 2 && cascadeType == CascadeType::Middle) ||
                (partId == 3 && cascadeType == CascadeType::End));
    };

    auto pInput = std::make_unique<MockPart>(graph.GeneratePartId(), false, true);
    auto pA = std::make_unique<NoWeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[0], filter,
                                             true, true);
    auto pB = std::make_unique<NoWeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[1], filter,
                                             true, true);
    auto pC = std::make_unique<NoWeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[2], filter,
                                             true, true);
    auto pOutput = std::make_unique<MockPart>(graph.GeneratePartId(), true, false);

    PartId partInputId  = pInput->GetPartId();
    PartId partAId      = pA->GetPartId();
    PartId partBId      = pB->GetPartId();
    PartId partCId      = pC->GetPartId();
    PartId partOutputId = pOutput->GetPartId();
    graph.AddPart(std::move(pInput));
    graph.AddPart(std::move(pA));
    graph.AddPart(std::move(pB));
    graph.AddPart(std::move(pC));
    graph.AddPart(std::move(pOutput));

    PartOutputSlot partInputOutputSlot0 = { partInputId, 0 };

    PartInputSlot partAInputSlot0   = { partAId, 0 };
    PartOutputSlot partAOutputSlot0 = { partAId, 0 };

    PartInputSlot partBInputSlot0   = { partBId, 0 };
    PartOutputSlot partBOutputSlot0 = { partBId, 0 };

    PartInputSlot partCInputSlot0   = { partCId, 0 };
    PartOutputSlot partCOutputSlot0 = { partCId, 0 };

    PartInputSlot partOutputInputSlot0 = { partOutputId, 0 };

    graph.AddConnection(partAInputSlot0, partInputOutputSlot0);
    graph.AddConnection(partBInputSlot0, partAOutputSlot0);
    graph.AddConnection(partCInputSlot0, partBOutputSlot0);
    graph.AddConnection(partOutputInputSlot0, partCOutputSlot0);

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();
    ThreadPool threadPool(0);

    FrozenGraphOfParts frozenGraph = FrozenGraphOfParts(std::move(graph));
    CombinerTest combiner(frozenGraph, hwCaps, compOpt, estOpt, debuggingContext);
    combiner.Run(threadPool);
    Combination comb = combiner.GetBestCombination();

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
    std::atomic<uint32_t> numPlansCounter(0);
    std::array<std::vector<uint32_t>, 3> planWeightBuffers;
    // Filter lambda function used to force the CombinerTest in generating specific Plans for specific Parts.
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
                                             true, true);
    auto pOutput = std::make_unique<MockPart>(graph.GeneratePartId(), true, false);

    PartId partInputId  = pInput->GetPartId();
    PartId partAId      = pA->GetPartId();
    PartId partBId      = pB->GetPartId();
    PartId partCId      = pC->GetPartId();
    PartId partOutputId = pOutput->GetPartId();
    graph.AddPart(std::move(pInput));
    graph.AddPart(std::move(pA));
    graph.AddPart(std::move(pB));
    graph.AddPart(std::move(pC));
    graph.AddPart(std::move(pOutput));

    PartOutputSlot partInputOutputSlot0 = { partInputId, 0 };

    PartInputSlot partAInputSlot0   = { partAId, 0 };
    PartOutputSlot partAOutputSlot0 = { partAId, 0 };

    PartInputSlot partBInputSlot0   = { partBId, 0 };
    PartOutputSlot partBOutputSlot0 = { partBId, 0 };

    PartInputSlot partCInputSlot0   = { partCId, 0 };
    PartOutputSlot partCOutputSlot0 = { partCId, 0 };

    PartInputSlot partOutputInputSlot0 = { partOutputId, 0 };

    graph.AddConnection(partAInputSlot0, partInputOutputSlot0);
    graph.AddConnection(partBInputSlot0, partAOutputSlot0);
    graph.AddConnection(partCInputSlot0, partBOutputSlot0);
    graph.AddConnection(partOutputInputSlot0, partCOutputSlot0);

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();
    ThreadPool threadPool(0);

    FrozenGraphOfParts frozenGraph = FrozenGraphOfParts(std::move(graph));
    CombinerTest combiner(frozenGraph, hwCaps, compOpt, estOpt, debuggingContext);
    combiner.Run(threadPool);
    Combination comb = combiner.GetBestCombination();

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
    std::atomic<uint32_t> numPlansCounter(0);
    std::array<std::vector<uint32_t>, 3> planWeightBuffers;
    // Filter lambda function used to force the CombinerTest in generating specific Plans for specific Parts.
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

    PartId partInputId  = pInput->GetPartId();
    PartId partAId      = pA->GetPartId();
    PartId partBId      = pB->GetPartId();
    PartId partCId      = pC->GetPartId();
    PartId partOutputId = pOutput->GetPartId();
    graph.AddPart(std::move(pInput));
    graph.AddPart(std::move(pA));
    graph.AddPart(std::move(pB));
    graph.AddPart(std::move(pC));
    graph.AddPart(std::move(pOutput));

    PartOutputSlot partInputOutputSlot0 = { partInputId, 0 };

    PartInputSlot partAInputSlot0   = { partAId, 0 };
    PartOutputSlot partAOutputSlot0 = { partAId, 0 };

    PartInputSlot partBInputSlot0   = { partBId, 0 };
    PartOutputSlot partBOutputSlot0 = { partBId, 0 };

    PartInputSlot partCInputSlot0   = { partCId, 0 };
    PartOutputSlot partCOutputSlot0 = { partCId, 0 };

    PartInputSlot partOutputInputSlot0 = { partOutputId, 0 };

    graph.AddConnection(partAInputSlot0, partInputOutputSlot0);
    graph.AddConnection(partBInputSlot0, partAOutputSlot0);
    graph.AddConnection(partCInputSlot0, partBOutputSlot0);
    graph.AddConnection(partOutputInputSlot0, partCOutputSlot0);

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();
    ThreadPool threadPool(0);

    FrozenGraphOfParts frozenGraph = FrozenGraphOfParts(std::move(graph));
    CombinerTest combiner(frozenGraph, hwCaps, compOpt, estOpt, debuggingContext);
    combiner.Run(threadPool);
    Combination comb = combiner.GetBestCombination();

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
    std::atomic<uint32_t> numPlansCounter(0);
    std::array<std::vector<uint32_t>, 3> planWeightBuffers;
    // Filter lambda function used to force the CombinerTest in generating specific Plans for specific Parts.
    auto filter = [](auto cascadeType, auto partId) {
        return ((partId == 1 && cascadeType == CascadeType::Beginning) ||
                (partId == 2 && cascadeType == CascadeType::Middle) ||
                (partId == 3 && cascadeType == CascadeType::End));
    };

    auto pInput = std::make_unique<MockPart>(graph.GeneratePartId(), false, true);
    auto pA = std::make_unique<NoWeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[0], filter,
                                             true, true);
    auto pB = std::make_unique<WeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[1], filter,
                                           true, true);
    auto pC = std::make_unique<WeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[2], filter,
                                           true, true);
    auto pOutput = std::make_unique<MockPart>(graph.GeneratePartId(), true, false);

    PartId partInputId  = pInput->GetPartId();
    PartId partAId      = pA->GetPartId();
    PartId partBId      = pB->GetPartId();
    PartId partCId      = pC->GetPartId();
    PartId partOutputId = pOutput->GetPartId();
    graph.AddPart(std::move(pInput));
    graph.AddPart(std::move(pA));
    graph.AddPart(std::move(pB));
    graph.AddPart(std::move(pC));
    graph.AddPart(std::move(pOutput));

    PartOutputSlot partInputOutputSlot0 = { partInputId, 0 };

    PartInputSlot partAInputSlot0   = { partAId, 0 };
    PartOutputSlot partAOutputSlot0 = { partAId, 0 };

    PartInputSlot partBInputSlot0   = { partBId, 0 };
    PartOutputSlot partBOutputSlot0 = { partBId, 0 };

    PartInputSlot partCInputSlot0   = { partCId, 0 };
    PartOutputSlot partCOutputSlot0 = { partCId, 0 };

    PartInputSlot partOutputInputSlot0 = { partOutputId, 0 };

    graph.AddConnection(partAInputSlot0, partInputOutputSlot0);
    graph.AddConnection(partBInputSlot0, partAOutputSlot0);
    graph.AddConnection(partCInputSlot0, partBOutputSlot0);
    graph.AddConnection(partOutputInputSlot0, partCOutputSlot0);

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();
    ThreadPool threadPool(0);

    FrozenGraphOfParts frozenGraph = FrozenGraphOfParts(std::move(graph));
    CombinerTest combiner(frozenGraph, hwCaps, compOpt, estOpt, debuggingContext);
    combiner.Run(threadPool);
    Combination comb = combiner.GetBestCombination();

    // The network consists of mocked Mce and Ple kernels with and without weights, hence only the first Mce should be double buffered.
    // Number of plans: 1 weightStripe * 1 PlePlan/weightStripe + 2 weightStripes * (1 McePlan/weightStripe + 1 McePlan/weightStripe) == 5.
    // Number of weightStripes per plan per part: e.g. PlePart generates 1x plans, with 1 weightStripe.
    //                                            e.g. McePart generates 2x plans, with 1 and 2 weightStripes respectively.
    REQUIRE(numPlansCounter == 5);
    REQUIRE(planWeightBuffers[0] == std::vector<uint32_t>{ 1 });
    REQUIRE(planWeightBuffers[1] == std::vector<uint32_t>{ 1, 2 });
    REQUIRE(planWeightBuffers[2] == std::vector<uint32_t>{ 2, 1 });
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
    std::atomic<uint32_t> numPlansCounter(0);
    std::array<std::vector<uint32_t>, 4> planWeightBuffers;
    // Filter lambda function used to force the CombinerTest in generating specific Plans for specific Parts.
    auto filter = [](auto cascadeType, auto partId) {
        return ((partId == 1 && cascadeType == CascadeType::Beginning) ||
                (partId == 2 && cascadeType == CascadeType::Middle) ||
                (partId == 3 && cascadeType == CascadeType::Middle) ||
                (partId == 4 && cascadeType == CascadeType::End));
    };

    auto pInput = std::make_unique<MockPart>(graph.GeneratePartId(), false, true);
    auto pA = std::make_unique<NoWeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[0], filter,
                                             true, true);
    auto pB = std::make_unique<WeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[1], filter,
                                           true, true);
    auto pC = std::make_unique<WeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[2], filter,
                                           true, true);
    auto pD = std::make_unique<NoWeightPart>(graph.GeneratePartId(), &numPlansCounter, &planWeightBuffers[3], filter,
                                             true, true);
    auto pOutput = std::make_unique<MockPart>(graph.GeneratePartId(), true, false);

    PartId partInputId  = pInput->GetPartId();
    PartId partAId      = pA->GetPartId();
    PartId partBId      = pB->GetPartId();
    PartId partCId      = pC->GetPartId();
    PartId partDId      = pD->GetPartId();
    PartId partOutputId = pOutput->GetPartId();
    graph.AddPart(std::move(pInput));
    graph.AddPart(std::move(pA));
    graph.AddPart(std::move(pB));
    graph.AddPart(std::move(pC));
    graph.AddPart(std::move(pD));
    graph.AddPart(std::move(pOutput));

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

    graph.AddConnection(partAInputSlot0, partInputOutputSlot0);
    graph.AddConnection(partBInputSlot0, partAOutputSlot0);
    graph.AddConnection(partCInputSlot0, partBOutputSlot0);
    graph.AddConnection(partDInputSlot0, partCOutputSlot0);
    graph.AddConnection(partOutputInputSlot0, partDOutputSlot0);

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();
    ThreadPool threadPool(0);

    FrozenGraphOfParts frozenGraph = FrozenGraphOfParts(std::move(graph));
    CombinerTest combiner(frozenGraph, hwCaps, compOpt, estOpt, debuggingContext);
    combiner.Run(threadPool);
    Combination comb = combiner.GetBestCombination();

    // The network consists of mocked Mce and Ple kernels with and without weights, hence only the first Mce should be double buffered.
    // Number of plans: 1 weightStripe * 1 PlePlan/weightStripe + 2 weightStripes * (1 McePlan/weightStripe + 1 McePlan/weightStripe + 1 PlePlan/weightStripe) == 7.
    // Number of weightStripes per plan per part: e.g. PlePart generates 1x plans, with 1 weightStripe.
    //                                            e.g. McePart generates 2x plans, with 1 and 2 weightStripes respectively.
    REQUIRE(numPlansCounter == 7);
    REQUIRE(planWeightBuffers[0] == std::vector<uint32_t>{ 1 });
    REQUIRE(planWeightBuffers[1] == std::vector<uint32_t>{ 1, 2 });
    REQUIRE(planWeightBuffers[2] == std::vector<uint32_t>{ 2, 1 });
    REQUIRE(planWeightBuffers[3] == std::vector<uint32_t>{ 2, 1 });
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

    auto pA = std::make_unique<MockPart>(graph.GeneratePartId());

    PartId partAId = pA->GetPartId();
    graph.AddPart(std::move(pA));

    PartInputSlot partAInputSlot0   = { partAId, 0 };
    PartOutputSlot partAOutputSlot0 = { partAId, 0 };

    const uint32_t ifmSize = 1 * 16 * 16 * 16;
    const uint32_t ofmSize = 1 * 16 * 16 * 16;

    // Plan A
    Plan planA;
    SramBuffer* inputSram    = planA.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    inputSram->m_Format      = CascadingBufferFormat::NHWCB;
    inputSram->m_TensorShape = TensorShape{ 1, 16, 16, 16 };
    inputSram->m_StripeShape = TensorShape{ 1, 16, 16, 16 };
    inputSram->m_Order       = TraversalOrder::Xyz;
    inputSram->m_SizeInBytes = ifmSize;
    inputSram->m_DebugTag    = "InputSram";

    PleInputSramBuffer* pleInputSram = planA.m_OpGraph.AddBuffer(std::make_unique<PleInputSramBuffer>());
    pleInputSram->m_Format           = CascadingBufferFormat::NHWCB;
    pleInputSram->m_TensorShape      = TensorShape{ 1, 16, 16, 16 };
    pleInputSram->m_StripeShape      = TensorShape{ 1, 16, 16, 16 };
    pleInputSram->m_SizeInBytes      = ifmSize;
    pleInputSram->m_DebugTag         = "PleInputSram";

    SramBuffer* mceWeightsSram    = planA.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    mceWeightsSram->m_Format      = CascadingBufferFormat::NHWCB;
    mceWeightsSram->m_TensorShape = TensorShape{ 1, 1, 1, 16 };
    mceWeightsSram->m_StripeShape = TensorShape{ 1, 1, 1, 16 };
    mceWeightsSram->m_Order       = TraversalOrder::Xyz;
    mceWeightsSram->m_SizeInBytes = (uint32_t)16;
    mceWeightsSram->m_DebugTag    = "MceWeightsSram";

    SramBuffer* outputSram    = planA.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    outputSram->m_Format      = CascadingBufferFormat::NHWCB;
    outputSram->m_TensorShape = TensorShape{ 1, 16, 16, 16 };
    outputSram->m_StripeShape = TensorShape{ 1, 16, 16, 16 };
    outputSram->m_Order       = TraversalOrder::Xyz;
    outputSram->m_SizeInBytes = ofmSize;
    outputSram->m_DebugTag    = "OutputSram";

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
    planA.m_OpGraph.AddConsumer(inputSram, planA.m_OpGraph.GetOps()[mceOpIndex], 0);
    planA.m_OpGraph.SetProducer(pleInputSram, planA.m_OpGraph.GetOps()[mceOpIndex]);
    planA.m_OpGraph.AddConsumer(pleInputSram, planA.m_OpGraph.GetOps()[pleOpIndex], 0);
    planA.m_OpGraph.AddConsumer(mceWeightsSram, planA.m_OpGraph.GetOps()[mceOpIndex], 1);
    planA.m_OpGraph.SetProducer(outputSram, planA.m_OpGraph.GetOps()[pleOpIndex]);
    planA.m_InputMappings  = { { inputSram, partAInputSlot0 } };
    planA.m_OutputMappings = { { outputSram, partAOutputSlot0 } };

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    FrozenGraphOfParts frozenGraph = FrozenGraphOfParts(std::move(graph));
    CombinerTest combiner(frozenGraph, hwCaps, compOpt, estOpt, debuggingContext);

    SectionContext context = { {}, SramAllocator(hwCaps.GetTotalSramSize() / hwCaps.GetNumberOfSrams()),
                               {}, {},
                               0,  false,
                               {}, BlockConfig{ 16u, 16u } };

    // Check that no buffers are allocated before calling AllocateSram().
    REQUIRE(context.alloc.GetAllocationSize() == 0);
    REQUIRE(combiner.AllocateSram(context, partAId, planA, { nullptr }) == true);
    // Check that all 4 buffers (Input, Mce Weights, Ple Code, Output) have been allocated.
    REQUIRE(context.alloc.GetAllocationSize() == 4);
    // Check that 2 buffers (Mce Weights, Input) have been deallocated.
    PartId nextPartId = 17;
    combiner.DeallocateUnusedBuffers(partAId, { { outputSram, partAOutputSlot0 } }, { nextPartId }, context);
    REQUIRE(context.alloc.GetAllocationSize() == 2);
    // Check that it is only the Input and Mce Weights buffers that have been deallocated.
    REQUIRE(context.alloc.TryFree(inputSram->m_Offset.value()) == false);
    REQUIRE(context.alloc.TryFree(mceWeightsSram->m_Offset.value()) == false);
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

    auto pA = std::make_unique<MockPart>(graph.GeneratePartId());

    PartId partAId = pA->GetPartId();
    graph.AddPart(std::move(pA));

    PartInputSlot partAInputSlot0   = { partAId, 0 };
    PartOutputSlot partAOutputSlot0 = { partAId, 0 };

    const uint32_t ifmSize = 1 * 16 * 16 * 16;
    const uint32_t ofmSize = 1 * 16 * 16 * 16;

    // Plan A
    Plan planA;
    SramBuffer* inputSram    = planA.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    inputSram->m_Format      = CascadingBufferFormat::NHWCB;
    inputSram->m_TensorShape = TensorShape{ 1, 16, 16, 16 };
    inputSram->m_StripeShape = TensorShape{ 1, 16, 16, 16 };
    inputSram->m_Order       = TraversalOrder::Xyz;
    inputSram->m_SizeInBytes = ifmSize;
    inputSram->m_DebugTag    = "InputSram";
    size_t inputBufferIndex  = planA.m_OpGraph.GetBuffers().size() - 1;

    PleInputSramBuffer* pleInputSram = planA.m_OpGraph.AddBuffer(std::make_unique<PleInputSramBuffer>());
    pleInputSram->m_Format           = CascadingBufferFormat::NHWCB;
    pleInputSram->m_TensorShape      = TensorShape{ 1, 16, 16, 16 };
    pleInputSram->m_StripeShape      = TensorShape{ 1, 16, 16, 16 };
    pleInputSram->m_SizeInBytes      = ifmSize;
    pleInputSram->m_DebugTag         = "PleInputSram";
    size_t pleInputSramIndex         = planA.m_OpGraph.GetBuffers().size() - 1;

    SramBuffer* mceWeightsSram    = planA.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    mceWeightsSram->m_Format      = CascadingBufferFormat::NHWCB;
    mceWeightsSram->m_TensorShape = TensorShape{ 1, 1, 1, 16 };
    mceWeightsSram->m_StripeShape = TensorShape{ 1, 1, 1, 16 };
    mceWeightsSram->m_Order       = TraversalOrder::Xyz;
    mceWeightsSram->m_SizeInBytes = (uint32_t)16;
    mceWeightsSram->m_DebugTag    = "MceWeightsSram";
    size_t mceWeightsBufferIndex  = planA.m_OpGraph.GetBuffers().size() - 1;

    SramBuffer* outputSram    = planA.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    outputSram->m_Format      = CascadingBufferFormat::NHWCB;
    outputSram->m_TensorShape = TensorShape{ 1, 32, 16, 16 };
    outputSram->m_StripeShape = TensorShape{ 1, 16, 16, 16 };
    outputSram->m_Order       = TraversalOrder::Xyz;
    outputSram->m_SizeInBytes = ofmSize;
    outputSram->m_DebugTag    = "OutputSram";
    size_t outputBufferIndex  = planA.m_OpGraph.GetBuffers().size() - 1;

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

    FrozenGraphOfParts frozenGraph = FrozenGraphOfParts(std::move(graph));
    CombinerTest combiner(frozenGraph, hwCaps, compOpt, estOpt, debuggingContext);

    SectionContext context = { {}, SramAllocator(hwCaps.GetTotalSramSize() / hwCaps.GetNumberOfSrams()),
                               {}, {},
                               0,  false,
                               {}, BlockConfig{ 16u, 16u } };

    // Check that no buffers are allocated before calling AllocateSram().
    REQUIRE(context.alloc.GetAllocationSize() == 0);
    REQUIRE(combiner.AllocateSram(context, partAId, planA, { nullptr }) == true);
    // Check that all 4 buffers (Input, Mce Weights, Ple Code, Output) have been allocated.
    REQUIRE(context.alloc.GetAllocationSize() == 4);
    // Check that none of the buffers have been deallocated.
    PartId nextPartId = 17;
    combiner.DeallocateUnusedBuffers(partAId, { { planA.m_OpGraph.GetBuffers()[outputBufferIndex], partAOutputSlot0 } },
                                     { nextPartId }, context);
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
    auto pA        = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB        = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pC        = std::make_unique<MockPart>(graph.GeneratePartId());
    PartId partAId = pA->GetPartId();
    PartId partBId = pB->GetPartId();
    PartId partCId = pC->GetPartId();
    graph.AddPart(std::move(pA));
    graph.AddPart(std::move(pB));
    graph.AddPart(std::move(pC));

    PartOutputSlot partAOutputSlot0 = { partAId, 0 };

    PartInputSlot partBInputSlot0   = { partBId, 0 };
    PartOutputSlot partBOutputSlot0 = { partBId, 0 };

    PartInputSlot partCInputSlot0 = { partCId, 0 };

    graph.AddConnection(partBInputSlot0, partAOutputSlot0);
    graph.AddConnection(partCInputSlot0, partBOutputSlot0);

    // Plan A
    Plan planA;
    SramBuffer* inputSramA     = planA.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    inputSramA->m_Format       = CascadingBufferFormat::NHWCB;
    inputSramA->m_TensorShape  = TensorShape{ 1, 17, 16, 16 };
    inputSramA->m_StripeShape  = TensorShape{ 1, 17, 16, 16 };
    inputSramA->m_Order        = TraversalOrder::Xyz;
    inputSramA->m_DebugTag     = "InputSramA";
    SramBuffer* outputSramA    = planA.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    outputSramA->m_Format      = CascadingBufferFormat::NHWCB;
    outputSramA->m_TensorShape = TensorShape{ 1, 17, 16, 16 };
    outputSramA->m_StripeShape = TensorShape{ 1, 17, 16, 16 };
    outputSramA->m_Order       = TraversalOrder::Xyz;
    outputSramA->m_DebugTag    = "OutputSramA";
    planA.m_OutputMappings     = { { outputSramA, partAOutputSlot0 } };
    planA.m_OpGraph.AddOp(std::make_unique<MceOp>(
        MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct, BlockConfig{ 16u, 16u }, TensorShape{ 1, 17, 16, 16 },
        TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 1, 1, 16 }, TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));
    planA.m_OpGraph.GetOps()[0]->m_DebugTag = "MceA";
    planA.m_OpGraph.AddConsumer(inputSramA, planA.m_OpGraph.GetOps()[0], 0);
    planA.m_OpGraph.SetProducer(outputSramA, planA.m_OpGraph.GetOps()[0]);

    // Plan B
    Plan planB;
    SramBuffer* inputSramB     = planB.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    inputSramB->m_Format       = CascadingBufferFormat::NHWCB;
    inputSramB->m_TensorShape  = TensorShape{ 1, 17, 16, 16 };
    inputSramB->m_StripeShape  = TensorShape{ 1, 17, 16, 16 };
    inputSramB->m_Order        = TraversalOrder::Xyz;
    inputSramB->m_SizeInBytes  = 4;
    inputSramB->m_DebugTag     = "InputSramB";
    SramBuffer* outputSramB    = planB.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    outputSramB->m_Format      = CascadingBufferFormat::NHWCB;
    outputSramB->m_TensorShape = TensorShape{ 1, 17, 16, 16 };
    outputSramB->m_StripeShape = TensorShape{ 1, 17, 16, 16 };
    outputSramB->m_Order       = TraversalOrder::Xyz;
    outputSramB->m_SizeInBytes = 4;
    outputSramB->m_DebugTag    = "OutputSramB";
    planB.m_InputMappings      = { { inputSramB, partBInputSlot0 } };
    planB.m_OutputMappings     = { { outputSramB, partBOutputSlot0 } };
    planB.m_OpGraph.AddOp(std::make_unique<MceOp>(
        MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct, BlockConfig{ 16u, 16u }, TensorShape{ 1, 17, 16, 16 },
        TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 1, 1, 16 }, TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));
    planB.m_OpGraph.GetOps()[0]->m_DebugTag = "MceB";
    planB.m_OpGraph.AddConsumer(inputSramB, planB.m_OpGraph.GetOps()[0], 0);
    planB.m_OpGraph.SetProducer(outputSramB, planB.m_OpGraph.GetOps()[0]);

    // Plan C
    Plan planC;
    SramBuffer* inputSramC    = planC.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    inputSramC->m_Format      = CascadingBufferFormat::NHWCB;
    inputSramC->m_TensorShape = TensorShape{ 1, 17, 16, 16 };
    inputSramC->m_StripeShape = TensorShape{ 1, 17, 16, 16 };
    inputSramC->m_Order       = TraversalOrder::Xyz;
    inputSramC->m_SizeInBytes = 4;
    inputSramC->m_DebugTag    = "InputSramC";
    planC.m_InputMappings     = { { inputSramC, partCInputSlot0 } };

    auto endingGlueA = std::make_shared<EndingGlue>();
    endingGlueA->m_Graph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
    endingGlueA->m_Graph.GetOps()[0]->m_DebugTag = "InputDma";

    std::unique_ptr<DramBuffer> dramBufferPtr = DramBuffer::Build()
                                                    .AddFormat(CascadingBufferFormat::NHWCB)
                                                    .AddTensorShape(TensorShape{ 1, 17, 16, 16 })
                                                    .AddBufferType(BufferType::Intermediate)
                                                    .AddDebugTag("DramBuffer");

    DramBuffer* dramBuffer = endingGlueA->m_Graph.AddBuffer(std::move(dramBufferPtr));

    endingGlueA->m_Graph.SetProducer(dramBuffer, endingGlueA->m_Graph.GetOps()[0]);
    endingGlueA->m_ExternalConnections.m_BuffersToOps.insert({ outputSramA, endingGlueA->m_Graph.GetOp(0) });

    auto startingGlueB = std::make_shared<StartingGlue>();
    startingGlueB->m_Graph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
    startingGlueB->m_Graph.GetOps()[0]->m_DebugTag = "OutputDma";
    startingGlueB->m_ExternalConnections.m_BuffersToOps.insert({ dramBuffer, startingGlueB->m_Graph.GetOp(0) });
    startingGlueB->m_ExternalConnections.m_OpsToBuffers.insert({ startingGlueB->m_Graph.GetOps()[0], inputSramB });

    auto endingGlueB = std::make_shared<EndingGlue>();

    auto startingGlueC = std::make_shared<StartingGlue>();
    startingGlueC->m_Graph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
    startingGlueC->m_Graph.GetOps()[0]->m_DebugTag = "InputDmaC";
    startingGlueC->m_ExternalConnections.m_BuffersToOps.insert({ outputSramB, startingGlueC->m_Graph.GetOp(0) });
    startingGlueC->m_ExternalConnections.m_OpsToBuffers.insert({ startingGlueC->m_Graph.GetOp(0), inputSramC });

    // Create Combination with all the plans and glues
    Combination combA(partAId, std::move(planA));
    combA.SetEndingGlue(std::move(*endingGlueA), partAOutputSlot0);

    Combination combB(partBId, std::move(planB));
    combB.SetStartingGlue(std::move(*startingGlueB), partBInputSlot0);
    combB.SetEndingGlue(std::move(*endingGlueB), partBOutputSlot0);

    Combination combC(partCId, std::move(planC));
    combC.SetStartingGlue(std::move(*startingGlueC), partCInputSlot0);

    Combination comb = combA + combB + combC;

    bool dumpInputGraphToFile = false;
    if (dumpInputGraphToFile)
    {
        std::ofstream stream("GetOpGraphForDfsCombinationPartialSram Input.dot");
        SaveCombinationToDot(comb, stream, DetailLevel::High);
    }

    // Call function under test
    FrozenGraphOfParts frozenGraph = FrozenGraphOfParts(std::move(graph));
    OpGraph combOpGraph            = GetOpGraphForCombination(comb, frozenGraph);

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

    REQUIRE(combOpGraph.GetSingleProducer(combOpGraph.GetBuffers()[0]) == nullptr);
    REQUIRE(combOpGraph.GetSingleProducer(combOpGraph.GetBuffers()[1])->m_DebugTag == "MceA");
    REQUIRE(combOpGraph.GetSingleProducer(combOpGraph.GetBuffers()[2])->m_DebugTag == "InputDma");
    REQUIRE(combOpGraph.GetSingleProducer(combOpGraph.GetBuffers()[3])->m_DebugTag == "OutputDma");
    REQUIRE(combOpGraph.GetSingleProducer(combOpGraph.GetBuffers()[4])->m_DebugTag == "MceB");
    REQUIRE(combOpGraph.GetSingleProducer(combOpGraph.GetBuffers()[5])->m_DebugTag == "InputDmaC");

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
    auto pA = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pC = std::make_unique<MockPart>(graph.GeneratePartId());

    BasePart& partA = *pA;
    BasePart& partB = *pB;
    BasePart& partC = *pC;

    graph.AddPart(std::move(pA));
    graph.AddPart(std::move(pB));
    graph.AddPart(std::move(pC));

    PartOutputSlot partAOutputSlot = { partA.GetPartId(), 0 };
    PartOutputSlot partBOutputSlot = { partB.GetPartId(), 0 };

    PartInputSlot partCInputSlot0 = { partC.GetPartId(), 0 };
    PartInputSlot partCInputSlot1 = { partC.GetPartId(), 1 };

    graph.AddConnection(partCInputSlot0, { partAOutputSlot });
    graph.AddConnection(partCInputSlot1, { partBOutputSlot });

    const CompilationOptions compOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);

    // Plan A
    Plan planA;
    SramBuffer* outputSramA    = planA.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    outputSramA->m_Format      = CascadingBufferFormat::NHWCB;
    outputSramA->m_TensorShape = TensorShape{ 1, 64, 64, 64 };
    outputSramA->m_StripeShape = TensorShape{ 1, 8, 8, 32 };
    outputSramA->m_Order       = TraversalOrder::Xyz;
    outputSramA->m_SizeInBytes = 4;
    outputSramA->m_DebugTag    = "OutputSramA";
    planA.m_OutputMappings     = { { outputSramA, partAOutputSlot } };
    auto dummyOpA              = std::make_unique<DummyOp>();
    dummyOpA->m_DebugTag       = "DummyA";
    planA.m_OpGraph.AddOp(std::move(dummyOpA));

    // Plan B
    Plan planB;
    SramBuffer* outputSramB    = planB.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    outputSramB->m_Format      = CascadingBufferFormat::NHWCB;
    outputSramB->m_TensorShape = TensorShape{ 1, 64, 64, 64 };
    outputSramB->m_StripeShape = TensorShape{ 1, 8, 8, 32 };
    outputSramB->m_Order       = TraversalOrder::Xyz;
    outputSramB->m_SizeInBytes = 4;
    outputSramB->m_DebugTag    = "OutputSramB";
    planB.m_OutputMappings     = { { outputSramB, partBOutputSlot } };
    auto dummyOpB              = std::make_unique<DummyOp>();
    dummyOpB->m_DebugTag       = "DummyB";
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
    std::unique_ptr<DramBuffer> input0DramCPtr = DramBuffer::Build()
                                                     .AddFormat(CascadingBufferFormat::NHWCB)
                                                     .AddTensorShape(TensorShape{ 1, 64, 64, 64 })
                                                     .AddSizeInBytes(4)
                                                     .AddBufferType(BufferType::Intermediate)
                                                     .AddDebugTag("Input0DramC");
    DramBuffer* input0DramC = planC.m_OpGraph.AddBuffer(std::move(input0DramCPtr));

    std::unique_ptr<DramBuffer> input1DramCPtr = DramBuffer::Build()
                                                     .AddFormat(CascadingBufferFormat::NHWCB)
                                                     .AddTensorShape(TensorShape{ 1, 64, 64, 64 })
                                                     .AddSizeInBytes(4)
                                                     .AddBufferType(BufferType::Intermediate)
                                                     .AddDebugTag("Input1DramC");
    DramBuffer* input1DramC = planC.m_OpGraph.AddBuffer(std::move(input1DramCPtr));

    planC.m_InputMappings = { { input0DramC, partCInputSlot0 }, { input1DramC, partCInputSlot1 } };
    auto dummyOpC         = std::make_unique<DummyOp>();
    dummyOpC->m_DebugTag  = "DummyC";
    planC.m_OpGraph.AddOp(std::move(dummyOpC));

    auto startingGlueC_A = std::make_shared<StartingGlue>();
    startingGlueC_A->m_ExternalConnections.m_OpsToBuffers.insert({ endingGlueA->m_Graph.GetOps().back(), input0DramC });

    auto startingGlueC_B = std::make_shared<StartingGlue>();
    startingGlueC_A->m_ExternalConnections.m_OpsToBuffers.insert({ endingGlueB->m_Graph.GetOps().back(), input1DramC });

    // Create Combination with all the plans and glues
    Combination combA(partA.GetPartId(), std::move(planA));
    combA.SetEndingGlue(std::move(*endingGlueA), partAOutputSlot);

    Combination combB(partB.GetPartId(), std::move(planB));
    combB.SetEndingGlue(std::move(*endingGlueB), partBOutputSlot);

    Combination combC(partC.GetPartId(), std::move(planC));
    combC.SetStartingGlue(std::move(*startingGlueC_A), partCInputSlot0);
    combC.SetStartingGlue(std::move(*startingGlueC_B), partCInputSlot1);

    Combination comb = combA + combB + combC;

    bool dumpInputGraphToFile = false;
    if (dumpInputGraphToFile)
    {
        std::ofstream stream("GetOpGraphForDfsMISOSramsToDrams.dot");
        SaveCombinationToDot(comb, stream, DetailLevel::High);
    }

    // Call function under test
    FrozenGraphOfParts frozenGraph = FrozenGraphOfParts(std::move(graph));
    OpGraph combOpGraph            = GetOpGraphForCombination(comb, frozenGraph);

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("GetOpGraphForDfsMISOSramsToDrams Output.dot");
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
    REQUIRE(combOpGraph.GetSingleProducer(combOpGraph.GetBuffers()[2])->m_DebugTag == "GlueAC_Dma");
    REQUIRE(combOpGraph.GetSingleProducer(combOpGraph.GetBuffers()[3])->m_DebugTag == "GlueBC_Dma");
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
    auto pA = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pC = std::make_unique<MockPart>(graph.GeneratePartId());

    BasePart& partA = *pA;
    BasePart& partB = *pB;
    BasePart& partC = *pC;

    graph.AddPart(std::move(pA));
    graph.AddPart(std::move(pB));
    graph.AddPart(std::move(pC));

    PartOutputSlot partAOutputSlot = { partA.GetPartId(), 0 };
    PartOutputSlot partBOutputSlot = { partB.GetPartId(), 0 };

    PartInputSlot partCInputSlot0 = { partC.GetPartId(), 0 };
    PartInputSlot partCInputSlot1 = { partC.GetPartId(), 1 };

    graph.AddConnection(partCInputSlot0, { partAOutputSlot });
    graph.AddConnection(partCInputSlot1, { partBOutputSlot });

    const CompilationOptions compOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);

    // Plan A
    Plan planA;
    std::unique_ptr<DramBuffer> outputSramAPtr = DramBuffer::Build()
                                                     .AddFormat(CascadingBufferFormat::NHWCB)
                                                     .AddTensorShape(TensorShape{ 1, 64, 64, 64 })
                                                     .AddSizeInBytes(4)
                                                     .AddBufferType(BufferType::Intermediate)
                                                     .AddDebugTag("OutputSramA");
    DramBuffer* outputSramA = planA.m_OpGraph.AddBuffer(std::move(outputSramAPtr));

    planA.m_OutputMappings = { { outputSramA, partAOutputSlot } };
    auto dummyOpA          = std::make_unique<DummyOp>();
    dummyOpA->m_DebugTag   = "DummyA";
    planA.m_OpGraph.AddOp(std::move(dummyOpA));
    planA.m_OpGraph.SetProducer(planA.m_OpGraph.GetBuffers().back(), planA.m_OpGraph.GetOps().back());

    // Plan B
    Plan planB;
    std::unique_ptr<DramBuffer> outputSramBPtr = DramBuffer::Build()
                                                     .AddFormat(CascadingBufferFormat::NHWCB)
                                                     .AddTensorShape(TensorShape{ 1, 64, 64, 64 })
                                                     .AddSizeInBytes(4)
                                                     .AddBufferType(BufferType::Intermediate)
                                                     .AddDebugTag("OutputSramB");
    DramBuffer* outputSramB = planB.m_OpGraph.AddBuffer(std::move(outputSramBPtr));

    planB.m_OutputMappings = { { outputSramB, partBOutputSlot } };
    auto dummyOpB          = std::make_unique<DummyOp>();
    dummyOpB->m_DebugTag   = "DummyB";
    planB.m_OpGraph.AddOp(std::move(dummyOpB));
    planB.m_OpGraph.SetProducer(outputSramB, planB.m_OpGraph.GetOps().back());

    // Plan C
    Plan planC;
    SramBuffer* input0DramC    = planC.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    input0DramC->m_Format      = CascadingBufferFormat::NHWCB;
    input0DramC->m_TensorShape = TensorShape{ 1, 64, 64, 64 };
    input0DramC->m_StripeShape = TensorShape{ 1, 8, 8, 32 };
    input0DramC->m_Order       = TraversalOrder::Xyz;
    input0DramC->m_SizeInBytes = 4;
    input0DramC->m_DebugTag    = "Input0DramC";
    SramBuffer* input1DramC    = planC.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    input1DramC->m_Format      = CascadingBufferFormat::NHWCB;
    input1DramC->m_TensorShape = TensorShape{ 1, 64, 64, 64 };
    input1DramC->m_StripeShape = TensorShape{ 1, 8, 8, 32 };
    input1DramC->m_Order       = TraversalOrder::Xyz;
    input1DramC->m_SizeInBytes = 4;
    input1DramC->m_DebugTag    = "Input1DramC";
    planC.m_InputMappings      = { { input0DramC, partCInputSlot0 }, { input1DramC, partCInputSlot1 } };
    auto dummyOpC              = std::make_unique<DummyOp>();
    dummyOpC->m_DebugTag       = "DummyC";
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
    Combination combA(partA.GetPartId(), std::move(planA));
    combA.SetEndingGlue(std::move(*endingGlueA), partAOutputSlot);

    Combination combB(partB.GetPartId(), std::move(planB));
    combB.SetEndingGlue(std::move(*endingGlueB), partBOutputSlot);

    Combination combC(partC.GetPartId(), std::move(planC));
    combC.SetStartingGlue(std::move(*startingGlueCA), partCInputSlot0);
    combC.SetStartingGlue(std::move(*startingGlueCB), partCInputSlot1);

    Combination comb = combA + combB + combC;

    bool dumpInputGraphToFile = false;
    if (dumpInputGraphToFile)
    {
        std::ofstream stream("GetOpGraphForDfsMISOSramsToDrams.dot");
        SaveCombinationToDot(comb, stream, DetailLevel::High);
    }

    // Call function under test
    FrozenGraphOfParts frozenGraph = FrozenGraphOfParts(std::move(graph));
    OpGraph combOpGraph            = GetOpGraphForCombination(comb, frozenGraph);

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("GetOpGraphForDfsMISODramsToSrams Output.dot");
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
    REQUIRE(combOpGraph.GetSingleProducer(combOpGraph.GetBuffers()[2])->m_DebugTag == "GlueAC_Dma");
    REQUIRE(combOpGraph.GetSingleProducer(combOpGraph.GetBuffers()[3])->m_DebugTag == "GlueBC_Dma");
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

    graph.AddPart(std::move(pA));
    graph.AddPart(std::move(pB));
    graph.AddPart(std::move(pC));
    graph.AddPart(std::move(pD));
    graph.AddPart(std::move(pE));

    PartOutputSlot partAOutputSlot = { partA.GetPartId(), 0 };
    PartOutputSlot partDOutputSlot = { partD.GetPartId(), 0 };

    PartInputSlot partBInputSlot = { partB.GetPartId(), 0 };
    PartInputSlot partCInputSlot = { partC.GetPartId(), 0 };
    PartInputSlot partDInputSlot = { partD.GetPartId(), 0 };
    PartInputSlot partEInputSlot = { partE.GetPartId(), 0 };

    graph.AddConnection(partBInputSlot, { partAOutputSlot });
    graph.AddConnection(partCInputSlot, { partAOutputSlot });
    graph.AddConnection(partDInputSlot, { partAOutputSlot });
    graph.AddConnection(partEInputSlot, { partDOutputSlot });

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    // Plan A
    Plan planA;
    SramBuffer* outputSramA    = planA.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    outputSramA->m_Format      = CascadingBufferFormat::NHWCB;
    outputSramA->m_TensorShape = TensorShape{ 1, 64, 64, 64 };
    outputSramA->m_StripeShape = TensorShape{ 1, 8, 8, 32 };
    outputSramA->m_Order       = TraversalOrder::Xyz;
    outputSramA->m_SizeInBytes = 4;
    outputSramA->m_DebugTag    = "OutputSramA";
    planA.m_OutputMappings     = { { planA.m_OpGraph.GetBuffers()[0], partAOutputSlot } };

    // Plan B
    Plan planB;
    SramBuffer* inputSramB    = planB.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    inputSramB->m_Format      = CascadingBufferFormat::NHWCB;
    inputSramB->m_TensorShape = TensorShape{ 1, 64, 64, 64 };
    inputSramB->m_StripeShape = TensorShape{ 1, 8, 8, 32 };
    inputSramB->m_Order       = TraversalOrder::Xyz;
    inputSramB->m_SizeInBytes = 4;
    inputSramB->m_DebugTag    = "InputSramB";
    planB.m_InputMappings     = { { planB.m_OpGraph.GetBuffers()[0], partBInputSlot } };

    // Plan C
    Plan planC;
    SramBuffer* inputSramC    = planC.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    inputSramC->m_Format      = CascadingBufferFormat::NHWCB;
    inputSramC->m_TensorShape = TensorShape{ 1, 64, 64, 64 };
    inputSramC->m_StripeShape = TensorShape{ 1, 8, 8, 32 };
    inputSramC->m_Order       = TraversalOrder::Xyz;
    inputSramC->m_SizeInBytes = 4;
    inputSramC->m_DebugTag    = "InputSramC";
    planC.m_InputMappings     = { { planC.m_OpGraph.GetBuffers()[0], partCInputSlot } };

    // Plan D
    Plan planD;
    std::unique_ptr<DramBuffer> inputDramDPtr = DramBuffer::Build()
                                                    .AddFormat(CascadingBufferFormat::NHWCB)
                                                    .AddTensorShape(TensorShape{ 1, 64, 64, 64 })
                                                    .AddSizeInBytes(4)
                                                    .AddBufferType(BufferType::Intermediate)
                                                    .AddDebugTag("InputDramD");
    planD.m_OpGraph.AddBuffer(std::move(inputDramDPtr));

    SramBuffer* outputSramD    = planD.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    outputSramD->m_Format      = CascadingBufferFormat::NHWCB;
    outputSramD->m_TensorShape = TensorShape{ 1, 64, 64, 64 };
    outputSramD->m_StripeShape = TensorShape{ 1, 8, 8, 32 };
    outputSramD->m_SizeInBytes = 4;
    outputSramD->m_DebugTag    = "OutputSramD";
    planD.m_InputMappings      = { { planD.m_OpGraph.GetBuffers()[0], partDInputSlot } };
    planD.m_OpGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
    planD.m_OpGraph.GetOps()[0]->m_DebugTag = "DmaToSramD";
    planD.m_OpGraph.AddConsumer(planD.m_OpGraph.GetBuffers()[0], planD.m_OpGraph.GetOps()[0], 0);
    planD.m_OpGraph.SetProducer(planD.m_OpGraph.GetBuffers()[1], planD.m_OpGraph.GetOps()[0]);

    // Create Combination with all the plans and glues
    Combination combA(partA.GetPartId(), std::move(planA));
    Combination combB(partB.GetPartId(), std::move(planB));
    Combination combC(partC.GetPartId(), std::move(planC));
    Combination combD(partD.GetPartId(), std::move(planD));

    // Merge the combinations
    Combination comb = combA + combB + combC + combD;

    FrozenGraphOfParts frozenGraph = FrozenGraphOfParts(std::move(graph));
    CombinerTest combiner(frozenGraph, hwCaps, compOpt, estOpt, debuggingContext);

    Combination combGlued = combiner.GluePartToCombinationSrcToDests(partA, comb, 0);

    bool dumpInputGraphToFile = false;
    if (dumpInputGraphToFile)
    {
        std::ofstream stream("Add shared glue between dram and sram Input.dot");
        SaveCombinationToDot(combGlued, stream, DetailLevel::High);
    }

    // The output should be the ending glue of part A contains a Dram buffer, This Dram buffer is used for part D's input buffer
    // And Part B and C's sram buffers are dma'd from that merged buffer.
    auto elemA              = combGlued.GetElem(partAId);
    EndingGlue* endingGlueA = elemA.m_EndingGlues.find(partAOutputSlot)->second.get();

    auto elemB                  = combGlued.GetElem(partBId);
    StartingGlue* startingGlueB = elemB.m_StartingGlues.find(partBInputSlot)->second.get();
    REQUIRE(
        startingGlueB->m_ExternalConnections.m_BuffersToOps.find(endingGlueA->m_Graph.GetBuffers().back())->second ==
        startingGlueB->m_Graph.GetOp(0));

    auto elemC                  = combGlued.GetElem(partCId);
    StartingGlue* startingGlueC = elemC.m_StartingGlues.find(partCInputSlot)->second.get();
    REQUIRE(
        startingGlueC->m_ExternalConnections.m_BuffersToOps.find(endingGlueA->m_Graph.GetBuffers().back())->second ==
        startingGlueC->m_Graph.GetOp(0));

    auto elemD                  = combGlued.GetElem(partDId);
    StartingGlue* startingGlueD = elemD.m_StartingGlues.find(partDInputSlot)->second.get();
    REQUIRE(
        startingGlueD->m_ExternalConnections.m_ReplacementBuffers.find(elemD.m_Plan->m_OpGraph.GetBuffers().front())
            ->second == endingGlueA->m_Graph.GetBuffers().back());
}

// Manually creates a partial combination starting in Dram with NHWC and going into sram.
// Glue will be generated which adds a conversion to NHWCB through sram
TEST_CASE("GetOpGraphCombinationDramSramConversion", "[CombinerDFS]")
{
    DebuggableObject::ms_IdCounter = 0;    // Reset counter so we get deterministic results

    GraphOfParts graph;
    auto pA = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB = std::make_unique<MockPart>(graph.GeneratePartId());

    BasePart& partA = *pA;
    BasePart& partB = *pB;

    graph.AddPart(std::move(pA));
    graph.AddPart(std::move(pB));

    PartOutputSlot partAOutputSlot = { partA.GetPartId(), 0 };
    PartInputSlot partBInputSlot   = { partB.GetPartId(), 0 };

    graph.AddConnection(partBInputSlot, { partAOutputSlot });

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Plan planA;
    std::unique_ptr<DramBuffer> startingDramBufferPtr = DramBuffer::Build()
                                                            .AddFormat(CascadingBufferFormat::NHWC)
                                                            .AddTensorShape(TensorShape{ 1, 64, 64, 64 })
                                                            .AddSizeInBytes(4)
                                                            .AddBufferType(BufferType::Intermediate)
                                                            .AddDebugTag("StartingDramBuffer");
    planA.m_OpGraph.AddBuffer(std::move(startingDramBufferPtr));

    planA.m_OutputMappings = { { planA.m_OpGraph.GetBuffers()[0], partAOutputSlot } };
    Buffer* startingBuffer = planA.m_OpGraph.GetBuffers().back();

    // Plan B
    Plan planB;
    SramBuffer* finalSramBuffer    = planB.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    finalSramBuffer->m_Format      = CascadingBufferFormat::NHWCB;
    finalSramBuffer->m_TensorShape = TensorShape{ 1, 64, 64, 64 };
    finalSramBuffer->m_StripeShape = TensorShape{ 1, 8, 8, 16 };
    finalSramBuffer->m_Order       = TraversalOrder::Xyz;
    finalSramBuffer->m_SizeInBytes = 4;
    finalSramBuffer->m_DebugTag    = "FinalSramBuffer";
    planB.m_InputMappings          = { { planB.m_OpGraph.GetBuffers()[0], partBInputSlot } };
    Buffer* finalBuffer            = planB.m_OpGraph.GetBuffers().back();

    // Create Combination with all the plans and glues
    Combination combA(partA.GetPartId(), std::move(planA));
    Combination combB(partB.GetPartId(), std::move(planB));

    // Merge the combinations
    Combination comb = combA + combB;

    FrozenGraphOfParts frozenGraph = FrozenGraphOfParts(std::move(graph));
    CombinerTest combiner(frozenGraph, hwCaps, compOpt, estOpt, debuggingContext);

    std::vector<PartConnection> destPartEdge;

    Combination combGlued = combiner.GluePartToCombinationSrcToDests(partA, comb, 0);

    // Access Glue Buffer and Ops and set an appropriate name for debugging purposes
    //auto elemAB = elemIt.m_Glues.find(edgeA2B.m_Destination);
    auto endingGlueA   = combGlued.GetElem(partA.GetPartId()).m_EndingGlues[partAOutputSlot];
    auto startingGlueB = combGlued.GetElem(partB.GetPartId()).m_StartingGlues[partBInputSlot];
    REQUIRE(endingGlueA->m_Graph.GetBuffers().size() == 2);
    REQUIRE(startingGlueB->m_Graph.GetBuffers().size() == 0);
    REQUIRE(endingGlueA->m_Graph.GetBuffers()[1]->m_Location == Location::Sram);
    REQUIRE(endingGlueA->m_Graph.GetBuffers()[1]->m_Format == CascadingBufferFormat::NHWCB);
    REQUIRE(endingGlueA->m_Graph.GetBuffers()[0]->m_Location == Location::Dram);
    REQUIRE(endingGlueA->m_Graph.GetBuffers()[0]->m_Format == CascadingBufferFormat::NHWCB);

    bool dumpInputGraphToFile = false;
    if (dumpInputGraphToFile)
    {
        std::ofstream stream("GetOpGraphCombinationDramSramConversion Input.dot");
        SaveCombinationToDot(combGlued, stream, DetailLevel::High);
    }

    // Call function under test
    OpGraph combOpGraph = GetOpGraphForCombination(combGlued, frozenGraph);

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("GetOpGraphCombinationDramSramConversion Output.dot");
        SaveOpGraphToDot(combOpGraph, stream, DetailLevel::High);
    }

    REQUIRE(combOpGraph.GetBuffers().size() == 4);
    REQUIRE(combOpGraph.GetBuffers()[0] == startingBuffer);
    REQUIRE(combOpGraph.GetBuffers()[1] == endingGlueA->m_Graph.GetBuffers()[0]);
    REQUIRE(combOpGraph.GetBuffers()[2] == endingGlueA->m_Graph.GetBuffers()[1]);
    REQUIRE(combOpGraph.GetBuffers()[3] == finalBuffer);

    REQUIRE(combOpGraph.GetOps().size() == 3);
    REQUIRE(combOpGraph.GetOps() == std::vector<Op*>{ endingGlueA->m_Graph.GetOps()[0],
                                                      endingGlueA->m_Graph.GetOps()[1],
                                                      startingGlueB->m_Graph.GetOps()[0] });

    REQUIRE(combOpGraph.GetSingleProducer(combOpGraph.GetBuffers()[0]) == nullptr);
    REQUIRE(combOpGraph.GetSingleProducer(combOpGraph.GetBuffers()[2]) == combOpGraph.GetOps()[0]);
    REQUIRE(combOpGraph.GetSingleProducer(combOpGraph.GetBuffers()[1]) == combOpGraph.GetOps()[1]);
    REQUIRE(combOpGraph.GetSingleProducer(combOpGraph.GetBuffers()[3]) == combOpGraph.GetOps()[2]);

    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[0]) ==
            OpGraph::ConsumersList{ { combOpGraph.GetOps()[0], 0 } });
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[2]) ==
            OpGraph::ConsumersList{ { combOpGraph.GetOps()[1], 0 } });
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[1]) ==
            OpGraph::ConsumersList{ { combOpGraph.GetOps()[2], 0 } });
    REQUIRE(combOpGraph.GetConsumers(combOpGraph.GetBuffers()[3]) == OpGraph::ConsumersList{});
}

// Manually creates a partial combination with two dram buffers being merged
TEST_CASE("GetOpGraphCombinationDramDramMerge", "[CombinerDFS]")
{
    DebuggableObject::ms_IdCounter = 0;    // Reset counter so we get deterministic results

    GraphOfParts graph;
    auto pA = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB = std::make_unique<MockPart>(graph.GeneratePartId());

    BasePart& partA = *pA;
    BasePart& partB = *pB;

    graph.AddPart(std::move(pA));
    graph.AddPart(std::move(pB));

    PartOutputSlot partAOutputSlot = { partA.GetPartId(), 0 };
    PartInputSlot partBInputSlot   = { partB.GetPartId(), 0 };

    graph.AddConnection(partBInputSlot, { partAOutputSlot });

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Plan planA;
    std::unique_ptr<DramBuffer> startingDramBufferPtr = DramBuffer::Build()
                                                            .AddFormat(CascadingBufferFormat::NHWC)
                                                            .AddTensorShape(TensorShape{ 1, 64, 64, 64 })
                                                            .AddSizeInBytes(4)
                                                            .AddBufferType(BufferType::Intermediate)
                                                            .AddDebugTag("StartingDramBuffer");
    DramBuffer* startingDramBuffer = planA.m_OpGraph.AddBuffer(std::move(startingDramBufferPtr));

    planA.m_OutputMappings = { { startingDramBuffer, partAOutputSlot } };

    // Plan B
    Plan planB;
    std::unique_ptr<DramBuffer> finalDramBufferPtr = DramBuffer::Build()
                                                         .AddFormat(CascadingBufferFormat::NHWC)
                                                         .AddTensorShape(TensorShape{ 1, 64, 64, 64 })
                                                         .AddSizeInBytes(4)
                                                         .AddBufferType(BufferType::Output)
                                                         .AddDebugTag("FinalDramBuffer");
    DramBuffer* finalDramBuffer = planB.m_OpGraph.AddBuffer(std::move(finalDramBufferPtr));

    planB.m_InputMappings = { { finalDramBuffer, partBInputSlot } };

    // Create Combination with all the plans and glues
    Combination combA(partA.GetPartId(), std::move(planA));
    Combination combB(partB.GetPartId(), std::move(planB));

    // Merge the combinations
    Combination comb = combA + combB;

    FrozenGraphOfParts frozenGraph = FrozenGraphOfParts(std::move(graph));
    CombinerTest combiner(frozenGraph, hwCaps, compOpt, estOpt, debuggingContext);

    // Part B and the edge that connects to its source Part A
    Combination combGlued = combiner.GluePartToCombinationSrcToDests(partA, comb, 0);

    bool dumpInputGraphToFile = false;
    if (dumpInputGraphToFile)
    {
        std::ofstream stream("GetOpGraphCombinationDramDramMerge Input.dot");
        SaveCombinationToDot(combGlued, stream, DetailLevel::High);
    }

    // Call function under test
    OpGraph combOpGraph = GetOpGraphForCombination(combGlued, frozenGraph);

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("GetOpGraphCombinationDramDramMerge Output.dot");
        SaveOpGraphToDot(combOpGraph, stream, DetailLevel::High);
    }

    // The buffers have been merged into one, and the buffer is an Output buffer
    REQUIRE(combOpGraph.GetBuffers().size() == 1);
    REQUIRE(combOpGraph.GetBuffers()[0]->Dram()->m_BufferType == BufferType::Output);
    REQUIRE(combOpGraph.GetOps().size() == 0);
}

TEST_CASE("GetOpGraphForDfsCombinationMergedBuffer", "[CombinerDFS]")
{
    GraphOfParts graph;
    auto pA        = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB        = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pC        = std::make_unique<MockPart>(graph.GeneratePartId());
    PartId partAId = pA->GetPartId();
    PartId partBId = pB->GetPartId();
    PartId partCId = pC->GetPartId();
    graph.AddPart(std::move(pA));
    graph.AddPart(std::move(pB));
    graph.AddPart(std::move(pC));

    PartOutputSlot partAOutputSlot0 = { partAId, 0 };
    PartInputSlot partBInputSlot0   = { partBId, 0 };
    PartOutputSlot partBOutputSlot0 = { partBId, 0 };

    PartInputSlot partCInputSlot0 = { partCId, 0 };

    graph.AddConnection(partBInputSlot0, partAOutputSlot0);
    graph.AddConnection(partCInputSlot0, partBOutputSlot0);

    // Plan A
    Plan planA;
    SramBuffer* planA_Buffer0    = planA.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    planA_Buffer0->m_Format      = CascadingBufferFormat::NHWCB;
    planA_Buffer0->m_TensorShape = TensorShape{ 1, 17, 16, 16 };
    planA_Buffer0->m_StripeShape = TensorShape{ 1, 17, 16, 16 };
    planA_Buffer0->m_Order       = TraversalOrder::Xyz;
    planA_Buffer0->m_DebugTag    = "PlanA_Buffer0";
    planA.m_OutputMappings       = { { planA_Buffer0, partAOutputSlot0 } };

    // Plan B
    Plan planB;
    SramBuffer* planB_Buffer0    = planB.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    planB_Buffer0->m_Format      = CascadingBufferFormat::NHWCB;
    planB_Buffer0->m_TensorShape = TensorShape{ 1, 17, 16, 16 };
    planB_Buffer0->m_StripeShape = TensorShape{ 1, 17, 16, 16 };
    planB_Buffer0->m_Order       = TraversalOrder::Xyz;
    planB_Buffer0->m_SizeInBytes = 4;
    planB_Buffer0->m_DebugTag    = "PlanB_Buffer0";
    planB.m_InputMappings        = { { planB_Buffer0, partBInputSlot0 } };
    planB.m_OutputMappings       = { { planB_Buffer0, partBOutputSlot0 } };

    // Plan C
    Plan planC;
    SramBuffer* planC_Buffer    = planC.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    planC_Buffer->m_Format      = CascadingBufferFormat::NHWCB;
    planC_Buffer->m_TensorShape = TensorShape{ 1, 17, 16, 16 };
    planC_Buffer->m_StripeShape = TensorShape{ 1, 17, 16, 16 };
    planC_Buffer->m_Order       = TraversalOrder::Xyz;
    planC_Buffer->m_SizeInBytes = 4;
    planC_Buffer->m_DebugTag    = "PlanC_Buffer0";
    planC.m_InputMappings       = { { planC_Buffer, partCInputSlot0 } };

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
    Combination combA(partAId, std::move(planA));
    combA.SetEndingGlue(std::move(*endingGlueA), partAOutputSlot0);

    Combination combB(partBId, std::move(planB));
    combB.SetStartingGlue(std::move(*startingGlueB), partBInputSlot0);
    combB.SetEndingGlue(std::move(*endingGlueB), partBOutputSlot0);

    Combination combC(partCId, std::move(planC));
    combC.SetStartingGlue(std::move(*startingGlueC), partCInputSlot0);

    Combination comb = combA + combB + combC;

    bool dumpInputGraphToFile = false;
    if (dumpInputGraphToFile)
    {
        std::ofstream stream("GetOpGraphForDfsCombinationPartialSram Input.dot");
        SaveCombinationToDot(comb, stream, DetailLevel::High);
    }

    // Call function under test
    FrozenGraphOfParts frozenGraph = FrozenGraphOfParts(std::move(graph));
    OpGraph combOpGraph            = GetOpGraphForCombination(comb, frozenGraph);

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
    auto pA         = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB         = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pDE        = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pF         = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pG         = std::make_unique<MockPart>(graph.GeneratePartId());
    PartId partAId  = pA->GetPartId();
    PartId partBId  = pB->GetPartId();
    PartId partDEId = pDE->GetPartId();
    PartId partFId  = pF->GetPartId();
    PartId partGId  = pG->GetPartId();
    graph.AddPart(std::move(pA));
    graph.AddPart(std::move(pB));
    graph.AddPart(std::move(pDE));
    graph.AddPart(std::move(pF));
    graph.AddPart(std::move(pG));

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

    graph.AddConnection(partBInputSlot0, partAOutputSlot0);
    graph.AddConnection(partDEInputSlot0, partBOutputSlot0);
    graph.AddConnection(partDEInputSlot1, partBOutputSlot0);
    graph.AddConnection(partFInputSlot0, partDEOutputSlot0);
    graph.AddConnection(partGInputSlot0, partDEOutputSlot0);
    graph.AddConnection(partGInputSlot1, partDEOutputSlot1);

    Plan planA;
    std::unique_ptr<DramBuffer> inputDramPtr = DramBuffer::Build()
                                                   .AddFormat(CascadingBufferFormat::NHWCB)
                                                   .AddTensorShape(TensorShape{ 1, 17, 16, 16 })
                                                   .AddBufferType(BufferType::Input)
                                                   .AddDebugTag("InputDram");
    DramBuffer* inputDram = planA.m_OpGraph.AddBuffer(std::move(inputDramPtr));

    planA.m_OutputMappings = { { inputDram, partAOutputSlot0 } };

    // Part consisting of node B
    Plan planB;
    SramBuffer* inputSram1    = planB.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    inputSram1->m_Format      = CascadingBufferFormat::NHWCB;
    inputSram1->m_TensorShape = TensorShape{ 1, 17, 16, 16 };
    inputSram1->m_StripeShape = TensorShape{ 1, 17, 16, 16 };
    inputSram1->m_Order       = TraversalOrder::Xyz;
    inputSram1->m_SizeInBytes = 4;
    inputSram1->m_DebugTag    = "InputSram1";
    planB.m_InputMappings     = { { inputSram1, partBInputSlot0 } };
    planB.m_OutputMappings    = { { inputSram1, partBOutputSlot0 } };

    // Part consisting of nodes D and E
    Plan planDE;
    SramBuffer* intermediateSramInput1    = planDE.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    intermediateSramInput1->m_Format      = CascadingBufferFormat::NHWCB;
    intermediateSramInput1->m_TensorShape = TensorShape{ 1, 17, 16, 16 };
    intermediateSramInput1->m_StripeShape = TensorShape{ 1, 17, 16, 16 };
    intermediateSramInput1->m_Order       = TraversalOrder::Xyz;
    intermediateSramInput1->m_SizeInBytes = 4;
    intermediateSramInput1->m_DebugTag    = "IntermediateSramInput1";
    SramBuffer* outputSram1               = planDE.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    outputSram1->m_Format                 = CascadingBufferFormat::NHWCB;
    outputSram1->m_TensorShape            = TensorShape{ 1, 17, 16, 16 };
    outputSram1->m_StripeShape            = TensorShape{ 1, 17, 16, 16 };
    outputSram1->m_Order                  = TraversalOrder::Xyz;
    outputSram1->m_DebugTag               = "OutputSram1";
    SramBuffer* intermediateSramInput2    = planDE.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    intermediateSramInput2->m_Format      = CascadingBufferFormat::NHWCB;
    intermediateSramInput2->m_TensorShape = TensorShape{ 1, 17, 16, 16 };
    intermediateSramInput2->m_StripeShape = TensorShape{ 1, 17, 16, 16 };
    intermediateSramInput2->m_Order       = TraversalOrder::Xyz;
    intermediateSramInput2->m_SizeInBytes = 4;
    intermediateSramInput2->m_DebugTag    = "IntermediateSramInput2";
    SramBuffer* outputSram2               = planDE.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    outputSram2->m_Format                 = CascadingBufferFormat::NHWCB;
    outputSram2->m_TensorShape            = TensorShape{ 1, 17, 16, 16 };
    outputSram2->m_StripeShape            = TensorShape{ 1, 17, 16, 16 };
    outputSram2->m_Order                  = TraversalOrder::Xyz;
    outputSram2->m_DebugTag               = "OutputSram2";
    planDE.m_InputMappings                = { { planDE.m_OpGraph.GetBuffers()[0], partDEInputSlot0 },
                               { planDE.m_OpGraph.GetBuffers()[2], partDEInputSlot1 } };
    planDE.m_OutputMappings               = { { planDE.m_OpGraph.GetBuffers()[1], partDEOutputSlot0 },
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
    std::unique_ptr<DramBuffer> outputDram1Ptr = DramBuffer::Build()
                                                     .AddFormat(CascadingBufferFormat::NHWCB)
                                                     .AddTensorShape(TensorShape{ 1, 17, 16, 16 })
                                                     .AddBufferType(BufferType::Output)
                                                     .AddDebugTag("OutputDram1");
    planF.m_OpGraph.AddBuffer(std::move(outputDram1Ptr));

    planF.m_InputMappings = { { planF.m_OpGraph.GetBuffers()[0], partFInputSlot0 } };

    // Part consisting of node G
    Plan planG;
    std::unique_ptr<DramBuffer> outputDram2Ptr = DramBuffer::Build()
                                                     .AddFormat(CascadingBufferFormat::NHWCB)
                                                     .AddTensorShape(TensorShape{ 1, 17, 16, 16 })
                                                     .AddBufferType(BufferType::Output)
                                                     .AddDebugTag("OutputDram2");
    planG.m_OpGraph.AddBuffer(std::move(outputDram2Ptr));

    std::unique_ptr<DramBuffer> outputDram3Ptr = DramBuffer::Build()
                                                     .AddFormat(CascadingBufferFormat::NHWCB)
                                                     .AddTensorShape(TensorShape{ 1, 17, 16, 16 })
                                                     .AddBufferType(BufferType::Output)
                                                     .AddDebugTag("OutputDram3");
    planG.m_OpGraph.AddBuffer(std::move(outputDram3Ptr));

    planG.m_InputMappings = { { planG.m_OpGraph.GetBuffers()[0], partGInputSlot0 },
                              { planG.m_OpGraph.GetBuffers()[1], partGInputSlot1 } };

    // The end glueing of A is empty. But the starting glue of B has the connections.
    auto startingGlueA = std::make_shared<StartingGlue>();
    auto endingGlueA   = std::make_shared<EndingGlue>();

    std::unique_ptr<DramBuffer> replacementBufferPtr = DramBuffer::Build()
                                                           .AddFormat(CascadingBufferFormat::NHWCB)
                                                           .AddTensorShape(TensorShape{ 1, 17, 16, 16 })
                                                           .AddDebugTag("ReplacementBuffer");
    DramBuffer* replacementBuffer = endingGlueA->m_Graph.AddBuffer(std::move(replacementBufferPtr));

    endingGlueA->m_ExternalConnections.m_ReplacementBuffers.insert(
        { planA.m_OpGraph.GetBuffers()[0], replacementBuffer });

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
    Combination combA(partAId, std::move(planA));
    combA.SetEndingGlue(std::move(*endingGlueA), partAOutputSlot0);

    Combination combB(partBId, std::move(planB));
    combB.SetStartingGlue(std::move(*startingGlueB), partBInputSlot0);
    combB.SetEndingGlue(std::move(*endingGlueB), partBOutputSlot0);

    Combination combDE(partDEId, std::move(planDE));
    combDE.SetStartingGlue(std::move(*startingGlueDE0), partDEInputSlot0);
    combDE.SetStartingGlue(std::move(*startingGlueDE1), partDEInputSlot1);
    combDE.SetEndingGlue(std::move(*endingGlueD), partDEOutputSlot0);
    combDE.SetEndingGlue(std::move(*endingGlueE), partDEOutputSlot1);

    Combination combF(partFId, std::move(planF));
    combF.SetStartingGlue(std::move(*startingGlueF), partFInputSlot0);

    Combination combG(partGId, std::move(planG));
    combG.SetStartingGlue(std::move(*startingGluefromDtoG), partGInputSlot0);
    combG.SetStartingGlue(std::move(*startingGluefromEtoG), partGInputSlot1);

    Combination comb = combA + combB + combDE + combF + combG;

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("GetOpGraphForCombination Input.dot");
        SaveCombinationToDot(comb, stream, DetailLevel::High);
    }

    // Call function under test
    FrozenGraphOfParts frozenGraph = FrozenGraphOfParts(std::move(graph));
    OpGraph combOpGraph            = GetOpGraphForCombination(comb, frozenGraph);

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

    REQUIRE(combOpGraph.GetSingleProducer(combOpGraph.GetBuffers()[0]) == nullptr);
    REQUIRE(combOpGraph.GetSingleProducer(combOpGraph.GetBuffers()[1])->m_DebugTag == "InputDma");
    REQUIRE(combOpGraph.GetSingleProducer(combOpGraph.GetBuffers()[2])->m_DebugTag == "Mce2");
    REQUIRE(combOpGraph.GetSingleProducer(combOpGraph.GetBuffers()[3])->m_DebugTag == "Mce2");
    REQUIRE(combOpGraph.GetSingleProducer(combOpGraph.GetBuffers()[4])->m_DebugTag == "OutputDma1");
    REQUIRE(combOpGraph.GetSingleProducer(combOpGraph.GetBuffers()[5])->m_DebugTag == "OutputDma2");
    REQUIRE(combOpGraph.GetSingleProducer(combOpGraph.GetBuffers()[6])->m_DebugTag == "OutputDma3");

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

    auto pA               = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB               = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pC               = std::make_unique<MockPart>(graph.GeneratePartId());
    const BasePart& partA = *pA;
    const BasePart& partB = *pB;
    const BasePart& partC = *pC;
    graph.AddPart(std::move(pA));
    graph.AddPart(std::move(pB));
    graph.AddPart(std::move(pC));

    Plan planA;
    Plan planB;
    Plan planC;

    Combination combA(partA.GetPartId(), std::move(planA));
    Combination combB(partB.GetPartId(), std::move(planB));
    Combination combC(partC.GetPartId(), std::move(planC));

    REQUIRE(combA.GetEndPartId() - combA.GetFirstPartId() == 1);
    REQUIRE(combB.GetEndPartId() - combB.GetFirstPartId() == 1);
    REQUIRE(combC.GetEndPartId() - combC.GetFirstPartId() == 1);

    Combination comb;
    REQUIRE(comb.GetEndPartId() - comb.GetFirstPartId() == 0);

    comb = combA + combB + combC;
    REQUIRE(comb.GetEndPartId() - comb.GetFirstPartId() == 3);
    // All parts are in the final combination
    for (const std::pair<const PartId, std::unique_ptr<BasePart>>& idAndPart : graph.GetParts())
    {
        REQUIRE_NOTHROW(comb.GetElem(idAndPart.first));
    }
}

TEST_CASE("Combination AddGlue", "[CombinerDFS]")
{
    // Create graph:
    //
    //  A - B
    //
    GraphOfParts graph;

    auto pA               = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB               = std::make_unique<MockPart>(graph.GeneratePartId());
    const BasePart& partA = *pA;
    const BasePart& partB = *pB;
    PartId partAId        = partA.GetPartId();
    PartId partBId        = partB.GetPartId();
    graph.AddPart(std::move(pA));
    graph.AddPart(std::move(pB));

    PartOutputSlot partAOutputSlot0 = { partAId, 0 };

    PartInputSlot partBInputSlot0 = { partBId, 0 };

    Plan planA;
    Plan planB;

    Combination combA(partA.GetPartId(), std::move(planA));
    Combination combB(partB.GetPartId(), std::move(planB));

    REQUIRE(combA.GetEndPartId() - combA.GetFirstPartId() == 1);
    REQUIRE(combB.GetEndPartId() - combB.GetFirstPartId() == 1);

    EndingGlue endingGlueA;
    combA.SetEndingGlue(std::move(endingGlueA), partAOutputSlot0);
    REQUIRE(combA.GetElem(partAOutputSlot0.m_PartId).m_EndingGlues.size() == 1);

    StartingGlue startingGlueB;
    combB.SetStartingGlue(std::move(startingGlueB), partBInputSlot0);
    REQUIRE(combB.GetElem(partBInputSlot0.m_PartId).m_StartingGlues.size() == 1);

    Combination comb = combA + combB;
    // Adding combinations adds their glues.
    REQUIRE(comb.GetElem(partAOutputSlot0.m_PartId).m_EndingGlues.size() == 1);
    REQUIRE(comb.GetElem(partBInputSlot0.m_PartId).m_StartingGlues.size() == 1);
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
    auto pA = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pC = std::make_unique<MockPart>(graph.GeneratePartId());

    BasePart& partA = *pA;
    BasePart& partB = *pB;
    BasePart& partC = *pC;

    PartId partAId = pA->GetPartId();
    PartId partBId = pB->GetPartId();
    PartId partCId = pC->GetPartId();

    graph.AddPart(std::move(pA));
    graph.AddPart(std::move(pB));
    graph.AddPart(std::move(pC));

    PartOutputSlot partAOutputSlot = { partA.GetPartId(), 0 };

    PartInputSlot partBInputSlot = { partB.GetPartId(), 0 };
    PartInputSlot partCInputSlot = { partC.GetPartId(), 0 };

    graph.AddConnection(partBInputSlot, { partAOutputSlot });
    graph.AddConnection(partCInputSlot, { partAOutputSlot });

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Plan planA;
    SramBuffer* bufferA    = planA.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    bufferA->m_Format      = CascadingBufferFormat::NHWCB;
    bufferA->m_TensorShape = TensorShape{ 1, 64, 64, 64 };
    bufferA->m_StripeShape = TensorShape{ 1, 8, 8, 32 };
    bufferA->m_Order       = TraversalOrder::Xyz;
    bufferA->m_SizeInBytes = 4;
    planA.m_OutputMappings = { { planA.m_OpGraph.GetBuffers()[0], partAOutputSlot } };

    Plan planB;
    SramBuffer* bufferB    = planB.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    bufferB->m_Format      = CascadingBufferFormat::NHWCB;
    bufferB->m_TensorShape = TensorShape{ 1, 64, 64, 64 };
    bufferB->m_StripeShape = TensorShape{ 1, 8, 8, 32 };
    bufferB->m_Order       = TraversalOrder::Xyz;
    bufferB->m_SizeInBytes = 4;
    planB.m_InputMappings  = { { planB.m_OpGraph.GetBuffers()[0], partBInputSlot } };

    Plan planC;
    SramBuffer* bufferC    = planC.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    bufferC->m_Format      = CascadingBufferFormat::NHWCB;
    bufferC->m_TensorShape = TensorShape{ 1, 64, 64, 64 };
    bufferC->m_StripeShape = TensorShape{ 1, 8, 8, 32 };
    bufferC->m_Order       = TraversalOrder::Xyz;
    bufferC->m_SizeInBytes = 4;
    planC.m_InputMappings  = { { planC.m_OpGraph.GetBuffers()[0], partCInputSlot } };

    Combination combA(partA.GetPartId(), std::move(planA));
    Combination combB(partB.GetPartId(), std::move(planB));
    Combination combC(partC.GetPartId(), std::move(planC));

    // Merge the combinations
    Combination comb = combA + combB + combC;

    // There is no glue
    for (const std::pair<const PartId, std::unique_ptr<BasePart>>& idAndPart : graph.GetParts())
    {
        REQUIRE(comb.GetElem(idAndPart.first).m_EndingGlues.size() == 0);
        REQUIRE(comb.GetElem(idAndPart.first).m_StartingGlues.size() == 0);
    }

    FrozenGraphOfParts frozenGraph = FrozenGraphOfParts(std::move(graph));
    CombinerTest combiner(frozenGraph, hwCaps, compOpt, estOpt, debuggingContext);

    Combination combGlued = combiner.GluePartToCombinationSrcToDests(partA, comb, 0);

    // One glue shared by A-B, A-C
    // The glue has (1) 1 x input DMA (2) DRAM buffer (3) 2 x ouput DMA

    REQUIRE(combGlued.GetEndPartId() - combGlued.GetFirstPartId() == 3);

    OpGraph opGraph = GetOpGraphForCombination(combGlued, frozenGraph);

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("GluePartToCombinationBranch0.dot");
        SaveOpGraphToDot(opGraph, stream, DetailLevel::High);
    }

    // Part A should have 1 ending glue containing a dma and a dram buffer
    {
        auto elemIt = combGlued.GetElem(partAId);
        REQUIRE(elemIt.m_StartingGlues.size() == 0);
        REQUIRE(elemIt.m_EndingGlues.size() == 1);
        REQUIRE(elemIt.m_EndingGlues.begin()->second->m_Graph.GetBuffers().size() == 1);
        REQUIRE(elemIt.m_EndingGlues.begin()->second->m_Graph.GetBuffers()[0]->m_Location == Location::Dram);
        REQUIRE(elemIt.m_EndingGlues.begin()->second->m_Graph.GetBuffers()[0]->m_Format ==
                CascadingBufferFormat::FCAF_DEEP);
        REQUIRE(elemIt.m_EndingGlues.begin()->second->m_Graph.GetOps().size() == 1);
    }
    // Part B and C should have 1 starting glue containing just 1 dma op each.
    {
        auto elemIt = combGlued.GetElem(partBId);
        REQUIRE(elemIt.m_StartingGlues.size() == 1);
        REQUIRE(elemIt.m_EndingGlues.size() == 0);
        REQUIRE(elemIt.m_StartingGlues.begin()->second->m_Graph.GetBuffers().size() == 0);
        REQUIRE(elemIt.m_StartingGlues.begin()->second->m_Graph.GetOps().size() == 1);
    }
    {
        auto elemIt = combGlued.GetElem(partCId);
        REQUIRE(elemIt.m_StartingGlues.size() == 1);
        REQUIRE(elemIt.m_EndingGlues.size() == 0);
        REQUIRE(elemIt.m_StartingGlues.begin()->second->m_Graph.GetBuffers().size() == 0);
        REQUIRE(elemIt.m_StartingGlues.begin()->second->m_Graph.GetOps().size() == 1);
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
    //  D is an output node in DRAM and cannot share
    //  glue with B, C
    GraphOfParts graph;
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

    graph.AddPart(std::move(pA));
    graph.AddPart(std::move(pB));
    graph.AddPart(std::move(pC));
    graph.AddPart(std::move(pD));

    PartOutputSlot partAOutputSlot = { partA.GetPartId(), 0 };

    PartInputSlot partBInputSlot = { partB.GetPartId(), 0 };
    PartInputSlot partCInputSlot = { partC.GetPartId(), 0 };
    PartInputSlot partDInputSlot = { partD.GetPartId(), 0 };

    graph.AddConnection(partBInputSlot, { partAOutputSlot });
    graph.AddConnection(partCInputSlot, { partAOutputSlot });
    graph.AddConnection(partDInputSlot, { partAOutputSlot });

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Plan planA;
    SramBuffer* bufferA    = planA.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    bufferA->m_Format      = CascadingBufferFormat::NHWCB;
    bufferA->m_TensorShape = TensorShape{ 1, 64, 64, 64 };
    bufferA->m_StripeShape = TensorShape{ 1, 8, 8, 32 };
    bufferA->m_Order       = TraversalOrder::Xyz;
    bufferA->m_SizeInBytes = 4;
    planA.m_OutputMappings = { { planA.m_OpGraph.GetBuffers()[0], partAOutputSlot } };

    Plan planB;
    SramBuffer* bufferB    = planB.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    bufferB->m_Format      = CascadingBufferFormat::NHWCB;
    bufferB->m_TensorShape = TensorShape{ 1, 64, 64, 64 };
    bufferB->m_StripeShape = TensorShape{ 1, 8, 8, 32 };
    bufferB->m_Order       = TraversalOrder::Xyz;
    bufferB->m_SizeInBytes = 4;
    planB.m_InputMappings  = { { planB.m_OpGraph.GetBuffers()[0], partBInputSlot } };

    Plan planC;
    SramBuffer* bufferC    = planC.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    bufferC->m_Format      = CascadingBufferFormat::NHWCB;
    bufferC->m_TensorShape = TensorShape{ 1, 64, 64, 64 };
    bufferC->m_StripeShape = TensorShape{ 1, 8, 8, 32 };
    bufferC->m_Order       = TraversalOrder::Xyz;
    bufferC->m_SizeInBytes = 4;
    planC.m_InputMappings  = { { planC.m_OpGraph.GetBuffers()[0], partCInputSlot } };

    Plan planD;
    std::unique_ptr<DramBuffer> bufferDPtr = DramBuffer::Build()
                                                 .AddFormat(CascadingBufferFormat::NHWCB)
                                                 .AddTensorShape(TensorShape{ 1, 64, 64, 64 })
                                                 .AddSizeInBytes(4);
    planD.m_OpGraph.AddBuffer(std::move(bufferDPtr));

    planD.m_InputMappings = { { planD.m_OpGraph.GetBuffers()[0], partDInputSlot } };
    auto dummyOpD         = std::make_unique<DummyOp>();
    dummyOpD->m_DebugTag  = "DummyD";
    planD.m_OpGraph.AddOp(std::move(dummyOpD));
    planD.m_OpGraph.AddConsumer(planD.m_OpGraph.GetBuffers().back(), planD.m_OpGraph.GetOps().back(), 0);

    Combination combA(partA.GetPartId(), std::move(planA));
    Combination combB(partB.GetPartId(), std::move(planB));
    Combination combC(partC.GetPartId(), std::move(planC));
    Combination combD(partD.GetPartId(), std::move(planD));

    // Merge the combinations
    Combination comb = combA + combB + combC + combD;

    // There is no glue
    for (const std::pair<const PartId, std::unique_ptr<BasePart>>& idAndPart : graph.GetParts())
    {
        REQUIRE(comb.GetElem(idAndPart.first).m_EndingGlues.size() == 0);
        REQUIRE(comb.GetElem(idAndPart.first).m_StartingGlues.size() == 0);
    }

    FrozenGraphOfParts frozenGraph = FrozenGraphOfParts(std::move(graph));
    CombinerTest combiner(frozenGraph, hwCaps, compOpt, estOpt, debuggingContext);

    std::vector<PartConnection> destPartEdge;

    Combination combGlued = combiner.GluePartToCombinationSrcToDests(partA, comb, 0);

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("GluePartToCombinationBranch1.dot");
        SaveCombinationToDot(combGlued, stream, DetailLevel::High);
    }

    REQUIRE(combGlued.GetEndPartId() - combGlued.GetFirstPartId() == 4);

    // Ending glue of A copies to an FCAF buffer for use by B and C, and has a DmaOp for D
    auto elemA = combGlued.GetElem(partAId);
    REQUIRE(elemA.m_EndingGlues.size() == 1);
    OpGraph& opGraphA = elemA.m_Plan->m_OpGraph;

    auto endingGlueA = elemA.m_EndingGlues[partAOutputSlot];
    REQUIRE(endingGlueA->m_Graph.GetBuffers().size() == 1);
    REQUIRE(endingGlueA->m_Graph.GetOps().size() == 2);
    REQUIRE(endingGlueA->m_Graph.GetSingleProducer(endingGlueA->m_Graph.GetBuffers().front()) ==
            endingGlueA->m_Graph.GetOp(1));
    const auto& planABuffers = opGraphA.GetBuffers();
    REQUIRE(endingGlueA->m_ExternalConnections.m_BuffersToOps ==
            std::multimap<Buffer*, Op*>{
                { planABuffers.back(), endingGlueA->m_Graph.GetOp(0) },
                { planABuffers.back(), endingGlueA->m_Graph.GetOp(1) },
            });

    // Starting glue of B and C copy from that FCAF buffer
    auto elemB               = combGlued.GetElem(partBId);
    const auto& planBBuffers = elemB.m_Plan->m_OpGraph.GetBuffers();
    auto startingGlueB       = elemB.m_StartingGlues[partBInputSlot];
    REQUIRE(startingGlueB->m_Graph.GetBuffers().size() == 0);
    REQUIRE(startingGlueB->m_Graph.GetOps().size() == 1);
    REQUIRE(
        startingGlueB->m_ExternalConnections.m_BuffersToOps.find(endingGlueA->m_Graph.GetBuffers().back())->second ==
        startingGlueB->m_Graph.GetOp(0));
    REQUIRE(startingGlueB->m_ExternalConnections.m_OpsToBuffers.find(startingGlueB->m_Graph.GetOp(0))->second ==
            planBBuffers.front());

    auto elemC               = combGlued.GetElem(partCId);
    const auto& planCBuffers = elemC.m_Plan->m_OpGraph.GetBuffers();
    auto startingGlueC       = elemC.m_StartingGlues[partCInputSlot];
    REQUIRE(startingGlueC->m_Graph.GetBuffers().size() == 0);
    REQUIRE(startingGlueC->m_Graph.GetOps().size() == 1);
    REQUIRE(
        startingGlueC->m_ExternalConnections.m_BuffersToOps.find(endingGlueA->m_Graph.GetBuffers().back())->second ==
        startingGlueC->m_Graph.GetOp(0));
    REQUIRE(startingGlueC->m_ExternalConnections.m_OpsToBuffers.find(startingGlueC->m_Graph.GetOp(0))->second ==
            planCBuffers.front());

    // Starting glue of D copies from the original SRAM buffer. Note that the buffer of D
    // cannot be re-used by the other glues, as it is an output buffer
    auto elemD               = combGlued.GetElem(partDId);
    const auto& planDBuffers = elemD.m_Plan->m_OpGraph.GetBuffers();
    auto startingGlueD       = elemD.m_StartingGlues[partDInputSlot];
    REQUIRE(startingGlueD->m_Graph.GetOps().size() == 0);
    REQUIRE(startingGlueD->m_ExternalConnections.m_BuffersToOps.size() == 0);
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

    graph.AddPart(std::move(pA));
    graph.AddPart(std::move(pB));
    graph.AddPart(std::move(pC));
    graph.AddPart(std::move(pD));
    graph.AddPart(std::move(pE));

    PartOutputSlot partAOutputSlot = { partA.GetPartId(), 0 };
    PartOutputSlot partDOutputSlot = { partD.GetPartId(), 0 };

    PartInputSlot partBInputSlot = { partB.GetPartId(), 0 };
    PartInputSlot partCInputSlot = { partC.GetPartId(), 0 };
    PartInputSlot partDInputSlot = { partD.GetPartId(), 0 };
    PartInputSlot partEInputSlot = { partE.GetPartId(), 0 };

    graph.AddConnection(partBInputSlot, { partAOutputSlot });
    graph.AddConnection(partCInputSlot, { partAOutputSlot });
    graph.AddConnection(partDInputSlot, { partAOutputSlot });
    graph.AddConnection(partEInputSlot, { partDOutputSlot });

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Plan planA;
    SramBuffer* bufferA    = planA.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    bufferA->m_Format      = CascadingBufferFormat::NHWCB;
    bufferA->m_TensorShape = TensorShape{ 1, 64, 64, 64 };
    bufferA->m_StripeShape = TensorShape{ 1, 8, 8, 32 };
    bufferA->m_Order       = TraversalOrder::Xyz;
    bufferA->m_SizeInBytes = 4;
    planA.m_OutputMappings = { { planA.m_OpGraph.GetBuffers()[0], partAOutputSlot } };

    Plan planB;
    SramBuffer* bufferB    = planB.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    bufferB->m_Format      = CascadingBufferFormat::NHWCB;
    bufferB->m_TensorShape = TensorShape{ 1, 64, 64, 64 };
    bufferB->m_StripeShape = TensorShape{ 1, 8, 8, 32 };
    bufferB->m_Order       = TraversalOrder::Xyz;
    bufferB->m_SizeInBytes = 4;
    planB.m_InputMappings  = { { planB.m_OpGraph.GetBuffers()[0], partBInputSlot } };

    Plan planC;
    SramBuffer* bufferC    = planC.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    bufferC->m_Format      = CascadingBufferFormat::NHWCB;
    bufferC->m_TensorShape = TensorShape{ 1, 64, 64, 64 };
    bufferC->m_StripeShape = TensorShape{ 1, 8, 8, 32 };
    bufferC->m_Order       = TraversalOrder::Xyz;
    bufferC->m_SizeInBytes = 4;
    planC.m_InputMappings  = { { planC.m_OpGraph.GetBuffers()[0], partCInputSlot } };

    Plan planD;
    std::unique_ptr<DramBuffer> bufferDPtr = DramBuffer::Build()
                                                 .AddFormat(CascadingBufferFormat::NHWC)
                                                 .AddTensorShape(TensorShape{ 1, 64, 64, 64 })
                                                 .AddSizeInBytes(4);
    planD.m_OpGraph.AddBuffer(std::move(bufferDPtr));

    planD.m_InputMappings = { { planD.m_OpGraph.GetBuffers()[0], partDInputSlot } };

    Plan planE;
    std::unique_ptr<DramBuffer> bufferEPtr = DramBuffer::Build()
                                                 .AddFormat(CascadingBufferFormat::NHWCB)
                                                 .AddTensorShape(TensorShape{ 1, 64, 64, 64 })
                                                 .AddSizeInBytes(4);
    planE.m_OpGraph.AddBuffer(std::move(bufferEPtr));

    planE.m_InputMappings = { { planE.m_OpGraph.GetBuffers()[0], partDInputSlot } };

    Combination combA(partA.GetPartId(), std::move(planA));
    Combination combB(partB.GetPartId(), std::move(planB));
    Combination combC(partC.GetPartId(), std::move(planC));
    Combination combD(partD.GetPartId(), std::move(planD));
    Combination combE(partE.GetPartId(), std::move(planE));

    // Merge the combinations
    Combination comb = combA + combB + combC + combD + combE;

    // There is no glue
    for (const std::pair<const PartId, std::unique_ptr<BasePart>>& idAndPart : graph.GetParts())
    {
        REQUIRE(comb.GetElem(idAndPart.first).m_EndingGlues.size() == 0);
        REQUIRE(comb.GetElem(idAndPart.first).m_StartingGlues.size() == 0);
    }

    FrozenGraphOfParts frozenGraph = FrozenGraphOfParts(std::move(graph));
    CombinerTest combiner(frozenGraph, hwCaps, compOpt, estOpt, debuggingContext);

    Combination combGlued = combiner.GluePartToCombinationSrcToDests(partA, comb, 0);

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("GluePartToCombinationBranch2.dot");
        SaveCombinationToDot(combGlued, stream, DetailLevel::High);
    }

    // One glue shared by A-B, A-C (SRAM - SRAM) and A-D
    // A-D partially shares this same glue, but requires some extra DMAs afterwards.
    // Note part D's input buffer in DRAM is NHWC so that it
    // cannot share glue with others.
    REQUIRE(combGlued.GetEndPartId() - combGlued.GetFirstPartId() == 5);

    auto elemA              = combGlued.GetElem(partAId);
    EndingGlue* endingGlueA = elemA.m_EndingGlues.find(partAOutputSlot)->second.get();
    OpGraph& opGraphA       = elemA.m_Plan->m_OpGraph;
    REQUIRE(endingGlueA->m_Graph.GetOps().size() == 3);
    REQUIRE(endingGlueA->m_Graph.GetBuffers().size() == 2);
    REQUIRE(endingGlueA->m_ExternalConnections.m_BuffersToOps ==
            std::multimap<Buffer*, Op*>{ { opGraphA.GetBuffers()[0], endingGlueA->m_Graph.GetOp(0) } });

    auto elemB                  = combGlued.GetElem(partBId);
    StartingGlue* startingGlueB = elemB.m_StartingGlues.find(partBInputSlot)->second.get();
    OpGraph& opGraphB           = elemB.m_Plan->m_OpGraph;
    REQUIRE(startingGlueB->m_Graph.GetOps().size() == 1);
    REQUIRE(startingGlueB->m_Graph.GetBuffers().size() == 0);
    REQUIRE(startingGlueB->m_ExternalConnections.m_BuffersToOps.find(endingGlueA->m_Graph.GetBuffers()[0])->second ==
            startingGlueB->m_Graph.GetOp(0));
    REQUIRE(startingGlueB->m_ExternalConnections.m_OpsToBuffers.find(startingGlueB->m_Graph.GetOp(0))->second ==
            opGraphB.GetBuffers()[0]);

    auto elemC                  = combGlued.GetElem(partCId);
    StartingGlue* startingGlueC = elemC.m_StartingGlues.find(partCInputSlot)->second.get();
    OpGraph& opGraphC           = elemC.m_Plan->m_OpGraph;
    REQUIRE(startingGlueC->m_Graph.GetOps().size() == 1);
    REQUIRE(startingGlueC->m_Graph.GetBuffers().size() == 0);
    REQUIRE(startingGlueC->m_ExternalConnections.m_BuffersToOps.find(endingGlueA->m_Graph.GetBuffers()[0])->second ==
            startingGlueC->m_Graph.GetOp(0));
    REQUIRE(startingGlueC->m_ExternalConnections.m_OpsToBuffers.find(startingGlueC->m_Graph.GetOp(0))->second ==
            opGraphC.GetBuffers()[0]);

    auto elemD                  = combGlued.GetElem(partDId);
    StartingGlue* startingGlueD = elemD.m_StartingGlues.find(partDInputSlot)->second.get();
    OpGraph& opGraphD           = elemD.m_Plan->m_OpGraph;
    REQUIRE(startingGlueD->m_Graph.GetOps().size() == 0);
    REQUIRE(startingGlueD->m_Graph.GetBuffers().size() == 0);
    REQUIRE(startingGlueD->m_ExternalConnections.m_BuffersToOps.size() == 0);
    REQUIRE(startingGlueD->m_ExternalConnections.m_OpsToBuffers ==
            std::multimap<Op*, Buffer*>{ { endingGlueA->m_Graph.GetOps()[2], opGraphD.GetBuffers()[0] } });
}

TEST_CASE("GluePartToCombinationDramToDramAndSramShare", "[CombinerDFS]")
{
    // Create graph:
    //
    //
    //   - - > B
    //  |
    //  A - -> C
    //
    //  A is DRAM NHWC
    //  B is DRAM NHWCB, and so requires a conversion from A
    //  C is SRAM, and cannot be copied directly from A because it splits in depth. It can however use the DRAM buffer B
    //  and copy straight from there.
    GraphOfParts graph;
    auto pA = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pC = std::make_unique<MockPart>(graph.GeneratePartId());

    BasePart& partA = *pA;
    BasePart& partB = *pB;
    BasePart& partC = *pC;

    PartId partAId = pA->GetPartId();
    PartId partBId = pB->GetPartId();
    PartId partCId = pC->GetPartId();

    graph.AddPart(std::move(pA));
    graph.AddPart(std::move(pB));
    graph.AddPart(std::move(pC));

    PartOutputSlot partAOutputSlot = { partA.GetPartId(), 0 };

    PartInputSlot partBInputSlot = { partB.GetPartId(), 0 };
    PartInputSlot partCInputSlot = { partC.GetPartId(), 0 };

    graph.AddConnection(partBInputSlot, { partAOutputSlot });
    graph.AddConnection(partCInputSlot, { partAOutputSlot });

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Plan planA;
    std::unique_ptr<DramBuffer> bufferAPtr = DramBuffer::Build()
                                                 .AddFormat(CascadingBufferFormat::NHWC)
                                                 .AddTensorShape(TensorShape{ 1, 64, 64, 64 })
                                                 .AddSizeInBytes(4)
                                                 .AddBufferType(BufferType::Intermediate);
    planA.m_OpGraph.AddBuffer(std::move(bufferAPtr));

    planA.m_OutputMappings = { { planA.m_OpGraph.GetBuffers()[0], partAOutputSlot } };

    Plan planB;
    std::unique_ptr<DramBuffer> bufferBPtr = DramBuffer::Build()
                                                 .AddFormat(CascadingBufferFormat::NHWCB)
                                                 .AddTensorShape(TensorShape{ 1, 64, 64, 64 })
                                                 .AddSizeInBytes(4)
                                                 .AddBufferType(BufferType::Intermediate);
    planB.m_OpGraph.AddBuffer(std::move(bufferBPtr));

    planB.m_InputMappings = { { planB.m_OpGraph.GetBuffers()[0], partBInputSlot } };

    Plan planC;
    SramBuffer* bufferC    = planC.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    bufferC->m_Format      = CascadingBufferFormat::NHWCB;
    bufferC->m_TensorShape = TensorShape{ 1, 64, 64, 64 };
    bufferC->m_StripeShape = TensorShape{ 1, 8, 8, 32 };
    bufferC->m_Order       = TraversalOrder::Xyz;
    bufferC->m_SizeInBytes = 4;
    planC.m_InputMappings  = { { planC.m_OpGraph.GetBuffers()[0], partCInputSlot } };

    Combination combA(partA.GetPartId(), std::move(planA));
    Combination combB(partB.GetPartId(), std::move(planB));
    Combination combC(partC.GetPartId(), std::move(planC));

    // Merge the combinations
    Combination comb = combA + combB + combC;

    FrozenGraphOfParts frozenGraph = FrozenGraphOfParts(std::move(graph));
    CombinerTest combiner(frozenGraph, hwCaps, compOpt, estOpt, debuggingContext);

    Combination combGlued = combiner.GluePartToCombinationSrcToDests(partA, comb, 0);

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("GluePartToCombinationDramToDramAndSramShare.dot");
        SaveCombinationToDot(combGlued, stream, DetailLevel::High);
    }

    // Ending glue of A converts to NHWCB in DRAM.
    // Starting glue of B replaces that NHWCB buffer
    // Starting glue of C copies from that NHWCB buffer.
    REQUIRE(combGlued.GetEndPartId() - combGlued.GetFirstPartId() == 3);

    auto elemA              = combGlued.GetElem(partAId);
    EndingGlue* endingGlueA = elemA.m_EndingGlues.find(partAOutputSlot)->second.get();
    OpGraph& opGraphA       = elemA.m_Plan->m_OpGraph;
    REQUIRE(endingGlueA->m_Graph.GetOps().size() == 2);
    REQUIRE(endingGlueA->m_Graph.GetBuffers().size() == 2);
    REQUIRE(endingGlueA->m_Graph.GetBuffers()[0]->m_Format == CascadingBufferFormat::NHWCB);
    REQUIRE(endingGlueA->m_ExternalConnections.m_BuffersToOps ==
            std::multimap<Buffer*, Op*>{ { opGraphA.GetBuffers()[0], endingGlueA->m_Graph.GetOp(0) } });

    auto elemB                  = combGlued.GetElem(partBId);
    StartingGlue* startingGlueB = elemB.m_StartingGlues.find(partBInputSlot)->second.get();
    OpGraph& opGraphB           = elemB.m_Plan->m_OpGraph;
    REQUIRE(startingGlueB->m_Graph.GetOps().size() == 0);
    REQUIRE(startingGlueB->m_Graph.GetBuffers().size() == 0);
    REQUIRE(startingGlueB->m_ExternalConnections.m_ReplacementBuffers ==
            std::unordered_map<Buffer*, Buffer*>{ { opGraphB.GetBuffers()[0], endingGlueA->m_Graph.GetBuffers()[0] } });

    auto elemC                  = combGlued.GetElem(partCId);
    StartingGlue* startingGlueC = elemC.m_StartingGlues.find(partCInputSlot)->second.get();
    OpGraph& opGraphC           = elemC.m_Plan->m_OpGraph;
    REQUIRE(startingGlueC->m_Graph.GetOps().size() == 1);
    REQUIRE(startingGlueC->m_Graph.GetBuffers().size() == 0);
    REQUIRE(startingGlueC->m_ExternalConnections.m_BuffersToOps.find(endingGlueA->m_Graph.GetBuffers()[0])->second ==
            startingGlueC->m_Graph.GetOp(0));
    REQUIRE(startingGlueC->m_ExternalConnections.m_OpsToBuffers.find(startingGlueC->m_Graph.GetOp(0))->second ==
            opGraphC.GetBuffers()[0]);
}

TEST_CASE("GluePartToCombinationDramToDramAndSramMergeShare", "[CombinerDFS]")
{
    // Create graph:
    //
    //
    //   - - > B
    //  |
    //  A - -> C
    //
    //  A is DRAM NHWCB
    //  B is DRAM NHWCB, and can be a simple replacement of A
    //  C is SRAM, and can be DMA'd from A
    GraphOfParts graph;
    auto pA = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pC = std::make_unique<MockPart>(graph.GeneratePartId());

    BasePart& partA = *pA;
    BasePart& partB = *pB;
    BasePart& partC = *pC;

    PartId partAId = pA->GetPartId();
    PartId partBId = pB->GetPartId();
    PartId partCId = pC->GetPartId();

    graph.AddPart(std::move(pA));
    graph.AddPart(std::move(pB));
    graph.AddPart(std::move(pC));

    PartOutputSlot partAOutputSlot = { partA.GetPartId(), 0 };

    PartInputSlot partBInputSlot = { partB.GetPartId(), 0 };
    PartInputSlot partCInputSlot = { partC.GetPartId(), 0 };

    graph.AddConnection(partBInputSlot, { partAOutputSlot });
    graph.AddConnection(partCInputSlot, { partAOutputSlot });

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Plan planA;
    std::unique_ptr<DramBuffer> bufferAPtr = DramBuffer::Build()
                                                 .AddFormat(CascadingBufferFormat::NHWCB)
                                                 .AddTensorShape(TensorShape{ 1, 64, 64, 64 })
                                                 .AddSizeInBytes(4)
                                                 .AddBufferType(BufferType::Intermediate);
    DramBuffer* bufferA = planA.m_OpGraph.AddBuffer(std::move(bufferAPtr));

    planA.m_OutputMappings = { { bufferA, partAOutputSlot } };

    Plan planB;
    std::unique_ptr<DramBuffer> bufferBPtr = DramBuffer::Build()
                                                 .AddFormat(CascadingBufferFormat::NHWCB)
                                                 .AddTensorShape(TensorShape{ 1, 64, 64, 64 })
                                                 .AddSizeInBytes(4)
                                                 .AddBufferType(BufferType::Intermediate);
    DramBuffer* bufferB = planB.m_OpGraph.AddBuffer(std::move(bufferBPtr));

    planB.m_InputMappings = { { bufferB, partBInputSlot } };

    Plan planC;
    SramBuffer* bufferC    = planC.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    bufferC->m_Format      = CascadingBufferFormat::NHWCB;
    bufferC->m_TensorShape = TensorShape{ 1, 64, 64, 64 };
    bufferC->m_StripeShape = TensorShape{ 1, 8, 8, 32 };
    bufferC->m_Order       = TraversalOrder::Xyz;
    bufferC->m_SizeInBytes = 4;
    planC.m_InputMappings  = { { bufferC, partCInputSlot } };

    Combination combA(partA.GetPartId(), std::move(planA));
    Combination combB(partB.GetPartId(), std::move(planB));
    Combination combC(partC.GetPartId(), std::move(planC));

    // Merge the combinations
    Combination comb = combA + combB + combC;

    FrozenGraphOfParts frozenGraph = FrozenGraphOfParts(std::move(graph));
    CombinerTest combiner(frozenGraph, hwCaps, compOpt, estOpt, debuggingContext);

    Combination combGlued = combiner.GluePartToCombinationSrcToDests(partA, comb, 0);

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("GluePartToCombinationDramToDramAndSramMergeShare.dot");
        SaveCombinationToDot(combGlued, stream, DetailLevel::High);
    }

    // Ending glue of A is empty
    // Starting glue of B links to the existing buffer in A
    // Starting glue of C copies from the original buffer in A.
    REQUIRE(combGlued.GetEndPartId() - combGlued.GetFirstPartId() == 3);

    auto elemA              = combGlued.GetElem(partAId);
    EndingGlue* endingGlueA = elemA.m_EndingGlues.find(partAOutputSlot)->second.get();
    OpGraph& opGraphA       = elemA.m_Plan->m_OpGraph;
    REQUIRE(endingGlueA->m_Graph.GetOps().size() == 0);
    REQUIRE(endingGlueA->m_Graph.GetBuffers().size() == 0);

    auto elemB                  = combGlued.GetElem(partBId);
    StartingGlue* startingGlueB = elemB.m_StartingGlues.find(partBInputSlot)->second.get();
    OpGraph& opGraphB           = elemB.m_Plan->m_OpGraph;
    REQUIRE(startingGlueB->m_Graph.GetOps().size() == 0);
    REQUIRE(startingGlueB->m_Graph.GetBuffers().size() == 0);
    REQUIRE(startingGlueB->m_ExternalConnections.m_ReplacementBuffers ==
            std::unordered_map<Buffer*, Buffer*>{ { opGraphB.GetBuffers()[0], opGraphA.GetBuffers()[0] } });

    auto elemC                  = combGlued.GetElem(partCId);
    StartingGlue* startingGlueC = elemC.m_StartingGlues.find(partCInputSlot)->second.get();
    OpGraph& opGraphC           = elemC.m_Plan->m_OpGraph;
    REQUIRE(startingGlueC->m_Graph.GetOps().size() == 1);
    REQUIRE(startingGlueC->m_Graph.GetBuffers().size() == 0);
    REQUIRE(startingGlueC->m_ExternalConnections.m_BuffersToOps.find(opGraphA.GetBuffers()[0])->second ==
            startingGlueC->m_Graph.GetOp(0));
    REQUIRE(startingGlueC->m_ExternalConnections.m_OpsToBuffers.find(startingGlueC->m_Graph.GetOp(0))->second ==
            opGraphC.GetBuffers()[0]);

    // Check that the replacements are handled correctly when converting to an OpGraph
    OpGraph combOpGraph = GetOpGraphForCombination(combGlued, frozenGraph);

    if (dumpToFile)
    {
        std::ofstream stream("GluePartToCombinationDramToDramAndSramMergeShare Merged.dot");
        SaveOpGraphToDot(combOpGraph, stream, DetailLevel::High);
    }

    REQUIRE(combOpGraph.GetBuffers().size() == 2);
    REQUIRE(combOpGraph.GetOps().size() == 1);
}

TEST_CASE("GluePartToCombinationDramToDramsMerge", "[CombinerDFS]")
{
    // Create graph:
    //
    //
    //   - - > B
    //  |
    //  A - -> C
    //
    //  A is DRAM NHWCB
    //  B is DRAM NHWCB, could be merged with A if it's not an output
    //  C is DRAM NHWCB  could be merged with A if it's not an output
    //
    //  Various combinations of B and C being output buffers are checked, to make sure that we don't merge
    //  buffers incorrectly.
    bool isBOutput = GENERATE(false, true);
    bool isCOutput = GENERATE(false, true);
    CAPTURE(isBOutput);
    CAPTURE(isCOutput);

    GraphOfParts graph;
    auto pA = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pC = std::make_unique<MockPart>(graph.GeneratePartId());

    BasePart& partA = *pA;
    BasePart& partB = *pB;
    BasePart& partC = *pC;

    PartId partAId = pA->GetPartId();
    PartId partBId = pB->GetPartId();
    PartId partCId = pC->GetPartId();

    graph.AddPart(std::move(pA));
    graph.AddPart(std::move(pB));
    graph.AddPart(std::move(pC));

    PartOutputSlot partAOutputSlot = { partA.GetPartId(), 0 };

    PartInputSlot partBInputSlot = { partB.GetPartId(), 0 };
    PartInputSlot partCInputSlot = { partC.GetPartId(), 0 };

    graph.AddConnection(partBInputSlot, { partAOutputSlot });
    graph.AddConnection(partCInputSlot, { partAOutputSlot });

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Plan planA;
    std::unique_ptr<DramBuffer> bufferAPtr = DramBuffer::Build()
                                                 .AddFormat(CascadingBufferFormat::NHWCB)
                                                 .AddTensorShape(TensorShape{ 1, 64, 64, 64 })
                                                 .AddSizeInBytes(4)
                                                 .AddBufferType(BufferType::Intermediate);
    planA.m_OpGraph.AddBuffer(std::move(bufferAPtr));

    planA.m_OutputMappings = { { planA.m_OpGraph.GetBuffers()[0], partAOutputSlot } };

    Plan planB;
    std::unique_ptr<DramBuffer> bufferBPtr =
        DramBuffer::Build()
            .AddFormat(CascadingBufferFormat::NHWCB)
            .AddTensorShape(TensorShape{ 1, 64, 64, 64 })
            .AddSizeInBytes(4)
            .AddBufferType(isBOutput ? BufferType::Output : BufferType::Intermediate);
    planB.m_OpGraph.AddBuffer(std::move(bufferBPtr));

    planB.m_InputMappings = { { planB.m_OpGraph.GetBuffers()[0], partBInputSlot } };

    Plan planC;
    std::unique_ptr<DramBuffer> bufferCPtr =
        DramBuffer::Build()
            .AddFormat(CascadingBufferFormat::NHWCB)
            .AddTensorShape(TensorShape{ 1, 64, 64, 64 })
            .AddSizeInBytes(4)
            .AddBufferType(isCOutput ? BufferType::Output : BufferType::Intermediate);
    planC.m_OpGraph.AddBuffer(std::move(bufferCPtr));

    planC.m_InputMappings = { { planC.m_OpGraph.GetBuffers()[0], partCInputSlot } };

    Combination combA(partA.GetPartId(), std::move(planA));
    Combination combB(partB.GetPartId(), std::move(planB));
    Combination combC(partC.GetPartId(), std::move(planC));

    // Merge the combinations
    Combination comb = combA + combB + combC;

    FrozenGraphOfParts frozenGraph = FrozenGraphOfParts(std::move(graph));
    CombinerTest combiner(frozenGraph, hwCaps, compOpt, estOpt, debuggingContext);

    Combination combGlued = combiner.GluePartToCombinationSrcToDests(partA, comb, 0);

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("GluePartToCombinationDramToDramsMerge " +
                             std::to_string(static_cast<uint32_t>(isBOutput)) +
                             std::to_string(static_cast<uint32_t>(isCOutput)) + ".dot");
        SaveCombinationToDot(combGlued, stream, DetailLevel::High);
    }

    // Ending glue of A contains copies to 0, 1 or 2 NHWCB DRAM buffers, depending on how many unique
    // outputs we need
    const uint32_t numOutputs = (isBOutput ? 1 : 0) + (isCOutput ? 1 : 0);
    auto elemA                = combGlued.GetElem(partAId);
    EndingGlue* endingGlueA   = elemA.m_EndingGlues.find(partAOutputSlot)->second.get();
    REQUIRE(endingGlueA->m_Graph.GetOps().size() == 2 * numOutputs);
    REQUIRE(endingGlueA->m_Graph.GetBuffers().size() == numOutputs);

    // Starting glue of B is either a Replacement or empty, depending on if could share or not
    auto elemB                  = combGlued.GetElem(partBId);
    StartingGlue* startingGlueB = elemB.m_StartingGlues.find(partBInputSlot)->second.get();
    if (isBOutput)
    {
        // Copy (empty)
        REQUIRE(startingGlueB->m_Graph.GetOps().size() == 0);
        REQUIRE(startingGlueB->m_Graph.GetBuffers().size() == 0);
        REQUIRE(startingGlueB->m_ExternalConnections.m_ReplacementBuffers.size() == 0);
        REQUIRE(startingGlueB->m_ExternalConnections.m_OpsToBuffers.size() == 1);
    }
    else
    {
        // Replacement
        REQUIRE(startingGlueB->m_Graph.GetOps().size() == 0);
        REQUIRE(startingGlueB->m_Graph.GetBuffers().size() == 0);
        REQUIRE(startingGlueB->m_ExternalConnections.m_ReplacementBuffers.size() == 1);
    }

    // Starting glue of C is either a Replacement or empty, depending on if could share or not
    auto elemC                  = combGlued.GetElem(partCId);
    StartingGlue* startingGlueC = elemC.m_StartingGlues.find(partCInputSlot)->second.get();
    if (isCOutput)
    {
        // Copy (empty)
        REQUIRE(startingGlueC->m_Graph.GetOps().size() == 0);
        REQUIRE(startingGlueC->m_Graph.GetBuffers().size() == 0);
        REQUIRE(startingGlueC->m_ExternalConnections.m_ReplacementBuffers.size() == 0);
        REQUIRE(startingGlueC->m_ExternalConnections.m_OpsToBuffers.size() == 1);
    }
    else
    {
        // Replacement
        REQUIRE(startingGlueC->m_Graph.GetOps().size() == 0);
        REQUIRE(startingGlueC->m_Graph.GetBuffers().size() == 0);
        REQUIRE(startingGlueC->m_ExternalConnections.m_ReplacementBuffers.size() == 1);
    }
}

TEST_CASE("GluePartToCombinationSramToDramsMerge", "[CombinerDFS]")
{
    // Create graph:
    //
    //
    //   - - > B
    //  |
    //  A - -> C
    //
    //  A is SRAM
    //  B is DRAM NHWCB, could be merged with C if neither is an output
    //  C is DRAM NHWCB, could be merged with B if neither is an output
    //
    //  Various combinations of B and C being output buffers are checked, to make sure that we don't merge
    //  buffers incorrectly.
    bool isBOutput = GENERATE(false, true);
    bool isCOutput = GENERATE(false, true);
    CAPTURE(isBOutput);
    CAPTURE(isCOutput);

    GraphOfParts graph;
    auto pA = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pC = std::make_unique<MockPart>(graph.GeneratePartId());

    BasePart& partA = *pA;
    BasePart& partB = *pB;
    BasePart& partC = *pC;

    PartId partAId = pA->GetPartId();
    PartId partBId = pB->GetPartId();
    PartId partCId = pC->GetPartId();

    graph.AddPart(std::move(pA));
    graph.AddPart(std::move(pB));
    graph.AddPart(std::move(pC));

    PartOutputSlot partAOutputSlot = { partA.GetPartId(), 0 };

    PartInputSlot partBInputSlot = { partB.GetPartId(), 0 };
    PartInputSlot partCInputSlot = { partC.GetPartId(), 0 };

    graph.AddConnection(partBInputSlot, { partAOutputSlot });
    graph.AddConnection(partCInputSlot, { partAOutputSlot });

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Plan planA;
    SramBuffer* bufferA    = planA.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    bufferA->m_Format      = CascadingBufferFormat::NHWCB;
    bufferA->m_TensorShape = TensorShape{ 1, 64, 64, 64 };
    bufferA->m_StripeShape = TensorShape{ 0, 0, 0, 0 };
    bufferA->m_Order       = TraversalOrder::Xyz;
    bufferA->m_SizeInBytes = 4;
    planA.m_OutputMappings = { { planA.m_OpGraph.GetBuffers()[0], partAOutputSlot } };

    Plan planB;
    std::unique_ptr<DramBuffer> bufferBPtr =
        DramBuffer::Build()
            .AddFormat(CascadingBufferFormat::NHWCB)
            .AddTensorShape(TensorShape{ 1, 64, 64, 64 })
            .AddSizeInBytes(4)
            .AddBufferType(isBOutput ? BufferType::Output : BufferType::Intermediate);
    planB.m_OpGraph.AddBuffer(std::move(bufferBPtr));

    planB.m_InputMappings = { { planB.m_OpGraph.GetBuffers()[0], partBInputSlot } };

    Plan planC;
    std::unique_ptr<DramBuffer> bufferCPtr =
        DramBuffer::Build()
            .AddFormat(CascadingBufferFormat::NHWCB)
            .AddTensorShape(TensorShape{ 1, 64, 64, 64 })
            .AddSizeInBytes(4)
            .AddBufferType(isCOutput ? BufferType::Output : BufferType::Intermediate);
    planC.m_OpGraph.AddBuffer(std::move(bufferCPtr));

    planC.m_InputMappings = { { planC.m_OpGraph.GetBuffers()[0], partCInputSlot } };

    Combination combA(partA.GetPartId(), std::move(planA));
    Combination combB(partB.GetPartId(), std::move(planB));
    Combination combC(partC.GetPartId(), std::move(planC));

    // Merge the combinations
    Combination comb = combA + combB + combC;

    FrozenGraphOfParts frozenGraph = FrozenGraphOfParts(std::move(graph));
    CombinerTest combiner(frozenGraph, hwCaps, compOpt, estOpt, debuggingContext);

    Combination combGlued = combiner.GluePartToCombinationSrcToDests(partA, comb, 0);

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("GluePartToCombinationSramToDramsMerge " +
                             std::to_string(static_cast<uint32_t>(isBOutput)) +
                             std::to_string(static_cast<uint32_t>(isCOutput)) + ".dot");
        SaveCombinationToDot(combGlued, stream, DetailLevel::High);
    }

    REQUIRE(combGlued.GetEndPartId() - combGlued.GetFirstPartId() == 3);

    // Ending glue of A contains copies to NHWCB DRAM, which can be shared if neither B nor C are outputs.
    auto elemA              = combGlued.GetElem(partAId);
    EndingGlue* endingGlueA = elemA.m_EndingGlues.find(partAOutputSlot)->second.get();
    REQUIRE(endingGlueA->m_Graph.GetOps().size() == ((!isBOutput && !isCOutput) ? 1 : 2));
    REQUIRE(endingGlueA->m_Graph.GetBuffers().size() == ((isBOutput && isCOutput) ? 0 : 1));
    if (!(isBOutput && isCOutput))
    {
        REQUIRE(endingGlueA->m_Graph.GetBuffers()[0]->m_Format == CascadingBufferFormat::NHWCB);
    }

    // Starting glue of B is either empty, or a Replacement, depending on if could share or not
    auto elemB                  = combGlued.GetElem(partBId);
    StartingGlue* startingGlueB = elemB.m_StartingGlues.find(partBInputSlot)->second.get();
    if (isBOutput)
    {
        // Empty
        REQUIRE(startingGlueB->m_Graph.GetOps().size() == 0);
        REQUIRE(startingGlueB->m_Graph.GetBuffers().size() == 0);
        REQUIRE(startingGlueB->m_ExternalConnections.m_ReplacementBuffers.size() == 0);
    }
    else
    {
        // Replacement
        REQUIRE(startingGlueB->m_Graph.GetOps().size() == 0);
        REQUIRE(startingGlueB->m_Graph.GetBuffers().size() == 0);
        REQUIRE(startingGlueB->m_ExternalConnections.m_ReplacementBuffers.size() == 1);
    }

    // Starting glue of C is either empty, or a Replacement, depending on if could share or not
    auto elemC                  = combGlued.GetElem(partCId);
    StartingGlue* startingGlueC = elemC.m_StartingGlues.find(partCInputSlot)->second.get();
    if (isCOutput)
    {
        // Empty
        REQUIRE(startingGlueC->m_Graph.GetOps().size() == 0);
        REQUIRE(startingGlueC->m_Graph.GetBuffers().size() == 0);
        REQUIRE(startingGlueC->m_ExternalConnections.m_ReplacementBuffers.size() == 0);
    }
    else
    {
        // Replacement
        REQUIRE(startingGlueC->m_Graph.GetOps().size() == 0);
        REQUIRE(startingGlueC->m_Graph.GetBuffers().size() == 0);
        REQUIRE(startingGlueC->m_ExternalConnections.m_ReplacementBuffers.size() == 1);
    }
}

TEST_CASE("GluePartToCombinationSramToDramConversion", "[CombinerDFS]")
{
    // Create graph:
    //
    //  A - -> B
    //
    //  A is SRAM
    //  B is DRAM NHWC, and cannot be copied directly from A because it splits in depth. It gets converted via another DRAM buffer.
    GraphOfParts graph;
    auto pA = std::make_unique<MockPart>(graph.GeneratePartId());
    auto pB = std::make_unique<MockPart>(graph.GeneratePartId());

    BasePart& partA = *pA;
    BasePart& partB = *pB;

    PartId partAId = pA->GetPartId();
    PartId partBId = pB->GetPartId();

    graph.AddPart(std::move(pA));
    graph.AddPart(std::move(pB));

    PartOutputSlot partAOutputSlot = { partA.GetPartId(), 0 };
    PartInputSlot partBInputSlot   = { partB.GetPartId(), 0 };

    graph.AddConnection(partBInputSlot, { partAOutputSlot });

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    Plan planA;
    SramBuffer* bufferA    = planA.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    bufferA->m_Format      = CascadingBufferFormat::NHWCB;
    bufferA->m_TensorShape = TensorShape{ 1, 64, 64, 64 };
    bufferA->m_StripeShape = TensorShape{ 1, 8, 8, 32 };
    bufferA->m_Order       = TraversalOrder::Xyz;
    bufferA->m_SizeInBytes = 4;
    planA.m_OutputMappings = { { planA.m_OpGraph.GetBuffers()[0], partAOutputSlot } };

    Plan planB;
    std::unique_ptr<DramBuffer> bufferBPtr = DramBuffer::Build()
                                                 .AddFormat(CascadingBufferFormat::NHWC)
                                                 .AddTensorShape(TensorShape{ 1, 64, 64, 64 })
                                                 .AddSizeInBytes(4)
                                                 .AddBufferType(BufferType::Intermediate);
    planB.m_OpGraph.AddBuffer(std::move(bufferBPtr));

    planB.m_InputMappings = { { planB.m_OpGraph.GetBuffers()[0], partBInputSlot } };

    Combination combA(partA.GetPartId(), std::move(planA));
    Combination combB(partB.GetPartId(), std::move(planB));

    // Merge the combinations
    Combination comb = combA + combB;

    FrozenGraphOfParts frozenGraph = FrozenGraphOfParts(std::move(graph));
    CombinerTest combiner(frozenGraph, hwCaps, compOpt, estOpt, debuggingContext);

    Combination combGlued = combiner.GluePartToCombinationSrcToDests(partA, comb, 0);

    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("GluePartToCombinationSramToDramConversion.dot");
        SaveCombinationToDot(combGlued, stream, DetailLevel::High);
    }

    // Ending glue of A converts to FCAF and then NHWC in DRAM.
    // Starting glue of B replaces that NHWC buffer

    REQUIRE(combGlued.GetEndPartId() - combGlued.GetFirstPartId() == 2);

    auto elemA              = combGlued.GetElem(partAId);
    EndingGlue* endingGlueA = elemA.m_EndingGlues.find(partAOutputSlot)->second.get();
    OpGraph& opGraphA       = elemA.m_Plan->m_OpGraph;
    REQUIRE(endingGlueA->m_Graph.GetOps().size() == 3);
    REQUIRE(endingGlueA->m_Graph.GetBuffers().size() == 3);
    REQUIRE(endingGlueA->m_Graph.GetBuffers()[0]->m_Format == CascadingBufferFormat::FCAF_DEEP);
    REQUIRE(endingGlueA->m_Graph.GetBuffers()[1]->m_Format == CascadingBufferFormat::NHWC);
    REQUIRE(endingGlueA->m_Graph.GetBuffers()[2]->m_Location == Location::Sram);
    REQUIRE(endingGlueA->m_ExternalConnections.m_BuffersToOps ==
            std::multimap<Buffer*, Op*>{ { opGraphA.GetBuffers()[0], endingGlueA->m_Graph.GetOp(0) } });

    auto elemB                  = combGlued.GetElem(partBId);
    StartingGlue* startingGlueB = elemB.m_StartingGlues.find(partBInputSlot)->second.get();
    OpGraph& opGraphB           = elemB.m_Plan->m_OpGraph;
    REQUIRE(startingGlueB->m_Graph.GetOps().size() == 0);
    REQUIRE(startingGlueB->m_Graph.GetBuffers().size() == 0);
    REQUIRE(startingGlueB->m_ExternalConnections.m_ReplacementBuffers ==
            std::unordered_map<Buffer*, Buffer*>{ { opGraphB.GetBuffers()[0], endingGlueA->m_Graph.GetBuffers()[1] } });
}

TEST_CASE("AllocateSram", "[CombinerDFS]")
{
    GraphOfParts graph;

    auto pA = std::make_unique<MockPart>(graph.GeneratePartId());

    const BasePart& partA = *pA;
    graph.AddPart(std::move(pA));

    PartOutputSlot partAOutputSlot = { partA.GetPartId(), 0 };
    PartInputSlot partAInputSlot   = { partA.GetPartId(), 0 };

    const uint32_t ifmSize = 524288;
    const uint32_t ofmSize = 65536;

    auto mockBuffer           = std::make_unique<SramBuffer>();
    mockBuffer->m_Format      = CascadingBufferFormat::NHWCB;
    mockBuffer->m_TensorShape = TensorShape{ 1, 32, 16, 1024 };
    mockBuffer->m_StripeShape = TensorShape{ 1, 32, 16, 1024 };
    mockBuffer->m_Order       = TraversalOrder::Xyz;
    mockBuffer->m_SizeInBytes = ifmSize;
    mockBuffer->m_Offset      = 0;

    Plan planA;
    SramBuffer* bufferA1    = planA.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    bufferA1->m_Format      = CascadingBufferFormat::NHWCB;
    bufferA1->m_TensorShape = TensorShape{ 1, 32, 16, 1024 };
    bufferA1->m_StripeShape = TensorShape{ 1, 32, 16, 1024 };
    bufferA1->m_Order       = TraversalOrder::Xyz;
    bufferA1->m_SizeInBytes = ifmSize;
    SramBuffer* bufferA2    = planA.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
    bufferA2->m_Format      = CascadingBufferFormat::NHWCB;
    bufferA2->m_TensorShape = TensorShape{ 1, 32, 16, 1024 };
    bufferA2->m_StripeShape = TensorShape{ 1, 4, 16, 1024 };
    bufferA2->m_Order       = TraversalOrder::Xyz;
    bufferA2->m_SizeInBytes = ofmSize;

    planA.m_OpGraph.AddOp(std::make_unique<MceOp>(
        MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct, BlockConfig{ 8u, 8u }, TensorShape{ 1, 32, 16, 1024 },
        TensorShape{ 1, 4, 16, 1024 }, TensorShape{ 1, 32, 16, 1024 }, TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));
    planA.m_OpGraph.SetProducer(bufferA2, planA.m_OpGraph.GetOps()[0]);
    planA.m_InputMappings  = { { bufferA1, partAInputSlot } };
    planA.m_OutputMappings = { { bufferA2, partAOutputSlot } };

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    ethosn::support_library::impl::NumMemoryStripes numMemoryStripes;
    const std::set<uint32_t> operationIds = { 0 };

    FrozenGraphOfParts frozenGraph = FrozenGraphOfParts(std::move(graph));
    CombinerTest combiner(frozenGraph, hwCaps, compOpt, estOpt, debuggingContext);

    SectionContext context = { {}, SramAllocator(hwCaps.GetTotalSramSize() / hwCaps.GetNumberOfSrams()),
                               {}, {},
                               0,  false,
                               {}, BlockConfig{ 8u, 8u } };

    // SRAM has enough space for ofm and the plan does not have a PLE kernel
    SectionContext context1 = context;
    REQUIRE(combiner.AllocateSram(context1, partA.GetPartId(), planA, { mockBuffer.get() }) == true);
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
    REQUIRE(combiner.AllocateSram(context2, partA.GetPartId(), planA, { mockBuffer.get() }) == true);
    REQUIRE(context2.pleOps.size() == 1);
    REQUIRE(actualPleOp->m_LoadKernel == true);

    // PLE kernel used previously has different block height
    // The plan is expected to be fit into SRAM and there is a need to Load the Kernel
    SectionContext context3 = context;
    PleKernelId pleKernel1  = PleKernelId::PASSTHROUGH_8X16_1;
    context3.pleOps         = { { pleKernel1, 0 } };
    REQUIRE(combiner.AllocateSram(context3, partA.GetPartId(), planA, { mockBuffer.get() }) == true);
    REQUIRE(context3.pleOps.size() == 2);
    REQUIRE(actualPleOp->m_LoadKernel == true);

    // PLE kernel passthrough is already used previously in the same
    // section, the plan is expected to be fit into SRAM and no need to Load the Kernel
    SectionContext context4 = context;
    PleKernelId pleKernel2  = PleKernelId::PASSTHROUGH_8X8_2;
    context4.pleOps         = { { pleKernel2, 0 } };
    REQUIRE(combiner.AllocateSram(context4, partA.GetPartId(), planA, { mockBuffer.get() }) == true);
    REQUIRE(context4.pleOps.size() == 1);
    REQUIRE(actualPleOp->m_LoadKernel == false);

    SectionContext context5 = context;
    // Allocate memory where the plan and the allocated memory exceeds the SRAM Size
    uint32_t planSize          = ofmSize + planA.m_OpGraph.GetBuffers()[2]->m_SizeInBytes + hwCaps.GetMaxPleSize();
    uint32_t remainingSramSize = hwCaps.GetTotalSramSize() - planSize;
    context5.alloc.Allocate((remainingSramSize + hwCaps.GetNumberOfSrams()) / hwCaps.GetNumberOfSrams(),
                            AllocationPreference::Start);
    REQUIRE(combiner.AllocateSram(context5, partA.GetPartId(), planA, { mockBuffer.get() }) == false);
    REQUIRE(context5.pleOps.size() == 0);
    REQUIRE(actualPleOp->m_LoadKernel == true);

    ETHOSN_UNUSED(outBufferAndPleOp);
}

TEST_CASE("SramAllocationForSinglePartSection", "[CombinerDFS]")
{
    GIVEN("A Graph of one part where its corresponding plan fits into a single section")
    {
        GraphOfParts graph;

        auto pA = std::make_unique<MockPart>(graph.GeneratePartId());

        const BasePart& partA = *pA;

        graph.AddPart(std::move(pA));

        PartInputSlot partAInputSlot   = { partA.GetPartId(), 0 };
        PartOutputSlot partAOutputSlot = { partA.GetPartId(), 0 };

        ethosn::support_library::impl::NumMemoryStripes numMemoryStripes;
        const std::set<uint32_t> operationIds = { 0 };

        const CompilationOptions compOpt;
        const EstimationOptions estOpt;
        const DebuggingContext debuggingContext(compOpt.m_DebugInfo);
        const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

        FrozenGraphOfParts frozenGraph = FrozenGraphOfParts(std::move(graph));
        CombinerTest combiner(frozenGraph, hwCaps, compOpt, estOpt, debuggingContext);
        SectionContext context     = { {}, SramAllocator(hwCaps.GetTotalSramSize() / hwCaps.GetNumberOfSrams()),
                                   {}, {},
                                   0,  false,
                                   {}, BlockConfig{ 16u, 16u } };
        uint32_t currentSramOffset = 0;

        Plan planA;

        const uint32_t inputBufferSize  = 512;
        const uint32_t outputBufferSize = 512;

        SramBuffer* bufferA          = planA.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
        bufferA->m_Format            = CascadingBufferFormat::NHWCB;
        bufferA->m_TensorShape       = TensorShape{ 1, 8, 8, 8 };
        bufferA->m_StripeShape       = TensorShape{ 1, 8, 8, 8 };
        bufferA->m_Order             = TraversalOrder::Xyz;
        bufferA->m_SizeInBytes       = inputBufferSize;
        PleInputSramBuffer* bufferA2 = planA.m_OpGraph.AddBuffer(std::make_unique<PleInputSramBuffer>());
        bufferA2->m_Format           = CascadingBufferFormat::NHWCB;
        bufferA2->m_TensorShape      = TensorShape{ 1, 8, 8, 8 };
        bufferA2->m_StripeShape      = TensorShape{ 1, 8, 8, 8 };
        bufferA2->m_SizeInBytes      = outputBufferSize;

        planA.m_OpGraph.AddOp(std::make_unique<MceOp>(
            MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct, BlockConfig{ 8u, 8u }, TensorShape{ 1, 8, 8, 8 },
            TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 8, 8, 8 }, TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));

        planA.m_InputMappings  = { { bufferA, partAInputSlot } };
        planA.m_OutputMappings = { { bufferA2, partAOutputSlot } };

        WHEN("Lonely section with a plan that has no Ple Op")
        {
            REQUIRE(bufferA->m_Offset.has_value() == false);
            REQUIRE(combiner.AllocateSram(context, partA.GetPartId(), planA, { nullptr }) == true);
            REQUIRE(bufferA->m_Offset.has_value() == true);
            REQUIRE(bufferA->m_Offset.value() == currentSramOffset);
            currentSramOffset = bufferA->m_SizeInBytes / hwCaps.GetNumberOfSrams();
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

            REQUIRE(bufferA->m_Offset.has_value() == false);
            REQUIRE(outBufferAndPleOp.first->m_Offset.has_value() == false);
            REQUIRE(combiner.AllocateSram(context, partA.GetPartId(), planA, { nullptr }) == true);

            REQUIRE(actualPleOp->m_Offset.has_value() == true);
            REQUIRE(actualPleOp->m_Offset.value() == currentSramOffset);

            currentSramOffset += hwCaps.GetMaxPleSize();
            REQUIRE(bufferA->m_Offset.has_value() == true);
            REQUIRE(bufferA->m_Offset.value() == currentSramOffset);
            currentSramOffset += bufferA->m_SizeInBytes / hwCaps.GetNumberOfSrams();
            // Note that Buffer 1 is the output from MceOp where its location is in PleInputSRAM not SRAM
            REQUIRE(outBufferAndPleOp.first->m_Offset.has_value() == true);
            REQUIRE(outBufferAndPleOp.first->m_Offset.value() == currentSramOffset);
            currentSramOffset += outBufferAndPleOp.first->m_SizeInBytes / hwCaps.GetNumberOfSrams();
            REQUIRE(context.pleOps.size() == 1);
        }
    }
}

TEST_CASE("SramAllocationForMultiplePartSection", "[CombinerDFS]")
{
    GIVEN("A Graph of three parts where their corresponding plans fits into a single section")
    {
        GraphOfParts graph;

        // Use even part IDs so that SRAM allocations all use AllocationPreference::Start
        auto pA = std::make_unique<MockPart>(0);
        auto pB = std::make_unique<MockPart>(2);
        auto pC = std::make_unique<MockPart>(4);

        const BasePart& partA = *pA;
        const BasePart& partB = *pB;
        const BasePart& partC = *pC;

        graph.AddPart(std::move(pA));
        graph.AddPart(std::move(pB));
        graph.AddPart(std::move(pC));

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

        FrozenGraphOfParts frozenGraph = FrozenGraphOfParts(std::move(graph));
        CombinerTest combiner(frozenGraph, hwCaps, compOpt, estOpt, debuggingContext);
        SectionContext context     = { {}, SramAllocator(hwCaps.GetTotalSramSize() / hwCaps.GetNumberOfSrams()),
                                   {}, {},
                                   0,  false,
                                   {}, BlockConfig{ 16u, 16u } };
        uint32_t currentSramOffset = 0;

        Plan planA;

        const uint32_t inputBufferSize  = 512;
        const uint32_t outputBufferSize = 512;

        SramBuffer* bufferA1         = planA.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
        bufferA1->m_Format           = CascadingBufferFormat::NHWCB;
        bufferA1->m_TensorShape      = TensorShape{ 1, 8, 8, 8 };
        bufferA1->m_StripeShape      = TensorShape{ 1, 8, 8, 8 };
        bufferA1->m_Order            = TraversalOrder::Xyz;
        bufferA1->m_SizeInBytes      = inputBufferSize;
        PleInputSramBuffer* bufferA2 = planA.m_OpGraph.AddBuffer(std::make_unique<PleInputSramBuffer>());
        bufferA2->m_Format           = CascadingBufferFormat::NHWCB;
        bufferA2->m_TensorShape      = TensorShape{ 1, 8, 8, 8 };
        bufferA2->m_StripeShape      = TensorShape{ 1, 8, 8, 8 };
        bufferA2->m_SizeInBytes      = outputBufferSize;

        planA.m_OpGraph.AddOp(std::make_unique<MceOp>(
            MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct, BlockConfig{ 8u, 8u }, TensorShape{ 1, 8, 8, 8 },
            TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 8, 8, 8 }, TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));

        planA.m_InputMappings  = { { bufferA1, partAInputSlot } };
        planA.m_OutputMappings = { { bufferA2, partAOutputSlot } };

        WHEN("Starting the section with the first plan that has no Ple Op")
        {
            REQUIRE(bufferA1->m_Offset.has_value() == false);
            REQUIRE(combiner.AllocateSram(context, partA.GetPartId(), planA, { nullptr }) == true);
            REQUIRE(bufferA1->m_Offset.has_value() == true);
            REQUIRE(bufferA1->m_Offset.value() == currentSramOffset);
            currentSramOffset = bufferA1->m_SizeInBytes / hwCaps.GetNumberOfSrams();
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

            REQUIRE(bufferA1->m_Offset.has_value() == false);
            REQUIRE(outBufferAndPleOp.first->m_Offset.has_value() == false);
            REQUIRE(combiner.AllocateSram(context, partA.GetPartId(), planA, { nullptr }) == true);

            REQUIRE(actualPleOp->m_Offset.has_value() == true);
            REQUIRE(actualPleOp->m_Offset.value() == currentSramOffset);

            currentSramOffset += hwCaps.GetMaxPleSize();
            REQUIRE(bufferA1->m_Offset.has_value() == true);
            REQUIRE(bufferA1->m_Offset.value() == currentSramOffset);
            currentSramOffset += bufferA1->m_SizeInBytes / hwCaps.GetNumberOfSrams();
            // Note that Buffer 1 is the output from MceOp where its location is in PleInputSRAM not SRAM
            REQUIRE(outBufferAndPleOp.first->m_Offset.has_value() == true);
            REQUIRE(outBufferAndPleOp.first->m_Offset.value() == currentSramOffset);
            currentSramOffset += outBufferAndPleOp.first->m_SizeInBytes / hwCaps.GetNumberOfSrams();
            REQUIRE(context.pleOps.size() == 1);

            Plan planB;

            const uint32_t InputBufferSize  = 512;
            const uint32_t OutputBufferSize = 512;

            SramBuffer* bufferB1    = planB.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
            bufferB1->m_Format      = CascadingBufferFormat::NHWCB;
            bufferB1->m_TensorShape = TensorShape{ 1, 8, 8, 8 };
            bufferB1->m_StripeShape = TensorShape{ 1, 8, 8, 8 };
            bufferB1->m_Order       = TraversalOrder::Xyz;
            bufferB1->m_SizeInBytes = InputBufferSize;

            PleInputSramBuffer* bufferB2 = planB.m_OpGraph.AddBuffer(std::make_unique<PleInputSramBuffer>());
            bufferB2->m_Format           = CascadingBufferFormat::NHWCB;
            bufferB2->m_TensorShape      = TensorShape{ 1, 8, 8, 8 };
            bufferB2->m_StripeShape      = TensorShape{ 1, 8, 8, 8 };
            bufferB2->m_SizeInBytes      = OutputBufferSize;

            planB.m_OpGraph.AddOp(std::make_unique<MceOp>(MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
                                                          BlockConfig{ 8u, 8u }, TensorShape{ 1, 8, 8, 8 },
                                                          TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 8, 8, 8 },
                                                          TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));

            planB.m_InputMappings  = { { bufferB1, partBInputSlot } };
            planB.m_OutputMappings = { { bufferB2, partBOutputSlot } };

            WHEN("Continuing the section with the second plan that has no Ple Op")
            {
                REQUIRE(bufferB1->m_Offset.has_value() == false);
                REQUIRE(combiner.AllocateSram(context, partB.GetPartId(), planB, { outBufferAndPleOp.first }) == true);
                REQUIRE(bufferB1->m_Offset.has_value() == true);
                REQUIRE(bufferB1->m_Offset.value() == outBufferAndPleOp.first->m_Offset.value());
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
                REQUIRE(combiner.AllocateSram(context, partB.GetPartId(), planB,
                                              { planA.m_OpGraph.GetBuffers()[2]->Sram() }) == true);
                REQUIRE(context.pleOps.size() == 1);

                REQUIRE(actualPleOpB->m_LoadKernel == false);
                REQUIRE(actualPleOpB->m_Offset == actualPleOpA->m_Offset);

                REQUIRE(outBufferAndPleOp.first->m_Offset.value() == currentSramOffset);
                currentSramOffset += outBufferAndPleOp.first->m_SizeInBytes / hwCaps.GetNumberOfSrams();
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

                REQUIRE(combiner.AllocateSram(context, partB.GetPartId(), planB,
                                              { planA.m_OpGraph.GetBuffers()[2]->Sram() }) == true);
                REQUIRE(context.pleOps.size() == 2);

                REQUIRE(actualPleOpB->m_LoadKernel == true);
                REQUIRE(actualPleOpB->m_Offset == currentSramOffset);

                currentSramOffset += hwCaps.GetMaxPleSize();
                REQUIRE(outBufferAndPleOp.first->m_Offset.value() == currentSramOffset);
                currentSramOffset += outBufferAndPleOp.first->m_SizeInBytes / hwCaps.GetNumberOfSrams();

                Plan planC;

                const uint32_t InputBufferSize  = 512;
                const uint32_t OutputBufferSize = 512;

                SramBuffer* bufferC1    = planC.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
                bufferC1->m_Format      = CascadingBufferFormat::NHWCB;
                bufferC1->m_TensorShape = TensorShape{ 1, 8, 8, 8 };
                bufferC1->m_StripeShape = TensorShape{ 1, 8, 8, 8 };
                bufferC1->m_Order       = TraversalOrder::Xyz;
                bufferC1->m_SizeInBytes = InputBufferSize;

                PleInputSramBuffer* bufferC2 = planC.m_OpGraph.AddBuffer(std::make_unique<PleInputSramBuffer>());
                bufferC2->m_Format           = CascadingBufferFormat::NHWCB;
                bufferC2->m_TensorShape      = TensorShape{ 1, 8, 8, 8 };
                bufferC2->m_StripeShape      = TensorShape{ 1, 8, 8, 8 };
                bufferC2->m_SizeInBytes      = OutputBufferSize;

                planC.m_OpGraph.AddOp(std::make_unique<MceOp>(MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
                                                              BlockConfig{ 8u, 8u }, TensorShape{ 1, 8, 8, 8 },
                                                              TensorShape{ 1, 8, 8, 8 }, TensorShape{ 1, 8, 8, 8 },
                                                              TraversalOrder::Xyz, Stride(), 0, 0, 0, 255));

                planC.m_InputMappings  = { { bufferC1, partCInputSlot } };
                planC.m_OutputMappings = { { bufferC2, partCOutputSlot } };

                WHEN("Ending the section with the third plan that has no Ple Op")
                {
                    REQUIRE(bufferC1->m_Offset.has_value() == false);
                    REQUIRE(combiner.AllocateSram(context, partC.GetPartId(), planC, { outBufferAndPleOp.first }) ==
                            true);
                    REQUIRE(bufferC1->m_Offset.has_value() == true);
                    REQUIRE(bufferC1->m_Offset.value() == outBufferAndPleOp.first->m_Offset.value());
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

                    REQUIRE(combiner.AllocateSram(context, partC.GetPartId(), planC,
                                                  { planB.m_OpGraph.GetBuffers()[2]->Sram() }) == true);
                    REQUIRE(context.pleOps.size() == 2);

                    REQUIRE(actualPleOpC->m_LoadKernel == false);

                    REQUIRE(actualPleOpC->m_Offset == actualPleOpB->m_Offset);

                    REQUIRE(outBufferAndPleOp.first->m_Offset.value() == currentSramOffset);
                    currentSramOffset += outBufferAndPleOp.first->m_SizeInBytes / hwCaps.GetNumberOfSrams();
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

                    REQUIRE(combiner.AllocateSram(context, partC.GetPartId(), planC,
                                                  { planB.m_OpGraph.GetBuffers()[2]->Sram() }) == true);
                    REQUIRE(context.pleOps.size() == 3);

                    REQUIRE(actualPleOpB->m_LoadKernel == true);

                    REQUIRE(actualPleOpC->m_Offset == currentSramOffset);

                    currentSramOffset += hwCaps.GetMaxPleSize();
                    REQUIRE(outBufferAndPleOp.first->m_Offset.value() == currentSramOffset);
                    currentSramOffset += outBufferAndPleOp.first->m_SizeInBytes / hwCaps.GetNumberOfSrams();
                }

                ETHOSN_UNUSED(outBufferAndPleOp);
            }

            ETHOSN_UNUSED(outBufferAndPleOp);
        }
    }
}
