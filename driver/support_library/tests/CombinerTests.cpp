//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/DebuggingContext.hpp"
#include "../src/GraphNodes.hpp"
#include "../src/cascading/Cascading.hpp"
#include "../src/cascading/Combiner.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>
#include <fstream>

using namespace ethosn::support_library;
namespace sl = ethosn::support_library;
using namespace ethosn::command_stream;

namespace
{

struct PlanConfigurator
{
    PlanConfigurator(Plan& plan, Node* node)
        : m_Plan(plan)
        , m_Node(node)
    {}

    virtual void SetMapping(Buffer* buffer) = 0;

    Plan& m_Plan;
    Node* m_Node{ nullptr };
};

struct InputPlanConfigurator : public PlanConfigurator
{
    InputPlanConfigurator(Plan& plan, Node* node)
        : PlanConfigurator(plan, node)
    {}

    void SetMapping(Buffer* buffer)
    {
        m_Plan.m_InputMappings[buffer] = m_Node->GetInput(0);
    }
};

struct OutputPlanConfigurator : public PlanConfigurator
{
    OutputPlanConfigurator(Plan& plan, Node* node)
        : PlanConfigurator(plan, node)
    {}

    void SetMapping(Buffer* buffer)
    {
        m_Plan.m_OutputMappings[buffer] = m_Node;
    }
};

void ConfigurePlan(PlanConfigurator&& configurator,
                   Lifetime lifetime,
                   Location location,
                   CascadingBufferFormat format,
                   const TensorShape& tensorShape,
                   const TensorShape& stripeShape,
                   TraversalOrder order,
                   uint32_t sizeInBytes,
                   const QuantizationInfo& quantization)
{
    Buffer tempBuffer(lifetime, location, format, tensorShape, stripeShape, order, sizeInBytes, quantization);
    Buffer* buffer = configurator.m_Plan.m_OpGraph.AddBuffer(std::make_unique<Buffer>(std::move(tempBuffer)));
    configurator.SetMapping(buffer);
}

void CheckCommonDRAMBuffer(const PlanCompatibilityResult& resultSramSram)
{
    REQUIRE(resultSramSram.m_IsCompatible == true);
    REQUIRE(resultSramSram.m_RequiresGlue == true);
    REQUIRE(resultSramSram.m_Glue.m_Graph.GetOps().size() == 2);
    REQUIRE(dynamic_cast<DmaOp*>(resultSramSram.m_Glue.m_Graph.GetOps()[0]) != nullptr);
    REQUIRE(dynamic_cast<DmaOp*>(resultSramSram.m_Glue.m_Graph.GetOps()[1]) != nullptr);
    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers().size() == 1);
    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_Location == Location::Dram);
    REQUIRE(resultSramSram.m_Glue.m_Graph.GetProducer(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]) ==
            resultSramSram.m_Glue.m_Graph.GetOps()[0]);
    REQUIRE(resultSramSram.m_Glue.m_Graph.GetConsumers(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]) ==
            std::vector<std::pair<Op*, uint32_t>>{ { resultSramSram.m_Glue.m_Graph.GetOps()[1], 0 } });
    REQUIRE(resultSramSram.m_Glue.m_InputSlot ==
            std::pair<Op*, uint32_t>{ resultSramSram.m_Glue.m_Graph.GetOps()[0], 0 });
    REQUIRE(resultSramSram.m_Glue.m_Output == resultSramSram.m_Glue.m_Graph.GetOps()[1]);
}

void CreateMceOpProducerWithBlockConfig(GraphOfParts& parts,
                                        Node* node,
                                        const BlockConfig& blockConfig,
                                        const EstimationOptions& estOpt,
                                        const CompilationOptions& compOpt,
                                        const HardwareCapabilities& hwCaps)
{
    parts.m_Parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    parts.m_Parts.back()->m_SubGraph.push_back(node);
    std::unique_ptr<Plan> plan = std::make_unique<Plan>();
    plan->m_OpGraph.AddBuffer(std::make_unique<Buffer>(
        Lifetime::Cascade, Location::PleInputSram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 16, 16, 16 },
        TensorShape{ 1, 16, 16, 16 }, TraversalOrder::Xyz, 0, QuantizationInfo()));
    plan->m_OutputMappings = { { plan->m_OpGraph.GetBuffers()[0], node } };
    MceOp op(Lifetime::Cascade, MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct, blockConfig,
             TensorShape{ 1, 16, 16, 16 }, TensorShape{ 1, 16, 16, 16 }, TensorShape{ 1, 1, 1, 16 },
             TraversalOrder::Xyz, Stride(), 0, 0);
    plan->m_OpGraph.AddOp(std::make_unique<MceOp>(std::move(op)));
    plan->m_OpGraph.SetProducer(plan->m_OpGraph.GetBuffers()[0], plan->m_OpGraph.GetOps()[0]);
    parts.m_Parts.back()->m_Plans.push_back(std::move(plan));
}

void CreatePleOpConsumerWithBlockConfig(GraphOfParts& parts,
                                        Node* node,
                                        std::vector<BlockConfig>& blockConfigs,
                                        const EstimationOptions& estOpt,
                                        const CompilationOptions& compOpt,
                                        const HardwareCapabilities& hwCaps)
{
    parts.m_Parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    parts.m_Parts.back()->m_SubGraph.push_back(node);
    std::unique_ptr<Plan> plan = std::make_unique<Plan>();
    plan->m_OpGraph.AddBuffer(std::make_unique<Buffer>(
        Lifetime::Cascade, Location::PleInputSram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 16, 16, 16 },
        TensorShape{ 1, 16, 16, 16 }, TraversalOrder::Xyz, 0, QuantizationInfo()));
    plan->m_InputMappings = { { plan->m_OpGraph.GetBuffers()[0], node->GetInput(0) } };
    for (auto& blockConfig : blockConfigs)
    {
        PleOp op(Lifetime::Cascade, PleOperation::PASSTHROUGH, blockConfig, 1U, { TensorShape{ 1, 16, 16, 16 } },
                 TensorShape{ 1, 16, 16, 16 });
        plan->m_OpGraph.AddOp(std::make_unique<PleOp>(std::move(op)));

        plan->m_OpGraph.AddConsumer(plan->m_OpGraph.GetBuffers()[0], plan->m_OpGraph.GetOps().back(), 0);
    }
    parts.m_Parts.back()->m_Plans.push_back(std::move(plan));
}

}    // namespace

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

    DotAttributes GetDotAttributes() override
    {
        return DotAttributes(std::to_string(m_Id), m_Name, "");
    }

    bool IsPrepared() override
    {
        return false;
    }

    std::string m_Name;
};

/// Checks that ArePlansCompatible correctly returns failure when given two unrelated plans and success
/// when given two adjacent plans that have compatible buffers (identical in this simple case)
TEST_CASE("ArePlansCompatible Simple")
{
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
    // Create simple graph A -> B -> C
    Graph graph;
    NameOnlyNode* nodeA = graph.CreateAndAddNode<NameOnlyNode>("a");
    NameOnlyNode* nodeB = graph.CreateAndAddNode<NameOnlyNode>("b");
    NameOnlyNode* nodeC = graph.CreateAndAddNode<NameOnlyNode>("c");
    graph.Connect(nodeA, nodeB, 0);
    graph.Connect(nodeB, nodeC, 0);

    // Generate a single plan for each node
    Buffer planAOutput(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(),
                       TraversalOrder::Xyz, 0, QuantizationInfo());
    Plan planA({}, { { &planAOutput, nodeA } });

    Buffer planBInput(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(),
                      TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer planBOutput(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(),
                       TraversalOrder::Xyz, 0, QuantizationInfo());
    Plan planB({ { &planBInput, nodeB->GetInput(0) } }, { { &planBOutput, nodeB } });

    Buffer planCInput(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(),
                      TraversalOrder::Xyz, 0, QuantizationInfo());
    Plan planC({ { &planCInput, nodeC->GetInput(0) } }, {});

    SECTION("Check compatibility for A -> B. These are adjacent so should be compatible.")
    {
        PlanCompatibilityResult resultAB = ArePlansCompatible(planA, planB, *nodeA->GetOutput(0), hwCaps);
        REQUIRE(resultAB.m_IsCompatible == true);
        REQUIRE(resultAB.m_RequiresGlue == false);
    }

    SECTION("Check compatibility for B -> C. These are adjacent so should be compatible.")
    {
        PlanCompatibilityResult resultBC = ArePlansCompatible(planB, planC, *nodeB->GetOutput(0), hwCaps);
        REQUIRE(resultBC.m_IsCompatible == true);
        REQUIRE(resultBC.m_RequiresGlue == false);
    }

    SECTION("Check compatibility for A -> C. These do not share an adjacent edge so should not be compatible.")
    {
        PlanCompatibilityResult resultAC = ArePlansCompatible(planA, planC, *nodeA->GetOutput(0), hwCaps);
        REQUIRE(resultAC.m_IsCompatible == false);
    }
}

/// Checks that ArePlansCompatible correctly returns success/failure when given adjacent buffers which
/// are compatible/incompatible.
TEST_CASE("ArePlansCompatible buffer compatibility")
{
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
    // Create simple graph A -> B
    Graph graph;
    NameOnlyNode* nodeA = graph.CreateAndAddNode<NameOnlyNode>("a");
    NameOnlyNode* nodeB = graph.CreateAndAddNode<NameOnlyNode>("b");
    graph.Connect(nodeA, nodeB, 0);

    // Generate a single plan for each node
    Buffer planAOutput(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 10, 10, 10 },
                       TensorShape(), TraversalOrder::Xyz, 0, QuantizationInfo());
    Plan planA({}, { { &planAOutput, nodeA } });

    Buffer planBInput(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 10, 10, 10 },
                      TensorShape(), TraversalOrder::Xyz, 0, QuantizationInfo());
    Plan planB({ { &planBInput, nodeB->GetInput(0) } }, {});

    SECTION("Modify the quant info on one of the buffers so they are different.")
    {
        planAOutput.m_QuantizationInfo = QuantizationInfo(100, 100.0f);
        // It is allowed to reinterpret the quant info of a buffer, so this should be successful
        PlanCompatibilityResult result = ArePlansCompatible(planA, planB, *nodeA->GetOutput(0), hwCaps);
        REQUIRE(result.m_IsCompatible == true);
        REQUIRE(result.m_RequiresGlue == false);
    }

    SECTION("Modify the shapes to be different (NHWCB)")
    {
        planBInput.m_TensorShape = TensorShape{ 1, 20, 10, 5 };
        // The buffers are NHWCB, so it is not allowed to reinterpret the shape like this
        PlanCompatibilityResult result = ArePlansCompatible(planA, planB, *nodeA->GetOutput(0), hwCaps);
        REQUIRE(result.m_IsCompatible == false);
    }

    SECTION("Modify the shapes to be different (NHWC and valid)")
    {
        planAOutput.m_Format     = CascadingBufferFormat::NHWC;
        planBInput.m_Format      = CascadingBufferFormat::NHWC;
        planBInput.m_TensorShape = TensorShape{ 1, 20, 10, 5 };
        // The buffers are NHWC, so it is allowed to reinterpret the shape like this
        PlanCompatibilityResult result = ArePlansCompatible(planA, planB, *nodeA->GetOutput(0), hwCaps);
        REQUIRE(result.m_IsCompatible == true);
        REQUIRE(result.m_RequiresGlue == false);
    }

    SECTION("Modify the shapes to be different (NHWC and invalid)")
    {
        planAOutput.m_Format     = CascadingBufferFormat::NHWC;
        planBInput.m_Format      = CascadingBufferFormat::NHWC;
        planBInput.m_TensorShape = TensorShape{ 1, 100, 100, 100 };
        // The buffers are NHWC, but the modified tensor shape is not compatible (same number of elements) so it is
        // not allowed to reinterpret the shape like this
        PlanCompatibilityResult result = ArePlansCompatible(planA, planB, *nodeA->GetOutput(0), hwCaps);
        REQUIRE(result.m_IsCompatible == false);
    }
}

/// Checks that ArePlansCompatible correctly returns glue when DMA ops are required.
TEST_CASE("ArePlansCompatible Glue")
{
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
    // Create simple graph A -> B
    Graph graph;
    NameOnlyNode* nodeA = graph.CreateAndAddNode<NameOnlyNode>("a");
    NameOnlyNode* nodeB = graph.CreateAndAddNode<NameOnlyNode>("b");
    graph.Connect(nodeA, nodeB, 0);

    // Generate some plans for each node
    Buffer planAOutputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 2, 3, 4 },
                           TensorShape{ 1, 1, 1, 1 }, TraversalOrder::Xyz, 0, QuantizationInfo());
    Plan planASram({}, { { &planAOutputSram, nodeA } });

    Buffer planAOutputDram(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 2, 3, 4 },
                           TensorShape(), TraversalOrder::Xyz, 0, QuantizationInfo());
    Plan planADram({}, { { &planAOutputDram, nodeA } });

    Buffer planBInputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 2, 3, 4 },
                          TensorShape{ 1, 1, 1, 2 },    // Note different stripe shape to above, to make incompatible
                          TraversalOrder::Xyz, 0, QuantizationInfo());
    Plan planBSram({ { &planBInputSram, nodeB->GetInput(0) } }, {});

    Buffer planBInputDram(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 2, 3, 4 },
                          TensorShape(), TraversalOrder::Xyz, 0, QuantizationInfo());
    Plan planBDram({ { &planBInputDram, nodeB->GetInput(0) } }, {});

    SECTION("Check compatibility for A Sram -> B Dram. This requires a DMA op to be compatible.")
    {
        PlanCompatibilityResult resultSramDram = ArePlansCompatible(planASram, planBDram, *nodeA->GetOutput(0), hwCaps);
        REQUIRE(resultSramDram.m_IsCompatible == true);
        REQUIRE(resultSramDram.m_RequiresGlue == true);
        REQUIRE(resultSramDram.m_Glue.m_Graph.GetOps().size() == 1);
        REQUIRE(dynamic_cast<DmaOp*>(resultSramDram.m_Glue.m_Graph.GetOps()[0]) != nullptr);
        REQUIRE(resultSramDram.m_Glue.m_Graph.GetBuffers().size() == 0);
        REQUIRE(resultSramDram.m_Glue.m_InputSlot ==
                std::pair<Op*, uint32_t>{ resultSramDram.m_Glue.m_Graph.GetOps()[0], 0 });
        REQUIRE(resultSramDram.m_Glue.m_Output == resultSramDram.m_Glue.m_Graph.GetOps()[0]);
    }

    SECTION("Check compatibility for A Dram -> B Sram. This requires a DMA op to be compatible.")
    {
        PlanCompatibilityResult resultDramSram = ArePlansCompatible(planADram, planBSram, *nodeA->GetOutput(0), hwCaps);
        REQUIRE(resultDramSram.m_IsCompatible == true);
        REQUIRE(resultDramSram.m_RequiresGlue == true);
        REQUIRE(resultDramSram.m_Glue.m_Graph.GetOps().size() == 1);
        REQUIRE(dynamic_cast<DmaOp*>(resultDramSram.m_Glue.m_Graph.GetOps()[0]) != nullptr);
        REQUIRE(resultDramSram.m_Glue.m_Graph.GetBuffers().size() == 0);
        REQUIRE(resultDramSram.m_Glue.m_InputSlot ==
                std::pair<Op*, uint32_t>{ resultDramSram.m_Glue.m_Graph.GetOps()[0], 0 });
        REQUIRE(resultDramSram.m_Glue.m_Output == resultDramSram.m_Glue.m_Graph.GetOps()[0]);
    }

    SECTION("Check compatibility for A Sram -> B Sram without activation compression. This requires two DMA ops as the "
            "Sram buffers are incompatible, so we need to go out to Dram and back.")
    {
        PlanCompatibilityResult resultSramSram = ArePlansCompatible(planASram, planBSram, *nodeA->GetOutput(0), hwCaps);
        CheckCommonDRAMBuffer(resultSramSram);
        REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_TensorShape == TensorShape{ 1, 2, 3, 4 });
        REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_SizeInBytes == 1 * 8 * 8 * 16);
        REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_Format == CascadingBufferFormat::NHWCB);
    }
}

TEST_CASE("ArePlansCompatible Glue with incompatible activation compression")
{
    GIVEN("A simple graph A -> B")
    {
        Graph graph;
        NameOnlyNode* nodeA = graph.CreateAndAddNode<NameOnlyNode>("a");
        NameOnlyNode* nodeB = graph.CreateAndAddNode<NameOnlyNode>("b");
        graph.Connect(nodeA, nodeB, 0);
        WHEN("SRAM Buffer A is NOT compressible and SRAM buffer B is compressible")
        {
            Buffer planAOutputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                   TensorShape{ 1, 8, 8, 32 }, TensorShape{ 1, 1, 1, 1 }, TraversalOrder::Xyz, 0,
                                   QuantizationInfo());
            Plan planASram({}, { { &planAOutputSram, nodeA } });

            Buffer planBInputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                  TensorShape{ 1, 8, 8, 32 }, TensorShape{ 1, 8, 8, 32 }, TraversalOrder::Xyz, 0,
                                  QuantizationInfo());
            Plan planBSram({ { &planBInputSram, nodeB->GetInput(0) } }, {});
            constexpr TensorShape expectedTensorShape            = TensorShape{ 1, 8, 8, 32 };
            constexpr uint32_t expectedSizeInByte                = 1 * 8 * 8 * 32;
            constexpr CascadingBufferFormat expectedBufferFormat = CascadingBufferFormat::NHWCB;

            AND_WHEN("Hardware configuration is Nx7")
            {
                HardwareCapabilities hwCaps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
                THEN("DRAM buffer is CascadingBufferFormat::NHWCB (not compressed)")
                {
                    PlanCompatibilityResult resultSramSram =
                        ArePlansCompatible(planASram, planBSram, *nodeA->GetOutput(0), hwCaps);
                    CheckCommonDRAMBuffer(resultSramSram);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_TensorShape == expectedTensorShape);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_SizeInBytes == expectedSizeInByte);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_Format == expectedBufferFormat);
                }
            }
            AND_WHEN("Hardware configuration is N78")
            {
                HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();
                THEN("DRAM buffer is CascadingBufferFormat::NHWCB (not compressed)")
                {
                    PlanCompatibilityResult resultSramSram =
                        ArePlansCompatible(planASram, planBSram, *nodeA->GetOutput(0), hwCaps);
                    CheckCommonDRAMBuffer(resultSramSram);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_TensorShape == expectedTensorShape);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_SizeInBytes == expectedSizeInByte);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_Format == expectedBufferFormat);
                }
            }
        }
        WHEN("SRAM Buffer A is compressible and SRAM buffer B is NOT compressible")
        {
            Buffer planAOutputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                   TensorShape{ 1, 8, 8, 32 }, TensorShape{ 1, 8, 8, 32 }, TraversalOrder::Xyz, 0,
                                   QuantizationInfo());
            Plan planASram({}, { { &planAOutputSram, nodeA } });

            Buffer planBInputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                  TensorShape{ 1, 8, 8, 32 }, TensorShape{ 1, 1, 1, 1 }, TraversalOrder::Xyz, 0,
                                  QuantizationInfo());
            Plan planBSram({ { &planBInputSram, nodeB->GetInput(0) } }, {});
            constexpr TensorShape expectedTensorShape            = TensorShape{ 1, 8, 8, 32 };
            constexpr uint32_t expectedSizeInByte                = 1 * 8 * 8 * 32;
            constexpr CascadingBufferFormat expectedBufferFormat = CascadingBufferFormat::NHWCB;

            AND_WHEN("Hardware configuration is Nx7")
            {
                HardwareCapabilities hwCaps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
                THEN("DRAM buffer is CascadingBufferFormat::NHWCB (not compressed)")
                {
                    PlanCompatibilityResult resultSramSram =
                        ArePlansCompatible(planASram, planBSram, *nodeA->GetOutput(0), hwCaps);
                    CheckCommonDRAMBuffer(resultSramSram);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_TensorShape == expectedTensorShape);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_SizeInBytes == expectedSizeInByte);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_Format == expectedBufferFormat);
                }
            }
            AND_WHEN("Hardware configuration is N78")
            {
                HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();
                THEN("DRAM buffer is CascadingBufferFormat::NHWCB (not compressed)")
                {
                    PlanCompatibilityResult resultSramSram =
                        ArePlansCompatible(planASram, planBSram, *nodeA->GetOutput(0), hwCaps);
                    CheckCommonDRAMBuffer(resultSramSram);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_TensorShape == expectedTensorShape);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_SizeInBytes == expectedSizeInByte);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_Format == expectedBufferFormat);
                }
            }
        }
        WHEN("SRAM Buffer A is compressible with FCAF_WIDE only compression and SRAM buffer B is compressible with "
             "FCAF_DEEP only compression only")
        {
            Buffer planAOutputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                   TensorShape{ 1, 16, 16, 64 }, TensorShape{ 1, 8, 16, 48 }, TraversalOrder::Xyz, 0,
                                   QuantizationInfo());
            Plan planASram({}, { { &planAOutputSram, nodeA } });
            Buffer planBInputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                  TensorShape{ 1, 16, 16, 64 }, TensorShape{ 1, 8, 8, 64 }, TraversalOrder::Xyz, 0,
                                  QuantizationInfo());
            Plan planBSram({ { &planBInputSram, nodeB->GetInput(0) } }, {});
            constexpr TensorShape expectedTensorShape            = TensorShape{ 1, 16, 16, 64 };
            constexpr uint32_t expectedSizeInByte                = 1 * 16 * 16 * 64;
            constexpr CascadingBufferFormat expectedBufferFormat = CascadingBufferFormat::NHWCB;
            AND_WHEN("Hardware configuration is N78")
            {
                HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();
                THEN("DRAM buffer is CascadingBufferFormat::NHWCB (not compressed)")
                {
                    PlanCompatibilityResult resultSramSram =
                        ArePlansCompatible(planASram, planBSram, *nodeA->GetOutput(0), hwCaps);
                    CheckCommonDRAMBuffer(resultSramSram);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_TensorShape == expectedTensorShape);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_SizeInBytes == expectedSizeInByte);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_Format == expectedBufferFormat);
                }
            }
        }
    }
}

TEST_CASE("ArePlansCompatible Glue with compatible activation compression")
{
    GIVEN("A simple graph A -> B")
    {
        Graph graph;
        NameOnlyNode* nodeA = graph.CreateAndAddNode<NameOnlyNode>("a");
        NameOnlyNode* nodeB = graph.CreateAndAddNode<NameOnlyNode>("b");
        graph.Connect(nodeA, nodeB, 0);
        WHEN("Hardware configuration is N78")
        {
            HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();
            AND_WHEN("SRAM Buffer A is compressible with FCAF_WIDE only compression and SRAM buffer B is compressible "
                     "with FCAF_WIDE only compression only")
            {
                Buffer planAOutputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                       TensorShape{ 1, 16, 16, 48 }, TensorShape{ 1, 8, 16, 48 }, TraversalOrder::Xyz,
                                       0, QuantizationInfo());
                Plan planASram({}, { { &planAOutputSram, nodeA } });

                Buffer planBInputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                      TensorShape{ 1, 16, 16, 48 }, TensorShape{ 1, 16, 16, 48 }, TraversalOrder::Xyz,
                                      0, QuantizationInfo());
                Plan planBSram({ { &planBInputSram, nodeB->GetInput(0) } }, {});
                constexpr TensorShape expectedTensorShape            = TensorShape{ 1, 16, 16, 48 };
                constexpr uint32_t expectedSizeInByte                = 1 * 16 * 16 * 48;
                constexpr CascadingBufferFormat expectedBufferFormat = CascadingBufferFormat::FCAF_WIDE;
                THEN("DRAM buffer is CascadingBufferFormat::FCAF_WIDE (compressed)")
                {
                    PlanCompatibilityResult resultSramSram =
                        ArePlansCompatible(planASram, planBSram, *nodeA->GetOutput(0), hwCaps);
                    CheckCommonDRAMBuffer(resultSramSram);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_TensorShape == expectedTensorShape);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_SizeInBytes == expectedSizeInByte);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_Format == expectedBufferFormat);
                }
            }
            AND_WHEN("SRAM Buffer A is compressible with FCAF_WIDE only compression and SRAM buffer B is compressible "
                     "with both FCAF compression")
            {
                Buffer planAOutputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                       TensorShape{ 1, 16, 16, 48 }, TensorShape{ 1, 8, 16, 48 }, TraversalOrder::Xyz,
                                       0, QuantizationInfo());
                Plan planASram({}, { { &planAOutputSram, nodeA } });

                Buffer planBInputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                      TensorShape{ 1, 16, 16, 48 }, TensorShape{ 1, 16, 16, 32 }, TraversalOrder::Xyz,
                                      0, QuantizationInfo());
                Plan planBSram({ { &planBInputSram, nodeB->GetInput(0) } }, {});
                constexpr TensorShape expectedTensorShape            = TensorShape{ 1, 16, 16, 48 };
                constexpr uint32_t expectedSizeInByte                = 1 * 16 * 16 * 48;
                constexpr CascadingBufferFormat expectedBufferFormat = CascadingBufferFormat::FCAF_WIDE;
                THEN("DRAM buffer is CascadingBufferFormat::FCAF_WIDE (compressed)")
                {
                    PlanCompatibilityResult resultSramSram =
                        ArePlansCompatible(planASram, planBSram, *nodeA->GetOutput(0), hwCaps);
                    CheckCommonDRAMBuffer(resultSramSram);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_TensorShape == expectedTensorShape);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_SizeInBytes == expectedSizeInByte);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_Format == expectedBufferFormat);
                }
            }
            AND_WHEN("SRAM Buffer A is compressible with both FCAF compression and SRAM buffer B is compressible with "
                     "FCAF_WIDE only")
            {
                Buffer planAOutputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                       TensorShape{ 1, 16, 16, 48 }, TensorShape{ 1, 8, 16, 32 }, TraversalOrder::Xyz,
                                       0, QuantizationInfo());
                Plan planASram({}, { { &planAOutputSram, nodeA } });

                Buffer planBInputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                      TensorShape{ 1, 16, 16, 48 }, TensorShape{ 1, 16, 16, 48 }, TraversalOrder::Xyz,
                                      0, QuantizationInfo());
                Plan planBSram({ { &planBInputSram, nodeB->GetInput(0) } }, {});
                constexpr TensorShape expectedTensorShape            = TensorShape{ 1, 16, 16, 48 };
                constexpr uint32_t expectedSizeInByte                = 1 * 16 * 16 * 48;
                constexpr CascadingBufferFormat expectedBufferFormat = CascadingBufferFormat::FCAF_WIDE;
                THEN("DRAM buffer is CascadingBufferFormat::FCAF_WIDE (compressed)")
                {
                    PlanCompatibilityResult resultSramSram =
                        ArePlansCompatible(planASram, planBSram, *nodeA->GetOutput(0), hwCaps);
                    CheckCommonDRAMBuffer(resultSramSram);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_TensorShape == expectedTensorShape);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_SizeInBytes == expectedSizeInByte);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_Format == expectedBufferFormat);
                }
            }
            AND_WHEN("SRAM Buffer A is compressible with FCAF_DEEP only compression and SRAM buffer B is compressible "
                     "with FCAF_DEEP only compression only")
            {
                Buffer planAOutputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                       TensorShape{ 1, 8, 8, 64 }, TensorShape{ 1, 8, 8, 32 }, TraversalOrder::Xyz, 0,
                                       QuantizationInfo());
                Plan planASram({}, { { &planAOutputSram, nodeA } });

                Buffer planBInputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                      TensorShape{ 1, 8, 8, 64 }, TensorShape{ 1, 8, 8, 64 }, TraversalOrder::Xyz, 0,
                                      QuantizationInfo());
                Plan planBSram({ { &planBInputSram, nodeB->GetInput(0) } }, {});
                constexpr TensorShape expectedTensorShape            = TensorShape{ 1, 8, 8, 64 };
                constexpr uint32_t expectedSizeInByte                = 1 * 8 * 8 * 64;
                constexpr CascadingBufferFormat expectedBufferFormat = CascadingBufferFormat::FCAF_DEEP;
                THEN("DRAM buffer is CascadingBufferFormat::FCAF_DEEP (compressed)")
                {
                    PlanCompatibilityResult resultSramSram =
                        ArePlansCompatible(planASram, planBSram, *nodeA->GetOutput(0), hwCaps);
                    CheckCommonDRAMBuffer(resultSramSram);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_TensorShape == expectedTensorShape);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_SizeInBytes == expectedSizeInByte);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_Format == expectedBufferFormat);
                }
            }
            AND_WHEN("SRAM Buffer A is compressible with FCAF_DEEP only compression and SRAM buffer B is compressible "
                     "with both FCAF compression")
            {
                Buffer planAOutputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                       TensorShape{ 1, 16, 16, 48 }, TensorShape{ 1, 8, 8, 32 }, TraversalOrder::Xyz, 0,
                                       QuantizationInfo());
                Plan planASram({}, { { &planAOutputSram, nodeA } });

                Buffer planBInputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                      TensorShape{ 1, 16, 16, 48 }, TensorShape{ 1, 8, 16, 32 }, TraversalOrder::Xyz, 0,
                                      QuantizationInfo());
                Plan planBSram({ { &planBInputSram, nodeB->GetInput(0) } }, {});
                constexpr TensorShape expectedTensorShape            = TensorShape{ 1, 16, 16, 48 };
                constexpr uint32_t expectedSizeInByte                = 1 * 16 * 16 * 48;
                constexpr CascadingBufferFormat expectedBufferFormat = CascadingBufferFormat::FCAF_DEEP;
                THEN("DRAM buffer is CascadingBufferFormat::FCAF_DEEP (compressed)")
                {
                    PlanCompatibilityResult resultSramSram =
                        ArePlansCompatible(planASram, planBSram, *nodeA->GetOutput(0), hwCaps);
                    CheckCommonDRAMBuffer(resultSramSram);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_TensorShape == expectedTensorShape);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_SizeInBytes == expectedSizeInByte);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_Format == expectedBufferFormat);
                }
            }
            AND_WHEN("SRAM Buffer A is compressible with both FCAF compression and SRAM buffer B is compressible with "
                     "FCAF_DEEP only")
            {
                Buffer planAOutputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                       TensorShape{ 1, 16, 16, 48 }, TensorShape{ 1, 8, 16, 32 }, TraversalOrder::Xyz,
                                       0, QuantizationInfo());
                Plan planASram({}, { { &planAOutputSram, nodeA } });

                Buffer planBInputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                      TensorShape{ 1, 16, 16, 48 }, TensorShape{ 1, 8, 8, 32 }, TraversalOrder::Xyz, 0,
                                      QuantizationInfo());
                Plan planBSram({ { &planBInputSram, nodeB->GetInput(0) } }, {});
                constexpr TensorShape expectedTensorShape            = TensorShape{ 1, 16, 16, 48 };
                constexpr uint32_t expectedSizeInByte                = 1 * 16 * 16 * 48;
                constexpr CascadingBufferFormat expectedBufferFormat = CascadingBufferFormat::FCAF_DEEP;
                THEN("DRAM buffer is CascadingBufferFormat::FCAF_DEEP (compressed)")
                {
                    PlanCompatibilityResult resultSramSram =
                        ArePlansCompatible(planASram, planBSram, *nodeA->GetOutput(0), hwCaps);
                    CheckCommonDRAMBuffer(resultSramSram);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_TensorShape == expectedTensorShape);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_SizeInBytes == expectedSizeInByte);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_Format == expectedBufferFormat);
                }
            }
            AND_WHEN("SRAM Buffer A is compressible with both FCAF compression and SRAM buffer B is compressible with "
                     "both FCAF compression")
            {
                Buffer planAOutputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                       TensorShape{ 1, 16, 16, 32 }, TensorShape{ 1, 8, 16, 32 }, TraversalOrder::Xyz,
                                       0, QuantizationInfo());
                Plan planASram({}, { { &planAOutputSram, nodeA } });

                Buffer planBInputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                      TensorShape{ 1, 16, 16, 32 }, TensorShape{ 1, 16, 16, 32 }, TraversalOrder::Xyz,
                                      0, QuantizationInfo());
                Plan planBSram({ { &planBInputSram, nodeB->GetInput(0) } }, {});
                constexpr TensorShape expectedTensorShape            = TensorShape{ 1, 16, 16, 32 };
                constexpr uint32_t expectedSizeInByte                = 1 * 16 * 16 * 32;
                constexpr CascadingBufferFormat expectedBufferFormat = CascadingBufferFormat::FCAF_DEEP;
                THEN("DRAM buffer is CascadingBufferFormat::FCAF_DEEP (compressed)")
                {
                    PlanCompatibilityResult resultSramSram =
                        ArePlansCompatible(planASram, planBSram, *nodeA->GetOutput(0), hwCaps);
                    CheckCommonDRAMBuffer(resultSramSram);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_TensorShape == expectedTensorShape);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_SizeInBytes == expectedSizeInByte);
                    REQUIRE(resultSramSram.m_Glue.m_Graph.GetBuffers()[0]->m_Format == expectedBufferFormat);
                }
            }
        }
    }
}

TEST_CASE("ArePlansCompatible matching BlockConfigs")
{
    Graph graph;
    NameOnlyNode* nodeA = graph.CreateAndAddNode<NameOnlyNode>("a");
    NameOnlyNode* nodeB = graph.CreateAndAddNode<NameOnlyNode>("b");

    graph.Connect(nodeA, nodeB, 0);

    GraphOfParts parts;

    const EstimationOptions estOpt;
    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    // Part consisting of node A
    CreateMceOpProducerWithBlockConfig(parts, nodeA, BlockConfig{ 16U, 16U }, estOpt, compOpt, hwCaps);
    Plan& planAToCheck = *(parts.m_Parts.back()->m_Plans.back());

    // Part consisting of node B
    std::vector<BlockConfig> configs = { BlockConfig{ 16U, 16U }, BlockConfig{ 16U, 16U } };
    CreatePleOpConsumerWithBlockConfig(parts, nodeB, configs, estOpt, compOpt, hwCaps);
    Plan& planBToCheck = *(parts.m_Parts.back()->m_Plans.back());

    SECTION("Check compatibility for A -> B")
    {
        PlanCompatibilityResult resultAB = ArePlansCompatible(planAToCheck, planBToCheck, *nodeA->GetOutput(0), hwCaps);
        REQUIRE(resultAB.m_IsCompatible == true);
        REQUIRE(resultAB.m_RequiresGlue == false);
    }
}

TEST_CASE("ArePlansCompatible non-matching BlockConfigs")
{
    Graph graph;
    NameOnlyNode* nodeA = graph.CreateAndAddNode<NameOnlyNode>("a");
    NameOnlyNode* nodeB = graph.CreateAndAddNode<NameOnlyNode>("b");

    graph.Connect(nodeA, nodeB, 0);

    GraphOfParts parts;

    const EstimationOptions estOpt;
    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    // Part consisting of node A
    CreateMceOpProducerWithBlockConfig(parts, nodeA, BlockConfig{ 16U, 16U }, estOpt, compOpt, hwCaps);
    Plan& planAToCheck = *(parts.m_Parts.back()->m_Plans.back());

    // Part consisting of node B
    std::vector<BlockConfig> configs = { BlockConfig{ 16U, 16U }, BlockConfig{ 16U, 8U } };
    CreatePleOpConsumerWithBlockConfig(parts, nodeB, configs, estOpt, compOpt, hwCaps);
    Plan& planBToCheck = *(parts.m_Parts.back()->m_Plans.back());

    SECTION("Check incompatibility for A -> B")
    {
        PlanCompatibilityResult resultAB = ArePlansCompatible(planAToCheck, planBToCheck, *nodeA->GetOutput(0), hwCaps);
        REQUIRE(resultAB.m_IsCompatible == false);
    }
}

TEST_CASE("CreateMetadata For Cascade With No Depthwise Splitting for Convolution")
{
    const EstimationOptions estOpt;
    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);

    // Create graph A -> B
    Graph graph;

    NameOnlyNode* nodeA = graph.CreateAndAddNode<NameOnlyNode>("a");

    MceOperationNode* nodeB = graph.CreateAndAddNode<MceOperationNode>(
        TensorShape(), TensorShape(), sl::DataType::UINT8_QUANTIZED, QuantizationInfo(),
        ethosn::support_library::TensorInfo({ 1, 1, 1, 1 }, ethosn::support_library::DataType::UINT8_QUANTIZED,
                                            ethosn::support_library::DataFormat::HWIO, QuantizationInfo(0, 0.9f)),
        std::vector<uint8_t>({ 1 }), ethosn::support_library::TensorInfo({ 1, 1, 1, 1 }), std::vector<int32_t>{ 0 },
        Stride(), 0, 0, ethosn::command_stream::MceOperation::CONVOLUTION, CompilerDataFormat::NHWCB,
        std::set<uint32_t>{ 1 });

    graph.Connect(nodeA, nodeB, 0);

    // Generate some plans for each node
    Buffer planAOutputSramFullDepth(Lifetime::Cascade, Location::Sram, CascadingBufferFormat::NHWCB,
                                    TensorShape{ 1, 112, 112, 64 }, TensorShape{ 1, 8, 8, 64 }, TraversalOrder::Xyz, 0,
                                    QuantizationInfo());
    Plan planASramFullDepth({}, { { &planAOutputSramFullDepth, nodeA } });

    Buffer planAOutputSramPartialDepth(Lifetime::Cascade, Location::Sram, CascadingBufferFormat::NHWCB,
                                       TensorShape{ 1, 112, 112, 64 }, TensorShape{ 1, 8, 8, 16 }, TraversalOrder::Xyz,
                                       0, QuantizationInfo());
    Plan planASramPartialDepth({}, { { &planAOutputSramPartialDepth, nodeA } });

    Buffer planAOutputDram(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB,
                           TensorShape{ 1, 112, 112, 64 }, TensorShape{ 1, 112, 112, 64 }, TraversalOrder::Xyz, 0,
                           QuantizationInfo());
    Plan planADram({}, { { &planAOutputDram, nodeA } });

    Buffer planBInputSramPartialDepth(Lifetime::Cascade, Location::Sram, CascadingBufferFormat::NHWCB,
                                      TensorShape{ 1, 112, 112, 64 }, TensorShape{ 1, 8, 8, 16 }, TraversalOrder::Xyz,
                                      0, QuantizationInfo());

    Plan planBSramPartialDepth({ { &planBInputSramPartialDepth, nodeB->GetInput(0) } }, { {} });

    Buffer planBInputSramFullDepth(Lifetime::Cascade, Location::Sram, CascadingBufferFormat::NHWCB,
                                   TensorShape{ 1, 112, 112, 64 }, TensorShape{ 1, 8, 8, 64 }, TraversalOrder::Xyz, 0,
                                   QuantizationInfo());

    Plan planBSramFullDepth({ { &planBInputSramFullDepth, nodeB->GetInput(0) } }, { {} });

    Buffer planBInputDram(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB,
                          TensorShape{ 1, 112, 112, 64 }, TensorShape{ 1, 112, 112, 64 }, TraversalOrder::Xyz, 0,
                          QuantizationInfo());

    Plan planBDram({ { &planBInputDram, nodeB->GetInput(0) } }, { {} });

    GraphOfParts gOfParts;
    Parts& parts = gOfParts.m_Parts;

    // Add nodeA and plans to partA
    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_SubGraph.push_back(nodeA);
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planASramFullDepth)));
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planASramPartialDepth)));
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planADram)));

    // Add nodeB and plans to partB
    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_SubGraph.push_back(nodeB);
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planBSramFullDepth)));
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planBSramPartialDepth)));
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planBDram)));

    Metadata metadata = CreateMetadata(gOfParts, hwCaps);

    // Number of parts in the metadata
    REQUIRE(metadata.size() == 2U);
    // Current part has three plans
    REQUIRE(metadata.front().m_Comp.begin()->second.size() == 3U);

    // The first plan ie planASramFullDepth is compatible with planBSramFullDepth, planBSramPartialdepth and planBDram
    REQUIRE(metadata.front().m_Comp.begin()->second.find(0)->second.size() == 4);
    // It gets merged with planBSramFullDepth first whose id is 0
    REQUIRE(metadata.front().m_Comp.begin()->second.find(0)->second.front().m_Id == 0);
    Glue* glue = &metadata.front().m_Comp.begin()->second.find(0)->second.front().m_Glue;
    // planASramFullDepth when merged with planBSramFullDepth should not need any glue
    REQUIRE(glue->m_Graph.GetOps().size() == 0U);

    // The second plan ie planASramPartialDepth is compatible with both planBSramFullDepth, planBSramPartialdepth and planBDram
    REQUIRE(metadata.front().m_Comp.begin()->second.find(1)->second.size() == 3);
    // It gets cascaded with planBSramPartialdepth first whose id is 1.
    REQUIRE(metadata.front().m_Comp.begin()->second.find(1)->second.at(1).m_Id == 1);
    // For which it needs a valid glue.
    glue = &metadata.front().m_Comp.begin()->second.find(1)->second.at(1).m_Glue;
    REQUIRE(glue->m_Graph.GetOps().size() == 2U);
    REQUIRE(glue->m_Graph.GetOps()[0] != nullptr);
    REQUIRE(glue->m_Graph.GetOps()[1] != nullptr);

    // The third plan ie planADram is compatible with both planBSramFullDepth, planBSramPartialdepth and planBDram
    REQUIRE(metadata.front().m_Comp.begin()->second.find(2)->second.size() == 3);
    // It gets cascaded with planBSramFullDepth first whose id is 0
    REQUIRE(metadata.front().m_Comp.begin()->second.find(2)->second.front().m_Id == 0);
    // For which it needs a valid glue
    glue = &metadata.front().m_Comp.begin()->second.find(2)->second.front().m_Glue;
    REQUIRE(glue->m_Graph.GetOps().size() == 1U);
    REQUIRE(glue->m_Graph.GetOps()[0] != nullptr);
}

TEST_CASE("CreateMetadata For Cascade With Depthwise Splitting for DepthwiseConvolution")
{
    const EstimationOptions estOpt;
    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);

    // Create graph A -> B
    Graph graph;

    NameOnlyNode* nodeA = graph.CreateAndAddNode<NameOnlyNode>("a");

    MceOperationNode* nodeB = graph.CreateAndAddNode<MceOperationNode>(
        TensorShape(), TensorShape(), sl::DataType::UINT8_QUANTIZED, QuantizationInfo(),
        ethosn::support_library::TensorInfo({ 1, 1, 1, 1 }, ethosn::support_library::DataType::UINT8_QUANTIZED,
                                            ethosn::support_library::DataFormat::HWIO, QuantizationInfo(0, 0.9f)),
        std::vector<uint8_t>({ 1 }), ethosn::support_library::TensorInfo({ 1, 1, 1, 1 }), std::vector<int32_t>{ 0 },
        Stride(), 0, 0, ethosn::command_stream::MceOperation::DEPTHWISE_CONVOLUTION, CompilerDataFormat::NHWCB,
        std::set<uint32_t>{ 1 });

    graph.Connect(nodeA, nodeB, 0);

    // Generate some plans for each node
    Buffer planAOutputSramFullDepth(Lifetime::Cascade, Location::Sram, CascadingBufferFormat::NHWCB,
                                    TensorShape{ 1, 112, 112, 64 }, TensorShape{ 1, 8, 8, 64 }, TraversalOrder::Xyz, 0,
                                    QuantizationInfo());
    Plan planASramFullDepth({}, { { &planAOutputSramFullDepth, nodeA } });

    Buffer planAOutputSramPartialDepth(Lifetime::Cascade, Location::Sram, CascadingBufferFormat::NHWCB,
                                       TensorShape{ 1, 112, 112, 64 }, TensorShape{ 1, 8, 8, 16 }, TraversalOrder::Xyz,
                                       0, QuantizationInfo());
    Plan planASramPartialDepth({}, { { &planAOutputSramPartialDepth, nodeA } });

    Buffer planAOutputDram(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB,
                           TensorShape{ 1, 112, 112, 64 }, TensorShape{ 1, 112, 112, 64 }, TraversalOrder::Xyz, 0,
                           QuantizationInfo());
    Plan planADram({}, { { &planAOutputDram, nodeA } });

    Buffer planBInputSramPartialDepth(Lifetime::Cascade, Location::Sram, CascadingBufferFormat::NHWCB,
                                      TensorShape{ 1, 112, 112, 64 }, TensorShape{ 1, 8, 8, 16 }, TraversalOrder::Xyz,
                                      0, QuantizationInfo());

    Plan planBSramPartialDepth({ { &planBInputSramPartialDepth, nodeB->GetInput(0) } }, { {} });

    Buffer planBInputSramFullDepth(Lifetime::Cascade, Location::Sram, CascadingBufferFormat::NHWCB,
                                   TensorShape{ 1, 112, 112, 64 }, TensorShape{ 1, 8, 8, 64 }, TraversalOrder::Xyz, 0,
                                   QuantizationInfo());

    Plan planBSramFullDepth({ { &planBInputSramFullDepth, nodeB->GetInput(0) } }, { {} });

    Buffer planBInputDram(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB,
                          TensorShape{ 1, 112, 112, 64 }, TensorShape{ 1, 112, 112, 64 }, TraversalOrder::Xyz, 0,
                          QuantizationInfo());

    Plan planBDram({ { &planBInputDram, nodeB->GetInput(0) } }, { {} });

    GraphOfParts gOfParts;
    Parts& parts = gOfParts.m_Parts;

    // Add nodeA and plans to partA
    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_SubGraph.push_back(nodeA);
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planASramFullDepth)));
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planASramPartialDepth)));
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planADram)));

    // Add nodeB and plans to partB
    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_SubGraph.push_back(nodeB);
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planBSramFullDepth)));
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planBSramPartialDepth)));
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planBDram)));

    Metadata metadata = CreateMetadata(gOfParts, hwCaps);

    // Number of parts in the metadata
    REQUIRE(metadata.size() == 2U);
    // Current part has three plans
    REQUIRE(metadata.front().m_Comp.begin()->second.size() == 3U);

    // The first plan ie planASramFullDepth is compatible with planBSramFullDepth, planBSramPartialdepth and planBDram
    REQUIRE(metadata.front().m_Comp.begin()->second.find(0)->second.size() == 4);
    // It gets merged with planBSramFullDepth first whose id is 0
    REQUIRE(metadata.front().m_Comp.begin()->second.find(0)->second.front().m_Id == 0);
    Glue* glue = &metadata.front().m_Comp.begin()->second.find(0)->second.front().m_Glue;
    // planASramFullDepth when merged with planBSramFullDepth should not need any glue
    REQUIRE(glue->m_Graph.GetOps().size() == 0U);

    // The second plan ie planASramPartialDepth is compatible with both planBSramFullDepth, planBSramPartialdepth and planBDram
    REQUIRE(metadata.front().m_Comp.begin()->second.find(1)->second.size() == 4);
    // It gets merged with planBSramPartialdepth first whose id is 1.
    REQUIRE(metadata.front().m_Comp.begin()->second.find(1)->second.at(1).m_Id == 1);
    // For which it does not need a valid glue.
    glue = &metadata.front().m_Comp.begin()->second.find(1)->second.at(1).m_Glue;
    REQUIRE(glue->m_Graph.GetOps().size() == 0U);

    // The third plan ie planADram is compatible with both planBSramFullDepth, planBSramPartialdepth and planBDram
    REQUIRE(metadata.front().m_Comp.begin()->second.find(2)->second.size() == 3);
    // It gets cascaded with planBSramFullDepth first whose id is 0
    REQUIRE(metadata.front().m_Comp.begin()->second.find(2)->second.front().m_Id == 0);
    // For which it needs a valid glue
    glue = &metadata.front().m_Comp.begin()->second.find(2)->second.front().m_Glue;
    REQUIRE(glue->m_Graph.GetOps().size() == 1U);
    REQUIRE(glue->m_Graph.GetOps()[0] != nullptr);
}

/// Checks that CreateMedata correctly populates the metadata structure.
TEST_CASE("CreateMetadata Simple")
{
    const EstimationOptions estOpt;
    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
    // Create simple graph A -> B -> C
    Graph graph;
    NameOnlyNode* nodeA = graph.CreateAndAddNode<NameOnlyNode>("a");
    NameOnlyNode* nodeB = graph.CreateAndAddNode<NameOnlyNode>("b");
    NameOnlyNode* nodeC = graph.CreateAndAddNode<NameOnlyNode>("c");
    graph.Connect(nodeA, nodeB, 0);
    graph.Connect(nodeB, nodeC, 0);

    // Generate some plans for each node
    Buffer planAOutputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(),
                           TensorShape{ 1, 2, 3, 4 }, TraversalOrder::Xyz, 0, QuantizationInfo());
    Plan planASram({}, { { &planAOutputSram, nodeA } });

    Buffer planAOutputDram(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(),
                           TraversalOrder::Xyz, 0, QuantizationInfo());
    Plan planADram({}, { { &planAOutputDram, nodeA } });

    Buffer planBInputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(),
                          TensorShape{ 5, 6, 7, 8 }, TraversalOrder::Xyz, 0, QuantizationInfo());

    Buffer planBOutputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(),
                           TensorShape{ 5, 6, 7, 8 }, TraversalOrder::Xyz, 0, QuantizationInfo());

    Plan planBSram({ { &planBInputSram, nodeB->GetInput(0) } }, { { &planBOutputSram, nodeB } });

    Buffer planBInputDram(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(),
                          TraversalOrder::Xyz, 0, QuantizationInfo());

    Buffer planBOutputDram(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(),
                           TraversalOrder::Xyz, 0, QuantizationInfo());

    Plan planBDram({ { &planBInputDram, nodeB->GetInput(0) } }, { { &planBOutputDram, nodeB } });

    Buffer planCInputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(),
                          TensorShape{ 5, 6, 7, 8 }, TraversalOrder::Xyz, 0, QuantizationInfo());
    Plan planCSram({ { &planCInputSram, nodeC->GetInput(0) } }, {});

    Buffer planCInputDram(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(),
                          TraversalOrder::Xyz, 0, QuantizationInfo());
    Plan planCDram({ { &planCInputDram, nodeC->GetInput(0) } }, {});

    GraphOfParts gOfParts;
    Parts& parts = gOfParts.m_Parts;

    // Add nodeA and plans to partA
    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_SubGraph.push_back(nodeA);
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planASram)));
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planADram)));

    // Add nodeB and plans to partB
    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_SubGraph.push_back(nodeB);
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planBSram)));
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planBDram)));

    // Add nodeC and plans to partC
    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_SubGraph.push_back(nodeC);
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planCSram)));
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planCDram)));

    Metadata metadata = CreateMetadata(gOfParts, hwCaps);

    // Number of parts in the metadata
    REQUIRE(metadata.size() == 3U);
    // First part has no input connected
    REQUIRE(metadata.front().m_Source.size() == 0);
    REQUIRE(metadata.front().m_Destination.size() == 1U);
    REQUIRE(metadata.front().m_Destination.find(nodeB->GetInput(0)) != metadata.front().m_Destination.end());
    // Only one output for this part
    REQUIRE(metadata.front().m_Comp.size() == 1U);
    // PartId of next part
    REQUIRE(metadata.front().m_Comp.begin()->first == nodeB->GetInput(0));
    // Current part has two plans
    REQUIRE(metadata.front().m_Comp.begin()->second.size() == 2U);
    REQUIRE(metadata.front().m_Comp.begin()->second.find(0) != metadata.front().m_Comp.begin()->second.end());
    // Plan 0
    REQUIRE(metadata.front().m_Comp.begin()->second.find(0)->first == 0);
    // Can be merged with plan 0 of next part
    REQUIRE(metadata.front().m_Comp.begin()->second.find(0)->second.front().m_Id == 0);
    //planASram, planBSram
    Glue* glue = &metadata.front().m_Comp.begin()->second.find(0)->second.front().m_Glue;
    REQUIRE(glue->m_Graph.GetOps().size() == 2U);
    REQUIRE(glue->m_Graph.GetOps()[0] != nullptr);
    REQUIRE(glue->m_Graph.GetOps()[1] != nullptr);
    REQUIRE(glue->m_Graph.GetBuffers().size() == 1U);
    REQUIRE(glue->m_Graph.GetBuffers()[0]->m_Location == Location::Dram);
    REQUIRE(glue->m_Graph.GetProducer(glue->m_Graph.GetBuffers()[0]) == glue->m_Graph.GetOps()[0]);
    REQUIRE(glue->m_Graph.GetConsumers(glue->m_Graph.GetBuffers()[0]) ==
            std::vector<std::pair<Op*, uint32_t>>{ { glue->m_Graph.GetOps()[1], 0 } });
    REQUIRE(glue->m_InputSlot == std::pair<Op*, uint32_t>{ glue->m_Graph.GetOps()[0], 0 });
    REQUIRE(glue->m_Output == glue->m_Graph.GetOps()[1]);

    // Can be merged with plan 1 of next part
    REQUIRE(metadata.front().m_Comp.begin()->second.find(0)->second.back().m_Id == 1U);
    //planASram, planBDram
    glue = &metadata.front().m_Comp.begin()->second.find(0)->second.back().m_Glue;
    REQUIRE(glue->m_Graph.GetOps().size() == 1U);
    REQUIRE(glue->m_Graph.GetOps()[0] != nullptr);
    REQUIRE(glue->m_Graph.GetBuffers().size() == 0);
    REQUIRE(glue->m_InputSlot == std::pair<Op*, uint32_t>{ glue->m_Graph.GetOps()[0], 0 });
    REQUIRE(glue->m_Output == glue->m_Graph.GetOps()[0]);

    REQUIRE(metadata.front().m_Comp.begin()->second.find(1U) != metadata.front().m_Comp.begin()->second.end());
    // Plan 0
    REQUIRE(metadata.front().m_Comp.begin()->second.find(1)->first == 1U);
    // Can be merged with plan 0 and 1 of next part
    REQUIRE(metadata.front().m_Comp.begin()->second.find(1)->second.front().m_Id == 0);
    REQUIRE(metadata.front().m_Comp.begin()->second.find(1)->second.back().m_Id == 1U);

    // Second part input is connected with part 0
    REQUIRE(metadata.at(1).m_Source.size() == 1U);
    REQUIRE(metadata.at(1).m_Source.find(nodeB->GetInput(0)) != metadata.at(1).m_Source.end());
    REQUIRE(metadata.at(1).m_Destination.size() == 1U);
    REQUIRE(metadata.at(1).m_Destination.find(nodeC->GetInput(0)) != metadata.at(1).m_Destination.end());
    // Only one output for this part
    REQUIRE(metadata.at(1).m_Comp.size() == 1U);
    // PartId of next part
    REQUIRE(metadata.at(1).m_Comp.begin()->first == nodeC->GetInput(0));
    // Current part has two plans
    REQUIRE(metadata.at(1).m_Comp.begin()->second.size() == 2U);
    // Both parts are in the metadata
    REQUIRE(metadata.at(1).m_Comp.begin()->second.find(0) != metadata.at(1).m_Comp.begin()->second.end());
    REQUIRE(metadata.at(1).m_Comp.begin()->second.find(1U) != metadata.at(1).m_Comp.begin()->second.end());
}

/// Checks that CreateMedata correctly populates the metadata structure.
TEST_CASE("CreateMetadata Of Graph With Branches")
{
    const EstimationOptions estOpt;
    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
    /* Create graph:

              B - D
            /      \
          A          F
            \      /
              C - E

    */
    Graph graph;
    NameOnlyNode* nodeA = graph.CreateAndAddNode<NameOnlyNode>("a");
    NameOnlyNode* nodeB = graph.CreateAndAddNode<NameOnlyNode>("b");
    NameOnlyNode* nodeC = graph.CreateAndAddNode<NameOnlyNode>("c");
    NameOnlyNode* nodeD = graph.CreateAndAddNode<NameOnlyNode>("d");
    NameOnlyNode* nodeE = graph.CreateAndAddNode<NameOnlyNode>("e");
    NameOnlyNode* nodeF = graph.CreateAndAddNode<NameOnlyNode>("f");
    graph.Connect(nodeA, nodeB, 0);
    graph.Connect(nodeA, nodeC, 0);
    graph.Connect(nodeB, nodeD, 0);
    graph.Connect(nodeC, nodeE, 0);
    graph.Connect(nodeD, nodeF, 0);
    graph.Connect(nodeE, nodeF, 0);

    // Generate some plans for each node

    // Node A
    Buffer planAOutputSramToB(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(),
                              TensorShape{ 1, 2, 3, 4 }, TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer planAOutputSramToC(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(),
                              TensorShape{ 1, 2, 3, 4 }, TraversalOrder::Xyz, 0, QuantizationInfo());

    Plan planASram({}, { { &planAOutputSramToB, nodeA }, { &planAOutputSramToC, nodeA } });

    Buffer planAOutputDramToB(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB, TensorShape(),
                              TensorShape(), TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer planAOutputDramToC(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB, TensorShape(),
                              TensorShape(), TraversalOrder::Xyz, 0, QuantizationInfo());

    Plan planADram({}, { { &planAOutputDramToB, nodeA }, { &planAOutputDramToC, nodeA } });

    // Node B
    Buffer planBInputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(),
                          TensorShape{ 1, 2, 3, 4 }, TraversalOrder::Xyz, 0, QuantizationInfo());

    Buffer planBOutputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(),
                           TensorShape{ 5, 6, 7, 8 }, TraversalOrder::Xyz, 0, QuantizationInfo());

    Plan planBSram({ { &planBInputSram, nodeB->GetInput(0) } }, { { &planBOutputSram, nodeB } });

    Buffer planBInputDram(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(),
                          TraversalOrder::Xyz, 0, QuantizationInfo());

    Buffer planBOutputDram(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(),
                           TraversalOrder::Xyz, 0, QuantizationInfo());

    Plan planBDram({ { &planBInputDram, nodeB->GetInput(0) } }, { { &planBOutputDram, nodeB } });

    // Node C
    Buffer planCInputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(),
                          TensorShape{ 1, 2, 3, 4 }, TraversalOrder::Xyz, 0, QuantizationInfo());

    Buffer planCOutputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(),
                           TensorShape{ 2, 2, 2, 2 }, TraversalOrder::Xyz, 0, QuantizationInfo());

    Plan planCSram({ { &planCInputSram, nodeC->GetInput(0) } }, { { &planCOutputSram, nodeC } });

    Buffer planCInputDram(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(),
                          TraversalOrder::Xyz, 0, QuantizationInfo());

    Buffer planCOutputDram(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(),
                           TraversalOrder::Xyz, 0, QuantizationInfo());

    Plan planCDram({ { &planCInputDram, nodeC->GetInput(0) } }, { { &planCOutputDram, nodeC } });

    // Node D
    Buffer planDInputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(),
                          TensorShape{ 1, 2, 3, 4 }, TraversalOrder::Xyz, 0, QuantizationInfo());

    Buffer planDOutputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(),
                           TensorShape{ 5, 6, 7, 8 }, TraversalOrder::Xyz, 0, QuantizationInfo());

    Plan planDSram({ { &planDInputSram, nodeD->GetInput(0) } }, { { &planDOutputSram, nodeD } });

    Buffer planDInputDram(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(),
                          TraversalOrder::Xyz, 0, QuantizationInfo());

    Buffer planDOutputDram(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(),
                           TraversalOrder::Xyz, 0, QuantizationInfo());

    Plan planDDram({ { &planDInputDram, nodeD->GetInput(0) } }, { { &planDOutputDram, nodeD } });

    // Node E
    Buffer planEInputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(),
                          TensorShape{ 1, 2, 3, 4 }, TraversalOrder::Xyz, 0, QuantizationInfo());

    Buffer planEOutputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(),
                           TensorShape{ 5, 6, 7, 8 }, TraversalOrder::Xyz, 0, QuantizationInfo());

    Plan planESram({ { &planEInputSram, nodeE->GetInput(0) } }, { { &planEOutputSram, nodeE } });

    Buffer planEInputDram(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(),
                          TraversalOrder::Xyz, 0, QuantizationInfo());

    Buffer planEOutputDram(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(),
                           TraversalOrder::Xyz, 0, QuantizationInfo());

    Plan planEDram({ { &planEInputDram, nodeE->GetInput(0) } }, { { &planEOutputDram, nodeE } });

    // Node F
    Buffer planFInputSramFromD(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(),
                               TensorShape{ 5, 6, 7, 8 }, TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer planFInputSramFromE(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(),
                               TensorShape{ 5, 6, 7, 8 }, TraversalOrder::Xyz, 0, QuantizationInfo());

    Plan planFSram({ { &planFInputSramFromD, nodeF->GetInput(1) }, { &planFInputSramFromE, nodeF->GetInput(0) } }, {});

    Buffer planFInputDramFromD(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB, TensorShape(),
                               TensorShape(), TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer planFInputDramFromE(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB, TensorShape(),
                               TensorShape(), TraversalOrder::Xyz, 0, QuantizationInfo());

    Plan planFDram({ { &planFInputDramFromD, nodeF->GetInput(1) }, { &planFInputDramFromE, nodeF->GetInput(0) } }, {});

    GraphOfParts gOfParts;
    Parts& parts = gOfParts.m_Parts;

    // Topological sort:  A, B, D, C, E, F
    // Add nodeA and plans to partA
    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_SubGraph.push_back(nodeA);
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planASram)));
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planADram)));

    // Add nodeB and plans to partB
    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_SubGraph.push_back(nodeB);
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planBSram)));
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planBDram)));

    // Add nodeC and plans to partD
    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_SubGraph.push_back(nodeD);
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planDSram)));
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planDDram)));

    // Add nodeC and plans to partC
    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_SubGraph.push_back(nodeC);
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planCSram)));
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planCDram)));

    // Add nodeC and plans to partE
    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_SubGraph.push_back(nodeE);
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planESram)));
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planEDram)));

    // Add nodeC and plans to partF
    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_SubGraph.push_back(nodeF);
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planFSram)));
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planFDram)));

    Metadata metadata = CreateMetadata(gOfParts, hwCaps);

    // Number of parts in the metadata
    REQUIRE(metadata.size() == 6U);
    // First part
    REQUIRE(metadata.at(0).m_Source.size() == 0);
    REQUIRE(metadata.at(0).m_Destination.size() == 2U);
    REQUIRE(metadata.at(0).m_Destination.find(nodeB->GetInput(0)) != metadata.at(0).m_Destination.end());
    REQUIRE(metadata.at(0).m_Destination.find(nodeC->GetInput(0)) != metadata.at(0).m_Destination.end());
    // Second part
    REQUIRE(metadata.at(1U).m_Source.size() == 1U);
    REQUIRE(metadata.at(1U).m_Source.find(nodeB->GetInput(0)) != metadata.at(1U).m_Source.end());
    REQUIRE(metadata.at(1U).m_Destination.size() == 1U);
    REQUIRE(metadata.at(1U).m_Destination.find(nodeD->GetInput(0)) != metadata.at(1U).m_Destination.end());
    // Third part
    REQUIRE(metadata.at(2U).m_Source.size() == 1U);
    REQUIRE(metadata.at(2U).m_Source.find(nodeD->GetInput(0)) != metadata.at(2U).m_Source.end());
    REQUIRE(metadata.at(2U).m_Destination.size() == 1U);
    REQUIRE(metadata.at(2U).m_Destination.find(nodeF->GetInput(1)) != metadata.at(2U).m_Destination.end());
    // Fourth part
    REQUIRE(metadata.at(3U).m_Source.size() == 1U);
    REQUIRE(metadata.at(3U).m_Source.find(nodeC->GetInput(0)) != metadata.at(3U).m_Source.end());
    REQUIRE(metadata.at(3U).m_Destination.size() == 1U);
    REQUIRE(metadata.at(3U).m_Destination.find(nodeE->GetInput(0)) != metadata.at(3U).m_Destination.end());
    // Fifth part
    REQUIRE(metadata.at(4U).m_Source.size() == 1U);
    REQUIRE(metadata.at(4U).m_Source.find(nodeE->GetInput(0)) != metadata.at(4U).m_Source.end());
    REQUIRE(metadata.at(4U).m_Destination.size() == 1U);
    REQUIRE(metadata.at(4U).m_Destination.find(nodeF->GetInput(0)) != metadata.at(4U).m_Destination.end());
    // Sixth part
    REQUIRE(metadata.at(5U).m_Source.size() == 2U);
    REQUIRE(metadata.at(5U).m_Source.find(nodeF->GetInput(0)) != metadata.at(5U).m_Source.end());
    REQUIRE(metadata.at(5U).m_Source.find(nodeF->GetInput(1)) != metadata.at(5U).m_Source.end());
    REQUIRE(metadata.at(5U).m_Destination.size() == 0);

    // Two outputs for this part
    REQUIRE(metadata.front().m_Comp.size() == 2U);
    // PartId of next part
    REQUIRE(metadata.front().m_Comp.find(nodeB->GetInput(0)) != metadata.front().m_Comp.end());
    REQUIRE(metadata.front().m_Comp.find(nodeC->GetInput(0)) != metadata.front().m_Comp.end());

    // Compatible plans with the destination part 1
    CompatiblePlansOfPart* cPlsOfPa = &metadata.front().m_Comp.find(nodeB->GetInput(0))->second;
    // Current part has two plans (Dram plan)
    REQUIRE(cPlsOfPa->size() == 2U);
    REQUIRE(cPlsOfPa->find(1U) != cPlsOfPa->end());
    {
        // Plan 1 has DRAM location since this part has multiple outputs
        const Edge* edge = metadata.front().m_Destination.find(nodeB->GetInput(0))->first;
        REQUIRE(edge != nullptr);
        Buffer* buf = (*(*parts.at(0).get()).m_Plans.at(1U).get()).GetOutputBuffer(edge->GetSource());
        REQUIRE(buf != nullptr);
        REQUIRE(buf->m_Location == Location::Dram);
        // This plan is compatible with all the plans (2) of next part
        CompatiblePlans* cPls = &(cPlsOfPa->find(1U)->second);
        REQUIRE(cPls != nullptr);
        REQUIRE(cPls->size() == 2U);
        CompatiblePlans::const_iterator it = cPls->begin();
        while (it != cPls->end())
        {
            const Glue& glue = it->m_Glue;
            REQUIRE(glue.m_Graph.GetOps().size() <= 1U);
            ++it;
        }
    }

    // Compatible plans with the destination part 3
    cPlsOfPa = &metadata.front().m_Comp.find(nodeC->GetInput(0))->second;
    // Current part has two compatible plans
    REQUIRE(cPlsOfPa->size() == 2U);
    REQUIRE(cPlsOfPa->find(1U) != cPlsOfPa->end());
    {
        // Plan 1 has DRAM location since this part has multiple outputs
        const Edge* edge = metadata.front().m_Destination.find(nodeB->GetInput(0))->first;
        REQUIRE(edge != nullptr);
        Buffer* buf = (*(*parts.at(0).get()).m_Plans.at(1U).get()).GetOutputBuffer(edge->GetSource());
        REQUIRE(buf != nullptr);
        REQUIRE(buf->m_Location == Location::Dram);
        // This plan is compatible with all the plans (2) of next part
        CompatiblePlans* cPls = &(cPlsOfPa->find(1U)->second);
        REQUIRE(cPls != nullptr);
        REQUIRE(cPls->size() == 2U);
        CompatiblePlans::const_iterator it = cPls->begin();
        while (it != cPls->end())
        {
            const Glue& glue = it->m_Glue;
            REQUIRE(glue.m_Graph.GetOps().size() <= 1U);
            ++it;
        }
    }
    REQUIRE(cPlsOfPa->find(0) != cPlsOfPa->end());
    {
        // This plan is compatible with only a plan of next part
        CompatiblePlans* cPls = &(cPlsOfPa->find(0U)->second);
        REQUIRE(cPls != nullptr);
        REQUIRE(cPls->size() == 1U);
    }

    // Go to part 2
    // One output for this part
    REQUIRE(metadata.at(2U).m_Comp.size() == 1U);
    // PartId of next part
    REQUIRE(metadata.at(2U).m_Comp.find(nodeF->GetInput(1)) != metadata.at(2).m_Comp.end());

    // Compatible plans with the destination part 1
    cPlsOfPa = &metadata.at(2U).m_Comp.find(nodeF->GetInput(1))->second;
    // Current part has two plans
    REQUIRE(cPlsOfPa->size() == 2U);
    REQUIRE(cPlsOfPa->find(1U) != cPlsOfPa->end());
    {
        // Plan 1 has DRAM location
        const Edge* edge = metadata.at(2U).m_Destination.find(nodeF->GetInput(1))->first;
        REQUIRE(edge != nullptr);
        Buffer* buf = (*(*parts.at(2U).get()).m_Plans.at(1U).get()).GetOutputBuffer(edge->GetSource());
        REQUIRE(buf != nullptr);
        REQUIRE(buf->m_Location == Location::Dram);
        // This plan is compatible with all the plans (2) of next part
        CompatiblePlans* cPls = &(cPlsOfPa->find(1U)->second);
        REQUIRE(cPls != nullptr);
        REQUIRE(cPls->size() == 2U);
        CompatiblePlans::const_iterator it = cPls->begin();
        while (it != cPls->end())
        {
            const Glue& glue = it->m_Glue;
            REQUIRE(glue.m_Graph.GetOps().size() <= 1U);
            ++it;
        }
    }
    REQUIRE(cPlsOfPa->find(0) != cPlsOfPa->end());
    {
        // Plan 0 has SRAM location
        const Edge* edge = metadata.at(2U).m_Destination.find(nodeF->GetInput(1))->first;
        REQUIRE(edge != nullptr);
        Buffer* buf = (*(*parts.at(2U).get()).m_Plans.at(0).get()).GetOutputBuffer(edge->GetSource());
        REQUIRE(buf != nullptr);
        REQUIRE(buf->m_Location == Location::Sram);
        // This plan is compatible with only a plan of next part
        CompatiblePlans* cPls = &(cPlsOfPa->find(0)->second);
        REQUIRE(cPls != nullptr);
        REQUIRE(cPls->size() == 1U);
    }

    // Go to part 4
    // One output for this part
    REQUIRE(metadata.at(4U).m_Comp.size() == 1U);
    // PartId of next part
    REQUIRE(metadata.at(4U).m_Comp.find(nodeF->GetInput(0)) != metadata.at(4).m_Comp.end());

    // Compatible plans with the destination part 1
    cPlsOfPa = &metadata.at(4U).m_Comp.find(nodeF->GetInput(0))->second;
    // Current part has two plans
    REQUIRE(cPlsOfPa->size() == 2U);
    REQUIRE(cPlsOfPa->find(1) != cPlsOfPa->end());
    {
        // Plan 1 has DRAM location
        const Edge* edge = metadata.at(4U).m_Destination.find(nodeF->GetInput(0))->first;
        REQUIRE(edge != nullptr);
        Buffer* buf = (*(*parts.at(4U).get()).m_Plans.at(1U).get()).GetOutputBuffer(edge->GetSource());
        REQUIRE(buf != nullptr);
        REQUIRE(buf->m_Location == Location::Dram);
        // This plan is compatible with all the plans (2) of next part
        CompatiblePlans* cPls = &(cPlsOfPa->find(1U)->second);
        REQUIRE(cPls != nullptr);
        REQUIRE(cPls->size() == 2U);
        CompatiblePlans::const_iterator it = cPls->begin();
        while (it != cPls->end())
        {
            const Glue& glue = it->m_Glue;
            REQUIRE(glue.m_Graph.GetOps().size() <= 1U);
            ++it;
        }
    }
    REQUIRE(cPlsOfPa->find(0) != cPlsOfPa->end());
    {
        // Plan 0 has SRAM location
        const Edge* edge = metadata.at(4U).m_Destination.find(nodeF->GetInput(0))->first;
        REQUIRE(edge != nullptr);
        Buffer* buf = (*(*parts.at(4U).get()).m_Plans.at(0).get()).GetOutputBuffer(edge->GetSource());
        REQUIRE(buf != nullptr);
        REQUIRE(buf->m_Location == Location::Sram);
        // This plan is compatible with only a plan of next part
        CompatiblePlans* cPls = &(cPlsOfPa->find(0)->second);
        REQUIRE(cPls != nullptr);
        REQUIRE(cPls->size() == 1U);
    }
}

/// Checks that CreateSeeds correctly generates the seeds
TEST_CASE("CreateSeeds Simple")
{
    const EstimationOptions estOpt;
    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
    // Create simple graph A -> B
    Graph graph;
    NameOnlyNode* nodeA = graph.CreateAndAddNode<NameOnlyNode>("a");
    NameOnlyNode* nodeB = graph.CreateAndAddNode<NameOnlyNode>("b");
    graph.Connect(nodeA, nodeB, 0);

    // Generate some plans for each node
    Buffer planAOutputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(),
                           TensorShape{ 1, 2, 3, 4 }, TraversalOrder::Xyz, 0, QuantizationInfo());
    Plan planASram({}, { { &planAOutputSram, nodeA } });

    Buffer planAOutputDram(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(),
                           TraversalOrder::Xyz, 0, QuantizationInfo());
    Plan planADram({}, { { &planAOutputDram, nodeA } });

    Buffer planBInputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(),
                          TensorShape{ 5, 6, 7, 8 }, TraversalOrder::Xyz, 0, QuantizationInfo());
    Plan planBSram({ { &planBInputSram, nodeB->GetInput(0) } }, {});

    Buffer planBInputDram(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(),
                          TraversalOrder::Xyz, 0, QuantizationInfo());
    Plan planBDram({ { &planBInputDram, nodeB->GetInput(0) } }, {});

    GraphOfParts gOfParts;
    Parts& parts = gOfParts.m_Parts;

    // Add nodeA and plans to partA
    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_SubGraph.push_back(nodeA);
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planASram)));
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planADram)));

    // Add nodeB and plans to partB
    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_SubGraph.push_back(nodeB);
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planBSram)));
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planBDram)));

    Metadata metadata = CreateMetadata(gOfParts, hwCaps);

    Combinations combs = CreateSeeds(gOfParts, metadata, hwCaps);

    // All plan are compatible, the total number of seeds is the product of the number of plans
    REQUIRE(combs.size() == 4U);
    // Seed 0
    REQUIRE(combs.at(0).m_Elems.size() == 1U);
    REQUIRE(combs.at(0).m_Elems.at(0).m_PartId == 0);
    REQUIRE(combs.at(0).m_Elems.at(0).m_PlanId == 0);
    REQUIRE(combs.at(0).m_Elems.at(0).m_Glues.size() != 0);
    REQUIRE((combs.at(0).m_Elems.at(0).m_Glues.begin()->second).m_Glue->m_Graph.GetOps().size() == 2U);
    REQUIRE((combs.at(0).m_Elems.at(0).m_Glues.begin()->second).m_Id == 0);
    // Seed 1
    REQUIRE(combs.at(1).m_Elems.size() == 1U);
    REQUIRE(combs.at(1).m_Elems.at(0).m_PartId == 0);
    REQUIRE(combs.at(1).m_Elems.at(0).m_PlanId == 0);
    REQUIRE(combs.at(1).m_Elems.at(0).m_Glues.size() != 0);
    REQUIRE((combs.at(1).m_Elems.at(0).m_Glues.begin()->second).m_Glue->m_Graph.GetOps().size() == 1U);
    REQUIRE((combs.at(1).m_Elems.at(0).m_Glues.begin()->second).m_Id == 1U);
    // Seed 2
    REQUIRE(combs.at(2).m_Elems.size() == 1U);
    REQUIRE(combs.at(2).m_Elems.at(0).m_PartId == 0);
    REQUIRE(combs.at(2).m_Elems.at(0).m_PlanId == 1U);
    REQUIRE(combs.at(2).m_Elems.at(0).m_Glues.size() != 0);
    REQUIRE((combs.at(2).m_Elems.at(0).m_Glues.begin()->second).m_Glue->m_Graph.GetOps().size() == 1U);
    REQUIRE((combs.at(2).m_Elems.at(0).m_Glues.begin()->second).m_Id == 0);
    // Seed 3
    REQUIRE(combs.at(3).m_Elems.size() == 1U);
    REQUIRE(combs.at(3).m_Elems.at(0).m_PartId == 0);
    REQUIRE(combs.at(3).m_Elems.at(0).m_PlanId == 1U);
    REQUIRE(combs.at(3).m_Elems.at(0).m_Glues.size() != 0);
    REQUIRE((combs.at(3).m_Elems.at(0).m_Glues.begin()->second).m_Glue->m_Graph.GetOps().size() == 0);
    REQUIRE((combs.at(3).m_Elems.at(0).m_Glues.begin()->second).m_Id == 1U);
}

/// Checks that GrowSeeds generates all the combinations
TEST_CASE("GrowSeeds Simple")
{
    const EstimationOptions estOpt;
    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
    // Create simple graph A -> B -> C
    Graph graph;
    NameOnlyNode* nodeA = graph.CreateAndAddNode<NameOnlyNode>("a");
    NameOnlyNode* nodeB = graph.CreateAndAddNode<NameOnlyNode>("b");
    NameOnlyNode* nodeC = graph.CreateAndAddNode<NameOnlyNode>("c");
    graph.Connect(nodeA, nodeB, 0);
    graph.Connect(nodeB, nodeC, 0);

    // Generate some plans for each node
    Plan planASram;
    ConfigurePlan(OutputPlanConfigurator(planASram, nodeA), Lifetime::Cascade, Location::Sram,
                  CascadingBufferFormat::NHWCB, TensorShape(), TensorShape{ 1, 2, 3, 4 }, TraversalOrder::Xyz, 4 * 16,
                  QuantizationInfo());

    Plan planADram;
    ConfigurePlan(OutputPlanConfigurator(planADram, nodeA), Lifetime::Cascade, Location::Dram,
                  CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(), TraversalOrder::Xyz, 4 * 16,
                  QuantizationInfo());

    Plan planBSram;
    ConfigurePlan(InputPlanConfigurator(planBSram, nodeB), Lifetime::Cascade, Location::Sram,
                  CascadingBufferFormat::NHWCB, TensorShape(), TensorShape{ 5, 6, 7, 8 }, TraversalOrder::Xyz, 4 * 16,
                  QuantizationInfo());
    ConfigurePlan(OutputPlanConfigurator(planBSram, nodeB), Lifetime::Cascade, Location::Sram,
                  CascadingBufferFormat::NHWCB, TensorShape(), TensorShape{ 5, 6, 7, 8 }, TraversalOrder::Xyz, 4 * 16,
                  QuantizationInfo());

    Plan planBDram;
    ConfigurePlan(InputPlanConfigurator(planBDram, nodeB), Lifetime::Cascade, Location::Dram,
                  CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(), TraversalOrder::Xyz, 4 * 16,
                  QuantizationInfo());
    ConfigurePlan(OutputPlanConfigurator(planBDram, nodeB), Lifetime::Cascade, Location::Dram,
                  CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(), TraversalOrder::Xyz, 4 * 16,
                  QuantizationInfo());

    Plan planCSram;
    ConfigurePlan(InputPlanConfigurator(planCSram, nodeC), Lifetime::Cascade, Location::Sram,
                  CascadingBufferFormat::NHWCB, TensorShape(), TensorShape{ 5, 6, 7, 8 }, TraversalOrder::Xyz, 4 * 16,
                  QuantizationInfo());

    Plan planCDram;
    ConfigurePlan(InputPlanConfigurator(planCDram, nodeC), Lifetime::Cascade, Location::Dram,
                  CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(), TraversalOrder::Xyz, 4 * 16,
                  QuantizationInfo());

    GraphOfParts gOfParts;
    Parts& parts = gOfParts.m_Parts;

    // Add nodeA and plans to partA
    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_SubGraph.push_back(nodeA);
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planASram)));
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planADram)));

    // Add nodeB and plans to partB
    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_SubGraph.push_back(nodeB);
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planBSram)));
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planBDram)));

    // Add nodeC and plans to partC
    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_SubGraph.push_back(nodeC);
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planCSram)));
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planCDram)));

    Metadata metadata = CreateMetadata(gOfParts, hwCaps);

    Combinations combs = CreateSeeds(gOfParts, metadata, hwCaps);
    // All plan are compatible, the total number of seeds is the product of the number of plans (plus "Back to Dram" plans)
    REQUIRE(combs.size() == 4U);

    GrownSeeds res = GrowSeeds(combs, gOfParts, metadata, hwCaps, GrowScheme::Default);
    REQUIRE(res.m_Combinations.size() == 10U);
    REQUIRE(res.m_Terminated == false);
    res = GrowSeeds(res.m_Combinations, gOfParts, metadata, hwCaps, GrowScheme::Default);
    REQUIRE(res.m_Combinations.size() == 10U);
    REQUIRE(res.m_Terminated == false);
    res = GrowSeeds(res.m_Combinations, gOfParts, metadata, hwCaps, GrowScheme::Default);
    REQUIRE(res.m_Combinations.size() == 10U);
    REQUIRE(res.m_Terminated == true);

    for (size_t i = 0; i < res.m_Combinations.size(); ++i)
    {
        INFO("Combination number is: " << i);
        // All the combinations are complete
        REQUIRE(res.m_Combinations.at(i).m_Elems.size() == 3U);
        // All the combinations have the correct sequence of parts
        for (size_t j = 0; j < parts.size(); ++j)
        {
            REQUIRE(res.m_Combinations.at(i).m_Elems.at(j).m_PartId == j);
        }
    }

    // All the combinations have the correct diagnostic
    REQUIRE(res.m_Combinations.at(0).m_Scratch.m_AllocatedSram == 8U * 16U);
    REQUIRE(res.m_Combinations.at(0).m_Scratch.m_Score == 1U);

    REQUIRE(res.m_Combinations.at(1).m_Scratch.m_AllocatedSram == 4U * 16U);

    REQUIRE(res.m_Combinations.at(2).m_Scratch.m_AllocatedSram == 0);

    REQUIRE(res.m_Combinations.at(3).m_Scratch.m_AllocatedSram == 4U * 16U);

    REQUIRE(res.m_Combinations.at(4).m_Scratch.m_AllocatedSram == 0);

    REQUIRE(res.m_Combinations.at(5).m_Scratch.m_AllocatedSram == 8U * 16U);

    REQUIRE(res.m_Combinations.at(6).m_Scratch.m_AllocatedSram == 4U * 16U);

    REQUIRE(res.m_Combinations.at(7).m_Scratch.m_AllocatedSram == 0);

    REQUIRE(res.m_Combinations.at(8).m_Scratch.m_AllocatedSram == 4U * 16U);

    REQUIRE(res.m_Combinations.at(9).m_Scratch.m_AllocatedSram == 0);
}

/// Checks GrowSeeds schemes mechanism
TEST_CASE("GrowSeeds Schemes")
{
    const EstimationOptions estOpt;
    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
    // Create simple graph A -> B -> C
    Graph graph;
    NameOnlyNode* nodeA = graph.CreateAndAddNode<NameOnlyNode>("a");
    NameOnlyNode* nodeB = graph.CreateAndAddNode<NameOnlyNode>("b");
    NameOnlyNode* nodeC = graph.CreateAndAddNode<NameOnlyNode>("c");
    graph.Connect(nodeA, nodeB, 0);
    graph.Connect(nodeB, nodeC, 0);

    // Generate some plans for each node
    Plan planASram;
    ConfigurePlan(OutputPlanConfigurator(planASram, nodeA), Lifetime::Cascade, Location::Sram,
                  CascadingBufferFormat::NHWCB, TensorShape(), TensorShape{ 1, 2, 3, 4 }, TraversalOrder::Xyz, 4 * 16,
                  QuantizationInfo());

    Plan planADram;
    ConfigurePlan(OutputPlanConfigurator(planADram, nodeA), Lifetime::Cascade, Location::Dram,
                  CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(), TraversalOrder::Xyz, 4 * 16,
                  QuantizationInfo());

    Plan planBSram;
    ConfigurePlan(InputPlanConfigurator(planBSram, nodeB), Lifetime::Cascade, Location::Sram,
                  CascadingBufferFormat::NHWCB, TensorShape(), TensorShape{ 5, 6, 7, 8 }, TraversalOrder::Xyz, 4 * 16,
                  QuantizationInfo());
    ConfigurePlan(OutputPlanConfigurator(planBSram, nodeB), Lifetime::Cascade, Location::Sram,
                  CascadingBufferFormat::NHWCB, TensorShape(), TensorShape{ 5, 6, 7, 8 }, TraversalOrder::Xyz, 4 * 16,
                  QuantizationInfo());

    Plan planBDram;
    ConfigurePlan(InputPlanConfigurator(planBDram, nodeB), Lifetime::Cascade, Location::Dram,
                  CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(), TraversalOrder::Xyz, 4 * 16,
                  QuantizationInfo());
    ConfigurePlan(OutputPlanConfigurator(planBDram, nodeB), Lifetime::Cascade, Location::Dram,
                  CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(), TraversalOrder::Xyz, 4 * 16,
                  QuantizationInfo());

    Plan planCSram;
    // Note that m_SizeInBytes is different to planBSram, these plans are not mergeable
    ConfigurePlan(InputPlanConfigurator(planCSram, nodeC), Lifetime::Cascade, Location::Sram,
                  CascadingBufferFormat::NHWCB, TensorShape(), TensorShape{ 5, 6, 7, 8 }, TraversalOrder::Xyz,
                  2 * 4 * 16, QuantizationInfo());

    Plan planCDram;
    ConfigurePlan(InputPlanConfigurator(planCDram, nodeC), Lifetime::Cascade, Location::Dram,
                  CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(), TraversalOrder::Xyz, 4 * 16,
                  QuantizationInfo());

    GraphOfParts gOfParts;
    Parts& parts = gOfParts.m_Parts;
    // Add nodeA and plans to partA
    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_SubGraph.push_back(nodeA);
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planASram)));
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planADram)));

    // Add nodeB and plans to partB
    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_SubGraph.push_back(nodeB);
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planBSram)));
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planBDram)));

    // Add nodeC and plans to partC
    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_SubGraph.push_back(nodeC);
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planCSram)));
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planCDram)));

    Metadata metadata = CreateMetadata(gOfParts, hwCaps);

    const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
    Combinations combs              = CreateSeeds(gOfParts, metadata, caps);
    // All plan are compatible, the total number of seeds is the product of the number of plans
    REQUIRE(combs.size() == 4U);

    // Get where it is with merging parts
    size_t maxScore = 0U;
    for (const auto& c : combs)
    {
        if (c.m_Scratch.m_Score > maxScore)
        {
            maxScore = c.m_Scratch.m_Score;
        }
    }

    GrownSeeds res = GrowSeeds(combs, gOfParts, metadata, caps, GrowScheme::MergeOnly);
    // B and C cannot be merged
    REQUIRE(res.m_Combinations.size() == 0U);

    // C output data need to go to Dram
    res = GrowSeeds(combs, gOfParts, metadata, caps, GrowScheme::DramOnly);
    REQUIRE(res.m_Combinations.size() == 8U);

    // Check that nothing has been merged
    for (const auto& c : res.m_Combinations)
    {
        REQUIRE(c.m_Scratch.m_Score <= maxScore);
    }
}

/// Checks that CreateMedata correctly populates the metadata structure.
TEST_CASE("GrowSeeds Of Graph With Branches")
{
    const EstimationOptions estOpt;
    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
    /* Create graph:

                  C
               `/
          A - B
                \
                  D

    */
    Graph graph;
    NameOnlyNode* nodeA = graph.CreateAndAddNode<NameOnlyNode>("a");
    NameOnlyNode* nodeB = graph.CreateAndAddNode<NameOnlyNode>("b");
    NameOnlyNode* nodeC = graph.CreateAndAddNode<NameOnlyNode>("c");
    NameOnlyNode* nodeD = graph.CreateAndAddNode<NameOnlyNode>("d");
    graph.Connect(nodeA, nodeB, 0);
    graph.Connect(nodeB, nodeC, 0);
    graph.Connect(nodeB, nodeD, 0);

    // Generate some plans for each node

    // Node A
    Buffer planAOutputSram(Lifetime::Cascade, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(),
                           TensorShape{ 1, 2, 3, 4 }, TraversalOrder::Xyz, 0, QuantizationInfo());

    Plan planASram({}, { { &planAOutputSram, nodeA } });

    Buffer planAOutputDram(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(),
                           TraversalOrder::Xyz, 0, QuantizationInfo());

    Plan planADram({}, { { &planAOutputDram, nodeA } });

    // Node B
    Buffer planBInputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(),
                          TensorShape{ 1, 2, 3, 4 }, TraversalOrder::Xyz, 0, QuantizationInfo());

    Buffer planBOutputSramToC(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(),
                              TensorShape{ 5, 6, 7, 8 }, TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer planBOutputSramToD(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(),
                              TensorShape{ 5, 6, 7, 8 }, TraversalOrder::Xyz, 0, QuantizationInfo());

    Plan planBSram({ { &planBInputSram, nodeB->GetInput(0) } },
                   { { &planBOutputSramToC, nodeB }, { &planBOutputSramToD, nodeB } });

    Buffer planBInputDram(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(),
                          TraversalOrder::Xyz, 0, QuantizationInfo());

    Buffer planBOutputDramToC(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB, TensorShape(),
                              TensorShape(), TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer planBOutputDramToD(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB, TensorShape(),
                              TensorShape(), TraversalOrder::Xyz, 0, QuantizationInfo());

    Plan planBDram({ { &planBInputDram, nodeB->GetInput(0) } },
                   { { &planBOutputDramToC, nodeB }, { &planBOutputDramToD, nodeB } });

    // Node C
    Buffer planCInputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(),
                          TensorShape{ 5, 6, 7, 8 }, TraversalOrder::Xyz, 0, QuantizationInfo());

    Buffer planCOutputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(),
                           TensorShape{ 2, 2, 2, 2 }, TraversalOrder::Xyz, 0, QuantizationInfo());

    Plan planCSram({ { &planCInputSram, nodeC->GetInput(0) } }, { { &planCOutputSram, nodeC } });

    Buffer planCInputDram(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(),
                          TraversalOrder::Xyz, 0, QuantizationInfo());

    Buffer planCOutputDram(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(),
                           TraversalOrder::Xyz, 0, QuantizationInfo());

    Plan planCDram({ { &planCInputDram, nodeC->GetInput(0) } }, { { &planCOutputDram, nodeC } });

    // Node D
    Buffer planDInputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(),
                          TensorShape{ 1, 2, 3, 4 }, TraversalOrder::Xyz, 0, QuantizationInfo());

    Buffer planDOutputSram(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(),
                           TensorShape{ 5, 6, 7, 8 }, TraversalOrder::Xyz, 0, QuantizationInfo());

    Plan planDSram({ { &planDInputSram, nodeD->GetInput(0) } }, { { &planDOutputSram, nodeD } });

    Buffer planDInputDram(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(),
                          TraversalOrder::Xyz, 0, QuantizationInfo());

    Buffer planDOutputDram(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(),
                           TraversalOrder::Xyz, 0, QuantizationInfo());

    Plan planDDram({ { &planDInputDram, nodeD->GetInput(0) } }, { { &planDOutputDram, nodeD } });

    GraphOfParts gOfParts;
    Parts& parts = gOfParts.m_Parts;

    // Topological sort:  A, B, C, D
    // Add nodeA and plans to partA
    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_SubGraph.push_back(nodeA);
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planASram)));
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planADram)));

    // Add nodeB and plans to partB
    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_SubGraph.push_back(nodeB);
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planBSram)));
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planBDram)));

    // Add nodeC and plans to partC
    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_SubGraph.push_back(nodeC);
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planCSram)));
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planCDram)));

    // Add nodeC and plans to partD
    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_SubGraph.push_back(nodeD);
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planDSram)));
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planDDram)));

    Metadata metadata = CreateMetadata(gOfParts, hwCaps);

    // Number of parts in the metadata
    REQUIRE(metadata.size() == 4U);

    Combinations combs = CreateSeeds(gOfParts, metadata, hwCaps);
    // All plan are compatible, the total number of seeds is the product of the number of plans (plus "Back to Dram" plans)
    REQUIRE(combs.size() == 5U);

    GrownSeeds res = GrowSeeds(combs, gOfParts, metadata, hwCaps, GrowScheme::Default);
    REQUIRE(res.m_Combinations.size() == 7U);
    REQUIRE(res.m_Terminated == false);
    res = GrowSeeds(res.m_Combinations, gOfParts, metadata, hwCaps, GrowScheme::Default);
    REQUIRE(res.m_Combinations.size() == 14U);
    REQUIRE(res.m_Terminated == false);
    res = GrowSeeds(res.m_Combinations, gOfParts, metadata, hwCaps, GrowScheme::Default);
    REQUIRE(res.m_Combinations.size() == 14U);
    REQUIRE(res.m_Terminated == false);
    res = GrowSeeds(res.m_Combinations, gOfParts, metadata, hwCaps, GrowScheme::Default);
    REQUIRE(res.m_Combinations.size() == 14U);
    REQUIRE(res.m_Terminated == false);
    res = GrowSeeds(res.m_Combinations, gOfParts, metadata, hwCaps, GrowScheme::Default);
    REQUIRE(res.m_Combinations.size() == 14U);
    REQUIRE(res.m_Terminated == true);

    size_t score = 0;

    for (size_t i = 0; i < res.m_Combinations.size(); ++i)
    {
        INFO("Combination number is: " << i);
        // All the combinations are complete
        REQUIRE(res.m_Combinations.at(i).m_Elems.size() == 4U);
        // Check that only two combinations can merge
        score += res.m_Combinations.at(i).m_Scratch.m_Score;
        REQUIRE(score <= 2U);
    }
}

/// Checks that Combine generates all the combinations
TEST_CASE("Combine Simple")
{
    const EstimationOptions estOpt;
    CompilationOptions compOpt;
    compOpt.m_DisableWinograd         = GENERATE(false, true);
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
    // Create simple graph A -> B -> C
    Graph graph;
    NameOnlyNode* nodeA = graph.CreateAndAddNode<NameOnlyNode>("a");
    NameOnlyNode* nodeB = graph.CreateAndAddNode<NameOnlyNode>("b");
    NameOnlyNode* nodeC = graph.CreateAndAddNode<NameOnlyNode>("c");
    graph.Connect(nodeA, nodeB, 0);
    graph.Connect(nodeB, nodeC, 0);

    // Generate some plans for each node
    Plan planASram;
    ConfigurePlan(OutputPlanConfigurator(planASram, nodeA), Lifetime::Atomic, Location::Sram,
                  CascadingBufferFormat::NHWCB, TensorShape(), TensorShape{ 1, 2, 3, 4 }, TraversalOrder::Xyz, 0,
                  QuantizationInfo());

    Plan planADram;
    ConfigurePlan(OutputPlanConfigurator(planADram, nodeA), Lifetime::Atomic, Location::Dram,
                  CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(), TraversalOrder::Xyz, 0,
                  QuantizationInfo());

    Plan planBSram;
    ConfigurePlan(InputPlanConfigurator(planBSram, nodeB), Lifetime::Atomic, Location::Sram,
                  CascadingBufferFormat::NHWCB, TensorShape(), TensorShape{ 5, 6, 7, 8 }, TraversalOrder::Xyz, 0,
                  QuantizationInfo());
    ConfigurePlan(OutputPlanConfigurator(planBSram, nodeB), Lifetime::Atomic, Location::Sram,
                  CascadingBufferFormat::NHWCB, TensorShape(), TensorShape{ 5, 6, 7, 8 }, TraversalOrder::Xyz, 0,
                  QuantizationInfo());

    Plan planBDram;
    ConfigurePlan(InputPlanConfigurator(planBDram, nodeB), Lifetime::Atomic, Location::Dram,
                  CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(), TraversalOrder::Xyz, 0,
                  QuantizationInfo());
    ConfigurePlan(OutputPlanConfigurator(planBDram, nodeB), Lifetime::Atomic, Location::Dram,
                  CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(), TraversalOrder::Xyz, 0,
                  QuantizationInfo());

    Plan planCSram;
    ConfigurePlan(InputPlanConfigurator(planCSram, nodeC), Lifetime::Atomic, Location::Sram,
                  CascadingBufferFormat::NHWCB, TensorShape(), TensorShape{ 5, 6, 7, 8 }, TraversalOrder::Xyz, 0,
                  QuantizationInfo());

    Plan planCDram;
    ConfigurePlan(InputPlanConfigurator(planCDram, nodeC), Lifetime::Atomic, Location::Dram,
                  CascadingBufferFormat::NHWCB, TensorShape(), TensorShape(), TraversalOrder::Xyz, 0,
                  QuantizationInfo());

    GraphOfParts gOfParts;
    Parts& parts = gOfParts.m_Parts;

    // Add nodeA and plans to partA
    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_SubGraph.push_back(nodeA);
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planASram)));
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planADram)));
    (*(parts.back())).m_NumInvalidPlans = 1;    // Disable the "avoid dram" mechanism.

    // Add nodeB and plans to partB
    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_SubGraph.push_back(nodeB);
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planBSram)));
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planBDram)));

    // Add nodeC and plans to partC
    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_SubGraph.push_back(nodeC);
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planCSram)));
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planCDram)));

    CompilationOptions compilationOptions;
    compilationOptions.m_DebugInfo.m_DumpDebugFiles = CompilationOptions::DebugLevel::None;
    DebuggingContext debuggingCtxt(&compilationOptions.m_DebugInfo);
    SetDebuggingContext(debuggingCtxt);
    Cascading cascading(estOpt, compOpt, hwCaps);
    Combinations combs = cascading.Combine(gOfParts);

    REQUIRE(combs.size() == 12U);

    size_t score = 0;
    for (size_t i = 0; i < combs.size(); ++i)
    {
        INFO("Combination number is: " << i);
        // All the combinations are complete
        REQUIRE(combs.at(i).m_Elems.size() == 3U);
        // All the combinations have the correct sequence of parts
        for (size_t j = 0; j < parts.size(); ++j)
        {
            REQUIRE(combs.at(i).m_Elems.at(j).m_PartId == j);
        }
        score += combs.at(i).m_Scratch.m_Score;
    }
    // Check that there is at least a merge
    REQUIRE(score > 0);
}

/// Checks that Combine back to Dram
TEST_CASE("Combine Simple back to dram")
{
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
    const EstimationOptions estOpt;
    CompilationOptions compOpt;
    // Create simple graph A -> B
    Graph graph;
    NameOnlyNode* nodeA = graph.CreateAndAddNode<NameOnlyNode>("a");
    NameOnlyNode* nodeB = graph.CreateAndAddNode<NameOnlyNode>("b");
    graph.Connect(nodeA, nodeB, 0);

    Plan planASram;
    ConfigurePlan(OutputPlanConfigurator(planASram, nodeA), Lifetime::Cascade, Location::Sram,
                  CascadingBufferFormat::NHWCB, TensorShape(), TensorShape{ 1, 2, 3, 4 }, TraversalOrder::Xyz,
                  2 * 1024 * 16, QuantizationInfo());
    Buffer planAWeightsSram(Lifetime::Cascade, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(),
                            TensorShape{ 1, 2, 3, 4 }, TraversalOrder::Xyz, 31 * 1024 * 16, QuantizationInfo());
    planASram.m_OpGraph.AddBuffer(std::make_unique<Buffer>(std::move(planAWeightsSram)));

    Plan planBSram;
    ConfigurePlan(InputPlanConfigurator(planBSram, nodeB), Lifetime::Cascade, Location::Sram,
                  CascadingBufferFormat::NHWCB, TensorShape(), TensorShape{ 1, 2, 3, 4 }, TraversalOrder::Xyz,
                  2 * 1024 * 16, QuantizationInfo());
    Buffer planBWeightsSram(Lifetime::Cascade, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(),
                            TensorShape{ 1, 2, 3, 4 }, TraversalOrder::Xyz, 61 * 1024 * 16, QuantizationInfo());
    planBSram.m_OpGraph.AddBuffer(std::make_unique<Buffer>(std::move(planBWeightsSram)));

    GraphOfParts gOfParts;
    Parts& parts = gOfParts.m_Parts;

    // Add nodeA and plans to partA
    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_SubGraph.push_back(nodeA);
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planASram)));

    // Add nodeB and plans to partB
    parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    (*(parts.back())).m_SubGraph.push_back(nodeB);
    (*(parts.back())).m_Plans.push_back(std::make_unique<Plan>(std::move(planBSram)));
    (*(parts.back())).m_NumInvalidPlans = 1;    // Plan B does not actually fit in Sram

    compOpt.m_DebugInfo.m_DumpDebugFiles = CompilationOptions::DebugLevel::None;
    DebuggingContext debuggingCtxt(&compOpt.m_DebugInfo);
    SetDebuggingContext(debuggingCtxt);
    Cascading cascading(estOpt, compOpt, hwCaps);
    Combinations combs = cascading.Combine(gOfParts);

    REQUIRE(combs.size() == 3U);
    for (size_t i = 0; i < combs.size(); ++i)
    {
        INFO("Combination number is: " << i);
        // Parts cannot be cascaded since Lifetime::Cascade data does not fit in Sram
        REQUIRE(combs.at(0).m_Scratch.m_Score == 0);
    }
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
TEST_CASE("GetOpGraphForCombination")
{
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

    // Part consisting of node A
    parts.m_Parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    parts.m_Parts.back()->m_SubGraph.push_back(nodeA);
    std::unique_ptr<Plan> planA = std::make_unique<Plan>();
    planA->m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB,
                                                        TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                        TraversalOrder::Xyz, 0, QuantizationInfo()));
    planA->m_OpGraph.GetBuffers().back()->m_DebugTag = "InputDram";
    planA->m_OutputMappings                          = { { planA->m_OpGraph.GetBuffers()[0], nodeA } };
    parts.m_Parts.back()->m_Plans.push_back(std::move(planA));

    // Glue between A and B
    Glue glueA_BC;
    glueA_BC.m_Graph.AddOp(std::make_unique<DmaOp>());
    glueA_BC.m_Graph.GetOps()[0]->m_DebugTag = "InputDma";
    glueA_BC.m_InputSlot                     = { glueA_BC.m_Graph.GetOps()[0], 0 };
    glueA_BC.m_Output                        = glueA_BC.m_Graph.GetOps()[0];

    // Part consisting of node B
    parts.m_Parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    parts.m_Parts.back()->m_SubGraph.push_back(nodeB);
    std::unique_ptr<Plan> planB = std::make_unique<Plan>();
    planB->m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                        TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                        TraversalOrder::Xyz, 4, QuantizationInfo()));
    planB->m_OpGraph.GetBuffers().back()->m_DebugTag = "InputSram1";
    planB->m_InputMappings                           = { { planB->m_OpGraph.GetBuffers()[0], nodeB->GetInput(0) } };
    planB->m_OutputMappings                          = { { planB->m_OpGraph.GetBuffers()[0], nodeB } };
    parts.m_Parts.back()->m_Plans.push_back(std::move(planB));

    // Part consisting of node C
    parts.m_Parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    parts.m_Parts.back()->m_SubGraph.push_back(nodeC);
    std::unique_ptr<Plan> planC = std::make_unique<Plan>();
    planC->m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                        TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                        TraversalOrder::Xyz, 4, QuantizationInfo()));
    planC->m_OpGraph.GetBuffers().back()->m_DebugTag = "InputSram2";
    planC->m_InputMappings                           = { { planC->m_OpGraph.GetBuffers()[0], nodeC->GetInput(0) } };
    planC->m_OutputMappings                          = { { planC->m_OpGraph.GetBuffers()[0], nodeC } };
    parts.m_Parts.back()->m_Plans.push_back(std::move(planC));

    // Part consisting of nodes D and E
    parts.m_Parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
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
    parts.m_Parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
    parts.m_Parts.back()->m_SubGraph.push_back(nodeF);
    std::unique_ptr<Plan> planF = std::make_unique<Plan>();
    planF->m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB,
                                                        TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                        TraversalOrder::Xyz, 0, QuantizationInfo()));
    planF->m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputDram1";
    planF->m_InputMappings                           = { { planF->m_OpGraph.GetBuffers()[0], nodeF->GetInput(0) } };
    parts.m_Parts.back()->m_Plans.push_back(std::move(planF));

    // Part consisting of node G
    parts.m_Parts.push_back(std::make_unique<Part>(estOpt, compOpt, hwCaps));
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
    Elem elemA  = { 0, 0, { { nodeB->GetInput(0), { 0, &glueA_BC } } } };
    Elem elemB  = { 1, 0, {} };
    Elem elemC  = { 2, 0, {} };
    Elem elemDE = { 3,
                    0,
                    { { nodeF->GetInput(0), { 0, &glueD_F } },
                      { nodeG->GetInput(0), { 0, &glueD_G } },
                      { nodeG->GetInput(1), { 0, &glueE_G } } } };
    Elem elemF  = { 4, 0, {} };
    Elem elemG  = { 5, 0, {} };
    comb.m_Elems.push_back(elemA);
    comb.m_Elems.push_back(elemB);
    comb.m_Elems.push_back(elemC);
    comb.m_Elems.push_back(elemDE);
    comb.m_Elems.push_back(elemF);
    comb.m_Elems.push_back(elemG);

    // For easier debugging of this test (and so that you can see the pretty graph!), dump the input to a file
    bool dumpToFile = false;
    if (dumpToFile)
    {
        std::ofstream stream("GetOpGraphForCombination Input.dot");
        SaveCombinationToDot(comb, parts, stream, DetailLevel::High);
    }

    // Call function under test
    OpGraph combOpGraph = GetOpGraphForCombination(comb, parts);

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
