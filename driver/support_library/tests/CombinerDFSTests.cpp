//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/DebuggingContext.hpp"
#include "../src/GraphNodes.hpp"
#include "../src/cascading/Cascading.hpp"
#include "../src/cascading/CombinerDFS.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>

#include <fstream>

namespace dfs = ethosn::support_library::depth_first_search;

using namespace ethosn::support_library;
using namespace ethosn::command_stream;

namespace
{

void AddNodesToPart(GraphOfParts& gOfParts,
                    std::vector<Node*> nodes,
                    const EstimationOptions& estOpt,
                    const CompilationOptions& compOpt,
                    const HardwareCapabilities& hwCaps)
{
    gOfParts.m_Parts.push_back(std::make_unique<Part>(gOfParts.GeneratePartId(), estOpt, compOpt, hwCaps));
    for (Node* node : nodes)
    {
        (*(gOfParts.m_Parts.back())).m_SubGraph.push_back(node);
    }
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
               ethosn::support_library::DataType::UINT8_QUANTIZED,
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

TEST_CASE("IsPartSiso", "[CombinerDFS]")
{
    Graph graph;
    /* Create graph:

                  D
                 /
        A - B - C
                 \
                  E

    */
    NameOnlyNode* nodeA = graph.CreateAndAddNode<NameOnlyNode>("a");
    NameOnlyNode* nodeB = graph.CreateAndAddNode<NameOnlyNode>("b");
    NameOnlyNode* nodeC = graph.CreateAndAddNode<NameOnlyNode>("c");
    NameOnlyNode* nodeD = graph.CreateAndAddNode<NameOnlyNode>("d");
    NameOnlyNode* nodeE = graph.CreateAndAddNode<NameOnlyNode>("e");

    graph.Connect(nodeA, nodeB, 0);
    graph.Connect(nodeB, nodeC, 0);
    graph.Connect(nodeC, nodeD, 0);
    graph.Connect(nodeC, nodeE, 0);

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    GraphOfParts gOfParts;
    AddNodesToPart(gOfParts, { nodeA }, estOpt, compOpt, hwCaps);
    AddNodesToPart(gOfParts, { nodeB }, estOpt, compOpt, hwCaps);
    AddNodesToPart(gOfParts, { nodeC }, estOpt, compOpt, hwCaps);
    AddNodesToPart(gOfParts, { nodeD }, estOpt, compOpt, hwCaps);
    AddNodesToPart(gOfParts, { nodeE }, estOpt, compOpt, hwCaps);

    size_t count = 0;
    for (auto&& p : gOfParts.m_Parts)
    {
        REQUIRE(p->m_PartId == count);
        ++count;
    }

    dfs::Combiner combiner(gOfParts, hwCaps, estOpt);

    REQUIRE(combiner.IsPartSiso(*gOfParts.m_Parts.at(0).get()) == false);
    REQUIRE(combiner.IsPartSiso(*gOfParts.m_Parts.at(1).get()) == true);
    REQUIRE(combiner.IsPartSiso(*gOfParts.m_Parts.at(2).get()) == false);
    REQUIRE(combiner.IsPartSiso(*gOfParts.m_Parts.at(3).get()) == false);
    REQUIRE(combiner.IsPartSiso(*gOfParts.m_Parts.at(4).get()) == false);

    // All parts have been cached
    REQUIRE(combiner.m_InOutMap.at(static_cast<uint32_t>(dfs::InOutFormat::SISO)).size() == gOfParts.m_Parts.size());
    // Other maps have been updated
    REQUIRE(combiner.m_InOutMap.at(static_cast<uint32_t>(dfs::InOutFormat::SIMO)).size() == 1);
    REQUIRE(combiner.m_InOutMap.at(static_cast<uint32_t>(dfs::InOutFormat::SIMO))[1] == false);
    REQUIRE(combiner.m_InOutMap.at(static_cast<uint32_t>(dfs::InOutFormat::MISO)).size() == 1);
    REQUIRE(combiner.m_InOutMap.at(static_cast<uint32_t>(dfs::InOutFormat::MISO))[1] == false);
    REQUIRE(combiner.m_InOutMap.at(static_cast<uint32_t>(dfs::InOutFormat::MIMO)).size() == 1);
    REQUIRE(combiner.m_InOutMap.at(static_cast<uint32_t>(dfs::InOutFormat::MIMO))[1] == false);
}

TEST_CASE("IsPartSimo", "[CombinerDFS]")
{
    Graph graph;
    /* Create graph:

                  D
                 /
        A - B - C
                 \
                  E

    */
    NameOnlyNode* nodeA = graph.CreateAndAddNode<NameOnlyNode>("a");
    NameOnlyNode* nodeB = graph.CreateAndAddNode<NameOnlyNode>("b");
    NameOnlyNode* nodeC = graph.CreateAndAddNode<NameOnlyNode>("c");
    NameOnlyNode* nodeD = graph.CreateAndAddNode<NameOnlyNode>("d");
    NameOnlyNode* nodeE = graph.CreateAndAddNode<NameOnlyNode>("e");

    graph.Connect(nodeA, nodeB, 0);
    graph.Connect(nodeB, nodeC, 0);
    graph.Connect(nodeC, nodeD, 0);
    graph.Connect(nodeC, nodeE, 0);

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    GraphOfParts gOfParts;
    AddNodesToPart(gOfParts, { nodeA }, estOpt, compOpt, hwCaps);
    AddNodesToPart(gOfParts, { nodeB }, estOpt, compOpt, hwCaps);
    AddNodesToPart(gOfParts, { nodeC }, estOpt, compOpt, hwCaps);
    AddNodesToPart(gOfParts, { nodeD }, estOpt, compOpt, hwCaps);
    AddNodesToPart(gOfParts, { nodeE }, estOpt, compOpt, hwCaps);

    size_t count = 0;
    for (auto&& p : gOfParts.m_Parts)
    {
        REQUIRE(p->m_PartId == count);
        ++count;
    }

    dfs::Combiner combiner(gOfParts, hwCaps, estOpt);

    REQUIRE(combiner.IsPartSimo(*gOfParts.m_Parts.at(0).get()) == false);
    REQUIRE(combiner.IsPartSimo(*gOfParts.m_Parts.at(1).get()) == false);
    REQUIRE(combiner.IsPartSimo(*gOfParts.m_Parts.at(2).get()) == true);
    REQUIRE(combiner.IsPartSimo(*gOfParts.m_Parts.at(3).get()) == false);
    REQUIRE(combiner.IsPartSimo(*gOfParts.m_Parts.at(4).get()) == false);

    // All parts have been cached
    REQUIRE(combiner.m_InOutMap.at(static_cast<uint32_t>(dfs::InOutFormat::SIMO)).size() == gOfParts.m_Parts.size());
    REQUIRE(combiner.m_InOutMap.at(static_cast<uint32_t>(dfs::InOutFormat::SIMO))[2] == true);
    // Other maps have been updated
    REQUIRE(combiner.m_InOutMap.at(static_cast<uint32_t>(dfs::InOutFormat::SISO)).size() == 1);
    REQUIRE(combiner.m_InOutMap.at(static_cast<uint32_t>(dfs::InOutFormat::SISO))[1] == false);
    REQUIRE(combiner.m_InOutMap.at(static_cast<uint32_t>(dfs::InOutFormat::MISO)).size() == 1);
    REQUIRE(combiner.m_InOutMap.at(static_cast<uint32_t>(dfs::InOutFormat::MISO))[2] == false);
    REQUIRE(combiner.m_InOutMap.at(static_cast<uint32_t>(dfs::InOutFormat::MIMO)).size() == 1);
    REQUIRE(combiner.m_InOutMap.at(static_cast<uint32_t>(dfs::InOutFormat::MIMO))[2] == false);
}

TEST_CASE("IsPartMiso", "[CombinerDFS]")
{
    Graph graph;
    /* Create graph:

      A
       \
        C - D
       /
      B

    */
    NameOnlyNode* nodeA = graph.CreateAndAddNode<NameOnlyNode>("a");
    NameOnlyNode* nodeB = graph.CreateAndAddNode<NameOnlyNode>("b");
    NameOnlyNode* nodeC = graph.CreateAndAddNode<NameOnlyNode>("c");
    NameOnlyNode* nodeD = graph.CreateAndAddNode<NameOnlyNode>("d");

    graph.Connect(nodeA, nodeC, 0);
    graph.Connect(nodeB, nodeC, 0);
    graph.Connect(nodeC, nodeD, 0);

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    GraphOfParts gOfParts;
    AddNodesToPart(gOfParts, { nodeA }, estOpt, compOpt, hwCaps);
    AddNodesToPart(gOfParts, { nodeB }, estOpt, compOpt, hwCaps);
    AddNodesToPart(gOfParts, { nodeC }, estOpt, compOpt, hwCaps);
    AddNodesToPart(gOfParts, { nodeD }, estOpt, compOpt, hwCaps);

    size_t count = 0;
    for (auto&& p : gOfParts.m_Parts)
    {
        REQUIRE(p->m_PartId == count);
        ++count;
    }

    dfs::Combiner combiner(gOfParts, hwCaps, estOpt);

    REQUIRE(combiner.IsPartMiso(*gOfParts.m_Parts.at(0).get()) == false);
    REQUIRE(combiner.IsPartMiso(*gOfParts.m_Parts.at(1).get()) == false);
    REQUIRE(combiner.IsPartMiso(*gOfParts.m_Parts.at(2).get()) == true);
    REQUIRE(combiner.IsPartMiso(*gOfParts.m_Parts.at(3).get()) == false);

    // All parts have been cached
    REQUIRE(combiner.m_InOutMap.at(static_cast<uint32_t>(dfs::InOutFormat::MISO)).size() == gOfParts.m_Parts.size());
    REQUIRE(combiner.m_InOutMap.at(static_cast<uint32_t>(dfs::InOutFormat::MISO))[2] == true);
    // Other maps have been updated
    REQUIRE(combiner.m_InOutMap.at(static_cast<uint32_t>(dfs::InOutFormat::SISO)).size() == 1);
    REQUIRE(combiner.m_InOutMap.at(static_cast<uint32_t>(dfs::InOutFormat::SISO))[2] == false);
    REQUIRE(combiner.m_InOutMap.at(static_cast<uint32_t>(dfs::InOutFormat::SIMO)).size() == 1);
    REQUIRE(combiner.m_InOutMap.at(static_cast<uint32_t>(dfs::InOutFormat::SIMO))[2] == false);
    REQUIRE(combiner.m_InOutMap.at(static_cast<uint32_t>(dfs::InOutFormat::MIMO)).size() == 1);
    REQUIRE(combiner.m_InOutMap.at(static_cast<uint32_t>(dfs::InOutFormat::MIMO))[2] == false);
}

TEST_CASE("IsPartMimo", "[CombinerDFS]")
{
    Graph graph;
    /* Create graph:

      A   D
       \ /
        C
       / \
      B   E

    */
    NameOnlyNode* nodeA = graph.CreateAndAddNode<NameOnlyNode>("a");
    NameOnlyNode* nodeB = graph.CreateAndAddNode<NameOnlyNode>("b");
    NameOnlyNode* nodeC = graph.CreateAndAddNode<NameOnlyNode>("c");
    NameOnlyNode* nodeD = graph.CreateAndAddNode<NameOnlyNode>("d");
    NameOnlyNode* nodeE = graph.CreateAndAddNode<NameOnlyNode>("e");

    graph.Connect(nodeA, nodeC, 0);
    graph.Connect(nodeB, nodeC, 0);
    graph.Connect(nodeC, nodeD, 0);
    graph.Connect(nodeC, nodeE, 0);

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    GraphOfParts gOfParts;
    AddNodesToPart(gOfParts, { nodeA }, estOpt, compOpt, hwCaps);
    AddNodesToPart(gOfParts, { nodeB }, estOpt, compOpt, hwCaps);
    AddNodesToPart(gOfParts, { nodeC }, estOpt, compOpt, hwCaps);
    AddNodesToPart(gOfParts, { nodeD }, estOpt, compOpt, hwCaps);
    AddNodesToPart(gOfParts, { nodeE }, estOpt, compOpt, hwCaps);

    size_t count = 0;
    for (auto&& p : gOfParts.m_Parts)
    {
        REQUIRE(p->m_PartId == count);
        ++count;
    }

    dfs::Combiner combiner(gOfParts, hwCaps, estOpt);

    REQUIRE(combiner.IsPartMimo(*gOfParts.m_Parts.at(0).get()) == false);
    REQUIRE(combiner.IsPartMimo(*gOfParts.m_Parts.at(1).get()) == false);
    REQUIRE(combiner.IsPartMimo(*gOfParts.m_Parts.at(2).get()) == true);
    REQUIRE(combiner.IsPartMimo(*gOfParts.m_Parts.at(3).get()) == false);
    REQUIRE(combiner.IsPartMimo(*gOfParts.m_Parts.at(4).get()) == false);

    // All parts have been cached
    REQUIRE(combiner.m_InOutMap.at(static_cast<uint32_t>(dfs::InOutFormat::MIMO)).size() == gOfParts.m_Parts.size());
    REQUIRE(combiner.m_InOutMap.at(static_cast<uint32_t>(dfs::InOutFormat::MIMO))[2] == true);
    // Other maps have been updated
    REQUIRE(combiner.m_InOutMap.at(static_cast<uint32_t>(dfs::InOutFormat::SISO)).size() == 1);
    REQUIRE(combiner.m_InOutMap.at(static_cast<uint32_t>(dfs::InOutFormat::SISO))[2] == false);
    REQUIRE(combiner.m_InOutMap.at(static_cast<uint32_t>(dfs::InOutFormat::SIMO)).size() == 1);
    REQUIRE(combiner.m_InOutMap.at(static_cast<uint32_t>(dfs::InOutFormat::SIMO))[2] == false);
    REQUIRE(combiner.m_InOutMap.at(static_cast<uint32_t>(dfs::InOutFormat::MISO)).size() == 1);
    REQUIRE(combiner.m_InOutMap.at(static_cast<uint32_t>(dfs::InOutFormat::MISO))[2] == false);
}

TEST_CASE("IsPartInput", "[CombinerDFS]")
{
    Graph graph;
    /* Create graph:

      A   D
       \ /
        C
       / \
      B   E

    */
    NameOnlyNode* nodeA = graph.CreateAndAddNode<NameOnlyNode>("a");
    NameOnlyNode* nodeB = graph.CreateAndAddNode<NameOnlyNode>("b");
    NameOnlyNode* nodeC = graph.CreateAndAddNode<NameOnlyNode>("c");
    NameOnlyNode* nodeD = graph.CreateAndAddNode<NameOnlyNode>("d");
    NameOnlyNode* nodeE = graph.CreateAndAddNode<NameOnlyNode>("e");

    graph.Connect(nodeA, nodeC, 0);
    graph.Connect(nodeB, nodeC, 0);
    graph.Connect(nodeC, nodeD, 0);
    graph.Connect(nodeC, nodeE, 0);

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    GraphOfParts gOfParts;
    AddNodesToPart(gOfParts, { nodeA }, estOpt, compOpt, hwCaps);
    AddNodesToPart(gOfParts, { nodeB }, estOpt, compOpt, hwCaps);
    AddNodesToPart(gOfParts, { nodeC }, estOpt, compOpt, hwCaps);
    AddNodesToPart(gOfParts, { nodeD }, estOpt, compOpt, hwCaps);
    AddNodesToPart(gOfParts, { nodeE }, estOpt, compOpt, hwCaps);

    size_t count = 0;
    for (auto&& p : gOfParts.m_Parts)
    {
        REQUIRE(p->m_PartId == count);
        ++count;
    }

    dfs::Combiner combiner(gOfParts, hwCaps, estOpt);

    REQUIRE(combiner.IsPartInput(*gOfParts.m_Parts.at(0).get()) == true);
    REQUIRE(combiner.IsPartInput(*gOfParts.m_Parts.at(1).get()) == true);
    REQUIRE(combiner.IsPartInput(*gOfParts.m_Parts.at(2).get()) == false);
    REQUIRE(combiner.IsPartInput(*gOfParts.m_Parts.at(3).get()) == false);
    REQUIRE(combiner.IsPartInput(*gOfParts.m_Parts.at(4).get()) == false);
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

    GraphOfParts gOfParts;

    const EstimationOptions estOpt;
    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);

    // Part consisting of node A
    AddNodesToPart(gOfParts, { nodeA }, estOpt, compOpt, hwCaps);
    std::unique_ptr<Plan> planA = std::make_unique<Plan>();
    planA->m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB,
                                                        TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                        TraversalOrder::Xyz, 0, QuantizationInfo()));
    planA->m_OpGraph.GetBuffers().back()->m_DebugTag = "InputDram";
    planA->m_OutputMappings                          = { { planA->m_OpGraph.GetBuffers()[0], nodeA } };
    gOfParts.m_Parts.back()->m_Plans.push_back(std::move(planA));

    // Glue between A and B
    dfs::Glue glueA_BC;
    glueA_BC.m_Graph.AddOp(std::make_unique<DmaOp>());
    glueA_BC.m_Graph.GetOps()[0]->m_DebugTag = "InputDma";
    glueA_BC.m_InputSlot                     = { glueA_BC.m_Graph.GetOps()[0], 0 };
    glueA_BC.m_Output                        = glueA_BC.m_Graph.GetOps()[0];

    // Part consisting of node B
    AddNodesToPart(gOfParts, { nodeB }, estOpt, compOpt, hwCaps);
    std::unique_ptr<Plan> planB = std::make_unique<Plan>();
    planB->m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                        TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                        TraversalOrder::Xyz, 4, QuantizationInfo()));
    planB->m_OpGraph.GetBuffers().back()->m_DebugTag = "InputSram1";
    planB->m_InputMappings                           = { { planB->m_OpGraph.GetBuffers()[0], nodeB->GetInput(0) } };
    planB->m_OutputMappings                          = { { planB->m_OpGraph.GetBuffers()[0], nodeB } };
    gOfParts.m_Parts.back()->m_Plans.push_back(std::move(planB));

    // Part consisting of node C
    AddNodesToPart(gOfParts, { nodeC }, estOpt, compOpt, hwCaps);
    std::unique_ptr<Plan> planC = std::make_unique<Plan>();
    planC->m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                        TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                        TraversalOrder::Xyz, 4, QuantizationInfo()));
    planC->m_OpGraph.GetBuffers().back()->m_DebugTag = "InputSram2";
    planC->m_InputMappings                           = { { planC->m_OpGraph.GetBuffers()[0], nodeC->GetInput(0) } };
    planC->m_OutputMappings                          = { { planC->m_OpGraph.GetBuffers()[0], nodeC } };
    gOfParts.m_Parts.back()->m_Plans.push_back(std::move(planC));

    // Part consisting of nodes D and E
    AddNodesToPart(gOfParts, { nodeD, nodeE }, estOpt, compOpt, hwCaps);
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
    gOfParts.m_Parts.back()->m_Plans.push_back(std::move(planDE));

    // Glue between D and F
    dfs::Glue glueD_F;
    glueD_F.m_Graph.AddOp(std::make_unique<DmaOp>());
    glueD_F.m_Graph.GetOps()[0]->m_DebugTag = "OutputDma1";
    glueD_F.m_InputSlot                     = { glueD_F.m_Graph.GetOps()[0], 0 };
    glueD_F.m_Output                        = glueD_F.m_Graph.GetOps()[0];

    // Glue between D and G
    dfs::Glue glueD_G;
    glueD_G.m_Graph.AddOp(std::make_unique<DmaOp>());
    glueD_G.m_Graph.GetOps()[0]->m_DebugTag = "OutputDma2";
    glueD_G.m_InputSlot                     = { glueD_G.m_Graph.GetOps()[0], 0 };
    glueD_G.m_Output                        = glueD_G.m_Graph.GetOps()[0];

    // Glue between E and G
    dfs::Glue glueE_G;
    glueE_G.m_Graph.AddOp(std::make_unique<DmaOp>());
    glueE_G.m_Graph.GetOps()[0]->m_DebugTag = "OutputDma3";
    glueE_G.m_InputSlot                     = { glueE_G.m_Graph.GetOps()[0], 0 };
    glueE_G.m_Output                        = glueE_G.m_Graph.GetOps()[0];

    // Part consisting of node F
    AddNodesToPart(gOfParts, { nodeF }, estOpt, compOpt, hwCaps);
    std::unique_ptr<Plan> planF = std::make_unique<Plan>();
    planF->m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB,
                                                        TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                        TraversalOrder::Xyz, 0, QuantizationInfo()));
    planF->m_OpGraph.GetBuffers().back()->m_DebugTag = "OutputDram1";
    planF->m_InputMappings                           = { { planF->m_OpGraph.GetBuffers()[0], nodeF->GetInput(0) } };
    gOfParts.m_Parts.back()->m_Plans.push_back(std::move(planF));

    // Part consisting of node G
    AddNodesToPart(gOfParts, { nodeG }, estOpt, compOpt, hwCaps);
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
    gOfParts.m_Parts.back()->m_Plans.push_back(std::move(planG));

    // Create Combination with all the plans and glues
    dfs::Combination comb;

    //using Glues = std::map<const Edge*, const Glue*>;
    //PlanId m_PlanId;
    //Glues m_Glues

    dfs::Elem elemA  = { 0, { { nodeB->GetInput(0), { &glueA_BC } } } };
    dfs::Elem elemB  = { 0, {} };
    dfs::Elem elemC  = { 0, {} };
    dfs::Elem elemDE = { 0,
                         { { nodeF->GetInput(0), { &glueD_F } },
                           { nodeG->GetInput(0), { &glueD_G } },
                           { nodeG->GetInput(1), { &glueE_G } } } };
    dfs::Elem elemF  = { 0, {} };
    dfs::Elem elemG  = { 0, {} };
    comb.m_Elems.insert(std::make_pair(0, elemA));
    comb.m_Elems.insert(std::make_pair(1, elemB));
    comb.m_Elems.insert(std::make_pair(2, elemC));
    comb.m_Elems.insert(std::make_pair(3, elemDE));
    comb.m_Elems.insert(std::make_pair(4, elemF));
    comb.m_Elems.insert(std::make_pair(5, elemG));

    // Call function under test
    OpGraph combOpGraph = GetOpGraphForCombination(comb, gOfParts);

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
