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

void CheckPartId(const GraphOfParts& gOfParts)
{
    size_t count = 0;
    for (auto&& p : gOfParts.m_Parts)
    {
        REQUIRE(p->m_PartId == count);
        ++count;
    }
}

Part& GetPart(const GraphOfParts& gOfParts, const PartId partId)
{
    return *gOfParts.m_Parts.at(partId).get();
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
    // Create graph:
    //
    //          D
    //          |
    //  A - B - C
    //          |
    //          E
    //
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

    CheckPartId(gOfParts);

    dfs::Combiner combiner(gOfParts, hwCaps, estOpt);

    REQUIRE(combiner.IsPartSiso(GetPart(gOfParts, 0)) == false);
    REQUIRE(combiner.IsPartSiso(GetPart(gOfParts, 1)) == true);
    REQUIRE(combiner.IsPartSiso(GetPart(gOfParts, 2)) == false);
    REQUIRE(combiner.IsPartSiso(GetPart(gOfParts, 3)) == false);
    REQUIRE(combiner.IsPartSiso(GetPart(gOfParts, 4)) == false);
}

TEST_CASE("IsPartSimo", "[CombinerDFS]")
{
    Graph graph;
    // Create graph:
    //
    //          D
    //          |
    //  A - B - C
    //          |
    //          E
    //
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

    CheckPartId(gOfParts);

    dfs::Combiner combiner(gOfParts, hwCaps, estOpt);

    REQUIRE(combiner.IsPartSimo(GetPart(gOfParts, 0)) == false);
    REQUIRE(combiner.IsPartSimo(GetPart(gOfParts, 1)) == false);
    REQUIRE(combiner.IsPartSimo(GetPart(gOfParts, 2)) == true);
    REQUIRE(combiner.IsPartSimo(GetPart(gOfParts, 3)) == false);
    REQUIRE(combiner.IsPartSimo(GetPart(gOfParts, 4)) == false);
}

TEST_CASE("IsPartMiso", "[CombinerDFS]")
{
    Graph graph;
    // Create graph:
    //
    //  A
    //  |
    //  C - D
    //  |
    //  B
    //
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

    CheckPartId(gOfParts);

    dfs::Combiner combiner(gOfParts, hwCaps, estOpt);

    REQUIRE(combiner.IsPartMiso(GetPart(gOfParts, 0)) == false);
    REQUIRE(combiner.IsPartMiso(GetPart(gOfParts, 1)) == false);
    REQUIRE(combiner.IsPartMiso(GetPart(gOfParts, 2)) == true);
    REQUIRE(combiner.IsPartMiso(GetPart(gOfParts, 3)) == false);
}

TEST_CASE("IsPartMimo", "[CombinerDFS]")
{
    Graph graph;
    // Create graph:
    //
    //  A    E
    //  |    |
    //   - - C - D
    //       |
    //       B
    //
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

    CheckPartId(gOfParts);

    dfs::Combiner combiner(gOfParts, hwCaps, estOpt);

    REQUIRE(combiner.IsPartMimo(GetPart(gOfParts, 0)) == false);
    REQUIRE(combiner.IsPartMimo(GetPart(gOfParts, 1)) == false);
    REQUIRE(combiner.IsPartMimo(GetPart(gOfParts, 2)) == true);
    REQUIRE(combiner.IsPartMimo(GetPart(gOfParts, 3)) == false);
    REQUIRE(combiner.IsPartMimo(GetPart(gOfParts, 4)) == false);
}

TEST_CASE("IsPartInput and IsPartOutput", "[CombinerDFS]")
{
    Graph graph;
    // Create graph:
    //
    //  A    E
    //  |    |
    //   - - C - D
    //       |
    //       B
    //
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

    CheckPartId(gOfParts);

    dfs::Combiner combiner(gOfParts, hwCaps, estOpt);

    REQUIRE(combiner.IsPartInput(GetPart(gOfParts, 0)) == true);
    REQUIRE(combiner.IsPartOutput(GetPart(gOfParts, 0)) == false);

    REQUIRE(combiner.IsPartInput(GetPart(gOfParts, 1)) == true);
    REQUIRE(combiner.IsPartOutput(GetPart(gOfParts, 1)) == false);

    REQUIRE(combiner.IsPartInput(GetPart(gOfParts, 2)) == false);
    REQUIRE(combiner.IsPartOutput(GetPart(gOfParts, 2)) == false);

    REQUIRE(combiner.IsPartInput(GetPart(gOfParts, 3)) == false);
    REQUIRE(combiner.IsPartOutput(GetPart(gOfParts, 3)) == true);

    REQUIRE(combiner.IsPartInput(GetPart(gOfParts, 4)) == false);
    REQUIRE(combiner.IsPartOutput(GetPart(gOfParts, 4)) == true);
}

TEST_CASE("IsPartSo and IsPartMo", "[CombinerDFS]")
{
    Graph graph;
    // Create graph:
    //
    //  A    E
    //  |    |
    //   - - C - D
    //       |
    //       B - F
    //
    NameOnlyNode* nodeA = graph.CreateAndAddNode<NameOnlyNode>("a");
    NameOnlyNode* nodeB = graph.CreateAndAddNode<NameOnlyNode>("b");
    NameOnlyNode* nodeC = graph.CreateAndAddNode<NameOnlyNode>("c");
    NameOnlyNode* nodeD = graph.CreateAndAddNode<NameOnlyNode>("d");
    NameOnlyNode* nodeE = graph.CreateAndAddNode<NameOnlyNode>("e");
    NameOnlyNode* nodeF = graph.CreateAndAddNode<NameOnlyNode>("f");

    graph.Connect(nodeA, nodeC, 0);
    graph.Connect(nodeB, nodeC, 0);
    graph.Connect(nodeB, nodeF, 0);
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
    AddNodesToPart(gOfParts, { nodeF }, estOpt, compOpt, hwCaps);

    CheckPartId(gOfParts);

    dfs::Combiner combiner(gOfParts, hwCaps, estOpt);

    REQUIRE(combiner.IsPartSo(GetPart(gOfParts, 0)) == true);
    REQUIRE(combiner.IsPartMo(GetPart(gOfParts, 0)) == false);

    REQUIRE(combiner.IsPartSo(GetPart(gOfParts, 1)) == false);
    REQUIRE(combiner.IsPartMo(GetPart(gOfParts, 1)) == true);

    REQUIRE(combiner.IsPartSo(GetPart(gOfParts, 2)) == false);
    REQUIRE(combiner.IsPartMo(GetPart(gOfParts, 2)) == true);

    REQUIRE(combiner.IsPartSo(GetPart(gOfParts, 3)) == false);
    REQUIRE(combiner.IsPartMo(GetPart(gOfParts, 3)) == false);

    REQUIRE(combiner.IsPartSo(GetPart(gOfParts, 4)) == false);
    REQUIRE(combiner.IsPartMo(GetPart(gOfParts, 4)) == false);

    REQUIRE(combiner.IsPartSo(GetPart(gOfParts, 5)) == false);
    REQUIRE(combiner.IsPartMo(GetPart(gOfParts, 5)) == false);
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

TEST_CASE("GetDestinationParts", "[CombinerDFS]")
{
    Graph graph;
    // Create graph:
    //
    //       C
    //       |
    //   A - B - D
    //
    NameOnlyNode* nodeA = graph.CreateAndAddNode<NameOnlyNode>("a");
    NameOnlyNode* nodeB = graph.CreateAndAddNode<NameOnlyNode>("b");
    NameOnlyNode* nodeC = graph.CreateAndAddNode<NameOnlyNode>("c");
    NameOnlyNode* nodeD = graph.CreateAndAddNode<NameOnlyNode>("d");

    graph.Connect(nodeA, nodeB, 0);
    graph.Connect(nodeB, nodeC, 0);
    graph.Connect(nodeB, nodeD, 0);

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    GraphOfParts gOfParts;
    AddNodesToPart(gOfParts, { nodeA }, estOpt, compOpt, hwCaps);
    AddNodesToPart(gOfParts, { nodeB }, estOpt, compOpt, hwCaps);
    AddNodesToPart(gOfParts, { nodeC }, estOpt, compOpt, hwCaps);
    AddNodesToPart(gOfParts, { nodeD }, estOpt, compOpt, hwCaps);

    CheckPartId(gOfParts);

    dfs::Combiner combiner(gOfParts, hwCaps, estOpt);

    REQUIRE(combiner.GetDestinationParts(GetPart(gOfParts, 0)).size() == 1);
    REQUIRE(combiner.GetDestinationParts(GetPart(gOfParts, 0)).at(0).first == &GetPart(gOfParts, 1));
    REQUIRE(combiner.GetDestinationParts(GetPart(gOfParts, 1)).size() == 2);
    REQUIRE(combiner.GetDestinationParts(GetPart(gOfParts, 1)).at(0).first == &GetPart(gOfParts, 2));
    REQUIRE(combiner.GetDestinationParts(GetPart(gOfParts, 1)).at(1).first == &GetPart(gOfParts, 3));
    REQUIRE(combiner.GetDestinationParts(GetPart(gOfParts, 2)).size() == 0);
    REQUIRE(combiner.GetDestinationParts(GetPart(gOfParts, 3)).size() == 0);
}

TEST_CASE("Combination operator+", "[CombinerDFS]")
{
    Graph graph;
    // Create graph:
    //
    //  A - B - C
    //
    NameOnlyNode* nodeA = graph.CreateAndAddNode<NameOnlyNode>("a");
    NameOnlyNode* nodeB = graph.CreateAndAddNode<NameOnlyNode>("b");
    NameOnlyNode* nodeC = graph.CreateAndAddNode<NameOnlyNode>("c");

    graph.Connect(nodeA, nodeB, 0);
    graph.Connect(nodeB, nodeC, 0);

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    GraphOfParts gOfParts;
    AddNodesToPart(gOfParts, { nodeA }, estOpt, compOpt, hwCaps);
    AddNodesToPart(gOfParts, { nodeB }, estOpt, compOpt, hwCaps);
    AddNodesToPart(gOfParts, { nodeC }, estOpt, compOpt, hwCaps);

    CheckPartId(gOfParts);

    Part& partA = GetPart(gOfParts, 0);
    Part& partB = GetPart(gOfParts, 1);
    Part& partC = GetPart(gOfParts, 2);

    Plan planA(0);
    Plan planB(1);
    Plan planC(2);

    dfs::Combination combA(partA, planA);
    dfs::Combination combB(partB, planB);
    dfs::Combination combC(partC, planC);

    REQUIRE(combA.m_Elems.size() == 1);
    REQUIRE(combB.m_Elems.size() == 1);
    REQUIRE(combC.m_Elems.size() == 1);

    dfs::Combination comb;
    REQUIRE(comb.m_Elems.size() == 0);

    comb = combA + combB + combC;
    REQUIRE(comb.m_Elems.size() == 3);
    // All parts are in the final combination
    for (size_t i = 0; i < gOfParts.m_Parts.size(); ++i)
    {
        Part& part = GetPart(gOfParts, i);
        REQUIRE(comb.m_Elems.find(part.m_PartId) != comb.m_Elems.end());
    }

    // Nothing changes if combA is added again
    comb = comb + combA;
    REQUIRE(comb.m_Elems.size() == 3);

    // There is no glue
    for (size_t i = 0; i < gOfParts.m_Parts.size(); ++i)
    {
        Part& part = GetPart(gOfParts, i);
        for (auto& glueIt : comb.m_Elems.at(part.m_PartId).m_Glues)
        {
            REQUIRE(glueIt.second == nullptr);
        }
    }

    // Simple glue between B and C
    dfs::Glue glueB_C;
    glueB_C.m_Graph.AddOp(std::make_unique<DmaOp>());
    glueB_C.m_Graph.GetOps()[0]->m_DebugTag = "DmaBC";
    glueB_C.m_InputSlot                     = { glueB_C.m_Graph.GetOps()[0], 0 };
    glueB_C.m_Output                        = glueB_C.m_Graph.GetOps()[0];

    dfs::Combination combBGlue(partB, nodeC->GetInput(0), &glueB_C);

    comb = comb + combBGlue;
    // Number of elemnts didn't change
    REQUIRE(comb.m_Elems.size() == 3);
    // Glue has been added
    REQUIRE(comb.m_Elems.at(partB.m_PartId).m_Glues.size() == 1);
    const dfs::Glue* glueTest = comb.m_Elems.at(partB.m_PartId).m_Glues.at(nodeC->GetInput(0));
    // It has the correct tag
    REQUIRE(glueTest->m_Graph.GetOps()[0]->m_DebugTag == "DmaBC");
    REQUIRE(comb.m_Elems.at(partB.m_PartId).m_PlanId == planB.m_PlanId);
}

TEST_CASE("FindBestCombinationForPart cache", "[CombinerDFS]")
{
    Graph graph;
    // Create graph:
    //
    //  A - B - C
    //
    NameOnlyNode* nodeA = graph.CreateAndAddNode<NameOnlyNode>("a");
    NameOnlyNode* nodeB = graph.CreateAndAddNode<NameOnlyNode>("b");
    NameOnlyNode* nodeC = graph.CreateAndAddNode<NameOnlyNode>("c");

    graph.Connect(nodeA, nodeB, 0);
    graph.Connect(nodeB, nodeC, 0);

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    GraphOfParts gOfParts;
    AddNodesToPart(gOfParts, { nodeA }, estOpt, compOpt, hwCaps);
    AddNodesToPart(gOfParts, { nodeB }, estOpt, compOpt, hwCaps);
    AddNodesToPart(gOfParts, { nodeC }, estOpt, compOpt, hwCaps);

    CheckPartId(gOfParts);

    dfs::Combiner combiner(gOfParts, hwCaps, estOpt);

    Part& partA = GetPart(gOfParts, 0);
    Part& partB = GetPart(gOfParts, 1);
    Part& partC = GetPart(gOfParts, 2);

    // Map is empty
    REQUIRE(combiner.m_CombinationPerPartMap.size() == 0);
    dfs::Combination comb = combiner.FindBestCombinationForPart(partA);
    // Map has partA
    REQUIRE(combiner.m_CombinationPerPartMap.size() == 1);
    auto mapIt = combiner.m_CombinationPerPartMap.find(&partA);
    REQUIRE(mapIt != combiner.m_CombinationPerPartMap.end());
    comb = combiner.FindBestCombinationForPart(partA);
    // Map has still only partA
    REQUIRE(combiner.m_CombinationPerPartMap.size() == 1);
    comb = combiner.FindBestCombinationForPart(partB);
    // Map has partB
    REQUIRE(combiner.m_CombinationPerPartMap.size() == 2);
    mapIt = combiner.m_CombinationPerPartMap.find(&partB);
    REQUIRE(mapIt != combiner.m_CombinationPerPartMap.end());
    comb = combiner.FindBestCombinationForPart(partB);
    // Map has still only partA and partB
    REQUIRE(combiner.m_CombinationPerPartMap.size() == 2);
    comb = combiner.FindBestCombinationForPart(partC);
    // Map has partC
    REQUIRE(combiner.m_CombinationPerPartMap.size() == 3);
    mapIt = combiner.m_CombinationPerPartMap.find(&partC);
    REQUIRE(mapIt != combiner.m_CombinationPerPartMap.end());
    comb = combiner.FindBestCombinationForPart(partC);
    // Map has still only partA, partB and partC
    REQUIRE(combiner.m_CombinationPerPartMap.size() == 3);
}

TEST_CASE("GetSourceParts", "[CombinerDFS]")
{
    Graph graph;
    // Create graph:
    //      A
    //      |
    //  B - C - D
    //
    //
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

    CheckPartId(gOfParts);

    dfs::Combiner combiner(gOfParts, hwCaps, estOpt);

    REQUIRE(combiner.GetSourceParts(*gOfParts.m_Parts.at(0).get()).size() == 0);
    REQUIRE(combiner.GetSourceParts(*gOfParts.m_Parts.at(1).get()).size() == 0);
    REQUIRE(combiner.GetSourceParts(*gOfParts.m_Parts.at(2).get()).size() == 2);
    REQUIRE(combiner.GetSourceParts(*gOfParts.m_Parts.at(2).get()).at(0).first == gOfParts.m_Parts.at(1).get());
    REQUIRE(combiner.GetSourceParts(*gOfParts.m_Parts.at(2).get()).at(1).first == gOfParts.m_Parts.at(0).get());
    REQUIRE(combiner.GetSourceParts(*gOfParts.m_Parts.at(3).get()).size() == 1);
    REQUIRE(combiner.GetSourceParts(*gOfParts.m_Parts.at(3).get()).at(0).first == gOfParts.m_Parts.at(2).get());
}

TEST_CASE("ArePlansCompatible", "[CombinerDFS]")
{
    Graph graph;
    NameOnlyNode* nodeA = graph.CreateAndAddNode<NameOnlyNode>("a");
    NameOnlyNode* nodeB = graph.CreateAndAddNode<NameOnlyNode>("b");

    graph.Connect(nodeA, nodeB, 0);

    GraphOfParts gOfParts;

    const EstimationOptions estOpt;
    const CompilationOptions compOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);

    // Part consisting of node A
    AddNodesToPart(gOfParts, { nodeA }, estOpt, compOpt, hwCaps);
    std::unique_ptr<Plan> planA = std::make_unique<Plan>();
    planA->m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                        TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 17, 16, 16 },
                                                        TraversalOrder::Xyz, 4, QuantizationInfo()));
    planA->m_OpGraph.GetBuffers().back()->m_DebugTag = "InputDram";
    planA->m_OutputMappings                          = { { planA->m_OpGraph.GetBuffers()[0], nodeA } };
    gOfParts.m_Parts.back()->m_Plans.push_back(std::move(planA));

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

    dfs::Combiner combiner(gOfParts, hwCaps, estOpt);

    const Edge* edge = nodeA->GetOutput(0);
    REQUIRE(combiner.ArePlansCompatible(*(gOfParts.m_Parts.at(0)->m_Plans.at(0)),
                                        *(gOfParts.m_Parts.at(1)->m_Plans.at(0)), *edge) == true);
}

TEST_CASE("GluePartToCombination", "[CombinerDFS]")
{
    Graph graph;
    // Create graph:
    //
    //        B
    //  A     |
    //  |     v
    //   - -> D <- - C
    //
    NameOnlyNode* nodeA = graph.CreateAndAddNode<NameOnlyNode>("a");
    NameOnlyNode* nodeB = graph.CreateAndAddNode<NameOnlyNode>("b");
    NameOnlyNode* nodeC = graph.CreateAndAddNode<NameOnlyNode>("c");
    NameOnlyNode* nodeD = graph.CreateAndAddNode<NameOnlyNode>("d");

    graph.Connect(nodeA, nodeD, 0);
    graph.Connect(nodeB, nodeD, 1);
    graph.Connect(nodeC, nodeD, 2);

    const CompilationOptions compOpt;
    const EstimationOptions estOpt;
    const HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();

    GraphOfParts gOfParts;
    AddNodesToPart(gOfParts, { nodeA }, estOpt, compOpt, hwCaps);
    std::unique_ptr<Plan> planA = std::make_unique<Plan>(gOfParts.m_Parts.back()->GeneratePlanId());
    planA->m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                        TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 8, 32 },
                                                        TraversalOrder::Xyz, 4, QuantizationInfo()));
    planA->m_OutputMappings = { { planA->m_OpGraph.GetBuffers()[0], nodeA } };
    // Add plan to last part
    gOfParts.m_Parts.back()->m_Plans.push_back(std::move(planA));

    AddNodesToPart(gOfParts, { nodeB }, estOpt, compOpt, hwCaps);
    std::unique_ptr<Plan> planB = std::make_unique<Plan>(gOfParts.m_Parts.back()->GeneratePlanId());
    planB->m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                        TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 16, 16 },
                                                        TraversalOrder::Xyz, 4, QuantizationInfo()));
    planB->m_OutputMappings = { { planB->m_OpGraph.GetBuffers()[0], nodeB } };
    // Add plan to last part
    gOfParts.m_Parts.back()->m_Plans.push_back(std::move(planB));

    AddNodesToPart(gOfParts, { nodeC }, estOpt, compOpt, hwCaps);
    std::unique_ptr<Plan> planC = std::make_unique<Plan>(gOfParts.m_Parts.back()->GeneratePlanId());
    planC->m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB,
                                                        TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 16, 16 },
                                                        TraversalOrder::Xyz, 4, QuantizationInfo()));
    planC->m_OutputMappings = { { planC->m_OpGraph.GetBuffers()[0], nodeC } };
    // Add plan to last part
    gOfParts.m_Parts.back()->m_Plans.push_back(std::move(planC));

    AddNodesToPart(gOfParts, { nodeD }, estOpt, compOpt, hwCaps);
    std::unique_ptr<Plan> planD = std::make_unique<Plan>(gOfParts.m_Parts.back()->GeneratePlanId());
    planD->m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                        TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 16, 16, 32 },
                                                        TraversalOrder::Xyz, 4, QuantizationInfo()));
    planD->m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                        TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 8, 16, 48 },
                                                        TraversalOrder::Xyz, 4, QuantizationInfo()));
    planD->m_OpGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB,
                                                        TensorShape{ 1, 64, 64, 64 }, TensorShape{ 1, 32, 16, 48 },
                                                        TraversalOrder::Xyz, 4, QuantizationInfo()));

    planD->m_InputMappings = { { planD->m_OpGraph.GetBuffers()[0], nodeD->GetInput(0) },
                               { planD->m_OpGraph.GetBuffers()[1], nodeD->GetInput(1) },
                               { planD->m_OpGraph.GetBuffers()[2], nodeD->GetInput(2) } };
    // Add plan to last part
    gOfParts.m_Parts.back()->m_Plans.push_back(std::move(planD));

    CheckPartId(gOfParts);

    const Part& partA = GetPart(gOfParts, 0);
    const Part& partB = GetPart(gOfParts, 1);
    const Part& partC = GetPart(gOfParts, 2);
    const Part& partD = GetPart(gOfParts, 3);

    dfs::Combination combA(partA, partA.GetPlan(0));
    dfs::Combination combB(partB, partB.GetPlan(0));
    dfs::Combination combC(partC, partC.GetPlan(0));
    dfs::Combination combD(partD, partD.GetPlan(0));

    // Merge the combinations
    dfs::Combination comb = combA + combB + combC + combD;

    // There is no glue
    for (size_t i = 0; i < gOfParts.m_Parts.size(); ++i)
    {
        Part& part = GetPart(gOfParts, i);
        for (auto& glueIt : comb.m_Elems.at(part.m_PartId).m_Glues)
        {
            REQUIRE(glueIt.second == nullptr);
        }
    }

    dfs::Combiner combiner(gOfParts, hwCaps, estOpt);

    const auto& sources = combiner.GetSourceParts(partD);

    dfs::Combination combGlued = combiner.GluePartToCombination(partD, comb, sources);

    REQUIRE(combGlued.m_Elems.size() == 4);
    // There is a glue for each input part
    REQUIRE(combiner.m_GluesVector.size() == 3);

    for (size_t i = 0; i < combiner.m_GluesVector.size(); ++i)
    {
        if (!(combiner.m_GluesVector.at(i).get())->m_Graph.GetBuffers().empty())
        {
            REQUIRE((combiner.m_GluesVector.at(i).get())->m_Graph.GetOps().size() == 2);
            REQUIRE((combiner.m_GluesVector.at(i).get())->m_Graph.GetBuffers().at(0)->m_Location == Location::Dram);
        }
        else
        {
            REQUIRE((combiner.m_GluesVector.at(i).get())->m_Graph.GetOps().size() == 1);
        }
    }

    // A and B have glue and the buffer in Dram is in the expected format
    auto elemIt = combGlued.m_Elems.find(partA.m_PartId);
    REQUIRE(elemIt != combGlued.m_Elems.end());
    REQUIRE(elemIt->second.m_Glues.begin()->second->m_Graph.GetBuffers().at(0)->m_Location == Location::Dram);
    REQUIRE(elemIt->second.m_Glues.begin()->second->m_Graph.GetBuffers().at(0)->m_Format ==
            CascadingBufferFormat::FCAF_DEEP);
    elemIt = combGlued.m_Elems.find(partB.m_PartId);
    REQUIRE(elemIt != combGlued.m_Elems.end());
    REQUIRE(elemIt->second.m_Glues.begin()->second->m_Graph.GetBuffers().at(0)->m_Location == Location::Dram);
    REQUIRE(elemIt->second.m_Glues.begin()->second->m_Graph.GetBuffers().at(0)->m_Format ==
            CascadingBufferFormat::FCAF_WIDE);
}
