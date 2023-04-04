//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/Graph.hpp"
#include "../src/GraphNodes.hpp"

#include <catch.hpp>

#include <fstream>

using namespace ethosn::support_library;
namespace sl = ethosn::support_library;

/// Simple Node type for tests.
/// Includes a friendly name and ignores shape, quantisation info etc. so that tests
/// can focus on graph topology.
class NameOnlyNode : public Node
{
public:
    // cppcheck-suppress passedByValue
    NameOnlyNode(NodeId id, std::string name)
        : Node(id,
               TensorShape(),
               sl::DataType::UINT8_QUANTIZED,
               QuantizationInfo(),
               CompilerDataFormat::NONE,
               std::set<uint32_t>{ 0 })
        , m_Name(std::move(name))
    {}

    DotAttributes GetDotAttributes() override
    {
        return DotAttributes(std::to_string(m_Id), m_Name, "");
    }

    bool IsPrepared() override
    {
        return false;
    }

    NodeType GetNodeType() override
    {
        return NodeType::NameOnlyNode;
    }

    std::string m_Name;
};

/// Checks that the CollapseEdge function correctly removes the given edge
/// and preserves the order of connections. The test creates a graph with the following topology
/// (all edges directed left-to-right and inputs ordered top-to-bottom):
///
/// I1 \       / M ------- O1
///     \     /          /
///      --- S --------D ---- O2
/// I2 /                    /
///                       I3
///
/// After calling CollapseEdge on S-D, the resulting graph should be:
///
/// I1 \       / M ------- O1
///     \     /          /
///      --- S ----------
/// I2 /      \___________ O2
///                       /
///                      I3
TEST_CASE("CollapseEdge")
{
    bool debug = false;    // Enable to dump dot files to help debug.

    // Build initial graph
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

    if (debug)
    {
        std::ofstream dotStream("before.dot");
        g.DumpToDotFormat(dotStream);
    }

    // Call function being tested
    g.CollapseEdge(d->GetInput(0));

    if (debug)
    {
        std::ofstream dotStream("after.dot");
        g.DumpToDotFormat(dotStream);
    }

    // Check resulting graph structure
    REQUIRE(g.GetNodes().size() == 7);    // D should have been removed
    REQUIRE(i1->GetOutputs() == std::vector<Edge*>{ s->GetInput(0) });
    REQUIRE(i2->GetOutputs() == std::vector<Edge*>{ s->GetInput(1) });
    REQUIRE(s->GetOutputs() == std::vector<Edge*>{ m->GetInput(0), o1->GetInput(1), o2->GetInput(0) });
    REQUIRE(m->GetOutputs() == std::vector<Edge*>{ o1->GetInput(0) });
    REQUIRE(i3->GetOutputs() == std::vector<Edge*>{ o2->GetInput(1) });
}

/// Checks that the InsertNode function operates correctly and preserves the order of connections.
/// The test creates a graph with the following topology
/// (all edges directed left-to-right and inputs ordered top-to-bottom):
///
/// I1 \     / O1
///     \   /
///      A ------O2
///     /   \_
/// I2 /      \ O3
///
/// After calling InsertNodeAfter to insert a node (N) after A, the resulting graph should be:
///
/// I1 \      / O1
///     \    /
///      A--N ------O2
///     /    \_
/// I2 /       \ O3
TEST_CASE("InsertNodeAfter")
{
    bool debug = false;    // Enable to dump dot files to help debug.

    // Build initial graph
    Graph g;
    NameOnlyNode* i1 = g.CreateAndAddNode<NameOnlyNode>("I1");
    NameOnlyNode* i2 = g.CreateAndAddNode<NameOnlyNode>("I2");
    NameOnlyNode* a  = g.CreateAndAddNode<NameOnlyNode>("A");
    NameOnlyNode* o1 = g.CreateAndAddNode<NameOnlyNode>("O1");
    NameOnlyNode* o2 = g.CreateAndAddNode<NameOnlyNode>("O2");
    NameOnlyNode* o3 = g.CreateAndAddNode<NameOnlyNode>("O3");

    g.Connect(i1, a, 0);
    g.Connect(i2, a, 1);
    g.Connect(a, o1, 0);
    g.Connect(a, o2, 0);
    g.Connect(a, o3, 0);

    if (debug)
    {
        std::ofstream dotStream("before.dot");
        g.DumpToDotFormat(dotStream);
    }

    // Create new node to be inserted
    NameOnlyNode* n = g.CreateAndAddNode<NameOnlyNode>("N");

    // Call function being tested
    g.InsertNodeAfter(a, n);

    if (debug)
    {
        std::ofstream dotStream("after.dot");
        g.DumpToDotFormat(dotStream);
    }

    // Check resulting graph structure
    REQUIRE(g.GetNodes().size() == 7);    // One new node should have been added
    REQUIRE(i1->GetOutputs() == std::vector<Edge*>{ a->GetInput(0) });
    REQUIRE(i2->GetOutputs() == std::vector<Edge*>{ a->GetInput(1) });
    REQUIRE(a->GetOutputs() == std::vector<Edge*>{ n->GetInput(0) });
    REQUIRE(n->GetOutputs() == std::vector<Edge*>{ o1->GetInput(0), o2->GetInput(0), o3->GetInput(0) });
}

/// Checks that setting FixGraphConvertOutputTo on a node leads to FormatConversionNodes being inserted into the graph.
TEST_CASE("FixGraph ConvertOutputTo")
{
    // Create the Graph
    Graph g;
    Node* input = g.CreateAndAddNode<InputNode>(
        TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC), std::set<uint32_t>{ 0 });
    Node* conv = g.CreateAndAddNode<NameOnlyNode>("C");
    g.Connect(input, conv);

    input->SetFixGraphConvertOutputTo(CompilerDataFormat::NHWCB);

    input->FixGraph(g, FixGraphSeverity::Highest);

    // Check resulting graph structure
    REQUIRE(g.GetNodes().size() == 4);    // Two new node should have been added
    auto fmtConv1 = g.GetNodes()[2].get();
    auto fmtConv2 = g.GetNodes()[3].get();
    REQUIRE(fmtConv1->GetNodeType() == NodeType::FormatConversionNode);
    REQUIRE(fmtConv1->GetFormat() == CompilerDataFormat::NHWCB);
    REQUIRE(fmtConv2->GetNodeType() == NodeType::FormatConversionNode);
    REQUIRE(fmtConv2->GetFormat() == CompilerDataFormat::NHWC);

    REQUIRE(input->GetOutputs()[0] == fmtConv1->GetInput(0));
    REQUIRE(fmtConv1->GetOutputs()[0] == fmtConv2->GetInput(0));
    REQUIRE(fmtConv2->GetOutputs()[0] == conv->GetInput(0));
}
/// Checks that setting FixGraphConvertOutputTo on a node which already has a FormatConversionNode
/// on its output doesn't add another. If it did this could lead to the preparation loop getting stuck
/// and repeatedly adding more nodes with no benefit.
TEST_CASE("FixGraph ConvertOutputTo avoids creating chains of FormatConversionNode")
{
    // Create the Graph
    Graph graph;
    Node* input = graph.CreateAndAddNode<InputNode>(
        TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC), std::set<uint32_t>{ 0 });
    Node* fc = graph.CreateAndAddNode<FormatConversionNode>(TensorShape{ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED,
                                                            QuantizationInfo(), CompilerDataFormat::NHWCB,
                                                            std::set<uint32_t>{ 0 });
    graph.Connect(input, fc);

    input->SetFixGraphConvertOutputTo(CompilerDataFormat::NHWCB);

    input->FixGraph(graph, FixGraphSeverity::Highest);

    REQUIRE(graph.GetNodes().size() == 2);    // No new nodes should have been added
}

/// Checks that going from InputNode -> OutputNode adds a Copy Node
TEST_CASE("FixGraph InputNode -> OutputNode Adds CopyNode")
{
    // Create the Graph
    Graph graph;
    Node* input = graph.CreateAndAddNode<InputNode>(
        TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC), std::set<uint32_t>{ 0 });
    Node* output = graph.CreateAndAddNode<OutputNode>(DataType::UINT8_QUANTIZED, std::set<uint32_t>{ 0 }, 0);
    graph.Connect(input, output);

    output->FixGraph(graph, FixGraphSeverity::Highest);

    REQUIRE(graph.GetNodes().size() == 3);
    REQUIRE(graph.GetNodes()[2]->GetNodeType() == NodeType::CopyNode);
}

/// Checks that going from InputNode -> ReinterpretNode -> OutputNode adds a Copy Node
TEST_CASE("FixGraph InputNode -> ReinterpretNode -> OutputNode Adds CopyNode")
{
    // Create the Graph
    Graph graph;
    Node* input = graph.CreateAndAddNode<InputNode>(
        TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC), std::set<uint32_t>{ 0 });
    Node* reinterpret =
        graph.CreateAndAddNode<ReinterpretNode>(TensorShape{ 1, 16, 32, 8 }, DataType::UINT8_QUANTIZED,
                                                QuantizationInfo(), CompilerDataFormat::NHWC, std::set<uint32_t>{ 0 });
    Node* output = graph.CreateAndAddNode<OutputNode>(DataType::UINT8_QUANTIZED, std::set<uint32_t>{ 0 }, 0);
    graph.Connect(input, reinterpret);
    graph.Connect(reinterpret, output);

    output->FixGraph(graph, FixGraphSeverity::Highest);

    REQUIRE(graph.GetNodes().size() == 4);
    REQUIRE(graph.GetNodes()[3].get()->GetNodeType() == NodeType::CopyNode);
}

/// Checks that going from any Node to ReinterpretNode works fine when the Node before
/// ReinterpretNode outputs in compressed format.
/// Fix graph should set the CompressionHint for the node prior to ReinterpretNode
/// to CompressionHint::RequiredUncompressed.
TEST_CASE("FixGraph modifies CompressionHint for ReinterpretNode")
{
    // Create the graph
    Graph graph;

    Node* input = graph.CreateAndAddNode<InputNode>(
        TensorInfo({ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, DataFormat::NHWC), std::set<uint32_t>{ 0 });

    Node* formatConversion = graph.CreateAndAddNode<FormatConversionNode>(
        TensorShape{ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, QuantizationInfo(), CompilerDataFormat::NHWCB,
        std::set<uint32_t>{ 0 });

    ethosn::support_library::utils::ShapeMultiplier shapeMultiplier = { utils::Fraction{ 1, 1 },
                                                                        utils::Fraction{ 1, 1 },
                                                                        utils::Fraction{ 1, 1 } };
    Node* fuseOnlyPleOperation                                      = graph.CreateAndAddNode<FuseOnlyPleOperationNode>(
        TensorShape{ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, QuantizationInfo(),
        ethosn::command_stream::PleOperation::SIGMOID, CompilerDataFormat::NHWCB, shapeMultiplier,
        std::set<uint32_t>{ 1 });
    fuseOnlyPleOperation->SetCompressedFormat(CompilerDataCompressedFormat::FCAF_WIDE);

    Node* reinterpret0 =
        graph.CreateAndAddNode<ReinterpretNode>(TensorShape{ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED,
                                                QuantizationInfo(), CompilerDataFormat::NHWCB, std::set<uint32_t>{ 2 });
    reinterpret0->SetCompressedFormat(CompilerDataCompressedFormat::FCAF_WIDE);

    Node* reinterpret1 =
        graph.CreateAndAddNode<ReinterpretNode>(TensorShape{ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED,
                                                QuantizationInfo(), CompilerDataFormat::NHWCB, std::set<uint32_t>{ 3 });
    reinterpret1->SetCompressedFormat(CompilerDataCompressedFormat::FCAF_WIDE);

    graph.Connect(input, formatConversion);
    graph.Connect(formatConversion, fuseOnlyPleOperation);
    graph.Connect(fuseOnlyPleOperation, reinterpret0);
    graph.Connect(reinterpret0, reinterpret1);

    auto fuseOnlyPleNode  = graph.GetNodes()[2].get();
    auto reinterpretNode0 = graph.GetNodes()[3].get();
    auto reinterpretNode1 = graph.GetNodes()[4].get();

    // Checks before fixing the graph
    REQUIRE(fuseOnlyPleNode->GetCompressionHint() != CompressionHint::RequiredUncompressed);
    REQUIRE(reinterpretNode0->GetCompressionHint() != CompressionHint::RequiredUncompressed);
    REQUIRE(reinterpretNode1->GetCompressionHint() != CompressionHint::RequiredUncompressed);

    // Fixing the node that outputs to 2nd ReinterpretNode
    reinterpret1->FixGraph(graph, FixGraphSeverity::Highest);

    REQUIRE(graph.GetNodes().size() == 5);
    fuseOnlyPleNode  = graph.GetNodes()[2].get();
    reinterpretNode0 = graph.GetNodes()[3].get();
    reinterpretNode1 = graph.GetNodes()[4].get();

    REQUIRE(fuseOnlyPleNode->GetCompressionHint() != CompressionHint::RequiredUncompressed);
    REQUIRE(reinterpretNode0->GetCompressionHint() == CompressionHint::RequiredUncompressed);
    REQUIRE(reinterpretNode1->GetCompressionHint() != CompressionHint::RequiredUncompressed);

    // Fixing the node that outputs to 1st ReinterpretNode
    reinterpret0->FixGraph(graph, FixGraphSeverity::Highest);

    REQUIRE(graph.GetNodes().size() == 5);
    fuseOnlyPleNode  = graph.GetNodes()[2].get();
    reinterpretNode0 = graph.GetNodes()[3].get();
    reinterpretNode1 = graph.GetNodes()[4].get();

    REQUIRE(fuseOnlyPleNode->GetCompressionHint() == CompressionHint::RequiredUncompressed);
    REQUIRE(reinterpretNode0->GetCompressionHint() == CompressionHint::RequiredUncompressed);
    REQUIRE(reinterpretNode1->GetCompressionHint() != CompressionHint::RequiredUncompressed);
}

/// Checks that a ReinterpretNode is prepared only when the previous node's
/// output is uncompressed.
TEST_CASE("IsPrepared returns an appropriate bool value for ReinterpretNode")
{
    // Create the graph
    Graph graph;

    ethosn::support_library::utils::ShapeMultiplier shapeMultiplier = { utils::Fraction{ 1, 1 },
                                                                        utils::Fraction{ 1, 1 },
                                                                        utils::Fraction{ 1, 1 } };
    Node* fuseOnlyPleOperation                                      = graph.CreateAndAddNode<FuseOnlyPleOperationNode>(
        TensorShape{ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED, QuantizationInfo(),
        ethosn::command_stream::PleOperation::SIGMOID, CompilerDataFormat::NHWCB, shapeMultiplier,
        std::set<uint32_t>{ 1 });
    fuseOnlyPleOperation->SetCompressedFormat(CompilerDataCompressedFormat::FCAF_WIDE);

    Node* reinterpret0 =
        graph.CreateAndAddNode<ReinterpretNode>(TensorShape{ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED,
                                                QuantizationInfo(), CompilerDataFormat::NHWCB, std::set<uint32_t>{ 2 });
    reinterpret0->SetCompressedFormat(CompilerDataCompressedFormat::FCAF_WIDE);

    Node* reinterpret1 =
        graph.CreateAndAddNode<ReinterpretNode>(TensorShape{ 1, 16, 16, 16 }, DataType::UINT8_QUANTIZED,
                                                QuantizationInfo(), CompilerDataFormat::NHWCB, std::set<uint32_t>{ 3 });
    reinterpret1->SetCompressedFormat(CompilerDataCompressedFormat::FCAF_WIDE);

    graph.Connect(fuseOnlyPleOperation, reinterpret0);
    graph.Connect(reinterpret0, reinterpret1);

    REQUIRE(!graph.GetNodes()[1].get()->IsPrepared());
    REQUIRE(!graph.GetNodes()[2].get()->IsPrepared());

    // Mimicking the effect of changing the first ReinterpretNode's previous node's output from
    // compressed to an uncompressed format.
    graph.GetNodes()[0].get()->SetCompressedFormat(CompilerDataCompressedFormat::NONE);

    REQUIRE(graph.GetNodes()[1].get()->IsPrepared());
    REQUIRE(!graph.GetNodes()[2].get()->IsPrepared());

    // Mimicking the effect of changing the second ReinterpretNode's previous node's output from
    // compressed to an uncompressed format.
    graph.GetNodes()[1].get()->SetCompressedFormat(CompilerDataCompressedFormat::NONE);
    REQUIRE(graph.GetNodes()[1].get()->IsPrepared());
    REQUIRE(graph.GetNodes()[2].get()->IsPrepared());
}
