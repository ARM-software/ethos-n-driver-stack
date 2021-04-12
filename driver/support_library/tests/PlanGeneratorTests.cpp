//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "GlobalParameters.hpp"
#include "GraphNodes.hpp"
#include "TestUtils.hpp"
#include "cascading/Cascading.hpp"
#include "cascading/Visualisation.hpp"
#include "ethosn_support_library/Support.hpp"

#include <catch.hpp>
#include <fstream>
#include <regex>
#include <sstream>

using namespace ethosn::support_library;
namespace sl = ethosn::support_library;

namespace
{

using TS = ethosn::support_library::TensorShape;
using TI = ethosn::support_library::TensorInfo;
using QI = ethosn::support_library::QuantizationInfo;

Node* CreateAndAddInputNode(Graph& g, TS tsIn = TS({ 1, 32, 32, 3 }))
{
    return g.CreateAndAddNode<InputNode>(tsIn, std::set<uint32_t>());
}

Node* CreateAndAddOutputNode(Graph& g)
{
    return g.CreateAndAddNode<OutputNode>(sl::DataType::UINT8_QUANTIZED, std::set<uint32_t>(), 0);
}

Node* CreateAndAddMceOperationNode(Graph& g, TS tsOut, const uint32_t kH, const uint32_t kW)
{
    const std::vector<uint8_t> weights(kH * kW, 1);
    return g.CreateAndAddNode<MceOperationNode>(
        TS(), tsOut, sl::DataType::UINT8_QUANTIZED, QI(),
        TI({ kH, kW, 1, 1 }, ethosn::support_library::DataType::UINT8_QUANTIZED,
           ethosn::support_library::DataFormat::HWIO, QuantizationInfo(0, 0.9f)),
        weights, TI({ 1, 1, 1, 1 }), std::vector<int32_t>{ 0 }, Stride(), 0, 0,
        ethosn::command_stream::MceOperation::CONVOLUTION, CompilerDataFormat::NHWCB, std::set<uint32_t>{ 1 });
}

Node* CreateAndAddMceOperationNode(Graph& g)
{
    return CreateAndAddMceOperationNode(g, TS(), 1, 1);
}

Node* CreateAndAddMceOperationNode(Graph& g, TS tsOut)
{
    return CreateAndAddMceOperationNode(g, tsOut, 1, 1);
}

Node* CreateAndAddMcePostProcessOperationNode(Graph& g)
{
    return g.CreateAndAddNode<McePostProcessOperationNode>(TS(), sl::DataType::UINT8_QUANTIZED, QI(), 0, 255,
                                                           CompilerDataFormat::NHWCB, std::set<uint32_t>());
}

Node* CreateAndAddConstantNode(Graph& g)
{
    return g.CreateAndAddNode<ConstantNode>(TensorShape(), std::vector<uint8_t>(), std::set<uint32_t>());
}

Node* CreateAndAddFuseOnlyPleOperationNode(Graph& g, TS tensorShape, utils::ShapeMultiplier shapeMultiplier)
{
    return g.CreateAndAddNode<FuseOnlyPleOperationNode>(
        tensorShape, sl::DataType::UINT8_QUANTIZED, QI(), ethosn::command_stream::PleOperation::INTERLEAVE_2X2_2_2,
        CompilerDataFormat::NHWCB, shapeMultiplier, std::set<uint32_t>{ 1 });
}

Node* CreateAndAddFormatConversionNode(Graph& g, CompilerDataFormat format, TS tsOut = TS())
{
    return g.CreateAndAddNode<FormatConversionNode>(tsOut, sl::DataType::UINT8_QUANTIZED, QI(), format,
                                                    std::set<uint32_t>());
}

Node* CreateAndAddReinterpretNode(Graph& g, TS tensorShape = TS())
{
    return g.CreateAndAddNode<ReinterpretNode>(tensorShape, sl::DataType::UINT8_QUANTIZED, QI(),
                                               CompilerDataFormat::NHWC, std::set<uint32_t>());
}

void BuildGraphWithoutBranchingBeforeMcePostProcessNode(Graph& g)
{
    /*
         Graph looks like this:

                    / O1
                   /
        I1--N1--MPP--N2--O2
                   \
                    \ O3
        */

    auto i1  = CreateAndAddInputNode(g);
    auto n1  = CreateAndAddMceOperationNode(g);
    auto mpp = CreateAndAddMcePostProcessOperationNode(g);
    auto n2  = CreateAndAddMceOperationNode(g);
    auto o1  = CreateAndAddOutputNode(g);
    auto o2  = CreateAndAddOutputNode(g);
    auto o3  = CreateAndAddOutputNode(g);

    g.Connect(i1, n1, 0);
    g.Connect(n1, mpp, 0);
    g.Connect(mpp, o1, 0);
    g.Connect(mpp, o3, 0);
    g.Connect(mpp, n2, 0);
    g.Connect(n2, o2, 0);
}

void BuildGraphWithBranchingBeforeMcePostProcessNode(Graph& g)
{
    /*
         Graph looks like this:

                    / O1
                   /
        I1--N1---N2--N3--O2
              \     /
               \   /
                MPP
                   \
                    \ O3
        */

    auto i1  = CreateAndAddInputNode(g);
    auto n1  = CreateAndAddMceOperationNode(g);
    auto mpp = CreateAndAddMcePostProcessOperationNode(g);
    auto n2  = CreateAndAddMceOperationNode(g);
    auto n3  = CreateAndAddConstantNode(g);
    auto o1  = CreateAndAddOutputNode(g);
    auto o2  = CreateAndAddOutputNode(g);
    auto o3  = CreateAndAddOutputNode(g);

    g.Connect(i1, n1, 0);
    g.Connect(n1, mpp, 0);
    g.Connect(n1, n2, 0);
    g.Connect(n2, o1, 0);
    g.Connect(n2, n3, 0);
    g.Connect(n3, o2, 0);
    g.Connect(mpp, n3, 0);
    g.Connect(mpp, o3, 0);
}

void BuildGraphWithNonMceOpNodeBeforeMcePostProcessNode(Graph& g)
{
    /*
         Graph looks like this:

                    / O1
                   /
        I1--N1--MPP--N2--O2
                   \
                    \ O3
        */

    auto i1  = CreateAndAddInputNode(g);
    auto n1  = CreateAndAddConstantNode(g);
    auto mpp = CreateAndAddMcePostProcessOperationNode(g);
    auto n2  = CreateAndAddMceOperationNode(g);
    auto o1  = CreateAndAddOutputNode(g);
    auto o2  = CreateAndAddOutputNode(g);
    auto o3  = CreateAndAddOutputNode(g);

    g.Connect(i1, n1, 0);
    g.Connect(n1, mpp, 0);
    g.Connect(mpp, o1, 0);
    g.Connect(mpp, o3, 0);
    g.Connect(mpp, n2, 0);
    g.Connect(n2, o2, 0);
}

Part BuildSinglePartWithOneNode(Graph& g,
                                const EstimationOptions& estOpt,
                                const CompilationOptions& compOpt,
                                const HardwareCapabilities& caps)
{
    TS tsIn   = { 1, 32, 32, 3 };
    TS tsOut  = { 1, 64, 64, 1 };
    auto in   = CreateAndAddInputNode(g, tsIn);
    auto node = CreateAndAddMceOperationNode(g, tsOut);
    auto out  = CreateAndAddOutputNode(g);

    // Graph must be connected to be valid
    g.Connect(in, node, 0);
    g.Connect(node, out, 0);

    Part part(estOpt, compOpt, caps);
    part.m_SubGraph.push_back(node);
    return part;
}

Part BuildSinglePartWithTwoNodes(Graph& g,
                                 const EstimationOptions& estOpt,
                                 const CompilationOptions& compOpt,
                                 const HardwareCapabilities& caps)
{
    TS tsIn    = { 1, 32, 32, 3 };
    TS tsOut   = { 1, 64, 64, 1 };
    auto in    = CreateAndAddInputNode(g, tsIn);
    auto node1 = CreateAndAddMceOperationNode(g, tsOut);
    auto node2 = CreateAndAddMcePostProcessOperationNode(g);
    auto out   = CreateAndAddOutputNode(g);

    // Graph must be connected to be valid
    g.Connect(in, node1, 0);
    g.Connect(node1, node2, 0);
    g.Connect(node2, out, 0);

    Part part(estOpt, compOpt, caps);
    part.m_SubGraph.push_back(node1);
    part.m_SubGraph.push_back(node2);
    return part;
}

Part BuildPartWithFuseOnlyPle(Graph& g,
                              const EstimationOptions& estOpt,
                              const CompilationOptions& compOpt,
                              const HardwareCapabilities& caps)
{
    TS ts                                  = { 1, 8, 1, 1 };
    utils::ShapeMultiplier shapeMultiplier = { { 1, 2 }, { 1, 2 }, 1 };
    auto in                                = CreateAndAddInputNode(g);
    auto node1                             = CreateAndAddFuseOnlyPleOperationNode(g, ts, shapeMultiplier);
    auto out                               = CreateAndAddOutputNode(g);

    // Graph must be connected to be valid
    g.Connect(in, node1, 0);
    g.Connect(node1, out, 0);

    Part part(estOpt, compOpt, caps);
    part.m_SubGraph.push_back(node1);
    return part;
}

Part BuildPartWithLeadingFormatConversionNode(Graph& g,
                                              const EstimationOptions& estOpt,
                                              const CompilationOptions& compOpt,
                                              const HardwareCapabilities& caps)
{
    TS tsIn   = { 1, 32, 32, 4 };
    TS tsOut  = { 1, 64, 64, 1 };
    auto in   = CreateAndAddInputNode(g, tsIn);
    auto node = CreateAndAddFormatConversionNode(g, CompilerDataFormat::NHWC, tsOut);
    auto out  = CreateAndAddReinterpretNode(g, tsOut);

    // Graph must be connected to be valid
    g.Connect(in, node, 0);
    g.Connect(node, out, 0);

    Part part(estOpt, compOpt, caps);
    part.m_SubGraph.push_back(node);
    return part;
}

Part BuildPartWithTrailingFormatConversionNode(Graph& g,
                                               const EstimationOptions& estOpt,
                                               const CompilationOptions& compOpt,
                                               const HardwareCapabilities& caps)
{
    TS tsIn   = { 1, 32, 32, 4 };
    TS tsOut  = { 1, 64, 64, 1 };
    auto in   = CreateAndAddReinterpretNode(g, tsIn);
    auto node = CreateAndAddFormatConversionNode(g, CompilerDataFormat::NHWCB, tsOut);
    auto out  = CreateAndAddOutputNode(g);

    // Graph must be connected to be valid
    g.Connect(in, node, 0);
    g.Connect(node, out, 0);

    Part part(estOpt, compOpt, caps);
    part.m_SubGraph.push_back(node);
    return part;
}

Part BuildPartWithReinterpretNode(Graph& g,
                                  const EstimationOptions& estOpt,
                                  const CompilationOptions& compOpt,
                                  const HardwareCapabilities& caps)
{
    TS tsIn   = { 1, 32, 32, 4 };
    TS tsOut  = { 1, 64, 64, 1 };
    auto in   = CreateAndAddFormatConversionNode(g, CompilerDataFormat::NHWC, tsIn);
    auto node = CreateAndAddReinterpretNode(g, tsOut);
    auto out  = CreateAndAddFormatConversionNode(g, CompilerDataFormat::NHWCB, tsOut);

    // Graph must be connected to be valid
    g.Connect(in, node, 0);
    g.Connect(node, out, 0);

    Part part(estOpt, compOpt, caps);
    part.m_SubGraph.push_back(node);
    return part;
}

void AssertPart(const Part& part,
                const Edge* input,
                const Node* output,
                const std::function<void(const Plan&)>& assertPlan)
{
    for (Plans::size_type i = 0; i < part.m_Plans.size(); ++i)
    {
        const auto& plan = part.GetPlan(i);
        assertPlan(plan);
        auto inputBuffer         = plan.GetInputBuffer(input);
        auto outputBuffer        = plan.GetOutputBuffer(output);
        inputBuffer->m_DebugTag  = std::string("Input buffer: ") + inputBuffer->m_DebugTag;
        outputBuffer->m_DebugTag = std::string("Output buffer: ") + outputBuffer->m_DebugTag;
    }
}

std::ostream& operator<<(std::ostream& os, const ethosn::support_library::TensorShape& ts)
{
    os << "{ " << ts[0] << ", " << ts[1] << ", " << ts[2] << ", " << ts[3] << " }";
    return os;
}

using StripeShape = TensorShape;

void RequireInputBuffer(const Part& part,
                        const Edge* edge,
                        CascadingBufferFormat format,
                        Location location,
                        uint32_t stripes,
                        TensorShape ts,
                        StripeShape ss)
{
    for (uint32_t i = 0; i < part.GetNumPlans(); ++i)
    {
        const Buffer* buffer = part.GetPlan(i).GetInputBuffer(edge);
        REQUIRE(buffer);
        if ((buffer->m_Format == format) && (buffer->m_Location == location) && (buffer->m_NumStripes == stripes) &&
            (buffer->m_TensorShape == ts) && (buffer->m_StripeShape == ss))
        {
            return;
        }
    }
    std::ostringstream oss;
    oss << "Looking for InputBuffer with tensor: " << ts << ", stripe: " << ss << ", num stripes: " << stripes
        << ", location: " << static_cast<int>(location) << ", format: " << static_cast<int>(format) << "\n";
    oss << "Plans for part: \n";
    for (uint32_t i = 0; i < part.GetNumPlans(); ++i)
    {
        const Buffer* buffer = part.GetPlan(i).GetInputBuffer(edge);
        oss << "Plan: " << i << ", buffer: " << buffer->m_DebugTag << ", tensor: ";
        oss << buffer->m_TensorShape << ", stripe: ";
        oss << buffer->m_StripeShape;
        oss << ", num stripes: " << buffer->m_NumStripes << ", location: " << static_cast<int>(buffer->m_Location)
            << ", format: " << static_cast<int>(buffer->m_Format) << "\n";
    }
    INFO(oss.str());
    REQUIRE(false);
}

void RequireOutputBuffer(const Part& part,
                         const Node* node,
                         CascadingBufferFormat format,
                         Location location,
                         uint32_t stripes,
                         TensorShape ts,
                         StripeShape ss)
{
    for (uint32_t i = 0; i < part.GetNumPlans(); ++i)
    {
        const Buffer* buffer = part.GetPlan(i).GetOutputBuffer(node);
        REQUIRE(buffer);
        if ((buffer->m_Format == format) && (buffer->m_Location == location) && (buffer->m_NumStripes == stripes) &&
            (buffer->m_TensorShape == ts) && (buffer->m_StripeShape == ss))
        {
            return;
        }
    }
    std::ostringstream oss;
    oss << "Looking for OutputBuffer with tensor: " << ts << ", stripe: " << ss << ", num stripes: " << stripes
        << ", location: " << static_cast<int>(location) << ", format: " << static_cast<int>(format) << "\n";
    oss << "Plans for part: \n";
    for (uint32_t i = 0; i < part.GetNumPlans(); ++i)
    {
        const Buffer* buffer = part.GetPlan(i).GetOutputBuffer(node);
        oss << "Plan: " << i << ", buffer: " << buffer->m_DebugTag << ", tensor: ";
        oss << buffer->m_TensorShape << ", stripe: ";
        oss << buffer->m_StripeShape;
        oss << ", num stripes: " << stripes << ", location: " << static_cast<int>(buffer->m_Location)
            << ", format: " << static_cast<int>(buffer->m_Format) << "\n";
    }
    INFO(oss.str());
    REQUIRE(false);
}

bool ContainsInputStripe(const Plan::InputMapping& inputMappings,
                         const TensorShape& stripe,
                         const uint32_t numInputStripes)
{
    for (auto input : inputMappings)
    {
        if (input.first->m_StripeShape == stripe && input.first->m_NumStripes == numInputStripes)
        {
            return true;
        }
    }
    return false;
}

bool ContainsOutputStripe(const Plan::OutputMapping& outputMappings,
                          const TensorShape& stripe,
                          const uint32_t numOutputStripes)
{
    for (auto output : outputMappings)
    {
        if (output.first->m_StripeShape == stripe && output.first->m_NumStripes == numOutputStripes)
        {
            return true;
        }
    }
    return false;
}

std::pair<bool, size_t> GetPlanIndexContainingStripes(const Plans& plans,
                                                      const TensorShape& inputStripe,
                                                      const uint32_t numInputStripes,
                                                      const TensorShape& outputStripe,
                                                      const uint32_t numOutputStripes)
{
    for (uint32_t i = 0; i < plans.size(); ++i)
    {
        bool containsInputStripe  = ContainsInputStripe(plans[i]->m_InputMappings, inputStripe, numInputStripes);
        bool containsOutputStripe = ContainsOutputStripe(plans[i]->m_OutputMappings, outputStripe, numOutputStripes);
        if (containsInputStripe && containsOutputStripe)
        {
            return { true, i };
        }
    }
    return { false, 0 };
}

bool ContainsPlanWithStripes(const Plans& plans,
                             const TensorShape& inputStripe,
                             const uint32_t numInputStripes,
                             const TensorShape& outputStripe,
                             const uint32_t numOutputStripes)
{
    return GetPlanIndexContainingStripes(plans, inputStripe, numInputStripes, outputStripe, numOutputStripes).first;
}

void SavePlansToDot(const Plans& plans, const std::string test)
{
    if (!g_AllowDotFileGenerationInTests)
    {
        return;
    }

    std::stringstream str;
    std::stringstream stripes;
    for (const auto& plan : plans)
    {
        SaveOpGraphToDot(plan->m_OpGraph, str, DetailLevel::High);

        SaveOpGraphToTxtFile(plan->m_OpGraph, stripes);
    }

    std::regex re("digraph");
    std::string s = std::regex_replace(str.str(), re, "subgraph");

    std::ofstream file(test + ".dot");
    std::ofstream stripesFile(test + "_stripes.txt");
    file << "digraph {" << std::endl << s << "}" << std::endl;
    stripesFile << stripes.str() << std::endl;
}

}    // namespace

TEST_CASE("PlanGenerator: Generate parts from graph without branching before MCE PP node")
{
    // Given
    const EstimationOptions estOpt;
    const CompilationOptions compOpt = GetDefaultCompilationOptions();
    const HardwareCapabilities caps  = GetEthosN77HwCapabilities();
    Graph g;
    BuildGraphWithoutBranchingBeforeMcePostProcessNode(g);
    REQUIRE(g.GetNodes().size() == 7);

    // When
    GraphOfParts gop   = CreateGraphOfParts(g, estOpt, compOpt, caps);
    const Parts& parts = gop.GetParts();

    // Then, no part has two nodes becuase of non-MCE Op before PP Op.
    REQUIRE(parts.size() == 6);
    REQUIRE(parts[0]->m_SubGraph.size() == 1);
    REQUIRE(parts[1]->m_SubGraph.size() == 2);
    REQUIRE(parts[2]->m_SubGraph.size() == 1);
    REQUIRE(parts[3]->m_SubGraph.size() == 1);
    REQUIRE(parts[4]->m_SubGraph.size() == 1);
    REQUIRE(parts[5]->m_SubGraph.size() == 1);
}

TEST_CASE("PlanGenerator: Generate parts from graph with branching before MCE PP node")
{
    // Given
    const EstimationOptions estOpt;
    const CompilationOptions compOpt = GetDefaultCompilationOptions();
    const HardwareCapabilities caps  = GetEthosN77HwCapabilities();
    Graph g;
    BuildGraphWithBranchingBeforeMcePostProcessNode(g);
    REQUIRE(g.GetNodes().size() == 8);

    // When
    GraphOfParts gop   = CreateGraphOfParts(g, estOpt, compOpt, caps);
    const Parts& parts = gop.GetParts();

    // Then, no parts have two nodes because of branching.
    REQUIRE(parts.size() == 8);
    REQUIRE(parts[0]->m_SubGraph.size() == 1);
    REQUIRE(parts[1]->m_SubGraph.size() == 1);
    REQUIRE(parts[2]->m_SubGraph.size() == 1);
    REQUIRE(parts[3]->m_SubGraph.size() == 1);
    REQUIRE(parts[4]->m_SubGraph.size() == 1);
    REQUIRE(parts[5]->m_SubGraph.size() == 1);
    REQUIRE(parts[6]->m_SubGraph.size() == 1);
    REQUIRE(parts[7]->m_SubGraph.size() == 1);
}

TEST_CASE("PlanGenerator: Generate parts from graph with non MCE Operation node before MCE PP node")
{
    // Given
    const EstimationOptions estOpt;
    const CompilationOptions compOpt = GetDefaultCompilationOptions();
    const HardwareCapabilities caps  = GetEthosN77HwCapabilities();
    Graph g;
    BuildGraphWithNonMceOpNodeBeforeMcePostProcessNode(g);
    REQUIRE(g.GetNodes().size() == 7);

    // When
    GraphOfParts gop   = CreateGraphOfParts(g, estOpt, compOpt, caps);
    const Parts& parts = gop.GetParts();

    // Then, no part has two nodes becuase of non-MCE Op before PP Op.
    REQUIRE(parts.size() == 7);
    REQUIRE(parts[0]->m_SubGraph.size() == 1);
    REQUIRE(parts[1]->m_SubGraph.size() == 1);
    REQUIRE(parts[2]->m_SubGraph.size() == 1);
    REQUIRE(parts[3]->m_SubGraph.size() == 1);
    REQUIRE(parts[4]->m_SubGraph.size() == 1);
    REQUIRE(parts[5]->m_SubGraph.size() == 1);
    REQUIRE(parts[6]->m_SubGraph.size() == 1);
}

TEST_CASE("PlanGenerator: Generate plans from a part with single node")
{
    // Given
    const EstimationOptions estOpt;
    CompilationOptions compOpt      = GetDefaultCompilationOptions();
    compOpt.m_DisableWinograd       = GENERATE(false, true);
    const HardwareCapabilities caps = GetEthosN77HwCapabilities();
    Graph g;
    auto part = BuildSinglePartWithOneNode(g, estOpt, compOpt, caps);
    std::string expected =
        R"(digraph SupportLibraryGraph
\{
.*
Buffer_.* -> MceOp_.*\[ label="Input 0"\]
MceOp_.* -> Buffer_.*
Buffer_.* -> DmaOp_.*
DmaOp_.* -> Buffer_.*
Buffer_.* -> MceOp_.*\[ label="Input 1"\]
\{.*\}?
\}.*
)";
    expected.erase(std::remove(expected.begin(), expected.end(), '\n'), expected.end());
    std::regex re(expected);
    std::smatch m;

    // When
    part.CreatePlans();

    // Then
    REQUIRE(part.m_Plans.size() == 74);
    SavePlansToDot(part.m_Plans, "plans_in_part_with_single_node");
    const auto& plan = part.GetPlan(0);
    std::stringstream str;
    SaveOpGraphToDot(plan.m_OpGraph, str, DetailLevel::Low);
    std::string s = str.str();
    s.erase(std::remove(s.begin(), s.end(), '\n'), s.end());
    bool result = std::regex_match(s, m, re);
    REQUIRE(result);
    REQUIRE(plan.m_OpGraph.GetOps().size() == 2);
    REQUIRE(plan.m_OpGraph.GetBuffers().size() == 4);
    REQUIRE(plan.m_InputMappings.size() == 1);
    REQUIRE(plan.m_OutputMappings.size() == 1);
}

TEST_CASE("PlanGenerator: Generate plans from a part with two fused nodes")
{
    // Given
    const EstimationOptions estOpt;
    CompilationOptions compOpt      = GetDefaultCompilationOptions();
    compOpt.m_DisableWinograd       = GENERATE(false, true);
    const HardwareCapabilities caps = GetEthosN77HwCapabilities();
    Graph g;
    auto part = BuildSinglePartWithTwoNodes(g, estOpt, compOpt, caps);
    std::string expected =
        R"(digraph SupportLibraryGraph
\{
.*
Buffer_.* -> MceOp_.*\[ label="Input 0"\]
MceOp_.* -> Buffer_.*
Buffer_.* -> MceOp_.*
MceOp_.* -> Buffer_.*
Buffer_.* -> DmaOp_.*
DmaOp_.* -> Buffer_.*
Buffer_.* -> MceOp_.*\[ label="Input 1"\]
\{.*\}?
\}.*
)";
    expected.erase(std::remove(expected.begin(), expected.end(), '\n'), expected.end());
    std::regex re(expected);
    std::smatch m;

    // When
    part.CreatePlans();

    // Then
    REQUIRE(part.m_Plans.size() == 19);
    SavePlansToDot(part.m_Plans, "plans_in_part_with_two_fused_nodes");
    const auto& plan = part.GetPlan(0);
    std::stringstream str;
    SaveOpGraphToDot(plan.m_OpGraph, str, DetailLevel::Low);
    std::string s = str.str();
    s.erase(std::remove(s.begin(), s.end(), '\n'), s.end());
    bool result = std::regex_match(s, m, re);
    REQUIRE(result);
    REQUIRE(plan.m_OpGraph.GetOps().size() == 3);
    REQUIRE(plan.m_OpGraph.GetBuffers().size() == 5);
    REQUIRE(plan.m_InputMappings.size() == 1);
    REQUIRE(plan.m_OutputMappings.size() == 1);
}

TEST_CASE("PlanGenerator:FuseOnlyPleNode")
{
    // Graph with FuseOnlyPleNode
    const EstimationOptions estOpt;
    const CompilationOptions compOpt = GetDefaultCompilationOptions();
    const HardwareCapabilities caps  = GetEthosN77HwCapabilities();
    Graph g;
    auto part = BuildPartWithFuseOnlyPle(g, estOpt, compOpt, caps);

    part.CreatePlans();
    SavePlansToDot(part.m_Plans, "plans_part_fuseonlyple");

    const auto& plan1 = part.GetPlan(0);
    REQUIRE(plan1.m_OpGraph.GetBuffers().size() == 2);
    auto buffers1 = plan1.m_OpGraph.GetBuffers();
    auto ops1     = plan1.m_OpGraph.GetOps();
    REQUIRE(buffers1[0]->m_Location == Location::PleInputSram);
    REQUIRE(buffers1[1]->m_Location == Location::Sram);
    REQUIRE(ops1.size() == 1);
    REQUIRE(dynamic_cast<PleOp*>(ops1.back()) != nullptr);

    const auto& plan2 = part.GetPlan(1);
    REQUIRE(plan2.m_OpGraph.GetBuffers().size() == 5);
    auto buffers2 = plan2.m_OpGraph.GetBuffers();
    auto ops2     = plan2.m_OpGraph.GetOps();
    REQUIRE(buffers2[0]->m_Location == Location::Sram);
    REQUIRE(buffers2[1]->m_Location == Location::PleInputSram);
    REQUIRE(buffers2[2]->m_Location == Location::Dram);
    REQUIRE(buffers2[3]->m_Location == Location::Sram);
    REQUIRE(buffers2[4]->m_Location == Location::Sram);

    REQUIRE(ops2.size() == 3);
    REQUIRE(dynamic_cast<MceOp*>(ops2[0]) != nullptr);
    REQUIRE(dynamic_cast<PleOp*>(ops2[2]) != nullptr);
}

TEST_CASE("PlanGenerator:MceOperationNode")
{
    // Graph with FuseOnlyPleNode
    const EstimationOptions estOpt;
    CompilationOptions compOpt      = GetDefaultCompilationOptions();
    compOpt.m_DisableWinograd       = GENERATE(false, true);
    const HardwareCapabilities caps = GetEthosN77HwCapabilities();
    Graph g;
    auto part = BuildSinglePartWithOneNode(g, estOpt, compOpt, caps);

    part.CreatePlans();
    SavePlansToDot(part.m_Plans, "plans_part_mceoperation");

    const auto& plan1 = part.GetPlan(0);
    REQUIRE(plan1.m_OpGraph.GetBuffers().size() == 4);
    auto buffers1 = plan1.m_OpGraph.GetBuffers();
    auto ops1     = plan1.m_OpGraph.GetOps();
    REQUIRE(buffers1[0]->m_Location == Location::Sram);
    REQUIRE(buffers1[1]->m_Location == Location::PleInputSram);
    REQUIRE(ops1.size() == 2);
    REQUIRE(dynamic_cast<MceOp*>(ops1.front()) != nullptr);

    const auto& plan2 = part.GetPlan(1);
    REQUIRE(plan2.m_OpGraph.GetBuffers().size() == 5);
    const OpGraph::BufferList& buffers2 = plan2.m_OpGraph.GetBuffers();
    auto ops2                           = plan2.m_OpGraph.GetOps();
    REQUIRE(buffers2[0]->m_Location == Location::Sram);
    REQUIRE(buffers2[1]->m_Location == Location::PleInputSram);
    REQUIRE(buffers2[2]->m_Location == Location::Dram);
    REQUIRE(buffers2[3]->m_Location == Location::Sram);
    REQUIRE(buffers2[4]->m_Location == Location::Sram);

    REQUIRE(ops2.size() == 3);
    REQUIRE(dynamic_cast<MceOp*>(ops2[0]) != nullptr);
    REQUIRE(dynamic_cast<PleOp*>(ops2[2]) != nullptr);

    // Check the weights have been encoded properly
    REQUIRE(buffers2[2]->m_Format == CascadingBufferFormat::WEIGHT);
    REQUIRE(buffers2[2]->m_EncodedWeights != nullptr);
    REQUIRE(buffers2[2]->m_EncodedWeights->m_Data.size() > 0);
    REQUIRE(buffers2[2]->m_SizeInBytes == buffers2[2]->m_EncodedWeights->m_Data.size());
    REQUIRE(buffers2[3]->m_SizeInBytes == buffers2[2]->m_EncodedWeights->m_MaxSize);
}

TEST_CASE("PlanGenerator: Generate plans from a part with single format conversion node")
{
    auto assertPlan = [](const Plan& plan) -> void {
        REQUIRE(plan.m_OpGraph.GetOps().size() == 1);
        REQUIRE(IsObjectOfType<DmaOp>(plan.m_OpGraph.GetOps()[0]));
        REQUIRE(plan.m_OpGraph.GetBuffers().size() == 2);
        REQUIRE(plan.m_InputMappings.size() == 1);
        REQUIRE(plan.m_OutputMappings.size() == 1);
    };

    // Given
    const EstimationOptions estOpt;
    const CompilationOptions compOpt = GetDefaultCompilationOptions();
    const HardwareCapabilities caps  = GetEthosN77HwCapabilities();
    Graph g;
    auto part         = BuildPartWithLeadingFormatConversionNode(g, estOpt, compOpt, caps);
    const auto& edges = g.GetEdges();
    const auto& nodes = g.GetNodes();
    REQUIRE(edges.size() == 2);
    REQUIRE(nodes.size() == 3);

    // When
    part.CreatePlans();

    // Then
    REQUIRE(part.m_Plans.size() == 8);
    SavePlansToDot(part.m_Plans, "plans_in_part_with_leading_format_conversion_node");

    const auto input  = edges[0].get();
    const auto output = nodes[1].get();
    AssertPart(part, input, output, assertPlan);

    RequireInputBuffer(part, input, CascadingBufferFormat::NHWCB, Location::Sram, 1, { 1, 32, 32, 4 }, { 1, 8, 8, 16 });
    RequireOutputBuffer(part, output, CascadingBufferFormat::NHWC, Location::Dram, 0, { 1, 64, 64, 1 }, { 0, 0, 0, 0 });

    RequireInputBuffer(part, input, CascadingBufferFormat::NHWCB, Location::Sram, 2, { 1, 32, 32, 4 }, { 1, 8, 8, 16 });
    RequireOutputBuffer(part, output, CascadingBufferFormat::NHWC, Location::Dram, 0, { 1, 64, 64, 1 }, { 0, 0, 0, 0 });

    RequireInputBuffer(part, input, CascadingBufferFormat::NHWCB, Location::Sram, 1, { 1, 32, 32, 4 },
                       { 1, 32, 32, 16 });
    RequireOutputBuffer(part, output, CascadingBufferFormat::NHWC, Location::Dram, 0, { 1, 64, 64, 1 }, { 0, 0, 0, 0 });

    RequireInputBuffer(part, input, CascadingBufferFormat::NHWCB, Location::Sram, 1, { 1, 32, 32, 4 },
                       { 1, 32, 32, 16 });
    RequireOutputBuffer(part, output, CascadingBufferFormat::NHWC, Location::VirtualSram, 1, { 1, 64, 64, 1 },
                        { 1, 64, 64, 16 });
}

TEST_CASE("PlanGenerator: Generate plans from a part with trailing format conversion node")
{
    auto assertPlan = [](const Plan& plan) -> void {
        REQUIRE(plan.m_OpGraph.GetOps().size() == 1);
        REQUIRE(IsObjectOfType<DmaOp>(plan.m_OpGraph.GetOps()[0]));
        REQUIRE(plan.m_OpGraph.GetBuffers().size() == 2);
        REQUIRE(plan.m_InputMappings.size() == 1);
        REQUIRE(plan.m_OutputMappings.size() == 1);
    };

    // Given
    const EstimationOptions estOpt;
    const CompilationOptions compOpt = GetDefaultCompilationOptions();
    const HardwareCapabilities caps  = GetEthosN77HwCapabilities();
    Graph g;
    auto part         = BuildPartWithTrailingFormatConversionNode(g, estOpt, compOpt, caps);
    const auto& edges = g.GetEdges();
    const auto& nodes = g.GetNodes();
    REQUIRE(edges.size() == 2);
    REQUIRE(nodes.size() == 3);

    // When
    part.CreatePlans();

    // Then
    REQUIRE(part.m_Plans.size() == 11);
    SavePlansToDot(part.m_Plans, "plans_in_part_with_trailing_format_conversion_node");

    const auto input  = edges[0].get();
    const auto output = nodes[1].get();
    AssertPart(part, input, output, assertPlan);

    RequireInputBuffer(part, input, CascadingBufferFormat::NHWC, Location::Dram, 0, { 1, 32, 32, 4 }, { 0, 0, 0, 0 });
    RequireOutputBuffer(part, output, CascadingBufferFormat::NHWCB, Location::Sram, 1, { 1, 64, 64, 1 },
                        { 1, 8, 8, 16 });

    RequireInputBuffer(part, input, CascadingBufferFormat::NHWC, Location::Dram, 0, { 1, 32, 32, 4 }, { 0, 0, 0, 0 });
    RequireOutputBuffer(part, output, CascadingBufferFormat::NHWCB, Location::Sram, 2, { 1, 64, 64, 1 },
                        { 1, 8, 8, 16 });

    RequireInputBuffer(part, input, CascadingBufferFormat::NHWC, Location::Dram, 0, { 1, 32, 32, 4 }, { 0, 0, 0, 0 });
    RequireOutputBuffer(part, output, CascadingBufferFormat::NHWCB, Location::Sram, 3, { 1, 64, 64, 1 },
                        { 1, 8, 8, 16 });

    RequireInputBuffer(part, input, CascadingBufferFormat::NHWC, Location::Dram, 0, { 1, 32, 32, 4 }, { 0, 0, 0, 0 });
    RequireOutputBuffer(part, output, CascadingBufferFormat::NHWCB, Location::Sram, 1, { 1, 64, 64, 1 },
                        { 1, 8, 16, 16 });

    RequireInputBuffer(part, input, CascadingBufferFormat::NHWC, Location::Dram, 0, { 1, 32, 32, 4 }, { 0, 0, 0, 0 });
    RequireOutputBuffer(part, output, CascadingBufferFormat::NHWCB, Location::Sram, 1, { 1, 64, 64, 1 },
                        { 1, 16, 8, 16 });

    RequireInputBuffer(part, input, CascadingBufferFormat::NHWC, Location::Dram, 0, { 1, 32, 32, 4 }, { 0, 0, 0, 0 });
    RequireOutputBuffer(part, output, CascadingBufferFormat::NHWCB, Location::Sram, 1, { 1, 64, 64, 1 },
                        { 1, 64, 64, 16 });

    RequireInputBuffer(part, input, CascadingBufferFormat::NHWC, Location::VirtualSram, 1, { 1, 32, 32, 4 },
                       { 1, 32, 32, 16 });
}

TEST_CASE("PlanGenerator: Generate plans from a part with reinterpret node")
{
    auto assertPlan = [](const Plan& plan) -> void {
        REQUIRE(plan.m_OpGraph.GetOps().size() == 1);
        REQUIRE(IsObjectOfType<DummyOp>(plan.m_OpGraph.GetOps()[0]));
        REQUIRE(plan.m_OpGraph.GetBuffers().size() == 2);
        REQUIRE(plan.m_InputMappings.size() == 1);
        REQUIRE(plan.m_OutputMappings.size() == 1);
    };

    // Given
    const EstimationOptions estOpt;
    const CompilationOptions compOpt = GetDefaultCompilationOptions();
    const HardwareCapabilities caps  = GetEthosN77HwCapabilities();
    Graph g;
    auto part         = BuildPartWithReinterpretNode(g, estOpt, compOpt, caps);
    const auto& edges = g.GetEdges();
    const auto& nodes = g.GetNodes();
    REQUIRE(edges.size() == 2);
    REQUIRE(nodes.size() == 3);

    // When
    part.CreatePlans();

    // Then
    REQUIRE(part.m_Plans.size() == 2);
    SavePlansToDot(part.m_Plans, "plans_in_part_with_reinterpret_node");

    const auto input  = edges[0].get();
    const auto output = nodes[1].get();
    AssertPart(part, input, output, assertPlan);

    RequireInputBuffer(part, input, CascadingBufferFormat::NHWC, Location::Dram, 0, { 1, 32, 32, 4 }, { 0, 0, 0, 0 });
    RequireInputBuffer(part, input, CascadingBufferFormat::NHWC, Location::VirtualSram, 1, { 1, 32, 32, 4 },
                       { 1, 32, 32, 16 });
    RequireOutputBuffer(part, output, CascadingBufferFormat::NHWC, Location::Dram, 0, { 1, 64, 64, 1 }, { 0, 0, 0, 0 });
    RequireOutputBuffer(part, output, CascadingBufferFormat::NHWC, Location::VirtualSram, 1, { 1, 64, 64, 1 },
                        { 1, 64, 64, 16 });
}

Part BuildPartWithMceNodeStride(Graph& g,
                                TS inputShape,
                                TS outputShape,
                                TS weightShape,
                                ethosn::command_stream::MceOperation op,
                                Stride stride,
                                const EstimationOptions& estOpt,
                                const CompilationOptions& compOpt,
                                const HardwareCapabilities& caps)
{
    TI weightInfo           = weightShape;
    weightInfo.m_DataFormat = op == ethosn::command_stream::MceOperation::DEPTHWISE_CONVOLUTION
                                  ? ethosn::support_library::DataFormat::HWIM
                                  : ethosn::support_library::DataFormat::HWIO;
    weightInfo.m_QuantizationInfo = { 0, 0.9f };
    std::vector<uint8_t> weightData(utils::GetNumElements(weightShape), 1);

    TI biasInfo = TensorShape{ 1, 1, 1, outputShape[3] };
    std::vector<int32_t> biasData(outputShape[3], 0);

    auto in      = g.CreateAndAddNode<InputNode>(inputShape, std::set<uint32_t>{ 1 });
    auto mceNode = g.CreateAndAddNode<MceOperationNode>(
        inputShape, outputShape, ethosn::support_library::DataType::UINT8_QUANTIZED, QI(), weightInfo, weightData,
        biasInfo, biasData, stride, 1, 1, op, CompilerDataFormat::NHWCB, std::set<uint32_t>{ 1 });
    auto out =
        g.CreateAndAddNode<OutputNode>(ethosn::support_library::DataType::UINT8_QUANTIZED, std::set<uint32_t>(), 0);

    g.Connect(in, mceNode, 0);
    g.Connect(mceNode, out, 0);

    Part part(estOpt, compOpt, caps);
    part.m_SubGraph.push_back(mceNode);

    return part;
}

Part BuildPartWithMceNode(Graph& g,
                          TS inputShape,
                          TS outputShape,
                          TS weightShape,
                          ethosn::command_stream::MceOperation op,
                          const EstimationOptions& estOpt,
                          const CompilationOptions& compOpt,
                          const HardwareCapabilities& caps)
{
    return BuildPartWithMceNodeStride(g, inputShape, outputShape, weightShape, op, Stride(1, 1), estOpt, compOpt, caps);
}

TEST_CASE("PlanGenerator: FuseOnly")
{
    const EstimationOptions estOpt;
    const CompilationOptions compOpt = GetDefaultCompilationOptions();
    const HardwareCapabilities caps  = GetEthosN77HwCapabilities();
    Graph g;

    auto in      = g.CreateAndAddNode<InputNode>(TS{ 1, 224, 224, 64 }, std::set<uint32_t>{ 1 });
    auto pleNode = g.CreateAndAddNode<FuseOnlyPleOperationNode>(
        TS{ 1, 112, 112, 256 }, ethosn::support_library::DataType::UINT8_QUANTIZED, QI(),
        ethosn::command_stream::PleOperation::INTERLEAVE_2X2_2_2, CompilerDataFormat::NHWCB,
        utils::ShapeMultiplier{ { 1, 2 }, { 1, 2 }, 4 }, std::set<uint32_t>{ 1 });
    auto out =
        g.CreateAndAddNode<OutputNode>(ethosn::support_library::DataType::UINT8_QUANTIZED, std::set<uint32_t>(), 0);

    g.Connect(in, pleNode, 0);
    g.Connect(pleNode, out, 0);

    Part part(estOpt, compOpt, caps);
    part.m_SubGraph.push_back(pleNode);

    part.CreatePlans();

    SavePlansToDot(part.m_Plans, "plans_in_part_with_fuse_only");

    // Ensure there is a plan with the correct stripes and the has an mce and ple op.
    {
        TS inputStripe{ 1, 16, 224, 64 };
        uint32_t numInputStripes = 1;
        TS outputStripe{ 1, 8, 112, 256 };
        uint32_t numOutputStripes = 1;

        auto planIndex =
            GetPlanIndexContainingStripes(part.m_Plans, inputStripe, numInputStripes, outputStripe, numOutputStripes);
        REQUIRE(planIndex.first);
        auto& plan      = part.m_Plans[planIndex.second];
        auto ops        = plan->m_OpGraph.GetOps();
        auto foundMceOp = utils::FindIndexIf(ops, [](Op* op) { return IsObjectOfType<MceOp>(op); });
        auto foundPleOp = utils::FindIndexIf(ops, [](Op* op) { return IsObjectOfType<PleOp>(op); });
        REQUIRE(foundMceOp.first);
        REQUIRE(foundPleOp.first);
    }
    // Ensure there is a plan with the correct ratio of input and output stripe sizes.
    {
        TS inputStripe{ 1, 16, 16, 16 };
        uint32_t numInputStripes = 1;
        TS outputStripe{ 1, 8, 8, 64 };
        uint32_t numOutputStripes = 1;

        // Ensure there is a plan with the correct stripes and the has an mce and ple op.
        REQUIRE(ContainsPlanWithStripes(part.m_Plans, inputStripe, numInputStripes, outputStripe, numOutputStripes));
    }
}

TEST_CASE("PlanGenerator: Mobilenet N57 Conv", "[slow]")
{
    const EstimationOptions estOpt;
    const CompilationOptions compOpt = GetDefaultCompilationOptions();
    const HardwareCapabilities caps  = GetEthosN57HwCapabilities();
    Graph g;

    TS inputShape{ 1, 14, 14, 256 };
    TS outputShape{ 1, 14, 14, 512 };
    TS weightShape{ 1, 1, 256, 512 };

    Part part = BuildPartWithMceNode(g, inputShape, outputShape, weightShape,
                                     ethosn::command_stream::MceOperation::CONVOLUTION, estOpt, compOpt, caps);
    part.CreatePlans();

    SavePlansToDot(part.m_Plans, "plans_mobilenet_conv");

    {
        TS inputStripe{ 1, 16, 16, 256 };
        uint32_t numInputStripes = 1;
        TS outputStripe{ 1, 16, 16, 512 };
        uint32_t numOutputStripes = 1;
        REQUIRE(ContainsPlanWithStripes(part.m_Plans, inputStripe, numInputStripes, outputStripe, numOutputStripes));
    }
}

TEST_CASE("PlanGenerator: Mobilenet N57 Depthwise")
{
    const EstimationOptions estOpt;
    const CompilationOptions compOpt = GetDefaultCompilationOptions();
    const HardwareCapabilities caps  = GetEthosN57HwCapabilities();
    Graph g;

    TS inputShape{ 1, 14, 14, 512 };
    TS outputShape{ 1, 14, 14, 512 };
    TS weightShape{ 1, 1, 512, 1 };
    Part part =
        BuildPartWithMceNode(g, inputShape, outputShape, weightShape,
                             ethosn::command_stream::MceOperation::DEPTHWISE_CONVOLUTION, estOpt, compOpt, caps);
    part.CreatePlans();

    SavePlansToDot(part.m_Plans, "plans_mobilenet_depthwise");
    {
        TS inputStripe{ 1, 16, 16, 512 };
        uint32_t numInputStripes = 1;
        TS outputStripe{ 1, 16, 16, 512 };
        uint32_t numOutputStripes = 1;
        REQUIRE(ContainsPlanWithStripes(part.m_Plans, inputStripe, numInputStripes, outputStripe, numOutputStripes));
    }
}

TEST_CASE("PlanGenerator: Mobilenet N57 1024", "[slow]")
{
    const EstimationOptions estOpt;
    const CompilationOptions compOpt = GetDefaultCompilationOptions();
    const HardwareCapabilities caps  = GetEthosN57HwCapabilities();
    Graph g;

    TS inputShape{ 1, 7, 7, 512 };
    TS outputShape{ 1, 7, 7, 1024 };
    TS weightShape{ 1, 1, 512, 1024 };
    Part part = BuildPartWithMceNode(g, inputShape, outputShape, weightShape,
                                     ethosn::command_stream::MceOperation::CONVOLUTION, estOpt, compOpt, caps);
    part.CreatePlans();

    SavePlansToDot(part.m_Plans, "plans_mobilenet_1024");
    {
        TS inputStripe{ 1, 8, 8, 512 };
        uint32_t numInputStripes = 1;
        TS outputStripe{ 1, 8, 8, 16 };
        uint32_t numOutputStripes = 2;
        REQUIRE(ContainsPlanWithStripes(part.m_Plans, inputStripe, numInputStripes, outputStripe, numOutputStripes));
    }
}

TEST_CASE("PlanGenerator:BlockConfig")
{
    // Graph with FuseOnlyPleNode
    const EstimationOptions estOpt;
    const CompilationOptions compOpt = GetDefaultCompilationOptions();
    const HardwareCapabilities caps  = GetEthosN78HwCapabilities();
    Graph g;
    auto part = BuildPartWithFuseOnlyPle(g, estOpt, compOpt, caps);

    part.CreatePlans();
    SavePlansToDot(part.m_Plans, "plans_part_blockconfig");

    const auto& plan1 = part.GetPlan(0);
    REQUIRE(plan1.m_OpGraph.GetBuffers().size() == 2);
    auto ops1 = plan1.m_OpGraph.GetOps();
    REQUIRE(ops1.size() == 1);
    auto pleOp = dynamic_cast<PleOp*>(ops1.back());
    REQUIRE(pleOp != nullptr);
    REQUIRE(pleOp->m_Op == ethosn::command_stream::PleOperation::INTERLEAVE_2X2_2_2);
    REQUIRE(pleOp->m_BlockConfig == ethosn::command_stream::BlockConfig{ 16U, 16U });

    const auto& plan2 = part.GetPlan(1);
    auto ops2         = plan2.m_OpGraph.GetOps();
    REQUIRE(ops2.size() == 3);
    auto mceOp = dynamic_cast<MceOp*>(ops2[0]);
    REQUIRE(mceOp != nullptr);
    REQUIRE(mceOp->m_BlockConfig == ethosn::command_stream::BlockConfig{ 16U, 16U });
    pleOp = dynamic_cast<PleOp*>(ops2[2]);
    REQUIRE(pleOp != nullptr);
    REQUIRE(pleOp->m_Op == ethosn::command_stream::PleOperation::INTERLEAVE_2X2_2_2);
    REQUIRE(pleOp->m_BlockConfig == ethosn::command_stream::BlockConfig{ 16U, 16U });
}

TEST_CASE("PlanGenerator:Winograd")
{
    const EstimationOptions estOpt;
    const CompilationOptions compOpt = GetDefaultCompilationOptions();
    const HardwareCapabilities caps  = GetEthosN78HwCapabilities();
    Graph g;

    TS tsIn   = { 1, 32, 32, 3 };
    TS tsOut  = { 1, 64, 64, 1 };
    auto in   = CreateAndAddInputNode(g, tsIn);
    auto node = CreateAndAddMceOperationNode(g, tsOut, 3, 3);
    auto out  = CreateAndAddOutputNode(g);

    // Graph must be connected to be valid
    g.Connect(in, node, 0);
    g.Connect(node, out, 0);

    Part part(estOpt, compOpt, caps);
    part.m_SubGraph.push_back(node);
    part.CreatePlans();
    SavePlansToDot(part.m_Plans, "plans_part_winograd");

    for (const auto& plan : part.m_Plans)
    {
        auto ops = plan->m_OpGraph.GetOps();
        REQUIRE(!ops.empty());
        auto mceOp = dynamic_cast<MceOp*>(ops[0]);
        REQUIRE(mceOp != nullptr);
        if (mceOp->m_BlockConfig == ethosn::command_stream::BlockConfig{ 16U, 16U })
        {
            REQUIRE(mceOp->m_Algo == CompilerMceAlgorithm::Direct);
        }
        if (mceOp->m_BlockConfig == ethosn::command_stream::BlockConfig{ 8U, 8U })
        {
            REQUIRE(mceOp->m_Algo == CompilerMceAlgorithm::Winograd);
        }
    }
}
