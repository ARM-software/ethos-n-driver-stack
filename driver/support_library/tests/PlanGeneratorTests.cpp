//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "CapabilitiesInternal.hpp"
#include "GlobalParameters.hpp"
#include "GraphNodes.hpp"
#include "TestUtils.hpp"
#include "Utils.hpp"
#include "cascading/Cascading.hpp"
#include "cascading/Visualisation.hpp"
#include "cascading/WeightEncoderCache.hpp"
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

Node* CreateAndAddMceOperationNode(
    Graph& g, const TS& tsIn, const TS& tsOut, const uint32_t kH, const uint32_t kW, const Stride& stride)
{
    const std::vector<uint8_t> weights(kH * kW * utils::GetChannels(tsIn) * utils::GetChannels(tsOut), 1);
    const std::vector<int32_t> bias(utils::GetChannels(tsOut), 0);
    return g.CreateAndAddNode<MceOperationNode>(
        TS(), tsOut, sl::DataType::UINT8_QUANTIZED, QI(),
        TI({ kH, kW, utils::GetChannels(tsIn), utils::GetChannels(tsOut) },
           ethosn::support_library::DataType::UINT8_QUANTIZED, ethosn::support_library::DataFormat::HWIO,
           QuantizationInfo(0, 0.9f)),
        weights, TI({ 1, 1, 1, utils::GetChannels(tsOut) }), bias, stride, 0, 0,
        ethosn::command_stream::MceOperation::CONVOLUTION, CompilerDataFormat::NHWCB, std::set<uint32_t>{ 1 });
}

Node* CreateAndAddMceOperationNode(
    Graph& g, const TS& tsOut, const uint32_t kH, const uint32_t kW, const Stride& stride)
{
    return CreateAndAddMceOperationNode(g, TS({ 1, 1, 1, 1 }), tsOut, kH, kW, stride);
}

Node* CreateAndAddMceOperationNode(Graph& g)
{
    return CreateAndAddMceOperationNode(g, TS(), 1, 1, Stride());
}

Node* CreateAndAddMceOperationNode(Graph& g, const TS& tsOut)
{
    return CreateAndAddMceOperationNode(g, tsOut, 1, 1, Stride());
}

Node* CreateAndAddMceOperationNode(Graph& g, const TS& tsOut, const Stride& stride)
{
    return CreateAndAddMceOperationNode(g, tsOut, 1, 1, stride);
}

Node* CreateAndAddMceOperationNode(Graph& g, const TS& tsIn, const TS& tsOut, const uint32_t kH, const uint32_t kW)
{
    return CreateAndAddMceOperationNode(g, tsIn, tsOut, kH, kW, Stride());
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

    Part part(0, estOpt, compOpt, caps);
    part.m_SubGraph.push_back(node);
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

    Part part(0, estOpt, compOpt, caps);
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

    Part part(0, estOpt, compOpt, caps);
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

    Part part(0, estOpt, compOpt, caps);
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

    Part part(0, estOpt, compOpt, caps);
    part.m_SubGraph.push_back(node);
    return part;
}

void AssertPart(const Plans& plans,
                const Edge* input,
                const Node* output,
                const std::function<void(const Plan&)>& assertPlan)
{
    for (const auto& plan : plans)
    {
        assertPlan(*plan);
        auto inputBuffer         = plan->GetInputBuffer(input);
        auto outputBuffer        = plan->GetOutputBuffer(output);
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

void RequireInputBuffer(const Plans& plans,
                        const Edge* edge,
                        CascadingBufferFormat format,
                        Location location,
                        uint32_t stripes,
                        TensorShape ts,
                        StripeShape ss)
{
    for (const auto& plan : plans)
    {
        const Buffer* buffer = plan->GetInputBuffer(edge);
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
    for (const auto& plan : plans)
    {
        const Buffer* buffer = plan->GetInputBuffer(edge);
        oss << ", buffer: " << buffer->m_DebugTag << ", tensor: ";
        oss << buffer->m_TensorShape << ", stripe: ";
        oss << buffer->m_StripeShape;
        oss << ", num stripes: " << buffer->m_NumStripes << ", location: " << static_cast<int>(buffer->m_Location)
            << ", format: " << static_cast<int>(buffer->m_Format) << "\n";
    }
    INFO(oss.str());
    REQUIRE(false);
}

void RequireOutputBuffer(const Plans& plans,
                         const Node* node,
                         CascadingBufferFormat format,
                         Location location,
                         uint32_t stripes,
                         TensorShape ts,
                         StripeShape ss)
{
    for (const auto& plan : plans)
    {
        const Buffer* buffer = plan->GetOutputBuffer(node);
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
    for (const auto& plan : plans)
    {
        const Buffer* buffer = plan->GetOutputBuffer(node);
        oss << ", buffer: " << buffer->m_DebugTag << ", tensor: ";
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

std::vector<size_t> GetPlanIndexContainingStripes(const Plans& plans,
                                                  const TensorShape& inputStripe,
                                                  const uint32_t numInputStripes,
                                                  const TensorShape& outputStripe,
                                                  const uint32_t numOutputStripes)
{
    std::vector<size_t> res;
    for (uint32_t i = 0; i < plans.size(); ++i)
    {
        bool containsInputStripe  = ContainsInputStripe(plans[i]->m_InputMappings, inputStripe, numInputStripes);
        bool containsOutputStripe = ContainsOutputStripe(plans[i]->m_OutputMappings, outputStripe, numOutputStripes);
        if (containsInputStripe && containsOutputStripe)
        {
            res.push_back(i);
        }
    }
    return res;
}

bool ContainsPlanWithStripes(const Plans& plans,
                             const TensorShape& inputStripe,
                             const uint32_t numInputStripes,
                             const TensorShape& outputStripe,
                             const uint32_t numOutputStripes)
{
    return !GetPlanIndexContainingStripes(plans, inputStripe, numInputStripes, outputStripe, numOutputStripes).empty();
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
    const CompilationOptions compOpt;
    const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
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
    const CompilationOptions compOpt;
    const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
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
    const CompilationOptions compOpt;
    const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
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

TEST_CASE("PlanGenerator:FuseOnlyPleNode")
{
    // Graph with FuseOnlyPleNode
    const EstimationOptions estOpt;
    const CompilationOptions compOpt;
    const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
    Graph g;
    auto part = BuildPartWithFuseOnlyPle(g, estOpt, compOpt, caps);

    Plans plans = part.GetPlans();
    SavePlansToDot(plans, "plans_part_fuseonlyple");

    REQUIRE(plans.size() == 15);

    constexpr size_t planId = 11;
    const auto& plan1       = plans[planId];
    REQUIRE(plan1->m_OpGraph.GetBuffers().size() == 2);
    auto buffers1 = plan1->m_OpGraph.GetBuffers();
    auto ops1     = plan1->m_OpGraph.GetOps();
    REQUIRE(buffers1[0]->m_Location == Location::PleInputSram);
    REQUIRE(buffers1[1]->m_Location == Location::Sram);
    REQUIRE(ops1.size() == 1);
    REQUIRE(dynamic_cast<PleOp*>(ops1.back()) != nullptr);

    const auto& plan2 = plans[1];
    REQUIRE(plan2->m_OpGraph.GetBuffers().size() == 5);
    auto buffers2 = plan2->m_OpGraph.GetBuffers();
    auto ops2     = plan2->m_OpGraph.GetOps();
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
    CompilationOptions compOpt;
    compOpt.m_DisableWinograd       = GENERATE(false, true);
    const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
    Graph g;
    auto part = BuildSinglePartWithOneNode(g, estOpt, compOpt, caps);

    Plans plans = part.GetPlans();
    SavePlansToDot(plans, "plans_part_mceoperation");

    const auto& plan1 = plans[0];
    REQUIRE(plan1->m_OpGraph.GetBuffers().size() == 5);
    auto buffers1 = plan1->m_OpGraph.GetBuffers();
    auto ops1     = plan1->m_OpGraph.GetOps();
    REQUIRE(buffers1[0]->m_Location == Location::PleInputSram);
    REQUIRE(buffers1[1]->m_Location == Location::Sram);
    REQUIRE(ops1.size() == 3);
    REQUIRE(dynamic_cast<MceOp*>(ops1.front()) != nullptr);

    const auto& plan2 = plans[2];
    REQUIRE(plan2->m_OpGraph.GetBuffers().size() == 5);
    const OpGraph::BufferList& buffers2 = plan2->m_OpGraph.GetBuffers();
    auto ops2                           = plan2->m_OpGraph.GetOps();
    REQUIRE(buffers2[0]->m_Location == Location::PleInputSram);
    REQUIRE(buffers2[1]->m_Location == Location::Sram);
    REQUIRE(buffers2[2]->m_Location == Location::Dram);
    REQUIRE(buffers2[3]->m_Location == Location::Sram);

    REQUIRE(ops2.size() == 3);
    REQUIRE(dynamic_cast<MceOp*>(ops2[0]) != nullptr);
    REQUIRE(dynamic_cast<DmaOp*>(ops2[1]) != nullptr);
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
    const CompilationOptions compOpt;
    const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
    Graph g;
    auto part         = BuildPartWithLeadingFormatConversionNode(g, estOpt, compOpt, caps);
    const auto& edges = g.GetEdges();
    const auto& nodes = g.GetNodes();
    REQUIRE(edges.size() == 2);
    REQUIRE(nodes.size() == 3);

    // When
    Plans plans = part.GetPlans();
    SavePlansToDot(plans, "plans_in_part_with_leading_format_conversion_node");

    // Then
    REQUIRE(plans.size() == 18);

    const auto input  = edges[0].get();
    const auto output = nodes[1].get();
    AssertPart(plans, input, output, assertPlan);

    RequireInputBuffer(plans, input, CascadingBufferFormat::NHWCB, Location::Sram, 1, { 1, 32, 32, 4 },
                       { 1, 8, 8, 16 });
    RequireOutputBuffer(plans, output, CascadingBufferFormat::NHWC, Location::Dram, 0, { 1, 64, 64, 1 },
                        { 0, 0, 0, 0 });

    RequireInputBuffer(plans, input, CascadingBufferFormat::NHWCB, Location::Sram, 2, { 1, 32, 32, 4 },
                       { 1, 8, 8, 16 });
    RequireOutputBuffer(plans, output, CascadingBufferFormat::NHWC, Location::Dram, 0, { 1, 64, 64, 1 },
                        { 0, 0, 0, 0 });

    RequireInputBuffer(plans, input, CascadingBufferFormat::NHWCB, Location::Sram, 1, { 1, 32, 32, 4 },
                       { 1, 32, 32, 16 });
    RequireOutputBuffer(plans, output, CascadingBufferFormat::NHWC, Location::Dram, 0, { 1, 64, 64, 1 },
                        { 0, 0, 0, 0 });

    RequireInputBuffer(plans, input, CascadingBufferFormat::NHWCB, Location::Sram, 1, { 1, 32, 32, 4 },
                       { 1, 32, 32, 16 });
    RequireOutputBuffer(plans, output, CascadingBufferFormat::NHWC, Location::VirtualSram, 1, { 1, 64, 64, 1 },
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
    const CompilationOptions compOpt;
    const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
    Graph g;
    auto part         = BuildPartWithTrailingFormatConversionNode(g, estOpt, compOpt, caps);
    const auto& edges = g.GetEdges();
    const auto& nodes = g.GetNodes();
    REQUIRE(edges.size() == 2);
    REQUIRE(nodes.size() == 3);

    // When
    Plans plans = part.GetPlans();

    // Then
    SavePlansToDot(plans, "plans_in_part_with_trailing_format_conversion_node");
    REQUIRE(plans.size() == 22);

    const auto input  = edges[0].get();
    const auto output = nodes[1].get();
    AssertPart(plans, input, output, assertPlan);

    RequireInputBuffer(plans, input, CascadingBufferFormat::NHWC, Location::Dram, 0, { 1, 32, 32, 4 }, { 0, 0, 0, 0 });
    RequireOutputBuffer(plans, output, CascadingBufferFormat::NHWCB, Location::Sram, 1, { 1, 64, 64, 1 },
                        { 1, 8, 8, 16 });

    RequireInputBuffer(plans, input, CascadingBufferFormat::NHWC, Location::Dram, 0, { 1, 32, 32, 4 }, { 0, 0, 0, 0 });
    RequireOutputBuffer(plans, output, CascadingBufferFormat::NHWCB, Location::Sram, 2, { 1, 64, 64, 1 },
                        { 1, 8, 8, 16 });

    RequireInputBuffer(plans, input, CascadingBufferFormat::NHWC, Location::Dram, 0, { 1, 32, 32, 4 }, { 0, 0, 0, 0 });
    RequireOutputBuffer(plans, output, CascadingBufferFormat::NHWCB, Location::Sram, 3, { 1, 64, 64, 1 },
                        { 1, 8, 8, 16 });

    RequireInputBuffer(plans, input, CascadingBufferFormat::NHWC, Location::Dram, 0, { 1, 32, 32, 4 }, { 0, 0, 0, 0 });
    RequireOutputBuffer(plans, output, CascadingBufferFormat::NHWCB, Location::Sram, 1, { 1, 64, 64, 1 },
                        { 1, 8, 16, 16 });

    RequireInputBuffer(plans, input, CascadingBufferFormat::NHWC, Location::Dram, 0, { 1, 32, 32, 4 }, { 0, 0, 0, 0 });
    RequireOutputBuffer(plans, output, CascadingBufferFormat::NHWCB, Location::Sram, 1, { 1, 64, 64, 1 },
                        { 1, 16, 8, 16 });

    RequireInputBuffer(plans, input, CascadingBufferFormat::NHWC, Location::Dram, 0, { 1, 32, 32, 4 }, { 0, 0, 0, 0 });
    RequireOutputBuffer(plans, output, CascadingBufferFormat::NHWCB, Location::Sram, 1, { 1, 64, 64, 1 },
                        { 1, 64, 64, 16 });

    RequireInputBuffer(plans, input, CascadingBufferFormat::NHWC, Location::VirtualSram, 1, { 1, 32, 32, 4 },
                       { 1, 32, 32, 16 });
}

TEST_CASE("PlanGenerator: Generate plans from a part with reinterpret node")
{
    // Given
    const EstimationOptions estOpt;
    const CompilationOptions compOpt;
    const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
    Graph g;
    auto part         = BuildPartWithReinterpretNode(g, estOpt, compOpt, caps);
    const auto& edges = g.GetEdges();
    const auto& nodes = g.GetNodes();
    REQUIRE(edges.size() == 2);
    REQUIRE(nodes.size() == 3);

    // When
    Plans plans = part.GetPlans();
    SavePlansToDot(plans, "plans_in_part_with_reinterpret_node");

    // Then
    const auto input  = edges[0].get();
    const auto output = nodes[1].get();
    REQUIRE(plans.size() == 2);

    {
        const Plan& dramPlan = *plans[0];
        CHECK(dramPlan.m_OpGraph.GetOps().empty());
        REQUIRE(dramPlan.m_OpGraph.GetBuffers().size() == 1);
        const Buffer* b = dramPlan.m_OpGraph.GetBuffers()[0];
        CHECK(dramPlan.GetInputBuffer(input) == b);
        CHECK(dramPlan.GetOutputBuffer(output) == b);
        CHECK(b->m_Location == Location::Dram);
    }

    {
        const Plan& virtualSramPlan = *plans[1];

        REQUIRE(virtualSramPlan.m_OpGraph.GetOps().size() == 1);
        CHECK(IsObjectOfType<DummyOp>(virtualSramPlan.m_OpGraph.GetOps()[0]));
        CHECK(virtualSramPlan.m_OpGraph.GetBuffers().size() == 2);
        CHECK(virtualSramPlan.m_InputMappings.size() == 1);
        CHECK(virtualSramPlan.m_OutputMappings.size() == 1);

        const Buffer* inputBuffer = virtualSramPlan.GetInputBuffer(input);
        CHECK((inputBuffer->m_Format == CascadingBufferFormat::NHWC &&
               inputBuffer->m_Location == Location::VirtualSram && inputBuffer->m_NumStripes == 1 &&
               inputBuffer->m_TensorShape == TensorShape{ 1, 32, 32, 4 } &&
               inputBuffer->m_StripeShape == TensorShape{ 1, 32, 32, 16 }));

        const Buffer* outputBuffer = virtualSramPlan.GetOutputBuffer(output);
        CHECK((outputBuffer->m_Format == CascadingBufferFormat::NHWC &&
               outputBuffer->m_Location == Location::VirtualSram && outputBuffer->m_NumStripes == 1 &&
               outputBuffer->m_TensorShape == TensorShape{ 1, 64, 64, 1 } &&
               outputBuffer->m_StripeShape == TensorShape{ 1, 64, 64, 16 }));
    }
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

    Part part(0, estOpt, compOpt, caps);
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
    const CompilationOptions compOpt;
    const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
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

    Part part(0, estOpt, compOpt, caps);
    part.m_SubGraph.push_back(pleNode);

    Plans plans = part.GetPlans();

    SavePlansToDot(plans, "plans_in_part_with_fuse_only");

    // Ensure there is a plan with the correct stripes and the has an mce and ple op.
    {
        TS inputStripe{ 1, 16, 224, 64 };
        uint32_t numInputStripes = 1;
        TS outputStripe{ 1, 8, 112, 256 };
        uint32_t numOutputStripes = 1;

        auto planIndices =
            GetPlanIndexContainingStripes(plans, inputStripe, numInputStripes, outputStripe, numOutputStripes);
        REQUIRE(!planIndices.empty());
        bool foundMceOp = false;
        bool foundPleOp = false;
        for (auto planIndex : planIndices)
        {
            auto& plan = plans[planIndex];
            auto ops   = plan->m_OpGraph.GetOps();
            foundMceOp = utils::FindIndexIf(ops, [](Op* op) { return IsObjectOfType<MceOp>(op); }).first;
            foundPleOp = utils::FindIndexIf(ops, [](Op* op) { return IsObjectOfType<PleOp>(op); }).first;
            if (foundMceOp && foundPleOp)
            {
                break;
            }
        }
        REQUIRE(foundMceOp);
        REQUIRE(foundPleOp);
    }
    // Ensure there is a plan with the correct ratio of input and output stripe sizes.
    {
        TS inputStripe{ 1, 16, 16, 16 };
        uint32_t numInputStripes = 1;
        TS outputStripe{ 1, 8, 8, 64 };
        uint32_t numOutputStripes = 1;

        // Ensure there is a plan with the correct stripes and the has an mce and ple op.
        REQUIRE(ContainsPlanWithStripes(plans, inputStripe, numInputStripes, outputStripe, numOutputStripes));
    }
}

TEST_CASE("PlanGenerator: Mobilenet N78_2TOPS_4PLE_RATIO Conv", "[slow]")
{
    const EstimationOptions estOpt;
    const CompilationOptions compOpt;
    const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_2TOPS_4PLE_RATIO);
    Graph g;

    TS inputShape{ 1, 14, 14, 256 };
    TS outputShape{ 1, 14, 14, 512 };
    TS weightShape{ 1, 1, 256, 512 };

    Part part = BuildPartWithMceNode(g, inputShape, outputShape, weightShape,
                                     ethosn::command_stream::MceOperation::CONVOLUTION, estOpt, compOpt, caps);

    Plans plans = part.GetPlans();

    SavePlansToDot(plans, "plans_mobilenet_conv");

    {
        TS inputStripe{ 1, 16, 16, 256 };
        uint32_t numInputStripes = 1;
        TS outputStripe{ 1, 16, 16, 512 };
        uint32_t numOutputStripes = 1;
        REQUIRE(ContainsPlanWithStripes(plans, inputStripe, numInputStripes, outputStripe, numOutputStripes));
    }
}

TEST_CASE("PlanGenerator: Mobilenet N78_2TOPS_4PLE_RATIO Depthwise")
{
    const EstimationOptions estOpt;
    const CompilationOptions compOpt;
    const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_2TOPS_4PLE_RATIO);
    Graph g;

    TS inputShape{ 1, 14, 14, 512 };
    TS outputShape{ 1, 14, 14, 512 };
    TS weightShape{ 1, 1, 512, 1 };
    Part part =
        BuildPartWithMceNode(g, inputShape, outputShape, weightShape,
                             ethosn::command_stream::MceOperation::DEPTHWISE_CONVOLUTION, estOpt, compOpt, caps);

    Plans plans = part.GetPlans();

    SavePlansToDot(plans, "plans_mobilenet_depthwise");
    {
        TS inputStripe{ 1, 16, 16, 512 };
        uint32_t numInputStripes = 1;
        TS outputStripe{ 1, 16, 16, 512 };
        uint32_t numOutputStripes = 1;
        REQUIRE(ContainsPlanWithStripes(plans, inputStripe, numInputStripes, outputStripe, numOutputStripes));
    }
}

TEST_CASE("PlanGenerator: Mobilenet N78_2TOPS_4PLE_RATIO 1024", "[slow]")
{
    const EstimationOptions estOpt;
    const CompilationOptions compOpt;
    const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_2TOPS_4PLE_RATIO);
    Graph g;

    TS inputShape{ 1, 7, 7, 512 };
    TS outputShape{ 1, 7, 7, 1024 };
    TS weightShape{ 1, 1, 512, 1024 };
    Part part = BuildPartWithMceNode(g, inputShape, outputShape, weightShape,
                                     ethosn::command_stream::MceOperation::CONVOLUTION, estOpt, compOpt, caps);

    Plans plans = part.GetPlans();

    SavePlansToDot(plans, "plans_mobilenet_1024");
    {
        TS inputStripe{ 1, 8, 8, 512 };
        uint32_t numInputStripes = 1;
        TS outputStripe{ 1, 8, 8, 16 };
        uint32_t numOutputStripes = 2;
        REQUIRE(ContainsPlanWithStripes(plans, inputStripe, numInputStripes, outputStripe, numOutputStripes));
    }
}

TEST_CASE("PlanGenerator:BlockConfig")
{
    // Graph with FuseOnlyPleNode
    const EstimationOptions estOpt;
    const CompilationOptions compOpt;
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    Graph g;
    auto part = BuildPartWithFuseOnlyPle(g, estOpt, compOpt, caps);

    Plans plans = part.GetPlans();
    SavePlansToDot(plans, "plans_part_blockconfig");

    const auto& plan1 = plans[0];
    REQUIRE(plan1->m_OpGraph.GetBuffers().size() == 5);
    auto ops1 = plan1->m_OpGraph.GetOps();
    REQUIRE(ops1.size() == 3);
    auto pleOp = dynamic_cast<PleOp*>(ops1.back());
    REQUIRE(pleOp != nullptr);
    REQUIRE(pleOp->m_Op == ethosn::command_stream::PleOperation::INTERLEAVE_2X2_2_2);
    REQUIRE(pleOp->m_BlockConfig == ethosn::command_stream::BlockConfig{ 16U, 16U });

    const auto& plan2 = plans[1];
    auto ops2         = plan2->m_OpGraph.GetOps();
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
    const CompilationOptions compOpt;
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    Graph g;

    const uint32_t numIfms = 128;
    const uint32_t numOfms = 256;
    TS tsIn                = { 1, 32, 32, numIfms };
    TS tsOut               = { 1, 64, 64, numOfms };
    auto in                = CreateAndAddInputNode(g, tsIn);
    auto node              = CreateAndAddMceOperationNode(g, tsIn, tsOut, 3, 3);
    auto out               = CreateAndAddOutputNode(g);

    // Graph must be connected to be valid
    g.Connect(in, node, 0);
    g.Connect(node, out, 0);

    Part part(0, estOpt, compOpt, caps);
    part.m_SubGraph.push_back(node);
    Plans plans = part.GetPlans();
    SavePlansToDot(plans, "plans_part_winograd");

    for (const auto& plan : plans)
    {
        auto ops = plan->m_OpGraph.GetOps();
        REQUIRE(!ops.empty());
        auto mceOp = dynamic_cast<MceOp*>(ops[0]);
        REQUIRE(mceOp != nullptr);
        if (mceOp->m_WeightsStripeShape[2] < numIfms)
        {
            REQUIRE(mceOp->m_Algo == CompilerMceAlgorithm::Direct);
        }
        else if ((mceOp->m_BlockConfig == ethosn::command_stream::BlockConfig{ 8U, 8U }) ||
                 (mceOp->m_BlockConfig == ethosn::command_stream::BlockConfig{ 8U, 16U }) ||
                 (mceOp->m_BlockConfig == ethosn::command_stream::BlockConfig{ 16U, 8U }))
        {
            REQUIRE(mceOp->m_Algo == CompilerMceAlgorithm::Winograd);
        }
        else
        {
            REQUIRE(mceOp->m_Algo == CompilerMceAlgorithm::Direct);
        }
    }
}

TEST_CASE("PlanGenerator:Split input in depth")
{
    const EstimationOptions estOpt;
    const CompilationOptions compOpt;
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    Graph g;

    TS tsIn   = { 1, 64, 64, 256 };
    TS tsOut  = { 1, 64, 64, 64 };
    auto in   = CreateAndAddInputNode(g, tsIn);
    auto node = CreateAndAddMceOperationNode(g, tsOut, Stride{ 2U, 2U });
    auto out  = CreateAndAddOutputNode(g);

    // Graph must be connected to be valid
    g.Connect(in, node, 0);
    g.Connect(node, out, 0);

    Part part(0, estOpt, compOpt, caps);
    part.m_SubGraph.push_back(node);
    Plans plans = part.GetPlans();
    SavePlansToDot(plans, "plans_part_split_input_depth");

    uint64_t match = 0;
    for (const auto& plan : plans)
    {
        REQUIRE(!ContainsInputStripe(plan->m_InputMappings, TensorShape{ 1, 16, 16, caps.GetNumberOfOgs() }, 1));
        REQUIRE(!ContainsInputStripe(plan->m_InputMappings, TensorShape{ 1, 16, 16, caps.GetNumberOfOgs() }, 2));

        if (ContainsInputStripe(plan->m_InputMappings, TensorShape{ 1, 16, 16, caps.GetNumberOfOgs() * 4 }, 1))
        {
            ++match;
        }

        if (ContainsInputStripe(plan->m_InputMappings, TensorShape{ 1, 16, 16, caps.GetNumberOfOgs() * 4 }, 2))
        {
            ++match;
        }
    }
    REQUIRE(match > 0);
}

TEST_CASE("PlanGenerator: Split output in depth")
{
    const EstimationOptions estOpt;
    const CompilationOptions compOpt;
    const HardwareCapabilities caps = GetEthosN78HwCapabilities();
    Graph g;

    TS inputShape{ 1, 8, 8, 32 };
    TS outputShape{ 1, 8, 8, 32 };
    TS weightShape{ 3, 3, 32, 32 };
    Part part   = BuildPartWithMceNode(g, inputShape, outputShape, weightShape,
                                     ethosn::command_stream::MceOperation::CONVOLUTION, estOpt, compOpt, caps);
    Plans plans = part.GetPlans();

    SavePlansToDot(plans, "plans_split_output_in_depth");
    {
        TS inputStripe{ 1, 8, 8, 32 };
        uint32_t numInputStripes = 1;
        TS outputStripe{ 1, 8, 8, 8 };
        uint32_t numOutputStripes = 2;
        REQUIRE(std::any_of(plans.begin(), plans.end(), [&](auto& p) {
            auto mceOp = dynamic_cast<MceOp*>(p->m_OpGraph.GetOps()[0]);
            REQUIRE(mceOp != nullptr);
            return ContainsInputStripe(p->m_InputMappings, inputStripe, numInputStripes) &&
                   ContainsOutputStripe(p->m_OutputMappings, outputStripe, numOutputStripes) &&
                   // Check also the algorithm, to make sure we include output-depth-split plans with Winograd enabled
                   // (these were previously missing)
                   mceOp->m_Algo == CompilerMceAlgorithm::Winograd;
        }));
    }
}
