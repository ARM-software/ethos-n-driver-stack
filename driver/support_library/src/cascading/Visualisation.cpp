//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "Visualisation.hpp"

#include "Combiner.hpp"
#include "Graph.hpp"
#include "GraphNodes.hpp"
#include "Part.hpp"
#include "Plan.hpp"
#include "Utils.hpp"

#include <iostream>

using namespace std;

namespace ethosn
{
namespace support_library
{

std::string ToString(Location l)
{
    switch (l)
    {
        case Location::Dram:
            return "Dram";
        case Location::Sram:
            return "Sram";
        case Location::PleInputSram:
            return "PleInputSram";
        default:
            assert(!"Unknown");
            return "";
    }
}

std::string ToString(Lifetime l)
{
    switch (l)
    {
        case Lifetime::Atomic:
            return "Atomic";
        case Lifetime::Cascade:
            return "Cascade";
        default:
            assert(!"Unknown");
            return "";
    }
}

std::string ToString(CompilerDataFormat f)
{
    switch (f)
    {
        case CompilerDataFormat::NONE:
            return "NONE";
        case CompilerDataFormat::NHWC:
            return "NHWC";
        case CompilerDataFormat::NCHW:
            return "NCHW";
        case CompilerDataFormat::NHWCB:
            return "NHWCB";
        case CompilerDataFormat::WEIGHT:
            return "WEIGHT";
        case CompilerDataFormat::NHWCB_COMPRESSED:
            return "NHWCB_COMPRESSED";
        case CompilerDataFormat::FCAF_DEEP:
            return "FCAF_DEEP";
        case CompilerDataFormat::FCAF_WIDE:
            return "FCAF_WIDE";
        default:
            assert(!"Unknown");
            return "";
    }
}

std::string ToString(const TensorShape& s)
{
    std::stringstream ss;
    ss << "[" << s[0] << ", " << s[1] << ", " << s[2] << ", " << s[3] << "]";
    return ss.str();
}

std::string ToString(TraversalOrder o)
{
    switch (o)
    {
        case TraversalOrder::Xyz:
            return "Xyz";
        case TraversalOrder::Zxy:
            return "Zxy";
        default:
            assert(!"Unknown");
            return "";
    }
}

std::string ToString(command_stream::MceOperation o)
{
    switch (o)
    {
        case ethosn::command_stream::MceOperation::CONVOLUTION:
            return "CONVOLUTION";
        case ethosn::command_stream::MceOperation::DEPTHWISE_CONVOLUTION:
            return "DEPTHWISE_CONVOLUTION";
        case ethosn::command_stream::MceOperation::FULLY_CONNECTED:
            return "FULLY_CONNECTED";
        default:
            assert(!"Unknown");
            return "";
    }
}

std::string ToString(command_stream::PleOperation o)
{
    switch (o)
    {
        case ethosn::command_stream::PleOperation::ADDITION:
            return "ADDITION";
        case ethosn::command_stream::PleOperation::ADDITION_RESCALE:
            return "ADDITION_RESCALE";
        case ethosn::command_stream::PleOperation::AVGPOOL_3X3_1_1_UDMA:
            return "AVGPOOL_3X3_1_1_UDMA";
        case ethosn::command_stream::PleOperation::FAULT:
            return "FAULT";
        case ethosn::command_stream::PleOperation::INTERLEAVE_2X2_2_2:
            return "INTERLEAVE_2X2_2_2";
        case ethosn::command_stream::PleOperation::MAXPOOL_2X2_2_2:
            return "MAXPOOL_2X2_2_2";
        case ethosn::command_stream::PleOperation::MAXPOOL_3X3_2_2_EVEN:
            return "MAXPOOL_3X3_2_2_EVEN";
        case ethosn::command_stream::PleOperation::MAXPOOL_3X3_2_2_ODD:
            return "MAXPOOL_3X3_2_2_ODD";
        case ethosn::command_stream::PleOperation::MEAN_XY_7X7:
            return "MEAN_XY_7X7";
        case ethosn::command_stream::PleOperation::MEAN_XY_8X8:
            return "MEAN_XY_8X8";
        case ethosn::command_stream::PleOperation::OFM_SCALING:
            return "OFM_SCALING";
        case ethosn::command_stream::PleOperation::PASSTHROUGH:
            return "PASSTHROUGH";
        case ethosn::command_stream::PleOperation::SIGMOID:
            return "SIGMOID";
        default:
            assert(!"Unknown");
            return "";
    }
}

std::string ToString(command_stream::BlockConfig b)
{
    return std::to_string(b.m_BlockWidth()) + "x" + std::to_string(b.m_BlockHeight());
}

DotAttributes::DotAttributes(std::string id, std::string label, std::string color)
    : m_Id(id)
    , m_Label(label)
    , m_Color(color)
{}

namespace
{

using NodeIds = std::unordered_map<void*, std::string>;

/// Replaces any illegal characters to form a valid .dot file "ID".
std::string SanitizeId(std::string s)
{
    return utils::ReplaceAll(s, " ", "_");
}

DotAttributes GetDotAttributes(Op* op, DetailLevel detailLevel)
{
    DotAttributes result;
    result.m_Id    = SanitizeId(op->m_DebugTag);
    result.m_Shape = "oval";

    std::stringstream label;
    label << op->m_DebugTag;
    if (detailLevel == DetailLevel::High)
    {
        label << "\n";
        label << "Lifetime = " << ToString(op->m_Lifetime) << "\n";

        DmaOp* dmaOp = dynamic_cast<DmaOp*>(op);
        MceOp* mceOp = dynamic_cast<MceOp*>(op);
        PleOp* pleOp = dynamic_cast<PleOp*>(op);
        if (dmaOp != nullptr)
        {
            label << "DmaOp\n";
            label << "Location = " << ToString(dmaOp->m_Location) << "\n";
            label << "Format = " << ToString(dmaOp->m_Format) << "\n";
        }
        else if (mceOp != nullptr)
        {
            label << "MceOp\n";
            label << "Op = " << ToString(mceOp->m_Op) << "\n";
            label << "Block Config = " << ToString(mceOp->m_BlockConfig) << "\n";
            label << "Input Stripe Shape = " << ToString(mceOp->m_InputStripeShape) << "\n";
            label << "Output Stripe Shape = " << ToString(mceOp->m_OutputStripeShape) << "\n";
            label << "Weights Stripe Shape = " << ToString(mceOp->m_WeightsStripeShape) << "\n";
            label << "Order = " << ToString(mceOp->m_Order) << "\n";
        }
        else if (pleOp != nullptr)
        {
            label << "PleOp\n";
            label << "Op = " << ToString(pleOp->m_Op) << "\n";
            label << "Block Config = " << ToString(pleOp->m_BlockConfig) << "\n";
            label << "Num Inputs = " << pleOp->m_NumInputs << "\n";
            label << "Input Stripe Shapes = " << ArrayToString(pleOp->m_InputStripeShapes) << "\n";
            label << "Output Stripe Shape = " << ToString(pleOp->m_OutputStripeShape) << "\n";
        }
    }
    result.m_Label = label.str();

    return result;
}

DotAttributes GetDotAttributes(Buffer* buffer, DetailLevel detailLevel)
{
    DotAttributes result;
    result.m_Id    = SanitizeId(buffer->m_DebugTag);
    result.m_Shape = "box";

    std::stringstream label;
    label << buffer->m_DebugTag;
    if (detailLevel == DetailLevel::High)
    {
        label << "\n";
        label << "Lifetime = " << ToString(buffer->m_Lifetime) << "\n";
        label << "Location = " << ToString(buffer->m_Location) << "\n";
        label << "Format = " << ToString(buffer->m_Format) << "\n";
        label << "Tensor shape = " << ToString(buffer->m_TensorShape) << "\n";
        label << "Stripe shape = " << ToString(buffer->m_StripeShape) << "\n";
        label << "Order = " << ToString(buffer->m_Order) << "\n";
        label << "Size in bytes = " << buffer->m_SizeInBytes << "\n";
    }
    result.m_Label = label.str();

    return result;
}

DotAttributes GetDotAttributes(Part* part, DetailLevel)
{
    DotAttributes result;
    result.m_Id    = SanitizeId(part->m_DebugTag);
    result.m_Label = part->m_DebugTag;
    return result;
}

DotAttributes GetDotAttributes(const Plan* plan, DetailLevel)
{
    DotAttributes result;
    result.m_Id    = SanitizeId(plan->m_DebugTag);
    result.m_Label = plan->m_DebugTag;
    return result;
}

std::string GetLabel(InputNode*, DetailLevel)
{
    return "InputNode";
}

std::string GetLabel(OutputNode*, DetailLevel)
{
    return "OutputNode";
}

std::string GetLabel(ConstantNode*, DetailLevel)
{
    return "ConstantNode";
}

std::string GetLabel(MceOperationNode* node, DetailLevel detailLevel)
{
    std::string label = "MceOperationNode";
    if (detailLevel == DetailLevel::High)
    {
        label += "\n";
        label += ToString(node->GetOperation());
    }
    return label;
}

std::string GetLabel(FuseOnlyPleOperationNode* node, DetailLevel detailLevel)
{
    std::string label = "FuseOnlyPleOperationNode";
    if (detailLevel == DetailLevel::High)
    {
        label += "\n";
        label += ToString(node->GetKernelOperation());
    }
    return label;
}

std::string GetLabel(StandalonePleOperationNode* node, DetailLevel detailLevel)
{
    std::string label = "StandalonePleOperationNode";
    if (detailLevel == DetailLevel::High)
    {
        label += "\n";
        label += ToString(node->GetKernelOperation());
    }
    return label;
}

std::string GetLabel(McePostProcessOperationNode*, DetailLevel)
{
    return "McePostProcessOperationNode";
}

std::string GetLabel(SoftmaxNode*, DetailLevel)
{
    return "SoftmaxNode";
}

std::string GetLabel(RequantizeNode*, DetailLevel)
{
    return "RequantizeNode";
}

std::string GetLabel(FormatConversionNode*, DetailLevel)
{
    return "FormatConversionNode";
}

std::string GetLabel(ReinterpretNode*, DetailLevel)
{
    return "ReinterpretNode";
}

std::string GetLabel(ConcatNode*, DetailLevel)
{
    return "ConcatNode";
}

std::string GetLabel(ExtractSubtensorNode*, DetailLevel)
{
    return "ExtractSubtensorNode";
}

std::string GetLabel(EstimateOnlyNode*, DetailLevel)
{
    return "EstimateOnlyNode";
}

DotAttributes GetDotAttributes(Node* node, DetailLevel detailLevel)
{
    DotAttributes result;
    result.m_Id    = SanitizeId(std::to_string(node->GetId()));
    result.m_Shape = "oval";

    std::stringstream label;
    label << "Node " + std::to_string(node->GetId()) + "\n";

    InputNode* inputNode                          = dynamic_cast<InputNode*>(node);
    OutputNode* outputNode                        = dynamic_cast<OutputNode*>(node);
    ConstantNode* constantNode                    = dynamic_cast<ConstantNode*>(node);
    MceOperationNode* mceNode                     = dynamic_cast<MceOperationNode*>(node);
    FuseOnlyPleOperationNode* fusePleNode         = dynamic_cast<FuseOnlyPleOperationNode*>(node);
    StandalonePleOperationNode* standalonePleNode = dynamic_cast<StandalonePleOperationNode*>(node);
    McePostProcessOperationNode* mcePpNode        = dynamic_cast<McePostProcessOperationNode*>(node);
    SoftmaxNode* softmaxNode                      = dynamic_cast<SoftmaxNode*>(node);
    RequantizeNode* requantNode                   = dynamic_cast<RequantizeNode*>(node);
    FormatConversionNode* formatNode              = dynamic_cast<FormatConversionNode*>(node);
    ReinterpretNode* reinterpretNode              = dynamic_cast<ReinterpretNode*>(node);
    ConcatNode* concatNode                        = dynamic_cast<ConcatNode*>(node);
    ExtractSubtensorNode* extractSubtensorNode    = dynamic_cast<ExtractSubtensorNode*>(node);
    EstimateOnlyNode* estimateNode                = dynamic_cast<EstimateOnlyNode*>(node);

    if (inputNode)
    {
        label << GetLabel(inputNode, detailLevel);
    }
    else if (inputNode)
    {
        label << GetLabel(inputNode, detailLevel);
    }
    else if (outputNode)
    {
        label << GetLabel(outputNode, detailLevel);
    }
    else if (constantNode)
    {
        label << GetLabel(constantNode, detailLevel);
    }
    else if (mceNode)
    {
        label << GetLabel(mceNode, detailLevel);
    }
    else if (fusePleNode)
    {
        label << GetLabel(fusePleNode, detailLevel);
    }
    else if (standalonePleNode)
    {
        label << GetLabel(standalonePleNode, detailLevel);
    }
    else if (mcePpNode)
    {
        label << GetLabel(mcePpNode, detailLevel);
    }
    else if (softmaxNode)
    {
        label << GetLabel(softmaxNode, detailLevel);
    }
    else if (requantNode)
    {
        label << GetLabel(requantNode, detailLevel);
    }
    else if (formatNode)
    {
        label << GetLabel(formatNode, detailLevel);
    }
    else if (reinterpretNode)
    {
        label << GetLabel(reinterpretNode, detailLevel);
    }
    else if (concatNode)
    {
        label << GetLabel(concatNode, detailLevel);
    }
    else if (extractSubtensorNode)
    {
        label << GetLabel(extractSubtensorNode, detailLevel);
    }
    else if (estimateNode)
    {
        label << GetLabel(estimateNode, detailLevel);
    }

    if (detailLevel == DetailLevel::High)
    {
        label << "\n";
        label << "CorrespondingOperationIds:";
        for (auto id : node->GetCorrespondingOperationIds())
        {
            label << " " << id;
        }
        label << "\n";

        label << "Shape = " << ToString(node->GetShape()) << "\n";
        label << "Format = " << ToString(node->GetFormat()) << "\n";
    }
    result.m_Label = label.str();

    return result;
}

void DumpNodeToDotFormat(DotAttributes attr, std::ostream& stream)
{
    std::string label = utils::ReplaceAll(attr.m_Label, "\n", "\\n");
    stream << attr.m_Id << "[";
    stream << "label = \"" << label << "\"";
    if (attr.m_Shape.size() > 0)
    {
        stream << ", shape = " << attr.m_Shape;
    }
    if (attr.m_Color.size() > 0)
    {
        stream << ", color = " << attr.m_Color;
    }
    stream << "]\n";
}

template <typename T>
std::string DumpToDotFormat(T* obj, std::ostream& stream, DetailLevel detailLevel)
{
    DotAttributes attr = GetDotAttributes(obj, detailLevel);
    DumpNodeToDotFormat(attr, stream);
    return attr.m_Id;
}

void DumpSubgraphHeaderToDotFormat(DotAttributes attr, std::ostream& stream)
{
    stream << "subgraph cluster" << attr.m_Id << "\n";
    stream << "{"
           << "\n";
    stream << "label=\"" << utils::ReplaceAll(attr.m_Label, "\n", "\\n") << "\""
           << "\n";
    if (attr.m_Color.size() > 0)
    {
        stream << ", color = " << attr.m_Color;
    }
    stream << "labeljust=l"
           << "\n";
}

NodeIds SaveOpGraphAsBody(const OpGraph& graph, std::ostream& stream, DetailLevel detailLevel)
{
    NodeIds nodeIds;

    // Define all the nodes and remember the node IDs, so we can link them with edges later.
    for (auto&& o : graph.GetOps())
    {
        std::string nodeId = DumpToDotFormat(o, stream, detailLevel);
        nodeIds[o]         = nodeId;
    }
    for (auto&& b : graph.GetBuffers())
    {
        std::string nodeId = DumpToDotFormat(b, stream, detailLevel);
        nodeIds[b]         = nodeId;
    }

    // Define all the edges
    for (auto&& b : graph.GetBuffers())
    {
        Op* producer = graph.GetProducer(b);
        if (producer != nullptr)
        {
            stream << nodeIds.at(producer) << " -> " << nodeIds.at(b) << "\n";
        }

        for (auto&& c : graph.GetConsumers(b))
        {
            stream << nodeIds.at(b) << " -> " << nodeIds.at(c.first);
            // If the consumer has multiple inputs, label each one as the order is important.
            if (graph.GetInputs(c.first).size() > 1)
            {
                stream << "[ label=\"Input " << c.second << "\"]";
            }
            stream << "\n";
        }
    }

    // Heuristic to make the 'weights' input of MceOps appear to the side of the MceOp so it doesn't interrupt
    // the general flow of the network from top to bottom:
    //    Input number 1 of every MceOp, and all its antecedents are placed on the same 'rank'
    for (auto&& o : graph.GetOps())
    {
        if (dynamic_cast<MceOp*>(o) != nullptr && graph.GetInputs(o).size() >= 2)
        {
            stream << "{ rank = \"same\"; " << nodeIds[o] << "; ";
            Buffer* buf = graph.GetInputs(o)[1];
            while (buf != nullptr)
            {
                stream << nodeIds[buf] << "; ";
                Op* op = graph.GetProducer(buf);
                if (op != nullptr)
                {
                    stream << nodeIds[op] << "; ";
                    if (graph.GetInputs(op).size() == 1)
                    {
                        buf = graph.GetInputs(op)[0];
                        continue;
                    }
                }
                break;
            }
            stream << "}\n";
        }
    }

    return nodeIds;
}

NodeIds SavePlanAsBody(const Plan& plan, std::ostream& stream, DetailLevel detailLevel)
{
    NodeIds nodeIds = SaveOpGraphAsBody(plan.m_OpGraph, stream, detailLevel);

    // Indicate what the inputs and outputs of the Plan are
    for (auto&& input : plan.m_InputMappings)
    {
        std::string bufferId = nodeIds.at(input.first);
        std::string id       = "InputLabel" + bufferId;
        std::string label    = "Input from " + GetDotAttributes(input.second->GetSource(), DetailLevel::Low).m_Label;
        stream << id << "[label = \"" << label << "\", shape = box]\n";
        stream << id << " -> " << bufferId << "[arrowhead = box]\n";
    }
    for (auto&& output : plan.m_OutputMappings)
    {
        std::string bufferId = nodeIds.at(output.first);
        std::string id       = "OutputLabel" + bufferId;
        std::string label    = "Output from " + GetDotAttributes(output.second, DetailLevel::Low).m_Label;
        stream << id << "[label = \"" << label << "\", shape = box]\n";
        stream << bufferId << " -> " << id << "[dir = back, arrowtail = box]\n";
    }

    return nodeIds;
}

}    // namespace

void SaveOpGraphToDot(const OpGraph& graph, std::ostream& stream, DetailLevel detailLevel)
{
    stream << "digraph SupportLibraryGraph"
           << "\n";
    stream << "{"
           << "\n";

    SaveOpGraphAsBody(graph, stream, detailLevel);

    stream << "}"
           << "\n";
}

void SaveGraphToDot(const Graph& graph, const GraphOfParts* graphOfParts, std::ostream& stream, DetailLevel detailLevel)
{
    stream << "digraph SupportLibraryGraph"
           << "\n";
    stream << "{"
           << "\n";

    std::unordered_map<Node*, std::string> nodeIds;

    // Process all parts that we were given (if any)
    const Parts& parts = graphOfParts != nullptr ? graphOfParts->m_Parts : static_cast<const Parts&>(Parts());
    for (const auto& part : parts)
    {
        DotAttributes attr = GetDotAttributes(part.get(), detailLevel);
        DumpSubgraphHeaderToDotFormat(attr, stream);

        for (auto&& n : part->m_SubGraph)
        {
            std::string nodeId = DumpToDotFormat(n, stream, detailLevel);
            nodeIds[n]         = nodeId;
        }

        stream << "}"
               << "\n";
    }

    // Process all nodes that aren't included in any Part
    for (auto&& n : graph.GetNodes())
    {
        if (nodeIds.find(n.get()) == nodeIds.end())
        {
            std::string nodeId = DumpToDotFormat(n.get(), stream, detailLevel);
            nodeIds[n.get()]   = nodeId;
        }
    }

    for (auto&& e : graph.GetEdges())
    {
        std::pair<bool, size_t> edgeInput = utils::FindIndex(e->GetDestination()->GetInputs(), e.get());
        stream << nodeIds.at(e->GetSource()) << " -> " << nodeIds.at(e->GetDestination());
        // If the consumer has multiple inputs, label each one as the order is important.
        if (e->GetDestination()->GetInputs().size() > 1)
        {
            stream << "[ label=\"Input " << edgeInput.second << "\"]";
        }
        stream << "\n";
    }
    stream << "}"
           << "\n";
}

void SavePlansToDot(const Part& part, std::ostream& stream, DetailLevel detailLevel)
{
    stream << "digraph SupportLibraryGraph"
           << "\n";
    stream << "{"
           << "\n";

    for (auto&& plan : part.m_Plans)
    {
        DotAttributes attr = GetDotAttributes(plan.get(), detailLevel);
        DumpSubgraphHeaderToDotFormat(attr, stream);
        SavePlanAsBody(*plan, stream, detailLevel);
        stream << "}"
               << "\n";
    }

    stream << "}"
           << "\n";
}

void SaveCombinationToDot(const Combination& combination,
                          const GraphOfParts& graphOfParts,
                          std::ostream& stream,
                          DetailLevel detailLevel)
{
    stream << "digraph SupportLibraryGraph"
           << "\n";
    stream << "{"
           << "\n";

    NodeIds nodeIds;
    std::unordered_map<const Edge*, std::string> edgeInputs;

    for (const Elem& elem : combination.m_Elems)
    {
        const Part& part = graphOfParts.GetPart(elem.m_PartId);
        const Plan& plan = part.GetPlan(elem.m_PlanId);

        // Save Plans as isolated subgraph
        DotAttributes attr = GetDotAttributes(&plan, detailLevel);
        DumpSubgraphHeaderToDotFormat(attr, stream);
        NodeIds newNodeIds = SaveOpGraphAsBody(plan.m_OpGraph, stream, detailLevel);
        nodeIds.insert(newNodeIds.begin(), newNodeIds.end());
        stream << "}"
               << "\n";

        // Connect plan to its inputs
        auto inputEdges = part.GetInputs();
        for (const Edge* inputEdge : inputEdges)
        {
            std::string source = edgeInputs.at(inputEdge);
            std::string dest   = nodeIds.at(plan.GetInputBuffer(inputEdge));
            stream << source << " -> " << dest << "\n";
        }

        // Deal with each output edge, which may have a glue attached
        uint32_t glueCounter = 0;
        auto outputEdges     = part.GetOutputs();
        for (const Edge* outputEdge : outputEdges)
        {
            auto glueIt      = elem.m_Glues.find(outputEdge);
            const Glue* glue = glueIt != elem.m_Glues.end() && !glueIt->second.m_Glue->m_Graph.GetOps().empty()
                                   ? glueIt->second.m_Glue
                                   : nullptr;
            if (glue != nullptr)
            {
                // Save Glue as isolated subgraph
                std::string glueLabel = plan.m_DebugTag + " Glue " + std::to_string(glueCounter);
                DotAttributes attr(SanitizeId(glueLabel), glueLabel, "");
                DumpSubgraphHeaderToDotFormat(attr, stream);
                NodeIds newNodeIds = SaveOpGraphAsBody(glue->m_Graph, stream, detailLevel);
                nodeIds.insert(newNodeIds.begin(), newNodeIds.end());
                stream << "}"
                       << "\n";

                // Connect the glue to its input plan
                stream << nodeIds.at(plan.GetOutputBuffer(outputEdge->GetSource())) << " -> "
                       << nodeIds.at(glue->m_InputSlot.first);
                if (glue->m_Graph.GetInputs(glue->m_InputSlot.first).size() > 1)
                {
                    stream << "[ label=\"Input " << glue->m_InputSlot.second << "\"]";
                }
                stream << "\n";

                edgeInputs[outputEdge] = nodeIds.at(glue->m_Output);
            }
            else
            {
                edgeInputs[outputEdge] = nodeIds.at(plan.GetOutputBuffer(outputEdge->GetSource()));
            }
            ++glueCounter;
        }
    }

    stream << "}"
           << "\n";
}

}    // namespace support_library
}    // namespace ethosn
