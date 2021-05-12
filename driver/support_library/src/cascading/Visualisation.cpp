//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "Visualisation.hpp"

#include "Combiner.hpp"
#include "Estimation.hpp"
#include "Graph.hpp"
#include "GraphNodes.hpp"
#include "Part.hpp"
#include "PerformanceData.hpp"
#include "Plan.hpp"
#include "Utils.hpp"

#include <ethosn_utils/Strings.hpp>

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
        case Location::VirtualSram:
            return "VirtualSram";
        default:
            assert(!"Unknown location");
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
            assert(!"Unknown lifetime");
            return "";
    }
}

std::string ToString(CascadingBufferFormat f)
{
    switch (f)
    {
        case CascadingBufferFormat::NHWC:
            return "NHWC";
        case CascadingBufferFormat::NCHW:
            return "NCHW";
        case CascadingBufferFormat::NHWCB:
            return "NHWCB";
        case CascadingBufferFormat::WEIGHT:
            return "WEIGHT";
        case CascadingBufferFormat::NHWCB_COMPRESSED:
            return "NHWCB_COMPRESSED";
        case CascadingBufferFormat::FCAF_DEEP:
            return "FCAF_DEEP";
        case CascadingBufferFormat::FCAF_WIDE:
            return "FCAF_WIDE";
        default:
            assert(!"Unknown data format");
            return "";
    }
}

std::string ToString(DataFormat f)
{
    switch (f)
    {
        case DataFormat::HWIM:
            return "HWIM";
        case DataFormat::HWIO:
            return "HWIO";
        case DataFormat::NCHW:
            return "NCHW";
        case DataFormat::NHWC:
            return "NHWC";
        case DataFormat::NHWCB:
            return "NHWCB";
        default:
            assert(!"Unknown data format");
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
        default:
            assert(!"Unknown data format");
            return "";
    }
}

std::string ToString(CompilerDataCompressedFormat f)
{
    switch (f)
    {
        case CompilerDataCompressedFormat::NONE:
            return "NONE";
        case CompilerDataCompressedFormat::NHWCB_COMPRESSED:
            return "NHWCB_COMPRESSED";
        case CompilerDataCompressedFormat::FCAF_DEEP:
            return "FCAF_DEEP";
        case CompilerDataCompressedFormat::FCAF_WIDE:
            return "FCAF_WIDE";
        default:
            assert(!"Unknown data compressed format");
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
            assert(!"Unknown traversal order");
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
            assert(!"Unknown MCE operation");
            return "";
    }
}

std::string ToString(CompilerMceAlgorithm a)
{
    switch (a)
    {
        case CompilerMceAlgorithm::None:
            return "NONE";
        case CompilerMceAlgorithm::Direct:
            return "DIRECT";
        case CompilerMceAlgorithm::Winograd:
            return "WINOGRAD";
        default:
            assert(!"Unknown MCE algorithm");
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
        case ethosn::command_stream::PleOperation::DOWNSAMPLE_2X2:
            return "DOWNSAMPLE_2X2";
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
        case ethosn::command_stream::PleOperation::PASSTHROUGH:
            return "PASSTHROUGH";
        case ethosn::command_stream::PleOperation::SIGMOID:
            return "SIGMOID";
        case ethosn::command_stream::PleOperation::TRANSPOSE_XY:
            return "TRANSPOSE_XY";
        default:
            assert(!"Unknown PLE operation");
            return "";
    }
}

std::string ToString(command_stream::BlockConfig b)
{
    return std::to_string(b.m_BlockWidth()) + "x" + std::to_string(b.m_BlockHeight());
}

std::string ToString(const QuantizationScales& scales)
{
    if (scales.size() == 1)
    {
        return "Scale = " + std::to_string(scales[0]);
    }
    else
    {
        std::string out("Scales = [ ");
        for (auto s : scales)
        {
            out += std::to_string(s) + " ";
        }
        out += "]";
        return out;
    }
}

std::string ToString(const QuantizationInfo& q)
{
    std::string out("ZeroPoint = " + std::to_string(q.GetZeroPoint()) + ", " + ToString(q.GetScales()));
    if (q.GetQuantizationDim().has_value())
    {
        out += ", Dim = " + std::to_string(q.GetQuantizationDim().value());
    }
    return out;
}

std::string ToString(const Stride& s)
{
    return std::to_string(s.m_X) + ", " + std::to_string(s.m_Y);
}

std::string ToString(command_stream::DataFormat f)
{
    switch (f)
    {
        case command_stream::DataFormat::FCAF_DEEP:
            return "FCAF_DEEP";
        case command_stream::DataFormat::FCAF_WIDE:
            return "FCAF_WIDE";
        case command_stream::DataFormat::NCHW:
            return "NCHW";
        case command_stream::DataFormat::NHWC:
            return "NHWC";
        case command_stream::DataFormat::NHWCB:
            return "NHWCB";
        case command_stream::DataFormat::NHWCB_COMPRESSED:
            return "NHWCB_COMPRESSED";
        case command_stream::DataFormat::WEIGHT_STREAM:
            return "WEIGHT_STREAM";
        default:
            assert(!"Unknown format");
            return "";
    }
}

std::string ToString(const uint32_t v)
{
    return std::to_string(v);
}

std::string ToString(DataType t)
{
    switch (t)
    {
        case DataType::UINT8_QUANTIZED:
            return "UINT8_QUANTIZED";
        case DataType::INT8_QUANTIZED:
            return "INT8_QUANTIZED";
        case DataType::INT32_QUANTIZED:
            return "INT32_QUANTIZED";
        default:
            assert(!"Unknown format");
            return "";
    }
}

DotAttributes::DotAttributes()
    : m_LabelAlignmentChar('n')
{}

DotAttributes::DotAttributes(std::string id, std::string label, std::string color)
    : m_Id(id)
    , m_Label(label)
    , m_LabelAlignmentChar('n')
    , m_Color(color)
{}

namespace
{

using NodeIds = std::unordered_map<const void*, std::string>;

/// Escapes any characters that have special meaning in the dot language.
/// Unfortunately the escape sequence for newline also encodes the alignment (left, centre, right) of the text.
/// The codes are 'l' -> left, 'r' -> right, 'n' -> centre
std::string Escape(std::string s, char alignmentChar = 'n')
{
    s = ethosn::utils::ReplaceAll(s, "\n", std::string("\\") + alignmentChar);
    s = ethosn::utils::ReplaceAll(s, "\"", "\\\"");
    s = ethosn::utils::ReplaceAll(s, "\t", "    ");    // Tabs don't seem to work at all (e.g. when used in JSON)
    return s;
}

/// Replaces any illegal characters to form a valid .dot file "ID".
std::string SanitizeId(std::string s)
{
    return ethosn::utils::ReplaceAll(s, " ", "_");
}

std::string GetOpString(Op* op)
{
    std::stringstream stream;
    DmaOp* dmaOp = dynamic_cast<DmaOp*>(op);
    MceOp* mceOp = dynamic_cast<MceOp*>(op);
    PleOp* pleOp = dynamic_cast<PleOp*>(op);
    if (dmaOp != nullptr)
    {
        stream << "DmaOp\n";
        stream << "Location = " << ToString(dmaOp->m_Location) << "\n";
    }
    else if (mceOp != nullptr)
    {
        stream << "MceOp\n";
        stream << "Op = " << ToString(mceOp->m_Op) << "\n";
        stream << "Algo = " << ToString(mceOp->m_Algo) << "\n";
        stream << "Block Config = " << ToString(mceOp->m_BlockConfig) << "\n";
        stream << "Input Stripe Shape = " << ToString(mceOp->m_InputStripeShape) << "\n";
        stream << "Output Stripe Shape = " << ToString(mceOp->m_OutputStripeShape) << "\n";
        stream << "Weights Stripe Shape = " << ToString(mceOp->m_WeightsStripeShape) << "\n";
        stream << "Order = " << ToString(mceOp->m_Order) << "\n";
        stream << "Stride = " << ToString(mceOp->m_Stride) << "\n";
        stream << "Pad L/T = " << mceOp->m_PadLeft << ", " << mceOp->m_PadTop << "\n";
    }
    else if (pleOp != nullptr)
    {
        stream << "PleOp\n";
        stream << "Op = " << ToString(pleOp->m_Op) << "\n";
        stream << "Block Config = " << ToString(pleOp->m_BlockConfig) << "\n";
        stream << "Num Inputs = " << pleOp->m_NumInputs << "\n";
        stream << "Input Stripe Shapes = " << ArrayToString(pleOp->m_InputStripeShapes) << "\n";
        stream << "Output Stripe Shape = " << ToString(pleOp->m_OutputStripeShape) << "\n";
    }
    stream << "Operation Ids = " << ArrayToString(op->m_OperationIds) << "\n";
    return stream.str();
}

std::string GetBufferString(Buffer* buffer)
{
    std::stringstream stream;
    stream << "\n";
    stream << "Lifetime = " << ToString(buffer->m_Lifetime) << "\n";
    stream << "Location = " << ToString(buffer->m_Location) << "\n";
    stream << "Format = " << ToString(buffer->m_Format) << "\n";
    stream << "Quant. Info = " << ToString(buffer->m_QuantizationInfo) << "\n";
    stream << "Tensor shape = " << ToString(buffer->m_TensorShape) << "\n";
    stream << "Stripe shape = " << ToString(buffer->m_StripeShape) << "\n";
    stream << "Num. Stripes = " << buffer->m_NumStripes << "\n";
    stream << "Order = " << ToString(buffer->m_Order) << "\n";
    stream << "Size in bytes = " << buffer->m_SizeInBytes << "\n";
    return stream.str();
}

std::string GetCombinationString(const Combination* comb)
{
    std::stringstream stream;
    stream << "\n";
    stream << "Current Part ID = " << std::to_string(comb->m_Scratch.m_CurrPartId) << "\n";
    stream << "Allocated Sram = " << std::to_string(comb->m_Scratch.m_AllocatedSram) << "\n";
    stream << "Score = " << std::to_string(comb->m_Scratch.m_Score) << "\n";
    return stream.str();
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

        label << GetOpString(op);
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
        label << GetBufferString(buffer);
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

DotAttributes GetDotAttributes(const Combination* comb, DetailLevel)
{
    DotAttributes result;

    std::stringstream label;
    label << "Scratch";
    label << GetCombinationString(comb);
    result.m_Label = label.str();

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
        label << "CompressedFormat = " << ToString(node->GetCompressedFormat()) << "\n";
    }
    result.m_Label = label.str();

    return result;
}

DotAttributes GetDotAttributes(Operation* operation, DetailLevel detailLevel)
{
    DotAttributes result;
    result.m_Id    = SanitizeId("Operation" + std::to_string(operation->GetId()));
    result.m_Shape = "oval";

    std::stringstream label;
    label << std::to_string(operation->GetId()) + ": " << operation->GetTypeName() << "\n";

    struct LabelVisitor : NetworkVisitor
    {
        using NetworkVisitor::Visit;

        LabelVisitor(std::stringstream& label, DetailLevel detailLevel)
            : m_Label(label)
            , m_DetailLevel(detailLevel)
        {}

        void Visit(Convolution& op) override
        {
            if (m_DetailLevel >= DetailLevel::High)
            {
                m_Label << "Weights: " << op.GetWeights().GetId() << "\n";
                m_Label << "Bias: " << op.GetBias().GetId() << "\n";
            }
        }

        void Visit(DepthwiseConvolution& op) override
        {
            if (m_DetailLevel >= DetailLevel::High)
            {
                m_Label << "Weights: " << op.GetWeights().GetId() << "\n";
                m_Label << "Bias: " << op.GetBias().GetId() << "\n";
            }
        }

        void Visit(TransposeConvolution& op) override
        {
            if (m_DetailLevel >= DetailLevel::High)
            {
                m_Label << "Weights: " << op.GetWeights().GetId() << "\n";
                m_Label << "Bias: " << op.GetBias().GetId() << "\n";
            }
        }

        void Visit(FullyConnected& op) override
        {
            if (m_DetailLevel >= DetailLevel::High)
            {
                m_Label << "Weights: " << op.GetWeights().GetId() << "\n";
                m_Label << "Bias: " << op.GetBias().GetId() << "\n";
            }
        }

        std::stringstream& m_Label;
        DetailLevel m_DetailLevel;
    } visitor(label, detailLevel);
    operation->Accept(visitor);

    result.m_Label = label.str();

    return result;
}

DotAttributes GetDotAttributes(Operand* operand, DetailLevel detailLevel)
{
    DotAttributes result;
    result.m_Id    = SanitizeId("Operand" + std::to_string(operand->GetProducer().GetId()) + "_" +
                             std::to_string(operand->GetProducerOutputIndex()));
    result.m_Shape = "box";

    std::stringstream label;
    label << "Operand\n";

    if (detailLevel == DetailLevel::High)
    {
        label << "Shape = " << ToString(operand->GetTensorInfo().m_Dimensions) << "\n";
        label << "Format = " << ToString(operand->GetTensorInfo().m_DataFormat) << "\n";
        label << "Type = " << ToString(operand->GetTensorInfo().m_DataType) << "\n";
        label << "Quant. info = " << ToString(operand->GetTensorInfo().m_QuantizationInfo) << "\n";
    }
    result.m_Label = label.str();

    return result;
}

void DumpNodeToDotFormat(DotAttributes attr, std::ostream& stream)
{
    std::string label = Escape(attr.m_Label, attr.m_LabelAlignmentChar);
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
    stream << "label=\"" << Escape(attr.m_Label) << "\""
           << "\n";
    if (attr.m_Color.size() > 0)
    {
        stream << ", color = " << attr.m_Color;
    }
    stream << "labeljust=l"
           << "\n";
}

void SaveOpGraphEdges(const OpGraph& graph, const NodeIds& nodeIds, std::ostream& stream)
{
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
}

/// Heuristic to make the 'weights' input of MceOps appear to the side of the MceOp so it doesn't interrupt
/// the general flow of the network from top to bottom:
///    Input number 1 of every MceOp, and all its antecedents are placed on the same 'rank'
void ApplyOpGraphRankHeuristic(const OpGraph& graph,
                               const std::vector<Op*>& opsSubset,
                               const NodeIds& nodeIds,
                               std::ostream& stream)
{
    for (auto&& o : opsSubset)
    {
        if (dynamic_cast<MceOp*>(o) != nullptr && graph.GetInputs(o).size() >= 2)
        {
            stream << "{ rank = \"same\"; " << nodeIds.at(o) << "; ";
            Buffer* buf = graph.GetInputs(o)[1];
            while (buf != nullptr)
            {
                stream << nodeIds.at(buf) << "; ";
                Op* op = graph.GetProducer(buf);
                if (op != nullptr)
                {
                    stream << nodeIds.at(op) << "; ";
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
    SaveOpGraphEdges(graph, nodeIds, stream);

    // Heuristic to make the 'weights' input of MceOps appear to the side of the MceOp so it doesn't interrupt
    // the general flow of the network from top to bottom:
    ApplyOpGraphRankHeuristic(graph, graph.GetOps(), nodeIds, stream);

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

void SaveNetworkToDot(const Network& network, std::ostream& stream, DetailLevel detailLevel)
{
    stream << "digraph SupportLibraryGraph"
           << "\n";
    stream << "{"
           << "\n";

    NodeIds nodeIds;
    for (auto&& operation : network)
    {
        std::string operationNodeId = DumpToDotFormat(operation.get(), stream, detailLevel);
        nodeIds[operation.get()]    = operationNodeId;

        // Edges to inputs
        uint32_t inputIdx = 0;
        for (auto&& operand : operation->GetInputs())
        {
            stream << nodeIds.at(operand) << " -> " << operationNodeId;
            // If the operation has multiple inputs, label each one as the order is important.
            if (operation->GetInputs().size() > 1)
            {
                stream << "[ label=\"Input " << inputIdx << "\"]";
            }
            stream << "\n";
            ++inputIdx;
        }

        // Output operands
        uint32_t outputIdx = 0;
        for (auto&& operand : operation->GetOutputs())
        {
            std::string operandNodeId = DumpToDotFormat(&operand, stream, detailLevel);
            nodeIds[&operand]         = operandNodeId;

            // Edge to output operand
            stream << operationNodeId << " -> " << operandNodeId;
            // If the operation has multiple outputs, label each one as the order is important.
            if (operation->GetOutputs().size() > 1)
            {
                stream << "[ label=\"Output " << outputIdx << "\"]";
            }
            stream << "\n";
            ++outputIdx;
        }
    }

    stream << "}"
           << "\n";
}

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

void SaveEstimatedOpGraphToDot(const OpGraph& graph,
                               const EstimatedOpGraph& estimationDetails,
                               std::ostream& stream,
                               DetailLevel detailLevel)
{
    stream << "digraph SupportLibraryGraph"
           << "\n";
    stream << "{"
           << "\n";

    // Decide which Pass each Buffer belongs to (if any). This information is derived from the EstimatedOpGraph.
    std::unordered_map<uint32_t, std::vector<Buffer*>> passToBuffers;
    std::unordered_set<Buffer*> unassignedBuffers;
    for (Buffer* b : graph.GetBuffers())
    {
        // If all the buffers inputs and outputs are in the same Pass, then we assign the buffer to that pass too.
        // Otherwise leave it unassigned
        std::vector<uint32_t> neighbourPassIdxs;
        if (graph.GetProducer(b) != nullptr)
        {
            neighbourPassIdxs.push_back(estimationDetails.m_OpToPass.at(graph.GetProducer(b)));
        }
        for (auto consumer : graph.GetConsumers(b))
        {
            neighbourPassIdxs.push_back(estimationDetails.m_OpToPass.at(consumer.first));
        }

        if (!neighbourPassIdxs.empty() && std::all_of(neighbourPassIdxs.begin(), neighbourPassIdxs.end(),
                                                      [&](uint32_t p) { return p == neighbourPassIdxs.front(); }))
        {
            passToBuffers[neighbourPassIdxs.front()].push_back(b);
        }
        else
        {
            unassignedBuffers.insert(b);
        }
    }

    NodeIds nodeIds;

    // Write a subgraph for each pass, containing just the nodes for now.
    // We'll add the edges later as we can do them all together (including edges between passes).
    size_t numPasses = estimationDetails.m_PerfData.m_Stream.size();
    for (uint32_t passIdx = 0; passIdx < numPasses; ++passIdx)
    {
        std::string passId = "Pass" + std::to_string(passIdx);
        DotAttributes passAttr(passId, passId, "");
        DumpSubgraphHeaderToDotFormat(passAttr, stream);

        // Ops
        std::vector<Op*> ops;
        for (auto kv : estimationDetails.m_OpToPass)
        {
            if (kv.second != passIdx)
            {
                continue;
            }

            ops.push_back(kv.first);
            std::string nodeId = DumpToDotFormat(kv.first, stream, detailLevel);
            nodeIds[kv.first]  = nodeId;
        }

        // Buffers
        for (Buffer* b : passToBuffers[passIdx])
        {
            std::string nodeId = DumpToDotFormat(b, stream, detailLevel);
            nodeIds[b]         = nodeId;
        }

        ApplyOpGraphRankHeuristic(graph, ops, nodeIds, stream);

        // Add a "dummy" node showing the perf data JSON
        std::stringstream perfJson;
        PrintPassPerformanceData(perfJson, ethosn::utils::Indent(0), estimationDetails.m_PerfData.m_Stream[passIdx]);
        DotAttributes perfAttr(passId + "_Perf", perfJson.str(), "");
        perfAttr.m_Shape              = "box";
        perfAttr.m_LabelAlignmentChar = 'l';
        DumpNodeToDotFormat(perfAttr, stream);

        stream << "}"
               << "\n";
    }

    // Buffers that aren't in a Pass
    for (Buffer* b : unassignedBuffers)
    {
        std::string nodeId = DumpToDotFormat(b, stream, detailLevel);
        nodeIds[b]         = nodeId;
    }

    // Edges
    SaveOpGraphEdges(graph, nodeIds, stream);

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
    for (const std::unique_ptr<Part>& part : parts)
    {
        DotAttributes attr = GetDotAttributes(part.get(), detailLevel);
        DumpSubgraphHeaderToDotFormat(attr, stream);

        for (Node*& n : part->m_SubGraph)
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

void SaveOpGraphToTxtFile(const OpGraph& graph, std::ostream& stream)
{
    auto ops = graph.GetOps();
    for (auto op : ops)
    {
        stream << GetOpString(op);
        stream << "\n";
        stream << "\nInput Buffers: \n";
        auto inputBufs = graph.GetInputs(op);
        for (auto inputBuf : inputBufs)
        {
            stream << GetBufferString(inputBuf);
        }
        stream << "Output Buffers: \n";
        auto outputBuf = graph.GetOutput(op);
        if (outputBuf)
        {
            stream << GetBufferString(outputBuf);
        }
        stream << "\n";
    }
    stream << "-------------------------------------------------------------------------\n";
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

    // Save Scratch at the top
    DotAttributes attr = GetDotAttributes(&combination, detailLevel);
    DumpSubgraphHeaderToDotFormat(attr, stream);
    stream << "{"
           << "\n";

    NodeIds nodeIds;
    std::unordered_map<const Edge*, std::string> edgeInputs;

    for (const Elem& elem : combination.m_Elems)
    {
        const Part& part = graphOfParts.GetPart(elem.m_PartId);
        const Plan& plan = part.GetPlan(elem.m_PlanId);

        // Save Plans as isolated subgraph
        attr = GetDotAttributes(&plan, detailLevel);
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
                // If the consumer has multiple inputs, label each one as the order is important.
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
    stream << "}"
           << "\n";
    stream << "}"
           << "\n";
}

}    // namespace support_library
}    // namespace ethosn
