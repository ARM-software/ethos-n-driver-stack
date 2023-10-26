//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "Visualisation.hpp"

#include "CombinerDFS.hpp"
#include "CommandStreamGenerator.hpp"
#include "ConcreteOperations.hpp"
#include "Estimation.hpp"
#include "EstimationUtils.hpp"
#include "Network.hpp"
#include "Part.hpp"
#include "PerformanceData.hpp"
#include "Plan.hpp"
#include "Utils.hpp"

#include <ethosn_utils/Strings.hpp>

#include <iostream>

using namespace std;
using namespace ethosn::command_stream;

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
            ETHOSN_FAIL_MSG("Unknown location");
            return "";
    }
}

std::string ToString(BufferFormat f)
{
    switch (f)
    {
        case BufferFormat::NHWC:
            return "NHWC";
        case BufferFormat::NCHW:
            return "NCHW";
        case BufferFormat::NHWCB:
            return "NHWCB";
        case BufferFormat::WEIGHT:
            return "WEIGHT";
        case BufferFormat::FCAF_DEEP:
            return "FCAF_DEEP";
        case BufferFormat::FCAF_WIDE:
            return "FCAF_WIDE";
        default:
            ETHOSN_FAIL_MSG("Unknown data format");
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
            ETHOSN_FAIL_MSG("Unknown data format");
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
            ETHOSN_FAIL_MSG("Unknown data format");
            return "";
    }
}

std::string ToString(CompilerDataCompressedFormat f)
{
    switch (f)
    {
        case CompilerDataCompressedFormat::NONE:
            return "NONE";
        case CompilerDataCompressedFormat::FCAF_DEEP:
            return "FCAF_DEEP";
        case CompilerDataCompressedFormat::FCAF_WIDE:
            return "FCAF_WIDE";
        default:
            ETHOSN_FAIL_MSG("Unknown data compressed format");
            return "";
    }
}

std::string ToString(const TensorInfo& i)
{
    return "(" + ToString(i.m_Dimensions) + ", " + ToString(i.m_DataType) + ", " + ToString(i.m_DataFormat) + ", " +
           ToString(i.m_QuantizationInfo) + ")";
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
            ETHOSN_FAIL_MSG("Unknown traversal order");
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
            ETHOSN_FAIL_MSG("Unknown MCE operation");
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
            ETHOSN_FAIL_MSG("Unknown MCE algorithm");
            return "";
    }
}

std::string ToString(PleOperation o)
{
    switch (o)
    {
        case PleOperation::ADDITION:
            return "ADDITION";
        case PleOperation::ADDITION_RESCALE:
            return "ADDITION_RESCALE";
        case PleOperation::MULTIPLICATION:
            return "MULTIPLICATION";
        case PleOperation::AVGPOOL_3X3_1_1_UDMA:
            return "AVGPOOL_3X3_1_1_UDMA";
        case PleOperation::DOWNSAMPLE_2X2:
            return "DOWNSAMPLE_2X2";
        case PleOperation::INTERLEAVE_2X2_2_2:
            return "INTERLEAVE_2X2_2_2";
        case PleOperation::MAXPOOL_2X2_2_2:
            return "MAXPOOL_2X2_2_2";
        case PleOperation::MAXPOOL_3X3_2_2_EVEN:
            return "MAXPOOL_3X3_2_2_EVEN";
        case PleOperation::MAXPOOL_3X3_2_2_ODD:
            return "MAXPOOL_3X3_2_2_ODD";
        case PleOperation::MEAN_XY_7X7:
            return "MEAN_XY_7X7";
        case PleOperation::MEAN_XY_8X8:
            return "MEAN_XY_8X8";
        case PleOperation::PASSTHROUGH:
            return "PASSTHROUGH";
        case PleOperation::SIGMOID:
            return "SIGMOID";
        case PleOperation::TRANSPOSE_XY:
            return "TRANSPOSE_XY";
        case PleOperation::LEAKY_RELU:
            return "LEAKY_RELU";
        case PleOperation::MAXPOOL1D:
            return "MAXPOOL1D";
        default:
            ETHOSN_FAIL_MSG("Unknown PLE operation");
            return "";
    }
}

std::string ToString(const BlockConfig& b)
{
    return std::to_string(b.m_Width) + "x" + std::to_string(b.m_Height);
}

std::string ToString(const QuantizationScales& scales)
{
    if (scales.size() == 1)
    {
        return "Scale = " + std::to_string(scales[0]);
    }
    else
    {
        // Keep the representation compact by showing the min and max, rather than every value
        return "Scales = [" + std::to_string(scales.size()) + "](min = " + std::to_string(scales.min()) +
               ", max = " + std::to_string(scales.max()) + ")";
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

std::string ToString(const bool b)
{
    return b ? "True" : "False";
}

std::string ToString(const uint16_t v)
{
    return std::to_string(static_cast<uint32_t>(v));
}

std::string ToString(const uint32_t v)
{
    return std::to_string(v);
}

std::string ToString(const int32_t v)
{
    return std::to_string(v);
}

std::string ToStringHex(const uint32_t v)
{
    std::stringstream ss;
    ss << std::hex << std::uppercase << "0x" << v;
    return ss.str();
}

std::string ToString(const std::string& s)
{
    return s;
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
            ETHOSN_FAIL_MSG("Unknown format");
            return "";
    }
}

std::string ToString(const utils::ShapeMultiplier& m)
{
    return "[" + ToString(m.m_H) + ", " + ToString(m.m_W) + ", " + ToString(m.m_C) + "]";
}

std::string ToString(const utils::Fraction& f)
{
    return ToString(f.m_Numerator) + "/" + ToString(f.m_Denominator);
}

std::string ToString(MceUpsampleType t)
{
    switch (t)
    {
        case MceUpsampleType::OFF:
            return "OFF";
        case MceUpsampleType::BILINEAR:
            return "BILINEAR";
        case MceUpsampleType::NEAREST_NEIGHBOUR:
            return "NEAREST_NEIGHBOUR";
        case MceUpsampleType::TRANSPOSE:
            return "TRANSPOSE";
        default:
            ETHOSN_FAIL_MSG("Unknown MceUpsampleType");
            return "";
    }
}

std::string ToString(command_stream::PleKernelId id)
{
    return command_stream::PleKernelId2String(id);
}

std::string ToString(const BufferType& t)
{
    switch (t)
    {
        case BufferType::Input:
            return "Input";
        case BufferType::Output:
            return "Output";
        case BufferType::ConstantDma:
            return "ConstantDma";
        case BufferType::ConstantControlUnit:
            return "ConstantControlUnit";
        case BufferType::Intermediate:
            return "Intermediate";
        default:
            ETHOSN_FAIL_MSG("Unknown type");
            return "";
    }
}

std::string ToString(PackedBoundaryThickness t)
{
    return "{ L: " + std::to_string(static_cast<int>(t.left)) + ", T: " + std::to_string(static_cast<int>(t.top)) +
           ", R: " + std::to_string(static_cast<int>(t.right)) + ", B: " + std::to_string(static_cast<int>(t.bottom)) +
           " }";
}

std::string ToString(const Padding& p)
{
    return "{ L: " + std::to_string(static_cast<int>(p.m_Left)) + ", T: " + std::to_string(static_cast<int>(p.m_Top)) +
           ", R: " + std::to_string(static_cast<int>(p.m_Right)) +
           ", B: " + std::to_string(static_cast<int>(p.m_Bottom)) + " }";
}

std::string ToString(const PoolingType& p)
{
    switch (p)
    {
        case PoolingType::AVG:
            return "AVG";
        case PoolingType::MAX:
            return "MAX";
        default:
            ETHOSN_FAIL_MSG("Unknown PoolingType");
            return "";
    }
}

/// Replaces any illegal characters to form a valid .dot file "ID".
std::string SanitizeId(std::string s)
{
    return ethosn::utils::ReplaceAll(s, " ", "_");
}

DotAttributes::DotAttributes()
    : m_LabelAlignmentChar('n')
{}

namespace
{

using NodeIds = std::unordered_map<const void*, std::string>;

/// Escapes any characters that have special meaning in the dot language.
/// Unfortunately the escape sequence for newline also encodes the alignment (left, centre, right) of the text.
/// The codes are 'l' -> left, 'r' -> right, 'n' -> centre
std::string Escape(std::string s, char alignmentChar = 'n')
{
    // If the string is multi-line, make sure it has a trailing newline, otherwise the resulting dot string will have
    // incorrect alignment on the last line (it seems to default to centered, so e.g. left-justified multi-line strings
    // will be wrong).
    if (!s.empty() && s.find('\n') != std::string::npos && s.back() != '\n')
    {
        s += '\n';
    }
    s = ethosn::utils::ReplaceAll(s, "\n", std::string("\\") + alignmentChar);
    s = ethosn::utils::ReplaceAll(s, "\"", "\\\"");
    s = ethosn::utils::ReplaceAll(s, "\t", "    ");    // Tabs don't seem to work at all (e.g. when used in JSON)
    return s;
}

DotAttributes GetDotAttributes(Op* op, DetailLevel detailLevel, uint32_t idxInOpGraph, std::string extra = "")
{
    DotAttributes result = op->GetDotAttributes(detailLevel);
    result.m_Id          = SanitizeId(op->m_DebugTag);
    result.m_Shape       = "oval";

    std::stringstream preLabel;
    preLabel << op->m_DebugTag;
    if (detailLevel == DetailLevel::High)
    {
        preLabel << "\n";
        preLabel << "Idx in OpGraph: " << idxInOpGraph << "\n";
        if (!extra.empty())
        {
            preLabel << extra << "\n";
        }
    }
    preLabel << result.m_Label;
    result.m_Label = preLabel.str();

    return result;
}

DotAttributes GetDotAttributes(Buffer* buffer, DetailLevel detailLevel, std::string extra = "")
{
    DotAttributes result = buffer->GetDotAttributes(detailLevel);
    result.m_Id          = SanitizeId(buffer->m_DebugTag);
    result.m_Shape       = "box";
    // Highlight buffer locations with colour to make it easier to see where cascading has taken place
    result.m_Color =
        buffer->m_Location == Location::Dram ? "brown" : buffer->m_Location == Location::Sram ? "blue" : "";

    std::stringstream preLabel;
    preLabel << buffer->m_DebugTag;
    if (detailLevel == DetailLevel::High)
    {
        preLabel << "\n";
        if (!extra.empty())
        {
            preLabel << extra << "\n";
        }
    }
    preLabel << result.m_Label;
    result.m_Label = preLabel.str();

    return result;
}

DotAttributes GetDotAttributes(const BasePart& part, DetailLevel detail)
{
    return part.GetDotAttributes(detail);
}

DotAttributes GetDotAttributes(const Plan* plan, DetailLevel)
{
    DotAttributes result;
    result.m_Id    = SanitizeId(plan->m_DebugTag);
    result.m_Label = plan->m_DebugTag;
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

                const uint64_t numIfms           = utils::GetChannels(op.GetInput(0).GetTensorInfo().m_Dimensions);
                const uint64_t numOfms           = utils::GetChannels(op.GetOutput(0).GetTensorInfo().m_Dimensions);
                const uint64_t weightsWidth      = op.GetWeights().GetTensorInfo().m_Dimensions[1];
                const uint64_t weightsHeight     = op.GetWeights().GetTensorInfo().m_Dimensions[0];
                const uint64_t numKernelElements = weightsWidth * weightsHeight;
                const uint64_t outputWidth       = utils::GetWidth(op.GetOutput(0).GetTensorInfo().m_Dimensions);
                const uint64_t outputHeight      = utils::GetHeight(op.GetOutput(0).GetTensorInfo().m_Dimensions);
                const uint64_t numOutputElementsPerOfm = outputWidth * outputHeight;
                // We count multiplies and adds separately, hence the factor of 2x.
                const uint64_t numOpsPerIfmPerOfm = numOutputElementsPerOfm * 2U * numKernelElements;
                m_Label << "Num MACs: " << numIfms * numOpsPerIfmPerOfm * numOfms << "\n";
            }
        }

        void Visit(DepthwiseConvolution& op) override
        {
            if (m_DetailLevel >= DetailLevel::High)
            {
                m_Label << "Weights: " << op.GetWeights().GetId() << "\n";
                m_Label << "Bias: " << op.GetBias().GetId() << "\n";

                const uint64_t numOfms           = utils::GetChannels(op.GetOutput(0).GetTensorInfo().m_Dimensions);
                const uint64_t weightsWidth      = op.GetWeights().GetTensorInfo().m_Dimensions[1];
                const uint64_t weightsHeight     = op.GetWeights().GetTensorInfo().m_Dimensions[0];
                const uint64_t numKernelElements = weightsWidth * weightsHeight;
                const uint64_t outputWidth       = utils::GetWidth(op.GetOutput(0).GetTensorInfo().m_Dimensions);
                const uint64_t outputHeight      = utils::GetHeight(op.GetOutput(0).GetTensorInfo().m_Dimensions);
                const uint64_t numOutputElementsPerOfm = outputWidth * outputHeight;
                // We count multiplies and adds separately, hence the factor of 2x.
                const uint64_t numOpsPerOfm = numOutputElementsPerOfm * 2U * numKernelElements;
                m_Label << "Num MACs: " << numOpsPerOfm * numOfms << "\n";
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

        void Visit(Pooling& op) override
        {
            if (m_DetailLevel >= DetailLevel::High)
            {
                m_Label << "Type: " << ToString(op.GetPoolingInfo().m_PoolingType) << "\n";
                m_Label << "Pooling size: " << op.GetPoolingInfo().m_PoolingSizeX << " x "
                        << op.GetPoolingInfo().m_PoolingSizeY << "\n";
                m_Label << "Stride: " << op.GetPoolingInfo().m_PoolingStrideX << " x "
                        << op.GetPoolingInfo().m_PoolingStrideY << "\n";
                m_Label << "Padding: " << ToString(op.GetPoolingInfo().m_Padding) << "\n";
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

template <typename T, typename... TArgs>
std::string DumpToDotFormat(T* obj, std::ostream& stream, DetailLevel detailLevel, TArgs&&... additionalArgs)
{
    DotAttributes attr = GetDotAttributes(obj, detailLevel, std::forward<TArgs>(additionalArgs)...);
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
        stream << "color = " << attr.m_Color << "\n";
    }
    stream << "labeljust=l"
           << "\n";
    if (attr.m_FontSize.size() > 0)
    {
        stream << "fontsize = " << attr.m_FontSize << "\n";
    }
}

void SaveOpGraphEdges(const OpGraph& graph, const NodeIds& nodeIds, std::ostream& stream)
{
    // Define all the edges
    for (auto&& b : graph.GetBuffers())
    {
        for (Op* producer : graph.GetProducers(b))
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

int HasWeightsBuffer(const OpGraph& graph, Op* const o)
{
    OpGraph::BufferList bufList = graph.GetInputs(o);
    int weightsBufferIndex      = -1;
    int bufferIndex             = 0;
    for (auto&& buf : bufList)
    {
        if (buf->m_Format == BufferFormat::WEIGHT)
        {
            weightsBufferIndex = bufferIndex;
        }
        bufferIndex++;
    }
    return weightsBufferIndex;
}

/// Heuristic to make the 'weights' input of MceOps appear to the side of the MceOp so it doesn't interrupt
/// the general flow of the network from top to bottom:
///    The weights input (usually input 1) of every MceOp, and all its antecedents are placed on the same 'rank'
void ApplyOpGraphRankHeuristic(const OpGraph& graph,
                               const std::vector<Op*>& opsSubset,
                               const NodeIds& nodeIds,
                               std::ostream& stream)
{
    for (auto&& o : opsSubset)
    {
        int hasWeightsBufferIdx = (HasWeightsBuffer(graph, o));
        if (hasWeightsBufferIdx != -1)
        {
            stream << "{ rank = \"same\"; " << nodeIds.at(o) << "; ";
            Buffer* buf = graph.GetInputs(o)[hasWeightsBufferIdx];
            while (buf != nullptr)
            {
                stream << nodeIds.at(buf) << "; ";
                Op* op = graph.GetSingleProducer(buf);
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
    uint32_t idx = 0;
    for (auto&& o : graph.GetOps())
    {
        std::string nodeId = DumpToDotFormat(o, stream, detailLevel, idx);
        nodeIds[o]         = nodeId;
        ++idx;
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
        std::string label    = "Input Slot " + std::to_string(input.second.m_InputIndex);
        stream << id << "[label = \"" << label << "\", shape = box]\n";
        stream << id << " -> " << bufferId << "[arrowhead = box]\n";
    }
    for (auto&& output : plan.m_OutputMappings)
    {
        std::string bufferId = nodeIds.at(output.first);
        std::string id       = "OutputLabel" + bufferId;
        std::string label    = "Output Slot " + std::to_string(output.second.m_OutputIndex);
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
                               DetailLevel detailLevel,
                               std::map<uint32_t, std::string> extraPassDetails,
                               std::map<Op*, std::string> extraOpDetails,
                               std::map<Buffer*, std::string> extraBufferDetails)
{
    stream << "digraph SupportLibraryGraph"
           << "\n";
    stream << "{"
           << "\n";

    // Add a title showing the total metric
    stream << "labelloc=\"t\";\n";
    stream << "label=\"Total metric = " << estimationDetails.m_Metric << "\";\n";

    std::map<Op*, uint32_t> opToOpGraphIdx;
    uint32_t idx = 0;
    for (Op* o : graph.GetOps())
    {
        opToOpGraphIdx[o] = idx;
        ++idx;
    }

    // Decide which Pass each Buffer belongs to (if any). This information is not directly available in EstimatedOpGraph
    // as that just contains the Pass for each *Op*, so we must derive this information.
    std::unordered_map<uint32_t, std::vector<Buffer*>> passToBuffers;
    std::vector<Buffer*> unassignedBuffers;
    for (Buffer* b : graph.GetBuffers())
    {
        // If all the buffer's inputs and outputs are in the same Pass, then we assign the buffer to that pass too.
        // Otherwise leave it unassigned
        std::vector<Op*> neighbourOps;
        for (auto producer : graph.GetProducers(b))
        {
            neighbourOps.push_back(producer);
        }
        for (auto consumer : graph.GetConsumers(b))
        {
            neighbourOps.push_back(consumer.first);
        }

        utils::Optional<uint32_t> commonPassIdx;
        for (Op* neighbour : neighbourOps)
        {
            auto it = estimationDetails.m_OpToPass.find(neighbour);
            // The Op may not be in a Pass, for example if it is an EstimateOnlyOp
            utils::Optional<uint32_t> passIdx = it == estimationDetails.m_OpToPass.end()
                                                    ? utils::Optional<uint32_t>()
                                                    : utils::Optional<uint32_t>(it->second);

            if (!commonPassIdx.has_value())
            {
                commonPassIdx = passIdx;
            }
            else if (commonPassIdx.has_value() && passIdx != commonPassIdx)
            {
                commonPassIdx = utils::Optional<uint32_t>();
                break;
            }
        }

        if (commonPassIdx.has_value())
        {
            passToBuffers[commonPassIdx.value()].push_back(b);
        }
        else
        {
            unassignedBuffers.push_back(b);
        }
    }

    NodeIds nodeIds;

    // Write a subgraph for each pass, containing just the nodes for now.
    // We'll add the edges later as we can do them all together (including edges between passes).
    size_t numPasses = estimationDetails.m_Passes.size();
    for (uint32_t passIdx = 0; passIdx < numPasses; ++passIdx)
    {
        std::string passId = "Pass" + std::to_string(passIdx);
        DotAttributes passAttr(passId, passId, "");
        // Passes tend to be large so it's nice to be able to see the names/indexes when zoomed far out
        passAttr.m_FontSize = "56";
        {
            const auto it = extraPassDetails.find(passIdx);
            if (it != extraPassDetails.end())
            {
                passAttr.m_Label += "\n" + it->second;
            }
        }
        DumpSubgraphHeaderToDotFormat(passAttr, stream);

        // Ops
        std::vector<Op*> ops;
        for (auto kv : estimationDetails.m_OpToPass)
        {
            if (kv.second != passIdx)
            {
                continue;
            }

            std::string extraDetails = utils::GetWithDefault(extraOpDetails, kv.first, std::string(""));

            ops.push_back(kv.first);
            std::string nodeId =
                DumpToDotFormat(kv.first, stream, detailLevel, opToOpGraphIdx.at(kv.first), extraDetails);
            nodeIds[kv.first] = nodeId;
        }

        // Buffers
        for (Buffer* b : passToBuffers[passIdx])
        {
            std::string extraDetails = utils::GetWithDefault(extraBufferDetails, b, std::string(""));
            std::string nodeId       = DumpToDotFormat(b, stream, detailLevel, extraDetails);
            nodeIds[b]               = nodeId;
        }

        ApplyOpGraphRankHeuristic(graph, ops, nodeIds, stream);

        // Add a "dummy" node showing the metric and debug info for this pass
        std::stringstream perfDetails;
        perfDetails << "Metric = " << estimationDetails.m_Passes[passIdx].m_Metric << "\n\n";
        perfDetails << estimationDetails.m_Passes[passIdx].m_DebugInfo << "\n\n";
        PrintPassPerformanceData(perfDetails, ethosn::utils::Indent(0),
                                 estimationDetails.m_LegacyPerfData.m_Stream[passIdx]);
        DotAttributes perfAttr(passId + "_Perf", perfDetails.str(), "");
        perfAttr.m_Shape              = "note";
        perfAttr.m_LabelAlignmentChar = 'l';
        DumpNodeToDotFormat(perfAttr, stream);

        stream << "}"
               << "\n";
    }

    // Ops that aren't in a Pass (e.g. EstimateOnlyOps).
    std::vector<Op*> unassignedOps;
    for (Op* op : graph.GetOps())
    {
        if (estimationDetails.m_OpToPass.find(op) == estimationDetails.m_OpToPass.end())
        {
            unassignedOps.push_back(op);
        }
    }
    // Sort these by something deterministic for reproducible behaviour (NOT pointer values!)
    std::sort(unassignedOps.begin(), unassignedOps.end(), [](auto a, auto b) { return a->m_DebugTag < b->m_DebugTag; });
    for (Op* o : unassignedOps)
    {
        std::string nodeId = DumpToDotFormat(o, stream, detailLevel, opToOpGraphIdx.at(o));
        nodeIds[o]         = nodeId;
    }

    // Buffers that aren't in a Pass
    for (Buffer* b : unassignedBuffers)
    {
        std::string extraDetails = utils::GetWithDefault(extraBufferDetails, b, std::string(""));
        std::string nodeId       = DumpToDotFormat(b, stream, detailLevel, extraDetails);
        nodeIds[b]               = nodeId;
    }

    // Edges
    SaveOpGraphEdges(graph, nodeIds, stream);

    stream << "}"
           << "\n";
}

void SaveCompiledOpGraphToDot(const OpGraph& graph,
                              const CompiledOpGraph& compilationDetails,
                              std::ostream& stream,
                              DetailLevel detailLevel)
{
    std::map<Op*, std::string> extraOpDetails;
    for (const auto& pair : compilationDetails.m_OpToAgentIdMapping)
    {
        extraOpDetails[pair.first] = "Agent ID: " + std::to_string(pair.second);
    }

    std::map<Buffer*, std::string> extraBufferDetails;
    for (const auto& pair : compilationDetails.m_BufferIds)
    {
        extraBufferDetails[pair.first] = "Buffer ID: " + std::to_string(pair.second);
    }

    std::map<uint32_t, std::pair<size_t, size_t>> passAgentIdRanges;
    for (const auto& pair : compilationDetails.m_EstimatedOpGraph.m_OpToPass)
    {
        auto it = passAgentIdRanges.insert({ pair.second, { std::numeric_limits<uint32_t>::max(), 0 } });

        std::pair<size_t, size_t>& p = it.first->second;
        size_t agentId               = compilationDetails.m_OpToAgentIdMapping.at(pair.first);
        p.first                      = std::min(p.first, agentId);
        p.second                     = std::max(p.second, agentId);
    }

    std::map<uint32_t, std::string> extraPassDetails;
    for (const auto& pair : passAgentIdRanges)
    {
        extraPassDetails[pair.first] =
            "Agent IDs: " + std::to_string(pair.second.first) + " - " + std::to_string(pair.second.second);
    }

    SaveEstimatedOpGraphToDot(graph, compilationDetails.m_EstimatedOpGraph, stream, detailLevel, extraPassDetails,
                              extraOpDetails, extraBufferDetails);
}

void SaveGraphOfPartsToDot(const GraphOfParts& graphOfParts, std::ostream& stream, DetailLevel detailLevel)
{
    stream << "digraph SupportLibraryGraph"
           << "\n";
    stream << "{"
           << "\n";

    std::unordered_map<PartId, std::string> partIds;

    // Process all parts that we were given (if any)
    const Parts& parts = graphOfParts.GetParts();
    for (const std::pair<const PartId, std::unique_ptr<BasePart>>& idAndPart : parts)
    {
        DotAttributes attr = GetDotAttributes(*idAndPart.second, detailLevel);
        DumpNodeToDotFormat(attr, stream);
        partIds[idAndPart.first] = attr.m_Id;
    }
    // Precompute the number of input and output parts for each part
    std::map<PartId, bool> partsMultipleOutputs;
    std::map<PartId, bool> partsMultipleInputs;
    for (const std::pair<const PartId, std::unique_ptr<BasePart>>& idAndPart : parts)
    {
        bool multipleOutputs = graphOfParts.GetPartOutputs(idAndPart.first).size() > 1 ? true : false;
        bool multipleInputs  = graphOfParts.GetPartInputs(idAndPart.first).size() > 1 ? true : false;
        partsMultipleOutputs.insert({ idAndPart.first, multipleOutputs });
        partsMultipleInputs.insert({ idAndPart.first, multipleInputs });
    }
    // Copy edges into a vector and sort so there are deterministic results.
    using InOutSlots = std::pair<PartInputSlot, PartOutputSlot>;
    std::vector<InOutSlots> edges;
    for (auto&& edge : graphOfParts.GetAllConnections())
    {
        edges.emplace_back(edge.first, edge.second);
    }
    std::sort(edges.begin(), edges.end(), [](const InOutSlots& left, const InOutSlots& right) {
        if (left.first.m_PartId < right.first.m_PartId)
            return true;
        if (right.first.m_PartId < left.first.m_PartId)
            return false;
        if (left.first.m_InputIndex < right.first.m_InputIndex)
            return true;
        if (right.first.m_InputIndex < left.first.m_InputIndex)
            return false;
        return false;
    });
    for (auto&& edge : edges)
    {
        stream << partIds[edge.second.m_PartId] << " -> " << partIds[edge.first.m_PartId];
        // Only print the slot number if there are more than 1 outputs for a part.
        if (partsMultipleOutputs.at(edge.second.m_PartId))
        {
            stream << "[ taillabel=\""
                   << "Slot " << edge.second.m_OutputIndex << "\"]";
        }
        // Only print the slot number if there are more than 1 inputs for a part.
        if (partsMultipleInputs.at(edge.first.m_PartId))
        {
            stream << "[ headlabel=\""
                   << "Slot " << edge.first.m_InputIndex << "\"]";
        }
        stream << "\n";
    }

    stream << "}"
           << "\n";
}

void SavePlansToDot(const Plans& plans, std::ostream& stream, DetailLevel detailLevel)
{
    stream << "digraph SupportLibraryGraph"
           << "\n";
    stream << "{"
           << "\n";

    for (auto&& plan : plans)
    {
        DotAttributes attr = GetDotAttributes(&plan, detailLevel);
        DumpSubgraphHeaderToDotFormat(attr, stream);
        SavePlanAsBody(plan, stream, detailLevel);
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
        stream << op->GetDotAttributes(DetailLevel::High).m_Label;
        stream << "\n";
        stream << "\nInput Buffers: \n\n";
        auto inputBufs = graph.GetInputs(op);
        for (auto inputBuf : inputBufs)
        {
            stream << inputBuf->GetDotAttributes(DetailLevel::High).m_Label;
        }
        stream << "Output Buffers: \n\n";
        auto outputBuf = graph.GetOutput(op);
        if (outputBuf)
        {
            stream << outputBuf->GetDotAttributes(DetailLevel::High).m_Label;
        }
        stream << "\n";
    }
    stream << "-------------------------------------------------------------------------\n";
}

// Dump a map to the output stream but sort the output so the results are deterministic.
template <typename T>
void DumpMapInSortedOrder(const T& map,
                          std::ostream& stream,
                          const NodeIds& nodeIds,
                          const std::string& additionalOptions = "")
{
    std::vector<std::string> nodeIdsOrdered;
    for (auto&& srcAndDest : map)
    {
        nodeIdsOrdered.push_back(nodeIds.at(srcAndDest.first) + " -> " + nodeIds.at(srcAndDest.second) +
                                 additionalOptions + "\n");
    }
    std::sort(nodeIdsOrdered.begin(), nodeIdsOrdered.end(),
              [](const std::string& left, const std::string& right) { return left < right; });
    for (const std::string& str : nodeIdsOrdered)
    {
        stream << str;
    }
}

template <typename T>
void DumpMapInSortedOrderReverse(const T& map,
                                 std::ostream& stream,
                                 const NodeIds& nodeIds,
                                 const std::string& additionalOptions = "")
{
    std::vector<std::string> nodeIdsOrdered;
    for (auto&& srcAndDest : map)
    {
        nodeIdsOrdered.push_back(nodeIds.at(srcAndDest.second) + " -> " + nodeIds.at(srcAndDest.first) +
                                 additionalOptions + "\n");
    }
    std::sort(nodeIdsOrdered.begin(), nodeIdsOrdered.end(),
              [](const std::string& left, const std::string& right) { return left < right; });
    for (const std::string& str : nodeIdsOrdered)
    {
        stream << str;
    }
}

void SaveCombinationToDot(const Combination& combination, std::ostream& stream, DetailLevel detailLevel)
{
    stream << "digraph SupportLibraryGraph"
           << "\n";
    stream << "{"
           << "\n";

    NodeIds nodeIds;
    std::unordered_map<PartInputSlot, std::string> edgeInputs;

    for (PartId partId = combination.GetFirstPartId(); partId < combination.GetEndPartId(); ++partId)
    {
        const Elem& elem = combination.GetElem(partId);
        const Plan& plan = *elem.m_Plan;

        // Save Plans as isolated subgraph
        DotAttributes attr = GetDotAttributes(&plan, detailLevel);
        attr.m_Label       = "Part " + std::to_string(partId) + ": " + attr.m_Label;
        DumpSubgraphHeaderToDotFormat(attr, stream);
        NodeIds newNodeIds = SaveOpGraphAsBody(plan.m_OpGraph, stream, detailLevel);
        nodeIds.insert(newNodeIds.begin(), newNodeIds.end());
        stream << "}"
               << "\n";

        // Construct an ordered map from the unordered map so we have consistent visualisation output
        const std::unordered_map<PartInputSlot, std::shared_ptr<StartingGlue>>& startingGlues = elem.m_StartingGlues;
        std::map<PartInputSlot, std::shared_ptr<StartingGlue>> startingGluesOrdered(startingGlues.begin(),
                                                                                    startingGlues.end());
        for (const auto& slotAndStartingGlue : startingGluesOrdered)
        {
            StartingGlue* startingGlue = slotAndStartingGlue.second.get();
            std::string glueLabel      = "Part " + std::to_string(partId) + " " + plan.m_DebugTag + " Starting Glue";
            DotAttributes attr(SanitizeId(glueLabel), glueLabel, "");
            DumpSubgraphHeaderToDotFormat(attr, stream);
            NodeIds newNodeIds = SaveOpGraphAsBody(startingGlue->m_Graph, stream, detailLevel);
            nodeIds.insert(newNodeIds.begin(), newNodeIds.end());
            stream << "}"
                   << "\n";

            // Add the connections
            // Note the replacement buffers are represented in the glue
            // as the key being the buffer to be replaced and the value is the buffer which replaces it
            // In the visualisation both buffers should be shown
            // but the buffer being replaced should be "on top" so the arrow of the connection needs to be swapped,
            // so that GraphViz arranges top-to-bottom, but we want the arrow to visually point up (from the buffer
            // in the plan to the replacement buffer), so we use dir=back.
            DumpMapInSortedOrder(startingGlue->m_ExternalConnections.m_BuffersToOps, stream, nodeIds);
            DumpMapInSortedOrder(startingGlue->m_ExternalConnections.m_OpsToBuffers, stream, nodeIds);
            DumpMapInSortedOrderReverse(startingGlue->m_ExternalConnections.m_ReplacementBuffers, stream, nodeIds,
                                        "[style = dashed, label=\"Replaced by\", dir=\"back\"]");
        }
        const std::unordered_map<PartOutputSlot, std::shared_ptr<EndingGlue>>& endingGlues = elem.m_EndingGlues;
        std::map<PartOutputSlot, std::shared_ptr<EndingGlue>> endingGluesOrdered(endingGlues.begin(),
                                                                                 endingGlues.end());
        for (const auto& slotAndEndingGlue : endingGluesOrdered)
        {
            EndingGlue* endingGlue = slotAndEndingGlue.second.get();
            std::string glueLabel  = "Part " + std::to_string(partId) + " " + plan.m_DebugTag + " Ending Glue";
            DotAttributes attr(SanitizeId(glueLabel), glueLabel, "");
            DumpSubgraphHeaderToDotFormat(attr, stream);
            NodeIds newNodeIds = SaveOpGraphAsBody(endingGlue->m_Graph, stream, detailLevel);
            nodeIds.insert(newNodeIds.begin(), newNodeIds.end());
            stream << "}"
                   << "\n";

            // Add the connections
            DumpMapInSortedOrder(endingGlue->m_ExternalConnections.m_BuffersToOps, stream, nodeIds);
            DumpMapInSortedOrder(endingGlue->m_ExternalConnections.m_OpsToBuffers, stream, nodeIds);
            DumpMapInSortedOrder(endingGlue->m_ExternalConnections.m_ReplacementBuffers, stream, nodeIds,
                                 "[style = dashed, label=\"Replaced by\"]");
        }
    }

    stream << "}"
           << "\n";
}

}    // namespace support_library
}    // namespace ethosn
