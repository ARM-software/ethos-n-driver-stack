//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "Graph.hpp"

#include "DebuggingContext.hpp"
#include "GraphNodes.hpp"
#include "NetworkToGraphConverter.hpp"
#include "SramAllocator.hpp"
#include "nonCascading/Pass.hpp"
#include "nonCascading/Section.hpp"

#include <ethosn_utils/Strings.hpp>

#include <sstream>
#include <unordered_map>

namespace ethosn
{
namespace support_library
{

bool IsCompressed(CompilerDataCompressedFormat compressedFormat)
{
    return compressedFormat != CompilerDataCompressedFormat::NONE;
}

Node::Node(NodeId id,
           const TensorShape& outputTensorShape,
           DataType outputDataType,
           const QuantizationInfo& outputQuantizationInfo,
           CompilerDataFormat format,
           std::set<uint32_t> correspondingOperationIds)
    : m_Id(id)
    , m_Shape(outputTensorShape)
    , m_DataType(outputDataType)
    , m_QuantizationInfo(outputQuantizationInfo)
    , m_Format(format)
    , m_OptimizationHint(OptimizationHint::DontCare)
    , m_LocationHint(LocationHint::PreferSram)
    , m_CompressionHint(CompressionHint::PreferCompressed)
    , m_FixGraphConvertOutputTo(CompilerDataFormat::NONE)
    , m_FixGraphLocationHint(LocationHint::PreferSram)
    , m_FixGraphCompressionHint(CompressionHint::PreferCompressed)
    , m_Pass(nullptr)
    , m_Location(BufferLocation::None)
    , m_CompressionFormat(CompilerDataCompressedFormat::NONE)
    , m_BufferId(0xFFFFFFFF)
    , m_CorrespondingOperationIds(correspondingOperationIds)
{
    Reset();
}

Node::~Node()
{}

NodeId Node::GetId() const
{
    return m_Id;
}

std::set<uint32_t> Node::GetCorrespondingOperationIds() const
{
    return m_CorrespondingOperationIds;
}

void Node::AddCorrespondingOperationIDs(std::set<uint32_t> newIds)
{
    for (auto it : newIds)
    {
        m_CorrespondingOperationIds.insert(it);
    }
}

const std::vector<Edge*>& Node::GetInputs() const
{
    return m_Inputs;
}

ethosn::support_library::Edge* Node::GetOutput(uint32_t idx)
{
    return m_Outputs[idx];
}

const ethosn::support_library::Edge* Node::GetOutput(uint32_t idx) const
{
    return m_Outputs[idx];
}

TensorShape Node::GetInputShape(uint32_t inputIdx) const
{
    return m_Inputs[inputIdx]->GetSource()->GetShape();
}

DataType Node::GetInputDataType(uint32_t inputIdx) const
{
    return m_Inputs[inputIdx]->GetSource()->GetDataType();
}

QuantizationInfo Node::GetInputQuantizationInfo(uint32_t inputIdx) const
{
    return m_Inputs[inputIdx]->GetSource()->GetQuantizationInfo();
}

CompilerDataFormat Node::GetInputFormat(uint32_t inputIdx) const
{
    return m_Inputs[inputIdx]->GetSource()->GetFormat();
}

CompilerDataCompressedFormat Node::GetInputCompressedFormat(uint32_t inputIdx) const
{
    return m_Inputs[inputIdx]->GetSource()->GetCompressedFormat();
}

BufferLocation Node::GetInputLocation(uint32_t inputIdx) const
{
    return m_Inputs[inputIdx]->GetSource()->GetLocation();
}

command_stream::DataFormat Node::GetInputBufferFormat(uint32_t inputIdx) const
{
    return m_Inputs[inputIdx]->GetSource()->GetBufferFormat();
}

command_stream::DataFormat Node::GetBufferFormat() const
{
    if (m_CompressionFormat == CompilerDataCompressedFormat::NONE)
    {
        switch (m_Format)
        {
            case CompilerDataFormat::NHWCB:
                return command_stream::DataFormat::NHWCB;
            case CompilerDataFormat::NHWC:
                return command_stream::DataFormat::NHWC;
            case CompilerDataFormat::NCHW:
                return command_stream::DataFormat::NCHW;
            default:
                ETHOSN_FAIL_MSG("Unknown buffer format");
                return command_stream::DataFormat::
                    WEIGHT_STREAM;    // Return something unusual to help indicate an error
        }
    }
    else
    {
        switch (m_CompressionFormat)
        {
            case CompilerDataCompressedFormat::FCAF_DEEP:
                return command_stream::DataFormat::FCAF_DEEP;
            case CompilerDataCompressedFormat::FCAF_WIDE:
                return command_stream::DataFormat::FCAF_WIDE;
            default:
                ETHOSN_FAIL_MSG("Unknown buffer compression format");
                return command_stream::DataFormat::
                    WEIGHT_STREAM;    // Return something unusual to help indicate an error
        }
    }
}

uint32_t Node::GetInputSramOffset(uint32_t inputIdx) const
{
    return m_Inputs[inputIdx]->GetSource()->GetOutputSramOffset();
}

uint32_t Node::GetOutputSramOffset() const
{
    return m_SramOffset;
}

void Node::SetOutputSramOffset(uint32_t offset)
{
    m_SramOffset = offset;
}

bool Node::GetInputCompressed(uint32_t inputIdx) const
{
    return m_Inputs[inputIdx]->GetSource()->GetCompressed();
}

ethosn::support_library::TensorShape Node::GetShape() const
{
    return m_Shape;
}

ethosn::support_library::DataType Node::GetDataType() const
{
    return m_DataType;
}

ethosn::support_library::QuantizationInfo Node::GetQuantizationInfo() const
{
    return m_QuantizationInfo;
}

ethosn::support_library::CompilerDataFormat Node::GetFormat() const
{
    return m_Format;
}

ethosn::support_library::BufferLocation Node::GetLocation() const
{
    return m_Location;
}

void Node::SetLocation(BufferLocation l)
{
    m_Location = l;
}

bool Node::GetCompressed() const
{
    return (m_CompressionFormat != CompilerDataCompressedFormat::NONE);
}

CompilerDataCompressedFormat Node::GetCompressedFormat() const
{
    return m_CompressionFormat;
}

void Node::SetCompressedFormat(CompilerDataCompressedFormat format)
{
    if (format != CompilerDataCompressedFormat::NONE)
    {
        assert(m_Format == CompilerDataFormat::NHWCB);
    }

    m_CompressionFormat = format;
}

ethosn::support_library::Pass* Node::GetPass() const
{
    return m_Pass;
}

void Node::SetPass(Pass* pass)
{
    m_Pass = pass;
}

void Node::PrepareAfterPassAssignment(SramAllocator& sramAllocator)
{
    // We can free the inputs to this node if outputs of the previous node are no longer needed.
    m_PreparationAttempted = true;
    // There are cases where more than one input to this node comes from the same node
    // Skip looking at nodes we have already looked at otherwise we may double free its output.
    std::set<Node*> nodesVisited;
    for (uint32_t i = 0; i < m_Inputs.size(); ++i)
    {
        Node* inputNode = GetInput(i)->GetSource();
        if (std::find(nodesVisited.begin(), nodesVisited.end(), inputNode) != nodesVisited.end())
        {
            break;
        }
        nodesVisited.insert(inputNode);
        if (GetInputLocation(i) == BufferLocation::Sram)
        {
            bool canDeallocateInput = true;
            for (const auto& node : inputNode->GetOutputs())
            {
                //Keep the node sram offset until all of its outputs nodes have been assigned a pass
                //we still need to deallocate its inputs otherwise nodes that fail to prepare will
                //leave their inputs in SRAM for the whole preparation iteration.
                if (!node->GetDestination()->m_PreparationAttempted)
                {
                    canDeallocateInput = false;
                    break;
                }
            }
            if (canDeallocateInput)
            {
                bool freed = sramAllocator.Free(inputNode->GetId(), inputNode->m_SramOffset);
                ETHOSN_UNUSED(freed);
                assert(freed);
            }
        }
    }
}

bool Node::FixGraph(Graph& graph, FixGraphSeverity)
{
    bool changed = false;
    if (m_FixGraphLocationHint == LocationHint::RequireDram && m_FixGraphLocationHint != m_LocationHint)
    {
        SetLocationHint(LocationHint::RequireDram);
        m_FixGraphLocationHint = LocationHint::PreferSram;
        changed                = true;
    }
    if (m_FixGraphCompressionHint == CompressionHint::RequiredUncompressed &&
        m_FixGraphCompressionHint != m_CompressionHint)
    {
        SetCompressionHint(CompressionHint::RequiredUncompressed);
        m_FixGraphCompressionHint = CompressionHint::PreferCompressed;
        changed                   = true;
    }
    if (m_FixGraphConvertOutputTo != CompilerDataFormat::NONE)
    {
        if (GetOutputs().size() == 1)    // Not supported for other cases
        {
            CompilerDataFormat requiredFormat = m_FixGraphConvertOutputTo;

            // First check if we already have a FormatConversionNode on our output.
            // If we do then don't add another otherwise it could lead to the preparation loop getting stuck
            // and repeatedly adding more nodes with no benefit.
            FormatConversionNode* existing = dynamic_cast<FormatConversionNode*>(GetOutput(0)->GetDestination());
            if (existing == nullptr || existing->GetFormat() != requiredFormat)
            {
                // Note that we need to add *two* FormatConversionNodes - one to convert to the format that we want
                // and then another to convert back to the original format.
                // The reason we need to convert back is that the format of layers in the graph is one of their fundamental
                // properties and will affect the operation of some nodes (e.g. Reinterpret, which relies on the layout
                // if its input).
                // Changing the format that is input into whatever node consumes our output could therefore invalidate
                // that node and change the meaning of the graph, which we don't want.
                // The McePlePass can simply include *one* of the two FormatConversionNodes for what it needs, and the other
                // can be handled by the preceding/following pass.
                FormatConversionNode* firstConversion = graph.CreateAndAddNodeWithDebug<FormatConversionNode>(
                    "FixGraphConvertOutputTo First", GetShape(), GetDataType(), GetQuantizationInfo(), requiredFormat,
                    GetCorrespondingOperationIds());
                firstConversion->SetOptimizationHint(
                    OptimizationHint::
                        DoNotMerge);    // Prevent the two nodes from being merged by optimization - otherwise we won't be able to use it in McePlePass.
                graph.SplitEdge(GetOutput(0), firstConversion);
                FormatConversionNode* secondConversion = graph.CreateAndAddNodeWithDebug<FormatConversionNode>(
                    "FixGraphConvertOutputTo Second", GetShape(), GetDataType(), GetQuantizationInfo(), GetFormat(),
                    GetCorrespondingOperationIds());
                graph.SplitEdge(firstConversion->GetOutput(0), secondConversion);

                m_FixGraphConvertOutputTo = CompilerDataFormat::NONE;    // Already done.
                changed                   = true;
            }
        }
    }

    return changed;
}

void Node::Generate(command_stream::CommandStreamBuffer& cmdStream, BufferManager& bufferManager, bool dumpRam)
{
    // Not all nodes are in passes
    if (m_Pass && !m_Pass->IsGenerated())
    {
        m_Pass->Generate(cmdStream, bufferManager, dumpRam);
    }
}

void Node::Estimate(NetworkPerformanceData& perfData, const EstimationOptions& estimationOptions)
{
    // If the node cannot be prepared it is recorded as a failure
    if (!IsPrepared())
    {
        for (const auto it : GetCorrespondingOperationIds())
        {
            perfData.m_OperationIdFailureReasons.emplace(it, "Support library failed to estimate operation");
        }
    }
    // Not all nodes are in passes
    if (m_Pass && !m_Pass->IsEstimated())
    {
        m_Pass->Estimate(perfData.m_Stream, estimationOptions);
    }
}

std::string Node::DumpToDotFormat(std::ostream& stream)
{
    DotAttributes attr = GetDotAttributes();
    std::string label  = ethosn::utils::ReplaceAll(attr.m_Label, "\n", "\\n");
    stream << attr.m_Id << "[";
    stream << "label = \"" << label << "\""
           << "\n";
    if (attr.m_Color.size() > 0)
    {
        stream << ", color = " << attr.m_Color;
    }
    stream << "]\n";
    return attr.m_Id;
}

void Node::Reset()
{
    m_PreparationAttempted = false;
    m_Pass                 = nullptr;
    m_Location             = BufferLocation::None;
    m_BufferId             = 0xFFFFFFFF;
    m_SramOffset           = 0;
    m_CompressionFormat    = CompilerDataCompressedFormat::NONE;
}

ethosn::support_library::OptimizationHint Node::GetOptimizationHint() const
{
    return m_OptimizationHint;
}

void Node::SetOptimizationHint(OptimizationHint v)
{
    m_OptimizationHint = v;
}

ethosn::support_library::LocationHint Node::GetLocationHint() const
{
    return m_LocationHint;
}

void Node::SetLocationHint(LocationHint v)
{
    m_LocationHint = v;
}

ethosn::support_library::CompressionHint Node::GetCompressionHint() const
{
    return m_CompressionHint;
}

void Node::SetCompressionHint(CompressionHint v)
{
    m_CompressionHint = v;
}

ethosn::support_library::CompilerDataFormat Node::GetFixGraphConvertOutputTo() const
{
    return m_FixGraphConvertOutputTo;
}

void Node::SetFixGraphConvertOutputTo(CompilerDataFormat v)
{
    m_FixGraphConvertOutputTo = v;
}

ethosn::support_library::LocationHint Node::GetFixGraphLocationHint() const
{
    return m_FixGraphLocationHint;
}

void Node::SetFixGraphLocationHint(LocationHint v)
{
    m_FixGraphLocationHint = v;
}

ethosn::support_library::CompressionHint Node::GetFixGraphCompressionHint() const
{
    return m_FixGraphCompressionHint;
}

void Node::SetFixGraphCompressionHint(CompressionHint v)
{
    m_FixGraphCompressionHint = v;
}

uint32_t Node::GetBufferId() const
{
    return m_BufferId;
}

void Node::SetBufferId(uint32_t v)
{
    m_BufferId = v;
}

DotAttributes Node::GetDotAttributes()
{
    std::stringstream result;
    const DebuggingContext& debuggingContext = GetDebuggingContext();

    result << "Node Id: " << m_Id << "\n";
    result << "Creation source:" << debuggingContext.GetStringFromNode(this) << "\n";
    result << "CorrespondingOperationIds:";
    for (auto id : m_CorrespondingOperationIds)
    {
        result << " " << id;
    }
    result << "\n";

    result << ToString(m_Shape) << " ";
    result << "Format = " << ToString(m_Format) << "\n";
    result << "CompressedFormat = " << ToString(m_CompressionFormat) << "\n";
    result << "Quant. Info = " << ToString(m_QuantizationInfo) << "\n";
    switch (m_OptimizationHint)
    {
        case OptimizationHint::DoNotMerge:
            result << "DO NOT MERGE\n";
            break;
        default:
            break;
    }
    switch (m_LocationHint)
    {
        case LocationHint::PreferSram:
            result << "PREFER SRAM\n";
            break;
        case LocationHint::RequireDram:
            result << "REQUIRE DRAM\n";
            break;
        default:
            ETHOSN_FAIL_MSG("Unknown location hint");
    }
    switch (m_Location)
    {
        case BufferLocation::None:
            result << "Location = NONE\n";
            break;
        case BufferLocation::Dram:
            result << "DRAM, BUFFER 0x" << std::hex << m_BufferId << std::dec << " (" << m_BufferId << ")\n";
            break;
        case BufferLocation::Sram:
            result << "SRAM, BUFFER 0x" << std::hex << m_BufferId << std::dec << " (" << m_BufferId << ")\n";
            break;
        default:
            ETHOSN_FAIL_MSG("Unknown location");
    }
    switch (m_CompressionHint)
    {
        case CompressionHint::PreferCompressed:
            result << "PREFER COMPRESSED\n";
            break;
        case CompressionHint::RequiredUncompressed:
            result << "REQUIRE UNCOMPRESSED\n";
            break;
        default:
            ETHOSN_FAIL_MSG("Unknown compression hint");
    }
    result << "Optimization Hint:";
    switch (m_OptimizationHint)
    {
        case OptimizationHint::DontCare:
            result << "DONT CARE\n";
            break;
        case OptimizationHint::DoNotMerge:
            result << "DO NOT MERGE\n";
            break;
        default:
            ETHOSN_FAIL_MSG("Unknown optimization hint");
    }
    std::string color = IsPrepared() ? "green" : "red";
    return DotAttributes(std::to_string(GetId()), result.str(), color);
}

const ethosn::support_library::Edge* Node::GetInput(uint32_t idx) const
{
    return m_Inputs[idx];
}

ethosn::support_library::Edge* Node::GetInput(uint32_t idx)
{
    return m_Inputs[idx];
}

const std::vector<Edge*>& Node::GetOutputs() const
{
    return m_Outputs;
}

Edge::Edge(Node* source, Node* destination)
    : m_Source(source)
    , m_Destination(destination)
{}

const ethosn::support_library::Node* Edge::GetSource() const
{
    return m_Source;
}

ethosn::support_library::Node* Edge::GetSource()
{
    return m_Source;
}

const ethosn::support_library::TensorShape Edge::GetSourceShape() const
{
    return m_Source->GetShape();
}

ethosn::support_library::TensorShape Edge::GetSourceShape()
{
    return m_Source->GetShape();
}

ethosn::support_library::Node* Edge::GetDestination()
{
    return m_Destination;
}

const ethosn::support_library::Node* Edge::GetDestination() const
{
    return m_Destination;
}

Graph::Graph(const Network& network,
             const HardwareCapabilities& capabilities,
             const EstimationOptions& estimationOptions,
             bool strictPrecision)
    : Graph()
{
    NetworkToGraphConverter converter(
        *this, capabilities,
        network.IsEstimationMode() ? estimationOptions : utils::Optional<const EstimationOptions&>(), strictPrecision);
    network.Accept(converter);
}

const std::vector<std::unique_ptr<ethosn::support_library::Node>>& Graph::GetNodes() const
{
    return m_Nodes;
}

std::vector<Node*> Graph::GetNodesSorted() const
{
    std::vector<Node*> targets;
    for (const auto& node : m_Nodes)
    {
        if (node.get()->GetOutputs().size() == 0)
        {
            targets.push_back(node.get());
        }
    }
    std::vector<Node*> sorted;
    utils::GraphTopologicalSort<Node*, std::vector<Node*>>(
        targets,
        [](Node* n) {
            std::vector<Node*> result;
            std::transform(n->GetInputs().begin(), n->GetInputs().end(), std::back_inserter(result),
                           [](auto x) -> Node* { return x->GetSource(); });
            return result;
        },
        sorted);
    return sorted;
}

const std::vector<std::unique_ptr<Edge>>& Graph::GetEdges() const
{
    return m_Edges;
}

void Graph::AddNode(std::unique_ptr<Node> node)
{
    m_Nodes.push_back(std::move(node));
}

void Graph::Connect(Node* source, Node* destination, int32_t insertionIdx)
{
    std::unique_ptr<Edge> e = std::make_unique<Edge>(source, destination);
    Edge* e2                = e.get();
    m_Edges.push_back(std::move(e));

    source->m_Outputs.push_back(e2);
    if (insertionIdx == -1)
    {
        destination->m_Inputs.push_back(e2);
    }
    else
    {
        destination->m_Inputs.insert(destination->m_Inputs.begin() + insertionIdx, e2);
    }
}

void Graph::RemoveNode(Node* node)
{
    std::vector<Edge*> edges = node->GetInputs();
    for (Edge* e : edges)
    {
        RemoveEdge(e);
    }
    edges = node->GetOutputs();
    for (Edge* e : edges)
    {
        RemoveEdge(e);
    }
    {
        auto it = std::find_if(m_Nodes.begin(), m_Nodes.end(),
                               [&](const std::unique_ptr<Node>& n) { return n.get() == node; });
        assert(it != m_Nodes.end());
        m_Nodes.erase(it);
    }
}

int32_t Graph::RemoveEdge(Edge* edge)
{
    {
        auto it = std::find(edge->GetSource()->m_Outputs.begin(), edge->GetSource()->m_Outputs.end(), edge);
        assert(it != edge->GetSource()->m_Outputs.end());
        edge->GetSource()->m_Outputs.erase(it);
    }
    int32_t index;
    {
        auto it = std::find(edge->GetDestination()->m_Inputs.begin(), edge->GetDestination()->m_Inputs.end(), edge);
        assert(it != edge->GetDestination()->m_Inputs.end());
        index = static_cast<int32_t>(it - edge->GetDestination()->m_Inputs.begin());
        edge->GetDestination()->m_Inputs.erase(it);
    }
    {
        auto it = std::find_if(m_Edges.begin(), m_Edges.end(),
                               [&](const std::unique_ptr<Edge>& e) { return e.get() == edge; });
        assert(it != m_Edges.end());
        m_Edges.erase(it);
    }
    return index;
}

void Graph::SplitEdge(Edge* edge, Node* newNode)
{
    Node* first   = edge->GetSource();
    Node* last    = edge->GetDestination();
    int32_t index = RemoveEdge(edge);
    Connect(first, newNode);
    Connect(newNode, last, index);
}

void Graph::CollapseEdge(Edge* edge)
{
    Node* source = edge->GetSource();
    Node* dest   = edge->GetDestination();
    std::vector<std::pair<Node*, int32_t>> newDestsAndIndices;
    for (const auto& e : dest->GetOutputs())
    {
        auto newDest  = e->GetDestination();
        int32_t index = static_cast<int32_t>(utils::FindIndex(newDest->GetInputs(), e).second);
        newDestsAndIndices.push_back({ newDest, index });
    }
    RemoveNode(dest);
    for (auto n : newDestsAndIndices)
    {
        Connect(source, n.first, n.second);
    }
}

void Graph::CollapseNode(Node* node)
{
    std::vector<Edge*> outgoingEdges = node->GetOutputs();
    for (Edge* outgoingEdge : outgoingEdges)
    {
        Node* outNode = outgoingEdge->GetDestination();

        int32_t inputIdx = static_cast<int32_t>(
            std::distance(outNode->GetInputs().begin(),
                          std::find(outNode->GetInputs().begin(), outNode->GetInputs().end(), outgoingEdge)));
        for (Edge* inputEdge : node->GetInputs())
        {
            Node* inputNode = inputEdge->GetSource();
            Connect(inputNode, outNode, inputIdx);
            inputIdx++;
        }
        RemoveEdge(outgoingEdge);
    }

    RemoveNode(node);
}

void Graph::InsertNodeAfter(Node* position, Node* newNode)
{
    std::vector<Edge*> outputs = position->GetOutputs();    // Copy the output edges as these will change as we loop
    for (Edge* e : outputs)
    {
        Node* dest       = e->GetDestination();
        int32_t inputIdx = RemoveEdge(e);
        Connect(newNode, dest, inputIdx);
    }
    Connect(position, newNode);
}

NodeId Graph::GenerateNodeId()
{
    return m_NextNodeId++;
}

void Graph::DumpToDotFormat(std::ostream& stream) const
{
    stream << "digraph SupportLibraryGraph"
           << "\n";
    stream << "{"
           << "\n";

    std::unordered_map<Node*, std::string> nodeIds;
    std::unordered_map<Pass*, std::vector<Node*>> passes;
    std::unordered_map<Section*, std::vector<Pass*>> sections;
    for (auto&& n : m_Nodes)
    {
        Pass* p = n->GetPass();
        passes[p].push_back(n.get());
    }

    for (auto&& p : passes)
    {
        Section* s = (p.first == nullptr) ? nullptr : p.first->GetSection();
        sections[s].push_back(p.first);
    }

    for (auto&& section : sections)
    {
        if (section.first != nullptr)
        {
            DotAttributes attr = section.first->GetDotAttributes();
            stream << "subgraph clusterSection" << attr.m_Id << "\n";
            stream << "{"
                   << "\n";
            stream << "label=\"" << ethosn::utils::ReplaceAll(attr.m_Label, "\n", "\\n") << "\""
                   << "\n";
            stream << "color = " << attr.m_Color << "\n";
            stream << "labeljust=l"
                   << "\n";
        }

        for (auto&& p : section.second)
        {
            if (p != nullptr)
            {
                DotAttributes attr = p->GetDotAttributes();
                stream << "subgraph clusterPass" << attr.m_Id << "\n";
                stream << "{"
                       << "\n";
                stream << "label=\"" << ethosn::utils::ReplaceAll(attr.m_Label, "\n", "\\n") << "\""
                       << "\n";
                stream << "color = " << attr.m_Color << "\n";
                stream << "labeljust=l"
                       << "\n";
            }

            for (auto&& n : passes[p])
            {
                std::string id = n->DumpToDotFormat(stream);
                nodeIds[n]     = id;
            }

            if (p != nullptr)
            {
                stream << "}"
                       << "\n";
            }
        }

        if (section.first != nullptr)
        {
            stream << "}"
                   << "\n";
        }
    }

    for (auto&& e : m_Edges)
    {
        std::pair<bool, size_t> edgeInput = utils::FindIndex(e->GetDestination()->GetInputs(), e.get());
        stream << nodeIds[e->GetSource()] << " -> " << nodeIds[e->GetDestination()] << "[ label=\"" << edgeInput.second
               << "\"]\n";
    }
    stream << "}"
           << "\n";
}

}    // namespace support_library
}    // namespace ethosn
