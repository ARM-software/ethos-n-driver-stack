//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "BufferManager.hpp"
#include "Network.hpp"
#include "cascading/Visualisation.hpp"

#include <memory>
#include <vector>

namespace ethosn
{
namespace support_library
{

class Network;

class Pass;

class Graph;
class Edge;

class BufferManager;
struct CompilerBufferInfo;
class SramAllocator;

enum class CompilerDataFormat
{
    NONE,
    NHWC,
    NCHW,
    NHWCB,
    WEIGHT,
    NHWCB_COMPRESSED,
    FCAF_DEEP,
    FCAF_WIDE,
};

inline CompilerDataFormat ConvertExternalToCompilerDataFormat(DataFormat dataFormat)
{
    assert(dataFormat == DataFormat::NHWC || dataFormat == DataFormat::NHWCB || dataFormat == DataFormat::HWIO ||
           dataFormat == DataFormat::HWIM);
    if (dataFormat == DataFormat::NHWC)
    {
        return CompilerDataFormat::NHWC;
    }
    else if (dataFormat == DataFormat::NHWCB)
    {
        return CompilerDataFormat::NHWCB;
    }
    else
    {
        return CompilerDataFormat::WEIGHT;
    }
}

enum class OptimizationHint
{
    DontCare,
    DoNotMerge
};

enum class LocationHint
{
    PreferSram,
    RequireDram,
};

enum class CompressionHint
{
    PreferCompressed,
    RequiredUncompressed,
};

enum class FixGraphSeverity
{
    Low,
    High,
    // 'Meta' values required for easy iteration over all severity values. Update these if the above values are changed.
    Lowest  = Low,
    Highest = High
};

using NodeId = size_t;

/// Has 0 or more input edges and produces exactly 1 output, which can be connected to zero or more output edges.
class Node
{
public:
    Node(NodeId id,
         const TensorShape& outputTensorShape,
         const QuantizationInfo& outputQuantizationInfo,
         CompilerDataFormat format,
         std::set<uint32_t> correspondingOperationIds);
    virtual ~Node();

    /// Connections
    /// @{
    const std::vector<Edge*>& GetInputs() const;
    const Edge* GetInput(uint32_t idx) const;
    Edge* GetInput(uint32_t idx);
    const std::vector<Edge*>& GetOutputs() const;
    const Edge* GetOutput(uint32_t idx) const;
    Edge* GetOutput(uint32_t idx);
    /// @}

    /// Fixed properties
    /// @{
    TensorShape GetShape() const;
    QuantizationInfo GetQuantizationInfo() const;
    CompilerDataFormat GetFormat() const;
    void SetFormat(CompilerDataFormat format);

    TensorShape GetInputShape(uint32_t inputIdx) const;
    QuantizationInfo GetInputQuantizationInfo(uint32_t inputIdx) const;
    CompilerDataFormat GetInputFormat(uint32_t inputIdx) const;
    command_stream::DataFormat GetInputBufferFormat(uint32_t inputIdx) const;
    bool GetInputCompressed(uint32_t inputIdx) const;
    CompilerDataFormat GetInputCompressedFormat(uint32_t inputIdx) const;
    /// @}

    /// Preparation hints
    /// @{
    OptimizationHint GetOptimizationHint() const;
    void SetOptimizationHint(OptimizationHint v);

    LocationHint GetLocationHint() const;
    void SetLocationHint(LocationHint v);

    CompressionHint GetCompressionHint() const;
    void SetCompressionHint(CompressionHint v);
    /// @}

    /// Fix graph hints
    /// @{
    CompilerDataFormat GetFixGraphConvertOutputTo() const;
    void SetFixGraphConvertOutputTo(CompilerDataFormat v);

    LocationHint GetFixGraphLocationHint() const;
    void SetFixGraphLocationHint(LocationHint v);

    CompressionHint GetFixGraphCompressionHint() const;
    void SetFixGraphCompressionHint(CompressionHint v);
    /// @}

    /// Preparation results
    /// @{
    Pass* GetPass() const;
    void SetPass(Pass* pass);

    BufferLocation GetLocation() const;
    void SetLocation(BufferLocation l);

    bool GetCompressed() const;
    CompilerDataFormat GetCompressedFormat() const;

    BufferLocation GetInputLocation(uint32_t inputIdx) const;

    command_stream::DataFormat GetBufferFormat() const;

    uint32_t GetInputSramOffset(uint32_t inputIdx) const;
    uint32_t GetOutputSramOffset() const;
    void SetOutputSramOffset(uint32_t offset);
    /// @}

    /// Generation results
    /// @{
    uint32_t GetBufferId() const;
    void SetBufferId(uint32_t v);
    /// @}

    /// Preparation methods
    /// @{
    virtual void Reset();
    virtual void PrepareAfterPassAssignment(SramAllocator& sramAllocator);
    virtual bool IsPrepared() = 0;    // Subclasses must implement this and perform their own checks.

    /// Attempts to make changes to the graph in order to allow this node to be prepared in the next iteration.
    /// This could, for example, change the hints on some nodes or add a new node to the graph.
    /// The severity parameter allows some modifications to be made only if absolutely necessary (i.e. no other
    /// changes to the graph were sufficient).
    virtual bool FixGraph(Graph& graph, FixGraphSeverity severity);
    /// @}

    /// Generation methods
    /// @{
    virtual void Generate(command_stream::CommandStreamBuffer& cmdStream, BufferManager& bufferManager, bool dumpRam);
    /// @}

    /// Performance estimation methods
    /// @{
    virtual void Estimate(NetworkPerformanceData& perfStream, const EstimationOptions& estimationOptions);
    /// @}

    /// Debugging methods
    /// @{
    std::string DumpToDotFormat(std::ostream& stream);
    /// @}

    NodeId GetId() const;
    std::set<uint32_t> GetCorrespondingOperationIds() const;
    // When a node is collapsed, we need to record the mapping between the dead node to the input network operation.
    void AddCorrespondingOperationIDs(std::set<uint32_t> newIds);

protected:
    friend Graph;    // Only the Graph can manipulate a Node's connections.

    virtual DotAttributes GetDotAttributes();

    NodeId m_Id;

    std::vector<Edge*> m_Inputs;
    std::vector<Edge*> m_Outputs;

    // Abstract properties of the output - don't require the tensor to actually exist anywhere in SRAM/DRAM
    TensorShape m_Shape;
    QuantizationInfo m_QuantizationInfo;
    CompilerDataFormat m_Format;

    // Preparation hints
    OptimizationHint m_OptimizationHint;
    LocationHint m_LocationHint;
    CompressionHint m_CompressionHint;

    // Fix graph hints
    CompilerDataFormat m_FixGraphConvertOutputTo;
    LocationHint m_FixGraphLocationHint;
    CompressionHint m_FixGraphCompressionHint;

    // Set during preparation, but cleared after each iteration
    bool m_PreparationAttempted;
    Pass* m_Pass;
    BufferLocation m_Location;
    //If this node's output will remain in SRAM then this is the offset at which it will be kept.
    //This is used by later nodes to determine where their inputs can be found.
    //At the generation stage this data will be placed into the BufferManager.
    uint32_t m_SramOffset;

    // Set during generation.
    uint32_t m_BufferId;

    // The ids of the operations in the input graph that this Node corresponds to
    std::set<uint32_t> m_CorrespondingOperationIds;
};

class Edge
{
public:
    Edge(Node* source, Node* destination);

    const Node* GetSource() const;
    const Node* GetDestination() const;
    Node* GetSource();
    Node* GetDestination();

private:
    Node* m_Source;
    Node* m_Destination;
};

class Graph
{
public:
    Graph()
        : m_Nodes()
        , m_Edges()
        , m_NextNodeId(0)
    {}

    Graph(const Network& network, const HardwareCapabilities& capabilities);

    const std::vector<std::unique_ptr<Node>>& GetNodes() const;
    std::vector<Node*> GetNodesSorted() const;

    const std::vector<std::unique_ptr<Edge>>& GetEdges() const;

    /// Constructs a new node of type TNode and adds it to this graph. The new node will initially have no connections.
    /// The arguments are forwarded to the node's constructor.
    template <typename TNode, typename... Args>
    TNode* CreateAndAddNode(Args&&... args);

    /// Connects two nodes together with a directed edge.
    /// insertionIdx specifies the index of the *incoming* connection to destination
    /// (the order of outgoing connections has no relevance).
    void Connect(Node* source, Node* destination, int32_t insertionIdx = -1);

    /// Removes a node from this graph, implicitly disconnecting it from all inputs and outputs.
    void RemoveNode(Node* node);
    /// Removes an edge and returns the index into the destination node of the removed edge.
    int32_t RemoveEdge(Edge* edge);
    /// Splits the given edge by inserting a new node along that edge.
    void SplitEdge(Edge* edge, Node* newNode);
    /// Removes the destination node of the given edge and moves the connections from the removed node
    /// to the source node of the removed edge.
    void CollapseEdge(Edge* edge);
    /// Removes a node and 'passes through' incoming edges to its outputs.
    void CollapseNode(Node* node);
    /// Inserts a node into this graph, immediately after the given node.
    /// A connection will be made between 'position' and 'newNode', and any outputs that 'position' used to have will be
    /// changed to come from 'position' instead.
    void InsertNodeAfter(Node* position, Node* newNode);

    void DumpToDotFormat(std::ostream& stream) const;

private:
    void AddNode(std::unique_ptr<Node> node);
    NodeId GenerateNodeId();

    std::vector<std::unique_ptr<Node>> m_Nodes;
    std::vector<std::unique_ptr<Edge>> m_Edges;
    NodeId m_NextNodeId;
};

template <typename TNode, typename... Args>
TNode* ethosn::support_library::Graph::CreateAndAddNode(Args&&... args)
{
    NodeId nodeId               = GenerateNodeId();
    std::unique_ptr<TNode> node = std::make_unique<TNode>(nodeId, std::forward<Args>(args)...);
    TNode* ptr                  = node.get();
    AddNode(std::move(node));
    return ptr;
}

template <typename Pred>
Node* SearchDependencies(Node* node, Pred pred)
{
    if (pred(node))
    {
        return node;
    }
    for (uint32_t i = 0; i < node->GetInputs().size(); ++i)
    {
        Node* found = SearchDependencies(node->GetInput(i)->GetSource(), pred);
        if (found)
        {
            return found;
        }
    }
    return nullptr;
}

}    // namespace support_library
}    // namespace ethosn
