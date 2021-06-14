//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../Graph.hpp"
#include "../GraphNodes.hpp"
#include "../WeightEncoder.hpp"

#include <ethosn_command_stream/CommandStream.hpp>

#include <map>
#include <unordered_map>

namespace ethosn
{
namespace support_library
{

enum class OpType
{
    DmaOp,
    MceOp,
    PleOp
};

enum class Lifetime
{
    Atomic,
    Cascade
};

enum class TraversalOrder
{
    Xyz,
    Zxy
};

enum class Location
{
    Dram,
    PleInputSram,
    Sram,
    VirtualSram
};

enum class CascadingBufferFormat
{
    NHWC,
    NCHW,
    NHWCB,
    WEIGHT,
    FCAF_DEEP,
    FCAF_WIDE
};

struct SizeInBytes
{
    uint32_t m_Tot       = 0;
    uint32_t m_TotAtomic = 0;
};

bool IsCompressed(CascadingBufferFormat format);

class Buffer;
class Op;

using PartId = size_t;
using PlanId = size_t;

/// A graph of connected Ops and Buffers.
///
/// Each Op takes as input zero or more Buffers, with each input associated with an index (i.e. 0th input,
/// 1st input etc), and produces zero or one Buffers. This can be used for example to represent an MceOp which takes an
/// IFM (0th input) and weights (1st input) and produces an OFM (output).
/// Each Buffer is produced by zero or one Ops and consumed by zero or more Ops. This can be used for example to
/// represent a tensor in Sram which is produced as the output of one MceOp and consumed as the IFM input by two
/// subsequent MceOps.
///
/// We do not currently need to support an Op producing multiple output Buffers, but this class could be extended to
/// support that if needed.
///
/// This is a non-intrusive graph in the sense that the elements of the graph (Ops and Buffers) do not store any
/// information about their existence in the graph. This makes it possible for the same element to be present in
/// multiple graphs, which may be very useful for Plans and Combinations etc.
/// This also means that OpGraph takes no ownership of the Ops and Buffers - the user is required to ensure they
/// outlive the OpGraph. See OwnedOpGraph for a way of doing this.
class OpGraph
{
public:
    using OpList        = std::vector<Op*>;
    using BufferList    = std::vector<Buffer*>;
    using ConsumersList = std::vector<std::pair<Op*, uint32_t>>;

    /// Simple queries
    /// @{
    const OpList& GetOps() const;
    const BufferList& GetBuffers() const;

    bool Contains(Op* op) const;
    bool Contains(Buffer* buffer) const;

    Op* GetProducer(Buffer* buffer) const;
    ConsumersList GetConsumers(Buffer* buffer) const;
    BufferList GetInputs(Op* op) const;
    Buffer* GetOutput(Op* op) const;
    /// @}

    /// Manipulation
    /// @{
    void AddOp(Op* op);
    void AddBuffer(Buffer* buffer);

    void SetProducer(Buffer* buffer, Op* producerOp);
    void ClearProducer(Buffer* buffer);

    void AddConsumer(Buffer* buffer, Op* consumerOp, uint32_t opInputIdx);
    /// @}

private:
    /// All of the Ops in the graph, in no particular order.
    OpList m_Ops;
    /// All of the Buffers in the graph, in no particular order.
    BufferList m_Buffers;

    /// For each Buffer in the graph, which Op produces it (if any).
    std::unordered_map<Buffer*, Op*> m_BufferProducers;
    /// For each Buffer in the graph, which Ops (and which input index of those Ops) consume it (if any).
    std::unordered_map<Buffer*, ConsumersList> m_BufferConsumers;
    /// For each Op in the graph, which Buffer does it produce (if any).
    std::unordered_map<Op*, Buffer*> m_OpOutputs;
    /// For each Op in the graph, which Buffers does it consume (if any), ordered by input index.
    std::unordered_map<Op*, BufferList> m_OpInputs;
};

/// An extension of OpGraph which additionally manages the lifetime of the Ops and Buffers.
class OwnedOpGraph : public OpGraph
{
public:
    Op* AddOp(std::unique_ptr<Op> op);
    Buffer* AddBuffer(std::unique_ptr<Buffer> buffer);

private:
    std::vector<std::unique_ptr<Op>> m_Ops;
    std::vector<std::unique_ptr<Buffer>> m_Buffers;
};

struct DebuggableObject
{
public:
    DebuggableObject(const char* defaultTagPrefix);

    /// This can be used to help identify this object for debugging purposes, and is used in visulisations (dot files)
    /// to identify this object. It shouldn't have any effect on network compilation or estimation.
    std::string m_DebugTag;
    int m_DebugId;

    /// Counter for generating unique debug tags (see DebuggableObject constructor).
    /// This is publicly exposed so can be manipulated by tests.
    static int ms_IdCounter;
};

class Plan : public DebuggableObject
{
public:
    using InputMapping  = std::map<Buffer*, Edge*>;
    using OutputMapping = std::map<Buffer*, Node*>;

    Plan();
    Plan(InputMapping&& inputMappings, OutputMapping&& outputMappings);

    /// Gets the Buffer corresponding to the given Edge, which should be an input to the Part that this Plan is for.
    /// Returns nullptr if the Edge is unrecognised.
    Buffer* GetInputBuffer(const Edge* inputEdge) const;
    /// Gets the Buffer corresponding to the given Node, which should be an output from the Part that this Plan is for.
    /// Returns nullptr if the Node is unrecognised.
    Buffer* GetOutputBuffer(const Node* outputNode) const;

    /// The graph of Ops and Buffers which define how this plan would be executed.
    OwnedOpGraph m_OpGraph;

    /// Specifies which of the Buffers in the above OpGraph are inputs to this plan, and which Edges from the Graph
    /// these correspond to.
    InputMapping m_InputMappings;
    /// Specifies which of the Buffers in the above OpGraph are outputs from this plan, and which Nodes from the Graph
    /// these correspond to.
    OutputMapping m_OutputMappings;
};

class Op : public DebuggableObject
{
public:
    Op(const char* defaultTagPrefix);
    Op(const char* defaultTagPrefix, Lifetime lifetime);
    virtual ~Op() = default;

    Lifetime m_Lifetime;
    std::set<uint32_t> m_OperationIds;
};

class DmaOp : public Op
{
public:
    DmaOp();
    DmaOp(Lifetime lifetime, Location location);

    Location m_Location;
};

class MceOp : public Op
{
public:
    MceOp();
    MceOp(Lifetime lifetime,
          command_stream::MceOperation op,
          CompilerMceAlgorithm algo,
          command_stream::BlockConfig blockConfig,
          TensorShape inputStripeShape,
          TensorShape outputStripeShape,
          TensorShape weightsStripeShape,
          TraversalOrder order,
          Stride stride,
          uint32_t padLeft,
          uint32_t padTop);

    command_stream::MceOperation m_Op;
    CompilerMceAlgorithm m_Algo;
    command_stream::BlockConfig m_BlockConfig;
    TensorShape m_InputStripeShape;
    TensorShape m_OutputStripeShape;
    TensorShape m_WeightsStripeShape;
    TraversalOrder m_Order;
    Stride m_Stride;
    uint32_t m_PadLeft;
    uint32_t m_PadTop;
};

class PleOp : public Op
{
public:
    PleOp();
    PleOp(Lifetime lifetime,
          command_stream::PleOperation op,
          command_stream::BlockConfig blockConfig,
          uint32_t numInputs,
          std::vector<TensorShape> inputStripeShapes,
          TensorShape outputStripeShape);

    command_stream::PleOperation m_Op;
    command_stream::BlockConfig m_BlockConfig;
    uint32_t m_NumInputs;
    std::vector<TensorShape> m_InputStripeShapes;
    TensorShape m_OutputStripeShape;
};

class DummyOp : public Op
{
public:
    DummyOp();
};

class Buffer : public DebuggableObject
{
public:
    Buffer();
    Buffer(Lifetime lifetime, Location location, CascadingBufferFormat format, TraversalOrder order);
    Buffer(Lifetime lifetime,
           Location location,
           CascadingBufferFormat format,
           TensorShape tensorShape,
           TensorShape stripeShape,
           TraversalOrder order,
           uint32_t sizeInBytes,
           QuantizationInfo quantInfo);

    Lifetime m_Lifetime;
    Location m_Location;
    CascadingBufferFormat m_Format;
    QuantizationInfo m_QuantizationInfo;
    TensorShape m_TensorShape;
    TensorShape m_StripeShape;
    TraversalOrder m_Order;
    uint32_t m_SizeInBytes;

    /// This value should be easily calculable from m_SizeInBytes and m_StripeShape (and possibly some format parameters),
    /// but is useful to store by itself nonetheless.
    uint32_t m_NumStripes;

    /// Relevant only if this is a weights buffer in Dram.
    std::shared_ptr<EncodedWeights> m_EncodedWeights;
};

bool IsOutputBufferInDram(const Plan& plan, const Edge& edge);
bool IsInputBufferInSram(const Plan& plan, const Edge& edge);
bool IsOutputBufferInSram(const Plan& plan, const Edge& edge);

SizeInBytes GetTotSizeInBytes(const Plan& plan);
SizeInBytes GetInputsSizeInBytes(const Plan& plan);

std::vector<Op*> GetSortedOps(const OpGraph& opGraph);

}    // namespace support_library
}    // namespace ethosn
