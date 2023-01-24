//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../WeightEncoder.hpp"
#include "DebuggableObject.hpp"
#include "Part.hpp"

#include "../include/ethosn_support_library/Optional.hpp"
#include <ethosn_command_stream/CommandStream.hpp>
#include <ethosn_command_stream/cascading/PleKernelId.hpp>

#include <map>
#include <unordered_map>

namespace ethosn
{
namespace support_library
{

class Buffer;
class Op;

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

bool IsCompressed(CascadingBufferFormat format);

/// A graph of connected Ops and Buffers.
///
/// Each Op takes as input zero or more Buffers, with each input associated with an index (i.e. 0th input,
/// 1st input etc), and produces zero or one Buffers. This can be used for example to represent an MceOp which takes an
/// IFM (0th input) and weights (1st input) and produces an OFM (output).
/// Each Buffer is produced by zero or more Ops and consumed by zero or more Ops. This can be used for example to
/// represent a tensor in Dram which is produced by two different DmaOps (both writing data into this same buffer)
/// and consumed as the input by two different subsequent DmaOps. Note that the producers of a buffer are _not_
/// ordered/numbered as they are for Op inputs.
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

    // Merge another OpGraph into the current one
    void MergeOpGraph(const OpGraph& other);

    /// Simple queries
    /// @{
    const OpList& GetOps() const;
    const BufferList& GetBuffers() const;
    Op* GetOp(uint32_t index) const;

    bool Contains(Op* op) const;
    bool Contains(Buffer* buffer) const;

    /// If the buffer has a single producer, returns it.
    /// If the buffer has no producer, returns nullptr.
    /// Otherwise (multiple producers), throws an exception. If the buffer might have
    /// multiple producers, use GetProducers instead.
    Op* GetSingleProducer(Buffer* buffer) const;
    OpList GetProducers(Buffer* buffer) const;
    ConsumersList GetConsumers(Buffer* buffer) const;
    std::pair<Op*, uint32_t> GetConsumer(Buffer* buffer, uint32_t index) const;
    BufferList GetInputs(Op* op) const;
    Buffer* GetOutput(Op* op) const;
    /// @}

    /// Manipulation
    /// @{
    void AddOp(Op* op);
    void AddBuffer(Buffer* buffer);

    void SetProducer(Buffer* buffer, Op* producerOp);
    void AddProducer(Buffer* buffer, Op* producerOp);
    void RemoveProducer(Buffer* buffer, Op* producerOp);
    void ClearProducers(Buffer* buffer);

    void AddConsumer(Buffer* buffer, Op* consumerOp, uint32_t opInputIdx);
    void RemoveConsumer(Buffer* buffer, Op* consumerOp, uint32_t opInputIdx);

    /// Removes the given Op or Buffer from this OpGraph, and then if that leaves
    /// any previously-connected Ops or Buffers without any input connections or
    /// without any output connections, then they will be removed too.
    /// This repeats recursively until encountering an Op or Buffer which has other connections.
    /// This means that calling this method on a linear graph (with no branching)
    /// will result in the OpGraph being completely emptied.
    /// For graphs with branching, this will result in an entire 'branch' being removed.
    /// @{
    void RemoveAndPrune(Op* op);
    void RemoveAndPrune(Buffer* b);
    /// @}
    /// @}

    /// Optimization step which removes sequences of Ops and Buffers which copy data into and out of SRAM
    /// multiple times and can be shortened to just a single copy.
    /// Such sequences can arise as a result of combining multiple plans together
    /// (in particular Reshape, Concat and Split) and lead to worse performance.
    /// By eliminating/simplifying these sequences, the NPU will have less work to do
    /// and so performance will be better.
    void RemoveRedundantCopies();

protected:
    void RemoveRedundantCopiesSramToDram();
    void RemoveRedundantCopiesDramToSram();

    /// All of the Ops in the graph, in no particular order.
    OpList m_Ops;
    /// All of the Buffers in the graph, in no particular order.
    BufferList m_Buffers;

    /// For each Buffer in the graph, which Ops produce it (if any).
    std::unordered_map<Buffer*, OpList> m_BufferProducers;
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
    OwnedOpGraph()
    {}
    OwnedOpGraph(const OwnedOpGraph&) = delete;
    OwnedOpGraph(OwnedOpGraph&&)      = default;
    OwnedOpGraph& operator=(OwnedOpGraph&&) = default;

    Op* AddOp(std::unique_ptr<Op> op);
    Buffer* AddBuffer(std::unique_ptr<Buffer> buffer);

    // Merge another OpGraph into the current one taking ownership of the other opgraphs ops and buffers
    void MergeOpGraph(OwnedOpGraph& other);

private:
    std::vector<std::unique_ptr<Op>> m_Ops;
    std::vector<std::unique_ptr<Buffer>> m_Buffers;
};

class Op : public DebuggableObject
{
public:
    Op(const char* defaultTagPrefix);
    virtual ~Op() = default;

    virtual utils::Optional<command_stream::BlockConfig> GetBlockConfig()
    {
        return utils::Optional<command_stream::BlockConfig>{};
    }
    virtual uint32_t GetNumberOfAgents() const
    {
        return 1;
    }

    virtual DotAttributes GetDotAttributes(DetailLevel) const override;

    std::set<uint32_t> m_OperationIds;
};

class DmaOp : public Op
{
public:
    DmaOp(CascadingBufferFormat transferFormat);
    DmaOp(const char* debugType, CascadingBufferFormat transferFormat);
    virtual DotAttributes GetDotAttributes(DetailLevel) const override;

    /// The *DRAM* format that this DmaOp converts to/from. SRAM format is always NHWCB.
    /// Normally this will match the actual format of the connected DRAM buffer, but in some
    /// cases we want to *reinterpret* the data (e.g. Fully Connected), in which case this might not match.
    CascadingBufferFormat m_TransferFormat;
    TensorShape m_Offset;
};

class MceOp : public Op
{
public:
    MceOp();
    MceOp(command_stream::MceOperation op,
          CompilerMceAlgorithm algo,
          command_stream::BlockConfig blockConfig,
          TensorShape inputStripeShape,
          TensorShape outputStripeShape,
          TensorShape weightsStripeShape,
          TraversalOrder order,
          Stride stride,
          uint32_t padLeft,
          uint32_t padTop,
          int16_t lowerBound,
          int16_t upperBound);

    utils::Optional<command_stream::BlockConfig> GetBlockConfig() override
    {
        return m_BlockConfig;
    }

    virtual DotAttributes GetDotAttributes(DetailLevel) const override;

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
    uint32_t m_UpscaleFactor;
    command_stream::cascading::UpsampleType m_UpsampleType;
    int16_t m_LowerBound;
    int16_t m_UpperBound;
    utils::Optional<TensorShape> m_uninterleavedInputShape;
};

class PleOp : public Op
{
public:
    PleOp();
    PleOp(command_stream::PleOperation op,
          command_stream::BlockConfig blockConfig,
          uint32_t numInputs,
          std::vector<TensorShape> inputStripeShapes,
          TensorShape outputStripeShape,
          DataType dataType,
          bool loadKernel);

    utils::Optional<command_stream::BlockConfig> GetBlockConfig() override
    {
        return m_BlockConfig;
    }

    virtual uint32_t GetNumberOfAgents() const override final;
    virtual DotAttributes GetDotAttributes(DetailLevel) const override;

    command_stream::PleOperation m_Op;
    command_stream::BlockConfig m_BlockConfig;
    uint32_t m_NumInputs;
    std::vector<TensorShape> m_InputStripeShapes;
    TensorShape m_OutputStripeShape;
    command_stream::cascading::PleKernelId m_PleKernelId;
    bool m_LoadKernel;
    utils::Optional<uint32_t> m_Offset;
    uint16_t m_Input0Multiplier;
    uint16_t m_Input0Shift;
    uint16_t m_Input1Multiplier;
    uint16_t m_Input1Shift;
};

class EstimateOnlyOp : public Op
{
public:
    EstimateOnlyOp(const std::string& reasonForEstimateOnly);

    std::string m_ReasonForEstimateOnly;
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
    Buffer(Location location, CascadingBufferFormat format, TraversalOrder order);
    Buffer(Location location,
           CascadingBufferFormat format,
           TensorShape tensorShape,
           TensorShape stripeShape,
           TraversalOrder order,
           uint32_t sizeInBytes,
           QuantizationInfo quantInfo);
    virtual ~Buffer()
    {}

    bool IsFullTensor() const
    {
        return m_Location == Location::Dram ||
               (m_Location == Location::Sram && utils::IsFullTensor(m_TensorShape, m_StripeShape));
    }

    Location m_Location;
    DataType m_DataType;
    CascadingBufferFormat m_Format;
    QuantizationInfo m_QuantizationInfo;
    TensorShape m_TensorShape;
    TensorShape m_StripeShape;
    TraversalOrder m_Order;
    /// The size of the entire buffer, in bytes. For DRAM buffers, this would be the size of the entire
    /// tensor, but for SRAM buffers this would be a rolling buffer and likely be smaller than the entire
    /// tensor.
    uint32_t m_SizeInBytes;
    /// Relevant only for SRAM buffers.
    /// The size of a single slot in the buffer, in bytes. This could be derived from m_StripeShape,
    /// m_Format, m_PackedBoundaryThickness etc., but it is useful to store by itself.
    uint32_t m_SlotSizeInBytes;

    /// This value is set by the different parts for DRAM buffers
    utils::Optional<BufferType> m_BufferType;

    /// This value is set by the Combiner for SRAM buffers
    utils::Optional<uint32_t> m_Offset;

    /// This value is set by the NetworkToGraphOfPartsConverter for Input/Output buffers
    utils::Optional<uint32_t> m_OperationId;
    utils::Optional<uint32_t> m_ProducerOutputIndx;

    /// This value should be easily calculable from m_SizeInBytes and m_SlotSizeInBytes,
    /// but is useful to store by itself nonetheless.
    uint32_t m_NumStripes;

    /// Relevant only if this is a weights buffer in Dram.
    std::shared_ptr<EncodedWeights> m_EncodedWeights;

    /// Relevant only for SRAM buffers.
    /// Defines how much boundary data on each side is packed into each stripe in this buffer.
    command_stream::cascading::PackedBoundaryThickness m_PackedBoundaryThickness;

    /// Relevant only for SRAM buffers.
    /// How many times the tensor is loaded into this buffer. Normally this would be 1,
    /// as we stream data in or out once. However, we sometimes need to re-load the same data
    /// from DRAM multiple times for more complicated streaming strategies, in which case
    /// this field can be >1 to indicate this.
    uint32_t m_NumLoads;
};

}    // namespace support_library
}    // namespace ethosn
