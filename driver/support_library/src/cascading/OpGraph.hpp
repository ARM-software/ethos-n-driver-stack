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
    const ConsumersList& GetConsumers(Buffer* buffer) const;
    std::pair<Op*, uint32_t> GetConsumer(Buffer* buffer, uint32_t index) const;
    const BufferList& GetInputs(Op* op) const;
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
    ///
    /// Such sequences can arise as a result of combining multiple plans together
    /// (in particular Reshape, Concat and Split) and lead to worse performance.
    /// By eliminating/simplifying these sequences, the NPU will have less work to do
    /// and so performance will be better.
    void RemoveRedundantCopies();

    /// Optimization step which reduces the amount of packed boundary data for cases where
    /// the DRAM format is now known to not be FCAF_WIDE.
    void ReducePackedBoundaryData();

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

    template <typename TOp>
    TOp* AddOp(std::unique_ptr<TOp> op);

    template <typename TBuffer>
    TBuffer* AddBuffer(std::unique_ptr<TBuffer> buffer);

    // Merge another OpGraph into the current one taking ownership of the other opgraphs ops and buffers
    void MergeOpGraph(OwnedOpGraph& other);

private:
    std::vector<std::unique_ptr<Op>> m_OwnedOps;
    std::vector<std::unique_ptr<Buffer>> m_OwnedBuffers;
};

class Op : public DebuggableObject
{
public:
    Op(const char* defaultTagPrefix);
    virtual ~Op() = default;

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
    MceUpsampleType m_UpsampleType;
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

    virtual DotAttributes GetDotAttributes(DetailLevel) const override;

    command_stream::PleOperation m_Op;
    command_stream::BlockConfig m_BlockConfig;
    uint32_t m_NumInputs;
    std::vector<TensorShape> m_InputStripeShapes;
    TensorShape m_OutputStripeShape;
    command_stream::cascading::PleKernelId m_PleKernelId;
    uint32_t m_BlockMultiplier;
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

class SramBuffer;
class DramBuffer;
class PleInputSramBuffer;

class SramBufferBuilder;
class DramBufferBuilder;
class PleInputSramBufferBuilder;

class Buffer : public DebuggableObject
{
protected:
    Buffer(const char* defaultTagPrefix, Location location);

public:
    virtual ~Buffer()
    {}

    bool IsFullTensor() const;

    const SramBuffer* Sram() const;
    SramBuffer* Sram();

    const DramBuffer* Dram() const;
    DramBuffer* Dram();

    const PleInputSramBuffer* PleInputSram() const;
    PleInputSramBuffer* PleInputSram();

    DotAttributes GetDotAttributes(DetailLevel) const override;

    /// The value of this determines the type of object this is (e.g. DramBuffer).
    const Location m_Location;

    DataType m_DataType;
    CascadingBufferFormat m_Format;
    QuantizationInfo m_QuantizationInfo;
    TensorShape m_TensorShape;

    /// The size of the entire buffer, in bytes. For DRAM buffers, this would be the size of the entire
    /// tensor, but for SRAM buffers this would be a rolling buffer and likely be smaller than the entire
    /// tensor.
    uint32_t m_SizeInBytes;
};

template <class TBuffer, class TBuilder>
class BufferBuilder
{
protected:
    BufferBuilder();

    void ValidateCommon();

    std::unique_ptr<TBuffer> m_Buffer;

public:
    TBuilder& AddDataType(DataType dataType);
    TBuilder& AddFormat(CascadingBufferFormat format);
    TBuilder& AddQuantization(const QuantizationInfo& info);
    TBuilder& AddTensorShape(const TensorShape& shape);
    TBuilder& AddDebugTag(std::string debug);
    TBuilder& AddSizeInBytes(uint32_t size);

    virtual operator std::unique_ptr<TBuffer>() = 0;
};

template <class TBuffer, class TBuilder>
BufferBuilder<TBuffer, TBuilder>::BufferBuilder()
    : m_Buffer(std::make_unique<TBuffer>())
{}

template <class TBuffer, class TBuilder>
TBuilder& BufferBuilder<TBuffer, TBuilder>::AddDataType(DataType dataType)
{
    m_Buffer->m_DataType = dataType;
    return static_cast<TBuilder&>(*this);
}

template <class TBuffer, class TBuilder>
TBuilder& BufferBuilder<TBuffer, TBuilder>::AddFormat(CascadingBufferFormat format)
{
    m_Buffer->m_Format = format;
    return static_cast<TBuilder&>(*this);
}

template <class TBuffer, class TBuilder>
TBuilder& BufferBuilder<TBuffer, TBuilder>::AddQuantization(const QuantizationInfo& info)
{
    m_Buffer->m_QuantizationInfo = info;
    return static_cast<TBuilder&>(*this);
}

template <class TBuffer, class TBuilder>
TBuilder& BufferBuilder<TBuffer, TBuilder>::AddTensorShape(const TensorShape& shape)
{
    m_Buffer->m_TensorShape = shape;
    return static_cast<TBuilder&>(*this);
}

template <class TBuffer, class TBuilder>
TBuilder& BufferBuilder<TBuffer, TBuilder>::AddDebugTag(std::string debug)
{
    m_Buffer->m_DebugTag = std::move(debug);
    return static_cast<TBuilder&>(*this);
}

template <class TBuffer, class TBuilder>
TBuilder& BufferBuilder<TBuffer, TBuilder>::AddSizeInBytes(uint32_t size)
{
    m_Buffer->m_SizeInBytes = size;
    return static_cast<TBuilder&>(*this);
}

template <class TBuffer, class TBuilder>
void BufferBuilder<TBuffer, TBuilder>::ValidateCommon()
{
    assert((Location::Dram <= m_Buffer->m_Location) && (m_Buffer->m_Location <= Location::VirtualSram));
    assert((DataType::UINT8_QUANTIZED <= m_Buffer->m_DataType) && (m_Buffer->m_DataType <= DataType::INT32_QUANTIZED));
    assert((CascadingBufferFormat::NHWC <= m_Buffer->m_Format) &&
           (m_Buffer->m_Format <= CascadingBufferFormat::FCAF_WIDE));
    assert((m_Buffer->m_TensorShape != TensorShape()));
    // m_QuantizationInfo is initialized to a valid value by default
    // m_SizeInBytes may be zero, asserts added in derived classes where it must be non-zero
}

class DramBuffer : public Buffer
{
private:
    DramBuffer();

public:
    static DramBufferBuilder Build();

    DotAttributes GetDotAttributes(DetailLevel) const override;

    utils::Optional<BufferType> m_BufferType;

    /// This value is set by the NetworkToGraphOfPartsConverter for Input/Output buffers
    utils::Optional<uint32_t> m_OperationId;
    utils::Optional<uint32_t> m_ProducerOutputIndx;

    /// Relevant only if this is a weights buffer.
    std::shared_ptr<EncodedWeights> m_EncodedWeights;

    /// Relevant only if this is a constant buffer.
    std::shared_ptr<std::vector<uint8_t>> m_ConstantData;

    friend std::unique_ptr<DramBuffer> std::make_unique<DramBuffer>();
};

class DramBufferBuilder : public BufferBuilder<DramBuffer, DramBufferBuilder>
{
public:
    DramBufferBuilder();

    DramBufferBuilder& AddBufferType(const utils::Optional<BufferType>& type);
    DramBufferBuilder& AddOperationId(const utils::Optional<uint32_t>& id);
    DramBufferBuilder& AddProducerOutputIndex(const utils::Optional<uint32_t>& index);
    DramBufferBuilder& AddEncodedWeights(std::shared_ptr<EncodedWeights> weights);
    DramBufferBuilder& AddConstantData(std::shared_ptr<std::vector<uint8_t>> constant);

    operator std::unique_ptr<DramBuffer>();
};

class SramBuffer : public Buffer
{
private:
    SramBuffer();

public:
    static SramBufferBuilder Build();

    DotAttributes GetDotAttributes(DetailLevel) const override;

    TensorShape m_StripeShape;

    TraversalOrder m_Order;

    /// The size of a single slot in the buffer, in bytes. This could be derived from m_StripeShape,
    /// m_Format, m_PackedBoundaryThickness etc., but it is useful to store by itself.
    uint32_t m_SlotSizeInBytes;

    /// This value is set by the Combiner.
    utils::Optional<uint32_t> m_Offset;

    /// This value should be easily calculable from m_SizeInBytes and m_SlotSizeInBytes,
    /// but is useful to store by itself nonetheless.
    uint32_t m_NumStripes;

    /// Defines how much boundary data on each side is packed into each stripe in this buffer.
    PackedBoundaryThickness m_PackedBoundaryThickness;

    /// How many times the tensor is loaded into this buffer. Normally this would be 1,
    /// as we stream data in or out once. However, we sometimes need to re-load the same data
    /// from DRAM multiple times for more complicated streaming strategies, in which case
    /// this field can be >1 to indicate this.
    uint32_t m_NumLoads;

    /// If set, this SRAM buffer has not been allocated enough space to be used as the DMA
    /// destination for an FCAF_WIDE DRAM buffer. Therefore using FCAF_WIDE would result in
    /// a buffer overflow.
    /// Note that just because this value is false, does not mean that FCAF_WIDE is compatible,
    /// as there are other compatibility criteria too.
    /// This value could be calculated based on other properties, but it's a bit complicated
    /// so we prefer to calculate it once when we first work out the buffer size.
    bool m_ForbidFcafWide;

    friend std::unique_ptr<SramBuffer> std::make_unique<SramBuffer>();
};

namespace impl
{
struct TileSizeCalculation;
}

class SramBufferBuilder : public BufferBuilder<SramBuffer, SramBufferBuilder>
{
public:
    SramBufferBuilder();

    SramBufferBuilder& AddStripeShape(const TensorShape& shape);
    SramBufferBuilder& AddTraversalOrder(TraversalOrder order);
    SramBufferBuilder& AddPackedBoundaryThickness(const PackedBoundaryThickness& boundary);
    SramBufferBuilder& AddNumLoads(uint32_t loads);
    SramBufferBuilder& ForbidFcafWide(bool forbid);
    SramBufferBuilder& AddSlotSize(uint32_t slotSize);
    SramBufferBuilder& AddNumStripes(uint32_t numStripes);

    // TileSizeCalculation covers forbid fcaf, buffer size, and slot size
    SramBufferBuilder& AddFromTileSize(const impl::TileSizeCalculation& tile);

    operator std::unique_ptr<SramBuffer>();
};

class PleInputSramBuffer : public Buffer
{
private:
    PleInputSramBuffer();

public:
    static PleInputSramBufferBuilder Build();

    TensorShape m_StripeShape;

    /// Doesn't really mean anything for this type of buffer, but we store it to preserve
    /// this value along a cascade to places where it does matter.
    uint32_t m_NumStripes;

    DotAttributes GetDotAttributes(DetailLevel) const override;

    friend std::unique_ptr<PleInputSramBuffer> std::make_unique<PleInputSramBuffer>();
};

class PleInputSramBufferBuilder : public BufferBuilder<PleInputSramBuffer, PleInputSramBufferBuilder>
{
public:
    PleInputSramBufferBuilder();

    PleInputSramBufferBuilder& AddStripeShape(const TensorShape& shape);
    PleInputSramBufferBuilder& AddNumStripes(uint32_t numStripes);

    operator std::unique_ptr<PleInputSramBuffer>();
};

}    // namespace support_library
}    // namespace ethosn
