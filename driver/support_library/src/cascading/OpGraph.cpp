//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "OpGraph.hpp"
#include "PleKernelDatabase.hpp"
#include "StripeHelper.hpp"

#include <unordered_set>

using namespace std;
using namespace ethosn::command_stream;

namespace ethosn
{
namespace support_library
{

bool IsCompressed(CascadingBufferFormat format)
{
    return format == CascadingBufferFormat::FCAF_DEEP || format == CascadingBufferFormat::FCAF_WIDE;
}

void OpGraph::MergeOpGraph(const OpGraph& other)
{
    m_Ops.insert(std::end(m_Ops), std::begin(other.m_Ops), std::end(other.m_Ops));
    m_Buffers.insert(std::end(m_Buffers), std::begin(other.m_Buffers), std::end(other.m_Buffers));
    m_BufferProducers.insert(std::begin(other.m_BufferProducers), std::end(other.m_BufferProducers));
    m_BufferConsumers.insert(std::begin(other.m_BufferConsumers), std::end(other.m_BufferConsumers));
    m_OpOutputs.insert(std::begin(other.m_OpOutputs), std::end(other.m_OpOutputs));
    m_OpInputs.insert(std::begin(other.m_OpInputs), std::end(other.m_OpInputs));
}

const OpGraph::OpList& OpGraph::GetOps() const
{
    return m_Ops;
}

Op* OpGraph::GetOp(uint32_t index) const
{
    assert(index < m_Ops.size());
    return m_Ops.at(index);
}

const OpGraph::BufferList& OpGraph::GetBuffers() const
{
    return m_Buffers;
}

bool OpGraph::Contains(Op* op) const
{
    return std::find(m_Ops.begin(), m_Ops.end(), op) != m_Ops.end();
}

bool OpGraph::Contains(Buffer* buffer) const
{
    return std::find(m_Buffers.begin(), m_Buffers.end(), buffer) != m_Buffers.end();
}

ethosn::support_library::Op* OpGraph::GetSingleProducer(Buffer* buffer) const
{
    auto it = m_BufferProducers.find(buffer);
    if (it == m_BufferProducers.end())
    {
        return nullptr;
    }
    else
    {
        if (it->second.size() == 0)
        {
            return nullptr;
        }
        else
        {
            if (it->second.size() > 1)
            {
                throw std::runtime_error(
                    "This buffer has multiple producers, can't use GetSingleProducer. Use GetProducers instead");
            }
            return it->second[0];
        }
    }
}

OpGraph::OpList OpGraph::GetProducers(Buffer* buffer) const
{
    auto it = m_BufferProducers.find(buffer);
    if (it == m_BufferProducers.end())
    {
        return {};
    }
    else
    {
        return it->second;
    }
}

const OpGraph::ConsumersList& OpGraph::GetConsumers(Buffer* buffer) const
{
    static OpGraph::ConsumersList empty;    // So that we can return a ref
    auto it = m_BufferConsumers.find(buffer);
    return it != m_BufferConsumers.end() ? it->second : empty;
}

std::pair<Op*, uint32_t> OpGraph::GetConsumer(Buffer* buffer, uint32_t index) const
{
    auto it = m_BufferConsumers.find(buffer);

    if (it != m_BufferConsumers.end())
    {
        assert(index < it->second.size());
        return it->second.at(index);
    }
    else
    {
        return std::make_pair(nullptr, 0);
    }
}

const OpGraph::BufferList& OpGraph::GetInputs(Op* op) const
{
    static OpGraph::BufferList empty;    // So that we can return a ref
    auto it = m_OpInputs.find(op);
    return it != m_OpInputs.end() ? it->second : empty;
}

Buffer* OpGraph::GetOutput(Op* op) const
{
    auto it = m_OpOutputs.find(op);
    return it != m_OpOutputs.end() ? it->second : nullptr;
}

void OpGraph::AddOp(Op* op)
{
    if (std::find(m_Ops.begin(), m_Ops.end(), op) != m_Ops.end())
    {
        throw std::runtime_error("Cannot add the same Op twice");
    }
    m_Ops.push_back(op);
}

void OpGraph::AddBuffer(Buffer* buffer)
{
    if (std::find(m_Buffers.begin(), m_Buffers.end(), buffer) != m_Buffers.end())
    {
        throw std::runtime_error("Cannot add the same Buffer twice");
    }
    m_Buffers.push_back(buffer);
}

void OpGraph::SetProducer(Buffer* buffer, Op* producerOp)
{
    if (!Contains(buffer))
    {
        throw std::runtime_error("`buffer` is not part of this graph (or is nullptr)");
    }
    if (!Contains(producerOp))
    {
        throw std::runtime_error("`producerOp` is not part of this graph (or is nullptr)");
    }
    auto it = m_BufferProducers.find(buffer);
    if (it != m_BufferProducers.end() && it->second.size() > 0)
    {
        throw std::runtime_error("Buffer is already produced by an Op. It must be disconnected first.");
    }
    m_BufferProducers[buffer] = { producerOp };
    m_OpOutputs[producerOp]   = buffer;
}

void OpGraph::AddProducer(Buffer* buffer, Op* producerOp)
{
    if (!Contains(buffer))
    {
        throw std::runtime_error("`buffer` is not part of this graph (or is nullptr)");
    }
    if (!Contains(producerOp))
    {
        throw std::runtime_error("`producerOp` is not part of this graph (or is nullptr)");
    }
    OpList& producerList = m_BufferProducers[buffer];
    if (utils::Find(producerList, producerOp).first)
    {
        throw std::runtime_error("`producerOp` is already a producer");
    }
    producerList.push_back(producerOp);
    m_OpOutputs[producerOp] = buffer;
}

void OpGraph::RemoveProducer(Buffer* buffer, Op* producerOp)
{
    if (!Contains(buffer))
    {
        throw std::runtime_error("`buffer` is not part of this graph (or is nullptr)");
    }
    if (!Contains(producerOp))
    {
        throw std::runtime_error("`producerOp` is not part of this graph (or is nullptr)");
    }

    auto oldProducerIt = m_BufferProducers.find(buffer);
    if (oldProducerIt == m_BufferProducers.end())
    {
        throw std::runtime_error("`producerOp` is not a producer of `buffer`");
    }
    OpList& producers             = oldProducerIt->second;
    std::pair<bool, size_t> found = utils::FindIndex(producers, producerOp);
    if (!found.first)
    {
        throw std::runtime_error("`producerOp` is not a producer of `buffer`");
    }
    producers.erase(producers.begin() + found.second);
    m_OpOutputs.erase(producerOp);
}

void OpGraph::ClearProducers(Buffer* buffer)
{
    if (!Contains(buffer))
    {
        throw std::runtime_error("`buffer` is not part of this graph (or is nullptr)");
    }
    auto oldProducerIt = m_BufferProducers.find(buffer);
    if (oldProducerIt != m_BufferProducers.end())
    {
        for (Op* producer : oldProducerIt->second)
        {
            m_OpOutputs.erase(producer);
        }
    }
    m_BufferProducers.erase(buffer);
}

void OpGraph::AddConsumer(Buffer* buffer, Op* consumerOp, uint32_t opInputIdx)
{
    if (!Contains(buffer))
    {
        throw std::runtime_error("`buffer` is not part of this graph (or is nullptr)");
    }
    if (!Contains(consumerOp))
    {
        throw std::runtime_error("`consumerOp` is not part of this graph (or is nullptr)");
    }
    auto it = m_OpInputs.find(consumerOp);
    if (it != m_OpInputs.end() && opInputIdx < it->second.size() && it->second[opInputIdx] != nullptr)
    {
        throw std::runtime_error(
            "`consumerOp` is already consuming a buffer at `opInputIdx`. It must be disconnected first.");
    }
    m_BufferConsumers[buffer].push_back({ consumerOp, opInputIdx });
    auto& inputs = m_OpInputs[consumerOp];
    if (opInputIdx < inputs.size())
    {
        inputs[opInputIdx] = buffer;
    }
    else if (opInputIdx == inputs.size())
    {
        inputs.push_back(buffer);
    }
    else
    {
        // Prevent leaving 'dangling' inputs - they must be connected properly first.
        // This means other code can be sure that input buffers are not set to null and so don't need to check.
        throw std::runtime_error("Cannot connect to this input index without connecting earlier inputs first.");
    }
}

void OpGraph::RemoveConsumer(Buffer* buffer, Op* consumerOp, uint32_t opInputIdx)
{
    if (!Contains(buffer))
    {
        throw std::runtime_error("`buffer` is not part of this graph (or is nullptr)");
    }
    if (!Contains(consumerOp))
    {
        throw std::runtime_error("`consumerOp` is not part of this graph (or is nullptr)");
    }

    auto consumerIt = m_BufferConsumers.find(buffer);
    if (consumerIt == m_BufferConsumers.end())
    {
        throw std::runtime_error("`consumerOp` is not a consumer of `buffer`");
    }

    ConsumersList& consumers      = consumerIt->second;
    std::pair<bool, size_t> found = utils::FindIndex(consumers, std::pair<Op*, uint32_t>{ consumerOp, opInputIdx });
    if (!found.first)
    {
        throw std::runtime_error("`consumerOp` is not a consumer of `buffer`");
    }
    consumers.erase(consumers.begin() + found.second);

    auto& inputs = m_OpInputs[consumerOp];
    assert(opInputIdx < inputs.size());
    if (opInputIdx == inputs.size() - 1)
    {
        inputs.pop_back();
    }
    else
    {
        // Prevent disconnecting anything other than the non-last input, as this would shuffle
        // the other inputs up and cause unintentional semantic changes to the graph
        throw std::runtime_error("Cannot disconnect from this input index without disconnecting later inputs first.");
    }
}

void OpGraph::RemoveAndPrune(Op* op)
{
    // Input side - disconnect from input buffers, and prune the input buffers
    // if this was their last consumer
    {
        // Take a copy of the input buffers array, as we will be disconnecting these as we loop.
        std::vector<Buffer*> inputs = GetInputs(op);

        // Loop in reverse order as inputs can only be disconnected in this order
        for (int inputIdx = static_cast<int>(inputs.size() - 1); inputIdx >= 0; --inputIdx)
        {
            RemoveConsumer(inputs[inputIdx], op, inputIdx);
        }

        for (Buffer* b : inputs)
        {
            if (GetConsumers(b).size() == 0)
            {
                RemoveAndPrune(b);
            }
        }
    }

    // Output side - disconnect from any output buffer, and prune the output buffers
    // if this was their last producer
    {
        Buffer* b = GetOutput(op);
        if (b != nullptr)
        {
            RemoveProducer(b, op);

            if (GetProducers(b).size() == 0)
            {
                RemoveAndPrune(b);
            }
        }
    }

    // Finally, remove the op itself
    std::pair<bool, size_t> found = utils::FindIndex(m_Ops, op);
    if (!found.first)
    {
        throw std::runtime_error("`op` is not part of this graph");
    }
    m_Ops.erase(m_Ops.begin() + found.second);
}

void OpGraph::RemoveAndPrune(Buffer* buffer)
{
    // Input side - disconnect from producers and prune them too
    {
        // Take a copy of the producers array, as we will be disconnecting these as we loop.
        std::vector<Op*> producers = GetProducers(buffer);
        for (Op* p : producers)
        {
            RemoveProducer(buffer, p);
        }

        for (Op* p : producers)
        {
            RemoveAndPrune(p);
        }
    }

    // Output side - disconnect from consumers, and prune the consumers too if this was their last
    // input buffer
    {
        // Take a copy of the consumers array, as we will be disconnecting these as we loop.
        ConsumersList consumers = GetConsumers(buffer);
        for (std::pair<Op*, uint32_t> c : consumers)
        {
            RemoveConsumer(buffer, c.first, c.second);
        }

        for (std::pair<Op*, uint32_t> c : consumers)
        {
            if (GetInputs(c.first).size() == 0)
            {
                RemoveAndPrune(c.first);
            }
        }
    }

    // Finally, remove the buffer itself
    std::pair<bool, size_t> found = utils::FindIndex(m_Buffers, buffer);
    if (!found.first)
    {
        throw std::runtime_error("`buffer` is not part of this graph");
    }
    m_Buffers.erase(m_Buffers.begin() + found.second);
}

template <typename TOp>
TOp* OwnedOpGraph::AddOp(std::unique_ptr<TOp> op)
{
    // Call base implementation first in case it errors, in which case we don't want to track this Op.
    TOp* raw = op.get();
    OpGraph::AddOp(raw);
    m_Ops.emplace_back(std::move(op));
    return raw;
}

// Explicit instantiations
template Op* OwnedOpGraph::AddOp<Op>(std::unique_ptr<Op> op);
template DummyOp* OwnedOpGraph::AddOp<DummyOp>(std::unique_ptr<DummyOp> op);
template DmaOp* OwnedOpGraph::AddOp<DmaOp>(std::unique_ptr<DmaOp> op);
template MceOp* OwnedOpGraph::AddOp<MceOp>(std::unique_ptr<MceOp> op);
template PleOp* OwnedOpGraph::AddOp<PleOp>(std::unique_ptr<PleOp> op);
template EstimateOnlyOp* OwnedOpGraph::AddOp<EstimateOnlyOp>(std::unique_ptr<EstimateOnlyOp> op);

template <typename TBuffer>
TBuffer* OwnedOpGraph::AddBuffer(std::unique_ptr<TBuffer> buffer)
{
    // Call base implementation first in case it errors, in which case we don't want to track this Op.
    TBuffer* raw = buffer.get();
    OpGraph::AddBuffer(raw);
    m_Buffers.emplace_back(std::move(buffer));
    return raw;
}

// Explicit instantiations
template Buffer* OwnedOpGraph::AddBuffer<Buffer>(std::unique_ptr<Buffer> op);
template DramBuffer* OwnedOpGraph::AddBuffer<DramBuffer>(std::unique_ptr<DramBuffer> op);
template SramBuffer* OwnedOpGraph::AddBuffer<SramBuffer>(std::unique_ptr<SramBuffer> op);
template PleInputSramBuffer* OwnedOpGraph::AddBuffer<PleInputSramBuffer>(std::unique_ptr<PleInputSramBuffer> op);

void OwnedOpGraph::MergeOpGraph(OwnedOpGraph& other)
{
    for (auto&& op : other.m_Ops)
    {
        AddOp(std::move(op));
    }
    for (auto&& buf : other.m_Buffers)
    {
        AddBuffer(std::move(buf));
    }
    m_BufferProducers.insert(std::begin(other.m_BufferProducers), std::end(other.m_BufferProducers));
    m_BufferConsumers.insert(std::begin(other.m_BufferConsumers), std::end(other.m_BufferConsumers));
    m_OpOutputs.insert(std::begin(other.m_OpOutputs), std::end(other.m_OpOutputs));
    m_OpInputs.insert(std::begin(other.m_OpInputs), std::end(other.m_OpInputs));
}

Op::Op(const char* defaultTagPrefix)
    : DebuggableObject(defaultTagPrefix)
    , m_OperationIds()
{}

DotAttributes Op::GetDotAttributes(DetailLevel) const
{
    return DotAttributes();
}

DmaOp::DmaOp(CascadingBufferFormat transferFormat)
    : Op("DmaOp")
    , m_TransferFormat(transferFormat)
    , m_Offset({ 0, 0, 0, 0 })
{}

DmaOp::DmaOp(const char* debugType, CascadingBufferFormat transferFormat)
    : Op(debugType)
    , m_TransferFormat(transferFormat)
    , m_Offset({ 0, 0, 0, 0 })
{}

DotAttributes DmaOp::GetDotAttributes(DetailLevel detail) const
{
    DotAttributes result;
    if (detail == DetailLevel::High)
    {
        result.m_Label = "DmaOp\n";
        result.m_Label += "Operation Ids = " + ArrayToString(m_OperationIds) + "\n";
        result.m_Label += "Transfer Format = " + ToString(m_TransferFormat) + "\n";
        result.m_Label += "Offset = " + ToString(m_Offset) + "\n";
    }

    result.m_Color = std::string("darkgoldenrod");
    return result;
}

MceOp::MceOp()
    : Op("MceOp")
    , m_Op(MceOperation::CONVOLUTION)
    , m_Algo(CompilerMceAlgorithm::Direct)
    , m_BlockConfig{ 0u, 0u }
    , m_InputStripeShape{ 0, 0, 0, 0 }
    , m_OutputStripeShape{ 0, 0, 0, 0 }
    , m_WeightsStripeShape{ 0, 0, 0, 0 }
    , m_Order(TraversalOrder::Xyz)
    , m_Stride()
    , m_PadLeft(0)
    , m_PadTop(0)
    , m_UpscaleFactor(1)
    , m_UpsampleType(command_stream::cascading::UpsampleType::OFF)
    , m_LowerBound(0)
    , m_UpperBound(255)
{}

MceOp::MceOp(MceOperation op,
             CompilerMceAlgorithm algo,
             BlockConfig blockConfig,
             TensorShape inputStripeShape,
             TensorShape outputStripeShape,
             TensorShape weightsStripeShape,
             TraversalOrder order,
             Stride stride,
             uint32_t padLeft,
             uint32_t padTop,
             int16_t lowerBound,
             int16_t upperBound)
    : Op("MceOp")
    , m_Op(op)
    , m_Algo(algo)
    , m_BlockConfig(blockConfig)
    , m_InputStripeShape(inputStripeShape)
    , m_OutputStripeShape(outputStripeShape)
    , m_WeightsStripeShape(weightsStripeShape)
    , m_Order(order)
    , m_Stride(stride)
    , m_PadLeft(padLeft)
    , m_PadTop(padTop)
    , m_UpscaleFactor(1)
    , m_UpsampleType(command_stream::cascading::UpsampleType::OFF)
    , m_LowerBound(lowerBound)
    , m_UpperBound(upperBound)
{}

DotAttributes MceOp::GetDotAttributes(DetailLevel detail) const
{
    DotAttributes result;
    if (detail == DetailLevel::High)
    {
        result.m_Label = "MceOp\n";
        result.m_Label += "Op = " + ToString(m_Op) + "\n";
        result.m_Label += "Algo = " + ToString(m_Algo) + "\n";
        result.m_Label += "Block Config = " + ToString(m_BlockConfig) + "\n";
        result.m_Label += "Input Stripe Shape = " + ToString(m_InputStripeShape) + "\n";
        result.m_Label += "Output Stripe Shape = " + ToString(m_OutputStripeShape) + "\n";
        result.m_Label += "Weights Stripe Shape = " + ToString(m_WeightsStripeShape) + "\n";
        result.m_Label += "Order = " + ToString(m_Order) + "\n";
        result.m_Label += "Stride = " + ToString(m_Stride) + "\n";
        result.m_Label += "Pad L/T = " + to_string(m_PadLeft) + ", " + to_string(m_PadTop) + "\n";
        result.m_Label += "UpscaleFactor = " + ToString(m_UpscaleFactor) + "\n";
        result.m_Label += "UpsampleType = " + ToString(m_UpsampleType) + "\n";
        result.m_Label += "Lower/Upper Bound = " + to_string(m_LowerBound) + ", " + to_string(m_UpperBound) + "\n";
        result.m_Label += "Operation Ids = " + ArrayToString(m_OperationIds) + "\n";
    }
    return result;
}

PleOp::PleOp()
    : Op("PleOp")
    , m_Op(PleOperation::FAULT)
    , m_BlockConfig{ 0u, 0u }
    , m_NumInputs(0)
    , m_InputStripeShapes{}
    , m_OutputStripeShape{ 0, 0, 0, 0 }
    , m_PleKernelId{ command_stream::cascading::PleKernelId::NOT_FOUND }
    , m_LoadKernel{ true }
    , m_Input0Multiplier(0)
    , m_Input0Shift(0)
    , m_Input1Multiplier(0)
    , m_Input1Shift(0)
{}

PleOp::PleOp(PleOperation op,
             BlockConfig blockConfig,
             uint32_t numInputs,
             std::vector<TensorShape> inputStripeShapes,
             TensorShape outputStripeShape,
             DataType dataType,
             bool loadKernel)
    : Op("PleOp")
    , m_Op(op)
    , m_BlockConfig(blockConfig)
    , m_NumInputs(numInputs)
    , m_InputStripeShapes(inputStripeShapes)
    , m_OutputStripeShape(outputStripeShape)
    , m_LoadKernel(loadKernel)
    , m_Input0Multiplier(0)
    , m_Input0Shift(0)
    , m_Input1Multiplier(0)
    , m_Input1Shift(0)
{
    m_PleKernelId = plelib::FindPleKernelIdFromDatabase(blockConfig, (inputStripeShapes.at(0))[2],
                                                        utils::GetCommandDataType(dataType), op);
}

uint32_t PleOp::GetNumberOfAgents() const
{
    return m_LoadKernel ? 2 : 1;
}

DotAttributes PleOp::GetDotAttributes(DetailLevel detail) const
{
    DotAttributes result;
    if (detail == DetailLevel::High)
    {
        result.m_Label = "PleOp\n";
        result.m_Label += "Op = " + ToString(m_Op) + "\n";
        result.m_Label += "Block Config = " + ToString(m_BlockConfig) + "\n";
        result.m_Label += "Num Inputs = " + to_string(m_NumInputs) + "\n";
        result.m_Label += "Input Stripe Shapes = " + ArrayToString(m_InputStripeShapes) + "\n";
        result.m_Label += "Output Stripe Shape = " + ToString(m_OutputStripeShape) + "\n";
        result.m_Label += "Ple kernel Id = " + ToString(m_PleKernelId) + "\n";
        result.m_Label += "Kernel Load = " + ToString(m_LoadKernel) + "\n";
        if (m_Offset.has_value())
        {
            result.m_Label += "Offset = " + ToString(m_Offset.value()) + " (" + ToStringHex(m_Offset.value()) + ")\n";
        }
        result.m_Label += "Operation Ids = " + ArrayToString(m_OperationIds) + "\n";
        result.m_Label += "Input0Multiplier = " + ToString(m_Input0Multiplier) + "\n";
        result.m_Label += "Input0Shift = " + ToString(m_Input0Shift) + "\n";
        result.m_Label += "Input1Multiplier = " + ToString(m_Input1Multiplier) + "\n";
        result.m_Label += "Input1Shift = " + ToString(m_Input1Shift) + "\n";
    }
    return result;
}

EstimateOnlyOp::EstimateOnlyOp(const std::string& reasonForEstimateOnly)
    : Op("EstimateOnlyOp")
    , m_ReasonForEstimateOnly(reasonForEstimateOnly)
{}

DummyOp::DummyOp()
    : Op("DummyOp")
{}

Buffer::Buffer(const char* defaultTagPrefix, Location location)
    : DebuggableObject(defaultTagPrefix)
    , m_Location(location)
    , m_DataType(DataType::UINT8_QUANTIZED)
    , m_Format(CascadingBufferFormat::NHWCB)
    , m_QuantizationInfo(QuantizationInfo())
    , m_TensorShape({ 0, 0, 0, 0 })
    , m_SizeInBytes(0)
{}

DotAttributes Buffer::GetDotAttributes(DetailLevel detail) const
{
    DotAttributes result;
    if (detail == DetailLevel::High)
    {
        result.m_Label += "Location = " + ToString(m_Location) + "\n";
        result.m_Label += "Format = " + ToString(m_Format) + "\n";
        result.m_Label += "Data Type = " + ToString(m_DataType) + "\n";
        result.m_Label += "Quant. Info = " + ToString(m_QuantizationInfo) + "\n";
        result.m_Label += "Tensor shape = " + ToString(m_TensorShape) + "\n";
        result.m_Label += "Size in bytes = " + ToString(m_SizeInBytes) + " (" + ToStringHex(m_SizeInBytes) + ")\n";
    }
    return result;
}

bool Buffer::IsFullTensor() const
{
    return m_Location == Location::Dram ||
           (m_Location == Location::Sram && utils::IsFullTensor(m_TensorShape, Sram()->m_StripeShape));
}

const SramBuffer* Buffer::Sram() const
{
    assert(m_Location == Location::Sram);
    return static_cast<const SramBuffer*>(this);
}

SramBuffer* Buffer::Sram()
{
    assert(m_Location == Location::Sram);
    return static_cast<SramBuffer*>(this);
}

const DramBuffer* Buffer::Dram() const
{
    assert(m_Location == Location::Dram);
    return static_cast<const DramBuffer*>(this);
}

DramBuffer* Buffer::Dram()
{
    assert(m_Location == Location::Dram);
    return static_cast<DramBuffer*>(this);
}

const PleInputSramBuffer* Buffer::PleInputSram() const
{
    assert(m_Location == Location::PleInputSram);
    return static_cast<const PleInputSramBuffer*>(this);
}

PleInputSramBuffer* Buffer::PleInputSram()
{
    assert(m_Location == Location::PleInputSram);
    return static_cast<PleInputSramBuffer*>(this);
}

SramBuffer::SramBuffer()
    : Buffer("SramBuffer", Location::Sram)
    , m_StripeShape({ 0, 0, 0, 0 })
    , m_Order(TraversalOrder::Xyz)
    , m_SlotSizeInBytes(0)
    , m_NumStripes(0)
    , m_PackedBoundaryThickness({ 0, 0, 0, 0 })
    , m_NumLoads(1)
    , m_ForbidFcafWide(false)
{}

DotAttributes SramBuffer::GetDotAttributes(DetailLevel detail) const
{
    DotAttributes result = Buffer::GetDotAttributes(detail);
    if (detail == DetailLevel::High)
    {
        result.m_Label += "Stripe shape = " + ToString(m_StripeShape) + "\n";
        result.m_Label += "Order = " + ToString(m_Order) + "\n";
        result.m_Label +=
            "Slot size in bytes = " + ToString(m_SlotSizeInBytes) + " (" + ToStringHex(m_SlotSizeInBytes) + ")\n";
        if (m_Offset.has_value())
        {
            result.m_Label += "Offset = " + ToString(m_Offset.value()) + " (" + ToStringHex(m_Offset.value()) + ")\n";
        }
        result.m_Label += "Num. Stripes = " + ToString(m_NumStripes) + "\n";
        result.m_Label += "Packed boundary thickness = " + ToString(m_PackedBoundaryThickness) + "\n";
        result.m_Label += "Num loads = " + ToString(m_NumLoads) + "\n";
        if (m_ForbidFcafWide)
        {
            result.m_Label += "Forbid FCAF_WIDE\n";
        }
    }
    return result;
}

DramBuffer::DramBuffer()
    : Buffer("DramBuffer", Location::Dram)
{}

DotAttributes DramBuffer::GetDotAttributes(DetailLevel detail) const
{
    DotAttributes result = Buffer::GetDotAttributes(detail);
    if (detail == DetailLevel::High)
    {
        if (m_EncodedWeights)
        {
            result.m_Label +=
                "Encoded weights = { " + ToString(static_cast<uint32_t>(m_EncodedWeights->m_Data.size())) +
                " bytes, max size = " + ToString(m_EncodedWeights->m_MaxSize) +
                ", num. metadata = " + ToString(static_cast<uint32_t>(m_EncodedWeights->m_Metadata.size())) +
                ", is wide filter = " + ToString(m_EncodedWeights->m_IsWideFilter) + " }\n";
        }
        if (m_ConstantData)
        {
            result.m_Label +=
                "Constant data = [ " + ToString(static_cast<uint32_t>(m_ConstantData->size())) + " bytes ]\n";
        }
        result.m_Label += "Type = " + (m_BufferType.has_value() ? ToString(m_BufferType.value()) : "None") + "\n";
        if (m_OperationId.has_value())
        {
            result.m_Label += "Operation ID = " + ToString(m_OperationId.value()) + "\n";
        }
        if (m_ProducerOutputIndx.has_value())
        {
            result.m_Label += "Producer Output Index = " + ToString(m_ProducerOutputIndx.value()) + "\n";
        }
    }
    return result;
}

PleInputSramBuffer::PleInputSramBuffer()
    : Buffer("PleInputSramBuffer", Location::PleInputSram)
    , m_StripeShape({ 0, 0, 0, 0 })
    , m_NumStripes(0)
{}

DotAttributes PleInputSramBuffer::GetDotAttributes(DetailLevel detail) const
{
    DotAttributes result = Buffer::GetDotAttributes(detail);
    if (detail == DetailLevel::High)
    {
        result.m_Label += "Stripe shape = " + ToString(m_StripeShape) + "\n";
        result.m_Label += "Num. Stripes = " + ToString(m_NumStripes) + "\n";
    }
    return result;
}

namespace remove_redundant_copies_impl
{

/// Gets a list of all the buffers in the given OpGraph, sorted topologically from
/// inputs to outputs.
std::vector<Buffer*> GetSortedBuffers(const OpGraph& g)
{
    // Find all buffers with no consumers, which we assume are the output buffers of the graph.
    std::vector<Buffer*> targets;
    for (Buffer* b : g.GetBuffers())
    {
        if (g.GetConsumers(b).size() == 0)
        {
            targets.push_back(b);
        }
    }

    // Function which gets the buffers which are needed as inputs (via the producing Ops) to the given buffer
    auto getIncomingEdges = [&g](Buffer* b) {
        std::vector<Buffer*> incomingBuffers;
        for (Op* c : g.GetProducers(b))
        {
            std::vector<Buffer*> inputs = g.GetInputs(c);
            incomingBuffers.insert(incomingBuffers.end(), inputs.begin(), inputs.end());
        }
        return incomingBuffers;
    };

    // Use our generic topological sort function
    std::vector<Buffer*> buffersSorted;
    bool result = utils::GraphTopologicalSort<Buffer*>(targets, getIncomingEdges, buffersSorted);
    assert(result);    // It should not be possible to have an OpGraph with cycles
    ETHOSN_UNUSED(result);

    return buffersSorted;
}

/// Describes a chain of Buffers with DmaOps connecting adjacent Buffers, e.g.:
///
///    Buffer1 -> DmaOp1 -> Buffer2 -> DmaOp2 -> Buffer3
///
struct DmaChain
{
    /// All the buffers in the chain, in order from first to last.
    /// This vector is one longer than the `dmas` vector.
    std::vector<Buffer*> buffers;
    /// All the DmaOps in the chain, in order from first to last.
    /// This vector is one shorter than the `buffers` vector.
    /// Element i in this vector is the DMA between buffers i and i+1 in the `buffers` vector.
    std::vector<DmaOp*> dmas;

    /// Sums up the DMA offsets along the whole chain
    /// From the validation done when finding the chain, we know that these offsets are all
    /// from SRAM to DRAM (or DRAM to SRAM), so it's meaningful to sum them all up like this.
    TensorShape GetTotalDmaOffset() const
    {
        TensorShape result = { 0, 0, 0, 0 };
        for (DmaOp* dma : dmas)
        {
            result += dma->m_Offset;
        }
        return result;
    }

    /// Gets all the operation IDs tagged anywhere on the chain.
    std::set<uint32_t> GetOperationIds() const
    {
        std::set<uint32_t> result;
        for (DmaOp* dma : dmas)
        {
            result.insert(dma->m_OperationIds.begin(), dma->m_OperationIds.end());
        }
        return result;
    }
};

/// Checks if a given buffer is valid to be included in a DMA chain.
bool IsBufferValid(const Buffer* b)
{
    return b != nullptr && (b->m_Location == Location::Dram || b->m_Location == Location::Sram);
}

enum Dir
{
    SramToDram,
    DramToSram,
};

/// Stored state about whether a chain includes a reshape and/or any subtensors.
/// This affects whether future DmaOps can be included or not (see IsOpValid).
struct ChainState
{
    bool hasReshape   = false;
    bool hasSubtensor = false;
};

/// Checks if a given Op is valid to be included in a DMA chain.
/// If it's valid, returns an updated copy of `inState`, otherwise returns an empty optional.
/// `allowedSubtensorDir` describes whether subtensors are allowed from SRAM -> DRAM or vice versa.
utils::Optional<ChainState> IsOpValid(const Op* op,
                                      const Buffer& inputBuffer,
                                      const Buffer& outputBuffer,
                                      Dir allowedSubtensorDir,
                                      const ChainState& inState)
{
    if (!IsObjectOfType<DmaOp>(op))
    {
        return {};
    }
    const DmaOp& dma = *static_cast<const DmaOp*>(op);

    // The DMA can't be reinterpreting the data (e.g. for fully connected)
    CascadingBufferFormat dramFormat;
    Dir transferDir;
    if (inputBuffer.m_Location == Location::Dram)
    {
        dramFormat = inputBuffer.m_Format;
        assert(outputBuffer.m_Location == Location::Sram);
        transferDir = Dir::DramToSram;
    }
    else
    {
        assert(inputBuffer.m_Location == Location::Sram);
        assert(outputBuffer.m_Location == Location::Dram);
        dramFormat  = outputBuffer.m_Format;
        transferDir = Dir::SramToDram;
    }
    if (dma.m_TransferFormat != dramFormat)
    {
        return {};
    }

    ChainState outState = inState;

    // Subtensors are only allowed in one 'direction' (taking only part of the input buffer, or placing the input into
    // part of an output buffer). We can't mix these in the same chain because it would make the calculation of the final
    // DMA offset (of the optimised chain) more difficult. Multiple subtensors of the same 'direction' are allowed though,
    // as this is simple to accumulate and allows us to merge multiple concats/splits together.
    bool isSubtensor =
        dma.m_Offset != TensorShape{ 0, 0, 0, 0 } ||
        utils::GetNumElements(inputBuffer.m_TensorShape) != utils::GetNumElements(outputBuffer.m_TensorShape);
    if (isSubtensor)
    {
        outState.hasSubtensor = true;
        if (transferDir != allowedSubtensorDir)
        {
            return {};
        }
    }
    bool isReshape =
        inputBuffer.m_TensorShape != outputBuffer.m_TensorShape &&
        utils::GetNumElements(inputBuffer.m_TensorShape) == utils::GetNumElements(outputBuffer.m_TensorShape);
    if (isReshape)
    {
        outState.hasReshape = true;
    }

    if (outState.hasReshape && outState.hasSubtensor)
    {
        // These don't play nice. If we combine subtensors and reshape, it becomes very difficult (impossible?)
        // to recover this information later, and so we can't tell if a chain optimisation is valid or not.
        // To keep things simpler, we simply stop before the chain includes both a reshape and subtensor,
        // and just optimise the bit we can.
        return {};
    }

    return outState;
}

/// Finds a chain of DMAs starting at the given SRAM buffer and ending in a DRAM buffer,
/// which together describe the operation of taking the entire SRAM buffer and copying it into
/// (possibly a sub-tensor of) the ending DRAM buffer.
///
/// For example, given the following OpGraph, we would find chains with the starting buffer as follows:
///
///     a:  a -> C -> e -> G -> j -> K -> i
///     b:  b -> D -> e -> G -> j -> K -> i
///     f:  f -> H -> i
///     j:  j -> K -> i
///
///  (capital letters are DmaOps, lowercase letters are Buffers)
///
///  a (Sram)  b (Sram)
///     |          |
///     C          D
///      \        /
///        e (Dram)       f (Sram)
///           |              |
///           G              |
///           |              |
///        j (Sram)          |
///           |              |
///           K              H
///             \           /
///                i (Dram)
///
DmaChain ExploreDmaChainStartingFromSram(const OpGraph& graph, Buffer* startingBuffer)
{
    // Start of the chain must be in SRAM
    if (!IsBufferValid(startingBuffer) || startingBuffer->m_Location != Location::Sram)
    {
        return {};
    }

    DmaChain result = { { startingBuffer }, {} };

    // Look "down" the graph to find the end of the chain
    Buffer* buffer = startingBuffer;
    ChainState state;
    while (true)
    {
        OpGraph::ConsumersList consumers = graph.GetConsumers(buffer);
        if (consumers.size() != 1)
        {
            // Branching or end of graph - end the chain.
            // Multiple consumers mean that the data we are following is needed elsewhere too,
            // so we won't be able to simply replace this chain.
            break;
        }

        std::pair<Op*, uint32_t> consumer = consumers[0];

        // Check that the buffer outputted by the consumer is valid to include in the chain
        Buffer* consumerOutput = graph.GetOutput(consumer.first);
        if (!IsBufferValid(consumerOutput))
        {
            // Buffer cannot be in this chain - end the chain here
            break;
        }

        // Check if the consumer is a valid Op to include in the chain
        utils::Optional<ChainState> newStateIfValid =
            IsOpValid(consumer.first, *buffer, *consumerOutput, Dir::SramToDram, state);
        if (!newStateIfValid.has_value())
        {
            // Op cannot be in this chain - end the chain here
            break;
        }

        // We're now happy to extend the chain to include the consumer and its output buffer
        state = newStateIfValid.value();
        result.buffers.push_back(consumerOutput);
        result.dmas.push_back(static_cast<DmaOp*>(consumer.first));

        // Keep walking down the graph
        buffer = consumerOutput;
    }

    // If the last buffer we found was SRAM, then pop this off the end so that we're back with a DRAM
    // at the end.
    if (result.buffers.size() >= 2 && result.buffers.back()->m_Location == Location::Sram)
    {
        result.buffers.pop_back();
        result.dmas.pop_back();
    }

    return result;
}

/// Finds a chain of DMAs ending at the given SRAM buffer and starting from a DRAM buffer,
/// which together describe the operation of taking (a sub-tensor of) the DRAM buffer and
/// copying it into the SRAM buffer.
///
/// For example, given the following OpGraph, we would find chains with the ending buffer as follows:
///
///     a:  m -> L -> i -> K -> j -> G -> e -> C -> a
///     b:  m -> L -> i -> K -> j -> G -> e -> D -> b
///     f:  m -> L -> i -> H -> f
///     j:  m -> L -> i -> K -> j
///
///  (capital letters are DmaOps, lowercase letters are Buffers)
///
///                m (Sram)
///                   |
///                   L
///                   |
///                i (Dram)
///             /           \_
///           K              H
///           |              |
///        j (Sram)          |
///           |              |
///           G              |
///           |              |
///        e (Dram)       f (Sram)
///      /        \_
///     C          D
///     |          |
///  a (Sram)   b (Sram)
///
DmaChain ExploreDmaChainEndingAtSram(const OpGraph& graph, Buffer* endingBuffer)
{
    // End of the chain must be in SRAM
    if (!IsBufferValid(endingBuffer) || endingBuffer->m_Location != Location::Sram)
    {
        return {};
    }

    DmaChain result = { { endingBuffer }, {} };

    // Look back "up" the graph to find the start of the chain.
    Buffer* buffer = endingBuffer;
    ChainState state;
    while (true)
    {
        std::vector<Op*> producers = graph.GetProducers(buffer);
        if (producers.size() != 1)
        {
            // Branching or end of graph - end the chain.
            // Multiple producers means that our data doesn't come from a single place, so we can't
            // simply replace this chain.
            break;
        }

        Op* producer = producers[0];

        // Check that the input buffer of the producer is valid to include in the chain
        std::vector<Buffer*> producerInputs = graph.GetInputs(producer);
        Buffer* producerInput               = producerInputs.size() == 0 ? nullptr : producerInputs[0];
        if (!IsBufferValid(producerInput))
        {
            // Buffer cannot be in a chain
            break;
        }

        // Check if the producer is a valid Op to include in the chain -
        utils::Optional<ChainState> newStateIfValid =
            IsOpValid(producer, *producerInput, *buffer, Dir::DramToSram, state);
        if (!newStateIfValid.has_value())
        {
            // Op cannot be in a chain
            break;
        }

        // We're now happy to extend the chain to include producerInput
        // Put the new buffer and op at start, as we are walking "up"
        state = newStateIfValid.value();
        result.buffers.insert(result.buffers.begin(), producerInput);
        result.dmas.insert(result.dmas.begin(), static_cast<DmaOp*>(producer));

        // Keep walking up the graph
        buffer = producerInput;
    }

    // If the last buffer we found was SRAM, then pop this off the start so that we're back with a DRAM
    // at the start.
    if (result.buffers.size() >= 2 && result.buffers.front()->m_Location == Location::Sram)
    {
        result.buffers.erase(result.buffers.begin());
        result.dmas.erase(result.dmas.begin());
    }

    return result;
}

}    // namespace remove_redundant_copies_impl

void OpGraph::RemoveRedundantCopies()
{
    // This optimisation is implemented in two complementary (but independent) halves because it was
    // too complicated to make a generic optimisation. There are different restrictions for what is valid
    // depending on whether you start or end in SRAM/DRAM, and the two cases implemented below are the
    // only ones that we actually need.

    // This one eliminates chains of copies that start in SRAM and end in DRAM (e.g. Concat)
    RemoveRedundantCopiesSramToDram();
    // This one eliminates chains of copies that start in DRAM and end in SRAM (e.g. Split)
    RemoveRedundantCopiesDramToSram();
}

/// Replaces chains of redundant DmaOps from Sram -> Dram.
///
/// For example:
///
/// (capital letters are DmaOps, lowercase letters are Buffers)
///
///  a (Sram)  b (Sram)                                  a (Sram)  b (Sram)
///     |          |                                        |          |
///     C          D                                        C          D
///      \        /                                          \         |
///        e (Dram)       f (Sram)                            \        |    f (Sram)
///           |              |                 =>              \       |       |
///           G              |                                  \      |       |
///           |              |                                   \     |       |
///        j (Sram)          |                                    \    |       |
///           |              |                                     \   |       |
///           K              H                                      \  |       H
///             \           /                                        \ |       /
///                i (Dram)                                            i (Dram)
///
void OpGraph::RemoveRedundantCopiesSramToDram()
{
    using namespace remove_redundant_copies_impl;

    // Look through the graph for chains consisting of just Buffers and DmaOps, starting in Sram
    // and ending in Dram.
    // Search in topological order from inputs -> outputs, so that we find the longest chains first
    std::vector<Buffer*> buffersSorted = GetSortedBuffers(*this);
    std::vector<DmaChain> chains;
    std::unordered_set<Buffer*> visited;
    for (Buffer* buffer : buffersSorted)
    {
        if (visited.count(buffer) > 0)
        {
            // Don't start a chain partway through another chain, otherwise we will have chains which are
            // subsets of each other and then collapsing one chain will affect the other leading to problems.
            // Note that we *can* have chains which share a tail though,
            // as is the case for example with nested concats where multiple SRAM buffers end up in the same DRAM
            // buffer. It's just the SRAM buffer at the start which can't be shared with another chain.
            continue;
        }
        DmaChain chain = ExploreDmaChainStartingFromSram(*this, buffer);
        if (chain.buffers.size() >= 2)
        {
            chains.push_back(chain);
        }
        visited.insert(chain.buffers.begin(), chain.buffers.end());
    }

    // Check which chains can actually be replaced.
    // These are additional criteria to check compared to what's done in ExploreDmaChainStartingFromSram,
    // which make more sense to be done separately.
    int chainIdx = static_cast<int>(chains.size()) - 1;    // Loop in reverse so we can remove invalid chains as we go
    while (chainIdx >= 0)
    {
        DmaChain& chain        = chains[chainIdx];
        bool chainOk           = false;
        bool restartValidation = false;
        // We might need to shorten the chain to make it valid, so keep trying until it gets too short.
        // Four buffers (Sram -> Dram -> Sram -> Dram) is the minimum length we can optimize.
        while (chain.buffers.size() >= 4)
        {
            SramBuffer* sramBuffer = chain.buffers[0]->Sram();
            DramBuffer* dramBuffer = chain.buffers.back()->Dram();

            // Sum up the DMA offsets along the whole chain, to get the total offset
            TensorShape combinedOffset = chain.GetTotalDmaOffset();

            // We have to be careful not to add an invalid DMA, so check that the stripe shapes etc.
            // are compatible, and if not then try shortening the chain to see if we can replace a
            // sub-section of the chain instead
            if (!impl::IsSramBufferCompatibleWithDramBuffer(*sramBuffer, *dramBuffer, combinedOffset))
            {
                chain.buffers.pop_back();    // Remove DRAM buffer
                chain.buffers.pop_back();    // Remove SRAM buffer
                chain.dmas.pop_back();
                chain.dmas.pop_back();

                // The buffer(s) that we removed may have been part of another chain too, and this could
                // lead to problems if a "concat buffer" now has lost some of its inputs, as that part of the
                // buffer would be unitialised and might overwrite some other valid data later.
                // See unit test "OpGraph RemoveRedundantCopiesSramToDram Concat one branch invalid"
                // To avoid this, when a buffer is removed from one chain, we remove that same buffer from _all_ chains that use it
                for (DmaChain& otherChain : chains)
                {
                    if (&otherChain == &chain)
                    {
                        continue;
                    }

                    if (otherChain.buffers.size() >= 2 && otherChain.buffers.back() == dramBuffer)
                    {
                        otherChain.buffers.pop_back();
                        otherChain.buffers.pop_back();
                        otherChain.dmas.pop_back();
                        otherChain.dmas.pop_back();
                    }
                }

                // We then re-validate all the chains from scratch as this has changed the other chains
                // and some of them may no longer be valid.
                restartValidation = true;

                // Try again with the shorter chain
                continue;
            }

            // This chain is good, move on to the next
            chainOk = true;
            break;
        }

        if (!chainOk)
        {
            chains.erase(chains.begin() + chainIdx);
        }
        // Move to the next chain, or if we need to restart validation, do that
        if (restartValidation)
        {
            chainIdx = static_cast<int>(chains.size() - 1);
        }
        else
        {
            --chainIdx;
        }
    }

    // We're now happy that all the remaining chains are valid to be optimised
    // Replace each chain with a single DMA between the starting SRAM and ending DRAM buffers
    for (DmaChain chain : chains)
    {
        assert(chain.buffers[0]->m_Location == Location::Sram);
        DramBuffer* dramBuffer = chain.buffers.back()->Dram();

        // Sum up the DMA offsets along the whole chain, to collapse into one.
        TensorShape combinedOffset = chain.GetTotalDmaOffset();

        // We can't add a new DmaOp as this isn't an OwnedOpGraph, so repurpose one of the existing ones.
        // We can't repurpose the last Dma, as that might be shared with other chains, so we repurpose the first one,
        // which should never be shared.
        DmaOp* firstDma = chain.dmas.front();
        RemoveProducer(chain.buffers[1], firstDma);

        firstDma->m_TransferFormat = dramBuffer->m_Format;
        firstDma->m_Offset         = combinedOffset;
        firstDma->m_OperationIds   = chain.GetOperationIds();
        AddProducer(dramBuffer, firstDma);

        // Prune from the top - we can't start at the end because that might be shared with other chains.
        if (GetProducers(chain.buffers[1]).size() == 0)
        {
            RemoveAndPrune(chain.buffers[1]);
        }
    }
}

/// Replaces chains of redundant DmaOps from Dram -> Sram.
///
/// For example:
///
/// (capital letters are DmaOps, lowercase letters are Buffers)
///
///                m (Sram)                                           m (Sram)
///                   |                                                  |
///                   L                                                  L
///                   |                                                  |
///                i (Dram)                                           i (Dram)
///             /           \_                                     /  |        \_
///           K              H                                    /   |        H
///           |              |                                   /    |        |
///        j (Sram)          |               =>                 /     |        |
///           |              |                                 /      |        |
///           G              |                                /       |        |
///           |              |                               /        |        |
///        e (Dram)       f (Sram)                          /         |     f (Sram)
///      /        \_                                       /          |
///     C          D                                       C          D
///     |          |                                       |          |
///  a (Sram)   b (Sram)                                a (Sram)   b (Sram)
///
void OpGraph::RemoveRedundantCopiesDramToSram()
{
    using namespace remove_redundant_copies_impl;

    // Look through the graph for chains consisting of just Buffers and DmaOps, starting in Dram
    // and ending in Sram.
    // Search in reverse topological order (from outputs -> inputs), so that we find the longest chains first
    // (as tthe graph exploration happens from bottom up)
    std::vector<Buffer*> buffersSorted = GetSortedBuffers(*this);
    std::vector<DmaChain> chains;
    std::unordered_set<Buffer*> visited;
    for (auto bufferIt = buffersSorted.rbegin(); bufferIt != buffersSorted.rend(); ++bufferIt)
    {
        Buffer* buffer = *bufferIt;
        if (visited.count(buffer) > 0)
        {
            // Don't start a chain partway through another chain, otherwise we will have chains which are
            // subsets of each other. Note that we *can* have chains which share a head though,
            // as is the case for example with Split where a DRAM buffer is split into multiple SRAM
            // buffers across several nested splits. It's just the SRAM buffer at the start which
            // can't be part of another chain.
            continue;
        }
        DmaChain chain = ExploreDmaChainEndingAtSram(*this, buffer);
        chains.push_back(chain);
        visited.insert(chain.buffers.begin(), chain.buffers.end());
    }

    // Check which chains can actually be replaced.
    // These are additional criteria to check compared to what's done in ExploreDmaChainEndingAtSram,
    // which make more sense to be done separately.
    for (DmaChain chain : chains)
    {
        // We might need to shorten the chain to make it valid, so keep trying until it gets too short.
        // Four buffers (Dram -> Sram -> Dram ->) is the minimum length we can optimize.
        while (chain.buffers.size() >= 4)
        {
            DramBuffer* dramBuffer = chain.buffers[0]->Dram();
            SramBuffer* sramBuffer = chain.buffers.back()->Sram();

            // Sum up the DMA offsets along the whole chain, to collapse into one
            TensorShape combinedOffset = chain.GetTotalDmaOffset();

            // We have to be careful not to add an invalid DMA, so check that the stripe shapes etc.
            // are compatible, and if not then try shortening the chain to see if we can replace a
            // sub-section of the chain instead
            if (!impl::IsSramBufferCompatibleWithDramBuffer(*sramBuffer, *dramBuffer, combinedOffset))
            {
                chain.buffers.pop_back();    // Remove SRAM buffer
                chain.buffers.pop_back();    // Remove DRAM buffer
                chain.dmas.pop_back();
                chain.dmas.pop_back();

                // Try again with the shorter chain
                continue;
            }

            // We're now happy to replace this chain as it is valid to be optimised
            // Replace it with a single DMA between the DRAM and SRAM buffers
            // We can't add a new DmaOp as this isn't an OwnedOpGraph, so repurpose one of the existing ones.
            // We can't repurpose the first Dma, as that might be shared, so we repurpose the last one.
            DmaOp* lastDma = chain.dmas.back();
            RemoveConsumer(chain.buffers[chain.buffers.size() - 2], lastDma, 0);

            lastDma->m_TransferFormat = dramBuffer->m_Format;
            lastDma->m_Offset         = combinedOffset;
            lastDma->m_OperationIds   = chain.GetOperationIds();
            AddConsumer(dramBuffer, lastDma, 0);

            // Prune from the bottom - we can't start at the top because that might be shared (somewhere along the chain).
            if (GetConsumers(chain.buffers[chain.buffers.size() - 2]).size() == 0)
            {
                RemoveAndPrune(chain.buffers[chain.buffers.size() - 2]);
            }

            break;    // Chain successfully replaced, move on to the next
        }
    }
}

}    // namespace support_library
}    // namespace ethosn
