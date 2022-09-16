//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "Plan.hpp"
#include "PleKernelDatabase.hpp"

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

OpGraph::ConsumersList OpGraph::GetConsumers(Buffer* buffer) const
{
    auto it = m_BufferConsumers.find(buffer);
    return it != m_BufferConsumers.end() ? it->second : OpGraph::ConsumersList{};
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

OpGraph::BufferList OpGraph::GetInputs(Op* op) const
{
    auto it = m_OpInputs.find(op);
    return it != m_OpInputs.end() ? it->second : OpGraph::BufferList{};
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

Plan::Plan()
    : Plan({}, {})
{}

Plan::Plan(PartInputMapping&& inputMappings, PartOutputMapping&& outputMappings)
    : DebuggableObject("Plan")
    , m_InputMappings(std::move(inputMappings))
    , m_OutputMappings(std::move(outputMappings))
{}

Buffer* Plan::GetInputBuffer(const PartInputSlot& partInputSlot) const
{
    for (const auto& pair : m_InputMappings)
    {
        if (pair.second == partInputSlot)
        {
            return pair.first;
        }
    }
    return nullptr;
}

Buffer* Plan::GetOutputBuffer(const PartOutputSlot& partOutputSlot) const
{
    for (const auto& pair : m_OutputMappings)
    {
        if (pair.second == partOutputSlot)
        {
            return pair.first;
        }
    }
    return nullptr;
}

ethosn::command_stream::BlockConfig Plan::GetBlockConfigures(const PartOutputSlot& partOutputSlot) const
{
    Buffer* outputBuffer = GetOutputBuffer(partOutputSlot);

    Op* opProducer = m_OpGraph.GetSingleProducer(outputBuffer);

    if (opProducer != nullptr && opProducer->GetBlockConfig().has_value())
    {
        return opProducer->GetBlockConfig().value();
    }
    else
    {
        return ethosn::command_stream::BlockConfig{};
    }
}

PleKernelInfo Plan::GetPleKernelInfo(const HardwareCapabilities& cap) const
{
    PleKernelInfo pleKernelInfo;
    pleKernelInfo.m_Size  = 0;
    pleKernelInfo.m_PleOp = nullptr;

    for (auto& op : m_OpGraph.GetOps())
    {
        if (IsObjectOfType<PleOp>(op))
        {
            PleOp* pleOp          = static_cast<PleOp*>(op);
            pleKernelInfo.m_Size  = cap.GetMaxPleSize();
            pleKernelInfo.m_PleOp = pleOp;
            break;
        }
    }

    return pleKernelInfo;
}

Op* OwnedOpGraph::AddOp(std::unique_ptr<Op> op)
{
    // Call base implementation first in case it errors, in which case we don't want to track this Op.
    Op* raw = op.get();
    OpGraph::AddOp(raw);
    m_Ops.emplace_back(std::move(op));
    return raw;
}

Buffer* OwnedOpGraph::AddBuffer(std::unique_ptr<Buffer> buffer)
{
    // Call base implementation first in case it errors, in which case we don't want to track this Op.
    Buffer* raw = buffer.get();
    OpGraph::AddBuffer(raw);
    m_Buffers.emplace_back(std::move(buffer));
    return raw;
}

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

Buffer::Buffer()
    : Buffer(Location::Dram,
             CascadingBufferFormat::NHWCB,
             { 0, 0, 0, 0 },
             { 0, 0, 0, 0 },
             TraversalOrder::Xyz,
             0,
             QuantizationInfo())
{}

Buffer::Buffer(Location location, CascadingBufferFormat format, TraversalOrder order)
    : Buffer(location, format, { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, order, 0, QuantizationInfo())
{}

Buffer::Buffer(Location location,
               CascadingBufferFormat format,
               TensorShape tensorShape,
               TensorShape stripeShape,
               TraversalOrder order,
               uint32_t sizeInBytes,
               QuantizationInfo quantInfo)
    : DebuggableObject("Buffer")
    , m_Location(location)
    , m_DataType(DataType::UINT8_QUANTIZED)
    , m_Format(format)
    , m_QuantizationInfo(quantInfo)
    , m_TensorShape(tensorShape)
    , m_StripeShape(stripeShape)
    , m_Order(order)
    , m_SizeInBytes(sizeInBytes)
    , m_SlotSizeInBytes(0)
    , m_NumStripes(0)
    , m_PackedBoundaryThickness({ 0, 0, 0, 0 })
    , m_NumLoads(1)
{}

bool IsOutputBufferInDram(const Plan& plan, const PartOutputSlot& outputSlot)
{
    const Buffer* buf = plan.GetOutputBuffer(outputSlot);
    return (buf == nullptr) ? true : ((buf->m_Location) == Location::Dram);
}

bool IsInputBufferInSram(const Plan& plan, const PartInputSlot& inputSlot)
{
    const Buffer* buf = plan.GetInputBuffer(inputSlot);
    return (buf == nullptr) ? false : ((buf->m_Location) == Location::Sram);
}

bool IsOutputBufferInSram(const Plan& plan, const PartOutputSlot& outputSlot)
{
    const Buffer* buf = plan.GetOutputBuffer(outputSlot);
    return (buf == nullptr) ? false : ((buf->m_Location) == Location::Sram);
}

SizeInBytes GetTotSizeInBytes(const Plan& plan)
{
    SizeInBytes result;
    const OpGraph::BufferList& bufs        = plan.m_OpGraph.GetBuffers();
    OpGraph::BufferList::const_iterator it = bufs.begin();
    while (it != bufs.end())
    {
        const Buffer* buf   = *it;
        const uint32_t size = buf->m_SizeInBytes;
        if (buf->m_Location == Location::Sram)
        {
            result.m_Tot += size;
        }
        ++it;
    }
    assert(result.m_TotAtomic <= result.m_Tot);
    return result;
}

SizeInBytes GetInputsSizeInBytes(const Plan& plan)
{
    SizeInBytes result;
    const PartInputMapping in           = plan.m_InputMappings;
    PartInputMapping::const_iterator it = in.begin();
    while (it != in.end())
    {
        const Buffer* buf   = it->first;
        const uint32_t size = buf->m_SizeInBytes;
        if (buf->m_Location == Location::Sram)
        {
            result.m_Tot += size;
        }
        ++it;
    }
    assert(result.m_TotAtomic <= result.m_Tot);
    return result;
}

}    // namespace support_library
}    // namespace ethosn
