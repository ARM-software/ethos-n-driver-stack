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

ethosn::support_library::Op* OpGraph::GetProducer(Buffer* buffer) const
{
    auto it = m_BufferProducers.find(buffer);
    return it != m_BufferProducers.end() ? it->second : nullptr;
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
        throw std::runtime_error("buffer is not part of this graph (or is nullptr)");
    }
    if (!Contains(producerOp))
    {
        throw std::runtime_error("producerOp is not part of this graph (or is nullptr)");
    }
    auto it = m_BufferProducers.find(buffer);
    if (it != m_BufferProducers.end() && it->second != nullptr)
    {
        throw std::runtime_error("Buffer is already produced by an Op. It must be disconnected first.");
    }
    m_BufferProducers[buffer] = producerOp;
    m_OpOutputs[producerOp]   = buffer;
}

void OpGraph::ClearProducer(Buffer* buffer)
{
    if (!Contains(buffer))
    {
        throw std::runtime_error("buffer is not part of this graph (or is nullptr)");
    }
    auto oldProducerIt = m_BufferProducers.find(buffer);
    if (oldProducerIt != m_BufferProducers.end())
    {
        m_OpOutputs.erase(oldProducerIt->second);
    }
    m_BufferProducers.erase(buffer);
}

void OpGraph::AddConsumer(Buffer* buffer, Op* consumerOp, uint32_t opInputIdx)
{
    if (!Contains(buffer))
    {
        throw std::runtime_error("buffer is not part of this graph (or is nullptr)");
    }
    if (!Contains(consumerOp))
    {
        throw std::runtime_error("consumerOp is not part of this graph (or is nullptr)");
    }
    auto it = m_OpInputs.find(consumerOp);
    if (it != m_OpInputs.end() && opInputIdx < it->second.size() && it->second[opInputIdx] != nullptr)
    {
        throw std::runtime_error(
            "consumerOp is already consuming a buffer at opInputIdx. It must be disconnected first.");
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

    Op* opProducer = m_OpGraph.GetProducer(outputBuffer);

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

Op::Op(const char* defaultTagPrefix)
    : DebuggableObject(defaultTagPrefix)
    , m_Lifetime(Lifetime::Cascade)
    , m_OperationIds()
{}

Op::Op(const char* defaultTagPrefix, Lifetime lifetime)
    : DebuggableObject(defaultTagPrefix)
    , m_Lifetime(lifetime)
    , m_OperationIds()
{}

DmaOp::DmaOp()
    : Op("DmaOp")
{}

DmaOp::DmaOp(Lifetime lifetime)
    : Op("DmaOp", lifetime)
{}

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
    , m_UpsampleType(command_stream::UpsampleType::OFF)
    , m_LowerBound(0)
    , m_UpperBound(255)
{}

MceOp::MceOp(Lifetime lifetime,
             MceOperation op,
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
    : Op("MceOp", lifetime)
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
    , m_UpsampleType(command_stream::UpsampleType::OFF)
    , m_LowerBound(lowerBound)
    , m_UpperBound(upperBound)
{}

PleOp::PleOp()
    : Op("PleOp")
    , m_Op(PleOperation::FAULT)
    , m_BlockConfig{ 0u, 0u }
    , m_NumInputs(0)
    , m_InputStripeShapes{}
    , m_OutputStripeShape{ 0, 0, 0, 0 }
    , m_OutputDataType{ command_stream::DataType::U8 }
    , m_PleKernelId{ command_stream::cascading::PleKernelId::NOT_FOUND }
    , m_LoadKernel{ true }
{}

PleOp::PleOp(Lifetime lifetime,
             PleOperation op,
             BlockConfig blockConfig,
             uint32_t numInputs,
             std::vector<TensorShape> inputStripeShapes,
             TensorShape outputStripeShape,
             command_stream::DataType dataType,
             bool loadKernel)
    : Op("PleOp", lifetime)
    , m_Op(op)
    , m_BlockConfig(blockConfig)
    , m_NumInputs(numInputs)
    , m_InputStripeShapes(inputStripeShapes)
    , m_OutputStripeShape(outputStripeShape)
    , m_OutputDataType(dataType)
    , m_LoadKernel(loadKernel)
{
    m_PleKernelId = plelib::FindPleKernelIdFromDatabase(blockConfig, (inputStripeShapes.at(0))[2], dataType, op);
}

ConcatOp::ConcatOp()
    : Op("ConcatOp")
{}

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
    , m_Format(format)
    , m_QuantizationInfo(quantInfo)
    , m_TensorShape(tensorShape)
    , m_StripeShape(stripeShape)
    , m_Order(order)
    , m_SizeInBytes(sizeInBytes)
    , m_NumStripes(0)
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
