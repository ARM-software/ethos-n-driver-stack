//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "Plan.hpp"

using namespace std;

namespace ethosn
{
namespace support_library
{

const OpGraph::OpList& OpGraph::GetOps() const
{
    return m_Ops;
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
    if (opInputIdx >= inputs.size())
    {
        inputs.resize(opInputIdx + 1, nullptr);
    }
    inputs[opInputIdx] = buffer;
}

Plan::Plan()
    : Plan({}, {})
{}

Plan::Plan(InputMapping&& inputMappings, OutputMapping&& outputMappings)
    : DebuggableObject("Plan")
    , m_InputMappings(std::move(inputMappings))
    , m_OutputMappings(std::move(outputMappings))
{}

Buffer* Plan::GetInputBuffer(const Edge* inputEdge) const
{
    for (const auto& pair : m_InputMappings)
    {
        if (pair.second == inputEdge)
        {
            return pair.first;
        }
    }
    return nullptr;
}

Buffer* Plan::GetOutputBuffer(const Node* outputNode) const
{
    for (const auto& pair : m_OutputMappings)
    {
        if (pair.second == outputNode)
        {
            return pair.first;
        }
    }
    return nullptr;
}

const OwnedOpGraph& Plan::getOwnedOpGraph() const
{
    return m_OpGraph;
}

void OwnedOpGraph::AddOp(std::unique_ptr<Op> op)
{
    // Call base implementation first in case it errors, in which case we don't want to track this Op.
    OpGraph::AddOp(op.get());
    m_Ops.push_back(std::move(op));
}

void OwnedOpGraph::AddBuffer(std::unique_ptr<Buffer> buffer)
{
    // Call base implementation first in case it errors, in which case we don't want to track this Op.
    OpGraph::AddBuffer(buffer.get());
    m_Buffers.push_back(std::move(buffer));
}

int DebuggableObject::ms_IdCounter = 0;

DebuggableObject::DebuggableObject(const char* defaultTagPrefix)
{
    // Generate an arbitrary and unique (but deterministic) default debug tag for this object.
    // This means that if no-one sets anything more useful, we still have a way to identify it.
    m_DebugTag = std::string(defaultTagPrefix) + " " + std::to_string(ms_IdCounter);
    ++ms_IdCounter;
}

Op::Op(const char* defaultTagPrefix)
    : DebuggableObject(defaultTagPrefix)
    , m_Lifetime(Lifetime::Atomic)
{}

Op::Op(const char* defaultTagPrefix, Lifetime lifetime)
    : DebuggableObject(defaultTagPrefix)
    , m_Lifetime(lifetime)
{}

DmaOp::DmaOp()
    : Op("DmaOp")
    , m_Location(Location::Dram)
    , m_Format(CompilerDataFormat::NONE)
{}

DmaOp::DmaOp(Lifetime lifetime, Location location, CompilerDataFormat format)
    : Op("DmaOp", lifetime)
    , m_Location(location)
    , m_Format(format)
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
{}

MceOp::MceOp(Lifetime lifetime,
             MceOperation op,
             CompilerMceAlgorithm algo,
             BlockConfig blockConfig,
             TensorShape inputStripeShape,
             TensorShape outputStripeShape,
             TensorShape weightsStripeShape,
             TraversalOrder order,
             Stride stride)
    : Op("MceOp", lifetime)
    , m_Op(op)
    , m_Algo(algo)
    , m_BlockConfig(blockConfig)
    , m_InputStripeShape(inputStripeShape)
    , m_OutputStripeShape(outputStripeShape)
    , m_WeightsStripeShape(weightsStripeShape)
    , m_Order(order)
    , m_Stride(stride)
{}

PleOp::PleOp()
    : Op("PleOp")
    , m_Op(PleOperation::FAULT)
    , m_BlockConfig{ 0u, 0u }
    , m_NumInputs(0)
    , m_InputStripeShapes{}
    , m_OutputStripeShape{ 0, 0, 0, 0 }
{}

PleOp::PleOp(Lifetime lifetime,
             PleOperation op,
             BlockConfig blockConfig,
             uint32_t numInputs,
             std::vector<TensorShape> inputStripeShapes,
             TensorShape outputStripeShape)
    : Op("PleOp", lifetime)
    , m_Op(op)
    , m_BlockConfig(blockConfig)
    , m_NumInputs(numInputs)
    , m_InputStripeShapes(inputStripeShapes)
    , m_OutputStripeShape(outputStripeShape)
{}

DummyOp::DummyOp()
    : Op("DummyOp")
{}

Buffer::Buffer()
    : Buffer(Lifetime::Atomic,
             Location::Dram,
             CompilerDataFormat::NONE,
             { 0, 0, 0, 0 },
             { 0, 0, 0, 0 },
             TraversalOrder::Xyz,
             0)
{}

Buffer::Buffer(Lifetime lifetime, Location location, CompilerDataFormat format, TraversalOrder order)
    : Buffer(lifetime, location, format, { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, order, 0)
{}

Buffer::Buffer(Lifetime lifetime,
               Location location,
               CompilerDataFormat format,
               TensorShape tensorShape,
               TensorShape stripeShape,
               TraversalOrder order,
               uint32_t sizeInBytes)
    : DebuggableObject("Buffer")
    , m_Lifetime(lifetime)
    , m_Location(location)
    , m_Format(format)
    , m_TensorShape(tensorShape)
    , m_StripeShape(stripeShape)
    , m_Order(order)
    , m_SizeInBytes(sizeInBytes)
{}

}    // namespace support_library
}    // namespace ethosn
