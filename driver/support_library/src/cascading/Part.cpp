//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "Part.hpp"

#include "../Graph.hpp"
#include "GraphNodes.hpp"
#include "Plan.hpp"

using namespace std;

namespace ethosn
{
namespace support_library
{

std::unique_ptr<Op> CreateOpFromNode(const Node* node)
{
    if (IsObjectOfType<MceOperationNode>(node))
    {
        const MceOperationNode* mceOperationNode = dynamic_cast<const MceOperationNode*>(node);
        MceOp op(Lifetime::Atomic, mceOperationNode->GetOperation(), mceOperationNode->GetAlgorithm(),
                 BlockConfig{ 8U, 8U }, TensorShape{}, TensorShape{}, TensorShape{}, TraversalOrder::Xyz,
                 mceOperationNode->GetStride());
        return std::make_unique<MceOp>(std::move(op));
    }
    else if (IsObjectOfType<McePostProcessOperationNode>(node))
    {
        return std::make_unique<MceOp>();
    }
    else if (IsObjectOfType<FuseOnlyPleOperationNode>(node))
    {
        const FuseOnlyPleOperationNode* fuseOnlyPleOperationNode = dynamic_cast<const FuseOnlyPleOperationNode*>(node);
        PleOp op(Lifetime::Atomic, fuseOnlyPleOperationNode->GetKernelOperation(), BlockConfig{ 8U, 8U },
                 static_cast<uint32_t>(fuseOnlyPleOperationNode->GetInputs().size()), std::vector<TensorShape>{},
                 TensorShape{});
        return std::make_unique<PleOp>(std::move(op));
    }
    else if (IsObjectOfType<StandalonePleOperationNode>(node))
    {
        const StandalonePleOperationNode* standalonePleOperationNode =
            dynamic_cast<const StandalonePleOperationNode*>(node);
        PleOp op(Lifetime::Atomic, standalonePleOperationNode->GetKernelOperation(), BlockConfig{ 8U, 8U },
                 static_cast<uint32_t>(standalonePleOperationNode->GetInputs().size()), std::vector<TensorShape>{},
                 TensorShape{});
        return std::make_unique<PleOp>(std::move(op));
    }
    else if (IsObjectOfType<FormatConversionNode>(node) || IsObjectOfType<ReinterpretNode>(node))
    {
        return std::make_unique<DmaOp>();
    }
    else if (IsObjectOfType<EstimateOnlyNode>(node))
    {
        return std::make_unique<DummyOp>();
    }

    std::cout
        << "Warning: Unsupported node type received during the plan generation. A dummy operation will be inserted."
        << std::endl;
    return std::make_unique<DummyOp>();
}

TensorInfo GetWeightsInfo(const Node* node)
{
    if (IsObjectOfType<MceOperationNode>(node))
    {
        return dynamic_cast<const MceOperationNode*>(node)->GetWeightsInfo();
    }

    return TensorInfo();
}

TensorShape GetWeightsShape(const Node* node)
{
    return GetWeightsInfo(node).m_Dimensions;
}

bool IsPlanValid(const Plan& plan)
{
    (void)plan;
    return true;
}

const Plan& Part::GetPlan(const PlanId id) const
{
    assert(id < m_Plans.size());
    return *m_Plans.at(id).get();
}

size_t Part::GetNumPlans() const
{
    return m_Plans.size();
}

std::vector<const Edge*> Part::GetInputs() const
{
    assert(m_SubGraph.size());
    std::vector<const Edge*> result;

    for (uint32_t n = 0; n < m_SubGraph.size(); ++n)
    {
        bool found        = false;
        const Node& nodeA = *m_SubGraph.at(n);
        for (uint32_t i = 0; i < nodeA.GetInputs().size(); ++i)
        {
            const Edge* in = nodeA.GetInput(i);
            for (uint32_t m = 0; m < m_SubGraph.size(); ++m)
            {
                if (m == n)
                {
                    continue;
                }
                const Node& nodeB = *m_SubGraph.at(m);
                for (uint32_t o = 0; o < nodeB.GetOutputs().size(); ++o)
                {
                    const Edge* out = nodeB.GetOutput(o);
                    if (in == out)
                    {
                        found = true;
                        break;
                    }
                    found = false;
                }
                if (found)
                {
                    break;
                }
            }
            if (!found)
            {
                result.push_back(in);
            }
        }
    }
    return result;
}

std::vector<const Edge*> Part::GetOutputs() const
{
    assert(m_SubGraph.size());
    std::vector<const Edge*> result;

    for (uint32_t n = 0; n < m_SubGraph.size(); ++n)
    {
        bool found        = false;
        const Node& nodeA = *m_SubGraph.at(n);
        for (uint32_t o = 0; o < nodeA.GetOutputs().size(); ++o)
        {
            const Edge* out = nodeA.GetOutput(o);
            for (uint32_t m = 0; m < m_SubGraph.size(); ++m)
            {
                if (m == n)
                {
                    continue;
                }
                const Node& nodeB = *m_SubGraph.at(m);
                for (uint32_t i = 0; i < nodeB.GetInputs().size(); ++i)
                {
                    const Edge* in = nodeB.GetInput(i);
                    if (in == out)
                    {
                        found = true;
                        break;
                    }
                    found = false;
                }
                if (found)
                {
                    break;
                }
            }
            if (!found)
            {
                result.push_back(out);
            }
        }
    }
    return result;
}

InPart GraphOfParts::GetInputPart(const Edge& e) const
{
    const size_t numParts = m_Parts.size();

    for (PartId p = 0; p < numParts; ++p)
    {
        const Part& part = GetPart(p);

        for (uint32_t i = 0; i < part.GetInputs().size(); ++i)
        {
            const Edge* frEdge = part.GetInputs().at(i);
            if (&e == frEdge)
            {
                return std::make_pair(true, p);
            }
        }
    }
    return std::make_pair(false, 0);
}

OutPart GraphOfParts::GetOutputPart(const Edge& e) const
{
    const size_t numParts = m_Parts.size();

    for (PartId p = 0; p < numParts; ++p)
    {
        const Part& part = GetPart(p);

        for (uint32_t i = 0; i < part.GetOutputs().size(); ++i)
        {
            const Edge* bkEdge = part.GetOutputs().at(i);
            if (&e == bkEdge)
            {
                return std::make_pair(true, p);
            }
        }
    }
    return std::make_pair(false, 0);
}

size_t GraphOfParts::GetNumParts() const
{
    return m_Parts.size();
}

const Part& GraphOfParts::GetPart(const PartId id) const
{
    assert(id < m_Parts.size());
    return *m_Parts.at(id).get();
}

const Parts& GraphOfParts::GetParts() const
{
    return m_Parts;
}

void Part::CreatePlans(const HardwareCapabilities& caps)
{
    using NumStripesType = uint32_t;

    auto InsertPlan = [this](Plan::InputMapping&& inputMappings, Plan::OutputMapping&& outputMappings,
                             OwnedOpGraph&& opGraph) -> void {
        auto plan       = std::make_unique<Plan>(std::move(inputMappings), std::move(outputMappings));
        plan->m_OpGraph = std::move(opGraph);
        if (IsPlanValid(*plan))
        {
            this->m_Plans.push_back(std::move(plan));
        }
    };

    auto CreatePlanForInputNode = [&](Node* node, Lifetime lifetime, TraversalOrder order) -> void {
        Plan::InputMapping inputMappings;
        Plan::OutputMapping outputMappings;
        OwnedOpGraph opGraph;

        auto buffer           = std::make_unique<Buffer>(lifetime, Location::Dram, CompilerDataFormat::NHWC, order);
        buffer->m_TensorShape = node->GetShape();
        buffer->m_SizeInBytes = 0;
        outputMappings[buffer.get()] = node;
        opGraph.AddBuffer(std::move(buffer));
        InsertPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph));
    };

    auto CreatePlanForOutputNode = [&](Node* node, Lifetime lifetime, TraversalOrder order) -> void {
        Plan::InputMapping inputMappings;
        Plan::OutputMapping outputMappings;
        OwnedOpGraph opGraph;

        auto buffer = std::make_unique<Buffer>(lifetime, Location::Dram, CompilerDataFormat::NHWC, order);
        assert(node->GetInputs().size() > 0);
        buffer->m_TensorShape       = node->GetInput(0)->GetSource()->GetShape();
        buffer->m_SizeInBytes       = 0;
        inputMappings[buffer.get()] = node->GetInput(0);
        opGraph.AddBuffer(std::move(buffer));
        InsertPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph));
    };

    auto CreatePlanForNode = [&, this](Node* node, Lifetime lifetime, HardwareCapabilities caps,
                                       CompilerDataFormat format, TraversalOrder order, StripeSizeType stripeSize,
                                       NumStripesType numStripes) -> void {
        assert(node->GetInputs().size() > 0);

        Plan::InputMapping inputMappings;
        Plan::OutputMapping outputMappings;
        OwnedOpGraph opGraph;
        auto& buffers = opGraph.GetBuffers();
        auto& ops     = opGraph.GetOps();

        opGraph.AddOp(CreateOpFromNode(node));
        auto op        = ops.back();
        op->m_Lifetime = lifetime;
        opGraph.AddBuffer(std::make_unique<Buffer>(lifetime, Location::Sram, format, order));
        auto inBuffer = buffers.back();
        opGraph.AddBuffer(std::make_unique<Buffer>(lifetime, Location::Sram, format, order));
        auto outBuffer = buffers.back();
        opGraph.AddConsumer(inBuffer, op, 0);
        opGraph.SetProducer(outBuffer, op);

        auto inputNode           = node->GetInput(0)->GetSource();
        auto outputNode          = m_SubGraph.back();
        inBuffer->m_TensorShape  = inputNode->GetShape();
        inBuffer->m_StripeShape  = inputNode->GetShape();
        outBuffer->m_TensorShape = outputNode->GetShape();
        outBuffer->m_StripeShape = outputNode->GetShape();

        switch (format)
        {
            case CompilerDataFormat::NHWC:    // Fall through
            case CompilerDataFormat::NHWCB:
                inBuffer->m_StripeShape[1]  = stripeSize;
                outBuffer->m_StripeShape[1] = stripeSize;
                break;
            default:
                break;
        }
        inBuffer->m_SizeInBytes  = numStripes * CalculateSizeInBytes(inBuffer->m_StripeShape);
        outBuffer->m_SizeInBytes = numStripes * CalculateSizeInBytes(outBuffer->m_StripeShape);

        if (this->m_SubGraph.size() > 1 && IsObjectOfType<McePostProcessOperationNode>(this->m_SubGraph[1]))
        {
            opGraph.AddOp(CreateOpFromNode(this->m_SubGraph[1]));
            auto mcePpOp = ops.back();
            opGraph.AddBuffer(std::make_unique<Buffer>(lifetime, Location::Sram, format, order));
            auto mcePpOpBuffer = buffers.back();
            opGraph.AddConsumer(outBuffer, mcePpOp, 0);
            opGraph.SetProducer(mcePpOpBuffer, mcePpOp);
        }

        inputMappings[buffers.front()] = node->GetInput(0);
        outputMappings[buffers.back()] = this->m_SubGraph.back();

        if (IsObjectOfType<MceOp>(op))
        {
            opGraph.AddBuffer(std::make_unique<Buffer>(lifetime, Location::Dram, format, order));
            auto weightsBufferInDram           = buffers.back();
            weightsBufferInDram->m_TensorShape = GetWeightsShape(node);
            weightsBufferInDram->m_StripeShape = {};
            opGraph.AddBuffer(std::make_unique<Buffer>(lifetime, Location::Sram, format, order));
            auto weightsBufferInSram           = buffers.back();
            weightsBufferInSram->m_TensorShape = weightsBufferInDram->m_TensorShape;
            weightsBufferInSram->m_StripeShape = GetWeightsShape(node);
            weightsBufferInSram->m_SizeInBytes = utils::EstimateWeightSizeBytes(
                GetWeightsShape(node), caps, GetWeightsInfo(node).m_DataFormat == DataFormat::HWIM);
            weightsBufferInSram->m_Format = CompilerDataFormat::WEIGHT;
            opGraph.AddOp(std::make_unique<DmaOp>());
            auto dmaOp                  = ops.back();
            MceOp* mceOp                = dynamic_cast<MceOp*>(op);
            mceOp->m_InputStripeShape   = inBuffer->m_StripeShape;
            mceOp->m_OutputStripeShape  = outBuffer->m_StripeShape;
            mceOp->m_WeightsStripeShape = weightsBufferInSram->m_StripeShape;
            opGraph.AddConsumer(weightsBufferInDram, dmaOp, 0);
            opGraph.SetProducer(weightsBufferInSram, dmaOp);
            opGraph.AddConsumer(weightsBufferInSram, op, 1);
        }
        if (IsObjectOfType<PleOp>(op))
        {
            PleOp* pleOp = dynamic_cast<PleOp*>(op);
            // Support only for single inputs
            pleOp->m_InputStripeShapes.push_back(inBuffer->m_StripeShape);
            pleOp->m_OutputStripeShape = outBuffer->m_StripeShape;
        }

        InsertPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph));
    };

    auto GetMaxStripeSize = [](CompilerDataFormat format, Node* node) -> StripeSizeType {
        switch (format)
        {
            case CompilerDataFormat::NHWC:    // Fall through
            case CompilerDataFormat::NHWCB:
                return node->GetShape()[1];
            default:
                return 0;
        }
    };

    using DataFormats                      = std::list<CompilerDataFormat>;
    const DataFormats supportedDataFormats = { CompilerDataFormat::NHWC, CompilerDataFormat::NHWCB };

    Node* node = m_SubGraph.front();
    if (IsObjectOfType<InputNode>(node))
    {
        CreatePlanForInputNode(node, Lifetime::Atomic, TraversalOrder::Xyz);
    }
    else if (IsObjectOfType<OutputNode>(node))
    {
        CreatePlanForOutputNode(node, Lifetime::Atomic, TraversalOrder::Xyz);
    }
    for (auto lifetime = Lifetime::Atomic; lifetime <= Lifetime::Cascade; lifetime = utils::NextEnumValue(lifetime))
    {
        for (auto order = TraversalOrder::Xyz; order <= TraversalOrder::Zxy; order = utils::NextEnumValue(order))
        {
            for (auto format = supportedDataFormats.cbegin(); format != supportedDataFormats.cend(); ++format)
            {
                StripeSizeType maxStripeSize =
                    utils::RoundUpToNearestMultiple(GetMaxStripeSize(*format, node), caps.GetBrickGroupShape()[1]);
                maxStripeSize =
                    (maxStripeSize > caps.GetBrickGroupShape()[1]) ? caps.GetBrickGroupShape()[1] : maxStripeSize;
                StripeSizeType minStripeSize = caps.GetBrickGroupShape()[1];
                for (StripeSizeType stripeSize = minStripeSize; stripeSize <= maxStripeSize; stripeSize += 8)
                {
                    CreatePlanForNode(node, lifetime, caps, *format, order, stripeSize, 4U);
                }
            }
        }
    }
    if (m_Plans.empty())
    {
        throw NotSupportedException("No plans generated for this part");
    }
}

uint32_t Part::CalculateSizeInBytes(const TensorShape& shape) const
{
    return utils::TotalSizeBytesNHWCB(shape);
}

}    // namespace support_library
}    // namespace ethosn
