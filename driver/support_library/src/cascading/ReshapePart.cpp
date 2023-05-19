//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../BufferManager.hpp"
#include "ReshapePart.hpp"

#include "PartUtils.hpp"
#include "Plan.hpp"

namespace ethosn
{
namespace support_library
{

Plans ReshapePart::GetPlans(CascadeType cascadeType, ethosn::command_stream::BlockConfig, Buffer*, uint32_t) const
{
    Plans plans;

    if (cascadeType == CascadeType::Lonely)
    {
        std::unique_ptr<DramBuffer> inputBuffer = DramBuffer::Build()
                                                      .AddFormat(CascadingBufferFormat::NHWC)
                                                      .AddDataType(m_DataType)
                                                      .AddTensorShape(m_InputTensorShape)
                                                      .AddQuantization(m_OutputQuantizationInfo)
                                                      .AddBufferType(BufferType::Intermediate);

        Buffer* inputBufferRaw = inputBuffer.get();

        auto dma1            = std::make_unique<DmaOp>(CascadingBufferFormat::NHWC);
        dma1->m_OperationIds = m_CorrespondingOperationIds;
        DmaOp* dma1Raw       = dma1.get();

        // Create a buffer with the best stripe shape
        std::unique_ptr<Buffer> sramBuffer = impl::MakeGlueIntermediateSramBuffer(
            m_InputTensorShape, m_OutputQuantizationInfo, m_DataType, { CascadingBufferFormat::NHWC }, m_Capabilities,
            m_StripeConfig.blockWidthMultiplier.min, m_StripeConfig.blockWidthMultiplier.max,
            m_StripeConfig.blockHeightMultiplier.min, m_StripeConfig.blockHeightMultiplier.max,
            m_StripeConfig.ofmDepthMultiplier.min, m_StripeConfig.ofmDepthMultiplier.max);
        Buffer* sramBufferRaw = sramBuffer.get();
        auto dma2             = std::make_unique<DmaOp>(CascadingBufferFormat::NHWC);
        dma2->m_OperationIds  = m_CorrespondingOperationIds;
        DmaOp* dma2Raw        = dma2.get();

        std::unique_ptr<DramBuffer> outputBuffer = DramBuffer::Build()
                                                       .AddFormat(CascadingBufferFormat::NHWC)
                                                       .AddDataType(m_DataType)
                                                       .AddTensorShape(m_OutputTensorShape)
                                                       .AddQuantization(m_OutputQuantizationInfo)
                                                       .AddBufferType(BufferType::Intermediate);

        Buffer* outputBufferRaw = outputBuffer.get();

        OwnedOpGraph graph;
        graph.AddOp(std::move(dma1));
        graph.AddOp(std::move(dma2));
        graph.AddBuffer(std::move(inputBuffer));
        graph.AddBuffer(std::move(sramBuffer));
        graph.AddBuffer(std::move(outputBuffer));
        graph.AddConsumer(inputBufferRaw, dma1Raw, 0);
        graph.SetProducer(sramBufferRaw, dma1Raw);
        graph.AddConsumer(sramBufferRaw, dma2Raw, 0);
        graph.SetProducer(outputBufferRaw, dma2Raw);

        PartInputMapping inputMappings;
        PartOutputMapping outputMappings;

        inputMappings[inputBufferRaw]   = PartInputSlot{ m_PartId, 0 };
        outputMappings[outputBufferRaw] = PartOutputSlot{ m_PartId, 0 };

        AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(graph), plans);
    }

    return plans;
}

bool ReshapePart::IsOutputGuaranteedNhwc() const
{
    // This allows ConcatPart to generate plans that should lead to a more efficient overall graph.
    return true;
}

ReshapePart::~ReshapePart()
{}

ethosn::support_library::DotAttributes ReshapePart::GetDotAttributes(DetailLevel detail) const
{
    DotAttributes result = BasePart::GetDotAttributes(detail);
    if (detail >= DetailLevel::High)
    {
        result.m_Label += "InputTensorShape = " + ToString(m_InputTensorShape) + "\n";
        result.m_Label += "OutputTensorShape = " + ToString(m_OutputTensorShape) + "\n";
        result.m_Label += "OutputQuantizationInfo = " + ToString(m_OutputQuantizationInfo) + "\n";
        result.m_Label += "DataType = " + ToString(m_DataType) + "\n";
    }
    return result;
}

std::vector<BoundaryRequirements> ReshapePart::GetInputBoundaryRequirements() const
{
    // We have a single input, that does not need any boundary data.
    // This is pretty much irrelevant anyway because we don't cascade into ReshapePart anyway.
    return { BoundaryRequirements{} };
}

std::vector<bool> ReshapePart::CanInputsTakePleInputSram() const
{
    // Our input must be in DRAM
    return { false };
}

}    // namespace support_library
}    // namespace ethosn
