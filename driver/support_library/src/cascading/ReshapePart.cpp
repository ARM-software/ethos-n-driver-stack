//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "ReshapePart.hpp"

#include "PartUtils.hpp"
#include "Plan.hpp"

namespace ethosn
{
namespace support_library
{

ReshapePart::ReshapePart(PartId id,
                         const TensorShape& inputTensorShape,
                         const TensorShape& outputTensorShape,
                         const CompilerDataFormat& compilerDataFormat,
                         const QuantizationInfo& quantizationInfo,
                         const std::set<uint32_t>& correspondingOperationIds,
                         const EstimationOptions& estOpt,
                         const CompilationOptions& compOpt,
                         const HardwareCapabilities& capabilities)
    : BasePart(id, "ReshapePart", compilerDataFormat, correspondingOperationIds, estOpt, compOpt, capabilities)
    , m_InputTensorShape{ inputTensorShape }
    , m_OutputTensorShape{ outputTensorShape }
    , m_OutputQuantizationInfo(quantizationInfo)
    , m_StripeConfig(impl::GetDefaultStripeConfig(compOpt, m_DebugTag.c_str()))
{}

Plans ReshapePart::GetPlans(CascadeType cascadeType,
                            ethosn::command_stream::BlockConfig blockConfig,
                            Buffer* sramBuffer,
                            uint32_t numWeightStripes) const
{
    ETHOSN_UNUSED(blockConfig);
    ETHOSN_UNUSED(sramBuffer);
    ETHOSN_UNUSED(numWeightStripes);

    Plans plans;

    if (cascadeType == CascadeType::Lonely)
    {
        const uint32_t minWidthMultiplier = m_StripeConfig.blockWidthMultiplier.min;
        const uint32_t maxWidthMultiplier = std::max(
            1U, std::min(utils::GetWidth(m_InputTensorShape) / utils::GetWidth(m_Capabilities.GetBrickGroupShape()),
                         m_StripeConfig.blockWidthMultiplier.max));
        const uint32_t minHeightMultiplier = m_StripeConfig.blockHeightMultiplier.min;
        const uint32_t maxHeightMultiplier = std::max(
            1U, std::min(utils::GetHeight(m_InputTensorShape) / utils::GetHeight(m_Capabilities.GetBrickGroupShape()),
                         m_StripeConfig.blockHeightMultiplier.max));
        for (uint32_t heightMultiplier = minHeightMultiplier; heightMultiplier <= maxHeightMultiplier;
             heightMultiplier *= 2)
        {
            for (uint32_t widthMultiplier = minWidthMultiplier; widthMultiplier <= maxWidthMultiplier;
                 widthMultiplier *= 2)
            {
                auto inputBuffer = std::make_unique<Buffer>(
                    Location::Dram, CascadingBufferFormat::NHWC, m_InputTensorShape, TensorShape{ 0, 0, 0, 0 },
                    TraversalOrder::Xyz, utils::TotalSizeBytes(m_InputTensorShape), m_OutputQuantizationInfo);
                inputBuffer->m_BufferType = BufferType::Intermediate;
                Buffer* inputBufferRaw    = inputBuffer.get();

                auto dma1            = std::make_unique<DmaOp>(CascadingBufferFormat::NHWC);
                dma1->m_OperationIds = m_CorrespondingOperationIds;
                DmaOp* dma1Raw       = dma1.get();

                // We can't split depth because if one of the buffers is NHWC that won't be compatible.
                TensorShape sramStripeShape = {
                    1, utils::GetHeight(m_Capabilities.GetBrickGroupShape()) * heightMultiplier,
                    utils::GetWidth(m_Capabilities.GetBrickGroupShape()) * widthMultiplier,
                    utils::RoundUpToNearestMultiple(m_InputTensorShape[3],
                                                    utils::GetChannels(m_Capabilities.GetBrickGroupShape()))
                };
                auto sramBuffer = std::make_unique<Buffer>(
                    Location::Sram, CascadingBufferFormat::NHWCB, m_InputTensorShape, sramStripeShape,
                    TraversalOrder::Xyz, utils::TotalSizeBytesNHWCB(sramStripeShape), m_OutputQuantizationInfo);
                sramBuffer->m_BufferType = BufferType::Intermediate;
                sramBuffer->m_Offset =
                    0;    // Nothing else should be resident in SRAM at this point, so we can use any address
                sramBuffer->m_NumStripes      = 1;
                sramBuffer->m_SlotSizeInBytes = sramBuffer->m_SizeInBytes;

                Buffer* sramBufferRaw = sramBuffer.get();
                auto dma2             = std::make_unique<DmaOp>(CascadingBufferFormat::NHWC);
                dma2->m_OperationIds  = m_CorrespondingOperationIds;
                DmaOp* dma2Raw        = dma2.get();

                auto outputBuffer = std::make_unique<Buffer>(
                    Location::Dram, CascadingBufferFormat::NHWC, m_OutputTensorShape, TensorShape{ 0, 0, 0, 0 },
                    TraversalOrder::Xyz, utils::TotalSizeBytes(m_OutputTensorShape), m_OutputQuantizationInfo);
                outputBuffer->m_BufferType = BufferType::Intermediate;
                Buffer* outputBufferRaw    = outputBuffer.get();

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
        }
    }

    return plans;
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
    }
    return result;
}

}    // namespace support_library
}    // namespace ethosn
