//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "ConcatPart.hpp"

#include "../Utils.hpp"
#include "ConcreteOperations.hpp"
#include "GraphNodes.hpp"
#include "PartUtils.hpp"
#include "Plan.hpp"

namespace ethosn
{
namespace support_library
{

ConcatPart::ConcatPart(PartId id,
                       const std::vector<TensorInfo>& inputTensorsInfo,
                       const ConcatenationInfo& concatInfo,
                       const CompilerDataFormat& compilerDataFormat,
                       const std::set<uint32_t>& correspondingOperationIds,
                       const EstimationOptions& estOpt,
                       const CompilationOptions& compOpt,
                       const HardwareCapabilities& capabilities)
    : BasePart(id, "ConcatPart", compilerDataFormat, correspondingOperationIds, estOpt, compOpt, capabilities)
    , m_InputTensorsInfo{ inputTensorsInfo }
    , m_ConcatInfo{ concatInfo }
    , m_StripeConfig(impl::GetDefaultStripeConfig(compOpt, m_DebugTag.c_str()))
{}

Plans ConcatPart::GetPlans(CascadeType cascadeType,
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
        CreateConcatDramPlans(plans);
    }

    return plans;
}

ConcatPart::~ConcatPart()
{}

void ConcatPart::CreateConcatDramPlans(Plans& plans) const
{
    CascadingBufferFormat format = impl::GetCascadingBufferFormatFromCompilerDataFormat(m_CompilerDataFormat);

    TensorInfo outputInfo = Concatenation::CalculateOutputTensorInfo(m_InputTensorsInfo, m_ConcatInfo);

    uint32_t minWidthMultiplier = m_StripeConfig.blockWidthMultiplier.min;
    uint32_t maxWidthMultiplier =
        std::max(1U, std::min(utils::DivRoundUp(utils::GetWidth(outputInfo.m_Dimensions),
                                                utils::GetWidth(m_Capabilities.GetBrickGroupShape())),
                              m_StripeConfig.blockWidthMultiplier.max));
    uint32_t minHeightMultiplier = m_StripeConfig.blockHeightMultiplier.min;
    uint32_t maxHeightMultiplier =
        std::max(1U, std::min(utils::DivRoundUp(utils::GetHeight(outputInfo.m_Dimensions),
                                                utils::GetHeight(m_Capabilities.GetBrickGroupShape())),
                              m_StripeConfig.blockHeightMultiplier.max));
    if (m_ConcatInfo.m_Axis == 3 &&
        std::any_of(m_InputTensorsInfo.begin(), m_InputTensorsInfo.end(), [&](const TensorInfo& t) {
            return t.m_Dimensions[3] % m_Capabilities.GetBrickGroupShape()[3] != 0;
        }))
    {
        // When splitting channels by multiples of less than 16, the firmware requires that the stripe shape is 8x8 (WxH)
        minWidthMultiplier  = 1;
        maxWidthMultiplier  = 1;
        minHeightMultiplier = 1;
        maxHeightMultiplier = 1;
    }

    PartInputMapping inputMappings;
    PartOutputMapping outputMappings;
    OwnedOpGraph opGraph;

    opGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, format, TraversalOrder::Xyz));
    Buffer* outputBuffer             = opGraph.GetBuffers().back();
    outputBuffer->m_DataType         = outputInfo.m_DataType;
    outputBuffer->m_TensorShape      = outputInfo.m_Dimensions;
    outputBuffer->m_SizeInBytes      = utils::CalculateBufferSize(outputBuffer->m_TensorShape, format);
    outputBuffer->m_QuantizationInfo = outputInfo.m_QuantizationInfo;
    outputBuffer->m_BufferType       = BufferType::Intermediate;
    outputMappings[outputBuffer]     = PartOutputSlot{ m_PartId, 0 };

    TensorShape offset = { 0, 0, 0, 0 };
    for (uint32_t inputIndex = 0; inputIndex < m_InputTensorsInfo.size(); inputIndex++)
    {
        opGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, format, TraversalOrder::Xyz));
        Buffer* inputBuffer             = opGraph.GetBuffers().back();
        inputBuffer->m_DataType         = m_InputTensorsInfo[inputIndex].m_DataType;
        inputBuffer->m_TensorShape      = m_InputTensorsInfo[inputIndex].m_Dimensions;
        inputBuffer->m_SizeInBytes      = utils::CalculateBufferSize(inputBuffer->m_TensorShape, format);
        inputBuffer->m_QuantizationInfo = m_InputTensorsInfo[inputIndex].m_QuantizationInfo;
        inputBuffer->m_BufferType       = BufferType::Intermediate;
        inputMappings[inputBuffer]      = PartInputSlot{ m_PartId, inputIndex };

        auto dma1            = std::make_unique<DmaOp>(format);
        dma1->m_OperationIds = m_CorrespondingOperationIds;
        DmaOp* dma1Raw       = dma1.get();
        opGraph.AddOp(std::move(dma1));

        const uint32_t stripeDepth =
            m_ConcatInfo.m_Axis == 3
                ? m_InputTensorsInfo[inputIndex].m_Dimensions[3]
                : utils::RoundUpToNearestMultiple(m_InputTensorsInfo[inputIndex].m_Dimensions[3],
                                                  utils::GetChannels(m_Capabilities.GetBrickGroupShape()));
        // Create a buffer with the best stripe shape
        std::unique_ptr<Buffer> sramBuffer = impl::MakeGlueIntermediateSramBuffer(
            m_InputTensorsInfo[inputIndex].m_Dimensions, outputInfo.m_QuantizationInfo, outputInfo.m_DataType,
            m_Capabilities, stripeDepth, minWidthMultiplier, maxWidthMultiplier, minHeightMultiplier,
            maxHeightMultiplier);
        Buffer* sramBufferRaw = sramBuffer.get();
        opGraph.AddBuffer(std::move(sramBuffer));

        auto dma2            = std::make_unique<DmaOp>(format);
        dma2->m_OperationIds = m_CorrespondingOperationIds;
        dma2->m_Offset       = offset;
        DmaOp* dma2Raw       = dma2.get();
        opGraph.AddOp(std::move(dma2));

        opGraph.AddConsumer(inputBuffer, dma1Raw, 0);
        opGraph.SetProducer(sramBufferRaw, dma1Raw);
        opGraph.AddConsumer(sramBufferRaw, dma2Raw, 0);
        opGraph.AddProducer(outputBuffer, dma2Raw);

        offset[m_ConcatInfo.m_Axis] =
            offset[m_ConcatInfo.m_Axis] + m_InputTensorsInfo[inputIndex].m_Dimensions[m_ConcatInfo.m_Axis];
    }

    // Note that we don't use AddNewPlan as the validation is wrong for SRAM (not all our buffers need to
    // be alive at the same time)
    Plan plan(std::move(inputMappings), std::move(outputMappings));
    plan.m_OpGraph = std::move(opGraph);
    // Prevent the Combiner from doing its own SRAM allocation for our SRAM buffers, as this makes pessimistic assumptions
    // about the lifetimes (that they must all be alive at the same time), which can lead to poor performance.
    plan.m_IsPreallocated = true;
    plans.push_back(std::move(plan));
}

ethosn::support_library::DotAttributes ConcatPart::GetDotAttributes(DetailLevel detail) const
{
    DotAttributes result = BasePart::GetDotAttributes(detail);
    if (detail >= DetailLevel::High)
    {
        result.m_Label += "InputTensorsInfo = " + ArrayToString(m_InputTensorsInfo) + "\n";
        result.m_Label += "ConcatInfo.Axis = " + ToString(m_ConcatInfo.m_Axis) + "\n";
        result.m_Label += "ConcatInfo.OutputQuantInfo = " + ToString(m_ConcatInfo.m_OutputQuantizationInfo) + "\n";
    }
    return result;
}

}    // namespace support_library
}    // namespace ethosn
