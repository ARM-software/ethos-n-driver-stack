//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../Utils.hpp"
#include "EstimateOnlyPart.hpp"
#include "GraphNodes.hpp"
#include "PartUtils.hpp"
#include "Plan.hpp"

namespace ethosn
{
namespace support_library
{

Plans EstimateOnlyPart::GetPlans(CascadeType cascadeType,
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
        CreatePlanForEstimateOnlyPart(plans);
    }

    return plans;
}

EstimateOnlyPart::~EstimateOnlyPart()
{}

void EstimateOnlyPart::CreatePlanForEstimateOnlyPart(Plans& plans) const
{
    PartInputMapping inputMappings;
    PartOutputMapping outputMappings;
    OwnedOpGraph opGraph;

    auto estimateOnlyOp = std::make_unique<EstimateOnlyOp>(m_ReasonForEstimateOnly);
    Op* op              = opGraph.AddOp(std::move(estimateOnlyOp));
    op->m_OperationIds  = m_CorrespondingOperationIds;

    CascadingBufferFormat format = impl::GetCascadingBufferFormatFromCompilerDataFormat(m_CompilerDataFormat);

    for (uint32_t inputIndex = 0; inputIndex < m_InputTensorsInfo.size(); inputIndex++)
    {
        DramBuffer* inputBuffer         = opGraph.AddBuffer(std::make_unique<DramBuffer>());
        inputBuffer->m_Format           = format;
        inputBuffer->m_DataType         = m_InputTensorsInfo[inputIndex].m_DataType;
        inputBuffer->m_TensorShape      = m_InputTensorsInfo[inputIndex].m_Dimensions;
        inputBuffer->m_SizeInBytes      = utils::CalculateBufferSize(inputBuffer->m_TensorShape, format);
        inputBuffer->m_QuantizationInfo = m_InputTensorsInfo[inputIndex].m_QuantizationInfo;
        inputBuffer->m_BufferType       = BufferType::Intermediate;
        opGraph.AddConsumer(inputBuffer, op, inputIndex);
        inputMappings[inputBuffer] = PartInputSlot{ m_PartId, inputIndex };
    }

    for (uint32_t outputIndex = 0; outputIndex < m_OutputTensorsInfo.size(); outputIndex++)
    {
        DramBuffer* outputBuffer         = opGraph.AddBuffer(std::make_unique<DramBuffer>());
        outputBuffer->m_Format           = format;
        outputBuffer->m_DataType         = m_OutputTensorsInfo[outputIndex].m_DataType;
        outputBuffer->m_TensorShape      = m_OutputTensorsInfo[outputIndex].m_Dimensions;
        outputBuffer->m_SizeInBytes      = utils::CalculateBufferSize(outputBuffer->m_TensorShape, format);
        outputBuffer->m_QuantizationInfo = m_OutputTensorsInfo[outputIndex].m_QuantizationInfo;
        outputBuffer->m_BufferType       = BufferType::Intermediate;
        opGraph.SetProducer(outputBuffer, op);
        outputMappings[outputBuffer] = PartOutputSlot{ m_PartId, outputIndex };
    }

    AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph), plans);
}

ethosn::support_library::DotAttributes EstimateOnlyPart::GetDotAttributes(DetailLevel detail) const
{
    DotAttributes result = BasePart::GetDotAttributes(detail);
    if (detail >= DetailLevel::High)
    {
        result.m_Label += "CompilerDataFormat = " + ToString(m_CompilerDataFormat) + "\n";
        result.m_Label += "InputTensorsInfo = " + ArrayToString(m_InputTensorsInfo) + "\n";
        result.m_Label += "OutputTensorsInfo = " + ArrayToString(m_OutputTensorsInfo) + "\n";
    }
    return result;
}

}    // namespace support_library
}    // namespace ethosn
