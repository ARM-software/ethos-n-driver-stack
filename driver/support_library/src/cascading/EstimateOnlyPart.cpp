//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "EstimateOnlyPart.hpp"

#include "../BufferManager.hpp"
#include "../Utils.hpp"
#include "PartUtils.hpp"
#include "Plan.hpp"

namespace ethosn
{
namespace support_library
{

Plans EstimateOnlyPart::GetPlans(CascadeType cascadeType,
                                 ethosn::command_stream::BlockConfig blockConfig,
                                 const std::vector<Buffer*>& sramBufferInputs,
                                 uint32_t numWeightStripes) const
{
    ETHOSN_UNUSED(blockConfig);
    ETHOSN_UNUSED(sramBufferInputs);
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
        std::unique_ptr<DramBuffer> inputBuffer =
            DramBuffer::Build()
                .AddFormat(format)
                .AddDataType(m_InputTensorsInfo[inputIndex].m_DataType)
                .AddTensorShape(m_InputTensorsInfo[inputIndex].m_Dimensions)
                .AddQuantization(m_InputTensorsInfo[inputIndex].m_QuantizationInfo)
                .AddBufferType(BufferType::Intermediate);

        DramBuffer* inputBufferRaw = opGraph.AddBuffer(std::move(inputBuffer));
        opGraph.AddConsumer(inputBufferRaw, op, inputIndex);
        inputMappings[inputBufferRaw] = PartInputSlot{ m_PartId, inputIndex };
    }

    for (uint32_t outputIndex = 0; outputIndex < m_OutputTensorsInfo.size(); outputIndex++)
    {
        std::unique_ptr<DramBuffer> outputBuffer =
            DramBuffer::Build()
                .AddFormat(format)
                .AddDataType(m_OutputTensorsInfo[outputIndex].m_DataType)
                .AddTensorShape(m_OutputTensorsInfo[outputIndex].m_Dimensions)
                .AddQuantization(m_OutputTensorsInfo[outputIndex].m_QuantizationInfo)
                .AddBufferType(BufferType::Intermediate);

        DramBuffer* outputBufferRaw = opGraph.AddBuffer(std::move(outputBuffer));
        opGraph.SetProducer(outputBufferRaw, op);
        outputMappings[outputBufferRaw] = PartOutputSlot{ m_PartId, outputIndex };
    }

    AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph), {}, plans);
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

std::vector<BoundaryRequirements> EstimateOnlyPart::GetInputBoundaryRequirements() const
{
    // We pessimistically assume that we will need boundary data for all of our inputs
    return std::vector<BoundaryRequirements>(m_InputTensorsInfo.size(), BoundaryRequirements{ true, true, true, true });
}

std::vector<bool> EstimateOnlyPart::CanInputsTakePleInputSram() const
{
    // We pessimistically assume that all our inputs need to come from DRAM.
    return std::vector<bool>(m_InputTensorsInfo.size(), false);
}

}    // namespace support_library
}    // namespace ethosn
