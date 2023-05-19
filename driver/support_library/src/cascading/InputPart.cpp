//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "InputPart.hpp"

#include "../BufferManager.hpp"
#include "../Utils.hpp"
#include "PartUtils.hpp"
#include "Plan.hpp"

namespace ethosn
{
namespace support_library
{

Plans InputPart::GetPlans(CascadeType cascadeType,
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
        CreatePlanForInputPart(plans);
    }

    return plans;
}

InputPart::~InputPart()
{}

void InputPart::CreatePlanForInputPart(Plans& plans) const
{
    PartInputMapping inputMappings;
    PartOutputMapping outputMappings;
    OwnedOpGraph opGraph;

    CascadingBufferFormat format = impl::GetCascadingBufferFormatFromCompilerDataFormat(m_CompilerDataFormat);

    std::unique_ptr<DramBuffer> buffer = DramBuffer::Build()
                                             .AddFormat(format)
                                             .AddDataType(m_OutputDataType)
                                             .AddTensorShape(m_OutputTensorShape)
                                             .AddQuantization(m_OutputQuantizationInfo)
                                             .AddBufferType(BufferType::Input)
                                             .AddOperationId(*m_CorrespondingOperationIds.begin());

    outputMappings[buffer.get()] = PartOutputSlot{ m_PartId, 0 };
    opGraph.AddBuffer(std::move(buffer));

    AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph), plans);
}

ethosn::support_library::DotAttributes InputPart::GetDotAttributes(DetailLevel detail) const
{
    DotAttributes result = BasePart::GetDotAttributes(detail);
    if (detail >= DetailLevel::High)
    {
        result.m_Label += "CompilerDataFormat = " + ToString(m_CompilerDataFormat) + "\n";
        result.m_Label += "OutputTensorShape = " + ToString(m_OutputTensorShape) + "\n";
        result.m_Label += "OutputQuantizationInfo = " + ToString(m_OutputQuantizationInfo) + "\n";
        result.m_Label += "OutputDataType = " + ToString(m_OutputDataType) + "\n";
    }
    return result;
}

std::vector<BoundaryRequirements> InputPart::GetInputBoundaryRequirements() const
{
    // InputParts have no inputs
    return {};
}

std::vector<bool> InputPart::CanInputsTakePleInputSram() const
{
    // InputParts have no inputs
    return {};
}

}    // namespace support_library
}    // namespace ethosn
