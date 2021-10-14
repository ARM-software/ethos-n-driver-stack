//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../Utils.hpp"
#include "GraphNodes.hpp"
#include "InputPart.hpp"
#include "PartUtils.hpp"
#include "Plan.hpp"

namespace ethosn
{
namespace support_library
{

InputPart::InputPart(PartId id,
                     const TensorShape& outputTensorShape,
                     const CompilerDataFormat& compilerDataFormat,
                     const QuantizationInfo& quantizationInfo,
                     const std::set<uint32_t>& correspondingOperationIds,
                     const EstimationOptions& estOpt,
                     const CompilationOptions& compOpt,
                     const HardwareCapabilities& capabilities)
    : BasePart(id, compilerDataFormat, quantizationInfo, correspondingOperationIds, estOpt, compOpt, capabilities)
    , m_OutputTensorShape{ outputTensorShape }
{}

Plans InputPart::GetPlans(CascadeType cascadeType,
                          ethosn::command_stream::BlockConfig blockConfig,
                          Buffer* sramBuffer,
                          uint32_t numWeightStripes) const
{
    ETHOSN_UNUSED(cascadeType);
    ETHOSN_UNUSED(blockConfig);
    ETHOSN_UNUSED(sramBuffer);
    ETHOSN_UNUSED(numWeightStripes);

    assert(cascadeType == CascadeType::Beginning || cascadeType == CascadeType::Lonely);

    Plans plans;

    CreatePlanForInputPart(Lifetime::Atomic, TraversalOrder::Xyz, plans);

    return plans;
}

InputPart::~InputPart()
{}

void InputPart::CreatePlanForInputPart(Lifetime lifetime, TraversalOrder order, Plans& plans) const
{
    PartInputMapping inputMappings;
    PartOutputMapping outputMappings;
    OwnedOpGraph opGraph;

    CascadingBufferFormat format = PartUtils::GetCascadingBufferFormatFromCompilerDataFormat(m_CompilerDataFormat);
    auto buffer                  = std::make_unique<Buffer>(lifetime, Location::Dram, format, order);
    buffer->m_TensorShape        = m_OutputTensorShape;
    buffer->m_SizeInBytes        = PartUtils::CalculateBufferSize(m_OutputTensorShape, format);
    buffer->m_QuantizationInfo   = m_QuantizationInfo;
    outputMappings[buffer.get()] = PartOutputSlot{ m_PartId, 0 };
    opGraph.AddBuffer(std::move(buffer));

    AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph), plans);
}

}    // namespace support_library
}    // namespace ethosn
