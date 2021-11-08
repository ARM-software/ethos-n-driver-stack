//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../Utils.hpp"
#include "GraphNodes.hpp"
#include "OutputPart.hpp"
#include "PartUtils.hpp"
#include "Plan.hpp"

namespace ethosn
{
namespace support_library
{

OutputPart::OutputPart(PartId id,
                       const TensorShape& inputTensorShape,
                       const CompilerDataFormat& compilerDataFormat,
                       const QuantizationInfo& quantizationInfo,
                       const std::set<uint32_t>& correspondingOperationIds,
                       const EstimationOptions& estOpt,
                       const CompilationOptions& compOpt,
                       const HardwareCapabilities& capabilities)
    : BasePart(id, compilerDataFormat, correspondingOperationIds, estOpt, compOpt, capabilities)
    , m_InputTensorShape{ inputTensorShape }
    , m_InputQuantizationInfo(quantizationInfo)
{}

Plans OutputPart::GetPlans(CascadeType cascadeType,
                           ethosn::command_stream::BlockConfig blockConfig,
                           Buffer* sramBuffer,
                           uint32_t numWeightStripes) const
{
    ETHOSN_UNUSED(blockConfig);
    ETHOSN_UNUSED(sramBuffer);
    ETHOSN_UNUSED(numWeightStripes);

    Plans plans;

    if (cascadeType == CascadeType::End || cascadeType == CascadeType::Lonely)
    {
        CreatePlanForOutputPart(Lifetime::Atomic, TraversalOrder::Xyz, plans);
    }

    return plans;
}

OutputPart::~OutputPart()
{}

void OutputPart::CreatePlanForOutputPart(Lifetime lifetime, TraversalOrder order, Plans& plans) const
{
    PartInputMapping inputMappings;
    PartOutputMapping outputMappings;
    OwnedOpGraph opGraph;

    CascadingBufferFormat format   = impl::GetCascadingBufferFormatFromCompilerDataFormat(m_CompilerDataFormat);
    std::unique_ptr<Buffer> buffer = std::make_unique<Buffer>(lifetime, Location::Dram, format, order);
    buffer->m_TensorShape          = m_InputTensorShape;
    buffer->m_SizeInBytes          = impl::CalculateBufferSize(m_InputTensorShape, format);
    buffer->m_QuantizationInfo     = m_InputQuantizationInfo;
    inputMappings[buffer.get()]    = PartInputSlot{ m_PartId, 0 };
    opGraph.AddBuffer(std::move(buffer));

    AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph), plans);
}

}    // namespace support_library
}    // namespace ethosn