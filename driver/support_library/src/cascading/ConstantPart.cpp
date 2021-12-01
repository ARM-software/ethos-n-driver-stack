//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../Utils.hpp"
#include "ConstantPart.hpp"
#include "GraphNodes.hpp"
#include "PartUtils.hpp"
#include "Plan.hpp"

namespace ethosn
{
namespace support_library
{

ConstantPart::ConstantPart(PartId id,
                           const TensorShape& outputTensorShape,
                           const CompilerDataFormat& compilerDataFormat,
                           const QuantizationInfo& quantizationInfo,
                           const std::set<uint32_t>& correspondingOperationIds,
                           const EstimationOptions& estOpt,
                           const CompilationOptions& compOpt,
                           const HardwareCapabilities& capabilities)
    : BasePart(id, compilerDataFormat, correspondingOperationIds, estOpt, compOpt, capabilities)
    , m_OutputTensorShape{ outputTensorShape }
    , m_OutputQuantizationInfo(quantizationInfo)
{}

Plans ConstantPart::GetPlans(CascadeType cascadeType,
                             ethosn::command_stream::BlockConfig blockConfig,
                             Buffer* sramBuffer,
                             uint32_t numWeightStripes) const
{
    ETHOSN_UNUSED(blockConfig);
    ETHOSN_UNUSED(sramBuffer);
    ETHOSN_UNUSED(numWeightStripes);

    Plans plans;

    if (cascadeType == CascadeType::Beginning || cascadeType == CascadeType::Lonely)
    {
        CreatePlanForConstantPart(Lifetime::Atomic, TraversalOrder::Xyz, plans);
    }

    return plans;
}

ConstantPart::~ConstantPart()
{}

void ConstantPart::CreatePlanForConstantPart(Lifetime lifetime, TraversalOrder order, Plans& plans) const
{
    PartInputMapping inputMappings;
    PartOutputMapping outputMappings;
    OwnedOpGraph opGraph;

    CascadingBufferFormat format = impl::GetCascadingBufferFormatFromCompilerDataFormat(m_CompilerDataFormat);
    auto buffer                  = std::make_unique<Buffer>(lifetime, Location::Dram, format, order);
    buffer->m_TensorShape        = m_OutputTensorShape;
    buffer->m_SizeInBytes        = impl::CalculateBufferSize(m_OutputTensorShape, format);
    buffer->m_QuantizationInfo   = m_OutputQuantizationInfo;
    outputMappings[buffer.get()] = PartOutputSlot{ m_PartId, 0 };
    opGraph.AddBuffer(std::move(buffer));

    AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph), plans);
}

ethosn::support_library::DotAttributes ConstantPart::GetDotAttributes(DetailLevel detail) const
{
    DotAttributes result = BasePart::GetDotAttributes(detail);
    result.m_Label       = "ConstantPart: " + result.m_Label;
    if (detail >= DetailLevel::High)
    {
        result.m_Label += "OutputTensorShape = " + ToString(m_OutputTensorShape) + "\n";
        result.m_Label += "OutputQuantizationInfo = " + ToString(m_OutputQuantizationInfo) + "\n";
    }
    return result;
}

}    // namespace support_library
}    // namespace ethosn
