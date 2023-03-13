//
// Copyright © 2021-2023 Arm Limited.
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
                           DataType dataType,
                           const std::set<uint32_t>& correspondingOperationIds,
                           const EstimationOptions& estOpt,
                           const CompilationOptions& compOpt,
                           const HardwareCapabilities& capabilities,
                           const std::vector<uint8_t>& constantData)
    : BasePart(id, "ConstantPart", correspondingOperationIds, estOpt, compOpt, capabilities)
    , m_OutputTensorShape{ outputTensorShape }
    , m_OutputQuantizationInfo(quantizationInfo)
    , m_OutputDataType(dataType)
    , m_CompilerDataFormat(compilerDataFormat)
    , m_ConstantData(std::make_shared<std::vector<uint8_t>>(constantData))
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
        CreatePlanForConstantPart(plans);
    }

    return plans;
}

ConstantPart::~ConstantPart()
{}

void ConstantPart::CreatePlanForConstantPart(Plans& plans) const
{
    PartInputMapping inputMappings;
    PartOutputMapping outputMappings;
    OwnedOpGraph opGraph;

    CascadingBufferFormat format = impl::GetCascadingBufferFormatFromCompilerDataFormat(m_CompilerDataFormat);
    auto buffer                  = std::make_unique<DramBuffer>();
    buffer->m_Format             = format;
    buffer->m_DataType           = m_OutputDataType;
    buffer->m_TensorShape        = m_OutputTensorShape;
    buffer->m_SizeInBytes        = utils::CalculateBufferSize(m_OutputTensorShape, format);
    buffer->m_QuantizationInfo   = m_OutputQuantizationInfo;
    buffer->m_BufferType         = BufferType::ConstantDma;
    buffer->m_ConstantData       = m_ConstantData;
    outputMappings[buffer.get()] = PartOutputSlot{ m_PartId, 0 };
    opGraph.AddBuffer(std::move(buffer));

    AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph), plans);
}

ethosn::support_library::DotAttributes ConstantPart::GetDotAttributes(DetailLevel detail) const
{
    DotAttributes result = BasePart::GetDotAttributes(detail);
    if (detail >= DetailLevel::High)
    {
        result.m_Label += "CompilerDataFormat = " + ToString(m_CompilerDataFormat) + "\n";
        result.m_Label += "OutputTensorShape = " + ToString(m_OutputTensorShape) + "\n";
        result.m_Label += "OutputQuantizationInfo = " + ToString(m_OutputQuantizationInfo) + "\n";
        result.m_Label += "OutputDataType = " + ToString(m_OutputDataType) + "\n";
        result.m_Label += "ConstantData = [ " + std::to_string(m_ConstantData->size()) + " bytes ]\n";
    }
    return result;
}

}    // namespace support_library
}    // namespace ethosn
