//
// Copyright © 2021-2022 Arm Limited.
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
                       DataType dataType,
                       const std::set<uint32_t>& correspondingOperationIds,
                       const uint32_t producerOutputIndx,
                       const EstimationOptions& estOpt,
                       const CompilationOptions& compOpt,
                       const HardwareCapabilities& capabilities)
    : BasePart(id, "OutputPart", correspondingOperationIds, estOpt, compOpt, capabilities)
    , m_InputTensorShape{ inputTensorShape }
    , m_InputQuantizationInfo(quantizationInfo)
    , m_InputDataType(dataType)
    , m_ProducerOutputIndx{ producerOutputIndx }
    , m_CompilerDataFormat(compilerDataFormat)
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

    if (cascadeType == CascadeType::Lonely)
    {
        CreatePlanForOutputPart(TraversalOrder::Xyz, plans);
    }

    return plans;
}

OutputPart::~OutputPart()
{}

void OutputPart::CreatePlanForOutputPart(TraversalOrder order, Plans& plans) const
{
    PartInputMapping inputMappings;
    PartOutputMapping outputMappings;
    OwnedOpGraph opGraph;

    CascadingBufferFormat format   = impl::GetCascadingBufferFormatFromCompilerDataFormat(m_CompilerDataFormat);
    std::unique_ptr<Buffer> buffer = std::make_unique<Buffer>(Location::Dram, format, order);
    buffer->m_DataType             = m_InputDataType;
    buffer->m_TensorShape          = m_InputTensorShape;
    buffer->m_SizeInBytes          = utils::CalculateBufferSize(m_InputTensorShape, format);
    buffer->m_QuantizationInfo     = m_InputQuantizationInfo;
    buffer->m_OperationId          = *m_CorrespondingOperationIds.begin();
    buffer->m_ProducerOutputIndx   = m_ProducerOutputIndx;
    buffer->m_BufferType           = BufferType::Output;
    inputMappings[buffer.get()]    = PartInputSlot{ m_PartId, 0 };
    opGraph.AddBuffer(std::move(buffer));

    AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph), plans);
}

ethosn::support_library::DotAttributes OutputPart::GetDotAttributes(DetailLevel detail) const
{
    DotAttributes result = BasePart::GetDotAttributes(detail);
    if (detail >= DetailLevel::High)
    {
        result.m_Label += "CompilerDataFormat = " + ToString(m_CompilerDataFormat) + "\n";
        result.m_Label += "InputTensorShape = " + ToString(m_InputTensorShape) + "\n";
        result.m_Label += "InputQuantizationInfo = " + ToString(m_InputQuantizationInfo) + "\n";
        result.m_Label += "InputDataType = " + ToString(m_InputDataType) + "\n";
    }
    return result;
}

}    // namespace support_library
}    // namespace ethosn
