//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../Utils.hpp"
#include "ConcatPart.hpp"
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
    : BasePart(id, compilerDataFormat, correspondingOperationIds, estOpt, compOpt, capabilities)
    , m_InputTensorsInfo{ inputTensorsInfo }
    , m_ConcatInfo{ concatInfo }
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
        CreateConcatDramPlan(plans);
    }

    return plans;
}

ConcatPart::~ConcatPart()
{}

void ConcatPart::CreateConcatDramPlan(Plans& plans) const
{
    PartInputMapping inputMappings;
    PartOutputMapping outputMappings;
    OwnedOpGraph opGraph;

    CascadingBufferFormat format = impl::GetCascadingBufferFormatFromCompilerDataFormat(m_CompilerDataFormat);

    auto concatOp      = std::make_unique<ConcatOp>(format);
    Op* op             = opGraph.AddOp(std::move(concatOp));
    op->m_OperationIds = m_CorrespondingOperationIds;

    for (uint32_t inputIndex = 0; inputIndex < m_InputTensorsInfo.size(); inputIndex++)
    {
        opGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, format, TraversalOrder::Xyz));
        Buffer* inputBuffer             = opGraph.GetBuffers()[inputIndex];
        inputBuffer->m_TensorShape      = m_InputTensorsInfo[inputIndex].m_Dimensions;
        inputBuffer->m_SizeInBytes      = impl::CalculateBufferSize(inputBuffer->m_TensorShape, format);
        inputBuffer->m_QuantizationInfo = m_InputTensorsInfo[inputIndex].m_QuantizationInfo;
        inputBuffer->m_BufferType       = BufferType::Intermediate;
        opGraph.AddConsumer(inputBuffer, op, inputIndex);
        inputMappings[inputBuffer] = PartInputSlot{ m_PartId, inputIndex };
    }

    TensorInfo expectedOutputInfo = Concatenation::CalculateOutputTensorInfo(m_InputTensorsInfo, m_ConcatInfo);
    opGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, format, TraversalOrder::Xyz));
    Buffer* outputBuffer             = opGraph.GetBuffers().back();
    outputBuffer->m_TensorShape      = expectedOutputInfo.m_Dimensions;
    outputBuffer->m_SizeInBytes      = impl::CalculateBufferSize(outputBuffer->m_TensorShape, format);
    outputBuffer->m_QuantizationInfo = expectedOutputInfo.m_QuantizationInfo;
    outputBuffer->m_BufferType       = BufferType::Intermediate;
    opGraph.SetProducer(outputBuffer, op);
    outputMappings[outputBuffer] = PartOutputSlot{ m_PartId, 0 };

    AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph), plans);
}

ethosn::support_library::DotAttributes ConcatPart::GetDotAttributes(DetailLevel detail) const
{
    DotAttributes result = BasePart::GetDotAttributes(detail);
    result.m_Label       = "ConcatPart: " + result.m_Label;
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
