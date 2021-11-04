//
// Copyright © 2021 Arm Limited.
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

    for (uint32_t inputIndex = 0; inputIndex < m_InputTensorsInfo.size(); inputIndex++)
    {
        opGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Dram, format, TraversalOrder::Xyz));
        Buffer* inputBuffer             = opGraph.GetBuffers()[inputIndex];
        inputBuffer->m_TensorShape      = m_InputTensorsInfo[inputIndex].m_Dimensions;
        inputBuffer->m_SizeInBytes      = impl::CalculateBufferSize(inputBuffer->m_TensorShape, format);
        inputBuffer->m_QuantizationInfo = m_InputTensorsInfo[inputIndex].m_QuantizationInfo;

        inputMappings[inputBuffer] = PartInputSlot{ m_PartId, inputIndex };
    }

    auto concatOp      = std::make_unique<ConcatOp>();
    Op* op             = opGraph.AddOp(std::move(concatOp));
    op->m_OperationIds = m_CorrespondingOperationIds;

    TensorInfo expectedOutputInfo = Concatenation::CalculateOutputTensorInfo(m_InputTensorsInfo, m_ConcatInfo);
    opGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Dram, format, TraversalOrder::Xyz));
    Buffer* outputBuffer             = opGraph.GetBuffers().back();
    outputBuffer->m_TensorShape      = expectedOutputInfo.m_Dimensions;
    outputBuffer->m_SizeInBytes      = impl::CalculateBufferSize(outputBuffer->m_TensorShape, format);
    outputBuffer->m_QuantizationInfo = expectedOutputInfo.m_QuantizationInfo;

    outputMappings[outputBuffer] = PartOutputSlot{ m_PartId, 0 };

    AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph), plans);
}

}    // namespace support_library
}    // namespace ethosn
