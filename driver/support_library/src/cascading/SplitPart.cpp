//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "PartUtils.hpp"
#include "Plan.hpp"
#include "SplitPart.hpp"

namespace ethosn
{
namespace support_library
{

SplitPart::SplitPart(PartId id,
                     const TensorInfo& inputTensorInfo,
                     const SplitInfo& splitInfo,
                     const CompilerDataFormat& compilerDataFormat,
                     const std::set<uint32_t>& correspondingOperationIds,
                     const EstimationOptions& estOpt,
                     const CompilationOptions& compOpt,
                     const HardwareCapabilities& capabilities)
    : BasePart(id, "SplitPart", compilerDataFormat, correspondingOperationIds, estOpt, compOpt, capabilities)
    , m_InputTensorInfo{ inputTensorInfo }
    , m_SplitInfo{ splitInfo }
{}

SplitPart::~SplitPart()
{}

Plans SplitPart::GetPlans(CascadeType cascadeType,
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
        CreateSplitDramPlan(plans);
    }

    return plans;
}

void SplitPart::CreateSplitDramPlan(Plans& plans) const
{
    PartInputMapping inputMappings;
    PartOutputMapping outputMappings;
    OwnedOpGraph opGraph;

    CascadingBufferFormat format = impl::GetCascadingBufferFormatFromCompilerDataFormat(m_CompilerDataFormat);

    opGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, format, TraversalOrder::Xyz));
    Buffer* inputBuffer             = opGraph.GetBuffers()[0];
    inputBuffer->m_TensorShape      = m_InputTensorInfo.m_Dimensions;
    inputBuffer->m_SizeInBytes      = utils::CalculateBufferSize(inputBuffer->m_TensorShape, format);
    inputBuffer->m_QuantizationInfo = m_InputTensorInfo.m_QuantizationInfo;
    inputBuffer->m_BufferType       = BufferType::Intermediate;

    inputMappings[inputBuffer] = PartInputSlot{ m_PartId, 0 };

    std::vector<TensorInfo> expectedOutputInfo = Split::CalculateOutputTensorInfos(m_InputTensorInfo, m_SplitInfo);
    TensorShape offset                         = { 0, 0, 0, 0 };
    for (uint32_t outputIndex = 0; outputIndex < m_SplitInfo.m_Sizes.size(); ++outputIndex)
    {
        auto splitOp       = std::make_unique<SplitOp>(format, offset);
        Op* op             = opGraph.AddOp(std::move(splitOp));
        op->m_OperationIds = m_CorrespondingOperationIds;

        opGraph.AddConsumer(inputBuffer, op, 0);

        opGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, format, TraversalOrder::Xyz));
        Buffer* outputBuffer             = opGraph.GetBuffers().back();
        outputBuffer->m_TensorShape      = expectedOutputInfo[outputIndex].m_Dimensions;
        outputBuffer->m_SizeInBytes      = utils::CalculateBufferSize(outputBuffer->m_TensorShape, format);
        outputBuffer->m_QuantizationInfo = expectedOutputInfo[outputIndex].m_QuantizationInfo;
        outputBuffer->m_BufferType       = BufferType::Intermediate;
        opGraph.SetProducer(outputBuffer, op);
        outputMappings[outputBuffer] = PartOutputSlot{ m_PartId, outputIndex };

        // Calculate the offset of each output index
        offset[m_SplitInfo.m_Axis] = offset[m_SplitInfo.m_Axis] + m_SplitInfo.m_Sizes[outputIndex];
    }

    AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph), plans);
}

DotAttributes SplitPart::GetDotAttributes(DetailLevel detail) const
{
    DotAttributes result = BasePart::GetDotAttributes(detail);
    if (detail >= DetailLevel::High)
    {
        result.m_Label += "InputTensorsInfo.Dimensions = " + ToString(m_InputTensorInfo.m_Dimensions) + "\n";
        result.m_Label += "InputTensorsInfo.DataFormat = " + ToString(m_InputTensorInfo.m_DataFormat) + "\n";
        result.m_Label += "InputTensorsInfo.DataType = " + ToString(m_InputTensorInfo.m_DataType) + "\n";
        result.m_Label +=
            "InputTensorsInfo.QuantizationInfo = " + ToString(m_InputTensorInfo.m_QuantizationInfo) + "\n";
        result.m_Label += "SplitInfo.Axis = " + ToString(m_SplitInfo.m_Axis) + "\n";
        result.m_Label += "SplitInfo.Sizes = " + ArrayToString(m_SplitInfo.m_Sizes) + "\n";
    }
    return result;
}

}    // namespace support_library
}    // namespace ethosn
