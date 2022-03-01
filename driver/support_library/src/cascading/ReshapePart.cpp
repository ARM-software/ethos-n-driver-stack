//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "PartUtils.hpp"
#include "Plan.hpp"
#include "ReshapePart.hpp"

namespace ethosn
{
namespace support_library
{

ReshapePart::ReshapePart(PartId id,
                         const TensorShape& inputTensorShape,
                         const TensorShape& outputTensorShape,
                         const CompilerDataFormat& compilerDataFormat,
                         const QuantizationInfo& quantizationInfo,
                         const std::set<uint32_t>& correspondingOperationIds,
                         const EstimationOptions& estOpt,
                         const CompilationOptions& compOpt,
                         const HardwareCapabilities& capabilities)
    : BasePart(id, compilerDataFormat, correspondingOperationIds, estOpt, compOpt, capabilities)
    , m_InputTensorShape{ inputTensorShape }
    , m_OutputTensorShape{ outputTensorShape }
    , m_OutputQuantizationInfo(quantizationInfo)
{}

Plans ReshapePart::GetPlans(CascadeType cascadeType,
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
        CreateReinterpretDramPlan(plans);
    }

    return plans;
}

ReshapePart::~ReshapePart()
{}

/// Creates a plan which simply reinterprets the input tensor properties of the given node with its output tensor
/// properties. No Ops are created - just a single Dram buffer which is tagged as both the input and output of the Plan.
void ReshapePart::CreateReinterpretDramPlan(Plans& plans) const
{
    CascadingBufferFormat format = impl::GetCascadingBufferFormatFromCompilerDataFormat(m_CompilerDataFormat);
    PartInputMapping inputMappings;
    PartOutputMapping outputMappings;
    OwnedOpGraph opGraph;
    opGraph.AddBuffer(std::make_unique<Buffer>(Lifetime::Atomic, Location::Dram, format, TraversalOrder::Xyz));
    Buffer* buffer             = opGraph.GetBuffers()[0];
    buffer->m_TensorShape      = m_OutputTensorShape;
    buffer->m_SizeInBytes      = impl::CalculateBufferSize(m_InputTensorShape, format);
    buffer->m_QuantizationInfo = m_OutputQuantizationInfo;
    buffer->m_BufferType       = BufferType::Intermediate;

    inputMappings[buffer]  = PartInputSlot{ m_PartId, 0 };
    outputMappings[buffer] = PartOutputSlot{ m_PartId, 0 };
    AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph), plans);
}

ethosn::support_library::DotAttributes ReshapePart::GetDotAttributes(DetailLevel detail) const
{
    DotAttributes result = BasePart::GetDotAttributes(detail);
    result.m_Label       = "ReshapePart: " + result.m_Label;
    if (detail >= DetailLevel::High)
    {
        result.m_Label += "InputTensorShape = " + ToString(m_InputTensorShape) + "\n";
        result.m_Label += "OutputTensorShape = " + ToString(m_OutputTensorShape) + "\n";
        result.m_Label += "OutputQuantizationInfo = " + ToString(m_OutputQuantizationInfo) + "\n";
    }
    return result;
}

}    // namespace support_library
}    // namespace ethosn
