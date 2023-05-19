//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "ConcatPart.hpp"

#include "../BufferManager.hpp"
#include "../Utils.hpp"
#include "ConcreteOperations.hpp"
#include "PartUtils.hpp"
#include "Plan.hpp"

namespace ethosn
{
namespace support_library
{

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
        CreateConcatDramPlans(plans);
    }

    return plans;
}

ConcatPart::~ConcatPart()
{}

void ConcatPart::CreateConcatDramPlans(Plans& plans) const
{
    const size_t numInputs = m_InputTensorsInfo.size();

    // Decide what format to use for the DRAM buffers.
    // Figure out if we need to use NHWC or if we can get away with NHWCB or FCAF (which should be more efficient).
    // We can use NHWCB/FCAF if the dimensions along the concat axis are all multiples of the brick group/cell size, so
    // that the DMA is capable of joining the tensors correctly from DRAM.
    bool canUseNhwcb      = true;
    const bool canUseNhwc = m_Axis != 3;    // DMA can't split along channels for NHWC
    bool canUseFcafDeep   = m_CompilationOptions.m_EnableIntermediateCompression;
    bool canUseFcafWide   = m_CompilationOptions.m_EnableIntermediateCompression;
    for (uint32_t i = 0; i < numInputs; ++i)
    {
        const uint32_t offset = m_Offsets[i];

        // Check compatibility with NHWCB
        if (offset % g_BrickGroupShape[m_Axis] != 0)
        {
            canUseNhwcb = false;
        }

        // Check compatibility with FCAF_DEEP
        if (offset % g_FcafDeepCellShape[m_Axis] != 0)
        {
            canUseFcafDeep = false;
        }

        // Check compatibility with FCAF_WIDE
        if (offset % g_FcafWideCellShape[m_Axis] != 0)
        {
            canUseFcafWide = false;
        }
    }

    // We prefer to use FCAF if possible, as it doesn't require chunking by the firmware and saves bandwidth
    // However, if all our inputs are likely to produce NHWC outputs, then it is probably better
    // to use NHWC, as it avoids the need for conversion.
    CascadingBufferFormat format;
    if (m_PreferNhwc && canUseNhwc)
    {
        format = CascadingBufferFormat::NHWC;
    }
    else if (canUseFcafDeep)
    {
        format = CascadingBufferFormat::FCAF_DEEP;
    }
    else if (canUseFcafWide)
    {
        format = CascadingBufferFormat::FCAF_WIDE;
    }
    else if (canUseNhwcb)
    {
        format = CascadingBufferFormat::NHWCB;
    }
    else if (canUseNhwc)
    {
        format = CascadingBufferFormat::NHWC;
    }
    else
    {
        // This shouldn't be possible, as all supported cases should be covered. However the logic is a bit tricky to
        // follow, so no harm in having this check.
        throw InternalErrorException("Unable to find a suitable format for Concat");
    }

    PartInputMapping inputMappings;
    PartOutputMapping outputMappings;
    OwnedOpGraph opGraph;

    std::unique_ptr<DramBuffer> outputBuffer = DramBuffer::Build()
                                                   .AddFormat(format)
                                                   .AddDataType(m_OutputTensorInfo.m_DataType)
                                                   .AddTensorShape(m_OutputTensorInfo.m_Dimensions)
                                                   .AddQuantization(m_OutputTensorInfo.m_QuantizationInfo)
                                                   .AddBufferType(BufferType::Intermediate);

    DramBuffer* outputBufferRaw     = opGraph.AddBuffer(std::move(outputBuffer));
    outputMappings[outputBufferRaw] = PartOutputSlot{ m_PartId, 0 };

    for (uint32_t inputIndex = 0; inputIndex < numInputs; inputIndex++)
    {
        TensorShape offset = { 0, 0, 0, 0 };
        offset[m_Axis]     = m_Offsets[inputIndex];

        std::unique_ptr<DramBuffer> inputBuffer =
            DramBuffer::Build()
                .AddFormat(format)
                .AddDataType(m_InputTensorsInfo[inputIndex].m_DataType)
                .AddTensorShape(m_InputTensorsInfo[inputIndex].m_Dimensions)
                .AddQuantization(m_InputTensorsInfo[inputIndex].m_QuantizationInfo)
                .AddBufferType(BufferType::Intermediate);

        DramBuffer* inputBufferRaw    = opGraph.AddBuffer(std::move(inputBuffer));
        inputMappings[inputBufferRaw] = PartInputSlot{ m_PartId, inputIndex };

        auto dma1            = std::make_unique<DmaOp>(format);
        dma1->m_OperationIds = m_CorrespondingOperationIds;
        DmaOp* dma1Raw       = dma1.get();
        opGraph.AddOp(std::move(dma1));

        // Create a buffer with the best stripe shape
        std::unique_ptr<SramBuffer> sramBuffer = impl::MakeGlueIntermediateSramBuffer(
            m_InputTensorsInfo[inputIndex].m_Dimensions, m_OutputTensorInfo.m_QuantizationInfo,
            m_OutputTensorInfo.m_DataType, { format }, m_Capabilities, m_StripeConfig.blockWidthMultiplier.min,
            m_StripeConfig.blockWidthMultiplier.max, m_StripeConfig.blockHeightMultiplier.min,
            m_StripeConfig.blockHeightMultiplier.max, m_StripeConfig.ofmDepthMultiplier.min,
            m_StripeConfig.ofmDepthMultiplier.max);
        SramBuffer* sramBufferRaw = sramBuffer.get();
        opGraph.AddBuffer(std::move(sramBuffer));

        auto dma2            = std::make_unique<DmaOp>(format);
        dma2->m_OperationIds = m_CorrespondingOperationIds;
        dma2->m_Offset       = offset;
        DmaOp* dma2Raw       = dma2.get();
        opGraph.AddOp(std::move(dma2));

        opGraph.AddConsumer(inputBufferRaw, dma1Raw, 0);
        opGraph.SetProducer(sramBufferRaw, dma1Raw);
        opGraph.AddConsumer(sramBufferRaw, dma2Raw, 0);
        opGraph.AddProducer(outputBufferRaw, dma2Raw);
    }

    // Note that we don't use AddNewPlan as the validation is wrong for SRAM (not all our buffers need to
    // be alive at the same time)
    Plan plan(std::move(inputMappings), std::move(outputMappings));
    plan.m_OpGraph = std::move(opGraph);
    // Prevent the Combiner from doing its own SRAM allocation for our SRAM buffers, as this makes pessimistic assumptions
    // about the lifetimes (that they must all be alive at the same time), which can lead to poor performance.
    plan.m_IsPreallocated = true;
    plans.push_back(std::move(plan));
}

ethosn::support_library::DotAttributes ConcatPart::GetDotAttributes(DetailLevel detail) const
{
    DotAttributes result = BasePart::GetDotAttributes(detail);
    if (detail >= DetailLevel::High)
    {
        result.m_Label += "PreferNhwc = " + ToString(m_PreferNhwc) + "\n";
        result.m_Label += "InputTensorsInfo = " + ArrayToString(m_InputTensorsInfo) + "\n";
        result.m_Label += "OutputTensorInfo = " + ToString(m_OutputTensorInfo) + "\n";
        result.m_Label += "Axis = " + ToString(m_Axis) + "\n";
        result.m_Label += "Offsets = " + ArrayToString(m_Offsets) + "\n";
    }
    return result;
}

const TensorShape& ConcatPart::GetOutputTensorShape() const
{
    return m_OutputTensorInfo.m_Dimensions;
}

const std::vector<uint32_t>& ConcatPart::GetOffsets() const
{
    return m_Offsets;
}

std::vector<BoundaryRequirements> ConcatPart::GetInputBoundaryRequirements() const
{
    // No boundary data required for any of our inputs
    return std::vector<BoundaryRequirements>(m_InputTensorsInfo.size(), BoundaryRequirements{});
}

std::vector<bool> ConcatPart::CanInputsTakePleInputSram() const
{
    // All our inputs need to be in DRAM
    return std::vector<bool>(m_InputTensorsInfo.size(), false);
}

}    // namespace support_library
}    // namespace ethosn
