//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "SplitPart.hpp"

#include "PartUtils.hpp"
#include "Plan.hpp"

namespace ethosn
{
namespace support_library
{

SplitPart::SplitPart(PartId id,
                     const TensorInfo& inputTensorInfo,
                     const std::vector<TensorInfo>& outputTensorInfos,
                     uint32_t axis,
                     const std::vector<uint32_t>& offsets,
                     const std::set<uint32_t>& correspondingOperationIds,
                     const EstimationOptions& estOpt,
                     const CompilationOptions& compOpt,
                     const HardwareCapabilities& capabilities)
    : BasePart(id, "SplitPart", correspondingOperationIds, estOpt, compOpt, capabilities)
    , m_InputTensorInfo{ inputTensorInfo }
    , m_OutputTensorInfos(outputTensorInfos)
    , m_Axis(axis)
    , m_Offsets(offsets)
    , m_StripeConfig(impl::GetDefaultStripeConfig(compOpt, m_DebugTag.c_str()))
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
        CreateSplitDramPlans(plans);
    }

    return plans;
}

void SplitPart::CreateSplitDramPlans(Plans& plans) const
{
    const size_t numOutputs = m_OutputTensorInfos.size();

    // Decide what format to use for the DRAM buffers.
    // Figure out if we need to use NHWC or if we can get away with NHWCB or FCAF (which should be more efficient).
    // We can use NHWCB/FCAF if the dimensions along the split axis are all multiples of the brick group/cell size, so
    // that the DMA is capable of extracting the tensors correctly from DRAM.
    bool canUseNhwcb      = true;
    const bool canUseNhwc = m_Axis != 3;    // DMA can't split along channels for NHWC
    bool canUseFcafDeep   = m_CompilationOptions.m_EnableIntermediateCompression;
    bool canUseFcafWide   = m_CompilationOptions.m_EnableIntermediateCompression;
    for (uint32_t i = 0; i < numOutputs; ++i)
    {
        const uint32_t offset = m_Offsets[i];

        // Check compatibility with NHWCB
        if (offset % m_Capabilities.GetBrickGroupShape()[m_Axis] != 0)
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
    CascadingBufferFormat format;
    if (canUseFcafDeep)
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
        throw InternalErrorException("Unable to find a suitable format for Split");
    }

    PartInputMapping inputMappings;
    PartOutputMapping outputMappings;
    OwnedOpGraph opGraph;

    opGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, format, TraversalOrder::Xyz));
    Buffer* inputBuffer             = opGraph.GetBuffers()[0];
    inputBuffer->m_DataType         = m_InputTensorInfo.m_DataType;
    inputBuffer->m_TensorShape      = m_InputTensorInfo.m_Dimensions;
    inputBuffer->m_SizeInBytes      = utils::CalculateBufferSize(inputBuffer->m_TensorShape, format);
    inputBuffer->m_QuantizationInfo = m_InputTensorInfo.m_QuantizationInfo;
    inputBuffer->m_BufferType       = BufferType::Intermediate;

    inputMappings[inputBuffer] = PartInputSlot{ m_PartId, 0 };

    for (uint32_t outputIndex = 0; outputIndex < numOutputs; outputIndex++)
    {
        const TensorInfo& outputTensorInfo = m_OutputTensorInfos[outputIndex];

        TensorShape offset = { 0, 0, 0, 0 };
        offset[m_Axis]     = m_Offsets[outputIndex];

        auto dma1            = std::make_unique<DmaOp>(format);
        dma1->m_OperationIds = m_CorrespondingOperationIds;
        dma1->m_Offset       = offset;
        DmaOp* dma1Raw       = dma1.get();
        opGraph.AddOp(std::move(dma1));

        // Create a buffer with the best stripe shape
        std::unique_ptr<Buffer> sramBuffer = impl::MakeGlueIntermediateSramBuffer(
            outputTensorInfo.m_Dimensions, outputTensorInfo.m_QuantizationInfo, outputTensorInfo.m_DataType, { format },
            m_Capabilities, m_StripeConfig.blockWidthMultiplier.min, m_StripeConfig.blockWidthMultiplier.max,
            m_StripeConfig.blockHeightMultiplier.min, m_StripeConfig.blockHeightMultiplier.max,
            m_StripeConfig.ofmDepthMultiplier.min, m_StripeConfig.ofmDepthMultiplier.max);
        Buffer* sramBufferRaw = sramBuffer.get();
        opGraph.AddBuffer(std::move(sramBuffer));

        auto dma2            = std::make_unique<DmaOp>(format);
        dma2->m_OperationIds = m_CorrespondingOperationIds;
        DmaOp* dma2Raw       = dma2.get();
        opGraph.AddOp(std::move(dma2));

        opGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, format, TraversalOrder::Xyz));
        Buffer* outputBuffer             = opGraph.GetBuffers().back();
        outputBuffer->m_DataType         = outputTensorInfo.m_DataType;
        outputBuffer->m_TensorShape      = outputTensorInfo.m_Dimensions;
        outputBuffer->m_SizeInBytes      = utils::CalculateBufferSize(outputBuffer->m_TensorShape, format);
        outputBuffer->m_QuantizationInfo = outputTensorInfo.m_QuantizationInfo;
        outputBuffer->m_BufferType       = BufferType::Intermediate;
        outputMappings[outputBuffer]     = PartOutputSlot{ m_PartId, outputIndex };

        opGraph.AddConsumer(inputBuffer, dma1Raw, 0);
        opGraph.SetProducer(sramBufferRaw, dma1Raw);
        opGraph.AddConsumer(sramBufferRaw, dma2Raw, 0);
        opGraph.AddProducer(outputBuffer, dma2Raw);
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

DotAttributes SplitPart::GetDotAttributes(DetailLevel detail) const
{
    DotAttributes result = BasePart::GetDotAttributes(detail);
    if (detail >= DetailLevel::High)
    {
        result.m_Label += "InputTensorInfo = " + ToString(m_InputTensorInfo) + "\n";
        result.m_Label += "OutputTensorInfos = " + ArrayToString(m_OutputTensorInfos) + "\n";
        result.m_Label += "Axis = " + ToString(m_Axis) + "\n";
        result.m_Label += "Offsets = " + ArrayToString(m_Offsets) + "\n";
    }
    return result;
}

const TensorShape& SplitPart::GetInputTensorShape() const
{
    return m_InputTensorInfo.m_Dimensions;
}

const std::vector<uint32_t>& SplitPart::GetOffsets() const
{
    return m_Offsets;
}

}    // namespace support_library
}    // namespace ethosn
