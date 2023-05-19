//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "FullyConnectedPart.hpp"
#include "PartUtils.hpp"

#include "../BufferManager.hpp"

using namespace ethosn::command_stream;

namespace ethosn
{
namespace support_library
{
using namespace impl;
using namespace utils;

Plans FullyConnectedPart::GetPlans(CascadeType cascadeType,
                                   BlockConfig blockConfig,
                                   Buffer* sramBuffer,
                                   uint32_t numWeightStripes) const
{
    ETHOSN_UNUSED(blockConfig);
    ETHOSN_UNUSED(sramBuffer);
    // Only Lonely plans are supported at the moment as fully connected layers
    // are rare and usually very large. This means the likelihood they can be
    // cascaded is reduced and their impact on performance is small.
    if (cascadeType == CascadeType::Lonely)
    {
        return GetLonelyPlans(numWeightStripes);
    }
    else
    {
        return Plans{};
    }
}

utils::Optional<MceOperation> FullyConnectedPart::GetMceOperation() const
{
    return ethosn::command_stream::MceOperation::FULLY_CONNECTED;
}

Plans FullyConnectedPart::GetLonelyPlans(uint32_t numWeightStripes) const
{
    Plans ret;

    // Fully connected only supports 8x8 block configs
    const BlockConfig blockConfig                   = { 8u, 8u };
    PackedBoundaryThickness packedBoundaryThickness = { 0, 0, 0, 0 };
    const uint32_t numWeightLoads                   = 1;

    StripeInfos stripeInfos;
    // Full IFM and Full OFM
    if (m_StripeConfig.splits.none)
    {
        TensorShape mceInputEncoding = { 0, 0, 0, 0 };
        TensorShape mceInputStripe   = CreateStripe(m_InputTensorShape, mceInputEncoding, g_BrickGroupShape[3]);

        TensorShape mceOutputEncoding = { 0, 0, 0, 0 };
        TensorShape mceOutputStripe =
            CreateStripe(m_OutputTensorShape, mceOutputEncoding, m_Capabilities.GetNumberOfOgs());

        TensorShape weightStripe = { 1, 1, GetNumElements(mceInputStripe), GetChannels(mceOutputStripe) };

        NumStripes numStripesInput   = { 1, 1 };
        NumStripes numStripesWeights = { 1, 1 };
        NumStripes numStripesOutput  = { 1, 1 };

        const uint32_t numIfmLoads = 1;

        MceAndPleInfo mceAndPleInfo;

        mceAndPleInfo.m_MceCompute.m_Input       = mceInputStripe;
        mceAndPleInfo.m_MceCompute.m_Output      = mceOutputStripe;
        mceAndPleInfo.m_MceCompute.m_Weight      = weightStripe;
        mceAndPleInfo.m_MceCompute.m_BlockConfig = blockConfig;
        mceAndPleInfo.m_PleCompute.m_Input       = mceOutputStripe;
        mceAndPleInfo.m_PleCompute.m_Output      = mceOutputStripe;
        mceAndPleInfo.m_PleCompute.m_BlockConfig = blockConfig;

        mceAndPleInfo.m_Memory.m_Input  = { { numStripesInput, mceInputStripe }, packedBoundaryThickness, numIfmLoads };
        mceAndPleInfo.m_Memory.m_Output = { numStripesOutput, mceOutputStripe };
        mceAndPleInfo.m_Memory.m_Weight = { { numStripesWeights, weightStripe }, numWeightLoads };
        mceAndPleInfo.m_Memory.m_PleInput = { { 0, 0 }, mceOutputStripe };
        stripeInfos.m_MceAndPleInfos.emplace(mceAndPleInfo);
    }
    // Full IFM and partial OFM
    if (m_StripeConfig.splits.mceAndPleOutputDepth)
    {
        // Exclusive loop as we already have a no-split plan above
        for (uint32_t ofmDepth :
             StripeShapeLoop::Exclusive(GetChannels(m_OutputTensorShape), m_Capabilities.GetNumberOfOgs(),
                                        m_StripeConfig.ofmDepthMultiplier.min, m_StripeConfig.ofmDepthMultiplier.max))
        {
            TensorShape mceInputEncoding = { 0, 0, 0, 0 };
            TensorShape mceInputStripe   = CreateStripe(m_InputTensorShape, mceInputEncoding, g_BrickGroupShape[3]);

            TensorShape mceOutputEncoding = { 0, 0, 0, ofmDepth };
            TensorShape mceOutputStripe =
                CreateStripe(m_OutputTensorShape, mceOutputEncoding, m_Capabilities.GetNumberOfOgs());

            TensorShape weightStripe = { 1, 1, GetNumElements(mceInputStripe), GetChannels(mceOutputStripe) };

            NumStripes numStripesInput   = { 1, 1 };
            NumStripes numStripesWeights = { 1, 2 };

            uint32_t maxNumStripesOutput = GetChannels(m_OutputTensorShape) > GetChannels(mceOutputStripe) ? 2 : 1;
            NumStripes numStripesOutput  = { 1, maxNumStripesOutput };

            const uint32_t numIfmLoads = 1;

            MceAndPleInfo mceAndPleInfo;

            mceAndPleInfo.m_MceCompute.m_Input       = mceInputStripe;
            mceAndPleInfo.m_MceCompute.m_Output      = mceOutputStripe;
            mceAndPleInfo.m_MceCompute.m_Weight      = weightStripe;
            mceAndPleInfo.m_MceCompute.m_BlockConfig = blockConfig;
            mceAndPleInfo.m_PleCompute.m_Input       = mceOutputStripe;
            mceAndPleInfo.m_PleCompute.m_Output      = mceOutputStripe;
            mceAndPleInfo.m_PleCompute.m_BlockConfig = blockConfig;

            mceAndPleInfo.m_Memory.m_Input    = { { numStripesInput, mceInputStripe },
                                               packedBoundaryThickness,
                                               numIfmLoads };
            mceAndPleInfo.m_Memory.m_Output   = { numStripesOutput, mceOutputStripe };
            mceAndPleInfo.m_Memory.m_Weight   = { { numStripesWeights, weightStripe }, numWeightLoads };
            mceAndPleInfo.m_Memory.m_PleInput = { { 0, 0 }, mceOutputStripe };
            stripeInfos.m_MceAndPleInfos.emplace(mceAndPleInfo);
        }
    }

    // Partial IFM and partial OFM
    if (m_StripeConfig.splits.outputDepthInputDepth)
    {
        // Exclusive loop as we already have a no-split plan above
        for (uint32_t ifmDepth :
             StripeShapeLoop::Exclusive(GetChannels(m_InputTensorShape),
                                        m_Capabilities.GetIgsPerEngine() * m_Capabilities.GetNumberOfEngines(),
                                        m_StripeConfig.ifmDepthMultiplier.min, m_StripeConfig.ifmDepthMultiplier.max))
        {
            TensorShape mceInputEncoding = { 0, 0, 0, ifmDepth };
            TensorShape mceInputStripe   = CreateStripe(m_InputTensorShape, mceInputEncoding, g_BrickGroupShape[3]);

            TensorShape mceOutputEncoding = { 0, 0, 0, m_Capabilities.GetNumberOfOgs() };
            TensorShape mceOutputStripe =
                CreateStripe(m_OutputTensorShape, mceOutputEncoding, m_Capabilities.GetNumberOfOgs());

            TensorShape weightStripe = { 1, 1, GetNumElements(mceInputStripe), GetChannels(mceOutputStripe) };

            uint32_t maxNumInputStripes = GetChannels(m_InputTensorShape) > GetChannels(mceInputStripe) ? 2 : 1;
            NumStripes numStripesInput  = { 1, maxNumInputStripes };

            uint32_t maxNumStripesOutput = GetChannels(m_OutputTensorShape) > GetChannels(mceOutputStripe) ? 2 : 1;
            NumStripes numStripesOutput  = { 1, maxNumStripesOutput };

            NumStripes numStripesWeights = { 1, 1 };

            const uint32_t numIfmLoads =
                utils::DivRoundUp(GetChannels(m_OutputTensorShape), GetChannels(mceOutputStripe));

            MceAndPleInfo mceAndPleInfo;

            mceAndPleInfo.m_MceCompute.m_Input       = mceInputStripe;
            mceAndPleInfo.m_MceCompute.m_Output      = mceOutputStripe;
            mceAndPleInfo.m_MceCompute.m_Weight      = weightStripe;
            mceAndPleInfo.m_MceCompute.m_BlockConfig = blockConfig;
            mceAndPleInfo.m_PleCompute.m_Input       = mceOutputStripe;
            mceAndPleInfo.m_PleCompute.m_Output      = mceOutputStripe;
            mceAndPleInfo.m_PleCompute.m_BlockConfig = blockConfig;

            mceAndPleInfo.m_Memory.m_Input    = { { numStripesInput, mceInputStripe },
                                               packedBoundaryThickness,
                                               numIfmLoads };
            mceAndPleInfo.m_Memory.m_Output   = { numStripesOutput, mceOutputStripe };
            mceAndPleInfo.m_Memory.m_Weight   = { { numStripesWeights, weightStripe }, numWeightLoads };
            mceAndPleInfo.m_Memory.m_PleInput = { { 0, 0 }, mceOutputStripe };
            stripeInfos.m_MceAndPleInfos.emplace(mceAndPleInfo);
        }
    }

    // Fully connected input cannot be de-compressed from FCAF
    const bool couldSourceBeFcaf = false;

    for (const MceAndPleInfo& info : stripeInfos.m_MceAndPleInfos)
    {
        for (auto numInputStripes = info.m_Memory.m_Input.m_Range.m_Min;
             numInputStripes <= info.m_Memory.m_Input.m_Range.m_Max; ++numInputStripes)
        {
            for (auto numOutputStripes = info.m_Memory.m_Output.m_Range.m_Min;
                 numOutputStripes <= info.m_Memory.m_Output.m_Range.m_Max; ++numOutputStripes)
            {
                for (auto numPleInputStripes = info.m_Memory.m_PleInput.m_Range.m_Min;
                     numPleInputStripes <= info.m_Memory.m_PleInput.m_Range.m_Max; ++numPleInputStripes)
                {
                    NumMemoryStripes numMemoryStripes;
                    numMemoryStripes.m_Input    = numInputStripes;
                    numMemoryStripes.m_Output   = numOutputStripes;
                    numMemoryStripes.m_Weight   = numWeightStripes;
                    numMemoryStripes.m_PleInput = numPleInputStripes;
                    OwnedOpGraph opGraph;
                    PartInputMapping inputMappings;
                    PartOutputMapping outputMappings;
                    impl::ConvData convData;
                    convData.weightInfo = m_WeightsInfo;
                    convData.weightData = m_WeightsData;
                    convData.biasInfo   = m_BiasInfo;
                    convData.biasData   = m_BiasData;

                    // The input buffer size of fully connected must be rounded up to the next 1024.
                    std::unique_ptr<DramBuffer> dramInput =
                        DramBuffer::Build()
                            .AddFormat(CascadingBufferFormat::NHWC)
                            .AddDataType(m_InputDataType)
                            .AddTensorShape(m_OriginalInputShape)
                            .AddQuantization(m_InputQuantizationInfo)
                            .AddBufferType(BufferType::Intermediate)
                            .AddSizeInBytes(utils::RoundUpToNearestMultiple(
                                utils::CalculateBufferSize(m_OriginalInputShape, CascadingBufferFormat::NHWC), 1024));

                    DramBuffer* dramInputRaw = opGraph.AddBuffer(std::move(dramInput));

                    // Use NHWCB specifically for Fully connected as the format in SRAM needs to be copied from an NHWC buffer byte by byte
                    Op* dmaOp             = opGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB));
                    dmaOp->m_OperationIds = m_CorrespondingOperationIds;

                    auto sramInputAndMceOp =
                        AddMceToOpGraph(opGraph, info.m_MceCompute, info.m_Memory, numMemoryStripes, m_InputTensorShape,
                                        m_InputQuantizationInfo, convData, m_WeightEncoderCache, couldSourceBeFcaf);
                    if (!sramInputAndMceOp.first || !sramInputAndMceOp.second)
                    {
                        continue;    // Weight compression failed (too big for SRAM) - abandon this plan
                    }

                    opGraph.AddConsumer(dramInputRaw, dmaOp, 0);
                    opGraph.SetProducer(sramInputAndMceOp.first, dmaOp);

                    auto pleInBuffer = impl::AddPleInputSramBuffer(opGraph, numPleInputStripes, m_OutputTensorShape,
                                                                   info.m_Memory.m_PleInput.m_Shape,
                                                                   m_OutputQuantizationInfo, m_OutputDataType);
                    opGraph.SetProducer(pleInBuffer, sramInputAndMceOp.second);

                    // Create an identity ple Op
                    std::unique_ptr<PleOp> pleOp =
                        std::make_unique<PleOp>(PleOperation::PASSTHROUGH, info.m_MceCompute.m_BlockConfig, 1,
                                                std::vector<TensorShape>{ info.m_PleCompute.m_Input },
                                                info.m_PleCompute.m_Output, m_OutputDataType, true);
                    auto outBufferAndPleOp = AddPleToOpGraph(
                        opGraph, info.m_Memory.m_Output.m_Shape, numMemoryStripes, std::move(pleOp),
                        m_OutputTensorShape, m_OutputQuantizationInfo, m_OutputDataType, m_CorrespondingOperationIds);
                    opGraph.AddConsumer(pleInBuffer, outBufferAndPleOp.second, 0);
                    inputMappings[dramInputRaw]             = PartInputSlot{ m_PartId, 0 };
                    outputMappings[outBufferAndPleOp.first] = PartOutputSlot{ m_PartId, 0 };
                    AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph), ret);
                }
            }
        }
    }
    return ret;
}

}    // namespace support_library
}    // namespace ethosn
