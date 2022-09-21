//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "FullyConnectedPart.hpp"
#include "PartUtils.hpp"

using namespace ethosn::command_stream;

namespace ethosn
{
namespace support_library
{
using namespace impl;
using namespace utils;

FullyConnectedPart::FullyConnectedPart(PartId id,
                                       const TensorShape& inputTensorShape,
                                       const TensorShape& reinterpretedInputShape,
                                       const TensorShape& outputTensorShape,
                                       const QuantizationInfo& inputQuantizationInfo,
                                       const QuantizationInfo& outputQuantizationInfo,
                                       const TensorInfo& weightsInfo,
                                       std::vector<uint8_t> weightsData,
                                       const TensorInfo& biasInfo,
                                       std::vector<int32_t> biasData,
                                       const EstimationOptions& estOpt,
                                       const CompilationOptions& compOpt,
                                       const HardwareCapabilities& capabilities,
                                       std::set<uint32_t> operationIds,
                                       DataType inputDataType,
                                       DataType outputDataType)
    : McePart(id,
              reinterpretedInputShape,
              outputTensorShape,
              inputQuantizationInfo,
              outputQuantizationInfo,
              weightsInfo,
              weightsData,
              biasInfo,
              biasData,
              Stride{},
              0,
              0,
              command_stream::MceOperation::FULLY_CONNECTED,
              estOpt,
              compOpt,
              capabilities,
              operationIds,
              inputDataType,
              outputDataType)
    , m_OriginalInputShape(inputTensorShape)
{}

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
    const BlockConfig blockConfig                                              = { 8u, 8u };
    command_stream::cascading::PackedBoundaryThickness packedBoundaryThickness = { 0, 0, 0, 0 };
    const uint32_t numIfmLoads                                                 = 1;
    const uint32_t numWeightLoads                                              = 1;

    StripeInfos stripeInfos;
    // Full IFM and Full OFM
    if (m_StripeConfig.splits.none)
    {
        TensorShape mceInputEncoding = { 0, 0, 0, 0 };
        TensorShape mceInputStripe =
            CreateStripe(m_InputTensorShape, mceInputEncoding, m_Capabilities.GetBrickGroupShape()[3]);

        TensorShape mceOutputEncoding = { 0, 0, 0, 0 };
        TensorShape mceOutputStripe =
            CreateStripe(m_OutputTensorShape, mceOutputEncoding, m_Capabilities.GetNumberOfOgs());

        TensorShape weightStripe = { 1, 1, GetNumElements(mceInputStripe), GetChannels(mceOutputStripe) };

        NumStripes numStripesInput   = { 1, 1 };
        NumStripes numStripesWeights = { 1, 1 };
        NumStripes numStripesOutput  = { 1, 1 };

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
        const uint32_t minOfmDepthMultiplier = std::max(1U, m_StripeConfig.ofmDepthMultiplier.min);
        const uint32_t maxOfmDepthMultiplier =
            std::max(1U, std::min(utils::DivRoundUp(GetChannels(m_OutputTensorShape), m_Capabilities.GetNumberOfOgs()),
                                  m_StripeConfig.ofmDepthMultiplier.max));
        for (uint32_t ofmDepthMultiplier = minOfmDepthMultiplier; ofmDepthMultiplier <= maxOfmDepthMultiplier;
             ofmDepthMultiplier *= 2)
        {
            TensorShape mceInputEncoding = { 0, 0, 0, 0 };
            TensorShape mceInputStripe =
                CreateStripe(m_InputTensorShape, mceInputEncoding, m_Capabilities.GetBrickGroupShape()[3]);

            TensorShape mceOutputEncoding = { 0, 0, 0, m_Capabilities.GetNumberOfOgs() * ofmDepthMultiplier };
            TensorShape mceOutputStripe =
                CreateStripe(m_OutputTensorShape, mceOutputEncoding, m_Capabilities.GetNumberOfOgs());

            TensorShape weightStripe = { 1, 1, GetNumElements(mceInputStripe), GetChannels(mceOutputStripe) };

            NumStripes numStripesInput   = { 1, 1 };
            NumStripes numStripesWeights = { 1, 2 };

            uint32_t maxNumStripesOutput = GetChannels(m_OutputTensorShape) > GetChannels(mceOutputStripe) ? 2 : 1;
            NumStripes numStripesOutput  = { 1, maxNumStripesOutput };

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
        const uint32_t minIfmDepthMultiplier = std::max(1U, m_StripeConfig.ifmDepthMultiplier.min);
        const uint32_t maxIfmDepthMultiplier = std::max(
            1U, std::min(utils::DivRoundUp(GetChannels(m_InputTensorShape),
                                           (m_Capabilities.GetIgsPerEngine() * m_Capabilities.GetNumberOfEngines())),
                         m_StripeConfig.ifmDepthMultiplier.max));
        for (uint32_t ifmDepthMultiplier = minIfmDepthMultiplier; ifmDepthMultiplier <= maxIfmDepthMultiplier;
             ifmDepthMultiplier *= 2)
        {
            TensorShape mceInputEncoding = {
                0, 0, 0, m_Capabilities.GetIgsPerEngine() * m_Capabilities.GetNumberOfEngines() * ifmDepthMultiplier
            };
            TensorShape mceInputStripe =
                CreateStripe(m_InputTensorShape, mceInputEncoding, m_Capabilities.GetBrickGroupShape()[3]);

            TensorShape mceOutputEncoding = { 0, 0, 0, m_Capabilities.GetNumberOfOgs() };
            TensorShape mceOutputStripe =
                CreateStripe(m_OutputTensorShape, mceOutputEncoding, m_Capabilities.GetNumberOfOgs());

            TensorShape weightStripe = { 1, 1, GetNumElements(mceInputStripe), GetChannels(mceOutputStripe) };

            uint32_t maxNumInputStripes = GetChannels(m_InputTensorShape) > GetChannels(mceInputStripe) ? 2 : 1;
            NumStripes numStripesInput  = { 1, maxNumInputStripes };

            uint32_t maxNumStripesOutput = GetChannels(m_OutputTensorShape) > GetChannels(mceOutputStripe) ? 2 : 1;
            NumStripes numStripesOutput  = { 1, maxNumStripesOutput };

            NumStripes numStripesWeights = { 1, 1 };

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

                    opGraph.AddBuffer(
                        std::make_unique<Buffer>(Location::Dram, CascadingBufferFormat::NHWC, TraversalOrder::Xyz));
                    Buffer* dramInput        = opGraph.GetBuffers().back();
                    dramInput->m_DataType    = m_InputDataType;
                    dramInput->m_TensorShape = m_OriginalInputShape;
                    // The input buffer size of fully connected must be rounded up to the next 1024.
                    dramInput->m_SizeInBytes = utils::RoundUpToNearestMultiple(
                        utils::CalculateBufferSize(m_OriginalInputShape, CascadingBufferFormat::NHWC), 1024);
                    dramInput->m_QuantizationInfo = m_InputQuantizationInfo;
                    dramInput->m_BufferType       = BufferType::Intermediate;

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

                    opGraph.AddConsumer(dramInput, dmaOp, 0);
                    opGraph.SetProducer(sramInputAndMceOp.first, dmaOp);

                    auto pleInBuffer = impl::AddPleInBuffer(opGraph, numPleInputStripes, m_OutputTensorShape,
                                                            info.m_Memory.m_PleInput.m_Shape, m_OutputQuantizationInfo,
                                                            m_OutputDataType, Location::PleInputSram);
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
                    inputMappings[dramInput]                = PartInputSlot{ m_PartId, 0 };
                    outputMappings[outBufferAndPleOp.first] = PartOutputSlot{ m_PartId, 0 };
                    AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph), ret, false,
                               true);
                }
            }
        }
    }
    return ret;
}

}    // namespace support_library
}    // namespace ethosn
