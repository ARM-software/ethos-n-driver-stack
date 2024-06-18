//
// Copyright © 2021-2024 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "FullyConnectedPart.hpp"
#include "PartUtils.hpp"

#include "BufferManager.hpp"

using namespace ethosn::command_stream;

namespace ethosn
{
namespace support_library
{
using namespace impl;
using namespace utils;

namespace
{

McePart::ConstructionParams ConvertConstructionParams(FullyConnectedPart::ConstructionParams&& fcParams)
{
    McePart::ConstructionParams mceParams(fcParams.m_EstOpt, fcParams.m_CompOpt, fcParams.m_Capabilities,
                                          fcParams.m_DebuggingContext, fcParams.m_ThreadPool);
    mceParams.m_Id = fcParams.m_Id;
    // Note the input shape as far as the McePart is concerned is the _reinterpreted_ input shape of the FC
    mceParams.m_InputTensorShape       = std::move(fcParams.m_ReinterpretedInputTensorShape);
    mceParams.m_OutputTensorShape      = std::move(fcParams.m_OutputTensorShape);
    mceParams.m_InputQuantizationInfo  = std::move(fcParams.m_InputQuantizationInfo);
    mceParams.m_OutputQuantizationInfo = std::move(fcParams.m_OutputQuantizationInfo);
    mceParams.m_WeightsInfo            = std::move(fcParams.m_WeightsInfo);
    mceParams.m_WeightsData            = std::move(fcParams.m_WeightsData);
    mceParams.m_BiasInfo               = std::move(fcParams.m_BiasInfo);
    mceParams.m_BiasData               = std::move(fcParams.m_BiasData);
    mceParams.m_Stride                 = { 1, 1 };
    mceParams.m_Padding                = Padding();
    mceParams.m_Op                     = command_stream::MceOperation::FULLY_CONNECTED;
    mceParams.m_OperationIds           = std::move(fcParams.m_OperationIds);
    mceParams.m_InputDataType          = std::move(fcParams.m_InputDataType);
    mceParams.m_OutputDataType         = std::move(fcParams.m_OutputDataType);
    mceParams.m_LowerBound             = mceParams.m_OutputDataType == DataType::UINT8_QUANTIZED ? 0 : -128;
    mceParams.m_UpperBound             = mceParams.m_OutputDataType == DataType::UINT8_QUANTIZED ? 255 : 127;
    return mceParams;
}

}    // namespace

FullyConnectedPart::FullyConnectedPart(ConstructionParams&& params)
    : McePart(ConvertConstructionParams(std::move(params)))
    , m_OriginalInputShape(params.m_InputTensorShape)
{}

Plans FullyConnectedPart::GetPlans(CascadeType cascadeType,
                                   BlockConfig blockConfig,
                                   const std::vector<Buffer*>& sramBufferInputs,
                                   uint32_t numWeightStripes) const
{
    ETHOSN_UNUSED(blockConfig);
    ETHOSN_UNUSED(sramBufferInputs);
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

StripeInfos FullyConnectedPart::GenerateStripeInfos() const
{
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

    return stripeInfos;
}

Plans FullyConnectedPart::GetLonelyPlans(uint32_t numWeightStripes) const
{
    Plans ret;

    StripeInfos stripeInfos = GenerateStripeInfos();

    // Fully connected input cannot be de-compressed from FCAF
    const bool couldSourceBeFcaf = false;

    for (const MceAndPleInfo& info : stripeInfos.m_MceAndPleInfos)
    {
        std::map<std::string, int> pleSelectionIntParams;
        pleSelectionIntParams["block_width"]  = info.m_PleCompute.m_BlockConfig.m_Width;
        pleSelectionIntParams["block_height"] = info.m_PleCompute.m_BlockConfig.m_Height;

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
                            .AddFormat(BufferFormat::NHWC)
                            .AddDataType(m_InputDataType)
                            .AddTensorShape(m_OriginalInputShape)
                            .AddQuantization(m_InputQuantizationInfo)
                            .AddBufferType(BufferType::Intermediate)
                            .AddSizeInBytes(utils::RoundUpToNearestMultiple(
                                utils::CalculateBufferSize(m_OriginalInputShape, BufferFormat::NHWC), 1024));

                    DramBuffer* dramInputRaw = opGraph.AddBuffer(std::move(dramInput));

                    // Use NHWCB specifically for Fully connected as the format in SRAM needs to be copied from an NHWC buffer byte by byte
                    Op* dmaOp             = opGraph.AddOp(std::make_unique<DmaOp>(BufferFormat::NHWCB));
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
                    std::unique_ptr<PleOp> pleOp = std::make_unique<PleOp>(
                        PleOperation::PASSTHROUGH, 1, std::vector<TensorShape>{ info.m_PleCompute.m_Input },
                        info.m_PleCompute.m_Output, true, m_Capabilities, std::map<std::string, std::string>{},
                        pleSelectionIntParams, std::map<std::string, int>{});
                    auto outBufferAndPleOp = AddPleToOpGraph(
                        opGraph, info.m_Memory.m_Output.m_Shape, numMemoryStripes, std::move(pleOp),
                        m_OutputTensorShape, m_OutputQuantizationInfo, m_OutputDataType, m_CorrespondingOperationIds);
                    opGraph.AddConsumer(pleInBuffer, outBufferAndPleOp.second, 0);
                    inputMappings[dramInputRaw]             = PartInputSlot{ m_PartId, 0 };
                    outputMappings[outBufferAndPleOp.first] = PartOutputSlot{ m_PartId, 0 };
                    AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph),
                               info.m_MceCompute.m_BlockConfig, ret);
                }
            }
        }
    }
    return ret;
}

void FullyConnectedPart::PreprocessWeightsAsync() const
{
    // Start encoding all the possible weight stripe and algorithm combinations that we might need later.

    WeightEncodingRequest request(m_Capabilities);
    request.m_WeightsTensorInfo      = m_WeightsInfo;
    request.m_WeightsData            = m_WeightsData;
    request.m_BiasTensorInfo         = m_BiasInfo;
    request.m_BiasData               = m_BiasData;
    request.m_InputQuantizationInfo  = m_InputQuantizationInfo;
    request.m_OutputQuantizationInfo = m_OutputQuantizationInfo;
    request.m_StripeDepth            = 0;
    request.m_StrideY                = m_Stride.m_Y;
    request.m_StrideX                = m_Stride.m_X;
    request.m_PaddingTop             = m_Padding.m_Top;
    request.m_PaddingLeft            = m_Padding.m_Left;
    request.m_IterationSize          = 0;
    request.m_Operation              = m_Operation;
    request.m_Algorithm              = CompilerMceAlgorithm::Direct;

    StripeInfos stripeInfos = GenerateStripeInfos();
    for (const MceAndPleInfo& i : stripeInfos.m_MceAndPleInfos)
    {
        WeightEncodingRequest modifiedRequest = request;
        modifiedRequest.m_StripeDepth         = GetWeightStripeDepth(m_WeightsInfo, i.m_MceCompute.m_Weight, m_Stride);
        modifiedRequest.m_IterationSize       = i.m_MceCompute.m_Weight[2];
        modifiedRequest.m_Algorithm = ResolveMceAlgorithm(i.m_MceCompute.m_BlockConfig, i.m_MceCompute.m_Weight[2]);

        m_WeightEncoderCache.EncodeStage1Async(std::move(modifiedRequest));
    }
}

}    // namespace support_library
}    // namespace ethosn
