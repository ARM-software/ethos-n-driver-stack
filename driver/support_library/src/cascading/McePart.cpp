//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../BufferManager.hpp"
#include "McePart.hpp"
#include "PartUtils.hpp"
#include "Plan.hpp"
#include "StripeHelper.hpp"

#include <algorithm>
#include <ethosn_utils/Macros.hpp>

#include <utility>

using namespace ethosn::command_stream;

namespace ethosn
{
namespace support_library
{
using namespace impl;
using namespace utils;

namespace
{

struct NumStripesGrouped
{
    NumStripes m_Input;
    NumStripes m_Output;
    NumStripes m_Weights;
    NumStripes m_PleInput;
};

utils::Optional<std::pair<MceAndPleInfo, MceOnlyInfo>>
    GenerateContinueSectionStripeInfos(NumStripesGrouped& numStripes,
                                       const SramBuffer* sramBuffer,
                                       uint32_t numWeightStripes,
                                       bool isDepthwise,
                                       const HardwareCapabilities& caps,
                                       const TensorShape& outputTensorShape,
                                       uint32_t kernelHeight,
                                       uint32_t kernelWidth,
                                       uint32_t strideMultiplier,
                                       const ShapeMultiplier& shapeMultiplier,
                                       const BlockConfig& blockConfig,
                                       CascadeType cascadeType,
                                       const StripeConfig& stripeConfig,
                                       BoundaryRequirements outputBoundaryRequirements)
{
    assert(cascadeType == CascadeType::Middle || cascadeType == CascadeType::End);
    const TensorShape& mceInputStripe = sramBuffer->m_StripeShape;
    bool fullHeight                   = GetHeight(sramBuffer->m_StripeShape) >= GetHeight(sramBuffer->m_TensorShape);
    bool fullWidth                    = GetWidth(sramBuffer->m_StripeShape) >= GetWidth(sramBuffer->m_TensorShape);
    bool fullTensor                   = fullHeight && fullWidth;
    bool fullInputDepth = GetChannels(sramBuffer->m_StripeShape) >= GetChannels(sramBuffer->m_TensorShape);

    if (!isDepthwise && !fullInputDepth)
    {
        return {};
    }

    TensorShape mceOutputEncoding = { 0, 0, 0, 0 };
    if (fullTensor)
    {
        if (isDepthwise)
        {
            mceOutputEncoding = { 0, 0, 0, GetChannels(mceInputStripe) / strideMultiplier };
        }
        else
        {
            // If we have the full plane, the number of weight stripes is 1 and we're in the middle of a cascade we can use the full tensor
            // But if we're at the end of a cascade we can split the output depth so we get more parallelism.
            if (numWeightStripes == 1 && cascadeType == CascadeType::Middle)
            {
                // strategy 3
                if (stripeConfig.splits.none)
                {
                    mceOutputEncoding = { 0, 0, 0, 0 };
                }
                else
                {
                    return {};
                }
            }
            else
            {
                // strategy 1
                if (stripeConfig.splits.mceAndPleOutputDepth || stripeConfig.splits.mceOutputDepthOnly)
                {
                    mceOutputEncoding = { 0, 0, 0, caps.GetNumberOfOgs() };
                }
                else
                {
                    return {};
                }
            }
        }
    }
    else
    {
        // Splitting width or height
        if (stripeConfig.splits.mceOutputHeightOnly || stripeConfig.splits.mceAndPleOutputHeight ||
            stripeConfig.splits.widthOnly || stripeConfig.splits.widthHeight)
        {
            mceOutputEncoding = { 0, fullHeight ? 0 : (GetHeight(mceInputStripe) * shapeMultiplier.m_H),
                                  fullWidth ? 0 : (GetWidth(mceInputStripe) * shapeMultiplier.m_W), 0 };
        }
        else
        {
            return {};
        }
    }
    TensorShape mceOutputStripe = impl::CreateStripe(outputTensorShape, mceOutputEncoding, g_BrickGroupShape[3]);

    TensorShape pleInputEncoding = mceOutputEncoding;
    if (cascadeType == CascadeType::Middle)
    {
        // PLE accumulates the full depth for the middle of an s1 cascade
        pleInputEncoding[3] = 0;
    }
    TensorShape pleInputStripe  = impl::CreateStripe(outputTensorShape, pleInputEncoding, g_BrickGroupShape[3]);
    TensorShape pleOutputStripe = pleInputStripe;    // PLE kernel is passthrough

    uint32_t mceWeightOutputStripe = mceOutputStripe[3];
    bool fullOutputDepth           = mceWeightOutputStripe >= GetChannels(outputTensorShape);
    if (fullOutputDepth && numWeightStripes != 1)
    {
        return {};
    }
    TensorShape mceWeightStripe;
    if (isDepthwise)
    {
        mceWeightStripe = TensorShape{ kernelHeight, kernelWidth, mceWeightOutputStripe * strideMultiplier, 1 };
    }
    else
    {
        mceWeightStripe = TensorShape{ kernelHeight, kernelWidth, mceInputStripe[3], mceWeightOutputStripe };
    }
    TensorShape memoryWeightStripe = mceWeightStripe;

    // Even if the MCE output is split in depth, we build up a full-depth tensor in SRAM for the next
    // layer of the cascade (if there is one)
    uint32_t memoryOutputChannelsEncoding = cascadeType == CascadeType::End ? GetChannels(mceOutputStripe) : 0;
    TensorShape memoryOutputStripeEncoding{ 0, fullHeight ? 0 : GetHeight(mceOutputStripe),
                                            fullWidth ? 0 : GetWidth(mceOutputStripe), memoryOutputChannelsEncoding };
    TensorShape memoryOutputStripe = CreateStripe(outputTensorShape, memoryOutputStripeEncoding, g_BrickGroupShape[3]);

    bool fullDepth      = memoryOutputStripe[3] >= outputTensorShape[3];
    bool isEndOfCascade = cascadeType == CascadeType::End;
    // strategy 0
    if (!fullTensor)
    {
        // if its the end of a cascade we can double buffer the output, if it's not we need to output up to 3 stripes for neighouring data.
        if (isEndOfCascade)
        {
            numStripes.m_Output = { 1, 2 };
        }
        else
        {
            if ((outputBoundaryRequirements.m_NeedsBeforeX || outputBoundaryRequirements.m_NeedsBeforeY) &&
                (outputBoundaryRequirements.m_NeedsAfterX || outputBoundaryRequirements.m_NeedsAfterY))
            {
                numStripes.m_Output = { 3, 3 };
            }
            else if (outputBoundaryRequirements.m_NeedsBeforeX || outputBoundaryRequirements.m_NeedsBeforeY ||
                     outputBoundaryRequirements.m_NeedsAfterX || outputBoundaryRequirements.m_NeedsAfterY)
            {
                numStripes.m_Output = { 2, 2 };
            }
            else
            {
                numStripes.m_Output = { 1, 1 };
            }
        }
    }
    // Strategy 1/3
    else if (isEndOfCascade && fullDepth)
    {
        assert(fullTensor);
        numStripes.m_Output = { 1, 1 };
    }
    else if (!isEndOfCascade)
    {
        numStripes.m_Output = { 1, 1 };
    }
    else if (!fullDepth)
    {
        numStripes.m_Output = { 1, 2 };
    }

    PackedBoundaryThickness packedBoundaryThickness = { 0, 0, 0, 0 };
    const uint32_t numIfmLoads                      = 1;
    const uint32_t numWeightLoads                   = 1;

    MceAndPleInfo mceAndPleInfo;

    mceAndPleInfo.m_MceCompute.m_Input       = mceInputStripe;
    mceAndPleInfo.m_MceCompute.m_Output      = mceOutputStripe;
    mceAndPleInfo.m_MceCompute.m_Weight      = mceWeightStripe;
    mceAndPleInfo.m_MceCompute.m_BlockConfig = blockConfig;
    mceAndPleInfo.m_PleCompute.m_Input       = pleInputStripe;
    mceAndPleInfo.m_PleCompute.m_Output      = pleOutputStripe;
    mceAndPleInfo.m_PleCompute.m_BlockConfig = blockConfig;

    mceAndPleInfo.m_Memory.m_Input  = { { numStripes.m_Input, mceInputStripe }, packedBoundaryThickness, numIfmLoads };
    mceAndPleInfo.m_Memory.m_Output = { numStripes.m_Output, memoryOutputStripe };
    mceAndPleInfo.m_Memory.m_Weight = { { numStripes.m_Weights, memoryWeightStripe }, numWeightLoads };
    mceAndPleInfo.m_Memory.m_PleInput = { numStripes.m_PleInput, mceOutputStripe };

    MceOnlyInfo mceOnlyInfo;

    mceOnlyInfo.m_MceCompute.m_Input       = mceInputStripe;
    mceOnlyInfo.m_MceCompute.m_Output      = mceOutputStripe;
    mceOnlyInfo.m_MceCompute.m_Weight      = mceWeightStripe;
    mceOnlyInfo.m_MceCompute.m_BlockConfig = blockConfig;

    mceOnlyInfo.m_Memory.m_Input    = { { numStripes.m_Input, mceInputStripe }, packedBoundaryThickness, numIfmLoads };
    mceOnlyInfo.m_Memory.m_Output   = { { 0, 0 }, { 0, 0, 0, 0 } };
    mceOnlyInfo.m_Memory.m_Weight   = { { numStripes.m_Weights, memoryWeightStripe }, numWeightLoads };
    mceOnlyInfo.m_Memory.m_PleInput = { numStripes.m_PleInput, mceOutputStripe };

    return std::make_pair(mceAndPleInfo, mceOnlyInfo);
}

}    // namespace

McePart::McePart(ConstructionParams&& params)
    : BasePart(params.m_Id,
               "McePart",
               std::move(params.m_OperationIds),
               params.m_EstOpt,
               params.m_CompOpt,
               params.m_Capabilities)
    , m_InputTensorShape(params.m_InputTensorShape)
    , m_OutputTensorShape(params.m_OutputTensorShape)
    , m_WeightEncoderCache{ params.m_Capabilities }
    , m_InputQuantizationInfo(params.m_InputQuantizationInfo)
    , m_OutputQuantizationInfo(params.m_OutputQuantizationInfo)
    , m_WeightsInfo(params.m_WeightsInfo)
    , m_WeightsData(std::make_shared<std::vector<uint8_t>>(std::move(params.m_WeightsData)))
    , m_BiasInfo(params.m_BiasInfo)
    , m_BiasData(std::move(params.m_BiasData))
    , m_Stride(params.m_Stride)
    , m_UpscaleFactor(params.m_UpscaleFactor)
    , m_UpsampleType(params.m_UpsampleType)
    , m_PadTop(params.m_PadTop)
    , m_PadLeft(params.m_PadLeft)
    , m_Operation(params.m_Op)
    , m_StripeConfig(GetDefaultStripeConfig(params.m_CompOpt, m_DebugTag.c_str()))
    , m_StripeGenerator(m_InputTensorShape,
                        m_OutputTensorShape,
                        m_OutputTensorShape,
                        m_WeightsInfo.m_Dimensions[0],
                        m_WeightsInfo.m_Dimensions[1],
                        m_PadTop,
                        m_PadLeft,
                        m_UpscaleFactor,
                        params.m_Op,
                        PleOperation::PASSTHROUGH,
                        ShapeMultiplier{ m_UpscaleFactor, m_UpscaleFactor, 1 },
                        ShapeMultiplier::Identity,
                        params.m_Capabilities,
                        m_StripeConfig)
    , m_InputDataType(params.m_InputDataType)
    , m_OutputDataType(params.m_OutputDataType)
    , m_LowerBound(params.m_LowerBound)
    , m_UpperBound(params.m_UpperBound)
    , m_IsChannelSelector(params.m_IsChannelSelector)
{}

Buffer* McePart::AddWeightBuffersAndDmaOpToMceOp(OwnedOpGraph& opGraph,
                                                 const impl::MceStripesInfo& mceComputeInfo,
                                                 const impl::NumStripesType& numMemoryWeightStripes,
                                                 const TensorShape& memoryWeightStripe,
                                                 uint32_t numLoads,
                                                 const impl::ConvData& convData,
                                                 WeightEncoderCache& weightEncoderCache,
                                                 CompilerMceAlgorithm mceOpAlgo) const
{
    // Encode weights
    const uint32_t weightStripeSize  = mceComputeInfo.m_Weight[2];
    const uint32_t weightStripeDepth = GetWeightStripeDepth(convData.weightInfo, mceComputeInfo.m_Weight, m_Stride);

    WeightEncodingRequest wp(m_Capabilities);
    wp.m_WeightsTensorInfo      = convData.weightInfo;
    wp.m_WeightsData            = convData.weightData;
    wp.m_BiasTensorInfo         = convData.biasInfo;
    wp.m_BiasData               = convData.biasData;
    wp.m_InputQuantizationInfo  = m_InputQuantizationInfo;
    wp.m_OutputQuantizationInfo = m_OutputQuantizationInfo;
    wp.m_StripeDepth            = weightStripeDepth;
    wp.m_StrideY                = m_Stride.m_Y;
    wp.m_StrideX                = m_Stride.m_X;
    wp.m_PaddingTop             = m_PadTop;
    wp.m_PaddingLeft            = m_PadLeft;
    wp.m_IterationSize          = weightStripeSize;
    wp.m_Operation              = m_Operation;
    wp.m_Algorithm              = mceOpAlgo;
    auto encodedWeights         = weightEncoderCache.Encode(std::move(wp));
    if (!encodedWeights)
    {
        return nullptr;    // Weight compression failed (too big for SRAM) - abandon this plan
    }

    auto weightShape = convData.weightInfo.m_Dimensions;
    weightShape[2]   = GetNumSubmapChannels(weightShape[2], m_Stride.m_X, m_Stride.m_Y, m_Capabilities);

    CascadingBufferFormat formatInSram = GetCascadingBufferFormatFromCompilerDataFormat(CompilerDataFormat::WEIGHT);

    std::unique_ptr<SramBuffer> sramWeightBuffer = SramBufferBuilder()
                                                       .AddFormat(formatInSram)
                                                       .AddDataType(convData.weightInfo.m_DataType)
                                                       .AddTensorShape(weightShape)
                                                       .AddQuantization(convData.weightInfo.m_QuantizationInfo)
                                                       .AddStripeShape(memoryWeightStripe)
                                                       .AddNumStripes(numMemoryWeightStripes)
                                                       .AddNumLoads(numLoads)
                                                       .AddSlotSize(encodedWeights->m_MaxSize)
                                                       .AddTraversalOrder(TraversalOrder::Xyz);

    CascadingBufferFormat formatInDram = impl::GetCascadingBufferFormatFromCompilerDataFormat(
        ConvertExternalToCompilerDataFormat(convData.weightInfo.m_DataFormat));

    std::unique_ptr<DramBuffer> dramWeightBuffer = DramBuffer::Build()
                                                       .AddFormat(formatInDram)
                                                       .AddDataType(convData.weightInfo.m_DataType)
                                                       .AddTensorShape(weightShape)
                                                       .AddQuantization(convData.weightInfo.m_QuantizationInfo)
                                                       .AddBufferType(BufferType::ConstantDma)
                                                       .AddEncodedWeights(std::move(encodedWeights));

    DramBuffer* dramWeightBufferRaw = opGraph.AddBuffer(std::move(dramWeightBuffer));
    SramBuffer* sramWeightBufferRaw = opGraph.AddBuffer(std::move(sramWeightBuffer));

    Op* dmaOp             = opGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::WEIGHT));
    dmaOp->m_OperationIds = m_CorrespondingOperationIds;

    opGraph.AddConsumer(dramWeightBufferRaw, dmaOp, 0);
    opGraph.SetProducer(sramWeightBufferRaw, dmaOp);

    // Use the encoded weights to determine the size of the sram and dram buffers
    return sramWeightBufferRaw;
}

CompilerMceAlgorithm McePart::ResolveMceAlgorithm(const ethosn::command_stream::BlockConfig& blockConfig,
                                                  uint32_t inputStripeChannels) const
{
    uint32_t kernelHeight   = m_WeightsInfo.m_Dimensions[0];
    uint32_t kernelWidth    = m_WeightsInfo.m_Dimensions[1];
    const bool isWinograd2d = (kernelHeight > 1) && (kernelWidth > 1);

    CompilerMceAlgorithm effectiveAlgo = CompilerMceAlgorithm::Direct;
    // Winograd and upscaling cannot be performed at the same time
    if (!m_CompilationOptions.m_DisableWinograd && m_Operation == command_stream::MceOperation::CONVOLUTION &&
        m_Stride == Stride{ 1, 1 } && m_UpsampleType == MceUpsampleType::OFF)
    {
        effectiveAlgo =
            utils::FindBestConvAlgorithm(m_Capabilities, m_WeightsInfo.m_Dimensions[0], m_WeightsInfo.m_Dimensions[1]);
    }

    std::vector<command_stream::BlockConfig> blockConfigs =
        FilterAlgoBlockConfigs(effectiveAlgo, isWinograd2d, { blockConfig }, m_Capabilities);

    CompilerMceAlgorithm mceOpAlgo = blockConfigs.empty() ? CompilerMceAlgorithm::Direct : effectiveAlgo;

    // Encoder doesn't support multiple iterations with Winograd enabled
    if (inputStripeChannels < m_WeightsInfo.m_Dimensions[2])
    {
        mceOpAlgo = CompilerMceAlgorithm::Direct;
    }

    return mceOpAlgo;
}

std::pair<Buffer*, Op*> McePart::AddMceToOpGraph(OwnedOpGraph& opGraph,
                                                 const impl::MceStripesInfo& mceStripeInfo,
                                                 const impl::MemoryStripesInfo& memoryStripesInfo,
                                                 impl::NumMemoryStripes& numMemoryStripes,
                                                 const TensorShape& inputShape,
                                                 const QuantizationInfo& inputQuantInfo,
                                                 impl::ConvData& convData,
                                                 WeightEncoderCache& weightEncoderCache,
                                                 bool couldSourceBeFcaf) const
{
    const CompilerMceAlgorithm mceOpAlgo = ResolveMceAlgorithm(mceStripeInfo.m_BlockConfig, mceStripeInfo.m_Weight[2]);

    const TraversalOrder ifmTraversalOrder =
        m_Operation == command_stream::MceOperation::DEPTHWISE_CONVOLUTION ? TraversalOrder::Xyz : TraversalOrder::Zxy;

    TileSizeCalculation tile = CalculateTileSize(m_Capabilities, inputShape, memoryStripesInfo.m_Input.m_Shape,
                                                 memoryStripesInfo.m_Input.m_PackedBoundaryThickness,
                                                 numMemoryStripes.m_Input, couldSourceBeFcaf);

    std::unique_ptr<SramBuffer> sramInBuffer =
        SramBufferBuilder()
            .AddFormat(CascadingBufferFormat::NHWCB)
            .AddDataType(m_InputDataType)
            .AddTensorShape(inputShape)
            .AddQuantization(inputQuantInfo)
            .AddStripeShape(memoryStripesInfo.m_Input.m_Shape)
            .AddNumStripes(numMemoryStripes.m_Input)
            .AddNumLoads(memoryStripesInfo.m_Input.m_NumLoads)
            .AddPackedBoundaryThickness(memoryStripesInfo.m_Input.m_PackedBoundaryThickness)
            .AddTraversalOrder(ifmTraversalOrder)
            .AddFromTileSize(tile);

    SramBuffer* sramInBufferRaw = opGraph.AddBuffer(std::move(sramInBuffer));

    Buffer* sramWeightBuffer = AddWeightBuffersAndDmaOpToMceOp(
        opGraph, mceStripeInfo, numMemoryStripes.m_Weight, memoryStripesInfo.m_Weight.m_Shape,
        memoryStripesInfo.m_Weight.m_NumLoads, convData, weightEncoderCache, mceOpAlgo);
    if (!sramWeightBuffer)
    {
        return { nullptr, nullptr };    // Weight compression failed (too big for SRAM) - abandon this plan
    }

    auto mceOp =
        std::make_unique<MceOp>(m_Operation, mceOpAlgo, mceStripeInfo.m_BlockConfig, mceStripeInfo.m_Input,
                                mceStripeInfo.m_Output, memoryStripesInfo.m_Weight.m_Shape, TraversalOrder::Xyz,
                                m_Stride, m_PadLeft, m_PadTop, m_LowerBound, m_UpperBound);
    mceOp->m_UpscaleFactor = m_UpscaleFactor;
    mceOp->m_UpsampleType  = m_UpsampleType;
    if (m_UninterleavedInputShape.has_value())
    {
        mceOp->m_uninterleavedInputShape = m_UninterleavedInputShape;
    }
    Op* op             = opGraph.AddOp(std::move(mceOp));
    op->m_OperationIds = m_CorrespondingOperationIds;
    opGraph.AddConsumer(sramInBufferRaw, op, 0);
    opGraph.AddConsumer(sramWeightBuffer, op, 1);

    return { sramInBufferRaw, op };
};

void McePart::CreateMceAndIdentityPlePlans(const impl::MceAndPleInfo& info,
                                           WeightEncoderCache& weightEncoderCache,
                                           Plans& plans,
                                           uint32_t numWeightStripes,
                                           bool couldSourceBeFcaf) const
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
                auto inBufferAndMceOp =
                    AddMceToOpGraph(opGraph, info.m_MceCompute, info.m_Memory, numMemoryStripes, m_InputTensorShape,
                                    m_InputQuantizationInfo, convData, weightEncoderCache, couldSourceBeFcaf);
                if (!inBufferAndMceOp.first || !inBufferAndMceOp.second)
                {
                    continue;    // Weight compression failed (too big for SRAM) - abandon this plan
                }

                auto pleInBuffer = impl::AddPleInputSramBuffer(opGraph, numPleInputStripes, m_OutputTensorShape,
                                                               info.m_Memory.m_PleInput.m_Shape,
                                                               m_OutputQuantizationInfo, m_OutputDataType);
                opGraph.SetProducer(pleInBuffer, inBufferAndMceOp.second);

                // Create an identity ple Op
                std::unique_ptr<PleOp> pleOp =
                    std::make_unique<PleOp>(PleOperation::PASSTHROUGH, info.m_MceCompute.m_BlockConfig, 1,
                                            std::vector<TensorShape>{ info.m_PleCompute.m_Input },
                                            info.m_PleCompute.m_Output, m_OutputDataType, true);
                auto outBufferAndPleOp = AddPleToOpGraph(
                    opGraph, info.m_Memory.m_Output.m_Shape, numMemoryStripes, std::move(pleOp), m_OutputTensorShape,
                    m_OutputQuantizationInfo, m_OutputDataType, m_CorrespondingOperationIds);
                opGraph.AddConsumer(pleInBuffer, outBufferAndPleOp.second, 0);
                inputMappings[inBufferAndMceOp.first]   = PartInputSlot{ m_PartId, 0 };
                outputMappings[outBufferAndPleOp.first] = PartOutputSlot{ m_PartId, 0 };
                AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph),
                           info.m_MceCompute.m_BlockConfig, plans);
            }
        }
    }
}

void McePart::CreateMceOnlyPlans(const impl::MceOnlyInfo& info,
                                 WeightEncoderCache& weightEncoderCache,
                                 Plans& plans,
                                 uint32_t numWeightStripes,
                                 bool couldSourceBeFcaf) const
{
    for (auto numInputStripes = info.m_Memory.m_Input.m_Range.m_Min;
         numInputStripes <= info.m_Memory.m_Input.m_Range.m_Max; ++numInputStripes)
    {
        for (auto numPleInputStripes = info.m_Memory.m_PleInput.m_Range.m_Min;
             numPleInputStripes <= info.m_Memory.m_PleInput.m_Range.m_Max; ++numPleInputStripes)
        {
            NumMemoryStripes numMemoryStripes;
            numMemoryStripes.m_Input    = numInputStripes;
            numMemoryStripes.m_Output   = 0;
            numMemoryStripes.m_Weight   = numWeightStripes;
            numMemoryStripes.m_PleInput = numPleInputStripes;
            OwnedOpGraph opGraph;
            PartInputMapping inputMappings;
            PartOutputMapping outputMappings;
            ConvData convData;
            convData.weightInfo = m_WeightsInfo;
            convData.weightData = m_WeightsData;
            convData.biasInfo   = m_BiasInfo;
            convData.biasData   = m_BiasData;
            auto inBufferAndMceOp =
                AddMceToOpGraph(opGraph, info.m_MceCompute, info.m_Memory, numMemoryStripes, m_InputTensorShape,
                                m_InputQuantizationInfo, convData, weightEncoderCache, couldSourceBeFcaf);
            if (!inBufferAndMceOp.first || !inBufferAndMceOp.second)
            {
                continue;    // Weight compression failed (too big for SRAM) - abandon this plan
            }

            // We need to add the output buffer first before adding mce to opgraph as it uses it.

            auto outBuffer = impl::AddPleInputSramBuffer(opGraph, numPleInputStripes, m_OutputTensorShape,
                                                         info.m_Memory.m_PleInput.m_Shape, m_OutputQuantizationInfo,
                                                         m_OutputDataType);
            opGraph.SetProducer(outBuffer, inBufferAndMceOp.second);
            inputMappings[inBufferAndMceOp.first] = PartInputSlot{ m_PartId, 0 };
            outputMappings[outBuffer]             = PartOutputSlot{ m_PartId, 0 };
            AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph),
                       info.m_MceCompute.m_BlockConfig, plans);
        }
    }
}

Plans McePart::GetLonelyPlans(uint32_t numWeightStripes) const
{
    Plans ret;

    if (!m_StripeConfig.planTypes.lonely)
    {
        return ret;
    }

    // Data could be de-compressed from FCAF
    const bool couldSourceBeFcaf = true;

    // Start by generating "high priority" plans. If any of these work, there is no point generating
    // any low priority plans as this will just waste time (e.g. weight encoding)
    const std::initializer_list<PlanPriority> allPriorities = { PlanPriority::High, PlanPriority::Low };
    for (PlanPriority priority : allPriorities)
    {
        StripeInfos stripeInfos =
            m_StripeGenerator.GenerateStripes(CascadeType::Lonely, m_OutputBoundaryRequirements.at(0), priority);
        for (const MceAndPleInfo& i : stripeInfos.m_MceAndPleInfos)
        {
            CreateMceAndIdentityPlePlans(i, m_WeightEncoderCache, ret, numWeightStripes, couldSourceBeFcaf);
        }
        if (!ret.empty())
        {
            break;
        }
    }

    return ret;
}

Plans McePart::GetBeginningPlans(uint32_t numWeightStripes) const
{
    Plans ret;

    if (!m_StripeConfig.planTypes.beginning)
    {
        return ret;
    }

    StripeInfos stripeInfos =
        m_StripeGenerator.GenerateStripes(CascadeType::Beginning, m_OutputBoundaryRequirements.at(0), {});

    // The plan will be "glued" to the end plan from the previous section.
    // Therefore the input buffer tile cannot be unconditionally clamped to the
    // tensor size since input data (from DRAM) could be FCAF compressed and the
    // HW always writes to the SRAM in full cell size. Clamping is only allowed
    // if the stripe shape is not multiple of FCAF cell size.
    const bool couldSourceBeFcaf = true;

    if (!m_OutputCanTakePleInputSram.at(0))
    {
        for (const MceAndPleInfo& i : stripeInfos.m_MceAndPleInfos)
        {
            CreateMceAndIdentityPlePlans(i, m_WeightEncoderCache, ret, numWeightStripes, couldSourceBeFcaf);
        }
    }
    else
    {
        for (const MceOnlyInfo& i : stripeInfos.m_MceOnlyInfos)
        {
            CreateMceOnlyPlans(i, m_WeightEncoderCache, ret, numWeightStripes, couldSourceBeFcaf);
        }
    }

    return ret;
}

Plans McePart::GetMiddlePlans(ethosn::command_stream::BlockConfig blockConfig,
                              const SramBuffer* sramBuffer,
                              uint32_t numWeightStripes) const
{
    assert(sramBuffer);
    Plans ret;

    if (!m_StripeConfig.planTypes.middle)
    {
        return ret;
    }

    uint32_t kernelHeight = m_WeightsInfo.m_Dimensions[0];
    uint32_t kernelWidth  = m_WeightsInfo.m_Dimensions[1];

    uint32_t strideMultiplier = m_Stride.m_X * m_Stride.m_Y;

    NumStripesGrouped numStripes;
    numStripes.m_Input    = { sramBuffer->m_NumStripes, sramBuffer->m_NumStripes };
    numStripes.m_Weights  = { numWeightStripes, numWeightStripes };
    numStripes.m_PleInput = { 0, 0 };

    bool isDepthwise = m_Operation == ethosn::command_stream::MceOperation::DEPTHWISE_CONVOLUTION;
    auto stripeInfos = GenerateContinueSectionStripeInfos(
        numStripes, sramBuffer, numWeightStripes, isDepthwise, m_Capabilities, m_OutputTensorShape, kernelHeight,
        kernelWidth, strideMultiplier, m_StripeGenerator.m_MceShapeMultiplier, blockConfig, CascadeType::Middle,
        m_StripeConfig, m_OutputBoundaryRequirements.at(0));

    if (!stripeInfos.has_value())
    {
        return ret;
    }

    // Data in the input buffer (SRAM) cannot be FCAF de-compressed.
    // Hence input tile is allowed to be clamped to tensor size.
    const bool couldSourceBeFcaf = false;

    if (!m_OutputCanTakePleInputSram.at(0))
    {
        CreateMceAndIdentityPlePlans(stripeInfos.value().first, m_WeightEncoderCache, ret, numWeightStripes,
                                     couldSourceBeFcaf);
    }
    else
    {
        CreateMceOnlyPlans(stripeInfos.value().second, m_WeightEncoderCache, ret, numWeightStripes, couldSourceBeFcaf);
    }
    return ret;
}

Plans McePart::GetEndPlans(ethosn::command_stream::BlockConfig blockConfig,
                           const SramBuffer* sramBuffer,
                           uint32_t numWeightStripes) const
{
    assert(sramBuffer);
    Plans ret;

    if (!m_StripeConfig.planTypes.end)
    {
        return ret;
    }

    uint32_t kernelHeight = m_WeightsInfo.m_Dimensions[0];
    uint32_t kernelWidth  = m_WeightsInfo.m_Dimensions[1];

    uint32_t strideMultiplier = m_Stride.m_X * m_Stride.m_Y;

    NumStripesGrouped numStripes;
    numStripes.m_Input    = { sramBuffer->m_NumStripes, sramBuffer->m_NumStripes };
    numStripes.m_Output   = { 1, 2 };
    numStripes.m_Weights  = { numWeightStripes, numWeightStripes };
    numStripes.m_PleInput = { 0, 0 };

    bool isDepthwise = m_Operation == ethosn::command_stream::MceOperation::DEPTHWISE_CONVOLUTION;
    auto stripeInfos = GenerateContinueSectionStripeInfos(
        numStripes, sramBuffer, numWeightStripes, isDepthwise, m_Capabilities, m_OutputTensorShape, kernelHeight,
        kernelWidth, strideMultiplier, m_StripeGenerator.m_MceShapeMultiplier, blockConfig, CascadeType::End,
        m_StripeConfig, m_OutputBoundaryRequirements.at(0));

    if (!stripeInfos.has_value())
    {
        return ret;
    }

    // Data in the input buffer (SRAM) cannot be FCAF de-compressed.
    // Hence input tile is allowed to be clamped to tensor size.
    const bool couldSourceBeFcaf = false;

    CreateMceAndIdentityPlePlans(stripeInfos.value().first, m_WeightEncoderCache, ret, numWeightStripes,
                                 couldSourceBeFcaf);

    return ret;
}

Plans McePart::GetPlans(CascadeType cascadeType,
                        ethosn::command_stream::BlockConfig blockConfig,
                        const std::vector<Buffer*>& sramBufferInputs,
                        uint32_t numWeightStripes) const
{
    switch (cascadeType)
    {
        case CascadeType::Lonely:
        {
            return GetLonelyPlans(numWeightStripes);
        }
        case CascadeType::Beginning:
        {
            return GetBeginningPlans(numWeightStripes);
        }
        case CascadeType::Middle:
        {
            Buffer* sramBuffer = sramBufferInputs[0];
            if (sramBuffer->m_Location != Location::Sram)
            {
                return Plans();
            }
            return GetMiddlePlans(blockConfig, sramBuffer->Sram(), numWeightStripes);
        }
        case CascadeType::End:
        {
            Buffer* sramBuffer = sramBufferInputs[0];
            if (sramBuffer->m_Location != Location::Sram)
            {
                return Plans();
            }
            return GetEndPlans(blockConfig, sramBuffer->Sram(), numWeightStripes);
        }
        default:
        {
            ETHOSN_FAIL_MSG("Invalid cascade type");
            return Plans();
        }
    }
}

utils::Optional<ethosn::command_stream::MceOperation> McePart::GetMceOperation() const
{
    return m_Operation;
}

bool McePart::HasActivationBounds() const
{
    return true;
}

bool McePart::CanDoubleBufferWeights() const
{
    return true;
}

void McePart::ApplyActivationBounds(int16_t lowerBound, int16_t upperBound)
{
    m_LowerBound = std::max(m_LowerBound, lowerBound);
    m_UpperBound = std::min(m_UpperBound, upperBound);
}

ethosn::support_library::DotAttributes McePart::GetDotAttributes(DetailLevel detail) const
{
    DotAttributes result = BasePart::GetDotAttributes(detail);
    if (detail >= DetailLevel::High)
    {
        result.m_Label += "InputTensorShape = " + ToString(m_InputTensorShape) + "\n";
        result.m_Label += "OutputTensorShape = " + ToString(m_OutputTensorShape) + "\n";
        result.m_Label += "InputQuantizationInfo = " + ToString(m_InputQuantizationInfo) + "\n";
        result.m_Label += "OutputQuantizationInfo = " + ToString(m_OutputQuantizationInfo) + "\n";
        result.m_Label += "InputDataType = " + ToString(m_InputDataType) + "\n";
        result.m_Label += "OutputDataType = " + ToString(m_OutputDataType) + "\n";
        result.m_Label += "WeightsInfo = " + ToString(m_WeightsInfo) + "\n";
        result.m_Label += "BiasInfo = " + ToString(m_BiasInfo) + "\n";
        result.m_Label += "Stride = " + ToString(m_Stride) + "\n";
        result.m_Label += "UpscaleFactor = " + ToString(m_UpscaleFactor) + "\n";
        result.m_Label += "UpsampleType = " + ToString(m_UpsampleType) + "\n";
        result.m_Label += "PadTop = " + ToString(m_PadTop) + "\n";
        result.m_Label += "PadLeft = " + ToString(m_PadLeft) + "\n";
        result.m_Label += "Operation = " + ToString(m_Operation) + "\n";

        result.m_Label +=
            "StripeGenerator.MceInputTensorShape = " + ToString(m_StripeGenerator.m_MceInputTensorShape) + "\n";
        result.m_Label +=
            "StripeGenerator.MceOutputTensorShape = " + ToString(m_StripeGenerator.m_MceOutputTensorShape) + "\n";
        result.m_Label +=
            "StripeGenerator.PleOutputTensorShape = " + ToString(m_StripeGenerator.m_PleOutputTensorShape) + "\n";
        result.m_Label += "StripeGenerator.KernelHeight = " + ToString(m_StripeGenerator.m_KernelHeight) + "\n";
        result.m_Label += "StripeGenerator.KernelWidth = " + ToString(m_StripeGenerator.m_KernelWidth) + "\n";
        result.m_Label += "StripeGenerator.UpscaleFactor = " + ToString(m_StripeGenerator.m_UpscaleFactor) + "\n";
        result.m_Label += "StripeGenerator.Operation = " + ToString(m_StripeGenerator.m_Operation) + "\n";
        result.m_Label +=
            "StripeGenerator.MceShapeMultiplier = " + ToString(m_StripeGenerator.m_MceShapeMultiplier) + "\n";
        result.m_Label +=
            "StripeGenerator.PleShapeMultiplier = " + ToString(m_StripeGenerator.m_PleShapeMultiplier) + "\n";

        result.m_Label += "LowerBound = " + ToString(m_LowerBound) + "\n";
        result.m_Label += "UpperBound = " + ToString(m_UpperBound) + "\n";
        result.m_Label += "IsChannelSelector = " + ToString(m_IsChannelSelector) + "\n";
    }
    return result;
}

void McePart::setUninterleavedInputShape(TensorShape uninterleavedInputShape)
{
    m_UninterleavedInputShape = uninterleavedInputShape;
}

const std::vector<uint8_t>& McePart::GetWeightsData() const
{
    return *m_WeightsData;
}

const TensorInfo& McePart::GetWeightsInfo() const
{
    return m_WeightsInfo;
}

const std::vector<int32_t>& McePart::GetBiasData() const
{
    return m_BiasData;
}

const TensorInfo& McePart::GetBiasInfo() const
{
    return m_BiasInfo;
}

const TensorShape& McePart::GetInputTensorShape() const
{
    return m_InputTensorShape;
}

const TensorShape& McePart::GetOutputTensorShape() const
{
    return m_OutputTensorShape;
}

utils::Optional<utils::ConstTensorData> McePart::GetChannelSelectorWeights() const
{
    if (m_IsChannelSelector)
    {
        // Run some checks that this part fulfils the criteria of a channel-selector part,
        // to catch cases where it was incorrectly tagged.
        assert(GetWidth(m_InputTensorShape) == GetWidth(m_OutputTensorShape));
        assert(GetHeight(m_InputTensorShape) == GetHeight(m_OutputTensorShape));
        assert(m_InputQuantizationInfo == m_OutputQuantizationInfo);
        assert(m_WeightsInfo.m_QuantizationInfo.GetZeroPoint() == 0);
        return ConstTensorData(m_WeightsData->data(), m_WeightsInfo.m_Dimensions);
    }
    return {};
}

// Note this function is quite similar to MergeWithChannelSelectorAfter, but has some differences
// with input and output channels being swapped, and not needing to update the biases or quant info.
bool McePart::MergeWithChannelSelectorBefore(const utils::ConstTensorData& channelSelectorWeights)
{
    if (m_Operation == command_stream::MceOperation::DEPTHWISE_CONVOLUTION)
    {
        // We need to be able to change which input channels go to which output channels, which depthwise can't do
        return false;
    }

    // Check if the merged layer would actually be computationally cheaper than two separatelayers.
    // If not, then merging may make performance worse!
    uint64_t separateMacsPerPixel =
        GetNumElements(channelSelectorWeights.GetShape()) + GetNumElements(m_WeightsInfo.m_Dimensions);
    uint64_t mergedMacsPerPixel = static_cast<uint64_t>(m_WeightsInfo.m_Dimensions[0]) * m_WeightsInfo.m_Dimensions[1] *
                                  channelSelectorWeights.GetShape()[2] * m_WeightsInfo.m_Dimensions[3];
    if (separateMacsPerPixel < mergedMacsPerPixel)
    {
        return false;
    }

    uint32_t oldInputDepth = GetChannels(m_InputTensorShape);
    uint32_t newInputDepth = channelSelectorWeights.GetShape()[2];
    assert(channelSelectorWeights.GetShape()[3] == oldInputDepth);
    m_InputTensorShape[3] = newInputDepth;

    // Update weights matrix to account for the channel selector
    ConstTensorData oldWeightsData(m_WeightsData->data(), m_WeightsInfo.m_Dimensions);
    m_WeightsInfo.m_Dimensions[2] = newInputDepth;
    int32_t weightZeroPoint       = m_WeightsInfo.m_QuantizationInfo.GetZeroPoint();
    std::vector<uint8_t> newWeightsDataRaw(GetNumElements(m_WeightsInfo.m_Dimensions),
                                           static_cast<uint8_t>(weightZeroPoint));
    TensorData newWeightsData(newWeightsDataRaw.data(), m_WeightsInfo.m_Dimensions);
    for (uint32_t newI = 0; newI < newInputDepth; ++newI)
    {
        // Find which old input channel (if any) this new input channel should select
        int32_t selectedOldInputChannel = -1;
        for (uint32_t oldI = 0; oldI < oldInputDepth; ++oldI)
        {
            if (channelSelectorWeights.GetElement(0, 0, newI, oldI) > 0)
            {
                selectedOldInputChannel = oldI;
                break;
            }
        }

        if (selectedOldInputChannel == -1)
        {
            // Not selected, so leave the weights for this new input channel as the default (zero point)
            continue;
        }

        for (uint32_t h = 0; h < m_WeightsInfo.m_Dimensions[0]; ++h)
        {
            for (uint32_t w = 0; w < m_WeightsInfo.m_Dimensions[1]; ++w)
            {
                for (uint32_t o = 0; o < m_WeightsInfo.m_Dimensions[3]; ++o)
                {
                    uint8_t v = oldWeightsData.GetElement(h, w, selectedOldInputChannel, o);
                    newWeightsData.SetElement(h, w, newI, o, v);
                }
            }
        }
    }
    m_WeightsData = std::make_shared<std::vector<uint8_t>>(std::move(newWeightsDataRaw));

    m_StripeGenerator.m_MceInputTensorShape[3] = newInputDepth;
    return true;
}

// Note this function is quite similar to MergeWithChannelSelectorBefore, but has some differences
// with input and output channels being swapped, and needing to update the biases and quant info as well.
bool McePart::MergeWithChannelSelectorAfter(const utils::ConstTensorData& channelSelectorWeights)
{
    if (m_Operation == command_stream::MceOperation::DEPTHWISE_CONVOLUTION)
    {
        // We need to be able to change which input channels go to which output channels, which depthwise can't do
        return false;
    }

    // Check if the merged layer would actually be computationally cheaper than two separated layers.
    // If not, then merging may make performance worse!
    uint64_t separateMacsPerPixel =
        GetNumElements(m_WeightsInfo.m_Dimensions) + GetNumElements(channelSelectorWeights.GetShape());
    uint64_t mergedMacsPerPixel = static_cast<uint64_t>(m_WeightsInfo.m_Dimensions[0]) * m_WeightsInfo.m_Dimensions[1] *
                                  m_WeightsInfo.m_Dimensions[2] * channelSelectorWeights.GetShape()[3];
    if (separateMacsPerPixel < mergedMacsPerPixel)
    {
        return false;
    }

    assert(channelSelectorWeights.GetShape()[2] == GetChannels(m_OutputTensorShape));
    uint32_t newOutputDepth = channelSelectorWeights.GetShape()[3];
    m_OutputTensorShape[3]  = newOutputDepth;

    // Update weights matrix, bias and quant infos to account for the channel selector
    ConstTensorData oldWeightsData(m_WeightsData->data(), m_WeightsInfo.m_Dimensions);
    m_WeightsInfo.m_Dimensions[3] = newOutputDepth;
    int32_t weightZeroPoint       = m_WeightsInfo.m_QuantizationInfo.GetZeroPoint();
    std::vector<uint8_t> newWeightsDataRaw(GetNumElements(m_WeightsInfo.m_Dimensions),
                                           static_cast<uint8_t>(weightZeroPoint));
    TensorData newWeightsData(newWeightsDataRaw.data(), m_WeightsInfo.m_Dimensions);

    m_BiasInfo.m_Dimensions[3] = newOutputDepth;
    std::vector<int32_t> newBiasData(GetNumElements(m_BiasInfo.m_Dimensions), 0);

    bool isPerChannelQuant = m_WeightsInfo.m_QuantizationInfo.GetQuantizationDim() == 3;
    utils::Optional<QuantizationScales> weightsPerChannelQuantScales;
    utils::Optional<QuantizationScales> biasPerChannelQuantScales;
    if (isPerChannelQuant)
    {
        weightsPerChannelQuantScales = QuantizationScales(0.0f, newOutputDepth);
        biasPerChannelQuantScales    = QuantizationScales(0.0f, newOutputDepth);
    }

    for (uint32_t newO = 0; newO < newOutputDepth; ++newO)
    {
        // Find which (if any) old output channel this new output channel is selecting
        int32_t selectedOldOutputChannel = -1;
        for (uint32_t i = 0; i < channelSelectorWeights.GetShape()[2]; ++i)
        {
            if (channelSelectorWeights.GetElement(0, 0, i, newO) > 0)
            {
                selectedOldOutputChannel = i;
                break;
            }
        }

        if (selectedOldOutputChannel == -1)
        {
            // Not selected, so leave the weights for this new output channel as the default (zero point)
            continue;
        }

        // Update weights
        for (uint32_t h = 0; h < m_WeightsInfo.m_Dimensions[0]; ++h)
        {
            for (uint32_t w = 0; w < m_WeightsInfo.m_Dimensions[1]; ++w)
            {
                for (uint32_t i = 0; i < m_WeightsInfo.m_Dimensions[2]; ++i)
                {
                    uint8_t v = oldWeightsData.GetElement(h, w, i, selectedOldOutputChannel);
                    newWeightsData.SetElement(h, w, i, newO, v);
                }
            }
        }

        // Update bias
        newBiasData[newO] = m_BiasData[selectedOldOutputChannel];

        // Update weights and bias per-channel quant (if used)
        if (isPerChannelQuant)
        {
            weightsPerChannelQuantScales.value()[newO] =
                m_WeightsInfo.m_QuantizationInfo.GetScales()[selectedOldOutputChannel];
            biasPerChannelQuantScales.value()[newO] =
                m_BiasInfo.m_QuantizationInfo.GetScales()[selectedOldOutputChannel];
        }
    }

    m_WeightsData = std::make_shared<std::vector<uint8_t>>(std::move(newWeightsDataRaw));
    m_BiasData    = std::move(newBiasData);
    if (isPerChannelQuant)
    {
        m_WeightsInfo.m_QuantizationInfo.SetScales(weightsPerChannelQuantScales.value());
        m_BiasInfo.m_QuantizationInfo.SetScales(biasPerChannelQuantScales.value());
    }

    m_StripeGenerator.m_MceOutputTensorShape[3] = newOutputDepth;
    m_StripeGenerator.m_PleOutputTensorShape[3] = newOutputDepth;
    return true;
}

std::vector<BoundaryRequirements> McePart::GetInputBoundaryRequirements() const
{
    uint32_t kernelHeight = m_WeightsInfo.m_Dimensions[0];
    uint32_t kernelWidth  = m_WeightsInfo.m_Dimensions[1];

    BoundaryRequirements result;
    result.m_NeedsBeforeX = kernelWidth >= 2 || m_UpscaleFactor > 1;
    result.m_NeedsAfterX  = kernelWidth >= 3 || m_UpscaleFactor > 1;
    result.m_NeedsBeforeY = kernelHeight >= 2 || m_UpscaleFactor > 1;
    result.m_NeedsAfterY  = kernelHeight >= 3 || m_UpscaleFactor > 1;

    return { result };
}

std::vector<bool> McePart::CanInputsTakePleInputSram() const
{
    // We can't take input that's in PLE input SRAM, as it needs to go to the MCE
    return { false };
}

void McePart::PreprocessWeightsAsync() const
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
    request.m_PaddingTop             = m_PadTop;
    request.m_PaddingLeft            = m_PadLeft;
    request.m_IterationSize          = 0;
    request.m_Operation              = m_Operation;
    request.m_Algorithm              = CompilerMceAlgorithm::Direct;

    // Note we only consider high priority lonely plans so that we don't encode a bunch of weights
    // which we might never consider (for low priority plans). If we do need these, they will encoded
    // later (serially).
    StripeInfos stripeInfosLonely =
        m_StripeGenerator.GenerateStripes(CascadeType::Lonely, m_OutputBoundaryRequirements.at(0), PlanPriority::High);
    for (const MceAndPleInfo& i : stripeInfosLonely.m_MceAndPleInfos)
    {
        WeightEncodingRequest modifiedRequest = request;
        modifiedRequest.m_StripeDepth         = GetWeightStripeDepth(m_WeightsInfo, i.m_MceCompute.m_Weight, m_Stride);
        modifiedRequest.m_IterationSize       = i.m_MceCompute.m_Weight[2];
        modifiedRequest.m_Algorithm = ResolveMceAlgorithm(i.m_MceCompute.m_BlockConfig, i.m_MceCompute.m_Weight[2]);

        m_WeightEncoderCache.EncodeStage1Async(std::move(modifiedRequest));
    }

    StripeInfos stripeInfosBeginning =
        m_StripeGenerator.GenerateStripes(CascadeType::Beginning, m_OutputBoundaryRequirements.at(0), {});
    for (const MceAndPleInfo& i : stripeInfosBeginning.m_MceAndPleInfos)
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
