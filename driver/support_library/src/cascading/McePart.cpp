//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "McePart.hpp"
#include "PartUtils.hpp"
#include "Plan.hpp"
#include "StripeHelper.hpp"

#include <algorithm>
#include <ethosn_utils/Macros.hpp>

using namespace ethosn::command_stream;

namespace ethosn
{
namespace support_library
{
using namespace impl;
using namespace utils;

namespace
{

bool IsSramBufferValid(bool needsBoundaryBeforeX,
                       bool needsBoundaryAfterX,
                       bool needsBoundaryBeforeY,
                       bool needsBoundaryAfterY,
                       Buffer* sramBuffer)
{
    uint32_t heightSplits = DivRoundUp(GetHeight(sramBuffer->m_TensorShape), GetHeight(sramBuffer->m_StripeShape));
    uint32_t widthSplits  = DivRoundUp(GetWidth(sramBuffer->m_TensorShape), GetWidth(sramBuffer->m_StripeShape));
    // In the middle of a casade:
    // If the kernel height/width is 1 we require only 1 input buffer because we don't need boundary data
    // If the kernel height/width is 2 we require 2 because we only require the top / left boundary data
    // If the kernel height/width is 3 or more we require 3 as we need top/left and bottom/right boundary data.
    if (heightSplits > 1 && widthSplits > 1)
    {
        // Splitting both width and height is not supported in a cascade
        return false;
    }
    uint32_t split           = 0;
    bool needsBoundaryBefore = false;
    bool needsBoundaryAfter  = false;
    if (heightSplits > 1)
    {
        split               = heightSplits;
        needsBoundaryBefore = needsBoundaryBeforeY;
        needsBoundaryAfter  = needsBoundaryAfterY;
    }
    else
    {
        split               = widthSplits;
        needsBoundaryBefore = needsBoundaryBeforeX;
        needsBoundaryAfter  = needsBoundaryAfterX;
    }
    if (needsBoundaryBefore && needsBoundaryAfter)
    {
        // For 3 height splits the number of stripes needs to be the number of splits
        if (split <= 3)
        {
            return sramBuffer->m_NumStripes == std::min(split, 3u);
        }
        if (sramBuffer->m_NumStripes < 3)
        {
            return false;
        }
    }
    else if (needsBoundaryBefore || needsBoundaryAfter)
    {
        // For 2 height splits the number of stripes needs to be the number of splits
        if (split <= 2)
        {
            return sramBuffer->m_NumStripes == std::min(split, 2u);
        }
        if (sramBuffer->m_NumStripes != 2)
        {
            return false;
        }
    }
    else
    {
        if (sramBuffer->m_NumStripes != 1)
        {
            return false;
        }
    }
    return true;
}

struct NumStripesGrouped
{
    NumStripes m_Input;
    NumStripes m_Output;
    NumStripes m_Weights;
    NumStripes m_PleInput;
};

utils::Optional<std::pair<MceAndPleInfo, MceOnlyInfo>>
    GenerateContinueSectionStripeInfos(NumStripesGrouped& numStripes,
                                       const Buffer* sramBuffer,
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
                                       const StripeConfig& stripeConfig)
{
    assert(cascadeType == CascadeType::Middle || cascadeType == CascadeType::End);
    const TensorShape& mceInputStripe = sramBuffer->m_StripeShape;
    bool fullHeight                   = GetHeight(sramBuffer->m_StripeShape) >= GetHeight(sramBuffer->m_TensorShape);
    bool fullWidth                    = GetWidth(sramBuffer->m_StripeShape) >= GetWidth(sramBuffer->m_TensorShape);
    bool fullTensor                   = fullHeight && fullWidth;
    TensorShape mceOutputEncoding     = { 0, 0, 0, 0 };
    // If we have the full plane, the number of weight stripes is 1 and we're in the middle of a cascade we can use the full tensor
    // But if we're at the end of a cascade we can split the output depth so we get more parallelism.
    if (fullTensor && numWeightStripes == 1 && cascadeType == CascadeType::Middle)
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
    else if (fullTensor)
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
    else
    {
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
    TensorShape mceOutputStripe =
        impl::CreateStripe(outputTensorShape, mceOutputEncoding, caps.GetBrickGroupShape()[3]);

    TensorShape pleInputEncoding = mceOutputEncoding;
    if (cascadeType == CascadeType::Middle)
    {
        // PLE accumulates the full depth for the middle of an s1 cascade
        pleInputEncoding[3] = 0;
    }
    TensorShape pleInputStripe  = impl::CreateStripe(outputTensorShape, pleInputEncoding, caps.GetBrickGroupShape()[3]);
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

    uint32_t memoryOutputChannelsEncoding = 0;
    if (fullTensor && cascadeType == CascadeType::End)
    {
        memoryOutputChannelsEncoding = caps.GetNumberOfOgs();
    }
    TensorShape memoryOutputStripeEncoding{ 0, fullHeight ? 0 : GetHeight(mceOutputStripe),
                                            fullWidth ? 0 : GetWidth(mceOutputStripe), memoryOutputChannelsEncoding };
    TensorShape memoryOutputStripe =
        CreateStripe(outputTensorShape, memoryOutputStripeEncoding, caps.GetBrickGroupShape()[3]);

    bool fullDepth            = memoryOutputStripe[3] >= outputTensorShape[3];
    bool isEndOfCascade       = cascadeType == CascadeType::End;
    uint32_t maxOutputStripes = 0;
    // strategy 0
    if (!fullTensor)
    {
        // if its the end of a cascade we can double buffer the output, if it's not we need to output up to 3 stripes for neighouring data.
        maxOutputStripes = isEndOfCascade ? 2 : 3;
    }
    // Strategy 1/3
    else if (isEndOfCascade && fullDepth)
    {
        assert(fullTensor);
        maxOutputStripes = 1;
    }
    else if (!isEndOfCascade)
    {
        assert(fullDepth);
        maxOutputStripes = 1;
    }
    else if (!fullDepth)
    {
        assert(fullTensor && isEndOfCascade);
        maxOutputStripes = 2;
    }
    numStripes.m_Output = { 1, maxOutputStripes };

    command_stream::cascading::PackedBoundaryThickness packedBoundaryThickness = { 0, 0, 0, 0 };
    const uint32_t numIfmLoads                                                 = 1;
    const uint32_t numWeightLoads                                              = 1;

    // Prevent too many MCE stripes per PLE (a firmware limitation)
    const uint32_t numMceStripesPerPle = utils::DivRoundUp(GetChannels(pleInputStripe), GetChannels(mceOutputStripe));
    if (numMceStripesPerPle > caps.GetMaxMceStripesPerPleStripe())
    {
        return {};
    }

    // Prevent too many IFM and Weight stripes per PLE (a firmware limitation)
    const uint32_t numIfmStripesPerMce       = 0;    // Continue section, so no IfmS
    const uint32_t numWgtStripesPerMce       = 1;
    const uint32_t numIfmAndWgtStripesPerPle = (numIfmStripesPerMce + numWgtStripesPerMce) * numMceStripesPerPle;
    if (numIfmAndWgtStripesPerPle > caps.GetMaxIfmAndWgtStripesPerPleStripe())
    {
        return {};
    }

    MceAndPleInfo mceAndPleInfo;

    mceAndPleInfo.m_MceCompute.m_Input       = sramBuffer->m_StripeShape;
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

McePart::McePart(PartId id,
                 const TensorShape& inputTensorShape,
                 const TensorShape& outputTensorShape,
                 const QuantizationInfo& inputQuantizationInfo,
                 const QuantizationInfo& outputQuantizationInfo,
                 const TensorInfo& weightsInfo,
                 std::vector<uint8_t> weightsData,
                 const TensorInfo& biasInfo,
                 std::vector<int32_t> biasData,
                 Stride stride,
                 uint32_t padTop,
                 uint32_t padLeft,
                 command_stream::MceOperation op,
                 const EstimationOptions& estOpt,
                 const CompilationOptions& compOpt,
                 const HardwareCapabilities& capabilities,
                 std::set<uint32_t> operationIds,
                 DataType inputDataType,
                 DataType outputDataType)
    : BasePart(id, "McePart", CompilerDataFormat::NONE, operationIds, estOpt, compOpt, capabilities)
    , m_InputTensorShape(inputTensorShape)
    , m_OutputTensorShape(outputTensorShape)
    , m_WeightEncoderCache{ capabilities, m_DebugTag.c_str() }
    , m_InputQuantizationInfo(inputQuantizationInfo)
    , m_OutputQuantizationInfo(outputQuantizationInfo)
    , m_WeightsInfo(weightsInfo)
    , m_WeightsData(std::make_shared<std::vector<uint8_t>>(std::move(weightsData)))
    , m_BiasInfo(biasInfo)
    , m_BiasData(std::move(biasData))
    , m_Stride(stride)
    , m_UpscaleFactor(1U)
    , m_UpsampleType(command_stream::cascading::UpsampleType::OFF)
    , m_PadTop(padTop)
    , m_PadLeft(padLeft)
    , m_Operation(op)
    , m_StripeConfig(GetDefaultStripeConfig(compOpt, m_DebugTag.c_str()))
    , m_StripeGenerator(m_InputTensorShape,
                        m_OutputTensorShape,
                        m_OutputTensorShape,
                        m_WeightsInfo.m_Dimensions[0],
                        m_WeightsInfo.m_Dimensions[1],
                        m_PadTop,
                        m_PadLeft,
                        m_UpscaleFactor,
                        op,
                        PleOperation::PASSTHROUGH,
                        ShapeMultiplier{ 1, 1, Fraction(1, stride.m_X * stride.m_Y) },
                        ShapeMultiplier::Identity,
                        capabilities,
                        m_StripeConfig)
    , m_InputDataType(inputDataType)
    , m_OutputDataType(outputDataType)
    , m_LowerBound(outputDataType == DataType::UINT8_QUANTIZED ? 0 : -128)
    , m_UpperBound(outputDataType == DataType::UINT8_QUANTIZED ? 255 : 127)
{}

McePart::McePart(ConstructionParams&& params)
    : BasePart(params.m_Id,
               "McePart",
               CompilerDataFormat::NONE,
               params.m_OperationIds,
               params.m_EstOpt,
               params.m_CompOpt,
               params.m_Capabilities)
    , m_InputTensorShape(params.m_InputTensorShape)
    , m_OutputTensorShape(params.m_OutputTensorShape)
    , m_WeightEncoderCache{ params.m_Capabilities, m_DebugTag.c_str() }
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

    WeightEncoderCache::Params wp;
    wp.weightsTensorInfo      = convData.weightInfo;
    wp.weightsData            = convData.weightData;
    wp.biasTensorInfo         = convData.biasInfo;
    wp.biasData               = convData.biasData;
    wp.inputQuantizationInfo  = m_InputQuantizationInfo;
    wp.outputQuantizationInfo = m_OutputQuantizationInfo;
    wp.stripeDepth            = weightStripeDepth;
    wp.strideY                = m_Stride.m_Y;
    wp.strideX                = m_Stride.m_X;
    wp.paddingTop             = m_PadTop;
    wp.paddingLeft            = m_PadLeft;
    wp.iterationSize          = weightStripeSize;
    wp.operation              = m_Operation;
    wp.algorithm              = mceOpAlgo;
    auto encodedWeights       = weightEncoderCache.Encode(wp);
    if (!encodedWeights)
    {
        return nullptr;    // Weight compression failed (too big for SRAM) - abandon this plan
    }

    CascadingBufferFormat formatInDram = impl::GetCascadingBufferFormatFromCompilerDataFormat(
        ConvertExternalToCompilerDataFormat(convData.weightInfo.m_DataFormat));
    Buffer* dramWeightBuffer =
        opGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, formatInDram, TraversalOrder::Xyz));
    dramWeightBuffer->m_DataType    = convData.weightInfo.m_DataType;
    dramWeightBuffer->m_TensorShape = convData.weightInfo.m_Dimensions;
    dramWeightBuffer->m_TensorShape[2] =
        GetNumSubmapChannels(dramWeightBuffer->m_TensorShape[2], wp.strideX, wp.strideY, m_Capabilities);
    dramWeightBuffer->m_EncodedWeights   = std::move(encodedWeights);
    dramWeightBuffer->m_SizeInBytes      = static_cast<uint32_t>(dramWeightBuffer->m_EncodedWeights->m_Data.size());
    dramWeightBuffer->m_QuantizationInfo = convData.weightInfo.m_QuantizationInfo;
    dramWeightBuffer->m_BufferType       = BufferType::ConstantDma;

    CascadingBufferFormat formatInSram = GetCascadingBufferFormatFromCompilerDataFormat(CompilerDataFormat::WEIGHT);
    Buffer* sramWeightBuffer =
        opGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, formatInSram, TraversalOrder::Xyz));
    sramWeightBuffer->m_DataType         = convData.weightInfo.m_DataType;
    sramWeightBuffer->m_TensorShape      = dramWeightBuffer->m_TensorShape;
    sramWeightBuffer->m_StripeShape      = memoryWeightStripe;
    sramWeightBuffer->m_QuantizationInfo = convData.weightInfo.m_QuantizationInfo;
    sramWeightBuffer->m_NumStripes       = numMemoryWeightStripes;
    sramWeightBuffer->m_SizeInBytes      = dramWeightBuffer->m_EncodedWeights->m_MaxSize * numMemoryWeightStripes;
    sramWeightBuffer->m_SlotSizeInBytes  = dramWeightBuffer->m_EncodedWeights->m_MaxSize;
    sramWeightBuffer->m_NumLoads         = numLoads;

    Op* dmaOp             = opGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::WEIGHT));
    dmaOp->m_OperationIds = m_CorrespondingOperationIds;

    opGraph.AddConsumer(dramWeightBuffer, dmaOp, 0);
    opGraph.SetProducer(sramWeightBuffer, dmaOp);

    // Use the encoded weights to determine the size of the sram and dram buffers
    return sramWeightBuffer;
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
    uint32_t kernelHeight   = m_WeightsInfo.m_Dimensions[0];
    uint32_t kernelWidth    = m_WeightsInfo.m_Dimensions[1];
    const bool isWinograd2d = (kernelHeight > 1) && (kernelWidth > 1);

    CompilerMceAlgorithm effectiveAlgo = CompilerMceAlgorithm::Direct;
    // Winograd and upscaling cannot be performed at the same time
    if (!m_CompilationOptions.m_DisableWinograd && m_Operation == command_stream::MceOperation::CONVOLUTION &&
        m_Stride == Stride{ 1, 1 } && m_UpsampleType == command_stream::cascading::UpsampleType::OFF)
    {
        effectiveAlgo =
            utils::FindBestConvAlgorithm(m_Capabilities, m_WeightsInfo.m_Dimensions[0], m_WeightsInfo.m_Dimensions[1]);
    }

    std::vector<command_stream::BlockConfig> blockConfigs =
        FilterAlgoBlockConfigs(effectiveAlgo, isWinograd2d, { mceStripeInfo.m_BlockConfig }, m_Capabilities);

    CompilerMceAlgorithm mceOpAlgo = blockConfigs.empty() ? CompilerMceAlgorithm::Direct : effectiveAlgo;

    // Encoder doesn't support multiple iterations with Winograd enabled
    if (mceStripeInfo.m_Weight[2] < convData.weightInfo.m_Dimensions[2])
    {
        mceOpAlgo = CompilerMceAlgorithm::Direct;
    }

    const TraversalOrder ifmTraversalOrder =
        m_Operation == command_stream::MceOperation::DEPTHWISE_CONVOLUTION ? TraversalOrder::Xyz : TraversalOrder::Zxy;
    Buffer* sramInBuffer =
        opGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB, ifmTraversalOrder));
    sramInBuffer->m_DataType                                               = m_InputDataType;
    sramInBuffer->m_TensorShape                                            = inputShape;
    sramInBuffer->m_StripeShape                                            = memoryStripesInfo.m_Input.m_Shape;
    sramInBuffer->m_NumStripes                                             = numMemoryStripes.m_Input;
    std::tie(sramInBuffer->m_SlotSizeInBytes, sramInBuffer->m_SizeInBytes) = CalculateTileSize(
        m_Capabilities, sramInBuffer->m_TensorShape, sramInBuffer->m_StripeShape,
        memoryStripesInfo.m_Input.m_PackedBoundaryThickness, sramInBuffer->m_NumStripes, couldSourceBeFcaf);
    sramInBuffer->m_QuantizationInfo        = inputQuantInfo;
    sramInBuffer->m_PackedBoundaryThickness = memoryStripesInfo.m_Input.m_PackedBoundaryThickness;
    sramInBuffer->m_NumLoads                = memoryStripesInfo.m_Input.m_NumLoads;

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
    opGraph.AddConsumer(sramInBuffer, op, 0);
    opGraph.AddConsumer(sramWeightBuffer, op, 1);

    return { sramInBuffer, op };
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

                auto pleInBuffer = impl::AddPleInBuffer(opGraph, numPleInputStripes, m_OutputTensorShape,
                                                        info.m_Memory.m_PleInput.m_Shape, m_OutputQuantizationInfo,
                                                        m_OutputDataType, Location::PleInputSram);
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
                AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph), plans, false, true);
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

            auto outBuffer =
                impl::AddPleInBuffer(opGraph, numPleInputStripes, m_OutputTensorShape, info.m_Memory.m_PleInput.m_Shape,
                                     m_OutputQuantizationInfo, m_OutputDataType, Location::PleInputSram);
            opGraph.SetProducer(outBuffer, inBufferAndMceOp.second);
            inputMappings[inBufferAndMceOp.first] = PartInputSlot{ m_PartId, 0 };
            outputMappings[outBuffer]             = PartOutputSlot{ m_PartId, 0 };
            AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph), plans);
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

    // Generate all possible plans.
    StripeInfos stripeInfos = m_StripeGenerator.GenerateStripes(CascadeType::Lonely);
    // Data could be de-compressed from FCAF
    const bool couldSourceBeFcaf = true;
    for (const MceAndPleInfo& i : stripeInfos.m_MceAndPleInfos)
    {
        CreateMceAndIdentityPlePlans(i, m_WeightEncoderCache, ret, numWeightStripes, couldSourceBeFcaf);
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

    StripeInfos stripeInfos = m_StripeGenerator.GenerateStripes(CascadeType::Beginning);

    // The plan will be "glued" to the end plan from the previous section.
    // Therefore the input buffer tile cannot be unconditionally clamped to the
    // tensor size since input data (from DRAM) could be FCAF compressed and the
    // HW always writes to the SRAM in full cell size. Clamping is only allowed
    // if the stripe shape is not multiple of FCAF cell size.
    const bool couldSourceBeFcaf = true;

    for (const MceAndPleInfo& i : stripeInfos.m_MceAndPleInfos)
    {
        CreateMceAndIdentityPlePlans(i, m_WeightEncoderCache, ret, numWeightStripes, couldSourceBeFcaf);
    }

    for (const MceOnlyInfo& i : stripeInfos.m_MceOnlyInfos)
    {
        CreateMceOnlyPlans(i, m_WeightEncoderCache, ret, numWeightStripes, couldSourceBeFcaf);
    }

    return ret;
}

Plans McePart::GetMiddlePlans(ethosn::command_stream::BlockConfig blockConfig,
                              Buffer* sramBuffer,
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

    bool needsBoundaryBeforeX = kernelWidth >= 2 || m_UpscaleFactor > 1;
    bool needsBoundaryAfterX  = kernelWidth >= 3 || m_UpscaleFactor > 1;
    bool needsBoundaryBeforeY = kernelHeight >= 2 || m_UpscaleFactor > 1;
    bool needsBoundaryAfterY  = kernelHeight >= 3 || m_UpscaleFactor > 1;

    if (!IsSramBufferValid(needsBoundaryBeforeX, needsBoundaryAfterX, needsBoundaryBeforeY, needsBoundaryAfterY,
                           sramBuffer))
    {
        return ret;
    }

    NumStripesGrouped numStripes;
    numStripes.m_Input = { sramBuffer->m_NumStripes, sramBuffer->m_NumStripes };
    // Multiple output stripes are needed because the follow layers may require multiple buffers due to boundary data.
    // These will be filtered out by the following layer
    numStripes.m_Output   = { 1, 3 };
    numStripes.m_Weights  = { numWeightStripes, numWeightStripes };
    numStripes.m_PleInput = { 0, 0 };

    bool isDepthwise = m_Operation == ethosn::command_stream::MceOperation::DEPTHWISE_CONVOLUTION;
    auto stripeInfos = GenerateContinueSectionStripeInfos(
        numStripes, sramBuffer, numWeightStripes, isDepthwise, m_Capabilities, m_OutputTensorShape, kernelHeight,
        kernelWidth, strideMultiplier, m_StripeGenerator.m_MceShapeMultiplier, blockConfig, CascadeType::Middle,
        m_StripeConfig);

    if (!stripeInfos.has_value())
    {
        return ret;
    }

    // Data in the input buffer (SRAM) cannot be FCAF de-compressed.
    // Hence input tile is allowed to be clamped to tensor size.
    const bool couldSourceBeFcaf = false;

    CreateMceAndIdentityPlePlans(stripeInfos.value().first, m_WeightEncoderCache, ret, numWeightStripes,
                                 couldSourceBeFcaf);
    CreateMceOnlyPlans(stripeInfos.value().second, m_WeightEncoderCache, ret, numWeightStripes, couldSourceBeFcaf);
    return ret;
}

Plans McePart::GetEndPlans(ethosn::command_stream::BlockConfig blockConfig,
                           Buffer* sramBuffer,
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

    bool needsBoundaryBeforeX = kernelWidth >= 2 || m_UpscaleFactor > 1;
    bool needsBoundaryAfterX  = kernelWidth >= 3 || m_UpscaleFactor > 1;
    bool needsBoundaryBeforeY = kernelHeight >= 2 || m_UpscaleFactor > 1;
    bool needsBoundaryAfterY  = kernelHeight >= 3 || m_UpscaleFactor > 1;

    if (!IsSramBufferValid(needsBoundaryBeforeX, needsBoundaryAfterX, needsBoundaryBeforeY, needsBoundaryAfterY,
                           sramBuffer))
    {
        return ret;
    }

    NumStripesGrouped numStripes;
    numStripes.m_Input    = { sramBuffer->m_NumStripes, sramBuffer->m_NumStripes };
    numStripes.m_Output   = { 1, 2 };
    numStripes.m_Weights  = { numWeightStripes, numWeightStripes };
    numStripes.m_PleInput = { 0, 0 };

    bool isDepthwise = m_Operation == ethosn::command_stream::MceOperation::DEPTHWISE_CONVOLUTION;
    auto stripeInfos = GenerateContinueSectionStripeInfos(
        numStripes, sramBuffer, numWeightStripes, isDepthwise, m_Capabilities, m_OutputTensorShape, kernelHeight,
        kernelWidth, strideMultiplier, m_StripeGenerator.m_MceShapeMultiplier, blockConfig, CascadeType::End,
        m_StripeConfig);

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
                        Buffer* sramBuffer,
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
            return GetMiddlePlans(blockConfig, sramBuffer, numWeightStripes);
        }
        case CascadeType::End:
        {
            return GetEndPlans(blockConfig, sramBuffer, numWeightStripes);
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

void McePart::ModifyActivationBounds(int16_t lowerBound, int16_t upperBound)
{
    m_LowerBound = lowerBound;
    m_UpperBound = upperBound;
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
    }
    return result;
}

void McePart::setUninterleavedInputShape(TensorShape uninterleavedInputShape)
{
    m_UninterleavedInputShape = uninterleavedInputShape;
}

}    // namespace support_library
}    // namespace ethosn
