//
// Copyright © 2021 Arm Limited.
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

bool IsSramBufferValid(uint32_t kernelHeight, uint32_t kernelWidth, Buffer* sramBuffer)
{
    uint32_t heightSplits = DivRoundUp(GetHeight(sramBuffer->m_TensorShape), GetHeight(sramBuffer->m_StripeShape));
    uint32_t widthSplits  = DivRoundUp(GetWidth(sramBuffer->m_TensorShape), GetWidth(sramBuffer->m_StripeShape));
    // In the middle of a casade:
    // If the kernel height/width is 1 we require only 1 input buffer because we don't need boundary data
    // If the kernel height/width is 2 we require 2 because we only require the top / left boundary data
    // If the kernel height/width is 3 or more we require 3 as we need top/left and bottom/right boundary data.
    if (kernelHeight >= 3 || kernelWidth >= 3)
    {
        // For 3 height splits the number of stripes needs to be the number of splits
        if (heightSplits <= 3 && widthSplits <= 3)
        {
            return sramBuffer->m_NumStripes == std::min(heightSplits, 3u);
        }
        if (sramBuffer->m_NumStripes < 3)
        {
            return false;
        }
    }
    else if (kernelHeight >= 2 || kernelWidth >= 2)
    {
        // For 2 height splits the number of stripes needs to be the number of splits
        if (heightSplits <= 2 && widthSplits <= 2)
        {
            return sramBuffer->m_NumStripes == std::min(heightSplits, 2u);
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
                                       BlockConfig& blockConfig,
                                       CascadeType cascadeType)
{
    assert(cascadeType == CascadeType::Middle || cascadeType == CascadeType::End);
    const TensorShape& mceInputStripe = sramBuffer->m_StripeShape;
    bool fullHeight                   = GetHeight(sramBuffer->m_StripeShape) >= GetHeight(sramBuffer->m_TensorShape);
    bool fullWidth                    = GetWidth(sramBuffer->m_StripeShape) >= GetWidth(sramBuffer->m_TensorShape);
    bool fullTensor                   = fullHeight && fullWidth;
    TensorShape mceOutputEncoding     = { 0, 0, 0, 0 };
    if (fullTensor && numWeightStripes == 1)
    {
        // strategy 3
        mceOutputEncoding = { 0, 0, 0, 0 };
    }
    else if (fullTensor)
    {
        // strategy 1
        mceOutputEncoding = { 0, 0, 0, caps.GetNumberOfOgs() };
    }
    else
    {
        mceOutputEncoding = { 0, fullHeight ? 0 : GetHeight(mceInputStripe), fullWidth ? 0 : GetWidth(mceInputStripe),
                              0 };
    }
    TensorShape mceOutputStripe    = impl::CreateStripe(outputTensorShape, mceOutputEncoding, caps.GetNumberOfOgs());
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

    MceAndPleInfo mceAndPleInfo;
    mceAndPleInfo.m_MceCompute.m_Input       = sramBuffer->m_StripeShape;
    mceAndPleInfo.m_MceCompute.m_Output      = mceOutputStripe;
    mceAndPleInfo.m_MceCompute.m_Weight      = mceWeightStripe;
    mceAndPleInfo.m_MceCompute.m_BlockConfig = blockConfig;
    mceAndPleInfo.m_PleCompute.m_Input       = mceOutputStripe;
    mceAndPleInfo.m_PleCompute.m_Output      = mceOutputStripe;
    mceAndPleInfo.m_PleCompute.m_BlockConfig = blockConfig;

    mceAndPleInfo.m_Memory.m_Input    = { numStripes.m_Input, mceInputStripe };
    mceAndPleInfo.m_Memory.m_Output   = { numStripes.m_Output, memoryOutputStripe };
    mceAndPleInfo.m_Memory.m_Weight   = { numStripes.m_Weights, memoryWeightStripe };
    mceAndPleInfo.m_Memory.m_PleInput = { numStripes.m_PleInput, mceOutputStripe };

    MceOnlyInfo mceOnlyInfo;

    mceOnlyInfo.m_MceCompute.m_Input       = mceInputStripe;
    mceOnlyInfo.m_MceCompute.m_Output      = mceOutputStripe;
    mceOnlyInfo.m_MceCompute.m_Weight      = mceWeightStripe;
    mceOnlyInfo.m_MceCompute.m_BlockConfig = blockConfig;

    mceOnlyInfo.m_Memory.m_Input    = { numStripes.m_Input, mceInputStripe };
    mceOnlyInfo.m_Memory.m_Output   = { { 0, 0 }, { 0, 0, 0, 0 } };
    mceOnlyInfo.m_Memory.m_Weight   = { numStripes.m_Weights, memoryWeightStripe };
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
                 command_stream::DataType dataType)
    : BasePart(id, CompilerDataFormat::NONE, operationIds, estOpt, compOpt, capabilities)
    , m_InputTensorShape(inputTensorShape)
    , m_OutputTensorShape(outputTensorShape)
    , m_InputQuantizationInfo(inputQuantizationInfo)
    , m_OutputQuantizationInfo(outputQuantizationInfo)
    , m_WeightsInfo(weightsInfo)
    , m_WeightsData(std::make_shared<std::vector<uint8_t>>(std::move(weightsData)))
    , m_BiasInfo(biasInfo)
    , m_BiasData(std::move(biasData))
    , m_Stride(stride)
    , m_UpscaleFactor(1U)
    , m_UpsampleType(command_stream::UpsampleType::OFF)
    , m_PadTop(padTop)
    , m_PadLeft(padLeft)
    , m_Operation(op)
    , m_StripeGenerator(m_InputTensorShape,
                        m_OutputTensorShape,
                        m_OutputTensorShape,
                        m_WeightsInfo.m_Dimensions[0],
                        m_WeightsInfo.m_Dimensions[1],
                        m_Stride,
                        m_UpscaleFactor,
                        op,
                        ShapeMultiplier{ 1, 1, 1 },
                        capabilities)
    , m_WeightEncoderCache{ capabilities }
    , m_DataType(dataType)
{}

Buffer* McePart::AddWeightBuffersAndDmaOpToMceOp(OwnedOpGraph& opGraph,
                                                 Lifetime lifetime,
                                                 const impl::MceStripesInfo& mceComputeInfo,
                                                 const impl::NumStripesType& numMemoryWeightStripes,
                                                 const TensorShape& memoryWeightStripe,
                                                 TraversalOrder order,
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

    CascadingBufferFormat formatInDram = impl::GetCascadingBufferFormatFromCompilerDataFormat(
        ConvertExternalToCompilerDataFormat(convData.weightInfo.m_DataFormat));
    Buffer* dramWeightBuffer =
        opGraph.AddBuffer(std::make_unique<Buffer>(lifetime, Location::Dram, formatInDram, order));
    dramWeightBuffer->m_TensorShape      = convData.weightInfo.m_Dimensions;
    dramWeightBuffer->m_EncodedWeights   = std::move(encodedWeights);
    dramWeightBuffer->m_SizeInBytes      = static_cast<uint32_t>(dramWeightBuffer->m_EncodedWeights->m_Data.size());
    dramWeightBuffer->m_QuantizationInfo = convData.weightInfo.m_QuantizationInfo;

    CascadingBufferFormat formatInSram = GetCascadingBufferFormatFromCompilerDataFormat(CompilerDataFormat::WEIGHT);
    Buffer* sramWeightBuffer =
        opGraph.AddBuffer(std::make_unique<Buffer>(lifetime, Location::Sram, formatInSram, order));
    sramWeightBuffer->m_TensorShape      = dramWeightBuffer->m_TensorShape;
    sramWeightBuffer->m_StripeShape      = memoryWeightStripe;
    sramWeightBuffer->m_QuantizationInfo = convData.weightInfo.m_QuantizationInfo;
    sramWeightBuffer->m_NumStripes       = numMemoryWeightStripes;
    sramWeightBuffer->m_SizeInBytes      = dramWeightBuffer->m_EncodedWeights->m_MaxSize * numMemoryWeightStripes;

    Op* dmaOp             = opGraph.AddOp(std::make_unique<DmaOp>());
    dmaOp->m_OperationIds = m_CorrespondingOperationIds;

    opGraph.AddConsumer(dramWeightBuffer, dmaOp, 0);
    opGraph.SetProducer(sramWeightBuffer, dmaOp);

    // Use the encoded weights to determine the size of the sram and dram buffers
    return sramWeightBuffer;
}

uint32_t McePart::CalculateTileSize(const HardwareCapabilities& caps,
                                    const TensorShape& inputTensorShape,
                                    const TensorShape& inputStripeShape,
                                    const TensorShape& outputStripeShape,
                                    uint32_t numStripes) const
{

    auto kernelHeight               = m_WeightsInfo.m_Dimensions[0];
    auto padTop                     = m_PadTop;
    const uint32_t brickGroupHeight = GetHeight(caps.GetBrickGroupShape());

    // Work out the tile sizes by deciding how many stripes we want in each tile
    const NeedBoundary needBoundaryY = ethosn::support_library::utils::GetBoundaryRequirements(
        padTop, GetHeight(inputTensorShape), GetHeight(inputStripeShape), GetHeight(outputStripeShape), kernelHeight);

    const bool isStreamingWidth = GetWidth(inputStripeShape) < GetWidth(inputTensorShape);

    const bool needsBoundarySlots = (needBoundaryY.m_Before || needBoundaryY.m_After) && (isStreamingWidth);
    const uint32_t inputStripeXZ  = GetWidth(inputStripeShape) * GetChannels(inputStripeShape);

    const uint32_t boundarySlotSize = needsBoundarySlots ? (brickGroupHeight * inputStripeXZ) : 0U;
    const uint32_t defaultSlotSize  = TotalSizeBytes(inputStripeShape);

    // We need the boundary slots both on the top and bottom of the stripe
    const uint32_t totalSlotSize = (2U * boundarySlotSize) + defaultSlotSize;

    uint32_t inputFullStripeSize = totalSlotSize * numStripes;

    const uint32_t inputTileSize = utils::MaxTileSize(inputTensorShape, caps);

    return std::min(inputTileSize, inputFullStripeSize);
}

std::pair<Buffer*, Op*> McePart::AddMceToOpGraph(OwnedOpGraph& opGraph,
                                                 Lifetime lifetime,
                                                 TraversalOrder order,
                                                 const impl::MceStripesInfo& mceStripeInfo,
                                                 const impl::MemoryStripesInfo& memoryStripesInfo,
                                                 impl::NumMemoryStripes& numMemoryStripes,
                                                 const TensorShape& inputShape,
                                                 const QuantizationInfo& inputQuantInfo,
                                                 impl::ConvData& convData,
                                                 WeightEncoderCache& weightEncoderCache) const
{
    uint32_t kernelHeight   = m_WeightsInfo.m_Dimensions[0];
    uint32_t kernelWidth    = m_WeightsInfo.m_Dimensions[1];
    const bool isWinograd2d = (kernelHeight > 1) && (kernelWidth > 1);

    CompilerMceAlgorithm effectiveAlgo = CompilerMceAlgorithm::Direct;
    // Winograd and upscaling cannot be performed at the same time
    if (!m_CompilationOptions.m_DisableWinograd && m_Operation == command_stream::MceOperation::CONVOLUTION &&
        m_Stride == Stride{ 1, 1 } && m_UpsampleType == command_stream::UpsampleType::OFF)
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

    Buffer* sramInBuffer =
        opGraph.AddBuffer(std::make_unique<Buffer>(lifetime, Location::Sram, CascadingBufferFormat::NHWCB, order));
    sramInBuffer->m_TensorShape = inputShape;
    sramInBuffer->m_StripeShape = memoryStripesInfo.m_Input.m_Shape;
    sramInBuffer->m_NumStripes  = numMemoryStripes.m_Input;
    sramInBuffer->m_SizeInBytes =
        CalculateTileSize(m_Capabilities, sramInBuffer->m_TensorShape, sramInBuffer->m_StripeShape,
                          memoryStripesInfo.m_PleInput.m_Shape, sramInBuffer->m_NumStripes);
    sramInBuffer->m_QuantizationInfo = inputQuantInfo;

    Buffer* sramWeightBuffer = AddWeightBuffersAndDmaOpToMceOp(
        opGraph, lifetime, mceStripeInfo, numMemoryStripes.m_Weight, memoryStripesInfo.m_Weight.m_Shape, order,
        convData, weightEncoderCache, mceOpAlgo);

    auto mceOp = std::make_unique<MceOp>(
        lifetime, m_Operation, mceOpAlgo, mceStripeInfo.m_BlockConfig, mceStripeInfo.m_Input, mceStripeInfo.m_Output,
        memoryStripesInfo.m_Weight.m_Shape, TraversalOrder::Xyz, m_Stride, m_PadLeft, m_PadTop);
    Op* op             = opGraph.AddOp(std::move(mceOp));
    op->m_OperationIds = m_CorrespondingOperationIds;
    opGraph.AddConsumer(sramInBuffer, op, 0);
    opGraph.AddConsumer(sramWeightBuffer, op, 1);

    return { sramInBuffer, op };
};

void McePart::CreateMceAndIdentityPlePlans(const impl::MceAndPleInfo& info,
                                           TraversalOrder order,
                                           WeightEncoderCache& weightEncoderCache,
                                           Plans& plans,
                                           uint32_t numWeightStripes) const
{
    auto lifetime = info.m_Lifetime;
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
                    AddMceToOpGraph(opGraph, lifetime, order, info.m_MceCompute, info.m_Memory, numMemoryStripes,
                                    m_InputTensorShape, m_InputQuantizationInfo, convData, weightEncoderCache);

                auto pleInBuffer =
                    impl::AddPleInBuffer(opGraph, numPleInputStripes, m_OutputTensorShape,
                                         info.m_Memory.m_PleInput.m_Shape, m_OutputQuantizationInfo, lifetime, order);
                opGraph.SetProducer(pleInBuffer, inBufferAndMceOp.second);

                // Create an identity ple Op
                std::unique_ptr<PleOp> pleOp = std::make_unique<PleOp>(
                    Lifetime::Cascade, PleOperation::PASSTHROUGH, info.m_MceCompute.m_BlockConfig, 1,
                    std::vector<TensorShape>{ info.m_PleCompute.m_Input }, info.m_PleCompute.m_Output, m_DataType);
                auto outBufferAndPleOp = AddPleToOpGraph(opGraph, lifetime, order, info.m_Memory.m_Output.m_Shape,
                                                         numMemoryStripes, std::move(pleOp), m_OutputTensorShape,
                                                         m_OutputQuantizationInfo, m_CorrespondingOperationIds);
                opGraph.AddConsumer(pleInBuffer, outBufferAndPleOp.second, 0);
                inputMappings[inBufferAndMceOp.first]   = PartInputSlot{ m_PartId, 0 };
                outputMappings[outBufferAndPleOp.first] = PartOutputSlot{ m_PartId, 0 };
                AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph), plans);
            }
        }
    }
}

void McePart::CreateMceOnlyPlans(const impl::MceOnlyInfo& info,
                                 TraversalOrder order,
                                 WeightEncoderCache& weightEncoderCache,
                                 Plans& plans,
                                 uint32_t numWeightStripes) const
{
    auto lifetime = info.m_Lifetime;
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
                AddMceToOpGraph(opGraph, lifetime, order, info.m_MceCompute, info.m_Memory, numMemoryStripes,
                                m_InputTensorShape, m_InputQuantizationInfo, convData, weightEncoderCache);
            // We need to add the output buffer first before adding mce to opgraph as it uses it.
            auto outBuffer =
                impl::AddPleInBuffer(opGraph, numPleInputStripes, m_OutputTensorShape, info.m_Memory.m_PleInput.m_Shape,
                                     m_OutputQuantizationInfo, lifetime, order);
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

    // Fully connected only supports 8x8 block configs
    const std::vector<BlockConfig> blockConfigs = [&]() -> std::vector<BlockConfig> {
        if (m_Operation == command_stream::MceOperation::FULLY_CONNECTED)
        {
            return { 8u, 8u };
        }
        else
        {
            return { { 16u, 16u },
                     { 16u, 8u },
                     { 8u, 16u },
                     { 8u, 8u },
                     {
                         32u,
                         8u,
                     },
                     { 8u, 32u } };
        }
    }();
    StripeInfos stripeInfos;
    for (auto&& blockConfig : blockConfigs)
    {
        // Todo generate all stripes again
        m_StripeGenerator.GenerateStripes(blockConfig, CascadeType::Lonely, &stripeInfos);
    }

    for (const MceAndPleInfo& i : stripeInfos.m_MceAndPleInfos)
    {
        CreateMceAndIdentityPlePlans(i, TraversalOrder::Xyz, m_WeightEncoderCache, ret, numWeightStripes);
    }

    return ret;
}

Plans McePart::GetBeginningPlans(uint32_t numWeightStripes) const
{
    Plans ret;

    // Fully connected only supports 8x8 block configs
    const std::vector<BlockConfig> blockConfigs = [&]() -> std::vector<BlockConfig> {
        if (m_Operation == command_stream::MceOperation::FULLY_CONNECTED)
        {
            return { 8u, 8u };
        }
        else
        {
            return { { 16u, 16u },
                     { 16u, 8u },
                     { 8u, 16u },
                     { 8u, 8u },
                     {
                         32u,
                         8u,
                     },
                     { 8u, 32u } };
        }
    }();
    StripeInfos stripeInfos;
    for (auto&& blockConfig : blockConfigs)
    {
        // Todo generate all stripes again
        m_StripeGenerator.GenerateStripes(blockConfig, CascadeType::Beginning, &stripeInfos);
    }

    for (const MceAndPleInfo& i : stripeInfos.m_MceAndPleInfos)
    {
        CreateMceAndIdentityPlePlans(i, TraversalOrder::Xyz, m_WeightEncoderCache, ret, numWeightStripes);
    }
    for (const MceOnlyInfo& i : stripeInfos.m_MceOnlyInfos)
    {
        CreateMceOnlyPlans(i, TraversalOrder::Xyz, m_WeightEncoderCache, ret, numWeightStripes);
    }

    return ret;
}

Plans McePart::GetMiddlePlans(ethosn::command_stream::BlockConfig blockConfig,
                              Buffer* sramBuffer,
                              uint32_t numWeightStripes) const
{
    assert(sramBuffer);
    Plans ret;

    uint32_t kernelHeight = m_WeightsInfo.m_Dimensions[0];
    uint32_t kernelWidth  = m_WeightsInfo.m_Dimensions[1];

    uint32_t strideMultiplier = m_Stride.m_X * m_Stride.m_Y;

    if (!IsSramBufferValid(kernelHeight, kernelWidth, sramBuffer))
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
        kernelWidth, strideMultiplier, blockConfig, CascadeType::Middle);

    if (!stripeInfos.has_value())
    {
        return ret;
    }

    CreateMceAndIdentityPlePlans(stripeInfos.value().first, TraversalOrder::Xyz, m_WeightEncoderCache, ret,
                                 numWeightStripes);
    CreateMceOnlyPlans(stripeInfos.value().second, TraversalOrder::Xyz, m_WeightEncoderCache, ret, numWeightStripes);
    return ret;
}

Plans McePart::GetEndPlans(ethosn::command_stream::BlockConfig blockConfig,
                           Buffer* sramBuffer,
                           uint32_t numWeightStripes) const
{

    assert(sramBuffer);
    Plans ret;

    uint32_t kernelHeight = m_WeightsInfo.m_Dimensions[0];
    uint32_t kernelWidth  = m_WeightsInfo.m_Dimensions[1];

    uint32_t strideMultiplier = m_Stride.m_X * m_Stride.m_Y;

    if (!IsSramBufferValid(kernelHeight, kernelWidth, sramBuffer))
    {
        return ret;
    }

    NumStripesGrouped numStripes;
    numStripes.m_Input    = { sramBuffer->m_NumStripes, sramBuffer->m_NumStripes };
    numStripes.m_Output   = { 1, 2 };
    numStripes.m_Weights  = { numWeightStripes, numWeightStripes };
    numStripes.m_PleInput = { 0, 0 };

    bool isDepthwise = m_Operation == ethosn::command_stream::MceOperation::DEPTHWISE_CONVOLUTION;
    auto stripeInfos = GenerateContinueSectionStripeInfos(numStripes, sramBuffer, numWeightStripes, isDepthwise,
                                                          m_Capabilities, m_OutputTensorShape, kernelHeight,
                                                          kernelWidth, strideMultiplier, blockConfig, CascadeType::End);

    if (!stripeInfos.has_value())
    {
        return ret;
    }

    CreateMceAndIdentityPlePlans(stripeInfos.value().first, TraversalOrder::Xyz, m_WeightEncoderCache, ret,
                                 numWeightStripes);

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

ethosn::support_library::DotAttributes McePart::GetDotAttributes(DetailLevel detail) const
{
    DotAttributes result = BasePart::GetDotAttributes(detail);
    result.m_Label       = "McePart: " + result.m_Label;
    if (detail >= DetailLevel::High)
    {
        result.m_Label += "InputTensorShape = " + ToString(m_InputTensorShape) + "\n";
        result.m_Label += "OutputTensorShape = " + ToString(m_OutputTensorShape) + "\n";
        result.m_Label += "InputQuantizationInfo = " + ToString(m_InputQuantizationInfo) + "\n";
        result.m_Label += "OutputQuantizationInfo = " + ToString(m_OutputQuantizationInfo) + "\n";
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
        result.m_Label += "StripeGenerator.Stride = " + ToString(m_StripeGenerator.m_Stride) + "\n";
        result.m_Label += "StripeGenerator.UpscaleFactor = " + ToString(m_StripeGenerator.m_UpscaleFactor) + "\n";
        result.m_Label += "StripeGenerator.Operation = " + ToString(m_StripeGenerator.m_Operation) + "\n";
        result.m_Label += "StripeGenerator.ShapeMultiplier = " + ToString(m_StripeGenerator.m_ShapeMultiplier) + "\n";
    }
    return result;
}

}    // namespace support_library
}    // namespace ethosn
