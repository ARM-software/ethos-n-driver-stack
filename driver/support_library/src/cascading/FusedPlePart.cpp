//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "FusedPlePart.hpp"
#include "PartUtils.hpp"
#include "Plan.hpp"
#include "StripeHelper.hpp"

#include <ethosn_utils/Macros.hpp>

#include <memory>

using namespace ethosn::command_stream;

namespace ethosn
{
namespace support_library
{
using namespace impl;
using namespace utils;

FusedPlePart::FusedPlePart(PartId id,
                           const TensorShape& inputTensorShape,
                           const TensorShape& outputTensorShape,
                           const QuantizationInfo& inputQuantizationInfo,
                           const QuantizationInfo& outputQuantizationInfo,
                           command_stream::PleOperation op,
                           utils::ShapeMultiplier shapeMultiplier,
                           const EstimationOptions& estOpt,
                           const CompilationOptions& compOpt,
                           const HardwareCapabilities& capabilities,
                           std::set<uint32_t> correspondingOperationIds,
                           command_stream::DataType dataType)
    : BasePart(id, "FusedPlePart", CompilerDataFormat::NONE, correspondingOperationIds, estOpt, compOpt, capabilities)
    , m_InputTensorShape(inputTensorShape)
    , m_OutputTensorShape(outputTensorShape)
    , m_InputQuantizationInfo(inputQuantizationInfo)
    , m_OutputQuantizationInfo(outputQuantizationInfo)
    , m_KernelOperation(op)
    , m_ShapeMultiplier(shapeMultiplier)
    , m_StripeGenerator(m_InputTensorShape,
                        m_InputTensorShape,
                        m_OutputTensorShape,
                        1,
                        1,
                        0,
                        0,
                        Stride{ 1, 1 },
                        1,
                        MceOperation::DEPTHWISE_CONVOLUTION,
                        op,
                        ShapeMultiplier::Identity,
                        shapeMultiplier,
                        capabilities,
                        GetDefaultStripeConfig(m_DebugTag.c_str()))
    , m_WeightEncoderCache{ capabilities }
    , m_DataType(dataType)
{
    m_StripeGenerator.m_StripeConfig.blockConfigs =
        FilterPleBlockConfigs(m_KernelOperation, m_StripeGenerator.m_StripeConfig.blockConfigs);
}

utils::Optional<ethosn::command_stream::MceOperation> FusedPlePart::GetMceOperation() const
{
    return {};
}

bool FusedPlePart::CanDoubleBufferWeights() const
{
    return true;
}

Buffer* FusedPlePart::AddIdentityWeights(OwnedOpGraph& opGraph,
                                         const impl::MceStripesInfo& mceComputeInfo,
                                         const impl::NumStripesType& numMemoryWeightStripes,
                                         const TensorShape& memoryWeightStripe,
                                         const impl::ConvData& convData,
                                         WeightEncoderCache& weightEncoderCache) const
{
    // Encode weights
    const uint32_t weightStripeSize = mceComputeInfo.m_Weight[2];
    const uint32_t weightStripeDepth =
        GetWeightStripeDepth(convData.weightInfo, mceComputeInfo.m_Weight, Stride{ 1, 1 });

    WeightEncoderCache::Params wp;
    wp.weightsTensorInfo     = convData.weightInfo;
    wp.weightsData           = convData.weightData;
    wp.biasTensorInfo        = convData.biasInfo;
    wp.biasData              = convData.biasData;
    wp.inputQuantizationInfo = m_InputQuantizationInfo;
    // An identity convolution is being added and hence, the Input/Output quantization information should be the same.
    wp.outputQuantizationInfo = m_InputQuantizationInfo;
    wp.stripeDepth            = weightStripeDepth;
    wp.strideY                = 1;
    wp.strideX                = 1;
    wp.paddingTop             = 0;
    wp.paddingLeft            = 0;
    wp.iterationSize          = weightStripeSize;
    wp.operation              = MceOperation::DEPTHWISE_CONVOLUTION;
    wp.algorithm              = CompilerMceAlgorithm::Direct;
    auto encodedWeights       = weightEncoderCache.Encode(wp);

    CascadingBufferFormat formatInDram = impl::GetCascadingBufferFormatFromCompilerDataFormat(
        ConvertExternalToCompilerDataFormat(convData.weightInfo.m_DataFormat));
    Buffer* dramWeightBuffer =
        opGraph.AddBuffer(std::make_unique<Buffer>(Location::Dram, formatInDram, TraversalOrder::Xyz));
    dramWeightBuffer->m_TensorShape      = convData.weightInfo.m_Dimensions;
    dramWeightBuffer->m_EncodedWeights   = std::move(encodedWeights);
    dramWeightBuffer->m_SizeInBytes      = static_cast<uint32_t>(dramWeightBuffer->m_EncodedWeights->m_Data.size());
    dramWeightBuffer->m_QuantizationInfo = convData.weightInfo.m_QuantizationInfo;
    dramWeightBuffer->m_BufferType       = BufferType::ConstantDma;

    CascadingBufferFormat formatInSram = GetCascadingBufferFormatFromCompilerDataFormat(CompilerDataFormat::WEIGHT);
    Buffer* sramWeightBuffer =
        opGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, formatInSram, TraversalOrder::Xyz));
    sramWeightBuffer->m_TensorShape      = dramWeightBuffer->m_TensorShape;
    sramWeightBuffer->m_StripeShape      = memoryWeightStripe;
    sramWeightBuffer->m_QuantizationInfo = convData.weightInfo.m_QuantizationInfo;
    sramWeightBuffer->m_NumStripes       = numMemoryWeightStripes;
    sramWeightBuffer->m_SizeInBytes      = dramWeightBuffer->m_EncodedWeights->m_MaxSize * numMemoryWeightStripes;
    sramWeightBuffer->m_SlotSizeInBytes  = dramWeightBuffer->m_EncodedWeights->m_MaxSize;

    Op* dmaOp             = opGraph.AddOp(std::make_unique<DmaOp>(CascadingBufferFormat::WEIGHT));
    dmaOp->m_OperationIds = m_CorrespondingOperationIds;

    opGraph.AddConsumer(dramWeightBuffer, dmaOp, 0);
    opGraph.SetProducer(sramWeightBuffer, dmaOp);

    // Use the encoded weights to determine the size of the sram and dram buffers
    return sramWeightBuffer;
}

std::pair<Buffer*, Buffer*> FusedPlePart::AddIdentityMceOpForSubGraph(OwnedOpGraph& opGraph,
                                                                      Lifetime lifetime,
                                                                      const impl::MceStripesInfo& mceComputeInfo,
                                                                      const impl::NumMemoryStripes& numMemoryStripes,
                                                                      const impl::MemoryStripesInfo& memoryStripes,
                                                                      const TensorShape& inpShape,
                                                                      const QuantizationInfo& inpQuantInfo,
                                                                      WeightEncoderCache& weightEncoderCache) const
{
    const OpGraph::BufferList& buffers = opGraph.GetBuffers();
    const OpGraph::OpList& ops         = opGraph.GetOps();

    const float weightScale = 0.5f;
    const float biasScale   = weightScale * inpQuantInfo.GetScale();
    const uint32_t numIfm   = inpShape[3];

    TensorInfo weightInfo{ { 1, 1, numIfm, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM, { 0, weightScale } };
    TensorInfo biasInfo{ { 1, 1, 1, numIfm }, DataType::INT32_QUANTIZED, DataFormat::NHWC, { 0, biasScale } };

    std::shared_ptr<std::vector<uint8_t>> weightsData = std::make_shared<std::vector<uint8_t>>(1 * 1 * 1 * numIfm, 2);
    std::vector<int32_t> biasData(numIfm, 0);

    // Add input Buffer.
    opGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB, TraversalOrder::Zxy));
    Buffer* idMceOpInBuff = buffers.back();

    // Add Weight buffers and DmaOp.
    ConvData convData;
    convData.weightInfo      = weightInfo;
    convData.weightData      = weightsData;
    convData.biasInfo        = biasInfo;
    convData.biasData        = std::move(biasData);
    Buffer* weightSramBuffer = AddIdentityWeights(opGraph, mceComputeInfo, numMemoryStripes.m_Weight,
                                                  memoryStripes.m_Weight.m_Shape, convData, weightEncoderCache);

    int16_t lowerBound = m_DataType == command_stream::DataType::U8 ? 0 : -128;
    int16_t upperBound = m_DataType == command_stream::DataType::U8 ? 255 : 127;

    // Add MceOp.
    opGraph.AddOp(std::make_unique<MceOp>(lifetime, MceOperation::DEPTHWISE_CONVOLUTION, CompilerMceAlgorithm::Direct,
                                          mceComputeInfo.m_BlockConfig, mceComputeInfo.m_Input, mceComputeInfo.m_Output,
                                          mceComputeInfo.m_Weight, TraversalOrder::Xyz, Stride(1, 1), 0, 0, lowerBound,
                                          upperBound));
    Op* idMceOp             = ops.back();
    idMceOp->m_OperationIds = m_CorrespondingOperationIds;

    // Add Output Buffer.
    opGraph.AddBuffer(
        std::make_unique<Buffer>(Location::PleInputSram, CascadingBufferFormat::NHWCB, TraversalOrder::Xyz));
    Buffer* idMceOpOutBuff = buffers.back();

    opGraph.AddConsumer(idMceOpInBuff, idMceOp, 0);
    opGraph.AddConsumer(weightSramBuffer, idMceOp, 1);
    opGraph.SetProducer(idMceOpOutBuff, idMceOp);

    // Set Input & Output buffer shapes and sizes.
    idMceOpOutBuff->m_TensorShape = inpShape;
    idMceOpInBuff->m_TensorShape  = inpShape;
    idMceOpOutBuff->m_StripeShape = memoryStripes.m_PleInput.m_Shape;
    idMceOpInBuff->m_StripeShape  = memoryStripes.m_Input.m_Shape;
    idMceOpOutBuff->m_SizeInBytes = 0;    // The output buffer is in ple sram so has no size in the tile
    // Data could be de-compressed from FCAF
    constexpr bool couldSourceBeFcaf = true;
    std::tie(idMceOpInBuff->m_SlotSizeInBytes, idMceOpInBuff->m_SizeInBytes) =
        CalculateTileSize(m_Capabilities, inpShape, idMceOpInBuff->m_StripeShape,
                          memoryStripes.m_Input.m_PackedBoundaryThickness, numMemoryStripes.m_Input, couldSourceBeFcaf);
    idMceOpOutBuff->m_QuantizationInfo       = inpQuantInfo;
    idMceOpInBuff->m_QuantizationInfo        = inpQuantInfo;
    idMceOpOutBuff->m_NumStripes             = numMemoryStripes.m_PleInput;
    idMceOpInBuff->m_NumStripes              = numMemoryStripes.m_Input;
    idMceOpInBuff->m_PackedBoundaryThickness = memoryStripes.m_Input.m_PackedBoundaryThickness;
    idMceOpInBuff->m_NumLoads                = memoryStripes.m_Input.m_NumLoads;

    return { idMceOpInBuff, idMceOpOutBuff };
}

void FusedPlePart::CreateIdentityMceAndFusedPlePlans(const MceAndPleInfo& info,
                                                     WeightEncoderCache& weightEncoderCache,
                                                     Plans& plans,
                                                     uint32_t numWeightStripes) const
{
    auto lifetime = info.m_Lifetime;
    // Create plan with identity mce op and ple op
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
                auto mceInAndOutBuffer =
                    AddIdentityMceOpForSubGraph(opGraph, lifetime, info.m_MceCompute, numMemoryStripes, info.m_Memory,
                                                m_InputTensorShape, m_InputQuantizationInfo, weightEncoderCache);

                // A fuse only ple operation only has 1 input
                auto op = std::make_unique<PleOp>(lifetime, m_KernelOperation, info.m_PleCompute.m_BlockConfig, 1,
                                                  std::vector<TensorShape>{ info.m_PleCompute.m_Input },
                                                  info.m_PleCompute.m_Output, m_DataType, true);

                auto outBufferAndPleOp =
                    AddPleToOpGraph(opGraph, lifetime, info.m_Memory.m_Output.m_Shape, numMemoryStripes, std::move(op),
                                    m_OutputTensorShape, m_OutputQuantizationInfo, m_CorrespondingOperationIds);
                opGraph.AddConsumer(mceInAndOutBuffer.second, outBufferAndPleOp.second, 0);
                inputMappings[mceInAndOutBuffer.first]  = PartInputSlot{ m_PartId, 0 };
                outputMappings[outBufferAndPleOp.first] = PartOutputSlot{ m_PartId, 0 };
                AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph), plans, true, false);
            }
        }
    }
}

void FusedPlePart::CreateFuseOnlyPlans(const PleOnlyInfo& info, Plans& plans) const
{
    auto lifetime = info.m_Lifetime;
    for (auto numOutputStripes = info.m_Memory.m_Output.m_Range.m_Min;
         numOutputStripes <= info.m_Memory.m_Output.m_Range.m_Max; ++numOutputStripes)
    {
        for (auto numPleInputStripes = info.m_Memory.m_PleInput.m_Range.m_Min;
             numPleInputStripes <= info.m_Memory.m_PleInput.m_Range.m_Max; ++numPleInputStripes)
        {
            NumMemoryStripes numMemoryStripes;
            numMemoryStripes.m_Input    = 0;
            numMemoryStripes.m_Output   = numOutputStripes;
            numMemoryStripes.m_Weight   = 0;
            numMemoryStripes.m_PleInput = numPleInputStripes;
            OwnedOpGraph opGraph;
            PartInputMapping inputMappings;
            PartOutputMapping outputMappings;
            auto pleInBuffer =
                AddPleInBuffer(opGraph, numPleInputStripes, m_InputTensorShape, info.m_Memory.m_PleInput.m_Shape,
                               m_InputQuantizationInfo, Location::PleInputSram);

            // A fuse only ple operation only has 1 input
            auto op = std::make_unique<PleOp>(lifetime, m_KernelOperation, info.m_PleCompute.m_BlockConfig, 1,
                                              std::vector<TensorShape>{ info.m_PleCompute.m_Input },
                                              info.m_PleCompute.m_Output, m_DataType, true);

            auto outBufferAndPleOp =
                AddPleToOpGraph(opGraph, lifetime, info.m_Memory.m_Output.m_Shape, numMemoryStripes, std::move(op),
                                m_OutputTensorShape, m_OutputQuantizationInfo, m_CorrespondingOperationIds);
            opGraph.AddConsumer(pleInBuffer, outBufferAndPleOp.second, 0);
            inputMappings[pleInBuffer]              = PartInputSlot{ m_PartId, 0 };
            outputMappings[outBufferAndPleOp.first] = PartOutputSlot{ m_PartId, 0 };
            AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph), plans);
        }
    }
}

Plans FusedPlePart::GetLonelyPlans(uint32_t numWeightStripes) const
{
    Plans ret;

    // Try to generate plans as per Beginning of a section. This guarantees larger stripes
    // and helps to reduce overhead.
    // The estimation doesn't take into account overheads so we need to use this heuristic
    StripeInfos stripeInfos = m_StripeGenerator.GenerateStripes(CascadeType::Beginning);

    for (const MceAndPleInfo& i : stripeInfos.m_MceAndPleInfos)
    {
        CreateIdentityMceAndFusedPlePlans(i, m_WeightEncoderCache, ret, numWeightStripes);
    }

    // Don't continue if at least a plan is valid
    if (!ret.empty())
    {
        return ret;
    }

    // Generate all possible plans.
    stripeInfos = m_StripeGenerator.GenerateStripes(CascadeType::Lonely);

    for (const MceAndPleInfo& i : stripeInfos.m_MceAndPleInfos)
    {
        CreateIdentityMceAndFusedPlePlans(i, m_WeightEncoderCache, ret, numWeightStripes);
    }

    return ret;
}

Plans FusedPlePart::GetBeginningPlans(uint32_t numWeightStripes) const
{
    Plans ret;

    StripeInfos stripeInfos = m_StripeGenerator.GenerateStripes(CascadeType::Beginning);

    for (const MceAndPleInfo& i : stripeInfos.m_MceAndPleInfos)
    {
        CreateIdentityMceAndFusedPlePlans(i, m_WeightEncoderCache, ret, numWeightStripes);
    }

    return ret;
}

Plans FusedPlePart::GenerateContinueSectionPlans(ethosn::command_stream::BlockConfig blockConfig,
                                                 Buffer* prevBuffer,
                                                 uint32_t numWeightStripes,
                                                 CascadeType cascadeType) const
{
    assert(cascadeType == CascadeType::Middle || cascadeType == CascadeType::End);
    assert(prevBuffer);
    Plans ret;

    if (!PleBlockConfigAllowed(m_KernelOperation, blockConfig))
    {
        return ret;
    }

    // Multiple output stripes are needed because the follow layers may require multiple buffers due to boundary data.
    // These will be filtered out by the following layer
    bool fullHeight = GetHeight(prevBuffer->m_StripeShape) >= GetHeight(prevBuffer->m_TensorShape);
    bool fullWidth  = GetWidth(prevBuffer->m_StripeShape) >= GetWidth(prevBuffer->m_TensorShape);
    bool fullPlane  = fullHeight && fullWidth;
    // At the end of a cascde we can double buffer

    const TensorShape& inputStripeShape = prevBuffer->m_StripeShape;

    // Output stripe size = min (Tensor dimension * Multiplier for that dimension, Tensor dimension rounded up to multiple of block size).
    //For Width: shapeW rounded up to multiple of brickWidth
    //For Height: shapeH rounded up to multiple of brickHeight
    //For Depth: shapeC rounded up to multiple of Number of Output Generators
    utils::ShapeMultiplier shapeMult = m_ShapeMultiplier;
    TensorShape pleOutputStripe      = TensorShape{
        inputStripeShape[0],
        std::min(inputStripeShape[1] * shapeMult.m_H, utils::RoundUpToNearestMultiple(m_OutputTensorShape[1], 8U)),
        std::min(inputStripeShape[2] * shapeMult.m_W, utils::RoundUpToNearestMultiple(m_OutputTensorShape[2], 8U)),
        std::min(inputStripeShape[3] * shapeMult.m_C,
                 utils::RoundUpToNearestMultiple(m_OutputTensorShape[3], m_Capabilities.GetNumberOfOgs()))
    };

    uint32_t memoryOutputChannelsEncoding = GetChannels(pleOutputStripe);
    bool isEndOfCascade                   = cascadeType == CascadeType::End;
    if (fullPlane && !isEndOfCascade)
    {
        memoryOutputChannelsEncoding = 0;
    }
    TensorShape memoryOutputStripeEncoding{ 0, fullHeight ? 0 : GetHeight(pleOutputStripe),
                                            fullWidth ? 0 : GetWidth(pleOutputStripe), memoryOutputChannelsEncoding };
    // Sram buffer takes the Stripe shape of the preceding Ple Op.
    TensorShape memoryOutputStripe =
        CreateStripe(m_OutputTensorShape, memoryOutputStripeEncoding, m_Capabilities.GetBrickGroupShape()[3]);
    bool fullDepth  = memoryOutputStripe[3] >= m_OutputTensorShape[3];
    bool fullTensor = fullPlane && fullDepth;

    // Do not generate Middle or End Plans, if there is a MAXPOOL_3x3_2_2 Ple Operation without a full tensor.
    if ((m_KernelOperation == command_stream::PleOperation::MAXPOOL_3X3_2_2_EVEN ||
         m_KernelOperation == command_stream::PleOperation::MAXPOOL_3X3_2_2_ODD) &&
        !fullTensor)
    {
        return ret;
    }

    uint32_t maxOutputStripes = 0;
    // strategy 0
    if (!fullPlane)
    {
        // if its the end of a cascade we can double buffer the output, if it's not we need to output up to 3 stripes for neighouring data.
        maxOutputStripes = isEndOfCascade ? 2 : 3;
    }
    // Strategy 1/3
    else if (isEndOfCascade && fullDepth)
    {
        assert(fullPlane);
        maxOutputStripes = 1;
    }
    else if (!isEndOfCascade)
    {
        assert(fullDepth);
        maxOutputStripes = 1;
    }
    else if (!fullDepth)
    {
        assert(fullPlane && isEndOfCascade);
        maxOutputStripes = 2;
    }

    NumStripes numStripesOutput = { 1, maxOutputStripes };
    const TensorShape mceInputStripe =
        TensorShape{ inputStripeShape[0], std::min(inputStripeShape[1], m_InputTensorShape[1]),
                     std::min(inputStripeShape[2], m_InputTensorShape[2]),
                     std::min(inputStripeShape[3], m_InputTensorShape[3]) };

    if (prevBuffer->m_Location == Location::Sram)
    {
        uint32_t kernelHeight = 1;
        uint32_t kernelWidth  = 1;

        if (prevBuffer->m_NumStripes != 1)
        {
            return ret;
        }

        NumStripes numStripesInput    = { prevBuffer->m_NumStripes, prevBuffer->m_NumStripes };
        NumStripes numStripesWeights  = { numWeightStripes, numWeightStripes };
        NumStripes numStripesPleInput = { 0, 0 };

        TensorShape mceOutputStripe    = mceInputStripe;
        TensorShape mceWeightStripe    = { kernelHeight, kernelWidth, mceInputStripe[3], 1 };
        TensorShape memoryWeightStripe = mceWeightStripe;

        command_stream::cascading::PackedBoundaryThickness packedBoundaryThickness = { 0, 0, 0, 0 };
        const uint32_t numIfmLoads                                                 = 1;
        const uint32_t numWeightLoads                                              = 1;

        MceAndPleInfo mceAndPleInfo;

        mceAndPleInfo.m_Lifetime = fullTensor ? Lifetime::Atomic : Lifetime::Cascade;

        mceAndPleInfo.m_MceCompute.m_Input       = mceInputStripe;
        mceAndPleInfo.m_MceCompute.m_Output      = mceOutputStripe;
        mceAndPleInfo.m_MceCompute.m_Weight      = mceWeightStripe;
        mceAndPleInfo.m_MceCompute.m_BlockConfig = blockConfig;
        mceAndPleInfo.m_PleCompute.m_Input       = mceInputStripe;
        mceAndPleInfo.m_PleCompute.m_Output      = pleOutputStripe;
        mceAndPleInfo.m_PleCompute.m_BlockConfig = blockConfig;

        mceAndPleInfo.m_Memory.m_Input    = { { numStripesInput, inputStripeShape },
                                           packedBoundaryThickness,
                                           numIfmLoads };
        mceAndPleInfo.m_Memory.m_Output   = { numStripesOutput, memoryOutputStripe };
        mceAndPleInfo.m_Memory.m_Weight   = { { numStripesWeights, memoryWeightStripe }, numWeightLoads };
        mceAndPleInfo.m_Memory.m_PleInput = { numStripesPleInput, mceOutputStripe };

        CreateIdentityMceAndFusedPlePlans(mceAndPleInfo, m_WeightEncoderCache, ret, numWeightStripes);
    }
    else if (prevBuffer->m_Location == Location::PleInputSram)
    {
        PleOnlyInfo pleOnlyInfo;

        pleOnlyInfo.m_Lifetime = fullTensor ? Lifetime::Atomic : Lifetime::Cascade;

        pleOnlyInfo.m_PleCompute.m_Input       = mceInputStripe;
        pleOnlyInfo.m_PleCompute.m_Output      = pleOutputStripe;
        pleOnlyInfo.m_PleCompute.m_BlockConfig = blockConfig;

        pleOnlyInfo.m_Memory.m_Input    = { { { 0, 0 }, { 0, 0, 0, 0 } }, { 0, 0, 0, 0 }, 0 };
        pleOnlyInfo.m_Memory.m_Output   = { numStripesOutput, memoryOutputStripe };
        pleOnlyInfo.m_Memory.m_Weight   = { { { 0, 0 }, { 0, 0, 0, 0 } }, 0 };
        pleOnlyInfo.m_Memory.m_PleInput = { { prevBuffer->m_NumStripes, prevBuffer->m_NumStripes }, inputStripeShape };
        CreateFuseOnlyPlans(pleOnlyInfo, ret);
    }

    return ret;
}    // namespace support_library

Plans FusedPlePart::GetPlans(CascadeType cascadeType,
                             ethosn::command_stream::BlockConfig blockConfig,
                             Buffer* prevBuffer,
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
            return GenerateContinueSectionPlans(blockConfig, prevBuffer, numWeightStripes, cascadeType);
        }
        case CascadeType::End:
        {
            return GenerateContinueSectionPlans(blockConfig, prevBuffer, numWeightStripes, cascadeType);
        }
        default:
        {
            ETHOSN_FAIL_MSG("Invalid cascade type");
            return Plans();
        }
    }
}

ethosn::support_library::DotAttributes FusedPlePart::GetDotAttributes(DetailLevel detail) const
{
    DotAttributes result = BasePart::GetDotAttributes(detail);
    if (detail >= DetailLevel::High)
    {
        result.m_Label += "InputTensorShape = " + ToString(m_InputTensorShape) + "\n";
        result.m_Label += "OutputTensorShape = " + ToString(m_OutputTensorShape) + "\n";
        result.m_Label += "InputQuantizationInfo = " + ToString(m_InputQuantizationInfo) + "\n";
        result.m_Label += "OutputQuantizationInfo = " + ToString(m_OutputQuantizationInfo) + "\n";
        result.m_Label += "KernelOperation = " + ToString(m_KernelOperation) + "\n";
        result.m_Label += "ShapeMultiplier = " + ToString(m_ShapeMultiplier) + "\n";

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
        result.m_Label +=
            "StripeGenerator.MceShapeMultiplier = " + ToString(m_StripeGenerator.m_MceShapeMultiplier) + "\n";
        result.m_Label +=
            "StripeGenerator.PleShapeMultiplier = " + ToString(m_StripeGenerator.m_PleShapeMultiplier) + "\n";
    }
    return result;
}

}    // namespace support_library
}    // namespace ethosn
