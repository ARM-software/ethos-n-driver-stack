//
// Copyright © 2021 Arm Limited.
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
    : BasePart(id, CompilerDataFormat::NONE, correspondingOperationIds, estOpt, compOpt, capabilities)
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
                        Stride{ 1, 1 },
                        1,
                        MceOperation::DEPTHWISE_CONVOLUTION,
                        shapeMultiplier,
                        capabilities)
    , m_WeightEncoderCache{ capabilities }
    , m_DataType(dataType)
{}

utils::Optional<ethosn::command_stream::MceOperation> FusedPlePart::GetMceOperation() const
{
    return {};
}

Buffer* FusedPlePart::AddIdentityWeights(OwnedOpGraph& opGraph,
                                         Lifetime lifetime,
                                         const impl::MceStripesInfo& mceComputeInfo,
                                         const impl::NumStripesType& numMemoryWeightStripes,
                                         const TensorShape& memoryWeightStripe,
                                         TraversalOrder order,
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

std::pair<Buffer*, Buffer*> FusedPlePart::AddIdentityMceOpForSubGraph(OwnedOpGraph& opGraph,
                                                                      Lifetime lifetime,
                                                                      const impl::MceStripesInfo& mceComputeInfo,
                                                                      const impl::NumMemoryStripes& numMemoryStripes,
                                                                      const impl::MemoryStripesInfo& memoryStripes,
                                                                      const TensorShape& inpShape,
                                                                      const QuantizationInfo& inpQuantInfo,
                                                                      TraversalOrder order,
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
    opGraph.AddBuffer(std::make_unique<Buffer>(lifetime, Location::Sram, CascadingBufferFormat::NHWCB, order));
    Buffer* idMceOpInBuff = buffers.back();

    // Add Weight buffers and DmaOp.
    ConvData convData;
    convData.weightInfo      = weightInfo;
    convData.weightData      = weightsData;
    convData.biasInfo        = biasInfo;
    convData.biasData        = std::move(biasData);
    Buffer* weightSramBuffer = AddIdentityWeights(opGraph, lifetime, mceComputeInfo, numMemoryStripes.m_Weight,
                                                  memoryStripes.m_Weight.m_Shape, order, convData, weightEncoderCache);

    // Add MceOp.
    opGraph.AddOp(std::make_unique<MceOp>(Lifetime::Cascade, MceOperation::DEPTHWISE_CONVOLUTION,
                                          CompilerMceAlgorithm::Direct, mceComputeInfo.m_BlockConfig,
                                          mceComputeInfo.m_Input, mceComputeInfo.m_Output, mceComputeInfo.m_Weight,
                                          order, Stride(1, 1), 0, 0));
    Op* idMceOp             = ops.back();
    idMceOp->m_OperationIds = m_CorrespondingOperationIds;

    // Add Output Buffer.
    opGraph.AddBuffer(std::make_unique<Buffer>(lifetime, Location::PleInputSram, CascadingBufferFormat::NHWCB, order));
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
    idMceOpInBuff->m_SizeInBytes =
        CalculateTileSize(m_Capabilities, inpShape, idMceOpInBuff->m_StripeShape, numMemoryStripes.m_Input);
    idMceOpOutBuff->m_QuantizationInfo = inpQuantInfo;
    idMceOpInBuff->m_QuantizationInfo  = inpQuantInfo;
    idMceOpOutBuff->m_NumStripes       = numMemoryStripes.m_PleInput;
    idMceOpInBuff->m_NumStripes        = numMemoryStripes.m_Input;

    return { idMceOpInBuff, idMceOpOutBuff };
}

void FusedPlePart::CreateIdentityMceAndFusedPlePlans(const MceAndPleInfo& info,
                                                     TraversalOrder order,
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
                                                m_InputTensorShape, m_InputQuantizationInfo, order, weightEncoderCache);

                // A fuse only ple operation only has 1 input
                auto op = std::make_unique<PleOp>(Lifetime::Cascade, m_KernelOperation, info.m_PleCompute.m_BlockConfig,
                                                  1, std::vector<TensorShape>{ info.m_PleCompute.m_Input },
                                                  info.m_PleCompute.m_Output, m_DataType);

                auto outBufferAndPleOp = AddPleToOpGraph(opGraph, lifetime, order, info.m_Memory.m_Output.m_Shape,
                                                         numMemoryStripes, std::move(op), m_OutputTensorShape,
                                                         m_OutputQuantizationInfo, m_CorrespondingOperationIds);
                opGraph.AddConsumer(mceInAndOutBuffer.second, outBufferAndPleOp.second, 0);
                inputMappings[mceInAndOutBuffer.first]  = PartInputSlot{ m_PartId, 0 };
                outputMappings[outBufferAndPleOp.first] = PartOutputSlot{ m_PartId, 0 };
                AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph), plans);
            }
        }
    }
}

void FusedPlePart::CreateFuseOnlyPlans(const PleOnlyInfo& info, TraversalOrder order, Plans& plans) const
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
                               m_InputQuantizationInfo, lifetime, order);

            // A fuse only ple operation only has 1 input
            auto op = std::make_unique<PleOp>(Lifetime::Cascade, m_KernelOperation, info.m_PleCompute.m_BlockConfig, 1,
                                              std::vector<TensorShape>{ info.m_PleCompute.m_Input },
                                              info.m_PleCompute.m_Output, m_DataType);

            auto outBufferAndPleOp = AddPleToOpGraph(opGraph, lifetime, order, info.m_Memory.m_Output.m_Shape,
                                                     numMemoryStripes, std::move(op), m_OutputTensorShape,
                                                     m_OutputQuantizationInfo, m_CorrespondingOperationIds);
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

    // Fully connected only supports 8x8 block configs
    const std::vector<BlockConfig> blockConfigs = { { 16u, 16u },
                                                    { 16u, 8u },
                                                    { 8u, 16u },
                                                    { 8u, 8u },
                                                    {
                                                        32u,
                                                        8u,
                                                    },
                                                    { 8u, 32u } };
    std::vector<BlockConfig> validBlockConfigs  = FilterPleBlockConfigs(m_KernelOperation, blockConfigs);

    if (validBlockConfigs.size() == 0)
    {
        throw InternalErrorException("Fused PLE part: no valid block size found");
    }

    StripeInfos stripeInfos;
    for (auto&& blockConfig : validBlockConfigs)
    {
        // Todo generate all stripes again
        m_StripeGenerator.GenerateStripes(blockConfig, CascadeType::Lonely, &stripeInfos);
    }

    for (const MceAndPleInfo& i : stripeInfos.m_MceAndPleInfos)
    {
        CreateIdentityMceAndFusedPlePlans(i, TraversalOrder::Xyz, m_WeightEncoderCache, ret, numWeightStripes);
    }

    return ret;
}

Plans FusedPlePart::GetBeginningPlans(uint32_t numWeightStripes) const
{
    Plans ret;

    // Fully connected only supports 8x8 block configs
    const std::vector<BlockConfig> blockConfigs = { { 16u, 16u },
                                                    { 16u, 8u },
                                                    { 8u, 16u },
                                                    { 8u, 8u },
                                                    {
                                                        32u,
                                                        8u,
                                                    },
                                                    { 8u, 32u } };

    std::vector<BlockConfig> validBlockConfigs = FilterPleBlockConfigs(m_KernelOperation, blockConfigs);

    if (validBlockConfigs.size() == 0)
    {
        throw InternalErrorException("Fused PLE part: no valid block size found");
    }

    StripeInfos stripeInfos;
    for (auto&& blockConfig : validBlockConfigs)
    {
        // Todo generate all stripes again
        m_StripeGenerator.GenerateStripes(blockConfig, CascadeType::Beginning, &stripeInfos);
    }

    for (const MceAndPleInfo& i : stripeInfos.m_MceAndPleInfos)
    {
        CreateIdentityMceAndFusedPlePlans(i, TraversalOrder::Xyz, m_WeightEncoderCache, ret, numWeightStripes);
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
    bool fullTensor = fullHeight && fullWidth;
    // At the end of a cascde we can double buffer

    const TensorShape& inputStripeShape = prevBuffer->m_StripeShape;

    utils::ShapeMultiplier shapeMult = m_ShapeMultiplier;
    TensorShape pleOutputStripe =
        TensorShape{ inputStripeShape[0], inputStripeShape[1] * shapeMult.m_H, inputStripeShape[2] * shapeMult.m_W,
                     inputStripeShape[3] * shapeMult.m_C };

    uint32_t memoryOutputChannelsEncoding = 0;
    bool isEndOfCascade                   = cascadeType == CascadeType::End;
    if (fullTensor && isEndOfCascade)
    {
        memoryOutputChannelsEncoding = m_Capabilities.GetNumberOfOgs();
    }
    TensorShape memoryOutputStripeEncoding{ 0, fullHeight ? 0 : GetHeight(pleOutputStripe),
                                            fullWidth ? 0 : GetWidth(pleOutputStripe), memoryOutputChannelsEncoding };
    TensorShape memoryOutputStripe =
        CreateStripe(m_OutputTensorShape, memoryOutputStripeEncoding, m_Capabilities.GetBrickGroupShape()[3]);

    bool fullDepth            = memoryOutputStripe[3] >= m_OutputTensorShape[3];
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

    NumStripes numStripesOutput = { 1, maxOutputStripes };

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

        const TensorShape& mceInputStripe = inputStripeShape;
        TensorShape mceOutputStripe       = mceInputStripe;
        TensorShape mceWeightStripe       = { kernelHeight, kernelWidth, mceInputStripe[3], 1 };
        TensorShape memoryWeightStripe    = mceWeightStripe;

        MceAndPleInfo mceAndPleInfo;
        mceAndPleInfo.m_MceCompute.m_Input       = prevBuffer->m_StripeShape;
        mceAndPleInfo.m_MceCompute.m_Output      = mceOutputStripe;
        mceAndPleInfo.m_MceCompute.m_Weight      = mceWeightStripe;
        mceAndPleInfo.m_MceCompute.m_BlockConfig = blockConfig;
        mceAndPleInfo.m_PleCompute.m_Input       = mceInputStripe;
        mceAndPleInfo.m_PleCompute.m_Output      = pleOutputStripe;
        mceAndPleInfo.m_PleCompute.m_BlockConfig = blockConfig;

        mceAndPleInfo.m_Memory.m_Input    = { numStripesInput, mceInputStripe };
        mceAndPleInfo.m_Memory.m_Output   = { numStripesOutput, memoryOutputStripe };
        mceAndPleInfo.m_Memory.m_Weight   = { numStripesWeights, memoryWeightStripe };
        mceAndPleInfo.m_Memory.m_PleInput = { numStripesPleInput, mceOutputStripe };

        CreateIdentityMceAndFusedPlePlans(mceAndPleInfo, TraversalOrder::Xyz, m_WeightEncoderCache, ret,
                                          numWeightStripes);
    }
    else if (prevBuffer->m_Location == Location::PleInputSram)
    {
        PleOnlyInfo pleOnlyInfo;
        pleOnlyInfo.m_PleCompute.m_Input       = inputStripeShape;
        pleOnlyInfo.m_PleCompute.m_Output      = pleOutputStripe;
        pleOnlyInfo.m_PleCompute.m_BlockConfig = blockConfig;

        pleOnlyInfo.m_Memory.m_Input    = { { 0, 0 }, { 0, 0, 0, 0 } };
        pleOnlyInfo.m_Memory.m_Output   = { numStripesOutput, memoryOutputStripe };
        pleOnlyInfo.m_Memory.m_Weight   = { { 0, 0 }, { 0, 0, 0, 0 } };
        pleOnlyInfo.m_Memory.m_PleInput = { { prevBuffer->m_NumStripes, prevBuffer->m_NumStripes }, inputStripeShape };
        CreateFuseOnlyPlans(pleOnlyInfo, TraversalOrder::Xyz, ret);
    }

    return ret;
}

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
    result.m_Label       = "FusedPlePart: " + result.m_Label;
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
        result.m_Label += "StripeGenerator.ShapeMultiplier = " + ToString(m_StripeGenerator.m_ShapeMultiplier) + "\n";
    }
    return result;
}

}    // namespace support_library
}    // namespace ethosn
