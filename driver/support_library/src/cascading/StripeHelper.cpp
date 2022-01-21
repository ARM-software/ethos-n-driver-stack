//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "PartUtils.hpp"
#include "StripeHelper.hpp"
#include "WeightEncoderCache.hpp"

namespace ethosn
{
namespace support_library
{
namespace impl
{

/// Generates a stripe shape given an encoding and an input tensor
/// Tries to create a stripe with the stripe shape in the encoding, if the dimension is 0 then it uses the full length of that dimension.
TensorShape CreateStripe(TensorShape input, TensorShape inputEncoding, uint32_t channelsRounding)
{
    TensorShape inputStripeShape;
    for (uint32_t i = 0; i < input.size(); ++i)
    {
        inputStripeShape[i] = inputEncoding[i] != 0 ? inputEncoding[i] : input[i];
        inputStripeShape[i] = std::min(inputStripeShape[i], input[i]);
    }
    inputStripeShape    = utils::RoundUpHeightAndWidthToBrickGroup(inputStripeShape);
    inputStripeShape[3] = utils::RoundUpToNearestMultiple(inputStripeShape[3], channelsRounding);
    return inputStripeShape;
}

StripeGenerator::StripeGenerator(const TensorShape& mceInput,
                                 const TensorShape& mceOutput,
                                 const TensorShape& pleOutput,
                                 uint32_t kernelHeight,
                                 uint32_t kernelWidth,
                                 const Stride& stride,
                                 uint32_t upscaleFactor,
                                 command_stream::MceOperation op,
                                 utils::ShapeMultiplier mceShapeMult,
                                 utils::ShapeMultiplier pleShapeMult,
                                 const HardwareCapabilities& m_Capabilities)
    : m_MceInputTensorShape(mceInput)
    , m_MceOutputTensorShape(mceOutput)
    , m_PleOutputTensorShape(pleOutput)
    , m_KernelHeight(kernelHeight)
    , m_KernelWidth(kernelWidth)
    , m_Stride(stride)
    , m_UpscaleFactor(upscaleFactor)
    , m_Operation(op)
    , m_MceShapeMultiplier(mceShapeMult)
    , m_PleShapeMultiplier(pleShapeMult)
    , m_Capabilities(m_Capabilities)
{}

void StripeGenerator::CreateNumStripes(CascadeType cascadeType,
                                       uint32_t kernelHeight,
                                       NumStripes& numStripesInput,
                                       NumStripes& numStripesOutput,
                                       NumStripes& numStripesWeights,
                                       NumStripes& numStripesPleInput) const
{
    // MceOperations output to PLE SRAM so are no "stripes"
    // At least 3 input stripes are needed because of
    // data on the top and bottom. Weights can
    // have 1 or 2 for double buffering.
    switch (cascadeType)
    {
        case CascadeType::Beginning:
        {
            if (kernelHeight == 1)
            {
                numStripesInput = { 1, 2 };
            }
            else
            {
                numStripesInput = { 3, 4 };
            }
            // Multiple output stripes are needed because the follow layers may require multiple buffers due to boundary data.
            // These will be filtered out by the following layer.
            numStripesOutput   = { 1, 3 };
            numStripesWeights  = { 1, 2 };
            numStripesPleInput = { 0, 0 };
            break;
        }
        case CascadeType::Lonely:
        {
            if (kernelHeight == 1)
            {
                numStripesInput = { 1, 2 };
            }
            else
            {
                numStripesInput = { 3, 4 };
            }
            numStripesOutput   = { 1, 2 };
            numStripesWeights  = { 1, 2 };
            numStripesPleInput = { 0, 0 };
            break;
        }
        default:
        {
            ETHOSN_FAIL_MSG("invalid cascade type");
            break;
        }
    }
}

void StripeGenerator::GenerateStripes(ethosn::command_stream::BlockConfig blockConfig,
                                      CascadeType cascadeType,
                                      StripeInfos* outStripeInfos) const
{
    using namespace utils;
    assert(outStripeInfos);

    uint32_t strideMultiplier  = 1U;
    bool isDepthwise           = false;
    TensorShape mceOutputShape = {};
    // MceOperations output to PLE SRAM so are no "stripes"
    // At least 3 input stripes are needed because of
    // data on the top and bottom. Weights can
    // have 1 or 2 for double buffering.
    NumStripes numStripesInput;
    NumStripes numStripesOutput;
    NumStripes numStripesWeights;
    NumStripes numStripesPleInput;
    CreateNumStripes(cascadeType, m_KernelHeight, numStripesInput, numStripesOutput, numStripesWeights,
                     numStripesPleInput);
    strideMultiplier = m_Stride.m_X * m_Stride.m_Y;
    isDepthwise      = m_Operation == ethosn::command_stream::MceOperation::DEPTHWISE_CONVOLUTION;
    mceOutputShape   = m_MceOutputTensorShape;

    auto ApplyMceShapeMult = [&](TensorShape shape) {
        return TensorShape{ shape[0], shape[1] * m_MceShapeMultiplier.m_H, shape[2] * m_MceShapeMultiplier.m_W,
                            shape[3] * m_MceShapeMultiplier.m_C };
    };
    auto ApplyPleShapeMult = [&](TensorShape shape) {
        return TensorShape{ shape[0], shape[1] * m_PleShapeMultiplier.m_H, shape[2] * m_PleShapeMultiplier.m_W,
                            shape[3] * m_PleShapeMultiplier.m_C };
    };

    auto AddStripeInfos = [&](const TensorShape& mceInputStripe, const TensorShape& mceOutputStripe,
                              const TensorShape& pleInputStripe, const TensorShape& pleOutputStripe,
                              const NumStripes& inputRange, const NumStripes& outputRange,
                              const NumStripes& weightRange, const NumStripes& pleInputRange,
                              const TensorShape& memoryInputStripe, const TensorShape& memoryOutputStripe,
                              const TensorShape& memoryPleInputStripe, const TensorShape& inputShape,
                              const TensorShape& outputShape) {
        // Limit the max number of stripes based on the size of the tensor - there is no point considering plans where
        // we can store more stripes in the tile than there are in the tensor!
        NumStripes inputCopy = inputRange;
        inputCopy.m_Max =
            std::min(inputCopy.m_Max, DivRoundUp(GetHeight(inputShape), GetHeight(memoryInputStripe)) *
                                          DivRoundUp(GetWidth(inputShape), GetWidth(memoryInputStripe)) *
                                          DivRoundUp(GetChannels(inputShape), GetChannels(memoryInputStripe)));
        NumStripes outputCopy = outputRange;
        outputCopy.m_Max =
            std::min(outputCopy.m_Max, DivRoundUp(GetHeight(outputShape), GetHeight(memoryOutputStripe)) *
                                           DivRoundUp(GetWidth(outputShape), GetWidth(memoryOutputStripe)) *
                                           DivRoundUp(GetChannels(outputShape), GetChannels(memoryOutputStripe)));

        // Prevent using stripes which have more elements than the entire tensor
        bool multipleStripes         = inputCopy.m_Max > 1 && outputCopy.m_Max > 1;
        bool stripesLargerThanTensor = utils::GetNumElements(memoryInputStripe) > utils::GetNumElements(inputShape) &&
                                       utils::GetNumElements(memoryOutputStripe) > utils::GetNumElements(outputShape);
        if (multipleStripes && stripesLargerThanTensor)
        {
            return;
        }
        TensorShape mceWeightStripe    = { m_KernelHeight, m_KernelWidth, mceInputStripe[3],
                                        isDepthwise ? 1 : mceOutputStripe[3] };
        TensorShape memoryWeightStripe = mceWeightStripe;
        NumStripes weightCopy          = weightRange;
        if (isDepthwise)
        {
            if (memoryWeightStripe[2] >= m_MceInputTensorShape[3])
            {
                weightCopy.m_Max = 1;
            }
        }
        else
        {
            if (memoryWeightStripe[3] >= mceOutputShape[3])
            {
                weightCopy.m_Max = 1;
            }
        }
        {
            MceAndPleInfo mceAndPleInfo;

            mceAndPleInfo.m_MceCompute.m_Input       = mceInputStripe;
            mceAndPleInfo.m_MceCompute.m_Output      = mceOutputStripe;
            mceAndPleInfo.m_MceCompute.m_Weight      = mceWeightStripe;
            mceAndPleInfo.m_MceCompute.m_BlockConfig = blockConfig;
            mceAndPleInfo.m_PleCompute.m_Input       = pleInputStripe;
            mceAndPleInfo.m_PleCompute.m_Output      = pleOutputStripe;
            mceAndPleInfo.m_PleCompute.m_BlockConfig = blockConfig;

            mceAndPleInfo.m_Memory.m_Input    = { inputCopy, memoryInputStripe };
            mceAndPleInfo.m_Memory.m_Output   = { outputCopy, memoryOutputStripe };
            mceAndPleInfo.m_Memory.m_Weight   = { weightCopy, memoryWeightStripe };
            mceAndPleInfo.m_Memory.m_PleInput = { pleInputRange, memoryPleInputStripe };
            outStripeInfos->m_MceAndPleInfos.insert(mceAndPleInfo);
        }
        {
            MceOnlyInfo mceOnlyInfo;

            mceOnlyInfo.m_MceCompute.m_Input       = mceInputStripe;
            mceOnlyInfo.m_MceCompute.m_Output      = mceOutputStripe;
            mceOnlyInfo.m_MceCompute.m_Weight      = mceWeightStripe;
            mceOnlyInfo.m_MceCompute.m_BlockConfig = blockConfig;

            mceOnlyInfo.m_Memory.m_Input    = { inputCopy, memoryInputStripe };
            mceOnlyInfo.m_Memory.m_Output   = { { 0, 0 }, { 0, 0, 0, 0 } };
            mceOnlyInfo.m_Memory.m_Weight   = { weightCopy, memoryWeightStripe };
            mceOnlyInfo.m_Memory.m_PleInput = { pleInputRange, memoryPleInputStripe };
            outStripeInfos->m_MceOnlyInfos.insert(mceOnlyInfo);
        }
        {
            PleOnlyInfo pleOnlyInfo;

            pleOnlyInfo.m_PleCompute.m_Input       = pleInputStripe;
            pleOnlyInfo.m_PleCompute.m_Output      = pleOutputStripe;
            pleOnlyInfo.m_PleCompute.m_BlockConfig = blockConfig;

            pleOnlyInfo.m_Memory.m_Input    = { { 0, 0 }, { 0, 0, 0, 0 } };
            pleOnlyInfo.m_Memory.m_Output   = { outputCopy, memoryOutputStripe };
            pleOnlyInfo.m_Memory.m_Weight   = { { 0, 0 }, { 0, 0, 0, 0 } };
            pleOnlyInfo.m_Memory.m_PleInput = { pleInputRange, memoryPleInputStripe };
            outStripeInfos->m_PleOnlyInfos.insert(pleOnlyInfo);
        }
        {
            DmaOnlyInfo dmaOnlyInfo;
            dmaOnlyInfo.m_Input  = { inputCopy, memoryInputStripe };
            dmaOnlyInfo.m_Output = { outputCopy, memoryOutputStripe };
            outStripeInfos->m_DmaOnlyInfos.insert(dmaOnlyInfo);
        }
    };

    // Use the minimum stripe size possible to minimize the time before processing
    // Try splitting height first
    {
        TensorShape mceInputEncoding  = { 0, blockConfig.m_BlockHeight(), 0, 0 };
        const TensorShape& inputShape = m_MceInputTensorShape;
        TensorShape mceInputStripe =
            CreateStripe(m_MceInputTensorShape, mceInputEncoding, m_Capabilities.GetBrickGroupShape()[3]);

        TensorShape mceOutputEncoding = ApplyMceShapeMult(mceInputEncoding);
        TensorShape mceOutputStripe = CreateStripe(mceOutputShape, mceOutputEncoding, m_Capabilities.GetNumberOfOgs());

        TensorShape pleOutputStripe = ApplyPleShapeMult(mceOutputStripe);

        TensorShape pleOutputEncoding = ApplyPleShapeMult(mceOutputEncoding);
        TensorShape memoryOutputStripe =
            CreateStripe(m_PleOutputTensorShape, pleOutputEncoding, m_Capabilities.GetBrickGroupShape()[3]);
        const TensorShape& outputShape   = m_PleOutputTensorShape;
        NumStripes numStripesWeightsCopy = numStripesWeights;
        numStripesWeightsCopy.m_Min      = std::min(numStripesWeights.m_Min, 1u);
        numStripesWeightsCopy.m_Max      = std::min(numStripesWeights.m_Max, 1u);

        AddStripeInfos(mceInputStripe, mceOutputStripe, mceInputStripe, pleOutputStripe, numStripesInput,
                       numStripesOutput, numStripesWeightsCopy, numStripesPleInput, mceInputStripe, memoryOutputStripe,
                       mceOutputStripe, inputShape, outputShape);
    }

    // Split only input in height while the output is full tensor
    {
        TensorShape mceInputEncoding  = { 0, blockConfig.m_BlockHeight(), 0, 0 };
        const TensorShape& inputShape = m_MceInputTensorShape;
        TensorShape mceInputStripe =
            CreateStripe(m_MceInputTensorShape, mceInputEncoding, m_Capabilities.GetBrickGroupShape()[3]);

        TensorShape mceOutputEncoding = ApplyMceShapeMult(mceInputEncoding);
        TensorShape mceOutputStripe = CreateStripe(mceOutputShape, mceOutputEncoding, m_Capabilities.GetNumberOfOgs());

        TensorShape pleOutputStripe = ApplyPleShapeMult(mceOutputStripe);

        const TensorShape& outputShape   = m_PleOutputTensorShape;
        TensorShape memoryOutputEncoding = { 0, 0, 0, 0 };
        TensorShape memoryOutputStripe =
            CreateStripe(outputShape, memoryOutputEncoding, m_Capabilities.GetBrickGroupShape()[3]);
        NumStripes numStripesWeightsCopy = numStripesWeights;
        numStripesWeightsCopy.m_Min      = std::min(numStripesWeights.m_Min, 1u);
        numStripesWeightsCopy.m_Max      = std::min(numStripesWeights.m_Max, 1u);
        NumStripes numStripesOutputCopy  = numStripesOutput;
        numStripesOutputCopy.m_Min       = std::min(numStripesOutput.m_Min, 1u);
        numStripesOutputCopy.m_Max       = std::min(numStripesOutput.m_Max, 1u);

        AddStripeInfos(mceInputStripe, mceOutputStripe, mceInputStripe, pleOutputStripe, numStripesInput,
                       numStripesOutputCopy, numStripesWeightsCopy, numStripesPleInput, mceInputStripe,
                       memoryOutputStripe, mceOutputStripe, inputShape, outputShape);
    }

    // Try splitting width
    {
        TensorShape mceInputEncoding  = { 0, 0, blockConfig.m_BlockWidth(), 0 };
        const TensorShape& inputShape = m_MceInputTensorShape;
        TensorShape mceInputStripe =
            CreateStripe(m_MceInputTensorShape, mceInputEncoding, m_Capabilities.GetBrickGroupShape()[3]);

        TensorShape mceOutputEncoding = ApplyMceShapeMult(mceInputEncoding);
        TensorShape mceOutputStripe = CreateStripe(mceOutputShape, mceOutputEncoding, m_Capabilities.GetNumberOfOgs());

        TensorShape pleOutputStripe = ApplyPleShapeMult(mceOutputStripe);

        TensorShape pleOutputEncoding = ApplyPleShapeMult(mceOutputEncoding);
        TensorShape memoryOutputStripe =
            CreateStripe(m_PleOutputTensorShape, pleOutputEncoding, m_Capabilities.GetBrickGroupShape()[3]);
        const TensorShape& outputShape = m_PleOutputTensorShape;

        NumStripes numStripesInputCopy = numStripesInput;

        if (m_KernelWidth == 1)
        {
            numStripesInputCopy.m_Min = 1;
            numStripesInputCopy.m_Max = 2;
        }

        NumStripes numStripesWeightCopy = numStripesWeights;
        numStripesWeightCopy.m_Min      = std::min(numStripesWeights.m_Min, 1u);
        numStripesWeightCopy.m_Max      = std::min(numStripesWeights.m_Max, 1u);

        AddStripeInfos(mceInputStripe, mceOutputStripe, mceInputStripe, pleOutputStripe, numStripesInputCopy,
                       numStripesOutput, numStripesWeightCopy, numStripesPleInput, mceInputStripe, memoryOutputStripe,
                       mceOutputStripe, inputShape, outputShape);
    }

    const uint32_t blockWidthMultiplier  = std::max(1U, GetWidth(m_MceInputTensorShape) / blockConfig.m_BlockWidth());
    const uint32_t blockHeightMultiplier = std::max(1U, GetHeight(m_MceInputTensorShape) / blockConfig.m_BlockHeight());
    if (cascadeType == CascadeType::Lonely)
    {
        for (uint32_t heightMultiplier = 1; heightMultiplier <= blockHeightMultiplier; ++heightMultiplier)
        {
            for (uint32_t widthMultiplier = 1; widthMultiplier <= blockWidthMultiplier; ++widthMultiplier)
            {
                // Try splitting width and height
                {
                    TensorShape mceInputEncoding  = { 0, heightMultiplier * blockConfig.m_BlockHeight(),
                                                     widthMultiplier * blockConfig.m_BlockWidth(), 0 };
                    const TensorShape& inputShape = m_MceInputTensorShape;
                    TensorShape mceInputStripe =
                        CreateStripe(m_MceInputTensorShape, mceInputEncoding, m_Capabilities.GetBrickGroupShape()[3]);

                    TensorShape mceOutputEncoding = ApplyMceShapeMult(mceInputEncoding);
                    TensorShape mceOutputStripe =
                        CreateStripe(mceOutputShape, mceOutputEncoding, m_Capabilities.GetNumberOfOgs());

                    TensorShape pleOutputStripe = ApplyPleShapeMult(mceOutputStripe);

                    TensorShape pleOutputEncoding = ApplyPleShapeMult(mceOutputEncoding);
                    TensorShape memoryOutputStripe =
                        CreateStripe(m_PleOutputTensorShape, pleOutputEncoding, m_Capabilities.GetBrickGroupShape()[3]);

                    const TensorShape& outputShape = m_PleOutputTensorShape;
                    NumStripes numStripesInputCopy = numStripesInput;

                    if (m_KernelWidth == 1)
                    {
                        numStripesInputCopy.m_Min = 1;
                        numStripesInputCopy.m_Max = 2;
                    }

                    NumStripes numStripesWeightCopy = numStripesWeights;
                    numStripesWeightCopy.m_Min      = std::min(numStripesWeights.m_Min, 1u);
                    numStripesWeightCopy.m_Max      = std::min(numStripesWeights.m_Max, 1u);

                    AddStripeInfos(mceInputStripe, mceOutputStripe, mceOutputStripe, pleOutputStripe,
                                   numStripesInputCopy, numStripesOutput, numStripesWeightCopy, numStripesPleInput,
                                   mceInputStripe, memoryOutputStripe, mceOutputStripe, inputShape, outputShape);
                }

                // Try split height width and output depth
                {
                    const uint32_t height = heightMultiplier * blockConfig.m_BlockHeight();
                    const uint32_t width  = widthMultiplier * blockConfig.m_BlockWidth();

                    TensorShape mceInputEncoding  = { 0, height, width, 0 };
                    const TensorShape& inputShape = m_MceInputTensorShape;
                    TensorShape mceInputStripe =
                        CreateStripe(m_MceInputTensorShape, mceInputEncoding, m_Capabilities.GetBrickGroupShape()[3]);

                    TensorShape mceOutputEncoding =
                        ApplyMceShapeMult({ 0, height, width, m_Capabilities.GetNumberOfOgs() });
                    TensorShape mceOutputStripe =
                        CreateStripe(mceOutputShape, mceOutputEncoding, m_Capabilities.GetNumberOfOgs());

                    TensorShape pleOutputStripe = ApplyPleShapeMult(mceOutputStripe);

                    TensorShape pleOutputEncoding = ApplyPleShapeMult(mceOutputEncoding);
                    TensorShape memoryOutputStripe =
                        CreateStripe(m_PleOutputTensorShape, pleOutputEncoding, m_Capabilities.GetBrickGroupShape()[3]);
                    const TensorShape& outputShape = m_PleOutputTensorShape;

                    AddStripeInfos(mceInputStripe, mceOutputStripe, mceOutputStripe, pleOutputStripe, numStripesInput,
                                   numStripesOutput, numStripesWeights, numStripesPleInput, mceInputStripe,
                                   memoryOutputStripe, mceOutputStripe, inputShape, outputShape);
                }

                // Try split input depth
                // note we have to limit the height and width to the block size
                {
                    TensorShape mceInputEncoding  = { 0, blockConfig.m_BlockHeight(), blockConfig.m_BlockWidth(),
                                                     m_Capabilities.GetNumberOfOgs() * strideMultiplier };
                    const TensorShape& inputShape = m_MceInputTensorShape;
                    TensorShape mceInputStripe =
                        CreateStripe(m_MceInputTensorShape, mceInputEncoding, m_Capabilities.GetBrickGroupShape()[3]);

                    TensorShape mceOutputEncoding = ApplyMceShapeMult(mceInputEncoding);
                    TensorShape mceOutputStripe =
                        CreateStripe(mceOutputShape, mceOutputEncoding, m_Capabilities.GetNumberOfOgs());

                    TensorShape pleOutputStripe = ApplyPleShapeMult(mceOutputStripe);

                    TensorShape pleOutputEncoding = ApplyPleShapeMult(mceOutputEncoding);
                    TensorShape memoryOutputStripe =
                        CreateStripe(m_PleOutputTensorShape, pleOutputEncoding, m_Capabilities.GetBrickGroupShape()[3]);
                    const TensorShape& outputShape = m_PleOutputTensorShape;

                    AddStripeInfos(mceInputStripe, mceOutputStripe, mceOutputStripe, pleOutputStripe, numStripesInput,
                                   numStripesOutput, numStripesWeights, numStripesPleInput, mceInputStripe,
                                   memoryOutputStripe, mceOutputStripe, inputShape, outputShape);
                }
            }
        }
    }

    if (isDepthwise)
    {

        if (cascadeType == CascadeType::Lonely)
        {
            // Try split output depth
            {
                // With depthwise each only OFM needs 1 IFM
                TensorShape mceInputEncoding  = { 0, 0, 0, m_Capabilities.GetNumberOfOgs() };
                const TensorShape& inputShape = m_MceInputTensorShape;
                TensorShape mceInputStripe =
                    CreateStripe(m_MceInputTensorShape, mceInputEncoding, m_Capabilities.GetBrickGroupShape()[3]);

                TensorShape mceOutputEncoding = ApplyMceShapeMult({ 0, 0, 0, m_Capabilities.GetNumberOfOgs() });
                TensorShape mceOutputStripe =
                    CreateStripe(mceOutputShape, mceOutputEncoding, m_Capabilities.GetNumberOfOgs());

                TensorShape pleOutputStripe = ApplyPleShapeMult(mceOutputStripe);

                TensorShape pleOutputEncoding = ApplyPleShapeMult(mceOutputEncoding);
                TensorShape memoryOutputStripe =
                    CreateStripe(m_PleOutputTensorShape, pleOutputEncoding, m_Capabilities.GetBrickGroupShape()[3]);

                const TensorShape& outputShape = m_PleOutputTensorShape;

                AddStripeInfos(mceInputStripe, mceOutputStripe, mceInputStripe, pleOutputStripe, numStripesInput,
                               numStripesOutput, numStripesWeights, numStripesPleInput, mceInputStripe,
                               memoryOutputStripe, mceOutputStripe, inputShape, outputShape);
            }

            for (uint32_t heightMultiplier = 1; heightMultiplier <= blockHeightMultiplier; ++heightMultiplier)
            {
                // Try split height width and output depth
                for (uint32_t widthMultiplier = 1; widthMultiplier <= blockWidthMultiplier; ++widthMultiplier)
                {
                    const uint32_t height = heightMultiplier * blockConfig.m_BlockHeight();
                    const uint32_t width  = widthMultiplier * blockConfig.m_BlockWidth();

                    TensorShape mceInputEncoding  = { 0, height, width,
                                                     m_Capabilities.GetNumberOfOgs() * strideMultiplier };
                    const TensorShape& inputShape = m_MceInputTensorShape;
                    TensorShape mceInputStripe =
                        CreateStripe(m_MceInputTensorShape, mceInputEncoding, m_Capabilities.GetBrickGroupShape()[3]);

                    TensorShape mceOutputEncoding =
                        ApplyMceShapeMult({ 0, height, width, m_Capabilities.GetNumberOfOgs() });
                    TensorShape mceOutputStripe =
                        CreateStripe(mceOutputShape, mceOutputEncoding, m_Capabilities.GetNumberOfOgs());

                    TensorShape pleOutputStripe = ApplyPleShapeMult(mceOutputStripe);

                    TensorShape pleOutputEncoding = ApplyPleShapeMult(mceOutputEncoding);
                    TensorShape memoryOutputStripe =
                        CreateStripe(m_PleOutputTensorShape, pleOutputEncoding, m_Capabilities.GetBrickGroupShape()[3]);
                    const TensorShape& outputShape = m_PleOutputTensorShape;

                    AddStripeInfos(mceInputStripe, mceOutputStripe, mceOutputStripe, pleOutputStripe, numStripesInput,
                                   numStripesOutput, numStripesWeights, numStripesPleInput, mceInputStripe,
                                   memoryOutputStripe, mceOutputStripe, inputShape, outputShape);
                }
            }
        }

        // Try split depth for compute but the memory buffer is the full tensor
        // e.g. strategy 1 cascading
        {
            TensorShape mceInputEncoding  = { 0, 0, 0, m_Capabilities.GetNumberOfOgs() };
            const TensorShape& inputShape = m_MceInputTensorShape;
            TensorShape mceInputStripe =
                CreateStripe(m_MceInputTensorShape, mceInputEncoding, m_Capabilities.GetBrickGroupShape()[3]);

            TensorShape mceOutputEncoding = ApplyMceShapeMult({ 0, 0, 0, m_Capabilities.GetNumberOfOgs() });
            TensorShape mceOutputStripe = CreateStripe(inputShape, mceOutputEncoding, m_Capabilities.GetNumberOfOgs());

            const TensorShape& outputShape = m_PleOutputTensorShape;
            TensorShape pleOutputStripe    = ApplyPleShapeMult(mceOutputStripe);

            TensorShape memoryOutputEncoding = { 0, 0, 0, 0 };
            TensorShape memoryOutputStripe =
                CreateStripe(outputShape, memoryOutputEncoding, m_Capabilities.GetBrickGroupShape()[3]);
            AddStripeInfos(mceInputStripe, mceOutputStripe, mceOutputStripe, pleOutputStripe, numStripesInput,
                           numStripesOutput, numStripesWeights, numStripesPleInput, mceInputStripe, memoryOutputStripe,
                           mceOutputStripe, inputShape, outputShape);
        }
    }
    else
    {

        if (cascadeType == CascadeType::Lonely)
        {
            // Try split output depth
            {
                TensorShape mceInputEncoding  = { 0, 0, 0, 0 };
                const TensorShape& inputShape = m_MceInputTensorShape;
                TensorShape mceInputStripe =
                    CreateStripe(m_MceInputTensorShape, mceInputEncoding, m_Capabilities.GetBrickGroupShape()[3]);

                TensorShape mceOutputEncoding = ApplyMceShapeMult({ 0, 0, 0, m_Capabilities.GetNumberOfOgs() });
                TensorShape mceOutputStripe =
                    CreateStripe(mceOutputShape, mceOutputEncoding, m_Capabilities.GetNumberOfOgs());

                TensorShape pleOutputStripe = ApplyPleShapeMult(mceOutputStripe);

                TensorShape pleOutputEncoding = ApplyPleShapeMult(mceOutputEncoding);
                TensorShape memoryOutputStripe =
                    CreateStripe(m_PleOutputTensorShape, pleOutputEncoding, m_Capabilities.GetBrickGroupShape()[3]);

                NumStripes numStripesInputCopy = numStripesInput;
                numStripesInputCopy.m_Min      = std::min(numStripesInputCopy.m_Min, 1u);
                numStripesInputCopy.m_Max      = std::min(numStripesInputCopy.m_Max, 1u);
                const TensorShape& outputShape = m_PleOutputTensorShape;

                AddStripeInfos(mceInputStripe, mceOutputStripe, mceInputStripe, pleOutputStripe, numStripesInputCopy,
                               numStripesOutput, numStripesWeights, numStripesPleInput, mceInputStripe,
                               memoryOutputStripe, mceOutputStripe, inputShape, outputShape);
            }
        }
        // Try split depth for compute but the memory buffer is the full tensor
        // e.g. strategy 1 cascading
        {
            TensorShape mceInputEncoding  = { 0, 0, 0, 0 };
            const TensorShape& inputShape = m_MceInputTensorShape;
            TensorShape mceInputStripe =
                CreateStripe(m_MceInputTensorShape, mceInputEncoding, m_Capabilities.GetBrickGroupShape()[3]);

            TensorShape mceOutputEncoding = ApplyMceShapeMult({ 0, 0, 0, m_Capabilities.GetNumberOfOgs() });
            TensorShape mceOutputStripe =
                CreateStripe(mceOutputShape, mceOutputEncoding, m_Capabilities.GetNumberOfOgs());

            const TensorShape& outputShape = m_PleOutputTensorShape;
            TensorShape pleOutputStripe    = ApplyPleShapeMult(mceOutputStripe);
            NumStripes numStripesInputCopy = numStripesInput;
            numStripesInputCopy.m_Min      = std::min(numStripesInputCopy.m_Min, 1u);
            numStripesInputCopy.m_Max      = std::min(numStripesInputCopy.m_Max, 1u);

            TensorShape memoryOutputEncoding = { 0, 0, 0, 0 };
            TensorShape memoryOutputStripe =
                CreateStripe(outputShape, memoryOutputEncoding, m_Capabilities.GetBrickGroupShape()[3]);
            AddStripeInfos(mceInputStripe, mceOutputStripe, mceOutputStripe, pleOutputStripe, numStripesInputCopy,
                           numStripesOutput, numStripesWeights, numStripesPleInput, mceInputStripe, memoryOutputStripe,
                           mceOutputStripe, inputShape, outputShape);
        }
    }

    // Don't split at all
    // This is needed if all of the stripes above are larger than the tensor
    // and none of them are added
    {
        TensorShape mceInputEncoding = { 0, 0, 0, 0 };
        TensorShape mceInputStripe =
            CreateStripe(m_MceInputTensorShape, mceInputEncoding, m_Capabilities.GetBrickGroupShape()[3]);
        const TensorShape& inputShape  = m_MceInputTensorShape;
        const TensorShape& outputShape = m_PleOutputTensorShape;

        TensorShape mceOutputEncoding = ApplyMceShapeMult(mceInputEncoding);
        TensorShape mceOutputStripe = CreateStripe(mceOutputShape, mceOutputEncoding, m_Capabilities.GetNumberOfOgs());

        TensorShape pleOutputEncoding = ApplyPleShapeMult(mceOutputEncoding);
        TensorShape pleOutputStripe =
            CreateStripe(m_PleOutputTensorShape, pleOutputEncoding, m_Capabilities.GetBrickGroupShape()[3]);
        NumStripes numStripesInputCopy   = numStripesInput;
        numStripesInputCopy.m_Min        = std::min(numStripesInput.m_Min, 1u);
        numStripesInputCopy.m_Max        = std::min(numStripesInput.m_Max, 1u);
        NumStripes numStripesWeightsCopy = numStripesWeights;
        numStripesWeightsCopy.m_Min      = std::min(numStripesWeights.m_Min, 1u);
        numStripesWeightsCopy.m_Max      = std::min(numStripesWeights.m_Max, 1u);
        NumStripes numStripesOutputCopy  = numStripesOutput;
        numStripesOutputCopy.m_Min       = std::min(numStripesOutput.m_Min, 1u);
        numStripesOutputCopy.m_Max       = std::min(numStripesOutput.m_Max, 1u);

        AddStripeInfos(mceInputStripe, mceOutputStripe, mceOutputStripe, pleOutputStripe, numStripesInputCopy,
                       numStripesOutputCopy, numStripesWeightsCopy, numStripesPleInput, mceInputStripe, pleOutputStripe,
                       mceOutputStripe, inputShape, outputShape);
    }
}

bool NumStripes::operator<(const NumStripes& rhs) const
{
    if (m_Min < rhs.m_Min)
        return true;
    if (rhs.m_Min < m_Min)
        return false;
    if (m_Max < rhs.m_Max)
        return true;
    if (rhs.m_Max < m_Max)
        return false;
    return false;
}

bool MceStripesInfo::operator<(const MceStripesInfo& rhs) const
{
    if (m_Input < rhs.m_Input)
        return true;
    if (rhs.m_Input < m_Input)
        return false;
    if (m_Output < rhs.m_Output)
        return true;
    if (rhs.m_Output < m_Output)
        return false;
    if (m_Weight < rhs.m_Weight)
        return true;
    if (rhs.m_Weight < m_Weight)
        return false;
    if (m_BlockConfig.m_BlockWidth() < rhs.m_BlockConfig.m_BlockWidth())
        return true;
    if (rhs.m_BlockConfig.m_BlockWidth() < m_BlockConfig.m_BlockWidth())
        return false;
    if (m_BlockConfig.m_BlockHeight() < rhs.m_BlockConfig.m_BlockHeight())
        return true;
    if (rhs.m_BlockConfig.m_BlockHeight() < m_BlockConfig.m_BlockHeight())
        return false;
    return false;
}

bool PleStripesInfo::operator<(const PleStripesInfo& rhs) const
{
    if (m_Input < rhs.m_Input)
        return true;
    if (rhs.m_Input < m_Input)
        return false;
    if (m_Output < rhs.m_Output)
        return true;
    if (rhs.m_Output < m_Output)
        return false;
    if (m_BlockConfig.m_BlockWidth() < rhs.m_BlockConfig.m_BlockWidth())
        return true;
    if (rhs.m_BlockConfig.m_BlockWidth() < m_BlockConfig.m_BlockWidth())
        return false;
    if (m_BlockConfig.m_BlockHeight() < rhs.m_BlockConfig.m_BlockHeight())
        return true;
    if (rhs.m_BlockConfig.m_BlockHeight() < m_BlockConfig.m_BlockHeight())
        return false;
    return false;
}

bool MemoryStripeInfo::operator<(const MemoryStripeInfo& rhs) const
{
    if (m_Range < rhs.m_Range)
        return true;
    if (rhs.m_Range < m_Range)
        return false;
    if (m_Shape < rhs.m_Shape)
        return true;
    if (rhs.m_Shape < m_Shape)
        return false;
    return false;
}

bool MemoryStripesInfo::operator<(const MemoryStripesInfo& rhs) const
{
    if (m_Input < rhs.m_Input)
        return true;
    if (rhs.m_Input < m_Input)
        return false;
    if (m_Output < rhs.m_Output)
        return true;
    if (rhs.m_Output < m_Output)
        return false;
    if (m_Weight < rhs.m_Weight)
        return true;
    if (rhs.m_Weight < m_Weight)
        return false;
    if (m_PleInput < rhs.m_PleInput)
        return true;
    if (rhs.m_PleInput < m_PleInput)
        return false;
    return false;
}

bool NumMemoryStripes::operator<(const NumMemoryStripes& rhs) const
{
    if (m_Input < rhs.m_Input)
        return true;
    if (rhs.m_Input < m_Input)
        return false;
    if (m_Output < rhs.m_Output)
        return true;
    if (rhs.m_Output < m_Output)
        return false;
    if (m_Weight < rhs.m_Weight)
        return true;
    if (rhs.m_Weight < m_Weight)
        return false;
    if (m_PleInput < rhs.m_PleInput)
        return true;
    if (rhs.m_PleInput < m_PleInput)
        return false;
    return false;
}

bool MceAndPleInfo::operator<(const MceAndPleInfo& rhs) const
{
    if (m_MceCompute < rhs.m_MceCompute)
        return true;
    if (rhs.m_MceCompute < m_MceCompute)
        return false;
    if (m_PleCompute < rhs.m_PleCompute)
        return true;
    if (rhs.m_PleCompute < m_PleCompute)
        return false;
    if (m_Memory < rhs.m_Memory)
        return true;
    if (rhs.m_Memory < m_Memory)
        return false;
    return false;
}

bool MceOnlyInfo::operator<(const MceOnlyInfo& rhs) const
{
    if (m_MceCompute < rhs.m_MceCompute)
        return true;
    if (rhs.m_MceCompute < m_MceCompute)
        return false;
    if (m_Memory < rhs.m_Memory)
        return true;
    if (rhs.m_Memory < m_Memory)
        return false;
    return false;
}

bool PleOnlyInfo::operator<(const PleOnlyInfo& rhs) const
{
    if (m_PleCompute < rhs.m_PleCompute)
        return true;
    if (rhs.m_PleCompute < m_PleCompute)
        return false;
    if (m_Memory < rhs.m_Memory)
        return true;
    if (rhs.m_Memory < m_Memory)
        return false;
    return false;
}

bool DmaOnlyInfo::operator<(const DmaOnlyInfo& rhs) const
{
    if (m_Input < rhs.m_Input)
        return true;
    if (rhs.m_Input < m_Input)
        return false;
    if (m_Output < rhs.m_Output)
        return true;
    if (rhs.m_Output < m_Output)
        return false;
    return false;
}

uint32_t GetWeightStripeDepth(const TensorInfo& weightInfo, const TensorShape& weightStripeShape, const Stride& stride)
{
    if (weightInfo.m_DataFormat == DataFormat::HWIO)
    {
        return weightStripeShape[3];
    }
    else if (weightInfo.m_DataFormat == DataFormat::HWIM)
    {
        return weightStripeShape[2] * weightStripeShape[3] / (stride.m_X * stride.m_Y);
    }
    else
    {
        assert(false);
        return 0;
    }
}

Buffer* AddPleInBuffer(OwnedOpGraph& opGraph,
                       NumStripesType& numPleInputMemoryStripes,
                       const TensorShape& tensorShape,
                       const TensorShape& pleInputMemoryShape,
                       const QuantizationInfo& quantInfo,
                       Lifetime lifetime,
                       TraversalOrder order,
                       Location location)
{
    assert(location == Location::Sram || location == Location::PleInputSram);

    opGraph.AddBuffer(std::make_unique<Buffer>(lifetime, location, GetFormat(location), order));
    auto buffer = opGraph.GetBuffers().back();

    buffer->m_TensorShape = tensorShape;
    buffer->m_StripeShape = pleInputMemoryShape;
    buffer->m_NumStripes  = numPleInputMemoryStripes;

    // number of stripes in tile is only relevant if the input buffer is in SRAM
    uint32_t numStripesInTile = location == Location::Sram ? numPleInputMemoryStripes : 1;
    buffer->m_SizeInBytes     = impl::CalculateBufferSize(buffer->m_StripeShape, buffer->m_Format) * numStripesInTile;

    buffer->m_QuantizationInfo = quantInfo;
    return buffer;
}

std::pair<Buffer*, Op*> AddPleToOpGraph(OwnedOpGraph& opGraph,
                                        Lifetime lifetime,
                                        TraversalOrder order,
                                        const TensorShape& memoryOutputShape,
                                        impl::NumMemoryStripes& numMemoryStripes,
                                        std::unique_ptr<Op> pleOp,
                                        const TensorShape& outputShape,
                                        const QuantizationInfo& outputQuantInfo,
                                        const std::set<uint32_t>& sourceOperationIds)
{
    auto& buffers      = opGraph.GetBuffers();
    Op* op             = opGraph.AddOp(std::move(pleOp));
    op->m_OperationIds = sourceOperationIds;
    op->m_Lifetime     = lifetime;

    opGraph.AddBuffer(std::make_unique<Buffer>(lifetime, Location::Sram, GetFormat(Location::Sram), order));
    auto pleOutBuffer = buffers.back();
    opGraph.SetProducer(pleOutBuffer, op);

    pleOutBuffer->m_TensorShape = outputShape;
    pleOutBuffer->m_StripeShape = memoryOutputShape;
    pleOutBuffer->m_NumStripes  = numMemoryStripes.m_Output;
    pleOutBuffer->m_SizeInBytes = numMemoryStripes.m_Output * utils::TotalSizeBytesNHWCB(memoryOutputShape);

    pleOutBuffer->m_QuantizationInfo = outputQuantInfo;

    return { pleOutBuffer, op };
};

}    // namespace impl
}    // namespace support_library
}    // namespace ethosn
