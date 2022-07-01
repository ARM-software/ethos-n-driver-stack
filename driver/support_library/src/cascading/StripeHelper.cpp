//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "StripeHelper.hpp"

#include "PartUtils.hpp"
#include "WeightEncoderCache.hpp"
#include <ethosn_utils/Strings.hpp>

#include <fstream>
#include <regex>

namespace ethosn
{
namespace support_library
{
namespace impl
{

StripeConfig GetDefaultStripeConfig(const char* identifier)
{
    // Start with a defaultly constructed StripeConfig, which has everything enabled
    StripeConfig result;

    // Apply the rules from the config file, if one is set
    const char* env = std::getenv("ETHOSN_SUPPORT_LIBRARY_DEBUG_STRIPE_CONFIG");
    if (env && strlen(env) > 0)
    {
        // The config file has a simple format. A list of sections with each section starting with a regex that defines
        // which parts that section applies to. The contents of each section are a series of commands, executed in order,
        // which enable/disable stripe config options.
        //
        // <regex>:
        // <command1>
        // <command2>
        // # more commands...
        //
        // <regex>:
        // <command1>
        // <command2>
        // # more commands...
        //
        // # more sections
        //
        // A simple example:
        //
        // McePart 3:
        //
        // DisableAll
        // Splits.WidthHeight=True
        // BlockConfig(8,8)=True

        std::ifstream file(env);
        if (!file.good())
        {
            throw std::runtime_error("Error opening stripe config file: " + std::string(env));
        }

        std::string line;
        uint32_t lineNumber = 0;
        auto reportError    = [&lineNumber](std::string msg) {
            throw std::runtime_error("Error in stripe config file at line " + std::to_string(lineNumber) + ": " + msg);
        };

        bool active = false;    // Does the section of the file we are in match the identifier given
        while (getline(file, line))
        {
            ++lineNumber;
            line = ethosn::utils::Trim(line);
            if (line.empty() || line[0] == '#')
            {
                // Empty (or whitespace) lines or comments - ignore
                continue;
            }

            if (line.back() == ':')
            {
                // Start of new section
                active = false;
                // Check if the regex for this section matches the identifier given
                std::regex regex(line.substr(0, line.length() - 1));
                if (std::regex_match(identifier, regex))
                {
                    active = true;
                }
            }
            else
            {
                // Command within a section. Only process if the regex matched
                if (active)
                {
                    std::vector<std::string> parts = ethosn::utils::Split(line, "=");
                    if (line == "DisableAll")
                    {
                        result.DisableAll();
                    }
                    else if (parts.size() == 2)
                    {
                        const std::string& name     = parts[0];
                        const std::string& valueStr = parts[1];

                        auto valueBool = [&]() {
                            if (valueStr == "True")
                            {
                                return true;
                            }
                            else if (valueStr == "False")
                            {
                                return false;
                            }
                            else
                            {
                                reportError("Invalid value '" + valueStr + "'. Must be True or False.");
                                return false;    // Avoid incorrect warning. This never executes, as the above line throws.
                            }
                        };
                        auto valueUInt = [&]() {
                            try
                            {
                                return static_cast<uint32_t>(std::stoul(valueStr));
                            }
                            catch (const std::exception&)
                            {
                                reportError("Invalid value '" + valueStr + "'. Must be an unsigned number.");
                                return 0u;    // Avoid incorrect warning. This never executes, as the above line throws.
                            }
                        };

                        const std::regex blockConfigRegex(R"(BlockConfig\((\d+),(\d+)\))");
                        std::smatch match;
                        if (name == "Splits.HeightOnly")
                        {
                            result.splits.heightOnly = valueBool();
                        }
                        else if (name == "Splits.WidthOnly")
                        {
                            result.splits.widthOnly = valueBool();
                        }
                        else if (name == "Splits.WidthHeight")
                        {
                            result.splits.widthHeight = valueBool();
                        }
                        else if (name == "Splits.WidthHeightOutputDepth")
                        {
                            result.splits.widthHeightOutputDepth = valueBool();
                        }
                        else if (name == "Splits.WidthHeightOutputDepthInputDepth")
                        {
                            result.splits.widthHeightOutputDepthInputDepth = valueBool();
                        }
                        else if (name == "Splits.OutputDepthInputDepth")
                        {
                            result.splits.outputDepthInputDepth = valueBool();
                        }
                        else if (name == "Splits.OutputDepthOnly")
                        {
                            result.splits.outputDepthOnly = valueBool();
                        }
                        else if (name == "Splits.InputDepthOnly")
                        {
                            result.splits.inputDepthOnly = valueBool();
                        }
                        else if (name == "Splits.None")
                        {
                            result.splits.none = valueBool();
                        }
                        else if (std::regex_match(name, match, blockConfigRegex))
                        {
                            uint32_t w = std::atoi(match[1].str().c_str());
                            uint32_t h = std::atoi(match[2].str().c_str());
                            ethosn::command_stream::BlockConfig b{ w, h };
                            auto it = std::find(result.blockConfigs.begin(), result.blockConfigs.end(), b);
                            if (valueBool())
                            {
                                if (it == result.blockConfigs.end())
                                {
                                    result.blockConfigs.push_back(b);
                                }
                            }
                            else
                            {
                                if (it != result.blockConfigs.end())
                                {
                                    result.blockConfigs.erase(it);
                                }
                            }
                        }
                        else if (name == "BlockWidthMultiplier.Min")
                        {
                            result.blockWidthMultiplier.min = valueUInt();
                        }
                        else if (name == "BlockWidthMultiplier.Max")
                        {
                            result.blockWidthMultiplier.max = valueUInt();
                        }
                        else if (name == "BlockHeightMultiplier.Min")
                        {
                            result.blockHeightMultiplier.min = valueUInt();
                        }
                        else if (name == "BlockHeightMultiplier.Max")
                        {
                            result.blockHeightMultiplier.max = valueUInt();
                        }
                        else
                        {
                            reportError("Unknown name in assignment: " + name);
                        }
                    }
                    else
                    {
                        reportError("Unexpected command syntax: " + line);
                    }
                }
            }
        }
    }

    return result;
}

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
                                 uint32_t padTop,
                                 uint32_t padLeft,
                                 const Stride& stride,
                                 uint32_t upscaleFactor,
                                 command_stream::MceOperation op,
                                 command_stream::PleOperation pleOp,
                                 utils::ShapeMultiplier mceShapeMult,
                                 utils::ShapeMultiplier pleShapeMult,
                                 const HardwareCapabilities& m_Capabilities,
                                 StripeConfig stripeConfig)
    : m_MceInputTensorShape(mceInput)
    , m_MceOutputTensorShape(mceOutput)
    , m_PleOutputTensorShape(pleOutput)
    , m_KernelHeight(kernelHeight)
    , m_KernelWidth(kernelWidth)
    , m_PadTop(padTop)
    , m_PadLeft(padLeft)
    , m_Stride(stride)
    , m_UpscaleFactor(upscaleFactor)
    , m_Operation(op)
    , m_KernelOperation(pleOp)
    , m_MceShapeMultiplier(mceShapeMult)
    , m_PleShapeMultiplier(pleShapeMult)
    , m_Capabilities(m_Capabilities)
    , m_StripeConfig(stripeConfig)
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

StripeConfig StripeGenerator::ApplyPleKernelSplitRestrictions(CascadeType cascadeType) const
{
    StripeConfig result = m_StripeConfig;

    // MaxPool_3x3_2_2 cannot be cascaded if it isn't the full tensor and can only be cascaded along height or depth.
    // This way, IFM streaming cannot cause data corruption in Ple Sram.
    if (m_KernelOperation == command_stream::PleOperation::MAXPOOL_3X3_2_2_EVEN ||
        m_KernelOperation == command_stream::PleOperation::MAXPOOL_3X3_2_2_ODD)
    {
        if (cascadeType == CascadeType::Beginning)
        {
            result.DisableSplitHeight();
            result.DisableSplitWidth();
            result.DisableSplitInputDepth();
            result.DisableSplitOutputDepth();
        }
        else
        {
            result.DisableSplitWidth();
        }
    }

    return result;
}
StripeInfos StripeGenerator::GenerateStripes(CascadeType cascadeType) const
{
    StripeInfos result;
    for (auto&& blockConfig : m_StripeConfig.blockConfigs)
    {
        GenerateStripes(blockConfig, cascadeType, result);
    }
    return result;
}

void StripeGenerator::GenerateStripes(const ethosn::command_stream::BlockConfig blockConfig,
                                      CascadeType cascadeType,
                                      StripeInfos& outStripeInfos) const
{
    using namespace utils;

    // Set Stripe split restrictions, depending on the Ple kernel type.
    StripeConfig stripeConfig = ApplyPleKernelSplitRestrictions(cascadeType);

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
        // Output stripe size = min (Tensor dimension * Multiplier for that dimension, Tensor dimension rounded up to multiple of block size).
        //For Width: shapeW rounded up to multiple of brickWidth
        //For Height: shapeH rounded up to multiple of brickHeight
        //For Depth: shapeC rounded up to multiple of Number of Output Generators
        return TensorShape{ shape[0],
                            std::min(shape[1] * m_PleShapeMultiplier.m_H,
                                     utils::RoundUpToNearestMultiple(m_PleOutputTensorShape[1], 8U)),
                            std::min(shape[2] * m_PleShapeMultiplier.m_W,
                                     utils::RoundUpToNearestMultiple(m_PleOutputTensorShape[2], 8U)),
                            std::min(shape[3] * m_PleShapeMultiplier.m_C,
                                     utils::RoundUpToNearestMultiple(m_PleOutputTensorShape[3],
                                                                     m_Capabilities.GetNumberOfOgs())) };
    };

    auto AddStripeInfos = [&](const TensorShape& mceInputStripe, const TensorShape& mceOutputStripe,
                              const TensorShape& pleInputStripe, const TensorShape& pleOutputStripe,
                              const NumStripes& inputRange, const NumStripes& outputRange,
                              const NumStripes& weightRange, const NumStripes& pleInputRange,
                              const TensorShape& memoryInputStripe, const TensorShape& memoryOutputStripe,
                              const TensorShape& memoryPleInputStripe, const TensorShape& inputShape,
                              const TensorShape& outputShape, const Lifetime lifetime = Lifetime::Cascade) {
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

        const NeedBoundary needBoundaryY = utils::GetBoundaryRequirements(
            m_PadTop, GetHeight(inputShape), GetHeight(mceInputStripe), GetHeight(mceOutputStripe), m_KernelHeight);
        const NeedBoundary needBoundaryX = utils::GetBoundaryRequirements(
            m_PadLeft, GetWidth(inputShape), GetWidth(mceInputStripe), GetWidth(mceOutputStripe), m_KernelWidth);
        const bool packBoundaryVertical   = (GetWidth(mceInputStripe) < GetWidth(inputShape));
        const bool packBoundaryHorizontal = (GetChannels(mceInputStripe) < GetChannels(inputShape));

        command_stream::cascading::PackedBoundaryThickness packedBoundaryThickness;
        packedBoundaryThickness.left   = (packBoundaryHorizontal && needBoundaryX.m_Before) ? 8 : 0;
        packedBoundaryThickness.top    = (packBoundaryVertical && needBoundaryY.m_Before) ? 8 : 0;
        packedBoundaryThickness.right  = (packBoundaryHorizontal && needBoundaryX.m_After) ? 8 : 0;
        packedBoundaryThickness.bottom = (packBoundaryVertical && needBoundaryY.m_After) ? 8 : 0;

        // OFM is always traversed in XYZ order and IFM always in ZXY. Therefore IFM data needs multiple loads if there
        // is more than one stripe in OFM depth, and the IFM has more than one stripe.
        const uint32_t numIfmLoads = !isDepthwise && (GetWidth(mceInputStripe) < GetWidth(inputShape) ||
                                                      GetHeight(mceInputStripe) < GetHeight(inputShape) ||
                                                      GetChannels(mceInputStripe) < GetChannels(inputShape))
                                         ? utils::DivRoundUp(GetChannels(mceOutputShape), GetChannels(mceOutputStripe))
                                         : 1;

        const uint32_t numWeightLoads = !isDepthwise && GetChannels(mceInputStripe) < GetChannels(inputShape)
                                            ? (utils::DivRoundUp(GetWidth(mceOutputShape), GetWidth(mceOutputStripe)) *
                                               utils::DivRoundUp(GetHeight(mceOutputShape), GetHeight(mceOutputStripe)))
                                            : 1;

        {
            MceAndPleInfo mceAndPleInfo;

            mceAndPleInfo.m_Lifetime = lifetime;

            mceAndPleInfo.m_MceCompute.m_Input       = mceInputStripe;
            mceAndPleInfo.m_MceCompute.m_Output      = mceOutputStripe;
            mceAndPleInfo.m_MceCompute.m_Weight      = mceWeightStripe;
            mceAndPleInfo.m_MceCompute.m_BlockConfig = blockConfig;
            mceAndPleInfo.m_PleCompute.m_Input       = pleInputStripe;
            mceAndPleInfo.m_PleCompute.m_Output      = pleOutputStripe;
            mceAndPleInfo.m_PleCompute.m_BlockConfig = blockConfig;

            mceAndPleInfo.m_Memory.m_Input = { { inputCopy, memoryInputStripe }, packedBoundaryThickness, numIfmLoads };
            mceAndPleInfo.m_Memory.m_Output   = { outputCopy, memoryOutputStripe };
            mceAndPleInfo.m_Memory.m_Weight   = { { weightCopy, memoryWeightStripe }, numWeightLoads };
            mceAndPleInfo.m_Memory.m_PleInput = { pleInputRange, memoryPleInputStripe };
            outStripeInfos.m_MceAndPleInfos.insert(mceAndPleInfo);
        }
        {
            MceOnlyInfo mceOnlyInfo;

            mceOnlyInfo.m_Lifetime = lifetime;

            mceOnlyInfo.m_MceCompute.m_Input       = mceInputStripe;
            mceOnlyInfo.m_MceCompute.m_Output      = mceOutputStripe;
            mceOnlyInfo.m_MceCompute.m_Weight      = mceWeightStripe;
            mceOnlyInfo.m_MceCompute.m_BlockConfig = blockConfig;

            mceOnlyInfo.m_Memory.m_Input  = { { inputCopy, memoryInputStripe }, packedBoundaryThickness, numIfmLoads };
            mceOnlyInfo.m_Memory.m_Output = { { 0, 0 }, { 0, 0, 0, 0 } };
            mceOnlyInfo.m_Memory.m_Weight = { { weightCopy, memoryWeightStripe }, numWeightLoads };
            mceOnlyInfo.m_Memory.m_PleInput = { pleInputRange, memoryPleInputStripe };
            outStripeInfos.m_MceOnlyInfos.insert(mceOnlyInfo);
        }
        {
            PleOnlyInfo pleOnlyInfo;

            pleOnlyInfo.m_Lifetime = lifetime;

            pleOnlyInfo.m_PleCompute.m_Input       = pleInputStripe;
            pleOnlyInfo.m_PleCompute.m_Output      = pleOutputStripe;
            pleOnlyInfo.m_PleCompute.m_BlockConfig = blockConfig;

            pleOnlyInfo.m_Memory.m_Input    = { { { 0, 0 }, { 0, 0, 0, 0 } }, { 0, 0, 0, 0 }, 0 };
            pleOnlyInfo.m_Memory.m_Output   = { outputCopy, memoryOutputStripe };
            pleOnlyInfo.m_Memory.m_Weight   = { { { 0, 0 }, { 0, 0, 0, 0 } }, 0 };
            pleOnlyInfo.m_Memory.m_PleInput = { pleInputRange, memoryPleInputStripe };
            outStripeInfos.m_PleOnlyInfos.insert(pleOnlyInfo);
        }
        {
            DmaOnlyInfo dmaOnlyInfo;
            dmaOnlyInfo.m_Lifetime = lifetime;
            dmaOnlyInfo.m_Input    = { inputCopy, memoryInputStripe };
            dmaOnlyInfo.m_Output   = { outputCopy, memoryOutputStripe };
            outStripeInfos.m_DmaOnlyInfos.insert(dmaOnlyInfo);
        }
    };

    // Use the minimum stripe size possible to minimize the time before processing.
    // Try splitting height first.
    if (stripeConfig.splits.heightOnly)
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

    // Split only input in height while the output is full tensor.
    if (stripeConfig.splits.heightOnly)
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

    // Try splitting width.
    if (stripeConfig.splits.widthOnly)
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
    const uint32_t minBlockWidthMultiplier = m_StripeConfig.blockWidthMultiplier.min;
    const uint32_t maxBlockWidthMultiplier =
        std::max(1U, std::min(GetWidth(m_MceInputTensorShape) / blockConfig.m_BlockWidth(),
                              m_StripeConfig.blockWidthMultiplier.max));
    const uint32_t minBlockHeightMultiplier = m_StripeConfig.blockHeightMultiplier.min;
    const uint32_t maxBlockHeightMultiplier =
        std::max(1U, std::min(GetHeight(m_MceInputTensorShape) / blockConfig.m_BlockHeight(),
                              m_StripeConfig.blockHeightMultiplier.max));

    if (cascadeType == CascadeType::Lonely)
    {
        for (uint32_t heightMultiplier = minBlockHeightMultiplier; heightMultiplier <= maxBlockHeightMultiplier;
             ++heightMultiplier)
        {
            for (uint32_t widthMultiplier = minBlockWidthMultiplier; widthMultiplier <= maxBlockWidthMultiplier;
                 ++widthMultiplier)
            {
                // Try splitting width and height.
                if (stripeConfig.splits.widthHeight)
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

                // Try split height width and output depth.
                if (stripeConfig.splits.widthHeightOutputDepth)
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

                // Try split input depth.
                // Note we have to limit the height and width to the block size.
                if (stripeConfig.splits.widthHeightOutputDepthInputDepth)
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
            // Try split output depth and input depth.
            if (stripeConfig.splits.outputDepthInputDepth)
            {
                // With depthwise each only OFM needs 1 IFM.
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

            // Try split height width and output depth and input depth.
            if (stripeConfig.splits.widthHeightOutputDepthInputDepth)
            {
                for (uint32_t heightMultiplier = minBlockHeightMultiplier; heightMultiplier <= maxBlockHeightMultiplier;
                     ++heightMultiplier)
                {
                    for (uint32_t widthMultiplier = minBlockWidthMultiplier; widthMultiplier <= maxBlockWidthMultiplier;
                         ++widthMultiplier)
                    {
                        const uint32_t height = heightMultiplier * blockConfig.m_BlockHeight();
                        const uint32_t width  = widthMultiplier * blockConfig.m_BlockWidth();

                        TensorShape mceInputEncoding  = { 0, height, width,
                                                         m_Capabilities.GetNumberOfOgs() * strideMultiplier };
                        const TensorShape& inputShape = m_MceInputTensorShape;
                        TensorShape mceInputStripe    = CreateStripe(m_MceInputTensorShape, mceInputEncoding,
                                                                  m_Capabilities.GetBrickGroupShape()[3]);

                        TensorShape mceOutputEncoding =
                            ApplyMceShapeMult({ 0, height, width, m_Capabilities.GetNumberOfOgs() });
                        TensorShape mceOutputStripe =
                            CreateStripe(mceOutputShape, mceOutputEncoding, m_Capabilities.GetNumberOfOgs());

                        TensorShape pleOutputStripe = ApplyPleShapeMult(mceOutputStripe);

                        TensorShape pleOutputEncoding  = ApplyPleShapeMult(mceOutputEncoding);
                        TensorShape memoryOutputStripe = CreateStripe(m_PleOutputTensorShape, pleOutputEncoding,
                                                                      m_Capabilities.GetBrickGroupShape()[3]);
                        const TensorShape& outputShape = m_PleOutputTensorShape;

                        AddStripeInfos(mceInputStripe, mceOutputStripe, mceOutputStripe, pleOutputStripe,
                                       numStripesInput, numStripesOutput, numStripesWeights, numStripesPleInput,
                                       mceInputStripe, memoryOutputStripe, mceOutputStripe, inputShape, outputShape);
                    }
                }
            }
        }

        // Try split depth for compute but the memory buffer is the full tensor
        // e.g. strategy 1 cascading.
        if (stripeConfig.splits.outputDepthInputDepth)
        {
            TensorShape mceInputEncoding  = { 0, 0, 0, m_Capabilities.GetNumberOfOgs() };
            const TensorShape& inputShape = m_MceInputTensorShape;
            TensorShape mceInputStripe =
                CreateStripe(m_MceInputTensorShape, mceInputEncoding, m_Capabilities.GetBrickGroupShape()[3]);

            TensorShape mceOutputEncoding = ApplyMceShapeMult({ 0, 0, 0, m_Capabilities.GetNumberOfOgs() });
            TensorShape mceOutputStripe =
                CreateStripe(mceOutputShape, mceOutputEncoding, m_Capabilities.GetNumberOfOgs());

            const TensorShape& outputShape = m_PleOutputTensorShape;
            TensorShape pleOutputStripe    = ApplyPleShapeMult(mceOutputStripe);

            TensorShape memoryOutputEncoding = { 0, 0, 0, 0 };
            TensorShape memoryOutputStripe =
                CreateStripe(outputShape, memoryOutputEncoding, m_Capabilities.GetBrickGroupShape()[3]);
            AddStripeInfos(mceInputStripe, mceOutputStripe, mceOutputStripe, pleOutputStripe, numStripesInput,
                           numStripesOutput, numStripesWeights, numStripesPleInput, mceInputStripe, memoryOutputStripe,
                           mceOutputStripe, inputShape, outputShape, Lifetime::Atomic);
        }
    }
    else
    {

        if (cascadeType == CascadeType::Lonely)
        {
            // Try split output depth.
            if (stripeConfig.splits.outputDepthOnly)
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
        // e.g. strategy 1 cascading.
        if (stripeConfig.splits.outputDepthOnly)
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
                           mceOutputStripe, inputShape, outputShape, Lifetime::Atomic);
        }
    }

    // Don't split at all.
    // This is needed if all of the stripes above are larger than the tensor
    // and none of them are added.
    if (stripeConfig.splits.none)
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
                       mceOutputStripe, inputShape, outputShape, Lifetime::Atomic);
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

bool InputMemoryStripeInfo::operator<(const InputMemoryStripeInfo& rhs) const
{
    auto lhsTuple = std::make_tuple(static_cast<const MemoryStripeInfo&>(*this), m_PackedBoundaryThickness.left,
                                    m_PackedBoundaryThickness.top, m_PackedBoundaryThickness.right,
                                    m_PackedBoundaryThickness.bottom, m_NumLoads);
    auto rhsTuple = std::make_tuple(static_cast<const MemoryStripeInfo&>(rhs), rhs.m_PackedBoundaryThickness.left,
                                    rhs.m_PackedBoundaryThickness.top, rhs.m_PackedBoundaryThickness.right,
                                    rhs.m_PackedBoundaryThickness.bottom, rhs.m_NumLoads);
    return lhsTuple < rhsTuple;
}

bool WeightMemoryStripeInfo::operator<(const WeightMemoryStripeInfo& rhs) const
{
    auto lhsTuple = std::make_tuple(static_cast<const MemoryStripeInfo&>(*this), m_NumLoads);
    auto rhsTuple = std::make_tuple(static_cast<const MemoryStripeInfo&>(rhs), rhs.m_NumLoads);
    return lhsTuple < rhsTuple;
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
                       Location location)
{
    assert(location == Location::Sram || location == Location::PleInputSram);

    opGraph.AddBuffer(std::make_unique<Buffer>(location, GetFormat(location), TraversalOrder::Xyz));
    auto buffer = opGraph.GetBuffers().back();

    buffer->m_TensorShape = tensorShape;
    buffer->m_StripeShape = pleInputMemoryShape;
    buffer->m_NumStripes  = numPleInputMemoryStripes;

    // number of stripes in tile is only relevant if the input buffer is in SRAM
    uint32_t numStripesInTile = location == Location::Sram ? numPleInputMemoryStripes : 1;
    buffer->m_SlotSizeInBytes = impl::CalculateBufferSize(buffer->m_StripeShape, buffer->m_Format);
    buffer->m_SizeInBytes     = buffer->m_SlotSizeInBytes * numStripesInTile;

    buffer->m_QuantizationInfo = quantInfo;
    return buffer;
}

std::pair<Buffer*, Op*> AddPleToOpGraph(OwnedOpGraph& opGraph,
                                        Lifetime lifetime,
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

    opGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, GetFormat(Location::Sram), TraversalOrder::Xyz));
    auto pleOutBuffer = buffers.back();
    opGraph.SetProducer(pleOutBuffer, op);

    pleOutBuffer->m_TensorShape     = outputShape;
    pleOutBuffer->m_StripeShape     = memoryOutputShape;
    pleOutBuffer->m_NumStripes      = numMemoryStripes.m_Output;
    pleOutBuffer->m_SizeInBytes     = numMemoryStripes.m_Output * utils::TotalSizeBytesNHWCB(memoryOutputShape);
    pleOutBuffer->m_SlotSizeInBytes = utils::TotalSizeBytesNHWCB(memoryOutputShape);

    pleOutBuffer->m_QuantizationInfo = outputQuantInfo;

    return { pleOutBuffer, op };
};

}    // namespace impl
}    // namespace support_library
}    // namespace ethosn
