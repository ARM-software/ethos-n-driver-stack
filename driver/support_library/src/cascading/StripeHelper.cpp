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

using namespace ethosn::support_library::utils;

namespace ethosn
{
namespace support_library
{
namespace impl
{

StripeConfig GetDefaultStripeConfig(const CompilationOptions& compilationOptions, const char* identifier)
{
    // Start with a defaultly constructed StripeConfig, which has everything enabled
    StripeConfig result;

    // For backwards compatibility with legacy code, apply the strategy and block config filtering
    // from the compilation options.
    // The cascading strategies don't match up 1:1 with the legacy strategies and so there isn't
    // a clear mapping. We assume that if the user disabled any strategies then all cascading strategies
    // are disabled apart from a rough mapping of the ones that the user left enabled.
    if (!compilationOptions.m_Strategy0 || !compilationOptions.m_Strategy1 || !compilationOptions.m_Strategy3 ||
        !compilationOptions.m_Strategy4 || !compilationOptions.m_Strategy6 || !compilationOptions.m_Strategy7)
    {
        result.DisableAllSplits();
        if (compilationOptions.m_Strategy0)
        {
            result.splits.mceAndPleOutputHeight = true;
        }
        if (compilationOptions.m_Strategy1)
        {
            result.splits.mceAndPleOutputDepth  = true;
            result.splits.outputDepthInputDepth = true;
        }
        if (compilationOptions.m_Strategy3)
        {
            result.splits.none = true;
        }
        if (compilationOptions.m_Strategy4)
        {
            // Legacy strategy 4 splitted width and output depth, but we don't have this in cascading.
            // Pick something close instead.
            result.splits.widthOnly = true;
        }
        if (compilationOptions.m_Strategy6)
        {
            result.splits.widthHeight            = true;
            result.splits.widthHeightOutputDepth = true;
        }
        if (compilationOptions.m_Strategy7)
        {
            result.splits.widthHeightOutputDepthInputDepth = true;
        }
    }

    auto removeBlockConfig = [&result](ethosn::command_stream::BlockConfig b) {
        result.blockConfigs.erase(std::remove(result.blockConfigs.begin(), result.blockConfigs.end(), b),
                                  result.blockConfigs.end());
    };

    if (!compilationOptions.m_BlockConfig8x8)
    {
        removeBlockConfig(ethosn::command_stream::BlockConfig{ 8u, 8u });
    }
    if (!compilationOptions.m_BlockConfig8x16)
    {
        removeBlockConfig(ethosn::command_stream::BlockConfig{ 8u, 16u });
    }
    if (!compilationOptions.m_BlockConfig16x8)
    {
        removeBlockConfig(ethosn::command_stream::BlockConfig{ 16u, 8u });
    }
    if (!compilationOptions.m_BlockConfig16x16)
    {
        removeBlockConfig(ethosn::command_stream::BlockConfig{ 16u, 16u });
    }
    if (!compilationOptions.m_BlockConfig32x8)
    {
        removeBlockConfig(ethosn::command_stream::BlockConfig{ 32u, 8u });
    }
    if (!compilationOptions.m_BlockConfig8x32)
    {
        removeBlockConfig(ethosn::command_stream::BlockConfig{ 8u, 832u });
    }

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
                    else if (line == "DisableAllSplits")
                    {
                        result.DisableAllSplits();
                    }
                    else if (line == "DisableAllBlockConfigs")
                    {
                        result.blockConfigs.clear();
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
                        if (name == "Splits.MceAndPleOutputHeight")
                        {
                            result.splits.mceAndPleOutputHeight = valueBool();
                        }
                        else if (name == "Splits.MceOutputHeightOnly")
                        {
                            result.splits.mceOutputHeightOnly = valueBool();
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
                        else if (name == "Splits.MceOutputDepthOnly")
                        {
                            result.splits.mceOutputDepthOnly = valueBool();
                        }
                        else if (name == "Splits.MceAndPleOutputDepth")
                        {
                            result.splits.mceAndPleOutputDepth = valueBool();
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
                            if (valueBool())
                            {
                                auto it = std::find(result.blockConfigs.begin(), result.blockConfigs.end(), b);
                                if (it == result.blockConfigs.end())
                                {
                                    result.blockConfigs.push_back(b);
                                }
                            }
                            else
                            {
                                removeBlockConfig(b);
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
                        else if (name == "IfmDepthMultiplier.Min")
                        {
                            result.ifmDepthMultiplier.min = valueUInt();
                        }
                        else if (name == "IfmDepthMultiplier.Max")
                        {
                            result.ifmDepthMultiplier.max = valueUInt();
                        }
                        else if (name == "OfmDepthMultiplier.Min")
                        {
                            result.ofmDepthMultiplier.min = valueUInt();
                        }
                        else if (name == "OfmDepthMultiplier.Max")
                        {
                            result.ofmDepthMultiplier.max = valueUInt();
                        }
                        else if (name == "PlanTypes.Beginning")
                        {
                            result.planTypes.beginning = valueBool();
                        }
                        else if (name == "PlanTypes.Middle")
                        {
                            result.planTypes.middle = valueBool();
                        }
                        else if (name == "PlanTypes.End")
                        {
                            result.planTypes.end = valueBool();
                        }
                        else if (name == "PlanTypes.Lonely")
                        {
                            result.planTypes.lonely = valueBool();
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

/// Checks if a given SRAM buffer could be DMA'd to or from a DRAM buffer of the given format.
/// For example, this checks that an SRAM buffer with a stripe shape that splits depth cannot be DMA'd
/// to an NHWC DRAM buffer (as the firmware does not support this).
bool IsSramBufferCompatibleWithDramFormat(const Buffer& sramBuffer, CascadingBufferFormat dramFormat)
{
    // NHWC can't split depth
    if (dramFormat == CascadingBufferFormat::NHWC &&
        utils::GetChannels(sramBuffer.m_StripeShape) < utils::GetChannels(sramBuffer.m_TensorShape))
    {
        return false;
    }

    // FCAF requires certain stripe shapes
    if (dramFormat == CascadingBufferFormat::FCAF_DEEP &&
        !IsCompressionFormatCompatibleWithStripeShape(CompilerDataCompressedFormat::FCAF_DEEP,
                                                      sramBuffer.m_StripeShape))
    {
        return false;
    }
    // FCAF requires certain stripe shapes
    if (dramFormat == CascadingBufferFormat::FCAF_WIDE &&
        !IsCompressionFormatCompatibleWithStripeShape(CompilerDataCompressedFormat::FCAF_WIDE,
                                                      sramBuffer.m_StripeShape))
    {
        return false;
    }

    // Packed boundary data only supported with NHWCB
    if (dramFormat != CascadingBufferFormat::NHWCB &&
        utils::AnyPackedBoundaryData(sramBuffer.m_PackedBoundaryThickness))
    {
        return false;
    }

    return true;
}

CascadingBufferFormat GetBestDramBufferFormat(const std::vector<const Buffer*>& sramBuffers,
                                              const CompilationOptions& compilationOptions)
{
    bool fcafDeep = compilationOptions.m_EnableIntermediateCompression;
    bool fcafWide = compilationOptions.m_EnableIntermediateCompression;

    for (const Buffer* b : sramBuffers)
    {
        if (!IsSramBufferCompatibleWithDramFormat(*b, CascadingBufferFormat::FCAF_DEEP))
        {
            fcafDeep = false;
        }
        if (!IsSramBufferCompatibleWithDramFormat(*b, CascadingBufferFormat::FCAF_WIDE))
        {
            fcafWide = false;
        }
    }

    if (fcafDeep)
    {
        return CascadingBufferFormat::FCAF_DEEP;
    }
    else if (fcafWide)
    {
        return CascadingBufferFormat::FCAF_WIDE;
    }
    return CascadingBufferFormat::NHWCB;
}

/// Creates an SRAM buffer for use in a glue which DMAs stuff into and out of SRAM.
/// The code attempts to choose an optimal stripe shape.
std::unique_ptr<Buffer>
    MakeGlueIntermediateSramBuffer(const TensorShape& shape,
                                   const QuantizationInfo& quantInfo,
                                   DataType dataType,
                                   const std::vector<CascadingBufferFormat>& compatibleDramBufferFormats,
                                   const HardwareCapabilities& caps,
                                   uint32_t minWidthMultiplier,
                                   uint32_t maxWidthMultiplier,
                                   uint32_t minHeightMultiplier,
                                   uint32_t maxHeightMultiplier,
                                   uint32_t minDepthMultiplier,
                                   uint32_t maxDepthMultiplier)
{
    // Calculate minimum stripe size, based on the DRAM format(s) that this buffer needs to be compatible with
    uint32_t baseWidth  = utils::GetWidth(caps.GetBrickGroupShape());
    uint32_t baseHeight = utils::GetHeight(caps.GetBrickGroupShape());
    uint32_t baseDepth  = utils::GetChannels(caps.GetBrickGroupShape());
    for (CascadingBufferFormat format : compatibleDramBufferFormats)
    {
        // We always need at least one brick group (even for NHWC)
        TensorShape minStripeShape = caps.GetBrickGroupShape();
        switch (format)
        {
            case CascadingBufferFormat::NHWC:
                // The firmware cannot split NHWC tensors along channels, so we must use the full depth.
                minStripeShape[3] =
                    utils::RoundUpToNearestMultiple(shape[3], utils::GetChannels(caps.GetBrickGroupShape()));
                break;
            case CascadingBufferFormat::NHWCB:
                minStripeShape = caps.GetBrickGroupShape();
                break;
            case CascadingBufferFormat::FCAF_DEEP:
                minStripeShape = g_FcafDeepCellShape;
                break;
            case CascadingBufferFormat::FCAF_WIDE:
                minStripeShape = g_FcafWideCellShape;
                break;
            default:
                assert(false);
        }
        // Note this simple std::max is only valid because we know the values are all multiples of each
        // other (8, 16 or 32). If we wanted this to be more generic, we would need to use a "least common multiple" algorithm.
        baseHeight = std::max(baseHeight, utils::GetHeight(minStripeShape));
        baseWidth  = std::max(baseWidth, utils::GetWidth(minStripeShape));
        baseDepth  = std::max(baseDepth, utils::GetChannels(minStripeShape));
    }

    // Set the SRAM buffer's stripe size to be the largest shape that fits in SRAM,
    // to minimise stripe processing overhead.
    TensorShape bestStripeShape;
    uint32_t bestScore = 0;
    // Inclusive loops so that we generate candidates that split only one or two of the dimensions, or none of them.
    for (uint32_t stripeHeight :
         StripeShapeLoop::Inclusive(utils::GetHeight(shape), baseHeight, minHeightMultiplier, maxHeightMultiplier))
    {
        for (uint32_t stripeWidth :
             StripeShapeLoop::Inclusive(utils::GetWidth(shape), baseWidth, minWidthMultiplier, maxWidthMultiplier))
        {
            for (uint32_t stripeDepth : StripeShapeLoop::Inclusive(utils::GetChannels(shape), baseDepth,
                                                                   minDepthMultiplier, maxDepthMultiplier))
            {
                TensorShape candidateStripeShape = { 1, stripeHeight, stripeWidth, stripeDepth };
                uint32_t score                   = utils::GetNumElements(candidateStripeShape);
                // Prefer full-channel and full-width stripes, as these are more efficient to transfer.
                if (utils::GetChannels(candidateStripeShape) >= utils::GetChannels(shape))
                {
                    score *= 2;
                    if (utils::GetWidth(candidateStripeShape) >= utils::GetWidth(shape))
                    {
                        score *= 2;
                    }
                }
                if (utils::TotalSizeBytesNHWCB(candidateStripeShape) <= caps.GetTotalSramSize() && score > bestScore)
                {
                    bestScore       = score;
                    bestStripeShape = candidateStripeShape;
                }
            }
        }
    }

    auto sramBuffer =
        std::make_unique<Buffer>(Location::Sram, CascadingBufferFormat::NHWCB, shape, bestStripeShape,
                                 TraversalOrder::Xyz, utils::TotalSizeBytesNHWCB(bestStripeShape), quantInfo);
    sramBuffer->m_DataType   = dataType;
    sramBuffer->m_BufferType = BufferType::Intermediate;
    sramBuffer->m_Offset     = 0;    // Nothing else should be resident in SRAM at this point, so we can use any address
    sramBuffer->m_NumStripes = 1;
    sramBuffer->m_SlotSizeInBytes = sramBuffer->m_SizeInBytes;

    return sramBuffer;
}

StripeGenerator::StripeGenerator(const TensorShape& mceInput,
                                 const TensorShape& mceOutput,
                                 const TensorShape& pleOutput,
                                 uint32_t kernelHeight,
                                 uint32_t kernelWidth,
                                 uint32_t padTop,
                                 uint32_t padLeft,
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
    , m_UpscaleFactor(upscaleFactor)
    , m_Operation(op)
    , m_KernelOperation(pleOp)
    , m_MceShapeMultiplier(mceShapeMult)
    , m_PleShapeMultiplier(pleShapeMult)
    , m_Capabilities(m_Capabilities)
    , m_StripeConfig(stripeConfig)
{}

void StripeGenerator::CreateNumStripes(CascadeType cascadeType,
                                       bool requiresBoundaryData,
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
            if (!requiresBoundaryData)
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
            if (!requiresBoundaryData)
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

    const uint32_t numOgs     = m_Capabilities.GetNumberOfOgs();
    const uint32_t brickDepth = GetChannels(m_Capabilities.GetBrickGroupShape());

    // Set Stripe split restrictions, depending on the Ple kernel type.
    StripeConfig stripeConfig = ApplyPleKernelSplitRestrictions(cascadeType);

    const bool isDepthwise           = m_Operation == ethosn::command_stream::MceOperation::DEPTHWISE_CONVOLUTION;
    const TensorShape mceOutputShape = m_MceOutputTensorShape;

    auto AddStripeInfos = [&](const TensorShape& mceInputStripe, const TensorShape& mceOutputStripe,
                              const TensorShape& pleInputStripe, const TensorShape& pleOutputStripe,
                              const TensorShape& memoryInputStripe, const TensorShape& memoryOutputStripe,
                              const TensorShape& memoryPleInputStripe, const TensorShape& inputShape,
                              const TensorShape& outputShape) {
        NumStripes inputRange;
        NumStripes outputRange;
        NumStripes weightRange;
        NumStripes pleInputRange;
        const bool requiresBoundaryData =
            (m_KernelHeight > 1 && GetHeight(mceInputStripe) < GetHeight(m_MceInputTensorShape)) ||
            (m_KernelWidth > 1 && GetWidth(mceInputStripe) < GetWidth(m_MceInputTensorShape)) || m_UpscaleFactor > 1;
        CreateNumStripes(cascadeType, requiresBoundaryData, inputRange, outputRange, weightRange, pleInputRange);

        // Limit the max number of stripes based on the size of the tensor - there is no point considering plans where
        // we can store more stripes in the tile than there are in the tensor!
        NumStripes inputCopy = inputRange;
        inputCopy.m_Max =
            std::min(inputCopy.m_Max, DivRoundUp(GetHeight(inputShape), GetHeight(memoryInputStripe)) *
                                          DivRoundUp(GetWidth(inputShape), GetWidth(memoryInputStripe)) *
                                          DivRoundUp(GetChannels(inputShape), GetChannels(memoryInputStripe)));
        inputCopy.m_Min = std::min(inputCopy.m_Min, inputCopy.m_Max);

        NumStripes outputCopy = outputRange;
        outputCopy.m_Max =
            std::min(outputCopy.m_Max, DivRoundUp(GetHeight(outputShape), GetHeight(memoryOutputStripe)) *
                                           DivRoundUp(GetWidth(outputShape), GetWidth(memoryOutputStripe)) *
                                           DivRoundUp(GetChannels(outputShape), GetChannels(memoryOutputStripe)));
        outputCopy.m_Min = std::min(outputCopy.m_Min, outputCopy.m_Max);

        // Prevent using stripes which have more elements than the entire tensor
        bool multipleStripes         = inputCopy.m_Max > 1 && outputCopy.m_Max > 1;
        bool stripesLargerThanTensor = utils::GetNumElements(memoryInputStripe) > utils::GetNumElements(inputShape) &&
                                       utils::GetNumElements(memoryOutputStripe) > utils::GetNumElements(outputShape);
        if (multipleStripes && stripesLargerThanTensor)
        {
            return;
        }

        // Prevent too many MCE stripes per PLE (a firmware limitation)
        const uint32_t numMceStripesPerPle =
            // Multiple stripes from output depth splitting, where the PLE accumulates the full depth
            utils::DivRoundUp(GetChannels(pleInputStripe), GetChannels(mceOutputStripe)) *
            // Multiple stripes from input depth splitting, where the MCE doesn't pass its result to the PLE until
            // after it has processed the whole IFM depth.
            utils::DivRoundUp(GetChannels(inputShape), GetChannels(mceInputStripe));
        if (numMceStripesPerPle > m_Capabilities.GetMaxMceStripesPerPleStripe())
        {
            return;
        }

        // Prevent too many IFM and Weight stripes per PLE (a firmware limitation)
        const uint32_t numIfmStripesPerMce =
            utils::DivRoundUp(GetWidth(mceInputStripe), GetWidth(memoryInputStripe)) *
            utils::DivRoundUp(GetHeight(mceInputStripe), GetHeight(memoryInputStripe)) *
            utils::DivRoundUp(GetChannels(mceInputStripe), GetChannels(memoryInputStripe));
        const uint32_t numWgtStripesPerMce       = 1;
        const uint32_t numIfmAndWgtStripesPerPle = (numIfmStripesPerMce + numWgtStripesPerMce) * numMceStripesPerPle;
        if (numIfmAndWgtStripesPerPle > m_Capabilities.GetMaxIfmAndWgtStripesPerPleStripe())
        {
            return;
        }

        TensorShape mceWeightStripe    = { m_KernelHeight, m_KernelWidth, mceInputStripe[3],
                                        isDepthwise ? 1 : mceOutputStripe[3] };
        TensorShape memoryWeightStripe = mceWeightStripe;
        // Limit the max number of stripes based on the size of the tensor - there is no point considering plans where
        // we can store more stripes in the tile than there are in the tensor!
        NumStripes weightCopy = weightRange;
        weightCopy.m_Max      = std::min(
            weightCopy.m_Max, DivRoundUp(m_MceInputTensorShape[2], memoryWeightStripe[2]) *
                                  (isDepthwise ? 1 : DivRoundUp(m_MceOutputTensorShape[3], memoryWeightStripe[3])));
        weightCopy.m_Min = std::min(weightCopy.m_Min, weightCopy.m_Max);
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
        const bool packBoundaryVertical = (GetWidth(mceInputStripe) < GetWidth(inputShape)) ||
                                          (GetChannels(mceInputStripe) < GetChannels(inputShape));
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
            dmaOnlyInfo.m_Input  = { inputCopy, memoryInputStripe };
            dmaOnlyInfo.m_Output = { outputCopy, memoryOutputStripe };
            outStripeInfos.m_DmaOnlyInfos.insert(dmaOnlyInfo);
        }
    };

    // Limit the minimum number of blocks per stripe to be such that the PLE outputs at least one brick group
    const uint32_t baseMceInputHeight = std::max(
        blockConfig.m_BlockHeight(), GetHeight(m_Capabilities.GetBrickGroupShape()) / m_PleShapeMultiplier.m_H);
    const uint32_t baseMceInputWidth =
        std::max(blockConfig.m_BlockWidth(), GetWidth(m_Capabilities.GetBrickGroupShape()) / m_PleShapeMultiplier.m_W);
    const uint32_t baseMceIfm = numOgs / m_MceShapeMultiplier.m_C;

    // Create some helpers to loop over potential stripe shapes. We create both 'inclusive' and 'exclusive' versions,
    // as in some cases we want to include stripes that cover the full tensor, and in others we don't.
    const StripeShapeLoop mceInputWidthLoopExcl =
        StripeShapeLoop::Exclusive(GetWidth(m_MceInputTensorShape), baseMceInputWidth,
                                   m_StripeConfig.blockWidthMultiplier.min, m_StripeConfig.blockWidthMultiplier.max);
    const StripeShapeLoop mceInputHeightLoopExcl =
        StripeShapeLoop::Exclusive(GetHeight(m_MceInputTensorShape), baseMceInputHeight,
                                   m_StripeConfig.blockHeightMultiplier.min, m_StripeConfig.blockHeightMultiplier.max);
    const StripeShapeLoop mceIfmLoopExcl =
        StripeShapeLoop::Exclusive(GetChannels(m_MceInputTensorShape), baseMceIfm,
                                   m_StripeConfig.ifmDepthMultiplier.min, m_StripeConfig.ifmDepthMultiplier.max);
    const StripeShapeLoop mceOfmLoopExcl =
        StripeShapeLoop::Exclusive(GetChannels(m_MceOutputTensorShape), baseMceIfm,
                                   m_StripeConfig.ofmDepthMultiplier.min, m_StripeConfig.ofmDepthMultiplier.max);
    const StripeShapeLoop mceInputWidthLoopIncl =
        StripeShapeLoop::Inclusive(GetWidth(m_MceInputTensorShape), baseMceInputWidth,
                                   m_StripeConfig.blockWidthMultiplier.min, m_StripeConfig.blockWidthMultiplier.max);
    const StripeShapeLoop mceInputHeightLoopIncl =
        StripeShapeLoop::Inclusive(GetHeight(m_MceInputTensorShape), baseMceInputHeight,
                                   m_StripeConfig.blockHeightMultiplier.min, m_StripeConfig.blockHeightMultiplier.max);
    const StripeShapeLoop mceIfmLoopIncl =
        StripeShapeLoop::Inclusive(GetChannels(m_MceInputTensorShape), baseMceIfm,
                                   m_StripeConfig.ifmDepthMultiplier.min, m_StripeConfig.ifmDepthMultiplier.max);
    const StripeShapeLoop mceOfmLoopIncl =
        StripeShapeLoop::Inclusive(GetChannels(m_MceOutputTensorShape), baseMceIfm,
                                   m_StripeConfig.ofmDepthMultiplier.min, m_StripeConfig.ofmDepthMultiplier.max);
    ETHOSN_UNUSED(mceInputWidthLoopExcl);    // Unused but kept above for consistency and potential future use.
    ETHOSN_UNUSED(mceInputHeightLoopExcl);
    ETHOSN_UNUSED(mceOfmLoopIncl);

    const TensorShape& outputShape = m_PleOutputTensorShape;

    // Use the minimum stripe size possible to minimize the time before processing.
    // Try splitting height first.
    if (stripeConfig.splits.mceAndPleOutputHeight)
    {
        TensorShape mceInputEncoding  = { 0, baseMceInputHeight, 0, 0 };
        const TensorShape& inputShape = m_MceInputTensorShape;
        TensorShape mceInputStripe    = CreateStripe(m_MceInputTensorShape, mceInputEncoding, brickDepth);

        TensorShape mceOutputEncoding = mceInputEncoding * m_MceShapeMultiplier;
        TensorShape mceOutputStripe   = CreateStripe(mceOutputShape, mceOutputEncoding, brickDepth);

        TensorShape pleInputStripe    = mceOutputStripe;
        TensorShape pleOutputEncoding = mceOutputEncoding * m_PleShapeMultiplier;
        TensorShape pleOutputStripe   = CreateStripe(outputShape, pleOutputEncoding, brickDepth);

        TensorShape memoryOutputStripe = CreateStripe(m_PleOutputTensorShape, pleOutputEncoding, brickDepth);

        AddStripeInfos(mceInputStripe, mceOutputStripe, pleInputStripe, pleOutputStripe, mceInputStripe,
                       memoryOutputStripe, mceOutputStripe, inputShape, outputShape);
    }

    // Split only input in height while the output is full tensor.
    if (stripeConfig.splits.mceOutputHeightOnly)
    {
        TensorShape mceInputEncoding  = { 0, baseMceInputHeight, 0, 0 };
        const TensorShape& inputShape = m_MceInputTensorShape;
        TensorShape mceInputStripe    = CreateStripe(m_MceInputTensorShape, mceInputEncoding, brickDepth);

        TensorShape mceOutputEncoding = mceInputEncoding * m_MceShapeMultiplier;
        TensorShape mceOutputStripe   = CreateStripe(mceOutputShape, mceOutputEncoding, brickDepth);

        TensorShape pleInputStripe    = mceOutputStripe;
        TensorShape pleOutputEncoding = mceOutputEncoding * m_PleShapeMultiplier;
        TensorShape pleOutputStripe   = CreateStripe(outputShape, pleOutputEncoding, brickDepth);

        TensorShape memoryOutputEncoding = { 0, 0, 0, 0 };
        TensorShape memoryOutputStripe   = CreateStripe(outputShape, memoryOutputEncoding, brickDepth);

        AddStripeInfos(mceInputStripe, mceOutputStripe, pleInputStripe, pleOutputStripe, mceInputStripe,
                       memoryOutputStripe, mceOutputStripe, inputShape, outputShape);
    }

    // Try splitting width.
    if (stripeConfig.splits.widthOnly)
    {
        TensorShape mceInputEncoding  = { 0, 0, baseMceInputWidth, 0 };
        const TensorShape& inputShape = m_MceInputTensorShape;
        TensorShape mceInputStripe    = CreateStripe(m_MceInputTensorShape, mceInputEncoding, brickDepth);

        TensorShape mceOutputEncoding = mceInputEncoding * m_MceShapeMultiplier;
        TensorShape mceOutputStripe   = CreateStripe(mceOutputShape, mceOutputEncoding, brickDepth);

        TensorShape pleInputStripe    = mceOutputStripe;
        TensorShape pleOutputEncoding = mceOutputEncoding * m_PleShapeMultiplier;
        TensorShape pleOutputStripe   = CreateStripe(outputShape, pleOutputEncoding, brickDepth);

        TensorShape memoryOutputStripe = CreateStripe(m_PleOutputTensorShape, pleOutputEncoding, brickDepth);

        AddStripeInfos(mceInputStripe, mceOutputStripe, pleInputStripe, pleOutputStripe, mceInputStripe,
                       memoryOutputStripe, mceOutputStripe, inputShape, outputShape);
    }

    if (cascadeType == CascadeType::Lonely)
    {
        // Inclusive loops so that we generate plans that split only in width or height, but with larger stripe shapes
        // than the non-lonely plans above.
        for (uint32_t mceInputStripeHeight : mceInputHeightLoopIncl)
        {
            for (uint32_t mceInputStripeWidth : mceInputWidthLoopIncl)
            {
                // Try splitting width and height.
                if (stripeConfig.splits.widthHeight)
                {
                    TensorShape mceInputEncoding  = { 0, mceInputStripeHeight, mceInputStripeWidth, 0 };
                    const TensorShape& inputShape = m_MceInputTensorShape;
                    TensorShape mceInputStripe    = CreateStripe(m_MceInputTensorShape, mceInputEncoding, brickDepth);

                    TensorShape mceOutputEncoding = mceInputEncoding * m_MceShapeMultiplier;
                    TensorShape mceOutputStripe   = CreateStripe(mceOutputShape, mceOutputEncoding, brickDepth);

                    TensorShape pleInputStripe    = mceOutputStripe;
                    TensorShape pleOutputEncoding = mceOutputEncoding * m_PleShapeMultiplier;
                    TensorShape pleOutputStripe   = CreateStripe(outputShape, pleOutputEncoding, brickDepth);

                    TensorShape memoryOutputStripe =
                        CreateStripe(m_PleOutputTensorShape, pleOutputEncoding, brickDepth);

                    AddStripeInfos(mceInputStripe, mceOutputStripe, pleInputStripe, pleOutputStripe, mceInputStripe,
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
                // Exclusive loop as we already have a no-split plan further down
                for (uint32_t mceIfmStripeDepth : mceIfmLoopExcl)
                {
                    // With depthwise each only OFM needs 1 IFM.
                    TensorShape mceInputEncoding  = { 0, 0, 0, mceIfmStripeDepth };
                    const TensorShape& inputShape = m_MceInputTensorShape;
                    TensorShape mceInputStripe    = CreateStripe(m_MceInputTensorShape, mceInputEncoding, brickDepth);

                    TensorShape mceOutputEncoding = mceInputEncoding * m_MceShapeMultiplier;
                    TensorShape mceOutputStripe   = CreateStripe(mceOutputShape, mceOutputEncoding, numOgs);

                    TensorShape pleInputStripe    = mceOutputStripe;
                    TensorShape pleOutputEncoding = mceOutputEncoding * m_PleShapeMultiplier;
                    TensorShape pleOutputStripe   = CreateStripe(outputShape, pleOutputEncoding, numOgs);

                    TensorShape memoryOutputStripe =
                        CreateStripe(m_PleOutputTensorShape, pleOutputEncoding, brickDepth);

                    AddStripeInfos(mceInputStripe, mceOutputStripe, pleInputStripe, pleOutputStripe, mceInputStripe,
                                   memoryOutputStripe, mceOutputStripe, inputShape, outputShape);
                }
            }

            // Try split height width and output depth and input depth.
            if (stripeConfig.splits.widthHeightOutputDepthInputDepth)
            {
                // Inclusive loops so that we generate plans that split only in one or two of the dimensions,
                // but with larger stripe shapes than the non-lonely plans above.
                for (uint32_t mceInputStripeHeight : mceInputHeightLoopIncl)
                {
                    for (uint32_t mceInputStripeWidth : mceInputWidthLoopIncl)
                    {
                        for (uint32_t mceIfmStripeDepth : mceIfmLoopIncl)
                        {
                            TensorShape mceInputEncoding  = { 0, mceInputStripeHeight, mceInputStripeWidth,
                                                             mceIfmStripeDepth };
                            const TensorShape& inputShape = m_MceInputTensorShape;
                            TensorShape mceInputStripe =
                                CreateStripe(m_MceInputTensorShape, mceInputEncoding, brickDepth);

                            TensorShape mceOutputEncoding = mceInputEncoding * m_MceShapeMultiplier;
                            TensorShape mceOutputStripe   = CreateStripe(mceOutputShape, mceOutputEncoding, numOgs);

                            TensorShape pleInputStripe    = mceOutputStripe;
                            TensorShape pleOutputEncoding = mceOutputEncoding * m_PleShapeMultiplier;
                            TensorShape pleOutputStripe   = CreateStripe(outputShape, pleOutputEncoding, numOgs);

                            TensorShape memoryOutputStripe =
                                CreateStripe(m_PleOutputTensorShape, pleOutputEncoding, brickDepth);

                            AddStripeInfos(mceInputStripe, mceOutputStripe, pleInputStripe, pleOutputStripe,

                                           mceInputStripe, memoryOutputStripe, mceOutputStripe, inputShape,
                                           outputShape);
                        }
                    }
                }
            }
        }

        // Try split depth for compute but the memory buffer is the full tensor
        // e.g. strategy 1 cascading.
        if (stripeConfig.splits.outputDepthInputDepth)
        {
            TensorShape mceInputEncoding  = { 0, 0, 0, baseMceIfm };
            const TensorShape& inputShape = m_MceInputTensorShape;
            TensorShape mceInputStripe    = CreateStripe(m_MceInputTensorShape, mceInputEncoding, brickDepth);

            TensorShape mceOutputEncoding = TensorShape{ 0, 0, 0, numOgs };
            TensorShape mceOutputStripe   = CreateStripe(mceOutputShape, mceOutputEncoding, numOgs);

            // PLE stripe is the full tensor, as it accumulates the full output depth
            TensorShape pleInputStripe  = CreateStripe(mceOutputShape, { 0, 0, 0, 0 }, brickDepth);
            TensorShape pleOutputStripe = CreateStripe(m_PleOutputTensorShape, { 0, 0, 0, 0 }, brickDepth);

            TensorShape memoryOutputEncoding = { 0, 0, 0, 0 };
            TensorShape memoryOutputStripe   = CreateStripe(outputShape, memoryOutputEncoding, brickDepth);
            AddStripeInfos(mceInputStripe, mceOutputStripe, pleInputStripe, pleOutputStripe, mceInputStripe,
                           memoryOutputStripe, mceOutputStripe, inputShape, outputShape);
        }
    }
    else
    {
        if (cascadeType == CascadeType::Lonely)
        {
            // Try split output depth.
            if (stripeConfig.splits.mceAndPleOutputDepth)
            {
                // Exclusive loop as we already have a no-split plan further down
                for (uint32_t mceOfmStripeDepth : mceOfmLoopExcl)
                {
                    TensorShape mceInputEncoding  = { 0, 0, 0, 0 };
                    const TensorShape& inputShape = m_MceInputTensorShape;
                    TensorShape mceInputStripe    = CreateStripe(m_MceInputTensorShape, mceInputEncoding, brickDepth);

                    TensorShape mceOutputEncoding = TensorShape{ 0, 0, 0, mceOfmStripeDepth };
                    TensorShape mceOutputStripe   = CreateStripe(mceOutputShape, mceOutputEncoding, numOgs);

                    TensorShape pleInputStripe    = mceOutputStripe;
                    TensorShape pleOutputEncoding = mceOutputEncoding * m_PleShapeMultiplier;
                    TensorShape pleOutputStripe   = CreateStripe(outputShape, pleOutputEncoding, numOgs);

                    TensorShape memoryOutputStripe =
                        CreateStripe(m_PleOutputTensorShape, pleOutputEncoding, brickDepth);

                    AddStripeInfos(mceInputStripe, mceOutputStripe, pleInputStripe, pleOutputStripe, mceInputStripe,
                                   memoryOutputStripe, mceOutputStripe, inputShape, outputShape);
                }
            }

            // Try split height width and output depth.
            if (stripeConfig.splits.widthHeightOutputDepth)
            {
                // Inclusive loops so that we generate plans that split only in width or height, but with larger stripe shapes
                // than the non-lonely plans above.
                for (uint32_t mceInputStripeHeight : mceInputHeightLoopIncl)
                {
                    for (uint32_t mceInputStripeWidth : mceInputWidthLoopIncl)
                    {
                        TensorShape mceInputEncoding  = { 0, mceInputStripeHeight, mceInputStripeWidth, 0 };
                        const TensorShape& inputShape = m_MceInputTensorShape;
                        TensorShape mceInputStripe = CreateStripe(m_MceInputTensorShape, mceInputEncoding, brickDepth);

                        TensorShape mceOutputEncoding =
                            TensorShape{ 0, mceInputStripeHeight * m_MceShapeMultiplier.m_H,
                                         mceInputStripeWidth * m_MceShapeMultiplier.m_W, numOgs };
                        TensorShape mceOutputStripe = CreateStripe(mceOutputShape, mceOutputEncoding, numOgs);

                        TensorShape pleInputStripe    = mceOutputStripe;
                        TensorShape pleOutputEncoding = mceOutputEncoding * m_PleShapeMultiplier;
                        TensorShape pleOutputStripe   = CreateStripe(outputShape, pleOutputEncoding, numOgs);

                        TensorShape memoryOutputStripe =
                            CreateStripe(m_PleOutputTensorShape, pleOutputEncoding, brickDepth);

                        AddStripeInfos(mceInputStripe, mceOutputStripe, pleInputStripe, pleOutputStripe,

                                       mceInputStripe, memoryOutputStripe, mceOutputStripe, inputShape, outputShape);
                    }
                }
            }

            // Try split input depth.
            // Note we have to limit the height and width to the block size.
            if (stripeConfig.splits.widthHeightOutputDepthInputDepth)
            {
                // Exclusive loop as we already have a no-split plan further down
                for (uint32_t mceIfmStripeDepth : mceIfmLoopExcl)
                {
                    TensorShape mceInputEncoding  = { 0, baseMceInputHeight, baseMceInputWidth, mceIfmStripeDepth };
                    const TensorShape& inputShape = m_MceInputTensorShape;
                    TensorShape mceInputStripe    = CreateStripe(m_MceInputTensorShape, mceInputEncoding, brickDepth);

                    TensorShape mceOutputEncoding = mceInputEncoding * m_MceShapeMultiplier;
                    // Because of the split in IFM depth, the MCE will have to hold and accumulate the MAC results
                    // between iterations. It can only do so across the number of OGs.
                    mceOutputEncoding[3]        = numOgs;
                    TensorShape mceOutputStripe = CreateStripe(mceOutputShape, mceOutputEncoding, numOgs);

                    TensorShape pleInputStripe    = mceOutputStripe;
                    TensorShape pleOutputEncoding = mceOutputEncoding * m_PleShapeMultiplier;
                    TensorShape pleOutputStripe   = CreateStripe(outputShape, pleOutputEncoding, numOgs);

                    TensorShape memoryOutputStripe =
                        CreateStripe(m_PleOutputTensorShape, pleOutputEncoding, brickDepth);

                    AddStripeInfos(mceInputStripe, mceOutputStripe, pleInputStripe, pleOutputStripe, mceInputStripe,
                                   memoryOutputStripe, mceOutputStripe, inputShape, outputShape);
                }
            }
        }
        // Try split depth for compute but the memory buffer is the full tensor
        // e.g. strategy 1 cascading.
        if (stripeConfig.splits.mceOutputDepthOnly)
        {
            TensorShape mceInputEncoding  = { 0, 0, 0, 0 };
            const TensorShape& inputShape = m_MceInputTensorShape;
            TensorShape mceInputStripe    = CreateStripe(m_MceInputTensorShape, mceInputEncoding, brickDepth);

            TensorShape mceOutputEncoding = TensorShape{ 0, 0, 0, numOgs };
            TensorShape mceOutputStripe   = CreateStripe(mceOutputShape, mceOutputEncoding, numOgs);

            // PLE stripe is the full tensor, as it accumulates the full output depth
            TensorShape pleInputStripe  = CreateStripe(mceOutputShape, { 0, 0, 0, 0 }, brickDepth);
            TensorShape pleOutputStripe = CreateStripe(m_PleOutputTensorShape, { 0, 0, 0, 0 }, brickDepth);

            TensorShape memoryOutputEncoding = { 0, 0, 0, 0 };
            TensorShape memoryOutputStripe   = CreateStripe(outputShape, memoryOutputEncoding, brickDepth);
            AddStripeInfos(mceInputStripe, mceOutputStripe, pleInputStripe, pleOutputStripe, mceInputStripe,
                           memoryOutputStripe, mceOutputStripe, inputShape, outputShape);
        }
    }

    // Don't split at all.
    // This is needed if all of the stripes above are larger than the tensor
    // and none of them are added.
    if (stripeConfig.splits.none)
    {
        TensorShape mceInputEncoding  = { 0, 0, 0, 0 };
        TensorShape mceInputStripe    = CreateStripe(m_MceInputTensorShape, mceInputEncoding, brickDepth);
        const TensorShape& inputShape = m_MceInputTensorShape;

        TensorShape mceOutputEncoding = mceInputEncoding * m_MceShapeMultiplier;
        TensorShape mceOutputStripe   = CreateStripe(mceOutputShape, mceOutputEncoding, brickDepth);

        TensorShape pleInputStripe = mceOutputStripe;

        TensorShape pleOutputEncoding = mceOutputEncoding * m_PleShapeMultiplier;
        TensorShape pleOutputStripe   = CreateStripe(m_PleOutputTensorShape, pleOutputEncoding, brickDepth);

        AddStripeInfos(mceInputStripe, mceOutputStripe, pleInputStripe, pleOutputStripe, mceInputStripe,
                       pleOutputStripe, mceOutputStripe, inputShape, outputShape);
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
                       NumStripesType numPleInputMemoryStripes,
                       const TensorShape& tensorShape,
                       const TensorShape& pleInputMemoryShape,
                       const QuantizationInfo& quantInfo,
                       DataType dataType,
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
    buffer->m_DataType        = dataType;
    buffer->m_SlotSizeInBytes = utils::CalculateBufferSize(buffer->m_StripeShape, buffer->m_Format);
    buffer->m_SizeInBytes     = buffer->m_SlotSizeInBytes * numStripesInTile;

    buffer->m_QuantizationInfo = quantInfo;
    return buffer;
}

std::pair<Buffer*, Op*> AddPleToOpGraph(OwnedOpGraph& opGraph,
                                        const TensorShape& memoryOutputShape,
                                        impl::NumMemoryStripes& numMemoryStripes,
                                        std::unique_ptr<Op> pleOp,
                                        const TensorShape& outputShape,
                                        const QuantizationInfo& outputQuantInfo,
                                        DataType outputDataType,
                                        const std::set<uint32_t>& sourceOperationIds)
{
    auto& buffers      = opGraph.GetBuffers();
    Op* op             = opGraph.AddOp(std::move(pleOp));
    op->m_OperationIds = sourceOperationIds;

    opGraph.AddBuffer(std::make_unique<Buffer>(Location::Sram, GetFormat(Location::Sram), TraversalOrder::Xyz));
    auto pleOutBuffer = buffers.back();
    opGraph.SetProducer(pleOutBuffer, op);

    pleOutBuffer->m_DataType        = outputDataType;
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
