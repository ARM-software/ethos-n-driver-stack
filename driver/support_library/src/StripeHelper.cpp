//
// Copyright © 2021-2025 Arm Limited.
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

    auto removeBlockConfig = [&result](BlockConfig b) {
        result.blockConfigs.erase(std::remove(result.blockConfigs.begin(), result.blockConfigs.end(), b),
                                  result.blockConfigs.end());
    };

    if (!compilationOptions.m_BlockConfig8x8)
    {
        removeBlockConfig(BlockConfig{ 8u, 8u });
    }
    if (!compilationOptions.m_BlockConfig8x16)
    {
        removeBlockConfig(BlockConfig{ 8u, 16u });
    }
    if (!compilationOptions.m_BlockConfig16x8)
    {
        removeBlockConfig(BlockConfig{ 16u, 8u });
    }
    if (!compilationOptions.m_BlockConfig16x16)
    {
        removeBlockConfig(BlockConfig{ 16u, 16u });
    }
    if (!compilationOptions.m_BlockConfig32x8)
    {
        removeBlockConfig(BlockConfig{ 32u, 8u });
    }
    if (!compilationOptions.m_BlockConfig8x32)
    {
        removeBlockConfig(BlockConfig{ 8u, 32u });
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
            throw std::runtime_error("Error in stripe config file at line " + std::to_string(lineNumber) + ": " +
                                     std::move(msg));
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
                        else if (name == "Splits.None")
                        {
                            result.splits.none = valueBool();
                        }
                        else if (std::regex_match(name, match, blockConfigRegex))
                        {
                            uint32_t w = std::atoi(match[1].str().c_str());
                            uint32_t h = std::atoi(match[2].str().c_str());
                            BlockConfig b{ w, h };
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
                        else if (name == "IfmNumStripes.Min")
                        {
                            result.ifmNumStripes.min = valueUInt();
                        }
                        else if (name == "IfmNumStripes.Max")
                        {
                            result.ifmNumStripes.max = valueUInt();
                        }
                        else if (name == "WeightNumStripes.Min")
                        {
                            result.weightNumStripes.min = valueUInt();
                        }
                        else if (name == "WeightNumStripes.Max")
                        {
                            result.weightNumStripes.max = valueUInt();
                        }
                        else if (name == "OfmNumStripes.Min")
                        {
                            result.ofmNumStripes.min = valueUInt();
                        }
                        else if (name == "OfmNumStripes.Max")
                        {
                            result.ofmNumStripes.max = valueUInt();
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

bool IsSramBufferCompatibleWithDramBuffer(const SramBuffer& sramBuffer,
                                          const DramBuffer& dramBuffer,
                                          const TensorShape& dramOffset)
{
    return IsSramBufferCompatibleWithDramBuffer(sramBuffer.m_TensorShape, sramBuffer.m_StripeShape,
                                                sramBuffer.m_ForbidFcafWide, sramBuffer.m_PackedBoundaryThickness,
                                                dramBuffer.m_Format, dramBuffer.m_TensorShape, dramOffset);
}

bool IsSramBufferCompatibleWithDramBuffer(const SramBuffer& sramBuffer,
                                          BufferFormat dramFormat,
                                          const TensorShape& dramTensorShape,
                                          const TensorShape& dramOffset)
{
    return IsSramBufferCompatibleWithDramBuffer(sramBuffer.m_TensorShape, sramBuffer.m_StripeShape,
                                                sramBuffer.m_ForbidFcafWide, sramBuffer.m_PackedBoundaryThickness,
                                                dramFormat, dramTensorShape, dramOffset);
}

bool IsSramBufferCompatibleWithDramBuffer(const TensorShape& sramTensorShape,
                                          const TensorShape& stripeShape,
                                          bool forbidFcafWide,
                                          const PackedBoundaryThickness& packedBoundaryThickness,
                                          BufferFormat dramFormat,
                                          const TensorShape& dramTensorShape,
                                          const TensorShape& dramOffset)
{
    // If the copy involves a reshape (tensor shape changes to one with the same number of elements,
    // not the same as a sub-tensor which has different number of elements), then it must be NHWC
    TensorShape dramTensorShapeNoReshape = dramTensorShape;
    if (sramTensorShape != dramTensorShape && GetNumElements(sramTensorShape) == GetNumElements(dramTensorShape))
    {
        if (dramFormat != BufferFormat::NHWC)
        {
            return false;
        }
        // Do the rest of the checks with the un-reshaped tensor, for stripe compatiblity checking etc.
        // This is because we use the SRAM tensor shape in the command we send to the firmware, not the
        // DRAM one.
        dramTensorShapeNoReshape = sramTensorShape;
    }

    // If there is an offset into the DRAM tensor, check that the offset is aligned appropriately for this
    // format.
    TensorShape requiredMultiple = { 0, 0, 0, 0 };
    switch (dramFormat)
    {
        case BufferFormat::NCHW:    // intentional fallthrough
        case BufferFormat::NHWC:
        {
            // No offset in C is allowed
            // However we allow splitting in depth only if the width is 1. When the width is 1 the firmware can support splitting in depth,
            // but for other cases it can't (this isn't strictly true, but is a conservative approximation - what matters
            // here is that we support at least the cases we claim to, which is when width == 1 - see IsTensorDepthSupported).
            uint32_t channelMultiple = GetWidth(dramTensorShapeNoReshape) == 1 ? 1 : 0xffffffff;
            requiredMultiple         = { 1, 1, 1, channelMultiple };
            break;
        }
        case BufferFormat::NHWCB:
            requiredMultiple = g_BrickGroupShape;
            break;
        case BufferFormat::FCAF_WIDE:
            requiredMultiple = g_FcafWideCellShape;
            break;
        case BufferFormat::FCAF_DEEP:
            requiredMultiple = g_FcafDeepCellShape;
            break;
        default:
            assert(false);
    }

    for (int axis = 1; axis <= 3; ++axis)
    {
        if (dramOffset[axis] % requiredMultiple[axis] != 0)
        {
            return false;
        }
    }

    // NHWC can't split depth except when width is 1 as described as above
    if (dramFormat == BufferFormat::NHWC &&
        utils::GetChannels(stripeShape) < utils::GetChannels(dramTensorShapeNoReshape) &&
        utils::GetWidth(dramTensorShapeNoReshape) > 1)
    {
        return false;
    }

    // FCAF requires certain stripe shapes
    if (dramFormat == BufferFormat::FCAF_DEEP &&
        !IsCompressionFormatCompatibleWithStripeShape(CompilerDataCompressedFormat::FCAF_DEEP, stripeShape,
                                                      dramTensorShapeNoReshape))
    {
        return false;
    }
    // FCAF requires certain stripe shapes
    if (dramFormat == BufferFormat::FCAF_WIDE &&
        !IsCompressionFormatCompatibleWithStripeShape(CompilerDataCompressedFormat::FCAF_WIDE, stripeShape,
                                                      dramTensorShapeNoReshape))
    {
        return false;
    }

    // Packed boundary data only supported with NHWCB and FCAF
    if (dramFormat != BufferFormat::NHWCB && dramFormat != BufferFormat::FCAF_DEEP &&
        dramFormat != BufferFormat::FCAF_WIDE && packedBoundaryThickness.AnyNonZero())
    {
        return false;
    }

    // Explicit foridding
    if (forbidFcafWide && dramFormat == BufferFormat::FCAF_WIDE)
    {
        return false;
    }

    return true;
}

BufferFormat GetBestDramBufferFormat(const std::vector<const SramBuffer*>& sramBuffers,
                                     const CompilationOptions& compilationOptions,
                                     const std::set<PartId>& debugPartIds,
                                     const DebuggingContext& debuggingContext)
{
    // cppcheck-suppress duplicateAssignExpression
    bool fcafDeep = compilationOptions.m_EnableIntermediateCompression;
    bool fcafWide = compilationOptions.m_EnableIntermediateCompression;

    // All the SRAM buffers should have the same shape, and this will be the same shape as the DRAM buffer.
    assert(sramBuffers.size() >= 1);
    TensorShape tensorShape = sramBuffers[0]->m_TensorShape;
    for (const Buffer* b : sramBuffers)
    {
        assert(b->m_TensorShape == tensorShape);
        ETHOSN_UNUSED(b);
    }

    // If a debug preferred dram format has been set, use that regardless of whether it is compatible or not
    // (intended for debugging only)
    utils::Optional<BufferFormat> preferred = debuggingContext.GetPreferredDramFormat(debugPartIds);
    if (preferred.has_value())
    {
        return preferred.value();
    }

    for (const SramBuffer* b : sramBuffers)
    {
        if (!IsSramBufferCompatibleWithDramBuffer(*b, BufferFormat::FCAF_DEEP, tensorShape, { 0, 0, 0, 0 }))
        {
            fcafDeep = false;
        }
        if (!IsSramBufferCompatibleWithDramBuffer(*b, BufferFormat::FCAF_WIDE, tensorShape, { 0, 0, 0, 0 }))
        {
            fcafWide = false;
        }
        // We'll fall back to NHWCB if neither FCAF formats work, so sanity check that NHWCB is valid.
        assert(IsSramBufferCompatibleWithDramBuffer(*b, BufferFormat::NHWCB, tensorShape, { 0, 0, 0, 0 }));
    }

    if (fcafDeep)
    {
        return BufferFormat::FCAF_DEEP;
    }
    else if (fcafWide)
    {
        return BufferFormat::FCAF_WIDE;
    }
    return BufferFormat::NHWCB;
}

/// Creates an SRAM buffer for use in a glue which DMAs stuff into and out of SRAM.
/// The code attempts to choose an optimal stripe shape.
std::unique_ptr<SramBuffer> MakeGlueIntermediateSramBuffer(const TensorShape& shape,
                                                           const QuantizationInfo& quantInfo,
                                                           DataType dataType,
                                                           const std::vector<BufferFormat>& compatibleDramBufferFormats,
                                                           const HardwareCapabilities& caps,
                                                           uint32_t minWidthMultiplier,
                                                           uint32_t maxWidthMultiplier,
                                                           uint32_t minHeightMultiplier,
                                                           uint32_t maxHeightMultiplier,
                                                           uint32_t minDepthMultiplier,
                                                           uint32_t maxDepthMultiplier)
{
    // Calculate minimum stripe size, based on the DRAM format(s) that this buffer needs to be compatible with
    uint32_t baseWidth  = utils::GetWidth(g_BrickGroupShape);
    uint32_t baseHeight = utils::GetHeight(g_BrickGroupShape);
    uint32_t baseDepth  = utils::GetChannels(g_BrickGroupShape);
    for (BufferFormat format : compatibleDramBufferFormats)
    {
        // We always need at least one brick group (even for NHWC)
        TensorShape minStripeShape = g_BrickGroupShape;
        switch (format)
        {
            case BufferFormat::NCHW:    // intentional fallthrough
            case BufferFormat::NHWC:
                // The firmware cannot split NHWC tensors along channels, so we must use the full depth.
                // However we allow splitting in depth only if the width is 1. When the width is 1 the firmware can support splitting in depth,
                // but for other cases it can't (this isn't strictly true, but is a conservative approximation - what matters
                // here is that we support at least the cases we claim to, which is when width == 1 - see IsTensorDepthSupported).
                minStripeShape[3] =
                    GetWidth(shape) == 1
                        ? GetChannels(g_BrickGroupShape)
                        : utils::RoundUpToNearestMultiple(shape[3], utils::GetChannels(g_BrickGroupShape));
                break;
            case BufferFormat::NHWCB:
                minStripeShape = g_BrickGroupShape;
                break;
            case BufferFormat::FCAF_DEEP:
                minStripeShape = g_FcafDeepCellShape;
                break;
            case BufferFormat::FCAF_WIDE:
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
    TensorShape bestStripeShape = {};
    uint32_t bestScore          = 0;
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

    if (bestStripeShape == TensorShape{})
    {
        throw InternalErrorException("Failed to find valid stripe shape for intermediate SRAM buffer");
    }

    std::unique_ptr<SramBuffer> sramBuffer = SramBufferBuilder()
                                                 .AddFormat(BufferFormat::NHWCB)
                                                 .AddDataType(dataType)
                                                 .AddTensorShape(shape)
                                                 .AddQuantization(quantInfo)
                                                 .AddStripeShape(bestStripeShape)
                                                 .AddNumStripes(1)
                                                 .AddSlotSize(utils::TotalSizeBytesNHWCB(bestStripeShape))
                                                 .AddTraversalOrder(TraversalOrder::Xyz);

    sramBuffer->m_Offset = 0;    // Nothing else should be resident in SRAM at this point, so we can use any address

    // Sanity check that the SRAM buffer we created is valid for DMAs to/from the DRAM buffers
    for (BufferFormat format : compatibleDramBufferFormats)
    {
        assert(IsSramBufferCompatibleWithDramBuffer(*sramBuffer, format, shape, { 0, 0, 0, 0 }));
        ETHOSN_UNUSED(format);
    }

    return sramBuffer;
}

StripeGenerator::StripeGenerator(const TensorShape& mceInput,
                                 const TensorShape& mceOutput,
                                 const TensorShape& pleOutput,
                                 uint32_t kernelHeight,
                                 uint32_t kernelWidth,
                                 Padding padding,
                                 uint32_t upscaleFactor,
                                 command_stream::MceOperation op,
                                 PleOperation pleOp,
                                 const utils::ShapeMultiplier& mceShapeMult,
                                 const utils::ShapeMultiplier& pleShapeMult,
                                 const HardwareCapabilities& m_Capabilities,
                                 const StripeConfig& stripeConfig)
    : m_MceInputTensorShape(mceInput)
    , m_MceOutputTensorShape(mceOutput)
    , m_PleOutputTensorShape(pleOutput)
    , m_KernelHeight(kernelHeight)
    , m_KernelWidth(kernelWidth)
    , m_Padding(padding)
    , m_UpscaleFactor(upscaleFactor)
    , m_Operation(op)
    , m_KernelOperation(pleOp)
    , m_MceShapeMultiplier(mceShapeMult)
    , m_PleShapeMultiplier(pleShapeMult)
    , m_Capabilities(m_Capabilities)
    , m_StripeConfig(stripeConfig)
{}

void StripeGenerator::CreateNumStripes(CascadeType cascadeType,
                                       uint32_t minStripesInIfmTile,
                                       BoundaryRequirements outputBoundaryRequirements,
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
            numStripesInput = { minStripesInIfmTile, minStripesInIfmTile + 1 };
            // Multiple output stripes may needed because the follow layers may require multiple buffers due to boundary data.
            if ((outputBoundaryRequirements.m_NeedsBeforeX || outputBoundaryRequirements.m_NeedsBeforeY) &&
                (outputBoundaryRequirements.m_NeedsAfterX || outputBoundaryRequirements.m_NeedsAfterY))
            {
                numStripesOutput = { 3, 3 };
            }
            else if (outputBoundaryRequirements.m_NeedsBeforeX || outputBoundaryRequirements.m_NeedsBeforeY ||
                     outputBoundaryRequirements.m_NeedsAfterX || outputBoundaryRequirements.m_NeedsAfterY)
            {
                numStripesOutput = { 2, 2 };
            }
            else
            {
                numStripesOutput = { 1, 1 };
            }
            numStripesWeights  = { 1, 2 };
            numStripesPleInput = { 0, 0 };
            break;
        }
        case CascadeType::Lonely:
        {
            numStripesInput    = { minStripesInIfmTile, minStripesInIfmTile + 1 };
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
    if (m_KernelOperation == PleOperation::MAXPOOL_3X3_2_2_EVEN ||
        m_KernelOperation == PleOperation::MAXPOOL_3X3_2_2_ODD)
    {
        if (cascadeType == CascadeType::Beginning)
        {
            result.DisableSplitHeight();
            result.DisableSplitWidth();
            result.DisableSplitInputDepth();
            result.DisableSplitOutputDepth();
        }
        // Note that there are also restrictions for Lonely plans, but these are applied in AddStripeInfos
        // as more information is needed than is available here.
    }

    /* The transpose operator requires a full tensor present. */
    if (m_KernelOperation == PleOperation::TRANSPOSE_XY)
    {
        result.DisableAllSplits();
        result.splits.none = 1;
    }

    return result;
}
StripeInfos StripeGenerator::GenerateStripes(CascadeType cascadeType,
                                             BoundaryRequirements outputBoundaryRequirements,
                                             utils::Optional<PlanPriority> priorityFilter) const
{
    StripeInfos result;
    for (auto&& blockConfig : m_StripeConfig.blockConfigs)
    {
        GenerateStripes(blockConfig, cascadeType, outputBoundaryRequirements, priorityFilter, result);
    }
    return result;
}

inline uint32_t getSpaceLeft(uint32_t tensor_size, uint32_t stripe_size)
{
    return (stripe_size - (tensor_size % stripe_size)) % stripe_size;
}

int32_t checkForPOSB(uint32_t tensor_size, uint32_t stripe_size, uint32_t padding, uint32_t block_config)
{
    if (getSpaceLeft(tensor_size, stripe_size) >= padding)
    {
        return 0;
    }

    uint32_t new_stripe_size = stripe_size;
    uint32_t space_left      = getSpaceLeft(tensor_size, stripe_size);
    do
    {
        new_stripe_size += block_config;
        space_left = getSpaceLeft(tensor_size, new_stripe_size);
    } while (new_stripe_size < 2 * stripe_size && space_left < padding);

    if (new_stripe_size >= 2 * stripe_size)
    {
        return -1;
    }
    return static_cast<int32_t>(new_stripe_size - stripe_size);
}

void StripeGenerator::GenerateStripes(const BlockConfig blockConfig,
                                      CascadeType cascadeType,
                                      BoundaryRequirements outputBoundaryRequirements,
                                      utils::Optional<PlanPriority> priorityFilter,
                                      StripeInfos& outStripeInfos) const
{
    using namespace utils;

    const uint32_t numOgs   = m_Capabilities.GetNumberOfOgs();
    const uint32_t numSrams = m_Capabilities.GetNumberOfSrams();

    // Set Stripe split restrictions, depending on the Ple kernel type.
    StripeConfig stripeConfig = ApplyPleKernelSplitRestrictions(cascadeType);

    const bool isDepthwise           = m_Operation == ethosn::command_stream::MceOperation::DEPTHWISE_CONVOLUTION;
    const bool isConv2d              = m_Operation == ethosn::command_stream::MceOperation::CONVOLUTION;
    const TensorShape mceOutputShape = m_MceOutputTensorShape;

    const bool isHeightIncreased =
        GetHeight(m_PleOutputTensorShape) > (GetHeight(m_MceInputTensorShape) * m_UpscaleFactor);
    const bool isWidthIncreased =
        GetWidth(m_PleOutputTensorShape) > (GetWidth(m_MceInputTensorShape) * m_UpscaleFactor);

    // Indicates that checks for padding over stripe boundaries should be performed.
    bool shouldCheckForPOSB =
        (isDepthwise || isConv2d) && (((m_Padding.GetVerticalPadding() > 0) && isHeightIncreased) ||
                                      ((m_Padding.GetHorizontalPadding() > 0) && isWidthIncreased));

    // This method is intended to be called first with PlanPriority::High and after and only if needed
    // with PlanPriority::Low.
    // Splitting input depth(for regular conv) is always worse, so these are low priority plans, for depthwise
    // conv we treat it all as HIGH
    if (priorityFilter == PlanPriority::High && !isDepthwise)
    {
        stripeConfig.DisableSplitInputDepth();
    }
    else if (priorityFilter == PlanPriority::Low && !isDepthwise)
    {
        stripeConfig.DisableAllSplits();
        stripeConfig.splits.widthHeightOutputDepthInputDepth = true;
        stripeConfig.splits.outputDepthInputDepth            = true;
    }

    // Note use of numSrams rather than numOgs when doing depthwise as only one OG per CE is used for depthwise.
    const uint32_t baseMceOfm = (isDepthwise ? numSrams : numOgs);

    // For configs with a smaller number of OGs or SRAMs, we can have stripe depth smaller than a brick group.
    const uint32_t channelRounding = std::min(GetChannels(g_BrickGroupShape), baseMceOfm);

    auto AddStripeInfos = [&](const TensorShape& mceInputStripe, const TensorShape& mceOutputStripe,
                              const TensorShape& pleInputStripe, const TensorShape& pleOutputStripe,
                              const TensorShape& memoryInputStripe, const TensorShape& memoryOutputStripe,
                              const TensorShape& memoryPleInputStripe, const TensorShape& inputShape,
                              const TensorShape& outputShape) {
        NumStripes inputRange;
        NumStripes outputRange;
        NumStripes weightRange;
        NumStripes pleInputRange;

        const NeedBoundary needBoundaryY =
            utils::GetBoundaryRequirements(m_Padding.m_Top, GetHeight(mceInputStripe), GetHeight(mceOutputStripe),
                                           m_KernelHeight, m_UpscaleFactor > 1);
        const NeedBoundary needBoundaryX = utils::GetBoundaryRequirements(
            m_Padding.m_Left, GetWidth(mceInputStripe), GetWidth(mceOutputStripe), m_KernelWidth, m_UpscaleFactor > 1);
        // IFM is traversed ZXY order (XYZ for depthwise though).
        // If the first dimension with more than one stripe needs boundary data, we need at least this many stripes in the tile.
        uint32_t minStripesInTile = 1;

        if (isDepthwise || GetChannels(mceInputStripe) >= GetChannels(inputShape))
        {
            // X first?
            if (GetWidth(mceInputStripe) < GetWidth(inputShape))
            {
                minStripesInTile = 1 + (needBoundaryX.m_Before ? 1 : 0) + (needBoundaryX.m_After ? 1 : 0);
                // If there is only 2 stripes in X, then we don't need 3 in the tile
                minStripesInTile =
                    std::min(minStripesInTile, utils::DivRoundUp(GetWidth(inputShape), GetWidth(mceInputStripe)));
            }
            // Y first?
            else if (GetHeight(mceInputStripe) < GetHeight(inputShape))
            {
                minStripesInTile = 1 + (needBoundaryY.m_Before ? 1 : 0) + (needBoundaryY.m_After ? 1 : 0);
                // If there is only 2 stripes in Y, then we don't need 3 in the tile
                minStripesInTile =
                    std::min(minStripesInTile, utils::DivRoundUp(GetHeight(inputShape), GetHeight(mceInputStripe)));
            }
        }

        CreateNumStripes(cascadeType, minStripesInTile, outputBoundaryRequirements, inputRange, outputRange,
                         weightRange, pleInputRange);

        // Limit the max number of stripes based on the size of the tensor - there is no point considering plans where
        // we can store more stripes in the tile than there are in the tensor!
        NumStripes inputCopy = inputRange;
        inputCopy.m_Max =
            std::min(inputCopy.m_Max, DivRoundUp(GetHeight(inputShape), GetHeight(memoryInputStripe)) *
                                          DivRoundUp(GetWidth(inputShape), GetWidth(memoryInputStripe)) *
                                          DivRoundUp(GetChannels(inputShape), GetChannels(memoryInputStripe)));
        inputCopy.m_Min = std::min(inputCopy.m_Min, inputCopy.m_Max);

        // Apply any stripe config overrides
        inputCopy.m_Min = std::max(inputCopy.m_Min, stripeConfig.ifmNumStripes.min);
        inputCopy.m_Max = std::min(inputCopy.m_Max, stripeConfig.ifmNumStripes.max);

        NumStripes outputCopy = outputRange;
        outputCopy.m_Max =
            std::min(outputCopy.m_Max, DivRoundUp(GetHeight(outputShape), GetHeight(memoryOutputStripe)) *
                                           DivRoundUp(GetWidth(outputShape), GetWidth(memoryOutputStripe)) *
                                           DivRoundUp(GetChannels(outputShape), GetChannels(memoryOutputStripe)));
        outputCopy.m_Min = std::min(outputCopy.m_Min, outputCopy.m_Max);

        // If splitting in height, maxpool requires at least two slots in the OFM tile because it can't
        // write a full stripe of output data until it starts the next output stripe (due to pooling windows
        // overlapping the stripe boundary).
        if (m_KernelOperation == PleOperation::MAXPOOL_3X3_2_2_EVEN ||
            m_KernelOperation == PleOperation::MAXPOOL_3X3_2_2_ODD)
        {
            if (GetHeight(pleInputStripe) < GetHeight(m_MceOutputTensorShape))
            {
                outputCopy.m_Min = std::max(outputCopy.m_Min, 2u);
            }
        }

        // Apply any stripe config overrides
        outputCopy.m_Min = std::max(outputCopy.m_Min, stripeConfig.ofmNumStripes.min);
        outputCopy.m_Max = std::min(outputCopy.m_Max, stripeConfig.ofmNumStripes.max);

        // Prevent unsupported splits for max pooling due to limitations of the PLE kernel
        if (m_KernelOperation == PleOperation::MAXPOOL_3X3_2_2_EVEN ||
            m_KernelOperation == PleOperation::MAXPOOL_3X3_2_2_ODD)
        {
            // Prevent having more than one channel per PLE, when it is also split in height.
            if (GetHeight(pleInputStripe) < GetHeight(m_MceOutputTensorShape) &&
                GetChannels(pleInputStripe) > baseMceOfm)
            {
                return;
            }

            // Prevent any splitting in width.
            // (Note this can't be done using StripeConfig::DisableSplitWidth because that is overly cautious and also
            //  disables splitting in all the dimensions, which is the only way to get a height+depth split, which is needed
            //  in some cases).
            if (GetWidth(pleInputStripe) < GetWidth(m_MceOutputTensorShape))
            {
                return;
            }
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

        // Apply any stripe config overrides
        weightCopy.m_Min = std::max(weightCopy.m_Min, stripeConfig.weightNumStripes.min);
        weightCopy.m_Max = std::min(weightCopy.m_Max, stripeConfig.weightNumStripes.max);

        const bool needMultipleIfmDepths = !isDepthwise && GetChannels(mceInputStripe) < GetChannels(inputShape);
        // Packed boundary is needed only if that dimension is not the fastest iterating
        const bool packBoundaryVertical = (GetHeight(mceInputStripe) < GetHeight(inputShape)) &&
                                          (needMultipleIfmDepths || GetWidth(mceInputStripe) < GetWidth(inputShape));
        const bool packBoundaryHorizontal = (GetWidth(mceInputStripe) < GetWidth(inputShape)) && needMultipleIfmDepths;

        PackedBoundaryThickness packedBoundaryThickness;
        // We set the packed boundary on the left and right to 16, so that it can work with FCAF_WIDE.
        // We don't yet know what DRAM format will be used, so we have to be conservative.
        // Later on, we will reduce this down to 8 if we don't end up using FCAF_WIDE
        packedBoundaryThickness.left   = (packBoundaryHorizontal && needBoundaryX.m_Before) ? 16 : 0;
        packedBoundaryThickness.top    = (packBoundaryVertical && needBoundaryY.m_Before) ? 8 : 0;
        packedBoundaryThickness.right  = (packBoundaryHorizontal && needBoundaryX.m_After) ? 16 : 0;
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

    // Determine the "base" shape of stripes - the stripe shapes we pick will be a whole multiple of this.
    // We choose a single block for this as this is the smallest size that will fully utilize the hardware.
    // Also make the base shape large enough such that the PLE outputs at least one brick group and
    // the MCE takes as input at least one brick group, which is a limitation of the firmware/hardware.
    const ShapeMultiplier mceAndPleShapeMultiplier = m_MceShapeMultiplier * m_PleShapeMultiplier;
    const uint32_t baseMceInputHeight =
        std::max({ blockConfig.m_Height / m_MceShapeMultiplier.m_H,
                   GetHeight(g_BrickGroupShape) / mceAndPleShapeMultiplier.m_H, GetHeight(g_BrickGroupShape) });
    const uint32_t baseMceInputWidth =
        std::max({ blockConfig.m_Width / m_MceShapeMultiplier.m_W,
                   GetWidth(g_BrickGroupShape) / mceAndPleShapeMultiplier.m_W, GetWidth(g_BrickGroupShape) });
    const uint32_t baseMceIfm = baseMceOfm / m_MceShapeMultiplier.m_C;

    // Create some helpers to loop over potential stripe shapes. We create both 'inclusive' and 'exclusive' versions,
    // as in some cases we want to include stripes that cover the full tensor, and in others we don't.
    const StripeShapeLoop mceInputWidthLoopExcl =
        StripeShapeLoop::Exclusive(GetWidth(m_MceInputTensorShape), baseMceInputWidth,
                                   stripeConfig.blockWidthMultiplier.min, stripeConfig.blockWidthMultiplier.max);
    const StripeShapeLoop mceInputHeightLoopExcl =
        StripeShapeLoop::Exclusive(GetHeight(m_MceInputTensorShape), baseMceInputHeight,
                                   stripeConfig.blockHeightMultiplier.min, stripeConfig.blockHeightMultiplier.max);
    const StripeShapeLoop mceIfmLoopExcl =
        StripeShapeLoop::Exclusive(GetChannels(m_MceInputTensorShape), baseMceIfm, stripeConfig.ifmDepthMultiplier.min,
                                   stripeConfig.ifmDepthMultiplier.max);
    const StripeShapeLoop mceOfmLoopExcl =
        StripeShapeLoop::Exclusive(GetChannels(m_MceOutputTensorShape), baseMceOfm, stripeConfig.ofmDepthMultiplier.min,
                                   stripeConfig.ofmDepthMultiplier.max);
    const StripeShapeLoop mceInputWidthLoopIncl =
        StripeShapeLoop::Inclusive(GetWidth(m_MceInputTensorShape), baseMceInputWidth,
                                   stripeConfig.blockWidthMultiplier.min, stripeConfig.blockWidthMultiplier.max);
    const StripeShapeLoop mceInputHeightLoopIncl =
        StripeShapeLoop::Inclusive(GetHeight(m_MceInputTensorShape), baseMceInputHeight,
                                   stripeConfig.blockHeightMultiplier.min, stripeConfig.blockHeightMultiplier.max);
    const StripeShapeLoop mceIfmLoopIncl =
        StripeShapeLoop::Inclusive(GetChannels(m_MceInputTensorShape), baseMceIfm, stripeConfig.ifmDepthMultiplier.min,
                                   stripeConfig.ifmDepthMultiplier.max);
    const StripeShapeLoop mceOfmLoopIncl =
        StripeShapeLoop::Inclusive(GetChannels(m_MceOutputTensorShape), baseMceIfm, stripeConfig.ofmDepthMultiplier.min,
                                   stripeConfig.ofmDepthMultiplier.max);
    ETHOSN_UNUSED(mceOfmLoopIncl);

    const TensorShape& outputShape = m_PleOutputTensorShape;

    if (stripeConfig.splits.mceAndPleOutputHeight)
    {
        // Exclusive loop as we already have a no-split plan further down
        for (uint32_t mceInputStripeHeight : mceInputHeightLoopExcl)
        {
            uint32_t inputHeight = mceInputStripeHeight;

            if (shouldCheckForPOSB)
            {
                int32_t heightDelta = checkForPOSB(GetHeight(m_MceInputTensorShape), inputHeight,
                                                   m_Padding.GetVerticalPadding(), blockConfig.m_Height);
                if (heightDelta < 0)
                {
                    continue;
                }
                inputHeight += heightDelta;
            }
            TensorShape mceInputEncoding  = { 0, inputHeight, 0, 0 };
            const TensorShape& inputShape = m_MceInputTensorShape;
            TensorShape mceInputStripe    = CreateStripe(m_MceInputTensorShape, mceInputEncoding, channelRounding);

            TensorShape mceOutputEncoding = mceInputEncoding * m_MceShapeMultiplier;
            TensorShape mceOutputStripe   = CreateStripe(mceOutputShape, mceOutputEncoding, channelRounding);

            TensorShape pleInputStripe    = mceOutputStripe;
            TensorShape pleOutputEncoding = mceOutputEncoding * m_PleShapeMultiplier;
            TensorShape pleOutputStripe   = CreateStripe(outputShape, pleOutputEncoding, channelRounding);

            TensorShape memoryOutputStripe = CreateStripe(m_PleOutputTensorShape, pleOutputEncoding, channelRounding);

            AddStripeInfos(mceInputStripe, mceOutputStripe, pleInputStripe, pleOutputStripe, mceInputStripe,
                           memoryOutputStripe, mceOutputStripe, inputShape, outputShape);
        }
    }

    // Split only input in height while the output is full tensor.
    if (stripeConfig.splits.mceOutputHeightOnly)
    {
        // Exclusive loop as we already have a no-split plan further down
        for (uint32_t mceInputStripeHeight : mceInputHeightLoopExcl)
        {
            uint32_t inputHeight = mceInputStripeHeight;

            if (shouldCheckForPOSB)
            {
                int32_t heightDelta = checkForPOSB(GetHeight(m_MceInputTensorShape), inputHeight,
                                                   m_Padding.GetVerticalPadding(), blockConfig.m_Height);
                if (heightDelta < 0)
                {
                    continue;
                }
                inputHeight += heightDelta;
            }
            TensorShape mceInputEncoding  = { 0, inputHeight, 0, 0 };
            const TensorShape& inputShape = m_MceInputTensorShape;
            TensorShape mceInputStripe    = CreateStripe(m_MceInputTensorShape, mceInputEncoding, channelRounding);

            TensorShape mceOutputEncoding = mceInputEncoding * m_MceShapeMultiplier;
            TensorShape mceOutputStripe   = CreateStripe(mceOutputShape, mceOutputEncoding, channelRounding);

            TensorShape pleInputStripe    = mceOutputStripe;
            TensorShape pleOutputEncoding = mceOutputEncoding * m_PleShapeMultiplier;
            TensorShape pleOutputStripe   = CreateStripe(outputShape, pleOutputEncoding, channelRounding);

            TensorShape memoryOutputEncoding = { 0, 0, 0, 0 };

            if (shouldCheckForPOSB)
            {
                memoryOutputEncoding = pleOutputEncoding;
            }

            TensorShape memoryOutputStripe = CreateStripe(outputShape, memoryOutputEncoding, channelRounding);

            AddStripeInfos(mceInputStripe, mceOutputStripe, pleInputStripe, pleOutputStripe, mceInputStripe,
                           memoryOutputStripe, mceOutputStripe, inputShape, outputShape);
        }
    }

    // Try splitting width.
    if (stripeConfig.splits.widthOnly)
    {
        // Exclusive loop as we already have a no-split plan further down
        for (uint32_t mceInputStripeWidth : mceInputWidthLoopExcl)
        {
            uint32_t inputWidth = mceInputStripeWidth;

            if (shouldCheckForPOSB)
            {
                int32_t widthDelta = checkForPOSB(GetWidth(m_MceInputTensorShape), mceInputStripeWidth,
                                                  m_Padding.GetHorizontalPadding(), blockConfig.m_Width);
                if (widthDelta < 0)
                {
                    continue;
                }
                inputWidth += widthDelta;
            }
            TensorShape mceInputEncoding  = { 0, 0, inputWidth, 0 };
            const TensorShape& inputShape = m_MceInputTensorShape;
            TensorShape mceInputStripe    = CreateStripe(m_MceInputTensorShape, mceInputEncoding, channelRounding);

            TensorShape mceOutputEncoding = mceInputEncoding * m_MceShapeMultiplier;
            TensorShape mceOutputStripe   = CreateStripe(mceOutputShape, mceOutputEncoding, channelRounding);

            TensorShape pleInputStripe    = mceOutputStripe;
            TensorShape pleOutputEncoding = mceOutputEncoding * m_PleShapeMultiplier;
            TensorShape pleOutputStripe   = CreateStripe(outputShape, pleOutputEncoding, channelRounding);

            TensorShape memoryOutputStripe = CreateStripe(m_PleOutputTensorShape, pleOutputEncoding, channelRounding);

            AddStripeInfos(mceInputStripe, mceOutputStripe, pleInputStripe, pleOutputStripe, mceInputStripe,
                           memoryOutputStripe, mceOutputStripe, inputShape, outputShape);
        }
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
                    uint32_t inputWidth  = mceInputStripeWidth;
                    uint32_t inputHeight = mceInputStripeHeight;

                    if (shouldCheckForPOSB)
                    {
                        int32_t widthDelta = checkForPOSB(GetWidth(m_MceInputTensorShape), mceInputStripeWidth,
                                                          m_Padding.GetHorizontalPadding(), blockConfig.m_Width);
                        if (widthDelta < 0)
                        {
                            continue;
                        }
                        inputWidth += widthDelta;
                        int32_t heightDelta = checkForPOSB(GetHeight(m_MceInputTensorShape), mceInputStripeHeight,
                                                           m_Padding.GetVerticalPadding(), blockConfig.m_Height);
                        if (heightDelta < 0)
                        {
                            continue;
                        }
                        inputHeight += heightDelta;
                    }

                    TensorShape mceInputEncoding  = { 0, inputHeight, inputWidth, 0 };
                    const TensorShape& inputShape = m_MceInputTensorShape;
                    TensorShape mceInputStripe = CreateStripe(m_MceInputTensorShape, mceInputEncoding, channelRounding);

                    TensorShape mceOutputEncoding = mceInputEncoding * m_MceShapeMultiplier;
                    TensorShape mceOutputStripe   = CreateStripe(mceOutputShape, mceOutputEncoding, channelRounding);

                    TensorShape pleInputStripe    = mceOutputStripe;
                    TensorShape pleOutputEncoding = mceOutputEncoding * m_PleShapeMultiplier;
                    TensorShape pleOutputStripe   = CreateStripe(outputShape, pleOutputEncoding, channelRounding);

                    TensorShape memoryOutputStripe =
                        CreateStripe(m_PleOutputTensorShape, pleOutputEncoding, channelRounding);

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
                    // With depthwise each OFM only needs 1 IFM.
                    TensorShape mceInputEncoding  = { 0, 0, 0, mceIfmStripeDepth };
                    const TensorShape& inputShape = m_MceInputTensorShape;
                    TensorShape mceInputStripe = CreateStripe(m_MceInputTensorShape, mceInputEncoding, channelRounding);

                    TensorShape mceOutputEncoding = mceInputEncoding * m_MceShapeMultiplier;
                    TensorShape mceOutputStripe   = CreateStripe(mceOutputShape, mceOutputEncoding, baseMceOfm);

                    TensorShape pleInputStripe    = mceOutputStripe;
                    TensorShape pleOutputEncoding = mceOutputEncoding * m_PleShapeMultiplier;
                    TensorShape pleOutputStripe   = CreateStripe(outputShape, pleOutputEncoding, baseMceOfm);

                    TensorShape memoryOutputStripe =
                        CreateStripe(m_PleOutputTensorShape, pleOutputEncoding, baseMceOfm);

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
                            uint32_t inputWidth  = mceInputStripeWidth;
                            uint32_t inputHeight = mceInputStripeHeight;

                            if (shouldCheckForPOSB)
                            {
                                int32_t widthDelta =
                                    checkForPOSB(GetWidth(m_MceInputTensorShape), mceInputStripeWidth,
                                                 m_Padding.GetHorizontalPadding(), blockConfig.m_Width);
                                if (widthDelta < 0)
                                {
                                    continue;
                                }
                                inputWidth += widthDelta;
                                int32_t heightDelta =
                                    checkForPOSB(GetHeight(m_MceInputTensorShape), mceInputStripeHeight,
                                                 m_Padding.GetVerticalPadding(), blockConfig.m_Height);
                                if (heightDelta < 0)
                                {
                                    continue;
                                }
                                inputHeight += heightDelta;
                            }
                            TensorShape mceInputEncoding  = { 0, inputHeight, inputWidth, mceIfmStripeDepth };
                            const TensorShape& inputShape = m_MceInputTensorShape;
                            TensorShape mceInputStripe =
                                CreateStripe(m_MceInputTensorShape, mceInputEncoding, channelRounding);

                            TensorShape mceOutputEncoding = mceInputEncoding * m_MceShapeMultiplier;
                            TensorShape mceOutputStripe   = CreateStripe(mceOutputShape, mceOutputEncoding, baseMceOfm);

                            TensorShape pleInputStripe    = mceOutputStripe;
                            TensorShape pleOutputEncoding = mceOutputEncoding * m_PleShapeMultiplier;
                            TensorShape pleOutputStripe   = CreateStripe(outputShape, pleOutputEncoding, baseMceOfm);

                            TensorShape memoryOutputStripe =
                                CreateStripe(m_PleOutputTensorShape, pleOutputEncoding, baseMceOfm);

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
            // Exclusive loop as we already have a no-split plan further down
            for (uint32_t mceIfmStripeDepth : mceIfmLoopExcl)
            {
                TensorShape mceInputEncoding  = { 0, 0, 0, mceIfmStripeDepth };
                const TensorShape& inputShape = m_MceInputTensorShape;
                TensorShape mceInputStripe    = CreateStripe(m_MceInputTensorShape, mceInputEncoding, channelRounding);

                TensorShape mceOutputEncoding = mceInputEncoding * m_MceShapeMultiplier;
                TensorShape mceOutputStripe   = CreateStripe(mceOutputShape, mceOutputEncoding, baseMceOfm);

                // PLE stripe is the full tensor, as it accumulates the full output depth
                TensorShape pleInputStripe  = CreateStripe(mceOutputShape, { 0, 0, 0, 0 }, baseMceOfm);
                TensorShape pleOutputStripe = CreateStripe(m_PleOutputTensorShape, { 0, 0, 0, 0 }, baseMceOfm);

                TensorShape memoryOutputEncoding = { 0, 0, 0, 0 };
                TensorShape memoryOutputStripe   = CreateStripe(outputShape, memoryOutputEncoding, baseMceOfm);
                AddStripeInfos(mceInputStripe, mceOutputStripe, pleInputStripe, pleOutputStripe, mceInputStripe,
                               memoryOutputStripe, mceOutputStripe, inputShape, outputShape);
            }
        }
    }
    else    // Convolution or Fully Connected
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
                    TensorShape mceInputStripe = CreateStripe(m_MceInputTensorShape, mceInputEncoding, channelRounding);

                    TensorShape mceOutputEncoding = TensorShape{ 0, 0, 0, mceOfmStripeDepth };
                    TensorShape mceOutputStripe   = CreateStripe(mceOutputShape, mceOutputEncoding, baseMceOfm);

                    TensorShape pleInputStripe    = mceOutputStripe;
                    TensorShape pleOutputEncoding = mceOutputEncoding * m_PleShapeMultiplier;
                    TensorShape pleOutputStripe   = CreateStripe(outputShape, pleOutputEncoding, baseMceOfm);

                    TensorShape memoryOutputStripe =
                        CreateStripe(m_PleOutputTensorShape, pleOutputEncoding, baseMceOfm);

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
                        uint32_t inputWidth  = mceInputStripeWidth;
                        uint32_t inputHeight = mceInputStripeHeight;

                        if (shouldCheckForPOSB)
                        {
                            int32_t widthDelta = checkForPOSB(GetWidth(m_MceInputTensorShape), mceInputStripeWidth,
                                                              m_Padding.GetHorizontalPadding(), blockConfig.m_Width);
                            if (widthDelta < 0)
                            {
                                continue;
                            }
                            inputWidth += widthDelta;
                            int32_t heightDelta = checkForPOSB(GetHeight(m_MceInputTensorShape), mceInputStripeHeight,
                                                               m_Padding.GetVerticalPadding(), blockConfig.m_Height);
                            if (heightDelta < 0)
                            {
                                continue;
                            }
                            inputHeight += heightDelta;
                        }

                        TensorShape mceInputEncoding  = { 0, inputHeight, inputWidth, 0 };
                        const TensorShape& inputShape = m_MceInputTensorShape;
                        TensorShape mceInputStripe =
                            CreateStripe(m_MceInputTensorShape, mceInputEncoding, channelRounding);

                        TensorShape mceOutputEncoding =
                            TensorShape{ 0, inputHeight * m_MceShapeMultiplier.m_H,
                                         inputWidth * m_MceShapeMultiplier.m_W, baseMceOfm };
                        TensorShape mceOutputStripe = CreateStripe(mceOutputShape, mceOutputEncoding, baseMceOfm);

                        TensorShape pleInputStripe    = mceOutputStripe;
                        TensorShape pleOutputEncoding = mceOutputEncoding * m_PleShapeMultiplier;
                        TensorShape pleOutputStripe   = CreateStripe(outputShape, pleOutputEncoding, baseMceOfm);

                        TensorShape memoryOutputStripe =
                            CreateStripe(m_PleOutputTensorShape, pleOutputEncoding, baseMceOfm);

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
                    TensorShape mceInputStripe = CreateStripe(m_MceInputTensorShape, mceInputEncoding, channelRounding);

                    TensorShape mceOutputEncoding = mceInputEncoding * m_MceShapeMultiplier;

                    // We need to check mceOutputEncoding here, because that might be more than one block, depending
                    // on baseMceInputWidth/Height (e.g. MCE/PLE shape multipliers).
                    // In this case we can't generate a valid plan, and we'd need to use a larger block config instead.
                    if (GetWidth(mceOutputEncoding) != blockConfig.m_Width ||
                        GetHeight(mceOutputEncoding) != blockConfig.m_Height)
                    {
                        continue;
                    }

                    // Because of the split in IFM depth, the MCE will have to hold and accumulate the MAC results
                    // between iterations. It can only do so across the number of OGs.
                    mceOutputEncoding[3]        = baseMceOfm;
                    TensorShape mceOutputStripe = CreateStripe(mceOutputShape, mceOutputEncoding, baseMceOfm);

                    TensorShape pleInputStripe    = mceOutputStripe;
                    TensorShape pleOutputEncoding = mceOutputEncoding * m_PleShapeMultiplier;
                    TensorShape pleOutputStripe   = CreateStripe(outputShape, pleOutputEncoding, baseMceOfm);

                    TensorShape memoryOutputStripe = CreateStripe(m_PleOutputTensorShape, pleOutputEncoding, numOgs);

                    AddStripeInfos(mceInputStripe, mceOutputStripe, pleInputStripe, pleOutputStripe, mceInputStripe,
                                   memoryOutputStripe, mceOutputStripe, inputShape, outputShape);
                }
            }
        }
        // Try split depth for compute but the memory buffer is the full tensor
        // e.g. strategy 1 cascading.
        if (stripeConfig.splits.mceOutputDepthOnly)
        {
            // Exclusive loop as we already have a no-split plan further down
            for (uint32_t mceOfmStripeDepth : mceOfmLoopExcl)
            {
                TensorShape mceInputEncoding  = { 0, 0, 0, 0 };
                const TensorShape& inputShape = m_MceInputTensorShape;
                TensorShape mceInputStripe    = CreateStripe(m_MceInputTensorShape, mceInputEncoding, channelRounding);

                TensorShape mceOutputEncoding = TensorShape{ 0, 0, 0, mceOfmStripeDepth };
                TensorShape mceOutputStripe   = CreateStripe(mceOutputShape, mceOutputEncoding, baseMceOfm);

                // PLE stripe is the full tensor, as it accumulates the full output depth
                TensorShape pleInputStripe  = CreateStripe(mceOutputShape, { 0, 0, 0, 0 }, baseMceOfm);
                TensorShape pleOutputStripe = CreateStripe(m_PleOutputTensorShape, { 0, 0, 0, 0 }, baseMceOfm);

                TensorShape memoryOutputEncoding = { 0, 0, 0, 0 };
                TensorShape memoryOutputStripe   = CreateStripe(outputShape, memoryOutputEncoding, baseMceOfm);
                AddStripeInfos(mceInputStripe, mceOutputStripe, pleInputStripe, pleOutputStripe, mceInputStripe,
                               memoryOutputStripe, mceOutputStripe, inputShape, outputShape);
            }
        }
    }

    // Don't split at all.
    // This is needed if all of the stripes above are larger than the tensor
    // and none of them are added.
    if (stripeConfig.splits.none)
    {
        TensorShape mceInputEncoding  = { 0, 0, 0, 0 };
        TensorShape mceInputStripe    = CreateStripe(m_MceInputTensorShape, mceInputEncoding, channelRounding);
        const TensorShape& inputShape = m_MceInputTensorShape;

        TensorShape mceOutputEncoding = mceInputEncoding * m_MceShapeMultiplier;
        TensorShape mceOutputStripe   = CreateStripe(mceOutputShape, mceOutputEncoding, channelRounding);

        TensorShape pleInputStripe = mceOutputStripe;

        TensorShape pleOutputEncoding = mceOutputEncoding * m_PleShapeMultiplier;
        TensorShape pleOutputStripe   = CreateStripe(m_PleOutputTensorShape, pleOutputEncoding, channelRounding);

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
    if (m_BlockConfig.m_Width < rhs.m_BlockConfig.m_Width)
        return true;
    if (rhs.m_BlockConfig.m_Width < m_BlockConfig.m_Width)
        return false;
    if (m_BlockConfig.m_Height < rhs.m_BlockConfig.m_Height)
        return true;
    if (rhs.m_BlockConfig.m_Height < m_BlockConfig.m_Height)
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
    if (m_BlockConfig.m_Width < rhs.m_BlockConfig.m_Width)
        return true;
    if (rhs.m_BlockConfig.m_Width < m_BlockConfig.m_Width)
        return false;
    if (m_BlockConfig.m_Height < rhs.m_BlockConfig.m_Height)
        return true;
    if (rhs.m_BlockConfig.m_Height < m_BlockConfig.m_Height)
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

Buffer* AddPleInputSramBuffer(OwnedOpGraph& opGraph,
                              NumStripesType numPleInputMemoryStripes,
                              const TensorShape& tensorShape,
                              const TensorShape& pleInputMemoryShape,
                              const QuantizationInfo& quantInfo,
                              DataType dataType)
{
    std::unique_ptr<PleInputSramBuffer> buffer =
        PleInputSramBufferBuilder()
            .AddFormat(BufferFormat::NHWCB)
            .AddDataType(dataType)
            .AddTensorShape(tensorShape)
            .AddQuantization(quantInfo)
            .AddStripeShape(pleInputMemoryShape)
            .AddNumStripes(numPleInputMemoryStripes)
            .AddSizeInBytes(utils::CalculateBufferSize(pleInputMemoryShape, BufferFormat::NHWCB));

    PleInputSramBuffer* bufferRaw = opGraph.AddBuffer(std::move(buffer));

    return bufferRaw;
}

std::pair<SramBuffer*, Op*> AddPleToOpGraph(OwnedOpGraph& opGraph,
                                            const TensorShape& memoryOutputShape,
                                            impl::NumMemoryStripes& numMemoryStripes,
                                            std::unique_ptr<Op> pleOp,
                                            const TensorShape& outputShape,
                                            const QuantizationInfo& outputQuantInfo,
                                            DataType outputDataType,
                                            const std::set<uint32_t>& sourceOperationIds)
{
    Op* op             = opGraph.AddOp(std::move(pleOp));
    op->m_OperationIds = sourceOperationIds;

    // Note that we don't need to account for FCAF here, because this SRAM buffer will never be decompressed
    // from FCAF. It may be compressed _into_ FCAF, but that's fine and doesn't require any special consideration.
    std::unique_ptr<SramBuffer> pleOutBuffer = SramBufferBuilder()
                                                   .AddFormat(GetFormat(Location::Sram))
                                                   .AddDataType(outputDataType)
                                                   .AddTensorShape(outputShape)
                                                   .AddQuantization(outputQuantInfo)
                                                   .AddStripeShape(memoryOutputShape)
                                                   .AddNumStripes(numMemoryStripes.m_Output)
                                                   .AddSlotSize(utils::TotalSizeBytesNHWCB(memoryOutputShape))
                                                   .AddTraversalOrder(TraversalOrder::Xyz);

    auto pleOutBufferRaw = opGraph.AddBuffer(std::move(pleOutBuffer));
    opGraph.SetProducer(pleOutBufferRaw, op);

    return { pleOutBufferRaw, op };
};

}    // namespace impl
}    // namespace support_library
}    // namespace ethosn
