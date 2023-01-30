//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "Strategies.hpp"

#include "../include/ethosn_support_library/Support.hpp"
#include "Compiler.hpp"
#include "McePlePass.hpp"
#include "Pass.hpp"
#include "StrategiesCommon.hpp"
#include "Utils.hpp"
#include "cascading/EstimationUtils.hpp"

#include <stdexcept>

namespace ethosn
{
namespace support_library
{

using namespace utils;

namespace
{

std::vector<command_stream::BlockConfig>
    SortBlockConfigsBasedOnShapeRemainder(const std::vector<command_stream::BlockConfig>& blockConfigs,
                                          const TensorShape& outputShape,
                                          const TensorShape& weightsShape)
{
    // Sort block configs so that the most efficient ones will be first
    std::vector<command_stream::BlockConfig> result = blockConfigs;

    const auto comp = [&](const command_stream::BlockConfig& blockConfig1,
                          const command_stream::BlockConfig& blockConfig2) {
        const uint32_t blockWidth1  = blockConfig1.m_BlockWidth();
        const uint32_t blockHeight1 = blockConfig1.m_BlockHeight();

        const uint32_t blockWidth2  = blockConfig2.m_BlockWidth();
        const uint32_t blockHeight2 = blockConfig2.m_BlockHeight();

        const bool outputFitsInBlock1 = (outputShape[1] <= blockHeight1) && (outputShape[2] <= blockWidth1);
        const bool outputFitsInBlock2 = (outputShape[1] <= blockHeight2) && (outputShape[2] <= blockWidth2);

        if (outputFitsInBlock1 && outputFitsInBlock2)
        {
            const uint32_t size1 = blockWidth1 * blockHeight1;
            const uint32_t size2 = blockWidth2 * blockHeight2;

            return size1 < size2;
        }
        else if (!outputFitsInBlock1 && !outputFitsInBlock2)
        {
            // We want to maximise the size of the partial blocks at the edge of the ofm XY planes.
            // We maximise the sum of the remainder of the ofm shape divided by the block size.
            //
            // Example on a 17x17 ofm shape:
            //   16x16 blocks: score = 17%16 + 17%16 = 2
            //   32x8  blocks: score = 17%32 + 17%8 = 18.

            const uint32_t remHeight1 = outputShape[1] % blockConfig1.m_BlockHeight();
            const uint32_t remWidth1  = outputShape[2] % blockConfig1.m_BlockWidth();

            const uint32_t remHeight2 = outputShape[1] % blockConfig2.m_BlockHeight();
            const uint32_t remWidth2  = outputShape[2] % blockConfig2.m_BlockWidth();

            const uint32_t rem1 = remHeight1 + remWidth1;
            const uint32_t rem2 = remHeight2 + remWidth2;

            if (rem1 == rem2)
            {
                // In case of a tie, we favor largest block width if (weightsWidth > weightsHeight)
                // or largest block height otherwise
                const uint32_t weightsWidth  = weightsShape[1];
                const uint32_t weightsHeight = weightsShape[0];

                if (weightsWidth > weightsHeight)
                {
                    return (blockWidth1 > blockWidth2) ||
                           ((blockWidth1 == blockWidth2) && (blockHeight1 > blockHeight2));
                }

                return (blockHeight1 > blockHeight2) || ((blockHeight1 == blockHeight2) && (blockWidth1 > blockWidth2));
            }

            return rem1 > rem2;
        }
        else
        {
            return outputFitsInBlock1;    // && !outputFitsBlock2
        }
    };

    std::stable_sort(result.begin(), result.end(), comp);
    return result;
}

}    // namespace

MceStrategySelectionReturnValue IStrategyDefaultBlockSelection::TrySetupAnyBlockConfig(
    const MceStrategySelectionParameters& strategySelectionParameters,
    const std::vector<command_stream::BlockConfig>& allowedBlockConfigs)
{
    // Sort block configs so that the most efficient ones will be first
    std::vector<command_stream::BlockConfig> sortedBlockConfigs = SortBlockConfigsBasedOnShapeRemainder(
        allowedBlockConfigs, strategySelectionParameters.outputShape, strategySelectionParameters.weightsShape);

    // Try each config in turn, and choose the first that works
    MceStrategySelectionReturnValue rv;
    rv.success = false;

    for (const auto& curBlockConfig : sortedBlockConfigs)
    {
        rv = TrySetup(strategySelectionParameters, curBlockConfig);
        if (rv.success)
        {
            rv.strategyConfig.blockWidth  = curBlockConfig.m_BlockWidth();
            rv.strategyConfig.blockHeight = curBlockConfig.m_BlockHeight();
            return rv;
        }
    }

    return rv;
}

namespace
{

// We limit the number of buffers in a tile to 3 because using 4 buffers in the tile on VGG16
// on the 1 MB SRAM configuration causes a performance regression.
// We need to further investigate this trade-off.
constexpr uint32_t g_DefaultMaxNumInputBuffersInTile  = 3;
constexpr uint32_t g_DefaultMaxNumWeightBuffersInTile = 2;

struct TryStripeShapesResult
{
    bool m_Success                       = false;
    StrategyConfig m_StrategyConfig      = {};
    InputStats m_InputStats              = {};
    SramAllocator m_UpdatedSramAllocator = {};
};

// Given a requested shape for the output stripe (which is not required to be rounded at all),
// calculates what the actual stripe sizes would be (accounting for hardware and firmware constraints)
// and what the tile sizes would be (accounting for double-buffering etc.) and checks if all this would
// fit into SRAM.
// By keeping all the logic of the confusing rounding in this one function it lets the per-Strategy functions
// be nice and simple and concentrate just on looping over possible stripe sizes.
TryStripeShapesResult TryStripeShapes(const MceStrategySelectionParameters& strategySelectionParameters,
                                      const TensorShape& requestedOutputStripe,
                                      const uint32_t maxNumWeightBuffersInTile = g_DefaultMaxNumWeightBuffersInTile,
                                      const uint32_t maxNumInputBuffersInTile  = g_DefaultMaxNumInputBuffersInTile)
{

    const HardwareCapabilities& capabilities = strategySelectionParameters.capabilities;
    const uint32_t patchWidth                = g_PatchShape[2];
    const uint32_t brickGroupHeight          = g_BrickGroupShape[1];
    const uint32_t brickGroupWidth           = g_BrickGroupShape[2];
    const uint32_t brickGroupChannels        = g_BrickGroupShape[3];

    // Sanity check to ensure the output shape width and height are not zero.
    const TensorShape& outputShape = strategySelectionParameters.outputShape;
    assert(outputShape[1] != 0);
    assert(outputShape[2] != 0);

    const ShapeMultiplier& mceShapeMultiplier     = strategySelectionParameters.mceShapeMultiplier;
    const ShapeMultiplier& pleShapeMultiplier     = strategySelectionParameters.pleShapeMultiplier;
    const utils::ShapeMultiplier& shapeMultiplier = mceShapeMultiplier * pleShapeMultiplier;

    // Round the requested output stripe shape to appropriate boundaries
    // Width and height must be a multiple of the brick group size in order to be DMA-able.
    // Additionally, if the input stripes are to be smaller than the input stripe then we must make sure the
    // input stripe sizes are also valid.
    const uint32_t outputStripeWidthMultiple = std::max(brickGroupWidth, brickGroupWidth * shapeMultiplier.m_W);
    const uint32_t outputStripeWidthMax      = RoundUpToNearestMultiple(outputShape[2], brickGroupWidth);
    const uint32_t outputStripeWidth =
        requestedOutputStripe[2] == patchWidth
            ? patchWidth    // Special case, originally supported only in strategy 4.
            : std::min(RoundUpToNearestMultiple(requestedOutputStripe[2], outputStripeWidthMultiple),
                       outputStripeWidthMax);

    const uint32_t outputStripeHeightMultiple = std::max(brickGroupHeight, brickGroupHeight * shapeMultiplier.m_H);
    const uint32_t outputStripeHeightMax      = RoundUpToNearestMultiple(outputShape[1], brickGroupHeight);
    const uint32_t outputStripeHeight =
        std::min(RoundUpToNearestMultiple(requestedOutputStripe[1], outputStripeHeightMultiple), outputStripeHeightMax);
    // The stripe depth must be a multiple of the number of the number of srams as this is required by the firmware and
    // PLE supports, although I think this limitation could be lifted in the future.
    // The stripe depth must also be such that no stripes may start on channels that aren't a multiple of 16 and pass
    // through into the next 16, which is not supported by the DMA (e.g. a stripe starting on channel 24
    // and going to channel 48).
    // Assert to ensure that rounding to a multiple of brickGroupChannels is ALSO a multiple of num SRAMS
    uint32_t outputStripeChannels =
        (DivRoundUp(outputShape[3], requestedOutputStripe[3]) > 1 &&
         requestedOutputStripe[3] > brickGroupChannels * shapeMultiplier.m_C)
            ? RoundUpToNearestMultiple(requestedOutputStripe[3], brickGroupChannels * shapeMultiplier.m_C)
            : RoundUpToNearestMultiple(requestedOutputStripe[3], capabilities.GetNumberOfSrams() * shapeMultiplier.m_C);

    const TensorShape inputShape = strategySelectionParameters.inputShape;
    const uint32_t inputStripeHeightPre =
        AccountForFullDimension(outputShape[1], inputShape[1], outputStripeHeight, shapeMultiplier.m_H);
    const uint32_t inputStripeHeight =
        RoundUpToNearestMultiple(std::min(inputStripeHeightPre, inputShape[1]), brickGroupHeight);

    const uint32_t inputStripeWidthPre =
        AccountForFullDimension(outputShape[2], inputShape[2], outputStripeWidth, shapeMultiplier.m_W);
    const uint32_t inputStripeWidth =
        RoundUpToNearestMultiple(std::min(inputStripeWidthPre, inputShape[2]), brickGroupWidth);

    const TensorShape& weightsShape = strategySelectionParameters.weightsShape;

    // Account for the boundary slots if required by the strategy and the kernel size. It uses the normal
    // slot triple buffering in the width dimension if needed.
    uint32_t usedBoundarySlotsHeight;

    if (inputShape[1] > inputStripeHeight && inputShape[2] > inputStripeWidth && weightsShape[0] > 1)
    {
        usedBoundarySlotsHeight = capabilities.GetBoundaryStripeHeight();
    }
    else
    {
        usedBoundarySlotsHeight = 0;
    }

    // Ensure that the input is large enough for the filter
    if (inputShape[1] > inputStripeHeight)    // streaming in Y
    {
        if (usedBoundarySlotsHeight != 0)
        {
            if ((2 * usedBoundarySlotsHeight) < (weightsShape[0] - 1))
            {
                // Without this restriction, wrong stripe height would be selected resulting in output being produced without doing a full convolution.
                return {};
            }
        }
        else
        {
            if ((2 * inputStripeHeight) < (weightsShape[0] - 1))
            {
                // Without this restriction, wrong stripe height would be selected resulting in output being produced without doing a full convolution.
                return {};
            }
        }
    }
    if (inputShape[2] > inputStripeWidth)    // streaming in X
    {
        if ((2 * inputStripeWidth) < (weightsShape[1] - 1))
        {
            // Without this restriction, wrong stripe width would be selected resulting in output being produced without doing a full convolution.
            return {};
        }
    }

    // Output stripe depth maximum is set for MAXPOOLING_3x3/(2,2)
    // so that the PLE can manage spilling if the number of stripes is more than 1.
    if (utils::DivRoundUp(inputShape[1], inputStripeHeight) > 1)
    {
        const uint32_t depthMax = strategySelectionParameters.depthMax;
        outputStripeChannels    = std::min(outputStripeChannels, depthMax);
    }

    const TensorShape outputStripe = { 1, outputStripeHeight, outputStripeWidth, outputStripeChannels };

    // Calculate input stripe from output stripe
    TensorShape inputStripe = { 1, inputStripeHeight, inputStripeWidth,
                                RoundUpToNearestMultiple(inputShape[3], capabilities.GetNumberOfSrams()) };

    // Calculate weight stripe from output stripe.
    TensorShape weightStripe;
    const DataFormat& weightsFormat                       = strategySelectionParameters.weightsFormat;
    const std::pair<bool, uint32_t>& inputStaticAndOffset = strategySelectionParameters.inputStaticAndOffset;
    if (weightsFormat == DataFormat::HWIO)
    {
        weightStripe = { weightsShape[0], weightsShape[1], inputShape[3], outputStripe[3] / shapeMultiplier.m_C };
    }
    else if (weightsFormat == DataFormat::HWIM)
    {
        uint32_t strideSize =
            utils::DivRoundUp(utils::RoundUpToNearestMultiple(inputShape[3], capabilities.GetNumberOfSrams()),
                              utils::RoundUpToNearestMultiple(weightsShape[2], capabilities.GetNumberOfSrams()));
        weightStripe = { weightsShape[0], weightsShape[1], outputStripe[3] / shapeMultiplier.m_C * strideSize,
                         weightsShape[3] };

        // Legacy code doesn't support splitting in width in this case.
        // Also this is not required when the whole input is already in Sram.
        if (!inputStaticAndOffset.first && GetWidth(inputStripe) >= GetWidth(inputShape))
        {
            inputStripe[3] = weightStripe[2];
        }
    }
    else
    {
        // Weight tensor must be HWIO or HWIM
        assert(false);
        throw std::runtime_error("Invalid weight data format");
    }

    // Work out the tile sizes by deciding how many stripes we want in each tile

    // Max number of stripes for the given input shape
    // Ifm. 1x1 kernel needs 1 stripe loaded to calculate the output, 2x2 needs 2 stripes (current + 1 above/below),
    // 3x3 and larger needs 3 (current + 1 above + 1 below). Add one for double buffering. The same applies
    // when streaming in the width direction and using boundary slots for height direction if necessary.
    const uint32_t weightsSize              = inputShape[2] > inputStripe[2] ? weightsShape[1] : weightsShape[0];
    const uint32_t maxNumInputStripesInTile = std::min(std::min(weightsSize, 3u) + 1, maxNumInputBuffersInTile);
    // Clamp this to the maximum number of stripes possible (i.e. if the image is small enough don't bother allocating
    // more space than we could use).
    const uint32_t numInputStripesTotalX = DivRoundUp(inputShape[2], inputStripe[2]);
    const uint32_t numInputStripesTotalY = DivRoundUp(inputShape[1], inputStripe[1]);
    const uint32_t numInputStripesTotal  = numInputStripesTotalY * numInputStripesTotalX;
    // If the input is already in SRAM then we must have all stripes of the image in the tile, regardless of how many.
    const uint32_t numInputStripesInTile =
        inputStaticAndOffset.first ? numInputStripesTotal : std::min(maxNumInputStripesInTile, numInputStripesTotal);
    // Check that the number of slots in the tile can be represented in HW
    if (numInputStripesInTile > capabilities.GetNumCentralSlots())
    {
        return {};
    }

    // Clamp the overall tile size to the size of the full tensor. This means that if we have a small number of stripes
    // and the last one is partial we don't waste space in the tile that will never be used.
    uint32_t inputTileMax =
        TotalSizeBytes(TensorShape{ 1, RoundUpToNearestMultiple(inputShape[1], brickGroupHeight),
                                    RoundUpToNearestMultiple(inputShape[2], brickGroupWidth),
                                    RoundUpToNearestMultiple(inputShape[3], capabilities.GetNumberOfSrams()) });

    if (inputShape[1] > inputStripe[1] && inputShape[2] > inputStripe[2])
    {
        // In case the input tensor is split in both x and y (strategy 6), the size of input tile max
        // will take into account of (partial width, full height) and (full width, partial height).
        inputTileMax = std::max(
            inputTileMax,
            TotalSizeBytes(TensorShape{ 1, RoundUpToNearestMultiple(inputShape[1], inputStripe[1]),
                                        RoundUpToNearestMultiple(inputShape[2], brickGroupWidth),
                                        RoundUpToNearestMultiple(inputShape[3], capabilities.GetNumberOfSrams()) }));

        inputTileMax = std::max(
            inputTileMax,
            TotalSizeBytes(TensorShape{ 1, RoundUpToNearestMultiple(inputShape[1], brickGroupHeight),
                                        RoundUpToNearestMultiple(inputShape[2], inputStripe[2]),
                                        RoundUpToNearestMultiple(inputShape[3], capabilities.GetNumberOfSrams()) }));
    }

    // Account for the boundary slots if required by the strategy and the kernel size. It uses the normal
    // slot triple buffering in the width dimension if needed.
    const uint32_t boundarySlotsSize =
        capabilities.GetNumBoundarySlots() * usedBoundarySlotsHeight * inputStripe[2] * inputStripe[3];
    const uint32_t inputTile =
        std::min(TotalSizeBytes(inputStripe) * numInputStripesInTile, inputTileMax) + boundarySlotsSize;

    // Clamp this to the maximum number of stripes possible (i.e. if the image is small enough don't bother allocating
    // more space than we could use).
    const uint32_t numWeightStripesTotal  = DivRoundUp(outputShape[3], outputStripe[3]);
    const uint32_t numWeightStripesInTile = std::min(maxNumWeightBuffersInTile, numWeightStripesTotal);
    const uint32_t weightTile =
        (TotalSizeBytes(weightStripe) == 0U)
            ? 0U
            : EstimateWeightSizeBytes(weightStripe, capabilities, weightsFormat == DataFormat::HWIM) *
                  numWeightStripesInTile;

    // Outputs. We need at most 2 at a time for double-buffering.
    const uint32_t maxNumOutputStripesInTile = 2;
    // Clamp this to the maximum number of stripes possible (i.e. if the image is small enough don't bother allocating
    // more space than we could use).
    const uint32_t numOutputStripesX      = DivRoundUp(outputShape[2], outputStripe[2]);
    const uint32_t numOutputStripesY      = DivRoundUp(outputShape[1], outputStripe[1]);
    const uint32_t numOutputStripesZ      = DivRoundUp(outputShape[3], outputStripe[3]);
    const uint32_t numOutputStripesTotal  = numOutputStripesX * numOutputStripesY * numOutputStripesZ;
    const uint32_t numOutputStripesInTile = std::min(maxNumOutputStripesInTile, numOutputStripesTotal);
    // Clamp the overall tile size to the size of the full tensor. This means that if we have a small number of stripes
    // and the last one is partial we don't waste space in the tile that will never be used.
    const uint32_t outputTileMax =
        TotalSizeBytes(TensorShape{ 1, RoundUpToNearestMultiple(outputShape[1], brickGroupHeight),
                                    RoundUpToNearestMultiple(outputShape[2], brickGroupWidth),
                                    RoundUpToNearestMultiple(outputShape[3], capabilities.GetNumberOfSrams()) });
    // For the special case of a 4-wide stripe, the tile must be rounded up to a brick group otherwise the DMA
    // will try to access outside of the tile. This may only be an issue in the model though.
    const uint32_t outputTileMin = TotalSizeBytes(RoundUpHeightAndWidthToBrickGroup(outputStripe));
    const uint32_t outputTile =
        std::max(std::min(TotalSizeBytes(outputStripe) * numOutputStripesInTile, outputTileMax), outputTileMin);

    if ((numInputStripesTotalX != numOutputStripesX && numOutputStripesY > 1) ||
        numInputStripesTotalY < numOutputStripesY)
    {
        // This is a limitation of the current StripeStreamer code in the firmware.
        // Note that there is only very limited support for the case where there are
        // more input stripes than output stripes, but it isn't clear what those
        // limitations are so this check is probably overly permissive for those cases.
        return {};
    }

    SramAllocator currentSramAllocator = strategySelectionParameters.sramAllocator;
    SramAllocator::UserId userId       = strategySelectionParameters.userId;
    AllocationResult allocationResults =
        FitsInSram(userId, currentSramAllocator, capabilities, inputTile, weightTile, outputTile, inputStaticAndOffset);
    if (!allocationResults.m_Success)
    {
        return {};
    }
    TryStripeShapesResult result;
    result.m_Success                                           = true;
    result.m_StrategyConfig.inputAllocation.stripeShape        = inputStripe;
    result.m_StrategyConfig.inputAllocation.tileSize           = inputTile;
    result.m_StrategyConfig.inputAllocation.numStripesInTile   = numInputStripesInTile;
    result.m_StrategyConfig.outputAllocation.stripeShape       = outputStripe;
    result.m_StrategyConfig.outputAllocation.tileSize          = outputTile;
    result.m_StrategyConfig.outputAllocation.numStripesInTile  = numOutputStripesInTile;
    result.m_StrategyConfig.weightsAllocation.stripeShape      = weightStripe;
    result.m_StrategyConfig.weightsAllocation.tileSize         = weightTile;
    result.m_StrategyConfig.weightsAllocation.numStripesInTile = numWeightStripesInTile;
    result.m_UpdatedSramAllocator                              = currentSramAllocator;
    FillStrategyConfigOffsets(allocationResults, result.m_StrategyConfig);

    result.m_InputStats = GetInputStatsLegacy(
        capabilities, inputShape, inputStripe, inputStaticAndOffset.first ? Location::Sram : Location::Dram, inputTile,
        TensorInfo(weightsShape, DataType::UINT8_QUANTIZED, weightsFormat), numOutputStripesZ);

    return result;
}

}    // namespace

using namespace utils;

MceStrategySelectionReturnValue Strategy0::TrySetup(const MceStrategySelectionParameters& strategySelectionParameters,
                                                    const ethosn::command_stream::BlockConfig& blockConfig)
{
    MceStrategySelectionReturnValue rv{};
    rv.success                     = false;
    StrategyConfig& strategyConfig = rv.strategyConfig;
    SramAllocator& sramAllocator   = rv.sramAllocator;

    // Calculate the range of stripe sizes we want to try. We want to make the MCE output stripe size a multiple of
    // the block size for performance reasons (partial blocks give poor PLE utilisation).
    // Try splitting into two stripes at first.
    const TensorShape& mceOutputShape                = strategySelectionParameters.mceOutputShape;
    const utils::ShapeMultiplier& pleShapeMultiplier = strategySelectionParameters.pleShapeMultiplier;
    const uint32_t maxMceOutputStripeHeight =
        RoundUpToNearestMultiple(mceOutputShape[1] / 2, blockConfig.m_BlockHeight());
    if (maxMceOutputStripeHeight >= mceOutputShape[1])
    {
        return rv;    // Can't use strategy 0, as the height is too small to split at all
    }
    // Decrease iteratively by one block at a time
    const uint32_t stepMceOutputStripeHeight = blockConfig.m_BlockHeight();
    // Stop when the stripe is a single block
    const uint32_t minMceOutputStripeHeight = blockConfig.m_BlockHeight();

    // TryStripeShapes is driven by the *output* stripe size rather than *mce output* stripe size, so convert.
    const uint32_t maxOutputStripeHeight  = maxMceOutputStripeHeight * pleShapeMultiplier.m_H;
    const uint32_t stepOutputStripeHeight = stepMceOutputStripeHeight * pleShapeMultiplier.m_H;
    const uint32_t minOutputStripeHeight  = minMceOutputStripeHeight * pleShapeMultiplier.m_H;

    // Generate a list of parameters we pass to TryStripeShapes so we can see all the stripe shapes which could be attempted.
    struct Strategy0Params
    {
        uint32_t outputStripeHeight;
        uint32_t numInputBuffers;
    };
    std::vector<Strategy0Params> paramsList;
    for (uint32_t outputStripeHeight = maxOutputStripeHeight; outputStripeHeight >= minOutputStripeHeight;
         outputStripeHeight -= stepOutputStripeHeight)
    {
        // First try a solution with 4 slots in the input tile.
        for (uint32_t numInputBuffers = 4; numInputBuffers >= g_DefaultMaxNumInputBuffersInTile; --numInputBuffers)
        {
            paramsList.push_back({ outputStripeHeight, numInputBuffers });
        }
    }

    const TensorShape& outputShape = strategySelectionParameters.outputShape;
    for (auto params : paramsList)
    {
        const TryStripeShapesResult tryResult = TryStripeShapes(
            strategySelectionParameters, { 1, params.outputStripeHeight, outputShape[2], outputShape[3] },
            g_DefaultMaxNumWeightBuffersInTile, params.numInputBuffers);
        if (tryResult.m_Success)
        {
            strategyConfig          = tryResult.m_StrategyConfig;
            strategyConfig.strategy = Strategy::STRATEGY_0;
            sramAllocator           = tryResult.m_UpdatedSramAllocator;
            rv.success              = true;
            return rv;
        }
    }

    return rv;
}

MceStrategySelectionReturnValue Strategy1::TrySetup(const MceStrategySelectionParameters& strategySelectionParameters,
                                                    const ethosn::command_stream::BlockConfig&)
{
    MceStrategySelectionReturnValue rv;
    rv.success                     = false;
    StrategyConfig& strategyConfig = rv.strategyConfig;
    SramAllocator& sramAllocator   = rv.sramAllocator;

    const TensorShape& outputShape = strategySelectionParameters.outputShape;
    auto TrySolution               = [&](const uint32_t outputStripeChannels, uint32_t numWeightBuffers) {
        const TryStripeShapesResult tryResult = TryStripeShapes(
            strategySelectionParameters, { 1, outputShape[1], outputShape[2], outputStripeChannels }, numWeightBuffers);
        if (tryResult.m_Success)
        {
            strategyConfig          = tryResult.m_StrategyConfig;
            strategyConfig.strategy = Strategy::STRATEGY_1;
            sramAllocator           = tryResult.m_UpdatedSramAllocator;
            return true;
        }
        return false;
    };

    struct Strategy1Params
    {
        uint32_t outputStripeChannels;
        uint32_t numWeightBuffers;
    };
    // Generate a list of parameters we pass to TryStripeShapes so we can see all the stripe shapes which could be attempted.
    std::vector<Strategy1Params> paramsList;
    for (uint32_t numDepthSplits = 2; numDepthSplits < outputShape[3]; ++numDepthSplits)
    {
        // First, try and find a solution with three stripes of weight in the tile.
        for (uint32_t numWeightBuffers = 3; numWeightBuffers >= g_DefaultMaxNumWeightBuffersInTile; --numWeightBuffers)
        {
            const uint32_t outputStripeChannels = outputShape[3] / numDepthSplits;
            paramsList.push_back({ outputStripeChannels, numWeightBuffers });
        }
    }
    // Attempt single buffering the weight stripes as a last resort for strategy 1
    for (uint32_t numDepthSplits = 2; numDepthSplits < outputShape[3]; ++numDepthSplits)
    {
        const uint32_t outputStripeChannels = outputShape[3] / numDepthSplits;
        paramsList.push_back({ outputStripeChannels, 1u });
    }
    for (auto params : paramsList)
    {

        if (TrySolution(params.outputStripeChannels, params.numWeightBuffers))
        {
            rv.success = true;
            return rv;
        }
    }
    return rv;
}

MceStrategySelectionReturnValue Strategy3::TrySetup(const MceStrategySelectionParameters& strategySelectionParameters,
                                                    const ethosn::command_stream::BlockConfig&)
{
    MceStrategySelectionReturnValue rv;
    rv.success                     = false;
    StrategyConfig& strategyConfig = rv.strategyConfig;
    SramAllocator& sramAllocator   = rv.sramAllocator;

    const TensorShape& outputShape        = strategySelectionParameters.outputShape;
    const TryStripeShapesResult tryResult = TryStripeShapes(strategySelectionParameters, outputShape);
    if (tryResult.m_Success)
    {
        strategyConfig          = tryResult.m_StrategyConfig;
        strategyConfig.strategy = Strategy::STRATEGY_3;
        sramAllocator           = tryResult.m_UpdatedSramAllocator;
        rv.success              = true;
        return rv;
    }

    return rv;
}

MceStrategySelectionReturnValue
    Strategy4::TrySetupAnyBlockConfig(const MceStrategySelectionParameters& strategySelectionParameters,
                                      const std::vector<command_stream::BlockConfig>& allowedBlockConfigs)
{
    MceStrategySelectionReturnValue rv;
    rv.success                     = false;
    StrategyConfig& strategyConfig = rv.strategyConfig;
    SramAllocator& sramAllocator   = rv.sramAllocator;

    // Force strategy 4 to use the minimum number of stripe depths
    const TensorShape& outputShape            = strategySelectionParameters.outputShape;
    const HardwareCapabilities& capabilities  = strategySelectionParameters.capabilities;
    const ShapeMultiplier& mceShapeMultiplier = strategySelectionParameters.mceShapeMultiplier;
    const ShapeMultiplier& pleShapeMultiplier = strategySelectionParameters.pleShapeMultiplier;
    const uint32_t ofmRegion                  = std::min(outputShape[3], capabilities.GetNumberOfOgs());
    const uint32_t stripeDepth                = RoundUpToNearestMultiple(ofmRegion, capabilities.GetNumberOfSrams());
    const uint32_t outStripeDepth             = stripeDepth * mceShapeMultiplier.m_C * pleShapeMultiplier.m_C;

    const uint32_t inputStripeWidth     = g_BrickGroupShape[2];
    const uint32_t mceOutputStripeWidth = inputStripeWidth * mceShapeMultiplier.m_W;
    const uint32_t outputStripeWidth    = mceOutputStripeWidth * pleShapeMultiplier.m_W;

    // Sort block configs first based on the common metric
    const TensorShape& weightsShape = strategySelectionParameters.weightsShape;
    std::vector<command_stream::BlockConfig> sortedBlockConfigs =
        SortBlockConfigsBasedOnShapeRemainder(allowedBlockConfigs, outputShape, weightsShape);

    // Then sort again (with higher priority) to favour those with a width matching our stripe width,
    // to avoid partial blocks (partial blocks give poor PLE utilisation).
    std::stable_sort(sortedBlockConfigs.begin(), sortedBlockConfigs.end(),
                     [&](const command_stream::BlockConfig& a, const command_stream::BlockConfig& b) {
                         const uint32_t scoreA = (a.m_BlockWidth() == mceOutputStripeWidth ? 1 : 0);
                         const uint32_t scoreB = (b.m_BlockWidth() == mceOutputStripeWidth ? 1 : 0);
                         return scoreA > scoreB;    // Higher scores should be placed earlier in the sorted list
                     });

    for (const command_stream::BlockConfig& blockConfig : sortedBlockConfigs)
    {
        // First try double-buffering the weight stripes (i.e. tile = 2 x stripe) but if
        // this does not fit then single-buffering will have to do.
        for (uint32_t numStripesInWeightTile = 2; numStripesInWeightTile >= 1; --numStripesInWeightTile)
        {
            const TryStripeShapesResult tryResult =
                TryStripeShapes(strategySelectionParameters, { 1, outputShape[1], outputStripeWidth, outStripeDepth },
                                numStripesInWeightTile);
            if (tryResult.m_Success)
            {
                strategyConfig             = tryResult.m_StrategyConfig;
                strategyConfig.strategy    = Strategy::STRATEGY_4;
                strategyConfig.blockWidth  = blockConfig.m_BlockWidth();
                strategyConfig.blockHeight = blockConfig.m_BlockHeight();
                sramAllocator              = tryResult.m_UpdatedSramAllocator;
                rv.success                 = true;
                return rv;
            }
        }
    }

    return rv;
}

MceStrategySelectionReturnValue
    Strategy6::TrySetupAnyBlockConfig(const MceStrategySelectionParameters& strategySelectionParameters,
                                      const std::vector<command_stream::BlockConfig>& allowedBlockConfigs)
{
    MceStrategySelectionReturnValue rv;
    rv.success                     = false;
    StrategyConfig& strategyConfig = rv.strategyConfig;
    SramAllocator& sramAllocator   = rv.sramAllocator;

    const std::pair<bool, uint32_t>& inputStaticAndOffset = strategySelectionParameters.inputStaticAndOffset;
    if (inputStaticAndOffset.first)
    {
        return rv;
    }

    // Sort block configs based on the common metric
    const TensorShape& outputShape  = strategySelectionParameters.outputShape;
    const TensorShape& weightsShape = strategySelectionParameters.weightsShape;
    std::vector<command_stream::BlockConfig> sortedBlockConfigs =
        SortBlockConfigsBasedOnShapeRemainder(allowedBlockConfigs, outputShape, weightsShape);

    // Generate a list of parameters we pass to TryStripeShapes so we can see all the stripe shapes which could be attempted.
    struct Strategy6Params
    {
        uint32_t outputStripeHeight;
        uint32_t outputStripeWidth;
        uint32_t outputStripeChannel;
        uint32_t blockWidth;
        uint32_t blockHeight;
    };
    std::vector<Strategy6Params> paramsList;

    // Consider all combinations of variables, in an order which we think will give the best performance first.
    // Even though we use a cost metric further down, this doesn't account for all aspects of performance and so the
    // order here does still matter.
    const TensorShape& mceOutputShape         = strategySelectionParameters.mceOutputShape;
    const ShapeMultiplier& pleShapeMultiplier = strategySelectionParameters.pleShapeMultiplier;
    for (uint32_t numChannelSplits = 1; numChannelSplits < outputShape[3]; ++numChannelSplits)
    {
        for (command_stream::BlockConfig blockConfig : sortedBlockConfigs)
        {
            // Calculate the range of stripe sizes we want to try. We want to make the MCE output stripe size a multiple of
            // the block size for performance reasons (partial blocks give poor PLE utilisation).
            // Try splitting into two stripes (for width and height) at first.
            const uint32_t maxMceOutputStripeHeight =
                RoundUpToNearestMultiple(mceOutputShape[1] / 2, blockConfig.m_BlockHeight());
            const uint32_t maxMceOutputStripeWidth =
                RoundUpToNearestMultiple(mceOutputShape[2] / 2, blockConfig.m_BlockWidth());
            if (maxMceOutputStripeHeight >= mceOutputShape[1] || maxMceOutputStripeWidth > mceOutputShape[2])
            {
                continue;    // Can't use strategy 6, as the width/height is too small to split at all
            }
            // Decrease iteratively by one block at a time
            const uint32_t stepMceOutputStripeHeight = blockConfig.m_BlockHeight();
            const uint32_t stepMceOutputStripeWidth  = blockConfig.m_BlockWidth();
            // Stop when the stripe is a single block
            const uint32_t minMceOutputStripeHeight = blockConfig.m_BlockHeight();
            const uint32_t minMceOutputStripeWidth  = blockConfig.m_BlockWidth();

            // TryStripeShapes is driven by the *output* stripe size rather than *mce output* stripe size, so convert.
            const uint32_t maxOutputStripeHeight  = maxMceOutputStripeHeight * pleShapeMultiplier.m_H;
            const uint32_t maxOutputStripeWidth   = maxMceOutputStripeWidth * pleShapeMultiplier.m_W;
            const uint32_t stepOutputStripeHeight = stepMceOutputStripeHeight * pleShapeMultiplier.m_H;
            const uint32_t stepOutputStripeWidth  = stepMceOutputStripeWidth * pleShapeMultiplier.m_W;
            const uint32_t minOutputStripeHeight  = minMceOutputStripeHeight * pleShapeMultiplier.m_H;
            const uint32_t minOutputStripeWidth   = minMceOutputStripeWidth * pleShapeMultiplier.m_W;

            for (uint32_t outputStripeWidth = maxOutputStripeWidth; outputStripeWidth >= minOutputStripeWidth;
                 outputStripeWidth -= stepOutputStripeWidth)
            {
                for (uint32_t outputStripeHeight = maxOutputStripeHeight; outputStripeHeight >= minOutputStripeHeight;
                     outputStripeHeight -= stepOutputStripeHeight)
                {
                    const uint32_t outputStripeChannel = outputShape[3] / numChannelSplits;
                    paramsList.push_back({ outputStripeHeight, outputStripeWidth, outputStripeChannel,
                                           blockConfig.m_BlockWidth(), blockConfig.m_BlockHeight() });
                }
            }
        }
    }

    Optional<std::pair<Strategy6Params, TryStripeShapesResult>> best;
    uint64_t bestCost = std::numeric_limits<uint64_t>::max();
    for (auto params : paramsList)
    {
        const TensorShape outputStripeShape   = { 1, params.outputStripeHeight, params.outputStripeWidth,
                                                params.outputStripeChannel };
        const TryStripeShapesResult tryResult = TryStripeShapes(strategySelectionParameters, outputStripeShape);
        if (tryResult.m_Success)
        {
            const uint64_t ifmBandwidth = tryResult.m_InputStats.m_MemoryStats.m_DramParallel +
                                          tryResult.m_InputStats.m_MemoryStats.m_DramNonParallel;
            const bool isOutputFcafCompatible = (IsCompressionFormatCompatibleWithStripeShapeLegacy(
                                                     CompilerDataCompressedFormat::FCAF_WIDE, outputStripeShape) ||
                                                 IsCompressionFormatCompatibleWithStripeShapeLegacy(
                                                     CompilerDataCompressedFormat::FCAF_DEEP, outputStripeShape));

            // Minimise IFM bandwidth, but also account for FCAF comatibility. FCAF is important not only for
            // bandwidth reduction, but reduces the chances that the firmware will need to do lots of small DMA chunks
            // for each stripe.
            const uint64_t cost = ifmBandwidth / (isOutputFcafCompatible ? 2 : 1);

            // Note that this strict inequality favours params earlier in the list, as we add them in a rough
            // best-first order. The above cost metric does not account for everything.
            if (cost < bestCost)
            {
                best     = std::make_pair(params, tryResult);
                bestCost = cost;
            }
        }
    }

    if (best.has_value())
    {
        strategyConfig             = best.value().second.m_StrategyConfig;
        strategyConfig.strategy    = Strategy::STRATEGY_6;
        strategyConfig.blockWidth  = best.value().first.blockWidth;
        strategyConfig.blockHeight = best.value().first.blockHeight;
        sramAllocator              = best.value().second.m_UpdatedSramAllocator;
        rv.success                 = true;
        return rv;
    }

    return rv;
}

// Scheduling strategy to support input tensor depth streaming
// Limitations:
// (1) Input tensor split in depth and height directions, no split in width.
// (2) only depthwise convolutions supported.
MceStrategySelectionReturnValue Strategy7::TrySetup(const MceStrategySelectionParameters& strategySelectionParameters,
                                                    const ethosn::command_stream::BlockConfig& blockConfig)
{
    MceStrategySelectionReturnValue rv;
    rv.success                     = false;
    StrategyConfig& strategyConfig = rv.strategyConfig;
    SramAllocator& sramAllocator   = rv.sramAllocator;

    const DataFormat weightsFormat = strategySelectionParameters.weightsFormat;
    const bool isHwim              = weightsFormat == DataFormat::HWIM;

    // This applies only to Depthwise convolutions.
    if (!isHwim)
    {
        return rv;
    }

    const std::pair<bool, uint32_t>& inputStaticAndOffset = strategySelectionParameters.inputStaticAndOffset;
    if (inputStaticAndOffset.first)
    {
        return rv;
    }

    const TensorShape& outputShape = strategySelectionParameters.outputShape;
    auto TrySolution               = [&](const uint32_t outputStripeHeight, const uint32_t outputStripeChannels,
                           uint32_t numWeightBuffers) {
        const TryStripeShapesResult tryResult =
            TryStripeShapes(strategySelectionParameters,
                            { 1, outputStripeHeight, outputShape[2], outputStripeChannels }, numWeightBuffers);
        if (tryResult.m_Success)
        {
            strategyConfig          = tryResult.m_StrategyConfig;
            strategyConfig.strategy = Strategy::STRATEGY_7;
            sramAllocator           = tryResult.m_UpdatedSramAllocator;
            return true;
        }
        return false;
    };

    // Calculate the range of stripe sizes we want to try. We want to make the MCE output stripe size a multiple of
    // the block size for performance reasons (partial blocks give poor PLE utilisation).
    // Try splitting into two stripes at first.
    const TensorShape& mceOutputShape = strategySelectionParameters.mceOutputShape;
    const uint32_t maxMceOutputStripeHeight =
        RoundUpToNearestMultiple(mceOutputShape[1] / 2, blockConfig.m_BlockHeight());
    // Decrease iteratively by one block at a time
    const uint32_t stepMceOutputStripeHeight = blockConfig.m_BlockHeight();
    // Stop when the stripe is a single block
    const uint32_t minMceOutputStripeHeight = blockConfig.m_BlockHeight();

    // TryStripeShapes is driven by the *output* stripe size rather than *mce output* stripe size, so convert.
    const ShapeMultiplier& pleShapeMultiplier = strategySelectionParameters.pleShapeMultiplier;
    const uint32_t maxOutputStripeHeight      = maxMceOutputStripeHeight * pleShapeMultiplier.m_H;
    const uint32_t stepOutputStripeHeight     = stepMceOutputStripeHeight * pleShapeMultiplier.m_H;
    const uint32_t minOutputStripeHeight      = minMceOutputStripeHeight * pleShapeMultiplier.m_H;

    struct Strategy7Params
    {
        uint32_t outputStripeHeight;
        uint32_t outputStripeChannels;
        uint32_t numWeightBuffers;
    };
    // Generate a list of parameters we pass to TryStripeShapes so we can see all the stripe shapes which could be attempted.
    std::vector<Strategy7Params> paramsList;
    for (uint32_t outputStripeHeight = maxOutputStripeHeight; outputStripeHeight >= minOutputStripeHeight;
         outputStripeHeight -= stepOutputStripeHeight)
    {
        for (uint32_t numDepthSplits = 2U; numDepthSplits < outputShape[3]; ++numDepthSplits)
        {
            // First, try and find a solution with three stripes of weight in the tile.
            for (uint32_t numWeightBuffers = 3U; numWeightBuffers >= g_DefaultMaxNumWeightBuffersInTile;
                 --numWeightBuffers)
            {
                const uint32_t outputStripeChannels = outputShape[3] / numDepthSplits;
                paramsList.push_back({ outputStripeHeight, outputStripeChannels, numWeightBuffers });
            }
        }
    }
    // Attempt single buffering the weight stripes as a last resort.
    for (uint32_t outputStripeHeight = maxOutputStripeHeight; outputStripeHeight >= minOutputStripeHeight;
         outputStripeHeight -= stepOutputStripeHeight)
    {
        for (uint32_t numDepthSplits = 2U; numDepthSplits < outputShape[3]; ++numDepthSplits)
        {
            const uint32_t outputStripeChannels = outputShape[3] / numDepthSplits;
            paramsList.push_back({ outputStripeHeight, outputStripeChannels, 1U });
        }
    }
    for (auto params : paramsList)
    {

        if (TrySolution(params.outputStripeHeight, params.outputStripeChannels, params.numWeightBuffers))
        {
            rv.success = true;
            return rv;
        }
    }
    return rv;
}

}    // namespace support_library
}    // namespace ethosn
