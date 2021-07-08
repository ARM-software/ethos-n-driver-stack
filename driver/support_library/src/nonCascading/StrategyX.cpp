//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "StrategyX.hpp"

#include "Strategies.hpp"
#include "StrategiesCommon.hpp"
#include "StrategyConfig.hpp"
#include "Utils.hpp"

#include <ethosn_command_stream/CommandData.hpp>

namespace ethosn
{
namespace support_library
{

using namespace utils;

namespace
{

enum class WeightsReloadingOptions
{
    NO_RELOADING,
    RELOADING_DOUBLE_BUFFERING,
    RELOADING_NO_DOUBLE_BUFFERING,
};

bool IsUpsampling(command_stream::UpsampleType upsampleType)
{
    return upsampleType != command_stream::UpsampleType::OFF;
}

bool IsFullyConnected(command_stream::MceOperation mceOperation)
{
    return mceOperation == command_stream::MceOperation::FULLY_CONNECTED;
}

bool IsBlockConfigCompatible(const command_stream::BlockConfig& blockConfig,
                             const HardwareCapabilities& capabilities,
                             command_stream::MceOperation mceOperation,
                             command_stream::UpsampleType upsampleType)
{
    const uint32_t numAccumulatorsPerOg     = capabilities.GetTotalAccumulatorsPerOg();
    const uint32_t currBlockWidth           = blockConfig.m_BlockWidth();
    const uint32_t currBlockHeight          = blockConfig.m_BlockHeight();
    const uint32_t numberOfElementsInABlock = currBlockWidth * currBlockHeight;

    bool isUpsampling     = IsUpsampling(upsampleType);
    bool isFullyConnected = IsFullyConnected(mceOperation);

    if (numberOfElementsInABlock > numAccumulatorsPerOg)
    {
        return false;
    }

    if (isFullyConnected && (currBlockWidth != 8U || currBlockHeight != 8U))
    {
        return false;
    }

    // When using upsampling, we need to have a block size of
    // 16x16 because the input tensor is DMA using the size
    // "BlockSizeW/2 X BlockSizeH/2" and the DMA cannot transfer block
    // smaller than 8x8
    if (isUpsampling && (currBlockWidth != 16U || currBlockHeight != 16U))
    {
        return false;
    }

    return true;
}

// Given a requested shape for the output stripe calculates what the actual stripe sizes would be
// (accounting for hardware and firmware constraints)
// and what the tile sizes would be (accounting for buffering etc.) and checks if all this would
// fit into SRAM.
MceStrategySelectionReturnValue
    TryStripeShapes(const StrategyXSelectionParameters& strategyXSelectionParameters,
                    const TensorShape& requestedOutputStripe,
                    const uint32_t requestedInputChannels,
                    const bool allowInputBuffering                 = false,
                    const bool avoidInputReloading                 = false,
                    const bool activationCompression               = false,
                    const WeightsReloadingOptions weightsReloading = WeightsReloadingOptions::NO_RELOADING)
{
    MceStrategySelectionReturnValue rv;
    rv.success = false;

    const command_stream::MceOperation& mceOperation = strategyXSelectionParameters.mceOperation;
    const DataFormat& weightsFormat                  = strategyXSelectionParameters.weightsFormat;
    const bool isFullyConnected                      = (mceOperation == command_stream::MceOperation::FULLY_CONNECTED);
    const bool isHwio                                = (weightsFormat == DataFormat::HWIO);

    const HardwareCapabilities& capabilities         = strategyXSelectionParameters.capabilities;
    const uint32_t brickGroupHeight                  = GetHeight(capabilities.GetBrickGroupShape());
    const uint32_t brickGroupWidth                   = GetWidth(capabilities.GetBrickGroupShape());
    const uint32_t brickGroupChannels                = GetChannels(capabilities.GetBrickGroupShape());
    const utils::ShapeMultiplier& mceShapeMultiplier = strategyXSelectionParameters.mceShapeMultiplier;
    const utils::ShapeMultiplier& pleShapeMultiplier = strategyXSelectionParameters.pleShapeMultiplier;
    const utils::ShapeMultiplier& shapeMultiplier    = mceShapeMultiplier * pleShapeMultiplier;

    // Allow output stripe width smaller then brickGroupHeight. This is going to be fixed later to make it DMA-able when pooling is supported.
    const uint32_t outputStripeWidthMin = brickGroupWidth * shapeMultiplier.m_W;
    const TensorShape& outputShape      = strategyXSelectionParameters.outputShape;
    const uint32_t outputStripeWidthMax = RoundUpToNearestMultiple(GetWidth(outputShape), brickGroupWidth);
    uint32_t outputStripeWidth =
        std::min(RoundUpToNearestMultiple(GetWidth(requestedOutputStripe), outputStripeWidthMin), outputStripeWidthMax);

    // Allow output stripe height smaller then brickGroupHeight. This is going to be fixed later to make it DMA-able when pooling is supported.
    const uint32_t outputStripeHeightMin = brickGroupHeight * shapeMultiplier.m_H;
    const uint32_t outputStripeHeightMax = RoundUpToNearestMultiple(GetHeight(outputShape), brickGroupHeight);
    uint32_t outputStripeHeight          = std::min(
        RoundUpToNearestMultiple(GetHeight(requestedOutputStripe), outputStripeHeightMin), outputStripeHeightMax);
    // The stripe depth must be a multiple of the number of srams as this is required by the firmware and
    // PLE supports.
    // The stripe depth must also be such that no stripes may start on channels that aren't a multiple of 16 and pass
    // through into the next 16, which is not supported by the DMA (e.g. a stripe starting on channel 24
    // and going to channel 48).
    // Ensure that rounding to a multiple of brickGroupChannels is ALSO a multiple of num SRAMS
    uint32_t outputStripeChannels =
        (DivRoundUp(GetChannels(outputShape), GetChannels(requestedOutputStripe)) > 1 &&
         GetChannels(requestedOutputStripe) > brickGroupChannels * shapeMultiplier.m_C)
            ? RoundUpToNearestMultiple(GetChannels(requestedOutputStripe), brickGroupChannels * shapeMultiplier.m_C)
            : RoundUpToNearestMultiple(GetChannels(requestedOutputStripe),
                                       capabilities.GetNumberOfSrams() * shapeMultiplier.m_C);

    // Calculate input stripe from output stripe
    const TensorShape& inputShape = strategyXSelectionParameters.inputShape;
    const uint32_t inputStripeHeightPre =
        AccountForFullDimension(GetHeight(outputShape), GetHeight(inputShape), outputStripeHeight, shapeMultiplier.m_H);
    const uint32_t inputStripeHeight =
        RoundUpToNearestMultiple(std::min(inputStripeHeightPre, GetHeight(inputShape)), brickGroupHeight);

    const uint32_t inputStripeWidthPre =
        AccountForFullDimension(GetWidth(outputShape), GetWidth(inputShape), outputStripeWidth, shapeMultiplier.m_W);
    const uint32_t inputStripeWidth =
        RoundUpToNearestMultiple(std::min(inputStripeWidthPre, GetWidth(inputShape)), brickGroupWidth);

    const TensorShape& weightsShape = strategyXSelectionParameters.weightsShape;

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
    const uint32_t& depthMax = strategyXSelectionParameters.depthMax;
    if (utils::DivRoundUp(GetHeight(inputShape), inputStripeHeight) > 1)
    {
        outputStripeChannels = std::min(outputStripeChannels, depthMax);
    }

    // MCE output stripe shape = requestedOutputStripe / PleShapeMultiplier
    const TensorShape mceOutputStripe = { 1, outputStripeHeight / pleShapeMultiplier.m_H,
                                          outputStripeWidth / pleShapeMultiplier.m_W,
                                          outputStripeChannels / pleShapeMultiplier.m_C };

    uint32_t strideSize =
        utils::DivRoundUp(utils::RoundUpToNearestMultiple(GetChannels(inputShape), capabilities.GetNumberOfSrams()),
                          utils::RoundUpToNearestMultiple(weightsShape[2], capabilities.GetNumberOfSrams()));

    // Same considerations done above for the outputStripeChannels.
    // The difference is that the input channels need to account the stride size
    // since all the de-interleaved input channels  need to go together.
    const uint32_t inputStripeChannels =
        (DivRoundUp(GetChannels(inputShape), requestedInputChannels) > 1 &&
         requestedInputChannels > brickGroupChannels * strideSize)
            ? RoundUpToNearestMultiple(requestedInputChannels, brickGroupChannels * strideSize)
            : RoundUpToNearestMultiple(requestedInputChannels, capabilities.GetNumberOfSrams() * strideSize);

    const TensorShape inputStripe = { 1, inputStripeHeight, inputStripeWidth, inputStripeChannels };

    // Make sure that input is DMA-able.
    if ((GetHeight(inputStripe) % brickGroupHeight != 0) || (GetWidth(inputStripe) % brickGroupWidth != 0))
    {
        return rv;
    }

    // Calculate weight stripe from output stripe.
    TensorShape weightStripe;
    if (isHwio)
    {
        const uint32_t weightStripeChannels =
            isFullyConnected
                ? (RoundUpToNearestMultiple(GetHeight(inputStripe) * GetWidth(inputStripe) * GetChannels(inputStripe),
                                            g_WeightsChannelVecProd))
                : GetChannels(inputStripe);

        weightStripe = { weightsShape[0], weightsShape[1], weightStripeChannels, GetChannels(mceOutputStripe) };
    }
    else
    {
        // Weight tensor must be HWIO
        assert(false);
    }

    // Work out the tile sizes by deciding how many stripes we want in each tile

    const std::pair<const uint32_t, const uint32_t>& pad = strategyXSelectionParameters.pad;
    const NeedBoundary needBoundaryY = GetBoundaryRequirements(pad.first, GetHeight(inputShape), GetHeight(inputStripe),
                                                               GetHeight(mceOutputStripe), weightsShape[0]);

    const bool needsBoundarySlots = needBoundaryY.m_Before || needBoundaryY.m_After;
    const uint32_t inputStripeXZ  = GetWidth(inputStripe) * GetChannels(inputStripe);

    const uint32_t boundarySlotSize = needsBoundarySlots ? (brickGroupHeight * inputStripeXZ) : 0U;
    const uint32_t defaultSlotSize  = TotalSizeBytes(inputStripe);

    const uint32_t totalSlotSize = (2U * boundarySlotSize) + defaultSlotSize;

    // Clamp this to the maximum number of stripes possible (i.e. if the image is small enough don't bother allocating
    // more space than we could use).
    const uint32_t numInputStripesTotalX = DivRoundUp(GetWidth(inputShape), GetWidth(inputStripe));
    const uint32_t numInputStripesTotalY = DivRoundUp(GetHeight(inputShape), GetHeight(inputStripe));
    const uint32_t numInputStripesTotalZ = DivRoundUp(GetChannels(inputShape), GetChannels(inputStripe));

    const NeedBoundary needBoundaryX = GetBoundaryRequirements(pad.second, GetWidth(inputShape), GetWidth(inputStripe),
                                                               GetWidth(mceOutputStripe), weightsShape[1]);

    uint32_t numInputSlots = 1U;
    numInputSlots += static_cast<uint32_t>(needBoundaryX.m_Before);
    numInputSlots += static_cast<uint32_t>(needBoundaryX.m_After);
    numInputSlots = std::min(numInputSlots, numInputStripesTotalX);

    const bool isFullHeight              = numInputStripesTotalY == 1U;
    const bool isFullWidth               = numInputStripesTotalX == 1U;
    const uint32_t numInputSlotGroupsMax = (avoidInputReloading && isFullHeight && isFullWidth)
                                               ? (numInputStripesTotalX * numInputStripesTotalY * numInputStripesTotalZ)
                                               : 2U;

    // It's better to use multiple queues if partial depth.
    const bool needSlotGroups = (GetChannels(inputShape) > GetChannels(inputStripe));
    const uint32_t numInputStripesInTile =
        numInputSlots * ((allowInputBuffering && needSlotGroups) ? numInputSlotGroupsMax : 1U);
    const uint32_t inputTile = totalSlotSize * numInputStripesInTile;

    uint32_t numWeightStripesInTile;
    if (!isFullyConnected)
    {
        if (weightsReloading == WeightsReloadingOptions::NO_RELOADING)
        {
            // First try to fit all ifm iterations in the weight tile to avoid weight reloading.
            numWeightStripesInTile = DivRoundUp(GetChannels(inputShape), GetChannels(inputStripe));
        }
        else
        {
            // If not try to weight reloading with double buffering.
            numWeightStripesInTile = weightsReloading == WeightsReloadingOptions::RELOADING_DOUBLE_BUFFERING ? 2U : 1U;
        }
    }
    else
    {
        // Fully connected: reserves two stripes for weight streaming.
        numWeightStripesInTile = 2U;
    }

    const uint32_t weightTile =
        EstimateWeightSizeBytes(weightStripe, capabilities, weightsFormat == DataFormat::HWIM) * numWeightStripesInTile;

    // To support activation compression, MCE and output stripes will need to be decoupled.
    if (activationCompression)
    {
        // The output stripe depth must be multiple of FCAF cell depth in
        // case it gets compressed.
        // FCAF wide (HxWxC=8x16x16) is the most likely format to be used for compression.
        uint32_t minFcafDepth = 16;

        // However, FCAF deep (8x8x32) will be preferred if the tensor's height and width are both less than
        // or equal to 8.
        if (outputShape[1] <= 8 && outputShape[2] <= 8)
        {
            minFcafDepth = 32;
        }

        if (minFcafDepth > outputStripeChannels)
        {
            // If the minimum output depth for FCAF is greater than the MCE output stripe depth,
            // multiple MCE stripes would need to be accumulated to form a output stripe that
            // is deep enough for FCAF.
            outputStripeChannels = minFcafDepth;
            outputStripeHeight   = RoundUpToNearestMultiple(outputShape[1], 8);
            outputStripeWidth    = RoundUpToNearestMultiple(outputShape[2], 8);
        }
    }

    const TensorShape outputStripe = { 1, outputStripeHeight, outputStripeWidth, outputStripeChannels };

    // Make sure that output is DMA-able.
    if ((GetHeight(outputStripe) % brickGroupHeight != 0) || (GetWidth(outputStripe) % brickGroupWidth != 0))
    {
        return rv;
    }

    // Outputs. We need at most 2 at a time for double-buffering.
    const uint32_t maxNumOutputStripesInTile = 2U;
    // Clamp this to the maximum number of stripes possible (i.e. if the image is small enough don't bother allocating
    // more space than we could use).
    const uint32_t numOutputStripesTotalX = DivRoundUp(GetWidth(outputShape), GetWidth(outputStripe));
    const uint32_t numOutputStripesTotalY = DivRoundUp(GetHeight(outputShape), GetHeight(outputStripe));
    const uint32_t numOutputStripesTotalZ = DivRoundUp(GetChannels(outputShape), GetChannels(outputStripe));
    const uint32_t numOutputStripesTotal  = numOutputStripesTotalX * numOutputStripesTotalY * numOutputStripesTotalZ;
    const uint32_t numOutputStripesInTile = std::min(maxNumOutputStripesInTile, numOutputStripesTotal);
    // Clamp the overall tile size to the size of the full tensor. This means that if we have a small number of stripes
    // and the last one is partial we don't waste space in the tile that will never be used.
    const uint32_t outputTileMax = TotalSizeBytes(
        TensorShape{ 1, RoundUpToNearestMultiple(GetHeight(outputShape), brickGroupHeight),
                     RoundUpToNearestMultiple(GetWidth(outputShape), brickGroupWidth),
                     RoundUpToNearestMultiple(GetChannels(outputShape), capabilities.GetNumberOfOgs()) });
    const uint32_t outputTile = std::min(TotalSizeBytes(outputStripe) * numOutputStripesInTile, outputTileMax);

    SramAllocator currentSramAllocator = strategyXSelectionParameters.sramAllocator;
    const std::pair<const bool, const uint32_t>& inputStaticAndOffset =
        strategyXSelectionParameters.inputStaticAndOffset;
    SramAllocator::UserId userId = strategyXSelectionParameters.userId;
    AllocationResult allocationResults =
        FitsInSram(userId, currentSramAllocator, capabilities, inputTile, weightTile, outputTile, inputStaticAndOffset);

    rv.success = allocationResults.m_Success;
    if (!rv.success)
    {
        return rv;
    }
    StrategyConfig& outStrategyConfig                    = rv.strategyConfig;
    outStrategyConfig.inputAllocation.stripeShape        = inputStripe;
    outStrategyConfig.inputAllocation.tileSize           = inputTile;
    outStrategyConfig.inputAllocation.numStripesInTile   = numInputStripesInTile;
    outStrategyConfig.outputAllocation.stripeShape       = outputStripe;
    outStrategyConfig.outputAllocation.tileSize          = outputTile;
    outStrategyConfig.outputAllocation.numStripesInTile  = numOutputStripesInTile;
    outStrategyConfig.weightsAllocation.stripeShape      = weightStripe;
    outStrategyConfig.weightsAllocation.tileSize         = weightTile;
    outStrategyConfig.weightsAllocation.numStripesInTile = numWeightStripesInTile;
    // If we succeeded in finding a strategy, update the sram allocation state
    rv.sramAllocator = currentSramAllocator;
    FillStrategyConfigOffsets(allocationResults, outStrategyConfig);
    return rv;
}

// Try ZXY input traversal: streaming in Z, in X and Y and XYZ output traversal (output traversal
// matters only for the Firmware).
MceStrategySelectionReturnValue TryInputZXYOutputXYZ(const StrategyXSelectionParameters& strategyXSelectionParameters)
{
    MceStrategySelectionReturnValue rv;

    rv.success = false;

    const std::pair<const bool, const uint32_t>& inputStaticAndOffset =
        strategyXSelectionParameters.inputStaticAndOffset;
    if (inputStaticAndOffset.first)
    {
        return rv;
    }

    const command_stream::MceOperation& mceOperation = strategyXSelectionParameters.mceOperation;
    const bool isFullyConnected                      = IsFullyConnected(mceOperation);

    // Sort the block config (allowedBlockConfigs is a copy)
    std::vector<command_stream::BlockConfig> allowedBlockConfigs = strategyXSelectionParameters.allowedBlockConfigs;
    std::sort(allowedBlockConfigs.begin(), allowedBlockConfigs.end(),
              [](command_stream::BlockConfig a, command_stream::BlockConfig b) {
                  return ((a.m_BlockWidth() > b.m_BlockWidth()) ||
                          ((a.m_BlockWidth() == b.m_BlockWidth()) && (a.m_BlockHeight() >= b.m_BlockHeight())));
              });

    struct Params
    {
        uint32_t blockHeight;
        uint32_t blockWidth;
        uint32_t inputStripeChannel;
        uint32_t outputStripeHeight;
        uint32_t outputStripeWidth;
        uint32_t outputStripeChannel;
        bool activationCompression;
    };

    std::vector<bool> activationCompressionOptions;

    // Activation compression options:
    // {true, false} --- not fully connected.
    // {false}       --- otherwise
    const HardwareCapabilities& capabilities = strategyXSelectionParameters.capabilities;
    if (!isFullyConnected)
    {
        activationCompressionOptions.push_back(true);
    }
    activationCompressionOptions.push_back(false);

    const WeightsReloadingOptions weightsReloading[] = { WeightsReloadingOptions::NO_RELOADING,
                                                         WeightsReloadingOptions::RELOADING_DOUBLE_BUFFERING,
                                                         WeightsReloadingOptions::RELOADING_NO_DOUBLE_BUFFERING };

    // Generate a list of parameters we pass to TryStripeShapes so we can see all the stripe shapes which could be attempted.
    std::vector<Params> paramsList;

    const TensorShape& inputShape                    = strategyXSelectionParameters.inputShape;
    const utils::ShapeMultiplier& pleShapeMultiplier = strategyXSelectionParameters.pleShapeMultiplier;
    const command_stream::UpsampleType& upsampleType = strategyXSelectionParameters.upsampleType;
    for (auto compression : activationCompressionOptions)
    {
        for (auto& currBlockConfig : allowedBlockConfigs)
        {
            if (!IsBlockConfigCompatible(currBlockConfig, capabilities, mceOperation, upsampleType))
            {
                continue;
            }

            const uint32_t currBlockWidth  = currBlockConfig.m_BlockWidth();
            const uint32_t currBlockHeight = currBlockConfig.m_BlockHeight();
            // Mce can produce a single block only.
            const uint32_t outputStripeHeight = currBlockHeight * pleShapeMultiplier.m_H;
            const uint32_t outputStripeWidth  = currBlockWidth * pleShapeMultiplier.m_W;

            for (uint32_t numInputChannelSplits = 2U; numInputChannelSplits < GetChannels(inputShape);
                 ++numInputChannelSplits)
            {
                const uint32_t inputStripeChannel  = GetChannels(inputShape) / numInputChannelSplits;
                const uint32_t outputStripeChannel = capabilities.GetNumberOfOgs() * pleShapeMultiplier.m_C;
                paramsList.push_back({ currBlockConfig.m_BlockHeight(), currBlockConfig.m_BlockWidth(),
                                       inputStripeChannel, outputStripeHeight, outputStripeWidth, outputStripeChannel,
                                       compression });
            }
        }
    }

    if (paramsList.size() == 0)
    {
        return rv;
    }

    SramAllocator sramAllocator = strategyXSelectionParameters.sramAllocator;
    auto TryConf = [&inputShape, &strategyXSelectionParameters](const Params params, const bool allowInputBuffering,
                                                                const bool avoidInputReloading,
                                                                const WeightsReloadingOptions weightsReloading) {
        assert(!avoidInputReloading || allowInputBuffering);
        MceStrategySelectionReturnValue rv =
            TryStripeShapes(strategyXSelectionParameters,
                            { 1, params.outputStripeHeight, params.outputStripeWidth, params.outputStripeChannel },
                            params.inputStripeChannel, allowInputBuffering, avoidInputReloading,
                            params.activationCompression, weightsReloading);
        if (rv.success)
        {
            StrategyConfig& outStrategyConfig = rv.strategyConfig;
            // Check that input stripe is partial depth.
            if (GetChannels(outStrategyConfig.inputAllocation.stripeShape) < GetChannels(inputShape))
            {
                outStrategyConfig.blockWidth  = params.blockWidth;
                outStrategyConfig.blockHeight = params.blockHeight;
                outStrategyConfig.strategy    = Strategy::STRATEGY_X;
            }
            else
            {
                rv.success = false;
            }
        }
        return rv;
    };

    // Below it is going to try:
    // a. Fit all input stripes in the tile to avoid reloading and allow buffering
    // b. Fit at least two input stripes (including neighbouring) for double buffering
    // c. No buffering
    // with all possible weights reloading options as following:
    // a. Fit all weight stripes in the tile (NO_RELOADING)
    // b. Fit at least two weight stripes stripes (RELOADING_DOUBLE_BUFFERING)
    // c. Only single weight stripe can fit so no buffering (RELOADING_NO_DOUBLE_BUFFERING)

    for (auto& tryWeightsReloading : weightsReloading)
    {
        // a. Try all configurations using input buffering.
        for (auto params : paramsList)
        {
            rv = TryConf(params, true, true, tryWeightsReloading);
            if (rv.success)
            {
                return rv;
            }
        }

        // b. If here it means that it cannot avoid input reloading.
        for (auto params : paramsList)
        {
            rv = TryConf(params, true, false, tryWeightsReloading);
            if (rv.success)
            {
                return rv;
            }
        }

        // c. If here it means that it cannot do input buffering.
        for (auto params : paramsList)
        {
            rv = TryConf(params, false, false, tryWeightsReloading);
            if (rv.success)
            {
                return rv;
            }
        }
    }

    return rv;
}

// Try XY input traversal: streaming in X and Y and XYZ output traversal (output traversal)
// matters only for the Firmware).
MceStrategySelectionReturnValue TryInputXYOutputXYZ(const StrategyXSelectionParameters& strategyXSelectionParameters)
{
    MceStrategySelectionReturnValue rv;
    rv.success = false;

    const std::pair<const bool, const uint32_t>& inputStaticAndOffset =
        strategyXSelectionParameters.inputStaticAndOffset;
    if (inputStaticAndOffset.first)
    {
        return rv;
    }

    const command_stream::MceOperation& mceOperation = strategyXSelectionParameters.mceOperation;
    const bool isFullyConnected                      = (mceOperation == command_stream::MceOperation::FULLY_CONNECTED);

    // Allow only fully connected since this is equivalent of strategy 1 not yet fully supported and
    // tested in strategy X.
    if (!isFullyConnected)
    {
        return rv;
    }

    // Sort the block config (allowedBlockConfigs is a copy)
    std::vector<command_stream::BlockConfig> allowedBlockConfigs = strategyXSelectionParameters.allowedBlockConfigs;
    std::sort(allowedBlockConfigs.begin(), allowedBlockConfigs.end(),
              [](command_stream::BlockConfig a, command_stream::BlockConfig b) {
                  return ((a.m_BlockWidth() > b.m_BlockWidth()) ||
                          ((a.m_BlockWidth() == b.m_BlockWidth()) && (a.m_BlockHeight() >= b.m_BlockHeight())));
              });

    struct Params
    {
        uint32_t blockHeight;
        uint32_t blockWidth;
        uint32_t inputStripeChannel;
        uint32_t outputStripeHeight;
        uint32_t outputStripeWidth;
        uint32_t outputStripeChannel;
    };

    // Generate a list of parameters we pass to TryStripeShapes so we can see all the stripe shapes which could be attempted.
    std::vector<Params> paramsList;
    const HardwareCapabilities& capabilities         = strategyXSelectionParameters.capabilities;
    const command_stream::UpsampleType& upsampleType = strategyXSelectionParameters.upsampleType;
    const utils::ShapeMultiplier& pleShapeMultiplier = strategyXSelectionParameters.pleShapeMultiplier;
    const TensorShape& inputShape                    = strategyXSelectionParameters.inputShape;
    for (auto& currBlockConfig : allowedBlockConfigs)
    {
        if (!IsBlockConfigCompatible(currBlockConfig, capabilities, mceOperation, upsampleType))
        {
            continue;
        }

        const uint32_t currBlockWidth  = currBlockConfig.m_BlockWidth();
        const uint32_t currBlockHeight = currBlockConfig.m_BlockHeight();
        // Use a single block only.
        const uint32_t outputStripeHeight = currBlockHeight * pleShapeMultiplier.m_H;
        const uint32_t outputStripeWidth  = currBlockWidth * pleShapeMultiplier.m_W;

        const uint32_t inputStripeChannel  = GetChannels(inputShape);
        const uint32_t outputStripeChannel = capabilities.GetNumberOfOgs() * pleShapeMultiplier.m_C;
        paramsList.push_back({ currBlockHeight, currBlockConfig.m_BlockWidth(), inputStripeChannel, outputStripeHeight,
                               outputStripeWidth, outputStripeChannel });
    }

    if (paramsList.size() == 0)
    {
        return rv;
    }

    auto TryConf = [&strategyXSelectionParameters](const Params params, const bool allowInputBuffering) {
        MceStrategySelectionReturnValue rv =
            TryStripeShapes(strategyXSelectionParameters,
                            { 1, params.outputStripeHeight, params.outputStripeWidth, params.outputStripeChannel },
                            params.inputStripeChannel, allowInputBuffering);
        StrategyConfig& outStrategyConfig = rv.strategyConfig;
        if (rv.success)
        {
            outStrategyConfig.blockWidth  = params.blockWidth;
            outStrategyConfig.blockHeight = params.blockHeight;
            outStrategyConfig.strategy    = Strategy::STRATEGY_X;
        }
        return rv;
    };

    // Try all configurations using input buffering.
    for (auto params : paramsList)
    {
        rv = TryConf(params, true);
        if (rv.success)
        {
            return rv;
        }
    }

    // If here it means that it cannot do input buffering.
    for (auto params : paramsList)
    {
        rv = TryConf(params, false);
        if (rv.success)
        {
            return rv;
        }
    }

    return rv;
}

template <typename T>
bool IsStrategyAllowed(const std::vector<IStrategy*>& strategies)
{
    for (IStrategy* s : strategies)
    {
        if (dynamic_cast<T*>(s))
        {
            return true;
        }
    }
    return false;
}

}    //namespace

bool IsStrategyX(const command_stream::MceOperation& mceOperation,
                 const StrategyConfig& strategyConfig,
                 const CompilerMceAlgorithm algorithm,
                 const std::vector<IStrategy*>& allowedStrategies)
{
    const bool isSupportedMceOperation = (mceOperation == command_stream::MceOperation::CONVOLUTION) ||
                                         (mceOperation == command_stream::MceOperation::FULLY_CONNECTED);
    const bool isSupportedAlgorithm = (algorithm == CompilerMceAlgorithm::Direct);
    const bool isSupportedStrategy =
        (strategyConfig.strategy == Strategy::STRATEGY_7) || (strategyConfig.strategy == Strategy::NONE);
    const bool isAllowedStrategy = (IsStrategyAllowed<Strategy7>(allowedStrategies)) ||
                                   (mceOperation == command_stream::MceOperation::FULLY_CONNECTED);
    return isSupportedMceOperation && isSupportedAlgorithm && isSupportedStrategy && isAllowedStrategy;
}

MceStrategySelectionReturnValue TryStrategyX(const StrategyXSelectionParameters& strategyXSelectionParameters)
{
    MceStrategySelectionReturnValue rv = TryInputXYOutputXYZ(strategyXSelectionParameters);
    if (rv.success)
    {
        return rv;
    }

    rv = TryInputZXYOutputXYZ(strategyXSelectionParameters);

    return rv;
}

}    // namespace support_library
}    // namespace ethosn
