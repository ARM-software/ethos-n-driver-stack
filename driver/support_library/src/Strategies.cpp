//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "Strategies.hpp"

#include "../include/ethosn_support_library/Support.hpp"
#include "Compiler.hpp"
#include "Pass.hpp"
#include "Utils.hpp"

namespace ethosn
{
namespace support_library
{

using namespace utils;

namespace
{

// We limit the number of buffers in a tile to 3 because using 4 buffers in the tile on VGG16
// on the 1 MB SRAM configuration causes a performance regression.
// We need to further investigate this trade-off.
constexpr uint32_t g_DefaultMaxNumInputBuffersInTile  = 3;
constexpr uint32_t g_DefaultMaxNumWeightBuffersInTile = 2;

struct AllocationResult
{
    bool m_Success;
    uint32_t m_InputOffset;
    uint32_t m_WeightOffset;
    uint32_t m_OutputOffset;
    uint32_t m_PleOffset;
};

AllocationResult FitsInSram(SramAllocator& sramAllocator,
                            const HardwareCapabilities& capabilities,
                            uint32_t input,
                            uint32_t weight,
                            uint32_t output,
                            std::pair<bool, uint32_t> inputStaticAndOffset)
{
    AllocationResult res;
    res.m_Success          = true;
    auto pleAllocateResult = sramAllocator.Allocate(capabilities.GetMaxPleSize(), AllocationPreference::Start, "ple");
    res.m_Success &= pleAllocateResult.first;
    res.m_PleOffset = pleAllocateResult.second;

    if (inputStaticAndOffset.first)
    {
        res.m_InputOffset = inputStaticAndOffset.second;
    }
    else
    {
        assert(input > 0);
        auto inputAllocateResult =
            sramAllocator.Allocate(input / capabilities.GetNumberOfSrams(), AllocationPreference::Start, "input");
        res.m_Success &= inputAllocateResult.first;
        res.m_InputOffset = inputAllocateResult.second;
    }

    // Try to allocate output and input tiles in opposite ends of SRAM, so we can overlap loading/saving
    AllocationPreference outputAllocationPreference;
    AllocationPreference weightAllocationPreference;
    if (res.m_InputOffset <= (capabilities.GetTotalSramSize() / capabilities.GetNumberOfSrams()) / 2)
    {
        outputAllocationPreference = AllocationPreference::End;
        weightAllocationPreference = AllocationPreference::Start;
    }
    else
    {
        outputAllocationPreference = AllocationPreference::Start;
        weightAllocationPreference = AllocationPreference::End;
    }

    // There are passes without weights but still need to decide on strategies i.e. PlePasses
    // We don't allocate anything if there are no weights.
    assert(weight > 0);
    auto weightAllocateResult =
        sramAllocator.Allocate(weight / capabilities.GetNumberOfSrams(), weightAllocationPreference, "weights");
    res.m_Success &= weightAllocateResult.first;
    res.m_WeightOffset = weightAllocateResult.second;

    assert(output > 0);
    auto outputAllocateResult =
        sramAllocator.Allocate(output / capabilities.GetNumberOfSrams(), outputAllocationPreference, "outputs");
    res.m_Success &= outputAllocateResult.first;
    res.m_OutputOffset = outputAllocateResult.second;

    return res;
}

void FillTensorConfigOffsets(const AllocationResult& allocationResults, TensorConfig& outTensorConfig)
{
    outTensorConfig.pleAllocation.offset     = allocationResults.m_PleOffset;
    outTensorConfig.inputAllocation.offset   = allocationResults.m_InputOffset;
    outTensorConfig.weightsAllocation.offset = allocationResults.m_WeightOffset;
    outTensorConfig.outputAllocation.offset  = allocationResults.m_OutputOffset;
}

// Given a requested shape for the output stripe (which is not required to be rounded at all),
// calculates what the actual stripe sizes would be (accounting for hardware and firmware constraints)
// and what the tile sizes would be (accounting for double-buffering etc.) and checks if all this would
// fit into SRAM.
// By keeping all the logic of the confusing rounding in this one function it lets the per-Strategy functions
// be nice and simple and concentrate just on looping over possible stripe sizes.
bool TryStripeShapes(SramAllocator& sramAllocator,
                     const TensorShape& requestedOutputStripe,
                     const TensorShape& inputShape,
                     const TensorShape& outputShape,
                     DataFormat weightsFormat,
                     const TensorShape& weightsShape,
                     const HardwareCapabilities& capabilities,
                     const utils::ShapeMultiplier& shapeMultiplier,
                     std::pair<bool, uint32_t> inputStaticAndOffset,
                     TensorConfig& outTensorConfig,
                     const uint32_t depthMax,
                     const uint32_t maxNumWeightBuffersInTile = g_DefaultMaxNumWeightBuffersInTile,
                     const uint32_t maxNumInputBuffersInTile  = g_DefaultMaxNumInputBuffersInTile)
{
    const uint32_t brickGroupWidth    = capabilities.GetBrickGroupShape()[1];
    const uint32_t brickGroupHeight   = capabilities.GetBrickGroupShape()[2];
    const uint32_t brickGroupChannels = capabilities.GetBrickGroupShape()[3];

    // Round the requested output stripe shape to appropriate boundaries
    // Width and height must be a multiple of the brick group size in order to be DMA-able.
    // Additionally, if the input stripes are to be smaller than the input stripe then we must make sure the
    // input stripe sizes are also valid.
    const uint32_t outputStripeWidthMultiple = std::max(brickGroupWidth, brickGroupWidth * shapeMultiplier.m_W);
    const uint32_t outputStripeWidthMax      = RoundUpToNearestMultiple(outputShape[2], brickGroupWidth);
    const uint32_t outputStripeWidth =
        std::min(RoundUpToNearestMultiple(requestedOutputStripe[2], outputStripeWidthMultiple), outputStripeWidthMax);

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

    // Local function to account for the fact that if the output stripe in a dimension is the entire tensor
    // we need to use the full input tensor in that dimension
    auto AccountForFullDimension = [&](auto outputTensorDim, auto inputTensorDim, auto outputStripeDim,
                                       auto multiplier) {
        if (outputStripeDim >= outputTensorDim)
        {
            return inputTensorDim;
        }
        else
        {
            return outputStripeDim / multiplier;
        }
    };
    const uint32_t inputStripeHeightPre =
        AccountForFullDimension(outputShape[1], inputShape[1], outputStripeHeight, shapeMultiplier.m_W);
    const uint32_t inputStripeHeight =
        RoundUpToNearestMultiple(std::min(inputStripeHeightPre, inputShape[1]), brickGroupHeight);

    const uint32_t inputStripeWidthPre =
        AccountForFullDimension(outputShape[2], inputShape[2], outputStripeWidth, shapeMultiplier.m_H);
    const uint32_t inputStripeWidth =
        RoundUpToNearestMultiple(std::min(inputStripeWidthPre, inputShape[2]), brickGroupWidth);

    // Output stripe depth maximum is set for MAXPOOLING_3x3/(2,2)
    // so that the PLE can manage spilling if the number of stripes is more than 1.
    if (utils::DivRoundUp(inputShape[1], inputStripeHeight) > 1)
    {
        outputStripeChannels = std::min(outputStripeChannels, depthMax);
    }

    const TensorShape outputStripe = { 1, outputStripeHeight, outputStripeWidth, outputStripeChannels };

    // Calculate input stripe from output stripe
    const TensorShape inputStripe = { 1, inputStripeHeight, inputStripeWidth,
                                      RoundUpToNearestMultiple(inputShape[3], capabilities.GetNumberOfSrams()) };

    // Calculate weight stripe from output stripe.
    TensorShape weightStripe;
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
    }
    else
    {
        // Weight tensor must be HWIO or HWIM
        assert(false);
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
        return false;
    }
    // Clamp the overall tile size to the size of the full tensor. This means that if we have a small number of stripes
    // and the last one is partial we don't waste space in the tile that will never be used.
    const uint32_t inputTileMax =
        TotalSizeBytes(TensorShape{ 1, RoundUpToNearestMultiple(inputShape[1], brickGroupWidth),
                                    RoundUpToNearestMultiple(inputShape[2], brickGroupHeight),
                                    RoundUpToNearestMultiple(inputShape[3], capabilities.GetNumberOfSrams()) });
    // Account for the boundary slots if required by the strategy and the kernel size. It uses the normal
    // slot triple buffering in the width dimension if needed.
    const uint32_t boundarySlotsSize =
        inputShape[1] > inputStripe[1] && inputShape[2] > inputStripe[2] && weightsShape[0] > 1
            ? capabilities.GetNumBoundarySlots() * capabilities.GetBoundaryStripeHeight() * inputStripe[2] *
                  inputStripe[3]
            : 0;
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
        TotalSizeBytes(TensorShape{ 1, RoundUpToNearestMultiple(outputShape[1], brickGroupWidth),
                                    RoundUpToNearestMultiple(outputShape[2], brickGroupHeight),
                                    RoundUpToNearestMultiple(outputShape[3], capabilities.GetNumberOfSrams()) });
    const uint32_t outputTile = std::min(TotalSizeBytes(outputStripe) * numOutputStripesInTile, outputTileMax);

    if (numInputStripesTotalX < numOutputStripesX || numInputStripesTotalY < numOutputStripesY)
    {
        // This is a limitation of the current StripeStreamer code in the firmware.
        // Note that there is only very limited support for the case where there are
        // more input stripes than output stripes, but it isn't clear what those
        // limitations are so this check is probably overly permissive for those cases.
        return false;
    }

    SramAllocator currentSramAllocator = sramAllocator;
    AllocationResult allocationResults =
        FitsInSram(currentSramAllocator, capabilities, inputTile, weightTile, outputTile, inputStaticAndOffset);
    if (!allocationResults.m_Success)
    {
        return false;
    }
    outTensorConfig.inputAllocation.stripeShape   = inputStripe;
    outTensorConfig.inputAllocation.tileSize      = inputTile;
    outTensorConfig.outputAllocation.stripeShape  = outputStripe;
    outTensorConfig.outputAllocation.tileSize     = outputTile;
    outTensorConfig.weightsAllocation.stripeShape = weightStripe;
    outTensorConfig.weightsAllocation.tileSize    = weightTile;
    // If we succeeded in finding a strategy, update the sram allocation state
    sramAllocator = currentSramAllocator;
    FillTensorConfigOffsets(allocationResults, outTensorConfig);
    return true;
}

}    // namespace

using namespace utils;

bool Strategy0::TrySetup(TensorConfig& tensorConfig,
                         SramAllocator& sramAllocator,
                         const TensorShape& inputShape,
                         const TensorShape& outputShape,
                         DataFormat weightsFormat,
                         const TensorShape& weightsShape,
                         const ethosn::command_stream::BlockConfig& blockConfig,
                         const HardwareCapabilities& capabilities,
                         const utils::ShapeMultiplier& shapeMultiplier,
                         std::pair<bool, uint32_t> inputStaticAndOffset,
                         CompilerMceAlgorithm,
                         const uint32_t depthMax)
{
    // Try splitting into two stripes at first, then move until we find something that works.
    // Stop when we reach the point where the MCE output stripe would be less than the block height.
    // Unfortunately we don't have the MCE output stripe here, so we have to make do with the input stripe.
    const uint32_t maxSplits = DivRoundUp(inputShape[1], blockConfig.m_BlockHeight());

    struct Strategy0Params
    {
        uint32_t outputStripeHeight;
        uint32_t numInputBuffers;
    };
    // Generate a list of parameters we pass to TryStripeShapes so we can see all the stripe shapes which could be attempted.
    std::vector<Strategy0Params> paramsList;
    for (uint32_t numHeightSplits = 2; numHeightSplits <= maxSplits; ++numHeightSplits)
    {
        // First try a solution with 4 slots in the input tile.
        for (uint32_t numInputBuffers = 4; numInputBuffers >= g_DefaultMaxNumInputBuffersInTile; --numInputBuffers)
        {
            const uint32_t outputStripeHeight = outputShape[1] / numHeightSplits;
            paramsList.push_back({ outputStripeHeight, numInputBuffers });
        }
    }

    for (auto params : paramsList)
    {
        if (TryStripeShapes(sramAllocator, { 1, params.outputStripeHeight, outputShape[2], outputShape[3] }, inputShape,
                            outputShape, weightsFormat, weightsShape, capabilities, shapeMultiplier,
                            inputStaticAndOffset, tensorConfig, depthMax, g_DefaultMaxNumWeightBuffersInTile,
                            params.numInputBuffers))
        {
            tensorConfig.blockWidth  = blockConfig.m_BlockWidth();
            tensorConfig.blockHeight = blockConfig.m_BlockHeight();
            tensorConfig.strategy    = Strategy::STRATEGY_0;
            return true;
        }
    }

    return false;
}

bool Strategy1::TrySetup(TensorConfig& tensorConfig,
                         SramAllocator& sramAllocator,
                         const TensorShape& inputShape,
                         const TensorShape& outputShape,
                         DataFormat weightsFormat,
                         const TensorShape& weightsShape,
                         const ethosn::command_stream::BlockConfig& blockConfig,
                         const HardwareCapabilities& capabilities,
                         const utils::ShapeMultiplier& shapeMultiplier,
                         std::pair<bool, uint32_t> inputStaticAndOffset,
                         CompilerMceAlgorithm,
                         const uint32_t depthMax)
{
    auto TrySolution = [&](const uint32_t outputStripeChannels, uint32_t numWeightBuffers) {
        if (TryStripeShapes(sramAllocator, { 1, outputShape[1], outputShape[2], outputStripeChannels }, inputShape,
                            outputShape, weightsFormat, weightsShape, capabilities, shapeMultiplier,
                            inputStaticAndOffset, tensorConfig, depthMax, numWeightBuffers))
        {
            tensorConfig.blockWidth  = blockConfig.m_BlockWidth();
            tensorConfig.blockHeight = blockConfig.m_BlockHeight();
            tensorConfig.strategy    = Strategy::STRATEGY_1;
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
            return true;
        }
    }
    return false;
}

bool Strategy3::TrySetup(TensorConfig& tensorConfig,
                         SramAllocator& sramAllocator,
                         const TensorShape& inputShape,
                         const TensorShape& outputShape,
                         DataFormat weightsFormat,
                         const TensorShape& weightsShape,
                         const ethosn::command_stream::BlockConfig& blockConfig,
                         const HardwareCapabilities& capabilities,
                         const utils::ShapeMultiplier& shapeMultiplier,
                         std::pair<bool, uint32_t> inputStaticAndOffset,
                         CompilerMceAlgorithm,
                         const uint32_t depthMax)
{
    if (TryStripeShapes(sramAllocator, outputShape, inputShape, outputShape, weightsFormat, weightsShape, capabilities,
                        shapeMultiplier, inputStaticAndOffset, tensorConfig, depthMax))
    {
        tensorConfig.blockWidth  = blockConfig.m_BlockWidth();
        tensorConfig.blockHeight = blockConfig.m_BlockHeight();
        tensorConfig.strategy    = Strategy::STRATEGY_3;
        return true;
    }

    return false;
}

bool Strategy4::TrySetup(TensorConfig& tensorConfig,
                         SramAllocator& originalSramAllocator,
                         const TensorShape& inputShape,
                         const TensorShape& outputShape,
                         DataFormat weightsFormat,
                         const TensorShape& weightsShape,
                         const ethosn::command_stream::BlockConfig& blockConfig,
                         const HardwareCapabilities& capabilities,
                         const utils::ShapeMultiplier& shapeMultiplier,
                         std::pair<bool, uint32_t> inputStaticAndOffset,
                         CompilerMceAlgorithm,
                         const uint32_t depthMax)
{
    using namespace ethosn::command_stream;
    using namespace ethosn;

    if (inputStaticAndOffset.first)
    {
        return false;
    }

    uint32_t inputStripeWidth = capabilities.GetBrickGroupShape()[2];
    // 3x3 conv needs a tile size that fits 3 stripes rather than just 2
    const uint32_t maxNumInputStripesInTile = (weightsShape[1] > 1) ? 3 : 2;

    // For strided convolutions or pooling the OFM size (width*height) is a fraction of the
    // IFM size. For example a 32x24 image might be scaled down by a factor of 2 to 16x12.
    // The output stripe width needs to be a multiple of patch width to allow OFM save operations
    // in the control unit firmware.
    uint32_t outputStripeWidth =
        RoundUpToNearestMultiple(inputStripeWidth * outputShape[2] / inputShape[2], capabilities.GetPatchShape()[2]);
    uint32_t outputTileWidth = utils::RoundUpToNearestMultiple(outputStripeWidth, capabilities.GetBrickGroupShape()[2]);

    // clang-format off
    TensorShape inputStripe = { inputShape[0],
                                RoundUpToNearestMultiple(inputShape[1], capabilities.GetBrickGroupShape()[1]),
                                inputStripeWidth,
                                RoundUpToNearestMultiple(inputShape[3], capabilities.GetNumberOfSrams()) };

    // Clamp this to the maximum number of stripes possible (i.e. if the image is small enough don't bother allocating
    // more space than we could use).
    const uint32_t numInputStripesTotal = DivRoundUp(inputShape[2], inputStripe[2]);
    const uint32_t numInputStripesInTile = std::min(maxNumInputStripesInTile, numInputStripesTotal);
    uint32_t inputTileWidth = inputStripeWidth * numInputStripesInTile;

    uint32_t inputTile = inputShape[0] *
                         RoundUpToNearestMultiple(inputShape[1], capabilities.GetBrickGroupShape()[1]) *
                         inputTileWidth *
                         RoundUpToNearestMultiple(inputShape[3], capabilities.GetNumberOfSrams());
    // clang-format on

    // NNXSW-1082: Force strategy 4 to use the minimum number of stripe depths
    uint32_t ofmRegion = std::min(outputShape[3], capabilities.GetNumberOfOfm());

    uint32_t stripeDepth = RoundUpToNearestMultiple(ofmRegion, capabilities.GetNumberOfSrams());
    stripeDepth          = std::min(stripeDepth, depthMax);
    uint32_t strideSize =
        utils::DivRoundUp(utils::RoundUpToNearestMultiple(inputShape[3], capabilities.GetNumberOfSrams()),
                          utils::RoundUpToNearestMultiple(weightsShape[2], capabilities.GetNumberOfSrams()));
    uint32_t tileDepth = stripeDepth * 2;

    uint32_t outStripeDepth = stripeDepth * shapeMultiplier.m_C;

    if (DivRoundUp(inputShape[1], inputStripe[1]) > 1)
    {
        outStripeDepth = std::min(depthMax, outStripeDepth);
    }

    // clang-format off
    // The OFM and weight tiles are double buffered, allowing the CEs to work on
    // one stripe at the same time as the MCE loads new weights and outputs finished OFMs
    TensorShape outputStripe = { outputShape[0],
                                 RoundUpToNearestMultiple(outputShape[1], capabilities.GetBrickGroupShape()[1]),
                                 outputStripeWidth,
                                 outStripeDepth };

    uint32_t outputTile = outputShape[0] *
                          RoundUpToNearestMultiple(outputShape[1], capabilities.GetBrickGroupShape()[1]) *
                          outputTileWidth *
                          tileDepth * shapeMultiplier.m_C;

    TensorShape weightStripe;
    if (weightsFormat == DataFormat::HWIO)
    {
        weightStripe = { weightsShape[0],
                         weightsShape[1],
                         inputShape[3],
                         stripeDepth };

    }
    else if (weightsFormat == DataFormat::HWIM)
    {
        weightStripe = { weightsShape[0],
                         weightsShape[1],
                         stripeDepth * strideSize,
                         weightsShape[3] };
    }
    else
    {
        // Weight tensor must be HWIO or HWIM
        assert(false);
    }

    // Choose the weight tile. First try double-buffering the weight stripes (i.e. tile = 2 x stripe) but if
    // this does not fit then single-buffering will have to do.
    for (uint32_t numStripesInWeightTile = 2; numStripesInWeightTile >= 1; --numStripesInWeightTile)
    {
        // clang-format on
        SramAllocator sramAllocator = originalSramAllocator;
        bool isHwim                 = weightsFormat == DataFormat::HWIM;
        const uint32_t weightTile =
            EstimateWeightSizeBytes(weightStripe, capabilities, isHwim) * numStripesInWeightTile;
        AllocationResult allocationResults =
            FitsInSram(sramAllocator, capabilities, inputTile, weightTile, outputTile, inputStaticAndOffset);
        if (allocationResults.m_Success)
        {
            tensorConfig.inputAllocation.stripeShape   = inputStripe;
            tensorConfig.inputAllocation.tileSize      = inputTile;
            tensorConfig.outputAllocation.stripeShape  = outputStripe;
            tensorConfig.outputAllocation.tileSize     = outputTile;
            tensorConfig.weightsAllocation.stripeShape = weightStripe;
            tensorConfig.weightsAllocation.tileSize    = weightTile;
            tensorConfig.blockWidth                    = blockConfig.m_BlockWidth();
            tensorConfig.blockHeight                   = blockConfig.m_BlockHeight();
            tensorConfig.strategy                      = Strategy::STRATEGY_4;
            originalSramAllocator                      = sramAllocator;
            FillTensorConfigOffsets(allocationResults, tensorConfig);
            return true;
        }
    }

    return false;
}

bool Strategy6::TrySetup(TensorConfig& tensorConfig,
                         SramAllocator& originalSramAllocator,
                         const TensorShape& inputShape,
                         const TensorShape& outputShape,
                         DataFormat weightsFormat,
                         const TensorShape& weightsShape,
                         const ethosn::command_stream::BlockConfig& blockConfig,
                         const HardwareCapabilities& capabilities,
                         const utils::ShapeMultiplier& shapeMultiplier,
                         std::pair<bool, uint32_t> inputStaticAndOffset,
                         CompilerMceAlgorithm,
                         const uint32_t depthMax)
{
    if (inputStaticAndOffset.first)
    {
        return false;
    }

    // Try splitting into two (for width and height) at first, then move until we find something that works.
    // Stop when we reach the point where the MCE output stripe would be less than the block sizes.
    // Unfortunately we don't have the MCE output stripe here, so we have to make do with the input stripe.
    const uint32_t maxHeightSplit = DivRoundUp(inputShape[1], blockConfig.m_BlockHeight());
    const uint32_t maxWidthSplit  = DivRoundUp(inputShape[2], blockConfig.m_BlockWidth());

    struct Strategy6Params
    {
        uint32_t outputStripeHeight;
        uint32_t outputStripeWidth;
        uint32_t outputStripeChannel;
    };
    // Generate a list of parameters we pass to TryStripeShapes so we can see all the stripe shapes which could be attempted.
    std::vector<Strategy6Params> paramsList;
    // Try without splitting the channels at first
    for (uint32_t numChannelSplits = 1; numChannelSplits < outputShape[3]; ++numChannelSplits)
    {
        for (uint32_t numWidthSplits = 2; numWidthSplits <= maxWidthSplit; ++numWidthSplits)
        {
            for (uint32_t numHeightSplits = 2; numHeightSplits <= maxHeightSplit; ++numHeightSplits)
            {
                const uint32_t outputStripeHeight  = outputShape[1] / numHeightSplits;
                const uint32_t outputStripeWidth   = outputShape[2] / numWidthSplits;
                const uint32_t outputStripeChannel = outputShape[3] / numChannelSplits;
                paramsList.push_back({ outputStripeHeight, outputStripeWidth, outputStripeChannel });
            }
        }
    }

    for (auto params : paramsList)
    {
        if (TryStripeShapes(originalSramAllocator,
                            { 1, params.outputStripeHeight, params.outputStripeWidth, params.outputStripeChannel },
                            inputShape, outputShape, weightsFormat, weightsShape, capabilities, shapeMultiplier,
                            inputStaticAndOffset, tensorConfig, depthMax))
        {
            tensorConfig.blockWidth  = blockConfig.m_BlockWidth();
            tensorConfig.blockHeight = blockConfig.m_BlockHeight();
            tensorConfig.strategy    = Strategy::STRATEGY_6;
            return true;
        }
    }

    return false;
}

// Scheduling strategy to support IFM depth streaming
// Limitations:
// (1) IFM split in Z direction only, no split in XY
// (2) Winograd is not supported
bool Strategy7::TrySetup(TensorConfig& tensorConfig,
                         SramAllocator& originalSramAllocator,
                         const TensorShape& inputShape,
                         const TensorShape& outputShape,
                         DataFormat weightsFormat,
                         const TensorShape& weightsShape,
                         const ethosn::command_stream::BlockConfig& blockConfig,
                         const HardwareCapabilities& capabilities,
                         const utils::ShapeMultiplier& shapeMultiplier,
                         std::pair<bool, uint32_t> inputStaticAndOffset,
                         CompilerMceAlgorithm algorithm,
                         const uint32_t)
{
    if (inputStaticAndOffset.first)
    {
        return false;
    }

    const uint32_t numAccumulatorsPerOg = capabilities.GetTotalAccumulatorsPerEngine();
    const uint32_t brickGroupChannels   = capabilities.GetBrickGroupShape()[3];

    // the depth is constrained by the number of OFMs that can be produced in one iteration
    const uint32_t depthStripe = capabilities.GetNumberOfOfm();

    if (algorithm == CompilerMceAlgorithm::Winograd)
    {
        return false;
    }
    if ((blockConfig.m_BlockWidth() * blockConfig.m_BlockHeight()) > numAccumulatorsPerOg ||
        (blockConfig.m_BlockWidth() < outputShape[2]))
    {
        // Because of the IFM streaming in Z, (1) the output block shape
        // in XY is limited by the number of OFMs which can be produced in one iteration
        // (2) only supports YZ streaming and no split in X.
        return false;
    }

    // Restriction when depth streaming (multiple iterations per output):
    // block dim = stripe dimension in XY plane.
    const TensorShape outputStripe = { outputShape[0], blockConfig.m_BlockHeight(), blockConfig.m_BlockWidth(),
                                       depthStripe * shapeMultiplier.m_C };

    uint32_t inputStripeHeight = std::min(inputShape[1], outputStripe[1] * inputShape[1] / outputShape[1]);
    uint32_t inputStripeWidth  = std::min(inputShape[2], outputStripe[2] * inputShape[2] / outputShape[2]);

    uint32_t strideSize =
        utils::DivRoundUp(utils::RoundUpToNearestMultiple(inputShape[3], capabilities.GetNumberOfSrams()),
                          utils::RoundUpToNearestMultiple(weightsShape[2], capabilities.GetNumberOfSrams()));
    TensorShape inputStripe = {
        inputShape[0], utils::RoundUpToNearestMultiple(inputStripeHeight, capabilities.GetBrickGroupShape()[1]),
        utils::RoundUpToNearestMultiple(inputStripeWidth, capabilities.GetBrickGroupShape()[2]),
        weightsFormat == DataFormat::HWIO
            ? utils::RoundUpToNearestMultiple(inputShape[3], capabilities.GetNumberOfSrams())
            : depthStripe * strideSize
    };

    uint32_t numInputStripesTile = 2;
    if (((inputShape[1] + inputStripeHeight - 1) / inputStripeHeight) > 2 && weightsShape[0] > 1)
    {
        // three stripes are required in the tile if
        // (1) StripeH > 2
        // (2) weightH > 1
        numInputStripesTile += 1;
    }

    // output stripe is also double buffered in the tile
    uint32_t outputTile = outputStripe[0] * outputStripe[1] * outputStripe[2] * outputStripe[3] * 2;

    // initialise weight stripe.
    TensorShape weightStripe;
    if (weightsFormat == DataFormat::HWIO)
    {
        // Note that the "I" dimension is left as zero for now and set during the below loop.
        weightStripe = { weightsShape[0], weightsShape[1], 0, outputStripe[3] };
    }
    else if (weightsFormat == DataFormat::HWIM)
    {
        weightStripe = { weightsShape[0], weightsShape[1], depthStripe * strideSize, weightsShape[3] };
    }
    else
    {
        // Weight tensor must be HWIO or HWIM
        assert(false);
    }

    uint32_t inputStripeDepth = inputStripe[3];

    SramAllocator sramAllocator;
    AllocationResult allocationResults;

    uint32_t inputTile;
    uint32_t weightTile;
    if (weightsFormat == DataFormat::HWIO)
    {
        // For regular convolution, iteratively reduce the input stripe depth until it fits.
        do
        {
            if (inputStripeDepth <= depthStripe * strideSize)
            {
                return false;
            }

            // halve the input stripe depth in order to fit IFM into the SRAM
            inputStripeDepth /= 2;
            // The stripe depth must also be such that no stripes may start on channels that aren't a multiple of 16 and pass
            // through into the next 16, which is not supported by the DMA (e.g. a stripe starting on channel 24
            // and going to channel 48).
            inputStripeDepth =
                (DivRoundUp(inputShape[3], inputStripeDepth) > 1 && inputStripeDepth > brickGroupChannels * strideSize)
                    ? RoundUpToNearestMultiple(inputStripeDepth, brickGroupChannels * strideSize)
                    : RoundUpToNearestMultiple(inputStripeDepth, capabilities.GetNumberOfSrams() * strideSize);

            // update the input and weight stripe tensor accordingly
            inputStripe[3] = inputStripeDepth;
            assert(weightsFormat == DataFormat::HWIO);
            weightStripe[2] = inputStripeDepth;
            utils::RoundUpToNearestMultiple(weightStripe[2] / 2, capabilities.GetNumberOfSrams() * strideSize);
            inputTile  = inputStripe[0] * inputStripe[1] * inputStripe[2] * inputStripe[3] * numInputStripesTile;
            weightTile = EstimateWeightSizeBytes(weightStripe, capabilities, false) * 2;

            sramAllocator = originalSramAllocator;
            allocationResults =
                FitsInSram(sramAllocator, capabilities, inputTile, weightTile, outputTile, inputStaticAndOffset);
        } while (allocationResults.m_Success == false);
    }
    else if (weightsFormat == DataFormat::HWIM)
    {
        // For depthwise, we start with the smallest input stripe depth anyway
        // (as it must be equal to the output stripe depth) so there is only one configuration to try.
        inputTile     = inputStripe[0] * inputStripe[1] * inputStripe[2] * inputStripe[3] * numInputStripesTile;
        weightTile    = EstimateWeightSizeBytes(weightStripe, capabilities, false) * 2;
        sramAllocator = originalSramAllocator;
        allocationResults =
            FitsInSram(sramAllocator, capabilities, inputTile, weightTile, outputTile, inputStaticAndOffset);
        if (!allocationResults.m_Success)
        {
            return false;
        }
    }
    else
    {
        // Weight tensor must be HWIO or HWIM
        assert(false);
    }

    tensorConfig.inputAllocation.stripeShape   = inputStripe;
    tensorConfig.inputAllocation.tileSize      = inputTile;
    tensorConfig.outputAllocation.stripeShape  = outputStripe;
    tensorConfig.outputAllocation.tileSize     = outputTile;
    tensorConfig.weightsAllocation.stripeShape = weightStripe;
    tensorConfig.weightsAllocation.tileSize    = weightTile;
    tensorConfig.blockWidth                    = blockConfig.m_BlockWidth();
    tensorConfig.blockHeight                   = blockConfig.m_BlockHeight();
    tensorConfig.strategy                      = Strategy::STRATEGY_7;

    originalSramAllocator = sramAllocator;
    FillTensorConfigOffsets(allocationResults, tensorConfig);

    return true;
}

bool StrategyFc::TrySetup(TensorConfig& tensorConfig,
                          SramAllocator& originalSramAllocator,
                          const TensorShape& inputShape,
                          const TensorShape& outputShape,
                          const DataFormat weightsFormat,
                          const TensorShape& weightsShape,
                          const ethosn::command_stream::BlockConfig& blockConfig,
                          const HardwareCapabilities& capabilities,
                          const ShapeMultiplier&,
                          std::pair<bool, uint32_t> inputStaticAndOffset,
                          CompilerMceAlgorithm,
                          uint32_t)
{
    using namespace ethosn::command_stream;
    using namespace ethosn;

    if (weightsFormat != DataFormat::HWIO)
    {
        return false;
    }

    // The minimum stripe depth depends on the number of compute engines and how
    // many OFMs each CE can output
    const uint32_t stripeSize = RoundUpToNearestMultiple(std::min(outputShape[3], capabilities.GetNumberOfOfm()),
                                                         capabilities.GetNumberOfSrams());

    uint32_t inputW     = inputShape[2];
    uint32_t inputH     = inputShape[1];
    uint32_t inputDepth = inputShape[3];

    // clang-format off
    TensorShape inputStripe = { inputShape[0],
                                utils::RoundUpToNearestMultiple(inputW,     capabilities.GetBrickGroupShape()[1]),
                                utils::RoundUpToNearestMultiple(inputH,     capabilities.GetBrickGroupShape()[2]),
                                utils::RoundUpToNearestMultiple(inputDepth, capabilities.GetNumberOfSrams()) };

    uint32_t inputTile = inputShape[0] *
                         utils::RoundUpToNearestMultiple(inputW,      capabilities.GetBrickGroupShape()[1]) *
                         utils::RoundUpToNearestMultiple(inputH,     capabilities.GetBrickGroupShape()[2]) *
                         utils::RoundUpToNearestMultiple(inputDepth, capabilities.GetNumberOfSrams()) ;

    TensorShape outputStripe = { outputShape[0],
                                 utils::RoundUpToNearestMultiple(outputShape[1], capabilities.GetBrickGroupShape()[1]),
                                 utils::RoundUpToNearestMultiple(outputShape[2], capabilities.GetBrickGroupShape()[2]),
                                 stripeSize };

    // The OFM and weight tiles are double buffered, allowing the CEs to work on
    // one stripe at the same time as the MCE loads new weights and outputs finished OFMs
    uint32_t outputTile = outputShape[0] *
                          utils::RoundUpToNearestMultiple(outputShape[1], capabilities.GetBrickGroupShape()[1]) *
                          utils::RoundUpToNearestMultiple(outputShape[2], capabilities.GetBrickGroupShape()[2]) *
                          stripeSize * 2 ;

    // dim[2] of weight tensor = input length
    // it is round to multipe of 1024
    uint32_t inputLength = RoundUpToNearestMultiple(weightsShape[2], 1024);

    // initialise both weight stripe and tile.
    TensorShape weightStripe = { weightsShape[0],
                                 weightsShape[1],
                                 inputLength,
                                 stripeSize };

    // clang-format on

    bool isHwim = weightsFormat == DataFormat::HWIM;

    // compute weight size including header
    uint32_t weightTile = EstimateWeightSizeBytes(weightStripe, capabilities, isHwim) * 2;

    if (inputTile >= (capabilities.GetTotalSramSize() / 2))
    {
        // The strategy only works if the input tile size is less than half of the total
        // SRAM size.
        return false;
    }

    SramAllocator sramAllocator = originalSramAllocator;
    // weight stripe tensor is adjusted so that input+weight+output tiles fit into SRAM.
    AllocationResult allocationResults =
        FitsInSram(sramAllocator, capabilities, inputTile, weightTile, outputTile, inputStaticAndOffset);
    while (allocationResults.m_Success == false)
    {
        // weight length per stripe is halved then re-aligned to multiple of 1024
        inputLength /= 2;
        inputLength = RoundUpToNearestMultiple(inputLength, 1024);

        // update stripe and tile tensors
        weightStripe[2] = inputLength;

        // recalculate the weight size
        weightTile                  = EstimateWeightSizeBytes(weightStripe, capabilities, isHwim) * 2;
        SramAllocator sramAllocator = originalSramAllocator;
        allocationResults =
            FitsInSram(sramAllocator, capabilities, inputTile, weightTile, outputTile, inputStaticAndOffset);
    }

    tensorConfig.inputAllocation.stripeShape   = inputStripe;
    tensorConfig.inputAllocation.tileSize      = inputTile;
    tensorConfig.outputAllocation.stripeShape  = outputStripe;
    tensorConfig.outputAllocation.tileSize     = outputTile;
    tensorConfig.weightsAllocation.stripeShape = weightStripe;
    tensorConfig.weightsAllocation.tileSize    = weightTile;
    tensorConfig.blockWidth                    = blockConfig.m_BlockWidth();
    tensorConfig.blockHeight                   = blockConfig.m_BlockHeight();
    tensorConfig.strategy                      = Strategy::STRATEGY_FC;
    originalSramAllocator                      = sramAllocator;
    FillTensorConfigOffsets(allocationResults, tensorConfig);
    return true;
}

const char* Strategy0::GetStrategyString()
{
    return "Strategy 0";
}

const char* Strategy1::GetStrategyString()
{
    return "Strategy 1";
}

const char* Strategy3::GetStrategyString()
{
    return "Strategy 3";
}

const char* Strategy4::GetStrategyString()
{
    return "Strategy 4";
}

const char* Strategy6::GetStrategyString()
{
    return "Strategy 6";
}

const char* Strategy7::GetStrategyString()
{
    return "Strategy 7";
}

const char* StrategyFc::GetStrategyString()
{
    return "Strategy Fc";
}

}    // namespace support_library
}    // namespace ethosn
