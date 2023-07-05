//
// Copyright © 2020-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "EstimationUtils.hpp"

#include "Plan.hpp"

#include <numeric>

namespace ethosn
{
namespace support_library
{

constexpr uint32_t GetMinNumSlots(bool needNeighbour, uint32_t numStripes)
{
    return std::min(needNeighbour ? 3U : 1U, numStripes);
}

constexpr uint32_t GetEffectiveSize(uint32_t size, uint32_t stripeSize, uint32_t borderBefore, uint32_t borderAfter)
{
    return size + (borderBefore + borderAfter) * ((size - 1U) / stripeSize);
}

uint32_t GetInputMinNumSlotsForBuffering(const bool isStreamingH,
                                         const bool isStreamingW,
                                         const bool isStreamingC,
                                         const bool needNeighbourStripeH,
                                         const bool needNeighbourStripeW,
                                         const uint32_t numStripesH,
                                         const uint32_t numStripesW)
{
    if (isStreamingC)
    {
        return 2U * GetMinNumSlots(needNeighbourStripeH, numStripesH) *
               GetMinNumSlots(needNeighbourStripeW, numStripesW);
    }
    else if (isStreamingW)
    {
        return GetMinNumSlots(needNeighbourStripeW, numStripesW) + 1U;
    }
    else if (isStreamingH)
    {
        return GetMinNumSlots(needNeighbourStripeH, numStripesH) + 1U;
    }
    return 1U;
}

uint32_t GetInputNumReloads(const bool isStreamingH,
                            const bool isStreamingW,
                            const bool isStreamingC,
                            const TensorInfo& weights,
                            const uint32_t ofmProduced,
                            const uint32_t numOutStripesC)
{
    assert(numOutStripesC > 0);

    if (isStreamingC)
    {
        // Round up the number of output channels (HWIO) or the channel multiplier (HWIM, where M=1).
        return utils::DivRoundUp(weights.m_Dimensions[3], ofmProduced) - 1U;
    }
    else if (isStreamingH || isStreamingW)
    {
        return weights.m_DataFormat == DataFormat::HWIM ? 0 : numOutStripesC - 1U;
    }

    return 0;
}

uint32_t GetInputTotalBytes(const HardwareCapabilities& caps,
                            const TensorShape& shape,
                            const TensorShape& stripeShape,
                            const bool isStreamingH,
                            const bool isStreamingW,
                            const bool isStreamingC,
                            const bool needNeighbourStripeH,
                            const bool needNeighbourStripeW,
                            const uint32_t reloads)
{
    uint32_t borderHeight = 0;
    uint32_t borderWidth  = 0;

    // Calculate the total amount of input data to be transferred included reloading.
    if (needNeighbourStripeW && isStreamingC)
    {
        borderWidth = stripeShape[2];
    }

    if (needNeighbourStripeH && (isStreamingC || (isStreamingH && isStreamingW)))
    {
        borderHeight = caps.GetBoundaryStripeHeight();
    }

    const uint32_t effectiveHeight = GetEffectiveSize(shape[1], stripeShape[1], borderHeight, borderHeight);
    const uint32_t effectiveWidth  = GetEffectiveSize(shape[2], stripeShape[2], borderWidth, borderWidth);

    // Total amount of data
    return (reloads + 1U) * shape[0] * effectiveHeight * effectiveWidth * shape[3];
}

InputStats GetInputStatsCascading(const SramBuffer& ifmBuffer,
                                  const TensorShape& weightsShape,
                                  utils::Optional<CascadingBufferFormat> dramBufferFormat)
{
    InputStats data;

    if (dramBufferFormat.has_value())
    {
        const uint32_t numStripes        = utils::GetNumStripesTotal(ifmBuffer.m_TensorShape, ifmBuffer.m_StripeShape);
        data.m_StripesStats.m_NumReloads = ifmBuffer.m_NumLoads - 1;

        // Calculate the total amount of input data to be transferred, including reloading and any packed boundary data.
        // Note that a simpler calculation of numStripes * m_SlotSizeInBytes is not accurate in cases where there
        // are partial stripes (in any of the three dimensions), because the slot size will be for the full stripe
        // shape and so this would overestimate.
        uint32_t effectiveHeight =
            GetEffectiveSize(ifmBuffer.m_TensorShape[1], ifmBuffer.m_StripeShape[1],
                             ifmBuffer.m_PackedBoundaryThickness.top, ifmBuffer.m_PackedBoundaryThickness.bottom);
        uint32_t effectiveWidth =
            GetEffectiveSize(ifmBuffer.m_TensorShape[2], ifmBuffer.m_StripeShape[2],
                             ifmBuffer.m_PackedBoundaryThickness.left, ifmBuffer.m_PackedBoundaryThickness.right);
        if (dramBufferFormat != CascadingBufferFormat::NHWC)
        {
            effectiveHeight = utils::RoundUpToNearestMultiple(effectiveHeight, 8);
            effectiveWidth  = utils::RoundUpToNearestMultiple(effectiveWidth, 8);
        }
        const uint32_t total = ifmBuffer.m_NumLoads * ifmBuffer.m_TensorShape[0] * effectiveHeight * effectiveWidth *
                               ifmBuffer.m_TensorShape[3];

        // Calculate the amount of input data to be transferred for a single stripe, including any packed boundary data.
        // Note that this is subtly different to m_SlotSizeInBytes because that is the amount of SRAM needed
        // to store the data, not the amount of data actually transferred. These can be different in cases of
        // partial stripes (in any of the three dimensions), because the slot size will be for the full stripe
        // shape and so this would overestimate.
        uint32_t effectiveStripeHeight =
            std::min(utils::GetHeight(ifmBuffer.m_TensorShape), utils::GetHeight(ifmBuffer.m_StripeShape));
        uint32_t effectiveStripeWidth =
            std::min(utils::GetWidth(ifmBuffer.m_TensorShape), utils::GetWidth(ifmBuffer.m_StripeShape));
        uint32_t effectiveStripeChannels =
            std::min(utils::GetChannels(ifmBuffer.m_TensorShape), utils::GetChannels(ifmBuffer.m_StripeShape));
        if (dramBufferFormat != CascadingBufferFormat::NHWC)
        {
            effectiveStripeHeight = utils::RoundUpToNearestMultiple(effectiveStripeHeight, 8);
            effectiveStripeWidth  = utils::RoundUpToNearestMultiple(effectiveStripeWidth, 8);
        }
        const uint32_t stripeBytes = effectiveStripeHeight * effectiveStripeWidth * effectiveStripeChannels;

        const bool boundaryStripesNeeded =
            ((weightsShape[0] > 1 && ifmBuffer.m_StripeShape[1] < ifmBuffer.m_TensorShape[1]) ||
             (weightsShape[1] > 1 && ifmBuffer.m_StripeShape[2] < ifmBuffer.m_TensorShape[2]));
        // Calculate the minimum amount of data required to start processing.
        // This is a conservative approximation (i.e. an overestimate).
        // For example we assume that the stripes needed are non-partial.
        const uint32_t numStripesNeededToStartProcessing = boundaryStripesNeeded ? 2 : 1;
        const uint32_t bytesNeededToStartProcessing = std::min(numStripesNeededToStartProcessing * stripeBytes, total);

        // Determine how much data can be transferred in parallel.
        const uint32_t numStripesNeededPerOfmStripe = boundaryStripesNeeded ? 3 : 1;
        const uint32_t minNumSlotsForBuffering      = numStripesNeededPerOfmStripe + 1;

        const bool buffering = ifmBuffer.m_NumStripes >= minNumSlotsForBuffering;

        if (buffering)
        {
            data.m_MemoryStats.m_DramNonParallel = bytesNeededToStartProcessing;
            data.m_MemoryStats.m_DramParallel    = total - bytesNeededToStartProcessing;
        }
        else
        {
            data.m_MemoryStats.m_DramNonParallel = total;
            data.m_MemoryStats.m_DramParallel    = 0;
        }

        data.m_StripesStats.m_NumCentralStripes  = numStripes;
        data.m_StripesStats.m_NumBoundaryStripes = 0;
    }
    else
    {
        data.m_MemoryStats.m_Sram = ifmBuffer.m_TensorShape[0] * ifmBuffer.m_TensorShape[1] *
                                    ifmBuffer.m_TensorShape[2] * ifmBuffer.m_TensorShape[3];
    }

    return data;
}

OutputStats GetOutputStatsCascading(const SramBuffer& ofmSramBuffer,
                                    utils::Optional<CascadingBufferFormat> dramBufferFormat)
{
    OutputStats data;

    TensorShape shape              = ofmSramBuffer.m_TensorShape;
    const TensorShape& stripeShape = ofmSramBuffer.m_StripeShape;

    if (dramBufferFormat.has_value() && dramBufferFormat != CascadingBufferFormat::NHWC)
    {
        shape = utils::RoundUpHeightAndWidthToBrickGroup(shape);
    }

    const TensorShape& stripeShapeValid = { std::min(stripeShape[0], shape[0]), std::min(stripeShape[1], shape[1]),
                                            std::min(stripeShape[2], shape[2]), std::min(stripeShape[3], shape[3]) };
    const uint32_t stripeSize = stripeShapeValid[0] * stripeShapeValid[1] * stripeShapeValid[2] * stripeShapeValid[3];

    // Total amount of data.
    const uint32_t total = shape[0] * shape[1] * shape[2] * shape[3];

    // Consider the output data transfer only if it is not already in Sram.
    if (dramBufferFormat.has_value())
    {
        const bool buffering = ofmSramBuffer.m_NumStripes >= 2;
        if (buffering)
        {
            data.m_MemoryStats.m_DramNonParallel = stripeSize;
            data.m_MemoryStats.m_DramParallel    = total - data.m_MemoryStats.m_DramNonParallel;
        }
        else
        {
            data.m_MemoryStats.m_DramNonParallel = total;
            data.m_MemoryStats.m_DramParallel    = 0;
        }

        data.m_StripesStats.m_NumCentralStripes = utils::GetNumStripesTotal(shape, stripeShape);
    }
    else
    {
        data.m_MemoryStats.m_Sram = total;
    }
    return data;
}

namespace
{

uint32_t GetPleCyclesPerPatch(command_stream::PleOperation op)
{
    // These numbers were estimated from some internal benchmarks running on the model.
    switch (op)
    {
        case command_stream::PleOperation::ADDITION:
            return 15;
        case command_stream::PleOperation::ADDITION_RESCALE:
            return 35;
        case command_stream::PleOperation::AVGPOOL_3X3_1_1_UDMA:
            return 97;
        case command_stream::PleOperation::DOWNSAMPLE_2X2:
            return 10;
        case command_stream::PleOperation::INTERLEAVE_2X2_2_2:
            return 13;
        case command_stream::PleOperation::LEAKY_RELU:
            return 37;
        case command_stream::PleOperation::MAXPOOL_2X2_2_2:
            return 13;
        case command_stream::PleOperation::MAXPOOL_3X3_2_2_EVEN:    // intentional fallthrough
        case command_stream::PleOperation::MAXPOOL_3X3_2_2_ODD:
            return 37;
        case command_stream::PleOperation::MEAN_XY_7X7:    // intentional fallthrough
        case command_stream::PleOperation::MEAN_XY_8X8:
            return 37;
        case command_stream::PleOperation::PASSTHROUGH:
            return 6;
        case command_stream::PleOperation::SIGMOID:
            return 76;
        case command_stream::PleOperation::TRANSPOSE_XY:
            return 14;
        default:
            return 0;
    }
}

}    // namespace

PleStats GetPleStats(const HardwareCapabilities& caps,
                     const std::vector<TensorShape>& inputShapes,
                     const TensorShape& outputShape,
                     const command_stream::PleOperation& pleOperation,
                     uint32_t blockMultiplier,
                     const ethosn::command_stream::BlockConfig& blockConfig)
{
    using namespace utils;

    PleStats pleStats;

    // Number of patches that need to be post processed by the Ple kernel
    uint32_t patchesH = 0;
    uint32_t patchesW = 0;
    uint32_t patchesC = 0;

    for (auto& inputShape : inputShapes)
    {
        uint32_t effectiveHeight = GetHeight(inputShape);
        uint32_t effectiveWidth  = GetWidth(inputShape);

        // Note that we round up to the block config, because the PLE always processes an entire block,
        // even if it is only partial.
        // Standalone operations (e.g. Addition) don't use a block config.
        if (blockConfig.m_BlockWidth() != 0 && blockConfig.m_BlockHeight() != 0)
        {
            effectiveHeight = utils::RoundUpToNearestMultiple(GetHeight(inputShape), blockConfig.m_BlockHeight());
            effectiveWidth  = utils::RoundUpToNearestMultiple(GetWidth(inputShape), blockConfig.m_BlockWidth());
        }

        patchesH = std::max(utils::DivRoundUp(effectiveHeight, GetHeight(g_PatchShape)), patchesH);
        patchesW = std::max(utils::DivRoundUp(effectiveWidth, GetWidth(g_PatchShape)), patchesW);

        patchesC =
            std::max(utils::DivRoundUp(GetChannels(inputShape), caps.GetNumberOfEngines() * caps.GetNumberOfPleLanes()),
                     patchesC);
    }

    pleStats.m_NumOfPatches = patchesW * patchesH * patchesC;
    pleStats.m_Operation    = static_cast<uint32_t>(pleOperation);

    // Standalone operations (e.g. Addition) don't use a block config.
    uint64_t blockOverhead = 0;
    if (blockConfig.m_BlockWidth() != 0 && blockConfig.m_BlockHeight() != 0)
    {
        uint64_t numBlocks =
            static_cast<uint64_t>(utils::DivRoundUp(utils::GetHeight(outputShape), blockConfig.m_BlockHeight())) *
            static_cast<uint64_t>(utils::DivRoundUp(utils::GetWidth(outputShape), blockConfig.m_BlockWidth())) *
            patchesC;
        uint64_t numMultipliedBlocks = numBlocks / blockMultiplier;

        constexpr uint32_t overheadPerBlock           = 10;
        constexpr uint32_t overheadPerMultipliedBlock = 100;
        blockOverhead = overheadPerBlock * numBlocks + overheadPerMultipliedBlock * numMultipliedBlocks;
    }

    pleStats.m_CycleCount = pleStats.m_NumOfPatches * GetPleCyclesPerPatch(pleOperation) + blockOverhead;

    return pleStats;
}

PassStats GetConversionStats(const ConversionData& input, const ConversionData& output, bool isDramToDram)
{
    PassStats perfData;

    const TensorShape& inputShape           = input.tensorShape;
    const TensorShape& roundedUpInputShape  = utils::RoundUpHeightAndWidthToBrickGroup(inputShape);
    const TensorShape& outputShape          = output.tensorShape;
    const TensorShape& roundedUpOutputShape = utils::RoundUpHeightAndWidthToBrickGroup(outputShape);

    const bool isInputNHWC  = input.isNhwc;
    const bool isOutputNHWC = output.isNhwc;

    const uint32_t inputSize  = inputShape[0] * inputShape[1] * inputShape[2] * inputShape[3];
    const uint32_t outputSize = outputShape[0] * outputShape[1] * outputShape[2] * outputShape[3];

    const uint32_t roundedUpInputSize =
        roundedUpInputShape[0] * roundedUpInputShape[1] * roundedUpInputShape[2] * roundedUpInputShape[3];
    const uint32_t roundedUpOutputSize =
        roundedUpOutputShape[0] * roundedUpOutputShape[1] * roundedUpOutputShape[2] * roundedUpOutputShape[3];

    if (isDramToDram)
    {
        perfData.m_Input.m_MemoryStats.m_DramNonParallel    = isInputNHWC ? inputSize : roundedUpInputSize;
        perfData.m_Input.m_StripesStats.m_NumCentralStripes = utils::GetNumStripesTotal(inputShape, input.stripeShape);

        perfData.m_Output.m_MemoryStats.m_DramNonParallel = isOutputNHWC ? outputSize : roundedUpOutputSize;
        perfData.m_Output.m_StripesStats.m_NumCentralStripes =
            utils::GetNumStripesTotal(outputShape, output.stripeShape);
    }
    else
    {
        // This is for Sram To Sram conversions. We only handle Dram To Dram or Sram to Sram.
        perfData.m_Input.m_MemoryStats.m_Sram  = roundedUpInputSize;
        perfData.m_Output.m_MemoryStats.m_Sram = roundedUpOutputSize;
    }
    return perfData;
}

InputStats AccountForActivationCompression(InputStats stats, float spaceSavingRatio)
{
    InputStats ret = stats;
    ret.m_MemoryStats.m_DramNonParallel =
        static_cast<uint32_t>(static_cast<float>(stats.m_MemoryStats.m_DramNonParallel) * (1 - spaceSavingRatio));
    ret.m_MemoryStats.m_DramParallel =
        static_cast<uint32_t>(static_cast<float>(stats.m_MemoryStats.m_DramParallel) * (1 - spaceSavingRatio));
    return ret;
}

StripesStats AccountForDmaChunking(StripesStats stats,
                                   const SramBuffer& sramBuffer,
                                   const DramBuffer& dramBuffer,
                                   bool dramStridingAllowed)
{
    using namespace utils;

    StripesStats result = stats;

    if (dramBuffer.m_Format == CascadingBufferFormat::NHWCB)
    {
        const uint32_t brickGroupWidth    = utils::GetWidth(g_BrickGroupShape);
        const uint32_t brickGroupHeight   = utils::GetHeight(g_BrickGroupShape);
        const uint32_t brickGroupChannels = utils::GetChannels(g_BrickGroupShape);

        const TensorShape& stripeSize             = sramBuffer.m_StripeShape;
        const TensorShape& supertensorSizeInCells = {
            1,
            DivRoundUp(GetHeight(dramBuffer.m_TensorShape), brickGroupHeight),
            DivRoundUp(GetWidth(dramBuffer.m_TensorShape), brickGroupWidth),
            DivRoundUp(GetChannels(dramBuffer.m_TensorShape), brickGroupChannels),
        };

        // Consistent non-zero DRAM stride needed for output streaming to use DRAM striding
        const bool canDramStride = dramStridingAllowed &&
                                   utils::DivRoundUp(GetChannels(stripeSize), brickGroupChannels) == 1U &&
                                   GetChannels(supertensorSizeInCells) > 1;

        uint32_t numChunksH = 1;
        uint32_t numChunksW = 1;
        uint32_t numChunksC = 1;

        const bool partialDepth =
            utils::DivRoundUp(GetChannels(stripeSize), brickGroupChannels) < GetChannels(supertensorSizeInCells);
        const bool partialWidth =
            utils::DivRoundUp(GetWidth(stripeSize), brickGroupWidth) < GetWidth(supertensorSizeInCells);

        // Input NHWCB cannot DRAM stride, output NHWCB can only dram stride with stripes
        // one brick group in depth

        // DRAM striding can be used for as much of the stripe that has a consistent stride
        // i.e. can cover the full stripe if it is full width, or each row if it is partial

        // Stride between X chunks if partial depth
        if (partialDepth && !canDramStride)
        {
            numChunksW = utils::DivRoundUp(GetWidth(stripeSize), brickGroupWidth);
        }

        // Stride between Y chunks if partial width or partial depth
        if ((partialDepth && !canDramStride) || partialWidth)
        {
            numChunksH = utils::DivRoundUp(GetHeight(stripeSize), brickGroupHeight);
        }

        result.m_NumCentralStripes *= (numChunksH * numChunksW * numChunksC);
    }

    return result;
}

double CalculateMetric(const PassStats& legacyPerfData, const PassDesc& passDesc)
{
    ETHOSN_UNUSED(passDesc);

    using namespace utils;

    // Casts to double may result in a loss of precision as doubles cannot represent all the values
    // in a uint64_t, however it is unlikely to occur and is fine anyway as the metric is an estimation.
    uint64_t nonParallelBytes = static_cast<uint64_t>(legacyPerfData.m_Input.m_MemoryStats.m_DramNonParallel) +
                                legacyPerfData.m_Output.m_MemoryStats.m_DramNonParallel +
                                legacyPerfData.m_Weights.m_MemoryStats.m_DramNonParallel;
    double nonParallelBytesDouble = static_cast<double>(nonParallelBytes);

    uint64_t parallelBytes = static_cast<uint64_t>(legacyPerfData.m_Input.m_MemoryStats.m_DramParallel) +
                             legacyPerfData.m_Output.m_MemoryStats.m_DramParallel +
                             legacyPerfData.m_Weights.m_MemoryStats.m_DramParallel;
    double parallelBytesDouble = static_cast<double>(parallelBytes);

    uint64_t mcePleCycleCount     = std::max(legacyPerfData.m_Mce.m_CycleCount, legacyPerfData.m_Ple.m_CycleCount);
    double mcePleCycleCountDouble = static_cast<double>(mcePleCycleCount);

    // Rough approximation for the number of stripes in a pass. This isn't measuring any exact number,
    // as the number of stripes may be different for the MCE, PLE, DMA etc., just a rough idea.
    uint32_t numStripes              = std::max({ legacyPerfData.m_Input.m_StripesStats.m_NumCentralStripes *
                                         (legacyPerfData.m_Input.m_StripesStats.m_NumReloads + 1),
                                     legacyPerfData.m_Weights.m_StripesStats.m_NumCentralStripes *
                                         (legacyPerfData.m_Weights.m_StripesStats.m_NumReloads + 1),
                                     legacyPerfData.m_Output.m_StripesStats.m_NumCentralStripes *
                                         (legacyPerfData.m_Output.m_StripesStats.m_NumReloads + 1) });
    double nonparallelOverheadCycles = 0 * numStripes;
    // This overhead was measured approximately from some profiling traces.
    double parallelOverheadCycles = 10000 * numStripes;

    // How many bytes the DMA can transfer for each cycle of the MCE/PLE.
    constexpr double bytesPerCycle = 16.0;

    // Non-buffered, multi-stripe DMA transfers can prevent the MCE from executing in parallel with buffered
    // DMA transfers when the MCE is waiting on DMA transfers already, as the MCE and non-buffered transfer
    // will end up waiting on each other, as they are unable to use the tile at the same time.
    // e.g. Non-buffered IFM stripe cannot load while the MCE is using the tile slot and vice versa.
    auto IsDmaBlocking = [](const InputStats& stats) {
        return (stats.m_MemoryStats.m_DramNonParallel > 0) && (stats.m_StripesStats.m_NumCentralStripes > 1) &&
               (stats.m_MemoryStats.m_DramParallel == 0);
    };

    const bool blockingDmaTransfers = IsDmaBlocking(legacyPerfData.m_Input) || IsDmaBlocking(legacyPerfData.m_Output) ||
                                      IsDmaBlocking(legacyPerfData.m_Weights);

    const bool dmaBlockingMce =
        ((parallelBytesDouble / bytesPerCycle) > std::max({ mcePleCycleCountDouble, parallelOverheadCycles })) &&
        blockingDmaTransfers;

    double metric = (nonParallelBytesDouble / bytesPerCycle) + (dmaBlockingMce ? mcePleCycleCountDouble : 0) +
                    std::max({ parallelBytesDouble / bytesPerCycle, dmaBlockingMce ? 0 : mcePleCycleCountDouble,
                               parallelOverheadCycles }) +
                    nonparallelOverheadCycles;
    return metric;
}

}    // namespace support_library
}    // namespace ethosn
