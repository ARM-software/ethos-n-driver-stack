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

InputStats GetInputStatsLegacy(const HardwareCapabilities& caps,
                               const TensorShape& shape,
                               const TensorShape& stripeShape,
                               const Location location,
                               const uint32_t tileSize,
                               const TensorInfo& weights,
                               const uint32_t numOutStripesC)
{
    InputStats data;

    if (location != Location::Sram)
    {
        const TensorShape stripeShapeValid = {
            std::min(stripeShape[0], shape[0]),
            std::min(stripeShape[1], shape[1]),
            std::min(stripeShape[2], shape[2]),
            std::min(stripeShape[3], shape[3]),
        };
        const uint32_t stripeSize = stripeShape[0] * stripeShape[1] * stripeShape[2] * stripeShape[3];
        assert(stripeSize != 0U);

        const uint32_t numStripesH = utils::GetNumStripesH(shape, stripeShape);
        const uint32_t numStripesW = utils::GetNumStripesW(shape, stripeShape);
        const uint32_t numStripesC = utils::GetNumStripesC(shape, stripeShape);

        const bool needNeighbourStripeH = weights.m_Dimensions[0] > 1U;
        const bool needNeighbourStripeW = weights.m_Dimensions[1] > 1U;

        // Number of ofm produced per iteration
        const uint32_t ofmProduced = caps.GetOgsPerEngine() * caps.GetNumberOfEngines();

        // This might change, it doesn't always need all the boundary slots.
        const uint32_t numBoundarySlots = caps.GetNumBoundarySlots();

        const bool isStreamingH = numStripesH > 1U;
        const bool isStreamingW = numStripesW > 1U;
        const bool isStreamingC = numStripesC > 1U;

        data.m_StripesStats.m_NumReloads =
            GetInputNumReloads(isStreamingH, isStreamingW, isStreamingC, weights, ofmProduced, numOutStripesC);

        // Calculate the total amount of input data to be transferred included reloading.
        const uint32_t total =
            GetInputTotalBytes(caps, shape, stripeShape, isStreamingH, isStreamingW, isStreamingC, needNeighbourStripeH,
                               needNeighbourStripeW, data.m_StripesStats.m_NumReloads);

        // Calculate the minimum amount of data required to start processing.
        uint32_t borderWidth  = 0;
        uint32_t borderHeight = 0;

        if (needNeighbourStripeH && isStreamingH)
        {
            borderHeight = (isStreamingC || isStreamingW) ? caps.GetBoundaryStripeHeight() : stripeShapeValid[1];
        }

        if (needNeighbourStripeW && isStreamingW)
        {
            borderWidth = stripeShapeValid[2];
        }

        const bool isUsingBoundarySlots = needNeighbourStripeH && isStreamingH && isStreamingW && !isStreamingC;
        const uint32_t boundarySize     = isUsingBoundarySlots ? borderHeight * stripeShape[2] * stripeShape[3] : 0;
        const uint32_t numStripesInTile = utils::DivRoundUp(tileSize - (boundarySize * numBoundarySlots), stripeSize);

        data.m_MemoryStats.m_DramNonParallel =
            (stripeShapeValid[1] + borderHeight) * (stripeShapeValid[2] + borderWidth) * stripeShapeValid[3];
        // Determine how much data can be transferred in parallel.
        const uint32_t minNumSlotsForBuffering =
            GetInputMinNumSlotsForBuffering(isStreamingH, isStreamingW, isStreamingC, needNeighbourStripeH,
                                            needNeighbourStripeW, numStripesH, numStripesW);

        const bool buffering = numStripesInTile >= minNumSlotsForBuffering;

        if (buffering)
        {
            data.m_MemoryStats.m_DramParallel = total - data.m_MemoryStats.m_DramNonParallel;
        }
        else
        {
            data.m_MemoryStats.m_DramNonParallel = total;
        }

        data.m_StripesStats.m_NumCentralStripes  = utils::GetNumStripesTotal(shape, stripeShape);
        data.m_StripesStats.m_NumBoundaryStripes = isUsingBoundarySlots ? (numStripesH - 1) * numStripesW : 0;
    }
    else
    {
        data.m_MemoryStats.m_Sram = shape[0] * shape[1] * shape[2] * shape[3];
    }

    return data;
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

        // Calculate the total amount of input data to be transferred, included reloading and any packed boundary data.
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

        const bool boundaryStripesNeeded =
            ((weightsShape[0] > 1 && ifmBuffer.m_StripeShape[1] < ifmBuffer.m_TensorShape[1]) ||
             (weightsShape[1] > 1 && ifmBuffer.m_StripeShape[2] < ifmBuffer.m_TensorShape[2]));
        // Calculate the minimum amount of data required to start processing.
        // This is a conservative approximation (i.e. an overestimate).
        // For example we assume that the stripes needed are non-partial.
        const uint32_t numStripesNeededToStartProcessing = boundaryStripesNeeded ? 2 : 1;
        const uint32_t bytesNeededToStartProcessing =
            std::min(numStripesNeededToStartProcessing * ifmBuffer.m_SlotSizeInBytes, total);

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

OutputStats GetOutputStatsLegacy(const TensorShape& shape, const TensorShape& stripeShape, const Location location)
{
    OutputStats data;

    const TensorShape& stripeShapeValid = { std::min(stripeShape[0], shape[0]), std::min(stripeShape[1], shape[1]),
                                            std::min(stripeShape[2], shape[2]), std::min(stripeShape[3], shape[3]) };
    const uint32_t stripeSize = stripeShapeValid[0] * stripeShapeValid[1] * stripeShapeValid[2] * stripeShapeValid[3];

    // Total amount of data.
    const uint32_t total = shape[0] * shape[1] * shape[2] * shape[3];

    // Consider the output data transfer only if it is not already in Sram.
    if (location != Location::Sram)
    {
        // Wait to the final stripe to be copied out if required.
        data.m_MemoryStats.m_DramNonParallel    = stripeSize;
        data.m_MemoryStats.m_DramParallel       = total - data.m_MemoryStats.m_DramNonParallel;
        data.m_StripesStats.m_NumCentralStripes = utils::GetNumStripesTotal(shape, stripeShape);
    }
    else
    {
        data.m_MemoryStats.m_Sram = total;
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

PleStats GetPleStats(const HardwareCapabilities& caps,
                     const std::vector<TensorShape>& inputShapes,
                     const command_stream::PleOperation& pleOperation)
{
    using namespace utils;

    PleStats pleststs;

    // Number of patches that need to be post processed by the Ple kernel
    uint32_t patchesH = 0;
    uint32_t patchesW = 0;
    uint32_t patchesC = 0;

    for (auto& inputShape : inputShapes)
    {
        patchesH = std::max(utils::DivRoundUp(GetHeight(inputShape), GetHeight(g_PatchShape)), patchesH);
        patchesW = std::max(utils::DivRoundUp(GetWidth(inputShape), GetWidth(g_PatchShape)), patchesW);
        patchesC =
            std::max(utils::DivRoundUp(GetChannels(inputShape), caps.GetNumberOfEngines() * caps.GetNumberOfPleLanes()),
                     patchesC);
    }

    pleststs.m_NumOfPatches = patchesW * patchesH * patchesC;
    pleststs.m_Operation    = static_cast<uint32_t>(pleOperation);
    return pleststs;
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

double CalculateMetric(const NetworkPerformanceData& networkPerfData)
{
    double totalMetric = 0;
    for (PassPerformanceData passPerfData : networkPerfData.m_Stream)
    {
        double metric = CalculateMetric(passPerfData);
        totalMetric += metric;
    }
    return totalMetric;
}

double CalculateMetric(const PassPerformanceData& passPerfData)
{
    uint64_t nonParallelBytes = passPerfData.m_Stats.m_Input.m_MemoryStats.m_DramNonParallel +
                                passPerfData.m_Stats.m_Output.m_MemoryStats.m_DramNonParallel +
                                passPerfData.m_Stats.m_Weights.m_MemoryStats.m_DramNonParallel;
    double nonParallelBytesDouble = static_cast<double>(nonParallelBytes);

    uint64_t parallelBytes = passPerfData.m_Stats.m_Input.m_MemoryStats.m_DramParallel +
                             passPerfData.m_Stats.m_Output.m_MemoryStats.m_DramParallel +
                             passPerfData.m_Stats.m_Weights.m_MemoryStats.m_DramParallel;
    double parallelBytesDouble = static_cast<double>(parallelBytes);

    uint64_t mceCycleCount     = passPerfData.m_Stats.m_Mce.m_CycleCount;
    double mceCycleCountDouble = static_cast<double>(mceCycleCount);

    // Rough approximation for the number of stripes in a pass. This isn't measuring any exact number,
    // as the number of stripes may be different for the MCE, PLE, DMA etc., just a rough idea.
    uint32_t numStripes              = std::max({ passPerfData.m_Stats.m_Input.m_StripesStats.m_NumCentralStripes *
                                         (passPerfData.m_Stats.m_Input.m_StripesStats.m_NumReloads + 1),
                                     passPerfData.m_Stats.m_Weights.m_StripesStats.m_NumCentralStripes *
                                         (passPerfData.m_Stats.m_Weights.m_StripesStats.m_NumReloads + 1),
                                     passPerfData.m_Stats.m_Output.m_StripesStats.m_NumCentralStripes *
                                         (passPerfData.m_Stats.m_Output.m_StripesStats.m_NumReloads + 1) });
    double nonparallelOverheadCycles = 0 * numStripes;
    // This overhead was measured approximately from some profiling traces.
    double parallelOverheadCycles = 10000 * numStripes;

    constexpr double dramBandwidth  = 12000000000;    // bytes/second
    constexpr double clockFrequency = 1250000000;     // cycles/second
    constexpr double bytesPerCycle  = dramBandwidth / clockFrequency;

    // Non-buffered, multi-stripe DMA transfers can prevent the MCE from executing in parallel with buffered
    // DMA transfers when the MCE is waiting on DMA transfers already, as the MCE and non-buffered transfer
    // will end up waiting on each other, as they are unable to use the tile at the same time.
    // e.g. Non-buffered IFM stripe cannot load while the MCE is using the tile slot and vice versa.
    auto IsDmaBlocking = [](const InputStats& stats) {
        return (stats.m_MemoryStats.m_DramNonParallel > 0) && (stats.m_StripesStats.m_NumCentralStripes > 1) &&
               (stats.m_MemoryStats.m_DramParallel == 0);
    };

    const bool blockingDmaTransfers = IsDmaBlocking(passPerfData.m_Stats.m_Input) ||
                                      IsDmaBlocking(passPerfData.m_Stats.m_Output) ||
                                      IsDmaBlocking(passPerfData.m_Stats.m_Weights);

    const bool dmaBlockingMce =
        ((parallelBytesDouble / bytesPerCycle) > std::max({ mceCycleCountDouble, parallelOverheadCycles })) &&
        blockingDmaTransfers;

    double metric = (nonParallelBytesDouble / bytesPerCycle) + (dmaBlockingMce ? mceCycleCountDouble : 0) +
                    std::max({ parallelBytesDouble / bytesPerCycle, dmaBlockingMce ? 0 : mceCycleCountDouble,
                               parallelOverheadCycles }) +
                    nonparallelOverheadCycles;
    return metric;
}

}    // namespace support_library
}    // namespace ethosn
