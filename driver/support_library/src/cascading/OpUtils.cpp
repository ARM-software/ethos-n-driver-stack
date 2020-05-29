//
// Copyright © 2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

// TODO:: Duplicated code from Pass.cpp Need to be deleted/restructured. Fix in NNXSW-2208
#include "OpUtils.hpp"

#include "Plan.hpp"

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
        borderWidth = caps.GetBrickGroupShape()[2];
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

uint32_t GetWeightsNumReloads(const HardwareCapabilities& caps,
                              const TensorShape& inShape,
                              const TensorShape& inStripeShape,
                              const TensorInfo& info,
                              const uint32_t tileSize)
{
    // The input data streaming affects the number of weights data reloads.
    const uint32_t numStripesH = utils::GetNumStripesH(inShape, inStripeShape);
    const uint32_t numStripesW = utils::GetNumStripesW(inShape, inStripeShape);
    const uint32_t numStripesC = utils::GetNumStripesC(inShape, inStripeShape);

    const uint32_t totalSize =
        utils::EstimateWeightSizeBytes(info.m_Dimensions, caps, info.m_DataFormat == DataFormat::HWIM);

    const bool isStreamingHC = numStripesH > 1U && numStripesW == 1U && numStripesC > 1U;

    // Account for the reloading of the weights data, this happens when streaming input data in depth and height.
    return isStreamingHC && (tileSize < totalSize) ? (numStripesW * numStripesH - 1U) : 0;
}

InputStats GetInputStats(const HardwareCapabilities& caps,
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

        const uint32_t numStripesH = utils::GetNumStripesH(shape, stripeShape);
        const uint32_t numStripesW = utils::GetNumStripesW(shape, stripeShape);
        const uint32_t numStripesC = utils::GetNumStripesC(shape, stripeShape);

        const bool needNeighbourStripeH = weights.m_Dimensions[0] > 1U;
        const bool needNeighbourStripeW = weights.m_Dimensions[1] > 1U;

        // Number of ofm produced per iteration
        const uint32_t ofmProduced = caps.GetOfmPerEngine() * caps.GetNumberOfEngines();

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
            borderWidth = isStreamingC ? caps.GetBrickGroupShape()[2] : stripeShapeValid[2];
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

OutputStats GetOutputStats(const TensorShape& shape, const TensorShape& stripeShape, const BufferLocation location)
{
    OutputStats data;

    const TensorShape& stripeShapeValid = { std::min(stripeShape[0], shape[0]), std::min(stripeShape[1], shape[1]),
                                            std::min(stripeShape[2], shape[2]), std::min(stripeShape[3], shape[3]) };
    const uint32_t stripeSize = stripeShapeValid[0] * stripeShapeValid[1] * stripeShapeValid[2] * stripeShapeValid[3];

    // Total amount of data.
    const uint32_t total = shape[0] * shape[1] * shape[2] * shape[3];

    // Consider the output data transfer only if it is not already in Sram.
    if (location != BufferLocation::Sram)
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

WeightsStats GetWeightsStats(const HardwareCapabilities& caps,
                             EncodedWeights& encodedWeights,
                             const TensorInfo& info,
                             const TensorShape& stripeShape,
                             const uint32_t tileSize,
                             const TensorShape& inShape,
                             const TensorShape& inStripeShape)
{
    WeightsStats data;

    const uint32_t stripeSize =
        utils::EstimateWeightSizeBytes(stripeShape, caps, info.m_DataFormat == DataFormat::HWIM);

    // Account for the reloading of the weights data, this happens when streaming input data in depth and height.
    data.m_StripesStats.m_NumCentralStripes = static_cast<uint32_t>(encodedWeights.m_Metadata.size());
    data.m_StripesStats.m_NumReloads        = GetWeightsNumReloads(caps, inShape, inStripeShape, info, tileSize);

    // Check if there is more than a stripe in the tile.
    const bool buffering = tileSize > stripeSize;

    if (buffering)
    {
        // At least a weights stripe needs to be in internal memory before starting the processing, use the metadata information
        // to get the amount of data.
        data.m_MemoryStats.m_DramNonParallel = encodedWeights.m_Metadata[0].m_Size;
        data.m_MemoryStats.m_DramParallel =
            (data.m_StripesStats.m_NumReloads + 1U) * static_cast<uint32_t>(encodedWeights.m_Data.size()) -
            data.m_MemoryStats.m_DramNonParallel;
    }
    else
    {
        data.m_MemoryStats.m_DramNonParallel =
            (data.m_StripesStats.m_NumReloads + 1U) * static_cast<uint32_t>(encodedWeights.m_Data.size());
    }
    // Clamp the savings to 0
    // if the weights are uncompressable then the encoded weight size is larger than the weights provided
    // because of the header
    data.m_WeightCompressionSavings =
        std::max(0.0f, 1.0f - (static_cast<float>(encodedWeights.m_Data.size()) /
                               static_cast<float>(utils::GetNumElements(info.m_Dimensions))));

    return data;
}

InputStats GetInputStats(const HardwareCapabilities& caps,
                         const Buffer* inpbuf,
                         const Buffer* outbuff,
                         const uint32_t& inputTileSize,
                         const TensorInfo& weightsInfo)
{
    // TODO: Assuming RoundUpHeightAndWidthToBrickGroup. Fix in NNXSW-2208
    assert(inpbuf);
    assert(outbuff);

    // Number of output stripes affects the number of input data reloads for some streaming strategies.
    uint32_t numOutStripeC = utils::DivRoundUp(outbuff->m_TensorShape[3], outbuff->m_StripeShape[3]);
    return GetInputStats(caps, inpbuf->m_TensorShape, inpbuf->m_StripeShape, inpbuf->m_Location, inputTileSize,
                         weightsInfo, numOutStripeC);
}

PleStats GetPleStats(const HardwareCapabilities& caps,
                     const std::vector<TensorShape>& inputStripeShapes,
                     const command_stream::PleOperation& pleOperation)
{
    PleStats pleststs;

    // Number of patches that need to be post processed by the Ple kernel
    uint32_t patchesH = 0;
    uint32_t patchesW = 0;
    uint32_t patchesC = 0;

    for (auto& inputshape : inputStripeShapes)
    {
        patchesH = std::max(utils::DivRoundUp(inputshape[1], caps.GetPatchShape()[1]), patchesH);
        patchesW = std::max(utils::DivRoundUp(inputshape[2], caps.GetPatchShape()[2]), patchesW);
        patchesC = std::max(utils::DivRoundUp(inputshape[3], caps.GetNumberOfEngines()), patchesC);
    }

    pleststs.m_NumOfPatches = patchesW * patchesH * patchesC;
    pleststs.m_Operation    = static_cast<uint32_t>(pleOperation);
    return pleststs;
}

}    // namespace support_library
}    // namespace ethosn
