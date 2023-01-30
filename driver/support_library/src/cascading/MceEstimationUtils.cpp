//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "MceEstimationUtils.hpp"

#include <random>

namespace ethosn
{
namespace support_library
{

uint64_t GetMceCycleCountWinograd(const HardwareCapabilities& caps,
                                  const TensorShape& inputShape,
                                  const TensorShape& outputShape,
                                  const uint32_t weightsHeight,
                                  const uint32_t weightsWidth)
{

    const uint32_t ifmConsumed = caps.GetIgsPerEngine() * caps.GetNumberOfEngines();
    const uint32_t ofmProduced = caps.GetOgsPerEngine() * caps.GetNumberOfEngines();

    // Winograd output block size:
    // 1D 1x3 [WxH]filter -> [WxH] 4x2
    // 1D 3x1 filter -> 2x4
    // 2D 3x3 filter -> 2x2
    WinogradOutputShape winogradOutputShape = caps.Get3x3WinogradOutputSize();

    if (weightsWidth == 1U)
    {
        winogradOutputShape = caps.Get1x3WinogradOutputSize();
    }

    if (weightsHeight == 1U)
    {
        winogradOutputShape = caps.Get3x1WinogradOutputSize();
    }

    uint32_t numIfms = inputShape[3];
    uint32_t numOfms = outputShape[3];

    const uint32_t numTotIfms = utils::RoundUpToNearestMultiple(numIfms, ifmConsumed);

    const uint32_t numWinogradOutputs = utils::DivRoundUp(outputShape[2], winogradOutputShape.m_Width) *
                                        utils::DivRoundUp(outputShape[1], winogradOutputShape.m_Height);

    const uint32_t winogradKernelSize = caps.GetWideKernelSize();
    // Always 16 MACs to process either a 2x4, 4x2 or 2x2 winograd block
    const uint64_t numMacsPerWinogradOutput = static_cast<uint64_t>(caps.GetMacsPerWinogradOutputBlock()) *
                                              utils::DivRoundUp(weightsWidth, winogradKernelSize) *
                                              utils::DivRoundUp(weightsHeight, winogradKernelSize);

    const uint64_t numMacOps       = numWinogradOutputs * numMacsPerWinogradOutput;
    const uint64_t numCyclesPerOfm = (numTotIfms * numMacOps) / (ifmConsumed * caps.GetMacUnitsPerOg());

    return numCyclesPerOfm * utils::DivRoundUp(numOfms, ofmProduced);
}

uint64_t GetMceCycleCountDirect(const HardwareCapabilities& caps,
                                const Stride& stride,
                                const ethosn::command_stream::MceOperation& convtype,
                                const TensorShape& inputShape,
                                const TensorShape& outputShape,
                                const uint32_t weightsHeight,
                                const uint32_t weightsWidth)
{
    const uint32_t numKernelElements = weightsWidth * weightsHeight;
    const uint32_t numEngines        = caps.GetNumberOfEngines();
    const uint32_t numIgsPerEngine   = caps.GetIgsPerEngine();
    const uint32_t numOgsPerEngine   = caps.GetOgsPerEngine();
    const uint32_t numMacUnitsPerOg  = caps.GetMacUnitsPerOg();
    const uint32_t halfPatchHeight   = g_PatchShape[1];
    const uint32_t halfPatchWidth    = utils::DivRoundUp(g_PatchShape[2], 2u);
    uint32_t numActiveOgs;
    uint32_t ifmChannelsPerMacUnit;
    uint32_t ifmChannelsPerOfm;

    if (convtype == ethosn::command_stream::MceOperation::DEPTHWISE_CONVOLUTION)
    {
        numActiveOgs          = numIgsPerEngine * numEngines;
        ifmChannelsPerMacUnit = 1;
        ifmChannelsPerOfm     = 1;
    }
    else
    {
        numActiveOgs          = numOgsPerEngine * numEngines;
        ifmChannelsPerMacUnit = numIgsPerEngine * numEngines;
        ifmChannelsPerOfm     = utils::GetNumOrigChannels(inputShape[3], stride.m_X, stride.m_Y, caps);
    }

    uint32_t h        = utils::RoundUpToNearestMultiple(outputShape[1], halfPatchHeight);
    uint32_t w        = utils::RoundUpToNearestMultiple(outputShape[2], halfPatchWidth);
    uint32_t i        = utils::RoundUpToNearestMultiple(ifmChannelsPerOfm, ifmChannelsPerMacUnit);
    uint32_t o        = utils::RoundUpToNearestMultiple(outputShape[3], numActiveOgs);
    uint64_t macCount = static_cast<uint64_t>(numKernelElements) * h * w * i * o;

    uint32_t macsPerCycle = ifmChannelsPerMacUnit * numMacUnitsPerOg * numActiveOgs;

    return macCount / macsPerCycle;
}

uint64_t GetMceCycleCount(const HardwareCapabilities& caps,
                          const Stride& stride,
                          const ethosn::command_stream::MceOperation& convtype,
                          const CompilerMceAlgorithm& algo,
                          const TensorShape& inputShape,
                          const TensorShape& outputShape,
                          const uint32_t weightsHeight,
                          const uint32_t weightsWidth)
{
    if (algo == CompilerMceAlgorithm::Winograd)
    {
        return GetMceCycleCountWinograd(caps, inputShape, outputShape, weightsHeight, weightsWidth);
    }
    else
    {
        return GetMceCycleCountDirect(caps, stride, convtype, inputShape, outputShape, weightsHeight, weightsWidth);
    }
}

uint64_t GetNumOperations(const HardwareCapabilities& caps,
                          const Stride& stride,
                          const command_stream::MceOperation& convtype,
                          const TensorShape& inputShape,
                          const TensorShape& outputShape,
                          const uint32_t weightsHeight,
                          const uint32_t weightsWidth)
{
    const uint64_t numKernelElements    = static_cast<uint64_t>(weightsWidth) * weightsHeight;
    const uint64_t numOpsPerElement     = 2U * numKernelElements;
    const uint64_t numUninterleavedIfms = utils::GetNumOrigChannels(inputShape[3], stride.m_X, stride.m_Y, caps);
    const uint64_t numOutputElements    = static_cast<uint64_t>(outputShape[1]) * outputShape[2];
    const uint64_t numOpsPerIfmPerOfm   = numOutputElements * numOpsPerElement;

    uint64_t numIfms = 0;
    uint64_t numOfms = 0;
    if (convtype == command_stream::MceOperation::CONVOLUTION)
    {
        numIfms = numUninterleavedIfms;
        numOfms = outputShape[3];
    }
    else if (convtype == command_stream::MceOperation::DEPTHWISE_CONVOLUTION)
    {
        numIfms = 1;
        numOfms = numUninterleavedIfms;
    }
    else if (convtype == command_stream::MceOperation::FULLY_CONNECTED)
    {
        // Fully connected has its input as a 3D tensor, but it needs to be treated as 1D
        numIfms = numUninterleavedIfms * inputShape[1] * inputShape[2];
        numOfms = outputShape[3];
    }
    else
    {
        ETHOSN_FAIL_MSG("Unexpected convtype");
    }

    return numIfms * numOpsPerIfmPerOfm * numOfms;
}

MceStats GetMceStats(const HardwareCapabilities& caps,
                     const Stride& stride,
                     const ethosn::command_stream::MceOperation& convtype,
                     const CompilerMceAlgorithm& algo,
                     const TensorShape& inputShape,
                     const TensorShape& outputShape,
                     const TensorShape& weightsShape)
{
    MceStats data;
    const uint32_t weightsHeight = weightsShape[0];
    const uint32_t weightsWidth  = weightsShape[1];

    data.m_CycleCount =
        GetMceCycleCount(caps, stride, convtype, algo, inputShape, outputShape, weightsHeight, weightsWidth);

    data.m_Operations = GetNumOperations(caps, stride, convtype, inputShape, outputShape, weightsHeight, weightsWidth);

    return data;
}

std::vector<uint8_t> GenerateCompressibleData(size_t numElements, float spaceSavingProportion, int32_t zeroPoint)
{
    std::vector<uint8_t> dummyWeightData(numElements);
    std::mt19937 gen;
    // Note that we use the generator manually rather than using a distribution, as distributions
    // are not guaranteed to give consistent results across STL implementations and therefore makes
    // the results harder to debug across machines/platforms/compilers etc.
    generate(dummyWeightData.begin(), dummyWeightData.end(),
             [&]() -> uint8_t { return static_cast<uint8_t>(gen() % 256); });

    // Generate zero data with the weight ratio provided
    std::bernoulli_distribution bernoulliDistribution(1.0f - spaceSavingProportion);
    std::vector<uint8_t> zeroData(numElements);
    generate(zeroData.begin(), zeroData.end(), [&]() -> uint8_t { return bernoulliDistribution(gen); });

    // Take into account the zero point in the quantization info.
    auto QuantizeIfZero = [zeroPoint](uint8_t a, uint8_t b) { return a == 0 ? static_cast<uint8_t>(zeroPoint) : b; };
    std::transform(zeroData.begin(), zeroData.end(), dummyWeightData.begin(), dummyWeightData.begin(), QuantizeIfZero);
    return dummyWeightData;
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

WeightsStats GetWeightsStats(const HardwareCapabilities& caps,
                             const EncodedWeights& encodedWeights,
                             const TensorInfo& info,
                             const uint32_t tileSize,
                             const TensorShape& inShape,
                             const TensorShape& inStripeShape)
{
    WeightsStats data;

    const uint32_t stripeSize = encodedWeights.m_MaxSize;

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

}    // namespace support_library
}    // namespace ethosn
