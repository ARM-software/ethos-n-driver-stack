//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
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

    const uint32_t ifmConsumed = caps.GetIfmPerEngine() * caps.GetNumberOfEngines();
    const uint32_t ofmProduced = caps.GetOfmPerEngine() * caps.GetNumberOfEngines();
    // Winograd output size can be 2x2 for 2D or 1x2 and 2x1 for 1D
    const uint32_t winogradOutputH =
        weightsHeight == 1U ? caps.GetOutputSizePerWinograd1D() : caps.GetOutputSizePerWinograd2D();
    const uint32_t winogradOutputW =
        weightsWidth == 1U ? caps.GetOutputSizePerWinograd1D() : caps.GetOutputSizePerWinograd2D();

    uint32_t numIfms = inputShape[3];
    uint32_t numOfms = outputShape[3];

    const uint32_t numTotIfms = utils::RoundUpToNearestMultiple(numIfms, ifmConsumed);
    // Number of Winograd output (i.e. 2x2, 1x2, 2x1) on HW plane
    const uint32_t numWinogradOutputs =
        utils::DivRoundUp(outputShape[2], winogradOutputW) * utils::DivRoundUp(outputShape[1], winogradOutputH);

    const uint32_t wideKernelSize = caps.GetWideKernelSize();
    const uint64_t numMacsPerElemHW =
        weightsHeight == 1 || weightsWidth == 1
            ? caps.GetMacsPerWinograd1D() * utils::DivRoundUp(weightsWidth * weightsHeight, wideKernelSize)
            : caps.GetMacsPerWinograd2D() * utils::DivRoundUp(weightsWidth, wideKernelSize) *
                  utils::DivRoundUp(weightsHeight, wideKernelSize);

    const uint64_t numMacOps       = numWinogradOutputs * numMacsPerElemHW;
    const uint64_t numCyclesPerOfm = (numTotIfms * numMacOps) / (ifmConsumed * caps.GetMacUnitsPerEngine());

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
    const uint32_t ifmConsumed       = caps.GetIfmPerEngine() * caps.GetNumberOfEngines();
    const uint32_t ofmProduced       = caps.GetOfmPerEngine() * caps.GetNumberOfEngines();
    const uint32_t halfPatchH        = caps.GetPatchShape()[1];
    const uint32_t halfPatchW        = utils::DivRoundUp(caps.GetPatchShape()[2], 2u);
    const uint32_t numActualIfms     = inputShape[3] / (stride.m_X * stride.m_Y);

    uint32_t numIfms = numActualIfms;
    uint32_t numOfms = outputShape[3];

    if (convtype == ethosn::command_stream::MceOperation::DEPTHWISE_CONVOLUTION)
    {
        numIfms = ifmConsumed;
        numOfms = numActualIfms;
    }

    const uint32_t numTotIfms = utils::RoundUpToNearestMultiple(numIfms, ifmConsumed);
    // Number of output elements on HW plane when the height and width are rounded up to half patches
    const uint32_t numOutputElements = utils::RoundUpToNearestMultiple(outputShape[2], halfPatchW) *
                                       utils::RoundUpToNearestMultiple(outputShape[1], halfPatchH);

    const uint64_t numMacOps       = numOutputElements * numKernelElements;
    const uint64_t numCyclesPerOfm = (numTotIfms * numMacOps) / (ifmConsumed * caps.GetMacUnitsPerEngine());

    return numCyclesPerOfm * utils::DivRoundUp(numOfms, ofmProduced);
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

uint64_t GetNumOperations(const Stride& stride,
                          const ethosn::command_stream::MceOperation& convtype,
                          const TensorShape& inputShape,
                          const TensorShape& outputShape,
                          const uint32_t weightsHeight,
                          const uint32_t weightsWidth)
{
    const uint64_t numKernelElements = weightsWidth * weightsHeight;
    const uint64_t numOpsPerElement  = numKernelElements + numKernelElements;
    const uint64_t numActualIfms     = utils::DivRoundUp(inputShape[3], (stride.m_X * stride.m_Y));
    const uint64_t numInputElements  = inputShape[1] * inputShape[2];
    const uint64_t numOpsPerIfm      = numInputElements * numOpsPerElement;

    uint64_t numIfms = numActualIfms;
    uint64_t numOfms = outputShape[3];

    if (convtype == ethosn::command_stream::MceOperation::DEPTHWISE_CONVOLUTION)
    {
        numIfms = 1;
        numOfms = numActualIfms;
    }

    return numIfms * numOpsPerIfm * numOfms;
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

    data.m_Operations = GetNumOperations(stride, convtype, inputShape, outputShape, weightsHeight, weightsWidth);

    return data;
}

std::vector<uint8_t> GenerateCompressibleData(size_t numElements, float spaceSavingProportion, int32_t zeroPoint)
{
    std::vector<uint8_t> dummyWeightData(numElements);
    std::mt19937 gen;
    std::uniform_int_distribution<> uniformDistribution(0, 255);
    generate(dummyWeightData.begin(), dummyWeightData.end(),
             [&]() -> uint8_t { return static_cast<uint8_t>(uniformDistribution(gen)); });

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

}    // namespace support_library
}    // namespace ethosn
