//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "OpUtils.hpp"

namespace ethosn
{
namespace support_library
{

// TODO: Slightly modified but duplicated functions from McePlePass.cpp.
// Need to adopt to only one version. Fix in NNXSW-2208.
uint32_t GetMceCycleCountWinograd(const HardwareCapabilities& caps,
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
    const uint32_t numMacsPerElemHW =
        weightsHeight == 1 || weightsWidth == 1
            ? caps.GetMacsPerWinograd1D() * utils::DivRoundUp(weightsWidth * weightsHeight, wideKernelSize)
            : caps.GetMacsPerWinograd2D() * utils::DivRoundUp(weightsWidth, wideKernelSize) *
                  utils::DivRoundUp(weightsHeight, wideKernelSize);

    const uint32_t numMacOps       = numWinogradOutputs * numMacsPerElemHW;
    const uint32_t numCyclesPerOfm = (numTotIfms * numMacOps) / (ifmConsumed * caps.GetMacUnitsPerEngine());

    return numCyclesPerOfm * utils::DivRoundUp(numOfms, ofmProduced);
}

uint32_t GetMceCycleCountDirect(const HardwareCapabilities& caps,
                                const Stride& stride,
                                const ethosn::command_stream::MceOperation& convtype,
                                //const MceOperationNode* mceOperation,
                                const TensorShape& inputShape,
                                const TensorShape& outputShape,
                                const uint32_t weightsHeight,
                                const uint32_t weightsWidth)
{
    //const Stride& stride             = mceOperation->GetStride();
    const uint32_t numKernelElements = weightsWidth * weightsHeight;
    const uint32_t ifmConsumed       = caps.GetIfmPerEngine() * caps.GetNumberOfEngines();
    const uint32_t ofmProduced       = caps.GetOfmPerEngine() * caps.GetNumberOfEngines();
    const uint32_t halfPatchH        = caps.GetPatchShape()[1];
    const uint32_t halfPatchW        = utils::DivRoundUp(caps.GetPatchShape()[2], 2);
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

    const uint32_t numMacOps       = numOutputElements * numKernelElements;
    const uint32_t numCyclesPerOfm = (numTotIfms * numMacOps) / (ifmConsumed * caps.GetMacUnitsPerEngine());

    return numCyclesPerOfm * utils::DivRoundUp(numOfms, ofmProduced);
}

uint32_t GetMceCycleCount(const HardwareCapabilities& caps,
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

uint32_t GetNumOperations(const Stride& stride,
                          const ethosn::command_stream::MceOperation& convtype,
                          const TensorShape& inputShape,
                          const TensorShape& outputShape,
                          const uint32_t weightsHeight,
                          const uint32_t weightsWidth)
{
    const uint32_t numKernelElements = weightsWidth * weightsHeight;
    const uint32_t numOpsPerElement  = numKernelElements + numKernelElements;
    const uint32_t numActualIfms     = utils::DivRoundUp(inputShape[3], (stride.m_X * stride.m_Y));
    const uint32_t numInputElements  = inputShape[1] * inputShape[2];
    const uint32_t numOpsPerIfm      = numInputElements * numOpsPerElement;

    uint32_t numIfms = numActualIfms;
    uint32_t numOfms = outputShape[3];

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

}    // namespace support_library
}    // namespace ethosn
