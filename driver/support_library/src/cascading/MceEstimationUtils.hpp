//
// Copyright © 2020-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "../WeightEncoder.hpp"
#include "Plan.hpp"

namespace ethosn
{
namespace support_library
{

class HardwareCapabilities;

uint64_t GetMceCycleCountWinograd(const HardwareCapabilities& caps,
                                  const TensorShape& inputShape,
                                  const TensorShape& outputShape,
                                  const uint32_t weightsHeight,
                                  const uint32_t weightsWidth,
                                  const ethosn::command_stream::BlockConfig& blockConfig);

uint64_t GetMceCycleCountDirect(const HardwareCapabilities& caps,
                                const Stride& stride,
                                const ethosn::command_stream::MceOperation& convtype,
                                const TensorShape& inputShape,
                                const TensorShape& outputShape,
                                const uint32_t weightsHeight,
                                const uint32_t weightsWidth);

MceStats GetMceStats(const HardwareCapabilities& caps,
                     const Stride& stride,
                     const ethosn::command_stream::MceOperation& convtype,
                     const CompilerMceAlgorithm& algo,
                     const TensorShape& inputShape,
                     const TensorShape& outputShape,
                     const TensorShape& weightsShape,
                     const ethosn::command_stream::BlockConfig& blockConfig);

WeightsStats GetWeightsStats(const HardwareCapabilities& caps,
                             const EncodedWeights& encodedWeights,
                             const TensorInfo& info,
                             const uint32_t tileSize,
                             const TensorShape& inShape,
                             const TensorShape& inStripeShape);

std::vector<uint8_t> GenerateCompressibleData(size_t numElements, float spaceSavingProportion, int32_t zeroPoint);

CompilerMceAlgorithm FindBestConvAlgorithm(const HardwareCapabilities& caps,
                                           const Stride& stride,
                                           const ethosn::command_stream::MceOperation& convtype,
                                           const TensorShape& inputShape,
                                           const TensorShape& outputShape,
                                           const uint32_t weightsHeight,
                                           const uint32_t weightsWidth,
                                           const ethosn::command_stream::BlockConfig& blockConfig);

}    //namespace support_library
}    //namespace ethosn
