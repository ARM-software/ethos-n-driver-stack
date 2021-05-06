//
// Copyright © 2020,2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "../GraphNodes.hpp"
#include "../WeightEncoder.hpp"
#include "Plan.hpp"

namespace ethosn
{
namespace support_library
{

class HardwareCapabilities;

MceStats GetMceStats(const HardwareCapabilities& caps,
                     const Stride& stride,
                     const ethosn::command_stream::MceOperation& convtype,
                     const CompilerMceAlgorithm& algo,
                     const TensorShape& inputShape,
                     const TensorShape& outputShape,
                     const TensorShape& weightsShape);

WeightsStats GetWeightsStats(const HardwareCapabilities& caps,
                             const EncodedWeights& encodedWeights,
                             const TensorInfo& info,
                             const TensorShape& stripeShape,
                             const uint32_t tileSize,
                             const TensorShape& inShape,
                             const TensorShape& inStripeShape);

std::vector<uint8_t> GenerateCompressibleData(size_t numElements, float spaceSavingProportion, int32_t zeroPoint);

}    //namespace support_library
}    //namespace ethosn
