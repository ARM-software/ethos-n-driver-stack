//
// Copyright © 2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "../GraphNodes.hpp"
#include "../Utils.hpp"
#include "../WeightEncoder.hpp"
#include "Plan.hpp"

namespace ethosn
{
namespace support_library
{

class Buffer;

MceStats GetMceStats(const HardwareCapabilities& caps,
                     const Stride& stride,
                     const ethosn::command_stream::MceOperation& convtype,
                     const CompilerMceAlgorithm& algo,
                     const TensorShape& inputShape,
                     const TensorShape& outputShape,
                     const TensorShape& weightsShape);

PleStats GetPleStats(const HardwareCapabilities& caps,
                     const std::vector<TensorShape>& inputStripeShapes,
                     const command_stream::PleOperation& pleoperation);

InputStats GetInputStats(const HardwareCapabilities& caps,
                         const Buffer* inpbuf,
                         const Buffer* outbuff,
                         const uint32_t& inputTileSize,
                         const TensorInfo& weights = {
                             { { 1, 1, 1, 1 } },
                             DataType::UINT8_QUANTIZED,
                             DataFormat::HWIM,
                             { 0, 0.1f },
                         });

OutputStats GetOutputStats(const TensorShape& shape, const TensorShape& stripeShape, const BufferLocation location);

WeightsStats GetWeightsStats(const HardwareCapabilities& caps,
                             EncodedWeights& encodedWeights,
                             const TensorInfo& info,
                             const TensorShape& stripeShape,
                             const uint32_t tileSize,
                             const TensorShape& inShape,
                             const TensorShape& inStripeShape);

}    //namespace support_library
}    //namespace ethosn
