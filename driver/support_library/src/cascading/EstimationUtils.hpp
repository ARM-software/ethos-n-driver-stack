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

PleStats GetPleStats(const HardwareCapabilities& caps,
                     const std::vector<TensorShape>& inputShapes,
                     const command_stream::PleOperation& pleoperation);

InputStats GetInputStats(const HardwareCapabilities& caps,
                         const TensorShape& shape,
                         const TensorShape& stripeShape,
                         const Location location,
                         const uint32_t tileSize,
                         const TensorInfo& weights =
                             {
                                 { { 1, 1, 1, 1 } },
                                 DataType::UINT8_QUANTIZED,
                                 DataFormat::HWIM,
                                 { 0, 0.1f },
                             },
                         const uint32_t numOutStripesC = 1);

OutputStats GetOutputStats(const TensorShape& shape, const TensorShape& stripeShape, const Location location);

InputStats AccountForActivationCompression(InputStats stats, float spaceSavingRatio);

}    //namespace support_library
}    //namespace ethosn
