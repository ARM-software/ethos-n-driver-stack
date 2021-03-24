//
// Copyright © 2020-2021 Arm Limited.
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

class Buffer;
class HardwareCapabilities;

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

uint64_t GetPerformanceDataMetric(const PassStats& passStat);
uint64_t GetMetric(const NetworkPerformanceData& netPerfData);
bool IsLeftMoreDataPerformantThanRight(const NetworkPerformanceData& left, const NetworkPerformanceData& right);

}    //namespace support_library
}    //namespace ethosn
