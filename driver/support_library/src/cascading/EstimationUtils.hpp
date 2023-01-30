//
// Copyright © 2020-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "../GraphNodes.hpp"
#include "../WeightEncoder.hpp"
#include "Plan.hpp"

#include <vector>

namespace ethosn
{
namespace support_library
{

class Buffer;
class HardwareCapabilities;

struct ConversionData
{
    ConversionData() = default;
    TensorShape tensorShape;
    TensorShape stripeShape;
    bool isNhwc;
};

PassStats GetConversionStats(const ConversionData& input, const ConversionData& output, bool isDramToDram);

PleStats GetPleStats(const HardwareCapabilities& caps,
                     const std::vector<TensorShape>& inputShapes,
                     const command_stream::PleOperation& pleoperation);

InputStats GetInputStatsLegacy(const HardwareCapabilities& caps,
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

InputStats GetInputStatsCascading(const Buffer& ifmBuffer,
                                  const TensorShape& weightsShape,
                                  utils::Optional<CascadingBufferFormat> dramBufferFormat);

OutputStats GetOutputStatsLegacy(const TensorShape& shape, const TensorShape& stripeShape, const Location location);
OutputStats GetOutputStatsCascading(const Buffer& ofmSramBuffer,
                                    utils::Optional<CascadingBufferFormat> dramBufferFormat);

InputStats AccountForActivationCompression(InputStats stats, float spaceSavingRatio);

/// Increases the number of stripes in the given stats if the transfer between the two buffers provided
/// would result in the DMA having to be split into multiple chunks. This is useful as the performance estimate
/// will then take this into account, and prefer to choose strategies that don't require chunking.
StripesStats AccountForDmaChunking(StripesStats stats,
                                   const Buffer& sramBuffer,
                                   const Buffer& dramBuffer,
                                   bool dramStridingAllowed);

double CalculateMetric(const NetworkPerformanceData& networkPerfData);
double CalculateMetric(const PassPerformanceData& passPerfData);

}    //namespace support_library
}    //namespace ethosn
