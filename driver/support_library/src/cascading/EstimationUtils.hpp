//
// Copyright © 2020-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

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

InputStats GetInputStatsCascading(const SramBuffer& ifmBuffer,
                                  const TensorShape& weightsShape,
                                  utils::Optional<CascadingBufferFormat> dramBufferFormat);

OutputStats GetOutputStatsCascading(const SramBuffer& ofmSramBuffer,
                                    utils::Optional<CascadingBufferFormat> dramBufferFormat);

InputStats AccountForActivationCompression(InputStats stats, float spaceSavingRatio);

/// Increases the number of stripes in the given stats if the transfer between the two buffers provided
/// would result in the DMA having to be split into multiple chunks. This is useful as the performance estimate
/// will then take this into account, and prefer to choose strategies that don't require chunking.
StripesStats AccountForDmaChunking(StripesStats stats,
                                   const SramBuffer& sramBuffer,
                                   const DramBuffer& dramBuffer,
                                   bool dramStridingAllowed);

double CalculateMetric(const NetworkPerformanceData& networkPerfData);
double CalculateMetric(const PassPerformanceData& passPerfData);

}    //namespace support_library
}    //namespace ethosn
