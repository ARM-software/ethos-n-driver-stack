//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "GraphNodes.hpp"
#include "Pass.hpp"

namespace ethosn
{
namespace support_library
{

bool IsStrategyX(const command_stream::MceOperation& mceOperation,
                 const StrategyConfig& strategyConfig,
                 const CompilerMceAlgorithm algorithm,
                 const std::vector<IStrategy*>& allowedStrategies);

bool TryStrategyX(const command_stream::MceOperation& mceOperation,
                  const command_stream::UpsampleType upsampleType,
                  StrategyConfig& strategyConfig,
                  SramAllocator& sramAllocator,
                  const TensorShape& inputShape,
                  const TensorShape& outputShape,
                  const DataFormat weightsFormat,
                  const TensorShape& weightsShape,
                  std::pair<const uint32_t, const uint32_t> pad,
                  const std::vector<command_stream::BlockConfig>& allowedBlockConfigs,
                  const HardwareCapabilities& capabilities,
                  const utils::ShapeMultiplier& mceShapeMultiplier,
                  const utils::ShapeMultiplier& pleShapeMultiplier,
                  std::pair<const bool, const uint32_t> inputStaticAndOffset,
                  const uint32_t depthMax);

}    // namespace support_library
}    // namespace ethosn
