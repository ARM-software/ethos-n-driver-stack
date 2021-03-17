//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "GraphNodes.hpp"
#include "Pass.hpp"
#include "StrategiesCommon.hpp"

namespace ethosn
{
namespace support_library
{

struct StrategyXSelectionParameters
{
    StrategyXSelectionParameters(command_stream::MceOperation mceOperation,
                                 command_stream::UpsampleType upsampleType,
                                 SramAllocator sramAllocator,
                                 TensorShape inputShape,
                                 TensorShape outputShape,
                                 DataFormat weightsFormat,
                                 TensorShape weightsShape,
                                 std::pair<const uint32_t, const uint32_t> pad,
                                 std::vector<command_stream::BlockConfig>& allowedBlockConfigs,
                                 HardwareCapabilities capabilities,
                                 utils::ShapeMultiplier mceShapeMultiplier,
                                 utils::ShapeMultiplier pleShapeMultiplier,
                                 std::pair<const bool, const uint32_t> inputStaticAndOffset,
                                 uint32_t depthMax)
        : mceOperation{ mceOperation }
        , upsampleType{ upsampleType }
        , sramAllocator{ sramAllocator }
        , inputShape{ inputShape }
        , outputShape{ outputShape }
        , weightsFormat{ weightsFormat }
        , weightsShape{ weightsShape }
        , pad{ pad }
        , allowedBlockConfigs{ allowedBlockConfigs }
        , capabilities{ capabilities }
        , mceShapeMultiplier{ mceShapeMultiplier }
        , pleShapeMultiplier{ pleShapeMultiplier }
        , inputStaticAndOffset{ inputStaticAndOffset }
        , depthMax{ depthMax }
    {}
    // Remove the copy constructor/assignment operator because there are not reason that this huge object
    // wrapper must be copied.
    StrategyXSelectionParameters(const StrategyXSelectionParameters&) = delete;
    StrategyXSelectionParameters& operator=(const StrategyXSelectionParameters&) = delete;

    command_stream::MceOperation mceOperation;
    command_stream::UpsampleType upsampleType;
    SramAllocator sramAllocator;
    TensorShape inputShape;
    TensorShape outputShape;
    DataFormat weightsFormat;
    TensorShape weightsShape;
    std::pair<const uint32_t, const uint32_t> pad;
    std::vector<command_stream::BlockConfig>& allowedBlockConfigs;
    HardwareCapabilities capabilities;
    utils::ShapeMultiplier mceShapeMultiplier;
    utils::ShapeMultiplier pleShapeMultiplier;
    std::pair<const bool, const uint32_t> inputStaticAndOffset;
    uint32_t depthMax;
};

bool IsStrategyX(const command_stream::MceOperation& mceOperation,
                 const StrategyConfig& strategyConfig,
                 const CompilerMceAlgorithm algorithm,
                 const std::vector<IStrategy*>& allowedStrategies);

MceStrategySelectionReturnValue TryStrategyX(const StrategyXSelectionParameters& strategyXSelectionParameters);

}    // namespace support_library
}    // namespace ethosn
