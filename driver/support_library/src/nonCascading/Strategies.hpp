//
// Copyright © 2018-2021 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "GraphNodes.hpp"
#include "Network.hpp"
#include "SramAllocator.hpp"

#include <ethosn_command_stream/CommandStream.hpp>

namespace ethosn
{
namespace support_library
{

class HardwareCapabilities;
struct TensorConfig;

enum class Strategy
{
    NONE,
    STRATEGY_0,
    STRATEGY_1,
    STRATEGY_3,
    STRATEGY_4,
    STRATEGY_6,
    STRATEGY_7,
    STRATEGY_X,
};

class IStrategy
{
public:
    virtual bool TrySetupAnyBlockConfig(TensorConfig& tensorConfig,
                                        SramAllocator& sramAllocator,
                                        const TensorShape& inputShape,
                                        const TensorShape& mceOutputShape,
                                        const TensorShape& outputShape,
                                        DataFormat weightsFormat,
                                        const TensorShape& weightsShape,
                                        const std::vector<command_stream::BlockConfig>& allowedBlockConfigs,
                                        const HardwareCapabilities& capabilities,
                                        const utils::ShapeMultiplier& mceShapeMultiplier,
                                        const utils::ShapeMultiplier& pleShapeMultiplier,
                                        std::pair<bool, uint32_t> inputStaticAndOffset,
                                        CompilerMceAlgorithm algorithm,
                                        const uint32_t depthMax = UINT32_MAX) = 0;

    virtual ~IStrategy()
    {}
};

/// An IStrategy which uses the default block config selection approach, which is to sort them by a metric and then
/// try them each in turn, choosing the first that works.
class IStrategyDefaultBlockSelection : public IStrategy
{
public:
    /// Implementation of IStrategy::TrySetupAnyBlockConfig
    bool TrySetupAnyBlockConfig(TensorConfig& tensorConfig,
                                SramAllocator& sramAllocator,
                                const TensorShape& inputShape,
                                const TensorShape& mceOutputShape,
                                const TensorShape& outputShape,
                                DataFormat weightsFormat,
                                const TensorShape& weightsShape,
                                const std::vector<command_stream::BlockConfig>& allowedBlockConfigs,
                                const HardwareCapabilities& capabilities,
                                const utils::ShapeMultiplier& mceShapeMultiplier,
                                const utils::ShapeMultiplier& pleShapeMultiplier,
                                std::pair<bool, uint32_t> inputStaticAndOffset,
                                CompilerMceAlgorithm algorithm,
                                const uint32_t depthMax = UINT32_MAX) final;

    /// Interface for derived classes to implement, which attempts a single block config.
    virtual bool TrySetup(TensorConfig& tensorConfig,
                          SramAllocator& sramAllocator,
                          const TensorShape& inputShape,
                          const TensorShape& mceOutputShape,
                          const TensorShape& outputShape,
                          DataFormat weightsFormat,
                          const TensorShape& weightsShape,
                          const ethosn::command_stream::BlockConfig& blockConfig,
                          const HardwareCapabilities& capabilities,
                          const utils::ShapeMultiplier& mceShapeMultiplier,
                          const utils::ShapeMultiplier& pleShapeMultiplier,
                          std::pair<bool, uint32_t> inputStaticAndOffset,
                          CompilerMceAlgorithm algorithm,
                          const uint32_t depthMax = UINT32_MAX) = 0;
};

/// SRAM allocation strategy where the input feature map is "streamed" in one stripe at a time.
/// Used when inputs are larger than what can fit in the SRAM.
/// Weights are not streamed in, but copied all at once.
class Strategy0 : public IStrategyDefaultBlockSelection
{
public:
    virtual bool TrySetup(TensorConfig& tensorConfig,
                          SramAllocator& sramAllocator,
                          const TensorShape& inputShape,
                          const TensorShape& mceOutputShape,
                          const TensorShape& outputShape,
                          DataFormat weightsFormat,
                          const TensorShape& weightsShape,
                          const ethosn::command_stream::BlockConfig& blockConfig,
                          const HardwareCapabilities& capabilities,
                          const utils::ShapeMultiplier& mceShapeMultiplier,
                          const utils::ShapeMultiplier& pleShapeMultiplier,
                          std::pair<bool, uint32_t> inputStaticAndOffset,
                          CompilerMceAlgorithm algorithm,
                          const uint32_t depthMax = UINT32_MAX) override;
};

/// SRAM allocation strategy where the weights are "streamed" in one depth stripe at a time.
/// Used when weights are larger than what can fit in the SRAM.
/// Input feature maps are not streamed in, but copied all at once.
class Strategy1 : public IStrategyDefaultBlockSelection
{
public:
    virtual bool TrySetup(TensorConfig& tensorConfig,
                          SramAllocator& sramAllocator,
                          const TensorShape& inputShape,
                          const TensorShape& mceOutputShape,
                          const TensorShape& outputShape,
                          DataFormat weightsFormat,
                          const TensorShape& weightsShape,
                          const ethosn::command_stream::BlockConfig& blockConfig,
                          const HardwareCapabilities& capabilities,
                          const utils::ShapeMultiplier& mceShapeMultiplier,
                          const utils::ShapeMultiplier& pleShapeMultiplier,
                          std::pair<bool, uint32_t> inputStaticAndOffset,
                          CompilerMceAlgorithm algorithm,
                          const uint32_t depthMax = UINT32_MAX) override;
};

/// SRAM allocation strategy where input feature maps and weights are copied all at once.
class Strategy3 : public IStrategyDefaultBlockSelection
{
public:
    virtual bool TrySetup(TensorConfig& tensorConfig,
                          SramAllocator& sramAllocator,
                          const TensorShape& inputShape,
                          const TensorShape& mceOutputShape,
                          const TensorShape& outputShape,
                          DataFormat weightsFormat,
                          const TensorShape& weightsShape,
                          const ethosn::command_stream::BlockConfig& blockConfig,
                          const HardwareCapabilities& capabilities,
                          const utils::ShapeMultiplier& mceShapeMultiplier,
                          const utils::ShapeMultiplier& pleShapeMultiplier,
                          std::pair<bool, uint32_t> inputStaticAndOffset,
                          CompilerMceAlgorithm algorithm,
                          const uint32_t depthMax = UINT32_MAX) override;
};

/// Implementation of the SRAM allocation strategy 4 where the input width
/// and the output depth are "streamed" one stripe at a time.
/// The full height is streamed in.
class Strategy4 : public IStrategy
{
public:
    virtual bool TrySetupAnyBlockConfig(TensorConfig& tensorConfig,
                                        SramAllocator& sramAllocator,
                                        const TensorShape& inputShape,
                                        const TensorShape& mceOutputShape,
                                        const TensorShape& outputShape,
                                        DataFormat weightsFormat,
                                        const TensorShape& weightsShape,
                                        const std::vector<command_stream::BlockConfig>& allowedBlockConfigs,
                                        const HardwareCapabilities& capabilities,
                                        const utils::ShapeMultiplier& mceShapeMultiplier,
                                        const utils::ShapeMultiplier& pleShapeMultiplier,
                                        std::pair<bool, uint32_t> inputStaticAndOffset,
                                        CompilerMceAlgorithm algorithm,
                                        const uint32_t depthMax = UINT32_MAX) override;
};

/// This strategy splits along width, height and depth
class Strategy6 : public IStrategy
{
public:
    virtual bool TrySetupAnyBlockConfig(TensorConfig& tensorConfig,
                                        SramAllocator& sramAllocator,
                                        const TensorShape& inputShape,
                                        const TensorShape& mceOutputShape,
                                        const TensorShape& outputShape,
                                        DataFormat weightsFormat,
                                        const TensorShape& weightsShape,
                                        const std::vector<command_stream::BlockConfig>& allowedBlockConfigs,
                                        const HardwareCapabilities& capabilities,
                                        const utils::ShapeMultiplier& mceShapeMultiplier,
                                        const utils::ShapeMultiplier& pleShapeMultiplier,
                                        std::pair<bool, uint32_t> inputStaticAndOffset,
                                        CompilerMceAlgorithm algorithm,
                                        const uint32_t depthMax = UINT32_MAX) override;
};

/// This strategy is similar to strategy 1, however splits the IFM along depth.
class Strategy7 : public IStrategyDefaultBlockSelection
{
public:
    virtual bool TrySetup(TensorConfig& tensorConfig,
                          SramAllocator& sramAllocator,
                          const TensorShape& inputShape,
                          const TensorShape& mceOutputShape,
                          const TensorShape& outputShape,
                          DataFormat weightsFormat,
                          const TensorShape& weightsShape,
                          const ethosn::command_stream::BlockConfig& blockConfig,
                          const HardwareCapabilities& capabilities,
                          const utils::ShapeMultiplier& mceShapeMultiplier,
                          const utils::ShapeMultiplier& pleShapeMultiplier,
                          std::pair<bool, uint32_t> inputStaticAndOffset,
                          CompilerMceAlgorithm algorithm,
                          const uint32_t depthMax = UINT32_MAX) override;
};

}    // namespace support_library
}    // namespace ethosn
