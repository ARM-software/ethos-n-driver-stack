//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
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
    STRATEGY_5,
    STRATEGY_6,
    STRATEGY_7,
    STRATEGY_X,
    STRATEGY_FC
};

class IStrategy
{
public:
    virtual bool TrySetup(TensorConfig& tensorConfig,
                          SramAllocator& sramAllocator,
                          const TensorShape& inputShape,
                          const TensorShape& outputShape,
                          DataFormat weightsFormat,
                          const TensorShape& weightsShape,
                          const ethosn::command_stream::BlockConfig& blockConfig,
                          const HardwareCapabilities& capabilities,
                          const utils::ShapeMultiplier& shapeMultiplier,
                          std::pair<bool, uint32_t> inputStaticAndOffset,
                          CompilerMceAlgorithm algorithm,
                          const uint32_t depthMax = UINT32_MAX) = 0;

    virtual ~IStrategy()
    {}

    virtual const char* GetStrategyString() = 0;
};

/// SRAM allocation strategy where the input feature map is "streamed" in one stripe at a time.
/// Used when inputs are larger than what can fit in the SRAM.
/// Weights are not streamed in, but copied all at once.
class Strategy0 : public IStrategy
{
public:
    virtual bool TrySetup(TensorConfig& tensorConfig,
                          SramAllocator& sramAllocator,
                          const TensorShape& inputShape,
                          const TensorShape& outputShape,
                          DataFormat weightsFormat,
                          const TensorShape& weightsShape,
                          const ethosn::command_stream::BlockConfig& blockConfig,
                          const HardwareCapabilities& capabilities,
                          const utils::ShapeMultiplier& shapeMultiplier,
                          std::pair<bool, uint32_t> inputStaticAndOffset,
                          CompilerMceAlgorithm algorithm,
                          const uint32_t depthMax = UINT32_MAX) override;

    virtual const char* GetStrategyString() override;
};

/// SRAM allocation strategy where the weights are "streamed" in one depth stripe at a time.
/// Used when weights are larger than what can fit in the SRAM.
/// Input feature maps are not streamed in, but copied all at once.
class Strategy1 : public IStrategy
{
public:
    virtual bool TrySetup(TensorConfig& tensorConfig,
                          SramAllocator& sramAllocator,
                          const TensorShape& inputShape,
                          const TensorShape& outputShape,
                          DataFormat weightsFormat,
                          const TensorShape& weightsShape,
                          const ethosn::command_stream::BlockConfig& blockConfig,
                          const HardwareCapabilities& capabilities,
                          const utils::ShapeMultiplier& shapeMultiplier,
                          std::pair<bool, uint32_t> inputStaticAndOffset,
                          CompilerMceAlgorithm algorithm,
                          const uint32_t depthMax = UINT32_MAX) override;

    virtual const char* GetStrategyString() override;
};

/// SRAM allocation strategy where input feature maps and weights are copied all at once.
class Strategy3 : public IStrategy
{
public:
    virtual bool TrySetup(TensorConfig& tensorConfig,
                          SramAllocator& sramAllocator,
                          const TensorShape& inputShape,
                          const TensorShape& outputShape,
                          DataFormat weightsFormat,
                          const TensorShape& weightsShape,
                          const ethosn::command_stream::BlockConfig& blockConfig,
                          const HardwareCapabilities& capabilities,
                          const utils::ShapeMultiplier& shapeMultiplier,
                          std::pair<bool, uint32_t> inputStaticAndOffset,
                          CompilerMceAlgorithm algorithm,
                          const uint32_t depthMax = UINT32_MAX) override;

    virtual const char* GetStrategyString() override;
};

/// Implementation of the SRAM allocation strategy 4 where the input width
/// and the output depth are "streamed" one stripe at a time.
/// The full height is streamed in.
class Strategy4 : public IStrategy
{
public:
    virtual bool TrySetup(TensorConfig& tensorConfig,
                          SramAllocator& sramAllocator,
                          const TensorShape& inputShape,
                          const TensorShape& outputShape,
                          DataFormat weightsFormat,
                          const TensorShape& weightsShape,
                          const ethosn::command_stream::BlockConfig& blockConfig,
                          const HardwareCapabilities& capabilities,
                          const utils::ShapeMultiplier& shapeMultiplier,
                          std::pair<bool, uint32_t> inputStaticAndOffset,
                          CompilerMceAlgorithm algorithm,
                          const uint32_t depthMax = UINT32_MAX) override;

    virtual const char* GetStrategyString() override;
};

/// This strategy splits along width, height and depth
class Strategy6 : public IStrategy
{
public:
    virtual bool TrySetup(TensorConfig& tensorConfig,
                          SramAllocator& sramAllocator,
                          const TensorShape& inputShape,
                          const TensorShape& outputShape,
                          DataFormat weightsFormat,
                          const TensorShape& weightsShape,
                          const ethosn::command_stream::BlockConfig& blockConfig,
                          const HardwareCapabilities& capabilities,
                          const utils::ShapeMultiplier& shapeMultiplier,
                          std::pair<bool, uint32_t> inputStaticAndOffset,
                          CompilerMceAlgorithm algorithm,
                          const uint32_t depthMax = UINT32_MAX) override;

    virtual const char* GetStrategyString() override;
};

/// This strategy is similar to strategy 1, however splits the IFM along depth.
/// .
class Strategy7 : public IStrategy
{
public:
    virtual bool TrySetup(TensorConfig& tensorConfig,
                          SramAllocator& sramAllocator,
                          const TensorShape& inputShape,
                          const TensorShape& outputShape,
                          DataFormat weightsFormat,
                          const TensorShape& weightsShape,
                          const ethosn::command_stream::BlockConfig& blockConfig,
                          const HardwareCapabilities& capabilities,
                          const utils::ShapeMultiplier& shapeMultiplier,
                          std::pair<bool, uint32_t> inputStaticAndOffset,
                          CompilerMceAlgorithm algorithm,
                          const uint32_t depthMax = UINT32_MAX) override;

    virtual const char* GetStrategyString() override;
};

/// SRAM allocation strategy for fully connected
class StrategyFc : public IStrategy
{
public:
    virtual bool TrySetup(TensorConfig& tensorConfig,
                          SramAllocator& sramAllocator,
                          const TensorShape& inputShape,
                          const TensorShape& outputShape,
                          DataFormat weightsFormat,
                          const TensorShape& weightsShape,
                          const ethosn::command_stream::BlockConfig& blockConfig,
                          const HardwareCapabilities& capabilities,
                          const utils::ShapeMultiplier& shapeMultiplier,
                          std::pair<bool, uint32_t> inputStaticAndOffset,
                          CompilerMceAlgorithm algorithm,
                          const uint32_t depthMax = UINT32_MAX) override;

    virtual const char* GetStrategyString() override;
};

}    // namespace support_library
}    // namespace ethosn
