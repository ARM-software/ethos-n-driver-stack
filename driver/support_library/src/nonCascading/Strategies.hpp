//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "GraphNodes.hpp"
#include "Network.hpp"
#include "SramAllocator.hpp"
#include "StrategiesCommon.hpp"
#include "StrategyConfig.hpp"

#include <ethosn_command_stream/CommandStream.hpp>

namespace ethosn
{
namespace support_library
{

class HardwareCapabilities;
class MceStrategySelectionParameters;

class IStrategy
{
public:
    virtual MceStrategySelectionReturnValue
        TrySetupAnyBlockConfig(const MceStrategySelectionParameters& strategySelectionParameters,
                               const std::vector<command_stream::BlockConfig>& allowedBlockConfigs) = 0;

    virtual ~IStrategy()
    {}
};

/// An IStrategy which uses the default block config selection approach, which is to sort them by a metric and then
/// try them each in turn, choosing the first that works.
class IStrategyDefaultBlockSelection : public IStrategy
{
public:
    /// Implementation of IStrategy::TrySetupAnyBlockConfig
    MceStrategySelectionReturnValue
        TrySetupAnyBlockConfig(const MceStrategySelectionParameters& strategySelectionParameters,
                               const std::vector<command_stream::BlockConfig>& allowedBlockConfigs) final;

    /// Interface for derived classes to implement, which attempts a single block config.
    virtual MceStrategySelectionReturnValue TrySetup(const MceStrategySelectionParameters& strategySelectionParameters,
                                                     const ethosn::command_stream::BlockConfig& blockConfig) = 0;
};

/// SRAM allocation strategy where the input feature map is "streamed" in one stripe at a time.
/// Used when inputs are larger than what can fit in the SRAM.
/// Weights are not streamed in, but copied all at once.
class Strategy0 : public IStrategyDefaultBlockSelection
{
public:
    virtual MceStrategySelectionReturnValue TrySetup(const MceStrategySelectionParameters& strategySelectionParameters,
                                                     const ethosn::command_stream::BlockConfig& blockConfig) override;
};

/// SRAM allocation strategy where the weights are "streamed" in one depth stripe at a time.
/// Used when weights are larger than what can fit in the SRAM.
/// Input feature maps are not streamed in, but copied all at once.
class Strategy1 : public IStrategyDefaultBlockSelection
{
public:
    virtual MceStrategySelectionReturnValue TrySetup(const MceStrategySelectionParameters& strategySelectionParameters,
                                                     const ethosn::command_stream::BlockConfig& blockConfig) override;
};

/// SRAM allocation strategy where input feature maps and weights are copied all at once.
class Strategy3 : public IStrategyDefaultBlockSelection
{
public:
    virtual MceStrategySelectionReturnValue TrySetup(const MceStrategySelectionParameters& strategySelectionParameters,
                                                     const ethosn::command_stream::BlockConfig& blockConfig) override;
};

/// Implementation of the SRAM allocation strategy 4 where the input width
/// and the output depth are "streamed" one stripe at a time.
/// The full height is streamed in.
class Strategy4 : public IStrategy
{
public:
    virtual MceStrategySelectionReturnValue
        TrySetupAnyBlockConfig(const MceStrategySelectionParameters& strategySelectionParameters,
                               const std::vector<command_stream::BlockConfig>& allowedBlockConfigs) override;
};

/// This strategy splits along width, height and depth
class Strategy6 : public IStrategy
{
public:
    virtual MceStrategySelectionReturnValue
        TrySetupAnyBlockConfig(const MceStrategySelectionParameters& strategySelectionParameters,
                               const std::vector<command_stream::BlockConfig>& allowedBlockConfigs) override;
};

/// This strategy is similar to strategy 1, however splits the IFM along depth.
class Strategy7 : public IStrategyDefaultBlockSelection
{
public:
    virtual MceStrategySelectionReturnValue TrySetup(const MceStrategySelectionParameters& strategySelectionParameters,
                                                     const ethosn::command_stream::BlockConfig& blockConfig) override;
};

}    // namespace support_library
}    // namespace ethosn
