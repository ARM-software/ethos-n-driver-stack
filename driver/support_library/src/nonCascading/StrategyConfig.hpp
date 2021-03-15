//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../../include/ethosn_support_library/Support.hpp"

namespace ethosn
{

namespace support_library
{
// Describes the allocation of a tensor in SRAM
struct SramTensorAllocation
{
    uint32_t tileSize;
    uint32_t numStripesInTile;
    TensorShape stripeShape;
    uint32_t offset;
};

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

// Output of ChooseAndSetupStrategy. Describes the allocation of input, output and weight tensors in SRAM
struct StrategyConfig
{
    SramTensorAllocation inputAllocation;
    SramTensorAllocation outputAllocation;
    SramTensorAllocation weightsAllocation;
    SramTensorAllocation pleAllocation;
    uint32_t blockWidth;
    uint32_t blockHeight;
    Strategy strategy = Strategy::NONE;
};

}    // namespace support_library

}    // namespace ethosn
