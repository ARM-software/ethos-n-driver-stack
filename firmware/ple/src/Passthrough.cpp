//
// Copyright © 2018-2020,2022-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#include "../include/ethosn_ple/BlockConstants.hpp"
#include "../include/ethosn_ple/Common.hpp"
#include "../include/ethosn_ple/MceStripeLoop.hpp"
#include "../include/ethosn_ple/PassthroughBase.hpp"

namespace
{
class Passthrough : public PassthroughBase<BlockSize, BlockSize, Passthrough>
{
public:
    Passthrough(PleState& pleState, const OperatorInfo& opInfo)
        : PassthroughBase<BlockSize, BlockSize, Passthrough>(
              pleState.GetActiveEvents(), opInfo.sizeInElements, opInfo.output.dfcAddr)
    {}

    void ProcessBlock()
    {}
};
}    // namespace

extern "C" void __attribute__((noreturn)) main()
{
    MainWithStripeLoop<MceStripeLoop<Passthrough>>();
}
