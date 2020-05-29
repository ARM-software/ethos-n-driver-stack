//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "StrategyX.hpp"

#include <ethosn_command_stream/CommandData.hpp>

#include <cassert>

namespace ethosn
{
namespace support_library
{

bool IsStrategyX(const MceOperationNode& node)
{
    bool isSupportedMceOperation{};
    {
        isSupportedMceOperation = (node.GetOperation() == command_stream::MceOperation::CONVOLUTION);
    }

    bool isSupportedUpsampling{};
    {
        isSupportedUpsampling = (node.GetUpscaleFactor() == 1U);
    }

    return isSupportedMceOperation && isSupportedUpsampling;
}

}    // namespace support_library
}    // namespace ethosn
