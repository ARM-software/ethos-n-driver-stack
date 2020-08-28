//
// Copyright © 22020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../include/ethosn_support_library/Support.hpp"
#include "Combiner.hpp"

#include <unordered_set>

namespace ethosn
{
namespace support_library
{

PassStats EstimatePassGrownFrom(const OpGraph& opGraph,
                                Op* op,
                                const HardwareCapabilities& capabilities,
                                const EstimationOptions& estimationOpts,
                                std::unordered_set<Op*>& unestimatedOps);

NetworkPerformanceData EstimateOpGraph(const OpGraph& opGraph,
                                       const HardwareCapabilities& capabilities,
                                       const EstimationOptions& estimationOpts);

}    // namespace support_library
}    // namespace ethosn
