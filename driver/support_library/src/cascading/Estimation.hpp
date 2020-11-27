//
// Copyright © 2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../include/ethosn_support_library/Support.hpp"
#include "Combiner.hpp"

#include <unordered_map>
#include <unordered_set>

namespace ethosn
{
namespace support_library
{

struct EstimatedPass
{
    PassStats m_Stats;
    /// The Ops included in this pass.
    std::unordered_set<Op*> m_Ops;
};

EstimatedPass EstimatePassGrownFrom(const OpGraph& opGraph,
                                    Op* op,
                                    const HardwareCapabilities& capabilities,
                                    const EstimationOptions& estimationOpts,
                                    std::unordered_set<Op*>& unestimatedOps);

struct EstimatedOpGraph
{
    NetworkPerformanceData m_PerfData;
    /// For each Op in the OpGraph that was estimated, which Pass in the NetworkPerformanceData it was included in.
    std::unordered_map<Op*, uint32_t> m_OpToPass;
};

EstimatedOpGraph EstimateOpGraph(const OpGraph& opGraph,
                                 const HardwareCapabilities& capabilities,
                                 const EstimationOptions& estimationOpts);

}    // namespace support_library
}    // namespace ethosn
