//
// Copyright © 2020-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../include/ethosn_support_library/Support.hpp"
#include "CombinerDFS.hpp"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ethosn
{
namespace support_library
{

struct EstimatedPass
{
    PassStats m_Stats;
    /// The Ops included in this pass.
    std::vector<Op*> m_Ops;
};

EstimatedPass EstimateConversionPassGrownFrom(const OpGraph& opGraph,
                                              Op* op,
                                              const EstimationOptions& estimationOpts,
                                              std::unordered_set<Op*>& unestimatedOps);

EstimatedPass EstimatePassGrownFrom(const OpGraph& opGraph,
                                    Op* op,
                                    const HardwareCapabilities& capabilities,
                                    const EstimationOptions& estimationOpts,
                                    std::unordered_set<Op*>& unestimatedOps);

/// Result of estimating the performance of an OpGraph.
struct EstimatedOpGraph
{
    double m_Metric;
    NetworkPerformanceData m_PerfData;
    /// For each Op in the OpGraph that was estimated, which Pass in the NetworkPerformanceData it was included in.
    std::unordered_map<Op*, uint32_t> m_OpToPass;
};

EstimatedOpGraph EstimateOpGraph(const OpGraph& opGraph,
                                 const HardwareCapabilities& capabilities,
                                 const EstimationOptions& estimationOpts);

}    // namespace support_library
}    // namespace ethosn
