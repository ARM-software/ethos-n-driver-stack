//
// Copyright © 2020-2024 Arm Limited.
// Copyright © 2024 Axis Communications AB.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../include/ethosn_support_library/Support.hpp"
#include "OpGraph.hpp"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ethosn
{
namespace support_library
{

struct EstimatedPass
{
    /// The estimated cycle count for this pass.
    double m_Metric;
    /// Additional information helpful for debugging the performance estimation, shown in dot files.
    PassDebugStats m_PassDebugStat = {};
    std::string m_DebugInfo;
    /// The Ops included in this pass.
    std::vector<Op*> m_Ops;

    /// Performance data in a format consumable by SPA, which is deprecated.
    PassStats m_LegacyStats;
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
    /// The total estimated cycle count for the entire OpGraph.
    double m_Metric;
    std::vector<EstimatedPass> m_Passes;
    /// Performance data in a format consumable by SPA, which is deprecated.
    NetworkPerformanceData m_LegacyPerfData;
    /// For each Op in the OpGraph that was estimated, which Pass in the m_Passes/m_LegacyPerfData it was included in.
    std::unordered_map<Op*, uint32_t> m_OpToPass;
};

EstimatedOpGraph EstimateOpGraph(const OpGraph& opGraph,
                                 const HardwareCapabilities& capabilities,
                                 const EstimationOptions& estimationOpts);

}    // namespace support_library
}    // namespace ethosn
