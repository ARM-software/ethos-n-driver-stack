//
// Copyright © 2020-2021 Arm Limited.
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
/// This may be incomplete - i.e. some parts of the OpGraph may not have been estimated due to missing features
/// in EstimateOpGraph(). Use IsComplete() to check this and handle as necessary.
struct EstimatedOpGraph
{
    NetworkPerformanceData m_PerfData;
    /// For each Op in the OpGraph that was estimated, which Pass in the NetworkPerformanceData it was included in.
    std::unordered_map<Op*, uint32_t> m_OpToPass;

    std::unordered_set<Op*> m_UnestimatedOps;    ///< Any Ops that couldn't be estimated.

    bool IsComplete() const
    {
        return m_UnestimatedOps.size() == 0;
    }
};

EstimatedOpGraph EstimateOpGraph(const OpGraph& opGraph,
                                 const HardwareCapabilities& capabilities,
                                 const EstimationOptions& estimationOpts);

}    // namespace support_library
}    // namespace ethosn
