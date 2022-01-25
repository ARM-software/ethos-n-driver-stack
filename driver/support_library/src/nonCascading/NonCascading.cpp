//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "NonCascading.hpp"

#include "DebuggingContext.hpp"
#include "Graph.hpp"
#include "Utils.hpp"
#include "ethosn_support_library/Support.hpp"

#include <fstream>
#include <sstream>

namespace ethosn
{

namespace support_library
{

NetworkPerformanceData NonCascadingEstimate(Graph& graph, const EstimationOptions& estOpt)
{
    NetworkPerformanceData performanceStream;
    std::vector<Node*> sorted = graph.GetNodesSorted();

    for (Node* n : sorted)
    {
        if (!n->IsPrepared())
        {
            std::stringstream result;
            for (auto id : n->GetCorrespondingOperationIds())
            {
                result << " " << id;
            }
            g_Logger.Error("Failed to prepare operation:%s", result.str().c_str());
        }
        n->Estimate(performanceStream, estOpt);
    }

    return performanceStream;
}

}    //namespace support_library

}    // namespace ethosn
