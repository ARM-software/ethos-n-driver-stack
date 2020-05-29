//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../include/ethosn_support_library/Support.hpp"
#include "cascading/Visualisation.hpp"

#include <string>

namespace ethosn
{
namespace support_library
{

class Graph;
class GraphOfParts;
struct DebuggingContext
{
public:
    DebuggingContext(const CompilationOptions::DebugInfo& compilationOptions);
    void DumpGraph(const Graph& graph, const std::string& fileName) const;
    void SaveGraphToDot(const Graph& graph,
                        const GraphOfParts* graphOfParts,
                        const std::string& fileName,
                        DetailLevel detailLevel) const;
    void SavePlansToDot(const Part& part, const std::string& fileName, DetailLevel detailLevel) const;
    void SaveOpGraphToDot(const OpGraph& opGraph, const std::string& fileName, DetailLevel detailLevel) const;
    void SaveCombinationToDot(const Combination& combination,
                              const GraphOfParts& graphOfParts,
                              const std::string& fileName,
                              DetailLevel detailLevel) const;

    const CompilationOptions::DebugInfo& m_DebugInfo;

    std::string GetAbsolutePathOutputFileName(const std::string& fileName) const;
};

}    // namespace support_library
}    // namespace ethosn
