//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../include/ethosn_support_library/Support.hpp"
#include "cascading/Visualisation.hpp"

#include <memory>
#include <string>
#include <unordered_map>

namespace ethosn
{
namespace support_library
{

class Graph;
class GraphOfParts;
class Node;
struct DebuggingContext
{
public:
    struct NodeToCreateSourceTuple
    {
        const Node* node;
        std::string creationSource;
    };
    DebuggingContext(const CompilationOptions::DebugInfo* compilationOptions);
    void DumpGraph(CompilationOptions::DebugLevel level, const Graph& graph, const std::string& fileName) const;
    void SaveGraphToDot(CompilationOptions::DebugLevel level,
                        const Graph& graph,
                        const GraphOfParts* graphOfParts,
                        const std::string& fileName,
                        DetailLevel detailLevel) const;
    void SavePlansToDot(CompilationOptions::DebugLevel level,
                        const Part& part,
                        const std::string& fileName,
                        DetailLevel detailLevel) const;
    void SaveOpGraphToDot(CompilationOptions::DebugLevel level,
                          const OpGraph& opGraph,
                          const std::string& fileName,
                          DetailLevel detailLevel) const;
    void SaveEstimatedOpGraphToDot(CompilationOptions::DebugLevel level,
                                   const OpGraph& opGraph,
                                   const EstimatedOpGraph& estimationDetails,
                                   const std::string& fileName,
                                   DetailLevel detailLevel) const;
    void SaveCombinationToDot(CompilationOptions::DebugLevel level,
                              const Combination& combination,
                              const GraphOfParts& graphOfParts,
                              const std::string& fileName,
                              DetailLevel detailLevel) const;

    const CompilationOptions::DebugInfo* m_DebugInfo;

    std::string GetAbsolutePathOutputFileName(const std::string& fileName) const;

    const std::string& GetStringFromNode(const Node* m) const;
    void AddNodeCreationSource(NodeToCreateSourceTuple tuple);

    uint32_t GetMaxNumDumps() const
    {
        return 100U;
    }

private:
    using NodeToCreationSourceContainer = std::unordered_map<const void*, std::string>;
    NodeToCreationSourceContainer m_NodeToCreationSource;
};

void SetDebuggingContext(const DebuggingContext& debuggingContext);
DebuggingContext& GetDebuggingContext();
const DebuggingContext& GetConstDebuggingContext();

}    // namespace support_library
}    // namespace ethosn
