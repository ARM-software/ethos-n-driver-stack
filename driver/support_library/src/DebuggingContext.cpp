//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "DebuggingContext.hpp"
#include "Graph.hpp"

#include <fstream>

namespace ethosn
{
namespace support_library
{

DebuggingContext::DebuggingContext(const CompilationOptions::DebugInfo& compilationOptions)
    : m_DebugInfo(compilationOptions)
{}

void DebuggingContext::DumpGraph(const Graph& graph, const std::string& fileName) const
{
    if (m_DebugInfo.m_DumpDebugFiles)
    {
        std::ofstream dotStream(GetAbsolutePathOutputFileName(fileName));
        graph.DumpToDotFormat(dotStream);
    }
}

void DebuggingContext::SaveGraphToDot(const Graph& graph,
                                      const GraphOfParts* graphOfParts,
                                      const std::string& fileName,
                                      DetailLevel detailLevel) const
{
    if (m_DebugInfo.m_DumpDebugFiles)
    {
        std::ofstream stream(GetAbsolutePathOutputFileName(fileName));
        ethosn::support_library::SaveGraphToDot(graph, graphOfParts, stream, detailLevel);
    }
}

void DebuggingContext::SavePlansToDot(const Part& part, const std::string& fileName, DetailLevel detailLevel) const
{
    if (m_DebugInfo.m_DumpDebugFiles)
    {
        std::ofstream stream(GetAbsolutePathOutputFileName(fileName));
        ethosn::support_library::SavePlansToDot(part, stream, detailLevel);
    }
}

void DebuggingContext::SaveOpGraphToDot(const OpGraph& opGraph,
                                        const std::string& fileName,
                                        DetailLevel detailLevel) const
{
    if (m_DebugInfo.m_DumpDebugFiles)
    {
        std::ofstream stream(GetAbsolutePathOutputFileName(fileName));
        ethosn::support_library::SaveOpGraphToDot(opGraph, stream, detailLevel);
    }
}

void DebuggingContext::SaveCombinationToDot(const Combination& combination,
                                            const GraphOfParts& graphOfParts,
                                            const std::string& fileName,
                                            DetailLevel detailLevel) const
{
    if (m_DebugInfo.m_DumpDebugFiles)
    {
        std::ofstream stream(GetAbsolutePathOutputFileName(fileName));
        ethosn::support_library::SaveCombinationToDot(combination, graphOfParts, stream, detailLevel);
    }
}

std::string DebuggingContext::GetAbsolutePathOutputFileName(const std::string& fileName) const
{
    std::string debugOutputFile("");
    if (!m_DebugInfo.m_DebugDir.empty())
    {
        debugOutputFile.append(m_DebugInfo.m_DebugDir + '/');
    }
    debugOutputFile.append(fileName);

    return debugOutputFile;
}

}    // namespace support_library
}    // namespace ethosn
