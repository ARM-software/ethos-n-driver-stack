//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "DebuggingContext.hpp"
#include "Graph.hpp"

#include <cassert>
#include <fstream>

namespace ethosn
{
namespace support_library
{

DebuggingContext::DebuggingContext(const CompilationOptions::DebugInfo& debugInfo)
    : m_DebugInfo(debugInfo)
{}

void DebuggingContext::SaveNetworkToDot(CompilationOptions::DebugLevel level,
                                        const Network& network,
                                        const std::string& fileName,
                                        DetailLevel detailLevel) const
{
    if (m_DebugInfo.m_DumpDebugFiles >= level)
    {
        std::ofstream dotStream(GetAbsolutePathOutputFileName(fileName));
        ethosn::support_library::SaveNetworkToDot(network, dotStream, detailLevel);
    }
}

void DebuggingContext::DumpGraph(CompilationOptions::DebugLevel level,
                                 const Graph& graph,
                                 const std::string& fileName) const
{
    if (m_DebugInfo.m_DumpDebugFiles >= level)
    {
        std::ofstream dotStream(GetAbsolutePathOutputFileName(fileName));
        graph.DumpToDotFormat(dotStream);
    }
}

void DebuggingContext::SaveGraphOfPartsToDot(CompilationOptions::DebugLevel level,
                                             const GraphOfParts& graphOfParts,
                                             const std::string& fileName,
                                             DetailLevel detailLevel) const
{
    if (m_DebugInfo.m_DumpDebugFiles >= level)
    {
        std::ofstream stream(GetAbsolutePathOutputFileName(fileName));
        ethosn::support_library::SaveGraphOfPartsToDot(graphOfParts, stream, detailLevel);
    }
}

void DebuggingContext::SavePlansToDot(CompilationOptions::DebugLevel level,
                                      const Plans& plans,
                                      const std::string& fileName,
                                      DetailLevel detailLevel) const
{
    if (m_DebugInfo.m_DumpDebugFiles >= level)
    {
        std::ofstream stream(GetAbsolutePathOutputFileName(fileName));
        ethosn::support_library::SavePlansToDot(plans, stream, detailLevel);
    }
}

void DebuggingContext::SaveOpGraphToDot(CompilationOptions::DebugLevel level,
                                        const OpGraph& opGraph,
                                        const std::string& fileName,
                                        DetailLevel detailLevel) const
{
    if (m_DebugInfo.m_DumpDebugFiles >= level)
    {
        std::ofstream stream(GetAbsolutePathOutputFileName(fileName));
        ethosn::support_library::SaveOpGraphToDot(opGraph, stream, detailLevel);
    }
}

void DebuggingContext::SaveEstimatedOpGraphToDot(CompilationOptions::DebugLevel level,
                                                 const OpGraph& opGraph,
                                                 const EstimatedOpGraph& estimationDetails,
                                                 const std::string& fileName,
                                                 DetailLevel detailLevel) const
{
    if (m_DebugInfo.m_DumpDebugFiles >= level)
    {
        std::ofstream stream(GetAbsolutePathOutputFileName(fileName));
        ethosn::support_library::SaveEstimatedOpGraphToDot(opGraph, estimationDetails, stream, detailLevel);
    }
}

void DebuggingContext::SaveCombinationToDot(CompilationOptions::DebugLevel level,
                                            const Combination& combination,
                                            const std::string& fileName,
                                            DetailLevel detailLevel) const
{
    if (m_DebugInfo.m_DumpDebugFiles >= level)
    {
        std::ofstream stream(GetAbsolutePathOutputFileName(fileName));
        ethosn::support_library::SaveCombinationToDot(combination, stream, detailLevel);
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

void DebuggingContext::AddNodeCreationSource(DebuggingContext::NodeToCreateSourceTuple tuple)
{
    this->m_NodeToCreationSource[tuple.node] = tuple.creationSource;
}

const std::string& DebuggingContext::GetStringFromNode(const Node* node) const
{
    const void* ptrToFind = static_cast<const void*>(node);
    static const std::string unknown("unknown");
    const NodeToCreationSourceContainer::const_iterator it = m_NodeToCreationSource.find(ptrToFind);
    if (it == m_NodeToCreationSource.end())
    {
        return unknown;
    }
    const std::string& value = it->second;
    return value;
}

}    // namespace support_library
}    // namespace ethosn
