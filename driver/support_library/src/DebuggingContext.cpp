//
// Copyright © 2018-2021 Arm Limited.
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

// s_DebuggingContext is declared as thread_local to be able to support
// parallel network compilation on different threads. Since the DebuggingContext
// object is set by the Compiler object, this allows each compilation to have
// its own debugging information.
static thread_local DebuggingContext s_DebuggingContext(nullptr);

DebuggingContext::DebuggingContext(const CompilationOptions::DebugInfo* compilationOptions)
    : m_DebugInfo(compilationOptions)
{}

void DebuggingContext::SaveNetworkToDot(CompilationOptions::DebugLevel level,
                                        const Network& network,
                                        const std::string& fileName,
                                        DetailLevel detailLevel) const
{
    if (m_DebugInfo->m_DumpDebugFiles >= level)
    {
        std::ofstream dotStream(GetAbsolutePathOutputFileName(fileName));
        ethosn::support_library::SaveNetworkToDot(network, dotStream, detailLevel);
    }
}

void DebuggingContext::DumpGraph(CompilationOptions::DebugLevel level,
                                 const Graph& graph,
                                 const std::string& fileName) const
{
    if (m_DebugInfo->m_DumpDebugFiles >= level)
    {
        std::ofstream dotStream(GetAbsolutePathOutputFileName(fileName));
        graph.DumpToDotFormat(dotStream);
    }
}

void DebuggingContext::SaveGraphToDot(CompilationOptions::DebugLevel level,
                                      const GraphOfParts& graphOfParts,
                                      const std::string& fileName,
                                      DetailLevel detailLevel) const
{
    if (m_DebugInfo->m_DumpDebugFiles >= level)
    {
        std::ofstream stream(GetAbsolutePathOutputFileName(fileName));
        ethosn::support_library::SaveGraphToDot(graphOfParts, stream, detailLevel);
    }
}

void DebuggingContext::SavePlansToDot(CompilationOptions::DebugLevel level,
                                      const Plans& plans,
                                      const std::string& fileName,
                                      DetailLevel detailLevel) const
{
    if (m_DebugInfo->m_DumpDebugFiles >= level)
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
    if (m_DebugInfo->m_DumpDebugFiles >= level)
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
    if (m_DebugInfo->m_DumpDebugFiles >= level)
    {
        std::ofstream stream(GetAbsolutePathOutputFileName(fileName));
        ethosn::support_library::SaveEstimatedOpGraphToDot(opGraph, estimationDetails, stream, detailLevel);
    }
}

void DebuggingContext::SaveCombinationToDot(CompilationOptions::DebugLevel level,
                                            const Combination& combination,
                                            const GraphOfParts& graphOfParts,
                                            const std::string& fileName,
                                            DetailLevel detailLevel) const
{
    if (m_DebugInfo->m_DumpDebugFiles >= level)
    {
        std::ofstream stream(GetAbsolutePathOutputFileName(fileName));
        ethosn::support_library::SaveCombinationToDot(combination, graphOfParts, stream, detailLevel);
    }
}

std::string DebuggingContext::GetAbsolutePathOutputFileName(const std::string& fileName) const
{
    std::string debugOutputFile("");
    if (!m_DebugInfo->m_DebugDir.empty())
    {
        debugOutputFile.append(m_DebugInfo->m_DebugDir + '/');
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

void SetDebuggingContext(const DebuggingContext& debuggingContext)
{
    s_DebuggingContext = debuggingContext;
}

DebuggingContext& GetDebuggingContext()
{
    return const_cast<DebuggingContext&>(GetConstDebuggingContext());
}

const DebuggingContext& GetConstDebuggingContext()
{
    static const CompilationOptions::DebugInfo defaultDebugInfo;
    if (!s_DebuggingContext.m_DebugInfo)
    {
        s_DebuggingContext = &defaultDebugInfo;
    }
    return s_DebuggingContext;
}

}    // namespace support_library
}    // namespace ethosn
