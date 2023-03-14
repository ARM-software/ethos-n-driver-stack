//
// Copyright © 2018-2023 Arm Limited.
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
    , m_TotalWeightCompressionTime(0)
{}

void DebuggingContext::Save(CompilationOptions::DebugLevel level,
                            const std::string& fileName,
                            std::function<void(std::ofstream&)> savingFunc) const
{
    if (m_DebugInfo.m_DumpDebugFiles >= level)
    {
        std::ofstream dotStream(GetAbsolutePathOutputFileName(fileName));
        savingFunc(dotStream);
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
