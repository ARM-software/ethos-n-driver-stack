//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "DebuggingContext.hpp"

#include <cassert>
#include <fstream>

namespace ethosn
{
namespace support_library
{

DebuggingContext::DebuggingContext(const CompilationOptions::DebugInfo& debugInfo)
    : m_DebugInfo(debugInfo)
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

}    // namespace support_library
}    // namespace ethosn
