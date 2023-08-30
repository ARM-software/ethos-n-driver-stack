//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "DebuggingContext.hpp"

#include <ethosn_utils/Strings.hpp>

#include <cassert>
#include <fstream>

namespace ethosn
{
namespace support_library
{

DebuggingContext::DebuggingContext(const CompilationOptions::DebugInfo& debugInfo)
    : m_DebugInfo(debugInfo)
{
    // Load the preferred DRAM formats debugging config
    const char* env = std::getenv("ETHOSN_SUPPORT_LIBRARY_DEBUG_PREFERRED_DRAM_FORMATS");
    if (env && strlen(env) > 0)
    {
        // The config file has a simple format with each line defining a set of part IDs and the preferred format
        //
        // A simple example:
        //
        // 1,10,12: FCAF_WIDE
        // 15: NHWCB

        std::ifstream file(env);
        if (!file.good())
        {
            throw std::runtime_error("Error opening preferred DRAM formats file: " + std::string(env));
        }

        std::string line;
        uint32_t lineNumber = 0;
        auto reportError    = [&lineNumber](std::string msg) {
            throw std::runtime_error("Error in preferred DRAM formats file at line " + std::to_string(lineNumber) +
                                     ": " + std::move(msg));
        };

        while (getline(file, line))
        {
            ++lineNumber;
            line = ethosn::utils::Trim(line);
            if (line.empty() || line[0] == '#')
            {
                // Empty (or whitespace) lines or comments - ignore
                continue;
            }

            std::vector<std::string> parts = ethosn::utils::Split(line, ":");
            if (parts.size() != 2)
            {
                reportError("Expected exactly one colon (':')");
            }
            std::string partIdsString = ethosn::utils::Trim(parts[0]);
            std::string formatString  = ethosn::utils::Trim(parts[1]);

            std::set<PartId> partIds;
            for (std::string partIdString : ethosn::utils::Split(partIdsString, ","))
            {
                try
                {
                    partIds.insert(std::stoi(ethosn::utils::Trim(partIdString)));
                }
                catch (const std::logic_error&)
                {
                    reportError(std::string("Invalid part ID: ") + partIdString);
                }
            }

            BufferFormat format = BufferFormat::WEIGHT;    // Something invalid
            if (formatString == "NHWCB")
            {
                format = BufferFormat::NHWCB;
            }
            else if (formatString == "NHWC")
            {
                format = BufferFormat::NHWC;
            }
            else if (formatString == "FCAF_WIDE")
            {
                format = BufferFormat::FCAF_WIDE;
            }
            else if (formatString == "FCAF_DEEP")
            {
                format = BufferFormat::FCAF_DEEP;
            }
            else
            {
                reportError(std::string("Invalid DRAM format: ") + formatString);
            }

            std::string key             = ethosn::utils::Join(",", partIds, [](PartId p) { return std::to_string(p); });
            m_PreferredDramFormats[key] = format;
        }
    }
}

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

utils::Optional<BufferFormat> DebuggingContext::GetPreferredDramFormat(const std::set<PartId>& partIds) const
{
    std::string key = ethosn::utils::Join(",", partIds, [](PartId p) { return std::to_string(p); });
    auto it         = m_PreferredDramFormats.find(key);
    return it == m_PreferredDramFormats.end() ? utils::EmptyOptional{} : utils::Optional<BufferFormat>(it->second);
}

}    // namespace support_library
}    // namespace ethosn
