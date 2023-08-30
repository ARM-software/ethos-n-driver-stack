//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../include/ethosn_support_library/Support.hpp"
#include "Part.hpp"
#include "Visualisation.hpp"

#include <memory>
#include <string>
#include <unordered_map>

namespace ethosn
{
namespace support_library
{

class GraphOfParts;
class Node;
struct DebuggingContext
{
public:
    DebuggingContext(const CompilationOptions::DebugInfo& debugInfo);

    void Save(CompilationOptions::DebugLevel level,
              const std::string& fileName,
              std::function<void(std::ofstream&)> savingFunc) const;

    CompilationOptions::DebugInfo m_DebugInfo;

    std::string GetAbsolutePathOutputFileName(const std::string& fileName) const;

    utils::Optional<BufferFormat> GetPreferredDramFormat(const std::set<PartId>& partIds) const;

private:
    /// For debugging, this can be used to store the preferred DRAM format (e.g. NHWCB, FCAF_WIDE)
    /// for a glue buffer which connects a particular set of parts.
    /// The key is a string with the part IDs joined together, e.g. "1,10,12"
    std::unordered_map<std::string, BufferFormat> m_PreferredDramFormats;
};

}    // namespace support_library
}    // namespace ethosn
