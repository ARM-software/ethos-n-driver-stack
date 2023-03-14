//
// Copyright © 2018-2023 Arm Limited.
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

namespace cascading_compiler
{
struct CompiledOpGraph;
}

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
    DebuggingContext(const CompilationOptions::DebugInfo& debugInfo);

    void Save(CompilationOptions::DebugLevel level,
              const std::string& fileName,
              std::function<void(std::ofstream&)> savingFunc) const;

    CompilationOptions::DebugInfo m_DebugInfo;

    std::string GetAbsolutePathOutputFileName(const std::string& fileName) const;

    const std::string& GetStringFromNode(const Node* m) const;
    void AddNodeCreationSource(NodeToCreateSourceTuple tuple);

    uint32_t GetMaxNumDumps() const
    {
        return 100U;
    }

    uint64_t m_TotalWeightCompressionTime;

private:
    using NodeToCreationSourceContainer = std::unordered_map<const void*, std::string>;
    NodeToCreationSourceContainer m_NodeToCreationSource;
};

}    // namespace support_library
}    // namespace ethosn
