//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Compiler.hpp"
#include "Strategies.hpp"

#include <ethosn_command_stream/CommandData.hpp>
#include <ethosn_command_stream/CommandStreamBuffer.hpp>

#include <cstdint>
#include <memory>
#include <unordered_map>

namespace ethosn
{
namespace support_library
{

class Pass;

class Section
{
public:
    Section(std::string id, command_stream::SectionType type, Pass* pass)
        : m_Id(id)
        , m_IsGenerated(false)
        , m_Passes()
        , m_SectionType(type)
    {
        m_Passes.push_back(pass);
    }

    std::string GetId()
    {
        return m_Id;
    }

    bool IsGenerated()
    {
        return m_IsGenerated;
    }

    const std::vector<Pass*>& GetPasses() const
    {
        return m_Passes;
    }

    /// Generates this Section by adding section delimiter to the given command stream
    void Generate(command_stream::CommandStreamBuffer& cmdStream);

    DotAttributes GetDotAttributes();

private:
    std::string m_Id;
    bool m_IsGenerated;
    std::vector<Pass*> m_Passes;
    command_stream::SectionType m_SectionType;
};

}    // namespace support_library
}    // namespace ethosn
