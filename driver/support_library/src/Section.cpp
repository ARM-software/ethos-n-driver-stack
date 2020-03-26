//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "Section.hpp"

#include "Compiler.hpp"
#include "Utils.hpp"

#include <vector>

namespace ethosn
{
namespace support_library
{

void Section::Generate(command_stream::CommandStreamBuffer& cmdStream)
{
    if (!m_IsGenerated)
    {
        ethosn::command_stream::Section secCmd;
        secCmd.m_Type() = m_SectionType;
        cmdStream.EmplaceBack(secCmd);
    }

    m_IsGenerated = true;
}

DotAttributes Section::GetDotAttributes()
{
    DotAttributes result = { "Section " + m_Id, "blue" };

    switch (m_SectionType)
    {
        case command_stream::SectionType::SISO:
            result.m_Label += " (SISO)";
            break;
        case command_stream::SectionType::SISO_CASCADED:
            result.m_Label += " (SISO_CASCADED)";
            break;
        case command_stream::SectionType::SIMO:
            result.m_Label += " (SIMO)";
            break;
        case command_stream::SectionType::SIMO_CASCADED:
            result.m_Label += " (SIMO_CASCADED)";
            break;
        case command_stream::SectionType::SISO_BRANCHED_CASCADED:
            result.m_Label += " (SISO_BRANCHED_CASCADED)";
            break;
        case command_stream::SectionType::MISO:
            result.m_Label += " (MISO)";
            break;
        default:
            break;
    }
    return result;
}

}    // namespace support_library
}    // namespace ethosn
