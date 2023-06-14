//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "DebuggableObject.hpp"

#include <string>

namespace ethosn
{
namespace support_library
{

std::atomic<int> DebuggableObject::ms_IdCounter(0);

DebuggableObject::DebuggableObject(const char* defaultTagPrefix)
{
    m_DebugId = ms_IdCounter++;
    // Generate an arbitrary and unique (but deterministic) default debug tag for this object.
    // This means that if no-one sets anything more useful, we still have a way to identify it.
    m_DebugTag = std::string(defaultTagPrefix) + " " + std::to_string(m_DebugId).c_str();
}

DebuggableObject::DebuggableObject(ExplicitDebugTag, const char* debugTag)
    : m_DebugTag(debugTag)
{
    //m_DebugId is very useful for conditional breakpoints
    m_DebugId = ms_IdCounter++;
}

}    // namespace support_library
}    // namespace ethosn
