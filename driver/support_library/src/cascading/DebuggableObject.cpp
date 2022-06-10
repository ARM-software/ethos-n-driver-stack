//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "DebuggableObject.hpp"

#include <string>

namespace ethosn
{
namespace support_library
{

int DebuggableObject::ms_IdCounter = 0;

DebuggableObject::DebuggableObject(const char* defaultTagPrefix)
    // Generate an arbitrary and unique (but deterministic) default debug tag for this object.
    // This means that if no-one sets anything more useful, we still have a way to identify it.
    // Forward to the explicit debug tag constructor.
    : DebuggableObject(ExplicitDebugTag(), (std::string(defaultTagPrefix) + " " + std::to_string(ms_IdCounter)).c_str())
{}

DebuggableObject::DebuggableObject(ExplicitDebugTag, const char* debugTag)
    : m_DebugTag(debugTag)
{
    //m_DebugId is very useful for conditional breakpoints
    m_DebugId = ms_IdCounter;
    ++ms_IdCounter;
}

}    // namespace support_library
}    // namespace ethosn
