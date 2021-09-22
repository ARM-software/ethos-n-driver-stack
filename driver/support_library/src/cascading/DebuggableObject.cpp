//
// Copyright © 2021 Arm Limited.
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
{
    // Generate an arbitrary and unique (but deterministic) default debug tag for this object.
    // This means that if no-one sets anything more useful, we still have a way to identify it.
    m_DebugTag = std::string(defaultTagPrefix) + " " + std::to_string(ms_IdCounter);
    //m_DebugId is very useful for conditional breakpoints
    m_DebugId = ms_IdCounter;
    ++ms_IdCounter;
}

}    // namespace support_library
}    // namespace ethosn
