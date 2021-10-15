//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

namespace ethosn
{
namespace support_library
{

struct DebuggableObject
{
public:
    DebuggableObject(const char* defaultTagPrefix);

    /// This can be used to help identify this object for debugging purposes, and is used in visulisations (dot files)
    /// to identify this object. It shouldn't have any effect on network compilation or estimation.
    std::string m_DebugTag;
    int m_DebugId;

    /// Counter for generating unique debug tags (see DebuggableObject constructor).
    /// This is publicly exposed so can be manipulated by tests.
    static int ms_IdCounter;
};

}    // namespace support_library
}    // namespace ethosn