//
// Copyright © 2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

inline bool IsLittleEndian()
{
    uint32_t i = 1;
    uint8_t* j = reinterpret_cast<uint8_t*>(&i);
    return *j == 1;
}