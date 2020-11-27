//
// Copyright © 2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if defined(__unix__)
#include <sys/stat.h>
#elif defined(_MSC_VER)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif

namespace ethosn
{
namespace utils
{

inline bool MakeDirectory(const char* dir)
{
#if defined(__unix__)
    return mkdir(dir, 0777) == 0;
#elif defined(_MSC_VER)
    return CreateDirectory(dir, nullptr);
#else
    return false;
#endif
}

}    // namespace utils
}    // namespace ethosn
