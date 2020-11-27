//
// Copyright © 2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

namespace ethosn
{
namespace utils
{

/// Converts each entry in the given 'entries' container to a string using the given 'toStringFunc'
/// functor, and joins them together into a list using the given 'separator'.
template <typename TContainer, typename TFunc>
std::string Join(const char* separator, const TContainer& entries, TFunc toStringFunc)
{
    std::string result;
    bool first = true;
    for (const auto& x : entries)
    {
        if (!first)
        {
            result += separator;
        }
        result += toStringFunc(x);
    }
    return result;
}

inline std::string ReplaceAll(std::string str, const std::string& from, const std::string& to)
{
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos)
    {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();    // Handles case where 'to' is a substring of 'from'
    }
    return str;
}

}    // namespace utils
}    // namespace ethosn
