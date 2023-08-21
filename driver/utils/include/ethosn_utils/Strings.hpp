//
// Copyright © 2020,2022-2023 Arm Limited.
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
        first = false;
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

static inline std::string Rtrim(const std::string& s)
{
    return s.substr(0, s.find_last_not_of(" \n\r\t") + 1);
}

static inline std::string Trim(const std::string& str)
{
    size_t start = str.find_first_not_of(" \n\r\t");
    if (start == std::string::npos)
    {
        start = 0;
    }
    size_t end = str.find_last_not_of(" \n\r\t");
    if (end == std::string::npos)
    {
        end = str.length();
    }
    return str.substr(start, end - start + 1);
}

static inline std::vector<std::string> Split(const std::string& s, const std::string& delim)
{
    std::vector<std::string> result;
    size_t pos = 0;
    while (true)
    {
        size_t newPos = s.find(delim, pos);
        if (newPos == std::string::npos)
        {
            result.push_back(s.substr(pos));
            break;
        }
        else
        {
            result.push_back(s.substr(pos, newPos - pos));
        }
        pos = newPos + 1;
    }

    return result;
}

inline bool EndsWith(const std::string& s, const std::string& q)
{
    if (s.size() < q.size())
    {
        return false;
    }
    return std::equal(s.end() - q.size(), s.end(), q.begin());
}

inline bool StartsWith(const std::string& string, const std::string& substring)
{
    if (string.size() < substring.size())
    {
        return false;
    }
    return std::equal(substring.begin(), substring.end(), string.begin());
}

}    // namespace utils
}    // namespace ethosn
