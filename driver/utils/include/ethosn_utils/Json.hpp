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

struct Indent
{
    explicit constexpr Indent(const size_t depth)
        : m_Depth(depth)
    {}

    constexpr operator size_t&()
    {
        return m_Depth;
    }

    constexpr operator size_t() const
    {
        return m_Depth;
    }

    size_t m_Depth;
};

inline std::ostream& operator<<(std::ostream& os, const Indent& indent)
{
    for (size_t i = 0; i < indent; ++i)
    {
        os << '\t';
    }

    return os;
}

template <typename T>
struct QuotedT
{
    explicit constexpr QuotedT(const T& value)
        : m_Value(value)
    {}

    const T& m_Value;
};

template <typename T>
QuotedT<T> Quoted(const T& value)
{
    return QuotedT<T>(value);
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const QuotedT<T>& field)
{
    return os << '"' << field.m_Value << '"';
}

template <typename T>
struct JsonFieldT
{
    explicit constexpr JsonFieldT(const T& value)
        : m_Value(value)
    {}

    const T& m_Value;
};

template <typename T>
JsonFieldT<T> JsonField(const T& value)
{
    return JsonFieldT<T>(value);
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const JsonFieldT<T>& field)
{
    return os << Quoted(field.m_Value) << ':';
}

template <typename T>
struct JsonArrayT
{
    explicit constexpr JsonArrayT(const T& value)
        : m_Value(value)
    {}

    const T& m_Value;
};

template <typename T>
JsonArrayT<T> JsonArray(const T& value)
{
    return JsonArrayT<T>(value);
}

template <typename T, typename PrintFn>
std::ostream& Print(
    std::ostream& os, const Indent indent, const JsonArrayT<T>& array, PrintFn&& printFn, const bool multiline = false)
{
    const char sep = multiline ? '\n' : ' ';

    os << indent << '[' << sep;

    for (auto it = array.m_Value.begin(); it != array.m_Value.end(); ++it)
    {
        printFn(os, *it);

        if (it != std::prev(array.m_Value.end()))
        {
            os << ',';
        }

        os << sep;
    }

    if (multiline)
    {
        os << indent;
    }

    os << ']';

    return os;
}

template <typename T>
std::ostream& Print(std::ostream& os, const Indent indent, const JsonArrayT<T>& array, const bool multiline = false)
{
    return Print(os, indent, array, [](std::ostream& os, const auto& value) { os << value; }, multiline);
}

}    // namespace utils
}    // namespace ethosn
