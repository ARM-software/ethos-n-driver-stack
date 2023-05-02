//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <array>
#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace ethosn
{
namespace control_unit
{

/// A fixed length buffer of characters to store a null-terminated string.
/// Provides methods to manipulate the string without overflowing the buffer.
/// The template parameter Capacity determines that maximum length of the string,
/// excluding the null-terminator.
/// A size/length value is maintained, indicating how much of the buffer is valid,
/// and this should always be consistent with the null-terminator. This speeds
/// up operations like appending, as we know where to start appending the new data.
template <size_t Capacity>
class FixedString
{
public:
    FixedString()
    {
        Clear();
    }
    explicit FixedString(const char* s)
    {
        m_Size = std::min(strlen(s), Capacity);
        // GCC 8 would give a warning here because we're potentially truncating s
#if __GNUC__ >= 8
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-truncation"
#endif
        strncpy(m_Buffer.data(), s, m_Size + 1);
#if __GNUC__ >= 8
#pragma GCC diagnostic pop
#endif
        m_Buffer[m_Size] = '\0';
    }

    template <typename... TArgs>
    static FixedString Format(const char* rhs, TArgs... args)
    {
        FixedString result;
        result.AppendFormat(rhs, std::forward<TArgs>(args)...);
        return result;
    }

    /// Gets the maximum possible length of the string, excluding the null-terminator.
    constexpr size_t GetCapacity()
    {
        return Capacity;
    }

    /// Gets the current length of the string, excluding the null-terminator.
    size_t GetSize() const
    {
        return m_Size;
    }

    const char* GetCString() const
    {
        return &m_Buffer[0];
    }

    void Clear()
    {
        m_Size      = 0;
        m_Buffer[0] = '\0';
    }

    FixedString& operator+=(const char* rhs)
    {
        // GCC 8 would give a warning here because we're potentially truncating *rhs
#if __GNUC__ >= 8
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-truncation"
#endif
        strncat(&m_Buffer[m_Size], rhs, Capacity - m_Size);
#if __GNUC__ >= 8
#pragma GCC diagnostic pop
#endif
        m_Size = std::min(m_Size + strlen(rhs), Capacity);
        return *this;
    }

    FixedString& AppendFormat(const char* rhs, ...)
    {
        va_list args;
        va_start(args, rhs);
        int numWritten = ::vsnprintf(&m_Buffer[m_Size], Capacity + 1 - m_Size, rhs, args);
        m_Size         = std::min(m_Size + static_cast<size_t>(numWritten), Capacity);
        va_end(args);
        return *this;
    }

private:
    /// Storage for the string. We add one byte for the null terminator.
    std::array<char, Capacity + 1> m_Buffer;
    /// Length of the string without the null terminator.
    size_t m_Size;
};

/// 'Null implementation' version of FixedString which has empty methods.
/// This is designed to be swapped out in-place of the regular FixedString in cases
/// where you don't want the overhead of creating and manipulating debug strings (e.g. release builds).
class NullFixedString
{
public:
    NullFixedString()
    {}

    explicit NullFixedString(const char*)
    {}

    template <typename... TArgs>
    static NullFixedString Format(const char*, TArgs...)
    {
        return {};
    }

    size_t GetCapacity()
    {
        return 0;
    }

    size_t GetSize() const
    {
        return 0;
    }

    const char* GetCString() const
    {
        return nullptr;
    }

    void Clear()
    {}

    NullFixedString& operator+=(const char*)
    {
        return *this;
    }

    NullFixedString& AppendFormat(const char*, ...)
    {
        return *this;
    }
};

}    // namespace control_unit
}    // namespace ethosn
