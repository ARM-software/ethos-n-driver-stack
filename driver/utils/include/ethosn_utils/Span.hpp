//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#if (__cplusplus >= 202002L) || (_MSVC_LANG >= 202002L)

#include <span>

#else

#if (__cplusplus >= 201703L) || (_MSVC_LANG >= 201703L)

// For std::data and std::size
#include <array>

#else

namespace std
{
template <typename C>
constexpr auto data(C& c) -> decltype(c.data())
{
    return c.data();
}

template <typename T, size_t N>
constexpr T* data(T (&array)[N]) noexcept
{
    return array;
}

template <typename C>
constexpr auto size(C& c) -> decltype(c.size())
{
    return c.size();
}

template <typename T, size_t N>
constexpr size_t size(T (&)[N]) noexcept
{
    return N;
}
}    // namespace std

#endif

#include <type_traits>

namespace std
{
template <typename T>
class span
{
public:
    constexpr span(T* data, size_t size) noexcept
        : m_Data(data)
        , m_Size(size)
    {}

    template <typename A>
    constexpr span(A& array) noexcept
        : span(std::data(array), std::size(array))
    {
        static_assert(sizeof(T) == sizeof(*std::data(array)), "");
    }

    constexpr auto data() const noexcept
    {
        return m_Data;
    }

    constexpr auto size() const noexcept
    {
        return m_Size;
    }

    constexpr auto begin() const noexcept
    {
        return m_Data;
    }

    constexpr auto end() const noexcept
    {
        return begin() + m_Size;
    }

    constexpr auto& operator[](const size_t i) const noexcept
    {
        return m_Data[i];
    }

private:
    T* m_Data;
    size_t m_Size;
};
}    // namespace std

#endif
