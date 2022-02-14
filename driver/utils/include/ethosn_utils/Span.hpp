//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#if (__cplusplus >= 202002L) || (_MSVC_LANG >= 202002L)

#include <span>

#else

#include <array>
#include <type_traits>
#include <vector>

namespace std
{
template <typename T>
class span
{
public:
    constexpr span() noexcept = default;

    template <typename U>
    constexpr span(U* const data, const size_t size) noexcept
        : m_Data(data)
        , m_Size(size)
    {
        static_assert(std::is_same<std::remove_cv_t<U>, std::remove_cv_t<T>>::value, "");
    }

    template <typename U, size_t N>
    constexpr span(const std::array<U, N>& array) noexcept
        : span(array.data(), N)
    {}

    template <typename U, size_t N>
    constexpr span(std::array<U, N>& array) noexcept
        : span(array.data(), N)
    {}

    template <typename U, size_t N>
    constexpr span(const U (&array)[N]) noexcept
        : span(array, N)
    {}

    template <typename U, size_t N>
    constexpr span(U (&array)[N]) noexcept
        : span(array, N)
    {}

    template <typename U>
    constexpr span(const std::vector<U>& vec) noexcept
        : span(vec.data(), vec.size())
    {}

    template <typename U>
    constexpr span(std::vector<U>& vec) noexcept
        : span(vec.data(), vec.size())
    {}

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
    T* m_Data{};
    size_t m_Size{};
};
}    // namespace std

#endif
