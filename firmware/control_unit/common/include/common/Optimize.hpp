//
// Copyright © 2018-2019,2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Utils.hpp"

namespace ethosn
{
namespace control_unit
{

#if defined(DISABLE_POW2)
using Pow2 = uint32_t;
#else

class Pow2
{
public:
    constexpr Pow2() noexcept
        : Pow2(1)
    {}

    constexpr explicit Pow2(const uint32_t v) noexcept
        : m_Value(v)
        , m_ModMask(0x7FFFFFFFU >> utils::CountLeadingZeros(v))
        , m_Log2(31 - utils::CountLeadingZeros(v))
    {}

    constexpr uint32_t GetValue() const noexcept
    {
        return m_Value;
    }

    constexpr uint32_t GetModMask() const noexcept
    {
        return m_ModMask;
    }

    constexpr uint32_t GetLog2() const noexcept
    {
        return m_Log2;
    }

    constexpr operator uint32_t() const noexcept
    {
        return m_Value;
    }

private:
    uint32_t m_Value;
    uint32_t m_ModMask;
    uint32_t m_Log2;
};

// MUL
template <typename T, std::enable_if_t<std::is_unsigned_v<T>, int> = 0>
constexpr uint32_t operator*(const Pow2 x, const T y) noexcept
{
    return y * x;
}

template <typename T, std::enable_if_t<std::is_unsigned_v<T>, int> = 0>
constexpr uint32_t operator*(const T x, const Pow2 y) noexcept
{
    return (1U * x) << y.GetLog2();
}

constexpr Pow2 operator*(const Pow2 x, const Pow2 y) noexcept
{
    return Pow2(x.GetValue() * y);
}

// DIV
template <typename T, std::enable_if_t<std::is_unsigned_v<T>, int> = 0>
constexpr uint32_t operator/(const T x, const Pow2 y) noexcept
{
    return (1U * x) >> y.GetLog2();
}

constexpr Pow2 operator/(const Pow2 x, const Pow2 y) noexcept
{
    return Pow2(x.GetValue() / y);
}

// MOD
template <typename T, std::enable_if_t<std::is_unsigned_v<T>, int> = 0>
constexpr uint32_t operator%(const T x, const Pow2 y) noexcept
{
    return (1U * x) & y.GetModMask();
}

#define POW2_POW2_TEST_CASE(shift0, shift1)                                                                            \
    static_assert(static_cast<uint32_t>(Pow2(1U << shift0) * Pow2(1U << shift1)) ==                                    \
                      static_cast<uint32_t>((1U << shift0) * (1U << shift1)),                                          \
                  "");                                                                                                 \
    static_assert(static_cast<uint32_t>(Pow2(1U << shift0) / Pow2(1U << shift1)) ==                                    \
                      static_cast<uint32_t>((1U << shift0) / (1U << shift1)),                                          \
                  "");                                                                                                 \
    static_assert(static_cast<uint32_t>(Pow2(1U << shift0) % Pow2(1U << shift1)) ==                                    \
                      static_cast<uint32_t>((1U << shift0) % (1U << shift1)),                                          \
                  "");

#define POW2_UINT32_TEST_CASE(test_val, shift)                                                                         \
    static_assert(                                                                                                     \
        static_cast<uint32_t>(test_val) * Pow2(1U << shift) == static_cast<uint32_t>(test_val) * (1U << shift), "");   \
    static_assert(                                                                                                     \
        Pow2(1U << shift) * static_cast<uint32_t>(test_val) == (1U << shift) * static_cast<uint32_t>(test_val), "");   \
    static_assert(                                                                                                     \
        static_cast<uint32_t>(test_val) / Pow2(1U << shift) == static_cast<uint32_t>(test_val) / (1U << shift), "");   \
    static_assert(                                                                                                     \
        static_cast<uint32_t>(test_val) % Pow2(1U << shift) == static_cast<uint32_t>(test_val) % (1U << shift), "");

POW2_POW2_TEST_CASE(6, 3)
POW2_POW2_TEST_CASE(4, 2)
POW2_POW2_TEST_CASE(3, 2)
POW2_UINT32_TEST_CASE(16, 2)
POW2_UINT32_TEST_CASE(32, 4)
POW2_UINT32_TEST_CASE(513, 3)
#endif
}    // namespace control_unit
}    // namespace ethosn
