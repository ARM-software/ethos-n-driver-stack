//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

namespace    // Internal linkage
{
// Use namespace to avoid polluting operator overloads
namespace xyz
{
// XY coordinates
struct Xy
{
    unsigned x;
    unsigned y;

    static constexpr Xy Dup(const unsigned x = 0)
    {
        return { x, x };
    }

    constexpr Xy()
        : Xy(0, 0)
    {}

    explicit constexpr Xy(const unsigned x)
        : Xy(x, 0)
    {}

    constexpr Xy(const unsigned x, const unsigned y)
        : x(x)
        , y(y)
    {}

    friend constexpr bool operator==(const Xy&, const Xy&) noexcept = default;
};

// XYZ coordinates
struct Xyz
{
    unsigned x;
    unsigned y;
    unsigned z;

    static constexpr Xyz Dup(const unsigned x = 0)
    {
        return { x, x, x };
    }

    constexpr Xyz()
        : Xyz(0, 0, 0)
    {}

    explicit constexpr Xyz(const unsigned x)
        : Xyz(x, 0, 0)
    {}

    constexpr Xyz(const unsigned x, const unsigned y, const unsigned z = 0)
        : x(x)
        , y(y)
        , z(z)
    {}

    constexpr Xyz(const Xy& xy, const unsigned z)
        : Xyz(xy.x, xy.y, z)
    {}

    constexpr operator Xy() const
    {
        return { x, y };
    }

    friend constexpr bool operator==(const Xyz&, const Xyz&) noexcept = default;
};

namespace impl
{
// Return coord.z if coord has field z.
template <typename T>
constexpr auto GetZ(unsigned, const T& coord) -> decltype(coord.z)
{
    return coord.z;
}
// Return default value dflt otherwise.
constexpr unsigned GetZ(const unsigned dflt, ...)
{
    return dflt;
}

// Set coord.z = z if coord has field z and return the reference to coord.
template <typename T>
constexpr T&& SetZ(T&& coord, const decltype(coord.z) z)
{
    coord.z = z;
    return std::forward<T>(coord);
}
// Simply return the reference to coord otherwise.
template <typename T>
constexpr T&& SetZ(T&& coord, ...)
{
    return std::forward<T>(coord);
}

}    // namespace impl

// Return coord.z if coord has field z.
// Return default value dflt otherwise.
template <typename T>
constexpr unsigned GetZ(const T& coord, const unsigned dflt = 0U)
{
    return impl::GetZ(dflt, coord);
}

// Set coord.z = z if coord has field z and return the reference to coord.
// Simply return the reference to coord otherwise.
template <typename T>
constexpr T&& SetZ(T&& coord, const unsigned z)
{
    return impl::SetZ(std::forward<T>(coord), z);
}

template <typename T>
constexpr unsigned TotalSize(const T& coord)
{
    return coord.x * coord.y * GetZ(coord, 1U);
}

template <typename T>
constexpr T operator+(const T& lhs, const std::type_identity_t<T>& rhs)
{
    return SetZ(T(lhs.x + rhs.x, lhs.y + rhs.y), GetZ(lhs) + GetZ(rhs));
}

template <typename T>
constexpr T operator+(const T& lhs, const unsigned rhs)
{
    return lhs + T::Dup(rhs);
}

template <typename T>
constexpr T& operator+=(T& lhs, const T& rhs)
{
    return (lhs = lhs + rhs);
}

template <typename T>
constexpr T operator-(const T& lhs, const std::type_identity_t<T>& rhs)
{
    return SetZ(T(lhs.x - rhs.x, lhs.y - rhs.y), GetZ(lhs) - GetZ(rhs));
}

template <typename T>
constexpr T operator-(const T& coord)
{
    return (T{} - coord);
}

template <typename T>
constexpr T operator-(const T& lhs, const unsigned rhs)
{
    return lhs - T::Dup(rhs);
}

template <typename T>
constexpr T& operator-=(T& lhs, const T& rhs)
{
    return (lhs = lhs - rhs);
}

template <typename T>
constexpr T operator*(const T& lhs, const std::type_identity_t<T>& rhs)
{
    return SetZ(T(lhs.x * rhs.x, lhs.y * rhs.y), GetZ(lhs) * GetZ(rhs));
}

template <typename T>
constexpr T operator*(const T& lhs, const unsigned rhs)
{
    return lhs * T::Dup(rhs);
}

template <typename T>
constexpr T& operator*=(T& lhs, const T& rhs)
{
    return (lhs = lhs * rhs);
}

template <typename T>
constexpr T operator/(const T& lhs, const std::type_identity_t<T>& rhs)
{
    return SetZ(T(lhs.x / rhs.x, lhs.y / rhs.y), GetZ(lhs) / GetZ(rhs, 1U));
}

template <typename T>
constexpr T operator/(const T& lhs, const unsigned rhs)
{
    return lhs / T::Dup(rhs);
}

template <typename T>
constexpr T& operator/=(T& lhs, const T& rhs)
{
    return (lhs = lhs / rhs);
}

template <typename T>
constexpr T operator%(const T& lhs, const std::type_identity_t<T>& rhs)
{
    return SetZ(T(lhs.x % rhs.x, lhs.y % rhs.y), GetZ(lhs) % GetZ(rhs, 1U));
}

template <typename T>
constexpr T operator%(const T& lhs, const unsigned rhs)
{
    return lhs % T::Dup(rhs);
}

template <typename T>
constexpr T& operator%=(T& lhs, const T& rhs)
{
    return (lhs = lhs % rhs);
}

template <typename T>
constexpr unsigned Dot(T lhs, const std::type_identity_t<T>& rhs)
{
    lhs *= rhs;
    return lhs.x + lhs.y + GetZ(lhs);
}

template <typename T>
constexpr T DivRoundUp(const T& numerator, const std::type_identity_t<T>& denominator)
{
    return (numerator + denominator - 1U) / denominator;
}

template <typename T>
constexpr T DivRoundUp(const T& numerator, const unsigned denominator)
{
    return (numerator + denominator - 1U) / denominator;
}

template <typename T>
constexpr T TransposeXY(const T& coord)
{
    return SetZ(T(coord.y, coord.x), GetZ(coord));
}

}    // namespace xyz

// Export main types to the global scope so they can be used without the namespace qualifier.
using Xy  = xyz::Xy;
using Xyz = xyz::Xyz;

// Static tests

static_assert((Xy{ 1, 5 } + Xy{ 1, 0 }) == Xy{ 2, 5 }, "");
static_assert((Xy{ 3, 5 } - Xy{ 1, 0 }) == Xy{ 2, 5 }, "");
static_assert(!((Xy{ 1, 5 } + Xy{ 1, 1 }) == Xy{ 2, 5 }), "");

static_assert(Xy{ 2, 4 } != Xy{ 2, 5 }, "");
static_assert(!(Xy{ 2, 5 } != Xy{ 2, 5 }), "");

static_assert((Xyz{ 1, 2, 0 } + Xyz{ 1, 3, 3 }) == Xyz{ 2, 5, 3 }, "");
static_assert((Xyz{ 5, 5, 5 } - Xyz{ 3, 0, 2 }) == Xyz{ 2, 5, 3 }, "");
static_assert(!(Xyz{ 2, 5, 2 } == Xyz{ 2, 5, 3 }), "");

static_assert(Xyz{ 2, 5, 2 } != Xyz{ 2, 5, 3 }, "");
static_assert(!(Xyz{ 2, 5, 3 } != Xyz{ 2, 5, 3 }), "");

static_assert(TransposeXY(Xy(2, 5)) == Xy(5, 2), "");
static_assert(TransposeXY(Xyz(2, 5, 3)) == Xyz(5, 2, 3), "");

}    // namespace
