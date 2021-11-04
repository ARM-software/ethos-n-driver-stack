//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
//
// This is intended to help write cleaner code for small vector objects with named elements.
//
// For example, consider a struct Xyz and the following desirable properties:
//
// - Fields x, y, z can be accessed by normal named member access.
//
//     Xyz tensorSize;
//     tensorSize.x = 16;
//     tensorSize.y = 16;
//     tensorSize.z = 64;
//
// - Arithmetic operators work element-wise as if Xyz was a vector of size 3.
//
//     constexpr auto GetLinearOffset(const Xyz& coord, const Xyz& strides)
//     {
//         return Sum(coord * strides);
//     }
//
// - Arithmetic operators with scalars work element-wise as if Xyz was a vector of size 3.
//
//     constexpr bool GetUpscaledSize(const Xyz& size)
//     {
//         return 2 * size;
//     }
//
// This file provides a class sv::Vector and a macro ETHOSN_USE_AS_SV_VECTOR that can be used
// to give these properties to a plain struct:
//
//     struct Xyz
//     {
//         uint32_t x;
//         uint32_t y;
//         uint32_t z;
//
//         ETHOSN_USE_AS_SV_VECTOR(Xyz, uint32_t, 3)
//     };
//
//
// NOTE
// ----
//
// Unary (+, -, ...) and binary operators (+, -, *, /, %, ...) on the sv::Vector-enabled structs
// always produce an expression of type sv::Vector<R, N>, with R = deduced type, N = same as the
// operands.
//
// If a result of type equal to a sv::Vector-enabled struct is desired, an object of type
// sv::Vector-enabled struct can be assigned from a compatible sv::Vector.
//
// Examples:
//
//     Xyz tensorSize     = { 16, 24, 48 };
//     Xyz dfltStripeSize = { 8, 16, 32 };
//
//     //Xyz edgeStripeSize = ((tensorSize - 1U) % dfltStripeSize) + 1U; // ERROR: Xyz cannot be
//                                                                       // constructed from
//                                                                       // sv::Vector expression
//
//     Xyz edgeStripeSize1;
//     edgeStripeSize = ((tensorSize - 1U) % dfltStripeSize) + 1U; // OK: assignment operator works
//                                                                 // with sv::Vector expression
//
#pragma once

#define ETHOSN_DECL_SV_VECTOR_STRUCT(Name, ...)                                                                        \
    template <typename T = int>                                                                                        \
    struct Name                                                                                                        \
    {                                                                                                                  \
        T __VA_ARGS__;                                                                                                 \
        ETHOSN_USE_AS_SV_VECTOR(Name, T, N_ARGS(__VA_ARGS__))                                                          \
    };                                                                                                                 \
    ETHOSN_DECL_SV_VECTOR_STRUCT_CTAD(Name)

// This is only supported for >=c++17 builds
#if !((__cplusplus >= 201703L) || (_MSVC_LANG >= 201703L))
#define ETHOSN_DECL_SV_VECTOR_STRUCT_CTAD(...)
#define ETHOSN_USE_AS_SV_VECTOR(...)
#else

#include "BitCast.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <type_traits>
#include <utility>

#define ETHOSN_DECL_SV_VECTOR_STRUCT_CTAD(Name)                                                                        \
    template <typename T, typename... Us>                                                                              \
    Name(T, Us...)->Name<T>;

#define ETHOSN_USE_AS_SV_VECTOR(Typename, T, N)                                                                        \
    using Vector = ethosn::utils::sv::Vector<T, N>;                                                                    \
    static Typename Dup(const T& value)                                                                                \
    {                                                                                                                  \
        Typename x;                                                                                                    \
        x = Vector::Dup(value);                                                                                        \
        return x;                                                                                                      \
    }                                                                                                                  \
    Typename& operator=(const Vector& vec)                                                                             \
    {                                                                                                                  \
        return (*this = vec.template To<Typename>());                                                                  \
    }                                                                                                                  \
    Vector AsVector() const                                                                                            \
    {                                                                                                                  \
        return Vector::ToVector(*this);                                                                                \
    }                                                                                                                  \
    std::array<T, N> AsArray() const                                                                                   \
    {                                                                                                                  \
        return AsVector().AsArray();                                                                                   \
    }                                                                                                                  \
    ETHOSN_DEF_SV_VECTOR_UNARY_OPERATOR(Typename, +)                                                                   \
    ETHOSN_DEF_SV_VECTOR_UNARY_OPERATOR(Typename, -)                                                                   \
    ETHOSN_DEF_SV_VECTOR_UNARY_OPERATOR(Typename, !)                                                                   \
    ETHOSN_DEF_SV_VECTOR_UNARY_OPERATOR(Typename, ~)                                                                   \
    ETHOSN_DEF_SV_VECTOR_BINARY_OPERATOR(Typename, N, +)                                                               \
    ETHOSN_DEF_SV_VECTOR_BINARY_OPERATOR(Typename, N, -)                                                               \
    ETHOSN_DEF_SV_VECTOR_BINARY_OPERATOR(Typename, N, *)                                                               \
    ETHOSN_DEF_SV_VECTOR_BINARY_OPERATOR(Typename, N, /)                                                               \
    ETHOSN_DEF_SV_VECTOR_BINARY_OPERATOR(Typename, N, %)                                                               \
    ETHOSN_DEF_SV_VECTOR_BINARY_OPERATOR(Typename, N, ==)                                                              \
    ETHOSN_DEF_SV_VECTOR_BINARY_OPERATOR(Typename, N, !=)                                                              \
    ETHOSN_DEF_SV_VECTOR_BINARY_OPERATOR(Typename, N, >)                                                               \
    ETHOSN_DEF_SV_VECTOR_BINARY_OPERATOR(Typename, N, <)                                                               \
    ETHOSN_DEF_SV_VECTOR_BINARY_OPERATOR(Typename, N, >=)                                                              \
    ETHOSN_DEF_SV_VECTOR_BINARY_OPERATOR(Typename, N, <=)                                                              \
    ETHOSN_DEF_SV_VECTOR_BINARY_OPERATOR(Typename, N, &&)                                                              \
    ETHOSN_DEF_SV_VECTOR_BINARY_OPERATOR(Typename, N, ||)                                                              \
    ETHOSN_DEF_SV_VECTOR_BINARY_OPERATOR(Typename, N, &)                                                               \
    ETHOSN_DEF_SV_VECTOR_BINARY_OPERATOR(Typename, N, |)                                                               \
    ETHOSN_DEF_SV_VECTOR_BINARY_OPERATOR(Typename, N, <<)                                                              \
    ETHOSN_DEF_SV_VECTOR_BINARY_OPERATOR(Typename, N, >>)                                                              \
    /* clang-format off */ ETHOSN_DEF_SV_VECTOR_BINARY_OPERATOR(Typename, N, ^) /* clang-format on */

#define ETHOSN_DEF_SV_VECTOR_UNARY_OPERATOR(Typename, op)                                                              \
    friend auto operator op(const Typename& t)                                                                         \
    {                                                                                                                  \
        return op t.AsVector();                                                                                        \
    }

#define ETHOSN_DEF_SV_VECTOR_BINARY_OPERATOR(Typename, N, op)                                                          \
    friend auto operator op(const Typename& lhs, const Typename& rhs)                                                  \
    {                                                                                                                  \
        return lhs.AsVector() op rhs.AsVector();                                                                       \
    }                                                                                                                  \
    template <typename U>                                                                                              \
    friend auto operator op(const Typename& lhs, const ethosn::utils::sv::Vector<U, N>& rhs)                           \
    {                                                                                                                  \
        return lhs.AsVector() op rhs;                                                                                  \
    }                                                                                                                  \
    template <typename U>                                                                                              \
    friend auto operator op(const ethosn::utils::sv::Vector<U, N>& lhs, const Typename& rhs)                           \
    {                                                                                                                  \
        return lhs op rhs.AsVector();                                                                                  \
    }                                                                                                                  \
    template <typename U>                                                                                              \
    friend auto operator op(const Typename& lhs, const U& rhs)->decltype(std::declval<Vector>() op rhs)                \
    {                                                                                                                  \
        return lhs.AsVector() op rhs;                                                                                  \
    }                                                                                                                  \
    template <typename U>                                                                                              \
    friend auto operator op(const U& lhs, const Typename& rhs)->decltype(lhs op std::declval<Vector>())                \
    {                                                                                                                  \
        return lhs op rhs.AsVector();                                                                                  \
    }

namespace ethosn::utils::sv
{
namespace detail
{
template <typename From, typename To>
inline constexpr bool is_non_narrowing_conversion_v =
    // clang-format off
        (
            (std::is_integral_v<From> && std::is_integral_v<To>) ||
            (std::is_floating_point_v<From> && std::is_floating_point_v<To>)
        ) &&
        !(std::is_signed_v<From> && std::is_unsigned_v<To>) &&
        (sizeof(From) < sizeof(To));
// clang-format on
}    // namespace detail

template <typename T, size_t N>
struct Vector : std::array<T, N>
{
    static_assert(N <= 4, "This class is meant for small vectors. This limit is arbitrary");

    static constexpr Vector Dup(const T& value)
    {
        return Vector{ Vector<T, 0>{}, value };
    }

    template <typename U>
    static Vector ToVector(const U& obj)
    {
        return std::bit_cast<Vector>(obj);
    }

    constexpr Vector(const Vector&) = default;

    constexpr Vector()
        : std::array<T, N>{}
    {}

    template <typename... Us>
    explicit constexpr Vector(const T& t, Us&&... us)
        : std::array<T, N>{ { t, std::forward<Us>(us)... } }
    {}

    template <typename U, std::enable_if_t<detail::is_non_narrowing_conversion_v<U, T>, int> = 0>
    constexpr Vector(const Vector<U, N>& other)
    {
        std::copy_n(std::begin(other), N, std::begin(*this));
    }

    template <typename U, size_t M>
    explicit constexpr Vector(const Vector<U, M>& other, const T fillValue = T{})
    {
        // Copy up to M elements from other and init the remaining (if any) elements up to N with fillValue
        std::fill(std::copy_n(std::begin(other), std::min(N, M), std::begin(*this)), std::end(*this), fillValue);
    }

    template <size_t pos, size_t M = std::max(N, pos) - pos>
    constexpr Vector<T, M> Slice(const T fillValue = T{}) const
    {
        constexpr size_t N2 = std::max(N, pos) - pos;
        Vector<T, M> slice;
        std::fill(std::copy_n(std::begin(*this) + pos, std::min(N2, M), std::begin(slice)), std::end(slice), fillValue);
        return slice;
    }

    template <size_t M>
    constexpr Vector<T, M> Resize(const T fillValue = T{}) const
    {
        return Slice<0, M>(fillValue);
    }

    template <typename U>
    auto To() const
    {
        return std::bit_cast<U>(*this);
    }

    constexpr std::array<T, N>& AsArray()
    {
        return *this;
    }

    constexpr const std::array<T, N>& AsArray() const
    {
        return *this;
    }
};

template <typename T, typename... Us>
Vector(T, Us...)->Vector<T, 1 + sizeof...(Us)>;

template <typename T, typename U, size_t N = sizeof(U) / sizeof(T)>
Vector<T, N> ToVector(const U& obj)
{
    return Vector<T, N>::ToVector(obj);
}

////////////////////////////////
// Unary operators            //
////////////////////////////////

template <typename T, size_t N, typename Fn>
constexpr auto Op(const Vector<T, N>& vec, Fn&& fn) -> Vector<decltype(fn(vec[0])), N>
{
    Vector<decltype(fn(vec[0])), N> result;
    for (size_t i = 0; i < N; ++i)
    {
        result[i] = fn(vec[i]);
    }
    return result;
}

#define OP_VECTOR_OPERATOR(op, fn)                                                                                     \
    template <typename T, size_t N>                                                                                    \
    constexpr auto operator op(const Vector<T, N>& vec)                                                                \
    {                                                                                                                  \
        return Op(vec, fn);                                                                                            \
    }

// arithmetic
OP_VECTOR_OPERATOR(+, ([](auto&& x) { return +x; }))
OP_VECTOR_OPERATOR(-, ([](auto&& x) { return -x; }))
// logical
OP_VECTOR_OPERATOR(!, ([](auto&& x) { return !x; }))
// bitwise
OP_VECTOR_OPERATOR(~, ([](auto&& x) { return ~x; }))

#undef OP_VECTOR_OPERATOR

///////////////////////////////////
// Vector op Vector operators    //
///////////////////////////////////

template <typename T, typename U, size_t N, typename Fn>
constexpr auto Op(const Vector<T, N>& lhs, const Vector<U, N>& rhs, Fn&& fn) -> Vector<decltype(fn(lhs[0], rhs[0])), N>
{
    Vector<decltype(fn(lhs[0], rhs[0])), N> result;
    for (size_t i = 0; i < N; ++i)
    {
        result[i] = fn(lhs[i], rhs[i]);
    }
    return result;
}

#define VECTOR_OP_VECTOR_OPERATOR(op, fn)                                                                              \
    template <typename T, typename U, size_t N>                                                                        \
    constexpr auto operator op(const Vector<T, N>& lhs, const Vector<U, N>& rhs)                                       \
    {                                                                                                                  \
        return Op(lhs, rhs, fn);                                                                                       \
    }

// arithmetic
VECTOR_OP_VECTOR_OPERATOR(+, std::plus<>{})
VECTOR_OP_VECTOR_OPERATOR(-, std::minus<>{})
VECTOR_OP_VECTOR_OPERATOR(*, std::multiplies<>{})
VECTOR_OP_VECTOR_OPERATOR(/, std::divides<>{})
VECTOR_OP_VECTOR_OPERATOR(%, std::modulus<>{})
// comparisons
VECTOR_OP_VECTOR_OPERATOR(==, std::equal_to<>{})
VECTOR_OP_VECTOR_OPERATOR(!=, std::not_equal_to<>{})
VECTOR_OP_VECTOR_OPERATOR(>, std::greater<>{})
VECTOR_OP_VECTOR_OPERATOR(<, std::less<>{})
VECTOR_OP_VECTOR_OPERATOR(>=, std::greater_equal<>{})
VECTOR_OP_VECTOR_OPERATOR(<=, std::less_equal<>{})
// logical
VECTOR_OP_VECTOR_OPERATOR(&&, std::logical_and<>{})
VECTOR_OP_VECTOR_OPERATOR(||, std::logical_or<>{})
// bitwise
VECTOR_OP_VECTOR_OPERATOR(&, std::bit_and<>{})
VECTOR_OP_VECTOR_OPERATOR(|, std::bit_or<>{})
// clang-format off
VECTOR_OP_VECTOR_OPERATOR(^, std::bit_xor<>{})
// clang-format on
// shift
VECTOR_OP_VECTOR_OPERATOR(<<, ([](auto&& x, auto&& y) { return x << y; }))
VECTOR_OP_VECTOR_OPERATOR(>>, ([](auto&& x, auto&& y) { return x >> y; }))

#undef VECTOR_OP_VECTOR_OPERATOR

///////////////////////////////////
// Vector op Scalar operators    //
///////////////////////////////////

template <typename T, typename U, size_t N, typename Fn>
constexpr auto Op(const Vector<T, N>& lhs, const U& rhs, Fn&& fn) -> Vector<decltype(fn(lhs[0], rhs)), N>
{
    Vector<decltype(fn(lhs[0], rhs)), N> result;
    for (size_t i = 0; i < N; ++i)
    {
        result[i] = fn(lhs[i], rhs);
    }
    return result;
}

#define VECTOR_OP_SCALAR_OPERATOR(op, fn)                                                                              \
    template <typename T, typename U, size_t N>                                                                        \
    constexpr auto operator op(const Vector<T, N>& lhs, const U& rhs)->Vector<decltype(lhs[0] op rhs), N>              \
    {                                                                                                                  \
        return Op(lhs, rhs, fn);                                                                                       \
    }

// arithmetic
VECTOR_OP_SCALAR_OPERATOR(+, std::plus<>{})
VECTOR_OP_SCALAR_OPERATOR(-, std::minus<>{})
VECTOR_OP_SCALAR_OPERATOR(*, std::multiplies<>{})
VECTOR_OP_SCALAR_OPERATOR(/, std::divides<>{})
VECTOR_OP_SCALAR_OPERATOR(%, std::modulus<>{})
// comparisons
VECTOR_OP_SCALAR_OPERATOR(==, std::equal_to<>{})
VECTOR_OP_SCALAR_OPERATOR(!=, std::not_equal_to<>{})
VECTOR_OP_SCALAR_OPERATOR(>, std::greater<>{})
VECTOR_OP_SCALAR_OPERATOR(<, std::less<>{})
VECTOR_OP_SCALAR_OPERATOR(>=, std::greater_equal<>{})
VECTOR_OP_SCALAR_OPERATOR(<=, std::less_equal<>{})
// logical
VECTOR_OP_SCALAR_OPERATOR(&&, std::logical_and<>{})
VECTOR_OP_SCALAR_OPERATOR(||, std::logical_or<>{})
// bitwise
VECTOR_OP_SCALAR_OPERATOR(&, std::bit_and<>{})
VECTOR_OP_SCALAR_OPERATOR(|, std::bit_or<>{})
// clang-format off
VECTOR_OP_SCALAR_OPERATOR(^, std::bit_xor<>{})
// clang-format on
// shift
VECTOR_OP_SCALAR_OPERATOR(<<, ([](auto&& x, auto&& y) { return x << y; }))
VECTOR_OP_SCALAR_OPERATOR(>>, ([](auto&& x, auto&& y) { return x >> y; }))

#undef VECTOR_OP_SCALAR_OPERATOR

///////////////////////////////////
// Scalar op Vector operators    //
///////////////////////////////////

template <typename T, typename U, size_t N, typename Fn>
constexpr auto Op(const T& lhs, const Vector<U, N>& rhs, Fn&& fn) -> Vector<decltype(fn(lhs, rhs[0])), N>
{
    Vector<decltype(fn(lhs, rhs[0])), N> result;
    for (size_t i = 0; i < N; ++i)
    {
        result[i] = fn(lhs, rhs[i]);
    }
    return result;
}

#define SCALAR_OP_VECTOR_OPERATOR(op, fn)                                                                              \
    template <typename T, typename U, size_t N>                                                                        \
    constexpr auto operator op(const T& lhs, const Vector<U, N>& rhs)->Vector<decltype(lhs op rhs[0]), N>              \
    {                                                                                                                  \
        return Op(lhs, rhs, fn);                                                                                       \
    }

// arithmetic
SCALAR_OP_VECTOR_OPERATOR(+, std::plus<>{})
SCALAR_OP_VECTOR_OPERATOR(-, std::minus<>{})
SCALAR_OP_VECTOR_OPERATOR(*, std::multiplies<>{})
SCALAR_OP_VECTOR_OPERATOR(/, std::divides<>{})
SCALAR_OP_VECTOR_OPERATOR(%, std::modulus<>{})
// comparisons
SCALAR_OP_VECTOR_OPERATOR(==, std::equal_to<>{})
SCALAR_OP_VECTOR_OPERATOR(!=, std::not_equal_to<>{})
SCALAR_OP_VECTOR_OPERATOR(>, std::greater<>{})
SCALAR_OP_VECTOR_OPERATOR(<, std::less<>{})
SCALAR_OP_VECTOR_OPERATOR(>=, std::greater_equal<>{})
SCALAR_OP_VECTOR_OPERATOR(<=, std::less_equal<>{})
// logical
SCALAR_OP_VECTOR_OPERATOR(&&, std::logical_and<>{})
SCALAR_OP_VECTOR_OPERATOR(||, std::logical_or<>{})
// bitwise
SCALAR_OP_VECTOR_OPERATOR(&, std::bit_and<>{})
SCALAR_OP_VECTOR_OPERATOR(|, std::bit_or<>{})
// clang-format off
SCALAR_OP_VECTOR_OPERATOR(^, std::bit_xor<>{})
// clang-format on
// shift
SCALAR_OP_VECTOR_OPERATOR(<<, ([](auto&& x, auto&& y) { return x << y; }))
SCALAR_OP_VECTOR_OPERATOR(>>, ([](auto&& x, auto&& y) { return x >> y; }))

#undef SCALAR_OP_VECTOR_OPERATOR

///////////////////////////////////
// Other functions on Vectors    //
///////////////////////////////////

template <typename T, typename U, size_t N>
constexpr auto CSel(const Vector<bool, N>& cond, const Vector<T, N>& vec1, const Vector<U, N>& vec2)
    -> Vector<std::decay_t<decltype(cond[0] ? vec1[0] : vec2[0])>, N>
{
    Vector<std::decay_t<decltype(cond[0] ? vec1[0] : vec2[0])>, N> result;
    for (size_t i = 0; i < N; ++i)
    {
        result[i] = cond[i] ? vec1[i] : vec2[i];
    }
    return result;
}

template <typename T, typename U, size_t N>
constexpr auto CSel(const Vector<bool, N>& cond, const Vector<T, N>& vec, const U& scalar)
    -> Vector<std::decay_t<decltype(cond[0] ? vec[0] : scalar)>, N>
{
    Vector<std::decay_t<decltype(cond[0] ? vec[0] : scalar)>, N> result;
    for (size_t i = 0; i < N; ++i)
    {
        result[i] = cond[i] ? vec[i] : scalar;
    }
    return result;
}

template <typename T, typename U, size_t N>
constexpr auto CSel(const Vector<bool, N>& cond, const T& scalar, const Vector<U, N>& vec)
    -> Vector<std::decay_t<decltype(cond[0] ? scalar : vec[0])>, N>
{
    Vector<std::decay_t<decltype(cond[0] ? scalar : vec[0])>, N> result;
    for (size_t i = 0; i < N; ++i)
    {
        result[i] = cond[i] ? scalar : vec[i];
    }
    return result;
}

template <typename T, size_t N, typename Fn = std::plus<>, typename U = decltype(Fn{}(T{}, T{}))>
constexpr U Reduce(const Vector<T, N>& vec, Fn&& fn = Fn{}, U init = U{})
{
    for (size_t i = 0; i < N; ++i)
    {
        init = fn(init, vec[i]);
    }
    return init;
}

template <typename T, size_t N, typename U = decltype(T{} + T{})>
constexpr auto Sum(const Vector<T, N>& vec, const U& init = U{})
{
    return Reduce(vec, std::plus<>{}, init);
}

template <typename T, size_t N, typename U = decltype(T{} * T{})>
constexpr auto Prod(const Vector<T, N>& vec, const U& init)
{
    return Reduce(vec, std::multiplies<>{}, init);
}

template <typename T, size_t N>
constexpr auto Prod(const Vector<T, N>& vec)
{
    return Prod(vec.template Slice<1>(), 1 * vec[0]);
}

template <typename T, size_t N>
constexpr auto Min(const Vector<T, N>& vec) -> std::decay_t<decltype(std::min(vec[0], vec[0]))>
{
    return Reduce(vec, [](auto&& x, auto&& y) { return std::min(x, y); }, vec[0]);
}

template <typename T, size_t N>
constexpr auto Max(const Vector<T, N>& vec) -> std::decay_t<decltype(std::max(vec[0], vec[0]))>
{
    return Reduce(vec, [](auto&& x, auto&& y) { return std::max(x, y); }, vec[0]);
}

template <size_t N>
constexpr bool All(const Vector<bool, N>& vec)
{
    return Min(vec);
}

template <size_t N>
constexpr bool Any(const Vector<bool, N>& vec)
{
    return Max(vec);
}

template <size_t N>
constexpr bool None(const Vector<bool, N>& vec)
{
    return !Any(vec);
}

}    // namespace ethosn::utils::sv

#endif
