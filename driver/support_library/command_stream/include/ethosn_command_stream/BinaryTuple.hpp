//
// Copyright © 2018-2021 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "CheckPlatform.hpp"
#include "MacroUtils.hpp"

#include <algorithm>
#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

// Helper macro to define `m_{Name}()` and `m_{Name}() const` methods for a single named property
#define DEFINE_ACCESSOR(name)                                                                                          \
    constexpr auto& m_##name() const                                                                                   \
    {                                                                                                                  \
        return Get<Properties::name>();                                                                                \
    }                                                                                                                  \
    constexpr auto& m_##name()                                                                                         \
    {                                                                                                                  \
        return Get<Properties::name>();                                                                                \
    }                                                                                                                  \
    constexpr auto& m_##name() const volatile                                                                          \
    {                                                                                                                  \
        return Get<Properties::name>();                                                                                \
    }                                                                                                                  \
    constexpr auto& m_##name() volatile                                                                                \
    {                                                                                                                  \
        return Get<Properties::name>();                                                                                \
    }

// Helper macro to define `m_{Name}()` access methods for all named properties.
// Call this macro with property names in order.
#define PROPERTIES(name, ...)                                                                                          \
    template <typename... Us>                                                                                          \
    constexpr name(const Us&... us)                                                                                    \
        : Base(us...)                                                                                                  \
    {}                                                                                                                 \
    enum class Properties : size_t                                                                                     \
    {                                                                                                                  \
        __VA_ARGS__                                                                                                    \
    };                                                                                                                 \
    template <Properties P>                                                                                            \
    constexpr auto& Get() const                                                                                        \
    {                                                                                                                  \
        return Base::template Get<static_cast<size_t>(P)>();                                                           \
    }                                                                                                                  \
    template <Properties P>                                                                                            \
    constexpr auto& Get()                                                                                              \
    {                                                                                                                  \
        return Base::template Get<static_cast<size_t>(P)>();                                                           \
    }                                                                                                                  \
    template <Properties P>                                                                                            \
    constexpr auto& Get() const volatile                                                                               \
    {                                                                                                                  \
        return Base::template Get<static_cast<size_t>(P)>();                                                           \
    }                                                                                                                  \
    template <Properties P>                                                                                            \
    constexpr auto& Get() volatile                                                                                     \
    {                                                                                                                  \
        return Base::template Get<static_cast<size_t>(P)>();                                                           \
    }                                                                                                                  \
    FOREACH_N(DEFINE_ACCESSOR, __VA_ARGS__)

// Aliases
#define TYPES ODD_ARGS
#define FIELDS EVEN_ARGS

// Defines name as a direct subclass of AlignedBinaryTuple with alignment and
// named properties as specified by the `type, field` pairs in the argument list.
#define NAMED_ALIGNED_BINARY_TUPLE(name, align, ...)                                                                   \
    struct name : public ethosn::command_stream::AlignedBinaryTuple<align, TYPES(__VA_ARGS__)>                         \
    {                                                                                                                  \
        using Base = ethosn::command_stream::AlignedBinaryTuple<align, TYPES(__VA_ARGS__)>;                            \
        PROPERTIES(name, FIELDS(__VA_ARGS__))                                                                          \
    }

// Expand as NAMED_ALIGNED_BINARY_TUPLE(4, ...)
#define NAMED_ALIGNED_BINARY_TUPLE_4(name, ...) NAMED_ALIGNED_BINARY_TUPLE(name, 4, __VA_ARGS__)

// Defines name as a direct subclass of BinaryTuple with properties as specified
// by the `type, field` pairs in the argument list.
#define NAMED_BINARY_TUPLE(name, ...)                                                                                  \
    struct name : public ethosn::command_stream::BinaryTuple<TYPES(__VA_ARGS__)>                                       \
    {                                                                                                                  \
        using Base = ethosn::command_stream::BinaryTuple<TYPES(__VA_ARGS__)>;                                          \
        PROPERTIES(name, FIELDS(__VA_ARGS__))                                                                          \
    }

#define NAMED_BINARY_TUPLE_SPECIALIZATION(nameWithTemplateArgs, nameWithoutTemplateArgs, ...)                          \
    template <>                                                                                                        \
    struct nameWithTemplateArgs : public ethosn::command_stream::BinaryTuple<TYPES(__VA_ARGS__)>                       \
    {                                                                                                                  \
        using Base = ethosn::command_stream::BinaryTuple<TYPES(__VA_ARGS__)>;                                          \
        PROPERTIES(nameWithoutTemplateArgs, FIELDS(__VA_ARGS__))                                                       \
    }

namespace ethosn
{
namespace command_stream
{
// Implementation details in impl namespace
namespace impl
{

// Returns true if T is a basic type
template <typename T>
constexpr bool IsBasicType()
{
    return std::is_integral<T>::value || std::is_enum<T>::value;
}

// BinaryTypeTraits specialization for AlignedBinaryTuple and derived types.
// Default definition for basic types.
template <typename T, bool B = IsBasicType<T>()>
struct BinaryTypeTraits
{
    // Returns true if T is a basic type
    constexpr static size_t Align = alignof(T);
    constexpr static size_t Size  = sizeof(T);
};

// BinaryTypeTraits specialization for std::array
template <typename U, size_t N>
struct BinaryTypeTraits<std::array<U, N>, false>
{
    constexpr static size_t Align = alignof(U);
    constexpr static size_t Size  = N * sizeof(U);
};

// BinaryTypeTraits specialization for AlignedBinaryTuple and derived types
template <typename T>
struct BinaryTypeTraits<T, false>
{
    constexpr static size_t Align = T::Align;
    constexpr static size_t Size  = T::Size;
};

// Check if T is a valid binary type: i.e. either a basic binary type
// or derived from AlignedBinaryTuple with no extra non-static data
template <typename T>
constexpr bool IsBinary()
{
    return (alignof(T) == BinaryTypeTraits<T>::Align) && (sizeof(T) == BinaryTypeTraits<T>::Size);
}

// Return value rounded-up to the next multiple of Align
template <size_t Align>
constexpr size_t RoundUp(size_t value)
{
    static_assert(((Align - 1) & Align) == 0, "Align must be a power of two");
    return (value + (Align - 1)) & ~(Align - 1);
}

// Helper struct to obtain expected alignment and size of fields in a binary structure
template <size_t A, typename... Ts>
struct StructTraits
{
    constexpr static size_t NFields   = sizeof...(Ts);
    constexpr static size_t LastField = (NFields > 0) ? (NFields - 1) : 0;

    // Helper struct (align=A, size=1)
    struct alignas(A) Next
    {};

    // OffsetOf needs the extra Next's so Type<NFields> and Type<NFields + 1> are well defined
    template <size_t I>
    using Type = std::tuple_element_t<I, std::tuple<Ts..., Next, Next>>;

    // Expected offset of field 0 is always 0
    template <size_t I>
    constexpr static std::enable_if_t<(I == 0), size_t> OffsetOf()
    {
        return 0;
    }
    // Expected offset of field I
    template <size_t I>
    constexpr static std::enable_if_t<(I > 0), size_t> OffsetOf()
    {
        return impl::RoundUp<alignof(Type<I>)>(OffsetOf<I - 1>() + sizeof(Type<I - 1>));
    }
    // Expected size (including padding) of field I
    template <size_t I>
    constexpr static size_t SizeOf()
    {
        return OffsetOf<I + 1>() - OffsetOf<I>();
    }

    // Expected alignment requirement of the whole binary structure
    constexpr static size_t Align = A;
    // Expected size of the whole binary structure
    constexpr static size_t Size = OffsetOf<LastField + 1>();
};

/// Wrapper around type T with padding of size P.
template <typename T, size_t P>
struct TypeAndPadding
{
    T m_Data;
    uint8_t m_Padding[P];

    /// Constructs the TypeAndPadding by initialising the Data with the given arguments and zeroes the leftover padding.
    template <typename... Us>
    constexpr TypeAndPadding(const Us&... us)
        : m_Data{ us... }
        , m_Padding{}    // Zero-initialise the padding
    {}
};

/// Specialisation of TypeAndPadding when there is no padding
template <typename T>
struct TypeAndPadding<T, 0>
{
    T m_Data;

    /// Constructs the TypeAndPadding by initialising the Data with the given arguments and zeroes the leftover padding.
    template <typename... Us>
    constexpr TypeAndPadding(const Us&... us)
        : m_Data{ us... }
    {}
};

/// Wrapper around type T with padding to increase its size to S.
template <typename T, size_t S>
struct PaddedType : public TypeAndPadding<T, S - sizeof(T)>
{
    using TypeAndPadding<T, S - sizeof(T)>::TypeAndPadding;
};

// Specializations implement a struct with explicit alignment/padding of fields
// and a single constexpr constructor taking exactly one argument per field.
template <size_t A, typename... Ts>
struct AlignedStruct;

// Specialisation of AlignedStruct for empty struct
template <size_t A>
struct alignas(A) AlignedStruct<A>
{};

// Helper macros to generate specialisations of AlignedStruct for a fixed number of fields

// clang-format off

// Expands to size_t A, typename T0, ..., typename T<n-1>
#define DECLARE_TEMPLATE_ARGS(n) size_t A FOREACH_N(DECLARE_TEMPLATE_ARG, SEQ(n))
#define DECLARE_TEMPLATE_ARG(id) , typename T##id

// Expands to A, T0, ..., T<n-1>
#define TEMPLATE_ARGS(n) A FOREACH_N(TEMPLATE_ARG, SEQ(n))
#define TEMPLATE_ARG(id) , T##id

// Expands to const T0& t0, ..., const T<n-1>& t<n-1>
#define CONSTRUCTOR_ARGS(n)      RAW_FOREACH_N(CONSTRUCTOR_ARG, LAST_CONSTRUCTOR_ARG, SEQ(n))
#define CONSTRUCTOR_ARG(id)      const std::remove_cv_t<T##id>& t##id = T##id{},
#define LAST_CONSTRUCTOR_ARG(id) const std::remove_cv_t<T##id>& t##id = T##id{}

// Expands to m_id{ t0 }, ..., m_<n-1>{ t<n-1> }
#define FIELD_INITS(n)      RAW_FOREACH_N(FIELD_INIT, LAST_FIELD_INIT, SEQ(n))
#define FIELD_INIT(id)      m_##id{ t##id },
#define LAST_FIELD_INIT(id) m_##id{ t##id }

// clang-format on

// Defines explicitly padded fields m_0, ..., m_<n-1>
#define UNION_FIELDS(n) FOREACH_N(UNION_FIELD, SEQ(n))
// Defines field id as PaddedType<Tid, SizeOf<id>()> m_id;
// Defines a pair of `Get<id>()`, `Get<id>() const` methods for index-based access
#define UNION_FIELD(id)                                                                                                \
    using U##id = PaddedType<T##id, AlignedStruct::SizeOf<id>()>;                                                      \
    U##id m_##id;                                                                                                      \
    template <size_t I>                                                                                                \
    constexpr std::enable_if_t<(I == id), const U##id&> Get() const                                                    \
    {                                                                                                                  \
        return m_##id;                                                                                                 \
    }                                                                                                                  \
    template <size_t I>                                                                                                \
    constexpr std::enable_if_t<(I == id), U##id&> Get()                                                                \
    {                                                                                                                  \
        return m_##id;                                                                                                 \
    }                                                                                                                  \
    template <size_t I>                                                                                                \
    constexpr std::enable_if_t<(I == id), const volatile U##id&> Get() const volatile                                  \
    {                                                                                                                  \
        return m_##id;                                                                                                 \
    }                                                                                                                  \
    template <size_t I>                                                                                                \
    constexpr std::enable_if_t<(I == id), volatile U##id&> Get() volatile                                              \
    {                                                                                                                  \
        return m_##id;                                                                                                 \
    }

// Define a specialisation of AlignedStruct that contains n fields
#define DEFINE_ALIGNED_STRUCT(n)                                                                                       \
    template <DECLARE_TEMPLATE_ARGS(n)>                                                                                \
    struct alignas(A) AlignedStruct<TEMPLATE_ARGS(n)>                                                                  \
    {                                                                                                                  \
        using Traits = StructTraits<TEMPLATE_ARGS(n)>;                                                                 \
        template <size_t I>                                                                                            \
        constexpr static size_t SizeOf()                                                                               \
        {                                                                                                              \
            return Traits::template SizeOf<I>();                                                                       \
        }                                                                                                              \
        constexpr AlignedStruct(CONSTRUCTOR_ARGS(n))                                                                   \
            : FIELD_INITS(n)                                                                                           \
        {}                                                                                                             \
        UNION_FIELDS(n)                                                                                                \
    };

// Explicit specialisations of AlignedStruct
DEFINE_ALIGNED_STRUCT(1)
DEFINE_ALIGNED_STRUCT(2)
DEFINE_ALIGNED_STRUCT(3)
DEFINE_ALIGNED_STRUCT(4)
DEFINE_ALIGNED_STRUCT(5)
DEFINE_ALIGNED_STRUCT(6)
DEFINE_ALIGNED_STRUCT(7)
DEFINE_ALIGNED_STRUCT(8)
DEFINE_ALIGNED_STRUCT(9)
DEFINE_ALIGNED_STRUCT(10)
DEFINE_ALIGNED_STRUCT(11)
DEFINE_ALIGNED_STRUCT(12)
DEFINE_ALIGNED_STRUCT(13)
DEFINE_ALIGNED_STRUCT(14)
DEFINE_ALIGNED_STRUCT(15)
DEFINE_ALIGNED_STRUCT(16)
// Add lines here to increase the maximum number of fields supported

}    // namespace impl

// Implementation of AlignedBinaryTuple: a simplified std::tuple
// with explicit alignment and memory layout.
template <size_t A, typename... Ts>
struct AlignedBinaryTuple : public impl::AlignedStruct<A, Ts...>
{
    using Base   = impl::AlignedStruct<A, Ts...>;
    using Traits = impl::StructTraits<A, Ts...>;

    template <size_t I>
    using Type = typename Traits::template Type<I>;

    constexpr static size_t NFields = Traits::NFields;
    constexpr static size_t Align   = Traits::Align;
    constexpr static size_t Size    = Traits::Size;

    // Initialise the first tuple elements with `us...` and zero-initialise the rest
    // (padding bytes are zero-initialised if any)
    template <typename... Us>
    constexpr AlignedBinaryTuple(const Us&... us)
        : AlignedBinaryTuple(std::index_sequence_for<Us...>{}, us...)
    {
        static_assert(std::min({ true, impl::IsBinary<Ts>()... }), "Not all Ts are binary type");
        static_assert(Align >= std::max({ size_t{ 1U }, alignof(Ts)... }), "Alignment requirements can't be relaxed");
        static_assert(alignof(AlignedBinaryTuple) == Align, "");
        static_assert(sizeof(AlignedBinaryTuple) == Size, "Unexpected padding in DataStruct");
    }

    // Read-only access of tuple element I
    template <size_t I>
    constexpr const Type<I>& Get() const
    {
        return Base::template Get<I>().m_Data;
    }
    // Read-write access of tuple element I
    template <size_t I>
    constexpr Type<I>& Get()
    {
        return Base::template Get<I>().m_Data;
    }
    // Read-only access of tuple element I volatile
    template <size_t I>
    constexpr const volatile Type<I>& Get() const volatile
    {
        return Base::template Get<I>().m_Data;
    }
    // Read-write access of tuple element I volatile
    template <size_t I>
    constexpr volatile Type<I>& Get() volatile
    {
        return Base::template Get<I>().m_Data;
    }

    // Return true if all corresponding tuple elements in this/other compare equal
    // (ignore padding)
    constexpr bool operator==(const AlignedBinaryTuple& other) const
    {
        return IsEqual(std::index_sequence_for<Ts...>{}, other);
    }
    constexpr bool operator!=(const AlignedBinaryTuple& other) const
    {
        return !(*this == other);
    }

private:
    // Helper constructor initialises the first tuple elements with us... and zero-initializes the rest.
    // Use std::index_sequence_for<Us...>{} to generate the helper parameter pack Is
    template <size_t... Is, typename... Us>
    constexpr AlignedBinaryTuple(const std::index_sequence<Is...>&, const Us&... us)
        : Base(Type<Is>{ us }...)
    {}

    // Return true if all fields in this and other compare equal.
    // Use std::index_sequence_for<Ts...>{} to generate the helper parameter pack Is
    template <size_t... Is>
    constexpr bool IsEqual(const std::index_sequence<Is...>&, const AlignedBinaryTuple& other) const
    {
        return std::min({ true, (Get<Is>() == other.Get<Is>())... });
    }
};

/// Returns the maximum of the natural alignment of all the given type parameters.
template <typename... Ts>
constexpr size_t MaxAlignOf()
{
    return std::max({ size_t{ 1U }, alignof(Ts)... });
}

template <typename... Ts>
using BinaryTuple = AlignedBinaryTuple<MaxAlignOf<Ts...>(), Ts...>;

}    // namespace command_stream
}    // namespace ethosn
