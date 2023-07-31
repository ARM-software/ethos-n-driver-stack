//
// Copyright © 2022-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef ETHOSN_ASSERT_MSG
#error "Please define ETHOSN_ASSERT_MSG"
#endif

/// Calls NumericCast (see below), but also masks the result into the specified
/// number of bits to silence compiler warnings.
/// A define is necessary for this as some compilers will still raise incorrect warnings
/// about narrowing into bitfields when the masking is done in a function.
#define ETHOSN_NUMERIC_CAST(source, TDest, NumBits)                                                                    \
    (ethosn::utils::NumericCast<TDest, (NumBits)>(source) & static_cast<TDest>((1U << (NumBits)) - 1U))

namespace ethosn
{
namespace utils
{

/// Casts from one numeric type to another, checking that the value is within the range of the destination type
/// and asserting if not.
/// This overload is for integer types only (enforced by using enable_if on source type).
template <typename TDest,
          uint32_t NumBits = 0,    // Optionally checks that the value fits within this number of bits
          typename TSource>
constexpr std::enable_if_t<std::is_integral<TSource>::value, TDest> NumericCast(const TSource source)
{
    static_assert(std::is_arithmetic<TDest>::value, "Destination type must be a numeric type.");
    static_assert(std::is_arithmetic<TSource>::value, "Source type must be a numeric type.");
    ETHOSN_ASSERT_MSG(source >= std::numeric_limits<TDest>::lowest() && source <= std::numeric_limits<TDest>::max(),
                      "Source value is out of range of destination type");
    if (NumBits > 0)
    {
        constexpr uint64_t maxValue = static_cast<uint64_t>(1) << NumBits;
        ETHOSN_ASSERT_MSG(source >= 0 && static_cast<uint64_t>(source) < maxValue,
                          "Source value requires more bits than available, or is negative");
        return static_cast<TDest>(static_cast<uint64_t>(source) & (maxValue - 1));
    }
    return static_cast<TDest>(source);
}

/// Casts from one numeric type to another, checking that the value is within the range of the destination type
/// and asserting if not.
/// This overload is for enum types only (enforced by using enable_if on return type).
template <typename TDest,
          uint32_t NumBits = 0,    // Optionally checks that the value fits within this number of bits
          typename TSource>
constexpr std::enable_if_t<std::is_enum<TSource>::value, TDest> NumericCast(const TSource source)
{
    auto underlying = static_cast<std::underlying_type_t<TSource>>(source);
    return NumericCast<TDest, NumBits>(underlying);
}
}    // namespace utils
}    // namespace ethosn
