//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#if (__cplusplus >= 202002L) || (_MSVC_LANG >= 202002L)

#include <bit>

#else

#include <cstddef>
#include <cstring>
#include <type_traits>

namespace std
{

template <typename To, typename From>
To bit_cast(const From& src) noexcept
{
    static_assert((sizeof(To) == sizeof(From)) && std::is_trivially_copyable<From>::value &&
                  std::is_trivially_copyable<To>::value);
    To dst;
    std::memcpy(reinterpret_cast<std::byte*>(&dst), reinterpret_cast<const std::byte*>(&src), sizeof(To));
    return dst;
}

}    // namespace std

#endif
