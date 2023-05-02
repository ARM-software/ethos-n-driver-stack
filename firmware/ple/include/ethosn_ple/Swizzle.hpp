//
// Copyright © 2018-2020 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
// Swz config helpers
//

#pragma once

#include <generated/mcr_opcodes.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace
{
template <typename T, size_t N>
class alignas(T) BitFieldArray
{
public:
    using ElementT                    = uint8_t;
    static constexpr unsigned NumBits = (CHAR_BIT * sizeof(T)) / N;

    static_assert(NumBits <= (CHAR_BIT * sizeof(ElementT)), "");
    static_assert((CHAR_BIT * sizeof(T)) == (N * NumBits), "");

    static constexpr BitFieldArray Dup(const ElementT value)
    {
        BitFieldArray bfa(value);

        for (size_t i = 1U; i < N; i *= 2U)
        {
            bfa.m_Word |= bfa.m_Word << (i * NumBits);
        }

        return bfa;
    }

    template <typename... Us>
    constexpr BitFieldArray(const Us... us)
        : BitFieldArray(std::index_sequence_for<Us...>{}, ElementT{ us }...)
    {}

    constexpr operator T() const
    {
        return m_Word;
    }

    constexpr BitFieldArray operator|(const BitFieldArray rhs) const
    {
        BitFieldArray ret;
        ret.m_Word = m_Word | rhs.m_Word;
        return ret;
    }

    constexpr ElementT Get(const size_t i) const
    {
        return static_cast<ElementT>(m_Word >> (i * NumBits)) & static_cast<ElementT>((1U << NumBits) - 1U);
    }

    constexpr void Set(const size_t i, const ElementT value)
    {
        const T mask = static_cast<T>((1U << NumBits) - 1U);
        m_Word       = (m_Word & ~(mask << (i * NumBits))) | ((value & mask) << (i * NumBits));
    }

private:
    template <size_t... Is, typename... Us>
    constexpr BitFieldArray(std::index_sequence<Is...>, const Us... us)
        : m_Word{}
    {
        (void)std::initializer_list<int>{ (Set(Is, us), 0)... };
    }

    T m_Word;
};

using SwzRegSel        = BitFieldArray<uint32_t, 16>;
using HalfSwzSubRegSel = BitFieldArray<uint32_t, 8>;
using SwzSubRegSel     = BitFieldArray<uint64_t, 16>;

template <typename RegSelT>
constexpr RegSelT ToSwzRegSelT(const typename RegSelT::ElementT (&regSel)[4][4])
{
    return {
        // clang-format off
        regSel[0][0], regSel[0][1], regSel[0][2], regSel[0][3],
        regSel[1][0], regSel[1][1], regSel[1][2], regSel[1][3],
        regSel[2][0], regSel[2][1], regSel[2][2], regSel[2][3],
        regSel[3][0], regSel[3][1], regSel[3][2], regSel[3][3],
        // clang-format on
    };
}

constexpr SwzRegSel ToSwzRegSel(const SwzRegSel::ElementT (&regSel)[4][4])
{
    return ToSwzRegSelT<SwzRegSel>(regSel);
}

constexpr SwzSubRegSel ToSwzSubRegSel(const SwzSubRegSel::ElementT (&subRegSel)[4][4])
{
    return ToSwzRegSelT<SwzSubRegSel>(subRegSel);
}

constexpr HalfSwzSubRegSel ToHalfSwzSubRegSel(const HalfSwzSubRegSel::ElementT (&subRegSel)[2][4])
{
    return {
        // clang-format off
        subRegSel[0][0], subRegSel[0][1], subRegSel[0][2], subRegSel[0][3],
        subRegSel[1][0], subRegSel[1][1], subRegSel[1][2], subRegSel[1][3],
        // clang-format on
    };
}

template <typename RegSelT>
constexpr RegSelT Transpose(const RegSelT regSel)
{
    return ToSwzRegSelT<RegSelT>({
        { regSel.Get(0), regSel.Get(4), regSel.Get(8), regSel.Get(12) },
        { regSel.Get(1), regSel.Get(5), regSel.Get(9), regSel.Get(13) },
        { regSel.Get(2), regSel.Get(6), regSel.Get(10), regSel.Get(14) },
        { regSel.Get(3), regSel.Get(7), regSel.Get(11), regSel.Get(15) },
    });
}

template <unsigned SwzId>
void SetSwzRegSel(const SwzRegSel regSel)
{
    ve_set_swzsel_reg_sel<SwzId>(regSel);
}

template <unsigned SwzId>
void SetSwzSubRegSel(const HalfSwzSubRegSel low, const HalfSwzSubRegSel high)
{
    ve_set_swzsel_subreg_sel<SwzId>(low, high);
}

template <unsigned SwzId>
void SetSwzSubRegSel(const SwzSubRegSel subRegSel)
{
    ve_set_swzsel_subreg_sel<SwzId>(subRegSel, subRegSel >> 32U);
}

}    // namespace
