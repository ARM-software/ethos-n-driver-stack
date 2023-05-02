//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "hw.h"
#include "utils.h"

#include <cstdint>
#include <utility>

namespace    // Internal linkage
{
namespace VE_TIMING
{
struct LOAD_INRAM_RF
{
    constexpr static unsigned int OP_READ    = 2;
    constexpr static unsigned int WRITE_BACK = 4;
    constexpr static unsigned int PIPELINE   = 1;
};

struct LOAD_HALF_INRAM_RF
{
    constexpr static unsigned int OP_READ    = 2;
    constexpr static unsigned int WRITE_BACK = 4;
    constexpr static unsigned int PIPELINE   = 1;
};

struct LOAD_OUTRAM_RF
{
    constexpr static unsigned int OP_READ    = 2;
    constexpr static unsigned int WRITE_BACK = 4;
    constexpr static unsigned int PIPELINE   = 1;
};

struct LOAD_HALF_OUTRAM_RF
{
    constexpr static unsigned int OP_READ    = 2;
    constexpr static unsigned int WRITE_BACK = 4;
    constexpr static unsigned int PIPELINE   = 1;
};

struct STORE_RF_OUTRAM
{
    constexpr static unsigned int OP_READ    = 1;
    constexpr static unsigned int WRITE_BACK = 3;
    constexpr static unsigned int PIPELINE   = 1;
};

struct STORE_HALF_RF_OUTRAM
{
    constexpr static unsigned int OP_READ    = 1;
    constexpr static unsigned int WRITE_BACK = 3;
    constexpr static unsigned int PIPELINE   = 1;
};

struct LOAD_MCU_RF
{
    constexpr static unsigned int OP_READ    = 0;
    constexpr static unsigned int WRITE_BACK = 1;
    constexpr static unsigned int PIPELINE   = 1;
};

struct STORE_RF_MCU
{
    constexpr static unsigned int OP_READ    = 0;
    constexpr static unsigned int WRITE_BACK = 1;
    constexpr static unsigned int PIPELINE   = 1;
};
}    // namespace VE_TIMING

namespace lsu
{

struct Address
{
    unsigned ramAddr : 16;
    unsigned rfAddr : 16;
};

struct Stride
{
    unsigned ramStride : 16;
    unsigned rfStride : 16;
};

inline Address operator+(const Address addr, const Stride stride)
{
    Address result;
    reinterpret_cast<unsigned&>(result) =
        reinterpret_cast<const unsigned&>(addr) + reinterpret_cast<const unsigned&>(stride);
    return result;
}

inline Address operator+(const Address addr, const unsigned stride)
{
    return addr + Stride{ stride, stride };
}

template <typename T>
inline Address& operator+=(Address& addr, const T stride)
{
    return (addr = addr + stride);
}

constexpr Stride operator*(const Stride stride, const unsigned scale)
{
    return { stride.ramStride * scale, stride.rfStride * scale };
}

constexpr Stride operator*(const unsigned scale, const Stride stride)
{
    return stride * scale;
}

constexpr Stride& operator*=(Stride& stride, const unsigned scale)
{
    return (stride = stride * scale);
}

namespace impl
{

constexpr unsigned CP_NUM = 6;
constexpr unsigned TIMING = 4 - 1;    // One cycle spent on issuing the instruction

// MCR opcodes
constexpr unsigned STORE_RF_OUTFRAM_OPC1     = 0b100;
constexpr unsigned STORE_HALF_RF_OUTRAM_OPC1 = 0b101;
constexpr unsigned LOAD_MCU_RF_OPC1          = 0b111;

// MCRR opcodes
constexpr unsigned LOAD_OUTFRAM_RF_OPC1     = 0b000;
constexpr unsigned LOAD_HALF_OUTRAM_RF_OPC1 = 0b001;
constexpr unsigned LOAD_INRAM_RF_OPC1       = 0b010;
constexpr unsigned LOAD_HALF_INRAM_RF_OPC1  = 0b011;

// MRC opcodes
constexpr unsigned STORE_RF_MCU_OPC1 = 0b111;

constexpr std::pair<unsigned, unsigned> GetAddrOffset(const unsigned regOffset, const unsigned maxRegOffset)
{
    const unsigned newRegOffset = regOffset % (maxRegOffset + 1U);
    const unsigned addrOffset   = (regOffset - newRegOffset) * WORDS_PER_REGISTER;
    return { newRegOffset, addrOffset };
}

}    // namespace impl

template <unsigned I>
void LoadInramRf(const unsigned ramId, const Address addr)
{
    static_assert((I % 2) == 0, "");

    using namespace impl;

    constexpr auto offset = GetAddrOffset(I, 15);
    mcrr<CP_NUM, LOAD_INRAM_RF_OPC1, offset.first / 2>(addr + offset.second, ramId);
    nop<COPRO_PIPELINE_DISABLE ? TIMING : 0>();
}

template <unsigned I>
void LoadHalfInramRf(const unsigned ramId, const Address addr)
{
    using namespace impl;

    constexpr auto offset = GetAddrOffset(I, 7);
    mcrr<CP_NUM, LOAD_HALF_INRAM_RF_OPC1, offset.first>(addr + offset.second, ramId);
    nop<COPRO_PIPELINE_DISABLE ? TIMING : 0>();
}

template <unsigned I>
void LoadOutramRf(const Address addr)
{
    static_assert((I % 2) == 0, "");

    using namespace impl;

    constexpr auto offset = impl::GetAddrOffset(I, 15);
    mcrr<CP_NUM, LOAD_OUTFRAM_RF_OPC1, offset.first / 2>(addr + offset.second);
    nop<COPRO_PIPELINE_DISABLE ? TIMING : 0>();
}

template <unsigned I>
void LoadHalfOutramRf(const Address addr)
{
    using namespace impl;

    constexpr auto offset = GetAddrOffset(I, 7);
    mcrr<CP_NUM, LOAD_HALF_OUTRAM_RF_OPC1, offset.first>(addr + offset.second);
    nop<COPRO_PIPELINE_DISABLE ? TIMING : 0>();
}

STATIC_LOOP_FN_WRAPPER(LoadInramRfFn, LoadInramRf);
STATIC_LOOP_FN_WRAPPER(LoadHalfInramRfFn, LoadHalfInramRf);

STATIC_LOOP_FN_WRAPPER(LoadOutramRfFn, LoadOutramRf);
STATIC_LOOP_FN_WRAPPER(LoadHalfOutramRfFn, LoadHalfOutramRf);

template <unsigned I>
void StoreRfOutram(const Address addr)
{
    static_assert((I % 2) == 0, "");

    constexpr auto offset = impl::GetAddrOffset(I, 15);
    mcr<impl::CP_NUM, impl::STORE_RF_OUTFRAM_OPC1, offset.first / 2, 0, 0>(addr + offset.second);
    nop<COPRO_PIPELINE_DISABLE ? impl::TIMING : 0>();
}

template <unsigned I>
void StoreHalfRfOutram(const Address addr)
{
    constexpr auto offset = impl::GetAddrOffset(I, 7);
    mcr<impl::CP_NUM, impl::STORE_HALF_RF_OUTRAM_OPC1, offset.first, 0, 0>(addr + offset.second);
    nop<COPRO_PIPELINE_DISABLE ? impl::TIMING : 0>();
}

STATIC_LOOP_FN_WRAPPER(StoreRfOutramFn, StoreRfOutram);
STATIC_LOOP_FN_WRAPPER(StoreHalfRfOutramFn, StoreHalfRfOutram);

// MCU_RAM -> RF
template <unsigned RfAddr, unsigned Offset = 0, typename T = unsigned>
void LoadMcuRf(const T data)
{
    static_assert(RfAddr < (NUM_REGISTERS * WORDS_PER_REGISTER), "Register address not supported");

    mcr<impl::CP_NUM, impl::LOAD_MCU_RF_OPC1, Offset, (RfAddr >> 4), (RfAddr & 0xFU)>(data);
    nop<COPRO_PIPELINE_DISABLE ? impl::TIMING : 0>();
}

// RF -> MCU_RAM
template <unsigned RfAddr, unsigned Offset = 0, typename T = unsigned>
T&& StoreRfMcu(T&& ret = {})
{
    static_assert(RfAddr < (NUM_REGISTERS * WORDS_PER_REGISTER), "Register address not supported");

    ret = mrc<impl::CP_NUM, impl::STORE_RF_MCU_OPC1, Offset, (RfAddr >> 4), (RfAddr & 0xFU), decltype(ret)>();
    nop<COPRO_PIPELINE_DISABLE ? impl::TIMING : 0>();
    return std::forward<T>(ret);
}

}    // namespace lsu

// =============================================================================
// Register file read
// =============================================================================

template <typename T>
struct alignas(uint32_t) RfReg
{
    using Row = T[4];

    uint8_t data[4][4][sizeof(T)];

    auto& operator[](const size_t i) const
    {
        return reinterpret_cast<const Row&>(data[i]);
    }

    auto& operator[](const size_t i)
    {
        return reinterpret_cast<Row&>(data[i]);
    }
};

template <unsigned BaseReg>
struct ReadRfWordFn
{
    template <unsigned I, unsigned J, typename T>
    void operator()(RfReg<T>& reg)
    {
        const uint32_t word = lsu::StoreRfMcu<WORDS_PER_REGISTER*(BaseReg + I), J>();

        for (size_t k = 0U; k < 4U; ++k)
        {
            auto& row         = reinterpret_cast<const uint8_t(&)[4U]>(word);
            reg.data[J][k][I] = row[k];
        }
    }
};

template <unsigned I, typename T = uint8_t>
__attribute__((noinline)) RfReg<T> ReadRfReg()
{
    using namespace static_loop;

    RfReg<T> reg;

    For<RangeN<0, sizeof(T)>, RangeN<0, WORDS_PER_REGISTER>>::Invoke(ReadRfWordFn<I>{}, reg);

    return reg;
}

}    // namespace
