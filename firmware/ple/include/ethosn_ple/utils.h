//
// Copyright © 2018-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
// Utility macros
//

#pragma once

#include "Cmsis.hpp"

#include <algorithm>
#include <type_traits>
#include <utility>

#include <scylla_addr_fields.h>
#include <scylla_regs.h>

// This define may be set by the compiler to force instructions to be issued
// with bubbles thus making any forwarding hazard impossible.
// Default is to allow instruction pipelining
#ifndef COPRO_PIPELINE_DISABLE
#define COPRO_PIPELINE_DISABLE false
#endif

// =============================================================================
// Helper macros
// =============================================================================
#ifndef likely
#define likely(x) __builtin_expect((x) != 0, 1)
#endif
#ifndef unlikely
#define unlikely(x) __builtin_expect((x), 0)
#endif
#ifndef __inline_always
#define __inline_always inline __attribute__((__always_inline__))
#endif
#ifndef __noinline
#define __noinline __attribute__((noinline))
#endif
#ifndef UNUSED
#define UNUSED(x) ((void)(x))
#endif
#ifndef __weak_linkage
#define __weak_linkage __attribute__((__weak__))
#endif

namespace    // Internal linkage
{

// Constants from the spec

enum : unsigned
{
    SWZ_SHIFT_LOW_0,
    SWZ_SHIFT_LOW_1,
    SWZ_SHIFT_HIGH_0,
    SWZ_SHIFT_HIGH_1,
};

static constexpr unsigned CSEL_EQ = 0;
static constexpr unsigned CSEL_NE = 1;
static constexpr unsigned CSEL_CS = 2;
static constexpr unsigned CSEL_CC = 3;
static constexpr unsigned CSEL_MI = 4;
static constexpr unsigned CSEL_PL = 5;
static constexpr unsigned CSEL_VS = 6;
static constexpr unsigned CSEL_VC = 7;
static constexpr unsigned CSEL_HI = 8;
static constexpr unsigned CSEL_LS = 9;
static constexpr unsigned CSEL_GE = 10;
static constexpr unsigned CSEL_LT = 11;
static constexpr unsigned CSEL_GT = 12;
static constexpr unsigned CSEL_LE = 13;
static constexpr unsigned CSEL_AL = 14;

template <typename T>
concept Integral = std::is_integral_v<T>;

/// Extracts a single bit at the given index from the given value.
/// Bit indices start at 0 for the LSB, increasing towards the MSB.
constexpr unsigned int GetBit(unsigned int value, unsigned int bitIdx)
{
    return (value >> bitIdx) & 0b1;
}

// Tests
static_assert(GetBit(0b0101, 0) == 0b1, "Trailing bit");
static_assert(GetBit(0b0101, 1) == 0b0, "Zero in the middle");
static_assert(GetBit(0b0101, 2) == 0b1, "Leading one");
static_assert(GetBit(0b0101, 3) == 0b0, "Leading zero");

/// Helper function to construct a number from a list of bit values (0/1).
/// i.e. Bits(1, 0, 1) == 0b101.
/// This is used for the generated coprocessor instructions to generate easier-to-read code.
/// /// @{
template <typename BitType>
constexpr unsigned int Bits(BitType b)
{
    return static_cast<unsigned int>(b);
}

template <typename FirstBitType, typename... BitTypes>
constexpr unsigned int Bits(FirstBitType msb, BitTypes... bits)
{
    return static_cast<unsigned int>(msb) << sizeof...(bits) | Bits(bits...);
}

// Tests
static_assert(Bits(1) == 0b1, "Single bit");
static_assert(Bits(0) == 0b0, "Single bit");
static_assert(Bits(1, 0, 1, 1) == 0b1011, "Multiple bits");
static_assert(Bits(0, 1, 0, 1, 1) == 0b1011, "Leading zero");

/// @}

// The bit field defined in struct Cdp2Inst doesn't match the CPD2 encoding that you find
// in the regular Arm spec. Some bits of the coproc field have been borrowed to actually
// make the CRn, CRd and CRm field 5 bits.
struct Cdp2Inst
{
    // We need to split the 32 bits instruction code into 2 * 16 bits
    // because when we want to write some fields into the SRAM it might
    // be that the instruction is located in an address which is not
    // 32 bits aligned.
    // Thumb instructions (such as MOV) are 16 bits (one word) while
    // T32 instructions (such as CDP2) are 32 bits (2 half-word).
    struct High
    {
        constexpr High() = delete;

        uint16_t Rn_0_3 : 4;
        uint16_t Opc1 : 4;
        uint16_t : 8;
    };

    struct Low
    {
        constexpr Low() = delete;

        uint16_t Rm_0_3 : 4;
        uint16_t : 1;
        uint16_t Opc2 : 3;
        uint16_t Rm_4 : 1;
        uint16_t Rn_4 : 1;
        uint16_t Rd_4 : 1;
        uint16_t : 1;
        uint16_t Rd_0_3 : 4;
    };

    constexpr Cdp2Inst() = delete;

    constexpr void SetRm(uint16_t rm) volatile
    {
        m_Low.Rm_4   = (rm >> 4) & 0x1;
        m_Low.Rm_0_3 = rm & 0xf;
    }

    High m_High;
    Low m_Low;
};

constexpr bool IsPow2(const unsigned n)
{
    return (n > 0) && ((n & (n - 1U)) == 0U);
}

template <Integral T, Integral U>
constexpr std::common_type_t<T, U> DivRoundUp(const T numerator, const U denominator)
{
    constexpr std::common_type_t<T, U> one = 1;
    return (numerator + denominator - one) / denominator;
}

constexpr auto RoundUpToMultiple(const Integral auto x, const Integral auto y)
{
    return DivRoundUp(x, y) * y;
}

// Return the number of elements in the last iteration if `num` is split in iterations of `den` elements
constexpr unsigned LastIter(unsigned num, unsigned den)
{
    return ((num - 1) % den) + 1;
}
static_assert(LastIter(60, 4) == 4);
static_assert(LastIter(61, 4) == 1);
static_assert(LastIter(62, 4) == 2);
static_assert(LastIter(63, 4) == 3);
static_assert(LastIter(64, 4) == 4);

template <unsigned TimeA, unsigned TimeB, unsigned Distance = 1>
constexpr unsigned VePipelineDelay()
{
    static_assert(Distance > 0, "");
    constexpr unsigned atDistanceTimeB = Distance + TimeB;
    return std::max(TimeA + 1U, atDistanceTimeB) - atDistanceTimeB;
}

template <typename InstA, typename InstB, unsigned Distance = 1>
constexpr unsigned RwHazardDelay()
{
    return VePipelineDelay<InstA::WRITE_BACK, InstB::OP_READ, Distance>();
}

template <typename InstA, typename InstB, unsigned Distance = 1>
constexpr unsigned ReadInOrderDelay()
{
    return VePipelineDelay<InstA::OP_READ, InstB::OP_READ, Distance>();
}

template <typename InstA, typename InstB, unsigned Distance = 1>
constexpr unsigned WriteInOrderDelay()
{
    return VePipelineDelay<InstA::WRITE_BACK, InstB::WRITE_BACK, Distance>();
}

inline __attribute__((noreturn)) void Hang()
{
#pragma unroll 1
    while (true)
    {
        __WFI();
    }
}

template <typename T>
void WriteToRegisters(volatile void* const regPtr, const T& data)
{
    static_assert((sizeof(data) % sizeof(uint32_t)) == 0, "");

    const auto dst = reinterpret_cast<volatile uint8_t*>(regPtr);
    const auto src = reinterpret_cast<const uint8_t*>(&data);

    for (unsigned i = 0; i < sizeof(data); i += sizeof(uint32_t))
    {
        reinterpret_cast<volatile uint32_t&>(dst[i]) = reinterpret_cast<const uint32_t&>(src[i]);
    }
}

template <typename T>
void WriteToRegisters(const uint32_t regAddr, const T& data)
{
    WriteToRegisters(reinterpret_cast<volatile void*>(regAddr), data);
}

// =============================================================================
// Arm specific
// =============================================================================

// Specialization for CycleCount <= 0
template <int CycleCount>
__inline_always std::enable_if_t<(CycleCount <= 0)> nop()
{}
// Inserts NOPs taking cc number of clock cycles
template <int CycleCount = 1>
__inline_always std::enable_if_t<(CycleCount > 0)> nop()
{
    // The MCU is able to dual-issue NOP instructions, so a MOV is used to cause
    // a delay of the requested number of cycles
    __ASM volatile("MOV r0, r0");
    nop<CycleCount - 1>();
}

// =============================================================================
// Raw CDP, CDP2, MCR, MRC, MCRR
// =============================================================================

template <unsigned Coproc, unsigned Opc1, unsigned Opc2, unsigned Rd, unsigned Rn, unsigned Rm>
__inline_always void cdp()
{
    __ASM volatile("CDP p%c[cp], %[op1], c%c[rd], c%c[rn], c%c[rm], %[op2]"
                   : /* No outputs */
                   : [cp] "n"(Coproc), [op1] "n"(Opc1), [op2] "n"(Opc2), [rd] "n"(Rd), [rn] "n"(Rn), [rm] "n"(Rm));
}

template <unsigned Coproc, unsigned Opc1, unsigned Opc2, unsigned Rd, unsigned Rn, unsigned Rm>
__inline_always void cdp2()
{
    __ASM volatile("CDP2 p%c[cp], %[op1], c%c[rd], c%c[rn], c%c[rm], %[op2]"
                   : /* No outputs */
                   : [cp] "n"(Coproc), [op1] "n"(Opc1), [op2] "n"(Opc2), [rd] "n"(Rd), [rn] "n"(Rn), [rm] "n"(Rm));
}

template <unsigned Coproc, unsigned Opc1, unsigned Opc2, unsigned Rn, unsigned Rm, typename T>
__inline_always void mcr(const T val)
{
    static_assert(sizeof(val) <= 4, "");

    __ASM volatile("MCR p%c[cp], %[op1], %[rt], c%c[rn], c%c[rm], %[op2]"
                   : /* No outputs */
                   : [cp] "n"(Coproc), [op1] "n"(Opc1), [op2] "n"(Opc2), [rt] "r"(val), [rn] "n"(Rn), [rm] "n"(Rm));
}

template <unsigned Coproc, unsigned Opc1, unsigned Opc2, unsigned Rn, unsigned Rm, typename T>
__inline_always void mcr2(const T val)
{
    static_assert(sizeof(val) <= 4, "");

    __ASM volatile("MCR2 p%c[cp], %[op1], %[rt], c%c[rn], c%c[rm], %[op2]"
                   : /* No outputs */
                   : [cp] "n"(Coproc), [op1] "n"(Opc1), [op2] "n"(Opc2), [rt] "r"(val), [rn] "n"(Rn), [rm] "n"(Rm));
}

template <unsigned Coproc, unsigned Opc1, unsigned Opc2, unsigned Rn, unsigned Rm, unsigned Rt = 0>
__inline_always void mcr()
{
    __ASM volatile("MCR p%c[cp], %[op1], r%c[rt], c%c[rn], c%c[rm], %[op2]"
                   : /* No outputs */
                   : [cp] "n"(Coproc), [op1] "n"(Opc1), [op2] "n"(Opc2), [rt] "n"(Rt), [rn] "n"(Rn), [rm] "n"(Rm));
}

template <unsigned Coproc, unsigned Opc1, unsigned Opc2, unsigned Rn, unsigned Rm, unsigned Rt = 0>
__inline_always void mcr2()
{
    __ASM volatile("MCR2 p%c[cp], %[op1], r%c[rt], c%c[rn], c%c[rm], %[op2]"
                   : /* No outputs */
                   : [cp] "n"(Coproc), [op1] "n"(Opc1), [op2] "n"(Opc2), [rt] "n"(Rt), [rn] "n"(Rn), [rm] "n"(Rm));
}

template <unsigned Coproc, unsigned Opc1, unsigned Opc2, unsigned Rn, unsigned Rm, typename T = unsigned>
__inline_always decltype(auto) mrc()
{
    std::remove_cv_t<std::remove_reference_t<T>> val;

    static_assert(sizeof(val) <= 4, "");

    __ASM volatile("MRC p%c[cp], %[op1], %[rt], c%c[rn], c%c[rm], %[op2]"
                   : [rt] "=r"(val)
                   : [cp] "n"(Coproc), [op1] "n"(Opc1), [op2] "n"(Opc2), [rn] "n"(Rn), [rm] "n"(Rm));

    return val;
}

template <unsigned Coproc, unsigned Opc1, unsigned Rm, typename T, typename T2>
__inline_always void mcrr(const T val, const T2 val2)
{
    static_assert(sizeof(val) <= 4, "");
    static_assert(sizeof(val2) <= 4, "");

    __ASM volatile("MCRR p%c[cp], %[op1], %[rt], %[rt2], c%c[rm]"
                   : /* No outputs */
                   : [cp] "n"(Coproc), [op1] "n"(Opc1), [rt] "r"(val), [rt2] "r"(val2), [rm] "n"(Rm));
}

template <unsigned Coproc, unsigned Opc1, unsigned Rm, unsigned Rt2 = 0, typename T>
__inline_always void mcrr(const T val)
{
    static_assert(sizeof(val) <= 4, "");

    __ASM volatile("MCRR p%c[cp], %[op1], %[rt], r%c[rt2], c%c[rm]"
                   : /* No outputs */
                   : [cp] "n"(Coproc), [op1] "n"(Opc1), [rt] "r"(val), [rt2] "n"(Rt2), [rm] "n"(Rm));
}

// =============================================================================
// Static loop support
// =============================================================================

namespace static_loop
{

//
// Use to declare a function object type that forwards calls to an existing templated function.
//
#define STATIC_LOOP_FN_WRAPPER(Wrapper, fn)                                                                            \
    struct Wrapper                                                                                                     \
    {                                                                                                                  \
        template <int... Is, typename... Args>                                                                         \
        void operator()(Args&&... args) const                                                                          \
        {                                                                                                              \
            fn<Is...>(std::forward<Args>(args)...);                                                                    \
        }                                                                                                              \
    }

//
// Represents a range of integers for use with static_loop::For.
//
// Example:
//
//     using namespace static_loop;
//     For<Range<0, 16, 2>>::Invoke(fn, arg); // Expands to fn.operator()<0>(arg); fn.operator()<2>(arg); ...
//
template <int _Start, int _End, int _Step = 1>
struct Range
{
    static_assert(_Step != 0, "");

    static constexpr int Start  = _Start;
    static constexpr int End    = _End;
    static constexpr int Step   = _Step;
    static constexpr unsigned N = (Step > 0) ? ((std::max(Start, End) - Start + Step - 1) / Step)
                                             : ((std::max(Start, End) - End - Step - 1) / -Step);

    using Next = Range<Start + Step, End, Step>;
};

template <int _Start, int _End, int _Step>
constexpr int Range<_Start, _End, _Step>::Start;

template <int _Start, int _End, int _Step>
constexpr int Range<_Start, _End, _Step>::End;

template <int _Start, int _End, int _Step>
constexpr int Range<_Start, _End, _Step>::Step;

template <int _Start, int _End, int _Step>
constexpr unsigned Range<_Start, _End, _Step>::N;

static_assert(Range<0, 5, 2>::N == 3, "");
static_assert(Range<5, 0, -2>::N == 3, "");

template <int _Start, unsigned _N, int _Step = 1>
using RangeN = Range<_Start, _Start + (_N * _Step), _Step>;

// Implementation details
namespace impl
{

// Function object type that binds I as the first argument to forwarded calls to the wrapped object
template <int I, typename Fn>
struct IndexBinder
{
    Fn&& m_Fn;

    constexpr IndexBinder(Fn&& fn)
        : m_Fn(fn)
    {}

    template <typename... Args>
    constexpr decltype(auto) operator()(Args&&... args) const
    {
        return m_Fn(I, std::forward<Args>(args)...);
    }

    template <int J, int... Js, typename... Args>
    constexpr decltype(auto) operator()(Args&&... args) const
    {
        return m_Fn.template operator()<I, J, Js...>(std::forward<Args>(args)...);
    }
};

// Create an IndexBinder for a callable
template <int I, typename Fn>
constexpr IndexBinder<I, Fn> Bind(Fn&& fn)
{
    return { std::forward<Fn>(fn) };
}

// Helper struct for static for-loop recursion. Recursion ends when a range is empty or there are no further ranges.
template <bool, typename... _Ranges>
struct ForImpl
{
    static constexpr void InvokeIf(...)
    {}

    static constexpr void Invoke(...)
    {}
};

}    // namespace impl

//
// Member static methods Invoke/InvokeIf iterate over the cartesian product of ranges.
//
// Example:
//
//     using namespace static_loop;
//
//     For<Range<0, 2>, Range<1, 3>>::Invoke(fn, arg); // Expands to:
//                                                     //     fn.operator()<0, 1>(arg);
//                                                     //     fn.operator()<0, 2>(arg);
//                                                     //     fn.operator()<1, 1>(arg);
//                                                     //     fn.operator()<1, 2>(arg);
//
//     For<Range<0, 2>, Range<1, 3>>::InvokeIf(cond, fn, arg); // Expands to:
//                                                             //     if (cond(0, 1)) fn.operator()<0, 1>(arg);
//                                                             //     if (cond(0, 2)) fn.operator()<0, 2>(arg);
//                                                             //     if (cond(1, 1)) fn.operator()<1, 1>(arg);
//                                                             //     if (cond(1, 2)) fn.operator()<1, 2>(arg);
//
template <typename... _Ranges>
using For = impl::ForImpl<std::min({ true, (_Ranges::N > 0)... }), _Ranges...>;

// Helper struct for static for-loop recursion: specialization for the in-range case.
template <typename _Range, typename... _Ranges>
struct impl::ForImpl<true, _Range, _Ranges...>
{
    template <typename Cond, typename Fn, typename... Args>
    static constexpr __inline_always void InvokeIf(Cond&& cond, Fn&& fn, Args&&... args)
    {
        // Run the body of the loop, only if this is a leaf node in the call graph.
        const int n = sizeof...(_Ranges);
        if (n == 0)
        {
            // Expansion for the current indices
            if (cond(_Range::Start, _Ranges::Start...))
            {
                // cppcheck-suppress constStatement
                fn.template operator()<_Range::Start, _Ranges::Start...>(std::forward<Args>(args)...);
            }
        }

        // Bind the first index and recurse with one dimension less
        // cppcheck-suppress accessForwarded
        For<_Ranges...>::InvokeIf(Bind<_Range::Start>(cond), Bind<_Range::Start>(fn), std::forward<Args>(args)...);
        // Advance the outer dimension
        // cppcheck-suppress accessForwarded
        For<typename _Range::Next, _Ranges...>::InvokeIf(cond, fn, std::forward<Args>(args)...);
    }

    template <typename Fn, typename... Args>
    static constexpr __inline_always void Invoke(Fn&& fn, Args&&... args)
    {
        struct TrueFn
        {
            constexpr bool operator()(...) const
            {
                return true;
            }
        };

        // We rely on the compiler to optimise away the condition check
        InvokeIf(TrueFn{}, fn, std::forward<Args>(args)...);
    }
};

}    // namespace static_loop

enum class Event : size_t
{
    SETIRQ_EVENT    = 4,
    BLOCK_DONE      = 8,
    UDMA_LOAD_DONE  = 22,
    UDMA_STORE_DONE = 23,
};

template <typename Enum, typename T = uint32_t>
class EnumBitset
{
    static_assert(std::is_same<std::underlying_type_t<Enum>, size_t>::value, "");
    static_assert(std::is_unsigned<T>::value, "");

public:
    constexpr EnumBitset()
        : EnumBitset{ 0 }
    {}

    explicit constexpr EnumBitset(const T bits)
        : m_Bits{ bits }
    {}

    template <typename U>
    constexpr EnumBitset(const EnumBitset<Enum, U>& other)
        : m_Bits{ other.GetBits() }
    {}

    template <typename U>
    constexpr EnumBitset& operator=(const EnumBitset<Enum, U>& other)
    {
        m_Bits = T{ other.GetBits() };
        return *this;
    }

    template <typename U>
    constexpr EnumBitset& operator|=(const EnumBitset<Enum, U>& other)
    {
        m_Bits |= T{ other.GetBits() };
        return *this;
    }

    constexpr EnumBitset& Set(const Enum bit, const bool value)
    {
        if (value)
        {
            m_Bits = static_cast<T>(m_Bits | MaskOf(bit));
        }
        else
        {
            m_Bits = static_cast<T>(m_Bits & ~MaskOf(bit));
        }

        return *this;
    }

    constexpr bool Get(const Enum bit) const
    {
        return (m_Bits & MaskOf(bit)) != 0;
    }

    constexpr bool operator[](const Enum bit) const
    {
        return Get(bit);
    }

    constexpr auto operator[](const Enum bit)
    {
        return BitReference(*this, bit);
    }

    constexpr T GetBits() const
    {
        return m_Bits;
    }

private:
    class BitReference
    {
    public:
        constexpr BitReference(EnumBitset& bitset, const Enum bit)
            : m_Bitset(bitset)
            , m_Bit(bit)
        {}

        constexpr const BitReference& operator=(const bool value) const
        {
            m_Bitset.Set(m_Bit, value);
            return *this;
        }

        constexpr const BitReference& operator|=(const bool value) const
        {
            return ((*this) = (*this) || value);
        }

        constexpr operator bool() const
        {
            return m_Bitset.Get(m_Bit);
        }

    private:
        EnumBitset& m_Bitset;
        const Enum m_Bit;
    };

    static constexpr T MaskOf(const Enum bit)
    {
        return static_cast<T>(1U << static_cast<size_t>(bit));
    }

    T m_Bits;
};

/// Waits until a specific HW event has happened since this function was last called.
template <Event E>
void WaitForEvent(EnumBitset<Event>& activeEvents)
{
    auto& statusReg = *reinterpret_cast<const volatile uint32_t*>(PLE_REG(CE_RP, CE_PLE_STATUS));

    constexpr EnumBitset<Event> mask = EnumBitset<Event>{}.Set(E, true);

    unsigned tmp = 0;

    __ASM volatile("0:\n"
                   "LDR %[tmp],%[statusReg]\n"
                   "ORR %[active],%[tmp]\n"
                   "TST %[active],%[mask]\n"
                   "ITT EQ\n"
                   "WFEEQ\n"
                   "BEQ 0b\n"
                   "BIC %[active],%[mask]\n"
                   : [active] "+r"(activeEvents), [tmp] "=&r"(tmp)
                   : [statusReg] "m"(statusReg), [mask] "n"(mask.GetBits())
                   : "cc");
}

}    // namespace
