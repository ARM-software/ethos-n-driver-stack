//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Log.hpp"

#include <ethosn_utils/Macros.hpp>
#include <scylla_addr_fields.h>
#include <scylla_regs.h>

#include <algorithm>
#include <array>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <string>

#define UNUSED(x) (void)(x)

// Asserts the given condition.
// Use this in preference to utils::Assert() as it will be compiled out depending on the CONTROL_UNIT_ASSERTS define.
#if defined(CONTROL_UNIT_ASSERTS)
#define ASSERT_MSG(cond, fmt, ...)                                                                                     \
    ethosn::control_unit::utils::Assert(cond, #cond, __FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#define ASSERT(cond) ethosn::control_unit::utils::Assert(cond, #cond, __FILE__, __LINE__, __func__)
#else
#define ASSERT_MSG(cond, fmt, ...) ETHOSN_UNUSED(cond)
#define ASSERT(cond) ETHOSN_UNUSED(cond)
#endif

// Fatal calls shall only be used for unrecoverable errors as they will never be compiled out and can therefore affect
// the performance of the released firmware. The message in the fatal call will be seen by the end user so that needs to
// be kept in mind when writing the message. When asserts are enabled, the fatal call will act as an assert to give more
// information about the issue and allow the debug monitor to be used.
#define FATAL_MSG(fmt, ...) FATAL_COND_MSG(false, fmt, ##__VA_ARGS__)
#if defined(CONTROL_UNIT_ASSERTS)
#define FATAL_COND_MSG(cond, fmt, ...) ASSERT_MSG(cond, "ERROR: " fmt, ##__VA_ARGS__)
#else
#define FATAL_COND_MSG(cond, fmt, ...) ethosn::control_unit::utils::FatalCond(cond, "ERROR: " fmt, ##__VA_ARGS__)
#endif

namespace ethosn
{
namespace control_unit
{
namespace utils
{

#if defined(CONTROL_UNIT_ASSERTS)

/// Callback function which can be set by the user of the library.
/// It is called whenever an assert fails.
using TAssertCallback =
    void (*)(const char* rep, const char* file, unsigned line, const char* function, const char* fmt, va_list args);
extern TAssertCallback g_AssertCallback;

// Leave Assert undefined if we haven't compiled with asserts enabled.
// This will cause compile errors to catch uses of asserts that don't use the ASSERT macro,
// and therefore wouldn't be compiled out as desired.

inline void Assert(bool condition,
                   const char* rep,
                   const char* file,
                   unsigned line,
                   const char* function,
                   const char* fmt = nullptr,
                   ...)
{
    if (g_AssertCallback && !condition)
    {
        va_list args;
        va_start(args, fmt);
        g_AssertCallback(rep, file, line, function, fmt, args);
        va_end(args);
    }
}

#else

void Fatal(const char* const fmt, va_list args);

inline void FatalCond(bool condition, const char* fmt, ...)
{
    if (!condition)
    {
        va_list args;
        va_start(args, fmt);
        Fatal(fmt, args);
        va_end(args);
    }
}

#endif
}    // namespace utils
}    // namespace control_unit
}    // namespace ethosn

#define ETHOSN_ASSERT_MSG(cond, msg) ASSERT_MSG(cond, msg)
#include <ethosn_utils/NumericCast.hpp>

namespace ethosn
{
namespace control_unit
{
namespace utils
{

// Helper class that provides array-like features (bounds-checking, iteration and indexing) on raw pointers/arrays.
template <typename T>
class ArrayRange
{
public:
    constexpr ArrayRange(T* begin, T* end)
        : m_Begin(begin)
        , m_End(end)
    {}

    constexpr ArrayRange(T* firstItem, const size_t numItems)
        : ArrayRange(firstItem, firstItem + numItems)
    {}

    // Lower case for usage in range-based for loop
    constexpr T* begin() const
    {
        return m_Begin;
    }

    // Lower case for usage in range-based for loop
    constexpr T* end() const
    {
        return m_End;
    }

    constexpr size_t GetSize() const
    {
        return static_cast<size_t>(m_End - m_Begin);
    }

    constexpr T& operator[](size_t idx) const
    {
        ASSERT_MSG(idx < GetSize(), "Index out of bounds (%i in array of size %i).", idx, GetSize());
        return m_Begin[idx];
    }

private:
    T* m_Begin;
    T* m_End;
};

void DefaultAssert(
    const char* rep, const char* file, unsigned line, const char* function, const char* const fmt, va_list args);

constexpr uint32_t CountLeadingZeros(uint32_t x)
{
#if defined(__GNUC__)
    return static_cast<uint32_t>(__builtin_clz(x));
#else
    uint32_t mask    = 0x80000000U;
    uint32_t counter = 0;
    while ((mask & x) ^ mask)
    {
        ++counter;
        mask = mask >> 1;
    }
    return counter;
#endif
}

const char* LookupRegisterName(uint32_t registerAddress);
#if !defined(CONTROL_UNIT_HARDWARE)
std::string GetRegisterName(uint32_t registerAddress);
#endif

template <typename HAL>
void LogUsefulRegisters(HAL& hal)
{
    static const uint32_t registers[] = {
        TOP_REG(DL2_RP, DL2_PWRCTLR),
        TOP_REG(DMA_RP, DMA_DMA_CHANNELS),
        TOP_REG(DMA_RP, DMA_DMA_COMP_CONFIG0),
        TOP_REG(DMA_RP, DMA_DMA_EMCS),
        TOP_REG(DMA_RP, DMA_DMA_RD_CMD),
        TOP_REG(DMA_RP, DMA_DMA_STRIDE0),
        TOP_REG(DMA_RP, DMA_DMA_STRIDE1),
        TOP_REG(DMA_RP, DMA_DMA_TOTAL_BYTES),
        TOP_REG(DMA_RP, DMA_DMA_WR_CMD),
        TOP_REG(DMA_RP, DMA_DRAM_ADDR_H),
        TOP_REG(DMA_RP, DMA_DRAM_ADDR_L),
        TOP_REG(DMA_RP, DMA_SRAM_ADDR),
        TOP_REG(GLOBAL_RP, GLOBAL_BLOCK_BANK_CONFIG),
        TOP_REG(GLOBAL_RP, GLOBAL_PLE_MCEIF_CONFIG),
        TOP_REG(GLOBAL_RP, GLOBAL_STRIPE_BANK_CONFIG),
        TOP_REG(GLOBAL_RP, GLOBAL_STRIPE_BANK_CONTROL),
        TOP_REG(PMU_RP, PMU_PMCNTENCLR),
        TOP_REG(PMU_RP, PMU_PMCR),
        TOP_REG(PMU_RP, PMU_PMINTENCLR),
        TOP_REG(PMU_RP, PMU_PMOVSCLR),
        TOP_REG(STRIPE_RP, CE_STRIPE_ACTIVATION_CONFIG),
        TOP_REG(STRIPE_RP, CE_STRIPE_CE_CONTROL),
        TOP_REG(STRIPE_RP, CE_STRIPE_DEPTHWISE_CONTROL),
        TOP_REG(STRIPE_RP, CE_STRIPE_FILTER),
        TOP_REG(STRIPE_RP, CE_STRIPE_IFM_BOTTOM_SLOTS),
        TOP_REG(STRIPE_RP, CE_STRIPE_IFM_CONFIG1),
        TOP_REG(STRIPE_RP, CE_STRIPE_IFM_CONFIG2_IG0),
        TOP_REG(STRIPE_RP, CE_STRIPE_IFM_DEFAULT_SLOT_SIZE),
        TOP_REG(STRIPE_RP, CE_STRIPE_IFM_MID_SLOTS),
        TOP_REG(STRIPE_RP, CE_STRIPE_IFM_PAD0_IG0),
        TOP_REG(STRIPE_RP, CE_STRIPE_IFM_PAD1_IG0),
        TOP_REG(STRIPE_RP, CE_STRIPE_IFM_PAD2_IG0),
        TOP_REG(STRIPE_RP, CE_STRIPE_IFM_PAD3_IG0),
        TOP_REG(STRIPE_RP, CE_STRIPE_IFM_ROW_STRIDE),
        TOP_REG(STRIPE_RP, CE_STRIPE_IFM_SLOT_BASE_ADDRESS_IG0),
        TOP_REG(STRIPE_RP, CE_STRIPE_IFM_SLOT_PAD_CONFIG),
        TOP_REG(STRIPE_RP, CE_STRIPE_IFM_SLOT_STRIDE),
        TOP_REG(STRIPE_RP, CE_STRIPE_IFM_TOP_SLOTS),
        TOP_REG(STRIPE_RP, CE_STRIPE_IFM_ZERO_POINT),
        TOP_REG(STRIPE_RP, CE_STRIPE_MUL_ENABLE_OG0),
        TOP_REG(STRIPE_RP, CE_STRIPE_OFM_CONFIG),
        TOP_REG(STRIPE_RP, CE_STRIPE_OFM_STRIPE_SIZE),
        TOP_REG(STRIPE_RP, CE_STRIPE_STRIPE_BLOCK_CONFIG),
        TOP_REG(STRIPE_RP, CE_STRIPE_VP_CONTROL),
        TOP_REG(STRIPE_RP, CE_STRIPE_WEIGHT_BASE_ADDR_OG0),
        TOP_REG(STRIPE_RP, CE_STRIPE_WIDE_KERNEL_CONTROL),
        TOP_REG(STRIPE_RP, CE_STRIPE_WIDE_KERNEL_OFFSET),
        TOP_REG(TSU_RP, TSU_TSU_CONTROL),
        TOP_REG(TSU_RP, TSU_TSU_EVENT_MSK),
    };

    for (uint32_t regAddress : registers)
    {
        const char* regName = LookupRegisterName(regAddress);
        uint32_t regValue   = hal.ReadReg(regAddress);
        if (regName)
        {
            hal.m_Logger.Info("%s = %x", regName, regValue);
        }
        else
        {
            hal.m_Logger.Info("%x = %x", regAddress, regValue);
        }
    }

    static const uint32_t ceRegisters[] = {
        CE_CE_ENABLES,   CE_PLE_CONTROL_0, CE_PLE_CONTROL_1,         CE_PLE_SCRATCH5,
        CE_PLE_SCRATCH7, CE_PLE_SETIRQ,    CE_PLE_UDMA_LOAD_COMMAND, CE_PLE_UDMA_LOAD_PARAMETERS,
    };

    for (uint32_t i = 0; i < hal.NumCes(); ++i)
    {
        for (uint32_t ceRegister : ceRegisters)
        {
            uint32_t regAddress = CE_REG(i, CE_RP, ceRegister);
            const char* regName = LookupRegisterName(regAddress);
            uint32_t regValue   = hal.ReadReg(regAddress);
            if (regName)
            {
                hal.m_Logger.Info("CE = %u, %s = %x", i, regName, regValue);
            }
            else
            {
                hal.m_Logger.Info("CE = %u, %x = %x", i, ceRegister, regValue);
            }
        }
    }
}

template <typename HAL>
void DisablePleMcuEvents(HAL& hal)
{
    hal.WriteReg(TOP_REG(CE_RP, CE_PLE_CONTROL_1), 0U);
}

template <typename HAL>
void EnablePleMcuEvents(HAL& hal)
{
    ple_control_1_r pleCtrl1;
    pleCtrl1.set_mcu_setevent(1);
    pleCtrl1.set_mceif_event(1);
    pleCtrl1.set_udma_event(1);
    pleCtrl1.set_txev_ncu(1);
    hal.WriteReg(TOP_REG(CE_RP, CE_PLE_CONTROL_1), pleCtrl1.word);
}

template <typename T1, typename T2>
constexpr auto DivRoundUp(const T1& numerator, const T2& denominator)
{
    return (numerator + denominator - 1) / denominator;
}

template <typename T1>
constexpr T1 RotateLeft(T1 val, T1 shift, T1 bits)
{
    ASSERT(shift < bits);
    T1 mask = (1U << bits) - 1;

    T1 ret = ((val << shift) & mask) | ((val & mask) >> (bits - shift));
    return ret & mask;
}

}    // namespace utils

}    // namespace control_unit
}    // namespace ethosn
