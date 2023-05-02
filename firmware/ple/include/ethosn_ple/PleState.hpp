//
// Copyright © 2018-2020,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "Sizes.hpp"
#include "hw.h"
#include "utils.h"

namespace
{
using MceBlockSize = sizes::BlockSize<PATCHES_PER_BLOCK_X, PATCHES_PER_BLOCK_Y>;

class PleState
{
public:
    // clang-format off
    PleState()
        : m_ActiveEvents{}
        , m_InramAddr{}
        , m_PleCounters{}
    {}

    EnumBitset<Event>& GetActiveEvents()
    {
        return m_ActiveEvents;
    }

    /// Waits until a specific HW event has happened since this method was last called.
    template <Event E>
    void WaitForEvent()
    {
        ::WaitForEvent<E>(m_ActiveEvents);
    }

    /// Waits for at least one MCE block to be in input SRAM since last call on Advance.
    void WaitForOneBlock()
    {
        const unsigned pleCounters = m_PleCounters;

        unsigned hwPleCounters = read_reg(CE_PLE_COUNTERS);

#pragma unroll 1
        while (hwPleCounters == pleCounters)
        {
            __WFE();
            hwPleCounters = read_reg(CE_PLE_COUNTERS);
        }
    }

    /// Waits for at least n MCE blocks to be in input SRAM since last call on Advance.
    void WaitForBlocks(unsigned n)
    {
        const unsigned tgtPleCounters = m_PleCounters + n;

        unsigned hwPleCounters = read_reg(CE_PLE_COUNTERS);

#pragma unroll 1
        while (static_cast<int8_t>(hwPleCounters - tgtPleCounters) < 0)
        {
            __WFE();
            hwPleCounters = read_reg(CE_PLE_COUNTERS);
        }
    }

    /// Advance internal pointers to data in input SRAM.
    unsigned Advance(const unsigned numMceBlocks)
    {
        constexpr unsigned WORDS_IN_MCE_BLOCK = WORDS_PER_REGISTER * TotalSize(MceBlockSize{});

        const unsigned oldInramAddr = m_InramAddr;

        m_InramAddr += WORDS_IN_MCE_BLOCK * numMceBlocks;
        m_PleCounters += numMceBlocks;

        return oldInramAddr;
    }

private:
    EnumBitset<Event> m_ActiveEvents;

    uint16_t m_InramAddr;
    uint8_t m_PleCounters;
};
}    // namespace
