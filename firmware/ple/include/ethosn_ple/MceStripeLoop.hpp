//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "Common.hpp"
#include "Input.hpp"
#include "PleState.hpp"

namespace
{

template <typename Operator, unsigned BlocksWait = k_BlockMultiplier, unsigned BlocksAdvance = k_BlockMultiplier>
class MceStripeLoop
{
    static constexpr bool FORCE_EDGE = BlocksWait != BlocksAdvance;

public:
    using Input = input::MceInput<BlocksWait, BlocksAdvance>;

    static constexpr Xy GetSizeInPatches(const Xy sizeInElements)
    {
        return DivRoundUp(sizeInElements, Xy::Dup(ELEMENTS_PER_PATCH_1D));
    }

    static constexpr Xy GetNumFullBlocks(const Xy sizeInElements)
    {
        const Xy sizeInPatches = GetSizeInPatches(sizeInElements);
        return FORCE_EDGE ? (sizeInPatches - Xy::Dup(1U)) / Xy(BlockSize{}) : sizeInPatches / Xy(BlockSize{});
    }

    static constexpr Xy GetNumEdgePatches(const Xy sizeInElements)
    {
        const Xy sizeInPatches = GetSizeInPatches(sizeInElements);
        return FORCE_EDGE ? ((sizeInPatches - Xy::Dup(1U)) % Xy(BlockSize{})) + Xy::Dup(1U)
                          : sizeInPatches % Xy(BlockSize{});
    }

    MceStripeLoop(PleState& pleState, const OperatorInfo& opInfo)
        : m_PleState(pleState)
        , m_OpInfo(opInfo)
        , m_NumFullBlocks(GetNumFullBlocks(Xy(m_OpInfo.sizeInElements)))
        , m_NumEdgePatches(GetNumEdgePatches(Xy(m_OpInfo.sizeInElements)))
        , m_DepthForThisCe(DivRoundUp(std::max(m_OpInfo.sizeInElements.z, g_CeId) - g_CeId, NUM_CES))
    {}

    ncu_ple_interface::PleMsg::StripeDone operator()() const
    {
        Input input(m_PleState);
        Operator op(m_PleState, m_OpInfo);

        const bool hasEdgeX = FORCE_EDGE || (m_NumEdgePatches.x != 0U);

        for (int z = static_cast<int>(m_DepthForThisCe); z > 0; z -= m_OpInfo.numActiveOgs)
        {
            const unsigned numActiveOgs =
                (NUM_MCEIF > 1) ? std::min(static_cast<unsigned>(z), m_OpInfo.numActiveOgs) : 1U;

            if (numActiveOgs == 1U)
            {
                SetPleLanesInUse(1U);
            }

#pragma unroll 1
            for (unsigned y = m_NumFullBlocks.y; y > 0; --y)
            {
                for (unsigned x = m_NumFullBlocks.x; x > 0; --x)
                {
                    const Xyz pos = Xyz(m_NumFullBlocks, m_DepthForThisCe) - Xyz(x, y, static_cast<unsigned>(z));

                    const unsigned inramAddr = input.WaitForFullWidthBlock();

                    op.ProcessFullBlock(0, numActiveOgs, inramAddr, pos);

                    input.SignalFullWidthBlockFreed();
                }

                if (hasEdgeX)
                {
                    const Xyz pos = Xyz(m_NumFullBlocks, m_DepthForThisCe) - Xyz(0, y, static_cast<unsigned>(z));

                    const unsigned inramAddr = input.WaitForPartialWidthBlock(m_NumEdgePatches.x);

                    op.ProcessPartialWidthBlock(0, numActiveOgs, inramAddr, pos, m_NumEdgePatches.x);

                    input.SignalPartialWidthBlockFreed(m_NumEdgePatches.x);
                }

                op.NextRow(numActiveOgs, m_NumFullBlocks.y - y);
            }

            if (m_NumEdgePatches.y > 0)
            {
                for (unsigned x = m_NumFullBlocks.x; x > 0; --x)
                {
                    const Xyz pos = Xyz(m_NumFullBlocks, m_DepthForThisCe) - Xyz(x, 0, static_cast<unsigned>(z));

                    const unsigned inramAddr = input.WaitForFullWidthBlock();

                    op.ProcessPartialHeightBlock(0, numActiveOgs, inramAddr, pos, m_NumEdgePatches.y);

                    input.SignalFullWidthBlockFreed();
                }

                if (hasEdgeX)
                {
                    const Xyz pos = Xyz(m_NumFullBlocks, m_DepthForThisCe) - Xyz(0, 0, static_cast<unsigned>(z));

                    const unsigned inramAddr = input.WaitForPartialWidthBlock(m_NumEdgePatches.x);

                    op.ProcessPartialBlock(0, numActiveOgs, inramAddr, pos, m_NumEdgePatches);

                    input.SignalPartialWidthBlockFreed(m_NumEdgePatches.x);
                }

                op.NextRow(numActiveOgs, m_NumFullBlocks.y);
            }

            op.NextDepth(numActiveOgs);
        }

        return ncu_ple_interface::PleMsg::StripeDone{};
    }

private:
    PleState& m_PleState;
    const OperatorInfo m_OpInfo;
    const Xy m_NumFullBlocks;
    const Xy m_NumEdgePatches;
    const unsigned m_DepthForThisCe;
};

template <typename StripeLoop, typename OutputToInput = OutputToInputIdentity>
void __attribute__((noreturn, __always_inline__)) MainWithStripeLoop()
{
    PleState pleState;

    Main([&]() { pleState.WaitForEvent<Event::SETIRQ_EVENT>(); },
         [&]() {
             return StripeLoop{ pleState, GetOperatorInfo<OutputToInput>() }();
         });
}

}    // namespace
