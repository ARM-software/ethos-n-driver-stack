//
// Copyright © 2018-2020 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "PleState.hpp"
#include "Sizes.hpp"
#include "coprocessor_opcodes.h"
#include "hw.h"
#include "utils.h"
#include "xyz.h"

#include <ncu_ple_interface_def.h>

namespace
{
namespace input
{
template <typename BlockSize>
class InramRfLoader;

template <unsigned BlockWidth, unsigned BlockHeight>
class InramRfLoader<sizes::BlockSize<BlockWidth, BlockHeight>>
{
public:
    using BlockSize = sizes::BlockSize<BlockWidth, BlockHeight>;

    void LoadFullBlock(const unsigned ramId, const lsu::Address lsuAddr) const
    {
        using namespace static_loop;
        For<Range<0, TotalSize(BlockSize{}), 2>>::Invoke(lsu::LoadInramRfFn{}, ramId, lsuAddr);
    }

    void LoadPartialWidthBlock(const unsigned ramId, const lsu::Address lsuAddr, const unsigned width) const
    {
        using namespace static_loop;
        For<Range<0, BlockWidth>>::Invoke(LoadPartialWidthBlockFn{}, ramId, lsuAddr, width);
    }

    void LoadPartialHeightBlock(const unsigned ramId, const lsu::Address lsuAddr, const unsigned height) const
    {
        using namespace static_loop;
        For<Range<0, BlockHeight, 2>>::Invoke(LoadPartialHeightBlockFn{}, ramId, lsuAddr, height);
    }

    void LoadPartialBlock(const unsigned ramId, const lsu::Address lsuAddr, const Xy& size) const
    {
        using namespace static_loop;
        For<Range<0, TotalSize(BlockSize{}), 2>>::Invoke(VeRegrep16Fn{}, 0U);
        For<Range<0, BlockWidth>, Range<0, BlockHeight, 2>>::Invoke(LoadPartialBlockFn{}, ramId, lsuAddr, size);
    }

private:
    STATIC_LOOP_FN_WRAPPER(VeRegrep8Fn, ve_regrep_8);
    STATIC_LOOP_FN_WRAPPER(VeRegrep16Fn, ve_regrep_16);

    struct LoadPartialWidthBlockFn
    {
        template <unsigned Xinv>
        void operator()(const unsigned ramId, const lsu::Address lsuAddr, const unsigned width)
        {
            using namespace static_loop;

            constexpr unsigned X = BlockWidth - 1U - Xinv;

            if (X < width)
            {
                For<RangeN<2 * X, BlockHeight / 2, 2 * BlockSize::X>>::Invoke(lsu::LoadInramRfFn{}, ramId, lsuAddr);
            }
            else
            {
                For<RangeN<2 * X, BlockHeight / 2, 2 * BlockSize::X>>::Invoke(VeRegrep16Fn{}, 0U);
            }
        }
    };

    struct LoadPartialHeightBlockFn
    {
        template <unsigned Yinv>
        void operator()(const unsigned ramId, const lsu::Address lsuAddr, const unsigned height)
        {
            using namespace static_loop;

            constexpr unsigned Y = BlockHeight - 2U - Yinv;
            constexpr unsigned I = Y * BlockWidth;

            if ((Y + 1) < height)
            {
                For<RangeN<I, BlockWidth, 2>>::Invoke(lsu::LoadInramRfFn{}, ramId, lsuAddr);
            }
            else if (Y < height)
            {
                For<RangeN<I + 1, BlockWidth, 2>>::Invoke(VeRegrep8Fn{}, 0U);
                For<RangeN<I, BlockWidth, 2>>::Invoke(lsu::LoadHalfInramRfFn{}, ramId, lsuAddr);
            }
            else
            {
                For<RangeN<I, BlockWidth, 2>>::Invoke(VeRegrep16Fn{}, 0U);
            }
        }
    };

    struct LoadPartialBlockFn
    {
        template <unsigned X, unsigned Y>
        void operator()(const unsigned ramId, const lsu::Address lsuAddr, const Xy& size)
        {
            constexpr unsigned I = (Y * BlockWidth) + (2U * X);

            if ((X < size.x) && ((Y + 1) < size.y))
            {
                lsu::LoadInramRf<I>(ramId, lsuAddr);
            }
            else if ((X < size.x) && (Y < size.y))
            {
                lsu::LoadHalfInramRf<I>(ramId, lsuAddr);
            }
            else
            {
                // Do nothing
            }
        }
    };
};

template <unsigned BlocksWait, unsigned BlocksAdvance>
class MceInput
{
    static_assert(BlocksWait >= BlocksAdvance, "");

public:
    MceInput(PleState& pleState)
        : m_PleState(pleState)
    {}

    unsigned WaitForFullWidthBlock() const
    {
        m_PleState.WaitForBlocks(BlocksWait);
        return m_PleState.Advance(BlocksAdvance);
    }

    unsigned WaitForPartialWidthBlock(const unsigned width) const
    {
        const unsigned numMceBlocks = DivRoundUp(width, MceBlockSize::X);
        m_PleState.WaitForBlocks(numMceBlocks);
        return m_PleState.Advance(numMceBlocks);
    }

    void SignalFullWidthBlockFreed() const
    {
#pragma unroll_completely
        for (unsigned i = 0; i < BlocksAdvance; ++i)
        {
            SignalBufferFreed();
        }
    }

    void SignalPartialWidthBlockFreed(const unsigned width) const
    {
        const unsigned numMceBlocks = DivRoundUp(width, MceBlockSize::X);
        SignalBufferFreed(numMceBlocks);
    }

private:
    PleState& m_PleState;
};

template <>
class MceInput<1, 1>
{
public:
    MceInput(PleState& pleState)
        : m_PleState(pleState)
    {}

    unsigned WaitForFullWidthBlock() const
    {
        m_PleState.WaitForOneBlock();
        return m_PleState.Advance(1U);
    }

    unsigned WaitForPartialWidthBlock(unsigned) const
    {
        return WaitForFullWidthBlock();
    }

    void SignalFullWidthBlockFreed() const
    {
        SignalBufferFreed();
    }

    void SignalPartialWidthBlockFreed(unsigned) const
    {
        SignalFullWidthBlockFreed();
    }

private:
    PleState& m_PleState;
};
}    // namespace input
}    // namespace
