//
// Copyright © 2018-2019 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "Sizes.hpp"
#include "coprocessor_opcodes.h"
#include "hw.h"
#include "utils.h"
#include "xyz.h"

namespace
{
namespace output
{
template <unsigned Offset, unsigned Size>
struct OutramSpace
{
    static_assert(IsPow2(Size), "");
};

template <typename BlockSize, typename GroupSize>
class RfOutramStorer;

template <unsigned BlockWidth,
          unsigned BlockHeight,
          unsigned BlockDepth,
          unsigned GroupWidth,
          unsigned GroupHeight,
          unsigned GroupDepth>
class RfOutramStorer<sizes::BlockSize<BlockWidth, BlockHeight, BlockDepth>,
                     sizes::GroupSize<GroupWidth, GroupHeight, GroupDepth>>
{
public:
    static constexpr Xyz mk_BlockSize = { BlockWidth, BlockHeight, BlockDepth };
    static constexpr Xyz mk_GroupSize = { GroupWidth, GroupHeight, GroupDepth };

    static constexpr Xyz mk_BlockSizeInGroups = mk_BlockSize / mk_GroupSize;

    static constexpr unsigned mk_PatchesInBlock = TotalSize(mk_BlockSize);
    static constexpr unsigned mk_PatchesInGroup = TotalSize(mk_GroupSize);

    static constexpr unsigned mk_WordsInBlock = mk_PatchesInBlock * WORDS_PER_REGISTER;
    static constexpr unsigned mk_WordsInGroup = mk_PatchesInGroup * WORDS_PER_REGISTER;

    static_assert((mk_BlockSizeInGroups * mk_GroupSize) == mk_BlockSize, "");
    static_assert(IsPow2(mk_PatchesInBlock), "");

    void StoreFullBlock(const lsu::Address lsuAddr) const
    {
        using namespace static_loop;
        For<Range<0, mk_PatchesInBlock, 2>>::Invoke(lsu::StoreRfOutramFn{}, lsuAddr);
    }

    void StorePartialWidthBlock(lsu::Address lsuAddr, const unsigned width) const
    {
        using namespace static_loop;

        for (unsigned i = mk_BlockSizeInGroups.y; i > 0; --i)
        {
            unsigned j = 0;
            for (; j < width; j += mk_GroupSize.x)
            {
                For<Range<0, mk_PatchesInGroup, 2>>::Invoke(lsu::StoreRfOutramFn{}, lsuAddr);
                lsuAddr += mk_WordsInGroup;
            }
            // Don't advance ramAddr so valid data is packed
            lsuAddr.rfAddr += (mk_BlockSize.x - j) * (mk_WordsInGroup / mk_GroupSize.x);
        }
    }

    void StorePartialHeightBlock(lsu::Address lsuAddr, const unsigned height) const
    {
        using namespace static_loop;

        constexpr unsigned patchesInRow = TotalSize(mk_GroupSize * Xyz{ mk_BlockSizeInGroups.x, 1U, 1U });

        for (unsigned i = 0; i < height; i += mk_GroupSize.y)
        {
            For<Range<0, patchesInRow, 2>>::Invoke(lsu::StoreRfOutramFn{}, lsuAddr);
            lsuAddr += WORDS_PER_REGISTER * patchesInRow;
        }
    }

    void StorePartialBlock(lsu::Address lsuAddr, const Xy& size) const
    {
        using namespace static_loop;

        for (unsigned i = 0; i < size.y; i += mk_GroupSize.y)
        {
            unsigned j = 0;
            for (; j < size.x; j += mk_GroupSize.x)
            {
                For<Range<0, mk_PatchesInGroup, 2>>::Invoke(lsu::StoreRfOutramFn{}, lsuAddr);
                lsuAddr += mk_WordsInGroup;
            }
            // Don't advance ramAddr so valid data is packed
            lsuAddr.rfAddr += (mk_BlockSize.x - j) * (mk_WordsInGroup / mk_GroupSize.x);
        }
    }

private:
    STATIC_LOOP_FN_WRAPPER(VeRegrep16Fn, ve_regrep_16);
};

}    // namespace output
}    // namespace
