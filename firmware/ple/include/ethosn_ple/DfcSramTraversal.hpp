//
// Copyright © 2018-2020,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "Sizes.hpp"
#include "hw.h"
#include "udma.h"
#include "utils.h"
#include "xyz.h"

namespace
{
namespace dfcsram
{

constexpr Xy GetNhwcbGroupStride(const Xyz& sizeInElements)
{
    Xy groupStride;

    //  Example of 32x32x1 stripe
    //      x →    0                1                2                3
    //    y G0=====+======+  G1=====+======+  G2=====+======+  G3=====+======+
    //    ↓ ‖  p0  |  p2  ‖  ‖  p4  |  p6  ‖  ‖  p8  |  p10 ‖  ‖  p12 |  p14 ‖
    //    0 +------+------+  +------+------+  +------+------+  +------+------+
    //      ‖  p1  |  p3  ‖  ‖  p5  |  p7  ‖  ‖  p9  |  p11 ‖  ‖  p13 |  p15 ‖
    //      +======+======+  +======+======+  +======+======+  +======+======+
    //
    //      G4=====+======+  G5=====+======+  G6=====+======+  G7=====+======+
    //      ‖  p16 |  p18 ‖  ‖  p20 |  p22 ‖  ‖  p24 |  p26 ‖  ‖  p28 |  p30 ‖
    //    1 +------+------+  +------+------+  +------+------+  +------+------+
    //      ‖  p17 |  p19 ‖  ‖  p21 |  p23 ‖  ‖  p25 |  p27 ‖  ‖  p29 |  p31 ‖
    //      +======+======+  +======+======+  +======+======+  +======+======+
    //
    //      G8=====+======+  G9=====+======+  G10====+======+  G11====+======+
    //      ‖  p32 |  p34 ‖  ‖  p36 |  p38 ‖  ‖  p40 |  p42 ‖  ‖  p44 |  p46 ‖
    //    2 +------+------+  +------+------+  +------+------+  +------+------+
    //      ‖  p33 |  p35 ‖  ‖  p37 |  p39 ‖  ‖  p41 |  p43 ‖  ‖  p45 |  p47 ‖
    //      +======+======+  +======+======+  +======+======+  +======+======+
    //
    //      G12====+======+  G13====+======+  G14====+======+  G15====+======+
    //      ‖  p48 |  p50 ‖  ‖  p52 |  p54 ‖  ‖  p56 |  p58 ‖  ‖  p60 |  p62 ‖
    //    3 +------+------+  +------+------+  +------+------+  +------+------+
    //      ‖  p49 |  p51 ‖  ‖  p53 |  p55 ‖  ‖  p57 |  p59 ‖  ‖  p61 |  p63 ‖
    //      +======+======+  +======+======+  +======+======+  +======+======+
    //
    // Normal processing order is row major, this means:
    //  - group stride X is distance between p0 and p4 in number of patches
    //  - group stride Y is distance between p0 and p16 in number of patches
    //
    // Transposed processing order is column major, this means:
    //  - group stride X is distance between p0 and p16 in number of patches
    //  - group stride Y is distance between p0 and p4 in number of patches

    groupStride.x = PATCHES_PER_GROUP * DivRoundUp(sizeInElements.z, TOTAL_NUM_SRAMS);
    groupStride.y = groupStride.x * DivRoundUp(sizeInElements.x, ELEMENTS_PER_GROUP_1D);

    return groupStride;
}

template <unsigned UdmaGroupSize, typename Derived>
class TraversalBase
{
public:
    constexpr TraversalBase(const Xyz& sizeInElements)
        : m_NhwcbGroupStride(GetNhwcbGroupStride(sizeInElements))
    {}

    void SetUdmaStoreParams(const Xy& sizeInGroups) const
    {
        udma::SetStoreParams<UdmaGroupSize>(GetUdmaParams(sizeInGroups));
    }

    void SetUdmaLoadParams(const Xy& sizeInGroups) const
    {
        udma::SetLoadParams<UdmaGroupSize>(GetUdmaParams(sizeInGroups));
    }

    const Xy& GetXyStride() const
    {
        return m_NhwcbGroupStride;
    }

    constexpr unsigned Advance(const Xyz& advInGroups) const
    {
        return static_cast<const Derived*>(this)->Advance(Xyz{}, advInGroups);
    }

private:
    constexpr udma::Params GetUdmaParams(const Xy& sizeInGroups) const
    {
        return static_cast<const Derived*>(this)->GetUdmaParams(sizeInGroups);
    }

    const Xy m_NhwcbGroupStride;
};

template <typename GroupSize>
class Traversal;

template <unsigned GroupDepth>
class Traversal<sizes::GroupSize<2, 2, GroupDepth>>
    : public TraversalBase<4 * GroupDepth, Traversal<sizes::GroupSize<2, 2, GroupDepth>>>
{
public:
    using GroupSize = sizes::GroupSize<2, 2, GroupDepth>;
    using Base      = TraversalBase<4 * GroupDepth, Traversal<GroupSize>>;

    using Base::Advance;
    using Base::Base;
    using Base::GetXyStride;

    constexpr unsigned Advance(const Xyz&, const Xyz& advInGroups) const
    {
        const Xyz advInNhwcbGroups = advInGroups / Xyz(1, 1, NUM_SRAMS);

        return Dot(Xyz::Dup(WORDS_PER_REGISTER), advInNhwcbGroups * Xyz(GetXyStride(), TotalSize(GroupSize{})));
    }

    constexpr udma::Params GetUdmaParams(const Xy& sizeInGroups) const
    {
        // The uDMA increments the write address between each write, as follows:
        // - Every write, a trivial stride of 1 is added to the address
        // - Every time GroupTransferSize is reset, ColumnGroupStride is added to the address
        //   (in addition to the trivial stride of 1)
        // - Every time ColumnGroupCount is reset, RowGroupStride is added to the address
        //   (in addition to the trivial stride of 1 and ColumnGroupStride)

        udma::Params params;

        params.colGrpCountMinusOne = sizeInGroups.x - 1U;
        params.rowGrpCountMinusOne = sizeInGroups.y - 1U;

        params.colGrpStride = GetXyStride().x - TotalSize(GroupSize{});
        params.rowGrpStride = GetXyStride().y - (GetXyStride().x * sizeInGroups.x);

        return params;
    }
};

template <>
class Traversal<sizes::GroupSize<1, 2, 1>> : public TraversalBase<1, Traversal<sizes::GroupSize<1, 2, 1>>>
{
public:
    using GroupSize = sizes::GroupSize<1, 2, 1>;
    using Base      = TraversalBase<1, Traversal<GroupSize>>;

    using Base::Advance;
    using Base::Base;
    using Base::GetXyStride;

    constexpr unsigned Advance(const Xyz& pos, Xyz advInGroups) const
    {
        const unsigned xMod2 = pos.x % 2;

        advInGroups += Xyz(xMod2, 0, 0);

        const Xyz advInNhwcbGroups = advInGroups / Xyz(2, 1, NUM_SRAMS);

        const unsigned advNhwcb =
            Dot(Xyz::Dup(WORDS_PER_REGISTER), advInNhwcbGroups * Xyz(GetXyStride(), PATCHES_PER_GROUP));

        const unsigned adv = (WORDS_PER_REGISTER * 2 * ((advInGroups.x % 2) - xMod2)) + advNhwcb;

        return adv;
    }

    constexpr udma::Params GetUdmaParams(const Xy& sizeInGroups) const
    {
        // The uDMA increments the write address between each write, as follows:
        // - Every write, a trivial stride of 1 is added to the address
        // - Every time GroupTransferSize is reset, ColumnGroupStride is added to the address
        //   (in addition to the trivial stride of 1)
        // - Every time ColumnGroupCount is reset, RowGroupStride is added to the address
        //   (in addition to the trivial stride of 1 and ColumnGroupStride)

        udma::Params params{};

        params.colGrpCountMinusOne = 1;
        params.rowGrpCountMinusOne = sizeInGroups.y - 1;

        params.colGrpStride = 0;
        params.rowGrpStride = GetXyStride().y - 2;

        return params;
    }
};

template <>
class Traversal<sizes::GroupSize<2, 1, 1>> : public TraversalBase<1, Traversal<sizes::GroupSize<2, 1, 1>>>
{
public:
    using GroupSize = sizes::GroupSize<2, 1, 1>;
    using Base      = TraversalBase<1, Traversal<GroupSize>>;

    using Base::Advance;
    using Base::Base;
    using Base::GetXyStride;

    constexpr unsigned Advance(const Xyz& pos, Xyz advInGroups) const
    {
        const unsigned yMod2 = pos.y % 2;

        advInGroups += Xyz(0, yMod2, 0);

        const Xyz advInNhwcbGroups = advInGroups / Xyz(1, 2, NUM_SRAMS);

        const unsigned advNhwcb =
            Dot(Xyz::Dup(WORDS_PER_REGISTER), advInNhwcbGroups * Xyz(GetXyStride(), PATCHES_PER_GROUP));

        const unsigned adv = (WORDS_PER_REGISTER * ((advInGroups.y % 2) - yMod2)) + advNhwcb;

        return adv;
    }

    constexpr udma::Params GetUdmaParams(const Xy& sizeInGroups) const
    {
        // The uDMA increments the write address between each write, as follows:
        // - Every write, a trivial stride of 1 is added to the address
        // - Every time GroupTransferSize is reset, ColumnGroupStride is added to the address
        //   (in addition to the trivial stride of 1)
        // - Every time ColumnGroupCount is reset, RowGroupStride is added to the address
        //   (in addition to the trivial stride of 1 and ColumnGroupStride)

        udma::Params params{};

        params.colGrpCountMinusOne = 1;
        params.rowGrpCountMinusOne = sizeInGroups.x - 1;

        params.colGrpStride = 1;
        params.rowGrpStride = GetXyStride().x - 4;

        return params;
    }
};

}    // namespace dfcsram
}    // namespace
