//
// Copyright © 2018-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "../include/ethosn_ple/BlockConstants.hpp"
#include "../include/ethosn_ple/Common.hpp"
#include "../include/ethosn_ple/MceStripeLoop.hpp"
#include "../include/ethosn_ple/SignedSupport.hpp"
#include "../include/ethosn_ple/Swizzle.hpp"

namespace
{
/// Implementation for Capacity a power of two
template <unsigned Capacity, bool = (Capacity & (Capacity - 1U)) == 0U>
class RingPosition
{
    static_assert((Capacity & (Capacity - 1U)) == 0U, "Capacity must be a power of 2");

public:
    explicit constexpr RingPosition(const unsigned pos = 0)
        : m_Pos(pos)
    {}

    constexpr unsigned Get() const
    {
        return m_Pos % Capacity;
    }

    constexpr RingPosition& operator+=(const unsigned rhs)
    {
        m_Pos += rhs;
        return *this;
    }

    constexpr RingPosition operator+(const unsigned rhs) const
    {
        return RingPosition(*this) += rhs;
    }

    constexpr RingPosition& operator-=(const unsigned rhs)
    {
        m_Pos -= rhs;
        return *this;
    }

    constexpr RingPosition operator-(const unsigned rhs) const
    {
        return RingPosition(*this) -= rhs;
    }

private:
    unsigned m_Pos;
};

/// Specialization for Capacity not a power of two
template <unsigned Capacity>
class RingPosition<Capacity, false>
{
public:
    explicit constexpr RingPosition(const unsigned pos = 0)
        : m_Pos(pos)
    {}

    constexpr unsigned Get() const
    {
        return m_Pos;
    }

    constexpr RingPosition& operator+=(const unsigned rhs)
    {
        m_Pos += rhs;
        if (m_Pos >= Capacity)
        {
            m_Pos -= Capacity;
        }
        return *this;
    }

    constexpr RingPosition operator+(const unsigned rhs) const
    {
        return RingPosition(*this) += rhs;
    }

    constexpr RingPosition& operator-=(const unsigned rhs)
    {
        if (m_Pos < rhs)
        {
            m_Pos += Capacity;
        }
        m_Pos -= rhs;
        return *this;
    }

    constexpr RingPosition operator-(const unsigned rhs) const
    {
        return RingPosition(*this) -= rhs;
    }

private:
    unsigned m_Pos;
};

template <unsigned Offset, unsigned Capacity>
class Stash
{
public:
    static constexpr unsigned GetOffset(const unsigned og)
    {
        return Offset + (og * Capacity);
    }

    void Reset()
    {
        m_Head = RingPosition<Capacity>{};
        m_Tail = RingPosition<Capacity>{};
    }

    RingPosition<Capacity> Front() const
    {
        return m_Head;
    }

    RingPosition<Capacity> Back() const
    {
        return m_Tail;
    }

    void PopFront(const unsigned n)
    {
        m_Head += n;
    }

    void PushBack(const unsigned n)
    {
        m_Tail += n;
    }

private:
    RingPosition<Capacity> m_Head;
    RingPosition<Capacity> m_Tail;
};

///////////////////////////////
//  Define useful constants  //
///////////////////////////////
using OutBlockSize = sizes::BlockSize<4, 1>;
using OutGroupSize = sizes::GroupSize<2, 1>;

constexpr Xy OUT_BLOCK_SIZE_IN_GROUPS = Xy(OutBlockSize{}) / Xy(OutGroupSize{});

constexpr unsigned OUT_BLOCK_SIZE_IN_WORDS = WORDS_PER_REGISTER * TotalSize(OutBlockSize{});
constexpr unsigned OUT_QUEUE_SIZE_IN_WORDS = 2U * OUT_BLOCK_SIZE_IN_WORDS;

constexpr unsigned WORDS_IN_STASH_GROUP = WORDS_PER_REGISTER * 2U;
constexpr unsigned WORDS_IN_STASH_BLOCK = OutBlockSize::X * WORDS_IN_STASH_GROUP;

constexpr unsigned REGS_ZEROS = 22U;

constexpr unsigned GetStashCapacity()
{
    constexpr unsigned WORDS_IN_HALF_KB   = WORDS_PER_REGISTER * ((1U << 9U) / ELEMENTS_PER_REGISTER);
    constexpr unsigned NUM_MCEIF_PER_LANE = NUM_MCEIF / NUM_PLE_LANES;
    constexpr unsigned OUTRAM_SIZE        = 3U * WORDS_IN_HALF_KB * (1U + NUM_MCEIF_PER_LANE);
    constexpr unsigned MIN_POW2_CAPACITY  = 4U * WORDS_IN_HALF_KB;

    unsigned capacity = (OUTRAM_SIZE - OUT_QUEUE_SIZE_IN_WORDS) / NUM_MCEIF_PER_LANE;

    capacity -= capacity % WORDS_IN_STASH_BLOCK;

    if (capacity > MIN_POW2_CAPACITY)
    {
        capacity = MIN_POW2_CAPACITY;
    }

    return capacity;
}

//////////////////////////////
//  Sanity-check constants  //
//////////////////////////////
static_assert((BlockSize::X == 8) && (BlockSize::Y == 2), "Only 8x2 blocks supported so far");
static_assert((OUT_QUEUE_SIZE_IN_WORDS & (OUT_QUEUE_SIZE_IN_WORDS - 1U)) == 0,
              "OUT_QUEUE_SIZE_IN_WORDS must be a power of 2");
static_assert((GetStashCapacity() % WORDS_IN_STASH_BLOCK) == 0, "");

/// Operator class for max pooling 3x3 stride (2,2)
class MaxPool_3x3_2_2
{
public:
    MaxPool_3x3_2_2(PleState& pleState, const OperatorInfo& opInfo)
        : m_OutputTraversal(opInfo.sizeInElements / Xyz(2, 2, 1))
        , m_Flags(opInfo.flags)
        , m_BlockPosEndX(DivRoundUp(opInfo.sizeInElements.x, BlockSize::X * ELEMENTS_PER_PATCH_1D) - 1U)
        , m_NumOutGroupsX(DivRoundUp(opInfo.sizeInElements.x / 2U, OutGroupSize::X * ELEMENTS_PER_PATCH_1D))
        // When the last patch in the X dimension only contains one column of valid elements, we only use it
        // to complete the pooling calculation of the patch before. It doesn't produce output data itself, i.e,
        // there are no valid pooling windows which centre is inside the patch.
        , m_SkipLastPatchX((opInfo.sizeInElements.x % ELEMENTS_PER_PATCH_1D) == 1U)
        , m_UdmaStorer(pleState.GetActiveEvents())
        , m_OutramAddr(0U)
        , m_OutDfcAddr(opInfo.output.dfcAddr)
    {
        constexpr SwzRegSel regSel0_1 = ToSwzRegSel({
            { 0, 0, 0, 0 },
            { 0, 0, 0, 0 },
            { 1, 1, 1, 1 },
            { 1, 1, 1, 1 },
        });

        constexpr SwzRegSel regSel2a = ToSwzRegSel({
            { 0, 0, 0, 0 },
            { 1, 1, 1, 1 },
            { 1, 1, 1, 1 },
            { 2, 2, 2, 2 },
        });

        constexpr SwzRegSel regSel2b = ToSwzRegSel({
            { 0, 0, 0, 0 },
            { 1, 1, 1, 1 },
            { 1, 1, 1, 1 },
            { 3, 3, 3, 3 },
        });

        constexpr HalfSwzSubRegSel subRegSel0 = ToHalfSwzSubRegSel({
            { 0, 4, 8, 12 },
            { 2, 6, 10, 14 },
        });

        constexpr HalfSwzSubRegSel subRegSel1 = ToHalfSwzSubRegSel({
            { 1, 5, 9, 13 },
            { 3, 7, 11, 15 },
        });

        constexpr HalfSwzSubRegSel subRegSel2 = ToHalfSwzSubRegSel({
            { 2, 6, 10, 14 },
            { 0, 4, 8, 12 },
        });

        // Init regs we'll use for zero-initialization
        ve_regrep_16<REGS_ZEROS>(0U);

        SetSwzRegSel<SWZ_COL_SELECT_TRANSPOSE_0>(regSel0_1);
        SetSwzRegSel<SWZ_COL_SELECT_TRANSPOSE_1>(regSel0_1);
        SetSwzRegSel<SWZ_COL_SELECT_TRANSPOSE_2A>(regSel2a);
        SetSwzRegSel<SWZ_COL_SELECT_TRANSPOSE_2B>(regSel2b);

        SetSwzSubRegSel<SWZ_COL_SELECT_TRANSPOSE_0>(subRegSel0, subRegSel0);
        SetSwzSubRegSel<SWZ_COL_SELECT_TRANSPOSE_1>(subRegSel1, subRegSel1);
        SetSwzSubRegSel<SWZ_COL_SELECT_TRANSPOSE_2A>(subRegSel2, subRegSel2);
        SetSwzSubRegSel<SWZ_COL_SELECT_TRANSPOSE_2B>(subRegSel2, subRegSel2);

        if (opInfo.flags[Flags::TOP])
        {
            GetStash().Reset();
            GetLayerHeightInElements() = 0;
            GetOutDfcAddrSave()        = m_OutDfcAddr;
        }
        else
        {
            std::swap(m_OutDfcAddr, GetOutDfcAddrSave());
        }

        m_OutDfcAddrZ = m_OutDfcAddr;

        if (m_Flags[Flags::RIGHT])
        {
            GetLayerHeightInElements() += opInfo.sizeInElements.y;
        }
    }

    ~MaxPool_3x3_2_2()
    {
        GetOutDfcAddrSave() = m_OutDfcAddr;
        m_UdmaStorer.WaitForUdma();
    }

    void ProcessFullBlock(const unsigned firstOg, const unsigned lastOg, const unsigned inramAddr, const Xyz& pos)
    {
        ProcessBlock(firstOg, lastOg, inramAddr, pos, Xy(BlockSize{}), false);
    }

    void ProcessPartialWidthBlock(
        const unsigned firstOg, const unsigned lastOg, const unsigned inramAddr, const Xyz& pos, unsigned width)
    {
        ProcessBlock(firstOg, lastOg, inramAddr, pos, Xy(width, BlockSize::Y), true);
    }

    void ProcessPartialHeightBlock(
        const unsigned firstOg, const unsigned lastOg, const unsigned inramAddr, const Xyz& pos, unsigned height)
    {
        ProcessBlock(firstOg, lastOg, inramAddr, pos, Xy(BlockSize::X, height), false);
    }

    void ProcessPartialBlock(
        const unsigned firstOg, const unsigned lastOg, const unsigned inramAddr, const Xyz& pos, const Xy& size)
    {
        ProcessBlock(firstOg, lastOg, inramAddr, pos, size, true);
    }

    void NextRow(const unsigned numActiveOgs, const unsigned posY)
    {
        if (posY == 0U)
        {
            m_OutDfcAddr = GetOutDfcAddrSave();
        }
        else
        {
            m_OutDfcAddr += m_OutputTraversal.Advance(Xyz(0, posY - 1U, 0), Xyz(0, OUT_BLOCK_SIZE_IN_GROUPS.y, 0)) -
                            (WORDS_PER_REGISTER * m_OutputTraversal.GetXyStride().y);
        }
    }

    void NextDepth(const unsigned numActiveOgs)
    {
        if (m_Flags[Flags::BOTTOM])
        {
            // When the last row of blocks in the Y dimension only contains one row of valid elements, we only
            // use it to complete the pooling calculation of the row above. It doesn't produce output data itself,
            // i.e, there are no valid pooling windows which centre is inside the row.
            if ((GetLayerHeightInElements() % (2U * ELEMENTS_PER_GROUP_1D)) != 1U)
            {
                if ((GetLayerHeightInElements() % ELEMENTS_PER_GROUP_1D) != 1U)
                {
                    ProcessLastRow(numActiveOgs);
                }
                else
                {
                    ZeroInitLastRow(numActiveOgs);
                }
            }
            {
                const unsigned adv = m_OutputTraversal.Advance(Xyz{}, Xyz(0, 0, numActiveOgs));
                m_OutDfcAddrZ += adv;
                GetOutDfcAddrSave() += adv;
                m_OutDfcAddr = m_OutDfcAddrZ;
            }
            GetStash().Reset();
        }
    }

private:
    enum : unsigned
    {
        SWZ_COL_SELECT_TRANSPOSE_0,
        SWZ_COL_SELECT_TRANSPOSE_1,
        SWZ_COL_SELECT_TRANSPOSE_2A,
        SWZ_COL_SELECT_TRANSPOSE_2B,
    };

    using Stash = Stash<OUT_QUEUE_SIZE_IN_WORDS, GetStashCapacity()>;

    struct State
    {
        Stash stash;
        uint16_t outDfcAddrSave;
        unsigned layerHeightInElements;
    };

    static State& GetState()
    {
        static std::aligned_storage_t<sizeof(State), alignof(State)> state;
        return reinterpret_cast<State&>(state);
    }

    static Stash& GetStash()
    {
        return GetState().stash;
    }

    static uint16_t& GetOutDfcAddrSave()
    {
        return GetState().outDfcAddrSave;
    }

    static unsigned& GetLayerHeightInElements()
    {
        return GetState().layerHeightInElements;
    }

    void Pool1D() const
    {
        static_assert(VE_TIMING::SWZ_8::WRITE_BACK > (1 + VE_TIMING::SWZ_8::OP_READ), "");
        static_assert(VE_TIMING::UMAX_8::WRITE_BACK > (1 + VE_TIMING::UMAX_8::OP_READ), "");

        // 6 input patches are expected in registers 0-5, corresponding to XY coordinates in the following order
        //
        //        cr0     cr1     cr2     cr3     cr4     cr5
        //     +-------+-------+-------+-------+-------+-------+
        //     | (0,0) | (1,0) | (0,1) | (1,1) | (2,0) | (2,1) |
        //     +-------+-------+-------+-------+-------+-------+
        //
        // In spatial representation:
        //
        //      x →
        //    y +-----+-----+-----+
        //    ↓ | cr0 | cr1 | cr4 |
        //      +-----+-----+-----+
        //      | cr2 | cr3 | cr5 |
        //      +-----+-----+-----+
        //

        ve_swz_8<6, 0, 0, SWZ_COL_SELECT_TRANSPOSE_0>();
        ve_swz_8<7, 0, 0, SWZ_COL_SELECT_TRANSPOSE_1>();
        ve_swz_8<8, 0, 4, SWZ_COL_SELECT_TRANSPOSE_2A>();

        ve_swz_8<9, 2, 2, SWZ_COL_SELECT_TRANSPOSE_0>();
        ve_swz_8<10, 2, 2, SWZ_COL_SELECT_TRANSPOSE_1>();
        ve_swz_8<11, 2, 4, SWZ_COL_SELECT_TRANSPOSE_2B>();

        Max8<6, 6, 7>();
        Max8<9, 9, 10>();

        nop<RwHazardDelay<MAX8_DELAY_TYPE, MAX8_DELAY_TYPE, 2>()>();

        Max8<0, 6, 8>();
        Max8<1, 9, 11>();
    }

    void HorizontalPoolGroup(const unsigned outramAddr) const
    {
        Pool1D();
        // Stash result
        nop<RwHazardDelay<VE_TIMING::UMAX_8, VE_TIMING::STORE_RF_OUTRAM>()>();
        lsu::StoreRfOutram<0>({ outramAddr, 0U });
    }

    void LoadInputPatches(const unsigned og, const unsigned inramAddr, const Xy& size) const
    {
        ve_regrep_16<0>(0U);
        ve_regrep_16<2>(0U);
        ve_regrep_16<4>(0U);

        const lsu::Address lsuAddr = { inramAddr, 0U };

        if (size.y > 1)
        {
            lsu::LoadInramRf<0>(og, lsuAddr);
            if (size.x > 1)
            {
                lsu::LoadInramRf<2>(og, lsuAddr);
            }
            if (size.x > 2)
            {
                lsu::LoadInramRf<4>(og, lsuAddr);
            }
        }
        else
        {
            lsu::LoadHalfInramRf<0>(og, lsuAddr);
            if (size.x > 1)
            {
                lsu::LoadHalfInramRf<2>(og, lsuAddr);
            }
            if (size.x > 2)
            {
                lsu::LoadHalfInramRf<4>(og, lsuAddr);
            }
        }

        // Swap regs 1 <-> 2
        static_assert(VE_TIMING::MOV_8::WRITE_BACK > (1 + VE_TIMING::MOV_8::OP_READ), "");
        ve_mov_8<1, 2>();
        ve_mov_8<2, 1>();
    }

    void HorizontalPoolBlock(const unsigned og, unsigned inramAddr, const Xy& size, const bool right) const
    {
        auto stashPos = GetStash().Back();

        // A group of 3x2 input patches are needed to produce a group of 1D-pooled 1x2 patches.
        // The calculation of coreNumGroups (considering that the last patch was already removed
        // from size according to m_SkipLastPatch if needed), ensures that all 3 in the width
        // dimension are available for those many groups.
        const unsigned coreNumGroups = (size.x - 1U) / 2U;

        for (unsigned i = coreNumGroups; i > 0; --i)
        {
            LoadInputPatches(og, inramAddr, Xy(3U, size.y));
            {
                const unsigned outramAddr = GetStash().GetOffset(og / NUM_PLE_LANES) + stashPos.Get();
                HorizontalPoolGroup(outramAddr);
            }
            inramAddr += 4U * WORDS_PER_REGISTER;
            stashPos += WORDS_IN_STASH_GROUP;
        }
        {
            const unsigned sizeX = size.x - (2U * coreNumGroups) + static_cast<unsigned>(!right || m_SkipLastPatchX);
            LoadInputPatches(og, inramAddr, Xy(sizeX, size.y));
        }
        {
            const unsigned outramAddr = GetStash().GetOffset(og / NUM_PLE_LANES) + stashPos.Get();
            HorizontalPoolGroup(outramAddr);
        }
        if ((coreNumGroups % 2U) == 0U)
        {
            // Stash zeros to complete group
            stashPos += WORDS_IN_STASH_GROUP;
            const unsigned outramAddr = GetStash().GetOffset(og / NUM_PLE_LANES) + stashPos.Get();
            lsu::StoreRfOutram<0>({ outramAddr, WORDS_PER_REGISTER * REGS_ZEROS });
        }
    }

    void VerticalPoolBlock(const unsigned og, const unsigned numGroups)
    {
        auto row0Addr       = GetStash().Front();
        auto row1Addr       = GetStash().Back() - WORDS_IN_STASH_BLOCK;
        unsigned outramAddr = m_OutramAddr;

        for (unsigned i = numGroups; i > 0; --i)
        {
            const lsu::Address lsuAddr0 = {
                GetStash().GetOffset(og / NUM_PLE_LANES) + row0Addr.Get(),
                0U,
            };
            const lsu::Address lsuAddr1a = {
                GetStash().GetOffset(og / NUM_PLE_LANES) + row1Addr.Get(),
                WORDS_PER_REGISTER * 4U,
            };
            const lsu::Address lsuAddr1b = {
                GetStash().GetOffset(og / NUM_PLE_LANES) + (row1Addr + (WORDS_PER_REGISTER * 2U)).Get(),
                WORDS_PER_REGISTER * 5U,
            };

            lsu::LoadOutramRf<0>(lsuAddr0);
            lsu::LoadOutramRf<2>(lsuAddr0);
            lsu::LoadHalfOutramRf<0>(lsuAddr1a);
            lsu::LoadHalfOutramRf<0>(lsuAddr1b);

            nop<RwHazardDelay<VE_TIMING::LOAD_HALF_OUTRAM_RF, VE_TIMING::SWZ_8>()>();

            Pool1D();

            nop<RwHazardDelay<VE_TIMING::UMAX_8, VE_TIMING::STORE_RF_OUTRAM>()>();

            lsu::StoreRfOutram<0>({ outramAddr, 0U });

            row0Addr += WORDS_PER_REGISTER * 4U;
            row1Addr += WORDS_PER_REGISTER * 4U;
            outramAddr += WORDS_PER_REGISTER * 2U;
        }
    }

    unsigned GetOutQueueSlot()
    {
        const unsigned slot = m_OutramAddr;
        m_OutramAddr        = (slot + OUT_BLOCK_SIZE_IN_WORDS) % OUT_QUEUE_SIZE_IN_WORDS;
        return slot;
    }

    void UdmaBlock(const unsigned og, const unsigned outramAddr, const Xy& blockSizeInGroups)
    {
        udma::Address udmaAddr = { m_OutDfcAddr, outramAddr };

        auto& udmaAddrAsU32 = reinterpret_cast<unsigned&>(udmaAddr);
        udmaAddrAsU32 += m_OutputTraversal.Advance(Xyz{}, Xyz(0, 0, og));

        m_UdmaStorer.WaitForUdma();
        m_OutputTraversal.SetUdmaStoreParams(blockSizeInGroups);
        m_UdmaStorer.Store(og % NUM_SRAMS, udmaAddr);
    }

    void ProcessBlock(const unsigned firstOg,
                      const unsigned lastOg,
                      const unsigned inramAddr,
                      const Xyz& pos,
                      Xy size,
                      const bool right)
    {
        if (right && m_SkipLastPatchX)
        {
            --size.x;

            if (size.x == 0U)
            {
                return;
            }
        }

        for (unsigned og = firstOg; og < lastOg; og += NUM_PLE_LANES)
        {
            HorizontalPoolBlock(og, inramAddr, size, right);
        }
        GetStash().PushBack(WORDS_IN_STASH_BLOCK);

        const bool top = m_Flags[Flags::TOP] && (pos.y == 0);

        if (!top)
        {
            const Xy blockSizeInGroups = {
                DivRoundUp(size.x, 2U * OutGroupSize::X),
                OutGroupSize::Y,
            };

            for (unsigned og = firstOg; og < lastOg; og += NUM_PLE_LANES)
            {
                VerticalPoolBlock(og, blockSizeInGroups.x);
                UdmaBlock(og, GetOutQueueSlot(), blockSizeInGroups);
            }
            GetStash().PopFront(WORDS_IN_STASH_BLOCK);

            m_OutDfcAddr += m_OutputTraversal.Advance(pos, Xyz(blockSizeInGroups.x));
        }
    }

    void ProcessLastRow(const unsigned numActiveOgs)
    {
        ve_regrep_16<4>(0U);

        const uint16_t outDfcAddrSave = m_OutDfcAddr;

        for (unsigned og = 0; og < numActiveOgs; og += NUM_PLE_LANES)
        {
            auto hpoolAddr = GetStash().Front();
            m_OutDfcAddr   = outDfcAddrSave;

#pragma unroll 1
            for (unsigned i = m_NumOutGroupsX; i > 0; --i)
            {
                {
                    const lsu::Address lsuHpoolAddr = { GetStash().GetOffset(og / NUM_PLE_LANES) + hpoolAddr.Get(),
                                                        0U };
                    lsu::LoadOutramRf<0>(lsuHpoolAddr);
                    lsu::LoadOutramRf<2>(lsuHpoolAddr);
                }

                nop<RwHazardDelay<VE_TIMING::LOAD_OUTRAM_RF, VE_TIMING::SWZ_8>()>();

                Pool1D();

                hpoolAddr += WORDS_PER_REGISTER * 4U;

                nop<RwHazardDelay<VE_TIMING::UMAX_8, VE_TIMING::STORE_RF_OUTRAM>()>();

                lsu::StoreRfOutram<0>({ m_OutramAddr, 0U });

                UdmaBlock(og, GetOutQueueSlot(), Xy(1, 1));

                m_OutDfcAddr += m_OutputTraversal.Advance(Xyz(m_NumOutGroupsX - i), Xyz(1));
            }
        }
    }

    void ZeroInitLastRow(const unsigned numActiveOgs)
    {
        const uint16_t outDfcAddrSave = m_OutDfcAddr;
        const unsigned outramAddr     = GetOutQueueSlot();

        lsu::StoreRfOutram<0>({ outramAddr, WORDS_PER_REGISTER * REGS_ZEROS });

        for (unsigned og = 0; og < numActiveOgs; og += NUM_PLE_LANES)
        {
            m_OutDfcAddr = outDfcAddrSave;

#pragma unroll 1
            for (unsigned i = m_NumOutGroupsX; i > 0; --i)
            {
                UdmaBlock(og, outramAddr, Xy(1, 1));
                m_OutDfcAddr += m_OutputTraversal.Advance(Xyz(m_NumOutGroupsX - i), Xyz(1));
            }
        }
    }

    const dfcsram::Traversal<OutGroupSize> m_OutputTraversal;

    const EnumBitset<Flags> m_Flags;

    const unsigned m_BlockPosEndX;
    const unsigned m_NumOutGroupsX;

    const bool m_SkipLastPatchX;

    udma::UdmaStorer m_UdmaStorer;

    uint16_t m_OutramAddr;
    uint16_t m_OutDfcAddr;
    uint16_t m_OutDfcAddrZ;
};

using MaxPool_3x3_2_2_StripeLoop = MceStripeLoop<MaxPool_3x3_2_2, k_BlockMultiplier + 1, k_BlockMultiplier>;
}    // namespace
