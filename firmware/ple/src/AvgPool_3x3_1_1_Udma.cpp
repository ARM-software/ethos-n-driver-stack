//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#include "../include/ethosn_ple/Common.hpp"
#include "../include/ethosn_ple/SignedSupport.hpp"
#include "../include/ethosn_ple/Swizzle.hpp"

namespace
{
struct DivInfo
{
    uint8_t off;

    union
    {
        uint16_t mul;

        struct
        {
            uint8_t mulLow;
            uint8_t mulHigh;
        };
    };
};

template <typename SwzRegSelT, unsigned N>
struct SwzTable
{
    using Type = SwzRegSelT;

    static constexpr unsigned numEntries = N;

    Type data[numEntries];

    constexpr const auto& operator[](const unsigned i) const
    {
        return data[i];
    }

    constexpr auto& operator[](const unsigned i)
    {
        return data[i];
    }
};

using DivSwzTable  = SwzTable<SwzSubRegSel, 1U + ELEMENTS_PER_PATCH_1D + 1U>;
using ZextSwzTable = SwzTable<SwzRegSel, 1U + ELEMENTS_PER_PATCH_1D + 1U>;

template <unsigned Divisor>
constexpr DivInfo GetDivInfo()
{
    constexpr unsigned offset     = Divisor / 2U;
    constexpr unsigned multiplier = ((1U << 16) / Divisor) + 1U;

    constexpr DivInfo divInfo = { offset, multiplier };

    static_assert(divInfo.off == offset, "offset overflow");
    static_assert(divInfo.mul == multiplier, "multiplier overflow");

    return divInfo;
}

constexpr SwzRegSel RegSelDown(const uint8_t srcReg0, const uint8_t srcReg1)
{
    return ToSwzRegSel({
        { srcReg0, srcReg0, srcReg0, srcReg0 },
        { srcReg1, srcReg1, srcReg1, srcReg1 },
        { srcReg1, srcReg1, srcReg1, srcReg1 },
        { srcReg1, srcReg1, srcReg1, srcReg1 },
    });
}

constexpr SwzRegSel RegSelUp(const uint8_t srcReg0, const uint8_t srcReg1)
{
    return ToSwzRegSel({
        { srcReg1, srcReg1, srcReg1, srcReg1 },
        { srcReg1, srcReg1, srcReg1, srcReg1 },
        { srcReg1, srcReg1, srcReg1, srcReg1 },
        { srcReg0, srcReg0, srcReg0, srcReg0 },
    });
}

template <typename RegSelT>
constexpr RegSelT
    GetPartialPatchSwzY(const unsigned size, const unsigned inValue, const unsigned edgeValue, const unsigned outValue)
{
    if (size == 0U)
    {
        return RegSelT::Dup(outValue);
    }

    const unsigned edge = size - 1U;

    uint8_t regSel[ELEMENTS_PER_PATCH_1D][ELEMENTS_PER_PATCH_1D] = {};

    for (unsigned i = 0U; i < std::min(edge, ELEMENTS_PER_PATCH_1D); ++i)
    {
        for (unsigned j = 0; j < ELEMENTS_PER_PATCH_1D; ++j)
        {
            regSel[i][j] = inValue;
        }
    }

    if (edge < ELEMENTS_PER_PATCH_1D)
    {
        for (unsigned j = 0; j < ELEMENTS_PER_PATCH_1D; ++j)
        {
            regSel[edge][j] = edgeValue;
        }

        for (unsigned i = edge + 1U; i < ELEMENTS_PER_PATCH_1D; ++i)
        {
            for (unsigned j = 0; j < ELEMENTS_PER_PATCH_1D; ++j)
            {
                regSel[i][j] = outValue;
            }
        }
    }

    return ToSwzRegSelT<RegSelT>(regSel);
}

template <typename SwzTable, bool Tr>
constexpr SwzTable GetSwzTable(const unsigned inValue, const unsigned edgeValue, const unsigned outValue)
{
    using SwzRegSelT = typename SwzTable::Type;

    SwzTable tbl{};

    for (unsigned i = 0U; i < tbl.numEntries; ++i)
    {
        tbl[i] = GetPartialPatchSwzY<SwzRegSelT>(i, inValue, edgeValue, outValue);

        if (Tr)
        {
            tbl[i] = Transpose(tbl[i]);
        }
    }

    return tbl;
}

constexpr DivSwzTable GetDivSwzTableX()
{
    return GetSwzTable<DivSwzTable, true>(0, 1, 4);
}

constexpr DivSwzTable GetDivSwzTableY()
{
    return GetSwzTable<DivSwzTable, false>(0, 2, 4);
}

constexpr ZextSwzTable GetZextSwzTableX()
{
    return GetSwzTable<ZextSwzTable, true>(0, 0, 3);
}

constexpr ZextSwzTable GetZextSwzTableY()
{
    return GetSwzTable<ZextSwzTable, false>(0, 0, 3);
}

template <unsigned I>
constexpr unsigned GetPartialSize(const unsigned size)
{
    constexpr unsigned offset  = I * ELEMENTS_PER_PATCH_1D;
    const unsigned partialSize = ((size - 1U) % ELEMENTS_PER_GROUP_1D) + 1U;
    return std::min(std::max(partialSize, offset) - offset, DivSwzTable::numEntries - 1U);
}

class AvgPool_3x3_1_1_Udma
{
    using GroupSize = sizes::GroupSize<PATCHES_PER_GROUP_1D, PATCHES_PER_GROUP_1D>;

    static constexpr unsigned OUT_QUEUE_SIZE = WORDS_PER_REGISTER * 2U * TotalSize(GroupSize{});

    static constexpr unsigned REG_DIV_OFF = 20;
    static constexpr unsigned REG_DIV_MUL = 22;

public:
    AvgPool_3x3_1_1_Udma(EnumBitset<Event>& activeEvents, const OperatorInfo& opInfo)
        : m_SizeInGroups{
            DivRoundUp(Xy(opInfo.sizeInElements), Xy::Dup(ELEMENTS_PER_GROUP_1D)),
            DivRoundUp(std::max(opInfo.sizeInElements.z, g_CeId) - g_CeId, NUM_CES),
        }
        , m_DfcTraversal(opInfo.sizeInElements)
        , m_InDfcAddrBase(opInfo.inputs[0].dfcAddr)
        , m_OutDfcAddrBase(opInfo.output.dfcAddr)
        , m_DivSwzX{
            {
                {
                    SwzSubRegSel::Dup(0U),
                    SwzSubRegSel::Dup(0U),
                },
                {
                    ms_DivSwzTableX[GetPartialSize<0>(opInfo.sizeInElements.x)],
                    ms_DivSwzTableX[GetPartialSize<1>(opInfo.sizeInElements.x)],
                },
            },
            {
                {
                    ms_DivSwzLeft,
                    SwzSubRegSel::Dup(0U),
                },
                {
                    ms_DivSwzLeft | ms_DivSwzTableX[GetPartialSize<0>(opInfo.sizeInElements.x)],
                    ms_DivSwzTableX[GetPartialSize<1>(opInfo.sizeInElements.x)],
                },
            },
        }
        , m_DivSwzY{
            {
                {
                    SwzSubRegSel::Dup(0U),
                    SwzSubRegSel::Dup(0U),
                },
                {
                    ms_DivSwzTableY[GetPartialSize<0>(opInfo.sizeInElements.y)],
                    ms_DivSwzTableY[GetPartialSize<1>(opInfo.sizeInElements.y)],
                },
            },
            {
                {
                    ms_DivSwzTop,
                    SwzSubRegSel::Dup(0U),
                },
                {
                    ms_DivSwzTop | ms_DivSwzTableY[GetPartialSize<0>(opInfo.sizeInElements.y)],
                    ms_DivSwzTableY[GetPartialSize<1>(opInfo.sizeInElements.y)],
                },
            },
        }
        , m_ZextSwz{
            {
                {
                    {
                        SwzRegSel::Dup(0U),
                        SwzRegSel::Dup(0U),
                    },
                    {
                        SwzRegSel::Dup(1U),
                        SwzRegSel::Dup(1U),
                    },
                },
                {
                    {
                        ms_ZextSwzTableX[GetPartialSize<0>(opInfo.sizeInElements.x)],
                        ms_ZextSwzTableX[GetPartialSize<1>(opInfo.sizeInElements.x)],
                    },
                    {
                        SwzRegSel::Dup(1U) | ms_ZextSwzTableX[GetPartialSize<0>(opInfo.sizeInElements.x)],
                        SwzRegSel::Dup(1U) | ms_ZextSwzTableX[GetPartialSize<1>(opInfo.sizeInElements.x)],
                    },
                },
            },
            {
                {
                    {
                        ms_ZextSwzTableY[GetPartialSize<0>(opInfo.sizeInElements.y)],
                        ms_ZextSwzTableY[GetPartialSize<0>(opInfo.sizeInElements.y)],
                    },
                    {
                        SwzRegSel::Dup(1U) | ms_ZextSwzTableY[GetPartialSize<1>(opInfo.sizeInElements.y)],
                        SwzRegSel::Dup(1U) | ms_ZextSwzTableY[GetPartialSize<1>(opInfo.sizeInElements.y)],
                    },
                },
                {
                    {
                        (ms_ZextSwzTableY[GetPartialSize<0>(opInfo.sizeInElements.y)] |
                            ms_ZextSwzTableX[GetPartialSize<0>(opInfo.sizeInElements.x)]),
                        (ms_ZextSwzTableY[GetPartialSize<0>(opInfo.sizeInElements.y)] |
                            ms_ZextSwzTableX[GetPartialSize<1>(opInfo.sizeInElements.x)]),
                    },
                    {
                        (SwzRegSel::Dup(1U) |
                            ms_ZextSwzTableY[GetPartialSize<1>(opInfo.sizeInElements.y)] |
                            ms_ZextSwzTableX[GetPartialSize<0>(opInfo.sizeInElements.x)]),
                        (SwzRegSel::Dup(1U) |
                            ms_ZextSwzTableY[GetPartialSize<1>(opInfo.sizeInElements.y)] |
                            ms_ZextSwzTableX[GetPartialSize<1>(opInfo.sizeInElements.x)]),
                    },
                },
            }
        }
        , m_Bottom(false)
        , m_Right(false)
        , m_InDfcAddr()
        , m_OutDfcAddr()
        , m_InramAddr(0)
        , m_OutramAddr(0)
        , m_UdmaLoader(activeEvents)
        , m_UdmaStorer(activeEvents)
    {
        ve_regrep_16<REG_DIV_OFF>(0U);
        ve_regrep_16<REG_DIV_MUL>(0U);

        m_DfcTraversal.SetUdmaStoreParams(Xy{ 1U, 1U });

        constexpr SwzSubRegSel subRegSelZeroExtend = ToSwzSubRegSel({
            // clang-format off
            {  0,  1,  2,  3 },
            {  4,  5,  6,  7 },
            {  8,  9, 10, 11 },
            { 12, 13, 14, 15 },
            // clang-format on
        });

        SetSwzRegSel<SWZ_ZERO_EXTEND_0_0>(SwzRegSel::Dup(0U));
        SetSwzRegSel<SWZ_ZERO_EXTEND_0_1>(SwzRegSel::Dup(0U));
        SetSwzRegSel<SWZ_ZERO_EXTEND_1_0>(SwzRegSel::Dup(1U));
        SetSwzRegSel<SWZ_ZERO_EXTEND_1_1>(SwzRegSel::Dup(1U));

        SetSwzSubRegSel<SWZ_ZERO_EXTEND_0_0>(subRegSelZeroExtend);
        SetSwzSubRegSel<SWZ_ZERO_EXTEND_0_1>(subRegSelZeroExtend);
        SetSwzSubRegSel<SWZ_ZERO_EXTEND_1_0>(subRegSelZeroExtend);
        SetSwzSubRegSel<SWZ_ZERO_EXTEND_1_1>(subRegSelZeroExtend);

        constexpr SwzSubRegSel subRegSelDown = ToSwzSubRegSel({
            // clang-format off
            { 12, 13, 14, 15 },
            {  0,  1,  2,  3 },
            {  4,  5,  6,  7 },
            {  8,  9, 10, 11 },
            // clang-format on
        });

        SetSwzRegSel<SWZ_LANE_SHIFT_DOWN_1_2_0>(RegSelDown(1, 2));
        SetSwzRegSel<SWZ_LANE_SHIFT_DOWN_1_2_1>(RegSelDown(1, 2));
        SetSwzRegSel<SWZ_LANE_SHIFT_DOWN_0_2_0>(RegSelDown(0, 2));
        SetSwzRegSel<SWZ_LANE_SHIFT_DOWN_0_2_1>(RegSelDown(0, 2));

        SetSwzSubRegSel<SWZ_LANE_SHIFT_DOWN_1_2_0>(subRegSelDown);
        SetSwzSubRegSel<SWZ_LANE_SHIFT_DOWN_1_2_1>(subRegSelDown);
        SetSwzSubRegSel<SWZ_LANE_SHIFT_DOWN_0_2_0>(subRegSelDown);
        SetSwzSubRegSel<SWZ_LANE_SHIFT_DOWN_0_2_1>(subRegSelDown);

        constexpr SwzSubRegSel subRegSelUp = ToSwzSubRegSel({
            // clang-format off
            {  4,  5,  6,  7 },
            {  8,  9, 10, 11 },
            { 12, 13, 14, 15 },
            {  0,  1,  2,  3 },
            // clang-format on
        });

        SetSwzRegSel<SWZ_LANE_SHIFT_UP_0_2_0>(RegSelUp(0, 2));
        SetSwzRegSel<SWZ_LANE_SHIFT_UP_0_2_1>(RegSelUp(0, 2));

        SetSwzSubRegSel<SWZ_LANE_SHIFT_UP_0_2_0>(subRegSelUp);
        SetSwzSubRegSel<SWZ_LANE_SHIFT_UP_0_2_1>(subRegSelUp);

        constexpr SwzSubRegSel subRegSelLeft = ToSwzSubRegSel({
            // clang-format off
            {   1,  2,  3,  0 },
            {   5,  6,  7,  4 },
            {   9, 10, 11,  8 },
            {  13, 14, 15, 12 },
            // clang-format on
        });

        SetSwzRegSel<SWZ_LANE_SHIFT_LEFT_0_2>(Transpose(RegSelUp(0, 2)));
        SetSwzRegSel<SWZ_LANE_SHIFT_LEFT_1_3>(Transpose(RegSelUp(1, 3)));

        SetSwzSubRegSel<SWZ_LANE_SHIFT_LEFT_0_2>(subRegSelLeft);
        SetSwzSubRegSel<SWZ_LANE_SHIFT_LEFT_1_3>(subRegSelLeft);

        constexpr SwzSubRegSel subRegSelRight = ToSwzSubRegSel({
            // clang-format off
            {   3,  0,  1,  2 },
            {   7,  4,  5,  6 },
            {  11,  8,  9, 10 },
            {  15, 12, 13, 14 },
            // clang-format on
        });

        SetSwzRegSel<SWZ_LANE_SHIFT_RIGHT_0_2>(Transpose(RegSelDown(0, 2)));
        SetSwzRegSel<SWZ_LANE_SHIFT_RIGHT_1_3>(Transpose(RegSelDown(1, 3)));

        SetSwzSubRegSel<SWZ_LANE_SHIFT_RIGHT_0_2>(subRegSelRight);
        SetSwzSubRegSel<SWZ_LANE_SHIFT_RIGHT_1_3>(subRegSelRight);

        constexpr DivInfo divInfo9 = GetDivInfo<9>();
        constexpr DivInfo divInfo6 = GetDivInfo<6>();
        constexpr DivInfo divInfo4 = GetDivInfo<4>();

        lsu::LoadMcuRf<WORDS_PER_REGISTER * REG_DIV_OFF>(divInfo9.off | (divInfo6.off << 8) | (divInfo6.off << 16) |
                                                         (divInfo4.off << 24));
        lsu::LoadMcuRf<WORDS_PER_REGISTER * REG_DIV_MUL>(divInfo9.mulLow | (divInfo6.mulLow << 8) |
                                                         (divInfo6.mulLow << 16) | (divInfo4.mulLow << 24));
        lsu::LoadMcuRf<WORDS_PER_REGISTER*(REG_DIV_MUL + 1)>(divInfo9.mulHigh | (divInfo6.mulHigh << 8) |
                                                             (divInfo6.mulHigh << 16) | (divInfo4.mulHigh << 24));

        SetSwzRegSel<SWZ_DIV_0>(SwzRegSel::Dup(0U));
        SetSwzRegSel<SWZ_DIV_1>(SwzRegSel::Dup(1U));
    }

    ncu_ple_interface::PleMsg::StripeDone operator()()
    {
        for (unsigned dfc = 0; dfc < NUM_SRAMS; dfc += NUM_PLE_LANES)
        {
            uint16_t inDfcAddrZ  = m_InDfcAddrBase;
            uint16_t outDfcAddrZ = m_OutDfcAddrBase;

            SetPleLanesInUse(NUM_PLE_LANES);

            for (unsigned z = dfc; z < m_SizeInGroups.z; z += NUM_SRAMS)
            {
                if ((m_SizeInGroups.z - z) == 1U)
                {
                    SetPleLanesInUse(1U);
                }

                m_InDfcAddr  = inDfcAddrZ;
                m_OutDfcAddr = outDfcAddrZ;

                for (unsigned y = m_SizeInGroups.y; y > 0; --y)
                {
                    const bool top    = y == m_SizeInGroups.y;
                    const bool bottom = y == 1U;

                    PoolRow(dfc, top, bottom);

                    if (top)
                    {
                        m_InDfcAddr = inDfcAddrZ;
                    }
                }

                const unsigned advZ = m_DfcTraversal.Advance(Xyz(0, 0, NUM_SRAMS));

                inDfcAddrZ += advZ;
                outDfcAddrZ += advZ;
            }
        }

        m_UdmaStorer.WaitForUdma();

        return ncu_ple_interface::PleMsg::StripeDone{};
    }

private:
    enum : unsigned
    {
        SWZ_ZERO_EXTEND_0_0,
        SWZ_ZERO_EXTEND_0_1,
        SWZ_ZERO_EXTEND_1_0,
        SWZ_ZERO_EXTEND_1_1,

        SWZ_LANE_SHIFT_DOWN_1_2_0,
        SWZ_LANE_SHIFT_DOWN_1_2_1,
        SWZ_LANE_SHIFT_DOWN_0_2_0,
        SWZ_LANE_SHIFT_DOWN_0_2_1,

        SWZ_LANE_SHIFT_UP_0_2_0,
        SWZ_LANE_SHIFT_UP_0_2_1,

        SWZ_LANE_SHIFT_LEFT_0_2,
        SWZ_LANE_SHIFT_LEFT_1_3,

        SWZ_LANE_SHIFT_RIGHT_0_2,
        SWZ_LANE_SHIFT_RIGHT_1_3,

        SWZ_DIV_0,
        SWZ_DIV_1,
    };

    static uint16_t Advance(uint16_t& addr, const unsigned adv, const unsigned mod = (1U << 16U))
    {
        const uint16_t oldAddr = addr;
        addr                   = (oldAddr + adv) % mod;
        return oldAddr;
    }

    void UdmaLoad(const unsigned dfcId)
    {
        udma::Address udmaAddr;

        udmaAddr.dfcAddrWords = m_InDfcAddr;
        udmaAddr.pleAddr      = m_InramAddr;

        m_UdmaLoader.WaitForUdma();
        m_UdmaLoader.Load(dfcId, udmaAddr);
    }

    void UdmaStore(const unsigned dfcId)
    {
        udma::Address udmaAddr;

        udmaAddr.dfcAddrWords = m_OutDfcAddr;
        udmaAddr.pleAddr      = m_OutramAddr;

        m_UdmaStorer.WaitForUdma();
        m_UdmaStorer.Store(dfcId, udmaAddr);
    }

    uint16_t AdvanceInput(const unsigned inGroupsY)
    {
        const uint16_t oldInramAddr = Advance(m_InramAddr, WORDS_PER_REGISTER * inGroupsY * TotalSize(GroupSize{}));
        Advance(m_InDfcAddr, m_DfcTraversal.Advance(Xyz(1U)));
        return oldInramAddr;
    }

    void AdvanceOutput()
    {
        Advance(m_OutramAddr, WORDS_PER_REGISTER * TotalSize(GroupSize{}), OUT_QUEUE_SIZE);
        Advance(m_OutDfcAddr, m_DfcTraversal.Advance(Xyz(1U)));
    }

    void LoadInputGroup(uint16_t inramId, uint16_t inramAddr, const bool top, const bool bottom)
    {
        // 6 input patches are loaded in registers 1,3,4-7,8,10 corresponding to XY coordinates in the order
        // as depicted below. We'll compute the vertical 1x3 pooling for the 4 patches in the center (cr4-cr7).
        //
        //        cr0     cr1     cr2     cr3     cr4     cr5     cr6     cr7     cr8     cr9     cr10    cr11
        //     +-------+-------+-------+-------+=======+=======+=======+=======+-------+-------+-------+-------+
        //     |       | (0,-1)|       | (1,-1)‖ (0,0) ‖ (0,1) ‖ (1,0) ‖ (1,1) ‖ (0,2) |       | (1,2) |       |
        //     +-------+-------+-------+-------+=======+=======+=======+=======+-------+-------+-------+-------+
        //
        // In spatial representation:
        //
        //      x →
        //    y +------+------+
        //    ↓ |      |      |
        //      +------+------+
        //      | cr1  | cr3  |
        //      +======+======+
        //      ‖ cr4  ‖ cr6  ‖
        //      +======+======+
        //      ‖ cr5  ‖ cr7  ‖
        //      +======+======+
        //      | cr8  | cr10 |
        //      +------+------+
        //      |      |      |
        //      +------+------+
        //

        if (top)
        {
            inramAddr -= WORDS_PER_REGISTER * TotalSize(GroupSize{});

            ve_regrep_8<1>(0U);
            ve_regrep_8<3>(0U);
        }

        if (bottom)
        {
            ve_regrep_8<8>(0U);
            ve_regrep_8<10>(0U);
        }

        if (!top)
        {
            lsu::LoadHalfInramRf<1>(inramId, { inramAddr, 0U });
            lsu::LoadHalfInramRf<3>(inramId, { inramAddr, 0U });
        }

        lsu::LoadInramRf<4>(inramId, { inramAddr, 0U });
        lsu::LoadInramRf<6>(inramId, { inramAddr, 0U });

        if (!bottom)
        {
            lsu::LoadHalfInramRf<8>(inramId, { inramAddr, 0U });
            lsu::LoadHalfInramRf<10>(inramId, { inramAddr, 0U });
        }
    }

    void VerticalPoolGroup(
        uint16_t inramId, const uint16_t inramAddr, const bool top, const bool bottom, const bool right)
    {
        LoadInputGroup(inramId, inramAddr, top, bottom);

        if ((bottom != m_Bottom) || (right != m_Right))
        {
            SetSwzRegSel<SWZ_ZERO_EXTEND_0_0>(m_ZextSwz[bottom][right][0][0]);
            SetSwzRegSel<SWZ_ZERO_EXTEND_0_1>(m_ZextSwz[bottom][right][0][1]);
            SetSwzRegSel<SWZ_ZERO_EXTEND_1_0>(m_ZextSwz[bottom][right][1][0]);
            SetSwzRegSel<SWZ_ZERO_EXTEND_1_1>(m_ZextSwz[bottom][right][1][1]);

            if (right != m_Right)
            {
                SetSwzRegSel<SWZ_LANE_SHIFT_DOWN_1_2_0>(RegSelDown(1, 2) | m_ZextSwz[0][right][0][0]);
                SetSwzRegSel<SWZ_LANE_SHIFT_DOWN_1_2_1>(RegSelDown(1, 2) | m_ZextSwz[0][right][0][1]);
                SetSwzRegSel<SWZ_LANE_SHIFT_DOWN_0_2_0>(RegSelDown(0, 2) | m_ZextSwz[0][right][0][0]);
                SetSwzRegSel<SWZ_LANE_SHIFT_DOWN_0_2_1>(RegSelDown(0, 2) | m_ZextSwz[0][right][0][1]);

                SetSwzRegSel<SWZ_LANE_SHIFT_UP_0_2_0>(RegSelUp(0, 2) | m_ZextSwz[0][right][0][0]);
                SetSwzRegSel<SWZ_LANE_SHIFT_UP_0_2_1>(RegSelUp(0, 2) | m_ZextSwz[0][right][0][1]);
            }

            m_Bottom = bottom;
            m_Right  = right;
        }

        if (k_IsSigned)
        {
            // extend signed 8 bit to signed 16 bit
            ve_swz_8<13, 4, REG_DIV_OFF, SWZ_ZERO_EXTEND_0_0, RwHazardDelay<VE_TIMING::SWZ_8, VE_TIMING::ASR_16>()>();
            ve_asr_16<12, 12, 8>();

            ve_swz_8<15, 6, REG_DIV_OFF, SWZ_ZERO_EXTEND_0_1, RwHazardDelay<VE_TIMING::SWZ_8, VE_TIMING::ASR_16>()>();
            ve_asr_16<14, 14, 8>();

            ve_swz_8<17, 4, REG_DIV_OFF, SWZ_ZERO_EXTEND_1_0, RwHazardDelay<VE_TIMING::SWZ_8, VE_TIMING::ASR_16>()>();
            ve_asr_16<16, 16, 8>();

            ve_swz_8<19, 6, REG_DIV_OFF, SWZ_ZERO_EXTEND_1_1, RwHazardDelay<VE_TIMING::SWZ_8, VE_TIMING::ASR_16>()>();
            ve_asr_16<18, 18, 8>();

            ve_swz_8<1, 0, 12, SWZ_LANE_SHIFT_DOWN_1_2_0, RwHazardDelay<VE_TIMING::SWZ_8, VE_TIMING::ASR_16>()>();
            ve_asr_16<0, 0, 8>();

            ve_swz_8<5, 2, 14, SWZ_LANE_SHIFT_DOWN_1_2_1, RwHazardDelay<VE_TIMING::SWZ_8, VE_TIMING::ASR_16>()>();
            ve_asr_16<4, 4, 8>();

            ve_swz_8<3, 12, 16, SWZ_LANE_SHIFT_DOWN_0_2_0, RwHazardDelay<VE_TIMING::SWZ_8, VE_TIMING::ASR_16>()>();
            ve_asr_16<2, 2, 8>();

            ve_swz_8<7, 14, 18, SWZ_LANE_SHIFT_DOWN_0_2_1, RwHazardDelay<VE_TIMING::SWZ_8, VE_TIMING::ASR_16>()>();
            ve_asr_16<6, 6, 8>();
        }

        else

        {
            ve_swz_8_zext_16<12, 4, REG_DIV_OFF, SWZ_ZERO_EXTEND_0_0>();
            ve_swz_8_zext_16<14, 6, REG_DIV_OFF, SWZ_ZERO_EXTEND_0_1>();

            ve_swz_8_zext_16<16, 4, REG_DIV_OFF, SWZ_ZERO_EXTEND_1_0>();
            ve_swz_8_zext_16<18, 6, REG_DIV_OFF, SWZ_ZERO_EXTEND_1_1>();

            ve_swz_8_zext_16<0, 0, 12, SWZ_LANE_SHIFT_DOWN_1_2_0>();
            ve_swz_8_zext_16<4, 2, 14, SWZ_LANE_SHIFT_DOWN_1_2_1>();

            ve_swz_8_zext_16<2, 12, 16, SWZ_LANE_SHIFT_DOWN_0_2_0>();
            ve_swz_8_zext_16<6, 14, 18, SWZ_LANE_SHIFT_DOWN_0_2_1>();
        }

        ve_add_16<0, 0, 12>();
        ve_add_16<4, 4, 14>();

        ve_add_16<2, 2, 16>();
        ve_add_16<6, 6, 18>();

        if (k_IsSigned)
        {
            ve_swz_8<13, 16, 12, SWZ_LANE_SHIFT_UP_0_2_0, RwHazardDelay<VE_TIMING::SWZ_8, VE_TIMING::ASR_16>()>();
            ve_asr_16<12, 12, 8>();

            ve_swz_8<15, 18, 14, SWZ_LANE_SHIFT_UP_0_2_1, RwHazardDelay<VE_TIMING::SWZ_8, VE_TIMING::ASR_16>()>();
            ve_asr_16<14, 14, 8>();

            ve_swz_8<17, 8, 16, SWZ_LANE_SHIFT_UP_0_2_0, RwHazardDelay<VE_TIMING::SWZ_8, VE_TIMING::ASR_16>()>();
            ve_asr_16<16, 16, 8>();

            ve_swz_8<19, 10, 18, SWZ_LANE_SHIFT_UP_0_2_1, RwHazardDelay<VE_TIMING::SWZ_8, VE_TIMING::ASR_16>()>();
            ve_asr_16<18, 18, 8>();
        }
        else
        {
            ve_swz_8<12, 16, 12, SWZ_LANE_SHIFT_UP_0_2_0>();
            ve_swz_8<14, 18, 14, SWZ_LANE_SHIFT_UP_0_2_1>();

            ve_swz_8<16, 8, 16, SWZ_LANE_SHIFT_UP_0_2_0>();
            ve_swz_8<18, 10, 18, SWZ_LANE_SHIFT_UP_0_2_1>();
        }

        ve_add_16<0, 0, 12>();
        ve_add_16<4, 4, 14>();
        ve_add_16<2, 2, 16>();
        ve_add_16<6, 6, 18>();
    }

    void LoadStashGroup(const uint16_t stashAddr, const bool left, const bool right)
    {
        // 12 input patches are loaded in registers 0-11 corresponding to 6 16-bit stashed vertical pooling results
        // as depicted below. We'll compute the horizontal 3x1 pooling for the patches in the center (cr4,cr6,cr8,cr10).
        //
        //      x →
        //    y +------+------+======+======+======+======+------+------+
        //    ↓ | cr0  |      ‖ cr4  |      ‖ c8   |      ‖ cr12 |      |
        //      +------+------+------+------+------+------+------+------+
        //      | cr2  |      ‖ cr6  |      ‖ cr10 |      ‖ cr14 |      |
        //      +------+------+======+======+======+======+------+------+
        //

        if (left)
        {
            ve_regrep_16<0>(0U);
            ve_regrep_16<2>(0U);
        }

        if (right)
        {
            ve_regrep_16<12>(0U);
            ve_regrep_16<14>(0U);
        }

        if (!left)
        {
            lsu::LoadOutramRf<0>({ stashAddr, 0U });
            lsu::LoadOutramRf<2>({ stashAddr, 0U });
        }

        lsu::LoadOutramRf<4>({ stashAddr, 0U });
        lsu::LoadOutramRf<6>({ stashAddr, 0U });
        lsu::LoadOutramRf<8>({ stashAddr, 0U });
        lsu::LoadOutramRf<10>({ stashAddr, 0U });

        if (!right)
        {
            lsu::LoadOutramRf<12>({ stashAddr, 0U });
            lsu::LoadOutramRf<14>({ stashAddr, 0U });
        }
    }

    template <unsigned Cr0, unsigned Cr1, unsigned Cr2, unsigned Cr3, unsigned Aux0, unsigned Aux1>
    void HorizontalPoolHalfGroup()
    {
        //      x →
        //    y +------+------+======+======+======+======+------+------+
        //    ↓ | Cr0  |      ‖ Cr1  |      ‖ Cr2  |      ‖ Cr3  |      |
        //      +------+------+======+======+======+======+------+------+
        //

        ve_swz_8<Aux0, Cr0, Cr1, SWZ_LANE_SHIFT_RIGHT_0_2>();
        ve_swz_8<Cr0 + 1, Cr0, Cr1, SWZ_LANE_SHIFT_RIGHT_1_3>();

        nop<1>();
        ve_mov_8<Cr0 + 0, Aux0>();

        ve_swz_8<Aux1, Cr3, Cr2, SWZ_LANE_SHIFT_LEFT_0_2>();
        ve_swz_8<Cr3 + 1, Cr3, Cr2, SWZ_LANE_SHIFT_LEFT_1_3>();

        nop<1>();
        ve_mov_8<Cr3 + 0, Aux1>();

        ve_add_16<Cr0, Cr0, Cr1>();

        nop<RwHazardDelay<VE_TIMING::SWZ_8, VE_TIMING::ADD_16, 2>()>();

        ve_add_16<Cr3, Cr3, Cr2>();

        ve_swz_8<Aux0 + 0, Cr2, Cr1, SWZ_LANE_SHIFT_LEFT_0_2>();
        ve_swz_8<Aux0 + 1, Cr2, Cr1, SWZ_LANE_SHIFT_LEFT_1_3>();

        ve_swz_8<Aux1 + 0, Cr1, Cr2, SWZ_LANE_SHIFT_RIGHT_0_2>();
        ve_swz_8<Aux1 + 1, Cr1, Cr2, SWZ_LANE_SHIFT_RIGHT_1_3>();

        ve_add_16<Cr0, Cr0, Aux0>();

        nop<RwHazardDelay<VE_TIMING::SWZ_8, VE_TIMING::ADD_16, 2>()>();

        ve_add_16<Cr3, Cr3, Aux1>();
    }

    template <unsigned Reg, unsigned Aux0, unsigned Aux1>
    void Normalize(const SwzSubRegSel divSubRegSel)
    {
        SetSwzSubRegSel<SWZ_DIV_0>(divSubRegSel);
        SetSwzSubRegSel<SWZ_DIV_1>(divSubRegSel);

        ve_swz_8_zext_16<Aux0, REG_DIV_OFF, REG_DIV_OFF, SWZ_DIV_0>();

        ve_swz_8<Aux1, REG_DIV_MUL, REG_DIV_MUL, SWZ_DIV_0>();
        ve_swz_8<Aux1 + 1U, REG_DIV_MUL, REG_DIV_MUL, SWZ_DIV_1>();

        ve_add_16<Reg, Reg, Aux0>();
        nop<RwHazardDelay<VE_TIMING::ADD_16, MMUL16_DELAY_TYPE>()>();
        MMul16<Reg, Reg, Aux1>();
    }

    void HorizontalPoolGroup(
        const uint16_t stashAddr, const bool top, const bool bottom, const bool left, const bool right)
    {
        LoadStashGroup(stashAddr, left, right);

        //      x →
        //    y +------+------+======+======+======+======+------+------+
        //    ↓ | cr0  |      ‖ cr4  |      ‖ c8   |      ‖ cr12 |      |
        //      +------+------+------+------+------+------+------+------+
        //      | cr2  |      ‖ cr6  |      ‖ cr10 |      ‖ cr14 |      |
        //      +------+------+======+======+======+======+------+------+
        //
        //      +------+------+
        //      | cr16 |      |
        //      +------+------+
        //      | cr18 |      |
        //      +------+------+
        //

        HorizontalPoolHalfGroup<0, 4, 8, 12, 16, 18>();
        HorizontalPoolHalfGroup<2, 6, 10, 14, 16, 18>();

        // xy = {0,0}
        Normalize<0, 16, 18>(m_DivSwzX[left][right][0] | m_DivSwzY[top][bottom][0]);
        // xy = {1,0}
        nop<WriteInOrderDelay<MMUL16_DELAY_TYPE, VE_TIMING::SWZ_8_ZEXT_16, 3>()>();
        Normalize<12, 16, 18>(m_DivSwzX[0][right][1] | m_DivSwzY[top][bottom][0]);
        // xy = {0,1}
        nop<WriteInOrderDelay<MMUL16_DELAY_TYPE, VE_TIMING::SWZ_8_ZEXT_16, 3>()>();
        Normalize<2, 16, 18>(m_DivSwzX[left][right][0] | m_DivSwzY[0][bottom][1]);
        // xy = {1,1}
        nop<WriteInOrderDelay<MMUL16_DELAY_TYPE, VE_TIMING::SWZ_8_ZEXT_16, 3>()>();
        Normalize<14, 16, 18>(m_DivSwzX[0][right][1] | m_DivSwzY[0][bottom][1]);

        ve_mov_8<1, 2>();
        ve_mov_8<2, 12>();
        nop<RwHazardDelay<MMUL16_DELAY_TYPE, VE_TIMING::MOV_8, 3>()>();
        ve_mov_8<3, 14>();

        nop<RwHazardDelay<VE_TIMING::MOV_8, VE_TIMING::STORE_RF_OUTRAM, 2>()>();

        lsu::StoreRfOutram<0>({ m_OutramAddr, 0U });
        lsu::StoreRfOutram<2>({ m_OutramAddr, 0U });

        nop<VE_TIMING::STORE_RF_OUTRAM::WRITE_BACK - 1U>();
    }

    void HorizontalPoolRow(const unsigned dfc, const bool top, const bool bottom)
    {
        uint16_t stashAddr = OUT_QUEUE_SIZE - (WORDS_PER_REGISTER * 4U);

        for (unsigned x = m_SizeInGroups.x; x > 0; --x)
        {
            const bool left  = x == m_SizeInGroups.x;
            const bool right = x == 1U;

            HorizontalPoolGroup(stashAddr, top, bottom, left, right);

            stashAddr += WORDS_PER_REGISTER * 8U;

            UdmaStore(dfc);
            AdvanceOutput();
        }
    }

    void PoolRow(const unsigned dfc, const bool top, const bool bottom)
    {
        const unsigned inGroupsY = 3U - top - bottom;

        m_DfcTraversal.SetUdmaLoadParams(Xy{ 1U, inGroupsY });
        UdmaLoad(dfc);

        uint16_t stashAddr = OUT_QUEUE_SIZE;

        for (unsigned x = m_SizeInGroups.x; x > 0; --x)
        {
            const bool right = x == 1U;

            const uint16_t inramAddr = AdvanceInput(inGroupsY);

            if (right)
            {
                m_UdmaLoader.WaitForUdma();
            }
            else
            {
                UdmaLoad(dfc);
            }

            // N77: dfc0->inram0
            // N57: dfc0->inram0, inram 1 not used
            // N37: dfc0->inram0, dfc1->inram1
            // N78: 1-to-1 mapping between dfc and inram. Udma can only transfer data
            // to an input sram from the matching dfc index
            VerticalPoolGroup(dfc, inramAddr, top, bottom, right);

            nop<ReadInOrderDelay<VE_TIMING::ADD_16, VE_TIMING::STORE_RF_OUTRAM>()>();

            // Stash result
            lsu::StoreRfOutram<0>({ stashAddr, 0U });
            lsu::StoreRfOutram<4>({ stashAddr, 0U });
            lsu::StoreRfOutram<2>({ stashAddr, 0U });
            lsu::StoreRfOutram<6>({ stashAddr, 0U });

            stashAddr += WORDS_PER_REGISTER * 8U;
        }

        HorizontalPoolRow(dfc, top, bottom);
    }

    static constexpr SwzSubRegSel ms_DivSwzLeft = Transpose(GetPartialPatchSwzY<SwzSubRegSel>(1, 0, 1, 0));
    static constexpr SwzSubRegSel ms_DivSwzTop  = GetPartialPatchSwzY<SwzSubRegSel>(1, 0, 2, 0);

    static constexpr DivSwzTable ms_DivSwzTableX = GetDivSwzTableX();
    static constexpr DivSwzTable ms_DivSwzTableY = GetDivSwzTableY();

    static constexpr ZextSwzTable ms_ZextSwzTableX = GetZextSwzTableX();
    static constexpr ZextSwzTable ms_ZextSwzTableY = GetZextSwzTableY();

    const Xyz m_SizeInGroups;

    const dfcsram::Traversal<GroupSize> m_DfcTraversal;

    const uint16_t m_InDfcAddrBase;
    const uint16_t m_OutDfcAddrBase;

    const SwzSubRegSel m_DivSwzX[2][2][2];
    const SwzSubRegSel m_DivSwzY[2][2][2];

    const SwzRegSel m_ZextSwz[2][2][2][2];

    bool m_Bottom;
    bool m_Right;

    uint16_t m_InDfcAddr;
    uint16_t m_OutDfcAddr;

    uint16_t m_InramAddr;
    uint16_t m_OutramAddr;

    udma::UdmaLoader m_UdmaLoader;
    udma::UdmaStorer m_UdmaStorer;
};

constexpr SwzSubRegSel AvgPool_3x3_1_1_Udma::ms_DivSwzLeft;
constexpr SwzSubRegSel AvgPool_3x3_1_1_Udma::ms_DivSwzTop;

constexpr DivSwzTable AvgPool_3x3_1_1_Udma::ms_DivSwzTableX;
constexpr DivSwzTable AvgPool_3x3_1_1_Udma::ms_DivSwzTableY;

constexpr ZextSwzTable AvgPool_3x3_1_1_Udma::ms_ZextSwzTableX;
constexpr ZextSwzTable AvgPool_3x3_1_1_Udma::ms_ZextSwzTableY;

}    // namespace

extern "C" __attribute__((noreturn)) void main()
{
    EnumBitset<Event> activeEvents;
    Main([&]() { WaitForEvent<Event::SETIRQ_EVENT>(activeEvents); },
         [&]() {
             return AvgPool_3x3_1_1_Udma{ activeEvents, GetOperatorInfo<OutputToInputIdentity>() }();
         });
}
