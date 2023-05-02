//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#include "../include/ethosn_ple/BlockConstants.hpp"
#include "../include/ethosn_ple/Common.hpp"
#include "../include/ethosn_ple/MceStripeLoop.hpp"
#include "../include/ethosn_ple/PassthroughBase.hpp"
#include "../include/ethosn_ple/SignedSupport.hpp"
#include "../include/ethosn_ple/Swizzle.hpp"

namespace
{
static_assert(((BlockSize::X == 4) && (BlockSize::Y == 4)) || ((BlockSize::X == 8) && (BlockSize::Y == 2)), "");

using OutBlockSize = sizes::BlockSize<BlockSize::X / 2, BlockSize::Y / 2>;

struct OutputToInput
{
    constexpr Xyz operator()(const Xyz out, const EnumBitset<Flags>) const
    {
        return { 2 * out.x, 2 * out.y, out.z };
    }
};

class MaxPool : public PassthroughBase<BlockSize, OutBlockSize, MaxPool>
{
public:
    using Base = PassthroughBase<BlockSize, OutBlockSize, MaxPool>;

    MaxPool(PleState& pleState, const OperatorInfo& opInfo)
        : Base(pleState.GetActiveEvents(), DivRoundUp(opInfo.sizeInElements, Xyz(2, 2, 1)), opInfo.output.dfcAddr)
    {
        constexpr SwzRegSel regSel = ToSwzRegSel({
            { 0, 0, 1, 1 },
            { 0, 0, 1, 1 },
            { 0, 0, 1, 1 },
            { 0, 0, 1, 1 },
        });

        constexpr SwzSubRegSel subRegSel0 = ToSwzSubRegSel({
            { 0, 8, 0, 8 },
            { 1, 9, 1, 9 },
            { 2, 10, 2, 10 },
            { 3, 11, 3, 11 },
        });

        constexpr SwzSubRegSel subRegSel1 = ToSwzSubRegSel({
            { 4, 12, 4, 12 },
            { 5, 13, 5, 13 },
            { 6, 14, 6, 14 },
            { 7, 15, 7, 15 },
        });

        SetSwzRegSel<SWZ_ROW_SELECT_TRANSPOSE_0>(regSel);
        SetSwzSubRegSel<SWZ_ROW_SELECT_TRANSPOSE_0>(subRegSel0);

        SetSwzRegSel<SWZ_ROW_SELECT_TRANSPOSE_1>(regSel);
        SetSwzSubRegSel<SWZ_ROW_SELECT_TRANSPOSE_1>(subRegSel1);
    }

    void ProcessBlock() const
    {
        using namespace VE_TIMING;

        VerticalMaxPoolTransposeGroup<0, 0>();
        VerticalMaxPoolTransposeGroup<2, 4>();
        VerticalMaxPoolTransposeGroup<0, 0>();

        VerticalMaxPoolTransposeGroup<2, 8>();
        VerticalMaxPoolTransposeGroup<4, 12>();
        VerticalMaxPoolTransposeGroup<2, 2>();

        if (BlockSize::X == 4)
        {
            static_assert(MOV_8::WRITE_BACK > (1 + MOV_8::OP_READ), "");

            nop<RwHazardDelay<UMAX_8, MOV_8, 2>()>();

            // Swap patches 1 and 2 for flipped-N order
            // The second mov reads operands before the first writes back. See the static assertion above.
            ve_mov_8<1, 2>();
            ve_mov_8<2, 1>();

            nop<RwHazardDelay<MOV_8, STORE_RF_OUTRAM>()>();
        }
        else
        {
            nop<RwHazardDelay<UMAX_8, STORE_RF_OUTRAM>()>();
        }
    }

private:
    enum : unsigned
    {
        SWZ_ROW_SELECT_TRANSPOSE_0,
        SWZ_ROW_SELECT_TRANSPOSE_1,
    };

    template <unsigned Dst, unsigned Src>
    static void VerticalMaxPoolTransposeGroup()
    {
        using namespace VE_TIMING;
        static_assert(SWZ_8::WRITE_BACK > (1 + SWZ_8::OP_READ), "");

        // The second swz reads operands before the first writes back. See the static assertion above.
        ve_swz_8<Src + 0, Src, Src, SWZ_ROW_SELECT_TRANSPOSE_0>();
        ve_swz_8<Src + 1, Src, Src, SWZ_ROW_SELECT_TRANSPOSE_1>();

        // The second swz reads operands before the first writes back. See the static assertion above.
        ve_swz_8<Src + 2, Src + 2, Src + 2, SWZ_ROW_SELECT_TRANSPOSE_0>();
        ve_swz_8<Src + 3, Src + 2, Src + 2, SWZ_ROW_SELECT_TRANSPOSE_1>();

        nop<RwHazardDelay<SWZ_8, MAX8_DELAY_TYPE, 2>()>();
        Max8<Dst + 0, Src + 0, Src + 1>();
        Max8<Dst + 1, Src + 2, Src + 3>();
    }
};
}    // namespace

extern "C" void __attribute__((noreturn)) main()
{
    MainWithStripeLoop<MceStripeLoop<MaxPool>, OutputToInput>();
}
