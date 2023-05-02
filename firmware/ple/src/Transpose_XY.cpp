//
// Copyright © 2020,2022-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#include "../include/ethosn_ple/BlockConstants.hpp"
#include "../include/ethosn_ple/Common.hpp"
#include "../include/ethosn_ple/MceStripeLoop.hpp"
#include "../include/ethosn_ple/PassthroughBase.hpp"
#include "../include/ethosn_ple/Swizzle.hpp"

namespace
{
using InBlockSize  = BlockSize;
using OutBlockSize = sizes::BlockSize<BlockSize::Y, BlockSize::X>;

struct OutputToInput
{
    constexpr Xyz operator()(const Xyz& out, const EnumBitset<Flags>) const
    {
        return TransposeXY(out);
    }
};

// Converts patch Xy coordinates to offset in RF.
//
// Example: 16x16 block with patch coordinates x, y
//      x→ 0      1      2      3
//    y
//    ↓ +======+======+======+======+
//    0 ‖  p0  |  p2  ‖  p4  |  p6  ‖
//      +------+------+------+------+
//    1 ‖  p1  |  p3  ‖  p5  |  p7  ‖
//      +======+======+======+======+
//    2 ‖  p8  |  p10 ‖  p12 |  p14 ‖
//      +------+------+------+------+
//    3 ‖  p9  |  p11 ‖  p13 |  p15 ‖
//      +======+======+======+======+
//
// Index of patch is equivalent to index of RF register holding patch data.
constexpr unsigned XyToLinear(const Xy& coord, const Xy& blockSize)
{
    const Xy groupStride   = { PATCHES_PER_GROUP, PATCHES_PER_GROUP_1D * blockSize.x };
    const Xy inGroupStride = { PATCHES_PER_GROUP_1D, 1U };

    const Xy groupXy   = coord / Xy(PATCHES_PER_GROUP_1D, PATCHES_PER_GROUP_1D);
    const Xy inGroupXy = coord % Xy(PATCHES_PER_GROUP_1D, PATCHES_PER_GROUP_1D);

    return Dot(groupXy, groupStride) + Dot(inGroupXy, inGroupStride);
}

class Transpose_XY;
using Base = PassthroughBase<BlockSize, OutBlockSize, Transpose_XY, true>;

class Transpose_XY : public Base
{
public:
    Transpose_XY(PleState& pleState, const OperatorInfo& opInfo)
        : Base(pleState.GetActiveEvents(), TransposeXY(opInfo.sizeInElements), opInfo.output.dfcAddr)
    {
        // The mask provides access to even RF registers
        constexpr SwzRegSel regSel0 = ToSwzRegSel({
            { 0, 0, 0, 0 },
            { 0, 0, 0, 0 },
            { 0, 0, 0, 0 },
            { 0, 0, 0, 0 },
        });

        // The mask provides access to odd RF registers
        constexpr SwzRegSel regSel1 = ToSwzRegSel({
            { 1, 1, 1, 1 },
            { 1, 1, 1, 1 },
            { 1, 1, 1, 1 },
            { 1, 1, 1, 1 },
        });

        // Swizzle subregister describing how to transpose single patch
        constexpr SwzSubRegSel subRegSel = Transpose(ToSwzSubRegSel({
            { 0, 1, 2, 3 },
            { 4, 5, 6, 7 },
            { 8, 9, 10, 11 },
            { 12, 13, 14, 15 },
        }));

        // Usage of 2 swizzle registers is needed because swz instruction source registers
        // are always need to be even
        SetSwzRegSel<SWZ_TRANSPOSE_0>(regSel0);
        SetSwzSubRegSel<SWZ_TRANSPOSE_0>(subRegSel);

        SetSwzRegSel<SWZ_TRANSPOSE_1>(regSel1);
        SetSwzSubRegSel<SWZ_TRANSPOSE_1>(subRegSel);
    }

    void ProcessBlock()
    {
        using namespace static_loop;

        For<Range<0, InBlockSize::X>, Range<0, InBlockSize::Y>>::Invoke(TransposeFn{});

        nop<RwHazardDelay<VE_TIMING::SWZ_8, VE_TIMING::STORE_RF_OUTRAM>()>();
    }

private:
    enum : unsigned
    {
        SWZ_TRANSPOSE_0,
        SWZ_TRANSPOSE_1,
    };

    // Transpose block
    struct TransposeFn
    {
        template <unsigned Src, unsigned Dst>
        static void TransposePatch()
        {
            // Src - (Src % 2) is needed because registers Rn, Rm for swz8 instruction supports only even RF registers.
            // Access to odd registers are done using appropriate swizzle index map.
            constexpr unsigned SrcAligned = Src - (Src % 2);
            constexpr unsigned SwzOp      = ((Src % 2) == 0) ? SWZ_TRANSPOSE_0 : SWZ_TRANSPOSE_1;

            ve_swz_8<Dst, SrcAligned, SrcAligned, SwzOp>();
        }

        // The main idea of the operator is to receive (SrcX, SrcY) coordinates of source patch and perform transpose
        // on one or two patches. In case if source patch does not change its position only one swizzle is needed.
        // Otherwise two patches need to be transposed together. This is required to avoid second patch data overwrite.
        template <unsigned SrcX, unsigned SrcY>
        void operator()() const
        {
            using namespace VE_TIMING;

            // Calculate coordinates of RF register for source patch.
            constexpr unsigned SrcRegIdx = XyToLinear({ SrcX, SrcY }, Xy(InBlockSize{}));

            // Calculate coordinates of RF register for destination patch using same formula as for source
            // but with transposed block size and patch coordinates.
            constexpr unsigned DstRegIdx = XyToLinear({ SrcY, SrcX }, Xy(OutBlockSize{}));

            // Only one swizzle is needed.
            if (SrcRegIdx == DstRegIdx)
            {
                TransposePatch<SrcRegIdx, DstRegIdx>();
                nop<1>();
            }
            // Pairs are only processed once.
            else if (SrcRegIdx < DstRegIdx)
            {
                static_assert(SWZ_8::WRITE_BACK > (1 + SWZ_8::OP_READ), "");
                // Temporary copy of dest patch, as it gets overwritten
                ve_mov_8<23, DstRegIdx>();
                TransposePatch<SrcRegIdx, DstRegIdx>();
                nop<1>();
                TransposePatch<23, SrcRegIdx>();
                nop<1>();
            }
        }
    };
};

}    // namespace

extern "C" void __attribute__((noreturn)) main()
{
    MainWithStripeLoop<MceStripeLoop<Transpose_XY>, OutputToInput>();
}
