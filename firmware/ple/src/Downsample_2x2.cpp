//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#include "../include/ethosn_ple/BlockConstants.hpp"
#include "../include/ethosn_ple/Common.hpp"
#include "../include/ethosn_ple/MceStripeLoop.hpp"
#include "../include/ethosn_ple/PassthroughBase.hpp"
#include "../include/ethosn_ple/Swizzle.hpp"

namespace    // Internal linkage
{
// PLE operator for downsampling IFM with a stride of (2,2) using the swizzle instruction.
//
// Each BlockOperator() call processes one block of 4x4 input patches, producing 1 blocks of 2x2 output patches
//                        or processes one block of 8x2 input patches, producing 1 blocks of 4x1 output patches
//                        or processes one block of 4x2 input patches, producing 1 blocks of 2x1 output patches
//                        or processes one block of 2x2 input patches, producing 1 blocks of 1x1 output patches.
//
// The downsample via swizzle is implemented as follows:
//
//   * For each group of 2x2 patches swizzle instruction extracts 1 patch.

enum DownsampleSwzId : unsigned
{
    DOWNSAMPLE_SWZ_0
};

struct OutputToInput
{
    constexpr Xyz operator()(const Xyz out, const EnumBitset<Flags>) const
    {
        return { 2 * out.x, 2 * out.y, out.z };
    }
};

using OutBlockSize = sizes::BlockSize<BlockSize::X / 2, BlockSize::Y / 2>;

class Downsample_2x2 : public PassthroughBase<BlockSize, OutBlockSize, Downsample_2x2>
{
public:
    using Base = PassthroughBase<BlockSize, OutBlockSize, Downsample_2x2>;

    Downsample_2x2(PleState& pleState, const OperatorInfo& opInfo)
        : Base(pleState.GetActiveEvents(), DivRoundUp(opInfo.sizeInElements, Xyz(2, 2, 1)), opInfo.output.dfcAddr)
    {
        // Register select represents the source register Cr to Cr+3 represented on 2 bits [0 to 3]
        //
        // 2-bit source per-lane:  3311331122002200
        //                         -> F5F5A0A0 in hex (every 4 bits contain 2 register-selects)
        //                         -> 11 11 01 01 11 11 01 01 10 10 00 00 10 10 00 00 (in binary)
        constexpr SwzRegSel regSel = ToSwzRegSel({
            { 0, 0, 2, 2 },
            { 0, 0, 2, 2 },
            { 1, 1, 3, 3 },
            { 1, 1, 3, 3 },
        });

        // Swizzle pattern A: Move lane 0 of source register into lane 0 of destination
        //                    Move lane 2 of source register into lane 1 of destination
        //                    ...
        //                    Move lane 8 of source register into lane 4 of destination
        //                    ...
        //
        //                                   Destination lane:
        // Source lane: A8A82020A8A82020  -> FEDCBA9876543210
        constexpr HalfSwzSubRegSel subRegSel = ToHalfSwzSubRegSel({
            { 0, 2, 0, 2 },
            { 8, 10, 8, 10 },
        });

        ve_set_swzsel_reg_sel<DOWNSAMPLE_SWZ_0>(regSel);
        ve_set_swzsel_subreg_sel<DOWNSAMPLE_SWZ_0>(subRegSel, subRegSel);
    }

    void ProcessBlock() const
    {
        using namespace VE_TIMING;
        using namespace static_loop;

        nop<1>();

        For<Range<0, OutBlockSize::Y>, Range<0, OutBlockSize::X>>::Invoke(ProcessGroup{});

        // Prevent read-before-write hazard when this result is stored to the output RAM.
        nop<RwHazardDelay<SWZ_8, STORE_RF_OUTRAM>()>();
    }

private:
    struct ProcessGroup
    {
        template <unsigned Y, unsigned X>
        void operator()() const
        {
            constexpr unsigned Src = PATCHES_PER_GROUP * ((OutBlockSize::X * Y) + X);
            constexpr unsigned Dst = (OutBlockSize::Y * X) + Y;
            ve_swz_8<Dst, Src, Src + 2, DOWNSAMPLE_SWZ_0>();
        }
    };
};
}    // namespace

extern "C" void __attribute__((noreturn)) main()
{
    MainWithStripeLoop<MceStripeLoop<Downsample_2x2>, OutputToInput>();
}
