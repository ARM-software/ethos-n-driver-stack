//
// Copyright © 2018-2020,2022-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#include "../include/ethosn_ple/BlockConstants.hpp"
#include "../include/ethosn_ple/Common.hpp"
#include "../include/ethosn_ple/MceStripeLoop.hpp"
#include "../include/ethosn_ple/PassthroughBase.hpp"

namespace    // Internal linkage
{
// PLE operator for interleaving [interleave-8/16] the IFM with a stride of (2,2), using the swizzle instruction.
//
// Each BlockOperator() call processes one block of 4x4 input patches, producing 4 blocks of 2x2 output patches,
// each output block corresponding to a different output channel.
//
// The interleave via swizzle is implemented as follows:
//
//   * For each of the 4 groups of 2x2 patches in the 4x4 input block, swizzle instructions extract 4 patches,
//     each corresponding to one of the output channels.
//   * The produced patches are moved so all patches of the same output channel are contiguous in flipped-N order
//     in their corresponding 2x2 output block.

static_assert(BlockSize::X == 4, "Only input blocks of width 4 supported so far");
static_assert(BlockSize::Y == 4, "Only input blocks of height 4 supported so far");

// Swizzle pattern A: Move lane 0 of source register into lane 0 of destination
//                    Move lane 2 of source register into lane 1 of destination
//                    ...
//                    Move lane 8 of source register into lane 4 of destination
//                    ...
//
//                                   Destination lane:
// Source lane: A8A82020A8A82020  -> FEDCBA9876543210
constexpr unsigned int SWIZZLE_PATTERN_A = 0xA8A82020;

// Swizzle pattern B: Move lane 1 of source register into lane 0 of destination
//                    Move lane 3 of source register into lane 1 of destination
//                    ...
//                    Move lane 9 of source register into lane 4 of destination
//                    ...
//
//                                   Destination lane:
// Source lane: B9B93131B9B93131  -> FEDCBA9876543210
constexpr unsigned int SWIZZLE_PATTERN_B = 0xB9B93131;

// Swizzle pattern C: Move lane 4 of source register into lane 0 of destination
//                    Move lane 6 of source register into lane 1 of destination
//                    ...
//                    Move lane C of source register into lane 4 of destination
//                    ...
//
//                                   Destination lane:
// Source lane: ECEC6464ECEC6464  -> FEDCBA9876543210
constexpr unsigned int SWIZZLE_PATTERN_C = 0xECEC6464;

// Swizzle pattern D: Move lane 5 of source register into lane 0 of destination
//                    Move lane 7 of source register into lane 1 of destination
//                    ...
//                    Move lane D of source register into lane 4 of destination
//                    ...
//
//                                   Destination lane:
// Source lane: FDFD7575FDFD7575  -> FEDCBA9876543210
constexpr unsigned int SWIZZLE_PATTERN_D = 0xFDFD7575;

// Register select represents the source register Cr to Cr+3 represented on 2 bits [0 to 3]
//
// 2-bit source per-lane:  3311331122002200
//                         -> F5F5A0A0 in hex (every 4 bits contain 2 register-selects)
//                         -> 11 11 01 01 11 11 01 01 10 10 00 00 10 10 00 00 (in binary)
constexpr unsigned int reg_sel = 0xF5F5A0A0;

enum InterleaveSwzId : unsigned
{
    INTERLEAVE_SWZ_0,
    INTERLEAVE_SWZ_1,
    INTERLEAVE_SWZ_2,
    INTERLEAVE_SWZ_3
};

struct OutputToInput
{
    constexpr Xyz operator()(const Xyz out, const EnumBitset<Flags>) const
    {
        return { 2 * out.x, 2 * out.y,
                 (out.z % TOTAL_NUM_SRAMS) + ((out.z / (TOTAL_NUM_SRAMS * 4)) * TOTAL_NUM_SRAMS) };
    }
};

template <unsigned Grp, unsigned Patch>
constexpr unsigned PatchOfGroup()
{
    static_assert(Grp < 6, "");
    static_assert(Patch < 4, "");

    return (4U * Grp) + Patch;
}

template <unsigned DstGrp, unsigned SrcGrp>
void InterleaveGroup()
{
    // We swap INTERLEAVE_SWZ_1 and INTERLEAVE_SWZ_2 intentionally
    ve_swz_8<PatchOfGroup<DstGrp, 0>(), PatchOfGroup<SrcGrp, 0>(), PatchOfGroup<SrcGrp, 2>(), INTERLEAVE_SWZ_0>();
    ve_swz_8<PatchOfGroup<DstGrp, 1>(), PatchOfGroup<SrcGrp, 0>(), PatchOfGroup<SrcGrp, 2>(), INTERLEAVE_SWZ_2>();
    ve_swz_8<PatchOfGroup<DstGrp, 2>(), PatchOfGroup<SrcGrp, 0>(), PatchOfGroup<SrcGrp, 2>(), INTERLEAVE_SWZ_1>();
    ve_swz_8<PatchOfGroup<DstGrp, 3>(), PatchOfGroup<SrcGrp, 0>(), PatchOfGroup<SrcGrp, 2>(), INTERLEAVE_SWZ_3>();
}

template <unsigned DstGrp, unsigned SrcGrp>
void MovGroup()
{
    ve_mov_16<PatchOfGroup<DstGrp, 0>(), PatchOfGroup<SrcGrp, 0>()>();
    ve_mov_16<PatchOfGroup<DstGrp, 2>(), PatchOfGroup<SrcGrp, 2>()>();
}

using OutBlockSize = sizes::BlockSize<BlockSize::X / 2, BlockSize::Y / 2, 4>;

class Interleave_2x2_2_2 : public PassthroughBase<BlockSize, OutBlockSize, Interleave_2x2_2_2>
{
public:
    using Base = PassthroughBase<BlockSize, OutBlockSize, Interleave_2x2_2_2>;

    Interleave_2x2_2_2(PleState& pleState, const OperatorInfo& opInfo)
        : Base(pleState.GetActiveEvents(),
               Xyz(DivRoundUp(Xy(opInfo.sizeInElements), Xy(2, 2)),
                   RoundUpToMultiple(opInfo.sizeInElements.z, TOTAL_NUM_SRAMS) * 4),
               opInfo.output.dfcAddr)

    {
        ve_set_swzsel_reg_sel<INTERLEAVE_SWZ_0>(reg_sel);
        ve_set_swzsel_reg_sel<INTERLEAVE_SWZ_1>(reg_sel);
        ve_set_swzsel_reg_sel<INTERLEAVE_SWZ_2>(reg_sel);
        ve_set_swzsel_reg_sel<INTERLEAVE_SWZ_3>(reg_sel);

        ve_set_swzsel_subreg_sel<INTERLEAVE_SWZ_0>(SWIZZLE_PATTERN_A, SWIZZLE_PATTERN_A);
        ve_set_swzsel_subreg_sel<INTERLEAVE_SWZ_1>(SWIZZLE_PATTERN_B, SWIZZLE_PATTERN_B);
        ve_set_swzsel_subreg_sel<INTERLEAVE_SWZ_2>(SWIZZLE_PATTERN_C, SWIZZLE_PATTERN_C);
        ve_set_swzsel_subreg_sel<INTERLEAVE_SWZ_3>(SWIZZLE_PATTERN_D, SWIZZLE_PATTERN_D);
    }

    void ProcessBlock()
    {
        using namespace VE_TIMING;

        InterleaveGroup<4, 0>();
        InterleaveGroup<5, 1>();

        MovGroup<0, 4>();

        InterleaveGroup<4, 2>();

        MovGroup<1, 5>();

        InterleaveGroup<5, 3>();

        // We need patches of the same interleave together.

        // Interleave 2 patches are in position 1 in each group
        ve_mov_8<PatchOfGroup<2, 0>(), PatchOfGroup<0, 1>()>();
        ve_mov_8<PatchOfGroup<2, 1>(), PatchOfGroup<4, 1>()>();
        ve_mov_8<PatchOfGroup<2, 2>(), PatchOfGroup<1, 1>()>();
        ve_mov_8<PatchOfGroup<2, 3>(), PatchOfGroup<5, 1>()>();

        // Interleave 3 patches are in position 3 in each group
        ve_mov_8<PatchOfGroup<3, 0>(), PatchOfGroup<0, 3>()>();
        ve_mov_8<PatchOfGroup<3, 1>(), PatchOfGroup<4, 3>()>();
        ve_mov_8<PatchOfGroup<3, 2>(), PatchOfGroup<1, 3>()>();
        ve_mov_8<PatchOfGroup<3, 3>(), PatchOfGroup<5, 3>()>();

        // Interleave 0 patches are in position 0 in each group
        ve_mov_8<PatchOfGroup<0, 1>(), PatchOfGroup<4, 0>()>();
        ve_mov_8<PatchOfGroup<0, 3>(), PatchOfGroup<5, 0>()>();

        // Interleave 1 patches are in position 2 in each group
        ve_mov_8<PatchOfGroup<1, 1>(), PatchOfGroup<4, 2>()>();
        ve_mov_8<PatchOfGroup<1, 3>(), PatchOfGroup<5, 2>()>();

        // Swap patches <0, 2> and <1, 0>, using 23 as a temporary (should be unused by now)
        ve_mov_8<23, PatchOfGroup<1, 0>()>();
        ve_mov_8<PatchOfGroup<1, 0>(), PatchOfGroup<0, 2>()>();
        nop<1>();
        ve_mov_8<PatchOfGroup<0, 2>(), 23>();

        // Prevent read-before-write hazard when this result is stored to the output RAM.
        nop<RwHazardDelay<MOV_8, STORE_RF_OUTRAM>()>();
    }
};
}    // namespace

extern "C" void __attribute__((noreturn)) main()
{
    MainWithStripeLoop<MceStripeLoop<Interleave_2x2_2_2>, OutputToInput>();
}
