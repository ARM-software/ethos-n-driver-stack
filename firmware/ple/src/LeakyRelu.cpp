//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#include "../include/ethosn_ple/BlockConstants.hpp"
#include "../include/ethosn_ple/MceStripeLoop.hpp"
#include "../include/ethosn_ple/Passthrough2.hpp"

namespace
{

// This symbol needs to be defined to avoid a link error.
// It is used to zero out the C++ classes but we don't need those to be zero initialised and it is slow
__attribute__((used)) extern "C" void __aeabi_memclr8(void* dest, size_t n)
{}

__noinline void SetEncodedShift(volatile Cdp2Inst& asrSat, const unsigned shift)
{
    asrSat.SetRm(shift);
}

template <unsigned I>
void SetEncodedShift(const unsigned shift)
{
    // This modifies the CDP2 instructions stored at the address of the
    // label ASRSat_32_16_<I> to execute the correct amount of right shift.
    // This is done by modifying the Rm field of the CDP2 instruction
    volatile Cdp2Inst* asrSat;
    // cppcheck-suppress uninitvar
    __ASM("ADR %[asrSat], ASRSat_32_16_%c[I]" : [asrSat] "=r"(asrSat) : [I] "n"(I));
    SetEncodedShift(*asrSat, shift);
}

// We can model leaky relu as a mul and max:
//                             +-----------+
//                             |           |
//                   +--------->    Mul    +-------------+
//                   |         |           |             |
//                   |         +-----------+             |
//                   |                                   |
//                   |                                   |
// +-----------+     |                             +-----v-----+         +-----v-----+
// |           |     |                             |           |         |           |
// |   Input   +-----+----------------------------->    Max    +--------->   Output  |
// |           |                                   |           |         |           |
// +-----------+                                   +-----------+         +-----------+
//
// The data has to be requantized to the output quantization before going into the max.
// The mul can be performed as part of the requantization for that branch.
// The "first" input in the OperatorInfo struct has quantization info for the input -> max path.
// The "second" input in the OperatorInfo struct has the combined quantization info for the input -> mul -> max path.
class LeakyRelu
{
public:
    void Init(const StripeInfo& info)
    {
        // Initialize the input zero point in reg 18
        ve_regrep_16<kZeroPointReg>(static_cast<uint32_t>(info.inputs[0].zeroPoint));
        // Initialize requantization multiplier of the first branch in reg 20
        ve_regrep_16<kMult0Reg>(info.inputs[0].multiplier);
        // Initialize requantization multiplier of the second branch in reg 22
        ve_regrep_16<kMult1Reg>(info.inputs[1].multiplier);

        // We need to implement a rounding rshift to avoid a bias in the error.
        // We do that by right-shifting by (shift - 1), adding 1 and right-shifting by 1.
        // Self-modifying code: modify the shift instructions that will be used for
        // requantization with the corresponding shift.
        SetEncodedShift<0>(info.inputs[0].shift - 1);
        SetEncodedShift<1>(info.inputs[1].shift - 1);
        SetEncodedShift<2>(info.inputs[0].shift - 1);
        SetEncodedShift<3>(info.inputs[1].shift - 1);
        SetEncodedShift<4>(info.inputs[0].shift - 1);
        SetEncodedShift<5>(info.inputs[1].shift - 1);
        SetEncodedShift<6>(info.inputs[0].shift - 1);
        SetEncodedShift<7>(info.inputs[1].shift - 1);

        // We keep the output zero point in a member variable because we're already using all the
        // 24 registers in the register file.
        outZeroPoint = static_cast<uint32_t>(info.output.zeroPoint);
    }

    template <Xy patchesInGroup>
    void ProcessGroup(const PassthroughState& ctx) const
    {
        // Load 4 input patches in registers 0-3
        cexec::UncheckedExec(LoadGroup<patchesInGroup>(ctx.og, ctx.inramAddrGroup));
        ProcessGroup(ctx);
    }

private:
    __noinline void ProcessGroup(const PassthroughState& ctx) const
    {
        using namespace cexec;

        // Extend the input patches in registers 0-3 to 16b and leave the result in regs 0-7.
        // Reverse order to avoid read-before-write conflicts.
        const auto extendTo16b =
            std::tuple_cat(ConvertTo16b<6, 3>(), ConvertTo16b<4, 2>(), ConvertTo16b<2, 1>(), ConvertTo16b<0, 0>());

        // Rescale the four 16b patches twice to produce the result of the two requantization branches before the max.
        // Destination and temporary registers are chosen in an effort to minimize conflicts. Results are in regs 2-17.
        //
        // Logical register usage sequence:
        //
        // |   0 |   2 |   4 |   6 |   8 |  10 |  12 |  14 |  16 |  18 |  20 |  22 |
        // *************************************************************************
        // |  i0 |  i1 |  i2 |  i3 |                             |  zp |  m1 |  m2 |
        // |  i0 |  i1 |  i2 |  i3 |                 |     tmp   |  zp |  m1 |  m2 |
        // |  i0 |  i1 |  i2 |  i3 |                       | r3a |  zp |  m1 |  m2 |
        // |  i0 |  i1 |  i2 |                 |    tmp    | r3a |  zp |  m1 |  m2 |
        // |  i0 |  i1 |  i2 |                       | r3b | r3a |  zp |  m1 |  m2 |
        // |  i0 |  i1 |  i2 |           |    tmp    | r3b | r3a |  zp |  m1 |  m2 |
        // |  i0 |  i1 |  i2 |                 | r2a | r3b | r3a |  zp |  m1 |  m2 |
        // |  i0 |  i1 |           |    tmp    | r2a | r3b | r3a |  zp |  m1 |  m2 |
        // |  i0 |  i1 |                 | r2b | r2a | r3b | r3a |  zp |  m1 |  m2 |
        // |  i0 |  i1 |     |    tmp    | r2b | r2a | r3b | r3a |  zp |  m1 |  m2 |
        // |  i0 |  i1 |           | r1a | r2b | r2a | r3b | r3a |  zp |  m1 |  m2 |
        // |  i0 |     |    tmp    | r1a | r2b | r2a | r3b | r3a |  zp |  m1 |  m2 |
        // |  i0 |           | r1b | r1a | r2b | r2a | r3b | r3a |  zp |  m1 |  m2 |
        // |  i0 |    tmp    | r1b | r1a | r2b | r2a | r3b | r3a |  zp |  m1 |  m2 |
        // |  i0 |     | r0a | r1b | r1a | r2b | r2a | r3b | r3a |  zp |  m1 |  m2 |
        // |    tmp    | r0a | r1b | r1a | r2b | r2a | r3b | r3a |  zp |  m1 |  m2 |
        //       | r0b | r0a | r1b | r1a | r2b | r2a | r3b | r3a |  zp |  m1 |  m2 |
        //
        const auto rescale0 = Rescale<16, 6, kZeroPointReg, kMult0Reg, 0, 14, 0>();
        const auto rescale1 = Rescale<14, 6, kZeroPointReg, kMult1Reg, 0, 12, 1>();
        const auto rescale2 = Rescale<12, 4, kZeroPointReg, kMult0Reg, 0, 10, 2>();
        const auto rescale3 = Rescale<10, 4, kZeroPointReg, kMult1Reg, 0, 8, 3>();
        const auto rescale4 = Rescale<8, 2, kZeroPointReg, kMult0Reg, 0, 6, 4>();
        const auto rescale5 = Rescale<6, 2, kZeroPointReg, kMult1Reg, 0, 4, 5>();
        const auto rescale6 = Rescale<4, 0, kZeroPointReg, kMult0Reg, 0, 2, 6>();
        const auto rescale7 = Rescale<2, 0, kZeroPointReg, kMult1Reg, 0, 0, 7>();

        const auto rescale =
            std::tuple_cat(rescale0, rescale1, rescale2, rescale3, rescale4, rescale5, rescale6, rescale7);

        // Take the max of the two requantization branches for each of the 4 pairs. Results in regs 0-7.
        const std::tuple takeMax = {
            SMax16<0, 2, 4>{},
            SMax16<2, 6, 8>{},
            SMax16<4, 10, 12>{},
            SMax16<6, 14, 16>{},
        };

        // Add the output zero point that was stored in the member variable. Results in regs 0-7.
        const std::tuple addOutZeroPoint = {
            RegrepAdd16<0, 0>{ outZeroPoint },
            RegrepAdd16<2, 2>{ outZeroPoint },
            RegrepAdd16<4, 4>{ outZeroPoint },
            RegrepAdd16<6, 6>{ outZeroPoint },
        };

        // Saturate to 8b and leave the result in regs 0-3.
        const auto saturate = std::tuple_cat(Sat_16_8<0, 0, 0, 8, true>(), Sat_16_8<1, 2, 2, 8>(),
                                             Sat_16_8<2, 4, 4, 8>(), Sat_16_8<3, 6, 6, 8>());

        // Store the result to output sram.
        const auto store = StoreGroup(ctx.outramAddrGroup);

        Exec(std::tuple_cat(extendTo16b, rescale, takeMax, addOutZeroPoint, saturate, store));
    }

    static constexpr unsigned kZeroPointReg = 18;
    static constexpr unsigned kMult0Reg     = 20;
    static constexpr unsigned kMult1Reg     = 22;

    uint32_t outZeroPoint;
};
}    // namespace

extern "C" __attribute__((noreturn)) void main()
{
    Passthrough<LeakyRelu>::Main();
}
