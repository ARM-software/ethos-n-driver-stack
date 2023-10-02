//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_ple/BlockConstants.hpp"
#include "../include/ethosn_ple/Common.hpp"
#include "../include/ethosn_ple/MceStripeLoop.hpp"
#include "../include/ethosn_ple/Passthrough2.hpp"
#include "../include/ethosn_ple/SignedSupport.hpp"
#include "../include/ethosn_ple/Swizzle.hpp"

// Sigmoid is an elementwise operation with f(x) = 1 / (1 + exp(-x))
// We do this in 16-bit fixed point precision (8 bits whole number, 8 bits fractional),
// so that the output has reasonable accuracy.
// First we convert the 8-bit quantized input into the 16-bit fixed point representation,
// then perform the exp, add and division, then convert back into 8-bit quantized for the output.
//
// To avoid having to add lots of NOPs, we process four patches "in parallel" by interleaving
// the VE instructions from each patch, so that while one patch is waiting for something to happen,
// another patch can be doing something useful.
// Each of the Stage0ForPatch, Stage1ForPatch, etc. functions do a couple of instructions from a
// particular patch (provided as a template argument) and then the ProcessGroupCommon() function
// interleaves all these together.
// We also 'stagger' the interleaved patches, so that we're not executing the equivalent instruction
// for the next patch immediately after the previous patch. This is because some VE instructions
// can't be repeated straight after (e.g. MUL), so by staggering we mix up the instructions and avoid more nops.
//
// Register usage:
//   0-3 start with the input values (4 patches) and the results are placed back here when done
//   4-7 are used for common constants, used for all patch calculations
//      4-5 16-bit signed zero point value
//      6-7 16-bit unsigned multiplier for rescale
//   8-23 are used to store intermediate values for calculations. As we do four patches in parallel:
//      8-11 are used by the first patch
//      12-15 are used by the second patch
//      16-19 are used by the third patch
//      20-23 are used by the fourth patch

// This symbol needs to be defined to avoid a link error.
// It does not appear to be used anywhere though, so it's a bit of a mystery.
__attribute__((used)) extern "C" void __aeabi_memclr4()
{}

namespace
{

// This symbol needs to be defined to avoid a link error.
// It is used to zero out the C++ classes but we don't need those to be zero initialised and it is slow
__attribute__((used)) extern "C" void __aeabi_memclr8(void* dest, size_t n)
{}

class Sigmoid
{
public:
    Sigmoid()
    {}

    void Init(const StripeInfo& info)
    {
        // Store common constants in registers for later
        ve_regrep_16<REG_ZERO_POINT>(static_cast<uint32_t>(info.inputs[0].zeroPoint));
        ve_regrep_16<REG_MULTIPLIER>(info.inputs[0].multiplier);

        // Update the 4 x ASR instructions with the shift value which we only know at runtime
        const uint32_t shift = info.inputs[0].shift;
        volatile Cdp2Inst* asrSat;
        // cppcheck-suppress uninitvar
        __ASM("ADR %[asrSat], INSTRUCTION_FOR_MODIFICATION_0" : [asrSat] "=r"(asrSat));
        asrSat->SetRm(shift);
        __ASM("ADR %[asrSat], INSTRUCTION_FOR_MODIFICATION_1" : [asrSat] "=r"(asrSat));
        asrSat->SetRm(shift);
        __ASM("ADR %[asrSat], INSTRUCTION_FOR_MODIFICATION_2" : [asrSat] "=r"(asrSat));
        asrSat->SetRm(shift);
        __ASM("ADR %[asrSat], INSTRUCTION_FOR_MODIFICATION_3" : [asrSat] "=r"(asrSat));
        asrSat->SetRm(shift);
    }

    template <Xy patchesInGroup>
    void ProcessGroup(const PassthroughState& ctx) const;

private:
    static constexpr int REG_ZERO_POINT = 4;
    static constexpr int REG_MULTIPLIER = 6;

    void ProcessGroupCommon(const PassthroughState& ctx) const;

    // See above comments on register usage. This gets the register to use as intermediate
    // number I (0-3) for patch number P (0-3).
    template <int P, int I>
    static constexpr unsigned int IReg()
    {
        static_assert(P < 4, "Only four patches available!");
        static_assert(I < 4, "Only four regs available per patch!");
        return 8 + P * 4 + I;
    }

    template <int P>
    void Stage0ForPatch() const
    {
        // Load input value from reg number P into our intermediate space (reg 1)
        ve_mov_8<IReg<P, 1>(), P>();
    }

    template <int P>
    void Stage1ForPatch() const
    {
        // Zero-extend or sign-extend (for uint8 or int8 input data respectively)
        // so that intermediate regs 0-1 represent the input data in int16 format.
        SR16<IReg<P, 0>(), IReg<P, 0>(), 8>();
    }

    template <int P>
    void Stage2ForPatch() const
    {
        // Take absolute value and requantize the input into fixed point 8.8
        //     y = (abs(x - zero_point) * multiplier ) >> shift
        // The multiplier and shift are calculated offline in the support library and passed to us

        // 0-1 = x - zeroPoint
        ve_sub_16<IReg<P, 0>(), IReg<P, 0>(), REG_ZERO_POINT>();
    }

    template <int P>
    void Stage3ForPatch() const
    {
        // Take abs value. We do this to avoid numerical imprecision with very negative input values.
        // This leads to the exp calculation returning a very large number, which is limited to 16-bits and so is
        // saturated and then when we do the reciprocal, we get the wrong answer (2/256, rather then 0/256 or 1/256).
        // Instead we use the fact that sigmoid is symmetrical in this way: f(-x) = 1 - f(x)
        // We have less precision issues with very positive input values, so this helps.
        // We take the absolute value here, and then do the "1 - " bit after we have the final answer,
        // for input values that were negative.

        // We need to remember which elements were positive/negative so we know which ones to invert later.
        // Conveniently we already have intermediate reg 1 which will be all 1s for negative and all 0s for positive,
        // so save this over the input value for later use (we have no other intermediate regs to use, and the input
        // register is no longer needed).
        ve_mov_8<P, IReg<P, 1>()>();

        ve_abs_16<IReg<P, 0>(), IReg<P, 0>()>();
    }

    template <int P>
    void Stage4ForPatch() const
    {
        // Multiply by multiplier (which is one part of the overall scale). This gives a 32-bit result in 0-3.
        ve_umull_16<IReg<P, 0>(), IReg<P, 0>(), REG_MULTIPLIER>();
    }

    template <int P>
    void Stage5ForPatch() const
    {
        // Shift right (which is one part of the overall scale) and saturate to 16-bit. This gives a 16-bit result in 0-1.
        // The shift amount here is set to zero, but is replaced at runtime by self-modifying code in Init().
        __ASM volatile("INSTRUCTION_FOR_MODIFICATION_%c0:" ::"n"(P));
        ve_lsrsat_32_16<IReg<P, 0>(), IReg<P, 0>(), 0>();

        // We now have y = abs((x - zero_point) * multiplier) >> shift in 0-1.
        // This is unsigned and so always >= 0
    }

    template <int P>
    void Stage6ForPatch() const
    {
        // Load constant zero into 2-3 for later use
        ve_regrep_16<IReg<P, 2>()>(0);
    }

    template <int P>
    void Stage7ForPatch() const
    {
        // Negate (y = 0 - x)
        ve_sub_16<IReg<P, 0>(), IReg<P, 2>(), IReg<P, 0>()>();
        // The value is now signed and always <= 0
    }

    template <int P>
    void Stage8ForPatch() const
    {
        // Exponential (y = e^x)
        ve_exp2_16<IReg<P, 0>(), IReg<P, 0>(), 0>();
        // Result is 0 <= x <= 1.0, as input is <= 0
    }

    template <int P>
    void Stage9ForPatch() const
    {
        // Add one. Note that because the 16-bit value in 0-1 is in 8.8 fixed point, we add one
        // to the upper byte. We know this can't overflow because the this byte will either be 0 or 1,
        // as the overall number is <= 1.
        ve_regrepadd_8<IReg<P, 1>(), IReg<P, 1>()>(1);
        // Result is 1 <= x <= 2
    }

    template <int P>
    void Stage10ForPatch() const
    {
        // Reciprocal (y = 1/x)
        ve_rcp_16<IReg<P, 0>(), IReg<P, 0>(), 0>();
        // Result is 0.5 <= x <= 1.0
    }

    template <int P>
    void Stage11ForPatch() const
    {
        // If result is exactly 1.0, saturate to 0.FF (as we only have one byte for our output)
        ve_sub_8<IReg<P, 0>(), IReg<P, 0>(), IReg<P, 1>()>();
    }

    template <int P>
    void Stage12ForPatch() const
    {
        // Encode into the output quantisation. The output quantisation is always fixed.
        if (k_IsSigned)
        {
            // Output zero point depends on the datatype (0 for uint8, -128 for int8)
            ve_regrepadd_16<IReg<P, 0>(), IReg<P, 0>()>(static_cast<uint32_t>(k_SmallestValue));
        }
    }

    template <int P>
    void Stage13ForPatch() const
    {
        // We now have the result in 0, but we may need to invert (256 - x) each individual element if
        // the original input was negative. We stored a mask (all 0s or 1s) for this earlier which we use.
        // When the mask is all zeroes, this xor and sub does nothing, when it is all 1s, it performs 256 - x!
        ve_xor_8<IReg<P, 0>(), IReg<P, 0>(), P>();
    }

    template <int P>
    void Stage14ForPatch() const
    {
        ve_sub_8<P, IReg<P, 0>(), P>();
    }
};

template <Xy patchesInGroup>
void Sigmoid::ProcessGroup(const PassthroughState& ctx) const
{
    // Load 4 input patches in registers 0-3
    cexec::UncheckedExec(LoadGroup<patchesInGroup>(ctx.og, ctx.inramAddrGroup));

    // Call into common code to avoid many copies of that function (as this function is templated x 4)
    ProcessGroupCommon(ctx);
}

void Sigmoid::ProcessGroupCommon(const PassthroughState& ctx) const
{
    // Process patches four at a time, interleaved and staggered

    Stage0ForPatch<0>();
    nop<2>();

    Stage1ForPatch<0>();
    Stage0ForPatch<1>();
    nop<1>();

    Stage2ForPatch<0>();
    Stage1ForPatch<1>();
    Stage0ForPatch<2>();

    Stage3ForPatch<0>();
    Stage2ForPatch<1>();
    Stage1ForPatch<2>();
    Stage0ForPatch<3>();

    Stage4ForPatch<0>();
    nop<3>();
    Stage3ForPatch<1>();
    Stage2ForPatch<2>();
    Stage1ForPatch<3>();

    Stage5ForPatch<0>();
    nop<1>();    // Not sure why this one is needed - the model doesn't seem to so there must be a difference with the HW
    Stage4ForPatch<1>();
    nop<3>();
    Stage3ForPatch<2>();
    Stage2ForPatch<3>();

    nop<2>();
    Stage6ForPatch<0>();
    Stage5ForPatch<1>();
    nop<1>();    // Not sure why this one is needed - the model doesn't seem to so there must be a difference with the HW
    Stage4ForPatch<2>();
    nop<3>();
    Stage3ForPatch<3>();

    nop<1>();
    Stage7ForPatch<0>();
    Stage6ForPatch<1>();
    Stage5ForPatch<2>();
    nop<1>();    // Not sure why this one is needed - the model doesn't seem to so there must be a difference with the HW
    Stage4ForPatch<3>();

    nop<3>();
    Stage8ForPatch<0>();
    Stage7ForPatch<1>();
    Stage6ForPatch<2>();
    Stage5ForPatch<3>();

    nop<1>();
    Stage9ForPatch<0>();
    Stage8ForPatch<1>();
    Stage7ForPatch<2>();
    Stage6ForPatch<3>();

    nop<1>();
    Stage10ForPatch<0>();
    nop<3>();
    Stage9ForPatch<1>();
    Stage8ForPatch<2>();
    Stage7ForPatch<3>();

    nop<2>();
    Stage11ForPatch<0>();
    Stage10ForPatch<1>();
    nop<3>();
    Stage9ForPatch<2>();
    Stage8ForPatch<3>();

    Stage12ForPatch<0>();
    nop<3>();
    Stage11ForPatch<1>();
    Stage10ForPatch<2>();
    nop<3>();
    Stage9ForPatch<3>();

    Stage13ForPatch<0>();
    Stage12ForPatch<1>();
    Stage11ForPatch<2>();
    Stage10ForPatch<3>();

    Stage14ForPatch<0>();
    Stage13ForPatch<1>();
    nop<1>();
    Stage12ForPatch<2>();
    nop<2>();
    Stage11ForPatch<3>();

    Stage14ForPatch<1>();
    Stage13ForPatch<2>();
    Stage12ForPatch<3>();

    nop<2>();
    Stage14ForPatch<2>();
    Stage13ForPatch<3>();

    nop<2>();
    Stage14ForPatch<3>();

    nop<1>();

    cexec::UncheckedExec(StoreGroup(ctx.outramAddrGroup));
}

}    // namespace

extern "C" void __attribute__((noreturn)) main()
{
    Passthrough<Sigmoid>::Main();
}
