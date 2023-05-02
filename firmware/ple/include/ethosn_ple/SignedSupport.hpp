//
// Copyright © 2020-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <generated/cdp_opcodes.h>
#include <generated/mcr_opcodes.h>

namespace
{
constexpr bool k_IsSigned = IS_SIGNED;

constexpr int k_SmallestValue = k_IsSigned ? -128 : 0;
constexpr int k_LargestValue  = k_IsSigned ? 127 : 255;

template <unsigned Dst, unsigned Src1, unsigned Src2, unsigned int post_cc = 0>
void Max8()
{
    if (k_IsSigned)
    {
        ve_smax_8<Dst, Src1, Src2, post_cc>();
    }
    else
    {
        ve_umax_8<Dst, Src1, Src2, post_cc>();
    }
}

using MAX8_DELAY_TYPE = std::conditional_t<k_IsSigned, VE_TIMING::SMAX_8, VE_TIMING::UMAX_8>;

template <unsigned Dst, unsigned Src1, unsigned Src2, unsigned int post_cc = 0>
void MMul16()
{
    if (k_IsSigned)
    {
        ve_smmul_16<Dst, Src1, Src2, post_cc>();
    }
    else
    {
        ve_ummul_16<Dst, Src1, Src2, post_cc>();
    }
}

using MMUL16_DELAY_TYPE = std::conditional_t<k_IsSigned, VE_TIMING::SMMUL_16, VE_TIMING::UMMUL_16>;

template <unsigned int Dest, unsigned int Src, unsigned int Shift, unsigned int post_cc = 0>
void SR16()
{
    if (k_IsSigned)
    {
        ve_asr_16<Dest, Src, Shift>();
    }
    else
    {
        ve_lsr_16<Dest, Src, Shift>();
    }
}

using SR16_DELAY_TYPE = std::conditional_t<k_IsSigned, VE_TIMING::ASR_16, VE_TIMING::LSR_16>;

struct Saturate_16_8_BeforeDelayUnsigned
{
    // The source register is only read by the SMAX_16 operation but we add
    // a distance of 1 to take the REGREP_16 operation into account
    static constexpr unsigned int OP_READ    = VE_TIMING::SMAX_16::OP_READ + 1;
    static constexpr unsigned int WRITE_BACK = VE_TIMING::REGREP_16::WRITE_BACK;
    static constexpr unsigned int PIPELINE   = VE_TIMING::REGREP_16::PIPELINE;
};

template <unsigned int Dst, unsigned int Src, unsigned int Scratch>
void Saturate_16_8()
{
    if (k_IsSigned)
    {
        // shift right by 0 has the effect to truncate to 8 bits
        // and saturates.
        ve_asrsat_16_8<Dst, Src, 0>();
    }
    else
    {
        ve_regrep_16<Scratch>(0);
        // No need to insert NOP here because they can go one after each other into the pipeline
        ve_smax_16<Dst, Src, Scratch>();
        nop<RwHazardDelay<VE_TIMING::SMAX_16, VE_TIMING::LSRSAT_16_8>()>();
        // shift right by 0 to truncate the value to 8 bits
        ve_lsrsat_16_8<Dst, Dst, 0>();
    }
}

using SATURATE_BEFORE_DELAY_TYPE =
    std::conditional_t<k_IsSigned, VE_TIMING::ASRSAT_16_8, Saturate_16_8_BeforeDelayUnsigned>;
using SATURATE_AFTER_DELAY_TYPE = std::conditional_t<k_IsSigned, VE_TIMING::ASRSAT_16_8, VE_TIMING::LSRSAT_16_8>;

}    // namespace
