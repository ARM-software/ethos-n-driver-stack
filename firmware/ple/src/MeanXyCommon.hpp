//
// Copyright © 2018-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "../include/ethosn_ple/BlockConstants.hpp"
#include "../include/ethosn_ple/Common.hpp"
#include "../include/ethosn_ple/MceStripeLoop.hpp"
#include "../include/ethosn_ple/PassthroughBase.hpp"
#include "../include/ethosn_ple/SignedSupport.hpp"

namespace
{
struct DivInfo
{
    unsigned offset;
    unsigned multiplier;
};

constexpr unsigned COMMON_SHIFT = 20;

constexpr DivInfo GetDivInfo(const unsigned ksize)
{
    return (ksize == 0) ? DivInfo{ 0, 0 }
                        : DivInfo{ (ksize * ksize) / 2U, ((1U << COMMON_SHIFT) / (ksize * ksize)) + 1U };
}

constexpr DivInfo g_DivInfos[] = {
    GetDivInfo(0), GetDivInfo(1), GetDivInfo(2), GetDivInfo(3), GetDivInfo(4),
    GetDivInfo(5), GetDivInfo(6), GetDivInfo(7), GetDivInfo(8),
};

class MeanXy : public PassthroughBase<BlockSize, BlockSize, MeanXy>
{
public:
    MeanXy(PleState& pleState, const OperatorInfo& opInfo)
        : PassthroughBase<BlockSize, BlockSize, MeanXy>(
              pleState.GetActiveEvents(), opInfo.sizeInElements, opInfo.output.dfcAddr)
        , m_DivInfo(g_DivInfos[opInfo.sizeInElements.x])
    {
        // setup swizzle to left-shift two registers by 8, 4, 2 or 1 lane(s)
        // right-most lane is used as shift in value
        ve_set_swzsel_reg_sel<SWZ_ASL8_SEL0>(0b00000000000000000000000000000000);
        ve_set_swzsel_reg_sel<SWZ_ASL8_SEL1>(0b01010101010101010101010101010101);
        ve_set_swzsel_subreg_sel<SWZ_ASL8_SEL0>(0xFEDCBA98, 0xFFFFFFFF);
        ve_set_swzsel_subreg_sel<SWZ_ASL8_SEL1>(0xFEDCBA98, 0xFFFFFFFF);

        ve_set_swzsel_reg_sel<SWZ_ASL4_SEL0>(0b00000000000000000000000000000000);
        ve_set_swzsel_reg_sel<SWZ_ASL4_SEL1>(0b01010101010101010101010101010101);
        ve_set_swzsel_subreg_sel<SWZ_ASL4_SEL0>(0xBA987654, 0xFFFFFEDC);
        ve_set_swzsel_subreg_sel<SWZ_ASL4_SEL1>(0xBA987654, 0xFFFFFEDC);

        ve_set_swzsel_reg_sel<SWZ_ASL2_SEL0>(0b00000000000000000000000000000000);
        ve_set_swzsel_reg_sel<SWZ_ASL2_SEL1>(0b01010101010101010101010101010101);
        ve_set_swzsel_subreg_sel<SWZ_ASL2_SEL0>(0x98765432, 0xFFFEDCBA);
        ve_set_swzsel_subreg_sel<SWZ_ASL2_SEL1>(0x98765432, 0xFFFEDCBA);

        ve_set_swzsel_reg_sel<SWZ_ASL1_SEL0>(0b00000000000000000000000000000000);
        ve_set_swzsel_reg_sel<SWZ_ASL1_SEL1>(0b01010101010101010101010101010101);
        ve_set_swzsel_subreg_sel<SWZ_ASL1_SEL0>(0x87654321, 0xFFEDCBA9);
        ve_set_swzsel_subreg_sel<SWZ_ASL1_SEL1>(0x87654321, 0xFFEDCBA9);

        // use swizzle to select lane0 which contains the averaged value for the
        // first quarter block, Src1 is used to replicate zeroes to lane 1-15
        ve_set_swzsel_reg_sel<SWZ_OUTPUT_SEL>(0b10101010101010101010101010101000);
        ve_set_swzsel_subreg_sel<SWZ_OUTPUT_SEL>(0x0, 0);
    }

    void ProcessBlock()
    {
        using namespace VE_TIMING;

        if (k_IsSigned)
        {
            ve_regrep_16<REG_ACC_OUTPUT>(0);

            // Load input 0 to scratch, extend to signed 16bit
            ve_mov_8<REG_SCRATCH + 1, 0, RwHazardDelay<MOV_8, ASR_16>()>();
            ve_asr_16<REG_SCRATCH, REG_SCRATCH, 8, RwHazardDelay<ASR_16, ADD_16>()>();

            // Accumulate in REG_ACC_OUTPUT
            ve_add_16<REG_ACC_OUTPUT, REG_ACC_OUTPUT, REG_SCRATCH, RwHazardDelay<ADD_16, MOV_8>()>();

            // Load input 1 to scratch, extend to signed 16bit
            ve_mov_8<REG_SCRATCH + 1, 1, RwHazardDelay<MOV_8, ASR_16>()>();
            ve_asr_16<REG_SCRATCH, REG_SCRATCH, 8, RwHazardDelay<ASR_16, ADD_16>()>();

            // Accumulate in REG_ACC_OUTPUT
            ve_add_16<REG_ACC_OUTPUT, REG_ACC_OUTPUT, REG_SCRATCH, RwHazardDelay<ADD_16, MOV_8>()>();

            // Load input 2 to scratch, extend to signed 16bit
            ve_mov_8<REG_SCRATCH + 1, 2, RwHazardDelay<MOV_8, ASR_16>()>();
            ve_asr_16<REG_SCRATCH, REG_SCRATCH, 8, RwHazardDelay<ASR_16, ADD_16>()>();

            // Accumulate in REG_ACC_OUTPUT
            ve_add_16<REG_ACC_OUTPUT, REG_ACC_OUTPUT, REG_SCRATCH, RwHazardDelay<ADD_16, MOV_8>()>();

            // Load input 3 to scratch, extend to signed 16bit
            ve_mov_8<REG_SCRATCH + 1, 3, RwHazardDelay<MOV_8, ASR_16>()>();
            ve_asr_16<REG_SCRATCH, REG_SCRATCH, 8, RwHazardDelay<ASR_16, ADD_16>()>();

            // Accumulate in REG_ACC_OUTPUT
            ve_add_16<REG_ACC_OUTPUT, REG_ACC_OUTPUT, REG_SCRATCH, RwHazardDelay<ADD_16, SWZ_8>()>();

            // clear registers 1-3 for output
            ve_regrep_8<1>(0);
            ve_regrep_16<2>(0);
        }
        else
        {
            // use right-shift to clear acc (no value larger than 16-bit is used)
            ve_lsr16acc();

            // accumulate values for first 8x8 elements
            ve_addacc_8<0, 0>();
            ve_addacc_8<1, 0>();
            ve_addacc_8<2, 0>();
            ve_addacc_8<3, 0>();

            // clear registers 1-3 for output
            ve_regrep_8<1>(0);
            ve_regrep_16<2>(0);

            ve_movreg_16<REG_ACC_OUTPUT, 0, RwHazardDelay<MOVREG_16, SWZ_8>()>();
        }

        constexpr unsigned delaySwzAdd        = RwHazardDelay<SWZ_8, ADD_16>();
        constexpr unsigned delayAddSwz        = RwHazardDelay<ADD_16, SWZ_8>();
        constexpr unsigned delayAddRegrepadd  = RwHazardDelay<ADD_16, REGREPADD_16>();
        constexpr unsigned delayRegrepaddMmul = RwHazardDelay<REGREPADD_16, MMUL16_DELAY_TYPE>();
        constexpr unsigned delayMmulLsr       = RwHazardDelay<MMUL16_DELAY_TYPE, LSR_16>();
        constexpr unsigned delayLsrSwz        = RwHazardDelay<LSR_16, SWZ_8>();

        // accumulate lane[0-7] and lane[8-15] in lane[0-7] of REG_ACC_OUTPUT
        rf_asl8_16<REG_ACC_INPUT, REG_ACC_OUTPUT, delaySwzAdd>();
        ve_add_16<REG_ACC_OUTPUT, REG_ACC_INPUT, REG_ACC_OUTPUT, delayAddSwz>();

        // accumulate lane[0-3] and lane[4-7] in lane[0-3] of REG_ACC_OUTPUT
        rf_asl4_16<REG_ACC_INPUT, REG_ACC_OUTPUT, delaySwzAdd>();
        ve_add_16<REG_ACC_OUTPUT, REG_ACC_INPUT, REG_ACC_OUTPUT, delayAddSwz>();

        // accumulate lane[0-1] and lane[2-3] in lane[0-1] of REG_ACC_OUTPUT
        rf_asl2_16<REG_ACC_INPUT, REG_ACC_OUTPUT, delaySwzAdd>();
        ve_add_16<REG_ACC_OUTPUT, REG_ACC_INPUT, REG_ACC_OUTPUT, delayAddSwz>();

        // accumulate lane[0] and lane[1] in lane[0] of REG_ACC_OUTPUT
        rf_asl1_16<REG_ACC_INPUT, REG_ACC_OUTPUT, delaySwzAdd>();
        ve_add_16<REG_ACC_OUTPUT, REG_ACC_INPUT, REG_ACC_OUTPUT, delayAddRegrepadd>();

        // divide - step 0: prepare register with magic value
        ve_regrep_16<REG_DIV>(m_DivInfo.multiplier);
        // divide - step 1: add rounding offset
        ve_regrepadd_16<REG_ACC_OUTPUT, REG_ACC_OUTPUT, delayRegrepaddMmul>(m_DivInfo.offset);
        // divide - step 2: multiply with magic value, extract upper 16 bits
        MMul16<REG_DIV, REG_DIV, REG_ACC_OUTPUT, delayMmulLsr>();
        // divide - step 4: perform remaining shift
        ve_lsr_16<REG_DIV, REG_DIV, COMMON_SHIFT - 16, delayLsrSwz>();

        // write lane 0 output, reg2 is used to replicate zeros to lane 1-15
        ve_swz_8<0, REG_DIV, 2, SWZ_OUTPUT_SEL>();

        // Prevent read-before-write hazard when this result is stored to the output RAM.
        nop<RwHazardDelay<SWZ_8, STORE_RF_OUTRAM>()>();
    }

private:
    static constexpr unsigned int SWZ_ASL8_SEL0  = 0;
    static constexpr unsigned int SWZ_ASL8_SEL1  = 1;
    static constexpr unsigned int SWZ_ASL4_SEL0  = 2;
    static constexpr unsigned int SWZ_ASL4_SEL1  = 3;
    static constexpr unsigned int SWZ_ASL2_SEL0  = 4;
    static constexpr unsigned int SWZ_ASL2_SEL1  = 5;
    static constexpr unsigned int SWZ_ASL1_SEL0  = 6;
    static constexpr unsigned int SWZ_ASL1_SEL1  = 7;
    static constexpr unsigned int SWZ_OUTPUT_SEL = 8;

    static constexpr unsigned int REG_DIV        = 18;
    static constexpr unsigned int REG_ACC_OUTPUT = 22;
    static constexpr unsigned int REG_ACC_INPUT  = 8;
    static constexpr unsigned int REG_SCRATCH    = 6;

    template <unsigned int dst, unsigned int src, unsigned int post_cc = 0>

    void rf_asl8_16()
    {
        ve_swz_8<dst, src, src, SWZ_ASL8_SEL0>();
        ve_swz_8<dst + 1, src, src, SWZ_ASL8_SEL1, post_cc>();
    }

    template <unsigned int dst, unsigned int src, unsigned int post_cc = 0>
    void rf_asl4_16()
    {
        ve_swz_8<dst, src, src, SWZ_ASL4_SEL0>();
        ve_swz_8<dst + 1, src, src, SWZ_ASL4_SEL1, post_cc>();
    }

    template <unsigned int dst, unsigned int src, unsigned int post_cc = 0>
    void rf_asl2_16()
    {
        ve_swz_8<dst, src, src, SWZ_ASL2_SEL0>();
        ve_swz_8<dst + 1, src, src, SWZ_ASL2_SEL1, post_cc>();
    }

    template <unsigned int dst, unsigned int src, unsigned int post_cc = 0>
    void rf_asl1_16()
    {
        ve_swz_8<dst, src, src, SWZ_ASL1_SEL0>();
        ve_swz_8<dst + 1, src, src, SWZ_ASL1_SEL1, post_cc>();
    }

    const DivInfo m_DivInfo;
};
}    // namespace
