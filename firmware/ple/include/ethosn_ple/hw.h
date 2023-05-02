//
// Copyright © 2018-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Cmsis.hpp"

#include <generated/mcr_opcodes.h>
#include <ncu_ple_interface_def.h>
#include <scylla_addr_fields.h>
#include <scylla_regs.h>

#include <cstddef>
#include <cstdint>

// ====================================================
// Defines for register/patch/group/block relationships
// ====================================================

namespace    // Internal linkage
{
// NUM_OFM is equal to the number of CE's times NUM_MCEIF
constexpr unsigned NUM_CES         = NUM_OFM / NUM_MCEIF;
constexpr unsigned TOTAL_NUM_SRAMS = NUM_CES * NUM_SRAMS;

constexpr unsigned int NUM_REGISTERS = 24;

// A register is 128-bits and an element is 8-bits
constexpr unsigned int ELEMENTS_PER_REGISTER = 16;
constexpr unsigned int WORDS_PER_REGISTER    = 4;

// A patch is 4x4 elements
constexpr unsigned int ELEMENTS_PER_PATCH_1D = 4;
constexpr unsigned int ELEMENTS_PER_PATCH    = ELEMENTS_PER_PATCH_1D * ELEMENTS_PER_PATCH_1D;

// A group is 2x2 patches
constexpr unsigned int PATCHES_PER_GROUP_1D  = 2;
constexpr unsigned int ELEMENTS_PER_GROUP_1D = ELEMENTS_PER_PATCH_1D * PATCHES_PER_GROUP_1D;
constexpr unsigned int ELEMENTS_PER_GROUP    = ELEMENTS_PER_GROUP_1D * ELEMENTS_PER_GROUP_1D;
constexpr unsigned int PATCHES_PER_GROUP     = PATCHES_PER_GROUP_1D * PATCHES_PER_GROUP_1D;
constexpr unsigned int REGISTERS_PER_GROUP   = PATCHES_PER_GROUP * ELEMENTS_PER_PATCH / ELEMENTS_PER_REGISTER;
constexpr unsigned int WORDS_PER_GROUP       = WORDS_PER_REGISTER * REGISTERS_PER_GROUP;

const uint32_t& g_CeId = *reinterpret_cast<const uint32_t*>(PLE_REG(CE_RP, CE_CE_ID));

inline void write_reg(uint32_t reg_offset, uint32_t value, uint32_t reg_page = CE_RP)
{
    *reinterpret_cast<volatile uint32_t*>(PLE_REG(reg_page, reg_offset)) = value;
}

inline uint32_t read_reg(uint32_t reg_offset, uint32_t reg_page = CE_RP)
{
    return *reinterpret_cast<volatile const uint32_t*>(PLE_REG(reg_page, reg_offset));
}

inline void SetPleLanesInUse(const unsigned numZ)
{
    if (NUM_PLE_LANES > 1U)
    {
        static_assert((NUM_PLE_LANES == 1U) || (NUM_PLE_LANES == 2U), "");

        if (numZ == 1U)
        {
            ve_set_ple_lane_sel<>(0b01);
        }
        else
        {
            ve_set_ple_lane_sel<>(0b11);
        }

        nop<VE_TIMING::SET_PLE_LANE_SEL::PIPELINE - 1U>();
    }
}

inline void SignalBufferFreed(unsigned n = 1U)
{
    ce_setirq_r irq;
    irq.set_buffer_freed(event_create_t::CREATE);

#pragma unroll 1
    do
    {
        write_reg(CE_CE_SETIRQ, irq.word);
        --n;
    } while (n > 0U);
}

inline void SignalPleStripeDone()
{
    ce_setirq_r irq;
    irq.set_stripe_done(event_create_t::CREATE);
    write_reg(CE_CE_SETIRQ, irq.word);
}

// Sanity check of the MCE Interface and SRAM counts.
// These depend on the target hardware and will affect how
// the OFMs are written to the SRAM.
//
// The number of SRAMs and MCE Interfaces varies between different
// products as follows:
//   Product  NUM_MCEIF  NUM_SRAMS
//   N77          1          1
//   N57          2          1
//   N37          2          2
//   N78          4          4
//                4          2
static_assert(NUM_SRAMS == 1 || NUM_SRAMS == 2 || NUM_SRAMS == 4, "Number of SRAMs unsupported");
static_assert(NUM_MCEIF == 1 || NUM_MCEIF == 2 || NUM_MCEIF == 4, "Number of MCE Interfaces unsupported");
static_assert(NUM_MCEIF >= NUM_SRAMS, "Number of SRAMs not compatible with number of MCE Interfaces");
}    // namespace
