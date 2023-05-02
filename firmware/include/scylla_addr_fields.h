//
// Copyright © 2017-2019 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

// This header file is NOT automatically generated


#pragma once

#ifdef __KERNEL__
    #include <linux/types.h>
#else
    #include <stdint.h>
#endif

#if !defined(__cplusplus) || __cplusplus < 201402L
#define CONSTEXPR
#else
#define CONSTEXPR constexpr
#endif

#define REGION_CODE      (0x0 >> 1)
#define REGION_SRAM      (0x2 >> 1)
#define REGION_REGISTERS (0x4 >> 1)
#define REGION_EXT_RAM0  (0x6 >> 1)
#define REGION_EXT_RAM1  (0x8 >> 1)
#define REGION_EXT_DEV0  (0xA >> 1)
#define REGION_EXT_DEV1  (0xC >> 1)
#define REGION_BUS       (0xE >> 1)

#define REGION_SHIFT 29
#define REGION_MASK 0x3

#define BROADCAST_SHIFT 28
#define BROADCAST_MASK 0x1

#define MEM_INDEX_SHIFT 25
#define MEM_INDEX_MASK 0x7

#define CE_SHIFT 20
#define CE_MASK 0x1F

#define REGPAGE_SHIFT 16
#define REGPAGE_MASK 0xF

#define REGOFFSET_SHIFT 0
#define REGOFFSET_MASK 0xFFFF

// Register page offsets
#define SEC_RP            0x0
#define DL1_RP            0x1
#define DL2_RP            0x2
#define DL3_RP            0x3
#define reserved0_RP      0x4
#define DMA_RP            0x5
#define TSU_RP            0x6
#define CE_RP             0x7
#define GLOBAL_RP         0x8
#define STRIPE_RP         0x9
#define BLOCK_RP          0xA
#define reserved1_RP      0xB
#define reserved2_RP      0xC
#define PMU_RP            0xD
#define DBG_RP            0xE
#define reserved3_RP      0xF

// Compose a Scylla register address from the bit components

#define SCYLLA_REG(broadcast, ce, page, offset)             \
	(((REGION_REGISTERS & REGION_MASK) << REGION_SHIFT)       |  \
     (((broadcast) & BROADCAST_MASK) << BROADCAST_SHIFT) |  \
     (((ce) & CE_MASK) << CE_SHIFT)                      |  \
     (((page) & REGPAGE_MASK) << REGPAGE_SHIFT)          |  \
     (((offset) & REGOFFSET_MASK) << REGOFFSET_SHIFT))

#define TOP_REG(page, offset) SCYLLA_REG(1, 0, (page), (offset))

#define CE_REG(ce, page, offset) SCYLLA_REG(0, (ce), (page), (offset))

#define PLE_REG(page, offset) \
    (((REGION_REGISTERS & REGION_MASK) << REGION_SHIFT) | \
     (((page) & REGPAGE_MASK) << REGPAGE_SHIFT) | \
     (((offset) & REGOFFSET_MASK) << REGOFFSET_SHIFT))

// Compose a Scylla SRAM address
struct scylla_sram_addr
{

    union
    {
        uint32_t addr;
        struct
        {
            uint32_t zero1       :  4;
            uint32_t sram_addr   : 15;
            uint32_t sram        :  1;
            uint32_t ce          :  5;
            uint32_t mem_index   :  3;
            uint32_t zero2       :  1;
            uint32_t region      :  3;
        } bits;
    };
#ifdef __cplusplus
    CONSTEXPR scylla_sram_addr(uint32_t init = 0) : addr(init) {}
    CONSTEXPR scylla_sram_addr(uint32_t ce, uint32_t sram, uint32_t mem_index, uint32_t sram_addr) : scylla_sram_addr()
        {
            bits.region      = REGION_SRAM;
            bits.zero2       = 0;
            bits.mem_index   = mem_index & 0x7;
            bits.ce          = ce & 0x1f;
            bits.sram        = sram & 0x1;
            bits.sram_addr   = sram_addr & 0x7fff;
            bits.zero1       = 0;
        }

    CONSTEXPR void set_sram_byte_addr(uint32_t value) { bits.sram_addr = (value >> 4) & 0x7fff; }
    CONSTEXPR void set_sram_addr(uint32_t value) { bits.sram_addr = value & 0x7fff; }
    CONSTEXPR void set_sram     (uint32_t value) { bits.sram      = value & 0x1; };
    CONSTEXPR void set_ce       (uint32_t value) { bits.ce        = value & 0x1f; };
    CONSTEXPR void set_mem_index(uint32_t value) { bits.mem_index = value & 0x7; };
    CONSTEXPR void set_region   (uint32_t value) { bits.region    = value & 0x7; };

    CONSTEXPR uint32_t get_sram_addr()     const { uint32_t value = static_cast<uint32_t>(bits.sram_addr); return value; }
    CONSTEXPR uint32_t get_sram_byte_addr()const { uint32_t value = static_cast<uint32_t>(bits.sram_addr) << 4; return value; }
    CONSTEXPR uint32_t get_sram()          const { uint32_t value = static_cast<uint32_t>(bits.sram);      return value; }
    CONSTEXPR uint32_t get_ce()            const { uint32_t value = static_cast<uint32_t>(bits.ce);        return value; }
    CONSTEXPR uint32_t get_mem_index()     const { uint32_t value = static_cast<uint32_t>(bits.mem_index); return value; }
    CONSTEXPR uint32_t get_region()        const { uint32_t value = static_cast<uint32_t>(bits.region);    return value; }
#endif
};


// Compose a Scylla top-level register address
struct scylla_top_addr
{
    union
    {
        uint32_t addr;
        struct
        {
            uint32_t page_offset : 16;
            uint32_t reg_page   :  4;
            uint32_t ce          :  8;
            uint32_t b           :  1;
            uint32_t region      :  3;
        } bits;
    };
#ifdef __cplusplus
    CONSTEXPR scylla_top_addr(uint32_t init = 0) : addr(init) {}

    // Default to broadcast for most registers
    CONSTEXPR scylla_top_addr(uint32_t reg_page, uint32_t page_offset) : scylla_top_addr()
        {
            bits.region      = REGION_REGISTERS;
            bits.b           = 1;
            bits.ce          = 0;
            bits.reg_page    = reg_page & 0xf;
            bits.page_offset = page_offset & 0xffff;
        }

    // CE-specific
    CONSTEXPR scylla_top_addr(uint32_t ce, uint32_t reg_page, uint32_t page_offset) : scylla_top_addr()
        {
            bits.region      = REGION_REGISTERS;
            bits.b           = 0;
            bits.ce          = ce & 0xff;
            bits.reg_page    = reg_page & 0xf;
            bits.page_offset = page_offset & 0xffff;
        }

    CONSTEXPR scylla_top_addr(uint32_t b, uint32_t ce, uint32_t reg_page, uint32_t page_offset) : scylla_top_addr()
        {
            bits.region      = REGION_REGISTERS;
            bits.b           = b & 0x1;
            bits.ce          = ce & 0xff;
            bits.reg_page    = reg_page & 0xf;
            bits.page_offset = page_offset & 0xffff;
        }

    CONSTEXPR void set_page_offset(uint32_t value) { bits.page_offset = value & 0xffff; }
    CONSTEXPR void set_reg_page   (uint32_t value) { bits.reg_page   = value & 0xf; };
    CONSTEXPR void set_ce         (uint32_t value) { bits.ce          = value & 0xff; };
    CONSTEXPR void set_b          (uint32_t value) { bits.b           = value & 0x1; };
    CONSTEXPR void set_region     (uint32_t value) { bits.region      = value & 0x3; };

    CONSTEXPR uint32_t get_page_offset() const { uint32_t value = static_cast<uint32_t>(bits.page_offset); return value; }
    CONSTEXPR uint32_t get_reg_page() const { uint32_t value = static_cast<uint32_t>(bits.reg_page);   return value; }
    CONSTEXPR uint32_t get_ce() const { uint32_t value = static_cast<uint32_t>(bits.ce);          return value; }
    CONSTEXPR uint32_t get_b() const { uint32_t value = static_cast<uint32_t>(bits.b);           return value; }
    CONSTEXPR uint32_t get_region() const { uint32_t value = static_cast<uint32_t>(bits.region);      return value; }
#endif
};
