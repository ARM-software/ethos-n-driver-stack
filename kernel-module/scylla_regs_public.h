/*
 *
 * (C) COPYRIGHT 2018-2019 Arm Limited. All rights reserved.
 *
 * This program is free software and is provided to you under the terms of the
 * GNU General Public License version 2 as published by the Free Software
 * Foundation, and any use by you of this program is subject to the terms
 * of such GNU licence.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, you can access it online at
 * http://www.gnu.org/licenses/gpl-2.0.html.
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 */


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

#define NPU_ARCH_VERSION_MAJOR 1
#define NPU_ARCH_VERSION_MINOR 4
#define NPU_ARCH_VERSION_PATCH 10
#define NPU_ARCH_BASENAME "SCYLLA"

// Register offsets

//
// Register subpage DL1
//
#define DL1_SYSCTLR0 0x0018
#define DL1_SYSCTLR1 0x001C
#define DL1_PWRCTLR 0x0020
#define DL1_CLRIRQ_EXT 0x0034
#define DL1_SETIRQ_INT 0x0040
#define DL1_IRQ_STATUS 0x00A0
#define DL1_GP0 0x1000
#define DL1_GP1 0x1004
#define DL1_GP2 0x1008
#define DL1_GP3 0x100C
#define DL1_GP4 0x1010
#define DL1_GP5 0x1014
#define DL1_GP6 0x1018
#define DL1_GP7 0x101C
#define DL1_STREAM0_STREAM_SECURITY 0x3000
#define DL1_STREAM0_NSAID 0x3004
#define DL1_STREAM0_MMUSID 0x3008
#define DL1_STREAM0_MMUSSID 0x300C
#define DL1_STREAM0_ATTR_CONTROL 0x3010
#define DL1_STREAM0_MEMATTR 0x3014
#define DL1_STREAM0_ADDRESS_EXTEND 0x3018
#define DL1_NPU_ID 0xF000
#define DL1_UNIT_COUNT 0xF004
#define DL1_MCE_FEATURES 0xF008
#define DL1_DFC_FEATURES 0xF00C
#define DL1_PLE_FEATURES 0xF010
#define DL1_WD_FEATURES 0xF014
#define DL1_VECTOR_ENGINE_FEATURES 0xF018
#define DL1_ECOID 0xF100
#define DL1_STREAMID_WIDTH 0xF104
#define DL1_REGISTERS_SIZE 0xF108


//
// dl1_sysctlr0_r - System control 0 - MCU Control and Status
//
struct dl1_sysctlr0_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t cpuwait : 1; // MCU CPUWAIT input
            uint32_t lockup : 1; // MCU LOCKUP output
            uint32_t halted : 1; // MCU HALTED output
            uint32_t rstreq : 1; // MCU SYSRESETREQ output
            uint32_t sleeping : 1; // MCU SLEEPING and TRCENA output
            uint32_t reserved0 : 2;
            uint32_t initvtor : 22; // MCU Vector Table address
            uint32_t soft_rstreq : 2; // Soft reset request
            uint32_t hard_rstreq : 1; // Hard reset request
        } bits;
    };
#ifdef __cplusplus
    CONSTEXPR dl1_sysctlr0_r(uint32_t init=0) : word(init) {}
    CONSTEXPR uint32_t get_cpuwait() const { uint32_t value = static_cast<uint32_t>(bits.cpuwait); return value;}
    CONSTEXPR void set_cpuwait(uint32_t value) { bits.cpuwait = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_lockup() const { uint32_t value = static_cast<uint32_t>(bits.lockup); return value;}
    CONSTEXPR void set_lockup(uint32_t value) { bits.lockup = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_halted() const { uint32_t value = static_cast<uint32_t>(bits.halted); return value;}
    CONSTEXPR void set_halted(uint32_t value) { bits.halted = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_rstreq() const { uint32_t value = static_cast<uint32_t>(bits.rstreq); return value;}
    CONSTEXPR void set_rstreq(uint32_t value) { bits.rstreq = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_sleeping() const { uint32_t value = static_cast<uint32_t>(bits.sleeping); return value;}
    CONSTEXPR void set_sleeping(uint32_t value) { bits.sleeping = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_initvtor() const { uint32_t value = static_cast<uint32_t>(bits.initvtor); return value;}
    CONSTEXPR void set_initvtor(uint32_t value) { bits.initvtor = ((1u<<22)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR soft_reset_t get_soft_rstreq() const { soft_reset_t value = static_cast<soft_reset_t>(bits.soft_rstreq); return value;}
    CONSTEXPR void set_soft_rstreq(soft_reset_t value) { bits.soft_rstreq = ((1u<<2)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_hard_rstreq() const { uint32_t value = static_cast<uint32_t>(bits.hard_rstreq); return value;}
    CONSTEXPR void set_hard_rstreq(uint32_t value) { bits.hard_rstreq = ((1u<<1)-1) & static_cast<uint32_t>(value); }
#endif
};


//
// dl1_sysctlr1_r - System control 1 - Event Control
//
struct dl1_sysctlr1_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t reserved0 : 4;
            uint32_t mcu_setevnt : 1; // MCU SET event
            uint32_t mcu_setirq : 1; // MCU SET interrupt
            uint32_t mcu_gpevnt : 1; // MCU GP event
            uint32_t reserved1 : 1;
            uint32_t tsu_evnt : 1; // TSU event
            uint32_t tsu_irq : 1; // TSU interrupt
            uint32_t tsu_dbg : 1; // TSU debug request
            uint32_t reserved2 : 5;
            uint32_t txev_ple : 1; // MCU TXEV sent to PLE
            uint32_t reserved3 : 1;
            uint32_t txev_dbg : 1; // MCU TXEV sent to Host
            uint32_t rxev_degroup : 1; // Degroup PLE TXEV sent to MCU
            uint32_t rxev_evnt : 1; // PLE TXEV sent to MCU
            uint32_t rxev_irq : 1; // PLE TXEV triggers MCU interrupt
            uint32_t reserved4 : 2;
            uint32_t pmu_evnt : 1; // PMU counter overflow event
            uint32_t pmu_irq : 1; // PMU counter overflow interrupt
            uint32_t pmu_dbg : 1; // PMU counter overflow debug request
            uint32_t pmu_eng : 1; // PMU engine counter overflow request
            uint32_t err_tolr_evnt : 1; // Tolerable error triggers MCU event
            uint32_t err_tolr_irq : 1; // Tolerable error triggers MCU interrupt
            uint32_t err_func_irq : 1; // Functional error triggers MCU interrupt
            uint32_t err_recv_irq : 1; // Recoverable error triggers MCU interrupt
        } bits;
    };
#ifdef __cplusplus
    CONSTEXPR dl1_sysctlr1_r(uint32_t init=0) : word(init) {}
    CONSTEXPR uint32_t get_mcu_setevnt() const { uint32_t value = static_cast<uint32_t>(bits.mcu_setevnt); return value;}
    CONSTEXPR void set_mcu_setevnt(uint32_t value) { bits.mcu_setevnt = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_mcu_setirq() const { uint32_t value = static_cast<uint32_t>(bits.mcu_setirq); return value;}
    CONSTEXPR void set_mcu_setirq(uint32_t value) { bits.mcu_setirq = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_mcu_gpevnt() const { uint32_t value = static_cast<uint32_t>(bits.mcu_gpevnt); return value;}
    CONSTEXPR void set_mcu_gpevnt(uint32_t value) { bits.mcu_gpevnt = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_tsu_evnt() const { uint32_t value = static_cast<uint32_t>(bits.tsu_evnt); return value;}
    CONSTEXPR void set_tsu_evnt(uint32_t value) { bits.tsu_evnt = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_tsu_irq() const { uint32_t value = static_cast<uint32_t>(bits.tsu_irq); return value;}
    CONSTEXPR void set_tsu_irq(uint32_t value) { bits.tsu_irq = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_tsu_dbg() const { uint32_t value = static_cast<uint32_t>(bits.tsu_dbg); return value;}
    CONSTEXPR void set_tsu_dbg(uint32_t value) { bits.tsu_dbg = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_txev_ple() const { uint32_t value = static_cast<uint32_t>(bits.txev_ple); return value;}
    CONSTEXPR void set_txev_ple(uint32_t value) { bits.txev_ple = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_txev_dbg() const { uint32_t value = static_cast<uint32_t>(bits.txev_dbg); return value;}
    CONSTEXPR void set_txev_dbg(uint32_t value) { bits.txev_dbg = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_rxev_degroup() const { uint32_t value = static_cast<uint32_t>(bits.rxev_degroup); return value;}
    CONSTEXPR void set_rxev_degroup(uint32_t value) { bits.rxev_degroup = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_rxev_evnt() const { uint32_t value = static_cast<uint32_t>(bits.rxev_evnt); return value;}
    CONSTEXPR void set_rxev_evnt(uint32_t value) { bits.rxev_evnt = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_rxev_irq() const { uint32_t value = static_cast<uint32_t>(bits.rxev_irq); return value;}
    CONSTEXPR void set_rxev_irq(uint32_t value) { bits.rxev_irq = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_pmu_evnt() const { uint32_t value = static_cast<uint32_t>(bits.pmu_evnt); return value;}
    CONSTEXPR void set_pmu_evnt(uint32_t value) { bits.pmu_evnt = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_pmu_irq() const { uint32_t value = static_cast<uint32_t>(bits.pmu_irq); return value;}
    CONSTEXPR void set_pmu_irq(uint32_t value) { bits.pmu_irq = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_pmu_dbg() const { uint32_t value = static_cast<uint32_t>(bits.pmu_dbg); return value;}
    CONSTEXPR void set_pmu_dbg(uint32_t value) { bits.pmu_dbg = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_pmu_eng() const { uint32_t value = static_cast<uint32_t>(bits.pmu_eng); return value;}
    CONSTEXPR void set_pmu_eng(uint32_t value) { bits.pmu_eng = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_err_tolr_evnt() const { uint32_t value = static_cast<uint32_t>(bits.err_tolr_evnt); return value;}
    CONSTEXPR void set_err_tolr_evnt(uint32_t value) { bits.err_tolr_evnt = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_err_tolr_irq() const { uint32_t value = static_cast<uint32_t>(bits.err_tolr_irq); return value;}
    CONSTEXPR void set_err_tolr_irq(uint32_t value) { bits.err_tolr_irq = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_err_func_irq() const { uint32_t value = static_cast<uint32_t>(bits.err_func_irq); return value;}
    CONSTEXPR void set_err_func_irq(uint32_t value) { bits.err_func_irq = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_err_recv_irq() const { uint32_t value = static_cast<uint32_t>(bits.err_recv_irq); return value;}
    CONSTEXPR void set_err_recv_irq(uint32_t value) { bits.err_recv_irq = ((1u<<1)-1) & static_cast<uint32_t>(value); }
#endif
};


//
// dl1_pwrctlr_r - Power Control
//
struct dl1_pwrctlr_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t active : 1; // NPU activity state
            uint32_t qreqn : 1; // Value of CLK Q-channel QREQn
            uint32_t reserved0 : 30;
        } bits;
    };
#ifdef __cplusplus
    CONSTEXPR dl1_pwrctlr_r(uint32_t init=0) : word(init) {}
    CONSTEXPR uint32_t get_active() const { uint32_t value = static_cast<uint32_t>(bits.active); return value;}
    CONSTEXPR void set_active(uint32_t value) { bits.active = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_qreqn() const { uint32_t value = static_cast<uint32_t>(bits.qreqn); return value;}
    CONSTEXPR void set_qreqn(uint32_t value) { bits.qreqn = ((1u<<1)-1) & static_cast<uint32_t>(value); }
#endif
};


//
// dl1_clrirq_ext_r - Clear external interrupts (to host)
//
struct dl1_clrirq_ext_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t err : 1; // Host error interrupt clear request
            uint32_t debug : 1; // Host debug interrupt clear request
            uint32_t job : 1; // Host job interrupt clear request
            uint32_t reserved0 : 29;
        } bits;
    };
#ifdef __cplusplus
    CONSTEXPR dl1_clrirq_ext_r(uint32_t init=0) : word(init) {}
    CONSTEXPR uint32_t get_err() const { uint32_t value = static_cast<uint32_t>(bits.err); return value;}
    CONSTEXPR void set_err(uint32_t value) { bits.err = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_debug() const { uint32_t value = static_cast<uint32_t>(bits.debug); return value;}
    CONSTEXPR void set_debug(uint32_t value) { bits.debug = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_job() const { uint32_t value = static_cast<uint32_t>(bits.job); return value;}
    CONSTEXPR void set_job(uint32_t value) { bits.job = ((1u<<1)-1) & static_cast<uint32_t>(value); }
#endif
};


//
// dl1_setirq_int_r - Raise internal interrupts and events
//
struct dl1_setirq_int_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t reserved0 : 4;
            uint32_t event : 1; // MCU event (edge-sensitive to MCU'S RXEV pin)
            uint32_t interrupt : 1; // MCU interrupt (edge-sensitive to MCU's IRQ pin)
            uint32_t reserved1 : 1;
            uint32_t nmi : 1; // MCU interrupt (edge-sensitive to MCU's NMI pin)
            uint32_t reserved2 : 24;
        } bits;
    };
#ifdef __cplusplus
    CONSTEXPR dl1_setirq_int_r(uint32_t init=0) : word(init) {}
    CONSTEXPR uint32_t get_event() const { uint32_t value = static_cast<uint32_t>(bits.event); return value;}
    CONSTEXPR void set_event(uint32_t value) { bits.event = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_interrupt() const { uint32_t value = static_cast<uint32_t>(bits.interrupt); return value;}
    CONSTEXPR void set_interrupt(uint32_t value) { bits.interrupt = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_nmi() const { uint32_t value = static_cast<uint32_t>(bits.nmi); return value;}
    CONSTEXPR void set_nmi(uint32_t value) { bits.nmi = ((1u<<1)-1) & static_cast<uint32_t>(value); }
#endif
};


//
// dl1_irq_status_r - Status register used by the Host system
//
struct dl1_irq_status_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t setirq_err : 1; // ERR interrupt caused by SETIRQ
            uint32_t setirq_dbg : 1; // DEBUG interrupt caused by SETIRQ
            uint32_t setirq_job : 1; // JOB interrupt caused by SETIRQ
            uint32_t reserved0 : 7;
            uint32_t tsu_dbg : 1; // DEBUG interrupt caused by TSU
            uint32_t reserved1 : 15;
            uint32_t pmu_dbg : 1; // DEBUG interrupt caused by top-level PMU
            uint32_t pmu_eng : 1; // DEBUG interrupt caused by engine-level PMU
            uint32_t tol_err : 1; // Tolerable error
            uint32_t func_err : 1; // Functional error
            uint32_t rec_err : 1; // Recoverable error
            uint32_t unrec_err : 1; // Unrecoverable error
        } bits;
    };
#ifdef __cplusplus
    CONSTEXPR dl1_irq_status_r(uint32_t init=0) : word(init) {}
    CONSTEXPR uint32_t get_setirq_err() const { uint32_t value = static_cast<uint32_t>(bits.setirq_err); return value;}
    CONSTEXPR void set_setirq_err(uint32_t value) { bits.setirq_err = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_setirq_dbg() const { uint32_t value = static_cast<uint32_t>(bits.setirq_dbg); return value;}
    CONSTEXPR void set_setirq_dbg(uint32_t value) { bits.setirq_dbg = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_setirq_job() const { uint32_t value = static_cast<uint32_t>(bits.setirq_job); return value;}
    CONSTEXPR void set_setirq_job(uint32_t value) { bits.setirq_job = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_tsu_dbg() const { uint32_t value = static_cast<uint32_t>(bits.tsu_dbg); return value;}
    CONSTEXPR void set_tsu_dbg(uint32_t value) { bits.tsu_dbg = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_pmu_dbg() const { uint32_t value = static_cast<uint32_t>(bits.pmu_dbg); return value;}
    CONSTEXPR void set_pmu_dbg(uint32_t value) { bits.pmu_dbg = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_pmu_eng() const { uint32_t value = static_cast<uint32_t>(bits.pmu_eng); return value;}
    CONSTEXPR void set_pmu_eng(uint32_t value) { bits.pmu_eng = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_tol_err() const { uint32_t value = static_cast<uint32_t>(bits.tol_err); return value;}
    CONSTEXPR void set_tol_err(uint32_t value) { bits.tol_err = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_func_err() const { uint32_t value = static_cast<uint32_t>(bits.func_err); return value;}
    CONSTEXPR void set_func_err(uint32_t value) { bits.func_err = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_rec_err() const { uint32_t value = static_cast<uint32_t>(bits.rec_err); return value;}
    CONSTEXPR void set_rec_err(uint32_t value) { bits.rec_err = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_unrec_err() const { uint32_t value = static_cast<uint32_t>(bits.unrec_err); return value;}
    CONSTEXPR void set_unrec_err(uint32_t value) { bits.unrec_err = ((1u<<1)-1) & static_cast<uint32_t>(value); }
#endif
};


//
// dl1_gp0_r - General purpose register 0
//
struct dl1_gp0_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t gp0 : 32; // General purpose register 0
        } bits;
    };
#ifdef __cplusplus
    CONSTEXPR dl1_gp0_r(uint32_t init=0) : word(init) {}
    CONSTEXPR uint32_t get_gp0() const { uint32_t value = static_cast<uint32_t>(bits.gp0); return value;}
    CONSTEXPR void set_gp0(uint32_t value) { bits.gp0 = static_cast<uint32_t>(value); }
#endif
};


//
// dl1_gp1_r - General purpose register 1
//
struct dl1_gp1_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t gp1 : 32; // General purpose register 1
        } bits;
    };
#ifdef __cplusplus
    CONSTEXPR dl1_gp1_r(uint32_t init=0) : word(init) {}
    CONSTEXPR uint32_t get_gp1() const { uint32_t value = static_cast<uint32_t>(bits.gp1); return value;}
    CONSTEXPR void set_gp1(uint32_t value) { bits.gp1 = static_cast<uint32_t>(value); }
#endif
};


//
// dl1_gp2_r - General purpose register 2
//
struct dl1_gp2_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t gp2 : 32; // General purpose register 2
        } bits;
    };
#ifdef __cplusplus
    CONSTEXPR dl1_gp2_r(uint32_t init=0) : word(init) {}
    CONSTEXPR uint32_t get_gp2() const { uint32_t value = static_cast<uint32_t>(bits.gp2); return value;}
    CONSTEXPR void set_gp2(uint32_t value) { bits.gp2 = static_cast<uint32_t>(value); }
#endif
};


//
// dl1_gp3_r - General purpose register 3
//
struct dl1_gp3_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t gp3 : 32; // General purpose register 3
        } bits;
    };
#ifdef __cplusplus
    CONSTEXPR dl1_gp3_r(uint32_t init=0) : word(init) {}
    CONSTEXPR uint32_t get_gp3() const { uint32_t value = static_cast<uint32_t>(bits.gp3); return value;}
    CONSTEXPR void set_gp3(uint32_t value) { bits.gp3 = static_cast<uint32_t>(value); }
#endif
};


//
// dl1_gp4_r - General purpose register 4
//
struct dl1_gp4_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t gp4 : 32; // General purpose register 4
        } bits;
    };
#ifdef __cplusplus
    CONSTEXPR dl1_gp4_r(uint32_t init=0) : word(init) {}
    CONSTEXPR uint32_t get_gp4() const { uint32_t value = static_cast<uint32_t>(bits.gp4); return value;}
    CONSTEXPR void set_gp4(uint32_t value) { bits.gp4 = static_cast<uint32_t>(value); }
#endif
};


//
// dl1_gp5_r - General purpose register 5
//
struct dl1_gp5_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t gp5 : 32; // General purpose register 5
        } bits;
    };
#ifdef __cplusplus
    CONSTEXPR dl1_gp5_r(uint32_t init=0) : word(init) {}
    CONSTEXPR uint32_t get_gp5() const { uint32_t value = static_cast<uint32_t>(bits.gp5); return value;}
    CONSTEXPR void set_gp5(uint32_t value) { bits.gp5 = static_cast<uint32_t>(value); }
#endif
};


//
// dl1_gp6_r - General purpose register 6
//
struct dl1_gp6_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t gp6 : 32; // General purpose register 6
        } bits;
    };
#ifdef __cplusplus
    CONSTEXPR dl1_gp6_r(uint32_t init=0) : word(init) {}
    CONSTEXPR uint32_t get_gp6() const { uint32_t value = static_cast<uint32_t>(bits.gp6); return value;}
    CONSTEXPR void set_gp6(uint32_t value) { bits.gp6 = static_cast<uint32_t>(value); }
#endif
};


//
// dl1_gp7_r - General purpose register 7
//
struct dl1_gp7_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t gp7 : 32; // General purpose register 7
        } bits;
    };
#ifdef __cplusplus
    CONSTEXPR dl1_gp7_r(uint32_t init=0) : word(init) {}
    CONSTEXPR uint32_t get_gp7() const { uint32_t value = static_cast<uint32_t>(bits.gp7); return value;}
    CONSTEXPR void set_gp7(uint32_t value) { bits.gp7 = static_cast<uint32_t>(value); }
#endif
};


//
// dl1_stream0_stream_security_r - Stream 0 - Security State
//
struct dl1_stream0_stream_security_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t mmusecsid : 1; // MMU stream security state
            uint32_t protns : 1; // AXI stream security state
            uint32_t reserved0 : 30;
        } bits;
    };
#ifdef __cplusplus
    CONSTEXPR dl1_stream0_stream_security_r(uint32_t init=0) : word(init) {}
    CONSTEXPR uint32_t get_mmusecsid() const { uint32_t value = static_cast<uint32_t>(bits.mmusecsid); return value;}
    CONSTEXPR void set_mmusecsid(uint32_t value) { bits.mmusecsid = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_protns() const { uint32_t value = static_cast<uint32_t>(bits.protns); return value;}
    CONSTEXPR void set_protns(uint32_t value) { bits.protns = ((1u<<1)-1) & static_cast<uint32_t>(value); }
#endif
};


//
// dl1_stream0_nsaid_r - Stream 0 - Non-secure Access Identifier
//
struct dl1_stream0_nsaid_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t nsaid : 4; // Non-Secure Address Identifier
            uint32_t reserved0 : 28;
        } bits;
    };
#ifdef __cplusplus
    CONSTEXPR dl1_stream0_nsaid_r(uint32_t init=0) : word(init) {}
    CONSTEXPR uint32_t get_nsaid() const { uint32_t value = static_cast<uint32_t>(bits.nsaid); return value;}
    CONSTEXPR void set_nsaid(uint32_t value) { bits.nsaid = ((1u<<4)-1) & static_cast<uint32_t>(value); }
#endif
};


//
// dl1_stream0_mmusid_r - Stream 0 - MMU Stream Identifier
//
struct dl1_stream0_mmusid_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t mmusid : 32; // MMU Stream ID (actual width is implementation defined)
        } bits;
    };
#ifdef __cplusplus
    CONSTEXPR dl1_stream0_mmusid_r(uint32_t init=0) : word(init) {}
    CONSTEXPR uint32_t get_mmusid() const { uint32_t value = static_cast<uint32_t>(bits.mmusid); return value;}
    CONSTEXPR void set_mmusid(uint32_t value) { bits.mmusid = static_cast<uint32_t>(value); }
#endif
};


//
// dl1_stream0_mmussid_r - Stream 0 - MMU Sub-stream Stream Identifier
//
struct dl1_stream0_mmussid_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t mmussid : 20; // MMU Sub-Stream ID (actual width is implementation defined)
            uint32_t reserved0 : 11;
            uint32_t mmussidv : 1; // MMUSSID valid bit
        } bits;
    };
#ifdef __cplusplus
    CONSTEXPR dl1_stream0_mmussid_r(uint32_t init=0) : word(init) {}
    CONSTEXPR uint32_t get_mmussid() const { uint32_t value = static_cast<uint32_t>(bits.mmussid); return value;}
    CONSTEXPR void set_mmussid(uint32_t value) { bits.mmussid = ((1u<<20)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_mmussidv() const { uint32_t value = static_cast<uint32_t>(bits.mmussidv); return value;}
    CONSTEXPR void set_mmussidv(uint32_t value) { bits.mmussidv = ((1u<<1)-1) & static_cast<uint32_t>(value); }
#endif
};


//
// dl1_stream0_attr_control_r - Stream 0 - Attribute Control
//
struct dl1_stream0_attr_control_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t attrlocked : 1; // Stream attributes locked
            uint32_t reserved0 : 31;
        } bits;
    };
#ifdef __cplusplus
    CONSTEXPR dl1_stream0_attr_control_r(uint32_t init=0) : word(init) {}
    CONSTEXPR uint32_t get_attrlocked() const { uint32_t value = static_cast<uint32_t>(bits.attrlocked); return value;}
    CONSTEXPR void set_attrlocked(uint32_t value) { bits.attrlocked = ((1u<<1)-1) & static_cast<uint32_t>(value); }
#endif
};


//
// dl1_stream0_memattr_r - Stream 0 - Memory Attributes
//
struct dl1_stream0_memattr_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t memattr : 4; // Memory attributes
            uint32_t reserved0 : 28;
        } bits;
    };
#ifdef __cplusplus
    CONSTEXPR dl1_stream0_memattr_r(uint32_t init=0) : word(init) {}
    CONSTEXPR memory_attributes_t get_memattr() const { memory_attributes_t value = static_cast<memory_attributes_t>(bits.memattr); return value;}
    CONSTEXPR void set_memattr(memory_attributes_t value) { bits.memattr = ((1u<<4)-1) & static_cast<uint32_t>(value); }
#endif
};


//
// dl1_stream0_address_extend_r - Stream 0 - Extended address bits per stream
//
struct dl1_stream0_address_extend_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t reserved0 : 9;
            uint32_t addrextend : 20; // Address extension bits [48:29]
            uint32_t reserved1 : 3;
        } bits;
    };
#ifdef __cplusplus
    CONSTEXPR dl1_stream0_address_extend_r(uint32_t init=0) : word(init) {}
    CONSTEXPR uint32_t get_addrextend() const { uint32_t value = static_cast<uint32_t>(bits.addrextend); return value;}
    CONSTEXPR void set_addrextend(uint32_t value) { bits.addrextend = ((1u<<20)-1) & static_cast<uint32_t>(value); }
#endif
};


//
// dl1_npu_id_r - NPU ID register
//
struct dl1_npu_id_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t version_status : 4; // Status of the NPU release
            uint32_t version_minor : 4; // Minor release version number
            uint32_t version_major : 4; // Major release version number
            uint32_t product_major : 4; // Product identifier
            uint32_t arch_rev : 8; // Architecture patch revision
            uint32_t arch_minor : 4; // Architecture minor revision
            uint32_t arch_major : 4; // Architecture major revision
        } bits;
    };
#ifdef __cplusplus
    CONSTEXPR dl1_npu_id_r(uint32_t init=0) : word(init) {}
    CONSTEXPR npu_version_status_t get_version_status() const { npu_version_status_t value = static_cast<npu_version_status_t>(bits.version_status); return value;}
    CONSTEXPR void set_version_status(npu_version_status_t value) { bits.version_status = ((1u<<4)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_version_minor() const { uint32_t value = static_cast<uint32_t>(bits.version_minor); return value;}
    CONSTEXPR void set_version_minor(uint32_t value) { bits.version_minor = ((1u<<4)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_version_major() const { uint32_t value = static_cast<uint32_t>(bits.version_major); return value;}
    CONSTEXPR void set_version_major(uint32_t value) { bits.version_major = ((1u<<4)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_product_major() const { uint32_t value = static_cast<uint32_t>(bits.product_major); return value;}
    CONSTEXPR void set_product_major(uint32_t value) { bits.product_major = ((1u<<4)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_arch_rev() const { uint32_t value = static_cast<uint32_t>(bits.arch_rev); return value;}
    CONSTEXPR void set_arch_rev(uint32_t value) { bits.arch_rev = ((1u<<8)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_arch_minor() const { uint32_t value = static_cast<uint32_t>(bits.arch_minor); return value;}
    CONSTEXPR void set_arch_minor(uint32_t value) { bits.arch_minor = ((1u<<4)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_arch_major() const { uint32_t value = static_cast<uint32_t>(bits.arch_major); return value;}
    CONSTEXPR void set_arch_major(uint32_t value) { bits.arch_major = ((1u<<4)-1) & static_cast<uint32_t>(value); }
#endif
};


//
// dl1_unit_count_r - Units present count
//
struct dl1_unit_count_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t quad_count : 8; // Number of quads
            uint32_t engines_per_quad : 8; // Number of engines per quad
            uint32_t dfc_emc_per_engine : 4; // Number of memory controllers per engine
            uint32_t reserved0 : 12;
        } bits;
    };
#ifdef __cplusplus
    CONSTEXPR dl1_unit_count_r(uint32_t init=0) : word(init) {}
    CONSTEXPR uint32_t get_quad_count() const { uint32_t value = static_cast<uint32_t>(bits.quad_count); return value;}
    CONSTEXPR void set_quad_count(uint32_t value) { bits.quad_count = ((1u<<8)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_engines_per_quad() const { uint32_t value = static_cast<uint32_t>(bits.engines_per_quad); return value;}
    CONSTEXPR void set_engines_per_quad(uint32_t value) { bits.engines_per_quad = ((1u<<8)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_dfc_emc_per_engine() const { uint32_t value = static_cast<uint32_t>(bits.dfc_emc_per_engine); return value;}
    CONSTEXPR void set_dfc_emc_per_engine(uint32_t value) { bits.dfc_emc_per_engine = ((1u<<4)-1) & static_cast<uint32_t>(value); }
#endif
};


//
// dl1_mce_features_r - MCE features
//
struct dl1_mce_features_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t ifm_generated_per_engine : 4; // IFMs sent to broadcast network per engine
            uint32_t reserved0 : 4;
            uint32_t ofm_generated_per_engine : 4; // OFMs generated per Engine
            uint32_t mce_num_macs : 8; // Number of MAC units per MCE
            uint32_t mce_num_acc : 8; // Number of accumulators per MAC unit
            uint32_t winograd_support : 1; // Winograd funcationality present
            uint32_t tsu_16bit_sequence_support : 1; // TSU support for automatically sequencing 16 bit IFM and weights
            uint32_t ofm_scaling_16bit_support : 1; // Hardware support for scaling results from 16-bit operations
            uint32_t reserved1 : 1;
        } bits;
    };
#ifdef __cplusplus
    CONSTEXPR dl1_mce_features_r(uint32_t init=0) : word(init) {}
    CONSTEXPR uint32_t get_ifm_generated_per_engine() const { uint32_t value = static_cast<uint32_t>(bits.ifm_generated_per_engine); return value;}
    CONSTEXPR void set_ifm_generated_per_engine(uint32_t value) { bits.ifm_generated_per_engine = ((1u<<4)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_ofm_generated_per_engine() const { uint32_t value = static_cast<uint32_t>(bits.ofm_generated_per_engine); return value;}
    CONSTEXPR void set_ofm_generated_per_engine(uint32_t value) { bits.ofm_generated_per_engine = ((1u<<4)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_mce_num_macs() const { uint32_t value = static_cast<uint32_t>(bits.mce_num_macs); return value;}
    CONSTEXPR void set_mce_num_macs(uint32_t value) { bits.mce_num_macs = ((1u<<8)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_mce_num_acc() const { uint32_t value = static_cast<uint32_t>(bits.mce_num_acc); return value;}
    CONSTEXPR void set_mce_num_acc(uint32_t value) { bits.mce_num_acc = ((1u<<8)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_winograd_support() const { uint32_t value = static_cast<uint32_t>(bits.winograd_support); return value;}
    CONSTEXPR void set_winograd_support(uint32_t value) { bits.winograd_support = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_tsu_16bit_sequence_support() const { uint32_t value = static_cast<uint32_t>(bits.tsu_16bit_sequence_support); return value;}
    CONSTEXPR void set_tsu_16bit_sequence_support(uint32_t value) { bits.tsu_16bit_sequence_support = ((1u<<1)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_ofm_scaling_16bit_support() const { uint32_t value = static_cast<uint32_t>(bits.ofm_scaling_16bit_support); return value;}
    CONSTEXPR void set_ofm_scaling_16bit_support(uint32_t value) { bits.ofm_scaling_16bit_support = ((1u<<1)-1) & static_cast<uint32_t>(value); }
#endif
};


//
// dl1_dfc_features_r - DFC features
//
struct dl1_dfc_features_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t dfc_mem_size_per_emc : 16; // DFC memory size per EMC
            uint32_t bank_count : 6; // Number of banks in DFC memory
            uint32_t activation_compression : 4; // Version of activation compression supported
            uint32_t reserved0 : 6;
        } bits;
    };
#ifdef __cplusplus
    CONSTEXPR dl1_dfc_features_r(uint32_t init=0) : word(init) {}
    CONSTEXPR uint32_t get_dfc_mem_size_per_emc() const { uint32_t value = static_cast<uint32_t>(bits.dfc_mem_size_per_emc); return (value << 12);}
    CONSTEXPR void set_dfc_mem_size_per_emc(uint32_t value) { bits.dfc_mem_size_per_emc = ((1u<<16)-1) & static_cast<uint32_t>((value >> 12)); }
    CONSTEXPR uint32_t get_bank_count() const { uint32_t value = static_cast<uint32_t>(bits.bank_count); return value;}
    CONSTEXPR void set_bank_count(uint32_t value) { bits.bank_count = ((1u<<6)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_activation_compression() const { uint32_t value = static_cast<uint32_t>(bits.activation_compression); return value;}
    CONSTEXPR void set_activation_compression(uint32_t value) { bits.activation_compression = ((1u<<4)-1) & static_cast<uint32_t>(value); }
#endif
};


//
// dl1_ple_features_r - PLE features
//
struct dl1_ple_features_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t ple_input_mem_size : 8; // PLE input memory size
            uint32_t ple_output_mem_size : 8; // PLE output memory size
            uint32_t ple_vrf_mem_size : 8; // PLE vector register file memory size
            uint32_t ple_mem_size : 8; // PLE base memory size
        } bits;
    };
#ifdef __cplusplus
    CONSTEXPR dl1_ple_features_r(uint32_t init=0) : word(init) {}
    CONSTEXPR uint32_t get_ple_input_mem_size() const { uint32_t value = static_cast<uint32_t>(bits.ple_input_mem_size); return (value << 8);}
    CONSTEXPR void set_ple_input_mem_size(uint32_t value) { bits.ple_input_mem_size = ((1u<<8)-1) & static_cast<uint32_t>((value >> 8)); }
    CONSTEXPR uint32_t get_ple_output_mem_size() const { uint32_t value = static_cast<uint32_t>(bits.ple_output_mem_size); return (value << 8);}
    CONSTEXPR void set_ple_output_mem_size(uint32_t value) { bits.ple_output_mem_size = ((1u<<8)-1) & static_cast<uint32_t>((value >> 8)); }
    CONSTEXPR uint32_t get_ple_vrf_mem_size() const { uint32_t value = static_cast<uint32_t>(bits.ple_vrf_mem_size); return (value << 4);}
    CONSTEXPR void set_ple_vrf_mem_size(uint32_t value) { bits.ple_vrf_mem_size = ((1u<<8)-1) & static_cast<uint32_t>((value >> 4)); }
    CONSTEXPR uint32_t get_ple_mem_size() const { uint32_t value = static_cast<uint32_t>(bits.ple_mem_size); return (value << 8);}
    CONSTEXPR void set_ple_mem_size(uint32_t value) { bits.ple_mem_size = ((1u<<8)-1) & static_cast<uint32_t>((value >> 8)); }
#endif
};


//
// dl1_wd_features_r - Weight Decoder features
//
struct dl1_wd_features_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t buffer_size : 8; // Weight decoder buffer size
            uint32_t max_dim : 8; // Weight decoder max dimension
            uint32_t compression_version : 4; // Version of weight compression implemented
            uint32_t reserved0 : 12;
        } bits;
    };
#ifdef __cplusplus
    CONSTEXPR dl1_wd_features_r(uint32_t init=0) : word(init) {}
    CONSTEXPR uint32_t get_buffer_size() const { uint32_t value = static_cast<uint32_t>(bits.buffer_size); return value;}
    CONSTEXPR void set_buffer_size(uint32_t value) { bits.buffer_size = ((1u<<8)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_max_dim() const { uint32_t value = static_cast<uint32_t>(bits.max_dim); return value;}
    CONSTEXPR void set_max_dim(uint32_t value) { bits.max_dim = ((1u<<8)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_compression_version() const { uint32_t value = static_cast<uint32_t>(bits.compression_version); return value;}
    CONSTEXPR void set_compression_version(uint32_t value) { bits.compression_version = ((1u<<4)-1) & static_cast<uint32_t>(value); }
#endif
};


//
// dl1_vector_engine_features_r - PLE VE features
//
struct dl1_vector_engine_features_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t vector_engine_version : 4; // Version of the vector engine implemented
            uint32_t ple_lanes : 2; // Number of lanes in the PLE
            uint32_t reserved0 : 26;
        } bits;
    };
#ifdef __cplusplus
    CONSTEXPR dl1_vector_engine_features_r(uint32_t init=0) : word(init) {}
    CONSTEXPR uint32_t get_vector_engine_version() const { uint32_t value = static_cast<uint32_t>(bits.vector_engine_version); return value;}
    CONSTEXPR void set_vector_engine_version(uint32_t value) { bits.vector_engine_version = ((1u<<4)-1) & static_cast<uint32_t>(value); }
    CONSTEXPR uint32_t get_ple_lanes() const { uint32_t value = static_cast<uint32_t>(bits.ple_lanes); return (value + 1);}
    CONSTEXPR void set_ple_lanes(uint32_t value) { bits.ple_lanes = ((1u<<2)-1) & static_cast<uint32_t>((value - 1)); }
#endif
};


//
// dl1_ecoid_r - Encoding describing ECOs implemented
//
struct dl1_ecoid_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t ecoid : 12; // Field for describing ECOs implemented
            uint32_t reserved0 : 20;
        } bits;
    };
#ifdef __cplusplus
    CONSTEXPR dl1_ecoid_r(uint32_t init=0) : word(init) {}
    CONSTEXPR uint32_t get_ecoid() const { uint32_t value = static_cast<uint32_t>(bits.ecoid); return value;}
    CONSTEXPR void set_ecoid(uint32_t value) { bits.ecoid = ((1u<<12)-1) & static_cast<uint32_t>(value); }
#endif
};


//
// dl1_streamid_width_r - Configured StreamID widths
//
struct dl1_streamid_width_r
{
    union
    {
        uint32_t word;
        struct
        {
            uint32_t mmusid_w : 5; // Configured width of StreamID (AxMMUSID)
            uint32_t reserved0 : 3;
            uint32_t mmussid_w : 5; // Configured width of SubstreamID (AxMMUSSID)
            uint32_t reserved1 : 19;
        } bits;
    };
#ifdef __cplusplus
    CONSTEXPR dl1_streamid_width_r(uint32_t init=0) : word(init) {}
    CONSTEXPR uint32_t get_mmusid_w() const { uint32_t value = static_cast<uint32_t>(bits.mmusid_w); return (value + 1);}
    CONSTEXPR void set_mmusid_w(uint32_t value) { bits.mmusid_w = ((1u<<5)-1) & static_cast<uint32_t>((value - 1)); }
    CONSTEXPR uint32_t get_mmussid_w() const { uint32_t value = static_cast<uint32_t>(bits.mmussid_w); return (value + 1);}
    CONSTEXPR void set_mmussid_w(uint32_t value) { bits.mmussid_w = ((1u<<5)-1) & static_cast<uint32_t>((value - 1)); }
#endif
};

