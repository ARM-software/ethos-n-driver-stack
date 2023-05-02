//
// Copyright © 2020-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "SignedSupport.hpp"
#include "lsu.h"

#include <generated/cdp_opcodes.h>
#include <generated/mcr_opcodes.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <tuple>

namespace
{
namespace cexec
{
namespace internals
{
enum class MicroOpType
{
    READ,
    WRITE,
};

enum class MicroOpTarget : unsigned
{
    MCU_PIPELINE,

    VE_MUL_PIPELINE,
    VE_SHIFT_PIPELINE,

    UDMA_PARAMS,

    VE_ACC,
    VE_FLAGS,

    SWZ_REG_SEL_0,
    SWZ_REG_SEL_1,
    SWZ_REG_SEL_2,
    SWZ_REG_SEL_3,
    SWZ_REG_SEL_4,
    SWZ_REG_SEL_5,
    SWZ_REG_SEL_6,
    SWZ_REG_SEL_7,
    SWZ_REG_SEL_8,
    SWZ_REG_SEL_9,
    SWZ_REG_SEL_10,
    SWZ_REG_SEL_11,
    SWZ_REG_SEL_12,
    SWZ_REG_SEL_13,
    SWZ_REG_SEL_14,
    SWZ_REG_SEL_15,

    SWZ_SUBREG_SEL_0,
    SWZ_SUBREG_SEL_1,
    SWZ_SUBREG_SEL_2,
    SWZ_SUBREG_SEL_3,
    SWZ_SUBREG_SEL_4,
    SWZ_SUBREG_SEL_5,
    SWZ_SUBREG_SEL_6,
    SWZ_SUBREG_SEL_7,
    SWZ_SUBREG_SEL_8,
    SWZ_SUBREG_SEL_9,
    SWZ_SUBREG_SEL_10,
    SWZ_SUBREG_SEL_11,
    SWZ_SUBREG_SEL_12,
    SWZ_SUBREG_SEL_13,
    SWZ_SUBREG_SEL_14,
    SWZ_SUBREG_SEL_15,

    RF_REG_0,
    RF_REG_1,
    RF_REG_2,
    RF_REG_3,
    RF_REG_4,
    RF_REG_5,
    RF_REG_6,
    RF_REG_7,
    RF_REG_8,
    RF_REG_9,
    RF_REG_10,
    RF_REG_11,
    RF_REG_12,
    RF_REG_13,
    RF_REG_14,
    RF_REG_15,
    RF_REG_16,
    RF_REG_17,
    RF_REG_18,
    RF_REG_19,
    RF_REG_20,
    RF_REG_21,
    RF_REG_22,
    RF_REG_23,
};

struct MicroOp
{
    unsigned cycle;
    MicroOpType type;
    MicroOpTarget target;
};

constexpr MicroOpTarget RfRegTarget(const unsigned reg)
{
    return static_cast<MicroOpTarget>(static_cast<unsigned>(MicroOpTarget::RF_REG_0) + reg);
}

constexpr MicroOpTarget SwzRegSelTarget(const unsigned sel)
{
    return static_cast<MicroOpTarget>(static_cast<unsigned>(MicroOpTarget::SWZ_REG_SEL_0) + sel);
}

constexpr MicroOpTarget SwzSubregSel(const unsigned sel)
{
    return static_cast<MicroOpTarget>(static_cast<unsigned>(MicroOpTarget::SWZ_SUBREG_SEL_0) + sel);
}

constexpr bool IsRfRegTarget(const MicroOpTarget target)
{
    return (static_cast<unsigned>(target) >= static_cast<unsigned>(MicroOpTarget::RF_REG_0)) &&
           (static_cast<unsigned>(target) <= static_cast<unsigned>(MicroOpTarget::RF_REG_23));
}

constexpr bool MayConflict(const MicroOp& uOp1, const MicroOp& uOp2)
{
    return (uOp1.target == uOp2.target) || (IsRfRegTarget(uOp1.target) && IsRfRegTarget(uOp2.target));
}

constexpr bool HasConflict(const MicroOp& uOp1, const unsigned cycle1, const MicroOp& uOp2, const unsigned cycle2)
{
    bool hasConflict = false;

    if (MayConflict(uOp1, uOp2))
    {
        const unsigned t1 = cycle1 + uOp1.cycle;
        const unsigned t2 = cycle2 + uOp2.cycle;

        if (uOp1.type == uOp2.type)
        {
            // Same op on same resource
            hasConflict = t1 == t2;
        }
        else if (uOp1.target == uOp2.target)
        {
            if (uOp1.type == MicroOpType::WRITE)
            {
                // Read after write
                // uOp2 wants to read after uOp1 writes
                hasConflict = t1 >= t2;
            }
            else
            {
                // Write after read
                // uOp1 wants to read before uOp2 writes
                hasConflict = t1 >= t2;
            }
        }
    }

    return hasConflict;
}

constexpr bool HasConflict(const std::span<const MicroOp>& op1,
                           const unsigned cycle1,
                           const std::span<const MicroOp>& op2,
                           const unsigned cycle2)
{
    for (const MicroOp& uOp1 : op1)
    {
        for (const MicroOp& uOp2 : op2)
        {
            if (HasConflict(uOp1, cycle1, uOp2, cycle2))
            {
                return true;
            }
        }
    }

    return false;
}

template <unsigned N>
constexpr auto ResolveDependencies(const std::span<const MicroOp> (&ops)[N])
{
    std::array<unsigned, N> opToCycle{};
    unsigned maxCycle = 0;

    for (unsigned op = 1; op < N; ++op)
    {
        for (unsigned cycle = 0;; ++cycle)
        {
            bool hasConflict = false;
            for (unsigned i = 0; !hasConflict && (i < op); ++i)
            {
                hasConflict |= HasConflict(ops[i], opToCycle[i], ops[op], cycle);
            }
            if (!hasConflict)
            {
                opToCycle[op] = cycle;
                if (cycle > maxCycle)
                {
                    maxCycle = cycle;
                }
                break;
            }
        }
    }

    return std::pair{ opToCycle, maxCycle + 1 };
}

template <unsigned Cycle, unsigned Op, typename OpsTuple, typename OpToCycle>
__inline_always bool Exec(const OpsTuple& ops, const OpToCycle& opToCycle)
{
    if (Cycle == opToCycle[Op])
    {
        std::get<Op>(ops)();
        return true;
    }
    return false;
}

template <unsigned Cycle, typename OpsTuple, typename OpToCycle, unsigned... Ops>
__inline_always void Exec(const OpsTuple& ops, const OpToCycle& opToCycle, std::index_sequence<Ops...>)
{
    const bool found = (... || Exec<Cycle, Ops>(ops, opToCycle));
    if (!found)
    {
        nop<>();
    }
}

template <typename Op, typename = decltype(std::declval<Op>().m_Rt)>
__inline_always void LoadRt(const Op& op)
{
    __ASM volatile("" ::"r"(op.m_Rt));
}

__inline_always void LoadRt(...)
{}

template <typename Op, typename = decltype(std::declval<Op>().m_Rt2)>
__inline_always void LoadRt2(const Op& op)
{
    __ASM volatile("" ::"r"(op.m_Rt2));
}

__inline_always void LoadRt2(...)
{}

template <typename... Ops, unsigned... Is>
__inline_always void LoadRts(const std::tuple<Ops...>& ops, std::index_sequence<Is...>)
{
    (..., LoadRt(std::get<Is>(ops)));
    (..., LoadRt2(std::get<Is>(ops)));
}

template <typename OpToCycle, typename... Ops, unsigned... Cycles>
__inline_always void Exec(const std::tuple<Ops...>& ops, const OpToCycle& opToCycle, std::index_sequence<Cycles...>)
{
    LoadRts(ops, std::index_sequence_for<Ops...>{});
    (..., Exec<Cycles>(ops, opToCycle, std::index_sequence_for<Ops...>{}));
}
}    // namespace internals

template <typename... Ops>
__inline_always void Exec(const std::tuple<Ops...>& ops)
{
    constexpr std::span<const internals::MicroOp> microOps[] = { Ops::ms_MicroOps... };

    constexpr auto opToCycleAndNumCycles = ResolveDependencies(microOps);

    constexpr auto opToCycle = opToCycleAndNumCycles.first;
    constexpr auto numCycles = opToCycleAndNumCycles.second;

    internals::Exec(ops, opToCycle, std::make_index_sequence<numCycles>{});
}

template <typename... Ops>
__inline_always void UncheckedExec(const std::tuple<Ops...>& ops)
{
    std::apply([](auto&&... x) { (..., x()); }, ops);
}

template <unsigned Dst, unsigned Src>
struct Mov8
{
    static constexpr internals::MicroOp ms_MicroOps[] = {
        { .cycle = 0, .type = internals::MicroOpType::WRITE, .target = internals::MicroOpTarget::MCU_PIPELINE },
        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src) },
        { .cycle = 4, .type = internals::MicroOpType::WRITE, .target = internals::RfRegTarget(Dst) },
    };

    void operator()() const
    {
        ve_mov_8<Dst, Src>();
    }
};

template <unsigned Dst, unsigned Src0, unsigned Src1>
struct Xor8
{
    static constexpr internals::MicroOp ms_MicroOps[] = {
        { .cycle = 0, .type = internals::MicroOpType::WRITE, .target = internals::MicroOpTarget::MCU_PIPELINE },
        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src0) },
        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src1) },
        { .cycle = 4, .type = internals::MicroOpType::WRITE, .target = internals::RfRegTarget(Dst) },
    };

    void operator()() const
    {
        ve_xor_8<Dst, Src0, Src1>();
    }
};

template <unsigned Dst, unsigned Src0, unsigned Src1>
struct SMax16
{
    static constexpr internals::MicroOp ms_MicroOps[] = {
        { .cycle = 0, .type = internals::MicroOpType::WRITE, .target = internals::MicroOpTarget::MCU_PIPELINE },

        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src0) },
        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src0 + 1) },
        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src1) },
        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src1 + 1) },

        { .cycle = 4, .type = internals::MicroOpType::WRITE, .target = internals::RfRegTarget(Dst) },
        { .cycle = 4, .type = internals::MicroOpType::WRITE, .target = internals::RfRegTarget(Dst + 1) },
    };

    void operator()() const
    {
        ve_smax_16<Dst, Src0, Src1>();
    }
};

template <unsigned Dst, unsigned Src0, unsigned Src1>
struct Add16
{
    static constexpr internals::MicroOp ms_MicroOps[] = {
        { .cycle = 0, .type = internals::MicroOpType::WRITE, .target = internals::MicroOpTarget::MCU_PIPELINE },

        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src0) },
        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src0 + 1) },
        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src1) },
        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src1 + 1) },

        { .cycle = 4, .type = internals::MicroOpType::WRITE, .target = internals::RfRegTarget(Dst) },
        { .cycle = 4, .type = internals::MicroOpType::WRITE, .target = internals::RfRegTarget(Dst + 1) },
    };

    void operator()() const
    {
        ve_add_16<Dst, Src0, Src1>();
    }
};

template <unsigned Dst, unsigned Src0, unsigned Src1>
struct Sub16
{
    static constexpr internals::MicroOp ms_MicroOps[] = {
        { .cycle = 0, .type = internals::MicroOpType::WRITE, .target = internals::MicroOpTarget::MCU_PIPELINE },

        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src0) },
        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src0 + 1) },
        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src1) },
        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src1 + 1) },

        { .cycle = 4, .type = internals::MicroOpType::WRITE, .target = internals::RfRegTarget(Dst) },
        { .cycle = 4, .type = internals::MicroOpType::WRITE, .target = internals::RfRegTarget(Dst + 1) },
    };

    void operator()() const
    {
        ve_sub_16<Dst, Src0, Src1>();
    }
};

template <unsigned Dst, unsigned Src, unsigned Shift>
struct ASR16
{
    static constexpr internals::MicroOp ms_MicroOps[] = {
        { .cycle = 0, .type = internals::MicroOpType::WRITE, .target = internals::MicroOpTarget::MCU_PIPELINE },
        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src) },
        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src + 1) },
        { .cycle = 4, .type = internals::MicroOpType::WRITE, .target = internals::RfRegTarget(Dst) },
        { .cycle = 4, .type = internals::MicroOpType::WRITE, .target = internals::RfRegTarget(Dst + 1) },
    };

    void operator()() const
    {
        ve_asr_16<Dst, Src, Shift>();
    }
};

template <unsigned Dst, unsigned Src, unsigned Shift>
struct ASRSat_16_8
{
    static constexpr internals::MicroOp ms_MicroOps[] = {
        { .cycle = 0, .type = internals::MicroOpType::WRITE, .target = internals::MicroOpTarget::MCU_PIPELINE },
        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src) },
        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src + 1) },
        { .cycle = 4, .type = internals::MicroOpType::WRITE, .target = internals::RfRegTarget(Dst) },
    };

    void operator()() const
    {
        ve_asrsat_16_8<Dst, Src, Shift>();
    }
};

template <unsigned Dst, unsigned Src, unsigned Shift>
struct LSRSat_16_8
{
    static constexpr internals::MicroOp ms_MicroOps[] = {
        { .cycle = 0, .type = internals::MicroOpType::WRITE, .target = internals::MicroOpTarget::MCU_PIPELINE },
        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src) },
        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src + 1) },
        { .cycle = 4, .type = internals::MicroOpType::WRITE, .target = internals::RfRegTarget(Dst) },
    };

    void operator()() const
    {
        ve_lsrsat_16_8<Dst, Src, Shift>();
    }
};

template <unsigned Dst, unsigned Src, unsigned Shift>
struct ShiftRight16
{
    static constexpr internals::MicroOp ms_MicroOps[] = {
        { .cycle = 0, .type = internals::MicroOpType::WRITE, .target = internals::MicroOpTarget::MCU_PIPELINE },
        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src) },
        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src + 1) },
        { .cycle = 4, .type = internals::MicroOpType::WRITE, .target = internals::RfRegTarget(Dst) },
        { .cycle = 4, .type = internals::MicroOpType::WRITE, .target = internals::RfRegTarget(Dst + 1) },
    };

    void operator()() const
    {
        SR16<Dst, Src, Shift>();
    }
};

template <unsigned Dst, unsigned Src0, unsigned Src1>
struct SUMull16
{
    static constexpr internals::MicroOp ms_MicroOps[] = {
        { .cycle = 0, .type = internals::MicroOpType::WRITE, .target = internals::MicroOpTarget::MCU_PIPELINE },

        { .cycle = 0, .type = internals::MicroOpType::WRITE, .target = internals::MicroOpTarget::VE_MUL_PIPELINE },
        { .cycle = 1, .type = internals::MicroOpType::WRITE, .target = internals::MicroOpTarget::VE_MUL_PIPELINE },
        { .cycle = 2, .type = internals::MicroOpType::WRITE, .target = internals::MicroOpTarget::VE_MUL_PIPELINE },
        { .cycle = 3, .type = internals::MicroOpType::WRITE, .target = internals::MicroOpTarget::VE_MUL_PIPELINE },

        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src0) },
        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src0 + 1) },
        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src1) },
        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src1 + 1) },

        { .cycle = 6, .type = internals::MicroOpType::WRITE, .target = internals::RfRegTarget(Dst) },
        { .cycle = 6, .type = internals::MicroOpType::WRITE, .target = internals::RfRegTarget(Dst + 1) },
        { .cycle = 7, .type = internals::MicroOpType::WRITE, .target = internals::RfRegTarget(Dst + 2) },
        { .cycle = 7, .type = internals::MicroOpType::WRITE, .target = internals::RfRegTarget(Dst + 3) },
    };

    void operator()() const
    {
        ve_sumull_16<Dst, Src0, Src1>();
    }
};

template <unsigned Dst, unsigned Src, unsigned Shift, unsigned Label = UINT32_MAX>
struct ASRSat_32_16
{
    static constexpr internals::MicroOp ms_MicroOps[] = {
        { .cycle = 0, .type = internals::MicroOpType::WRITE, .target = internals::MicroOpTarget::MCU_PIPELINE },

        { .cycle = 0, .type = internals::MicroOpType::WRITE, .target = internals::MicroOpTarget::VE_SHIFT_PIPELINE },
        { .cycle = 1, .type = internals::MicroOpType::WRITE, .target = internals::MicroOpTarget::VE_SHIFT_PIPELINE },

        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src) },
        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src + 1) },
        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src + 2) },
        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src + 3) },

        { .cycle = 5, .type = internals::MicroOpType::WRITE, .target = internals::RfRegTarget(Dst) },
        { .cycle = 5, .type = internals::MicroOpType::WRITE, .target = internals::RfRegTarget(Dst + 1) },
    };

    void operator()() const
    {
        if constexpr (Label != UINT32_MAX)
        {
            __ASM volatile("ASRSat_32_16_%c0:" ::"n"(Label));
        }
        ve_asrsat_32_16<Dst, Src, Shift>();
    }
};

template <unsigned Dst>
struct Regrep16
{
    static constexpr internals::MicroOp ms_MicroOps[] = {
        { .cycle = 0, .type = internals::MicroOpType::WRITE, .target = internals::MicroOpTarget::MCU_PIPELINE },
        { .cycle = 2, .type = internals::MicroOpType::WRITE, .target = internals::RfRegTarget(Dst) },
        { .cycle = 2, .type = internals::MicroOpType::WRITE, .target = internals::RfRegTarget(Dst + 1) },
    };

    explicit constexpr Regrep16(const unsigned rt)
        : m_Rt(rt)
    {}

    void operator()() const
    {
        ve_regrep_16<Dst>(m_Rt);
    }

    unsigned m_Rt;
};

template <unsigned Dst, unsigned Src>
struct RegrepAdd16
{
    static constexpr internals::MicroOp ms_MicroOps[] = {
        { .cycle = 0, .type = internals::MicroOpType::WRITE, .target = internals::MicroOpTarget::MCU_PIPELINE },
        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src) },
        { .cycle = 2, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src + 1) },
        { .cycle = 4, .type = internals::MicroOpType::WRITE, .target = internals::RfRegTarget(Dst) },
        { .cycle = 4, .type = internals::MicroOpType::WRITE, .target = internals::RfRegTarget(Dst + 1) },
    };

    explicit constexpr RegrepAdd16(const unsigned rt)
        : m_Rt(rt)
    {}

    void operator()() const
    {
        ve_regrepadd_16<Dst, Src>(m_Rt);
    }

    unsigned m_Rt;
};

template <unsigned Dst>
struct LoadInramRf
{
    static constexpr internals::MicroOp ms_MicroOps[] = {
        { .cycle = 0, .type = internals::MicroOpType::WRITE, .target = internals::MicroOpTarget::MCU_PIPELINE },
        { .cycle = 4, .type = internals::MicroOpType::WRITE, .target = internals::RfRegTarget(Dst) },
        { .cycle = 4, .type = internals::MicroOpType::WRITE, .target = internals::RfRegTarget(Dst + 1) },
    };

    explicit constexpr LoadInramRf(const unsigned ramId, const lsu::Address inramAddr)
        : m_Rt(ramId)
        , m_Rt2(inramAddr)
    {}

    void operator()() const
    {
        lsu::LoadInramRf<Dst>(m_Rt, m_Rt2);
    }

    unsigned m_Rt;
    lsu::Address m_Rt2;
};

template <unsigned Dst>
struct LoadHalfInramRf
{
    static constexpr internals::MicroOp ms_MicroOps[] = {
        { .cycle = 0, .type = internals::MicroOpType::WRITE, .target = internals::MicroOpTarget::MCU_PIPELINE },
        { .cycle = 4, .type = internals::MicroOpType::WRITE, .target = internals::RfRegTarget(Dst) },
    };

    explicit constexpr LoadHalfInramRf(const unsigned ramId, const lsu::Address inramAddr)
        : m_Rt(ramId)
        , m_Rt2(inramAddr)
    {}

    void operator()() const
    {
        lsu::LoadHalfInramRf<Dst>(m_Rt, m_Rt2);
    }

    unsigned m_Rt;
    lsu::Address m_Rt2;
};

template <unsigned Src>
struct StoreRfOutram
{
    static constexpr internals::MicroOp ms_MicroOps[] = {
        { .cycle = 0, .type = internals::MicroOpType::WRITE, .target = internals::MicroOpTarget::MCU_PIPELINE },
        { .cycle = 1, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src) },
        { .cycle = 1, .type = internals::MicroOpType::READ, .target = internals::RfRegTarget(Src + 1) },
    };

    explicit constexpr StoreRfOutram(const lsu::Address outramAddr)
        : m_Rt(outramAddr)
    {}

    void operator()() const
    {
        lsu::StoreRfOutram<Src>(m_Rt);
    }

    lsu::Address m_Rt;
};

template <unsigned Dst, unsigned Src>
auto ConvertTo16b()
{
    static_assert((Dst % 2) == 0);

    using namespace cexec;

    if constexpr ((Src % 2) == 0)
    {
        return std::tuple{
            Mov8<Dst + 1, Src>{},
            ShiftRight16<Dst, Dst, 8>{},
        };
    }
    else
    {
        return std::tuple{
            ShiftRight16<Dst, Src - 1, 8>{},
        };
    }
}

template <unsigned Dst,
          unsigned Src,
          unsigned ZeroPoint,
          unsigned Multiplier,
          unsigned Shift,
          unsigned Tmp,
          unsigned Label>
auto Rescale()
{
    using namespace cexec;

    return std::tuple{
        Sub16<Dst, Src, ZeroPoint>{},
        SUMull16<Tmp, Dst, Multiplier>{},
        // To be modified by self-modifying code
        // Labelled ASRSat_32_16_<Label>
        ASRSat_32_16<Dst, Tmp, Shift, Label>{},
        RegrepAdd16<Dst, Dst>{ 1U },
        ASR16<Dst, Dst, 1>{},
    };
}

template <unsigned Dst, unsigned Src, unsigned Tmp, unsigned Zero, bool InitZero = false>
auto Sat_16_8()
{
    if constexpr (k_IsSigned)
    {
        return std::tuple{
            ASRSat_16_8<Dst, Src, 0>{},
        };
    }
    else if constexpr (InitZero)
    {
        return std::tuple{
            Sub16<Zero, Zero, Zero>{},
            SMax16<Tmp, Src, Zero>{},
            LSRSat_16_8<Dst, Tmp, 0>{},
        };
    }
    else
    {
        return std::tuple{
            SMax16<Tmp, Src, Zero>{},
            LSRSat_16_8<Dst, Tmp, 0>{},
        };
    }
}
}    // namespace cexec
}    // namespace
