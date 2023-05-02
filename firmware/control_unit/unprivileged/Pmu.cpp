//
// Copyright © 2018-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "Pmu.hpp"

#include <scylla_addr_fields.h>
#include <scylla_regs.h>

#include <common/Utils.hpp>

namespace ethosn
{
namespace control_unit
{

namespace
{
pm_top_event_type_t ConvertPublicEventTypeToInternal(ethosn_profiling_hw_counter_types publicEvent)
{
    switch (publicEvent)
    {
        case ethosn_profiling_hw_counter_types::BUS_ACCESS_RD_TRANSFERS:
            return pm_top_event_type_t::BUS_ACCESS_RD_TRANSFERS;
        case ethosn_profiling_hw_counter_types::BUS_RD_COMPLETE_TRANSFERS:
            return pm_top_event_type_t::BUS_RD_COMPLETE_TRANSFERS;
        case ethosn_profiling_hw_counter_types::BUS_READ_BEATS:
            return pm_top_event_type_t::BUS_READ_BEATS;
        case ethosn_profiling_hw_counter_types::BUS_READ_TXFR_STALL_CYCLES:
            return pm_top_event_type_t::BUS_READ_TXFR_STALL_CYCLES;
        case ethosn_profiling_hw_counter_types::BUS_ACCESS_WR_TRANSFERS:
            return pm_top_event_type_t::BUS_ACCESS_WR_TRANSFERS;
        case ethosn_profiling_hw_counter_types::BUS_WR_COMPLETE_TRANSFERS:
            return pm_top_event_type_t::BUS_WR_COMPLETE_TRANSFERS;
        case ethosn_profiling_hw_counter_types::BUS_WRITE_BEATS:
            return pm_top_event_type_t::BUS_WRITE_BEATS;
        case ethosn_profiling_hw_counter_types::BUS_WRITE_TXFR_STALL_CYCLES:
            return pm_top_event_type_t::BUS_WRITE_TXFR_STALL_CYCLES;
        case ethosn_profiling_hw_counter_types::BUS_WRITE_STALL_CYCLES:
            return pm_top_event_type_t::BUS_WRITE_STALL_CYCLES;
        case ethosn_profiling_hw_counter_types::BUS_ERROR_COUNT:
            return pm_top_event_type_t::BUS_ERROR_COUNT;
        case ethosn_profiling_hw_counter_types::NCU_MCU_ICACHE_MISS:
            return pm_top_event_type_t::NCU_MCU_ICACHE_MISS;
        case ethosn_profiling_hw_counter_types::NCU_MCU_DCACHE_MISS:
            return pm_top_event_type_t::NCU_MCU_DCACHE_MISS;
        case ethosn_profiling_hw_counter_types::NCU_MCU_BUS_READ_BEATS:
            return pm_top_event_type_t::NCU_MCU_BUS_READ_BEATS;
        case ethosn_profiling_hw_counter_types::NCU_MCU_BUS_WRITE_BEATS:
            return pm_top_event_type_t::NCU_MCU_BUS_WRITE_BEATS;
        default:
        {
            ASSERT_MSG(false, "ConvertPublicEventTypeToInternal: ethosn_profiling_hw_counter_types out of sync with "
                              "pm_top_event_type_t");
            return static_cast<pm_top_event_type_t>(0);
        }
    }
}
}    // namespace

template <typename HAL>
Pmu<HAL>::Pmu(HAL& hal)
    : m_Hal(hal)
{
    // Enable the PMU even if not compiled with CONTROL_UNIT_PROFILING,
    // to enable the simple reporting of the inference cycle count.

    // PMU ignores all register reads and writes before enable.
    // So make sure to enable it as a separate write, before anything else
    pmcr_r pmcr;
    pmcr.set_cnt_en(1);
    m_Hal.WriteReg(TOP_REG(PMU_RP, PMU_PMCR), pmcr.word);

    // Start the cycle counter running immediately. It can be reset later as desired.
    Reset(0, {});
}

template <typename HAL>
void Pmu<HAL>::Reset(uint32_t numCounters, const ethosn_profiling_hw_counter_types* counters)
{
    ASSERT_MSG(numCounters < 6, "Only 6 counters supported at a time");

    // Reset all counters to zero
    pmcr_r pmcr;
    pmcr.set_cnt_en(1);
    pmcr.set_event_cnt_rst(1);
    pmcr.set_cycle_cnt_rst(1);
    m_Hal.WriteReg(TOP_REG(PMU_RP, PMU_PMCR), pmcr.word);

    // Enable the cycle count plus any other requested counters
    pmcntenset_r counterEnable;
    counterEnable.set_cycle_cnt(1);

    for (uint32_t i = 0; i < numCounters; i++)
    {
        uint32_t internalEventBits = static_cast<uint32_t>(ConvertPublicEventTypeToInternal(counters[i]));

        pmevtyper0_r typeReg;
        typeReg.set_event_type(internalEventBits);
        m_Hal.WriteReg(TOP_REG(PMU_RP, PMU_PMEVTYPER0 + i * (PMU_PMEVTYPER1 - PMU_PMEVTYPER0)), typeReg.word);

        counterEnable.word |= static_cast<uint32_t>(1 << i);
    }

    m_Hal.WriteReg(TOP_REG(PMU_RP, PMU_PMCNTENSET), counterEnable.word);
}

template <typename HAL>
uint32_t Pmu<HAL>::GetCycleCount32()
{
    return m_Hal.ReadReg(TOP_REG(PMU_RP, PMU_PMCCNTR_LO));
}

template <typename HAL>
uint64_t Pmu<HAL>::GetCycleCount64()
{
    uint64_t cycleCount = m_Hal.ReadReg(TOP_REG(PMU_RP, PMU_PMCCNTR_LO));
    cycleCount |= (static_cast<uint64_t>(m_Hal.ReadReg(TOP_REG(PMU_RP, PMU_PMCCNTR_HI))) << 32U);
    return cycleCount;
}

template <typename HAL>
uint32_t Pmu<HAL>::ReadCounter(uint32_t counter)
{
    ASSERT_MSG(counter < 6, "Only 6 counters supported at a time");
    pmevcntr0_r countReg(m_Hal.ReadReg(TOP_REG(PMU_RP, PMU_PMEVCNTR0 + counter * (PMU_PMEVCNTR1 - PMU_PMEVCNTR0))));
    return countReg.get_event_cnt();
}

}    // namespace control_unit
}    // namespace ethosn

// Because we are defining template methods in this cpp file we need to explicitly instantiate all versions
// that we intend to use.
#if defined(CONTROL_UNIT_MODEL)
#include <model/ModelHal.hpp>
template class ethosn::control_unit::Pmu<ethosn::control_unit::ModelHal>;

#include <model/UscriptHal.hpp>
template class ethosn::control_unit::Pmu<ethosn::control_unit::UscriptHal<ethosn::control_unit::ModelHal>>;

#include <model/LoggingHal.hpp>
template class ethosn::control_unit::Pmu<ethosn::control_unit::LoggingHal>;
#endif

#if defined(CONTROL_UNIT_HARDWARE)
#include <common/hals/HardwareHal.hpp>
template class ethosn::control_unit::Pmu<ethosn::control_unit::HardwareHal>;
#endif
