//
// Copyright © 2018-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include <common/FirmwareApi.hpp>

namespace ethosn
{
namespace control_unit
{

template <typename HAL>
class Pmu final
{
public:
    /// Enables the PMU and starts the cycle counter running.
    /// Call Reset() to enable further counters.
    Pmu(HAL& hal);

    /// Resets the PMU, clearing and resetting all counters.
    /// Enables the given set of counters if requested, which can then later be
    /// queried with ReadCounter.
    void Reset(uint32_t numCounters, const ethosn_profiling_hw_counter_types* counters);

    uint32_t GetCycleCount32();
    uint64_t GetCycleCount64();

    uint32_t ReadCounter(uint32_t counter);

private:
    HAL& m_Hal;
};

}    // namespace control_unit
}    // namespace ethosn
