//
// Copyright © 2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/profiling/IBackendProfilingContext.hpp>
#include <ethosn_driver_library/Profiling.hpp>

namespace armnn
{
namespace profiling
{

class EthosNBackendProfilingContext : public IBackendProfilingContext
{
public:
    EthosNBackendProfilingContext(IBackendInternal::IBackendProfilingPtr& backendProfiling)
        : m_BackendProfiling(backendProfiling)
        , m_CapturePeriod(0)
    {}

    // The following is a rough sequence of calls from armnn :-
    // 1. RegisterCounters()
    // 2. EnableProfiling( flag = true)
    // 3. ActivateCounters()
    // 4. ReportCounterValues() called multiple times when inference is running as well as finished
    // 5. EnableProfiling(flag = false)
    // 6. goto step 2
    uint16_t RegisterCounters(uint16_t currentMaxGlobalCounterID);
    Optional<std::string> ActivateCounters(uint32_t capturePeriod, const std::vector<uint16_t>& counterIds);
    std::vector<Timestamp> ReportCounterValues();
    void EnableProfiling(bool flag);

private:
    IBackendInternal::IBackendProfilingPtr& m_BackendProfiling;
    uint32_t m_CapturePeriod;
    ethosn::driver_library::profiling::Configuration m_config;
    std::vector<uint16_t> m_ActiveCounters;
};

}    // namespace profiling

}    // namespace armnn
