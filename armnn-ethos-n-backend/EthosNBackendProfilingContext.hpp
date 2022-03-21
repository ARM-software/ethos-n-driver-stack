//
// Copyright © 2020-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/profiling/IBackendProfilingContext.hpp>
#include <ethosn_driver_library/Network.hpp>
#include <ethosn_driver_library/Profiling.hpp>

namespace armnn
{
namespace profiling
{

class EthosNBackendProfilingContext : public arm::pipe::IBackendProfilingContext
{
public:
    EthosNBackendProfilingContext(IBackendInternal::IBackendProfilingPtr& backendProfiling)
        : m_ProfilingEnabled(backendProfiling->IsProfilingEnabled())
        , m_GuidGenerator(backendProfiling->GetProfilingGuidGenerator())
        , m_SendTimelinePacket(backendProfiling->GetSendTimelinePacket())
        , m_BackendProfiling(backendProfiling)
        , m_CapturePeriod(0)
    {
        if (!ethosn::driver_library::VerifyKernel())
        {
            throw RuntimeException("Kernel version is not supported");
        }
    }

    // The following is a rough sequence of calls from armnn :-
    // 1. RegisterCounters()
    // 2. EnableProfiling( flag = true)
    // 3. ActivateCounters()
    // 4. ReportCounterValues() called multiple times when inference is running as well as finished
    // 5. EnableProfiling(flag = false)
    // 6. goto step 2
    uint16_t RegisterCounters(uint16_t currentMaxGlobalCounterID) override;
    arm::pipe::Optional<std::string> ActivateCounters(uint32_t capturePeriod,
                                                      const std::vector<uint16_t>& counterIds) override;
    std::vector<arm::pipe::Timestamp> ReportCounterValues() override;
    bool EnableProfiling(bool flag) override;
    bool EnableTimelineReporting(bool flag) override;

    bool IsProfilingEnabled() const;
    arm::pipe::IProfilingGuidGenerator& GetGuidGenerator() const;
    arm::pipe::ISendTimelinePacket* GetSendTimelinePacket() const;

    std::map<uint64_t, arm::pipe::ProfilingDynamicGuid>& GetIdToEntityGuids();

private:
    bool m_ProfilingEnabled;
    arm::pipe::IProfilingGuidGenerator& m_GuidGenerator;
    std::unique_ptr<arm::pipe::ISendTimelinePacket> m_SendTimelinePacket;
    IBackendInternal::IBackendProfilingPtr& m_BackendProfiling;
    uint32_t m_CapturePeriod;
    ethosn::driver_library::profiling::Configuration m_Config;
    std::vector<uint16_t> m_ActiveCounters;
    std::map<uint64_t, arm::pipe::ProfilingDynamicGuid> m_IdToEntityGuids;
};

}    // namespace profiling

}    // namespace armnn
