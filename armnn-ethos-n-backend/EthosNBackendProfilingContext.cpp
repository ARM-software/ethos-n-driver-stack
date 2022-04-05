//
// Copyright © 2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "EthosNBackendProfilingContext.hpp"

#include <armnn/backends/profiling/IBackendProfiling.hpp>

namespace armnn
{

namespace profiling
{

uint16_t EthosNBackendProfilingContext::RegisterCounters(uint16_t currentMaxGlobalCounterID)
{
    std::unique_ptr<IRegisterBackendCounters> counterRegistrar =
        m_BackendProfiling->GetCounterRegistrationInterface(currentMaxGlobalCounterID);

    std::string categoryName("DriverLibraryCounters");
    counterRegistrar->RegisterCategory(categoryName);
    uint16_t nextMaxGlobalCounterId = counterRegistrar->RegisterCounter(
        static_cast<uint16_t>(ethosn::driver_library::profiling::PollCounterName::DriverLibraryNumLiveBuffers),
        categoryName, 0, 0, 1.f, "DriverLibraryNumLiveBuffers",
        "The number of currently live instances of the Buffer class.");
    nextMaxGlobalCounterId = counterRegistrar->RegisterCounter(
        static_cast<uint16_t>(ethosn::driver_library::profiling::PollCounterName::DriverLibraryNumLiveInferences),
        categoryName, 0, 0, 1.f, "DriverLibraryNumLiveInferences",
        "The number of currently live instances of the Inference class.");

    categoryName = "KernelDriverCounters";
    counterRegistrar->RegisterCategory(categoryName);
    nextMaxGlobalCounterId = counterRegistrar->RegisterCounter(
        static_cast<uint16_t>(ethosn::driver_library::profiling::PollCounterName::KernelDriverNumMailboxMessagesSent),
        categoryName, 0, 0, 1.f, "KernelDriverNumMailboxMessagesSent",
        "The number of mailbox messages sent by the kernel driver.");
    nextMaxGlobalCounterId = counterRegistrar->RegisterCounter(
        static_cast<uint16_t>(
            ethosn::driver_library::profiling::PollCounterName::KernelDriverNumMailboxMessagesReceived),
        categoryName, 0, 0, 1.f, "KernelDriverNumMailboxMessagesReceived",
        "The number of mailbox messages received by the kernel driver.");

    return nextMaxGlobalCounterId;
}

Optional<std::string> EthosNBackendProfilingContext::ActivateCounters(uint32_t capturePeriod,
                                                                      const std::vector<uint16_t>& counterIds)
{
    if (capturePeriod == 0 || counterIds.size() == 0)
    {
        m_ActiveCounters.clear();
    }
    m_CapturePeriod  = capturePeriod;
    m_ActiveCounters = counterIds;
    return Optional<std::string>();
}

std::vector<Timestamp> EthosNBackendProfilingContext::ReportCounterValues()
{
    std::vector<CounterValue> counterValues;

    for (auto counterId : m_ActiveCounters)
    {
        counterValues.emplace_back(CounterValue{
            counterId, static_cast<uint32_t>(ethosn::driver_library::profiling::GetCounterValue(
                           static_cast<ethosn::driver_library::profiling::PollCounterName>(counterId))) });
    }

    uint64_t timestamp = m_CapturePeriod;
    return { Timestamp{ timestamp, counterValues } };
}

bool EthosNBackendProfilingContext::EnableProfiling(bool flag)
{
    // Create a temporary config so if Configure fails we don't update the actual config
    auto config              = m_Config;
    config.m_EnableProfiling = flag;
    bool configured          = ethosn::driver_library::profiling::Configure(config);
    if (!configured)
    {
        return false;
    }
    m_Config           = config;
    m_ProfilingEnabled = flag;
    return true;
}

// Currently armnn doesn't use this api. So we are setting up the timeline events just before we send them
// See EthosNPreCompiledWorkload.cpp Execute() for more details.
// IVGCVSW-4708 needed before we can use this.
bool EthosNBackendProfilingContext::EnableTimelineReporting(bool)
{
    return true;
}

bool EthosNBackendProfilingContext::IsProfilingEnabled() const
{
    return m_ProfilingEnabled;
}

IProfilingGuidGenerator& EthosNBackendProfilingContext::GetGuidGenerator() const
{
    return m_GuidGenerator;
}

ISendTimelinePacket* EthosNBackendProfilingContext::GetSendTimelinePacket() const
{
    return m_SendTimelinePacket.get();
}

std::map<uint64_t, ProfilingDynamicGuid>& EthosNBackendProfilingContext::GetIdToEntityGuids()
{
    return m_IdToEntityGuids;
}

}    // namespace profiling

}    // namespace armnn
