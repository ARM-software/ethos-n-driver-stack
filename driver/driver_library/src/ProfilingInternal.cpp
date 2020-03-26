//
// Copyright © 2019-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "ProfilingInternal.hpp"

#include <uapi/ethosn.h>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <vector>

namespace ethosn
{
namespace driver_library
{
namespace profiling
{

uint64_t GetNextTimeLineEventId()
{
    g_NextTimelineEventId = std::max(DRIVER_LIBRARY_EVENT_ID_BASE, g_NextTimelineEventId + 1);
    return g_NextTimelineEventId;
}

bool ApplyConfiguration(Configuration config)
{
    bool hasKernelConfigureSucceeded = ConfigureKernelDriver(config);

    if (hasKernelConfigureSucceeded && g_CurrentConfiguration.m_EnableProfiling && !config.m_EnableProfiling)
    {
        g_ProfilingEntries.clear();
        g_BufferToLifetimeEventId.clear();
        g_InferenceToLifetimeEventId.clear();
        g_NextTimelineEventId = DRIVER_LIBRARY_EVENT_ID_BASE;
    }

    return hasKernelConfigureSucceeded;
}

namespace
{
std::vector<std::string> Split(std::string s, char delim)
{
    std::stringstream ss(s);
    std::string token;
    std::vector<std::string> results;
    while (std::getline(ss, token, delim))
    {
        results.push_back(token);
    }
    return results;
}
}    // namespace

Configuration GetConfigFromString(const char* str)
{
    if (!str)
    {
        return Configuration();
    }
    Configuration config     = {};
    config.m_EnableProfiling = true;
    for (auto option : Split(str, ' '))
    {
        auto optionPair         = Split(option, '=');
        std::string optionName  = optionPair.size() >= 1 ? optionPair[0] : "";
        std::string optionValue = optionPair.size() >= 2 ? optionPair[1] : "";
        if (optionName == "dumpFile")
        {
            g_DumpFile = optionValue;
        }
        else if (optionName == "firmwareBufferSize")
        {
            config.m_FirmwareBufferSize = static_cast<uint32_t>(std::stoul(optionValue));
        }
        else if (optionName == "hwCounters")
        {
            auto hwCounters = Split(optionValue, ',');
            if (hwCounters.size() > 6)
            {
                std::cerr << "There can only be at most 6 hardware counters\n";
                continue;
            }
            for (auto counter : hwCounters)
            {
                if (counter == "busAccessRdTransfers")
                {
                    config.m_HardwareCounters[config.m_NumHardwareCounters++] =
                        HardwareCounters::FirmwareBusAccessRdTransfers;
                }
                else if (counter == "busRdCompleteTransfers")
                {
                    config.m_HardwareCounters[config.m_NumHardwareCounters++] =
                        HardwareCounters::FirmwareBusRdCompleteTransfers;
                }
                else if (counter == "busReadBeats")
                {
                    config.m_HardwareCounters[config.m_NumHardwareCounters++] = HardwareCounters::FirmwareBusReadBeats;
                }
                else if (counter == "busReadTxfrStallCycles")
                {
                    config.m_HardwareCounters[config.m_NumHardwareCounters++] =
                        HardwareCounters::FirmwareBusReadTxfrStallCycles;
                }
                else if (counter == "busAccessWrTransfers")
                {
                    config.m_HardwareCounters[config.m_NumHardwareCounters++] =
                        HardwareCounters::FirmwareBusAccessWrTransfers;
                }
                else if (counter == "busWrCompleteTransfers")
                {
                    config.m_HardwareCounters[config.m_NumHardwareCounters++] =
                        HardwareCounters::FirmwareBusWrCompleteTransfers;
                }
                else if (counter == "busWriteBeats")
                {
                    config.m_HardwareCounters[config.m_NumHardwareCounters++] = HardwareCounters::FirmwareBusWriteBeats;
                }
                else if (counter == "busWriteTxfrStallCycles")
                {
                    config.m_HardwareCounters[config.m_NumHardwareCounters++] =
                        HardwareCounters::FirmwareBusWriteTxfrStallCycles;
                }
                else if (counter == "busWriteStallCycles")
                {
                    config.m_HardwareCounters[config.m_NumHardwareCounters++] =
                        HardwareCounters::FirmwareBusWriteStallCycles;
                }
                else if (counter == "busErrorCount")
                {
                    config.m_HardwareCounters[config.m_NumHardwareCounters++] = HardwareCounters::FirmwareBusErrorCount;
                }
                else if (counter == "ncuMcuIcacheMiss")
                {
                    config.m_HardwareCounters[config.m_NumHardwareCounters++] =
                        HardwareCounters::FirmwareNcuMcuIcacheMiss;
                }
                else if (counter == "ncuMcuDcacheMiss")
                {
                    config.m_HardwareCounters[config.m_NumHardwareCounters++] =
                        HardwareCounters::FirmwareNcuMcuDcacheMiss;
                }
                else if (counter == "ncuMcuBusReadBeats")
                {
                    config.m_HardwareCounters[config.m_NumHardwareCounters++] =
                        HardwareCounters::FirmwareNcuMcuBusReadBeats;
                }
                else if (counter == "ncuMcuBusWriteBeats")
                {
                    config.m_HardwareCounters[config.m_NumHardwareCounters++] =
                        HardwareCounters::FirmwareNcuMcuBusWriteBeats;
                }
            }
        }
    }
    return config;
}

Configuration GetDefaultConfiguration()
{
    const char* profilingConfigEnv = std::getenv("ETHOSN_DRIVER_LIBRARY_PROFILING_CONFIG");
    if (!profilingConfigEnv)
    {
        return Configuration();
    }
    auto config = GetConfigFromString(profilingConfigEnv);

    if (!ApplyConfiguration(config))
    {
        return Configuration();
    }
    return config;
}

std::string g_DumpFile               = "";
Configuration g_CurrentConfiguration = GetDefaultConfiguration();

std::vector<ProfilingEntry> g_ProfilingEntries              = {};
std::map<Buffer*, uint64_t> g_BufferToLifetimeEventId       = {};
std::map<Inference*, uint64_t> g_InferenceToLifetimeEventId = {};
uint64_t g_NextTimelineEventId                              = DRIVER_LIBRARY_EVENT_ID_BASE;

bool Configure(Configuration config)
{
    bool isConfigurationApplied = ApplyConfiguration(config);
    if (isConfigurationApplied)
    {
        g_CurrentConfiguration = config;
    }
    return isConfigurationApplied;
}

std::vector<ProfilingEntry> ReportNewProfilingData()
{
    std::vector<ProfilingEntry> res(std::move(g_ProfilingEntries));

    g_ProfilingEntries.clear();
    return res;
}

uint64_t GetCounterValue(PollCounterName counter)
{
    if (!g_CurrentConfiguration.m_EnableProfiling)
    {
        return 0;
    }

    switch (counter)
    {
        case PollCounterName::DriverLibraryNumLiveBuffers:
            return g_BufferToLifetimeEventId.size();
        case PollCounterName::DriverLibraryNumLiveInferences:
            return g_InferenceToLifetimeEventId.size();
        case PollCounterName::KernelDriverNumMailboxMessagesSent:    // Deliberate fallthrough
        case PollCounterName::KernelDriverNumMailboxMessagesReceived:
            return GetKernelDriverCounterValue(counter);
        default:
            assert(!"Invalid counter");
            return 0;
    }
}

}    // namespace profiling
}    // namespace driver_library
}    // namespace ethosn
