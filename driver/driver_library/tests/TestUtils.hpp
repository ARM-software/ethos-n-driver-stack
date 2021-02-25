//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "../include/ethosn_driver_library/Profiling.hpp"

#include <chrono>
#include <cstring>
#include <fstream>
#include <thread>

inline bool PollForConfigure(ethosn::driver_library::profiling::Configuration config, int8_t attempts = 2)
{
    using namespace std::chrono_literals;
    // Make sure that the configuration is successfull. There might be some tear down
    // or set up still in flight from previous/current test.
    bool ret = false;
    do
    {
        ret = ethosn::driver_library::profiling::Configure(config);
        if (ret)
            break;
        // Wait before polling again.
        std::this_thread::sleep_for(100ms);
        --attempts;
    } while (attempts > 0);
    return ret;
}

/// Utility to enable profiling with the given options and then automatically disable it at the end of
/// the scope. This is useful so that the profiling state does not affect other tests.
struct ScopedProfilingEnablement
{
    ScopedProfilingEnablement(ethosn::driver_library::profiling::Configuration config)
    {
        using namespace std::chrono_literals;
        PollForConfigure(config);
    }

    ~ScopedProfilingEnablement()
    {
        using namespace std::chrono_literals;
        PollForConfigure(ethosn::driver_library::profiling::Configuration());
    }
};

struct ScopedModuleParameterAccessor
{
    ScopedModuleParameterAccessor(const char* syspath, const std::string parameterValue)
        : m_Syspath(syspath)
    {
        std::fstream param(m_Syspath);
        if (!param.is_open())
        {
            throw std::runtime_error(std::string("Unable to open ") + m_Syspath + std::string(": ") +
                                     std::strerror(errno));
        }
        param >> m_OriginalState;
        param << parameterValue;
    }

    ~ScopedModuleParameterAccessor()
    {
        std::fstream param(m_Syspath, std::ios_base::out | std::ios_base::trunc);
        if (!param.is_open())
        {
            return;
        }
        param << m_OriginalState;
    }

private:
    std::string m_Syspath;
    std::string m_OriginalState;
};
