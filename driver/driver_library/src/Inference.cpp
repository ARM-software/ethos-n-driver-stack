//
// Copyright © 2018-2020,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_driver_library/Inference.hpp"

#include "DumpProfiling.hpp"
#include "ProfilingInternal.hpp"
#include "Utils.hpp"

#if defined(__unix__)
#include <uapi/ethosn.h>
#endif

#include <cstdint>
#include <fstream>
#include <iostream>
#include <thread>
#if defined(__unix__)
#include <poll.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace ethosn
{
namespace driver_library
{

class Inference::InferenceImpl
{
public:
    InferenceImpl(int fileDescriptor)
        : m_FileDescriptor(fileDescriptor)
    {}

    ~InferenceImpl()
    {
#if defined(__unix__)
        close(m_FileDescriptor);
#endif
    };

    int GetFileDescriptor() const
    {
        return m_FileDescriptor;
    }

private:
    int m_FileDescriptor;
};

Inference::Inference(int fileDescriptor)
    : m_Impl{ std::make_unique<InferenceImpl>(fileDescriptor) }
{
    if (profiling::g_CurrentConfiguration.m_EnableProfiling)
    {
        RecordLifetimeEvent(this, profiling::g_InferenceToLifetimeEventId,
                            profiling::ProfilingEntry::Type::TimelineEventStart,
                            profiling::ProfilingEntry::MetadataCategory::InferenceLifetime);
    }
}

Inference::~Inference()
{
    if (profiling::g_CurrentConfiguration.m_EnableProfiling)
    {
        try
        {
            RecordLifetimeEvent(this, profiling::g_InferenceToLifetimeEventId,
                                profiling::ProfilingEntry::Type::TimelineEventEnd,
                                profiling::ProfilingEntry::MetadataCategory::InferenceLifetime);

            // Include profiling entries from the firmware if any.
            profiling::AppendKernelDriverEntries();
            // Dumping profiling data at inference destruction is convenient because
            // this is called frequently enough such that there is a good amount of data dumped
            // but not frequently enough to cause performance regressions.
            if (profiling::g_DumpFile.size() > 0)
            {
                std::ofstream file(profiling::g_DumpFile.c_str(), std::ios_base::out | std::ofstream::binary);
                profiling::DumpAllProfilingData(file);
            }
        }
        catch (const std::exception& e)
        {
            std::cerr << "Exception in ~Inference: " << e.what() << std::endl;
            return;
        }
    }
}

int Inference::GetFileDescriptor() const
{
    return m_Impl->GetFileDescriptor();
}

uint64_t Inference::GetCycleCount() const
{
    uint64_t cycleCount = 0;
#if defined(TARGET_KMOD)
    int result = ioctl(GetFileDescriptor(), ETHOSN_IOCTL_GET_CYCLE_COUNT, &cycleCount);
    if (result != 0)
    {
        throw std::runtime_error(std::string("Error querying cycle count."));
    }
#endif
    return cycleCount;
}

InferenceResult Inference::Wait(uint32_t timeoutMs) const
{
#if defined(__unix__)
    struct pollfd fds;
    memset(&fds, 0, sizeof(fds));
    fds.fd     = GetFileDescriptor();
    fds.events = POLLIN;    // Wait for any available input

    int pollResult = poll(&fds, 1, timeoutMs);

    if (pollResult < 0)
    {
        int err = errno;
        g_Logger.Error("Failed to read inference result status (poll returned %s)", strerror(err));
        return InferenceResult::Error;
    }
    else
    {
        // Either poll timed out or it finished successfully. Either way, read and return the final status
        ethosn::driver_library::InferenceResult result;
        if (read(GetFileDescriptor(), &result, sizeof(result)) != static_cast<ssize_t>(sizeof(result)))
        {
            int err = errno;
            g_Logger.Error("Failed to read inference result status (read returned %s)", strerror(err));
            return InferenceResult::Error;
        }
        else if (result == ethosn::driver_library::InferenceResult::Completed ||
                 result == ethosn::driver_library::InferenceResult::Error)
        {
            return result;
        }
        else if (result == ethosn::driver_library::InferenceResult::Scheduled ||
                 result == ethosn::driver_library::InferenceResult::Running)
        {
            g_Logger.Error("Inference timed out");
            return result;
        }
        else
        {
            g_Logger.Error("Inference failed with unknown status %d", static_cast<uint32_t>(result));
            return ethosn::driver_library::InferenceResult::Error;
        }
    }

#else
    // Default to success as for platforms other than Linux we assume we are running on the model and therefore
    // there is no need to wait.
    ETHOSN_UNUSED(timeoutMs);
    return InferenceResult::Completed;

#endif
}

}    // namespace driver_library
}    // namespace ethosn
