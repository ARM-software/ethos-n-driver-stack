//
// Copyright © 2018-2020,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_driver_library/Inference.hpp"

#include "DumpProfiling.hpp"
#include "ProfilingInternal.hpp"

#if defined(__unix__)
#include <uapi/ethosn.h>
#endif

#include <cstdint>
#include <fstream>
#include <iostream>
#if defined(__unix__)
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
    : inferenceImpl{ std::make_unique<InferenceImpl>(fileDescriptor) }
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
    return inferenceImpl->GetFileDescriptor();
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

}    // namespace driver_library
}    // namespace ethosn
