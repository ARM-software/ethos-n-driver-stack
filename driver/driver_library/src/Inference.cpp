//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_driver_library/Inference.hpp"

#include "DumpProfiling.hpp"
#include "ProfilingInternal.hpp"

#include <cstdint>
#include <fstream>
#if defined(__unix__)
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
        : m_fileDescriptor(fileDescriptor)
    {}

    ~InferenceImpl()
    {
#if defined(__unix__)
        close(m_fileDescriptor);
#endif
    };

    int GetFileDescriptor()
    {
        return m_fileDescriptor;
    }

private:
    int m_fileDescriptor;
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
}

int Inference::GetFileDescriptor()
{
    return inferenceImpl->GetFileDescriptor();
}

}    // namespace driver_library
}    // namespace ethosn
