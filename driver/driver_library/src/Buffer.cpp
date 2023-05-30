//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_driver_library/Buffer.hpp"

#include "ProfilingInternal.hpp"
#ifdef TARGET_KMOD
#include "KmodBuffer.hpp"
#else
#include "ModelBuffer.hpp"
#endif

#include <chrono>

namespace ethosn
{
namespace driver_library
{

Buffer::Buffer(Buffer&& otherBuffer)
    : bufferImpl(std::move(otherBuffer.bufferImpl))
{
    if (profiling::g_CurrentConfiguration.m_EnableProfiling)
    {
        RecordLifetimeEvent(this, profiling::g_BufferToLifetimeEventId,
                            profiling::ProfilingEntry::Type::TimelineEventStart,
                            profiling::ProfilingEntry::MetadataCategory::BufferLifetime);
    }
}

Buffer::Buffer(std::unique_ptr<BufferImpl> otherBufferImpl)
    : bufferImpl(std::move(otherBufferImpl))
{
    if (profiling::g_CurrentConfiguration.m_EnableProfiling)
    {
        RecordLifetimeEvent(this, profiling::g_BufferToLifetimeEventId,
                            profiling::ProfilingEntry::Type::TimelineEventStart,
                            profiling::ProfilingEntry::MetadataCategory::BufferLifetime);
    }
}

Buffer::~Buffer()
{
    if (profiling::g_CurrentConfiguration.m_EnableProfiling)
    {
        RecordLifetimeEvent(this, profiling::g_BufferToLifetimeEventId,
                            profiling::ProfilingEntry::Type::TimelineEventEnd,
                            profiling::ProfilingEntry::MetadataCategory::BufferLifetime);
    }
}

uint32_t Buffer::GetSize()
{
    if (!bufferImpl)
    {
        throw std::runtime_error("Unable to GetSize as BufferImpl is null");
    }
    return bufferImpl->GetSize();
}

const int& Buffer::GetBufferHandle() const
{
    if (!bufferImpl)
    {
        throw std::runtime_error("Unable to GetBufferHandle as BufferImpl is null");
    }
    return bufferImpl->GetBufferHandle();
}

uint8_t* Buffer::Map()
{
    if (!bufferImpl)
    {
        throw std::runtime_error("Unable to Map as BufferImpl is null");
    }
    return bufferImpl->Map();
}

void Buffer::Unmap()
{
    if (!bufferImpl)
    {
        throw std::runtime_error("Unable to Unmap as BufferImpl is null");
    }
    bufferImpl->Unmap();
}

}    // namespace driver_library
}    // namespace ethosn
