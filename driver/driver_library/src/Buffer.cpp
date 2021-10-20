//
// Copyright © 2018-2021 Arm Limited.
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

Buffer::Buffer(Buffer&& buffer)
    : bufferImpl(std::move(buffer.bufferImpl.get()))
{
    if (profiling::g_CurrentConfiguration.m_EnableProfiling)
    {
        RecordLifetimeEvent(this, profiling::g_BufferToLifetimeEventId,
                            profiling::ProfilingEntry::Type::TimelineEventStart,
                            profiling::ProfilingEntry::MetadataCategory::BufferLifetime);
    }
}

Buffer::Buffer(uint32_t size, DataFormat format, const std::string& device)
    : bufferImpl{ std::make_unique<BufferImpl>(size, format, device) }
{
    if (profiling::g_CurrentConfiguration.m_EnableProfiling)
    {
        RecordLifetimeEvent(this, profiling::g_BufferToLifetimeEventId,
                            profiling::ProfilingEntry::Type::TimelineEventStart,
                            profiling::ProfilingEntry::MetadataCategory::BufferLifetime);
    }
}

Buffer::Buffer(uint32_t size, DataFormat format)
    : Buffer(size, format, DEVICE_NODE)
{}

Buffer::Buffer(const uint8_t* src, uint32_t size, DataFormat format, const std::string& device)
    : bufferImpl{ std::make_unique<BufferImpl>(src, size, format, device) }
{
    if (profiling::g_CurrentConfiguration.m_EnableProfiling)
    {
        RecordLifetimeEvent(this, profiling::g_BufferToLifetimeEventId,
                            profiling::ProfilingEntry::Type::TimelineEventStart,
                            profiling::ProfilingEntry::MetadataCategory::BufferLifetime);
    }
}

Buffer::Buffer(const uint8_t* src, uint32_t size, DataFormat format)
    : Buffer(src, size, format, DEVICE_NODE)
{}

uint32_t Buffer::GetSize()
{
    return bufferImpl->GetSize();
}

DataFormat Buffer::GetDataFormat()
{
    return bufferImpl->GetDataFormat();
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

const int& Buffer::GetBufferHandle() const
{
    return bufferImpl->GetBufferHandle();
}

uint8_t* Buffer::Map()
{
    return bufferImpl->Map();
}

void Buffer::Unmap()
{
    bufferImpl->Unmap();
}

}    // namespace driver_library
}    // namespace ethosn
