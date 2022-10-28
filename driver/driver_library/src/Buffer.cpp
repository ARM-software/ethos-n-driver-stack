//
// Copyright © 2018-2022 Arm Limited.
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

Buffer::Buffer(int fd, uint32_t size, const std::string& device)
    : bufferImpl{ std::make_unique<BufferImpl>(fd, size, device) }
{
    if (profiling::g_CurrentConfiguration.m_EnableProfiling)
    {
        RecordLifetimeEvent(this, profiling::g_BufferToLifetimeEventId,
                            profiling::ProfilingEntry::Type::TimelineEventStart,
                            profiling::ProfilingEntry::MetadataCategory::BufferLifetime);
    }
}

Buffer::Buffer(int fd, uint32_t size)
    : Buffer(fd, size, DEVICE_NODE)
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

DataFormat Buffer::GetDataFormat()
{
    if (!bufferImpl)
    {
        throw std::runtime_error("Unable to GetDataFormat as BufferImpl is null");
    }
    return bufferImpl->GetDataFormat();
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
