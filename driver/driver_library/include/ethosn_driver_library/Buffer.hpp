//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace ethosn
{
namespace driver_library
{

class ProcMemAllocator;

enum class MemType
{
    ALLOCATE,
    IMPORT,
    NONE
};

struct IntermediateBufferReq
{

    IntermediateBufferReq()
        : type(MemType::ALLOCATE)
        , fd(0)
        , flags(0)
    {}

    IntermediateBufferReq(const MemType type, const uint32_t fd, const uint32_t flags)
        : type(type)
        , fd(fd)
        , flags(flags)
    {}

    IntermediateBufferReq(const MemType type)
        : type(type)
        , fd(0)
        , flags(0)
    {}

    MemType type;
    uint32_t fd;
    uint32_t flags;
};

class Buffer
{
public:
    Buffer(Buffer&& otherBuffer);

    ~Buffer();

    explicit operator bool() const
    {
        return bufferImpl != nullptr;
    }

    // Returns the size of the buffer.
    uint32_t GetSize();

    // Returns the raw file descriptor.
    const int& GetBufferHandle() const;

    // Syncs for cpu and returns a pointer to the mapped buffer.
    // To be used together with Unmap().
    uint8_t* Map();

    // Unmaps the buffer and syncs for device.
    // To be used together with Map().
    void Unmap();

private:
    friend ProcMemAllocator;
    class BufferImpl;

    Buffer(std::unique_ptr<BufferImpl> otherBufferImpl);

    std::unique_ptr<BufferImpl> bufferImpl;
};
}    // namespace driver_library
}    // namespace ethosn
