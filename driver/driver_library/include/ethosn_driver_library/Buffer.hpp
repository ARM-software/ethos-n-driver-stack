//
// Copyright © 2018-2022 Arm Limited.
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

// Buffer formats.
// (N = batch, H = height, W = width, C = channel)
enum class DataFormat
{
    NHWC,
    NHWCB
};

class Buffer
{
public:
    Buffer(Buffer&& otherBuffer);

    // Device allocates the buffer.
    Buffer(uint32_t size, DataFormat format);
    Buffer(uint32_t size, DataFormat format, const std::string& device);

    // Device imports the buffer.
    Buffer(int fd, uint32_t size);
    Buffer(int fd, uint32_t size, const std::string& device);

    // Data is copied from src into the buffer. Any access after creation is via Map()/Unmap():
    //
    // Buffer input(mem, size, format);
    //
    // ... inference is executed ...
    //
    // uint8_t data = input.Map();
    //
    // ... fill in more data ...
    //
    // input.Unmap();
    //
    // ... another inference is executed ...
    //
    Buffer(const uint8_t* src, uint32_t size, DataFormat format);
    Buffer(const uint8_t* src, uint32_t size, DataFormat format, const std::string& device);

    ~Buffer();

    explicit operator bool() const
    {
        return bufferImpl != nullptr;
    }

    // Returns the size of the buffer.
    uint32_t GetSize();

    // Returns the data format of the buffer.
    DataFormat GetDataFormat();

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
