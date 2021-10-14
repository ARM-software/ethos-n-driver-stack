//
// Copyright © 2018-2021 Arm Limited.
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
    Buffer(Buffer&& buffer);

    // FIXME: Need an API where an external buffer is provided, without any copy. (Jira NNXSW-605)

    // Device allocates the buffer.
    Buffer(uint32_t size, DataFormat format);
    Buffer(uint32_t size, DataFormat format, const std::string& device);

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
    // FIXME: Fix as part of Jira NNXSW-610 - Refactor Driver Library
    Buffer(uint8_t* src, uint32_t size, DataFormat format);
    Buffer(uint8_t* src, uint32_t size, DataFormat format, const std::string& device);

    ~Buffer();

    // Returns the size of the buffer.
    uint32_t GetSize();

    // Returns the data format of the buffer.
    DataFormat GetDataFormat();

    // Returns the raw file descriptor.
    const int& GetBufferHandle() const;

    // Returns a pointer to the mapped kernel buffer.
    // This is deprecated, use Map()/Unmap() instead.
    uint8_t* GetMappedBuffer();

    // Syncs for cpu and returns a pointer to the mapped buffer.
    // To be used together with Unmap().
    uint8_t* Map();

    // Unmaps the buffer and syncs for device.
    // To be used together with Map().
    void Unmap();

private:
    class BufferImpl;
    std::unique_ptr<BufferImpl> bufferImpl;
};
}    // namespace driver_library
}    // namespace ethosn
