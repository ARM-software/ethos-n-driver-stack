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

    // Ethos-N allocates the buffer.
    Buffer(uint32_t size, DataFormat format);
    Buffer(uint32_t size, DataFormat format, const std::string& device);

    // Data is copied from src into the buffer.
    // This won't work for output buffers if using kmod backend unless any access after creation is via GetMappedBuffer().
    // The input data will only be copied-in at creation time, and the output data won't be copied-out to
    // the original memory location.
    // The important thing is that we make sure the latest input data is copied into our copy before inference and
    // that output data is copied to the user-space location after inference.
    // For input data, it would be fine to document that the user is supposed to fill data with our API and not via the
    // original pointer. Otherwise, there's no guarantee the inference will run with the correct data.
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
    uint8_t* GetMappedBuffer();

private:
    class BufferImpl;
    std::unique_ptr<BufferImpl> bufferImpl;
};
}    // namespace driver_library
}    // namespace ethosn
