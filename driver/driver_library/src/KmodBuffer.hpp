//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../include/ethosn_driver_library/Buffer.hpp"
#include "../include/ethosn_driver_library/Network.hpp"
#include "Utils.hpp"

#include <uapi/ethosn.h>

#include <algorithm>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <sys/ioctl.h>
#include <sys/mman.h>
#if defined(__unix__)
#include <unistd.h>
#endif

namespace ethosn
{
namespace driver_library
{

class Buffer::BufferImpl
{
public:
    BufferImpl(uint32_t size, DataFormat format, const std::string& device)
        : m_Data(nullptr)
        , m_Size(size)
        , m_Format(format)
    {
        const ethosn_buffer_req outputBufReq = {
            size,
            MB_RDWR,
        };

        int ethosnFd = open(device.c_str(), O_RDONLY);
        if (ethosnFd < 0)
        {
            throw std::runtime_error(std::string("Unable to open " + device + ": ") + strerror(errno));
        }

        // Check compatibility between driver library and the kernel
        if (!VerifyKernel(device))
        {
            throw std::runtime_error(std::string("Wrong kernel module version\n"));
        }

        m_BufferFd = ioctl(ethosnFd, ETHOSN_IOCTL_CREATE_BUFFER, &outputBufReq);
        int err    = errno;
        close(ethosnFd);
        if (m_BufferFd < 0)
        {
            throw std::runtime_error(std::string("Failed to create buffer: ") + strerror(err));
        }

        m_Data = reinterpret_cast<uint8_t*>(mmap(nullptr, size, PROT_WRITE, MAP_SHARED, m_BufferFd, 0));
        if (m_Data == MAP_FAILED)
        {
            err = errno;
            close(m_BufferFd);
            throw std::runtime_error(std::string("Failed to map memory: ") + strerror(err));
        }
    }

    BufferImpl(uint8_t* src, uint32_t size, DataFormat format, const std::string& device)
        : BufferImpl(size, format, device)
    {
        std::copy_n(src, size, m_Data);
    }

    ~BufferImpl()
    {
        munmap(m_Data, m_Size);
        close(m_BufferFd);
    }

    uint32_t GetSize()
    {
        return m_Size;
    }

    DataFormat GetDataFormat()
    {
        return m_Format;
    }

    const int& GetBufferHandle() const
    {
        return m_BufferFd;
    }

    uint8_t* GetMappedBuffer()
    {
        return m_Data;
    }

private:
    int m_BufferFd;
    uint8_t* m_Data;
    uint32_t m_Size;
    DataFormat m_Format;
};

}    // namespace driver_library
}    // namespace ethosn
