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
        : m_MappedData(nullptr)
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
            close(ethosnFd);
            throw std::runtime_error(std::string("Wrong kernel module version\n"));
        }

        m_BufferFd = ioctl(ethosnFd, ETHOSN_IOCTL_CREATE_BUFFER, &outputBufReq);
        int err    = errno;
        close(ethosnFd);
        if (m_BufferFd < 0)
        {
            throw std::runtime_error(std::string("Failed to create buffer: ") + strerror(err));
        }
    }

    BufferImpl(const uint8_t* src, uint32_t size, DataFormat format, const std::string& device)
        : BufferImpl(size, format, device)
    {
        uint8_t* data = Map();
        std::copy_n(src, size, data);
        Unmap();
    }

    ~BufferImpl()
    {
        try
        {
            // It can throw and it needs to close the file descriptor
            Unmap();
        }
        catch (...)
        {
            assert(false);
        }
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

    uint8_t* Map()
    {
        int ret = ioctl(m_BufferFd, ETHOSN_IOCTL_SYNC_FOR_CPU);
        if (ret < 0)
        {
            throw std::runtime_error(std::string("Failed to sync for cpu: ") + strerror(errno));
        }

        if (m_MappedData)
        {
            return m_MappedData;
        }

        m_MappedData = reinterpret_cast<uint8_t*>(mmap(nullptr, m_Size, PROT_WRITE, MAP_SHARED, m_BufferFd, 0));
        if (m_MappedData == MAP_FAILED)
        {
            m_MappedData = nullptr;
            throw std::runtime_error(std::string("Failed to map memory: ") + strerror(errno));
        }

        return m_MappedData;
    }

    void Unmap()
    {
        if (!m_MappedData)
        {
            return;
        }

        munmap(m_MappedData, m_Size);
        m_MappedData = nullptr;
        int ret      = ioctl(m_BufferFd, ETHOSN_IOCTL_SYNC_FOR_DEVICE);
        if (ret < 0)
        {
            throw std::runtime_error(std::string("Failed to sync for device: ") + strerror(errno));
        }
    }

private:
    int m_BufferFd;
    uint8_t* m_MappedData;
    uint32_t m_Size;
    DataFormat m_Format;
};

}    // namespace driver_library
}    // namespace ethosn
