//
// Copyright © 2018-2023 Arm Limited.
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
    BufferImpl(uint32_t size, int allocatorFd)
        : m_MappedData(nullptr)
        , m_Size(size)
    {
        const ethosn_buffer_req bufferReq = {
            size,
            MB_RDWR,
        };

        m_BufferFd = ioctl(allocatorFd, ETHOSN_IOCTL_CREATE_BUFFER, &bufferReq);
        int err    = errno;
        if (m_BufferFd < 0)
        {
            throw std::runtime_error(std::string("Failed to create buffer: ") + strerror(err));
        }
    }

    BufferImpl(const uint8_t* src, uint32_t size, int allocatorFd)
        : BufferImpl(size, allocatorFd)
    {
        uint8_t* data = Map();
        std::copy_n(src, size, data);
        Unmap();
    }

    BufferImpl(int fd, uint32_t size, int allocatorFd)
        : m_MappedData(nullptr)
        , m_Size(size)
    {
        const ethosn_dma_buf_req importedBufferReq = {
            static_cast<__u32>(fd),
            O_RDWR | O_CLOEXEC,
            size,
        };

        m_BufferFd = ioctl(allocatorFd, ETHOSN_IOCTL_IMPORT_BUFFER, &importedBufferReq);
        int err    = errno;
        if (m_BufferFd < 0)
        {
            throw std::runtime_error(std::string("Failed to import  buffer: ") + strerror(err));
        }
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

    const int& GetBufferHandle() const
    {
        return m_BufferFd;
    }

    uint8_t* Map()
    {
        if (m_MappedData)
        {
            return m_MappedData;
        }

        m_MappedData =
            reinterpret_cast<uint8_t*>(mmap(nullptr, m_Size, PROT_WRITE | PROT_READ, MAP_SHARED, m_BufferFd, 0));
        if (m_MappedData == MAP_FAILED)
        {
            m_MappedData = nullptr;
            throw std::runtime_error(std::string("Failed to map memory: ") + strerror(errno));
        }

        int ret = ioctl(m_BufferFd, ETHOSN_IOCTL_SYNC_FOR_CPU);
        if (ret < 0)
        {
            throw std::runtime_error(std::string("Failed to sync for cpu: ") + strerror(errno));
        }

        return m_MappedData;
    }

    void Unmap()
    {
        if (!m_MappedData)
        {
            return;
        }

        int ret = ioctl(m_BufferFd, ETHOSN_IOCTL_SYNC_FOR_DEVICE);
        if (ret < 0)
        {
            throw std::runtime_error(std::string("Failed to sync for device: ") + strerror(errno));
        }

        munmap(m_MappedData, m_Size);
        m_MappedData = nullptr;
    }

private:
    int m_BufferFd;
    uint8_t* m_MappedData;
    uint32_t m_Size;
};

}    // namespace driver_library
}    // namespace ethosn
