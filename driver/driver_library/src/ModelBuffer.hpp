//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../include/ethosn_driver_library/Buffer.hpp"
#include <ethosn_utils/Macros.hpp>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#ifdef _WIN32
#include <io.h>
// Alias for ssize_t so that we can write code with the correct types for Linux, and it also
// works for Windows. On Windows, read() and write() return int whereas on Linux they return
// ssize_t.
using ssize_t = int;
#endif
#if defined(__unix__)
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace ethosn
{
namespace driver_library
{

class Buffer::BufferImpl
{
public:
    BufferImpl(uint32_t size, const std::string&)
        : m_Data(std::make_unique<uint8_t[]>(size))
        , m_Ptr(&m_Data[0])
        , m_Size(size)
        , m_Fd({ false, -1 })
    {}

    BufferImpl(uint32_t size, int)
        : m_Data(std::make_unique<uint8_t[]>(size))
        , m_Ptr(&m_Data[0])
        , m_Size(size)
        , m_Fd({ false, -1 })
    {}

    BufferImpl(const uint8_t* src, uint32_t size, const std::string&)
        : m_Data(std::make_unique<uint8_t[]>(size))
        , m_Ptr(&m_Data[0])
        , m_Size(size)
        , m_Fd({ false, -1 })
    {
        std::copy_n(src, size, m_Ptr);
    }

    BufferImpl(const uint8_t* src, uint32_t size, int)
        : m_Data(std::make_unique<uint8_t[]>(size))
        , m_Ptr(&m_Data[0])
        , m_Size(size)
        , m_Fd({ false, -1 })
    {
        std::copy_n(src, size, m_Ptr);
    }

    // Overload for creating a buffer with a file descriptor, i.e. an "imported buffer"
    // This is useful for testing importing file descriptors without needing the kernel module
    BufferImpl(int fd, uint32_t size, const std::string&)
        : m_Data(std::make_unique<uint8_t[]>(size))
        , m_Ptr(&m_Data[0])
        , m_Size(size)
        , m_Fd({ true, fd })
    {
        // Apart from storing the FD, we don't populate the m_Data yet.
        // We defer this until Map(), as this is more similar to how a "real" imported buffer
        // would behave.
    }

    BufferImpl(int fd, uint32_t size, int)
        : m_Data(std::make_unique<uint8_t[]>(size))
        , m_Ptr(&m_Data[0])
        , m_Size(size)
        , m_Fd({ true, fd })
    {
        // Apart from storing the FD, we don't populate the m_Data yet.
        // We defer this until Map(), as this is more similar to how a "real" imported buffer
        // would behave.
    }

    uint32_t GetSize()
    {
        return m_Size;
    }

    // Not used for the model backend
    const int& GetBufferHandle() const
    {
        static int dummy = 0;
        return dummy;
    }

    uint8_t* Map()
    {
        if (!m_Fd.first)
        {
            return m_Ptr;
        }

        // If we're using a file descriptor we need to update our data to reflect
        // any changes in the fd contents.
        if (lseek(m_Fd.second, 0, SEEK_SET) < 0)
        {
            int err = errno;
            throw std::runtime_error("ModelBuffer lseek failed. errno = " + std::to_string(err) + ": " +
                                     std::strerror(err));
        }
        ssize_t bytesRead = read(m_Fd.second, m_Data.get(), m_Size);
        if (bytesRead < 0)
        {
            int err = errno;
            throw std::runtime_error("ModelBuffer read failed. errno = " + std::to_string(err) + ": " +
                                     std::strerror(err));
        }
        if (static_cast<size_t>(bytesRead) != m_Size)
        {
            throw std::runtime_error("ModelBuffer read failed. Expected return value of " + std::to_string(m_Size) +
                                     " but got " + std::to_string(bytesRead));
        }
        if (lseek(m_Fd.second, 0, SEEK_SET) < 0)
        {
            int err = errno;
            throw std::runtime_error("ModelBuffer lseek failed. errno = " + std::to_string(err) + ": " +
                                     std::strerror(err));
        }
        return m_Ptr;
    }

    void Unmap()
    {
        if (!m_Fd.first)
        {
            return;
        }

        // If we're using a file descriptor we need to write the result back to that file
        // Write from the beginning of the file
        if (lseek(m_Fd.second, 0, SEEK_SET) < 0)
        {
            int err = errno;
            throw std::runtime_error("ModelBuffer lseek failed. errno = " + std::to_string(err) + ": " +
                                     std::strerror(err));
        }
        ssize_t bytesWritten = write(m_Fd.second, m_Data.get(), m_Size);
        if (bytesWritten < 0)
        {
            int err = errno;
            throw std::runtime_error("ModelBuffer write failed. errno = " + std::to_string(errno) + ": " +
                                     std::strerror(err));
        }
        if (static_cast<size_t>(bytesWritten) != m_Size)
        {
            throw std::runtime_error("ModelBuffer write failed. Expected return value of " + std::to_string(m_Size) +
                                     " but got " + std::to_string(bytesWritten));
        }
        if (lseek(m_Fd.second, 0, SEEK_SET) < 0)
        {
            int err = errno;
            throw std::runtime_error("ModelBuffer lseek failed. errno = " + std::to_string(err) + ": " +
                                     std::strerror(err));
        }
    }

private:
    std::unique_ptr<uint8_t[]> m_Data;
    uint8_t* m_Ptr;
    uint32_t m_Size;
    std::pair<bool, int> m_Fd;
};

}    // namespace driver_library
}    // namespace ethosn
