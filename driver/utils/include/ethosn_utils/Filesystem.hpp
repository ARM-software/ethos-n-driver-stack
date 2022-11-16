//
// Copyright © 2020,2022 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if defined(__unix__)
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#elif defined(_MSC_VER)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <io.h>
#endif
#include <cstring>

namespace ethosn
{
namespace utils
{

inline bool MakeDirectory(const char* dir)
{
#if defined(__unix__)
    return mkdir(dir, 0777) == 0;
#elif defined(_MSC_VER)
    return CreateDirectory(dir, nullptr);
#else
    return false;
#endif
}

class Fd
{
public:
    Fd()
        : m_Fd(-EINVAL)
    {}
    explicit Fd(int fd)
        : m_Fd(fd)
    {}
    Fd(const Fd&) = delete;
    Fd(Fd&& other)
        : Fd(other.m_Fd)
    {
        other.m_Fd = -EINVAL;
    }

    Fd& operator=(const Fd&) = delete;
    Fd& operator             =(Fd&& rhs)
    {
        if (m_Fd > 0)
        {
            close(m_Fd);
        }
        m_Fd     = rhs.m_Fd;
        rhs.m_Fd = -EINVAL;
        return *this;
    }

    ~Fd()
    {
        if (m_Fd > 0)
        {
            close(m_Fd);
        }
    }

    const int& GetRawFd() const
    {
        return m_Fd;
    }

    template <typename... Ts>
    int ioctl(const int cmd, Ts&&... ts) const
    {
        return ::ioctl(m_Fd, cmd, std::forward<Ts>(ts)...);
    }

    template <typename... Ts>
    int CheckedIoctl(const int cmd, Ts&&... ts) const
    {
        int result = ::ioctl(m_Fd, cmd, std::forward<Ts>(ts)...);
        if (result < 0)
        {
            int err = errno;
            throw std::runtime_error(std::string("errno: ") + std::strerror(err));
        }
        return result;
    }

private:
    int m_Fd;
};

#if defined(__unix__)

template <typename T, size_t N>
class MMap
{
public:
    MMap(const Fd& fd, const int flags = PROT_READ | PROT_WRITE)
    {
        m_Data = reinterpret_cast<T*>(mmap(nullptr, N * sizeof(T), flags, MAP_SHARED, fd.GetRawFd(), 0));
    }

    MMap(const MMap&) = delete;
    MMap& operator=(const MMap&) = delete;

    ~MMap()
    {
        munmap(m_Data, N);
    }

    const T& operator[](const size_t i) const
    {
        return m_Data[i];
    }

    T* begin()
    {
        return m_Data;
    }

    T* end()
    {
        return m_Data + N;
    }

    bool IsValid() const
    {
        return m_Data != MAP_FAILED;
    }

private:
    T* m_Data;
};

#endif

}    // namespace utils
}    // namespace ethosn
