//
// Copyright © 2022-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_driver_library/ProcMemAllocator.hpp"

#include "NetworkImpl.hpp"
#if defined(TARGET_KMOD)
#include "KmodBuffer.hpp"
#include "KmodNetwork.hpp"
#elif defined(TARGET_MODEL)
#include "ModelBuffer.hpp"
#include "ModelNetwork.hpp"
#endif

#include <ethosn_utils/Macros.hpp>
#if defined(__unix__)
#include <uapi/ethosn.h>

#include <errno.h>
#include <fcntl.h>
#include <stdexcept>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace ethosn
{
namespace driver_library
{

ProcMemAllocator::ProcMemAllocator(const char* device, bool is_protected)
    : m_isProtected(is_protected)
{
#ifdef TARGET_KMOD
    int ethosnFd = open(device, O_RDONLY);
    if (ethosnFd < 0)
    {
        throw std::runtime_error(std::string("Unable to open " + std::string(device) + ": ") + strerror(errno));
    }

    // Check compatibility between driver library and the kernel
    try
    {
        if (!VerifyKernel(device))
        {
            throw std::runtime_error(std::string("Wrong kernel module version\n"));
        }
    }
    catch (const std::runtime_error& error)
    {
        close(ethosnFd);
        throw;
    }

    struct ethosn_proc_mem_allocator_req proc_mem_req;
    proc_mem_req.is_protected = is_protected;

    m_AllocatorFd = ioctl(ethosnFd, ETHOSN_IOCTL_CREATE_PROC_MEM_ALLOCATOR, &proc_mem_req);
    int err       = errno;
    close(ethosnFd);
    if (m_AllocatorFd < 0)
    {
        throw std::runtime_error(std::string("Failed to create process memory allocator: ") + strerror(err));
    }
#else
    ETHOSN_UNUSED(is_protected);
    m_AllocatorFd = -1;
#endif
    m_deviceId = device;
}
ProcMemAllocator::ProcMemAllocator(const char* device)
    : ProcMemAllocator(device, false)
{}

ProcMemAllocator::ProcMemAllocator()
    : ProcMemAllocator(DEVICE_NODE, false)
{}

ProcMemAllocator::ProcMemAllocator(bool is_protected)
    : ProcMemAllocator(DEVICE_NODE, is_protected)
{}

ProcMemAllocator::ProcMemAllocator(ProcMemAllocator&& otherAllocator)
    : m_AllocatorFd(otherAllocator.m_AllocatorFd)
    , m_deviceId(otherAllocator.m_deviceId)
    , m_isProtected(otherAllocator.m_isProtected)
{
    // Invalidate fd of other allocator to prevent early closing
    otherAllocator.m_AllocatorFd = -1;
}

ProcMemAllocator::~ProcMemAllocator()
{
#ifdef TARGET_KMOD
    if (m_AllocatorFd > 0)
    {
        close(m_AllocatorFd);
    }
#endif
}

Buffer ProcMemAllocator::CreateBuffer(uint32_t size)
{
    return Buffer(std::make_unique<Buffer::BufferImpl>(size, m_AllocatorFd));
}

Buffer ProcMemAllocator::CreateBuffer(const uint8_t* src, uint32_t size)
{
    return Buffer(std::make_unique<Buffer::BufferImpl>(src, size, m_AllocatorFd));
}

void CheckImportMemorySize(int fd, uint32_t size)
{
    int64_t memSize = lseek(fd, 0, SEEK_END);

    if (memSize < 0)
    {
        throw std::runtime_error(std::string("Failed to get memory size from fd. ") + strerror(errno));
    }

    if (lseek(fd, 0, SEEK_SET))
    {
        throw std::runtime_error(std::string("Failed to seek start of file from fd. ") + strerror(errno));
    }

    if (static_cast<uint64_t>(memSize) < static_cast<uint64_t>(size))
    {
        throw std::runtime_error("Source buffer is smaller than the size specified");
    }
}

Buffer ProcMemAllocator::ImportBuffer(int fd, uint32_t size)
{
    CheckImportMemorySize(fd, size);

    return Buffer(std::make_unique<Buffer::BufferImpl>(fd, size, m_AllocatorFd));
}

Network ProcMemAllocator::CreateNetwork(const char* compiledNetworkData,
                                        size_t compiledNetworkSize,
                                        const IntermediateBufferReq& desc)
{
#if !defined(TARGET_KMOD)
    ETHOSN_UNUSED(desc);
#endif
    return Network(
#if defined(TARGET_MODEL)
        std::make_unique<ModelNetworkImpl>(compiledNetworkData, compiledNetworkSize)
#elif defined(TARGET_KMOD)
        std::make_unique<KmodNetworkImpl>(compiledNetworkData, compiledNetworkSize, m_AllocatorFd, desc)
#elif defined(TARGET_DUMPONLY)
        std::make_unique<NetworkImpl>(compiledNetworkData, compiledNetworkSize, false)
#else
#error "Unknown target backend."
#endif
    );
}

std::string ProcMemAllocator::GetDeviceId()
{
    return m_deviceId;
}

bool ProcMemAllocator::GetProtected()
{
    return m_isProtected;
}

}    // namespace driver_library
}    // namespace ethosn
