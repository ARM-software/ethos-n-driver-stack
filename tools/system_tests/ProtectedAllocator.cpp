//
// Copyright © 2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "ProtectedAllocator.hpp"

#include <sstream>
#include <stdexcept>
#include <string>

namespace ethosn
{
namespace system_tests
{

ProtectedAllocator::ProtectedAllocator()
    : m_DmaBufDev(new DmaBufferDevice("/dev/ethosn-tzmp1-test-module"))
{}

void* ProtectedAllocator::allocate(size_t size, size_t)
{
    if (size == 0)
    {
        throw std::invalid_argument("Invalid zero size allocation");
    }

    std::unique_ptr<DmaBuffer> dataDmaBuf = std::make_unique<DmaBuffer>(m_DmaBufDev, size);
    const int fd                          = dataDmaBuf->GetFd();
    auto pos                              = m_Allocations.emplace(fd, Allocation{ std::move(dataDmaBuf), fd });
    return static_cast<void*>(&(pos.first->second.m_Fd));
}

void ProtectedAllocator::free(void* ptr)
{
    if (ptr == nullptr)
    {
        throw std::invalid_argument("ptr is null");
    }

    const int fd = *static_cast<int*>(ptr);
    // To detect double free issues an exception will be thrown for unknown allocations
    auto allocation = m_Allocations.find(fd);
    if (allocation == m_Allocations.end())
    {
        std::ostringstream oss;
        oss << ptr;
        throw std::invalid_argument("No allocation exists for ptr: " + oss.str() + " fd: " + std::to_string(fd));
    }

    m_Allocations.erase(allocation);
}

armnn::MemorySource ProtectedAllocator::GetMemorySourceType()
{
    return armnn::MemorySource::DmaBufProtected;
}

void ProtectedAllocator::PopulateData(void* ptr, const uint8_t* inData, size_t len)
{
    if (ptr == nullptr)
    {
        throw std::invalid_argument("ptr is null");
    }

    if (inData == nullptr)
    {
        throw std::invalid_argument("inData is null");
    }

    // Throwing exception rather than doing nothing for zero length to easier
    // detect incorrect usage of the allocator.
    if (len == 0U)
    {
        throw std::invalid_argument("Zero length population not allowed");
    }

    const int fd = *static_cast<int*>(ptr);
    m_Allocations.at(fd).m_DmaBuf->PopulateData(inData, len);
}

void ProtectedAllocator::RetrieveData(void* ptr, uint8_t* outData, size_t len)
{
    if (ptr == nullptr)
    {
        throw std::invalid_argument("ptr is null");
    }

    if (outData == nullptr)
    {
        throw std::invalid_argument("outData is null");
    }

    // Throwing exception rather than doing nothing for zero length to easier
    // detect incorrect usage of the allocator.
    if (len == 0U)
    {
        throw std::invalid_argument("Zero length retrieve not allowed");
    }

    const int fd = *static_cast<int*>(ptr);
    m_Allocations.at(fd).m_DmaBuf->RetrieveData(outData, len);
}

}    // namespace system_tests
}    // namespace ethosn
