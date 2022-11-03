//
// Copyright © 2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Buffer.hpp"
#include "Network.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace ethosn
{
namespace driver_library
{

class ProcMemAllocator
{
public:
    ProcMemAllocator();
    ProcMemAllocator(const std::string& device);

    // Disable copy and assignment to prevent accidental duplication
    // as only one instance of this class should exist per process
    ProcMemAllocator(const ProcMemAllocator&) = delete;
    ProcMemAllocator& operator=(const ProcMemAllocator&) = delete;

    // Buffer Creation
    Buffer CreateBuffer(uint32_t size, DataFormat format);
    Buffer CreateBuffer(const uint8_t* src, uint32_t size, DataFormat format);
    Buffer ImportBuffer(int fd, uint32_t size);

    // Network Creation
    Network CreateNetwork(const char* compiledNetworkData, size_t compiledNetworkSize);

    std::string GetDeviceId();

    ~ProcMemAllocator();

private:
    int m_AllocatorFd;
    std::string m_deviceId;
};

}    // namespace driver_library
}    // namespace ethosn
