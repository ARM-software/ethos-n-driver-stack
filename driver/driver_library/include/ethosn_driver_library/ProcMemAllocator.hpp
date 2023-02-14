//
// Copyright © 2022-2023 Arm Limited.
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
    ProcMemAllocator(const char* device);

    ProcMemAllocator(bool is_protected);
    ProcMemAllocator(const char* device, bool is_protected);

    ProcMemAllocator(ProcMemAllocator&& otherAllocator);

    // Disable copy and assignment to prevent accidental duplication
    // as only one instance of this class should exist per process
    ProcMemAllocator(const ProcMemAllocator&) = delete;
    ProcMemAllocator& operator=(const ProcMemAllocator&) = delete;

    // Buffer Creation
    Buffer CreateBuffer(uint32_t size, DataFormat format);

    /// Create buffer filled with the data from src. The buffer's data can later be accessed via the
    /// buffer's Map() function
    ///
    /// ProcMemAllocator processMemAllocator;
    /// Buffer input = processMemAllocator.CreateBuffer(mem, size, format);
    ///
    /// ... inference is executed ...
    ///
    /// uint8_t data = input.Map();
    ///
    /// ... fill in more data ...
    ///
    /// input.Unmap();
    ///
    /// ... another inference is executed ...
    ///
    Buffer CreateBuffer(const uint8_t* src, uint32_t size, DataFormat format);

    /// Import dma-buf based buffer to be used by the device
    Buffer ImportBuffer(int fd, uint32_t size);

    /// Loads a Network into the driver so that it is ready for inferences.
    /// The Compiled Network data should be obtained from the Support Library, by serializing the
    /// ethosn::support_library::CompiledNetwork object (by calling its Serialize() method).
    /// This data is copied into the driver where necessary and does not need to kept alive by the caller.
    /// @throws CompiledNetworkException if the given Compiled Network data is not valid.
    Network CreateNetwork(const char* compiledNetworkData,
                          size_t compiledNetworkSize,
                          const IntermediateBufferReq& desc = { MemType::ALLOCATE, 0, 0 });

    std::string GetDeviceId();

    bool GetProtected();

    ~ProcMemAllocator();

private:
    int m_AllocatorFd;
    std::string m_deviceId;
    const bool m_isProtected;
};

}    // namespace driver_library
}    // namespace ethosn
