//
// Copyright © 2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_driver_library/Buffer.hpp"
#include "../include/ethosn_driver_library/Network.hpp"
#include "../include/ethosn_driver_library/ProcMemAllocator.hpp"
#include "../src/Utils.hpp"

#include <ethosn_utils/KernelUtils.hpp>

#include <linux/version.h>
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 6, 0)
#include <linux/dma-heap.h>
#include <uapi/ethosn.h>

#include <catch.hpp>

#include <cstring>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

using namespace ethosn::driver_library;

namespace
{
class DmaHeapBuffer
{
public:
    DmaHeapBuffer(uint64_t bufferSize)
        : m_HeapData{ bufferSize, 0, O_RDWR | O_CLOEXEC, 0 }
    {
        m_DmaHeapFd = open("/dev/dma_heap/system", O_RDONLY | O_CLOEXEC);
        REQUIRE(m_DmaHeapFd >= 0);
        int result = ::ioctl(m_DmaHeapFd, DMA_HEAP_IOCTL_ALLOC, &m_HeapData);
        if (result < 0)
        {
            int err = errno;
            FAIL(std::string("errno: ") + strerror(err));
        }
    }

    int GetRawFd()
    {
        return m_HeapData.fd;
    }

    int GetFlags()
    {
        return m_HeapData.fd_flags;
    }

    ~DmaHeapBuffer()
    {
        if (m_DmaHeapFd >= 0)
        {
            close(m_DmaHeapFd);
        }
    }

private:
    dma_heap_allocation_data m_HeapData;
    int m_DmaHeapFd;
};

}    // namespace

TEST_CASE("SimpleImportedBufferAllocation")
{
    // check the kernel version to be higher or equal to 5.6.
    if (!ethosn::utils::IsKernelVersionHigherOrEqualTo(5, 6))
    {
        INFO("Kernel version lower than 5.6.");
        INFO("No tests will be performed.");
        return;
    }

    // check that NPU core is behind a IOMMU.
    if (!ethosn::utils::IsNpuCoreBehindIommus())
    {
        INFO("No NPU core is behind a IOMMU or \"ethosn@xxxxxxx\" not found in the device tree.");
        INFO("No tests will be performed.");
        return;
    }

    ProcMemAllocator test_allocator;
    {
        constexpr uint32_t bufSize = 1024;
        DmaHeapBuffer dmaHeapData(bufSize);
        {
            // Create Simple buffer
            Buffer test_buffer = test_allocator.ImportBuffer(dmaHeapData.GetRawFd(), bufSize);

            // Verify Buffer properties
            REQUIRE(test_buffer.GetSize() == bufSize);
        }
    }
}

TEST_CASE("ImportedBufferSource")
{
    // check the kernel version to be higher or equal to 5.6.
    if (!ethosn::utils::IsKernelVersionHigherOrEqualTo(5, 6))
    {
        INFO("Kernel version lower than 5.6.");
        INFO("No tests will be performed.");
        return;
    }

    // check that NPU core is behind a IOMMU.
    if (!ethosn::utils::IsNpuCoreBehindIommus())
    {
        INFO("No NPU core is behind a IOMMU or \"ethosn@xxxxxxx\" not found in the device tree.");
        INFO("No tests will be performed.");
        return;
    }

    ProcMemAllocator test_allocator;
    {
        uint8_t test_src[]     = "This is a test source data";
        uint32_t test_src_size = sizeof(test_src);
        DmaHeapBuffer dmaHeapData(test_src_size);
        {
            // Create a buffer with test source data
            Buffer test_buffer = test_allocator.ImportBuffer(dmaHeapData.GetRawFd(), test_src_size);
            uint8_t* data      = test_buffer.Map();
            std::copy_n(test_src, test_src_size, data);
            test_buffer.Unmap();

            // Verify Buffer properties
            REQUIRE(test_buffer.GetSize() == test_src_size);
            REQUIRE(std::memcmp(test_buffer.Map(), test_src, test_src_size) == 0);
        }
    }
}

TEST_CASE("ImportedBufferMap/Unmap")
{
    // check the kernel version to be higher or equal to 5.6.
    if (!ethosn::utils::IsKernelVersionHigherOrEqualTo(5, 6))
    {
        INFO("Kernel version lower than 5.6.");
        INFO("No tests will be performed.");
        return;
    }

    // check that NPU core is behind a IOMMU.
    if (!ethosn::utils::IsNpuCoreBehindIommus())
    {
        INFO("No NPU core is behind a IOMMU or \"ethosn@xxxxxxx\" not found in the device tree.");
        INFO("No tests will be performed.");
        return;
    }

    ProcMemAllocator test_allocator;
    {
        uint8_t test_src[]     = "This is a test source data";
        uint32_t test_src_size = sizeof(test_src);
        DmaHeapBuffer dmaHeapData(test_src_size);
        {
            // Create a buffer with test source data
            Buffer test_buffer = test_allocator.ImportBuffer(dmaHeapData.GetRawFd(), test_src_size);
            uint8_t* data      = test_buffer.Map();
            std::copy_n(test_src, test_src_size, data);
            test_buffer.Unmap();

            // Verify Buffer properties
            REQUIRE(test_buffer.GetSize() == test_src_size);
            REQUIRE(std::memcmp(test_buffer.Map(), test_src, test_src_size) == 0);
            REQUIRE_NOTHROW(test_buffer.Unmap());
            // Check that it is not going to munmap twice
            REQUIRE_NOTHROW(test_buffer.Unmap());
        }
    }
}

TEST_CASE("Input/Output/IntermediateBuffers-ProcMemAllocImport")
{
    // clang-format off
    std::vector<uint8_t> serialized = {
        // 0: FourCC
        'E', 'N', 'C', 'N',

        // 4: Version (Major)
        1, 0, 0, 0,
        // 8: Version (Minor)
        0, 0, 0, 0,
        // 12: Version (Patch)
        0, 0, 0, 0,

        // 16: Constant DMA data (size)
        3, 0, 0, 0,
        // 20: Constant DMA data (values)
        1, 2, 3,

        // 23: Constant control unit data (size)
        2, 0, 0, 0,
        // 27: Constant control unit data (values)
        4, 5,

        // Input buffer infos (size)
        1, 0, 0, 0,
        // Input buffer info 0
        3, 0, 0, 0, 11, 0, 0, 0, 12, 0, 0, 0,

        // Output buffer infos (size)
        2, 0, 0, 0,
        // Output buffer info 0
        4, 0, 0, 0, 21, 0, 0, 0, 22, 0, 0, 0,
        // Output buffer info 1
        5, 0, 0, 0, 23, 0, 0, 0, 24, 0, 0, 0,

        // Constant control unit data buffer infos (size)
        1, 0, 0, 0,
        // Constant control unit data buffer info 0 (buffer 1, offset 0, size 2)
        1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,

        // Constant DMA data buffer infos (size)
        1, 0, 0, 0,
        // Constant DMA data buffer info 0 (buffer 0, offset 0, size 3)
        0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0,

        // Intermediate data buffer infos (size)
        1, 0, 0, 0,
        // Intermediate data buffer info 0
        2, 0, 0, 0, 51, 0, 0, 0, 52, 0, 0, 0,
    };
    // clang-format on

    // check the kernel version to be higher or equal to 5.6.
    if (!ethosn::utils::IsKernelVersionHigherOrEqualTo(5, 6))
    {
        INFO("Kernel version lower than 5.6.");
        INFO("No tests will be performed.");
        return;
    }

    // check that NPU core is behind a IOMMU.
    if (!ethosn::utils::IsNpuCoreBehindIommus())
    {
        INFO("No NPU core is behind a IOMMU or \"ethosn@xxxxxxx\" not found in the device tree.");
        INFO("No tests will be performed.");
        return;
    }

    constexpr uint32_t bufSize = 103;
    DmaHeapBuffer dmaHeapData(bufSize);
    IntermediateBufferReq req(MemType::IMPORT, dmaHeapData.GetRawFd(), dmaHeapData.GetFlags());
    ProcMemAllocator procMemAlloc("/dev/ethosn0");
    procMemAlloc.CreateNetwork(reinterpret_cast<const char*>(serialized.data()), serialized.size(), req);
}
#endif
