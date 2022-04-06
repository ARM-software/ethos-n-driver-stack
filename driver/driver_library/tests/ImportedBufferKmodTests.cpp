//
// Copyright © 2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_driver_library/Buffer.hpp"
#include "../include/ethosn_driver_library/Network.hpp"
#include "../src/Utils.hpp"

#include <linux/version.h>
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 6, 0)
#include <linux/dma-heap.h>
#include <uapi/ethosn.h>

#include <catch.hpp>

#include <chrono>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <thread>
#include <unistd.h>

using namespace ethosn::driver_library;

namespace
{

bool IsDeviceStatusOkay(const std::string& filePath)
{
    std::ifstream fileStream(filePath);
    if (!fileStream.is_open())
        return false;
    std::string isOkay;
    getline(fileStream, isOkay);
    constexpr char okay[] = "okay";
    return isOkay.compare(0, strlen(okay), okay) == 0;
}

bool IsCore0IommuAvailable(const std::string& filePath)
{
    std::ifstream fileStream(filePath + "/core0/iommus");
    return fileStream.is_open();
}

bool IsKernelVersionHigherOrEqualTo(int kernelVersion, int kernelPatchLevel)
{
    utsname linuxReleaseInfo = {};
    REQUIRE(!uname(&linuxReleaseInfo));
    int actualKernelVersion, actualKernelPatchLevel;
    REQUIRE(sscanf(linuxReleaseInfo.release, "%d.%d", &actualKernelVersion, &actualKernelPatchLevel) == 2);
    return ((kernelVersion < actualKernelVersion) ||
            ((kernelVersion == actualKernelVersion) && (kernelPatchLevel <= actualKernelPatchLevel)));
}

bool IsNpuCoreBehindIommus()
{
    constexpr char deviceTreePath[] = "/proc/device-tree";
    DIR* dir                        = opendir(deviceTreePath);
    REQUIRE(dir != nullptr);
    dirent* ent;
    constexpr char deviceBindingPrefix[] = "ethosn@";
    bool coreFound                       = false;
    while ((ent = readdir(dir)) != nullptr)
    {
        const std::string dirName(ent->d_name);
        if (dirName.find(deviceBindingPrefix, 0, strlen(deviceBindingPrefix)) == std::string::npos)
        {
            continue;
        }
        const std::string devicePath = deviceTreePath + std::string("/") + dirName;
        if (!IsDeviceStatusOkay(devicePath + "/status"))
        {
            continue;
        }
        if (!IsDeviceStatusOkay(devicePath + "/core0/status"))
        {
            continue;
        }
        coreFound = true;
        if (IsCore0IommuAvailable(devicePath))
        {
            break;
        }
    }
    if (ent == nullptr)
    {
        if (coreFound)
        {
            INFO("No NPU core is behind a IOMMU.");
        }
        else
        {
            INFO("No \"ethosn@xxxxxxx\" device tree node was found.");
        }
    }
    closedir(dir);
    return ent != nullptr;
}

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
    if (!IsKernelVersionHigherOrEqualTo(5, 6))
    {
        INFO("Kernel version lower than 5.6.");
        INFO("No tests will be performed.");
        return;
    }

    // check that NPU core is behind a IOMMU.
    if (!IsNpuCoreBehindIommus())
    {
        INFO("No tests will be performed.");
        return;
    }

    {
        constexpr uint32_t bufSize = 1024;
        DmaHeapBuffer dmaHeapData(bufSize);
        {
            // Create Simple buffer
            Buffer test_buffer(dmaHeapData.GetRawFd(), bufSize);

            // Verify Buffer properties
            REQUIRE(test_buffer.GetSize() == bufSize);
        }
    }
}

TEST_CASE("ImportedBufferSource")
{
    // check the kernel version to be higher or equal to 5.6.
    if (!IsKernelVersionHigherOrEqualTo(5, 6))
    {
        INFO("Kernel version lower than 5.6.");
        INFO("No tests will be performed.");
        return;
    }

    // check that NPU core is behind a IOMMU.
    if (!IsNpuCoreBehindIommus())
    {
        INFO("No tests will be performed.");
        return;
    }

    {
        uint8_t test_src[]     = "This is a test source data";
        uint32_t test_src_size = sizeof(test_src);
        DmaHeapBuffer dmaHeapData(test_src_size);
        {
            // Create a buffer with test source data
            Buffer test_buffer(dmaHeapData.GetRawFd(), test_src_size);
            uint8_t* data = test_buffer.Map();
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
    if (!IsKernelVersionHigherOrEqualTo(5, 6))
    {
        INFO("Kernel version lower than 5.6.");
        INFO("No tests will be performed.");
        return;
    }

    // check that NPU core is behind a IOMMU.
    if (!IsNpuCoreBehindIommus())
    {
        INFO("No tests will be performed.");
        return;
    }

    {
        uint8_t test_src[]     = "This is a test source data";
        uint32_t test_src_size = sizeof(test_src);
        DmaHeapBuffer dmaHeapData(test_src_size);
        {
            // Create a buffer with test source data
            Buffer test_buffer(dmaHeapData.GetRawFd(), test_src_size);
            uint8_t* data = test_buffer.Map();
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
#endif
