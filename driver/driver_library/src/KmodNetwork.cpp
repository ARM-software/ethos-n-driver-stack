//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "KmodNetwork.hpp"

#include "../include/ethosn_driver_library/Network.hpp"
#include "Utils.hpp"

#include <uapi/ethosn.h>

#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <sys/ioctl.h>
#include <sys/mman.h>
#if defined(__unix__)
#include <unistd.h>
#endif

using namespace ethosn;
using namespace ethosn::driver_library;

static_assert(ETHOSN_INFERENCE_SCHEDULED == static_cast<int>(InferenceResult::Scheduled),
              "ethosn.h != InferenceResult");
static_assert(ETHOSN_INFERENCE_RUNNING == static_cast<int>(InferenceResult::Running), "ethosn.h != InferenceResult");
static_assert(ETHOSN_INFERENCE_COMPLETED == static_cast<int>(InferenceResult::Completed),
              "ethosn.h != InferenceResult");
static_assert(ETHOSN_INFERENCE_ERROR == static_cast<int>(InferenceResult::Error), "ethosn.h != InferenceResult");

namespace
{

ethosn_buffer_info ToKmodBufInfo(const BufferInfo& info)
{
    return { info.m_Id, info.m_Offset, info.m_Size };
}

std::vector<ethosn_buffer_info> ToKmodBufInfos(const std::vector<BufferInfo>& infos)
{
    std::vector<ethosn_buffer_info> kmodInfos(infos.size());
    std::transform(infos.begin(), infos.end(), kmodInfos.begin(), ToKmodBufInfo);
    return kmodInfos;
}

}    // namespace

namespace ethosn
{
namespace driver_library
{

std::vector<char> GetFirmwareAndHardwareCapabilities(const std::string& device)
{
    int fd = open(device.c_str(), O_RDONLY);
    if (fd < 0)
    {
        throw std::runtime_error(std::string("Unable to open " + device + ": ") + strerror(errno));
    }

    // Check compatibility between driver library and the kernel
    if (!VerifyKernel(device))
    {
        close(fd);
        throw std::runtime_error(std::string("Wrong kernel module version\n"));
    }

    // Query how big the capabilities data is.
    int capsSize = ioctl(fd, ETHOSN_IOCTL_FW_HW_CAPABILITIES, NULL);
    if (capsSize <= 0)
    {
        close(fd);
        throw std::runtime_error(std::string("Failed to retrieve the size of firmware capabilities: ") +
                                 strerror(errno));
    }

    // Allocate a buffer of this size
    std::vector<char> caps(capsSize);

    // Get the kernel to fill it in
    int ret = ioctl(fd, ETHOSN_IOCTL_FW_HW_CAPABILITIES, caps.data());
    if (ret != 0)
    {
        close(fd);
        throw std::runtime_error(std::string("Failed to retrieve firmware and hardware information data: ") +
                                 strerror(errno));
    }

    close(fd);
    return caps;
}

bool IsKernelVersionMatching(const struct Version& ver, const std::string& device)
{
    // The actual kernel module version obtained from a running system,
    struct Version actKmodVer;

    int fd = open(device.c_str(), O_RDONLY);
    if (fd < 0)
    {
        throw std::runtime_error(std::string("Unable to open " + device + ": ") + strerror(errno));
    }

    int match = ioctl(fd, ETHOSN_IOCTL_GET_VERSION, &actKmodVer);

    close(fd);

    if (match < 0)
    {
        throw std::runtime_error(std::string("Kernel version cannot be obtained \n"));
    }

    if (ver == actKmodVer)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool IsKernelVersionMatching(const struct Version& ver)
{
    return IsKernelVersionMatching(ver, DEVICE_NODE);
}

constexpr bool IsKernelVersionSupported(const uint32_t& majorVersion)
{
    if ((majorVersion <= MAX_ETHOSN_KERNEL_MODULE_MAJOR_VERSION_SUPPORTED) &&
        (majorVersion >= MIN_ETHOSN_KERNEL_MODULE_MAJOR_VERSION_SUPPORTED))
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool VerifyKernel(const std::string& device)
{
    // The kernel module version that is defined in uapi/ethosn.h
    static constexpr struct Version uapiKmodVer(ETHOSN_KERNEL_MODULE_VERSION_MAJOR, ETHOSN_KERNEL_MODULE_VERSION_MINOR,
                                                ETHOSN_KERNEL_MODULE_VERSION_PATCH);

    static_assert((IsKernelVersionSupported(ETHOSN_KERNEL_MODULE_VERSION_MAJOR)),
                  "Kernel module version defined in ethosn.h is not supported");

    return IsKernelVersionMatching(uapiKmodVer, device);
}

bool VerifyKernel()
{
    return VerifyKernel(DEVICE_NODE);
}

KmodNetworkImpl::KmodNetworkImpl(const char* compiledNetworkData,
                                 size_t compiledNetworkSize,
                                 int allocatorFd,
                                 const IntermediateBufferReq& desc)
    : NetworkImpl(compiledNetworkData, compiledNetworkSize, false)
    , m_IntermediateBufferFd(0)
{
    CompiledNetworkInfo compiledNetwork = DeserializeCompiledNetwork(compiledNetworkData, compiledNetworkSize);

    std::vector<ethosn_buffer_info> constantCuInfos =
        ToKmodBufInfos(compiledNetwork.m_ConstantControlUnitDataBufferInfos);
    std::vector<ethosn_buffer_info> constantDmaInfos  = ToKmodBufInfos(compiledNetwork.m_ConstantDmaDataBufferInfos);
    std::vector<ethosn_buffer_info> inputInfos        = ToKmodBufInfos(compiledNetwork.m_InputBufferInfos);
    std::vector<ethosn_buffer_info> outputInfos       = ToKmodBufInfos(compiledNetwork.m_OutputBufferInfos);
    std::vector<ethosn_buffer_info> intermediateInfos = ToKmodBufInfos(compiledNetwork.m_IntermediateDataBufferInfos);

    ethosn_network_req netReq = {};

    netReq.dma_buffers.num  = static_cast<uint32_t>(constantDmaInfos.size());
    netReq.dma_buffers.info = constantDmaInfos.data();
    netReq.dma_data.size    = static_cast<uint32_t>(compiledNetwork.m_ConstantDmaDataSize);
    netReq.dma_data.data    = compiledNetwork.CalculateConstantDmaDataPtr(compiledNetworkData);

    switch (desc.type)
    {
        case MemType::ALLOCATE:
            netReq.intermediate_desc.memory.type      = ethosn_intermediate_desc::ethosn_memory::ALLOCATE;
            netReq.intermediate_desc.memory.data_size = static_cast<uint32_t>(compiledNetwork.m_IntermediateDataSize);
            break;
        case MemType::IMPORT:
            netReq.intermediate_desc.memory.type    = ethosn_intermediate_desc::ethosn_memory::IMPORT;
            netReq.intermediate_desc.memory.dma_req = { desc.fd, desc.flags,
                                                        static_cast<uint32_t>(compiledNetwork.m_IntermediateDataSize) };
            break;
        case MemType::NONE:
            netReq.intermediate_desc.memory = {};
            break;
        default:
            throw std::runtime_error(std::string("Wrong value of memory type of Intermediate Buffers\n"));
            break;
    }

    netReq.intermediate_desc.buffers.num  = static_cast<uint32_t>(intermediateInfos.size());
    netReq.intermediate_desc.buffers.info = intermediateInfos.data();

    netReq.input_buffers.num  = static_cast<uint32_t>(inputInfos.size());
    netReq.input_buffers.info = inputInfos.data();

    netReq.output_buffers.num  = static_cast<uint32_t>(outputInfos.size());
    netReq.output_buffers.info = outputInfos.data();

    netReq.cu_buffers.num  = static_cast<uint32_t>(constantCuInfos.size());
    netReq.cu_buffers.info = constantCuInfos.data();
    netReq.cu_data.size    = static_cast<uint32_t>(compiledNetwork.m_ConstantControlUnitDataSize);
    netReq.cu_data.data    = compiledNetwork.CalculateConstantControlUnitDataPtr(compiledNetworkData);

    m_NetworkFd = ioctl(allocatorFd, ETHOSN_IOCTL_REGISTER_NETWORK, &netReq);
    int err     = errno;
    if (m_NetworkFd < 0)
    {
        throw std::runtime_error(std::string("Unable to create network: ") + strerror(err));
    }
}

KmodNetworkImpl::~KmodNetworkImpl()
{
    try
    {
        // It can throw and it needs to close the file descriptor.
        // Dump intermediate buffer files, if requested
        DumpIntermediateBuffersBasedOnEnvVar();
    }
    catch (const std::exception& e)
    {
        g_Logger.Error("%s", e.what());
    }

    close(m_NetworkFd);
}

Inference* KmodNetworkImpl::ScheduleInference(Buffer* const inputBuffers[],
                                              uint32_t numInputBuffers,
                                              Buffer* const outputBuffers[],
                                              uint32_t numOutputBuffers)
{
    DumpCmmBasedOnEnvVar(inputBuffers, numInputBuffers);

    ethosn_inference_req ifrReq = {};
    std::vector<int> inputFds(numInputBuffers, -1);
    std::vector<int> outputFds(numOutputBuffers, -1);

    for (size_t i = 0; i < numInputBuffers; ++i)
    {
        inputFds[i] = inputBuffers[i]->GetBufferHandle();
    }

    for (size_t i = 0; i < numOutputBuffers; ++i)
    {
        outputFds[i] = outputBuffers[i]->GetBufferHandle();
    }

    ifrReq.num_inputs = numInputBuffers;
    ifrReq.input_fds  = inputFds.data();

    ifrReq.num_outputs = numOutputBuffers;
    ifrReq.output_fds  = outputFds.data();

    // FIXME: Get rid of raw pointers (requires API change)
    int inference_fd = ioctl(m_NetworkFd, ETHOSN_IOCTL_SCHEDULE_INFERENCE, &ifrReq);
    if (inference_fd < 0)
    {
        throw std::runtime_error(std::string("Failed to create inference: ") + strerror(errno));
    }

    return new Inference(inference_fd);
}

std::pair<const char*, size_t> KmodNetworkImpl::MapIntermediateBuffers()
{
    // Get the file handle from the kernel module
    m_IntermediateBufferFd = ioctl(m_NetworkFd, ETHOSN_IOCTL_GET_INTERMEDIATE_BUFFER);
    if (m_IntermediateBufferFd < 0)
    {
        int err = errno;
        throw std::runtime_error(std::string("Unable to get intermediate buffer: ") + strerror(err));
    }

    // Find the size of the buffer
    off_t size;
    size = lseek(m_IntermediateBufferFd, 0, SEEK_END);
    if (size < 0)
    {
        int err = errno;
        close(m_IntermediateBufferFd);
        throw std::runtime_error(std::string("Unable to seek intermediate buffer: ") + strerror(err));
    }

    // Map the buffer so we can read its data
    void* data = mmap(nullptr, size, PROT_READ, MAP_SHARED, m_IntermediateBufferFd, 0);
    if (data == MAP_FAILED)
    {
        int err = errno;
        close(m_IntermediateBufferFd);
        throw std::runtime_error(std::string("Unable to map buffer: ") + strerror(err));
    }

    return { reinterpret_cast<const char*>(data), size };
}

void KmodNetworkImpl::UnmapIntermediateBuffers(std::pair<const char*, size_t> mappedPtr)
{
    munmap(const_cast<char*>(mappedPtr.first), mappedPtr.second);
    close(m_IntermediateBufferFd);
}

}    // namespace driver_library
}    // namespace ethosn
