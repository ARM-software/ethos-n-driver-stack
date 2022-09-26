//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "KmodNetwork.hpp"

#include "../include/ethosn_driver_library/Network.hpp"
#include "Utils.hpp"

#include <ethosn_command_stream/CommandStreamBuffer.hpp>
#include <ethosn_utils/Strings.hpp>
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

KmodNetworkImpl::KmodNetworkImpl(const char* compiledNetworkData, size_t compiledNetworkSize, const std::string& device)
    : NetworkImpl(compiledNetworkData, compiledNetworkSize, false)
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

    netReq.intermediate_buffers.num  = static_cast<uint32_t>(intermediateInfos.size());
    netReq.intermediate_buffers.info = intermediateInfos.data();

    netReq.intermediate_data_size = compiledNetwork.m_IntermediateDataSize;

    netReq.input_buffers.num  = static_cast<uint32_t>(inputInfos.size());
    netReq.input_buffers.info = inputInfos.data();

    netReq.output_buffers.num  = static_cast<uint32_t>(outputInfos.size());
    netReq.output_buffers.info = outputInfos.data();

    netReq.cu_buffers.num  = static_cast<uint32_t>(constantCuInfos.size());
    netReq.cu_buffers.info = constantCuInfos.data();
    netReq.cu_data.size    = static_cast<uint32_t>(compiledNetwork.m_ConstantControlUnitDataSize);
    netReq.cu_data.data    = compiledNetwork.CalculateConstantControlUnitDataPtr(compiledNetworkData);

    int ethosnFd = open(device.c_str(), O_RDONLY);
    if (ethosnFd < 0)
    {
        throw std::runtime_error(std::string("Unable to open " + device + ": ") + strerror(errno));
    }

    // Check compatibility between driver library and the kernel
    if (!VerifyKernel(device))
    {
        close(ethosnFd);
        throw std::runtime_error(std::string("Wrong kernel module version\n"));
    }

    m_NetworkFd = ioctl(ethosnFd, ETHOSN_IOCTL_REGISTER_NETWORK, &netReq);
    int err     = errno;
    close(ethosnFd);

    // Check file descriptor after closing ethosn fd, as not to leak data.
    // Need to cache errno, as to not be overwritten by `close`
    if (m_NetworkFd < 0)
    {
        throw std::runtime_error(std::string("Unable to create network: ") + strerror(err));
    }
}

KmodNetworkImpl::KmodNetworkImpl(const char* compiledNetworkData, size_t compiledNetworkSize, int allocatorFd)
    : NetworkImpl(compiledNetworkData, compiledNetworkSize, false)
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

    netReq.intermediate_buffers.num  = static_cast<uint32_t>(intermediateInfos.size());
    netReq.intermediate_buffers.info = intermediateInfos.data();

    netReq.intermediate_data_size = compiledNetwork.m_IntermediateDataSize;

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
        const char* const debugEnv = std::getenv("ETHOSN_DRIVER_LIBRARY_DEBUG");
        if (debugEnv && strstr(debugEnv, "dump-intermediate") != nullptr)
        {
            DumpIntermediateBuffers();
        }
    }
    catch (...)
    {
        assert(false);
    }

    close(m_NetworkFd);
}

Inference* KmodNetworkImpl::ScheduleInference(Buffer* const inputBuffers[],
                                              uint32_t numInputBuffers,
                                              Buffer* const outputBuffers[],
                                              uint32_t numOutputBuffers) const
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

void KmodNetworkImpl::DumpIntermediateBuffers()
{
    if (!m_CompiledNetwork)
    {
        throw std::runtime_error("Missing m_CompiledNetwork");
    }
    g_Logger.Debug("Dumping intermediate buffers...");

    // Check if any intermediate buffers overlap memory with each another, and warn in this case that the
    // developer should probably modify support library to use non-overlapping intermediate buffers, otherwise
    // the intermediate dump will likely be corrupted.
    {
        std::vector<BufferInfo> buffers = m_CompiledNetwork->m_IntermediateDataBufferInfos;
        std::sort(buffers.begin(), buffers.end(),
                  [](const BufferInfo& a, const BufferInfo& b) { return a.m_Offset < b.m_Offset; });
        for (uint32_t i = 1; i < buffers.size(); ++i)
        {
            if (buffers[i - 1].m_Offset + buffers[i - 1].m_Size > buffers[i].m_Offset)
            {
                g_Logger.Warning(
                    "Intermediate buffers are overlapping and so the data about to be dumped may "
                    "be corrupted. Consider enabling the debugDisableBufferReuse option in the Support Library to "
                    "prevent this.");
            }
        }
    }

    // Get the file handle from the kernel module
    int intermediateBufferFd = ioctl(m_NetworkFd, ETHOSN_IOCTL_GET_INTERMEDIATE_BUFFER);
    if (intermediateBufferFd < 0)
    {
        int err = errno;
        g_Logger.Error("Unable to get intermediate buffer: %s", strerror(err));
        return;
    }

    // Find the size of the buffer and validate it
    off_t size;
    size = lseek(intermediateBufferFd, 0, SEEK_END);
    if (size < 0)
    {
        int err = errno;
        g_Logger.Error("Unable to seek intermediate buffer: %s", strerror(err));
        close(intermediateBufferFd);
        return;
    }
    if (size != m_CompiledNetwork->m_IntermediateDataSize)
    {
        g_Logger.Error("Intermediate data was of unexpected size: CompiledNetwork: %d, Kernel: %zu",
                       m_CompiledNetwork->m_IntermediateDataSize, static_cast<size_t>(size));
    }

    if (size > 0)    // There may not be any intermediate data at all
    {
        // Map the buffer so we can read its data
        uint8_t* data = reinterpret_cast<uint8_t*>(mmap(nullptr, size, PROT_READ, MAP_SHARED, intermediateBufferFd, 0));
        if (data == MAP_FAILED)
        {
            int err = errno;
            g_Logger.Error("Unable to map buffer: %s", strerror(err));
            close(intermediateBufferFd);
            return;
        }

        // Parse the command stream to find the DUMP_DRAM commands
        BufferInfo cmdStreamInfo = m_CompiledNetwork->m_ConstantControlUnitDataBufferInfos[0];
        const uint8_t* rawCmdStreamData =
            m_CompiledNetwork->CalculateConstantControlUnitDataPtr(m_CompiledNetworkData.data());
        command_stream::CommandStream cmdStream(rawCmdStreamData + cmdStreamInfo.m_Offset,
                                                rawCmdStreamData + cmdStreamInfo.m_Offset + cmdStreamInfo.m_Size);
        for (auto it = cmdStream.begin(); it != cmdStream.end(); ++it)
        {
            if (it->m_Opcode() == command_stream::Opcode::DUMP_DRAM)
            {
                const command_stream::CommandData<command_stream::Opcode::DUMP_DRAM>& cmd =
                    it->GetCommand<command_stream::Opcode::DUMP_DRAM>()->m_Data();

                // Find where this buffer is in the intermediate data
                auto bufferInfoIt = std::find_if(m_CompiledNetwork->m_IntermediateDataBufferInfos.begin(),
                                                 m_CompiledNetwork->m_IntermediateDataBufferInfos.end(),
                                                 [&](const auto& b) { return b.m_Id == cmd.m_DramBufferId(); });
                if (bufferInfoIt == m_CompiledNetwork->m_IntermediateDataBufferInfos.end())
                {
                    g_Logger.Error("Can't find buffer info for buffer ID %d, which would have been dumped to %s",
                                   cmd.m_DramBufferId(), cmd.m_Filename().data());
                }
                else
                {
                    // Modify the filename to include the network name, so we don't overwrite files for example when running multiple subgraphs
                    std::string dumpFilename = cmd.m_Filename().data();
                    dumpFilename             = utils::ReplaceAll(dumpFilename, "EthosNIntermediateBuffer_",
                                                     "EthosNIntermediateBuffer_" + m_DebugName + "_");

                    std::ofstream fs(dumpFilename.c_str());
                    WriteHex(fs, 0, data + bufferInfoIt->m_Offset, bufferInfoIt->m_Size);
                    g_Logger.Debug("Dumped intermediate buffer %d to %s", bufferInfoIt->m_Id, dumpFilename.c_str());
                }
            }
        }

        munmap(data, size);
    }
    else
    {
        g_Logger.Error("No intermediate data to dump");
    }

    close(intermediateBufferFd);

    g_Logger.Debug("Finished dumping intermediate buffers");
}

}    // namespace driver_library
}    // namespace ethosn
