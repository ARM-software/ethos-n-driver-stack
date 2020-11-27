//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "KmodNetwork.hpp"

#include "../include/ethosn_driver_library/Network.hpp"
#include "Utils.hpp"

#include <ethosn_command_stream/CommandStreamBuffer.hpp>
#include <ethosn_support_library/Support.hpp>
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

ethosn_buffer_info ToKmodBufInfo(const support_library::BufferInfo& info)
{
    return { info.m_Id, info.m_Offset, info.m_Size };
}

template <typename TBufferInfo>
std::vector<ethosn_buffer_info> ToKmodBufInfos(const std::vector<TBufferInfo>& infos)
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

std::vector<char> GetFirmwareAndHardwareCapabilities()
{
    int fd = open(ETHOSN_STRINGIZE_VALUE_OF(DEVICE_NODE), O_RDONLY);
    if (fd < 0)
    {
        throw std::runtime_error(std::string("Unable to open ") + std::string(ETHOSN_STRINGIZE_VALUE_OF(DEVICE_NODE)) +
                                 std::string(": ") + strerror(errno));
    }

    // Query how big the capabilities data is.
    int capsSize = ioctl(fd, ETHOSN_IOCTL_FW_HW_CAPABILITIES, NULL);
    if (capsSize <= 0)
    {
        throw std::runtime_error(std::string("Failed to retrieve the size of firmware capabilities, errno = ") +
                                 strerror(errno));
    }

    // Allocate a buffer of this size
    std::vector<char> caps(capsSize);

    // Get the kernel to fill it in
    int ret = ioctl(fd, ETHOSN_IOCTL_FW_HW_CAPABILITIES, caps.data());
    if (ret != 0)
    {
        throw std::runtime_error(std::string("Failed to retrieve firmware and hardware information data, errno = ") +
                                 strerror(errno));
    }

    close(fd);
    return caps;
}

KmodNetworkImpl::KmodNetworkImpl(support_library::CompiledNetwork& compiledNetwork)
    : NetworkImpl(compiledNetwork)
{
    std::vector<ethosn_buffer_info> constantCuInfos =
        ToKmodBufInfos(compiledNetwork.GetConstantControlUnitDataBufferInfos());
    std::vector<ethosn_buffer_info> constantDmaInfos = ToKmodBufInfos(compiledNetwork.GetConstantDmaDataBufferInfos());
    std::vector<ethosn_buffer_info> inputInfos       = ToKmodBufInfos(compiledNetwork.GetInputBufferInfos());
    std::vector<ethosn_buffer_info> outputInfos      = ToKmodBufInfos(compiledNetwork.GetOutputBufferInfos());
    std::vector<ethosn_buffer_info> intermediateInfos =
        ToKmodBufInfos(compiledNetwork.GetIntermediateDataBufferInfos());

    ethosn_network_req netReq = {};

    netReq.dma_buffers.num  = static_cast<uint32_t>(constantDmaInfos.size());
    netReq.dma_buffers.info = constantDmaInfos.data();
    netReq.dma_data.size    = static_cast<uint32_t>(compiledNetwork.GetConstantDmaData().size());
    netReq.dma_data.data    = compiledNetwork.GetConstantDmaData().data();

    netReq.intermediate_buffers.num  = static_cast<uint32_t>(intermediateInfos.size());
    netReq.intermediate_buffers.info = intermediateInfos.data();

    netReq.intermediate_data_size = compiledNetwork.GetIntermediateDataSize();

    netReq.input_buffers.num  = static_cast<uint32_t>(inputInfos.size());
    netReq.input_buffers.info = inputInfos.data();

    netReq.output_buffers.num  = static_cast<uint32_t>(outputInfos.size());
    netReq.output_buffers.info = outputInfos.data();

    netReq.cu_buffers.num  = static_cast<uint32_t>(constantCuInfos.size());
    netReq.cu_buffers.info = constantCuInfos.data();
    netReq.cu_data.size    = static_cast<uint32_t>(compiledNetwork.GetConstantControlUnitData().size());
    netReq.cu_data.data    = compiledNetwork.GetConstantControlUnitData().data();

    int ethosnFd = open(ETHOSN_STRINGIZE_VALUE_OF(DEVICE_NODE), O_RDONLY);
    if (ethosnFd < 0)
    {
        throw std::runtime_error(std::string("Unable to open ") + std::string(ETHOSN_STRINGIZE_VALUE_OF(DEVICE_NODE)) +
                                 std::string(": ") + strerror(errno));
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

KmodNetworkImpl::~KmodNetworkImpl()
{
    // Dump intermediate buffer files, if requested
    const char* const debugEnv = std::getenv("ETHOSN_DRIVER_LIBRARY_DEBUG");
    if (debugEnv && strstr(debugEnv, "dump-intermediate") != nullptr)
    {
        DumpIntermediateBuffers();
    }

    close(m_NetworkFd);
}

Inference* KmodNetworkImpl::ScheduleInference(Buffer* const inputBuffers[],
                                              uint32_t numInputBuffers,
                                              Buffer* const outputBuffers[],
                                              uint32_t numOutputBuffers) const
{
    const char* const debugEnv = std::getenv("ETHOSN_DRIVER_LIBRARY_DEBUG");
    if (debugEnv && (strcmp(debugEnv, "1") == 0 || strstr(debugEnv, "cmm") != nullptr))
    {
        DumpCmm(inputBuffers, numInputBuffers, (std::string("CombinedMemoryMap_") + m_DebugName + ".hex").c_str(),
                Cmm_All);
    }
    else if (debugEnv && strstr(debugEnv, "cmdstream") != nullptr)
    {
        DumpCmm(inputBuffers, numInputBuffers, (std::string("CombinedMemoryMap_") + m_DebugName + ".hex").c_str(),
                Cmm_Inference | Cmm_ConstantControlUnit);
    }

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
    std::cout << "Dumping intermediate buffers..." << std::endl;

    // Get the file handle from the kernel module
    int intermediateBufferFd = ioctl(m_NetworkFd, ETHOSN_IOCTL_GET_INTERMEDIATE_BUFFER);
    if (intermediateBufferFd < 0)
    {
        int err = errno;
        std::cerr << "Unable to get intermediate buffer: " << strerror(err) << std::endl;
        return;
    }

    // Find the size of the buffer and validate it
    off_t size;
    size = lseek(intermediateBufferFd, 0, SEEK_END);
    if (size < 0)
    {
        int err = errno;
        std::cerr << "Unable to seek intermediate buffer: " << strerror(err) << std::endl;
        close(intermediateBufferFd);
        return;
    }
    if (size != m_CompiledNetwork.GetIntermediateDataSize())
    {
        std::cerr << "Intermediate data was of unexpected size: CompiledNetwork: "
                  << m_CompiledNetwork.GetIntermediateDataSize() << ", Kernel: " << size << std::endl;
    }

    if (size > 0)    // There may not be any intermediate data at all
    {
        // Map the buffer so we can read its data
        uint8_t* data = reinterpret_cast<uint8_t*>(mmap(nullptr, size, PROT_READ, MAP_SHARED, intermediateBufferFd, 0));
        if (data == MAP_FAILED)
        {
            int err = errno;
            std::cerr << "Unable to map buffer: " << strerror(err) << std::endl;
            close(intermediateBufferFd);
            return;
        }

        // Parse the command stream to find the DUMP_DRAM commands
        support_library::BufferInfo cmdStreamInfo = m_CompiledNetwork.GetConstantControlUnitDataBufferInfos()[0];
        const uint8_t* rawCmdStreamData           = m_CompiledNetwork.GetConstantControlUnitData().data();
        command_stream::CommandStream cmdStream(rawCmdStreamData + cmdStreamInfo.m_Offset,
                                                rawCmdStreamData + cmdStreamInfo.m_Offset + cmdStreamInfo.m_Size);
        for (auto it = cmdStream.begin(); it != cmdStream.end(); ++it)
        {
            if (it->m_Opcode() == command_stream::Opcode::DUMP_DRAM)
            {
                const command_stream::CommandData<command_stream::Opcode::DUMP_DRAM>& cmd =
                    it->GetCommand<command_stream::Opcode::DUMP_DRAM>()->m_Data();

                // Find where this buffer is in the intermediate data
                auto bufferInfoIt = std::find_if(m_CompiledNetwork.GetIntermediateDataBufferInfos().begin(),
                                                 m_CompiledNetwork.GetIntermediateDataBufferInfos().end(),
                                                 [&](const auto& b) { return b.m_Id == cmd.m_DramBufferId(); });
                if (bufferInfoIt == m_CompiledNetwork.GetIntermediateDataBufferInfos().end())
                {
                    std::cerr << "Can't find buffer info for buffer ID " << cmd.m_DramBufferId()
                              << ", which would have been dumped to " << cmd.m_Filename().data() << std::endl;
                }
                else
                {
                    // Modify the filename to include the network name, so we don't overwrite files for example when running multiple subgraphs
                    std::string dumpFilename = cmd.m_Filename().data();
                    dumpFilename             = utils::ReplaceAll(dumpFilename, "EthosNIntermediateBuffer_",
                                                     "EthosNIntermediateBuffer_" + m_DebugName + "_");

                    std::ofstream fs(dumpFilename.c_str());
                    WriteHex(fs, 0, data + bufferInfoIt->m_Offset, bufferInfoIt->m_Size);
                    std::cout << "Dumped EthosN intermediate buffer " << bufferInfoIt->m_Id << " to " << dumpFilename
                              << std::endl;
                }
            }
        }

        munmap(data, size);
    }
    else
    {
        std::cerr << "No intermediate data to dump" << std::endl;
    }

    close(intermediateBufferFd);

    std::cout << "Finished dumping intermediate buffers" << std::endl;
}

}    // namespace driver_library
}    // namespace ethosn
