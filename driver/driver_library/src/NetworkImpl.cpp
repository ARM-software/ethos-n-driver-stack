//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "NetworkImpl.hpp"

#include "Utils.hpp"

#include <ethosn_command_stream/CommandStream.hpp>
#include <ethosn_command_stream/CommandStreamBuffer.hpp>
#include <ethosn_firmware.h>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

using namespace ethosn;

using MemoryMap = std::map<uint64_t, std::array<uint32_t, 4>>;

namespace
{

uint64_t AddToMemoryMap(MemoryMap& mm,
                        const uint64_t baseAddr,
                        const std::array<uint32_t, 4>* first,
                        const std::array<uint32_t, 4>* last)
{
    uint64_t addr = baseAddr;

    for (auto values = first; values != last; ++values)
    {
        mm[addr] = *values;
        addr += sizeof(*values);
    }

    return addr;
}

uint64_t AddToMemoryMap(MemoryMap& mm, const uint32_t baseAddr, const void* data, const size_t size)
{
    constexpr size_t sizeOfLine    = sizeof(mm[0]);
    constexpr size_t sizeOfElement = sizeof(mm[0][0]);

    // see use of remaining below
    assert(sizeOfLine / sizeOfElement == 4);

    const std::array<uint32_t, 4>* mmBegin = reinterpret_cast<const std::array<uint32_t, 4>*>(data);
    const std::array<uint32_t, 4>* mmEnd   = mmBegin + (size / sizeOfLine);

    uint64_t addr = AddToMemoryMap(mm, baseAddr, mmBegin, mmEnd);

    // there are up to sizeOfLine bytes left, group them in sizeOfElement
    const size_t remaining = (((baseAddr + size) - addr) + sizeOfElement - 1) / sizeOfElement;

    if (remaining > 0)
    {
        mm[addr] = { (*mmEnd)[0], ((remaining > 1) ? (*mmEnd)[1] : 0), ((remaining > 2) ? (*mmEnd)[2] : 0),
                     ((remaining > 3) ? (*mmEnd)[3] : 0) };

        addr += sizeOfLine;
    }

    return addr;
}

template <typename T>
uint64_t AddToMemoryMap(MemoryMap& mm, const uint32_t baseAddr, const T& data)
{
    return AddToMemoryMap(mm, baseAddr, data.data(), data.size() * sizeof(data[0]));
}

MemoryMap GetFirmwareMemMap(const char* firmwareFile)
{
    if (!driver_library::FileExists(firmwareFile))
    {
        return MemoryMap();
    }
    // Produce combined memory map
    MemoryMap memMap;

    // Add firmware data
    {
        std::ifstream fwHex(firmwareFile);
        std::string line;

        while (std::getline(fwHex, line))
        {
            size_t pos;

            const uint32_t addr = static_cast<uint32_t>(std::stoul(line, &pos, 16));

            std::array<uint32_t, 4> values;
            for (uint32_t& v : values)
            {
                line = line.substr(pos + 1);
                v    = static_cast<uint32_t>(std::stoul(line, &pos, 16));
            }

            memMap[addr] = values;
        }
    }
    return memMap;
}
}    // namespace

namespace ethosn
{
namespace driver_library
{

NetworkImpl::NetworkImpl(support_library::CompiledNetwork& compiledNetwork)
    : m_CompiledNetwork(compiledNetwork)
{}

Inference* NetworkImpl::ScheduleInference(Buffer* const inputBuffers[],
                                          uint32_t numInputBuffers,
                                          Buffer* const[],
                                          uint32_t) const
{
    DumpCmm(inputBuffers, numInputBuffers, "CombinedMemoryMap.hex", Cmm_All);

    // Simulate an inference result for the user by creating a memory stream containing the result status.
    FILE* tempFile         = std::tmpfile();
    InferenceResult status = InferenceResult::Completed;
    if (fwrite(&status, sizeof(status), 1, tempFile) != 1)
    {
        return nullptr;
    }
    fseek(tempFile, 0, SEEK_SET);

    return new Inference(fileno(tempFile));
}

void NetworkImpl::SetDebugName(const char* name)
{
    m_DebugName = name;
}

void NetworkImpl::DumpCmm(Buffer* const inputBuffers[],
                          uint32_t numInputBuffers,
                          const char* cmmFilename,
                          uint8_t sections) const
{
    constexpr uint32_t defaultMailboxAddr = 0x60000000;
    constexpr uint32_t defaultBaseAddr    = 0x60100000;

    const char* const baseAddressStr    = std::getenv("BASE_ADDRESS");
    const char* const cuBaseAddressStr  = std::getenv("CU_BASE_ADDRESS");
    const char* const mailboxAddressStr = std::getenv("MAILBOX_ADDRESS");

    uint64_t baseAddress   = (baseAddressStr != nullptr) ? std::stoul(baseAddressStr, nullptr, 0) : defaultBaseAddr;
    uint64_t cuBaseAddress = (cuBaseAddressStr != nullptr) ? std::stoul(cuBaseAddressStr, nullptr, 0) : baseAddress;
    uint64_t mailboxAddress =
        (mailboxAddressStr != nullptr) ? std::stoul(mailboxAddressStr, nullptr, 0) : defaultMailboxAddr;

    // Get size of firmware from env, if it doesn't exist assume we are running on the model and do not need a firmware file.
    const char* const firmwareFile = std::getenv("FIRMWARE_FILE");

    // Decide where each type of buffer is going to be placed.
    // Other buffer types need allocations in the functional model's address space.
    uint64_t constantDmaDataBaseAddress = ethosn::driver_library::RoundUpToNearestMultiple(baseAddress, 64);
    uint64_t inputBuffersBaseAddress    = ethosn::driver_library::RoundUpToNearestMultiple(
        constantDmaDataBaseAddress + m_CompiledNetwork.GetConstantDmaData().size(), 64);
    uint64_t outputBuffersBaseAddress = ethosn::driver_library::RoundUpToNearestMultiple(
        inputBuffersBaseAddress + GetLastAddressedMemory(m_CompiledNetwork.GetInputBufferInfos()), 64);
    uint64_t intermediateDataBaseAddress = ethosn::driver_library::RoundUpToNearestMultiple(
        outputBuffersBaseAddress + GetLastAddressedMemory(m_CompiledNetwork.GetOutputBufferInfos()), 64);

    uint64_t cmmConstantControlUnitDataBaseAddress = ethosn::driver_library::RoundUpToNearestMultiple(
        intermediateDataBaseAddress + m_CompiledNetwork.GetIntermediateDataSize(), 64);

    const std::vector<uint32_t> combinedMemMapInferenceData = BuildInferenceData(
        cuBaseAddress + cmmConstantControlUnitDataBaseAddress - baseAddress, constantDmaDataBaseAddress,
        inputBuffersBaseAddress, outputBuffersBaseAddress, intermediateDataBaseAddress);

    // Produce combined memory map
    MemoryMap cmm = GetFirmwareMemMap(firmwareFile);

    // Add "memory map"
    if (sections & Cmm_ConstantDma)
    {
        AddToMemoryMap(cmm, static_cast<uint32_t>(constantDmaDataBaseAddress), m_CompiledNetwork.GetConstantDmaData());
    }
    if (sections & Cmm_ConstantControlUnit)
    {
        AddToMemoryMap(cmm, static_cast<uint32_t>(cmmConstantControlUnitDataBaseAddress),
                       m_CompiledNetwork.GetConstantControlUnitData());
    }

    // Write the inference data, which includes the binding table
    const uint32_t inferenceAddr = static_cast<uint32_t>(mailboxAddress) + 16;
    AddToMemoryMap(cmm, static_cast<uint32_t>(mailboxAddress), &inferenceAddr, sizeof(inferenceAddr));
    if (sections & Cmm_Inference)
    {
        AddToMemoryMap(cmm, inferenceAddr, combinedMemMapInferenceData);
    }

    // Then load in the IFM data
    if (sections & Cmm_Ifm)
    {
        for (uint32_t i = 0; i < numInputBuffers; ++i)
        {
            auto& info = m_CompiledNetwork.GetInputBufferInfos()[i];
            auto ifm   = inputBuffers[i];
            AddToMemoryMap(cmm, static_cast<uint32_t>(inputBuffersBaseAddress) + info.m_Offset, ifm->GetMappedBuffer(),
                           info.m_Size);
        }
    }

    // Write cmm to file
    {
        std::ofstream cmmStream(cmmFilename);
        cmmStream << std::hex << std::setfill('0');

        for (const auto& addrValues : cmm)
        {
            cmmStream << std::setw(8) << addrValues.first << ':';

            for (const auto& v : addrValues.second)
            {
                cmmStream << ' ' << std::setw(8) << v;
            }

            cmmStream << std::endl;
        }
    }
}

// TBufferInfo can be either a BufferInfo, an InputBufferInfo or an OutputBufferInfo.
template <typename TBufferInfo>
void FillBufferTable(std::vector<ethosn_buffer_desc>& bufferTable,
                     uint64_t baseAddress,
                     const std::vector<TBufferInfo>& bufferInfos)
{
    for (auto&& buffer : bufferInfos)
    {
        bufferTable[buffer.m_Id] = { baseAddress + buffer.m_Offset, buffer.m_Size };
    }
}

/// Constructs the raw data for an inference, corresponding to the control_unit::Inference class.
std::vector<uint32_t> NetworkImpl::BuildInferenceData(uint64_t constantControlUnitDataBaseAddress,
                                                      uint64_t constantDmaDataBaseAddress,
                                                      uint64_t inputBuffersBaseAddress,
                                                      uint64_t outputBuffersBaseAddress,
                                                      uint64_t intermediateDataBaseAddress) const
{
    std::vector<uint32_t> inferenceData;

    // Calculate and append total number of buffers to place in the buffer table.
    size_t numCuBufs           = m_CompiledNetwork.GetConstantControlUnitDataBufferInfos().size();
    size_t numDmaBufs          = m_CompiledNetwork.GetConstantDmaDataBufferInfos().size();
    size_t numInputBufs        = m_CompiledNetwork.GetInputBufferInfos().size();
    size_t numOutputBufs       = m_CompiledNetwork.GetOutputBufferInfos().size();
    size_t numIntermediateBufs = m_CompiledNetwork.GetIntermediateDataBufferInfos().size();

    const uint32_t numBuffers =
        static_cast<uint32_t>(numCuBufs + numDmaBufs + numInputBufs + numOutputBufs + numIntermediateBufs);
    ethosn_buffer_array buffers;
    buffers.num_buffers = numBuffers;
    command_stream::EmplaceBack<ethosn_buffer_array>(inferenceData, buffers);

    // Fill in the buffer table, which is ordered by buffer ID.
    std::vector<ethosn_buffer_desc> bufferTable(numBuffers);
    FillBufferTable(bufferTable, constantControlUnitDataBaseAddress,
                    m_CompiledNetwork.GetConstantControlUnitDataBufferInfos());
    FillBufferTable(bufferTable, constantDmaDataBaseAddress, m_CompiledNetwork.GetConstantDmaDataBufferInfos());
    FillBufferTable(bufferTable, inputBuffersBaseAddress, m_CompiledNetwork.GetInputBufferInfos());
    FillBufferTable(bufferTable, outputBuffersBaseAddress, m_CompiledNetwork.GetOutputBufferInfos());
    FillBufferTable(bufferTable, intermediateDataBaseAddress, m_CompiledNetwork.GetIntermediateDataBufferInfos());

    // Append buffer table to raw data.
    for (const auto& bufferInfo : bufferTable)
    {
        command_stream::EmplaceBack(inferenceData, bufferInfo);
    }

    return inferenceData;
}

}    // namespace driver_library
}    // namespace ethosn
