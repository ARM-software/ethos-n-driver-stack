//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "NetworkImpl.hpp"

#include "../include/ethosn_driver_library/Network.hpp"
#include "Utils.hpp"
#if defined(ETHOSN_ALLOW_COMMAND_STREAM_DUMP)
#include "BinaryParser.hpp"
#endif

#include <ethosn_command_stream/CommandStreamBuilder.hpp>
#include <ethosn_firmware.h>
#include <ethosn_utils/Strings.hpp>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#if defined(__unix__)
#include <unistd.h>
#elif defined(_MSC_VER)
#include <io.h>
#endif

using namespace ethosn;
using namespace ethosn::driver_library;

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

/// Reads values from a raw byte array.
class Reader
{
public:
    Reader(const uint8_t* data, size_t size)
        : m_Data(data)
        , m_Size(size)
        , m_Pos(0)
    {}

    size_t GetPosition()
    {
        return m_Pos;
    }

    bool ReadUint8(uint8_t& outValue)
    {
        if (m_Pos + 1 > m_Size)
        {
            return false;
        }

        outValue = m_Data[m_Pos];
        m_Pos += 1;
        return true;
    }

    /// Assumes little-endian encoding, regardless of the host platform's endian-ness.
    bool ReadUint32(uint32_t& outValue)
    {
        if (m_Pos + 4 > m_Size)
        {
            return false;
        }

        uint32_t data =
            m_Data[m_Pos + 3] << 24 | m_Data[m_Pos + 2] << 16 | m_Data[m_Pos + 1] << 8 | m_Data[m_Pos + 0] << 0;
        m_Pos += 4;
        outValue = data;
        return true;
    }

    bool Skip(uint32_t numBytes)
    {
        if (m_Pos + numBytes > m_Size)
        {
            return false;
        }

        m_Pos += numBytes;
        return true;
    }

private:
    const uint8_t* m_Data;
    size_t m_Size;
    size_t m_Pos;
};

/// Note this does not copy the data - it just returns an offset to the beginning of the array and a size.
/// Therefore the Reader's underlying data must be kept available for the caller to read the array contents.
bool ReadByteArray(Reader& reader, size_t& outOffset, size_t& outSize)
{
    uint32_t size;
    if (!reader.ReadUint32(size))
    {
        return false;
    }
    outSize = size;

    outOffset = reader.GetPosition();
    if (!reader.Skip(size))
    {
        return false;
    }

    return true;
}

bool ReadString(Reader& reader, std::string& out)
{
    uint32_t size;
    if (!reader.ReadUint32(size))
    {
        return false;
    }

    out.clear();
    for (uint32_t i = 0; i < size; ++i)
    {
        uint8_t b;
        if (!reader.ReadUint8(b))
        {
            return false;
        }
        out += b;
    }

    return true;
}

bool ReadBufferInfoArray(Reader& reader, std::vector<BufferInfo>& outData)
{
    uint32_t size;
    if (!reader.ReadUint32(size))
    {
        return false;
    }

    for (uint32_t i = 0; i < size; ++i)
    {
        BufferInfo item;
        if (!reader.ReadUint32(item.m_Id) || !reader.ReadUint32(item.m_Offset) || !reader.ReadUint32(item.m_Size) ||
            !ReadString(reader, item.m_DebugName))
        {
            return false;
        }

        outData.emplace_back(item);
    }

    return true;
}

}    // namespace

namespace ethosn
{
namespace driver_library
{

CompiledNetworkInfo DeserializeCompiledNetwork(const char* data, size_t size)
{
    Reader reader(reinterpret_cast<const uint8_t*>(data), size);

    CompiledNetworkInfo result;

    // Verify "FourCC"
    uint8_t fourcc[4] = {};
    if (!reader.ReadUint8(fourcc[0]) || !reader.ReadUint8(fourcc[1]) || !reader.ReadUint8(fourcc[2]) ||
        !reader.ReadUint8(fourcc[3]))
    {
        throw CompiledNetworkException("Data too short");
    }
    if (fourcc[0] != 'E' || fourcc[1] != 'N' || fourcc[2] != 'C' || fourcc[3] != 'N')
    {
        throw CompiledNetworkException("Not a serialized CompiledNetwork");
    }

    // Verify version
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
    if (!reader.ReadUint32(major) || !reader.ReadUint32(minor) || !reader.ReadUint32(patch))
    {
        throw CompiledNetworkException("Data too short");
    }
    if (major < MIN_ETHOSN_COMPILED_NETWORK_MAJOR_VERSION_SUPPORTED ||
        major > MAX_ETHOSN_COMPILED_NETWORK_MAJOR_VERSION_SUPPORTED)
    {
        throw CompiledNetworkException("Unsupported version");
    }

    // Read main data
    bool success = true;
    success      = success && ReadByteArray(reader, result.m_ConstantDmaDataOffset, result.m_ConstantDmaDataSize);
    success =
        success && ReadByteArray(reader, result.m_ConstantControlUnitDataOffset, result.m_ConstantControlUnitDataSize);
    success = success && ReadBufferInfoArray(reader, result.m_InputBufferInfos);
    success = success && ReadBufferInfoArray(reader, result.m_OutputBufferInfos);
    success = success && ReadBufferInfoArray(reader, result.m_ConstantControlUnitDataBufferInfos);
    success = success && ReadBufferInfoArray(reader, result.m_ConstantDmaDataBufferInfos);
    success = success && ReadBufferInfoArray(reader, result.m_IntermediateDataBufferInfos);

    if (!success)
    {
        throw CompiledNetworkException("Corrupted");
    }

    // Calculate intermediate data size
    if (!result.m_IntermediateDataBufferInfos.empty())
    {
        auto GetLastBufferAddress = [](const auto& buf) { return buf.m_Offset + buf.m_Size; };
        auto maxBuffer            = std::max_element(
            result.m_IntermediateDataBufferInfos.begin(), result.m_IntermediateDataBufferInfos.end(),
            [&](const auto& a, const auto& b) { return GetLastBufferAddress(a) < GetLastBufferAddress(b); });
        result.m_IntermediateDataSize = GetLastBufferAddress(*maxBuffer);
    }

    return result;
}

NetworkImpl::NetworkImpl(const char* compiledNetworkData,
                         size_t compiledNetworkSizeData,
                         bool alwaysCopyCompiledNetwork)
{
    // Copy and store the compiled network if we might need it later for debugging use, or if we've been explicitly
    // told to (i.e. for the model backend).
    // We cannot simply store the user's pointers as they are not obliged to keep this data alive.
    const char* const debugEnv = std::getenv("ETHOSN_DRIVER_LIBRARY_DEBUG");
    if (alwaysCopyCompiledNetwork || debugEnv)
    {
        m_CompiledNetworkData = std::vector<char>(compiledNetworkData, compiledNetworkData + compiledNetworkSizeData);
        m_CompiledNetwork     = std::make_unique<CompiledNetworkInfo>(
            DeserializeCompiledNetwork(m_CompiledNetworkData.data(), m_CompiledNetworkData.size()));
    }
}

NetworkImpl::~NetworkImpl()
{}

Inference*
    NetworkImpl::ScheduleInference(Buffer* const inputBuffers[], uint32_t numInputBuffers, Buffer* const[], uint32_t)
{
    DumpCmmBasedOnEnvVar(inputBuffers, numInputBuffers);

    // Simulate an inference result for the user by creating a memory stream containing the result status.
    FILE* tempFile         = std::tmpfile();
    InferenceResult status = InferenceResult::Completed;
    if (fwrite(&status, sizeof(status), 1, tempFile) != 1)
    {
        std::fclose(tempFile);
        return nullptr;
    }
    fseek(tempFile, 0, SEEK_SET);

    // Duplicate the file descriptor so the file pointer can be freed.
    int tmpFd = dup(fileno(tempFile));
    std::fclose(tempFile);
    if (tmpFd < 0)
    {
        return nullptr;
    }

    return new Inference(tmpFd);
}

void NetworkImpl::SetDebugName(const char* name)
{
    m_DebugName = name;
}

void NetworkImpl::DumpCmmBasedOnEnvVar(Buffer* const inputBuffers[], uint32_t numInputBuffers) const
{
    const char* const debugEnv    = std::getenv("ETHOSN_DRIVER_LIBRARY_DEBUG");
    const std::string cmmFilename = std::string("CombinedMemoryMap_") + m_DebugName + ".hex";
    uint8_t cmmSections           = 0;
    if (debugEnv && (strcmp(debugEnv, "1") == 0 || strstr(debugEnv, "cmm") != nullptr))
    {
        cmmSections = Cmm_All;
    }
    else if (debugEnv && strstr(debugEnv, "cmdstream") != nullptr)
    {
        cmmSections = Cmm_Inference | Cmm_ConstantControlUnit;
    }
    if (cmmSections != 0)
    {
        DumpCmm(inputBuffers, numInputBuffers, cmmFilename.c_str(), cmmSections);
        DumpCommandStream((std::string("CommandStream_") + m_DebugName + ".xml").c_str());
    }
}

void NetworkImpl::DumpCmm(Buffer* const inputBuffers[],
                          uint32_t numInputBuffers,
                          const char* cmmFilename,
                          uint8_t sections) const
{
    if (!m_CompiledNetwork)
    {
        throw std::runtime_error("Missing m_CompiledNetwork");
    }

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
        constantDmaDataBaseAddress + m_CompiledNetwork->m_ConstantDmaDataSize, 64);
    uint64_t outputBuffersBaseAddress = ethosn::driver_library::RoundUpToNearestMultiple(
        inputBuffersBaseAddress + GetLastAddressedMemory(m_CompiledNetwork->m_InputBufferInfos), 64);
    uint64_t intermediateDataBaseAddress = ethosn::driver_library::RoundUpToNearestMultiple(
        outputBuffersBaseAddress + GetLastAddressedMemory(m_CompiledNetwork->m_OutputBufferInfos), 64);

    uint64_t cmmConstantControlUnitDataBaseAddress = ethosn::driver_library::RoundUpToNearestMultiple(
        intermediateDataBaseAddress + m_CompiledNetwork->m_IntermediateDataSize, 64);

    const std::vector<uint32_t> combinedMemMapInferenceData = BuildInferenceData(
        cuBaseAddress + cmmConstantControlUnitDataBaseAddress - baseAddress, constantDmaDataBaseAddress,
        inputBuffersBaseAddress, outputBuffersBaseAddress, intermediateDataBaseAddress);

    // Produce combined memory map
    MemoryMap cmm = GetFirmwareMemMap(firmwareFile);

    // Add "memory map"
    if (sections & Cmm_ConstantDma)
    {
        AddToMemoryMap(cmm, static_cast<uint32_t>(constantDmaDataBaseAddress),
                       m_CompiledNetwork->CalculateConstantDmaDataPtr(m_CompiledNetworkData.data()),
                       m_CompiledNetwork->m_ConstantDmaDataSize);
    }
    if (sections & Cmm_ConstantControlUnit)
    {
        AddToMemoryMap(cmm, static_cast<uint32_t>(cmmConstantControlUnitDataBaseAddress),
                       m_CompiledNetwork->CalculateConstantControlUnitDataPtr(m_CompiledNetworkData.data()),
                       m_CompiledNetwork->m_ConstantControlUnitDataSize);
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
            auto& info       = m_CompiledNetwork->m_InputBufferInfos[i];
            auto ifm         = inputBuffers[i];
            uint8_t* ifmData = ifm->Map();
            AddToMemoryMap(cmm, static_cast<uint32_t>(inputBuffersBaseAddress) + info.m_Offset, ifmData, info.m_Size);
            ifm->Unmap();
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
                     const std::vector<TBufferInfo>& bufferInfos,
                     const ethosn_buffer_type bufferType)
{
    for (auto&& buffer : bufferInfos)
    {
        bufferTable[buffer.m_Id] = { baseAddress + buffer.m_Offset, buffer.m_Size, static_cast<uint32_t>(bufferType) };
    }
}

/// Constructs the raw data for an inference, corresponding to the control_unit::Inference class.
std::vector<uint32_t> NetworkImpl::BuildInferenceData(uint64_t constantControlUnitDataBaseAddress,
                                                      uint64_t constantDmaDataBaseAddress,
                                                      uint64_t inputBuffersBaseAddress,
                                                      uint64_t outputBuffersBaseAddress,
                                                      uint64_t intermediateDataBaseAddress) const
{
    if (!m_CompiledNetwork)
    {
        throw std::runtime_error("Missing m_CompiledNetwork");
    }
    std::vector<uint32_t> inferenceData;

    // Calculate and append total number of buffers to place in the buffer table.
    size_t numCuBufs           = m_CompiledNetwork->m_ConstantControlUnitDataBufferInfos.size();
    size_t numDmaBufs          = m_CompiledNetwork->m_ConstantDmaDataBufferInfos.size();
    size_t numInputBufs        = m_CompiledNetwork->m_InputBufferInfos.size();
    size_t numOutputBufs       = m_CompiledNetwork->m_OutputBufferInfos.size();
    size_t numIntermediateBufs = m_CompiledNetwork->m_IntermediateDataBufferInfos.size();

    const uint32_t numBuffers =
        static_cast<uint32_t>(numCuBufs + numDmaBufs + numInputBufs + numOutputBufs + numIntermediateBufs);
    ethosn_buffer_array buffers;
    buffers.num_buffers = numBuffers;
    command_stream::EmplaceBack<ethosn_buffer_array>(inferenceData, buffers);

    // Fill in the buffer table, which is ordered by buffer ID.
    std::vector<ethosn_buffer_desc> bufferTable(numBuffers);
    FillBufferTable(bufferTable, constantControlUnitDataBaseAddress,
                    m_CompiledNetwork->m_ConstantControlUnitDataBufferInfos, ETHOSN_BUFFER_CMD_FW);
    FillBufferTable(bufferTable, constantDmaDataBaseAddress, m_CompiledNetwork->m_ConstantDmaDataBufferInfos,
                    ETHOSN_BUFFER_CONSTANT);
    FillBufferTable(bufferTable, inputBuffersBaseAddress, m_CompiledNetwork->m_InputBufferInfos, ETHOSN_BUFFER_INPUT);
    FillBufferTable(bufferTable, outputBuffersBaseAddress, m_CompiledNetwork->m_OutputBufferInfos,
                    ETHOSN_BUFFER_OUTPUT);
    FillBufferTable(bufferTable, intermediateDataBaseAddress, m_CompiledNetwork->m_IntermediateDataBufferInfos,
                    ETHOSN_BUFFER_INTERMEDIATE);

    // Append buffer table to raw data.
    for (const auto& bufferInfo : bufferTable)
    {
        command_stream::EmplaceBack(inferenceData, bufferInfo);
    }

    return inferenceData;
}

void NetworkImpl::DumpCommandStream(const char* cmdStreamFilename) const
{
    if (!m_CompiledNetwork)
    {
        throw std::runtime_error("Missing m_CompiledNetwork");
    }
#if defined(ETHOSN_ALLOW_COMMAND_STREAM_DUMP)
    BufferInfo cmdStreamInfo = m_CompiledNetwork->m_ConstantControlUnitDataBufferInfos[0];
    const uint8_t* rawCmdStreamData =
        m_CompiledNetwork->CalculateConstantControlUnitDataPtr(m_CompiledNetworkData.data());
    std::stringstream binaryIn;
    binaryIn.write(reinterpret_cast<const char*>(rawCmdStreamData) + cmdStreamInfo.m_Offset, cmdStreamInfo.m_Size);
    std::ofstream xmlOut(cmdStreamFilename);
    BinaryParser(binaryIn).WriteXml(xmlOut);
#else
    ETHOSN_UNUSED(cmdStreamFilename);
    g_Logger.Error(
        "Command stream dump requested but feature is not enabled. Please enable this feature at build time.");
#endif
}

void NetworkImpl::DumpIntermediateBuffersBasedOnEnvVar()
{
    const char* const debugEnv = std::getenv("ETHOSN_DRIVER_LIBRARY_DEBUG");
    if (debugEnv && strstr(debugEnv, "dump-intermediate") != nullptr)
    {
        DumpIntermediateBuffers();
    }
}

void NetworkImpl::DumpIntermediateBuffers()
{
    if (!m_CompiledNetwork)
    {
        throw std::runtime_error("Missing m_CompiledNetwork");
    }
    g_Logger.Debug("Dumping intermediate buffers...");

    if (m_CompiledNetwork->m_IntermediateDataSize == 0)    // There may not be any intermediate data at all
    {
        g_Logger.Debug("No intermediate data to dump");
        return;
    }

    std::vector<BufferInfo> buffers = m_CompiledNetwork->m_IntermediateDataBufferInfos;

    // Check if any intermediate buffers overlap memory with each another, and warn in this case that the
    // developer should probably modify support library to use non-overlapping intermediate buffers, otherwise
    // the intermediate dump will likely be corrupted.
    {
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

    // Map the buffer so we can read its data. This implementation depends on the backend.
    std::pair<const char*, size_t> mapped = MapIntermediateBuffers();

    // Validate the size
    if (mapped.second != m_CompiledNetwork->m_IntermediateDataSize)
    {
        throw std::runtime_error(std::string("Intermediate data was of unexpected size: CompiledNetwork: ") +
                                 std::to_string(m_CompiledNetwork->m_IntermediateDataSize) +
                                 ", mapped: " + std::to_string(mapped.second));
    }

    for (const BufferInfo& bufferInfo : buffers)
    {
        // Find where this buffer is in the intermediate data
        // Modify the filename to include the network name, so we don't overwrite files for example when running multiple subgraphs
        std::string dumpFilename = bufferInfo.m_DebugName;
        dumpFilename             = utils::ReplaceAll(dumpFilename, "EthosNIntermediateBuffer_",
                                         "EthosNIntermediateBuffer_" + m_DebugName + "_");

        std::ofstream fs(dumpFilename.c_str());
        WriteHex(fs, 0, reinterpret_cast<const uint8_t*>(mapped.first) + bufferInfo.m_Offset, bufferInfo.m_Size);
        g_Logger.Debug("Dumped intermediate buffer %d to %s", bufferInfo.m_Id, dumpFilename.c_str());
    }

    UnmapIntermediateBuffers(mapped);

    g_Logger.Debug("Finished dumping intermediate buffers");
}

}    // namespace driver_library
}    // namespace ethosn
