//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "ModelNetwork.hpp"

#include "../include/ethosn_driver_library/Network.hpp"
#include "ProfilingInternal.hpp"
#include "Utils.hpp"

#include <ModelFirmwareInterface.h>
#include <ethosn_firmware.h>

#include <cassert>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#if defined(__unix__)
#include <unistd.h>
#elif defined(_MSC_VER)
#include <io.h>
#endif

using namespace ethosn;

namespace ethosn
{

namespace control_unit
{
// Defined in PleKernelBinaries.hpp
extern const size_t g_PleKernelBinariesSize;
extern const uint8_t g_PleKernelBinaries[];
}    // namespace control_unit

namespace driver_library
{

namespace
{

uint64_t GetFirmwareSize(const char* firmwareFile)
{
    if (!driver_library::FileExists(firmwareFile))
    {
        throw std::runtime_error("Firmware file cannot be found");
    }

    constexpr const uint32_t hexLineLength = sizeof("01234567: 01234567 01234567 01234567 01234567\n") - 1U;
    uint64_t firmwareSize = (std::ifstream(firmwareFile, std::ifstream::ate).tellg() / hexLineLength) * 16U;
    return firmwareSize;
}

}    // namespace

std::vector<char> GetFirmwareAndHardwareCapabilities(const std::string&)
{
    const char* const modelOptions = std::getenv("ETHOSN_DRIVER_LIBRARY_MODEL_OPTIONS");
    return control_unit::GetFirmwareAndHardwareCapabilities(modelOptions);
}

bool VerifyKernel(const std::string&)
{
    return true;
}

bool VerifyKernel()
{
    return true;
}

ModelNetworkImpl::ModelNetworkImpl(const char* compiledNetworkData, size_t compiledNetworkSize)
    // Note we pass true here so that the compiled network data is stored by the base class,
    // as we need it for each inference.
    : NetworkImpl(compiledNetworkData, compiledNetworkSize, true)
    , m_IntermediateDataBaseAddress(0)
{
    m_MappedIntermediateBuffer.resize(m_CompiledNetwork->m_IntermediateDataSize);
}

ModelNetworkImpl::~ModelNetworkImpl()
{
    try
    {
        // Dump intermediate buffer files, if requested
        DumpIntermediateBuffersBasedOnEnvVar();
        m_MappedIntermediateBuffer.clear();
    }
    catch (const std::exception& e)
    {
        g_Logger.Error("%s", e.what());
    }
}

ethosn::driver_library::Inference* ModelNetworkImpl::ScheduleInference(Buffer* const inputBuffers[],
                                                                       uint32_t numInputBuffers,
                                                                       Buffer* const outputBuffers[],
                                                                       uint32_t numOutputBuffers)
{
    DumpCmmBasedOnEnvVar(inputBuffers, numInputBuffers);

    // Constant data for the Control Unit doesn't need a new allocation as it is already loaded into host DRAM.

    const char* const firmwareFile        = std::getenv("FIRMWARE_FILE");
    const char* uscriptFile               = "config.txt";
    const bool uscriptUseFriendlyRegNames = std::getenv("ETHOSN_DRIVER_LIBRARY_USCRIPT_FRIENDLY_REGS") != nullptr;
    const bool enableOutputBufferDump     = std::getenv("ETHOSN_DRIVER_LIBRARY_OUTPUT_BUFFER_DUMP") != nullptr;

    constexpr uint32_t defaultBaseAddr = 0x60100000;
    const char* const baseAddressStr   = std::getenv("BASE_ADDRESS");

    uint64_t baseAddress = (baseAddressStr != nullptr) ? std::stoul(baseAddressStr, nullptr, 0) : defaultBaseAddr;

    // Create the Firmware model.
    const char* const modelOptions = std::getenv("ETHOSN_DRIVER_LIBRARY_MODEL_OPTIONS");

    // Load PLE kernel data into bennto
    uint64_t pleKernelDataAddr;
    if (firmwareFile != nullptr && strlen(firmwareFile) > 0)
    {
        pleKernelDataAddr = GetFirmwareSize(firmwareFile) - control_unit::g_PleKernelBinariesSize;
    }
    else
    {
        pleKernelDataAddr = 0x10000000 - control_unit::g_PleKernelBinariesSize;
    }

    std::unique_ptr<control_unit::IModelFirmwareInterface> firmwareInterface =
        control_unit::IModelFirmwareInterface::Create(modelOptions, uscriptFile, uscriptUseFriendlyRegNames,
                                                      pleKernelDataAddr);
    if (std::getenv("ETHOSN_DRIVER_LIBRARY_DUMP_SRAM"))
    {
        firmwareInterface->DumpSram("initial_ce");
    }
    // Record the loading of the CMM so that when replaying the uscript the appropriate data is loaded.
    // This is used by the HW verification team as they run the model alongside the RTL simulation.
    firmwareInterface->RecordDramLoad(0x0, "CombinedMemoryMap.hex");

    if (!firmwareInterface->LoadDram(pleKernelDataAddr, control_unit::g_PleKernelBinaries,
                                     control_unit::g_PleKernelBinariesSize))
    {
        g_Logger.Error("Failed to load PLE kernel data");
        return nullptr;
    }
    uint64_t constantControlUnitDataBaseAddress = reinterpret_cast<uint64_t>(
        m_CompiledNetwork->CalculateConstantControlUnitDataPtr(m_CompiledNetworkData.data()));
    // If profiling was enabled, setup a buffer for the firmware to write events into.
    // This simulates what the kernel would do.
    // See also comments at top of NullKmodProfiling.cpp for why this can't be implemented in ConfigureKernelDriver().
    std::vector<uint8_t> firmwareProfilingBuffer;    // Storage for the buffer
    uint64_t timestampOffset = 0;
    if (profiling::g_CurrentConfiguration.m_EnableProfiling)
    {
        firmwareProfilingBuffer.resize(profiling::g_CurrentConfiguration.m_FirmwareBufferSize);
        ethosn_firmware_profiling_configuration profilingConfig = {};
        profilingConfig.enable_profiling                        = true;
        profilingConfig.buffer_address  = reinterpret_cast<size_t>(firmwareProfilingBuffer.data());
        profilingConfig.buffer_size     = static_cast<uint32_t>(firmwareProfilingBuffer.size());
        profilingConfig.num_hw_counters = profiling::g_CurrentConfiguration.m_NumHardwareCounters;
        for (uint32_t i = 0; i < profilingConfig.num_hw_counters; ++i)
        {
            profilingConfig.hw_counters[i] =
                ConvertHwCountersToKernel(profiling::g_CurrentConfiguration.m_HardwareCounters[i]);
        }

        firmwareInterface->ResetAndEnableProfiling(profilingConfig);
        // This is the point which the firmware considers zero for profiling timestamps, so
        // record the time offset to wall clock time for later conversion
        timestampOffset = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }

    // Decide where each type of buffer is going to be placed.
    // Other buffer types need allocations in bennto's address space.
    uint64_t constantDmaDataBaseAddress = ethosn::driver_library::RoundUpToNearestMultiple(baseAddress, 64);
    uint64_t inputBuffersBaseAddress    = ethosn::driver_library::RoundUpToNearestMultiple(
        constantDmaDataBaseAddress + m_CompiledNetwork->m_ConstantDmaDataSize, 64);
    uint64_t outputBuffersBaseAddress = ethosn::driver_library::RoundUpToNearestMultiple(
        inputBuffersBaseAddress + GetLastAddressedMemory(m_CompiledNetwork->m_InputBufferInfos), 64);
    m_IntermediateDataBaseAddress = ethosn::driver_library::RoundUpToNearestMultiple(
        outputBuffersBaseAddress + GetLastAddressedMemory(m_CompiledNetwork->m_OutputBufferInfos), 64);

    // Load DMA data into bennto.
    if (m_CompiledNetwork->m_ConstantDmaDataSize > 0)
    {
        if (!firmwareInterface->LoadDram(constantDmaDataBaseAddress,
                                         m_CompiledNetwork->CalculateConstantDmaDataPtr(m_CompiledNetworkData.data()),
                                         m_CompiledNetwork->m_ConstantDmaDataSize))
        {
            g_Logger.Error("Failed to load memory map into Bennto.");
            return nullptr;
        }
    }
    // Then load in the IFM data into bennto
    for (uint32_t i = 0; i < numInputBuffers; ++i)
    {
        auto offset = m_CompiledNetwork->m_InputBufferInfos[i].m_Offset;
        auto ifm    = inputBuffers[i];

        // Use both offset and size returned by the compiler to load input data into DRAM.
        // The driver library for model uses the offsets returned by the compiler to
        // align buffers.
        const bool retVal = firmwareInterface->LoadDram(inputBuffersBaseAddress + offset, ifm->Map(),
                                                        m_CompiledNetwork->m_InputBufferInfos[i].m_Size);
        ifm->Unmap();
        if (!retVal)
        {
            g_Logger.Error("Failed to load IFM into Bennto.");
            return nullptr;
        }
    }

    const std::vector<uint32_t> inferenceData =
        BuildInferenceData(constantControlUnitDataBaseAddress, constantDmaDataBaseAddress, inputBuffersBaseAddress,
                           outputBuffersBaseAddress, m_IntermediateDataBaseAddress);

    // The call to RunInference below is synchronous so there's nothing for a user to wait on.
    // Simulate an inference result for them by creating a memory stream containing the result status.
    FILE* tempFile         = std::tmpfile();
    InferenceResult status = InferenceResult::Completed;
    if (fwrite(&status, sizeof(status), 1, tempFile) != 1)
    {
        g_Logger.Error("Failed to write inference result to temporary file");
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

    std::unique_ptr<Inference> res = std::make_unique<Inference>(tmpFd);

    g_Logger.Debug("About to run inference");

    if (!firmwareInterface->RunInference(inferenceData))
    {
        g_Logger.Error("Failed to execute command stream");
        return nullptr;
    }

    // Copy memory back from bennto to the output buffer
    for (uint32_t i = 0; i < numOutputBuffers; ++i)
    {
        const BufferInfo& outputBuf = m_CompiledNetwork->m_OutputBufferInfos[i];
        firmwareInterface->DumpDram(const_cast<uint8_t*>(outputBuffers[i]->Map()),
                                    outputBuffersBaseAddress + outputBuf.m_Offset, outputBuf.m_Size);
        outputBuffers[i]->Unmap();
    }

    // If requested, dump the output buffers. This is used by the HW verification team to compare results with the RTL.
    if (enableOutputBufferDump)
    {
        std::ofstream bufferDetailsFile("OutputBufferDetails.txt");
        std::ofstream bufferDataFile("OutputBufferData.hex");

        for (uint32_t i = 0; i < numOutputBuffers; ++i)
        {
            const BufferInfo& outputBuf = m_CompiledNetwork->m_OutputBufferInfos[i];

            bufferDetailsFile << std::hex;
            bufferDetailsFile << "0x" << static_cast<uint32_t>(outputBuffersBaseAddress) + outputBuf.m_Offset << " 0x"
                              << outputBuf.m_Size << std::endl;

            WriteHex(bufferDataFile, static_cast<uint32_t>(outputBuffersBaseAddress) + outputBuf.m_Offset,
                     outputBuffers[i]->Map(), outputBuf.m_Size);
            outputBuffers[i]->Unmap();
        }
    }

    // Gather any profiling entries written by the firmware and add them to the global buffer
    if (profiling::g_CurrentConfiguration.m_EnableProfiling &&
        firmwareProfilingBuffer.size() > sizeof(ethosn_profiling_buffer))
    {
        std::map<uint8_t, profiling::ProfilingEntry> inProgressTimelineEvents;
        uint64_t mostRecentCorrectedKernelTimestamp = 0;
        const ethosn_profiling_buffer& data =
            reinterpret_cast<const ethosn_profiling_buffer&>(firmwareProfilingBuffer[0]);
        for (uint32_t i = 0; i < data.firmware_write_index; i++)
        {
            const ethosn_profiling_entry& entry = data.entries[i];
            // Timestamps from the model are in nanoseconds, rather than cycles
            constexpr int clockFrequencyMhz = 1000;
            std::pair<bool, profiling::ProfilingEntry> successAndEntry =
                profiling::ConvertProfilingEntry(entry, inProgressTimelineEvents, mostRecentCorrectedKernelTimestamp,
                                                 clockFrequencyMhz, timestampOffset);
            // Not all firmware profiling entries yield an entry we expose from the driver library
            if (successAndEntry.first)
            {
                profiling::g_ProfilingEntries.push_back(successAndEntry.second);
            }
        }
    }

    if (m_CompiledNetwork->m_IntermediateDataSize > 0)
    {
        if (!firmwareInterface->DumpDram(m_MappedIntermediateBuffer.data(), m_IntermediateDataBaseAddress,
                                         m_CompiledNetwork->m_IntermediateDataSize))
        {
            throw std::runtime_error("Failed to map intermediate buffer");
        }
    }
    return res.release();
}

std::pair<const char*, size_t> ModelNetworkImpl::MapIntermediateBuffers()
{
    /* Does nothing, as the data is always available after an inference. */
    return { reinterpret_cast<const char*>(m_MappedIntermediateBuffer.data()), m_MappedIntermediateBuffer.size() };
}

void ModelNetworkImpl::UnmapIntermediateBuffers(std::pair<const char*, size_t> mappedPtr)
{
    ETHOSN_UNUSED(mappedPtr);
    /* Does nothing, as the data is always available after an inference. */
}

}    // namespace driver_library
}    // namespace ethosn
