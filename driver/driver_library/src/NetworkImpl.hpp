//
// Copyright © 2018-2021 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../include/ethosn_driver_library/Buffer.hpp"
#include "../include/ethosn_driver_library/Inference.hpp"

#include <cstdint>
#include <string>
#include <vector>

// Version information range to check against when deserializing a Compiled Network
#define MAX_ETHOSN_COMPILED_NETWORK_MAJOR_VERSION_SUPPORTED 1
#define MIN_ETHOSN_COMPILED_NETWORK_MAJOR_VERSION_SUPPORTED 1

namespace ethosn
{
namespace driver_library
{

struct BufferInfo
{
public:
    constexpr BufferInfo()
        : BufferInfo(0, 0, 0)
    {}

    constexpr BufferInfo(uint32_t id, uint32_t offset, uint32_t size)
        : m_Id(id)
        , m_Offset(offset)
        , m_Size(size)
    {}

    bool operator==(const BufferInfo& rhs) const
    {
        return m_Id == rhs.m_Id && m_Offset == rhs.m_Offset && m_Size == rhs.m_Size;
    }

    uint32_t m_Id;
    uint32_t m_Offset;
    uint32_t m_Size;
};

/// The result of deserializing a Compiled Network from the Support Library (see DeserializeCompiledNetwork).
/// This contains offsets to data in the byte array from which this object was parsed, so you will likely need to
/// keep that data available if you want to use this object.
/// This is done to avoid copying the potentially large constant data buffers.
struct CompiledNetworkInfo
{
    size_t m_ConstantDmaDataOffset = 0;
    size_t m_ConstantDmaDataSize   = 0;

    size_t m_ConstantControlUnitDataOffset = 0;
    size_t m_ConstantControlUnitDataSize   = 0;

    std::vector<BufferInfo> m_InputBufferInfos;
    std::vector<BufferInfo> m_OutputBufferInfos;
    std::vector<BufferInfo> m_ConstantControlUnitDataBufferInfos;
    std::vector<BufferInfo> m_ConstantDmaDataBufferInfos;
    std::vector<BufferInfo> m_IntermediateDataBufferInfos;

    uint32_t m_IntermediateDataSize = 0;

    const uint8_t* CalculateConstantDmaDataPtr(const char* compiledNetworkData)
    {
        return reinterpret_cast<const uint8_t*>(compiledNetworkData) + m_ConstantDmaDataOffset;
    }

    const uint8_t* CalculateConstantControlUnitDataPtr(const char* compiledNetworkData)
    {
        return reinterpret_cast<const uint8_t*>(compiledNetworkData) + m_ConstantControlUnitDataOffset;
    }
};

/// @throws CompiledNetworkException if the given Compiled Network data is not valid.
CompiledNetworkInfo DeserializeCompiledNetwork(const char* data, size_t size);

/// Base class for all NetworkImpls.
/// This provides the functionality to dump a combined memory map.
class NetworkImpl
{
public:
    NetworkImpl(const char* compiledNetworkData, size_t compiledNetworkSizeData, bool alwaysCopyCompiledNetwork);

    virtual ~NetworkImpl()
    {}

    /// This simple base implementation only dumps the CMM file, rather than scheduling inferences.
    virtual Inference* ScheduleInference(Buffer* const inputBuffers[],
                                         uint32_t numInputBuffers,
                                         Buffer* const outputBuffers[],
                                         uint32_t numOutputBuffers) const;

    void SetDebugName(const char* name);

protected:
    enum CmmSection : uint8_t
    {
        Cmm_ConstantDma         = 0x1,
        Cmm_ConstantControlUnit = 0x2,
        Cmm_Inference           = 0x4,
        Cmm_Ifm                 = 0x8,
        Cmm_All                 = 0xFF,
    };

    void DumpCmmBasedOnEnvVar(Buffer* const inputBuffers[], uint32_t numInputBuffers) const;

    void DumpCmm(Buffer* const inputBuffers[],
                 uint32_t numInputBuffers,
                 const char* cmmFilename,
                 uint8_t sections) const;

    std::vector<uint32_t> BuildInferenceData(uint64_t constantControlUnitDataBaseAddress,
                                             uint64_t constantDmaDataBaseAddress,
                                             uint64_t inputBuffersBaseAddress,
                                             uint64_t outputBuffersBaseAddress,
                                             uint64_t intermediateDataBaseAddress) const;

    /// Some debugging operations and some backends require keeping around a copy of the compiled network,
    /// but we don't want to incur this memory cost for the standard case, so these fields may be left empty.
    /// @{
    std::vector<char> m_CompiledNetworkData;
    std::unique_ptr<CompiledNetworkInfo> m_CompiledNetwork;
    /// @}

    std::string m_DebugName;
};

}    // namespace driver_library
}    // namespace ethosn
