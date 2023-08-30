//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "BufferManager.hpp"
#include "CombinerDFS.hpp"
#include "CommandStreamGenerator.hpp"
#include "DebuggingContext.hpp"
#include "Utils.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

namespace ethosn
{
namespace support_library
{

struct CompilerResult
{
    OpGraph opGraph;
    /// This is necessary to keep data alive which is referenced inside `compiledOpGraph` and `opGraph`.
    Combination combination;
    /// Some fields of this will be empty/null if estimation was requested.
    CompiledOpGraph compiledOpGraph;

    const NetworkPerformanceData& GetLegacyNetworkPerformanceData() const
    {
        return compiledOpGraph.m_EstimatedOpGraph.m_LegacyPerfData;
    }
};

/// Compiles a user-constructed Network into CompilerResult,
/// which contains the compiled network.
class Compiler
{
public:
    /// The presence (or lack) of `estimationOptions` determines if estimation or compilation is performed.
    Compiler(const Network& network,
             const FirmwareAndHardwareCapabilities& fwAndHwCapabilities,
             const CompilationOptions& compilationOptions,
             utils::Optional<const EstimationOptions&> estimationOptions);
    Compiler(const Compiler&) = delete;
    Compiler& operator=(const Compiler&) = delete;

    CompilerResult Compile();

    ~Compiler();

private:
    /// The input Network constructed by the user, set at creation time.
    const Network& m_Network;

    /// Compilation parameters, set at creation time.
    /// @{
    const HardwareCapabilities m_Capabilities;
    CompilationOptions m_CompilationOptions;
    DebuggingContext m_DebuggingContext;
    /// @}

    /// Only present for performance estimation.
    utils::Optional<const EstimationOptions&> m_EstimationOptions;
};

class CompiledNetworkImpl : public CompiledNetwork
{
public:
    struct BufferInfoInternal
    {
    public:
        BufferInfoInternal(uint32_t id,
                           uint32_t offset,
                           uint32_t size,
                           uint32_t sourceOperationId,
                           uint32_t sourceOperationOutputIndex,
                           std::string debugName)
            : m_Id(id)
            , m_Offset(offset)
            , m_Size(size)
            , m_SourceOperationId(sourceOperationId)
            , m_SourceOperationOutputIndex(sourceOperationOutputIndex)
            , m_DebugName(std::move(debugName))
        {}

        bool operator==(const BufferInfoInternal& rhs) const
        {
            return m_Id == rhs.m_Id && m_Offset == rhs.m_Offset && m_Size == rhs.m_Size &&
                   m_SourceOperationId == rhs.m_SourceOperationId &&
                   m_SourceOperationOutputIndex == rhs.m_SourceOperationOutputIndex;
        }

        /// Unique ID for this buffer, across all types of buffers exposed by CompiledNetwork.
        /// IDs are contiguous across all buffer types and start at zero.
        /// IDs are *not* necessarily contiguous within each type of buffer.
        uint32_t m_Id;
        /// Offset of the start of this buffer relative to a block of data containing all buffers of this type.
        uint32_t m_Offset;
        /// Size (in bytes) of this buffer.
        uint32_t m_Size;
        uint32_t m_SourceOperationId;             ///< Only relevant for input and output buffer infos.
        uint32_t m_SourceOperationOutputIndex;    ///< Only relevant for input and output buffer infos.
        /// Used for dumping buffers as files in the driver library.
        std::string m_DebugName;
    };

    CompiledNetworkImpl();

    CompiledNetworkImpl(const std::vector<uint8_t>& constantDmaData,
                        const std::vector<uint8_t>& constantControlUnitData,
                        const std::map<uint32_t, CompilerBufferInfo>& buffers,
                        const std::set<uint32_t>& operationIds);

    /// Public API implementation
    /// @{
    virtual const std::set<uint32_t>& GetOperationIds() const override
    {
        return m_OperationIds;
    }

    const std::vector<InputBufferInfo>& GetInputBufferInfos() const override
    {
        return m_InputBufferInfosPublic;
    }

    const std::vector<OutputBufferInfo>& GetOutputBufferInfos() const override
    {
        return m_OutputBufferInfosPublic;
    }

    uint32_t GetIntermediateBufferSize() const override
    {
        return m_IntermediateBufferSizePublic;
    }

    void Serialize(std::ostream& out) const override;
    /// @}

    const std::vector<BufferInfoInternal>& GetInputBufferInfosInternal() const
    {
        return m_InputBufferInfos;
    }

    const std::vector<BufferInfoInternal>& GetOutputBufferInfosInternal() const
    {
        return m_OutputBufferInfos;
    }
    const std::vector<uint8_t>& GetConstantDmaData() const
    {
        return m_ConstantDmaData;
    }

    const std::vector<uint8_t>& GetConstantControlUnitData() const
    {
        return m_ConstantControlUnitData;
    }

    const std::vector<BufferInfoInternal>& GetConstantControlUnitDataBufferInfos() const
    {
        return m_ConstantControlUnitDataBufferInfos;
    }

    const std::vector<BufferInfoInternal>& GetConstantDmaDataBufferInfos() const
    {
        return m_ConstantDmaDataBufferInfos;
    }

    const std::vector<BufferInfoInternal>& GetIntermediateDataBufferInfos() const
    {
        return m_IntermediateDataBufferInfos;
    }

private:
    /// Data exposed via public API.
    /// @{
    std::set<uint32_t> m_OperationIds;

    std::vector<InputBufferInfo> m_InputBufferInfosPublic;
    std::vector<OutputBufferInfo> m_OutputBufferInfosPublic;
    uint32_t m_IntermediateBufferSizePublic;
    /// @}

    /// Internal use only
    /// @{
    std::vector<uint8_t> m_ConstantDmaData;
    std::vector<uint8_t> m_ConstantControlUnitData;

    std::vector<BufferInfoInternal> m_InputBufferInfos;
    std::vector<BufferInfoInternal> m_OutputBufferInfos;
    std::vector<BufferInfoInternal> m_ConstantControlUnitDataBufferInfos;
    std::vector<BufferInfoInternal> m_ConstantDmaDataBufferInfos;
    std::vector<BufferInfoInternal> m_IntermediateDataBufferInfos;
    /// @}
};

template <class IOBufferInfo>
bool SortByOperationId(const IOBufferInfo& buf1, const IOBufferInfo& buf2)
{
    return buf1.m_SourceOperationId < buf2.m_SourceOperationId;
}

}    // namespace support_library
}    // namespace ethosn
