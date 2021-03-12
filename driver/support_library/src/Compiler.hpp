//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "DebuggingContext.hpp"
#include "Graph.hpp"
#include "Utils.hpp"
#include "nonCascading/BufferManager.hpp"

#include <ethosn_command_stream/CommandStreamBuffer.hpp>

#include <cstdint>
#include <cstring>
#include <memory>

namespace ethosn
{
namespace support_library
{

class CompiledNetwork;
class IStrategy;
class Pass;
class Section;
class IEstimationStrategy;

/// Compiles a user-constructed Network into a CompiledNetwork.
/// This is done in three stages:
///    - Conversion - converts the Network into an internal Graph.
///    - Preparation - modifies the Graph so that it can be split into Passes.
///    - Generation - produces the final outputs from the Graph and Passes.
class Compiler
{
public:
    Compiler(const Network& network,
             const FirmwareAndHardwareCapabilities& fwAndHwCapabilities,
             const CompilationOptions& compilationOptions,
             const EstimationOptions& estimationOptions);
    Compiler(const Compiler&) = delete;
    Compiler& operator=(const Compiler&) = delete;

    std::unique_ptr<CompiledNetwork> Compile();
    NetworkPerformanceData EstimatePerformance();
    ~Compiler();

private:
    /// Conversion
    /// @{
    void Convert();
    /// @}

    /// Preparation
    /// @{
    void Prepare();
    void Optimize();
    void CreatePasses();
    bool IsPrepared();
    void CreateSections();
    ///@}

    /// Generation
    /// @{
    void Generate();
    /// @}

    /// Debugging
    /// @{
    void DumpGraph(const std::string& filename);
    /// @}

    /// The input Network constructed by the user, set at creation time.
    const Network& m_Network;

    /// Compilation parameters, set at creation time.
    /// @{
    std::vector<std::unique_ptr<IStrategy>> m_AllowedStrategies;
    std::vector<command_stream::BlockConfig> m_AllowedBlockConfigs;
    HardwareCapabilities m_Capabilities;
    const CompilationOptions& m_CompilationOptions;
    bool m_EnableCascading;
    /// @}

    /// Performance estimation
    /// @{
    const EstimationOptions& m_EstimationOptions;
    bool m_PerfEstimate;
    NetworkPerformanceData PrivateEstimatePerformance();
    /// @}

    /// Intermediate data/results
    /// @{
    /// The internal graph of nodes. Modified as we progress through compilation.
    Graph m_Graph;
    /// The list of Passes we have built up so far.
    std::vector<std::unique_ptr<Pass>> m_Passes;
    /// The list of Sections we have built up so far.
    std::vector<std::unique_ptr<Section>> m_Sections;
    BufferManager m_BufferManager;
    /// @}

    /// Performance information
    /// @{
    NetworkPerformanceData m_PerformanceStream;
    /// @}

    /// Outputs
    /// @{
    command_stream::CommandStreamBuffer m_CommandStream;
    /// @}
};

class CompiledNetworkImpl : public CompiledNetwork
{
public:
    struct BufferInfoInternal
    {
    public:
        constexpr BufferInfoInternal()
            : m_Id(0)
            , m_Offset(0)
            , m_Size(0)
            , m_SourceOperationId(0)
            , m_SourceOperationOutputIndex(0)
        {}

        constexpr BufferInfoInternal(uint32_t id, uint32_t offset, uint32_t size)
            : m_Id(id)
            , m_Offset(offset)
            , m_Size(size)
            , m_SourceOperationId(0xFFFFFFFF)
            , m_SourceOperationOutputIndex(0xFFFFFFFF)
        {}

        constexpr BufferInfoInternal(uint32_t id,
                                     uint32_t offset,
                                     uint32_t size,
                                     uint32_t sourceOperationId,
                                     uint32_t sourceOperationOutputIndex)
            : m_Id(id)
            , m_Offset(offset)
            , m_Size(size)
            , m_SourceOperationId(sourceOperationId)
            , m_SourceOperationOutputIndex(sourceOperationOutputIndex)
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
    };

    CompiledNetworkImpl()
    {}

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

std::vector<std::unique_ptr<IStrategy>> GenerateAllowedStrategies(const CompilationOptions& m_Options);
std::vector<command_stream::BlockConfig> GenerateAllowedBlockConfigs(const CompilationOptions& m_Options);

uint32_t CalculateBufferSize(const TensorShape& shape, command_stream::DataFormat dataFormat);

}    // namespace support_library
}    // namespace ethosn
