//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
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
    CompiledNetworkImpl()
        : m_ConstantDmaData()
        , m_ConstantControlUnitData()
        , m_InputBufferInfos()
        , m_OutputBufferInfos()
        , m_ConstantControlUnitDataBufferInfos()
        , m_ConstantDmaDataBufferInfos()
        , m_IntermediateDataBufferInfos()
        , m_OperationIds()
    {}

    CompiledNetworkImpl(const std::vector<uint8_t>& constantDmaData,
                        const std::vector<uint8_t>& constantControlUnitData,
                        const std::map<uint32_t, CompilerBufferInfo>& buffers,
                        const std::set<uint32_t>& operationIds);

    virtual const std::vector<uint8_t>& GetConstantDmaData() const override
    {
        return m_ConstantDmaData;
    }

    virtual const std::vector<uint8_t>& GetConstantControlUnitData() const override
    {
        return m_ConstantControlUnitData;
    }

    virtual const std::set<uint32_t>& GetOperationIds() const override
    {
        return m_OperationIds;
    }

    virtual const std::vector<InputBufferInfo>& GetInputBufferInfos() const override
    {
        return m_InputBufferInfos;
    }

    virtual const std::vector<OutputBufferInfo>& GetOutputBufferInfos() const override
    {
        return m_OutputBufferInfos;
    }

    virtual const std::vector<BufferInfo>& GetConstantControlUnitDataBufferInfos() const override
    {
        return m_ConstantControlUnitDataBufferInfos;
    }

    virtual const std::vector<BufferInfo>& GetConstantDmaDataBufferInfos() const override
    {
        return m_ConstantDmaDataBufferInfos;
    }

    virtual const std::vector<BufferInfo>& GetIntermediateDataBufferInfos() const override
    {
        return m_IntermediateDataBufferInfos;
    }

    virtual uint32_t GetIntermediateDataSize() const override;

    template <typename T>
    void Serialize(std::ostream& out, const std::vector<T>& data) const;

    virtual void Serialize(std::ostream& out) const override;

    template <typename T>
    void Deserialize(std::istream& in, std::vector<T>& data);

    virtual void Deserialize(std::istream& in);

private:
    template <typename T>
    T Read(std::istream& in);

    std::vector<uint8_t> m_ConstantDmaData;
    std::vector<uint8_t> m_ConstantControlUnitData;

    std::vector<InputBufferInfo> m_InputBufferInfos;
    std::vector<OutputBufferInfo> m_OutputBufferInfos;
    std::vector<BufferInfo> m_ConstantControlUnitDataBufferInfos;
    std::vector<BufferInfo> m_ConstantDmaDataBufferInfos;
    std::vector<BufferInfo> m_IntermediateDataBufferInfos;

    std::set<uint32_t> m_OperationIds;
    std::map<uint32_t, std::string> m_OperationIdsFailureReasons;
};

std::vector<std::unique_ptr<IStrategy>> GenerateAllowedStrategies(const CompilationOptions& m_Options);
std::vector<command_stream::BlockConfig> GenerateAllowedBlockConfigs(const CompilationOptions& m_Options);

uint32_t CalculateBufferSize(const TensorShape& shape, command_stream::DataFormat dataFormat);

}    // namespace support_library
}    // namespace ethosn
