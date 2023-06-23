//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "Compiler.hpp"
#include "ConcreteOperations.hpp"
#include "SramAllocator.hpp"
#include "cascading/Cascading.hpp"

#include <ethosn_utils/Enums.hpp>

#include <fstream>
#include <numeric>
#include <sstream>
#include <vector>

#include <ethosn_utils/Filesystem.hpp>

using namespace ethosn::utils;

namespace ethosn
{
namespace support_library
{

using namespace utils;

namespace
{

void DumpNetwork(const DebuggingContext& debuggingContext, const Network& network)
{
    debuggingContext.Save(CompilationOptions::DebugLevel::Medium, "Network.dot",
                          [&](std::ofstream& s) { SaveNetworkToDot(network, s, DetailLevel::Low); });
    debuggingContext.Save(CompilationOptions::DebugLevel::Medium, "NetworkDetailed.dot",
                          [&](std::ofstream& s) { SaveNetworkToDot(network, s, DetailLevel::High); });
}

/// Check that the network is valid:
/// * Ensure that all the operations which produce an operand have at least 1 consumer
///   (i.e. There are no dangling outputs).
bool ValidateNetwork(const Network& network)
{
    for (detail::OperationList::const_iterator operation = network.begin(); operation != network.end(); ++operation)
    {
        auto outputOperands = (*operation)->GetOutputs();
        for (auto&& operand : outputOperands)
        {
            // Constants are special because they can correspond to convolutions but we don't actually connect them in the graph
            // These constants will have no outputs and the network will still be valid.
            bool isConstant = dynamic_cast<Constant*>(operation->get()) != nullptr;
            if (operand.GetConsumers().empty() && !isConstant)
            {
                return false;
            }
        }
    }
    return true;
}

}    // namespace

Compiler::Compiler(const Network& network,
                   const FirmwareAndHardwareCapabilities& fwAndHwCapabilities,
                   const CompilationOptions& compilationOptions,
                   const EstimationOptions& estimationOptions)
    : m_Network(network)
    , m_Capabilities(fwAndHwCapabilities)
    , m_CompilationOptions(compilationOptions)
    , m_DebuggingContext(compilationOptions.m_DebugInfo)
    , m_EstimationOptions(estimationOptions)
{
    if (m_Capabilities.GetNumberOfSrams() < 16)
    {
        // The FCAF channel rounding (SetStripeChannelsInfo in CascadingCommandStreamGeneratorUtils.hpp)
        // causes problems with small HW configs. We don't support these anyway, so disable FCAF so that
        // tests pass.
        m_CompilationOptions.m_EnableIntermediateCompression = false;
    }
}

Compiler::~Compiler()
{}

std::unique_ptr<CompiledNetwork> Compiler::Compile()
{
    DumpNetwork(m_DebuggingContext, m_Network);

    bool validNetwork = ValidateNetwork(m_Network);
    if (!validNetwork)
    {
        return std::unique_ptr<CompiledNetworkImpl>(nullptr);
    }

    try
    {
        return RunCascading(m_Network, utils::EmptyOptional{}, m_CompilationOptions, m_Capabilities, m_DebuggingContext)
            .compiledOpGraph.m_CompiledNetwork;
    }
    catch (const std::runtime_error& e)
    {
        // Either we failed compilation or there was not enough SRAM to convert NHWCB to NHWC
        // NNXSW-2802: Temporary fix to print the error but need better approach  for error reporting from support library.
        g_Logger.Error("Error: %s", e.what());
        return std::unique_ptr<CompiledNetworkImpl>(nullptr);
    }
}

NetworkPerformanceData Compiler::EstimatePerformance()
{
    DumpNetwork(m_DebuggingContext, m_Network);

    NetworkPerformanceData performance;

    try
    {
        performance =
            RunCascading(m_Network, m_EstimationOptions, m_CompilationOptions, m_Capabilities, m_DebuggingContext)
                .GetNetworkPerformanceData();
    }
    catch (const std::exception& e)
    {
        g_Logger.Warning("Cascading estimation failed with: %s", e.what());
        throw NotSupportedException("Estimation didn't find any valid performance data to return");
    }

    return performance;
}

CompiledNetworkImpl::CompiledNetworkImpl()
    : m_IntermediateBufferSizePublic(0U)
{}

CompiledNetworkImpl::CompiledNetworkImpl(const std::vector<uint8_t>& constantDmaData,
                                         const std::vector<uint8_t>& constantControlUnitData,
                                         const std::map<uint32_t, CompilerBufferInfo>& buffers,
                                         const std::set<uint32_t>& operationIds)
    : m_OperationIds(operationIds)
    , m_IntermediateBufferSizePublic(0U)
    , m_ConstantDmaData(constantDmaData)
    , m_ConstantControlUnitData(constantControlUnitData)
{
    // Convert the set of buffers from the BufferManager into the format that CompiledNetwork exposes.
    for (auto internalBufferIt : buffers)
    {
        uint32_t bufferId = internalBufferIt.first;

        const CompilerBufferInfo& compilerBuffer = internalBufferIt.second;
        if (compilerBuffer.m_Location != BufferLocation::Dram)
        {
            // Sram buffers do not need to be exposed.
            continue;
        }

        BufferInfoInternal buffer(bufferId, compilerBuffer.m_Offset, compilerBuffer.m_Size,
                                  compilerBuffer.m_SourceOperationId, compilerBuffer.m_SourceOperationOutputIndex);
        switch (compilerBuffer.m_Type)
        {
            case BufferType::Input:
            {
                InputBufferInfo inputBuffer(compilerBuffer.m_Size, compilerBuffer.m_SourceOperationId,
                                            compilerBuffer.m_SourceOperationOutputIndex);

                m_InputBufferInfos.push_back(buffer);
                m_InputBufferInfosPublic.push_back(inputBuffer);

                // The input buffers need to sorted by m_SourceOperationId.
                // m_SourceOperationId increases sequentially as the caller adds operands.
                // This will ensure that the user can pass their buffers to the driver library API
                // (ScheduleInference()) in the same order as they were added to the original
                // network.
                std::sort(m_InputBufferInfos.begin(), m_InputBufferInfos.end(), SortByOperationId<BufferInfoInternal>);
                std::sort(m_InputBufferInfosPublic.begin(), m_InputBufferInfosPublic.end(),
                          SortByOperationId<InputBufferInfo>);
                break;
            }
            case BufferType::Output:
            {
                OutputBufferInfo outputBuffer(compilerBuffer.m_Size, compilerBuffer.m_SourceOperationId,
                                              compilerBuffer.m_SourceOperationOutputIndex);
                m_OutputBufferInfos.push_back(buffer);
                m_OutputBufferInfosPublic.push_back(outputBuffer);

                // The output buffers need to sorted by m_SourceOperationId.
                // m_SourceOperationId increases sequentially as the caller adds operands.
                // This will ensure that the user can pass their buffers to the driver library API
                // (ScheduleInference()) in the same order as they were added to the original
                // network.
                std::sort(m_OutputBufferInfos.begin(), m_OutputBufferInfos.end(),
                          SortByOperationId<BufferInfoInternal>);
                std::sort(m_OutputBufferInfosPublic.begin(), m_OutputBufferInfosPublic.end(),
                          SortByOperationId<OutputBufferInfo>);
                break;
            }
            case BufferType::Intermediate:
            {
                m_IntermediateDataBufferInfos.push_back(buffer);
                m_IntermediateBufferSizePublic =
                    std::max(m_IntermediateBufferSizePublic, buffer.m_Offset + buffer.m_Size);
                break;
            }
            case BufferType::ConstantControlUnit:
            {
                m_ConstantControlUnitDataBufferInfos.push_back(buffer);
                break;
            }
            case BufferType::ConstantDma:
            {
                m_ConstantDmaDataBufferInfos.push_back(buffer);
                break;
            }
            default:
                assert(false);
        }
    }
}

namespace
{

void Write(std::ostream& out, const uint32_t data)
{
    // Write in little-endian order, regardless of host endianness
    out.put(static_cast<char>(data & 0xFF));
    out.put(static_cast<char>((data >> 8) & 0xFF));
    out.put(static_cast<char>((data >> 16) & 0xFF));
    out.put(static_cast<char>((data >> 24) & 0xFF));
}

void WriteByteArray(std::ostream& out, const std::vector<uint8_t>& data)
{
    Write(out, static_cast<uint32_t>(data.size()));
    out.write(reinterpret_cast<const char*>(data.data()), data.size());
}

void WriteBufferInfoArray(std::ostream& out, const std::vector<CompiledNetworkImpl::BufferInfoInternal>& data)
{
    Write(out, static_cast<uint32_t>(data.size()));
    for (size_t i = 0; i < data.size(); ++i)
    {
        Write(out, data[i].m_Id);
        Write(out, data[i].m_Offset);
        Write(out, data[i].m_Size);
    }
}

}    // namespace

void CompiledNetworkImpl::Serialize(std::ostream& out) const
{
    // Tag to identify the compiled network data structure using "FourCC" style
    out.write("ENCN", 4);

    // Version of data structure
    constexpr uint32_t major = 1;
    constexpr uint32_t minor = 0;
    constexpr uint32_t patch = 0;

    Write(out, major);
    Write(out, minor);
    Write(out, patch);

    // Main data
    WriteByteArray(out, m_ConstantDmaData);
    WriteByteArray(out, m_ConstantControlUnitData);
    WriteBufferInfoArray(out, m_InputBufferInfos);
    WriteBufferInfoArray(out, m_OutputBufferInfos);
    WriteBufferInfoArray(out, m_ConstantControlUnitDataBufferInfos);
    WriteBufferInfoArray(out, m_ConstantDmaDataBufferInfos);
    WriteBufferInfoArray(out, m_IntermediateDataBufferInfos);
}

}    // namespace support_library
}    // namespace ethosn
