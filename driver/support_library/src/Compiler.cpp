//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "Compiler.hpp"
#include "ConcreteOperations.hpp"
#include "NetworkToGraphOfPartsConverter.hpp"
#include "SramAllocator.hpp"
#include "ThreadPool.hpp"

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

/// Check that the network is valid and throw a reason if not
/// * Ensure that all the operations which produce an operand have at least 1 consumer
///   (i.e. There are no dangling outputs).
void ValidateNetworkAndThrowIfBad(const Network& network)
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
                throw NotSupportedException(
                    "Network contains operations without any consumer i.e. There are dangling outputs");
            }
        }
    }
    // All check pass just return without throwing an error
}

}    // namespace

FrozenGraphOfParts CreateGraphOfParts(const Network& network,
                                      const HardwareCapabilities& capabilities,
                                      const EstimationOptions& estOpt,
                                      const CompilationOptions& compOpt,
                                      DebuggingContext& debuggingContext,
                                      ThreadPool& threadPool)
{
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(network, capabilities, estOpt, compOpt,
                                                                  debuggingContext, threadPool);
    GraphOfParts g = networkToGraphOfPartsConverter.ReleaseGraphOfParts();

    // Dump the GraphOfParts both before and after we optimize it.
    debuggingContext.Save(CompilationOptions::DebugLevel::Medium, "PreOptimizeGraphOfParts.dot",
                          [&](std::ofstream& s) { SaveGraphOfPartsToDot(g, s, DetailLevel::Low); });
    debuggingContext.Save(CompilationOptions::DebugLevel::Medium, "PreOptimizeGraphOfPartsDetailed.dot",
                          [&](std::ofstream& s) { SaveGraphOfPartsToDot(g, s, DetailLevel::High); });

    // Perform some optimizations on the GraphOfParts, to simplify it before generating any plans
    g.MergeChannelSelectors();

    g.SortAndCompact();

    debuggingContext.Save(CompilationOptions::DebugLevel::Medium, "GraphOfParts.dot",
                          [&](std::ofstream& s) { SaveGraphOfPartsToDot(g, s, DetailLevel::Low); });
    debuggingContext.Save(CompilationOptions::DebugLevel::Medium, "GraphOfPartsDetailed.dot",
                          [&](std::ofstream& s) { SaveGraphOfPartsToDot(g, s, DetailLevel::High); });

    return FrozenGraphOfParts(std::move(g));
}

Compiler::Compiler(const Network& network,
                   const FirmwareAndHardwareCapabilities& fwAndHwCapabilities,
                   const CompilationOptions& compilationOptions,
                   utils::Optional<const EstimationOptions&> estimationOptions)
    : m_Network(network)
    , m_Capabilities(fwAndHwCapabilities)
    , m_CompilationOptions(compilationOptions)
    , m_DebuggingContext(compilationOptions.m_DebugInfo)
    , m_EstimationOptions(estimationOptions)
{
    ValidateNetworkAndThrowIfBad(m_Network);

    if (m_Capabilities.GetNumberOfSrams() < 16)
    {
        // The FCAF channel rounding (SetStripeChannelsInfo in CommandStreamGeneratorUtils.hpp)
        // causes problems with small HW configs. We don't support these anyway, so disable FCAF so that
        // tests pass.
        m_CompilationOptions.m_EnableIntermediateCompression = false;
    }
}

Compiler::~Compiler()
{}

CompilerResult Compiler::Compile()
{
    DumpNetwork(m_DebuggingContext, m_Network);

    if (m_DebuggingContext.m_DebugInfo.m_DumpDebugFiles >= CompilationOptions::DebugLevel::Medium)
    {
        MakeDirectory(m_DebuggingContext.GetAbsolutePathOutputFileName("BestCombination").c_str());
    }

    // Default estimation options when none are provided (i.e. for compilation API rather than estimation API)
    EstimationOptions estimationOptions;
    if (m_EstimationOptions.has_value())
    {
        estimationOptions = m_EstimationOptions.value();
    }
    else
    {
        // We want the current numbers, as we are compiling for the current hardware
        estimationOptions.m_Current = true;
        // Estimate of the expected savings. We can't know this for sure as we don't have any input data.
        estimationOptions.m_ActivationCompressionSaving = 0.5f;
        // We have real weights, so use them rather than the override.
        estimationOptions.m_UseWeightCompressionOverride = false;
    }

    // ThreadPool object to be shared for all parallel computation for this compilation.
    // Uses an automatic number of threads based on environment variable
    ThreadPool threadPool(-1);

    auto startTime = std::chrono::high_resolution_clock::now();

    FrozenGraphOfParts graphOfParts = CreateGraphOfParts(m_Network, m_Capabilities, estimationOptions,
                                                         m_CompilationOptions, m_DebuggingContext, threadPool);

    auto duration = std::chrono::high_resolution_clock::now() - startTime;
    g_Logger.Debug("CreateGraphOfParts: %llu ms", duration.count() / (1000ULL * 1000ULL));

    startTime = std::chrono::high_resolution_clock::now();

    Combiner combiner(graphOfParts, m_Capabilities, m_CompilationOptions, estimationOptions, m_DebuggingContext);
    combiner.Run(threadPool);
    OpGraph opGraph = combiner.GetMergedOpGraphForBestCombination();

    duration = std::chrono::high_resolution_clock::now() - startTime;
    g_Logger.Debug("Combiner: %llu ms", duration.count() / (1000ULL * 1000ULL));
    g_Logger.Debug("Weights encoded: stage 1: %u, stage 2: %u", g_NumWeightEncodingsStage1, g_NumWeightEncodingsStage2);

    m_DebuggingContext.Save(
        CompilationOptions::DebugLevel::Medium, "BestCombination/1_CombinationBasic.dot",
        [&](std::ofstream& s) { SaveCombinationToDot(combiner.GetBestCombination(), s, DetailLevel::Low); });
    m_DebuggingContext.Save(
        CompilationOptions::DebugLevel::Medium, "BestCombination/1_CombinationDetailed.dot",
        [&](std::ofstream& s) { SaveCombinationToDot(combiner.GetBestCombination(), s, DetailLevel::High); });

    m_DebuggingContext.Save(CompilationOptions::DebugLevel::Medium, "BestCombination/2_MergedBasic.dot",
                            [&](std::ofstream& s) { SaveOpGraphToDot(opGraph, s, DetailLevel::Low); });
    m_DebuggingContext.Save(CompilationOptions::DebugLevel::Medium, "BestCombination/2_MergedDetailed.dot",
                            [&](std::ofstream& s) { SaveOpGraphToDot(opGraph, s, DetailLevel::High); });

    startTime = std::chrono::high_resolution_clock::now();

    // Perform optimisation steps on the merged OpGraph.
    // These optimisations would not have affected the choice of combination as they would apply equally
    // to all combinations, and so it is much more efficient to perform them after the Combiner has finished.
    opGraph.RemoveRedundantCopies();
    opGraph.ReducePackedBoundaryData();

    duration = std::chrono::high_resolution_clock::now() - startTime;
    g_Logger.Debug("RemoveRedundantCopies: %llu ms", duration.count() / (1000ULL * 1000ULL));

    m_DebuggingContext.Save(CompilationOptions::DebugLevel::Medium, "BestCombination/3_OptimisedBasic.dot",
                            [&](std::ofstream& s) { SaveOpGraphToDot(opGraph, s, DetailLevel::Low); });
    m_DebuggingContext.Save(CompilationOptions::DebugLevel::Medium, "BestCombination/3_OptimisedDetailed.dot",
                            [&](std::ofstream& s) { SaveOpGraphToDot(opGraph, s, DetailLevel::High); });

    startTime = std::chrono::high_resolution_clock::now();

    EstimatedOpGraph estimatedOpGraph =
        ethosn::support_library::EstimateOpGraph(opGraph, m_Capabilities, estimationOptions);

    duration = std::chrono::high_resolution_clock::now() - startTime;
    g_Logger.Debug("EstimateOpGraph: %llu ms", duration.count() / (1000ULL * 1000ULL));

    m_DebuggingContext.Save(CompilationOptions::DebugLevel::Medium, "BestCombination/4_EstimatedBasic.dot",
                            [&](std::ofstream& s) {
                                SaveEstimatedOpGraphToDot(opGraph, estimatedOpGraph, s, DetailLevel::Low, {}, {}, {});
                            });
    m_DebuggingContext.Save(CompilationOptions::DebugLevel::Medium, "BestCombination/4_EstimatedDetailed.dot",
                            [&](std::ofstream& s) {
                                SaveEstimatedOpGraphToDot(opGraph, estimatedOpGraph, s, DetailLevel::High, {}, {}, {});
                            });

    if (m_EstimationOptions.has_value())
    {
        // Not requesting compilation, so stop here.
        return { opGraph, combiner.GetBestCombination(), { estimatedOpGraph, {}, {}, {} } };
    }

    std::set<uint32_t> operationIds = m_Network.GetOperationIds();

    startTime = std::chrono::high_resolution_clock::now();

    CommandStreamGenerator commandStreamGenerator(opGraph, estimatedOpGraph, operationIds, m_Capabilities,
                                                  m_CompilationOptions, m_DebuggingContext);
    CompiledOpGraph compiledOpGraph = commandStreamGenerator.Generate();

    duration = std::chrono::high_resolution_clock::now() - startTime;
    g_Logger.Debug("CommandStreamGenerator: %llu ms", duration.count() / (1000ULL * 1000ULL));

    m_DebuggingContext.Save(
        CompilationOptions::DebugLevel::Medium, "BestCombination/5_CompiledBasic.dot",
        [&](std::ofstream& s) { SaveCompiledOpGraphToDot(opGraph, compiledOpGraph, s, DetailLevel::Low); });
    m_DebuggingContext.Save(
        CompilationOptions::DebugLevel::Medium, "BestCombination/5_CompiledDetailed.dot",
        [&](std::ofstream& s) { SaveCompiledOpGraphToDot(opGraph, compiledOpGraph, s, DetailLevel::High); });

    return { std::move(opGraph), combiner.GetBestCombination(), std::move(compiledOpGraph) };
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
                                  compilerBuffer.m_SourceOperationId, compilerBuffer.m_SourceOperationOutputIndex,
                                  compilerBuffer.m_DebugName);
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

void WriteString(std::ostream& out, const std::string& data)
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
        WriteString(out, data[i].m_DebugName);
    }
}

}    // namespace

void CompiledNetworkImpl::Serialize(std::ostream& out) const
{
    // Tag to identify the compiled network data structure using "FourCC" style
    out.write("ENCN", 4);

    // Version of data structure
    constexpr uint32_t major = 2;
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
