//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "Compiler.hpp"

#include "GraphNodes.hpp"
#include "IEstimationStrategy.hpp"
#include "Optimization.hpp"
#include "SramAllocator.hpp"
#include "cascading/Cascading.hpp"
#include "nonCascading/ConversionPass.hpp"
#include "nonCascading/McePlePass.hpp"
#include "nonCascading/NonCascading.hpp"
#include "nonCascading/PlePass.hpp"
#include "nonCascading/Section.hpp"
#include "nonCascading/Strategies.hpp"

#include <fstream>
#include <numeric>
#include <sstream>
#include <vector>

namespace ethosn
{
namespace support_library
{

using namespace utils;

uint32_t CalculateBufferSize(const TensorShape& shape, command_stream::DataFormat dataFormat)
{
    assert(dataFormat == command_stream::DataFormat::NHWC || dataFormat == command_stream::DataFormat::NCHW ||
           dataFormat == command_stream::DataFormat::NHWCB ||
           dataFormat == command_stream::DataFormat::NHWCB_COMPRESSED ||
           dataFormat == command_stream::DataFormat::FCAF_WIDE || dataFormat == command_stream::DataFormat::FCAF_DEEP);

    switch (dataFormat)
    {
        case command_stream::DataFormat::NHWCB_COMPRESSED:
            return TotalSizeBytesNHWCBCompressed(shape);
        case command_stream::DataFormat::FCAF_DEEP:
            return TotalSizeBytesFCAFDeep(shape);
        case command_stream::DataFormat::FCAF_WIDE:
            return TotalSizeBytesFCAFWide(shape);
        case command_stream::DataFormat::NHWCB:
            return TotalSizeBytesNHWCB(shape);
        default:
            return TotalSizeBytes(shape);
    }
}

std::vector<std::unique_ptr<IStrategy>> GenerateAllowedStrategies(const CompilationOptions& m_Options)
{
    std::vector<std::unique_ptr<IStrategy>> result;
    // We try the "best" strategies first until we find one which is appropriate
    // This may change in the future when we use a dynamic programming approach
    if (m_Options.m_Strategy3)
    {
        result.push_back(std::make_unique<Strategy3>());
    }
    if (m_Options.m_Strategy0)
    {
        result.push_back(std::make_unique<Strategy0>());
    }
    if (m_Options.m_Strategy1)
    {
        result.push_back(std::make_unique<Strategy1>());
    }
    if (m_Options.m_Strategy6)
    {
        result.push_back(std::make_unique<Strategy6>());
    }
    if (m_Options.m_Strategy4)
    {
        result.push_back(std::make_unique<Strategy4>());
    }
    if (m_Options.m_Strategy7)
    {
        result.push_back(std::make_unique<Strategy7>());
    }
    return result;
}

std::vector<command_stream::BlockConfig> GenerateAllowedBlockConfigs(const CompilationOptions& m_Options)
{
    using namespace command_stream;
    std::vector<BlockConfig> result;

    if (m_Options.m_BlockConfig16x16)
    {
        result.emplace_back(16u, 16u);
    }
    if (m_Options.m_BlockConfig32x8)
    {
        result.emplace_back(32u, 8u);
    }
    if (m_Options.m_BlockConfig8x32)
    {
        result.emplace_back(8u, 32u);
    }
    if (m_Options.m_BlockConfig16x8)
    {
        result.emplace_back(16u, 8u);
    }
    if (m_Options.m_BlockConfig8x16)
    {
        result.emplace_back(8u, 16u);
    }
    if (m_Options.m_BlockConfig8x8)
    {
        result.emplace_back(8u, 8u);
    }
    return result;
}

Compiler::Compiler(const Network& network,
                   const FirmwareAndHardwareCapabilities& fwAndHwCapabilities,
                   const CompilationOptions& compilationOptions,
                   const EstimationOptions& estimationOptions)
    : m_Network(network)
    , m_AllowedStrategies(GenerateAllowedStrategies(compilationOptions))
    , m_AllowedBlockConfigs(GenerateAllowedBlockConfigs(compilationOptions))
    , m_Capabilities(fwAndHwCapabilities)
    , m_CompilationOptions(compilationOptions)
    , m_EnableCascading(false)
    , m_EstimationOptions(estimationOptions)
    , m_PerfEstimate(false)
{
    SetDebuggingContext(DebuggingContext(&compilationOptions.m_DebugInfo));
}

Compiler::~Compiler()
{}

std::unique_ptr<CompiledNetwork> Compiler::Compile()
{
    m_PerfEstimate = false;

    try
    {
        Convert();
        Prepare();
        Generate();
    }
    catch (const NotSupportedException& e)
    {
        // Either we failed compilation or there was not enough SRAM to convert NHWCB to NHWC
        // NNXSW-2802: Temporary fix to print the error but need better approach  for error reporting from support library.
        g_Logger.Error("Error: %s", e.what());
        return std::unique_ptr<CompiledNetworkImpl>(nullptr);
    }

    // The compiler will need to split the network into supported subgraphs and have the appropriate ids for each.
    // See the Support Library public interface design note for more details.
    // For now we're just passing the full network ids through.
    std::set<uint32_t> compiledOperationIds = m_Network.GetOperationIds();

    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = std::make_unique<CompiledNetworkImpl>(
        m_BufferManager.GetConstantDmaData(), m_BufferManager.GetConstantControlUnitData(),
        m_BufferManager.GetBuffers(), compiledOperationIds);

    return compiledNetwork;
}

NetworkPerformanceData Compiler::EstimatePerformance()
{
    bool nonCascadedPerformanceValid = false;
    bool cascadedPerformanceValid    = false;
    NetworkPerformanceData nonCascadedPerformance, cascadedPerformance;
    const CompilerAlgorithm& compilerAlgorithm = m_CompilationOptions.m_CompilerAlgorithm;
    // An engineer can force to use non cascaded estimation only by setting
    // 'COMPILER_ALGORITHM = NonCascadingOnly' into the configuration file
    if (compilerAlgorithm == CompilerAlgorithm::Auto || compilerAlgorithm == CompilerAlgorithm::NonCascadingOnly)
    {
        try
        {
            m_EnableCascading           = false;
            nonCascadedPerformance      = PrivateEstimatePerformance();
            nonCascadedPerformanceValid = true;
        }
        catch (...)
        {
            //Nothing to do. nonCascadedPerformanceValid == false already
        }
    }
    // An engineer can force to use cascaded estimation only by setting
    // 'COMPILER_ALGORITHM = CascadingOnly' into the configuration file
    if (compilerAlgorithm == CompilerAlgorithm::Auto || compilerAlgorithm == CompilerAlgorithm::CascadingOnly)
    {
        if (m_EstimationOptions.m_Current == false)
        {
            try
            {
                m_EnableCascading        = true;
                cascadedPerformance      = PrivateEstimatePerformance();
                cascadedPerformanceValid = true;
            }
            catch (...)
            {
                //Nothing to do. cascadedPerformanceValid == false already
            }
        }
    }
    if (!nonCascadedPerformanceValid && !cascadedPerformanceValid)
    {
        throw NotSupportedException("Estimation didn't find any valid performance data to return");
    }
    if (nonCascadedPerformanceValid && !cascadedPerformanceValid)
    {
        return nonCascadedPerformance;
    }
    if (!nonCascadedPerformanceValid && cascadedPerformanceValid)
    {
        return cascadedPerformance;
    }
    // Both of the performances are valid, try to see which one is the best
    if (utils::IsLeftMoreDataPerformantThanRight(nonCascadedPerformance, cascadedPerformance))
    {
        return nonCascadedPerformance;
    }
    else
    {
        return cascadedPerformance;
    }
}

NetworkPerformanceData Compiler::PrivateEstimatePerformance()
{
    // Sets the performance estimate flag
    m_PerfEstimate = true;

    try
    {
        Convert();
        if (!m_EnableCascading)
        {
            Prepare();
        }
    }
    catch (const NotSupportedException&)
    {
        // Conversion and preparation can throw by not creating a valid graph but we should still be able to estimate it.
    }
    if (m_EnableCascading)
    {
        Optimize();
    }
    if (!m_EnableCascading)
    {
        NonCascading nonCascadingEstimate(m_EstimationOptions, m_CompilationOptions, m_Capabilities);
        m_PerformanceStream = nonCascadingEstimate.Estimate(m_Graph);
    }
    else
    {
        Cascading cascadingEstimate(m_EstimationOptions, m_CompilationOptions, m_Capabilities);
        m_PerformanceStream = cascadingEstimate.Estimate(m_Graph);
    }

    return m_PerformanceStream;
}

void Compiler::Convert()
{
    m_Graph = Graph(m_Network, m_Capabilities, m_EstimationOptions, m_CompilationOptions.m_StrictPrecision);

    DumpGraph("GraphInitial");
}

void Compiler::Optimize()
{
    OptimizeGraph(m_Graph);
}

void Compiler::Prepare()
{
    // This is an iterative process, where we modify the graph as necessary to prepare it for Generation.
    uint32_t numIterations = 0;
    // Set an upper limit for the number of iterations in case we have a bug somewhere.
    // This should not be required because we only keep iterating if we make a change to the graph, and we should
    // only change something if we know it will help. However if we have a bug we may get stuck in a case where we
    // repeatedly modify the graph thinking it will help, but it does not.
    // Note that this limit is set based on the size of the *initial* graph (the graph may grow in size).
    const uint32_t maxIterations = static_cast<uint32_t>(m_Graph.GetNodes().size()) * 10;
    while (true)
    {
        DumpGraph(std::string("GraphPrepareIteration") + std::to_string(numIterations) + "_Pre");

        Optimize();
        CreatePasses();

        DumpGraph(std::string("GraphPrepareIteration") + std::to_string(numIterations) + "_Post");

        if (IsPrepared())
        {
            CreateSections();
            break;
        }

        ++numIterations;

        // Modify graph based on previous attempt. Make a copy as we may add/remove nodes as we fix.
        std::vector<Node*> nodes = m_Graph.GetNodesSorted();
        bool madeChange          = false;    // Record if we were able to make a change to the graph
        // First try making less severe changes and then only escalate to more severe changes if necessary.
        // This prevents making potentially suboptimal changes to the graph that aren't necessary.
        for (FixGraphSeverity severity = FixGraphSeverity::Lowest; severity <= FixGraphSeverity::Highest;
             severity                  = utils::NextEnumValue(severity))
        {
            for (auto& n : nodes)
            {
                madeChange |= n->FixGraph(m_Graph, severity);
                // Note we don't break immedately if a change was made because for large graphs it might be very
                // slow making only one change at a time.
            }
            if (madeChange)
            {
                break;
            }
        }

        if (!madeChange || numIterations > maxIterations)
        {
            std::string errorMsg = std::string("Unable to prepare graph after ") + std::to_string(numIterations) +
                                   " iterations (max: " + std::to_string(maxIterations) +
                                   "). madeChange = " + (madeChange ? "true" : "false") + ".";

            errorMsg += "The operation(s) with the following ids have failed to compile:";

            std::vector<uint32_t> failedOpIds;

            // Find which nodes failed and correlate them with the original operations
            for (auto& n : nodes)
            {
                if (!(n->IsPrepared()))
                {
                    for (auto id : n->GetCorrespondingOperationIds())
                    {
                        failedOpIds.push_back(id);
                    }
                }
            }

            // Remove duplicates
            std::sort(failedOpIds.begin(), failedOpIds.end());

            std::vector<uint32_t>::iterator iter =
                std::unique(failedOpIds.begin(), failedOpIds.begin() + failedOpIds.size());

            failedOpIds.resize(std::distance(failedOpIds.begin(), iter));

            for (auto o : failedOpIds)
            {
                errorMsg += " " + std::to_string(o);
            }

            throw NotSupportedException(errorMsg.c_str());
        }

        // Clear passes for next attempt
        m_Passes.clear();
        for (auto& n : m_Graph.GetNodes())
        {
            n->Reset();
        }
    }
}

bool Compiler::IsPrepared()
{
    for (const auto& n : m_Graph.GetNodes())
    {
        if (!n->IsPrepared())
        {
            return false;
        }
    }

    return true;
}

void Compiler::CreatePasses()
{
    std::vector<IStrategy*> strategies = utils::GetRawPointers(m_AllowedStrategies);
    std::vector<Node*> sortedNodes     = m_Graph.GetNodesSorted();
    SramAllocator sramAllocator(m_Capabilities.GetTotalSramSize() / m_Capabilities.GetNumberOfSrams());

    // forward estimate flag is passed on to the function CreateGreedily to allow FCAF for
    // strategies 6, 7 and arbitrary tensor shape. This happens if the forward-looking
    // SPA is configured.
    bool forwardEst = m_PerfEstimate && !m_EstimationOptions.m_Current;

    for (Node* n : sortedNodes)
    {
        if (n->GetPass() == nullptr)
        {
            const size_t passId = m_Passes.size();
            std::unique_ptr<Pass> p;
            if (!p)
            {
                p = McePlePass::CreateGreedily(m_Capabilities, passId, strategies, m_AllowedBlockConfigs,
                                               m_CompilationOptions.m_EnableIntermediateCompression,
                                               !m_CompilationOptions.m_DisableWinograd, n, sramAllocator, forwardEst);
            }
            if (!p)
            {
                p = PlePass::CreateGreedily(m_Capabilities, passId, n, sramAllocator);
            }
            if (!p)
            {
                p = ConversionPass::CreateGreedily(m_Capabilities, passId, n, sramAllocator);
            }

            if (p)
            {
                m_Passes.push_back(std::move(p));
            }
            n->PrepareAfterPassAssignment(sramAllocator);
        }
    }
}

void Compiler::CreateSections()
{
    // NNXSW-1221: Implement a search algorithm to partition the network into sections
    // For now, each section will only have one pass (SISO or MISO)
    for (auto& p : m_Passes)
    {
        command_stream::SectionType sectionType = (p->GetNodes().front()->GetInputs().size() > 1)
                                                      ? command_stream::SectionType::MISO
                                                      : command_stream::SectionType::SISO;

        std::string sectionId = std::to_string(m_Sections.size());

        std::unique_ptr<ethosn::support_library::Section> section =
            std::make_unique<Section>(sectionId, sectionType, p.get());

        p->SetSection(section.get());

        m_Sections.push_back(std::move(section));
    }
}

void Compiler::Generate()
{
    const DebuggingContext& debuggingContext = GetConstDebuggingContext();
    std::vector<Node*> sorted                = m_Graph.GetNodesSorted();

    // If an initial dump is requested, add the sram dump command at the head of the stream.
    if (debuggingContext.m_DebugInfo->m_InitialSramDump)
    {
        ethosn::command_stream::DumpSram cmdStrDumpSram;
        const char dumpName[] = "initial_ce";
        static_assert(sizeof(dumpName) <= sizeof(cmdStrDumpSram.m_Filename()), "");
        std::copy(std::begin(dumpName), std::end(dumpName), cmdStrDumpSram.m_Filename().begin());
        m_CommandStream.EmplaceBack(cmdStrDumpSram);
    }

    for (Node* n : sorted)
    {
        n->Generate(m_CommandStream, m_BufferManager, debuggingContext.m_DebugInfo->m_DumpRam);
    }

    DumpGraph("GraphFinal");

    m_BufferManager.AddCommandStream(m_CommandStream);

    m_BufferManager.Allocate();
}

void Compiler::DumpGraph(const std::string& filename)
{
    const DebuggingContext& debuggingContext = GetConstDebuggingContext();
    std::string finalFileName("");
    if (m_EnableCascading)
    {
        finalFileName += "Cascaded_";
    }
    else
    {
        finalFileName += "NonCascaded_";
    }
    finalFileName += filename;
    finalFileName += ".dot";
    debuggingContext.DumpGraph(CompilationOptions::DebugLevel::Medium, m_Graph, finalFileName);
}

CompiledNetworkImpl::CompiledNetworkImpl(const std::vector<uint8_t>& constantDmaData,
                                         const std::vector<uint8_t>& constantControlUnitData,
                                         const std::map<uint32_t, CompilerBufferInfo>& buffers,
                                         const std::set<uint32_t>& operationIds)
    : m_OperationIds(operationIds)
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
                m_InputBufferInfos.push_back(buffer);
                m_InputBufferInfosPublic.emplace_back(compilerBuffer.m_Size, compilerBuffer.m_SourceOperationId,
                                                      compilerBuffer.m_SourceOperationOutputIndex);
                break;
            }
            case BufferType::Output:
            {
                m_OutputBufferInfos.push_back(buffer);
                m_OutputBufferInfosPublic.emplace_back(compilerBuffer.m_Size, compilerBuffer.m_SourceOperationId,
                                                       compilerBuffer.m_SourceOperationOutputIndex);
                break;
            }
            case BufferType::Intermediate:
            {
                m_IntermediateDataBufferInfos.push_back(buffer);
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
