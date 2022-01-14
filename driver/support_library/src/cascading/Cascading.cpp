//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "Cascading.hpp"

#include "../Graph.hpp"
#include "../GraphNodes.hpp"
#include "../Utils.hpp"
#include "DebuggingContext.hpp"
#include "Estimation.hpp"
#include "EstimationUtils.hpp"
#include "NetworkToGraphOfPartsConverter.hpp"
#include "Part.hpp"

#include "../include/ethosn_support_library/Optional.hpp"
#include <ethosn_utils/Filesystem.hpp>

#include <fstream>
#include <iostream>

using namespace std;
using namespace ethosn::utils;

namespace ethosn
{
namespace support_library
{

namespace
{

template <typename T>
bool IsNodeOfType(const Node* node)
{
    return (dynamic_cast<const T*>(node) != nullptr);
}

void SaveDebugFilesForUnestimatedCombination(std::string folder,
                                             const DebuggingContext& debuggingContext,
                                             const Combination& comb,
                                             const OpGraph& opGraph,
                                             const GraphOfParts& graphOfParts)
{
    MakeDirectory(debuggingContext.GetAbsolutePathOutputFileName(folder).c_str());

    debuggingContext.SaveCombinationToDot(CompilationOptions::DebugLevel::None, comb, graphOfParts,
                                          folder + "/Simple.dot", DetailLevel::Low);
    debuggingContext.SaveCombinationToDot(CompilationOptions::DebugLevel::None, comb, graphOfParts,
                                          folder + "/Detailed.dot", DetailLevel::High);

    debuggingContext.SaveOpGraphToDot(CompilationOptions::DebugLevel::None, opGraph, folder + "/MergedSimple.dot",
                                      DetailLevel::Low);
    debuggingContext.SaveOpGraphToDot(CompilationOptions::DebugLevel::None, opGraph, folder + "/MergedDetailed.dot",
                                      DetailLevel::High);
}

void SaveDebugFilesForEstimatedCombination(std::string folder,
                                           const DebuggingContext& debuggingContext,
                                           const OpGraph& opGraph,
                                           const EstimatedOpGraph& estimationDetails)
{
    MakeDirectory(debuggingContext.GetAbsolutePathOutputFileName(folder).c_str());

    debuggingContext.SaveEstimatedOpGraphToDot(CompilationOptions::DebugLevel::None, opGraph, estimationDetails,
                                               folder + "/EstimatedSimple.dot", DetailLevel::Low);
    debuggingContext.SaveEstimatedOpGraphToDot(CompilationOptions::DebugLevel::None, opGraph, estimationDetails,
                                               folder + "/EstimatedDetailed.dot", DetailLevel::High);
}

}    // namespace

GraphOfParts CreateGraphOfParts(const Network& network,
                                const HardwareCapabilities& capabilities,
                                const EstimationOptions& estOpt,
                                const CompilationOptions& compOpt)
{
    NetworkToGraphOfPartsConverter m_NetworkToGraphOfPartsConverter(network, capabilities, estOpt, compOpt);
    return m_NetworkToGraphOfPartsConverter.ReleaseGraphOfParts();
}

Cascading::Cascading(const EstimationOptions& estOpt,
                     const CompilationOptions& compOpt,
                     const HardwareCapabilities& hwCap)
    : m_EstimationOptions(estOpt)
    , m_CompilationOptions(compOpt)
    , m_Capabilities(hwCap)
    , m_DebuggingContext(GetDebuggingContext())
    , m_BestCombination(nullptr)
    , m_Combiner(m_GraphOfParts, hwCap, estOpt, m_DebuggingContext)
{
    // Constructor
}

NetworkPerformanceData Cascading::EstimateNetwork(const Network& network)
{
    m_GraphOfParts = CreateGraphOfParts(network, m_Capabilities, m_EstimationOptions, m_CompilationOptions);

    m_DebuggingContext.SaveGraphOfPartsToDot(CompilationOptions::DebugLevel::Medium, m_GraphOfParts,
                                             "Cascaded_GraphOfParts.dot", DetailLevel::Low);
    m_DebuggingContext.SaveGraphOfPartsToDot(CompilationOptions::DebugLevel::Medium, m_GraphOfParts,
                                             "Cascaded_GraphOfPartsDetailed.dot", DetailLevel::High);

    if (m_DebuggingContext.m_DebugInfo->m_DumpDebugFiles >= CompilationOptions::DebugLevel::Medium)
    {
        MakeDirectory(m_DebuggingContext.GetAbsolutePathOutputFileName("Parts").c_str());
    }

    m_ValidCombinations = Combine(m_GraphOfParts);

    if (m_DebuggingContext.m_DebugInfo->m_DumpDebugFiles >= CompilationOptions::DebugLevel::High)
    {
        MakeDirectory(m_DebuggingContext.GetAbsolutePathOutputFileName("Combinations").c_str());
        uint32_t counter = 0;
        for (const Combination& comb : m_ValidCombinations)
        {
            std::string folder = "Combinations/" + std::to_string(counter);
            OpGraph g          = GetOpGraphForCombination(comb, m_GraphOfParts);
            SaveDebugFilesForUnestimatedCombination(folder, m_DebuggingContext, comb, g, m_GraphOfParts);
            ++counter;
        }
    }

    if (m_ValidCombinations.empty())
    {
        throw NotSupportedException("No valid combinations were found.");
    }

    EstimatePerformance();
    return m_PerformanceStream;
}

const GraphOfParts& Cascading::GetGraphOfParts() const
{
    return m_GraphOfParts;
}

const ethosn::support_library::Combination* Cascading::GetBestCombination()
{
    return m_BestCombination;
}

void Cascading::EstimatePerformance()
{
    std::ofstream debugPerformanceDumpFile;
    if (m_DebuggingContext.m_DebugInfo->m_DumpDebugFiles >= CompilationOptions::DebugLevel::Medium)
    {
        debugPerformanceDumpFile.open(m_DebuggingContext.GetAbsolutePathOutputFileName("Cascaded_Performance.txt"));
    }
    uint32_t combinationIdx = 0;
    utils::Optional<uint32_t> bestCombinationIdx;
    for (const Combination& combination : m_ValidCombinations)
    {
        OpGraph combiOpGraph = GetOpGraphForCombination(combination, m_GraphOfParts);
        EstimatedOpGraph estimatedOpGraph =
            ethosn::support_library::EstimateOpGraph(combiOpGraph, m_Capabilities, m_EstimationOptions);
        if (m_DebuggingContext.m_DebugInfo->m_DumpDebugFiles >= CompilationOptions::DebugLevel::Medium)
        {
            debugPerformanceDumpFile << combinationIdx << ": "
                                     << GetPerformanceTotalDataMetric(estimatedOpGraph.m_PerfData) << std::endl;
            if (m_DebuggingContext.m_DebugInfo->m_DumpDebugFiles >= CompilationOptions::DebugLevel::High)
            {
                std::string folder = "Combinations/" + std::to_string(combinationIdx);
                SaveDebugFilesForEstimatedCombination(folder, m_DebuggingContext, combiOpGraph, estimatedOpGraph);
            }
        }

        if (!bestCombinationIdx.has_value() ||
            ComparePerformanceData(estimatedOpGraph.m_PerfData, m_PerformanceStream) ==
                PerformanceComparisonResult::LeftBetter)
        {
            m_PerformanceStream = estimatedOpGraph.m_PerfData;
            m_BestCombination   = &combination;
            bestCombinationIdx  = combinationIdx;
        }

        ++combinationIdx;
    }

    if (m_DebuggingContext.m_DebugInfo->m_DumpDebugFiles >= CompilationOptions::DebugLevel::Medium)
    {
        debugPerformanceDumpFile << "\nBest: "
                                 << (bestCombinationIdx.has_value() ? std::to_string(bestCombinationIdx.value())
                                                                    : "NONE")
                                 << std::endl;

        // Save the details of the best combination. Note this is done at Medium debug level, so we do this even though
        // we save out details for ALL the combinations on High debug level.
        if (bestCombinationIdx.has_value())
        {
            MakeDirectory(m_DebuggingContext.GetAbsolutePathOutputFileName("Combinations").c_str());
            std::string folder   = "Combinations/Best(" + std::to_string(bestCombinationIdx.value()) + ")";
            OpGraph combiOpGraph = GetOpGraphForCombination(*m_BestCombination, m_GraphOfParts);
            EstimatedOpGraph estimatedOpGraph =
                ethosn::support_library::EstimateOpGraph(combiOpGraph, m_Capabilities, m_EstimationOptions);
            SaveDebugFilesForUnestimatedCombination(folder, m_DebuggingContext, *m_BestCombination, combiOpGraph,
                                                    m_GraphOfParts);
            SaveDebugFilesForEstimatedCombination(folder, m_DebuggingContext, combiOpGraph, estimatedOpGraph);
        }
    }
}

}    // namespace support_library
}    // namespace ethosn
