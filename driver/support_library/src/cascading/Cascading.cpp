//
// Copyright © 2018-2023 Arm Limited.
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
                                             const OpGraph& opGraph)
{
    MakeDirectory(debuggingContext.GetAbsolutePathOutputFileName(folder).c_str());

    debuggingContext.Save(CompilationOptions::DebugLevel::None, folder + "/Simple.dot",
                          [&](std::ofstream& s) { SaveCombinationToDot(comb, s, DetailLevel::Low); });
    debuggingContext.Save(CompilationOptions::DebugLevel::None, folder + "/Detailed.dot",
                          [&](std::ofstream& s) { SaveCombinationToDot(comb, s, DetailLevel::High); });

    debuggingContext.Save(CompilationOptions::DebugLevel::None, folder + "/MergedSimple.dot",
                          [&](std::ofstream& s) { SaveOpGraphToDot(opGraph, s, DetailLevel::Low); });
    debuggingContext.Save(CompilationOptions::DebugLevel::None, folder + "/MergedDetailed.dot",
                          [&](std::ofstream& s) { SaveOpGraphToDot(opGraph, s, DetailLevel::High); });
}

void SaveDebugFilesForEstimatedCombination(std::string folder,
                                           const DebuggingContext& debuggingContext,
                                           const OpGraph& opGraph,
                                           const EstimatedOpGraph& estimationDetails)
{
    MakeDirectory(debuggingContext.GetAbsolutePathOutputFileName(folder).c_str());

    debuggingContext.Save(CompilationOptions::DebugLevel::None, folder + "/EstimatedSimple.dot", [&](std::ofstream& s) {
        SaveEstimatedOpGraphToDot(opGraph, estimationDetails, s, DetailLevel::Low, {}, {}, {});
    });
    debuggingContext.Save(CompilationOptions::DebugLevel::None, folder + "/EstimatedDetailed.dot",
                          [&](std::ofstream& s) {
                              SaveEstimatedOpGraphToDot(opGraph, estimationDetails, s, DetailLevel::High, {}, {}, {});
                          });
}

}    // namespace

GraphOfParts CreateGraphOfParts(const Network& network,
                                const HardwareCapabilities& capabilities,
                                const EstimationOptions& estOpt,
                                const CompilationOptions& compOpt,
                                const DebuggingContext& debuggingContext)
{
    NetworkToGraphOfPartsConverter networkToGraphOfPartsConverter(network, capabilities, estOpt, compOpt);
    GraphOfParts g = networkToGraphOfPartsConverter.ReleaseGraphOfParts();

    debuggingContext.Save(CompilationOptions::DebugLevel::Medium, "Cascaded_GraphOfParts.dot",
                          [&](std::ofstream& s) { SaveGraphOfPartsToDot(g, s, DetailLevel::Low); });
    debuggingContext.Save(CompilationOptions::DebugLevel::Medium, "Cascaded_GraphOfPartsDetailed.dot",
                          [&](std::ofstream& s) { SaveGraphOfPartsToDot(g, s, DetailLevel::High); });

    return g;
}

Cascading::Cascading(const EstimationOptions& estOpt,
                     const CompilationOptions& compOpt,
                     const HardwareCapabilities& hwCap,
                     const DebuggingContext& debuggingContext)
    : m_EstimationOptions(estOpt)
    , m_CompilationOptions(compOpt)
    , m_Capabilities(hwCap)
    , m_DebuggingContext(debuggingContext)
    , m_Combiner(m_GraphOfParts, hwCap, compOpt, estOpt, m_DebuggingContext)
{}

NetworkPerformanceData Cascading::EstimateNetwork(const Network& network)
{
    m_GraphOfParts =
        CreateGraphOfParts(network, m_Capabilities, m_EstimationOptions, m_CompilationOptions, m_DebuggingContext);

    m_Combiner.Run();

    if (m_DebuggingContext.m_DebugInfo.m_DumpDebugFiles >= CompilationOptions::DebugLevel::High)
    {
        MakeDirectory(m_DebuggingContext.GetAbsolutePathOutputFileName("BestCombination").c_str());
        OpGraph g = GetOpGraphForCombination(m_Combiner.GetBestCombination(), m_GraphOfParts);
        SaveDebugFilesForUnestimatedCombination("BestCombination", m_DebuggingContext, m_Combiner.GetBestCombination(),
                                                g);
    }

    EstimatePerformance();
    return m_PerformanceStream;
}

const GraphOfParts& Cascading::GetGraphOfParts() const
{
    return m_GraphOfParts;
}

const ethosn::support_library::Combination& Cascading::GetBestCombination()
{
    return m_Combiner.GetBestCombination();
}

void Cascading::EstimatePerformance()
{
    OpGraph combiOpGraph = GetOpGraphForCombination(m_Combiner.GetBestCombination(), m_GraphOfParts);
    EstimatedOpGraph estimatedOpGraph =
        ethosn::support_library::EstimateOpGraph(combiOpGraph, m_Capabilities, m_EstimationOptions);
    m_PerformanceStream = estimatedOpGraph.m_PerfData;
    if (m_DebuggingContext.m_DebugInfo.m_DumpDebugFiles >= CompilationOptions::DebugLevel::Medium)
    {
        SaveDebugFilesForEstimatedCombination("BestCombination", m_DebuggingContext, combiOpGraph, estimatedOpGraph);
    }
}

}    // namespace support_library
}    // namespace ethosn
