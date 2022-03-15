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

    m_Combiner.Run();

    if (m_DebuggingContext.m_DebugInfo->m_DumpDebugFiles >= CompilationOptions::DebugLevel::High)
    {
        MakeDirectory(m_DebuggingContext.GetAbsolutePathOutputFileName("BestCombination").c_str());
        OpGraph g = GetOpGraphForCombination(m_Combiner.GetBestCombination(), m_GraphOfParts);
        SaveDebugFilesForUnestimatedCombination("BestCombination", m_DebuggingContext, m_Combiner.GetBestCombination(),
                                                g, m_GraphOfParts);
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
    if (m_DebuggingContext.m_DebugInfo->m_DumpDebugFiles >= CompilationOptions::DebugLevel::Medium)
    {
        SaveDebugFilesForEstimatedCombination("BestCombination", m_DebuggingContext, combiOpGraph, estimatedOpGraph);
    }
}

}    // namespace support_library
}    // namespace ethosn
