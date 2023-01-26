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

    // Dump the GraphOfParts both before and after we optimize it.
    debuggingContext.Save(CompilationOptions::DebugLevel::Medium, "Cascaded_PreOptimizeGraphOfParts.dot",
                          [&](std::ofstream& s) { SaveGraphOfPartsToDot(g, s, DetailLevel::Low); });
    debuggingContext.Save(CompilationOptions::DebugLevel::Medium, "Cascaded_PreOptimizeGraphOfPartsDetailed.dot",
                          [&](std::ofstream& s) { SaveGraphOfPartsToDot(g, s, DetailLevel::High); });

    // Perform some optimizations on the GraphOfParts, to simplify it before generating any plans
    g.MergeChannelSelectors();

    debuggingContext.Save(CompilationOptions::DebugLevel::Medium, "Cascaded_GraphOfParts.dot",
                          [&](std::ofstream& s) { SaveGraphOfPartsToDot(g, s, DetailLevel::Low); });
    debuggingContext.Save(CompilationOptions::DebugLevel::Medium, "Cascaded_GraphOfPartsDetailed.dot",
                          [&](std::ofstream& s) { SaveGraphOfPartsToDot(g, s, DetailLevel::High); });

    return g;
}

RunCascadingResult RunCascading(const Network& network,
                                utils::Optional<const EstimationOptions&> estOpt,
                                const CompilationOptions& compOpt,
                                const HardwareCapabilities& caps,
                                const DebuggingContext& debuggingContext)
{
    // Use default estimation options for compilation
    EstimationOptions estimationOptions = estOpt.has_value() ? estOpt.value() : EstimationOptions();

    GraphOfParts graphOfParts = CreateGraphOfParts(network, caps, estimationOptions, compOpt, debuggingContext);
    Combiner combiner(graphOfParts, caps, compOpt, estimationOptions, debuggingContext);
    combiner.Run();
    OpGraph opGraph = combiner.GetMergedOpGraphForBestCombination();

    if (debuggingContext.m_DebugInfo.m_DumpDebugFiles >= CompilationOptions::DebugLevel::Medium)
    {
        MakeDirectory(debuggingContext.GetAbsolutePathOutputFileName("BestCombination").c_str());
        SaveDebugFilesForUnestimatedCombination("BestCombination", debuggingContext, combiner.GetBestCombination(),
                                                opGraph);
    }

    EstimatedOpGraph estimatedOpGraph = ethosn::support_library::EstimateOpGraph(opGraph, caps, estimationOptions);
    if (debuggingContext.m_DebugInfo.m_DumpDebugFiles >= CompilationOptions::DebugLevel::Medium)
    {
        SaveDebugFilesForEstimatedCombination("BestCombination", debuggingContext, opGraph, estimatedOpGraph);
    }

    if (estOpt.has_value())
    {
        // Not requesting compilation, so stop here.
        return { opGraph, combiner.GetBestCombination(), { estimatedOpGraph, {}, {}, {} } };
    }

    std::set<uint32_t> operationIds = network.GetOperationIds();

    cascading_compiler::CascadingCommandStreamGenerator commandStreamGenerator(opGraph, operationIds, caps, compOpt,
                                                                               debuggingContext);
    cascading_compiler::CompiledOpGraph compiledOpGraph = commandStreamGenerator.Generate();

    debuggingContext.Save(
        CompilationOptions::DebugLevel::Medium, "BestCombination/CompiledSimple.dot",
        [&](std::ofstream& s) { SaveCompiledOpGraphToDot(opGraph, compiledOpGraph, s, DetailLevel::Low); });
    debuggingContext.Save(
        CompilationOptions::DebugLevel::Medium, "BestCombination/CompiledDetailed.dot",
        [&](std::ofstream& s) { SaveCompiledOpGraphToDot(opGraph, compiledOpGraph, s, DetailLevel::High); });

    return { std::move(opGraph), combiner.GetBestCombination(), std::move(compiledOpGraph) };
}

}    // namespace support_library
}    // namespace ethosn
