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
    if (debuggingContext.m_DebugInfo.m_DumpDebugFiles >= CompilationOptions::DebugLevel::Medium)
    {
        MakeDirectory(debuggingContext.GetAbsolutePathOutputFileName("BestCombination").c_str());
    }

    // Use default estimation options for compilation
    EstimationOptions estimationOptions = estOpt.has_value() ? estOpt.value() : EstimationOptions();

    GraphOfParts graphOfParts = CreateGraphOfParts(network, caps, estimationOptions, compOpt, debuggingContext);
    Combiner combiner(graphOfParts, caps, compOpt, estimationOptions, debuggingContext);
    combiner.Run();
    OpGraph opGraph = combiner.GetMergedOpGraphForBestCombination();

    debuggingContext.Save(
        CompilationOptions::DebugLevel::Medium, "BestCombination/1_CombinationBasic.dot",
        [&](std::ofstream& s) { SaveCombinationToDot(combiner.GetBestCombination(), s, DetailLevel::Low); });
    debuggingContext.Save(
        CompilationOptions::DebugLevel::Medium, "BestCombination/1_CombinationDetailed.dot",
        [&](std::ofstream& s) { SaveCombinationToDot(combiner.GetBestCombination(), s, DetailLevel::High); });

    debuggingContext.Save(CompilationOptions::DebugLevel::Medium, "BestCombination/2_MergedBasic.dot",
                          [&](std::ofstream& s) { SaveOpGraphToDot(opGraph, s, DetailLevel::Low); });
    debuggingContext.Save(CompilationOptions::DebugLevel::Medium, "BestCombination/2_MergedDetailed.dot",
                          [&](std::ofstream& s) { SaveOpGraphToDot(opGraph, s, DetailLevel::High); });

    // Perform optimisation steps on the merged OpGraph.
    // These optimisations would not have affected the choice of combination as they would apply equally
    // to all combinations, and so it is much more efficient to perform them after the Combiner has finished.
    opGraph.RemoveRedundantCopies();

    debuggingContext.Save(CompilationOptions::DebugLevel::Medium, "BestCombination/3_OptimisedBasic.dot",
                          [&](std::ofstream& s) { SaveOpGraphToDot(opGraph, s, DetailLevel::Low); });
    debuggingContext.Save(CompilationOptions::DebugLevel::Medium, "BestCombination/3_OptimisedDetailed.dot",
                          [&](std::ofstream& s) { SaveOpGraphToDot(opGraph, s, DetailLevel::High); });

    EstimatedOpGraph estimatedOpGraph = ethosn::support_library::EstimateOpGraph(opGraph, caps, estimationOptions);

    debuggingContext.Save(CompilationOptions::DebugLevel::Medium, "BestCombination/4_EstimatedBasic.dot",
                          [&](std::ofstream& s) {
                              SaveEstimatedOpGraphToDot(opGraph, estimatedOpGraph, s, DetailLevel::Low, {}, {}, {});
                          });
    debuggingContext.Save(CompilationOptions::DebugLevel::Medium, "BestCombination/4_EstimatedDetailed.dot",
                          [&](std::ofstream& s) {
                              SaveEstimatedOpGraphToDot(opGraph, estimatedOpGraph, s, DetailLevel::High, {}, {}, {});
                          });

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
        CompilationOptions::DebugLevel::Medium, "BestCombination/5_CompiledBasic.dot",
        [&](std::ofstream& s) { SaveCompiledOpGraphToDot(opGraph, compiledOpGraph, s, DetailLevel::Low); });
    debuggingContext.Save(
        CompilationOptions::DebugLevel::Medium, "BestCombination/5_CompiledDetailed.dot",
        [&](std::ofstream& s) { SaveCompiledOpGraphToDot(opGraph, compiledOpGraph, s, DetailLevel::High); });

    return { std::move(opGraph), combiner.GetBestCombination(), std::move(compiledOpGraph) };
}

}    // namespace support_library
}    // namespace ethosn
