//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "Cascading.hpp"

#include "../Utils.hpp"
#include "DebuggingContext.hpp"
#include "Estimation.hpp"
#include "EstimationUtils.hpp"
#include "NetworkToGraphOfPartsConverter.hpp"
#include "Part.hpp"
#include "ThreadPool.hpp"

#include "../include/ethosn_support_library/Optional.hpp"
#include <ethosn_utils/Filesystem.hpp>

#include <chrono>
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
    debuggingContext.Save(CompilationOptions::DebugLevel::Medium, "Cascaded_PreOptimizeGraphOfParts.dot",
                          [&](std::ofstream& s) { SaveGraphOfPartsToDot(g, s, DetailLevel::Low); });
    debuggingContext.Save(CompilationOptions::DebugLevel::Medium, "Cascaded_PreOptimizeGraphOfPartsDetailed.dot",
                          [&](std::ofstream& s) { SaveGraphOfPartsToDot(g, s, DetailLevel::High); });

    // Perform some optimizations on the GraphOfParts, to simplify it before generating any plans
    g.MergeChannelSelectors();

    g.SortAndCompact();

    debuggingContext.Save(CompilationOptions::DebugLevel::Medium, "Cascaded_GraphOfParts.dot",
                          [&](std::ofstream& s) { SaveGraphOfPartsToDot(g, s, DetailLevel::Low); });
    debuggingContext.Save(CompilationOptions::DebugLevel::Medium, "Cascaded_GraphOfPartsDetailed.dot",
                          [&](std::ofstream& s) { SaveGraphOfPartsToDot(g, s, DetailLevel::High); });

    return FrozenGraphOfParts(std::move(g));
}

RunCascadingResult RunCascading(const Network& network,
                                utils::Optional<const EstimationOptions&> estOpt,
                                const CompilationOptions& compOpt,
                                const HardwareCapabilities& caps,
                                DebuggingContext& debuggingContext)
{
    if (debuggingContext.m_DebugInfo.m_DumpDebugFiles >= CompilationOptions::DebugLevel::Medium)
    {
        MakeDirectory(debuggingContext.GetAbsolutePathOutputFileName("BestCombination").c_str());
    }

    // Default estimation options when none are provided (i.e. for compilation API rather than estimation API)
    EstimationOptions estimationOptions;
    if (estOpt.has_value())
    {
        estimationOptions = estOpt.value();
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

    FrozenGraphOfParts graphOfParts =
        CreateGraphOfParts(network, caps, estimationOptions, compOpt, debuggingContext, threadPool);

    auto duration = std::chrono::high_resolution_clock::now() - startTime;
    g_Logger.Debug("CreateGraphOfParts: %llu ms", duration.count() / (1000ULL * 1000ULL));

    startTime = std::chrono::high_resolution_clock::now();

    Combiner combiner(graphOfParts, caps, compOpt, estimationOptions, debuggingContext);
    combiner.Run(threadPool);
    OpGraph opGraph = combiner.GetMergedOpGraphForBestCombination();

    duration = std::chrono::high_resolution_clock::now() - startTime;
    g_Logger.Debug("Combiner: %llu ms", duration.count() / (1000ULL * 1000ULL));
    g_Logger.Debug("Weights encoded: stage 1: %u, stage 2: %u", g_NumWeightEncodingsStage1, g_NumWeightEncodingsStage2);

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

    startTime = std::chrono::high_resolution_clock::now();

    // Perform optimisation steps on the merged OpGraph.
    // These optimisations would not have affected the choice of combination as they would apply equally
    // to all combinations, and so it is much more efficient to perform them after the Combiner has finished.
    opGraph.RemoveRedundantCopies();

    duration = std::chrono::high_resolution_clock::now() - startTime;
    g_Logger.Debug("RemoveRedundantCopies: %llu ms", duration.count() / (1000ULL * 1000ULL));

    debuggingContext.Save(CompilationOptions::DebugLevel::Medium, "BestCombination/3_OptimisedBasic.dot",
                          [&](std::ofstream& s) { SaveOpGraphToDot(opGraph, s, DetailLevel::Low); });
    debuggingContext.Save(CompilationOptions::DebugLevel::Medium, "BestCombination/3_OptimisedDetailed.dot",
                          [&](std::ofstream& s) { SaveOpGraphToDot(opGraph, s, DetailLevel::High); });

    startTime = std::chrono::high_resolution_clock::now();

    EstimatedOpGraph estimatedOpGraph = ethosn::support_library::EstimateOpGraph(opGraph, caps, estimationOptions);

    duration = std::chrono::high_resolution_clock::now() - startTime;
    g_Logger.Debug("EstimateOpGraph: %llu ms", duration.count() / (1000ULL * 1000ULL));

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

    startTime = std::chrono::high_resolution_clock::now();

    cascading_compiler::CascadingCommandStreamGenerator commandStreamGenerator(opGraph, operationIds, caps, compOpt,
                                                                               debuggingContext);
    cascading_compiler::CompiledOpGraph compiledOpGraph = commandStreamGenerator.Generate();

    duration = std::chrono::high_resolution_clock::now() - startTime;
    g_Logger.Debug("CommandStreamGenerator: %llu ms", duration.count() / (1000ULL * 1000ULL));

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
