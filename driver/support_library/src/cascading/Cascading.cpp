//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "Cascading.hpp"

#include "../Graph.hpp"
#include "../GraphNodes.hpp"
#include "../Utils.hpp"
#include "DebuggingContext.hpp"
#include "Estimation.hpp"
#include "EstimationUtils.hpp"
#include "Part.hpp"
#include "PartV1.hpp"

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

GraphOfParts CreateGraphOfParts(const Graph& graph,
                                const EstimationOptions& estOpt,
                                const CompilationOptions& compOpt,
                                const HardwareCapabilities& capabilities)
{
    GraphOfParts graphOfParts;
    Parts& parts = graphOfParts.m_Parts;

    auto AddNodeToPart    = [](Node* node, PartV1& part) -> void { part.m_SubGraph.push_back(node); };
    auto AddNodeToNewPart = [&](Node* node) -> void {
        // Insert node into new part.
        auto part = std::make_unique<PartV1>(graphOfParts.GeneratePartId(), estOpt, compOpt, capabilities);
        AddNodeToPart(node, *part);
        parts.push_back(std::unique_ptr<BasePart>(std::move(part)));
    };
    auto FindPartFromSourceAndAddNode = [&](Node* ppOpNode) -> void {
        // Iterate in reverse, it will be quicker.
        for (auto part = parts.rbegin(); part != parts.rend(); ++part)
        {
            // Connect PP Op nodes only if the parent node has a single output.
            const auto partOutputNode = static_cast<PartV1*>(part->get())->m_SubGraph.back();
            for (const auto input : ppOpNode->GetInputs())
            {
                if (input->GetSource() == partOutputNode)
                {
                    // Case 1)
                    AddNodeToPart(ppOpNode, static_cast<PartV1&>(**part));
                    return;
                }
            }
        }
        ETHOSN_FAIL_MSG("MCE Post-Process node has not been added to any Part");
    };

    for (Node* node : graph.GetNodesSorted())
    {
        assert(node);
        if (IsNodeOfType<McePostProcessOperationNode>(node))
        {
            // There are two possible cases with PP Op nodes:
            // 1) The node is connected to an MCE operation node with a single output.
            // 2) The node is connected to a non PP Op node with multiple outputs.
            // If 1), then find the part with the source node and add this node.
            // If 2), then create a new part with that single PP Op node.

            auto source = node->GetInputs()[0]->GetSource();
            if (IsNodeOfType<MceOperationNode>(source) && source->GetOutputs().size() == 1)
            {
                // Case 1)
                FindPartFromSourceAndAddNode(node);
            }
            else
            {
                // Case 2)
                AddNodeToNewPart(node);
            }
        }
        else
        {
            AddNodeToNewPart(node);
        }
    }

    auto GetPartIdFromNode = [&](const Node* node) {
        for (auto&& part : parts)
        {
            const auto partV1 = static_cast<PartV1*>(part.get());
            for (Node* n : partV1->m_SubGraph)
            {
                if (node == n)
                {
                    return partV1->GetPartId();
                }
            }
        }
        assert(false);
        return static_cast<PartId>(-1);
    };

    for (auto&& part : parts)
    {
        const auto partV1         = static_cast<PartV1*>(part.get());
        auto edges                = partV1->GetOutputs();
        PartOutputSlot outputSlot = { partV1->GetPartId(), 0 };
        for (auto&& edge : edges)
        {
            const Node* dest    = edge->GetDestination();
            auto destInputs     = dest->GetInputs();
            uint32_t inputIndex = 0;
            for (uint32_t i = 0; i < destInputs.size(); ++i)
            {
                if (edge == destInputs[i])
                {
                    inputIndex = i;
                    break;
                }
            }

            assert(GetPartIdFromNode(edge->GetSource()) == partV1->GetPartId());
            auto destPart           = GetPartIdFromNode(dest);
            PartInputSlot inputSlot = { destPart, inputIndex };
            graphOfParts.AddConnection(inputSlot, outputSlot);
        }
    }

    // Validate that every node has been assigned to a Part.
    std::set<Node*> nodes;
    std::transform(graph.GetNodes().begin(), graph.GetNodes().end(), std::inserter(nodes, nodes.end()),
                   [](auto&& n) { return n.get(); });
    for (auto&& p : graphOfParts.m_Parts)
    {
        for (auto&& n : static_cast<PartV1*>(p.get())->m_SubGraph)
        {
            nodes.erase(n);
        }
    }
    if (!nodes.empty())
    {
        throw NotSupportedException("Some nodes could not be assigned to a Part");
    }

    return graphOfParts;
}

Cascading::Cascading(const EstimationOptions& estOpt,
                     const CompilationOptions& compOpt,
                     const HardwareCapabilities& hwCap)
    : IEstimationStrategy(estOpt, compOpt, hwCap)
    , m_BestCombination(nullptr)
    , m_Combiner(m_GraphOfParts, hwCap, estOpt, m_DebuggingContext)
{
    // Constructor
}

Cascading::~Cascading()
{}

NetworkPerformanceData Cascading::Estimate(Graph& graph)
{
    m_GraphOfParts = CreateGraphOfParts(graph, m_EstimationOptions, m_CompilationOptions, m_Capabilities);

    m_DebuggingContext.SaveGraphToDot(CompilationOptions::DebugLevel::Medium, m_GraphOfParts,
                                      "Cascaded_GraphOfParts.dot", DetailLevel::Low);
    m_DebuggingContext.SaveGraphToDot(CompilationOptions::DebugLevel::Medium, m_GraphOfParts,
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
            ethosn::support_library::EstimateOpGraph(combiOpGraph, m_Capabilities, GetEstimationOptions());
        if (m_DebuggingContext.m_DebugInfo->m_DumpDebugFiles >= CompilationOptions::DebugLevel::Medium)
        {
            debugPerformanceDumpFile << combinationIdx << ": "
                                     << GetPerformanceTotalDataMetric(estimatedOpGraph.m_PerfData)
                                     << (estimatedOpGraph.IsComplete() ? "" : " (INCOMPLETE)") << std::endl;
            if (m_DebuggingContext.m_DebugInfo->m_DumpDebugFiles >= CompilationOptions::DebugLevel::High)
            {
                std::string folder = "Combinations/" + std::to_string(combinationIdx);
                SaveDebugFilesForEstimatedCombination(folder, m_DebuggingContext, combiOpGraph, estimatedOpGraph);
            }
        }

        // The estimation may not have been complete, in which case we can't consider this combination.
        if (estimatedOpGraph.IsComplete())
        {
            if (!bestCombinationIdx.has_value() ||
                ComparePerformanceData(estimatedOpGraph.m_PerfData, m_PerformanceStream) ==
                    PerformanceComparisonResult::LeftBetter)
            {
                m_PerformanceStream = estimatedOpGraph.m_PerfData;
                m_BestCombination   = &combination;
                bestCombinationIdx  = combinationIdx;
            }
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
                ethosn::support_library::EstimateOpGraph(combiOpGraph, m_Capabilities, GetEstimationOptions());
            SaveDebugFilesForUnestimatedCombination(folder, m_DebuggingContext, *m_BestCombination, combiOpGraph,
                                                    m_GraphOfParts);
            SaveDebugFilesForEstimatedCombination(folder, m_DebuggingContext, combiOpGraph, estimatedOpGraph);
        }
    }
}

}    // namespace support_library
}    // namespace ethosn
