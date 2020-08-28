//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "Cascading.hpp"

#include "../Graph.hpp"
#include "../GraphNodes.hpp"
#include "../McePlePass.hpp"
#include "../Utils.hpp"
#include "DebuggingContext.hpp"
#include "Estimation.hpp"
#include "Part.hpp"

#include "../include/ethosn_support_library/Optional.hpp"

#include <fstream>
#include <iostream>

using namespace std;

namespace ethosn
{
namespace support_library
{

template <typename T>
bool IsNodeOfType(const Node* node)
{
    return (dynamic_cast<const T*>(node) != nullptr);
}

GraphOfParts CreateGraphOfParts(const Graph& graph)
{
    GraphOfParts graphOfParts;
    Parts& parts = graphOfParts.m_Parts;

    auto AddNodeToPart    = [](Node* node, Part& part) -> void { part.m_SubGraph.push_back(node); };
    auto AddNodeToNewPart = [&](Node* node) -> void {
        // Insert node into new part.
        parts.push_back(std::make_unique<Part>());
        AddNodeToPart(node, *(parts.back()));
    };
    auto FindPartFromSourceAndAddNode = [&](Node* ppOpNode) -> void {
        // Iterate in reverse, it will be quicker.
        for (auto part = parts.rbegin(); part != parts.rend(); ++part)
        {
            // Connect PP Op nodes only if the parent node has a single output.
            const auto partOutputNode = (*part)->m_SubGraph.back();
            for (const auto input : ppOpNode->GetInputs())
            {
                if (input->GetSource() == partOutputNode)
                {
                    // Case 1)
                    AddNodeToPart(ppOpNode, **part);
                    return;
                }
            }
        }
        assert(!"MCE Post-Process node has not been added to any Part");
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

    // Validate that every node has been assigned to a Part.
    std::set<Node*> nodes;
    std::transform(graph.GetNodes().begin(), graph.GetNodes().end(), std::inserter(nodes, nodes.end()),
                   [](auto&& n) { return n.get(); });
    for (auto&& p : graphOfParts.m_Parts)
    {
        for (auto&& n : p->m_SubGraph)
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

void CreatePlans(Parts& parts, const HardwareCapabilities& caps)
{
    for (auto& part : parts)
    {
        part->CreatePlans(caps);
    }

    return;
}

Cascading::Cascading(const EstimationOptions& estOpt,
                     const HardwareCapabilities& hwCap,
                     const DebuggingContext& debuggingContext)
    : IEstimationStrategy(estOpt, hwCap, debuggingContext)
{
    // Constructor
}

Cascading::~Cascading()
{}

NetworkPerformanceData Cascading::Estimate(Graph& graph)
{
    m_GraphOfParts = CreateGraphOfParts(graph);

    m_DebuggingContext.SaveGraphToDot(graph, &m_GraphOfParts, "Cascaded_GraphOfParts.dot", DetailLevel::Low);
    m_DebuggingContext.SaveGraphToDot(graph, &m_GraphOfParts, "Cascaded_GraphOfPartsDetailed.dot", DetailLevel::High);

    CreatePlans(m_GraphOfParts.m_Parts, m_Capabilities);

    if (m_DebuggingContext.m_DebugInfo.m_DumpDebugFiles)
    {
        std::ofstream debugPlanCountsDumpFile(
            m_DebuggingContext.GetAbsolutePathOutputFileName("Cascaded_PlanCounts.txt"));

        for (auto&& part : m_GraphOfParts.m_Parts)
        {
            debugPlanCountsDumpFile << part->m_DebugTag << ": " << part->GetNumPlans() << std::endl;

            m_DebuggingContext.SavePlansToDot(*part, "Cascaded_" + part->m_DebugTag + " Plans.dot", DetailLevel::Low);
            m_DebuggingContext.SavePlansToDot(*part, "Cascaded_" + part->m_DebugTag + " PlansDetailed.dot",
                                              DetailLevel::High);
        }
    }

    m_ValidCombinations = Combine(m_GraphOfParts);

    if (m_DebuggingContext.m_DebugInfo.m_DumpDebugFiles)
    {
        uint32_t counter = 0;
        for (const Combination& comb : m_ValidCombinations)
        {
            m_DebuggingContext.SaveCombinationToDot(
                comb, m_GraphOfParts, std::string("Cascaded_Combination") + std::to_string(counter) + ".dot",
                DetailLevel::Low);
            m_DebuggingContext.SaveCombinationToDot(
                comb, m_GraphOfParts, std::string("Cascaded_Combination") + std::to_string(counter) + "Detailed.dot",
                DetailLevel::High);

            OpGraph g = GetOpGraphForCombination(comb, m_GraphOfParts);
            m_DebuggingContext.SaveOpGraphToDot(
                g, std::string("Cascaded_Combination") + std::to_string(counter) + "Merged.dot", DetailLevel::Low);
            m_DebuggingContext.SaveOpGraphToDot(
                g, std::string("Cascaded_Combination") + std::to_string(counter) + "MergedDetailed.dot",
                DetailLevel::High);
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

const GraphOfParts& Cascading::getGraphOfParts() const
{
    return m_GraphOfParts;
}

void Cascading::EstimatePerformance()
{
    std::ofstream debugPerformanceDumpFile;
    if (m_DebuggingContext.m_DebugInfo.m_DumpDebugFiles)
    {
        debugPerformanceDumpFile.open(m_DebuggingContext.GetAbsolutePathOutputFileName("Cascaded_Performance.txt"));
    }
    uint32_t combinationIdx = 0;
    utils::Optional<uint32_t> bestCombinationIdx;
    for (const Combination& combination : m_ValidCombinations)
    {
        try
        {
            NetworkPerformanceData curNetPerfData = EstimateCombination(combination);

            if (m_DebuggingContext.m_DebugInfo.m_DumpDebugFiles)
            {
                debugPerformanceDumpFile << combinationIdx << ": " << utils::GetMetric(curNetPerfData) << std::endl;
            }

            if (!bestCombinationIdx.has_value() ||
                utils::IsLeftMoreDataPerformantThanRight(curNetPerfData, m_PerformanceStream))
            {
                m_PerformanceStream = curNetPerfData;
                bestCombinationIdx  = combinationIdx;
            }
        }
        catch (const NotSupportedException& e)
        {
            // Ignore this combination - others may still be valid
            if (m_DebuggingContext.m_DebugInfo.m_DumpDebugFiles)
            {
                debugPerformanceDumpFile << combinationIdx << ": Error: " << e.what() << std::endl;
            }
        }

        ++combinationIdx;
    }

    if (m_DebuggingContext.m_DebugInfo.m_DumpDebugFiles)
    {
        debugPerformanceDumpFile << "\nBest: "
                                 << (bestCombinationIdx.has_value() ? std::to_string(bestCombinationIdx.value())
                                                                    : "NONE")
                                 << std::endl;
    }
}

NetworkPerformanceData Cascading::EstimateCombination(const Combination& combination)
{
    OpGraph combiOpGraph = GetOpGraphForCombination(combination, m_GraphOfParts);
    return ethosn::support_library::EstimateOpGraph(combiOpGraph, m_Capabilities, GetEstimationOptions());
}

}    // namespace support_library
}    // namespace ethosn
