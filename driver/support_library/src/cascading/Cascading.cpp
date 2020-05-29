//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "Cascading.hpp"

#include "../Graph.hpp"
#include "../GraphNodes.hpp"
#include "../McePlePass.hpp"
#include "DebuggingContext.hpp"
#include "Part.hpp"

#include <iostream>

using namespace std;

namespace ethosn
{
namespace support_library
{

namespace
{

uint64_t GetPerformanceDataMetric(const PassStats& passStat)
{
    return passStat.m_Input.m_MemoryStats.m_DramParallel + passStat.m_Input.m_MemoryStats.m_DramNonParallel +
           passStat.m_Output.m_MemoryStats.m_DramParallel + passStat.m_Output.m_MemoryStats.m_DramNonParallel +
           passStat.m_Weights.m_MemoryStats.m_DramParallel + passStat.m_Output.m_MemoryStats.m_DramNonParallel;
}

uint64_t GetMetric(const NetworkPerformanceData& netPerfData)
{
    uint64_t performanceMetric = 0;
    for (PassPerformanceData passPerfData : netPerfData.m_Stream)
    {
        performanceMetric += GetPerformanceDataMetric(passPerfData.m_Stats);
    }
    return performanceMetric;
}

bool IsLeftMoreDataPerformantThanRight(const NetworkPerformanceData& left, const NetworkPerformanceData& right)
{
    return GetMetric(left) < GetMetric(right);
}

}    //namespace

template <typename T>
bool IsNodeOfType(const Node* node)
{
    return (dynamic_cast<const T*>(node) != nullptr);
}

GraphOfParts CreateGraphOfParts(const Graph& graph)
{
    GraphOfParts graphOfParts;
    auto& parts = graphOfParts.m_Parts;

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

    m_DebuggingContext.SaveGraphToDot(graph, &m_GraphOfParts, "GraphOfParts.dot", DetailLevel::Low);
    m_DebuggingContext.SaveGraphToDot(graph, &m_GraphOfParts, "GraphOfPartsDetailed.dot", DetailLevel::High);

    CreatePlans(m_GraphOfParts.m_Parts, m_Capabilities);

    for (auto&& part : m_GraphOfParts.m_Parts)
    {
        m_DebuggingContext.SavePlansToDot(*part, part->m_DebugTag + " Plans.dot", DetailLevel::Low);
        m_DebuggingContext.SavePlansToDot(*part, part->m_DebugTag + " PlansDetailed.dot", DetailLevel::High);
    }

    m_ValidCombinations = Combine(m_GraphOfParts);

    if (m_DebuggingContext.m_DebugInfo.m_DumpDebugFiles)
    {
        uint32_t counter = 0;
        for (const Combination& comb : m_ValidCombinations)
        {
            m_DebuggingContext.SaveCombinationToDot(
                comb, m_GraphOfParts, std::string("Combination") + std::to_string(counter) + ".dot", DetailLevel::Low);
            m_DebuggingContext.SaveCombinationToDot(
                comb, m_GraphOfParts, std::string("Combination") + std::to_string(counter) + "Detailed.dot",
                DetailLevel::High);

            OpGraph g = GetOpGraphForCombination(comb, m_GraphOfParts);
            m_DebuggingContext.SaveOpGraphToDot(g, std::string("Combination") + std::to_string(counter) + "Merged.dot",
                                                DetailLevel::Low);
            m_DebuggingContext.SaveOpGraphToDot(
                g, std::string("Combination") + std::to_string(counter) + "MergedDetailed.dot", DetailLevel::High);
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
    bool isFirst = true;
    for (const Combination& combination : m_ValidCombinations)
    {
        NetworkPerformanceData curNetPerfData = EstimateCombination(combination);
        if (isFirst || IsLeftMoreDataPerformantThanRight(curNetPerfData, m_PerformanceStream))
        {
            isFirst             = false;
            m_PerformanceStream = curNetPerfData;
        }
    }
}

NetworkPerformanceData EstimateCombination(const Combination& combination,
                                           const GraphOfParts& parts,
                                           const HardwareCapabilities& capabilities)
{
    NetworkPerformanceData netPerfData;
    OpGraph combiOpGraph  = GetOpGraphForCombination(combination, parts);
    OpGraph::OpList opLst = combiOpGraph.GetOps();
    assert(opLst.size() >= 1);

    netPerfData.m_Stream.emplace_back(std::move(PassPerformanceData()));
    PassStats& planStats = netPerfData.m_Stream.front().m_Stats;

    for (auto opIt = opLst.begin(); opIt != opLst.end(); ++opIt)
    {
        Op* op     = *opIt;
        Op* prevOp = (opIt == opLst.begin()) ? nullptr : *(opIt - 1);
        Op* nextOp = (opIt == opLst.end()) ? nullptr : *(opIt + 1);

        OpGraph::BufferList bufList = combiOpGraph.GetInputs(op);
        assert(bufList.size() >= 1);
        Buffer* inpBuf = bufList[0];
        Buffer* outBuf = combiOpGraph.GetOutput(op);
        assert(outBuf);

        // TODO: Fix this in NNXSW-2194
        uint32_t inputTilesize = 1;

        if (IsObjectOfType<DmaOp>(op))
        {
            if (Location::Dram == inpBuf->m_Location && Location::Sram == outBuf->m_Location)
            {
                // Calculate Input stats
                // TODO: Get Weights Info buffer if required. No Activation Compression. Fix in NNXSW-2194
                planStats.m_Input += support_library::GetInputStats(capabilities, inpBuf, outBuf, inputTilesize);
            }
            if (Location::Sram == inpBuf->m_Location && Location::Dram == outBuf->m_Location)
            {
                // Calculate Output stats
                // TODO: RoundedUpOutputShape, Location. NNXSW-2194
                planStats.m_Output +=
                    support_library::GetOutputStats(inpBuf->m_TensorShape, inpBuf->m_StripeShape, BufferLocation::Dram);
                // If prev op is MCe or Ple stop the pass here for generation of stats.
                //TODO: operatorID and ParentID to be filled in by NNXSW-2190
                if ((nextOp != nullptr) && (IsObjectOfType<MceOp>(prevOp) || IsObjectOfType<PleOp>(prevOp)))
                {
                    netPerfData.m_Stream.emplace_back(std::move(PassPerformanceData()));
                    planStats = netPerfData.m_Stream.front().m_Stats;
                }
            }
        }
        else if (IsObjectOfType<MceOp>(op))
        {
            MceOp* mceOp    = dynamic_cast<MceOp*>(op);
            planStats.m_Mce = support_library::GetMceStats(capabilities, mceOp->m_Stride, mceOp->m_Op, mceOp->m_Algo,
                                                           mceOp->m_InputStripeShape, mceOp->m_OutputStripeShape,
                                                           mceOp->m_WeightsStripeShape);
        }
        else if (IsObjectOfType<PleOp>(op))
        {
            PleOp* pleOp    = dynamic_cast<PleOp*>(op);
            planStats.m_Ple = support_library::GetPleStats(capabilities, pleOp->m_InputStripeShapes, pleOp->m_Op);
        }
    }
    return netPerfData;
}

NetworkPerformanceData Cascading::EstimateCombination(const Combination& combination)
{
    return ethosn::support_library::EstimateCombination(combination, m_GraphOfParts, m_Capabilities);
}

}    // namespace support_library
}    // namespace ethosn
