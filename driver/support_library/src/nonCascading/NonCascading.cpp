//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "NonCascading.hpp"

#include "DebuggingContext.hpp"
#include "Graph.hpp"
#include "Utils.hpp"
#include "ethosn_support_library/Support.hpp"

#include <fstream>
#include <sstream>

namespace ethosn
{

namespace support_library
{

namespace
{

void UpdateWithCascadingHeuristic(NetworkPerformanceData& performanceStream, const HardwareCapabilities& hwCaps)
{
    std::vector<PassPerformanceData> perfStream = performanceStream.m_Stream;
    constexpr double factor                     = 0.2f;

    uint32_t sramFootprint            = 0;
    uint32_t numCascadingNodes        = 0;
    PassPerformanceData* previousNode = nullptr;

    // There are two possible cascading strategies:
    // - Input feature map streaming, only for the first node of the section
    // - Weight streaming while all the input feature maps are stationary
    for (PassPerformanceData& node : perfStream)
    {
        PassStats& current = node.m_Stats;

        sramFootprint += static_cast<uint32_t>(
            (current.m_Input.m_MemoryStats.m_DramParallel + current.m_Input.m_MemoryStats.m_DramNonParallel) * factor);
        sramFootprint += static_cast<uint32_t>(current.m_Weights.m_MemoryStats.m_DramParallel +
                                               current.m_Weights.m_MemoryStats.m_DramNonParallel);

        // This is a sequence of cascade-able nodes.
        if (numCascadingNodes > 0 && previousNode)
        {
            PassStats& previous = previousNode->m_Stats;

            // The current node is not already cascaded with the previous node and the cascaded section fits in
            // Sram.
            if (current.m_Input.m_MemoryStats.m_Sram == 0 && sramFootprint <= hwCaps.GetTotalSramSize())
            {
                // Two or more nodes can be cascaded
                if (numCascadingNodes == 1)
                {
                    const uint32_t dramNonParallel = previous.m_Input.m_MemoryStats.m_DramNonParallel;
                    const uint32_t dramParallel    = previous.m_Input.m_MemoryStats.m_DramParallel;

                    // Update inputs statistics
                    previous.m_Input.m_MemoryStats.m_DramNonParallel =
                        static_cast<uint32_t>((dramNonParallel + dramParallel) * factor);
                    previous.m_Input.m_MemoryStats.m_DramParallel =
                        static_cast<uint32_t>((dramNonParallel + dramParallel) * (1 - factor));
                }
                else
                {
                    // Update inputs statistics
                    previous.m_Input.m_MemoryStats.m_Sram = previous.m_Input.m_MemoryStats.m_DramNonParallel +
                                                            previous.m_Input.m_MemoryStats.m_DramParallel;
                    previous.m_Input.m_MemoryStats.m_DramNonParallel = 0;
                    previous.m_Input.m_MemoryStats.m_DramParallel    = 0;
                    // Update weights statistics
                    previous.m_Weights.m_MemoryStats.m_DramParallel =
                        previous.m_Weights.m_MemoryStats.m_DramNonParallel +
                        previous.m_Weights.m_MemoryStats.m_DramParallel;
                    previous.m_Weights.m_MemoryStats.m_DramNonParallel = 0;
                }

                // Update outputs statistics
                previous.m_Output.m_MemoryStats.m_Sram =
                    previous.m_Output.m_MemoryStats.m_DramNonParallel + previous.m_Output.m_MemoryStats.m_DramParallel;
                previous.m_Output.m_MemoryStats.m_DramNonParallel = 0;
                previous.m_Output.m_MemoryStats.m_DramParallel    = 0;
                ++numCascadingNodes;
            }
            else
            {
                // The current node cannot be cascaded with the previous node, update the statistics for the
                // previous node to account for this.
                if (previous.m_Input.m_MemoryStats.m_Sram == 0)
                {
                    // Update inputs statistics
                    previous.m_Input.m_MemoryStats.m_Sram = previous.m_Input.m_MemoryStats.m_DramNonParallel +
                                                            previous.m_Input.m_MemoryStats.m_DramParallel;
                    previous.m_Input.m_MemoryStats.m_DramNonParallel = 0;
                    previous.m_Input.m_MemoryStats.m_DramParallel    = 0;

                    // Update outputs statistics
                    const uint32_t dramNonParallel = previous.m_Output.m_MemoryStats.m_DramNonParallel;
                    const uint32_t dramParallel    = previous.m_Output.m_MemoryStats.m_DramParallel;

                    previous.m_Output.m_MemoryStats.m_DramNonParallel =
                        static_cast<uint32_t>((dramParallel + dramNonParallel) * factor);
                    previous.m_Output.m_MemoryStats.m_DramParallel =
                        static_cast<uint32_t>((dramParallel + dramNonParallel) * (1 - factor));

                    // Update weights statistics
                    previous.m_Weights.m_MemoryStats.m_DramParallel =
                        previous.m_Weights.m_MemoryStats.m_DramNonParallel +
                        previous.m_Weights.m_MemoryStats.m_DramParallel;
                    previous.m_Weights.m_MemoryStats.m_DramNonParallel = 0;
                }
                // Check if it can do at least weight streaming
                else if (current.m_Input.m_MemoryStats.m_Sram != 0)
                {
                    // Update weights statistics
                    current.m_Weights.m_MemoryStats.m_DramParallel = current.m_Weights.m_MemoryStats.m_DramNonParallel +
                                                                     current.m_Weights.m_MemoryStats.m_DramParallel;
                    current.m_Weights.m_MemoryStats.m_DramNonParallel = 0;
                }

                numCascadingNodes = 0;
                sramFootprint     = 0;
            }
        }
        else
        {
            // This is the first node of a potential section.
            if (numCascadingNodes == 0 && previousNode)
            {
                PassStats& previous = previousNode->m_Stats;

                // Check if weight streaming
                if (previous.m_Input.m_MemoryStats.m_Sram != 0 && current.m_Input.m_MemoryStats.m_Sram != 0)
                {
                    // Update weights statistics
                    current.m_Weights.m_MemoryStats.m_DramParallel = current.m_Weights.m_MemoryStats.m_DramNonParallel +
                                                                     current.m_Weights.m_MemoryStats.m_DramParallel;
                    current.m_Weights.m_MemoryStats.m_DramNonParallel = 0;
                }
            }
            ++numCascadingNodes;
        }

        previousNode = &node;
    }

    // It has finished going through all the nodes, update the last node statistics if it has been cascaded.
    if (numCascadingNodes > 0)
    {
        PassStats& previous = previousNode->m_Stats;

        // Update input statistics
        previous.m_Input.m_MemoryStats.m_Sram =
            previous.m_Input.m_MemoryStats.m_DramNonParallel + previous.m_Input.m_MemoryStats.m_DramParallel;

        // Update weights statistics
        previous.m_Weights.m_MemoryStats.m_DramParallel =
            previous.m_Weights.m_MemoryStats.m_DramNonParallel + previous.m_Weights.m_MemoryStats.m_DramParallel;
        previous.m_Weights.m_MemoryStats.m_DramNonParallel = 0;

        // Update outputs statistics
        const uint32_t dramNonParallel = previous.m_Output.m_MemoryStats.m_DramNonParallel;
        const uint32_t dramParallel    = previous.m_Output.m_MemoryStats.m_DramParallel;

        previous.m_Output.m_MemoryStats.m_DramNonParallel =
            static_cast<uint32_t>((dramParallel + dramNonParallel) * factor);
        previous.m_Output.m_MemoryStats.m_DramParallel =
            static_cast<uint32_t>((dramParallel + dramNonParallel) * (1 - factor));
    }

    performanceStream.m_Stream = perfStream;
}

}    // namespace

NonCascading::NonCascading(const EstimationOptions& estOpt,
                           const CompilationOptions& compOpt,
                           const HardwareCapabilities& hwCap)
    : IEstimationStrategy(estOpt, compOpt, hwCap)
{}

NetworkPerformanceData NonCascading::Estimate(Graph& graph)
{
    std::vector<Node*> sorted = graph.GetNodesSorted();

    for (Node* n : sorted)
    {
        if (!n->IsPrepared())
        {
            std::stringstream result;
            for (auto id : n->GetCorrespondingOperationIds())
            {
                result << " " << id;
            }
            g_Logger.Error("Failed to prepare operation:%s", result.str().c_str());
        }
        n->Estimate(m_PerformanceStream, m_EstimationOptions);
    }

    bool current = m_EstimationOptions.m_Current;
    if (!current)
    {
        UpdateWithCascadingHeuristic(m_PerformanceStream, m_Capabilities);
    }

    m_DebuggingContext.DumpGraph(CompilationOptions::DebugLevel::Medium, graph, "NonCascaded_GraphFinal.dot");

    return m_PerformanceStream;
}

}    //namespace support_library

}    // namespace ethosn
