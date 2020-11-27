//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../../include/ethosn_support_library/Support.hpp"

#include <ethosn_command_stream/CommandData.hpp>

#include <string>

namespace ethosn
{
namespace support_library
{

class Graph;
class OpGraph;
class GraphOfParts;
class Part;
struct EstimatedOpGraph;
struct Combination;
enum class Location;
enum class Lifetime;
enum class CompilerDataFormat;
enum class CompilerDataCompressedFormat;
enum class TraversalOrder;
enum class CompilerMceAlgorithm;

std::string ToString(Location l);
std::string ToString(Lifetime l);
std::string ToString(CompilerDataFormat f);
std::string ToString(CompilerDataCompressedFormat f);
std::string ToString(const TensorShape& s);
std::string ToString(TraversalOrder o);
std::string ToString(command_stream::MceOperation o);
std::string ToString(CompilerMceAlgorithm a);
std::string ToString(command_stream::PleOperation o);
std::string ToString(command_stream::BlockConfig b);
std::string ToString(const QuantizationInfo& q);
std::string ToString(const Stride& s);
std::string ToString(command_stream::DataFormat f);
std::string ToString(const uint32_t v);
std::string ToString(DataType t);

template <typename C>
std::string ArrayToString(const C& container)
{
    std::stringstream ss;
    ss << "[";
    for (auto it = container.begin(); it != container.end(); ++it)
    {
        ss << ToString(*it);
        if (it != std::prev(container.end()))
        {
            ss << ", ";
        }
    }
    ss << "]";
    return ss.str();
}

struct DotAttributes
{
    DotAttributes();
    DotAttributes(std::string id, std::string label, std::string color);

    std::string m_Id;
    std::string m_Label;
    char m_LabelAlignmentChar;
    std::string m_Shape;
    std::string m_Color;
};

enum class DetailLevel
{
    Low,
    High
};

/// Save OpGraph information to a text file
void SaveOpGraphToTxtFile(const OpGraph& graph, std::ostream& stream);

/// Saves a graph of Ops and Buffers to a dot file format to visualise the graph.
/// detailLevel controls how much detail is shown on the visualisation.
void SaveOpGraphToDot(const OpGraph& graph, std::ostream& stream, DetailLevel detailLevel);

/// Saves a graph of Ops and Buffers to a dot file format to visualise the graph.
/// Includes details of how the performance of the OpGraph was estimated.
/// detailLevel controls how much detail is shown on the visualisation.
void SaveEstimatedOpGraphToDot(const OpGraph& graph,
                               const EstimatedOpGraph& estimationDetails,
                               std::ostream& stream,
                               DetailLevel detailLevel);

/// Saves a Graph of Nodes to a dot file format to visualise the graph.
/// Optionally includes groupings of Nodes into Parts, if provided a GraphOfParts object.
/// detailLevel controls how much detail is shown on the visualisation.
void SaveGraphToDot(const Graph& graph,
                    const GraphOfParts* graphOfParts,
                    std::ostream& stream,
                    DetailLevel detailLevel);

/// Saves all the plans generated for the given part to a dot file format to visualise them.
/// detailLevel controls how much detail is shown on the visualisation.
void SavePlansToDot(const Part& part, std::ostream& stream, DetailLevel detailLevel);

/// Saves a Combination of Plans and Glues to a dot file format to visualise it.
/// detailLevel controls how much detail is shown on the visualisation.
void SaveCombinationToDot(const Combination& combination,
                          const GraphOfParts& graphOfParts,
                          std::ostream& stream,
                          DetailLevel detailLevel);

}    // namespace support_library
}    // namespace ethosn
