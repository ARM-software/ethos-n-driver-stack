//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../include/ethosn_support_library/Support.hpp"

#include "DmaRegisters.hpp"
#include "MceRegisters.hpp"
#include "Utils.hpp"

#include <ethosn_command_stream/CommandStream.hpp>

#include <string>

namespace ethosn
{
namespace support_library
{

struct CompiledOpGraph;
class OpGraph;
class Op;
class Buffer;
class GraphOfParts;
class Plan;
struct EstimatedOpGraph;
class Combination;
enum class Location;
enum class CompilerDataFormat;
enum class CompilerDataCompressedFormat;
enum class TraversalOrder;
enum class CompilerMceAlgorithm;
enum class BufferFormat;
enum class BufferType;

using Plans = std::vector<Plan>;

std::string ToString(Location l);
std::string ToString(BufferFormat f);
std::string ToString(DataFormat f);
std::string ToString(CompilerDataFormat f);
std::string ToString(CompilerDataCompressedFormat f);
std::string ToString(const TensorInfo& i);
std::string ToString(const TensorShape& s);
std::string ToString(TraversalOrder o);
std::string ToString(command_stream::MceOperation o);
std::string ToString(CompilerMceAlgorithm a);
std::string ToString(PleOperation o);
std::string ToString(const BlockConfig& b);
std::string ToString(command_stream::PleKernelId id);
std::string ToString(const QuantizationInfo& q);
std::string ToString(const Stride& s);
std::string ToString(const bool b);
std::string ToString(const uint16_t v);
std::string ToString(const uint32_t v);
std::string ToString(const int32_t v);
std::string ToString(const std::string& s);
std::string ToStringHex(const uint32_t v);
std::string ToString(DataType t);
std::string ToString(const utils::ShapeMultiplier& m);
std::string ToString(const utils::Fraction& f);
std::string ToString(MceUpsampleType t);
std::string ToString(PackedBoundaryThickness t);
std::string ToString(const BufferType& t);
std::string ToString(const Padding& p);
std::string ToString(const PoolingType& p);

/// Replaces any illegal characters to form a valid .dot file "ID".
std::string SanitizeId(std::string s);

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

template <typename C>
std::string MapToString(const C& container)
{
    std::stringstream ss;
    ss << "{ ";
    for (auto it = container.begin(); it != container.end(); ++it)
    {
        ss << ToString(it->first) << ": " << ToString(it->second);
        if (it != std::prev(container.end()))
        {
            ss << ", ";
        }
    }
    ss << " }";
    return ss.str();
}

struct DotAttributes
{
    DotAttributes();
    template <typename Id, typename Label, typename Color>
    DotAttributes(Id&& id, Label&& label, Color&& color)
        : m_Id(std::forward<Id>(id))
        , m_Label(std::forward<Label>(label))
        , m_LabelAlignmentChar('n')
        , m_Color(std::forward<Color>(color))
    {}

    std::string m_Id;
    std::string m_Label;
    char m_LabelAlignmentChar;
    std::string m_Shape;
    std::string m_Color;
    std::string m_FontSize;
};

enum class DetailLevel
{
    Low,
    High
};

/// Saves a Network of Operations to a dot file format to visualise the network.
/// detailLevel controls how much detail is shown on the visualisation.
void SaveNetworkToDot(const Network& network, std::ostream& stream, DetailLevel detailLevel);

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
                               DetailLevel detailLevel,
                               std::map<uint32_t, std::string> extraPassDetails,
                               std::map<Op*, std::string> extraOpDetails,
                               std::map<Buffer*, std::string> extraBufferDetails);

/// Saves a graph of Ops and Buffers to a dot file format to visualise the graph.
/// Includes details of how the performance of the OpGraph was estimated
/// and the agent IDs associated with each Op.
/// detailLevel controls how much detail is shown on the visualisation.
void SaveCompiledOpGraphToDot(const OpGraph& graph,
                              const CompiledOpGraph& compilationDetails,
                              std::ostream& stream,
                              DetailLevel detailLevel);

/// Saves a Graph of Parts to a dot file format to visualise the graph.
/// detailLevel controls how much detail is shown on the visualisation.
void SaveGraphOfPartsToDot(const GraphOfParts& graphOfParts, std::ostream& stream, DetailLevel detailLevel);

/// Saves all the plans generated for the given part to a dot file format to visualise them.
/// detailLevel controls how much detail is shown on the visualisation.
void SavePlansToDot(const Plans& plans, std::ostream& stream, DetailLevel detailLevel);

/// Saves a Combination of Plans and Glues to a dot file format to visualise it.
/// detailLevel controls how much detail is shown on the visualisation.
void SaveCombinationToDot(const Combination& combination, std::ostream& stream, DetailLevel detailLevel);

}    // namespace support_library
}    // namespace ethosn
