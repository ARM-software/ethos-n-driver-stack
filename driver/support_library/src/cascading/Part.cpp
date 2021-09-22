//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "Part.hpp"

#include "../Graph.hpp"
#include "../Utils.hpp"
#include "GraphNodes.hpp"
#include "Plan.hpp"
#include "WeightEncoder.hpp"
#include "WeightEncoderCache.hpp"

#include <unordered_map>

using namespace ethosn::command_stream;

namespace ethosn
{
namespace support_library
{

using namespace utils;

PartId BasePart::GetPartId() const
{
    return m_PartId;
}

std::vector<PartInputSlot> GraphOfParts::GetPartInputs(PartId p) const
{
    std::vector<PartInputSlot> res;
    for (auto&& part : m_Connections)
    {
        if (part.first.m_PartId == p)
        {
            res.push_back(part.first);
        }
    }
    std::sort(res.begin(), res.end());
    return res;
}

std::vector<PartOutputSlot> GraphOfParts::GetPartOutputs(PartId p) const
{
    std::vector<PartOutputSlot> res;
    for (auto&& part : m_Connections)
    {
        if (part.second.m_PartId == p)
        {
            res.push_back(part.second);
        }
    }
    std::sort(res.begin(), res.end());
    return res;
}

std::vector<PartOutputSlot> GraphOfParts::GetSourceParts(PartId p) const
{
    // the source part will be connected via one of it's output slots.
    // e.g.
    // P0 0---->0 P1
    // the sources of P1 will be {0, 0} which corresponds to Part0's output slot
    std::vector<PartOutputSlot> res;
    for (auto&& connection : m_Connections)
    {
        if (connection.first.m_PartId == p)
        {
            res.push_back(connection.second);
        }
    }
    std::sort(res.begin(), res.end());
    return res;
}

std::vector<PartInputSlot> GraphOfParts::GetDestinationParts(PartId p) const
{
    // the destination part will be connected via one of it's input slots.
    // e.g.
    // P0 0---->0 P1
    // the destinations of P0 will be {1, 0} which corresponds to Part1's output slot
    std::vector<PartInputSlot> res;
    for (auto&& connection : m_Connections)
    {
        if (connection.second.m_PartId == p)
        {
            res.push_back(connection.first);
        }
    }
    std::sort(res.begin(), res.end());
    return res;
}

std::vector<PartConnection> GraphOfParts::GetSourceConnections(PartId p) const
{
    std::vector<PartConnection> res;
    for (auto&& connection : m_Connections)
    {
        if (connection.first.m_PartId == p)
        {
            res.push_back(PartConnection{ connection.first, connection.second });
        }
    }
    std::sort(res.begin(), res.end());
    return res;
}

std::vector<PartConnection> GraphOfParts::GetDestinationConnections(PartId p) const
{
    std::vector<PartConnection> res;
    for (auto&& connection : m_Connections)
    {
        if (connection.second.m_PartId == p)
        {
            res.push_back(PartConnection{ connection.first, connection.second });
        }
    }
    std::sort(res.begin(), res.end());
    return res;
}

size_t GraphOfParts::GetNumParts() const
{
    return m_Parts.size();
}

const BasePart& GraphOfParts::GetPart(const PartId id) const
{
    assert(id < m_Parts.size());
    const auto& part = *m_Parts.at(id);
    assert(part.GetPartId() == id);
    return part;
}

const Parts& GraphOfParts::GetParts() const
{
    return m_Parts;
}

std::vector<PartInputSlot> GraphOfParts::GetConnectedInputSlots(const PartOutputSlot& outputSlot) const
{
    std::vector<PartInputSlot> res;
    for (auto&& connection : m_Connections)
    {
        if (connection.second == outputSlot)
        {
            res.push_back(connection.first);
        }
    }
    std::sort(res.begin(), res.end());
    return res;
}

utils::Optional<PartOutputSlot> GraphOfParts::GetConnectedOutputSlot(const PartInputSlot& inputSlot) const
{
    utils::Optional<PartOutputSlot> res;
    auto connection = m_Connections.find(inputSlot);
    if (connection != m_Connections.end())
    {
        res = connection->second;
    }
    return res;
}

std::vector<PartConnection> GraphOfParts::GetConnectionsBetween(PartId source, PartId dest) const
{
    std::vector<PartConnection> res;
    res.reserve(1);
    auto inOutSlots = GetDestinationConnections(source);
    for (auto&& inOut : inOutSlots)
    {
        if (inOut.m_Destination.m_PartId == dest)
        {
            res.push_back(inOut);
        }
    }
    return res;
}

void GraphOfParts::AddConnection(PartInputSlot inputSlot, PartOutputSlot outputSlot)
{
    assert(m_Connections.count(inputSlot) == 0);
    m_Connections[inputSlot] = outputSlot;
}

}    // namespace support_library
}    // namespace ethosn
