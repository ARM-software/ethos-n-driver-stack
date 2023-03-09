//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "GraphOfParts.hpp"

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
    const BasePart& part = *m_Parts.at(id);
    assert(part.GetPartId() == id);
    return part;
}

const Parts& GraphOfParts::GetParts() const
{
    return m_Parts;
}

void GraphOfParts::AddPart(std::unique_ptr<BasePart> p)
{
    PartId id                               = p->GetPartId();
    std::pair<Parts::iterator, bool> result = m_Parts.insert(std::make_pair(id, std::move(p)));
    assert(result.second);
    ETHOSN_UNUSED(result);
}

const std::unordered_map<PartInputSlot, PartOutputSlot>& GraphOfParts::GetAllConnections() const
{
    return m_Connections;
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

void GraphOfParts::AddConnection(PartInputSlot inputSlot, PartOutputSlot outputSlot)
{
    assert(m_Connections.count(inputSlot) == 0);
    m_Connections[inputSlot] = outputSlot;
}

void GraphOfParts::RemoveConnection(PartInputSlot inputSlot)
{
    auto it = m_Connections.find(inputSlot);
    if (it != m_Connections.end())
    {
        m_Connections.erase(it);
    }
}

void GraphOfParts::MergeChannelSelectors()
{
    auto partIt = m_Parts.begin();
    while (partIt != m_Parts.end())
    {
        const std::unique_ptr<BasePart>& channelSelectorPart = partIt->second;
        utils::Optional<ConstTensorData> o                   = channelSelectorPart->GetChannelSelectorWeights();
        if (!o.has_value())
        {
            // Not a channel selector, check the next instead
            ++partIt;
            continue;
        }
        ConstTensorData channelSelectorWeights = o.value();

        // Check if can be merged with the part afterwards
        {
            std::vector<PartConnection> connections = GetDestinationConnections(channelSelectorPart->GetPartId());
            // The channel selector part's output must only be consumed by the part we are going to merge it with,
            if (connections.size() == 1)
            {
                PartId destPartId = connections[0].m_Destination.m_PartId;
                if (m_Parts[destPartId]->MergeWithChannelSelectorBefore(channelSelectorWeights))
                {
                    // Merge successful

                    // Merge operation IDs
                    for (uint32_t operationId : channelSelectorPart->GetOperationIds())
                    {
                        m_Parts[destPartId]->AddOperationId(operationId);
                    }

                    // Remove the channel selector part and reroute its input connections to the modified layer.
                    std::vector<PartConnection> channelSelectorInputConnections =
                        GetSourceConnections(channelSelectorPart->GetPartId());
                    // Channel selectors are single-input single-output
                    assert(channelSelectorInputConnections.size() == 1);
                    RemoveConnection(channelSelectorInputConnections[0].m_Destination);
                    RemoveConnection(PartInputSlot{ destPartId, 0 });

                    AddConnection(PartInputSlot{ destPartId, 0 }, channelSelectorInputConnections[0].m_Source);

                    partIt = m_Parts.erase(partIt);
                    continue;
                }
            }
        }

        // Check if can be merged with the part beforehand
        {
            std::vector<PartConnection> connections = GetSourceConnections(channelSelectorPart->GetPartId());
            assert(connections.size() == 1);    // Channel selectors are single-input single-output
            PartId srcPartId = connections[0].m_Source.m_PartId;
            // The part we are going to merge it with can't have a shared output - it must only be connected with the channel selector part
            if (GetDestinationConnections(srcPartId).size() == 1)
            {
                if (m_Parts[srcPartId]->MergeWithChannelSelectorAfter(channelSelectorWeights))
                {
                    // Merge successful

                    // Merge operation IDs
                    for (uint32_t operationId : channelSelectorPart->GetOperationIds())
                    {
                        m_Parts[srcPartId]->AddOperationId(operationId);
                    }

                    // Remove the channel selector part and reroute its output connections to the modified layer.
                    std::vector<PartConnection> channelSelectorOutputConnections =
                        GetDestinationConnections(channelSelectorPart->GetPartId());
                    // Channel selectors are single-input single-output
                    assert(channelSelectorOutputConnections.size() == 1);
                    RemoveConnection(channelSelectorOutputConnections[0].m_Destination);
                    RemoveConnection(PartInputSlot{ channelSelectorPart->GetPartId(), 0 });

                    AddConnection(channelSelectorOutputConnections[0].m_Destination, PartOutputSlot{ srcPartId, 0 });

                    partIt = m_Parts.erase(partIt);
                    continue;
                }
            }
        }

        ++partIt;
    }
}

}    // namespace support_library
}    // namespace ethosn
