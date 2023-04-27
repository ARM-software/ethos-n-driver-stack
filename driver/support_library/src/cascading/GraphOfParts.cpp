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
            if (std::find(res.begin(), res.end(), part.second) == res.end())
            {
                res.push_back(part.second);
            }
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

Parts GraphOfParts::ReleaseParts()
{
    return std::move(m_Parts);
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

void GraphOfParts::SortAndCompact()
{
    // Find a topological sort of the part IDs
    std::vector<PartId> targets;
    for (auto&& part : m_Parts)
    {
        if (GetPartOutputs(part.first).size() == 0)
        {
            targets.push_back(part.first);
        }
    }

    auto GetIncomingEdges = [&](PartId p) -> std::vector<PartId> {
        std::vector<PartId> result;
        for (const PartConnection& c : GetSourceConnections(p))
        {
            result.push_back(c.m_Source.m_PartId);
        }
        return result;
    };

    std::vector<PartId> sorted;
    bool success = utils::GraphTopologicalSort<PartId, std::vector<PartId>>(targets, GetIncomingEdges, sorted);
    if (!success)
    {
        throw InternalErrorException("Topological sort failed");
    }

    // Use the sorted list to re-number the parts, updating the Part IDs stored in the Parts themselves
    // as well as all the connections between them.
    std::map<PartId, PartId> oldToNew;
    for (PartId newId = 0; newId < sorted.size(); ++newId)
    {
        oldToNew[sorted[newId]] = newId;
    }

    Parts oldParts = std::move(m_Parts);
    m_Parts.clear();
    for (std::pair<const PartId, std::unique_ptr<BasePart>>& p : oldParts)
    {
        PartId oldPartId = p.first;
        PartId newPartId = oldToNew[oldPartId];
        p.second->ChangePartId(newPartId);
        m_Parts[newPartId] = std::move(p.second);
    }

    std::unordered_map<PartInputSlot, PartOutputSlot> oldConnections = std::move(m_Connections);
    m_Connections.clear();
    for (const std::pair<const PartInputSlot, PartOutputSlot>& c : oldConnections)
    {
        PartId oldDestPartId = c.first.m_PartId;
        PartId newDestPartId = oldToNew[oldDestPartId];

        PartId oldSrcPartId = c.second.m_PartId;
        PartId newSrcPartId = oldToNew[oldSrcPartId];

        m_Connections[PartInputSlot{ newDestPartId, c.first.m_InputIndex }] =
            PartOutputSlot{ newSrcPartId, c.second.m_OutputIndex };
    }

    // Fill the boundary requirements for all parts. This is only possible once all connections
    // have been made so that we know which part(s) consume the output of each part.
    for (std::pair<const PartId, std::unique_ptr<BasePart>>& p : m_Parts)
    {
        BasePart* part = p.second.get();

        const std::vector<PartOutputSlot>& outputSlots = GetPartOutputs(part->GetPartId());

        std::vector<BoundaryRequirements> req(outputSlots.size());

        for (PartOutputSlot outputSlot : outputSlots)
        {
            // We should produce boundary data for this output slot, if any of the consuming parts require it.
            BoundaryRequirements boundaryRequirement;
            for (PartInputSlot connectedInputSlot : GetConnectedInputSlots(outputSlot))
            {
                const std::vector<BoundaryRequirements>& inputReqs =
                    GetPart(connectedInputSlot.m_PartId).GetInputBoundaryRequirements();
                BoundaryRequirements inputReq = inputReqs.at(connectedInputSlot.m_InputIndex);
                boundaryRequirement.m_NeedsBeforeX |= inputReq.m_NeedsBeforeX;
                boundaryRequirement.m_NeedsAfterX |= inputReq.m_NeedsAfterX;
                boundaryRequirement.m_NeedsBeforeY |= inputReq.m_NeedsBeforeY;
                boundaryRequirement.m_NeedsAfterY |= inputReq.m_NeedsAfterY;
            }
            req[outputSlot.m_OutputIndex] = boundaryRequirement;
        }

        part->SetOutputBoundaryRequirements(std::move(req));
    }
}

FrozenGraphOfParts::FrozenGraphOfParts(GraphOfParts graph)
{
    // Take ownership of all the Parts from the GraphOfParts
    for (auto&& p : graph.ReleaseParts())
    {
        m_Parts.push_back(std::move(p.second));
    }

    // Copy all the connection information in our arrays for fast lookups
    for (const auto& c : graph.GetAllConnections())
    {
        m_Connections.insert(c);
    }

    m_PartInputs.resize(m_Parts.size());
    for (PartId p = 0; p < m_Parts.size(); ++p)
    {
        m_PartInputs[p] = graph.GetPartInputs(p);
    }

    m_PartOutputs.resize(m_Parts.size());
    for (PartId p = 0; p < m_Parts.size(); ++p)
    {
        m_PartOutputs[p] = graph.GetPartOutputs(p);
    }

    m_SourceParts.resize(m_Parts.size());
    for (PartId p = 0; p < m_Parts.size(); ++p)
    {
        m_SourceParts[p] = graph.GetSourceParts(p);
    }

    m_DestinationParts.resize(m_Parts.size());
    for (PartId p = 0; p < m_Parts.size(); ++p)
    {
        m_DestinationParts[p] = graph.GetDestinationParts(p);
    }

    m_SourceConnections.resize(m_Parts.size());
    for (PartId p = 0; p < m_Parts.size(); ++p)
    {
        m_SourceConnections[p] = graph.GetSourceConnections(p);
    }

    m_DestinationConnections.resize(m_Parts.size());
    for (PartId p = 0; p < m_Parts.size(); ++p)
    {
        m_DestinationConnections[p] = graph.GetDestinationConnections(p);
    }

    m_ConnectedInputSlots.resize(m_Parts.size());
    for (PartId p = 0; p < m_Parts.size(); ++p)
    {
        const std::vector<PartOutputSlot> outputSlots = graph.GetPartOutputs(p);
        m_ConnectedInputSlots[p].resize(outputSlots.size());
        for (PartOutputSlot slot : graph.GetPartOutputs(p))
        {
            m_ConnectedInputSlots[p][slot.m_OutputIndex] = graph.GetConnectedInputSlots(slot);
        }
    }

    m_ConnectedOutputSlot.resize(m_Parts.size());
    for (PartId p = 0; p < m_Parts.size(); ++p)
    {
        const std::vector<PartInputSlot> inputSlots = graph.GetPartInputs(p);
        m_ConnectedOutputSlot[p].resize(inputSlots.size());
        for (PartInputSlot slot : graph.GetPartInputs(p))
        {
            m_ConnectedOutputSlot[p][slot.m_InputIndex] = graph.GetConnectedOutputSlot(slot);
        }
    }
}

size_t FrozenGraphOfParts::GetNumParts() const
{
    return m_Parts.size();
}

const BasePart& FrozenGraphOfParts::GetPart(const PartId id) const
{
    return *m_Parts[id];
}

const std::vector<std::unique_ptr<BasePart>>& FrozenGraphOfParts::GetParts() const
{
    return m_Parts;
}

const std::unordered_map<PartInputSlot, PartOutputSlot>& FrozenGraphOfParts::GetAllConnections() const
{
    return m_Connections;
}

const std::vector<PartInputSlot>& FrozenGraphOfParts::GetPartInputs(PartId p) const
{
    return m_PartInputs[p];
}

const std::vector<PartOutputSlot>& FrozenGraphOfParts::GetPartOutputs(PartId p) const
{
    return m_PartOutputs[p];
}

const std::vector<PartOutputSlot>& FrozenGraphOfParts::GetSourceParts(PartId p) const
{
    return m_SourceParts[p];
}

const std::vector<PartInputSlot>& FrozenGraphOfParts::GetDestinationParts(PartId p) const
{
    return m_DestinationParts[p];
}

const std::vector<PartConnection>& FrozenGraphOfParts::GetSourceConnections(PartId p) const
{
    return m_SourceConnections[p];
}

const std::vector<PartConnection>& FrozenGraphOfParts::GetDestinationConnections(PartId p) const
{
    return m_DestinationConnections[p];
}

const std::vector<PartInputSlot>& FrozenGraphOfParts::GetConnectedInputSlots(const PartOutputSlot& outputSlot) const
{
    return m_ConnectedInputSlots[outputSlot.m_PartId][outputSlot.m_OutputIndex];
}

const utils::Optional<PartOutputSlot>& FrozenGraphOfParts::GetConnectedOutputSlot(const PartInputSlot& inputSlot) const
{
    return m_ConnectedOutputSlot[inputSlot.m_PartId][inputSlot.m_InputIndex];
}

}    // namespace support_library
}    // namespace ethosn
