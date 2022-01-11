//
// Copyright © 2018-2022 Arm Limited.
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

utils::Optional<ethosn::command_stream::MceOperation> BasePart::GetMceOperation() const
{
    utils::Optional<ethosn::command_stream::MceOperation> mceOperationWithNoValue;
    return mceOperationWithNoValue;
}

bool IsPlanValid(const HardwareCapabilities& caps, const Plan& plan)
{
    const uint32_t sizeInBytes = GetTotSizeInBytes(plan).m_Tot;

    if (sizeInBytes > caps.GetTotalSramSize())
    {
        // There is no space
        return false;
    }

    return true;
}

void BasePart::AddNewPlan(PartInputMapping&& inputMappings,
                          PartOutputMapping&& outputMappings,
                          OwnedOpGraph&& opGraph,
                          Plans& plans) const
{
    Plan plan(std::move(inputMappings), std::move(outputMappings));
    plan.m_OpGraph = std::move(opGraph);

    if (IsPlanValid(m_Capabilities, plan))
    {
        plans.push_back(std::move(plan));
    }
}

ethosn::support_library::DotAttributes BasePart::GetDotAttributes(DetailLevel detail) const
{
    DotAttributes result = DebuggableObject::GetDotAttributes(detail);
    if (detail >= DetailLevel::High)
    {
        result.m_Label += "\n";
        result.m_Label += "PartId = " + ToString(m_PartId) + "\n";
        result.m_Label += "CompilerDataFormat = " + ToString(m_CompilerDataFormat) + "\n";
        result.m_Label += "CorrespondingOperationIds = " + ArrayToString(m_CorrespondingOperationIds) + "\n";
    }
    return result;
}

bool BasePart::HasActivationBounds() const
{
    return false;
}

void BasePart::ModifyActivationBounds(int16_t, int16_t)
{}

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
