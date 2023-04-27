//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../Graph.hpp"
#include "../Utils.hpp"
#include "DebuggableObject.hpp"
#include "Part.hpp"

#include <functional>
#include <map>
#include <typeinfo>
#include <utility>
#include <vector>

namespace ethosn
{
namespace support_library
{

/// PartConnection describes a connection between parts.
/// The source of a connection is an output slot of a part
/// The destination of a connection is the input slot to a part.
/// e.g. Part0 output slot 0 is connected to Part1 input slot 0
/// P0 0------>0 P1
/// The source of the connection is P0 output slot 0 {0,0} and the destination is P1 input slot 0 {0,1}.
struct PartConnection
{
    PartInputSlot m_Destination;
    PartOutputSlot m_Source;

    bool operator==(const PartConnection& r) const
    {
        return m_Destination == r.m_Destination && m_Source == r.m_Source;
    }

    bool operator<(const PartConnection& r) const
    {
        if (m_Destination < r.m_Destination)
            return true;
        if (r.m_Destination < m_Destination)
            return false;
        if (m_Source < r.m_Source)
            return true;
        if (r.m_Source < m_Source)
            return false;
        return false;
    }
};

class WeightEncoderCache;

}    // namespace support_library

}    // namespace ethosn

namespace ethosn_impl
{
inline size_t HashCombine(ethosn::support_library::PartId partId, uint32_t index)
{
    std::hash<uint64_t> hasher;
    uint64_t combinedKey = (static_cast<uint64_t>(partId) << 32) | (index);
    size_t ret           = hasher(combinedKey);

    return ret;
}
}    // namespace ethosn_impl

namespace std
{
template <>
struct hash<ethosn::support_library::PartInputSlot>
{
    size_t operator()(const ethosn::support_library::PartInputSlot& p) const noexcept
    {
        return ethosn_impl::HashCombine(p.m_PartId, p.m_InputIndex);
    }
};

template <>
struct hash<ethosn::support_library::PartOutputSlot>
{
    size_t operator()(const ethosn::support_library::PartOutputSlot& p) const noexcept
    {
        return ethosn_impl::HashCombine(p.m_PartId, p.m_OutputIndex);
    }
};

template <>
struct hash<ethosn::support_library::PartConnection>
{
    size_t operator()(const ethosn::support_library::PartConnection& s) const noexcept
    {
        uint64_t destHash = ethosn_impl::HashCombine(s.m_Destination.m_PartId, s.m_Destination.m_InputIndex);
        uint64_t srcHash  = ethosn_impl::HashCombine(s.m_Source.m_PartId, s.m_Source.m_OutputIndex);

        uint64_t hash = 17;
        hash          = hash * 31 + destHash;
        hash          = hash * 31 + srcHash;
        return hash;
    }
};
}    // namespace std

namespace ethosn
{
namespace support_library
{

/// The GraphOfParts contains the parts and the connections between them.
/// The connection between parts is stored as a map from PartInputSlot and PartOutputSlot as an input slot can only have 1 output slot.
///
/// e.g. A graph of parts with two part output slots {0,0} and {0,1} (corresponding to P0)
///      and 2 part input slots {1,0} (corresponding to P1) and {2,0} (corresponding to P2)
///
/// P0 0------>0 P1
///  |
///    1------>0 P2
///
class GraphOfParts
{
public:
    GraphOfParts() = default;

    size_t GetNumParts() const;
    const BasePart& GetPart(const PartId id) const;
    const Parts& GetParts() const;

    /// Takes ownership of the internal array of parts, leaving this object empty.
    Parts ReleaseParts();

    void AddPart(std::unique_ptr<BasePart> p);

    const std::unordered_map<PartInputSlot, PartOutputSlot>& GetAllConnections() const;

    /// Methods to retrieve the input / output slots for a part
    std::vector<PartInputSlot> GetPartInputs(PartId p) const;
    std::vector<PartOutputSlot> GetPartOutputs(PartId p) const;

    /// Methods to retrieve the corresponding output / input slots for adjacent parts
    /// Retrieves the OutputSlots for the parts which are sources to Part p
    std::vector<PartOutputSlot> GetSourceParts(PartId p) const;
    /// Retrieves the InputSlots for the parts which are destinations to Part p
    std::vector<PartInputSlot> GetDestinationParts(PartId p) const;

    /// Methods to retrieve the connections for the source and destination parts of p
    std::vector<PartConnection> GetSourceConnections(PartId p) const;
    std::vector<PartConnection> GetDestinationConnections(PartId p) const;

    /// Methods to get the corresponding connected input/output slots of an input/output slot
    std::vector<PartInputSlot> GetConnectedInputSlots(const PartOutputSlot& outputSlot) const;
    utils::Optional<PartOutputSlot> GetConnectedOutputSlot(const PartInputSlot& inputSlot) const;

    /// Adds a connection between input slot and output slot to the graph of parts
    /// asserts if the input slot is already connected to an output slot.
    void AddConnection(PartInputSlot inputSlot, PartOutputSlot outputSlot);

    void RemoveConnection(PartInputSlot inputSlot);

    /// Where possible, merge parts which are tagged as channel selectors with neighbouring
    /// parts, to simplify and speed up the graph. See BasePart::IsChannelSelector() for details.
    void MergeChannelSelectors();

    PartId GeneratePartId()
    {
        PartId currId = m_NextPartId;
        ++m_NextPartId;
        return currId;
    }

    /// Sort the Parts into a topological order suitable for further compilation steps, and compact the Part IDs
    /// such that they are contiguous and start from zero. This is important as some parts may have been
    /// removed as part of other optimisation steps, leaving "gaps" in the part IDs. Having contiguous
    /// Part IDs makes them easier to use for further compilation steps.
    void SortAndCompact();

private:
    Parts m_Parts;
    std::unordered_map<PartInputSlot, PartOutputSlot> m_Connections;
    PartId m_NextPartId = 0;
};

/// An immutable equivalent of GraphOfParts, with faster accessors.
/// This stores cached versions of all the accessor methods, so it makes the accessors much faster
/// at the expense of not being able to change any parts or connections.
class FrozenGraphOfParts
{
public:
    /// Takes a GraphOfParts and "freezes" it.
    explicit FrozenGraphOfParts(GraphOfParts graph);
    FrozenGraphOfParts(const FrozenGraphOfParts& rhs) = delete;
    FrozenGraphOfParts(FrozenGraphOfParts&&)          = default;

    size_t GetNumParts() const;
    const BasePart& GetPart(const PartId id) const;
    const std::vector<std::unique_ptr<BasePart>>& GetParts() const;

    const std::unordered_map<PartInputSlot, PartOutputSlot>& GetAllConnections() const;

    /// Methods to retrieve the input / output slots for a part
    const std::vector<PartInputSlot>& GetPartInputs(PartId p) const;
    const std::vector<PartOutputSlot>& GetPartOutputs(PartId p) const;

    /// Methods to retrieve the corresponding output / input slots for adjacent parts
    /// Retrieves the OutputSlots for the parts which are sources to Part p
    const std::vector<PartOutputSlot>& GetSourceParts(PartId p) const;
    /// Retrieves the InputSlots for the parts which are destinations to Part p
    const std::vector<PartInputSlot>& GetDestinationParts(PartId p) const;

    /// Methods to retrieve the connections for the source and destination parts of p
    const std::vector<PartConnection>& GetSourceConnections(PartId p) const;
    const std::vector<PartConnection>& GetDestinationConnections(PartId p) const;

    /// Methods to get the corresponding connected input/output slots of an input/output slot
    const std::vector<PartInputSlot>& GetConnectedInputSlots(const PartOutputSlot& outputSlot) const;
    const utils::Optional<PartOutputSlot>& GetConnectedOutputSlot(const PartInputSlot& inputSlot) const;

private:
    std::vector<std::unique_ptr<BasePart>> m_Parts;
    std::unordered_map<PartInputSlot, PartOutputSlot> m_Connections;
    std::vector<std::vector<PartInputSlot>> m_PartInputs;
    std::vector<std::vector<PartOutputSlot>> m_PartOutputs;
    std::vector<std::vector<PartOutputSlot>> m_SourceParts;
    std::vector<std::vector<PartInputSlot>> m_DestinationParts;
    std::vector<std::vector<PartConnection>> m_SourceConnections;
    std::vector<std::vector<PartConnection>> m_DestinationConnections;
    std::vector<std::vector<std::vector<PartInputSlot>>> m_ConnectedInputSlots;
    std::vector<std::vector<utils::Optional<PartOutputSlot>>> m_ConnectedOutputSlot;
};

}    // namespace support_library
}    // namespace ethosn
