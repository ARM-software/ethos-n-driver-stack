//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../Graph.hpp"
#include "DebuggableObject.hpp"

#include <functional>
#include <utility>
#include <vector>

namespace ethosn
{
namespace support_library
{

class Buffer;

enum class CascadeType
{
    Beginning,
    Middle,
    End,
    Lonely
};

enum class CascadingBufferFormat
{
    NHWC,
    NCHW,
    NHWCB,
    WEIGHT,
    FCAF_DEEP,
    FCAF_WIDE
};

template <typename D, typename B>
D* GetObjectAs(B* obj)
{
    return dynamic_cast<D*>(obj);
}

template <typename D, typename B>
const D* GetObjectAs(const B* obj)
{
    return dynamic_cast<const D*>(obj);
}

template <typename D, typename B>
bool IsObjectOfType(const B* obj)
{
    return (GetObjectAs<D>(obj) != nullptr);
}

using PartId         = uint32_t;
using StripeSizeType = TensorShape::value_type;
using Plans          = std::vector<Plan>;
using InPart         = std::pair<bool, PartId>;
using OutPart        = std::pair<bool, PartId>;
using Nodes          = std::vector<Node*>;

// Object which represents the input to a part
// This consists of the PartId of the part connected
// and the index of that input into the part
struct PartInputSlot
{
    PartId m_PartId;
    uint32_t m_InputIndex;
    bool operator==(const PartInputSlot& r) const
    {
        return m_PartId == r.m_PartId && m_InputIndex == r.m_InputIndex;
    }
    bool operator<(const PartInputSlot& r) const
    {
        if (m_PartId < r.m_PartId)
            return true;
        if (r.m_PartId < m_PartId)
            return false;
        if (m_InputIndex < r.m_InputIndex)
            return true;
        if (r.m_InputIndex < m_InputIndex)
            return false;
        return false;
    }
};

// Object which represents the output to a part
// This consists of the PartId of the part connected
// and the index of that output out of the part
struct PartOutputSlot
{
    PartId m_PartId;
    uint32_t m_OutputIndex;
    bool operator==(const PartOutputSlot& r) const
    {
        return m_PartId == r.m_PartId && m_OutputIndex == r.m_OutputIndex;
    }
    bool operator<(const PartOutputSlot& r) const
    {
        if (m_PartId < r.m_PartId)
            return true;
        if (r.m_PartId < m_PartId)
            return false;
        if (m_OutputIndex < r.m_OutputIndex)
            return true;
        if (r.m_OutputIndex < m_OutputIndex)
            return false;
        return false;
    }
};

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

using PartInputMapping  = std::map<Buffer*, PartInputSlot>;
using PartOutputMapping = std::map<Buffer*, PartOutputSlot>;
class OwnedOpGraph;

class BasePart : public DebuggableObject
{
public:
    BasePart(PartId id,
             const EstimationOptions& estOpt,
             const CompilationOptions& compOpt,
             const HardwareCapabilities& capabilities)
        : DebuggableObject("BasePart")
        , m_PartId{ id }
        , m_CompilerDataFormat{ CompilerDataFormat::NONE }
        , m_EstimationOptions{ estOpt }
        , m_CompilationOptions{ compOpt }
        , m_Capabilities{ capabilities }
    {}
    BasePart(PartId id,
             const CompilerDataFormat compilerDataFormat,
             const std::set<uint32_t> correspondingOperationIds,
             const EstimationOptions& estOpt,
             const CompilationOptions& compOpt,
             const HardwareCapabilities& capabilities)
        : DebuggableObject("BasePart")
        , m_PartId{ id }
        , m_CompilerDataFormat{ compilerDataFormat }
        , m_CorrespondingOperationIds{ correspondingOperationIds }
        , m_EstimationOptions{ estOpt }
        , m_CompilationOptions{ compOpt }
        , m_Capabilities{ capabilities }
    {}
    PartId GetPartId() const;
    virtual Plans GetPlans(CascadeType cascadeType,
                           ethosn::command_stream::BlockConfig blockConfig,
                           Buffer* sramBuffer,
                           uint32_t numWeightStripes) const = 0;
    virtual utils::Optional<ethosn::command_stream::MceOperation> GetMceOperation() const;

    virtual bool HasActivationBounds() const;
    virtual void ModifyActivationBounds(int16_t lowerBound, int16_t upperBound);

    DotAttributes GetDotAttributes(DetailLevel) const override;

    virtual ~BasePart()
    {}

protected:
    PartId m_PartId;
    const CompilerDataFormat m_CompilerDataFormat;
    const std::set<uint32_t> m_CorrespondingOperationIds;
    const EstimationOptions& m_EstimationOptions;
    const CompilationOptions& m_CompilationOptions;
    const HardwareCapabilities& m_Capabilities;
    void AddNewPlan(PartInputMapping&& inputMappings,
                    PartOutputMapping&& outputMappings,
                    OwnedOpGraph&& opGraph,
                    Plans& plans) const;
};

using Parts = std::vector<std::unique_ptr<BasePart>>;

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

    /// Retrieves the connections between source and dest PartIds
    std::vector<PartConnection> GetConnectionsBetween(PartId source, PartId dest) const;

    /// Adds a connection between input slot and output slot to the graph of parts
    /// asserts if the input slot is already connected to an output slot.
    void AddConnection(PartInputSlot inputSlot, PartOutputSlot outputSlot);

    PartId GeneratePartId()
    {
        PartId currId = m_NextPartId;
        ++m_NextPartId;
        return currId;
    }

    Parts m_Parts;
    std::unordered_map<PartInputSlot, PartOutputSlot> m_Connections;
    PartId m_NextPartId = 0;
};

}    // namespace support_library
}    // namespace ethosn
