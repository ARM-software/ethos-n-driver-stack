//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../MacroUtils.hpp"

#include <ethosn_utils/SmallVector.hpp>
#include <ethosn_utils/Span.hpp>

#include <cstdint>

namespace ethosn
{
namespace command_stream
{
namespace cascading
{

/// Slot info for data in SRAM
struct Tile
{
    uint16_t baseAddr;
    uint16_t numSlots;
    uint16_t slotSize;
};

ETHOSN_DECL_SV_VECTOR_STRUCT(TensorSize, height, width, channels)

/// Ifm/Ofm Streamer work size
ETHOSN_DECL_SV_VECTOR_STRUCT(FmSWorkSize, height, width, channels)

/// Ifm/Ofm Streamer common data
struct FmSData
{
    /// Starting offset of the tensor inside the supertensor
    uint32_t dramOffset;
    /// Buffer ID of the supertensor
    uint16_t bufferId;
    /// IFM/OFM SRAM tile info
    Tile tile;
    /// Default stripe size. Actual stripe size could be smaller at the tensor edges
    TensorSize<uint16_t> dfltStripeSize;
    /// Size of the stripes at the edge of each dimension
    TensorSize<uint16_t> edgeStripeSize;
    /// Strides in dram for stripe coordinates
    FmSWorkSize<uint16_t> stripeDramStrides;
    /// Number of unique stripes in each tensor dimension (numStripesTotal will be
    /// a larger multiple of the product of all dimensions if reloading is needed)
    FmSWorkSize<uint16_t> numStripes;
    /// Stride info for stripe ID (scalar) to stripe coord (ND) conversion
    FmSWorkSize<uint16_t> stripeIdStrides;
};

/// Ifm Streamer data
struct IfmS
{
    FmSData fmData;
    // add read-specific fields as needed
};

/// Output Streamer data
struct OfmS
{
    FmSData fmData;
    // add write-specific fields as needed
};

/// Weight Streamer data
struct WgtS
{
    /// Buffer ID of the weights tensor
    uint16_t bufferId;
    /// Buffer ID of the weights metadata array of (offset, size) pairs
    uint16_t metadataBufferId;
    /// Weight SRAM tile info
    Tile tile;
    /// Number of unique stripes (numStripesTotal will be a larger multiple if reloading is needed)
    uint16_t numStripes;
};

struct BlockSize
{
    uint8_t width;
    uint8_t height;
};

/// Mce Scheduler work size
ETHOSN_DECL_SV_VECTOR_STRUCT(MceSWorkSize, ofmHeight, ofmWidth, ofmChannels, ifmChannels)

/// Mce Scheduler data
struct MceS
{
    using WorkSize = MceSWorkSize<uint16_t>;

    /// IFM SRAM tile info
    Tile ifmTile;
    /// Weight SRAM tile info
    Tile wgtTile;
    /// Mce block size
    BlockSize blockSize;
    WorkSize dfltStripeSize;
    WorkSize edgeStripeSize;
    /// Number of stripes for each "work" dimension
    WorkSize numStripes;
    /// Stride info for stripe ID (scalar) to stripe coord (ND) conversion
    WorkSize stripeIdStrides;
};

/// Ple Loader data
struct PleL
{
    // Add fields as needed
};

/// Ple Scheduler work size
ETHOSN_DECL_SV_VECTOR_STRUCT(PleSWorkSize, ofmHeight, ofmWidth, ofmChannels)

/// Ple Scheduler data
struct PleS
{
    using WorkSize = PleSWorkSize<uint16_t>;

    // Add fields as needed

    /// PLE SRAM OFM tile info
    Tile ofmTile;
    /// Number of unique stripes in each ofm tensor dimension
    WorkSize numStripes;
    /// Stride info for stripe ID (scalar) to stripe coord (ND) conversion
    WorkSize stripeIdStrides;
};

/// Enum tag for agent data
enum class AgentType : uint32_t
{
    IFM_STREAMER,
    WGT_STREAMER,
    MCE_SCHEDULER,
    PLE_LOADER,
    PLE_SCHEDULER,
    OFM_STREAMER,
};

/// Immutable tagged union of agent data that can only be constructed from the concrete agent data type.
/// The corresponding constructor overload will set the enum tag accordingly. Note that constructors are
/// intentionally not explicit because implicit conversion is desirable for cleaner code.
struct AgentData
{
    const AgentType type;

    union
    {
        const IfmS ifm;
        const WgtS wgt;
        const MceS mce;
        const PleL pleL;
        const PleS ple;
        const OfmS ofm;
    };

    constexpr AgentData(const IfmS& data)
        : type{ AgentType::IFM_STREAMER }
        , ifm{ data }
    {}

    constexpr AgentData(const WgtS& data)
        : type{ AgentType::WGT_STREAMER }
        , wgt{ data }
    {}

    constexpr AgentData(const MceS& data)
        : type{ AgentType::MCE_SCHEDULER }
        , mce{ data }
    {}

    constexpr AgentData(const PleL& data)
        : type{ AgentType::PLE_LOADER }
        , pleL{ data }
    {}

    constexpr AgentData(const PleS& data)
        : type{ AgentType::PLE_SCHEDULER }
        , ple{ data }
    {}

    constexpr AgentData(const OfmS& data)
        : type{ AgentType::OFM_STREAMER }
        , ofm{ data }
    {}
};

/// Used to represent a ratio in the number of stripes of this/other agent
/// that are needed by other/this agent
struct Ratio
{
    uint8_t other;
    uint8_t self;
};

/// Used to represent a dependency between this agent and some other agent
struct Dependency
{
    /// Relative position of the other agent wrt the agent that owns this Dependency object.
    /// We can use unsigned type because it always references another agent, down the sequence
    /// for schedule and write-after-read dependencies, and up the sequence for read-after-write
    /// dependencies. The sign is implicit in that way. Using unsigned for extra range.
    uint8_t relativeAgentId;
    /// In the presence of reloads, the number of stripes in self/other in each reload.
    Ratio outerRatio;
    /// Ratio between stripe counters. E.g. two Ifm Streamer stripes might be needed for each
    /// stripe of the consumer Mce Scheduler
    Ratio innerRatio;
    /// Extra number of stripes that are needed. E.g. 3x3 conv:
    ///    IfmS stripes  MceS stripes
    ///            +        *
    ///            |        |
    ///            +        | +
    ///            |        | |
    ///            +        * *
    ///            |        | |
    ///            +        + | +
    ///            |          | |
    ///            +          * *
    ///            |          | |
    ///            +          + |  <- innerRatio[IfmS] = 1 / 2
    ///            |            |
    ///            +            *
    ///            |            |  <- boundary = 1
    ///            +            +
    int8_t boundary;
};

/// Contains dependency info for an agent
struct AgentDependencyInfo
{
    // Total number of stripes for this Agent including reloads (if any)
    uint16_t numStripesTotal;
    /// Array of schedule dependencies. Size 1 for now, could change if we identify a use case for it.
    std::array<Dependency, 1> scheduleDependencies;
    /// Array of read-after-write dependencies. Size 2 for mce and ple-only with two inputs,
    /// could change if we identify a use case for it.
    std::array<Dependency, 2> readDependencies;
    /// Array of write-after-read dependencies related to a tile size. The agent should pause progress before
    /// overwriting a slot in the tile until the existing data is no longer needed by any reader agent.
    /// Size 1 for now, could change if we identify a use case for it.
    std::array<Dependency, 1> writeDependencies;
};

/// Contains tagged agent data and dependency info for an agent
struct Agent
{
    /// Agent-type-specific data
    AgentData data;
    /// Dependency info
    AgentDependencyInfo info;
};

/// A command stream is nothing more than a contiguous sequence of Agent objects.
/// This enables index-based, random access to the different objects in the sequence.
using CommandStream = std::span<const Agent>;

}    // namespace cascading
}    // namespace command_stream
}    // namespace ethosn
