//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Part.hpp"
#include "Plan.hpp"

#include <deque>
#include <memory>
#include <unordered_map>

namespace ethosn
{
namespace support_library
{
/// The graph of Ops and Buffers that would need to be inserted between two plans to make the compatible,
/// for example some DmaOps.
struct Glue
{
    Glue() noexcept
    {}

    OwnedOpGraph m_Graph;
    /// The Op (and which of its inputs) of m_Graph that need to be connected to the output buffer of 'plan1'.
    /// Unused if no glue is required.
    std::pair<Op*, uint32_t> m_InputSlot = { nullptr, 0 };
    /// The Op of m_Graph that needs to connected to the input buffer of 'plan2'.
    /// Unused if no glue is required.
    Op* m_Output = nullptr;
};

/// The result of ArePlansCompatible.
/// This is more complicated than a simple yes/no because some plans will need Dma ops inserting between them
/// to make them compatible.
struct PlanCompatibilityResult
{
    PlanCompatibilityResult() noexcept
    {}

    bool m_IsCompatible = false;

    bool m_RequiresGlue = false;
    /// The graph of Ops and Buffers that would need to be inserted between the two plans to make the compatible,
    /// for example some DmaOps.
    /// This may be empty if no glue is required.
    Glue m_Glue;
};

/// A single element in a combination
struct Elem
{
    struct Link
    {
        PlanId m_Id;
        const Glue* m_Glue;
    };
    using Glues = std::unordered_map<const Edge*, Link>;

    PartId m_PartId;
    PlanId m_PlanId;
    Glues m_Glues;
};

struct Scratch
{
    using Indexes = std::unordered_map<PartId, size_t>;
    using Dst     = std::vector<const Edge*>;
    using Edges   = std::unordered_map<PartId, Dst>;

    uint32_t m_AllocatedSram;

    Indexes m_Idx;
    Edges m_Edges;
    PartId m_CurrPartId;

    size_t m_Score = 0;
};

struct Combination
{
    using Elems          = std::vector<Elem>;
    Combination& operator=(const Combination& c) = default;

    /// Helpers
    /// @{
    size_t GetNumElems() const
    {
        return m_Elems.size();
    }
    /// @}

    /// Scratch
    /// @{
    Scratch m_Scratch;
    /// @}

    Elems m_Elems;
};

using Combinations = std::vector<Combination>;

// Compatible plan of a destination part given the source part and
// its plan. The glue member tells how the plans are connected.
struct CompatiblePlan
{
    CompatiblePlan()                          = default;
    CompatiblePlan(CompatiblePlan&& rhs)      = default;
    CompatiblePlan(const CompatiblePlan& rhs) = delete;
    CompatiblePlan& operator=(CompatiblePlan&& rhs) = delete;

    Glue m_Glue;
    PlanId m_Id;
};

// Vector of all incompatible plans given the part.
// The index of the outer vector represents the part id.
using IncompatiblePlans = std::vector<std::vector<PlanId>>;

// Vector of all compatible plans of a destination part given the
// source part and its plan.
using CompatiblePlans = std::vector<CompatiblePlan>;

// |-------------------------------------------------------------------|
// |     Id         |               CompatiblePlans                    |
// |-------------------------------------------------------------------|
// |                |                                                  |
// |   PlanIdX      |     {{ PlanIdA, Glue1}, ... , {PlanIdW, GlueN}}  |
// |                |                                                  |
// |-------------------------------------------------------------------|
// |    ...         |                  ...                             |
// Note this is an *ordered* map to give deterministic results.
using CompatiblePlansOfPart = std::map<PlanId, CompatiblePlans>;

// |-----------------------------------------------------------------|
// |     Edge       |               CompatiblePlansOfPart            |
// |-----------------------------------------------------------------|
// |                |   Id     |           CompatiblePlans           |
// |                |------------------------------------------------|
// |     EdgeY      |  PlanIdX |     {{ PlanIdA, Glue1}, ... }       |
// |                |------------------------------------------------|
// |                |   ...    |               ...                   |
// |-----------------------------------------------------------------|
// |    ...         |   ...    |               ...                   |
class CompatiblePlansOfParts : public std::unordered_map<const Edge*, CompatiblePlansOfPart>
{
public:
    CompatiblePlansOfParts()                             = default;
    CompatiblePlansOfParts(CompatiblePlansOfParts&& rhs) = default;
};

using SrcPart = std::unordered_map<const Edge*, PartId>;
using DstPart = std::unordered_map<const Edge*, PartId>;

struct MetadataOfPart
{
    SrcPart m_Source;
    DstPart m_Destination;
    PartId m_PartId;
    CompatiblePlansOfParts m_Comp;
};

// |-----------------------------------------------------------|----
// |                    Part0                                  | ...
// |-----------------------------------------------------------|----
// |                                                           |
// |  { PartIdG, PartIdQ, ... , ... , ..., PartIdY }           | ...
// |                                                           |
// |-----------------------------------------------------------|----
// |    Edge        |               CompatiblePlansOfPart      | ...
// |-----------------------------------------------------------|----
// |                |   Key    |           CompatiblePlans     | ...
// |                |------------------------------------------|----
// |    EdgeY       |  PlanIdX |     {{ PlanIdA, Glue1}, ... } | ...
// |                |------------------------------------------|----
// |                |   ...    |               ...             | ...
// |-----------------------------------------------------------|----
// |    ...         |   ...    |               ...             | ...
using Metadata = std::deque<MetadataOfPart>;

struct GrownSeeds
{
    bool m_Terminated           = true;
    Combinations m_Combinations = {};
};

enum class GrowScheme
{
    MergeOnly,
    DramOnly,
    Default
};

/// Checks whether two given plans are compatible, i.e. whether plan1 could be joined to plan2 along the given Edge.
PlanCompatibilityResult ArePlansCompatible(
    const Plan& plan1, const Plan& plan2, const Edge& edge, const HardwareCapabilities&, const bool forceGlue = false);

// Create a Metadata structure containing all the compatible
// succession of plans of two topologically consecutive parts.
// E.g.:
//                   PartX -> PartY
//
// For each plan in PartX list all the compatible plans of PartY.
// No SRAM allocation verification is performed at this stage.
Metadata CreateMetadata(const GraphOfParts&, const HardwareCapabilities&);

// Create the seeds from which all the combinations are going to be derived.
// The seeds are created from the first part in topological order.
//  E.g.:
//                    PartX -> PartY
//
// This represents all the combinations of all the compatible plans of
// PartX and PartY. At this stage two plans can be merged if they
// meet the SRAM allocation requirements (e.g. all required buffers fit
// in the SRAM).
Combinations CreateSeeds(const GraphOfParts&, const Metadata&, const HardwareCapabilities&);

// The input combinations seeds are grown by one plan at each iteration
// until all the combinations have length equal to the number of parts.
GrownSeeds
    GrowSeeds(const Combinations&, const GraphOfParts&, const Metadata&, const HardwareCapabilities&, const GrowScheme);

/// Creates a single OpGraph which contains the full graph of Ops and Buffers for the given Combination.
/// This handles merging of adjacent Plans and Glues to give a homogenous structure, suitable for
/// Estimation or Generation into a command stream.
OpGraph GetOpGraphForCombination(const Combination& combination, const GraphOfParts& parts);

}    // namespace support_library
}    // namespace ethosn
