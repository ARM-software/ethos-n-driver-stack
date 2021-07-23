//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Part.hpp"
#include "Plan.hpp"

#include <map>

namespace ethosn
{
namespace support_library
{
namespace depth_first_search
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
    Combination()
    {}

    Combination(const Part& part, const Plan& plan)
    {
        m_Elems.push_back({ part.m_PartId, plan.m_PlanId, {} });
    }

    Combination operator+(const Combination& rhs) const
    {
        return rhs;
    }

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

enum class InOutFormat : uint8_t
{
    SISO = 0,
    SIMO,
    MISO,
    MIMO,
    NUM_INOUT_FORMATS
};

using Combinations = std::vector<Combination>;

struct Combiner
{
    Combiner(const GraphOfParts& graphOfParts,
             const HardwareCapabilities& capabilities,
             const EstimationOptions& estOpt);

    bool IsPartInput(const Part& part);

    bool IsPartSiso(const Part& part);
    bool IsPartSisoImpl(const Part& part) const;

    bool IsPartSimo(const Part& part);
    bool IsPartSimoImpl(const Part& part) const;

    bool IsPartMiso(const Part& part);
    bool IsPartMisoImpl(const Part& part) const;

    bool IsPartMimo(const Part& part);
    bool IsPartMimoImpl(const Part& part) const;

    bool IsPlanMergeable(const Combination& comb, const Part& part, const Plan& plan);
    bool IsPlanMergeableImpl(const Combination& comb, const Part& part, const Plan& plan) const;

    bool IsPlanAllowed(const Combination& comb, const Part& part, const Plan& plan);
    bool IsPlanAllowedImpl(const Combination& comb, const Part& part, const Plan& plan) const;

    bool IsPlanAllocated(SramAllocator& alloc, const Combination& comb, const Part& part, const Plan& plan);

    template <InOutFormat format>
    bool IsPartFormat(const Part& part);

    Combination GetBestCombination() const;
    Combination GetBestCombination(Combinations& combs) const;

    const Part* GetNextPart(const Part& part) const;
    std::vector<const Part*> GetDestinationParts(const Part& part);

    Combination FindBestCombinationsForPart(const Part& part);
    Combination FindBestCombinationsForPartImpl(const Part& part);

    Combination ContinueSection(const Part& part, const Combination& comb, const SramAllocator& alloc);

    void Run();

    const GraphOfParts& m_GraphOfParts;
    const HardwareCapabilities& m_Caps;
    const EstimationOptions& m_EstOpt;

    Combination m_BestCombination;

    std::vector<std::map<const PartId, const bool>> m_InOutMap;
};

OpGraph GetOpGraphForCombination(const Combination& combination, const GraphOfParts& parts);

}    // namespace depth_first_search
}    // namespace support_library
}    // namespace ethosn
