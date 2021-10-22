//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Part.hpp"
#include "Plan.hpp"

#include "DebuggingContext.hpp"

namespace ethosn
{
namespace support_library
{

using PlanCache = std::unordered_map<PartId, Plans>;

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
    std::vector<Op*> m_Output;

    uint32_t m_OutDmaOffset = 0;
};

struct GlueInfo
{
    const Glue* m_Glue;
    bool m_OutDma;
};

/// A single element in a combination
struct Elem
{
    using Glues = std::unordered_map<PartInputSlot, GlueInfo>;

    std::shared_ptr<Plan> m_Plan;
    Glues m_Glues;
};

constexpr size_t g_InvalidCombRank = std::numeric_limits<size_t>::max();

struct Combination
{
    Combination()
    {}

    // Create a combination with a single element without any edge/glue information
    Combination(const BasePart& part, std::shared_ptr<Plan> plan, size_t orderRank, const GraphOfParts& graphOfParts)
        : Combination(part, plan, nullptr, nullptr, orderRank, false, graphOfParts)
    {}

    // Create a combination with a single element without plan information,
    // this is used when updating edge/glue information for a part with
    // multiple outputs where the plan has been already selected and
    // won't be changed when merging combinations
    // Note glue should not change the header ID and rank of the
    // combination
    Combination(const BasePart& part, const PartInputSlot* edge, const Glue* glue, const GraphOfParts& graphOfParts)
        : Combination(part, nullptr, edge, glue, SIZE_MAX, true, graphOfParts)
    {}

    Combination(const BasePart& part,
                const PartInputSlot* edge,
                const Glue* glue,
                bool outDma,
                const GraphOfParts& graphOfParts)
        : Combination(part, nullptr, edge, glue, SIZE_MAX, outDma, graphOfParts)
    {}

    // Create a combination with a single element with edge/glue information,
    // if no edge/glue information is provided (e.g. nullptr) the combination
    // will consider the case where no glue is required on any output edge of
    // the part
    Combination(const BasePart& part,
                std::shared_ptr<Plan> plan,
                const PartInputSlot* slot,
                const Glue* glue,
                size_t orderRank,
                bool outDma,
                const GraphOfParts& graphOfParts)
    {
        // Create a new element
        Elem elem = { plan, {} };
        // Insert glue value (it can be null if no glue is required)
        // if a valid edge is provided
        if (slot)
        {
            GlueInfo glueInfo = { glue, outDma };
            elem.m_Glues.insert(std::make_pair(*slot, glueInfo));
        }
        else
        {
            // Consider no glue on all the output edges (i.e. mergeable)
            const auto& destSlots = graphOfParts.GetDestinationConnections(part.GetPartId());
            for (auto& slots : destSlots)
            {
                GlueInfo glueInfo = { nullptr, false };
                elem.m_Glues.insert(std::make_pair(slots.m_Destination, glueInfo));
            }
        }
        m_Elems.insert(std::make_pair(part.GetPartId(), elem));

        // Update the Header's rank in topological order
        m_HeadOrderRank = orderRank;

        // The partId is not pushed to the part ID list
        // if this is a glue.
        if (orderRank != g_InvalidCombRank)
        {
            m_PartIdsInOrder.push_back(part.GetPartId());
        }
    }

    Combination operator+(const Combination& rhs) const
    {
        Combination result = *this;

        // The header order rank decides the order
        // how part ID vectors are merged.
        if (result.m_HeadOrderRank > rhs.m_HeadOrderRank)
        {
            result.m_HeadOrderRank = rhs.m_HeadOrderRank;
            result.m_PartIdsInOrder.insert(result.m_PartIdsInOrder.begin(), rhs.m_PartIdsInOrder.begin(),
                                           rhs.m_PartIdsInOrder.end());
        }
        else if (!rhs.m_PartIdsInOrder.empty())
        {
            result.m_PartIdsInOrder.insert(result.m_PartIdsInOrder.end(), rhs.m_PartIdsInOrder.begin(),
                                           rhs.m_PartIdsInOrder.end());
        }

        for (auto& rhsElemIt : rhs.m_Elems)
        {
            auto resultElemIt = result.m_Elems.find(rhsElemIt.first);
            if (resultElemIt != result.m_Elems.end())
            {
                assert(rhsElemIt.second.m_Plan.get() == nullptr ||
                       resultElemIt->second.m_Plan == rhsElemIt.second.m_Plan);
                for (auto& glueIt : rhsElemIt.second.m_Glues)
                {
                    auto edgeIt = resultElemIt->second.m_Glues.find(glueIt.first);
                    if (edgeIt != resultElemIt->second.m_Glues.end())
                    {
                        // Take the glue value of rhs
                        edgeIt->second = glueIt.second;
                    }
                    else
                    {
                        // Add edge/glue information
                        resultElemIt->second.m_Glues.insert(glueIt);
                    }
                }
            }
            else
            {
                result.m_Elems.insert(rhsElemIt);
            }
        }
        return result;
    }

    using Elems          = std::unordered_map<PartId, Elem>;
    Combination& operator=(const Combination& c) = default;

    /// Helpers
    /// @{
    size_t GetNumElems() const
    {
        return m_Elems.size();
    }
    /// @}

    Elems m_Elems;
    size_t m_HeadOrderRank = g_InvalidCombRank;
    std::vector<PartId> m_PartIdsInOrder;
};

enum class StatsType
{
    SinglePartSection,
    StartSection,
    ContinueSection,
    EndSection,
    FindBestCombinationForPart,
    NumStats,
};

using Combinations = std::vector<Combination>;

class Combiner
{
public:
    Combiner(const GraphOfParts& graphOfParts,
             const HardwareCapabilities& capabilities,
             const EstimationOptions& estOpt,
             const DebuggingContext& debuggingContext);

    bool IsPartInput(const BasePart& part) const;
    bool IsPartOutput(const BasePart& part) const;

    bool IsPartSi(const BasePart& part) const;
    bool IsPartSo(const BasePart& part) const;
    bool IsPartMo(const BasePart& part) const;
    bool IsPartSiso(const BasePart& part) const;
    bool IsPartSimo(const BasePart& part) const;
    bool IsPartMiso(const BasePart& part) const;
    bool IsPartMimo(const BasePart& part) const;

    bool AreMceOperationsCompatible(const Buffer* plan1OutputBuffer,
                                    const Buffer* plan2InputBuffer,
                                    const PartOutputSlot& outputSlot) const;

    bool AreBlockConfigsCompatible(const Plan& plan1, const Plan& plan2, const PartOutputSlot& outputSlot) const;

    bool ArePlansCompatible(const Plan& sPlan, const Plan& dPlan, const PartConnection& outputSlot);
    bool ArePlansCompatibleImpl(const Plan& sPlan, const Plan& dPlan, const PartConnection& outputSlot) const;

    bool IsPlanAllocated(SramAllocator& alloc, const Plan& plan) const;
    bool IsPlanInputGlueable(const Plan& plan) const;
    bool IsPlanOutputGlueable(const Plan& plan) const;
    bool ArePlansAllowedToMerge(const Plan& reference, const Plan& current, const PartConnection& outputSlot) const;
    bool ArePlansStreamingStrategiesCompatible(const Plan& reference,
                                               const Plan& current,
                                               const PartConnection& slots) const;

    Combination GetBestCombination() const;
    Combination GetBestCombination(Combinations& combs) const;
    CascadingBufferFormat
        GetBestCascadingBufferDramFormat(const std::array<TensorShape, 2> inputOutputStripeShapes) const;

    const Plan& GetPlanForPartFromCombination(const BasePart& part, const Combination& comb) const;
    std::pair<bool, const Glue*> GetGlue(const Buffer* outputBuffer, const Buffer* inputBuffer);
    std::pair<bool, const Glue*> GetSharedGlue(const Buffer* outputBuffer, std::vector<const Buffer*>& inputBuffer);

    Combination FindBestCombinationForPart(const BasePart& part);
    virtual Combination FindBestCombinationForPartImpl(const BasePart& part);

    Combination ContinueSection(const BasePart& part,
                                const BasePart& sPart,
                                const Combination& comb,
                                const SramAllocator& alloc);

    Combination SinglePartSection(const BasePart& part);

    Combination
        EndSection(const BasePart& part, const BasePart& sPart, const Combination& comb, const SramAllocator& alloc);

    Combination StartSection(const BasePart& part, const BasePart& nextPart, const SramAllocator& alloc);

    std::unique_ptr<Glue> GenerateGlueBetweenSramAndDram() const;
    std::unique_ptr<Glue> GenerateGlueBetweenSramAndSram(const Buffer* buffer,
                                                         const CascadingBufferFormat cascadingBufferFormat) const;
    std::unique_ptr<Glue> GenerateGlueBetweenSramAndSrams(const Buffer* buffer,
                                                          const CascadingBufferFormat cascadingBufferFormat,
                                                          uint32_t numOfOuputs) const;
    Combination GluePartToCombinationDestToSrcs(const BasePart& part,
                                                const Combination& comb,
                                                const std::vector<PartConnection>& sources);

    Combination GluePartToCombinationSrcToDests(const BasePart& sPart,
                                                const Combination& comb,
                                                const std::vector<PartConnection>& destPartEdge);

    const BasePart* GetNextPart(const BasePart* part)
    {
        return m_PartOrderTable[part->GetPartId()].second;
    }

    void SavePartsPlans(const BasePart& part, const Plans& plans) const;

    void UpdateStats(const StatsType type);

    void Run();

    enum class PartState
    {
        Visiting,
        Visited,
    };

    bool Visit(const BasePart* current,
               std::vector<const BasePart*>& outSorted,
               std::map<const BasePart*, PartState>& partStates);

    bool TopologicalSortParts();

    template <typename... Args>
    Plans GetPlansCached(const BasePart& part, Args&&... args)
    {
        // Note the cache only uses the part id (instead of all the plan parameters)
        // because when specific plan generation is used the plans should be unique and
        // the cache can be removed.
        auto planInCache = m_PlanCache.find(part.GetPartId());
        if (planInCache != m_PlanCache.end())
        {
            return planInCache->second;
        }
        else
        {
            auto plans = part.GetPlans(std::forward<Args>(args)...);
            SavePartsPlans(part, plans);
            m_PlanCache.emplace(part.GetPartId(), plans);
            return plans;
        }
    }

private:
    const GraphOfParts& m_GraphOfParts;
    const HardwareCapabilities& m_Caps;
    const EstimationOptions& m_EstOpt;
    const DebuggingContext& m_DebuggingContext;

    const BasePart* m_FirstPartAfterSort = nullptr;
    std::vector<std::pair<size_t, const BasePart*>> m_PartOrderTable;

    Combination m_BestCombination;

    std::vector<std::unique_ptr<Glue>> m_GluesVector;
    std::unordered_map<const BasePart*, const Combination> m_CombinationPerPartMap;
    PlanCache m_PlanCache;

    std::vector<size_t> m_Stats{ std::vector<size_t>(static_cast<size_t>(StatsType::NumStats), 0) };
};

OpGraph GetOpGraphForCombination(const Combination& combination, const GraphOfParts& parts);

}    // namespace support_library
}    // namespace ethosn
