//
// Copyright © 2021-2022 Arm Limited.
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

// Store the connections between the glues
struct GlueConnections
{
    // Store a map of buffer replacements
    // e.g. used when merging output and input buffers when cascading plans in a section
    // The key is a Buffer in the Plan, and the value is the Buffer that it should be replaced with.
    std::unordered_map<Buffer*, Buffer*> m_ReplacementBuffers;
    // Store the connection between ops and buffers.
    // This can point to ops and buffers which are not within the glue
    std::multimap<Op*, Buffer*> m_OpsToBuffers;
    // Store the connection between buffers and ops.
    // This can point to ops and buffers which are not within the glue
    std::multimap<Buffer*, Op*> m_BuffersToOps;
};

// The end of a plan which connects to another plan
// E.g. for two plans connected as: planA -> planB
// planA -> EndingGlue -> StartingGlue -> planB
struct EndingGlue
{
    EndingGlue() noexcept
    {}

    OwnedOpGraph m_Graph;

    // Store how this enidng glue connects to the previous plan / ending glue
    // Note this is different to the connections in the starting glue as it only stores the connection to its plan
    GlueConnections m_ExternalConnections;
};

// The start of a plan which connects to another plan
// E.g. for two plans connected as: planA -> planB
// planA -> EndingGlue -> StartingGlue -> planB
struct StartingGlue
{
    StartingGlue() noexcept
    {}

    OwnedOpGraph m_Graph;

    // Store how this starting glue connects to the previous plan / ending glue and the following plan
    // Note this is different to the connections in the ending glue as it stores how the connectoion to both its plan
    // AND the previous ending glue/plan
    GlueConnections m_ExternalConnections;
};

// Specifies an ending glue and multiple starting glues
// Used as a return type for functions which generate glue
struct StartingAndEndingGlues
{
    // There can be multiple starting glues when there are branches and multiple input buffers to the same plan.
    std::vector<StartingGlue> m_StartingGlues;
    EndingGlue m_EndingGlue;
};

/// A single element in a combination
struct Elem
{
    std::shared_ptr<Plan> m_Plan;

    // The starting glue attachs to inputs of a plan
    std::unordered_map<PartInputSlot, std::shared_ptr<StartingGlue>> m_StartingGlues;
    // The ending glue attachs to outputs of a plan
    std::unordered_map<PartOutputSlot, std::shared_ptr<EndingGlue>> m_EndingGlues;
};

constexpr size_t g_InvalidCombRank = std::numeric_limits<size_t>::max();

using PleOperations = std::vector<std::pair<command_stream::cascading::PleKernelId, uint32_t>>;

struct Combination
{
    Combination()
    {}

    Combination(const BasePart& part)
        : Combination(part, Plan(), SIZE_MAX)
    {}

    // Create a combination with a single element plan
    Combination(const BasePart& part, Plan&& plan, size_t orderRank)
    {
        // Create a new element
        Elem elem = { std::make_shared<Plan>(std::move(plan)), {}, {} };
        m_Elems.insert({ part.GetPartId(), elem });

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
            // Can't add combinations which share part id's.
            assert(result.m_Elems.find(rhsElemIt.first) == result.m_Elems.end());
            result.m_Elems.insert(rhsElemIt);
        }
        return result;
    }

    void AddEndingGlue(EndingGlue&& glue, PartOutputSlot outputSlot)
    {
        const auto& it = m_Elems.find(outputSlot.m_PartId);
        assert(it != m_Elems.end());
        it->second.m_EndingGlues.insert({ outputSlot, std::make_shared<EndingGlue>(std::move(glue)) });
    }

    void SetStartingGlue(StartingGlue&& glue, PartInputSlot inputSlot)
    {
        const auto& it = m_Elems.find(inputSlot.m_PartId);
        assert(it != m_Elems.end());
        it->second.m_StartingGlues.insert({ inputSlot, std::make_shared<StartingGlue>(std::move(glue)) });
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

    bool IsPlanAllocated(SramAllocator& alloc,
                         const Plan& plan,
                         PleOperations& pleOps,
                         const Buffer* const outBufOfPrevPlanInSection,
                         const StatsType sectionType) const;
    bool IsPlanInputGlueable(const Plan& plan) const;
    bool IsPlanOutputGlueable(const Plan& plan) const;
    bool ArePlansAllowedToMerge(const Plan& reference, const Plan& current, const PartConnection& outputSlot) const;
    bool ArePlansStreamingStrategiesCompatible(const Plan& reference,
                                               const Plan& current,
                                               const PartConnection& slots) const;
    void DeallocateUnusedBuffers(const Plan& sPlan, SramAllocator& allocator);

    const Combination& GetBestCombination() const;
    Combination GetBestCombination(const Combinations& combs);
    OpGraph GetMergedOpGraphForBestCombination() const;
    CascadingBufferFormat GetBestCascadingBufferDramFormat(const std::array<Buffer*, 2> sramBuffers) const;

    const Plan& GetPlanForPartFromCombination(const BasePart& part, const Combination& comb) const;
    std::pair<bool, StartingAndEndingGlues> GetGlue(Buffer* outputBuffer, Buffer* inputBuffer);
    std::pair<bool, StartingAndEndingGlues> GetSharedGlue(Buffer* outputBuffer, std::vector<Buffer*>& inputBuffer);

    Combination FindBestCombinationForPart(const BasePart& part);
    virtual Combination FindBestCombinationForPartImpl(const BasePart& part);

    Combination ContinueSection(const BasePart& part,
                                const BasePart& sPart,
                                const Combination& comb,
                                const SramAllocator& alloc,
                                uint32_t prevNumWeightStripes,
                                bool prevDoubleBuffered,
                                const PleOperations& pleOps,
                                uint32_t totalAgents);

    Combination SinglePartSection(const BasePart& part);

    Combination EndSection(const BasePart& part,
                           const BasePart& sPart,
                           const Combination& comb,
                           const SramAllocator& alloc,
                           uint32_t prevNumWeightStripes,
                           bool prevDoubleBuffered,
                           const PleOperations& pleOps,
                           uint32_t totalAgents);

    Combination StartSection(const BasePart& part, const BasePart& nextPart, const SramAllocator& alloc);

    StartingAndEndingGlues GenerateGlueBetweenSramAndDram(Buffer* sramBuffer,
                                                          Buffer* dramBuffer,
                                                          const CascadingBufferFormat cascadingBufferFormat) const;
    StartingAndEndingGlues GenerateGlueBetweenDramAndSram(Buffer* dramBuffer,
                                                          Buffer* sramBuffer,
                                                          const CascadingBufferFormat cascadingBufferFormat) const;
    StartingAndEndingGlues GenerateGlueBetweenDramAndSramWithConversion(Buffer* inputBuffer,
                                                                        Buffer* outputBuffer) const;
    StartingAndEndingGlues GenerateGlueBetweenSramAndDramWithConversion(Buffer* sourceBuffer, Buffer* destBuffer) const;
    StartingAndEndingGlues GenerateGlueBetweenSramAndSram(Buffer* sourceBuffer,
                                                          Buffer* destBuffer,
                                                          const CascadingBufferFormat cascadingBufferFormat) const;
    StartingAndEndingGlues GenerateSharedGlue(Buffer* sourceBuffer,
                                              std::vector<Buffer*>& destBuffers,
                                              const CascadingBufferFormat cascadingBufferFormat) const;
    StartingAndEndingGlues GenerateGlueBetweenDramAndDram(Buffer* inputBuffer, Buffer* outputBuffer) const;

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

    bool IsSectionSizeSupported(const StatsType sectionInfo, const Plan& plan, uint32_t& totalAgents);
    bool DoAgentsInGraphFitInsideWindow(const OpGraph& opGraph, Buffer* inputBuffer, uint32_t& totalAgents);

private:
    // Add glue to input slots and output slots which do not have glue already
    // This is needed so it can estimate partial combinations
    Combination AddTempGlues(const Combination& combination);
    const GraphOfParts& m_GraphOfParts;
    const HardwareCapabilities& m_Caps;
    const EstimationOptions& m_EstOpt;
    const DebuggingContext& m_DebuggingContext;

    const BasePart* m_FirstPartAfterSort = nullptr;
    std::vector<std::pair<size_t, const BasePart*>> m_PartOrderTable;

    Combination m_BestCombination;
    OpGraph m_MergedOpGraphForBestCombination;
    bool m_MergedOpGraphReady;

    std::vector<StartingAndEndingGlues> m_GluesVector;
    std::unordered_map<const BasePart*, const Combination> m_CombinationPerPartMap;

    std::vector<size_t> m_Stats{ std::vector<size_t>(static_cast<size_t>(StatsType::NumStats), 0) };
};

OpGraph GetOpGraphForCombination(const Combination& combination, const GraphOfParts& parts);

}    // namespace support_library
}    // namespace ethosn
