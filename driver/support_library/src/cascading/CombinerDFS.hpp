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
    using Glues = std::map<const Edge*, const Glue*>;

    PlanId m_PlanId;
    Glues m_Glues;
};

struct Combination
{
    Combination()
    {}

    // Create a combination with a single element without any edge/glue information
    Combination(const Part& part, const Plan& plan)
        : Combination(part, plan, nullptr, nullptr)
    {}

    // Create a combination with a single element without plan information,
    // this is used when updating edge/glue information for a part with
    // multiple outputs where the plan has been already selected and
    // won't be changed when merging combinations
    Combination(const Part& part, const Edge* edge, const Glue* glue)
        : Combination(part, g_InvalidPlanId, edge, glue)
    {}

    // Create a combination with a single element with edge/glue information,
    // if no edge/glue information is provided (e.g. nullptr) the combination
    // will consider the case where no glue is required on any output edge of
    // the part
    Combination(const Part& part, const Plan& plan, const Edge* edge, const Glue* glue)
    {
        // Create a new element
        Elem elem = { plan.m_PlanId, {} };
        // Insert glue value (it can be null if no glue is required)
        // if a valid edge is provided
        if (edge)
        {
            elem.m_Glues.insert(std::make_pair(edge, glue));
        }
        else
        {
            // Consider no glue on all the output edges (i.e. mergeable)
            for (auto& edge : part.GetOutputs())
            {
                elem.m_Glues.insert(std::make_pair(edge, nullptr));
            }
        }
        m_Elems.insert(std::make_pair(part.m_PartId, elem));
    }

    Combination operator+(const Combination& rhs) const
    {
        Combination result = *this;
        for (auto& rhsElemIt : rhs.m_Elems)
        {
            auto resultElemIt = result.m_Elems.find(rhsElemIt.first);
            if (resultElemIt != result.m_Elems.end())
            {
                assert(resultElemIt->second.m_PlanId == rhsElemIt.second.m_PlanId ||
                       rhsElemIt.second.m_PlanId == g_InvalidPlanId);
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

    using Elems          = std::map<PartId, Elem>;
    Combination& operator=(const Combination& c) = default;

    /// Helpers
    /// @{
    size_t GetNumElems() const
    {
        return m_Elems.size();
    }
    /// @}

    Elems m_Elems;
};

enum class StatsType
{
    ContinueSection,
    FindBestCombinationForPart,
    NumStats,
};

using Combinations = std::vector<Combination>;

struct Combiner
{
    Combiner(const GraphOfParts& graphOfParts,
             const HardwareCapabilities& capabilities,
             const EstimationOptions& estOpt,
             const DebuggingContext& debuggingContext);

    bool IsPartInput(const Part& part) const;
    bool IsPartOutput(const Part& part) const;

    bool IsPartSo(const Part& part) const;
    bool IsPartMo(const Part& part) const;
    bool IsPartSiso(const Part& part) const;
    bool IsPartSimo(const Part& part) const;
    bool IsPartMiso(const Part& part) const;
    bool IsPartMimo(const Part& part) const;

    bool AreMceOperationsCompatible(const Buffer* plan1OutputBuffer,
                                    const Buffer* plan2InputBuffer,
                                    const Node* destination) const;

    bool AreBlockConfigsCompatible(const Plan& plan1, const Plan& plan2, const Edge& edge) const;

    bool ArePlansCompatible(const Plan& sPlan, const Plan& dPlan, const Edge& edge);
    bool ArePlansCompatibleImpl(const Plan& sPlan, const Plan& dPlan, const Edge& edge) const;

    bool IsPlanAllocated(SramAllocator& alloc, const Plan& plan) const;
    bool IsPlanInputGlueable(const Plan& plan) const;
    bool ArePlansAllowedToMerge(const Plan& reference, const Plan& current, const Edge& edge) const;

    Combination GetBestCombination() const;
    Combination GetBestCombination(Combinations& combs) const;
    CascadingBufferFormat
        GetBestCascadingBufferDramFormat(const std::array<TensorShape, 2> inputOutputStripeShapes) const;

    const Plan& GetPlanForPartFromCombination(const Part& part, const Combination& comb) const;
    std::vector<std::pair<const Part*, const Edge*>> GetSourceParts(const Part& part) const;
    std::vector<std::pair<const Part*, const Edge*>> GetDestinationParts(const Part& part) const;
    std::pair<bool, const Glue*> GetGlue(const Buffer* outputBuffer, const Buffer* inputBuffer);

    Combination FindBestCombinationForPart(const Part& part);
    Combination FindBestCombinationForPartImpl(const Part& part);

    Combination ContinueSection(const Part& part, const Combination& comb, const SramAllocator& alloc);

    std::unique_ptr<Glue> GenerateGlueBetweenSramAndDram() const;
    std::unique_ptr<Glue> GenerateGlueBetweenSramAndSram(const Buffer* buffer,
                                                         const CascadingBufferFormat cascadingBufferFormat) const;
    Combination GluePartToCombination(const Part& part,
                                      const Combination& comb,
                                      const std::vector<std::pair<const Part*, const Edge*>>& sources);

    void UpdateStats(const StatsType type);

    void Run();

    const GraphOfParts& m_GraphOfParts;
    const HardwareCapabilities& m_Caps;
    const EstimationOptions& m_EstOpt;
    const DebuggingContext& m_DebuggingContext;

    Combination m_BestCombination;

    std::map<const Part*, const Combination> m_CombinationPerPartMap;
    std::vector<std::unique_ptr<Glue>> m_GluesVector;

    std::vector<size_t> m_Stats{ std::vector<size_t>(static_cast<size_t>(StatsType::NumStats), 0) };
};

OpGraph GetOpGraphForCombination(const Combination& combination, const GraphOfParts& parts);

}    // namespace support_library
}    // namespace ethosn