//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "DebuggingContext.hpp"
#include "Estimation.hpp"
#include "GraphOfParts.hpp"
#include "Part.hpp"
#include "Plan.hpp"
#include "SramAllocator.hpp"

#include <set>
#include <unordered_map>

namespace ethosn
{
namespace support_library
{

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

using PleOperations = std::vector<std::pair<command_stream::PleKernelId, uint32_t>>;

/// A Combination stores which Plans have been chosen for a set of Parts.
/// It also stores Glues which connect adjacent Plans to each other.
/// The Parts that it stores Plans for must have contiguous IDs (e.g. Parts 1, 2 and 3).
class Combination
{
public:
    /// Creates an empty/invalid Combination, which contains no chosen Plans.
    Combination();

    /// Creates a Combination storing a single Part with an associated Plan.
    /// No Glues are needed, as there is only a single Plan.
    Combination(PartId partId, Plan&& plan);

    /// Combines this Combination and another into a new Combination, containing the chosen Plans
    /// and glues from each. The `rhs` must contain Parts that continue the contiguous ID numbering
    /// from the current Combination, e.g. { 1, 2, 3 } + { 4, 5 } is valid, but { 1, 2, 3 } + { 5, 6 } is not.
    Combination operator+(const Combination& rhs) const;

    /// Sets the ending or starting Glue for a given Part in this Combination.
    /// This can only be done once - a Glue can't be changed once set.
    /// @{
    void SetEndingGlue(EndingGlue&& glue, PartOutputSlot outputSlot);
    void SetStartingGlue(StartingGlue&& glue, PartInputSlot inputSlot);
    /// @}

    bool IsEmpty() const;

    /// Gets the first/last Part ID which this Combination is storing a Plan for.
    /// All Parts inbetween these will also have a Plan stored, because we always store a contiguous range.
    /// @{
    PartId GetFirstPartId() const;
    PartId GetEndPartId() const;
    /// @}

    Elem& GetElem(PartId partId);
    const Elem& GetElem(PartId partId) const;

    double GetMetric() const;
    void SetMetric(double metric);

private:
    /// The ID of the first Part that we're storing a Plan for.
    PartId m_PartIdOffset;
    /// The Plans and Glues for each Part in the contiguous range of Parts that we're storing.
    std::vector<Elem> m_Elems;
    /// The combined estimated performance metric for the set of Plans that we're storing.
    double m_Metric;
};

/// Information about a partially-complete section, which is created in StartSection and through
/// ContinueSection(s) and finally into EndSection.
struct SectionContext
{
    /// All the plans chosen so far.
    Combination comb;
    /// Tracks which parts of SRAM are in use by buffers that need to be kept alive.
    SramAllocator alloc;
    /// Tracks which PLE kernels have already been loaded into SRAM.
    PleOperations pleOps;
    /// Tracks which buffers have live allocations in `alloc`, along with a list of which parts
    /// have ownership of the buffer (ownership is passed between parts as we progress through the section).
    std::unordered_map<SramBuffer*, std::set<PartId>> allocatedBuffers;
    /// Whether or not we are double-buffering weight stripes.
    uint32_t currNumWeightStripes;
    bool hasSectionDoubleBuffered;
    /// @}
    /// When partway through a section, we might have several Parts whose outputs haven't yet
    /// been processed. These are tracked here. This should be empty once the section is finished.
    std::unordered_map<PartConnection, Buffer*> unresolvedOutputs;
    /// Which block config to use for this section (we use the same block config for the whole section).
    BlockConfig blockConfig;
};

using Combinations = std::vector<Combination>;

class Combiner
{
public:
    Combiner(const FrozenGraphOfParts& graphOfParts,
             const HardwareCapabilities& capabilities,
             const CompilationOptions& compilationOptions,
             const EstimationOptions& estOpt,
             const DebuggingContext& debuggingContext);

    const Combination& GetBestCombination() const;
    OpGraph GetMergedOpGraphForBestCombination() const;

    void Run(ThreadPool& threadPool);

protected:
    bool AllocateSram(SectionContext& context,
                      PartId partId,
                      const Plan& plan,
                      const std::vector<Buffer*>& outputBuffersOfPrevPlan) const;
    void DeallocateUnusedBuffers(PartId partId,
                                 const PartOutputMapping& planOutputBuffers,
                                 const std::vector<PartId>& consumingPartIds,
                                 SectionContext& context);

    Combination GluePartToCombinationSrcToDests(const BasePart& sPart, const Combination& comb, uint32_t outputSlotIdx);

private:
    struct BestCombinationResults
    {
        size_t m_BestIdx;
        double m_BestMetric;
        /// Only used for debugging
        /// @{
        std::vector<Combination> m_CompletedCombinations;
        std::vector<OpGraph> m_OpGraphs;
        std::vector<EstimatedOpGraph> m_EstimatedOpGraphs;
        /// @}
    };
    BestCombinationResults EstimateAndChooseBestCombination(const Combinations& combs) const;

    struct EstimatedCombination
    {
        Combination m_CombinationWithTempGlues;
        OpGraph m_OpGraph;
        EstimatedOpGraph m_EstimatedOpGraph;
    };
    EstimatedCombination EstimateCombination(const Combination& comb) const;

    Combination ChooseBestLonelyPlan(const BasePart& part);

    std::vector<SectionContext> StartSection(const BasePart& part);
    std::vector<SectionContext> ContinueSection(const BasePart& part, const SectionContext& context);
    std::vector<SectionContext> EndSection(const BasePart& part, const SectionContext& context);

    std::vector<SectionContext> ContinueOrEndSection(bool isEnd, const BasePart& part, const SectionContext& context);

    std::vector<Combination> CalculateSectionsOfAllLengths(const BasePart& startingPart);

    // Add glue to input slots and output slots which do not have glue already
    // This is needed so it can estimate partial combinations
    Combination AddTempGlues(const Combination& combination) const;

    void DumpDebugInfo(const Combinations& combs,
                       const Combiner::BestCombinationResults& bestCombinationResults,
                       const std::string& folder);

    const FrozenGraphOfParts& m_GraphOfParts;
    const HardwareCapabilities& m_Caps;
    const CompilationOptions& m_CompilationOptions;
    const EstimationOptions& m_EstOpt;
    const DebuggingContext& m_DebuggingContext;

    Combination m_BestCombination;
    OpGraph m_MergedOpGraphForBestCombination;
};

OpGraph GetOpGraphForCombination(const Combination& combination, const FrozenGraphOfParts& parts);

}    // namespace support_library
}    // namespace ethosn
