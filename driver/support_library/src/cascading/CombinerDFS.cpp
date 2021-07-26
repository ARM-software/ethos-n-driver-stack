//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "CombinerDFS.hpp"

#include "../SramAllocator.hpp"
#include "../Utils.hpp"
#include "Estimation.hpp"
#include "EstimationUtils.hpp"
#include "Part.hpp"
#include "Plan.hpp"

namespace ethosn
{
namespace support_library
{
namespace depth_first_search
{

using namespace utils;

bool Combiner::IsPartInput(const Part& part) const
{
    return (0 == part.GetInputs().size());
}

bool Combiner::IsPartOutput(const Part& part) const
{
    return (0 == part.GetOutputs().size());
}

bool Combiner::IsPartSo(const Part& part) const
{
    return (part.GetOutputs().size() == 1);
}

bool Combiner::IsPartMo(const Part& part) const
{
    return (part.GetOutputs().size() > 1);
}

bool Combiner::IsPartSiso(const Part& part) const
{
    return (part.GetInputs().size() == 1 && part.GetOutputs().size() == 1);
}

bool Combiner::IsPartSimo(const Part& part) const
{
    return (part.GetInputs().size() == 1 && part.GetOutputs().size() > 1);
}

bool Combiner::IsPartMiso(const Part& part) const
{
    return (part.GetInputs().size() > 1 && part.GetOutputs().size() == 1);
}

bool Combiner::IsPartMimo(const Part& part) const
{
    return (part.GetInputs().size() > 1 && part.GetOutputs().size() > 1);
}

const Part* Combiner::GetNextPart(const Part& part) const
{
    // Sanity check. This function only supports the
    // use cases SISO, input and ouput parts required
    // in this file.
    assert(part.GetInputs().size() <= 1 && part.GetOutputs().size() <= 1);

    // It is an output part
    if (part.GetOutputs().size() == 0)
    {
        return nullptr;
    }

    // output edge of this part
    const Edge* edge = part.GetOutputs().at(0);

    // Find the next part that takes the output edge as its input
    InPart nextPart = m_GraphOfParts.GetInputPart(*edge);

    if (nextPart.first)
    {
        PartId id = nextPart.second;
        return &(m_GraphOfParts.GetPart(id));
    }
    else
    {
        return nullptr;
    }
}

std::vector<const Part*> Combiner::GetDestinationParts(const Part& part)
{
    std::vector<const Part*> result;

    std::vector<const Edge*> outputEdges = part.GetOutputs();

    for (auto& edge : outputEdges)
    {
        InPart nextPart = m_GraphOfParts.GetInputPart(*edge);

        if (nextPart.first)
        {
            PartId id = nextPart.second;
            result.push_back(&(m_GraphOfParts.GetPart(id)));
        }
    }

    return result;
}

bool Combiner::IsPlanMergeableImpl(const Combination& comb, const Part& part, const Plan& plan) const
{
    ETHOSN_UNUSED(comb);
    ETHOSN_UNUSED(part);
    ETHOSN_UNUSED(plan);
    return false;
}

bool Combiner::IsPlanMergeable(const Combination& comb, const Part& part, const Plan& plan)
{
    return IsPlanMergeableImpl(comb, part, plan);
}

bool Combiner::IsPlanAllowedImpl(const Combination& comb, const Part& part, const Plan& plan) const
{
    ETHOSN_UNUSED(comb);
    ETHOSN_UNUSED(part);
    ETHOSN_UNUSED(plan);
    return false;
}

bool Combiner::IsPlanAllowed(const Combination& comb, const Part& part, const Plan& plan)
{
    return IsPlanAllowedImpl(comb, part, plan);
}

bool Combiner::IsPlanAllocated(SramAllocator& alloc, const Combination& comb, const Part& part, const Plan& plan)
{
    ETHOSN_UNUSED(alloc);
    ETHOSN_UNUSED(comb);
    ETHOSN_UNUSED(part);
    ETHOSN_UNUSED(plan);
    return false;
}

Combination Combiner::GetBestCombination(Combinations& combs) const
{
    if (combs.size() > 0)
    {
        utils::Optional<Combination> result;
        NetworkPerformanceData refNetPerfData;

        for (const Combination& combination : combs)
        {
            if (!combination.m_Elems.empty())
            {
                OpGraph combiOpGraph = GetOpGraphForCombination(combination, m_GraphOfParts);

                EstimatedOpGraph estimatedOpGraph =
                    ethosn::support_library::EstimateOpGraph(combiOpGraph, m_Caps, m_EstOpt);

                if (!estimatedOpGraph.IsComplete())
                {
                    continue;
                }

                if (!result.has_value() || ComparePerformanceData(estimatedOpGraph.m_PerfData, refNetPerfData) ==
                                               PerformanceComparisonResult::LeftBetter)
                {
                    refNetPerfData = estimatedOpGraph.m_PerfData;
                    result         = combination;
                }
            }
        }

        if (!result.has_value())
        {
            // If Estimation failed, pick the first combination
            return combs.front();
        }
        return result.value();
    }

    return Combination{};
}

Combination Combiner::GetBestCombination() const
{
    return m_BestCombination;
}

// Try to merge plans from the given Part onto the given Combination.
// This may not happen because:
//  - Plan cannot be merged e.g. different strategies
//  - Plan is not allowed
//  - Plan buffers do not fit in SRAM i.e. merged plans
//    in the seciton take up all the memory
Combination Combiner::ContinueSection(const Part& part, const Combination& comb, const SramAllocator& alloc)
{
    // End the current section and start a new one.
    // There is a single edge between the combination comb and
    // and the current part
    Combination result = comb + FindBestCombinationForPart(part);

    if (IsPartSiso(part))
    {
        // SISO part:
        //
        // Try to continue this section with next part.
        // Make sure that the chosen next plan is in the order:
        //  - Compatible with the last plan in the section
        //  - Allowed i.e. some restriction could be applied
        //    to reduce the search space, for example it
        //    could consider only plans that have identical
        //    block configurations etc.
        //  - Allocated i.e. there is space in SRAM to accomodate
        //    all the buffers required by the plan

        const Part& nextPart = *GetNextPart(part);
        for (const auto& plan : part.m_Plans)
        {
            // Make a copy of the allocator since every plan needs to have its own,
            // each potential section won't allocate from the same allocator.
            SramAllocator tempAlloc = alloc;

            if (!IsPlanMergeable(comb, part, *plan.get()))
            {
                continue;
            }

            if (!IsPlanAllowed(comb, part, *plan.get()))
            {
                continue;
            }

            if (!IsPlanAllocated(tempAlloc, comb, part, *plan.get()))
            {
                continue;
            }

            // Add current part and plan to the combination,
            // no glue is required. Current part is SISO and
            // has a single input/output
            Combination section = comb + Combination(part, *plan.get());
            // Options to be estimated
            Combinations options = { result, ContinueSection(nextPart, section, tempAlloc) };
            result               = GetBestCombination(options);
        }
    }
    return result;
}

// This function finds the best combination from the current part
// to the end of the graph. The resul is unique given the part.
//
// The retuned value of this function should be cached
//
//      PART       ||    COMBINATION
//  ===================================
//      partA      ||    CombinationX
//  -----------------------------------
//      partB      ||    CombinationY
//  -----------------------------------
//       ...       ||         ...
//  -----------------------------------
//      partN      ||    CombinationW
//  -----------------------------------
//
Combination Combiner::FindBestCombinationForPartImpl(const Part& part)
{
    // This is going to be a new combination, so this
    // is empty initialized
    Combination result = {};
    // There are some scenarios:
    //  - Part is Single Input Single Output i.e. SISO
    //  - Part is Single Input Multiple Output i.e. SIMO
    //  - Part is Multiple Input Multiple Output i.e. MIMO
    //  - Part is Multiple Input Sinlge Output i.e. MISO
    //  - Part is Output i.e. no next part
    //  - Part is Input i.e. SO or MO
    if (IsPartSo(part))
    {
        // SISO and MISO are equivalent since what counts
        // is the number of output parts which in both cases
        // is one
        const Part& nextPart = *GetNextPart(part);
        for (const auto& plan : part.m_Plans)
        {
            // This is the start of a new section, reset the allocated Sram
            SramAllocator alloc(m_Caps.GetTotalSramSize() / m_Caps.GetNumberOfSrams());
            Combination head(part, *plan.get());
            Combinations options = { result, ContinueSection(nextPart, head, alloc) };
            result               = GetBestCombination(options);
        }
    }
    else
    {
        // ContinueSection operates only on SISO parts
        // so Output parts and Multiple Output parts
        // cannot be merged for now

        // Select best plan for the part
        for (const auto& plan : part.m_Plans)
        {
            // Glue will be added later on
            Combination head(part, *plan.get());
            Combinations options = { result, head };
            result               = GetBestCombination(options);
        }

        // SIMO part:
        //
        // It cannot create a section, it needs to start as
        // many new sections as the number of output parts
        //
        // MIMO part:
        //
        // This part is a lonely one, it needs to start
        // as many new sections as the number of output parts
        // Some of the ongoing sections might not be ended, the
        // recursion goes depth first and does not walk the parts
        // necessarily in a topological order that allows to end
        // all the input sections to a MIMO/MISO part. For exmaple
        // the input edge into a MISO part might come from a differnt
        // input of the whole graph. This should not be a concern

        for (const auto& destPart : GetDestinationParts(part))
        {
            // Glue needs to be added here for each destination
            result = result + FindBestCombinationForPart(*destPart);
        }
    }
    return result;
}

// TODO: This implement a caching mechanism on part
//       PartId -> Best Combination
//
//      PART       ||    COMBINATION
//  ===================================
//      partA      ||    CombinationX
//  -----------------------------------
//      partB      ||    CombinationY
//  -----------------------------------
//       ...       ||         ...
//  -----------------------------------
//      partN      ||    CombinationW
//  -----------------------------------
//
Combination Combiner::FindBestCombinationForPart(const Part& part)
{
    Combination result;
    auto combIt = m_CombinationPerPartMap.find(&part);
    if (combIt != m_CombinationPerPartMap.end())
    {
        result = combIt->second;
    }
    else
    {
        result = FindBestCombinationForPartImpl(part);
        m_CombinationPerPartMap.insert(std::make_pair(&part, result));
    }
    return result;
}

Combiner::Combiner(const GraphOfParts& graphOfParts, const HardwareCapabilities& caps, const EstimationOptions& estOpt)
    : m_GraphOfParts(graphOfParts)
    , m_Caps(caps)
    , m_EstOpt(estOpt)
{}

void Combiner::Run()
{
    for (auto&& part : m_GraphOfParts.m_Parts)
    {
        // Process only parts that have an input node
        if (!IsPartInput(*part.get()))
        {
            continue;
        }
        // Result combinations (each per input) can just be merged
        m_BestCombination = m_BestCombination + FindBestCombinationForPart(*part.get());
    }
}

// Take in input a combination and generate an OpGraph.
// This is used in:
//  - Combiner logic:   it needs to estimate the combination and this is done on an
//                      OpGraph in order to select the best combination between two
//                      or more
//  - Estimation logic: it can only estimate OpGraphs and not raw combinations.
OpGraph GetOpGraphForCombination(const Combination& combination, const GraphOfParts& parts)
{
    OpGraph result;

    // When adjacent plans are connected without any glue, the output buffer of one plan becomes the input buffer of the
    // next plan. In the merged graph representation that we are creating, we therefore need only one buffer object.
    // This map is used to get the buffer that we are using to represent two buffers that have been merged.
    std::map<Buffer*, Buffer*> mergedBuffers;
    auto getEffectiveBuffer = [&mergedBuffers](Buffer* b) {
        auto it = mergedBuffers.find(b);
        return it != mergedBuffers.end() ? it->second : b;
    };

    // For each Edge connecting two Parts, which Buffer should the destination part connect to, in order to get that input.
    // A glue may also need to be inserted which connects to this buffer.
    // If there is no glue between two parts, then the source
    // part's output buffer should be re-used directly (as that buffer is then shared between the two plans).
    std::map<const Edge*, Buffer*> edgeConnectionBuffers;

    // For each outgoing edge from a plan, the glue that needs to be inserted there (if any)
    std::map<const Edge*, const Glue*> glues;

    // Add each Elem, one at a time. It is assumed that these are toplogically sorted, so we can assume that all
    // parts used as input to each part have already been processed.
    for (auto& elemIt : combination.m_Elems)
    {
        const Part& part = parts.GetPart(elemIt.first);
        const Plan& plan = part.GetPlan(elemIt.second.m_PlanId);

        // Add any glues for each incoming edge of this Part, and remember which Op we will need to connect the plan's
        // input buffers to
        std::map<const Edge*, Op*> incomingGlueOps;
        std::vector<const Edge*> inputEdges = part.GetInputs();
        for (auto inputEdge : inputEdges)
        {
            auto glueIt      = glues.find(inputEdge);
            const Glue* glue = glueIt != glues.end() ? glueIt->second : nullptr;
            if (glue != nullptr)
            {
                // Add Ops and Buffers from the glue, no connections yet.
                for (Buffer* b : glue->m_Graph.GetBuffers())
                {
                    result.AddBuffer(b);
                }
                for (Op* o : glue->m_Graph.GetOps())
                {
                    result.AddOp(o);
                }

                // Add internal connections within the glue
                for (Buffer* b : glue->m_Graph.GetBuffers())
                {
                    Op* producer = glue->m_Graph.GetProducer(b);
                    if (producer)
                    {
                        result.SetProducer(b, producer);
                    }

                    for (auto consumer : glue->m_Graph.GetConsumers(b))
                    {
                        result.AddConsumer(b, consumer.first, consumer.second);
                    }
                }

                // Connect to the input plan
                result.AddConsumer(edgeConnectionBuffers.at(inputEdge), glue->m_InputSlot.first,
                                   glue->m_InputSlot.second);

                // Remember the output Op from this glue, to connect to our plan
                incomingGlueOps[inputEdge] = glue->m_Output;
            }
        }
        for (Buffer* b : plan.m_OpGraph.GetBuffers())
        {
            // Don't add a buffer if its an input to the plan, and it is shared with the input plan
            // (i.e. no glue between them).
            // Instead, remap it to the one we already have
            Buffer* sharedBuffer = nullptr;
            auto inputEdgeIt     = plan.m_InputMappings.find(b);
            if (inputEdgeIt != plan.m_InputMappings.end())
            {
                Edge* inputEdge = inputEdgeIt->second;
                if (incomingGlueOps.find(inputEdge) == incomingGlueOps.end())
                {
                    sharedBuffer = edgeConnectionBuffers.find(inputEdge)->second;
                    // This buffer itself may have been merged (e.g. for plans that have a single buffer for both
                    // input and output, like reinterpret Dram)
                    sharedBuffer = getEffectiveBuffer(sharedBuffer);
                }
            }
            if (sharedBuffer)
            {
                assert(result.Contains(sharedBuffer));
                // Record the fact that this buffer has been shared, so that when making connections (below), we
                // connect to the correct buffer.
                mergedBuffers[b] = sharedBuffer;
            }
            else
            {
                result.AddBuffer(b);
            }
        }
        // Add Ops from the Plan
        for (Op* o : plan.m_OpGraph.GetOps())
        {
            result.AddOp(o);
        }

        // Add internal connections (within the Plan), noting that some buffers will have been merged and
        // that we need to make the connection to the correct one.
        for (Buffer* b : plan.m_OpGraph.GetBuffers())
        {
            Op* producer = plan.m_OpGraph.GetProducer(b);
            if (producer)
            {
                result.SetProducer(getEffectiveBuffer(b), producer);
            }

            for (auto consumer : plan.m_OpGraph.GetConsumers(b))
            {
                result.AddConsumer(getEffectiveBuffer(b), consumer.first, consumer.second);
            }
        }

        // Connect this Plan's inputs to the glues we take input from.
        // If we are instead connected to a plan directly (without any glue), then nothing needs to be done
        // because our input buffer will have been replaced by the output buffer from that plan,
        // so we are already connected
        for (auto input : plan.m_InputMappings)
        {
            Buffer* ourBuffer = input.first;
            Edge* inputEdge   = input.second;
            auto glueOpIt     = incomingGlueOps.find(inputEdge);
            if (glueOpIt != incomingGlueOps.end())
            {
                result.SetProducer(ourBuffer, glueOpIt->second);
            }
        }

        // Store our output connections for future plans, and any glues on our outputs
        for (auto output : plan.m_OutputMappings)
        {
            for (Edge* outputEdge : output.second->GetOutputs())
            {
                edgeConnectionBuffers[outputEdge] = output.first;
                auto glueIt                       = elemIt.second.m_Glues.find(outputEdge);
                if (glueIt != elemIt.second.m_Glues.end() && !glueIt->second->m_Graph.GetOps().empty())
                {
                    glues[outputEdge] = glueIt->second;
                }
            }
        }
    }

    return result;
}

}    // namespace depth_first_search
}    // namespace support_library
}    // namespace ethosn
