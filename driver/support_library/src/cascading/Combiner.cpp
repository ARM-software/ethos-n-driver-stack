//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "Combiner.hpp"

#include "../SramAllocator.hpp"
#include "Cascading.hpp"
#include "Part.hpp"
#include "Plan.hpp"

namespace ethosn
{
namespace support_library
{

namespace
{

bool IsOutputBufferInDram(const Plan& plan, const Edge& edge)
{
    const Buffer* buf = plan.GetOutputBuffer(edge.GetSource());
    return (buf == nullptr) ? true : ((buf->m_Location) == Location::Dram);
}

bool IsOutputBufferAtomic(const Plan& plan, const Edge& edge)
{
    const Buffer* buf = plan.GetOutputBuffer(edge.GetSource());
    return (buf == nullptr) ? true : ((buf->m_Lifetime) == Lifetime::Atomic);
}

struct SizeInBytes
{
    uint32_t m_Tot       = 0;
    uint32_t m_TotAtomic = 0;
};

SizeInBytes GetTotSizeInBytes(const Plan& plan)
{
    SizeInBytes result;
    const OpGraph::BufferList& bufs        = plan.m_OpGraph.GetBuffers();
    OpGraph::BufferList::const_iterator it = bufs.begin();
    while (it != bufs.end())
    {
        const Buffer* buf   = *it;
        const uint32_t size = buf->m_SizeInBytes;
        result.m_Tot += size;
        if (buf->m_Lifetime == Lifetime::Atomic)
        {
            result.m_TotAtomic += size;
        }
        ++it;
    }
    assert(result.m_TotAtomic <= result.m_Tot);
    return result;
}

SizeInBytes GetInputsSizeInBytes(const Plan& plan)
{
    SizeInBytes result;
    const Plan::InputMapping in           = plan.m_InputMappings;
    Plan::InputMapping::const_iterator it = in.begin();
    while (it != in.end())
    {
        const Buffer* buf   = it->first;
        const uint32_t size = buf->m_SizeInBytes;
        result.m_Tot += size;
        if (buf->m_Lifetime == Lifetime::Atomic)
        {
            result.m_TotAtomic += size;
        }
        ++it;
    }
    assert(result.m_TotAtomic <= result.m_Tot);
    return result;
}

using Allocated = std::pair<bool, uint32_t>;

struct AddedSeed
{
    bool m_Added;
    uint32_t m_MinSizeInBytes;
    Combination m_Combination;
};

AddedSeed AddSeed(const PlanId fPlId,
                  const PlanId sPlId,
                  const PartId fPaId,
                  const Edge* sEdge,
                  const Part& part,
                  const Glue* glue,
                  const Combination& comb,
                  const uint32_t baseSizeInBytes,
                  const uint32_t minSizeInBytes,
                  SramAllocator& alloc,
                  const HardwareCapabilities& caps,
                  const bool canMerge)
{
    Allocated allocated;
    Combination result            = comb;
    Scratch::Indexes::iterator it = result.m_Scratch.m_Idx.find(fPaId);

    alloc.Reset();

    const bool update = (it != result.m_Scratch.m_Idx.end());

    const Plan& sPl            = part.GetPlan(sPlId);
    const SizeInBytes sTotSize = GetTotSizeInBytes(sPl);
    const SizeInBytes sInSize  = GetInputsSizeInBytes(sPl);
    assert(sTotSize.m_Tot >= sInSize.m_Tot);
    const uint32_t addSizeInBytes = canMerge ? (baseSizeInBytes + sTotSize.m_Tot - sInSize.m_Tot) : sTotSize.m_Tot;

    allocated = alloc.Allocate(addSizeInBytes / caps.GetNumberOfSrams(), AllocationPreference::Start);

    if (!allocated.first)
    {
        // There is no space
        return AddedSeed{ false, std::numeric_limits<uint32_t>::max(), Combination{} };
    }

    const uint32_t allocatedSram = addSizeInBytes - sTotSize.m_TotAtomic;
    const bool isMinSize         = (allocatedSram <= minSizeInBytes);

    if (canMerge && !isMinSize)
    {
        return AddedSeed{ false, std::numeric_limits<uint32_t>::max(), Combination{} };
    }

    if (update)
    {
        Elem& el = result.m_Elems.at(it->second);
        el.m_Glues.insert(std::make_pair(sEdge, Elem::Link{ sPlId, glue }));
    }
    else
    {
        Elem el;
        el.m_PartId = fPaId;
        el.m_PlanId = fPlId;
        el.m_Glues.insert(std::make_pair(sEdge, Elem::Link{ sPlId, glue }));
        result.m_Elems.push_back(el);
    }

    result.m_Scratch.m_AllocatedSram = allocatedSram;

    if (!update)
    {
        // Update the current combination indexes
        result.m_Scratch.m_Idx.insert(
            std::make_pair(fPaId, (result.GetNumElems() > 0) ? (result.GetNumElems() - 1U) : 0));
    }

    if (canMerge)
    {
        ++result.m_Scratch.m_Score;
    }

    return AddedSeed{ true, (canMerge ? std::min(minSizeInBytes, allocatedSram) : minSizeInBytes), result };
}

struct NxtPa
{
    bool m_Found = false;
    const Edge* m_Dst;
    Combination m_Comb;
};

NxtPa GetNxtPart(const PartId id, const size_t max, const Combination& comb, const Metadata& metadata)
{
    NxtPa result;

    result.m_Comb = comb;

    if (id >= max)
    {
        return result;
    }

    const MetadataOfPart& mOfPa = metadata.at(id);

    if (mOfPa.m_Destination.size() == 0)
    {
        result.m_Dst   = nullptr;
        result.m_Found = false;
        result.m_Comb.m_Scratch.m_Edges.insert(std::make_pair(id, Scratch::Dst{}));
        result.m_Comb.m_Scratch.m_CurrPartId = id + 1U;
        return result;
    }

    // It assumes that the parts are in topological sort
    // Find the destination with the lowest numerical PartId that we haven't already done - this will give deterministic results
    std::pair<const Edge*, PartId> lowestDstPart(nullptr, std::numeric_limits<PartId>::max());
    for (const auto& it : mOfPa.m_Destination)
    {
        const Edge* edge             = it.first;
        Scratch::Edges::iterator eIt = result.m_Comb.m_Scratch.m_Edges.find(id);
        const bool saved             = (eIt != result.m_Comb.m_Scratch.m_Edges.end());
        Scratch::Dst dst             = saved ? (eIt->second) : Scratch::Dst{};
        const bool done              = (std::find(std::begin(dst), std::end(dst), edge) != std::end(dst));
        if (!done && it.second < lowestDstPart.second)
        {
            lowestDstPart.first  = it.first;
            lowestDstPart.second = it.second;
        }
    }
    if (lowestDstPart.second == std::numeric_limits<PartId>::max())
    {
        return result;
    }

    const Edge* edge             = lowestDstPart.first;
    Scratch::Edges::iterator eIt = result.m_Comb.m_Scratch.m_Edges.find(id);
    const bool saved             = (eIt != result.m_Comb.m_Scratch.m_Edges.end());
    if (saved)
    {
        (eIt->second).push_back(edge);
    }
    else
    {
        result.m_Comb.m_Scratch.m_Edges.insert(std::make_pair(id, Scratch::Dst{ { edge } }));
    }
    const bool last = (mOfPa.m_Destination.size() == (result.m_Comb.m_Scratch.m_Edges.find(id)->second).size());
    result.m_Dst    = edge;
    result.m_Found  = true;
    result.m_Comb.m_Scratch.m_CurrPartId = last ? (id + 1U) : id;
    return result;
}

struct PlanFromSource
{
    bool m_Found = false;
    PlanId m_Id;
};

PlanFromSource GetPlanFromSource(PartId id, const Combination& comb, const Metadata& metadata)
{
    PlanFromSource result;

    const MetadataOfPart& mOfPa = metadata.at(id);

    if (mOfPa.m_Source.size() == 0)
    {
        return result;
    }

    SrcPart::const_iterator it = mOfPa.m_Source.begin();
    while (it != mOfPa.m_Source.end())
    {
        const PartId src                       = it->second;
        Scratch::Indexes::const_iterator idxIt = comb.m_Scratch.m_Idx.find(src);
        if (idxIt != comb.m_Scratch.m_Idx.end())
        {
            size_t combIdx                   = idxIt->first;
            const Elem& el                   = comb.m_Elems.at(combIdx);
            Elem::Glues::const_iterator glIt = el.m_Glues.find(it->first);
            if (glIt != el.m_Glues.end())
            {
                result.m_Id    = (glIt->second).m_Id;
                result.m_Found = true;
                return result;
            }
        }
        ++it;
    }
    return result;
}

constexpr PlanId unassignedPlanId = static_cast<PlanId>(-1);

Combinations CombineSeeds(const PlanId fPlId,
                          const CompatiblePlans& fComPls,
                          const Combination& comb,
                          const PartId fPartId,
                          const Edge* sEdge,
                          const GraphOfParts& parts,
                          const PlanId reqPlan,
                          const Metadata& metadata,
                          SramAllocator& alloc,
                          const HardwareCapabilities& caps,
                          const bool create = true)
{
    Combinations result;

    // First part
    const Part& fPart            = parts.GetPart(fPartId);
    const MetadataOfPart& fMOfPa = metadata.at(fPartId);

    // Second part
    DstPart::const_iterator eIt = fMOfPa.m_Destination.find(sEdge);
    assert(eIt != fMOfPa.m_Destination.end());
    const Part& sPart = parts.GetPart(eIt->second);

    const bool checkReqPlan = (reqPlan != unassignedPlanId);

    const Plan& fPl      = fPart.GetPlan(fPlId);
    const bool outInDram = IsOutputBufferInDram(fPl, *sEdge);
    const bool outAtomic = IsOutputBufferAtomic(fPl, *sEdge);

    const SizeInBytes fTotSize     = GetTotSizeInBytes(fPl);
    const uint32_t baseSizeInBytes = create ? (fTotSize.m_Tot - fTotSize.m_TotAtomic) : comb.m_Scratch.m_AllocatedSram;

    uint32_t minSizeInBytes = std::numeric_limits<uint32_t>::max();

    // Process all the list of compatible plans
    for (uint32_t i = 0; i < fComPls.size(); ++i)
    {
        const CompatiblePlan& fComPl = fComPls.at(i);

        if (checkReqPlan && (fComPl.m_Id == reqPlan))
        {
            continue;
        }

        const bool hasGlue = (fComPl.m_Glue.m_Graph.GetOps().size() > 0);

        const bool canMerge = !hasGlue && !outInDram && !outAtomic;

        AddedSeed addedSeed = AddSeed(fPlId, fComPl.m_Id, fPartId, sEdge, sPart, &fComPl.m_Glue, comb, baseSizeInBytes,
                                      minSizeInBytes, alloc, caps, canMerge);

        if (addedSeed.m_Added)
        {
            // Add seed
            result.push_back(addedSeed.m_Combination);
            // Update min size in bytes
            minSizeInBytes = addedSeed.m_MinSizeInBytes;
        }
    }

    return result;
}

}    // namespace

PlanCompatibilityResult ArePlansCompatible(const Plan& plan1, const Plan& plan2, const Edge& edge)
{
    // Sanity tests - make sure the two Plans are for adjacent Parts.
    // Note we lookup both buffers by the same Node, as the Graph does not explicitly store intermediate tensors -
    // they are implicitly attached to each Node (which are defined to have a single output).
    const Buffer* plan1OutputBuffer = plan1.GetOutputBuffer(edge.GetSource());
    const Buffer* plan2InputBuffer  = plan2.GetInputBuffer(&edge);

    if (plan1OutputBuffer == nullptr || plan2InputBuffer == nullptr)
    {
        // Not compatible as these two plans aren't connected along the edge we were told about.
        PlanCompatibilityResult result;
        result.m_IsCompatible = false;
        return result;
    }

    // Some properties of the buffers must match, for example if the quantisation info is different then even inserting
    // a DMA glue isn't going to help.
    bool areBuffersCompatible = plan1OutputBuffer->m_QuantizationInfo == plan2InputBuffer->m_QuantizationInfo &&
                                plan1OutputBuffer->m_TensorShape == plan2InputBuffer->m_TensorShape;
    if (!areBuffersCompatible)
    {
        // Not compatible as the output buffer can't be used directly as the input buffer, and we can't convert
        // between them using a glue (at least not with the current implementation of this function).
        PlanCompatibilityResult result;
        result.m_IsCompatible = false;
        return result;
    }

    // Check if the buffers on the boundary are compatible, i.e. the same, such that the plans could be directly merged
    // without any additional DMA ops required
    bool areBuffersEquivalent = plan1OutputBuffer->m_Location == plan2InputBuffer->m_Location &&
                                plan1OutputBuffer->m_Format == plan2InputBuffer->m_Format &&
                                plan1OutputBuffer->m_StripeShape == plan2InputBuffer->m_StripeShape &&
                                plan1OutputBuffer->m_Order == plan2InputBuffer->m_Order &&
                                plan1OutputBuffer->m_SizeInBytes == plan2InputBuffer->m_SizeInBytes;
    if (areBuffersEquivalent)
    {
        PlanCompatibilityResult result;
        result.m_IsCompatible = true;
        result.m_RequiresGlue = false;
        return result;
    }

    // One buffer may be in SRAM and the other in DRAM, in which case we can insert a single DMA op
    if (plan1OutputBuffer->m_Location == Location::Sram && plan2InputBuffer->m_Location == Location::Dram)
    {
        PlanCompatibilityResult result;
        result.m_IsCompatible = true;
        result.m_RequiresGlue = true;

        auto dma      = std::make_unique<DmaOp>();
        DmaOp* dmaRaw = dma.get();
        result.m_Glue.m_Graph.AddOp(std::move(dma));
        result.m_Glue.m_InputSlot = { dmaRaw, 0 };
        result.m_Glue.m_Output    = dmaRaw;

        return result;
    }
    else if (plan1OutputBuffer->m_Location == Location::Dram && plan2InputBuffer->m_Location == Location::Sram)
    {
        PlanCompatibilityResult result;
        result.m_IsCompatible = true;
        result.m_RequiresGlue = true;

        auto dma      = std::make_unique<DmaOp>();
        DmaOp* dmaRaw = dma.get();
        result.m_Glue.m_Graph.AddOp(std::move(dma));
        result.m_Glue.m_InputSlot = { dmaRaw, 0 };
        result.m_Glue.m_Output    = dmaRaw;

        return result;
    }

    // If both buffers are in SRAM (but not equivalent, as checked above), we can DMA out to DRAM and back in again.
    else if (plan1OutputBuffer->m_Location == Location::Sram && plan2InputBuffer->m_Location == Location::Sram)
    {
        PlanCompatibilityResult result;
        result.m_IsCompatible = true;
        result.m_RequiresGlue = true;

        auto dma1       = std::make_unique<DmaOp>();
        DmaOp* dma1Raw  = dma1.get();
        auto dramBuffer = std::make_unique<Buffer>(
            Lifetime::Atomic, Location::Dram, CompilerDataFormat::NHWCB, plan1OutputBuffer->m_TensorShape,
            TensorShape{ 0, 0, 0, 0 }, TraversalOrder::Xyz,
            utils::TotalSizeBytesNHWCB(plan1OutputBuffer->m_TensorShape), plan1OutputBuffer->m_QuantizationInfo);
        Buffer* dramBufferRaw = dramBuffer.get();
        auto dma2             = std::make_unique<DmaOp>();
        DmaOp* dma2Raw        = dma2.get();
        result.m_Glue.m_Graph.AddOp(std::move(dma1));
        result.m_Glue.m_Graph.AddOp(std::move(dma2));
        result.m_Glue.m_Graph.AddBuffer(std::move(dramBuffer));
        result.m_Glue.m_Graph.SetProducer(dramBufferRaw, dma1Raw);
        result.m_Glue.m_Graph.AddConsumer(dramBufferRaw, dma2Raw, 0);
        result.m_Glue.m_InputSlot = { dma1Raw, 0 };
        result.m_Glue.m_Output    = dma2Raw;

        return result;
    }

    return PlanCompatibilityResult{};
}

Metadata CreateMetadata(const GraphOfParts& parts)
{
    const size_t numParts = parts.GetNumParts();
    assert(numParts > 1U);

    Metadata result;

    MetadataOfPart mOfPa;
    CompatiblePlansOfPart comPlsOfPa;
    CompatiblePlans cPls;

    for (PartId p = 0; p < numParts; ++p)
    {
        mOfPa.m_Comp.clear();
        mOfPa.m_Source.clear();
        mOfPa.m_Destination.clear();
        comPlsOfPa.clear();

        const Part& fPart                      = parts.GetPart(p);
        mOfPa.m_PartId                         = p;
        const std::vector<const Edge*> dsEdges = fPart.GetOutputs();
        for (uint32_t n = 0; n < dsEdges.size(); ++n)
        {
            const Edge* dsEdge = dsEdges.at(n);

            const InPart inPa = parts.GetInputPart(*dsEdge);
            assert(inPa.first == true);

            const Part& sPart = parts.GetPart(inPa.second);
            // Requires Dram if the part is not directly connected with next part
            // in topological sort or if the part has multiple outputs or it the next
            // part has multiple inputs
            const bool reqDram = ((p + 1U) != inPa.second) || (dsEdges.size() > 1U) || (sPart.GetInputs().size() > 1U);

            for (uint32_t f = 0; f < fPart.GetNumPlans(); ++f)
            {
                cPls.clear();

                const Plan& fPl = fPart.GetPlan(f);

                for (uint32_t s = 0; s < sPart.GetNumPlans(); ++s)
                {
                    const Plan& sPl                   = sPart.GetPlan(s);
                    PlanCompatibilityResult plCompRes = ArePlansCompatible(fPl, sPl, *dsEdge);
                    if (plCompRes.m_IsCompatible)
                    {
                        if (reqDram && !IsOutputBufferInDram(fPl, *dsEdge) && !plCompRes.m_RequiresGlue)
                        {
                            continue;
                        }
                        cPls.push_back(CompatiblePlan{ std::move(plCompRes.m_Glue), s });
                    }
                }
                if (cPls.size() > 0)
                {
                    comPlsOfPa.insert(std::make_pair(f, std::move(cPls)));
                }
            }
            size_t comSize = comPlsOfPa.size();
            if (comSize > 0)
            {
                mOfPa.m_Comp.insert(std::make_pair(dsEdge, std::move(comPlsOfPa)));
            }
        }
        size_t sizeOfCompatiblePlan = mOfPa.m_Comp.size();
        bool isLastPartInNetwork    = dsEdges.size() == 0;
        if (!isLastPartInNetwork && sizeOfCompatiblePlan == 0)
        {
            std::string errorMessage =
                "No compatible plan was found for part with ID " + std::to_string(mOfPa.m_PartId);
            throw NotSupportedException(errorMessage.c_str());
        }
        // Fill up sources and destinations
        const std::vector<const Edge*> srEdges = fPart.GetInputs();
        for (uint32_t s = 0; s < srEdges.size(); ++s)
        {
            const Edge* srEdge = srEdges.at(s);
            OutPart outPa      = parts.GetOutputPart(*srEdge);
            assert(outPa.first == true);
            mOfPa.m_Source.insert(std::make_pair(srEdge, outPa.second));
        }
        for (uint32_t d = 0; d < dsEdges.size(); ++d)
        {
            const Edge* dsEdge = dsEdges.at(d);
            InPart inPa        = parts.GetInputPart(*dsEdge);
            assert(inPa.first == true);
            mOfPa.m_Destination.insert(std::make_pair(dsEdge, inPa.second));
        }
        result.push_back(std::move(mOfPa));
    }

    return result;
}

Combinations CreateSeeds(const GraphOfParts& parts, const Metadata& metadata, const HardwareCapabilities& caps)
{
    const size_t numParts = parts.GetNumParts();
    assert(numParts > 1U);

    Combinations result;

    SramAllocator alloc(caps.GetTotalSramSize() / caps.GetNumberOfSrams());
    Combination comb;

    // First part in topological order
    const PartId fPartId = 0;

    // Second part in topological order
    NxtPa next = GetNxtPart(fPartId, numParts, comb, metadata);
    assert(next.m_Found);
    const Edge* sEdge = next.m_Dst;

    const CompatiblePlansOfParts& comPlsOfPas = metadata.at(fPartId).m_Comp;
    assert(comPlsOfPas.size());

    CompatiblePlansOfParts::const_iterator itPa = comPlsOfPas.find(sEdge);
    const CompatiblePlansOfPart& comPlsOfPa     = itPa->second;
    assert(comPlsOfPa.size());

    CompatiblePlansOfPart::const_iterator it = comPlsOfPa.begin();
    while (it != comPlsOfPa.end())
    {
        // Take the planId and the list of compatible plans of a connected part
        Combinations temp = CombineSeeds(it->first, it->second, next.m_Comb, fPartId, sEdge, parts, unassignedPlanId,
                                         metadata, alloc, caps);
        result.insert(std::end(result), std::begin(temp), std::end(temp));
        ++it;
    }
    return result;
}

GrownSeeds GrowSeeds(const Combinations& combs,
                     const GraphOfParts& parts,
                     const size_t minScore,
                     const Metadata& metadata,
                     const HardwareCapabilities& caps)
{
    const size_t numParts = parts.GetNumParts();
    assert(numParts > 1U);

    GrownSeeds result;
    result.m_Terminated = true;

    SramAllocator alloc(caps.GetTotalSramSize() / caps.GetNumberOfSrams());

    for (uint32_t c = 0; c < combs.size(); ++c)
    {

        if (combs.at(c).m_Scratch.m_Score < minScore)
        {
            continue;
        }

        // Get where it is with the combination
        const PartId fPartId = combs.at(c).m_Scratch.m_CurrPartId;
        if (fPartId < numParts)
        {

            const NxtPa next             = GetNxtPart(fPartId, numParts, combs.at(c), metadata);
            const MetadataOfPart& fMOfPa = metadata.at(fPartId);

            result.m_Terminated = false;

            if (next.m_Found)
            {
                // Second part in topological order
                const Edge* sEdge           = next.m_Dst;
                DstPart::const_iterator eIt = fMOfPa.m_Destination.find(sEdge);
                assert(eIt != fMOfPa.m_Destination.end());
                const PartId sPartId = eIt->second;

                const CompatiblePlansOfParts& comPlsOfPas = fMOfPa.m_Comp;
                assert(comPlsOfPas.size());

                const CompatiblePlansOfPart& comPlsOfPa = (comPlsOfPas.find(sEdge))->second;

                const PlanFromSource fPl = GetPlanFromSource(fPartId, next.m_Comb, metadata);
                const PlanFromSource sPl = GetPlanFromSource(sPartId, next.m_Comb, metadata);
                const PlanId reqPlan     = sPl.m_Found ? sPl.m_Id : unassignedPlanId;

                if (!fPl.m_Found)
                {
                    CompatiblePlansOfPart::const_iterator it = comPlsOfPa.begin();
                    while (it != comPlsOfPa.end())
                    {
                        // Take the planId and the list of compatible plans of a connected part
                        Combinations temp = CombineSeeds(it->first, it->second, next.m_Comb, fPartId, sEdge, parts,
                                                         reqPlan, metadata, alloc, caps);
                        result.m_Combinations.insert(std::end(result.m_Combinations), std::begin(temp), std::end(temp));
                        ++it;
                    }
                }
                else
                {
                    CompatiblePlansOfPart::const_iterator it = comPlsOfPa.find(fPl.m_Id);
                    if (it != comPlsOfPa.end())
                    {
                        // Take the planId and the list of compatible plans of a connected part
                        Combinations temp = CombineSeeds(it->first, it->second, next.m_Comb, fPartId, sEdge, parts,
                                                         reqPlan, metadata, alloc, caps, false);
                        result.m_Combinations.insert(std::end(result.m_Combinations), std::begin(temp), std::end(temp));
                    }
                }
            }
            else
            {
                // Output part
                Combination grownComb    = next.m_Comb;
                const PlanFromSource fPl = GetPlanFromSource(fPartId, next.m_Comb, metadata);
                grownComb.m_Elems.push_back(Elem{ fPartId, fPl.m_Id, {} });
                result.m_Combinations.push_back(grownComb);
            }
        }
        else
        {
            result.m_Combinations.push_back(combs.at(c));
        }
        // Record best score for this iteration
        if ((result.m_Combinations.size() > 0) && (result.m_Combinations.back().m_Scratch.m_Score > result.m_BestScore))
        {
            result.m_BestScore = result.m_Combinations.back().m_Scratch.m_Score;
        }
    }
    return result;
}

Combinations Cascading::Combine(const GraphOfParts& parts)
{
    m_Metadata = CreateMetadata(parts);

    Combinations currSeeds = CreateSeeds(parts, m_Metadata, m_Capabilities);

    GrownSeeds grownSeeds;

    do
    {
        const size_t limit = (grownSeeds.m_BestScore > 1U) ? (grownSeeds.m_BestScore - 1U) : grownSeeds.m_BestScore;
        grownSeeds         = GrowSeeds(currSeeds, parts, limit, m_Metadata, m_Capabilities);
        currSeeds          = grownSeeds.m_Combinations;
    } while (!grownSeeds.m_Terminated);

    return currSeeds;
}

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
    for (const Elem& elem : combination.m_Elems)
    {
        const Part& part = parts.GetPart(elem.m_PartId);
        const Plan& plan = part.GetPlan(elem.m_PlanId);

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

        // Add Buffers from the Plan
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
                }
            }
            if (sharedBuffer)
            {
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
                auto glueIt                       = elem.m_Glues.find(outputEdge);
                if (glueIt != elem.m_Glues.end() && !glueIt->second.m_Glue->m_Graph.GetOps().empty())
                {
                    glues[outputEdge] = glueIt->second.m_Glue;
                }
            }
        }
    }

    return result;
}

}    // namespace support_library
}    // namespace ethosn
