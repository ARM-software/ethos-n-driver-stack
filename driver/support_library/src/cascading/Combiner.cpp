//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "Combiner.hpp"

#include "../SramAllocator.hpp"
#include "../Utils.hpp"
#include "Cascading.hpp"
#include "DebuggingContext.hpp"
#include "Estimation.hpp"
#include "EstimationUtils.hpp"
#include "Part.hpp"
#include "Plan.hpp"

#include <ethosn_utils/Filesystem.hpp>

#include <array>
#include <fstream>
#include <list>

namespace ethosn
{
namespace support_library
{

using namespace utils;

namespace
{

using Allocated = std::pair<bool, uint32_t>;

struct AddedSeed
{
    bool m_Added;
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

    // We are not yet sure what could be a good userId here so we are using zero
    SramAllocator::UserId userId = 0;
    allocated = alloc.Allocate(userId, addSizeInBytes / caps.GetNumberOfSrams(), AllocationPreference::Start);

    if (!allocated.first)
    {
        // There is no space
        return AddedSeed{ false, Combination{} };
    }

    const uint32_t allocatedSram = addSizeInBytes;

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

    if (canMerge && (allocatedSram < sInSize.m_TotAtomic))
    {
        std::string errorMessage =
            "Sram allocation incorrect " + std::to_string(allocatedSram) + " < " + std::to_string(sInSize.m_TotAtomic);
        throw NotSupportedException(errorMessage.c_str());
    }

    result.m_Scratch.m_AllocatedSram = allocatedSram - (canMerge ? sInSize.m_TotAtomic : 0U);

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

    return AddedSeed{ true, result };
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
                          const GrowScheme scheme,
                          const bool create,
                          const bool oneSeed)
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

    const SizeInBytes fTotSize     = GetTotSizeInBytes(fPl);
    const uint32_t baseSizeInBytes = create ? (fTotSize.m_Tot) : comb.m_Scratch.m_AllocatedSram;

    // Process all the list of compatible plans
    for (const CompatiblePlan& fComPl : fComPls)
    {
        if (checkReqPlan && (fComPl.m_Id == reqPlan))
        {
            continue;
        }

        const bool hasGlue = (fComPl.m_Glue.m_Graph.GetOps().size() > 0);

        const bool canMerge = !hasGlue && !outInDram;

        if (scheme == GrowScheme::MergeOnly && !canMerge)
        {
            continue;
        }

        if (scheme == GrowScheme::DramOnly && canMerge)
        {
            continue;
        }

        AddedSeed addedSeed = AddSeed(fPlId, fComPl.m_Id, fPartId, sEdge, sPart, &fComPl.m_Glue, comb, baseSizeInBytes,
                                      alloc, caps, canMerge);

        if (addedSeed.m_Added)
        {
            // Add seed
            result.push_back(addedSeed.m_Combination);
            if (oneSeed)
            {
                break;
            }
        }
    }

    return result;
}

Combinations CombineSeeds(const PlanId fPlId,
                          const CompatiblePlans& fComPls,
                          const Combination& comb,
                          const PartId fPartId,
                          const Edge* sEdge,
                          const GraphOfParts& parts,
                          const PlanId reqPlan,
                          const Metadata& metadata,
                          SramAllocator& alloc,
                          const HardwareCapabilities& caps)
{
    return CombineSeeds(fPlId, fComPls, comb, fPartId, sEdge, parts, reqPlan, metadata, alloc, caps,
                        GrowScheme::Default, true, false);
}

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
                          const GrowScheme scheme,
                          const bool oneSeed)
{
    return CombineSeeds(fPlId, fComPls, comb, fPartId, sEdge, parts, reqPlan, metadata, alloc, caps, scheme, true,
                        oneSeed);
}

CascadingBufferFormat GetBestCascadingBufferDramFormat(const std::array<TensorShape, 2> inputOutputStripeShapes)
{
    using SupportedCompressedFormats = std::vector<CascadingBufferFormat>;

    constexpr size_t sramStripeShapesSize = inputOutputStripeShapes.size();
    SupportedCompressedFormats cascadingBufferSupportedTypePerStripe[sramStripeShapesSize];
    for (size_t sramStripeShapesIdx = 0; sramStripeShapesIdx < sramStripeShapesSize; sramStripeShapesIdx++)
    {
        const TensorShape& currentStripeShape = inputOutputStripeShapes[sramStripeShapesIdx];
        SupportedCompressedFormats& currentCascadedSupportedTypeList =
            cascadingBufferSupportedTypePerStripe[sramStripeShapesIdx];

        if (IsCompressionFormatCompatibleWithStripeAndShape(CompilerDataCompressedFormat::FCAF_DEEP,
                                                            currentStripeShape))
        {
            currentCascadedSupportedTypeList.push_back(CascadingBufferFormat::FCAF_DEEP);
        }
        if (IsCompressionFormatCompatibleWithStripeAndShape(CompilerDataCompressedFormat::FCAF_WIDE,
                                                            currentStripeShape))
        {
            currentCascadedSupportedTypeList.push_back(CascadingBufferFormat::FCAF_WIDE);
        }
    }

    SupportedCompressedFormats supportedTypes;
    static_assert(ETHOSN_ARRAY_SIZE(cascadingBufferSupportedTypePerStripe) == 2, "");
    std::set_intersection(cascadingBufferSupportedTypePerStripe[0].begin(),
                          cascadingBufferSupportedTypePerStripe[0].end(),
                          cascadingBufferSupportedTypePerStripe[1].begin(),
                          cascadingBufferSupportedTypePerStripe[1].end(), std::back_inserter(supportedTypes));

    if (!supportedTypes.empty())
    {
        return supportedTypes.front();
    }

    return CascadingBufferFormat::NHWCB;
}

bool AreMceOperationsCompatible(const Buffer* plan1OutputBuffer,
                                const Buffer* plan2InputBuffer,
                                const Node* destination)
{

    const MceOperationNode* mceOperationNode = dynamic_cast<const MceOperationNode*>(destination);
    if ((mceOperationNode) && (plan1OutputBuffer->m_Location != Location::Dram))
    {
        const TensorShape& inputBufferShape = plan2InputBuffer->m_TensorShape;
        const TensorShape& inputStripeShape = plan2InputBuffer->m_StripeShape;

        if ((mceOperationNode->GetOperation() == ethosn::command_stream::MceOperation::CONVOLUTION) ||
            (mceOperationNode->GetOperation() == ethosn::command_stream::MceOperation::FULLY_CONNECTED))
        {
            if (GetChannels(inputStripeShape) < GetChannels(inputBufferShape))
            {
                return false;
            }
        }
    }
    return true;
}

bool AreBlockConfigsCompatible(const Plan& plan1, const Plan& plan2, const Edge& edge)
{
    Buffer* bufferProduced = plan1.GetOutputBuffer(edge.GetSource());
    Buffer* bufferConsumed = plan2.GetInputBuffer(&edge);

    const bool areBuffersInPleInputSram =
        bufferProduced->m_Location == Location::PleInputSram && bufferConsumed->m_Location == Location::PleInputSram;

    if (areBuffersInPleInputSram)
    {
        ethosn::command_stream::BlockConfig producerBlockConfig = {};
        size_t matching                                         = 0;

        Op* opProducer = plan1.m_OpGraph.GetProducer(bufferProduced);

        const MceOp* mceOp = dynamic_cast<const MceOp*>(opProducer);
        if (!mceOp)
        {
            return true;
        }
        producerBlockConfig = mceOp->m_BlockConfig;

        auto consumers = plan2.m_OpGraph.GetConsumers(bufferConsumed);
        for (auto& consumer : consumers)
        {
            Op* opConsumer                                          = consumer.first;
            ethosn::command_stream::BlockConfig consumerBlockConfig = {};

            const PleOp* pleOp = dynamic_cast<const PleOp*>(opConsumer);
            if (pleOp)
            {
                consumerBlockConfig = pleOp->m_BlockConfig;
            }
            if (producerBlockConfig == consumerBlockConfig)
            {
                ++matching;
            }
        }
        return matching == consumers.size();
    }
    return true;
}

void DumpDebugInfo(const GraphOfParts& parts,
                   const Combinations& combs,
                   std::vector<size_t> stats,
                   const DebuggingContext& debuggingContext,
                   const std::string folder)
{
    using namespace ethosn::utils;

    if (debuggingContext.m_DebugInfo->m_DumpDebugFiles >= CompilationOptions::DebugLevel::High)
    {
        MakeDirectory(debuggingContext.GetAbsolutePathOutputFileName(folder).c_str());

        if (!stats.empty())
        {
            std::ofstream debugIterationStatsDumpFile(
                debuggingContext.GetAbsolutePathOutputFileName(folder + "/Stats.txt"));
            for (auto& val : stats)
            {
                debugIterationStatsDumpFile << "Val : " << val << std::endl;
            }
        }

        size_t combinationNumber = 0;
        for (const Combination& comb : combs)
        {
            std::string subfolder = folder + "/" + std::to_string(combinationNumber);
            MakeDirectory(debuggingContext.GetAbsolutePathOutputFileName(subfolder).c_str());

            debuggingContext.SaveCombinationToDot(CompilationOptions::DebugLevel::None, comb, parts,
                                                  subfolder + "/Detailed.dot", DetailLevel::High);

            ++combinationNumber;
            if (combinationNumber > debuggingContext.GetMaxNumDumps())
            {
                break;
            }
        }
    }
}

void DumpDebugInfo(const GraphOfParts& parts,
                   const Metadata& metadata,
                   const DebuggingContext& debuggingContext,
                   const std::string folder)
{
    using namespace ethosn::utils;
    if (debuggingContext.m_DebugInfo->m_DumpDebugFiles >= CompilationOptions::DebugLevel::High)
    {
        MakeDirectory(debuggingContext.GetAbsolutePathOutputFileName(folder).c_str());

        for (const MetadataOfPart& fMOfPa : metadata)
        {
            const Part& srcPart       = parts.GetPart(fMOfPa.m_PartId);
            std::string srcPartFolder = folder + "/" + srcPart.m_DebugTag;
            // Create source part folder
            MakeDirectory(debuggingContext.GetAbsolutePathOutputFileName(srcPartFolder).c_str());

            std::ofstream debugMergeablePlanDumpFile(
                debuggingContext.GetAbsolutePathOutputFileName(srcPartFolder + "/Cascaded_MergeablePlans.txt"));
            std::ofstream debugGluedPlanDumpFile(
                debuggingContext.GetAbsolutePathOutputFileName(srcPartFolder + "/Cascaded_GluedPlans.txt"));
            std::ofstream debugOutDramPlanDumpFile(
                debuggingContext.GetAbsolutePathOutputFileName(srcPartFolder + "/Cascaded_OutDramPlans.txt"));

            size_t edgeCounter      = 0;
            size_t mergeCounter     = 0;
            size_t outInDramCounter = 0;
            size_t gluedCounter     = 0;
            for (const auto& itPa : fMOfPa.m_Comp)
            {
                const InPart inPa   = parts.GetInputPart(*(itPa.first));
                const Part& dstPart = parts.GetPart(inPa.second);
                for (const auto& itPls : itPa.second)
                {
                    const Plan& srcPlan  = srcPart.GetPlan(itPls.first);
                    const bool outInDram = IsOutputBufferInDram(srcPlan, *(itPa.first));
                    for (const auto& itPl : itPls.second)
                    {
                        const Plan& dstPlan        = dstPart.GetPlan(itPl.m_Id);
                        const size_t fileId        = mergeCounter + outInDramCounter + gluedCounter;
                        const std::string filename = dstPart.m_DebugTag + "_" + srcPlan.m_DebugTag + "_" +
                                                     dstPlan.m_DebugTag + "_Edge" + std::to_string(edgeCounter) +
                                                     "_Detailed_" + std::to_string(fileId) + ".dot";
                        debuggingContext.SaveOpGraphToDot(CompilationOptions::DebugLevel::None, itPl.m_Glue.m_Graph,
                                                          srcPartFolder + "/" + filename, DetailLevel::High);
                        if (!outInDram && itPl.m_Glue.m_Graph.GetOps().empty())
                        {
                            ++mergeCounter;
                            debugMergeablePlanDumpFile << srcPlan.m_DebugTag << ": " << dstPlan.m_DebugTag << std::endl;
                        }
                        if (outInDram)
                        {
                            ++outInDramCounter;
                            debugOutDramPlanDumpFile << srcPlan.m_DebugTag << ": " << dstPlan.m_DebugTag << std::endl;
                        }
                        if (!itPl.m_Glue.m_Graph.GetOps().empty())
                        {
                            ++gluedCounter;
                            debugGluedPlanDumpFile << srcPlan.m_DebugTag << ": " << dstPlan.m_DebugTag << std::endl;
                        }
                    }
                }
                ++edgeCounter;
                debugMergeablePlanDumpFile << "Tot: " << mergeCounter << std::endl;
                debugOutDramPlanDumpFile << "Tot: " << outInDramCounter << std::endl;
                debugGluedPlanDumpFile << "Tot: " << gluedCounter << std::endl;
            }
        }
    }
}

GrownSeeds GrowSeeds(const Combinations& combs,
                     const GraphOfParts& parts,
                     const Metadata& metadata,
                     const HardwareCapabilities& caps,
                     const GrowScheme scheme,
                     const bool oneSeed)
{
    const size_t numParts = parts.GetNumParts();
    assert(numParts > 1U);

    GrownSeeds result;

    SramAllocator alloc(caps.GetTotalSramSize() / caps.GetNumberOfSrams());

    for (const auto& currComb : combs)
    {
        if (currComb.m_Elems.empty())
        {
            continue;
        }

        // Get where it is with the combination
        const PartId fPartId = currComb.m_Scratch.m_CurrPartId;
        if (fPartId < numParts)
        {

            const NxtPa next             = GetNxtPart(fPartId, numParts, currComb, metadata);
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
                    for (const auto& it : comPlsOfPa)
                    {
                        // Take the planId and the list of compatible plans of a connected part
                        Combinations temp = CombineSeeds(it.first, it.second, next.m_Comb, fPartId, sEdge, parts,
                                                         reqPlan, metadata, alloc, caps, scheme, oneSeed);
                        result.m_Combinations.insert(std::end(result.m_Combinations), std::begin(temp), std::end(temp));
                        if (oneSeed && !result.m_Combinations.empty())
                        {
                            break;
                        }
                    }
                }
                else
                {
                    CompatiblePlansOfPart::const_iterator it = comPlsOfPa.find(fPl.m_Id);
                    if (it != comPlsOfPa.end())
                    {
                        // Take the planId and the list of compatible plans of a connected part
                        Combinations temp = CombineSeeds(it->first, it->second, next.m_Comb, fPartId, sEdge, parts,
                                                         reqPlan, metadata, alloc, caps, scheme, false, oneSeed);
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
            result.m_Combinations.push_back(currComb);
        }
    }
    return result;
}

Combination PruneCombinations(const GraphOfParts& parts,
                              const HardwareCapabilities& caps,
                              const Metadata& metadata,
                              const Combinations& combs,
                              const EstimationOptions& estimationOpts,
                              const DebuggingContext& debuggingContext,
                              const std::string folder)
{
    if (combs.size() > 0)
    {
        utils::Optional<Combination> result;
        NetworkPerformanceData refNetPerfData;
        std::vector<uint64_t> stats = {};
        size_t combinationNumber    = 0;
        for (const Combination& combination : combs)
        {
            try
            {
                GrownSeeds local = GrowSeeds({ combination }, parts, metadata, caps, GrowScheme::DramOnly, true);
                if (!local.m_Combinations.empty())
                {
                    OpGraph combiOpGraph = GetOpGraphForCombination(local.m_Combinations.front(), parts);
                    EstimatedOpGraph estimatedOpGraph =
                        ethosn::support_library::EstimateOpGraph(combiOpGraph, caps, estimationOpts);
                    if (!estimatedOpGraph.IsComplete())
                    {
                        throw NotSupportedException("Incomplete estimation");
                    }
                    if (debuggingContext.m_DebugInfo->m_DumpDebugFiles >= CompilationOptions::DebugLevel::High)
                    {
                        stats.push_back(combinationNumber);
                        std::vector<uint64_t> metrics = GetPerformanceMetrics(estimatedOpGraph.m_PerfData);
                        stats.insert(stats.end(), metrics.begin(), metrics.end());
                    }

                    if (!result.has_value() || ComparePerformanceData(estimatedOpGraph.m_PerfData, refNetPerfData) ==
                                                   PerformanceComparisonResult::LeftBetter)
                    {
                        refNetPerfData = estimatedOpGraph.m_PerfData;
                        result         = combination;
                    }
                }
            }
            catch (const NotSupportedException&)
            {
                // Skip this combination
                if (debuggingContext.m_DebugInfo->m_DumpDebugFiles >= CompilationOptions::DebugLevel::High)
                {
                    stats.push_back(combinationNumber);
                    stats.push_back(0UL);
                }
            }
            ++combinationNumber;
        }
        DumpDebugInfo(parts, {}, stats, debuggingContext, folder);

        if (!result.has_value())
        {
            // If Estimation failed, pick the first combination
            return combs.front();
        }
        return result.value();
    }
    return Combination{};
}

}    // namespace

PlanCompatibilityResult ArePlansCompatible(
    const Plan& plan1, const Plan& plan2, const Edge& edge, const HardwareCapabilities& hwCap, const bool forceGlue)
{
    ETHOSN_UNUSED(hwCap);

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

    // Some properties of the buffers must match, as we can't fix everything these by inserting a glue.
    // This would normally indicate there is an issue with the plans generated, and is more of a sanity check.
    //
    // Note that m_QuantizationInfo does not need to match between the buffers, as it is possible to *reinterpret* the
    // quantisation of a buffer without having to insert any glue (i.e. it's a no-op). We will use this to implement the
    // ReinterpretQuantization Operation.

    // The same goes for shape, but only in limited circumstances (e.g. you can't reinterpret a 1x1x1x1 as a 1x100x100x100
    // because there wouldn't be enough data, and there are probably additional limitations for non-linear formats like
    // NHWCB, FCAF). For now we are conservative and only allow this for simple NHWC cases where the full tensor is
    // reinterpreted with a different shape, which we use to implement "DRAM Reshape" Operations as a no-op.
    bool areShapesDifferent = plan1OutputBuffer->m_TensorShape != plan2InputBuffer->m_TensorShape;
    bool isValidNhwcReinterpret =
        plan1OutputBuffer->m_Format == CascadingBufferFormat::NHWC &&
        plan2InputBuffer->m_Format == CascadingBufferFormat::NHWC &&
        GetNumElements(plan1OutputBuffer->m_TensorShape) == GetNumElements(plan2InputBuffer->m_TensorShape);

    bool areBuffersIncompatible = areShapesDifferent && !isValidNhwcReinterpret;
    if (areBuffersIncompatible)
    {
        // Not compatible as the output buffer can't be used directly as the input buffer, and we can't convert
        // between them using a glue (at least not with the current implementation of this function).
        PlanCompatibilityResult result;
        result.m_IsCompatible = false;
        return result;
    }

    // Check if the buffers on the boundary are compatible, i.e. the same (or similar enough that they can be reinterpreted),
    // such that the plans could be directly merged without any additional DMA ops required
    bool areBuffersEquivalent = plan1OutputBuffer->m_Location == plan2InputBuffer->m_Location &&
                                plan1OutputBuffer->m_Format == plan2InputBuffer->m_Format &&
                                plan1OutputBuffer->m_StripeShape == plan2InputBuffer->m_StripeShape &&
                                plan1OutputBuffer->m_Order == plan2InputBuffer->m_Order &&
                                plan1OutputBuffer->m_SizeInBytes == plan2InputBuffer->m_SizeInBytes &&
                                plan1OutputBuffer->m_NumStripes == plan2InputBuffer->m_NumStripes;
    // For some MceOperations (ie Convolution, FullyConnected), we cannot merge plan2's input buffer stripe
    // with plan1's output buffer stripe which splits the full tensor in depth. The reason being
    // Mce cannot keep partial results. So we need to have a glue (ie dma operation) between these plans to
    // stop merging them.
    if ((areBuffersEquivalent) &&
        AreMceOperationsCompatible(plan1OutputBuffer, plan2InputBuffer, edge.GetDestination()) &&
        AreBlockConfigsCompatible(plan1, plan2, edge) && !forceGlue)
    {
        PlanCompatibilityResult result;
        result.m_IsCompatible = true;
        result.m_RequiresGlue = false;
        return result;
    }

    // One buffer may be in SRAM and the other in DRAM, in which case we can insert a single DMA op
    if (plan1OutputBuffer->m_Location == Location::Sram && plan2InputBuffer->m_Location == Location::Dram)
    {
        // Data is going to DRAM, it only requires double buffering
        if (plan1OutputBuffer->m_NumStripes > 2U)
        {
            return PlanCompatibilityResult{};
        }
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
        assert(plan1OutputBuffer->m_Format == CascadingBufferFormat::NHWCB);
        assert(plan2InputBuffer->m_Format == CascadingBufferFormat::NHWCB);

        // Data is going to DRAM, it only requires double buffering
        if (plan1OutputBuffer->m_NumStripes > 2U)
        {
            return PlanCompatibilityResult{};
        }

        PlanCompatibilityResult result;
        result.m_IsCompatible = true;
        result.m_RequiresGlue = true;

        auto dma1      = std::make_unique<DmaOp>();
        DmaOp* dma1Raw = dma1.get();

        CascadingBufferFormat cascadingBufferFormat =
            GetBestCascadingBufferDramFormat({ plan1OutputBuffer->m_StripeShape, plan2InputBuffer->m_StripeShape });
        auto dramBuffer = std::make_unique<Buffer>(
            Lifetime::Atomic, Location::Dram, cascadingBufferFormat, plan1OutputBuffer->m_TensorShape,
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

Metadata CreateMetadata(const GraphOfParts& parts, const HardwareCapabilities& hwCap)
{
    const size_t numParts = parts.GetNumParts();
    assert(numParts > 1U);
    assert(numParts <= static_cast<size_t>(std::numeric_limits<int32_t>::max()));

    Metadata result;
    IncompatiblePlans incompPlans(numParts);

    MetadataOfPart mOfPa;
    CompatiblePlansOfPart comPlsOfPa;
    CompatiblePlans cPls;

    // This loop goes backward to remove all incompatible
    // plans before they are used by any source part.
    const int32_t last = static_cast<int32_t>(numParts) - 1;
    for (int32_t p = last; p >= 0; --p)
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
            const std::vector<PlanId> incompPlansOfDstPart = incompPlans.at(inPa.second);

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
                    if (incompPlansOfDstPart.end() !=
                        std::find(incompPlansOfDstPart.begin(), incompPlansOfDstPart.end(), s))
                    {
                        // Skip this plan.
                        continue;
                    }
                    const Plan& sPl                   = sPart.GetPlan(s);
                    PlanCompatibilityResult plCompRes = ArePlansCompatible(fPl, sPl, *dsEdge, hwCap);
                    if (plCompRes.m_IsCompatible)
                    {
                        if (reqDram && !IsOutputBufferInDram(fPl, *dsEdge) && !plCompRes.m_RequiresGlue)
                        {
                            continue;
                        }
                        cPls.push_back(CompatiblePlan{ std::move(plCompRes.m_Glue), s });
                        // Make sure that there is a "Back to Dram" combination of these two plans
                        // if they are "Sram to Sram"
                        if (!plCompRes.m_RequiresGlue && IsOutputBufferInSram(fPl, *dsEdge) &&
                            IsInputBufferInSram(sPl, *dsEdge))
                        {
                            PlanCompatibilityResult plCompResForceGlue =
                                ArePlansCompatible(fPl, sPl, *dsEdge, hwCap, true);
                            // There is a restriction on number of stripes for plan when going "Back to Dram"
                            if (plCompResForceGlue.m_IsCompatible)
                            {
                                cPls.push_back(CompatiblePlan{ std::move(plCompResForceGlue.m_Glue), s });
                            }
                        }
                    }
                }
                if (cPls.size() > 0)
                {
                    comPlsOfPa.insert(std::make_pair(f, std::move(cPls)));
                }
                else
                {
                    // Add to the list of incompatible plans
                    incompPlans.at(p).push_back(f);
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
        result.push_front(std::move(mOfPa));
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
                     const Metadata& metadata,
                     const HardwareCapabilities& caps,
                     const GrowScheme scheme)
{
    return GrowSeeds(combs, parts, metadata, caps, scheme, false);
}

Combinations Cascading::Combine(const GraphOfParts& parts)
{
    m_Metadata = CreateMetadata(parts, m_Capabilities);

    DumpDebugInfo(parts, m_Metadata, m_DebuggingContext, "Metadata");

    Combinations currSeeds = CreateSeeds(parts, m_Metadata, m_Capabilities);

    // It contains "Merged in Sram" combinations
    GrownSeeds grownSeeds = {};
    // It contains "Back to Dram" combinations
    GrownSeeds haltedSeeds = GrowSeeds(currSeeds, parts, m_Metadata, m_Capabilities, GrowScheme::DramOnly);

    const bool avoidBackToDram = parts.GetNumInvalidPlans() == 0;

    size_t iteration = 0;
    do
    {
        // Grow combinations "Merged in Sram"
        grownSeeds = GrowSeeds(currSeeds, parts, m_Metadata, m_Capabilities, GrowScheme::MergeOnly);

        currSeeds = grownSeeds.m_Combinations;

        if (!avoidBackToDram || grownSeeds.m_Combinations.empty())
        {
            // Concatenate "Merged in Sram" and "Back to Dram"
            currSeeds.insert(std::end(currSeeds), std::begin(haltedSeeds.m_Combinations),
                             std::end(haltedSeeds.m_Combinations));
        }

        if (m_DebuggingContext.m_DebugInfo->m_DumpDebugFiles >= CompilationOptions::DebugLevel::High)
        {
            using namespace ethosn::utils;
            MakeDirectory(m_DebuggingContext.GetAbsolutePathOutputFileName("IntermediateCombinations").c_str());
            MakeDirectory(m_DebuggingContext.GetAbsolutePathOutputFileName("IntermediateHaltedCombinations").c_str());
            MakeDirectory(m_DebuggingContext.GetAbsolutePathOutputFileName("IntermediatePrunedCombinations").c_str());
        }

        // Take the best combination of the lot
        Combination pruned =
            PruneCombinations(parts, m_Capabilities, m_Metadata, currSeeds, GetEstimationOptions(), m_DebuggingContext,
                              "IntermediatePrunedCombinations/Iteration" + std::to_string(iteration));
        // Grow combinations "Back to Dram"
        haltedSeeds = GrowSeeds({ pruned }, parts, m_Metadata, m_Capabilities, GrowScheme::DramOnly);

        DumpDebugInfo(parts, currSeeds, { currSeeds.size() }, m_DebuggingContext,
                      "IntermediateCombinations/Iteration" + std::to_string(iteration));
        DumpDebugInfo(parts, haltedSeeds.m_Combinations, { haltedSeeds.m_Combinations.size() }, m_DebuggingContext,
                      "IntermediateHaltedCombinations/Iteration" + std::to_string(iteration));
        DumpDebugInfo(parts, { pruned }, {}, m_DebuggingContext,
                      "IntermediatePrunedCombinations/Iteration" + std::to_string(iteration));

        ++iteration;
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
