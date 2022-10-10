//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "CombinerDFS.hpp"

#include "../SramAllocator.hpp"
#include "../Utils.hpp"
#include "Cascading.hpp"
#include "Estimation.hpp"
#include "EstimationUtils.hpp"
#include "Part.hpp"
#include "Plan.hpp"
#include "StripeHelper.hpp"

#include <ethosn_utils/Filesystem.hpp>

#include <fstream>

namespace ethosn
{
namespace support_library
{

using namespace utils;

namespace
{

constexpr uint32_t g_NumWeightStripesMin = 1;
constexpr uint32_t g_NumWeightStripesMax = 2;

void DumpDebugInfo(const Combinations& combs,
                   const Combiner::BestCombinationResults& bestCombinationResults,
                   const DebuggingContext& debuggingContext,
                   const std::string folder)
{
    using namespace ethosn::utils;
    if (debuggingContext.m_DebugInfo.m_DumpDebugFiles >= CompilationOptions::DebugLevel::High)
    {
        MakeDirectory(debuggingContext.GetAbsolutePathOutputFileName(folder).c_str());

        for (size_t i = 0; i < combs.size(); ++i)
        {
            std::string prefix    = i == bestCombinationResults.m_BestIdx ? "(BEST) " : "";
            std::string subfolder = folder + "/" + prefix + std::to_string(i);
            MakeDirectory(debuggingContext.GetAbsolutePathOutputFileName(subfolder).c_str());

            if (!combs[i].m_Elems.empty())
            {
                debuggingContext.Save(CompilationOptions::DebugLevel::None, subfolder + "/Detailed.dot",
                                      [&](std::ofstream& s) { SaveCombinationToDot(combs[i], s, DetailLevel::High); });

                debuggingContext.Save(CompilationOptions::DebugLevel::None, subfolder + "/EstimatedDetailed.dot",
                                      [&](std::ofstream& s) {
                                          SaveEstimatedOpGraphToDot(bestCombinationResults.m_OpGraphs[i],
                                                                    bestCombinationResults.m_EstimatedOpGraphs[i], s,
                                                                    DetailLevel::High, {}, {}, {});
                                      });
            }
        }
    }
}

}    // namespace

void Combiner::UpdateStats(const StatsType type)
{
    assert(type < StatsType::NumStats);
    ++m_Stats[static_cast<size_t>(type)];
}

bool Combiner::IsPartInput(const BasePart& part) const
{
    return (0 == m_GraphOfParts.GetPartInputs(part.GetPartId()).size());
}

bool Combiner::IsPartOutput(const BasePart& part) const
{
    return (0 == m_GraphOfParts.GetPartOutputs(part.GetPartId()).size());
}

bool Combiner::IsPartSo(const BasePart& part) const
{
    return (m_GraphOfParts.GetPartOutputs(part.GetPartId()).size() == 1);
}

bool Combiner::IsPartSi(const BasePart& part) const
{
    return (m_GraphOfParts.GetPartInputs(part.GetPartId()).size() == 1);
}

bool Combiner::IsPartMo(const BasePart& part) const
{
    return (m_GraphOfParts.GetPartOutputs(part.GetPartId()).size() > 1);
}

bool Combiner::IsPartSiso(const BasePart& part) const
{
    return (m_GraphOfParts.GetPartInputs(part.GetPartId()).size() == 1 &&
            m_GraphOfParts.GetPartOutputs(part.GetPartId()).size() == 1);
}

bool Combiner::IsPartSimo(const BasePart& part) const
{
    return (m_GraphOfParts.GetPartInputs(part.GetPartId()).size() == 1 &&
            m_GraphOfParts.GetPartOutputs(part.GetPartId()).size() > 1);
}

bool Combiner::IsPartMiso(const BasePart& part) const
{
    return (m_GraphOfParts.GetPartInputs(part.GetPartId()).size() > 1 &&
            m_GraphOfParts.GetPartOutputs(part.GetPartId()).size() == 1);
}

bool Combiner::IsPartMimo(const BasePart& part) const
{
    return (m_GraphOfParts.GetPartInputs(part.GetPartId()).size() > 1 &&
            m_GraphOfParts.GetPartOutputs(part.GetPartId()).size() > 1);
}

const Plan& Combiner::GetPlanForPartFromCombination(const BasePart& part, const Combination& comb) const
{
    // Combination comb must contain part already
    auto elemIt = comb.m_Elems.find(part.GetPartId());
    assert(elemIt != comb.m_Elems.end());

    // Get the plan for the part
    return *elemIt->second.m_Plan;
}

// Check if there is sufficient SRAM for plan to fit
// into the SRAM allocation for the combination that
// is compatible with the plan
bool Combiner::IsPlanAllocated(SectionContext& context,
                               const Plan& plan,
                               const Buffer* const outBufOfPrevPlanInSection,
                               const StatsType sectionType) const
{
    // Some plans (e.g. from ConcatPart) do their own SRAM allocation, as the algorithm here
    // makes some assumptions which are sub-optimal
    if (plan.m_IsPreallocated)
    {
        return true;
    }

    PleKernelInfo pleKernelInfo = plan.GetPleKernelInfo(m_Caps);
    uint32_t pleKernelSize      = 0;
    bool newPleKernel           = false;
    bool isSramAllocated        = true;

    using Allocated = std::pair<bool, uint32_t>;
    Allocated bufferAllocated, pleKernelAllocated;
    SramAllocator localAlloc = context.alloc;

    // We are not yet sure what could be a good userId here so we are using zero
    SramAllocator::UserId userId = 0;

    if (pleKernelInfo.m_PleOp != nullptr)
    {
        // If PLE kernel of the current plan is already used by previous part of the same
        // section, then its size is not counted.

        auto CheckPleKernel =
            [&pleKernelInfo](const std::pair<command_stream::cascading::PleKernelId, uint32_t>& plePair) {
                return (pleKernelInfo.m_PleOp->m_PleKernelId == plePair.first);
            };

        auto pleIterator = std::find_if(context.pleOps.begin(), context.pleOps.end(), CheckPleKernel);

        if (pleIterator == context.pleOps.end())
        {
            pleKernelSize                       = pleKernelInfo.m_Size;
            newPleKernel                        = true;
            pleKernelInfo.m_PleOp->m_LoadKernel = true;
            assert(pleKernelSize != 0);
            assert(pleKernelSize <= m_Caps.GetMaxPleSize());

            // Allocate the PleKernel
            pleKernelAllocated = localAlloc.Allocate(userId, (pleKernelSize), AllocationPreference::Start,
                                                     pleKernelInfo.m_PleOp->m_DebugTag);

            isSramAllocated = pleKernelAllocated.first;

            if (isSramAllocated == true)
            {
                pleKernelInfo.m_PleOp->m_Offset = pleKernelAllocated.second;
            }
        }
        else
        {
            pleKernelInfo.m_PleOp->m_LoadKernel = false;
            pleKernelInfo.m_PleOp->m_Offset     = pleIterator->second;
        }
    }

    if (isSramAllocated)
    {
        // Allocate the Buffers
        // Note this function assumes the plan can be merged with the combination
        // that is associated with the sram allocation. Therefore, the additional
        // sram usage of this plan is the total size - input size in case it is
        // not a start of a section.
        const OpGraph::BufferList& buffers         = plan.m_OpGraph.GetBuffers();
        const PartInputMapping inputBuffersMapping = plan.m_InputMappings;

        OpGraph::BufferList::const_iterator buffersIterator = buffers.begin();

        bool inputBufferNeedAllocation = false;

        if (sectionType == StatsType::StartSection || sectionType == StatsType::SinglePartSection)
        {
            inputBufferNeedAllocation = true;
        }

        while (buffersIterator != buffers.end())
        {
            Buffer* const buf         = *buffersIterator;
            const uint32_t bufferSize = buf->m_SizeInBytes;

            if (buf->m_Location == Location::Sram)
            {
                // If an input buffer is in start of a section, or it's other buffer (i.e output buffer) in start/continue/end of section
                if (inputBufferNeedAllocation || inputBuffersMapping.count(buf) == 0)
                {
                    assert(bufferSize != 0);

                    bufferAllocated = localAlloc.Allocate(userId, (bufferSize / m_Caps.GetNumberOfSrams()),
                                                          AllocationPreference::Start, buf->m_DebugTag);

                    isSramAllocated = bufferAllocated.first;

                    if (isSramAllocated == true)
                    {
                        buf->m_Offset = bufferAllocated.second;
                        context.allocatedBuffers.push_back(buf);
                    }
                    else
                    {
                        break;
                    }
                }
                // If an input buffer in a continue or end section
                else
                {
                    assert(outBufOfPrevPlanInSection != nullptr && outBufOfPrevPlanInSection->m_Offset.has_value());
                    buf->m_Offset = outBufOfPrevPlanInSection->m_Offset;
                }
            }

            ++buffersIterator;
        }
    }

    if (isSramAllocated)
    {
        context.alloc = localAlloc;

        if (newPleKernel)
        {
            context.pleOps.push_back(std::make_pair(pleKernelInfo.m_PleOp->m_PleKernelId, pleKernelAllocated.second));
        }
    }

    return isSramAllocated;
}

bool Combiner::ArePlansAllowedToMerge(const Plan& reference, const Plan& current) const
{
    return !(reference.m_HasIdentityPle && current.m_HasIdentityMce);
}

Combination Combiner::AddTempGlues(const Combination& combination)
{
    Combination result        = combination;
    const GraphOfParts& parts = m_GraphOfParts;
    for (PartId partId : result.m_PartIdsInOrder)
    {
        auto elemIt = result.m_Elems.find(partId);
        assert(elemIt != result.m_Elems.end());
        const Plan& plan = *elemIt->second.m_Plan;

        std::vector<PartInputSlot> inputSlots = parts.GetPartInputs(partId);
        const std::unordered_map<PartInputSlot, std::shared_ptr<StartingGlue>>& startingGlues =
            elemIt->second.m_StartingGlues;
        // All parts needs starting glues in order to be estimated / create an opgraph
        for (PartInputSlot& inputSlot : inputSlots)
        {
            // If there isn't a starting glue on an input slot we have to add a temporary one
            if (startingGlues.find(inputSlot) == startingGlues.end())
            {
                Buffer* buffer    = plan.GetInputBuffer(inputSlot);
                auto startingGlue = std::make_shared<StartingGlue>();
                if (buffer->m_Location == Location::Sram)
                {
                    // Choose the best format for the DRAM buffer. Note that this format won't necessarily
                    // be the same as the format used in the final compilation, because we don't know what other
                    // users of this buffer will require. We could simply assume NHWCB which would be the most
                    // conservative in terms of performance and compatibility, but this might lead to pessimistic
                    // performance estimates due to chunking.
                    CascadingBufferFormat dramFormat = impl::GetBestDramBufferFormat({ buffer }, m_CompilationOptions);
                    auto dramBuffer        = std::make_unique<Buffer>(Location::Dram, dramFormat, buffer->m_TensorShape,
                                                               TensorShape{ 0, 0, 0, 0 }, TraversalOrder::Xyz,
                                                               utils::TotalSizeBytesNHWCB(buffer->m_TensorShape),
                                                               buffer->m_QuantizationInfo);
                    dramBuffer->m_DataType = buffer->m_DataType;
                    dramBuffer->m_BufferType = BufferType::Intermediate;
                    Buffer* dramBufferRaw    = dramBuffer.get();
                    auto dma                 = std::make_unique<DmaOp>(buffer->m_Format);
                    DmaOp* dmaRaw            = dma.get();
                    startingGlue->m_Graph.AddBuffer(std::move(dramBuffer));
                    startingGlue->m_Graph.AddOp(std::move(dma));
                    startingGlue->m_Graph.AddConsumer(dramBufferRaw, dmaRaw, 0);
                    startingGlue->m_ExternalConnections.m_OpsToBuffers.insert({ dmaRaw, buffer });
                }
                elemIt->second.m_StartingGlues.insert({ inputSlot, startingGlue });
            }
        }

        std::vector<PartOutputSlot> outputSlots = parts.GetPartOutputs(partId);
        const std::unordered_map<PartOutputSlot, std::shared_ptr<EndingGlue>>& endingGlues =
            elemIt->second.m_EndingGlues;
        for (PartOutputSlot& outputSlot : outputSlots)
        {
            // Same for output slots and ending glue
            if (endingGlues.find(outputSlot) == endingGlues.end())
            {
                Buffer* buffer  = plan.GetOutputBuffer(outputSlot);
                auto endingGlue = std::make_shared<EndingGlue>();
                if (buffer->m_Location == Location::Sram)
                {
                    // Choose the best format for the DRAM buffer. Note that this format won't necessarily
                    // be the same as the format used in the final compilation, because we don't know what other
                    // users of this buffer will require. We could simply assume NHWCB which would be the most
                    // conservative in terms of performance and compatibility, but this might lead to pessimistic
                    // performance estimates due to chunking.
                    CascadingBufferFormat dramFormat = impl::GetBestDramBufferFormat({ buffer }, m_CompilationOptions);
                    auto dramBuffer        = std::make_unique<Buffer>(Location::Dram, dramFormat, buffer->m_TensorShape,
                                                               TensorShape{ 0, 0, 0, 0 }, TraversalOrder::Xyz,
                                                               utils::TotalSizeBytesNHWCB(buffer->m_TensorShape),
                                                               buffer->m_QuantizationInfo);
                    dramBuffer->m_DataType = buffer->m_DataType;
                    dramBuffer->m_BufferType = BufferType::Intermediate;
                    Buffer* dramBufferRaw    = dramBuffer.get();
                    auto dma                 = std::make_unique<DmaOp>(buffer->m_Format);
                    DmaOp* dmaRaw            = dma.get();
                    endingGlue->m_Graph.AddBuffer(std::move(dramBuffer));
                    endingGlue->m_Graph.AddOp(std::move(dma));
                    endingGlue->m_Graph.SetProducer(dramBufferRaw, dmaRaw);
                    endingGlue->m_ExternalConnections.m_BuffersToOps.insert({ buffer, dmaRaw });
                }
                elemIt->second.m_EndingGlues.insert({ outputSlot, endingGlue });
            }
        }
    }
    return result;
}

Combiner::BestCombinationResults Combiner::GetBestCombination(const Combinations& combs)
{
    assert(combs.size() > 0);
    // If there is only one combination to estimate, then there's nothing it to so don't bother estimating it.
    // However when debugging it is useful to see the estimation, so we skip this optimisation
    if (combs.size() == 1 && m_DebuggingContext.m_DebugInfo.m_DumpDebugFiles < CompilationOptions::DebugLevel::High)
    {
        return { 0, {}, {}, {} };
    }

    BestCombinationResults result;
    utils::Optional<size_t> bestIdx;
    utils::Optional<double> bestMetric;
    for (size_t i = 0; i < combs.size(); ++i)
    {
        const Combination& combination = combs[i];
        // Add temporary glues to partial combinations so we can estimate performance
        Combination comb     = AddTempGlues(combination);
        OpGraph combiOpGraph = GetOpGraphForCombination(comb, m_GraphOfParts);

        // Estimate the combination we're considering
        EstimatedOpGraph estimatedOpGraph = ethosn::support_library::EstimateOpGraph(combiOpGraph, m_Caps, m_EstOpt);

        if (m_DebuggingContext.m_DebugInfo.m_DumpDebugFiles >= CompilationOptions::DebugLevel::High)
        {
            result.m_EstimatedOpGraphs.push_back(estimatedOpGraph);
            result.m_OpGraphs.push_back(combiOpGraph);
            result.m_CompletedCombinations.push_back(comb);
        }

        if (!bestIdx.has_value() || estimatedOpGraph.m_Metric < bestMetric.value())
        {
            bestIdx    = i;
            bestMetric = estimatedOpGraph.m_Metric;
        }
    }
    result.m_BestIdx = bestIdx.value();
    return result;
}

Combination Combiner::GetBestCombinationSafe(Combinations& combs)
{
    // Filter invalid combinations
    Combinations filteredCombs = utils::Filter(combs, [](const Combination& c) { return c.GetNumElems() > 0; });
    if (filteredCombs.size() == 0)
    {
        return Combination();
    }
    return std::move(filteredCombs[GetBestCombination(filteredCombs).m_BestIdx]);
}

const Combination& Combiner::GetBestCombination() const
{
    return m_BestCombination;
}

OpGraph Combiner::GetMergedOpGraphForBestCombination() const
{
    assert(m_MergedOpGraphReady == true);
    return m_MergedOpGraphForBestCombination;
}

/// Adds DmaOps (and possibly Buffers) to the given OpGraph to copy the given
/// existing `source` buffer to the given existing `dest` buffer.
/// Sram -> Dram and Dram -> Sram copies are done with a single DmaOp, and Dram -> Dram copies
/// are done with a DMA through Sram.
/// If external connection objects are provided, these are used to store the connections from the corresponding DMA op(s)
/// to the existing buffers. If not provided, these connections are made internally in the given OpGraph.
void AddCopyBetweenBuffers(OwnedOpGraph& graph,
                           Buffer* source,
                           GlueConnections* sourceExternalConnections,
                           Buffer* dest,
                           GlueConnections* destExternalConnections,
                           const HardwareCapabilities& caps)
{
    DmaOp* sourceDma = nullptr;
    DmaOp* destDma   = nullptr;
    if ((source->m_Location == Location::Dram) ^ (dest->m_Location == Location::Dram))
    {
        // Dram -> Sram or Sram -> Dram. Just need a single DMA.
        CascadingBufferFormat dramFormat = source->m_Location == Location::Dram ? source->m_Format : dest->m_Format;
        auto dma                         = std::make_unique<DmaOp>(dramFormat);
        sourceDma                        = dma.get();
        destDma                          = dma.get();
        graph.AddOp(std::move(dma));
    }
    else if (source->m_Location == Location::Dram && dest->m_Location == Location::Dram)
    {
        // Dram -> Dram. Copy via SRAM
        auto dma1 = std::make_unique<DmaOp>(source->m_Format);
        sourceDma = dma1.get();

        std::unique_ptr<Buffer> sramBuffer =
            impl::MakeGlueIntermediateSramBuffer(dest->m_TensorShape, dest->m_QuantizationInfo, dest->m_DataType,
                                                 { dest->m_Format, source->m_Format }, caps);
        Buffer* sramBufferRaw = sramBuffer.get();
        auto dma2             = std::make_unique<DmaOp>(dest->m_Format);
        destDma               = dma2.get();

        graph.AddOp(std::move(dma1));
        graph.AddOp(std::move(dma2));
        graph.AddBuffer(std::move(sramBuffer));
        graph.SetProducer(sramBufferRaw, sourceDma);
        graph.AddConsumer(sramBufferRaw, destDma, 0);
    }
    else
    {
        assert(false);    // Sram -> Sram. Not supported by this function.
    }

    // Connect the source and dest DmaOps to the source and dest buffers.
    // These might be internal connections or external connections.
    if (sourceExternalConnections == nullptr)
    {
        graph.AddConsumer(source, sourceDma, 0);
    }
    else
    {
        sourceExternalConnections->m_BuffersToOps.insert({ source, sourceDma });
    }

    if (destExternalConnections == nullptr)
    {
        graph.AddProducer(dest, destDma);
    }
    else
    {
        destExternalConnections->m_OpsToBuffers.insert({ destDma, dest });
    }
}

// A source part is glued to its destinations
Combination Combiner::GluePartToCombinationSrcToDests(const BasePart& sPart,
                                                      const Combination& comb,
                                                      const std::vector<PartConnection>& destPartEdge)
{
    assert(destPartEdge.size() != 0);
    Combination result = comb;

    // Find element belonging to source part in the combination
    auto elemIt = comb.m_Elems.find(sPart.GetPartId());
    assert(elemIt != comb.m_Elems.end());
    const Plan& sourcePlan = *elemIt->second.m_Plan;
    // Find the output buffer of the source node.
    // Note all destination nodes are branched off from the same source node
    Buffer* producedBuffer = sourcePlan.GetOutputBuffer(destPartEdge.at(0).m_Source);
    assert(producedBuffer != nullptr);

    // Find the input buffers in the destination plans
    std::vector<std::pair<PartConnection, Buffer*>> consumerBuffers;
    for (const auto& partEdge : destPartEdge)
    {
        const BasePart& part   = m_GraphOfParts.GetPart(partEdge.m_Destination.m_PartId);
        const Plan& plan       = GetPlanForPartFromCombination(part, comb);
        Buffer* consumerBuffer = plan.GetInputBuffer(partEdge.m_Destination);
        assert(consumerBuffer != nullptr);
        consumerBuffers.push_back(std::make_pair(partEdge, consumerBuffer));
    }

    // Sort the consumers so that DRAM consumers are processed first. This is because these buffers could be re-used
    // as part of the glue for other consumers, so we avoid having to create as many new buffers (and thus make a more efficient
    // graph).
    // Note that a stable sort is used, so that the order is deterministic when there are multiple SRAM or DRAM consumers.
    std::stable_sort(consumerBuffers.begin(), consumerBuffers.end(),
                     [](std::pair<PartConnection, Buffer*> a, std::pair<PartConnection, Buffer*> b) {
                         return (a.second->m_Location == Location::Dram) > (b.second->m_Location == Location::Dram);
                     });

    // Maintain a set of DRAM buffers that are available for use in the glue.
    // These are used if possible, rather than adding new buffers.
    std::map<CascadingBufferFormat, Buffer*> dramBuffers;
    if (producedBuffer->m_Location == Location::Dram)
    {
        dramBuffers[producedBuffer->m_Format] = producedBuffer;
    }

    EndingGlue endingGlue;    // We'll populate this as we go with any ending glue for the source part

    // Adds a new DRAM buffer of the given format to the ending glue, so that it can be used in any starting glues of consumers.
    // Also adds the DmaOps to connect this buffer to where it is copied from.
    auto addNewBuffer = [&](CascadingBufferFormat format, Buffer* copiedFrom) {
        auto dramBuffer = std::make_unique<Buffer>(
            Location::Dram, format, producedBuffer->m_TensorShape, TensorShape{ 0, 0, 0, 0 }, TraversalOrder::Xyz,
            CalculateBufferSize(producedBuffer->m_TensorShape, format), producedBuffer->m_QuantizationInfo);
        dramBuffer->m_DataType   = producedBuffer->m_DataType;
        dramBuffer->m_BufferType = BufferType::Intermediate;
        Buffer* dramBufferRaw    = dramBuffer.get();
        endingGlue.m_Graph.AddBuffer(std::move(dramBuffer));

        // If the new buffer is being copied from the original producedBuffer, then the connections to the DmaOp
        // need to be in the external connections of the ending glue (as they connect something in the glue to something
        // in the plan). Otherwise we assume the `copiedFrom` buffer is part of the ending glue, and so it needs an internal
        // connection.
        GlueConnections* connections = (copiedFrom == producedBuffer) ? &endingGlue.m_ExternalConnections : nullptr;
        AddCopyBetweenBuffers(endingGlue.m_Graph, copiedFrom, connections, dramBufferRaw, nullptr, m_Caps);

        // Store the buffer - we may be able to re-use this buffer later.
        dramBuffers[format] = dramBufferRaw;
        return dramBufferRaw;
    };

    // Returns a DRAM buffer suitable for copying to/from the given set of SRAM buffers.
    // This will be an existing DRAM buffer from `dramBuffers` if one exists and is compatible, otherwise
    // it will make a new one and return that.
    auto getOrAddCompatibleDramBuffer = [&](const std::initializer_list<const Buffer*>& sramBuffers) {
        // First check if we have an existing buffer that is usable, to avoid adding any more
        for (std::pair<CascadingBufferFormat, Buffer*> formatAndBuffer : dramBuffers)
        {
            if (std::all_of(sramBuffers.begin(), sramBuffers.end(), [&](const Buffer* b) {
                    return impl::IsSramBufferCompatibleWithDramFormat(*b, formatAndBuffer.first);
                }))
            {
                return formatAndBuffer.second;
            }
        }
        // Need to add a new buffer of a compatible format.
        CascadingBufferFormat format = impl::GetBestDramBufferFormat(sramBuffers, m_CompilationOptions);
        Buffer* newBuffer            = addNewBuffer(format, producedBuffer);
        return newBuffer;
    };

    // Go through every consumer and connect it up with appropriate glue.
    for (std::pair<PartConnection, Buffer*> consumerBufferPair : consumerBuffers)
    {
        PartConnection partEdge = consumerBufferPair.first;
        Buffer* consumerBuffer  = consumerBufferPair.second;

        StartingGlue startingGlue;    // We will fill this in with any starting glue that this consumer needs

        // Consider each case of Sram/Dram producer/consumer separately.
        // Although there is some overlap between these cases, this was found to be the least confusing approach.
        if (producedBuffer->m_Location == Location::Sram && consumerBuffer->m_Location == Location::Dram)
        {
            // There might already be an existing DRAM buffer of the right format, so we can avoid adding anything.
            // This can only be done for intermediate buffers though, as outputs need to have their own buffer
            auto dramBufferIt = dramBuffers.find(consumerBuffer->m_Format);
            if (dramBufferIt != dramBuffers.end() && consumerBuffer->m_BufferType == BufferType::Intermediate)
            {
                // Re-use this existing buffer by adding a replacement link
                startingGlue.m_ExternalConnections.m_ReplacementBuffers[consumerBuffer] = dramBufferIt->second;
            }
            else
            {
                // We might be able to add a single DMA to copy directly from the producer buffer,
                Buffer* dramBufferToCopyFrom = producedBuffer;
                if (!impl::IsSramBufferCompatibleWithDramFormat(*producedBuffer, consumerBuffer->m_Format))
                {
                    // If the SRAM buffer is not compatible though, then we'll need to do a conversion.
                    // We may be lucky and there is already a DRAM buffer that is compatible that we can copy from, or we may need to add a new one.
                    dramBufferToCopyFrom = getOrAddCompatibleDramBuffer({ producedBuffer });
                }

                // We could re-use this consumer DRAM buffer for other consumers, to save them doing their own conversion.
                // Only intermediate buffers can be shared though (Outputs, for example, don't allow reading).
                if (consumerBuffer->m_BufferType == BufferType::Intermediate)
                {
                    // In order for DRAM buffers in consuming plans to be available for sharing, a new copy of this buffer
                    // must be made in the ending glue of the producer, and then linked to the existing consumer buffer via a replacement.
                    Buffer* replacementBuffer = addNewBuffer(consumerBuffer->m_Format, dramBufferToCopyFrom);
                    startingGlue.m_ExternalConnections.m_ReplacementBuffers[consumerBuffer] = replacementBuffer;
                }
                else
                {
                    // This consumer buffer can't be re-used, so just copy from the buffer we chose above in the starting glue.
                    AddCopyBetweenBuffers(startingGlue.m_Graph, dramBufferToCopyFrom,
                                          &startingGlue.m_ExternalConnections, consumerBuffer,
                                          &startingGlue.m_ExternalConnections, m_Caps);
                }
            }
        }
        else if (producedBuffer->m_Location == Location::Dram && consumerBuffer->m_Location == Location::Sram)
        {
            // We might be able to add a single DMA to copy directly from the producer buffer,
            Buffer* dramBufferToCopyFrom = producedBuffer;
            if (!impl::IsSramBufferCompatibleWithDramFormat(*consumerBuffer, producedBuffer->m_Format))
            {
                // If the SRAM buffer is not compatible though, then we'll need to do a conversion.
                // We may be lucky and there is already a DRAM buffer that is compatible that we can copy from, or we may need to add a new one.
                dramBufferToCopyFrom = getOrAddCompatibleDramBuffer({ consumerBuffer });
            }

            // Add a DMA to the starting glue, to copy from the chosen DRAM buffer.
            AddCopyBetweenBuffers(startingGlue.m_Graph, dramBufferToCopyFrom, &startingGlue.m_ExternalConnections,
                                  consumerBuffer, &startingGlue.m_ExternalConnections, m_Caps);
        }
        else if (producedBuffer->m_Location == Location::Sram && consumerBuffer->m_Location == Location::Sram)
        {
            // SRAM to SRAM always needs to go via DRAM (note that this isn't a cascade!).
            // We may be lucky and there is already a DRAM buffer that is compatible that we can copy from, or we may need to add a new one.
            Buffer* dramBufferToCopyFrom = getOrAddCompatibleDramBuffer({ producedBuffer, consumerBuffer });
            // Add a DMA to the starting glue, to copy from the chosen DRAM buffer.
            AddCopyBetweenBuffers(startingGlue.m_Graph, dramBufferToCopyFrom, &startingGlue.m_ExternalConnections,
                                  consumerBuffer, &startingGlue.m_ExternalConnections, m_Caps);
        }
        else if (producedBuffer->m_Location == Location::Dram && consumerBuffer->m_Location == Location::Dram)
        {
            // There might already be an existing DRAM buffer of the right format, so we can avoid adding anything.
            // This can only be done for intermediate buffers though, as outputs need to have their own buffer
            auto dramBufferIt = dramBuffers.find(consumerBuffer->m_Format);
            if (dramBufferIt != dramBuffers.end() && consumerBuffer->m_BufferType == BufferType::Intermediate)
            {
                // Re-use this existing buffer by adding a replacement link
                startingGlue.m_ExternalConnections.m_ReplacementBuffers[consumerBuffer] = dramBufferIt->second;
            }
            // In the case that consumerBuffer is an output buffer, it can't be a simple replacement of the
            // producedBuffer, but we might be able to make a new "merged" buffer that is an output buffer,
            // and replace both with this new buffer.
            // Merging gets complicated if we have multiple consumers, as the merging may invalidate other
            // decisions. Therefore we only do this for simple single-consumer cases at the moment.
            else if (consumerBuffers.size() == 1 && consumerBuffer->m_BufferType == BufferType::Output &&
                     consumerBuffer->m_Format == producedBuffer->m_Format &&
                     consumerBuffer->m_QuantizationInfo == producedBuffer->m_QuantizationInfo &&
                     consumerBuffer->m_TensorShape == producedBuffer->m_TensorShape &&
                     consumerBuffer->m_SizeInBytes == producedBuffer->m_SizeInBytes)
            {
                std::unique_ptr<Buffer> mergedBuffer = std::make_unique<Buffer>(
                    Location::Dram, consumerBuffer->m_Format, consumerBuffer->m_TensorShape, TensorShape{ 0, 0, 0, 0 },
                    TraversalOrder::Xyz, consumerBuffer->m_SizeInBytes, consumerBuffer->m_QuantizationInfo);
                mergedBuffer->m_DebugTag           = "Merged " + consumerBuffer->m_DebugTag;
                mergedBuffer->m_BufferType         = consumerBuffer->m_BufferType;
                mergedBuffer->m_OperationId        = consumerBuffer->m_OperationId;
                mergedBuffer->m_ProducerOutputIndx = consumerBuffer->m_ProducerOutputIndx;
                mergedBuffer->m_DataType           = consumerBuffer->m_DataType;
                Buffer* mergedBufferRaw            = mergedBuffer.get();

                endingGlue.m_Graph.AddBuffer(std::move(mergedBuffer));
                // Mark both buffers as being replaced by the new merged buffer (the other is done later)
                endingGlue.m_ExternalConnections.m_ReplacementBuffers.insert({ producedBuffer, mergedBufferRaw });
                startingGlue.m_ExternalConnections.m_ReplacementBuffers.insert({ consumerBuffer, mergedBufferRaw });
            }
            // We could re-use this consumer DRAM buffer for other consumers, to save them doing their own conversion.
            // Only intermediate buffers can be shared though (Outputs, for example, don't allow reading).
            else if (consumerBuffer->m_BufferType == BufferType::Intermediate)
            {
                // In order for DRAM buffers in consuming plans to be available for sharing, a new copy of this buffer
                // must be made in the ending glue of the producer, and then linked to the existing consumer buffer via a replacement.
                Buffer* replacementBuffer = addNewBuffer(consumerBuffer->m_Format, producedBuffer);
                startingGlue.m_ExternalConnections.m_ReplacementBuffers.insert({ consumerBuffer, replacementBuffer });
            }
            else
            {
                // The consumer buffer must be an output buffer, and thus requires its own copy.
                AddCopyBetweenBuffers(endingGlue.m_Graph, producedBuffer, &endingGlue.m_ExternalConnections,
                                      consumerBuffer, &startingGlue.m_ExternalConnections, m_Caps);
            }
        }
        result.SetStartingGlue(std::move(startingGlue), partEdge.m_Destination);
    }

    result.AddEndingGlue(std::move(endingGlue), destPartEdge.at(0).m_Source);
    return result;
}

void Combiner::DeallocateUnusedBuffers(const Buffer& prevPlanBuffer, SectionContext& context)
{
    // If the output buffer from the previous plan contains the full tensor (either in SRAM like in
    // a strategy 1/3 cascade or in DRAM), then we can safely free everything else in SRAM.
    if (prevPlanBuffer.IsFullTensor())
    {
        for (size_t i = context.allocatedBuffers.size() - 1; i < context.allocatedBuffers.size(); --i)
        {
            Buffer* b = context.allocatedBuffers[i];
            if (b != &prevPlanBuffer)
            {
                context.alloc.Free(0, b->m_Offset.value());
                context.allocatedBuffers.erase(context.allocatedBuffers.begin() + i);
            }
        }
    }
}

// Try to end a section of the combination.
// This is called only when a section needs to be ended since the plan
// requirements are different to ContinueSection
//
// See diagram in StartSection.
Combination Combiner::EndSection(const BasePart& part,
                                 const BasePart& sPart,
                                 const Combination& comb,
                                 const SectionContext& context,
                                 uint32_t prevNumWeightStripes,
                                 bool prevDoubleBuffered,
                                 uint32_t totalAgents)
{
    UpdateStats(StatsType::EndSection);

    Combination result = {};

    if (IsPartSi(part))
    {
        std::vector<PartConnection> connections =
            m_GraphOfParts.GetConnectionsBetween(sPart.GetPartId(), part.GetPartId());

        // Sanity check: section is continued. It must be the single output of
        // its source part.
        assert(connections.size() == 1);

        const Plan& sPlan = GetPlanForPartFromCombination(sPart, comb);

        const PartConnection& connection = connections.at(0);

        ethosn::command_stream::BlockConfig blkConfig = sPlan.GetBlockConfigures(connection.m_Source);
        Buffer* sramBuffer                            = sPlan.GetOutputBuffer(connection.m_Source);

        SectionContext contextCopy = context;
        DeallocateUnusedBuffers(*sramBuffer, contextCopy);

        // Check if this Part can double buffer.
        // By default, no double buffering is performed.
        uint32_t currNumWeightStripesMax = g_NumWeightStripesMin;
        if (part.CanDoubleBufferWeights() && !prevDoubleBuffered)
        {
            currNumWeightStripesMax = g_NumWeightStripesMax;
        }

        // Double buffering is performed on a per Section basis, i.e. either the entire Section double buffers weights
        // (if the Parts allow it) or the Section single buffers weights. This double buffering is considered when the
        // Part being evaluated can be double buffered.
        for (uint32_t currNumWeightStripes = g_NumWeightStripesMin; currNumWeightStripes <= currNumWeightStripesMax;
             ++currNumWeightStripes)
        {
            // Determine which numWeightStripes to use, based on the history of double-buffering.
            // If previous Part was double-buffered, then:
            //      1. Pass that number of weightStripes during current plan generation
            //      2. Pass the same number to the next Parts, during the recursive plan generation calls.
            // Otherwise, pass the current weightStripe number from the local for-loop.
            // This is necessary, because if there was no double-buffering in the past and there is the possibility
            // to double buffer now, then multiple plans must be created for both single buffering and double buffering weights.
            uint32_t numWeightStripes = prevDoubleBuffered ? prevNumWeightStripes : currNumWeightStripes;
            Plans plans               = part.GetPlans(CascadeType::End, blkConfig, sramBuffer, numWeightStripes);

            for (Plan& plan : plans)
            {
                // Make a copy of the allocator since every plan needs to have its own,
                // each potential section won't allocate from the same allocator.
                SectionContext tempContext = contextCopy;

                if (!ArePlansAllowedToMerge(sPlan, plan))
                {
                    continue;
                }

                if (!IsPlanAllocated(tempContext, plan, sramBuffer, StatsType::EndSection))
                {
                    continue;
                }

                if (!IsSectionSizeSupported(StatsType::EndSection, plan, totalAgents))
                {
                    continue;
                }

                // Add current part and plan to the combination
                StartingGlue startingGlue;
                EndingGlue endingGlue;
                startingGlue.m_ExternalConnections.m_ReplacementBuffers.insert(
                    { plan.GetInputBuffer(connection.m_Destination), sramBuffer });
                Combination section =
                    comb + Combination(part, std::move(plan), m_PartOrderTable[part.GetPartId()].first);
                section.SetStartingGlue(std::move(startingGlue), connection.m_Destination);
                section.AddEndingGlue(std::move(endingGlue), connection.m_Source);

                Combinations options = { result, section };
                result               = GetBestCombinationSafe(options);
            }
        }

        //  Next part in the graph
        const BasePart* nextPartGraph = GetNextPart(&part);

        if (!result.m_Elems.empty() && nextPartGraph != nullptr)
        {
            result = result + FindBestCombinationForPart(*nextPartGraph);

            // Each of it destination part will start its own new section.
            // Therefore they all need to be glued with their source.
            std::vector<PartConnection> destConnections = m_GraphOfParts.GetDestinationConnections(part.GetPartId());

            if (destConnections.empty() == false)
            {
                result = GluePartToCombinationSrcToDests(part, result, destConnections);
            }
        }
    }

    return result;
}

bool Combiner::IsSectionSizeSupported(const StatsType sectionInfo, const Plan& plan, uint32_t& totalAgents)
{
    bool result = true;

    // Account for any Dma Ops in the glue logic at the input edge of the plan
    if (sectionInfo == StatsType::StartSection || sectionInfo == StatsType::SinglePartSection)
    {
        for (auto const& inputMapping : plan.m_InputMappings)
        {
            // A corresponding Dma Op in glue logic is not needed if the buffer is in Dram
            if (inputMapping.first->m_Location != Location::Dram)
            {
                // If any of the input buffer's consumers is Cascade, the buffer's producer Dma Op
                // must also be cascade. Assume that the Dma Op would result in a single agent.
                // Therefore, increment the agent count by one.
                if (!inputMapping.first->IsFullTensor())
                {
                    totalAgents += 1;
                }
            }
        }
    }

    // Count Agents for each Op in the graph. The Ops should be in execution order.
    for (Op* op : plan.m_OpGraph.GetOps())
    {
        totalAgents += op->GetNumberOfAgents();
        result &= totalAgents <= m_Caps.GetAgentWindowSize();

        // The total is to be reset when all preceding Agents have finished execution.
        // All preceding Agents must finish execution when an Atomic Op finishes
        // execution. This Atomic Op must be in the path from IFM to OFM. We can
        // identify whether an Op is in the path from IFM to OFM by checking its
        // output buffer's format. If the buffer's format is WEIGHT, it means that the
        // buffer's producer loads weights and hence it is not in the IFM to OFM path.
        if (plan.m_OpGraph.GetOutput(op)->m_Format != CascadingBufferFormat::WEIGHT)
        {
            totalAgents = plan.m_OpGraph.GetOutput(op)->IsFullTensor() ? 0 : totalAgents;
        }
    }

    // Account for any Dma Ops in the glue logic at the output edge of the plan
    if (sectionInfo == StatsType::EndSection || sectionInfo == StatsType::SinglePartSection)
    {
        for (auto const& outputMapping : plan.m_OutputMappings)
        {
            // A corresponding Dma Op in glue logic is not needed if the buffer is in Dram
            if (outputMapping.first->m_Location != Location::Dram)
            {
                // If the output buffer's producer is cascade, the buffer's consumer Dma Op must also
                // be cascade. Assume that the Dma Op would result in a single agent. Therefore,
                // increment the agent count by one.
                if (!outputMapping.first->IsFullTensor())
                {
                    totalAgents += 1;
                }
            }
        }
    }

    result &= totalAgents <= m_Caps.GetAgentWindowSize();
    return result;
}

// Try to start a section
//
//            Section A                             Section B
// - - - ------------------------            --------------------- - - -
//                               |          |
//             -------           |          |           -------            -------
//            |       |  ------  |  ------  |  ------  |       |  ------  |       |
//  - - - ----|   X   |-| SRAM |-|-| DRAM |-|-| SRAM |-|   Y   |-| SRAM |-|   Z   |
//            |       |  ------  |  ------  |  ------  |       |  ------  |       |
//             -------           |          |           -------            -------
//                ^              |          |              ^                  ^
// - - - ---------|--------------            --------------|------ - - -      |
//                |                                        |                  |
//          End of Section                         Start of a section         |
//                                                                            |
//                                                 Continue Section ----------
//
Combination Combiner::StartSection(const BasePart& part, const BasePart& nextPart)
{
    UpdateStats(StatsType::StartSection);

    // Sanity check
    // This is a section allowing at least two parts (not a lonely one)
    // Therefore the next part must be the destination part.
    assert(nextPart.GetPartId() == m_GraphOfParts.GetDestinationParts(part.GetPartId()).at(0).m_PartId);

    Combination result = {};

    if (IsPartSo(part))
    {
        // sanity check SISO is the only use case.
        assert(m_GraphOfParts.GetPartOutputs(part.GetPartId()).size() == 1);

        // Check if this Part can double buffer.
        // By default, no double buffering is performed.
        uint32_t currNumWeightStripesMax = g_NumWeightStripesMin;
        bool hasSectionDoubleBuffered    = false;
        if (part.CanDoubleBufferWeights())
        {
            currNumWeightStripesMax  = g_NumWeightStripesMax;
            hasSectionDoubleBuffered = true;
        }

        // Double buffering is performed on a per Section basis, i.e. either the entire Section double buffers weights
        // (if the Parts allow it) or the Section single buffers weights. This double buffering is considered when the
        // Part being evaluated can be double buffered.
        for (uint32_t currNumWeightStripes = g_NumWeightStripesMin; currNumWeightStripes <= currNumWeightStripesMax;
             ++currNumWeightStripes)
        {
            Plans plans = part.GetPlans(CascadeType::Beginning, ethosn::command_stream::BlockConfig{}, nullptr,
                                        currNumWeightStripes);

            // SISO part:
            //
            // Try to start a section
            // Make sure that the chosen next plan is in the order:
            //  - Compatible with the last plan in the section
            //  - Allowed i.e. some restriction could be applied
            //    to reduce the search space, for example it
            //    could consider only plans that have identical
            //    block configurations etc.
            //  - Allocated i.e. there is space in SRAM to accomodate
            //    all the buffers required by the plan
            for (Plan& plan : plans)
            {
                // Make a copy of the allocator since every plan needs to have its own,
                // each potential section won't allocate from the same allocator.
                SramAllocator alloc(m_Caps.GetTotalSramSize() / m_Caps.GetNumberOfSrams());
                // A list of PLE kernels that have been loaded into the SRAM
                // for this section. Once loaded, a PLE kernel will remain
                // in the SRAM as kernel reload is deemed to be costly.
                // The list is updated whenever a new kernel is encountered.
                PleOperations pleOps = {};
                SectionContext context{ alloc, pleOps, {} };

                // Allocation requirement are different for start of section
                if (!IsPlanAllocated(context, plan, nullptr, StatsType::StartSection))
                {
                    continue;
                }

                // Start counting total agents at 0 becasue this is the start of a section
                uint32_t totalAgents = 0;
                if (!IsSectionSizeSupported(StatsType::StartSection, plan, totalAgents))
                {
                    continue;
                }

                Combination head(part, std::move(plan), m_PartOrderTable[part.GetPartId()].first);

                // Options to be estimated: consider continuing and ending the current section
                // in the next part
                Combination ended     = EndSection(nextPart, part, head, context, currNumWeightStripes,
                                               hasSectionDoubleBuffered, totalAgents);
                Combination continued = ContinueSection(nextPart, part, head, context, currNumWeightStripes,
                                                        hasSectionDoubleBuffered, totalAgents);
                Combinations options  = { result, continued, ended };
                result                = GetBestCombinationSafe(options);
            }
        }
    }

    return result;
}

// This is a single part not merged with any other part.
// It does not need to check if the plan is compatible
// with the available SRAM since only valid plans are generated.
//
// - - - ---            -----------------------------            --- - - -
//          |          |                             |          |
//          |          |           -------           |          |
//          |  ------  |  ------  |       |  ------  |  ------  |
//          |-| DRAM |-|-| SRAM |-|   Y   |-| SRAM |-|-| DRAM |-|
//          |  ------  |  ------  |       |  ------  |  ------  |
//          |          |           -------           |          |
//          |          |                             |          |
// - - - ---            -----------------------------            --- - - -
//                                    ^
//                                    |
//                            Single part section
//
Combination Combiner::SinglePartSection(const BasePart& part)
{
    UpdateStats(StatsType::SinglePartSection);

    // Check if this Part can double buffer.
    // By default, no double buffering is performed.
    uint32_t currNumWeightStripesMax = g_NumWeightStripesMin;
    if (part.CanDoubleBufferWeights())
    {
        currNumWeightStripesMax = g_NumWeightStripesMax;
    }

    Combinations options;

    // Double buffering is performed on a per Section basis, i.e. either the entire Section double buffers weights
    // (if the Parts allow it) or the Section single buffers weights. This double buffering is considered when the
    // Part being evaluated can be double buffered.
    for (uint32_t currNumWeightStripes = g_NumWeightStripesMin; currNumWeightStripes <= currNumWeightStripesMax;
         ++currNumWeightStripes)
    {
        Plans plans =
            part.GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, currNumWeightStripes);

        for (Plan& plan : plans)
        {
            SramAllocator alloc(m_Caps.GetTotalSramSize() / m_Caps.GetNumberOfSrams());
            PleOperations pleOps = {};
            SectionContext context{ alloc, pleOps, {} };

            if (!IsPlanAllocated(context, plan, nullptr, StatsType::SinglePartSection))
            {
                continue;
            }
            // Start counting total agents from 0 because this is a single part section
            uint32_t totalAgents = 0;
            if (!IsSectionSizeSupported(StatsType::SinglePartSection, plan, totalAgents))
            {
                continue;
            }
            // Glue will be added later on.
            // In this case local optimum = global optimum so
            // it can get the best plan for the part.
            Combination head(part, std::move(plan), m_PartOrderTable[part.GetPartId()].first);
            options.push_back(std::move(head));
        }
    }

    Combination result;
    // There should always be at least one valid plan, but for testability we support the case where
    // no lonely plans are valid.
    if (options.size() > 0)
    {
        BestCombinationResults bestCombinationResults = GetBestCombination(options);
        // Include the part debug tag so that we know what type of part it is, but prepend the part ID so that
        // the folders are displayed in the right order.
        DumpDebugInfo(options, bestCombinationResults, m_DebuggingContext,
                      std::string("Lonely/") + std::to_string(part.GetPartId()) + " - " + part.m_DebugTag);
        result = options[bestCombinationResults.m_BestIdx];
    }

    //  Next part in the graph
    const BasePart* nextPartGraph = GetNextPart(&part);

    if (!result.m_Elems.empty() && nextPartGraph != nullptr)
    {
        result = result + FindBestCombinationForPart(*nextPartGraph);

        // Each of it destination part will start its own new section.
        // Therefore they all need to be glued with their source.
        auto outputSlots = m_GraphOfParts.GetPartOutputs(part.GetPartId());
        for (PartOutputSlot outputSlot : outputSlots)
        {
            std::vector<PartConnection> conns;
            std::vector<PartInputSlot> inputSlots = m_GraphOfParts.GetConnectedInputSlots(outputSlot);
            for (auto inputSlot : inputSlots)
            {
                conns.push_back({ inputSlot, outputSlot });
            }
            result = GluePartToCombinationSrcToDests(part, result, conns);
        }
    }

    return result;
}

Combination Combiner::ContinueSection(const BasePart& part,
                                      const BasePart& sPart,
                                      const Combination& comb,
                                      const SectionContext& context,
                                      uint32_t prevNumWeightStripes,
                                      bool prevDoubleBuffered,
                                      uint32_t totalAgents)
{
    UpdateStats(StatsType::ContinueSection);

    // Next Part in graph that is sorted in topological order
    const BasePart* nextPartGraph = GetNextPart(&part);

    const PartId partId = part.GetPartId();
    // flag to indicate if the next part can be in the same section of the current part
    bool nextPartSameSection = false;

    if (nextPartGraph != nullptr && m_GraphOfParts.GetDestinationParts(partId).size() != 0)
    {
        nextPartSameSection = nextPartGraph->GetPartId() == m_GraphOfParts.GetDestinationParts(partId).at(0).m_PartId;
    }

    Combination result = {};

    // A part can only be in the middle of a section
    // if the next part in the sorted graph is also
    // its destination.
    // Otherwise the next part will have to start
    // a new section which is already covered
    // by EndPart(part) --- where the section
    // ends in this part.
    if (IsPartSiso(part) && nextPartSameSection)
    {
        const Plan& sPlan = GetPlanForPartFromCombination(sPart, comb);

        std::vector<PartConnection> connections =
            m_GraphOfParts.GetConnectionsBetween(sPart.GetPartId(), part.GetPartId());

        // Sanity check: section is continued. It must be the single output of
        // its source part.
        assert(connections.size() == 1);

        // SISO part:
        //
        // Try to continue this section with next part.
        // Make sure that the chosen next plan is in the order:
        //  - Compatible with the last plan in the section
        //  - Allowed i.e. some restriction could be applied
        //    to reduce the search space, for example it
        //    could consider only plans that have identical
        //    block configurations etc.
        //  - Allocated i.e. there is space in SRAM to accommodate
        //    all the buffers required by the plan

        // sanity check SISO is the only use case.
        // destination part
        assert(m_GraphOfParts.GetDestinationParts(partId).size() == 1);
        assert(m_GraphOfParts.GetPartInputs(part.GetPartId()).size() == 1 &&
               m_GraphOfParts.GetPartOutputs(part.GetPartId()).size() == 1);

        const PartConnection& connection = connections.at(0);

        ethosn::command_stream::BlockConfig blkConfig = sPlan.GetBlockConfigures(connection.m_Source);
        Buffer* sramBuffer                            = sPlan.GetOutputBuffer(connection.m_Source);

        SectionContext contextCopy = context;
        DeallocateUnusedBuffers(*sramBuffer, contextCopy);

        // Check if this Part can double buffer.
        // By default, no double buffering is performed.
        uint32_t currNumWeightStripesMax = g_NumWeightStripesMin;
        bool hasSectionDoubleBuffered    = false;
        if (part.CanDoubleBufferWeights() && !prevDoubleBuffered)
        {
            currNumWeightStripesMax = g_NumWeightStripesMax;
        }

        if (part.CanDoubleBufferWeights() || prevDoubleBuffered)
        {
            hasSectionDoubleBuffered = true;
        }

        // Double buffering is performed on a per Section basis, i.e. either the entire Section double buffers weights
        // (if the Parts allow it) or the Section single buffers weights. This double buffering is considered when the
        // Part being evaluated can be double buffered.
        for (uint32_t currNumWeightStripes = g_NumWeightStripesMin; currNumWeightStripes <= currNumWeightStripesMax;
             ++currNumWeightStripes)
        {
            // Determine which numWeightStripes to use, based on the history of double-buffering.
            // If previous Part was double-buffered, then:
            //      1. Pass that number of weightStripes during current plan generation
            //      2. Pass the same number to the next Parts, during the recursive plan generation calls.
            // Otherwise, pass the current weightStripe number from the local for-loop.
            // This is necessary, because if there was no double-buffering in the past and there is the possibility
            // to double buffer now, then multiple plans must be created for both single buffering and double buffering weights.
            uint32_t numWeightStripes = prevDoubleBuffered ? prevNumWeightStripes : currNumWeightStripes;
            Plans plans               = part.GetPlans(CascadeType::Middle, blkConfig, sramBuffer, numWeightStripes);

            for (Plan& plan : plans)
            {
                // Make a copy of the allocator since every plan needs to have its own,
                // each potential section won't allocate from the same allocator.
                SectionContext tempContext = contextCopy;

                if (!ArePlansAllowedToMerge(sPlan, plan))
                {
                    continue;
                }

                if (!IsPlanAllocated(tempContext, plan, sramBuffer, StatsType::EndSection))
                {
                    continue;
                }

                if (!IsSectionSizeSupported(StatsType::ContinueSection, plan, totalAgents))
                {
                    continue;
                }

                // Add current part and plan to the combination,
                // no glue is required. Current part is SISO and
                // has a single input/output
                StartingGlue startingGlue;
                EndingGlue endingGlue;
                startingGlue.m_ExternalConnections.m_ReplacementBuffers.insert(
                    { plan.GetInputBuffer(connection.m_Destination), sramBuffer });
                Combination section =
                    comb + Combination(part, std::move(plan), m_PartOrderTable[part.GetPartId()].first);
                section.SetStartingGlue(std::move(startingGlue), connection.m_Destination);
                section.AddEndingGlue(std::move(endingGlue), connection.m_Source);

                // Options to be estimated
                Combinations options;

                // Next one is the last part of the section
                Combination ended = EndSection(*nextPartGraph, part, section, tempContext, numWeightStripes,
                                               hasSectionDoubleBuffered, totalAgents);

                // Next one is the middle part of the section
                Combination continued = ContinueSection(*nextPartGraph, part, section, tempContext, numWeightStripes,
                                                        hasSectionDoubleBuffered, totalAgents);
                options               = { result, continued, ended };

                result = GetBestCombinationSafe(options);
            }
        }
    }

    return result;
}

// This function finds the best combination from the current part
// to the end of the graph. The result is unique given the part.
//
// The returned value of this function should be cached
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
Combination Combiner::FindBestCombinationForPartImpl(const BasePart& part)
{
    PartId partId = part.GetPartId();
    // This is going to be a new combination, so this
    // is empty initialized
    Combination result = {};

    //  Next part in the graph
    const BasePart* nextPartGraph = GetNextPart(&part);

    // A section with more than one part can only be
    // possible if two parts are in the same branch
    bool nextPartSameSection = false;
    if (!m_GraphOfParts.GetDestinationParts(partId).empty())
    {
        assert(nextPartGraph != nullptr);
        nextPartSameSection = nextPartGraph->GetPartId() == m_GraphOfParts.GetDestinationParts(partId).at(0).m_PartId;
    }

    Combination start = {};

    // There are some scenarios:
    //  - Part is Single Input Single Output i.e. SISO
    //  - Part is Single Input Multiple Output i.e. SIMO
    //  - Part is Multiple Input Multiple Output i.e. MIMO
    //  - Part is Multiple Input Sinlge Output i.e. MISO
    //  - Part is Output i.e. no next part
    //  - Part is Input i.e. SO or MO
    if (IsPartSo(part) && nextPartSameSection)
    {
        // SISO and MISO are equivalent since what counts
        // is the number of output parts which in both cases
        // is one
        assert(m_GraphOfParts.GetDestinationParts(partId).size() == 1);

        // Start of a new section
        start = StartSection(part, *nextPartGraph);
    }

    // Lonely part
    Combination lonely = SinglePartSection(part);

    Combinations options = { start, lonely };

    result = GetBestCombinationSafe(options);

    assert(result.m_Elems.count(part.GetPartId()) == 1);

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
Combination Combiner::FindBestCombinationForPart(const BasePart& part)
{
    Combination result;
    UpdateStats(StatsType::FindBestCombinationForPart);

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

Combiner::Combiner(const GraphOfParts& graphOfParts,
                   const HardwareCapabilities& caps,
                   const CompilationOptions& compilationOptions,
                   const EstimationOptions& estOpt,
                   const DebuggingContext& debuggingContext)
    : m_GraphOfParts(graphOfParts)
    , m_Caps(caps)
    , m_CompilationOptions(compilationOptions)
    , m_EstOpt(estOpt)
    , m_DebuggingContext(debuggingContext)
    , m_MergedOpGraphReady(false)
{}

bool Combiner::Visit(const BasePart* current,
                     std::vector<const BasePart*>& outSorted,
                     std::map<const BasePart*, PartState>& partStates)
{
    auto currentStateIt = partStates.find(current);
    if (currentStateIt != partStates.end())
    {
        if (currentStateIt->second == PartState::Visited)
        {
            return true;
        }
        if (currentStateIt->second == PartState::Visiting)
        {
            return false;
        }
        else
        {
            assert(false);
        }
    }

    partStates[current] = PartState::Visiting;

    std::vector<PartOutputSlot> srcParts = m_GraphOfParts.GetSourceParts(current->GetPartId());

    for (auto& srcPart : srcParts)
    {
        Visit(&m_GraphOfParts.GetPart(srcPart.m_PartId), outSorted, partStates);
    }

    partStates[current] = PartState::Visited;

    outSorted.push_back(current);
    return true;
}

bool Combiner::TopologicalSortParts()
{
    // sort the parts in topological order

    if (m_GraphOfParts.m_Parts.size() == 0)
    {
        return true;
    }

    std::vector<const BasePart*> targets;

    // Sort starts from the output parts
    for (auto&& part : m_GraphOfParts.m_Parts)
    {
        if (m_GraphOfParts.GetDestinationParts(part->GetPartId()).size() == 0)
        {
            targets.push_back(part.get());
        }
    }

    std::map<const BasePart*, PartState> partState;
    std::vector<const BasePart*> sortedParts;

    for (auto& target : targets)
    {
        if (!Visit(target, sortedParts, partState))
        {
            return false;
        }
    }

    assert(sortedParts.size() == m_GraphOfParts.m_Parts.size());

    m_FirstPartAfterSort = sortedParts.at(0);
    assert(m_FirstPartAfterSort != nullptr);

    m_PartOrderTable.resize(sortedParts.size());

    // Sanity check although impossible
    assert(sortedParts.size() < g_InvalidCombRank);

    // Index: Part ID
    // Table content: (order, pointer to the next part)
    size_t loop;
    for (loop = 0; loop < (sortedParts.size() - 1); ++loop)
    {
        m_PartOrderTable[sortedParts[loop]->GetPartId()] = std::make_pair(loop, sortedParts[loop + 1]);
    }
    m_PartOrderTable[sortedParts[loop]->GetPartId()] = std::make_pair(loop, nullptr);

    return true;
}

void Combiner::Run()
{
    using namespace ethosn::utils;
    if (m_DebuggingContext.m_DebugInfo.m_DumpDebugFiles >= CompilationOptions::DebugLevel::High)
    {
        MakeDirectory(m_DebuggingContext.GetAbsolutePathOutputFileName("Lonely").c_str());
    }

    TopologicalSortParts();

    assert(IsPartInput(*m_FirstPartAfterSort));
    m_BestCombination = m_BestCombination + FindBestCombinationForPart(*m_FirstPartAfterSort);

    m_MergedOpGraphForBestCombination = GetOpGraphForCombination(m_BestCombination, m_GraphOfParts);

    m_MergedOpGraphReady = true;
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
    std::unordered_map<Buffer*, Buffer*> mergedBuffers;
    auto getEffectiveBuffer = [&mergedBuffers](Buffer* b) {
        auto it = mergedBuffers.find(b);
        return it != mergedBuffers.end() ? it->second : b;
    };

    assert(combination.m_PartIdsInOrder.size() == combination.m_Elems.size());

    // Add each Elem, one at a time. It is assumed that these are toplogically sorted, so we can assume that all
    // parts used as input to each part have already been processed.
    for (auto& partId : combination.m_PartIdsInOrder)
    {
        auto elemIt = combination.m_Elems.find(partId);
        assert(elemIt != combination.m_Elems.end());
        const Plan& plan = *elemIt->second.m_Plan;

        // Add any starting glues for each incoming edge of this Part
        const std::unordered_map<PartInputSlot, std::shared_ptr<StartingGlue>>& startingGlues =
            elemIt->second.m_StartingGlues;
        std::vector<PartInputSlot> inputSlots = parts.GetPartInputs(partId);
        for (PartInputSlot& inputSlot : inputSlots)
        {
            const StartingGlue* glue = startingGlues.at(inputSlot).get();
            result.MergeOpGraph(glue->m_Graph);
        }

        const std::unordered_map<PartOutputSlot, std::shared_ptr<EndingGlue>>& endingGlues =
            elemIt->second.m_EndingGlues;

        // Add buffers from the plan
        for (Buffer* b : plan.m_OpGraph.GetBuffers())
        {
            // Check if the buffer needs special treatment - if it is an input or output from this plan,
            // and the glue states that it needs replacing with something else then we shouldn't add this buffer
            // at all.
            auto inputSlotIt = plan.m_InputMappings.find(b);
            if (inputSlotIt != plan.m_InputMappings.end())
            {
                // Get the glue for this input buffer
                const StartingGlue* glue = startingGlues.at(inputSlotIt->second).get();
                // Look up the buffer replacement, if there is one
                auto bufferIt = glue->m_ExternalConnections.m_ReplacementBuffers.find(b);
                if (bufferIt != glue->m_ExternalConnections.m_ReplacementBuffers.end())
                {
                    // Don't add the buffer, just record it as being merged with its replacement.
                    mergedBuffers[b] = bufferIt->second;
                    continue;
                }
            }

            auto outputSlotIt = plan.m_OutputMappings.find(b);
            if (outputSlotIt != plan.m_OutputMappings.end())
            {
                // Get the glue for this output buffer
                const EndingGlue* glue = endingGlues.at(outputSlotIt->second).get();
                // Look up the buffer replacement, if there is one
                auto bufferIt = glue->m_ExternalConnections.m_ReplacementBuffers.find(b);
                if (bufferIt != glue->m_ExternalConnections.m_ReplacementBuffers.end())
                {
                    // Don't add the buffer, just record it as being merged with its replacement.
                    mergedBuffers[b] = bufferIt->second;
                    continue;
                }
            }

            // Normal buffer (not replaced with anything), just add it
            result.AddBuffer(b);
        }
        // Add Ops from the Plan
        for (Op* o : plan.m_OpGraph.GetOps())
        {
            result.AddOp(o);
        }

        // Add any ending glues to the OpGraph.
        // This must be done before we do any connections within the plan because we might need to connect
        // to buffers that are contained in the EndingGlue (merged buffers)
        // Note that the order of iteration here needs to be deterministic because we may add some Ops
        // to the OpGraph (and these need to be added in a consistent order).
        // Therefore we don't use plan.m_OutputMappings directly, as it does not have a deterministic order.
        std::vector<PartOutputSlot> outputSlots = parts.GetPartOutputs(partId);
        // GetPartOutputs will return duplicate values if the output slot has multiple connections.
        // The below logic requires not to have duplicates, so we remove these first.
        auto newEnd = std::unique(outputSlots.begin(), outputSlots.end());
        outputSlots.resize(std::distance(outputSlots.begin(), newEnd));
        for (auto outputSlot : outputSlots)
        {
            const EndingGlue* glue = endingGlues.at(outputSlot).get();
            result.MergeOpGraph(glue->m_Graph);
        }

        // Connect the starting glue to the previous plan (and/or its ending glue),
        // and the starting glue to the current plan.
        for (PartInputSlot& inputSlot : inputSlots)
        {
            // Get the glue for the input buffer
            const StartingGlue* glue = startingGlues.at(inputSlot).get();
            // Connect the plan, the starting glue and the previous plan's ending glue together.
            for (std::pair<Buffer*, Op*> bufAndOp : glue->m_ExternalConnections.m_BuffersToOps)
            {
                result.AddConsumer(getEffectiveBuffer(bufAndOp.first), bufAndOp.second, 0);
            }
            for (std::pair<Op*, Buffer*> opAndBuffer : glue->m_ExternalConnections.m_OpsToBuffers)
            {
                result.SetProducer(getEffectiveBuffer(opAndBuffer.second), opAndBuffer.first);
            }
        }

        // Add internal connections (within the Plan), noting that some buffers will have been merged and
        // that we need to make the connection to the correct one.
        for (Buffer* b : plan.m_OpGraph.GetBuffers())
        {
            for (auto producer : plan.m_OpGraph.GetProducers(b))
            {
                result.AddProducer(getEffectiveBuffer(b), producer);
            }

            for (auto consumer : plan.m_OpGraph.GetConsumers(b))
            {
                result.AddConsumer(getEffectiveBuffer(b), consumer.first, consumer.second);
            }
        }

        // Connect the ending glues to the current plan
        for (auto outputSlot : outputSlots)
        {
            EndingGlue* glue = endingGlues.at(outputSlot).get();
            // Connect the ending glue to the plan
            for (std::pair<Buffer*, Op*> bufAndOp : glue->m_ExternalConnections.m_BuffersToOps)
            {
                result.AddConsumer(getEffectiveBuffer(bufAndOp.first), bufAndOp.second, 0);
            }
            for (std::pair<Op*, Buffer*> opAndBuffer : glue->m_ExternalConnections.m_OpsToBuffers)
            {
                result.SetProducer(getEffectiveBuffer(opAndBuffer.second), opAndBuffer.first);
            }
        }
    }

    return result;
}

}    // namespace support_library
}    // namespace ethosn
