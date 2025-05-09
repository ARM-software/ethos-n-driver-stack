//
// Copyright © 2021-2024 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "CombinerDFS.hpp"

#include "BufferManager.hpp"
#include "Estimation.hpp"
#include "EstimationUtils.hpp"
#include "Plan.hpp"
#include "SramAllocator.hpp"
#include "StripeHelper.hpp"
#include "ThreadPool.hpp"
#include "Utils.hpp"

#include <ethosn_utils/Filesystem.hpp>
#include <ethosn_utils/Strings.hpp>

#include <algorithm>
#include <fstream>
#include <future>

using namespace ethosn::utils;

namespace ethosn
{
namespace support_library
{

using namespace utils;

namespace
{

constexpr uint32_t g_NumWeightStripesMin = 1;
constexpr uint32_t g_NumWeightStripesMax = 2;

}    // namespace

Combination::Combination()
    : m_PartIdOffset(0)
    , m_Metric(std::numeric_limits<double>::max())
{}

Combination::Combination(PartId partId, Plan&& plan)
    : m_PartIdOffset(partId)
    , m_Elems{ Elem{ std::make_shared<Plan>(std::move(plan)), {}, {} } }
    , m_Metric(0)
{}

Combination Combination::operator+(const Combination& rhs) const
{
    // If either Combination is invalid, propagate this to an invalid result. This means that if we fail to
    // find a valid plan for some case, this error is propagated upwards.
    if (rhs.IsEmpty() || this->IsEmpty())
    {
        return {};
    }

    // Part IDs must be contiguous between the LHS and RHS
    assert(this->GetEndPartId() == rhs.GetFirstPartId());

    Combination result = *this;
    result.m_Elems.insert(result.m_Elems.end(), rhs.m_Elems.begin(), rhs.m_Elems.end());
    result.m_Metric += rhs.m_Metric;

    return result;
}

void Combination::SetEndingGlue(EndingGlue&& glue, PartOutputSlot outputSlot)
{
    Elem& elem  = m_Elems[outputSlot.m_PartId - m_PartIdOffset];
    auto result = elem.m_EndingGlues.insert({ outputSlot, std::make_shared<EndingGlue>(std::move(glue)) });
    assert(result.second);    // Glue should only be set once
    ETHOSN_UNUSED(result);
}

void Combination::SetStartingGlue(StartingGlue&& glue, PartInputSlot inputSlot)
{
    Elem& elem  = m_Elems[inputSlot.m_PartId - m_PartIdOffset];
    auto result = elem.m_StartingGlues.insert({ inputSlot, std::make_shared<StartingGlue>(std::move(glue)) });
    assert(result.second);    // Glue should only be set once
    ETHOSN_UNUSED(result);
}

bool Combination::IsEmpty() const
{
    return m_Elems.empty();
}

PartId Combination::GetFirstPartId() const
{
    return m_PartIdOffset;
}

PartId Combination::GetEndPartId() const
{
    return static_cast<PartId>(m_PartIdOffset + m_Elems.size());
}

Elem& Combination::GetElem(PartId partId)
{
    assert(partId >= m_PartIdOffset);
    return m_Elems[partId - m_PartIdOffset];
}

const Elem& Combination::GetElem(PartId partId) const
{
    assert(partId >= m_PartIdOffset);
    return m_Elems[partId - m_PartIdOffset];
}

double Combination::GetMetric() const
{
    return m_Metric;
}

void Combination::SetMetric(double metric)
{
    m_Metric = metric;
}

Combiner::Combiner(const FrozenGraphOfParts& graphOfParts,
                   const HardwareCapabilities& caps,
                   const CompilationOptions& compilationOptions,
                   const EstimationOptions& estOpt,
                   const DebuggingContext& debuggingContext)
    : m_GraphOfParts(graphOfParts)
    , m_Caps(caps)
    , m_CompilationOptions(compilationOptions)
    , m_EstOpt(estOpt)
    , m_DebuggingContext(debuggingContext)
{}

void Combiner::DumpDebugInfo(const Combinations& combs,
                             const Combiner::BestCombinationResults& bestCombinationResults,
                             const std::string& folder)
{
    if (m_DebuggingContext.m_DebugInfo.m_DumpDebugFiles >= CompilationOptions::DebugLevel::High)
    {
        MakeDirectory(m_DebuggingContext.GetAbsolutePathOutputFileName(folder).c_str());

        for (size_t i = 0; i < combs.size(); ++i)
        {
            std::string prefix    = i == bestCombinationResults.m_BestIdx ? "(BEST) " : "";
            std::string subfolder = folder + "/" + prefix + std::to_string(i);
            MakeDirectory(m_DebuggingContext.GetAbsolutePathOutputFileName(subfolder).c_str());

            if (!combs[i].IsEmpty())
            {
                m_DebuggingContext.Save(
                    CompilationOptions::DebugLevel::None, subfolder + "/Detailed.dot",
                    [&](std::ofstream& s) { SaveCombinationToDot(combs[i], s, DetailLevel::High); });

                m_DebuggingContext.Save(CompilationOptions::DebugLevel::None, subfolder + "/EstimatedDetailed.dot",
                                        [&](std::ofstream& s) {
                                            SaveEstimatedOpGraphToDot(bestCombinationResults.m_OpGraphs[i],
                                                                      bestCombinationResults.m_EstimatedOpGraphs[i], s,
                                                                      DetailLevel::High, {}, {}, {});
                                        });
            }
        }
    }
}

// Check if there is sufficient SRAM for plan to fit
// into the SRAM allocation for the combination that
// is compatible with the plan, and makes those allocations.
bool Combiner::AllocateSram(SectionContext& context,
                            PartId partId,
                            const Plan& plan,
                            const std::vector<Buffer*>& outputBuffersOfPrevPlans) const
{
    // Some plans (e.g. from ConcatPart) do their own SRAM allocation, as the algorithm here
    // makes some assumptions which are sub-optimal
    if (plan.m_IsPreallocated)
    {
        return true;
    }
    constexpr uint32_t numBytesPerBeat = 16;
    PleKernelInfo pleKernelInfo        = plan.GetPleKernelInfo(m_Caps);
    uint32_t pleKernelSize             = 0;
    bool newPleKernel                  = false;
    bool isSramAllocated               = true;

    using Allocated = std::pair<bool, uint32_t>;
    Allocated bufferAllocated, pleKernelAllocated;
    SramAllocator localAlloc = context.alloc;

    // To get more benefit from preloading of weights from later layers, we want to minimise the overlap of SRAM
    // buffers with earlier buffers, even when those buffers are no longer being used. This will allow the
    // command stream generation to detect that the later buffers can be loaded early, leading to faster inferences.
    // The following is a very simple strategy to achieve this, which is to alternately allocate new buffers at the
    // start and end of the available SRAM space.
    AllocationPreference allocPref = (partId % 2 == 0) ? AllocationPreference::Start : AllocationPreference::End;

    if (pleKernelInfo.m_PleOp != nullptr)
    {
        // If PLE kernel of the current plan is already used by previous part of the same
        // section, then its size is not counted.

        auto CheckPleKernel = [&pleKernelInfo](const std::pair<command_stream::PleKernelId, uint32_t>& plePair) {
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
            pleKernelAllocated = localAlloc.Allocate(pleKernelSize, allocPref, pleKernelInfo.m_PleOp->m_DebugTag,
                                                     numBytesPerBeat * m_Caps.GetNumberOfSrams());

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

        while (buffersIterator != buffers.end())
        {
            Buffer* const buf         = *buffersIterator;
            const uint32_t bufferSize = buf->m_SizeInBytes;

            if (buf->m_Location == Location::Sram)
            {
                // If an input buffer is in start of a section, or it's other buffer (i.e output buffer) in start/continue/end of section
                if (inputBuffersMapping.count(buf) == 0 || outputBuffersOfPrevPlans.empty() ||
                    outputBuffersOfPrevPlans[inputBuffersMapping.at(buf).m_InputIndex] == nullptr)
                {
                    assert(bufferSize != 0);

                    bufferAllocated = localAlloc.Allocate(bufferSize / m_Caps.GetNumberOfSrams(), allocPref,
                                                          buf->m_DebugTag, numBytesPerBeat * m_Caps.GetNumberOfSrams());

                    isSramAllocated = bufferAllocated.first;

                    if (isSramAllocated == true)
                    {
                        buf->Sram()->m_Offset                 = bufferAllocated.second;
                        context.allocatedBuffers[buf->Sram()] = { partId };
                    }
                    else
                    {
                        break;
                    }
                }
                // Input buffer, but in a continue or end section so we just copy the address of the output buffer
                // from the incoming part.
                else
                {
                    buf->Sram()->m_Offset =
                        outputBuffersOfPrevPlans[inputBuffersMapping.at(buf).m_InputIndex]->Sram()->m_Offset.value();
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

Combination Combiner::AddTempGlues(const Combination& combination) const
{
    Combination result              = combination;
    const FrozenGraphOfParts& parts = m_GraphOfParts;
    for (PartId partId = result.GetFirstPartId(); partId < result.GetEndPartId(); ++partId)
    {
        Elem& elem       = result.GetElem(partId);
        const Plan& plan = *elem.m_Plan;

        const std::vector<PartInputSlot>& inputSlots = parts.GetPartInputs(partId);
        const std::unordered_map<PartInputSlot, std::shared_ptr<StartingGlue>>& startingGlues = elem.m_StartingGlues;
        // All parts needs starting glues in order to be estimated / create an opgraph
        for (const PartInputSlot& inputSlot : inputSlots)
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
                    BufferFormat dramFormat = impl::GetBestDramBufferFormat({ buffer->Sram() }, m_CompilationOptions,
                                                                            { partId }, m_DebuggingContext);

                    std::unique_ptr<DramBuffer> dramBuffer = DramBuffer::Build()
                                                                 .AddFormat(dramFormat)
                                                                 .AddDataType(buffer->m_DataType)
                                                                 .AddTensorShape(buffer->m_TensorShape)
                                                                 .AddQuantization(buffer->m_QuantizationInfo)
                                                                 .AddBufferType(BufferType::Intermediate);

                    Buffer* dramBufferRaw = dramBuffer.get();
                    auto dma              = std::make_unique<DmaOp>(buffer->m_Format);
                    DmaOp* dmaRaw         = dma.get();
                    startingGlue->m_Graph.AddBuffer(std::move(dramBuffer));
                    startingGlue->m_Graph.AddOp(std::move(dma));
                    startingGlue->m_Graph.AddConsumer(dramBufferRaw, dmaRaw, 0);
                    startingGlue->m_ExternalConnections.m_OpsToBuffers.insert({ dmaRaw, buffer });
                }
                elem.m_StartingGlues.insert({ inputSlot, startingGlue });
            }
        }

        const std::vector<PartOutputSlot>& outputSlots = parts.GetPartOutputs(partId);
        const std::unordered_map<PartOutputSlot, std::shared_ptr<EndingGlue>>& endingGlues = elem.m_EndingGlues;
        for (const PartOutputSlot& outputSlot : outputSlots)
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
                    BufferFormat dramFormat = impl::GetBestDramBufferFormat({ buffer->Sram() }, m_CompilationOptions,
                                                                            { partId }, m_DebuggingContext);

                    std::unique_ptr<DramBuffer> dramBuffer = DramBuffer::Build()
                                                                 .AddFormat(dramFormat)
                                                                 .AddDataType(buffer->m_DataType)
                                                                 .AddTensorShape(buffer->m_TensorShape)
                                                                 .AddQuantization(buffer->m_QuantizationInfo)
                                                                 .AddBufferType(BufferType::Intermediate);

                    Buffer* dramBufferRaw = dramBuffer.get();
                    auto dma              = std::make_unique<DmaOp>(buffer->m_Format);
                    DmaOp* dmaRaw         = dma.get();
                    endingGlue->m_Graph.AddBuffer(std::move(dramBuffer));
                    endingGlue->m_Graph.AddOp(std::move(dma));
                    endingGlue->m_Graph.SetProducer(dramBufferRaw, dmaRaw);
                    endingGlue->m_ExternalConnections.m_BuffersToOps.insert({ buffer, dmaRaw });
                }
                elem.m_EndingGlues.insert({ outputSlot, endingGlue });
            }
        }
    }
    return result;
}

Combiner::EstimatedCombination Combiner::EstimateCombination(const Combination& comb) const
{
    // Add temporary glues to partial combinations so we can estimate performance
    Combination combinationWithTempGlues = AddTempGlues(comb);
    OpGraph combiOpGraph                 = GetOpGraphForCombination(combinationWithTempGlues, m_GraphOfParts);

    // Estimate the combination we're considering
    EstimatedOpGraph estimatedOpGraph = ethosn::support_library::EstimateOpGraph(combiOpGraph, m_Caps, m_EstOpt);

    return { combinationWithTempGlues, combiOpGraph, estimatedOpGraph };
}

Combiner::BestCombinationResults Combiner::EstimateAndChooseBestCombination(const Combinations& combs) const
{
    assert(combs.size() > 0);

    BestCombinationResults result;
    utils::Optional<size_t> bestIdx;
    utils::Optional<double> bestMetric;
    for (size_t i = 0; i < combs.size(); ++i)
    {
        const Combination& combination            = combs[i];
        EstimatedCombination estimatedCombination = EstimateCombination(combination);

        if (m_DebuggingContext.m_DebugInfo.m_DumpDebugFiles >= CompilationOptions::DebugLevel::High)
        {
            result.m_EstimatedOpGraphs.push_back(estimatedCombination.m_EstimatedOpGraph);
            result.m_OpGraphs.push_back(estimatedCombination.m_OpGraph);
            result.m_CompletedCombinations.push_back(estimatedCombination.m_CombinationWithTempGlues);
        }

        if (!bestIdx.has_value() || estimatedCombination.m_EstimatedOpGraph.m_Metric < bestMetric.value())
        {
            bestIdx    = i;
            bestMetric = estimatedCombination.m_EstimatedOpGraph.m_Metric;
        }
    }
    result.m_BestIdx    = bestIdx.value();
    result.m_BestMetric = bestMetric.value();
    return result;
}

const Combination& Combiner::GetBestCombination() const
{
    return m_BestCombination;
}

OpGraph Combiner::GetMergedOpGraphForBestCombination() const
{
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
        BufferFormat dramFormat = source->m_Location == Location::Dram ? source->m_Format : dest->m_Format;
        auto dma                = std::make_unique<DmaOp>(dramFormat);
        sourceDma               = dma.get();
        destDma                 = dma.get();
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
Combination
    Combiner::GluePartToCombinationSrcToDests(const BasePart& sPart, const Combination& comb, uint32_t outputSlotIdx)
{
    Combination result = comb;

    PartOutputSlot outputSlot{ sPart.GetPartId(), outputSlotIdx };
    // Find element belonging to source part in the combination
    const Plan& sourcePlan = *comb.GetElem(sPart.GetPartId()).m_Plan;
    // Find the output buffer of the source node.
    // Note all destination nodes are branched off from the same source node
    Buffer* producedBuffer = sourcePlan.GetOutputBuffer(outputSlot);
    assert(producedBuffer != nullptr);

    // Find the input buffers in the destination plans
    std::set<PartId> debugPartIds = { sPart.GetPartId() };
    std::vector<std::pair<PartInputSlot, Buffer*>> consumerBuffers;
    for (const PartInputSlot& inputSlot : m_GraphOfParts.GetConnectedInputSlots(outputSlot))
    {
        const BasePart& part   = m_GraphOfParts.GetPart(inputSlot.m_PartId);
        const Plan& plan       = *comb.GetElem(part.GetPartId()).m_Plan;
        Buffer* consumerBuffer = plan.GetInputBuffer(inputSlot);
        assert(consumerBuffer != nullptr);
        consumerBuffers.push_back(std::make_pair(inputSlot, consumerBuffer));
        debugPartIds.insert(part.GetPartId());
    }

    // Sort the consumers so that DRAM consumers are processed first. This is because these buffers could be re-used
    // as part of the glue for other consumers, so we avoid having to create as many new buffers (and thus make a more efficient
    // graph).
    // Note that a stable sort is used, so that the order is deterministic when there are multiple SRAM or DRAM consumers.
    std::stable_sort(consumerBuffers.begin(), consumerBuffers.end(),
                     [](std::pair<PartInputSlot, Buffer*> a, std::pair<PartInputSlot, Buffer*> b) {
                         return (a.second->m_Location == Location::Dram) > (b.second->m_Location == Location::Dram);
                     });

    // Maintain a set of DRAM buffers that are available for use in the glue.
    // These are used if possible, rather than adding new buffers.
    std::map<BufferFormat, DramBuffer*> dramBuffers;
    if (producedBuffer->m_Location == Location::Dram)
    {
        dramBuffers[producedBuffer->m_Format] = producedBuffer->Dram();
    }

    EndingGlue endingGlue;    // We'll populate this as we go with any ending glue for the source part

    // Adds a new DRAM buffer of the given format to the ending glue, so that it can be used in any starting glues of consumers.
    // Also adds the DmaOps to connect this buffer to where it is copied from.
    auto addNewBuffer = [&](BufferFormat format, Buffer* copiedFrom) {
        std::unique_ptr<DramBuffer> dramBuffer = DramBuffer::Build()
                                                     .AddFormat(format)
                                                     .AddDataType(producedBuffer->m_DataType)
                                                     .AddTensorShape(producedBuffer->m_TensorShape)
                                                     .AddQuantization(producedBuffer->m_QuantizationInfo)
                                                     .AddBufferType(BufferType::Intermediate);

        DramBuffer* dramBufferRaw = dramBuffer.get();
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
    auto getOrAddCompatibleDramBuffer = [&](const std::initializer_list<const SramBuffer*>& sramBuffers) {
        // First check if we have an existing buffer that is usable, to avoid adding any more
        for (std::pair<BufferFormat, DramBuffer*> formatAndBuffer : dramBuffers)
        {
            if (std::all_of(sramBuffers.begin(), sramBuffers.end(), [&](const SramBuffer* b) {
                    return impl::IsSramBufferCompatibleWithDramBuffer(*b, *formatAndBuffer.second, { 0, 0, 0, 0 });
                }))
            {
                return formatAndBuffer.second;
            }
        }
        // Need to add a new buffer of a compatible format.
        BufferFormat format =
            impl::GetBestDramBufferFormat(sramBuffers, m_CompilationOptions, debugPartIds, m_DebuggingContext);
        DramBuffer* newBuffer = addNewBuffer(format, producedBuffer);
        return newBuffer;
    };

    // Go through every consumer and connect it up with appropriate glue.
    for (std::pair<PartInputSlot, Buffer*> consumerBufferPair : consumerBuffers)
    {
        PartInputSlot inputSlot = consumerBufferPair.first;
        Buffer* consumerBuffer  = consumerBufferPair.second;

        StartingGlue startingGlue;    // We will fill this in with any starting glue that this consumer needs

        // Consider each case of Sram/Dram producer/consumer separately.
        // Although there is some overlap between these cases, this was found to be the least confusing approach.
        if (producedBuffer->m_Location == Location::Sram && consumerBuffer->m_Location == Location::Dram)
        {
            // There might already be an existing DRAM buffer of the right format, so we can avoid adding anything.
            // This can only be done for intermediate buffers though, as outputs need to have their own buffer
            auto dramBufferIt = dramBuffers.find(consumerBuffer->m_Format);
            if (dramBufferIt != dramBuffers.end() && consumerBuffer->Dram()->m_BufferType == BufferType::Intermediate)
            {
                // Re-use this existing buffer by adding a replacement link
                startingGlue.m_ExternalConnections.m_ReplacementBuffers[consumerBuffer] = dramBufferIt->second;
            }
            else
            {
                // We might be able to add a single DMA to copy directly from the producer buffer,
                Buffer* bufferToCopyFrom = producedBuffer;
                if (!impl::IsSramBufferCompatibleWithDramBuffer(*producedBuffer->Sram(), *consumerBuffer->Dram(),
                                                                { 0, 0, 0, 0 }))
                {
                    // If the SRAM buffer is not compatible though, then we'll need to do a conversion.
                    // We may be lucky and there is already a DRAM buffer that is compatible that we can copy from, or we may need to add a new one.
                    bufferToCopyFrom = getOrAddCompatibleDramBuffer({ producedBuffer->Sram() });
                }

                // We could re-use this consumer DRAM buffer for other consumers, to save them doing their own conversion.
                // Only intermediate buffers can be shared though (Outputs, for example, don't allow reading).
                if (consumerBuffer->Dram()->m_BufferType == BufferType::Intermediate)
                {
                    // In order for DRAM buffers in consuming plans to be available for sharing, a new copy of this buffer
                    // must be made in the ending glue of the producer, and then linked to the existing consumer buffer via a replacement.
                    Buffer* replacementBuffer = addNewBuffer(consumerBuffer->m_Format, bufferToCopyFrom);
                    startingGlue.m_ExternalConnections.m_ReplacementBuffers[consumerBuffer] = replacementBuffer;
                }
                else
                {
                    // This consumer buffer can't be re-used, so just copy from the buffer we chose above in the starting glue.
                    // Note that we put the DmaOp in the ending glue not the starting glue, so that the data is copied out of
                    // SRAM as soon as possible (before any branching).
                    // If the new buffer is being copied from the original producedBuffer, then the connections to the DmaOp
                    // need to be in the external connections of the ending glue (as they connect something in the glue to something
                    // in the plan). Otherwise we assume the `copiedFrom` buffer is part of the ending glue, and so it needs an internal
                    // connection.
                    GlueConnections* connections =
                        (bufferToCopyFrom == producedBuffer) ? &endingGlue.m_ExternalConnections : nullptr;
                    AddCopyBetweenBuffers(endingGlue.m_Graph, bufferToCopyFrom, connections, consumerBuffer,
                                          &startingGlue.m_ExternalConnections, m_Caps);
                }
            }
        }
        else if (producedBuffer->m_Location == Location::Dram && consumerBuffer->m_Location == Location::Sram)
        {
            // We might be able to add a single DMA to copy directly from the producer buffer,
            Buffer* dramBufferToCopyFrom = producedBuffer;
            if (!impl::IsSramBufferCompatibleWithDramBuffer(*consumerBuffer->Sram(), *producedBuffer->Dram(),
                                                            { 0, 0, 0, 0 }))
            {
                // If the SRAM buffer is not compatible though, then we'll need to do a conversion.
                // We may be lucky and there is already a DRAM buffer that is compatible that we can copy from, or we may need to add a new one.
                dramBufferToCopyFrom = getOrAddCompatibleDramBuffer({ consumerBuffer->Sram() });
            }

            // Add a DMA to the starting glue, to copy from the chosen DRAM buffer.
            AddCopyBetweenBuffers(startingGlue.m_Graph, dramBufferToCopyFrom, &startingGlue.m_ExternalConnections,
                                  consumerBuffer, &startingGlue.m_ExternalConnections, m_Caps);
        }
        else if (producedBuffer->m_Location == Location::Sram && consumerBuffer->m_Location == Location::Sram)
        {
            // SRAM to SRAM always needs to go via DRAM (note that this isn't a cascade!).
            // We may be lucky and there is already a DRAM buffer that is compatible that we can copy from, or we may need to add a new one.
            Buffer* dramBufferToCopyFrom =
                getOrAddCompatibleDramBuffer({ producedBuffer->Sram(), consumerBuffer->Sram() });
            // Add a DMA to the starting glue, to copy from the chosen DRAM buffer.
            AddCopyBetweenBuffers(startingGlue.m_Graph, dramBufferToCopyFrom, &startingGlue.m_ExternalConnections,
                                  consumerBuffer, &startingGlue.m_ExternalConnections, m_Caps);
        }
        else if (producedBuffer->m_Location == Location::Dram && consumerBuffer->m_Location == Location::Dram)
        {
            // There might already be an existing DRAM buffer of the right format, so we can avoid adding anything.
            // This can only be done for intermediate buffers though, as outputs need to have their own buffer
            auto dramBufferIt = dramBuffers.find(consumerBuffer->m_Format);
            if (dramBufferIt != dramBuffers.end() && consumerBuffer->Dram()->m_BufferType == BufferType::Intermediate)
            {
                // Re-use this existing buffer by adding a replacement link
                startingGlue.m_ExternalConnections.m_ReplacementBuffers[consumerBuffer] = dramBufferIt->second;
            }
            // In the case that consumerBuffer is an output buffer, it can't be a simple replacement of the
            // producedBuffer, but we might be able to make a new "merged" buffer that is an output buffer,
            // and replace both with this new buffer.
            // Merging gets complicated if we have multiple consumers, as the merging may invalidate other
            // decisions. Therefore we only do this for simple single-consumer cases at the moment.
            else if (consumerBuffers.size() == 1 && consumerBuffer->Dram()->m_BufferType == BufferType::Output &&
                     producedBuffer->Dram()->m_BufferType == BufferType::Intermediate &&
                     consumerBuffer->m_Format == producedBuffer->m_Format &&
                     consumerBuffer->m_QuantizationInfo == producedBuffer->m_QuantizationInfo &&
                     consumerBuffer->m_TensorShape == producedBuffer->m_TensorShape &&
                     consumerBuffer->m_SizeInBytes == producedBuffer->m_SizeInBytes)
            {
                std::unique_ptr<DramBuffer> mergedBuffer =
                    DramBuffer::Build()
                        .AddFormat(consumerBuffer->m_Format)
                        .AddDataType(consumerBuffer->m_DataType)
                        .AddTensorShape(consumerBuffer->m_TensorShape)
                        .AddQuantization(consumerBuffer->m_QuantizationInfo)
                        .AddBufferType(consumerBuffer->Dram()->m_BufferType)
                        .AddSizeInBytes(consumerBuffer->m_SizeInBytes)
                        .AddDebugTag("Merged " + consumerBuffer->m_DebugTag)
                        .AddOperationId(consumerBuffer->Dram()->m_OperationId)
                        .AddProducerOutputIndex(consumerBuffer->Dram()->m_ProducerOutputIndx);

                Buffer* mergedBufferRaw = endingGlue.m_Graph.AddBuffer(std::move(mergedBuffer));

                // Mark both buffers as being replaced by the new merged buffer (the other is done later)
                endingGlue.m_ExternalConnections.m_ReplacementBuffers.insert({ producedBuffer, mergedBufferRaw });
                startingGlue.m_ExternalConnections.m_ReplacementBuffers.insert({ consumerBuffer, mergedBufferRaw });
            }
            // We could re-use this consumer DRAM buffer for other consumers, to save them doing their own conversion.
            // Only intermediate buffers can be shared though (Outputs, for example, don't allow reading).
            else if (consumerBuffer->Dram()->m_BufferType == BufferType::Intermediate)
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
        result.SetStartingGlue(std::move(startingGlue), inputSlot);
    }

    result.SetEndingGlue(std::move(endingGlue), outputSlot);
    return result;
}

void Combiner::DeallocateUnusedBuffers(PartId partId,
                                       const PartOutputMapping& planOutputBuffers,
                                       const std::vector<PartId>& consumingPartIds,
                                       SectionContext& context)
{
    // If the output buffer(s) from the plan contains the full tensor (either in SRAM like in
    // a strategy 1/3 cascade or in DRAM), then we can safely free everything else in SRAM.
    bool allOutputBuffersFullTensor = true;
    for (auto x : planOutputBuffers)
    {
        Buffer* b = x.first;
        if (!b->IsFullTensor())
        {
            allOutputBuffersFullTensor = false;
        }
    }

    // Pass on the ownership to consumers of this part - they will handle deallocation when they are finished with it
    // (recursively). This is important for e.g. strategy 0 cascades, as we can't free the buffers
    // until the end of the section. We will keep passing on responsibility for the buffers down the cascade,
    // accumulating more and more, until the end at which they will all be deallocated (see below block).
    // For some cases like strategy 1 into strategy 3 cascading, buffers may be freed partway through a cascade.
    std::vector<SramBuffer*> buffersToRemove;
    for (std::pair<SramBuffer* const, std::set<PartId>>& bufferAndOwners : context.allocatedBuffers)
    {
        if (bufferAndOwners.second.count(partId) > 0)
        {
            bool isOutputBuffer = planOutputBuffers.count(bufferAndOwners.first) > 0;
            if (!allOutputBuffersFullTensor || isOutputBuffer)
            {
                for (PartId consumingPartId : consumingPartIds)
                {
                    bufferAndOwners.second.insert(consumingPartId);
                }
            }

            // Decrement ref count on all allocated buffers which we were a user of
            // If we were the last user this will actually free it after this loop
            bufferAndOwners.second.erase(partId);
            if (bufferAndOwners.second.empty())
            {
                buffersToRemove.push_back(bufferAndOwners.first);
            }
        }
    }

    for (SramBuffer* b : buffersToRemove)
    {
        context.allocatedBuffers.erase(b);
        context.alloc.Free(b->m_Offset.value());
    }
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
Combination Combiner::ChooseBestLonelyPlan(const BasePart& part)
{
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
        Plans plans = part.GetPlans(CascadeType::Lonely, BlockConfig{}, {}, currNumWeightStripes);

        for (Plan& plan : plans)
        {
            SramAllocator alloc(m_Caps.GetTotalSramSize() / m_Caps.GetNumberOfSrams());
            PleOperations pleOps = {};
            SectionContext context{ {}, alloc, pleOps, {}, 0, false, {}, BlockConfig{} };

            if (!AllocateSram(context, part.GetPartId(), plan, {}))
            {
                continue;
            }
            // Glue will be added later on.
            // In this case local optimum = global optimum so
            // it can get the best plan for the part.
            Combination head(part.GetPartId(), std::move(plan));
            options.push_back(std::move(head));
        }
    }

    Combination result;
    // There should always be at least one valid plan, but for testability we support the case where
    // no lonely plans are valid.
    if (options.size() > 0)
    {
        BestCombinationResults bestCombinationResults = EstimateAndChooseBestCombination(options);
        // Include the part debug tag so that we know what type of part it is, but prepend the part ID so that
        // the folders are displayed in the right order.
        DumpDebugInfo(options, bestCombinationResults,
                      std::string("Lonely/") + std::to_string(part.GetPartId()) + " - " + part.m_DebugTag);
        result = options[bestCombinationResults.m_BestIdx];
        result.SetMetric(bestCombinationResults.m_BestMetric);
    }

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
std::vector<SectionContext> Combiner::StartSection(const BasePart& part)
{
    std::vector<SectionContext> result = {};

    // Check if this Part can double buffer.
    // By default, no double buffering is performed.
    uint32_t currNumWeightStripesMax = g_NumWeightStripesMin;
    bool hasSectionDoubleBuffered    = false;
    if (part.CanDoubleBufferWeights())
    {
        currNumWeightStripesMax  = g_NumWeightStripesMax;
        hasSectionDoubleBuffered = true;
    }

    std::vector<PartConnection> outgoingEdges = m_GraphOfParts.GetDestinationConnections(part.GetPartId());
    std::vector<PartId> consumingParts;
    for (PartConnection c : outgoingEdges)
    {
        consumingParts.push_back(c.m_Destination.m_PartId);
    }

    // Double buffering is performed on a per Section basis, i.e. either the entire Section double buffers weights
    // (if the Parts allow it) or the Section single buffers weights. This double buffering is considered when the
    // Part being evaluated can be double buffered.
    for (uint32_t currNumWeightStripes = g_NumWeightStripesMin; currNumWeightStripes <= currNumWeightStripesMax;
         ++currNumWeightStripes)
    {
        Plans plans = part.GetPlans(CascadeType::Beginning, BlockConfig{}, {}, currNumWeightStripes);

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

            // Default to 16x16 block if this plan doesn't have one. This means the rest of the section
            // will never consider other block sizes which isn't ideal.
            BlockConfig blockConfig =
                plan.m_BlockConfig.has_value() ? plan.m_BlockConfig.value() : BlockConfig{ 16u, 16u };
            SectionContext context{
                {}, alloc, {}, {}, currNumWeightStripes, hasSectionDoubleBuffered, {}, blockConfig
            };

            // Allocation requirement are different for start of section
            if (!AllocateSram(context, part.GetPartId(), plan, {}))
            {
                continue;
            }
            DeallocateUnusedBuffers(part.GetPartId(), plan.m_OutputMappings, consumingParts, context);

            for (const PartConnection& connection : m_GraphOfParts.GetDestinationConnections(part.GetPartId()))
            {
                context.unresolvedOutputs.insert({ connection, plan.GetOutputBuffer(connection.m_Source) });
            }

            context.comb = Combination(part.GetPartId(), std::move(plan));

            result.push_back(context);
        }
    }

    return result;
}

std::vector<SectionContext> Combiner::ContinueSection(const BasePart& part, const SectionContext& context)
{
    return ContinueOrEndSection(false, part, context);
}

std::vector<SectionContext> Combiner::EndSection(const BasePart& part, const SectionContext& context)
{
    return ContinueOrEndSection(true, part, context);
}

std::vector<SectionContext>
    Combiner::ContinueOrEndSection(bool isEnd, const BasePart& part, const SectionContext& context)
{
    std::vector<SectionContext> result = {};

    // Check if this Part can double buffer.
    // By default, no double buffering is performed.
    uint32_t currNumWeightStripesMax = g_NumWeightStripesMin;
    bool hasSectionDoubleBuffered    = false;
    if (!isEnd)
    {
        // ContinueSection
        if (part.CanDoubleBufferWeights() && !context.hasSectionDoubleBuffered)
        {
            currNumWeightStripesMax = g_NumWeightStripesMax;
        }

        if (part.CanDoubleBufferWeights() || context.hasSectionDoubleBuffered)
        {
            hasSectionDoubleBuffered = true;
        }
    }
    else
    {
        // EndSection
        if (part.CanDoubleBufferWeights() && !context.hasSectionDoubleBuffered)
        {
            currNumWeightStripesMax = g_NumWeightStripesMax;
        }
    }

    SectionContext contextCopy = context;

    // Resolve the output buffers of any previous parts that are used as inputs by this part
    std::vector<Buffer*> sramBufferInputs;
    std::vector<PartConnection> sourceConnection = m_GraphOfParts.GetSourceConnections(part.GetPartId());
    sramBufferInputs.resize(sourceConnection.size(), nullptr);
    bool anyInputSramBuffers = false;
    for (const PartConnection& connection : sourceConnection)
    {
        // Since we visit the parts in topological order, all connections should be available.
        // If one isn't available it means that it isn't in this section.
        auto bufferIt = contextCopy.unresolvedOutputs.find(connection);
        if (bufferIt != contextCopy.unresolvedOutputs.end())
        {
            sramBufferInputs[connection.m_Destination.m_InputIndex] = bufferIt->second;
            contextCopy.unresolvedOutputs.erase(connection);
            anyInputSramBuffers = true;
        }
    }

    // It might be that this part isn't connected to the existing section at all, which is considered
    // invalid. In future it might be possible to support this (e.g. a section with two separate DRAM
    // inputs which then merge together).
    if (!anyInputSramBuffers)
    {
        return result;
    }

    if (isEnd)
    {
        // We prevent ending sections if there are any unresolved buffers, because by definition
        // this would not be the end of a section.
        if (contextCopy.unresolvedOutputs.size() != 0)
        {
            return result;
        }
    }

    CascadeType planType = isEnd ? CascadeType::End : CascadeType::Middle;

    std::vector<PartConnection> outgoingEdges = m_GraphOfParts.GetDestinationConnections(part.GetPartId());
    std::vector<PartId> consumingParts;
    for (PartConnection c : outgoingEdges)
    {
        consumingParts.push_back(c.m_Destination.m_PartId);
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
        uint32_t numWeightStripes =
            context.hasSectionDoubleBuffered ? context.currNumWeightStripes : currNumWeightStripes;
        Plans plans = part.GetPlans(planType, context.blockConfig, sramBufferInputs, numWeightStripes);

        if (planType == CascadeType::Middle)
        {
            // We shouldn't generate multiple plans here, as it could lead to an explosion of combinations.
            if (plans.size() > 1)
            {
                throw InternalErrorException("Multiple Middle plans generated - could lead to combinatorial explosion");
            }
        }

        for (Plan& plan : plans)
        {
            // Make a copy of the allocator since every plan needs to have its own -
            // each potential section won't allocate from the same allocator.
            SectionContext tempContext           = contextCopy;
            tempContext.hasSectionDoubleBuffered = hasSectionDoubleBuffered;
            tempContext.currNumWeightStripes     = numWeightStripes;

            if (!AllocateSram(tempContext, part.GetPartId(), plan, sramBufferInputs))
            {
                continue;
            }
            DeallocateUnusedBuffers(part.GetPartId(), plan.m_OutputMappings, consumingParts, tempContext);

            if (!isEnd)
            {
                for (const PartConnection& connection : m_GraphOfParts.GetDestinationConnections(part.GetPartId()))
                {
                    tempContext.unresolvedOutputs.insert({ connection, plan.GetOutputBuffer(connection.m_Source) });
                }
            }

            // Remember the input buffers of the plan before we std::move it away.
            std::unordered_map<PartInputSlot, Buffer*> planInputBuffers;
            for (PartConnection c : m_GraphOfParts.GetSourceConnections(part.GetPartId()))
            {
                planInputBuffers[c.m_Destination] = plan.GetInputBuffer(c.m_Destination);
            }

            tempContext.comb = context.comb + Combination(part.GetPartId(), std::move(plan));

            // Add empty glues for cascaded inputs
            for (PartConnection c : sourceConnection)
            {
                if (sramBufferInputs[c.m_Destination.m_InputIndex] != nullptr)
                {
                    StartingGlue startingGlue;
                    EndingGlue endingGlue;
                    startingGlue.m_ExternalConnections.m_ReplacementBuffers.insert(
                        { planInputBuffers.at(c.m_Destination), sramBufferInputs[c.m_Destination.m_InputIndex] });
                    tempContext.comb.SetStartingGlue(std::move(startingGlue), c.m_Destination);
                    // Multiple parts can share the same source (i.e. a branch), and we can't add the glue twice (even though it's empty)
                    if (tempContext.comb.GetElem(c.m_Source.m_PartId).m_EndingGlues.count(c.m_Source) == 0)
                    {
                        tempContext.comb.SetEndingGlue(std::move(endingGlue), c.m_Source);
                    }
                }
            }

            // Once a section is finished, we can estimate performance for it
            if (isEnd)
            {
                EstimatedCombination estimatedCombination = EstimateCombination(tempContext.comb);
                tempContext.comb.SetMetric(estimatedCombination.m_EstimatedOpGraph.m_Metric);
            }

            result.push_back(tempContext);
        }
    }

    return result;
}

void Combiner::Run(ThreadPool& threadPool)
{
    if (m_DebuggingContext.m_DebugInfo.m_DumpDebugFiles >= CompilationOptions::DebugLevel::High)
    {
        ethosn::utils::MakeDirectory(m_DebuggingContext.GetAbsolutePathOutputFileName("Lonely").c_str());
    }

    const int numParts = static_cast<int>(m_GraphOfParts.GetParts().size());

    // Kick off all stage 1 weight encoding asynchronously for maximum parallelism.
    // We can't do this inside ChooseBestLonelyPlan, because that is being run on the worker threads
    // and we can't queue background work from a worker thread (see ThreadPool implementation).
    {
        auto startTime = std::chrono::high_resolution_clock::now();

        for (int partIdx = 0; partIdx < numParts; ++partIdx)
        {
            m_GraphOfParts.GetPart(partIdx).PreprocessWeightsAsync();
        }

        auto duration = std::chrono::high_resolution_clock::now() - startTime;
        g_Logger.Debug("PreprocessWeightsAsync (kick-off): %llu ms", duration.count() / (1000ULL * 1000ULL));
    }

    // Find the best lonely plan for each part. This is done up front so can be all done in parallel
    // with each other, as each part is independent.
    std::vector<Combination> bestLonely(numParts, Combination{});
    {
        auto startTime = std::chrono::high_resolution_clock::now();

        std::vector<std::future<void>> waitHandles(numParts);
        for (int partIdx = 0; partIdx < numParts; ++partIdx)
        {
            waitHandles[partIdx] = threadPool.AddToQueue(
                [&](int partIdx) { bestLonely[partIdx] = ChooseBestLonelyPlan(m_GraphOfParts.GetPart(partIdx)); },
                partIdx);
        }
        for (const auto& h : waitHandles)
        {
            h.wait();
        }

        auto duration = std::chrono::high_resolution_clock::now() - startTime;
        g_Logger.Debug("ChooseBestLonelyPlans: %llu ms", duration.count() / (1000ULL * 1000ULL));
    }

    // Find the best sections of each length, for each different starting part.
    // This is done up front so can be all done in parallel with each other, as sections from each starting part are independent
    // so we can do in parallel
    std::vector<std::vector<Combination>> sectionsOfAllLengthsForStartingPart(numParts);
    {
        auto startTime = std::chrono::high_resolution_clock::now();
        // Loop until the second to last part as the last will never be the start of a section.
        std::vector<std::future<void>> waitHandles(numParts - 1);
        for (int partIdx = 0; partIdx < numParts - 1; ++partIdx)
        {
            waitHandles[partIdx] = threadPool.AddToQueue(
                [&](int partIdx) {
                    sectionsOfAllLengthsForStartingPart[partIdx] =
                        CalculateSectionsOfAllLengths(m_GraphOfParts.GetPart(partIdx));
                },
                partIdx);
        }
        for (const auto& h : waitHandles)
        {
            h.wait();
        }

        // Dump best section if debug enabled
        const char* env = std::getenv("ETHOSN_SUPPORT_LIBRARY_DEBUG_PART_IDS");
        if (env && strlen(env) > 0)
        {
            for (std::string partIdString : ethosn::utils::Split(env, ","))
            {
                PartId partId = std::stoi(ethosn::utils::Trim(partIdString));

                std::string folder = "Sections";
                MakeDirectory(m_DebuggingContext.GetAbsolutePathOutputFileName(folder).c_str());
                folder += "/" + std::to_string(partId);
                MakeDirectory(m_DebuggingContext.GetAbsolutePathOutputFileName(folder).c_str());

                for (uint32_t sectionLength = 0; sectionLength < sectionsOfAllLengthsForStartingPart[partId].size();
                     ++sectionLength)
                {
                    if (!sectionsOfAllLengthsForStartingPart[partId][sectionLength].IsEmpty())
                    {
                        EstimatedCombination estimatedCombination =
                            EstimateCombination(sectionsOfAllLengthsForStartingPart[partId][sectionLength]);
                        m_DebuggingContext.Save(CompilationOptions::DebugLevel::None,
                                                folder + "/Length" + std::to_string(sectionLength) + ".dot",
                                                [&](std::ofstream& s) {
                                                    SaveEstimatedOpGraphToDot(estimatedCombination.m_OpGraph,
                                                                              estimatedCombination.m_EstimatedOpGraph,
                                                                              s, DetailLevel::High, {}, {}, {});
                                                });
                    }
                }
            }
        }

        auto duration = std::chrono::high_resolution_clock::now() - startTime;
        g_Logger.Debug("SectionsOfAllLengthsForStartingPart: %llu ms", duration.count() / (1000ULL * 1000ULL));
    }

    // We iterate through all possible (and valid) combinations of lonely (L), start (S), continue (C) and end (E)
    // sections for every Part, and pick the one with the best performance.
    // This is done in a deliberately non-recursive manner to aid
    // debugging and performance profiling, and also it was found to run faster than a recursive solution,
    // and avoids stack overflows from large networks.
    // There is a lot of repetition between different combinations, which we exploit by avoiding re-calculating
    // things that we've already done to keep compilation times down.
    // We treat the parts as a simple list indexed from 0 to n, ignoring any branching/graph structure.
    // This keeps the algorithm here simple, but should still allow us to make sections across branches in the future.

    // This array will be filled in with the best solution for the "tail" of the graph from the given part
    // onwards. For example in a graph with 4 parts (0,1,2,3), then best[1] will be filled in with the best
    // combination for parts 1,2 and 3 which will be one of LLL, LSE, SEL, SCE.
    // We fill this in reverse order, starting with the shortest tail.
    // Note we have an extra empty element at the end to avoid having to do a bounds check when the section length is
    // the full size of the graph
    std::vector<Combination> best(numParts + 1, Combination{});

    // The best combination for the final part can only be lonely, so fill this in immediately.
    assert(numParts >= 1);
    best[numParts - 1] = bestLonely[numParts - 1];

    // Now consider longer tails, working our way up from the shortest
    auto startTime = std::chrono::high_resolution_clock::now();
    for (int partIdx = numParts - 2; partIdx >= 0; --partIdx)
    {
        g_Logger.Verbose("Combiner progress: %u/%u", (numParts - partIdx), numParts);

        // Options for this tail are:
        //   - L followed by the best for the rest of the tail, which we will have just calculated on the previous iteration
        //   - SE followed by the best for the rest of the tail, which we will have just calculated on the previous-but-one iteration
        //   - SCE ..
        //   - SCCE ..
        //   - ...
        //   - SCC...CCE which will be entirely new and needs calculating
        //
        // We calculate the total metric for each of these, and pick the best

        Combination L = bestLonely[partIdx] + best[partIdx + 1];

        // Assume for now that L is the best, we'll replace this as necessary
        Combination bestTail = L;

        // Retrieve the SE, SCE, SCCE, etc., combinations (calculated up-front),
        // and check the performance of each of them when combined with the rest of the tail.
        const std::vector<Combination>& sections = sectionsOfAllLengthsForStartingPart[partIdx];
        if (sections.size() >= 2)
        {
            for (int sectionLength = 2; sectionLength <= numParts - partIdx; ++sectionLength)
            {
                const Combination& section = sections.at(sectionLength);
                if (section.IsEmpty())
                {
                    // No valid section of this length could be found. That doesn't mean that longer ones won't work though,
                    // so keep checking the longer lengths.
                    continue;
                }
                Combination sectionAndRest = section + best[partIdx + sectionLength];

                // Check if this is the new best
                if (sectionAndRest.GetMetric() < bestTail.GetMetric())
                {
                    bestTail = sectionAndRest;
                }
            }
        }

        // Store the best combination from this part onwards - we'll re-use this for all the longer tails.
        best[partIdx] = bestTail;
    }

    auto duration = std::chrono::high_resolution_clock::now() - startTime;
    g_Logger.Debug("FindingBest: %llu ms", duration.count() / (1000ULL * 1000ULL));

    // The best combination for the whole graph is simply the one where the tail is the whole graph.
    m_BestCombination = best[0];
    if (m_BestCombination.IsEmpty())
    {
        throw InternalErrorException((std::string("Failed to find a valid combination!").c_str()));
    }

    // Add glues at section boundaries - these are only needed at the end as they don't affect any of the other decisions
    for (PartId p = 0; p < static_cast<PartId>(numParts); ++p)
    {
        const std::vector<PartOutputSlot>& outputSlots = m_GraphOfParts.GetPartOutputs(p);
        for (uint32_t outputIndex = 0; outputIndex < outputSlots.size(); ++outputIndex)
        {
            if (m_BestCombination.GetElem(p).m_EndingGlues.find(PartOutputSlot{ p, outputIndex }) ==
                m_BestCombination.GetElem(p).m_EndingGlues.end())
            {
                m_BestCombination =
                    GluePartToCombinationSrcToDests(m_GraphOfParts.GetPart(p), m_BestCombination, outputIndex);
            }
        }
    }

    m_MergedOpGraphForBestCombination = GetOpGraphForCombination(m_BestCombination, m_GraphOfParts);
}

/// Starting from the given part, generates the best section of each possible length.
std::vector<Combination> Combiner::CalculateSectionsOfAllLengths(const BasePart& startingPart)
{
    const int numParts = static_cast<int>(m_GraphOfParts.GetParts().size());

    // Initialize result with empty/invalid combinations, for every possible section length.
    // We'll replace these with valid combinations if/when we find them.
    std::vector<Combination> best(static_cast<size_t>(numParts - startingPart.GetPartId() + 1), Combination{});

    // This stores the state for what is essentially an iterative implementation of a recursive algorithm.
    // This was found to be faster than the recursive approach, and is easier to debug and analyze performance.
    // The outer vector is for each part, with the first being the startingPart, and this will grow and shrink
    // as we go deeper into the tail and then come out again.
    // The inner vector is the list of plans generated for that part, given all the previous plans in the previous parts.
    // Plans are removed from the list once they have been considered (i.e. we have already looked deeper into the graph to continue
    // this section as far as we can, and therefore have no further use for it)
    //
    // Example partway through the algorithm (assuming the startingPart is part 0):
    //
    //       ----------> outer vector
    //      |
    //      |         Part 0         Part 1         Part 2
    //      |        =========      ========       ========
    //      |
    //      |          S0-0           C1-0           C2-0
    //      v          S0-1           C1-1           C2-1
    // inner vector    S0-2                          C2-2
    //
    //
    // With this state, we are looking at starting plan number 3 for Part 0 (S0-3) (note this is the one after the end,
    // and having already looked at the higher numbered ones which have already been removed from the list),
    // and continue plan 2 for Part 1 (C1-2) (having already looked at and removed the later ones, and removed the current one),
    // and continue plan 2 for Part 2 (C2-2) (having already looked at and removed the later ones, and NOT YET removed the current one).
    // We'll remove C2-2 from the list as we're about to process it.
    // With these three plans in our context, we're then looking at Part 3 and we will generate ends plans,
    // choose the best one, and store this in our result.
    // We'll also generate continue plans and add a new column at the right with all of these, and then
    // move on to Part 4, which will then be considering the last of these new continue plans ("recursing").
    // Once we exhaust the list of possible plans for a part, we'll go to the previous part and consider the next
    // plan there (this is like 'returning' in the recursive version).
    std::vector<std::vector<SectionContext>> contexts;

    // Start by generate all possible starting plans for the first Part.
    // We reverse the order so that the order in which we consider plans is the same as an older version of the
    // Combiner code. This is relevant when multiple plans have the same metric, as it determines which is preferred.
    std::vector<SectionContext> startingPlans = StartSection(startingPart);
    std::reverse(startingPlans.begin(), startingPlans.end());
    contexts.push_back(std::move(startingPlans));

    uint32_t numIterations = 0;
    while (!contexts.empty())
    {
        ++numIterations;
        if (contexts.back().empty())
        {
            // No more plans to consider for the previous part, so go back to the previous one so we can pick the next plan there
            contexts.pop_back();
            continue;
        }

        // The current part we're looking at, relative to the starting part. This is always the one immediately after the last "column" (see diagram above)
        int partIdxOffset = static_cast<int>(contexts.size());
        PartId partId     = startingPart.GetPartId() + partIdxOffset;

        // Take the next plan to consider from the previous part, removing it from the list
        SectionContext c = std::move(contexts.back().back());
        contexts.back().pop_back();

        // Try ending the section on this part, storing the best option
        Combination& bestOfThisLength        = best[partIdxOffset + 1];
        std::vector<SectionContext> endPlans = EndSection(m_GraphOfParts.GetPart(partId), c);
        for (SectionContext& endPlan : endPlans)
        {
            if (bestOfThisLength.IsEmpty() || endPlan.comb.GetMetric() < bestOfThisLength.GetMetric())
            {
                bestOfThisLength = endPlan.comb;
            }
        }

        // Generate all the continue plans and add these into a new "column" (see above diagram), so we can "recurse" into the next part
        if (partId < numParts - 1U)
        {
            std::vector<SectionContext> continuePlans = ContinueSection(m_GraphOfParts.GetPart(partId), c);
            if (continuePlans.empty())
            {
                // If the section has gotten too long (e.g. not enough SRAM), no point adding the empty vector then immediately popping it off
                continue;
            }
            contexts.push_back(continuePlans);
        }
    }

    g_Logger.Verbose("CalculateSectionsOfAllLengths: %u iterations", numIterations);

    return best;
}

// Take in input a combination and generate an OpGraph.
// This is used in:
//  - Combiner logic:   it needs to estimate the combination and this is done on an
//                      OpGraph in order to select the best combination between two
//                      or more
//  - Estimation logic: it can only estimate OpGraphs and not raw combinations.
OpGraph GetOpGraphForCombination(const Combination& combination, const FrozenGraphOfParts& parts)
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

    // Add each Elem, one at a time. It is assumed that these are toplogically sorted, so we can assume that all
    // parts used as input to each part have already been processed.
    for (PartId partId = combination.GetFirstPartId(); partId < combination.GetEndPartId(); ++partId)
    {
        const Elem& elem = combination.GetElem(partId);
        const Plan& plan = *elem.m_Plan;

        // Add any starting glues for each incoming edge of this Part
        const std::unordered_map<PartInputSlot, std::shared_ptr<StartingGlue>>& startingGlues = elem.m_StartingGlues;
        const std::vector<PartInputSlot>& inputSlots = parts.GetPartInputs(partId);
        for (const PartInputSlot& inputSlot : inputSlots)
        {
            const StartingGlue* glue = startingGlues.at(inputSlot).get();
            result.MergeOpGraph(glue->m_Graph);
        }

        const std::unordered_map<PartOutputSlot, std::shared_ptr<EndingGlue>>& endingGlues = elem.m_EndingGlues;

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
        const std::vector<PartOutputSlot>& outputSlots = parts.GetPartOutputs(partId);
        for (auto outputSlot : outputSlots)
        {
            const EndingGlue* glue = endingGlues.at(outputSlot).get();
            result.MergeOpGraph(glue->m_Graph);
        }

        // Connect the starting glue to the previous plan (and/or its ending glue),
        // and the starting glue to the current plan.
        for (const PartInputSlot& inputSlot : inputSlots)
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
