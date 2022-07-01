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
                   std::vector<size_t> stats,
                   const DebuggingContext& debuggingContext,
                   const std::string folder)
{
    using namespace ethosn::utils;
    if (debuggingContext.m_DebugInfo.m_DumpDebugFiles >= CompilationOptions::DebugLevel::High)
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

            if (!comb.m_Elems.empty())
            {
                debuggingContext.SaveCombinationToDot(CompilationOptions::DebugLevel::None, comb,
                                                      subfolder + "/Detailed.dot", DetailLevel::High);
            }

            ++combinationNumber;
            if (combinationNumber > debuggingContext.GetMaxNumDumps())
            {
                break;
            }
        }
    }
}

bool MatchingBlocks(const Plan& planProducer, const Plan& planConsumer, Buffer* produced, Buffer* consumed)
{
    size_t matching = 0;

    Op* opProducer = planProducer.m_OpGraph.GetProducer(produced);
    if (!opProducer)
    {
        // There is no producer for this buffer
        return true;
    }

    const auto producerBlockConfig = opProducer->GetBlockConfig();

    if (!producerBlockConfig.has_value())
    {
        // It's something else that does not have
        // the concept of block config
        return true;
    }

    auto consumers = planConsumer.m_OpGraph.GetConsumers(consumed);
    for (auto& consumer : consumers)
    {
        Op* opConsumer                 = consumer.first;
        const auto consumerBlockConfig = opConsumer->GetBlockConfig();

        if (!consumerBlockConfig.has_value())
        {
            // It's something else that does not have
            // the concept of block config
            ++matching;
        }
        // If here producerBlockConfig is not empty, while
        // consumerBlockConfig is empty if matching has been
        // already incremented in the else above, there is
        // no risk of incrementing matching twice
        else if (producerBlockConfig.value() == consumerBlockConfig.value())
        {
            ++matching;
        }
    }
    return matching == consumers.size();
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

bool Combiner::AreMceOperationsCompatible(const Buffer* plan1OutputBuffer,
                                          const Buffer* plan2InputBuffer,
                                          const PartOutputSlot& outputSlot) const
{
    const auto& part = m_GraphOfParts.GetPart(outputSlot.m_PartId);
    auto mceOp       = part.GetMceOperation();
    if ((mceOp.has_value()) && (plan1OutputBuffer->m_Location != Location::Dram))
    {
        const TensorShape& inputBufferShape = plan2InputBuffer->m_TensorShape;
        const TensorShape& inputStripeShape = plan2InputBuffer->m_StripeShape;

        if ((mceOp == ethosn::command_stream::MceOperation::CONVOLUTION) ||
            (mceOp == ethosn::command_stream::MceOperation::FULLY_CONNECTED))
        {
            if (GetChannels(inputStripeShape) < GetChannels(inputBufferShape))
            {
                return false;
            }
        }
    }
    return true;
}

bool Combiner::AreBlockConfigsCompatible(const Plan& plan1, const Plan& plan2, const PartOutputSlot& outputSlot) const
{
    Buffer* bufferProduced = plan1.GetOutputBuffer(outputSlot);
    auto inputSlots        = m_GraphOfParts.GetConnectedInputSlots(outputSlot);
    assert(inputSlots.size() == 1);
    const PartInputSlot& inputSlot = inputSlots[0];
    Buffer* bufferConsumed         = plan2.GetInputBuffer(inputSlot);

    const bool areBuffersInPleInputSram =
        bufferProduced->m_Location == Location::PleInputSram && bufferConsumed->m_Location == Location::PleInputSram;

    if (areBuffersInPleInputSram)
    {
        return MatchingBlocks(plan1, plan2, bufferProduced, bufferConsumed);
    }
    return true;
}

bool Combiner::ArePlansCompatibleImpl(const Plan& sPlan, const Plan& dPlan, const PartConnection& slots) const
{
    const PartInputSlot& inputSlot   = slots.m_Destination;
    const PartOutputSlot& outputSlot = slots.m_Source;
    const Buffer* planInputBuffer    = dPlan.GetInputBuffer(inputSlot);
    const Buffer* sPlanOutputBuffer  = sPlan.GetOutputBuffer(outputSlot);

    // two plans should be connected along the edge we were told about.
    if (sPlanOutputBuffer == nullptr || planInputBuffer == nullptr)
    {
        return false;
    }

    // Note that m_QuantizationInfo does not need to match between the buffers, as it is possible to *reinterpret* the
    // quantisation of a buffer without having to insert any glue (i.e. it's a no-op). We will use this to implement the
    // ReinterpretQuantization Operation.

    // The same goes for shape, but only in limited circumstances (e.g. you can't reinterpret a 1x1x1x1 as a 1x100x100x100
    // because there wouldn't be enough data, and there are probably additional limitations for non-linear formats like
    // NHWCB, FCAF). For now we are conservative and only allow this for simple NHWC cases where the full tensor is
    // reinterpreted with a different shape, which we use to implement "DRAM Reshape" Operations as a no-op.
    bool areShapesDifferent = sPlanOutputBuffer->m_TensorShape != planInputBuffer->m_TensorShape;
    bool isValidNhwcReinterpret =
        sPlanOutputBuffer->m_Format == CascadingBufferFormat::NHWC &&
        planInputBuffer->m_Format == CascadingBufferFormat::NHWC &&
        GetNumElements(sPlanOutputBuffer->m_TensorShape) == GetNumElements(planInputBuffer->m_TensorShape);

    bool areBuffersIncompatible = areShapesDifferent && !isValidNhwcReinterpret;

    if (areBuffersIncompatible)
    {
        return false;
    }

    // Check if the buffers on the boundary are compatible, i.e. the same (or similar enough that they can be reinterpreted),
    // such that the plans could be directly merged without any additional DMA ops required. Both locations must
    // be on SRAM.
    bool areOrdersEquivalent;
    if (sPlanOutputBuffer->m_Location == Location::Sram && planInputBuffer->m_Location == Location::Sram)
    {
        const uint32_t numStripesOutputZ = utils::DivRoundUp(GetChannels(sPlanOutputBuffer->m_TensorShape),
                                                             GetChannels(sPlanOutputBuffer->m_StripeShape));
        const uint32_t numStripesInputZ =
            utils::DivRoundUp(GetChannels(planInputBuffer->m_TensorShape), GetChannels(planInputBuffer->m_StripeShape));
        areOrdersEquivalent =
            (sPlanOutputBuffer->m_Order == planInputBuffer->m_Order) ||
            (numStripesInputZ == 1 && numStripesOutputZ == 1 &&
             (sPlanOutputBuffer->m_Order == TraversalOrder::Xyz || sPlanOutputBuffer->m_Order == TraversalOrder::Zxy) &&
             (planInputBuffer->m_Order == TraversalOrder::Xyz || planInputBuffer->m_Order == TraversalOrder::Zxy));
    }
    else
    {
        areOrdersEquivalent = sPlanOutputBuffer->m_Order == planInputBuffer->m_Order;
    }
    bool areBuffersEquivalent =
        sPlanOutputBuffer->m_Location == planInputBuffer->m_Location && planInputBuffer->m_Location != Location::Dram &&
        sPlanOutputBuffer->m_Location != Location::Dram && sPlanOutputBuffer->m_Format == planInputBuffer->m_Format &&
        sPlanOutputBuffer->m_StripeShape == planInputBuffer->m_StripeShape && areOrdersEquivalent &&
        sPlanOutputBuffer->m_SizeInBytes == planInputBuffer->m_SizeInBytes &&
        sPlanOutputBuffer->m_SlotSizeInBytes == planInputBuffer->m_SlotSizeInBytes &&
        sPlanOutputBuffer->m_NumStripes == planInputBuffer->m_NumStripes &&
        EqualPackedBoundaryData(sPlanOutputBuffer->m_PackedBoundaryThickness,
                                planInputBuffer->m_PackedBoundaryThickness) &&
        sPlanOutputBuffer->m_NumLoads == planInputBuffer->m_NumLoads;

    if ((!areBuffersEquivalent) || !AreMceOperationsCompatible(sPlanOutputBuffer, planInputBuffer, outputSlot) ||
        !AreBlockConfigsCompatible(sPlan, dPlan, outputSlot))
    {
        return false;
    }

    return true;
}

bool Combiner::ArePlansCompatible(const Plan& sPlan, const Plan& dPlan, const PartConnection& slots)
{
    return ArePlansCompatibleImpl(sPlan, dPlan, slots);
}

// Check if there is sufficient SRAM for plan to fit
// into the SRAM allocation for the combination that
// is compatible with the plan
bool Combiner::IsPlanAllocated(SramAllocator& alloc,
                               const Plan& plan,
                               PleOperations& pleOps,
                               const Buffer* const outBufOfPrevPlanInSection,
                               const StatsType sectionType) const
{
    PleKernelInfo pleKernelInfo = plan.GetPleKernelInfo(m_Caps);
    uint32_t pleKernelSize      = 0;
    bool newPleKernel           = false;
    bool isSramAllocated        = true;

    using Allocated = std::pair<bool, uint32_t>;
    Allocated bufferAllocated, pleKernelAllocated;
    SramAllocator localAlloc = alloc;

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

        auto pleIterator = std::find_if(pleOps.begin(), pleOps.end(), CheckPleKernel);

        if (pleIterator == pleOps.end())
        {
            pleKernelSize                       = pleKernelInfo.m_Size;
            newPleKernel                        = true;
            pleKernelInfo.m_PleOp->m_LoadKernel = true;
            assert(pleKernelSize != 0);
            assert(pleKernelSize <= m_Caps.GetMaxPleSize());

            // Allocate the PleKernel
            pleKernelAllocated = localAlloc.Allocate(userId, (pleKernelSize), AllocationPreference::Start);

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
                                                          AllocationPreference::Start);

                    isSramAllocated = bufferAllocated.first;

                    if (isSramAllocated == true)
                    {
                        buf->m_Offset = bufferAllocated.second;
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
        alloc = localAlloc;

        if (newPleKernel)
        {
            pleOps.push_back(std::make_pair(pleKernelInfo.m_PleOp->m_PleKernelId, pleKernelAllocated.second));
        }
    }

    return isSramAllocated;
}

bool Combiner::IsPlanInputGlueable(const Plan& plan) const
{
    for (auto inputMapping : plan.m_InputMappings)
    {
        const Buffer* buf = inputMapping.first;
        switch (buf->m_Location)
        {
            case Location::Dram:
            case Location::Sram:
                continue;
            default:
                return false;
        }
    }
    return true;
}

bool Combiner::IsPlanOutputGlueable(const Plan& plan) const
{
    for (auto outputMapping : plan.m_OutputMappings)
    {
        const Buffer* buf = outputMapping.first;
        switch (buf->m_Location)
        {
            case Location::Dram:
            case Location::Sram:
                continue;
            default:
                return false;
        }
    }
    return true;
}

bool Combiner::ArePlansAllowedToMerge(const Plan& reference, const Plan& current, const PartConnection& slots) const
{
    const PartOutputSlot& outputSlot = slots.m_Source;
    Buffer* referenceOutBuffer       = reference.GetOutputBuffer(outputSlot);
    const PartInputSlot& inputSlot   = slots.m_Destination;
    Buffer* currentInBuffer          = current.GetInputBuffer(inputSlot);

    // Plans in a section must use the same block configuration
    if (!MatchingBlocks(reference, current, referenceOutBuffer, currentInBuffer))
    {
        return false;
    }

    if (reference.m_HasIdentityPle && current.m_HasIdentityMce)
    {
        return false;
    }

    return true;
}

bool Combiner::ArePlansStreamingStrategiesCompatible(const Plan& reference,
                                                     const Plan& current,
                                                     const PartConnection& slots) const
{
    const PartInputSlot& inputSlot = slots.m_Destination;
    Buffer* currentInBuffer        = current.GetInputBuffer(inputSlot);

    // Plans in a section must use the same streaming strategy
    for (auto inputMapping : reference.m_InputMappings)
    {
        const Buffer* referenceInBuffer = inputMapping.first;
        if (currentInBuffer->m_Location != Location::Sram)
        {
            continue;
        }
        const auto refSplit   = IsSplitting(referenceInBuffer->m_TensorShape, referenceInBuffer->m_StripeShape);
        const bool refSplitH  = std::get<0>(refSplit);
        const bool refSplitW  = std::get<1>(refSplit);
        const bool refSplitC  = std::get<2>(refSplit);
        const auto currSplit  = IsSplitting(currentInBuffer->m_TensorShape, currentInBuffer->m_StripeShape);
        const bool currSplitH = std::get<0>(currSplit);
        const bool currSplitW = std::get<1>(currSplit);
        const bool currSplitC = std::get<2>(currSplit);

        if ((refSplitH != currSplitH || refSplitW != currSplitW || refSplitC != currSplitC) &&
            referenceInBuffer->m_Location != Location::Dram)
        {
            return false;
        }
    }
    return true;
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
                    // Assume NHWCB for now, we could use try and use a more optimal format
                    // NHWCB is most conservative in terms of performance and compatibility.
                    auto dramBuffer = std::make_unique<Buffer>(
                        Location::Dram, CascadingBufferFormat::NHWCB, buffer->m_TensorShape, TensorShape{ 0, 0, 0, 0 },
                        TraversalOrder::Xyz, utils::TotalSizeBytesNHWCB(buffer->m_TensorShape),
                        buffer->m_QuantizationInfo);
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
                    auto dramBuffer = std::make_unique<Buffer>(
                        Location::Dram, CascadingBufferFormat::NHWCB, buffer->m_TensorShape, TensorShape{ 0, 0, 0, 0 },
                        TraversalOrder::Xyz, utils::TotalSizeBytesNHWCB(buffer->m_TensorShape),
                        buffer->m_QuantizationInfo);
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

Combination Combiner::GetBestCombination(const Combinations& combs)
{
    utils::Optional<Combination> result;
    utils::Optional<NetworkPerformanceData> refNetPerfData;

    for (const Combination& combination : combs)
    {
        if (!combination.m_Elems.empty())
        {
            // If there has been no valid result so far just use the first one
            if (!result.has_value())
            {
                result = combination;
            }
            else
            {
                // Estimate the "result" if we haven't done it before
                // Store it so we can compare to it later
                if (!refNetPerfData.has_value())
                {
                    assert(result.has_value());
                    Combination comb     = AddTempGlues(result.value());
                    OpGraph combiOpGraph = GetOpGraphForCombination(comb, m_GraphOfParts);
                    EstimatedOpGraph estimatedOpGraph =
                        ethosn::support_library::EstimateOpGraph(combiOpGraph, m_Caps, m_EstOpt);
                    refNetPerfData = estimatedOpGraph.m_PerfData;
                }

                // Add temporary glues to partial combinations so we can estimate performance
                Combination comb     = AddTempGlues(combination);
                OpGraph combiOpGraph = GetOpGraphForCombination(comb, m_GraphOfParts);

                // Estimate the combination we're considering
                EstimatedOpGraph estimatedOpGraph =
                    ethosn::support_library::EstimateOpGraph(combiOpGraph, m_Caps, m_EstOpt);
                if (ComparePerformanceData(estimatedOpGraph.m_PerfData, refNetPerfData.value()) ==
                    PerformanceComparisonResult::LeftBetter)
                {
                    refNetPerfData = estimatedOpGraph.m_PerfData;
                    result         = combination;
                }
            }
        }
    }
    return result.has_value() ? result.value() : Combination();
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

CascadingBufferFormat Combiner::GetBestCascadingBufferDramFormat(const std::array<Buffer*, 2> sramBuffers) const
{
    using SupportedCompressedFormats = std::vector<CascadingBufferFormat>;

    constexpr size_t sramBuffersSize = sramBuffers.size();
    SupportedCompressedFormats cascadingBufferSupportedTypePerStripe[sramBuffersSize];
    for (size_t sramBufferIdx = 0; sramBufferIdx < sramBuffersSize; sramBufferIdx++)
    {
        const Buffer* currentBuffer = sramBuffers[sramBufferIdx];
        SupportedCompressedFormats& currentCascadedSupportedTypeList =
            cascadingBufferSupportedTypePerStripe[sramBufferIdx];

        if (IsCompressionFormatCompatibleWithStripeAndShape(CompilerDataCompressedFormat::FCAF_DEEP,
                                                            currentBuffer->m_StripeShape) &&
            !AnyPackedBoundaryData(currentBuffer->m_PackedBoundaryThickness))
        {
            currentCascadedSupportedTypeList.push_back(CascadingBufferFormat::FCAF_DEEP);
        }
        if (IsCompressionFormatCompatibleWithStripeAndShape(CompilerDataCompressedFormat::FCAF_WIDE,
                                                            currentBuffer->m_StripeShape) &&
            !AnyPackedBoundaryData(currentBuffer->m_PackedBoundaryThickness))
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

// Generate a simple glue between sram and dram which just contains a dma op
StartingAndEndingGlues Combiner::GenerateGlueBetweenSramAndDram(Buffer* sramBuffer,
                                                                Buffer* dramBuffer,
                                                                const CascadingBufferFormat transferFormat) const
{
    StartingAndEndingGlues result;
    auto dma      = std::make_unique<DmaOp>(transferFormat);
    DmaOp* dmaRaw = dma.get();
    EndingGlue endingGlue;
    endingGlue.m_Graph.AddOp(std::move(dma));
    endingGlue.m_ExternalConnections.m_BuffersToOps.insert({ sramBuffer, dmaRaw });
    result.m_EndingGlue = std::move(endingGlue);

    StartingGlue startingGlue;
    startingGlue.m_ExternalConnections.m_OpsToBuffers.insert({ dmaRaw, dramBuffer });
    result.m_StartingGlues.push_back(std::move(startingGlue));
    return result;
}

// Generate a simple glue between dram and sram which just contains a dma op
StartingAndEndingGlues Combiner::GenerateGlueBetweenDramAndSram(Buffer* dramBuffer,
                                                                Buffer* sramBuffer,
                                                                const CascadingBufferFormat transferFormat) const
{
    StartingAndEndingGlues result;
    auto dma      = std::make_unique<DmaOp>(transferFormat);
    DmaOp* dmaRaw = dma.get();
    EndingGlue endingGlue;
    result.m_EndingGlue = std::move(endingGlue);

    StartingGlue startingGlue;
    startingGlue.m_Graph.AddOp(std::move(dma));
    startingGlue.m_ExternalConnections.m_BuffersToOps.insert({ dramBuffer, dmaRaw });
    startingGlue.m_ExternalConnections.m_OpsToBuffers.insert({ dmaRaw, sramBuffer });
    result.m_StartingGlues.push_back(std::move(startingGlue));
    return result;
}

// Generate glue between DRAM and SRAM which includes a conversion from the source buffer into DRAM.
// DRAM --DmaOp-> SRAM --DmaOp-> DRAM (NHWCB) --DmaOp-> SRAM
StartingAndEndingGlues Combiner::GenerateGlueBetweenDramAndSramWithConversion(Buffer* inputBuffer,
                                                                              Buffer* outputBuffer) const
{
    StartingAndEndingGlues result;
    auto dma1      = std::make_unique<DmaOp>(outputBuffer->m_Format);
    DmaOp* dma1Raw = dma1.get();

    TensorShape outputShapeNHWCB = utils::RoundUpHeightAndWidthToBrickGroup(inputBuffer->m_TensorShape);
    outputShapeNHWCB[3] =
        utils::RoundUpToNearestMultiple(outputShapeNHWCB[3], utils::GetChannels(m_Caps.GetBrickGroupShape()));

    // Set the SRAM buffer's stripe size to be the smallest height and width to be the most compatible.
    // We can't split depth because if one of the buffers is NHWC that won't be compatible.
    TensorShape sramStripeShape = { 1, utils::GetHeight(m_Caps.GetBrickGroupShape()),
                                    utils::GetWidth(m_Caps.GetBrickGroupShape()),
                                    utils::GetChannels(outputShapeNHWCB) };
    auto sramBuffer             = std::make_unique<Buffer>(
        Location::Sram, CascadingBufferFormat::NHWCB, outputShapeNHWCB, sramStripeShape, TraversalOrder::Xyz,
        utils::TotalSizeBytesNHWCB(sramStripeShape), inputBuffer->m_QuantizationInfo);
    sramBuffer->m_BufferType = BufferType::Intermediate;
    sramBuffer->m_Offset     = 0;    // Nothing else should be resident in SRAM at this point, so we can use any address
    sramBuffer->m_NumStripes = 1;
    sramBuffer->m_SlotSizeInBytes = sramBuffer->m_SizeInBytes;
    Buffer* sramBufferRaw         = sramBuffer.get();

    auto dma2      = std::make_unique<DmaOp>(CascadingBufferFormat::NHWCB);
    DmaOp* dma2Raw = dma2.get();

    auto intermediateDramBuffer = std::make_unique<Buffer>(
        Location::Dram, CascadingBufferFormat::NHWCB, outputShapeNHWCB, TensorShape{ 0, 0, 0, 0 }, TraversalOrder::Xyz,
        utils::TotalSizeBytesNHWCB(m_Caps.GetBrickGroupShape()), inputBuffer->m_QuantizationInfo);
    intermediateDramBuffer->m_BufferType = BufferType::Intermediate;
    Buffer* intermediateDramBufferRaw    = intermediateDramBuffer.get();

    auto dma3      = std::make_unique<DmaOp>(inputBuffer->m_Format);
    DmaOp* dma3Raw = dma3.get();

    // We can choose to the dram buffer in the either in starting or ending
    // for now just put it in the starting glue
    // We still need an ending glue but it is empty
    EndingGlue endingGlue;
    result.m_EndingGlue = std::move(endingGlue);

    StartingGlue startingGlue;
    startingGlue.m_Graph.AddOp(std::move(dma1));
    startingGlue.m_Graph.AddOp(std::move(dma2));
    startingGlue.m_Graph.AddOp(std::move(dma3));
    startingGlue.m_Graph.AddBuffer(std::move(sramBuffer));
    startingGlue.m_Graph.SetProducer(sramBufferRaw, dma1Raw);
    startingGlue.m_Graph.AddConsumer(sramBufferRaw, dma2Raw, 0);
    startingGlue.m_Graph.AddBuffer(std::move(intermediateDramBuffer));
    startingGlue.m_Graph.SetProducer(intermediateDramBufferRaw, dma2Raw);
    startingGlue.m_Graph.AddConsumer(intermediateDramBufferRaw, dma3Raw, 0);
    startingGlue.m_ExternalConnections.m_BuffersToOps.insert({ outputBuffer, dma1Raw });
    startingGlue.m_ExternalConnections.m_OpsToBuffers.insert({ dma3Raw, inputBuffer });
    result.m_StartingGlues.push_back(std::move(startingGlue));

    return result;
}

// Generate glue between sram and sram, this should only be called if we need to generate a conversion back to dram
// as the sram buffers aren't compatible.
// For entry 3 (see table above) there are as many glues possible as the
// number of buffer formats in DRAM i.e. :
//  - NHWCB
//  - FCAF_DEEP
//  - FCAF_WIDE
//
StartingAndEndingGlues Combiner::GenerateGlueBetweenSramAndSram(Buffer* sourceBuffer,
                                                                Buffer* destBuffer,
                                                                const CascadingBufferFormat cascadingBufferFormat) const
{
    StartingAndEndingGlues result;

    auto dramBuffer = std::make_unique<Buffer>(
        Location::Dram, cascadingBufferFormat, destBuffer->m_TensorShape, TensorShape{ 0, 0, 0, 0 },
        TraversalOrder::Xyz, utils::TotalSizeBytesNHWCB(destBuffer->m_TensorShape), destBuffer->m_QuantizationInfo);
    dramBuffer->m_BufferType = BufferType::Intermediate;

    auto dma1             = std::make_unique<DmaOp>(cascadingBufferFormat);
    DmaOp* dma1Raw        = dma1.get();
    Buffer* dramBufferRaw = dramBuffer.get();
    auto dma2             = std::make_unique<DmaOp>(cascadingBufferFormat);
    DmaOp* dma2Raw        = dma2.get();
    EndingGlue endingGlue;
    endingGlue.m_Graph.AddOp(std::move(dma1));
    endingGlue.m_Graph.AddBuffer(std::move(dramBuffer));
    endingGlue.m_Graph.SetProducer(dramBufferRaw, dma1Raw);
    endingGlue.m_ExternalConnections.m_BuffersToOps.insert({ sourceBuffer, dma1Raw });
    result.m_EndingGlue = std::move(endingGlue);

    StartingGlue startingGlue;
    startingGlue.m_Graph.AddOp(std::move(dma2));
    startingGlue.m_ExternalConnections.m_BuffersToOps.insert({ dramBufferRaw, dma2Raw });
    startingGlue.m_ExternalConnections.m_OpsToBuffers.insert({ dma2Raw, destBuffer });
    result.m_StartingGlues.push_back(std::move(startingGlue));

    return result;
}

StartingAndEndingGlues Combiner::GenerateSharedGlue(Buffer* sourceBuffer,
                                                    std::vector<Buffer*>& destBuffers,
                                                    const CascadingBufferFormat cascadingBufferFormat) const
{
    // A single glue is used to stitch beteween a source SRAM and multiple destination SRAMs
    StartingAndEndingGlues result;

    // A single DRAM buffer is shared
    auto dramBuffer = std::make_unique<Buffer>(
        Location::Dram, cascadingBufferFormat, sourceBuffer->m_TensorShape, TensorShape{ 0, 0, 0, 0 },
        TraversalOrder::Xyz, utils::TotalSizeBytesNHWCB(sourceBuffer->m_TensorShape), sourceBuffer->m_QuantizationInfo);
    dramBuffer->m_BufferType = BufferType::Intermediate;

    // A input DMA is shared to move data from source SRAM
    // to the DRAM buffer.
    auto dma1             = std::make_unique<DmaOp>(cascadingBufferFormat);
    DmaOp* dma1Raw        = dma1.get();
    Buffer* dramBufferRaw = dramBuffer.get();
    EndingGlue endingGlue;
    endingGlue.m_Graph.AddOp(std::move(dma1));
    endingGlue.m_Graph.AddBuffer(std::move(dramBuffer));
    endingGlue.m_Graph.SetProducer(dramBufferRaw, dma1Raw);
    endingGlue.m_ExternalConnections.m_BuffersToOps.insert({ sourceBuffer, dma1Raw });
    result.m_EndingGlue = std::move(endingGlue);

    // Each destination uses its own output DMA
    // to move data from DRAM to its SRAM
    for (uint32_t i = 0; i < destBuffers.size(); i++)
    {
        if (destBuffers[i]->m_Location != Location::Sram)
        {
            StartingGlue startingGlue;
            startingGlue.m_ExternalConnections.m_ReplacementBuffers.insert({ destBuffers[i], dramBufferRaw });
            result.m_StartingGlues.push_back(std::move(startingGlue));
            continue;
        }
        auto dma2      = std::make_unique<DmaOp>(cascadingBufferFormat);
        DmaOp* dma2Raw = dma2.get();

        StartingGlue startingGlue;
        startingGlue.m_Graph.AddOp(std::move(dma2));
        startingGlue.m_ExternalConnections.m_BuffersToOps.insert({ dramBufferRaw, dma2Raw });
        startingGlue.m_ExternalConnections.m_OpsToBuffers.insert({ dma2Raw, destBuffers[i] });
        result.m_StartingGlues.push_back(std::move(startingGlue));
    }

    return result;
}

// Generate glue between DRAM buffers
// DRAM --DmaOp--> SRAM --DmaOp--> DRAM
StartingAndEndingGlues Combiner::GenerateGlueBetweenDramAndDram(Buffer* inputBuffer, Buffer* outputBuffer) const
{
    StartingAndEndingGlues result;
    auto dma1      = std::make_unique<DmaOp>(inputBuffer->m_Format);
    DmaOp* dma1Raw = dma1.get();

    TensorShape outputShapeNHWCB = utils::RoundUpHeightAndWidthToBrickGroup(inputBuffer->m_TensorShape);
    outputShapeNHWCB[3] =
        utils::RoundUpToNearestMultiple(outputShapeNHWCB[3], utils::GetChannels(m_Caps.GetBrickGroupShape()));

    // Set the SRAM buffer's stripe size to be the smallest height and width to be the most compatible.
    // We can't split depth because if one of the buffers is NHWC that won't be compatible.
    TensorShape sramStripeShape = { 1, utils::GetHeight(m_Caps.GetBrickGroupShape()),
                                    utils::GetWidth(m_Caps.GetBrickGroupShape()),
                                    utils::GetChannels(outputShapeNHWCB) };
    auto sramBuffer             = std::make_unique<Buffer>(
        Location::Sram, CascadingBufferFormat::NHWCB, outputShapeNHWCB, sramStripeShape, TraversalOrder::Xyz,
        utils::TotalSizeBytesNHWCB(sramStripeShape), inputBuffer->m_QuantizationInfo);
    sramBuffer->m_BufferType = BufferType::Intermediate;

    Buffer* sramBufferRaw = sramBuffer.get();
    auto dma2             = std::make_unique<DmaOp>(outputBuffer->m_Format);
    DmaOp* dma2Raw        = dma2.get();

    // We can choose to the dram buffer in the either in starting or ending
    // for now just put it in the starting glue
    // We still need an ending glue but it is empty
    EndingGlue endingGlue;
    result.m_EndingGlue = std::move(endingGlue);

    StartingGlue startingGlue;
    startingGlue.m_Graph.AddOp(std::move(dma1));
    startingGlue.m_Graph.AddOp(std::move(dma2));
    startingGlue.m_Graph.AddBuffer(std::move(sramBuffer));
    startingGlue.m_Graph.SetProducer(sramBufferRaw, dma1Raw);
    startingGlue.m_Graph.AddConsumer(sramBufferRaw, dma2Raw, 0);
    startingGlue.m_ExternalConnections.m_BuffersToOps.insert({ outputBuffer, dma1Raw });
    startingGlue.m_ExternalConnections.m_OpsToBuffers.insert({ dma2Raw, inputBuffer });

    result.m_StartingGlues.push_back(std::move(startingGlue));

    return result;
}

std::pair<bool, StartingAndEndingGlues> Combiner::GetGlue(Buffer* outputBuffer, Buffer* inputBuffer)
{
    if ((outputBuffer->m_Location == Location::Sram && inputBuffer->m_Location == Location::Dram))
    {
        StartingAndEndingGlues glues = GenerateGlueBetweenSramAndDram(outputBuffer, inputBuffer, inputBuffer->m_Format);
        return std::make_pair(true, std::move(glues));
    }
    else if (outputBuffer->m_Location == Location::Dram && inputBuffer->m_Location == Location::Sram)
    {
        // Going from DRAM to SRAM with NHWC can't be done with splitting in channels so we must add a conversion to NHWCB through SRAM
        // The firmware doesn't currently support loading NHWC data from DRAM with packed boundary data, so in this case we must also add a conversion
        StartingAndEndingGlues glues;
        if (outputBuffer->m_Format == CascadingBufferFormat::NHWC &&
            (utils::GetChannels(inputBuffer->m_StripeShape) < utils::GetChannels(inputBuffer->m_TensorShape) ||
             utils::AnyPackedBoundaryData(inputBuffer->m_PackedBoundaryThickness)))
        {
            glues = GenerateGlueBetweenDramAndSramWithConversion(inputBuffer, outputBuffer);
            return std::make_pair(true, std::move(glues));
        }
        glues = GenerateGlueBetweenDramAndSram(outputBuffer, inputBuffer, outputBuffer->m_Format);
        return std::make_pair(true, std::move(glues));
    }
    else if (outputBuffer->m_Location == Location::Sram && inputBuffer->m_Location == Location::Sram)
    {
        CascadingBufferFormat cascadingBufferFormat = GetBestCascadingBufferDramFormat({ outputBuffer, inputBuffer });

        StartingAndEndingGlues glues = GenerateGlueBetweenSramAndSram(outputBuffer, inputBuffer, cascadingBufferFormat);
        return std::make_pair(true, std::move(glues));
    }
    else if (outputBuffer->m_Location == Location::Dram && inputBuffer->m_Location == Location::Dram)
    {

        bool areBuffersEquivalent = outputBuffer->m_Format == inputBuffer->m_Format &&
                                    outputBuffer->m_Order == inputBuffer->m_Order &&
                                    outputBuffer->m_SizeInBytes == inputBuffer->m_SizeInBytes;
        // Provide an empty Glue in this case, there is nothing to do
        if (areBuffersEquivalent)
        {
            StartingAndEndingGlues glues;
            EndingGlue endingGlue;
            StartingGlue startingGlue;
            startingGlue.m_ExternalConnections.m_ReplacementBuffers.insert({ inputBuffer, outputBuffer });
            glues.m_StartingGlues.push_back(std::move(startingGlue));
            glues.m_EndingGlue = std::move(endingGlue);
            return std::make_pair(true, std::move(glues));
        }
        else
        {
            StartingAndEndingGlues glues = GenerateGlueBetweenDramAndDram(inputBuffer, outputBuffer);
            return std::make_pair(true, std::move(glues));
        }
    }
    // If here it means that buffers are not glue-able
    // e.g. input buffer location is PleInputSram
    return std::make_pair(false, StartingAndEndingGlues());
}

std::pair<bool, StartingAndEndingGlues> Combiner::GetSharedGlue(Buffer* outputBuffer,
                                                                std::vector<Buffer*>& inputBuffers)
{
    // number of input buffers must be larger than 1
    assert(inputBuffers.size() > 1);

    Buffer* inputBuffer = inputBuffers.at(0);
    // Sanity check: only source in SRAM can share the buffer
    assert(outputBuffer->m_Location == Location::Sram);

    // Use NHWCB if the input buffer is in DRAM, otherwise tries to find a compressed format
    CascadingBufferFormat cascadingBufferFormat = inputBuffer->m_Location == Location::Dram
                                                      ? CascadingBufferFormat::NHWCB
                                                      : GetBestCascadingBufferDramFormat({ outputBuffer, inputBuffer });

    uint32_t numBufferSrams = inputBuffer->m_Location == Location::Sram;

    for (uint32_t i = 1; i < inputBuffers.size(); ++i)
    {
        inputBuffer = inputBuffers.at(i);

        // Continues looking for compressed format only if the format
        // chosen so far is not NHWCB
        if (cascadingBufferFormat != CascadingBufferFormat::NHWCB)
        {
            CascadingBufferFormat cascadingBufferFormatLocal =
                inputBuffer->m_Location == Location::Dram
                    ? CascadingBufferFormat::NHWCB
                    : GetBestCascadingBufferDramFormat({ outputBuffer, inputBuffer });

            // All input buffers must share the same format
            if (cascadingBufferFormatLocal != cascadingBufferFormat)
            {
                cascadingBufferFormat = CascadingBufferFormat::NHWCB;
            }
        }

        numBufferSrams += inputBuffer->m_Location == Location::Sram;
    }

    StartingAndEndingGlues glues = GenerateSharedGlue(outputBuffer, inputBuffers, cascadingBufferFormat);
    return std::make_pair(true, std::move(glues));
}

// A source part is glued to its destinations
Combination Combiner::GluePartToCombinationSrcToDests(const BasePart& sPart,
                                                      const Combination& comb,
                                                      const std::vector<PartConnection>& destPartEdge)
{
    assert(destPartEdge.size() != 0);
    Combination result = comb;

    // Find an element belonging to source part in the combination
    auto elemIt = comb.m_Elems.find(sPart.GetPartId());
    assert(elemIt != comb.m_Elems.end());
    const Plan& sourcePlan = *elemIt->second.m_Plan;

    // Find the output buffer of the source node.
    // Note all destination nodes are branched off from
    // the same source node
    Buffer* outputBuffer = sourcePlan.GetOutputBuffer(destPartEdge.at(0).m_Source);
    assert(outputBuffer != nullptr);

    bool isSrcLocationSram = outputBuffer->m_Location == Location::Sram;

    std::vector<Buffer*> buffersSharingGlue;
    std::vector<std::pair<PartConnection, bool>> edgesSharingGlue;
    std::vector<bool> inputBufferSram;
    std::vector<std::pair<PartConnection, Buffer*>> buffersEdgesUseOwnGlue;

    auto canUseSharedGlue = [&](const BasePart& part, const Buffer* inputBuffer) -> bool {
        return ((!IsPartOutput(part) && inputBuffer->m_Format != CascadingBufferFormat::NHWC) ||
                inputBuffer->m_Location == Location::Sram);
    };

    // Gets the number of branches that are not output or input buffer in SRAM
    uint32_t noOfBranchesToShareGlue = 0;
    for (const auto& partEdge : destPartEdge)
    {
        const BasePart& part = m_GraphOfParts.GetPart(partEdge.m_Destination.m_PartId);
        const Plan& plan     = GetPlanForPartFromCombination(part, comb);

        const Buffer* inputBuffer = plan.GetInputBuffer(partEdge.m_Destination);
        assert(inputBuffer != nullptr);

        noOfBranchesToShareGlue += canUseSharedGlue(part, inputBuffer);
    }

    for (const auto& partEdge : destPartEdge)
    {
        const BasePart& part = m_GraphOfParts.GetPart(partEdge.m_Destination.m_PartId);
        const Plan& plan     = GetPlanForPartFromCombination(part, comb);

        Buffer* inputBuffer = plan.GetInputBuffer(partEdge.m_Destination);
        assert(inputBuffer != nullptr);

        // A branch is attached to a shared glue if the following conditions are met:
        // it is either not a output part and its input buffer format is not NHWC,
        // or its input buffer is SRAM
        // and there are at least 2 such branches
        // otherwise it uses its own glue
        // Reason for not assigning a branch that is an output in DRAM:
        // Intermediate and output buffers cannot be shared and using its
        // own glue prevents the output DRAM buffer being merged with
        // the intermediate buffer (created for the glue).
        // A branch with input buffer formant in NHWC cannot use shared glue
        // because the format of the intermediate buffer of a shared glue is NHWCB.
        bool useSharedGlue = noOfBranchesToShareGlue > 1 && canUseSharedGlue(part, inputBuffer);

        if (isSrcLocationSram && useSharedGlue)
        {
            buffersSharingGlue.push_back(inputBuffer);
            edgesSharingGlue.push_back(std::make_pair(partEdge, inputBuffer->m_Location == Location::Sram));
        }
        else
        {
            buffersEdgesUseOwnGlue.push_back(std::make_pair(partEdge, inputBuffer));
        }
    }

    assert(buffersSharingGlue.size() == edgesSharingGlue.size());

    StartingAndEndingGlues startingAndEndingGlues;
    for (auto branch : buffersEdgesUseOwnGlue)
    {
        std::pair<bool, StartingAndEndingGlues> glueResult = GetGlue(outputBuffer, branch.second);
        // There should only be 1 starting and 1 ending glue as these aren't shared.
        assert(glueResult.second.m_StartingGlues.size() == 1);
        if (!glueResult.first)
        {
            // This combination is not valid, clear it
            return Combination{};
        }
        startingAndEndingGlues.m_EndingGlue.m_Graph.MergeOpGraph(glueResult.second.m_EndingGlue.m_Graph);
        startingAndEndingGlues.m_EndingGlue.m_ExternalConnections.m_BuffersToOps.insert(
            glueResult.second.m_EndingGlue.m_ExternalConnections.m_BuffersToOps.begin(),
            glueResult.second.m_EndingGlue.m_ExternalConnections.m_BuffersToOps.end());
        startingAndEndingGlues.m_EndingGlue.m_ExternalConnections.m_OpsToBuffers.insert(
            glueResult.second.m_EndingGlue.m_ExternalConnections.m_OpsToBuffers.begin(),
            glueResult.second.m_EndingGlue.m_ExternalConnections.m_OpsToBuffers.end());
        startingAndEndingGlues.m_EndingGlue.m_ExternalConnections.m_ReplacementBuffers.insert(
            glueResult.second.m_EndingGlue.m_ExternalConnections.m_ReplacementBuffers.begin(),
            glueResult.second.m_EndingGlue.m_ExternalConnections.m_ReplacementBuffers.end());

        result.SetStartingGlue(std::move(glueResult.second.m_StartingGlues[0]), branch.first.m_Destination);
    }

    if (buffersSharingGlue.size() != 0)
    {
        std::pair<bool, StartingAndEndingGlues> glueResult = GetSharedGlue(outputBuffer, buffersSharingGlue);

        assert(glueResult.first == true);
        // The number of starting glues must be the same as the number of destinations
        assert(glueResult.second.m_StartingGlues.size() == edgesSharingGlue.size());

        startingAndEndingGlues.m_EndingGlue.m_Graph.MergeOpGraph(glueResult.second.m_EndingGlue.m_Graph);
        startingAndEndingGlues.m_EndingGlue.m_ExternalConnections.m_BuffersToOps.insert(
            glueResult.second.m_EndingGlue.m_ExternalConnections.m_BuffersToOps.begin(),
            glueResult.second.m_EndingGlue.m_ExternalConnections.m_BuffersToOps.end());
        startingAndEndingGlues.m_EndingGlue.m_ExternalConnections.m_OpsToBuffers.insert(
            glueResult.second.m_EndingGlue.m_ExternalConnections.m_OpsToBuffers.begin(),
            glueResult.second.m_EndingGlue.m_ExternalConnections.m_OpsToBuffers.end());
        startingAndEndingGlues.m_EndingGlue.m_ExternalConnections.m_ReplacementBuffers.insert(
            glueResult.second.m_EndingGlue.m_ExternalConnections.m_ReplacementBuffers.begin(),
            glueResult.second.m_EndingGlue.m_ExternalConnections.m_ReplacementBuffers.end());

        for (uint32_t i = 0; i < edgesSharingGlue.size(); ++i)
        {
            result.SetStartingGlue(std::move(glueResult.second.m_StartingGlues[i]),
                                   edgesSharingGlue[i].first.m_Destination);
        }
    }
    result.AddEndingGlue(std::move(startingAndEndingGlues.m_EndingGlue), destPartEdge.at(0).m_Source);

    return result;
}

void Combiner::DeallocateUnusedBuffers(const Plan& sPlan, SramAllocator& allocator)
{
    for (auto&& buffer : sPlan.m_OpGraph.GetBuffers())
    {
        if (buffer->m_Location != Location::Sram)
        {
            continue;
        }

        assert(buffer->m_Offset.has_value() == true);

        bool allConsumersAreAtomic          = false;
        OpGraph::ConsumersList consumerList = sPlan.m_OpGraph.GetConsumers(buffer);

        if (!consumerList.empty())
        {
            allConsumersAreAtomic =
                std::all_of(consumerList.begin(), consumerList.end(),
                            [](std::pair<Op*, uint32_t> x) { return x.first->m_Lifetime == Lifetime::Atomic; });
        }

        if (allConsumersAreAtomic == true)
        {
            allocator.Free(0, buffer->m_Offset.value());
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
                                 const SramAllocator& alloc,
                                 uint32_t prevNumWeightStripes,
                                 bool prevDoubleBuffered,
                                 const PleOperations& pleOps,
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

        SramAllocator allocCopy = alloc;
        DeallocateUnusedBuffers(sPlan, allocCopy);

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
                SramAllocator tempAlloc = allocCopy;

                PleOperations tempPleOps = pleOps;

                if (!IsPlanOutputGlueable(plan))
                {
                    continue;
                }

                if (!ArePlansCompatible(sPlan, plan, connection))
                {
                    continue;
                }

                if (!ArePlansAllowedToMerge(sPlan, plan, connection))
                {
                    continue;
                }

                if (!IsPlanAllocated(tempAlloc, plan, tempPleOps, sramBuffer, StatsType::EndSection))
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
                result               = GetBestCombination(options);
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
                OpGraph::ConsumersList consumers = plan.m_OpGraph.GetConsumers(inputMapping.first);
                bool aConsumerIsCascade          = std::any_of(consumers.begin(), consumers.end(),
                                                      [](auto& p) { return p.first->m_Lifetime == Lifetime::Cascade; });
                totalAgents += aConsumerIsCascade;
            }
        }
    }

    // Count Agents for each Op in the graph. The Ops should be in execution order.
    for (Op* op : plan.m_OpGraph.GetOps())
    {
        uint32_t numberOfInputs = static_cast<uint32_t>(plan.m_OpGraph.GetInputs(op).size());
        totalAgents += op->GetNumberOfAgents(numberOfInputs);
        result &= totalAgents <= m_Caps.GetAgentWindowSize();

        // The total is to be reset when all preceding Agents have finished execution.
        // All preceding Agents must finish execution when an Atomic Op finishes
        // execution. This Atomic Op must be in the path from IFM to OFM. We can
        // identify whether an Op is in the path from IFM to OFM by checking its
        // output buffer's format. If the buffer's format is WEIGHT, it means that the
        // buffer's producer loads weights and hence it is not in the IFM to OFM path.
        if (plan.m_OpGraph.GetOutput(op)->m_Format != CascadingBufferFormat::WEIGHT)
        {
            totalAgents = op->m_Lifetime == Lifetime::Atomic ? 0 : totalAgents;
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
                Op* producer = plan.m_OpGraph.GetProducer(outputMapping.first);
                if (producer != nullptr)
                {
                    bool isCascade = producer->m_Lifetime == Lifetime::Cascade;
                    totalAgents += isCascade;
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
Combination Combiner::StartSection(const BasePart& part, const BasePart& nextPart, const SramAllocator& alloc)
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
                SramAllocator tempAlloc = alloc;
                if (!IsPlanInputGlueable(plan))
                {
                    continue;
                }

                // A list of PLE kernels that have been loaded into the SRAM
                // for this section. Once loaded, a PLE kernel will remain
                // in the SRAM as kernel reload is deemed to be costly.
                // The list is updated whenever a new kernel is encountered.
                PleOperations pleOps = {};

                // Allocation requirement are different for start of section
                if (!IsPlanAllocated(tempAlloc, plan, pleOps, nullptr, StatsType::StartSection))
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
                Combination ended     = EndSection(nextPart, part, head, tempAlloc, currNumWeightStripes,
                                               hasSectionDoubleBuffered, pleOps, totalAgents);
                Combination continued = ContinueSection(nextPart, part, head, tempAlloc, currNumWeightStripes,
                                                        hasSectionDoubleBuffered, pleOps, totalAgents);
                Combinations options  = { result, continued, ended };
                result                = GetBestCombination(options);
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

    Combination result = {};

    // Check if this Part can double buffer.
    // By default, no double buffering is performed.
    uint32_t currNumWeightStripesMax = g_NumWeightStripesMin;
    if (part.CanDoubleBufferWeights())
    {
        currNumWeightStripesMax = g_NumWeightStripesMax;
    }

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

            if (!IsPlanInputGlueable(plan))
            {
                continue;
            }
            if (!IsPlanOutputGlueable(plan))
            {
                continue;
            }
            if (!IsPlanAllocated(alloc, plan, pleOps, nullptr, StatsType::SinglePartSection))
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
            Combinations options = { result, head };
            result               = GetBestCombination(options);
        }
    }

    //  Next part in the graph
    const BasePart* nextPartGraph = GetNextPart(&part);

    if (!result.m_Elems.empty() && nextPartGraph != nullptr)
    {
        result = result + FindBestCombinationForPart(*nextPartGraph);

        // Each of it destination part will start its own new section.
        // Therefore they all need to be glued with their source.
        std::vector<PartConnection> destPartEdge = m_GraphOfParts.GetDestinationConnections(part.GetPartId());

        if (destPartEdge.empty() == false)
        {
            result = GluePartToCombinationSrcToDests(part, result, destPartEdge);
        }
    }

    return result;
}

Combination Combiner::ContinueSection(const BasePart& part,
                                      const BasePart& sPart,
                                      const Combination& comb,
                                      const SramAllocator& alloc,
                                      uint32_t prevNumWeightStripes,
                                      bool prevDoubleBuffered,
                                      const PleOperations& pleOps,
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

        SramAllocator allocCopy = alloc;
        DeallocateUnusedBuffers(sPlan, allocCopy);

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
                SramAllocator tempAlloc = allocCopy;

                PleOperations tempPleOps = pleOps;

                if (!ArePlansCompatible(sPlan, plan, connection))
                {
                    continue;
                }

                if (!ArePlansAllowedToMerge(sPlan, plan, connection))
                {
                    continue;
                }

                if (!ArePlansStreamingStrategiesCompatible(sPlan, plan, connection))
                {
                    continue;
                }

                if (!IsPlanAllocated(tempAlloc, plan, tempPleOps, sramBuffer, StatsType::EndSection))
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
                Combination ended = EndSection(*nextPartGraph, part, section, tempAlloc, numWeightStripes,
                                               hasSectionDoubleBuffered, tempPleOps, totalAgents);

                // Next one is the middle part of the section
                Combination continued = ContinueSection(*nextPartGraph, part, section, tempAlloc, numWeightStripes,
                                                        hasSectionDoubleBuffered, tempPleOps, totalAgents);
                options               = { result, continued, ended };

                result = GetBestCombination(options);
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

        // This is the start of a new section, reset the allocated Sram
        SramAllocator alloc(m_Caps.GetTotalSramSize() / m_Caps.GetNumberOfSrams());

        // Start of a new section
        start = StartSection(part, *nextPartGraph, alloc);
    }

    // Lonely part
    Combination lonely = SinglePartSection(part);

    Combinations options = { start, lonely };

    result = GetBestCombination(options);

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

        DumpDebugInfo({ result }, m_Stats, m_DebuggingContext,
                      "FindBestCombinationForPart/Part" + std::to_string(part.GetPartId()));
    }
    return result;
}

Combiner::Combiner(const GraphOfParts& graphOfParts,
                   const HardwareCapabilities& caps,
                   const EstimationOptions& estOpt,
                   const DebuggingContext& debuggingContext)
    : m_GraphOfParts(graphOfParts)
    , m_Caps(caps)
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
        MakeDirectory(m_DebuggingContext.GetAbsolutePathOutputFileName("FindBestCombinationForPart").c_str());
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
        // Add buffers from the plan
        for (Buffer* b : plan.m_OpGraph.GetBuffers())
        {
            // Don't add a buffer if its an input to the plan, and the glue states it needs to be replaced with another buffer
            // Instead, remap it to the one we already have
            auto inputSlotIt = plan.m_InputMappings.find(b);
            if (inputSlotIt != plan.m_InputMappings.end())
            {
                // Get the glue for the input buffer
                StartingGlue* glue = startingGlues.at(inputSlotIt->second).get();
                // Look up the buffer replacement
                auto bufferIt = glue->m_ExternalConnections.m_ReplacementBuffers.find(b);
                // If the buffer replacement exists add it to the merged buffers
                if (bufferIt != glue->m_ExternalConnections.m_ReplacementBuffers.end())
                {
                    mergedBuffers[b] = getEffectiveBuffer(bufferIt->second);
                }
                else
                {
                    result.AddBuffer(b);
                }
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
        for (PartInputSlot& inputSlot : inputSlots)
        {
            // Get the glue for the input buffer
            StartingGlue* glue = startingGlues.at(inputSlot).get();
            // Connect the plan, the starting glue and the previous plans ending glue together.
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

        // Connect the ending glue
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
            const std::unordered_map<PartOutputSlot, std::shared_ptr<EndingGlue>>& endingGlues =
                elemIt->second.m_EndingGlues;

            EndingGlue* glue = endingGlues.at(outputSlot).get();
            result.MergeOpGraph(glue->m_Graph);
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

void Combiner::SavePartsPlans(const BasePart& part, const Plans& plans) const
{
    if (m_DebuggingContext.m_DebugInfo.m_DumpDebugFiles >= CompilationOptions::DebugLevel::Medium)
    {
        std::ofstream debugPlanCountsDumpFile(
            m_DebuggingContext.GetAbsolutePathOutputFileName("Cascaded_PlanCounts.txt"));

        std::string folder = "Parts/" + part.m_DebugTag;
        ethosn::utils::MakeDirectory(m_DebuggingContext.GetAbsolutePathOutputFileName(folder).c_str());

        debugPlanCountsDumpFile << part.m_DebugTag << ": " << plans.size() << std::endl;

        m_DebuggingContext.SavePlansToDot(CompilationOptions::DebugLevel::Medium, plans, folder + "/Plans.dot",
                                          DetailLevel::Low);
        m_DebuggingContext.SavePlansToDot(CompilationOptions::DebugLevel::Medium, plans, folder + "/PlansDetailed.dot",
                                          DetailLevel::High);
    }
}

}    // namespace support_library
}    // namespace ethosn
