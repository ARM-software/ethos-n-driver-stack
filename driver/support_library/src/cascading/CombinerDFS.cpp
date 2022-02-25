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

            if (!comb.m_Elems.empty())
            {
                debuggingContext.SaveCombinationToDot(CompilationOptions::DebugLevel::None, comb, parts,
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

std::pair<DmaOp*, Buffer*> CreateDramBufferAndDmaOp(Buffer* b, OwnedOpGraph& opGraph)
{
    // Create a DRAM buffer where the data is coming from or going to
    auto dramBuffer = std::make_unique<Buffer>(Lifetime::Atomic, Location::Dram, CascadingBufferFormat::NHWCB,
                                               b->m_TensorShape, TensorShape{ 0, 0, 0, 0 }, TraversalOrder::Xyz,
                                               utils::TotalSizeBytesNHWCB(b->m_TensorShape), b->m_QuantizationInfo);
    // Create a DMA operatoin that is moving data from DRAM to SRAM or viceversa
    auto dma              = std::make_unique<DmaOp>();
    DmaOp* dmaRaw         = dma.get();
    Buffer* dramBufferRaw = dramBuffer.get();
    // Add to the OwnedOpGraph
    opGraph.AddOp(std::move(dma));
    opGraph.AddBuffer(std::move(dramBuffer));
    return std::make_pair(dmaRaw, dramBufferRaw);
}

void CompleteOpGraph(OpGraph& combiOpGraph, OwnedOpGraph& tempOpGraph)
{
    // Make sure that the graph is complete in the sense that
    // all the input and output data is moved from and to DRAM

    // Note that we *copy* the vector of Buffers, as we may modify the collection as we enumerate
    std::vector<Buffer*> buffers = combiOpGraph.GetBuffers();
    for (Buffer* b : buffers)
    {
        Op* producer = combiOpGraph.GetProducer(b);
        if (!producer)
        {
            // This buffer does not have a producer and it is in SRAM
            if (b->m_Location == Location::Sram)
            {
                // Create a DRAM buffer where the data is coming from
                auto result           = CreateDramBufferAndDmaOp(b, tempOpGraph);
                DmaOp* dmaRaw         = result.first;
                Buffer* dramBufferRaw = result.second;
                combiOpGraph.AddOp(dmaRaw);
                combiOpGraph.AddBuffer(dramBufferRaw);
                combiOpGraph.SetProducer(b, dmaRaw);
                combiOpGraph.AddConsumer(dramBufferRaw, dmaRaw, 0);
            }
        }
        if (combiOpGraph.GetConsumers(b).empty())
        {
            // This buffer does not have a consumer and it is in SRAM
            if (b->m_Location == Location::Sram)
            {
                // Create a DRAM buffer where the data is going to
                auto result           = CreateDramBufferAndDmaOp(b, tempOpGraph);
                DmaOp* dmaRaw         = result.first;
                Buffer* dramBufferRaw = result.second;
                combiOpGraph.AddOp(dmaRaw);
                combiOpGraph.AddBuffer(dramBufferRaw);
                combiOpGraph.SetProducer(dramBufferRaw, dmaRaw);
                combiOpGraph.AddConsumer(b, dmaRaw, 0);
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
    bool areBuffersEquivalent =
        sPlanOutputBuffer->m_Location == planInputBuffer->m_Location && planInputBuffer->m_Location != Location::Dram &&
        sPlanOutputBuffer->m_Location != Location::Dram && sPlanOutputBuffer->m_Format == planInputBuffer->m_Format &&
        sPlanOutputBuffer->m_StripeShape == planInputBuffer->m_StripeShape &&
        sPlanOutputBuffer->m_Order == planInputBuffer->m_Order &&
        sPlanOutputBuffer->m_SizeInBytes == planInputBuffer->m_SizeInBytes &&
        sPlanOutputBuffer->m_NumStripes == planInputBuffer->m_NumStripes;

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
            pleKernelAllocated =
                localAlloc.Allocate(userId, (pleKernelSize / m_Caps.GetNumberOfSrams()), AllocationPreference::Start);

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

                OwnedOpGraph tempOpGraph;

                CompleteOpGraph(combiOpGraph, tempOpGraph);

                EstimatedOpGraph estimatedOpGraph =
                    ethosn::support_library::EstimateOpGraph(combiOpGraph, m_Caps, m_EstOpt);

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
            // If estimation failed, pick the first non empty combination
            for (const Combination& combination : combs)
            {
                if (!combination.m_Elems.empty())
                {
                    return combination;
                }
            }
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

CascadingBufferFormat
    Combiner::GetBestCascadingBufferDramFormat(const std::array<TensorShape, 2> inputOutputStripeShapes) const
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

// This table shows all possible buffer location permutations
// that requires glue.
//
//   Entry  |    Out Plan Location     ||      In Plan Location
//  ===========================================================
//          |                          ||
//     1    |         SRAM             ||         DRAM
//          |                          ||
//  -----------------------------------------------------------
//          |                          ||
//     2    |         DRAM             ||         SRAM
//          |                          ||
//  -----------------------------------------------------------
//          |                          ||
//     3    |         SRAM             ||         SRAM
//          |                          ||
//  -----------------------------------------------------------
//
// Entries 1 and 2 are pratically the same, there is a need
// of a DMA operation to bring data from a given input to
// an output, buffer in DRAM has been already allocated and
// for that reason there is no choice to make about format.
//
std::unique_ptr<Glue> Combiner::GenerateGlueBetweenSramAndDram() const
{
    auto result     = std::make_unique<Glue>();
    Glue* resultRaw = result.get();
    auto dma        = std::make_unique<DmaOp>();
    DmaOp* dmaRaw   = dma.get();
    resultRaw->m_Graph.AddOp(std::move(dma));
    resultRaw->m_InputSlot = { dmaRaw, 0 };
    resultRaw->m_Output.push_back(dmaRaw);

    // sanity check
    assert(resultRaw->m_OutDmaOffset == 0);
    return result;
}

// For entry 3 (see table above) there are as many glues possible as the
// number of buffer formats in DRAM i.e. :
//  - NHWCB
//  - FCAF_DEEP
//  - FCAF_WIDE
//
std::unique_ptr<Glue> Combiner::GenerateGlueBetweenSramAndSram(const Buffer* buffer,
                                                               const CascadingBufferFormat cascadingBufferFormat) const
{
    auto result     = std::make_unique<Glue>();
    Glue* resultRaw = result.get();

    auto dramBuffer = std::make_unique<Buffer>(
        Lifetime::Atomic, Location::Dram, cascadingBufferFormat, buffer->m_TensorShape, TensorShape{ 0, 0, 0, 0 },
        TraversalOrder::Xyz, utils::TotalSizeBytesNHWCB(buffer->m_TensorShape), buffer->m_QuantizationInfo);

    auto dma1             = std::make_unique<DmaOp>();
    DmaOp* dma1Raw        = dma1.get();
    Buffer* dramBufferRaw = dramBuffer.get();
    auto dma2             = std::make_unique<DmaOp>();
    DmaOp* dma2Raw        = dma2.get();
    resultRaw->m_Graph.AddOp(std::move(dma1));
    resultRaw->m_Graph.AddOp(std::move(dma2));
    resultRaw->m_Graph.AddBuffer(std::move(dramBuffer));
    resultRaw->m_Graph.SetProducer(dramBufferRaw, dma1Raw);
    resultRaw->m_Graph.AddConsumer(dramBufferRaw, dma2Raw, 0);
    resultRaw->m_InputSlot = { dma1Raw, 0 };
    resultRaw->m_Output.push_back(dma2Raw);
    resultRaw->m_OutDmaOffset = 1;

    return result;
}

std::unique_ptr<Glue> Combiner::GenerateGlueBetweenSramAndSrams(const Buffer* buffer,
                                                                const CascadingBufferFormat cascadingBufferFormat,
                                                                uint32_t numOfOutputs) const
{
    // A single glue is used to stitch beteween a source SRAM and multiple destination SRAMs
    auto result     = std::make_unique<Glue>();
    Glue* resultRaw = result.get();

    // A single DRAM buffer is shared
    auto dramBuffer = std::make_unique<Buffer>(
        Lifetime::Atomic, Location::Dram, cascadingBufferFormat, buffer->m_TensorShape, TensorShape{ 0, 0, 0, 0 },
        TraversalOrder::Xyz, utils::TotalSizeBytesNHWCB(buffer->m_TensorShape), buffer->m_QuantizationInfo);

    // A input DMA is shared to move data from source SRAM
    // to the DRAM buffer.
    auto dma1             = std::make_unique<DmaOp>();
    DmaOp* dma1Raw        = dma1.get();
    Buffer* dramBufferRaw = dramBuffer.get();
    resultRaw->m_Graph.AddOp(std::move(dma1));
    resultRaw->m_Graph.AddBuffer(std::move(dramBuffer));
    resultRaw->m_Graph.SetProducer(dramBufferRaw, dma1Raw);
    resultRaw->m_InputSlot = { dma1Raw, 0 };

    // Each destination uses its own output DMA
    // to move data from DRAM to its SRAM
    for (uint32_t i = 0; i < numOfOutputs; i++)
    {
        auto dma2      = std::make_unique<DmaOp>();
        DmaOp* dma2Raw = dma2.get();

        resultRaw->m_Graph.AddOp(std::move(dma2));
        resultRaw->m_Graph.AddConsumer(dramBufferRaw, dma2Raw, 0);

        resultRaw->m_Output.push_back(dma2Raw);
    }

    resultRaw->m_OutDmaOffset = 1;

    return result;
}

std::pair<bool, const Glue*> Combiner::GetGlue(const Buffer* outputBuffer, const Buffer* inputBuffer)
{
    if ((outputBuffer->m_Location == Location::Sram && inputBuffer->m_Location == Location::Dram) ||
        (outputBuffer->m_Location == Location::Dram && inputBuffer->m_Location == Location::Sram))
    {
        m_GluesVector.push_back(GenerateGlueBetweenSramAndDram());
        return std::make_pair(true, m_GluesVector.back().get());
    }
    else if (outputBuffer->m_Location == Location::Sram && inputBuffer->m_Location == Location::Sram)
    {
        CascadingBufferFormat cascadingBufferFormat =
            GetBestCascadingBufferDramFormat({ outputBuffer->m_StripeShape, inputBuffer->m_StripeShape });

        m_GluesVector.push_back(GenerateGlueBetweenSramAndSram(inputBuffer, cascadingBufferFormat));
        return std::make_pair(true, m_GluesVector.back().get());
    }
    else if (outputBuffer->m_Location == Location::Dram && inputBuffer->m_Location == Location::Dram)
    {
        // Provide an empty Glue in this case, there is nothing to do
        return std::make_pair(true, nullptr);
    }
    // If here it means that buffers are not glue-able
    // e.g. input buffer location is PleInputSram
    return std::make_pair(false, nullptr);
}

std::pair<bool, const Glue*> Combiner::GetSharedGlue(const Buffer* outputBuffer,
                                                     std::vector<const Buffer*>& inputBuffers)
{
    // number of input buffers must be larger than 1
    assert(inputBuffers.size() > 1);

    const Buffer* inputBuffer = inputBuffers.at(0);
    // Sanity check: only source in SRAM can share the buffer
    assert(outputBuffer->m_Location == Location::Sram);

    // Use NHWCB if the input buffer is in DRAM, otherwise tries to find a compressed format
    CascadingBufferFormat cascadingBufferFormat =
        inputBuffer->m_Location == Location::Dram
            ? CascadingBufferFormat::NHWCB
            : GetBestCascadingBufferDramFormat({ outputBuffer->m_StripeShape, inputBuffer->m_StripeShape });

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
                    : GetBestCascadingBufferDramFormat({ outputBuffer->m_StripeShape, inputBuffer->m_StripeShape });

            // All input buffers must share the same format
            if (cascadingBufferFormatLocal != cascadingBufferFormat)
            {
                cascadingBufferFormat = CascadingBufferFormat::NHWCB;
            }
        }

        numBufferSrams += inputBuffer->m_Location == Location::Sram;
    }

    m_GluesVector.push_back(GenerateGlueBetweenSramAndSrams(inputBuffer, cascadingBufferFormat, numBufferSrams));
    return std::make_pair(true, m_GluesVector.back().get());
}

// A destination part is glued to its sources
Combination Combiner::GluePartToCombinationDestToSrcs(const BasePart& part,
                                                      const Combination& comb,
                                                      const std::vector<PartConnection>& sources)
{
    Combination result = comb;

    // Get the plan for the part to be glued with all source parts
    const Plan& destPlan = GetPlanForPartFromCombination(part, comb);

    // Iterate on all the source parts slots
    for (const auto& source : sources)
    {
        // Find the source part in the combination,
        // it might happen that some branches haven't
        // been populated yet, that's fine, it will
        // just skip them
        auto elemIt = comb.m_Elems.find(source.m_Source.m_PartId);
        if (elemIt != comb.m_Elems.end())
        {
            const Plan& sourcePlan = *elemIt->second.m_Plan;

            // Sanity tests - make sure the two Plans are for adjacent Parts.
            const Buffer* outputBuffer     = sourcePlan.GetOutputBuffer(source.m_Source);
            const PartInputSlot& inputSlot = source.m_Destination;
            const Buffer* inputBuffer      = destPlan.GetInputBuffer(inputSlot);
            assert(outputBuffer != nullptr && inputBuffer != nullptr);

            auto glueResult = GetGlue(outputBuffer, inputBuffer);
            if (!glueResult.first)
            {
                // This combination is not valid, clear it
                return Combination{};
            }

            if (glueResult.second == nullptr)
            {
                continue;
            }

            const BasePart& part = m_GraphOfParts.GetPart(source.m_Source.m_PartId);
            result               = result + Combination(part, &source.m_Destination, glueResult.second, m_GraphOfParts);
        }
    }
    return result;
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
    const Buffer* outputBuffer = sourcePlan.GetOutputBuffer(destPartEdge.at(0).m_Source);
    assert(outputBuffer != nullptr);

    bool isSrcLocationSram = outputBuffer->m_Location == Location::Sram;

    std::vector<const Buffer*> buffersSharingGlue;
    std::vector<std::pair<PartConnection, bool>> edgesSharingGlue;
    std::vector<bool> inputBufferSram;
    std::vector<std::pair<PartConnection, const Buffer*>> buffersEdgesUseOwnGlue;

    // (1) source location SRAM and number of edges > 1 --- a shared glue
    //     will be used
    // (2) otherwise each edge uses its own glue
    for (const auto& partEdge : destPartEdge)
    {
        const BasePart& part = m_GraphOfParts.GetPart(partEdge.m_Destination.m_PartId);
        const Plan& plan     = GetPlanForPartFromCombination(part, comb);

        const Buffer* inputBuffer = plan.GetInputBuffer(partEdge.m_Destination);
        assert(inputBuffer != nullptr);

        if (isSrcLocationSram && destPartEdge.size() > 1)
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
    assert(buffersSharingGlue.size() == 0 || buffersEdgesUseOwnGlue.size() == 0);

    for (const auto branch : buffersEdgesUseOwnGlue)
    {
        auto glueResult = GetGlue(outputBuffer, branch.second);

        if (!glueResult.first)
        {
            // This combination is not valid, clear it
            return Combination{};
        }

        if (glueResult.second == nullptr)
        {
            continue;
        }

        result = result + Combination(sPart, &branch.first.m_Destination, glueResult.second, m_GraphOfParts);
    }

    if (buffersSharingGlue.size() != 0)
    {
        auto glueResult = GetSharedGlue(outputBuffer, buffersSharingGlue);

        assert(glueResult.first == true);

        for (const auto edge : edgesSharingGlue)
        {
            result =
                result + Combination(sPart, &edge.first.m_Destination, glueResult.second, edge.second, m_GraphOfParts);
        }
    }

    return result;
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
                                 uint32_t numWeightStripes,
                                 const PleOperations& pleOps)
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

        Plans plans = part.GetPlans(CascadeType::End, blkConfig, sramBuffer, numWeightStripes);

        for (Plan& plan : plans)
        {
            // Make a copy of the allocator since every plan needs to have its own,
            // each potential section won't allocate from the same allocator.
            SramAllocator tempAlloc = alloc;

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

            // Add current part and plan to the combination,
            Combination section =
                comb + Combination(part, std::move(plan), m_PartOrderTable[part.GetPartId()].first, m_GraphOfParts);

            Combinations options = { result, section };
            result               = GetBestCombination(options);
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

    if (IsPartSiso(part))
    {
        // sanity check SISO is the only use case.
        assert(m_GraphOfParts.GetPartInputs(part.GetPartId()).size() == 1 &&
               m_GraphOfParts.GetPartOutputs(part.GetPartId()).size() == 1);

        // The weight buffering will be improved by NNXSW-3628
        for (uint32_t numWeightStripes = g_NumWeightStripesMin; numWeightStripes <= g_NumWeightStripesMax;
             numWeightStripes++)
        {
            Plans plans =
                part.GetPlans(CascadeType::Beginning, ethosn::command_stream::BlockConfig{}, nullptr, numWeightStripes);

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
                Combination head(part, std::move(plan), m_PartOrderTable[part.GetPartId()].first, m_GraphOfParts);

                // Options to be estimated: consider continuing and ending the current section
                // in the next part
                Combination ended     = EndSection(nextPart, part, head, tempAlloc, numWeightStripes, pleOps);
                Combination continued = ContinueSection(nextPart, part, head, tempAlloc, numWeightStripes, pleOps);
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

    for (uint32_t numWeightStripes = g_NumWeightStripesMin; numWeightStripes <= g_NumWeightStripesMax;
         numWeightStripes++)
    {
        Plans plans =
            part.GetPlans(CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, numWeightStripes);

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
            // Glue will be added later on.
            // In this case local optimum = global optimum so
            // it can get the best plan for the part.
            Combination head(part, std::move(plan), m_PartOrderTable[part.GetPartId()].first, m_GraphOfParts);
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
                                      uint32_t numWeightStripes,
                                      const PleOperations& pleOps)
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

        Plans plans = part.GetPlans(CascadeType::Middle, blkConfig, sramBuffer, numWeightStripes);

        for (Plan& plan : plans)
        {
            // Make a copy of the allocator since every plan needs to have its own,
            // each potential section won't allocate from the same allocator.
            SramAllocator tempAlloc = alloc;

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

            if (!IsPlanAllocated(tempAlloc, plan, tempPleOps, sramBuffer, StatsType::ContinueSection))
            {
                continue;
            }

            // Add current part and plan to the combination,
            // no glue is required. Current part is SISO and
            // has a single input/output
            Combination section =
                comb + Combination(part, std::move(plan), m_PartOrderTable[part.GetPartId()].first, m_GraphOfParts);

            // Options to be estimated
            Combinations options;

            // Next one is the last part of the section
            Combination ended = EndSection(*nextPartGraph, part, section, tempAlloc, numWeightStripes, tempPleOps);

            // Next one is the middle part of the section
            Combination continued =
                ContinueSection(*nextPartGraph, part, section, tempAlloc, numWeightStripes, tempPleOps);

            options = { result, continued, ended };

            result = GetBestCombination(options);
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

        DumpDebugInfo(m_GraphOfParts, { result }, m_Stats, m_DebuggingContext,
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
    if (m_DebuggingContext.m_DebugInfo->m_DumpDebugFiles >= CompilationOptions::DebugLevel::High)
    {
        MakeDirectory(m_DebuggingContext.GetAbsolutePathOutputFileName("FindBestCombinationForPart").c_str());
    }

    TopologicalSortParts();

    assert(IsPartInput(*m_FirstPartAfterSort));
    m_BestCombination = m_BestCombination + FindBestCombinationForPart(*m_FirstPartAfterSort);
}

Combinations Cascading::Combine(const GraphOfParts& parts)
{
    ETHOSN_UNUSED(parts);

    m_Combiner.Run();

    Combination bestComb = m_Combiner.GetBestCombination();

    return { bestComb };
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

    // For each Edge connecting two Parts, which Buffer should the destination part connect to, in order to get that input.
    // A glue may also need to be inserted which connects to this buffer.
    // If there is no glue between two parts, then the source
    // part's output buffer should be re-used directly (as that buffer is then shared between the two plans).
    std::unordered_map<PartInputSlot, Buffer*> edgeConnectionBuffers;

    // For each outgoing edge from a plan, the glue that needs to be inserted there (if any)
    std::unordered_map<PartInputSlot, const Glue*> glues;

    // A glue may be shared between multiple edges
    // each of which should be assigned to a unique
    // output dma of the glue.
    // This is controlled by
    // (1) numEdgesGlue: total number of edges that
    // shares the glue. Shared glue if it is larger than 1
    // (2) incomingGlueCnt: counter of input edge using this glue
    // (3) inbufSramCnt: counter of input edge with buffer location
    //     in SRAM.
    //
    // DRAM ---> SRAM: dedicated glue
    // SRAM --> SRAM or SRAM --> DRAM: shared glue if multiple edges
    //  (1) shared input DMA and DRAM buffer
    //  (2) * output DMA for each SRAM destination
    //      * no DMA for DRAM destination and
    //        and its connecting plan's buffer
    //        is merged with the shared DRAM buffer.
    std::map<const Glue*, uint32_t> numEdgesGlue;

    std::map<const Glue*, uint32_t> incomingGlueCnt;
    std::map<const Glue*, uint32_t> inbufSramCnt;
    std::map<PartInputSlot, Buffer*> dramSharedBuf;

    assert(combination.m_PartIdsInOrder.size() == combination.m_Elems.size());

    // Add each Elem, one at a time. It is assumed that these are toplogically sorted, so we can assume that all
    // parts used as input to each part have already been processed.
    for (auto& partId : combination.m_PartIdsInOrder)
    {
        auto elemIt = combination.m_Elems.find(partId);
        assert(elemIt != combination.m_Elems.end());
        const Plan& plan = *elemIt->second.m_Plan;

        // Add any glues for each incoming edge of this Part, and remember which Op we will need to connect the plan's
        // input buffers to
        std::unordered_map<PartInputSlot, Op*> incomingGlueOps;
        std::vector<PartInputSlot> inputSlots = parts.GetPartInputs(partId);

        for (auto inputSlot : inputSlots)
        {
            auto glueIt      = glues.find(inputSlot);
            const Glue* glue = glueIt != glues.end() ? glueIt->second : nullptr;

            // input buffer in SRAM flag
            Buffer* inputBuffer = plan.GetInputBuffer(inputSlot);
            bool isLocSram      = inputBuffer->m_Location == Location::Sram;

            // shared dram flag is raised if the input buffer location is in DRAM and the glue is shared
            // by multiple edges.
            bool isDramShared = (inputBuffer->m_Location == Location::Dram) && (numEdgesGlue[glue] > 1);

            if (glue != nullptr)
            {
                // A glue can be shared between multiple edges.
                // The shared buffer and input DMA are added to
                // the graph when the first edge is visited.
                if (incomingGlueCnt[glue] == 0)
                {
                    // Add Ops and Buffers from the glue, no connections yet.
                    for (Buffer* b : glue->m_Graph.GetBuffers())
                    {
                        result.AddBuffer(b);
                    }

                    // Only add the input DMA if the edge connect buffer
                    // is not in the SRAM.
                    //
                    // Note the input DMA should be already added
                    // when the glue is first encountered at the
                    // end of the input part.
                    //
                    // If the edge connect buffer is in SRAM, the data
                    // stored in this buffer must be transfered to a
                    // DRAM buffer right after its producing operation.
                    // This is because the part linked to
                    // a glue is always the last of  a section and the data
                    // cannot be retained in the SRAM.
                    // If the edge connect buffer is already in the DRAM,
                    // then the transfer of the data from DRAM to SRAM
                    // should start just before its consuming operation.
                    if (edgeConnectionBuffers.at(inputSlot)->m_Location != Location::Sram)
                    {
                        result.AddOp(glue->m_Graph.GetOp(0));
                    }
                    else
                    {
                        // sanity check. DMA should already be in the op-graph
                        assert(std::find(result.GetOps().begin(), result.GetOps().end(), glue->m_Graph.GetOp(0)) !=
                               result.GetOps().end());
                    }

                    // Add internal connections within the glue
                    // There is only once produce for the buffer
                    for (Buffer* b : glue->m_Graph.GetBuffers())
                    {
                        Op* producer = glue->m_Graph.GetProducer(b);
                        if (producer)
                        {
                            result.SetProducer(b, producer);
                        }
                    }

                    // Connect to the input plan
                    result.AddConsumer(getEffectiveBuffer(edgeConnectionBuffers.at(inputSlot)), glue->m_InputSlot.first,
                                       glue->m_InputSlot.second);
                }

                // Add the output DMA and the corresponding consumer of the buffer
                // associated with this edge
                if (glue->m_OutDmaOffset > 0 && isLocSram)
                {
                    assert(inbufSramCnt[glue] < glue->m_Output.size());
                    result.AddOp(glue->m_Graph.GetOp(glue->m_OutDmaOffset + inbufSramCnt[glue]));

                    // Add internal connections within the glue
                    for (Buffer* b : glue->m_Graph.GetBuffers())
                    {
                        std::pair<Op*, uint32_t> consumer = glue->m_Graph.GetConsumer(b, inbufSramCnt[glue]);
                        assert(consumer.first != nullptr);
                        result.AddConsumer(b, consumer.first, consumer.second);
                    }
                }

                // Remember the output Op from this glue, to connect to our plan
                if (!isDramShared)
                {
                    // No output DMA is used if destination is DRAM and shared glue
                    incomingGlueOps[inputSlot] = glue->m_Output.at(inbufSramCnt[glue]);
                }
                else
                {
                    // Destination DRAM and shared glue:
                    // The glue's share DRAM buffer will be
                    // merged with its connecting plan's buffer
                    assert(glue->m_Graph.GetBuffers().size() == 1);
                    dramSharedBuf[inputSlot] = glue->m_Graph.GetBuffers().at(0);
                }

                incomingGlueCnt[glue] += 1;
                inbufSramCnt[glue] += isLocSram;
                assert(inbufSramCnt[glue] <= incomingGlueCnt[glue]);
            }
        }
        for (Buffer* b : plan.m_OpGraph.GetBuffers())
        {
            // Don't add a buffer if its an input to the plan, and it is shared with the input plan
            // (i.e. no glue between them).
            // Instead, remap it to the one we already have
            Buffer* sharedBuffer = nullptr;
            auto inputSlotIt     = plan.m_InputMappings.find(b);
            if (inputSlotIt != plan.m_InputMappings.end())
            {
                PartInputSlot inputSlot = inputSlotIt->second;
                if (incomingGlueOps.find(inputSlot) == incomingGlueOps.end() &&
                    dramSharedBuf.find(inputSlot) == dramSharedBuf.end())
                {
                    auto edgeBuffer = edgeConnectionBuffers.find(inputSlot);
                    // This code assumed that the combination spans the entire network
                    // The following check is needed as there can be input and outputs nodes
                    // which means there wouldn't be a shared buffer.
                    // This is okay as this combination won't be able to be estimated and thus
                    // another combination will be picked.
                    if (edgeBuffer != edgeConnectionBuffers.end())
                    {
                        // This buffer itself may have been merged (e.g. for plans that have a single buffer for both
                        // input and output, like reinterpret Dram)
                        sharedBuffer = getEffectiveBuffer(edgeBuffer->second);
                    }
                }
                else if (dramSharedBuf.find(inputSlot) != dramSharedBuf.end())
                {
                    // Plan's buffer is shared with saved glue's DRAM buffer
                    sharedBuffer = dramSharedBuf[inputSlot];
                }
            }

            if (sharedBuffer && result.Contains(sharedBuffer))
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
            Buffer* ourBuffer       = input.first;
            PartInputSlot inputSlot = input.second;
            auto glueOpIt           = incomingGlueOps.find(inputSlot);
            if (glueOpIt != incomingGlueOps.end())
            {
                result.SetProducer(ourBuffer, glueOpIt->second);
            }
        }

        // Store our output connections for future plans, and any glues on our outputs
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
            Buffer* outputBuffer = plan.GetOutputBuffer(outputSlot);
            auto inputSlots      = parts.GetConnectedInputSlots(outputSlot);
            for (auto&& inputSlot : inputSlots)
            {
                edgeConnectionBuffers[inputSlot] = outputBuffer;
                auto glueIt                      = elemIt->second.m_Glues.find(inputSlot);
                if (glueIt != elemIt->second.m_Glues.end() && glueIt->second.m_Glue &&
                    !glueIt->second.m_Glue->m_Graph.GetOps().empty())
                {
                    const Glue* outGlue = glueIt->second.m_Glue;

                    glues[inputSlot] = outGlue;

                    // If the glue is visited for the first time, then
                    // initialise the counter.
                    if (numEdgesGlue.find(outGlue) == numEdgesGlue.end())
                    {
                        numEdgesGlue[outGlue] = 1;

                        assert(incomingGlueCnt.find(outGlue) == incomingGlueCnt.end());
                        assert(inbufSramCnt.find(outGlue) == inbufSramCnt.end());
                        incomingGlueCnt[outGlue] = 0;
                        inbufSramCnt[outGlue]    = 0;

                        // If the output buffer is SRAM, then the glue's DMA
                        // that moves the data to the destination DRAM must be
                        // added now. The part connected to a glue is the end
                        // of a section.Output data cannot be kept in the SRAM.
                        if (outputBuffer->m_Location == Location::Sram)
                        {
                            result.AddOp(outGlue->m_Graph.GetOp(0));
                        }
                    }
                    else
                    {
                        numEdgesGlue[outGlue] += 1;
                    }
                }
            }
        }
    }

    return result;
}

void Combiner::SavePartsPlans(const BasePart& part, const Plans& plans) const
{
    if (m_DebuggingContext.m_DebugInfo->m_DumpDebugFiles >= CompilationOptions::DebugLevel::Medium)
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
