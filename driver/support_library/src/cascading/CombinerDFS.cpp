//
// Copyright © 2021 Arm Limited.
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
bool Combiner::IsPlanAllocated(SramAllocator& alloc, const Plan& plan) const
{
    // Get input and total SRAM sizes required for the plan
    const SizeInBytes sTotSize = GetTotSizeInBytes(plan);
    const SizeInBytes sInSize  = GetInputsSizeInBytes(plan);

    using Allocated = std::pair<bool, uint32_t>;
    Allocated allocated;
    SramAllocator localAlloc = alloc;

    // We are not yet sure what could be a good userId here so we are using zero
    SramAllocator::UserId userId = 0;

    // Note this function assumes the plan can be merged with the combination
    // that is associated with the sram allocation. Therefore, the additional
    // sram usage of this plan is the total size - input size.
    allocated = localAlloc.Allocate(userId, (sTotSize.m_Tot - sInSize.m_Tot) / m_Caps.GetNumberOfSrams(),
                                    AllocationPreference::Start);

    if (allocated.first)
    {
        alloc = localAlloc;
        return true;
    }
    else
    {
        return false;
    }
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

    // Plans in a section must use the same streaming strategy
    for (auto& inputMapping : reference.m_InputMappings)
    {
        const Buffer* referenceInBuffer = inputMapping.first;
        const bool refSplitH =
            GetHeight(referenceInBuffer->m_StripeShape) < GetHeight(referenceInBuffer->m_TensorShape);
        const bool refSplitW = GetWidth(referenceInBuffer->m_StripeShape) < GetWidth(referenceInBuffer->m_TensorShape);
        const bool refSplitC =
            GetChannels(referenceInBuffer->m_StripeShape) < GetChannels(referenceInBuffer->m_TensorShape);

        const bool currSplitH = GetHeight(currentInBuffer->m_StripeShape) < GetHeight(currentInBuffer->m_TensorShape);
        const bool currSplitW = GetWidth(currentInBuffer->m_StripeShape) < GetWidth(currentInBuffer->m_TensorShape);
        const bool currSplitC =
            GetChannels(currentInBuffer->m_StripeShape) < GetChannels(currentInBuffer->m_TensorShape);
        if (refSplitH != currSplitH || refSplitW != currSplitW || refSplitC != currSplitC)
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

// Try to merge plans from the given Part onto the given Combination.
// This may not happen because:
//  - Plan cannot be merged e.g. different strategies
//  - Plan is not allowed
//  - Plan buffers do not fit in SRAM i.e. merged plans
//    in the seciton take up all the memory
Combination Combiner::ContinueSection(const BasePart& part,
                                      const BasePart& sPart,
                                      const Combination& comb,
                                      const SramAllocator& alloc)
{
    UpdateStats(StatsType::ContinueSection);
    const PartId partId = part.GetPartId();

    const Plan& sPlan = GetPlanForPartFromCombination(sPart, comb);

    std::vector<PartConnection> edges = m_GraphOfParts.GetConnectionsBetween(sPart.GetPartId(), part.GetPartId());

    // Sanity check: section is continued. It must be the single output of
    // its source part.
    assert(edges.size() == 1);

    // End the current section and start a new one.
    // There is a single edge between the combination comb and
    // and the current part
    // Note only gluing to the source part that belongs to the same section.
    // Other glue to other source parts will be taken care of by (1) either
    // themselves when branching (2) or by this part later when other sections
    // also end here.
    // This is because we can only guarantee that the part is the only destination
    // of its source part from the same section.
    // Others could have multiple destinations (branching)
    Combination result = GluePartToCombinationDestToSrcs(part, comb + FindBestCombinationForPart(part), edges);

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

        // sanity check SISO is the only use case.

        assert(m_GraphOfParts.GetPartInputs(part.GetPartId()).size() == 1 &&
               m_GraphOfParts.GetPartOutputs(part.GetPartId()).size() == 1);

        // Next Part in graph that is sorted in topological order
        const BasePart* nextPartGraph = GetNextPart(&part);

        // destination part
        assert(m_GraphOfParts.GetDestinationParts(partId).size() == 1);
        const BasePart& destPart = m_GraphOfParts.GetPart(m_GraphOfParts.GetDestinationParts(partId).at(0).m_PartId);

        // flag to indicate if the next part can be in the same section of the current part
        const bool nextPartSameSection =
            nextPartGraph->GetPartId() == m_GraphOfParts.GetDestinationParts(partId).at(0).m_PartId;

        Plans plans = GetPlansCached(part, CascadeType::Middle, ethosn::command_stream::BlockConfig{}, nullptr, 0);

        const PartConnection& edge = edges.at(0);

        for (const auto& plan : plans)
        {
            // Make a copy of the allocator since every plan needs to have its own,
            // each potential section won't allocate from the same allocator.
            SramAllocator tempAlloc = alloc;

            if (!ArePlansCompatible(sPlan, *plan.get(), edge))
            {
                continue;
            }

            if (!IsPlanAllocated(tempAlloc, *plan.get()))
            {
                continue;
            }

            if (!ArePlansAllowedToMerge(sPlan, *plan.get(), edge))
            {
                continue;
            }

            // Add current part and plan to the combination,
            // no glue is required. Current part is SISO and
            // has a single input/output
            Combination section =
                comb + Combination(part, plan, m_PartOrderTable[part.GetPartId()].first, m_GraphOfParts);
            // Options to be estimated
            Combinations options;
            if (nextPartSameSection)
            {
                // A section can only continue if the next part in graph
                // is the same as its destination.
                options = { result, ContinueSection(destPart, part, section, tempAlloc) };
            }
            else
            {
                // Will have to start a new section with the next part.
                // It is likely in a different branch of the graph.
                section = section + FindBestCombinationForPart(*nextPartGraph);
                options = { result, section };
            }
            result = GetBestCombination(options);
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
Combination Combiner::FindBestCombinationForPartImpl(const BasePart& part)
{
    PartId partId = part.GetPartId();
    // This is going to be a new combination, so this
    // is empty initialized
    Combination result = {};

    Plans plans = GetPlansCached(part, CascadeType::Lonely, ethosn::command_stream::BlockConfig{}, nullptr, 0);

    //  Next part
    const BasePart* nextPartGraph = GetNextPart(&part);

    // A section is continued only if the next part in the graph
    // is the same as its only destination part.
    bool nextPartSameSection = false;
    if (!m_GraphOfParts.GetDestinationParts(partId).empty())
    {
        nextPartSameSection = nextPartGraph->GetPartId() == m_GraphOfParts.GetDestinationParts(partId).at(0).m_PartId;
    }

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
        const BasePart& nextPart = m_GraphOfParts.GetPart(m_GraphOfParts.GetDestinationParts(partId).at(0).m_PartId);

        for (const auto& plan : plans)
        {
            if (!IsPlanInputGlueable(*plan.get()))
            {
                continue;
            }

            // This is the start of a new section, reset the allocated Sram
            SramAllocator alloc(m_Caps.GetTotalSramSize() / m_Caps.GetNumberOfSrams());

            Combination head(part, plan, m_PartOrderTable[part.GetPartId()].first, m_GraphOfParts);
            Combinations options = { result, ContinueSection(nextPart, part, head, alloc) };
            result               = GetBestCombination(options);
        }
    }
    else
    {
        // ContinueSection operates only on SISO parts
        // so Output parts and Multiple Output parts
        // cannot be merged for now

        // Select best plan for the part
        for (const auto& plan : plans)
        {
            if (!IsPlanInputGlueable(*plan.get()))
            {
                continue;
            }

            // Glue will be added later on
            Combination head(part, plan, m_PartOrderTable[part.GetPartId()].first, m_GraphOfParts);
            Combinations options = { result, head };
            result               = GetBestCombination(options);
        }

        if (nextPartGraph != nullptr)
        {
            result = result + FindBestCombinationForPart(*nextPartGraph);

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

            // Each of it destination part will start its own new section.
            // Therefore they all need to be glued with their source.
            std::vector<PartConnection> destPartEdge = m_GraphOfParts.GetDestinationConnections(partId);

            if (destPartEdge.empty() == false)
            {
                result = GluePartToCombinationSrcToDests(part, result, destPartEdge);
            }
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

                    // Input DMA is always added when the glue
                    // is visited for the first time
                    result.AddOp(glue->m_Graph.GetOp(0));

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
                    result.AddConsumer(edgeConnectionBuffers.at(inputSlot), glue->m_InputSlot.first,
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
                if (incomingGlueOps.find(inputSlot) == incomingGlueOps.end())
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
        for (auto output : plan.m_OutputMappings)
        {
            PartOutputSlot outputSlot = output.second;
            auto inputSlots           = parts.GetConnectedInputSlots(outputSlot);
            for (auto&& inputSlot : inputSlots)
            {
                edgeConnectionBuffers[inputSlot] = output.first;
                auto glueIt                      = elemIt->second.m_Glues.find(inputSlot);
                if (glueIt != elemIt->second.m_Glues.end() && glueIt->second.m_Glue &&
                    !glueIt->second.m_Glue->m_Graph.GetOps().empty())
                {
                    const Glue* outGlue = glueIt->second.m_Glue;

                    glues[inputSlot] = outGlue;

                    // If the glue is visisted for the first time, then
                    // initialise the counter.
                    if (numEdgesGlue.find(outGlue) == numEdgesGlue.end())
                    {
                        numEdgesGlue[outGlue] = 1;

                        assert(incomingGlueCnt.find(outGlue) == incomingGlueCnt.end());
                        assert(inbufSramCnt.find(outGlue) == inbufSramCnt.end());
                        incomingGlueCnt[outGlue] = 0;
                        inbufSramCnt[outGlue]    = 0;
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
