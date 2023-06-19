//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "Estimation.hpp"

#include "../Utils.hpp"
#include "DebuggingContext.hpp"
#include "EstimationUtils.hpp"
#include "MceEstimationUtils.hpp"
#include "Part.hpp"

#include <ethosn_utils/Strings.hpp>

using namespace std;
using namespace ethosn::support_library::utils;

namespace ethosn
{
namespace support_library
{

namespace
{

DataFormat GetWeightsFormat(const MceOp& mceOp)
{
    return mceOp.m_Op == command_stream::MceOperation::DEPTHWISE_CONVOLUTION ? DataFormat::HWIM : DataFormat::HWIO;
}

}    // namespace

/// Estimates a conversion pass that contains the given DmaOp and possibly some of its neighbours.
/// Removes Ops from the given unprocessed set that it has included in its estimation.
EstimatedPass EstimateConversionPassGrownFrom(const OpGraph& opGraph,
                                              Op* op,
                                              const EstimationOptions& estimationOpts,
                                              std::unordered_set<Op*>& unprocessed)
{
    EstimatedPass result;

    auto includeOp = [&](Op* op) {
        unprocessed.erase(op);
        result.m_Ops.push_back(op);
    };

    assert(unprocessed.count(op) > 0);
    DmaOp* dmaOp = GetObjectAs<DmaOp>(op);
    assert(dmaOp != nullptr);

    auto inputBuffers = opGraph.GetInputs(dmaOp);
    if (inputBuffers.size() != 1)
    {
        throw NotSupportedException("The DmaOp must have only 1 input buffer");
    }
    Buffer* inputBuffer = inputBuffers[0];

    Buffer* sramBuffer = opGraph.GetOutput(dmaOp);
    if (sramBuffer == nullptr)
    {
        throw NotSupportedException("The DmaOp must have an output buffer");
    }

    if (sramBuffer->m_Location != Location::Sram)
    {
        throw NotSupportedException("The DmaOp's output buffer must be in Sram");
    }

    const auto& sramBufferConsumers = opGraph.GetConsumers(sramBuffer);
    if (sramBufferConsumers.size() != 1)
    {
        throw NotSupportedException("The DmaOps output buffer must have only 1 consumer");
    }

    DmaOp* secondDmaOp = GetObjectAs<DmaOp>(sramBufferConsumers[0].first);
    if (secondDmaOp == nullptr)
    {
        throw NotSupportedException("DmaOp must have a second Dma Op for a conversion pass");
    }

    Buffer* outputBuffer = opGraph.GetOutput(secondDmaOp);
    if (outputBuffer == nullptr)
    {
        throw NotSupportedException("The second DmaOp must have an output buffer");
    }

    auto isInputCompressed  = IsCompressed(inputBuffer->m_Format);
    auto isOutputCompressed = IsCompressed(outputBuffer->m_Format);
    includeOp(dmaOp);
    includeOp(secondDmaOp);

    ConversionData inputConversionData;
    // Use the SRAM tensor shape, which might be different from the DRAM tensor shape for reshapes
    inputConversionData.tensorShape = sramBuffer->m_TensorShape;
    // The input and output buffers are in DRAM so don't have stripes, use the sram buffer to get the stripe information
    inputConversionData.stripeShape = sramBuffer->Sram()->m_StripeShape;
    inputConversionData.isNhwc      = inputBuffer->m_Format == CascadingBufferFormat::NHWC;
    bool isDramToDram = inputBuffer->m_Location == Location::Dram && outputBuffer->m_Location == Location::Dram;

    if (!isDramToDram)
    {
        throw NotSupportedException("Only DRAM to DRAM conversion passes are supported at the moment");
    }

    ConversionData outputConversionData;
    // Use the SRAM tensor shape, which might be different from the DRAM tensor shape for reshapes
    outputConversionData.tensorShape = sramBuffer->m_TensorShape;
    outputConversionData.stripeShape = sramBuffer->Sram()->m_StripeShape;
    outputConversionData.isNhwc      = outputBuffer->m_Format == CascadingBufferFormat::NHWC;

    result.m_Stats = GetConversionStats(inputConversionData, outputConversionData, isDramToDram);

    result.m_Stats.m_Input.m_StripesStats =
        AccountForDmaChunking(result.m_Stats.m_Input.m_StripesStats, *sramBuffer->Sram(), *inputBuffer->Dram(), false);
    result.m_Stats.m_Output.m_StripesStats =
        AccountForDmaChunking(result.m_Stats.m_Output.m_StripesStats, *sramBuffer->Sram(), *outputBuffer->Dram(), true);

    if (isInputCompressed)
    {
        result.m_Stats.m_Input =
            AccountForActivationCompression(result.m_Stats.m_Input, estimationOpts.m_ActivationCompressionSaving);
    }
    if (isOutputCompressed)
    {
        result.m_Stats.m_Output =
            AccountForActivationCompression(result.m_Stats.m_Output, estimationOpts.m_ActivationCompressionSaving);
    }
    return result;
}

/// Estimates a pass that contains the given Op and possibly some of its neighbours.
/// Removes Ops from the given unprocessed set that it has included in its estimation.
EstimatedPass EstimatePassGrownFrom(const OpGraph& opGraph,
                                    Op* op,
                                    const HardwareCapabilities& capabilities,
                                    const EstimationOptions& estimationOpts,
                                    std::unordered_set<Op*>& unprocessed)
{
    EstimatedPass result;

    auto includeOp = [&](Op* op) {
        unprocessed.erase(op);
        result.m_Ops.push_back(op);
    };

    assert(unprocessed.count(op) > 0);
    MceOp* mceOp = GetObjectAs<MceOp>(op);
    PleOp* pleOp = GetObjectAs<PleOp>(op);
    assert(mceOp != nullptr || pleOp != nullptr);

    if (mceOp != nullptr)
    {
        // We require a PleOp immediately after
        Buffer* mceOutput = opGraph.GetOutput(mceOp);
        if (mceOutput == nullptr || mceOutput->m_Location != Location::PleInputSram)
        {
            throw NotSupportedException("MceOp must have an output buffer in PleInputSram");
        }
        if (opGraph.GetConsumers(mceOutput).size() != 1)
        {
            throw NotSupportedException("MceOp output buffer must be consumed by exactly one Op");
        }
        pleOp = GetObjectAs<PleOp>(opGraph.GetConsumers(mceOutput)[0].first);
        if (pleOp == nullptr || unprocessed.count(pleOp) == 0)
        {
            throw NotSupportedException(
                "MceOp output buffer consumer must be a PleOp which hasn't already been estimated");
        }
    }
    else if (pleOp != nullptr)
    {
        // We may have an MceOp before us
        if (opGraph.GetInputs(pleOp).size() == 1)
        {
            Buffer* pleInput = opGraph.GetInputs(pleOp)[0];
            mceOp            = GetObjectAs<MceOp>(opGraph.GetSingleProducer(pleInput));
            if (mceOp != nullptr && unprocessed.count(mceOp) == 0)
            {
                throw NotSupportedException(
                    "If PleOp's input is from an MceOp, that MceOp can't already have been estimated");
            }
        }
    }

    // Calculate MCE and weight stats if we have an MceOp
    // Remember weights info as we need it for the input stats. Set a default in case we have no weights (i.e. Ple-only)
    TensorInfo weightsTensorInfo = {
        { { 1, 1, 1, 1 } },
        DataType::UINT8_QUANTIZED,
        DataFormat::HWIM,
        { 0, 0.1f },
    };
    if (mceOp != nullptr)
    {
        // Check for weights as second input to the MceOp
        if (opGraph.GetInputs(mceOp).size() != 2)
        {
            throw NotSupportedException("MceOp must have exactly 2 inputs");
        }
        Buffer* inputBuffer     = opGraph.GetInputs(mceOp)[0];
        Buffer* weightsSram     = opGraph.GetInputs(mceOp)[1];
        Buffer* mceOutputBuffer = opGraph.GetOutput(mceOp);    // Validated above that this is non-null

        result.m_Stats.m_Mce =
            GetMceStats(capabilities, mceOp->m_Stride, mceOp->m_Op, mceOp->m_Algo, inputBuffer->m_TensorShape,
                        mceOutputBuffer->m_TensorShape, weightsSram->m_TensorShape, mceOp->m_BlockConfig);

        if (weightsSram->m_Location != Location::Sram)
        {
            throw NotSupportedException("Weights buffer must be in Sram");
        }
        DmaOp* dmaOp = GetObjectAs<DmaOp>(opGraph.GetSingleProducer(weightsSram));
        if (dmaOp == nullptr || unprocessed.count(dmaOp) == 0)
        {
            throw NotSupportedException("Weights buffer must be Dma'd");
        }
        if (opGraph.GetInputs(dmaOp).size() != 1)
        {
            throw NotSupportedException("DmaOp must have exactly one input");
        }
        Buffer* weightsDram = opGraph.GetInputs(dmaOp)[0];
        if (weightsDram->m_Location != Location::Dram)
        {
            throw NotSupportedException("Weights buffer must be Dma'd from Dram");
        }
        if (opGraph.GetSingleProducer(weightsDram) != nullptr)
        {
            throw NotSupportedException("Weights Dram buffer must not have a producer");
        }

        weightsTensorInfo = TensorInfo(weightsDram->m_TensorShape, DataType::UINT8_QUANTIZED, GetWeightsFormat(*mceOp),
                                       weightsDram->m_QuantizationInfo);
        result.m_Stats.m_Weights =
            GetWeightsStats(capabilities, *weightsDram->Dram()->m_EncodedWeights, weightsTensorInfo,
                            weightsSram->m_SizeInBytes, inputBuffer->m_TensorShape, inputBuffer->Sram()->m_StripeShape);

        includeOp(dmaOp);
        includeOp(mceOp);
    }

    // Calculate PLE stats if we have a PleOp
    if (pleOp != nullptr)
    {
        std::vector<TensorShape> inputShapes;
        for (Buffer* inputBuffer : opGraph.GetInputs(pleOp))
        {
            inputShapes.push_back(inputBuffer->m_TensorShape);
        }

        result.m_Stats.m_Ple = GetPleStats(capabilities, inputShapes, pleOp->m_Op, pleOp->m_BlockConfig);
        includeOp(pleOp);
    }

    Op* frontOp = mceOp ? static_cast<Op*>(mceOp) : pleOp;
    Op* backOp  = pleOp;

    Buffer* sramOutputBuffer = opGraph.GetOutput(backOp);
    if (sramOutputBuffer == nullptr)
    {
        throw NotSupportedException("Must have an output buffer");
    }

    // Check for a DmaOp beforehand, and use that to calculate input stats
    // Do this for each input
    for (uint32_t inputIdx = 0; inputIdx < opGraph.GetInputs(frontOp).size(); ++inputIdx)
    {
        if (frontOp == mceOp && inputIdx > 0)
        {
            // MceOps have only a single "regular" input - the second is for the weights which have already been handled
            // specially - see above.
            break;
        }

        // Check if this input is DMA'd into Sram, as this will affect the calculation of input stats.
        Buffer* sramInputBuffer = opGraph.GetInputs(frontOp)[inputIdx];
        if (sramInputBuffer->m_Location != Location::Sram)
        {
            throw NotSupportedException("Input buffer to PleOp/MceOp must be in Sram");
        }
        Buffer* dramBuffer = nullptr;
        DmaOp* dmaOp       = GetObjectAs<DmaOp>(opGraph.GetSingleProducer(sramInputBuffer));
        if (dmaOp != nullptr && unprocessed.count(dmaOp) > 0)
        {
            if (opGraph.GetInputs(dmaOp).size() != 1)
            {
                throw NotSupportedException("DmaOp must have exactly one input");
            }
            dramBuffer = opGraph.GetInputs(dmaOp)[0];
            includeOp(dmaOp);
        }

        InputStats stats = GetInputStatsCascading(*sramInputBuffer->Sram(), weightsTensorInfo.m_Dimensions,
                                                  dramBuffer != nullptr ? dramBuffer->m_Format
                                                                        : utils::Optional<CascadingBufferFormat>{});
        if (dramBuffer != nullptr)
        {
            stats.m_StripesStats =
                AccountForDmaChunking(stats.m_StripesStats, *sramInputBuffer->Sram(), *dramBuffer->Dram(), false);
            if (IsCompressed(dramBuffer->m_Format))
            {
                stats = AccountForActivationCompression(stats, estimationOpts.m_ActivationCompressionSaving);
            }
        }
        result.m_Stats.m_Input += stats;
    }

    // Check for a DmaOp afterwards, and use that to calculate output stats
    {
        if (sramOutputBuffer->m_Location != Location::Sram)
        {
            throw NotSupportedException("Output buffer from PleOp must be in Sram");
        }
        Buffer* dramBuffer = nullptr;
        for (uint32_t i = 0; i < uint32_t(opGraph.GetConsumers(sramOutputBuffer).size()); i++)
        {
            DmaOp* dmaOp = GetObjectAs<DmaOp>(opGraph.GetConsumers(sramOutputBuffer)[i].first);
            if (dmaOp != nullptr && unprocessed.count(dmaOp) > 0)
            {
                dramBuffer = opGraph.GetOutput(dmaOp);
                if (dramBuffer == nullptr)
                {
                    throw NotSupportedException("Output Dma op must have an output");
                }
                includeOp(dmaOp);
            }
        }

        OutputStats stats = GetOutputStatsCascading(*sramOutputBuffer->Sram(),
                                                    dramBuffer != nullptr ? dramBuffer->m_Format
                                                                          : utils::Optional<CascadingBufferFormat>{});
        if (dramBuffer != nullptr)
        {
            stats.m_StripesStats =
                AccountForDmaChunking(stats.m_StripesStats, *sramOutputBuffer->Sram(), *dramBuffer->Dram(), true);
            if (IsCompressed(dramBuffer->m_Format))
            {
                stats = AccountForActivationCompression(stats, estimationOpts.m_ActivationCompressionSaving);
            }
        }
        result.m_Stats.m_Output = stats;
    }

    return result;
}

namespace
{

std::string GetParentIds(const std::vector<Op*>& ops, const EstimatedOpGraph& estimatedOpGraph, const OpGraph& opGraph);

std::string GetIdOfPass(Op* op, const EstimatedOpGraph& estimatedOpGraph, const OpGraph& opGraph)
{
    auto passIt = estimatedOpGraph.m_OpToPass.find(op);
    if (passIt != estimatedOpGraph.m_OpToPass.end())
    {
        return std::to_string(passIt->second);
    }

    return GetParentIds({ op }, estimatedOpGraph, opGraph);
}

std::string GetParentIds(const std::vector<Op*>& ops, const EstimatedOpGraph& estimatedOpGraph, const OpGraph& opGraph)
{
    std::unordered_set<Op*> opsSet(ops.begin(), ops.end());    // For fast lookups

    std::vector<std::string> parts;
    for (Op* op : ops)
    {
        OpGraph::BufferList inputs = opGraph.GetInputs(op);

        for (auto&& input : inputs)
        {
            for (Op* producer : opGraph.GetProducers(input))
            {
                // Don't traverse any further if the Buffer is not connected (e.g. network input) or
                // it's connected to something else in the same Part.
                if (opsSet.count(producer) > 0)
                {
                    continue;
                }

                parts.push_back(GetIdOfPass(producer, estimatedOpGraph, opGraph));
            }
        }
    }

    if (parts.size() == 0)
    {
        return "[ [] ]";
    }
    else
    {
        return "[ " + ethosn::utils::Join(", ", parts, [](const std::string& x) { return x; }) + " ]";
    }
}

}    // namespace

EstimatedOpGraph EstimateOpGraph(const OpGraph& opGraph,
                                 const HardwareCapabilities& capabilities,
                                 const EstimationOptions& estimationOpts)
{
    // In order to estimate performance using our existing estimation framework, we need to split up the graph into
    // a set of passes, and report stats for each pass independently.
    // An MCE/PLE pass consists of an MceOp and/or PleOp, and optional DmaOps before and/or after.
    // A Conversion pass consists of 2 DmaOps from Dram to Dram.

    // We traverse the graph looking for Mce/PleOps, and then look outwards for neighbouring DmaOps to include in that
    // pass.
    std::vector<EstimatedPass> unsortedPasses;
    std::unordered_set<Op*> unprocessedOps(opGraph.GetOps().begin(), opGraph.GetOps().end());
    std::map<uint32_t, std::string> operationIdFailureReasons;

    for (Op* op : opGraph.GetOps())
    {
        if (unprocessedOps.count(op) == 0)
        {
            continue;    // This Op already estimated as part of another pass, so nothing else to do for it
        }

        if (IsObjectOfType<MceOp>(op) || IsObjectOfType<PleOp>(op))
        {
            EstimatedPass estimatedPass;

            try
            {
                estimatedPass = EstimatePassGrownFrom(opGraph, op, capabilities, estimationOpts, unprocessedOps);
            }
            catch (const NotSupportedException&)
            {
                // Some Ops will go unestimated, but this is fine. They will be reported in the result from this function
                continue;
            }
            unsortedPasses.push_back(estimatedPass);
        }
    }

    // Once we've found all the MCE/PLE passes we now estimate conversion passes from any remaining unestimated ops.
    if (!unprocessedOps.empty())
    {
        for (Op* op : opGraph.GetOps())
        {
            if (unprocessedOps.count(op) == 0)
            {
                continue;    // This Op already estimated as part of another pass, so nothing else to do for it
            }

            EstimatedPass estimatedPass;

            if (IsObjectOfType<DmaOp>(op))
            {
                try
                {
                    estimatedPass = EstimateConversionPassGrownFrom(opGraph, op, estimationOpts, unprocessedOps);
                }
                catch (const NotSupportedException&)
                {
                    // Some Ops will go unestimated, but this is fine. They will be reported in the result from this function
                    continue;
                }

                unsortedPasses.push_back(estimatedPass);
            }
            else if (IsObjectOfType<EstimateOnlyOp>(op))
            {
                unprocessedOps.erase(op);
                EstimateOnlyOp* estimateOnlyOp = GetObjectAs<EstimateOnlyOp>(op);

                for (auto it : op->m_OperationIds)
                {
                    operationIdFailureReasons[it] = "Could not be estimated and has zero performance impact. Reason: " +
                                                    estimateOnlyOp->m_ReasonForEstimateOnly;
                }
            }
        }
    }

    // The estimated passes we created above are not necessarily in topological order,
    // so now we sort them, whilst also turning them into PassPerformanceData structs
    // for our final result
    std::map<Op*, uint32_t> opToUnsortedPassIdx;
    for (uint32_t unsortedPassIdx = 0; unsortedPassIdx < unsortedPasses.size(); ++unsortedPassIdx)
    {
        for (Op* o : unsortedPasses[unsortedPassIdx].m_Ops)
        {
            opToUnsortedPassIdx[o] = unsortedPassIdx;
        }
    }

    EstimatedOpGraph result;
    result.m_PerfData.m_OperationIdFailureReasons = operationIdFailureReasons;
    // The Ops in the OpGraph should already be sorted into execution order, so go through this order
    // to determine the order of the passes
    std::unordered_set<uint32_t> unsortedPassIdxsAdded;    // Tracks the passes already added
    for (Op* op : opGraph.GetOps())
    {
        auto unsortedPassIdxIt = opToUnsortedPassIdx.find(op);
        // Not all Ops will have been placed in a Pass, for example EstimateOnlyOps, or ops which we failed to estimate
        if (unsortedPassIdxIt == opToUnsortedPassIdx.end())
        {
            continue;
        }
        // Don't add the same pass again (multiple Ops will belong to the same pass)
        bool alreadyAdded = unsortedPassIdxsAdded.count(unsortedPassIdxIt->second) > 0;
        if (alreadyAdded)
        {
            continue;
        }

        // Create the PassPerformanceData for this pass and it to the result
        PassPerformanceData passData;
        const uint32_t sortedPassIdx       = static_cast<uint32_t>(result.m_PerfData.m_Stream.size());
        const EstimatedPass& estimatedPass = unsortedPasses[unsortedPassIdxIt->second];
        passData.m_ParentIds               = GetParentIds(estimatedPass.m_Ops, result, opGraph);
        passData.m_Stats                   = estimatedPass.m_Stats;

        for (Op* op : estimatedPass.m_Ops)
        {
            passData.m_OperationIds.insert(op->m_OperationIds.begin(), op->m_OperationIds.end());
            result.m_OpToPass[op] = sortedPassIdx;
        }

        result.m_PerfData.m_Stream.push_back(passData);

        // Record that we processed this pass, to prevent doing so again.
        unsortedPassIdxsAdded.insert(unsortedPassIdxIt->second);
    }

    // Check that all Ops have been estimated.
    if (!unprocessedOps.empty())
    {
        throw NotSupportedException("Not all Ops could be estimated");
    }

    result.m_Metric = CalculateMetric(result.m_PerfData);

    return result;
}

}    // namespace support_library
}    // namespace ethosn
