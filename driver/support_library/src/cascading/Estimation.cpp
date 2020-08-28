//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "Estimation.hpp"

#include "../Graph.hpp"
#include "../GraphNodes.hpp"
#include "../McePlePass.hpp"
#include "../Utils.hpp"
#include "DebuggingContext.hpp"
#include "EstimationUtils.hpp"
#include "MceEstimationUtils.hpp"
#include "Part.hpp"

#include <iostream>

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
    return mceOp.m_Op == MceOperation::DEPTHWISE_CONVOLUTION ? DataFormat::HWIM : DataFormat::HWIO;
}

}    // namespace

/// Estimates a pass that contains the given Op and possibly some of its neighbours.
/// Removes Ops from the given unestimatedOps set that it has included in its estimation.
PassStats EstimatePassGrownFrom(const OpGraph& opGraph,
                                Op* op,
                                const HardwareCapabilities& capabilities,
                                const EstimationOptions& estimationOpts,
                                std::unordered_set<Op*>& unestimatedOps)
{
    PassStats result;

    assert(unestimatedOps.count(op) > 0);
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
        if (pleOp == nullptr || unestimatedOps.count(pleOp) == 0)
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
            mceOp            = GetObjectAs<MceOp>(opGraph.GetProducer(pleInput));
            if (mceOp != nullptr && unestimatedOps.count(mceOp) == 0)
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
        result.m_Mce = GetMceStats(capabilities, mceOp->m_Stride, mceOp->m_Op, mceOp->m_Algo, mceOp->m_InputStripeShape,
                                   mceOp->m_OutputStripeShape, mceOp->m_WeightsStripeShape);

        // Check for weights as second input to the MceOp
        if (opGraph.GetInputs(mceOp).size() != 2)
        {
            throw NotSupportedException("MceOp must have exactly 2 inputs");
        }
        Buffer* inputBuffer = opGraph.GetInputs(mceOp)[0];
        Buffer* weightsSram = opGraph.GetInputs(mceOp)[1];
        if (weightsSram->m_Location != Location::Sram)
        {
            throw NotSupportedException("Weights buffer must be in Sram");
        }
        DmaOp* dmaOp = GetObjectAs<DmaOp>(opGraph.GetProducer(weightsSram));
        if (dmaOp == nullptr || unestimatedOps.count(dmaOp) == 0)
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
        if (opGraph.GetProducer(weightsDram) != nullptr)
        {
            throw NotSupportedException("Weights Dram buffer must not have a producer");
        }

        weightsTensorInfo = TensorInfo(weightsDram->m_TensorShape, DataType::UINT8_QUANTIZED, GetWeightsFormat(*mceOp),
                                       weightsDram->m_QuantizationInfo);
        result.m_Weights =
            GetWeightsStats(capabilities, *weightsDram->m_EncodedWeights, weightsTensorInfo, weightsSram->m_StripeShape,
                            weightsSram->m_SizeInBytes, inputBuffer->m_TensorShape, inputBuffer->m_StripeShape);

        unestimatedOps.erase(dmaOp);
        unestimatedOps.erase(mceOp);
    }

    // Calculate PLE stats if we have a PleOp
    if (pleOp != nullptr)
    {
        std::vector<TensorShape> inputShapes;
        for (Buffer* inputBuffer : opGraph.GetInputs(pleOp))
        {
            inputShapes.push_back(inputBuffer->m_TensorShape);
        }

        result.m_Ple = GetPleStats(capabilities, inputShapes, pleOp->m_Op);
        unestimatedOps.erase(pleOp);
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
        Location inputLocation = Location::Sram;
        bool isCompressed      = false;
        DmaOp* dmaOp           = GetObjectAs<DmaOp>(opGraph.GetProducer(sramInputBuffer));
        if (dmaOp != nullptr && unestimatedOps.count(dmaOp) > 0)
        {
            if (opGraph.GetInputs(dmaOp).size() != 1)
            {
                throw NotSupportedException("DmaOp must have exactly one input");
            }
            Buffer* dramBuffer = opGraph.GetInputs(dmaOp)[0];
            inputLocation      = dramBuffer->m_Location;
            unestimatedOps.erase(dmaOp);
        }

        // Number of output stripes affects the number of input data reloads for some streaming strategies.
        uint32_t numOutStripeC =
            utils::DivRoundUp(sramOutputBuffer->m_TensorShape[3], sramOutputBuffer->m_StripeShape[3]);

        const InputStats uncompressedStats =
            GetInputStats(capabilities, sramInputBuffer->m_TensorShape, sramInputBuffer->m_StripeShape, inputLocation,
                          sramInputBuffer->m_SizeInBytes, weightsTensorInfo, numOutStripeC);
        const InputStats inputStats =
            isCompressed
                ? AccountForActivationCompression(uncompressedStats, estimationOpts.m_ActivationCompressionSaving)
                : uncompressedStats;
        result.m_Input += inputStats;
    }

    // Check for a DmaOp afterwards, and use that to calculate output stats
    {
        if (sramOutputBuffer->m_Location != Location::Sram)
        {
            throw NotSupportedException("Output buffer from PleOp must be in Sram");
        }
        Location outputLocation   = Location::Sram;
        CompilerDataFormat format = sramOutputBuffer->m_Format;
        bool isCompressed         = false;
        if (opGraph.GetConsumers(sramOutputBuffer).size() == 1)
        {
            DmaOp* dmaOp = GetObjectAs<DmaOp>(opGraph.GetConsumers(sramOutputBuffer)[0].first);
            if (dmaOp != nullptr && unestimatedOps.count(dmaOp) > 0)
            {
                Buffer* dmaOutput = opGraph.GetOutput(dmaOp);
                if (dmaOutput == nullptr)
                {
                    throw NotSupportedException("Output Dma op must have an output");
                }
                outputLocation = dmaOutput->m_Location;
                format         = dmaOutput->m_Format;
                unestimatedOps.erase(dmaOp);
            }
        }

        const TensorShape& roundedUpOutputShape =
            format != CompilerDataFormat::NHWC ? RoundUpHeightAndWidthToBrickGroup(sramOutputBuffer->m_TensorShape)
                                               : sramOutputBuffer->m_TensorShape;

        const OutputStats uncompressedStats =
            GetOutputStats(roundedUpOutputShape, sramOutputBuffer->m_StripeShape, outputLocation);
        result.m_Output = isCompressed ? AccountForActivationCompression(uncompressedStats,
                                                                         estimationOpts.m_ActivationCompressionSaving)
                                       : uncompressedStats;
    }

    return result;
}

NetworkPerformanceData EstimateOpGraph(const OpGraph& opGraph,
                                       const HardwareCapabilities& capabilities,
                                       const EstimationOptions& estimationOpts)
{
    NetworkPerformanceData result;

    // In order to estimate performance using our existing estimation framework, we need to split up the graph into
    // a set of passes, and report stats for each pass independently.
    // In general, a pass consists of an MceOp and/or PleOp, and optional DmaOps before and/or after.

    // We traverse the graph looking for Mce/PleOps, and then look outwards for neighbouring DmaOps to include in that
    // pass. Once we've found them all, we check that there aren't any leftover Ops that haven't been estimated.

    std::unordered_set<Op*> unestimatedOps(opGraph.GetOps().begin(), opGraph.GetOps().end());
    for (Op* op : opGraph.GetOps())
    {
        if (unestimatedOps.count(op) == 0)
        {
            continue;    // This Op already estimated as part of another pass, so nothing else to do for it
        }

        if (IsObjectOfType<MceOp>(op) || IsObjectOfType<PleOp>(op))
        {
            PassPerformanceData passData;
            passData.m_Stats = EstimatePassGrownFrom(opGraph, op, capabilities, estimationOpts, unestimatedOps);
            result.m_Stream.push_back(passData);
        }
    }

    // Check that all Ops have been estimated.
    if (!unestimatedOps.empty())
    {
        throw NotSupportedException("Not all Ops could be estimated");
    }

    return result;
}

}    // namespace support_library
}    // namespace ethosn
