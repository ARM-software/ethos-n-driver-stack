//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "Pass.hpp"

#include "Compiler.hpp"
#include "Utils.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <vector>

namespace ethosn
{
namespace support_library
{

using namespace utils;

command_stream::DataLocation GetCommandDataLocation(BufferLocation bufferLocation)
{
    assert(bufferLocation == BufferLocation::Dram || bufferLocation == BufferLocation::Sram);

    if (bufferLocation == BufferLocation::Sram)
    {
        return command_stream::DataLocation::SRAM;
    }
    else
    {
        return command_stream::DataLocation::DRAM;
    }
}

namespace
{
std::string GetParentIds(const Node& node);

std::string GetIdOfPass(const Node& node)
{
    if (node.GetPass() != nullptr)
    {
        return std::to_string(node.GetPass()->GetId());
    }

    return GetParentIds(node);
}

std::string GetParentIds(const Node& node)
{
    std::stringstream ss;

    ss << '[';
    for (auto it = node.GetInputs().begin(); it != node.GetInputs().end(); ++it)
    {
        const bool isLast = it == std::prev(node.GetInputs().end());
        ss << ' ' << GetIdOfPass(*(*it)->GetSource()) << (isLast ? ' ' : ',');
    }
    ss << ']';

    return ss.str();
}
}    // namespace

void Pass::Estimate(std::vector<PassPerformanceData>& perfStream, const EstimationOptions& estimationOptions)
{
    PassPerformanceData perfData;

    perfData.m_OperationIds = GetCorrespondingOperationIds();
    perfData.m_ParentIds    = GetParentIds(*m_Nodes.front());
    perfData.m_Stats        = GetStats(estimationOptions);

    perfStream.emplace_back(std::move(perfData));

    m_IsEstimated = true;
}

void Pass::PreGenerate(command_stream::CommandStreamBuffer& cmdStream)
{
    m_CommandStreamFirstCommandIdx = cmdStream.GetCount();
}

void Pass::PostGenerate(command_stream::CommandStreamBuffer& cmdStream, bool dumpRam)
{
    m_IsGenerated = true;

    if (dumpRam)
    {
        if (m_Nodes.back()->GetLocation() == ethosn::support_library::BufferLocation::Dram)
        {
            // In order for the end-to-end tests to only validate dram dumps when the output is actually in DRAM,
            // have a different dumpName for passes that have static outputs (output in SRAM).
            const char* const ignoreStr = (m_Nodes.back()->GetLocation() == BufferLocation::Sram) ? "IGNORE_" : "";
            const std::string dumpName  = ignoreStr + std::to_string(m_Nodes.back()->GetShape()[0]) + "_" +
                                         std::to_string(m_Nodes.back()->GetShape()[1]) + "_" +
                                         std::to_string(m_Nodes.back()->GetShape()[2]) + "_" +
                                         std::to_string(m_Nodes.back()->GetShape()[3]) + "_CommandStream_Operation_" +
                                         std::to_string(m_Id) + "_OutputModel_NHWCB.hex";

            ethosn::command_stream::DumpDram cmdStrDumpDram;
            cmdStrDumpDram.m_DramBufferId() = m_Nodes.back()->GetBufferId();

            std::copy(dumpName.begin(), dumpName.end(), cmdStrDumpDram.m_Filename().begin());
            cmdStream.EmplaceBack(cmdStrDumpDram);
        }

        ethosn::command_stream::DumpSram cmdStrDumpSram;
        const std::string dumpName = "output_ce_" + std::to_string(m_Id);
        std::copy(dumpName.begin(), dumpName.end(), cmdStrDumpSram.m_Filename().begin());
        cmdStream.EmplaceBack(cmdStrDumpSram);
    }

    m_CommandStreamLastCommandIdx = cmdStream.GetCount() - 1;
}

std::set<uint32_t> Pass::GetCorrespondingOperationIds() const
{
    std::set<uint32_t> result;
    for (const Node* n : m_Nodes)
    {
        std::set<uint32_t> nodeOperationIds = n->GetCorrespondingOperationIds();
        result.insert(nodeOperationIds.begin(), nodeOperationIds.end());
    }
    return result;
}

namespace
{
constexpr uint32_t GetMinNumSlots(bool needNeighbour, uint32_t numStripes)
{
    return std::min(needNeighbour ? 3U : 1U, numStripes);
}

constexpr uint32_t GetEffectiveSize(uint32_t size, uint32_t stripeSize, uint32_t borderBefore, uint32_t borderAfter)
{
    return size + (borderBefore + borderAfter) * ((size - 1U) / stripeSize);
}

uint32_t GetInputMinNumSlotsForBuffering(const bool isStreamingH,
                                         const bool isStreamingW,
                                         const bool isStreamingC,
                                         const bool needNeighbourStripeH,
                                         const bool needNeighbourStripeW,
                                         const uint32_t numStripesH,
                                         const uint32_t numStripesW)
{
    if (isStreamingC)
    {
        return 2U * GetMinNumSlots(needNeighbourStripeH, numStripesH) *
               GetMinNumSlots(needNeighbourStripeW, numStripesW);
    }
    else if (isStreamingW)
    {
        return GetMinNumSlots(needNeighbourStripeW, numStripesW) + 1U;
    }
    else if (isStreamingH)
    {
        return GetMinNumSlots(needNeighbourStripeH, numStripesH) + 1U;
    }
    return 1U;
}

uint32_t GetInputNumReloads(const bool isStreamingH,
                            const bool isStreamingW,
                            const bool isStreamingC,
                            const TensorInfo& weights,
                            const uint32_t ofmProduced,
                            const uint32_t numOutStripesC)
{
    assert(numOutStripesC > 0);

    if (isStreamingC)
    {
        // Round up the number of output channels (HWIO) or the channel multiplier (HWIM, where M=1).
        return utils::DivRoundUp(weights.m_Dimensions[3], ofmProduced) - 1U;
    }
    else if (isStreamingH || isStreamingW)
    {
        return weights.m_DataFormat == DataFormat::HWIM ? 0 : numOutStripesC - 1U;
    }

    return 0;
}

uint32_t GetInputTotalBytes(const HardwareCapabilities& caps,
                            const TensorShape& shape,
                            const TensorShape& stripeShape,
                            const bool isStreamingH,
                            const bool isStreamingW,
                            const bool isStreamingC,
                            const bool needNeighbourStripeH,
                            const bool needNeighbourStripeW,
                            const uint32_t reloads)
{
    uint32_t borderHeight = 0;
    uint32_t borderWidth  = 0;

    // Calculate the total amount of input data to be transferred included reloading.
    if (needNeighbourStripeW && isStreamingC)
    {
        borderWidth = caps.GetBrickGroupShape()[2];
    }

    if (needNeighbourStripeH && (isStreamingC || (isStreamingH && isStreamingW)))
    {
        borderHeight = caps.GetBoundaryStripeHeight();
    }

    const uint32_t effectiveHeight = GetEffectiveSize(shape[1], stripeShape[1], borderHeight, borderHeight);
    const uint32_t effectiveWidth  = GetEffectiveSize(shape[2], stripeShape[2], borderWidth, borderWidth);

    // Total amount of data
    return (reloads + 1U) * shape[0] * effectiveHeight * effectiveWidth * shape[3];
}

uint32_t GetWeightsNumReloads(const HardwareCapabilities& caps,
                              const TensorShape& inShape,
                              const TensorShape& inStripeShape,
                              const TensorInfo& info,
                              const uint32_t tileSize)
{
    // The input data streaming affects the number of weights data reloads.
    const uint32_t numStripesH = utils::GetNumStripesH(inShape, inStripeShape);
    const uint32_t numStripesW = utils::GetNumStripesW(inShape, inStripeShape);
    const uint32_t numStripesC = utils::GetNumStripesC(inShape, inStripeShape);

    const uint32_t totalSize =
        utils::EstimateWeightSizeBytes(info.m_Dimensions, caps, info.m_DataFormat == DataFormat::HWIM);

    const bool isStreamingHC = numStripesH > 1U && numStripesW == 1U && numStripesC > 1U;

    // Account for the reloading of the weights data, this happens when streaming input data in depth and height.
    return isStreamingHC && (tileSize < totalSize) ? (numStripesW * numStripesH - 1U) : 0;
}

}    // namespace

InputStats Pass::AccountForActivationCompression(InputStats stats, float spaceSavingRatio) const
{
    InputStats ret = stats;
    ret.m_MemoryStats.m_DramNonParallel =
        static_cast<uint32_t>(static_cast<float>(stats.m_MemoryStats.m_DramNonParallel) * (1 - spaceSavingRatio));
    ret.m_MemoryStats.m_DramParallel =
        static_cast<uint32_t>(static_cast<float>(stats.m_MemoryStats.m_DramParallel) * (1 - spaceSavingRatio));
    return ret;
}

InputStats Pass::GetInputStats(const TensorShape& shape,
                               const TensorShape& stripeShape,
                               const BufferLocation location,
                               const uint32_t tileSize,
                               const TensorInfo& weights,
                               const uint32_t numOutStripesC)
{
    InputStats data;

    if (location != BufferLocation::Sram)
    {
        const TensorShape stripeShapeValid = {
            std::min(stripeShape[0], shape[0]),
            std::min(stripeShape[1], shape[1]),
            std::min(stripeShape[2], shape[2]),
            std::min(stripeShape[3], shape[3]),
        };
        const uint32_t stripeSize = stripeShape[0] * stripeShape[1] * stripeShape[2] * stripeShape[3];

        const uint32_t numStripesH = utils::GetNumStripesH(shape, stripeShape);
        const uint32_t numStripesW = utils::GetNumStripesW(shape, stripeShape);
        const uint32_t numStripesC = utils::GetNumStripesC(shape, stripeShape);

        const bool needNeighbourStripeH = weights.m_Dimensions[0] > 1U;
        const bool needNeighbourStripeW = weights.m_Dimensions[1] > 1U;

        // Number of ofm produced per iteration
        const uint32_t ofmProduced = m_Capabilities.GetOfmPerEngine() * m_Capabilities.GetNumberOfEngines();

        // This might change, it doesn't always need all the boundary slots.
        const uint32_t numBoundarySlots = m_Capabilities.GetNumBoundarySlots();

        const bool isStreamingH = numStripesH > 1U;
        const bool isStreamingW = numStripesW > 1U;
        const bool isStreamingC = numStripesC > 1U;

        data.m_StripesStats.m_NumReloads =
            GetInputNumReloads(isStreamingH, isStreamingW, isStreamingC, weights, ofmProduced, numOutStripesC);

        // Calculate the total amount of input data to be transferred included reloading.
        const uint32_t total =
            GetInputTotalBytes(m_Capabilities, shape, stripeShape, isStreamingH, isStreamingW, isStreamingC,
                               needNeighbourStripeH, needNeighbourStripeW, data.m_StripesStats.m_NumReloads);

        // Calculate the minimum amount of data required to start processing.
        uint32_t borderWidth  = 0;
        uint32_t borderHeight = 0;

        if (needNeighbourStripeH && isStreamingH)
        {
            borderHeight =
                (isStreamingC || isStreamingW) ? m_Capabilities.GetBoundaryStripeHeight() : stripeShapeValid[1];
        }

        if (needNeighbourStripeW && isStreamingW)
        {
            borderWidth = isStreamingC ? m_Capabilities.GetBrickGroupShape()[2] : stripeShapeValid[2];
        }

        const bool isUsingBoundarySlots = needNeighbourStripeH && isStreamingH && isStreamingW && !isStreamingC;
        const uint32_t boundarySize     = isUsingBoundarySlots ? borderHeight * stripeShape[2] * stripeShape[3] : 0;
        const uint32_t numStripesInTile = utils::DivRoundUp(tileSize - (boundarySize * numBoundarySlots), stripeSize);

        data.m_MemoryStats.m_DramNonParallel =
            (stripeShapeValid[1] + borderHeight) * (stripeShapeValid[2] + borderWidth) * stripeShapeValid[3];

        // Determine how much data can be transferred in parallel.
        const uint32_t minNumSlotsForBuffering =
            GetInputMinNumSlotsForBuffering(isStreamingH, isStreamingW, isStreamingC, needNeighbourStripeH,
                                            needNeighbourStripeW, numStripesH, numStripesW);

        const bool buffering = numStripesInTile >= minNumSlotsForBuffering;

        if (buffering)
        {
            data.m_MemoryStats.m_DramParallel = total - data.m_MemoryStats.m_DramNonParallel;
        }
        else
        {
            data.m_MemoryStats.m_DramNonParallel = total;
        }

        data.m_StripesStats.m_NumCentralStripes  = utils::GetNumStripesTotal(shape, stripeShape);
        data.m_StripesStats.m_NumBoundaryStripes = isUsingBoundarySlots ? (numStripesH - 1) * numStripesW : 0;
    }
    else
    {
        data.m_MemoryStats.m_Sram = shape[0] * shape[1] * shape[2] * shape[3];
    }

    return data;
}

OutputStats
    Pass::GetOutputStats(const TensorShape& shape, const TensorShape& stripeShape, const BufferLocation location)
{
    OutputStats data;

    const TensorShape& stripeShapeValid = { std::min(stripeShape[0], shape[0]), std::min(stripeShape[1], shape[1]),
                                            std::min(stripeShape[2], shape[2]), std::min(stripeShape[3], shape[3]) };
    const uint32_t stripeSize = stripeShapeValid[0] * stripeShapeValid[1] * stripeShapeValid[2] * stripeShapeValid[3];

    // Total amount of data.
    const uint32_t total = shape[0] * shape[1] * shape[2] * shape[3];

    // Consider the output data transfer only if it is not already in Sram.
    if (location != BufferLocation::Sram)
    {
        // Wait to the final stripe to be copied out if required.
        data.m_MemoryStats.m_DramNonParallel    = stripeSize;
        data.m_MemoryStats.m_DramParallel       = total - data.m_MemoryStats.m_DramNonParallel;
        data.m_StripesStats.m_NumCentralStripes = utils::GetNumStripesTotal(shape, stripeShape);
    }
    else
    {
        data.m_MemoryStats.m_Sram = total;
    }
    return data;
}

WeightsStats Pass::GetWeightsStats(EncodedWeights& encodedWeights,
                                   const TensorInfo& info,
                                   const TensorShape& stripeShape,
                                   const uint32_t tileSize,
                                   const TensorShape& inShape,
                                   const TensorShape& inStripeShape)
{
    WeightsStats data;

    const uint32_t stripeSize =
        utils::EstimateWeightSizeBytes(stripeShape, m_Capabilities, info.m_DataFormat == DataFormat::HWIM);

    // Account for the reloading of the weights data, this happens when streaming input data in depth and height.
    data.m_StripesStats.m_NumCentralStripes = static_cast<uint32_t>(encodedWeights.m_Metadata.size());
    data.m_StripesStats.m_NumReloads = GetWeightsNumReloads(m_Capabilities, inShape, inStripeShape, info, tileSize);

    // Check if there is more than a stripe in the tile.
    const bool buffering = tileSize > stripeSize;

    if (buffering)
    {
        // At least a weights stripe needs to be in internal memory before starting the processing, use the metadata information
        // to get the amount of data.
        data.m_MemoryStats.m_DramNonParallel = encodedWeights.m_Metadata[0].m_Size;
        data.m_MemoryStats.m_DramParallel =
            (data.m_StripesStats.m_NumReloads + 1U) * static_cast<uint32_t>(encodedWeights.m_Data.size()) -
            data.m_MemoryStats.m_DramNonParallel;
    }
    else
    {
        data.m_MemoryStats.m_DramNonParallel =
            (data.m_StripesStats.m_NumReloads + 1U) * static_cast<uint32_t>(encodedWeights.m_Data.size());
    }
    // Clamp the savings to 0
    // if the weights are uncompressable then the encoded weight size is larger than the weights provided
    // because of the header
    data.m_WeightCompressionSavings = std::max(0.0f, 1.0f - (static_cast<float>(encodedWeights.m_Data.size()) /
                                                             static_cast<float>(GetNumElements(info.m_Dimensions))));

    return data;
}

ethosn::support_library::DotAttributes Pass::GetDotAttributes()
{
    std::stringstream stream;
    stream << std::hex << m_Nodes.back()->GetOutputSramOffset();
    std::string outputSramOffset =
        m_Nodes.back()->GetLocation() == BufferLocation::Sram ? "\nOutputSramOffset " + stream.str() : "";
    return DotAttributes(std::to_string(m_Id),
                         "Pass " + std::to_string(m_Id) + "\nCommands " +
                             std::to_string(m_CommandStreamFirstCommandIdx) + "-" +
                             std::to_string(m_CommandStreamLastCommandIdx) + "\nOutputSramOffset " + outputSramOffset,
                         "black");
}

ConcatNode* FindConcatNode(Node* node)
{
    for (const auto& n : node->GetOutputs())
    {
        if (dynamic_cast<ConcatNode*>(n->GetDestination()))
        {
            return dynamic_cast<ConcatNode*>(n->GetDestination());
        }
    }
    return nullptr;
}

std::pair<TensorShape, TensorShape> CalculateConcatSupertensorInfo(Node* inputToConcat, ConcatNode* concatNode)
{
    assert(inputToConcat);
    assert(concatNode);
    uint32_t axis = concatNode->GetAxis();

    TensorShape offset = { 0, 0, 0, 0 };
    for (uint32_t inputIdx = 0; inputIdx < concatNode->GetInputs().size(); ++inputIdx)
    {
        if (concatNode->GetInput(inputIdx)->GetSource() == inputToConcat)
        {
            break;
        }
        offset[axis] += concatNode->GetInputShape(inputIdx)[axis];
    }
    std::pair<TensorShape, TensorShape> res;
    res.first  = offset;
    res.second = concatNode->GetShape();
    return res;
}

}    // namespace support_library
}    // namespace ethosn
