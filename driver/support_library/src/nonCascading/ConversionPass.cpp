//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "ConversionPass.hpp"

#include "Compiler.hpp"
#include "GraphNodes.hpp"
#include "SramAllocator.hpp"
#include "Utils.hpp"
#include "cascading/EstimationUtils.hpp"

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

bool ConversionPass::ChooseAndSetupStripe(const HardwareCapabilities& capabilities,
                                          SramAllocator& sramAllocator,
                                          TensorShape& outputStripe,
                                          const TensorShape& outputShape)
{
    std::pair<bool, uint32_t> outputAllocateResult;
    AllocationPreference outputAllocationPreference = AllocationPreference::Start;
    outputAllocateResult.first                      = false;

    // Try taking the whole size first, then move until we find something that works.
    const uint32_t maxHeightSplit = DivRoundUp(outputShape[1], capabilities.GetBrickGroupShape()[1]);
    const uint32_t maxWidthSplits = DivRoundUp(outputShape[2], capabilities.GetBrickGroupShape()[2]);
    // Allow splitting in depth only if the width is 1. When the width is 1 the firmware can support splitting in depth,
    // but for other cases it can't (this isn't strictly true, but is a conservative approximation - what matters
    // here is that we support at least the cases we claim to, which is when width == 1 - see IsTensorDepthSupported).
    const uint32_t maxDepthSplits =
        outputShape[2] == 1 ? DivRoundUp(outputShape[3], capabilities.GetBrickGroupShape()[3]) : 1;

    for (uint32_t numDepthSplits = 1; numDepthSplits <= maxDepthSplits && !outputAllocateResult.first; ++numDepthSplits)
    {
        for (uint32_t numWidthSplits = 1; numWidthSplits <= maxWidthSplits && !outputAllocateResult.first;
             ++numWidthSplits)
        {
            for (uint32_t numHeightSplits = 1; numHeightSplits <= maxHeightSplit && !outputAllocateResult.first;
                 ++numHeightSplits)
            {
                const uint32_t outputStripeHeight = outputShape[1] / numHeightSplits;
                const uint32_t outputStripeWidth  = outputShape[2] / numWidthSplits;
                const uint32_t outputStripeDepth  = outputShape[3] / numDepthSplits;

                outputStripe = {
                    1, utils::RoundUpToNearestMultiple(outputStripeHeight, capabilities.GetBrickGroupShape()[1]),
                    utils::RoundUpToNearestMultiple(outputStripeWidth, capabilities.GetBrickGroupShape()[2]),
                    utils::RoundUpToNearestMultiple(outputStripeDepth, capabilities.GetBrickGroupShape()[3])
                };
                const uint32_t output = TotalSizeBytesNHWCB(outputStripe);

                outputAllocateResult = sramAllocator.Allocate(output / capabilities.GetNumberOfSrams(),
                                                              outputAllocationPreference, "outputs attempt");
            }
        }
    }

    return outputAllocateResult.first;
}

std::unique_ptr<ethosn::support_library::ConversionPass> ConversionPass::CreateGreedily(
    const HardwareCapabilities& capabilities, size_t id, Node* firstNode, SramAllocator& sramAllocator)
{
    Node* current = firstNode;
    std::vector<Node*> definiteNodes;
    std::vector<Node*> potentialNodes;

    if (firstNode->GetInputs().empty())
    {
        return std::unique_ptr<ConversionPass>();    // InputNode
    }
    if (!capabilities.GetIsNchwSupported() && (firstNode->GetInputFormat(0) == CompilerDataFormat::NCHW))
    {
        return std::unique_ptr<ConversionPass>();    // Support NCHW conversion based on hardware capability
    }

    // If our input is in DRAM then we can support any linear sequence of Conversion nodes (i.e. convert from NHWCB to NHWC or vice versa).
    bool isInputDram = (firstNode->GetInputLocation(0) == BufferLocation::Dram);
    if (isInputDram && firstNode->GetInputCompressed(0) &&
        (firstNode->GetInputCompressedFormat(0) == CompilerDataCompressedFormat::FCAF_DEEP ||
         firstNode->GetInputCompressedFormat(0) == CompilerDataCompressedFormat::FCAF_WIDE))
    {
        // Firmware doesn't support loading FCAF formats from Dram in for OPERATION_CONVERT.
        return std::unique_ptr<ConversionPass>();
    }
    // If our input is in SRAM then we can also support NHWC reinterprets (i.e. reshapes) as long as the sequence ends in NHWCB
    bool isInputSram = (firstNode->GetInputLocation(0) == BufferLocation::Sram);

    while (current != nullptr)
    {
        if (isInputDram && dynamic_cast<FormatConversionNode*>(current))
        {
            definiteNodes.push_back(current);
        }
        else if (isInputDram && dynamic_cast<CopyNode*>(current))
        {
            definiteNodes.push_back(current);
        }
        else if (isInputSram)
        {
            if ((dynamic_cast<FormatConversionNode*>(current) ||
                 (dynamic_cast<ReinterpretNode*>(current) && current->GetInputFormat(0) == CompilerDataFormat::NHWC &&
                  current->GetFormat() == CompilerDataFormat::NHWC)) &&
                current->GetLocationHint() != LocationHint::RequireDram)
            {
                potentialNodes.push_back(current);
            }
            else
            {
                break;
            }

            if (current->GetFormat() == CompilerDataFormat::NHWCB)
            {
                std::copy(potentialNodes.begin(), potentialNodes.end(), std::back_inserter(definiteNodes));
                potentialNodes.clear();
            }
        }
        else
        {
            break;
        }

        current = GetNextLinearNodeForInclusionInPass<Node>(current);
    }

    if (!definiteNodes.empty())
    {
        // Allocate some SRAM for the output.

        TensorShape stripeShape;
        AllocationPreference outputSramAllocationPreference = AllocationPreference::Start;
        if (definiteNodes.front()->GetInputLocation(0) == BufferLocation::Sram)
        {
            // For SRAM -> SRAM conversion we perform the whole operation in one stripe
            stripeShape = definiteNodes.back()->GetShape();
            // If the input is already in SRAM then change our allocation preference to help overlap loading/saving.
            const uint32_t inputSramOffset = definiteNodes.front()->GetInputSramOffset(0);
            outputSramAllocationPreference =
                inputSramOffset <= (capabilities.GetTotalSramSize() / capabilities.GetNumberOfSrams()) / 2
                    ? AllocationPreference::End
                    : AllocationPreference::Start;
        }
        else if (definiteNodes.front()->GetInputLocation(0) == BufferLocation::Dram)
        {
            // For DRAM -> DRAM conversion we use the biggest possible stripe shape in the Y-direction.
            SramAllocator currentAllocator = sramAllocator;
            ChooseAndSetupStripe(capabilities, currentAllocator, stripeShape, definiteNodes.back()->GetShape());

            if (!capabilities.GetIsNchwSupported() && (definiteNodes.back()->GetFormat() == CompilerDataFormat::NCHW))
            {
                return std::unique_ptr<ConversionPass>();    // Support NCHW conversion based on hardware capability
            }

            // Conversion pass involving NCHW only supports strategy 3
            if (((definiteNodes.front()->GetInputFormat(0) == CompilerDataFormat::NCHW) ||
                 (definiteNodes.back()->GetFormat() == CompilerDataFormat::NCHW)) &&
                (stripeShape[1] < definiteNodes.front()->GetInputShape(0)[1]))
            {
                return std::unique_ptr<ConversionPass>();
            }

            outputSramAllocationPreference = AllocationPreference::Start;
        }
        else
        {
            assert(!"Unexpected location");
        }

        uint32_t outputSize                            = TotalSizeBytesNHWCB(stripeShape);
        std::pair<bool, uint32_t> outputAllocateResult = sramAllocator.Allocate(
            outputSize / capabilities.GetNumberOfSrams(), outputSramAllocationPreference, "conversion pass output");

        if (!outputAllocateResult.first)
        {
            // We may have been unable to find a strategy because SRAM is full
            // Therefore try find a node in SRAM and force it to DRAM to see if that helps.
            auto NodeInSramPredicate = [](Node* node) { return node->GetLocation() == BufferLocation::Sram; };
            Node* nodeToChange       = SearchDependencies(definiteNodes.front(), NodeInSramPredicate);
            if (nodeToChange != nullptr)
            {
                nodeToChange->SetFixGraphLocationHint(LocationHint::RequireDram);
            }
            return std::unique_ptr<ConversionPass>();
        }
        uint32_t sramOffset = outputAllocateResult.second;

        if (definiteNodes.front()->GetInputLocation(0) == BufferLocation::Dram)
        {
            sramAllocator.Free(outputAllocateResult.second);
        }

        std::unique_ptr<ethosn::support_library::ConversionPass> result =
            std::make_unique<ConversionPass>(capabilities, id, definiteNodes, stripeShape, sramOffset);
        return result;
    }
    else
    {
        return std::unique_ptr<ConversionPass>();
    }
}    // namespace support_library

ConversionPass::ConversionPass(const HardwareCapabilities& capabilities,
                               size_t id,
                               const std::vector<Node*>& nodes,
                               TensorShape stripeShape,
                               uint32_t sramOffset)
    : Pass(capabilities, id)
    , m_StripeShape(stripeShape)
{
    m_Nodes = nodes;
    for (Node* n : m_Nodes)
    {
        n->SetPass(this);
    }
    m_Nodes.back()->SetOutputSramOffset(sramOffset);
    m_Nodes.back()->SetLocation(m_Nodes.front()->GetInputLocation(0));
}

void ConversionPass::Generate(command_stream::CommandStreamBuffer& cmdStream,
                              BufferManager& bufferManager,
                              bool dumpRam)
{
    using namespace command_stream;

    Pass::PreGenerate(cmdStream);

    uint32_t inputBufferId                             = m_Nodes.front()->GetInput(0)->GetSource()->GetBufferId();
    const TensorShape& inputShape                      = m_Nodes.front()->GetInputShape(0);
    CompilerDataFormat inputFormat                     = m_Nodes.front()->GetInputFormat(0);
    BufferLocation inputLocation                       = m_Nodes.front()->GetInputLocation(0);
    const TensorShape& outputShape                     = m_Nodes.back()->GetShape();
    CompilerDataFormat outputFormat                    = m_Nodes.back()->GetFormat();
    BufferLocation outputLocation                      = m_Nodes.back()->GetLocation();
    command_stream::DataFormat commandOutputDataFormat = m_Nodes.back()->GetBufferFormat();
    TensorShape outputSupertensorShape                 = outputShape;
    TensorShape outputSupertensorOffset                = { 0, 0, 0, 0 };

    uint32_t inputSramOffset = 0;

    uint32_t outputBufferId   = 0;
    uint32_t outputSize       = CalculateBufferSize(m_Nodes.back()->GetShape(), commandOutputDataFormat);
    uint32_t outputSramOffset = m_Nodes.back()->GetOutputSramOffset();
    if (outputLocation == BufferLocation::Sram && outputFormat == CompilerDataFormat::NHWCB &&
        inputLocation == BufferLocation::Sram && inputFormat == CompilerDataFormat::NHWCB)
    {
        outputBufferId  = bufferManager.AddSram(outputSize, outputSramOffset);
        inputSramOffset = bufferManager.GetSramOffset(inputBufferId);
    }
    else if (outputLocation == BufferLocation::Dram && inputLocation == BufferLocation::Dram)
    {
        inputSramOffset = outputSramOffset;    // For Dram -> Dram conversion the Sram is shared for inputs & outputs
        ConcatNode* concatNode = FindConcatNode(m_Nodes.back());
        if (concatNode)
        {
            std::pair<TensorShape, TensorShape> superTensorInfo =
                CalculateConcatSupertensorInfo(m_Nodes.back(), concatNode);
            outputSupertensorOffset = superTensorInfo.first;
            outputSupertensorShape  = superTensorInfo.second;

            uint32_t totalSize = CalculateBufferSize(concatNode->GetShape(), concatNode->GetBufferFormat());
            outputBufferId     = concatNode->GetBufferId();
            if (outputBufferId == 0xffffffff)
            {
                outputBufferId = bufferManager.AddDram(BufferType::Intermediate, totalSize);
                concatNode->SetBufferId(outputBufferId);
            }
        }
        else
        {
            outputBufferId = bufferManager.AddDram(BufferType::Intermediate, outputSize);
        }
    }
    else
    {
        assert(!"not supported");
    }
    m_Nodes.back()->SetBufferId(outputBufferId);

    Convert convert;
    convert.m_InputInfo().m_DataType()          = GetCommandDataType(m_Nodes.front()->GetInputDataType(0));
    convert.m_InputInfo().m_DataFormat()        = m_Nodes.front()->GetInputBufferFormat(0);
    convert.m_InputInfo().m_TensorShape()       = inputShape;
    convert.m_InputInfo().m_SupertensorShape()  = inputShape;
    convert.m_InputInfo().m_SupertensorOffset() = { 0, 0, 0, 0 };
    convert.m_InputInfo().m_DramBufferId()      = inputBufferId;
    convert.m_InputInfo().m_ZeroPoint() =
        static_cast<int16_t>(m_Nodes.front()->GetInputQuantizationInfo(0).GetZeroPoint());
    convert.m_InputInfo().m_DataLocation() = GetCommandDataLocation(inputLocation);
    convert.m_InputInfo().m_SramOffset()   = inputSramOffset;
    convert.m_InputInfo().m_StripeShape()  = m_StripeShape;
    convert.m_InputInfo().m_TileSize()     = utils::TotalSizeBytesNHWCB(m_StripeShape);

    convert.m_OutputInfo().m_DataType()          = GetCommandDataType(m_Nodes.back()->GetDataType());
    convert.m_OutputInfo().m_DataFormat()        = commandOutputDataFormat;
    convert.m_OutputInfo().m_TensorShape()       = outputShape;
    convert.m_OutputInfo().m_SupertensorShape()  = outputSupertensorShape;
    convert.m_OutputInfo().m_SupertensorOffset() = outputSupertensorOffset;
    convert.m_OutputInfo().m_DramBufferId()      = outputBufferId;
    convert.m_OutputInfo().m_ZeroPoint() = static_cast<int16_t>(m_Nodes.back()->GetQuantizationInfo().GetZeroPoint());
    convert.m_OutputInfo().m_DataLocation() = GetCommandDataLocation(outputLocation);
    convert.m_OutputInfo().m_SramOffset()   = outputSramOffset;
    convert.m_OutputInfo().m_StripeShape()  = m_StripeShape;
    convert.m_OutputInfo().m_TileSize()     = utils::TotalSizeBytesNHWCB(m_StripeShape);

    cmdStream.EmplaceBack(convert);

    Pass::PostGenerate(cmdStream, dumpRam);
}

PassStats ConversionPass::GetStats(const EstimationOptions& estimationOptions)
{
    PassStats perfData;

    const TensorShape& inputShape           = m_Nodes.front()->GetInputShape(0);
    const TensorShape& roundedUpInputShape  = RoundUpHeightAndWidthToBrickGroup(inputShape);
    const BufferLocation inputLocation      = m_Nodes.front()->GetInputLocation(0);
    const TensorShape& outputShape          = m_Nodes.back()->GetShape();
    const TensorShape& roundedUpOutputShape = RoundUpHeightAndWidthToBrickGroup(outputShape);

    const bool isInputNHWC  = m_Nodes.front()->GetInputBufferFormat(0) == command_stream::DataFormat::NHWC;
    const bool isOutputNHWC = m_Nodes.back()->GetBufferFormat() == command_stream::DataFormat::NHWC;

    const uint32_t inputSize  = inputShape[0] * inputShape[1] * inputShape[2] * inputShape[3];
    const uint32_t outputSize = outputShape[0] * outputShape[1] * outputShape[2] * outputShape[3];

    const uint32_t roundedUpInputSize =
        roundedUpInputShape[0] * roundedUpInputShape[1] * roundedUpInputShape[2] * roundedUpInputShape[3];
    const uint32_t roundedUpOutputSize =
        roundedUpOutputShape[0] * roundedUpOutputShape[1] * roundedUpOutputShape[2] * roundedUpOutputShape[3];

    if (inputLocation != BufferLocation::Sram)
    {
        perfData.m_Input.m_MemoryStats.m_DramNonParallel    = isInputNHWC ? inputSize : roundedUpInputSize;
        perfData.m_Input.m_StripesStats.m_NumCentralStripes = utils::GetNumStripesTotal(inputShape, m_StripeShape);

        perfData.m_Output.m_MemoryStats.m_DramNonParallel    = isOutputNHWC ? outputSize : roundedUpOutputSize;
        perfData.m_Output.m_StripesStats.m_NumCentralStripes = utils::GetNumStripesTotal(outputShape, m_StripeShape);
    }
    else
    {
        perfData.m_Input.m_MemoryStats.m_Sram  = roundedUpInputSize;
        perfData.m_Output.m_MemoryStats.m_Sram = roundedUpOutputSize;
    }

    if (m_Nodes.front()->GetInputCompressed(0))
    {
        perfData.m_Input =
            AccountForActivationCompression(perfData.m_Input, estimationOptions.m_ActivationCompressionSaving);
    }
    if (m_Nodes.back()->GetCompressed())
    {
        perfData.m_Output =
            AccountForActivationCompression(perfData.m_Output, estimationOptions.m_ActivationCompressionSaving);
    }

    return perfData;
}

ethosn::support_library::DotAttributes ConversionPass::GetDotAttributes()
{
    DotAttributes result = Pass::GetDotAttributes();
    result.m_Label       = "ConversionPass\n" + result.m_Label;
    return result;
}

}    // namespace support_library
}    // namespace ethosn
