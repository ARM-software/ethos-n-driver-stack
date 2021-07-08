//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "PlePass.hpp"

#include "Compiler.hpp"
#include "GraphNodes.hpp"
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

std::unique_ptr<PlePass> PlePass::CreateGreedily(const HardwareCapabilities& capabilities,
                                                 size_t id,
                                                 Node* firstNode,
                                                 SramAllocator& sramAllocator)
{
    // Go through nodes in a linear order
    Node* current = firstNode;
    std::vector<Node*> nodes;
    // Keep track of the last set of nodes which can create a pass.
    // This is to prevent the case where we are able to create a pass then try and add an additional node
    // This then fails to create a pass which fails to prepare all the nodes. It should use the previously sucessful pass.
    std::vector<Node*> workingNodes;

    StandalonePleOperationNode* pleOperation = nullptr;
    FormatConversionNode* postConversion     = nullptr;
    bool lastWorkingStrategySelected         = false;

    CompilerDataFormat requiredOutputFormat  = CompilerDataFormat::NONE;
    BufferLocation outputLocation            = BufferLocation::None;
    BufferLocation lastWorkingOutputLocation = BufferLocation::None;
    std::vector<uint32_t> allocationOffsets;
    SramAllocator currentSramAllocator(sramAllocator);

    std::vector<SramTensorAllocation> inputSramAllocations;
    SramTensorAllocation pleSramAllocation;
    SramTensorAllocation outputSramAllocation;

    while (current != nullptr)
    {
        if (pleOperation == nullptr && dynamic_cast<StandalonePleOperationNode*>(current))
        {
            pleOperation = dynamic_cast<StandalonePleOperationNode*>(current);
            nodes.push_back(current);
        }
        else if (pleOperation != nullptr && postConversion == nullptr &&
                 (requiredOutputFormat == CompilerDataFormat::NONE || current->GetFormat() == requiredOutputFormat) &&
                 dynamic_cast<FormatConversionNode*>(current))
        {
            postConversion = dynamic_cast<FormatConversionNode*>(current);
            nodes.push_back(current);
        }
        else
        {
            break;
        }

        // Analyze the current set of nodes that we have (calculate the strategies etc.), as this will determine whether we want to merge more.
        outputLocation       = BufferLocation::None;
        requiredOutputFormat = CompilerDataFormat::NONE;
        if (pleOperation)
        {
            Node* pleOpFirstNode = nodes.front();
            Node* pleOpLastNode  = nodes.back();

            std::vector<TensorShape> inputShapes;
            inputShapes.reserve(pleOpFirstNode->GetInputs().size());
            std::vector<std::pair<bool, uint32_t>> inputsStaticAndOffset;
            inputsStaticAndOffset.reserve(pleOpFirstNode->GetInputs().size());
            for (uint32_t i = 0; i < pleOpFirstNode->GetInputs().size(); ++i)
            {
                inputShapes.push_back(pleOpFirstNode->GetInputShape(i));
                inputsStaticAndOffset.push_back({ pleOpFirstNode->GetInputLocation(i) == BufferLocation::Sram,
                                                  pleOpFirstNode->GetInputSramOffset(i) });
            }
            inputSramAllocations.resize(pleOpFirstNode->GetInputs().size());

            const TensorShape& outputShape = pleOpLastNode->GetShape();
            // Reset the SramAllocator used to calculate strategies to the base one originally passed in.

            currentSramAllocator       = sramAllocator;
            TensorShape splittableDims = {};
            switch (pleOperation->GetKernelOperation())
            {
                case (command_stream::PleOperation::ADDITION):
                case (command_stream::PleOperation::ADDITION_RESCALE):
                {
                    splittableDims = { 1, 1, 1, 1 };
                    break;
                }
                case (command_stream::PleOperation::AVGPOOL_3X3_1_1_UDMA):
                {
                    splittableDims = { 0, 0, 0, 1 };
                    break;
                }
                default:
                {
                    assert(false);
                    break;
                }
            }

            if (((pleOpFirstNode->GetInputFormat(0) == CompilerDataFormat::NCHW) ||
                 (nodes.back()->GetFormat() == CompilerDataFormat::NCHW)))
            {
                splittableDims = { 0, 0, 0, 0 };
            }

            PleStrategySelectionParameter pleStrategySelectionParameter{
                pleOpLastNode->GetId(), capabilities,  currentSramAllocator,
                inputSramAllocations,   inputShapes,   outputShape,
                inputsStaticAndOffset,  splittableDims
            };
            PleStrategySelectionReturnValue rv = ChooseAndSetupStrategy(pleStrategySelectionParameter);

            if (rv.success)
            {
                inputSramAllocations = rv.inputSramAllocations;
                currentSramAllocator = rv.sramAllocator;
                pleSramAllocation    = rv.pleSramAllocation;
                outputSramAllocation = rv.outputSramAllocation;
                if ((outputSramAllocation.stripeShape[3] < outputShape[3] ||
                     outputSramAllocation.stripeShape[2] < outputShape[2]))
                {
                    // The Firmware does not support outputting NHWC when the OFMs stripes are not contiguous in DRAM.
                    requiredOutputFormat = CompilerDataFormat::NHWCB;
                }

                if (pleOpLastNode->GetFormat() == CompilerDataFormat::NHWCB &&
                    pleOpLastNode->GetLocationHint() != LocationHint::RequireDram &&
                    outputSramAllocation.stripeShape[1] >= outputShape[1] &&
                    outputSramAllocation.stripeShape[2] >= outputShape[2] &&
                    outputSramAllocation.stripeShape[3] >= outputShape[3])
                {
                    // If we can keep the output in SRAM then do so.
                    outputLocation            = BufferLocation::Sram;
                    requiredOutputFormat      = CompilerDataFormat::NHWCB;
                    lastWorkingOutputLocation = outputLocation;
                }
                else
                {
                    outputLocation            = BufferLocation::Dram;
                    lastWorkingOutputLocation = outputLocation;
                }
                lastWorkingStrategySelected = true;
                workingNodes                = nodes;
            }
        }

        current = GetNextLinearNodeForInclusionInPass<Node>(current);
    }

    if (pleOperation)
    {
        if (lastWorkingStrategySelected)
        {
            // Compression format can't be used for the IFM, we need to give a hint to the previous
            // node that its output needs to be uncompressed.
            // Non legacy code does not support it quite yet.
            bool requiredUncompressed    = false;
            bool splitInDepthUnsupported = false;
            for (uint32_t i = 0; i < firstNode->GetInputs().size(); ++i)
            {
                if (firstNode->GetInputCompressed(i))
                {
                    firstNode->GetInput(i)->GetSource()->SetFixGraphCompressionHint(
                        CompressionHint::RequiredUncompressed);
                    requiredUncompressed = true;
                }

                auto inputSramAllocation = inputSramAllocations[i];
                auto inputShape          = firstNode->GetInputShape(i);
                auto source              = firstNode->GetInput(i)->GetSource();
                if (firstNode->GetInputFormat(i) == CompilerDataFormat::NHWC &&
                    (inputSramAllocation.stripeShape[3] < inputShape[3]))
                {
                    // The firmware does not support non contiguous IFM stripes in DRAM for NHWC input.
                    source->SetFixGraphConvertOutputTo(CompilerDataFormat::NHWCB);
                    splitInDepthUnsupported = true;
                }
            }
            if (requiredUncompressed || splitInDepthUnsupported)
            {
                return std::unique_ptr<PlePass>();
            }

            // Once we've found a valid strategy we can set the old SramAllocator to the updated one.
            sramAllocator = currentSramAllocator;
            sramAllocator.Free(nodes.back()->GetId(), pleSramAllocation.offset);
            for (uint32_t i = 0; i < firstNode->GetInputs().size(); ++i)
            {
                if (firstNode->GetInputLocation(i) != BufferLocation::Sram)
                {
                    sramAllocator.Free(nodes.back()->GetId(), inputSramAllocations[i].offset);
                }
            }
            // Set the output sram offset for the final node in the pass. To be used as the input for the next node
            if (lastWorkingOutputLocation == BufferLocation::Dram)
            {
                sramAllocator.Free(nodes.back()->GetId(), outputSramAllocation.offset);
            }
            uint32_t sramOffset = outputSramAllocation.offset;

            std::unique_ptr<PlePass> result =
                std::make_unique<PlePass>(capabilities, id, pleOperation, postConversion, inputSramAllocations,
                                          pleSramAllocation, outputSramAllocation, outputLocation, sramOffset);
            return result;
        }
        else
        {
            // We may have been unable to find a strategy because SRAM is full
            // Therefore try find a node in SRAM and force it to DRAM to see if that helps.
            auto NodeInSramPredicate = [](Node* node) { return node->GetLocation() == BufferLocation::Sram; };
            Node* nodeToChange       = SearchDependencies(firstNode, NodeInSramPredicate);
            if (nodeToChange != nullptr)
            {
                nodeToChange->SetFixGraphLocationHint(LocationHint::RequireDram);
            }
            return std::unique_ptr<PlePass>();
        }
    }
    else
    {
        return std::unique_ptr<PlePass>();
    }
}

PleStrategySelectionReturnValue
    PlePass::ChooseAndSetupStrategy(const PleStrategySelectionParameter& pleStrategySelectionParameter)
{
    using namespace command_stream;

    const std::vector<TensorShape>& inputShapes = pleStrategySelectionParameter.inputShapes;
    assert(inputShapes.size() > 0);
    // This function assumes we have setup the output parameters correctly
    assert(pleStrategySelectionParameter.inputSramAllocations.size() == inputShapes.size());
    const std::vector<std::pair<bool, uint32_t>>& inputsStaticAndOffset =
        pleStrategySelectionParameter.inputsStaticAndOffset;
    assert(inputsStaticAndOffset.size() == inputShapes.size());

    const TensorShape& inputShape0 = inputShapes[0];

    if (!std::all_of(++inputShapes.begin(), inputShapes.end(),
                     [&inputShape0](const TensorShape& t) { return t == inputShape0; }))
    {
        return {};
    }

    const std::pair<bool, uint32_t> inputsStaticAndOffset0 = inputsStaticAndOffset[0];

    // Check that all the inputs have the same locations (e.g. either Dram or Sram) since
    // control unit cannot handle different locations for ple only operations.
    if (!std::all_of(++inputsStaticAndOffset.begin(), inputsStaticAndOffset.end(),
                     [&inputsStaticAndOffset0](const std::pair<bool, uint32_t> t) {
                         return t.first == inputsStaticAndOffset0.first;
                     }))
    {
        return {};
    }

    const TensorShape& outputShape = pleStrategySelectionParameter.outputShape;
    if (inputShape0[3] > outputShape[3])
    {
        return {};
    }

    SramAllocator sramAllocator             = pleStrategySelectionParameter.sramAllocator;
    const HardwareCapabilities capabilities = pleStrategySelectionParameter.capabilities;
    SramAllocator::UserId userId            = pleStrategySelectionParameter.userId;
    auto pleAllocateResult =
        sramAllocator.Allocate(userId, capabilities.GetMaxPleSize(), AllocationPreference::Start, "ple");

    if (!pleAllocateResult.first)
    {
        return {};
    }

    SramTensorAllocation pleSramAllocation;
    pleSramAllocation.tileSize = capabilities.GetMaxPleSize();
    pleSramAllocation.offset   = pleAllocateResult.second;

    // Generate all the stripes we want to try
    using TensorShapeList = std::vector<TensorShape>;
    TensorShapeList outStripes;
    const TensorShape& splittableDims = pleStrategySelectionParameter.splittableDims;
    std::vector<TensorShapeList> inStripes;
    {
        // The stripe depth must be such that no stripes may start on channels that aren't a multiple of 16 and pass
        // through into the next 16, which is not supported by the DMA (e.g. a stripe starting on channel 24
        // and going to channel 48).
        const TensorShape minimumStripeSize =
            TensorShape{ 1, GetHeight(capabilities.GetBrickGroupShape()), GetWidth(capabilities.GetBrickGroupShape()),
                         GetChannels(capabilities.GetBrickGroupShape()) };
        const uint32_t maxHeightSplits = splittableDims[1] ? DivRoundUp(outputShape[1], minimumStripeSize[1]) : 1U;
        const uint32_t maxWidthSplits  = splittableDims[2] ? DivRoundUp(outputShape[2], minimumStripeSize[2]) : 1U;
        const uint32_t maxDepthSplits  = splittableDims[3] ? DivRoundUp(outputShape[3], minimumStripeSize[3]) : 1U;

        for (uint32_t numChannelSplits = 1; numChannelSplits <= maxDepthSplits; ++numChannelSplits)
        {
            for (uint32_t numWidthSplits = 1; numWidthSplits <= maxWidthSplits; ++numWidthSplits)
            {
                for (uint32_t numHeightSplits = 1; numHeightSplits <= maxHeightSplits; ++numHeightSplits)
                {
                    const uint32_t outStripeHeight =
                        RoundUpToNearestMultiple(outputShape[1] / numHeightSplits, minimumStripeSize[1]);
                    const uint32_t outStripeWidth =
                        RoundUpToNearestMultiple(outputShape[2] / numWidthSplits, minimumStripeSize[2]);
                    const uint32_t outStripeChannel =
                        RoundUpToNearestMultiple(outputShape[3] / numChannelSplits, minimumStripeSize[3]);
                    const TensorShape outShape = TensorShape{ 1, outStripeHeight, outStripeWidth, outStripeChannel };

                    TensorShapeList stripesForEachInput;
                    stripesForEachInput.reserve(inputShapes.size());
                    for (uint32_t inputIndex = 0; inputIndex < inputShapes.size(); ++inputIndex)
                    {
                        const auto& inputShape = inputShapes[inputIndex];
                        const uint32_t inStripeHeight =
                            RoundUpToNearestMultiple(inputShape[1] / numHeightSplits, minimumStripeSize[1]);
                        const uint32_t inStripeWidth =
                            RoundUpToNearestMultiple(inputShape[2] / numWidthSplits, minimumStripeSize[2]);
                        const uint32_t inStripeChannel =
                            RoundUpToNearestMultiple(inputShape[3] / numChannelSplits, minimumStripeSize[3]);
                        stripesForEachInput.push_back(TensorShape{ 1, inStripeHeight, inStripeWidth, inStripeChannel });
                    }

                    // Prevent duplicate stripes being added to the list being generated
                    if (outStripes.empty() || outShape != outStripes.back())
                    {
                        outStripes.push_back(std::move(outShape));
                        inStripes.push_back(std::move(stripesForEachInput));
                    }
                }
            }
        }
    }

    auto tryAlloc = [&sramAllocator, &pleStrategySelectionParameter,
                     &pleSramAllocation](const TensorShapeList& inputStripes, const TensorShape& outputStripe,
                                         const uint32_t maxNumStripesInTile) {
        PleStrategySelectionReturnValue rv;
        rv.pleSramAllocation = pleSramAllocation;
        rv.success           = false;

        SramAllocator trySramAllocator = sramAllocator;

        const uint32_t outStripeSizeInSram = GetNumElements(outputStripe);

        // We don't need multiple stripes in the tile if the stripe is already the full tensor
        const TensorShape& outputShape  = pleStrategySelectionParameter.outputShape;
        const uint32_t numStripesInTile = outStripeSizeInSram >= GetNumElements(outputShape) ? 1u : maxNumStripesInTile;
        const HardwareCapabilities& capabilities = pleStrategySelectionParameter.capabilities;
        SramAllocator::UserId userId             = pleStrategySelectionParameter.userId;
        auto outputAllocateResult                = trySramAllocator.Allocate(
            userId, (numStripesInTile * outStripeSizeInSram) / capabilities.GetNumberOfSrams(),
            AllocationPreference::End, "output");

        if (!outputAllocateResult.first)
        {
            return rv;
        }

        const std::vector<std::pair<bool, uint32_t>>& inputsStaticAndOffset =
            pleStrategySelectionParameter.inputsStaticAndOffset;
        const std::vector<TensorShape>& inputShapes             = pleStrategySelectionParameter.inputShapes;
        rv.inputSramAllocations                                 = pleStrategySelectionParameter.inputSramAllocations;
        std::vector<SramTensorAllocation>& inputSramAllocations = rv.inputSramAllocations;
        for (uint32_t inputIndex = 0; inputIndex < inputStripes.size(); ++inputIndex)
        {
            const TensorShape& inputStripe = inputStripes[inputIndex];
            uint32_t inStripeSizeInSram    = GetNumElements(inputStripe);
            // Don't allocate inputs if they are already in SRAM
            if (!inputsStaticAndOffset[inputIndex].first)
            {
                auto allocateResult = trySramAllocator.Allocate(
                    userId, (numStripesInTile * inStripeSizeInSram) / capabilities.GetNumberOfSrams(),
                    AllocationPreference::Start, "input" + std::to_string(inputIndex));

                if (!allocateResult.first)
                {
                    return rv;
                }

                inputSramAllocations[inputIndex].offset = allocateResult.second;
            }
            else if ((GetHeight(inputStripe) >= GetHeight(inputShapes[inputIndex])) &&
                     (GetWidth(inputStripe) >= GetWidth(inputShapes[inputIndex])) &&
                     (GetChannels(inputStripe) >= GetChannels(inputShapes[inputIndex])))
            {
                // A static input must fit entirely in SRAM (multi-stripe not supported)
                inputSramAllocations[inputIndex].offset = inputsStaticAndOffset[inputIndex].second;
            }
            else
            {
                return rv;
            }
            inputSramAllocations[inputIndex].stripeShape = inputStripe;
            inputSramAllocations[inputIndex].tileSize    = numStripesInTile * inStripeSizeInSram;
        }

        SramTensorAllocation& outputSramAllocation = rv.outputSramAllocation;
        outputSramAllocation.stripeShape           = outputStripe;
        outputSramAllocation.tileSize              = numStripesInTile * outStripeSizeInSram;
        outputSramAllocation.offset                = outputAllocateResult.second;
        rv.sramAllocator                           = trySramAllocator;
        rv.success                                 = true;
        return rv;
    };

    for (uint32_t i = 0; i < outStripes.size(); ++i)
    {
        PleStrategySelectionReturnValue rv = tryAlloc(inStripes[i], outStripes[i], 2u);
        // Double buffer all the stripes in each tile.
        if (rv.success)
        {
            return rv;
        }
    }

    return {};
}

PlePass::PlePass(const HardwareCapabilities& capabilities,
                 size_t id,
                 StandalonePleOperationNode* pleOperation,
                 FormatConversionNode* postConversionNode,
                 std::vector<SramTensorAllocation>& inputSramAllocations,
                 SramTensorAllocation& pleSramAllocation,
                 SramTensorAllocation& outputSramAllocation,
                 BufferLocation outputLocation,
                 uint32_t sramOffset)
    : Pass(capabilities, id)
    , m_PleOperation(pleOperation)
    , m_InputSramAllocations(inputSramAllocations)
    , m_PleSramAllocation(pleSramAllocation)
    , m_OutputSramAllocation(outputSramAllocation)
{
    m_Nodes.push_back(pleOperation);
    if (postConversionNode)
    {
        m_Nodes.push_back(postConversionNode);
    }

    for (Node* n : m_Nodes)
    {
        n->SetPass(this);
    }

    m_Nodes.back()->SetLocation(outputLocation);
    m_Nodes.back()->SetOutputSramOffset(sramOffset);
    m_Nodes.back()->SetCompressedFormat(CompilerDataCompressedFormat::NONE);
}

command_stream::PleOperation PlePass::GetPleOperation() const
{
    return m_PleOperation->GetKernelOperation();
}

void PlePass::Generate(command_stream::CommandStreamBuffer& cmdStream, BufferManager& bufferManager, bool dumpRam)
{
    Pass::PreGenerate(cmdStream);

    const TensorShape& inputShape  = m_Nodes.front()->GetInputShape(0);
    const TensorShape& outputShape = m_Nodes.back()->GetShape();

    // Set up command for command stream
    using namespace command_stream;
    PleOnly pleCmd;

    pleCmd.m_SramConfig().m_AllocationStrategy() = SramAllocationStrategy::STRATEGY_X;
    pleCmd.m_OutputInfo().m_TileSize()           = m_OutputSramAllocation.tileSize;
    pleCmd.m_OutputInfo().m_StripeShape()        = m_OutputSramAllocation.stripeShape;

    command_stream::DataFormat commandOutputDataFormat = m_Nodes.back()->GetBufferFormat();
    BufferLocation outputBufferLocation                = m_Nodes.back()->GetLocation();

    // Calculate input Buffer id
    uint32_t inputBufferId = m_Nodes.front()->GetInput(0)->GetSource()->GetBufferId();

    // Continue setting up command
    pleCmd.m_NumInputInfos() = static_cast<uint32_t>(m_PleOperation->GetInputs().size());

    pleCmd.m_InputInfo().m_DataType()          = GetCommandDataType(m_PleOperation->GetInputDataType(0));
    pleCmd.m_InputInfo().m_DataFormat()        = m_PleOperation->GetInputBufferFormat(0);
    pleCmd.m_InputInfo().m_TensorShape()       = inputShape;
    pleCmd.m_InputInfo().m_SupertensorShape()  = inputShape;
    pleCmd.m_InputInfo().m_SupertensorOffset() = { 0, 0, 0, 0 };
    pleCmd.m_InputInfo().m_TileSize()          = m_InputSramAllocations[0].tileSize;
    pleCmd.m_InputInfo().m_StripeShape()       = m_InputSramAllocations[0].stripeShape;
    pleCmd.m_InputInfo().m_DramBufferId()      = inputBufferId;
    pleCmd.m_InputInfo().m_ZeroPoint() =
        static_cast<int16_t>(m_PleOperation->GetInputQuantizationInfo(0).GetZeroPoint());
    pleCmd.m_InputInfo().m_DataLocation() = GetCommandDataLocation(m_PleOperation->GetInputLocation(0));
    pleCmd.m_InputInfo().m_SramOffset()   = m_InputSramAllocations[0].offset;

    // If the tensor is in Sram it needs to use Legacy code.
    if (pleCmd.m_InputInfo().m_DataLocation() != command_stream::DataLocation::DRAM)
    {
        pleCmd.m_SramConfig().m_AllocationStrategy() = SramAllocationStrategy::STRATEGY_3;
    }

    if (m_PleOperation->GetInputs().size() == 2)
    {
        const TensorShape& inputShape2              = m_Nodes.front()->GetInputShape(1);
        pleCmd.m_InputInfo2().m_DataType()          = GetCommandDataType(m_PleOperation->GetInputDataType(1));
        pleCmd.m_InputInfo2().m_DataFormat()        = m_PleOperation->GetInputBufferFormat(1);
        pleCmd.m_InputInfo2().m_StripeShape()       = m_InputSramAllocations[1].stripeShape;
        pleCmd.m_InputInfo2().m_TileSize()          = m_InputSramAllocations[1].tileSize;
        pleCmd.m_InputInfo2().m_TensorShape()       = inputShape2;
        pleCmd.m_InputInfo2().m_SupertensorShape()  = inputShape2;
        pleCmd.m_InputInfo2().m_SupertensorOffset() = { 0, 0, 0, 0 };
        pleCmd.m_InputInfo2().m_DramBufferId()      = m_PleOperation->GetInput(1)->GetSource()->GetBufferId();
        pleCmd.m_InputInfo2().m_ZeroPoint() =
            static_cast<int16_t>(m_PleOperation->GetInputQuantizationInfo(1).GetZeroPoint());
        pleCmd.m_InputInfo2().m_DataLocation() = GetCommandDataLocation(m_PleOperation->GetInputLocation(1));
        pleCmd.m_InputInfo2().m_SramOffset()   = m_InputSramAllocations[1].offset;

        // If the tensor is in Sram it needs to use Legacy code.
        if (pleCmd.m_InputInfo2().m_DataLocation() != command_stream::DataLocation::DRAM)
        {
            pleCmd.m_SramConfig().m_AllocationStrategy() = SramAllocationStrategy::STRATEGY_3;
        }
    }

    uint32_t outputBufferId;
    // Output is static in SRAM
    if (outputBufferLocation == BufferLocation::Sram)
    {
        outputBufferId = bufferManager.AddSram(TotalSizeBytesNHWCB(outputShape), m_OutputSramAllocation.offset);
    }
    else    // Output buffer space is required only when output is not static in SRAM
    {
        outputBufferId =
            bufferManager.AddDram(BufferType::Intermediate, CalculateBufferSize(outputShape, commandOutputDataFormat));
    }
    m_Nodes.back()->SetBufferId(outputBufferId);

    pleCmd.m_OutputInfo().m_DataType()          = GetCommandDataType(m_Nodes.back()->GetDataType());
    pleCmd.m_OutputInfo().m_DataFormat()        = m_Nodes.back()->GetBufferFormat();
    pleCmd.m_OutputInfo().m_TensorShape()       = outputShape;
    pleCmd.m_OutputInfo().m_SupertensorShape()  = outputShape;
    pleCmd.m_OutputInfo().m_SupertensorOffset() = { 0, 0, 0, 0 };
    pleCmd.m_OutputInfo().m_DramBufferId()      = outputBufferId;
    pleCmd.m_OutputInfo().m_ZeroPoint()    = static_cast<int16_t>(m_Nodes.back()->GetQuantizationInfo().GetZeroPoint());
    pleCmd.m_OutputInfo().m_DataLocation() = GetCommandDataLocation(outputBufferLocation);
    pleCmd.m_OutputInfo().m_SramOffset()   = m_OutputSramAllocation.offset;

    // If the tensor is in Sram it needs to use Legacy code.
    if (pleCmd.m_OutputInfo().m_DataLocation() != command_stream::DataLocation::DRAM)
    {
        pleCmd.m_SramConfig().m_AllocationStrategy() = SramAllocationStrategy::STRATEGY_3;
    }

    pleCmd.m_PleData().m_CeSram()    = m_PleSramAllocation.offset;
    pleCmd.m_PleData().m_PleSram()   = 0x0;
    pleCmd.m_PleData().m_Operation() = GetPleOperation();

    if (m_PleOperation->GetKernelOperation() == command_stream::PleOperation::ADDITION_RESCALE)
    {
        assert(m_PleOperation->GetInputs().size() == 2);

        float outputQuantScale = m_Nodes.back()->GetQuantizationInfo().GetScale();

        float inputQuantScale = m_PleOperation->GetInputQuantizationInfo(0).GetScale();
        CalculateRescaleMultiplierAndShift(inputQuantScale / outputQuantScale,
                                           pleCmd.m_PleData().m_RescaleMultiplier0(),
                                           pleCmd.m_PleData().m_RescaleShift0());

        float inputQuantScale1 = m_PleOperation->GetInputQuantizationInfo(1).GetScale();
        CalculateRescaleMultiplierAndShift(inputQuantScale1 / outputQuantScale,
                                           pleCmd.m_PleData().m_RescaleMultiplier1(),
                                           pleCmd.m_PleData().m_RescaleShift1());
    }

    cmdStream.EmplaceBack(pleCmd);

    Pass::PostGenerate(cmdStream, dumpRam);
}

PassStats PlePass::GetStats(const EstimationOptions& estimationOptions)
{
    PassStats perfData;

    InputStats inputStats;

    std::vector<TensorShape> inputShapes;

    for (uint32_t i = 0; i < m_Nodes.front()->GetInputs().size(); ++i)
    {
        TensorShape inputShape          = m_Nodes.front()->GetInputShape(i);
        TensorShape roundedUpInputShape = m_Nodes.front()->GetInputBufferFormat(i) != command_stream::DataFormat::NHWC
                                              ? RoundUpHeightAndWidthToBrickGroup(inputShape)
                                              : inputShape;
        TensorShape inputStripeShape = m_InputSramAllocations[i].stripeShape;
        BufferLocation inputLocation = m_Nodes.front()->GetInput(i)->GetSource()->GetLocation();
        uint32_t inputTileSize       = m_InputSramAllocations[i].tileSize;

        // Input data streaming statistics
        InputStats uncompressedInputStats =
            GetInputStats(m_Capabilities, roundedUpInputShape, inputStripeShape,
                          inputLocation == BufferLocation::Dram ? Location::Dram : Location::Sram, inputTileSize);

        if (m_Nodes.front()->GetInputCompressed(i))
        {
            inputStats += AccountForActivationCompression(uncompressedInputStats,
                                                          estimationOptions.m_ActivationCompressionSaving);
        }
        else
        {
            inputStats += uncompressedInputStats;
        }

        inputShapes.push_back(inputShape);
    }

    perfData.m_Input = inputStats;

    const TensorShape& outputShape          = m_Nodes.back()->GetShape();
    const TensorShape& roundedUpOutputShape = m_Nodes.back()->GetBufferFormat() != command_stream::DataFormat::NHWC
                                                  ? RoundUpHeightAndWidthToBrickGroup(outputShape)
                                                  : outputShape;
    const BufferLocation outputLocation  = m_Nodes.back()->GetLocation();
    const TensorShape& outputStripeShape = m_OutputSramAllocation.stripeShape;

    // Output data streaming statistics
    OutputStats uncompressedOutputStats =
        GetOutputStats(roundedUpOutputShape, outputStripeShape,
                       outputLocation == BufferLocation::Dram ? Location::Dram : Location::Sram);

    if (m_Nodes.back()->GetCompressed())
    {
        perfData.m_Output =
            AccountForActivationCompression(uncompressedOutputStats, estimationOptions.m_ActivationCompressionSaving);
    }
    else
    {
        perfData.m_Output = uncompressedOutputStats;
    }

    perfData.m_Ple = GetPleStats(m_Capabilities, inputShapes, GetPleOperation());

    return perfData;
}

DotAttributes PlePass::GetDotAttributes()
{
    DotAttributes result = Pass::GetDotAttributes();
    result.m_Label       = "PlePass\n" + result.m_Label;
    return result;
}

}    // namespace support_library
}    // namespace ethosn
