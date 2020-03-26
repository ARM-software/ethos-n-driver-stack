//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "PlePass.hpp"

#include "Compiler.hpp"
#include "GraphNodes.hpp"
#include "SramAllocator.hpp"
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
    bool strategySelected                    = false;
    bool lastWorkingStrategySelected         = false;
    std::vector<std::pair<bool, uint32_t>> inputsStaticAndOffset;

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
        strategySelected     = false;
        outputLocation       = BufferLocation::None;
        requiredOutputFormat = CompilerDataFormat::NONE;
        if (pleOperation)
        {
            Node* firstNode = nodes.front();
            Node* lastNode  = nodes.back();

            std::vector<TensorShape> inputShapes;
            inputShapes.reserve(firstNode->GetInputs().size());
            for (uint32_t i = 0; i < firstNode->GetInputs().size(); ++i)
            {
                inputShapes.push_back(firstNode->GetInputShape(i));
                inputsStaticAndOffset.push_back(
                    { firstNode->GetInputLocation(i) == BufferLocation::Sram, firstNode->GetInputSramOffset(i) });
            }
            inputSramAllocations.resize(firstNode->GetInputs().size());

            const TensorShape& outputShape = lastNode->GetShape();
            // Reset the SramAllocator used to calculate strategies to the base one originally passed in.

            currentSramAllocator = sramAllocator;
            strategySelected =
                ChooseAndSetupStrategy(capabilities, currentSramAllocator, inputSramAllocations, pleSramAllocation,
                                       outputSramAllocation, inputShapes, outputShape, inputsStaticAndOffset);

            if (strategySelected)
            {
                if (lastNode->GetFormat() == CompilerDataFormat::NHWCB &&
                    lastNode->GetLocationHint() != LocationHint::RequireDram &&
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
            // Once we've found a valid strategy we can set the old SramAllocator to the updated one.
            sramAllocator = currentSramAllocator;
            sramAllocator.Free(pleSramAllocation.offset);
            for (uint32_t i = 0; i < firstNode->GetInputs().size(); ++i)
            {
                if (firstNode->GetInputLocation(i) != BufferLocation::Sram)
                {
                    sramAllocator.Free(inputSramAllocations[i].offset);
                }
            }
            // Set the output sram offset for the final node in the pass. To be used as the input for the next node
            if (lastWorkingOutputLocation == BufferLocation::Dram)
            {
                sramAllocator.Free(outputSramAllocation.offset);
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

bool PlePass::ChooseAndSetupStrategy(const HardwareCapabilities& capabilities,
                                     SramAllocator& sramAllocator,
                                     std::vector<SramTensorAllocation>& inputSramAllocations,
                                     SramTensorAllocation& pleSramAllocation,
                                     SramTensorAllocation& outputSramAllocation,
                                     const std::vector<TensorShape>& inputShapes,
                                     const TensorShape& outputShape,
                                     const std::vector<std::pair<bool, uint32_t>>& inputsStaticAndOffset)
{
    using namespace command_stream;

    assert(inputShapes.size() > 0);

    const TensorShape& inputShape0 = inputShapes[0];

    if (!std::all_of(++inputShapes.begin(), inputShapes.end(),
                     [&inputShape0](const TensorShape& t) { return t == inputShape0; }))
    {
        return false;
    }

    const std::pair<bool, uint32_t> inputsStaticAndOffset0 = inputsStaticAndOffset[0];

    // Check that all the inputs have the same locations (e.g. either Dram or Sram) since
    // control unit cannot handle different locations for ple only operations.
    if (!std::all_of(++inputsStaticAndOffset.begin(), inputsStaticAndOffset.end(),
                     [&inputsStaticAndOffset0](const std::pair<bool, uint32_t> t) {
                         return t.first == inputsStaticAndOffset0.first;
                     }))
    {
        return false;
    }

    if (inputShape0[3] > outputShape[3])
    {
        return false;
    }

    auto pleAllocateResult = sramAllocator.Allocate(capabilities.GetMaxPleSize(), AllocationPreference::Start, "ple");

    if (!pleAllocateResult.first)
    {
        return false;
    }

    pleSramAllocation.tileSize = capabilities.GetMaxPleSize();
    pleSramAllocation.offset   = pleAllocateResult.second;

    const TensorShape inSramShape = {
        1,
        RoundUpToNearestMultiple(inputShape0[1], capabilities.GetBrickGroupShape()[1]),
        RoundUpToNearestMultiple(inputShape0[2], capabilities.GetBrickGroupShape()[2]),
        DivRoundUp(inputShape0[3], capabilities.GetNumberOfSrams()),
    };

    const TensorShape outSramShape = {
        1,
        RoundUpToNearestMultiple(outputShape[1], capabilities.GetBrickGroupShape()[1]),
        RoundUpToNearestMultiple(outputShape[2], capabilities.GetBrickGroupShape()[2]),
        DivRoundUp(outputShape[3], capabilities.GetNumberOfSrams()),
    };

    const uint32_t outDepthMult = outSramShape[3] / inSramShape[3];

    auto tryAlloc = [&](const uint32_t inSramStripeDepth, const uint32_t numStripesInTile) {
        SramAllocator trySramAllocator = sramAllocator;

        const uint32_t inStripeSizeInSram = inSramShape[1] * inSramShape[2] * inSramStripeDepth;

        const TensorShape inStripeShape = {
            1,
            inSramShape[1],
            inSramShape[2],
            inSramStripeDepth * capabilities.GetNumberOfSrams(),
        };

        for (uint32_t i = 0; i < inputShapes.size(); ++i)
        {
            inputSramAllocations[i].stripeShape = inStripeShape;
            inputSramAllocations[i].tileSize = numStripesInTile * inStripeSizeInSram * capabilities.GetNumberOfSrams();

            // Don't allocate inputs if they are already in SRAM
            if (!inputsStaticAndOffset[i].first)
            {
                auto allocateResult = trySramAllocator.Allocate(
                    numStripesInTile * inStripeSizeInSram, AllocationPreference::Start, "input" + std::to_string(i));

                if (!allocateResult.first)
                {
                    return false;
                }

                inputSramAllocations[i].offset = allocateResult.second;
            }
            else if (inStripeShape[3] >= inputShape0[3])
            {
                // A static input must fit entirely in SRAM (multi-stripe not supported)
                inputSramAllocations[i].offset = inputsStaticAndOffset[i].second;
            }
            else
            {
                return false;
            }
        }

        const uint32_t outSramStripeDepth = inSramStripeDepth * outDepthMult;

        const uint32_t outStripeSizeInSram = outSramShape[1] * outSramShape[2] * outSramStripeDepth;

        const TensorShape outStripeShape = {
            1,
            outSramShape[1],
            outSramShape[2],
            outSramStripeDepth * capabilities.GetNumberOfSrams(),
        };

        outputSramAllocation.stripeShape = outStripeShape;
        outputSramAllocation.tileSize    = numStripesInTile * outStripeSizeInSram * capabilities.GetNumberOfSrams();

        auto outputAllocateResult =
            trySramAllocator.Allocate(numStripesInTile * outStripeSizeInSram, AllocationPreference::End, "output");

        if (!outputAllocateResult.first)
        {
            return false;
        }

        outputSramAllocation.offset = outputAllocateResult.second;

        sramAllocator = trySramAllocator;

        return true;
    };

    const bool noInputIsStatic = std::all_of(inputsStaticAndOffset.begin(), inputsStaticAndOffset.end(),
                                             [](const std::pair<bool, uint32_t>& p) { return !p.first; });

    bool success = tryAlloc(inSramShape[3], 1U);

    if (noInputIsStatic)
    {
        const uint32_t sramDepthsInBrick = capabilities.GetBrickGroupShape()[3] / capabilities.GetNumberOfSrams();

        for (uint32_t inSramStripeDepth = RoundUpToNearestMultiple(DivRoundUp(inSramShape[3], 3U), sramDepthsInBrick);
             !success && (inSramStripeDepth != 0); inSramStripeDepth -= sramDepthsInBrick)
        {
            success = tryAlloc(inSramStripeDepth, 2U);
        }

        if (!success && (sramDepthsInBrick > 1U))
        {
            success = tryAlloc(1U, 2U);
        }

        if (!success)
        {
            success = tryAlloc(1U, 1U);
        }
    }

    return success;
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

    pleCmd.m_SramConfig().m_AllocationStrategy() = SramAllocationStrategy::STRATEGY_3;
    pleCmd.m_OutputInfo().m_TileSize()           = m_OutputSramAllocation.tileSize;
    pleCmd.m_OutputInfo().m_StripeShape()        = m_OutputSramAllocation.stripeShape;

    command_stream::DataFormat commandOutputDataFormat = m_Nodes.back()->GetBufferFormat();
    BufferLocation outputBufferLocation                = m_Nodes.back()->GetLocation();

    // Calculate input Buffer id
    uint32_t inputBufferId = m_Nodes.front()->GetInput(0)->GetSource()->GetBufferId();

    // Continue setting up command
    pleCmd.m_NumInputInfos() = static_cast<uint32_t>(m_PleOperation->GetInputs().size());

    pleCmd.m_InputInfo().m_DataType()          = command_stream::DataType::QASYMM8;
    pleCmd.m_InputInfo().m_DataFormat()        = m_PleOperation->GetInputBufferFormat(0);
    pleCmd.m_InputInfo().m_TensorShape()       = inputShape;
    pleCmd.m_InputInfo().m_SupertensorShape()  = inputShape;
    pleCmd.m_InputInfo().m_SupertensorOffset() = { 0, 0, 0, 0 };
    pleCmd.m_InputInfo().m_TileSize()          = m_InputSramAllocations[0].tileSize;
    pleCmd.m_InputInfo().m_StripeShape()       = m_InputSramAllocations[0].stripeShape;
    pleCmd.m_InputInfo().m_DramBufferId()      = inputBufferId;
    pleCmd.m_InputInfo().m_ZeroPoint() = static_cast<uint8_t>(m_PleOperation->GetInputQuantizationInfo(0).m_ZeroPoint);
    pleCmd.m_InputInfo().m_DataLocation() = GetCommandDataLocation(m_PleOperation->GetInputLocation(0));
    pleCmd.m_InputInfo().m_SramOffset()   = m_InputSramAllocations[0].offset;

    if (m_PleOperation->GetInputs().size() == 2)
    {
        const TensorShape& inputShape2              = m_Nodes.front()->GetInputShape(1);
        pleCmd.m_InputInfo2().m_DataType()          = command_stream::DataType::QASYMM8;
        pleCmd.m_InputInfo2().m_DataFormat()        = m_PleOperation->GetInputBufferFormat(1);
        pleCmd.m_InputInfo2().m_StripeShape()       = m_InputSramAllocations[1].stripeShape;
        pleCmd.m_InputInfo2().m_TileSize()          = m_InputSramAllocations[1].tileSize;
        pleCmd.m_InputInfo2().m_TensorShape()       = inputShape2;
        pleCmd.m_InputInfo2().m_SupertensorShape()  = inputShape2;
        pleCmd.m_InputInfo2().m_SupertensorOffset() = { 0, 0, 0, 0 };
        pleCmd.m_InputInfo2().m_DramBufferId()      = m_PleOperation->GetInput(1)->GetSource()->GetBufferId();
        pleCmd.m_InputInfo2().m_ZeroPoint() =
            static_cast<uint8_t>(m_PleOperation->GetInputQuantizationInfo(1).m_ZeroPoint);
        pleCmd.m_InputInfo2().m_DataLocation() = GetCommandDataLocation(m_PleOperation->GetInputLocation(1));
        pleCmd.m_InputInfo2().m_SramOffset()   = m_InputSramAllocations[1].offset;
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

    pleCmd.m_OutputInfo().m_DataType()          = command_stream::DataType::QASYMM8;
    pleCmd.m_OutputInfo().m_DataFormat()        = m_Nodes.back()->GetBufferFormat();
    pleCmd.m_OutputInfo().m_TensorShape()       = outputShape;
    pleCmd.m_OutputInfo().m_SupertensorShape()  = outputShape;
    pleCmd.m_OutputInfo().m_SupertensorOffset() = { 0, 0, 0, 0 };
    pleCmd.m_OutputInfo().m_DramBufferId()      = outputBufferId;
    pleCmd.m_OutputInfo().m_ZeroPoint()    = static_cast<uint8_t>(m_Nodes.back()->GetQuantizationInfo().m_ZeroPoint);
    pleCmd.m_OutputInfo().m_DataLocation() = GetCommandDataLocation(outputBufferLocation);
    pleCmd.m_OutputInfo().m_SramOffset()   = m_OutputSramAllocation.offset;

    pleCmd.m_PleData().m_CeSram()    = m_PleSramAllocation.offset;
    pleCmd.m_PleData().m_PleSram()   = 0x0;
    pleCmd.m_PleData().m_Operation() = GetPleOperation();

    if (m_PleOperation->GetKernelOperation() == command_stream::PleOperation::ADDITION_RESCALE)
    {
        assert(m_PleOperation->GetInputs().size() == 2);

        float outputQuantScale = m_Nodes.back()->GetQuantizationInfo().m_Scale;

        float inputQuantScale = m_PleOperation->GetInputQuantizationInfo(0).m_Scale;
        CalculateRescaleMultiplierAndShift(inputQuantScale / outputQuantScale, pleCmd.m_InputRescaleMultiplier0(),
                                           pleCmd.m_InputRescaleShift0());

        float inputQuantScale1 = m_PleOperation->GetInputQuantizationInfo(1).m_Scale;
        CalculateRescaleMultiplierAndShift(inputQuantScale1 / outputQuantScale, pleCmd.m_InputRescaleMultiplier1(),
                                           pleCmd.m_InputRescaleShift1());
    }
    cmdStream.EmplaceBack(pleCmd);

    Pass::PostGenerate(cmdStream, dumpRam);
}

PassStats PlePass::GetStats(const EstimationOptions& estimationOptions)
{
    PassStats perfData;

    // Number of patches that need to be post processed by the Ple kernel
    uint32_t patchesH = 0;
    uint32_t patchesW = 0;
    uint32_t patchesC = 0;

    InputStats inputStats;

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
            GetInputStats(roundedUpInputShape, inputStripeShape, inputLocation, inputTileSize);

        if (m_Nodes.front()->GetInputCompressed(i))
        {
            inputStats += AccountForActivationCompression(uncompressedInputStats,
                                                          estimationOptions.m_ActivationCompressionSaving);
        }
        else
        {
            inputStats += uncompressedInputStats;
        }

        // Number of patches that need to be post processed by the Ple kernel
        patchesH = std::max(utils::DivRoundUp(inputShape[1], m_Capabilities.GetPatchShape()[1]), patchesH);
        patchesW = std::max(utils::DivRoundUp(inputShape[2], m_Capabilities.GetPatchShape()[2]), patchesW);
        patchesC = std::max(utils::DivRoundUp(inputShape[3], m_Capabilities.GetNumberOfEngines()), patchesC);
    }

    perfData.m_Input = inputStats;

    const TensorShape& outputShape          = m_Nodes.back()->GetShape();
    const TensorShape& roundedUpOutputShape = m_Nodes.back()->GetBufferFormat() != command_stream::DataFormat::NHWC
                                                  ? RoundUpHeightAndWidthToBrickGroup(outputShape)
                                                  : outputShape;
    const BufferLocation outputLocation  = m_Nodes.back()->GetLocation();
    const TensorShape& outputStripeShape = m_OutputSramAllocation.stripeShape;

    // Output data streaming statistics
    OutputStats uncompressedOutputStats = GetOutputStats(roundedUpOutputShape, outputStripeShape, outputLocation);

    if (m_Nodes.back()->GetCompressed())
    {
        perfData.m_Output =
            AccountForActivationCompression(uncompressedOutputStats, estimationOptions.m_ActivationCompressionSaving);
    }
    else
    {
        perfData.m_Output = uncompressedOutputStats;
    }

    // Total number of patches
    perfData.m_Ple.m_NumOfPatches = patchesW * patchesH * patchesC;
    perfData.m_Ple.m_Operation    = static_cast<uint32_t>(GetPleOperation());

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
