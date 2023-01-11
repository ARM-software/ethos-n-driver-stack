//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "McePlePass.hpp"

#include "Compiler.hpp"
#include "Strategies.hpp"
#include "StrategyX.hpp"
#include "Utils.hpp"
#include "cascading/EstimationUtils.hpp"
#include "cascading/MceEstimationUtils.hpp"

#include <algorithm>
#include <cmath>
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

namespace
{

bool IsCompressionFormatCompatible(const CompilerDataCompressedFormat& compressionFormat,
                                   const TensorShape& stripeShape,
                                   const Strategy& strategy,
                                   bool forwardEst)
{
    // If SPA "forward-looking" estimate is configured, activation compression for Ethos-N78 will be
    // allowed for arbitrary tensor shapes except for Strategy 7, which are not supported by FCAF.
    bool estimateOverride   = forwardEst && (strategy != Strategy::STRATEGY_7);
    bool tensorCompressible = IsCompressionFormatCompatibleWithStripeShapeLegacy(compressionFormat, stripeShape);

    switch (compressionFormat)
    {
        case CompilerDataCompressedFormat::FCAF_DEEP:
            // The stripe shape must be a multiple of the cells height (8), width (8) and depth (32)
            return (tensorCompressible || estimateOverride);
        case CompilerDataCompressedFormat::FCAF_WIDE:
            // The stripe shape must be a multiple of the cells height (8), width (16) and depth (16)
            return (tensorCompressible || estimateOverride);
        default:
            return false;
    }
}

bool IsNodeCompressible(const Node& node)
{
    const CompressionHint nodeCompressionHint = node.GetCompressionHint();
    const CompilerDataFormat nodeFormat       = node.GetFormat();

    const bool hintIsOk             = nodeCompressionHint != CompressionHint::RequiredUncompressed;
    const bool isFormatCompressible = nodeFormat == CompilerDataFormat::NHWCB;

    return hintIsOk && isFormatCompressible;
}

CompilerDataCompressedFormat GetIntermediateOutputCompressedFormat(bool enableIntermediateCompression,
                                                                   const LinearNodesOutput& linearOutputNodes,
                                                                   bool forwardEst)
{
    const Node& outputNode = *linearOutputNodes.m_WorkingNodes.back();

    // Only attempt to compress if the format is compatible and there is a transfer to the DRAM
    if (!IsNodeCompressible(outputNode) || linearOutputNodes.m_OutputLocation != BufferLocation::Dram)
    {
        return CompilerDataCompressedFormat::NONE;
    }

    // Attempt to compress if it was requested
    if (enableIntermediateCompression)
    {
        const Strategy strategy              = linearOutputNodes.m_StrategyConfig.strategy;
        const TensorShape& outputStripeShape = linearOutputNodes.m_StrategyConfig.outputAllocation.stripeShape;

        // Attempt to find a compatible compression to use
        if (IsCompressionFormatCompatible(CompilerDataCompressedFormat::FCAF_DEEP, outputStripeShape, strategy,
                                          forwardEst))
        {
            return CompilerDataCompressedFormat::FCAF_DEEP;
        }

        if (IsCompressionFormatCompatible(CompilerDataCompressedFormat::FCAF_WIDE, outputStripeShape, strategy,
                                          forwardEst))
        {
            return CompilerDataCompressedFormat::FCAF_WIDE;
        }
    }

    // Output can't or should not be compressed
    return CompilerDataCompressedFormat::NONE;
}

}    // namespace

std::vector<command_stream::BlockConfig>
    McePlePass::FilterValidBlockConfigs(MceOperationNode* mceOperation,
                                        FuseOnlyPleOperationNode* pleOperation,
                                        const std::vector<command_stream::BlockConfig>& allowedBlockConfigs,
                                        const HardwareCapabilities& capabilities,
                                        CompilerMceAlgorithm algorithm)
{
    using namespace std::placeholders;

    const uint32_t weightsWidth  = mceOperation->GetWeightsInfo().m_Dimensions[1];
    const uint32_t weightsHeight = mceOperation->GetWeightsInfo().m_Dimensions[0];
    const bool isWinograd2d      = (weightsHeight > 1) && (weightsWidth > 1);

    std::vector<command_stream::BlockConfig> res = allowedBlockConfigs;

    // Filter for algorithm
    res = FilterAlgoBlockConfigs(algorithm, isWinograd2d, res, capabilities);

    // Filter for Mce operation
    res = FilterMceBlockConfigs(mceOperation, res);

    // Filter for Ple operation
    res = FilterPleBlockConfigs(pleOperation, res);

    return res;
}

std::vector<IStrategy*> McePlePass::GetValidStrategies(MceOperationNode* mceOperation,
                                                       std::vector<IStrategy*> allowedStrategies)
{
    if (mceOperation->GetOperation() == ethosn::command_stream::MceOperation::FULLY_CONNECTED)
    {
        // Strategy X will be used.
        allowedStrategies.clear();
    }
    return allowedStrategies;
}

std::vector<IStrategy*> FilterStrategiesForPle(command_stream::PleOperation operation,
                                               std::vector<IStrategy*> strategies)
{
    // MaxPool 3x3 assumes block traversal will happen in X-Y-Z order.
    // This means we cannot split the tensor in width.
    if (operation == command_stream::PleOperation::MAXPOOL_3X3_2_2_EVEN ||
        operation == command_stream::PleOperation::MAXPOOL_3X3_2_2_ODD)
    {
        auto IsPartialWidthStrategies = [](IStrategy* s) {
            return (dynamic_cast<Strategy4*>(s) || dynamic_cast<Strategy6*>(s));
        };
        strategies.erase(std::remove_if(strategies.begin(), strategies.end(), IsPartialWidthStrategies),
                         strategies.end());
    }

    // TransposeXY doesn't support any strategy that splits tensor in width or height
    if (operation == command_stream::PleOperation::TRANSPOSE_XY)
    {
        auto IsPartialStrategies = [](IStrategy* s) {
            return (dynamic_cast<Strategy0*>(s) || dynamic_cast<Strategy4*>(s) || dynamic_cast<Strategy6*>(s) ||
                    dynamic_cast<Strategy7*>(s));
        };
        strategies.erase(std::remove_if(strategies.begin(), strategies.end(), IsPartialStrategies), strategies.end());
    }
    return strategies;
}

LinearNodesOutput McePlePass::FindLinearWorkingNodes(Node* firstNode,
                                                     const SramAllocator& sramAllocator,
                                                     const HardwareCapabilities& capabilities,
                                                     std::vector<IStrategy*> allowedStrategies,
                                                     std::vector<command_stream::BlockConfig> allowedBlockConfigs,
                                                     bool enableWinograd)
{
    Node* current                              = firstNode;
    ExtractSubtensorNode* extractSubtensorNode = nullptr;
    MceOperationNode* mceOperation             = nullptr;
    FuseOnlyPleOperationNode* fuseOnlyPle      = nullptr;
    bool foundPostConversions                  = false;
    bool foundRequantizes                      = false;
    std::vector<Node*> currentSetOfNodes;
    CompilerDataFormat requiredOutputFormat = CompilerDataFormat::NONE;

    LinearNodesOutput res;
    while (current != nullptr)
    {
        if (mceOperation == nullptr && dynamic_cast<FormatConversionNode*>(current))
        {
            currentSetOfNodes.push_back(current);
        }
        else if (mceOperation == nullptr && extractSubtensorNode == nullptr &&
                 dynamic_cast<ExtractSubtensorNode*>(current))
        {
            extractSubtensorNode = dynamic_cast<ExtractSubtensorNode*>(current);
            currentSetOfNodes.push_back(current);
        }
        // MceOperation if we don't have one already
        else if (mceOperation == nullptr && dynamic_cast<MceOperationNode*>(current))
        {
            mceOperation = dynamic_cast<MceOperationNode*>(current);
            currentSetOfNodes.push_back(current);
        }
        else if (mceOperation != nullptr && fuseOnlyPle == nullptr && !foundPostConversions &&
                 dynamic_cast<McePostProcessOperationNode*>(current) && !foundRequantizes)
        {
            currentSetOfNodes.push_back(current);
        }
        else if (mceOperation != nullptr && fuseOnlyPle == nullptr && !foundPostConversions &&
                 dynamic_cast<FuseOnlyPleOperationNode*>(current))
        {
            fuseOnlyPle = dynamic_cast<FuseOnlyPleOperationNode*>(current);
            currentSetOfNodes.push_back(current);
        }
        else if (mceOperation != nullptr && dynamic_cast<RequantizeNode*>(current))
        {
            // The requantize will be implemented by modifying the requantization performed by the MCE which is before the PLE.
            // Therefore the requantize node must be before the PLE node.
            // However some PLE nodes are agnostic to different quantisation parameters and so we can conceptually reorder them.
            if (fuseOnlyPle != nullptr)
            {
                using namespace command_stream;
                if (fuseOnlyPle->IsAgnosticToRequantisation())
                {
                    foundRequantizes = true;
                    currentSetOfNodes.push_back(current);
                }
            }
            else
            {
                foundRequantizes = true;
                currentSetOfNodes.push_back(current);
            }
        }
        else if (mceOperation != nullptr && dynamic_cast<FormatConversionNode*>(current))
        {
            // Before we blindly include this conversion node, check if it would be a bad idea.
            // If we require a certain output format (as set below) and this conversion would break that, then don't merge it.
            bool shouldMergeConversion =
                requiredOutputFormat == CompilerDataFormat::NONE || current->GetFormat() == requiredOutputFormat;
            if (shouldMergeConversion)
            {
                foundPostConversions = true;
                currentSetOfNodes.push_back(current);
            }
            else
            {
                break;
            }
        }
        else if (mceOperation != nullptr && dynamic_cast<CopyNode*>(current))
        {
            currentSetOfNodes.push_back(current);
        }
        else
        {
            break;
        }

        // Analyze the current set of nodes that we have (calculate the strategies etc.), as this will determine whether we want to merge more.
        MceStrategySelectionReturnValue selectedStrategy;
        selectedStrategy.success = false;
        requiredOutputFormat     = CompilerDataFormat::NONE;
        if (mceOperation)
        {
            Node* firstNode = currentSetOfNodes.front();
            Node* lastNode  = currentSetOfNodes.back();
            std::pair<bool, uint32_t> inputStaticAndOffset;
            inputStaticAndOffset.first  = firstNode->GetInputLocation(0) == BufferLocation::Sram;
            inputStaticAndOffset.second = firstNode->GetInput(0)->GetSource()->GetOutputSramOffset();

            res.m_Algorithm = mceOperation->GetEffectiveAlgorithm(capabilities, enableWinograd);

            TensorShape weightsShape = GetRoundedWeights(mceOperation->GetWeightsInfo().m_Dimensions, res.m_Algorithm);

            uint32_t depthMax = UINT32_MAX;
            if ((fuseOnlyPle != nullptr) &&
                ((fuseOnlyPle->GetKernelOperation() == command_stream::PleOperation::MAXPOOL_3X3_2_2_EVEN) ||
                 (fuseOnlyPle->GetKernelOperation() == command_stream::PleOperation::MAXPOOL_3X3_2_2_ODD)))
            {
                // The stripe depth is limited since the PLE needs to buffer data
                // from the neighbouring stripe.
                if (mceOperation->GetOperation() == ethosn::command_stream::MceOperation::DEPTHWISE_CONVOLUTION)
                {
                    depthMax = capabilities.GetNumberOfSrams();
                }
                else
                {
                    depthMax = capabilities.GetNumberOfOgs();
                }
            }
            auto validStrategies = GetValidStrategies(mceOperation, allowedStrategies);
            if (fuseOnlyPle)
            {
                validStrategies = FilterStrategiesForPle(fuseOnlyPle->GetKernelOperation(), validStrategies);
            }
            auto validBlockConfigs =
                FilterValidBlockConfigs(mceOperation, fuseOnlyPle, allowedBlockConfigs, capabilities, res.m_Algorithm);
            // The shape we pass to strategy selection is the *MCE* input shape.
            // Note this may be different to firstNode->GetShape() if we are taking our input from a supertensor.
            const TensorShape mceInputShape  = mceOperation->GetInputShape(0);
            const TensorShape mceOutputShape = mceOperation->GetShape();

            MceStrategySelectionParameters strategySelectionParameters{
                lastNode->GetId(), capabilities,
                // Reset the SramAllocator used to calculate strategies to the base one originally passed in.
                sramAllocator, mceInputShape, mceOutputShape, lastNode->GetShape(),
                mceOperation->GetWeightsInfo().m_DataFormat, weightsShape, mceOperation->GetShapeMultiplier(),
                (fuseOnlyPle != nullptr ? fuseOnlyPle->GetShapeMultiplier() : g_IdentityShapeMultiplier),
                inputStaticAndOffset, res.m_Algorithm, depthMax
            };
            selectedStrategy = ChooseAndSetupStrategy(strategySelectionParameters, validStrategies, validBlockConfigs);

            if (IsStrategyX(mceOperation->GetOperation(), selectedStrategy.strategyConfig, res.m_Algorithm,
                            validStrategies))
            {
                StrategyXSelectionParameters strategyXSelectionParameters{
                    lastNode->GetId(),
                    mceOperation->GetOperation(),
                    mceOperation->GetUpsampleType(),
                    sramAllocator,
                    mceInputShape,
                    lastNode->GetShape(),
                    mceOperation->GetWeightsInfo().m_DataFormat,
                    weightsShape,
                    std::make_pair(mceOperation->GetPadTop(), mceOperation->GetPadLeft()),
                    validBlockConfigs,
                    capabilities,
                    mceOperation->GetShapeMultiplier(),
                    (fuseOnlyPle != nullptr ? fuseOnlyPle->GetShapeMultiplier() : g_IdentityShapeMultiplier),
                    inputStaticAndOffset,
                    depthMax
                };
                selectedStrategy = TryStrategyX(strategyXSelectionParameters);
            }

            if (selectedStrategy.success)
            {
                const StrategyConfig& selectedStrategySramConfig = selectedStrategy.strategyConfig;
                // The StrategyConfig that we chose may have restrictions on future conversions operations we can merge.
                if ((selectedStrategySramConfig.outputAllocation.stripeShape[3] < lastNode->GetShape()[3] ||
                     selectedStrategySramConfig.outputAllocation.stripeShape[2] < lastNode->GetShape()[2]) &&
                    mceOperation->GetOperation() != ethosn::command_stream::MceOperation::FULLY_CONNECTED)
                {
                    // The Firmware does not support outputting NHWC when the OFMs stripes are not contiguous in DRAM.
                    requiredOutputFormat = CompilerDataFormat::NHWCB;
                }
                else if (mceOperation->GetOperation() == ethosn::command_stream::MceOperation::FULLY_CONNECTED)
                {
                    // The Firmware only supports writing the output of a fully connected operation as NHWC.
                    requiredOutputFormat = CompilerDataFormat::NHWC;
                }

                if (selectedStrategySramConfig.strategy == Strategy::STRATEGY_3 &&
                    lastNode->GetFormat() == CompilerDataFormat::NHWCB &&
                    lastNode->GetLocationHint() != LocationHint::RequireDram)
                {
                    // If we can keep the output in SRAM then do so.
                    requiredOutputFormat = CompilerDataFormat::NHWCB;
                    res.m_OutputLocation = BufferLocation::Sram;
                }
                else
                {
                    res.m_OutputLocation = BufferLocation::Dram;
                }
                res.m_WorkingNodes         = currentSetOfNodes;
                res.m_SramAllocator        = selectedStrategy.sramAllocator;
                res.m_RequiredOutputFormat = requiredOutputFormat;
                res.m_StrategyConfig       = selectedStrategySramConfig;
                res.m_ValidBlockConfigs    = validBlockConfigs;
            }
            res.m_StrategySelected = selectedStrategy.success;
            res.m_MceOperation     = mceOperation;
            res.m_FuseOnlyPle      = fuseOnlyPle;
        }

        current = GetNextLinearNodeForInclusionInPass<Node>(current);
    }
    return res;
}

std::unique_ptr<ethosn::support_library::McePlePass>
    McePlePass::CreateGreedily(const HardwareCapabilities& capabilities,
                               size_t id,
                               std::vector<IStrategy*> allowedStrategies,
                               std::vector<command_stream::BlockConfig> allowedBlockConfigs,
                               bool enableIntermediateCompression,
                               bool enableWinograd,
                               Node* firstNode,
                               SramAllocator& sramAllocator,
                               bool forwardEst)
{
    // Find the largest set of linear nodes which can be formed into a pass
    LinearNodesOutput linearNodes = FindLinearWorkingNodes(firstNode, sramAllocator, capabilities, allowedStrategies,
                                                           allowedBlockConfigs, enableWinograd);

    // If we haven't found an MceOperation we can't do anything
    if (!linearNodes.m_MceOperation)
    {
        return std::unique_ptr<McePlePass>();
    }

    // If the output format of the last working node is not the same as the required format needed,
    // we give a hint that it needs to be converted.
    if (linearNodes.m_RequiredOutputFormat != CompilerDataFormat::NONE &&
        linearNodes.m_WorkingNodes.back()->GetFormat() != linearNodes.m_RequiredOutputFormat)
    {
        linearNodes.m_WorkingNodes.back()->SetFixGraphConvertOutputTo(linearNodes.m_RequiredOutputFormat);
        return std::unique_ptr<McePlePass>();
    }

    // If we can't find a valid block config or a working strategy and we are in winograd
    // we give a hint to set the convolution algorithm to direct mode
    if ((linearNodes.m_ValidBlockConfigs.empty() || !linearNodes.m_StrategySelected) &&
        linearNodes.m_Algorithm == CompilerMceAlgorithm::Winograd)
    {
        linearNodes.m_MceOperation->SetFixGraphAlgorithmHint(AlgorithmHint::RequireDirect);
        return std::unique_ptr<McePlePass>();
    }

    // If deep convolution followed by MaxPool 3x3 and the number of input channels is too large the ifm will
    // be split in width and since the max pool PLE kernel does not support splitting in width the network
    // will fail to compile so we need to insert identity depthwise before the max pool whenever we find this
    // pattern.
    if (!linearNodes.m_StrategySelected && linearNodes.m_FuseOnlyPle &&
        ((linearNodes.m_FuseOnlyPle->GetKernelOperation() == command_stream::PleOperation::MAXPOOL_3X3_2_2_EVEN) ||
         (linearNodes.m_FuseOnlyPle->GetKernelOperation() == command_stream::PleOperation::MAXPOOL_3X3_2_2_ODD)) &&
        dynamic_cast<MceOperationNode*>(linearNodes.m_FuseOnlyPle->GetInput(0)->GetSource()) != nullptr &&
        dynamic_cast<MceOperationNode*>(linearNodes.m_FuseOnlyPle->GetInput(0)->GetSource())->GetOperation() !=
            command_stream::MceOperation::DEPTHWISE_CONVOLUTION)
    {
        linearNodes.m_FuseOnlyPle->SetFixGraphInsertIdentityNodeHint(true);
        return std::unique_ptr<McePlePass>();
    }

    if (!linearNodes.m_StrategySelected)
    {
        // We may have been unable to find a strategy because SRAM is full
        // Therefore try find a node in SRAM and force it to DRAM to see if that helps.
        auto NodeInSramPredicate = [](Node* node) { return node->GetLocation() == BufferLocation::Sram; };
        Node* nodeToChange       = SearchDependencies(linearNodes.m_MceOperation, NodeInSramPredicate);
        if (nodeToChange != nullptr)
        {
            nodeToChange->SetFixGraphLocationHint(LocationHint::RequireDram);
        }

        return std::unique_ptr<McePlePass>();
    }

    // reading/writing in NCHW format, only strategy3 is allowed
    if (((linearNodes.m_WorkingNodes.front()->GetInputFormat(0) == CompilerDataFormat::NCHW) ||
         (linearNodes.m_WorkingNodes.back()->GetFormat() == CompilerDataFormat::NCHW)) &&
        (linearNodes.m_StrategyConfig.strategy != Strategy::STRATEGY_3))
    {
        return std::unique_ptr<McePlePass>();
    }

    if (linearNodes.m_WorkingNodes.front()->GetInputFormat(0) == CompilerDataFormat::NHWC &&
        (linearNodes.m_StrategyConfig.inputAllocation.stripeShape[3] <
             linearNodes.m_WorkingNodes.front()->GetInputShape(0)[3] ||
         (linearNodes.m_StrategyConfig.inputAllocation.stripeShape[1] <
              linearNodes.m_WorkingNodes.front()->GetInputShape(0)[1] &&
          linearNodes.m_StrategyConfig.inputAllocation.stripeShape[2] <
              linearNodes.m_WorkingNodes.front()->GetInputShape(0)[2])))
    {
        // The firmware does not support either boundary stripe loading or non contiguous IFM stripes in DRAM for NHWC input.
        linearNodes.m_WorkingNodes.front()->GetInput(0)->GetSource()->SetFixGraphConvertOutputTo(
            CompilerDataFormat::NHWCB);
        return std::unique_ptr<McePlePass>();
    }
    if (linearNodes.m_WorkingNodes.empty())
    {
        return std::unique_ptr<McePlePass>();
    }

    const Strategy strategy             = linearNodes.m_StrategyConfig.strategy;
    const TensorShape& inputStripeShape = linearNodes.m_StrategyConfig.inputAllocation.stripeShape;
    Node& inputNode                     = *linearNodes.m_WorkingNodes.front();

    // If the compression format can't be used for the IFM, we need to give a hint to the previous
    // node that its output needs to be uncompressed.
    if (inputNode.GetInputCompressed(0) &&
        !IsCompressionFormatCompatible(inputNode.GetInputCompressedFormat(0), inputStripeShape, strategy, forwardEst))
    {
        inputNode.GetInput(0)->GetSource()->SetFixGraphCompressionHint(CompressionHint::RequiredUncompressed);
        return std::unique_ptr<McePlePass>();
    }
    assert(linearNodes.m_OutputLocation != BufferLocation::None);

    const CompilerDataCompressedFormat intermediateOutputCompressedFormat =
        GetIntermediateOutputCompressedFormat(enableIntermediateCompression, linearNodes, forwardEst);

    // Once we've found a valid strategy we can set the old SramAllocator to the updated one.
    sramAllocator = linearNodes.m_SramAllocator;
    // We can deallocate the weights and ple now.
    const Node* lastNode = linearNodes.m_WorkingNodes.back();
    sramAllocator.Free(lastNode->GetId(), linearNodes.m_StrategyConfig.weightsAllocation.offset);
    sramAllocator.Free(lastNode->GetId(), linearNodes.m_StrategyConfig.pleAllocation.offset);
    if (firstNode->GetInputLocation(0) != BufferLocation::Sram)
    {
        sramAllocator.Free(lastNode->GetId(), linearNodes.m_StrategyConfig.inputAllocation.offset);
    }
    // Set the output sram offset for the final node in the pass. To be used as the input for the next node
    if (linearNodes.m_OutputLocation == BufferLocation::Dram)
    {
        sramAllocator.Free(lastNode->GetId(), linearNodes.m_StrategyConfig.outputAllocation.offset);
    }
    uint32_t sramOffset = linearNodes.m_StrategyConfig.outputAllocation.offset;

    std::unique_ptr<ethosn::support_library::McePlePass> result = std::make_unique<McePlePass>(
        capabilities, id, linearNodes.m_WorkingNodes, linearNodes.m_StrategyConfig, linearNodes.m_OutputLocation,
        intermediateOutputCompressedFormat, linearNodes.m_Algorithm, sramOffset);

    return result;
}

McePlePass::McePlePass(const HardwareCapabilities& capabilities,
                       size_t id,
                       std::vector<Node*> nodes,
                       const StrategyConfig& strategyConfig,
                       BufferLocation outputLocation,
                       CompilerDataCompressedFormat intermediateCompressedFormat,
                       CompilerMceAlgorithm algorithm,
                       uint32_t sramOffset)
    : Pass(capabilities, id)
    , m_ExtractSubtensorNode(nullptr)
    , m_MceOperation(nullptr)
    , m_PleOperation(nullptr)
    , m_WeightEncoder(capabilities)
    , m_StrategyConfig(strategyConfig)
{
    m_Nodes = nodes;
    for (auto node : nodes)
    {
        FormatConversionNode* formatConversionNode               = dynamic_cast<FormatConversionNode*>(node);
        ExtractSubtensorNode* extractSubtensorNode               = dynamic_cast<ExtractSubtensorNode*>(node);
        MceOperationNode* mceOperationNode                       = dynamic_cast<MceOperationNode*>(node);
        McePostProcessOperationNode* mcePostProcessOperationNode = dynamic_cast<McePostProcessOperationNode*>(node);
        FuseOnlyPleOperationNode* fuseOnlyPleOperationNode       = dynamic_cast<FuseOnlyPleOperationNode*>(node);
        RequantizeNode* requantizeNode                           = dynamic_cast<RequantizeNode*>(node);
        CopyNode* copyNode                                       = dynamic_cast<CopyNode*>(node);

        node->SetPass(this);
        if (formatConversionNode && m_MceOperation == nullptr)
        {
            m_PreConversionNodes.push_back(formatConversionNode);
        }
        else if (extractSubtensorNode && m_ExtractSubtensorNode == nullptr)
        {
            m_ExtractSubtensorNode = extractSubtensorNode;
        }
        else if (mceOperationNode && m_MceOperation == nullptr)
        {
            m_MceOperation = mceOperationNode;
        }
        else if (mcePostProcessOperationNode)
        {
            m_McePostProcessOperations.push_back(mcePostProcessOperationNode);
        }
        else if (fuseOnlyPleOperationNode)
        {
            m_PleOperation = fuseOnlyPleOperationNode;
        }
        else if (formatConversionNode)
        {
            m_PostConversionNodes.push_back(formatConversionNode);
        }
        else if (requantizeNode)
        {
            m_RequantizeNodes.push_back(requantizeNode);
        }
        else if (copyNode)
        {
            m_CopyNodes.push_back(copyNode);
        }
        else
        {
            ETHOSN_FAIL_MSG("Unexpected node type");
        }
    }

    m_Nodes.back()->SetOutputSramOffset(sramOffset);
    m_Nodes.back()->SetLocation(outputLocation);
    // We can use compression only in the case when:
    // NHWCB tensors in DRAM where the output stripe is the full width and depth.
    m_Nodes.back()->SetCompressedFormat(intermediateCompressedFormat);

    m_MceOperation->SetAlgorithm(algorithm);
}

command_stream::PleOperation McePlePass::GetPleOperation() const
{
    // Get PLE code buffer - passthrough unless we have been fused with a PLE operation
    return m_PleOperation ? m_PleOperation->GetKernelOperation() : command_stream::PleOperation::PASSTHROUGH;
}

MceStrategySelectionReturnValue
    McePlePass::ChooseAndSetupStrategy(const MceStrategySelectionParameters& strategySelectionParameters,
                                       std::vector<IStrategy*> allowedStrategies,
                                       std::vector<command_stream::BlockConfig> allowedBlockConfigs)
{
    // We try the "best" strategies first until we find one which is appropriate
    // This may change in the future when we use a dynamic programming approach
    MceStrategySelectionReturnValue rv;
    rv.success = false;

    for (IStrategy* strategy : allowedStrategies)
    {
        rv = strategy->TrySetupAnyBlockConfig(strategySelectionParameters, allowedBlockConfigs);
        if (rv.success)
        {
            break;
        }
    }

    return rv;
}

ethosn::support_library::DotAttributes McePlePass::GetDotAttributes()
{
    DotAttributes result = Pass::GetDotAttributes();
    result.m_Label       = "McePlePass\n" + result.m_Label;
    switch (m_StrategyConfig.strategy)
    {
        case Strategy::STRATEGY_0:
            result.m_Label += "\nSTRATEGY_0";
            break;
        case Strategy::STRATEGY_1:
            result.m_Label += "\nSTRATEGY_1";
            break;
        case Strategy::STRATEGY_3:
            result.m_Label += "\nSTRATEGY_3";
            break;
        case Strategy::STRATEGY_4:
            result.m_Label += "\nSTRATEGY_4";
            break;
        case Strategy::STRATEGY_6:
            result.m_Label += "\nSTRATEGY_6";
            break;
        case Strategy::STRATEGY_7:
            result.m_Label += "\nSTRATEGY_7";
            break;
        default:
            break;
    }
    return result;
}

std::pair<uint32_t, uint32_t> McePlePass::GetWeightStripeSizeAndDepth()
{
    const TensorInfo& weightsInfo = m_MceOperation->GetWeightsInfo();
    // weight stripe size is needed for weight encoder if weight streaming.
    uint32_t weightStripeSize = m_StrategyConfig.weightsAllocation.stripeShape[2];

    // Encode weights
    uint32_t weightStripeDepth = 0;
    if (weightsInfo.m_DataFormat == DataFormat::HWIO)
    {
        weightStripeDepth = m_StrategyConfig.weightsAllocation.stripeShape[3];
    }
    else if (weightsInfo.m_DataFormat == DataFormat::HWIM)
    {
        weightStripeDepth = m_StrategyConfig.weightsAllocation.stripeShape[2] *
                            m_StrategyConfig.weightsAllocation.stripeShape[3] /
                            (m_MceOperation->GetStride().m_X * m_MceOperation->GetStride().m_Y);
    }
    else
    {
        // Weight tensor must be HWIO or HWIM
        assert(false);
    }
    return { weightStripeSize, weightStripeDepth };
}

void McePlePass::Generate(command_stream::CommandStreamBuffer& cmdStream, BufferManager& bufferManager, bool dumpRam)
{
    Pass::PreGenerate(cmdStream);

    const TensorShape& mceUninterleavedInputShape = m_MceOperation->GetUninterleavedInputShape();
    const TensorShape& mceOutputShape             = m_MceOperation->GetShape();
    const TensorShape& mceInputShape              = m_MceOperation->GetInputShape(0);
    const TensorInfo& weightsInfo                 = m_MceOperation->GetWeightsInfo();

    // Get SRAM output info
    const TensorShape& outputShape = m_Nodes.back()->GetShape();

    const BufferLocation inputLocation = m_Nodes.front()->GetInput(0)->GetSource()->GetLocation();
    BufferLocation outputLocation      = m_Nodes.back()->GetLocation();
    // Set up command for command stream
    using namespace command_stream;
    McePle convCmd;

    // The allocation has been executed in the Translation
    SramAllocationStrategy strategy;
    switch (m_StrategyConfig.strategy)
    {
        case Strategy::STRATEGY_0:
            strategy = SramAllocationStrategy::STRATEGY_0;
            break;
        case Strategy::STRATEGY_1:
            strategy = SramAllocationStrategy::STRATEGY_1;
            break;
        case Strategy::STRATEGY_3:
            strategy = SramAllocationStrategy::STRATEGY_3;
            break;
        case Strategy::STRATEGY_4:
            strategy = SramAllocationStrategy::STRATEGY_4;
            break;
        case Strategy::STRATEGY_6:
            strategy = SramAllocationStrategy::STRATEGY_6;
            break;
        case Strategy::STRATEGY_7:
            strategy = SramAllocationStrategy::STRATEGY_7;
            break;
        case Strategy::STRATEGY_X:
            strategy = SramAllocationStrategy::STRATEGY_X;
            break;
        default:
            // Invalid strategy
            assert(false);
            // Set strategy so we don't get errors with asserts disabled
            strategy = SramAllocationStrategy::STRATEGY_0;
    }

    convCmd.m_SramConfig().m_AllocationStrategy() = strategy;

    // Propagate tile/stripe shapes to command stream structs
    convCmd.m_InputInfo().m_StripeShape()   = m_StrategyConfig.inputAllocation.stripeShape;
    convCmd.m_InputInfo().m_TileSize()      = m_StrategyConfig.inputAllocation.tileSize;
    convCmd.m_OutputInfo().m_StripeShape()  = m_StrategyConfig.outputAllocation.stripeShape;
    convCmd.m_OutputInfo().m_TileSize()     = m_StrategyConfig.outputAllocation.tileSize;
    convCmd.m_WeightInfo().m_StripeShape()  = m_StrategyConfig.weightsAllocation.stripeShape;
    convCmd.m_WeightInfo().m_TileSize()     = m_StrategyConfig.weightsAllocation.tileSize;
    convCmd.m_BlockConfig().m_BlockWidth()  = m_StrategyConfig.blockWidth;
    convCmd.m_BlockConfig().m_BlockHeight() = m_StrategyConfig.blockHeight;

    uint32_t inputBufferId = m_Nodes.front()->GetInput(0)->GetSource()->GetBufferId();

    const QuantizationInfo& quantizationInfo = m_RequantizeNodes.empty()
                                                   ? m_MceOperation->GetQuantizationInfo()
                                                   : m_RequantizeNodes.back()->GetQuantizationInfo();
    // Encode and add weights to memory map and binding table
    uint32_t weightStripeSize;
    uint32_t weightStripeDepth;
    std::tie(weightStripeSize, weightStripeDepth) = GetWeightStripeSizeAndDepth();
    EncodedWeights encodedWeights =
        m_WeightEncoder.Encode(*m_MceOperation, weightStripeDepth, weightStripeSize, quantizationInfo);

    // Check that the weight tile can hold the expected number of stripes
    if (m_StrategyConfig.weightsAllocation.tileSize <
        (encodedWeights.m_MaxSize * m_StrategyConfig.weightsAllocation.numStripesInTile))
    {
        throw InternalErrorException("Weight tile too small for the expected number of stripes");
    }

    std::vector<uint8_t>& compressedWeights = encodedWeights.m_Data;
    uint32_t weightBufferId                 = bufferManager.AddDramConstant(BufferType::ConstantDma, compressedWeights);

    // Add weight metadata to buffer table and command stream
    std::vector<uint8_t> metadataBytes;
    metadataBytes.assign(
        reinterpret_cast<const uint8_t*>(encodedWeights.m_Metadata.data()),
        reinterpret_cast<const uint8_t*>(encodedWeights.m_Metadata.data() + encodedWeights.m_Metadata.size()));

    uint32_t weightMetadataBufferId    = bufferManager.AddDramConstant(BufferType::ConstantControlUnit, metadataBytes);
    convCmd.m_WeightMetadataBufferId() = weightMetadataBufferId;

    convCmd.m_InputInfo().m_DataType()         = GetCommandDataType(m_Nodes.front()->GetInputDataType(0));
    convCmd.m_InputInfo().m_DataFormat()       = m_Nodes.front()->GetInputBufferFormat(0);
    convCmd.m_InputInfo().m_TensorShape()      = mceInputShape;
    convCmd.m_InputInfo().m_SupertensorShape() = m_Nodes.front()->GetInputShape(0);

    TensorShape supertensorOffset =
        m_ExtractSubtensorNode ? m_ExtractSubtensorNode->GetSupertensorOffset() : TensorShape{ 0, 0, 0, 0 };

    convCmd.m_InputInfo().m_SupertensorOffset() = supertensorOffset;
    convCmd.m_InputInfo().m_DramBufferId()      = inputBufferId;
    convCmd.m_InputInfo().m_ZeroPoint() =
        static_cast<uint16_t>(m_Nodes.front()->GetInputQuantizationInfo(0).GetZeroPoint());
    convCmd.m_InputInfo().m_DataLocation() = GetCommandDataLocation(inputLocation);

    convCmd.m_WeightInfo().m_DataType()   = GetCommandDataType(weightsInfo.m_DataType);
    convCmd.m_WeightInfo().m_DataFormat() = command_stream::DataFormat::WEIGHT_STREAM;

    TensorShape weightsShape = weightsInfo.m_Dimensions;
    if (m_MceOperation->GetAlgorithm() == CompilerMceAlgorithm::Winograd)
    {
        // We don't use winograd for depthwise convolution
        assert(weightsInfo.m_DataFormat != DataFormat::HWIM);

        // WINOGRAD: width and height are rounded up to multiple of 3 if it is not equal to 1.
        for (uint8_t dimension = 0; dimension < 2; ++dimension)
        {
            if ((weightsShape[dimension] != 1) && (weightsShape[dimension] % 3 != 0))
            {
                weightsShape[dimension] = utils::RoundUpToNearestMultiple(weightsShape[dimension], 3);
            }
        }
    }
    convCmd.m_WeightInfo().m_TensorShape()       = weightsShape;
    convCmd.m_WeightInfo().m_SupertensorShape()  = weightsShape;
    convCmd.m_WeightInfo().m_SupertensorOffset() = { 0, 0, 0, 0 };
    convCmd.m_WeightInfo().m_DramBufferId()      = weightBufferId;
    convCmd.m_WeightInfo().m_ZeroPoint()         = static_cast<int16_t>(weightsInfo.m_QuantizationInfo.GetZeroPoint());

    convCmd.m_OutputInfo().m_DataType()          = GetCommandDataType(m_Nodes.back()->GetDataType());
    convCmd.m_OutputInfo().m_DataFormat()        = m_Nodes.back()->GetBufferFormat();
    convCmd.m_OutputInfo().m_TensorShape()       = outputShape;
    convCmd.m_OutputInfo().m_SupertensorShape()  = outputShape;
    convCmd.m_OutputInfo().m_SupertensorOffset() = { 0, 0, 0, 0 };
    convCmd.m_OutputInfo().m_ZeroPoint() = static_cast<int16_t>(m_Nodes.back()->GetQuantizationInfo().GetZeroPoint());
    convCmd.m_OutputInfo().m_DataLocation() = GetCommandDataLocation(outputLocation);

    const uint32_t inputSramOffset = inputLocation == BufferLocation::Sram ? bufferManager.GetSramOffset(inputBufferId)
                                                                           : m_StrategyConfig.inputAllocation.offset;
    const uint32_t outputSramOffset = m_StrategyConfig.outputAllocation.offset;
    const uint32_t weightSramOffset = m_StrategyConfig.weightsAllocation.offset;
    const uint32_t pleSramOffset    = m_StrategyConfig.pleAllocation.offset;
    SramOffsets sramOffsets         = { inputSramOffset, outputSramOffset, weightSramOffset, pleSramOffset };

    uint32_t outputBufferId;
    const uint32_t outputSize = CalculateBufferSize(outputShape, m_Nodes.back()->GetBufferFormat());
    if (outputLocation == BufferLocation::Sram)
    {
        outputBufferId = bufferManager.AddSram(outputSize, sramOffsets.outputOffset);
    }
    else    // Output buffer space is required only when output is not static in SRAM
    {
        ConcatNode* concatNode = FindConcatNode(m_Nodes.back());
        if (concatNode)
        {
            std::pair<TensorShape, TensorShape> superTensorInfo =
                CalculateConcatSupertensorInfo(m_Nodes.back(), concatNode);
            convCmd.m_OutputInfo().m_SupertensorOffset() = superTensorInfo.first;
            convCmd.m_OutputInfo().m_SupertensorShape()  = superTensorInfo.second;

            // Allocate a new buffer for the concat result if this is the first input to it that we've prepared,
            // otherwise re-use the existing buffer.
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

    m_Nodes.back()->SetBufferId(outputBufferId);

    convCmd.m_OutputInfo().m_DramBufferId() = outputBufferId;

    // Only strategy X decouples MCE and output (PLE) stripes
    // and its MCE depth = weight stripe depth
    // Note strategy X does not support HWIW.
    assert(weightsInfo.m_DataFormat != DataFormat::HWIM || strategy != SramAllocationStrategy::STRATEGY_X);
    const TensorShape& mceOutputStripe = {
        m_StrategyConfig.inputAllocation.stripeShape[0],
        utils::RoundUpToNearestMultiple(m_StrategyConfig.inputAllocation.stripeShape[1] * mceOutputShape[1] /
                                            mceInputShape[1],
                                        m_Capabilities.GetBrickGroupShape()[1]),
        utils::RoundUpToNearestMultiple(m_StrategyConfig.inputAllocation.stripeShape[2] * mceOutputShape[2] /
                                            mceInputShape[2],
                                        m_Capabilities.GetBrickGroupShape()[2]),
        strategy == SramAllocationStrategy::STRATEGY_X
            ? m_StrategyConfig.weightsAllocation.stripeShape[3]
            : (GetPleOperation() == command_stream::PleOperation::INTERLEAVE_2X2_2_2)
                  ? m_StrategyConfig.outputAllocation.stripeShape[3] / 4
                  : m_StrategyConfig.outputAllocation.stripeShape[3]
    };

    convCmd.m_MceData() = m_MceOperation->GetMceData();

    DataTypeRange activationBounds        = GetRangeOfDataType(m_MceOperation->GetDataType());
    convCmd.m_MceData().m_ActivationMin() = static_cast<int16_t>(activationBounds.min);
    convCmd.m_MceData().m_ActivationMax() = static_cast<int16_t>(activationBounds.max);

    assert(m_MceOperation->GetUpscaleFactor() <= 2);
    convCmd.m_MceData().m_UpsampleType() = m_MceOperation->GetUpsampleType();

    if (convCmd.m_MceData().m_UpsampleType() == UpsampleType::BILINEAR)
    {
        // As only 2x resize is supported, drop mode is only possible for odd output width/height.
        convCmd.m_MceData().m_UpsampleEdgeModeRow() =
            (outputShape[1] & 1) ? UpsampleEdgeMode::DROP : UpsampleEdgeMode::GENERATE;
        convCmd.m_MceData().m_UpsampleEdgeModeCol() =
            (outputShape[2] & 1) ? UpsampleEdgeMode::DROP : UpsampleEdgeMode::GENERATE;
    }
    else
    {
        convCmd.m_MceData().m_UpsampleEdgeModeRow() = UpsampleEdgeMode::GENERATE;
        convCmd.m_MceData().m_UpsampleEdgeModeCol() = UpsampleEdgeMode::GENERATE;
    }

    convCmd.m_MceData().m_UninterleavedInputShape() = mceUninterleavedInputShape;
    convCmd.m_MceData().m_OutputShape()             = mceOutputShape;
    convCmd.m_MceData().m_OutputStripeShape()       = mceOutputStripe;
    convCmd.m_MceData().m_OutputZeroPoint()         = static_cast<int16_t>(quantizationInfo.GetZeroPoint());

    QuantizationInfo preRequantizationInfo = m_MceOperation->GetQuantizationInfo();
    for (const McePostProcessOperationNode* mcePostProcessOperation : m_McePostProcessOperations)
    {
        mcePostProcessOperation->Apply(convCmd.m_MceData());
        preRequantizationInfo = mcePostProcessOperation->GetQuantizationInfo();
    }

    for (auto const& requantizeNodes : m_RequantizeNodes)
    {
        requantizeNodes->Apply(convCmd.m_MceData(), preRequantizationInfo);
    }

    if (GetPleOperation() == command_stream::PleOperation::SIGMOID)
    {
        constexpr double log2e = 1.4426950408889634;

        const int inputZeroPoint = quantizationInfo.GetZeroPoint();
        const double inputScale  = quantizationInfo.GetScale();

        const double rescaleFactor = inputScale * (log2e * 256.);

        // Note that tanh shares the same PLE kernel with sigmoid
        // by applying different scaling factor to input and output
        // The output tensor scaling factor is 1/256 for sigmoid
        // and 1/128 for tanh.
        assert(m_Nodes.back()->GetQuantizationInfo().GetScale() == (1.f / 128) ||
               m_Nodes.back()->GetQuantizationInfo().GetScale() == (1.f / 256));
        const double tanhFactor = (m_Nodes.back()->GetQuantizationInfo().GetScale() == (1.f / 128)) ? 2.0f : 1.0f;

        uint16_t mult;
        uint16_t shift;
        CalculateRescaleMultiplierAndShift(rescaleFactor * tanhFactor, mult, shift);

        int absMax = static_cast<int>(std::ceil(std::ldexp(1., 15U + shift) / mult)) - 1;

        if (absMax == 0)
        {
            absMax = 1;

            mult  = INT16_MAX;
            shift = 0;
        }

        const int lowerBound = std::max<int>(convCmd.m_MceData().m_ActivationMin(), inputZeroPoint - absMax);
        const int upperBound =
            std::max(lowerBound, std::min<int>(convCmd.m_MceData().m_ActivationMax(), inputZeroPoint + absMax));

        convCmd.m_MceData().m_ActivationMin() = static_cast<int16_t>(lowerBound);
        convCmd.m_MceData().m_ActivationMax() = static_cast<int16_t>(upperBound);

        convCmd.m_PleData().m_RescaleMultiplier0() = mult;
        convCmd.m_PleData().m_RescaleShift0()      = shift;
    }
    else if (GetPleOperation() == command_stream::PleOperation::LEAKY_RELU)
    {
        m_PleOperation->SetOperationSpecificData(convCmd);
    }

    convCmd.m_InputInfo().m_SramOffset()  = sramOffsets.inputOffset;
    convCmd.m_OutputInfo().m_SramOffset() = sramOffsets.outputOffset;
    convCmd.m_WeightInfo().m_SramOffset() = sramOffsets.weightOffset;

    convCmd.m_PleData().m_CeSram()    = sramOffsets.pleCodeOffset;
    convCmd.m_PleData().m_PleSram()   = 0x0;
    convCmd.m_PleData().m_Operation() = GetPleOperation();

    cmdStream.EmplaceBack(convCmd);

    Pass::PostGenerate(cmdStream, dumpRam, bufferManager);
}

PassStats McePlePass::GetStats(const EstimationOptions& estimationOptions)
{
    PassStats perfData;

    const TensorShape& inputShape = m_MceOperation->GetInputShape(0);
    const TensorShape& roundedUpInputShape =
        m_Nodes.front()->GetInputBufferFormat(0) != command_stream::DataFormat::NHWC
            ? RoundUpHeightAndWidthToBrickGroup(inputShape)
            : inputShape;
    const TensorShape& inputStripeShape = m_StrategyConfig.inputAllocation.stripeShape;
    const BufferLocation inputLocation  = m_Nodes.front()->GetInput(0)->GetSource()->GetLocation();
    const uint32_t inputTileSize        = m_StrategyConfig.inputAllocation.tileSize;

    const TensorInfo& weightsInfo  = m_MceOperation->GetWeightsInfo();
    const uint32_t weightsTileSize = m_StrategyConfig.weightsAllocation.tileSize;

    const TensorShape& mceOutputShape = m_MceOperation->GetShape();

    const TensorShape& outputShape          = m_Nodes.back()->GetShape();
    const TensorShape& roundedUpOutputShape = m_Nodes.back()->GetBufferFormat() != command_stream::DataFormat::NHWC
                                                  ? RoundUpHeightAndWidthToBrickGroup(outputShape)
                                                  : outputShape;
    const TensorShape& outputStripeShape = m_StrategyConfig.outputAllocation.stripeShape;
    const BufferLocation outputLocation  = m_Nodes.back()->GetLocation();

    // Number of output stripes affects the number of input data reloads for some streaming strategies.
    uint32_t numOutStripeC = utils::DivRoundUp(outputShape[3], outputStripeShape[3]);

    // Input data streaming statistics.
    InputStats uncompressedInput =
        GetInputStatsLegacy(m_Capabilities, roundedUpInputShape, inputStripeShape,
                            inputLocation == BufferLocation::Dram ? Location::Dram : Location::Sram, inputTileSize,
                            weightsInfo, numOutStripeC);

    if (m_Nodes.front()->GetInputCompressed(0))
    {
        perfData.m_Input =
            AccountForActivationCompression(uncompressedInput, estimationOptions.m_ActivationCompressionSaving);
    }
    else
    {
        perfData.m_Input = uncompressedInput;
    }

    // Output data streaming statistics.
    OutputStats uncompressedOutput =
        GetOutputStatsLegacy(roundedUpOutputShape, outputStripeShape,
                             outputLocation == BufferLocation::Dram ? Location::Dram : Location::Sram);

    if (m_Nodes.back()->GetCompressed())
    {
        perfData.m_Output =
            AccountForActivationCompression(uncompressedOutput, estimationOptions.m_ActivationCompressionSaving);
    }
    else
    {
        perfData.m_Output = uncompressedOutput;
    }

    const QuantizationInfo& quantizationInfo = m_RequantizeNodes.empty()
                                                   ? m_MceOperation->GetQuantizationInfo()
                                                   : m_RequantizeNodes.back()->GetQuantizationInfo();

    // Encode weights to know the actual amount of data including headers.
    uint32_t weightStripeSize;
    uint32_t weightStripeDepth;
    std::tie(weightStripeSize, weightStripeDepth) = GetWeightStripeSizeAndDepth();
    EncodedWeights encodedWeights =
        m_WeightEncoder.Encode(*m_MceOperation, weightStripeDepth, weightStripeSize, quantizationInfo);

    perfData.m_Weights =
        GetWeightsStats(m_Capabilities, encodedWeights, weightsInfo, weightsTileSize, inputShape, inputStripeShape);

    perfData.m_Mce = GetMceStats(m_Capabilities, m_MceOperation->GetStride(), m_MceOperation->GetOperation(),
                                 m_MceOperation->GetAlgorithm(), inputShape, mceOutputShape, weightsInfo.m_Dimensions);

    perfData.m_Ple = GetPleStats(m_Capabilities, { mceOutputShape }, GetPleOperation());

    return perfData;
}

}    // namespace support_library
}    // namespace ethosn
