//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "McePlePass.hpp"

#include "Compiler.hpp"
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

CompilerMceAlgorithm ConvAlgorithm(const HardwareCapabilities& caps, uint32_t w, uint32_t h)
{
    uint32_t numMultsDirect;
    uint32_t numMultsWinograd;

    // Only chooses WINOGRAD if it reduces the number of
    // multiplications because it adds some additional overheads
    // See the 2x2 Winograd Support Specification for further details

    // Decompose kernels with width and height > 3 into multiple 3x3, 3x1 or 1x3 sub-kernels.
    const uint32_t wideKernelSize = caps.GetWideKernelSize();
    if (w == 1 || h == 1)
    {
        // 1D convolution kernel dim w x 1 or 1 x h
        // numOfMultiplications = 2 * w or 2 * h                   DIRECT
        //                      = 4 * CEIL(W/3) or 4 * CEIL(H/3)   WINOGRAD
        numMultsDirect   = w * h * caps.GetOutputSizePerWinograd2D() * caps.GetOutputSizePerWinograd1D();
        numMultsWinograd = caps.GetMacsPerWinograd1D() * utils::DivRoundUp(w * h, wideKernelSize);
    }
    else
    {
        // 2D convolution kernel dim w x h
        // numOfMultiplications = 4 * w * h                    DIRECT
        //                      = 16 * CEIL(W/3) * CEIL(H/3)   WINOGRAD
        numMultsDirect = w * h * caps.GetOutputSizePerWinograd2D() * caps.GetOutputSizePerWinograd2D();
        numMultsWinograd =
            caps.GetMacsPerWinograd2D() * utils::DivRoundUp(w, wideKernelSize) * utils::DivRoundUp(h, wideKernelSize);
    }

    if (numMultsWinograd < numMultsDirect)
    {
        return CompilerMceAlgorithm::Winograd;
    }
    else
    {
        return CompilerMceAlgorithm::Direct;
    }
}

bool IsCompressionFormatCompatible(CompilerDataCompressedFormat compressionFormat,
                                   const TensorShape& nodeShape,
                                   const TensorShape& stripeShape,
                                   Strategy strategy,
                                   bool forwardEst)
{
    // FCAF is not supported for strategy 7 and FC
    bool fcafCompStrategy = (strategy != Strategy::STRATEGY_7 && strategy != Strategy::STRATEGY_FC);

    // If SPA "forward-looking" estimate is configured, activation compression for Ethos-N78 will
    // be allowed for strategies 6, 7 and arbitrary tensor shape.
    bool estimateOverride = forwardEst && (strategy != Strategy::STRATEGY_FC);

    switch (compressionFormat)
    {
        case CompilerDataCompressedFormat::NHWCB_COMPRESSED:
            // The stripe must be the full width and depth of the node input/output shape
            return stripeShape[2] >= nodeShape[2] && stripeShape[3] >= nodeShape[3];
        case CompilerDataCompressedFormat::FCAF_DEEP:
            // The stripe shape must be a multiple of the cells height (8), width (8) and depth (32)
            return (fcafCompStrategy &&
                    (((stripeShape[1] % 8) == 0) && ((stripeShape[2] % 8) == 0) && ((stripeShape[3] % 32) == 0))) ||
                   estimateOverride;
        case CompilerDataCompressedFormat::FCAF_WIDE:
            // The stripe shape must be a multiple of the cells height (8), width (16) and depth (16)
            return (fcafCompStrategy &&
                    (((stripeShape[1] % 8) == 0) && ((stripeShape[2] % 16) == 0) && ((stripeShape[3] % 16) == 0))) ||
                   estimateOverride;
        default:
            return false;
    }
}

CompilerDataCompressedFormat GetIntermediateOutputCompressedFormat(const HardwareCapabilities& capabilities,
                                                                   bool enableIntermediateCompression,
                                                                   const LinearNodesOutput& linearOutputNodes,
                                                                   bool forwardEst)
{
    const Node& outputNode                       = *linearOutputNodes.m_WorkingNodes.back();
    const CompilerDataFormat currentOutputFormat = outputNode.GetFormat();

    // Output must be uncompressed
    if (outputNode.GetCompressionHint() == CompressionHint::RequiredUncompressed)
    {
        return CompilerDataCompressedFormat::NONE;
    }

    // Only attempt to compress if the format is compatible and there is a transfer to the DRAM
    if (currentOutputFormat != CompilerDataFormat::NHWCB || linearOutputNodes.m_OutputLocation != BufferLocation::Dram)
    {
        return CompilerDataCompressedFormat::NONE;
    }

    // Attempt to compress if it was requested
    if (enableIntermediateCompression)
    {
        const Strategy strategy              = linearOutputNodes.m_TensorConfig.strategy;
        const TensorShape& outputStripeShape = linearOutputNodes.m_TensorConfig.outputAllocation.stripeShape;
        const TensorShape& outputNodeShape   = outputNode.GetShape();

        // Attempt to find a compatible compression to use
        if (capabilities.GetActivationCompressionVersion() == 0)
        {
            if (IsCompressionFormatCompatible(CompilerDataCompressedFormat::NHWCB_COMPRESSED, outputNodeShape,
                                              outputStripeShape, strategy, forwardEst))
            {
                return CompilerDataCompressedFormat::NHWCB_COMPRESSED;
            }
        }
        else
        {
            if (IsCompressionFormatCompatible(CompilerDataCompressedFormat::FCAF_DEEP, outputNodeShape,
                                              outputStripeShape, strategy, forwardEst))
            {
                return CompilerDataCompressedFormat::FCAF_DEEP;
            }

            if (IsCompressionFormatCompatible(CompilerDataCompressedFormat::FCAF_WIDE, outputNodeShape,
                                              outputStripeShape, strategy, forwardEst))
            {
                return CompilerDataCompressedFormat::FCAF_WIDE;
            }
        }
    }

    // Output can't or should not be compressed
    return CompilerDataCompressedFormat::NONE;
}

}    // namespace

std::vector<command_stream::BlockConfig>
    McePlePass::FilterValidAndSortBlockConfigs(MceOperationNode* mceOperation,
                                               FuseOnlyPleOperationNode* pleOperation,
                                               const std::vector<command_stream::BlockConfig>& allowedBlockConfigs,
                                               const HardwareCapabilities& capabilities,
                                               const TensorShape& outputShape,
                                               CompilerMceAlgorithm algorithm)
{
    using namespace std::placeholders;

    const uint32_t weightsWidth  = mceOperation->GetWeightsInfo().m_Dimensions[1];
    const uint32_t weightsHeight = mceOperation->GetWeightsInfo().m_Dimensions[0];

    std::vector<command_stream::BlockConfig> res = allowedBlockConfigs;

    if (algorithm == CompilerMceAlgorithm::Winograd)
    {
        const bool isWinograd2d = (weightsHeight > 1) && (weightsWidth > 1);

        // The maximum block size depends on if we are performing a 1D or 2D convolution
        // We can do twice the number of outputs elements with 1D compared to 2D
        // See the Block size limitations sections in the 2x2 Winograd Support document for further details

        const uint32_t maxAllowedWxH = capabilities.GetTotalAccumulatorsPerEngine() / (isWinograd2d ? 4U : 2U);

        auto FilterMaxSize = [maxAllowedWxH](const command_stream::BlockConfig& blockConfig) {
            return (blockConfig.m_BlockWidth() * blockConfig.m_BlockHeight()) <= maxAllowedWxH;
        };

        res = Filter(res, FilterMaxSize);

        const auto comp = [&](const command_stream::BlockConfig& blockConfig1,
                              const command_stream::BlockConfig& blockConfig2) {
            const uint32_t blockWidth1  = blockConfig1.m_BlockWidth();
            const uint32_t blockHeight1 = blockConfig1.m_BlockHeight();

            const uint32_t blockWidth2  = blockConfig2.m_BlockWidth();
            const uint32_t blockHeight2 = blockConfig2.m_BlockHeight();

            const bool outputFitsInBlock1 = (outputShape[1] <= blockHeight1) && (outputShape[2] <= blockWidth1);
            const bool outputFitsInBlock2 = (outputShape[1] <= blockHeight2) && (outputShape[2] <= blockWidth2);

            if (outputFitsInBlock1 && outputFitsInBlock2)
            {
                const uint32_t size1 = blockWidth1 * blockHeight1;
                const uint32_t size2 = blockWidth2 * blockHeight2;

                return size1 < size2;
            }
            else if (!outputFitsInBlock1 && !outputFitsInBlock2)
            {
                // We want to maximise the size of the partial blocks at the edge of the ofm XY planes.
                // We maximise the sum of the remainder of the ofm shape divided by the block size.
                //
                // Example on a 17x17 ofm shape:
                //   16x16 blocks: score = 17%16 + 17%16 = 2
                //   32x8  blocks: score = 17%32 + 17%8 = 18.

                const uint32_t remHeight1 = outputShape[1] % blockConfig1.m_BlockHeight();
                const uint32_t remWidth1  = outputShape[2] % blockConfig1.m_BlockWidth();

                const uint32_t remHeight2 = outputShape[1] % blockConfig2.m_BlockHeight();
                const uint32_t remWidth2  = outputShape[2] % blockConfig2.m_BlockWidth();

                const uint32_t rem1 = remHeight1 + remWidth1;
                const uint32_t rem2 = remHeight2 + remWidth2;

                if (rem1 == rem2)
                {
                    // In case of a tie, we favor largest block width if (weightsWidth > weightsHeight)
                    // or largest block height otherwise

                    if (weightsWidth > weightsHeight)
                    {
                        return (blockWidth1 > blockWidth2) ||
                               ((blockWidth1 == blockWidth2) && (blockHeight1 > blockHeight2));
                    }

                    return (blockHeight1 > blockHeight2) ||
                           ((blockHeight1 == blockHeight2) && (blockWidth1 > blockWidth2));
                }

                return rem1 > rem2;
            }
            else
            {
                return outputFitsInBlock1;    // && !outputFitsBlock2
            }
        };

        std::sort(res.begin(), res.end(), comp);
    }

    const auto FilterToSize = [](const command_stream::BlockConfig& blockConfig, uint32_t width, uint32_t height) {
        return blockConfig == command_stream::BlockConfig{ width, height };
    };

    if (mceOperation->GetOperation() == ethosn::command_stream::MceOperation::FULLY_CONNECTED)
    {
        auto FilterTo8x8 = [FilterToSize](const command_stream::BlockConfig& blockConfig) {
            return FilterToSize(blockConfig, 8, 8);
        };
        // Fully Connected wants to force a 8x8 block size. We'll do this here by limiting the block configs.
        res = Filter(res, FilterTo8x8);
    }

    if (pleOperation != nullptr)
    {
        const auto FilterToSizes = [](const command_stream::BlockConfig& blockConfig,
                                      const std::initializer_list<command_stream::BlockConfig> allowedConfigs) {
            return std::find(allowedConfigs.begin(), allowedConfigs.end(), blockConfig) != allowedConfigs.end();
        };

        const command_stream::PleOperation pleOp = pleOperation->GetKernelOperation();

        if (pleOp == command_stream::PleOperation::INTERLEAVE_2X2_2_2)
        {
            auto filter = [FilterToSize](const command_stream::BlockConfig& blockConfig) {
                return FilterToSize(blockConfig, 16, 16);
            };
            res = Filter(res, filter);
        }
        else if (pleOp == command_stream::PleOperation::MAXPOOL_2X2_2_2)
        {
            // MaxPool 2x2 2,2 supports only 16x16, 32x8, 8x8
            auto filter = [&](const auto& blockConfig) {
                return FilterToSizes(blockConfig, { { 16U, 16U }, { 32U, 8U }, { 8U, 8U } });
            };
            res = Filter(res, filter);
        }
        else if ((pleOp == command_stream::PleOperation::MEAN_XY_7X7) ||
                 (pleOp == command_stream::PleOperation::MEAN_XY_8X8))
        {
            auto filter = [FilterToSize](const command_stream::BlockConfig& blockConfig) {
                return FilterToSize(blockConfig, 8, 8);
            };
            res = Filter(res, filter);
        }
        else if (pleOp == command_stream::PleOperation::MAXPOOL_3X3_2_2_EVEN ||
                 pleOp == command_stream::PleOperation::MAXPOOL_3X3_2_2_ODD)
        {
            // The maxpool 3x3_2_2 and avgpool 3x3_1_1 ple kernels only support 8x8, 32x8 blocks
            auto filter = [&](const auto& blockConfig) {
                return FilterToSizes(blockConfig, { { 32U, 8U }, { 8U, 8U } });
            };
            res = Filter(res, filter);
        }
    }

    return res;
}

std::vector<IStrategy*> McePlePass::GetValidStrategies(MceOperationNode* mceOperation,
                                                       std::vector<IStrategy*> allowedStrategies)
{
    if (mceOperation->GetOperation() == ethosn::command_stream::MceOperation::FULLY_CONNECTED)
    {
        // FC specific scheduling strategies will be used.
        allowedStrategies.clear();
        allowedStrategies.push_back(new StrategyFc());
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
        else
        {
            break;
        }

        // Analyze the current set of nodes that we have (calculate the strategies etc.), as this will determine whether we want to merge more.
        bool strategySelected = false;
        requiredOutputFormat  = CompilerDataFormat::NONE;
        if (mceOperation)
        {
            Node* firstNode = currentSetOfNodes.front();
            Node* lastNode  = currentSetOfNodes.back();
            std::pair<bool, uint32_t> inputStaticAndOffset;
            inputStaticAndOffset.first  = firstNode->GetInputLocation(0) == BufferLocation::Sram;
            inputStaticAndOffset.second = firstNode->GetInput(0)->GetSource()->GetOutputSramOffset();
            utils::ShapeMultiplier shapeMultiplier =
                mceOperation->GetShapeMultiplier() *
                (fuseOnlyPle != nullptr ? fuseOnlyPle->GetShapeMultiplier() : g_IdentityShapeMultiplier);

            TensorShape weightsShape = mceOperation->GetWeightsInfo().m_Dimensions;
            if (mceOperation->GetAlgorithmHint() == AlgorithmHint::AllowWinograd && enableWinograd &&
                mceOperation->GetOperation() == command_stream::MceOperation::CONVOLUTION &&
                mceOperation->GetStride() == Stride{ 1, 1 } &&
                // Winograd and upscaling cannot be performed at the same time
                mceOperation->GetUpsampleType() == UpsampleType::OFF)
            {
                res.m_Algorithm = ConvAlgorithm(capabilities, weightsShape[0], weightsShape[1]);
            }
            else
            {
                res.m_Algorithm = CompilerMceAlgorithm::Direct;
            }

            if (res.m_Algorithm == CompilerMceAlgorithm::Winograd ||
                (res.m_Algorithm == CompilerMceAlgorithm::Direct && ((weightsShape[0] > 7) || (weightsShape[1] > 7))))
            {
                // WINOGRAD: width and height are rounded up to multiple of 3
                // if it is not equal to 1
                // This needs to be taken into consideration in selecting
                // memory strategy.
                // DIRECT: wide kernel mode (H or W, both > 7)
                // then both H,W are rounded up to multiple of 3
                // unless H, W = 1
                if (weightsShape[0] != 1)
                {
                    weightsShape[0] = utils::RoundUpToNearestMultiple(weightsShape[0], 3);
                }

                if (weightsShape[1] != 1)
                {
                    weightsShape[1] = utils::RoundUpToNearestMultiple(weightsShape[1], 3);
                }
            }

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
                    depthMax = capabilities.GetNumberOfOfm();
                }
            }
            auto validStrategies = GetValidStrategies(mceOperation, allowedStrategies);
            if (fuseOnlyPle)
            {
                validStrategies = FilterStrategiesForPle(fuseOnlyPle->GetKernelOperation(), validStrategies);
            }
            auto validBlockConfigs = FilterValidAndSortBlockConfigs(
                mceOperation, fuseOnlyPle, allowedBlockConfigs, capabilities, lastNode->GetShape(), res.m_Algorithm);
            TensorConfig tensorConfig;
            // Reset the SramAllocator used to calculate strategies to the base one originally passed in.
            SramAllocator currentSramAllocator = sramAllocator;
            // The shape we pass to strategy selection is the *MCE* input shape.
            // Note this may be different to firstNode->GetShape() if we are taking our input from a supertensor.
            TensorShape mceInputShape = mceOperation->GetInputShape(0);
            strategySelected          = ChooseAndSetupStrategy(
                capabilities, currentSramAllocator, validStrategies, validBlockConfigs, tensorConfig, mceInputShape,
                lastNode->GetShape(), mceOperation->GetWeightsInfo().m_DataFormat, weightsShape, shapeMultiplier,
                inputStaticAndOffset, res.m_Algorithm, depthMax);

            if (IsStrategyX(mceOperation->GetOperation(), mceOperation->GetUpsampleType(), tensorConfig,
                            res.m_Algorithm, validStrategies))
            {
                currentSramAllocator = sramAllocator;
                strategySelected     = TryStrategyX(
                    mceOperation->GetOperation(), tensorConfig, currentSramAllocator, mceInputShape,
                    lastNode->GetShape(), mceOperation->GetWeightsInfo().m_DataFormat, weightsShape,
                    std::make_pair(mceOperation->GetPadTop(), mceOperation->GetPadLeft()), validBlockConfigs,
                    capabilities, mceOperation->GetShapeMultiplier(),
                    (fuseOnlyPle != nullptr ? fuseOnlyPle->GetShapeMultiplier() : g_IdentityShapeMultiplier),
                    inputStaticAndOffset, depthMax);
            }

            if (strategySelected)
            {
                // The TensorConfig that we chose may have restrictions on future conversions operations we can merge.
                if ((tensorConfig.outputAllocation.stripeShape[3] < lastNode->GetShape()[3] ||
                     tensorConfig.outputAllocation.stripeShape[2] < lastNode->GetShape()[2]) &&
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

                if (tensorConfig.strategy == Strategy::STRATEGY_3 &&
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
                res.m_SramAllocator        = currentSramAllocator;
                res.m_RequiredOutputFormat = requiredOutputFormat;
                res.m_TensorConfig         = tensorConfig;
                res.m_ValidBlockConfigs    = validBlockConfigs;
            }
            res.m_StrategySelected = strategySelected;
            res.m_MceOperation     = mceOperation;
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

    // Support NCHW as input or output only if hardware capability supports
    if (!capabilities.GetIsNchwSupported() &&
        ((linearNodes.m_WorkingNodes.front()->GetInputFormat(0) == CompilerDataFormat::NCHW) ||
         (linearNodes.m_WorkingNodes.back()->GetFormat() == CompilerDataFormat::NCHW)))
    {
        return std::unique_ptr<McePlePass>();
    }

    // reading/writing in NCHW format, only strategy3 is allowed
    if (((linearNodes.m_WorkingNodes.front()->GetInputFormat(0) == CompilerDataFormat::NCHW) ||
         (linearNodes.m_WorkingNodes.back()->GetFormat() == CompilerDataFormat::NCHW)) &&
        (linearNodes.m_TensorConfig.strategy != Strategy::STRATEGY_3))
    {
        return std::unique_ptr<McePlePass>();
    }

    if (linearNodes.m_WorkingNodes.front()->GetInputFormat(0) == CompilerDataFormat::NHWC &&
        (linearNodes.m_TensorConfig.inputAllocation.stripeShape[3] <
             linearNodes.m_WorkingNodes.front()->GetInputShape(0)[3] ||
         (linearNodes.m_TensorConfig.inputAllocation.stripeShape[1] <
              linearNodes.m_WorkingNodes.front()->GetInputShape(0)[1] &&
          linearNodes.m_TensorConfig.inputAllocation.stripeShape[2] <
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

    const Strategy strategy             = linearNodes.m_TensorConfig.strategy;
    const TensorShape& inputStripeShape = linearNodes.m_TensorConfig.inputAllocation.stripeShape;
    Node& inputNode                     = *linearNodes.m_WorkingNodes.front();

    // If the compression format can't be used for the IFM, we need to give a hint to the previous
    // node that its output needs to be uncompressed.
    if (inputNode.GetInputCompressed(0) &&
        !IsCompressionFormatCompatible(inputNode.GetInputCompressedFormat(0), inputNode.GetInputShape(0),
                                       inputStripeShape, strategy, forwardEst))
    {
        inputNode.GetInput(0)->GetSource()->SetFixGraphCompressionHint(CompressionHint::RequiredUncompressed);
        return std::unique_ptr<McePlePass>();
    }
    assert(linearNodes.m_OutputLocation != BufferLocation::None);

    const CompilerDataCompressedFormat intermediateOutputCompressedFormat =
        GetIntermediateOutputCompressedFormat(capabilities, enableIntermediateCompression, linearNodes, forwardEst);

    // Once we've found a valid strategy we can set the old SramAllocator to the updated one.
    sramAllocator = linearNodes.m_SramAllocator;
    // We can deallocate the weights and ple now.
    sramAllocator.Free(linearNodes.m_TensorConfig.weightsAllocation.offset);
    sramAllocator.Free(linearNodes.m_TensorConfig.pleAllocation.offset);
    if (firstNode->GetInputLocation(0) != BufferLocation::Sram)
    {
        sramAllocator.Free(linearNodes.m_TensorConfig.inputAllocation.offset);
    }
    // Set the output sram offset for the final node in the pass. To be used as the input for the next node
    if (linearNodes.m_OutputLocation == BufferLocation::Dram)
    {
        sramAllocator.Free(linearNodes.m_TensorConfig.outputAllocation.offset);
    }
    uint32_t sramOffset = linearNodes.m_TensorConfig.outputAllocation.offset;

    std::unique_ptr<ethosn::support_library::McePlePass> result = std::make_unique<McePlePass>(
        capabilities, id, linearNodes.m_WorkingNodes, linearNodes.m_TensorConfig, linearNodes.m_OutputLocation,
        intermediateOutputCompressedFormat, linearNodes.m_Algorithm, sramOffset);

    return result;
}

McePlePass::McePlePass(const HardwareCapabilities& capabilities,
                       size_t id,
                       std::vector<Node*> nodes,
                       const TensorConfig& tensorConfig,
                       BufferLocation outputLocation,
                       CompilerDataCompressedFormat intermediateCompressedFormat,
                       CompilerMceAlgorithm algorithm,
                       uint32_t sramOffset)
    : Pass(capabilities, id)
    , m_ExtractSubtensorNode(nullptr)
    , m_MceOperation(nullptr)
    , m_PleOperation(nullptr)
    , m_WeightEncoder(WeightEncoder::CreateWeightEncoder(capabilities))
    , m_TensorConfig(tensorConfig)
{
    m_Nodes = nodes;
    for (auto node : nodes)
    {
        node->SetPass(this);
        if (dynamic_cast<FormatConversionNode*>(node) && m_MceOperation == nullptr)
        {
            m_PreConversionNodes.push_back(dynamic_cast<FormatConversionNode*>(node));
        }
        else if (dynamic_cast<ExtractSubtensorNode*>(node) && m_ExtractSubtensorNode == nullptr)
        {
            m_ExtractSubtensorNode = dynamic_cast<ExtractSubtensorNode*>(node);
        }
        else if (dynamic_cast<MceOperationNode*>(node) && m_MceOperation == nullptr)
        {
            m_MceOperation = dynamic_cast<MceOperationNode*>(node);
        }
        else if (dynamic_cast<McePostProcessOperationNode*>(node))
        {
            m_McePostProcessOperations.push_back(dynamic_cast<McePostProcessOperationNode*>(node));
        }
        else if (dynamic_cast<FuseOnlyPleOperationNode*>(node))
        {
            m_PleOperation = dynamic_cast<FuseOnlyPleOperationNode*>(node);
        }
        else if (dynamic_cast<FormatConversionNode*>(node))
        {
            m_PostConversionNodes.push_back(dynamic_cast<FormatConversionNode*>(node));
        }
        else if (dynamic_cast<RequantizeNode*>(node))
        {
            m_RequantizeNodes.push_back(dynamic_cast<RequantizeNode*>(node));
        }
        else
        {
            assert(!"Unexpected node type");
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
#if OFM_SCALING_BYPASS_ENABLE
    return command_stream::PleOperation::OFM_SCALING;
#else
    return m_PleOperation ? m_PleOperation->GetKernelOperation() : command_stream::PleOperation::PASSTHROUGH;
#endif
}

bool McePlePass::ChooseAndSetupStrategy(const HardwareCapabilities& capabilities,
                                        SramAllocator& sramAllocator,
                                        std::vector<IStrategy*> allowedStrategies,
                                        std::vector<command_stream::BlockConfig> allowedBlockConfigs,
                                        TensorConfig& tensorConfig,
                                        const TensorShape& inputShape,
                                        const TensorShape& outputShape,
                                        DataFormat weightsFormat,
                                        const TensorShape& weightsShape,
                                        const utils::ShapeMultiplier& shapeMultiplier,
                                        std::pair<bool, uint32_t> inputStaticAndOffset,
                                        CompilerMceAlgorithm algorithm,
                                        const uint32_t depthMax)
{
    // We try the "best" strategies first until we find one which is appropriate
    // This may change in the future when we use a dynamic programming approach
    bool strategySelected = false;

    for (IStrategy* strategy : allowedStrategies)
    {
        for (auto& currBlockConfig : allowedBlockConfigs)
        {
            if (strategy->TrySetup(tensorConfig, sramAllocator, inputShape, outputShape, weightsFormat, weightsShape,
                                   currBlockConfig, capabilities, shapeMultiplier, inputStaticAndOffset, algorithm,
                                   depthMax))
            {
                strategySelected = true;
                break;
            }
        }

        if (strategySelected)
        {
            break;
        }
    }

    return strategySelected;
}

ethosn::support_library::DotAttributes McePlePass::GetDotAttributes()
{
    DotAttributes result = Pass::GetDotAttributes();
    result.m_Label       = "McePlePass\n" + result.m_Label;
    switch (m_TensorConfig.strategy)
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
        case Strategy::STRATEGY_5:
            result.m_Label += "\nSTRATEGY_5";
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
    uint32_t weightStripeSize = m_TensorConfig.weightsAllocation.stripeShape[2];

    // Encode weights
    uint32_t weightStripeDepth;
    if (weightsInfo.m_DataFormat == DataFormat::HWIO)
    {
        weightStripeDepth = m_TensorConfig.weightsAllocation.stripeShape[3];
    }
    else if (weightsInfo.m_DataFormat == DataFormat::HWIM)
    {
        weightStripeDepth = m_TensorConfig.weightsAllocation.stripeShape[2] *
                            m_TensorConfig.weightsAllocation.stripeShape[3] /
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
    switch (m_TensorConfig.strategy)
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
        case Strategy::STRATEGY_5:
            strategy = SramAllocationStrategy::STRATEGY_5;
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
        case Strategy::STRATEGY_FC:
            // Fully connected strategy is still mapped on to
            // command stream's STRATEGY_1. This shouldn't matter
            // because the firmware doesn't check the strategy names
            // but makes decisions based on the stripe and tile sizes.
            strategy = SramAllocationStrategy::STRATEGY_1;
            break;
        default:
            // Invalid strategy
            assert(false);
    }

    convCmd.m_SramConfig().m_AllocationStrategy() = strategy;

    // Propagate tile/stripe shapes to command stream structs
    convCmd.m_InputInfo().m_StripeShape()   = m_TensorConfig.inputAllocation.stripeShape;
    convCmd.m_InputInfo().m_TileSize()      = m_TensorConfig.inputAllocation.tileSize;
    convCmd.m_OutputInfo().m_StripeShape()  = m_TensorConfig.outputAllocation.stripeShape;
    convCmd.m_OutputInfo().m_TileSize()     = m_TensorConfig.outputAllocation.tileSize;
    convCmd.m_WeightInfo().m_StripeShape()  = m_TensorConfig.weightsAllocation.stripeShape;
    convCmd.m_WeightInfo().m_TileSize()     = m_TensorConfig.weightsAllocation.tileSize;
    convCmd.m_BlockConfig().m_BlockWidth()  = m_TensorConfig.blockWidth;
    convCmd.m_BlockConfig().m_BlockHeight() = m_TensorConfig.blockHeight;

    uint32_t inputBufferId = m_Nodes.front()->GetInput(0)->GetSource()->GetBufferId();

    const QuantizationInfo& quantizationInfo = m_RequantizeNodes.empty()
                                                   ? m_MceOperation->GetQuantizationInfo()
                                                   : m_RequantizeNodes.back()->GetQuantizationInfo();
    // Encode and add weights to memory map and binding table
    uint32_t weightStripeSize;
    uint32_t weightStripeDepth;
    std::tie(weightStripeSize, weightStripeDepth) = GetWeightStripeSizeAndDepth();
    EncodedWeights encodedWeights =
        m_WeightEncoder->Encode(*m_MceOperation, weightStripeDepth, weightStripeSize, quantizationInfo);
    std::vector<uint8_t>& compressedWeights = encodedWeights.m_Data;
    uint32_t weightBufferId                 = bufferManager.AddDramConstant(BufferType::ConstantDma, compressedWeights);

    // Add weight metadata to buffer table and command stream
    std::vector<uint8_t> metadataBytes;
    metadataBytes.assign(
        reinterpret_cast<const uint8_t*>(encodedWeights.m_Metadata.data()),
        reinterpret_cast<const uint8_t*>(encodedWeights.m_Metadata.data() + encodedWeights.m_Metadata.size()));

    uint32_t weightMetadataBufferId    = bufferManager.AddDramConstant(BufferType::ConstantControlUnit, metadataBytes);
    convCmd.m_WeightMetadataBufferId() = weightMetadataBufferId;

    convCmd.m_InputInfo().m_DataType()         = ConvertDataType(m_Nodes.front()->GetInputDataType(0));
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

    convCmd.m_OutputInfo().m_DataType()          = ConvertDataType(m_Nodes.back()->GetDataType());
    convCmd.m_OutputInfo().m_DataFormat()        = m_Nodes.back()->GetBufferFormat();
    convCmd.m_OutputInfo().m_TensorShape()       = outputShape;
    convCmd.m_OutputInfo().m_SupertensorShape()  = outputShape;
    convCmd.m_OutputInfo().m_SupertensorOffset() = { 0, 0, 0, 0 };
    convCmd.m_OutputInfo().m_ZeroPoint() = static_cast<int16_t>(m_Nodes.back()->GetQuantizationInfo().GetZeroPoint());
    convCmd.m_OutputInfo().m_DataLocation() = GetCommandDataLocation(outputLocation);

    const uint32_t inputSramOffset = inputLocation == BufferLocation::Sram ? bufferManager.GetSramOffset(inputBufferId)
                                                                           : m_TensorConfig.inputAllocation.offset;
    const uint32_t outputSramOffset = m_TensorConfig.outputAllocation.offset;
    const uint32_t weightSramOffset = m_TensorConfig.weightsAllocation.offset;
    const uint32_t pleSramOffset    = m_TensorConfig.pleAllocation.offset;
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
        m_TensorConfig.inputAllocation.stripeShape[0],
        utils::RoundUpToNearestMultiple(m_TensorConfig.inputAllocation.stripeShape[1] * mceOutputShape[1] /
                                            mceInputShape[1],
                                        m_Capabilities.GetBrickGroupShape()[1]),
        utils::RoundUpToNearestMultiple(m_TensorConfig.inputAllocation.stripeShape[2] * mceOutputShape[2] /
                                            mceInputShape[2],
                                        m_Capabilities.GetBrickGroupShape()[2]),
        strategy == SramAllocationStrategy::STRATEGY_X
            ? m_TensorConfig.weightsAllocation.stripeShape[3]
            : (GetPleOperation() == command_stream::PleOperation::INTERLEAVE_2X2_2_2)
                  ? m_TensorConfig.outputAllocation.stripeShape[3] / 4
                  : m_TensorConfig.outputAllocation.stripeShape[3]
    };

    convCmd.m_MceData() = m_MceOperation->GetMceData();

    DataTypeRange activationBounds        = GetRangeOfDataType(m_MceOperation->GetDataType());
    convCmd.m_MceData().m_ActivationMin() = static_cast<int16_t>(activationBounds.min);
    convCmd.m_MceData().m_ActivationMax() = static_cast<int16_t>(activationBounds.max);

    assert(m_MceOperation->GetUpscaleFactor() <= 2);
    convCmd.m_MceData().m_UpsampleMode()            = m_MceOperation->GetUpsampleType();
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

        uint16_t mult;
        uint16_t shift;
        CalculateRescaleMultiplierAndShift(rescaleFactor, mult, shift);

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

    Pass::PostGenerate(cmdStream, dumpRam);
}

PassStats McePlePass::GetStats(const EstimationOptions& estimationOptions)
{
    PassStats perfData;

    const TensorShape& inputShape = m_MceOperation->GetInputShape(0);
    const TensorShape& roundedUpInputShape =
        m_Nodes.front()->GetInputBufferFormat(0) != command_stream::DataFormat::NHWC
            ? RoundUpHeightAndWidthToBrickGroup(inputShape)
            : inputShape;
    const TensorShape& inputStripeShape = m_TensorConfig.inputAllocation.stripeShape;
    const BufferLocation inputLocation  = m_Nodes.front()->GetInput(0)->GetSource()->GetLocation();
    const uint32_t inputTileSize        = m_TensorConfig.inputAllocation.tileSize;

    const TensorInfo& weightsInfo         = m_MceOperation->GetWeightsInfo();
    const TensorShape& weightsStripeShape = m_TensorConfig.weightsAllocation.stripeShape;
    const uint32_t weightsTileSize        = m_TensorConfig.weightsAllocation.tileSize;

    const TensorShape& mceOutputShape = m_MceOperation->GetShape();

    const TensorShape& outputShape          = m_Nodes.back()->GetShape();
    const TensorShape& roundedUpOutputShape = m_Nodes.back()->GetBufferFormat() != command_stream::DataFormat::NHWC
                                                  ? RoundUpHeightAndWidthToBrickGroup(outputShape)
                                                  : outputShape;
    const TensorShape& outputStripeShape = m_TensorConfig.outputAllocation.stripeShape;
    const BufferLocation outputLocation  = m_Nodes.back()->GetLocation();

    // Number of output stripes affects the number of input data reloads for some streaming strategies.
    uint32_t numOutStripeC = utils::DivRoundUp(outputShape[3], outputStripeShape[3]);

    // Input data streaming statistics.
    InputStats uncompressedInput =
        GetInputStats(m_Capabilities, roundedUpInputShape, inputStripeShape,
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
        GetOutputStats(roundedUpOutputShape, outputStripeShape,
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
        m_WeightEncoder->Encode(*m_MceOperation, weightStripeDepth, weightStripeSize, quantizationInfo);

    perfData.m_Weights = GetWeightsStats(m_Capabilities, encodedWeights, weightsInfo, weightsStripeShape,
                                         weightsTileSize, inputShape, inputStripeShape);

    perfData.m_Mce = GetMceStats(m_Capabilities, m_MceOperation->GetStride(), m_MceOperation->GetOperation(),
                                 m_MceOperation->GetAlgorithm(), inputShape, mceOutputShape, weightsInfo.m_Dimensions);

    perfData.m_Ple = GetPleStats(m_Capabilities, { mceOutputShape }, GetPleOperation());

    return perfData;
}

}    // namespace support_library
}    // namespace ethosn
