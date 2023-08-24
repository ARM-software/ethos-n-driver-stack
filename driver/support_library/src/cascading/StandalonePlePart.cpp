//
// Copyright © 2022-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "StandalonePlePart.hpp"

#include "../Utils.hpp"
#include "PartUtils.hpp"
#include "Plan.hpp"
#include "StripeHelper.hpp"

#include <ethosn_utils/Macros.hpp>

#include <memory>

using namespace ethosn::command_stream;
using namespace ethosn::support_library::utils;

namespace ethosn
{
namespace support_library
{
using namespace impl;

StandalonePlePart::StandalonePlePart(PartId id,
                                     const std::vector<TensorShape>& inputTensorShapes,
                                     const TensorShape& outputTensorShape,
                                     const std::vector<QuantizationInfo>& inputQuantizationInfos,
                                     const QuantizationInfo& outputQuantizationInfo,
                                     PleOperation op,
                                     const EstimationOptions& estOpt,
                                     const CompilationOptions& compOpt,
                                     const HardwareCapabilities& capabilities,
                                     std::set<uint32_t> correspondingOperationIds,
                                     DataType dataType,
                                     std::map<std::string, std::string> selectionStringParams,
                                     std::map<std::string, int> selectionIntParams,
                                     std::map<std::string, int> runtimeParams)
    : BasePart(id, "StandalonePlePart", std::move(correspondingOperationIds), estOpt, compOpt, capabilities)
    , m_InputTensorShapes(inputTensorShapes)
    , m_OutputTensorShape(outputTensorShape)
    , m_InputQuantizationInfos(inputQuantizationInfos)
    , m_OutputQuantizationInfo(outputQuantizationInfo)
    , m_KernelOperation(op)
    , m_DataType(dataType)
    , m_StripeConfig(impl::GetDefaultStripeConfig(compOpt, m_DebugTag.c_str()))
    , m_SelectionStringParams(std::move(selectionStringParams))
    , m_SelectionIntParams(std::move(selectionIntParams))
    , m_RuntimeParams(std::move(runtimeParams))
{
    assert(m_InputQuantizationInfos.size() == m_InputTensorShapes.size());
}

Plans StandalonePlePart::GetPlans(CascadeType cascadeType,
                                  ethosn::command_stream::BlockConfig blockConfig,
                                  const std::vector<Buffer*>& sramBufferInputs,
                                  uint32_t numWeightStripes) const
{
    ETHOSN_UNUSED(numWeightStripes);
    ETHOSN_UNUSED(blockConfig);

    if (cascadeType == CascadeType::Middle || cascadeType == CascadeType::End)
    {
        for (auto p : sramBufferInputs)
        {
            if (p != nullptr && p->m_Location != Location::Sram)
            {
                return {};    // Can't continue section from e.g. PleInputSram
            }
        }
    }

    Plans plans;
    StripeConfig stripeConfig = m_StripeConfig;

    switch (m_KernelOperation)
    {
        case PleOperation::ADDITION:
        case PleOperation::ADDITION_RESCALE:
            // All split are possible as these operations are elementwise
            break;
        case PleOperation::AVGPOOL_3X3_1_1_UDMA:
        {
            // AVGPOOL_3X3_1_1_UDMA: only split in D is allowed.
            // This makes it cascadalbe only if the whole input, output
            // tensors are fit into SRAM (in other words no split)
            stripeConfig.DisableSplitWidth();
            stripeConfig.DisableSplitHeight();

            if (cascadeType != CascadeType::Lonely)
            {
                stripeConfig.DisableSplitInputDepth();
                stripeConfig.DisableSplitOutputDepth();
            }
            if (cascadeType == CascadeType::Middle || cascadeType == CascadeType::End)
            {
                Buffer* prevBuffer = sramBufferInputs[0];

                // A cascadable plan is not possible if the stripe shape of the previous buffer
                // is smaller than the input tensor (in other words a full tensor plan is NOT
                // compatible with its predecessors).
                if (prevBuffer->Sram()->m_StripeShape[1] < m_InputTensorShapes[0][1] ||
                    prevBuffer->Sram()->m_StripeShape[2] < m_InputTensorShapes[0][2] ||
                    prevBuffer->Sram()->m_StripeShape[3] < m_InputTensorShapes[0][3])
                {
                    return Plans{};
                }
            }
            break;
        }
        default:
        {
            assert(false);
            break;
        }
    }

    const uint32_t brickGroupHeight = g_BrickGroupShape[1];
    const uint32_t brickGroupWidth  = g_BrickGroupShape[2];
    const uint32_t brickGroupDepth  = g_BrickGroupShape[3];

    auto addPlan = [&](const TensorShape& outputStripeShape) {
        std::vector<TensorShape> inputStripes;
        for (uint32_t i = 0; i < m_InputTensorShapes.size(); ++i)
        {
            inputStripes.push_back(outputStripeShape);
        }

        // Multiple output stripes might be needed because the following layers may require multiple buffers due to boundary data.
        uint32_t minNumOutputStripes;
        BoundaryRequirements outputBoundaryRequirements = m_OutputBoundaryRequirements.at(0);
        if ((outputBoundaryRequirements.m_NeedsBeforeX || outputBoundaryRequirements.m_NeedsBeforeY) &&
            (outputBoundaryRequirements.m_NeedsAfterX || outputBoundaryRequirements.m_NeedsAfterY))
        {
            minNumOutputStripes = 3;
        }
        else if (outputBoundaryRequirements.m_NeedsBeforeX || outputBoundaryRequirements.m_NeedsBeforeY ||
                 outputBoundaryRequirements.m_NeedsAfterX || outputBoundaryRequirements.m_NeedsAfterY)
        {
            minNumOutputStripes = 2;
        }
        else
        {
            minNumOutputStripes = 1;
        }
        uint32_t maxNumOutputStripes = minNumOutputStripes;
        if (cascadeType == CascadeType::Lonely || cascadeType == CascadeType::End)
        {
            maxNumOutputStripes = minNumOutputStripes + 1;    // For double-buffering.
        }
        // Limit the max number of stripes based on the size of the tensor - there is no point considering plans where
        // we can store more stripes in the tile than there are in the tensor!
        maxNumOutputStripes = std::min(
            maxNumOutputStripes, DivRoundUp(GetHeight(m_OutputTensorShape), GetHeight(outputStripeShape)) *
                                     DivRoundUp(GetWidth(m_OutputTensorShape), GetWidth(outputStripeShape)) *
                                     DivRoundUp(GetChannels(m_OutputTensorShape), GetChannels(outputStripeShape)));
        minNumOutputStripes = std::min(minNumOutputStripes, maxNumOutputStripes);

        for (uint32_t numOutputStripes = minNumOutputStripes; numOutputStripes <= maxNumOutputStripes;
             ++numOutputStripes)
        {
            NumMemoryStripes numMemoryStripes;
            numMemoryStripes.m_Output = numOutputStripes;

            auto op = std::make_unique<PleOp>(m_KernelOperation, static_cast<uint32_t>(m_InputTensorShapes.size()),
                                              inputStripes, outputStripeShape, true, m_Capabilities,
                                              m_SelectionStringParams, m_SelectionIntParams, m_RuntimeParams);

            OwnedOpGraph opGraph;
            PartInputMapping inputMappings;
            PartOutputMapping outputMappings;

            std::vector<Buffer*> pleInputBuffers;
            pleInputBuffers.resize(m_InputTensorShapes.size());

            // PLE input buffers
            for (size_t i = 0; i < m_InputTensorShapes.size(); ++i)
            {
                TileSizeCalculation tileSize =
                    impl::CalculateTileSize(m_Capabilities, m_InputTensorShapes.at(i), outputStripeShape,
                                            PackedBoundaryThickness{ 0, 0, 0, 0 }, 2u, true);

                std::unique_ptr<SramBuffer> buffer = SramBufferBuilder()
                                                         .AddFormat(CascadingBufferFormat::NHWCB)
                                                         .AddDataType(m_DataType)
                                                         .AddTensorShape(m_InputTensorShapes.at(i))
                                                         .AddQuantization(m_InputQuantizationInfos.at(i))
                                                         .AddStripeShape(outputStripeShape)
                                                         .AddNumStripes(2)
                                                         .AddFromTileSize(tileSize);

                SramBuffer* bufferRaw = opGraph.AddBuffer(std::move(buffer));
                pleInputBuffers[i]    = bufferRaw;
            }

            // Output buffer
            auto outBufferAndPleOp =
                AddPleToOpGraph(opGraph, outputStripeShape, numMemoryStripes, std::move(op), m_OutputTensorShape,
                                m_OutputQuantizationInfo, m_DataType, m_CorrespondingOperationIds);

            for (size_t i = 0; i < m_InputTensorShapes.size(); ++i)
            {
                opGraph.AddConsumer(pleInputBuffers[i], outBufferAndPleOp.second, static_cast<uint32_t>(i));
                inputMappings[pleInputBuffers[i]] = PartInputSlot{ m_PartId, static_cast<uint32_t>(i) };
            }

            outputMappings[outBufferAndPleOp.first] = PartOutputSlot{ m_PartId, 0 };
            AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph), {}, plans);
        }
    };

    if (cascadeType == CascadeType::Middle || cascadeType == CascadeType::End)
    {
        TensorShape stripeShape = { 0, 0, 0, 0 };
        for (auto s : sramBufferInputs)
        {
            if (s == nullptr)
            {
                continue;
            }
            // Make sure all the input buffers' stripe shapes match.
            if (stripeShape != TensorShape{ 0, 0, 0, 0 } && s->Sram()->m_StripeShape != stripeShape)
            {
                return Plans{};
            }
            stripeShape = s->Sram()->m_StripeShape;
        }

        addPlan(stripeShape);
    }
    else
    {
        if (stripeConfig.splits.none)
        {
            TensorShape outputStripeShappe = CreateStripe(m_OutputTensorShape, { 0, 0, 0, 0 }, brickGroupDepth);
            addPlan(outputStripeShappe);
        }
        if (stripeConfig.splits.widthOnly)
        {
            TensorShape outputStripeShappe =
                CreateStripe(m_OutputTensorShape, { 0, 0, brickGroupWidth, 0 }, brickGroupDepth);
            addPlan(outputStripeShappe);
        }
        if (stripeConfig.splits.mceAndPleOutputHeight)
        {
            TensorShape outputStripeShappe =
                CreateStripe(m_OutputTensorShape, { 0, brickGroupHeight, 0, 0 }, brickGroupDepth);
            addPlan(outputStripeShappe);
        }

        if (cascadeType == CascadeType::Lonely)
        {
            if (stripeConfig.splits.outputDepthInputDepth)
            {
                TensorShape outputStripeShappe =
                    CreateStripe(m_OutputTensorShape, { 0, 0, 0, brickGroupDepth }, brickGroupDepth);
                addPlan(outputStripeShappe);
            }

            if (stripeConfig.splits.widthHeightOutputDepthInputDepth)
            {
                // Inclusive loops so that we generate plans that split only one or two of the dimensions,
                // but with larger stripe shapes than the non-lonely plans above.
                for (uint32_t stripeHeight : StripeShapeLoop::Inclusive(
                         utils::GetHeight(m_OutputTensorShape), brickGroupHeight,
                         stripeConfig.blockHeightMultiplier.min, stripeConfig.blockHeightMultiplier.max))
                {
                    for (uint32_t stripeWidth : StripeShapeLoop::Inclusive(
                             utils::GetWidth(m_OutputTensorShape), brickGroupWidth,
                             stripeConfig.blockWidthMultiplier.min, stripeConfig.blockWidthMultiplier.max))
                    {
                        for (uint32_t stripeDepth : StripeShapeLoop::Inclusive(
                                 utils::GetChannels(m_OutputTensorShape), brickGroupDepth,
                                 stripeConfig.ofmDepthMultiplier.min, stripeConfig.ofmDepthMultiplier.max))
                        {
                            TensorShape outputStripeShappe = CreateStripe(
                                m_OutputTensorShape, { 0, stripeHeight, stripeWidth, stripeDepth }, brickGroupDepth);
                            addPlan(outputStripeShappe);
                        }
                    }
                }
            }
        }
    }

    return plans;
}

ethosn::support_library::DotAttributes StandalonePlePart::GetDotAttributes(DetailLevel detail) const
{
    DotAttributes result = BasePart::GetDotAttributes(detail);
    if (detail >= DetailLevel::High)
    {
        result.m_Label += "InputTensorShape = " + ArrayToString(m_InputTensorShapes) + "\n";
        result.m_Label += "OutputTensorShape = " + ToString(m_OutputTensorShape) + "\n";
        result.m_Label += "InputQuantizationInfo = " + ArrayToString(m_InputQuantizationInfos) + "\n";
        result.m_Label += "OutputQuantizationInfo = " + ToString(m_OutputQuantizationInfo) + "\n";
        result.m_Label += "KernelOperation = " + ToString(m_KernelOperation) + "\n";
        result.m_Label += "DataType = " + ToString(m_DataType) + "\n";
        result.m_Label += "SelectionStringParams = " + MapToString(m_SelectionStringParams) + "\n";
        result.m_Label += "SelectionIntParams = " + MapToString(m_SelectionIntParams) + "\n";
        result.m_Label += "RuntimeParams = " + MapToString(m_RuntimeParams) + "\n";
    }
    return result;
}

std::vector<BoundaryRequirements> StandalonePlePart::GetInputBoundaryRequirements() const
{
    // We can have multiple inputs, but none of them require boundary data because even
    // for the avgpool kernel, we don't support splitting in width or height.
    return std::vector<BoundaryRequirements>(m_InputTensorShapes.size(), BoundaryRequirements{});
}

std::vector<bool> StandalonePlePart::CanInputsTakePleInputSram() const
{
    // All our inputs need to be in SRAM or DRAM.
    return std::vector<bool>(m_InputTensorShapes.size(), false);
}

}    // namespace support_library
}    // namespace ethosn
