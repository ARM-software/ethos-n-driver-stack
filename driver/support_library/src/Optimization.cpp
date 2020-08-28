//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "Optimization.hpp"

#include "GraphNodes.hpp"

namespace ethosn
{
namespace support_library
{

bool MergeFormatConversionNodes(Graph& graph, Node* node)
{
    // Two adjacent format conversions which perform opposite conversions can be eliminated:
    //
    //   X (NHWCB) -->  FormatConversionNode to NHWC  -->  FormatConversionNode to NHWCB -->
    //
    //  Becomes
    //
    //  X (NHWCB) -->
    FormatConversionNode* conversionNode = dynamic_cast<FormatConversionNode*>(node);
    if (conversionNode && conversionNode->GetOutputs().size() == 1 &&
        conversionNode->GetOptimizationHint() != OptimizationHint::DoNotMerge)
    {
        FormatConversionNode* nextFormatConversionNode =
            dynamic_cast<FormatConversionNode*>(conversionNode->GetOutput(0)->GetDestination());
        if (nextFormatConversionNode && nextFormatConversionNode->GetOptimizationHint() != OptimizationHint::DoNotMerge)
        {
            if (conversionNode->GetInputFormat(0) == nextFormatConversionNode->GetFormat())
            {
                graph.CollapseEdge(conversionNode->GetInput(0));
                graph.CollapseEdge(nextFormatConversionNode->GetInput(0));
                return true;
            }
        }
    }
    return false;
}

bool MergeRequantizeNodes(Graph& graph, Node* node)
{
    // Two adjacent requantize nodes can be merged:
    //
    //   X -->  RequantizeNode to (0.1, 74) --> RequantizeNode to (1, -84)  -->
    //
    //  Becomes
    //
    //  X -->  RequantizeNode to (1, -84) -->
    RequantizeNode* requantizeNode = dynamic_cast<RequantizeNode*>(node);
    if (requantizeNode && requantizeNode->GetOutputs().size() == 1 &&
        dynamic_cast<RequantizeNode*>(requantizeNode->GetOutput(0)->GetDestination()))
    {
        // Add the corresponding ids from the first requantize node (the removed one) to the second one (the one we are keeping)
        RequantizeNode* nextNode = dynamic_cast<RequantizeNode*>(requantizeNode->GetOutput(0)->GetDestination());
        nextNode->AddCorrespondingOperationIDs(requantizeNode->GetCorrespondingOperationIds());

        graph.CollapseNode(requantizeNode);
        return true;
    }
    return false;
}

bool ReorderReinterpretAndRequantizeNodes(Graph& graph, Node* node)
{
    // A reinterpret followed by a requantize can be reordered so the requantize is first.
    // This is required to be able to do the requantize as part of a preceding MceOperation
    //
    //  X -->  ReinterpretNode --> RequantizeNode to (-1, 84) -->
    //
    //  Becomes
    //
    //  X --> RequantizeNode to (-1, 84) --> ReinterpretNode -->
    ReinterpretNode* reinterpetNode = dynamic_cast<ReinterpretNode*>(node);
    if (reinterpetNode && reinterpetNode->GetOutputs().size() == 1 &&
        dynamic_cast<RequantizeNode*>(reinterpetNode->GetOutput(0)->GetDestination()))
    {
        Node* oldRequantNode = dynamic_cast<RequantizeNode*>(reinterpetNode->GetOutput(0)->GetDestination());
        Node* newRequant     = graph.CreateAndAddNode<RequantizeNode>(
            reinterpetNode->GetInputShape(0), oldRequantNode->GetDataType(), oldRequantNode->GetQuantizationInfo(),
            oldRequantNode->GetInputFormat(0), oldRequantNode->GetCorrespondingOperationIds());
        graph.SplitEdge(reinterpetNode->GetInput(0), newRequant);
        graph.CollapseNode(oldRequantNode);
        return true;
    }
    return false;
}

bool ReorderConcatAndRequantizeNodes(Graph& graph, Node* node)
{
    // A concat followed by a requantize can be reordered so that the requantize occurs on each input of the concat.
    // This is required to be able to do the requantize as part of a preceding MceOperation
    //
    //  X0 -->
    //  X1 -->  ConcatNode  --> RequantizeNode to (-1, 84) -->
    //  X2 -->
    //
    //  Becomes
    //
    //  X0 --> RequantizeNode to (-1, 84) -->
    //  X1 --> RequantizeNode to (-1, 84) --> ConcatNode -->
    //  X2 --> RequantizeNode to (-1, 84) -->
    ConcatNode* concatenationNode = dynamic_cast<ConcatNode*>(node);
    if (concatenationNode && concatenationNode->GetOutputs().size() == 1 &&
        dynamic_cast<RequantizeNode*>(concatenationNode->GetOutput(0)->GetDestination()))
    {
        Node* oldRequantNode = dynamic_cast<RequantizeNode*>(concatenationNode->GetOutput(0)->GetDestination());
        for (uint32_t i = 0; i < concatenationNode->GetInputs().size(); ++i)
        {
            Node* newRequant = graph.CreateAndAddNode<RequantizeNode>(
                concatenationNode->GetInputShape(i), oldRequantNode->GetDataType(),
                oldRequantNode->GetQuantizationInfo(), concatenationNode->GetInputFormat(i),
                oldRequantNode->GetCorrespondingOperationIds());
            graph.SplitEdge(concatenationNode->GetInput(i), newRequant);
        }
        graph.CollapseNode(oldRequantNode);
        return true;
    }
    return false;
}

bool MergeConcatNodes(Graph& graph, Node* node)
{
    // This is for use case of concatenation to concatenation in the graph, for example
    // Before:
    // concatNode0      concatNode1
    //     \                /
    //         concatNode2
    // After:
    //         concatNode2
    //
    ConcatNode* concatenationNode = dynamic_cast<ConcatNode*>(node);
    if (concatenationNode && concatenationNode->GetInputs().size() > 1 &&
        concatenationNode->GetOptimizationHint() != OptimizationHint::DoNotMerge)
    {
        for (uint32_t i = 0; i < concatenationNode->GetInputs().size(); ++i)
        {
            ConcatNode* prevConcatenationNode = dynamic_cast<ConcatNode*>(concatenationNode->GetInput(i)->GetSource());
            if (prevConcatenationNode)
            {
                // preserve the corresponding ID from the concat node we are removing
                concatenationNode->AddCorrespondingOperationIDs(prevConcatenationNode->GetCorrespondingOperationIds());
                graph.CollapseNode(prevConcatenationNode);
                return true;
            }
        }
    }
    return false;
}

bool RemoveUnconnectedNode(Graph& graph, Node* node)
{
    // Remove unconnected nodes
    // Before:
    // Node0   Node1
    //         /
    //      Node2
    // After:
    //        Node1
    //         /
    //      Node2
    //
    OutputNode* outputNode = dynamic_cast<OutputNode*>(node);
    if (outputNode == nullptr && node->GetOutputs().size() == 0)
    {
        graph.RemoveNode(node);
        return true;
    }
    return false;
}

bool MergeConstantAndReinterpretNodes(Graph& graph, Node* node)
{
    // Merge Constant node with ReinterpretNode if any.
    // Before:
    //         ConstantNode
    //         /
    //      ReinterpretNode
    // After:
    //        ConstantNode
    //
    ConstantNode* constantNode = dynamic_cast<ConstantNode*>(node);
    if (constantNode && constantNode->GetOutputs().size() == 1 &&
        constantNode->GetFormat() == CompilerDataFormat::NHWC &&
        dynamic_cast<ReinterpretNode*>(constantNode->GetOutput(0)->GetDestination()))
    {
        // Statically reshape the constant node shape.
        ReinterpretNode* reinterpetNode = dynamic_cast<ReinterpretNode*>(constantNode->GetOutput(0)->GetDestination());
        const TensorInfo constantInfo(reinterpetNode->GetShape(), constantNode->GetConstantDataType(), DataFormat::NHWC,
                                      constantNode->GetQuantizationInfo());
        Node* newConstantNode = graph.CreateAndAddNode<ConstantNode>(constantInfo, constantNode->GetConstantData(),
                                                                     node->GetCorrespondingOperationIds());
        // preserve the operation ids from the nodes that are being removed
        newConstantNode->AddCorrespondingOperationIDs(reinterpetNode->GetCorrespondingOperationIds());

        graph.InsertNodeAfter(reinterpetNode, newConstantNode);
        graph.CollapseNode(reinterpetNode);
        graph.CollapseNode(constantNode);
        return true;
    }
    return false;
}

bool MergeConstantAndFormatConversionNodes(Graph& graph, Node* node)
{

    // Merge Constant node with FormatConversionNode if any.
    // Before:
    //         ConstantNode
    //         /
    //      FormatConversionNode
    // After:
    //        ConstantNode
    //
    ConstantNode* constantNode = dynamic_cast<ConstantNode*>(node);
    if (constantNode && constantNode->GetOutputs().size() == 1 &&
        constantNode->GetFormat() == CompilerDataFormat::NHWC &&
        dynamic_cast<FormatConversionNode*>(constantNode->GetOutput(0)->GetDestination()))
    {
        graph.CollapseEdge(constantNode->GetOutput(0));
        return true;
    }
    return false;
}

bool ReplaceConstantAdditionWithDepthwise(Graph& graph, Node* node)
{
    // Replace Constant node and Addition node with a new MceOperationNode.
    // Before:
    // constantNode          inputNode
    //          \                /
    //      StandalonePleOperationNode
    // After:
    //                inputNode
    //                   /
    //   MceOperationNode (identity depthwise where the bias is the constant)
    //
    ConstantNode* constantNode = dynamic_cast<ConstantNode*>(node);
    if (constantNode && constantNode->GetOutputs().size() == 1 &&
        constantNode->GetFormat() == CompilerDataFormat::NHWC &&
        dynamic_cast<StandalonePleOperationNode*>(constantNode->GetOutput(0)->GetDestination()))
    {
        StandalonePleOperationNode* pleOperationNode =
            dynamic_cast<StandalonePleOperationNode*>(constantNode->GetOutput(0)->GetDestination());

        if (pleOperationNode->GetKernelOperation() == command_stream::PleOperation::ADDITION ||
            pleOperationNode->GetKernelOperation() == command_stream::PleOperation::ADDITION_RESCALE)
        {
            // if input shape is { 1, 1, 1, C } add an identity depthwise instead where the bias values are the constant vals from the bias add
            bool isConstantBroadcastAddChannels = constantNode->GetShape()[0] == 1 &&
                                                  constantNode->GetShape()[1] == 1 && constantNode->GetShape()[2] == 1;

            if (isConstantBroadcastAddChannels)
            {
                const TensorInfo constantLayerInfo(constantNode->GetShape(), constantNode->GetConstantDataType(),
                                                   DataFormat::NHWC, constantNode->GetQuantizationInfo());

                std::vector<uint8_t> constantLayerData = constantNode->GetConstantData();
                const Padding& padding                 = { 0, 0, 0, 0 };

                // Assume there is only one constant input (and only 2 inputs total).
                // In this case the input to the depthwise will be the non constant one.
                uint8_t idxOfInput = 0;

                // If the constant one is at idx 0, then it must be the other one.
                if (dynamic_cast<ConstantNode*>(pleOperationNode->GetInput(0)->GetSource()))
                {
                    idxOfInput = 1;
                }

                Node* inputNode = pleOperationNode->GetInput(idxOfInput)->GetSource();

                const TensorShape inputShape = inputNode->GetShape();

                if (inputShape[3] == constantNode->GetShape()[3])
                {

                    const QuantizationInfo& outputQuantInfo =
                        pleOperationNode->GetOutput(0)->GetSource()->GetQuantizationInfo();

                    TensorShape outputShape = pleOperationNode->GetOutput(0)->GetSource()->GetShape();

                    const uint32_t numIfm = inputShape[3];
                    // Since the constant input is being requantized, the weight scale and values must be chosen
                    // A weight scale and data must satisify the following requirements:
                    //   - the resulting weight data for the identity convolution doesn't saturate
                    //       (i.e. must be between 1 and 255)
                    //   - inputQuantScale * weightQuantScale needs to be less than the outputQuantScale
                    //       (See CalculateQuantizedMultiplierSmallerThanOne in Utils.hpp)
                    // TODO: Add support for per-channel quantization
                    const float weightScaleUpperBound =
                        std::min(outputQuantInfo.GetScale() / inputNode->GetQuantizationInfo().GetScale(), 1.f);
                    constexpr float weightScaleLowerBound = (1.f / 255.f);
                    if (weightScaleUpperBound < weightScaleLowerBound)
                    {
                        throw NotSupportedException("Couldn't choose appropriate weight scale for bias add");
                    }
                    const float weightScaleTarget = (weightScaleUpperBound + weightScaleLowerBound) / 2.f;
                    // The reciprical of the scale needs to be a whole number to minimize rounding error.
                    const float weightScaleRecipRounded = std::round(1.f / weightScaleTarget);
                    const float weightScale             = 1.f / weightScaleRecipRounded;
                    const float newConstantLayerScale   = weightScale * inputNode->GetQuantizationInfo().GetScale();

                    std::vector<uint8_t> weightsData(1 * 1 * 1 * numIfm, static_cast<uint8_t>(weightScaleRecipRounded));

                    TensorInfo weightInfo{
                        { 1, 1, numIfm, 1 }, DataType::UINT8_QUANTIZED, DataFormat::HWIM, { 0, weightScale }
                    };

                    QuantizationInfo constantNodeQuantizationInfo = constantNode->GetQuantizationInfo();
                    auto dataType                                 = constantNode->GetConstantDataType();
                    if (dataType == DataType::UINT8_QUANTIZED)
                    {
                        std::vector<int32_t> newConstantLayerData;
                        for (uint32_t k = 0; k < constantLayerData.size(); ++k)
                        {
                            float fpValue = constantNodeQuantizationInfo.GetScale() *
                                            static_cast<float>((constantLayerData.at(k) -
                                                                constantNodeQuantizationInfo.GetZeroPoint()));
                            newConstantLayerData.push_back(
                                static_cast<int32_t>(std::round(fpValue / newConstantLayerScale)));
                        }
                        Node* mceNode = graph.CreateAndAddNode<MceOperationNode>(
                            inputShape, outputShape, dataType, outputQuantInfo, weightInfo, weightsData,
                            constantLayerInfo, newConstantLayerData, Stride{ 1, 1 }, padding.m_Top, padding.m_Left,
                            ethosn::command_stream::MceOperation::DEPTHWISE_CONVOLUTION, CompilerDataFormat::NHWCB,
                            node->GetCorrespondingOperationIds());

                        mceNode->AddCorrespondingOperationIDs(pleOperationNode->GetCorrespondingOperationIds());

                        graph.InsertNodeAfter(inputNode, mceNode);
                        graph.CollapseEdge(mceNode->GetOutput(0));
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

}    // namespace support_library
}    // namespace ethosn
