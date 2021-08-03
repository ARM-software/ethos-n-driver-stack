//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "Optimization.hpp"

#include "GraphNodes.hpp"
#include "Utils.hpp"

namespace ethosn
{
namespace support_library
{

void OptimizeGraph(Graph& graph)
{
    using OptimizationFunc                     = bool (*)(Graph&, Node*);
    const OptimizationFunc optimizationFuncs[] = {
        &MergeFormatConversionNodes,
        &ReorderReinterpretAndRequantizeNodes,
        &ReorderConcatAndRequantizeNodes,
        &ReorderConcatAndCopyNodes,
        &MergeCopyAndRequantizeNodes,
        &MergeRequantizeNodes,
        &MergeCopyNodes,
        &MergeConcatNodes,
        &RemoveUnconnectedNode,
        &MergeConstantAndReinterpretNodes,
        &MergeConstantAndFormatConversionNodes,
    };

    bool madeChange;
    do
    {
        madeChange = false;
        for (Node* node : graph.GetNodesSorted())
        {
            for (const OptimizationFunc f : optimizationFuncs)
            {
                madeChange = f(graph, node);
                if (madeChange)
                {
                    goto nextIteration;
                }
            }
        }
    nextIteration:;
    } while (madeChange);
}

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

    if (requantizeNode && requantizeNode->GetOutputs().size() == 1)
    {
        // Add the corresponding ids from the first requantize node (the removed one) to the second one (the one we are keeping)
        RequantizeNode* nextNode = dynamic_cast<RequantizeNode*>(requantizeNode->GetOutput(0)->GetDestination());

        if (nextNode)
        {
            nextNode->AddCorrespondingOperationIDs(requantizeNode->GetCorrespondingOperationIds());
            graph.CollapseNode(requantizeNode);
            return true;
        }
    }
    return false;
}

bool MergeCopyNodes(Graph& graph, Node* node)
{
    // Two adjacent copy nodes can be merged:
    //
    //   X --> CopyNode --> CopyNode -->
    //
    //  Becomes
    //
    //  X --> CopyNode -->
    CopyNode* copyNode = dynamic_cast<CopyNode*>(node);
    if (copyNode && copyNode->GetOutputs().size() == 1 &&
        dynamic_cast<CopyNode*>(copyNode->GetOutput(0)->GetDestination()))
    {
        // Add the corresponding ids from the first copy node (the removed one) to the second one (the one we are keeping)
        CopyNode* nextNode = dynamic_cast<CopyNode*>(copyNode->GetOutput(0)->GetDestination());
        nextNode->AddCorrespondingOperationIDs(copyNode->GetCorrespondingOperationIds());

        graph.CollapseNode(copyNode);
        return true;
    }
    return false;
}

bool MergeCopyAndRequantizeNodes(Graph& graph, Node* node)
{
    // Two adjacent Copy and requantize nodes can be merged
    //
    //   X --> CopyNode --> RequantizeNode to (1, -84)  -->
    //
    //  Becomes
    //
    //  X -->  RequantizeNode to (1, -84) -->
    CopyNode* copyNode = dynamic_cast<CopyNode*>(node);
    if (copyNode && copyNode->GetOutputs().size() == 1 &&
        dynamic_cast<RequantizeNode*>(copyNode->GetOutput(0)->GetDestination()))
    {
        // Add the corresponding ids from the copy node to the requantize node
        RequantizeNode* nextNode = dynamic_cast<RequantizeNode*>(copyNode->GetOutput(0)->GetDestination());
        nextNode->AddCorrespondingOperationIDs(copyNode->GetCorrespondingOperationIds());

        graph.CollapseNode(copyNode);
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
        Node* newRequant     = graph.CreateAndAddNodeWithDebug<RequantizeNode>(
            "ReorderReinterpretAndRequantizeNodes", reinterpetNode->GetInputShape(0), oldRequantNode->GetDataType(),
            oldRequantNode->GetQuantizationInfo(), oldRequantNode->GetInputFormat(0),
            oldRequantNode->GetCorrespondingOperationIds());
        graph.SplitEdge(reinterpetNode->GetInput(0), newRequant);
        graph.CollapseNode(oldRequantNode);
        return true;
    }
    return false;
}

bool ReorderConcatAndCopyNodes(Graph& graph, Node* node)
{
    // A concat followed by a copy can be reordered so that the copy occurs on each input of the concat.
    // This is required to be able to merge concat followed by another concat
    //
    //  X0 -->
    //  X1 -->  ConcatNode  --> CopyNode -->
    //  X2 -->
    //
    //  Becomes
    //
    //  X0 --> CopyNode -->
    //  X1 --> CopyNode --> ConcatNode -->
    //  X2 --> CopyNode -->
    ConcatNode* concatenationNode = dynamic_cast<ConcatNode*>(node);
    if (concatenationNode && concatenationNode->GetOutputs().size() == 1 &&
        dynamic_cast<CopyNode*>(concatenationNode->GetOutput(0)->GetDestination()))
    {
        Node* oldCopyNode = dynamic_cast<CopyNode*>(concatenationNode->GetOutput(0)->GetDestination());

        for (uint32_t i = 0; i < concatenationNode->GetInputs().size(); ++i)
        {
            Node* newCopy = graph.CreateAndAddNodeWithDebug<CopyNode>(
                "ReorderConcatAndCopyNodes", concatenationNode->GetInputShape(i), oldCopyNode->GetDataType(),
                oldCopyNode->GetQuantizationInfo(), concatenationNode->GetInputFormat(i),
                oldCopyNode->GetCorrespondingOperationIds());
            graph.SplitEdge(concatenationNode->GetInput(i), newCopy);
        }
        graph.CollapseNode(oldCopyNode);
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
            Node* newRequant = graph.CreateAndAddNodeWithDebug<RequantizeNode>(
                "ReorderConcatAndRequantizeNodes", concatenationNode->GetInputShape(i), oldRequantNode->GetDataType(),
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
        Node* newConstantNode = graph.CreateAndAddNodeWithDebug<ConstantNode>(
            "MergeConstantAndReinterpretNodes", constantInfo, constantNode->GetConstantData(),
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

}    // namespace support_library
}    // namespace ethosn
