//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ethosn
{
namespace support_library
{

class Node;
class Graph;

void OptimizeGraph(Graph& graph);

bool MergeFormatConversionNodes(Graph& graph, Node* node);
bool MergeRequantizeNodes(Graph& graph, Node* node);
bool ReorderReinterpretAndRequantizeNodes(Graph& graph, Node* node);
bool ReorderConcatAndRequantizeNodes(Graph& graph, Node* node);
bool ReorderConcatAndCopyNodes(Graph& graph, Node* node);
bool MergeCopyAndRequantizeNodes(Graph& graph, Node* node);
bool MergeCopyNodes(Graph& graph, Node* node);
bool MergeConcatNodes(Graph& graph, Node* node);
bool RemoveUnconnectedNode(Graph& graph, Node* node);
bool MergeConstantAndReinterpretNodes(Graph& graph, Node* node);
bool MergeConstantAndFormatConversionNodes(Graph& graph, Node* node);

}    // namespace support_library
}    // namespace ethosn
