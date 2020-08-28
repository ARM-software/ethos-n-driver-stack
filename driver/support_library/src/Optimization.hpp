//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ethosn
{
namespace support_library
{

class Node;
class Graph;

bool MergeFormatConversionNodes(Graph& graph, Node* node);
bool MergeRequantizeNodes(Graph& graph, Node* node);
bool ReorderReinterpretAndRequantizeNodes(Graph& graph, Node* node);
bool ReorderConcatAndRequantizeNodes(Graph& graph, Node* node);
bool MergeConcatNodes(Graph& graph, Node* node);
bool RemoveUnconnectedNode(Graph& graph, Node* node);
bool MergeConstantAndReinterpretNodes(Graph& graph, Node* node);
bool MergeConstantAndFormatConversionNodes(Graph& graph, Node* node);
bool ReplaceConstantAdditionWithDepthwise(Graph& graph, Node* node);

}    // namespace support_library
}    // namespace ethosn
