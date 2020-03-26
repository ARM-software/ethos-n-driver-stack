//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Network.hpp"

#include <unordered_map>

namespace ethosn
{
namespace support_library
{

class Graph;
class Node;

class NetworkToGraphConverter : public INetworkVisitor
{
public:
    NetworkToGraphConverter(Graph& graph, const HardwareCapabilities& capabilities, bool estimationMode)
        : m_Graph(graph)
        , m_Capabilities(capabilities)
        , m_EstimationMode(estimationMode)
    {}

    void Visit(Input& input) final;
    void Visit(Output& output) final;
    void Visit(Convolution& convolution) final;
    void Visit(DepthwiseConvolution& depthwiseConvolution) final;
    void Visit(TransposeConvolution& transposeConvolution) final;
    void Visit(Constant& constant) final;
    void Visit(Concatenation& concatenation) final;
    void Visit(Split& split) final;
    void Visit(Addition& addition) final;
    void Visit(FullyConnected& fullyConnected) final;
    void Visit(Relu& relu) final;
    void Visit(Softmax& softmax) final;
    void Visit(Sigmoid& sigmoid) final;
    void Visit(Pooling& pooling) final;
    void Visit(Reshape& reshape) final;
    void Visit(DepthToSpace& depthToSpace) final;
    void Visit(EstimateOnly& estimateOnly) final;

private:
    /// Connects a node into the Graph to represent the given single/zero-output operation.
    /// The node will have its inputs connected to the nodes representing the inputs of the Operation.
    void ConnectNode(const Operation& operation, Node* node);

    /// Connects a linear list of nodes into the Graph to represent the given single/zero-output operation.
    /// The first node in the list will have its inputs connected to the nodes representing the inputs of the Operation.
    void ConnectNodeChain(const Operation& operation, const std::vector<Node*>& linearNodes);

    /// For each Operand in the input Network that we have visited,
    /// this contains the corresponding Node in the resulting Graph that produces the equivalent of that Operand.
    std::unordered_map<const Operand*, Node*> m_OperandToNode;

    Graph& m_Graph;
    const HardwareCapabilities& m_Capabilities;
    const bool m_EstimationMode;
};

}    // namespace support_library
}    // namespace ethosn
