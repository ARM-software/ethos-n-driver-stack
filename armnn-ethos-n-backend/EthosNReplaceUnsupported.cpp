//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "EthosNReplaceUnsupported.hpp"

#include <SubgraphView.hpp>
#include <backendsCommon/TensorHandle.hpp>

using namespace armnn;

namespace
{
// Replaces the pattern Constant-Multiplication with an optimized DepthwiseConvolution2d operation.
// Original pattern:
// Input    ->
//              Multiplication -> Output
// Constant ->
// Expected modified pattern:
// Input -> DepthwiseConvolution2d -> Output
//
bool ReplaceConstantMultiplicationWithDepthwise(Graph& graph, Layer* layer)
{
    if (layer->GetType() == LayerType::Multiplication)
    {
        InputSlot* patternSubgraphInput = &layer->GetInputSlot(0);

        Layer* inputLayer    = &patternSubgraphInput->GetConnectedOutputSlot()->GetOwningLayer();
        Layer* constantLayer = &layer->GetInputSlots()[1].GetConnectedOutputSlot()->GetOwningLayer();

        if (constantLayer->GetType() != LayerType::Constant)
        {
            patternSubgraphInput = &layer->GetInputSlot(1);
            std::swap(inputLayer, constantLayer);
        }

        if (constantLayer->GetType() == LayerType::Constant)
        {
            const TensorInfo& inputInfo = inputLayer->GetOutputSlot().GetTensorInfo();
            const TensorInfo& constInfo = constantLayer->GetOutputSlot().GetTensorInfo();

            // Add a Depthwise only where the constant input is a scalar that takes the form { 1, 1, 1, C }.
            // The scalar is used as weights for the convolution.
            if (constInfo.GetShape() == TensorShape({ 1, 1, 1, inputInfo.GetShape()[3] }))
            {
                Graph replacementGraph;

                DepthwiseConvolution2dDescriptor desc;
                desc.m_DataLayout = DataLayout::NHWC;

                const auto depthwiseLayer = replacementGraph.AddLayer<DepthwiseConvolution2dLayer>(
                    desc, "Replacement for Constant-Multiplication");

                TensorInfo weightInfo = constInfo;
                weightInfo.SetShape({ 1, constInfo.GetShape()[3], 1, 1 });

                const void* weightData = PolymorphicPointerDowncast<const ConstantLayer>(constantLayer)
                                             ->m_LayerOutput->GetConstTensor<void>();

                const ConstTensor weights(weightInfo, weightData);

                depthwiseLayer->m_Weight = std::make_unique<ScopedTensorHandle>(weights);

                SubgraphView patternSubgraph({ patternSubgraphInput }, { &layer->GetOutputSlot() },
                                             { layer, constantLayer });

                graph.SubstituteSubgraph(patternSubgraph, SubgraphView{ depthwiseLayer });

                return true;
            }
        }
    }
    return false;
}
}    // namespace

namespace armnn
{
namespace ethosnbackend
{

void ReplaceUnsupportedLayers(Graph& graph)
{
    using ReplacementFunc                    = bool (*)(Graph&, Layer*);
    const ReplacementFunc replacementFuncs[] = {
        &ReplaceConstantMultiplicationWithDepthwise,
    };

    bool madeChange;
    do
    {
        madeChange = false;
        for (Layer* layer : graph)
        {
            for (const ReplacementFunc f : replacementFuncs)
            {
                madeChange = f(graph, layer);
                if (madeChange)
                {
                    goto nextIteration;
                }
            }
        }
    nextIteration:;
    } while (madeChange);
}
}    // namespace ethosnbackend

}    // namespace armnn
