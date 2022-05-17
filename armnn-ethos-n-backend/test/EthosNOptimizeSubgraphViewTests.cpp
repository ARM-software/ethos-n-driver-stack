//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include <CommonTestUtils.hpp>
#include <EthosNBackend.hpp>
#include <EthosNBackendId.hpp>
#include <EthosNSubgraphViewConverter.hpp>
#include <Graph.hpp>
#include <armnn/BackendRegistry.hpp>
#include <doctest/doctest.h>

using namespace armnn;

namespace
{

// Creates a subgraph containing unsupported layers (the pooling layers have an unsupported configuration)
SubgraphView::SubgraphViewPtr BuildUnsupportedSubgraph(Graph& graph)
{
    const TensorInfo inputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo outputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo weightInfo({ 16, 1, 1, 16 }, DataType::QAsymmU8, 0.9f, 0, true);
    const TensorInfo biasInfo({ 1, 1, 1, 16 }, DataType::Signed32, 0.9f, 0, true);

    Convolution2dDescriptor convolutionDescriptor;
    convolutionDescriptor.m_StrideX     = 1;
    convolutionDescriptor.m_StrideY     = 1;
    convolutionDescriptor.m_BiasEnabled = true;
    convolutionDescriptor.m_DataLayout  = DataLayout::NHWC;

    Pooling2dDescriptor poolingDescriptor;
    poolingDescriptor.m_PoolType      = armnn::PoolingAlgorithm::Average;
    poolingDescriptor.m_PoolWidth     = 2;
    poolingDescriptor.m_PoolHeight    = 2;
    poolingDescriptor.m_StrideX       = 2;
    poolingDescriptor.m_StrideY       = 2;
    poolingDescriptor.m_PadLeft       = 1;
    poolingDescriptor.m_PadRight      = 1;
    poolingDescriptor.m_PadTop        = 1;
    poolingDescriptor.m_PadBottom     = 1;
    poolingDescriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;
    poolingDescriptor.m_DataLayout    = DataLayout::NHWC;

    // Construct the graph
    Layer* const inputLayer = graph.AddLayer<InputLayer>(0, "input layer");
    CHECK(inputLayer);
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    Convolution2dLayer* const conv1Layer = graph.AddLayer<Convolution2dLayer>(convolutionDescriptor, "conv1 layer");
    CHECK(conv1Layer);

    // Arm NN is transitioning from having weights/bias as intrinsic properties of the layer to having them
    // as separate layers with connections. For now, we need to do both.
    auto weights           = graph.AddLayer<ConstantLayer>("Weights");
    weights->m_LayerOutput = std::make_unique<ScopedTensorHandle>(weightInfo);
    weights->m_LayerOutput->Allocate();
    weights->GetOutputSlot().SetTensorInfo(weightInfo);
    weights->GetOutputSlot().Connect(conv1Layer->GetInputSlot(1));
    auto bias           = graph.AddLayer<ConstantLayer>("Bias");
    bias->m_LayerOutput = std::make_unique<ScopedTensorHandle>(biasInfo);
    bias->m_LayerOutput->Allocate();
    bias->GetOutputSlot().SetTensorInfo(biasInfo);
    bias->GetOutputSlot().Connect(conv1Layer->GetInputSlot(2));

    SetWeightAndBias(conv1Layer, weightInfo, biasInfo);
    conv1Layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    Pooling2dLayer* const pooling1Layer = graph.AddLayer<Pooling2dLayer>(poolingDescriptor, "pooling1 layer");
    CHECK(pooling1Layer);
    pooling1Layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    Pooling2dLayer* const pooling2Layer = graph.AddLayer<Pooling2dLayer>(poolingDescriptor, "pooling2 layer");
    CHECK(pooling2Layer);
    pooling2Layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    Convolution2dLayer* const conv2Layer = graph.AddLayer<Convolution2dLayer>(convolutionDescriptor, "conv2 layer");
    CHECK(conv2Layer);

    // Arm NN is transitioning from having weights/bias as intrinsic properties of the layer to having them
    // as separate layers with connections. For now, we need to do both.
    auto weights2           = graph.AddLayer<ConstantLayer>("Weights");
    weights2->m_LayerOutput = std::make_unique<ScopedTensorHandle>(weightInfo);
    weights2->m_LayerOutput->Allocate();
    weights2->GetOutputSlot().SetTensorInfo(weightInfo);
    weights2->GetOutputSlot().Connect(conv2Layer->GetInputSlot(1));
    auto bias2           = graph.AddLayer<ConstantLayer>("Bias");
    bias2->m_LayerOutput = std::make_unique<ScopedTensorHandle>(biasInfo);
    bias2->m_LayerOutput->Allocate();
    bias2->GetOutputSlot().SetTensorInfo(biasInfo);
    bias2->GetOutputSlot().Connect(conv2Layer->GetInputSlot(2));

    SetWeightAndBias(conv2Layer, weightInfo, biasInfo);
    conv2Layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    Pooling2dLayer* const pooling3Layer = graph.AddLayer<Pooling2dLayer>(poolingDescriptor, "pooling3 layer");
    CHECK(pooling3Layer);
    pooling3Layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    Layer* const outputLayer = graph.AddLayer<OutputLayer>(0, "output layer");
    CHECK(outputLayer);

    // Connect the network
    inputLayer->GetOutputSlot(0).Connect(conv1Layer->GetInputSlot(0));
    conv1Layer->GetOutputSlot(0).Connect(pooling1Layer->GetInputSlot(0));
    pooling1Layer->GetOutputSlot(0).Connect(pooling2Layer->GetInputSlot(0));
    pooling2Layer->GetOutputSlot(0).Connect(conv2Layer->GetInputSlot(0));
    conv2Layer->GetOutputSlot(0).Connect(pooling3Layer->GetInputSlot(0));
    pooling3Layer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Create the subgraph view for the whole network
    armnn::SubgraphView view(
        { conv1Layer, weights, bias, pooling1Layer, pooling2Layer, conv2Layer, weights2, bias2, pooling3Layer },
        { &conv1Layer->GetInputSlot(0) }, { &pooling3Layer->GetOutputSlot(0) });
    return std::make_unique<SubgraphView>(view);
}

// Creates a simple subgraph with only one convolution layer, supported by the Ethos-N backend
SubgraphView::SubgraphViewPtr BuildFullyOptimizableSubgraph1(Graph& graph)
{
    const TensorInfo inputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo outputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo weightInfo({ 16, 1, 1, 16 }, DataType::QAsymmU8, 0.9f, 0, true);
    const TensorInfo biasInfo({ 1, 1, 1, 16 }, DataType::Signed32, 0.9f, 0, true);

    Convolution2dDescriptor convolutionDescriptor;
    convolutionDescriptor.m_StrideX     = 1;
    convolutionDescriptor.m_StrideY     = 1;
    convolutionDescriptor.m_BiasEnabled = true;
    convolutionDescriptor.m_DataLayout  = DataLayout::NHWC;

    // Construct the graph
    Layer* const inputLayer = graph.AddLayer<InputLayer>(0, "input layer");
    CHECK(inputLayer);
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    Convolution2dLayer* const convLayer = graph.AddLayer<Convolution2dLayer>(convolutionDescriptor, "conv layer");
    CHECK(convLayer);

    // Arm NN is transitioning from having weights/bias as intrinsic properties of the layer to having them
    // as separate layers with connections. For now, we need to do both.
    auto weights           = graph.AddLayer<ConstantLayer>("Weights");
    weights->m_LayerOutput = std::make_unique<ScopedTensorHandle>(weightInfo);
    weights->m_LayerOutput->Allocate();
    weights->GetOutputSlot().SetTensorInfo(weightInfo);
    weights->GetOutputSlot().Connect(convLayer->GetInputSlot(1));
    auto bias           = graph.AddLayer<ConstantLayer>("Bias");
    bias->m_LayerOutput = std::make_unique<ScopedTensorHandle>(biasInfo);
    bias->m_LayerOutput->Allocate();
    bias->GetOutputSlot().SetTensorInfo(biasInfo);
    bias->GetOutputSlot().Connect(convLayer->GetInputSlot(2));

    SetWeightAndBias(convLayer, weightInfo, biasInfo);
    convLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    Layer* const outputLayer = graph.AddLayer<OutputLayer>(0, "output layer");
    CHECK(outputLayer);

    // Connect the network
    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    convLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Create the subgraph view for the whole network
    armnn::SubgraphView view({ convLayer, weights, bias }, { &convLayer->GetInputSlot(0) },
                             { &convLayer->GetOutputSlot(0) });
    return std::make_unique<SubgraphView>(view);
}

// Creates a more complex subgraph with five convolutions layers, all supported by the Ethos-N backend
SubgraphView::SubgraphViewPtr BuildFullyOptimizableSubgraph2(Graph& graph)
{
    const TensorInfo inputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo outputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo weightInfo({ 16, 1, 1, 16 }, DataType::QAsymmU8, 0.9f, 0, true);
    const TensorInfo biasInfo({ 1, 1, 1, 16 }, DataType::Signed32, 0.9f, 0, true);

    Convolution2dDescriptor convolutionDescriptor;
    convolutionDescriptor.m_StrideX     = 1;
    convolutionDescriptor.m_StrideY     = 1;
    convolutionDescriptor.m_BiasEnabled = true;
    convolutionDescriptor.m_DataLayout  = DataLayout::NHWC;

    // Construct the graph
    Layer* const inputLayer = graph.AddLayer<InputLayer>(0, "input layer");
    CHECK(inputLayer);
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    Convolution2dLayer* const conv1Layer = graph.AddLayer<Convolution2dLayer>(convolutionDescriptor, "conv1 layer");
    CHECK(conv1Layer);

    // Arm NN is transitioning from having weights/bias as intrinsic properties of the layer to having them
    // as separate layers with connections. For now, we need to do both.
    auto weights           = graph.AddLayer<ConstantLayer>("Weights");
    weights->m_LayerOutput = std::make_unique<ScopedTensorHandle>(weightInfo);
    weights->m_LayerOutput->Allocate();
    weights->GetOutputSlot().SetTensorInfo(weightInfo);
    weights->GetOutputSlot().Connect(conv1Layer->GetInputSlot(1));
    auto bias           = graph.AddLayer<ConstantLayer>("Bias");
    bias->m_LayerOutput = std::make_unique<ScopedTensorHandle>(biasInfo);
    bias->m_LayerOutput->Allocate();
    bias->GetOutputSlot().SetTensorInfo(biasInfo);
    bias->GetOutputSlot().Connect(conv1Layer->GetInputSlot(2));

    SetWeightAndBias(conv1Layer, weightInfo, biasInfo);
    conv1Layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    Convolution2dLayer* const conv2Layer = graph.AddLayer<Convolution2dLayer>(convolutionDescriptor, "conv2 layer");
    CHECK(conv2Layer);

    // Arm NN is transitioning from having weights/bias as intrinsic properties of the layer to having them
    // as separate layers with connections. For now, we need to do both.
    auto weights2           = graph.AddLayer<ConstantLayer>("Weights");
    weights2->m_LayerOutput = std::make_unique<ScopedTensorHandle>(weightInfo);
    weights2->m_LayerOutput->Allocate();
    weights2->GetOutputSlot().SetTensorInfo(weightInfo);
    weights2->GetOutputSlot().Connect(conv2Layer->GetInputSlot(1));
    auto bias2           = graph.AddLayer<ConstantLayer>("Bias");
    bias2->m_LayerOutput = std::make_unique<ScopedTensorHandle>(biasInfo);
    bias2->m_LayerOutput->Allocate();
    bias2->GetOutputSlot().SetTensorInfo(biasInfo);
    bias2->GetOutputSlot().Connect(conv2Layer->GetInputSlot(2));

    SetWeightAndBias(conv2Layer, weightInfo, biasInfo);
    conv2Layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    Convolution2dLayer* const conv3Layer = graph.AddLayer<Convolution2dLayer>(convolutionDescriptor, "conv3 layer");
    CHECK(conv3Layer);

    // Arm NN is transitioning from having weights/bias as intrinsic properties of the layer to having them
    // as separate layers with connections. For now, we need to do both.
    auto weights3           = graph.AddLayer<ConstantLayer>("Weights");
    weights3->m_LayerOutput = std::make_unique<ScopedTensorHandle>(weightInfo);
    weights3->m_LayerOutput->Allocate();
    weights3->GetOutputSlot().SetTensorInfo(weightInfo);
    weights3->GetOutputSlot().Connect(conv3Layer->GetInputSlot(1));
    auto bias3           = graph.AddLayer<ConstantLayer>("Bias");
    bias3->m_LayerOutput = std::make_unique<ScopedTensorHandle>(biasInfo);
    bias3->m_LayerOutput->Allocate();
    bias3->GetOutputSlot().SetTensorInfo(biasInfo);
    bias3->GetOutputSlot().Connect(conv3Layer->GetInputSlot(2));

    SetWeightAndBias(conv3Layer, weightInfo, biasInfo);
    conv3Layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    Convolution2dLayer* const conv4Layer = graph.AddLayer<Convolution2dLayer>(convolutionDescriptor, "conv4 layer");
    CHECK(conv4Layer);

    // Arm NN is transitioning from having weights/bias as intrinsic properties of the layer to having them
    // as separate layers with connections. For now, we need to do both.
    auto weights4           = graph.AddLayer<ConstantLayer>("Weights");
    weights4->m_LayerOutput = std::make_unique<ScopedTensorHandle>(weightInfo);
    weights4->m_LayerOutput->Allocate();
    weights4->GetOutputSlot().SetTensorInfo(weightInfo);
    weights4->GetOutputSlot().Connect(conv4Layer->GetInputSlot(1));
    auto bias4           = graph.AddLayer<ConstantLayer>("Bias");
    bias4->m_LayerOutput = std::make_unique<ScopedTensorHandle>(biasInfo);
    bias4->m_LayerOutput->Allocate();
    bias4->GetOutputSlot().SetTensorInfo(biasInfo);
    bias4->GetOutputSlot().Connect(conv4Layer->GetInputSlot(2));

    SetWeightAndBias(conv4Layer, weightInfo, biasInfo);
    conv4Layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    Convolution2dLayer* const conv5Layer = graph.AddLayer<Convolution2dLayer>(convolutionDescriptor, "conv5 layer");
    CHECK(conv5Layer);

    // Arm NN is transitioning from having weights/bias as intrinsic properties of the layer to having them
    // as separate layers with connections. For now, we need to do both.
    auto weights5           = graph.AddLayer<ConstantLayer>("Weights");
    weights5->m_LayerOutput = std::make_unique<ScopedTensorHandle>(weightInfo);
    weights5->m_LayerOutput->Allocate();
    weights5->GetOutputSlot().SetTensorInfo(weightInfo);
    weights5->GetOutputSlot().Connect(conv5Layer->GetInputSlot(1));
    auto bias5           = graph.AddLayer<ConstantLayer>("Bias");
    bias5->m_LayerOutput = std::make_unique<ScopedTensorHandle>(biasInfo);
    bias5->m_LayerOutput->Allocate();
    bias5->GetOutputSlot().SetTensorInfo(biasInfo);
    bias5->GetOutputSlot().Connect(conv5Layer->GetInputSlot(2));

    SetWeightAndBias(conv5Layer, weightInfo, biasInfo);
    conv5Layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    Layer* const outputLayer = graph.AddLayer<OutputLayer>(0, "output layer");
    CHECK(outputLayer);

    // Connect the network
    inputLayer->GetOutputSlot(0).Connect(conv1Layer->GetInputSlot(0));
    conv1Layer->GetOutputSlot(0).Connect(conv2Layer->GetInputSlot(0));
    conv2Layer->GetOutputSlot(0).Connect(conv3Layer->GetInputSlot(0));
    conv3Layer->GetOutputSlot(0).Connect(conv4Layer->GetInputSlot(0));
    conv4Layer->GetOutputSlot(0).Connect(conv5Layer->GetInputSlot(0));
    conv5Layer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Create the subgraph view for the whole network
    armnn::SubgraphView view({ conv1Layer, weights, bias, conv2Layer, weights2, bias2, conv3Layer, weights3, bias3,
                               conv4Layer, weights4, bias4, conv5Layer, weights5, bias5 },
                             { &conv1Layer->GetInputSlot(0) }, { &conv5Layer->GetOutputSlot(0) });
    return std::make_unique<SubgraphView>(view);
}

// Creates a network with only one supported convolution layer,
// but using large tensors in order to force the compile step to fail
SubgraphView::SubgraphViewPtr BuildNonOptimizableSubgraph(Graph& graph)
{
    // Using very large tensors to force the subgraph compilation to fail on the Ethos-N backend
    const TensorInfo inputInfo({ 1, 16, 16, 10000 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo outputInfo({ 1, 16, 16, 10000 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo weightInfo({ 16, 1, 1, 10000 }, DataType::QAsymmU8, 0.9f, 0, true);
    const TensorInfo biasInfo({ 1, 1, 1, 10000 }, DataType::Signed32, 0.9f, 0, true);

    Convolution2dDescriptor convolutionDescriptor;
    convolutionDescriptor.m_StrideX     = 1;
    convolutionDescriptor.m_StrideY     = 1;
    convolutionDescriptor.m_BiasEnabled = true;
    convolutionDescriptor.m_DataLayout  = DataLayout::NHWC;

    // Construct the graph
    Layer* const inputLayer = graph.AddLayer<InputLayer>(0, "input layer");
    CHECK(inputLayer);
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    Convolution2dLayer* const convLayer = graph.AddLayer<Convolution2dLayer>(convolutionDescriptor, "conv layer");
    CHECK(convLayer);

    // Arm NN is transitioning from having weights/bias as intrinsic properties of the layer to having them
    // as separate layers with connections. For now, we need to do both.
    auto weights           = graph.AddLayer<ConstantLayer>("Weights");
    weights->m_LayerOutput = std::make_unique<ScopedTensorHandle>(weightInfo);
    weights->m_LayerOutput->Allocate();
    weights->GetOutputSlot().SetTensorInfo(weightInfo);
    weights->GetOutputSlot().Connect(convLayer->GetInputSlot(1));
    auto bias           = graph.AddLayer<ConstantLayer>("Bias");
    bias->m_LayerOutput = std::make_unique<ScopedTensorHandle>(biasInfo);
    bias->m_LayerOutput->Allocate();
    bias->GetOutputSlot().SetTensorInfo(biasInfo);
    bias->GetOutputSlot().Connect(convLayer->GetInputSlot(2));

    SetWeightAndBias(convLayer, weightInfo, biasInfo);
    convLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    Layer* const outputLayer = graph.AddLayer<OutputLayer>(0, "output layer");
    CHECK(outputLayer);

    // Connect the network
    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    convLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Create the subgraph view for the whole network
    armnn::SubgraphView view(
        {
            convLayer,
            weights,
            bias,
        },
        { &convLayer->GetInputSlot(0) }, { &convLayer->GetOutputSlot(0) });
    return std::make_unique<SubgraphView>(view);
}

// The input subgraph contains unsupported layers (the pooling layers have an unsupported configuration)
void UnsupportedSubgraphTestImpl()
{
    Graph graph;

    // Create an unsupported subgraph
    SubgraphView::SubgraphViewPtr subgraphPtr = BuildUnsupportedSubgraph(graph);
    CHECK((subgraphPtr != nullptr));

    const SubgraphView::IInputSlots& subgraphInputSlots    = subgraphPtr->GetIInputSlots();
    const SubgraphView::IOutputSlots& subgraphOutputSlots  = subgraphPtr->GetIOutputSlots();
    const SubgraphView::IConnectableLayers& subgraphLayers = subgraphPtr->GetIConnectableLayers();

    CHECK(subgraphInputSlots.size() == 1);
    CHECK(subgraphOutputSlots.size() == 1);
    CHECK(subgraphLayers.size() == 9);

    // Create a backend object
    auto backendObjPtr = CreateBackendObject(EthosNBackendId());
    CHECK((backendObjPtr != nullptr));

    // Optimize the subgraph
    OptimizationViews optimizationViews;

    // Check that the optimization is carried out correctly, but no optimization is performed
    CHECK_NOTHROW(optimizationViews = backendObjPtr->OptimizeSubgraphView(*subgraphPtr));

    // The expected results are:
    //  - No substitutions
    //  - Exactly one failed subgraph, identical to the given one
    //  - No untouched subgraphs

    CHECK(optimizationViews.GetSubstitutions().empty());
    CHECK(optimizationViews.GetUntouchedSubgraphs().empty());

    const OptimizationViews::Subgraphs& failedSubgraphs = optimizationViews.GetFailedSubgraphs();
    CHECK(failedSubgraphs.size() == 1);

    const SubgraphView& failedSubgraph                           = failedSubgraphs.at(0);
    const SubgraphView::IInputSlots& failedSubgraphInputSlots    = failedSubgraph.GetIInputSlots();
    const SubgraphView::IOutputSlots& failedSubgraphOutputSlots  = failedSubgraph.GetIOutputSlots();
    const SubgraphView::IConnectableLayers& failedSubgraphLayers = failedSubgraph.GetIConnectableLayers();

    CHECK(failedSubgraphInputSlots.size() == subgraphInputSlots.size());
    CHECK(failedSubgraphOutputSlots.size() == subgraphOutputSlots.size());
    CHECK(failedSubgraphLayers.size() == subgraphLayers.size());

    CHECK(failedSubgraphInputSlots == subgraphInputSlots);
    CHECK(failedSubgraphOutputSlots == subgraphOutputSlots);
    CHECK(failedSubgraphLayers == subgraphLayers);
}

// A simple case with only one layer (convolution) to optimize, supported by the Ethos-N backend
void FullyOptimizableSubgraphTestImpl1()
{
    Graph graph;

    // Create a fully optimizable subgraph
    SubgraphViewSelector::SubgraphViewPtr subgraphPtr = BuildFullyOptimizableSubgraph1(graph);
    CHECK((subgraphPtr != nullptr));

    const SubgraphView::IInputSlots& subgraphInputSlots    = subgraphPtr->GetIInputSlots();
    const SubgraphView::IOutputSlots& subgraphOutputSlots  = subgraphPtr->GetIOutputSlots();
    const SubgraphView::IConnectableLayers& subgraphLayers = subgraphPtr->GetIConnectableLayers();

    CHECK(subgraphPtr->GetIInputSlots().size() == 1);
    CHECK(subgraphPtr->GetIOutputSlots().size() == 1);
    CHECK(subgraphPtr->GetIConnectableLayers().size() == 3);

    // Create a backend object
    auto backendObjPtr = CreateBackendObject(EthosNBackendId());
    CHECK((backendObjPtr != nullptr));

    // Optimize the subgraph
    OptimizationViews optimizationViews;

    // Check that the optimization is carried out correctly
    CHECK_NOTHROW(optimizationViews = backendObjPtr->OptimizeSubgraphView(*subgraphPtr));

    // The expected results are:
    //  - Exactly one substitution, mapping the whole input subgraph to a new replacement subgraph
    //  - No failed subgraphs
    //  - No untouched subgraphs

    CHECK(optimizationViews.GetFailedSubgraphs().empty());
    CHECK(optimizationViews.GetUntouchedSubgraphs().empty());

    const OptimizationViews::Substitutions& substitutions = optimizationViews.GetSubstitutions();
    CHECK(substitutions.size() == 1);

    const OptimizationViews::SubstitutionPair& substitution = substitutions.at(0);

    const SubgraphView& substitutableSubgraph                           = substitution.m_SubstitutableSubgraph;
    const SubgraphView::IInputSlots& substitutableSubgraphInputSlots    = substitutableSubgraph.GetIInputSlots();
    const SubgraphView::IOutputSlots& substitutableSubgraphOutputSlots  = substitutableSubgraph.GetIOutputSlots();
    const SubgraphView::IConnectableLayers& substitutableSubgraphLayers = substitutableSubgraph.GetIConnectableLayers();

    const SubgraphView& replacementSubgraph                           = substitution.m_ReplacementSubgraph;
    const SubgraphView::IInputSlots& replacementSubgraphInputSlots    = replacementSubgraph.GetIInputSlots();
    const SubgraphView::IOutputSlots& replacementSubgraphOutputSlots  = replacementSubgraph.GetIOutputSlots();
    const SubgraphView::IConnectableLayers& replacementSubgraphLayers = replacementSubgraph.GetIConnectableLayers();

    CHECK(substitutableSubgraphInputSlots.size() == subgraphInputSlots.size());
    CHECK(substitutableSubgraphOutputSlots.size() == subgraphOutputSlots.size());
    CHECK(substitutableSubgraphLayers.size() == subgraphLayers.size());

    CHECK(substitutableSubgraphInputSlots == subgraphInputSlots);
    CHECK(substitutableSubgraphOutputSlots == subgraphOutputSlots);
    CHECK(substitutableSubgraphLayers == subgraphLayers);

    CHECK(replacementSubgraphInputSlots.size() == subgraphInputSlots.size());
    CHECK(replacementSubgraphOutputSlots.size() == subgraphOutputSlots.size());
    CHECK(replacementSubgraphLayers.size() == 1);

    CHECK(replacementSubgraphInputSlots != subgraphInputSlots);
    CHECK(replacementSubgraphOutputSlots != subgraphOutputSlots);
    CHECK(replacementSubgraphLayers != subgraphLayers);
    CHECK((replacementSubgraphLayers.front()->GetType() == LayerType::PreCompiled));
}

// A more complex case with five layers (all convolutions) to optimize, all supported by the Ethos-N backend
void FullyOptimizableSubgraphTestImpl2()
{
    Graph graph;

    // Create a fully optimizable subgraph
    SubgraphViewSelector::SubgraphViewPtr subgraphPtr = BuildFullyOptimizableSubgraph2(graph);
    CHECK((subgraphPtr != nullptr));

    const SubgraphView::IInputSlots& subgraphInputSlots    = subgraphPtr->GetIInputSlots();
    const SubgraphView::IOutputSlots& subgraphOutputSlots  = subgraphPtr->GetIOutputSlots();
    const SubgraphView::IConnectableLayers& subgraphLayers = subgraphPtr->GetIConnectableLayers();

    CHECK(subgraphPtr->GetIInputSlots().size() == 1);
    CHECK(subgraphPtr->GetIOutputSlots().size() == 1);
    CHECK(subgraphPtr->GetIConnectableLayers().size() == 15);

    // Create a backend object
    auto backendObjPtr = CreateBackendObject(EthosNBackendId());
    CHECK((backendObjPtr != nullptr));

    // Optimize the subgraph
    OptimizationViews optimizationViews;

    // Check that the optimization is carried out correctly
    CHECK_NOTHROW(optimizationViews = backendObjPtr->OptimizeSubgraphView(*subgraphPtr));

    // The expected results are:
    //  - Exactly one substitution, mapping the whole input subgraph to a new replacement subgraph
    //  - No failed subgraphs
    //  - No untouched subgraphs

    CHECK(optimizationViews.GetFailedSubgraphs().empty());
    CHECK(optimizationViews.GetUntouchedSubgraphs().empty());

    const OptimizationViews::Substitutions& substitutions = optimizationViews.GetSubstitutions();
    CHECK(substitutions.size() == 1);

    const OptimizationViews::SubstitutionPair& substitution = substitutions.at(0);

    const SubgraphView& substitutableSubgraph                           = substitution.m_SubstitutableSubgraph;
    const SubgraphView::IInputSlots& substitutableSubgraphInputSlots    = substitutableSubgraph.GetIInputSlots();
    const SubgraphView::IOutputSlots& substitutableSubgraphOutputSlots  = substitutableSubgraph.GetIOutputSlots();
    const SubgraphView::IConnectableLayers& substitutableSubgraphLayers = substitutableSubgraph.GetIConnectableLayers();

    const SubgraphView& replacementSubgraph                           = substitution.m_ReplacementSubgraph;
    const SubgraphView::IInputSlots& replacementSubgraphInputSlots    = replacementSubgraph.GetIInputSlots();
    const SubgraphView::IOutputSlots& replacementSubgraphOutputSlots  = replacementSubgraph.GetIOutputSlots();
    const SubgraphView::IConnectableLayers& replacementSubgraphLayers = replacementSubgraph.GetIConnectableLayers();

    CHECK(substitutableSubgraphInputSlots.size() == subgraphInputSlots.size());
    CHECK(substitutableSubgraphOutputSlots.size() == subgraphOutputSlots.size());
    CHECK(substitutableSubgraphLayers.size() == subgraphLayers.size());

    CHECK(substitutableSubgraphInputSlots == subgraphInputSlots);
    CHECK(substitutableSubgraphOutputSlots == subgraphOutputSlots);
    CHECK(substitutableSubgraphLayers == subgraphLayers);

    CHECK(replacementSubgraphInputSlots.size() == subgraphInputSlots.size());
    CHECK(replacementSubgraphOutputSlots.size() == subgraphOutputSlots.size());
    CHECK(replacementSubgraphLayers.size() == 1);

    CHECK(replacementSubgraphInputSlots != subgraphInputSlots);
    CHECK(replacementSubgraphOutputSlots != subgraphOutputSlots);
    CHECK(replacementSubgraphLayers != subgraphLayers);
    CHECK((replacementSubgraphLayers.front()->GetType() == LayerType::PreCompiled));
}

// A network with only one convolution layer is supported,
// but we use large tensors in order to force the compile step to fail
void NonOptimizableSubgraphTestImpl()
{
    Graph graph;

    // Create a non-optimizable subgraph
    SubgraphViewSelector::SubgraphViewPtr subgraphPtr = BuildNonOptimizableSubgraph(graph);
    CHECK((subgraphPtr != nullptr));

    const SubgraphView::IInputSlots& subgraphInputSlots    = subgraphPtr->GetIInputSlots();
    const SubgraphView::IOutputSlots& subgraphOutputSlots  = subgraphPtr->GetIOutputSlots();
    const SubgraphView::IConnectableLayers& subgraphLayers = subgraphPtr->GetIConnectableLayers();

    CHECK(subgraphPtr->GetIInputSlots().size() == 1);
    CHECK(subgraphPtr->GetIOutputSlots().size() == 1);
    CHECK(subgraphPtr->GetIConnectableLayers().size() == 3);

    // Create a backend object
    auto backendObjPtr = CreateBackendObject(EthosNBackendId());
    CHECK((backendObjPtr != nullptr));

    // Optimize the subgraph
    OptimizationViews optimizationViews;

    // Check that the optimization is carried out correctly
    CHECK_NOTHROW(optimizationViews = backendObjPtr->OptimizeSubgraphView(*subgraphPtr));

    // The expected results are:
    //  - No substitutions
    //  - Exactly one failed subgraph, corresponding to the whole input subgraph
    //  - No untouched subgraphs

    CHECK(optimizationViews.GetSubstitutions().empty());
    CHECK(optimizationViews.GetUntouchedSubgraphs().empty());

    const OptimizationViews::Subgraphs& failedSubgraphs = optimizationViews.GetFailedSubgraphs();
    CHECK(failedSubgraphs.size() == 1);

    const SubgraphView& failedSubgraph                           = failedSubgraphs.at(0);
    const SubgraphView::IInputSlots& failedSubgraphInputSlots    = failedSubgraph.GetIInputSlots();
    const SubgraphView::IOutputSlots& failedSubgraphOutputSlots  = failedSubgraph.GetIOutputSlots();
    const SubgraphView::IConnectableLayers& failedSubgraphLayers = failedSubgraph.GetIConnectableLayers();

    CHECK(failedSubgraphInputSlots.size() == subgraphInputSlots.size());
    CHECK(failedSubgraphOutputSlots.size() == subgraphOutputSlots.size());
    CHECK(failedSubgraphLayers.size() == subgraphLayers.size());

    CHECK(failedSubgraphInputSlots == subgraphInputSlots);
    CHECK(failedSubgraphOutputSlots == subgraphOutputSlots);
    CHECK(failedSubgraphLayers == subgraphLayers);
}

}    // Anonymous namespace

TEST_SUITE("EthosNOptimizeSubGraph")
{

    TEST_CASE("UnsupportedSubgraph")
    {
        UnsupportedSubgraphTestImpl();
    }
    TEST_CASE("FullyOptimizableSubgraph1")
    {
        FullyOptimizableSubgraphTestImpl1();
    }
    TEST_CASE("FullyOptimizableSubgraph2")
    {
        FullyOptimizableSubgraphTestImpl2();
    }
    TEST_CASE("NonOptimizableSubgraph")
    {
        NonOptimizableSubgraphTestImpl();
    }

    /// Checks that GetCompilationOptions correctly handles user-provided ModelOptions.
    TEST_CASE("TestGetCompilationOptions")
    {
        EthosNConfig config;

        // Default (winograd enabled and strictPrecision disabled)
        CHECK(GetCompilationOptions(config, {}, 0).m_DisableWinograd == false);
        CHECK(GetCompilationOptions(config, {}, 0).m_StrictPrecision == false);

        // Disable winograd and enabled strictPrecision explicitly
        BackendOptions backendOptions(EthosNBackend::GetIdStatic(),
                                      { { "DisableWinograd", true }, { "StrictPrecision", true } });
        CHECK(GetCompilationOptions(config, { backendOptions }, 0).m_DisableWinograd == true);
        CHECK(GetCompilationOptions(config, { backendOptions }, 0).m_StrictPrecision == true);

        // Other backend options are ignored
        BackendOptions optOtherBackend("OtherBackend", { { "DisableWinograd", true }, { "StrictPrecision", true } });
        CHECK(GetCompilationOptions(config, { optOtherBackend }, 0).m_DisableWinograd == false);
        CHECK(GetCompilationOptions(config, { optOtherBackend }, 0).m_StrictPrecision == false);

        // Invalid option (unknown name)
        BackendOptions optInvalidName(EthosNBackend::GetIdStatic(), { { "TestInvalidOption", true } });
        CHECK_THROWS_AS(GetCompilationOptions(config, { optInvalidName }, 0), InvalidArgumentException);

        // Invalid option (wrong option type)
        BackendOptions optInvalidTypeWinograd(EthosNBackend::GetIdStatic(), { { "DisableWinograd", "hello" } });
        BackendOptions optInvalidTypeStrictPrecision(EthosNBackend::GetIdStatic(), { { "StrictPrecision", "test" } });
        CHECK_THROWS_AS(GetCompilationOptions(config, { optInvalidTypeWinograd }, 0), InvalidArgumentException);
        CHECK_THROWS_AS(GetCompilationOptions(config, { optInvalidTypeStrictPrecision }, 0), InvalidArgumentException);
    }

    /// Checks that the m_DisableWinograd option is correctly passed through to the support library.
    TEST_CASE("TestDisableWinograd")
    {
        // Set up mock support library, which records the m_DisableWinograd option
        class MockSupportLibrary : public EthosNSupportLibraryInterface
        {
        public:
            std::vector<std::unique_ptr<ethosn_lib::CompiledNetwork>>
                Compile(const ethosn_lib::Network&, const ethosn_lib::CompilationOptions& options) final
            {
                m_RecordedDisableWinograd.push_back(options.m_DisableWinograd);
                return {};
            }

            std::vector<bool> m_RecordedDisableWinograd;
        };
        g_EthosNSupportLibraryInterface        = std::make_unique<MockSupportLibrary>();
        MockSupportLibrary& mockSupportLibrary = static_cast<MockSupportLibrary&>(*g_EthosNSupportLibraryInterface);

        // Make an arbitrary network
        armnn::INetworkPtr net = armnn::INetwork::Create();
        TensorInfo inputInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0);
        TensorInfo outputInfo({ 1, 4, 4, 1 }, DataType::QAsymmU8, 1.0f, 0);

        IConnectableLayer* const inputLayer = net->AddInputLayer(0, "input");
        inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

        DepthToSpaceDescriptor desc(2, DataLayout::NHWC);
        IConnectableLayer* const spaceToDepthLayer = net->AddDepthToSpaceLayer(desc, "depthToSpace");
        spaceToDepthLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);
        inputLayer->GetOutputSlot(0).Connect(spaceToDepthLayer->GetInputSlot(0));

        IConnectableLayer* const outputLayer = net->AddOutputLayer(0, "output");
        spaceToDepthLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

        // Optimize for EthosNAcc with default options. This is expected to throw due to mock support library.
        std::vector<BackendId> backends = { EthosNBackendId() };
        IRuntimePtr runtime(IRuntime::Create(IRuntime::CreationOptions()));
        OptimizerOptions optOpts;
        CHECK_THROWS_AS(Optimize(*net, backends, runtime->GetDeviceSpec(), optOpts), armnn::InvalidArgumentException);

        // Check that support library was called correctly
        CHECK(mockSupportLibrary.m_RecordedDisableWinograd.back() == false);

        // Optimize for EthosNAcc (disable Winograd)
        optOpts.m_ModelOptions = { BackendOptions(EthosNBackend::GetIdStatic(), { { "DisableWinograd", true } }) };
        CHECK_THROWS_AS(Optimize(*net, backends, runtime->GetDeviceSpec(), optOpts), armnn::InvalidArgumentException);

        // Check that support library was called correctly
        CHECK(mockSupportLibrary.m_RecordedDisableWinograd.back() == true);
    }

    /// Checks that the m_StrictPrecision option is correctly passed through to the support library.
    TEST_CASE("TestStrictPrecision")
    {
        // Set up mock support library, which records the m_DisableWinograd option
        class MockSupportLibrary : public EthosNSupportLibraryInterface
        {
        public:
            std::vector<std::unique_ptr<ethosn_lib::CompiledNetwork>>
                Compile(const ethosn_lib::Network&, const ethosn_lib::CompilationOptions& options) final
            {
                m_RecordedStrictPrecision.push_back(options.m_StrictPrecision);
                return {};
            }

            std::vector<bool> m_RecordedStrictPrecision;
        };
        g_EthosNSupportLibraryInterface        = std::make_unique<MockSupportLibrary>();
        MockSupportLibrary& mockSupportLibrary = static_cast<MockSupportLibrary&>(*g_EthosNSupportLibraryInterface);

        // Set up tensor infos
        TensorInfo inputInfo({ 1, 8, 8, 16 }, DataType::QAsymmU8, 1.0f, 0);
        TensorInfo intermediateInfo({ 1, 8, 8, 16 }, DataType::QAsymmU8, 1.0f, 0);
        TensorInfo outputInfo({ 1, 8, 8, 32 }, DataType::QAsymmU8, 1.0f, 0);

        ActivationDescriptor reluDesc;
        reluDesc.m_Function = ActivationFunction::BoundedReLu;
        reluDesc.m_A        = 255.0f;
        reluDesc.m_B        = 0.0f;

        // Construct network
        armnn::INetworkPtr net               = armnn::INetwork::Create();
        IConnectableLayer* const input0Layer = net->AddInputLayer(0, "input0");
        input0Layer->GetOutputSlot(0).SetTensorInfo(inputInfo);
        IConnectableLayer* const relu0Layer = net->AddActivationLayer(reluDesc, "relu0");
        relu0Layer->GetOutputSlot(0).SetTensorInfo(intermediateInfo);
        input0Layer->GetOutputSlot(0).Connect(relu0Layer->GetInputSlot(0));

        IConnectableLayer* const input1Layer = net->AddInputLayer(1, "input1");
        input1Layer->GetOutputSlot(0).SetTensorInfo(inputInfo);
        IConnectableLayer* const relu1Layer = net->AddActivationLayer(reluDesc, "relu1");
        relu1Layer->GetOutputSlot(0).SetTensorInfo(intermediateInfo);
        input1Layer->GetOutputSlot(0).Connect(relu1Layer->GetInputSlot(0));

        std::array<TensorShape, 2> concatInputShapes = { intermediateInfo.GetShape(), intermediateInfo.GetShape() };
        IConnectableLayer* const concatLayer         = net->AddConcatLayer(
            CreateDescriptorForConcatenation(concatInputShapes.begin(), concatInputShapes.end(), 3), "concat");
        concatLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);
        relu0Layer->GetOutputSlot(0).Connect(concatLayer->GetInputSlot(0));
        relu1Layer->GetOutputSlot(0).Connect(concatLayer->GetInputSlot(1));

        IConnectableLayer* const outputLayer = net->AddOutputLayer(0, "output");
        concatLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

        // Optimize for EthosNAcc with default options. This is expected to throw due to mock support library.
        std::vector<BackendId> backends = { EthosNBackendId() };
        IRuntimePtr runtime(IRuntime::Create(IRuntime::CreationOptions()));
        OptimizerOptions optOpts;
        CHECK_THROWS_AS(Optimize(*net, backends, runtime->GetDeviceSpec(), optOpts), armnn::InvalidArgumentException);

        // Check that support library was called correctly
        CHECK(mockSupportLibrary.m_RecordedStrictPrecision.back() == false);

        // Optimize for EthosNAcc (enable StrictPrecision)
        optOpts.m_ModelOptions = { BackendOptions(EthosNBackend::GetIdStatic(), { { "StrictPrecision", true } }) };
        CHECK_THROWS_AS(Optimize(*net, backends, runtime->GetDeviceSpec(), optOpts), armnn::InvalidArgumentException);

        // Check that support library was called correctly
        CHECK(mockSupportLibrary.m_RecordedStrictPrecision.back() == true);
    }
}
