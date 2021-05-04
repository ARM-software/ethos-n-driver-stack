//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include <EthosNBackend.hpp>
#include <EthosNBackendId.hpp>
#include <EthosNSubgraphViewConverter.hpp>
#include <Graph.hpp>
#include <Network.hpp>
#include <armnn/BackendRegistry.hpp>
#include <backendsCommon/test/CommonTestUtils.hpp>
#include <boost/test/unit_test.hpp>

using namespace armnn;

namespace
{

// Creates a subgraph containing unsupported layers (the pooling layers have an unsupported configuration)
SubgraphView::SubgraphViewPtr BuildUnsupportedSubgraph(Graph& graph)
{
    const TensorInfo inputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo outputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo weightInfo({ 16, 1, 1, 16 }, DataType::QAsymmU8, 0.9f, 0);
    const TensorInfo biasInfo({ 1, 1, 1, 16 }, DataType::Signed32, 0.9f, 0);

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
    BOOST_TEST(inputLayer);
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    Convolution2dLayer* const conv1Layer = graph.AddLayer<Convolution2dLayer>(convolutionDescriptor, "conv1 layer");
    BOOST_TEST(conv1Layer);
    SetWeightAndBias(conv1Layer, weightInfo, biasInfo);
    conv1Layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    Pooling2dLayer* const pooling1Layer = graph.AddLayer<Pooling2dLayer>(poolingDescriptor, "pooling1 layer");
    BOOST_TEST(pooling1Layer);
    pooling1Layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    Pooling2dLayer* const pooling2Layer = graph.AddLayer<Pooling2dLayer>(poolingDescriptor, "pooling2 layer");
    BOOST_TEST(pooling2Layer);
    pooling2Layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    Convolution2dLayer* const conv2Layer = graph.AddLayer<Convolution2dLayer>(convolutionDescriptor, "conv2 layer");
    BOOST_TEST(conv2Layer);
    SetWeightAndBias(conv2Layer, weightInfo, biasInfo);
    conv2Layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    Pooling2dLayer* const pooling3Layer = graph.AddLayer<Pooling2dLayer>(poolingDescriptor, "pooling3 layer");
    BOOST_TEST(pooling3Layer);
    pooling3Layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    Layer* const outputLayer = graph.AddLayer<OutputLayer>(0, "output layer");
    BOOST_TEST(outputLayer);

    // Connect the network
    inputLayer->GetOutputSlot(0).Connect(conv1Layer->GetInputSlot(0));
    conv1Layer->GetOutputSlot(0).Connect(pooling1Layer->GetInputSlot(0));
    pooling1Layer->GetOutputSlot(0).Connect(pooling2Layer->GetInputSlot(0));
    pooling2Layer->GetOutputSlot(0).Connect(conv2Layer->GetInputSlot(0));
    conv2Layer->GetOutputSlot(0).Connect(pooling3Layer->GetInputSlot(0));
    pooling3Layer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Create the subgraph view for the whole network
    return CreateSubgraphViewFrom(CreateInputsFrom({ conv1Layer }), CreateOutputsFrom({ pooling3Layer }),
                                  { conv1Layer, pooling1Layer, pooling2Layer, conv2Layer, pooling3Layer });
}

// Creates a simple subgraph with only one convolution layer, supported by the Ethos-N backend
SubgraphView::SubgraphViewPtr BuildFullyOptimizableSubgraph1(Graph& graph)
{
    const TensorInfo inputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo outputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo weightInfo({ 16, 1, 1, 16 }, DataType::QAsymmU8, 0.9f, 0);
    const TensorInfo biasInfo({ 1, 1, 1, 16 }, DataType::Signed32, 0.9f, 0);

    Convolution2dDescriptor convolutionDescriptor;
    convolutionDescriptor.m_StrideX     = 1;
    convolutionDescriptor.m_StrideY     = 1;
    convolutionDescriptor.m_BiasEnabled = true;
    convolutionDescriptor.m_DataLayout  = DataLayout::NHWC;

    // Construct the graph
    Layer* const inputLayer = graph.AddLayer<InputLayer>(0, "input layer");
    BOOST_TEST(inputLayer);
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    Convolution2dLayer* const convLayer = graph.AddLayer<Convolution2dLayer>(convolutionDescriptor, "conv layer");
    BOOST_TEST(convLayer);
    SetWeightAndBias(convLayer, weightInfo, biasInfo);
    convLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    Layer* const outputLayer = graph.AddLayer<OutputLayer>(0, "output layer");
    BOOST_TEST(outputLayer);

    // Connect the network
    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    convLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Create the subgraph view for the whole network
    return CreateSubgraphViewFrom(CreateInputsFrom({ convLayer }), CreateOutputsFrom({ convLayer }), { convLayer });
}

// Creates a more complex subgraph with five convolutions layers, all supported by the Ethos-N backend
SubgraphView::SubgraphViewPtr BuildFullyOptimizableSubgraph2(Graph& graph)
{
    const TensorInfo inputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo outputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo weightInfo({ 16, 1, 1, 16 }, DataType::QAsymmU8, 0.9f, 0);
    const TensorInfo biasInfo({ 1, 1, 1, 16 }, DataType::Signed32, 0.9f, 0);

    Convolution2dDescriptor convolutionDescriptor;
    convolutionDescriptor.m_StrideX     = 1;
    convolutionDescriptor.m_StrideY     = 1;
    convolutionDescriptor.m_BiasEnabled = true;
    convolutionDescriptor.m_DataLayout  = DataLayout::NHWC;

    // Construct the graph
    Layer* const inputLayer = graph.AddLayer<InputLayer>(0, "input layer");
    BOOST_TEST(inputLayer);
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    Convolution2dLayer* const conv1Layer = graph.AddLayer<Convolution2dLayer>(convolutionDescriptor, "conv1 layer");
    BOOST_TEST(conv1Layer);
    SetWeightAndBias(conv1Layer, weightInfo, biasInfo);
    conv1Layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    Convolution2dLayer* const conv2Layer = graph.AddLayer<Convolution2dLayer>(convolutionDescriptor, "conv2 layer");
    BOOST_TEST(conv2Layer);
    SetWeightAndBias(conv2Layer, weightInfo, biasInfo);
    conv2Layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    Convolution2dLayer* const conv3Layer = graph.AddLayer<Convolution2dLayer>(convolutionDescriptor, "conv3 layer");
    BOOST_TEST(conv3Layer);
    SetWeightAndBias(conv3Layer, weightInfo, biasInfo);
    conv3Layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    Convolution2dLayer* const conv4Layer = graph.AddLayer<Convolution2dLayer>(convolutionDescriptor, "conv4 layer");
    BOOST_TEST(conv4Layer);
    SetWeightAndBias(conv4Layer, weightInfo, biasInfo);
    conv4Layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    Convolution2dLayer* const conv5Layer = graph.AddLayer<Convolution2dLayer>(convolutionDescriptor, "conv5 layer");
    BOOST_TEST(conv5Layer);
    SetWeightAndBias(conv5Layer, weightInfo, biasInfo);
    conv5Layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    Layer* const outputLayer = graph.AddLayer<OutputLayer>(0, "output layer");
    BOOST_TEST(outputLayer);

    // Connect the network
    inputLayer->GetOutputSlot(0).Connect(conv1Layer->GetInputSlot(0));
    conv1Layer->GetOutputSlot(0).Connect(conv2Layer->GetInputSlot(0));
    conv2Layer->GetOutputSlot(0).Connect(conv3Layer->GetInputSlot(0));
    conv3Layer->GetOutputSlot(0).Connect(conv4Layer->GetInputSlot(0));
    conv4Layer->GetOutputSlot(0).Connect(conv5Layer->GetInputSlot(0));
    conv5Layer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Create the subgraph view for the whole network
    return CreateSubgraphViewFrom(CreateInputsFrom({ conv1Layer }), CreateOutputsFrom({ conv5Layer }),
                                  { conv1Layer, conv2Layer, conv3Layer, conv4Layer, conv5Layer });
}

// Creates a network with only one supported convolution layer,
// but using large tensors in order to force the compile step to fail
SubgraphView::SubgraphViewPtr BuildNonOptimizableSubgraph(Graph& graph)
{
    // Using very large tensors to force the subgraph compilation to fail on the Ethos-N backend
    const TensorInfo inputInfo({ 1, 16, 16, 10000 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo outputInfo({ 1, 16, 16, 10000 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo weightInfo({ 16, 1, 1, 10000 }, DataType::QAsymmU8, 0.9f, 0);
    const TensorInfo biasInfo({ 1, 1, 1, 10000 }, DataType::Signed32, 0.9f, 0);

    Convolution2dDescriptor convolutionDescriptor;
    convolutionDescriptor.m_StrideX     = 1;
    convolutionDescriptor.m_StrideY     = 1;
    convolutionDescriptor.m_BiasEnabled = true;
    convolutionDescriptor.m_DataLayout  = DataLayout::NHWC;

    // Construct the graph
    Layer* const inputLayer = graph.AddLayer<InputLayer>(0, "input layer");
    BOOST_TEST(inputLayer);
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    Convolution2dLayer* const convLayer = graph.AddLayer<Convolution2dLayer>(convolutionDescriptor, "conv layer");
    BOOST_TEST(convLayer);
    SetWeightAndBias(convLayer, weightInfo, biasInfo);
    convLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    Layer* const outputLayer = graph.AddLayer<OutputLayer>(0, "output layer");
    BOOST_TEST(outputLayer);

    // Connect the network
    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    convLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Create the subgraph view for the whole network
    return CreateSubgraphViewFrom(CreateInputsFrom({ convLayer }), CreateOutputsFrom({ convLayer }), { convLayer });
}

// The input subgraph contains unsupported layers (the pooling layers have an unsupported configuration)
void UnsupportedSubgraphTestImpl()
{
    Graph graph;

    // Create an unsupported subgraph
    SubgraphView::SubgraphViewPtr subgraphPtr = BuildUnsupportedSubgraph(graph);
    BOOST_TEST((subgraphPtr != nullptr));

    const SubgraphView::InputSlots& subgraphInputSlots   = subgraphPtr->GetInputSlots();
    const SubgraphView::OutputSlots& subgraphOutputSlots = subgraphPtr->GetOutputSlots();
    const SubgraphView::Layers& subgraphLayers           = subgraphPtr->GetLayers();

    BOOST_TEST(subgraphInputSlots.size() == 1);
    BOOST_TEST(subgraphOutputSlots.size() == 1);
    BOOST_TEST(subgraphLayers.size() == 5);

    // Create a backend object
    auto backendObjPtr = CreateBackendObject(EthosNBackendId());
    BOOST_TEST((backendObjPtr != nullptr));

    // Optimize the subgraph
    OptimizationViews optimizationViews;

    // Check that the optimization is carried out correctly, but no optimization is performed
    BOOST_CHECK_NO_THROW(optimizationViews = backendObjPtr->OptimizeSubgraphView(*subgraphPtr));

    // The expected results are:
    //  - No substitutions
    //  - Exactly one failed subgraph, identical to the given one
    //  - No untouched subgraphs

    BOOST_TEST(optimizationViews.GetSubstitutions().empty());
    BOOST_TEST(optimizationViews.GetUntouchedSubgraphs().empty());

    const OptimizationViews::Subgraphs& failedSubgraphs = optimizationViews.GetFailedSubgraphs();
    BOOST_TEST(failedSubgraphs.size() == 1);

    const SubgraphView& failedSubgraph                         = failedSubgraphs.at(0);
    const SubgraphView::InputSlots& failedSubgraphInputSlots   = failedSubgraph.GetInputSlots();
    const SubgraphView::OutputSlots& failedSubgraphOutputSlots = failedSubgraph.GetOutputSlots();
    const SubgraphView::Layers& failedSubgraphLayers           = failedSubgraph.GetLayers();

    BOOST_TEST(failedSubgraphInputSlots.size() == subgraphInputSlots.size());
    BOOST_TEST(failedSubgraphOutputSlots.size() == subgraphOutputSlots.size());
    BOOST_TEST(failedSubgraphLayers.size() == subgraphLayers.size());

    BOOST_TEST(failedSubgraphInputSlots == subgraphInputSlots);
    BOOST_TEST(failedSubgraphOutputSlots == subgraphOutputSlots);
    BOOST_TEST(failedSubgraphLayers == subgraphLayers);
}

// A simple case with only one layer (convolution) to optimize, supported by the Ethos-N backend
void FullyOptimizableSubgraphTestImpl1()
{
    Graph graph;

    // Create a fully optimizable subgraph
    SubgraphViewSelector::SubgraphViewPtr subgraphPtr = BuildFullyOptimizableSubgraph1(graph);
    BOOST_TEST((subgraphPtr != nullptr));

    const SubgraphView::InputSlots& subgraphInputSlots   = subgraphPtr->GetInputSlots();
    const SubgraphView::OutputSlots& subgraphOutputSlots = subgraphPtr->GetOutputSlots();
    const SubgraphView::Layers& subgraphLayers           = subgraphPtr->GetLayers();

    BOOST_TEST(subgraphPtr->GetInputSlots().size() == 1);
    BOOST_TEST(subgraphPtr->GetOutputSlots().size() == 1);
    BOOST_TEST(subgraphPtr->GetLayers().size() == 1);

    // Create a backend object
    auto backendObjPtr = CreateBackendObject(EthosNBackendId());
    BOOST_TEST((backendObjPtr != nullptr));

    // Optimize the subgraph
    OptimizationViews optimizationViews;

    // Check that the optimization is carried out correctly
    BOOST_CHECK_NO_THROW(optimizationViews = backendObjPtr->OptimizeSubgraphView(*subgraphPtr));

    // The expected results are:
    //  - Exactly one substitution, mapping the whole input subgraph to a new replacement subgraph
    //  - No failed subgraphs
    //  - No untouched subgraphs

    BOOST_TEST(optimizationViews.GetFailedSubgraphs().empty());
    BOOST_TEST(optimizationViews.GetUntouchedSubgraphs().empty());

    const OptimizationViews::Substitutions& substitutions = optimizationViews.GetSubstitutions();
    BOOST_TEST(substitutions.size() == 1);

    const OptimizationViews::SubstitutionPair& substitution = substitutions.at(0);

    const SubgraphView& substitutableSubgraph                         = substitution.m_SubstitutableSubgraph;
    const SubgraphView::InputSlots& substitutableSubgraphInputSlots   = substitutableSubgraph.GetInputSlots();
    const SubgraphView::OutputSlots& substitutableSubgraphOutputSlots = substitutableSubgraph.GetOutputSlots();
    const SubgraphView::Layers& substitutableSubgraphLayers           = substitutableSubgraph.GetLayers();

    const SubgraphView& replacementSubgraph                         = substitution.m_ReplacementSubgraph;
    const SubgraphView::InputSlots& replacementSubgraphInputSlots   = replacementSubgraph.GetInputSlots();
    const SubgraphView::OutputSlots& replacementSubgraphOutputSlots = replacementSubgraph.GetOutputSlots();
    const SubgraphView::Layers& replacementSubgraphLayers           = replacementSubgraph.GetLayers();

    BOOST_TEST(substitutableSubgraphInputSlots.size() == subgraphInputSlots.size());
    BOOST_TEST(substitutableSubgraphOutputSlots.size() == subgraphOutputSlots.size());
    BOOST_TEST(substitutableSubgraphLayers.size() == subgraphLayers.size());

    BOOST_TEST(substitutableSubgraphInputSlots == subgraphInputSlots);
    BOOST_TEST(substitutableSubgraphOutputSlots == subgraphOutputSlots);
    BOOST_TEST(substitutableSubgraphLayers == subgraphLayers);

    BOOST_TEST(replacementSubgraphInputSlots.size() == subgraphInputSlots.size());
    BOOST_TEST(replacementSubgraphOutputSlots.size() == subgraphOutputSlots.size());
    BOOST_TEST(replacementSubgraphLayers.size() == 1);

    BOOST_TEST(replacementSubgraphInputSlots != subgraphInputSlots);
    BOOST_TEST(replacementSubgraphOutputSlots != subgraphOutputSlots);
    BOOST_TEST(replacementSubgraphLayers != subgraphLayers);
    BOOST_TEST((replacementSubgraphLayers.front()->GetType() == LayerType::PreCompiled));
}

// A more complex case with five layers (all convolutions) to optimize, all supported by the Ethos-N backend
void FullyOptimizableSubgraphTestImpl2()
{
    Graph graph;

    // Create a fully optimizable subgraph
    SubgraphViewSelector::SubgraphViewPtr subgraphPtr = BuildFullyOptimizableSubgraph2(graph);
    BOOST_TEST((subgraphPtr != nullptr));

    const SubgraphView::InputSlots& subgraphInputSlots   = subgraphPtr->GetInputSlots();
    const SubgraphView::OutputSlots& subgraphOutputSlots = subgraphPtr->GetOutputSlots();
    const SubgraphView::Layers& subgraphLayers           = subgraphPtr->GetLayers();

    BOOST_TEST(subgraphPtr->GetInputSlots().size() == 1);
    BOOST_TEST(subgraphPtr->GetOutputSlots().size() == 1);
    BOOST_TEST(subgraphPtr->GetLayers().size() == 5);

    // Create a backend object
    auto backendObjPtr = CreateBackendObject(EthosNBackendId());
    BOOST_TEST((backendObjPtr != nullptr));

    // Optimize the subgraph
    OptimizationViews optimizationViews;

    // Check that the optimization is carried out correctly
    BOOST_CHECK_NO_THROW(optimizationViews = backendObjPtr->OptimizeSubgraphView(*subgraphPtr));

    // The expected results are:
    //  - Exactly one substitution, mapping the whole input subgraph to a new replacement subgraph
    //  - No failed subgraphs
    //  - No untouched subgraphs

    BOOST_TEST(optimizationViews.GetFailedSubgraphs().empty());
    BOOST_TEST(optimizationViews.GetUntouchedSubgraphs().empty());

    const OptimizationViews::Substitutions& substitutions = optimizationViews.GetSubstitutions();
    BOOST_TEST(substitutions.size() == 1);

    const OptimizationViews::SubstitutionPair& substitution = substitutions.at(0);

    const SubgraphView& substitutableSubgraph                         = substitution.m_SubstitutableSubgraph;
    const SubgraphView::InputSlots& substitutableSubgraphInputSlots   = substitutableSubgraph.GetInputSlots();
    const SubgraphView::OutputSlots& substitutableSubgraphOutputSlots = substitutableSubgraph.GetOutputSlots();
    const SubgraphView::Layers& substitutableSubgraphLayers           = substitutableSubgraph.GetLayers();

    const SubgraphView& replacementSubgraph                         = substitution.m_ReplacementSubgraph;
    const SubgraphView::InputSlots& replacementSubgraphInputSlots   = replacementSubgraph.GetInputSlots();
    const SubgraphView::OutputSlots& replacementSubgraphOutputSlots = replacementSubgraph.GetOutputSlots();
    const SubgraphView::Layers& replacementSubgraphLayers           = replacementSubgraph.GetLayers();

    BOOST_TEST(substitutableSubgraphInputSlots.size() == subgraphInputSlots.size());
    BOOST_TEST(substitutableSubgraphOutputSlots.size() == subgraphOutputSlots.size());
    BOOST_TEST(substitutableSubgraphLayers.size() == subgraphLayers.size());

    BOOST_TEST(substitutableSubgraphInputSlots == subgraphInputSlots);
    BOOST_TEST(substitutableSubgraphOutputSlots == subgraphOutputSlots);
    BOOST_TEST(substitutableSubgraphLayers == subgraphLayers);

    BOOST_TEST(replacementSubgraphInputSlots.size() == subgraphInputSlots.size());
    BOOST_TEST(replacementSubgraphOutputSlots.size() == subgraphOutputSlots.size());
    BOOST_TEST(replacementSubgraphLayers.size() == 1);

    BOOST_TEST(replacementSubgraphInputSlots != subgraphInputSlots);
    BOOST_TEST(replacementSubgraphOutputSlots != subgraphOutputSlots);
    BOOST_TEST(replacementSubgraphLayers != subgraphLayers);
    BOOST_TEST((replacementSubgraphLayers.front()->GetType() == LayerType::PreCompiled));
}

// A network with only one convolution layer is supported,
// but we use large tensors in order to force the compile step to fail
void NonOptimizableSubgraphTestImpl()
{
    Graph graph;

    // Create a non-optimizable subgraph
    SubgraphViewSelector::SubgraphViewPtr subgraphPtr = BuildNonOptimizableSubgraph(graph);
    BOOST_TEST((subgraphPtr != nullptr));

    const SubgraphView::InputSlots& subgraphInputSlots   = subgraphPtr->GetInputSlots();
    const SubgraphView::OutputSlots& subgraphOutputSlots = subgraphPtr->GetOutputSlots();
    const SubgraphView::Layers& subgraphLayers           = subgraphPtr->GetLayers();

    BOOST_TEST(subgraphPtr->GetInputSlots().size() == 1);
    BOOST_TEST(subgraphPtr->GetOutputSlots().size() == 1);
    BOOST_TEST(subgraphPtr->GetLayers().size() == 1);

    // Create a backend object
    auto backendObjPtr = CreateBackendObject(EthosNBackendId());
    BOOST_TEST((backendObjPtr != nullptr));

    // Optimize the subgraph
    OptimizationViews optimizationViews;

    // Check that the optimization is carried out correctly
    BOOST_CHECK_NO_THROW(optimizationViews = backendObjPtr->OptimizeSubgraphView(*subgraphPtr));

    // The expected results are:
    //  - No substitutions
    //  - Exactly one failed subgraph, corresponding to the whole input subgraph
    //  - No untouched subgraphs

    BOOST_TEST(optimizationViews.GetSubstitutions().empty());
    BOOST_TEST(optimizationViews.GetUntouchedSubgraphs().empty());

    const OptimizationViews::Subgraphs& failedSubgraphs = optimizationViews.GetFailedSubgraphs();
    BOOST_TEST(failedSubgraphs.size() == 1);

    const SubgraphView& failedSubgraph                         = failedSubgraphs.at(0);
    const SubgraphView::InputSlots& failedSubgraphInputSlots   = failedSubgraph.GetInputSlots();
    const SubgraphView::OutputSlots& failedSubgraphOutputSlots = failedSubgraph.GetOutputSlots();
    const SubgraphView::Layers& failedSubgraphLayers           = failedSubgraph.GetLayers();

    BOOST_TEST(failedSubgraphInputSlots.size() == subgraphInputSlots.size());
    BOOST_TEST(failedSubgraphOutputSlots.size() == subgraphOutputSlots.size());
    BOOST_TEST(failedSubgraphLayers.size() == subgraphLayers.size());

    BOOST_TEST(failedSubgraphInputSlots == subgraphInputSlots);
    BOOST_TEST(failedSubgraphOutputSlots == subgraphOutputSlots);
    BOOST_TEST(failedSubgraphLayers == subgraphLayers);
}

}    // Anonymous namespace

BOOST_AUTO_TEST_SUITE(EthosNOptimizeSubGraph)

BOOST_AUTO_TEST_CASE(UnsupportedSubgraph)
{
    UnsupportedSubgraphTestImpl();
}
BOOST_AUTO_TEST_CASE(FullyOptimizableSubgraph1)
{
    FullyOptimizableSubgraphTestImpl1();
}
BOOST_AUTO_TEST_CASE(FullyOptimizableSubgraph2)
{
    FullyOptimizableSubgraphTestImpl2();
}
BOOST_AUTO_TEST_CASE(NonOptimizableSubgraph)
{
    NonOptimizableSubgraphTestImpl();
}

/// Checks that GetCompilationOptions correctly handles user-provided ModelOptions.
BOOST_AUTO_TEST_CASE(TestGetCompilationOptions)
{
    EthosNConfig config;

    // Default (winograd enabled)
    BOOST_TEST(GetCompilationOptions(config, {}, 0).m_DisableWinograd == false);

    // Disable winograd explicitly
    BackendOptions optDisableWinograd(EthosNBackend::GetIdStatic(), { { "DisableWinograd", true } });
    BOOST_TEST(GetCompilationOptions(config, { optDisableWinograd }, 0).m_DisableWinograd == true);

    // Other backend options are ignored
    BackendOptions optOtherBackend("OtherBackend", { { "DisableWinograd", true } });
    BOOST_TEST(GetCompilationOptions(config, { optOtherBackend }, 0).m_DisableWinograd == false);

    // Invalid option (unknown name)
    BackendOptions optInvalidName(EthosNBackend::GetIdStatic(), { { "TestInvalidOption", true } });
    BOOST_CHECK_THROW(GetCompilationOptions(config, { optInvalidName }, 0), InvalidArgumentException);

    // Invalid option (wrong option type)
    BackendOptions optInvalidType(EthosNBackend::GetIdStatic(), { { "DisableWinograd", "hello" } });
    BOOST_CHECK_THROW(GetCompilationOptions(config, { optInvalidType }, 0), InvalidArgumentException);
}

/// Checks that the m_DisableWinograd option is correctly passed through to the support library.
BOOST_AUTO_TEST_CASE(TestDisableWinograd)
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
    BOOST_CHECK_THROW(Optimize(*net, backends, runtime->GetDeviceSpec(), optOpts), armnn::InvalidArgumentException);

    // Check that support library was called correctly
    BOOST_TEST(mockSupportLibrary.m_RecordedDisableWinograd.back() == false);

    // Optimize for EthosNAcc (disable Winograd)
    optOpts.m_ModelOptions = { BackendOptions(EthosNBackend::GetIdStatic(), { { "DisableWinograd", true } }) };
    BOOST_CHECK_THROW(Optimize(*net, backends, runtime->GetDeviceSpec(), optOpts), armnn::InvalidArgumentException);

    // Check that support library was called correctly
    BOOST_TEST(mockSupportLibrary.m_RecordedDisableWinograd.back() == true);
}

BOOST_AUTO_TEST_SUITE_END()
