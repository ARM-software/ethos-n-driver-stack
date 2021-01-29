//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "EthosNLayerSupport.hpp"
#include "EthosNTestUtils.hpp"

#include <EthosNBackendId.hpp>
#include <EthosNSubgraphViewConverter.hpp>
#include <EthosNTensorUtils.hpp>
#include <Graph.hpp>
#include <armnn/ArmNN.hpp>
#include <armnn/utility/Assert.hpp>
#include <backendsCommon/test/CommonTestUtils.hpp>
#include <boost/test/unit_test.hpp>
#include <ethosn_support_library/Support.hpp>

using namespace armnn;

namespace
{

SubgraphView::SubgraphViewPtr BuildActivationSubgraph(Graph& graph, ActivationFunction activationFunction)
{
    const TensorInfo inputTensorInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);

    Layer* const inputLayer = graph.AddLayer<InputLayer>(0, "input");
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    Layer* outputLayer = graph.AddLayer<OutputLayer>(0, "output");

    // Set up activation layer
    ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = activationFunction;
    if (activationFunction == ActivationFunction::BoundedReLu)
    {
        activationDescriptor.m_A = 6.0f;    // ReLu6
        activationDescriptor.m_B = 0.0f;
    }
    if (activationFunction == ActivationFunction::LeakyReLu)
    {
        activationDescriptor.m_A = 0.1f;
        activationDescriptor.m_B = 0.0f;
    }

    const std::string layerName = "activation" + std::string(GetActivationFunctionAsCString(activationFunction));
    ActivationLayer* const activationLayer = graph.AddLayer<ActivationLayer>(activationDescriptor, layerName.c_str());
    activationLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    // Set up connections
    inputLayer->GetOutputSlot(0).Connect(activationLayer->GetInputSlot(0));
    activationLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Construct sub-graph
    return CreateSubgraphViewFrom(CreateInputsFrom({ activationLayer }), CreateOutputsFrom({ activationLayer }),
                                  { activationLayer });
}

class TestEthosNSubgraphViewConverter final : public EthosNSubgraphViewConverter
{
public:
    TestEthosNSubgraphViewConverter(const SubgraphView& subgraph)
        : EthosNSubgraphViewConverter(subgraph)
    {}

    void TestCreateUncompiledNetwork()
    {
        CreateUncompiledNetwork();
    }
};

}    // Anonymous namespace

BOOST_AUTO_TEST_SUITE(EthosNSupport)

// Simple test to check whether the Ethos-N support library is accessible
BOOST_AUTO_TEST_CASE(LibraryAccess)
{
    const std::string version = ethosn_lib::GetLibraryVersion().ToString();
    BOOST_TEST(version == "1.0.0");
}

BOOST_AUTO_TEST_CASE(ConvertAdditionLayer)
{
    Graph graph;

    // Create tensorinfo
    const TensorInfo inputTensorInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);

    // Construct graph
    Layer* inputLayer1 = graph.AddLayer<InputLayer>(0, "input1");
    inputLayer1->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    Layer* inputLayer2 = graph.AddLayer<InputLayer>(1, "input2");
    inputLayer2->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    Layer* additionLayer = graph.AddLayer<AdditionLayer>("addition");
    Layer* outputLayer   = graph.AddLayer<OutputLayer>(0, "output");

    // Set up connections
    inputLayer1->GetOutputSlot(0).Connect(additionLayer->GetInputSlot(0));
    inputLayer2->GetOutputSlot(0).Connect(additionLayer->GetInputSlot(1));
    additionLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Construct sub-graph
    SubgraphView::SubgraphViewPtr subgraphPtr = CreateSubgraphViewFrom(
        CreateInputsFrom({ additionLayer }), CreateOutputsFrom({ additionLayer }), { additionLayer });

    // Set up Ethos-N sub-graph converter
    TestEthosNSubgraphViewConverter converter(*subgraphPtr);

    // Check that we are able to convert the sub-graph
    BOOST_CHECK_NO_THROW(converter.TestCreateUncompiledNetwork());

    // Check that Ethos-N is able to compile the converted sub-graph
    BOOST_CHECK_NO_THROW(converter.CompileNetwork());
}

BOOST_AUTO_TEST_CASE(ConvertConcatLayer)
{
    Graph graph;

    const TensorInfo inputTensorInfo({ 1, 64, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo splitTensorInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);

    // Construct graph
    Layer* inputLayer = graph.AddLayer<InputLayer>(0, "input");
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    Layer* splitterLayer = graph.AddLayer<SplitterLayer>(ViewsDescriptor(4), "splitter");
    splitterLayer->GetOutputSlot(0).SetTensorInfo(splitTensorInfo);
    splitterLayer->GetOutputSlot(1).SetTensorInfo(splitTensorInfo);
    splitterLayer->GetOutputSlot(2).SetTensorInfo(splitTensorInfo);
    splitterLayer->GetOutputSlot(3).SetTensorInfo(splitTensorInfo);

    OriginsDescriptor concatDesc(4);
    concatDesc.SetConcatAxis(3);
    Layer* concatLayer = graph.AddLayer<ConcatLayer>(concatDesc, "concat");
    concatLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    Layer* outputLayer = graph.AddLayer<OutputLayer>(0, "output");

    // Set up connections
    inputLayer->GetOutputSlot(0).Connect(splitterLayer->GetInputSlot(0));
    splitterLayer->GetOutputSlot(0).Connect(concatLayer->GetInputSlot(0));
    splitterLayer->GetOutputSlot(1).Connect(concatLayer->GetInputSlot(1));
    splitterLayer->GetOutputSlot(2).Connect(concatLayer->GetInputSlot(2));
    splitterLayer->GetOutputSlot(3).Connect(concatLayer->GetInputSlot(3));
    concatLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Construct sub-graph
    SubgraphView::SubgraphViewPtr subgraphPtr =
        CreateSubgraphViewFrom(CreateInputsFrom({ concatLayer }), CreateOutputsFrom({ concatLayer }), { concatLayer });

    // Set up Ethos-N sub-graph converter
    TestEthosNSubgraphViewConverter converter(*subgraphPtr);

    // Check that we are able to convert the sub-graph
    BOOST_CHECK_NO_THROW(converter.TestCreateUncompiledNetwork());
}

// Tests focused on the BuildEthosNSplitInfo function, used as part of IsSplitterSupported.
BOOST_AUTO_TEST_CASE(IsSplitterSupported)
{
    using namespace ethosntensorutils;

    auto SetViewOriginAndSize = [](ViewsDescriptor& views, uint32_t viewIdx, std::array<uint32_t, 4> origin,
                                   std::array<uint32_t, 4> size) {
        for (uint32_t d = 0; d < origin.size(); ++d)
        {
            views.SetViewOriginCoord(viewIdx, d, origin[d]);
        }
        for (uint32_t d = 0; d < size.size(); ++d)
        {
            views.SetViewSize(viewIdx, d, size[d]);
        };
    };

    // Not enough views
    {
        ViewsDescriptor views(1, 4);
        BOOST_CHECK(!BuildEthosNSplitInfo({ 10, 10, 10, 10 }, views).has_value());
    }

    // First origin not at zero
    {
        ViewsDescriptor views(2, 4);
        SetViewOriginAndSize(views, 0, { 0, 0, 0, 1 }, { 10, 10, 10, 5 });
        BOOST_CHECK(!BuildEthosNSplitInfo({ 10, 10, 10, 10 }, views).has_value());
    }

    // Second origin at zero
    {
        ViewsDescriptor views(2, 4);
        SetViewOriginAndSize(views, 0, { 0, 0, 0, 0 }, { 10, 10, 10, 5 });
        SetViewOriginAndSize(views, 1, { 0, 0, 0, 0 }, { 10, 10, 10, 5 });
        BOOST_CHECK(!BuildEthosNSplitInfo({ 10, 10, 10, 10 }, views).has_value());
    }

    // Second origin non-zero in more than one dimension
    {
        ViewsDescriptor views(2, 4);
        SetViewOriginAndSize(views, 0, { 0, 0, 0, 0 }, { 10, 10, 10, 5 });
        SetViewOriginAndSize(views, 1, { 0, 0, 5, 5 }, { 10, 10, 10, 5 });
        BOOST_CHECK(!BuildEthosNSplitInfo({ 10, 10, 10, 10 }, views).has_value());
    }

    // Third origin non-zero in a dimension other than the split dimension
    {
        ViewsDescriptor views(3, 4);
        SetViewOriginAndSize(views, 0, { 0, 0, 0, 0 }, { 10, 10, 10, 3 });
        SetViewOriginAndSize(views, 1, { 0, 0, 0, 3 }, { 10, 10, 10, 3 });
        SetViewOriginAndSize(views, 2, { 0, 0, 1, 6 }, { 10, 10, 10, 4 });
        BOOST_CHECK(!BuildEthosNSplitInfo({ 10, 10, 10, 10 }, views).has_value());
    }

    // Gaps/overlaps along split axis
    {
        ViewsDescriptor views(2, 4);
        SetViewOriginAndSize(views, 0, { 0, 0, 0, 0 }, { 10, 10, 10, 5 });
        SetViewOriginAndSize(views, 1, { 0, 0, 0, 3 }, { 10, 10, 10, 5 });
        BOOST_CHECK(!BuildEthosNSplitInfo({ 10, 10, 10, 10 }, views).has_value());
    }

    // Other dimensions not filling the input tensor shape
    {
        ViewsDescriptor views(2, 4);
        SetViewOriginAndSize(views, 0, { 0, 0, 0, 0 }, { 10, 10, 10, 3 });
        SetViewOriginAndSize(views, 1, { 0, 0, 0, 3 }, { 10, 10, 9, 7 });
        BOOST_CHECK(!BuildEthosNSplitInfo({ 10, 10, 10, 10 }, views).has_value());
    }

    // Sucesss!
    {
        ViewsDescriptor views(2, 4);
        SetViewOriginAndSize(views, 0, { 0, 0, 0, 0 }, { 10, 10, 10, 3 });
        SetViewOriginAndSize(views, 1, { 0, 0, 0, 3 }, { 10, 10, 10, 7 });
        BOOST_CHECK(BuildEthosNSplitInfo({ 10, 10, 10, 10 }, views).value() == ethosn_lib::SplitInfo(3, { 3, 7 }));
    }
}

BOOST_AUTO_TEST_CASE(ConvertFullyConnectedLayer)
{
    Graph graph;

    const unsigned int width    = 8u;
    const unsigned int height   = width;
    const unsigned int channels = 1u;

    const unsigned int numInputs  = width * height * channels;
    const unsigned int numOutputs = 1u;

    const TensorInfo inputInfo({ 1, numInputs }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo outputInfo({ 1, numOutputs }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo weightInfo({ numInputs, numOutputs }, DataType::QAsymmU8, 0.9f, 0);
    const TensorInfo biasesInfo({ 1, numOutputs }, DataType::Signed32, 0.9f, 0);

    // Add InputLayer
    Layer* const inputLayer = graph.AddLayer<InputLayer>(0, "input");
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    // Add FullyConnectedLayer
    FullyConnectedDescriptor fullyConnectedDescriptor;
    fullyConnectedDescriptor.m_BiasEnabled = true;

    FullyConnectedLayer* const fullyConnectedLayer =
        graph.AddLayer<FullyConnectedLayer>(fullyConnectedDescriptor, "fullyConn");
    fullyConnectedLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    SetWeightAndBias(fullyConnectedLayer, weightInfo, biasesInfo);

    // Add OutputLayer
    Layer* const outputLayer = graph.AddLayer<OutputLayer>(0, "output");

    // Set up connections
    inputLayer->GetOutputSlot(0).Connect(fullyConnectedLayer->GetInputSlot(0));
    fullyConnectedLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    SubgraphView::SubgraphViewPtr subgraphPtr = CreateSubgraphViewFrom(
        CreateInputsFrom({ fullyConnectedLayer }), CreateOutputsFrom({ fullyConnectedLayer }), { fullyConnectedLayer });

    TestEthosNSubgraphViewConverter converter(*subgraphPtr);

    BOOST_CHECK_NO_THROW(converter.TestCreateUncompiledNetwork());
}

BOOST_AUTO_TEST_CASE(ConvertSigmoidLayer)
{
    Graph graph;

    SubgraphViewSelector::SubgraphViewPtr subgraphPtr = BuildActivationSubgraph(graph, ActivationFunction::Sigmoid);
    TestEthosNSubgraphViewConverter converter(*subgraphPtr);

    // Check that we are able to convert the sub-graph
    BOOST_CHECK_NO_THROW(converter.TestCreateUncompiledNetwork());
}

BOOST_AUTO_TEST_CASE(ConvertReLuLayer)
{
    Graph graph;

    SubgraphViewSelector::SubgraphViewPtr subgraphPtr = BuildActivationSubgraph(graph, ActivationFunction::ReLu);
    TestEthosNSubgraphViewConverter converter(*subgraphPtr);

    // Check that we are able to convert the sub-graph
    BOOST_CHECK_NO_THROW(converter.TestCreateUncompiledNetwork());
}

BOOST_AUTO_TEST_CASE(ConvertBoundedReLuLayer)
{
    Graph graph;

    SubgraphViewSelector::SubgraphViewPtr subgraphPtr = BuildActivationSubgraph(graph, ActivationFunction::BoundedReLu);
    TestEthosNSubgraphViewConverter converter(*subgraphPtr);

    // Check that we are able to convert the sub-graph
    BOOST_CHECK_NO_THROW(converter.TestCreateUncompiledNetwork());
}

BOOST_AUTO_TEST_CASE(ConvertLeakyReLuLayer)
{
    using namespace testing_utils;

    Graph graph;

    const TempDir tmpDir;
    const std::string configFile = tmpDir.Str() + "/config.txt";

    armnn::EthosNConfig config{};
    config.m_PerfOnly    = true;
    config.m_PerfOutDir  = tmpDir.Str();
    config.m_PerfCurrent = true;

    CreateConfigFile(configFile, config);
    SetEnv(armnn::EthosNConfig::CONFIG_FILE_ENV, configFile.c_str());

    SubgraphViewSelector::SubgraphViewPtr subgraphPtr = BuildActivationSubgraph(graph, ActivationFunction::LeakyReLu);
    TestEthosNSubgraphViewConverter converter(*subgraphPtr);

    // Check that we are able to convert the sub-graph when performance only mode.
    BOOST_CHECK_NO_THROW(converter.TestCreateUncompiledNetwork());
}

BOOST_AUTO_TEST_CASE(ConvertExecutionLeakyReLuLayer)
{
    Graph graph;

    SubgraphViewSelector::SubgraphViewPtr subgraphPtr = BuildActivationSubgraph(graph, ActivationFunction::LeakyReLu);
    TestEthosNSubgraphViewConverter converter(*subgraphPtr);

    BOOST_CHECK_NO_THROW(converter.TestCreateUncompiledNetwork());
}

BOOST_AUTO_TEST_CASE(ConvertDepthwiseConvolutionLayer)
{
    Graph graph;

    const TensorInfo inputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo outputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo weightInfo({ 1, 16, 1, 1 }, DataType::QAsymmU8, 0.9f, 0);
    const TensorInfo biasInfo({ 1, 1, 1, 16 }, DataType::Signed32, 0.9f, 0);

    Layer* const inputLayer = graph.AddLayer<InputLayer>(0, "input");
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    DepthwiseConvolution2dDescriptor depthwiseConvolutionDescriptor;
    depthwiseConvolutionDescriptor.m_BiasEnabled = true;
    depthwiseConvolutionDescriptor.m_DataLayout  = DataLayout::NHWC;
    depthwiseConvolutionDescriptor.m_StrideX     = 1;
    depthwiseConvolutionDescriptor.m_StrideY     = 1;

    DepthwiseConvolution2dLayer* const depthwiseConvolutionLayer =
        graph.AddLayer<DepthwiseConvolution2dLayer>(depthwiseConvolutionDescriptor, "depthWiseConv");
    depthwiseConvolutionLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    SetWeightAndBias(depthwiseConvolutionLayer, weightInfo, biasInfo);

    Layer* const outputLayer = graph.AddLayer<OutputLayer>(0, "output");

    inputLayer->GetOutputSlot(0).Connect(depthwiseConvolutionLayer->GetInputSlot(0));
    depthwiseConvolutionLayer->GetOutputSlot().Connect(outputLayer->GetInputSlot(0));

    SubgraphView::SubgraphViewPtr subgraphPtr =
        CreateSubgraphViewFrom(CreateInputsFrom({ depthwiseConvolutionLayer }),
                               CreateOutputsFrom({ depthwiseConvolutionLayer }), { depthwiseConvolutionLayer });

    TestEthosNSubgraphViewConverter converter(*subgraphPtr);

    // Check that we are able to convert the sub-graph
    BOOST_CHECK_NO_THROW(converter.TestCreateUncompiledNetwork());

    // Check that Ethos-N is able to compile the converted sub-graph
    BOOST_CHECK_NO_THROW(converter.CompileNetwork());
}

BOOST_AUTO_TEST_CASE(ConvertConvolutionLayer)
{
    Graph graph;

    const TensorInfo inputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo outputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo weightInfo({ 16, 1, 1, 16 }, DataType::QAsymmU8, 0.9f, 0);
    const TensorInfo biasInfo({ 1, 1, 1, 16 }, DataType::Signed32, 0.9f, 0);

    // Construct Graph
    Layer* const inputLayer = graph.AddLayer<InputLayer>(0, "input");
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    Convolution2dDescriptor convDescriptor;
    convDescriptor.m_BiasEnabled = true;
    convDescriptor.m_DataLayout  = DataLayout::NHWC;
    convDescriptor.m_StrideX     = 1;
    convDescriptor.m_StrideY     = 1;

    Convolution2dLayer* const convLayer = graph.AddLayer<Convolution2dLayer>(convDescriptor, "conv");

    convLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    SetWeightAndBias(convLayer, weightInfo, biasInfo);

    Layer* const outputLayer = graph.AddLayer<OutputLayer>(0, "output");

    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    convLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Retrieve Subgraph
    SubgraphView::SubgraphViewPtr subgraphPtr =
        CreateSubgraphViewFrom(CreateInputsFrom({ convLayer }), CreateOutputsFrom({ convLayer }), { convLayer });

    // Set up Ethos-N sub-graph converter
    TestEthosNSubgraphViewConverter converter(*subgraphPtr);

    // Check that we are able to convert the sub-graph
    BOOST_CHECK_NO_THROW(converter.TestCreateUncompiledNetwork());

    // Check that Ethos-N is able to compile the converted sub-graph
    BOOST_CHECK_NO_THROW(converter.CompileNetwork());
}

BOOST_AUTO_TEST_CASE(ConvertTransposeConvolutionLayer)
{
    Graph graph;

    const TensorInfo inputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo outputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo weightInfo({ 16, 3, 3, 16 }, DataType::QAsymmU8, 0.9f, 0);
    const TensorInfo biasInfo({ 1, 1, 1, 16 }, DataType::Signed32, 0.9f, 0);

    // Construct Graph
    Layer* const inputLayer = graph.AddLayer<InputLayer>(0, "input");
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    TransposeConvolution2dDescriptor convDescriptor;
    convDescriptor.m_BiasEnabled = true;
    convDescriptor.m_DataLayout  = DataLayout::NHWC;
    convDescriptor.m_StrideX     = 1;
    convDescriptor.m_StrideY     = 1;
    convDescriptor.m_PadTop      = 1;
    convDescriptor.m_PadLeft     = 1;

    TransposeConvolution2dLayer* const convLayer = graph.AddLayer<TransposeConvolution2dLayer>(convDescriptor, "conv");

    convLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    SetWeightAndBias(convLayer, weightInfo, biasInfo);

    Layer* const outputLayer = graph.AddLayer<OutputLayer>(0, "output");

    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    convLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Retrieve Subgraph
    SubgraphView::SubgraphViewPtr subgraphPtr =
        CreateSubgraphViewFrom(CreateInputsFrom({ convLayer }), CreateOutputsFrom({ convLayer }), { convLayer });

    // Set up Ethos-N sub-graph converter
    TestEthosNSubgraphViewConverter converter(*subgraphPtr);

    // This is not supported for now
    BOOST_CHECK_THROW(converter.TestCreateUncompiledNetwork(), ethosn_lib::NotSupportedException);
}

BOOST_AUTO_TEST_CASE(ConvertSoftmaxLayer)
{
    Graph graph;

    // Create tensorinfo
    const TensorInfo inputTensorInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);

    // Construct graph
    Layer* inputLayer = graph.AddLayer<InputLayer>(0, "input");
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    SoftmaxDescriptor softmaxDescriptor;
    Layer* softmaxLayer = graph.AddLayer<SoftmaxLayer>(softmaxDescriptor, "softmax");

    Layer* outputLayer = graph.AddLayer<OutputLayer>(0, "output");

    // Set up connections
    inputLayer->GetOutputSlot(0).Connect(softmaxLayer->GetInputSlot(0));
    softmaxLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Construct sub-graph
    SubgraphView::SubgraphViewPtr subgraphPtr = CreateSubgraphViewFrom(
        CreateInputsFrom({ softmaxLayer }), CreateOutputsFrom({ softmaxLayer }), { softmaxLayer });

    // Set up Ethos-N sub-graph converter
    TestEthosNSubgraphViewConverter converter(*subgraphPtr);

    BOOST_CHECK_THROW(converter.TestCreateUncompiledNetwork(), ethosn_lib::NotSupportedException);
}

SubgraphViewSelector::SubgraphViewPtr CreatePooling2dLayerSubgraph(Graph& graph,
                                                                   const TensorShape& inputTensorShape,
                                                                   const Pooling2dDescriptor& descriptor)
{
    // Create the input tensor info
    const TensorInfo inputTensorInfo(inputTensorShape, DataType::QAsymmU8, 1.0f, 0);

    // Construct the graph
    Layer* inputLayer     = graph.AddLayer<InputLayer>(0, "input");
    Layer* pooling2dLayer = graph.AddLayer<Pooling2dLayer>(descriptor, "pooling");
    Layer* outputLayer    = graph.AddLayer<OutputLayer>(0, "output");
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    // Set up the connections
    inputLayer->GetOutputSlot(0).Connect(pooling2dLayer->GetInputSlot(0));
    pooling2dLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Construct the sub-graph
    SubgraphView::SubgraphViewPtr subgraphPtr = CreateSubgraphViewFrom(
        CreateInputsFrom({ pooling2dLayer }), CreateOutputsFrom({ pooling2dLayer }), { pooling2dLayer });

    return subgraphPtr;
}

BOOST_AUTO_TEST_CASE(ConvertAvgPooling2dLayerUnsupported)
{
    TensorShape inputTensorShape{ 1, 16, 16, 16 };

    Pooling2dDescriptor descriptor;
    descriptor.m_PoolType      = armnn::PoolingAlgorithm::Average;
    descriptor.m_PoolWidth     = 2;
    descriptor.m_PoolHeight    = 2;
    descriptor.m_StrideX       = 2;
    descriptor.m_StrideY       = 2;
    descriptor.m_PadLeft       = 1;
    descriptor.m_PadRight      = 1;
    descriptor.m_PadTop        = 1;
    descriptor.m_PadBottom     = 1;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;
    descriptor.m_DataLayout    = DataLayout::NHWC;

    // The graph must be kept alive within scope for as long as we're gonna need a subgraph view to it
    Graph graph;

    // Construct the sub-graph
    SubgraphViewSelector::SubgraphViewPtr subgraphPtr =
        CreatePooling2dLayerSubgraph(graph, inputTensorShape, descriptor);

    // Get up the Ethos-N sub-graph converter
    TestEthosNSubgraphViewConverter converter(*subgraphPtr);

    // NOTE: Currently average 7x7 pooling for 7x7 input is supported, any stride is allowed
    BOOST_CHECK_THROW(converter.TestCreateUncompiledNetwork(), ethosn_lib::NotSupportedException);
}

BOOST_AUTO_TEST_CASE(ConvertAvgPooling2dLayerSupported)
{
    TensorShape inputTensorShape{ 1, 7, 7, 1 };

    Pooling2dDescriptor descriptor;
    descriptor.m_PoolType      = armnn::PoolingAlgorithm::Average;
    descriptor.m_PoolWidth     = 7;
    descriptor.m_PoolHeight    = 7;
    descriptor.m_StrideX       = 2;
    descriptor.m_StrideY       = 2;
    descriptor.m_PadLeft       = 0;
    descriptor.m_PadRight      = 0;
    descriptor.m_PadTop        = 0;
    descriptor.m_PadBottom     = 0;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;
    descriptor.m_DataLayout    = DataLayout::NHWC;

    // The graph must be kept alive within scope for as long as we're gonna need a subgraph view to it
    Graph graph;

    // Construct the sub-graph
    SubgraphViewSelector::SubgraphViewPtr subgraphPtr =
        CreatePooling2dLayerSubgraph(graph, inputTensorShape, descriptor);

    // Set up Ethos-N sub-graph converter
    TestEthosNSubgraphViewConverter converter(*subgraphPtr);

    // Check that we are able to convert the sub-graph
    BOOST_CHECK_NO_THROW(converter.TestCreateUncompiledNetwork());

    // Check that Ethos-N is able to compile the converted sub-graph
    BOOST_CHECK_NO_THROW(converter.CompileNetwork());
}

BOOST_AUTO_TEST_CASE(ConvertMaxPooling2dLayerSupported)
{
    TensorShape inputTensorShape{ 1, 8, 8, 1 };

    Pooling2dDescriptor descriptor;
    descriptor.m_PoolType      = armnn::PoolingAlgorithm::Max;
    descriptor.m_PoolWidth     = 2;
    descriptor.m_PoolHeight    = 2;
    descriptor.m_StrideX       = 2;
    descriptor.m_StrideY       = 2;
    descriptor.m_PadLeft       = 0;
    descriptor.m_PadRight      = 0;
    descriptor.m_PadTop        = 0;
    descriptor.m_PadBottom     = 0;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;
    descriptor.m_DataLayout    = DataLayout::NHWC;

    // The graph must be kept alive within scope for as long as we're gonna need a subgraph view to it
    Graph graph;

    // Construct the sub-graph
    SubgraphViewSelector::SubgraphViewPtr subgraphPtr =
        CreatePooling2dLayerSubgraph(graph, inputTensorShape, descriptor);

    // Set up Ethos-N sub-graph converter
    TestEthosNSubgraphViewConverter converter(*subgraphPtr);

    // Check that we are able to convert the sub-graph
    BOOST_CHECK_NO_THROW(converter.TestCreateUncompiledNetwork());

    // Check that Ethos-N is able to compile the converted sub-graph
    BOOST_CHECK_NO_THROW(converter.CompileNetwork());
}

BOOST_AUTO_TEST_CASE(ConvertReshapeLayer)
{
    Graph graph;

    // Create tensorinfo
    const TensorInfo inputTensorInfo({ 1, 4, 4, 16 }, DataType::QAsymmU8, 1.0f, 0);

    // Construct graph
    Layer* inputLayer = graph.AddLayer<InputLayer>(0, "input");
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    ReshapeDescriptor descriptor;
    descriptor.m_TargetShape = { 1, 1, 16, 16 };

    Layer* reshapeLayer = graph.AddLayer<ReshapeLayer>(descriptor, "reshape");

    Layer* outputLayer = graph.AddLayer<OutputLayer>(0, "output");

    // Set up connections
    inputLayer->GetOutputSlot(0).Connect(reshapeLayer->GetInputSlot(0));
    reshapeLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Construct sub-graph
    SubgraphView::SubgraphViewPtr subgraphPtr = CreateSubgraphViewFrom(
        CreateInputsFrom({ reshapeLayer }), CreateOutputsFrom({ reshapeLayer }), { reshapeLayer });

    // Set up Ethos-N  sub-graph converter
    TestEthosNSubgraphViewConverter converter(*subgraphPtr);

    // Check that we are able to convert the sub-graph
    BOOST_CHECK_NO_THROW(converter.TestCreateUncompiledNetwork());

    // Check that Ethos-N is able to compile the converted sub-graph
    BOOST_CHECK_NO_THROW(converter.CompileNetwork());
}

BOOST_AUTO_TEST_CASE(ConvertTransposeLayer)
{
    //Removed from the test because it is not part of the 20.08 delivery.
    //To be re-enabled in 20.11
    if (false)
    {
        Graph graph;

        // Create tensorinfo
        const TensorInfo inputTensorInfo({ 1, 32, 16, 8 }, DataType::QAsymmU8, 1.0f, 0);

        // Construct graph
        Layer* inputLayer = graph.AddLayer<InputLayer>(0, "input");
        inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

        TransposeDescriptor descriptor;
        descriptor.m_DimMappings = { 0, 2, 3, 1 };

        Layer* transposeLayer = graph.AddLayer<TransposeLayer>(descriptor, "transpose");

        Layer* outputLayer = graph.AddLayer<OutputLayer>(0, "output");

        // Set up connections
        inputLayer->GetOutputSlot(0).Connect(transposeLayer->GetInputSlot(0));
        transposeLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

        // Construct sub-graph
        SubgraphView::SubgraphViewPtr subgraphPtr = CreateSubgraphViewFrom(
            CreateInputsFrom({ transposeLayer }), CreateOutputsFrom({ transposeLayer }), { transposeLayer });

        // Set up Ethos-N  sub-graph converter
        TestEthosNSubgraphViewConverter converter(*subgraphPtr);

        // Check that we are able to convert the sub-graph
        BOOST_CHECK_NO_THROW(converter.TestCreateUncompiledNetwork());

        // Check that Ethos-N is able to compile the converted sub-graph
        BOOST_CHECK_NO_THROW(converter.CompileNetwork());
    }
}

BOOST_AUTO_TEST_CASE(ConvertQuantizeLayer)
{
    Graph graph;

    // Create tensorinfo
    const TensorInfo inputTensorInfo({ 1, 32, 16, 8 }, DataType::QAsymmU8, 0.7f, 127);
    const TensorInfo outputTensorInfo({ 1, 32, 16, 8 }, DataType::QAsymmU8, 0.5f, 30);

    // Construct graph
    Layer* inputLayer = graph.AddLayer<InputLayer>(0, "input");
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    Layer* quantizeLayer = graph.AddLayer<QuantizeLayer>("quantize");

    Layer* outputLayer = graph.AddLayer<OutputLayer>(0, "output");

    // Set up connections
    inputLayer->GetOutputSlot(0).Connect(quantizeLayer->GetInputSlot(0));
    quantizeLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    quantizeLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // Construct sub-graph
    SubgraphView::SubgraphViewPtr subgraphPtr = CreateSubgraphViewFrom(
        CreateInputsFrom({ quantizeLayer }), CreateOutputsFrom({ quantizeLayer }), { quantizeLayer });

    // Set up Ethos-N  sub-graph converter
    TestEthosNSubgraphViewConverter converter(*subgraphPtr);

    // Check that we are able to convert the sub-graph
    BOOST_CHECK_NO_THROW(converter.TestCreateUncompiledNetwork());

    // Check that Ethos-N is able to compile the converted sub-graph
    BOOST_CHECK_NO_THROW(converter.CompileNetwork());
}

BOOST_AUTO_TEST_CASE(ConvertResizeLayer)
{
    Graph graph;

    // Create tensorinfo
    const TensorInfo inputTensorInfo({ 1, 32, 16, 8 }, DataType::QAsymmU8, 1.0f, 0);

    // Construct graph
    Layer* inputLayer = graph.AddLayer<InputLayer>(0, "input");
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    ResizeDescriptor descriptor;
    descriptor.m_Method       = ResizeMethod::Bilinear;
    descriptor.m_TargetHeight = 64;
    descriptor.m_TargetWidth  = 32;

    Layer* resizeLayer = graph.AddLayer<ResizeLayer>(descriptor, "resize");

    Layer* outputLayer = graph.AddLayer<OutputLayer>(0, "output");

    // Set up connections
    inputLayer->GetOutputSlot(0).Connect(resizeLayer->GetInputSlot(0));
    resizeLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Construct sub-graph
    SubgraphView::SubgraphViewPtr subgraphPtr =
        CreateSubgraphViewFrom(CreateInputsFrom({ resizeLayer }), CreateOutputsFrom({ resizeLayer }), { resizeLayer });

    // Set up Ethos-N  sub-graph converter
    TestEthosNSubgraphViewConverter converter(*subgraphPtr);

    // Check that we are able to convert the sub-graph
    BOOST_CHECK_NO_THROW(converter.TestCreateUncompiledNetwork());

    // Check that Ethos-N is able to compile the converted sub-graph
    BOOST_CHECK_NO_THROW(converter.CompileNetwork());
}

BOOST_AUTO_TEST_CASE(TestConvolutionLayerWithLargeTensors)
{
    Graph graph;

    // Since we are supporting splitting in width and depth in conversion pass
    // Ethos-N should be able to compile sub-graph with large input tensors
    const TensorInfo inputInfo({ 1, 16, 10000, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo outputInfo({ 1, 16, 10000, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo weightInfo({ 16, 1, 1, 16 }, DataType::QAsymmU8, 0.9f, 0);
    const TensorInfo biasInfo({ 1, 1, 1, 16 }, DataType::Signed32, 0.9f, 0);

    // Construct Graph
    Layer* const inputLayer = graph.AddLayer<InputLayer>(0, "input");
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    Convolution2dDescriptor convDescriptor;
    convDescriptor.m_BiasEnabled = true;
    convDescriptor.m_DataLayout  = DataLayout::NHWC;
    convDescriptor.m_StrideX     = 1;
    convDescriptor.m_StrideY     = 1;

    Convolution2dLayer* const convLayer = graph.AddLayer<Convolution2dLayer>(convDescriptor, "conv");

    convLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    SetWeightAndBias(convLayer, weightInfo, biasInfo);

    Layer* const outputLayer = graph.AddLayer<OutputLayer>(0, "output");

    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    convLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Retrieve Subgraph
    SubgraphView::SubgraphViewPtr subgraphPtr =
        CreateSubgraphViewFrom(CreateInputsFrom({ convLayer }), CreateOutputsFrom({ convLayer }), { convLayer });

    // Set up Ethos-N sub-graph converter
    TestEthosNSubgraphViewConverter converter(*subgraphPtr);

    // Check that we are able to convert the sub-graph
    BOOST_CHECK_NO_THROW(converter.TestCreateUncompiledNetwork());

    // Check that the Ethos-N is able to compile the converted sub-graph
    BOOST_CHECK_NO_THROW(converter.CompileNetwork());
}

BOOST_AUTO_TEST_CASE(TestEthosNBackendFail)
{
    using namespace armnn;

    // build up the structure of the network
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0);

    NormalizationDescriptor descriptor;
    IConnectableLayer* pooling = net->AddNormalizationLayer(descriptor);

    IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(pooling->GetInputSlot(0));
    pooling->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 4, 4 }, DataType::Float32));
    pooling->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 4, 4 }, DataType::Float32));

    // optimize the network
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));
    std::vector<BackendId> backends = { EthosNBackendId() };
    // Optimize should throw the Ethos-N backend will never support Float32 normalization
    BOOST_CHECK_THROW(Optimize(*net, backends, runtime->GetDeviceSpec()), armnn::InvalidArgumentException);
}

BOOST_AUTO_TEST_CASE(EstimateOnly5dFail)
{
    using namespace armnn;
    using namespace testing_utils;

    const TempDir tmpDir;

    const std::string configFile = tmpDir.Str() + "/config.txt";
    const EthosNConfig config    = { true, ethosn_lib::EthosNVariant::ETHOS_N77, 0, tmpDir.Str() };

    CreateConfigFile(configFile, config);

    SetEnv(armnn::EthosNConfig::CONFIG_FILE_ENV, configFile.c_str());
    EthosNLayerSupport layerSupport;
    TensorInfo input  = TensorInfo({ 1, 1, 1, 1, 4 }, DataType::QAsymmU8, 1.f, 0);
    TensorInfo output = TensorInfo({ 1, 1, 1, 1, 4 }, DataType::QAsymmU8, 1.f, 0);
    std::string reasonIfUnsupported;
    ARMNN_ASSERT(!layerSupport.IsRsqrtSupported(input, output, reasonIfUnsupported));
    ARMNN_ASSERT(reasonIfUnsupported == "The ethosn can only support up to 4D tensors");
}

/// Checks the error message produced when the backend fails to claim support for Multiplication
/// by attempting to substitute the operation with DepthwiseConvolution2d.
BOOST_AUTO_TEST_CASE(TestMulSubstitutionFail)
{
    using namespace armnn;
    using namespace testing_utils;

    const TempDir tmpDir;

    const std::string configFile = tmpDir.Str() + "/config.txt";
    const EthosNConfig config    = { true, ethosn_lib::EthosNVariant::ETHOS_N77, 0, tmpDir.Str() };

    CreateConfigFile(configFile, config);

    SetEnv(armnn::EthosNConfig::CONFIG_FILE_ENV, configFile.c_str());
    EthosNLayerSupport layerSupport;

    // input1 is assumed to be a constant and will be used for the weights of the convolution
    TensorInfo input0 = TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0);
    TensorInfo input1 = TensorInfo({ 1, 1, 1, 4 }, DataType::Signed32, 1.0f, 0);
    TensorInfo output = TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 0.9f, 0);

    std::string reasonIfUnsupported;
    std::string expectedReasonIfSupported =
        "Multiplication operation is not supported on Arm Ethos-N NPU backend and an attempt was made to substitute "
        "for DepthwiseConvolution2d, however the following error occured when checking for Depthwise support: Weight "
        "for conv must be UINT8_QUANTIZED or INT8_QUANTIZED";

    ARMNN_ASSERT(!layerSupport.IsMultiplicationSupported(input0, input1, output, reasonIfUnsupported));
    ARMNN_ASSERT(reasonIfUnsupported == expectedReasonIfSupported);
}

BOOST_AUTO_TEST_SUITE_END()
