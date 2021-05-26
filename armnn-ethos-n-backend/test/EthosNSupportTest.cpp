//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "EthosNLayerSupport.hpp"
#include "EthosNTestUtils.hpp"

#include <EthosNBackend.hpp>
#include <EthosNBackendId.hpp>
#include <EthosNSubgraphViewConverter.hpp>
#include <EthosNTensorUtils.hpp>
#include <Graph.hpp>
#include <armnn/ArmNN.hpp>
#include <armnn/utility/Assert.hpp>
#include <backendsCommon/test/CommonTestUtils.hpp>
#include <doctest/doctest.h>
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
    TestEthosNSubgraphViewConverter(const SubgraphView& subgraph,
                                    const EthosNConfig& config,
                                    const std::vector<char>& capabilities)
        : EthosNSubgraphViewConverter(subgraph, {}, config, capabilities)
    {}

    void TestCreateUncompiledNetwork()
    {
        CreateUncompiledNetwork();
    }
};

}    // Anonymous namespace

TEST_SUITE("EthosNSupport")
{

    // Simple test to check whether the Ethos-N support library is accessible
    TEST_CASE("LibraryAccess")
    {
        const std::string version = ethosn_lib::GetLibraryVersion().ToString();
        const std::string macroVer =
            ethosn_lib::Version(ETHOSN_SUPPORT_LIBRARY_VERSION_MAJOR, ETHOSN_SUPPORT_LIBRARY_VERSION_MINOR,
                                ETHOSN_SUPPORT_LIBRARY_VERSION_PATCH)
                .ToString();
        CHECK(version == macroVer);
    }

    TEST_CASE("LibrarySupport")
    {
        CHECK(ethosnbackend::VerifyLibraries());
    }

    TEST_CASE("ConvertAdditionLayer")
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
        TestEthosNSubgraphViewConverter converter(*subgraphPtr, EthosNConfig(), EthosNConfig().QueryCapabilities());

        // Check that we are able to convert the sub-graph
        CHECK_NOTHROW(converter.TestCreateUncompiledNetwork());

        // Check that Ethos-N is able to compile the converted sub-graph
        CHECK_NOTHROW(converter.CompileNetwork());
    }

    TEST_CASE("ConvertConcatLayer")
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
        SubgraphView::SubgraphViewPtr subgraphPtr = CreateSubgraphViewFrom(
            CreateInputsFrom({ concatLayer }), CreateOutputsFrom({ concatLayer }), { concatLayer });

        // Set up Ethos-N sub-graph converter
        TestEthosNSubgraphViewConverter converter(*subgraphPtr, EthosNConfig(), EthosNConfig().QueryCapabilities());

        // Check that we are able to convert the sub-graph
        CHECK_NOTHROW(converter.TestCreateUncompiledNetwork());
    }

    // Tests focused on the BuildEthosNSplitInfo function, used as part of IsSplitterSupported.
    TEST_CASE("IsSplitterSupported")
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
            CHECK(!BuildEthosNSplitInfo({ 10, 10, 10, 10 }, views).has_value());
        }

        // First origin not at zero
        {
            ViewsDescriptor views(2, 4);
            SetViewOriginAndSize(views, 0, { 0, 0, 0, 1 }, { 10, 10, 10, 5 });
            CHECK(!BuildEthosNSplitInfo({ 10, 10, 10, 10 }, views).has_value());
        }

        // Second origin at zero
        {
            ViewsDescriptor views(2, 4);
            SetViewOriginAndSize(views, 0, { 0, 0, 0, 0 }, { 10, 10, 10, 5 });
            SetViewOriginAndSize(views, 1, { 0, 0, 0, 0 }, { 10, 10, 10, 5 });
            CHECK(!BuildEthosNSplitInfo({ 10, 10, 10, 10 }, views).has_value());
        }

        // Second origin non-zero in more than one dimension
        {
            ViewsDescriptor views(2, 4);
            SetViewOriginAndSize(views, 0, { 0, 0, 0, 0 }, { 10, 10, 10, 5 });
            SetViewOriginAndSize(views, 1, { 0, 0, 5, 5 }, { 10, 10, 10, 5 });
            CHECK(!BuildEthosNSplitInfo({ 10, 10, 10, 10 }, views).has_value());
        }

        // Third origin non-zero in a dimension other than the split dimension
        {
            ViewsDescriptor views(3, 4);
            SetViewOriginAndSize(views, 0, { 0, 0, 0, 0 }, { 10, 10, 10, 3 });
            SetViewOriginAndSize(views, 1, { 0, 0, 0, 3 }, { 10, 10, 10, 3 });
            SetViewOriginAndSize(views, 2, { 0, 0, 1, 6 }, { 10, 10, 10, 4 });
            CHECK(!BuildEthosNSplitInfo({ 10, 10, 10, 10 }, views).has_value());
        }

        // Gaps/overlaps along split axis
        {
            ViewsDescriptor views(2, 4);
            SetViewOriginAndSize(views, 0, { 0, 0, 0, 0 }, { 10, 10, 10, 5 });
            SetViewOriginAndSize(views, 1, { 0, 0, 0, 3 }, { 10, 10, 10, 5 });
            CHECK(!BuildEthosNSplitInfo({ 10, 10, 10, 10 }, views).has_value());
        }

        // Other dimensions not filling the input tensor shape
        {
            ViewsDescriptor views(2, 4);
            SetViewOriginAndSize(views, 0, { 0, 0, 0, 0 }, { 10, 10, 10, 3 });
            SetViewOriginAndSize(views, 1, { 0, 0, 0, 3 }, { 10, 10, 9, 7 });
            CHECK(!BuildEthosNSplitInfo({ 10, 10, 10, 10 }, views).has_value());
        }

        // Sucesss!
        {
            ViewsDescriptor views(2, 4);
            SetViewOriginAndSize(views, 0, { 0, 0, 0, 0 }, { 10, 10, 10, 3 });
            SetViewOriginAndSize(views, 1, { 0, 0, 0, 3 }, { 10, 10, 10, 7 });
            CHECK(BuildEthosNSplitInfo({ 10, 10, 10, 10 }, views).value() == ethosn_lib::SplitInfo(3, { 3, 7 }));
        }
    }

    TEST_CASE("ConvertFullyConnectedLayer")
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

        SubgraphView::SubgraphViewPtr subgraphPtr =
            CreateSubgraphViewFrom(CreateInputsFrom({ fullyConnectedLayer }),
                                   CreateOutputsFrom({ fullyConnectedLayer }), { fullyConnectedLayer });

        TestEthosNSubgraphViewConverter converter(*subgraphPtr, EthosNConfig(), EthosNConfig().QueryCapabilities());

        CHECK_NOTHROW(converter.TestCreateUncompiledNetwork());
    }

    TEST_CASE("ConvertSigmoidLayer")
    {
        Graph graph;

        SubgraphViewSelector::SubgraphViewPtr subgraphPtr = BuildActivationSubgraph(graph, ActivationFunction::Sigmoid);
        TestEthosNSubgraphViewConverter converter(*subgraphPtr, EthosNConfig(), EthosNConfig().QueryCapabilities());

        // Check that we are able to convert the sub-graph
        CHECK_NOTHROW(converter.TestCreateUncompiledNetwork());
    }

    TEST_CASE("ConvertTanhLayer")
    {
        Graph graph;

        SubgraphViewSelector::SubgraphViewPtr subgraphPtr = BuildActivationSubgraph(graph, ActivationFunction::TanH);
        TestEthosNSubgraphViewConverter converter(*subgraphPtr, EthosNConfig(), EthosNConfig().QueryCapabilities());

        // Check that we are able to convert the sub-graph
        CHECK_NOTHROW(converter.TestCreateUncompiledNetwork());
    }

    TEST_CASE("ConvertReLuLayer")
    {
        Graph graph;

        SubgraphViewSelector::SubgraphViewPtr subgraphPtr = BuildActivationSubgraph(graph, ActivationFunction::ReLu);
        TestEthosNSubgraphViewConverter converter(*subgraphPtr, EthosNConfig(), EthosNConfig().QueryCapabilities());

        // Check that we are able to convert the sub-graph
        CHECK_NOTHROW(converter.TestCreateUncompiledNetwork());
    }

    TEST_CASE("ConvertBoundedReLuLayer")
    {
        Graph graph;

        SubgraphViewSelector::SubgraphViewPtr subgraphPtr =
            BuildActivationSubgraph(graph, ActivationFunction::BoundedReLu);
        TestEthosNSubgraphViewConverter converter(*subgraphPtr, EthosNConfig(), EthosNConfig().QueryCapabilities());

        // Check that we are able to convert the sub-graph
        CHECK_NOTHROW(converter.TestCreateUncompiledNetwork());
    }

    TEST_CASE("ConvertLeakyReLuLayer")
    {
        using namespace testing_utils;

        Graph graph;

        armnn::EthosNConfig config{};
        config.m_PerfOnly    = true;
        config.m_PerfCurrent = true;

        SubgraphViewSelector::SubgraphViewPtr subgraphPtr =
            BuildActivationSubgraph(graph, ActivationFunction::LeakyReLu);
        TestEthosNSubgraphViewConverter converter(*subgraphPtr, config, config.QueryCapabilities());

        // Check that we are able to convert the sub-graph when performance only mode.
        CHECK_NOTHROW(converter.TestCreateUncompiledNetwork());
    }

    TEST_CASE("ConvertExecutionLeakyReLuLayer")
    {
        Graph graph;

        SubgraphViewSelector::SubgraphViewPtr subgraphPtr =
            BuildActivationSubgraph(graph, ActivationFunction::LeakyReLu);
        TestEthosNSubgraphViewConverter converter(*subgraphPtr, EthosNConfig(), EthosNConfig().QueryCapabilities());

        CHECK_NOTHROW(converter.TestCreateUncompiledNetwork());
    }

    TEST_CASE("ConvertDepthwiseConvolutionLayer")
    {
        Graph graph;

        const TensorInfo inputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
        const TensorInfo outputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
        const TensorInfo weightInfo({ 1, 1, 1, 16 }, DataType::QAsymmU8, 0.9f, 0);
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

        TestEthosNSubgraphViewConverter converter(*subgraphPtr, EthosNConfig(), EthosNConfig().QueryCapabilities());

        // Check that we are able to convert the sub-graph
        CHECK_NOTHROW(converter.TestCreateUncompiledNetwork());

        // Check that Ethos-N is able to compile the converted sub-graph
        CHECK_NOTHROW(converter.CompileNetwork());
    }

    TEST_CASE("ConvertConvolutionLayer")
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
        TestEthosNSubgraphViewConverter converter(*subgraphPtr, EthosNConfig(), EthosNConfig().QueryCapabilities());

        // Check that we are able to convert the sub-graph
        CHECK_NOTHROW(converter.TestCreateUncompiledNetwork());

        // Check that Ethos-N is able to compile the converted sub-graph
        CHECK_NOTHROW(converter.CompileNetwork());
    }

    TEST_CASE("ConvertTransposeConvolutionLayer")
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

        TransposeConvolution2dLayer* const convLayer =
            graph.AddLayer<TransposeConvolution2dLayer>(convDescriptor, "conv");

        convLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

        SetWeightAndBias(convLayer, weightInfo, biasInfo);

        Layer* const outputLayer = graph.AddLayer<OutputLayer>(0, "output");

        inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
        convLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

        // Retrieve Subgraph
        SubgraphView::SubgraphViewPtr subgraphPtr =
            CreateSubgraphViewFrom(CreateInputsFrom({ convLayer }), CreateOutputsFrom({ convLayer }), { convLayer });

        // Set up Ethos-N sub-graph converter
        TestEthosNSubgraphViewConverter converter(*subgraphPtr, EthosNConfig(), EthosNConfig().QueryCapabilities());

        // This is not supported for now
        CHECK_THROWS_AS(converter.TestCreateUncompiledNetwork(), ethosn_lib::NotSupportedException);
    }

    TEST_CASE("ConvertSoftmaxLayer")
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
        TestEthosNSubgraphViewConverter converter(*subgraphPtr, EthosNConfig(), EthosNConfig().QueryCapabilities());

        CHECK_THROWS_AS(converter.TestCreateUncompiledNetwork(), ethosn_lib::NotSupportedException);
    }

    SubgraphViewSelector::SubgraphViewPtr CreatePooling2dLayerSubgraph(
        Graph & graph, const TensorShape& inputTensorShape, const Pooling2dDescriptor& descriptor)
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

    TEST_CASE("ConvertAvgPooling2dLayerUnsupported")
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
        TestEthosNSubgraphViewConverter converter(*subgraphPtr, EthosNConfig(), EthosNConfig().QueryCapabilities());

        // NOTE: Currently average 7x7 pooling for 7x7 input is supported, any stride is allowed
        CHECK_THROWS_AS(converter.TestCreateUncompiledNetwork(), ethosn_lib::NotSupportedException);
    }

    TEST_CASE("ConvertAvgPooling2dLayerSupported")
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
        TestEthosNSubgraphViewConverter converter(*subgraphPtr, EthosNConfig(), EthosNConfig().QueryCapabilities());

        // Check that we are able to convert the sub-graph
        CHECK_NOTHROW(converter.TestCreateUncompiledNetwork());

        // Check that Ethos-N is able to compile the converted sub-graph
        CHECK_NOTHROW(converter.CompileNetwork());
    }

    TEST_CASE("ConvertMaxPooling2dLayerSupported")
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
        TestEthosNSubgraphViewConverter converter(*subgraphPtr, EthosNConfig(), EthosNConfig().QueryCapabilities());

        // Check that we are able to convert the sub-graph
        CHECK_NOTHROW(converter.TestCreateUncompiledNetwork());

        // Check that Ethos-N is able to compile the converted sub-graph
        CHECK_NOTHROW(converter.CompileNetwork());
    }

    TEST_CASE("ConvertReshapeLayer")
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
        TestEthosNSubgraphViewConverter converter(*subgraphPtr, EthosNConfig(), EthosNConfig().QueryCapabilities());

        // Check that we are able to convert the sub-graph
        CHECK_NOTHROW(converter.TestCreateUncompiledNetwork());

        // Check that Ethos-N is able to compile the converted sub-graph
        CHECK_NOTHROW(converter.CompileNetwork());
    }

    TEST_CASE("ConvertTransposeLayer")
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
        TestEthosNSubgraphViewConverter converter(*subgraphPtr, EthosNConfig(), EthosNConfig().QueryCapabilities());

        // Check that we are able to convert the sub-graph
        CHECK_NOTHROW(converter.TestCreateUncompiledNetwork());

        // Check that Ethos-N is able to compile the converted sub-graph
        CHECK_NOTHROW(converter.CompileNetwork());
    }

    TEST_CASE("ConvertQuantizeLayer")
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
        TestEthosNSubgraphViewConverter converter(*subgraphPtr, EthosNConfig(), EthosNConfig().QueryCapabilities());

        // Check that we are able to convert the sub-graph
        CHECK_NOTHROW(converter.TestCreateUncompiledNetwork());

        // Check that Ethos-N is able to compile the converted sub-graph
        CHECK_NOTHROW(converter.CompileNetwork());
    }

    TEST_CASE("ConvertResizeLayer")
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
        SubgraphView::SubgraphViewPtr subgraphPtr = CreateSubgraphViewFrom(
            CreateInputsFrom({ resizeLayer }), CreateOutputsFrom({ resizeLayer }), { resizeLayer });

        // Set up Ethos-N  sub-graph converter
        TestEthosNSubgraphViewConverter converter(*subgraphPtr, EthosNConfig(), EthosNConfig().QueryCapabilities());

        // Check that we are able to convert the sub-graph
        CHECK_NOTHROW(converter.TestCreateUncompiledNetwork());

        // Check that Ethos-N is able to compile the converted sub-graph
        CHECK_NOTHROW(converter.CompileNetwork());
    }

    TEST_CASE("TestConvolutionLayerWithLargeTensors")
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
        TestEthosNSubgraphViewConverter converter(*subgraphPtr, EthosNConfig(), EthosNConfig().QueryCapabilities());

        // Check that we are able to convert the sub-graph
        CHECK_NOTHROW(converter.TestCreateUncompiledNetwork());

        // Check that the Ethos-N is able to compile the converted sub-graph
        CHECK_NOTHROW(converter.CompileNetwork());
    }

    TEST_CASE("TestStandInFail")
    {
        using namespace armnn;

        Graph graph;

        const TensorInfo inputInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0);
        const TensorInfo outputInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0);

        // Construct Graph
        Layer* const inputLayer = graph.AddLayer<InputLayer>(0, "input");
        inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

        StandInDescriptor desc;
        desc.m_NumInputs  = 1;
        desc.m_NumOutputs = 1;

        const auto standInLayer = graph.AddLayer<StandInLayer>(desc, "RandomStandInLayer");

        standInLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

        Layer* const outputLayer = graph.AddLayer<OutputLayer>(0, "output");

        inputLayer->GetOutputSlot(0).Connect(standInLayer->GetInputSlot(0));
        standInLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

        // Retrieve Subgraph
        SubgraphView::SubgraphViewPtr subgraphPtr = CreateSubgraphViewFrom(
            CreateInputsFrom({ standInLayer }), CreateOutputsFrom({ standInLayer }), { standInLayer });

        armnn::EthosNConfig config{};

        // Set up Ethos-N sub-graph converter
        TestEthosNSubgraphViewConverter converter(*subgraphPtr, config, config.QueryCapabilities());

        // We won't be able to convert the sub-graph since StandIn layer is not supported with the provided name parameter.
        CHECK_THROWS_WITH_AS(converter.TestCreateUncompiledNetwork(), "Conversion not supported for layer type StandIn",
                             armnn::Exception);
    }

    TEST_CASE("TestStandInPerfOnlyPass")
    {
        using namespace armnn;

        Graph graph;

        const TensorInfo inputInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0);
        const TensorInfo outputInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0);

        // Construct Graph
        Layer* const inputLayer = graph.AddLayer<InputLayer>(0, "input");
        inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

        StandInDescriptor desc;
        desc.m_NumInputs  = 1;
        desc.m_NumOutputs = 1;

        const auto standInLayer = graph.AddLayer<StandInLayer>(desc, "Random:StandInLayer");

        standInLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

        Layer* const outputLayer = graph.AddLayer<OutputLayer>(0, "output");

        inputLayer->GetOutputSlot(0).Connect(standInLayer->GetInputSlot(0));
        standInLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

        // Retrieve Subgraph
        SubgraphView::SubgraphViewPtr subgraphPtr = CreateSubgraphViewFrom(
            CreateInputsFrom({ standInLayer }), CreateOutputsFrom({ standInLayer }), { standInLayer });

        armnn::EthosNConfig config{};
        config.m_PerfOnly    = true;
        config.m_PerfCurrent = true;

        // Set up Ethos-N sub-graph converter
        TestEthosNSubgraphViewConverter converter(*subgraphPtr, config, config.QueryCapabilities());

        // Check that we are able to convert the sub-graph
        CHECK_NOTHROW(converter.TestCreateUncompiledNetwork());
    }

    TEST_CASE("TestStandInPerfOnlyFail")
    {
        using namespace armnn;

        Graph graph;

        const TensorInfo inputInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0);
        const TensorInfo outputInfo({ 1, 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0);

        // Construct Graph
        Layer* const inputLayer = graph.AddLayer<InputLayer>(0, "input");
        inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

        StandInDescriptor desc;
        desc.m_NumInputs  = 1;
        desc.m_NumOutputs = 1;

        const auto standInLayer = graph.AddLayer<StandInLayer>(desc, "Random:StandInLayer");

        standInLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

        Layer* const outputLayer = graph.AddLayer<OutputLayer>(0, "output");

        inputLayer->GetOutputSlot(0).Connect(standInLayer->GetInputSlot(0));
        standInLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

        // Retrieve Subgraph
        SubgraphView::SubgraphViewPtr subgraphPtr = CreateSubgraphViewFrom(
            CreateInputsFrom({ standInLayer }), CreateOutputsFrom({ standInLayer }), { standInLayer });

        armnn::EthosNConfig config{};
        config.m_PerfOnly    = true;
        config.m_PerfCurrent = true;

        // Set up Ethos-N sub-graph converter
        TestEthosNSubgraphViewConverter converter(*subgraphPtr, config, config.QueryCapabilities());

        // Invalid TensorShape: max number of dimensions exceeded in EthosNAcc backend 5 > 4
        CHECK_THROWS_AS(converter.TestCreateUncompiledNetwork(), armnn::InvalidArgumentException);
    }

    TEST_CASE("TestEthosNBackendFail")
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
        CHECK_THROWS_AS(Optimize(*net, backends, runtime->GetDeviceSpec()), armnn::InvalidArgumentException);
    }

    TEST_CASE("EstimateOnly5dFail")
    {
        using namespace armnn;
        using namespace testing_utils;

        const EthosNConfig config = { true, ethosn_lib::EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO, 0 };

        EthosNLayerSupport layerSupport(config, EthosNMappings(), config.QueryCapabilities());
        TensorInfo input  = TensorInfo({ 1, 1, 1, 1, 4 }, DataType::QAsymmU8, 1.f, 0);
        TensorInfo output = TensorInfo({ 1, 1, 1, 1, 4 }, DataType::QAsymmU8, 1.f, 0);
        std::string reasonIfUnsupported;
        CHECK(!layerSupport.IsRsqrtSupported(input, output, reasonIfUnsupported));
        CHECK(reasonIfUnsupported == "The ethosn can only support up to 4D tensors");
    }

    /// Checks the error message produced when the backend fails to claim support for Multiplication
    /// by attempting to substitute the operation with DepthwiseConvolution2d.
    TEST_CASE("MulSubstitutionFail")
    {
        EthosNLayerSupport layerSupport(EthosNConfig(), EthosNMappings(), EthosNConfig().QueryCapabilities());

        // input1 is assumed to be a constant and will be used for the weights of the convolution
        TensorInfo input0 = TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0);
        TensorInfo input1 = TensorInfo({ 1, 1, 1, 4 }, DataType::Signed32, 1.0f, 0);
        TensorInfo output = TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 0.9f, 0);

        std::string reasonIfUnsupported;
        std::string expectedReasonIfSupported = "Multiplication operation is not supported on Arm Ethos-N NPU backend "
                                                "and an attempt was made to substitute "
                                                "for DepthwiseConvolution2d, however the following error occurred when "
                                                "checking for Depthwise support: Weight "
                                                "for conv must be UINT8_QUANTIZED or INT8_QUANTIZED";

        CHECK(!layerSupport.IsMultiplicationSupported(input0, input1, output, reasonIfUnsupported));
        CHECK(reasonIfUnsupported == expectedReasonIfSupported);
    }

    TEST_CASE("IsMultiplicationSupported")
    {
        EthosNLayerSupport layerSupport(EthosNConfig(), EthosNMappings(), EthosNConfig().QueryCapabilities());

        auto ExpectFail = [&layerSupport](const TensorInfo& input0, const TensorInfo& input1, const TensorInfo& output,
                                          const char* expectedFailureReason) {
            std::string failureReason;
            CHECK(!layerSupport.IsMultiplicationSupported(input0, input1, output, failureReason));
            CHECK(failureReason.find(expectedFailureReason) != std::string::npos);
        };

        // Failure case - 5D tensor
        ExpectFail(TensorInfo({ 1, 2, 2, 4, 9 }, DataType::QAsymmU8, 1.0f, 0),
                   TensorInfo({ 1, 1, 1, 4 }, DataType::Signed32, 0.9f, 0),
                   TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0),
                   "The ethosn can only support up to 4D tensors");

        // Success case - multiplication supported by replacing it with Depthwise
        // Additionally, verifying that the correct MultiplicationSupportedMode value is returned
        CHECK(layerSupport.GetMultiplicationSupportedMode(TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0),
                                                          TensorInfo({ 1, 1, 1, 4 }, DataType::QAsymmU8, 0.9f, 0),
                                                          TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0)) ==
              EthosNLayerSupport::MultiplicationSupportedMode::ReplaceWithDepthwise);

        // Success case - multiplication supported by replacing it with ReinterpretQuantize
        // Additionally, verifying that the correct MultiplicationSupportedMode value is returned
        CHECK(layerSupport.GetMultiplicationSupportedMode(TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0),
                                                          TensorInfo({ 1, 1, 1, 1 }, DataType::QAsymmU8, 0.009f, 0),
                                                          TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0)) ==
              EthosNLayerSupport::MultiplicationSupportedMode::ReplaceWithReinterpretQuantize);

        // Failure case - multiplication could be supported by replacing it with ReinterpretQuantize
        // but due to zero points of input and output info being not equal we get multiplication
        // as unsupported operation.
        // Additionally, verifying that the correct MultiplicationSupportedMode value is returned
        ExpectFail(TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0),
                   TensorInfo({ 1, 1, 1, 1 }, DataType::QAsymmU8, 0.009f, 0),
                   TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 1),
                   "Input and output quantization offsets are not equal");

        // Failure case - multiplication could be supported by replacing it with ReinterpretQuantize
        // but due to data types of input and output info being not equal we get multiplication
        // as unsupported operation.
        // Additionally, verifying that the correct MultiplicationSupportedMode value is returned
        ExpectFail(TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0),
                   TensorInfo({ 1, 1, 1, 1 }, DataType::QAsymmU8, 1.0f, 1),
                   TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmS8, 1.0f, 0), "Provided outputInfo is incorrect");

        // Failure case - multiplication not supported
        // Additionally, verifying that the correct MultiplicationSupportedMode value is returned
        CHECK(layerSupport.GetMultiplicationSupportedMode(TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0),
                                                          TensorInfo({ 1, 2, 2, 1 }, DataType::QAsymmU8, 0.009f, 0),
                                                          TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0)) ==
              EthosNLayerSupport::MultiplicationSupportedMode::None);

        // Failure case - broadcasting in an a way that can't be covered by the replacement
        ExpectFail(TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0),
                   TensorInfo({ 1, 2, 2, 1 }, DataType::QAsymmU8, 0.9f, 0),
                   TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0), "");

        // Failure case - could be replaced by depthwise but we can't find a valid weight scale
        ExpectFail(
            TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 100000.0f, 0),
            TensorInfo({ 1, 1, 1, 4 }, DataType::QAsymmU8, 0.9f, 0),
            TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0),
            "Multiplication operation is not supported on Arm Ethos-N NPU backend and an attempt was made to "
            "substitute "
            "for DepthwiseConvolution2d, however the following error occurred when checking for Depthwise support: "
            "Depthwise Convolution: Overall scale (of the input * weights / output) should be in the range");

        // Failure case - could be replaced by reinterpret quantize but support library rejects the reinterpret quantize
        // config (in this case, input tensor too deep)
        ExpectFail(TensorInfo({ 1, 2, 2, 100000 }, DataType::QAsymmU8, 1.0f, 0),
                   TensorInfo({ 1, 1, 1, 1 }, DataType::QAsymmU8, 0.009f, 0),
                   TensorInfo({ 1, 2, 2, 100000 }, DataType::QAsymmU8, 1.0f, 0),
                   "Input to reinterpret quantization: Tensor max depth cannot fit in SRAM");

        // Failure case - could be replaced by depthwise but support library rejects the depthwise config
        // (in this case, input tensor too deep)
        ExpectFail(TensorInfo({ 1, 2, 2, 100000 }, DataType::QAsymmU8, 1.0f, 0),
                   TensorInfo({ 1, 1, 1, 100000 }, DataType::QAsymmU8, 0.9f, 0),
                   TensorInfo({ 1, 2, 2, 100000 }, DataType::QAsymmU8, 1.0f, 0),
                   "Multiplication operation is not supported on Arm Ethos-N NPU backend and an attempt was made to "
                   "substitute for DepthwiseConvolution2d, however the following error occurred when checking for "
                   "Depthwise support: Input to depthwise conv: Tensor max depth cannot fit in SRAM");
    }

    TEST_CASE("IsMultiplicationSupportedPerfOnly")
    {
        EthosNConfig config;
        config.m_PerfOnly = true;
        EthosNLayerSupport layerSupport(config, EthosNMappings(), config.QueryCapabilities());

        // Success case - multiplication supported by replacing it with Depthwise
        // Additionally, Verifying that the correct MultiplicationSupportedMode value is returned
        CHECK(layerSupport.GetMultiplicationSupportedMode(TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0),
                                                          TensorInfo({ 1, 1, 1, 4 }, DataType::QAsymmU8, 0.9f, 0),
                                                          TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0)) ==
              EthosNLayerSupport::MultiplicationSupportedMode::ReplaceWithDepthwise);

        // Success case - multiplication supported by replacing it with ReinterpretQuantize
        // Additionally, Verifying that the correct MultiplicationSupportedMode value is returned
        CHECK(layerSupport.GetMultiplicationSupportedMode(TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0),
                                                          TensorInfo({ 1, 1, 1, 1 }, DataType::QAsymmU8, 0.009f, 0),
                                                          TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0)) ==
              EthosNLayerSupport::MultiplicationSupportedMode::ReplaceWithReinterpretQuantize);

        // Success case - multiplication supported in EstimateOnly mode
        // Additionally, Verifying that the correct MultiplicationSupportedMode value is returned
        CHECK(layerSupport.GetMultiplicationSupportedMode(TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0),
                                                          TensorInfo({ 1, 2, 2, 1 }, DataType::QAsymmU8, 0.009f, 0),
                                                          TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0)) ==
              EthosNLayerSupport::MultiplicationSupportedMode::EstimateOnly);
    }

    TEST_CASE("IsAdditionSupported")
    {
        EthosNLayerSupport layerSupport(EthosNConfig(), EthosNMappings(), EthosNConfig().QueryCapabilities());

        auto ExpectFail = [&layerSupport](const TensorInfo& input0, const TensorInfo& input1, const TensorInfo& output,
                                          const char* expectedFailureReason) {
            std::string failureReason;
            CHECK(!layerSupport.IsAdditionSupported(input0, input1, output, failureReason));
            CHECK(failureReason.find(expectedFailureReason) != std::string::npos);
        };

        // Failure case - 5D tensor
        ExpectFail(TensorInfo({ 1, 2, 2, 4, 9 }, DataType::QAsymmU8, 1.0f, 0),
                   TensorInfo({ 1, 1, 1, 4 }, DataType::Signed32, 1.0f, 0),
                   TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 0.9f, 0),
                   "The ethosn can only support up to 4D tensors");

        // Success case - regular addition supported natively
        CHECK(layerSupport.GetAdditionSupportedMode(TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0),
                                                    TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0),
                                                    TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0)) ==
              EthosNLayerSupport::AdditionSupportedMode::Native);

        // Success case - addition supported by replacing it with ReinterpretQuantize
        // Additionally, verifying that the correct AdditionSupportedMode value is returned
        CHECK(layerSupport.GetAdditionSupportedMode(TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0),
                                                    TensorInfo({ 1, 1, 1, 1 }, DataType::QAsymmU8, 1.0f, 0),
                                                    TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0)) ==
              EthosNLayerSupport::AdditionSupportedMode::ReplaceWithReinterpretQuantize);

        // Failure case - addition could be supported by replacing it with ReinterpretQuantize
        // but due to quantization scales of input and output info being not equal we get addition
        // as unsupported operation.
        ExpectFail(TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0),
                   TensorInfo({ 1, 1, 1, 1 }, DataType::QAsymmU8, 1.0f, 0),
                   TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 2.0f, 1),
                   "Input and output quantization scales are not equal");

        // Failure case - broadcasting in an a way that can't be covered by the depthwise replacement
        ExpectFail(TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0),
                   TensorInfo({ 1, 2, 2, 1 }, DataType::QAsymmU8, 1.0f, 0),
                   TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0),
                   "Cannot stretch along the requested dimensions.");

        // Failure case - could be replaced by depthwise but we can't find a valid weight scale
        ExpectFail(TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 100000.0f, 0),
                   TensorInfo({ 1, 1, 1, 4 }, DataType::QAsymmU8, 1.0f, 0),
                   TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0),
                   "Addition operation was attempted to be substituted for DepthwiseConvolution2d, "
                   "however the following error occurred in the substitution: Couldn't find valid weight scale");

        // Failure case - could be replaced by reinterpret quantize but support library rejects the reinterpret quantize
        // config (in this case, input tensor too deep)
        ExpectFail(TensorInfo({ 1, 2, 2, 100000 }, DataType::QAsymmU8, 1.0f, 0),
                   TensorInfo({ 1, 1, 1, 1 }, DataType::QAsymmU8, 1.0f, 0),
                   TensorInfo({ 1, 2, 2, 100000 }, DataType::QAsymmU8, 1.0f, 0),
                   "Input to reinterpret quantization: Tensor max depth cannot fit in SRAM");

        // Failure case - could be replaced by depthwise but support library rejects the depthwise config
        // (in this case, input tensor too deep)
        ExpectFail(TensorInfo({ 1, 2, 2, 100000 }, DataType::QAsymmU8, 1.0f, 0),
                   TensorInfo({ 1, 1, 1, 100000 }, DataType::QAsymmU8, 1.0f, 0),
                   TensorInfo({ 1, 2, 2, 100000 }, DataType::QAsymmU8, 1.0f, 0),
                   "Addition operation was attempted to be substituted for DepthwiseConvolution2d, "
                   "however the following error occurred when checking for Depthwise support: "
                   "Input to depthwise conv: Tensor max depth cannot fit in SRAM");

        // Success case - supported by replacement with depthwise
        CHECK(layerSupport.GetAdditionSupportedMode(TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0),
                                                    TensorInfo({ 1, 1, 1, 4 }, DataType::QAsymmU8, 1.0f, 0),
                                                    TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0)) ==
              EthosNLayerSupport::AdditionSupportedMode::ReplaceWithDepthwise);

        // Success case - supported by replacement with Reinterpret Quantization
        CHECK(layerSupport.GetAdditionSupportedMode(TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0),
                                                    TensorInfo({ 1, 1, 1, 1 }, DataType::QAsymmU8, 1.0f, 0),
                                                    TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0)) ==
              EthosNLayerSupport::AdditionSupportedMode::ReplaceWithReinterpretQuantize);
    }

    // Checks the behaviour is IsAdditionSupported when in perf-only mode.
    // Because we call multiple support library IsSupported checks (due to the potential depthwise replacement),
    // the logic relating to perf-only is a bit complicated and warrants explicit testing.
    TEST_CASE("IsAdditionSupportedPerfOnly")
    {
        EthosNConfig config;
        config.m_PerfOnly = true;
        EthosNLayerSupport layerSupport(config, EthosNMappings(), config.QueryCapabilities());

        // Broadcast add (over width & height) is reported as EstimateOnly by the support library,
        // but by replacing it with a depthwise we can support it fully, which is preferable.
        // Therefore GetAdditionSupportedMode should request replacement with a depthwise even in perf-only mode.
        CHECK(layerSupport.GetAdditionSupportedMode(TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0),
                                                    TensorInfo({ 1, 1, 1, 4 }, DataType::QAsymmU8, 1.0f, 0),
                                                    TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0)) ==
              EthosNLayerSupport::AdditionSupportedMode::ReplaceWithDepthwise);

        // A case where native Addition is not supported at all (even in EstimateOnly, because the input data types are
        // different), but replacement with depthwise can be done
        CHECK(layerSupport.GetAdditionSupportedMode(TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0),
                                                    TensorInfo({ 1, 1, 1, 4 }, DataType::QAsymmS8, 1.0f, 0),
                                                    TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0)) ==
              EthosNLayerSupport::AdditionSupportedMode::ReplaceWithDepthwise);

        // Native addition is EstimateOnly (broadcast across channels) and no depthwise replacement possible
        // because it's not the right kind of broadcast (this is NOT a case where the support library's IsDepthwiseSupported
        // fails - it doesn't even get this far).
        CHECK(layerSupport.GetAdditionSupportedMode(TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0),
                                                    TensorInfo({ 1, 2, 2, 1 }, DataType::QAsymmU8, 1.0f, 0),
                                                    TensorInfo({ 1, 2, 2, 4 }, DataType::QAsymmU8, 1.0f, 0)) ==
              EthosNLayerSupport::AdditionSupportedMode::Native);

        // There are some theoretically possible cases that can't be tested in practice because of the current
        // support library IsSupported checks. If we mocked the support library IsSupported checks then we could better
        // test the logic of the backend's GetAdditionSupportedMode() (a potential future improvement):
        //
        // 1. I do not believe it is currently possible for the depthwise to be EstimateOnly - it is either fully supported or
        // not supported at all, because the depthwise layer that we would replace with never uses any weird strides
        // or anything like that. Hence there are no tests for this case.
        //
        // 2. I do not believe it is currently possible for the native Addition to be EstimateOnly and the replacement
        // depthwise to be rejected by the support library, because the only way I can find to make the depthwise
        // rejected is to have a large tensor depth, but then would also results in the native Addition to be rejected.
    }

    TEST_CASE("IsDepthwiseConvolutionSupported")
    {
        EthosNLayerSupport layerSupport(EthosNConfig(), EthosNMappings(), EthosNConfig().QueryCapabilities());
        auto ExpectFail = [&layerSupport](const TensorInfo& input, const TensorInfo& output,
                                          const DepthwiseConvolution2dDescriptor& descriptor, const TensorInfo& weights,
                                          const Optional<TensorInfo>& biases, const char* expectedFailureReason) {
            std::string failureReason;
            CHECK(!layerSupport.IsDepthwiseConvolutionSupported(input, output, descriptor, weights, biases,
                                                                failureReason));
            CHECK(failureReason.find(expectedFailureReason) != std::string::npos);
        };

        const TensorInfo inputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
        const TensorInfo outputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
        const TensorInfo weightInfo({ 1, 1, 1, 16 }, DataType::QAsymmU8, 0.9f, 0);
        const TensorInfo biasInfo({ 1, 1, 1, 16 }, DataType::Signed32, 0.9f, 0);

        DepthwiseConvolution2dDescriptor depthwiseConvolutionDescriptor;
        depthwiseConvolutionDescriptor.m_BiasEnabled = true;
        depthwiseConvolutionDescriptor.m_DataLayout  = DataLayout::NHWC;
        depthwiseConvolutionDescriptor.m_StrideX     = 1;
        depthwiseConvolutionDescriptor.m_StrideY     = 1;

        SUBCASE("Working IsDepthwiseConvolutionSupported()")
        {
            // Check a good case
            std::string failureReason;
            CHECK(layerSupport.IsDepthwiseConvolutionSupported(inputInfo, outputInfo, depthwiseConvolutionDescriptor,
                                                               weightInfo, biasInfo, failureReason));
        }

        SUBCASE("IsDepthwiseConvolutionSupported() Don't handle 16 bit")
        {
            const TensorInfo inputInfo16({ 1, 16, 16, 16 }, DataType::QSymmS16, 1.0f, 0);
            const TensorInfo outputInfo16({ 1, 16, 16, 16 }, DataType::QSymmS16, 1.0f, 0);
            const TensorInfo weightInfo16({ 1, 1, 1, 16 }, DataType::QSymmS16, 0.9f, 0);
            const TensorInfo biasInfo16({ 1, 1, 1, 16 }, DataType::QSymmS16, 0.9f, 0);
            ExpectFail(inputInfo16, outputInfo, depthwiseConvolutionDescriptor, weightInfo, biasInfo,
                       "Unsupported data type: QSymm16");
            ExpectFail(inputInfo, outputInfo16, depthwiseConvolutionDescriptor, weightInfo, biasInfo,
                       "Unsupported data type: QSymm16");
            ExpectFail(inputInfo, outputInfo, depthwiseConvolutionDescriptor, weightInfo16, biasInfo,
                       "Unsupported data type: QSymm16");
            ExpectFail(inputInfo, outputInfo, depthwiseConvolutionDescriptor, weightInfo, biasInfo16,
                       "Unsupported data type: QSymm16");
        }

        SUBCASE("IsDepthwiseConvolutionSupported() only handle NHWC")
        {
            DepthwiseConvolution2dDescriptor depthwiseConvolutionDescriptorNCHW;
            depthwiseConvolutionDescriptorNCHW.m_BiasEnabled = true;
            depthwiseConvolutionDescriptorNCHW.m_DataLayout  = DataLayout::NCHW;
            depthwiseConvolutionDescriptorNCHW.m_StrideX     = 1;
            depthwiseConvolutionDescriptorNCHW.m_StrideY     = 1;
            ExpectFail(inputInfo, outputInfo, depthwiseConvolutionDescriptorNCHW, weightInfo, biasInfo,
                       "DataLayout must be NHWC");
        }

        SUBCASE("IsDepthwiseConvolutionSupported() should not handle PerAxisQuantization on dim other than O (I*M)")
        {
            TensorInfo weightInfoPerAxisQuantization({ 1, 1, 1, 16 }, DataType::QAsymmU8, 0.9f, 0);
            weightInfoPerAxisQuantization.SetQuantizationDim(Optional<unsigned int>(2));
            ExpectFail(inputInfo, outputInfo, depthwiseConvolutionDescriptor, weightInfoPerAxisQuantization, biasInfo,
                       "Can't convert tensor from [1,H,W,Cout] to [H,W,Cin,M] when per channel "
                       "quantization is applied on a dimension other than the last, or M != 1.");
        }

        SUBCASE("IsDepthwiseConvolutionSupported() should not handle PerAxisQuantization when M != 1")
        {
            TensorInfo weightInfoPerAxisQuantization({ 1, 1, 1, 16 }, DataType::QAsymmU8, 0.9f, 0);
            const TensorInfo inputInfo8Channels({ 1, 16, 16, 8 }, DataType::QAsymmU8, 1.0f, 0);
            weightInfoPerAxisQuantization.SetQuantizationDim(Optional<unsigned int>(3));
            ExpectFail(inputInfo8Channels, outputInfo, depthwiseConvolutionDescriptor, weightInfoPerAxisQuantization,
                       biasInfo,
                       "Can't convert tensor from [1,H,W,Cout] to [H,W,Cin,M] when per channel "
                       "quantization is applied on a dimension other than the last, or M != 1.");
        }
    }
}
