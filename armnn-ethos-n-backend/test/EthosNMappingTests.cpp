//
// Copyright © 2019-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "EthosNConfig.hpp"
#include "EthosNLayerSupport.hpp"
#include "EthosNMapping.hpp"
#include "EthosNTestUtils.hpp"

#include <EthosNBackend.hpp>
#include <EthosNBackendId.hpp>
#include <Network.hpp>
#include <backendsCommon/test/CommonTestUtils.hpp>
#include <boost/test/unit_test.hpp>
#include <test/EthosNTestUtils.hpp>

#include <sstream>

using Tensors  = std::map<std::string, armnn::SimpleInputOutput>;
using Layers   = std::vector<armnn::SimpleLayer>;
using Shape    = std::vector<uint32_t>;
using Mappings = std::vector<armnn::Mapping>;

using namespace armnn;
using namespace testing_utils;

namespace
{

struct TestLayerType
{
    LayerType layer;
    // For ActivationLayer, this will be the name of the activation function
    std::string name;
};

// Creates mappings for substitution
Mappings CreateSubstitutionMappings(TestLayerType original, TestLayerType replacement)
{
    Mappings mappings;
    Layers orgLayers, replacementLayers;

    auto shape   = Shape({ 1, 16, 16, 16 });
    auto tensors = Tensors({ std::make_pair("firstInput", armnn::SimpleInputOutput("firstInput", shape)),
                             std::make_pair("firstOutput", armnn::SimpleInputOutput("firstOutput", shape)) });

    std::map<std::string, ActivationFunction> mapStringToActivationFunction =
        armnn::ethosnbackend::GetMapStringToActivationFunction();
    if (!original.name.empty())
    {
        BOOST_TEST((mapStringToActivationFunction.find(original.name) != mapStringToActivationFunction.end()));
    }

    if (!replacement.name.empty())
    {
        BOOST_TEST((mapStringToActivationFunction.find(replacement.name) != mapStringToActivationFunction.end()));
    }

    switch (original.layer)
    {
        case LayerType::Activation:
            BOOST_TEST(original.name.empty() != true);
            orgLayers =
                Layers({ armnn::SimpleLayer("Activation", { armnn::SimpleInputOutput("firstInput", shape) },
                                            { "firstOutput" }, { std::make_pair("function", original.name) }) });
            break;
        default:
            BOOST_ASSERT("Unsupported substititable layer type\n");
            break;
    }

    switch (replacement.layer)
    {
        case LayerType::Activation:
            BOOST_TEST(replacement.name.empty() != true);
            replacementLayers =
                Layers({ armnn::SimpleLayer("Activation", { armnn::SimpleInputOutput("firstInput", shape) },
                                            { "firstOutput" }, { std::make_pair("function", replacement.name) }) });
            break;
        case LayerType::Convolution2d:
            replacementLayers = Layers({ armnn::SimpleLayer(
                "Convolution2d", { armnn::SimpleInputOutput("firstInput", shape) }, { "firstOutput" }, {}) });
            break;
        case LayerType::TransposeConvolution2d:
            replacementLayers = Layers({ armnn::SimpleLayer(
                "TransposeConvolution2d", { armnn::SimpleInputOutput("firstInput", shape) }, { "firstOutput" }, {}) });
            break;
        default:
            BOOST_ASSERT("Unsupported replacement layer type\n");
            break;
    }

    mappings.emplace_back(tensors, orgLayers, replacementLayers);
    return mappings;
}

std::string CreateExclusionMappings(const EthosNConfig config = EthosNConfig())
{
    std::string mappings;

    mappings += "pattern:\n";
    mappings += "input firstInput 1x16x16x16\n";
    mappings += "output firstOutput 1x_x_x_\n";
    mappings += "Activation (firstInput) (firstOutput) (function=TanH)\n";
    mappings += "graph-replacement:\n";
    mappings += "Excluded (firstInput) (firstOutput)\n";
    mappings += "pattern:\n";
    mappings += "input firstInput 1x_x_x_\n";
    mappings += "output firstOutput 1x_x_x_\n";
    mappings += "StandIn (firstInput) (firstOutput)\n";
    mappings += "graph-replacement:\n";
    mappings += "Excluded (firstInput) (firstOutput)\n";

    std::ofstream mappingStream(config.m_PerfMappingFile);
    if (mappingStream.is_open())
    {
        mappingStream << mappings;
    }

    return mappings;
}

EthosNConfig CreateEthosNConfig(TempDir& tmpDir)
{
    const std::string configFile = tmpDir.Str() + "/config.txt";
    {
        std::ofstream configStream(configFile);
        armnn::EthosNConfig config;
        config.m_PerfOnly        = true;
        config.m_PerfMappingFile = tmpDir.Str() + "/mapping.txt";
        configStream << config;
        std::ofstream mappingStream(config.m_PerfMappingFile);
        mappingStream << "";
    }
    SetEnv(armnn::EthosNConfig::CONFIG_FILE_ENV, configFile.c_str());
    return armnn::GetEthosNConfig();
}

void CreateUnoptimizedNetwork(INetwork& net)
{
    armnn::IConnectableLayer* const inputLayer = net.AddInputLayer(0, "input layer");
    BOOST_TEST(inputLayer);

    // Arm NN weights tensor shape is OHWI (out channels, height, width, in channels) for NHWC
    const TensorInfo convTensorInfo(TensorShape({ 1, 16, 16, 16 }), DataType::QAsymmU8, 0.9f, 0);
    const TensorInfo convWeightsInfo(TensorShape({ 16, 1, 1, 16 }), armnn::DataType::QAsymmU8, 0.9f, 0);
    const std::vector<uint8_t> convWeightsData(convWeightsInfo.GetNumElements());
    const armnn::ConstTensor convWeights(convWeightsInfo, convWeightsData);

    armnn::Convolution2dDescriptor convDesc{};
    convDesc.m_StrideX    = 1;
    convDesc.m_StrideY    = 1;
    convDesc.m_DataLayout = armnn::DataLayout::NHWC;

    armnn::IConnectableLayer* convLayer =
        net.AddConvolution2dLayer(convDesc, convWeights, EmptyOptional(), "convolution layer");

    ActivationDescriptor tanDesc{};
    tanDesc.m_A                               = 100;
    tanDesc.m_B                               = 0;
    tanDesc.m_Function                        = ActivationFunction::TanH;
    armnn::IConnectableLayer* const tanhLayer = net.AddActivationLayer(tanDesc, "TanH layer");
    BOOST_TEST(tanhLayer);

    armnn::IConnectableLayer* const outputLayer = net.AddOutputLayer(0, "output layer");
    BOOST_TEST(outputLayer);

    TensorInfo inputTensorInfo(TensorShape({ 1, 16, 16, 16 }), armnn::DataType::QAsymmU8);
    inputTensorInfo.SetQuantizationOffset(0);
    inputTensorInfo.SetQuantizationScale(0.9f);

    TensorInfo outputTensorInfo(TensorShape({ 1, 16, 16, 16 }), armnn::DataType::QAsymmU8);
    outputTensorInfo.SetQuantizationOffset(0);
    outputTensorInfo.SetQuantizationScale(1.0f / 256);

    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    convLayer->GetOutputSlot(0).SetTensorInfo(convTensorInfo);
    convLayer->GetOutputSlot(0).Connect(tanhLayer->GetInputSlot(0));

    tanhLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    tanhLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);
}

}    // namespace

BOOST_AUTO_TEST_SUITE(EthosNMapping)

//
// Tests that the Ethos-N  mapping file is parsed correctly
//

BOOST_AUTO_TEST_CASE(TestTrimm)
{
    BOOST_TEST(armnn::Trim(std::string("")).size() == 0);
    BOOST_TEST(armnn::Trim(std::string("\t ")).size() == 0);
    BOOST_TEST(armnn::Trim(std::string(" pattern:\t")) == std::string("pattern:"));
    BOOST_TEST(armnn::Trim(std::string("input firstInput, 1x_x_x_  \n\t")) == std::string("input firstInput, 1x_x_x_"));
}

BOOST_AUTO_TEST_CASE(TestPrune)
{
    // Given
    std::string s = "\n\tHello, world! \r";

    // When
    armnn::Prune(s);

    // Then
    BOOST_TEST(s == std::string("Hello,world!"));
}

BOOST_AUTO_TEST_CASE(TestProcessPattern)
{
    // Given
    const std::vector<std::string> buf1 = {
        "input firstInput, 1x_x_x_",
        "\toutput  firstOutput, 1x_x_x_",
        "Activation, (firstInput), (firstOutput), (function=TanH)",
    };
    Tensors tensors1;
    Layers layers1;
    const std::vector<std::string> buf2 = {
        "input firstInput, 1x_x_x_",
        "input secondInput, 1x1x2x3",
        "output firstOutput, 1x_x_x_",
        "StandIn, (firstInput, secondInput), (firstOutput), (\tname= CustomOp)",
    };
    Tensors tensors2;
    Layers layers2;
    const std::vector<std::string> buf3 = {
        "input firstInput, 1x_x_x_",
        "output  firstOutput, 1x_x_x_",
        "output  secondOutput, 1x_x_x_",
        "Excluded, (firstInput), (firstOutput, secondOutput)",
    };
    Tensors tensors3;
    Layers layers3;

    // When
    armnn::ProcessPattern(buf1, tensors1, layers1);

    // Then
    BOOST_TEST(
        tensors1 ==
        Tensors({ std::make_pair("firstInput", armnn::SimpleInputOutput("firstInput", Shape({ 1, 0, 0, 0 }))),
                  std::make_pair("firstOutput", armnn::SimpleInputOutput("firstOutput", Shape({ 1, 0, 0, 0 }))) }));
    BOOST_TEST(layers1 == Layers({ armnn::SimpleLayer("Activation",
                                                      { armnn::SimpleInputOutput("firstInput", Shape({ 1, 0, 0, 0 })) },
                                                      { "firstOutput" }, { std::make_pair("function", "TanH") }) }));

    // And when
    armnn::ProcessPattern(buf2, tensors2, layers2);

    // Then
    BOOST_TEST(
        tensors2 ==
        Tensors({ std::make_pair("firstInput", armnn::SimpleInputOutput("firstInput", Shape({ 1, 0, 0, 0 }))),
                  std::make_pair("secondInput", armnn::SimpleInputOutput("secondInput", Shape({ 1, 1, 2, 3 }))),
                  std::make_pair("firstOutput", armnn::SimpleInputOutput("firstOutput", Shape({ 1, 0, 0, 0 }))) }));
    BOOST_TEST(layers2 ==
               Layers({ armnn::SimpleLayer("StandIn",
                                           { armnn::SimpleInputOutput("firstInput", Shape({ 1, 0, 0, 0 })),
                                             armnn::SimpleInputOutput("secondInput", Shape({ 1, 1, 2, 3 })) },
                                           { "firstOutput" }, { std::make_pair("name", "CustomOp") }) }));

    // And when
    armnn::ProcessPattern(buf3, tensors3, layers3);

    // Then
    BOOST_TEST(
        tensors3 ==
        Tensors({ std::make_pair("firstInput", armnn::SimpleInputOutput("firstInput", Shape({ 1, 0, 0, 0 }))),
                  std::make_pair("firstOutput", armnn::SimpleInputOutput("firstOutput", Shape({ 1, 0, 0, 0 }))),
                  std::make_pair("secondOutput", armnn::SimpleInputOutput("secondOutput", Shape({ 1, 0, 0, 0 }))) }));
    BOOST_TEST(layers3 == Layers({ armnn::SimpleLayer("Excluded",
                                                      { armnn::SimpleInputOutput("firstInput", Shape({ 1, 0, 0, 0 })) },
                                                      { "firstOutput", "secondOutput" }, {}) }));
}

BOOST_AUTO_TEST_CASE(TestProcessBadInput)
{
    const std::vector<std::string> buf = {
        "input_ firstInput, 1x_x_x_",
        "output?  firstOutput, 1x_x_x_",
        "Activation, (firstInput), (firstOutput), (function=TanH)",
    };
    Tensors tensors;
    Layers layers;

    BOOST_CHECK_THROW(armnn::ProcessPattern(buf, tensors, layers), armnn::ParseException);

    try
    {
        armnn::ProcessPattern(buf, tensors, layers);
    }
    catch (const armnn::ParseException& e)
    {
        std::string err = "Syntax error:\ninput_ firstInput, 1x_x_x_\nSyntax error:\noutput?  firstOutput, "
                          "1x_x_x_\nUndefined input: 'firstInput'\n";
        BOOST_CHECK_EQUAL(err, e.what());
    }
}

Mappings CreateMappings(TestLayerType originalType, TestLayerType replacementType)
{
    Mappings ethosNMappings;

    std::map<std::string, LayerType> mapStringToLayerType = armnn::ethosnbackend::GetMapStringToLayerType();
    std::map<std::string, ActivationFunction> mapStringToActivationFunction =
        armnn::ethosnbackend::GetMapStringToActivationFunction();

    ethosNMappings = CreateSubstitutionMappings(originalType, replacementType);

    //Test if the mapping layer types are as intended
    BOOST_TEST(((mapStringToLayerType.find(ethosNMappings[0].m_ReplacementLayers[0].m_Name)->second) ==
                replacementType.layer));
    BOOST_TEST(
        ((mapStringToLayerType.find(ethosNMappings[0].m_PatternLayers[0].m_Name))->second == originalType.layer));

    //Test for single layer mappings
    BOOST_TEST((ethosNMappings.size() == 1));
    BOOST_TEST((ethosNMappings[0].m_PatternLayers.size() == 1));
    BOOST_TEST((ethosNMappings[0].m_ReplacementLayers.size() == 1));

    return ethosNMappings;
}

armnn::SubgraphView::SubgraphViewPtr
    CreateUnoptimizedSubgraph(Graph& graph, LayerType orgLayerType, Mappings& ethosNMappings)
{
    const TensorInfo inputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo outputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    Layer *inputLayer, *outputLayer, *operationLayer;

    if (orgLayerType == LayerType::Activation)
    {
        std::string activationFuncOriginalLayer =
            ethosNMappings[0].m_PatternLayers[0].m_ExtraArgs.find("function")->second;

        operationLayer = ethosnbackend::CreateActivationLayer(graph, activationFuncOriginalLayer);
    }
    else if ((orgLayerType == LayerType::Convolution2d) || (orgLayerType == LayerType::TransposeConvolution2d))
    {
        unsigned int inputChannels = inputInfo.GetShape()[3];
        DataType weightDataType    = inputInfo.GetDataType();
        operationLayer =
            ethosnbackend::CreateConvolutionLayer(ethosNMappings[0].m_ReplacementLayers[0].m_Name, graph, inputChannels,
                                                  1, 1, 1, 1, weightDataType, DataType::Signed32);
    }

    BOOST_TEST(operationLayer);
    operationLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    // Construct the graph
    inputLayer = graph.AddLayer<InputLayer>(0, "input layer");
    BOOST_TEST(inputLayer);
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    outputLayer = graph.AddLayer<OutputLayer>(0, "output layer");
    BOOST_TEST(outputLayer);

    // Connect the network
    inputLayer->GetOutputSlot(0).Connect(operationLayer->GetInputSlot(0));
    operationLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Create the subgraph view for the whole network
    return CreateSubgraphViewFrom(CreateInputsFrom({ operationLayer }), CreateOutputsFrom({ operationLayer }),
                                  { operationLayer });
}

// This function assumes that there is only one operation layer in the subgraph.
// That is because CreateUnoptimizedSubgraph() creates a subgraph with one input
// layer , one operation layer and one output layer. If in future, we want to
// validate subgraphs with multiple operation layers, then this function should
// be changed accordingly.
bool IsLayerPresentInSubgraph(Graph& graph, LayerType replacementType)
{
    const std::list<Layer*> newGraphLayers(graph.begin(), graph.end());

    for (Layer* layer : newGraphLayers)
    {
        if (layer->GetType() == replacementType)
        {
            return true;
        }
    }
    return false;
}

void TestSubgraphSubstitution(TestLayerType originalType, TestLayerType replacementType)
{
    using namespace testing_utils;
    Graph graph, graph2;

    TempDir tmpDir;
    EthosNConfig ethosnConfig = CreateEthosNConfig(tmpDir);
    auto backendObjPtr        = CreateBackendObject(EthosNBackendId());
    BOOST_TEST((backendObjPtr != nullptr));

    auto ethosNMappings = CreateMappings(originalType, replacementType);

    auto subGraphOriginal  = CreateUnoptimizedSubgraph(graph, originalType.layer, ethosNMappings);
    auto subGraphOriginal2 = CreateUnoptimizedSubgraph(graph2, originalType.layer, ethosNMappings);
    //Validate that the graph2 had the layer of the original type
    BOOST_TEST(IsLayerPresentInSubgraph(graph2, originalType.layer));

    // When
    OptimizationViews optimizationViews;
    armnn::CreatePreCompiledLayerInGraph(optimizationViews, *subGraphOriginal, ethosNMappings);
    ethosnbackend::ApplyMappings(ethosNMappings, graph2);

    // Then validate that armnn was able to compile the graph successfully
    BOOST_TEST(optimizationViews.Validate(*subGraphOriginal));
    BOOST_TEST(optimizationViews.GetSubstitutions().size() == 1);
    BOOST_TEST(optimizationViews.GetFailedSubgraphs().size() == 0);
    BOOST_TEST(optimizationViews.GetUntouchedSubgraphs().size() == 0);
    auto substitutions = optimizationViews.GetSubstitutions();
    BOOST_TEST(substitutions.size() == 1);
    bool subgraphsAreSame = (*subGraphOriginal == substitutions[0].m_SubstitutableSubgraph);
    BOOST_TEST(subgraphsAreSame);
    //Currently we replace a single layer with another single layer
    BOOST_TEST((substitutions[0].m_ReplacementSubgraph.GetLayers().size() == 1));

    // Validate that the substitution really took place. We need to do this as armnn
    // changes the layer type to pre-compiled
    BOOST_TEST(IsLayerPresentInSubgraph(graph2, replacementType.layer));
}

BOOST_AUTO_TEST_CASE(TestAllSubgraphSubstitution)
{
    TestLayerType org, replacement;

    org.layer         = LayerType::Activation;
    org.name          = "BoundedReLu";
    replacement.layer = LayerType::Activation;
    replacement.name  = "Sigmoid";
    TestSubgraphSubstitution(org, replacement);

    org.layer         = LayerType::Activation;
    org.name          = "BoundedReLu";
    replacement.layer = LayerType::Convolution2d;
    replacement.name  = "";
    TestSubgraphSubstitution(org, replacement);

    org.layer         = LayerType::Activation;
    org.name          = "BoundedReLu";
    replacement.layer = LayerType::Activation;
    replacement.name  = "ReLu";
    TestSubgraphSubstitution(org, replacement);

    org.layer         = LayerType::Activation;
    org.name          = "TanH";
    replacement.layer = LayerType::TransposeConvolution2d;
    replacement.name  = "";
    TestSubgraphSubstitution(org, replacement);
}

BOOST_AUTO_TEST_CASE(TestLayerInclusion)
{
    // Given
    TempDir tmpDir;
    armnn::g_EthosNConfig = CreateEthosNConfig(tmpDir);
    TensorInfo inputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    TensorInfo outputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f / 256, 0);
    ActivationDescriptor activationDescriptor;
    StandInDescriptor standInDescriptor{ 1, 1 };
    std::string reason;

    // When
    auto backendObjPtr = CreateBackendObject(EthosNBackendId());
    BOOST_TEST((backendObjPtr != nullptr));
    auto layerSupport = backendObjPtr->GetLayerSupport();

    // Then
    BOOST_TEST(layerSupport->IsActivationSupported(inputInfo, outputInfo, activationDescriptor, reason) == true);
    BOOST_TEST(reason.empty());
    BOOST_TEST(layerSupport->IsStandInSupported(std::vector<const TensorInfo*>{ &inputInfo },
                                                std::vector<const TensorInfo*>{ &outputInfo }, standInDescriptor,
                                                reason) == true);
    BOOST_TEST(reason.empty());
}

BOOST_AUTO_TEST_CASE(TestLayerExclusion)
{
    // Given
    TempDir tmpDir;
    armnn::g_EthosNConfig = CreateEthosNConfig(tmpDir);
    CreateExclusionMappings(armnn::g_EthosNConfig);
    armnn::g_EthosNMappings = GetMappings(armnn::g_EthosNConfig.m_PerfMappingFile);
    TensorInfo inputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    TensorInfo outputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f / 256, 0);
    ActivationDescriptor activationDescriptor1;
    ActivationDescriptor activationDescriptor2;
    activationDescriptor1.m_Function = ActivationFunction::Sigmoid;
    activationDescriptor2.m_Function = ActivationFunction::TanH;
    StandInDescriptor standInDescriptor{ 1, 1 };
    std::string reason;

    // When
    auto backendObjPtr = CreateBackendObject(EthosNBackendId());
    BOOST_TEST((backendObjPtr != nullptr));
    auto layerSupport = backendObjPtr->GetLayerSupport();

    // Then
    BOOST_TEST(layerSupport->IsActivationSupported(inputInfo, outputInfo, activationDescriptor1, reason) == true);
    BOOST_TEST(layerSupport->IsActivationSupported(inputInfo, outputInfo, activationDescriptor2, reason) == false);
    BOOST_TEST(reason == "Layer declared excluded in mapping file");
    BOOST_TEST(layerSupport->IsStandInSupported(std::vector<const TensorInfo*>{ &inputInfo },
                                                std::vector<const TensorInfo*>{ &outputInfo }, standInDescriptor,
                                                reason) == false);
    BOOST_TEST(reason == "Layer declared excluded in mapping file");
}

BOOST_AUTO_TEST_CASE(TestLayerExclusionViaArmnn)
{
    // Given
    TempDir tmpDir;
    EthosNConfig ethosnConfig = CreateEthosNConfig(tmpDir);
    CreateExclusionMappings(ethosnConfig);
    auto backendObjPtr = CreateBackendObject(EthosNBackendId());
    BOOST_TEST((backendObjPtr != nullptr));
    INetworkPtr net(INetwork::Create());
    CreateUnoptimizedNetwork(*net);

    // When
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));
    std::vector<BackendId> backends = { EthosNBackendId(), "CpuRef" };
    IOptimizedNetworkPtr optNet     = Optimize(*net, backends, runtime->GetDeviceSpec());

    // Then
    IOptimizedNetwork* optimizedNetwork = optNet.get();
    auto optNetPtr                      = boost::polymorphic_downcast<OptimizedNetwork*>(optimizedNetwork);
    auto& optimizedGraph                = optNetPtr->GetGraph();
    Graph::ConstIterator layer          = optimizedGraph.cbegin();
    auto inputLayer                     = *layer;
    BOOST_TEST((inputLayer->GetBackendId() == EthosNBackendId()));
    ++layer;
    auto convolutionLayer = *layer;
    BOOST_TEST((convolutionLayer->GetBackendId() == EthosNBackendId()));
    ++layer;
    auto activationLayer = *layer;
    BOOST_TEST((activationLayer->GetBackendId() == BackendId(Compute::CpuRef)));
    ++layer;
    auto outputLayer = *layer;
    BOOST_TEST((outputLayer->GetBackendId() == BackendId(Compute::CpuRef)));
}

BOOST_AUTO_TEST_SUITE_END()
