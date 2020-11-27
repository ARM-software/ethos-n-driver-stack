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
#include <armnn/Logging.hpp>
#include <armnn/utility/Assert.hpp>
#include <backendsCommon/test/CommonTestUtils.hpp>
#include <test/EthosNTestUtils.hpp>

// The include order is important. Turn off clang-format
// clang-format off
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
// clang-format on

#include <sstream>

using Tensors  = std::map<std::string, armnn::SimpleInputOutput>;
using Layers   = std::vector<armnn::SimpleLayer>;
using Shape    = std::vector<uint32_t>;
using Mappings = std::vector<armnn::Mapping>;

using namespace armnn;
using namespace testing_utils;

namespace bdata = boost::unit_test::data;

namespace
{

struct TestLayerType
{
    LayerType layer;
    // For ActivationLayer, this will be the name of the activation function
    std::string name;
};

using TestLayerTypeElem = std::tuple<TestLayerType, TestLayerType>;
using TestLayerTypeList = std::vector<TestLayerTypeElem>;

enum ExceptionCases
{
    NoException              = 0,
    ParseException           = 1,
    InvalidArgumentException = 2,
    First                    = NoException,
    Last                     = InvalidArgumentException
};

AdditionalLayerParams CreateAdditionalParams(LayerType type)
{
    AdditionalLayerParams params;

    if (type == LayerType::Pooling2d)
    {
        params.insert(std::make_pair("padding", "1x1x1x1"));
        params.insert(std::make_pair("kernel", "3x3"));
        params.insert(std::make_pair("stride", "1x1"));
        params.insert(std::make_pair("function", "Average"));
    }
    else if (type == LayerType::TransposeConvolution2d)
    {
        params.insert(std::make_pair("stride", "2x2"));
        params.insert(std::make_pair("padding", "0x0x0x0"));
        params.insert(std::make_pair("kernel", "1x1"));
    }
    else if ((type == LayerType::DepthwiseConvolution2d) || (type == LayerType::Convolution2d))
    {
        params.insert(std::make_pair("stride", "1x1"));
        params.insert(std::make_pair("kernel", "1x1"));
        params.insert(std::make_pair("padding", "0x0x0x0"));
        params.insert(std::make_pair("dilation", "1x1"));
    }

    return params;
}

// Creates mappings for substitution
template <const unsigned int SIZE>
Mappings CreateSubstitutionMappings(TestLayerType original,
                                    TestLayerType replacement,
                                    std::array<const unsigned int, SIZE> inputDimensions,
                                    std::array<const unsigned int, SIZE> outputDimensions)
{
    Mappings mappings;
    Layers orgLayers, replacementLayers;
    Shape inputTensorShape, outputTensorShape;
    AdditionalLayerParams params;

    inputTensorShape  = Shape(inputDimensions.data(), (inputDimensions.data() + inputDimensions.size()));
    outputTensorShape = Shape(outputDimensions.data(), (outputDimensions.data() + outputDimensions.size()));

    auto tensors =
        Tensors({ std::make_pair("firstInput", armnn::SimpleInputOutput("firstInput", inputTensorShape)),
                  std::make_pair("firstOutput", armnn::SimpleInputOutput("firstOutput", outputTensorShape)) });

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
                Layers({ armnn::SimpleLayer("Activation", { armnn::SimpleInputOutput("firstInput", inputTensorShape) },
                                            { "firstOutput" }, { std::make_pair("function", original.name) }) });
            break;
        case LayerType::DepthwiseConvolution2d:
            orgLayers = Layers({ armnn::SimpleLayer("DepthwiseConvolution2d",
                                                    { armnn::SimpleInputOutput("firstInput", inputTensorShape) },
                                                    { "firstOutput" }, {}) });
            break;
        case LayerType::L2Normalization:
            orgLayers = Layers(
                { armnn::SimpleLayer("L2Normalization", { armnn::SimpleInputOutput("firstInput", inputTensorShape) },
                                     { "firstOutput" }, {}) });
            break;
        case LayerType::Floor:
            orgLayers = Layers({ armnn::SimpleLayer(
                "Floor", { armnn::SimpleInputOutput("firstInput", inputTensorShape) }, { "firstOutput" }, {}) });
            break;
        case LayerType::Softmax:
            orgLayers = Layers({ armnn::SimpleLayer(
                "Softmax", { armnn::SimpleInputOutput("firstInput", inputTensorShape) }, { "firstOutput" }, {}) });
            break;
        case LayerType::Output:
            orgLayers = Layers({ armnn::SimpleLayer(
                "Output", { armnn::SimpleInputOutput("firstInput", inputTensorShape) }, { "firstOutput" }, {}) });
            break;
        case LayerType::LogSoftmax:
            orgLayers = Layers({ armnn::SimpleLayer(
                "LogSoftmax", { armnn::SimpleInputOutput("firstInput", inputTensorShape) }, { "firstOutput" }, {}) });
            break;
        case LayerType::DepthToSpace:
            orgLayers = Layers({ armnn::SimpleLayer(
                "DepthToSpace", { armnn::SimpleInputOutput("firstInput", inputTensorShape) }, { "firstOutput" }, {}) });
            break;
        case LayerType::Convolution2d:
            orgLayers = Layers(
                { armnn::SimpleLayer("Convolution2d", { armnn::SimpleInputOutput("firstInput", inputTensorShape) },
                                     { "firstOutput" }, {}) });
            break;
        default:
            ARMNN_ASSERT_MSG(false, "Unsupported substitutable layer type\n");
            break;
    }

    switch (replacement.layer)
    {
        case LayerType::Activation:
            BOOST_TEST(replacement.name.empty() != true);
            replacementLayers =
                Layers({ armnn::SimpleLayer("Activation", { armnn::SimpleInputOutput("firstInput", inputTensorShape) },
                                            { "firstOutput" }, { std::make_pair("function", replacement.name) }) });
            break;
        case LayerType::Convolution2d:
            params            = CreateAdditionalParams(LayerType::Convolution2d);
            replacementLayers = Layers(
                { armnn::SimpleLayer("Convolution2d", { armnn::SimpleInputOutput("firstInput", inputTensorShape) },
                                     { "firstOutput" }, params) });
            break;
        case LayerType::TransposeConvolution2d:
            params            = CreateAdditionalParams(LayerType::TransposeConvolution2d);
            replacementLayers = Layers({ armnn::SimpleLayer(
                "TransposeConvolution2d", { armnn::SimpleInputOutput("firstInput", inputTensorShape) },
                { "firstOutput" }, params) });
            break;
        case LayerType::DepthwiseConvolution2d:
            params            = CreateAdditionalParams(LayerType::DepthwiseConvolution2d);
            replacementLayers = Layers({ armnn::SimpleLayer(
                "DepthwiseConvolution2d", { armnn::SimpleInputOutput("firstInput", inputTensorShape) },
                { "firstOutput" }, params) });
            break;
        case LayerType::FullyConnected:
            ARMNN_LOG(info) << "Create a fully connected mapping\n";
            replacementLayers = Layers(
                { armnn::SimpleLayer("FullyConnected", { armnn::SimpleInputOutput("firstInput", inputTensorShape) },
                                     { "firstOutput" }, {}) });
            break;
        case LayerType::Pooling2d:
            ARMNN_LOG(info) << "Create a Pooling2d mapping\n";
            params = CreateAdditionalParams(LayerType::Pooling2d);
            replacementLayers =
                Layers({ armnn::SimpleLayer("Pooling2d", { armnn::SimpleInputOutput("firstInput", inputTensorShape) },
                                            { "firstOutput" }, params) });
            break;
        default:
            ARMNN_ASSERT_MSG(false, "Unsupported replacement layer type\n");
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
    mappings += "Activation  (firstInput) (firstOutput) ((function=TanH))\n";
    mappings += "graph-replacement:\n";
    mappings += "Excluded  (firstInput) (firstOutput)\n";
    mappings += "pattern:\n";
    mappings += "input firstInput 1x_x_x_\n";
    mappings += "output firstOutput 1x_x_x_\n";
    mappings += "StandIn  (firstInput) (firstOutput) ((name=namew))\n";
    mappings += "graph-replacement:\n";
    mappings += "Excluded  (firstInput) (firstOutput)\n";

    std::ofstream mappingStream(config.m_PerfMappingFile);
    if (mappingStream.is_open())
    {
        mappingStream << mappings;
    }

    return mappings;
}

std::string CreateMappingsWithLayerName(const EthosNConfig config = EthosNConfig())
{
    std::string mappings;

    mappings += "pattern:\n";
    mappings += "input firstInput 1x16x16x16\n";
    mappings += "output firstOutput 1x16x16x16\n";
    mappings += "DepthwiseConvolution2d  (firstInput) (firstOutput) ((name=depth))\n";
    mappings += "graph-replacement:\n";
    mappings += "Convolution2d  (firstInput) (firstOutput)\n";

    std::ofstream mappingStream(config.m_PerfMappingFile);
    if (mappingStream.is_open())
    {
        mappingStream << mappings;
    }

    return mappings;
}

void CreateMappingsWithInvalidAdditionalArguments1(const EthosNConfig config = EthosNConfig())
{
    std::string mapping;

    // Invalid additional parameter name ie kernell
    mapping = "pattern:\n";
    mapping += "input firstInput 1x16x16x16\n";
    mapping += "output firstOutput 1x16x16x16\n";
    mapping += "DepthwiseConvolution2d  (firstInput) (firstOutput) ((name=depth))\n";
    mapping += "graph-replacement:\n";
    mapping += "Convolution2d  (firstInput) (firstOutput) ((kernell=1x1))\n";

    std::ofstream mappingStream(config.m_PerfMappingFile);
    if (mappingStream.is_open())
    {
        mappingStream << mapping;
    }
}

void CreateMappingsWithInvalidAdditionalArguments2(const EthosNConfig config = EthosNConfig())
{
    std::string mapping;

    // Invalid value of additional parameter ie stride=1
    mapping = "pattern:\n";
    mapping += "input firstInput 1x16x16x16\n";
    mapping += "output firstOutput 1x16x16x16\n";
    mapping += "Activation  (firstInput) (firstOutput) ((function=TanH))\n";
    mapping += "graph-replacement:\n";
    mapping += "DepthwiseConvolution2d  (firstInput) (firstOutput) ((stride=1))\n";

    std::ofstream mappingStream(config.m_PerfMappingFile);
    if (mappingStream.is_open())
    {
        mappingStream << mapping;
    }
}

void CreateMappingsWithInvalidAdditionalArguments3(const EthosNConfig config = EthosNConfig())
{
    std::string mapping;

    // Required additional parameters not provided
    // Pooling2d requires ((function=something))
    mapping = "pattern:\n";
    mapping += "input firstInput 1x16x16x16\n";
    mapping += "output firstOutput 1x16x16x16\n";
    mapping += "Pooling2d  (firstInput) (firstOutput) ((name=depth))\n";
    mapping += "graph-replacement:\n";
    mapping += "Activation  (firstInput) (firstOutput) ((function=Sigmoid))\n";

    std::ofstream mappingStream(config.m_PerfMappingFile);
    if (mappingStream.is_open())
    {
        mappingStream << mapping;
    }
}

void CreateMappingsWithInvalidAdditionalArguments4(const EthosNConfig config = EthosNConfig())
{
    std::string mapping;

    // Unsupported value provided for additional parameter
    // Pooling2d is only supported with ((function=Average))
    mapping = "pattern:\n";
    mapping += "input firstInput 1x16x16x16\n";
    mapping += "output firstOutput 1x16x16x16\n";
    mapping += "Activation  (firstInput) (firstOutput) ((name=depth), (function=ReLu))\n";
    mapping += "graph-replacement:\n";
    mapping += "Pooling2d  (firstInput) (firstOutput) ((function=Max))\n";

    std::ofstream mappingStream(config.m_PerfMappingFile);
    if (mappingStream.is_open())
    {
        mappingStream << mapping;
    }
}

void CreateMappingsWithValidAdditionalArguments(const EthosNConfig config = EthosNConfig())
{
    std::string mappings;

    mappings += "pattern:\n";
    mappings += "input firstInput 1x16x16x16\n";
    mappings += "output firstOutput 1x16x16x16\n";
    mappings += "Activation,  (firstInput), (firstOutput), ((name=myact), (function=ReLu))\n";
    mappings += "graph-replacement:\n";
    mappings += "Pooling2d,  (firstInput), (firstOutput), ((kernel=3x3), (stride=2x2), (padding=2x2x2x2), "
                "(function=Average), (name=mypool))\n";

    std::ofstream mappingStream(config.m_PerfMappingFile);
    if (mappingStream.is_open())
    {
        mappingStream << mappings;
    }
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
        "input, firstInput, 1x_x_x_",
        "\toutput,  firstOutput, 1x_x_x_",
        "Activation,  (firstInput), (firstOutput), ((function=TanH))",
    };
    Tensors tensors1;
    Layers layers1;

    const std::vector<std::string> buf2 = {
        "input firstInput, 1x_x_x_",
        "input secondInput, 1x1x2x3",
        "output firstOutput, 1x_x_x_",
        "StandIn, (firstInput, secondInput), (firstOutput), ((\tfunction= CustomOp), (name=somename))",
    };
    Tensors tensors2;
    Layers layers2;

    const std::vector<std::string> buf3 = {
        "input firstInput, 1x_x_x_",
        "output  firstOutput, 1x_x_x_",
        "output  secondOutput, 1x_x_x_",
        "Excluded,  (firstInput), (firstOutput, secondOutput)",
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
    BOOST_TEST(layers2 == Layers({ armnn::SimpleLayer(
                              "StandIn",
                              { armnn::SimpleInputOutput("firstInput", Shape({ 1, 0, 0, 0 })),
                                armnn::SimpleInputOutput("secondInput", Shape({ 1, 1, 2, 3 })) },
                              { "firstOutput" },
                              { std::map<std::string, std::string>{ std::make_pair("function", "CustomOp"),
                                                                    std::make_pair("name", "somename") } }) }));

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
        "Activation, (firstInput), (firstOutput), ((function=TanH))",
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

template <const unsigned int SIZE>
Mappings CreateMappings(TestLayerType originalType,
                        TestLayerType replacementType,
                        std::array<const unsigned int, SIZE> inputDimensions,
                        std::array<const unsigned int, SIZE> outputDimensions)
{
    Mappings ethosNMappings;
    std::map<std::string, LayerType> mapStringToLayerType = armnn::ethosnbackend::GetMapStringToLayerType();

    ethosNMappings = CreateSubstitutionMappings<SIZE>(originalType, replacementType, inputDimensions, outputDimensions);

    //Test if there is atleast one mapping
    ARMNN_ASSERT((ethosNMappings.size() >= 1));
    //Test if the mapping layer types are as intended
    BOOST_TEST(((mapStringToLayerType.find(ethosNMappings[0].m_ReplacementLayers[0].m_LayerTypeName)->second) ==
                replacementType.layer));
    BOOST_TEST(((mapStringToLayerType.find(ethosNMappings[0].m_PatternLayers[0].m_LayerTypeName))->second ==
                originalType.layer));

    //Test for single layer mappings
    BOOST_TEST((ethosNMappings.size() == 1));
    BOOST_TEST((ethosNMappings[0].m_PatternLayers.size() == 1));
    BOOST_TEST((ethosNMappings[0].m_ReplacementLayers.size() == 1));

    return ethosNMappings;
}

Mappings CreateMappingsFromList(TestLayerTypeList testLayerTypeList)
{
    Mappings allMaps;

    for (TestLayerTypeElem elem : testLayerTypeList)
    {
        TestLayerType input  = std::get<0>(elem);
        TestLayerType output = std::get<1>(elem);
        Mappings currentMap;

        // We need to create the inputDimensions and outputDimensions as per
        // those written in mapping-tests/*.txt files
        if ((input.layer == LayerType::FullyConnected) || (output.layer == LayerType::FullyConnected))
        {
            std::array<const unsigned int, 2> inputDimensions{ { 1, 16 } };
            std::array<const unsigned int, 2> outputDimensions{ { 1, 1 } };

            currentMap = CreateMappings<2>(input, output, inputDimensions, outputDimensions);
        }
        else
        {
            std::array<const unsigned int, 4> inputDimensions{ { 1, 16, 16, 16 } };
            std::array<const unsigned int, 4> outputDimensions{ { 1, 16, 16, 16 } };

            currentMap = CreateMappings<4>(input, output, inputDimensions, outputDimensions);
        }

        allMaps.insert(allMaps.end(), currentMap.begin(), currentMap.end());
    }

    return allMaps;
}

template <const unsigned int SIZE>
armnn::SubgraphView::SubgraphViewPtr CreateUnoptimizedSubgraph(Graph& graph,
                                                               SimpleLayer layer,
                                                               std::array<const unsigned int, SIZE> inputDimensions,
                                                               std::array<const unsigned int, SIZE> outputDimensions)
{
    Layer *inputLayer, *outputLayer, *operationLayer;
    Shape inputOutputTensorShape;
    std::map<std::string, LayerType> mapStringToLayerType = armnn::ethosnbackend::GetMapStringToLayerType();
    LayerType type                                        = mapStringToLayerType.find(layer.m_LayerTypeName)->second;

    TensorInfo inputInfo(static_cast<unsigned int>(inputDimensions.size()), inputDimensions.data(), DataType::QAsymmU8,
                         1.0f, 0);
    TensorInfo outputInfo(static_cast<unsigned int>(outputDimensions.size()), outputDimensions.data(),
                          DataType::QAsymmU8, 1.0f, 0);

    if (type == LayerType::Activation)
    {
        std::string activationFuncOriginalLayer = layer.m_LayerParams.find("function")->second;
        std::string name                        = layer.m_LayerParams["name"];

        operationLayer = ethosnbackend::CreateActivationLayer(graph, activationFuncOriginalLayer, name);
    }
    else if ((type == LayerType::Convolution2d) || (type == LayerType::TransposeConvolution2d) ||
             (type == LayerType::DepthwiseConvolution2d))
    {
        unsigned int inputChannels = inputInfo.GetShape()[3];
        DataType weightDataType    = inputInfo.GetDataType();
        operationLayer = ethosnbackend::CreateConvolutionLayer(type, graph, inputChannels, layer.m_LayerParams,
                                                               weightDataType, DataType::Signed32);
    }
    else if (type == LayerType::FullyConnected)
    {
        operationLayer = ethosnbackend::CreateFullyConnectedLayer(graph, inputInfo, outputInfo, layer.m_LayerParams);
    }
    else if (type == LayerType::Pooling2d)
    {
        operationLayer = ethosnbackend::CreatePooling2dLayer(graph, layer.m_LayerParams);
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
bool IsLayerPresentInSubgraph(armnn::Graph& graph, LayerType type, AdditionalLayerParams params = { { "", "" } })
{
    const std::list<Layer*> newGraphLayers(graph.begin(), graph.end());
    bool match = false;

    for (Layer* layer : newGraphLayers)
    {
        if (layer->GetType() == type)
        {
            match = true;

            // Check if the caller has passed any additional layer parameters
            if (!((*params.begin()).first.empty()))
            {
                // Note:- Currently we only check for those additionalParameters which are
                // provided by CreateMappingsWithValidAdditionalArguments(). This has been
                // done to reduce the scope for the test.
                // Our aim is to validate that the layer has set the correct values for the
                // additional parameters which are specified by the mapping file.

                Pooling2dLayer* poolLayer = nullptr;
                ActivationLayer* actLayer = nullptr;
                Pooling2dDescriptor poolDesc;
                ActivationDescriptor actDesc;

                switch (layer->GetType())
                {
                    case armnn::LayerType::Pooling2d:
                        poolLayer = PolymorphicDowncast<Pooling2dLayer*>(layer);
                        break;
                    case armnn::LayerType::Activation:
                        actLayer = PolymorphicDowncast<ActivationLayer*>(layer);
                        break;
                    default:
                        return true;
                }

                if (poolLayer != nullptr)
                {
                    poolDesc = poolLayer->GetParameters();

                    if (!params["function"].empty())
                    {
                        auto poolAlgo   = armnn::ethosnbackend::GetMapStringToPoolingAlgorithm();
                        auto m_PoolType = poolAlgo.find(params["function"])->second;
                        BOOST_TEST((m_PoolType == poolDesc.m_PoolType));
                    }

                    if (!params["stride"].empty())
                    {
                        auto stride = GetLayerParameterValue(params, "stride");
                        BOOST_TEST((poolDesc.m_StrideX == stride[ethosnbackend::STRIDE_X]));
                        BOOST_TEST((poolDesc.m_StrideY == stride[ethosnbackend::STRIDE_Y]));
                    }

                    if (!params["kernel"].empty())
                    {
                        auto kernel = GetLayerParameterValue(params, "kernel");
                        BOOST_TEST((poolDesc.m_PoolHeight = kernel[ethosnbackend::KERNEL_HEIGHT]));
                        BOOST_TEST((poolDesc.m_PoolWidth = kernel[ethosnbackend::KERNEL_WIDTH]));
                    }

                    if (!params["padding"].empty())
                    {
                        auto padding = GetLayerParameterValue(params, "padding");
                        BOOST_TEST((poolDesc.m_PadBottom == padding[ethosnbackend::PAD_BOTTOM]));
                        BOOST_TEST((poolDesc.m_PadLeft == padding[ethosnbackend::PAD_LEFT]));
                        BOOST_TEST((poolDesc.m_PadRight == padding[ethosnbackend::PAD_RIGHT]));
                        BOOST_TEST((poolDesc.m_PadTop == padding[ethosnbackend::PAD_TOP]));
                    }
                }

                if (actLayer != nullptr)
                {
                    actDesc = actLayer->GetParameters();

                    if (!params["function"].empty())
                    {
                        auto actFunc    = armnn::ethosnbackend::GetMapStringToActivationFunction();
                        auto m_Function = actFunc.find(params["function"])->second;
                        BOOST_TEST((m_Function == actDesc.m_Function));
                    }
                }

                // Check for the common parameters ie 'name'
                if (!params["name"].empty())
                {
                    BOOST_TEST((params["name"] == layer->GetNameStr()));
                }
            }
        }
    }
    return match;
}

template <const unsigned int SIZE>
void TestSubgraphSubstitution(TestLayerType originalType,
                              TestLayerType replacementType,
                              std::array<const unsigned int, SIZE> inputDimensions,
                              std::array<const unsigned int, SIZE> outputDimensions,
                              bool validSubstitution = true)
{
    using namespace testing_utils;
    Graph graph, graph2;

    TempDir tmpDir;
    EthosNConfig ethosnConfig = CreateEthosNConfig(tmpDir);
    auto backendObjPtr        = CreateBackendObject(EthosNBackendId());
    BOOST_TEST((backendObjPtr != nullptr));

    auto ethosNMappings = CreateMappings<SIZE>(originalType, replacementType, inputDimensions, outputDimensions);

    auto subGraphOriginal =
        CreateUnoptimizedSubgraph<SIZE>(graph, ethosNMappings[0].m_PatternLayers[0], inputDimensions, outputDimensions);
    auto subGraphOriginal2 = CreateUnoptimizedSubgraph<SIZE>(graph2, ethosNMappings[0].m_PatternLayers[0],
                                                             inputDimensions, outputDimensions);

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
    BOOST_TEST((IsLayerPresentInSubgraph(graph2, replacementType.layer) == validSubstitution));
}

static const char* MAPPING_FILE_TEST_DIRECTORY = "armnn-ethos-n-backend/test/mapping-tests/";
struct TestParseMappingFileData
{
    const char* fileName;
    TestLayerTypeList layers;
    ExceptionCases exception     = NoException;
    std::string exceptionMessage = "";
};

static TestParseMappingFileData TestParseMappingFileDataset[] = {
    { //.fileName =
      "inActivationBoundedReLu_outActivationSigmoid.txt",
      //.layers =
      {
          std::make_tuple(
              //             .layer, .name
              TestLayerType({ LayerType::Activation, "BoundedReLu" }),
              TestLayerType({ LayerType::Activation, "Sigmoid" })),
      } },
    { //.fileName =
      "inActivationBoundedReLu_outConvolution2d.txt",
      //.layers =
      {
          std::make_tuple(
              //             .layer, .name
              TestLayerType({ LayerType::Activation, "BoundedReLu" }),
              TestLayerType({ LayerType::Convolution2d, "" })),
      } },
    { //.fileName =
      "inActivationBoundedReLu_outActivationReLu.txt",
      //.layers =
      { std::make_tuple(
          //             .layer, .name
          TestLayerType({ LayerType::Activation, "BoundedReLu" }),
          TestLayerType({ LayerType::Activation, "ReLu" })) } },
    { //.fileName =
      "inDepthToSpace_outTransposeConvolution2d.txt",
      //.layers =
      { std::make_tuple(
          //             .layer, .name
          TestLayerType({ LayerType::DepthToSpace, "" }),
          TestLayerType({ LayerType::TransposeConvolution2d, "" })) } },
    { //.fileName =
      "inActivationBoundedReLu_outDepthwiseConvolution2d.txt",
      //.layers =
      { std::make_tuple(
          //             .layer, .name
          TestLayerType({ LayerType::Activation, "BoundedReLu" }),
          TestLayerType({ LayerType::DepthwiseConvolution2d, "" })) } },
    { //.fileName =
      "inActivationBoundedReLu_outFullyConnected.txt",
      //.layers =
      { std::make_tuple(
          //             .layer, .name
          TestLayerType({ LayerType::Activation, "BoundedReLu" }),
          TestLayerType({ LayerType::FullyConnected, "" })) } },
    { //.fileName =
      "inActivationBoundedReLu_outPooling2d.txt",
      //.layers =
      { std::make_tuple(
          //             .layer, .name
          TestLayerType({ LayerType::Activation, "BoundedReLu" }),
          TestLayerType({ LayerType::Pooling2d, "" })) } },
    { //.fileName =
      "inDepthwiseConvolution2d_outConvolution2d.txt",
      //.layers =
      { std::make_tuple(
          //             .layer, .name
          TestLayerType({ LayerType::DepthwiseConvolution2d, "" }),
          TestLayerType({ LayerType::Convolution2d, "" })) } },
    { //.fileName =
      "inL2Normalization_outDepthwiseConvolution2d.txt",
      //.layers =
      { std::make_tuple(
          //             .layer, .name
          TestLayerType({ LayerType::L2Normalization, "" }),
          TestLayerType({ LayerType::DepthwiseConvolution2d, "" })) } },
    { //.fileName =
      "inFloor_outActivationReLu.txt",
      //.layers =
      { std::make_tuple(
          //             .layer, .name
          TestLayerType({ LayerType::Floor, "" }),
          TestLayerType({ LayerType::Activation, "ReLu" })) } },
    { //.fileName =
      "inSoftmax_outActivationSigmoid.txt",
      //.layers =
      { std::make_tuple(
          //             .layer, .name
          TestLayerType({ LayerType::Softmax, "" }),
          TestLayerType({ LayerType::Activation, "Sigmoid" })) } },
    { //.fileName =
      "inConvolution2d_outPooling2d.txt",
      //.layers =
      { std::make_tuple(
          //             .layer, .name
          TestLayerType({ LayerType::Convolution2d, "" }),
          TestLayerType({ LayerType::Pooling2d, "" })) } },
    { //.fileName =
      "inLogSoftmax_outFullyConnected.txt",
      //.layers =
      { std::make_tuple(
          //             .layer, .name
          TestLayerType({ LayerType::LogSoftmax, "" }),
          TestLayerType({ LayerType::FullyConnected, "" })) } },
    { //.fileName =
      "multiLayerMapping.txt",
      //.layers =
      { std::make_tuple(
            //             .layer, .name
            TestLayerType({ LayerType::DepthwiseConvolution2d, "" }),
            TestLayerType({ LayerType::Convolution2d, "" })),
        std::make_tuple(
            //             .layer, .name
            TestLayerType({ LayerType::Output, "" }),
            TestLayerType({ LayerType::Pooling2d, "" })),
        std::make_tuple(
            //             .layer, .name
            TestLayerType({ LayerType::L2Normalization, "" }),
            TestLayerType({ LayerType::DepthwiseConvolution2d, "" })) } },
    { //.fileName =
      "wrongSourceMapping.txt",
      //.layers =
      {},
      // .exception
      ExceptionCases::ParseException,
      // .exceptionMessage
      "L2Normalization_XYZ, (firstInput), (firstOutput)" },
    { //.fileName =
      "wrongReplacementMapping.txt",
      //.layers =
      {},
      // .exception
      ExceptionCases::ParseException,
      // .exceptionMessage
      "DepthwiseConvolution2d_XYZ, (firstInput), (firstOutput)" },
    { //.fileName =
      "wrongSyntaxAdditionalParams.txt",
      // .layers =
      {},
      // .exception
      ExceptionCases::ParseException,
      // .exceptionMessage
      "Additional parameters are to be enclosed in (( ))" },
    { //.fileName =
      "wrongSyntaxTooManyParams.txt",
      // .layers =
      {},
      // .exception
      ExceptionCases::ParseException,
      // .exceptionMessage
      "Too many parameters specified" },
    { //.fileName =
      "wrongSyntaxAdditionalParams2.txt",
      // .layers =
      {},
      // .exception
      ExceptionCases::ParseException,
      // .exceptionMessage
      "Syntax error: Additional parameters should be in (name1=value1),(name2=value2) format" },
    { //.fileName =
      "wrongSyntaxAdditionalParams3.txt",
      // .layers =
      {},
      // .exception
      ExceptionCases::ParseException,
      // .exceptionMessage
      "Syntax error: Additional parameters should be in (name1=value1),(name2=value2) format" }
};

std::ostream& boost_test_print_type(std::ostream& os, const TestParseMappingFileData& data)
{
    os << "filename: " << data.fileName << std::endl;
    for (size_t elemIdx = 0; elemIdx < data.layers.size(); elemIdx++)
    {
        TestLayerTypeElem elem = data.layers[elemIdx];
        TestLayerType input    = std::get<0>(elem);
        TestLayerType output   = std::get<1>(elem);

        os << "Layer " << elemIdx << ":" << std::endl;
        os << "\tInputLayerType: " << GetLayerTypeAsCString(input.layer) << " ";
        os << "\tInputLayerName: " << input.name << std::endl;
        os << "\tOutputLayerType: " << GetLayerTypeAsCString(output.layer) << " ";
        os << "\tOutputLayerName: " << output.name << std::endl;
    }
    return os;
}

BOOST_DATA_TEST_CASE(
    TestParseMappingFile,    // Test case name
    // see https://www.boost.org/doc/libs/1_72_0/libs/test/doc/html/boost_test/tests_organization/test_cases/test_case_generation/generators.html#boost_test.tests_organization.test_cases.test_case_generation.generators.c_arrays
    // for some explanation on how the date sets are generated
    bdata::xrange(ARRAY_SIZE(TestParseMappingFileDataset)) ^ bdata::make(TestParseMappingFileDataset),
    xr,              // Test number (not used here)
    arrayElement)    // Current entry of the TestParseMappingFileDataset array to be tested
{
    (void)(xr);

    //Get the input parameter of the tests
    const char* fileName = arrayElement.fileName;
    std::string fullFileName(MAPPING_FILE_TEST_DIRECTORY);
    fullFileName.append(fileName);
    TestLayerTypeList layers = arrayElement.layers;
    ExceptionCases gotException;
    ExceptionCases expectException = arrayElement.exception;
    std::string exceptionMessage   = arrayElement.exceptionMessage;

    //Execute the test code
    Mappings inputMapping = CreateMappingsFromList(layers);

    Mappings parsedMapping;
    try
    {
        parsedMapping = GetMappings(fullFileName);
        gotException  = ExceptionCases::NoException;
    }
    catch (const armnn::ParseException& e)
    {
        gotException = ExceptionCases::ParseException;
        BOOST_TEST((std::string(e.what()).find(exceptionMessage) != std::string::npos));
    }
    catch (const armnn::InvalidArgumentException& e)
    {
        gotException = ExceptionCases::InvalidArgumentException;
        BOOST_TEST((std::string(e.what()).find(exceptionMessage) != std::string::npos));
    }

    //Check the result
    BOOST_CHECK_EQUAL(gotException, expectException);
    BOOST_TEST(inputMapping == parsedMapping);
}

BOOST_AUTO_TEST_CASE(TestAllSubgraphSubstitution)
{
    TestLayerType org, replacement;

    {
        std::array<const unsigned int, 4> inputDimensions{ { 1, 16, 16, 16 } };
        std::array<const unsigned int, 4> outputDimensions{ { 1, 16, 16, 16 } };

        org.layer         = LayerType::Activation;
        org.name          = "BoundedReLu";
        replacement.layer = LayerType::Activation;
        replacement.name  = "Sigmoid";
        TestSubgraphSubstitution<4>(org, replacement, inputDimensions, outputDimensions);
    }

    {
        std::array<const unsigned int, 4> inputDimensions{ { 1, 16, 16, 16 } };
        std::array<const unsigned int, 4> outputDimensions{ { 1, 16, 16, 16 } };

        org.layer         = LayerType::Activation;
        org.name          = "BoundedReLu";
        replacement.layer = LayerType::Convolution2d;
        replacement.name  = "";
        TestSubgraphSubstitution<4>(org, replacement, inputDimensions, outputDimensions);
    }

    {
        std::array<const unsigned int, 3> inputDimensions{ { 1, 16, 16 } };
        std::array<const unsigned int, 3> outputDimensions{ { 1, 16, 16 } };

        org.layer         = LayerType::Activation;
        org.name          = "BoundedReLu";
        replacement.layer = LayerType::Activation;
        replacement.name  = "ReLu";
        TestSubgraphSubstitution<3>(org, replacement, inputDimensions, outputDimensions);
    }

    {
        // This is going to increase the size of the output by two times
        // as we are using a fixed value of strideX (which is 2) and
        // strideY (which is 2)
        std::array<const unsigned int, 4> inputDimensions{ { 1, 16, 16, 16 } };
        std::array<const unsigned int, 4> outputDimensions{ { 1, 32, 32, 16 } };

        org.layer         = LayerType::Activation;
        org.name          = "TanH";
        replacement.layer = LayerType::TransposeConvolution2d;
        replacement.name  = "";
        TestSubgraphSubstitution<4>(org, replacement, inputDimensions, outputDimensions);
    }

    {
        std::array<const unsigned int, 3> inputDimensions{ { 1, 16, 16 } };
        std::array<const unsigned int, 3> outputDimensions{ { 1, 16, 16 } };

        // Test an invalid mapping
        // Here Activation is to be substituted with Convolution2d when input/ouput
        // tensor shape is of three dimensions.
        // This is invalid as convolution expects input/output tensor shape to be of
        // four dimensions.
        try
        {
            org.layer         = LayerType::Activation;
            org.name          = "BoundedReLu";
            replacement.layer = LayerType::Convolution2d;
            replacement.name  = "";
            TestSubgraphSubstitution<3>(org, replacement, inputDimensions, outputDimensions, false);
        }
        catch (const armnn::InvalidArgumentException& e)
        {
            std::string err = "Invalid dimension index: 3 (number of dimensions is 3)";
            BOOST_TEST((std::string(e.what()).find(err) != std::string::npos));
        }
    }

    {
        std::array<const unsigned int, 4> inputDimensions{ { 1, 16, 16, 16 } };
        std::array<const unsigned int, 4> outputDimensions{ { 1, 16, 16, 16 } };

        org.layer         = LayerType::Activation;
        org.name          = "BoundedReLu";
        replacement.layer = LayerType::DepthwiseConvolution2d;
        replacement.name  = "";
        TestSubgraphSubstitution<4>(org, replacement, inputDimensions, outputDimensions);
    }

    {
        std::array<const unsigned int, 2> inputDimensions{ { 1, 16 } };
        std::array<const unsigned int, 2> outputDimensions{ { 1, 1 } };

        org.layer         = LayerType::Activation;
        org.name          = "BoundedReLu";
        replacement.layer = LayerType::FullyConnected;
        replacement.name  = "";
        TestSubgraphSubstitution<2>(org, replacement, inputDimensions, outputDimensions);
    }

    {
        std::array<const unsigned int, 4> inputDimensions{ { 1, 16, 16, 16 } };
        std::array<const unsigned int, 4> outputDimensions{ { 1, 16, 16, 16 } };

        org.layer         = LayerType::Activation;
        org.name          = "BoundedReLu";
        replacement.layer = LayerType::Pooling2d;
        replacement.name  = "";
        TestSubgraphSubstitution<4>(org, replacement, inputDimensions, outputDimensions);
    }
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

BOOST_AUTO_TEST_CASE(TestAdditionalParameters)
{
    // Given
    TempDir tmpDir;
    armnn::g_EthosNConfig = CreateEthosNConfig(tmpDir);

    typedef void (*CreateMappingsWithAdditionalArgs)(const EthosNConfig);
    typedef struct
    {
        CreateMappingsWithAdditionalArgs createMappingFunc;
        std::string exceptionMessage;
        ExceptionCases exception;
    } mappingTestCases;

    std::vector<mappingTestCases> testCases;

    testCases.push_back({ CreateMappingsWithInvalidAdditionalArguments1,
                          "Invalid Argument: Layer Parameter \"kernell\"is unknown",
                          ExceptionCases::InvalidArgumentException });
    testCases.push_back({ CreateMappingsWithInvalidAdditionalArguments2,
                          "Invalid Value: The expected format is ((stride=_x_))",
                          ExceptionCases::InvalidArgumentException });

    testCases.push_back({ CreateMappingsWithInvalidAdditionalArguments3,
                          "Invalid Argument: ((function=somefunction)) is needed",
                          ExceptionCases::InvalidArgumentException });

    testCases.push_back({ CreateMappingsWithInvalidAdditionalArguments4,
                          "Invalid Value: Only Average Pooling is supported",
                          ExceptionCases::InvalidArgumentException });

    testCases.push_back({ CreateMappingsWithValidAdditionalArguments, "", ExceptionCases::NoException });

    for (auto test : testCases)
    {
        test.createMappingFunc(armnn::g_EthosNConfig);

        auto ethosNMappings = GetMappings(armnn::g_EthosNConfig.m_PerfMappingFile);
        Graph graph;
        std::string exceptionMessage = test.exceptionMessage;
        auto expectException         = test.exception;
        auto gotException            = ExceptionCases::NoException;

        // When
        auto originalLayerType =
            armnn::ethosnbackend::GetLayerType(ethosNMappings[0].m_PatternLayers[0].m_LayerTypeName);
        auto replacementLayerType =
            armnn::ethosnbackend::GetLayerType(ethosNMappings[0].m_ReplacementLayers[0].m_LayerTypeName);
        std::array<const unsigned int, 4> inputDimensions{ { 1, 16, 16, 16 } };
        std::array<const unsigned int, 4> outputDimensions{ { 1, 16, 16, 16 } };

        auto subGraphOriginal = CreateUnoptimizedSubgraph<inputDimensions.size()>(
            graph, ethosNMappings[0].m_PatternLayers[0], inputDimensions, outputDimensions);

        BOOST_TEST(
            IsLayerPresentInSubgraph(graph, originalLayerType, ethosNMappings[0].m_PatternLayers[0].m_LayerParams));

        // Then
        try
        {
            ethosnbackend::ApplyMappings(ethosNMappings, graph);
            BOOST_TEST(IsLayerPresentInSubgraph(graph, replacementLayerType,
                                                ethosNMappings[0].m_ReplacementLayers[0].m_LayerParams));
        }
        catch (const armnn::InvalidArgumentException& e)
        {
            gotException = ExceptionCases::InvalidArgumentException;
            BOOST_TEST((std::string(e.what()).find(exceptionMessage) != std::string::npos));
        }
        catch (const armnn::ParseException& e)
        {
            gotException = ExceptionCases::InvalidArgumentException;
            BOOST_TEST((std::string(e.what()).find(exceptionMessage) != std::string::npos));
        }
        BOOST_TEST((gotException == expectException));
    }
}

BOOST_AUTO_TEST_CASE(TestLayerSubstitutionWithName)
{
    // Given
    TempDir tmpDir;
    Graph graph;
    armnn::g_EthosNConfig = CreateEthosNConfig(tmpDir);
    CreateMappingsWithLayerName(armnn::g_EthosNConfig);
    std::array<const unsigned int, 4> inputDimensions{ { 1, 16, 16, 16 } };
    std::array<const unsigned int, 4> outputDimensions{ { 1, 16, 16, 16 } };
    auto ethosNMappings    = GetMappings(armnn::g_EthosNConfig.m_PerfMappingFile);
    auto originalLayerType = armnn::ethosnbackend::GetLayerType(ethosNMappings[0].m_PatternLayers[0].m_LayerTypeName);
    auto replacementLayerType =
        armnn::ethosnbackend::GetLayerType(ethosNMappings[0].m_ReplacementLayers[0].m_LayerTypeName);

    // When
    auto subGraphOriginal = CreateUnoptimizedSubgraph<inputDimensions.size()>(
        graph, ethosNMappings[0].m_PatternLayers[0], inputDimensions, outputDimensions);
    BOOST_TEST(IsLayerPresentInSubgraph(graph, originalLayerType));
    ethosnbackend::ApplyMappings(ethosNMappings, graph);

    // Then
    BOOST_TEST(IsLayerPresentInSubgraph(graph, replacementLayerType));
}

BOOST_AUTO_TEST_CASE(TestLayerSubstitutionWithNameMismatch)
{
    // Given
    TempDir tmpDir;
    Graph graph;
    armnn::g_EthosNConfig = CreateEthosNConfig(tmpDir);
    CreateMappingsWithLayerName(armnn::g_EthosNConfig);
    std::array<const unsigned int, 4> inputDimensions{ { 1, 16, 16, 16 } };
    std::array<const unsigned int, 4> outputDimensions{ { 1, 16, 16, 16 } };
    auto ethosNMappings    = GetMappings(armnn::g_EthosNConfig.m_PerfMappingFile);
    auto originalLayerType = armnn::ethosnbackend::GetLayerType(ethosNMappings[0].m_PatternLayers[0].m_LayerTypeName);
    auto replacementLayerType =
        armnn::ethosnbackend::GetLayerType(ethosNMappings[0].m_ReplacementLayers[0].m_LayerTypeName);

    // When
    // Get the original layer name from the mapping parameters
    std::string name = ethosNMappings[0].m_PatternLayers[0].m_LayerParams["name"];
    // Change the layer name in the mapping parameters
    ethosNMappings[0].m_PatternLayers[0].m_LayerParams["name"] = "abcd";
    auto subGraphOriginal                                      = CreateUnoptimizedSubgraph<inputDimensions.size()>(
        graph, ethosNMappings[0].m_PatternLayers[0], inputDimensions, outputDimensions);
    // Revert the layer name in the mapping parameters back to its original.
    // This will ensure that there is a mismatch of layer name between the
    // graph's layer and the mapping parameters.
    ethosNMappings[0].m_PatternLayers[0].m_LayerParams["name"] = name;
    BOOST_TEST(IsLayerPresentInSubgraph(graph, originalLayerType));
    ethosnbackend::ApplyMappings(ethosNMappings, graph);

    // Then
    //Then the substitution should fail
    BOOST_TEST(!(IsLayerPresentInSubgraph(graph, replacementLayerType)));
    // And the graph should still contain the original layer
    BOOST_TEST(IsLayerPresentInSubgraph(graph, originalLayerType));
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

    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());

    // Then
    IOptimizedNetwork* optimizedNetwork = optNet.get();
    auto optNetPtr                      = PolymorphicDowncast<OptimizedNetwork*>(optimizedNetwork);
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
