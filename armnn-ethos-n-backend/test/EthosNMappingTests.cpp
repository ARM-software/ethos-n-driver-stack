//
// Copyright © 2019-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "EthosNConfig.hpp"
#include "EthosNLayerSupport.hpp"
#include "EthosNMapping.hpp"
#include "EthosNTestUtils.hpp"

#include "replacement-tests/SISOCatOneGraphFactory.hpp"
#include <EthosNBackend.hpp>
#include <EthosNBackendId.hpp>
#include <EthosNBackendUtils.hpp>
#include <armnn/Logging.hpp>
#include <armnn/utility/Assert.hpp>
#include <backendsCommon/test/CommonTestUtils.hpp>
#include <test/EthosNTestUtils.hpp>

#include <doctest/doctest.h>

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
        CHECK((mapStringToActivationFunction.find(original.name) != mapStringToActivationFunction.end()));
    }

    if (!replacement.name.empty())
    {
        CHECK((mapStringToActivationFunction.find(replacement.name) != mapStringToActivationFunction.end()));
    }

    switch (original.layer)
    {
        case LayerType::Activation:
            CHECK(original.name.empty() != true);
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
            CHECK(replacement.name.empty() != true);
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

std::string CreateExclusionMappings()
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

    return mappings;
}

std::string CreateMappingsWithLayerName()
{
    std::string mappings;

    mappings += "pattern:\n";
    mappings += "input firstInput 1x16x16x16\n";
    mappings += "output firstOutput 1x16x16x16\n";
    mappings += "DepthwiseConvolution2d  (firstInput) (firstOutput) ((name=depth))\n";
    mappings += "graph-replacement:\n";
    mappings += "Convolution2d  (firstInput) (firstOutput)\n";

    return mappings;
}

std::string CreateMappingsWithInvalidAdditionalArguments1()
{
    std::string mapping;

    // Invalid additional parameter name ie kernell
    mapping = "pattern:\n";
    mapping += "input firstInput 1x16x16x16\n";
    mapping += "output firstOutput 1x16x16x16\n";
    mapping += "DepthwiseConvolution2d  (firstInput) (firstOutput) ((name=depth))\n";
    mapping += "graph-replacement:\n";
    mapping += "Convolution2d  (firstInput) (firstOutput) ((kernell=1x1))\n";

    return mapping;
}

std::string CreateMappingsWithInvalidAdditionalArguments2()
{
    std::string mapping;

    // Invalid value of additional parameter ie stride=1
    mapping = "pattern:\n";
    mapping += "input firstInput 1x16x16x16\n";
    mapping += "output firstOutput 1x16x16x16\n";
    mapping += "Activation  (firstInput) (firstOutput) ((function=TanH))\n";
    mapping += "graph-replacement:\n";
    mapping += "DepthwiseConvolution2d  (firstInput) (firstOutput) ((stride=1))\n";

    return mapping;
}

std::string CreateMappingsWithInvalidAdditionalArguments3()
{
    std::string mapping;

    // Required additional parameters not provided
    // Pooling2d requires ((function=something))
    mapping = "pattern:\n";
    mapping += "input firstInput 1x16x16x16\n";
    mapping += "output firstOutput 1x16x16x16\n";
    mapping += "Activation  (firstInput) (firstOutput) ((function=Sigmoid))\n";
    mapping += "graph-replacement:\n";
    mapping += "Pooling2d  (firstInput) (firstOutput) ((name=depth))\n";

    return mapping;
}

std::string CreateMappingsWithInvalidAdditionalArguments4()
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

    return mapping;
}

std::string CreateMappingsWithValidAdditionalArguments()
{
    std::string mappings;

    mappings += "pattern:\n";
    mappings += "input firstInput 1x16x16x16\n";
    mappings += "output firstOutput 1x16x16x16\n";
    mappings += "Activation,  (firstInput), (firstOutput), ((name=myact), (function=ReLu))\n";
    mappings += "graph-replacement:\n";
    mappings += "Pooling2d,  (firstInput), (firstOutput), ((kernel=3x3), (stride=2x2), (padding=2x2x2x2), "
                "(function=Average), (name=mypool))\n";

    return mappings;
}

EthosNConfig CreateEthosNConfig()
{
    armnn::EthosNConfig config;
    config.m_PerfOnly = true;
    return config;
}

void CreateUnoptimizedNetwork(INetwork& net)
{
    armnn::IConnectableLayer* const inputLayer = net.AddInputLayer(0, "input layer");
    CHECK(inputLayer);

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
    CHECK(tanhLayer);

    armnn::IConnectableLayer* const outputLayer = net.AddOutputLayer(0, "output layer");
    CHECK(outputLayer);

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

template <typename Parameter>
void CheckLayerWithParametersEquals(const Layer* modLayer,
                                    const Layer* expLayer,
                                    std::string paramName,
                                    size_t layerIdx)
{
    const LayerWithParameters<Parameter>* modLayerWithParam =
        PolymorphicDowncast<const LayerWithParameters<Parameter>*>(modLayer);
    const LayerWithParameters<Parameter>* expLayerWithParam =
        PolymorphicDowncast<const LayerWithParameters<Parameter>*>(expLayer);

    bool areParamsEquals = modLayerWithParam->GetParameters() == expLayerWithParam->GetParameters();
    CHECK_MESSAGE(areParamsEquals, paramName << " at layer index: " << layerIdx << " nameMod: " << modLayer->GetName()
                                             << " nameExp: " << expLayer->GetName());
}

template <typename ConvolutionLayer>
void CheckConvolutionLayerDataEquals(const Layer* modLayer,
                                     const Layer* expLayer,
                                     std::string paramName,
                                     size_t layerIdx)
{
    const ConvolutionLayer* modLayerWithParam = PolymorphicDowncast<const ConvolutionLayer*>(modLayer);
    const ConvolutionLayer* expLayerWithParam = PolymorphicDowncast<const ConvolutionLayer*>(expLayer);

    const std::shared_ptr<ConstTensorHandle> modWeight = GetWeight(modLayerWithParam);
    const std::shared_ptr<ConstTensorHandle> expWeight = GetWeight(expLayerWithParam);

    bool weightEquals = modWeight->GetTensorInfo() == expWeight->GetTensorInfo();
    CHECK_MESSAGE(weightEquals, paramName << " weights doesn't match at layer index: " << layerIdx << " nameMod: "
                                          << modLayer->GetName() << " nameExp: " << expLayer->GetName());

    const std::shared_ptr<ConstTensorHandle> modBias = GetBias(modLayerWithParam);
    const std::shared_ptr<ConstTensorHandle> expBias = GetBias(expLayerWithParam);

    bool biasEquals = modBias->GetTensorInfo() == expBias->GetTensorInfo();
    CHECK_MESSAGE(biasEquals, paramName << " bias doesn't match at layer index: " << layerIdx
                                        << " nameMod: " << modLayer->GetName() << " nameExp: " << expLayer->GetName());
}

void CheckLayerEquals(const Layer* modLayer, const Layer* expLayer, std::string paramName, size_t layerIdx)
{
    CHECK_EQ(modLayer->GetNameStr(), expLayer->GetNameStr());

    LayerType modLayerType    = modLayer->GetType();
    LayerType expLayerType    = expLayer->GetType();
    std::string modTypeString = GetLayerTypeAsCString(modLayerType);
    CHECK_MESSAGE(static_cast<int>(modLayerType) == static_cast<int>(expLayerType),
                  paramName << " At layer index " << layerIdx << ": " << modTypeString
                            << " != " << GetLayerTypeAsCString(expLayerType));

    std::string subTestParamName(paramName);
    subTestParamName.append(modTypeString);
    if (modLayerType == LayerType::Input || modLayerType == LayerType::Output)
    {
        //No extra tests to be done
        return;
    }
    else if (modLayerType == LayerType::Activation)
    {
        CheckLayerWithParametersEquals<ActivationDescriptor>(modLayer, expLayer, subTestParamName, layerIdx);
    }
    else if (modLayerType == LayerType::Convolution2d)
    {
        CheckLayerWithParametersEquals<Convolution2dDescriptor>(modLayer, expLayer, subTestParamName, layerIdx);
        CheckConvolutionLayerDataEquals<Convolution2dLayer>(modLayer, expLayer, subTestParamName, layerIdx);
    }
    else if (modLayerType == LayerType::DepthwiseConvolution2d)
    {
        CheckLayerWithParametersEquals<DepthwiseConvolution2dDescriptor>(modLayer, expLayer, subTestParamName,
                                                                         layerIdx);
        CheckConvolutionLayerDataEquals<DepthwiseConvolution2dLayer>(modLayer, expLayer, subTestParamName, layerIdx);
    }
    else if (modLayerType == LayerType::TransposeConvolution2d)
    {
        CheckLayerWithParametersEquals<TransposeConvolution2dDescriptor>(modLayer, expLayer, subTestParamName,
                                                                         layerIdx);
        CheckConvolutionLayerDataEquals<TransposeConvolution2dLayer>(modLayer, expLayer, subTestParamName, layerIdx);
    }
    else if (modLayerType == LayerType::Pooling2d)
    {
        CheckLayerWithParametersEquals<Pooling2dDescriptor>(modLayer, expLayer, subTestParamName, layerIdx);
    }
    else
    {
        std::string assertMessage("Unsupported layer type (");
        assertMessage.append(modTypeString);
        assertMessage.append(") given to");
        assertMessage.append(__FUNCTION__);
        assertMessage.append(". Please add\n\r");
        REQUIRE_MESSAGE(false, assertMessage);
        ARMNN_ASSERT(false);
    }
}

}    // namespace

TEST_SUITE("EthosNMapping")
{
    //
    // Tests that the Ethos-N  mapping file is parsed correctly
    //

    TEST_CASE("TestTrimm")
    {
        CHECK(armnn::Trim(std::string("")).size() == 0);
        CHECK(armnn::Trim(std::string("\t ")).size() == 0);
        CHECK(armnn::Trim(std::string(" pattern:\t")) == std::string("pattern:"));
        CHECK(armnn::Trim(std::string("input firstInput, 1x_x_x_  \n\t")) == std::string("input firstInput, 1x_x_x_"));
    }

    TEST_CASE("TestPrune")
    {
        // Given
        std::string s = "\n\tHello, world! \r";

        // When
        armnn::Prune(s);

        // Then
        CHECK(s == std::string("Hello,world!"));
    }

    TEST_CASE("TestProcessPattern")
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
        CHECK(
            tensors1 ==
            Tensors({ std::make_pair("firstInput", armnn::SimpleInputOutput("firstInput", Shape({ 1, 0, 0, 0 }))),
                      std::make_pair("firstOutput", armnn::SimpleInputOutput("firstOutput", Shape({ 1, 0, 0, 0 }))) }));
        CHECK(layers1 == Layers({ armnn::SimpleLayer("Activation",
                                                     { armnn::SimpleInputOutput("firstInput", Shape({ 1, 0, 0, 0 })) },
                                                     { "firstOutput" }, { std::make_pair("function", "TanH") }) }));

        // And when
        armnn::ProcessPattern(buf2, tensors2, layers2);

        // Then
        CHECK(
            tensors2 ==
            Tensors({ std::make_pair("firstInput", armnn::SimpleInputOutput("firstInput", Shape({ 1, 0, 0, 0 }))),
                      std::make_pair("secondInput", armnn::SimpleInputOutput("secondInput", Shape({ 1, 1, 2, 3 }))),
                      std::make_pair("firstOutput", armnn::SimpleInputOutput("firstOutput", Shape({ 1, 0, 0, 0 }))) }));
        CHECK(layers2 == Layers({ armnn::SimpleLayer(
                             "StandIn",
                             { armnn::SimpleInputOutput("firstInput", Shape({ 1, 0, 0, 0 })),
                               armnn::SimpleInputOutput("secondInput", Shape({ 1, 1, 2, 3 })) },
                             { "firstOutput" },
                             { std::map<std::string, std::string>{ std::make_pair("function", "CustomOp"),
                                                                   std::make_pair("name", "somename") } }) }));

        // And when
        armnn::ProcessPattern(buf3, tensors3, layers3);

        // Then
        CHECK(tensors3 ==
              Tensors(
                  { std::make_pair("firstInput", armnn::SimpleInputOutput("firstInput", Shape({ 1, 0, 0, 0 }))),
                    std::make_pair("firstOutput", armnn::SimpleInputOutput("firstOutput", Shape({ 1, 0, 0, 0 }))),
                    std::make_pair("secondOutput", armnn::SimpleInputOutput("secondOutput", Shape({ 1, 0, 0, 0 }))) }));
        CHECK(layers3 ==
              Layers({ armnn::SimpleLayer("Excluded", { armnn::SimpleInputOutput("firstInput", Shape({ 1, 0, 0, 0 })) },
                                          { "firstOutput", "secondOutput" }, {}) }));
    }

    TEST_CASE("TestProcessBadInput")
    {
        const std::vector<std::string> buf = {
            "input_ firstInput, 1x_x_x_",
            "output?  firstOutput, 1x_x_x_",
            "Activation, (firstInput), (firstOutput), ((function=TanH))",
        };
        Tensors tensors;
        Layers layers;

        CHECK_THROWS_AS(armnn::ProcessPattern(buf, tensors, layers), armnn::ParseException);

        try
        {
            armnn::ProcessPattern(buf, tensors, layers);
        }
        catch (const armnn::ParseException& e)
        {
            std::string err = "Syntax error:\ninput_ firstInput, 1x_x_x_\nSyntax error:\noutput?  firstOutput, "
                              "1x_x_x_\nUndefined input: 'firstInput'\n";
            CHECK_EQ(err, e.what());
        }
    }

    template <const unsigned int SIZE>
    Mappings CreateMappings(TestLayerType originalType, TestLayerType replacementType,
                            std::array<const unsigned int, SIZE> inputDimensions,
                            std::array<const unsigned int, SIZE> outputDimensions)
    {
        Mappings ethosNMappings;
        std::map<std::string, LayerType> mapStringToLayerType = armnn::ethosnbackend::GetMapStringToLayerType();

        ethosNMappings =
            CreateSubstitutionMappings<SIZE>(originalType, replacementType, inputDimensions, outputDimensions);

        //Test if there is at least one mapping
        ARMNN_ASSERT((ethosNMappings.size() >= 1));
        //Test if the mapping layer types are as intended
        CHECK(((mapStringToLayerType.find(ethosNMappings[0].m_ReplacementLayers[0].m_LayerTypeName)->second) ==
               replacementType.layer));
        CHECK(((mapStringToLayerType.find(ethosNMappings[0].m_PatternLayers[0].m_LayerTypeName))->second ==
               originalType.layer));

        //Test for single layer mappings
        CHECK((ethosNMappings.size() == 1));
        CHECK((ethosNMappings[0].m_PatternLayers.size() == 1));
        CHECK((ethosNMappings[0].m_ReplacementLayers.size() == 1));

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
    armnn::SubgraphView::SubgraphViewPtr CreateUnoptimizedSubgraph(
        Graph & graph, SimpleLayer layer, std::array<const unsigned int, SIZE> inputDimensions,
        std::array<const unsigned int, SIZE> outputDimensions)
    {
        Layer *inputLayer, *outputLayer;
        SubgraphView operationSubgraph({}, {}, {});
        Shape inputOutputTensorShape;
        std::map<std::string, LayerType> mapStringToLayerType = armnn::ethosnbackend::GetMapStringToLayerType();
        LayerType type = mapStringToLayerType.find(layer.m_LayerTypeName)->second;

        TensorInfo inputInfo(static_cast<unsigned int>(inputDimensions.size()), inputDimensions.data(),
                             DataType::QAsymmU8, 1.0f, 0);
        TensorInfo outputInfo(static_cast<unsigned int>(outputDimensions.size()), outputDimensions.data(),
                              DataType::QAsymmU8, 1.0f, 0);

        if (type == LayerType::Activation)
        {
            std::string activationFuncOriginalLayer = layer.m_LayerParams.find("function")->second;
            std::string name                        = layer.m_LayerParams["name"];

            operationSubgraph =
                SubgraphView(ethosnbackend::CreateActivationLayer(graph, activationFuncOriginalLayer, name));
        }
        else if ((type == LayerType::Convolution2d) || (type == LayerType::TransposeConvolution2d) ||
                 (type == LayerType::DepthwiseConvolution2d))
        {
            unsigned int inputChannels = inputInfo.GetShape()[3];
            DataType weightDataType    = inputInfo.GetDataType();
            operationSubgraph          = SubgraphView(ethosnbackend::CreateConvolutionLayer(
                type, graph, inputChannels, layer.m_LayerParams, weightDataType, DataType::Signed32));
        }
        else if (type == LayerType::FullyConnected)
        {
            operationSubgraph =
                ethosnbackend::CreateFullyConnectedLayer(graph, inputInfo, outputInfo, layer.m_LayerParams);
        }
        else if (type == LayerType::Pooling2d)
        {
            operationSubgraph = SubgraphView(ethosnbackend::CreatePooling2dLayer(graph, layer.m_LayerParams));
        }

        CHECK(operationSubgraph.GetLayers().size() > 0);
        Layer* operationLayer = *operationSubgraph.GetLayers().begin();
        CHECK(operationLayer);
        operationLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

        // Construct the graph
        inputLayer = graph.AddLayer<InputLayer>(0, "input layer");
        CHECK(inputLayer);
        inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

        outputLayer = graph.AddLayer<OutputLayer>(0, "output layer");
        CHECK(outputLayer);

        // Connect the network
        inputLayer->GetOutputSlot(0).Connect(operationLayer->GetInputSlot(0));
        operationLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

        // Create the subgraph view for the whole network
        return std::make_unique<armnn::SubgraphView>(operationSubgraph);
    }

    // This function assumes that there is only one operation layer in the subgraph.
    // That is because CreateUnoptimizedSubgraph() creates a subgraph with one input
    // layer , one operation layer and one output layer. If in future, we want to
    // validate subgraphs with multiple operation layers, then this function should
    // be changed accordingly.
    bool IsLayerPresentInSubgraph(armnn::Graph & graph, LayerType type, AdditionalLayerParams params = { { "", "" } })
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
                            CHECK((m_PoolType == poolDesc.m_PoolType));
                        }

                        if (!params["stride"].empty())
                        {
                            auto stride = GetLayerParameterValue(params, "stride");
                            CHECK((poolDesc.m_StrideX == stride[ethosnbackend::STRIDE_X]));
                            CHECK((poolDesc.m_StrideY == stride[ethosnbackend::STRIDE_Y]));
                        }

                        if (!params["kernel"].empty())
                        {
                            auto kernel = GetLayerParameterValue(params, "kernel");
                            CHECK((poolDesc.m_PoolHeight = kernel[ethosnbackend::KERNEL_HEIGHT]));
                            CHECK((poolDesc.m_PoolWidth = kernel[ethosnbackend::KERNEL_WIDTH]));
                        }

                        if (!params["padding"].empty())
                        {
                            auto padding = GetLayerParameterValue(params, "padding");
                            CHECK((poolDesc.m_PadBottom == padding[ethosnbackend::PAD_BOTTOM]));
                            CHECK((poolDesc.m_PadLeft == padding[ethosnbackend::PAD_LEFT]));
                            CHECK((poolDesc.m_PadRight == padding[ethosnbackend::PAD_RIGHT]));
                            CHECK((poolDesc.m_PadTop == padding[ethosnbackend::PAD_TOP]));
                        }
                    }

                    if (actLayer != nullptr)
                    {
                        actDesc = actLayer->GetParameters();

                        if (!params["function"].empty())
                        {
                            auto actFunc    = armnn::ethosnbackend::GetMapStringToActivationFunction();
                            auto m_Function = actFunc.find(params["function"])->second;
                            CHECK((m_Function == actDesc.m_Function));
                        }
                    }

                    // Check for the common parameters ie 'name'
                    if (!params["name"].empty())
                    {
                        CHECK((params["name"] == layer->GetNameStr()));
                    }
                }
            }
        }
        return match;
    }

    template <const unsigned int SIZE>
    void TestSubgraphSubstitution(TestLayerType originalType, TestLayerType replacementType,
                                  std::array<const unsigned int, SIZE> inputDimensions,
                                  std::array<const unsigned int, SIZE> outputDimensions, bool validSubstitution = true)
    {
        using namespace testing_utils;
        Graph graph, graph2;

        EthosNConfig ethosnConfig = CreateEthosNConfig();

        auto ethosNMappings = CreateMappings<SIZE>(originalType, replacementType, inputDimensions, outputDimensions);

        auto subGraphOriginal  = CreateUnoptimizedSubgraph<SIZE>(graph, ethosNMappings[0].m_PatternLayers[0],
                                                                inputDimensions, outputDimensions);
        auto subGraphOriginal2 = CreateUnoptimizedSubgraph<SIZE>(graph2, ethosNMappings[0].m_PatternLayers[0],
                                                                 inputDimensions, outputDimensions);

        //Validate that the graph2 had the layer of the original type
        CHECK(IsLayerPresentInSubgraph(graph2, originalType.layer));

        // When
        OptimizationViews optimizationViews;
        armnn::CreatePreCompiledLayerInGraph(optimizationViews, *subGraphOriginal, ethosnConfig, ethosNMappings,
                                             ethosnConfig.QueryCapabilities(), {});
        ethosnbackend::ApplyMappings(ethosNMappings, graph2);

        // Then validate that armnn was able to compile the graph successfully
        CHECK(optimizationViews.Validate(*subGraphOriginal));
        CHECK(optimizationViews.GetSubstitutions().size() == 1);
        CHECK(optimizationViews.GetFailedSubgraphs().size() == 0);
        CHECK(optimizationViews.GetUntouchedSubgraphs().size() == 0);
        auto substitutions = optimizationViews.GetSubstitutions();
        CHECK(substitutions.size() == 1);
        bool subgraphsAreSame = (*subGraphOriginal == substitutions[0].m_SubstitutableSubgraph);
        CHECK(subgraphsAreSame);
        //Currently we replace a single layer with another single layer
        CHECK((substitutions[0].m_ReplacementSubgraph.GetLayers().size() == 1));

        // Validate that the substitution really took place. We need to do this as armnn
        // changes the layer type to pre-compiled
        CHECK((IsLayerPresentInSubgraph(graph2, replacementType.layer) == validSubstitution));
    }

    static const char* MAPPING_FILE_TEST_DIRECTORY = "armnn-ethos-n-backend/test/mapping-tests/";
    struct TestParseMappingFileData
    {
        const char* fileName;
        TestLayerTypeList layers;
        ExceptionCases exception     = NoException;
        std::string exceptionMessage = "";
    };

    // WARNING: If new entrys are added to this array a corresponding SUBCASE() needs to be added to TEST_CASE("TestParseMappingFile")

    static std::vector<TestParseMappingFileData> TestParseMappingFileDataset = {
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
          "inActivationBoundedReLu_outActivationTanh.txt",
          //.layers =
          {
              std::make_tuple(
                  //             .layer, .name
                  TestLayerType({ LayerType::Activation, "BoundedReLu" }),
                  TestLayerType({ LayerType::Activation, "TanH" })),
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
              TestLayerType({ LayerType::Activation, "BoundedReLu" }), TestLayerType({ LayerType::Pooling2d, "" })) } },
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
              TestLayerType({ LayerType::Floor, "" }), TestLayerType({ LayerType::Activation, "ReLu" })) } },
        { //.fileName =
          "inSoftmax_outActivationSigmoid.txt",
          //.layers =
          { std::make_tuple(
              //             .layer, .name
              TestLayerType({ LayerType::Softmax, "" }), TestLayerType({ LayerType::Activation, "Sigmoid" })) } },
        { //.fileName =
          "inConvolution2d_outPooling2d.txt",
          //.layers =
          { std::make_tuple(
              //             .layer, .name
              TestLayerType({ LayerType::Convolution2d, "" }), TestLayerType({ LayerType::Pooling2d, "" })) } },
        { //.fileName =
          "inLogSoftmax_outFullyConnected.txt",
          //.layers =
          { std::make_tuple(
              //             .layer, .name
              TestLayerType({ LayerType::LogSoftmax, "" }), TestLayerType({ LayerType::FullyConnected, "" })) } },
        { //.fileName =
          "multiLayerMapping.txt",
          //.layers =
          { std::make_tuple(
                //             .layer, .name
                TestLayerType({ LayerType::DepthwiseConvolution2d, "" }),
                TestLayerType({ LayerType::Convolution2d, "" })),
            std::make_tuple(
                //             .layer, .name
                TestLayerType({ LayerType::Output, "" }), TestLayerType({ LayerType::Pooling2d, "" })),
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
        // WARNING: If new entrys are added to this array a corresponding SUBCASE() needs to be added to TEST_CASE("TestParseMappingFile")
    };

    TEST_CASE("TestParseMappingFile")
    {
        TestParseMappingFileData arrayElement;
        SUBCASE(TestParseMappingFileDataset.at(0).fileName)
        {
            arrayElement = TestParseMappingFileDataset.at(0);
        }
        SUBCASE(TestParseMappingFileDataset.at(1).fileName)
        {
            arrayElement = TestParseMappingFileDataset.at(1);
        }
        SUBCASE(TestParseMappingFileDataset.at(2).fileName)
        {
            arrayElement = TestParseMappingFileDataset.at(2);
        }
        SUBCASE(TestParseMappingFileDataset.at(3).fileName)
        {
            arrayElement = TestParseMappingFileDataset.at(3);
        }
        SUBCASE(TestParseMappingFileDataset.at(4).fileName)
        {
            arrayElement = TestParseMappingFileDataset.at(4);
        }
        SUBCASE(TestParseMappingFileDataset.at(5).fileName)
        {
            arrayElement = TestParseMappingFileDataset.at(5);
        }
        SUBCASE(TestParseMappingFileDataset.at(6).fileName)
        {
            arrayElement = TestParseMappingFileDataset.at(6);
        }
        SUBCASE(TestParseMappingFileDataset.at(7).fileName)
        {
            arrayElement = TestParseMappingFileDataset.at(7);
        }
        SUBCASE(TestParseMappingFileDataset.at(8).fileName)
        {
            arrayElement = TestParseMappingFileDataset.at(8);
        }
        SUBCASE(TestParseMappingFileDataset.at(9).fileName)
        {
            arrayElement = TestParseMappingFileDataset.at(9);
        }
        SUBCASE(TestParseMappingFileDataset.at(10).fileName)
        {
            arrayElement = TestParseMappingFileDataset.at(10);
        }
        SUBCASE(TestParseMappingFileDataset.at(11).fileName)
        {
            arrayElement = TestParseMappingFileDataset.at(11);
        }
        SUBCASE(TestParseMappingFileDataset.at(12).fileName)
        {
            arrayElement = TestParseMappingFileDataset.at(12);
        }
        SUBCASE(TestParseMappingFileDataset.at(13).fileName)
        {
            arrayElement = TestParseMappingFileDataset.at(13);
        }
        SUBCASE(TestParseMappingFileDataset.at(14).fileName)
        {
            arrayElement = TestParseMappingFileDataset.at(14);
        }
        SUBCASE(TestParseMappingFileDataset.at(15).fileName)
        {
            arrayElement = TestParseMappingFileDataset.at(15);
        }
        SUBCASE(TestParseMappingFileDataset.at(16).fileName)
        {
            arrayElement = TestParseMappingFileDataset.at(16);
        }
        SUBCASE(TestParseMappingFileDataset.at(17).fileName)
        {
            arrayElement = TestParseMappingFileDataset.at(17);
        }
        SUBCASE(TestParseMappingFileDataset.at(18).fileName)
        {
            arrayElement = TestParseMappingFileDataset.at(18);
        }
        SUBCASE(TestParseMappingFileDataset.at(19).fileName)
        {
            arrayElement = TestParseMappingFileDataset.at(19);
        }
        SUBCASE(TestParseMappingFileDataset.at(20).fileName)
        {
            arrayElement = TestParseMappingFileDataset.at(20);
        }

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
            parsedMapping = ReadMappingsFromFile(fullFileName.c_str());
            gotException  = ExceptionCases::NoException;
        }
        catch (const armnn::ParseException& e)
        {
            gotException = ExceptionCases::ParseException;
            CHECK((std::string(e.what()).find(exceptionMessage) != std::string::npos));
        }
        catch (const armnn::InvalidArgumentException& e)
        {
            gotException = ExceptionCases::InvalidArgumentException;
            CHECK((std::string(e.what()).find(exceptionMessage) != std::string::npos));
        }
        //Check the result
        CHECK_EQ(gotException, expectException);
        CHECK(inputMapping == parsedMapping);
    }

    TEST_CASE("TestAllSubgraphSubstitution")
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
            replacement.layer = LayerType::Activation;
            replacement.name  = "TanH";
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
                CHECK((std::string(e.what()).find(err) != std::string::npos));
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

    TEST_CASE("TestLayerInclusion")
    {
        // Given
        EthosNConfig config = CreateEthosNConfig();
        TensorInfo inputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
        TensorInfo outputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f / 256, 0);
        ActivationDescriptor activationDescriptor;
        StandInDescriptor standInDescriptor{ 1, 1 };
        std::string reason;

        // When
        EthosNLayerSupport layerSupport(config, EthosNMappings(), config.QueryCapabilities());

        // Then
        CHECK(layerSupport.IsActivationSupported(inputInfo, outputInfo, activationDescriptor, reason) == true);
        CHECK(reason.empty());
        CHECK(layerSupport.IsStandInSupported(std::vector<const TensorInfo*>{ &inputInfo },
                                              std::vector<const TensorInfo*>{ &outputInfo }, standInDescriptor,
                                              reason) == true);
        CHECK(reason.empty());
    }

    TEST_CASE("TestAdditionalParameters")
    {
        // Given
        EthosNConfig config = CreateEthosNConfig();

        typedef std::string (*CreateMappingsWithAdditionalArgs)();
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
            auto ethosNMappings = ParseMappings(test.createMappingFunc().c_str());
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

            CHECK(
                IsLayerPresentInSubgraph(graph, originalLayerType, ethosNMappings[0].m_PatternLayers[0].m_LayerParams));

            // Then
            try
            {
                ethosnbackend::ApplyMappings(ethosNMappings, graph);
                CHECK(IsLayerPresentInSubgraph(graph, replacementLayerType,
                                               ethosNMappings[0].m_ReplacementLayers[0].m_LayerParams));
            }
            catch (const armnn::InvalidArgumentException& e)
            {
                gotException = ExceptionCases::InvalidArgumentException;
                CHECK((std::string(e.what()).find(exceptionMessage) != std::string::npos));
            }
            catch (const armnn::ParseException& e)
            {
                gotException = ExceptionCases::InvalidArgumentException;
                CHECK((std::string(e.what()).find(exceptionMessage) != std::string::npos));
            }
            CHECK((gotException == expectException));
        }
    }

    // A test which parses a syntactically incorrect mapping file.
    // The file is syntactically incorrect as "pattern:" is missing as the first line.
    TEST_CASE("TestIncorrectSyntaxMappingFile1")
    {
        using namespace testing_utils;

        // Given
        std::stringstream os;
        os << "input firstInput, 1x_x_x_\n";
        os << "output firstOutput, 1x_x_x_\n";
        os << "Activation, (firstInput), (firstOutput), ((function=TanH))\n";
        os << "graph-replacement:\n";
        os << "Activation, (firstInput), (firstOutput), ((function=Sigmoid), (name=SigmoidFunc))";
        os.seekg(0);

        // When
        try
        {
            EthosNMappings mappings = ParseMappings(os);
        }
        catch (const armnn::ParseException& e)
        {
            std::string err = "Syntax error in mapping file";
            CHECK((e.what() == err));
        }
    }

    // A test which parses a syntactically incorrect mapping file.
    // The file is syntactically incorrect as "graph-replacement:" is missing as the subsequent section after "pattern:".
    TEST_CASE("TestIncorrectSyntaxMappingFile2")
    {
        using namespace testing_utils;

        // Given
        std::stringstream os;
        os << "pattern:\n";
        os << "input firstInput, 1x_x_x_\n";
        os.seekg(0);

        // When
        try
        {
            EthosNMappings mappings = ParseMappings(os);
        }
        catch (const armnn::ParseException& e)
        {
            std::string err = "Syntax error in mapping file";
            CHECK((e.what() == err));
        }
    }

    // A test which parses a syntactically incorrect mapping file.
    // The file is syntactically incorrect as "pattern:" is missing as the first line.
    TEST_CASE("TestIncorrectSyntaxMappingFile3")
    {
        using namespace testing_utils;

        // Given
        std::stringstream os;
        os << "patternn:\n";
        os << "input firstInput, 1x_x_x_\n";
        os << "output firstOutput, 1x_x_x_\n";
        os << "Activation, (firstInput), (firstOutput), ((function=TanH))\n";
        os << "graph-replacement:\n";
        os << "Activation, (firstInput), (firstOutput), ((function=Sigmoid), (name=SigmoidFunc))";
        os.seekg(0);

        // When
        try
        {
            EthosNMappings mappings = ParseMappings(os);
        }
        catch (const armnn::ParseException& e)
        {
            std::string err = "Syntax error in mapping file";
            CHECK((e.what() == err));
        }
    }

    // A test which parses a syntactically incorrect mapping file.
    // The file is syntactically incorrect as "pattern:" is missing as the first line.
    TEST_CASE("TestIncorrectSyntaxMappingFile4")
    {
        using namespace testing_utils;

        // Given
        std::stringstream os;
        os << "graph-replacement:\n";
        os << "Activation, (firstInput), (firstOutput), ((function=Sigmoid), (name=SigmoidFunc))";
        os.seekg(0);

        // When
        try
        {
            EthosNMappings mappings = ParseMappings(os);
        }
        catch (const armnn::ParseException& e)
        {
            std::string err = "Syntax error in mapping file";
            CHECK((e.what() == err));
        }
    }

    // A test which parses a syntactically incorrect mapping file.
    // The file is syntactically incorrect as "pattern:" is missing as the first line.
    TEST_CASE("TestIncorrectSyntaxMappingFile5")
    {
        using namespace testing_utils;

        // Given
        std::stringstream os;
        os << "graph-replacement:\n";
        os << "Activation, (firstInput), (firstOutput), ((function=Sigmoid), (name=SigmoidFunc))";
        os << "pattern:\n";
        os.seekg(0);

        // When
        try
        {
            EthosNMappings mappings = ParseMappings(os);
        }
        catch (const armnn::ParseException& e)
        {
            std::string err = "Syntax error in mapping file";
            CHECK((e.what() == err));
        }
    }

    // A test which parses an empty mapping file.
    TEST_CASE("TestEmptyMappingFile")
    {
        using namespace testing_utils;

        // Given
        std::stringstream os;
        os << "\n\t\n";
        os.seekg(0);

        // When
        EthosNMappings mappings = ParseMappings(os);

        // Then
        CHECK((mappings.size() == 0));
    }

    // A test which parses a mapping file containing only comments.
    TEST_CASE("TestCommentsOnlyMappingFile")
    {
        using namespace testing_utils;

        // Given
        std::stringstream os;
        os << "# This is a mapping file";
        os << "# This does not contain any mappings";
        os.seekg(0);

        // When
        EthosNMappings mappings = ParseMappings(os);

        // Then
        CHECK((mappings.size() == 0));
    }

    // A test which parses a mapping file containing mappings and comments.
    TEST_CASE("TestMappingFileWithComments")
    {
        using namespace testing_utils;

        // Given
        std::stringstream os;
        os << "pattern:\n";
        os << "# First input \n";
        os << "input firstInput, 1x_x_x_\n";
        os << "# First output \n";
        os << "output firstOutput, 1x_x_x_\n";
        os << "# Layer to be replaced \n";
        os << "Activation, (firstInput), (firstOutput), ((function=TanH))\n";
        os << "graph-replacement:\n";
        os << "# Replacement layer \n";
        os << "Activation, (firstInput), (firstOutput), ((function=Sigmoid), (name=SigmoidFunc))";

        os.seekg(0);

        // When
        EthosNMappings mappings = ParseMappings(os);

        // Then
        CHECK((mappings.size() == 1));
    }

    TEST_CASE("TestLayerSubstitutionWithName")
    {
        // Given
        Graph graph;
        EthosNConfig config  = CreateEthosNConfig();
        std::string mappings = CreateMappingsWithLayerName();
        std::array<const unsigned int, 4> inputDimensions{ { 1, 16, 16, 16 } };
        std::array<const unsigned int, 4> outputDimensions{ { 1, 16, 16, 16 } };
        auto ethosNMappings = ParseMappings(mappings.c_str());
        auto originalLayerType =
            armnn::ethosnbackend::GetLayerType(ethosNMappings[0].m_PatternLayers[0].m_LayerTypeName);
        auto replacementLayerType =
            armnn::ethosnbackend::GetLayerType(ethosNMappings[0].m_ReplacementLayers[0].m_LayerTypeName);

        // When
        auto subGraphOriginal = CreateUnoptimizedSubgraph<inputDimensions.size()>(
            graph, ethosNMappings[0].m_PatternLayers[0], inputDimensions, outputDimensions);
        CHECK(IsLayerPresentInSubgraph(graph, originalLayerType));
        ethosnbackend::ApplyMappings(ethosNMappings, graph);

        // Then
        CHECK(IsLayerPresentInSubgraph(graph, replacementLayerType));
    }

    TEST_CASE("TestLayerSubstitutionWithNameMismatch")
    {
        // Given
        Graph graph;
        EthosNConfig config  = CreateEthosNConfig();
        std::string mappings = CreateMappingsWithLayerName();
        std::array<const unsigned int, 4> inputDimensions{ { 1, 16, 16, 16 } };
        std::array<const unsigned int, 4> outputDimensions{ { 1, 16, 16, 16 } };
        auto ethosNMappings = ParseMappings(mappings.c_str());
        auto originalLayerType =
            armnn::ethosnbackend::GetLayerType(ethosNMappings[0].m_PatternLayers[0].m_LayerTypeName);
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
        CHECK(IsLayerPresentInSubgraph(graph, originalLayerType));
        ethosnbackend::ApplyMappings(ethosNMappings, graph);

        // Then
        //Then the substitution should fail
        CHECK(!(IsLayerPresentInSubgraph(graph, replacementLayerType)));
        // And the graph should still contain the original layer
        CHECK(IsLayerPresentInSubgraph(graph, originalLayerType));
    }

    TEST_CASE("TestLayerExclusion")
    {
        // Given
        EthosNConfig config     = CreateEthosNConfig();
        std::string mappingsStr = CreateExclusionMappings();
        EthosNMappings mappings = ParseMappings(mappingsStr.c_str());
        TensorInfo inputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
        TensorInfo outputInfo({ 1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f / 256, 0);
        ActivationDescriptor activationDescriptor1;
        ActivationDescriptor activationDescriptor2;
        activationDescriptor1.m_Function = ActivationFunction::Sigmoid;
        activationDescriptor2.m_Function = ActivationFunction::TanH;
        StandInDescriptor standInDescriptor{ 1, 1 };
        std::string reason;

        // When
        EthosNLayerSupport layerSupport(config, mappings, config.QueryCapabilities());

        // Then
        CHECK(layerSupport.IsActivationSupported(inputInfo, outputInfo, activationDescriptor1, reason) == true);
        CHECK(layerSupport.IsActivationSupported(inputInfo, outputInfo, activationDescriptor2, reason) == false);
        CHECK(reason == "Layer declared excluded in mapping file");
        CHECK(layerSupport.IsStandInSupported(std::vector<const TensorInfo*>{ &inputInfo },
                                              std::vector<const TensorInfo*>{ &outputInfo }, standInDescriptor,
                                              reason) == false);
        CHECK(reason == "Layer declared excluded in mapping file");
    }

    TEST_CASE("TestLayerExclusionViaArmnn")
    {
        // Given
        EthosNConfig ethosnConfig = CreateEthosNConfig();
        std::string mappings      = CreateExclusionMappings();
        INetworkPtr net(INetwork::Create());
        CreateUnoptimizedNetwork(*net);

        // When
        IRuntime::CreationOptions options;
        IRuntimePtr runtime(IRuntime::Create(options));
        std::vector<BackendId> backends = { EthosNBackendId(), "CpuRef" };

        IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());

        // Then
        armnn::Graph& optimizedGraph = GetGraphForTesting(optNet.get());
        Graph::ConstIterator layer   = optimizedGraph.cbegin();
        auto inputLayer              = *layer;
        CHECK((inputLayer->GetBackendId() == EthosNBackendId()));
        ++layer;
        auto convolutionLayer = *layer;
        CHECK((convolutionLayer->GetBackendId() == EthosNBackendId()));
        ++layer;
        auto activationLayer = *layer;
        CHECK((activationLayer->GetBackendId() == BackendId(Compute::CpuRef)));
        ++layer;
        auto outputLayer = *layer;
        CHECK((outputLayer->GetBackendId() == BackendId(Compute::CpuRef)));
    }

    TEST_CASE("TestLayerInvalidExclusionViaArmnn")
    {
        // Given
        EthosNConfig ethosnConfig                = CreateEthosNConfig();
        const std::vector<std::string> mappings1 = {
            "input firstInput, 1x_x_x_",
            "output  firstOutput, 1x_x_x_",
            "Excluded1, (firstInput), (firstOutput), ((function=TanH))",
        };
        Tensors tensors;
        Layers layers;

        CHECK_THROWS_AS(armnn::ProcessPattern(mappings1, tensors, layers), armnn::ParseException);

        // When
        try
        {
            armnn::ProcessPattern(mappings1, tensors, layers);
        }
        // Then
        catch (const armnn::ParseException& e)
        {
            std::string err = "Syntax error:\nExcluded1, (firstInput), (firstOutput), ((function=TanH))\n";
            CHECK_EQ(err, e.what());
        }
    }

    const char* REPLACEMENT_FILE_TEST_DIRECTORY = "armnn-ethos-n-backend/test/replacement-tests/";

    TEST_CASE("TestGraphReplace")
    {

        //Get the input parameter of the tests
        const SISOCatOneGraphFactory& factory = SISOCatOneGraphFactory();
        std::string mappingFileName(REPLACEMENT_FILE_TEST_DIRECTORY);
        mappingFileName.append(factory.GetMappingFileName());

        std::unique_ptr<NetworkImpl> initNetImplPtr = factory.GetInitialGraph();
        Graph modifiedGraph                         = initNetImplPtr->GetGraph();

        std::unique_ptr<NetworkImpl> expectedNetImplPtr = factory.GetExpectedModifiedGraph();
        Graph expectedGraph                             = expectedNetImplPtr->GetGraph();

        const SubgraphView expectedGraphView(expectedGraph);

        EthosNMappings parsedMapping = ReadMappingsFromFile(mappingFileName.c_str());

        ethosnbackend::ApplyMappings(parsedMapping, modifiedGraph);
        SubgraphView modifiedGraphView(modifiedGraph);

        const SubgraphView::Layers& modifiedGraphLayers = modifiedGraphView.GetLayers();
        const SubgraphView::Layers& expectedGraphLayers = expectedGraphView.GetLayers();

        CHECK_EQ(modifiedGraphLayers.size(), expectedGraphLayers.size());

        SubgraphView::Layers::const_iterator modGraphLayerItr = modifiedGraphLayers.begin();
        SubgraphView::Layers::const_iterator expGraphLayerItr = expectedGraphLayers.begin();
        const Layer* previousLayer                            = nullptr;
        size_t modGraphLayerSize                              = modifiedGraphLayers.size();
        for (size_t layerIdx = 0; layerIdx < modGraphLayerSize; ++layerIdx, ++modGraphLayerItr, ++expGraphLayerItr)
        {
            bool isFirstLayer     = layerIdx == 0;
            const Layer* modLayer = *modGraphLayerItr;
            const Layer* expLayer = *expGraphLayerItr;
            unsigned int expectedNumInSlot;
            unsigned int expectedNumOutSlot;
            ARMNN_ASSERT_MSG(modLayer->GetNumInputSlots() <= 1,
                             "Multi input layers are not yet supported by this test\n\r");
            ARMNN_ASSERT_MSG(modLayer->GetNumOutputSlots() <= 1,
                             "Multi output layers are not yet supported by this test\n\r");

            CheckLayerEquals(modLayer, expLayer, "Mod == Exp ", layerIdx);

            if (previousLayer)
            {
                CheckLayerEquals(previousLayer, modLayer, "Mod == Prev ", layerIdx);
            }

            if (layerIdx < (modGraphLayerSize - 1))    //first to n-1 layer
            {
                expectedNumOutSlot = 1;
                if (isFirstLayer)
                {
                    expectedNumInSlot = 0;
                }
                else
                {
                    expectedNumInSlot = 1;
                }
                previousLayer = &(modLayer->GetOutputSlot(0).GetConnection(0)->GetOwningLayer());
            }
            else    //last layer
            {
                expectedNumOutSlot = 0;
                expectedNumInSlot  = 1;
            }
            CHECK_EQ(modLayer->GetNumInputSlots(), expectedNumInSlot);
            CHECK_EQ(modLayer->GetNumOutputSlots(), expectedNumOutSlot);
        }
    }
}
