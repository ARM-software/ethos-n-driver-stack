//
// Copyright © 2020-2021 Arm Ltd.
// SPDX-License-Identifier: Apache-2.0
//
#include "EthosNBackend.hpp"
#include "EthosNBackendUtils.hpp"
#include "EthosNMapping.hpp"
#include "EthosNTestUtils.hpp"
#include "Network.hpp"
#include "replacement-tests/SISOCatOneGraphFactory.hpp"

#include <armnn/INetwork.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <backendsCommon/TensorHandle.hpp>
#include <backendsCommon/test/CommonTestUtils.hpp>

// The include order is important. Turn off clang-format
// clang-format off
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
// clang-format on

#include <numeric>

using namespace armnn;

namespace bdata = boost::unit_test::data;

template <typename Parameter>
static void CheckLayerWithParametersEquals(const Layer* modLayer,
                                           const Layer* expLayer,
                                           std::string paramName,
                                           size_t layerIdx);

template <typename ConvolutionLayer>
static void CheckConvolutionLayerDataEquals(const Layer* modLayer,
                                            const Layer* expLayer,
                                            std::string paramName,
                                            size_t layerIdx);

static void CheckLayerEquals(const Layer* modLayer, const Layer* expLayer, std::string paramName, size_t layerIdx);

BOOST_AUTO_TEST_SUITE(EthosNReplacement)

const char* REPLACEMENT_FILE_TEST_DIRECTORY = "armnn-ethos-n-backend/test/replacement-tests/";
struct TestReplacementData
{
    const IReplacementTestGraphFactory& factory;
};

static TestReplacementData TestParseMappingFileDataset[] = { { //.factory
                                                               SISOCatOneGraphFactory() } };

std::ostream& boost_test_print_type(std::ostream& os, const TestReplacementData& data)
{
    os << "Factory: " << data.factory.GetName();
    return os;
}

BOOST_DATA_TEST_CASE(TestGraphReplace,
                     bdata::xrange(ARRAY_SIZE(TestParseMappingFileDataset)) ^ bdata::make(TestParseMappingFileDataset),
                     xr,
                     arrayElement)
{
    (void)(xr);

    //Get the input parameter of the tests
    const IReplacementTestGraphFactory& factory = arrayElement.factory;
    std::string mappingFileName(REPLACEMENT_FILE_TEST_DIRECTORY);
    mappingFileName.append(factory.GetMappingFileName());

    std::unique_ptr<NetworkImpl> initNetImplPtr = factory.GetInitialGraph();
    Graph modifiedGraph                         = initNetImplPtr->GetGraph();

    std::unique_ptr<NetworkImpl> expectedNetImplPtr = factory.GetExpectedModifiedGraph();
    Graph expectedGraph                             = expectedNetImplPtr->GetGraph();

    const SubgraphView expectedGraphView(expectedGraph);

    EthosNMappings parsedMapping = GetMappings(mappingFileName);

    ethosnbackend::ApplyMappings(parsedMapping, modifiedGraph);
    SubgraphView modifiedGraphView(modifiedGraph);

    const SubgraphView::Layers& modifiedGraphLayers = modifiedGraphView.GetLayers();
    const SubgraphView::Layers& expectedGraphLayers = expectedGraphView.GetLayers();

    BOOST_CHECK_EQUAL(modifiedGraphLayers.size(), expectedGraphLayers.size());

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
        BOOST_CHECK_EQUAL(modLayer->GetNumInputSlots(), expectedNumInSlot);
        BOOST_CHECK_EQUAL(modLayer->GetNumOutputSlots(), expectedNumOutSlot);
    }
}

BOOST_AUTO_TEST_SUITE_END()

// By default, specific unsupported layer patterns are substituted for patterns
// that can be optimized on the backend.
BOOST_AUTO_TEST_SUITE(EthosNDefaultLayerReplacement)

// Multiplication operations that take as input a Constant tensor in the shape
// { 1, 1, 1, C } can be substituted for DepthwiseConvolution2d.
//
// Original pattern:
// Input    ->
//              Multiplication -> Output
// Constant ->
//
// Expected modified pattern:
// Input -> DepthwiseConvolution2d -> Output
BOOST_AUTO_TEST_CASE(ConstMulToDepthwiseReplacement)
{
    auto net = std::make_unique<NetworkImpl>();

    TensorInfo inputInfo({ 1, 8, 8, 16 }, DataType::QAsymmU8, 1.0f, 0);
    TensorInfo constInfo({ 1, 1, 1, 16 }, DataType::QAsymmU8, 0.9f, 0);
    TensorInfo outputInfo({ 1, 8, 8, 16 }, DataType::QAsymmU8, 1.0f, 0);

    std::vector<uint8_t> constData(constInfo.GetNumElements(), 0);
    std::iota(constData.begin(), constData.end(), 0);
    ConstTensor constTensor(constInfo, constData);

    // Add the original pattern
    IConnectableLayer* const input    = net->AddInputLayer(0, "input");
    IConnectableLayer* const constant = net->AddConstantLayer(constTensor, "const");
    IConnectableLayer* const mul      = net->AddMultiplicationLayer("mul");
    IConnectableLayer* const output   = net->AddOutputLayer(0, "output");

    // Create connections between layers
    input->GetOutputSlot(0).SetTensorInfo(inputInfo);
    constant->GetOutputSlot(0).SetTensorInfo(constInfo);
    mul->GetOutputSlot(0).SetTensorInfo(outputInfo);

    input->GetOutputSlot(0).Connect(mul->GetInputSlot(0));
    constant->GetOutputSlot(0).Connect(mul->GetInputSlot(1));
    mul->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    // Substitute the subgraph and check for expected pattern and connections
    Graph pattern = net->GetGraph();
    ethosnbackend::ReplaceUnsupportedLayers(pattern);

    BOOST_CHECK(pattern.GetNumLayers() == 3);

    const std::vector<Layer*> vecPattern(pattern.begin(), pattern.end());

    Layer* inputLayer     = vecPattern[0];
    Layer* depthwiseLayer = vecPattern[1];
    Layer* outputLayer    = vecPattern[2];

    BOOST_CHECK(inputLayer->GetType() == LayerType::Input);
    BOOST_CHECK(depthwiseLayer->GetType() == LayerType::DepthwiseConvolution2d);
    BOOST_CHECK(outputLayer->GetType() == LayerType::Output);

    Layer* depthwiseInput  = &depthwiseLayer->GetInputSlots()[0].GetConnectedOutputSlot()->GetOwningLayer();
    Layer* depthwiseOutput = &depthwiseLayer->GetOutputSlots()[0].GetConnections()[0]->GetOwningLayer();
    BOOST_CHECK(depthwiseInput == inputLayer);
    BOOST_CHECK(depthwiseOutput == outputLayer);

    Layer* inputNextLayer  = &inputLayer->GetOutputSlots()[0].GetConnections()[0]->GetOwningLayer();
    Layer* outputPrevLayer = &outputLayer->GetInputSlots()[0].GetConnectedOutputSlot()->GetOwningLayer();
    BOOST_CHECK(inputNextLayer == depthwiseLayer);
    BOOST_CHECK(outputPrevLayer == depthwiseLayer);

    // Depthwise weights should be exact with the Constant data
    const uint8_t* dwWeightData =
        PolymorphicPointerDowncast<DepthwiseConvolution2dLayer>(depthwiseLayer)->m_Weight->GetConstTensor<uint8_t>();
    std::vector<uint8_t> depthwiseWeights(dwWeightData, dwWeightData + constData.size());
    BOOST_CHECK(depthwiseWeights == constData);
}

BOOST_AUTO_TEST_SUITE_END()

//Helper functions
template <typename Parameter>
static void
    CheckLayerWithParametersEquals(const Layer* modLayer, const Layer* expLayer, std::string paramName, size_t layerIdx)
{
    const LayerWithParameters<Parameter>* modLayerWithParam =
        PolymorphicDowncast<const LayerWithParameters<Parameter>*>(modLayer);
    const LayerWithParameters<Parameter>* expLayerWithParam =
        PolymorphicDowncast<const LayerWithParameters<Parameter>*>(expLayer);

    bool areParamsEquals = modLayerWithParam->GetParameters() == expLayerWithParam->GetParameters();
    BOOST_TEST(areParamsEquals, paramName << " at layer index: " << layerIdx << " nameMod: " << modLayer->GetName()
                                          << " nameExp: " << expLayer->GetName());
}

template <typename ConvolutionLayer>
static void CheckConvolutionLayerDataEquals(const Layer* modLayer,
                                            const Layer* expLayer,
                                            std::string paramName,
                                            size_t layerIdx)
{
    const ConvolutionLayer* modLayerWithParam = PolymorphicDowncast<const ConvolutionLayer*>(modLayer);
    const ConvolutionLayer* expLayerWithParam = PolymorphicDowncast<const ConvolutionLayer*>(expLayer);

    const std::shared_ptr<ConstTensorHandle> modWeight = GetWeight(modLayerWithParam);
    const std::shared_ptr<ConstTensorHandle> expWeight = GetWeight(expLayerWithParam);

    bool weightEquals = modWeight->GetTensorInfo() == expWeight->GetTensorInfo();
    BOOST_TEST(weightEquals, paramName << " weights doesn't match at layer index: " << layerIdx
                                       << " nameMod: " << modLayer->GetName() << " nameExp: " << expLayer->GetName());

    const std::shared_ptr<ConstTensorHandle> modBias = GetBias(modLayerWithParam);
    const std::shared_ptr<ConstTensorHandle> expBias = GetBias(expLayerWithParam);

    bool biasEquals = modBias->GetTensorInfo() == expBias->GetTensorInfo();
    BOOST_TEST(biasEquals, paramName << " bias doesn't match at layer index: " << layerIdx
                                     << " nameMod: " << modLayer->GetName() << " nameExp: " << expLayer->GetName());
}

static void CheckLayerEquals(const Layer* modLayer, const Layer* expLayer, std::string paramName, size_t layerIdx)
{
    BOOST_CHECK_EQUAL(modLayer->GetNameStr(), expLayer->GetNameStr());

    LayerType modLayerType    = modLayer->GetType();
    LayerType expLayerType    = expLayer->GetType();
    std::string modTypeString = GetLayerTypeAsCString(modLayerType);
    BOOST_TEST(static_cast<int>(modLayerType) == static_cast<int>(expLayerType),
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
        BOOST_ERROR(assertMessage);
        ARMNN_ASSERT(false);
    }
}
