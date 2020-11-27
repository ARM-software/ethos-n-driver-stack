//
// Copyright © 2020 Arm Ltd. All rights reserved.
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
#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/test/CommonTestUtils.hpp>

// The include order is important. Turn off clang-format
// clang-format off
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
// clang-format on

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

    INetworkPtr initINetPtr  = factory.GetInitialGraph();
    const INetwork& initINet = *initINetPtr;
    const Network& initNet   = *PolymorphicDowncast<const Network*>(&initINet);
    Graph modifiedGraph      = initNet.GetGraph();

    INetworkPtr expectINetPtr    = factory.GetExpectedModifiedGraph();
    const INetwork& expectedINet = *expectINetPtr;
    const Network& expectedNet   = *PolymorphicDowncast<const Network*>(&expectedINet);
    Graph expectedGraph          = expectedNet.GetGraph();
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

    const std::unique_ptr<ScopedCpuTensorHandle>& modWeight = GetWeight(modLayerWithParam);
    const std::unique_ptr<ScopedCpuTensorHandle>& expWeight = GetWeight(expLayerWithParam);

    bool weightEquals = modWeight->GetTensorInfo() == expWeight->GetTensorInfo();
    BOOST_TEST(weightEquals, paramName << " weights doesn't match at layer index: " << layerIdx
                                       << " nameMod: " << modLayer->GetName() << " nameExp: " << expLayer->GetName());

    const std::unique_ptr<ScopedCpuTensorHandle>& modBias = GetBias(modLayerWithParam);
    const std::unique_ptr<ScopedCpuTensorHandle>& expBias = GetBias(expLayerWithParam);

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