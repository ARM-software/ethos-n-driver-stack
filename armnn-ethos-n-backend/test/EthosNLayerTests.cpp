//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#include "EthosNLayerTests.hpp"

#include "EthosNWorkloadFactoryHelper.hpp"

#include <test/UnitTests.hpp>

#include <EthosNWorkloadFactory.hpp>
#include <backendsCommon/test/LayerTests.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(Compute_EthosN)

using FactoryType = armnn::EthosNWorkloadFactory;

// ============================================================================
// UNIT tests

ARMNN_AUTO_TEST_CASE(PreCompiledActivationRelu, PreCompiledActivationReluTest)
ARMNN_AUTO_TEST_CASE(PreCompiledActivationRelu1, PreCompiledActivationRelu1Test)
ARMNN_AUTO_TEST_CASE(PreCompiledActivationRelu6, PreCompiledActivationRelu6Test)

ARMNN_AUTO_TEST_CASE(PreCompiledConvolution2d, PreCompiledConvolution2dTest)
ARMNN_AUTO_TEST_CASE(PreCompiledConvolution2dStride2x2, PreCompiledConvolution2dStride2x2Test)

ARMNN_AUTO_TEST_CASE(PreCompiledDepthwiseConvolution2d, PreCompiledDepthwiseConvolution2dTest)
ARMNN_AUTO_TEST_CASE(PreCompiledDepthwiseConvolution2dStride2x2, PreCompiledDepthwiseConvolution2dStride2x2Test)

ARMNN_AUTO_TEST_CASE(PreCompiledTransposeConvolution2dStride2x2, PreCompiledTransposeConvolution2dStride2x2Test)

ARMNN_AUTO_TEST_CASE(PreCompiledConvolution2dWithAssymetricSignedWeights,
                     PreCompiledConvolution2dWithAssymetricSignedWeightsTest)

ARMNN_AUTO_TEST_CASE(PreCompiledConvolution2dWithSymetricSignedWeights,
                     PreCompiledConvolution2dWithSymetricSignedWeightsTest)

ARMNN_AUTO_TEST_CASE(PreCompiledFullyConnected, PreCompiledFullyConnectedTest)
ARMNN_AUTO_TEST_CASE(PreCompiledFullyConnected4d, PreCompiledFullyConnected4dTest)

ARMNN_AUTO_TEST_CASE(PreCompiledMaxPooling2d, PreCompiledMaxPooling2dTest)

ARMNN_AUTO_TEST_CASE(PreCompiledSplitter, PreCompiledSplitterTest)

ARMNN_AUTO_TEST_CASE(PreCompiledDepthToSpace, PreCompiledDepthToSpaceTest)

ARMNN_AUTO_TEST_CASE(PreCompiledLeakyRelu, PreCompiledLeakyReluTest)

ARMNN_AUTO_TEST_CASE(PreCompiledAddition, PreCompiledAdditionTest)

ARMNN_AUTO_TEST_CASE(PreCompiledMultiInput, PreCompiledMultiInputTest)
ARMNN_AUTO_TEST_CASE(PreCompiledMultiOutput, PreCompiledMultiOutputTest)

ARMNN_AUTO_TEST_CASE(PreCompiled1dTensor, PreCompiled1dTensorTest)
ARMNN_AUTO_TEST_CASE(PreCompiled2dTensor, PreCompiled2dTensorTest)
ARMNN_AUTO_TEST_CASE(PreCompiled3dTensor, PreCompiled3dTensorTest)

ARMNN_AUTO_TEST_CASE(PreCompiledConstMulToDepthwise, PreCompiledConstMulToDepthwiseTest)

BOOST_AUTO_TEST_CASE(TestInvalidLayerName)
{
    BOOST_CHECK_THROW(armnn::ethosnbackend::GetLayerType("Excluded"), armnn::InvalidArgumentException);

    try
    {
        armnn::ethosnbackend::GetLayerType("Excluded");
    }
    catch (const armnn::InvalidArgumentException& e)
    {
        std::string err = "layername \"Excluded\" is not valid";
        BOOST_CHECK_EQUAL(err, e.what());
    }
}

BOOST_AUTO_TEST_SUITE_END()
