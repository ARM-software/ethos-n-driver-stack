//
// Copyright © 2020-2021 Arm Ltd.
// SPDX-License-Identifier: Apache-2.0
//
#include "EthosNReplaceUnsupported.hpp"

#include <armnn/INetwork.hpp>
#include <backendsCommon/test/CommonTestUtils.hpp>

#include <boost/test/unit_test.hpp>

#include <numeric>

using namespace armnn;

// By default, specific unsupported layer patterns are substituted for patterns
// that can be optimized on the backend.
BOOST_AUTO_TEST_SUITE(EthosNReplaceUnsupported)

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
