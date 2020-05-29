//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
#include "SISOCatOneGraphFactory.hpp"

#include "EthosNConfig.hpp"
#include "EthosNLayerSupport.hpp"
#include "EthosNMapping.hpp"
#include "EthosNTestUtils.hpp"

#include <boost/test/unit_test.hpp>

using namespace armnn;

const std::string& SISOCatOneGraphFactory::GetName() const
{
    static const std::string name("SISOCatOneGraphFactory");
    return name;
}

//input-->Activation(TanH)-->Softmax-->Rsqrt-->FullyCoonected-->output
INetworkPtr SISOCatOneGraphFactory::GetInitialGraph() const
{
    INetworkPtr iNetPtr(INetwork::Create());
    INetwork& net = *iNetPtr;

    armnn::IConnectableLayer* const inputLayer = net.AddInputLayer(0, "input layer");
    BOOST_TEST(inputLayer);

    //Layer 1
    ActivationDescriptor tanDesc{};
    tanDesc.m_A                               = 100;
    tanDesc.m_B                               = 0;
    tanDesc.m_Function                        = ActivationFunction::TanH;
    armnn::IConnectableLayer* const tanhLayer = net.AddActivationLayer(tanDesc, "TanH layer");
    BOOST_TEST(tanhLayer);

    //Layer 2
    armnn::SoftmaxDescriptor softMaxDesc{};
    armnn::IConnectableLayer* const softmaxLayer = net.AddSoftmaxLayer(softMaxDesc, "Softmax");
    BOOST_TEST(softmaxLayer);

    //Layer 3
    armnn::ElementwiseUnaryDescriptor rsqrtDesc;
    rsqrtDesc.m_Operation                = UnaryOperation::Rsqrt;
    armnn::IConnectableLayer* rsqrtLayer = net.AddElementwiseUnaryLayer(rsqrtDesc, "Rsqrt");
    BOOST_TEST(rsqrtLayer);

    //Layer 4
    unsigned int dims[] = { 10, 1, 1, 1 };
    std::vector<float> convWeightsData(10);
    armnn::ConstTensor weights(armnn::TensorInfo(4, dims, armnn::DataType::Float32), convWeightsData);
    armnn::FullyConnectedDescriptor fullyConnectedDesc;
    armnn::IConnectableLayer* const fullyConnectedLayer =
        net.AddFullyConnectedLayer(fullyConnectedDesc, weights, armnn::EmptyOptional(), "fully connected");
    BOOST_TEST(fullyConnectedLayer);

    armnn::IConnectableLayer* const outputLayer = net.AddOutputLayer(0, "output layer");
    BOOST_TEST(outputLayer);

    TensorInfo inputTensorInfo(TensorShape({ 1, 16, 16, 16 }), armnn::DataType::QAsymmU8);
    inputTensorInfo.SetQuantizationOffset(0);
    inputTensorInfo.SetQuantizationScale(0.9f);

    TensorInfo outputTensorInfo(TensorShape({ 1, 16, 16, 16 }), armnn::DataType::QAsymmU8);
    outputTensorInfo.SetQuantizationOffset(0);
    outputTensorInfo.SetQuantizationScale(0.9f);

    inputLayer->GetOutputSlot(0).Connect(tanhLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    tanhLayer->GetOutputSlot(0).Connect(softmaxLayer->GetInputSlot(0));
    tanhLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    softmaxLayer->GetOutputSlot(0).Connect(rsqrtLayer->GetInputSlot(0));
    softmaxLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    rsqrtLayer->GetOutputSlot(0).Connect(fullyConnectedLayer->GetInputSlot(0));
    rsqrtLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    fullyConnectedLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    fullyConnectedLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    return iNetPtr;
}

INetworkPtr SISOCatOneGraphFactory::GetExpectedModifiedGraph() const
{
    INetworkPtr iNetPtr(INetwork::Create());
    INetwork& net = *iNetPtr;

    armnn::IConnectableLayer* const inputLayer = net.AddInputLayer(0, "input layer");
    BOOST_TEST(inputLayer);

    //Layer 1
    ActivationDescriptor tanDesc{};
    tanDesc.m_A                                  = 100;
    tanDesc.m_B                                  = 0;
    tanDesc.m_Function                           = ActivationFunction::Sigmoid;
    armnn::IConnectableLayer* const sigmoidLayer = net.AddActivationLayer(tanDesc, "Sigmoid");
    BOOST_TEST(sigmoidLayer);

    //Layer 2
    armnn::Pooling2dDescriptor pooling2dDesc{};
    pooling2dDesc.m_PadBottom                      = 1;
    pooling2dDesc.m_PadLeft                        = 1;
    pooling2dDesc.m_PadRight                       = 1;
    pooling2dDesc.m_PadTop                         = 1;
    pooling2dDesc.m_StrideX                        = 1;
    pooling2dDesc.m_StrideY                        = 1;
    pooling2dDesc.m_PoolHeight                     = 3;
    pooling2dDesc.m_PoolWidth                      = 3;
    pooling2dDesc.m_PoolType                       = PoolingAlgorithm::Average;
    pooling2dDesc.m_DataLayout                     = DataLayout::NHWC;
    armnn::IConnectableLayer* const pooling2dLayer = net.AddPooling2dLayer(pooling2dDesc, "Pooling2d");
    BOOST_TEST(pooling2dLayer);

    //Layer 3
    armnn::Convolution2dDescriptor convolution2dDesc;
    convolution2dDesc.m_DilationX   = 1;
    convolution2dDesc.m_DilationY   = 1;
    convolution2dDesc.m_PadBottom   = 0;
    convolution2dDesc.m_PadLeft     = 0;
    convolution2dDesc.m_PadRight    = 0;
    convolution2dDesc.m_PadTop      = 0;
    convolution2dDesc.m_StrideX     = 1;
    convolution2dDesc.m_StrideY     = 1;
    convolution2dDesc.m_BiasEnabled = true;
    convolution2dDesc.m_DataLayout  = DataLayout::NHWC;

    std::vector<uint8_t> weightDataConv2d(16 * 1 * 1 * 16);
    std::vector<unsigned int> weightDimensionsConv2d = { 16, 1, 1, 16 };
    ConstTensor weightsConv2d(TensorInfo(4, weightDimensionsConv2d.data(), DataType::QAsymmU8, 0.5), weightDataConv2d);

    std::vector<unsigned int> biasesDataConv2d(1 * 1 * 1 * 16);
    std::vector<unsigned int> biasDimensionsConv2d = { 1, 1, 1, 16 };
    ConstTensor biasesConv2d(TensorInfo(4, biasDimensionsConv2d.data(), DataType::Signed32, 0.899999976f),
                             biasesDataConv2d);

    armnn::IConnectableLayer* convolution2dLayer = net.AddConvolution2dLayer(
        convolution2dDesc, weightsConv2d, Optional<ConstTensor>(biasesConv2d), "Convolution2d");
    BOOST_TEST(convolution2dLayer);

    //Layer 4
    armnn::DepthwiseConvolution2dDescriptor depthwiseConvolution2dDesc;
    depthwiseConvolution2dDesc.m_DilationX   = 1;
    depthwiseConvolution2dDesc.m_DilationY   = 1;
    depthwiseConvolution2dDesc.m_PadBottom   = 0;
    depthwiseConvolution2dDesc.m_PadLeft     = 0;
    depthwiseConvolution2dDesc.m_PadRight    = 0;
    depthwiseConvolution2dDesc.m_PadTop      = 0;
    depthwiseConvolution2dDesc.m_StrideX     = 1;
    depthwiseConvolution2dDesc.m_StrideY     = 1;
    depthwiseConvolution2dDesc.m_BiasEnabled = true;
    depthwiseConvolution2dDesc.m_DataLayout  = DataLayout::NHWC;

    std::vector<uint8_t> weightDataDepthConv2d(1 * 16 * 1 * 1);
    std::vector<unsigned int> weightDimensionsDepthConv2d = { 1, 16, 1, 1 };
    ConstTensor weightsDepthConv2d(TensorInfo(4, weightDimensionsDepthConv2d.data(), DataType::QAsymmU8, 0.5),
                                   weightDataDepthConv2d);

    std::vector<unsigned int> biasesDataDepthConv2d(1 * 1 * 1 * 16);
    std::vector<unsigned int> biasDimensionsDepthConv2d = { 1, 1, 1, 16 };
    ConstTensor biasesDepthConv2d(TensorInfo(4, biasDimensionsDepthConv2d.data(), DataType::Signed32, 0.899999976f),
                                  biasesDataDepthConv2d);

    armnn::IConnectableLayer* depthwiseConvolution2dLayer =
        net.AddDepthwiseConvolution2dLayer(depthwiseConvolution2dDesc, weightsDepthConv2d,
                                           Optional<ConstTensor>(biasesDepthConv2d), "DepthwiseConvolution2d");
    BOOST_TEST(depthwiseConvolution2dLayer);

    armnn::IConnectableLayer* const outputLayer = net.AddOutputLayer(0, "output layer");
    BOOST_TEST(outputLayer);

    TensorInfo inputTensorInfo(TensorShape({ 1, 16, 16, 16 }), armnn::DataType::QAsymmU8);
    inputTensorInfo.SetQuantizationOffset(0);
    inputTensorInfo.SetQuantizationScale(0.9f);

    TensorInfo outputTensorInfo(TensorShape({ 1, 16, 16, 16 }), armnn::DataType::QAsymmU8);
    outputTensorInfo.SetQuantizationOffset(0);
    outputTensorInfo.SetQuantizationScale(0.9f);

    inputLayer->GetOutputSlot(0).Connect(sigmoidLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    sigmoidLayer->GetOutputSlot(0).Connect(pooling2dLayer->GetInputSlot(0));
    sigmoidLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    pooling2dLayer->GetOutputSlot(0).Connect(convolution2dLayer->GetInputSlot(0));
    pooling2dLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    convolution2dLayer->GetOutputSlot(0).Connect(depthwiseConvolution2dLayer->GetInputSlot(0));
    convolution2dLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    depthwiseConvolution2dLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    depthwiseConvolution2dLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    return iNetPtr;
}

std::string SISOCatOneGraphFactory::GetMappingFileName() const
{
    return "SISOCatOneMapping.txt";
}
