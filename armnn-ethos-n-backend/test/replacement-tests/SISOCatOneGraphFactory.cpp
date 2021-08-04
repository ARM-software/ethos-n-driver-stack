//
// Copyright © 2020-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#include "SISOCatOneGraphFactory.hpp"

#include "EthosNConfig.hpp"
#include "EthosNLayerSupport.hpp"
#include "EthosNMapping.hpp"
#include "EthosNTestUtils.hpp"
#include "Network.hpp"

#include <doctest/doctest.h>

using namespace armnn;

const std::string& SISOCatOneGraphFactory::GetName() const
{
    static const std::string name("SISOCatOneGraphFactory");
    return name;
}

//input-->Activation(TanH)-->Softmax-->Rsqrt-->FullyCoonected-->output
std::unique_ptr<NetworkImpl> SISOCatOneGraphFactory::GetInitialGraph() const
{
    std::unique_ptr<NetworkImpl> net = std::make_unique<NetworkImpl>();

    armnn::IConnectableLayer* const inputLayer = net->AddInputLayer(0, "input layer");
    CHECK(inputLayer);

    //Layer 1
    ActivationDescriptor tanDesc{};
    tanDesc.m_A                               = 100;
    tanDesc.m_B                               = 0;
    tanDesc.m_Function                        = ActivationFunction::TanH;
    armnn::IConnectableLayer* const tanhLayer = net->AddActivationLayer(tanDesc, "TanH layer");
    CHECK(tanhLayer);

    //Layer 2
    armnn::SoftmaxDescriptor softMaxDesc{};
    armnn::IConnectableLayer* const softmaxLayer = net->AddSoftmaxLayer(softMaxDesc, "Softmax");
    CHECK(softmaxLayer);

    //Layer 3
    armnn::ElementwiseUnaryDescriptor rsqrtDesc;
    rsqrtDesc.m_Operation                = UnaryOperation::Rsqrt;
    armnn::IConnectableLayer* rsqrtLayer = net->AddElementwiseUnaryLayer(rsqrtDesc, "Rsqrt");
    CHECK(rsqrtLayer);

    armnn::IConnectableLayer* const outputLayer = net->AddOutputLayer(0, "output layer");
    CHECK(outputLayer);

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

    rsqrtLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    rsqrtLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    return net;
}

std::unique_ptr<NetworkImpl> SISOCatOneGraphFactory::GetExpectedModifiedGraph() const
{
    std::unique_ptr<NetworkImpl> net           = std::make_unique<NetworkImpl>();
    armnn::IConnectableLayer* const inputLayer = net->AddInputLayer(0, "input layer");
    CHECK(inputLayer);

    //Layer 1
    ActivationDescriptor tanDesc{};
    tanDesc.m_A                                  = 100;
    tanDesc.m_B                                  = 0;
    tanDesc.m_Function                           = ActivationFunction::Sigmoid;
    armnn::IConnectableLayer* const sigmoidLayer = net->AddActivationLayer(tanDesc, "Sigmoid");
    CHECK(sigmoidLayer);

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
    armnn::IConnectableLayer* const pooling2dLayer = net->AddPooling2dLayer(pooling2dDesc, "Pooling2d");
    CHECK(pooling2dLayer);

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

    armnn::IConnectableLayer* convolution2dLayer = net->AddConvolution2dLayer(
        convolution2dDesc, weightsConv2d, Optional<ConstTensor>(biasesConv2d), "Convolution2d");
    CHECK(convolution2dLayer);

    armnn::IConnectableLayer* const outputLayer = net->AddOutputLayer(0, "output layer");
    CHECK(outputLayer);

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

    convolution2dLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    convolution2dLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    return net;
}

std::string SISOCatOneGraphFactory::GetMappingFileName() const
{
    return "SISOCatOneMapping.txt";
}
