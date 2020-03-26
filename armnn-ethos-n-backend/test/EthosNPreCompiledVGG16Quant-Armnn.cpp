//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#define BOOST_TEST_MODULE EthosNPreCompiledVGG16Quant_Armnn

#include <EthosNBackendId.hpp>
#include <armnn/ArmNN.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/test/included/unit_test.hpp>
#include <reference/RefBackendId.hpp>

#include <cstdlib>
#include <vector>

using namespace armnn;

// Generate weights data with a selection of values
std::vector<uint8_t> GenerateWeightsData(const TensorInfo& info)
{
    std::vector<uint8_t> data;
    const TensorShape shape = info.GetShape();
    data.resize(shape.GetNumElements());

    // Only OHWI layout is supported
    const unsigned int outChannels = shape[0];
    const unsigned int height      = shape[1];
    const unsigned int width       = shape[2];
    const unsigned int inChannels  = shape[3];
    const uint8_t minValue         = height == 1 && width == 1 ? 1 : 0;
    const uint8_t maxValue         = 4;
    for (unsigned int ic = 0; ic < inChannels; ++ic)
    {
        for (unsigned int oc = 0; oc < outChannels; ++oc)
        {
            for (unsigned int h = 0; h < height; ++h)
            {
                for (unsigned int w = 0; w < width; ++w)
                {
                    auto index  = (oc * height * width * inChannels) + (h * width * inChannels) + (w * inChannels) + ic;
                    data[index] = boost::numeric_cast<uint8_t>(rand() % (maxValue - minValue) + minValue);
                }
            }
        }
    }
    return data;
}

void RunNetwork(const INetwork& net,
                const std::vector<BackendId>& backends,
                const std::vector<uint8_t>& inputData,
                std::vector<uint8_t>& outputData)
{
    // Optimise the network
    IRuntimePtr runtime(IRuntime::Create(IRuntime::CreationOptions()));
    IOptimizedNetworkPtr optimizedNet = Optimize(net, backends, runtime->GetDeviceSpec(), OptimizerOptions());
    if (!optimizedNet)
    {
        throw RuntimeException(std::string("Failed to optimize network for ") + std::string(backends[0]),
                               CHECK_LOCATION());
    }

    // Load the optimised network into the runtime
    NetworkId networkIdentifier;
    Status loadStatus = runtime->LoadNetwork(networkIdentifier, std::move(optimizedNet));
    BOOST_TEST(loadStatus == Status::Success, "Failed to load network");

    // Create the input and output Tensors
    InputTensors inputTensors{ { 0,
                                 ConstTensor(runtime->GetInputTensorInfo(networkIdentifier, 0), inputData.data()) } };
    OutputTensors outputTensors{ { 0, Tensor(runtime->GetOutputTensorInfo(networkIdentifier, 0), outputData.data()) } };

    // Execute network
    runtime->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);
}

void TestVGG16Network()
{
    INetworkPtr net = INetwork::Create();

    constexpr int weightsHeight = 3;
    constexpr int weightsWidth  = 3;

    // Convolution descriptor (for all convolution layers)
    Convolution2dDescriptor convDescriptor;
    convDescriptor.m_StrideX     = 1;
    convDescriptor.m_StrideY     = 1;
    convDescriptor.m_PadLeft     = 1;
    convDescriptor.m_PadRight    = 1;
    convDescriptor.m_PadTop      = 1;
    convDescriptor.m_PadBottom   = 1;
    convDescriptor.m_BiasEnabled = true;
    convDescriptor.m_DataLayout  = DataLayout::NHWC;

    // Pooling descriptor (For all pooling layers)
    Pooling2dDescriptor poolDescriptor;
    poolDescriptor.m_PoolType      = PoolingAlgorithm::Max;
    poolDescriptor.m_PoolWidth     = 2;
    poolDescriptor.m_PoolHeight    = 2;
    poolDescriptor.m_StrideX       = 2;
    poolDescriptor.m_StrideY       = 2;
    poolDescriptor.m_PaddingMethod = PaddingMethod::Exclude;
    poolDescriptor.m_DataLayout    = DataLayout::NHWC;

    // ======== Input Layer ========
    // Set up the input - NHWC
    TensorInfo inputInfo(TensorShape({ 1, 224, 224, 3 }), DataType::QAsymmU8, 1.0f, 0);
    const unsigned int inputDataSize = inputInfo.GetNumElements();
    std::vector<uint8_t> inputData(inputDataSize);
    for (unsigned int i = 0; i < inputDataSize; ++i)
    {
        inputData[i] = boost::numeric_cast<uint8_t>(i % 253);
    }

    IConnectableLayer* const inputLayer = net->AddInputLayer(0, "input");
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    // ======== Layer 01 ========
    // Convolution weights - OHWI
    TensorInfo weightsInfo01(TensorShape({ 64, weightsHeight, weightsWidth, 3 }), DataType::QAsymmU8, 2.0f, 0);
    std::vector<uint8_t> weightsData01 = GenerateWeightsData(weightsInfo01);
    ConstTensor weights01(weightsInfo01, weightsData01);
    // Convolution bias
    TensorInfo biasInfo01(TensorShape({ 64 }), DataType::Signed32, 1.0f * 2.0f, 0);
    std::vector<int32_t> biasData01(biasInfo01.GetNumElements(), 0);
    ConstTensor bias01(biasInfo01, biasData01);
    // Add the layer and set the output tensor info
    IConnectableLayer* const conv01Layer =
        net->AddConvolution2dLayer(convDescriptor, weights01, Optional<ConstTensor>(bias01), "conv01");
    TensorInfo convInfo01(TensorShape({ 1, 224, 224, 64 }), DataType::QAsymmU8, 4.0f, 0);
    conv01Layer->GetOutputSlot(0).SetTensorInfo(convInfo01);

    // ======== Layer 02 ========
    // Convolution weights - OHWI
    TensorInfo weightsInfo02(TensorShape({ 64, weightsHeight, weightsWidth, 64 }), DataType::QAsymmU8, 2.0f, 0);
    std::vector<uint8_t> weightsData02 = GenerateWeightsData(weightsInfo02);
    ConstTensor weights02(weightsInfo02, weightsData02);
    // Convolution bias
    TensorInfo biasInfo02(TensorShape({ 64 }), DataType::Signed32, 4.0f * 2.0f, 0);
    std::vector<int32_t> biasData02(biasInfo02.GetNumElements(), 0);
    ConstTensor bias02(biasInfo02, biasData02);
    // Add the layer and set the output tensor info
    IConnectableLayer* const conv02Layer =
        net->AddConvolution2dLayer(convDescriptor, weights02, Optional<ConstTensor>(bias02), "conv02");
    TensorInfo convInfo02(TensorShape({ 1, 224, 224, 64 }), DataType::QAsymmU8, 16.0f, 0);
    conv02Layer->GetOutputSlot(0).SetTensorInfo(convInfo02);

    // ======== Layer 03 ========
    // Add the layer and set the output tensor info
    IConnectableLayer* const pool03Layer = net->AddPooling2dLayer(poolDescriptor, "pool03");
    TensorInfo poolInfo03(TensorShape({ 1, 112, 112, 64 }), DataType::QAsymmU8, 16.0f, 0);
    pool03Layer->GetOutputSlot(0).SetTensorInfo(poolInfo03);

    // ======== Layer 04 ========
    // Convolution weights - OHWI
    TensorInfo weightsInfo04(TensorShape({ 128, weightsHeight, weightsWidth, 64 }), DataType::QAsymmU8, 2.0f, 0);
    std::vector<uint8_t> weightsData04 = GenerateWeightsData(weightsInfo04);
    ConstTensor weights04(weightsInfo04, weightsData04);
    // Convolution bias
    TensorInfo biasInfo04(TensorShape({ 128 }), DataType::Signed32, 16.0f * 2.0f, 0);
    std::vector<int32_t> biasData04(biasInfo04.GetNumElements(), 0);
    ConstTensor bias04(biasInfo04, biasData04);
    // Add the layer and set the output tensor info
    IConnectableLayer* const conv04Layer =
        net->AddConvolution2dLayer(convDescriptor, weights04, Optional<ConstTensor>(bias04), "conv04");
    TensorInfo convInfo04(TensorShape({ 1, 112, 112, 128 }), DataType::QAsymmU8, 64.0f, 0);
    conv04Layer->GetOutputSlot(0).SetTensorInfo(convInfo04);

    // ======== Layer 05 ========
    // Convolution weights - OHWI
    TensorInfo weightsInfo05(TensorShape({ 128, weightsHeight, weightsWidth, 128 }), DataType::QAsymmU8, 2.0f, 0);
    std::vector<uint8_t> weightsData05 = GenerateWeightsData(weightsInfo05);
    ConstTensor weights05(weightsInfo05, weightsData05);
    // Convolution bias
    TensorInfo biasInfo05(TensorShape({ 128 }), DataType::Signed32, 64.0f * 2.0f, 0);
    std::vector<int32_t> biasData05(biasInfo05.GetNumElements(), 0);
    ConstTensor bias05(biasInfo05, biasData05);
    // Add the layer and set the output tensor info
    IConnectableLayer* const conv05Layer =
        net->AddConvolution2dLayer(convDescriptor, weights05, Optional<ConstTensor>(bias05), "conv05");
    TensorInfo convInfo05(TensorShape({ 1, 112, 112, 128 }), DataType::QAsymmU8, 256.0f, 0);
    conv05Layer->GetOutputSlot(0).SetTensorInfo(convInfo05);

    // ======== Layer 06 ========
    // Add the layer and set the output tensor info
    IConnectableLayer* const pool06Layer = net->AddPooling2dLayer(poolDescriptor, "pool06");
    TensorInfo poolInfo06(TensorShape({ 1, 56, 56, 128 }), DataType::QAsymmU8, 256.0f, 0);
    pool06Layer->GetOutputSlot(0).SetTensorInfo(poolInfo06);

    // ======== Layer 07 ========
    // Convolution weights - OHWI
    TensorInfo weightsInfo07(TensorShape({ 256, weightsHeight, weightsWidth, 128 }), DataType::QAsymmU8, 2.0f, 0);
    std::vector<uint8_t> weightsData07 = GenerateWeightsData(weightsInfo07);
    ConstTensor weights07(weightsInfo07, weightsData07);
    // Convolution bias
    TensorInfo biasInfo07(TensorShape({ 256 }), DataType::Signed32, 256.0f * 2.0f, 0);
    std::vector<int32_t> biasData07(biasInfo07.GetNumElements(), 0);
    ConstTensor bias07(biasInfo07, biasData07);
    // Add the layer and set the output tensor info
    IConnectableLayer* const conv07Layer =
        net->AddConvolution2dLayer(convDescriptor, weights07, Optional<ConstTensor>(bias07), "conv07");
    TensorInfo convInfo07(TensorShape({ 1, 56, 56, 256 }), DataType::QAsymmU8, 1024.0f, 0);
    conv07Layer->GetOutputSlot(0).SetTensorInfo(convInfo07);

    // ======== Layer 08 ========
    // Convolution weights - OHWI
    TensorInfo weightsInfo08(TensorShape({ 256, weightsHeight, weightsWidth, 256 }), DataType::QAsymmU8, 2.0f, 0);
    std::vector<uint8_t> weightsData08 = GenerateWeightsData(weightsInfo08);
    ConstTensor weights08(weightsInfo08, weightsData08);
    // Convolution bias
    TensorInfo biasInfo08(TensorShape({ 256 }), DataType::Signed32, 1024.0f * 2.0f, 0);
    std::vector<int32_t> biasData08(biasInfo08.GetNumElements(), 0);
    ConstTensor bias08(biasInfo08, biasData08);
    // Add the layer and set the output tensor info
    IConnectableLayer* const conv08Layer =
        net->AddConvolution2dLayer(convDescriptor, weights08, Optional<ConstTensor>(bias08), "conv08");
    TensorInfo convInfo08(TensorShape({ 1, 56, 56, 256 }), DataType::QAsymmU8, 4096.0f, 0);
    conv08Layer->GetOutputSlot(0).SetTensorInfo(convInfo08);

    // ======== Layer 09 ========
    // Convolution weights - OHWI
    TensorInfo weightsInfo09(TensorShape({ 256, weightsHeight, weightsWidth, 256 }), DataType::QAsymmU8, 2.0f, 0);
    std::vector<uint8_t> weightsData09 = GenerateWeightsData(weightsInfo09);
    ConstTensor weights09(weightsInfo09, weightsData09);
    // Convolution bias
    TensorInfo biasInfo09(TensorShape({ 256 }), DataType::Signed32, 4096.0f * 2.0f, 0);
    std::vector<int32_t> biasData09(biasInfo09.GetNumElements(), 0);
    ConstTensor bias09(biasInfo09, biasData09);
    // Add the layer and set the output tensor info
    IConnectableLayer* const conv09Layer =
        net->AddConvolution2dLayer(convDescriptor, weights09, Optional<ConstTensor>(bias09), "conv09");
    TensorInfo convInfo09(TensorShape({ 1, 56, 56, 256 }), DataType::QAsymmU8, 16384.0f, 0);
    conv09Layer->GetOutputSlot(0).SetTensorInfo(convInfo09);

    // ======== Layer 10 ========
    // Add the layer and set the output tensor info
    IConnectableLayer* const pool10Layer = net->AddPooling2dLayer(poolDescriptor, "pool10");
    TensorInfo poolInfo10(TensorShape({ 1, 28, 28, 256 }), DataType::QAsymmU8, 16384.0f, 0);
    pool10Layer->GetOutputSlot(0).SetTensorInfo(poolInfo10);

    // ======== Layer 11 ========
    // Convolution weights - OHWI
    TensorInfo weightsInfo11(TensorShape({ 512, weightsHeight, weightsWidth, 256 }), DataType::QAsymmU8, 2.0f, 0);
    std::vector<uint8_t> weightsData11 = GenerateWeightsData(weightsInfo11);
    ConstTensor weights11(weightsInfo11, weightsData11);
    // Convolution bias
    TensorInfo biasInfo11(TensorShape({ 512 }), DataType::Signed32, 16384.0f * 2.0f, 0);
    std::vector<int32_t> biasData11(biasInfo11.GetNumElements(), 0);
    ConstTensor bias11(biasInfo11, biasData11);
    // Add the layer and set the output tensor info
    IConnectableLayer* const conv11Layer =
        net->AddConvolution2dLayer(convDescriptor, weights11, Optional<ConstTensor>(bias11), "conv11");
    TensorInfo convInfo11(TensorShape({ 1, 28, 28, 512 }), DataType::QAsymmU8, 65536.0f, 0);
    conv11Layer->GetOutputSlot(0).SetTensorInfo(convInfo11);

    // ======== Layer 12 ========
    // Convolution weights - OHWI
    TensorInfo weightsInfo12(TensorShape({ 512, weightsHeight, weightsWidth, 512 }), DataType::QAsymmU8, 2.0f, 0);
    std::vector<uint8_t> weightsData12 = GenerateWeightsData(weightsInfo12);
    ConstTensor weights12(weightsInfo12, weightsData12);
    // Convolution bias
    TensorInfo biasInfo12(TensorShape({ 512 }), DataType::Signed32, 65536.0f * 2.0f, 0);
    std::vector<int32_t> biasData12(biasInfo12.GetNumElements(), 0);
    ConstTensor bias12(biasInfo12, biasData12);
    // Add the layer and set the output tensor info
    IConnectableLayer* const conv12Layer =
        net->AddConvolution2dLayer(convDescriptor, weights12, Optional<ConstTensor>(bias12), "conv12");
    TensorInfo convInfo12(TensorShape({ 1, 28, 28, 512 }), DataType::QAsymmU8, 262144.0f, 0);
    conv12Layer->GetOutputSlot(0).SetTensorInfo(convInfo12);

    // ======== Layer 13 ========
    // Convolution weights - OHWI
    TensorInfo weightsInfo13(TensorShape({ 512, weightsHeight, weightsWidth, 512 }), DataType::QAsymmU8, 2.0f, 0);
    std::vector<uint8_t> weightsData13 = GenerateWeightsData(weightsInfo13);
    ConstTensor weights13(weightsInfo13, weightsData13);
    // Convolution bias
    TensorInfo biasInfo13(TensorShape({ 512 }), DataType::Signed32, 262144.0f * 2.0f, 0);
    std::vector<int32_t> biasData13(biasInfo13.GetNumElements(), 0);
    ConstTensor bias13(biasInfo13, biasData13);
    // Add the layer and set the output tensor info
    IConnectableLayer* const conv13Layer =
        net->AddConvolution2dLayer(convDescriptor, weights13, Optional<ConstTensor>(bias13), "conv13");
    TensorInfo convInfo13(TensorShape({ 1, 28, 28, 512 }), DataType::QAsymmU8, 1048576.0f, 0);
    conv13Layer->GetOutputSlot(0).SetTensorInfo(convInfo13);

    // ======== Layer 14 ========
    // Add the layer and set the output tensor info
    IConnectableLayer* const pool14Layer = net->AddPooling2dLayer(poolDescriptor, "pool14");
    TensorInfo poolInfo14(TensorShape({ 1, 14, 14, 512 }), DataType::QAsymmU8, 1048576.0f, 0);
    pool14Layer->GetOutputSlot(0).SetTensorInfo(poolInfo14);

    // ======== Layer 15 ========
    // Convolution weights - OHWI
    TensorInfo weightsInfo15(TensorShape({ 512, weightsHeight, weightsWidth, 512 }), DataType::QAsymmU8, 2.0f, 0);
    std::vector<uint8_t> weightsData15 = GenerateWeightsData(weightsInfo15);
    ConstTensor weights15(weightsInfo15, weightsData15);
    // Convolution bias
    TensorInfo biasInfo15(TensorShape({ 512 }), DataType::Signed32, 1048576.0f * 2.0f, 0);
    std::vector<int32_t> biasData15(biasInfo15.GetNumElements(), 0);
    ConstTensor bias15(biasInfo15, biasData15);
    // Add the layer and set the output tensor info
    IConnectableLayer* const conv15Layer =
        net->AddConvolution2dLayer(convDescriptor, weights15, Optional<ConstTensor>(bias15), "conv15");
    TensorInfo convInfo15(TensorShape({ 1, 14, 14, 512 }), DataType::QAsymmU8, 4194304.0f, 0);
    conv15Layer->GetOutputSlot(0).SetTensorInfo(convInfo15);

    // ======== Layer 16 ========
    // Convolution weights - OHWI
    TensorInfo weightsInfo16(TensorShape({ 512, weightsHeight, weightsWidth, 512 }), DataType::QAsymmU8, 2.0f, 0);
    std::vector<uint8_t> weightsData16 = GenerateWeightsData(weightsInfo16);
    ConstTensor weights16(weightsInfo16, weightsData16);
    // Convolution bias
    TensorInfo biasInfo16(TensorShape({ 512 }), DataType::Signed32, 4194304.0f * 2.0f, 0);
    std::vector<int32_t> biasData16(biasInfo16.GetNumElements(), 0);
    ConstTensor bias16(biasInfo16, biasData16);
    // Add the layer and set the output tensor info
    IConnectableLayer* const conv16Layer =
        net->AddConvolution2dLayer(convDescriptor, weights16, Optional<ConstTensor>(bias16), "conv16");
    TensorInfo convInfo16(TensorShape({ 1, 14, 14, 512 }), DataType::QAsymmU8, 16777216.0f, 0);
    conv16Layer->GetOutputSlot(0).SetTensorInfo(convInfo16);

    // ======== Layer 17 ========
    // Convolution weights - OHWI
    TensorInfo weightsInfo17(TensorShape({ 512, weightsHeight, weightsWidth, 512 }), DataType::QAsymmU8, 2.0f, 0);
    std::vector<uint8_t> weightsData17 = GenerateWeightsData(weightsInfo17);
    ConstTensor weights17(weightsInfo17, weightsData17);
    // Convolution bias
    TensorInfo biasInfo17(TensorShape({ 512 }), DataType::Signed32, 16777216.0f * 2.0f, 0);
    std::vector<int32_t> biasData17(biasInfo17.GetNumElements(), 0);
    ConstTensor bias17(biasInfo17, biasData17);
    // Add the layer and set the output tensor info
    IConnectableLayer* const conv17Layer =
        net->AddConvolution2dLayer(convDescriptor, weights17, Optional<ConstTensor>(bias17), "conv17");
    TensorInfo convInfo17(TensorShape({ 1, 14, 14, 512 }), DataType::QAsymmU8, 67108864.0f, 0);
    conv17Layer->GetOutputSlot(0).SetTensorInfo(convInfo17);

    // ======== Layer 18 ========
    // Add the layer and set the output tensor info
    IConnectableLayer* const pool18Layer = net->AddPooling2dLayer(poolDescriptor, "pool18");
    TensorInfo poolInfo18(TensorShape({ 1, 7, 7, 512 }), DataType::QAsymmU8, 67108864.0f, 0);
    pool18Layer->GetOutputSlot(0).SetTensorInfo(poolInfo18);

    // ======== Output Layer ========
    IConnectableLayer* const outputLayer = net->AddOutputLayer(0, "output");

    // Reserve space for the expected outputs
    TensorInfo outputInfo(poolInfo18);
    std::vector<uint8_t> refOutputData(outputInfo.GetNumElements());
    std::vector<uint8_t> ethosnOutputData(outputInfo.GetNumElements());

    // Connect the layers
    inputLayer->GetOutputSlot(0).Connect(conv01Layer->GetInputSlot(0));
    conv01Layer->GetOutputSlot(0).Connect(conv02Layer->GetInputSlot(0));
    conv02Layer->GetOutputSlot(0).Connect(pool03Layer->GetInputSlot(0));
    pool03Layer->GetOutputSlot(0).Connect(conv04Layer->GetInputSlot(0));
    conv04Layer->GetOutputSlot(0).Connect(conv05Layer->GetInputSlot(0));
    conv05Layer->GetOutputSlot(0).Connect(pool06Layer->GetInputSlot(0));
    pool06Layer->GetOutputSlot(0).Connect(conv07Layer->GetInputSlot(0));
    conv07Layer->GetOutputSlot(0).Connect(conv08Layer->GetInputSlot(0));
    conv08Layer->GetOutputSlot(0).Connect(conv09Layer->GetInputSlot(0));
    conv09Layer->GetOutputSlot(0).Connect(pool10Layer->GetInputSlot(0));
    pool10Layer->GetOutputSlot(0).Connect(conv11Layer->GetInputSlot(0));
    conv11Layer->GetOutputSlot(0).Connect(conv12Layer->GetInputSlot(0));
    conv12Layer->GetOutputSlot(0).Connect(conv13Layer->GetInputSlot(0));
    conv13Layer->GetOutputSlot(0).Connect(pool14Layer->GetInputSlot(0));
    pool14Layer->GetOutputSlot(0).Connect(conv15Layer->GetInputSlot(0));
    conv15Layer->GetOutputSlot(0).Connect(conv16Layer->GetInputSlot(0));
    conv16Layer->GetOutputSlot(0).Connect(conv17Layer->GetInputSlot(0));
    conv17Layer->GetOutputSlot(0).Connect(pool18Layer->GetInputSlot(0));
    pool18Layer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    RunNetwork(*net, { EthosNBackendId() }, inputData, ethosnOutputData);
    RunNetwork(*net, { RefBackendId() }, inputData, refOutputData);

    BOOST_CHECK_EQUAL_COLLECTIONS(ethosnOutputData.begin(), ethosnOutputData.end(), refOutputData.begin(),
                                  refOutputData.end());
}

BOOST_AUTO_TEST_SUITE(EthosNPreCompiledVGG16)

BOOST_AUTO_TEST_CASE(EthosNPreCompiledVGG16Quant)
{
    TestVGG16Network();
}

BOOST_AUTO_TEST_SUITE_END()
