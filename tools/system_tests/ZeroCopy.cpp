//
// Copyright © 2022-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "SystemTestsUtils.hpp"

#include "ArmnnUtils.hpp"
#include "GlobalParameters.hpp"
#include <armnn/ArmNN.hpp>
#include <catch.hpp>
#include <ethosn_driver_library/Network.hpp>
#include <ethosn_driver_library/ProcMemAllocator.hpp>
#include <ethosn_support_library/Support.hpp>
#include <ethosn_utils/KernelUtils.hpp>
#include <ethosn_utils/VectorStream.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string.h>

namespace ethosn
{
namespace system_tests
{

struct ConvParams
{
    uint32_t m_NumIfm;
    uint32_t m_NumOfm;
    uint32_t m_IfmWidth;
    uint32_t m_IfmHeight;
    uint32_t m_KernelWidth;
    uint32_t m_KernelHeight;
    uint32_t m_PadLeft;
    uint32_t m_PadRight;
    uint32_t m_PadBottom;
    uint32_t m_PadTop;
    ethosn::support_library::DataFormat m_Format;
    uint32_t m_StrideX;
    uint32_t m_StrideY;
    bool m_Debug;
};

static InferenceOutputs CreateMultipleInferenceRef(ConvParams params,
                                                   const uint8_t* inputData1,
                                                   const uint8_t* inputData2,
                                                   const BaseTensor& weightsData,
                                                   const BaseTensor& biasData)
{
    using namespace armnn;
    // Construct Arm NN network
    INetworkPtr network = INetwork::Create();

    TensorInfo weightInfo(
        TensorShape({ params.m_NumOfm, params.m_KernelHeight, params.m_KernelWidth, params.m_NumIfm }),
        armnn::DataType::QAsymmU8, 1.0f, 0, true);

    ConstTensor weights(weightInfo, weightsData.GetByteData());

    uint32_t biasDims[1] = { params.m_NumOfm };
    TensorInfo biasInfo(TensorShape(1, biasDims), armnn::DataType::Signed32, 1.0f / 256.0f, 0, true);
    ConstTensor bias(biasInfo, biasData.GetByteData());

    Convolution2dDescriptor convDesc;
    convDesc.m_BiasEnabled  = true;
    convDesc.m_DataLayout   = DataLayout::NHWC;
    convDesc.m_PadLeft      = params.m_PadLeft;
    convDesc.m_PadRight     = params.m_PadRight;
    convDesc.m_PadTop       = params.m_PadTop;
    convDesc.m_PadBottom    = params.m_PadBottom;
    convDesc.m_StrideX      = params.m_StrideX;
    convDesc.m_StrideY      = params.m_StrideY;
    IConnectableLayer* conv = network->AddConvolution2dLayer(convDesc, "conv");

    armnn::IConnectableLayer* weightsLayer = network->AddConstantLayer(weights, "Conv2dWeights");
    weightsLayer->GetOutputSlot(0).SetTensorInfo(weightInfo);
    weightsLayer->GetOutputSlot(0).Connect((*conv).GetInputSlot(1));
    armnn::IConnectableLayer* biasLayer = network->AddConstantLayer(bias, "Conv2dBias");
    biasLayer->GetOutputSlot(0).SetTensorInfo(biasInfo);
    biasLayer->GetOutputSlot(0).Connect((*conv).GetInputSlot(2));

    IConnectableLayer* InputLayer  = network->AddInputLayer(0);
    IConnectableLayer* OutputLayer = network->AddOutputLayer(0);

    InputLayer->GetOutputSlot(0).Connect(conv->GetInputSlot(0));
    conv->GetOutputSlot(0).Connect(OutputLayer->GetInputSlot(0));

    // Create Arm NN runtime
    IRuntime::CreationOptions options;    // default options
    IRuntimePtr run = IRuntime::Create(options);

    // Set the tensors in the network.
    TensorInfo inputTensorInfo(TensorShape({ 1, params.m_IfmHeight, params.m_IfmWidth, params.m_NumIfm }),
                               armnn::DataType::QAsymmU8, 1.0f / 256.0f);
    InputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    uint32_t outHeight =
        (((params.m_IfmHeight + params.m_PadTop + params.m_PadBottom) - params.m_KernelHeight) / params.m_StrideY) + 1;
    uint32_t outWidth =
        (((params.m_IfmWidth + params.m_PadLeft + params.m_PadRight) - params.m_KernelWidth) / params.m_StrideX) + 1;

    TensorInfo outputTensorInfo(TensorShape({ 1, outHeight, outWidth, params.m_NumOfm }), armnn::DataType::QAsymmU8,
                                1.00001f / 256.0f);
    conv->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // Optimise Arm NN network
    IOptimizedNetworkPtr optNet = Optimize(*network, { Compute::CpuRef }, run->GetDeviceSpec());

    // Load graph into runtime
    NetworkId networkIdentifier;
    run->LoadNetwork(networkIdentifier, std::move(optNet));

    // Creates structures for inputs and outputs.
    InferenceOutputs outputData;

    OutputTensor outputData1 = MakeTensor(outputTensorInfo);
    outputData.push_back(std::move(outputData1));
    OutputTensor outputData2 = MakeTensor(outputTensorInfo);
    outputData.push_back(std::move(outputData2));

    TensorInfo runtimeInputTensorInfo = run->GetInputTensorInfo(networkIdentifier, 0);
    runtimeInputTensorInfo.SetConstant();
    InputTensors inputTensors1{ { 0, ConstTensor(runtimeInputTensorInfo, inputData1) } };
    InputTensors inputTensors2{ { 0, ConstTensor(runtimeInputTensorInfo, inputData2) } };
    OutputTensors outputTensors1{ { 0, armnn::Tensor(run->GetOutputTensorInfo(networkIdentifier, 0),
                                                     outputData[0]->GetByteData()) } };
    OutputTensors outputTensors2{ { 0, armnn::Tensor(run->GetOutputTensorInfo(networkIdentifier, 0),
                                                     outputData[1]->GetByteData()) } };

    // Execute network
    run->EnqueueWorkload(networkIdentifier, inputTensors1, outputTensors1);
    run->EnqueueWorkload(networkIdentifier, inputTensors2, outputTensors2);

    return outputData;
}

static InferenceOutputs
    CreateEthosNMultipleInferenceOutput(ConvParams params,
                                        std::shared_ptr<ethosn::driver_library::Buffer> ifm1,
                                        std::shared_ptr<ethosn::driver_library::Buffer> ifm2,
                                        const BaseTensor& weightData,
                                        const BaseTensor& biasData,
                                        const ethosn::support_library::CompilationOptions& options,
                                        std::shared_ptr<ethosn::driver_library::ProcMemAllocator> processMemAllocator)
{
    using namespace ethosn::support_library;

    if (!ethosn::driver_library::VerifyKernel())
    {
        throw std::runtime_error("Kernel version is not supported");
    }
    std::shared_ptr<Network> network = CreateNetwork(ethosn::driver_library::GetFirmwareAndHardwareCapabilities());
    std::shared_ptr<Operand> prevLayer;

    // Layer 0 Input
    {
        TensorInfo inputTensor =
            TensorInfo({ 1, params.m_IfmHeight, params.m_IfmWidth, params.m_NumIfm },
                       support_library::DataType::UINT8_QUANTIZED, params.m_Format, { 0, 1.0f / 256.0f });
        prevLayer = AddInput(network, inputTensor).tensor;
    }

    // Layer 2 Conv
    {
        TensorInfo weightInfo{ { params.m_KernelHeight, params.m_KernelWidth, params.m_NumIfm, params.m_NumOfm },
                               support_library::DataType::UINT8_QUANTIZED,
                               DataFormat::HWIO,
                               { 0, 1.0f } };
        TensorInfo biasInfo{ { 1, 1, 1, params.m_NumOfm },
                             support_library::DataType::INT32_QUANTIZED,
                             DataFormat::NHWC,
                             { 0, 1.0f / 256.0f } };
        ConvolutionInfo convInfo{ { params.m_PadTop, params.m_PadBottom, params.m_PadLeft, params.m_PadRight },
                                  { params.m_StrideX, params.m_StrideY },
                                  { 0, 1.00001f / 256.0f } };
        std::shared_ptr<Constant> bias    = AddConstant(network, biasInfo, biasData.GetByteData()).tensor;
        std::shared_ptr<Constant> weights = AddConstant(network, weightInfo, weightData.GetByteData()).tensor;
        prevLayer                         = AddConvolution(network, *prevLayer, *bias, *weights, convInfo).tensor;
    }

    {
        std::shared_ptr<Output> output = AddOutput(network, *prevLayer, params.m_Format).tensor;
    }
    std::vector<std::unique_ptr<CompiledNetwork>> compiledNetworks = Compile(*network, options);

    REQUIRE(compiledNetworks[0]);
    std::vector<char> compiledNetworkData;
    {
        ethosn::utils::VectorStream compiledNetworkStream(compiledNetworkData);
        compiledNetworks[0]->Serialize(compiledNetworkStream);
    }

    ethosn::driver_library::Network netInst =
        processMemAllocator->CreateNetwork(compiledNetworkData.data(), compiledNetworkData.size());

    std::unique_ptr<ethosn::driver_library::Network> ethosn =
        std::make_unique<ethosn::driver_library::Network>(std::move(netInst));

    uint32_t bufferSize;

    uint32_t outHeight =
        (((params.m_IfmHeight + params.m_PadTop + params.m_PadBottom) - params.m_KernelHeight) / params.m_StrideY) + 1;
    uint32_t outWidth =
        (((params.m_IfmWidth + params.m_PadLeft + params.m_PadRight) - params.m_KernelWidth) / params.m_StrideX) + 1;

    REQUIRE(params.m_Format == DataFormat::NHWC);
    bufferSize = params.m_NumOfm * outHeight * outWidth;

    // Allocate space for a copy of the output buffer
    // The OFM is assumed to be the last buffer in the binding table
    InferenceOutputs outputBuffer;

    OutputTensor outputBuffer1 = MakeTensor(DataType::U8, bufferSize);

    OutputTensor outputBuffer2 = MakeTensor(DataType::U8, bufferSize);

    // Create output buffers.
    ethosn::driver_library::Buffer bufInst1 =
        processMemAllocator->CreateBuffer(outputBuffer1->GetByteData(), outputBuffer1->GetNumBytes());
    std::shared_ptr<ethosn::driver_library::Buffer> ofm1 =
        std::make_shared<ethosn::driver_library::Buffer>(std::move(bufInst1));

    ethosn::driver_library::Buffer bufInst2 =
        processMemAllocator->CreateBuffer(outputBuffer2->GetByteData(), outputBuffer2->GetNumBytes());
    std::shared_ptr<ethosn::driver_library::Buffer> ofm2 =
        std::make_shared<ethosn::driver_library::Buffer>(std::move(bufInst2));

    // Input and output buffer
    ethosn::driver_library::Buffer* ifmRaw1[1] = { ifm1.get() };
    ethosn::driver_library::Buffer* ifmRaw2[1] = { ifm2.get() };
    ethosn::driver_library::Buffer* ofmRaw1[1] = { ofm1.get() };
    ethosn::driver_library::Buffer* ofmRaw2[1] = { ofm2.get() };

    // Execute the inferences.
    std::unique_ptr<ethosn::driver_library::Inference> result1(ethosn->ScheduleInference(ifmRaw1, 1, ofmRaw1, 1));
    driver_library::InferenceResult inferenceResult = result1->Wait(60 * 1000);
    REQUIRE(inferenceResult == driver_library::InferenceResult::Completed);
    CopyBuffers({ ofmRaw1[0] }, { outputBuffer1->GetByteData() });

    std::unique_ptr<ethosn::driver_library::Inference> result2(ethosn->ScheduleInference(ifmRaw2, 1, ofmRaw2, 1));
    inferenceResult = result2->Wait(60 * 1000);
    REQUIRE(inferenceResult == driver_library::InferenceResult::Completed);
    CopyBuffers({ ofmRaw2[0] }, { outputBuffer2->GetByteData() });

    outputBuffer.push_back(std::move(outputBuffer1));
    outputBuffer.push_back(std::move(outputBuffer2));

    return outputBuffer;
}

TEST_CASE("ZeroCopy")
{
    ConvParams params;
    params.m_NumIfm       = 16;
    params.m_NumOfm       = 16;
    params.m_IfmWidth     = 16;
    params.m_IfmHeight    = 16;
    params.m_KernelWidth  = 1;
    params.m_KernelHeight = 1;
    params.m_PadLeft      = 0;
    params.m_PadRight     = 0;
    params.m_PadBottom    = 0;
    params.m_PadTop       = 0;
    params.m_Format       = ethosn::support_library::DataFormat::NHWC;
    params.m_StrideX      = 1;
    params.m_StrideY      = 1;
    params.m_Debug        = false;

    // check the kernel version to be higher or equal to 5.6.
    if (!ethosn::utils::IsKernelVersionHigherOrEqualTo(5, 6))
    {
        INFO("Kernel version lower than 5.6.");
        INFO("No tests will be performed.");
        return;
    }

    // check that NPU core is behind a IOMMU.
    if (!ethosn::utils::IsNpuCoreBehindIommus())
    {
        INFO("No NPU core is behind a IOMMU or \"ethosn@xxxxxxx\" not found in the device tree.");
        INFO("No tests will be performed.");
        return;
    }

    std::unique_ptr<DmaBufferDevice> dmaBufHeap(new DmaBufferDevice(g_DmaBufHeap.c_str()));

    srand(42);

    const uint32_t size1 = params.m_IfmHeight * params.m_IfmWidth * params.m_NumIfm;
    DmaBuffer inputDataDmaBuf1(dmaBufHeap, size1);
    // Fill it with random data
    std::vector<uint8_t> inputData1(size1);
    for (uint32_t i = 0; i < size1; i++)
    {
        inputData1[i] = static_cast<uint8_t>(rand() % 8);
    }
    inputDataDmaBuf1.PopulateData(inputData1.data(), size1);

    const uint32_t size2 = params.m_IfmHeight * params.m_IfmWidth * params.m_NumIfm;
    DmaBuffer inputDataDmaBuf2(dmaBufHeap, size2);
    std::vector<uint8_t> inputData2(size2);
    // Fill it with random data
    for (uint32_t i = 0; i < size2; i++)
    {
        inputData2[i] = static_cast<uint8_t>(rand() % 8);
    }
    inputDataDmaBuf2.PopulateData(inputData2.data(), size2);

    std::vector<uint8_t> ethosnWeightData, armnnWeightData;
    std::tie(ethosnWeightData, armnnWeightData) =
        GenerateWeightData({ params.m_KernelHeight, params.m_KernelWidth, params.m_NumIfm, params.m_NumOfm }, 7);

    std::vector<int32_t> biasData(params.m_NumOfm);
    generate(biasData.begin(), biasData.end(), []() -> uint8_t { return static_cast<uint8_t>(rand() % 32); });

    InferenceOutputs refOutput = CreateMultipleInferenceRef(params, inputData1.data(), inputData2.data(),
                                                            *MakeTensor(armnnWeightData), *MakeTensor(biasData));

    std::shared_ptr<ethosn::driver_library::ProcMemAllocator> processMemAllocator =
        std::make_shared<ethosn::driver_library::ProcMemAllocator>();
    // Create input buffers.
    ethosn::driver_library::Buffer bufInst1 = processMemAllocator->ImportBuffer(inputDataDmaBuf1.GetFd(), size1);
    std::shared_ptr<ethosn::driver_library::Buffer> ifm1 =
        std::make_shared<ethosn::driver_library::Buffer>(std::move(bufInst1));

    ethosn::driver_library::Buffer bufInst2 = processMemAllocator->ImportBuffer(inputDataDmaBuf2.GetFd(), size2);
    std::shared_ptr<ethosn::driver_library::Buffer> ifm2 =
        std::make_shared<ethosn::driver_library::Buffer>(std::move(bufInst2));

    ethosn::support_library::CompilationOptions options;
    options.m_DebugInfo.m_DumpRam = params.m_Debug;
    InferenceOutputs actual = CreateEthosNMultipleInferenceOutput(params, ifm1, ifm2, *MakeTensor(ethosnWeightData),
                                                                  *MakeTensor(biasData), options, processMemAllocator);

    if (params.m_Debug)
    {
        DumpData("armnn1.hex", *refOutput[0]);
        DumpData("ethosn1.hex", *actual[0]);
        DumpData("armnn2.hex", *refOutput[1]);
        DumpData("ethosn2.hex", *actual[1]);
    }

    REQUIRE(CompareTensors(*actual[0], *refOutput[0], 0));
    REQUIRE(CompareTensors(*actual[1], *refOutput[1], 0));
    REQUIRE(!CompareTensors(*actual[0], *actual[1], 0));
}

}    // namespace system_tests
}    // namespace ethosn
