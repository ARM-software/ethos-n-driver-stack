//
// Copyright © 2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "ProtectedAllocator.hpp"
#include "SystemTestsUtils.hpp"
#include <armnn/ArmNN.hpp>
#include <catch.hpp>

using namespace armnn;
using namespace ethosn::system_tests;

// Test using the Preimport and Arm NN custom allocator API for both importing inputs and outputs
TEST_CASE("ProtectedCustomAllocatorTest", "[TZMP1-Test-Module]")
{

    // To create a PreCompiled layer, create a network and Optimize it.
    armnn::INetworkPtr net = armnn::INetwork::Create();

    armnn::IConnectableLayer* const inputLayer = net->AddInputLayer(0, "input layer");
    CHECK(inputLayer);

    ActivationDescriptor reluDesc;
    reluDesc.m_A                              = 255;
    reluDesc.m_B                              = 0;
    reluDesc.m_Function                       = ActivationFunction::BoundedReLu;
    armnn::IConnectableLayer* const reluLayer = net->AddActivationLayer(reluDesc, "relu layer");
    CHECK(reluLayer);

    armnn::IConnectableLayer* const outputLayer = net->AddOutputLayer(0, "output layer");
    CHECK(outputLayer);

    TensorInfo inputTensorInfo(TensorShape({ 1, 16, 16, 16 }), armnn::DataType::QAsymmU8);
    inputTensorInfo.SetQuantizationOffset(0);
    inputTensorInfo.SetQuantizationScale(1.f);
    inputTensorInfo.SetConstant(true);

    TensorInfo outputTensorInfo(TensorShape({ 1, 16, 16, 16 }), armnn::DataType::QAsymmU8);
    outputTensorInfo.SetQuantizationOffset(0);
    outputTensorInfo.SetQuantizationScale(1.f);

    inputLayer->GetOutputSlot(0).Connect(reluLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    reluLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    reluLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    std::string id = "EthosNAcc";
    armnn::IRuntime::CreationOptions options;
    auto customAllocator         = std::make_shared<ProtectedAllocator>();
    options.m_CustomAllocatorMap = { { id, std::move(customAllocator) } };
    options.m_ProtectedMode      = true;
    auto customAllocatorRef      = static_cast<ProtectedAllocator*>(options.m_CustomAllocatorMap[id].get());
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));
    armnn::OptimizerOptionsOpaque optimizerOptions;
    optimizerOptions.SetImportEnabled(true);
    optimizerOptions.SetExportEnabled(true);
    armnn::IOptimizedNetworkPtr optimizedNet =
        armnn::Optimize(*net, { id }, runtime->GetDeviceSpec(), optimizerOptions);
    CHECK(optimizedNet);

    // Load graph into runtime
    armnn::NetworkId networkIdentifier;
    INetworkProperties networkProperties(false, customAllocatorRef->GetMemorySourceType(),
                                         customAllocatorRef->GetMemorySourceType());
    std::string errMsgs;
    armnn::Status loadNetworkRes =
        runtime->LoadNetwork(networkIdentifier, std::move(optimizedNet), errMsgs, networkProperties);
    CHECK(loadNetworkRes == Status::Success);

    // Create some data and fill in the buffers
    unsigned int numElements = inputTensorInfo.GetNumElements();
    size_t totalBytes        = numElements * sizeof(uint8_t);

    void* inputFd = customAllocatorRef->allocate(totalBytes, 0);
    std::vector<uint8_t> inputBuffer(totalBytes, 127);
    customAllocatorRef->PopulateData(inputFd, inputBuffer.data(), totalBytes);

    // Explicitly initialize the output buffer to 0 to be different from the input
    // so we don't assume that the input is correct.
    void* outputFd = customAllocatorRef->allocate(totalBytes, 0);
    std::vector<uint8_t> outputBuffer(totalBytes, 0);
    customAllocatorRef->PopulateData(outputFd, outputBuffer.data(), totalBytes);

    InputTensors inputTensors{
        { 0, ConstTensor(runtime->GetInputTensorInfo(networkIdentifier, 0), inputFd) },
    };
    OutputTensors outputTensors{
        { 0, Tensor(runtime->GetOutputTensorInfo(networkIdentifier, 0), outputFd) },
    };

    auto ret = runtime->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);
    REQUIRE(ret == Status::Success);

    ret = runtime->UnloadNetwork(networkIdentifier);
    REQUIRE(ret == Status::Success);

    customAllocatorRef->RetrieveData(inputFd, inputBuffer.data(), totalBytes);
    customAllocatorRef->RetrieveData(outputFd, outputBuffer.data(), totalBytes);
    for (unsigned int i = 0; i < numElements; i++)
    {
        CHECK(outputBuffer[i] == inputBuffer[i]);
    }
}
