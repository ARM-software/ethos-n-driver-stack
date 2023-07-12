//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "SystemTestsUtils.hpp"
#include <armnn/ArmNN.hpp>
#include <catch.hpp>

using namespace armnn;
using namespace ethosn::system_tests;

// Test using the Preimport and Arm NN custom allocator API for both importing inputs and outputs
TEST_CASE("CustomAllocatorTest")
{
    class CustomAllocator : public armnn::ICustomAllocator
    {
    public:
        CustomAllocator()
            : m_DmaBufHeap(new DmaBufferDevice("/dev/dma_heap/system"))
        {}

        void* allocate(size_t size, size_t alignment)
        {
            IgnoreUnused(alignment);    // This function implementation does not support alignment
            std::unique_ptr<DmaBuffer> dataDmaBuf = std::make_unique<DmaBuffer>(m_DmaBufHeap, size);
            int fd                                = dataDmaBuf->GetFd();
            REQUIRE(fd >= 0);
            m_Map[fd].m_DataDmaBuf = std::move(dataDmaBuf);
            m_Map[fd].m_Fd         = fd;
            return static_cast<void*>(&m_Map[fd].m_Fd);
        }

        void free(void* ptr)
        {
            int index = *static_cast<int*>(ptr);
            m_Map[index].m_DataDmaBuf.reset();
            m_Map.erase(index);
        }

        armnn::MemorySource GetMemorySourceType()
        {
            return armnn::MemorySource::DmaBuf;
        }

        void PopulateData(void* ptr, const uint8_t* inData, size_t len)
        {
            int index = *static_cast<int*>(ptr);
            m_Map[index].m_DataDmaBuf->PopulateData(inData, len);
        }

        void RetrieveData(void* ptr, uint8_t* outData, size_t len)
        {
            int index = *static_cast<int*>(ptr);
            m_Map[index].m_DataDmaBuf->RetrieveData(outData, len);
        }

    private:
        std::unique_ptr<DmaBufferDevice> m_DmaBufHeap;
        struct m_MapStruct
        {
            std::unique_ptr<DmaBuffer> m_DataDmaBuf;
            int m_Fd;
        };
        std::map<int, m_MapStruct> m_Map;
    };

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
    auto customAllocator         = std::make_shared<CustomAllocator>();
    options.m_CustomAllocatorMap = { { id, std::move(customAllocator) } };
    auto customAllocatorRef      = static_cast<CustomAllocator*>(options.m_CustomAllocatorMap[id].get());
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
