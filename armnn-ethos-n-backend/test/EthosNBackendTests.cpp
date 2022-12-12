//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#include "EthosNBackend.hpp"
#include "EthosNSubgraphViewConverter.hpp"
#include "EthosNTensorHandle.hpp"
#include "EthosNTestUtils.hpp"
#include "EthosNWorkloadFactory.hpp"
#include "EthosNWorkloads.hpp"

#include <doctest/doctest.h>

#include <fcntl.h>
#include <fstream>
#ifdef _WIN32
#include <io.h>
#endif
#if defined(__unix__)
#include <sys/stat.h>
#include <unistd.h>
#endif

using namespace armnn;

TEST_SUITE("EthosNBackend")
{
    TEST_CASE("GetCapabilitiesTest")
    {
        EthosNBackend backend;
        BackendCapabilities backendCap = backend.GetCapabilities();
        uint32_t defaultDeviceId       = 0;
        uint32_t numberOfDevices       = 1;

        BackendCapabilities expectedCap("EthosNAcc");
        expectedCap.AddOption(BackendOptions::BackendOption("DeviceNamePrefix", "/dev/ethosn"));
        expectedCap.AddOption(BackendOptions::BackendOption("DeviceBaseId", defaultDeviceId));
        expectedCap.AddOption(BackendOptions::BackendOption("NumberOfDevices", numberOfDevices));
        expectedCap.AddOption(BackendOptions::BackendOption("ConstantTensorsAsInputs", true));
        expectedCap.AddOption(BackendOptions::BackendOption("AsyncExecution", true));
        expectedCap.AddOption(BackendOptions::BackendOption("ExternallyManagedMemory", true));
        expectedCap.AddOption(BackendOptions::BackendOption("PreImportIOTensors", true));

        CHECK(backendCap.GetBackendId().Get() == expectedCap.GetBackendId().Get());
        CHECK(backendCap.GetOptionCount() == expectedCap.GetOptionCount());
        CHECK(backendCap.GetOption(0).GetName() == expectedCap.GetOption(0).GetName());
        CHECK(backendCap.GetOption(0).GetValue().AsString() == expectedCap.GetOption(0).GetValue().AsString());
        CHECK(backendCap.GetOption(1).GetName() == expectedCap.GetOption(1).GetName());
        CHECK(backendCap.GetOption(1).GetValue().AsUnsignedInt() ==
              expectedCap.GetOption(1).GetValue().AsUnsignedInt());
        CHECK(backendCap.GetOption(2).GetName() == expectedCap.GetOption(2).GetName());
        CHECK(backendCap.GetOption(2).GetValue().AsUnsignedInt() ==
              expectedCap.GetOption(2).GetValue().AsUnsignedInt());
        CHECK(backendCap.GetOption(3).GetName() == expectedCap.GetOption(3).GetName());
        CHECK(backendCap.GetOption(3).GetValue().AsBool() == expectedCap.GetOption(3).GetValue().AsBool());
        CHECK(backendCap.GetOption(4).GetName() == expectedCap.GetOption(4).GetName());
        CHECK(backendCap.GetOption(4).GetValue().AsBool() == expectedCap.GetOption(4).GetValue().AsBool());
        CHECK(backendCap.GetOption(5).GetName() == expectedCap.GetOption(5).GetName());
        CHECK(backendCap.GetOption(5).GetValue().AsBool() == expectedCap.GetOption(5).GetValue().AsBool());
        CHECK(backendCap.GetOption(6).GetName() == expectedCap.GetOption(6).GetName());
        CHECK(backendCap.GetOption(6).GetValue().AsBool() == expectedCap.GetOption(6).GetValue().AsBool());
    }

    TEST_CASE("CreateWorkloadFactoryModelOptions")
    {
        EthosNBackend backend;
        ModelOptions modelOptions;
        const armnn::BackendOptions option("EthosNAcc", { { "Device", "/dev/ethosn0" } });
        modelOptions.push_back(option);

        IBackendInternal::IMemoryManagerSharedPtr memoryManager = backend.CreateMemoryManager();

        IBackendInternal::IWorkloadFactoryPtr workLoadFactory =
            backend.CreateWorkloadFactory(memoryManager, modelOptions);

        const std::string expectedDeviceId           = "/dev/ethosn0";
        EthosNWorkloadFactory* ethosnWorkLoadFactory = static_cast<EthosNWorkloadFactory*>(workLoadFactory.get());

        CHECK(expectedDeviceId == ethosnWorkLoadFactory->GetDeviceId());
    }

    TEST_CASE("CreateWorkloadFactoryModelOptionsNegative0")
    {
        // Negative test case where the backend option is "NeonAcc"
        // The expected device ID returned would be ""
        EthosNBackend backend;
        ModelOptions modelOptions;
        const armnn::BackendOptions option("NeonNAcc", { { "Device", "/dev/ethosn0" } });
        modelOptions.push_back(option);

        IBackendInternal::IMemoryManagerSharedPtr memoryManager = backend.CreateMemoryManager();

        IBackendInternal::IWorkloadFactoryPtr workLoadFactory =
            backend.CreateWorkloadFactory(memoryManager, modelOptions);

        EthosNWorkloadFactory* ethosnWorkLoadFactory = reinterpret_cast<EthosNWorkloadFactory*>(workLoadFactory.get());

        CHECK(ethosnWorkLoadFactory->GetDeviceId() == "");
    }

    TEST_CASE("CreateWorkloadFactoryModelOptionsNegative1")
    {
        // Negative test case where the option value is an integer
        // instead of string as expected.
        // An invalid argument exception would be expected.
        EthosNBackend backend;
        ModelOptions modelOptions;
        const armnn::BackendOptions option("EthosNAcc", { { "Device", 2 } });
        modelOptions.push_back(option);

        IBackendInternal::IMemoryManagerSharedPtr memoryManager = backend.CreateMemoryManager();

        CHECK_THROWS_AS(IBackendInternal::IWorkloadFactoryPtr workLoadFactory =
                            backend.CreateWorkloadFactory(memoryManager, modelOptions),
                        armnn::InvalidArgumentException);
    }

    TEST_CASE("GetLayerSupport")
    {
        EthosNBackend backend;
        ModelOptions modelOptions;
        const armnn::BackendOptions option("EthosNAcc", { { "Device", "/dev/ethosn0" } });
        modelOptions.push_back(option);

        IBackendInternal::ILayerSupportSharedPtr supportPtr = backend.GetLayerSupport(modelOptions);

        CHECK(supportPtr.get() != nullptr);
    }

    TEST_CASE("GetLayerSupportNegative0")
    {
        EthosNBackend backend;
        ModelOptions modelOptions;
        const armnn::BackendOptions option("EthosNAcc", { { "Device", 100 } });
        modelOptions.push_back(option);

        CHECK_THROWS_AS(IBackendInternal::ILayerSupportSharedPtr supportPtr = backend.GetLayerSupport(modelOptions),
                        armnn::InvalidArgumentException);
    }
}

bool IsOnHardware()
{
    std::ifstream f("/dev/ethosn0");
    return f.good();
}

TEST_SUITE("EthosNImportTensorHandle")
{
    TEST_CASE("Import")
    {
        bool onHardware = IsOnHardware();
        if (onHardware)
        {
            return;
        }
        // Create a file to be used as the file descriptor passed in
        // We don't create a dma_buf here but use a normal file descriptor for this test
        const char* path = "ImportTensorHandleTestFile";
        int fd           = open(path, O_RDWR | O_CREAT, S_IREAD | S_IWRITE);
        CHECK(fd > 0);

        // Register and get allocators to ensure the allocators exist
        EthosNConfig config;
        EthosNBackendAllocatorService::GetInstance().RegisterAllocator(config, {});
        EthosNBackendAllocatorService::GetInstance().GetAllocators();

        // Create an ethosn import tensor handle factory with dma buf
        EthosNImportTensorHandleFactory handleFactory(config);
        CHECK(handleFactory.GetImportFlags() == static_cast<MemorySourceFlags>(MemorySource::DmaBuf));
        CHECK(handleFactory.GetExportFlags() == static_cast<MemorySourceFlags>(MemorySource::DmaBuf));

        // Create a tensor info needed to create the tensor handle
        TensorInfo info({ 1, 16, 16, 16 }, DataType::QAsymmU8);
        unsigned int numElements = info.GetNumElements();
        // Create some data and fill in the file defined above
        std::vector<uint8_t> data(numElements, 127);
        int numBytesWritten = static_cast<int>(write(fd, data.data(), numElements));
        CHECK(numBytesWritten == numElements);
        // reset the file descriptor to the beginning so when we Import we read the correct data
        lseek(fd, 0, SEEK_SET);

        auto handle = handleFactory.CreateTensorHandle(info);
        CHECK(handle->GetImportFlags() == static_cast<MemorySourceFlags>(MemorySource::DmaBuf));

        bool imported = handle->Import(reinterpret_cast<void*>(&fd), MemorySource::DmaBuf);
        CHECK(imported);

        // Check that the imported data is actually the data we added to the file.
        uint8_t* buf = reinterpret_cast<uint8_t*>(handle->Map());
        for (unsigned int i = 0; i < numElements; i++)
        {
            CHECK(buf[i] == data[i]);
        }

        // Modify the internal data
        for (unsigned int i = 0; i < numElements; i++)
        {
            buf[i]++;
        }

        handle->Unmap();

        handle->Unimport();

        lseek(fd, 0, SEEK_SET);
        std::vector<uint8_t> outputData(numElements);
        int bytesRead = static_cast<int>(read(fd, outputData.data(), numElements));
        CHECK(bytesRead == static_cast<int>(numElements));

        for (unsigned int i = 0; i < numElements; i++)
        {
            CHECK(outputData[i] == data[i] + 1);
        }

        close(fd);

        EthosNBackendAllocatorService::GetInstance().PutAllocators();
    }

    TEST_CASE("ExecutionWithImportInputsAndOutputs")
    {
        bool onHardware = IsOnHardware();
        if (onHardware)
        {
            return;
        }

        // Create a file to be used as the file descriptor passed in
        // We don't create a dma_buf here but use a normal file descriptor for this test
        const char* path = "ExecutionWithImportInput";
        int fd           = open(path, O_RDWR | O_CREAT, S_IREAD | S_IWRITE);
        CHECK(fd > 0);

        // Create a file to be used as the file descriptor for output tensors
        const char* outputPath = "ExecutionWithImportOutput";
        int outputFd           = open(outputPath, O_RDWR | O_CREAT, S_IREAD | S_IWRITE);
        CHECK(outputFd > 0);

        using namespace testing_utils;

        armnn::EthosNConfig config{};
        config.m_PerfVariant = ethosn::support_library::EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO;
        config.m_PerfOnly    = false;

        BackendGlobalConfigSetter configSetter(config, config.QueryCapabilities());

        // Register allocators
        EthosNBackendAllocatorService::GetInstance().RegisterAllocator(config, {});

        armnn::EthosNWorkloadFactory factory(config);
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
        inputTensorInfo.SetConstant();

        TensorInfo outputTensorInfo(TensorShape({ 1, 16, 16, 16 }), armnn::DataType::QAsymmU8);
        outputTensorInfo.SetQuantizationOffset(0);
        outputTensorInfo.SetQuantizationScale(1.f);

        inputLayer->GetOutputSlot(0).Connect(reluLayer->GetInputSlot(0));
        inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

        reluLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
        reluLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

        std::vector<armnn::BackendId> backends = { factory.GetBackendId() };
        armnn::IRuntime::CreationOptions options;
        armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));
        armnn::OptimizerOptions optimizerOptions;
        optimizerOptions.m_ImportEnabled = true;
        optimizerOptions.m_ExportEnabled = true;
        armnn::IOptimizedNetworkPtr optimizedNet =
            armnn::Optimize(*net, backends, runtime->GetDeviceSpec(), optimizerOptions);
        CHECK(optimizedNet);

        // Load graph into runtime
        armnn::NetworkId networkIdentifier;
        INetworkProperties networkProperties(false, MemorySource::DmaBuf, MemorySource::DmaBuf);
        std::string errMsgs;
        runtime->LoadNetwork(networkIdentifier, std::move(optimizedNet), errMsgs, networkProperties);

        // Create some data and fill in the file descriptor
        unsigned int numElements = inputTensorInfo.GetNumElements();
        std::vector<uint8_t> inputData(numElements, 127);
        int numBytesWritten = static_cast<int>(write(fd, inputData.data(), numElements));
        CHECK(numBytesWritten == numElements);
        // reset the file descriptor to the beginning so when we Import we read the correct data
        lseek(fd, 0, SEEK_SET);

        // Explicitly initialize the output data to 0 to be different from the input
        // so we don't assume that the input is correct.
        std::vector<uint8_t> outputData(outputTensorInfo.GetNumElements(), 0);
        numBytesWritten = static_cast<int>(write(outputFd, outputData.data(), numElements));
        CHECK(numBytesWritten == numElements);
        lseek(outputFd, 0, SEEK_SET);

        InputTensors inputTensors{
            { 0, ConstTensor(runtime->GetInputTensorInfo(networkIdentifier, 0), &fd) },
        };
        OutputTensors outputTensors{ { 0, Tensor(runtime->GetOutputTensorInfo(networkIdentifier, 0), &outputFd) } };

        // Do the inference
        auto ret = runtime->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);
        CHECK(ret == Status::Success);

        // reset the file descriptor to the beginning
        // and check the output is expected
        lseek(outputFd, 0, SEEK_SET);
        int numBytesRead = static_cast<int>(read(outputFd, outputData.data(), numElements));
        CHECK(numBytesRead == numElements);

        for (unsigned int i = 0; i < numElements; i++)
        {
            CHECK(outputData[i] == inputData[i]);
        }
    }

    // Test using the Preimport API for both importing inputs and outputs
    TEST_CASE("ExecutionWithImportInputsAndOutputsPreImport")
    {
        bool onHardware = IsOnHardware();
        if (onHardware)
        {
            return;
        }

        // Create a file to be used as the file descriptor passed in
        // We don't create a dma_buf here but use a normal file descriptor for this test
        const char* path = "ExecutionWithImportInput";
        int fd           = open(path, O_RDWR | O_CREAT, S_IREAD | S_IWRITE);
        CHECK(fd > 0);

        // Create a file to be used as the file descriptor for output tensors
        const char* outputPath = "ExecutionWithImportOutput";
        int outputFd           = open(outputPath, O_RDWR | O_CREAT, S_IREAD | S_IWRITE);
        CHECK(outputFd > 0);

        using namespace testing_utils;

        armnn::EthosNConfig config{};
        config.m_PerfVariant = ethosn::support_library::EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO;
        config.m_PerfOnly    = false;

        BackendGlobalConfigSetter configSetter(config, config.QueryCapabilities());

        // Register allocators
        EthosNBackendAllocatorService::GetInstance().RegisterAllocator(config, {});

        armnn::EthosNWorkloadFactory factory(config);
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
        inputTensorInfo.SetConstant();

        TensorInfo outputTensorInfo(TensorShape({ 1, 16, 16, 16 }), armnn::DataType::QAsymmU8);
        outputTensorInfo.SetQuantizationOffset(0);
        outputTensorInfo.SetQuantizationScale(1.f);

        inputLayer->GetOutputSlot(0).Connect(reluLayer->GetInputSlot(0));
        inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

        reluLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
        reluLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

        std::vector<armnn::BackendId> backends = { factory.GetBackendId() };
        armnn::IRuntime::CreationOptions options;
        armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));
        armnn::OptimizerOptions optimizerOptions;
        optimizerOptions.m_ImportEnabled = false;
        optimizerOptions.m_ExportEnabled = false;
        armnn::IOptimizedNetworkPtr optimizedNet =
            armnn::Optimize(*net, backends, runtime->GetDeviceSpec(), optimizerOptions);
        CHECK(optimizedNet);

        // Load graph into runtime
        armnn::NetworkId networkIdentifier;
        INetworkProperties networkProperties(false, MemorySource::Undefined, MemorySource::Undefined);
        std::string errMsgs;
        armnn::Status loadNetworkRes =
            runtime->LoadNetwork(networkIdentifier, std::move(optimizedNet), errMsgs, networkProperties);
        CHECK(loadNetworkRes == Status::Success);

        // Create some data and fill in the file descriptor
        unsigned int numElements = inputTensorInfo.GetNumElements();
        std::vector<uint8_t> inputData(numElements, 127);
        int numBytesWritten = static_cast<int>(write(fd, inputData.data(), numElements));
        CHECK(numBytesWritten == numElements);
        // reset the file descriptor to the beginning so when we Import we read the correct data
        lseek(fd, 0, SEEK_SET);

        // Explicitly initialize the output data to 0 to be different from the input
        // so we don't assume that the input is correct.
        std::vector<uint8_t> outputData(outputTensorInfo.GetNumElements(), 0);
        numBytesWritten = static_cast<int>(write(outputFd, outputData.data(), numElements));
        CHECK(numBytesWritten == numElements);
        lseek(outputFd, 0, SEEK_SET);

        InputTensors inputTensors{
            { 0, ConstTensor(runtime->GetInputTensorInfo(networkIdentifier, 0), &fd) },
        };
        OutputTensors outputTensors{ { 0, Tensor(runtime->GetOutputTensorInfo(networkIdentifier, 0), &outputFd) } };

        std::vector<ImportedInputId> importInputsIds =
            runtime->ImportInputs(networkIdentifier, inputTensors, MemorySource::DmaBuf);

        std::vector<ImportedInputId> importOutputsIds =
            runtime->ImportOutputs(networkIdentifier, outputTensors, MemorySource::DmaBuf);

        // Do the inference
        auto ret = runtime->EnqueueWorkload(networkIdentifier, {}, {}, importInputsIds, importOutputsIds);
        REQUIRE(ret == Status::Success);

        lseek(fd, 0, SEEK_SET);
        runtime->ClearImportedInputs(networkIdentifier, importInputsIds);
        runtime->ClearImportedOutputs(networkIdentifier, importOutputsIds);

        // reset the file descriptor to the beginning
        // and check the output is expected
        lseek(outputFd, 0, SEEK_SET);
        int numBytesRead = static_cast<int>(read(outputFd, outputData.data(), numElements));
        CHECK(numBytesRead == numElements);

        for (unsigned int i = 0; i < numElements; i++)
        {
            CHECK(outputData[i] == inputData[i]);
        }
    }

    // Test using the Preimport API for only importing inputs. Not outputs.
    TEST_CASE("ExecutionWithImportOnlyInputsPreimport")
    {
        bool onHardware = IsOnHardware();
        if (onHardware)
        {
            return;
        }

        // Create a file to be used as the file descriptor passed in
        // We don't create a dma_buf here but use a normal file descriptor for this test
        const char* path = "ExecutionWithImportInput";
        int fd           = open(path, O_RDWR | O_CREAT, S_IREAD | S_IWRITE);
        CHECK(fd > 0);

        using namespace testing_utils;

        armnn::EthosNConfig config{};
        config.m_PerfVariant = ethosn::support_library::EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO;
        config.m_PerfOnly    = false;

        BackendGlobalConfigSetter configSetter(config, config.QueryCapabilities());

        // Register allocators
        EthosNBackendAllocatorService::GetInstance().RegisterAllocator(config, {});

        armnn::EthosNWorkloadFactory factory(config);
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
        inputTensorInfo.SetConstant();

        TensorInfo outputTensorInfo(TensorShape({ 1, 16, 16, 16 }), armnn::DataType::QAsymmU8);
        outputTensorInfo.SetQuantizationOffset(0);
        outputTensorInfo.SetQuantizationScale(1.f);

        inputLayer->GetOutputSlot(0).Connect(reluLayer->GetInputSlot(0));
        inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

        reluLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
        reluLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

        std::vector<armnn::BackendId> backends = { factory.GetBackendId() };
        armnn::IRuntime::CreationOptions options;
        armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));
        armnn::OptimizerOptions optimizerOptions;
        optimizerOptions.m_ImportEnabled = false;
        armnn::IOptimizedNetworkPtr optimizedNet =
            armnn::Optimize(*net, backends, runtime->GetDeviceSpec(), optimizerOptions);
        CHECK(optimizedNet);

        // Load graph into runtime
        armnn::NetworkId networkIdentifier;
        INetworkProperties networkProperties(false, MemorySource::Undefined, MemorySource::Undefined);
        std::string errMsgs;
        auto loadNetwork = runtime->LoadNetwork(networkIdentifier, std::move(optimizedNet), errMsgs, networkProperties);
        CHECK(loadNetwork == armnn::Status::Success);

        // Create some data and fill in the file descriptor
        unsigned int numElements = inputTensorInfo.GetNumElements();
        std::vector<uint8_t> inputData(numElements, 127);
        int numBytesWritten = static_cast<int>(write(fd, inputData.data(), numElements));
        CHECK(numBytesWritten == numElements);
        // reset the file descriptor to the beginning so when we Import we read the correct data
        lseek(fd, 0, SEEK_SET);

        std::vector<uint8_t> outputData(outputTensorInfo.GetNumElements(), 0);

        InputTensors inputTensors{
            { 0, ConstTensor(runtime->GetInputTensorInfo(networkIdentifier, 0), &fd) },
        };
        OutputTensors outputTensors{ { 0, Tensor(runtime->GetOutputTensorInfo(networkIdentifier, 0),
                                                 outputData.data()) } };

        std::vector<ImportedInputId> importInputsIds =
            runtime->ImportInputs(networkIdentifier, inputTensors, MemorySource::DmaBuf);

        // Do the inference
        auto ret = runtime->EnqueueWorkload(networkIdentifier, {}, outputTensors, importInputsIds);
        CHECK(ret == Status::Success);

        runtime->ClearImportedInputs(networkIdentifier, importInputsIds);

        for (unsigned int i = 0; i < numElements; i++)
        {
            CHECK(outputData[i] == inputData[i]);
        }
    }

    // Test using the Preimport and Arm NN custom allocator API for both importing inputs and outputs
    TEST_CASE("CustomAllocatorTest")
    {
        class CustomAllocator : public armnn::ICustomAllocator
        {
        public:
            CustomAllocator()
                : m_NameCount(0)
            {}

            void* allocate(size_t size, size_t alignment)
            {
                IgnoreUnused(alignment);    // This function implementation does not support alignment
                IgnoreUnused(size);
                // Create a file to be used as the file descriptor passed in
                // We don't create a dma_buf here but use a normal file descriptor for this test
                std::string fileName = "/tmp/bufferFile" + std::to_string(m_NameCount++) + ".bin";
                int fd               = open(fileName.c_str(), O_RDWR | O_CREAT, S_IREAD | S_IWRITE);
                REQUIRE(fd >= 0);
                m_Map[fd] = fd;
                return static_cast<void*>(&m_Map[fd]);
            }

            void free(void* ptr)
            {
                int fd = *static_cast<int*>(ptr);
                close(fd);
                m_Map.erase(fd);
            }

            armnn::MemorySource GetMemorySourceType()
            {
                return armnn::MemorySource::DmaBuf;
            }

            void PopulateData(void* ptr, const uint8_t* inData, size_t len)
            {
                int fd = *static_cast<int*>(ptr);
                lseek(fd, 0, SEEK_SET);
                auto ret = write(fd, inData, len);
                REQUIRE(ret == len);
                lseek(fd, 0, SEEK_SET);
            }

            void RetrieveData(void* ptr, uint8_t* outData, size_t len)
            {
                int fd = *static_cast<int*>(ptr);
                lseek(fd, 0, SEEK_SET);
                auto ret = read(fd, outData, len);
                REQUIRE(ret == len);
                lseek(fd, 0, SEEK_SET);
            }

            ~CustomAllocator()
            {
                for (const auto& it : m_Map)
                {
                    close(it.second);
                }
            }

        private:
            int m_NameCount;
            std::map<int, int> m_Map;
        };

        // Ensure to run this test on the model only
        bool onHardware = IsOnHardware();
        if (onHardware)
        {
            return;
        }

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
        armnn::OptimizerOptions optimizerOptions;
        optimizerOptions.m_ImportEnabled = true;
        optimizerOptions.m_ExportEnabled = true;
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

        customAllocatorRef->RetrieveData(inputFd, inputBuffer.data(), totalBytes);
        customAllocatorRef->RetrieveData(outputFd, outputBuffer.data(), totalBytes);
        for (unsigned int i = 0; i < numElements; i++)
        {
            CHECK(outputBuffer[i] == inputBuffer[i]);
        }
    }
}
