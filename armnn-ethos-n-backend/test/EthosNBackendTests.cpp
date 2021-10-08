//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#include "EthosNBackend.hpp"
#include "EthosNTensorHandle.hpp"
#include "EthosNWorkloadFactory.hpp"
#include "EthosNWorkloads.hpp"

#include <doctest/doctest.h>

using namespace armnn;

TEST_SUITE("BackendTests")
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
