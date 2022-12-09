//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include <EthosNBackend.hpp>
#include <EthosNConfig.hpp>
#include <EthosNTensorHandleFactory.hpp>
#include <EthosNWorkloadFactory.hpp>

#include <armnnTestUtils/LayerTestResult.hpp>
#include <armnnTestUtils/MemCopyTestImpl.hpp>
#include <armnnTestUtils/MockBackend.hpp>

#include <doctest/doctest.h>

namespace
{

template <>
struct MemCopyTestHelper<armnn::EthosNWorkloadFactory>
{
    static armnn::IBackendInternal::IMemoryManagerSharedPtr GetMemoryManager()
    {
        armnn::EthosNBackend backend;
        return backend.CreateMemoryManager();
    }

    static armnn::EthosNWorkloadFactory GetFactory(const armnn::IBackendInternal::IMemoryManagerSharedPtr&)
    {
        armnn::EthosNConfig config{};

        // Create process memory allocator if it does not already exist
        // Register and get allocators as this test doesn't call LoadNetwork
        armnn::EthosNBackendAllocatorService::GetInstance().RegisterAllocator(config, {});

        return armnn::EthosNWorkloadFactory(config);
    }
};

template <armnn::DataType dataType, typename T = armnn::ResolveType<dataType>>
LayerTestResult<T, 4> EthosNMemCopyTest(armnn::IWorkloadFactory& srcWorkloadFactory,
                                        armnn::IWorkloadFactory& dstWorkloadFactory,
                                        bool withSubtensors)
{
    const std::array<unsigned int, 4> shapeData = { { 1u, 1u, 6u, 5u } };
    const armnn::TensorShape tensorShape(4, shapeData.data());
    const armnn::TensorInfo tensorInfo(tensorShape, dataType);
    std::vector<T> inputData = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    };

    LayerTestResult<T, 4> ret(tensorInfo);
    ret.m_ExpectedData = inputData;

    std::vector<T> actualOutput(tensorInfo.GetNumElements());

    // Register and get allocators
    armnn::EthosNConfig config;
    armnn::EthosNBackendAllocatorService::GetInstance().RegisterAllocator(config, {});
    armnn::EthosNBackendAllocatorService::GetInstance().GetAllocators();

    armnn::EthosNWorkloadFactory factory(config);

    auto tensorHandleFactory = std::make_unique<armnn::EthosNImportTensorHandleFactory>(config);

    auto inputTensorHandle  = tensorHandleFactory->CreateTensorHandle(tensorInfo);
    auto outputTensorHandle = tensorHandleFactory->CreateTensorHandle(tensorInfo);

    AllocateAndCopyDataToITensorHandle(inputTensorHandle.get(), inputData.data());
    outputTensorHandle->Allocate();

    armnn::MemCopyQueueDescriptor memCopyQueueDesc;
    armnn::WorkloadInfo workloadInfo;

    const unsigned int origin[4] = {};

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    auto workloadInput = (withSubtensors && srcWorkloadFactory.SupportsSubTensors())
                             ? srcWorkloadFactory.CreateSubTensorHandle(*inputTensorHandle, tensorShape, origin)
                             : std::move(inputTensorHandle);
    auto workloadOutput = (withSubtensors && dstWorkloadFactory.SupportsSubTensors())
                              ? dstWorkloadFactory.CreateSubTensorHandle(*outputTensorHandle, tensorShape, origin)
                              : std::move(outputTensorHandle);
    ARMNN_NO_DEPRECATE_WARN_END

    AddInputToWorkload(memCopyQueueDesc, workloadInfo, tensorInfo, workloadInput.get());
    AddOutputToWorkload(memCopyQueueDesc, workloadInfo, tensorInfo, workloadOutput.get());

    dstWorkloadFactory.CreateWorkload(armnn::LayerType::MemCopy, memCopyQueueDesc, workloadInfo)->Execute();

    CopyDataFromITensorHandle(actualOutput.data(), workloadOutput.get());
    ret.m_ActualData = actualOutput;

    armnn::EthosNBackendAllocatorService::GetInstance().PutAllocators();

    return ret;
}

template <typename SrcWorkloadFactory,
          typename DstWorkloadFactory,
          armnn::DataType dataType,
          typename T = armnn::ResolveType<dataType>>
LayerTestResult<T, 4> EthosNMemCopyTest(bool withSubtensors)
{

    armnn::IBackendInternal::IMemoryManagerSharedPtr srcMemoryManager =
        MemCopyTestHelper<SrcWorkloadFactory>::GetMemoryManager();

    armnn::IBackendInternal::IMemoryManagerSharedPtr dstMemoryManager =
        MemCopyTestHelper<DstWorkloadFactory>::GetMemoryManager();

    SrcWorkloadFactory srcWorkloadFactory = MemCopyTestHelper<SrcWorkloadFactory>::GetFactory(srcMemoryManager);
    DstWorkloadFactory dstWorkloadFactory = MemCopyTestHelper<DstWorkloadFactory>::GetFactory(dstMemoryManager);

    return EthosNMemCopyTest<dataType>(srcWorkloadFactory, dstWorkloadFactory, withSubtensors);
}

}    // namespace

// Remove the following tests because the helper functions we use depend on the deprecated API.
TEST_SUITE("EthosNMemCopy")
{

    TEST_CASE("CopyBetweenCpuAndEthosN")
    {
        // NOTE: Ethos-N only supports QAsymmU8 data
        LayerTestResult<uint8_t, 4> result =
            EthosNMemCopyTest<armnn::MockWorkloadFactory, armnn::EthosNWorkloadFactory, armnn::DataType::QAsymmU8>(
                false);
        auto predResult =
            CompareTensors(result.m_ActualData, result.m_ExpectedData, result.m_ActualShape, result.m_ExpectedShape);
        CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    }

    TEST_CASE("CopyBetweenEthosNAndCpu")
    {
        LayerTestResult<uint8_t, 4> result =
            EthosNMemCopyTest<armnn::EthosNWorkloadFactory, armnn::MockWorkloadFactory, armnn::DataType::QAsymmU8>(
                false);
        auto predResult =
            CompareTensors(result.m_ActualData, result.m_ExpectedData, result.m_ActualShape, result.m_ExpectedShape);
        CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    }

    TEST_CASE("CopyBetweenCpuAndEthosNWithSubtensors")
    {
        LayerTestResult<uint8_t, 4> result =
            EthosNMemCopyTest<armnn::MockWorkloadFactory, armnn::EthosNWorkloadFactory, armnn::DataType::QAsymmU8>(
                true);
        auto predResult =
            CompareTensors(result.m_ActualData, result.m_ExpectedData, result.m_ActualShape, result.m_ExpectedShape);
        CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    }

    TEST_CASE("CopyBetweenEthosNAndCpuWithSubtensors")
    {
        LayerTestResult<uint8_t, 4> result =
            EthosNMemCopyTest<armnn::EthosNWorkloadFactory, armnn::MockWorkloadFactory, armnn::DataType::QAsymmU8>(
                true);
        auto predResult =
            CompareTensors(result.m_ActualData, result.m_ExpectedData, result.m_ActualShape, result.m_ExpectedShape);
        CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    }
}
