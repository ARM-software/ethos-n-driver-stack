//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include <EthosNBackend.hpp>
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
        return armnn::EthosNWorkloadFactory(config);
    }
};

}    // namespace

TEST_SUITE("EthosNMemCopy")
{

    TEST_CASE("CopyBetweenCpuAndEthosN")
    {
        // NOTE: Ethos-N only supports QAsymmU8 data
        LayerTestResult<uint8_t, 4> result =
            MemCopyTest<armnn::MockWorkloadFactory, armnn::EthosNWorkloadFactory, armnn::DataType::QAsymmU8>(false);
        auto predResult =
            CompareTensors(result.m_ActualData, result.m_ExpectedData, result.m_ActualShape, result.m_ExpectedShape);
        CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    }

    TEST_CASE("CopyBetweenEthosNAndCpu")
    {
        LayerTestResult<uint8_t, 4> result =
            MemCopyTest<armnn::EthosNWorkloadFactory, armnn::MockWorkloadFactory, armnn::DataType::QAsymmU8>(false);
        auto predResult =
            CompareTensors(result.m_ActualData, result.m_ExpectedData, result.m_ActualShape, result.m_ExpectedShape);
        CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    }

    TEST_CASE("CopyBetweenCpuAndEthosNWithSubtensors")
    {
        LayerTestResult<uint8_t, 4> result =
            MemCopyTest<armnn::MockWorkloadFactory, armnn::EthosNWorkloadFactory, armnn::DataType::QAsymmU8>(true);
        auto predResult =
            CompareTensors(result.m_ActualData, result.m_ExpectedData, result.m_ActualShape, result.m_ExpectedShape);
        CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    }

    TEST_CASE("CopyBetweenEthosNAndCpuWithSubtensors")
    {
        LayerTestResult<uint8_t, 4> result =
            MemCopyTest<armnn::EthosNWorkloadFactory, armnn::MockWorkloadFactory, armnn::DataType::QAsymmU8>(true);
        auto predResult =
            CompareTensors(result.m_ActualData, result.m_ExpectedData, result.m_ActualShape, result.m_ExpectedShape);
        CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    }
}
