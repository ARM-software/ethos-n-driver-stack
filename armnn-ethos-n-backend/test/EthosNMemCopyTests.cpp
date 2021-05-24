//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "EthosNWorkloadFactoryHelper.hpp"

#include <EthosNWorkloadFactory.hpp>
#include <aclCommon/test/MemCopyTestImpl.hpp>
#include <doctest/doctest.h>
#include <reference/RefWorkloadFactory.hpp>
#include <reference/test/RefWorkloadFactoryHelper.hpp>

TEST_SUITE("EthosNMemCopy")
{

    TEST_CASE("CopyBetweenCpuAndEthosN")
    {
        // NOTE: Ethos-N  only supports QAsymmU8 data
        LayerTestResult<uint8_t, 4> result =
            MemCopyTest<armnn::RefWorkloadFactory, armnn::EthosNWorkloadFactory, armnn::DataType::QAsymmU8>(false);
        auto predResult =
            CompareTensors(result.m_ActualData, result.m_ExpectedData, result.m_ActualShape, result.m_ExpectedShape);
        CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    }

    TEST_CASE("CopyBetweenEthosNAndCpu")
    {
        LayerTestResult<uint8_t, 4> result =
            MemCopyTest<armnn::EthosNWorkloadFactory, armnn::RefWorkloadFactory, armnn::DataType::QAsymmU8>(false);
        auto predResult =
            CompareTensors(result.m_ActualData, result.m_ExpectedData, result.m_ActualShape, result.m_ExpectedShape);
        CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    }

    TEST_CASE("CopyBetweenCpuAndEthosNWithSubtensors")
    {
        LayerTestResult<uint8_t, 4> result =
            MemCopyTest<armnn::RefWorkloadFactory, armnn::EthosNWorkloadFactory, armnn::DataType::QAsymmU8>(true);
        auto predResult =
            CompareTensors(result.m_ActualData, result.m_ExpectedData, result.m_ActualShape, result.m_ExpectedShape);
        CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    }

    TEST_CASE("CopyBetweenEthosNAndCpuWithSubtensors")
    {
        LayerTestResult<uint8_t, 4> result =
            MemCopyTest<armnn::EthosNWorkloadFactory, armnn::RefWorkloadFactory, armnn::DataType::QAsymmU8>(true);
        auto predResult =
            CompareTensors(result.m_ActualData, result.m_ExpectedData, result.m_ActualShape, result.m_ExpectedShape);
        CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    }
}
