//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "EthosNWorkloadFactoryHelper.hpp"

#include <EthosNWorkloadFactory.hpp>
#include <aclCommon/test/MemCopyTestImpl.hpp>
#include <boost/test/unit_test.hpp>
#include <reference/RefWorkloadFactory.hpp>
#include <reference/test/RefWorkloadFactoryHelper.hpp>

BOOST_AUTO_TEST_SUITE(EthosNMemCopy)

BOOST_AUTO_TEST_CASE(CopyBetweenCpuAndEthosN)
{
    // NOTE: Ethos-N  only supports QAsymmU8 data
    LayerTestResult<uint8_t, 4> result =
        MemCopyTest<armnn::RefWorkloadFactory, armnn::EthosNWorkloadFactory, armnn::DataType::QAsymmU8>(false);
    BOOST_TEST(CompareTensors(result.output, result.outputExpected));
}

BOOST_AUTO_TEST_CASE(CopyBetweenEthosNAndCpu)
{
    LayerTestResult<uint8_t, 4> result =
        MemCopyTest<armnn::EthosNWorkloadFactory, armnn::RefWorkloadFactory, armnn::DataType::QAsymmU8>(false);
    BOOST_TEST(CompareTensors(result.output, result.outputExpected));
}

BOOST_AUTO_TEST_CASE(CopyBetweenCpuAndEthosNWithSubtensors)
{
    LayerTestResult<uint8_t, 4> result =
        MemCopyTest<armnn::RefWorkloadFactory, armnn::EthosNWorkloadFactory, armnn::DataType::QAsymmU8>(true);
    BOOST_TEST(CompareTensors(result.output, result.outputExpected));
}

BOOST_AUTO_TEST_CASE(CopyBetweenEthosNAndCpuWithSubtensors)
{
    LayerTestResult<uint8_t, 4> result =
        MemCopyTest<armnn::EthosNWorkloadFactory, armnn::RefWorkloadFactory, armnn::DataType::QAsymmU8>(true);
    BOOST_TEST(CompareTensors(result.output, result.outputExpected));
}

BOOST_AUTO_TEST_SUITE_END()
