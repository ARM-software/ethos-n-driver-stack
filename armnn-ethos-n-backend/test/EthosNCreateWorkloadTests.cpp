//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#include "EthosNTensorHandle.hpp"
#include "EthosNWorkloadFactory.hpp"
#include "EthosNWorkloads.hpp"

#include <test/CreateWorkload.hpp>

using namespace armnn;

BOOST_AUTO_TEST_SUITE(CreateWorkloadEthosN)

namespace
{

bool TestEthosNTensorHandleInfo(EthosNTensorHandle& handle, const TensorInfo& expectedInfo)
{
    return handle.GetTensorInfo() == expectedInfo;
}

template <typename PreCompiledWorkloadType, typename armnn::DataType dataType>
void EthosNCreatePreCompiledWorkloadTest(bool withBias = false)
{
    Graph graph;
    EthosNWorkloadFactory factory{ EthosNConfig() };
    auto workload = CreatePreCompiledWorkloadTest<PreCompiledWorkloadType, dataType>(factory, graph, withBias);

    // Checks that inputs/outputs are as we expect them (see definition of CreatePreCompiledWorkloadTest).
    PreCompiledQueueDescriptor queueDescriptor = workload.second->GetData();

    auto inputHandle  = PolymorphicPointerDowncast<EthosNTensorHandle>(queueDescriptor.m_Inputs[0]);
    auto outputHandle = PolymorphicPointerDowncast<EthosNTensorHandle>(queueDescriptor.m_Outputs[0]);

    BOOST_TEST(TestEthosNTensorHandleInfo(*inputHandle, TensorInfo({ 1, 16, 16, 16 }, dataType, 0.9f, 0)));
    BOOST_TEST(TestEthosNTensorHandleInfo(*outputHandle, TensorInfo({ 1, 16, 16, 16 }, dataType, 0.9f, 0)));
}

}    // namespace

BOOST_AUTO_TEST_CASE(CreatePreCompiledUint8Workload)
{
    EthosNCreatePreCompiledWorkloadTest<EthosNPreCompiledWorkload, armnn::DataType::QAsymmU8>();
}

BOOST_AUTO_TEST_CASE(CreatePreCompiledUint8WorkloadWithBiases)
{
    EthosNCreatePreCompiledWorkloadTest<EthosNPreCompiledWorkload, armnn::DataType::QAsymmU8>(true);
}

BOOST_AUTO_TEST_SUITE_END()
