//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <EthosNBackend.hpp>
#include <EthosNConfig.hpp>
#include <EthosNWorkloadFactory.hpp>
#include <backendsCommon/test/WorkloadFactoryHelper.hpp>

namespace
{

template <>
struct WorkloadFactoryHelper<armnn::EthosNWorkloadFactory>
{
    static armnn::IBackendInternal::IMemoryManagerSharedPtr GetMemoryManager()
    {
        armnn::EthosNBackend backend;
        return backend.CreateMemoryManager();
    }

    static armnn::EthosNWorkloadFactory GetFactory(const armnn::IBackendInternal::IMemoryManagerSharedPtr& = nullptr)
    {
        return armnn::EthosNWorkloadFactory(armnn::EthosNConfig());
    }
};

using EthosNWorkloadFactoryHelper = WorkloadFactoryHelper<armnn::EthosNWorkloadFactory>;

}    // anonymous namespace
