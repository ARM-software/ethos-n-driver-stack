//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <EthosNBackend.hpp>
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
        return armnn::EthosNWorkloadFactory();
    }
};

using EthosNWorkloadFactoryHelper = WorkloadFactoryHelper<armnn::EthosNWorkloadFactory>;

}    // anonymous namespace
