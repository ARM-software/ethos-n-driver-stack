//
// Copyright © 2022-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "EthosNTensorHandle.hpp"
#include "EthosNTensorHandleFactory.hpp"

namespace armnn
{

const ITensorHandleFactory::FactoryId& EthosNImportTensorHandleFactory::GetIdStatic()
{
    static const ITensorHandleFactory::FactoryId s_Id("EthosNImportTensorHandleFactory");
    return s_Id;
}

std::unique_ptr<ITensorHandle> EthosNImportTensorHandleFactory::CreateSubTensorHandle(ITensorHandle&,
                                                                                      TensorShape const&,
                                                                                      unsigned int const*) const
{
    return nullptr;
}

std::unique_ptr<ITensorHandle> EthosNImportTensorHandleFactory::CreateTensorHandle(const TensorInfo& tensorInfo) const
{
    return CreateTensorHandle(tensorInfo, DataLayout::NHWC);
}

std::unique_ptr<ITensorHandle> EthosNImportTensorHandleFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                                   DataLayout dataLayout) const
{
    // only NHWC format is supported
    if (dataLayout != DataLayout::NHWC)
    {
        return nullptr;
    }
    if (m_EthosNConfig.m_PerfOnly)
    {
        return std::make_unique<ScopedTensorHandle>(tensorInfo);
    }

    return std::make_unique<EthosNTensorHandle>(tensorInfo, m_DeviceId);
}

const ITensorHandleFactory::FactoryId& EthosNImportTensorHandleFactory::GetId() const
{
    return GetIdStatic();
}

bool EthosNImportTensorHandleFactory::SupportsSubTensors() const
{
    return false;
}

const ITensorHandleFactory::FactoryId& EthosNProtectedTensorHandleFactory::GetIdStatic()
{
    static const ITensorHandleFactory::FactoryId s_Id("EthosNProtectedTensorHandleFactory");
    return s_Id;
}

std::unique_ptr<ITensorHandle> EthosNProtectedTensorHandleFactory::CreateSubTensorHandle(ITensorHandle&,
                                                                                         TensorShape const&,
                                                                                         unsigned int const*) const
{
    return nullptr;
}

std::unique_ptr<ITensorHandle>
    EthosNProtectedTensorHandleFactory::CreateTensorHandle(const TensorInfo& tensorInfo) const
{
    return CreateTensorHandle(tensorInfo, DataLayout::NHWC);
}

std::unique_ptr<ITensorHandle> EthosNProtectedTensorHandleFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                                      DataLayout dataLayout) const
{
    // only NHWC format is supported
    if (dataLayout != DataLayout::NHWC)
    {
        return nullptr;
    }
    if (m_EthosNConfig.m_PerfOnly)
    {
        return std::make_unique<ScopedTensorHandle>(tensorInfo);
    }

    return std::make_unique<EthosNProtectedTensorHandle>(tensorInfo, m_DeviceId);
}

const ITensorHandleFactory::FactoryId& EthosNProtectedTensorHandleFactory::GetId() const
{
    return GetIdStatic();
}

bool EthosNProtectedTensorHandleFactory::SupportsSubTensors() const
{
    return false;
}
}    // namespace armnn
