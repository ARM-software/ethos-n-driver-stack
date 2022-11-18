//
// Copyright © 2019,2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "EthosNConfig.hpp"
#include "armnn/backends/ITensorHandleFactory.hpp"
#include <ethosn_driver_library/ProcMemAllocator.hpp>

namespace armnn
{
/// The TensorHandleFactory for non import tensors
class EthosNTensorHandleFactory : public ITensorHandleFactory
{
public:
    virtual ~EthosNTensorHandleFactory()
    {}

    EthosNTensorHandleFactory(const EthosNConfig& config)
        : m_EthosNConfig(config)
    {}

    EthosNTensorHandleFactory(const EthosNConfig& config, const std::string& deviceId)
        : m_EthosNConfig(config)
        , m_DeviceId(deviceId)
    {}

    virtual std::unique_ptr<ITensorHandle> CreateSubTensorHandle(ITensorHandle& parent,
                                                                 TensorShape const& subTensorShape,
                                                                 unsigned int const* subTensorOrigin) const override;

    virtual std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo) const override;

    virtual std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo,
                                                              DataLayout dataLayout) const override;

    static const FactoryId& GetIdStatic();

    virtual const FactoryId& GetId() const override;

    virtual bool SupportsSubTensors() const override;

private:
    EthosNConfig m_EthosNConfig;
    std::string m_DeviceId;
};

/// The TensorHandleFactory for import tensors
class EthosNImportTensorHandleFactory : public ITensorHandleFactory
{
public:
    virtual ~EthosNImportTensorHandleFactory()
    {}

    EthosNImportTensorHandleFactory(const EthosNConfig& config,
                                    MemorySourceFlags importFlags,
                                    MemorySourceFlags exportFlags)
        : m_EthosNConfig(config)
        , m_ImportFlags(importFlags)
        , m_ExportFlags(exportFlags)
    {}

    EthosNImportTensorHandleFactory(const EthosNConfig& config,
                                    const std::string& deviceId,
                                    MemorySourceFlags importFlags,
                                    MemorySourceFlags exportFlags)
        : m_EthosNConfig(config)
        , m_DeviceId(deviceId)
        , m_ImportFlags(importFlags)
        , m_ExportFlags(exportFlags)
    {}

    virtual std::unique_ptr<ITensorHandle> CreateSubTensorHandle(ITensorHandle& parent,
                                                                 TensorShape const& subTensorShape,
                                                                 unsigned int const* subTensorOrigin) const override;

    virtual std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo) const override;

    virtual std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo,
                                                              DataLayout dataLayout) const override;

    static const FactoryId& GetIdStatic();

    const FactoryId& GetId() const override;

    bool SupportsSubTensors() const override;

    MemorySourceFlags GetImportFlags() const override
    {
        return m_ImportFlags;
    }
    MemorySourceFlags GetExportFlags() const override
    {
        return m_ExportFlags;
    }

private:
    EthosNConfig m_EthosNConfig;
    std::string m_DeviceId;
    MemorySourceFlags m_ImportFlags;
    MemorySourceFlags m_ExportFlags;
};
}    // namespace armnn
