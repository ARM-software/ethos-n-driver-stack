//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "EthosNConfig.hpp"
#include <armnn/backends/WorkloadFactory.hpp>
#include <ethosn_driver_library/ProcMemAllocator.hpp>

namespace armnn
{

// Ethos-N workload factory
class EthosNWorkloadFactory : public IWorkloadFactory
{
public:
    EthosNWorkloadFactory(const EthosNConfig& config,
                          const std::shared_ptr<ethosn::driver_library::ProcMemAllocator> procMemAllocator)
        : m_EthosNConfig(config)
        , m_ProcMemAllocator(procMemAllocator)
    {}

    EthosNWorkloadFactory(const EthosNConfig& config,
                          const std::string deviceId,
                          const std::shared_ptr<ethosn::driver_library::ProcMemAllocator> procMemAllocator)
        : m_EthosNConfig(config)
        , m_DeviceId(deviceId)
        , m_ProcMemAllocator(procMemAllocator)
    {}

    const BackendId& GetBackendId() const override;

    bool SupportsSubTensors() const override
    {
        return false;
    }

    std::unique_ptr<ITensorHandle> CreateSubTensorHandle(ITensorHandle& parent,
                                                         TensorShape const& subTensorShape,
                                                         unsigned int const* subTensorOrigin) const override;

    std::unique_ptr<IWorkload> CreateInput(const InputQueueDescriptor& descriptor,
                                           const WorkloadInfo& info) const override;

    std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo,
                                                      const bool IsMemoryManaged = true) const override;

    std::unique_ptr<ITensorHandle> CreateTensorHandle(const TensorInfo& tensorInfo,
                                                      DataLayout dataLayout,
                                                      const bool IsMemoryManaged = true) const override;

    std::unique_ptr<IWorkload> CreateOutput(const OutputQueueDescriptor& descriptor,
                                            const WorkloadInfo& info) const override;

    std::unique_ptr<IWorkload> CreateMemCopy(const MemCopyQueueDescriptor& descriptor,
                                             const WorkloadInfo& info) const override;

    std::unique_ptr<IWorkload> CreatePreCompiled(const PreCompiledQueueDescriptor& descriptor,
                                                 const WorkloadInfo& info) const override;

    std::string GetDeviceId() const
    {
        return m_DeviceId;
    }

private:
    template <typename Workload, typename QueueDescriptorType, typename... Args>
    static std::unique_ptr<IWorkload>
        MakeWorkload(const QueueDescriptorType& descriptor, const WorkloadInfo& info, Args&&... args);

    EthosNConfig m_EthosNConfig;
    std::string m_DeviceId;
    std::shared_ptr<ethosn::driver_library::ProcMemAllocator> m_ProcMemAllocator;
};

}    //namespace armnn
