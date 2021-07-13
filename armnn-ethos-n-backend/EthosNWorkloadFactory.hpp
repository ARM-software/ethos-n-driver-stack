//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "EthosNConfig.hpp"
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

// Ethos-N workload factory
class EthosNWorkloadFactory : public IWorkloadFactory
{
public:
    EthosNWorkloadFactory(const EthosNConfig& config)
        : m_EthosNConfig(config)
    {}

    const BackendId& GetBackendId() const override;

    static bool IsLayerSupported(const Layer& layer, Optional<DataType> dataType, std::string& outReasonIfUnsupported);

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

private:
    template <typename Workload, typename QueueDescriptorType, typename... Args>
    static std::unique_ptr<IWorkload>
        MakeWorkload(const QueueDescriptorType& descriptor, const WorkloadInfo& info, Args&&... args);

    EthosNConfig m_EthosNConfig;
};

}    //namespace armnn
