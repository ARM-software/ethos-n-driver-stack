//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "EthosNWorkloadFactory.hpp"

#include "EthosNBackendId.hpp"
#include "EthosNTensorHandle.hpp"
#include "EthosNWorkloads.hpp"

#include <armnn/backends/MemCopyWorkload.hpp>

namespace armnn
{

namespace
{
static const BackendId s_Id{ EthosNBackendId() };
}

const BackendId& EthosNWorkloadFactory::GetBackendId() const
{
    return s_Id;
}

std::unique_ptr<ITensorHandle>
    EthosNWorkloadFactory::CreateSubTensorHandle(ITensorHandle&, TensorShape const&, unsigned int const*) const
{
    assert(false);
    return nullptr;
}

std::unique_ptr<IWorkload> EthosNWorkloadFactory::CreateInput(const InputQueueDescriptor& descriptor,
                                                              const WorkloadInfo& info) const
{
    return std::make_unique<CopyMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<ITensorHandle>
    EthosNWorkloadFactory::CreateTensorHandle(const TensorInfo&, DataLayout, const bool) const
{
    // This API is now deprecated so it shouldn't be called.
    assert(false);
    return nullptr;
}

std::unique_ptr<ITensorHandle> EthosNWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo, const bool) const
{
    return CreateTensorHandle(tensorInfo, DataLayout::NHWC);
}

std::unique_ptr<IWorkload> EthosNWorkloadFactory::CreatePreCompiled(const PreCompiledQueueDescriptor& descriptor,
                                                                    const WorkloadInfo& info) const
{

    return std::make_unique<EthosNPreCompiledWorkload>(descriptor, info, m_DeviceId, m_InternalAllocator);
}

std::unique_ptr<IWorkload> EthosNWorkloadFactory::CreateOutput(const OutputQueueDescriptor& descriptor,
                                                               const WorkloadInfo& info) const
{
    return std::make_unique<CopyMemGenericWorkload>(descriptor, info);
}
std::unique_ptr<IWorkload> EthosNWorkloadFactory::CreateMemCopy(const MemCopyQueueDescriptor& descriptor,
                                                                const WorkloadInfo& info) const
{
    return std::make_unique<CopyMemGenericWorkload>(descriptor, info);
}

}    // namespace armnn
