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

std::unique_ptr<IWorkload> EthosNWorkloadFactory::CreateWorkload(LayerType type,
                                                                 const QueueDescriptor& descriptor,
                                                                 const WorkloadInfo& info) const
{
    switch (type)
    {
        case LayerType::Input:
        {
            auto inputQueueDescriptor = PolymorphicDowncast<const InputQueueDescriptor*>(&descriptor);
            return std::make_unique<CopyMemGenericWorkload>(*inputQueueDescriptor, info);
        }
        case LayerType::MemCopy:
        {
            auto memCopyQueueDescriptor = PolymorphicDowncast<const MemCopyQueueDescriptor*>(&descriptor);
            return std::make_unique<CopyMemGenericWorkload>(*memCopyQueueDescriptor, info);
        }
        case LayerType::Output:
        {
            auto outputQueueDescriptor = PolymorphicDowncast<const OutputQueueDescriptor*>(&descriptor);
            return std::make_unique<CopyMemGenericWorkload>(*outputQueueDescriptor, info);
        }
        case LayerType::PreCompiled:
        {
            auto preCompiledQueueDescriptor = PolymorphicDowncast<const PreCompiledQueueDescriptor*>(&descriptor);
            return std::make_unique<EthosNPreCompiledWorkload>(*preCompiledQueueDescriptor, info, m_DeviceId,
                                                               m_InternalAllocator);
        }
        default:
        {
            return nullptr;
        }
    }
}

}    // namespace armnn
