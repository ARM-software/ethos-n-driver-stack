//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "EthosNWorkloadFactory.hpp"

#include "EthosNBackendId.hpp"
#include "EthosNTensorHandle.hpp"
#include "EthosNWorkloads.hpp"

#include <Layer.hpp>
#include <armnn/Utils.hpp>
#include <backendsCommon/MakeWorkloadHelper.hpp>
#include <backendsCommon/MemCopyWorkload.hpp>

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

bool EthosNWorkloadFactory::IsLayerSupported(const Layer& layer,
                                             Optional<DataType> dataType,
                                             std::string& outReasonIfUnsupported)
{
    return IWorkloadFactory::IsLayerSupported(s_Id, layer, dataType, outReasonIfUnsupported);
}

std::unique_ptr<ITensorHandle>
    EthosNWorkloadFactory::CreateSubTensorHandle(ITensorHandle&, TensorShape const&, unsigned int const*) const
{
    return nullptr;
}

std::unique_ptr<IWorkload> EthosNWorkloadFactory::CreateInput(const InputQueueDescriptor& descriptor,
                                                              const WorkloadInfo& info) const
{
    return std::make_unique<CopyMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<ITensorHandle>
    EthosNWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo, DataLayout dataLayout, const bool) const
{
    // only NHWC format is supported
    if (dataLayout != DataLayout::NHWC)
    {
        return nullptr;
    }
    auto ethosnConfig = GetEthosNConfig();
    if (ethosnConfig.m_PerfOnly)
    {
        return std::make_unique<ScopedCpuTensorHandle>(tensorInfo);
    }
    return std::make_unique<EthosNTensorHandle>(tensorInfo);
}

std::unique_ptr<ITensorHandle> EthosNWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo, const bool) const
{
    return CreateTensorHandle(tensorInfo, DataLayout::NHWC);
}

std::unique_ptr<IWorkload> EthosNWorkloadFactory::CreatePreCompiled(const PreCompiledQueueDescriptor& descriptor,
                                                                    const WorkloadInfo& info) const
{
    return std::make_unique<EthosNPreCompiledWorkload>(descriptor, info);
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
