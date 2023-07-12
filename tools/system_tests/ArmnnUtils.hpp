//
// Copyright © 2018-2020,2022-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "SystemTestsUtils.hpp"

#include <armnn/ArmNN.hpp>

namespace ethosn
{
namespace system_tests
{

void ConfigureArmnnLogging();

InferenceOutputs ArmnnRunNetwork(armnn::INetwork* network,
                                 const std::vector<armnn::BackendId>& devices,
                                 const std::vector<armnn::LayerBindingId>& inputBindings,
                                 const std::vector<armnn::LayerBindingId>& outputBindings,
                                 const InferenceInputs& inputData,
                                 const std::vector<armnn::BackendOptions>& backendOptions,
                                 const char* dmaBufHeapDevFilename,
                                 bool runProtected,
                                 size_t numInferences);

/// Creates a new heap-allocated tensor with size and data type matching the given Arm NN description.
OwnedTensor MakeTensor(const armnn::TensorInfo& t);

class CustomAllocator : public armnn::ICustomAllocator
{
public:
    CustomAllocator()
        : m_DmaBufHeap(new DmaBufferDevice("/dev/dma_heap/system"))
    {}

    void* allocate(size_t size, size_t alignment);

    void free(void* ptr);

    armnn::MemorySource GetMemorySourceType();

private:
    std::unique_ptr<DmaBufferDevice> m_DmaBufHeap;
    struct m_MapStruct
    {
        std::unique_ptr<DmaBuffer> m_DataDmaBuf;
        int m_Fd;
    };
    std::map<int, m_MapStruct> m_Map;
};

}    // namespace system_tests
}    // namespace ethosn
