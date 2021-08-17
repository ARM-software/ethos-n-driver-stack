//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "NetworkImpl.hpp"

namespace ethosn
{
namespace driver_library
{

#define MAX_ETHOSN_KERNEL_MODULE_MAJOR_VERSION_SUPPORTED 1
#define MIN_ETHOSN_KERNEL_MODULE_MAJOR_VERSION_SUPPORTED 1

class KmodNetworkImpl : public NetworkImpl
{
public:
    KmodNetworkImpl(const char* compiledNetworkData, size_t compiledNetworkSize, const std::string& device);

    ~KmodNetworkImpl() override;

    Inference* ScheduleInference(Buffer* const inputBuffers[],
                                 uint32_t numInputBuffers,
                                 Buffer* const outputBuffers[],
                                 uint32_t numOutputBuffers) const override;

private:
    void DumpIntermediateBuffers();

    int m_NetworkFd;
};

bool IsKernelVersionMatching(const struct Version& ver);
constexpr bool IsKernelVersionSupported(const uint32_t& majorVersion);

}    // namespace driver_library
}    // namespace ethosn
