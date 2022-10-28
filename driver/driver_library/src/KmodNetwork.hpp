//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "NetworkImpl.hpp"

namespace ethosn
{
namespace driver_library
{

#define MAX_ETHOSN_KERNEL_MODULE_MAJOR_VERSION_SUPPORTED 4
#define MIN_ETHOSN_KERNEL_MODULE_MAJOR_VERSION_SUPPORTED 4

class KmodNetworkImpl : public NetworkImpl
{
public:
    KmodNetworkImpl(const char* compiledNetworkData, size_t compiledNetworkSize, const std::string& device);
    KmodNetworkImpl(const char* compiledNetworkData, size_t compiledNetworkSize, int allocatorFd);

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
bool IsKernelVersionMatching(const struct Version& ver, const std::string& device);
constexpr bool IsKernelVersionSupported(const uint32_t& majorVersion);

}    // namespace driver_library
}    // namespace ethosn
