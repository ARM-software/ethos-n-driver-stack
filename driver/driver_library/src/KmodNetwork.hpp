//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "NetworkImpl.hpp"

namespace ethosn
{
namespace driver_library
{

#define MAX_ETHOSN_KERNEL_MODULE_MAJOR_VERSION_SUPPORTED 6
#define MIN_ETHOSN_KERNEL_MODULE_MAJOR_VERSION_SUPPORTED 6

class KmodNetworkImpl : public NetworkImpl
{
public:
    KmodNetworkImpl(const char* compiledNetworkData,
                    size_t compiledNetworkSize,
                    int allocatorFd,
                    const IntermediateBufferReq& desc);

    ~KmodNetworkImpl() override;

    Inference* ScheduleInference(Buffer* const inputBuffers[],
                                 uint32_t numInputBuffers,
                                 Buffer* const outputBuffers[],
                                 uint32_t numOutputBuffers) override;

protected:
    std::pair<const char*, size_t> MapIntermediateBuffers() override;
    void UnmapIntermediateBuffers(std::pair<const char*, size_t> mappedPtr) override;

private:
    int m_NetworkFd;
    int m_IntermediateBufferFd;
};

bool IsKernelVersionMatching(const struct Version& ver);
bool IsKernelVersionMatching(const struct Version& ver, const std::string& device);
constexpr bool IsKernelVersionSupported(const uint32_t& majorVersion);

}    // namespace driver_library
}    // namespace ethosn
