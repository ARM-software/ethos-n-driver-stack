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

class KmodNetworkImpl : public NetworkImpl
{
public:
    KmodNetworkImpl(const char* compiledNetworkData, size_t compiledNetworkSize);

    ~KmodNetworkImpl() override;

    Inference* ScheduleInference(Buffer* const inputBuffers[],
                                 uint32_t numInputBuffers,
                                 Buffer* const outputBuffers[],
                                 uint32_t numOutputBuffers) const override;

private:
    void DumpIntermediateBuffers();

    int m_NetworkFd;
};

}    // namespace driver_library
}    // namespace ethosn
