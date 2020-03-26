//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
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
    KmodNetworkImpl(support_library::CompiledNetwork& compiledNetwork);

    ~KmodNetworkImpl() override;

    Inference* ScheduleInference(Buffer* const inputBuffers[],
                                 uint32_t numInputBuffers,
                                 Buffer* const outputBuffers[],
                                 uint32_t numOutputBuffers) const override;

private:
    int m_NetworkFd;
};

}    // namespace driver_library
}    // namespace ethosn
