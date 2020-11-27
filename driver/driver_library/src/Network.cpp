//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_driver_library/Network.hpp"

#include "NetworkImpl.hpp"
#ifdef TARGET_MODEL
#include "ModelNetwork.hpp"
#elif defined(TARGET_KMOD)
#include "KmodNetwork.hpp"
#endif

namespace ethosn
{
namespace driver_library
{

const Version GetLibraryVersion()
{
    return Version(ETHOSN_DRIVER_LIBRARY_VERSION_MAJOR, ETHOSN_DRIVER_LIBRARY_VERSION_MINOR,
                   ETHOSN_DRIVER_LIBRARY_VERSION_PATCH);
}

Version::Version()
    : Major(0)
    , Minor(0)
    , Patch(0)
{}

Version::Version(const uint32_t Major, const uint32_t Minor, const uint32_t Patch)
    : Major(Major)
    , Minor(Minor)
    , Patch(Patch)
{}

Network::Network(support_library::CompiledNetwork& compiledNetwork)
    : m_NetworkImpl(
#if defined(TARGET_MODEL)
          std::make_unique<ModelNetworkImpl>(compiledNetwork)
#elif defined(TARGET_KMOD)
          std::make_unique<KmodNetworkImpl>(compiledNetwork)
#elif defined(TARGET_DUMPONLY)
          std::make_unique<NetworkImpl>(compiledNetwork)
#else
#error "Unknown target backend."
#endif
      )
{}

Network::~Network() = default;

Inference* Network::ScheduleInference(Buffer* const inputBuffers[],
                                      uint32_t numInputBuffers,
                                      Buffer* const outputBuffers[],
                                      uint32_t numOutputBuffers) const
{
    return m_NetworkImpl->ScheduleInference(inputBuffers, numInputBuffers, outputBuffers, numOutputBuffers);
}

void Network::SetDebugName(const char* name)
{
    m_NetworkImpl->SetDebugName(name);
}

}    // namespace driver_library
}    // namespace ethosn
