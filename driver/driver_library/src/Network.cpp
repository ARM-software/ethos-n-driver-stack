//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_driver_library/Network.hpp"

#include "NetworkImpl.hpp"
#include <ethosn_utils/Macros.hpp>
#ifdef TARGET_MODEL
#include "ModelNetwork.hpp"
#elif defined(TARGET_KMOD)
#include "KmodNetwork.hpp"
#endif

#include <algorithm>

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

Network::Network(Network&& otherNetwork)
    : m_NetworkImpl(std::move(otherNetwork.m_NetworkImpl))
{}

Network::Network(std::unique_ptr<NetworkImpl> otherNetworkImpl)
    : m_NetworkImpl(std::move(otherNetworkImpl))
{}

Network::~Network() = default;

Inference* Network::ScheduleInference(Buffer* const inputBuffers[],
                                      uint32_t numInputBuffers,
                                      Buffer* const outputBuffers[],
                                      uint32_t numOutputBuffers)
{
    return m_NetworkImpl->ScheduleInference(inputBuffers, numInputBuffers, outputBuffers, numOutputBuffers);
}

void Network::SetDebugName(const char* name)
{
    m_NetworkImpl->SetDebugName(name);
}

/// This verifies if the version of the kernel module is compatible or not.
/// The check is performed at compile time as well as at run time.
/// At compile time, it checks if the version defined in ethosn.h is supported
/// or not. The supported version is defined in KmodNetwork.hpp.
/// At run time, it checks if the version obtained from the kernel, matches
/// with the version defined in ethosn.h or not.
/// Returns True if there is a match else False.
bool VerifyKernel();
bool VerifyKernel(const std::string& device);

std::vector<char> GetFirmwareAndHardwareCapabilities()
{
    return GetFirmwareAndHardwareCapabilities(DEVICE_NODE);
}

}    // namespace driver_library
}    // namespace ethosn
