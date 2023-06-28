//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Buffer.hpp"
#include "Inference.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Version information
#define ETHOSN_DRIVER_LIBRARY_VERSION_MAJOR 7
#define ETHOSN_DRIVER_LIBRARY_VERSION_MINOR 1
#define ETHOSN_DRIVER_LIBRARY_VERSION_PATCH 0

namespace ethosn
{
namespace driver_library
{

class NetworkImpl;

class ProcMemAllocator;

struct Version
{
    Version();

    constexpr Version(const uint32_t Major, const uint32_t Minor, const uint32_t Patch)
        : Major(Major)
        , Minor(Minor)
        , Patch(Patch)
    {}

    bool operator==(const Version& ver) const
    {
        return Major == ver.Major && Minor == ver.Minor && Patch == ver.Patch;
    }

    uint32_t Major;
    uint32_t Minor;
    uint32_t Patch;
};

const Version GetLibraryVersion();

/// Gets an opaque block of data representing the capabilities of the firmware and hardware.
/// This data should be passed to the Support Library (in its CompilationOptions constructor)
/// to provide details of what features of the hardware it should compile for.
std::vector<char> GetFirmwareAndHardwareCapabilities();
std::vector<char> GetFirmwareAndHardwareCapabilities(const std::string& device);

/// Exception type thrown when there is a problem with a compiled network passed to the API.
class CompiledNetworkException : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

// A single network, loaded and ready to execute inferences.
class Network
{
public:
    Network(Network&& otherNetwork);

    ~Network();

    // Schedule an inference with the network and the input & output buffers supplied.
    // Returns a Inference object.
    // The order of inputs/outputs correspond exactly to that in the compiled network.
    Inference* ScheduleInference(Buffer* const inputBuffers[],
                                 uint32_t numInputBuffers,
                                 Buffer* const outputBuffers[],
                                 uint32_t numOutputBuffers) const;

    void SetDebugName(const char* name);

private:
    friend ProcMemAllocator;
    Network(std::unique_ptr<NetworkImpl> otherNetworkImpl);

    std::unique_ptr<NetworkImpl> m_NetworkImpl;
};

bool VerifyKernel();
bool VerifyKernel(const std::string& device);

}    // namespace driver_library
}    // namespace ethosn
