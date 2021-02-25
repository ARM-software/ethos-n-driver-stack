//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Buffer.hpp"
#include "Inference.hpp"

#include <ethosn_support_library/Support.hpp>

// Version information
#define ETHOSN_DRIVER_LIBRARY_VERSION_MAJOR 0
#define ETHOSN_DRIVER_LIBRARY_VERSION_MINOR 1
#define ETHOSN_DRIVER_LIBRARY_VERSION_PATCH 2

namespace ethosn
{
namespace driver_library
{

class NetworkImpl;

struct Version
{
    Version();
    Version(const uint32_t Major, const uint32_t Minor, const uint32_t Patch);

    uint32_t Major;
    uint32_t Minor;
    uint32_t Patch;
};

const Version GetLibraryVersion();

/// Gets an opaque block of data representing the capabilities of the firmware and hardware.
/// This data should be passed to the Support Library (in its CompilationOptions constructor)
/// to provide details of what features of the hardware it should compile for.
std::vector<char> GetFirmwareAndHardwareCapabilities();

// The Network class maintains references to the command stream, ple kernels & weights.
class Network
{
public:
    Network(support_library::CompiledNetwork&);

    ~Network();

    // Schedule an inference with the network and the input & output buffers supplied.
    // Returns a Inference object.
    Inference* ScheduleInference(Buffer* const inputBuffers[],
                                 uint32_t numInputBuffers,
                                 Buffer* const outputBuffers[],
                                 uint32_t numOutputBuffers) const;

    void SetDebugName(const char* name);

private:
    std::unique_ptr<NetworkImpl> m_NetworkImpl;
};
}    // namespace driver_library
}    // namespace ethosn
