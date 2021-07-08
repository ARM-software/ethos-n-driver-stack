//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../include/ethosn_support_library/Support.hpp"
#include "Capabilities.hpp"

namespace ethosn
{
namespace support_library
{

FirmwareAndHardwareCapabilities GetEthosN78FwHwCapabilities(EthosNVariant variant, uint32_t sramSize = 0);

/// Checks that capabilities vector is valid
///
/// @exception Throws VersionMismatchException if version or size mismatch
void ValidateCapabilities(const std::vector<char>& rawCaps);

/// Validates capabilities and get a FirmwareAndHardwareCapabilities object from raw vector
///
/// @exception Throws VersionMismatchException if version or size mismatch
FirmwareAndHardwareCapabilities GetValidCapabilities(const std::vector<char>& rawCaps);

/// Runtime check that command stream version is within range of given capabilities
bool IsCommandStreamInRange(const FirmwareAndHardwareCapabilities& caps, const uint32_t& major, const uint32_t& minor);

/// Checks that command stream version is supported
bool VerifySupportedCommandStream(const FirmwareAndHardwareCapabilities& caps);

/// Checks if the given FirmwareAndHardwareCapabilities are supported by the support library.
bool AreCapabilitiesSupported(const FirmwareAndHardwareCapabilities& caps);

}    // namespace support_library
}    // namespace ethosn
