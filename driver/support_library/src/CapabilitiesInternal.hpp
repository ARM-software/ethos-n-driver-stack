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
FirmwareAndHardwareCapabilities GetEthosN77FwHwCapabilities();

/// Checks that capabilities vector is valid
///
/// @exception Throws VersionMismatchException if version or size missmatch
void ValidateCapabilities(const std::vector<char>& rawCaps);

/// Validates capabilities and get a FirmwareAndHardwareCapabilities object from raw vector
///
/// @returns FirmwareAndHardwareCapabilities
/// @exception Throws VersionMismatchException if version or size missmatch
FirmwareAndHardwareCapabilities GetValidCapabilities(const std::vector<char>& rawCaps);

/// Runtime check that command stream version is within range of given capabilities
///
/// @returns bool
bool IsCommandStreamInRange(const FirmwareAndHardwareCapabilities& caps, const uint32_t& major, const uint32_t& minor);

/// Checks that command stream version is suported
///
/// @returns bool
bool VerifySupportedCommandStream(const FirmwareAndHardwareCapabilities& caps);

}    // namespace support_library
}    // namespace ethosn
