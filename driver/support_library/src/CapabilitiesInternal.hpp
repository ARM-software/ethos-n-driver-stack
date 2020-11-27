//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../include/ethosn_support_library/Support.hpp"
#include "Capabilities.hpp"

namespace ethosn
{
namespace support_library
{

FirmwareAndHardwareCapabilities GetEthosN78FwHwCapabilities(EthosNVariant variant, uint32_t sramSize);
FirmwareAndHardwareCapabilities GetEthosN77FwHwCapabilities();
FirmwareAndHardwareCapabilities GetEthosN57FwHwCapabilities();
FirmwareAndHardwareCapabilities GetEthosN37FwHwCapabilities();

/// Checks that capabilities vector is valid
///
/// @exception Throws VersionMismatchException if version or size missmatch
void ValidateCapabilities(const std::vector<char>& rawCaps);

/// Validates capabilities and get a FirmwareAndHardwareCapabilities object from raw vector
///
/// @returns FirmwareAndHardwareCapabilities
/// @exception Throws VersionMismatchException if version or size missmatch
FirmwareAndHardwareCapabilities GetValidCapabilities(const std::vector<char>& rawCaps);

}    // namespace support_library
}    // namespace ethosn
