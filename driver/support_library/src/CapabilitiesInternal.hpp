//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Capabilities.hpp"

namespace ethosn
{
namespace support_library
{

FirmwareAndHardwareCapabilities GetEthosN77FwHwCapabilities();
FirmwareAndHardwareCapabilities GetEthosN57FwHwCapabilities();
FirmwareAndHardwareCapabilities GetEthosN37FwHwCapabilities();

}    // namespace support_library
}    // namespace ethosn
