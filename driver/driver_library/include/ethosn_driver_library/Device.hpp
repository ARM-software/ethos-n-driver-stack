//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <string>

namespace ethosn
{
namespace driver_library
{

// It is assumed that all device ids are consecutive.
// The device prefix and the device base identifier
// are compile time parameters, please refer to Scons
// files.
//
// For a system with the following devices:
//
// /dev/ethosn4
// /dev/ethosn5
// /dev/ethosn6
//
// GetDeviceNamePrefix returns /dev/ethosn
// GetDeviceBaseId returns 4
// GetNumberOfDevices returns 3
std::string GetDeviceNamePrefix();
uint16_t GetDeviceBaseId();
uint16_t GetNumberOfDevices();

}    // namespace driver_library
}    // namespace ethosn
