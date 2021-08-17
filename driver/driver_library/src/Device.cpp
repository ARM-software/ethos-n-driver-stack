//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_driver_library/Device.hpp"

#include <limits>
#if defined(__unix__)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace ethosn
{
namespace driver_library
{

uint16_t GetNumberOfDevices()
{
#ifdef TARGET_KMOD
    uint16_t i = 0;
    do
    {
        const std::string device = GetDeviceNamePrefix() + std::to_string(i + GetDeviceBaseId());
        if (access(device.c_str(), F_OK) != 0)
        {
            return i;
        }
        ++i;
    } while (i < std::numeric_limits<uint16_t>::max());
#else
    uint16_t i = 1;
#endif
    return i;
}

std::string GetDeviceNamePrefix()
{
    return DEVICE_NODE_PREFIX;
}

uint16_t GetDeviceBaseId()
{
    return DEVICE_NODE_BASE_ID;
}

}    // namespace driver_library
}    // namespace ethosn
