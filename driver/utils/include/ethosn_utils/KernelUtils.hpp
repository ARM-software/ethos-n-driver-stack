//
// Copyright © 2022-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstring>
#include <fstream>
#include <stdexcept>

#if defined(__unix__)
#include <dirent.h>
#include <fstream>
#include <stdio.h>
#include <sys/types.h>
#include <sys/utsname.h>
#endif

namespace ethosn
{
namespace utils
{

inline bool IsDeviceStatusOkay(const std::string& filePath)
{
    std::ifstream fileStream(filePath);
    if (!fileStream.is_open())
    {
        return false;
    }
    std::string isOkay;
    getline(fileStream, isOkay);
    constexpr char okay[] = "okay";
    return isOkay.compare(0, strlen(okay), okay) == 0;
}

inline bool IsCore0IommuAvailable(const std::string& filePath)
{
    std::ifstream fileStream(filePath + "/core0/main_allocator/firmware/iommus");
    return fileStream.is_open();
}

inline bool IsKernelVersionHigherOrEqualTo(int kernelVersion, int kernelPatchLevel)
{
#if defined(__unix__)
    utsname linuxReleaseInfo = {};
    if (uname(&linuxReleaseInfo))
    {
        throw std::runtime_error("Util.cpp: IsKernelVersionHigherOrEqualTo() uname() failed");
    }
    int actualKernelVersion, actualKernelPatchLevel;
    if (sscanf(linuxReleaseInfo.release, "%d.%d", &actualKernelVersion, &actualKernelPatchLevel) != 2)
    {
        throw std::runtime_error("Util.cpp: IsKernelVersionHigherOrEqualTo() sscan for version from uname failed");
    }
    return ((kernelVersion < actualKernelVersion) ||
            ((kernelVersion == actualKernelVersion) && (kernelPatchLevel <= actualKernelPatchLevel)));
#else
    ETHOSN_UNUSED(kernelVersion);
    ETHOSN_UNUSED(kernelPatchLevel);
    throw std::runtime_error("Not supported on this platform");
    return false;
#endif
}

inline bool IsNpuCoreBehindIommus()
{
#if defined(__unix__)
    constexpr char deviceTreePath[] = "/proc/device-tree";
    DIR* dir                        = opendir(deviceTreePath);
    if (dir == nullptr)
    {
        return false;
    }
    dirent* ent;
    constexpr char deviceBindingPrefix[] = "ethosn@";
    while ((ent = readdir(dir)) != nullptr)
    {
        const std::string dirName(ent->d_name);
        if (dirName.find(deviceBindingPrefix, 0, strlen(deviceBindingPrefix)) == std::string::npos)
        {
            continue;
        }
        const std::string devicePath = deviceTreePath + std::string("/") + dirName;
        if (!IsDeviceStatusOkay(devicePath + "/status"))
        {
            continue;
        }
        if (!IsDeviceStatusOkay(devicePath + "/core0/status"))
        {
            continue;
        }
        if (IsCore0IommuAvailable(devicePath))
        {
            break;
        }
    }
    closedir(dir);
    return ent != nullptr;
#else
    throw std::runtime_error("Not supported on this platform");
    return false;
#endif
}

/// Checks if the system appears to be configured for TZMP1.
/// This doesn't necessarily mean that all the components in the driver stack and configured.
inline bool IsTzmp1Configured()
{
#if defined(__unix__)
    constexpr char reservedMemoryPath[] = "/proc/device-tree/reserved-memory/";
    DIR* dir                            = opendir(reservedMemoryPath);
    if (dir == nullptr)
    {
        return false;
    }
    dirent* ent;
    constexpr char deviceBindingPrefix[] = "ethosn_protected_reserved@";
    while ((ent = readdir(dir)) != nullptr)
    {
        const std::string dirName(ent->d_name);
        if (dirName.find(deviceBindingPrefix, 0, strlen(deviceBindingPrefix)) == std::string::npos)
        {
            continue;
        }
        const std::string devicePath = reservedMemoryPath + std::string("/") + dirName;
        if (!IsDeviceStatusOkay(devicePath + "/status"))
        {
            continue;
        }
    }
    closedir(dir);
    return ent != nullptr;
#else
    throw std::runtime_error("Not supported on this platform");
    return false;
#endif
}

}    // namespace utils
}    // namespace ethosn
