//
// Copyright © 2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstring>
#include <stdexcept>

#include <dirent.h>
#include <fstream>
#include <stdio.h>
#include <sys/types.h>
#include <sys/utsname.h>

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
    std::ifstream fileStream(filePath + "/core0/iommus");
    return fileStream.is_open();
}

inline bool IsKernelVersionHigherOrEqualTo(int kernelVersion, int kernelPatchLevel)
{
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
}

inline bool IsNpuCoreBehindIommus()
{
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
}

}    // namespace utils
}    // namespace ethosn
