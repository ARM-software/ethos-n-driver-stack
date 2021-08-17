//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

// This file implements some of internal profiling functions with a null implementation (that does nothing).
// This is for the model and other backends that do not have access to a kernel module.
// These functions are declared in ProfilingInternal.hpp.

// Note that the model backend *can* report profiling entries, but due to the differences in the way the model backend
// works (it creates a temporary Firmware object just for the inference), there is no global state for these functions
// to affect, like there is for the kernel backend. Hence profiling is handled for the model backend inside
// ModelNetwork::ScheduleInference().

#include "ProfilingInternal.hpp"

namespace ethosn
{
namespace driver_library
{
namespace profiling
{

bool ConfigureKernelDriver(Configuration, const std::string&)
{
    return true;
}

uint64_t GetKernelDriverCounterValue(PollCounterName, const std::string&)
{
    return 0;
}

bool AppendKernelDriverEntries()
{
    return true;
}

}    // namespace profiling
}    // namespace driver_library

}    // namespace ethosn
