//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

// This file implements some of internal profiling functions with a null implementation (that does nothing).
// This is for the model and other backends that do not have access to a kernel module.
// These functions are declared in ProfilingInternal.hpp.

#include "ProfilingInternal.hpp"

namespace ethosn
{
namespace driver_library
{
namespace profiling
{

bool ConfigureKernelDriver(Configuration)
{
    return true;
}

uint64_t GetKernelDriverCounterValue(PollCounterName)
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
