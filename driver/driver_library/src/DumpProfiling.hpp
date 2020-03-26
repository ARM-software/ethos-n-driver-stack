//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../include/ethosn_driver_library/Profiling.hpp"

#include <ostream>
#include <vector>

namespace ethosn
{
namespace driver_library
{
namespace profiling
{

void DumpAllProfilingData(std::ostream& outStream);
void DumpProfilingData(const std::vector<ProfilingEntry>& profilingData, std::ostream& outStream);

}    // namespace profiling
}    // namespace driver_library
}    // namespace ethosn