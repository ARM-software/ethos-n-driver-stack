//
// Copyright © 2018-2020,2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#include "Utils.hpp"

namespace ethosn
{
namespace driver_library
{

#if !defined(NDEBUG)
constexpr const char DriverLibraryName[] = "driver_library";
LoggerType g_Logger({ &ethosn::utils::log::sinks::StdOut<DriverLibraryName> });
#else
LoggerType g_Logger;
#endif

}    // namespace driver_library
}    // namespace ethosn
