//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "FixedString.hpp"

#include <ethosn_utils/Log.hpp>

namespace ethosn
{
namespace control_unit
{

#if defined(ETHOSN_LOGGING)
constexpr utils::log::Severity g_LogCompileTimeMaxSeverity = utils::log::Severity::Debug;
#else
constexpr utils::log::Severity g_LogCompileTimeMaxSeverity = utils::log::Severity::Info;
#endif

/// Declare the LoggingString type which will map to either a regular FixedString or a
/// dummy null-implementation version depending on if logging is enabled.
/// This allows code to use LoggingString unconditionally, knowing that it will be
/// disabled on builds without debug logging.
#if defined(ETHOSN_LOGGING)
using LoggingString = FixedString<1024>;
#else
using LoggingString                                        = NullFixedString;
#endif

using LoggerType = utils::log::Logger<g_LogCompileTimeMaxSeverity>;

void LogSink(ethosn::utils::log::Severity severity, const char* msg);

}    // namespace control_unit
}    // namespace ethosn
