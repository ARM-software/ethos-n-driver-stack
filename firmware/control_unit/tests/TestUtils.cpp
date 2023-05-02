//
// Copyright © 2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "TestUtils.hpp"

#include <common/Utils.hpp>

#include <catch.hpp>

#include <cstring>

namespace ethosn
{
namespace control_unit
{
namespace tests
{

#ifdef CONTROL_UNIT_ASSERTS
void RequireFatalCall(std::function<void()> testFunc)
{
    // Verify the fault call by replacing the asset callback
    auto assertCallbackBackup                     = ethosn::control_unit::utils::g_AssertCallback;
    ethosn::control_unit::utils::g_AssertCallback = [](const char*, const char* file, unsigned line, const char*,
                                                       const char* fmt, va_list) {
        // All fatal messages should start with "ERROR:"
        if (!strncmp(fmt, "ERROR:", 6))
        {
            throw std::runtime_error("Fatal called");
        }
        else
        {
            throw std::runtime_error(std::string("Unknown assert in ") + file + ":" + std::to_string(line) + " " + fmt);
        }
    };

    CHECK_THROWS_WITH(testFunc(), "Fatal called");

    ethosn::control_unit::utils::g_AssertCallback = assertCallbackBackup;
}
#endif    // CONTROL_UNIT_ASSERTS

}    // namespace tests
}    // namespace control_unit
}    // namespace ethosn
