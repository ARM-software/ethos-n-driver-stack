//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"

#include <catch.hpp>

using namespace ethosn::support_library;

TEST_CASE("AccessSupportLib")
{
    // Simple test to ensure the library can be loaded.
    const Version version  = GetLibraryVersion();
    const Version macroVer = Version(ETHOSN_SUPPORT_LIBRARY_VERSION_MAJOR, ETHOSN_SUPPORT_LIBRARY_VERSION_MINOR,
                                     ETHOSN_SUPPORT_LIBRARY_VERSION_PATCH);
    REQUIRE(version.ToString() == macroVer.ToString());
}
