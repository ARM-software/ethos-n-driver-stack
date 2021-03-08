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
    const Version version = GetLibraryVersion();
    REQUIRE(version.ToString() == "1.0.0");
}
