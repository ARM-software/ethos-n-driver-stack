//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_driver_library/Network.hpp"

#include <catch.hpp>

using namespace ethosn::driver_library;

TEST_CASE("TestLibraryVersion")
{
    // Simple test to ensure the library can be loaded.
    const Version version = GetLibraryVersion();
    REQUIRE(version.Major == ETHOSN_DRIVER_LIBRARY_VERSION_MAJOR);
    REQUIRE(version.Minor == ETHOSN_DRIVER_LIBRARY_VERSION_MINOR);
    REQUIRE(version.Patch == ETHOSN_DRIVER_LIBRARY_VERSION_PATCH);
}

TEST_CASE("GetFirmwareAndHardwareCapabilities")
{
    std::vector<char> capsRaw = GetFirmwareAndHardwareCapabilities();
    REQUIRE(capsRaw.size() > 0);
}
