//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_driver_library/Device.hpp"
#include "../include/ethosn_driver_library/Network.hpp"
#include "../src/KmodNetwork.hpp"

#include <uapi/ethosn.h>

#include <catch.hpp>

using namespace ethosn::driver_library;

TEST_CASE("TestVersionMismatch")
{
    GIVEN("A random version")
    {
        const struct Version ver(UINT32_MAX, UINT32_MAX, UINT32_MAX);

        WHEN("It is matched against the actual version on the system")
        {
            bool ret = IsKernelVersionMatching(ver);

            THEN("The match should return false")
            {
                REQUIRE(!ret);
            }
        }
    }
}

TEST_CASE("TestVersionMatch")
{
    GIVEN("Kernel version defined in ethosn.h")
    {
        const struct Version ver(ETHOSN_KERNEL_MODULE_VERSION_MAJOR, ETHOSN_KERNEL_MODULE_VERSION_MINOR,
                                 ETHOSN_KERNEL_MODULE_VERSION_PATCH);

        WHEN("It is matched against the actual version on the system")
        {
            bool ret = IsKernelVersionMatching(ver);

            THEN("The match should return true")
            {
                REQUIRE(ret);
            }
        }
    }
}

TEST_CASE("GetNumberOfDevices")
{
    REQUIRE(GetNumberOfDevices() >= 1U);
}
