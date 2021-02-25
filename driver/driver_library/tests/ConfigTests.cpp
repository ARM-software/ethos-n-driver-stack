//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_driver_library/Profiling.hpp"
#include "../src/ProfilingInternal.hpp"

#include <catch.hpp>

using namespace ethosn::driver_library;

TEST_CASE("GetConfigFromString hardware counters")
{
    using namespace profiling;
    auto configString = "hwCounters=busAccessRdTransfers,busReadBeats";

    auto config = GetConfigFromString(configString);

    REQUIRE(config.m_EnableProfiling == true);
    REQUIRE(config.m_NumHardwareCounters == 2);
    REQUIRE(config.m_HardwareCounters[0] == HardwareCounters::FirmwareBusAccessRdTransfers);
    REQUIRE(config.m_HardwareCounters[1] == HardwareCounters::FirmwareBusReadBeats);
}

TEST_CASE("GetConfigFromString hardware counters > 6")
{
    using namespace profiling;
    auto configString = "hwCounters=busAccessRdTransfers,busReadBeats,busReadTxfrStallCycles,busAccessWrTransfers,"
                        "busWrCompleteTransfers,busWriteBeats,busWriteTxfrStallCycles,busWriteStallCycles";

    auto config = GetConfigFromString(configString);

    REQUIRE(config.m_EnableProfiling == true);
    REQUIRE(config.m_NumHardwareCounters == 0);
}
