//
// Copyright © 2018-2021 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"
#include "Capabilities.hpp"
#include "CapabilitiesInternal.hpp"

#include <catch.hpp>

#include <cstring>

using namespace ethosn::support_library;

std::vector<char> GetRawCapabilities(const FirmwareAndHardwareCapabilities& caps)
{
    return std::vector<char>(reinterpret_cast<const char*>(&caps), reinterpret_cast<const char*>(&caps) + sizeof(caps));
}

TEST_CASE("Invalid capabilities")
{
    SECTION("Capabilities data is too short for header")
    {
        REQUIRE_THROWS_AS(CreateNetwork({}), VersionMismatchException);
    }

    SECTION("Capabilities data has an unsupported size in the header")
    {
        FirmwareAndHardwareCapabilities caps;
        caps.m_Header.m_Size    = 1234;
        caps.m_Header.m_Version = FW_AND_HW_CAPABILITIES_VERSION;
        REQUIRE_THROWS_AS(CreateNetwork(GetRawCapabilities(caps)), VersionMismatchException);
    }

    SECTION("Capabilities data has an unsupported version in the header")
    {
        FirmwareAndHardwareCapabilities caps;
        caps.m_Header.m_Size    = sizeof(FirmwareAndHardwareCapabilities);
        caps.m_Header.m_Version = 17;
        REQUIRE_THROWS_AS(CreateNetwork(GetRawCapabilities(caps)), VersionMismatchException);
    }
}

TEST_CASE("Command stream compatibility")
{
    SECTION("Major version below range")
    {
        FirmwareAndHardwareCapabilities caps;
        caps.m_CommandStreamBeginRangeMajor = 5;
        caps.m_CommandStreamEndRangeMajor   = 10;
        caps.m_CommandStreamBeginRangeMinor = 5;
        caps.m_CommandStreamEndRangeMinor   = 10;

        REQUIRE(IsCommandStreamInRange(caps, 0, 5) == false);
    }

    SECTION("Major version at start of range, minor out of range")
    {
        FirmwareAndHardwareCapabilities caps;
        caps.m_CommandStreamBeginRangeMajor = 5;
        caps.m_CommandStreamEndRangeMajor   = 10;
        caps.m_CommandStreamBeginRangeMinor = 5;
        caps.m_CommandStreamEndRangeMinor   = 10;

        REQUIRE(IsCommandStreamInRange(caps, 5, 0) == false);
    }

    SECTION("Major version at start of range, minor in range")
    {
        FirmwareAndHardwareCapabilities caps;
        caps.m_CommandStreamBeginRangeMajor = 5;
        caps.m_CommandStreamEndRangeMajor   = 10;
        caps.m_CommandStreamBeginRangeMinor = 5;
        caps.m_CommandStreamEndRangeMinor   = 10;

        REQUIRE(IsCommandStreamInRange(caps, 5, 7) == true);
    }

    SECTION("Major version within range, but not start or end")
    {
        FirmwareAndHardwareCapabilities caps;
        caps.m_CommandStreamBeginRangeMajor = 5;
        caps.m_CommandStreamEndRangeMajor   = 10;
        caps.m_CommandStreamBeginRangeMinor = 5;
        caps.m_CommandStreamEndRangeMinor   = 10;

        REQUIRE(IsCommandStreamInRange(caps, 7, 0) == true);
    }

    SECTION("Major version at end of range, minor in range")
    {
        FirmwareAndHardwareCapabilities caps;
        caps.m_CommandStreamBeginRangeMajor = 5;
        caps.m_CommandStreamEndRangeMajor   = 10;
        caps.m_CommandStreamBeginRangeMinor = 5;
        caps.m_CommandStreamEndRangeMinor   = 10;

        REQUIRE(IsCommandStreamInRange(caps, 10, 5) == true);
    }

    SECTION("Major version at end of range, minor out of range")
    {
        FirmwareAndHardwareCapabilities caps;
        caps.m_CommandStreamBeginRangeMajor = 5;
        caps.m_CommandStreamEndRangeMajor   = 10;
        caps.m_CommandStreamBeginRangeMinor = 5;
        caps.m_CommandStreamEndRangeMinor   = 10;

        REQUIRE(IsCommandStreamInRange(caps, 10, 15) == false);
    }

    SECTION("Test with valid configuration")
    {
        std::vector<char> capsFromSupportLibVect = GetFwAndHwCapabilities(EthosNVariant::ETHOS_N77, 0);

        FirmwareAndHardwareCapabilities capsFromOptsSL;
        memcpy(&capsFromOptsSL, capsFromSupportLibVect.data(), sizeof(FirmwareAndHardwareCapabilities));

        REQUIRE(VerifySupportedCommandStream(capsFromOptsSL) == true);
    }
}

TEST_CASE("Correct capabilities")
{
    std::vector<char> capsFromSupportLibVect = GetFwAndHwCapabilities(EthosNVariant::ETHOS_N77, 0);

    FirmwareAndHardwareCapabilities capsFromOptsSL;
    memcpy(&capsFromOptsSL, capsFromSupportLibVect.data(), sizeof(FirmwareAndHardwareCapabilities));

    // Test some random values after being turned from vector into FirmwareAndHardwareCapabilities again.
    REQUIRE(capsFromOptsSL.m_Header.m_Version == 3);
    REQUIRE(capsFromOptsSL.m_TotalSramSize == 1048576);
    REQUIRE(capsFromOptsSL.m_TotalAccumulatorsPerOg == 512);
}

TEST_CASE("Capabilities different variant")
{
    uint32_t n57SizeSram = 524288;

    std::vector<char> capsFromSupportLibVect = GetFwAndHwCapabilities(EthosNVariant::ETHOS_N57, 0);

    FirmwareAndHardwareCapabilities capsFromOptsSL;
    memcpy(&capsFromOptsSL, capsFromSupportLibVect.data(), sizeof(FirmwareAndHardwareCapabilities));

    // Test some random values after being turned from vector into FirmwareAndHardwareCapabilities again.
    REQUIRE(capsFromOptsSL.m_TotalSramSize == n57SizeSram);
}

TEST_CASE("Capabilities different SRAM size")
{
    uint32_t n57SizeSram                     = 524288;
    std::vector<char> capsFromSupportLibVect = GetFwAndHwCapabilities(EthosNVariant::ETHOS_N77, n57SizeSram);

    FirmwareAndHardwareCapabilities capsFromOptsSL;
    memcpy(&capsFromOptsSL, capsFromSupportLibVect.data(), sizeof(FirmwareAndHardwareCapabilities));

    // Test some random values after being turned from vector into FirmwareAndHardwareCapabilities again.
    REQUIRE(capsFromOptsSL.m_TotalSramSize == n57SizeSram);
}
