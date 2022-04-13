//
// Copyright © 2018-2022 Arm Limited.
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
        caps.m_Header.m_Version = FW_AND_HW_CAPABILITIES_VERSION + 10;
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
        std::vector<char> capsFromSupportLibVect = GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO, 0);

        FirmwareAndHardwareCapabilities capsFromOptsSL;
        memcpy(&capsFromOptsSL, capsFromSupportLibVect.data(), sizeof(FirmwareAndHardwareCapabilities));

        REQUIRE(VerifySupportedCommandStream(capsFromOptsSL) == true);
    }
}

TEST_CASE("Correct capabilities")
{
    std::vector<char> capsFromSupportLibVect = GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO, 0);

    FirmwareAndHardwareCapabilities capsFromOptsSL;
    memcpy(&capsFromOptsSL, capsFromSupportLibVect.data(), sizeof(FirmwareAndHardwareCapabilities));

    // Test some random values after being turned from vector into FirmwareAndHardwareCapabilities again.
    REQUIRE(capsFromOptsSL.m_Header.m_Version == FW_AND_HW_CAPABILITIES_VERSION);
    REQUIRE(capsFromOptsSL.m_TotalSramSize == 1048576);
    REQUIRE(capsFromOptsSL.m_TotalAccumulatorsPerOg == 512);
}

TEST_CASE("Capabilities different variant")
{
    uint32_t eightTopsSizeSram = 2048 * 1024;

    std::vector<char> capsFromSupportLibVect = GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_8TOPS_2PLE_RATIO, 0);

    FirmwareAndHardwareCapabilities capsFromOptsSL;
    memcpy(&capsFromOptsSL, capsFromSupportLibVect.data(), sizeof(FirmwareAndHardwareCapabilities));

    // Test some random values after being turned from vector into FirmwareAndHardwareCapabilities again.
    REQUIRE(capsFromOptsSL.m_TotalSramSize == eightTopsSizeSram);
}

TEST_CASE("Capabilities different SRAM size")
{
    uint32_t overrideSramSize = 2048 * 1024;
    std::vector<char> capsFromSupportLibVect =
        GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO, overrideSramSize);

    FirmwareAndHardwareCapabilities capsFromOptsSL;
    memcpy(&capsFromOptsSL, capsFromSupportLibVect.data(), sizeof(FirmwareAndHardwareCapabilities));

    // Test some random values after being turned from vector into FirmwareAndHardwareCapabilities again.
    REQUIRE(capsFromOptsSL.m_TotalSramSize == overrideSramSize);
}

TEST_CASE("GetFwAndHwCapabilities unsupported")
{
    REQUIRE_THROWS_AS(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N77), NotSupportedException);
}

TEST_CASE("GetFwAndHwCapabilities with unsupported SRAM sizes")
{
    // SRAM too small
    REQUIRE_THROWS_WITH(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO, 2048),
                        "User configured SRAM size is smaller than the minimum allowed for this variant");
    // SRAM too large
    REQUIRE_THROWS_WITH(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO, 16 * 512 * 1024),
                        "User configured SRAM size is larger than the maximum allowed for this variant");
    // SRAM not a multiple of 16
    REQUIRE_THROWS_WITH(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO, 16 * 33 * 1024),
                        "User configured SRAM size per Emc is not a multiple of 16");
}

TEST_CASE("GetFwAndHwCapabilities with supported SRAM sizes")
{
    // Check edge of supported range
    REQUIRE_NOTHROW(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO, 16 * 32 * 1024));
    REQUIRE_NOTHROW(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO, 16 * 128 * 1024));

    // Exception should not be thrown on additional min and max SRAM size
    REQUIRE_NOTHROW(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO, 16 * 56 * 1024));
    REQUIRE_NOTHROW(GetFwAndHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO, 16 * 256 * 1024));
}
