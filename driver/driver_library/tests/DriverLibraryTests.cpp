//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_driver_library/Network.hpp"
#include "../src/NetworkImpl.hpp"

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

TEST_CASE("DeserializeCompiledNetwork")
{
    GIVEN("A serialized compiled network")
    {
        // clang-format off
        std::vector<uint8_t> serialized = {
            // 0: FourCC
            'E', 'N', 'C', 'N',

            // 4: Version (Major)
            1, 0, 0, 0,
            // 8: Version (Minor)
            0, 0, 0, 0,
            // 12: Version (Patch)
            0, 0, 0, 0,

            // 16: Constant DMA data (size)
            3, 0, 0, 0,
            // 20: Constant DMA data (values)
            1, 2, 3,

            // 23: Constant control unit data (size)
            2, 0, 0, 0,
            // 27: Constant control unit data (values)
            4, 5,

            // Input buffer infos (size)
            1, 0, 0, 0,
            // Input buffer info 0
            10, 0, 0, 0, 11, 0, 0, 0, 12, 0, 0, 0,

            // Output buffer infos (size)
            2, 0, 0, 0,
            // Output buffer info 0
            20, 0, 0, 0, 21, 0, 0, 0, 22, 0, 0, 0,
            // Output buffer info 1
            21, 0, 0, 0, 23, 0, 0, 0, 24, 0, 0, 0,

            // Constant control unit data buffer infos (size)
            1, 0, 0, 0,
            // Constant control unit data buffer info 0
            30, 0, 0, 0, 31, 0, 0, 0, 32, 0, 0, 0,

            // Constant DMA data buffer infos (size)
            1, 0, 0, 0,
            // Constant DMA data buffer info 0
            40, 0, 0, 0, 41, 0, 0, 0, 42, 0, 0, 0,

            // Intermediate data buffer infos (size)
            1, 0, 0, 0,
            // Intermediate data buffer info 0
            50, 0, 0, 0, 51, 0, 0, 0, 52, 0, 0, 0,
        };
        // clang-format on

        WHEN("Calling DeserializeCompiledNetwork")
        {
            CompiledNetworkInfo compiledNetwork =
                DeserializeCompiledNetwork(reinterpret_cast<const char*>(serialized.data()), serialized.size());

            THEN("The result is as expected")
            {
                REQUIRE(compiledNetwork.m_ConstantDmaDataOffset == 20);
                REQUIRE(compiledNetwork.m_ConstantDmaDataSize == 3);

                REQUIRE(compiledNetwork.m_ConstantControlUnitDataOffset == 27);
                REQUIRE(compiledNetwork.m_ConstantControlUnitDataSize == 2);

                REQUIRE(compiledNetwork.m_InputBufferInfos == std::vector<BufferInfo>{ { 10, 11, 12 } });
                REQUIRE(compiledNetwork.m_OutputBufferInfos ==
                        std::vector<BufferInfo>{ { 20, 21, 22 }, { 21, 23, 24 } });
                REQUIRE(compiledNetwork.m_ConstantControlUnitDataBufferInfos ==
                        std::vector<BufferInfo>{ { 30, 31, 32 } });
                REQUIRE(compiledNetwork.m_ConstantDmaDataBufferInfos == std::vector<BufferInfo>{ { 40, 41, 42 } });
                REQUIRE(compiledNetwork.m_IntermediateDataBufferInfos == std::vector<BufferInfo>{ { 50, 51, 52 } });

                REQUIRE(compiledNetwork.m_IntermediateDataSize == 103);
            }
        }
    }
}

TEST_CASE("DeserializeCompiledNetwork Errors")
{
    using Catch::Matchers::Message;

    // clang-format off
    std::vector<uint8_t> serializedValid = {
        // 0: FourCC
        'E', 'N', 'C', 'N',

        // 4: Version (Major)
        1, 0, 0, 0,
        // 8: Version (Minor)
        0, 0, 0, 0,
        // 12: Version (Patch)
        0, 0, 0, 0,

        // 16: Constant DMA data (size)
        3, 0, 0, 0,
        // 20: Constant DMA data (values)
        1, 2, 3,

        // 23: Constant control unit data (size)
        2, 0, 0, 0,
        // 27: Constant control unit data (values)
        4, 5,

        // Input buffer infos (size)
        1, 0, 0, 0,
        // Input buffer info 0
        10, 0, 0, 0, 11, 0, 0, 0, 12, 0, 0, 0,

        // Output buffer infos (size)
        2, 0, 0, 0,
        // Output buffer info 0
        20, 0, 0, 0, 21, 0, 0, 0, 22, 0, 0, 0,
        // Output buffer info 1
        21, 0, 0, 0, 23, 0, 0, 0, 24, 0, 0, 0,

        // Constant control unit data buffer infos (size)
        1, 0, 0, 0,
        // Constant control unit data buffer info 0
        30, 0, 0, 0, 31, 0, 0, 0, 32, 0, 0, 0,

        // Constant DMA data buffer infos (size)
        1, 0, 0, 0,
        // Constant DMA data buffer info 0
        40, 0, 0, 0, 41, 0, 0, 0, 42, 0, 0, 0,

        // Intermediate data buffer infos (size)
        1, 0, 0, 0,
        // Intermediate data buffer info 0
        50, 0, 0, 0, 51, 0, 0, 0, 52, 0, 0, 0,
    };
    // clang-format on

    GIVEN("A serialized compiled network that is too short (no FourCC code)")
    {
        std::vector<uint8_t> serialized{ 'E', 'N' };

        WHEN("Calling DeserializeCompiledNetwork")
        {
            THEN("An error is raised")
            {
                REQUIRE_THROWS_MATCHES(
                    DeserializeCompiledNetwork(reinterpret_cast<const char*>(serialized.data()), serialized.size()),
                    CompiledNetworkException, Message("Data too short"));
            }
        }
    }

    GIVEN("A serialized compiled network with the wrong FourCC code")
    {
        std::vector<uint8_t> serialized = serializedValid;
        serialized[0]                   = 'X';

        WHEN("Calling DeserializeCompiledNetwork")
        {
            THEN("An error is raised")
            {
                REQUIRE_THROWS_MATCHES(
                    DeserializeCompiledNetwork(reinterpret_cast<const char*>(serialized.data()), serialized.size()),
                    CompiledNetworkException, Message("Not a serialized CompiledNetwork"));
            }
        }
    }

    GIVEN("A serialized compiled network that is too short (no version fields)")
    {
        std::vector<uint8_t> serialized{ 'E', 'N', 'C', 'N', 12, 25 };

        WHEN("Calling DeserializeCompiledNetwork")
        {
            THEN("An error is raised")
            {
                REQUIRE_THROWS_MATCHES(
                    DeserializeCompiledNetwork(reinterpret_cast<const char*>(serialized.data()), serialized.size()),
                    CompiledNetworkException, Message("Data too short"));
            }
        }
    }

    GIVEN("A serialized compiled network with an unsupported version")
    {
        std::vector<uint8_t> serialized = serializedValid;
        serialized[4]                   = 82;

        WHEN("Calling DeserializeCompiledNetwork")
        {
            THEN("An error is raised")
            {
                REQUIRE_THROWS_MATCHES(
                    DeserializeCompiledNetwork(reinterpret_cast<const char*>(serialized.data()), serialized.size()),
                    CompiledNetworkException, Message("Unsupported version"));
            }
        }
    }

    GIVEN("A serialized compiled network that has been corrupted (truncated)")
    {
        std::vector<uint8_t> serialized(serializedValid.begin(), serializedValid.begin() + 40);

        WHEN("Calling DeserializeCompiledNetwork")
        {
            THEN("An error is raised")
            {
                REQUIRE_THROWS_MATCHES(
                    DeserializeCompiledNetwork(reinterpret_cast<const char*>(serialized.data()), serialized.size()),
                    CompiledNetworkException, Message("Corrupted"));
            }
        }
    }
}
