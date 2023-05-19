//
// Copyright © 2018-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_command_stream/CommandStreamBuffer.hpp"

#include <catch.hpp>

#include <iostream>

using namespace ethosn::command_stream;

TEST_CASE("CommandStreamBuffer Version Header")
{
    GIVEN("An empty command stream")
    {
        CommandStreamBuffer cs;
        WHEN("The raw data is inspected")
        {
            const std::vector<uint32_t>& data = cs.GetData();

            THEN("There is a header with the version information")
            {
                const std::vector<uint32_t> expectedData = {
                    static_cast<uint32_t>('E') | (static_cast<uint32_t>('N') << 8) |
                        (static_cast<uint32_t>('C') << 16) | (static_cast<uint32_t>('S') << 24),
                    ETHOSN_COMMAND_STREAM_VERSION_MAJOR, ETHOSN_COMMAND_STREAM_VERSION_MINOR,
                    ETHOSN_COMMAND_STREAM_VERSION_PATCH
                };

                REQUIRE(data == expectedData);
            }
        }
    }
}

TEST_CASE("CommandStream Version Header")
{
    std::vector<uint32_t> validCmdStreamData = {
        static_cast<uint32_t>('E') | (static_cast<uint32_t>('N') << 8) | (static_cast<uint32_t>('C') << 16) |
            (static_cast<uint32_t>('S') << 24),
        ETHOSN_COMMAND_STREAM_VERSION_MAJOR, ETHOSN_COMMAND_STREAM_VERSION_MINOR, ETHOSN_COMMAND_STREAM_VERSION_PATCH
    };

    GIVEN("A valid command stream")
    {
        std::vector<uint32_t> data = validCmdStreamData;

        WHEN("Constructing a CommandStream object around this data")
        {
            CommandStream cs(data.data(), data.data() + data.size());

            THEN("The CommandStream is valid and reports the correct version")
            {
                REQUIRE(cs.IsValid());
                REQUIRE(cs.GetVersionMajor() == ETHOSN_COMMAND_STREAM_VERSION_MAJOR);
                REQUIRE(cs.GetVersionMinor() == ETHOSN_COMMAND_STREAM_VERSION_MINOR);
                REQUIRE(cs.GetVersionPatch() == ETHOSN_COMMAND_STREAM_VERSION_PATCH);
            }
        }
    }

    GIVEN("A command stream that is too short")
    {
        std::vector<uint32_t> data = { 0, 1, 2 };

        WHEN("Constructing a CommandStream object around this data")
        {
            CommandStream cs(data.data(), data.data() + data.size());

            THEN("The CommandStream is invalid and has no version information")
            {
                REQUIRE(!cs.IsValid());
                REQUIRE(cs.GetVersionMajor() == 0);
                REQUIRE(cs.GetVersionMinor() == 0);
                REQUIRE(cs.GetVersionPatch() == 0);
            }
        }
    }

    GIVEN("A command stream that has the wrong fourcc code")
    {
        std::vector<uint32_t> data = validCmdStreamData;
        data[0]                    = 1234;

        WHEN("Constructing a CommandStream object around this data")
        {
            CommandStream cs(data.data(), data.data() + data.size());

            THEN("The CommandStream is invalid and has no version information")
            {
                REQUIRE(!cs.IsValid());
                REQUIRE(cs.GetVersionMajor() == 0);
                REQUIRE(cs.GetVersionMinor() == 0);
                REQUIRE(cs.GetVersionPatch() == 0);
            }
        }
    }

    GIVEN("A command stream that has the wrong version")
    {
        std::vector<uint32_t> data = validCmdStreamData;
        data[1]                    = ETHOSN_COMMAND_STREAM_VERSION_MAJOR + 1;

        WHEN("Constructing a CommandStream object around this data")
        {
            CommandStream cs(data.data(), data.data() + data.size());

            THEN("The CommandStream is invalid but the version is reported correctly")
            {
                REQUIRE(!cs.IsValid());
                REQUIRE(cs.GetVersionMajor() == ETHOSN_COMMAND_STREAM_VERSION_MAJOR + 1);
                REQUIRE(cs.GetVersionMinor() == ETHOSN_COMMAND_STREAM_VERSION_MINOR);
                REQUIRE(cs.GetVersionPatch() == ETHOSN_COMMAND_STREAM_VERSION_PATCH);
            }
        }
    }
}
