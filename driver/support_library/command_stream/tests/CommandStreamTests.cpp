//
// Copyright © 2018-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_command_stream/CommandStreamBuilder.hpp"

#include <catch.hpp>

#include <iostream>

using namespace ethosn::command_stream;

TEST_CASE("CommandStreamBuilder Version Header")
{
    GIVEN("An empty command stream")
    {
        const std::vector<uint32_t>& data = BuildCommandStream({}, {}, {}, {}, {});

        WHEN("The raw data is inspected")
        {
            THEN("There is a header with the version information")
            {
                const std::vector<uint32_t> expectedData = {
                    static_cast<uint32_t>('E') | (static_cast<uint32_t>('N') << 8) |
                        (static_cast<uint32_t>('C') << 16) | (static_cast<uint32_t>('S') << 24),
                    ETHOSN_COMMAND_STREAM_VERSION_MAJOR, ETHOSN_COMMAND_STREAM_VERSION_MINOR,
                    ETHOSN_COMMAND_STREAM_VERSION_PATCH
                };

                REQUIRE(std::equal(expectedData.begin(), expectedData.end(), data.begin()));
            }
        }
    }
}

TEST_CASE("CommandStreamParser Version Header")
{
    std::vector<uint32_t> validCmdStreamData = {
        static_cast<uint32_t>('E') | (static_cast<uint32_t>('N') << 8) | (static_cast<uint32_t>('C') << 16) |
            (static_cast<uint32_t>('S') << 24),
        ETHOSN_COMMAND_STREAM_VERSION_MAJOR, ETHOSN_COMMAND_STREAM_VERSION_MINOR, ETHOSN_COMMAND_STREAM_VERSION_PATCH
    };

    GIVEN("A valid command stream")
    {
        std::vector<uint32_t> data = validCmdStreamData;

        WHEN("Constructing a CommandStreamParser object around this data")
        {
            CommandStreamParser parser(data.data(), data.data() + data.size());

            THEN("The CommandStream is valid and reports the correct version")
            {
                REQUIRE(parser.IsValid());
                REQUIRE(parser.GetVersionMajor() == ETHOSN_COMMAND_STREAM_VERSION_MAJOR);
                REQUIRE(parser.GetVersionMinor() == ETHOSN_COMMAND_STREAM_VERSION_MINOR);
                REQUIRE(parser.GetVersionPatch() == ETHOSN_COMMAND_STREAM_VERSION_PATCH);
            }
        }
    }

    GIVEN("A command stream that is too short")
    {
        std::vector<uint32_t> data = { 0, 1, 2 };

        WHEN("Constructing a CommandStreamParser object around this data")
        {
            CommandStreamParser parser(data.data(), data.data() + data.size());

            THEN("The CommandStreamParser is invalid and has no version information")
            {
                REQUIRE(!parser.IsValid());
                REQUIRE(parser.GetVersionMajor() == 0);
                REQUIRE(parser.GetVersionMinor() == 0);
                REQUIRE(parser.GetVersionPatch() == 0);
            }
        }
    }

    GIVEN("A command stream that has the wrong fourcc code")
    {
        std::vector<uint32_t> data = validCmdStreamData;
        data[0]                    = 1234;

        WHEN("Constructing a CommandStreamParser object around this data")
        {
            CommandStreamParser parser(data.data(), data.data() + data.size());

            THEN("The CommandStreamParser is invalid and has no version information")
            {
                REQUIRE(!parser.IsValid());
                REQUIRE(parser.GetVersionMajor() == 0);
                REQUIRE(parser.GetVersionMinor() == 0);
                REQUIRE(parser.GetVersionPatch() == 0);
            }
        }
    }

    GIVEN("A command stream that has the wrong version")
    {
        std::vector<uint32_t> data = validCmdStreamData;
        data[1]                    = ETHOSN_COMMAND_STREAM_VERSION_MAJOR + 1;

        WHEN("Constructing a CommandStreamParser object around this data")
        {
            CommandStreamParser parser(data.data(), data.data() + data.size());

            THEN("The CommandStreamParser is invalid but the version is reported correctly")
            {
                REQUIRE(!parser.IsValid());
                REQUIRE(parser.GetVersionMajor() == ETHOSN_COMMAND_STREAM_VERSION_MAJOR + 1);
                REQUIRE(parser.GetVersionMinor() == ETHOSN_COMMAND_STREAM_VERSION_MINOR);
                REQUIRE(parser.GetVersionPatch() == ETHOSN_COMMAND_STREAM_VERSION_PATCH);
            }
        }
    }
}

TEST_CASE("BuildCommandStream")
{
    std::vector<uint32_t> data = BuildCommandStream(
        {
            Agent{ IfmS{} },
            Agent{ WgtS{} },
            Agent{ MceS{} },
            Agent{ PleS{} },
            Agent{ OfmS{} },
        },
        {}, {}, {}, {});

    CommandStreamParser parser(data.data(), data.data() + data.size());
    REQUIRE(parser.IsValid());

    const CommandStream* cs = parser.GetData();
    REQUIRE(cs != nullptr);

    const Agent* agents = cs->GetAgentsArray();

    CHECK(agents[0].type == AgentType::IFM_STREAMER);
    CHECK(agents[1].type == AgentType::WGT_STREAMER);
    CHECK(agents[2].type == AgentType::MCE_SCHEDULER);
    CHECK(agents[3].type == AgentType::PLE_SCHEDULER);
    CHECK(agents[4].type == AgentType::OFM_STREAMER);
}
