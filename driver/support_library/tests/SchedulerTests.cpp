//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/DebuggingContext.hpp"
#include "../src/cascading/Scheduler.hpp"

#include <catch.hpp>

using namespace ethosn::support_library;
using namespace ethosn::command_stream::cascading;

const char* CommandTypeToString(CommandType t)
{
    using namespace ethosn::command_stream::cascading;
    switch (t)
    {
        case CommandType::WaitForAgent:
            return "WaitForAgent";
        case CommandType::LoadIfmStripe:
            return "LoadIfmStripe";
        case CommandType::LoadWgtStripe:
            return "LoadWgtStripe";
        case CommandType::ProgramMceStripe:
            return "ProgramMceStripe";
        case CommandType::StartMceStripe:
            return "StartMceStripe";
        case CommandType::LoadPleCode:
            return "LoadPleCode";
        case CommandType::StartPleStripe:
            return "StartPleStripe";
        case CommandType::StoreOfmStripe:
            return "StoreOfmStripe";
        default:
            FAIL("Invalid cascading command type: " + std::to_string(static_cast<uint32_t>(t)));
            return "?";
    }
}

namespace ethosn
{
namespace command_stream
{
namespace cascading
{
bool operator==(const Command& lhs, const Command& rhs)
{
    static_assert(sizeof(Command) ==
                      sizeof(lhs.type) + sizeof(lhs.agentId) + sizeof(lhs.stripeId) + sizeof(lhs.extraDataOffset),
                  "New fields added");
    return lhs.type == rhs.type && lhs.agentId == rhs.agentId && lhs.stripeId == rhs.stripeId &&
           lhs.extraDataOffset == rhs.extraDataOffset;
}

}    // namespace cascading
}    // namespace command_stream
}    // namespace ethosn

namespace Catch
{
template <>
struct StringMaker<Command>
{
    static std::string convert(const Command& c)
    {
        return "\n  Command { CommandType::" + std::string(CommandTypeToString(c.type)) + ", " +
               std::to_string(c.agentId) + ", " + std::to_string(c.stripeId) + ", 0 }";
    }
};
}    // namespace Catch

struct StripeQueue
{
    bool limitedQueue   = false;
    uint32_t numStripes = 0;
};

TEST_CASE("Cascading/Scheduler/ComplexSingleLayer")
{
    //        IfmS               WgtS                MceS                      PleL/PleS/OfmS
    //       (load x3)          (load x1)           (xyz order)               (accumulate all mce stripes)
    //                                               +----------+              +----------+
    //                                              /          /|             /          /|
    //       +----------+                          +----------+ |            /          / |
    //      /          /|            +-+          /          /| |           /          /  |
    //     /          / +           / /|         +----------+ | +          /          /   |
    //    /          / /|          +-+ +        /          /| |/|         /          /    |
    //   +----------+ / +         / /|/        +----------+ | + |        +----------+     |
    //   |          |/ /|        +-+ +         |          | |/| |        |          |     |
    //   +----------+ / +       / /|/          |          | + | +        |          |     |
    //   |          |/ /|      +-+ +           |          |/| |/|        |          |     |
    //   +----------+ / +      | |/            +----------+ | + |        |          |     |
    //   |          |/ /|      +-+             |          | |/| |        |          |     |
    //   +----------+ / +                      |          | + | +        |          |     +
    //   |          |/ /|                      |          |/| |/         |          |    /
    //   +----------+ / +                      +----------+ | +          |          |   /
    //   |          |/ /                       |          | |/           |          |  /
    //   +----------+ /                        |          | +            |          | /
    //   |          |/                         |          |/             |          |/
    //   +----------+                          +----------+              +----------+
    //
    IfmSDesc ifms{};
    ifms.fmData.tile.numSlots = 4;

    WgtSDesc wgts{};
    wgts.tile.numSlots = 2;

    PleSDesc ples{};
    ples.ofmTile.numSlots = 2;

    OfmSDesc ofms{};
    ofms.fmData.tile.numSlots = 2;

    std::vector<AgentDescAndDeps> complexSingleLayerCmdStream{
        AgentDescAndDeps{
            AgentDesc(18, ifms),
            {
                /* .readDependencies     = */ {},
                /* .writeDependencies    =*/{ { { 3, { 3, 6 }, { 1, 2 }, 1 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(3, wgts),
            {
                /* .readDependencies     = */ {},
                /* .writeDependencies    =*/{ { { 2, { 3, 1 }, { 3, 1 }, 0 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, PleLDesc{}),
            {
                /* .readDependencies     = */ {},
                /* .writeDependencies    =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(9, MceSDesc{}),
            {
                /* .readDependencies     = */
                { {
                    { 3, { 6, 3 }, { 2, 1 }, 1 },
                    { 2, { 1, 3 }, { 1, 3 }, 0 },
                } },
                /* .writeDependencies    =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, ples),
            {
                /* .readDependencies     = */
                { {
                    { 1, { 9, 1 }, { 9, 1 }, 0 },
                    { 2, { 1, 1 }, { 1, 1 }, 0 },
                } },
                /* .writeDependencies    =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, ofms),
            {
                /* .readDependencies     = */ { { { 1, { 1, 1 }, { 1, 1 }, 0 } } },
                /*.writeDependencies =*/{},
            },
        },
    };

    const std::vector<Command> expectedDmaRdCommands{
        // clang-format off
        Command { CommandType::LoadIfmStripe, 0, 0, 0 },
        Command { CommandType::LoadIfmStripe, 0, 1, 0 },
        Command { CommandType::LoadIfmStripe, 0, 2, 0 },
        Command { CommandType::LoadWgtStripe, 1, 0, 0 },
        Command { CommandType::LoadIfmStripe, 0, 3, 0 },
        Command { CommandType::WaitForAgent, 3, 0, 0 },
        Command { CommandType::LoadIfmStripe, 0, 4, 0 },
        Command { CommandType::WaitForAgent, 3, 1, 0 },
        Command { CommandType::LoadIfmStripe, 0, 5, 0 },
        Command { CommandType::LoadIfmStripe, 0, 6, 0 },
        Command { CommandType::WaitForAgent, 3, 2, 0 },
        Command { CommandType::LoadIfmStripe, 0, 7, 0 },
        Command { CommandType::LoadIfmStripe, 0, 8, 0 },
        Command { CommandType::LoadWgtStripe, 1, 1, 0 },
        Command { CommandType::LoadIfmStripe, 0, 9, 0 },
        Command { CommandType::WaitForAgent, 3, 3, 0 },
        Command { CommandType::LoadIfmStripe, 0, 10, 0 },
        Command { CommandType::WaitForAgent, 3, 4, 0 },
        Command { CommandType::LoadIfmStripe, 0, 11, 0 },
        Command { CommandType::LoadIfmStripe, 0, 12, 0 },
        Command { CommandType::WaitForAgent, 3, 5, 0 },
        Command { CommandType::LoadIfmStripe, 0, 13, 0 },
        Command { CommandType::LoadIfmStripe, 0, 14, 0 },
        Command { CommandType::WaitForAgent, 3, 2, 0 },
        Command { CommandType::LoadWgtStripe, 1, 2, 0 },
        Command { CommandType::LoadIfmStripe, 0, 15, 0 },
        Command { CommandType::WaitForAgent, 3, 6, 0 },
        Command { CommandType::LoadIfmStripe, 0, 16, 0 },
        Command { CommandType::WaitForAgent, 3, 7, 0 },
        Command { CommandType::LoadIfmStripe, 0, 17, 0 },
        Command { CommandType::LoadPleCode, 2, 0, 0 }
        // clang-format on
    };

    const std::vector<Command> expectedDmaWrCommands{
        Command{ CommandType::WaitForAgent, 4, 0, 0 },
        Command{ CommandType::StoreOfmStripe, 5, 0, 0 },
    };

    const std::vector<Command> expectedMceCommands{
        // clang-format off
        Command { CommandType::ProgramMceStripe, 3, 0, 0 },
        Command { CommandType::WaitForAgent, 0, 2, 0 },
        Command { CommandType::WaitForAgent, 1, 0, 0 },
        Command { CommandType::StartMceStripe, 3, 0, 0 },
        Command { CommandType::ProgramMceStripe, 3, 1, 0 },
        Command { CommandType::WaitForAgent, 0, 4, 0 },
        Command { CommandType::StartMceStripe, 3, 1, 0 },
        Command { CommandType::ProgramMceStripe, 3, 2, 0 },
        Command { CommandType::WaitForAgent, 0, 5, 0 },
        Command { CommandType::StartMceStripe, 3, 2, 0 },
        Command { CommandType::ProgramMceStripe, 3, 3, 0 },
        Command { CommandType::WaitForAgent, 0, 8, 0 },
        Command { CommandType::WaitForAgent, 1, 1, 0 },
        Command { CommandType::StartMceStripe, 3, 3, 0 },
        Command { CommandType::ProgramMceStripe, 3, 4, 0 },
        Command { CommandType::WaitForAgent, 0, 10, 0 },
        Command { CommandType::StartMceStripe, 3, 4, 0 },
        Command { CommandType::ProgramMceStripe, 3, 5, 0 },
        Command { CommandType::WaitForAgent, 0, 11, 0 },
        Command { CommandType::StartMceStripe, 3, 5, 0 },
        Command { CommandType::ProgramMceStripe, 3, 6, 0 },
        Command { CommandType::WaitForAgent, 0, 14, 0 },
        Command { CommandType::WaitForAgent, 1, 2, 0 },
        Command { CommandType::StartMceStripe, 3, 6, 0 },
        Command { CommandType::ProgramMceStripe, 3, 7, 0 },
        Command { CommandType::WaitForAgent, 0, 16, 0 },
        Command { CommandType::StartMceStripe, 3, 7, 0 },
        Command { CommandType::ProgramMceStripe, 3, 8, 0 },
        Command { CommandType::WaitForAgent, 0, 17, 0 },
        Command { CommandType::StartMceStripe, 3, 8,  0}
        // clang-format on
    };

    const std::vector<Command> expectedPleCommands{
        Command{ CommandType::WaitForAgent, 2, 0, 0 },
        Command{ CommandType::StartPleStripe, 4, 0, 0 },
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(complexSingleLayerCmdStream, debuggingContext);
    scheduler.Schedule();

    CHECK(scheduler.GetDmaRdCommands() == expectedDmaRdCommands);
    CHECK(scheduler.GetDmaWrCommands() == expectedDmaWrCommands);
    CHECK(scheduler.GetMceCommands() == expectedMceCommands);
    CHECK(scheduler.GetPleCommands() == expectedPleCommands);
}

TEST_CASE("Cascading/Scheduler/Strategy7")
{
    //        IfmS                       WgtS                MceS                            PleL/PleS/OfmS
    //       (load x3)                  (load x1)           (xyz order)                     (accumulate all mce stripes)
    //                                                        +----+----+----+----+              +-------------------+
    //                                                       /    /    /    /    /|             /                   /|
    //       +----+----+----+----+                          +----+----+----+----+ +            +-------------------+ +
    //      /    /    /    /    /|            +-+          /    /    /    /    /|/|           /                   /|/|
    //     +----+----+----+----+ +           / /|         +--- +--- +----+----+ + +          +-------------------+ + +
    //    /    /    /    /    /|/|          +-+ +        /    /    /    /    /|/|/|         /                   /|/|/|
    //   +----+----+----+----+ + +         / /|/        +----+----+----+----+ + + +        +-------------------+ + + +
    //   |    |    |    |    |/|/|        +-+ +         |    |    |    |    |/|/|/         |                   |/|/|/
    //   +----+----+----+----+ + +       / /|/          +----+----+----+----+ + +          +-------------------+ + +
    //   |    |    |    |    |/|/       +-+ +           |    |    |    |    |/|/           |                   |/|/
    //   +----+----+----+----+ +        | |/            +----+----+----+----+ +            +-------------------+ +
    //   |    |    |    |    |/         +-+             |    |    |    |    |/             |                   |/
    //   +----+----+----+----+                          +----+----+----+----+              +-------------------+
    //
    IfmSDesc ifms{};
    ifms.fmData.tile.numSlots = 4;

    WgtSDesc wgts{};
    wgts.tile.numSlots = 2;

    PleSDesc ples{};
    ples.ofmTile.numSlots = 2;

    OfmSDesc ofms{};
    ofms.fmData.tile.numSlots = 2;

    std::vector<AgentDescAndDeps> strategy7CmdStream{
        AgentDescAndDeps{
            AgentDesc(72, ifms),
            {
                /*.readDependencies =*/{},
                /*.writeDependencies =*/{ { { 3, { 1, 1 }, { 1, 1 }, 0 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(6, wgts),
            {
                /*.readDependencies =*/{},
                /*.writeDependencies =*/{ { { 2, { 24, 2 }, { 1, 1 }, 0 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, PleLDesc{}),
            {
                /*.readDependencies =*/{},
                /*.writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(72, MceSDesc{}),
            {
                /*.readDependencies =*/
                { {
                    { 3, { 1, 1 }, { 1, 1 }, 0 },
                    { 2, { 2, 24 }, { 1, 1 }, 0 },
                } },
                /*.writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(9, ples),
            {
                /*.readDependencies =*/
                { {
                    { 2, { 0, 1 }, { 0, 1 }, 0 },
                    { 1, { 8, 1 }, { 8, 1 }, 0 },
                } },
                /*.writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(9, ofms),
            {
                /*.readDependencies =*/{ { { 1, { 1, 1 }, { 1, 1 }, 0 } } },
                /*.writeDependencies =*/{},
            },
        },
    };

    const std::vector<Command> expectedDmaRdCommands{
        // clang-format off
        Command { CommandType::LoadIfmStripe, 0, 0, 0 },
        Command { CommandType::LoadWgtStripe, 1, 0, 0 },
        Command { CommandType::LoadIfmStripe, 0, 1, 0 },
        Command { CommandType::LoadWgtStripe, 1, 1, 0 },
        Command { CommandType::LoadIfmStripe, 0, 2, 0 },
        Command { CommandType::LoadIfmStripe, 0, 3, 0 },
        Command { CommandType::WaitForAgent, 3, 0, 0 },
        Command { CommandType::LoadIfmStripe, 0, 4, 0 },
        Command { CommandType::WaitForAgent, 3, 1, 0 },
        Command { CommandType::LoadIfmStripe, 0, 5, 0 },
        Command { CommandType::WaitForAgent, 3, 2, 0 },
        Command { CommandType::LoadIfmStripe, 0, 6, 0 },
        Command { CommandType::WaitForAgent, 3, 3, 0 },
        Command { CommandType::LoadIfmStripe, 0, 7, 0 },
        Command { CommandType::WaitForAgent, 3, 4, 0 },
        Command { CommandType::LoadIfmStripe, 0, 8, 0 },
        Command { CommandType::WaitForAgent, 3, 5, 0 },
        Command { CommandType::LoadIfmStripe, 0, 9, 0 },
        Command { CommandType::WaitForAgent, 3, 6, 0 },
        Command { CommandType::LoadIfmStripe, 0, 10, 0 },
        Command { CommandType::WaitForAgent, 3, 7, 0 },
        Command { CommandType::LoadIfmStripe, 0, 11, 0 },
        Command { CommandType::WaitForAgent, 3, 8, 0 },
        Command { CommandType::LoadIfmStripe, 0, 12, 0 },
        Command { CommandType::WaitForAgent, 3, 9, 0 },
        Command { CommandType::LoadIfmStripe, 0, 13, 0 },
        Command { CommandType::WaitForAgent, 3, 10, 0 },
        Command { CommandType::LoadIfmStripe, 0, 14, 0 },
        Command { CommandType::WaitForAgent, 3, 11, 0 },
        Command { CommandType::LoadIfmStripe, 0, 15, 0 },
        Command { CommandType::WaitForAgent, 3, 12, 0 },
        Command { CommandType::LoadIfmStripe, 0, 16, 0 },
        Command { CommandType::WaitForAgent, 3, 13, 0 },
        Command { CommandType::LoadIfmStripe, 0, 17, 0 },
        Command { CommandType::WaitForAgent, 3, 14, 0 },
        Command { CommandType::LoadIfmStripe, 0, 18, 0 },
        Command { CommandType::WaitForAgent, 3, 15, 0 },
        Command { CommandType::LoadIfmStripe, 0, 19, 0 },
        Command { CommandType::WaitForAgent, 3, 16, 0 },
        Command { CommandType::LoadIfmStripe, 0, 20, 0 },
        Command { CommandType::WaitForAgent, 3, 17, 0 },
        Command { CommandType::LoadIfmStripe, 0, 21, 0 },
        Command { CommandType::WaitForAgent, 3, 18, 0 },
        Command { CommandType::LoadIfmStripe, 0, 22, 0 },
        Command { CommandType::WaitForAgent, 3, 19, 0 },
        Command { CommandType::LoadIfmStripe, 0, 23, 0 },
        Command { CommandType::WaitForAgent, 3, 20, 0 },
        Command { CommandType::LoadIfmStripe, 0, 24, 0 },
        Command { CommandType::WaitForAgent, 3, 0, 0 },
        Command { CommandType::LoadWgtStripe, 1, 2, 0 },
        Command { CommandType::WaitForAgent, 3, 21, 0 },
        Command { CommandType::LoadIfmStripe, 0, 25, 0 },
        Command { CommandType::WaitForAgent, 3, 1, 0 },
        Command { CommandType::LoadWgtStripe, 1, 3, 0 },
        Command { CommandType::WaitForAgent, 3, 22, 0 },
        Command { CommandType::LoadIfmStripe, 0, 26, 0 },
        Command { CommandType::WaitForAgent, 3, 23, 0 },
        Command { CommandType::LoadIfmStripe, 0, 27, 0 },
        Command { CommandType::WaitForAgent, 3, 24, 0 },
        Command { CommandType::LoadIfmStripe, 0, 28, 0 },
        Command { CommandType::WaitForAgent, 3, 25, 0 },
        Command { CommandType::LoadIfmStripe, 0, 29, 0 },
        Command { CommandType::WaitForAgent, 3, 26, 0 },
        Command { CommandType::LoadIfmStripe, 0, 30, 0 },
        Command { CommandType::WaitForAgent, 3, 27, 0 },
        Command { CommandType::LoadIfmStripe, 0, 31, 0 },
        Command { CommandType::WaitForAgent, 3, 28, 0 },
        Command { CommandType::LoadIfmStripe, 0, 32, 0 },
        Command { CommandType::WaitForAgent, 3, 29, 0 },
        Command { CommandType::LoadIfmStripe, 0, 33, 0 },
        Command { CommandType::WaitForAgent, 3, 30, 0 },
        Command { CommandType::LoadIfmStripe, 0, 34, 0 },
        Command { CommandType::WaitForAgent, 3, 31, 0 },
        Command { CommandType::LoadIfmStripe, 0, 35, 0 },
        Command { CommandType::WaitForAgent, 3, 32, 0 },
        Command { CommandType::LoadIfmStripe, 0, 36, 0 },
        Command { CommandType::WaitForAgent, 3, 33, 0 },
        Command { CommandType::LoadIfmStripe, 0, 37, 0 },
        Command { CommandType::WaitForAgent, 3, 34, 0 },
        Command { CommandType::LoadIfmStripe, 0, 38, 0 },
        Command { CommandType::WaitForAgent, 3, 35, 0 },
        Command { CommandType::LoadIfmStripe, 0, 39, 0 },
        Command { CommandType::WaitForAgent, 3, 36, 0 },
        Command { CommandType::LoadIfmStripe, 0, 40, 0 },
        Command { CommandType::WaitForAgent, 3, 37, 0 },
        Command { CommandType::LoadIfmStripe, 0, 41, 0 },
        Command { CommandType::WaitForAgent, 3, 38, 0 },
        Command { CommandType::LoadIfmStripe, 0, 42, 0 },
        Command { CommandType::WaitForAgent, 3, 39, 0 },
        Command { CommandType::LoadIfmStripe, 0, 43, 0 },
        Command { CommandType::WaitForAgent, 3, 40, 0 },
        Command { CommandType::LoadIfmStripe, 0, 44, 0 },
        Command { CommandType::WaitForAgent, 3, 41, 0 },
        Command { CommandType::LoadIfmStripe, 0, 45, 0 },
        Command { CommandType::WaitForAgent, 3, 42, 0 },
        Command { CommandType::LoadIfmStripe, 0, 46, 0 },
        Command { CommandType::WaitForAgent, 3, 43, 0 },
        Command { CommandType::LoadIfmStripe, 0, 47, 0 },
        Command { CommandType::WaitForAgent, 3, 44, 0 },
        Command { CommandType::LoadIfmStripe, 0, 48, 0 },
        Command { CommandType::WaitForAgent, 3, 24, 0 },
        Command { CommandType::LoadWgtStripe, 1, 4, 0 },
        Command { CommandType::WaitForAgent, 3, 45, 0 },
        Command { CommandType::LoadIfmStripe, 0, 49, 0 },
        Command { CommandType::WaitForAgent, 3, 25, 0 },
        Command { CommandType::LoadWgtStripe, 1, 5, 0 },
        Command { CommandType::WaitForAgent, 3, 46, 0 },
        Command { CommandType::LoadIfmStripe, 0, 50, 0 },
        Command { CommandType::WaitForAgent, 3, 47, 0 },
        Command { CommandType::LoadIfmStripe, 0, 51, 0 },
        Command { CommandType::WaitForAgent, 3, 48, 0 },
        Command { CommandType::LoadIfmStripe, 0, 52, 0 },
        Command { CommandType::WaitForAgent, 3, 49, 0 },
        Command { CommandType::LoadIfmStripe, 0, 53, 0 },
        Command { CommandType::WaitForAgent, 3, 50, 0 },
        Command { CommandType::LoadIfmStripe, 0, 54, 0 },
        Command { CommandType::WaitForAgent, 3, 51, 0 },
        Command { CommandType::LoadIfmStripe, 0, 55, 0 },
        Command { CommandType::WaitForAgent, 3, 52, 0 },
        Command { CommandType::LoadIfmStripe, 0, 56, 0 },
        Command { CommandType::WaitForAgent, 3, 53, 0 },
        Command { CommandType::LoadIfmStripe, 0, 57, 0 },
        Command { CommandType::WaitForAgent, 3, 54, 0 },
        Command { CommandType::LoadIfmStripe, 0, 58, 0 },
        Command { CommandType::WaitForAgent, 3, 55, 0 },
        Command { CommandType::LoadIfmStripe, 0, 59, 0 },
        Command { CommandType::WaitForAgent, 3, 56, 0 },
        Command { CommandType::LoadIfmStripe, 0, 60, 0 },
        Command { CommandType::WaitForAgent, 3, 57, 0 },
        Command { CommandType::LoadIfmStripe, 0, 61, 0 },
        Command { CommandType::WaitForAgent, 3, 58, 0 },
        Command { CommandType::LoadIfmStripe, 0, 62, 0 },
        Command { CommandType::WaitForAgent, 3, 59, 0 },
        Command { CommandType::LoadIfmStripe, 0, 63, 0 },
        Command { CommandType::WaitForAgent, 3, 60, 0 },
        Command { CommandType::LoadIfmStripe, 0, 64, 0 },
        Command { CommandType::WaitForAgent, 3, 61, 0 },
        Command { CommandType::LoadIfmStripe, 0, 65, 0 },
        Command { CommandType::WaitForAgent, 3, 62, 0 },
        Command { CommandType::LoadIfmStripe, 0, 66, 0 },
        Command { CommandType::WaitForAgent, 3, 63, 0 },
        Command { CommandType::LoadIfmStripe, 0, 67, 0 },
        Command { CommandType::WaitForAgent, 3, 64, 0 },
        Command { CommandType::LoadIfmStripe, 0, 68, 0 },
        Command { CommandType::WaitForAgent, 3, 65, 0 },
        Command { CommandType::LoadIfmStripe, 0, 69, 0 },
        Command { CommandType::WaitForAgent, 3, 66, 0 },
        Command { CommandType::LoadIfmStripe, 0, 70, 0 },
        Command { CommandType::WaitForAgent, 3, 67, 0 },
        Command { CommandType::LoadIfmStripe, 0, 71, 0 }
        // clang-format on
    };
    const std::vector<Command> expectedDmaWrCommands{
        //clang-format off
        Command{ CommandType::WaitForAgent, 4, 0, 0 }, Command{ CommandType::StoreOfmStripe, 5, 0, 0 },
        Command{ CommandType::WaitForAgent, 4, 1, 0 }, Command{ CommandType::StoreOfmStripe, 5, 1, 0 },
        Command{ CommandType::WaitForAgent, 4, 2, 0 }, Command{ CommandType::StoreOfmStripe, 5, 2, 0 },
        Command{ CommandType::WaitForAgent, 4, 3, 0 }, Command{ CommandType::StoreOfmStripe, 5, 3, 0 },
        Command{ CommandType::WaitForAgent, 4, 4, 0 }, Command{ CommandType::StoreOfmStripe, 5, 4, 0 },
        Command{ CommandType::WaitForAgent, 4, 5, 0 }, Command{ CommandType::StoreOfmStripe, 5, 5, 0 },
        Command{ CommandType::WaitForAgent, 4, 6, 0 }, Command{ CommandType::StoreOfmStripe, 5, 6, 0 },
        Command{ CommandType::WaitForAgent, 4, 7, 0 }, Command{ CommandType::StoreOfmStripe, 5, 7, 0 },
        Command{ CommandType::WaitForAgent, 4, 8, 0 }, Command{ CommandType::StoreOfmStripe, 5, 8, 0 },
        //clang-format on
    };

    const std::vector<Command> expectedMceCommands{
        // clang-format off
        Command { CommandType::ProgramMceStripe, 3, 0, 0 },
        Command { CommandType::WaitForAgent, 0, 0, 0 },
        Command { CommandType::WaitForAgent, 1, 0, 0 },
        Command { CommandType::StartMceStripe, 3, 0, 0 },
        Command { CommandType::ProgramMceStripe, 3, 1, 0 },
        Command { CommandType::WaitForAgent, 0, 1, 0 },
        Command { CommandType::WaitForAgent, 1, 1, 0 },
        Command { CommandType::StartMceStripe, 3, 1, 0 },
        Command { CommandType::ProgramMceStripe, 3, 2, 0 },
        Command { CommandType::WaitForAgent, 0, 2, 0 },
        Command { CommandType::StartMceStripe, 3, 2, 0 },
        Command { CommandType::ProgramMceStripe, 3, 3, 0 },
        Command { CommandType::WaitForAgent, 0, 3, 0 },
        Command { CommandType::StartMceStripe, 3, 3, 0 },
        Command { CommandType::ProgramMceStripe, 3, 4, 0 },
        Command { CommandType::WaitForAgent, 0, 4, 0 },
        Command { CommandType::StartMceStripe, 3, 4, 0 },
        Command { CommandType::ProgramMceStripe, 3, 5, 0 },
        Command { CommandType::WaitForAgent, 0, 5, 0 },
        Command { CommandType::StartMceStripe, 3, 5, 0 },
        Command { CommandType::ProgramMceStripe, 3, 6, 0 },
        Command { CommandType::WaitForAgent, 0, 6, 0 },
        Command { CommandType::StartMceStripe, 3, 6, 0 },
        Command { CommandType::ProgramMceStripe, 3, 7, 0 },
        Command { CommandType::WaitForAgent, 0, 7, 0 },
        Command { CommandType::StartMceStripe, 3, 7, 0 },
        Command { CommandType::ProgramMceStripe, 3, 8, 0 },
        Command { CommandType::WaitForAgent, 0, 8, 0 },
        Command { CommandType::StartMceStripe, 3, 8, 0 },
        Command { CommandType::ProgramMceStripe, 3, 9, 0 },
        Command { CommandType::WaitForAgent, 0, 9, 0 },
        Command { CommandType::StartMceStripe, 3, 9, 0 },
        Command { CommandType::ProgramMceStripe, 3, 10, 0 },
        Command { CommandType::WaitForAgent, 0, 10, 0 },
        Command { CommandType::StartMceStripe, 3, 10, 0 },
        Command { CommandType::ProgramMceStripe, 3, 11, 0 },
        Command { CommandType::WaitForAgent, 0, 11, 0 },
        Command { CommandType::StartMceStripe, 3, 11, 0 },
        Command { CommandType::ProgramMceStripe, 3, 12, 0 },
        Command { CommandType::WaitForAgent, 0, 12, 0 },
        Command { CommandType::StartMceStripe, 3, 12, 0 },
        Command { CommandType::ProgramMceStripe, 3, 13, 0 },
        Command { CommandType::WaitForAgent, 0, 13, 0 },
        Command { CommandType::StartMceStripe, 3, 13, 0 },
        Command { CommandType::ProgramMceStripe, 3, 14, 0 },
        Command { CommandType::WaitForAgent, 0, 14, 0 },
        Command { CommandType::StartMceStripe, 3, 14, 0 },
        Command { CommandType::ProgramMceStripe, 3, 15, 0 },
        Command { CommandType::WaitForAgent, 0, 15, 0 },
        Command { CommandType::StartMceStripe, 3, 15, 0 },
        Command { CommandType::ProgramMceStripe, 3, 16, 0 },
        Command { CommandType::WaitForAgent, 0, 16, 0 },
        Command { CommandType::StartMceStripe, 3, 16, 0 },
        Command { CommandType::ProgramMceStripe, 3, 17, 0 },
        Command { CommandType::WaitForAgent, 0, 17, 0 },
        Command { CommandType::StartMceStripe, 3, 17, 0 },
        Command { CommandType::ProgramMceStripe, 3, 18, 0 },
        Command { CommandType::WaitForAgent, 0, 18, 0 },
        Command { CommandType::StartMceStripe, 3, 18, 0 },
        Command { CommandType::ProgramMceStripe, 3, 19, 0 },
        Command { CommandType::WaitForAgent, 0, 19, 0 },
        Command { CommandType::StartMceStripe, 3, 19, 0 },
        Command { CommandType::ProgramMceStripe, 3, 20, 0 },
        Command { CommandType::WaitForAgent, 0, 20, 0 },
        Command { CommandType::StartMceStripe, 3, 20, 0 },
        Command { CommandType::ProgramMceStripe, 3, 21, 0 },
        Command { CommandType::WaitForAgent, 0, 21, 0 },
        Command { CommandType::StartMceStripe, 3, 21, 0 },
        Command { CommandType::ProgramMceStripe, 3, 22, 0 },
        Command { CommandType::WaitForAgent, 0, 22, 0 },
        Command { CommandType::StartMceStripe, 3, 22, 0 },
        Command { CommandType::ProgramMceStripe, 3, 23, 0 },
        Command { CommandType::WaitForAgent, 0, 23, 0 },
        Command { CommandType::StartMceStripe, 3, 23, 0 },
        Command { CommandType::ProgramMceStripe, 3, 24, 0 },
        Command { CommandType::WaitForAgent, 0, 24, 0 },
        Command { CommandType::WaitForAgent, 1, 2, 0 },
        Command { CommandType::StartMceStripe, 3, 24, 0 },
        Command { CommandType::ProgramMceStripe, 3, 25, 0 },
        Command { CommandType::WaitForAgent, 0, 25, 0 },
        Command { CommandType::WaitForAgent, 1, 3, 0 },
        Command { CommandType::StartMceStripe, 3, 25, 0 },
        Command { CommandType::ProgramMceStripe, 3, 26, 0 },
        Command { CommandType::WaitForAgent, 0, 26, 0 },
        Command { CommandType::StartMceStripe, 3, 26, 0 },
        Command { CommandType::ProgramMceStripe, 3, 27, 0 },
        Command { CommandType::WaitForAgent, 0, 27, 0 },
        Command { CommandType::StartMceStripe, 3, 27, 0 },
        Command { CommandType::ProgramMceStripe, 3, 28, 0 },
        Command { CommandType::WaitForAgent, 0, 28, 0 },
        Command { CommandType::StartMceStripe, 3, 28, 0 },
        Command { CommandType::ProgramMceStripe, 3, 29, 0 },
        Command { CommandType::WaitForAgent, 0, 29, 0 },
        Command { CommandType::StartMceStripe, 3, 29, 0 },
        Command { CommandType::ProgramMceStripe, 3, 30, 0 },
        Command { CommandType::WaitForAgent, 0, 30, 0 },
        Command { CommandType::StartMceStripe, 3, 30, 0 },
        Command { CommandType::ProgramMceStripe, 3, 31, 0 },
        Command { CommandType::WaitForAgent, 0, 31, 0 },
        Command { CommandType::StartMceStripe, 3, 31, 0 },
        Command { CommandType::ProgramMceStripe, 3, 32, 0 },
        Command { CommandType::WaitForAgent, 0, 32, 0 },
        Command { CommandType::StartMceStripe, 3, 32, 0 },
        Command { CommandType::ProgramMceStripe, 3, 33, 0 },
        Command { CommandType::WaitForAgent, 0, 33, 0 },
        Command { CommandType::StartMceStripe, 3, 33, 0 },
        Command { CommandType::ProgramMceStripe, 3, 34, 0 },
        Command { CommandType::WaitForAgent, 0, 34, 0 },
        Command { CommandType::StartMceStripe, 3, 34, 0 },
        Command { CommandType::ProgramMceStripe, 3, 35, 0 },
        Command { CommandType::WaitForAgent, 0, 35, 0 },
        Command { CommandType::StartMceStripe, 3, 35, 0 },
        Command { CommandType::ProgramMceStripe, 3, 36, 0 },
        Command { CommandType::WaitForAgent, 0, 36, 0 },
        Command { CommandType::StartMceStripe, 3, 36, 0 },
        Command { CommandType::ProgramMceStripe, 3, 37, 0 },
        Command { CommandType::WaitForAgent, 0, 37, 0 },
        Command { CommandType::StartMceStripe, 3, 37, 0 },
        Command { CommandType::ProgramMceStripe, 3, 38, 0 },
        Command { CommandType::WaitForAgent, 0, 38, 0 },
        Command { CommandType::StartMceStripe, 3, 38, 0 },
        Command { CommandType::ProgramMceStripe, 3, 39, 0 },
        Command { CommandType::WaitForAgent, 0, 39, 0 },
        Command { CommandType::StartMceStripe, 3, 39, 0 },
        Command { CommandType::ProgramMceStripe, 3, 40, 0 },
        Command { CommandType::WaitForAgent, 0, 40, 0 },
        Command { CommandType::StartMceStripe, 3, 40, 0 },
        Command { CommandType::ProgramMceStripe, 3, 41, 0 },
        Command { CommandType::WaitForAgent, 0, 41, 0 },
        Command { CommandType::StartMceStripe, 3, 41, 0 },
        Command { CommandType::ProgramMceStripe, 3, 42, 0 },
        Command { CommandType::WaitForAgent, 0, 42, 0 },
        Command { CommandType::StartMceStripe, 3, 42, 0 },
        Command { CommandType::ProgramMceStripe, 3, 43, 0 },
        Command { CommandType::WaitForAgent, 0, 43, 0 },
        Command { CommandType::StartMceStripe, 3, 43, 0 },
        Command { CommandType::ProgramMceStripe, 3, 44, 0 },
        Command { CommandType::WaitForAgent, 0, 44, 0 },
        Command { CommandType::StartMceStripe, 3, 44, 0 },
        Command { CommandType::ProgramMceStripe, 3, 45, 0 },
        Command { CommandType::WaitForAgent, 0, 45, 0 },
        Command { CommandType::StartMceStripe, 3, 45, 0 },
        Command { CommandType::ProgramMceStripe, 3, 46, 0 },
        Command { CommandType::WaitForAgent, 0, 46, 0 },
        Command { CommandType::StartMceStripe, 3, 46, 0 },
        Command { CommandType::ProgramMceStripe, 3, 47, 0 },
        Command { CommandType::WaitForAgent, 0, 47, 0 },
        Command { CommandType::StartMceStripe, 3, 47, 0 },
        Command { CommandType::ProgramMceStripe, 3, 48, 0 },
        Command { CommandType::WaitForAgent, 0, 48, 0 },
        Command { CommandType::WaitForAgent, 1, 4, 0 },
        Command { CommandType::StartMceStripe, 3, 48, 0 },
        Command { CommandType::ProgramMceStripe, 3, 49, 0 },
        Command { CommandType::WaitForAgent, 0, 49, 0 },
        Command { CommandType::WaitForAgent, 1, 5, 0 },
        Command { CommandType::StartMceStripe, 3, 49, 0 },
        Command { CommandType::ProgramMceStripe, 3, 50, 0 },
        Command { CommandType::WaitForAgent, 0, 50, 0 },
        Command { CommandType::StartMceStripe, 3, 50, 0 },
        Command { CommandType::ProgramMceStripe, 3, 51, 0 },
        Command { CommandType::WaitForAgent, 0, 51, 0 },
        Command { CommandType::StartMceStripe, 3, 51, 0 },
        Command { CommandType::ProgramMceStripe, 3, 52, 0 },
        Command { CommandType::WaitForAgent, 0, 52, 0 },
        Command { CommandType::StartMceStripe, 3, 52, 0 },
        Command { CommandType::ProgramMceStripe, 3, 53, 0 },
        Command { CommandType::WaitForAgent, 0, 53, 0 },
        Command { CommandType::StartMceStripe, 3, 53, 0 },
        Command { CommandType::ProgramMceStripe, 3, 54, 0 },
        Command { CommandType::WaitForAgent, 0, 54, 0 },
        Command { CommandType::StartMceStripe, 3, 54, 0 },
        Command { CommandType::ProgramMceStripe, 3, 55, 0 },
        Command { CommandType::WaitForAgent, 0, 55, 0 },
        Command { CommandType::StartMceStripe, 3, 55, 0 },
        Command { CommandType::ProgramMceStripe, 3, 56, 0 },
        Command { CommandType::WaitForAgent, 0, 56, 0 },
        Command { CommandType::StartMceStripe, 3, 56, 0 },
        Command { CommandType::ProgramMceStripe, 3, 57, 0 },
        Command { CommandType::WaitForAgent, 0, 57, 0 },
        Command { CommandType::StartMceStripe, 3, 57, 0 },
        Command { CommandType::ProgramMceStripe, 3, 58, 0 },
        Command { CommandType::WaitForAgent, 0, 58, 0 },
        Command { CommandType::StartMceStripe, 3, 58, 0 },
        Command { CommandType::ProgramMceStripe, 3, 59, 0 },
        Command { CommandType::WaitForAgent, 0, 59, 0 },
        Command { CommandType::StartMceStripe, 3, 59, 0 },
        Command { CommandType::ProgramMceStripe, 3, 60, 0 },
        Command { CommandType::WaitForAgent, 0, 60, 0 },
        Command { CommandType::StartMceStripe, 3, 60, 0 },
        Command { CommandType::ProgramMceStripe, 3, 61, 0 },
        Command { CommandType::WaitForAgent, 0, 61, 0 },
        Command { CommandType::StartMceStripe, 3, 61, 0 },
        Command { CommandType::ProgramMceStripe, 3, 62, 0 },
        Command { CommandType::WaitForAgent, 0, 62, 0 },
        Command { CommandType::StartMceStripe, 3, 62, 0 },
        Command { CommandType::ProgramMceStripe, 3, 63, 0 },
        Command { CommandType::WaitForAgent, 0, 63, 0 },
        Command { CommandType::StartMceStripe, 3, 63, 0 },
        Command { CommandType::ProgramMceStripe, 3, 64, 0 },
        Command { CommandType::WaitForAgent, 0, 64, 0 },
        Command { CommandType::StartMceStripe, 3, 64, 0 },
        Command { CommandType::ProgramMceStripe, 3, 65, 0 },
        Command { CommandType::WaitForAgent, 0, 65, 0 },
        Command { CommandType::StartMceStripe, 3, 65, 0 },
        Command { CommandType::ProgramMceStripe, 3, 66, 0 },
        Command { CommandType::WaitForAgent, 0, 66, 0 },
        Command { CommandType::StartMceStripe, 3, 66, 0 },
        Command { CommandType::ProgramMceStripe, 3, 67, 0 },
        Command { CommandType::WaitForAgent, 0, 67, 0 },
        Command { CommandType::StartMceStripe, 3, 67, 0 },
        Command { CommandType::ProgramMceStripe, 3, 68, 0 },
        Command { CommandType::WaitForAgent, 0, 68, 0 },
        Command { CommandType::StartMceStripe, 3, 68, 0 },
        Command { CommandType::ProgramMceStripe, 3, 69, 0 },
        Command { CommandType::WaitForAgent, 0, 69, 0 },
        Command { CommandType::StartMceStripe, 3, 69, 0 },
        Command { CommandType::ProgramMceStripe, 3, 70, 0 },
        Command { CommandType::WaitForAgent, 0, 70, 0 },
        Command { CommandType::StartMceStripe, 3, 70, 0 },
        Command { CommandType::ProgramMceStripe, 3, 71, 0 },
        Command { CommandType::WaitForAgent, 0, 71, 0 },
        Command { CommandType::StartMceStripe, 3, 71,  0}
        // clang-format on
    };

    const std::vector<Command> expectedPleCommands{
        // clang-format off
        Command { CommandType::StartPleStripe, 4, 0, 0 },
        Command { CommandType::StartPleStripe, 4, 1, 0 },
        Command { CommandType::StartPleStripe, 4, 2, 0 },
        Command { CommandType::StartPleStripe, 4, 3, 0 },
        Command { CommandType::StartPleStripe, 4, 4, 0 },
        Command { CommandType::StartPleStripe, 4, 5, 0 },
        Command { CommandType::StartPleStripe, 4, 6, 0 },
        Command { CommandType::StartPleStripe, 4, 7, 0 },
        Command { CommandType::StartPleStripe, 4, 8,  0}
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(strategy7CmdStream, debuggingContext);
    scheduler.Schedule();

    CHECK(scheduler.GetDmaRdCommands() == expectedDmaRdCommands);
    CHECK(scheduler.GetDmaWrCommands() == expectedDmaWrCommands);
    CHECK(scheduler.GetMceCommands() == expectedMceCommands);
    CHECK(scheduler.GetPleCommands() == expectedPleCommands);
}

TEST_CASE("Cascading/Scheduler/MultipleNonCascadedLayers")
{
    //       IfmS             WgtS         PleL/MceS/PleS/OfmS        IfmS             WgtS         PleL/MceS/PleS/OfmS
    //
    //       +----------+                      +----------+           +----------+                      +----------+
    //      /          /|          +-+        /          /|          /          /|          +-+        /          /|
    //     /          / +         / /|       /          / |         /          / +         / /|       /          / |
    //    /          / /|        / / +      /          /  |        /          / /|        / / +      /          /  |
    //   +----------+ / +       / / /      +----------+   +       +----------+ / +       / / /      +----------+   +
    //   |          |/ /|      / / /       |          |  /|       |          |/ /|      / / /       |          |  /|
    //   +----------+ / +     / / /        |          | / |       +----------+ / +     / / /        |          | / |
    //   |          |/ /|    +-+ /         |          |/  |       |          |/ /|    +-+ /         |          |/  |
    //   +----------+ / +    | |/          +----------+   +       +----------+ / +    | |/          +----------+   +
    //   |          |/ /|    +-+           |          |  /|       |          |/ /|    +-+           |          |  /|
    //   +----------+ / +                  |          | / |       +----------+ / +                  |          | / |
    //   |          |/ /|                  |          |/  |       |          |/ /|                  |          |/  |
    //   +----------+ / +                  +----------+   +       +----------+ / +                  +----------+   +
    //   |          |/ /                   |          |  /        |          |/ /                   |          |  /
    //   +----------+ /                    |          | /         +----------+ /                    |          | /
    //   |          |/                     |          |/          |          |/                     |          |/
    //   +----------+                      +----------+           +----------+                      +----------+
    //
    IfmSDesc ifms{};
    ifms.fmData.tile.numSlots = 4;

    WgtSDesc wgts{};
    wgts.tile.numSlots = 2;

    PleSDesc ples{};
    ples.ofmTile.numSlots = 2;

    OfmSDesc ofms{};
    ofms.fmData.tile.numSlots = 2;

    std::vector<AgentDescAndDeps> multipleNonCascadedLayersCmdStream{
        AgentDescAndDeps{
            AgentDesc(6, ifms),
            {
                /*.readDependencies =*/{},
                /*.writeDependencies =*/{ { { 3, { 3, 6 }, { 1, 2 }, 1 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, wgts),
            {
                /*.readDependencies =*/{},
                /*.writeDependencies =*/{ { { 2, { 3, 1 }, { 3, 1 }, 0 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, PleLDesc{}),
            {
                /*.readDependencies =*/{},
                /*.writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(3, MceSDesc{}),
            {
                /*.readDependencies =*/
                { {
                    { 3, { 6, 3 }, { 2, 1 }, 1 },
                    { 2, { 1, 3 }, { 1, 3 }, 0 },
                } },
                /*.writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(3, ples),
            {
                /*.readDependencies =*/
                { {
                    { 1, { 1, 1 }, { 1, 1 }, 0 },
                    { 2, { 1, 3 }, { 1, 3 }, 0 },
                } },
                /*.writeDependencies =*/{ { { 1, { 1, 1 }, { 1, 1 }, 0 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(3, ofms),
            {
                /*.readDependencies =*/{ { { 1, { 1, 1 }, { 1, 1 }, 0 } } },
                /*.writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(6, ifms),
            {
                /*.readDependencies =*/
                { {
                    { 3, { 3, 6 }, { 3, 6 }, 0 },
                    { 1, { 3, 6 }, { 1, 2 }, 0 },
                } },
                /*.writeDependencies =*/{ { { 3, { 3, 6 }, { 1, 2 }, 1 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, wgts),
            {
                /*.readDependencies =*/{},
                /*.writeDependencies =*/{ { { 2, { 3, 1 }, { 3, 1 }, 0 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, PleLDesc{}),
            {
                /*.readDependencies =*/{},
                /*.writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(3, MceSDesc{}),
            {
                /*.readDependencies =*/
                { {
                    { 3, { 6, 3 }, { 2, 1 }, 1 },
                    { 2, { 1, 3 }, { 1, 3 }, 0 },
                } },
                /*.writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(3, ples),
            {
                /*.readDependencies =*/
                { {
                    { 1, { 1, 1 }, { 1, 1 }, 0 },
                    { 2, { 1, 3 }, { 1, 3 }, 0 },
                } },
                /*.writeDependencies =*/{ { { 1, { 1, 1 }, { 1, 1 }, 0 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(3, ofms),
            {
                /*.readDependencies =*/{ { { 1, { 1, 1 }, { 1, 1 }, 0 } } },
                /*.writeDependencies =*/{},
            },
        },
    };

    const std::vector<Command> expectedDmaRdCommands{
        // clang-format off
        Command { CommandType::LoadIfmStripe, 0, 0, 0 },
        Command { CommandType::LoadIfmStripe, 0, 1, 0 },
        Command { CommandType::LoadIfmStripe, 0, 2, 0 },
        Command { CommandType::LoadWgtStripe, 1, 0, 0 },
        Command { CommandType::LoadPleCode, 2, 0, 0 },
        Command { CommandType::LoadIfmStripe, 0, 3, 0 },
        Command { CommandType::WaitForAgent, 3, 0, 0 },
        Command { CommandType::LoadIfmStripe, 0, 4, 0 },
        Command { CommandType::WaitForAgent, 3, 1, 0 },
        Command { CommandType::LoadIfmStripe, 0, 5, 0 },
        Command { CommandType::WaitForAgent, 3, 2, 0 },
        Command { CommandType::WaitForAgent, 5, 0, 0 },
        Command { CommandType::LoadIfmStripe, 6, 0, 0 },
        Command { CommandType::LoadIfmStripe, 6, 1, 0 },
        Command { CommandType::WaitForAgent, 5, 1, 0 },
        Command { CommandType::LoadIfmStripe, 6, 2, 0 },
        Command { CommandType::LoadWgtStripe, 7, 0, 0 },
        Command { CommandType::LoadPleCode, 8, 0, 0 },
        Command { CommandType::LoadIfmStripe, 6, 3, 0 },
        Command { CommandType::WaitForAgent, 9, 0, 0 },
        Command { CommandType::WaitForAgent, 5, 2, 0 },
        Command { CommandType::LoadIfmStripe, 6, 4, 0 },
        Command { CommandType::WaitForAgent, 9, 1, 0 },
        Command { CommandType::LoadIfmStripe, 6, 5, 0 }
        // clang-format on
    };
    const std::vector<Command> expectedDmaWrCommands{
        // clang-format off
        Command { CommandType::WaitForAgent, 4, 0, 0 },
        Command { CommandType::StoreOfmStripe, 5, 0, 0 },
        Command { CommandType::WaitForAgent, 4, 1, 0 },
        Command { CommandType::StoreOfmStripe, 5, 1, 0 },
        Command { CommandType::WaitForAgent, 4, 2, 0 },
        Command { CommandType::StoreOfmStripe, 5, 2, 0 },
        Command { CommandType::WaitForAgent, 10, 0, 0 },
        Command { CommandType::StoreOfmStripe, 11, 0, 0 },
        Command { CommandType::WaitForAgent, 10, 1, 0 },
        Command { CommandType::StoreOfmStripe, 11, 1, 0 },
        Command { CommandType::WaitForAgent, 10, 2, 0 },
        Command { CommandType::StoreOfmStripe, 11, 2,  0}
        // clang-format on
    };

    const std::vector<Command> expectedMceCommands{
        // clang-format off
        Command { CommandType::ProgramMceStripe, 3, 0, 0 },
        Command { CommandType::WaitForAgent, 0, 2, 0 },
        Command { CommandType::WaitForAgent, 1, 0, 0 },
        Command { CommandType::StartMceStripe, 3, 0, 0 },
        Command { CommandType::ProgramMceStripe, 3, 1, 0 },
        Command { CommandType::WaitForAgent, 0, 4, 0 },
        Command { CommandType::StartMceStripe, 3, 1, 0 },
        Command { CommandType::ProgramMceStripe, 3, 2, 0 },
        Command { CommandType::WaitForAgent, 0, 5, 0 },
        Command { CommandType::StartMceStripe, 3, 2, 0 },
        Command { CommandType::ProgramMceStripe, 9, 0, 0 },
        Command { CommandType::WaitForAgent, 6, 2, 0 },
        Command { CommandType::WaitForAgent, 7, 0, 0 },
        Command { CommandType::StartMceStripe, 9, 0, 0 },
        Command { CommandType::ProgramMceStripe, 9, 1, 0 },
        Command { CommandType::WaitForAgent, 6, 4, 0 },
        Command { CommandType::StartMceStripe, 9, 1, 0 },
        Command { CommandType::ProgramMceStripe, 9, 2, 0 },
        Command { CommandType::WaitForAgent, 6, 5, 0 },
        Command { CommandType::StartMceStripe, 9, 2, 0 },
        // clang-format on
    };

    const std::vector<Command> expectedPleCommands{
        // clang-format off
        Command { CommandType::WaitForAgent, 2, 0, 0 },
        Command { CommandType::StartPleStripe, 4, 0, 0 },
        Command { CommandType::StartPleStripe, 4, 1, 0 },
        Command { CommandType::WaitForAgent, 5, 0, 0 },
        Command { CommandType::StartPleStripe, 4, 2, 0 },
        Command { CommandType::WaitForAgent, 8, 0, 0 },
        Command { CommandType::StartPleStripe, 10, 0, 0 },
        Command { CommandType::StartPleStripe, 10, 1, 0 },
        Command { CommandType::WaitForAgent, 11, 0, 0 },
        Command { CommandType::StartPleStripe, 10, 2, 0 }
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(multipleNonCascadedLayersCmdStream, debuggingContext);
    scheduler.Schedule();

    CHECK(scheduler.GetDmaRdCommands() == expectedDmaRdCommands);
    CHECK(scheduler.GetDmaWrCommands() == expectedDmaWrCommands);
    CHECK(scheduler.GetMceCommands() == expectedMceCommands);
    CHECK(scheduler.GetPleCommands() == expectedPleCommands);
}

TEST_CASE("Cascading/Scheduler/Strategy1Cascade")
{
    //        IfmS             WgtS             MceS                PleL/PleS          WgtS         PleL/MceS/PleS/OfmS
    //      (load x3)        (load x1)       (xyz order)         (all mce stripes)   (load x1)        (xyz order)
    //                                          +----------+          +----------+                       +----------+
    //                                         /          /|         /          /|                      /          /|
    //       +----------+                     +----------+ |        /          / |                     +----------+ |
    //      /          /|          +-+       /          /| |       /          /  |          +-+       /          /| +
    //     /          / +         / /|      +----------+ | +      /          /   |         / /|      +----------+ |/|
    //    /          / /|        +-+ +     /          /| |/|     /          /    |        +-+ +     /          /| + |
    //   +----------+ / +       / /|/     +----------+ | + |    +----------+     |       / /|/     +----------+ |/| +
    //   |          |/ /|      +-+ +      |          | |/| |    |          |     |      +-+ +      |          | + |/|
    //   +----------+ / +     / /|/       |          | + | +    |          |     |     / /|/       |          |/| + |
    //   |          |/ /|    +-+ +        |          |/| |/|    |          |     |    +-+ +        +----------+ |/| +
    //   +----------+ / +    | |/         +----------+ | + |    |          |     |    | |/         |          | + |/|
    //   |          |/ /|    +-+          |          | |/| |    |          |     |    +-+          |          |/| + |
    //   +----------+ / +                 |          | + | +    |          |     +                 +----------+ |/| +
    //   |          |/ /|                 |          |/| |/     |          |    /                  |          | + |/
    //   +----------+ / +                 +----------+ | +      |          |   /                   |          |/| +
    //   |          |/ /                  |          | |/       |          |  /                    +----------+ |/
    //   +----------+ /                   |          | +        |          | /                     |          | +
    //   |          |/                    |          |/         |          |/                      |          |/
    //   +----------+                     +----------+          +----------+                       +----------+
    //
    IfmSDesc ifms{};
    ifms.fmData.tile.numSlots = 4;

    WgtSDesc wgts{};
    wgts.tile.numSlots = 2;

    PleSDesc ples{};
    ples.ofmTile.numSlots = 2;

    OfmSDesc ofms{};
    ofms.fmData.tile.numSlots = 2;

    std::vector<AgentDescAndDeps> strategy1CascadeCmdStream{
        AgentDescAndDeps{
            AgentDesc(18, ifms),
            {
                /*.readDependencies =*/{},
                /*.writeDependencies =*/{ { { 3, { 3, 6 }, { 1, 2 }, 1 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(3, wgts),
            {
                /*.readDependencies =*/{},
                /*.writeDependencies =*/{ { { 2, { 3, 1 }, { 3, 1 }, 0 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, PleLDesc{}),
            {
                /*.readDependencies =*/{},
                /*.writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(9, MceSDesc{}),
            {
                /*.readDependencies =*/
                { {
                    { 3, { 6, 3 }, { 2, 1 }, 1 },
                    { 2, { 1, 3 }, { 1, 3 }, 0 },
                } },
                /*.writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, ples),
            {
                /*.readDependencies =*/
                { {
                    { 1, { 9, 1 }, { 9, 1 }, 0 },
                    { 2, { 1, 1 }, { 1, 1 }, 0 },
                } },
                /*.writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(3, wgts),
            {
                /*.readDependencies =*/{},
                /*.writeDependencies =*/{ { { 2, { 4, 1 }, { 4, 1 }, 0 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, PleLDesc{}),
            {
                /*.readDependencies =*/{},
                /*.writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(12, MceSDesc{}),
            {
                /*.readDependencies =*/
                { {
                    { 3, { 1, 12 }, { 1, 12 }, 0 },
                    { 2, { 1, 4 }, { 1, 4 }, 0 },
                } },
                /*.writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(12, ples),
            {
                /*.readDependencies =*/
                { {
                    { 1, { 1, 1 }, { 1, 1 }, 0 },
                    { 2, { 1, 12 }, { 1, 12 }, 0 },
                } },
                /*.writeDependencies =*/{ { { 1, { 1, 1 }, { 1, 1 }, 0 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(12, ofms),
            {
                /*.readDependencies =*/{ { { 1, { 1, 1 }, { 1, 1 }, 0 } } },
                /*.writeDependencies =*/{},
            },
        },
    };

    const std::vector<Command> expectedDmaRdCommands{
        // clang-format off
        Command { CommandType::LoadIfmStripe, 0, 0, 0 },
        Command { CommandType::LoadIfmStripe, 0, 1, 0 },
        Command { CommandType::LoadIfmStripe, 0, 2, 0 },
        Command { CommandType::LoadWgtStripe, 1, 0, 0 },
        Command { CommandType::LoadIfmStripe, 0, 3, 0 },
        Command { CommandType::WaitForAgent, 3, 0, 0 },
        Command { CommandType::LoadIfmStripe, 0, 4, 0 },
        Command { CommandType::WaitForAgent, 3, 1, 0 },
        Command { CommandType::LoadIfmStripe, 0, 5, 0 },
        Command { CommandType::LoadIfmStripe, 0, 6, 0 },
        Command { CommandType::WaitForAgent, 3, 2, 0 },
        Command { CommandType::LoadIfmStripe, 0, 7, 0 },
        Command { CommandType::LoadIfmStripe, 0, 8, 0 },
        Command { CommandType::LoadWgtStripe, 1, 1, 0 },
        Command { CommandType::LoadIfmStripe, 0, 9, 0 },
        Command { CommandType::WaitForAgent, 3, 3, 0 },
        Command { CommandType::LoadIfmStripe, 0, 10, 0 },
        Command { CommandType::WaitForAgent, 3, 4, 0 },
        Command { CommandType::LoadIfmStripe, 0, 11, 0 },
        Command { CommandType::LoadIfmStripe, 0, 12, 0 },
        Command { CommandType::WaitForAgent, 3, 5, 0 },
        Command { CommandType::LoadIfmStripe, 0, 13, 0 },
        Command { CommandType::LoadIfmStripe, 0, 14, 0 },
        Command { CommandType::WaitForAgent, 3, 2, 0 },
        Command { CommandType::LoadWgtStripe, 1, 2, 0 },
        Command { CommandType::LoadIfmStripe, 0, 15, 0 },
        Command { CommandType::WaitForAgent, 3, 6, 0 },
        Command { CommandType::LoadIfmStripe, 0, 16, 0 },
        Command { CommandType::WaitForAgent, 3, 7, 0 },
        Command { CommandType::LoadIfmStripe, 0, 17, 0 },
        Command { CommandType::LoadPleCode, 2, 0, 0 },
        Command { CommandType::LoadWgtStripe, 5, 0, 0 },
        Command { CommandType::LoadPleCode, 6, 0, 0 },
        Command { CommandType::LoadWgtStripe, 5, 1, 0 },
        Command { CommandType::WaitForAgent, 7, 3, 0 },
        Command { CommandType::LoadWgtStripe, 5, 2, 0 }
        // clang-format on
    };
    const std::vector<Command> expectedDmaWrCommands{
        // clang-format off
        Command { CommandType::WaitForAgent, 8, 0, 0 },
        Command { CommandType::StoreOfmStripe, 9, 0, 0 },
        Command { CommandType::WaitForAgent, 8, 1, 0 },
        Command { CommandType::StoreOfmStripe, 9, 1, 0 },
        Command { CommandType::WaitForAgent, 8, 2, 0 },
        Command { CommandType::StoreOfmStripe, 9, 2, 0 },
        Command { CommandType::WaitForAgent, 8, 3, 0 },
        Command { CommandType::StoreOfmStripe, 9, 3, 0 },
        Command { CommandType::WaitForAgent, 8, 4, 0 },
        Command { CommandType::StoreOfmStripe, 9, 4, 0 },
        Command { CommandType::WaitForAgent, 8, 5, 0 },
        Command { CommandType::StoreOfmStripe, 9, 5, 0 },
        Command { CommandType::WaitForAgent, 8, 6, 0 },
        Command { CommandType::StoreOfmStripe, 9, 6, 0 },
        Command { CommandType::WaitForAgent, 8, 7, 0 },
        Command { CommandType::StoreOfmStripe, 9, 7, 0 },
        Command { CommandType::WaitForAgent, 8, 8, 0 },
        Command { CommandType::StoreOfmStripe, 9, 8, 0 },
        Command { CommandType::WaitForAgent, 8, 9, 0 },
        Command { CommandType::StoreOfmStripe, 9, 9, 0 },
        Command { CommandType::WaitForAgent, 8, 10, 0 },
        Command { CommandType::StoreOfmStripe, 9, 10, 0 },
        Command { CommandType::WaitForAgent, 8, 11, 0 },
        Command { CommandType::StoreOfmStripe, 9, 11,  0}
        // clang-format on
    };

    const std::vector<Command> expectedMceCommands{
        // clang-format off
        Command { CommandType::ProgramMceStripe, 3, 0, 0 },
        Command { CommandType::WaitForAgent, 0, 2, 0 },
        Command { CommandType::WaitForAgent, 1, 0, 0 },
        Command { CommandType::StartMceStripe, 3, 0, 0 },
        Command { CommandType::ProgramMceStripe, 3, 1, 0 },
        Command { CommandType::WaitForAgent, 0, 4, 0 },
        Command { CommandType::StartMceStripe, 3, 1, 0 },
        Command { CommandType::ProgramMceStripe, 3, 2, 0 },
        Command { CommandType::WaitForAgent, 0, 5, 0 },
        Command { CommandType::StartMceStripe, 3, 2, 0 },
        Command { CommandType::ProgramMceStripe, 3, 3, 0 },
        Command { CommandType::WaitForAgent, 0, 8, 0 },
        Command { CommandType::WaitForAgent, 1, 1, 0 },
        Command { CommandType::StartMceStripe, 3, 3, 0 },
        Command { CommandType::ProgramMceStripe, 3, 4, 0 },
        Command { CommandType::WaitForAgent, 0, 10, 0 },
        Command { CommandType::StartMceStripe, 3, 4, 0 },
        Command { CommandType::ProgramMceStripe, 3, 5, 0 },
        Command { CommandType::WaitForAgent, 0, 11, 0 },
        Command { CommandType::StartMceStripe, 3, 5, 0 },
        Command { CommandType::ProgramMceStripe, 3, 6, 0 },
        Command { CommandType::WaitForAgent, 0, 14, 0 },
        Command { CommandType::WaitForAgent, 1, 2, 0 },
        Command { CommandType::StartMceStripe, 3, 6, 0 },
        Command { CommandType::ProgramMceStripe, 3, 7, 0 },
        Command { CommandType::WaitForAgent, 0, 16, 0 },
        Command { CommandType::StartMceStripe, 3, 7, 0 },
        Command { CommandType::ProgramMceStripe, 3, 8, 0 },
        Command { CommandType::WaitForAgent, 0, 17, 0 },
        Command { CommandType::StartMceStripe, 3, 8, 0 },
        Command { CommandType::ProgramMceStripe, 7, 0, 0 },
        Command { CommandType::WaitForAgent, 4, 0, 0 },
        Command { CommandType::WaitForAgent, 5, 0, 0 },
        Command { CommandType::StartMceStripe, 7, 0, 0 },
        Command { CommandType::ProgramMceStripe, 7, 1, 0 },
        Command { CommandType::StartMceStripe, 7, 1, 0 },
        Command { CommandType::ProgramMceStripe, 7, 2, 0 },
        Command { CommandType::StartMceStripe, 7, 2, 0 },
        Command { CommandType::ProgramMceStripe, 7, 3, 0 },
        Command { CommandType::StartMceStripe, 7, 3, 0 },
        Command { CommandType::ProgramMceStripe, 7, 4, 0 },
        Command { CommandType::WaitForAgent, 5, 1, 0 },
        Command { CommandType::StartMceStripe, 7, 4, 0 },
        Command { CommandType::ProgramMceStripe, 7, 5, 0 },
        Command { CommandType::StartMceStripe, 7, 5, 0 },
        Command { CommandType::ProgramMceStripe, 7, 6, 0 },
        Command { CommandType::StartMceStripe, 7, 6, 0 },
        Command { CommandType::ProgramMceStripe, 7, 7, 0 },
        Command { CommandType::StartMceStripe, 7, 7, 0 },
        Command { CommandType::ProgramMceStripe, 7, 8, 0 },
        Command { CommandType::WaitForAgent, 5, 2, 0 },
        Command { CommandType::StartMceStripe, 7, 8, 0 },
        Command { CommandType::ProgramMceStripe, 7, 9, 0 },
        Command { CommandType::StartMceStripe, 7, 9, 0 },
        Command { CommandType::ProgramMceStripe, 7, 10, 0 },
        Command { CommandType::StartMceStripe, 7, 10, 0 },
        Command { CommandType::ProgramMceStripe, 7, 11, 0 },
        Command { CommandType::StartMceStripe, 7, 11,  0}
        // clang-format on
    };

    const std::vector<Command> expectedPleCommands{
        // clang-format off
        Command { CommandType::WaitForAgent, 2, 0, 0 },
        Command { CommandType::StartPleStripe, 4, 0, 0 },
        Command { CommandType::WaitForAgent, 6, 0, 0 },
        Command { CommandType::StartPleStripe, 8, 0, 0 },
        Command { CommandType::StartPleStripe, 8, 1, 0 },
        Command { CommandType::WaitForAgent, 9, 0, 0 },
        Command { CommandType::StartPleStripe, 8, 2, 0 },
        Command { CommandType::WaitForAgent, 9, 1, 0 },
        Command { CommandType::StartPleStripe, 8, 3, 0 },
        Command { CommandType::WaitForAgent, 9, 2, 0 },
        Command { CommandType::StartPleStripe, 8, 4, 0 },
        Command { CommandType::WaitForAgent, 9, 3, 0 },
        Command { CommandType::StartPleStripe, 8, 5, 0 },
        Command { CommandType::WaitForAgent, 9, 4, 0 },
        Command { CommandType::StartPleStripe, 8, 6, 0 },
        Command { CommandType::WaitForAgent, 9, 5, 0 },
        Command { CommandType::StartPleStripe, 8, 7, 0 },
        Command { CommandType::WaitForAgent, 9, 6, 0 },
        Command { CommandType::StartPleStripe, 8, 8, 0 },
        Command { CommandType::WaitForAgent, 9, 7, 0 },
        Command { CommandType::StartPleStripe, 8, 9, 0 },
        Command { CommandType::WaitForAgent, 9, 8, 0 },
        Command { CommandType::StartPleStripe, 8, 10, 0 },
        Command { CommandType::WaitForAgent, 9, 9, 0 },
        Command { CommandType::StartPleStripe, 8, 11, 0 }
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(strategy1CascadeCmdStream, debuggingContext);
    scheduler.Schedule();

    CHECK(scheduler.GetDmaRdCommands() == expectedDmaRdCommands);
    CHECK(scheduler.GetDmaWrCommands() == expectedDmaWrCommands);
    CHECK(scheduler.GetMceCommands() == expectedMceCommands);
    CHECK(scheduler.GetPleCommands() == expectedPleCommands);
}

TEST_CASE("Cascading/Scheduler/Strategy0Cascade")
{
    //        IfmS               WgtS                PleL                     MceS/PleS              WgtS           PleL/MceS/PleS/OfmS
    //      (load x1)          (load x1)                                     (xyz order)           (load x1)          (xyz order)
    //                                             +----------+              +----------+                            +----------+
    //                                            /          /|             /          /|                           /          /|
    //       +----------+                        /          / |            /          / |                          /          / |
    //      /          /|            +-+        /          /  |           /          /  |             +-+         /          /  |
    //     /          / +           / /|       /          /   +          /          /   +            / /|        /          /   +
    //    /          / /|          / / +      /          /    |         /          /   /|           / / +       /          /   /|
    //   +----------+ / +         / / /      +----------+     |        +----------+   / |          / / /       +----------+   / |
    //   |          |/ /|        / / /       |          |     |        |          |  /  |         / / /        |          |  /  |
    //   +----------+ / +       / / /        |          |     +        |          | /   +        / / /         |          | /   +
    //   |          |/ /|      +-+ /         |          |    /|        |          |/   /|       +-+ /          |          |/   /|
    //   +----------+ / +      | |/          |          |   / |        +----------+   / |       | |/           +----------+   / |
    //   |          |/ /|      +-+           |          |  /  |        |          |  /  |       +-+            |          |  /  |
    //   +----------+ / +                    |          | /   +        |          | /   +                      |          | /   +
    //   |          |/ /|                    |          |/   /|        |          |/   /|                      |          |/   /|
    //   +----------+ / +                    +----------+   / |        +----------+   / |                      +----------+   / |
    //   |          |/ /|                    |          |  /  |        |          |  /  |                      |          |  /  |
    //   +----------+ / +                    |          | /   +        |          | /   +                      |          | /   +
    //   |          |/ /|                    |          |/   /|        |          |/   /|                      |          |/   /
    //   +----------+ / +                    +----------+   / +        +----------+   / +                      +----------+   /
    //   |          |/ /|                    |          |  / /         |          |  / /                       |          |  /
    //   +----------+ / +                    |          | / /          |          | / /                        |          | /
    //   |          |/ /                     |          |/ /           |          |/ /                         |          |/
    //   +----------+ /                      +----------+ /            +----------+ /                          +----------+
    //   |          |/                       |          |/             |          |/
    //   +----------+                        +----------+              +----------+
    //
    IfmSDesc ifms{};
    ifms.fmData.tile.numSlots = 4;

    WgtSDesc wgts{};
    wgts.tile.numSlots = 2;

    PleSDesc ples{};
    ples.ofmTile.numSlots = 2;

    OfmSDesc ofms{};
    ofms.fmData.tile.numSlots = 2;

    std::vector<AgentDescAndDeps> strategy0CascadeCmdStream{
        AgentDescAndDeps{
            AgentDesc(9, ifms),
            {
                /*.readDependencies =*/{},
                /*.writeDependencies =*/{ { { 3, { 5, 9 }, { 1, 2 }, 1 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, wgts),
            {
                /*.readDependencies =*/{},
                /*.writeDependencies =*/{ { { 2, { 5, 1 }, { 5, 1 }, 0 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(4, PleLDesc{}),
            {
                /*.readDependencies =*/{},
                // Wait until the second PleS has finished its stripe before overwriting the PLE kernel code in SRAM,
                // which it might still be using (PleS also does the code uDMA).
                /*.writeDependencies =*/{ { { 6, { 1, 1 }, { 1, 1 }, 0 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(5, MceSDesc{}),
            {
                /*.readDependencies =*/
                { {
                    { 3, { 9, 5 }, { 2, 1 }, 1 },
                    { 2, { 1, 5 }, { 1, 5 }, 0 },
                } },
                /*.writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(5, ples),
            {
                /*.readDependencies =*/
                { {
                    { 1, { 1, 1 }, { 1, 1 }, 0 },
                    { 2, { 4, 5 }, { 1, 1 }, -1 },
                } },
                /*.writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, wgts),
            {
                /*.readDependencies =*/{},
                /*.writeDependencies =*/{ { { 2, { 4, 1 }, { 4, 1 }, 0 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(4, PleLDesc{}),
            {
                // Wait until the first PleS has finished its stripe before overwriting the PLE kernel code in SRAM,
                // which it might still be using (PleS also does the code uDMA).
                /*.readDependencies =*/{ { { 2, { 5, 4 }, { 1, 1 }, 1 } } },
                /*.writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(4, MceSDesc{}),
            {
                /*.readDependencies =*/
                { {
                    { 3, { 5, 4 }, { 1, 1 }, 1 },
                    { 2, { 1, 4 }, { 1, 4 }, 0 },
                } },
                /*.writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(4, ples),
            {
                /*.readDependencies =*/
                { {
                    { 1, { 1, 1 }, { 1, 1 }, 0 },
                    { 2, { 1, 1 }, { 1, 1 }, 0 },
                } },
                /*.writeDependencies =*/{ { { 1, { 1, 1 }, { 1, 1 }, 0 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(4, ofms),
            {
                /*.readDependencies =*/{ { { 1, { 1, 1 }, { 1, 1 }, 0 } } },
                /*.writeDependencies =*/{},
            },
        },
    };

    const std::vector<Command> expectedDmaRdCommands{
        // clang-format off
        Command { CommandType::LoadIfmStripe, 0, 0, 0 },
        Command { CommandType::LoadIfmStripe, 0, 1, 0 },
        Command { CommandType::LoadIfmStripe, 0, 2, 0 },
        Command { CommandType::LoadWgtStripe, 1, 0, 0 },
        Command { CommandType::LoadPleCode, 2, 0, 0 },
        Command { CommandType::LoadIfmStripe, 0, 3, 0 },
        Command { CommandType::WaitForAgent, 3, 0, 0 },
        Command { CommandType::LoadIfmStripe, 0, 4, 0 },
        Command { CommandType::LoadWgtStripe, 5, 0, 0 },
        Command { CommandType::WaitForAgent, 4, 1, 0 },
        Command { CommandType::LoadPleCode, 6, 0, 0 },
        Command { CommandType::WaitForAgent, 3, 1, 0 },
        Command { CommandType::LoadIfmStripe, 0, 5, 0 },
        Command { CommandType::LoadIfmStripe, 0, 6, 0 },
        Command { CommandType::WaitForAgent, 8, 0, 0 },
        Command { CommandType::LoadPleCode, 2, 1, 0 },
        Command { CommandType::WaitForAgent, 4, 2, 0 },
        Command { CommandType::LoadPleCode, 6, 1, 0 },
        Command { CommandType::WaitForAgent, 3, 2, 0 },
        Command { CommandType::LoadIfmStripe, 0, 7, 0 },
        Command { CommandType::LoadIfmStripe, 0, 8, 0 },
        Command { CommandType::WaitForAgent, 8, 1, 0 },
        Command { CommandType::LoadPleCode, 2, 2, 0 },
        Command { CommandType::WaitForAgent, 4, 3, 0 },
        Command { CommandType::LoadPleCode, 6, 2, 0 },
        Command { CommandType::WaitForAgent, 8, 2, 0 },
        Command { CommandType::LoadPleCode, 2, 3, 0 },
        Command { CommandType::WaitForAgent, 4, 4, 0 },
        Command { CommandType::LoadPleCode, 6, 3, 0 }
        // clang-format on
    };
    const std::vector<Command> expectedDmaWrCommands{
        // clang-format off
        Command { CommandType::WaitForAgent, 8, 0, 0 },
        Command { CommandType::StoreOfmStripe, 9, 0, 0 },
        Command { CommandType::WaitForAgent, 8, 1, 0 },
        Command { CommandType::StoreOfmStripe, 9, 1, 0 },
        Command { CommandType::WaitForAgent, 8, 2, 0 },
        Command { CommandType::StoreOfmStripe, 9, 2, 0 },
        Command { CommandType::WaitForAgent, 8, 3, 0 },
        Command { CommandType::StoreOfmStripe, 9, 3,  0}
        // clang-format on
    };

    const std::vector<Command> expectedMceCommands{
        // clang-format off
        Command { CommandType::ProgramMceStripe, 3, 0, 0 },
        Command { CommandType::WaitForAgent, 0, 2, 0 },
        Command { CommandType::WaitForAgent, 1, 0, 0 },
        Command { CommandType::StartMceStripe, 3, 0, 0 },
        Command { CommandType::ProgramMceStripe, 3, 1, 0 },
        Command { CommandType::WaitForAgent, 0, 4, 0 },
        Command { CommandType::StartMceStripe, 3, 1, 0 },
        Command { CommandType::ProgramMceStripe, 7, 0, 0 },
        Command { CommandType::WaitForAgent, 4, 1, 0 },
        Command { CommandType::WaitForAgent, 5, 0, 0 },
        Command { CommandType::StartMceStripe, 7, 0, 0 },
        Command { CommandType::ProgramMceStripe, 3, 2, 0 },
        Command { CommandType::WaitForAgent, 0, 6, 0 },
        Command { CommandType::StartMceStripe, 3, 2, 0 },
        Command { CommandType::ProgramMceStripe, 7, 1, 0 },
        Command { CommandType::WaitForAgent, 4, 2, 0 },
        Command { CommandType::StartMceStripe, 7, 1, 0 },
        Command { CommandType::ProgramMceStripe, 3, 3, 0 },
        Command { CommandType::WaitForAgent, 0, 8, 0 },
        Command { CommandType::StartMceStripe, 3, 3, 0 },
        Command { CommandType::ProgramMceStripe, 7, 2, 0 },
        Command { CommandType::WaitForAgent, 4, 3, 0 },
        Command { CommandType::StartMceStripe, 7, 2, 0 },
        Command { CommandType::ProgramMceStripe, 3, 4, 0 },
        Command { CommandType::StartMceStripe, 3, 4, 0 },
        Command { CommandType::ProgramMceStripe, 7, 3, 0 },
        Command { CommandType::WaitForAgent, 4, 4, 0 },
        Command { CommandType::StartMceStripe, 7, 3,  0}
        // clang-format on
    };

    const std::vector<Command> expectedPleCommands{
        // clang-format off
        Command { CommandType::WaitForAgent, 2, 0, 0 },
        Command { CommandType::StartPleStripe, 4, 0, 0 },
        Command { CommandType::StartPleStripe, 4, 1, 0 },
        Command { CommandType::WaitForAgent, 6, 0, 0 },
        Command { CommandType::StartPleStripe, 8, 0, 0 },
        Command { CommandType::WaitForAgent, 2, 1, 0 },
        Command { CommandType::StartPleStripe, 4, 2, 0 },
        Command { CommandType::WaitForAgent, 6, 1, 0 },
        Command { CommandType::StartPleStripe, 8, 1, 0 },
        Command { CommandType::WaitForAgent, 2, 2, 0 },
        Command { CommandType::StartPleStripe, 4, 3, 0 },
        Command { CommandType::WaitForAgent, 9, 0, 0 },
        Command { CommandType::WaitForAgent, 6, 2, 0 },
        Command { CommandType::StartPleStripe, 8, 2, 0 },
        Command { CommandType::WaitForAgent, 2, 3, 0 },
        Command { CommandType::StartPleStripe, 4, 4, 0 },
        Command { CommandType::WaitForAgent, 9, 1, 0 },
        Command { CommandType::WaitForAgent, 6, 3, 0 },
        Command { CommandType::StartPleStripe, 8, 3, 0 }
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(strategy0CascadeCmdStream, debuggingContext);
    scheduler.Schedule();

    CHECK(scheduler.GetDmaRdCommands() == expectedDmaRdCommands);
    CHECK(scheduler.GetDmaWrCommands() == expectedDmaWrCommands);
    CHECK(scheduler.GetMceCommands() == expectedMceCommands);
    CHECK(scheduler.GetPleCommands() == expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/IfmStreamer/WriteDependencies/FirstTile")
{
    const uint32_t tileSize = 4;
    IfmSDesc ifms{};
    ifms.fmData.tile.numSlots = tileSize;

    WgtSDesc wgts{};
    wgts.tile.numSlots = 2;

    std::vector<AgentDescAndDeps> cmdStream{
        // The first agent in the command stream is dummy, and it is there just
        // to make sure that we don't use agent ID 0. This help to validate
        // that the relative agent id field is properly used by the
        // scheduler function
        AgentDescAndDeps{ AgentDesc(0, IfmSDesc{}), {} },
        AgentDescAndDeps{
            AgentDesc(18, ifms),
            {
                /* .readDependencies =*/{},
                /* .writeDependencies =*/{ { { 3, { 3, 6 }, { 1, 2 }, 1 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(3, wgts),

            {
                /* .readDependencies =*/{},
                /* .writeDependencies =*/{ { { 2, { 9, 3 }, { 3, 1 }, 0 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, PleLDesc{}),

            {
                /* .readDependencies =*/{},
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(9, MceSDesc{}),

            {
                /* .readDependencies =*/
                { {
                    { 3, { 6, 3 }, { 2, 1 }, 1 },
                    { 2, { 3, 9 }, { 1, 3 }, 0 },
                } },
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, OfmSDesc{}),
            {
                /* .readDependencies =*/{ { { 1, { 9, 1 }, { 9, 1 }, 0 } } },
                /* .writeDependencies =*/{},
            },
        },
    };

    const std::vector<Command> expectedDmaWrCommands{
        //clang-format off
        Command{ CommandType::WaitForAgent, 4, 8, 0 }, Command{ CommandType::StoreOfmStripe, 5, 0, 0 }
        //clang-format on
    };

    const std::vector<Command> expectedPleCommands{
        // clang-format off
        // clang-format on
    };

    const std::vector<Command> expectedDmaRdCommands{
        // clang-format off
        Command { CommandType::LoadIfmStripe, 1, 0, 0 },
        Command { CommandType::LoadIfmStripe, 1, 1, 0 },
        Command { CommandType::LoadIfmStripe, 1, 2, 0 },
        Command { CommandType::LoadWgtStripe, 2, 0, 0 },
        Command { CommandType::LoadIfmStripe, 1, 3, 0 },
        Command { CommandType::WaitForAgent, 4, 0, 0 },
        Command { CommandType::LoadIfmStripe, 1, 4, 0 },
        Command { CommandType::WaitForAgent, 4, 1, 0 },
        Command { CommandType::LoadIfmStripe, 1, 5, 0 },
        Command { CommandType::LoadIfmStripe, 1, 6, 0 },
        Command { CommandType::WaitForAgent, 4, 2, 0 },
        Command { CommandType::LoadIfmStripe, 1, 7, 0 },
        Command { CommandType::LoadIfmStripe, 1, 8, 0 },
        Command { CommandType::LoadWgtStripe, 2, 1, 0 },
        Command { CommandType::LoadIfmStripe, 1, 9, 0 },
        Command { CommandType::WaitForAgent, 4, 3, 0 },
        Command { CommandType::LoadIfmStripe, 1, 10, 0 },
        Command { CommandType::WaitForAgent, 4, 4, 0 },
        Command { CommandType::LoadIfmStripe, 1, 11, 0 },
        Command { CommandType::LoadIfmStripe, 1, 12, 0 },
        Command { CommandType::WaitForAgent, 4, 5, 0 },
        Command { CommandType::LoadIfmStripe, 1, 13, 0 },
        Command { CommandType::LoadIfmStripe, 1, 14, 0 },
        Command { CommandType::WaitForAgent, 4, 2, 0 },
        Command { CommandType::LoadWgtStripe, 2, 2, 0 },
        Command { CommandType::LoadIfmStripe, 1, 15, 0 },
        Command { CommandType::WaitForAgent, 4, 6, 0 },
        Command { CommandType::LoadIfmStripe, 1, 16, 0 },
        Command { CommandType::WaitForAgent, 4, 7, 0 },
        Command { CommandType::LoadIfmStripe, 1, 17, 0 }
        // clang-format on
    };

    const std::vector<Command> expectedMceCommands{
        // clang-format off
        Command { CommandType::ProgramMceStripe, 4, 0, 0 },
        Command { CommandType::WaitForAgent, 1, 2, 0 },
        Command { CommandType::WaitForAgent, 2, 0, 0 },
        Command { CommandType::StartMceStripe, 4, 0, 0 },
        Command { CommandType::ProgramMceStripe, 4, 1, 0 },
        Command { CommandType::WaitForAgent, 1, 4, 0 },
        Command { CommandType::StartMceStripe, 4, 1, 0 },
        Command { CommandType::ProgramMceStripe, 4, 2, 0 },
        Command { CommandType::WaitForAgent, 1, 5, 0 },
        Command { CommandType::StartMceStripe, 4, 2, 0 },
        Command { CommandType::ProgramMceStripe, 4, 3, 0 },
        Command { CommandType::WaitForAgent, 1, 8, 0 },
        Command { CommandType::WaitForAgent, 2, 1, 0 },
        Command { CommandType::StartMceStripe, 4, 3, 0 },
        Command { CommandType::ProgramMceStripe, 4, 4, 0 },
        Command { CommandType::WaitForAgent, 1, 10, 0 },
        Command { CommandType::StartMceStripe, 4, 4, 0 },
        Command { CommandType::ProgramMceStripe, 4, 5, 0 },
        Command { CommandType::WaitForAgent, 1, 11, 0 },
        Command { CommandType::StartMceStripe, 4, 5, 0 },
        Command { CommandType::ProgramMceStripe, 4, 6, 0 },
        Command { CommandType::WaitForAgent, 1, 14, 0 },
        Command { CommandType::WaitForAgent, 2, 2, 0 },
        Command { CommandType::StartMceStripe, 4, 6, 0 },
        Command { CommandType::ProgramMceStripe, 4, 7, 0 },
        Command { CommandType::WaitForAgent, 1, 16, 0 },
        Command { CommandType::StartMceStripe, 4, 7, 0 },
        Command { CommandType::ProgramMceStripe, 4, 8, 0 },
        Command { CommandType::WaitForAgent, 1, 17, 0 },
        Command { CommandType::StartMceStripe, 4, 8,  0}
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, debuggingContext);
    scheduler.Schedule();

    CHECK(scheduler.GetDmaRdCommands() == expectedDmaRdCommands);
    CHECK(scheduler.GetDmaWrCommands() == expectedDmaWrCommands);
    CHECK(scheduler.GetMceCommands() == expectedMceCommands);
    CHECK(scheduler.GetPleCommands() == expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/IfmStreamer/WriteDependencies/AfterFirstTile")
{
    const uint32_t tileSize                  = 18;
    const uint32_t relativeAgentIdDependency = 3;
    const uint32_t numStripesTotal           = 18;
    IfmSDesc ifms{};
    ifms.fmData.tile.numSlots = tileSize;

    WgtSDesc wgts{};
    wgts.tile.numSlots = 2;

    std::vector<AgentDescAndDeps> cmdStream{
        // The first agent in the command stream is dummy, and it is there just
        // to make sure that we don't use agent ID 0. This help to validate
        // that the relative agent id field is properly used by the
        // scheduler function
        AgentDescAndDeps{ AgentDesc(0, IfmSDesc{}), {} },
        AgentDescAndDeps{
            AgentDesc(numStripesTotal, ifms),

            {
                /* .readDependencies =*/{},
                /* .writeDependencies =*/{ { { relativeAgentIdDependency, { 3, 6 }, { 1, 2 }, 1 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(3, wgts),

            {
                /* .readDependencies =*/{},
                /* .writeDependencies =*/{ { { 2, { 9, 3 }, { 3, 1 }, 0 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, PleLDesc{}),

            {
                /* .readDependencies =*/{},
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(9, MceSDesc{}),

            {
                /* .readDependencies =*/
                { {
                    { 3, { 6, 3 }, { 2, 1 }, 1 },
                    { 2, { 3, 9 }, { 1, 3 }, 0 },
                } },
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, OfmSDesc{}),
            {
                /* .readDependencies =*/{ { { 1, { 9, 1 }, { 9, 1 }, 0 } } },
                /* .writeDependencies =*/{},
            },
        },
    };

    const std::vector<Command> expectedDmaWrCommands{
        //clang-format off
        Command{ CommandType::WaitForAgent, 4, 8, 0 }, Command{ CommandType::StoreOfmStripe, 5, 0, 0 }
        //clang-format on
    };

    const std::vector<Command> expectedPleCommands{
        // clang-format off
        // clang-format on
    };
    const std::vector<Command> expectedDmaRdCommands{
        // clang-format off
        Command { CommandType::LoadIfmStripe, 1, 0, 0 },
        Command { CommandType::LoadIfmStripe, 1, 1, 0 },
        Command { CommandType::LoadIfmStripe, 1, 2, 0 },
        Command { CommandType::LoadWgtStripe, 2, 0, 0 },
        Command { CommandType::LoadIfmStripe, 1, 3, 0 },
        Command { CommandType::LoadIfmStripe, 1, 4, 0 },
        Command { CommandType::LoadIfmStripe, 1, 5, 0 },
        Command { CommandType::LoadIfmStripe, 1, 6, 0 },
        Command { CommandType::LoadIfmStripe, 1, 7, 0 },
        Command { CommandType::LoadIfmStripe, 1, 8, 0 },
        Command { CommandType::LoadWgtStripe, 2, 1, 0 },
        Command { CommandType::LoadIfmStripe, 1, 9, 0 },
        Command { CommandType::LoadIfmStripe, 1, 10, 0 },
        Command { CommandType::LoadIfmStripe, 1, 11, 0 },
        Command { CommandType::LoadIfmStripe, 1, 12, 0 },
        Command { CommandType::LoadIfmStripe, 1, 13, 0 },
        Command { CommandType::LoadIfmStripe, 1, 14, 0 },
        Command { CommandType::WaitForAgent, 4, 2, 0 },
        Command { CommandType::LoadWgtStripe, 2, 2, 0 },
        Command { CommandType::LoadIfmStripe, 1, 15, 0 },
        Command { CommandType::LoadIfmStripe, 1, 16, 0 },
        Command { CommandType::LoadIfmStripe, 1, 17, 0 }
        // clang-format on
    };

    const std::vector<Command> expectedMceCommands{
        // clang-format off
        Command { CommandType::ProgramMceStripe, 4, 0, 0 },
        Command { CommandType::WaitForAgent, 1, 2, 0 },
        Command { CommandType::WaitForAgent, 2, 0, 0 },
        Command { CommandType::StartMceStripe, 4, 0, 0 },
        Command { CommandType::ProgramMceStripe, 4, 1, 0 },
        Command { CommandType::WaitForAgent, 1, 4, 0 },
        Command { CommandType::StartMceStripe, 4, 1, 0 },
        Command { CommandType::ProgramMceStripe, 4, 2, 0 },
        Command { CommandType::WaitForAgent, 1, 5, 0 },
        Command { CommandType::StartMceStripe, 4, 2, 0 },
        Command { CommandType::ProgramMceStripe, 4, 3, 0 },
        Command { CommandType::WaitForAgent, 1, 8, 0 },
        Command { CommandType::WaitForAgent, 2, 1, 0 },
        Command { CommandType::StartMceStripe, 4, 3, 0 },
        Command { CommandType::ProgramMceStripe, 4, 4, 0 },
        Command { CommandType::WaitForAgent, 1, 10, 0 },
        Command { CommandType::StartMceStripe, 4, 4, 0 },
        Command { CommandType::ProgramMceStripe, 4, 5, 0 },
        Command { CommandType::WaitForAgent, 1, 11, 0 },
        Command { CommandType::StartMceStripe, 4, 5, 0 },
        Command { CommandType::ProgramMceStripe, 4, 6, 0 },
        Command { CommandType::WaitForAgent, 1, 14, 0 },
        Command { CommandType::WaitForAgent, 2, 2, 0 },
        Command { CommandType::StartMceStripe, 4, 6, 0 },
        Command { CommandType::ProgramMceStripe, 4, 7, 0 },
        Command { CommandType::WaitForAgent, 1, 16, 0 },
        Command { CommandType::StartMceStripe, 4, 7, 0 },
        Command { CommandType::ProgramMceStripe, 4, 8, 0 },
        Command { CommandType::WaitForAgent, 1, 17, 0 },
        Command { CommandType::StartMceStripe, 4, 8,  0}
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, debuggingContext);
    scheduler.Schedule();

    CHECK(scheduler.GetDmaRdCommands() == expectedDmaRdCommands);
    CHECK(scheduler.GetDmaWrCommands() == expectedDmaWrCommands);
    CHECK(scheduler.GetMceCommands() == expectedMceCommands);
    CHECK(scheduler.GetPleCommands() == expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/IfmStreamer/WithReadAndWriteDependency/FirstTile")
{
    const uint32_t numStripesTotal = 6;
    const uint32_t tileSize        = 4;
    IfmSDesc ifms{};
    ifms.fmData.tile.numSlots = tileSize;

    WgtSDesc wgts{};
    wgts.tile.numSlots = 2;

    PleSDesc ples{};
    ples.ofmTile.numSlots = 2;

    OfmSDesc ofms{};
    ofms.fmData.tile.numSlots = 2;

    std::vector<AgentDescAndDeps> cmdStream{
        AgentDescAndDeps{
            AgentDesc(3, MceSDesc{}),

            {
                /* .readDependencies =*/
                {},
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(3, ples),

            {
                /* .readDependencies =*/
                { {
                    { 1, { 3, 3 }, { 1, 1 }, 0 },
                } },
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(3, ofms),

            {
                /* .readDependencies =*/{ { { 1, { 3, 3 }, { 1, 1 }, 0 } } },
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(numStripesTotal, ifms),

            {
                /* .readDependencies =*/
                { {
                    { 3, { 3, 6 }, { 3, 6 }, 0 },
                    { 1, { 3, 6 }, { 1, 2 }, 0 },
                } },
                /* .writeDependencies =*/{ { { 3, { 3, 6 }, { 1, 2 }, 1 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, wgts),

            {
                /* .readDependencies =*/{},
                /* .writeDependencies =*/{ { { 2, { 3, 1 }, { 3, 1 }, 0 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, PleLDesc{}),

            {
                /* .readDependencies =*/{},
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(3, MceSDesc{}),

            {
                /* .readDependencies =*/
                { {
                    { 3, { 6, 3 }, { 2, 1 }, 1 },
                    { 2, { 1, 3 }, { 1, 3 }, 0 },
                } },
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, OfmSDesc{}),
            {
                /* .readDependencies =*/{ { { 1, { 3, 1 }, { 3, 1 }, 0 } } },
                /* .writeDependencies =*/{},
            },
        },
    };
    const std::vector<Command> expectedDmaRdCommands{
        // clang-format off
        Command { CommandType::WaitForAgent, 0, 2, 0 },
        Command { CommandType::WaitForAgent, 2, 0, 0 },
        Command { CommandType::LoadIfmStripe, 3, 0, 0 },
        Command { CommandType::LoadIfmStripe, 3, 1, 0 },
        Command { CommandType::WaitForAgent, 2, 1, 0 },
        Command { CommandType::LoadIfmStripe, 3, 2, 0 },
        Command { CommandType::LoadWgtStripe, 4, 0, 0 },
        Command { CommandType::LoadIfmStripe, 3, 3, 0 },
        Command { CommandType::WaitForAgent, 6, 0, 0 },
        Command { CommandType::WaitForAgent, 2, 2, 0 },
        Command { CommandType::LoadIfmStripe, 3, 4, 0 },
        Command { CommandType::WaitForAgent, 6, 1, 0 },
        Command { CommandType::LoadIfmStripe, 3, 5, 0 }
        // clang-format on
    };

    const std::vector<Command> expectedDmaWrCommands{
        //clang-format off
        Command{ CommandType::WaitForAgent, 1, 0, 0 },
        Command{ CommandType::StoreOfmStripe, 2, 0, 0 },
        Command{ CommandType::WaitForAgent, 1, 1, 0 },
        Command{ CommandType::StoreOfmStripe, 2, 1, 0 },
        Command{ CommandType::WaitForAgent, 1, 2, 0 },
        Command{ CommandType::StoreOfmStripe, 2, 2, 0 },
        Command{ CommandType::WaitForAgent, 6, 2, 0 },
        Command{ CommandType::StoreOfmStripe, 7, 0, 0 }
        // clang-format on
    };

    const std::vector<Command> expectedMceCommands{
        // clang-format off
        Command { CommandType::ProgramMceStripe, 0, 0, 0 },
        Command { CommandType::StartMceStripe, 0, 0, 0 },
        Command { CommandType::ProgramMceStripe, 0, 1, 0 },
        Command { CommandType::StartMceStripe, 0, 1, 0 },
        Command { CommandType::ProgramMceStripe, 0, 2, 0 },
        Command { CommandType::StartMceStripe, 0, 2, 0 },
        Command { CommandType::ProgramMceStripe, 6, 0, 0 },
        Command { CommandType::WaitForAgent, 3, 2, 0 },
        Command { CommandType::WaitForAgent, 4, 0, 0 },
        Command { CommandType::StartMceStripe, 6, 0, 0 },
        Command { CommandType::ProgramMceStripe, 6, 1, 0 },
        Command { CommandType::WaitForAgent, 3, 4, 0 },
        Command { CommandType::StartMceStripe, 6, 1, 0 },
        Command { CommandType::ProgramMceStripe, 6, 2, 0 },
        Command { CommandType::WaitForAgent, 3, 5, 0 },
        Command { CommandType::StartMceStripe, 6, 2,  0}
        // clang-format on
    };

    const std::vector<Command> expectedPleCommands{
        // clang-format off
        Command { CommandType::StartPleStripe, 1, 0, 0 },
        Command { CommandType::StartPleStripe, 1, 1, 0 },
        Command { CommandType::StartPleStripe, 1, 2,  0}
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, debuggingContext);
    scheduler.Schedule();

    CHECK(scheduler.GetDmaRdCommands() == expectedDmaRdCommands);
    CHECK(scheduler.GetDmaWrCommands() == expectedDmaWrCommands);
    CHECK(scheduler.GetMceCommands() == expectedMceCommands);
    CHECK(scheduler.GetPleCommands() == expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/WithReadAndWriteDependency/AfterFirstTile")
{
    const uint32_t tileSize        = 4;
    const uint32_t numStripesTotal = 6;
    IfmSDesc ifms{};
    ifms.fmData.tile.numSlots = tileSize;

    WgtSDesc wgts{};
    wgts.tile.numSlots = 2;

    PleSDesc ples{};
    ples.ofmTile.numSlots = 2;

    OfmSDesc ofms{};
    ofms.fmData.tile.numSlots = 2;

    std::vector<AgentDescAndDeps> cmdStream{
        AgentDescAndDeps{
            AgentDesc(3, MceSDesc{}),

            {
                /* .readDependencies =*/
                {},
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(3, ples),

            {
                /* .readDependencies =*/
                { {
                    { 1, { 3, 3 }, { 1, 1 }, 0 },
                } },
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(3, ofms),

            {
                /* .readDependencies =*/{ { { 1, { 3, 3 }, { 1, 1 }, 0 } } },
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(numStripesTotal, ifms),

            {
                /* .readDependencies =*/
                { {
                    { 3, { 3, 6 }, { 3, 6 }, 0 },
                    { 1, { 3, 6 }, { 1, 2 }, 0 },
                } },
                /* .writeDependencies =*/{ { { 3, { 3, 6 }, { 1, 2 }, 1 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, wgts),

            {
                /* .readDependencies =*/{},
                /* .writeDependencies =*/{ { { 2, { 3, 1 }, { 3, 1 }, 0 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, PleLDesc{}),

            {
                /* .readDependencies =*/{},
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(3, MceSDesc{}),

            {
                /* .readDependencies =*/
                { {
                    { 3, { 6, 3 }, { 2, 1 }, 1 },
                    { 2, { 1, 3 }, { 1, 3 }, 0 },
                } },
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, OfmSDesc{}),
            {
                /* .readDependencies =*/{ { { 1, { 3, 1 }, { 3, 1 }, 0 } } },
                /* .writeDependencies =*/{},
            },
        },

    };
    const std::vector<Command> expectedDmaRdCommands{
        // clang-format off
        Command { CommandType::WaitForAgent, 0, 2, 0 },
        Command { CommandType::WaitForAgent, 2, 0, 0 },
        Command { CommandType::LoadIfmStripe, 3, 0, 0 },
        Command { CommandType::LoadIfmStripe, 3, 1, 0 },
        Command { CommandType::WaitForAgent, 2, 1, 0 },
        Command { CommandType::LoadIfmStripe, 3, 2, 0 },
        Command { CommandType::LoadWgtStripe, 4, 0, 0 },
        Command { CommandType::LoadIfmStripe, 3, 3, 0 },
        Command { CommandType::WaitForAgent, 6, 0, 0 },
        Command { CommandType::WaitForAgent, 2, 2, 0 },
        Command { CommandType::LoadIfmStripe, 3, 4, 0 },
        Command { CommandType::WaitForAgent, 6, 1, 0 },
        Command { CommandType::LoadIfmStripe, 3, 5, 0 }
        // clang-format on
    };

    const std::vector<Command> expectedDmaWrCommands{
        //clang-format off
        Command{ CommandType::WaitForAgent, 1, 0, 0 },
        Command{ CommandType::StoreOfmStripe, 2, 0, 0 },
        Command{ CommandType::WaitForAgent, 1, 1, 0 },
        Command{ CommandType::StoreOfmStripe, 2, 1, 0 },
        Command{ CommandType::WaitForAgent, 1, 2, 0 },
        Command{ CommandType::StoreOfmStripe, 2, 2, 0 },
        Command{ CommandType::WaitForAgent, 6, 2, 0 },
        Command{ CommandType::StoreOfmStripe, 7, 0, 0 }
        // clang-format on
    };

    const std::vector<Command> expectedMceCommands{
        // clang-format off
        Command { CommandType::ProgramMceStripe, 0, 0, 0 },
        Command { CommandType::StartMceStripe, 0, 0, 0 },
        Command { CommandType::ProgramMceStripe, 0, 1, 0 },
        Command { CommandType::StartMceStripe, 0, 1, 0 },
        Command { CommandType::ProgramMceStripe, 0, 2, 0 },
        Command { CommandType::StartMceStripe, 0, 2, 0 },
        Command { CommandType::ProgramMceStripe, 6, 0, 0 },
        Command { CommandType::WaitForAgent, 3, 2, 0 },
        Command { CommandType::WaitForAgent, 4, 0, 0 },
        Command { CommandType::StartMceStripe, 6, 0, 0 },
        Command { CommandType::ProgramMceStripe, 6, 1, 0 },
        Command { CommandType::WaitForAgent, 3, 4, 0 },
        Command { CommandType::StartMceStripe, 6, 1, 0 },
        Command { CommandType::ProgramMceStripe, 6, 2, 0 },
        Command { CommandType::WaitForAgent, 3, 5, 0 },
        Command { CommandType::StartMceStripe, 6, 2,  0}
        // clang-format on
    };

    const std::vector<Command> expectedPleCommands{
        // clang-format off
        Command { CommandType::StartPleStripe, 1, 0, 0 },
        Command { CommandType::StartPleStripe, 1, 1, 0 },
        Command { CommandType::StartPleStripe, 1, 2,  0}
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, debuggingContext);
    scheduler.Schedule();

    CHECK(scheduler.GetDmaRdCommands() == expectedDmaRdCommands);
    CHECK(scheduler.GetDmaWrCommands() == expectedDmaWrCommands);
    CHECK(scheduler.GetMceCommands() == expectedMceCommands);
    CHECK(scheduler.GetPleCommands() == expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/WgtStreamer/AllFitInOneTile/WithWriteDependency")
{
    IfmSDesc ifms{};
    ifms.fmData.tile.numSlots = 4;

    const uint32_t numStripesTotal = 3;
    // When there is a write dependency, the tileSize needs to be set with the right value. i.e. 3
    const uint16_t tileSize = 3;
    WgtSDesc wgtS{};
    wgtS.tile.numSlots = tileSize;

    std::vector<AgentDescAndDeps> cmdStream{
        AgentDescAndDeps{
            AgentDesc(18, ifms),

            {
                /* .readDependencies =*/{},
                /* .writeDependencies =*/{ { { 3, { 3, 6 }, { 1, 2 }, 1 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(numStripesTotal, wgtS),

            {
                /* .readDependencies =*/{},
                /* .writeDependencies =*/{ { { 2, { 9, 3 }, { 3, 1 }, 0 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, PleLDesc{}),

            {
                /* .readDependencies =*/{},
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(9, MceSDesc{}),

            {
                /* .readDependencies =*/
                { {
                    { 3, { 6, 3 }, { 2, 1 }, 1 },
                    { 2, { 3, 9 }, { 1, 3 }, 0 },
                } },
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, OfmSDesc{}),
            {
                /* .readDependencies =*/{ { { 1, { 9, 1 }, { 9, 1 }, 0 } } },
                /* .writeDependencies =*/{},
            },
        },

    };

    const std::vector<Command> expectedDmaWrCommands{
        //clang-format off
        Command{ CommandType::WaitForAgent, 3, 8, 0 }, Command{ CommandType::StoreOfmStripe, 4, 0, 0 }
        //clang-format on
    };

    const std::vector<Command> expectedPleCommands{
        // clang-format off
        // clang-format on
    };
    const std::vector<Command> expectedDmaRdCommands{
        // clang-format off
        Command { CommandType::LoadIfmStripe, 0, 0, 0 },
        Command { CommandType::LoadIfmStripe, 0, 1, 0 },
        Command { CommandType::LoadIfmStripe, 0, 2, 0 },
        Command { CommandType::LoadWgtStripe, 1, 0, 0 },
        Command { CommandType::LoadIfmStripe, 0, 3, 0 },
        Command { CommandType::WaitForAgent, 3, 0, 0 },
        Command { CommandType::LoadIfmStripe, 0, 4, 0 },
        Command { CommandType::WaitForAgent, 3, 1, 0 },
        Command { CommandType::LoadIfmStripe, 0, 5, 0 },
        Command { CommandType::LoadIfmStripe, 0, 6, 0 },
        Command { CommandType::WaitForAgent, 3, 2, 0 },
        Command { CommandType::LoadIfmStripe, 0, 7, 0 },
        Command { CommandType::LoadIfmStripe, 0, 8, 0 },
        Command { CommandType::LoadWgtStripe, 1, 1, 0 },
        Command { CommandType::LoadIfmStripe, 0, 9, 0 },
        Command { CommandType::WaitForAgent, 3, 3, 0 },
        Command { CommandType::LoadIfmStripe, 0, 10, 0 },
        Command { CommandType::WaitForAgent, 3, 4, 0 },
        Command { CommandType::LoadIfmStripe, 0, 11, 0 },
        Command { CommandType::LoadIfmStripe, 0, 12, 0 },
        Command { CommandType::WaitForAgent, 3, 5, 0 },
        Command { CommandType::LoadIfmStripe, 0, 13, 0 },
        Command { CommandType::LoadIfmStripe, 0, 14, 0 },
        Command { CommandType::LoadWgtStripe, 1, 2, 0 },
        Command { CommandType::LoadIfmStripe, 0, 15, 0 },
        Command { CommandType::WaitForAgent, 3, 6, 0 },
        Command { CommandType::LoadIfmStripe, 0, 16, 0 },
        Command { CommandType::WaitForAgent, 3, 7, 0 },
        Command { CommandType::LoadIfmStripe, 0, 17, 0 }
        // clang-format on
    };

    const std::vector<Command> expectedMceCommands{
        // clang-format off
        Command { CommandType::ProgramMceStripe, 3, 0, 0 },
        Command { CommandType::WaitForAgent, 0, 2, 0 },
        Command { CommandType::WaitForAgent, 1, 0, 0 },
        Command { CommandType::StartMceStripe, 3, 0, 0 },
        Command { CommandType::ProgramMceStripe, 3, 1, 0 },
        Command { CommandType::WaitForAgent, 0, 4, 0 },
        Command { CommandType::StartMceStripe, 3, 1, 0 },
        Command { CommandType::ProgramMceStripe, 3, 2, 0 },
        Command { CommandType::WaitForAgent, 0, 5, 0 },
        Command { CommandType::StartMceStripe, 3, 2, 0 },
        Command { CommandType::ProgramMceStripe, 3, 3, 0 },
        Command { CommandType::WaitForAgent, 0, 8, 0 },
        Command { CommandType::WaitForAgent, 1, 1, 0 },
        Command { CommandType::StartMceStripe, 3, 3, 0 },
        Command { CommandType::ProgramMceStripe, 3, 4, 0 },
        Command { CommandType::WaitForAgent, 0, 10, 0 },
        Command { CommandType::StartMceStripe, 3, 4, 0 },
        Command { CommandType::ProgramMceStripe, 3, 5, 0 },
        Command { CommandType::WaitForAgent, 0, 11, 0 },
        Command { CommandType::StartMceStripe, 3, 5, 0 },
        Command { CommandType::ProgramMceStripe, 3, 6, 0 },
        Command { CommandType::WaitForAgent, 0, 14, 0 },
        Command { CommandType::WaitForAgent, 1, 2, 0 },
        Command { CommandType::StartMceStripe, 3, 6, 0 },
        Command { CommandType::ProgramMceStripe, 3, 7, 0 },
        Command { CommandType::WaitForAgent, 0, 16, 0 },
        Command { CommandType::StartMceStripe, 3, 7, 0 },
        Command { CommandType::ProgramMceStripe, 3, 8, 0 },
        Command { CommandType::WaitForAgent, 0, 17, 0 },
        Command { CommandType::StartMceStripe, 3, 8,  0}
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, debuggingContext);
    scheduler.Schedule();

    CHECK(scheduler.GetDmaRdCommands() == expectedDmaRdCommands);
    CHECK(scheduler.GetDmaWrCommands() == expectedDmaWrCommands);
    CHECK(scheduler.GetMceCommands() == expectedMceCommands);
    CHECK(scheduler.GetPleCommands() == expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/WgtStreamer/AllFitInOneTile/NoWriteDependency")
{
    IfmSDesc ifms{};
    ifms.fmData.tile.numSlots = 4;

    const uint32_t numStripesTotal = 3;

    std::vector<AgentDescAndDeps> cmdStream{
        AgentDescAndDeps{
            AgentDesc(18, ifms),

            {
                /* .readDependencies =*/{},
                /* .writeDependencies =*/{ { { 3, { 3, 6 }, { 1, 2 }, 1 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(numStripesTotal, WgtSDesc{}),

            {
                /* .readDependencies =*/{},
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, PleLDesc{}),

            {
                /* .readDependencies =*/{},
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(9, MceSDesc{}),

            {
                /* .readDependencies =*/
                { {
                    { 3, { 6, 3 }, { 2, 1 }, 1 },
                    { 2, { 3, 9 }, { 1, 3 }, 0 },
                } },
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, OfmSDesc{}),
            {
                /* .readDependencies =*/{ { { 1, { 9, 1 }, { 9, 1 }, 0 } } },
                /* .writeDependencies =*/{},
            },
        },

    };

    const std::vector<Command> expectedDmaWrCommands{
        //clang-format off
        Command{ CommandType::WaitForAgent, 3, 8, 0 }, Command{ CommandType::StoreOfmStripe, 4, 0, 0 }
        //clang-format on
    };

    const std::vector<Command> expectedPleCommands{
        // clang-format off
        // clang-format on
    };

    const std::vector<Command> expectedDmaRdCommands{
        // clang-format off
        Command { CommandType::LoadIfmStripe, 0, 0, 0 },
        Command { CommandType::LoadIfmStripe, 0, 1, 0 },
        Command { CommandType::LoadIfmStripe, 0, 2, 0 },
        Command { CommandType::LoadWgtStripe, 1, 0, 0 },
        Command { CommandType::LoadIfmStripe, 0, 3, 0 },
        Command { CommandType::WaitForAgent, 3, 0, 0 },
        Command { CommandType::LoadIfmStripe, 0, 4, 0 },
        Command { CommandType::WaitForAgent, 3, 1, 0 },
        Command { CommandType::LoadIfmStripe, 0, 5, 0 },
        Command { CommandType::LoadIfmStripe, 0, 6, 0 },
        Command { CommandType::WaitForAgent, 3, 2, 0 },
        Command { CommandType::LoadIfmStripe, 0, 7, 0 },
        Command { CommandType::LoadIfmStripe, 0, 8, 0 },
        Command { CommandType::LoadWgtStripe, 1, 1, 0 },
        Command { CommandType::LoadIfmStripe, 0, 9, 0 },
        Command { CommandType::WaitForAgent, 3, 3, 0 },
        Command { CommandType::LoadIfmStripe, 0, 10, 0 },
        Command { CommandType::WaitForAgent, 3, 4, 0 },
        Command { CommandType::LoadIfmStripe, 0, 11, 0 },
        Command { CommandType::LoadIfmStripe, 0, 12, 0 },
        Command { CommandType::WaitForAgent, 3, 5, 0 },
        Command { CommandType::LoadIfmStripe, 0, 13, 0 },
        Command { CommandType::LoadIfmStripe, 0, 14, 0 },
        Command { CommandType::LoadWgtStripe, 1, 2, 0 },
        Command { CommandType::LoadIfmStripe, 0, 15, 0 },
        Command { CommandType::WaitForAgent, 3, 6, 0 },
        Command { CommandType::LoadIfmStripe, 0, 16, 0 },
        Command { CommandType::WaitForAgent, 3, 7, 0 },
        Command { CommandType::LoadIfmStripe, 0, 17, 0 }
        // clang-format on
    };

    const std::vector<Command> expectedMceCommands{
        // clang-format off
        Command { CommandType::ProgramMceStripe, 3, 0, 0 },
        Command { CommandType::WaitForAgent, 0, 2, 0 },
        Command { CommandType::WaitForAgent, 1, 0, 0 },
        Command { CommandType::StartMceStripe, 3, 0, 0 },
        Command { CommandType::ProgramMceStripe, 3, 1, 0 },
        Command { CommandType::WaitForAgent, 0, 4, 0 },
        Command { CommandType::StartMceStripe, 3, 1, 0 },
        Command { CommandType::ProgramMceStripe, 3, 2, 0 },
        Command { CommandType::WaitForAgent, 0, 5, 0 },
        Command { CommandType::StartMceStripe, 3, 2, 0 },
        Command { CommandType::ProgramMceStripe, 3, 3, 0 },
        Command { CommandType::WaitForAgent, 0, 8, 0 },
        Command { CommandType::WaitForAgent, 1, 1, 0 },
        Command { CommandType::StartMceStripe, 3, 3, 0 },
        Command { CommandType::ProgramMceStripe, 3, 4, 0 },
        Command { CommandType::WaitForAgent, 0, 10, 0 },
        Command { CommandType::StartMceStripe, 3, 4, 0 },
        Command { CommandType::ProgramMceStripe, 3, 5, 0 },
        Command { CommandType::WaitForAgent, 0, 11, 0 },
        Command { CommandType::StartMceStripe, 3, 5, 0 },
        Command { CommandType::ProgramMceStripe, 3, 6, 0 },
        Command { CommandType::WaitForAgent, 0, 14, 0 },
        Command { CommandType::WaitForAgent, 1, 2, 0 },
        Command { CommandType::StartMceStripe, 3, 6, 0 },
        Command { CommandType::ProgramMceStripe, 3, 7, 0 },
        Command { CommandType::WaitForAgent, 0, 16, 0 },
        Command { CommandType::StartMceStripe, 3, 7, 0 },
        Command { CommandType::ProgramMceStripe, 3, 8, 0 },
        Command { CommandType::WaitForAgent, 0, 17, 0 },
        Command { CommandType::StartMceStripe, 3, 8,  0}
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, debuggingContext);
    scheduler.Schedule();

    CHECK(scheduler.GetDmaRdCommands() == expectedDmaRdCommands);
    CHECK(scheduler.GetDmaWrCommands() == expectedDmaWrCommands);
    CHECK(scheduler.GetMceCommands() == expectedMceCommands);
    CHECK(scheduler.GetPleCommands() == expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/WgtStreamer/WithWriteDependency/TileSize=1")
{
    IfmSDesc ifms{};
    ifms.fmData.tile.numSlots = 4;

    const uint32_t numStripesTotal = 3;
    const uint32_t tileSize        = 1;
    WgtSDesc wgtS{};
    wgtS.tile.numSlots = tileSize;
    std::vector<AgentDescAndDeps> cmdStream{
        AgentDescAndDeps{
            AgentDesc(18, ifms),

            {
                /* .readDependencies =*/{},
                /* .writeDependencies =*/{ { { 3, { 3, 6 }, { 1, 2 }, 1 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(numStripesTotal, wgtS),

            {
                /* .readDependencies =*/{},
                /* .writeDependencies =*/{ { { 2, { 9, 3 }, { 3, 1 }, 0 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, PleLDesc{}),

            {
                /* .readDependencies =*/{},
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(9, MceSDesc{}),

            {
                /* .readDependencies =*/
                { {
                    { 3, { 6, 3 }, { 2, 1 }, 1 },
                    { 2, { 3, 9 }, { 1, 3 }, 0 },
                } },
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, OfmSDesc{}),
            {
                /* .readDependencies =*/{ { { 1, { 9, 1 }, { 9, 1 }, 0 } } },
                /* .writeDependencies =*/{},
            },
        },

    };

    const std::vector<Command> expectedDmaWrCommands{
        //clang-format off
        Command{ CommandType::WaitForAgent, 3, 8, 0 }, Command{ CommandType::StoreOfmStripe, 4, 0, 0 }
        //clang-format on
    };

    const std::vector<Command> expectedPleCommands{
        // clang-format off
        // clang-format on
    };

    const std::vector<Command> expectedDmaRdCommands{
        // clang-format off
        Command { CommandType::LoadIfmStripe, 0, 0, 0 },
        Command { CommandType::LoadIfmStripe, 0, 1, 0 },
        Command { CommandType::LoadIfmStripe, 0, 2, 0 },
        Command { CommandType::LoadWgtStripe, 1, 0, 0 },
        Command { CommandType::LoadIfmStripe, 0, 3, 0 },
        Command { CommandType::WaitForAgent, 3, 0, 0 },
        Command { CommandType::LoadIfmStripe, 0, 4, 0 },
        Command { CommandType::WaitForAgent, 3, 1, 0 },
        Command { CommandType::LoadIfmStripe, 0, 5, 0 },
        Command { CommandType::LoadIfmStripe, 0, 6, 0 },
        Command { CommandType::WaitForAgent, 3, 2, 0 },
        Command { CommandType::LoadIfmStripe, 0, 7, 0 },
        Command { CommandType::LoadIfmStripe, 0, 8, 0 },
        Command { CommandType::WaitForAgent, 3, 2, 0 },
        Command { CommandType::LoadWgtStripe, 1, 1, 0 },
        Command { CommandType::LoadIfmStripe, 0, 9, 0 },
        Command { CommandType::WaitForAgent, 3, 3, 0 },
        Command { CommandType::LoadIfmStripe, 0, 10, 0 },
        Command { CommandType::WaitForAgent, 3, 4, 0 },
        Command { CommandType::LoadIfmStripe, 0, 11, 0 },
        Command { CommandType::LoadIfmStripe, 0, 12, 0 },
        Command { CommandType::WaitForAgent, 3, 5, 0 },
        Command { CommandType::LoadIfmStripe, 0, 13, 0 },
        Command { CommandType::LoadIfmStripe, 0, 14, 0 },
        Command { CommandType::WaitForAgent, 3, 5, 0 },
        Command { CommandType::LoadWgtStripe, 1, 2, 0 },
        Command { CommandType::LoadIfmStripe, 0, 15, 0 },
        Command { CommandType::WaitForAgent, 3, 6, 0 },
        Command { CommandType::LoadIfmStripe, 0, 16, 0 },
        Command { CommandType::WaitForAgent, 3, 7, 0 },
        Command { CommandType::LoadIfmStripe, 0, 17, 0 }
        // clang-format on
    };

    const std::vector<Command> expectedMceCommands{
        // clang-format off
        Command { CommandType::ProgramMceStripe, 3, 0, 0 },
        Command { CommandType::WaitForAgent, 0, 2, 0 },
        Command { CommandType::WaitForAgent, 1, 0, 0 },
        Command { CommandType::StartMceStripe, 3, 0, 0 },
        Command { CommandType::ProgramMceStripe, 3, 1, 0 },
        Command { CommandType::WaitForAgent, 0, 4, 0 },
        Command { CommandType::StartMceStripe, 3, 1, 0 },
        Command { CommandType::ProgramMceStripe, 3, 2, 0 },
        Command { CommandType::WaitForAgent, 0, 5, 0 },
        Command { CommandType::StartMceStripe, 3, 2, 0 },
        Command { CommandType::ProgramMceStripe, 3, 3, 0 },
        Command { CommandType::WaitForAgent, 0, 8, 0 },
        Command { CommandType::WaitForAgent, 1, 1, 0 },
        Command { CommandType::StartMceStripe, 3, 3, 0 },
        Command { CommandType::ProgramMceStripe, 3, 4, 0 },
        Command { CommandType::WaitForAgent, 0, 10, 0 },
        Command { CommandType::StartMceStripe, 3, 4, 0 },
        Command { CommandType::ProgramMceStripe, 3, 5, 0 },
        Command { CommandType::WaitForAgent, 0, 11, 0 },
        Command { CommandType::StartMceStripe, 3, 5, 0 },
        Command { CommandType::ProgramMceStripe, 3, 6, 0 },
        Command { CommandType::WaitForAgent, 0, 14, 0 },
        Command { CommandType::WaitForAgent, 1, 2, 0 },
        Command { CommandType::StartMceStripe, 3, 6, 0 },
        Command { CommandType::ProgramMceStripe, 3, 7, 0 },
        Command { CommandType::WaitForAgent, 0, 16, 0 },
        Command { CommandType::StartMceStripe, 3, 7, 0 },
        Command { CommandType::ProgramMceStripe, 3, 8, 0 },
        Command { CommandType::WaitForAgent, 0, 17, 0 },
        Command { CommandType::StartMceStripe, 3, 8,  0}
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, debuggingContext);
    scheduler.Schedule();

    CHECK(scheduler.GetDmaRdCommands() == expectedDmaRdCommands);
    CHECK(scheduler.GetDmaWrCommands() == expectedDmaWrCommands);
    CHECK(scheduler.GetMceCommands() == expectedMceCommands);
    CHECK(scheduler.GetPleCommands() == expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/WgtStreamer/WithReadAndWriteDependencies/TileSize=2")
{
    const uint32_t numStripesTotal = 3;
    const uint32_t tileSize        = 2;
    WgtSDesc wgtS{};
    wgtS.tile.numSlots = tileSize;

    PleSDesc ples{};
    ples.ofmTile.numSlots = 2;

    std::vector<AgentDescAndDeps> cmdStream{
        AgentDescAndDeps{
            AgentDesc(9, MceSDesc{}),

            {
                /* .readDependencies =*/
                {},
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, ples),

            {
                /* .readDependencies =*/
                { {
                    { 1, { 9, 1 }, { 9, 1 }, 0 },
                } },
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(numStripesTotal, wgtS),

            {
                /* .readDependencies =*/{ { { 2, { 9, 3 }, { 9, 3 }, 0 } } },
                /* .writeDependencies =*/{ { { 2, { 12, 3 }, { 4, 1 }, 0 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, PleLDesc{}),

            {
                /* .readDependencies =*/{},
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(12, MceSDesc{}),

            {
                /* .readDependencies =*/
                { {
                    { 3, { 1, 12 }, { 1, 12 }, 0 },
                    { 2, { 3, 12 }, { 1, 4 }, 0 },
                } },
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, OfmSDesc{}),
            {
                /* .readDependencies =*/{ { { 1, { 12, 1 }, { 12, 1 }, 0 } } },
                /* .writeDependencies =*/{},
            },
        },

    };

    const std::vector<Command> expectedDmaWrCommands{
        //clang-format off
        Command{ CommandType::WaitForAgent, 4, 11, 0 }, Command{ CommandType::StoreOfmStripe, 5, 0, 0 }
        //clang-format on
    };

    const std::vector<Command> expectedDmaRdCommands{
        // clang-format off
        Command { CommandType::WaitForAgent, 0, 8, 0 },
        Command { CommandType::LoadWgtStripe, 2, 0, 0 },
        Command { CommandType::LoadWgtStripe, 2, 1, 0 },
        Command { CommandType::WaitForAgent, 4, 3, 0 },
        Command { CommandType::LoadWgtStripe, 2, 2, 0 }
        // clang-format on
    };

    const std::vector<Command> expectedMceCommands{
        // clang-format off
        Command { CommandType::ProgramMceStripe, 0, 0, 0 },
        Command { CommandType::StartMceStripe, 0, 0, 0 },
        Command { CommandType::ProgramMceStripe, 0, 1, 0 },
        Command { CommandType::StartMceStripe, 0, 1, 0 },
        Command { CommandType::ProgramMceStripe, 0, 2, 0 },
        Command { CommandType::StartMceStripe, 0, 2, 0 },
        Command { CommandType::ProgramMceStripe, 0, 3, 0 },
        Command { CommandType::StartMceStripe, 0, 3, 0 },
        Command { CommandType::ProgramMceStripe, 0, 4, 0 },
        Command { CommandType::StartMceStripe, 0, 4, 0 },
        Command { CommandType::ProgramMceStripe, 0, 5, 0 },
        Command { CommandType::StartMceStripe, 0, 5, 0 },
        Command { CommandType::ProgramMceStripe, 0, 6, 0 },
        Command { CommandType::StartMceStripe, 0, 6, 0 },
        Command { CommandType::ProgramMceStripe, 0, 7, 0 },
        Command { CommandType::StartMceStripe, 0, 7, 0 },
        Command { CommandType::ProgramMceStripe, 0, 8, 0 },
        Command { CommandType::StartMceStripe, 0, 8, 0 },
        Command { CommandType::ProgramMceStripe, 4, 0, 0 },
        Command { CommandType::WaitForAgent, 1, 0, 0 },
        Command { CommandType::WaitForAgent, 2, 0, 0 },
        Command { CommandType::StartMceStripe, 4, 0, 0 },
        Command { CommandType::ProgramMceStripe, 4, 1, 0 },
        Command { CommandType::StartMceStripe, 4, 1, 0 },
        Command { CommandType::ProgramMceStripe, 4, 2, 0 },
        Command { CommandType::StartMceStripe, 4, 2, 0 },
        Command { CommandType::ProgramMceStripe, 4, 3, 0 },
        Command { CommandType::StartMceStripe, 4, 3, 0 },
        Command { CommandType::ProgramMceStripe, 4, 4, 0 },
        Command { CommandType::WaitForAgent, 2, 1, 0 },
        Command { CommandType::StartMceStripe, 4, 4, 0 },
        Command { CommandType::ProgramMceStripe, 4, 5, 0 },
        Command { CommandType::StartMceStripe, 4, 5, 0 },
        Command { CommandType::ProgramMceStripe, 4, 6, 0 },
        Command { CommandType::StartMceStripe, 4, 6, 0 },
        Command { CommandType::ProgramMceStripe, 4, 7, 0 },
        Command { CommandType::StartMceStripe, 4, 7, 0 },
        Command { CommandType::ProgramMceStripe, 4, 8, 0 },
        Command { CommandType::WaitForAgent, 2, 2, 0 },
        Command { CommandType::StartMceStripe, 4, 8, 0 },
        Command { CommandType::ProgramMceStripe, 4, 9, 0 },
        Command { CommandType::StartMceStripe, 4, 9, 0 },
        Command { CommandType::ProgramMceStripe, 4, 10, 0 },
        Command { CommandType::StartMceStripe, 4, 10, 0 },
        Command { CommandType::ProgramMceStripe, 4, 11, 0 },
        Command { CommandType::StartMceStripe, 4, 11,  0}
        // clang-format on
    };

    const std::vector<Command> expectedPleCommands{
        // clang-format off
        Command { CommandType::StartPleStripe, 1, 0,  0}
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, debuggingContext);
    scheduler.Schedule();

    CHECK(scheduler.GetDmaRdCommands() == expectedDmaRdCommands);
    CHECK(scheduler.GetDmaWrCommands() == expectedDmaWrCommands);
    CHECK(scheduler.GetMceCommands() == expectedMceCommands);
    CHECK(scheduler.GetPleCommands() == expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/MceSchedulerStripe")
{
    IfmSDesc ifms{};
    ifms.fmData.tile.numSlots = 4;

    WgtSDesc wgts{};
    wgts.tile.numSlots = 2;

    const uint32_t numStripesTotal = 9;
    std::vector<AgentDescAndDeps> cmdStream{
        AgentDescAndDeps{
            AgentDesc(18, ifms),

            {
                /* .readDependencies =*/{},
                /* .writeDependencies =*/{ { { 3, { 3, 6 }, { 1, 2 }, 1 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(3, wgts),

            {
                /* .readDependencies =*/{},
                /* .writeDependencies =*/{ { { 2, { 9, 3 }, { 3, 1 }, 0 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, PleLDesc{}),

            {
                /* .readDependencies =*/{},
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(numStripesTotal, MceSDesc{}),

            {
                /* .readDependencies =*/
                { {
                    { 3, { 6, 3 }, { 2, 1 }, 1 },
                    { 2, { 3, 9 }, { 1, 3 }, 0 },
                } },
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, OfmSDesc{}),
            {
                /* .readDependencies =*/{ { { 1, { numStripesTotal, 1 }, { numStripesTotal, 1 }, 0 } } },
                /* .writeDependencies =*/{},
            },
        },

    };

    const std::vector<Command> expectedDmaWrCommands{
        //clang-format off
        Command{ CommandType::WaitForAgent, 3, 8, 0 }, Command{ CommandType::StoreOfmStripe, 4, 0, 0 }
        //clang-format on
    };

    const std::vector<Command> expectedPleCommands{
        // clang-format off
        // clang-format on
    };
    const std::vector<Command> expectedDmaRdCommands{
        // clang-format off
        Command { CommandType::LoadIfmStripe, 0, 0, 0 },
        Command { CommandType::LoadIfmStripe, 0, 1, 0 },
        Command { CommandType::LoadIfmStripe, 0, 2, 0 },
        Command { CommandType::LoadWgtStripe, 1, 0, 0 },
        Command { CommandType::LoadIfmStripe, 0, 3, 0 },
        Command { CommandType::WaitForAgent, 3, 0, 0 },
        Command { CommandType::LoadIfmStripe, 0, 4, 0 },
        Command { CommandType::WaitForAgent, 3, 1, 0 },
        Command { CommandType::LoadIfmStripe, 0, 5, 0 },
        Command { CommandType::LoadIfmStripe, 0, 6, 0 },
        Command { CommandType::WaitForAgent, 3, 2, 0 },
        Command { CommandType::LoadIfmStripe, 0, 7, 0 },
        Command { CommandType::LoadIfmStripe, 0, 8, 0 },
        Command { CommandType::LoadWgtStripe, 1, 1, 0 },
        Command { CommandType::LoadIfmStripe, 0, 9, 0 },
        Command { CommandType::WaitForAgent, 3, 3, 0 },
        Command { CommandType::LoadIfmStripe, 0, 10, 0 },
        Command { CommandType::WaitForAgent, 3, 4, 0 },
        Command { CommandType::LoadIfmStripe, 0, 11, 0 },
        Command { CommandType::LoadIfmStripe, 0, 12, 0 },
        Command { CommandType::WaitForAgent, 3, 5, 0 },
        Command { CommandType::LoadIfmStripe, 0, 13, 0 },
        Command { CommandType::LoadIfmStripe, 0, 14, 0 },
        Command { CommandType::WaitForAgent, 3, 2, 0 },
        Command { CommandType::LoadWgtStripe, 1, 2, 0 },
        Command { CommandType::LoadIfmStripe, 0, 15, 0 },
        Command { CommandType::WaitForAgent, 3, 6, 0 },
        Command { CommandType::LoadIfmStripe, 0, 16, 0 },
        Command { CommandType::WaitForAgent, 3, 7, 0 },
        Command { CommandType::LoadIfmStripe, 0, 17, 0 }
        // clang-format on
    };

    const std::vector<Command> expectedMceCommands{
        // clang-format off
        Command { CommandType::ProgramMceStripe, 3, 0, 0 },
        Command { CommandType::WaitForAgent, 0, 2, 0 },
        Command { CommandType::WaitForAgent, 1, 0, 0 },
        Command { CommandType::StartMceStripe, 3, 0, 0 },
        Command { CommandType::ProgramMceStripe, 3, 1, 0 },
        Command { CommandType::WaitForAgent, 0, 4, 0 },
        Command { CommandType::StartMceStripe, 3, 1, 0 },
        Command { CommandType::ProgramMceStripe, 3, 2, 0 },
        Command { CommandType::WaitForAgent, 0, 5, 0 },
        Command { CommandType::StartMceStripe, 3, 2, 0 },
        Command { CommandType::ProgramMceStripe, 3, 3, 0 },
        Command { CommandType::WaitForAgent, 0, 8, 0 },
        Command { CommandType::WaitForAgent, 1, 1, 0 },
        Command { CommandType::StartMceStripe, 3, 3, 0 },
        Command { CommandType::ProgramMceStripe, 3, 4, 0 },
        Command { CommandType::WaitForAgent, 0, 10, 0 },
        Command { CommandType::StartMceStripe, 3, 4, 0 },
        Command { CommandType::ProgramMceStripe, 3, 5, 0 },
        Command { CommandType::WaitForAgent, 0, 11, 0 },
        Command { CommandType::StartMceStripe, 3, 5, 0 },
        Command { CommandType::ProgramMceStripe, 3, 6, 0 },
        Command { CommandType::WaitForAgent, 0, 14, 0 },
        Command { CommandType::WaitForAgent, 1, 2, 0 },
        Command { CommandType::StartMceStripe, 3, 6, 0 },
        Command { CommandType::ProgramMceStripe, 3, 7, 0 },
        Command { CommandType::WaitForAgent, 0, 16, 0 },
        Command { CommandType::StartMceStripe, 3, 7, 0 },
        Command { CommandType::ProgramMceStripe, 3, 8, 0 },
        Command { CommandType::WaitForAgent, 0, 17, 0 },
        Command { CommandType::StartMceStripe, 3, 8,  0}
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, debuggingContext);
    scheduler.Schedule();

    CHECK(scheduler.GetDmaRdCommands() == expectedDmaRdCommands);
    CHECK(scheduler.GetDmaWrCommands() == expectedDmaWrCommands);
    CHECK(scheduler.GetMceCommands() == expectedMceCommands);
    CHECK(scheduler.GetPleCommands() == expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/PleLoaderStripe/NoDependencies")
{
    IfmSDesc ifms{};
    ifms.fmData.tile.numSlots = 4;

    WgtSDesc wgts{};
    wgts.tile.numSlots = 2;

    std::vector<AgentDescAndDeps> cmdStream{ AgentDescAndDeps{
                                                 AgentDesc(18, ifms),

                                                 {
                                                     /* .readDependencies =*/{},
                                                     /* .writeDependencies =*/{ { { 3, { 3, 6 }, { 1, 2 }, 1 } } },
                                                 },
                                             },
                                             AgentDescAndDeps{
                                                 AgentDesc(3, wgts),

                                                 {
                                                     /* .readDependencies =*/{},
                                                     /* .writeDependencies =*/{ { { 2, { 9, 3 }, { 3, 1 }, 0 } } },
                                                 },
                                             },
                                             AgentDescAndDeps{
                                                 AgentDesc(1, PleLDesc{}),

                                                 {
                                                     /* .readDependencies =*/{},
                                                     /* .writeDependencies =*/{},
                                                 },
                                             },
                                             AgentDescAndDeps{
                                                 AgentDesc(9, MceSDesc{}),

                                                 {
                                                     /* .readDependencies =*/
                                                     { {
                                                         { 3, { 6, 3 }, { 2, 1 }, 1 },
                                                         { 2, { 3, 9 }, { 1, 3 }, 0 },
                                                     } },
                                                     /* .writeDependencies =*/{},
                                                 },
                                             },
                                             AgentDescAndDeps{
                                                 AgentDesc(1, OfmSDesc{}),
                                                 {
                                                     /* .readDependencies =*/{ { { 1, { 9, 1 }, { 9, 1 }, 0 } } },
                                                     /* .writeDependencies =*/{},
                                                 },
                                             } };

    const std::vector<Command> expectedDmaWrCommands{
        //clang-format off
        Command{ CommandType::WaitForAgent, 3, 8, 0 }, Command{ CommandType::StoreOfmStripe, 4, 0, 0 }
        //clang-format on
    };

    const std::vector<Command> expectedPleCommands{
        // clang-format off
        // clang-format on
    };

    const std::vector<Command> expectedDmaRdCommands{
        // clang-format off
        Command { CommandType::LoadIfmStripe, 0, 0, 0 },
        Command { CommandType::LoadIfmStripe, 0, 1, 0 },
        Command { CommandType::LoadIfmStripe, 0, 2, 0 },
        Command { CommandType::LoadWgtStripe, 1, 0, 0 },
        Command { CommandType::LoadIfmStripe, 0, 3, 0 },
        Command { CommandType::WaitForAgent, 3, 0, 0 },
        Command { CommandType::LoadIfmStripe, 0, 4, 0 },
        Command { CommandType::WaitForAgent, 3, 1, 0 },
        Command { CommandType::LoadIfmStripe, 0, 5, 0 },
        Command { CommandType::LoadIfmStripe, 0, 6, 0 },
        Command { CommandType::WaitForAgent, 3, 2, 0 },
        Command { CommandType::LoadIfmStripe, 0, 7, 0 },
        Command { CommandType::LoadIfmStripe, 0, 8, 0 },
        Command { CommandType::LoadWgtStripe, 1, 1, 0 },
        Command { CommandType::LoadIfmStripe, 0, 9, 0 },
        Command { CommandType::WaitForAgent, 3, 3, 0 },
        Command { CommandType::LoadIfmStripe, 0, 10, 0 },
        Command { CommandType::WaitForAgent, 3, 4, 0 },
        Command { CommandType::LoadIfmStripe, 0, 11, 0 },
        Command { CommandType::LoadIfmStripe, 0, 12, 0 },
        Command { CommandType::WaitForAgent, 3, 5, 0 },
        Command { CommandType::LoadIfmStripe, 0, 13, 0 },
        Command { CommandType::LoadIfmStripe, 0, 14, 0 },
        Command { CommandType::WaitForAgent, 3, 2, 0 },
        Command { CommandType::LoadWgtStripe, 1, 2, 0 },
        Command { CommandType::LoadIfmStripe, 0, 15, 0 },
        Command { CommandType::WaitForAgent, 3, 6, 0 },
        Command { CommandType::LoadIfmStripe, 0, 16, 0 },
        Command { CommandType::WaitForAgent, 3, 7, 0 },
        Command { CommandType::LoadIfmStripe, 0, 17, 0 }
        // clang-format on
    };

    const std::vector<Command> expectedMceCommands{
        // clang-format off
        Command { CommandType::ProgramMceStripe, 3, 0, 0 },
        Command { CommandType::WaitForAgent, 0, 2, 0 },
        Command { CommandType::WaitForAgent, 1, 0, 0 },
        Command { CommandType::StartMceStripe, 3, 0, 0 },
        Command { CommandType::ProgramMceStripe, 3, 1, 0 },
        Command { CommandType::WaitForAgent, 0, 4, 0 },
        Command { CommandType::StartMceStripe, 3, 1, 0 },
        Command { CommandType::ProgramMceStripe, 3, 2, 0 },
        Command { CommandType::WaitForAgent, 0, 5, 0 },
        Command { CommandType::StartMceStripe, 3, 2, 0 },
        Command { CommandType::ProgramMceStripe, 3, 3, 0 },
        Command { CommandType::WaitForAgent, 0, 8, 0 },
        Command { CommandType::WaitForAgent, 1, 1, 0 },
        Command { CommandType::StartMceStripe, 3, 3, 0 },
        Command { CommandType::ProgramMceStripe, 3, 4, 0 },
        Command { CommandType::WaitForAgent, 0, 10, 0 },
        Command { CommandType::StartMceStripe, 3, 4, 0 },
        Command { CommandType::ProgramMceStripe, 3, 5, 0 },
        Command { CommandType::WaitForAgent, 0, 11, 0 },
        Command { CommandType::StartMceStripe, 3, 5, 0 },
        Command { CommandType::ProgramMceStripe, 3, 6, 0 },
        Command { CommandType::WaitForAgent, 0, 14, 0 },
        Command { CommandType::WaitForAgent, 1, 2, 0 },
        Command { CommandType::StartMceStripe, 3, 6, 0 },
        Command { CommandType::ProgramMceStripe, 3, 7, 0 },
        Command { CommandType::WaitForAgent, 0, 16, 0 },
        Command { CommandType::StartMceStripe, 3, 7, 0 },
        Command { CommandType::ProgramMceStripe, 3, 8, 0 },
        Command { CommandType::WaitForAgent, 0, 17, 0 },
        Command { CommandType::StartMceStripe, 3, 8,  0}
        // clang-format on
    };
    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, debuggingContext);
    scheduler.Schedule();

    CHECK(scheduler.GetDmaRdCommands() == expectedDmaRdCommands);
    CHECK(scheduler.GetDmaWrCommands() == expectedDmaWrCommands);
    CHECK(scheduler.GetMceCommands() == expectedMceCommands);
    CHECK(scheduler.GetPleCommands() == expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/PleLoaderStripe/WithReadDependency")
{
    WgtSDesc wgts{};
    wgts.tile.numSlots = 2;

    PleSDesc ples{};
    ples.ofmTile.numSlots = 2;

    std::vector<AgentDescAndDeps> cmdStream{
        AgentDescAndDeps{
            AgentDesc(9, MceSDesc{}),

            {
                /* .readDependencies =*/
                {},
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, ples),

            {
                /* .readDependencies =*/
                { {
                    { 1, { 9, 1 }, { 9, 1 }, 0 },
                } },
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(3, wgts),

            {
                /* .readDependencies =*/{},
                /* .writeDependencies =*/{ { { 2, { 12, 3 }, { 4, 1 }, 0 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, PleLDesc{}),

            {
                /* .readDependencies =*/{ { { 3, { 9, 1 }, { 9, 1 }, 0 } } },
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(12, MceSDesc{}),

            {
                /* .readDependencies =*/
                { {
                    { 3, { 1, 12 }, { 1, 12 }, 0 },
                    { 2, { 3, 12 }, { 1, 4 }, 0 },
                } },
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{ AgentDesc(1, OfmSDesc{}),
                          {
                              /* .readDependencies =*/{ { { 1, { 12, 1 }, { 12, 1 }, 0 } } },
                              /* .writeDependencies =*/{},
                          } },

    };

    const std::vector<Command> expectedDmaWrCommands{
        //clang-format off
        Command{ CommandType::WaitForAgent, 4, 11, 0 }, Command{ CommandType::StoreOfmStripe, 5, 0, 0 }
        //clang-format on
    };

    const std::vector<Command> expectedDmaRdCommands{
        // clang-format off
        Command { CommandType::LoadWgtStripe, 2, 0, 0 },
        Command { CommandType::LoadWgtStripe, 2, 1, 0 },
        Command { CommandType::WaitForAgent, 4, 3, 0 },
        Command { CommandType::LoadWgtStripe, 2, 2, 0 }
        // clang-format on
    };

    const std::vector<Command> expectedMceCommands{
        // clang-format off
        Command { CommandType::ProgramMceStripe, 0, 0, 0 },
        Command { CommandType::StartMceStripe, 0, 0, 0 },
        Command { CommandType::ProgramMceStripe, 0, 1, 0 },
        Command { CommandType::StartMceStripe, 0, 1, 0 },
        Command { CommandType::ProgramMceStripe, 0, 2, 0 },
        Command { CommandType::StartMceStripe, 0, 2, 0 },
        Command { CommandType::ProgramMceStripe, 0, 3, 0 },
        Command { CommandType::StartMceStripe, 0, 3, 0 },
        Command { CommandType::ProgramMceStripe, 0, 4, 0 },
        Command { CommandType::StartMceStripe, 0, 4, 0 },
        Command { CommandType::ProgramMceStripe, 0, 5, 0 },
        Command { CommandType::StartMceStripe, 0, 5, 0 },
        Command { CommandType::ProgramMceStripe, 0, 6, 0 },
        Command { CommandType::StartMceStripe, 0, 6, 0 },
        Command { CommandType::ProgramMceStripe, 0, 7, 0 },
        Command { CommandType::StartMceStripe, 0, 7, 0 },
        Command { CommandType::ProgramMceStripe, 0, 8, 0 },
        Command { CommandType::StartMceStripe, 0, 8, 0 },
        Command { CommandType::ProgramMceStripe, 4, 0, 0 },
        Command { CommandType::WaitForAgent, 1, 0, 0 },
        Command { CommandType::WaitForAgent, 2, 0, 0 },
        Command { CommandType::StartMceStripe, 4, 0, 0 },
        Command { CommandType::ProgramMceStripe, 4, 1, 0 },
        Command { CommandType::StartMceStripe, 4, 1, 0 },
        Command { CommandType::ProgramMceStripe, 4, 2, 0 },
        Command { CommandType::StartMceStripe, 4, 2, 0 },
        Command { CommandType::ProgramMceStripe, 4, 3, 0 },
        Command { CommandType::StartMceStripe, 4, 3, 0 },
        Command { CommandType::ProgramMceStripe, 4, 4, 0 },
        Command { CommandType::WaitForAgent, 2, 1, 0 },
        Command { CommandType::StartMceStripe, 4, 4, 0 },
        Command { CommandType::ProgramMceStripe, 4, 5, 0 },
        Command { CommandType::StartMceStripe, 4, 5, 0 },
        Command { CommandType::ProgramMceStripe, 4, 6, 0 },
        Command { CommandType::StartMceStripe, 4, 6, 0 },
        Command { CommandType::ProgramMceStripe, 4, 7, 0 },
        Command { CommandType::StartMceStripe, 4, 7, 0 },
        Command { CommandType::ProgramMceStripe, 4, 8, 0 },
        Command { CommandType::WaitForAgent, 2, 2, 0 },
        Command { CommandType::StartMceStripe, 4, 8, 0 },
        Command { CommandType::ProgramMceStripe, 4, 9, 0 },
        Command { CommandType::StartMceStripe, 4, 9, 0 },
        Command { CommandType::ProgramMceStripe, 4, 10, 0 },
        Command { CommandType::StartMceStripe, 4, 10, 0 },
        Command { CommandType::ProgramMceStripe, 4, 11, 0 },
        Command { CommandType::StartMceStripe, 4, 11,  0}
        // clang-format on
    };

    const std::vector<Command> expectedPleCommands{
        // clang-format off
        Command { CommandType::StartPleStripe, 1, 0,  0}
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, debuggingContext);
    scheduler.Schedule();

    CHECK(scheduler.GetDmaRdCommands() == expectedDmaRdCommands);
    CHECK(scheduler.GetDmaWrCommands() == expectedDmaWrCommands);
    CHECK(scheduler.GetMceCommands() == expectedMceCommands);
    CHECK(scheduler.GetPleCommands() == expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/PleLoaderStripe/WithWriteDependency")
{
    PleSDesc ples{};
    ples.ofmTile.numSlots = 2;

    // Create a small command stream that contains a PleL agent with a write dependency.
    // We will confirm that this dependency results in the expected wait command being
    // inserted in the command queue.
    std::vector<AgentDescAndDeps> cmdStream{
        AgentDescAndDeps{
            AgentDesc(2, PleLDesc{}),

            {
                /* .readDependencies =*/{},
                // This is the dependency we are testing
                /* .writeDependencies =*/{ { { 1, { 1, 1 }, { 1, 1 }, 0 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, ples),

            {
                /* .readDependencies =*/{ { 1, { 1, 1 }, { 1, 1 }, 0 } },
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{ AgentDesc(1, OfmSDesc{}),
                          {
                              /* .readDependencies =*/{ { { 2, { 2, 1 }, { 2, 1 }, 0 } } },
                              /* .writeDependencies =*/{},
                          } },

    };

    const std::vector<Command> expectedDmaWrCommands{
        //clang-format off
        Command{ CommandType::WaitForAgent, 0, 1, 0 }, Command{ CommandType::StoreOfmStripe, 2, 0, 0 }
        //clang-format on
    };

    const std::vector<Command> expectedMceCommands{
        // clang-format off
        // clang-format on
    };

    const std::vector<Command> expectedDmaRdCommands{
        // clang-format off
        Command { CommandType::LoadPleCode, 0, 0, 0 },
        Command { CommandType::WaitForAgent, 1, 0, 0 },
        Command { CommandType::LoadPleCode, 0, 1,  0}
        // clang-format on
    };

    const std::vector<Command> expectedPleCommands{
        // clang-format off
        Command { CommandType::WaitForAgent, 0, 0, 0 },
        Command { CommandType::StartPleStripe, 1, 0,  0}
        // clang-format on
    };
    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, debuggingContext);
    scheduler.Schedule();

    CHECK(scheduler.GetDmaRdCommands() == expectedDmaRdCommands);
    CHECK(scheduler.GetDmaWrCommands() == expectedDmaWrCommands);
    CHECK(scheduler.GetMceCommands() == expectedMceCommands);
    CHECK(scheduler.GetPleCommands() == expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/PleSchedulerStripe/NoWriteDependencies")
{
    PleSDesc ples{};
    ples.ofmTile.numSlots = 2;

    const uint32_t numStripesTotal = 3;
    std::vector<AgentDescAndDeps> cmdStream{
        AgentDescAndDeps{
            AgentDesc(1, PleLDesc{}),

            {
                /* .readDependencies =*/{},
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(3, MceSDesc{}),

            {
                /* .readDependencies =*/
                {},
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(numStripesTotal, ples),

            {
                /* .readDependencies =*/
                { {
                    { 2, { 1, 3 }, { 1, 3 }, 0 },
                    { 1, { 3, 3 }, { 1, 1 }, 0 },
                } },
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{ AgentDesc(1, OfmSDesc{}),
                          {
                              /* .readDependencies =*/{ { { 1, { numStripesTotal, 1 }, { numStripesTotal, 1 }, 0 } } },
                              /* .writeDependencies =*/{},
                          } },

    };

    const std::vector<Command> expectedDmaWrCommands{
        //clang-format off
        Command{ CommandType::WaitForAgent, 2, 2, 0 }, Command{ CommandType::StoreOfmStripe, 3, 0, 0 }
        //clang-format on
    };

    const std::vector<Command> expectedDmaRdCommands{
        // clang-format off
        Command { CommandType::LoadPleCode, 0, 0,  0}
        // clang-format on
    };

    const std::vector<Command> expectedMceCommands{
        // clang-format off
        Command { CommandType::ProgramMceStripe, 1, 0, 0 },
        Command { CommandType::StartMceStripe, 1, 0, 0 },
        Command { CommandType::ProgramMceStripe, 1, 1, 0 },
        Command { CommandType::StartMceStripe, 1, 1, 0 },
        Command { CommandType::ProgramMceStripe, 1, 2, 0 },
        Command { CommandType::StartMceStripe, 1, 2,  0}
        // clang-format on
    };

    const std::vector<Command> expectedPleCommands{
        // clang-format off
        Command { CommandType::WaitForAgent, 0, 0, 0 },
        Command { CommandType::StartPleStripe, 2, 0, 0 },
        Command { CommandType::StartPleStripe, 2, 1, 0 },
        Command { CommandType::StartPleStripe, 2, 2,  0}
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, debuggingContext);
    scheduler.Schedule();

    CHECK(scheduler.GetDmaRdCommands() == expectedDmaRdCommands);
    CHECK(scheduler.GetDmaWrCommands() == expectedDmaWrCommands);
    CHECK(scheduler.GetMceCommands() == expectedMceCommands);
    CHECK(scheduler.GetPleCommands() == expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/PleSchedulerStripe/WithWriteDependency/TileSize 1..4")
{
    const uint32_t numStripesTotal = 12;

    auto tileSize = GENERATE_COPY(Catch::Generators::range<uint16_t>(1, 4));

    OfmSDesc ofms{};
    ofms.fmData.tile.numSlots = 2;

    DYNAMIC_SECTION("For tileSize " << tileSize)
    {
        PleSDesc pleS{};
        pleS.ofmTile.numSlots = tileSize;
        std::vector<AgentDescAndDeps> cmdStream{
            AgentDescAndDeps{
                AgentDesc(1, PleLDesc{}),

                {
                    /* .readDependencies =*/{},
                    /* .writeDependencies =*/{},
                },
            },
            AgentDescAndDeps{
                AgentDesc(12, MceSDesc{}),

                {
                    /* .readDependencies =*/
                    {},
                    /* .writeDependencies =*/{},
                },
            },
            AgentDescAndDeps{
                AgentDesc(numStripesTotal, pleS),

                {
                    /* .readDependencies =*/
                    { {
                        { 2, { 1, 12 }, { 1, 12 }, 0 },
                        { 1, { 12, 12 }, { 1, 1 }, 0 },
                    } },
                    /* .writeDependencies =*/{ { { 1, { 12, 12 }, { 1, 1 }, 0 } } },
                },
            },
            AgentDescAndDeps{
                AgentDesc(12, ofms),

                {
                    /* .readDependencies =*/{ { { 1, { 12, 12 }, { 1, 1 }, 0 } } },
                    /* .writeDependencies =*/{},
                },
            },
        };

        std::vector<Command> expectedDmaRdCommands;
        std::vector<Command> expectedDmaWrCommands;
        std::vector<Command> expectedMceCommands;
        std::vector<Command> expectedPleCommands;

        switch (tileSize)
        {
            case 1:
                expectedDmaRdCommands = std::vector<Command>{
                    // clang-format off
            Command { CommandType::LoadPleCode, 0, 0,  0}
                    // clang-format on
                };

                expectedDmaWrCommands = std::vector<Command>{
                    //clang-format off
                    Command{ CommandType::WaitForAgent, 2, 0, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 0, 0 },
                    Command{ CommandType::WaitForAgent, 2, 1, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 1, 0 },
                    Command{ CommandType::WaitForAgent, 2, 2, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 2, 0 },
                    Command{ CommandType::WaitForAgent, 2, 3, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 3, 0 },
                    Command{ CommandType::WaitForAgent, 2, 4, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 4, 0 },
                    Command{ CommandType::WaitForAgent, 2, 5, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 5, 0 },
                    Command{ CommandType::WaitForAgent, 2, 6, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 6, 0 },
                    Command{ CommandType::WaitForAgent, 2, 7, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 7, 0 },
                    Command{ CommandType::WaitForAgent, 2, 8, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 8, 0 },
                    Command{ CommandType::WaitForAgent, 2, 9, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 9, 0 },
                    Command{ CommandType::WaitForAgent, 2, 10, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 10, 0 },
                    Command{ CommandType::WaitForAgent, 2, 11, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 11, 0 }
                    // clang-format on
                };

                expectedMceCommands = std::vector<Command>{
                    // clang-format off
            Command { CommandType::ProgramMceStripe, 1, 0, 0 },
            Command { CommandType::StartMceStripe, 1, 0, 0 },
            Command { CommandType::ProgramMceStripe, 1, 1, 0 },
            Command { CommandType::StartMceStripe, 1, 1, 0 },
            Command { CommandType::ProgramMceStripe, 1, 2, 0 },
            Command { CommandType::StartMceStripe, 1, 2, 0 },
            Command { CommandType::ProgramMceStripe, 1, 3, 0 },
            Command { CommandType::StartMceStripe, 1, 3, 0 },
            Command { CommandType::ProgramMceStripe, 1, 4, 0 },
            Command { CommandType::StartMceStripe, 1, 4, 0 },
            Command { CommandType::ProgramMceStripe, 1, 5, 0 },
            Command { CommandType::StartMceStripe, 1, 5, 0 },
            Command { CommandType::ProgramMceStripe, 1, 6, 0 },
            Command { CommandType::StartMceStripe, 1, 6, 0 },
            Command { CommandType::ProgramMceStripe, 1, 7, 0 },
            Command { CommandType::StartMceStripe, 1, 7, 0 },
            Command { CommandType::ProgramMceStripe, 1, 8, 0 },
            Command { CommandType::StartMceStripe, 1, 8, 0 },
            Command { CommandType::ProgramMceStripe, 1, 9, 0 },
            Command { CommandType::StartMceStripe, 1, 9, 0 },
            Command { CommandType::ProgramMceStripe, 1, 10, 0 },
            Command { CommandType::StartMceStripe, 1, 10, 0 },
            Command { CommandType::ProgramMceStripe, 1, 11, 0 },
            Command { CommandType::StartMceStripe, 1, 11,  0}
                    // clang-format on
                };

                expectedPleCommands = std::vector<Command>{
                    // clang-format off
            Command { CommandType::WaitForAgent, 0, 0, 0 },
            Command { CommandType::StartPleStripe, 2, 0, 0 },
            Command { CommandType::WaitForAgent, 3, 0, 0 },
            Command { CommandType::StartPleStripe, 2, 1, 0 },
            Command { CommandType::WaitForAgent, 3, 1, 0 },
            Command { CommandType::StartPleStripe, 2, 2, 0 },
            Command { CommandType::WaitForAgent, 3, 2, 0 },
            Command { CommandType::StartPleStripe, 2, 3, 0 },
            Command { CommandType::WaitForAgent, 3, 3, 0 },
            Command { CommandType::StartPleStripe, 2, 4, 0 },
            Command { CommandType::WaitForAgent, 3, 4, 0 },
            Command { CommandType::StartPleStripe, 2, 5, 0 },
            Command { CommandType::WaitForAgent, 3, 5, 0 },
            Command { CommandType::StartPleStripe, 2, 6, 0 },
            Command { CommandType::WaitForAgent, 3, 6, 0 },
            Command { CommandType::StartPleStripe, 2, 7, 0 },
            Command { CommandType::WaitForAgent, 3, 7, 0 },
            Command { CommandType::StartPleStripe, 2, 8, 0 },
            Command { CommandType::WaitForAgent, 3, 8, 0 },
            Command { CommandType::StartPleStripe, 2, 9, 0 },
            Command { CommandType::WaitForAgent, 3, 9, 0 },
            Command { CommandType::StartPleStripe, 2, 10, 0 },
            Command { CommandType::WaitForAgent, 3, 10, 0 },
            Command { CommandType::StartPleStripe, 2, 11,  0}
                    // clang-format on
                };
                break;
            case 2:

                expectedDmaRdCommands = std::vector<Command>{
                    // clang-format off
            Command { CommandType::LoadPleCode, 0, 0,  0}
                    // clang-format on
                };

                expectedDmaWrCommands = std::vector<Command>{
                    //clang-format off
                    Command{ CommandType::WaitForAgent, 2, 0, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 0, 0 },
                    Command{ CommandType::WaitForAgent, 2, 1, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 1, 0 },
                    Command{ CommandType::WaitForAgent, 2, 2, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 2, 0 },
                    Command{ CommandType::WaitForAgent, 2, 3, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 3, 0 },
                    Command{ CommandType::WaitForAgent, 2, 4, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 4, 0 },
                    Command{ CommandType::WaitForAgent, 2, 5, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 5, 0 },
                    Command{ CommandType::WaitForAgent, 2, 6, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 6, 0 },
                    Command{ CommandType::WaitForAgent, 2, 7, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 7, 0 },
                    Command{ CommandType::WaitForAgent, 2, 8, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 8, 0 },
                    Command{ CommandType::WaitForAgent, 2, 9, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 9, 0 },
                    Command{ CommandType::WaitForAgent, 2, 10, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 10, 0 },
                    Command{ CommandType::WaitForAgent, 2, 11, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 11, 0 }
                    // clang-format on
                };

                expectedMceCommands = std::vector<Command>{
                    // clang-format off
            Command { CommandType::ProgramMceStripe, 1, 0, 0 },
            Command { CommandType::StartMceStripe, 1, 0, 0 },
            Command { CommandType::ProgramMceStripe, 1, 1, 0 },
            Command { CommandType::StartMceStripe, 1, 1, 0 },
            Command { CommandType::ProgramMceStripe, 1, 2, 0 },
            Command { CommandType::StartMceStripe, 1, 2, 0 },
            Command { CommandType::ProgramMceStripe, 1, 3, 0 },
            Command { CommandType::StartMceStripe, 1, 3, 0 },
            Command { CommandType::ProgramMceStripe, 1, 4, 0 },
            Command { CommandType::StartMceStripe, 1, 4, 0 },
            Command { CommandType::ProgramMceStripe, 1, 5, 0 },
            Command { CommandType::StartMceStripe, 1, 5, 0 },
            Command { CommandType::ProgramMceStripe, 1, 6, 0 },
            Command { CommandType::StartMceStripe, 1, 6, 0 },
            Command { CommandType::ProgramMceStripe, 1, 7, 0 },
            Command { CommandType::StartMceStripe, 1, 7, 0 },
            Command { CommandType::ProgramMceStripe, 1, 8, 0 },
            Command { CommandType::StartMceStripe, 1, 8, 0 },
            Command { CommandType::ProgramMceStripe, 1, 9, 0 },
            Command { CommandType::StartMceStripe, 1, 9, 0 },
            Command { CommandType::ProgramMceStripe, 1, 10, 0 },
            Command { CommandType::StartMceStripe, 1, 10, 0 },
            Command { CommandType::ProgramMceStripe, 1, 11, 0 },
            Command { CommandType::StartMceStripe, 1, 11,  0}
                    // clang-format on
                };

                expectedPleCommands = std::vector<Command>{
                    // clang-format off
            Command { CommandType::WaitForAgent, 0, 0, 0 },
            Command { CommandType::StartPleStripe, 2, 0, 0 },
            Command { CommandType::StartPleStripe, 2, 1, 0 },
            Command { CommandType::WaitForAgent, 3, 0, 0 },
            Command { CommandType::StartPleStripe, 2, 2, 0 },
            Command { CommandType::WaitForAgent, 3, 1, 0 },
            Command { CommandType::StartPleStripe, 2, 3, 0 },
            Command { CommandType::WaitForAgent, 3, 2, 0 },
            Command { CommandType::StartPleStripe, 2, 4, 0 },
            Command { CommandType::WaitForAgent, 3, 3, 0 },
            Command { CommandType::StartPleStripe, 2, 5, 0 },
            Command { CommandType::WaitForAgent, 3, 4, 0 },
            Command { CommandType::StartPleStripe, 2, 6, 0 },
            Command { CommandType::WaitForAgent, 3, 5, 0 },
            Command { CommandType::StartPleStripe, 2, 7, 0 },
            Command { CommandType::WaitForAgent, 3, 6, 0 },
            Command { CommandType::StartPleStripe, 2, 8, 0 },
            Command { CommandType::WaitForAgent, 3, 7, 0 },
            Command { CommandType::StartPleStripe, 2, 9, 0 },
            Command { CommandType::WaitForAgent, 3, 8, 0 },
            Command { CommandType::StartPleStripe, 2, 10, 0 },
            Command { CommandType::WaitForAgent, 3, 9, 0 },
            Command { CommandType::StartPleStripe, 2, 11,  0}
                    // clang-format on
                };
                break;
            case 3:

                expectedDmaRdCommands = std::vector<Command>{
                    // clang-format off
            Command { CommandType::LoadPleCode, 0, 0,  0}
                    // clang-format on
                };

                expectedDmaWrCommands = std::vector<Command>{
                    //clang-format off
                    Command{ CommandType::WaitForAgent, 2, 0, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 0, 0 },
                    Command{ CommandType::WaitForAgent, 2, 1, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 1, 0 },
                    Command{ CommandType::WaitForAgent, 2, 2, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 2, 0 },
                    Command{ CommandType::WaitForAgent, 2, 3, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 3, 0 },
                    Command{ CommandType::WaitForAgent, 2, 4, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 4, 0 },
                    Command{ CommandType::WaitForAgent, 2, 5, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 5, 0 },
                    Command{ CommandType::WaitForAgent, 2, 6, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 6, 0 },
                    Command{ CommandType::WaitForAgent, 2, 7, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 7, 0 },
                    Command{ CommandType::WaitForAgent, 2, 8, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 8, 0 },
                    Command{ CommandType::WaitForAgent, 2, 9, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 9, 0 },
                    Command{ CommandType::WaitForAgent, 2, 10, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 10, 0 },
                    Command{ CommandType::WaitForAgent, 2, 11, 0 },
                    Command{ CommandType::StoreOfmStripe, 3, 11, 0 }
                    // clang-format on
                };

                expectedMceCommands = std::vector<Command>{
                    // clang-format off
            Command { CommandType::ProgramMceStripe, 1, 0, 0 },
            Command { CommandType::StartMceStripe, 1, 0, 0 },
            Command { CommandType::ProgramMceStripe, 1, 1, 0 },
            Command { CommandType::StartMceStripe, 1, 1, 0 },
            Command { CommandType::ProgramMceStripe, 1, 2, 0 },
            Command { CommandType::StartMceStripe, 1, 2, 0 },
            Command { CommandType::ProgramMceStripe, 1, 3, 0 },
            Command { CommandType::StartMceStripe, 1, 3, 0 },
            Command { CommandType::ProgramMceStripe, 1, 4, 0 },
            Command { CommandType::StartMceStripe, 1, 4, 0 },
            Command { CommandType::ProgramMceStripe, 1, 5, 0 },
            Command { CommandType::StartMceStripe, 1, 5, 0 },
            Command { CommandType::ProgramMceStripe, 1, 6, 0 },
            Command { CommandType::StartMceStripe, 1, 6, 0 },
            Command { CommandType::ProgramMceStripe, 1, 7, 0 },
            Command { CommandType::StartMceStripe, 1, 7, 0 },
            Command { CommandType::ProgramMceStripe, 1, 8, 0 },
            Command { CommandType::StartMceStripe, 1, 8, 0 },
            Command { CommandType::ProgramMceStripe, 1, 9, 0 },
            Command { CommandType::StartMceStripe, 1, 9, 0 },
            Command { CommandType::ProgramMceStripe, 1, 10, 0 },
            Command { CommandType::StartMceStripe, 1, 10, 0 },
            Command { CommandType::ProgramMceStripe, 1, 11, 0 },
            Command { CommandType::StartMceStripe, 1, 11,  0}
                    // clang-format on
                };

                expectedPleCommands = std::vector<Command>{
                    // clang-format off
            Command { CommandType::WaitForAgent, 0, 0, 0 },
            Command { CommandType::StartPleStripe, 2, 0, 0 },
            Command { CommandType::StartPleStripe, 2, 1, 0 },
            Command { CommandType::StartPleStripe, 2, 2, 0 },
            Command { CommandType::WaitForAgent, 3, 0, 0 },
            Command { CommandType::StartPleStripe, 2, 3, 0 },
            Command { CommandType::WaitForAgent, 3, 1, 0 },
            Command { CommandType::StartPleStripe, 2, 4, 0 },
            Command { CommandType::WaitForAgent, 3, 2, 0 },
            Command { CommandType::StartPleStripe, 2, 5, 0 },
            Command { CommandType::WaitForAgent, 3, 3, 0 },
            Command { CommandType::StartPleStripe, 2, 6, 0 },
            Command { CommandType::WaitForAgent, 3, 4, 0 },
            Command { CommandType::StartPleStripe, 2, 7, 0 },
            Command { CommandType::WaitForAgent, 3, 5, 0 },
            Command { CommandType::StartPleStripe, 2, 8, 0 },
            Command { CommandType::WaitForAgent, 3, 6, 0 },
            Command { CommandType::StartPleStripe, 2, 9, 0 },
            Command { CommandType::WaitForAgent, 3, 7, 0 },
            Command { CommandType::StartPleStripe, 2, 10, 0 },
            Command { CommandType::WaitForAgent, 3, 8, 0 },
            Command { CommandType::StartPleStripe, 2, 11,  0}
                    // clang-format on
                };

                break;
            default:
                FAIL("Invalid tile size");
        }

        ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

        Scheduler scheduler(cmdStream, debuggingContext);
        scheduler.Schedule();

        CHECK(scheduler.GetDmaRdCommands() == expectedDmaRdCommands);
        CHECK(scheduler.GetDmaWrCommands() == expectedDmaWrCommands);
        CHECK(scheduler.GetMceCommands() == expectedMceCommands);
        CHECK(scheduler.GetPleCommands() == expectedPleCommands);
    }
}

TEST_CASE("Cascading/StripeScheduler/PleSchedulerStripe/ReadDependencyToMceSIsFirst")
{
    PleSDesc ples{};
    ples.ofmTile.numSlots = 2;

    const uint32_t numStripesTotal = 3;
    std::vector<AgentDescAndDeps> cmdStream{
        AgentDescAndDeps{
            AgentDesc(1, PleLDesc{}),

            {
                /* .readDependencies =*/{},
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(3, MceSDesc{}),

            {
                /* .readDependencies =*/
                {},
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(numStripesTotal, ples),

            {
                /* .readDependencies =*/
                { {
                    // The order of those dependencies is different from the other test
                    { 1, { 3, 3 }, { 1, 1 }, 0 },
                    { 2, { 1, 3 }, { 1, 3 }, 0 },

                } },
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{ AgentDesc(1, OfmSDesc{}),
                          {
                              /* .readDependencies =*/{ { { 1, { numStripesTotal, 1 }, { numStripesTotal, 1 }, 0 } } },
                              /* .writeDependencies =*/{},
                          } },
    };
    const std::vector<Command> expectedDmaWrCommands{
        //clang-format off
        Command{ CommandType::WaitForAgent, 2, 2, 0 }, Command{ CommandType::StoreOfmStripe, 3, 0, 0 }
        //clang-format on
    };

    const std::vector<Command> expectedDmaRdCommands{
        // clang-format off
        Command { CommandType::LoadPleCode, 0, 0,  0}
        // clang-format on
    };

    const std::vector<Command> expectedMceCommands{
        // clang-format off
        Command { CommandType::ProgramMceStripe, 1, 0, 0 },
        Command { CommandType::StartMceStripe, 1, 0, 0 },
        Command { CommandType::ProgramMceStripe, 1, 1, 0 },
        Command { CommandType::StartMceStripe, 1, 1, 0 },
        Command { CommandType::ProgramMceStripe, 1, 2, 0 },
        Command { CommandType::StartMceStripe, 1, 2,  0}
        // clang-format on
    };

    const std::vector<Command> expectedPleCommands{
        // clang-format off
        Command { CommandType::WaitForAgent, 0, 0, 0 },
        Command { CommandType::StartPleStripe, 2, 0, 0 },
        Command { CommandType::StartPleStripe, 2, 1, 0 },
        Command { CommandType::StartPleStripe, 2, 2,  0}
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, debuggingContext);
    scheduler.Schedule();

    CHECK(scheduler.GetDmaRdCommands() == expectedDmaRdCommands);
    CHECK(scheduler.GetDmaWrCommands() == expectedDmaWrCommands);
    CHECK(scheduler.GetMceCommands() == expectedMceCommands);
    CHECK(scheduler.GetPleCommands() == expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/PleSchedulerStripe/ReadDependencyTowardsIfmS")
{
    PleSDesc ples{};
    ples.ofmTile.numSlots = 2;

    const uint32_t numStripesTotal = 3;
    std::vector<AgentDescAndDeps> cmdStream{
        AgentDescAndDeps{
            AgentDesc(1, PleLDesc{}),

            {
                /* .readDependencies =*/{},
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(3, IfmSDesc{}),

            {
                /* .readDependencies =*/
                {},
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(numStripesTotal, ples),

            {
                /* .readDependencies =*/
                { {
                    { 2, { 1, 3 }, { 1, 3 }, 0 },
                    { 1, { 3, 3 }, { 1, 1 }, 0 },
                } },
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{ AgentDesc(1, OfmSDesc{}),
                          {
                              /* .readDependencies =*/{ { { 1, { numStripesTotal, 1 }, { numStripesTotal, 1 }, 0 } } },
                              /* .writeDependencies =*/{},
                          } },
    };
    const std::vector<Command> expectedDmaWrCommands{
        //clang-format off
        Command{ CommandType::WaitForAgent, 2, 2, 0 }, Command{ CommandType::StoreOfmStripe, 3, 0, 0 }
        //clang-format on
    };

    const std::vector<Command> expectedMceCommands{
        // clang-format off
        // clang-format on
    };

    const std::vector<Command> expectedDmaRdCommands{
        // clang-format off
        Command { CommandType::LoadPleCode, 0, 0, 0 },
        Command { CommandType::LoadIfmStripe, 1, 0, 0 },
        Command { CommandType::LoadIfmStripe, 1, 1, 0 },
        Command { CommandType::LoadIfmStripe, 1, 2,  0}
        // clang-format on
    };

    const std::vector<Command> expectedPleCommands{
        // clang-format off
        Command { CommandType::WaitForAgent, 0, 0, 0 },
        Command { CommandType::WaitForAgent, 1, 0, 0 },
        Command { CommandType::StartPleStripe, 2, 0, 0 },
        Command { CommandType::WaitForAgent, 1, 1, 0 },
        Command { CommandType::StartPleStripe, 2, 1, 0 },
        Command { CommandType::WaitForAgent, 1, 2, 0 },
        Command { CommandType::StartPleStripe, 2, 2,  0}
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, debuggingContext);
    scheduler.Schedule();

    CHECK(scheduler.GetDmaRdCommands() == expectedDmaRdCommands);
    CHECK(scheduler.GetDmaWrCommands() == expectedDmaWrCommands);
    CHECK(scheduler.GetMceCommands() == expectedMceCommands);
    CHECK(scheduler.GetPleCommands() == expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/PleSchedulerStripe/Strategy0Cascading/FirstPle")
{
    PleSDesc ples{};
    ples.ofmTile.numSlots = 2;

    const uint32_t numStripesTotal = 5;
    std::vector<AgentDescAndDeps> cmdStream{
        AgentDescAndDeps{
            AgentDesc(4, PleLDesc{}),

            {
                /* .readDependencies =*/{},
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(5, MceSDesc{}),

            {
                /* .readDependencies =*/
                {},
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(numStripesTotal, ples),

            {
                /* .readDependencies =*/
                { {
                    { 2, { 4, 5 }, { 1, 1 }, -1 },
                    { 1, { 5, 5 }, { 1, 1 }, 0 },
                } },
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{ AgentDesc(1, OfmSDesc{}),
                          {
                              /* .readDependencies =*/{ { { 1, { numStripesTotal, 1 }, { numStripesTotal, 1 }, 0 } } },
                              /* .writeDependencies =*/{},
                          } },
    };

    const std::vector<Command> expectedDmaWrCommands{
        //clang-format off
        Command{ CommandType::WaitForAgent, 2, 4, 0 }, Command{ CommandType::StoreOfmStripe, 3, 0, 0 }
        //clang-format on
    };

    const std::vector<Command> expectedDmaRdCommands{
        // clang-format off
        Command { CommandType::LoadPleCode, 0, 0, 0 },
        Command { CommandType::LoadPleCode, 0, 1, 0 },
        Command { CommandType::LoadPleCode, 0, 2, 0 },
        Command { CommandType::LoadPleCode, 0, 3,  0}
        // clang-format on
    };

    const std::vector<Command> expectedMceCommands{
        // clang-format off
        Command { CommandType::ProgramMceStripe, 1, 0, 0 },
        Command { CommandType::StartMceStripe, 1, 0, 0 },
        Command { CommandType::ProgramMceStripe, 1, 1, 0 },
        Command { CommandType::StartMceStripe, 1, 1, 0 },
        Command { CommandType::ProgramMceStripe, 1, 2, 0 },
        Command { CommandType::StartMceStripe, 1, 2, 0 },
        Command { CommandType::ProgramMceStripe, 1, 3, 0 },
        Command { CommandType::StartMceStripe, 1, 3, 0 },
        Command { CommandType::ProgramMceStripe, 1, 4, 0 },
        Command { CommandType::StartMceStripe, 1, 4,  0}
        // clang-format on
    };

    const std::vector<Command> expectedPleCommands{
        // clang-format off
        Command { CommandType::WaitForAgent, 0, 0, 0 },
        Command { CommandType::StartPleStripe, 2, 0, 0 },
        Command { CommandType::StartPleStripe, 2, 1, 0 },
        Command { CommandType::WaitForAgent, 0, 1, 0 },
        Command { CommandType::StartPleStripe, 2, 2, 0 },
        Command { CommandType::WaitForAgent, 0, 2, 0 },
        Command { CommandType::StartPleStripe, 2, 3, 0 },
        Command { CommandType::WaitForAgent, 0, 3, 0 },
        Command { CommandType::StartPleStripe, 2, 4,  0}
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, debuggingContext);
    scheduler.Schedule();

    CHECK(scheduler.GetDmaRdCommands() == expectedDmaRdCommands);
    CHECK(scheduler.GetDmaWrCommands() == expectedDmaWrCommands);
    CHECK(scheduler.GetMceCommands() == expectedMceCommands);
    CHECK(scheduler.GetPleCommands() == expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/PleSchedulerStripe/Strategy0Cascading/SecondPle")
{
    const uint32_t numStripesTotal = 4;
    PleSDesc ples{};
    ples.ofmTile.numSlots = 4;

    OfmSDesc ofms{};
    ofms.fmData.tile.numSlots = 2;

    std::vector<AgentDescAndDeps> cmdStream{
        AgentDescAndDeps{
            AgentDesc(4, PleLDesc{}),

            {
                /* .readDependencies =*/{},
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(4, MceSDesc{}),

            {
                /* .readDependencies =*/
                {},
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(numStripesTotal, ples),

            {
                /* .readDependencies =*/
                { {
                    { 2, { 4, 4 }, { 1, 1 }, 0 },
                    { 1, { 4, 4 }, { 1, 1 }, 0 },
                } },
                /* .writeDependencies =*/{ { { 1, { 4, 4 }, { 1, 1 }, 0 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(4, ofms),

            {
                /* .readDependencies =*/{ { { 1, { 4, 4 }, { 1, 1 }, 0 } } },
                /* .writeDependencies =*/{},
            },
        },
    };

    const std::vector<Command> expectedDmaRdCommands{
        // clang-format off
        Command { CommandType::LoadPleCode, 0, 0, 0 },
        Command { CommandType::LoadPleCode, 0, 1, 0 },
        Command { CommandType::LoadPleCode, 0, 2, 0 },
        Command { CommandType::LoadPleCode, 0, 3,  0}
        // clang-format on
    };

    const std::vector<Command> expectedDmaWrCommands{
        //clang-format off
        Command{ CommandType::WaitForAgent, 2, 0, 0 },
        Command{ CommandType::StoreOfmStripe, 3, 0, 0 },
        Command{ CommandType::WaitForAgent, 2, 1, 0 },
        Command{ CommandType::StoreOfmStripe, 3, 1, 0 },
        Command{ CommandType::WaitForAgent, 2, 2, 0 },
        Command{ CommandType::StoreOfmStripe, 3, 2, 0 },
        Command{ CommandType::WaitForAgent, 2, 3, 0 },
        Command{ CommandType::StoreOfmStripe, 3, 3, 0 }
        // clang-format on
    };

    const std::vector<Command> expectedMceCommands{
        // clang-format off
        Command { CommandType::ProgramMceStripe, 1, 0, 0 },
        Command { CommandType::StartMceStripe, 1, 0, 0 },
        Command { CommandType::ProgramMceStripe, 1, 1, 0 },
        Command { CommandType::StartMceStripe, 1, 1, 0 },
        Command { CommandType::ProgramMceStripe, 1, 2, 0 },
        Command { CommandType::StartMceStripe, 1, 2, 0 },
        Command { CommandType::ProgramMceStripe, 1, 3, 0 },
        Command { CommandType::StartMceStripe, 1, 3,  0}
        // clang-format on
    };

    const std::vector<Command> expectedPleCommands{
        // clang-format off
        Command { CommandType::WaitForAgent, 0, 0, 0 },
        Command { CommandType::StartPleStripe, 2, 0, 0 },
        Command { CommandType::WaitForAgent, 0, 1, 0 },
        Command { CommandType::StartPleStripe, 2, 1, 0 },
        Command { CommandType::WaitForAgent, 0, 2, 0 },
        Command { CommandType::StartPleStripe, 2, 2, 0 },
        Command { CommandType::WaitForAgent, 0, 3, 0 },
        Command { CommandType::StartPleStripe, 2, 3,  0}
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, debuggingContext);
    scheduler.Schedule();

    CHECK(scheduler.GetDmaRdCommands() == expectedDmaRdCommands);
    CHECK(scheduler.GetDmaWrCommands() == expectedDmaWrCommands);
    CHECK(scheduler.GetMceCommands() == expectedMceCommands);
    CHECK(scheduler.GetPleCommands() == expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/OfmStreamerStripe")
{
    PleSDesc ples{};
    ples.ofmTile.numSlots = 2;

    OfmSDesc ofms{};
    ofms.fmData.tile.numSlots = 2;

    std::vector<AgentDescAndDeps> cmdStream{
        AgentDescAndDeps{
            AgentDesc(12, MceSDesc{}),

            {
                /* .readDependencies =*/
                {},
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{
            AgentDesc(12, ples),

            {
                /* .readDependencies =*/
                { { { 1, { 1, 1 }, { 1, 1 }, 0 } } },
                /* .writeDependencies =*/{ { { 1, { 12, 12 }, { 1, 1 }, 0 } } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(12, ofms),

            {
                /* .readDependencies =*/{ { { 1, { 12, 12 }, { 1, 1 }, 0 } } },
                /* .writeDependencies =*/{},
            },
        },
    };

    const std::vector<Command> expectedDmaRdCommands{
        // clang-format off
        // clang-format on
    };

    const std::vector<Command> expectedDmaWrCommands{
        //clang-format off
        Command{ CommandType::WaitForAgent, 1, 0, 0 },
        Command{ CommandType::StoreOfmStripe, 2, 0, 0 },
        Command{ CommandType::WaitForAgent, 1, 1, 0 },
        Command{ CommandType::StoreOfmStripe, 2, 1, 0 },
        Command{ CommandType::WaitForAgent, 1, 2, 0 },
        Command{ CommandType::StoreOfmStripe, 2, 2, 0 },
        Command{ CommandType::WaitForAgent, 1, 3, 0 },
        Command{ CommandType::StoreOfmStripe, 2, 3, 0 },
        Command{ CommandType::WaitForAgent, 1, 4, 0 },
        Command{ CommandType::StoreOfmStripe, 2, 4, 0 },
        Command{ CommandType::WaitForAgent, 1, 5, 0 },
        Command{ CommandType::StoreOfmStripe, 2, 5, 0 },
        Command{ CommandType::WaitForAgent, 1, 6, 0 },
        Command{ CommandType::StoreOfmStripe, 2, 6, 0 },
        Command{ CommandType::WaitForAgent, 1, 7, 0 },
        Command{ CommandType::StoreOfmStripe, 2, 7, 0 },
        Command{ CommandType::WaitForAgent, 1, 8, 0 },
        Command{ CommandType::StoreOfmStripe, 2, 8, 0 },
        Command{ CommandType::WaitForAgent, 1, 9, 0 },
        Command{ CommandType::StoreOfmStripe, 2, 9, 0 },
        Command{ CommandType::WaitForAgent, 1, 10, 0 },
        Command{ CommandType::StoreOfmStripe, 2, 10, 0 },
        Command{ CommandType::WaitForAgent, 1, 11, 0 },
        Command{ CommandType::StoreOfmStripe, 2, 11, 0 }
        // clang-format on
    };

    const std::vector<Command> expectedMceCommands{
        // clang-format off
        Command { CommandType::ProgramMceStripe, 0, 0, 0 },
        Command { CommandType::StartMceStripe, 0, 0, 0 },
        Command { CommandType::ProgramMceStripe, 0, 1, 0 },
        Command { CommandType::StartMceStripe, 0, 1, 0 },
        Command { CommandType::ProgramMceStripe, 0, 2, 0 },
        Command { CommandType::StartMceStripe, 0, 2, 0 },
        Command { CommandType::ProgramMceStripe, 0, 3, 0 },
        Command { CommandType::StartMceStripe, 0, 3, 0 },
        Command { CommandType::ProgramMceStripe, 0, 4, 0 },
        Command { CommandType::StartMceStripe, 0, 4, 0 },
        Command { CommandType::ProgramMceStripe, 0, 5, 0 },
        Command { CommandType::StartMceStripe, 0, 5, 0 },
        Command { CommandType::ProgramMceStripe, 0, 6, 0 },
        Command { CommandType::StartMceStripe, 0, 6, 0 },
        Command { CommandType::ProgramMceStripe, 0, 7, 0 },
        Command { CommandType::StartMceStripe, 0, 7, 0 },
        Command { CommandType::ProgramMceStripe, 0, 8, 0 },
        Command { CommandType::StartMceStripe, 0, 8, 0 },
        Command { CommandType::ProgramMceStripe, 0, 9, 0 },
        Command { CommandType::StartMceStripe, 0, 9, 0 },
        Command { CommandType::ProgramMceStripe, 0, 10, 0 },
        Command { CommandType::StartMceStripe, 0, 10, 0 },
        Command { CommandType::ProgramMceStripe, 0, 11, 0 },
        Command { CommandType::StartMceStripe, 0, 11,  0}
        // clang-format on
    };

    const std::vector<Command> expectedPleCommands{
        // clang-format off
        Command { CommandType::StartPleStripe, 1, 0, 0 },
        Command { CommandType::StartPleStripe, 1, 1, 0 },
        Command { CommandType::WaitForAgent, 2, 0, 0 },
        Command { CommandType::StartPleStripe, 1, 2, 0 },
        Command { CommandType::WaitForAgent, 2, 1, 0 },
        Command { CommandType::StartPleStripe, 1, 3, 0 },
        Command { CommandType::WaitForAgent, 2, 2, 0 },
        Command { CommandType::StartPleStripe, 1, 4, 0 },
        Command { CommandType::WaitForAgent, 2, 3, 0 },
        Command { CommandType::StartPleStripe, 1, 5, 0 },
        Command { CommandType::WaitForAgent, 2, 4, 0 },
        Command { CommandType::StartPleStripe, 1, 6, 0 },
        Command { CommandType::WaitForAgent, 2, 5, 0 },
        Command { CommandType::StartPleStripe, 1, 7, 0 },
        Command { CommandType::WaitForAgent, 2, 6, 0 },
        Command { CommandType::StartPleStripe, 1, 8, 0 },
        Command { CommandType::WaitForAgent, 2, 7, 0 },
        Command { CommandType::StartPleStripe, 1, 9, 0 },
        Command { CommandType::WaitForAgent, 2, 8, 0 },
        Command { CommandType::StartPleStripe, 1, 10, 0 },
        Command { CommandType::WaitForAgent, 2, 9, 0 },
        Command { CommandType::StartPleStripe, 1, 11, 0 }
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, debuggingContext);
    scheduler.Schedule();

    CHECK(scheduler.GetDmaRdCommands() == expectedDmaRdCommands);
    CHECK(scheduler.GetDmaWrCommands() == expectedDmaWrCommands);
    CHECK(scheduler.GetMceCommands() == expectedMceCommands);
    CHECK(scheduler.GetPleCommands() == expectedPleCommands);
}
