//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/DebuggingContext.hpp"
#include "../src/cascading/Scheduler.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>

using namespace ethosn::support_library;
using namespace ethosn::command_stream::cascading;
using CommandVariant = ethosn::command_stream::CommandVariant;

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

struct TestCommand
{
    CommandType type;
    uint32_t agentId;
    uint32_t stripeId;
};

inline bool operator==(const TestCommand& lhs, const TestCommand& rhs)
{
    static_assert(sizeof(TestCommand) == sizeof(lhs.type) + sizeof(lhs.agentId) + sizeof(lhs.stripeId),
                  "New fields added");
    return lhs.type == rhs.type && lhs.agentId == rhs.agentId && lhs.stripeId == rhs.stripeId;
}

namespace Catch
{
template <>
struct StringMaker<TestCommand>
{
    static std::string convert(const TestCommand& c)
    {
        return "\n  TestCommand { CommandType::" + std::string(CommandTypeToString(c.type)) + ", " +
               std::to_string(c.agentId) + ", " + std::to_string(c.stripeId) + ", 0 }";
    }
};
}    // namespace Catch

void CompareCommandArrays(const std::vector<CommandVariant>& a, const std::vector<TestCommand>& b)
{
    std::vector<TestCommand> converted;
    for (const CommandVariant& cmd : a)
    {
        TestCommand c;
        c.type = cmd.type;
        switch (cmd.type)
        {
            case CommandType::WaitForAgent:
                c.agentId  = cmd.waitForAgent.agentId;
                c.stripeId = cmd.waitForAgent.stripeId;
                break;
            case CommandType::LoadIfmStripe:
                c.agentId  = cmd.dma.agentId;
                c.stripeId = cmd.dma.stripeId;
                break;
            case CommandType::LoadWgtStripe:
                c.agentId  = cmd.dma.agentId;
                c.stripeId = cmd.dma.stripeId;
                break;
            case CommandType::ProgramMceStripe:
                c.agentId  = cmd.programMceStripe.agentId;
                c.stripeId = cmd.programMceStripe.stripeId;
                break;
            case CommandType::StartMceStripe:
                c.agentId  = cmd.startMceStripe.agentId;
                c.stripeId = cmd.startMceStripe.stripeId;
                break;
            case CommandType::LoadPleCode:
                c.agentId  = cmd.dma.agentId;
                c.stripeId = cmd.dma.stripeId;
                break;
            case CommandType::StartPleStripe:
                c.agentId  = cmd.startPleStripe.agentId;
                c.stripeId = cmd.startPleStripe.stripeId;
                break;
            case CommandType::StoreOfmStripe:
                c.agentId  = cmd.dma.agentId;
                c.stripeId = cmd.dma.stripeId;
                break;
            default:
                break;
        }
        converted.push_back(c);
    }
    CHECK(converted == b);
}

IfmSDesc MakeIfmSDesc()
{
    IfmSDesc result{};
    result.fmData.tile.numSlots          = 1;
    result.fmData.stripeIdStrides        = { 1, 1, 1 };
    result.fmData.numStripes             = { 1, 1, 1 };
    result.fmData.edgeStripeSize         = { 1, 1, 1 };
    result.fmData.supertensorSizeInCells = { 1, 1 };
    return result;
}

WgtSDesc MakeWgtSDesc()
{
    WgtSDesc result{};
    result.tile.numSlots   = 2;
    result.stripeIdStrides = { 1, 1 };
    result.numStripes      = { 1, 1 };
    static std::vector<WeightsMetadata> wgtMetadata{ { 0, 0 } };
    result.metadata = &wgtMetadata;
    return result;
}

MceSDesc MakeMceSDesc()
{
    MceSDesc result{};
    result.stripeIdStrides  = { 1, 1, 1, 1 };
    result.numStripes       = { 1, 1, 1, 1 };
    result.edgeStripeSize   = { 1, 1, 1, 1 };
    result.convStrideXy     = { 1, 1 };
    result.ifmTile.numSlots = 1;
    result.wgtTile.numSlots = 1;
    result.blockSize        = { 16, 16 };
    return result;
}

PleSDesc MakePleSDesc()
{
    PleSDesc result{};
    result.ofmTile.numSlots = 1;
    result.stripeIdStrides  = { 1, 1, 1 };
    result.numStripes       = { 1, 1, 1 };
    return result;
}

OfmSDesc MakeOfmSDesc()
{
    OfmSDesc result{};
    result.fmData.tile.numSlots          = 1;
    result.fmData.stripeIdStrides        = { 1, 1, 1 };
    result.fmData.numStripes             = { 1, 1, 1 };
    result.fmData.edgeStripeSize         = { 1, 1, 1 };
    result.fmData.supertensorSizeInCells = { 1, 1 };
    return result;
}

TEST_CASE("Cascading/Scheduler/ComplexSingleLayer")
{
    HardwareCapabilities caps = GetEthosN78HwCapabilities();
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
    IfmSDesc ifms             = MakeIfmSDesc();
    ifms.fmData.tile.numSlots = 4;

    WgtSDesc wgts      = MakeWgtSDesc();
    wgts.tile.numSlots = 2;

    PleSDesc ples         = MakePleSDesc();
    ples.ofmTile.numSlots = 2;

    OfmSDesc ofms             = MakeOfmSDesc();
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
            AgentDesc(9, MakeMceSDesc()),
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

    const std::vector<TestCommand> expectedDmaRdCommands{
        // clang-format off
        TestCommand { CommandType::LoadIfmStripe, 0, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 1 },
        TestCommand { CommandType::LoadIfmStripe, 0, 2 },
        TestCommand { CommandType::LoadWgtStripe, 1, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 3 },
        TestCommand { CommandType::WaitForAgent, 3, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 4 },
        TestCommand { CommandType::WaitForAgent, 3, 1 },
        TestCommand { CommandType::LoadIfmStripe, 0, 5 },
        TestCommand { CommandType::LoadIfmStripe, 0, 6 },
        TestCommand { CommandType::WaitForAgent, 3, 2 },
        TestCommand { CommandType::LoadIfmStripe, 0, 7 },
        TestCommand { CommandType::LoadIfmStripe, 0, 8 },
        TestCommand { CommandType::LoadWgtStripe, 1, 1 },
        TestCommand { CommandType::LoadIfmStripe, 0, 9 },
        TestCommand { CommandType::WaitForAgent, 3, 3 },
        TestCommand { CommandType::LoadIfmStripe, 0, 10 },
        TestCommand { CommandType::WaitForAgent, 3, 4 },
        TestCommand { CommandType::LoadIfmStripe, 0, 11 },
        TestCommand { CommandType::LoadIfmStripe, 0, 12 },
        TestCommand { CommandType::WaitForAgent, 3, 5 },
        TestCommand { CommandType::LoadIfmStripe, 0, 13 },
        TestCommand { CommandType::LoadIfmStripe, 0, 14 },
        TestCommand { CommandType::LoadWgtStripe, 1, 2 },
        TestCommand { CommandType::LoadIfmStripe, 0, 15 },
        TestCommand { CommandType::WaitForAgent, 3, 6 },
        TestCommand { CommandType::LoadIfmStripe, 0, 16 },
        TestCommand { CommandType::WaitForAgent, 3, 7 },
        TestCommand { CommandType::LoadIfmStripe, 0, 17 },
        TestCommand { CommandType::LoadPleCode, 2, 0 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedDmaWrCommands{
        TestCommand{ CommandType::WaitForAgent, 4, 0 },
        TestCommand{ CommandType::StoreOfmStripe, 5, 0 },
    };

    const std::vector<TestCommand> expectedMceCommands{
        // clang-format off
        TestCommand { CommandType::ProgramMceStripe, 3, 0 },
        TestCommand { CommandType::WaitForAgent, 0, 2 },
        TestCommand { CommandType::WaitForAgent, 1, 0 },
        TestCommand { CommandType::StartMceStripe, 3, 0 },
        TestCommand { CommandType::ProgramMceStripe, 3, 1 },
        TestCommand { CommandType::WaitForAgent, 0, 4 },
        TestCommand { CommandType::StartMceStripe, 3, 1 },
        TestCommand { CommandType::ProgramMceStripe, 3, 2 },
        TestCommand { CommandType::WaitForAgent, 0, 5 },
        TestCommand { CommandType::StartMceStripe, 3, 2 },
        TestCommand { CommandType::ProgramMceStripe, 3, 3 },
        TestCommand { CommandType::WaitForAgent, 0, 8 },
        TestCommand { CommandType::WaitForAgent, 1, 1 },
        TestCommand { CommandType::StartMceStripe, 3, 3 },
        TestCommand { CommandType::ProgramMceStripe, 3, 4 },
        TestCommand { CommandType::WaitForAgent, 0, 10 },
        TestCommand { CommandType::StartMceStripe, 3, 4 },
        TestCommand { CommandType::ProgramMceStripe, 3, 5 },
        TestCommand { CommandType::WaitForAgent, 0, 11 },
        TestCommand { CommandType::StartMceStripe, 3, 5 },
        TestCommand { CommandType::ProgramMceStripe, 3, 6 },
        TestCommand { CommandType::WaitForAgent, 0, 14 },
        TestCommand { CommandType::WaitForAgent, 1, 2 },
        TestCommand { CommandType::StartMceStripe, 3, 6 },
        TestCommand { CommandType::ProgramMceStripe, 3, 7 },
        TestCommand { CommandType::WaitForAgent, 0, 16 },
        TestCommand { CommandType::StartMceStripe, 3, 7 },
        TestCommand { CommandType::ProgramMceStripe, 3, 8 },
        TestCommand { CommandType::WaitForAgent, 0, 17 },
        TestCommand { CommandType::StartMceStripe, 3, 8 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedPleCommands{
        TestCommand{ CommandType::WaitForAgent, 2, 0 },
        TestCommand{ CommandType::StartPleStripe, 4, 0 },
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(complexSingleLayerCmdStream, caps, debuggingContext);
    scheduler.Schedule();

    CompareCommandArrays(scheduler.GetDmaRdCommands(), expectedDmaRdCommands);
    CompareCommandArrays(scheduler.GetDmaWrCommands(), expectedDmaWrCommands);
    CompareCommandArrays(scheduler.GetMceCommands(), expectedMceCommands);
    CompareCommandArrays(scheduler.GetPleCommands(), expectedPleCommands);
}

TEST_CASE("Cascading/Scheduler/Strategy7")
{
    HardwareCapabilities caps = GetEthosN78HwCapabilities();

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
    IfmSDesc ifms             = MakeIfmSDesc();
    ifms.fmData.tile.numSlots = 4;

    WgtSDesc wgts      = MakeWgtSDesc();
    wgts.tile.numSlots = 2;

    PleSDesc ples         = MakePleSDesc();
    ples.ofmTile.numSlots = 2;

    OfmSDesc ofms             = MakeOfmSDesc();
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
            AgentDesc(72, MakeMceSDesc()),
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

    const std::vector<TestCommand> expectedDmaRdCommands{
        // clang-format off
        TestCommand { CommandType::LoadIfmStripe, 0, 0 },
        TestCommand { CommandType::LoadWgtStripe, 1, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 1 },
        TestCommand { CommandType::LoadWgtStripe, 1, 1 },
        TestCommand { CommandType::LoadIfmStripe, 0, 2 },
        TestCommand { CommandType::LoadIfmStripe, 0, 3 },
        TestCommand { CommandType::WaitForAgent, 3, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 4 },
        TestCommand { CommandType::WaitForAgent, 3, 1 },
        TestCommand { CommandType::LoadIfmStripe, 0, 5 },
        TestCommand { CommandType::WaitForAgent, 3, 2 },
        TestCommand { CommandType::LoadIfmStripe, 0, 6 },
        TestCommand { CommandType::WaitForAgent, 3, 3 },
        TestCommand { CommandType::LoadIfmStripe, 0, 7 },
        TestCommand { CommandType::WaitForAgent, 3, 4 },
        TestCommand { CommandType::LoadIfmStripe, 0, 8 },
        TestCommand { CommandType::WaitForAgent, 3, 5 },
        TestCommand { CommandType::LoadIfmStripe, 0, 9 },
        TestCommand { CommandType::WaitForAgent, 3, 6 },
        TestCommand { CommandType::LoadIfmStripe, 0, 10 },
        TestCommand { CommandType::WaitForAgent, 3, 7 },
        TestCommand { CommandType::LoadIfmStripe, 0, 11 },
        TestCommand { CommandType::WaitForAgent, 3, 8 },
        TestCommand { CommandType::LoadIfmStripe, 0, 12 },
        TestCommand { CommandType::WaitForAgent, 3, 9 },
        TestCommand { CommandType::LoadIfmStripe, 0, 13 },
        TestCommand { CommandType::WaitForAgent, 3, 10 },
        TestCommand { CommandType::LoadIfmStripe, 0, 14 },
        TestCommand { CommandType::WaitForAgent, 3, 11 },
        TestCommand { CommandType::LoadIfmStripe, 0, 15 },
        TestCommand { CommandType::WaitForAgent, 3, 12 },
        TestCommand { CommandType::LoadIfmStripe, 0, 16 },
        TestCommand { CommandType::WaitForAgent, 3, 13 },
        TestCommand { CommandType::LoadIfmStripe, 0, 17 },
        TestCommand { CommandType::WaitForAgent, 3, 14 },
        TestCommand { CommandType::LoadIfmStripe, 0, 18 },
        TestCommand { CommandType::WaitForAgent, 3, 15 },
        TestCommand { CommandType::LoadIfmStripe, 0, 19 },
        TestCommand { CommandType::WaitForAgent, 3, 16 },
        TestCommand { CommandType::LoadIfmStripe, 0, 20 },
        TestCommand { CommandType::WaitForAgent, 3, 17 },
        TestCommand { CommandType::LoadIfmStripe, 0, 21 },
        TestCommand { CommandType::WaitForAgent, 3, 18 },
        TestCommand { CommandType::LoadIfmStripe, 0, 22 },
        TestCommand { CommandType::WaitForAgent, 3, 19 },
        TestCommand { CommandType::LoadIfmStripe, 0, 23 },
        TestCommand { CommandType::WaitForAgent, 3, 20 },
        TestCommand { CommandType::LoadIfmStripe, 0, 24 },
        TestCommand { CommandType::LoadWgtStripe, 1, 2 },
        TestCommand { CommandType::WaitForAgent, 3, 21 },
        TestCommand { CommandType::LoadIfmStripe, 0, 25 },
        TestCommand { CommandType::LoadWgtStripe, 1, 3 },
        TestCommand { CommandType::WaitForAgent, 3, 22 },
        TestCommand { CommandType::LoadIfmStripe, 0, 26 },
        TestCommand { CommandType::WaitForAgent, 3, 23 },
        TestCommand { CommandType::LoadIfmStripe, 0, 27 },
        TestCommand { CommandType::WaitForAgent, 3, 24 },
        TestCommand { CommandType::LoadIfmStripe, 0, 28 },
        TestCommand { CommandType::WaitForAgent, 3, 25 },
        TestCommand { CommandType::LoadIfmStripe, 0, 29 },
        TestCommand { CommandType::WaitForAgent, 3, 26 },
        TestCommand { CommandType::LoadIfmStripe, 0, 30 },
        TestCommand { CommandType::WaitForAgent, 3, 27 },
        TestCommand { CommandType::LoadIfmStripe, 0, 31 },
        TestCommand { CommandType::WaitForAgent, 3, 28 },
        TestCommand { CommandType::LoadIfmStripe, 0, 32 },
        TestCommand { CommandType::WaitForAgent, 3, 29 },
        TestCommand { CommandType::LoadIfmStripe, 0, 33 },
        TestCommand { CommandType::WaitForAgent, 3, 30 },
        TestCommand { CommandType::LoadIfmStripe, 0, 34 },
        TestCommand { CommandType::WaitForAgent, 3, 31 },
        TestCommand { CommandType::LoadIfmStripe, 0, 35 },
        TestCommand { CommandType::WaitForAgent, 3, 32 },
        TestCommand { CommandType::LoadIfmStripe, 0, 36 },
        TestCommand { CommandType::WaitForAgent, 3, 33 },
        TestCommand { CommandType::LoadIfmStripe, 0, 37 },
        TestCommand { CommandType::WaitForAgent, 3, 34 },
        TestCommand { CommandType::LoadIfmStripe, 0, 38 },
        TestCommand { CommandType::WaitForAgent, 3, 35 },
        TestCommand { CommandType::LoadIfmStripe, 0, 39 },
        TestCommand { CommandType::WaitForAgent, 3, 36 },
        TestCommand { CommandType::LoadIfmStripe, 0, 40 },
        TestCommand { CommandType::WaitForAgent, 3, 37 },
        TestCommand { CommandType::LoadIfmStripe, 0, 41 },
        TestCommand { CommandType::WaitForAgent, 3, 38 },
        TestCommand { CommandType::LoadIfmStripe, 0, 42 },
        TestCommand { CommandType::WaitForAgent, 3, 39 },
        TestCommand { CommandType::LoadIfmStripe, 0, 43 },
        TestCommand { CommandType::WaitForAgent, 3, 40 },
        TestCommand { CommandType::LoadIfmStripe, 0, 44 },
        TestCommand { CommandType::WaitForAgent, 3, 41 },
        TestCommand { CommandType::LoadIfmStripe, 0, 45 },
        TestCommand { CommandType::WaitForAgent, 3, 42 },
        TestCommand { CommandType::LoadIfmStripe, 0, 46 },
        TestCommand { CommandType::WaitForAgent, 3, 43 },
        TestCommand { CommandType::LoadIfmStripe, 0, 47 },
        TestCommand { CommandType::WaitForAgent, 3, 44 },
        TestCommand { CommandType::LoadIfmStripe, 0, 48 },
        TestCommand { CommandType::LoadWgtStripe, 1, 4 },
        TestCommand { CommandType::WaitForAgent, 3, 45 },
        TestCommand { CommandType::LoadIfmStripe, 0, 49 },
        TestCommand { CommandType::LoadWgtStripe, 1, 5 },
        TestCommand { CommandType::WaitForAgent, 3, 46 },
        TestCommand { CommandType::LoadIfmStripe, 0, 50 },
        TestCommand { CommandType::WaitForAgent, 3, 47 },
        TestCommand { CommandType::LoadIfmStripe, 0, 51 },
        TestCommand { CommandType::WaitForAgent, 3, 48 },
        TestCommand { CommandType::LoadIfmStripe, 0, 52 },
        TestCommand { CommandType::WaitForAgent, 3, 49 },
        TestCommand { CommandType::LoadIfmStripe, 0, 53 },
        TestCommand { CommandType::WaitForAgent, 3, 50 },
        TestCommand { CommandType::LoadIfmStripe, 0, 54 },
        TestCommand { CommandType::WaitForAgent, 3, 51 },
        TestCommand { CommandType::LoadIfmStripe, 0, 55 },
        TestCommand { CommandType::WaitForAgent, 3, 52 },
        TestCommand { CommandType::LoadIfmStripe, 0, 56 },
        TestCommand { CommandType::WaitForAgent, 3, 53 },
        TestCommand { CommandType::LoadIfmStripe, 0, 57 },
        TestCommand { CommandType::WaitForAgent, 3, 54 },
        TestCommand { CommandType::LoadIfmStripe, 0, 58 },
        TestCommand { CommandType::WaitForAgent, 3, 55 },
        TestCommand { CommandType::LoadIfmStripe, 0, 59 },
        TestCommand { CommandType::WaitForAgent, 3, 56 },
        TestCommand { CommandType::LoadIfmStripe, 0, 60 },
        TestCommand { CommandType::WaitForAgent, 3, 57 },
        TestCommand { CommandType::LoadIfmStripe, 0, 61 },
        TestCommand { CommandType::WaitForAgent, 3, 58 },
        TestCommand { CommandType::LoadIfmStripe, 0, 62 },
        TestCommand { CommandType::WaitForAgent, 3, 59 },
        TestCommand { CommandType::LoadIfmStripe, 0, 63 },
        TestCommand { CommandType::WaitForAgent, 3, 60 },
        TestCommand { CommandType::LoadIfmStripe, 0, 64 },
        TestCommand { CommandType::WaitForAgent, 3, 61 },
        TestCommand { CommandType::LoadIfmStripe, 0, 65 },
        TestCommand { CommandType::WaitForAgent, 3, 62 },
        TestCommand { CommandType::LoadIfmStripe, 0, 66 },
        TestCommand { CommandType::WaitForAgent, 3, 63 },
        TestCommand { CommandType::LoadIfmStripe, 0, 67 },
        TestCommand { CommandType::WaitForAgent, 3, 64 },
        TestCommand { CommandType::LoadIfmStripe, 0, 68 },
        TestCommand { CommandType::WaitForAgent, 3, 65 },
        TestCommand { CommandType::LoadIfmStripe, 0, 69 },
        TestCommand { CommandType::WaitForAgent, 3, 66 },
        TestCommand { CommandType::LoadIfmStripe, 0, 70 },
        TestCommand { CommandType::WaitForAgent, 3, 67 },
        TestCommand { CommandType::LoadIfmStripe, 0, 71 }
        // clang-format on
    };
    const std::vector<TestCommand> expectedDmaWrCommands{
        //clang-format off
        TestCommand{ CommandType::WaitForAgent, 4, 0 }, TestCommand{ CommandType::StoreOfmStripe, 5, 0 },
        TestCommand{ CommandType::WaitForAgent, 4, 1 }, TestCommand{ CommandType::StoreOfmStripe, 5, 1 },
        TestCommand{ CommandType::WaitForAgent, 4, 2 }, TestCommand{ CommandType::StoreOfmStripe, 5, 2 },
        TestCommand{ CommandType::WaitForAgent, 4, 3 }, TestCommand{ CommandType::StoreOfmStripe, 5, 3 },
        TestCommand{ CommandType::WaitForAgent, 4, 4 }, TestCommand{ CommandType::StoreOfmStripe, 5, 4 },
        TestCommand{ CommandType::WaitForAgent, 4, 5 }, TestCommand{ CommandType::StoreOfmStripe, 5, 5 },
        TestCommand{ CommandType::WaitForAgent, 4, 6 }, TestCommand{ CommandType::StoreOfmStripe, 5, 6 },
        TestCommand{ CommandType::WaitForAgent, 4, 7 }, TestCommand{ CommandType::StoreOfmStripe, 5, 7 },
        TestCommand{ CommandType::WaitForAgent, 4, 8 }, TestCommand{ CommandType::StoreOfmStripe, 5, 8 },
        //clang-format on
    };

    const std::vector<TestCommand> expectedMceCommands{
        // clang-format off
        TestCommand { CommandType::ProgramMceStripe, 3, 0 },
        TestCommand { CommandType::WaitForAgent, 0, 0 },
        TestCommand { CommandType::WaitForAgent, 1, 0 },
        TestCommand { CommandType::StartMceStripe, 3, 0 },
        TestCommand { CommandType::ProgramMceStripe, 3, 1 },
        TestCommand { CommandType::WaitForAgent, 0, 1 },
        TestCommand { CommandType::WaitForAgent, 1, 1 },
        TestCommand { CommandType::StartMceStripe, 3, 1 },
        TestCommand { CommandType::ProgramMceStripe, 3, 2 },
        TestCommand { CommandType::WaitForAgent, 0, 2 },
        TestCommand { CommandType::StartMceStripe, 3, 2 },
        TestCommand { CommandType::ProgramMceStripe, 3, 3 },
        TestCommand { CommandType::WaitForAgent, 0, 3 },
        TestCommand { CommandType::StartMceStripe, 3, 3 },
        TestCommand { CommandType::ProgramMceStripe, 3, 4 },
        TestCommand { CommandType::WaitForAgent, 0, 4 },
        TestCommand { CommandType::StartMceStripe, 3, 4 },
        TestCommand { CommandType::ProgramMceStripe, 3, 5 },
        TestCommand { CommandType::WaitForAgent, 0, 5 },
        TestCommand { CommandType::StartMceStripe, 3, 5 },
        TestCommand { CommandType::ProgramMceStripe, 3, 6 },
        TestCommand { CommandType::WaitForAgent, 0, 6 },
        TestCommand { CommandType::StartMceStripe, 3, 6 },
        TestCommand { CommandType::ProgramMceStripe, 3, 7 },
        TestCommand { CommandType::WaitForAgent, 0, 7 },
        TestCommand { CommandType::StartMceStripe, 3, 7 },
        TestCommand { CommandType::ProgramMceStripe, 3, 8 },
        TestCommand { CommandType::WaitForAgent, 0, 8 },
        TestCommand { CommandType::StartMceStripe, 3, 8 },
        TestCommand { CommandType::ProgramMceStripe, 3, 9 },
        TestCommand { CommandType::WaitForAgent, 0, 9 },
        TestCommand { CommandType::StartMceStripe, 3, 9 },
        TestCommand { CommandType::ProgramMceStripe, 3, 10 },
        TestCommand { CommandType::WaitForAgent, 0, 10 },
        TestCommand { CommandType::StartMceStripe, 3, 10 },
        TestCommand { CommandType::ProgramMceStripe, 3, 11 },
        TestCommand { CommandType::WaitForAgent, 0, 11 },
        TestCommand { CommandType::StartMceStripe, 3, 11 },
        TestCommand { CommandType::ProgramMceStripe, 3, 12 },
        TestCommand { CommandType::WaitForAgent, 0, 12 },
        TestCommand { CommandType::StartMceStripe, 3, 12 },
        TestCommand { CommandType::ProgramMceStripe, 3, 13 },
        TestCommand { CommandType::WaitForAgent, 0, 13 },
        TestCommand { CommandType::StartMceStripe, 3, 13 },
        TestCommand { CommandType::ProgramMceStripe, 3, 14 },
        TestCommand { CommandType::WaitForAgent, 0, 14 },
        TestCommand { CommandType::StartMceStripe, 3, 14 },
        TestCommand { CommandType::ProgramMceStripe, 3, 15 },
        TestCommand { CommandType::WaitForAgent, 0, 15 },
        TestCommand { CommandType::StartMceStripe, 3, 15 },
        TestCommand { CommandType::ProgramMceStripe, 3, 16 },
        TestCommand { CommandType::WaitForAgent, 0, 16 },
        TestCommand { CommandType::StartMceStripe, 3, 16 },
        TestCommand { CommandType::ProgramMceStripe, 3, 17 },
        TestCommand { CommandType::WaitForAgent, 0, 17 },
        TestCommand { CommandType::StartMceStripe, 3, 17 },
        TestCommand { CommandType::ProgramMceStripe, 3, 18 },
        TestCommand { CommandType::WaitForAgent, 0, 18 },
        TestCommand { CommandType::StartMceStripe, 3, 18 },
        TestCommand { CommandType::ProgramMceStripe, 3, 19 },
        TestCommand { CommandType::WaitForAgent, 0, 19 },
        TestCommand { CommandType::StartMceStripe, 3, 19 },
        TestCommand { CommandType::ProgramMceStripe, 3, 20 },
        TestCommand { CommandType::WaitForAgent, 0, 20 },
        TestCommand { CommandType::StartMceStripe, 3, 20 },
        TestCommand { CommandType::ProgramMceStripe, 3, 21 },
        TestCommand { CommandType::WaitForAgent, 0, 21 },
        TestCommand { CommandType::StartMceStripe, 3, 21 },
        TestCommand { CommandType::ProgramMceStripe, 3, 22 },
        TestCommand { CommandType::WaitForAgent, 0, 22 },
        TestCommand { CommandType::StartMceStripe, 3, 22 },
        TestCommand { CommandType::ProgramMceStripe, 3, 23 },
        TestCommand { CommandType::WaitForAgent, 0, 23 },
        TestCommand { CommandType::StartMceStripe, 3, 23 },
        TestCommand { CommandType::ProgramMceStripe, 3, 24 },
        TestCommand { CommandType::WaitForAgent, 0, 24 },
        TestCommand { CommandType::WaitForAgent, 1, 2 },
        TestCommand { CommandType::StartMceStripe, 3, 24 },
        TestCommand { CommandType::ProgramMceStripe, 3, 25 },
        TestCommand { CommandType::WaitForAgent, 0, 25 },
        TestCommand { CommandType::WaitForAgent, 1, 3 },
        TestCommand { CommandType::StartMceStripe, 3, 25 },
        TestCommand { CommandType::ProgramMceStripe, 3, 26 },
        TestCommand { CommandType::WaitForAgent, 0, 26 },
        TestCommand { CommandType::StartMceStripe, 3, 26 },
        TestCommand { CommandType::ProgramMceStripe, 3, 27 },
        TestCommand { CommandType::WaitForAgent, 0, 27 },
        TestCommand { CommandType::StartMceStripe, 3, 27 },
        TestCommand { CommandType::ProgramMceStripe, 3, 28 },
        TestCommand { CommandType::WaitForAgent, 0, 28 },
        TestCommand { CommandType::StartMceStripe, 3, 28 },
        TestCommand { CommandType::ProgramMceStripe, 3, 29 },
        TestCommand { CommandType::WaitForAgent, 0, 29 },
        TestCommand { CommandType::StartMceStripe, 3, 29 },
        TestCommand { CommandType::ProgramMceStripe, 3, 30 },
        TestCommand { CommandType::WaitForAgent, 0, 30 },
        TestCommand { CommandType::StartMceStripe, 3, 30 },
        TestCommand { CommandType::ProgramMceStripe, 3, 31 },
        TestCommand { CommandType::WaitForAgent, 0, 31 },
        TestCommand { CommandType::StartMceStripe, 3, 31 },
        TestCommand { CommandType::ProgramMceStripe, 3, 32 },
        TestCommand { CommandType::WaitForAgent, 0, 32 },
        TestCommand { CommandType::StartMceStripe, 3, 32 },
        TestCommand { CommandType::ProgramMceStripe, 3, 33 },
        TestCommand { CommandType::WaitForAgent, 0, 33 },
        TestCommand { CommandType::StartMceStripe, 3, 33 },
        TestCommand { CommandType::ProgramMceStripe, 3, 34 },
        TestCommand { CommandType::WaitForAgent, 0, 34 },
        TestCommand { CommandType::StartMceStripe, 3, 34 },
        TestCommand { CommandType::ProgramMceStripe, 3, 35 },
        TestCommand { CommandType::WaitForAgent, 0, 35 },
        TestCommand { CommandType::StartMceStripe, 3, 35 },
        TestCommand { CommandType::ProgramMceStripe, 3, 36 },
        TestCommand { CommandType::WaitForAgent, 0, 36 },
        TestCommand { CommandType::StartMceStripe, 3, 36 },
        TestCommand { CommandType::ProgramMceStripe, 3, 37 },
        TestCommand { CommandType::WaitForAgent, 0, 37 },
        TestCommand { CommandType::StartMceStripe, 3, 37 },
        TestCommand { CommandType::ProgramMceStripe, 3, 38 },
        TestCommand { CommandType::WaitForAgent, 0, 38 },
        TestCommand { CommandType::StartMceStripe, 3, 38 },
        TestCommand { CommandType::ProgramMceStripe, 3, 39 },
        TestCommand { CommandType::WaitForAgent, 0, 39 },
        TestCommand { CommandType::StartMceStripe, 3, 39 },
        TestCommand { CommandType::ProgramMceStripe, 3, 40 },
        TestCommand { CommandType::WaitForAgent, 0, 40 },
        TestCommand { CommandType::StartMceStripe, 3, 40 },
        TestCommand { CommandType::ProgramMceStripe, 3, 41 },
        TestCommand { CommandType::WaitForAgent, 0, 41 },
        TestCommand { CommandType::StartMceStripe, 3, 41 },
        TestCommand { CommandType::ProgramMceStripe, 3, 42 },
        TestCommand { CommandType::WaitForAgent, 0, 42 },
        TestCommand { CommandType::StartMceStripe, 3, 42 },
        TestCommand { CommandType::ProgramMceStripe, 3, 43 },
        TestCommand { CommandType::WaitForAgent, 0, 43 },
        TestCommand { CommandType::StartMceStripe, 3, 43 },
        TestCommand { CommandType::ProgramMceStripe, 3, 44 },
        TestCommand { CommandType::WaitForAgent, 0, 44 },
        TestCommand { CommandType::StartMceStripe, 3, 44 },
        TestCommand { CommandType::ProgramMceStripe, 3, 45 },
        TestCommand { CommandType::WaitForAgent, 0, 45 },
        TestCommand { CommandType::StartMceStripe, 3, 45 },
        TestCommand { CommandType::ProgramMceStripe, 3, 46 },
        TestCommand { CommandType::WaitForAgent, 0, 46 },
        TestCommand { CommandType::StartMceStripe, 3, 46 },
        TestCommand { CommandType::ProgramMceStripe, 3, 47 },
        TestCommand { CommandType::WaitForAgent, 0, 47 },
        TestCommand { CommandType::StartMceStripe, 3, 47 },
        TestCommand { CommandType::ProgramMceStripe, 3, 48 },
        TestCommand { CommandType::WaitForAgent, 0, 48 },
        TestCommand { CommandType::WaitForAgent, 1, 4 },
        TestCommand { CommandType::StartMceStripe, 3, 48 },
        TestCommand { CommandType::ProgramMceStripe, 3, 49 },
        TestCommand { CommandType::WaitForAgent, 0, 49 },
        TestCommand { CommandType::WaitForAgent, 1, 5 },
        TestCommand { CommandType::StartMceStripe, 3, 49 },
        TestCommand { CommandType::ProgramMceStripe, 3, 50 },
        TestCommand { CommandType::WaitForAgent, 0, 50 },
        TestCommand { CommandType::StartMceStripe, 3, 50 },
        TestCommand { CommandType::ProgramMceStripe, 3, 51 },
        TestCommand { CommandType::WaitForAgent, 0, 51 },
        TestCommand { CommandType::StartMceStripe, 3, 51 },
        TestCommand { CommandType::ProgramMceStripe, 3, 52 },
        TestCommand { CommandType::WaitForAgent, 0, 52 },
        TestCommand { CommandType::StartMceStripe, 3, 52 },
        TestCommand { CommandType::ProgramMceStripe, 3, 53 },
        TestCommand { CommandType::WaitForAgent, 0, 53 },
        TestCommand { CommandType::StartMceStripe, 3, 53 },
        TestCommand { CommandType::ProgramMceStripe, 3, 54 },
        TestCommand { CommandType::WaitForAgent, 0, 54 },
        TestCommand { CommandType::StartMceStripe, 3, 54 },
        TestCommand { CommandType::ProgramMceStripe, 3, 55 },
        TestCommand { CommandType::WaitForAgent, 0, 55 },
        TestCommand { CommandType::StartMceStripe, 3, 55 },
        TestCommand { CommandType::ProgramMceStripe, 3, 56 },
        TestCommand { CommandType::WaitForAgent, 0, 56 },
        TestCommand { CommandType::StartMceStripe, 3, 56 },
        TestCommand { CommandType::ProgramMceStripe, 3, 57 },
        TestCommand { CommandType::WaitForAgent, 0, 57 },
        TestCommand { CommandType::StartMceStripe, 3, 57 },
        TestCommand { CommandType::ProgramMceStripe, 3, 58 },
        TestCommand { CommandType::WaitForAgent, 0, 58 },
        TestCommand { CommandType::StartMceStripe, 3, 58 },
        TestCommand { CommandType::ProgramMceStripe, 3, 59 },
        TestCommand { CommandType::WaitForAgent, 0, 59 },
        TestCommand { CommandType::StartMceStripe, 3, 59 },
        TestCommand { CommandType::ProgramMceStripe, 3, 60 },
        TestCommand { CommandType::WaitForAgent, 0, 60 },
        TestCommand { CommandType::StartMceStripe, 3, 60 },
        TestCommand { CommandType::ProgramMceStripe, 3, 61 },
        TestCommand { CommandType::WaitForAgent, 0, 61 },
        TestCommand { CommandType::StartMceStripe, 3, 61 },
        TestCommand { CommandType::ProgramMceStripe, 3, 62 },
        TestCommand { CommandType::WaitForAgent, 0, 62 },
        TestCommand { CommandType::StartMceStripe, 3, 62 },
        TestCommand { CommandType::ProgramMceStripe, 3, 63 },
        TestCommand { CommandType::WaitForAgent, 0, 63 },
        TestCommand { CommandType::StartMceStripe, 3, 63 },
        TestCommand { CommandType::ProgramMceStripe, 3, 64 },
        TestCommand { CommandType::WaitForAgent, 0, 64 },
        TestCommand { CommandType::StartMceStripe, 3, 64 },
        TestCommand { CommandType::ProgramMceStripe, 3, 65 },
        TestCommand { CommandType::WaitForAgent, 0, 65 },
        TestCommand { CommandType::StartMceStripe, 3, 65 },
        TestCommand { CommandType::ProgramMceStripe, 3, 66 },
        TestCommand { CommandType::WaitForAgent, 0, 66 },
        TestCommand { CommandType::StartMceStripe, 3, 66 },
        TestCommand { CommandType::ProgramMceStripe, 3, 67 },
        TestCommand { CommandType::WaitForAgent, 0, 67 },
        TestCommand { CommandType::StartMceStripe, 3, 67 },
        TestCommand { CommandType::ProgramMceStripe, 3, 68 },
        TestCommand { CommandType::WaitForAgent, 0, 68 },
        TestCommand { CommandType::StartMceStripe, 3, 68 },
        TestCommand { CommandType::ProgramMceStripe, 3, 69 },
        TestCommand { CommandType::WaitForAgent, 0, 69 },
        TestCommand { CommandType::StartMceStripe, 3, 69 },
        TestCommand { CommandType::ProgramMceStripe, 3, 70 },
        TestCommand { CommandType::WaitForAgent, 0, 70 },
        TestCommand { CommandType::StartMceStripe, 3, 70 },
        TestCommand { CommandType::ProgramMceStripe, 3, 71 },
        TestCommand { CommandType::WaitForAgent, 0, 71 },
        TestCommand { CommandType::StartMceStripe, 3, 71 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedPleCommands{
        // clang-format off
        TestCommand { CommandType::StartPleStripe, 4, 0 },
        TestCommand { CommandType::StartPleStripe, 4, 1 },
        TestCommand { CommandType::StartPleStripe, 4, 2 },
        TestCommand { CommandType::StartPleStripe, 4, 3 },
        TestCommand { CommandType::StartPleStripe, 4, 4 },
        TestCommand { CommandType::StartPleStripe, 4, 5 },
        TestCommand { CommandType::StartPleStripe, 4, 6 },
        TestCommand { CommandType::StartPleStripe, 4, 7 },
        TestCommand { CommandType::StartPleStripe, 4, 8 }
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(strategy7CmdStream, caps, debuggingContext);
    scheduler.Schedule();

    CompareCommandArrays(scheduler.GetDmaRdCommands(), expectedDmaRdCommands);
    CompareCommandArrays(scheduler.GetDmaWrCommands(), expectedDmaWrCommands);
    CompareCommandArrays(scheduler.GetMceCommands(), expectedMceCommands);
    CompareCommandArrays(scheduler.GetPleCommands(), expectedPleCommands);
}

TEST_CASE("Cascading/Scheduler/MultipleNonCascadedLayers")
{
    HardwareCapabilities caps = GetEthosN78HwCapabilities();

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
    IfmSDesc ifms             = MakeIfmSDesc();
    ifms.fmData.tile.numSlots = 4;

    WgtSDesc wgts      = MakeWgtSDesc();
    wgts.tile.numSlots = 2;

    PleSDesc ples         = MakePleSDesc();
    ples.ofmTile.numSlots = 2;

    OfmSDesc ofms             = MakeOfmSDesc();
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
            AgentDesc(3, MakeMceSDesc()),
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
            AgentDesc(3, MakeMceSDesc()),
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

    const std::vector<TestCommand> expectedDmaRdCommands{
        // clang-format off
        TestCommand { CommandType::LoadIfmStripe, 0, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 1 },
        TestCommand { CommandType::LoadIfmStripe, 0, 2 },
        TestCommand { CommandType::LoadWgtStripe, 1, 0 },
        TestCommand { CommandType::LoadPleCode, 2, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 3 },
        TestCommand { CommandType::WaitForAgent, 3, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 4 },
        TestCommand { CommandType::WaitForAgent, 3, 1 },
        TestCommand { CommandType::LoadIfmStripe, 0, 5 },
        TestCommand { CommandType::WaitForAgent, 3, 2 },
        TestCommand { CommandType::WaitForAgent, 5, 0 },
        TestCommand { CommandType::LoadIfmStripe, 6, 0 },
        TestCommand { CommandType::LoadIfmStripe, 6, 1 },
        TestCommand { CommandType::WaitForAgent, 5, 1 },
        TestCommand { CommandType::LoadIfmStripe, 6, 2 },
        TestCommand { CommandType::LoadWgtStripe, 7, 0 },
        TestCommand { CommandType::LoadPleCode, 8, 0 },
        TestCommand { CommandType::LoadIfmStripe, 6, 3 },
        TestCommand { CommandType::WaitForAgent, 9, 0 },
        TestCommand { CommandType::WaitForAgent, 5, 2 },
        TestCommand { CommandType::LoadIfmStripe, 6, 4 },
        TestCommand { CommandType::WaitForAgent, 9, 1 },
        TestCommand { CommandType::LoadIfmStripe, 6, 5 }
        // clang-format on
    };
    const std::vector<TestCommand> expectedDmaWrCommands{
        // clang-format off
        TestCommand { CommandType::WaitForAgent, 4, 0 },
        TestCommand { CommandType::StoreOfmStripe, 5, 0 },
        TestCommand { CommandType::WaitForAgent, 4, 1 },
        TestCommand { CommandType::StoreOfmStripe, 5, 1 },
        TestCommand { CommandType::WaitForAgent, 4, 2 },
        TestCommand { CommandType::StoreOfmStripe, 5, 2 },
        TestCommand { CommandType::WaitForAgent, 10, 0 },
        TestCommand { CommandType::StoreOfmStripe, 11, 0 },
        TestCommand { CommandType::WaitForAgent, 10, 1 },
        TestCommand { CommandType::StoreOfmStripe, 11, 1 },
        TestCommand { CommandType::WaitForAgent, 10, 2 },
        TestCommand { CommandType::StoreOfmStripe, 11, 2 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedMceCommands{
        // clang-format off
        TestCommand { CommandType::ProgramMceStripe, 3, 0 },
        TestCommand { CommandType::WaitForAgent, 0, 2 },
        TestCommand { CommandType::WaitForAgent, 1, 0 },
        TestCommand { CommandType::StartMceStripe, 3, 0 },
        TestCommand { CommandType::ProgramMceStripe, 3, 1 },
        TestCommand { CommandType::WaitForAgent, 0, 4 },
        TestCommand { CommandType::StartMceStripe, 3, 1 },
        TestCommand { CommandType::ProgramMceStripe, 3, 2 },
        TestCommand { CommandType::WaitForAgent, 0, 5 },
        TestCommand { CommandType::StartMceStripe, 3, 2 },
        TestCommand { CommandType::ProgramMceStripe, 9, 0 },
        TestCommand { CommandType::WaitForAgent, 6, 2 },
        TestCommand { CommandType::WaitForAgent, 7, 0 },
        TestCommand { CommandType::StartMceStripe, 9, 0 },
        TestCommand { CommandType::ProgramMceStripe, 9, 1 },
        TestCommand { CommandType::WaitForAgent, 6, 4 },
        TestCommand { CommandType::StartMceStripe, 9, 1 },
        TestCommand { CommandType::ProgramMceStripe, 9, 2 },
        TestCommand { CommandType::WaitForAgent, 6, 5 },
        TestCommand { CommandType::StartMceStripe, 9, 2 },
        // clang-format on
    };

    const std::vector<TestCommand> expectedPleCommands{
        // clang-format off
        TestCommand { CommandType::WaitForAgent, 2, 0 },
        TestCommand { CommandType::StartPleStripe, 4, 0 },
        TestCommand { CommandType::StartPleStripe, 4, 1 },
        TestCommand { CommandType::WaitForAgent, 5, 0 },
        TestCommand { CommandType::StartPleStripe, 4, 2 },
        TestCommand { CommandType::WaitForAgent, 8, 0 },
        TestCommand { CommandType::StartPleStripe, 10, 0 },
        TestCommand { CommandType::StartPleStripe, 10, 1 },
        TestCommand { CommandType::WaitForAgent, 11, 0 },
        TestCommand { CommandType::StartPleStripe, 10, 2 }
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(multipleNonCascadedLayersCmdStream, caps, debuggingContext);
    scheduler.Schedule();

    CompareCommandArrays(scheduler.GetDmaRdCommands(), expectedDmaRdCommands);
    CompareCommandArrays(scheduler.GetDmaWrCommands(), expectedDmaWrCommands);
    CompareCommandArrays(scheduler.GetMceCommands(), expectedMceCommands);
    CompareCommandArrays(scheduler.GetPleCommands(), expectedPleCommands);
}

TEST_CASE("Cascading/Scheduler/Strategy1Cascade")
{
    HardwareCapabilities caps = GetEthosN78HwCapabilities();

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
    IfmSDesc ifms             = MakeIfmSDesc();
    ifms.fmData.tile.numSlots = 4;

    WgtSDesc wgts      = MakeWgtSDesc();
    wgts.tile.numSlots = 2;

    PleSDesc ples         = MakePleSDesc();
    ples.ofmTile.numSlots = 2;

    OfmSDesc ofms             = MakeOfmSDesc();
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
            AgentDesc(9, MakeMceSDesc()),
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
            AgentDesc(12, MakeMceSDesc()),
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

    const std::vector<TestCommand> expectedDmaRdCommands{
        // clang-format off
        TestCommand { CommandType::LoadIfmStripe, 0, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 1 },
        TestCommand { CommandType::LoadIfmStripe, 0, 2 },
        TestCommand { CommandType::LoadWgtStripe, 1, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 3 },
        TestCommand { CommandType::WaitForAgent, 3, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 4 },
        TestCommand { CommandType::WaitForAgent, 3, 1 },
        TestCommand { CommandType::LoadIfmStripe, 0, 5 },
        TestCommand { CommandType::LoadIfmStripe, 0, 6 },
        TestCommand { CommandType::WaitForAgent, 3, 2 },
        TestCommand { CommandType::LoadIfmStripe, 0, 7 },
        TestCommand { CommandType::LoadIfmStripe, 0, 8 },
        TestCommand { CommandType::LoadWgtStripe, 1, 1 },
        TestCommand { CommandType::LoadIfmStripe, 0, 9 },
        TestCommand { CommandType::WaitForAgent, 3, 3 },
        TestCommand { CommandType::LoadIfmStripe, 0, 10 },
        TestCommand { CommandType::WaitForAgent, 3, 4 },
        TestCommand { CommandType::LoadIfmStripe, 0, 11 },
        TestCommand { CommandType::LoadIfmStripe, 0, 12 },
        TestCommand { CommandType::WaitForAgent, 3, 5 },
        TestCommand { CommandType::LoadIfmStripe, 0, 13 },
        TestCommand { CommandType::LoadIfmStripe, 0, 14 },
        TestCommand { CommandType::LoadWgtStripe, 1, 2 },
        TestCommand { CommandType::LoadIfmStripe, 0, 15 },
        TestCommand { CommandType::WaitForAgent, 3, 6 },
        TestCommand { CommandType::LoadIfmStripe, 0, 16 },
        TestCommand { CommandType::WaitForAgent, 3, 7 },
        TestCommand { CommandType::LoadIfmStripe, 0, 17 },
        TestCommand { CommandType::LoadPleCode, 2, 0 },
        TestCommand { CommandType::LoadWgtStripe, 5, 0 },
        TestCommand { CommandType::LoadPleCode, 6, 0 },
        TestCommand { CommandType::LoadWgtStripe, 5, 1 },
        TestCommand { CommandType::WaitForAgent, 7, 3 },
        TestCommand { CommandType::LoadWgtStripe, 5, 2 }
        // clang-format on
    };
    const std::vector<TestCommand> expectedDmaWrCommands{
        // clang-format off
        TestCommand { CommandType::WaitForAgent, 8, 0 },
        TestCommand { CommandType::StoreOfmStripe, 9, 0 },
        TestCommand { CommandType::WaitForAgent, 8, 1 },
        TestCommand { CommandType::StoreOfmStripe, 9, 1 },
        TestCommand { CommandType::WaitForAgent, 8, 2 },
        TestCommand { CommandType::StoreOfmStripe, 9, 2 },
        TestCommand { CommandType::WaitForAgent, 8, 3 },
        TestCommand { CommandType::StoreOfmStripe, 9, 3 },
        TestCommand { CommandType::WaitForAgent, 8, 4 },
        TestCommand { CommandType::StoreOfmStripe, 9, 4 },
        TestCommand { CommandType::WaitForAgent, 8, 5 },
        TestCommand { CommandType::StoreOfmStripe, 9, 5 },
        TestCommand { CommandType::WaitForAgent, 8, 6 },
        TestCommand { CommandType::StoreOfmStripe, 9, 6 },
        TestCommand { CommandType::WaitForAgent, 8, 7 },
        TestCommand { CommandType::StoreOfmStripe, 9, 7 },
        TestCommand { CommandType::WaitForAgent, 8, 8 },
        TestCommand { CommandType::StoreOfmStripe, 9, 8 },
        TestCommand { CommandType::WaitForAgent, 8, 9 },
        TestCommand { CommandType::StoreOfmStripe, 9, 9 },
        TestCommand { CommandType::WaitForAgent, 8, 10 },
        TestCommand { CommandType::StoreOfmStripe, 9, 10 },
        TestCommand { CommandType::WaitForAgent, 8, 11 },
        TestCommand { CommandType::StoreOfmStripe, 9, 11 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedMceCommands{
        // clang-format off
        TestCommand { CommandType::ProgramMceStripe, 3, 0 },
        TestCommand { CommandType::WaitForAgent, 0, 2 },
        TestCommand { CommandType::WaitForAgent, 1, 0 },
        TestCommand { CommandType::StartMceStripe, 3, 0 },
        TestCommand { CommandType::ProgramMceStripe, 3, 1 },
        TestCommand { CommandType::WaitForAgent, 0, 4 },
        TestCommand { CommandType::StartMceStripe, 3, 1 },
        TestCommand { CommandType::ProgramMceStripe, 3, 2 },
        TestCommand { CommandType::WaitForAgent, 0, 5 },
        TestCommand { CommandType::StartMceStripe, 3, 2 },
        TestCommand { CommandType::ProgramMceStripe, 3, 3 },
        TestCommand { CommandType::WaitForAgent, 0, 8 },
        TestCommand { CommandType::WaitForAgent, 1, 1 },
        TestCommand { CommandType::StartMceStripe, 3, 3 },
        TestCommand { CommandType::ProgramMceStripe, 3, 4 },
        TestCommand { CommandType::WaitForAgent, 0, 10 },
        TestCommand { CommandType::StartMceStripe, 3, 4 },
        TestCommand { CommandType::ProgramMceStripe, 3, 5 },
        TestCommand { CommandType::WaitForAgent, 0, 11 },
        TestCommand { CommandType::StartMceStripe, 3, 5 },
        TestCommand { CommandType::ProgramMceStripe, 3, 6 },
        TestCommand { CommandType::WaitForAgent, 0, 14 },
        TestCommand { CommandType::WaitForAgent, 1, 2 },
        TestCommand { CommandType::StartMceStripe, 3, 6 },
        TestCommand { CommandType::ProgramMceStripe, 3, 7 },
        TestCommand { CommandType::WaitForAgent, 0, 16 },
        TestCommand { CommandType::StartMceStripe, 3, 7 },
        TestCommand { CommandType::ProgramMceStripe, 3, 8 },
        TestCommand { CommandType::WaitForAgent, 0, 17 },
        TestCommand { CommandType::StartMceStripe, 3, 8 },
        TestCommand { CommandType::ProgramMceStripe, 7, 0 },
        TestCommand { CommandType::WaitForAgent, 4, 0 },
        TestCommand { CommandType::WaitForAgent, 5, 0 },
        TestCommand { CommandType::StartMceStripe, 7, 0 },
        TestCommand { CommandType::ProgramMceStripe, 7, 1 },
        TestCommand { CommandType::StartMceStripe, 7, 1 },
        TestCommand { CommandType::ProgramMceStripe, 7, 2 },
        TestCommand { CommandType::StartMceStripe, 7, 2 },
        TestCommand { CommandType::ProgramMceStripe, 7, 3 },
        TestCommand { CommandType::StartMceStripe, 7, 3 },
        TestCommand { CommandType::ProgramMceStripe, 7, 4 },
        TestCommand { CommandType::WaitForAgent, 5, 1 },
        TestCommand { CommandType::StartMceStripe, 7, 4 },
        TestCommand { CommandType::ProgramMceStripe, 7, 5 },
        TestCommand { CommandType::StartMceStripe, 7, 5 },
        TestCommand { CommandType::ProgramMceStripe, 7, 6 },
        TestCommand { CommandType::StartMceStripe, 7, 6 },
        TestCommand { CommandType::ProgramMceStripe, 7, 7 },
        TestCommand { CommandType::StartMceStripe, 7, 7 },
        TestCommand { CommandType::ProgramMceStripe, 7, 8 },
        TestCommand { CommandType::WaitForAgent, 5, 2 },
        TestCommand { CommandType::StartMceStripe, 7, 8 },
        TestCommand { CommandType::ProgramMceStripe, 7, 9 },
        TestCommand { CommandType::StartMceStripe, 7, 9 },
        TestCommand { CommandType::ProgramMceStripe, 7, 10 },
        TestCommand { CommandType::StartMceStripe, 7, 10 },
        TestCommand { CommandType::ProgramMceStripe, 7, 11 },
        TestCommand { CommandType::StartMceStripe, 7, 11 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedPleCommands{
        // clang-format off
        TestCommand { CommandType::WaitForAgent, 2, 0 },
        TestCommand { CommandType::StartPleStripe, 4, 0 },
        TestCommand { CommandType::WaitForAgent, 6, 0 },
        TestCommand { CommandType::StartPleStripe, 8, 0 },
        TestCommand { CommandType::StartPleStripe, 8, 1 },
        TestCommand { CommandType::WaitForAgent, 9, 0 },
        TestCommand { CommandType::StartPleStripe, 8, 2 },
        TestCommand { CommandType::WaitForAgent, 9, 1 },
        TestCommand { CommandType::StartPleStripe, 8, 3 },
        TestCommand { CommandType::WaitForAgent, 9, 2 },
        TestCommand { CommandType::StartPleStripe, 8, 4 },
        TestCommand { CommandType::WaitForAgent, 9, 3 },
        TestCommand { CommandType::StartPleStripe, 8, 5 },
        TestCommand { CommandType::WaitForAgent, 9, 4 },
        TestCommand { CommandType::StartPleStripe, 8, 6 },
        TestCommand { CommandType::WaitForAgent, 9, 5 },
        TestCommand { CommandType::StartPleStripe, 8, 7 },
        TestCommand { CommandType::WaitForAgent, 9, 6 },
        TestCommand { CommandType::StartPleStripe, 8, 8 },
        TestCommand { CommandType::WaitForAgent, 9, 7 },
        TestCommand { CommandType::StartPleStripe, 8, 9 },
        TestCommand { CommandType::WaitForAgent, 9, 8 },
        TestCommand { CommandType::StartPleStripe, 8, 10 },
        TestCommand { CommandType::WaitForAgent, 9, 9 },
        TestCommand { CommandType::StartPleStripe, 8, 11 }
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(strategy1CascadeCmdStream, caps, debuggingContext);
    scheduler.Schedule();

    CompareCommandArrays(scheduler.GetDmaRdCommands(), expectedDmaRdCommands);
    CompareCommandArrays(scheduler.GetDmaWrCommands(), expectedDmaWrCommands);
    CompareCommandArrays(scheduler.GetMceCommands(), expectedMceCommands);
    CompareCommandArrays(scheduler.GetPleCommands(), expectedPleCommands);
}

TEST_CASE("Cascading/Scheduler/Strategy0Cascade")
{
    HardwareCapabilities caps = GetEthosN78HwCapabilities();

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
    IfmSDesc ifms             = MakeIfmSDesc();
    ifms.fmData.tile.numSlots = 4;

    WgtSDesc wgts      = MakeWgtSDesc();
    wgts.tile.numSlots = 2;

    PleSDesc ples         = MakePleSDesc();
    ples.ofmTile.numSlots = 2;

    OfmSDesc ofms             = MakeOfmSDesc();
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
            AgentDesc(5, MakeMceSDesc()),
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
            AgentDesc(4, MakeMceSDesc()),
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

    const std::vector<TestCommand> expectedDmaRdCommands{
        // clang-format off
        TestCommand { CommandType::LoadIfmStripe, 0, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 1 },
        TestCommand { CommandType::LoadIfmStripe, 0, 2 },
        TestCommand { CommandType::LoadWgtStripe, 1, 0 },
        TestCommand { CommandType::LoadPleCode, 2, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 3 },
        TestCommand { CommandType::WaitForAgent, 3, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 4 },
        TestCommand { CommandType::LoadWgtStripe, 5, 0 },
        TestCommand { CommandType::WaitForAgent, 4, 1 },
        TestCommand { CommandType::LoadPleCode, 6, 0 },
        TestCommand { CommandType::WaitForAgent, 3, 1 },
        TestCommand { CommandType::LoadIfmStripe, 0, 5 },
        TestCommand { CommandType::LoadIfmStripe, 0, 6 },
        TestCommand { CommandType::WaitForAgent, 8, 0 },
        TestCommand { CommandType::LoadPleCode, 2, 1 },
        TestCommand { CommandType::WaitForAgent, 4, 2 },
        TestCommand { CommandType::LoadPleCode, 6, 1 },
        TestCommand { CommandType::WaitForAgent, 3, 2 },
        TestCommand { CommandType::LoadIfmStripe, 0, 7 },
        TestCommand { CommandType::LoadIfmStripe, 0, 8 },
        TestCommand { CommandType::WaitForAgent, 8, 1 },
        TestCommand { CommandType::LoadPleCode, 2, 2 },
        TestCommand { CommandType::WaitForAgent, 4, 3 },
        TestCommand { CommandType::LoadPleCode, 6, 2 },
        TestCommand { CommandType::WaitForAgent, 8, 2 },
        TestCommand { CommandType::LoadPleCode, 2, 3 },
        TestCommand { CommandType::WaitForAgent, 4, 4 },
        TestCommand { CommandType::LoadPleCode, 6, 3 }
        // clang-format on
    };
    const std::vector<TestCommand> expectedDmaWrCommands{
        // clang-format off
        TestCommand { CommandType::WaitForAgent, 8, 0 },
        TestCommand { CommandType::StoreOfmStripe, 9, 0 },
        TestCommand { CommandType::WaitForAgent, 8, 1 },
        TestCommand { CommandType::StoreOfmStripe, 9, 1 },
        TestCommand { CommandType::WaitForAgent, 8, 2 },
        TestCommand { CommandType::StoreOfmStripe, 9, 2 },
        TestCommand { CommandType::WaitForAgent, 8, 3 },
        TestCommand { CommandType::StoreOfmStripe, 9, 3 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedMceCommands{
        // clang-format off
        TestCommand { CommandType::ProgramMceStripe, 3, 0 },
        TestCommand { CommandType::WaitForAgent, 0, 2 },
        TestCommand { CommandType::WaitForAgent, 1, 0 },
        TestCommand { CommandType::StartMceStripe, 3, 0 },
        TestCommand { CommandType::ProgramMceStripe, 3, 1 },
        TestCommand { CommandType::WaitForAgent, 0, 4 },
        TestCommand { CommandType::StartMceStripe, 3, 1 },
        TestCommand { CommandType::ProgramMceStripe, 7, 0 },
        TestCommand { CommandType::WaitForAgent, 4, 1 },
        TestCommand { CommandType::WaitForAgent, 5, 0 },
        TestCommand { CommandType::StartMceStripe, 7, 0 },
        TestCommand { CommandType::ProgramMceStripe, 3, 2 },
        TestCommand { CommandType::WaitForAgent, 0, 6 },
        TestCommand { CommandType::StartMceStripe, 3, 2 },
        TestCommand { CommandType::ProgramMceStripe, 7, 1 },
        TestCommand { CommandType::WaitForAgent, 4, 2 },
        TestCommand { CommandType::StartMceStripe, 7, 1 },
        TestCommand { CommandType::ProgramMceStripe, 3, 3 },
        TestCommand { CommandType::WaitForAgent, 0, 8 },
        TestCommand { CommandType::StartMceStripe, 3, 3 },
        TestCommand { CommandType::ProgramMceStripe, 7, 2 },
        TestCommand { CommandType::WaitForAgent, 4, 3 },
        TestCommand { CommandType::StartMceStripe, 7, 2 },
        TestCommand { CommandType::ProgramMceStripe, 3, 4 },
        TestCommand { CommandType::StartMceStripe, 3, 4 },
        TestCommand { CommandType::ProgramMceStripe, 7, 3 },
        TestCommand { CommandType::WaitForAgent, 4, 4 },
        TestCommand { CommandType::StartMceStripe, 7, 3 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedPleCommands{
        // clang-format off
        TestCommand { CommandType::WaitForAgent, 2, 0 },
        TestCommand { CommandType::StartPleStripe, 4, 0 },
        TestCommand { CommandType::StartPleStripe, 4, 1 },
        TestCommand { CommandType::WaitForAgent, 6, 0 },
        TestCommand { CommandType::StartPleStripe, 8, 0 },
        TestCommand { CommandType::WaitForAgent, 2, 1 },
        TestCommand { CommandType::StartPleStripe, 4, 2 },
        TestCommand { CommandType::WaitForAgent, 6, 1 },
        TestCommand { CommandType::StartPleStripe, 8, 1 },
        TestCommand { CommandType::WaitForAgent, 2, 2 },
        TestCommand { CommandType::StartPleStripe, 4, 3 },
        TestCommand { CommandType::WaitForAgent, 9, 0 },
        TestCommand { CommandType::WaitForAgent, 6, 2 },
        TestCommand { CommandType::StartPleStripe, 8, 2 },
        TestCommand { CommandType::WaitForAgent, 2, 3 },
        TestCommand { CommandType::StartPleStripe, 4, 4 },
        TestCommand { CommandType::WaitForAgent, 9, 1 },
        TestCommand { CommandType::WaitForAgent, 6, 3 },
        TestCommand { CommandType::StartPleStripe, 8, 3 }
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(strategy0CascadeCmdStream, caps, debuggingContext);
    scheduler.Schedule();

    CompareCommandArrays(scheduler.GetDmaRdCommands(), expectedDmaRdCommands);
    CompareCommandArrays(scheduler.GetDmaWrCommands(), expectedDmaWrCommands);
    CompareCommandArrays(scheduler.GetMceCommands(), expectedMceCommands);
    CompareCommandArrays(scheduler.GetPleCommands(), expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/IfmStreamer/WriteDependencies/FirstTile")
{
    HardwareCapabilities caps = GetEthosN78HwCapabilities();

    const uint32_t tileSize   = 4;
    IfmSDesc ifms             = MakeIfmSDesc();
    ifms.fmData.tile.numSlots = tileSize;

    WgtSDesc wgts      = MakeWgtSDesc();
    wgts.tile.numSlots = 2;

    std::vector<AgentDescAndDeps> cmdStream{
        // The first agent in the command stream is dummy, and it is there just
        // to make sure that we don't use agent ID 0. This help to validate
        // that the relative agent id field is properly used by the
        // scheduler function
        AgentDescAndDeps{ AgentDesc(0, MakeIfmSDesc()), {} },
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
            AgentDesc(9, MakeMceSDesc()),
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
            AgentDesc(1, MakeOfmSDesc()),
            {
                /* .readDependencies =*/{ { { 1, { 9, 1 }, { 9, 1 }, 0 } } },
                /* .writeDependencies =*/{},
            },
        },
    };

    const std::vector<TestCommand> expectedDmaWrCommands{
        //clang-format off
        TestCommand{ CommandType::WaitForAgent, 4, 8 }, TestCommand{ CommandType::StoreOfmStripe, 5, 0 }
        //clang-format on
    };

    const std::vector<TestCommand> expectedPleCommands{
        // clang-format off
        // clang-format on
    };

    const std::vector<TestCommand> expectedDmaRdCommands{
        // clang-format off
        TestCommand { CommandType::LoadIfmStripe, 1, 0 },
        TestCommand { CommandType::LoadIfmStripe, 1, 1 },
        TestCommand { CommandType::LoadIfmStripe, 1, 2 },
        TestCommand { CommandType::LoadWgtStripe, 2, 0 },
        TestCommand { CommandType::LoadIfmStripe, 1, 3 },
        TestCommand { CommandType::WaitForAgent, 4, 0 },
        TestCommand { CommandType::LoadIfmStripe, 1, 4 },
        TestCommand { CommandType::WaitForAgent, 4, 1 },
        TestCommand { CommandType::LoadIfmStripe, 1, 5 },
        TestCommand { CommandType::LoadIfmStripe, 1, 6 },
        TestCommand { CommandType::WaitForAgent, 4, 2 },
        TestCommand { CommandType::LoadIfmStripe, 1, 7 },
        TestCommand { CommandType::LoadIfmStripe, 1, 8 },
        TestCommand { CommandType::LoadWgtStripe, 2, 1 },
        TestCommand { CommandType::LoadIfmStripe, 1, 9 },
        TestCommand { CommandType::WaitForAgent, 4, 3 },
        TestCommand { CommandType::LoadIfmStripe, 1, 10 },
        TestCommand { CommandType::WaitForAgent, 4, 4 },
        TestCommand { CommandType::LoadIfmStripe, 1, 11 },
        TestCommand { CommandType::LoadIfmStripe, 1, 12 },
        TestCommand { CommandType::WaitForAgent, 4, 5 },
        TestCommand { CommandType::LoadIfmStripe, 1, 13 },
        TestCommand { CommandType::LoadIfmStripe, 1, 14 },
        TestCommand { CommandType::LoadWgtStripe, 2, 2 },
        TestCommand { CommandType::LoadIfmStripe, 1, 15 },
        TestCommand { CommandType::WaitForAgent, 4, 6 },
        TestCommand { CommandType::LoadIfmStripe, 1, 16 },
        TestCommand { CommandType::WaitForAgent, 4, 7 },
        TestCommand { CommandType::LoadIfmStripe, 1, 17 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedMceCommands{
        // clang-format off
        TestCommand { CommandType::ProgramMceStripe, 4, 0 },
        TestCommand { CommandType::WaitForAgent, 1, 2 },
        TestCommand { CommandType::WaitForAgent, 2, 0 },
        TestCommand { CommandType::StartMceStripe, 4, 0 },
        TestCommand { CommandType::ProgramMceStripe, 4, 1 },
        TestCommand { CommandType::WaitForAgent, 1, 4 },
        TestCommand { CommandType::StartMceStripe, 4, 1 },
        TestCommand { CommandType::ProgramMceStripe, 4, 2 },
        TestCommand { CommandType::WaitForAgent, 1, 5 },
        TestCommand { CommandType::StartMceStripe, 4, 2 },
        TestCommand { CommandType::ProgramMceStripe, 4, 3 },
        TestCommand { CommandType::WaitForAgent, 1, 8 },
        TestCommand { CommandType::WaitForAgent, 2, 1 },
        TestCommand { CommandType::StartMceStripe, 4, 3 },
        TestCommand { CommandType::ProgramMceStripe, 4, 4 },
        TestCommand { CommandType::WaitForAgent, 1, 10 },
        TestCommand { CommandType::StartMceStripe, 4, 4 },
        TestCommand { CommandType::ProgramMceStripe, 4, 5 },
        TestCommand { CommandType::WaitForAgent, 1, 11 },
        TestCommand { CommandType::StartMceStripe, 4, 5 },
        TestCommand { CommandType::ProgramMceStripe, 4, 6 },
        TestCommand { CommandType::WaitForAgent, 1, 14 },
        TestCommand { CommandType::WaitForAgent, 2, 2 },
        TestCommand { CommandType::StartMceStripe, 4, 6 },
        TestCommand { CommandType::ProgramMceStripe, 4, 7 },
        TestCommand { CommandType::WaitForAgent, 1, 16 },
        TestCommand { CommandType::StartMceStripe, 4, 7 },
        TestCommand { CommandType::ProgramMceStripe, 4, 8 },
        TestCommand { CommandType::WaitForAgent, 1, 17 },
        TestCommand { CommandType::StartMceStripe, 4, 8 }
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, caps, debuggingContext);
    scheduler.Schedule();

    CompareCommandArrays(scheduler.GetDmaRdCommands(), expectedDmaRdCommands);
    CompareCommandArrays(scheduler.GetDmaWrCommands(), expectedDmaWrCommands);
    CompareCommandArrays(scheduler.GetMceCommands(), expectedMceCommands);
    CompareCommandArrays(scheduler.GetPleCommands(), expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/IfmStreamer/WriteDependencies/AfterFirstTile")
{
    HardwareCapabilities caps = GetEthosN78HwCapabilities();

    const uint32_t tileSize                  = 18;
    const uint32_t relativeAgentIdDependency = 3;
    const uint32_t numStripesTotal           = 18;
    IfmSDesc ifms                            = MakeIfmSDesc();
    ifms.fmData.tile.numSlots                = tileSize;

    WgtSDesc wgts      = MakeWgtSDesc();
    wgts.tile.numSlots = 2;

    std::vector<AgentDescAndDeps> cmdStream{
        // The first agent in the command stream is dummy, and it is there just
        // to make sure that we don't use agent ID 0. This help to validate
        // that the relative agent id field is properly used by the
        // scheduler function
        AgentDescAndDeps{ AgentDesc(0, MakeIfmSDesc()), {} },
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
            AgentDesc(9, MakeMceSDesc()),
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
            AgentDesc(1, MakeOfmSDesc()),
            {
                /* .readDependencies =*/{ { { 1, { 9, 1 }, { 9, 1 }, 0 } } },
                /* .writeDependencies =*/{},
            },
        },
    };

    const std::vector<TestCommand> expectedDmaWrCommands{
        //clang-format off
        TestCommand{ CommandType::WaitForAgent, 4, 8 }, TestCommand{ CommandType::StoreOfmStripe, 5, 0 }
        //clang-format on
    };

    const std::vector<TestCommand> expectedPleCommands{
        // clang-format off
        // clang-format on
    };
    const std::vector<TestCommand> expectedDmaRdCommands{
        // clang-format off
        TestCommand { CommandType::LoadIfmStripe, 1, 0 },
        TestCommand { CommandType::LoadIfmStripe, 1, 1 },
        TestCommand { CommandType::LoadIfmStripe, 1, 2 },
        TestCommand { CommandType::LoadWgtStripe, 2, 0 },
        TestCommand { CommandType::LoadIfmStripe, 1, 3 },
        TestCommand { CommandType::LoadIfmStripe, 1, 4 },
        TestCommand { CommandType::LoadIfmStripe, 1, 5 },
        TestCommand { CommandType::LoadIfmStripe, 1, 6 },
        TestCommand { CommandType::LoadIfmStripe, 1, 7 },
        TestCommand { CommandType::LoadIfmStripe, 1, 8 },
        TestCommand { CommandType::LoadWgtStripe, 2, 1 },
        TestCommand { CommandType::LoadIfmStripe, 1, 9 },
        TestCommand { CommandType::LoadIfmStripe, 1, 10 },
        TestCommand { CommandType::LoadIfmStripe, 1, 11 },
        TestCommand { CommandType::LoadIfmStripe, 1, 12 },
        TestCommand { CommandType::LoadIfmStripe, 1, 13 },
        TestCommand { CommandType::LoadIfmStripe, 1, 14 },
        TestCommand { CommandType::WaitForAgent, 4, 2 },
        TestCommand { CommandType::LoadWgtStripe, 2, 2 },
        TestCommand { CommandType::LoadIfmStripe, 1, 15 },
        TestCommand { CommandType::LoadIfmStripe, 1, 16 },
        TestCommand { CommandType::LoadIfmStripe, 1, 17 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedMceCommands{
        // clang-format off
        TestCommand { CommandType::ProgramMceStripe, 4, 0 },
        TestCommand { CommandType::WaitForAgent, 1, 2 },
        TestCommand { CommandType::WaitForAgent, 2, 0 },
        TestCommand { CommandType::StartMceStripe, 4, 0 },
        TestCommand { CommandType::ProgramMceStripe, 4, 1 },
        TestCommand { CommandType::WaitForAgent, 1, 4 },
        TestCommand { CommandType::StartMceStripe, 4, 1 },
        TestCommand { CommandType::ProgramMceStripe, 4, 2 },
        TestCommand { CommandType::WaitForAgent, 1, 5 },
        TestCommand { CommandType::StartMceStripe, 4, 2 },
        TestCommand { CommandType::ProgramMceStripe, 4, 3 },
        TestCommand { CommandType::WaitForAgent, 1, 8 },
        TestCommand { CommandType::WaitForAgent, 2, 1 },
        TestCommand { CommandType::StartMceStripe, 4, 3 },
        TestCommand { CommandType::ProgramMceStripe, 4, 4 },
        TestCommand { CommandType::WaitForAgent, 1, 10 },
        TestCommand { CommandType::StartMceStripe, 4, 4 },
        TestCommand { CommandType::ProgramMceStripe, 4, 5 },
        TestCommand { CommandType::WaitForAgent, 1, 11 },
        TestCommand { CommandType::StartMceStripe, 4, 5 },
        TestCommand { CommandType::ProgramMceStripe, 4, 6 },
        TestCommand { CommandType::WaitForAgent, 1, 14 },
        TestCommand { CommandType::WaitForAgent, 2, 2 },
        TestCommand { CommandType::StartMceStripe, 4, 6 },
        TestCommand { CommandType::ProgramMceStripe, 4, 7 },
        TestCommand { CommandType::WaitForAgent, 1, 16 },
        TestCommand { CommandType::StartMceStripe, 4, 7 },
        TestCommand { CommandType::ProgramMceStripe, 4, 8 },
        TestCommand { CommandType::WaitForAgent, 1, 17 },
        TestCommand { CommandType::StartMceStripe, 4, 8 }
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, caps, debuggingContext);
    scheduler.Schedule();

    CompareCommandArrays(scheduler.GetDmaRdCommands(), expectedDmaRdCommands);
    CompareCommandArrays(scheduler.GetDmaWrCommands(), expectedDmaWrCommands);
    CompareCommandArrays(scheduler.GetMceCommands(), expectedMceCommands);
    CompareCommandArrays(scheduler.GetPleCommands(), expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/IfmStreamer/WithReadAndWriteDependency/FirstTile")
{
    HardwareCapabilities caps = GetEthosN78HwCapabilities();

    const uint32_t numStripesTotal = 6;
    const uint32_t tileSize        = 4;
    IfmSDesc ifms                  = MakeIfmSDesc();
    ifms.fmData.tile.numSlots      = tileSize;

    WgtSDesc wgts      = MakeWgtSDesc();
    wgts.tile.numSlots = 2;

    PleSDesc ples         = MakePleSDesc();
    ples.ofmTile.numSlots = 2;

    OfmSDesc ofms             = MakeOfmSDesc();
    ofms.fmData.tile.numSlots = 2;

    std::vector<AgentDescAndDeps> cmdStream{
        AgentDescAndDeps{
            AgentDesc(3, MakeMceSDesc()),
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
            AgentDesc(3, MakeMceSDesc()),
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
            AgentDesc(1, MakeOfmSDesc()),
            {
                /* .readDependencies =*/{ { { 1, { 3, 1 }, { 3, 1 }, 0 } } },
                /* .writeDependencies =*/{},
            },
        },
    };
    const std::vector<TestCommand> expectedDmaRdCommands{
        // clang-format off
        TestCommand { CommandType::WaitForAgent, 0, 2 },
        TestCommand { CommandType::WaitForAgent, 2, 0 },
        TestCommand { CommandType::LoadIfmStripe, 3, 0 },
        TestCommand { CommandType::LoadIfmStripe, 3, 1 },
        TestCommand { CommandType::WaitForAgent, 2, 1 },
        TestCommand { CommandType::LoadIfmStripe, 3, 2 },
        TestCommand { CommandType::LoadWgtStripe, 4, 0 },
        TestCommand { CommandType::LoadIfmStripe, 3, 3 },
        TestCommand { CommandType::WaitForAgent, 6, 0 },
        TestCommand { CommandType::WaitForAgent, 2, 2 },
        TestCommand { CommandType::LoadIfmStripe, 3, 4 },
        TestCommand { CommandType::WaitForAgent, 6, 1 },
        TestCommand { CommandType::LoadIfmStripe, 3, 5 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedDmaWrCommands{
        //clang-format off
        TestCommand{ CommandType::WaitForAgent, 1, 0 },
        TestCommand{ CommandType::StoreOfmStripe, 2, 0 },
        TestCommand{ CommandType::WaitForAgent, 1, 1 },
        TestCommand{ CommandType::StoreOfmStripe, 2, 1 },
        TestCommand{ CommandType::WaitForAgent, 1, 2 },
        TestCommand{ CommandType::StoreOfmStripe, 2, 2 },
        TestCommand{ CommandType::WaitForAgent, 6, 2 },
        TestCommand{ CommandType::StoreOfmStripe, 7, 0 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedMceCommands{
        // clang-format off
        TestCommand { CommandType::ProgramMceStripe, 0, 0 },
        TestCommand { CommandType::StartMceStripe, 0, 0 },
        TestCommand { CommandType::ProgramMceStripe, 0, 1 },
        TestCommand { CommandType::StartMceStripe, 0, 1 },
        TestCommand { CommandType::ProgramMceStripe, 0, 2 },
        TestCommand { CommandType::StartMceStripe, 0, 2 },
        TestCommand { CommandType::ProgramMceStripe, 6, 0 },
        TestCommand { CommandType::WaitForAgent, 3, 2 },
        TestCommand { CommandType::WaitForAgent, 4, 0 },
        TestCommand { CommandType::StartMceStripe, 6, 0 },
        TestCommand { CommandType::ProgramMceStripe, 6, 1 },
        TestCommand { CommandType::WaitForAgent, 3, 4 },
        TestCommand { CommandType::StartMceStripe, 6, 1 },
        TestCommand { CommandType::ProgramMceStripe, 6, 2 },
        TestCommand { CommandType::WaitForAgent, 3, 5 },
        TestCommand { CommandType::StartMceStripe, 6, 2 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedPleCommands{
        // clang-format off
        TestCommand { CommandType::StartPleStripe, 1, 0 },
        TestCommand { CommandType::StartPleStripe, 1, 1 },
        TestCommand { CommandType::StartPleStripe, 1, 2 }
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, caps, debuggingContext);
    scheduler.Schedule();

    CompareCommandArrays(scheduler.GetDmaRdCommands(), expectedDmaRdCommands);
    CompareCommandArrays(scheduler.GetDmaWrCommands(), expectedDmaWrCommands);
    CompareCommandArrays(scheduler.GetMceCommands(), expectedMceCommands);
    CompareCommandArrays(scheduler.GetPleCommands(), expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/WithReadAndWriteDependency/AfterFirstTile")
{
    HardwareCapabilities caps = GetEthosN78HwCapabilities();

    const uint32_t tileSize        = 4;
    const uint32_t numStripesTotal = 6;
    IfmSDesc ifms                  = MakeIfmSDesc();
    ifms.fmData.tile.numSlots      = tileSize;

    WgtSDesc wgts      = MakeWgtSDesc();
    wgts.tile.numSlots = 2;

    PleSDesc ples         = MakePleSDesc();
    ples.ofmTile.numSlots = 2;

    OfmSDesc ofms             = MakeOfmSDesc();
    ofms.fmData.tile.numSlots = 2;

    std::vector<AgentDescAndDeps> cmdStream{
        AgentDescAndDeps{
            AgentDesc(3, MakeMceSDesc()),
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
            AgentDesc(3, MakeMceSDesc()),
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
            AgentDesc(1, MakeOfmSDesc()),
            {
                /* .readDependencies =*/{ { { 1, { 3, 1 }, { 3, 1 }, 0 } } },
                /* .writeDependencies =*/{},
            },
        },

    };
    const std::vector<TestCommand> expectedDmaRdCommands{
        // clang-format off
        TestCommand { CommandType::WaitForAgent, 0, 2 },
        TestCommand { CommandType::WaitForAgent, 2, 0 },
        TestCommand { CommandType::LoadIfmStripe, 3, 0 },
        TestCommand { CommandType::LoadIfmStripe, 3, 1 },
        TestCommand { CommandType::WaitForAgent, 2, 1 },
        TestCommand { CommandType::LoadIfmStripe, 3, 2 },
        TestCommand { CommandType::LoadWgtStripe, 4, 0 },
        TestCommand { CommandType::LoadIfmStripe, 3, 3 },
        TestCommand { CommandType::WaitForAgent, 6, 0 },
        TestCommand { CommandType::WaitForAgent, 2, 2 },
        TestCommand { CommandType::LoadIfmStripe, 3, 4 },
        TestCommand { CommandType::WaitForAgent, 6, 1 },
        TestCommand { CommandType::LoadIfmStripe, 3, 5 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedDmaWrCommands{
        //clang-format off
        TestCommand{ CommandType::WaitForAgent, 1, 0 },
        TestCommand{ CommandType::StoreOfmStripe, 2, 0 },
        TestCommand{ CommandType::WaitForAgent, 1, 1 },
        TestCommand{ CommandType::StoreOfmStripe, 2, 1 },
        TestCommand{ CommandType::WaitForAgent, 1, 2 },
        TestCommand{ CommandType::StoreOfmStripe, 2, 2 },
        TestCommand{ CommandType::WaitForAgent, 6, 2 },
        TestCommand{ CommandType::StoreOfmStripe, 7, 0 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedMceCommands{
        // clang-format off
        TestCommand { CommandType::ProgramMceStripe, 0, 0 },
        TestCommand { CommandType::StartMceStripe, 0, 0 },
        TestCommand { CommandType::ProgramMceStripe, 0, 1 },
        TestCommand { CommandType::StartMceStripe, 0, 1 },
        TestCommand { CommandType::ProgramMceStripe, 0, 2 },
        TestCommand { CommandType::StartMceStripe, 0, 2 },
        TestCommand { CommandType::ProgramMceStripe, 6, 0 },
        TestCommand { CommandType::WaitForAgent, 3, 2 },
        TestCommand { CommandType::WaitForAgent, 4, 0 },
        TestCommand { CommandType::StartMceStripe, 6, 0 },
        TestCommand { CommandType::ProgramMceStripe, 6, 1 },
        TestCommand { CommandType::WaitForAgent, 3, 4 },
        TestCommand { CommandType::StartMceStripe, 6, 1 },
        TestCommand { CommandType::ProgramMceStripe, 6, 2 },
        TestCommand { CommandType::WaitForAgent, 3, 5 },
        TestCommand { CommandType::StartMceStripe, 6, 2 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedPleCommands{
        // clang-format off
        TestCommand { CommandType::StartPleStripe, 1, 0 },
        TestCommand { CommandType::StartPleStripe, 1, 1 },
        TestCommand { CommandType::StartPleStripe, 1, 2 }
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, caps, debuggingContext);
    scheduler.Schedule();

    CompareCommandArrays(scheduler.GetDmaRdCommands(), expectedDmaRdCommands);
    CompareCommandArrays(scheduler.GetDmaWrCommands(), expectedDmaWrCommands);
    CompareCommandArrays(scheduler.GetMceCommands(), expectedMceCommands);
    CompareCommandArrays(scheduler.GetPleCommands(), expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/WgtStreamer/AllFitInOneTile/WithWriteDependency")
{
    HardwareCapabilities caps = GetEthosN78HwCapabilities();

    IfmSDesc ifms             = MakeIfmSDesc();
    ifms.fmData.tile.numSlots = 4;

    const uint32_t numStripesTotal = 3;
    // When there is a write dependency, the tileSize needs to be set with the right value. i.e. 3
    const uint16_t tileSize = 3;
    WgtSDesc wgtS           = MakeWgtSDesc();
    wgtS.tile.numSlots      = tileSize;

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
            AgentDesc(9, MakeMceSDesc()),
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
            AgentDesc(1, MakeOfmSDesc()),
            {
                /* .readDependencies =*/{ { { 1, { 9, 1 }, { 9, 1 }, 0 } } },
                /* .writeDependencies =*/{},
            },
        },

    };

    const std::vector<TestCommand> expectedDmaWrCommands{
        //clang-format off
        TestCommand{ CommandType::WaitForAgent, 3, 8 }, TestCommand{ CommandType::StoreOfmStripe, 4, 0 }
        //clang-format on
    };

    const std::vector<TestCommand> expectedPleCommands{
        // clang-format off
        // clang-format on
    };
    const std::vector<TestCommand> expectedDmaRdCommands{
        // clang-format off
        TestCommand { CommandType::LoadIfmStripe, 0, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 1 },
        TestCommand { CommandType::LoadIfmStripe, 0, 2 },
        TestCommand { CommandType::LoadWgtStripe, 1, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 3 },
        TestCommand { CommandType::WaitForAgent, 3, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 4 },
        TestCommand { CommandType::WaitForAgent, 3, 1 },
        TestCommand { CommandType::LoadIfmStripe, 0, 5 },
        TestCommand { CommandType::LoadIfmStripe, 0, 6 },
        TestCommand { CommandType::WaitForAgent, 3, 2 },
        TestCommand { CommandType::LoadIfmStripe, 0, 7 },
        TestCommand { CommandType::LoadIfmStripe, 0, 8 },
        TestCommand { CommandType::LoadWgtStripe, 1, 1 },
        TestCommand { CommandType::LoadIfmStripe, 0, 9 },
        TestCommand { CommandType::WaitForAgent, 3, 3 },
        TestCommand { CommandType::LoadIfmStripe, 0, 10 },
        TestCommand { CommandType::WaitForAgent, 3, 4 },
        TestCommand { CommandType::LoadIfmStripe, 0, 11 },
        TestCommand { CommandType::LoadIfmStripe, 0, 12 },
        TestCommand { CommandType::WaitForAgent, 3, 5 },
        TestCommand { CommandType::LoadIfmStripe, 0, 13 },
        TestCommand { CommandType::LoadIfmStripe, 0, 14 },
        TestCommand { CommandType::LoadWgtStripe, 1, 2 },
        TestCommand { CommandType::LoadIfmStripe, 0, 15 },
        TestCommand { CommandType::WaitForAgent, 3, 6 },
        TestCommand { CommandType::LoadIfmStripe, 0, 16 },
        TestCommand { CommandType::WaitForAgent, 3, 7 },
        TestCommand { CommandType::LoadIfmStripe, 0, 17 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedMceCommands{
        // clang-format off
        TestCommand { CommandType::ProgramMceStripe, 3, 0 },
        TestCommand { CommandType::WaitForAgent, 0, 2 },
        TestCommand { CommandType::WaitForAgent, 1, 0 },
        TestCommand { CommandType::StartMceStripe, 3, 0 },
        TestCommand { CommandType::ProgramMceStripe, 3, 1 },
        TestCommand { CommandType::WaitForAgent, 0, 4 },
        TestCommand { CommandType::StartMceStripe, 3, 1 },
        TestCommand { CommandType::ProgramMceStripe, 3, 2 },
        TestCommand { CommandType::WaitForAgent, 0, 5 },
        TestCommand { CommandType::StartMceStripe, 3, 2 },
        TestCommand { CommandType::ProgramMceStripe, 3, 3 },
        TestCommand { CommandType::WaitForAgent, 0, 8 },
        TestCommand { CommandType::WaitForAgent, 1, 1 },
        TestCommand { CommandType::StartMceStripe, 3, 3 },
        TestCommand { CommandType::ProgramMceStripe, 3, 4 },
        TestCommand { CommandType::WaitForAgent, 0, 10 },
        TestCommand { CommandType::StartMceStripe, 3, 4 },
        TestCommand { CommandType::ProgramMceStripe, 3, 5 },
        TestCommand { CommandType::WaitForAgent, 0, 11 },
        TestCommand { CommandType::StartMceStripe, 3, 5 },
        TestCommand { CommandType::ProgramMceStripe, 3, 6 },
        TestCommand { CommandType::WaitForAgent, 0, 14 },
        TestCommand { CommandType::WaitForAgent, 1, 2 },
        TestCommand { CommandType::StartMceStripe, 3, 6 },
        TestCommand { CommandType::ProgramMceStripe, 3, 7 },
        TestCommand { CommandType::WaitForAgent, 0, 16 },
        TestCommand { CommandType::StartMceStripe, 3, 7 },
        TestCommand { CommandType::ProgramMceStripe, 3, 8 },
        TestCommand { CommandType::WaitForAgent, 0, 17 },
        TestCommand { CommandType::StartMceStripe, 3, 8 }
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, caps, debuggingContext);
    scheduler.Schedule();

    CompareCommandArrays(scheduler.GetDmaRdCommands(), expectedDmaRdCommands);
    CompareCommandArrays(scheduler.GetDmaWrCommands(), expectedDmaWrCommands);
    CompareCommandArrays(scheduler.GetMceCommands(), expectedMceCommands);
    CompareCommandArrays(scheduler.GetPleCommands(), expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/WgtStreamer/AllFitInOneTile/NoWriteDependency")
{
    HardwareCapabilities caps = GetEthosN78HwCapabilities();

    IfmSDesc ifms             = MakeIfmSDesc();
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
            AgentDesc(numStripesTotal, MakeWgtSDesc()),
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
            AgentDesc(9, MakeMceSDesc()),
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
            AgentDesc(1, MakeOfmSDesc()),
            {
                /* .readDependencies =*/{ { { 1, { 9, 1 }, { 9, 1 }, 0 } } },
                /* .writeDependencies =*/{},
            },
        },

    };

    const std::vector<TestCommand> expectedDmaWrCommands{
        //clang-format off
        TestCommand{ CommandType::WaitForAgent, 3, 8 }, TestCommand{ CommandType::StoreOfmStripe, 4, 0 }
        //clang-format on
    };

    const std::vector<TestCommand> expectedPleCommands{
        // clang-format off
        // clang-format on
    };

    const std::vector<TestCommand> expectedDmaRdCommands{
        // clang-format off
        TestCommand { CommandType::LoadIfmStripe, 0, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 1 },
        TestCommand { CommandType::LoadIfmStripe, 0, 2 },
        TestCommand { CommandType::LoadWgtStripe, 1, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 3 },
        TestCommand { CommandType::WaitForAgent, 3, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 4 },
        TestCommand { CommandType::WaitForAgent, 3, 1 },
        TestCommand { CommandType::LoadIfmStripe, 0, 5 },
        TestCommand { CommandType::LoadIfmStripe, 0, 6 },
        TestCommand { CommandType::WaitForAgent, 3, 2 },
        TestCommand { CommandType::LoadIfmStripe, 0, 7 },
        TestCommand { CommandType::LoadIfmStripe, 0, 8 },
        TestCommand { CommandType::LoadWgtStripe, 1, 1 },
        TestCommand { CommandType::LoadIfmStripe, 0, 9 },
        TestCommand { CommandType::WaitForAgent, 3, 3 },
        TestCommand { CommandType::LoadIfmStripe, 0, 10 },
        TestCommand { CommandType::WaitForAgent, 3, 4 },
        TestCommand { CommandType::LoadIfmStripe, 0, 11 },
        TestCommand { CommandType::LoadIfmStripe, 0, 12 },
        TestCommand { CommandType::WaitForAgent, 3, 5 },
        TestCommand { CommandType::LoadIfmStripe, 0, 13 },
        TestCommand { CommandType::LoadIfmStripe, 0, 14 },
        TestCommand { CommandType::LoadWgtStripe, 1, 2 },
        TestCommand { CommandType::LoadIfmStripe, 0, 15 },
        TestCommand { CommandType::WaitForAgent, 3, 6 },
        TestCommand { CommandType::LoadIfmStripe, 0, 16 },
        TestCommand { CommandType::WaitForAgent, 3, 7 },
        TestCommand { CommandType::LoadIfmStripe, 0, 17 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedMceCommands{
        // clang-format off
        TestCommand { CommandType::ProgramMceStripe, 3, 0 },
        TestCommand { CommandType::WaitForAgent, 0, 2 },
        TestCommand { CommandType::WaitForAgent, 1, 0 },
        TestCommand { CommandType::StartMceStripe, 3, 0 },
        TestCommand { CommandType::ProgramMceStripe, 3, 1 },
        TestCommand { CommandType::WaitForAgent, 0, 4 },
        TestCommand { CommandType::StartMceStripe, 3, 1 },
        TestCommand { CommandType::ProgramMceStripe, 3, 2 },
        TestCommand { CommandType::WaitForAgent, 0, 5 },
        TestCommand { CommandType::StartMceStripe, 3, 2 },
        TestCommand { CommandType::ProgramMceStripe, 3, 3 },
        TestCommand { CommandType::WaitForAgent, 0, 8 },
        TestCommand { CommandType::WaitForAgent, 1, 1 },
        TestCommand { CommandType::StartMceStripe, 3, 3 },
        TestCommand { CommandType::ProgramMceStripe, 3, 4 },
        TestCommand { CommandType::WaitForAgent, 0, 10 },
        TestCommand { CommandType::StartMceStripe, 3, 4 },
        TestCommand { CommandType::ProgramMceStripe, 3, 5 },
        TestCommand { CommandType::WaitForAgent, 0, 11 },
        TestCommand { CommandType::StartMceStripe, 3, 5 },
        TestCommand { CommandType::ProgramMceStripe, 3, 6 },
        TestCommand { CommandType::WaitForAgent, 0, 14 },
        TestCommand { CommandType::WaitForAgent, 1, 2 },
        TestCommand { CommandType::StartMceStripe, 3, 6 },
        TestCommand { CommandType::ProgramMceStripe, 3, 7 },
        TestCommand { CommandType::WaitForAgent, 0, 16 },
        TestCommand { CommandType::StartMceStripe, 3, 7 },
        TestCommand { CommandType::ProgramMceStripe, 3, 8 },
        TestCommand { CommandType::WaitForAgent, 0, 17 },
        TestCommand { CommandType::StartMceStripe, 3, 8 }
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, caps, debuggingContext);
    scheduler.Schedule();

    CompareCommandArrays(scheduler.GetDmaRdCommands(), expectedDmaRdCommands);
    CompareCommandArrays(scheduler.GetDmaWrCommands(), expectedDmaWrCommands);
    CompareCommandArrays(scheduler.GetMceCommands(), expectedMceCommands);
    CompareCommandArrays(scheduler.GetPleCommands(), expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/WgtStreamer/WithWriteDependency/TileSize=1")
{
    HardwareCapabilities caps = GetEthosN78HwCapabilities();

    IfmSDesc ifms             = MakeIfmSDesc();
    ifms.fmData.tile.numSlots = 4;

    const uint32_t numStripesTotal = 3;
    const uint32_t tileSize        = 1;
    WgtSDesc wgtS                  = MakeWgtSDesc();
    wgtS.tile.numSlots             = tileSize;
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
            AgentDesc(9, MakeMceSDesc()),
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
            AgentDesc(1, MakeOfmSDesc()),
            {
                /* .readDependencies =*/{ { { 1, { 9, 1 }, { 9, 1 }, 0 } } },
                /* .writeDependencies =*/{},
            },
        },

    };

    const std::vector<TestCommand> expectedDmaWrCommands{
        //clang-format off
        TestCommand{ CommandType::WaitForAgent, 3, 8 }, TestCommand{ CommandType::StoreOfmStripe, 4, 0 }
        //clang-format on
    };

    const std::vector<TestCommand> expectedPleCommands{
        // clang-format off
        // clang-format on
    };

    const std::vector<TestCommand> expectedDmaRdCommands{
        // clang-format off
        TestCommand { CommandType::LoadIfmStripe, 0, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 1 },
        TestCommand { CommandType::LoadIfmStripe, 0, 2 },
        TestCommand { CommandType::LoadWgtStripe, 1, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 3 },
        TestCommand { CommandType::WaitForAgent, 3, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 4 },
        TestCommand { CommandType::WaitForAgent, 3, 1 },
        TestCommand { CommandType::LoadIfmStripe, 0, 5 },
        TestCommand { CommandType::LoadIfmStripe, 0, 6 },
        TestCommand { CommandType::WaitForAgent, 3, 2 },
        TestCommand { CommandType::LoadIfmStripe, 0, 7 },
        TestCommand { CommandType::LoadIfmStripe, 0, 8 },
        TestCommand { CommandType::LoadWgtStripe, 1, 1 },
        TestCommand { CommandType::LoadIfmStripe, 0, 9 },
        TestCommand { CommandType::WaitForAgent, 3, 3 },
        TestCommand { CommandType::LoadIfmStripe, 0, 10 },
        TestCommand { CommandType::WaitForAgent, 3, 4 },
        TestCommand { CommandType::LoadIfmStripe, 0, 11 },
        TestCommand { CommandType::LoadIfmStripe, 0, 12 },
        TestCommand { CommandType::WaitForAgent, 3, 5 },
        TestCommand { CommandType::LoadIfmStripe, 0, 13 },
        TestCommand { CommandType::LoadIfmStripe, 0, 14 },
        TestCommand { CommandType::LoadWgtStripe, 1, 2 },
        TestCommand { CommandType::LoadIfmStripe, 0, 15 },
        TestCommand { CommandType::WaitForAgent, 3, 6 },
        TestCommand { CommandType::LoadIfmStripe, 0, 16 },
        TestCommand { CommandType::WaitForAgent, 3, 7 },
        TestCommand { CommandType::LoadIfmStripe, 0, 17 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedMceCommands{
        // clang-format off
        TestCommand { CommandType::ProgramMceStripe, 3, 0 },
        TestCommand { CommandType::WaitForAgent, 0, 2 },
        TestCommand { CommandType::WaitForAgent, 1, 0 },
        TestCommand { CommandType::StartMceStripe, 3, 0 },
        TestCommand { CommandType::ProgramMceStripe, 3, 1 },
        TestCommand { CommandType::WaitForAgent, 0, 4 },
        TestCommand { CommandType::StartMceStripe, 3, 1 },
        TestCommand { CommandType::ProgramMceStripe, 3, 2 },
        TestCommand { CommandType::WaitForAgent, 0, 5 },
        TestCommand { CommandType::StartMceStripe, 3, 2 },
        TestCommand { CommandType::ProgramMceStripe, 3, 3 },
        TestCommand { CommandType::WaitForAgent, 0, 8 },
        TestCommand { CommandType::WaitForAgent, 1, 1 },
        TestCommand { CommandType::StartMceStripe, 3, 3 },
        TestCommand { CommandType::ProgramMceStripe, 3, 4 },
        TestCommand { CommandType::WaitForAgent, 0, 10 },
        TestCommand { CommandType::StartMceStripe, 3, 4 },
        TestCommand { CommandType::ProgramMceStripe, 3, 5 },
        TestCommand { CommandType::WaitForAgent, 0, 11 },
        TestCommand { CommandType::StartMceStripe, 3, 5 },
        TestCommand { CommandType::ProgramMceStripe, 3, 6 },
        TestCommand { CommandType::WaitForAgent, 0, 14 },
        TestCommand { CommandType::WaitForAgent, 1, 2 },
        TestCommand { CommandType::StartMceStripe, 3, 6 },
        TestCommand { CommandType::ProgramMceStripe, 3, 7 },
        TestCommand { CommandType::WaitForAgent, 0, 16 },
        TestCommand { CommandType::StartMceStripe, 3, 7 },
        TestCommand { CommandType::ProgramMceStripe, 3, 8 },
        TestCommand { CommandType::WaitForAgent, 0, 17 },
        TestCommand { CommandType::StartMceStripe, 3, 8 }
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, caps, debuggingContext);
    scheduler.Schedule();

    CompareCommandArrays(scheduler.GetDmaRdCommands(), expectedDmaRdCommands);
    CompareCommandArrays(scheduler.GetDmaWrCommands(), expectedDmaWrCommands);
    CompareCommandArrays(scheduler.GetMceCommands(), expectedMceCommands);
    CompareCommandArrays(scheduler.GetPleCommands(), expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/WgtStreamer/WithReadAndWriteDependencies/TileSize=2")
{
    HardwareCapabilities caps = GetEthosN78HwCapabilities();

    const uint32_t numStripesTotal = 3;
    const uint32_t tileSize        = 2;
    WgtSDesc wgtS                  = MakeWgtSDesc();
    wgtS.tile.numSlots             = tileSize;

    PleSDesc ples         = MakePleSDesc();
    ples.ofmTile.numSlots = 2;

    std::vector<AgentDescAndDeps> cmdStream{
        AgentDescAndDeps{
            AgentDesc(9, MakeMceSDesc()),
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
            AgentDesc(12, MakeMceSDesc()),
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
            AgentDesc(1, MakeOfmSDesc()),
            {
                /* .readDependencies =*/{ { { 1, { 12, 1 }, { 12, 1 }, 0 } } },
                /* .writeDependencies =*/{},
            },
        },

    };

    const std::vector<TestCommand> expectedDmaWrCommands{
        //clang-format off
        TestCommand{ CommandType::WaitForAgent, 4, 11 }, TestCommand{ CommandType::StoreOfmStripe, 5, 0 }
        //clang-format on
    };

    const std::vector<TestCommand> expectedDmaRdCommands{
        // clang-format off
        TestCommand { CommandType::WaitForAgent, 0, 8 },
        TestCommand { CommandType::LoadWgtStripe, 2, 0 },
        TestCommand { CommandType::LoadWgtStripe, 2, 1 },
        TestCommand { CommandType::WaitForAgent, 4, 3 },
        TestCommand { CommandType::LoadWgtStripe, 2, 2 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedMceCommands{
        // clang-format off
        TestCommand { CommandType::ProgramMceStripe, 0, 0 },
        TestCommand { CommandType::StartMceStripe, 0, 0 },
        TestCommand { CommandType::ProgramMceStripe, 0, 1 },
        TestCommand { CommandType::StartMceStripe, 0, 1 },
        TestCommand { CommandType::ProgramMceStripe, 0, 2 },
        TestCommand { CommandType::StartMceStripe, 0, 2 },
        TestCommand { CommandType::ProgramMceStripe, 0, 3 },
        TestCommand { CommandType::StartMceStripe, 0, 3 },
        TestCommand { CommandType::ProgramMceStripe, 0, 4 },
        TestCommand { CommandType::StartMceStripe, 0, 4 },
        TestCommand { CommandType::ProgramMceStripe, 0, 5 },
        TestCommand { CommandType::StartMceStripe, 0, 5 },
        TestCommand { CommandType::ProgramMceStripe, 0, 6 },
        TestCommand { CommandType::StartMceStripe, 0, 6 },
        TestCommand { CommandType::ProgramMceStripe, 0, 7 },
        TestCommand { CommandType::StartMceStripe, 0, 7 },
        TestCommand { CommandType::ProgramMceStripe, 0, 8 },
        TestCommand { CommandType::StartMceStripe, 0, 8 },
        TestCommand { CommandType::ProgramMceStripe, 4, 0 },
        TestCommand { CommandType::WaitForAgent, 1, 0 },
        TestCommand { CommandType::WaitForAgent, 2, 0 },
        TestCommand { CommandType::StartMceStripe, 4, 0 },
        TestCommand { CommandType::ProgramMceStripe, 4, 1 },
        TestCommand { CommandType::StartMceStripe, 4, 1 },
        TestCommand { CommandType::ProgramMceStripe, 4, 2 },
        TestCommand { CommandType::StartMceStripe, 4, 2 },
        TestCommand { CommandType::ProgramMceStripe, 4, 3 },
        TestCommand { CommandType::StartMceStripe, 4, 3 },
        TestCommand { CommandType::ProgramMceStripe, 4, 4 },
        TestCommand { CommandType::WaitForAgent, 2, 1 },
        TestCommand { CommandType::StartMceStripe, 4, 4 },
        TestCommand { CommandType::ProgramMceStripe, 4, 5 },
        TestCommand { CommandType::StartMceStripe, 4, 5 },
        TestCommand { CommandType::ProgramMceStripe, 4, 6 },
        TestCommand { CommandType::StartMceStripe, 4, 6 },
        TestCommand { CommandType::ProgramMceStripe, 4, 7 },
        TestCommand { CommandType::StartMceStripe, 4, 7 },
        TestCommand { CommandType::ProgramMceStripe, 4, 8 },
        TestCommand { CommandType::WaitForAgent, 2, 2 },
        TestCommand { CommandType::StartMceStripe, 4, 8 },
        TestCommand { CommandType::ProgramMceStripe, 4, 9 },
        TestCommand { CommandType::StartMceStripe, 4, 9 },
        TestCommand { CommandType::ProgramMceStripe, 4, 10 },
        TestCommand { CommandType::StartMceStripe, 4, 10 },
        TestCommand { CommandType::ProgramMceStripe, 4, 11 },
        TestCommand { CommandType::StartMceStripe, 4, 11 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedPleCommands{
        // clang-format off
        TestCommand { CommandType::StartPleStripe, 1, 0 }
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, caps, debuggingContext);
    scheduler.Schedule();

    CompareCommandArrays(scheduler.GetDmaRdCommands(), expectedDmaRdCommands);
    CompareCommandArrays(scheduler.GetDmaWrCommands(), expectedDmaWrCommands);
    CompareCommandArrays(scheduler.GetMceCommands(), expectedMceCommands);
    CompareCommandArrays(scheduler.GetPleCommands(), expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/MceSchedulerStripe")
{
    HardwareCapabilities caps = GetEthosN78HwCapabilities();

    IfmSDesc ifms             = MakeIfmSDesc();
    ifms.fmData.tile.numSlots = 4;

    WgtSDesc wgts      = MakeWgtSDesc();
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
            AgentDesc(numStripesTotal, MakeMceSDesc()),
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
            AgentDesc(1, MakeOfmSDesc()),
            {
                /* .readDependencies =*/{ { { 1, { numStripesTotal, 1 }, { numStripesTotal, 1 }, 0 } } },
                /* .writeDependencies =*/{},
            },
        },

    };

    const std::vector<TestCommand> expectedDmaWrCommands{
        //clang-format off
        TestCommand{ CommandType::WaitForAgent, 3, 8 }, TestCommand{ CommandType::StoreOfmStripe, 4, 0 }
        //clang-format on
    };

    const std::vector<TestCommand> expectedPleCommands{
        // clang-format off
        // clang-format on
    };
    const std::vector<TestCommand> expectedDmaRdCommands{
        // clang-format off
        TestCommand { CommandType::LoadIfmStripe, 0, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 1 },
        TestCommand { CommandType::LoadIfmStripe, 0, 2 },
        TestCommand { CommandType::LoadWgtStripe, 1, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 3 },
        TestCommand { CommandType::WaitForAgent, 3, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 4 },
        TestCommand { CommandType::WaitForAgent, 3, 1 },
        TestCommand { CommandType::LoadIfmStripe, 0, 5 },
        TestCommand { CommandType::LoadIfmStripe, 0, 6 },
        TestCommand { CommandType::WaitForAgent, 3, 2 },
        TestCommand { CommandType::LoadIfmStripe, 0, 7 },
        TestCommand { CommandType::LoadIfmStripe, 0, 8 },
        TestCommand { CommandType::LoadWgtStripe, 1, 1 },
        TestCommand { CommandType::LoadIfmStripe, 0, 9 },
        TestCommand { CommandType::WaitForAgent, 3, 3 },
        TestCommand { CommandType::LoadIfmStripe, 0, 10 },
        TestCommand { CommandType::WaitForAgent, 3, 4 },
        TestCommand { CommandType::LoadIfmStripe, 0, 11 },
        TestCommand { CommandType::LoadIfmStripe, 0, 12 },
        TestCommand { CommandType::WaitForAgent, 3, 5 },
        TestCommand { CommandType::LoadIfmStripe, 0, 13 },
        TestCommand { CommandType::LoadIfmStripe, 0, 14 },
        TestCommand { CommandType::LoadWgtStripe, 1, 2 },
        TestCommand { CommandType::LoadIfmStripe, 0, 15 },
        TestCommand { CommandType::WaitForAgent, 3, 6 },
        TestCommand { CommandType::LoadIfmStripe, 0, 16 },
        TestCommand { CommandType::WaitForAgent, 3, 7 },
        TestCommand { CommandType::LoadIfmStripe, 0, 17 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedMceCommands{
        // clang-format off
        TestCommand { CommandType::ProgramMceStripe, 3, 0 },
        TestCommand { CommandType::WaitForAgent, 0, 2 },
        TestCommand { CommandType::WaitForAgent, 1, 0 },
        TestCommand { CommandType::StartMceStripe, 3, 0 },
        TestCommand { CommandType::ProgramMceStripe, 3, 1 },
        TestCommand { CommandType::WaitForAgent, 0, 4 },
        TestCommand { CommandType::StartMceStripe, 3, 1 },
        TestCommand { CommandType::ProgramMceStripe, 3, 2 },
        TestCommand { CommandType::WaitForAgent, 0, 5 },
        TestCommand { CommandType::StartMceStripe, 3, 2 },
        TestCommand { CommandType::ProgramMceStripe, 3, 3 },
        TestCommand { CommandType::WaitForAgent, 0, 8 },
        TestCommand { CommandType::WaitForAgent, 1, 1 },
        TestCommand { CommandType::StartMceStripe, 3, 3 },
        TestCommand { CommandType::ProgramMceStripe, 3, 4 },
        TestCommand { CommandType::WaitForAgent, 0, 10 },
        TestCommand { CommandType::StartMceStripe, 3, 4 },
        TestCommand { CommandType::ProgramMceStripe, 3, 5 },
        TestCommand { CommandType::WaitForAgent, 0, 11 },
        TestCommand { CommandType::StartMceStripe, 3, 5 },
        TestCommand { CommandType::ProgramMceStripe, 3, 6 },
        TestCommand { CommandType::WaitForAgent, 0, 14 },
        TestCommand { CommandType::WaitForAgent, 1, 2 },
        TestCommand { CommandType::StartMceStripe, 3, 6 },
        TestCommand { CommandType::ProgramMceStripe, 3, 7 },
        TestCommand { CommandType::WaitForAgent, 0, 16 },
        TestCommand { CommandType::StartMceStripe, 3, 7 },
        TestCommand { CommandType::ProgramMceStripe, 3, 8 },
        TestCommand { CommandType::WaitForAgent, 0, 17 },
        TestCommand { CommandType::StartMceStripe, 3, 8 }
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, caps, debuggingContext);
    scheduler.Schedule();

    CompareCommandArrays(scheduler.GetDmaRdCommands(), expectedDmaRdCommands);
    CompareCommandArrays(scheduler.GetDmaWrCommands(), expectedDmaWrCommands);
    CompareCommandArrays(scheduler.GetMceCommands(), expectedMceCommands);
    CompareCommandArrays(scheduler.GetPleCommands(), expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/PleLoaderStripe/NoDependencies")
{
    HardwareCapabilities caps = GetEthosN78HwCapabilities();

    IfmSDesc ifms             = MakeIfmSDesc();
    ifms.fmData.tile.numSlots = 4;

    WgtSDesc wgts      = MakeWgtSDesc();
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
                                                 AgentDesc(9, MakeMceSDesc()),

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
                                                 AgentDesc(1, MakeOfmSDesc()),
                                                 {
                                                     /* .readDependencies =*/{ { { 1, { 9, 1 }, { 9, 1 }, 0 } } },
                                                     /* .writeDependencies =*/{},
                                                 },
                                             } };

    const std::vector<TestCommand> expectedDmaWrCommands{
        //clang-format off
        TestCommand{ CommandType::WaitForAgent, 3, 8 }, TestCommand{ CommandType::StoreOfmStripe, 4, 0 }
        //clang-format on
    };

    const std::vector<TestCommand> expectedPleCommands{
        // clang-format off
        // clang-format on
    };

    const std::vector<TestCommand> expectedDmaRdCommands{
        // clang-format off
        TestCommand { CommandType::LoadIfmStripe, 0, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 1 },
        TestCommand { CommandType::LoadIfmStripe, 0, 2 },
        TestCommand { CommandType::LoadWgtStripe, 1, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 3 },
        TestCommand { CommandType::WaitForAgent, 3, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 4 },
        TestCommand { CommandType::WaitForAgent, 3, 1 },
        TestCommand { CommandType::LoadIfmStripe, 0, 5 },
        TestCommand { CommandType::LoadIfmStripe, 0, 6 },
        TestCommand { CommandType::WaitForAgent, 3, 2 },
        TestCommand { CommandType::LoadIfmStripe, 0, 7 },
        TestCommand { CommandType::LoadIfmStripe, 0, 8 },
        TestCommand { CommandType::LoadWgtStripe, 1, 1 },
        TestCommand { CommandType::LoadIfmStripe, 0, 9 },
        TestCommand { CommandType::WaitForAgent, 3, 3 },
        TestCommand { CommandType::LoadIfmStripe, 0, 10 },
        TestCommand { CommandType::WaitForAgent, 3, 4 },
        TestCommand { CommandType::LoadIfmStripe, 0, 11 },
        TestCommand { CommandType::LoadIfmStripe, 0, 12 },
        TestCommand { CommandType::WaitForAgent, 3, 5 },
        TestCommand { CommandType::LoadIfmStripe, 0, 13 },
        TestCommand { CommandType::LoadIfmStripe, 0, 14 },
        TestCommand { CommandType::LoadWgtStripe, 1, 2 },
        TestCommand { CommandType::LoadIfmStripe, 0, 15 },
        TestCommand { CommandType::WaitForAgent, 3, 6 },
        TestCommand { CommandType::LoadIfmStripe, 0, 16 },
        TestCommand { CommandType::WaitForAgent, 3, 7 },
        TestCommand { CommandType::LoadIfmStripe, 0, 17 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedMceCommands{
        // clang-format off
        TestCommand { CommandType::ProgramMceStripe, 3, 0 },
        TestCommand { CommandType::WaitForAgent, 0, 2 },
        TestCommand { CommandType::WaitForAgent, 1, 0 },
        TestCommand { CommandType::StartMceStripe, 3, 0 },
        TestCommand { CommandType::ProgramMceStripe, 3, 1 },
        TestCommand { CommandType::WaitForAgent, 0, 4 },
        TestCommand { CommandType::StartMceStripe, 3, 1 },
        TestCommand { CommandType::ProgramMceStripe, 3, 2 },
        TestCommand { CommandType::WaitForAgent, 0, 5 },
        TestCommand { CommandType::StartMceStripe, 3, 2 },
        TestCommand { CommandType::ProgramMceStripe, 3, 3 },
        TestCommand { CommandType::WaitForAgent, 0, 8 },
        TestCommand { CommandType::WaitForAgent, 1, 1 },
        TestCommand { CommandType::StartMceStripe, 3, 3 },
        TestCommand { CommandType::ProgramMceStripe, 3, 4 },
        TestCommand { CommandType::WaitForAgent, 0, 10 },
        TestCommand { CommandType::StartMceStripe, 3, 4 },
        TestCommand { CommandType::ProgramMceStripe, 3, 5 },
        TestCommand { CommandType::WaitForAgent, 0, 11 },
        TestCommand { CommandType::StartMceStripe, 3, 5 },
        TestCommand { CommandType::ProgramMceStripe, 3, 6 },
        TestCommand { CommandType::WaitForAgent, 0, 14 },
        TestCommand { CommandType::WaitForAgent, 1, 2 },
        TestCommand { CommandType::StartMceStripe, 3, 6 },
        TestCommand { CommandType::ProgramMceStripe, 3, 7 },
        TestCommand { CommandType::WaitForAgent, 0, 16 },
        TestCommand { CommandType::StartMceStripe, 3, 7 },
        TestCommand { CommandType::ProgramMceStripe, 3, 8 },
        TestCommand { CommandType::WaitForAgent, 0, 17 },
        TestCommand { CommandType::StartMceStripe, 3, 8 }
        // clang-format on
    };
    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, caps, debuggingContext);
    scheduler.Schedule();

    CompareCommandArrays(scheduler.GetDmaRdCommands(), expectedDmaRdCommands);
    CompareCommandArrays(scheduler.GetDmaWrCommands(), expectedDmaWrCommands);
    CompareCommandArrays(scheduler.GetMceCommands(), expectedMceCommands);
    CompareCommandArrays(scheduler.GetPleCommands(), expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/PleLoaderStripe/WithReadDependency")
{
    HardwareCapabilities caps = GetEthosN78HwCapabilities();

    WgtSDesc wgts      = MakeWgtSDesc();
    wgts.tile.numSlots = 2;

    PleSDesc ples         = MakePleSDesc();
    ples.ofmTile.numSlots = 2;

    std::vector<AgentDescAndDeps> cmdStream{
        AgentDescAndDeps{
            AgentDesc(9, MakeMceSDesc()),
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
            AgentDesc(12, MakeMceSDesc()),
            {
                /* .readDependencies =*/
                { {
                    { 3, { 1, 12 }, { 1, 12 }, 0 },
                    { 2, { 3, 12 }, { 1, 4 }, 0 },
                } },
                /* .writeDependencies =*/{},
            },
        },
        AgentDescAndDeps{ AgentDesc(1, MakeOfmSDesc()),
                          {
                              /* .readDependencies =*/{ { { 1, { 12, 1 }, { 12, 1 }, 0 } } },
                              /* .writeDependencies =*/{},
                          } },

    };

    const std::vector<TestCommand> expectedDmaWrCommands{
        //clang-format off
        TestCommand{ CommandType::WaitForAgent, 4, 11 }, TestCommand{ CommandType::StoreOfmStripe, 5, 0 }
        //clang-format on
    };

    const std::vector<TestCommand> expectedDmaRdCommands{
        // clang-format off
        TestCommand { CommandType::LoadWgtStripe, 2, 0 },
        TestCommand { CommandType::LoadWgtStripe, 2, 1 },
        TestCommand { CommandType::WaitForAgent, 4, 3 },
        TestCommand { CommandType::LoadWgtStripe, 2, 2 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedMceCommands{
        // clang-format off
        TestCommand { CommandType::ProgramMceStripe, 0, 0 },
        TestCommand { CommandType::StartMceStripe, 0, 0 },
        TestCommand { CommandType::ProgramMceStripe, 0, 1 },
        TestCommand { CommandType::StartMceStripe, 0, 1 },
        TestCommand { CommandType::ProgramMceStripe, 0, 2 },
        TestCommand { CommandType::StartMceStripe, 0, 2 },
        TestCommand { CommandType::ProgramMceStripe, 0, 3 },
        TestCommand { CommandType::StartMceStripe, 0, 3 },
        TestCommand { CommandType::ProgramMceStripe, 0, 4 },
        TestCommand { CommandType::StartMceStripe, 0, 4 },
        TestCommand { CommandType::ProgramMceStripe, 0, 5 },
        TestCommand { CommandType::StartMceStripe, 0, 5 },
        TestCommand { CommandType::ProgramMceStripe, 0, 6 },
        TestCommand { CommandType::StartMceStripe, 0, 6 },
        TestCommand { CommandType::ProgramMceStripe, 0, 7 },
        TestCommand { CommandType::StartMceStripe, 0, 7 },
        TestCommand { CommandType::ProgramMceStripe, 0, 8 },
        TestCommand { CommandType::StartMceStripe, 0, 8 },
        TestCommand { CommandType::ProgramMceStripe, 4, 0 },
        TestCommand { CommandType::WaitForAgent, 1, 0 },
        TestCommand { CommandType::WaitForAgent, 2, 0 },
        TestCommand { CommandType::StartMceStripe, 4, 0 },
        TestCommand { CommandType::ProgramMceStripe, 4, 1 },
        TestCommand { CommandType::StartMceStripe, 4, 1 },
        TestCommand { CommandType::ProgramMceStripe, 4, 2 },
        TestCommand { CommandType::StartMceStripe, 4, 2 },
        TestCommand { CommandType::ProgramMceStripe, 4, 3 },
        TestCommand { CommandType::StartMceStripe, 4, 3 },
        TestCommand { CommandType::ProgramMceStripe, 4, 4 },
        TestCommand { CommandType::WaitForAgent, 2, 1 },
        TestCommand { CommandType::StartMceStripe, 4, 4 },
        TestCommand { CommandType::ProgramMceStripe, 4, 5 },
        TestCommand { CommandType::StartMceStripe, 4, 5 },
        TestCommand { CommandType::ProgramMceStripe, 4, 6 },
        TestCommand { CommandType::StartMceStripe, 4, 6 },
        TestCommand { CommandType::ProgramMceStripe, 4, 7 },
        TestCommand { CommandType::StartMceStripe, 4, 7 },
        TestCommand { CommandType::ProgramMceStripe, 4, 8 },
        TestCommand { CommandType::WaitForAgent, 2, 2 },
        TestCommand { CommandType::StartMceStripe, 4, 8 },
        TestCommand { CommandType::ProgramMceStripe, 4, 9 },
        TestCommand { CommandType::StartMceStripe, 4, 9 },
        TestCommand { CommandType::ProgramMceStripe, 4, 10 },
        TestCommand { CommandType::StartMceStripe, 4, 10 },
        TestCommand { CommandType::ProgramMceStripe, 4, 11 },
        TestCommand { CommandType::StartMceStripe, 4, 11 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedPleCommands{
        // clang-format off
        TestCommand { CommandType::StartPleStripe, 1, 0 }
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, caps, debuggingContext);
    scheduler.Schedule();

    CompareCommandArrays(scheduler.GetDmaRdCommands(), expectedDmaRdCommands);
    CompareCommandArrays(scheduler.GetDmaWrCommands(), expectedDmaWrCommands);
    CompareCommandArrays(scheduler.GetMceCommands(), expectedMceCommands);
    CompareCommandArrays(scheduler.GetPleCommands(), expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/PleLoaderStripe/WithWriteDependency")
{
    HardwareCapabilities caps = GetEthosN78HwCapabilities();

    PleSDesc ples         = MakePleSDesc();
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
        AgentDescAndDeps{ AgentDesc(1, MakeOfmSDesc()),
                          {
                              /* .readDependencies =*/{ { { 2, { 2, 1 }, { 2, 1 }, 0 } } },
                              /* .writeDependencies =*/{},
                          } },

    };

    const std::vector<TestCommand> expectedDmaWrCommands{
        //clang-format off
        TestCommand{ CommandType::WaitForAgent, 0, 1 }, TestCommand{ CommandType::StoreOfmStripe, 2, 0 }
        //clang-format on
    };

    const std::vector<TestCommand> expectedMceCommands{
        // clang-format off
        // clang-format on
    };

    const std::vector<TestCommand> expectedDmaRdCommands{
        // clang-format off
        TestCommand { CommandType::LoadPleCode, 0, 0 },
        TestCommand { CommandType::WaitForAgent, 1, 0 },
        TestCommand { CommandType::LoadPleCode, 0, 1 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedPleCommands{
        // clang-format off
        TestCommand { CommandType::WaitForAgent, 0, 0 },
        TestCommand { CommandType::StartPleStripe, 1, 0 }
        // clang-format on
    };
    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, caps, debuggingContext);
    scheduler.Schedule();

    CompareCommandArrays(scheduler.GetDmaRdCommands(), expectedDmaRdCommands);
    CompareCommandArrays(scheduler.GetDmaWrCommands(), expectedDmaWrCommands);
    CompareCommandArrays(scheduler.GetMceCommands(), expectedMceCommands);
    CompareCommandArrays(scheduler.GetPleCommands(), expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/PleSchedulerStripe/NoWriteDependencies")
{
    HardwareCapabilities caps = GetEthosN78HwCapabilities();

    PleSDesc ples         = MakePleSDesc();
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
            AgentDesc(3, MakeMceSDesc()),
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
        AgentDescAndDeps{ AgentDesc(1, MakeOfmSDesc()),
                          {
                              /* .readDependencies =*/{ { { 1, { numStripesTotal, 1 }, { numStripesTotal, 1 }, 0 } } },
                              /* .writeDependencies =*/{},
                          } },

    };

    const std::vector<TestCommand> expectedDmaWrCommands{
        //clang-format off
        TestCommand{ CommandType::WaitForAgent, 2, 2 }, TestCommand{ CommandType::StoreOfmStripe, 3, 0 }
        //clang-format on
    };

    const std::vector<TestCommand> expectedDmaRdCommands{
        // clang-format off
        TestCommand { CommandType::LoadPleCode, 0, 0 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedMceCommands{
        // clang-format off
        TestCommand { CommandType::ProgramMceStripe, 1, 0 },
        TestCommand { CommandType::StartMceStripe, 1, 0 },
        TestCommand { CommandType::ProgramMceStripe, 1, 1 },
        TestCommand { CommandType::StartMceStripe, 1, 1 },
        TestCommand { CommandType::ProgramMceStripe, 1, 2 },
        TestCommand { CommandType::StartMceStripe, 1, 2 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedPleCommands{
        // clang-format off
        TestCommand { CommandType::WaitForAgent, 0, 0 },
        TestCommand { CommandType::StartPleStripe, 2, 0 },
        TestCommand { CommandType::StartPleStripe, 2, 1 },
        TestCommand { CommandType::StartPleStripe, 2, 2 }
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, caps, debuggingContext);
    scheduler.Schedule();

    CompareCommandArrays(scheduler.GetDmaRdCommands(), expectedDmaRdCommands);
    CompareCommandArrays(scheduler.GetDmaWrCommands(), expectedDmaWrCommands);
    CompareCommandArrays(scheduler.GetMceCommands(), expectedMceCommands);
    CompareCommandArrays(scheduler.GetPleCommands(), expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/PleSchedulerStripe/WithWriteDependency/TileSize 1..4")
{
    HardwareCapabilities caps = GetEthosN78HwCapabilities();

    const uint32_t numStripesTotal = 12;

    auto tileSize = GENERATE_COPY(Catch::Generators::range<uint16_t>(1, 4));

    OfmSDesc ofms             = MakeOfmSDesc();
    ofms.fmData.tile.numSlots = 2;

    DYNAMIC_SECTION("For tileSize " << tileSize)
    {
        PleSDesc pleS         = MakePleSDesc();
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
                AgentDesc(12, MakeMceSDesc()),

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

        std::vector<TestCommand> expectedDmaRdCommands;
        std::vector<TestCommand> expectedDmaWrCommands;
        std::vector<TestCommand> expectedMceCommands;
        std::vector<TestCommand> expectedPleCommands;

        switch (tileSize)
        {
            case 1:
                expectedDmaRdCommands = std::vector<TestCommand>{
                    // clang-format off
                    TestCommand { CommandType::LoadPleCode, 0, 0 }
                    // clang-format on
                };

                expectedDmaWrCommands = std::vector<TestCommand>{
                    //clang-format off
                    TestCommand{ CommandType::WaitForAgent, 2, 0 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 0 },
                    TestCommand{ CommandType::WaitForAgent, 2, 1 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 1 },
                    TestCommand{ CommandType::WaitForAgent, 2, 2 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 2 },
                    TestCommand{ CommandType::WaitForAgent, 2, 3 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 3 },
                    TestCommand{ CommandType::WaitForAgent, 2, 4 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 4 },
                    TestCommand{ CommandType::WaitForAgent, 2, 5 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 5 },
                    TestCommand{ CommandType::WaitForAgent, 2, 6 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 6 },
                    TestCommand{ CommandType::WaitForAgent, 2, 7 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 7 },
                    TestCommand{ CommandType::WaitForAgent, 2, 8 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 8 },
                    TestCommand{ CommandType::WaitForAgent, 2, 9 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 9 },
                    TestCommand{ CommandType::WaitForAgent, 2, 10 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 10 },
                    TestCommand{ CommandType::WaitForAgent, 2, 11 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 11 }
                    // clang-format on
                };

                expectedMceCommands = std::vector<TestCommand>{
                    // clang-format off
                    TestCommand { CommandType::ProgramMceStripe, 1, 0 },
                    TestCommand { CommandType::StartMceStripe, 1, 0 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 1 },
                    TestCommand { CommandType::StartMceStripe, 1, 1 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 2 },
                    TestCommand { CommandType::StartMceStripe, 1, 2 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 3 },
                    TestCommand { CommandType::StartMceStripe, 1, 3 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 4 },
                    TestCommand { CommandType::StartMceStripe, 1, 4 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 5 },
                    TestCommand { CommandType::StartMceStripe, 1, 5 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 6 },
                    TestCommand { CommandType::StartMceStripe, 1, 6 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 7 },
                    TestCommand { CommandType::StartMceStripe, 1, 7 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 8 },
                    TestCommand { CommandType::StartMceStripe, 1, 8 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 9 },
                    TestCommand { CommandType::StartMceStripe, 1, 9 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 10 },
                    TestCommand { CommandType::StartMceStripe, 1, 10 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 11 },
                    TestCommand { CommandType::StartMceStripe, 1, 11 }
                    // clang-format on
                };

                expectedPleCommands = std::vector<TestCommand>{
                    // clang-format off
                    TestCommand { CommandType::WaitForAgent, 0, 0 },
                    TestCommand { CommandType::StartPleStripe, 2, 0 },
                    TestCommand { CommandType::WaitForAgent, 3, 0 },
                    TestCommand { CommandType::StartPleStripe, 2, 1 },
                    TestCommand { CommandType::WaitForAgent, 3, 1 },
                    TestCommand { CommandType::StartPleStripe, 2, 2 },
                    TestCommand { CommandType::WaitForAgent, 3, 2 },
                    TestCommand { CommandType::StartPleStripe, 2, 3 },
                    TestCommand { CommandType::WaitForAgent, 3, 3 },
                    TestCommand { CommandType::StartPleStripe, 2, 4 },
                    TestCommand { CommandType::WaitForAgent, 3, 4 },
                    TestCommand { CommandType::StartPleStripe, 2, 5 },
                    TestCommand { CommandType::WaitForAgent, 3, 5 },
                    TestCommand { CommandType::StartPleStripe, 2, 6 },
                    TestCommand { CommandType::WaitForAgent, 3, 6 },
                    TestCommand { CommandType::StartPleStripe, 2, 7 },
                    TestCommand { CommandType::WaitForAgent, 3, 7 },
                    TestCommand { CommandType::StartPleStripe, 2, 8 },
                    TestCommand { CommandType::WaitForAgent, 3, 8 },
                    TestCommand { CommandType::StartPleStripe, 2, 9 },
                    TestCommand { CommandType::WaitForAgent, 3, 9 },
                    TestCommand { CommandType::StartPleStripe, 2, 10 },
                    TestCommand { CommandType::WaitForAgent, 3, 10 },
                    TestCommand { CommandType::StartPleStripe, 2, 11 }
                    // clang-format on
                };
                break;
            case 2:

                expectedDmaRdCommands = std::vector<TestCommand>{
                    // clang-format off
                    TestCommand { CommandType::LoadPleCode, 0, 0 }
                    // clang-format on
                };

                expectedDmaWrCommands = std::vector<TestCommand>{
                    //clang-format off
                    TestCommand{ CommandType::WaitForAgent, 2, 0 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 0 },
                    TestCommand{ CommandType::WaitForAgent, 2, 1 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 1 },
                    TestCommand{ CommandType::WaitForAgent, 2, 2 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 2 },
                    TestCommand{ CommandType::WaitForAgent, 2, 3 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 3 },
                    TestCommand{ CommandType::WaitForAgent, 2, 4 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 4 },
                    TestCommand{ CommandType::WaitForAgent, 2, 5 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 5 },
                    TestCommand{ CommandType::WaitForAgent, 2, 6 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 6 },
                    TestCommand{ CommandType::WaitForAgent, 2, 7 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 7 },
                    TestCommand{ CommandType::WaitForAgent, 2, 8 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 8 },
                    TestCommand{ CommandType::WaitForAgent, 2, 9 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 9 },
                    TestCommand{ CommandType::WaitForAgent, 2, 10 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 10 },
                    TestCommand{ CommandType::WaitForAgent, 2, 11 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 11 }
                    // clang-format on
                };

                expectedMceCommands = std::vector<TestCommand>{
                    // clang-format off
                    TestCommand { CommandType::ProgramMceStripe, 1, 0 },
                    TestCommand { CommandType::StartMceStripe, 1, 0 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 1 },
                    TestCommand { CommandType::StartMceStripe, 1, 1 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 2 },
                    TestCommand { CommandType::StartMceStripe, 1, 2 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 3 },
                    TestCommand { CommandType::StartMceStripe, 1, 3 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 4 },
                    TestCommand { CommandType::StartMceStripe, 1, 4 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 5 },
                    TestCommand { CommandType::StartMceStripe, 1, 5 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 6 },
                    TestCommand { CommandType::StartMceStripe, 1, 6 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 7 },
                    TestCommand { CommandType::StartMceStripe, 1, 7 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 8 },
                    TestCommand { CommandType::StartMceStripe, 1, 8 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 9 },
                    TestCommand { CommandType::StartMceStripe, 1, 9 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 10 },
                    TestCommand { CommandType::StartMceStripe, 1, 10 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 11 },
                    TestCommand { CommandType::StartMceStripe, 1, 11 }
                    // clang-format on
                };

                expectedPleCommands = std::vector<TestCommand>{
                    // clang-format off
                    TestCommand { CommandType::WaitForAgent, 0, 0 },
                    TestCommand { CommandType::StartPleStripe, 2, 0 },
                    TestCommand { CommandType::StartPleStripe, 2, 1 },
                    TestCommand { CommandType::WaitForAgent, 3, 0 },
                    TestCommand { CommandType::StartPleStripe, 2, 2 },
                    TestCommand { CommandType::WaitForAgent, 3, 1 },
                    TestCommand { CommandType::StartPleStripe, 2, 3 },
                    TestCommand { CommandType::WaitForAgent, 3, 2 },
                    TestCommand { CommandType::StartPleStripe, 2, 4 },
                    TestCommand { CommandType::WaitForAgent, 3, 3 },
                    TestCommand { CommandType::StartPleStripe, 2, 5 },
                    TestCommand { CommandType::WaitForAgent, 3, 4 },
                    TestCommand { CommandType::StartPleStripe, 2, 6 },
                    TestCommand { CommandType::WaitForAgent, 3, 5 },
                    TestCommand { CommandType::StartPleStripe, 2, 7 },
                    TestCommand { CommandType::WaitForAgent, 3, 6 },
                    TestCommand { CommandType::StartPleStripe, 2, 8 },
                    TestCommand { CommandType::WaitForAgent, 3, 7 },
                    TestCommand { CommandType::StartPleStripe, 2, 9 },
                    TestCommand { CommandType::WaitForAgent, 3, 8 },
                    TestCommand { CommandType::StartPleStripe, 2, 10 },
                    TestCommand { CommandType::WaitForAgent, 3, 9 },
                    TestCommand { CommandType::StartPleStripe, 2, 11 }
                    // clang-format on
                };
                break;
            case 3:

                expectedDmaRdCommands = std::vector<TestCommand>{
                    // clang-format off
                    TestCommand { CommandType::LoadPleCode, 0, 0 }
                    // clang-format on
                };

                expectedDmaWrCommands = std::vector<TestCommand>{
                    //clang-format off
                    TestCommand{ CommandType::WaitForAgent, 2, 0 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 0 },
                    TestCommand{ CommandType::WaitForAgent, 2, 1 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 1 },
                    TestCommand{ CommandType::WaitForAgent, 2, 2 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 2 },
                    TestCommand{ CommandType::WaitForAgent, 2, 3 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 3 },
                    TestCommand{ CommandType::WaitForAgent, 2, 4 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 4 },
                    TestCommand{ CommandType::WaitForAgent, 2, 5 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 5 },
                    TestCommand{ CommandType::WaitForAgent, 2, 6 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 6 },
                    TestCommand{ CommandType::WaitForAgent, 2, 7 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 7 },
                    TestCommand{ CommandType::WaitForAgent, 2, 8 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 8 },
                    TestCommand{ CommandType::WaitForAgent, 2, 9 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 9 },
                    TestCommand{ CommandType::WaitForAgent, 2, 10 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 10 },
                    TestCommand{ CommandType::WaitForAgent, 2, 11 },
                    TestCommand{ CommandType::StoreOfmStripe, 3, 11 }
                    // clang-format on
                };

                expectedMceCommands = std::vector<TestCommand>{
                    // clang-format off
                    TestCommand { CommandType::ProgramMceStripe, 1, 0 },
                    TestCommand { CommandType::StartMceStripe, 1, 0 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 1 },
                    TestCommand { CommandType::StartMceStripe, 1, 1 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 2 },
                    TestCommand { CommandType::StartMceStripe, 1, 2 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 3 },
                    TestCommand { CommandType::StartMceStripe, 1, 3 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 4 },
                    TestCommand { CommandType::StartMceStripe, 1, 4 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 5 },
                    TestCommand { CommandType::StartMceStripe, 1, 5 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 6 },
                    TestCommand { CommandType::StartMceStripe, 1, 6 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 7 },
                    TestCommand { CommandType::StartMceStripe, 1, 7 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 8 },
                    TestCommand { CommandType::StartMceStripe, 1, 8 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 9 },
                    TestCommand { CommandType::StartMceStripe, 1, 9 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 10 },
                    TestCommand { CommandType::StartMceStripe, 1, 10 },
                    TestCommand { CommandType::ProgramMceStripe, 1, 11 },
                    TestCommand { CommandType::StartMceStripe, 1, 11 }
                    // clang-format on
                };

                expectedPleCommands = std::vector<TestCommand>{
                    // clang-format off
                    TestCommand { CommandType::WaitForAgent, 0, 0 },
                    TestCommand { CommandType::StartPleStripe, 2, 0 },
                    TestCommand { CommandType::StartPleStripe, 2, 1 },
                    TestCommand { CommandType::StartPleStripe, 2, 2 },
                    TestCommand { CommandType::WaitForAgent, 3, 0 },
                    TestCommand { CommandType::StartPleStripe, 2, 3 },
                    TestCommand { CommandType::WaitForAgent, 3, 1 },
                    TestCommand { CommandType::StartPleStripe, 2, 4 },
                    TestCommand { CommandType::WaitForAgent, 3, 2 },
                    TestCommand { CommandType::StartPleStripe, 2, 5 },
                    TestCommand { CommandType::WaitForAgent, 3, 3 },
                    TestCommand { CommandType::StartPleStripe, 2, 6 },
                    TestCommand { CommandType::WaitForAgent, 3, 4 },
                    TestCommand { CommandType::StartPleStripe, 2, 7 },
                    TestCommand { CommandType::WaitForAgent, 3, 5 },
                    TestCommand { CommandType::StartPleStripe, 2, 8 },
                    TestCommand { CommandType::WaitForAgent, 3, 6 },
                    TestCommand { CommandType::StartPleStripe, 2, 9 },
                    TestCommand { CommandType::WaitForAgent, 3, 7 },
                    TestCommand { CommandType::StartPleStripe, 2, 10 },
                    TestCommand { CommandType::WaitForAgent, 3, 8 },
                    TestCommand { CommandType::StartPleStripe, 2, 11 }
                    // clang-format on
                };

                break;
            default:
                FAIL("Invalid tile size");
        }

        ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

        Scheduler scheduler(cmdStream, caps, debuggingContext);
        scheduler.Schedule();

        CompareCommandArrays(scheduler.GetDmaRdCommands(), expectedDmaRdCommands);
        CompareCommandArrays(scheduler.GetDmaWrCommands(), expectedDmaWrCommands);
        CompareCommandArrays(scheduler.GetMceCommands(), expectedMceCommands);
        CompareCommandArrays(scheduler.GetPleCommands(), expectedPleCommands);
    }
}

TEST_CASE("Cascading/StripeScheduler/PleSchedulerStripe/ReadDependencyToMceSIsFirst")
{
    HardwareCapabilities caps = GetEthosN78HwCapabilities();

    PleSDesc ples         = MakePleSDesc();
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
            AgentDesc(3, MakeMceSDesc()),
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
        AgentDescAndDeps{ AgentDesc(1, MakeOfmSDesc()),
                          {
                              /* .readDependencies =*/{ { { 1, { numStripesTotal, 1 }, { numStripesTotal, 1 }, 0 } } },
                              /* .writeDependencies =*/{},
                          } },
    };
    const std::vector<TestCommand> expectedDmaWrCommands{
        //clang-format off
        TestCommand{ CommandType::WaitForAgent, 2, 2 }, TestCommand{ CommandType::StoreOfmStripe, 3, 0 }
        //clang-format on
    };

    const std::vector<TestCommand> expectedDmaRdCommands{
        // clang-format off
        TestCommand { CommandType::LoadPleCode, 0, 0 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedMceCommands{
        // clang-format off
        TestCommand { CommandType::ProgramMceStripe, 1, 0 },
        TestCommand { CommandType::StartMceStripe, 1, 0 },
        TestCommand { CommandType::ProgramMceStripe, 1, 1 },
        TestCommand { CommandType::StartMceStripe, 1, 1 },
        TestCommand { CommandType::ProgramMceStripe, 1, 2 },
        TestCommand { CommandType::StartMceStripe, 1, 2 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedPleCommands{
        // clang-format off
        TestCommand { CommandType::WaitForAgent, 0, 0 },
        TestCommand { CommandType::StartPleStripe, 2, 0 },
        TestCommand { CommandType::StartPleStripe, 2, 1 },
        TestCommand { CommandType::StartPleStripe, 2, 2 }
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, caps, debuggingContext);
    scheduler.Schedule();

    CompareCommandArrays(scheduler.GetDmaRdCommands(), expectedDmaRdCommands);
    CompareCommandArrays(scheduler.GetDmaWrCommands(), expectedDmaWrCommands);
    CompareCommandArrays(scheduler.GetMceCommands(), expectedMceCommands);
    CompareCommandArrays(scheduler.GetPleCommands(), expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/PleSchedulerStripe/ReadDependencyTowardsIfmS")
{
    HardwareCapabilities caps = GetEthosN78HwCapabilities();

    PleSDesc ples         = MakePleSDesc();
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
            AgentDesc(3, MakeIfmSDesc()),
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
        AgentDescAndDeps{ AgentDesc(1, MakeOfmSDesc()),
                          {
                              /* .readDependencies =*/{ { { 1, { numStripesTotal, 1 }, { numStripesTotal, 1 }, 0 } } },
                              /* .writeDependencies =*/{},
                          } },
    };
    const std::vector<TestCommand> expectedDmaWrCommands{
        //clang-format off
        TestCommand{ CommandType::WaitForAgent, 2, 2 }, TestCommand{ CommandType::StoreOfmStripe, 3, 0 }
        //clang-format on
    };

    const std::vector<TestCommand> expectedMceCommands{
        // clang-format off
        // clang-format on
    };

    const std::vector<TestCommand> expectedDmaRdCommands{
        // clang-format off
        TestCommand { CommandType::LoadPleCode, 0, 0 },
        TestCommand { CommandType::LoadIfmStripe, 1, 0 },
        TestCommand { CommandType::LoadIfmStripe, 1, 1 },
        TestCommand { CommandType::LoadIfmStripe, 1, 2 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedPleCommands{
        // clang-format off
        TestCommand { CommandType::WaitForAgent, 0, 0 },
        TestCommand { CommandType::WaitForAgent, 1, 0 },
        TestCommand { CommandType::StartPleStripe, 2, 0 },
        TestCommand { CommandType::WaitForAgent, 1, 1 },
        TestCommand { CommandType::StartPleStripe, 2, 1 },
        TestCommand { CommandType::WaitForAgent, 1, 2 },
        TestCommand { CommandType::StartPleStripe, 2, 2 }
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, caps, debuggingContext);
    scheduler.Schedule();

    CompareCommandArrays(scheduler.GetDmaRdCommands(), expectedDmaRdCommands);
    CompareCommandArrays(scheduler.GetDmaWrCommands(), expectedDmaWrCommands);
    CompareCommandArrays(scheduler.GetMceCommands(), expectedMceCommands);
    CompareCommandArrays(scheduler.GetPleCommands(), expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/PleSchedulerStripe/Strategy0Cascading/FirstPle")
{
    HardwareCapabilities caps = GetEthosN78HwCapabilities();

    PleSDesc ples         = MakePleSDesc();
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
            AgentDesc(5, MakeMceSDesc()),
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
        AgentDescAndDeps{ AgentDesc(1, MakeOfmSDesc()),
                          {
                              /* .readDependencies =*/{ { { 1, { numStripesTotal, 1 }, { numStripesTotal, 1 }, 0 } } },
                              /* .writeDependencies =*/{},
                          } },
    };

    const std::vector<TestCommand> expectedDmaWrCommands{
        //clang-format off
        TestCommand{ CommandType::WaitForAgent, 2, 4 }, TestCommand{ CommandType::StoreOfmStripe, 3, 0 }
        //clang-format on
    };

    const std::vector<TestCommand> expectedDmaRdCommands{
        // clang-format off
        TestCommand { CommandType::LoadPleCode, 0, 0 },
        TestCommand { CommandType::LoadPleCode, 0, 1 },
        TestCommand { CommandType::LoadPleCode, 0, 2 },
        TestCommand { CommandType::LoadPleCode, 0, 3 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedMceCommands{
        // clang-format off
        TestCommand { CommandType::ProgramMceStripe, 1, 0 },
        TestCommand { CommandType::StartMceStripe, 1, 0 },
        TestCommand { CommandType::ProgramMceStripe, 1, 1 },
        TestCommand { CommandType::StartMceStripe, 1, 1 },
        TestCommand { CommandType::ProgramMceStripe, 1, 2 },
        TestCommand { CommandType::StartMceStripe, 1, 2 },
        TestCommand { CommandType::ProgramMceStripe, 1, 3 },
        TestCommand { CommandType::StartMceStripe, 1, 3 },
        TestCommand { CommandType::ProgramMceStripe, 1, 4 },
        TestCommand { CommandType::StartMceStripe, 1, 4 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedPleCommands{
        // clang-format off
        TestCommand { CommandType::WaitForAgent, 0, 0 },
        TestCommand { CommandType::StartPleStripe, 2, 0 },
        TestCommand { CommandType::StartPleStripe, 2, 1 },
        TestCommand { CommandType::WaitForAgent, 0, 1 },
        TestCommand { CommandType::StartPleStripe, 2, 2 },
        TestCommand { CommandType::WaitForAgent, 0, 2 },
        TestCommand { CommandType::StartPleStripe, 2, 3 },
        TestCommand { CommandType::WaitForAgent, 0, 3 },
        TestCommand { CommandType::StartPleStripe, 2, 4 }
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, caps, debuggingContext);
    scheduler.Schedule();

    CompareCommandArrays(scheduler.GetDmaRdCommands(), expectedDmaRdCommands);
    CompareCommandArrays(scheduler.GetDmaWrCommands(), expectedDmaWrCommands);
    CompareCommandArrays(scheduler.GetMceCommands(), expectedMceCommands);
    CompareCommandArrays(scheduler.GetPleCommands(), expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/PleSchedulerStripe/Strategy0Cascading/SecondPle")
{
    HardwareCapabilities caps = GetEthosN78HwCapabilities();

    const uint32_t numStripesTotal = 4;
    PleSDesc ples                  = MakePleSDesc();
    ples.ofmTile.numSlots          = 4;

    OfmSDesc ofms             = MakeOfmSDesc();
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
            AgentDesc(4, MakeMceSDesc()),
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

    const std::vector<TestCommand> expectedDmaRdCommands{
        // clang-format off
        TestCommand { CommandType::LoadPleCode, 0, 0 },
        TestCommand { CommandType::LoadPleCode, 0, 1 },
        TestCommand { CommandType::LoadPleCode, 0, 2 },
        TestCommand { CommandType::LoadPleCode, 0, 3 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedDmaWrCommands{
        //clang-format off
        TestCommand{ CommandType::WaitForAgent, 2, 0 },
        TestCommand{ CommandType::StoreOfmStripe, 3, 0 },
        TestCommand{ CommandType::WaitForAgent, 2, 1 },
        TestCommand{ CommandType::StoreOfmStripe, 3, 1 },
        TestCommand{ CommandType::WaitForAgent, 2, 2 },
        TestCommand{ CommandType::StoreOfmStripe, 3, 2 },
        TestCommand{ CommandType::WaitForAgent, 2, 3 },
        TestCommand{ CommandType::StoreOfmStripe, 3, 3 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedMceCommands{
        // clang-format off
        TestCommand { CommandType::ProgramMceStripe, 1, 0 },
        TestCommand { CommandType::StartMceStripe, 1, 0 },
        TestCommand { CommandType::ProgramMceStripe, 1, 1 },
        TestCommand { CommandType::StartMceStripe, 1, 1 },
        TestCommand { CommandType::ProgramMceStripe, 1, 2 },
        TestCommand { CommandType::StartMceStripe, 1, 2 },
        TestCommand { CommandType::ProgramMceStripe, 1, 3 },
        TestCommand { CommandType::StartMceStripe, 1, 3 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedPleCommands{
        // clang-format off
        TestCommand { CommandType::WaitForAgent, 0, 0 },
        TestCommand { CommandType::StartPleStripe, 2, 0 },
        TestCommand { CommandType::WaitForAgent, 0, 1 },
        TestCommand { CommandType::StartPleStripe, 2, 1 },
        TestCommand { CommandType::WaitForAgent, 0, 2 },
        TestCommand { CommandType::StartPleStripe, 2, 2 },
        TestCommand { CommandType::WaitForAgent, 0, 3 },
        TestCommand { CommandType::StartPleStripe, 2, 3 }
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, caps, debuggingContext);
    scheduler.Schedule();

    CompareCommandArrays(scheduler.GetDmaRdCommands(), expectedDmaRdCommands);
    CompareCommandArrays(scheduler.GetDmaWrCommands(), expectedDmaWrCommands);
    CompareCommandArrays(scheduler.GetMceCommands(), expectedMceCommands);
    CompareCommandArrays(scheduler.GetPleCommands(), expectedPleCommands);
}

TEST_CASE("Cascading/StripeScheduler/OfmStreamerStripe")
{
    HardwareCapabilities caps = GetEthosN78HwCapabilities();

    PleSDesc ples         = MakePleSDesc();
    ples.ofmTile.numSlots = 2;

    OfmSDesc ofms             = MakeOfmSDesc();
    ofms.fmData.tile.numSlots = 2;

    std::vector<AgentDescAndDeps> cmdStream{
        AgentDescAndDeps{
            AgentDesc(12, MakeMceSDesc()),
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

    const std::vector<TestCommand> expectedDmaRdCommands{
        // clang-format off
        // clang-format on
    };

    const std::vector<TestCommand> expectedDmaWrCommands{
        //clang-format off
        TestCommand{ CommandType::WaitForAgent, 1, 0 },
        TestCommand{ CommandType::StoreOfmStripe, 2, 0 },
        TestCommand{ CommandType::WaitForAgent, 1, 1 },
        TestCommand{ CommandType::StoreOfmStripe, 2, 1 },
        TestCommand{ CommandType::WaitForAgent, 1, 2 },
        TestCommand{ CommandType::StoreOfmStripe, 2, 2 },
        TestCommand{ CommandType::WaitForAgent, 1, 3 },
        TestCommand{ CommandType::StoreOfmStripe, 2, 3 },
        TestCommand{ CommandType::WaitForAgent, 1, 4 },
        TestCommand{ CommandType::StoreOfmStripe, 2, 4 },
        TestCommand{ CommandType::WaitForAgent, 1, 5 },
        TestCommand{ CommandType::StoreOfmStripe, 2, 5 },
        TestCommand{ CommandType::WaitForAgent, 1, 6 },
        TestCommand{ CommandType::StoreOfmStripe, 2, 6 },
        TestCommand{ CommandType::WaitForAgent, 1, 7 },
        TestCommand{ CommandType::StoreOfmStripe, 2, 7 },
        TestCommand{ CommandType::WaitForAgent, 1, 8 },
        TestCommand{ CommandType::StoreOfmStripe, 2, 8 },
        TestCommand{ CommandType::WaitForAgent, 1, 9 },
        TestCommand{ CommandType::StoreOfmStripe, 2, 9 },
        TestCommand{ CommandType::WaitForAgent, 1, 10 },
        TestCommand{ CommandType::StoreOfmStripe, 2, 10 },
        TestCommand{ CommandType::WaitForAgent, 1, 11 },
        TestCommand{ CommandType::StoreOfmStripe, 2, 11 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedMceCommands{
        // clang-format off
        TestCommand { CommandType::ProgramMceStripe, 0, 0 },
        TestCommand { CommandType::StartMceStripe, 0, 0 },
        TestCommand { CommandType::ProgramMceStripe, 0, 1 },
        TestCommand { CommandType::StartMceStripe, 0, 1 },
        TestCommand { CommandType::ProgramMceStripe, 0, 2 },
        TestCommand { CommandType::StartMceStripe, 0, 2 },
        TestCommand { CommandType::ProgramMceStripe, 0, 3 },
        TestCommand { CommandType::StartMceStripe, 0, 3 },
        TestCommand { CommandType::ProgramMceStripe, 0, 4 },
        TestCommand { CommandType::StartMceStripe, 0, 4 },
        TestCommand { CommandType::ProgramMceStripe, 0, 5 },
        TestCommand { CommandType::StartMceStripe, 0, 5 },
        TestCommand { CommandType::ProgramMceStripe, 0, 6 },
        TestCommand { CommandType::StartMceStripe, 0, 6 },
        TestCommand { CommandType::ProgramMceStripe, 0, 7 },
        TestCommand { CommandType::StartMceStripe, 0, 7 },
        TestCommand { CommandType::ProgramMceStripe, 0, 8 },
        TestCommand { CommandType::StartMceStripe, 0, 8 },
        TestCommand { CommandType::ProgramMceStripe, 0, 9 },
        TestCommand { CommandType::StartMceStripe, 0, 9 },
        TestCommand { CommandType::ProgramMceStripe, 0, 10 },
        TestCommand { CommandType::StartMceStripe, 0, 10 },
        TestCommand { CommandType::ProgramMceStripe, 0, 11 },
        TestCommand { CommandType::StartMceStripe, 0, 11 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedPleCommands{
        // clang-format off
        TestCommand { CommandType::StartPleStripe, 1, 0 },
        TestCommand { CommandType::StartPleStripe, 1, 1 },
        TestCommand { CommandType::WaitForAgent, 2, 0 },
        TestCommand { CommandType::StartPleStripe, 1, 2 },
        TestCommand { CommandType::WaitForAgent, 2, 1 },
        TestCommand { CommandType::StartPleStripe, 1, 3 },
        TestCommand { CommandType::WaitForAgent, 2, 2 },
        TestCommand { CommandType::StartPleStripe, 1, 4 },
        TestCommand { CommandType::WaitForAgent, 2, 3 },
        TestCommand { CommandType::StartPleStripe, 1, 5 },
        TestCommand { CommandType::WaitForAgent, 2, 4 },
        TestCommand { CommandType::StartPleStripe, 1, 6 },
        TestCommand { CommandType::WaitForAgent, 2, 5 },
        TestCommand { CommandType::StartPleStripe, 1, 7 },
        TestCommand { CommandType::WaitForAgent, 2, 6 },
        TestCommand { CommandType::StartPleStripe, 1, 8 },
        TestCommand { CommandType::WaitForAgent, 2, 7 },
        TestCommand { CommandType::StartPleStripe, 1, 9 },
        TestCommand { CommandType::WaitForAgent, 2, 8 },
        TestCommand { CommandType::StartPleStripe, 1, 10 },
        TestCommand { CommandType::WaitForAgent, 2, 9 },
        TestCommand { CommandType::StartPleStripe, 1, 11 }
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(cmdStream, caps, debuggingContext);
    scheduler.Schedule();

    CompareCommandArrays(scheduler.GetDmaRdCommands(), expectedDmaRdCommands);
    CompareCommandArrays(scheduler.GetDmaWrCommands(), expectedDmaWrCommands);
    CompareCommandArrays(scheduler.GetMceCommands(), expectedMceCommands);
    CompareCommandArrays(scheduler.GetPleCommands(), expectedPleCommands);
}
