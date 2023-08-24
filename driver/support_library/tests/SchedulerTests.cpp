//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/DebuggingContext.hpp"
#include "../src/cascading/Scheduler.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>

using namespace ethosn::support_library;
using namespace ethosn::command_stream;
using namespace ethosn::command_stream::cascading;
using CommandVariant = ethosn::command_stream::CommandVariant;

const char* CommandTypeToString(CommandType t)
{
    using namespace ethosn::command_stream::cascading;
    switch (t)
    {
        case CommandType::WaitForCounter:
            return "WaitForCounter";
        case CommandType::LoadIfmStripe:
            return "LoadIfmStripe";
        case CommandType::LoadWgtStripe:
            return "LoadWgtStripe";
        case CommandType::ProgramMceStripe:
            return "ProgramMceStripe";
        case CommandType::ConfigMceif:
            return "ConfigMceif";
        case CommandType::StartMceStripe:
            return "StartMceStripe";
        case CommandType::LoadPleCodeIntoSram:
            return "LoadPleCodeIntoSram";
        case CommandType::LoadPleCodeIntoPleSram:
            return "LoadPleCodeIntoPleSram";
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
    uint32_t paramA;
    uint32_t paramB;
};

inline bool operator==(const TestCommand& lhs, const TestCommand& rhs)
{
    static_assert(sizeof(TestCommand) == sizeof(lhs.type) + sizeof(lhs.paramA) + sizeof(lhs.paramB),
                  "New fields added");
    return lhs.type == rhs.type && lhs.paramA == rhs.paramA && lhs.paramB == rhs.paramB;
}

namespace Catch
{
template <>
struct StringMaker<TestCommand>
{
    static std::string convert(const TestCommand& c)
    {
        return "\n  TestCommand { CommandType::" + std::string(CommandTypeToString(c.type)) + ", " +
               std::to_string(c.paramA) + ", " + std::to_string(c.paramB) + " }";
    }
};
}    // namespace Catch

void CompareCommandArrays(const std::vector<CommandVariant>& a, const std::vector<TestCommand>& b)
{
    std::vector<TestCommand> converted;
    for (const CommandVariant& cmd : a)
    {
        TestCommand c = {};
        c.type        = cmd.type;
        switch (cmd.type)
        {
            case CommandType::WaitForCounter:
                c.paramA = static_cast<uint32_t>(cmd.waitForCounter.counterName);
                c.paramB = cmd.waitForCounter.counterValue;
                break;
            case CommandType::LoadIfmStripe:
                c.paramA = cmd.dma.agentId;
                break;
            case CommandType::LoadWgtStripe:
                c.paramA = cmd.dma.agentId;
                break;
            case CommandType::ProgramMceStripe:
                c.paramA = cmd.programMceStripe.agentId;
                break;
            case CommandType::ConfigMceif:
                c.paramA = cmd.configMceif.agentId;
                break;
            case CommandType::StartMceStripe:
                c.paramA = cmd.startMceStripe.agentId;
                break;
            case CommandType::LoadPleCodeIntoSram:
                c.paramA = cmd.dma.agentId;
                break;
            case CommandType::LoadPleCodeIntoPleSram:
                c.paramA = cmd.loadPleCodeIntoPleSram.agentId;
                break;
            case CommandType::StartPleStripe:
                c.paramA = cmd.startPleStripe.agentId;
                break;
            case CommandType::StoreOfmStripe:
                c.paramA = cmd.dma.agentId;
                break;
            default:
                REQUIRE(false);
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
    result.pleKernelId      = PleKernelId::V4442_PASSTHROUGH_bw8_bh8_bm1;
    return result;
}

PleSDesc MakePleSDesc()
{
    PleSDesc result{};
    static HardwareCapabilities hwCaps = GetEthosN78HwCapabilities();
    static PleOp pleOp(
        PleOperation::PASSTHROUGH, 1, std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 8 } }, TensorShape{ 1, 8, 8, 8 },
        true, hwCaps, std::map<std::string, std::string>{},
        std::map<std::string, int>{ { "block_width", 8 }, { "block_height", 32 }, { "block_multiplier", 1 } },
        std::map<std::string, int>{});
    result.m_PleOp          = &pleOp;
    result.ofmTile.numSlots = 1;
    result.pleKernelId      = PleKernelId::V4442_PASSTHROUGH_bw8_bh8_bm1;
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
    IfmSDesc ifms = MakeIfmSDesc();
    WgtSDesc wgts = MakeWgtSDesc();
    PleSDesc ples = MakePleSDesc();
    OfmSDesc ofms = MakeOfmSDesc();

    std::vector<AgentDescAndDeps> complexSingleLayerCmdStream{
        AgentDescAndDeps{
            AgentDesc(18, ifms),
            {
                { { 3, { 3, 6 }, { 1, 2 }, 1, 4, true, true } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(3, wgts),
            {
                { { 3, { 3, 1 }, { 3, 1 }, 0, 2, true, true } },
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, PleLDesc{}),
            {},
        },
        AgentDescAndDeps{
            AgentDesc(9, MakeMceSDesc()),
            {
                {
                    { 0, { 6, 3 }, { 2, 1 }, 1, -1, true, true },
                    { 1, { 1, 3 }, { 1, 3 }, 0, -1, true, true },
                },
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, ples),
            {
                {
                    { 3, { 9, 1 }, { 9, 1 }, 0, -1, true, false },
                    { 2, { 1, 1 }, { 1, 1 }, 0, -1, true, true },
                },
            },
        },
        AgentDescAndDeps{
            AgentDesc(1, ofms),
            {
                { { 4, { 1, 1 }, { 1, 1 }, 0, -1, true, true } },
            },
        },
    };

    const std::vector<TestCommand> expectedDmaRdCommands{
        // clang-format off
        TestCommand { CommandType::LoadIfmStripe, 0, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 0 },
        TestCommand { CommandType::LoadWgtStripe, 1, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 0 },
        TestCommand { CommandType::WaitForCounter, 3, 1 },
        TestCommand { CommandType::LoadIfmStripe, 0, 0 },
        TestCommand { CommandType::WaitForCounter, 3, 2 },
        TestCommand { CommandType::LoadIfmStripe, 0, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 0 },
        TestCommand { CommandType::WaitForCounter, 3, 3 },
        TestCommand { CommandType::LoadIfmStripe, 0, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 0 },
        TestCommand { CommandType::LoadWgtStripe, 1, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 0 },
        TestCommand { CommandType::WaitForCounter, 3, 4 },
        TestCommand { CommandType::LoadIfmStripe, 0, 0 },
        TestCommand { CommandType::WaitForCounter, 3, 5 },
        TestCommand { CommandType::LoadIfmStripe, 0, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 0 },
        TestCommand { CommandType::WaitForCounter, 3, 6 },
        TestCommand { CommandType::LoadIfmStripe, 0, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 0 },
        TestCommand { CommandType::LoadWgtStripe, 1, 0 },
        TestCommand { CommandType::LoadIfmStripe, 0, 0 },
        TestCommand { CommandType::WaitForCounter, 3, 7 },
        TestCommand { CommandType::LoadIfmStripe, 0, 0 },
        TestCommand { CommandType::WaitForCounter, 3, 8 },
        TestCommand { CommandType::LoadIfmStripe, 0, 0 },
        TestCommand { CommandType::LoadPleCodeIntoSram, 2, 0 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedDmaWrCommands{ TestCommand{ CommandType::WaitForCounter, 5, 1 },
                                                          TestCommand{ CommandType::StoreOfmStripe, 5, 0 } };

    const std::vector<TestCommand> expectedMceCommands{
        // clang-format off
        TestCommand { CommandType::ProgramMceStripe, 3, 0 },
        TestCommand { CommandType::WaitForCounter, 0, 4 },
        TestCommand { CommandType::ConfigMceif, 3, 0 },
        TestCommand { CommandType::StartMceStripe, 3, 0 },
        TestCommand { CommandType::ProgramMceStripe, 3, 0 },
        TestCommand { CommandType::WaitForCounter, 0, 6 },
        TestCommand { CommandType::StartMceStripe, 3, 0 },
        TestCommand { CommandType::ProgramMceStripe, 3, 0 },
        TestCommand { CommandType::WaitForCounter, 0, 7 },
        TestCommand { CommandType::StartMceStripe, 3, 0 },
        TestCommand { CommandType::ProgramMceStripe, 3, 0 },
        TestCommand { CommandType::WaitForCounter, 0, 11 },
        TestCommand { CommandType::StartMceStripe, 3, 0 },
        TestCommand { CommandType::ProgramMceStripe, 3, 0 },
        TestCommand { CommandType::WaitForCounter, 0, 13 },
        TestCommand { CommandType::StartMceStripe, 3, 0 },
        TestCommand { CommandType::ProgramMceStripe, 3, 0 },
        TestCommand { CommandType::WaitForCounter, 0, 14 },
        TestCommand { CommandType::StartMceStripe, 3, 0 },
        TestCommand { CommandType::ProgramMceStripe, 3, 0 },
        TestCommand { CommandType::WaitForCounter, 0, 18 },
        TestCommand { CommandType::StartMceStripe, 3, 0 },
        TestCommand { CommandType::ProgramMceStripe, 3, 0 },
        TestCommand { CommandType::WaitForCounter, 0, 20 },
        TestCommand { CommandType::StartMceStripe, 3, 0 },
        TestCommand { CommandType::ProgramMceStripe, 3, 0 },
        TestCommand { CommandType::WaitForCounter, 0, 21 },
        TestCommand { CommandType::StartMceStripe, 3, 0 }
        // clang-format on
    };

    const std::vector<TestCommand> expectedPleCommands{
        // clang-format off
        TestCommand { CommandType::WaitForCounter, 0, 22 },
        TestCommand { CommandType::LoadPleCodeIntoPleSram, 4, 0 },
        TestCommand { CommandType::WaitForCounter, 4, 1 },
        TestCommand { CommandType::WaitForCounter, 2, 1 },
        TestCommand { CommandType::StartPleStripe, 4, 0 },
        // clang-format on
    };

    ethosn::support_library::DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});

    Scheduler scheduler(complexSingleLayerCmdStream, caps, debuggingContext);
    scheduler.Schedule();

    CompareCommandArrays(scheduler.GetDmaRdCommands(), expectedDmaRdCommands);
    CompareCommandArrays(scheduler.GetDmaWrCommands(), expectedDmaWrCommands);
    CompareCommandArrays(scheduler.GetMceCommands(), expectedMceCommands);
    CompareCommandArrays(scheduler.GetPleCommands(), expectedPleCommands);
}
