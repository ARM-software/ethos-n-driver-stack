//
// Copyright © 2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../../include/ethosn_command_stream/CommandStreamBuffer.hpp"

#include <catch.hpp>

using namespace ethosn::command_stream;

TEST_CASE("Cascading/CommandStream")
{
    CommandStreamBuffer csbuffer;

    csbuffer.EmplaceBack(McePle{});

    AddCascade(csbuffer,
               {
                   cascading::Agent{ 0, cascading::IfmS{} },
                   cascading::Agent{ 0, cascading::WgtS{} },
                   cascading::Agent{ 0, cascading::MceS{} },
                   cascading::Agent{ 0, cascading::PleS{} },
                   cascading::Agent{ 0, cascading::OfmS{} },
               },
               { { cascading::Command{ cascading::CommandType::LoadIfmStripe, 0, 0, 0 } } },
               { { cascading::Command{ cascading::CommandType::StoreOfmStripe, 2, 3, 0 } } },
               { { cascading::Command{ cascading::CommandType::StartMceStripe, 0, 0, 0 } } },
               { { cascading::Command{ cascading::CommandType::WaitForAgent, 0, 0, 0 } } },
               { { cascading::DmaExtraData{} } }, { { cascading::ProgramMceExtraData{} } },
               { { cascading::StartMceExtraData{} } }, { { cascading::StartPleExtraData{} } });

    csbuffer.EmplaceBack(Fence{});
    csbuffer.EmplaceBack(McePle{});
    csbuffer.EmplaceBack(Convert{});
    csbuffer.EmplaceBack(SpaceToDepth{});

    CommandStream cstream(&*csbuffer.begin(), &*csbuffer.end());
    REQUIRE(cstream.IsValid());

    auto it = cstream.begin();

    REQUIRE(it->GetCommand<Opcode::OPERATION_MCE_PLE>() != nullptr);

    ++it;

    {
        const auto command = it->GetCommand<Opcode::CASCADE>();

        REQUIRE(command != nullptr);

        const auto cascadeBegin = static_cast<const cascading::Agent*>(static_cast<const void*>(command + 1));

        std::span<const cascading::Agent> agents{ cascadeBegin, 5 };

        CHECK(agents[0].data.type == cascading::AgentType::IFM_STREAMER);
        CHECK(agents[1].data.type == cascading::AgentType::WGT_STREAMER);
        CHECK(agents[2].data.type == cascading::AgentType::MCE_SCHEDULER);
        CHECK(agents[3].data.type == cascading::AgentType::PLE_SCHEDULER);
        CHECK(agents[4].data.type == cascading::AgentType::OFM_STREAMER);
    }

    ++it;

    REQUIRE(it->GetCommand<Opcode::FENCE>() != nullptr);

    ++it;

    REQUIRE(it->GetCommand<Opcode::OPERATION_MCE_PLE>() != nullptr);

    ++it;

    REQUIRE(it->GetCommand<Opcode::OPERATION_CONVERT>() != nullptr);

    ++it;

    REQUIRE(it->GetCommand<Opcode::OPERATION_SPACE_TO_DEPTH>() != nullptr);
}
