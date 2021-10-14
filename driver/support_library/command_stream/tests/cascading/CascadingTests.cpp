//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../../include/ethosn_command_stream/CommandStreamBuffer.hpp"

#include <catch.hpp>

using namespace ethosn::command_stream;

TEST_CASE("Cascading/CommandStream")
{
    CommandStreamBuffer csbuffer;

    csbuffer.EmplaceBack(McePle{});
    csbuffer.EmplaceBack(Cascade{ 5U });
    csbuffer.EmplaceBack(cascading::Agent{ cascading::IfmS{}, {} });
    csbuffer.EmplaceBack(cascading::Agent{ cascading::WgtS{}, {} });
    csbuffer.EmplaceBack(cascading::Agent{ cascading::MceS{}, {} });
    csbuffer.EmplaceBack(cascading::Agent{ cascading::PleS{}, {} });
    csbuffer.EmplaceBack(cascading::Agent{ cascading::OfmS{}, {} });
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

        cascading::CommandStream cascade{ cascadeBegin, command->m_Data().m_NumAgents() };

        CHECK(cascade[0].data.type == cascading::AgentType::IFM_STREAMER);
        CHECK(cascade[1].data.type == cascading::AgentType::WGT_STREAMER);
        CHECK(cascade[2].data.type == cascading::AgentType::MCE_SCHEDULER);
        CHECK(cascade[3].data.type == cascading::AgentType::PLE_SCHEDULER);
        CHECK(cascade[4].data.type == cascading::AgentType::OFM_STREAMER);
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
