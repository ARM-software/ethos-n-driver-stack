//
// Copyright © 2020-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/Graph.hpp"
#include "../src/GraphNodes.hpp"
#include "../src/Utils.hpp"
#include "../src/cascading/Cascading.hpp"
#include "../src/cascading/PartUtils.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>

using namespace ethosn::support_library;
using namespace ethosn::support_library::utils;

TEST_CASE("GraphOfParts simple linear")
{
    GraphOfParts graph;

    // p1 -> p2 -> p3

    auto p1 = std::make_unique<MockPart>(1);
    auto p2 = std::make_unique<MockPart>(2);
    auto p3 = std::make_unique<MockPart>(3);
    graph.AddPart(std::move(p1));
    graph.AddPart(std::move(p2));
    graph.AddPart(std::move(p3));

    // connect up the parts
    graph.AddConnection(PartInputSlot{ 2, 0 }, PartOutputSlot{ 1, 0 });
    graph.AddConnection(PartInputSlot{ 3, 0 }, PartOutputSlot{ 2, 0 });

    auto sourcep1 = graph.GetSourceParts(1);
    REQUIRE(sourcep1.size() == 0);
    auto sourcep2 = graph.GetSourceParts(2);
    REQUIRE(sourcep2.size() == 1);
    REQUIRE(sourcep2.at(0) == PartOutputSlot{ 1, 0 });
    auto sourcep3 = graph.GetSourceParts(3);
    REQUIRE(sourcep3.size() == 1);
    REQUIRE(sourcep3.at(0) == PartOutputSlot{ 2, 0 });

    auto destp1 = graph.GetDestinationParts(1);
    REQUIRE(destp1.size() == 1);
    REQUIRE(destp1.at(0) == PartInputSlot{ 2, 0 });
    auto destp2 = graph.GetDestinationParts(2);
    REQUIRE(destp2.size() == 1);
    REQUIRE(destp2.at(0) == PartInputSlot{ 3, 0 });
    auto destp3 = graph.GetDestinationParts(3);
    REQUIRE(destp3.size() == 0);
}

TEST_CASE("GraphOfParts multiple input slots for one output slot")
{
    GraphOfParts graph;

    // p1 "0th" output connects to p2 and p3.
    //
    // p1 0->0 p2
    //    0->0 p3

    auto p1   = std::make_unique<MockPart>(1);
    auto p2   = std::make_unique<MockPart>(2);
    auto p3   = std::make_unique<MockPart>(3);
    auto p1Id = p1->GetPartId();
    auto p2Id = p2->GetPartId();
    auto p3Id = p3->GetPartId();
    graph.AddPart(std::move(p1));
    graph.AddPart(std::move(p2));
    graph.AddPart(std::move(p3));

    PartOutputSlot p1OutputSlot = PartOutputSlot{ p1Id, 0 };
    PartInputSlot p2InputSlot   = PartInputSlot{ p2Id, 0 };
    PartInputSlot p3InputSlot   = PartInputSlot{ p3Id, 0 };

    // connect up the parts
    graph.AddConnection(p2InputSlot, p1OutputSlot);
    graph.AddConnection(p3InputSlot, p1OutputSlot);

    auto inputSlots = graph.GetConnectedInputSlots(p1OutputSlot);
    REQUIRE(inputSlots.size() == 2);
    REQUIRE(utils::Find(inputSlots, p2InputSlot).first);
    REQUIRE(utils::Find(inputSlots, p3InputSlot).first);

    {
        auto outputSlot = graph.GetConnectedOutputSlot(p2InputSlot);
        REQUIRE(outputSlot.has_value());
        REQUIRE(outputSlot.value() == p1OutputSlot);
    }
    {
        auto outputSlot = graph.GetConnectedOutputSlot(p3InputSlot);
        REQUIRE(outputSlot.has_value());
        REQUIRE(outputSlot.value() == p1OutputSlot);
    }
}

TEST_CASE("GraphOfParts GetPartInputs/Outputs")
{
    GraphOfParts graph;

    // p1 "0th" output connects to p2 and p3
    // p1 "1st" output connects to p3's 0th and 1st input
    //
    // p1 0->0 p2
    //    0->0 p3
    //    1->1 p3

    auto p1   = std::make_unique<MockPart>(1);
    auto p2   = std::make_unique<MockPart>(2);
    auto p3   = std::make_unique<MockPart>(3);
    auto p1Id = p1->GetPartId();
    auto p2Id = p2->GetPartId();
    auto p3Id = p3->GetPartId();
    graph.AddPart(std::move(p1));
    graph.AddPart(std::move(p2));
    graph.AddPart(std::move(p3));

    PartOutputSlot p1OutputSlot0 = PartOutputSlot{ p1Id, 0 };
    PartOutputSlot p1OutputSlot1 = PartOutputSlot{ p1Id, 1 };
    PartInputSlot p2InputSlot    = PartInputSlot{ p2Id, 0 };
    PartInputSlot p3InputSlot0   = PartInputSlot{ p3Id, 0 };
    PartInputSlot p3InputSlot1   = PartInputSlot{ p3Id, 1 };

    // connect up the parts
    graph.AddConnection(p2InputSlot, p1OutputSlot0);
    graph.AddConnection(p3InputSlot0, p1OutputSlot0);
    graph.AddConnection(p3InputSlot1, p1OutputSlot1);

    {
        auto inputSlots = graph.GetPartInputs(p1Id);
        REQUIRE(inputSlots.size() == 0);
        auto outputSlots = graph.GetPartOutputs(p1Id);
        REQUIRE(outputSlots.size() == 2);
    }
    {
        auto inputSlots = graph.GetPartInputs(p2Id);
        REQUIRE(inputSlots.size() == 1);
        REQUIRE(utils::Find(inputSlots, p2InputSlot).first);

        auto outputSlots = graph.GetPartOutputs(p2Id);
        REQUIRE(outputSlots.size() == 0);
    }
    {
        auto inputSlots = graph.GetPartInputs(p3Id);
        REQUIRE(inputSlots.size() == 2);
        REQUIRE(utils::Find(inputSlots, p3InputSlot0).first);
        REQUIRE(utils::Find(inputSlots, p3InputSlot1).first);

        auto outputSlots = graph.GetPartOutputs(p3Id);
        REQUIRE(outputSlots.size() == 0);
    }
}

TEST_CASE("GraphOfParts/MergeChannelSelectors/CantMergeSharedOutput")
{
    GraphOfParts g;

    // 1 -> 2 (cs) -> 3
    //   \       \_
    //    4         5

    auto part1                                = std::make_unique<MockPart>(1);
    part1->m_CanMergeWithChannelSelectorAfter = true;
    g.AddPart(std::move(part1));

    auto part2                      = std::make_unique<MockPart>(2);
    part2->m_ChannelSelectorWeights = ConstTensorData(nullptr, TensorShape());
    g.AddPart(std::move(part2));

    auto part3                                 = std::make_unique<MockPart>(3);
    part3->m_CanMergeWithChannelSelectorBefore = true;
    g.AddPart(std::move(part3));

    auto part4 = std::make_unique<MockPart>(4);
    g.AddPart(std::move(part4));

    auto part5 = std::make_unique<MockPart>(5);
    g.AddPart(std::move(part5));

    g.AddConnection(PartInputSlot{ 2, 0 }, PartOutputSlot{ 1, 0 });
    g.AddConnection(PartInputSlot{ 4, 0 }, PartOutputSlot{ 1, 0 });
    g.AddConnection(PartInputSlot{ 3, 0 }, PartOutputSlot{ 2, 0 });
    g.AddConnection(PartInputSlot{ 5, 0 }, PartOutputSlot{ 2, 0 });

    g.MergeChannelSelectors();

    // No optimisation possible on either side, due to shared outputs
    CHECK(g.GetParts().size() == 5);
}

TEST_CASE("GraphOfParts/MergeChannelSelectors/CantMergeWithUnsupportedParts")
{
    GraphOfParts g;

    // 1 -> 2 (cs) -> 3

    auto part1                                = std::make_unique<MockPart>(1);
    part1->m_CanMergeWithChannelSelectorAfter = false;
    g.AddPart(std::move(part1));

    auto part2                      = std::make_unique<MockPart>(2);
    part2->m_ChannelSelectorWeights = ConstTensorData(nullptr, TensorShape());
    g.AddPart(std::move(part2));

    auto part3                                 = std::make_unique<MockPart>(3);
    part3->m_CanMergeWithChannelSelectorBefore = false;
    g.AddPart(std::move(part3));

    g.AddConnection(PartInputSlot{ 2, 0 }, PartOutputSlot{ 1, 0 });
    g.AddConnection(PartInputSlot{ 3, 0 }, PartOutputSlot{ 2, 0 });

    g.MergeChannelSelectors();

    // No optimisation possible on either side, as neither neighbouring part supports merging
    CHECK(g.GetParts().size() == 3);
}

TEST_CASE("GraphOfParts/MergeChannelSelectors/MergeBefore")
{
    GraphOfParts g;

    // 1 -> 2 (cs) -> 3

    auto part1 = std::make_unique<MockPart>(1);
    part1->AddOperationId(1);
    part1->m_CanMergeWithChannelSelectorAfter = true;
    g.AddPart(std::move(part1));

    auto part2 = std::make_unique<MockPart>(2);
    part2->AddOperationId(2);
    part2->m_ChannelSelectorWeights = ConstTensorData(nullptr, TensorShape());
    g.AddPart(std::move(part2));

    auto part3 = std::make_unique<MockPart>(3);
    part3->AddOperationId(3);
    part3->m_CanMergeWithChannelSelectorBefore = false;
    g.AddPart(std::move(part3));

    g.AddConnection(PartInputSlot{ 2, 0 }, PartOutputSlot{ 1, 0 });
    g.AddConnection(PartInputSlot{ 3, 0 }, PartOutputSlot{ 2, 0 });

    g.MergeChannelSelectors();

    // 2 should have been merged with 1
    CHECK(g.GetParts().size() == 2);
    CHECK(static_cast<MockPart*>(g.GetParts().at(1).get())->GetOperationIds() == std::set<uint32_t>{ 1, 2 });

    CHECK(g.GetAllConnections().size() == 1);
    CHECK(g.GetAllConnections().at(PartInputSlot{ 3, 0 }) == PartOutputSlot{ 1, 0 });
}

TEST_CASE("GraphOfParts/MergeChannelSelectors/MergeAfter")
{
    GraphOfParts g;

    // 1 -> 2 (cs) -> 3

    auto part1 = std::make_unique<MockPart>(1);
    part1->AddOperationId(1);
    part1->m_CanMergeWithChannelSelectorAfter = false;
    g.AddPart(std::move(part1));

    auto part2 = std::make_unique<MockPart>(2);
    part2->AddOperationId(2);
    part2->m_ChannelSelectorWeights = ConstTensorData(nullptr, TensorShape());
    g.AddPart(std::move(part2));

    auto part3 = std::make_unique<MockPart>(3);
    part3->AddOperationId(3);
    part3->m_CanMergeWithChannelSelectorBefore = true;
    g.AddPart(std::move(part3));

    g.AddConnection(PartInputSlot{ 2, 0 }, PartOutputSlot{ 1, 0 });
    g.AddConnection(PartInputSlot{ 3, 0 }, PartOutputSlot{ 2, 0 });

    g.MergeChannelSelectors();

    // 2 should have been merged with 3
    CHECK(g.GetParts().size() == 2);
    CHECK(static_cast<MockPart*>(g.GetParts().at(3).get())->GetOperationIds() == std::set<uint32_t>{ 2, 3 });

    CHECK(g.GetAllConnections().size() == 1);
    CHECK(g.GetAllConnections().at(PartInputSlot{ 3, 0 }) == PartOutputSlot{ 1, 0 });
}

TEST_CASE("GraphOfParts/SortAndCompact")
{
    GraphOfParts g;

    // 3 -> 1 -> 5

    auto part3 = std::make_unique<MockPart>(3);
    part3->AddOperationId(3);
    part3->m_DebugTag = "Part 3";
    g.AddPart(std::move(part3));

    auto part1 = std::make_unique<MockPart>(1);
    part1->AddOperationId(1);
    part1->m_DebugTag = "Part 1";
    g.AddPart(std::move(part1));

    auto part5 = std::make_unique<MockPart>(5);
    part5->AddOperationId(5);
    part5->m_DebugTag = "Part 5";
    g.AddPart(std::move(part5));

    g.AddConnection(PartInputSlot{ 1, 0 }, PartOutputSlot{ 3, 0 });
    g.AddConnection(PartInputSlot{ 5, 0 }, PartOutputSlot{ 1, 0 });

    g.SortAndCompact();

    // 3 is the first in the graph, so becomes Part 0, 1 stays the same and 5 becomes 2
    CHECK(g.GetParts().size() == 3);

    CHECK(g.GetParts().at(0)->GetPartId() == 0);
    // Debug tag is renamed, so it's consistent with the Part ID
    CHECK(g.GetParts().at(0)->m_DebugTag == "Part 0");
    // But the other data (e.g. operation IDs remains the same)
    CHECK(static_cast<MockPart&>(*g.GetParts().at(0)).GetOperationIds() == std::set<uint32_t>{ 3 });

    CHECK(g.GetParts().at(1)->GetPartId() == 1);
    CHECK(g.GetParts().at(1)->m_DebugTag == "Part 1");
    CHECK(static_cast<MockPart&>(*g.GetParts().at(1)).GetOperationIds() == std::set<uint32_t>{ 1 });

    CHECK(g.GetParts().at(2)->GetPartId() == 2);
    CHECK(g.GetParts().at(2)->m_DebugTag == "Part 2");
    CHECK(static_cast<MockPart&>(*g.GetParts().at(2)).GetOperationIds() == std::set<uint32_t>{ 5 });

    CHECK(g.GetAllConnections().size() == 2);
    CHECK(g.GetAllConnections().at(PartInputSlot{ 1, 0 }) == PartOutputSlot{ 0, 0 });
    CHECK(g.GetAllConnections().at(PartInputSlot{ 2, 0 }) == PartOutputSlot{ 1, 0 });
}
