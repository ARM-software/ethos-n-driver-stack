//
// Copyright © 2020-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/Graph.hpp"
#include "../src/GraphNodes.hpp"
#include "../src/cascading/Cascading.hpp"
#include "../src/cascading/PartUtils.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>

using namespace ethosn::support_library;

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
        REQUIRE(outputSlots.size() == 3);
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
