//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/cascading/Plan.hpp"

#include <catch.hpp>

using namespace ethosn::support_library;

TEST_CASE("OpGraph Contains")
{
    OpGraph graph;

    Buffer buffer;
    MceOp op;

    // Initially these are not in the graph
    REQUIRE(!graph.Contains(&buffer));
    REQUIRE(!graph.Contains(&op));

    // Add them and check that they are reported as contained
    graph.AddBuffer(&buffer);
    graph.AddOp(&op);
    REQUIRE(graph.Contains(&buffer));
    REQUIRE(graph.Contains(&op));
}

/// Checks GetProducer correctly returns zero/one producers.
TEST_CASE("OpGraph GetProducer")
{
    OpGraph graph;

    // Start with just a single buffer and nothing that produces it
    Buffer buffer;
    graph.AddBuffer(&buffer);
    REQUIRE(graph.GetProducer(&buffer) == nullptr);

    // Add an Op as a producer
    MceOp op;
    graph.AddOp(&op);
    graph.SetProducer(&buffer, &op);
    REQUIRE(graph.GetProducer(&buffer) == &op);
}

/// Checks GetConsumers correctly returns zero or many consumers, along with their input indices
TEST_CASE("OpGraph GetConsumers")
{
    OpGraph graph;

    // Start with just a single buffer and nothing that consumes it
    Buffer buffer;
    graph.AddBuffer(&buffer);
    REQUIRE(graph.GetConsumers(&buffer) == std::vector<std::pair<Op*, uint32_t>>{});

    // Add an Op as a consumer
    MceOp op1;
    graph.AddOp(&op1);
    graph.AddConsumer(&buffer, &op1, 0);
    REQUIRE(graph.GetConsumers(&buffer) == std::vector<std::pair<Op*, uint32_t>>{ { &op1, 0 } });

    // Add another Op as a consumer, but using its 2nd input.
    // Note we must first connect the 1st input of the op to something else
    MceOp op2;
    graph.AddOp(&op2);
    graph.AddConsumer(&buffer, &op2, 0);
    graph.AddConsumer(&buffer, &op2, 1);
    REQUIRE(graph.GetConsumers(&buffer) ==
            std::vector<std::pair<Op*, uint32_t>>{ { &op1, 0 }, { &op2, 0 }, { &op2, 1 } });
}

/// Checks GetInputs correctly returns zero or many inputs, along with their input indices
TEST_CASE("OpGraph GetInputs")
{
    OpGraph graph;

    // Start with just a single op that has no inputs
    MceOp op;
    graph.AddOp(&op);
    REQUIRE(graph.GetInputs(&op) == std::vector<Buffer*>{});

    // Add a Buffer as the first input
    Buffer buffer1;
    graph.AddBuffer(&buffer1);
    graph.AddConsumer(&buffer1, &op, 0);
    REQUIRE(graph.GetInputs(&op) == std::vector<Buffer*>{ &buffer1 });

    // Add a Buffer as the second input
    Buffer buffer2;
    graph.AddBuffer(&buffer2);
    graph.AddConsumer(&buffer2, &op, 1);
    REQUIRE(graph.GetInputs(&op) == std::vector<Buffer*>{ &buffer1, &buffer2 });
}

/// Checks GetOutput correctly returns zero or one output
TEST_CASE("OpGraph GetOutput")
{
    OpGraph graph;

    // Start with just a single op that has no output
    MceOp op;
    graph.AddOp(&op);
    REQUIRE(graph.GetOutput(&op) == nullptr);

    // Add a Buffer as the output
    Buffer buffer;
    graph.AddBuffer(&buffer);
    graph.SetProducer(&buffer, &op);
    REQUIRE(graph.GetOutput(&op) == &buffer);
}

/// Adds a single Op to the graph, checking both the successful and unsuccessful cases
TEST_CASE("OpGraph AddOp")
{
    OpGraph graph;
    MceOp op;

    // Add the op and check it has been added
    graph.AddOp(&op);
    REQUIRE(graph.GetOps() == std::vector<Op*>{ &op });

    // Attempt to add it again and check that this failed
    REQUIRE_THROWS(graph.AddOp(&op));
}

/// Adds a single Buffer to the graph, checking both the successful and unsuccessful cases
TEST_CASE("OpGraph AddBuffer")
{
    OpGraph graph;
    Buffer buffer;

    // Add the buffer and check it has been added
    graph.AddBuffer(&buffer);
    REQUIRE(graph.GetBuffers() == std::vector<Buffer*>{ &buffer });

    // Attempt to add it again and check that this failed
    REQUIRE_THROWS(graph.AddBuffer(&buffer));
}

/// Checks SetProducer correctly validates
TEST_CASE("OpGraph SetProducer")
{
    // Try calling with an Op that isn't part of the graph
    {
        OpGraph graph;
        MceOp op;
        Buffer buffer;
        graph.AddBuffer(&buffer);
        REQUIRE_THROWS(graph.SetProducer(&buffer, &op));
    }

    // Try calling with a Buffer that isn't part of the graph
    {
        OpGraph graph;
        MceOp op;
        graph.AddOp(&op);
        Buffer buffer;
        REQUIRE_THROWS(graph.SetProducer(&buffer, &op));
    }

    // Try setting the producer for a buffer that already has a producer
    {
        OpGraph graph;
        MceOp op1;
        graph.AddOp(&op1);
        Buffer buffer;
        graph.AddBuffer(&buffer);
        graph.SetProducer(&buffer, &op1);

        MceOp op2;
        graph.AddOp(&op2);
        REQUIRE_THROWS(graph.SetProducer(&buffer, &op2));
    }

    // Successful case
    {
        OpGraph graph;
        MceOp op1;
        graph.AddOp(&op1);
        Buffer buffer;
        graph.AddBuffer(&buffer);
        graph.SetProducer(&buffer, &op1);

        REQUIRE(graph.GetProducer(&buffer) == &op1);
    }
}

/// Checks ClearProducer correctly validates and does the right thing
TEST_CASE("OpGraph ClearProducer")
{
    SECTION("Try calling with a nullptr")
    {
        OpGraph graph;
        REQUIRE_THROWS(graph.ClearProducer(nullptr));
    }

    SECTION("Try calling with a Buffer that isn't part of the graph")
    {
        OpGraph graph;
        Buffer b;
        REQUIRE_THROWS(graph.ClearProducer(&b));
    }

    SECTION("Clear the producer for a buffer that doesn't already have one. This should be a no-op")
    {
        OpGraph graph;
        Buffer buffer;
        graph.AddBuffer(&buffer);
        REQUIRE_NOTHROW(graph.ClearProducer(&buffer));
        REQUIRE(graph.GetProducer(&buffer) == nullptr);
    }

    SECTION("Clear the producer for a buffer that already has one")
    {
        OpGraph graph;
        MceOp op1;
        graph.AddOp(&op1);
        Buffer buffer;
        graph.AddBuffer(&buffer);
        graph.SetProducer(&buffer, &op1);

        graph.ClearProducer(&buffer);
        REQUIRE(graph.GetProducer(&buffer) == nullptr);
        REQUIRE(graph.GetOutput(&op1) == nullptr);
    }
}

/// Checks AddConsumer correctly validates and deals with multiple input slots.
TEST_CASE("OpGraph AddConsumer")
{
    // Try calling with an Op that isn't part of the graph
    {
        OpGraph graph;
        MceOp op;
        Buffer buffer;
        graph.AddBuffer(&buffer);
        REQUIRE_THROWS(graph.AddConsumer(&buffer, &op, 0));
    }

    // Try calling with a Buffer that isn't part of the graph
    {
        OpGraph graph;
        MceOp op;
        graph.AddOp(&op);
        Buffer buffer;
        REQUIRE_THROWS(graph.AddConsumer(&buffer, &op, 0));
    }

    // Try adding an op as a consumer that is already linked to another buffer
    {
        OpGraph graph;
        MceOp op1;
        graph.AddOp(&op1);
        Buffer buffer1;
        graph.AddBuffer(&buffer1);
        graph.AddConsumer(&buffer1, &op1, 0);

        Buffer buffer2;
        graph.AddBuffer(&buffer2);
        REQUIRE_THROWS(graph.AddConsumer(&buffer2, &op1, 0));
    }

    // Connect a second input slot of an Op where the lower-numbered slots is already connected
    // This requires the vector of inputs to be appended to
    {
        OpGraph graph;
        MceOp op1;
        graph.AddOp(&op1);
        Buffer buffer1;
        graph.AddBuffer(&buffer1);
        graph.AddConsumer(&buffer1, &op1, 0);
        graph.AddConsumer(&buffer1, &op1, 1);

        REQUIRE(graph.GetInputs(&op1) == std::vector<Buffer*>{ &buffer1, &buffer1 });
    }

    // Connect a higher-numbered input slot of an Op where the lower-numbered slots are not yet connected
    // This is an error, as the earlier-numbered slots would be unconnected.
    {
        OpGraph graph;
        MceOp op1;
        graph.AddOp(&op1);
        Buffer buffer1;
        graph.AddBuffer(&buffer1);
        REQUIRE_THROWS(graph.AddConsumer(&buffer1, &op1, 2));
    }
}

TEST_CASE("Get size in bytes helpers")
{
    Buffer* buffer;
    Buffer tempBuffer;

    Plan planASram;
    tempBuffer = Buffer(Lifetime::Atomic, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(),
                        TensorShape{ 1, 8, 8, 16 }, TraversalOrder::Xyz, 4 * 8 * 8 * 16, QuantizationInfo());
    buffer     = planASram.m_OpGraph.AddBuffer(std::make_unique<Buffer>(std::move(tempBuffer)));
    planASram.m_OutputMappings[buffer] = nullptr;

    REQUIRE(GetTotSizeInBytes(planASram).m_Tot == 4 * 8 * 8 * 16);
    REQUIRE(GetTotSizeInBytes(planASram).m_TotAtomic == 4 * 8 * 8 * 16);
    REQUIRE(GetInputsSizeInBytes(planASram).m_Tot == 0);
    REQUIRE(GetInputsSizeInBytes(planASram).m_TotAtomic == 0);

    Plan planBSram;
    tempBuffer = Buffer(Lifetime::Cascade, Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(),
                        TensorShape{ 1, 8, 8, 8 }, TraversalOrder::Xyz, 4 * 8 * 8 * 8, QuantizationInfo());
    buffer     = planBSram.m_OpGraph.AddBuffer(std::make_unique<Buffer>(std::move(tempBuffer)));
    planBSram.m_InputMappings[buffer] = nullptr;

    REQUIRE(GetTotSizeInBytes(planBSram).m_Tot == 4 * 8 * 8 * 8);
    REQUIRE(GetTotSizeInBytes(planBSram).m_TotAtomic == 0);
    REQUIRE(GetInputsSizeInBytes(planBSram).m_Tot == 4 * 8 * 8 * 8);
    REQUIRE(GetInputsSizeInBytes(planBSram).m_TotAtomic == 0);
}
