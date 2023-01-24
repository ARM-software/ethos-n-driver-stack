//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/cascading/OpGraph.hpp"

#include <catch.hpp>

#include <fstream>

using namespace ethosn::support_library;
using namespace ethosn::command_stream::cascading;

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

/// Checks GetSingleProducer correctly returns zero/one producers and
/// GetProducers correctly returns zero/one/many producers.
TEST_CASE("OpGraph GetSingleProducer and GetProducers")
{
    OpGraph graph;

    // Start with just a single buffer and nothing that produces it
    Buffer buffer;
    graph.AddBuffer(&buffer);
    REQUIRE(graph.GetSingleProducer(&buffer) == nullptr);
    REQUIRE(graph.GetProducers(&buffer) == std::vector<Op*>{});

    // Add an Op as a producer
    MceOp op;
    graph.AddOp(&op);
    graph.SetProducer(&buffer, &op);
    REQUIRE(graph.GetSingleProducer(&buffer) == &op);
    REQUIRE(graph.GetProducers(&buffer) == std::vector<Op*>{ &op });

    // Add a second Op as a producer
    MceOp op2;
    graph.AddOp(&op2);
    graph.AddProducer(&buffer, &op2);
    REQUIRE_THROWS(graph.GetSingleProducer(&buffer));
    REQUIRE(graph.GetProducers(&buffer) == std::vector<Op*>{ &op, &op2 });
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

    // Try adding a producer that is already a producer
    {
        OpGraph graph;
        MceOp op1;
        graph.AddOp(&op1);
        Buffer buffer;
        graph.AddBuffer(&buffer);
        graph.SetProducer(&buffer, &op1);

        REQUIRE_THROWS(graph.SetProducer(&buffer, &op1));
    }

    // Successful case
    {
        OpGraph graph;
        MceOp op1;
        graph.AddOp(&op1);
        Buffer buffer;
        graph.AddBuffer(&buffer);
        graph.SetProducer(&buffer, &op1);

        REQUIRE(graph.GetSingleProducer(&buffer) == &op1);
    }
}

/// Checks  AddProducer correctly validates
TEST_CASE("OpGraph AddProducer")
{
    // Try calling with an Op that isn't part of the graph
    {
        OpGraph graph;
        MceOp op;
        Buffer buffer;
        graph.AddBuffer(&buffer);
        REQUIRE_THROWS(graph.AddProducer(&buffer, &op));
    }

    // Try calling with a Buffer that isn't part of the graph
    {
        OpGraph graph;
        MceOp op;
        graph.AddOp(&op);
        Buffer buffer;
        REQUIRE_THROWS(graph.AddProducer(&buffer, &op));
    }

    // Try adding a producer for a buffer that already has a producer
    {
        OpGraph graph;
        MceOp op1;
        graph.AddOp(&op1);
        Buffer buffer;
        graph.AddBuffer(&buffer);
        graph.SetProducer(&buffer, &op1);

        MceOp op2;
        graph.AddOp(&op2);
        graph.AddProducer(&buffer, &op2);
        REQUIRE(graph.GetProducers(&buffer) == std::vector<Op*>{ &op1, &op2 });
    }

    // Try adding a producer that is already a producer
    {
        OpGraph graph;
        MceOp op1;
        graph.AddOp(&op1);
        Buffer buffer;
        graph.AddBuffer(&buffer);
        graph.SetProducer(&buffer, &op1);

        REQUIRE_THROWS(graph.AddProducer(&buffer, &op1));
    }

    // Successful case
    {
        OpGraph graph;
        MceOp op1;
        graph.AddOp(&op1);
        Buffer buffer;
        graph.AddBuffer(&buffer);
        graph.AddProducer(&buffer, &op1);

        REQUIRE(graph.GetSingleProducer(&buffer) == &op1);
    }
}

/// Checks RemoveProducer correctly validates and does the right thing
TEST_CASE("OpGraph RemoveProducer")
{
    SECTION("Try calling with nullptr Buffer")
    {
        OpGraph graph;
        MceOp o;
        graph.AddOp(&o);
        REQUIRE_THROWS(graph.RemoveProducer(nullptr, &o));
    }
    SECTION("Try calling with nullptr Op")
    {
        OpGraph graph;
        Buffer b;
        graph.AddBuffer(&b);
        REQUIRE_THROWS(graph.RemoveProducer(&b, nullptr));
    }

    SECTION("Try calling with a Buffer that isn't part of the graph")
    {
        OpGraph graph;
        MceOp o;
        graph.AddOp(&o);
        Buffer b;
        REQUIRE_THROWS(graph.RemoveProducer(&b, &o));
    }
    SECTION("Try calling with an Op that isn't part of the graph")
    {
        OpGraph graph;
        Buffer b;
        graph.AddBuffer(&b);
        MceOp o;
        REQUIRE_THROWS(graph.RemoveProducer(&b, &o));
    }

    SECTION("Try calling with a Buffer that has no producers")
    {
        OpGraph graph;
        Buffer b;
        graph.AddBuffer(&b);
        MceOp o;
        graph.AddOp(&o);
        REQUIRE_THROWS(graph.RemoveProducer(&b, &o));
    }
    SECTION("Try calling with an Op that isn't a producer of the Buffer (but the Buffer has other producers)")
    {
        OpGraph graph;
        Buffer b;
        graph.AddBuffer(&b);
        MceOp o1;
        graph.AddOp(&o1);
        MceOp o2;
        graph.AddOp(&o2);
        graph.SetProducer(&b, &o1);

        REQUIRE_THROWS(graph.RemoveProducer(&b, &o2));
    }

    SECTION("Remove a producer from a buffer that has only one")
    {
        OpGraph graph;
        MceOp op1;
        graph.AddOp(&op1);
        Buffer buffer;
        graph.AddBuffer(&buffer);
        graph.SetProducer(&buffer, &op1);

        graph.RemoveProducer(&buffer, &op1);
        REQUIRE(graph.GetProducers(&buffer).size() == 0);
        REQUIRE(graph.GetOutput(&op1) == nullptr);
    }

    SECTION("Remove a producer from a buffer that has two")
    {
        OpGraph graph;
        MceOp op1;
        graph.AddOp(&op1);
        MceOp op2;
        graph.AddOp(&op2);
        Buffer buffer;
        graph.AddBuffer(&buffer);
        graph.AddProducer(&buffer, &op1);
        graph.AddProducer(&buffer, &op2);

        graph.RemoveProducer(&buffer, &op1);
        REQUIRE(graph.GetProducers(&buffer) == OpGraph::OpList{ &op2 });
        REQUIRE(graph.GetOutput(&op1) == nullptr);
        REQUIRE(graph.GetOutput(&op2) == &buffer);
    }
}

/// Checks ClearProducers correctly validates and does the right thing
TEST_CASE("OpGraph ClearProducers")
{
    SECTION("Try calling with a nullptr")
    {
        OpGraph graph;
        REQUIRE_THROWS(graph.ClearProducers(nullptr));
    }

    SECTION("Try calling with a Buffer that isn't part of the graph")
    {
        OpGraph graph;
        Buffer b;
        REQUIRE_THROWS(graph.ClearProducers(&b));
    }

    SECTION("Clear the producer for a buffer that doesn't already have one. This should be a no-op")
    {
        OpGraph graph;
        Buffer buffer;
        graph.AddBuffer(&buffer);
        REQUIRE_NOTHROW(graph.ClearProducers(&buffer));
        REQUIRE(graph.GetSingleProducer(&buffer) == nullptr);
    }

    SECTION("Clear the producer for a buffer that already has one")
    {
        OpGraph graph;
        MceOp op1;
        graph.AddOp(&op1);
        Buffer buffer;
        graph.AddBuffer(&buffer);
        graph.SetProducer(&buffer, &op1);

        graph.ClearProducers(&buffer);
        REQUIRE(graph.GetSingleProducer(&buffer) == nullptr);
        REQUIRE(graph.GetOutput(&op1) == nullptr);
    }

    SECTION("Clear the producers for a buffer that has two")
    {
        OpGraph graph;
        MceOp op1;
        graph.AddOp(&op1);
        MceOp op2;
        graph.AddOp(&op2);
        Buffer buffer;
        graph.AddBuffer(&buffer);
        graph.AddProducer(&buffer, &op1);
        graph.AddProducer(&buffer, &op2);

        graph.ClearProducers(&buffer);
        REQUIRE(graph.GetSingleProducer(&buffer) == nullptr);
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

/// Checks RemoveConsumer correctly validates and deals with multiple input slots.
TEST_CASE("OpGraph RemoveConsumer")
{
    SECTION("Try calling with nullptr Buffer")
    {
        OpGraph graph;
        MceOp o;
        graph.AddOp(&o);
        REQUIRE_THROWS(graph.RemoveConsumer(nullptr, &o, 0));
    }
    SECTION("Try calling with nullptr Op")
    {
        OpGraph graph;
        Buffer b;
        graph.AddBuffer(&b);
        REQUIRE_THROWS(graph.RemoveConsumer(&b, nullptr, 0));
    }

    SECTION("Try calling with a Buffer that isn't part of the graph")
    {
        OpGraph graph;
        MceOp o;
        graph.AddOp(&o);
        Buffer b;
        REQUIRE_THROWS(graph.RemoveConsumer(&b, &o, 0));
    }
    SECTION("Try calling with an Op that isn't part of the graph")
    {
        OpGraph graph;
        Buffer b;
        graph.AddBuffer(&b);
        MceOp o;
        REQUIRE_THROWS(graph.RemoveConsumer(&b, &o, 0));
    }

    SECTION("Try calling with a Buffer that has no consumers")
    {
        OpGraph graph;
        Buffer b;
        graph.AddBuffer(&b);
        MceOp o1;
        graph.AddOp(&o1);
        REQUIRE_THROWS(graph.RemoveConsumer(&b, &o1, 0));
    }

    SECTION("Try calling with an Op that isn't a consumer of the Buffer (but the Buffer has other consumers)")
    {
        OpGraph graph;
        Buffer b;
        graph.AddBuffer(&b);
        MceOp o1;
        graph.AddOp(&o1);
        MceOp o2;
        graph.AddOp(&o2);
        graph.AddConsumer(&b, &o1, 0);

        REQUIRE_THROWS(graph.RemoveConsumer(&b, &o2, 0));
    }

    SECTION("Try calling with an Op that is a consumer of the Buffer, but with a different input index")
    {
        OpGraph graph;
        Buffer b;
        graph.AddBuffer(&b);
        MceOp o1;
        graph.AddOp(&o1);
        graph.AddConsumer(&b, &o1, 0);

        REQUIRE_THROWS(graph.RemoveConsumer(&b, &o1, 1));
    }

    SECTION("Try removing a consumer Op which has other (later-numbered) inputs connected too")
    {
        OpGraph graph;
        Buffer b;
        graph.AddBuffer(&b);
        MceOp o1;
        graph.AddOp(&o1);
        graph.AddConsumer(&b, &o1, 0);
        graph.AddConsumer(&b, &o1, 1);

        REQUIRE_THROWS(graph.RemoveConsumer(&b, &o1, 0));
    }

    SECTION("Remove a consumer from a buffer that has only one")
    {
        OpGraph graph;
        MceOp op1;
        graph.AddOp(&op1);
        Buffer buffer;
        graph.AddBuffer(&buffer);
        graph.AddConsumer(&buffer, &op1, 0);

        graph.RemoveConsumer(&buffer, &op1, 0);
        REQUIRE(graph.GetConsumers(&buffer).size() == 0);
        REQUIRE(graph.GetInputs(&op1).size() == 0);
    }

    SECTION("Remove a consumer from a buffer that has two")
    {
        OpGraph graph;
        MceOp op1;
        graph.AddOp(&op1);
        MceOp op2;
        graph.AddOp(&op2);
        Buffer buffer;
        graph.AddBuffer(&buffer);
        graph.AddConsumer(&buffer, &op1, 0);
        graph.AddConsumer(&buffer, &op2, 0);

        graph.RemoveConsumer(&buffer, &op1, 0);
        REQUIRE(graph.GetConsumers(&buffer) == OpGraph::ConsumersList{ { &op2, 0 } });
        REQUIRE(graph.GetInputs(&op1).size() == 0);
        REQUIRE(graph.GetInputs(&op2) == std::vector<Buffer*>{ &buffer });
    }
}

/// Checks RemoveAndPrune behaves correctly.
TEST_CASE("OpGraph RemoveAndPrune")
{
    // Create test graph. We will prune from various points in this graph
    // and check the result.
    // (capital letters are Ops, lowercase letters are Buffers)
    // Note there are two (disjoint) "subgraphs" within the OpGraph
    //
    //  j  a  i
    //   \ | /
    //     B                q
    //     |                |
    //     c                Z
    //     |  \             |
    //     D   E            w
    //     |   |
    //     k   |
    //     |   |
    //     L   |
    //     |  /
    //     f
    //     | \_
    //     G   H
    //

    OpGraph graph;
    MceOp B, D, E, G, H, L, Z;
    Buffer a, c, f, i, j, k, q, w;
    graph.AddOp(&B);
    graph.AddOp(&D);
    graph.AddOp(&E);
    graph.AddOp(&G);
    graph.AddOp(&H);
    graph.AddOp(&L);
    graph.AddOp(&Z);

    graph.AddBuffer(&a);
    graph.AddBuffer(&c);
    graph.AddBuffer(&f);
    graph.AddBuffer(&i);
    graph.AddBuffer(&j);
    graph.AddBuffer(&k);
    graph.AddBuffer(&q);
    graph.AddBuffer(&w);

    graph.AddConsumer(&j, &B, 0);
    graph.AddConsumer(&a, &B, 1);
    graph.AddConsumer(&i, &B, 2);
    graph.AddProducer(&c, &B);
    graph.AddConsumer(&c, &D, 0);
    graph.AddConsumer(&c, &E, 0);
    graph.AddProducer(&k, &D);
    graph.AddConsumer(&k, &L, 0);
    graph.AddProducer(&f, &L);
    graph.AddProducer(&f, &E);
    graph.AddConsumer(&f, &G, 0);
    graph.AddConsumer(&f, &H, 0);

    graph.AddConsumer(&q, &Z, 0);
    graph.AddProducer(&w, &Z);

    bool debug = false;
    if (debug)
    {
        std::ofstream s("OpGraph RemoveAndPrune.dot");
        SaveOpGraphToDot(graph, s, DetailLevel::Low);
    }

    SECTION("Prune j")
    {
        // Not valid, as this would disconnect a non-last input of B
        REQUIRE_THROWS(graph.RemoveAndPrune(&j));
    }
    SECTION("Prune a")
    {
        // Not valid, as this would disconnect a non-last input of B
        REQUIRE_THROWS(graph.RemoveAndPrune(&a));
    }
    SECTION("Prune i")
    {
        graph.RemoveAndPrune(&i);
        // Only i is removed as B has other inputs
        CHECK(graph.GetOps() == std::vector<Op*>{ &B, &D, &E, &G, &H, &L, &Z });
        CHECK(graph.GetBuffers() == std::vector<Buffer*>{ &a, &c, &f, &j, &k, &q, &w });
    }

    SECTION("Prune B")
    {
        graph.RemoveAndPrune(&B);
        // The entire left sub-graph gets pruned
        CHECK(graph.GetOps() == std::vector<Op*>{ &Z });
        CHECK(graph.GetBuffers() == std::vector<Buffer*>{ &q, &w });
    }
    SECTION("Prune c")
    {
        graph.RemoveAndPrune(&c);
        // The entire left sub-graph gets pruned
        CHECK(graph.GetOps() == std::vector<Op*>{ &Z });
        CHECK(graph.GetBuffers() == std::vector<Buffer*>{ &q, &w });
    }

    SECTION("Prune D")
    {
        graph.RemoveAndPrune(&D);
        // The branch D-L gets removed but c and f don't, because they have other connections
        CHECK(graph.GetOps() == std::vector<Op*>{ &B, &E, &G, &H, &Z });
        CHECK(graph.GetBuffers() == std::vector<Buffer*>{ &a, &c, &f, &i, &j, &q, &w });
    }
    SECTION("Prune k")
    {
        graph.RemoveAndPrune(&k);
        // The branch D-L gets removed but c and f don't, because they have other connections
        CHECK(graph.GetOps() == std::vector<Op*>{ &B, &E, &G, &H, &Z });
        CHECK(graph.GetBuffers() == std::vector<Buffer*>{ &a, &c, &f, &i, &j, &q, &w });
    }
    SECTION("Prune L")
    {
        graph.RemoveAndPrune(&L);
        // The branch D-L gets removed but c and f don't, because they have other connections
        CHECK(graph.GetOps() == std::vector<Op*>{ &B, &E, &G, &H, &Z });
        CHECK(graph.GetBuffers() == std::vector<Buffer*>{ &a, &c, &f, &i, &j, &q, &w });
    }

    SECTION("Prune E")
    {
        graph.RemoveAndPrune(&E);
        // Only E gets removed but c and f don't, because they have other connections
        CHECK(graph.GetOps() == std::vector<Op*>{ &B, &D, &G, &H, &L, &Z });
        CHECK(graph.GetBuffers() == std::vector<Buffer*>{ &a, &c, &f, &i, &j, &k, &q, &w });
    }

    SECTION("Prune f")
    {
        graph.RemoveAndPrune(&f);
        // The entire left sub-graph gets pruned
        CHECK(graph.GetOps() == std::vector<Op*>{ &Z });
        CHECK(graph.GetBuffers() == std::vector<Buffer*>{ &q, &w });
    }

    SECTION("Prune G")
    {
        graph.RemoveAndPrune(&G);
        // Only G gets removed but f doesn't, because it has other connections
        CHECK(graph.GetOps() == std::vector<Op*>{ &B, &D, &E, &H, &L, &Z });
        CHECK(graph.GetBuffers() == std::vector<Buffer*>{ &a, &c, &f, &i, &j, &k, &q, &w });
    }
    SECTION("Prune H")
    {
        graph.RemoveAndPrune(&H);
        // Only H gets removed but f doesn't, because it has other connections
        CHECK(graph.GetOps() == std::vector<Op*>{ &B, &D, &E, &G, &L, &Z });
        CHECK(graph.GetBuffers() == std::vector<Buffer*>{ &a, &c, &f, &i, &j, &k, &q, &w });
    }

    SECTION("Prune q")
    {
        graph.RemoveAndPrune(&q);
        // The entire right sub-graph gets pruned
        CHECK(graph.GetOps() == std::vector<Op*>{ &B, &D, &E, &G, &H, &L });
        CHECK(graph.GetBuffers() == std::vector<Buffer*>{ &a, &c, &f, &i, &j, &k });
    }
    SECTION("Prune Z")
    {
        graph.RemoveAndPrune(&Z);
        // The entire right sub-graph gets pruned
        CHECK(graph.GetOps() == std::vector<Op*>{ &B, &D, &E, &G, &H, &L });
        CHECK(graph.GetBuffers() == std::vector<Buffer*>{ &a, &c, &f, &i, &j, &k });
    }
    SECTION("Prune w")
    {
        graph.RemoveAndPrune(&w);
        // The entire right sub-graph gets pruned
        CHECK(graph.GetOps() == std::vector<Op*>{ &B, &D, &E, &G, &H, &L });
        CHECK(graph.GetBuffers() == std::vector<Buffer*>{ &a, &c, &f, &i, &j, &k });
    }
}

TEST_CASE("OpGraph RemoveRedundantCopiesSramToDram Linear")
{
    // Create test graph with a chain of copies.
    // (capital letters are DmaOps, lowercase letters are Buffers)
    //
    //  c (Sram)
    //     |
    //     D
    //     |
    //  e (Dram)
    //     |
    //     F
    //     |
    //  g (Sram)
    //     |
    //     H
    //     |
    //  i (Dram)
    //     |
    //     J
    //     |
    //  k (Sram)      The chain is long so that we check that we avoid optimising overlapping
    //     |          chains.
    //     L
    //     |
    //  m (Dram)
    //     |
    //     N
    //     |
    //  o (Sram)       Put a trailing SRAM buffer on the end that won't be optimised,
    //                 to check that the chain search stops at the DRAM buffer above.

    OpGraph graph;
    DmaOp D(CascadingBufferFormat::NHWCB);
    DmaOp F(CascadingBufferFormat::NHWCB);
    DmaOp H(CascadingBufferFormat::NHWCB);
    DmaOp J(CascadingBufferFormat::NHWCB);
    DmaOp L(CascadingBufferFormat::NHWCB);
    DmaOp N(CascadingBufferFormat::NHWCB);
    Buffer c(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 16, 16, 32 }, TensorShape{ 1, 16, 16, 16 },
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer e(Location::Dram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 16, 16, 32 }, TensorShape{},
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer g(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 16, 16, 32 }, TensorShape{ 1, 16, 16, 16 },
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer i(Location::Dram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 16, 16, 32 }, TensorShape{},
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer k(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 16, 16, 32 }, TensorShape{ 1, 16, 16, 16 },
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer m(Location::Dram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 16, 16, 32 }, TensorShape{},
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer o(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 16, 16, 32 }, TensorShape{ 1, 16, 16, 16 },
             TraversalOrder::Xyz, 0, QuantizationInfo());
    graph.AddOp(&D);
    graph.AddOp(&F);
    graph.AddOp(&H);
    graph.AddOp(&J);
    graph.AddOp(&L);
    graph.AddOp(&N);

    // Add the buffers in a weird order, to confirm that the topological sort works
    graph.AddBuffer(&g);
    graph.AddBuffer(&e);
    graph.AddBuffer(&c);
    graph.AddBuffer(&i);
    graph.AddBuffer(&m);
    graph.AddBuffer(&k);
    graph.AddBuffer(&o);

    graph.AddConsumer(&c, &D, 0);
    graph.AddProducer(&e, &D);
    graph.AddConsumer(&e, &F, 0);
    graph.AddProducer(&g, &F);
    graph.AddConsumer(&g, &H, 0);
    graph.AddProducer(&i, &H);
    graph.AddConsumer(&i, &J, 0);
    graph.AddProducer(&k, &J);
    graph.AddConsumer(&k, &L, 0);
    graph.AddProducer(&m, &L);
    graph.AddConsumer(&m, &N, 0);
    graph.AddProducer(&o, &N);

    bool debug = false;
    if (debug)
    {
        std::ofstream s("OpGraph RemoveRedundantCopiesSramToDram Linear Pre.dot");
        SaveOpGraphToDot(graph, s, DetailLevel::Low);
    }

    SECTION("All good")
    {
        graph.RemoveRedundantCopies();

        if (debug)
        {
            std::ofstream s("OpGraph RemoveRedundantCopiesSramToDram Linear Post.dot");
            SaveOpGraphToDot(graph, s, DetailLevel::Low);
        }

        // e, F, g, H, i, J, k, L are removed
        //
        //  c (Sram)
        //     |
        //     D
        //     |
        //  m (Dram)
        //     |
        //     N
        //     |
        //  o (Sram)
        //
        CHECK(graph.GetBuffers() == std::vector<Buffer*>{ &c, &m, &o });
        CHECK(graph.GetOps() == std::vector<Op*>{ &D, &N });

        CHECK(graph.GetConsumers(&c) == OpGraph::ConsumersList{ { &D, 0 } });
        CHECK(graph.GetProducers(&m) == std::vector<Op*>{ &D });
        CHECK(graph.GetConsumers(&m) == OpGraph::ConsumersList{ { &N, 0 } });
        CHECK(graph.GetProducers(&o) == std::vector<Op*>{ &N });
    }

    SECTION("Chain shortened due to incompatible DMA")
    {
        // Change the final DRAM buffer to NHWC, which is then incompatible with the
        // starting SRAM buffer as it would require a depth split which NHWC doesn't support.
        // This means the chain will be shortened to the previous DRAM buffer.
        L.m_TransferFormat = CascadingBufferFormat::NHWC;
        m.m_Format         = CascadingBufferFormat::NHWC;

        graph.RemoveRedundantCopies();

        // e, F, g, H are removed
        //
        //  c (Sram)
        //     |
        //     D
        //     |
        //  i (Dram)
        //     |
        //     J
        //     |
        //  k (Sram)
        //     |
        //     L
        //     |
        //  m (Dram)
        //     |
        //     N
        //     |
        //  o (Sram)
        //
        CHECK(graph.GetBuffers() == std::vector<Buffer*>{ &c, &i, &m, &k, &o });
        CHECK(graph.GetOps() == std::vector<Op*>{ &D, &J, &L, &N });

        CHECK(graph.GetConsumers(&c) == OpGraph::ConsumersList{ { &D, 0 } });
        CHECK(graph.GetProducers(&i) == std::vector<Op*>{ &D });
        CHECK(graph.GetConsumers(&i) == OpGraph::ConsumersList{ { &J, 0 } });
        CHECK(graph.GetProducers(&k) == std::vector<Op*>{ &J });
        CHECK(graph.GetConsumers(&k) == OpGraph::ConsumersList{ { &L, 0 } });
        CHECK(graph.GetProducers(&m) == std::vector<Op*>{ &L });
        CHECK(graph.GetConsumers(&m) == OpGraph::ConsumersList{ { &N, 0 } });
        CHECK(graph.GetProducers(&o) == std::vector<Op*>{ &N });
    }
}

TEST_CASE("OpGraph RemoveRedundantCopiesDramToSram Linear")
{
    // Create test graph with a chain of copies.
    // (capital letters are DmaOps, lowercase letters are Buffers)
    //
    //  a (Dram)
    //     |
    //     B
    //     |
    //  c (Sram)
    //     |
    //     D
    //     |
    //  e (Dram)               The chain is long so that we check that we avoid optimising overlapping
    //     |                   chains.
    //     F
    //     |
    //  g (Sram)
    //     |
    //     H
    //     |
    //  i (Dram)
    //     |
    //     J
    //     |
    //  k (Sram)

    OpGraph graph;
    DmaOp B(CascadingBufferFormat::FCAF_DEEP);
    DmaOp D(CascadingBufferFormat::NHWCB);
    DmaOp F(CascadingBufferFormat::NHWCB);
    // This is a bit of a hack to prevent the Sram -> Dram optimisation from kicking in first, before we have a chance for
    // the Dram -> Sram optimisation to happen.
    F.m_Offset = TensorShape{ 0, 0, 0, 32 };
    DmaOp H(CascadingBufferFormat::NHWCB);
    DmaOp J(CascadingBufferFormat::NHWCB);
    Buffer a(Location::Dram, CascadingBufferFormat::FCAF_DEEP, TensorShape{ 1, 16, 16, 32 }, TensorShape{},
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer c(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 16, 16, 32 }, TensorShape{ 1, 16, 16, 32 },
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer e(Location::Dram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 16, 16, 32 }, TensorShape{},
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer g(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 16, 16, 32 }, TensorShape{ 1, 16, 16, 32 },
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer i(Location::Dram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 16, 16, 32 }, TensorShape{},
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer k(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 16, 16, 32 }, TensorShape{ 1, 16, 16, 32 },
             TraversalOrder::Xyz, 0, QuantizationInfo());
    graph.AddOp(&B);
    graph.AddOp(&D);
    graph.AddOp(&F);
    graph.AddOp(&H);
    graph.AddOp(&J);

    // Add the buffers in a weird order, to confirm that the topological sort works
    graph.AddBuffer(&g);
    graph.AddBuffer(&e);
    graph.AddBuffer(&c);
    graph.AddBuffer(&a);
    graph.AddBuffer(&k);
    graph.AddBuffer(&i);

    graph.AddConsumer(&a, &B, 0);
    graph.AddProducer(&c, &B);
    graph.AddConsumer(&c, &D, 0);
    graph.AddProducer(&e, &D);
    graph.AddConsumer(&e, &F, 0);
    graph.AddProducer(&g, &F);
    graph.AddConsumer(&g, &H, 0);
    graph.AddProducer(&i, &H);
    graph.AddConsumer(&i, &J, 0);
    graph.AddProducer(&k, &J);

    bool debug = true;
    if (debug)
    {
        std::ofstream s("OpGraph RemoveRedundantCopiesDramToSram Linear Pre.dot");
        SaveOpGraphToDot(graph, s, DetailLevel::Low);
    }

    SECTION("All good")
    {
        graph.RemoveRedundantCopies();

        if (debug)
        {
            std::ofstream s("OpGraph RemoveRedundantCopiesDramToSram Linear Post.dot");
            SaveOpGraphToDot(graph, s, DetailLevel::Low);
        }

        // B, c, D, e, F, g, H, i are removed
        //
        //
        //  a (Dram)
        //     |
        //     J
        //     |
        //  k (Sram)
        //
        CHECK(graph.GetBuffers() == std::vector<Buffer*>{ &a, &k });
        CHECK(graph.GetOps() == std::vector<Op*>{ &J });

        CHECK(graph.GetConsumers(&a) == OpGraph::ConsumersList{ { &J, 0 } });
        CHECK(graph.GetProducers(&k) == std::vector<Op*>{ &J });
    }

    SECTION("Chain shortened due to incompatible DMA")
    {
        // Change the last SRAM to use packed boundary data, which is then incompatible with the
        // starting DRAM buffer as it FCAF and we don't support packed boundary data with FCAF.
        // This means the chain will be shortened to the next SRAM buffer.
        k.m_PackedBoundaryThickness = { 1, 1, 1, 1 };

        graph.RemoveRedundantCopies();

        if (debug)
        {
            std::ofstream s("OpGraph RemoveRedundantCopiesDramToSram Linear Post2.dot");
            SaveOpGraphToDot(graph, s, DetailLevel::Low);
        }

        // B, c, D, e are removed
        //
        //  a (Dram)
        //     |
        //     F
        //     |
        //  g (Sram)
        //     |
        //     H
        //     |
        //  i (Dram)
        //     |
        //     J
        //     |
        //  k (Sram)
        //
        CHECK(graph.GetBuffers() == std::vector<Buffer*>{ &g, &a, &k, &i });
        CHECK(graph.GetOps() == std::vector<Op*>{ &F, &H, &J });

        CHECK(graph.GetConsumers(&a) == OpGraph::ConsumersList{ { &F, 0 } });
        CHECK(graph.GetProducers(&g) == std::vector<Op*>{ &F });
        CHECK(graph.GetConsumers(&g) == OpGraph::ConsumersList{ { &H, 0 } });
        CHECK(graph.GetProducers(&i) == std::vector<Op*>{ &H });
        CHECK(graph.GetConsumers(&i) == OpGraph::ConsumersList{ { &J, 0 } });
        CHECK(graph.GetProducers(&k) == std::vector<Op*>{ &J });
    }
}

TEST_CASE("OpGraph RemoveRedundantCopies Reshape")
{
    // Create test graph with a chain of reshapes
    // (capital letters are DmaOps, lowercase letters are Buffers)
    //
    //  a (Dram)
    //     |
    //     B
    //     |
    //  c (Sram)
    //     |
    //     D
    //     |
    //  e (Dram)
    //     |
    //     F
    //     |
    //  g (Sram)
    //     |
    //     H
    //     |
    //  i (Dram)

    OpGraph graph;
    DmaOp B(CascadingBufferFormat::NHWC);
    DmaOp D(CascadingBufferFormat::NHWC);
    DmaOp F(CascadingBufferFormat::NHWC);
    DmaOp H(CascadingBufferFormat::NHWC);
    Buffer a(Location::Dram, CascadingBufferFormat::NHWC, TensorShape{ 1, 10, 10, 10 }, TensorShape{},
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer c(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 10, 10, 10 }, TensorShape{ 1, 16, 16, 16 },
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer e(Location::Dram, CascadingBufferFormat::NHWC, TensorShape{ 1, 100, 10, 1 }, TensorShape{},
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer g(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 100, 10, 1 }, TensorShape{ 1, 112, 16, 16 },
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer i(Location::Dram, CascadingBufferFormat::NHWC, TensorShape{ 1, 1, 1000, 1 }, TensorShape{},
             TraversalOrder::Xyz, 0, QuantizationInfo());
    graph.AddOp(&B);
    graph.AddOp(&D);
    graph.AddOp(&F);
    graph.AddOp(&H);

    graph.AddBuffer(&a);
    graph.AddBuffer(&c);
    graph.AddBuffer(&e);
    graph.AddBuffer(&g);
    graph.AddBuffer(&i);

    graph.AddConsumer(&a, &B, 0);
    graph.AddProducer(&c, &B);
    graph.AddConsumer(&c, &D, 0);
    graph.AddProducer(&e, &D);
    graph.AddConsumer(&e, &F, 0);
    graph.AddProducer(&g, &F);
    graph.AddConsumer(&g, &H, 0);
    graph.AddProducer(&i, &H);

    bool debug = false;
    if (debug)
    {
        std::ofstream s("OpGraph RemoveRedundantCopies Reshape Pre.dot");
        SaveOpGraphToDot(graph, s, DetailLevel::Low);
    }

    graph.RemoveRedundantCopies();

    if (debug)
    {
        std::ofstream s("OpGraph RemoveRedundantCopies Reshape Post.dot");
        SaveOpGraphToDot(graph, s, DetailLevel::Low);
    }

    // e, F, g, H are removed
    // //
    //  a (Dram)
    //     |
    //     B
    //     |
    //  c (Sram)
    //     |
    //     D
    //     |
    //  i (Dram)
    //
    CHECK(graph.GetBuffers() == std::vector<Buffer*>{ &a, &c, &i });
    CHECK(graph.GetOps() == std::vector<Op*>{ &B, &D });

    CHECK(graph.GetConsumers(&a) == OpGraph::ConsumersList{ { &B, 0 } });
    CHECK(graph.GetProducers(&c) == std::vector<Op*>{ &B });
    CHECK(graph.GetConsumers(&c) == OpGraph::ConsumersList{ { &D, 0 } });
    CHECK(graph.GetProducers(&i) == std::vector<Op*>{ &D });
}

TEST_CASE("OpGraph RemoveRedundantCopies Invalid Buffers and Ops")
{
    // Create test graph with a chain of copies.
    // (capital letters are DmaOps, lowercase letters are Buffers)
    //
    //  a (Dram)
    //     |
    //     B
    //     |
    //  c (Sram)
    //     |
    //     D
    //     |
    //  e (Dram)
    //     |
    //     F
    //     |
    //  g (Sram)
    //     |
    //     H
    //     |
    //  i (Dram)

    OpGraph graph;
    DmaOp B(CascadingBufferFormat::NHWC);
    DmaOp D(CascadingBufferFormat::NHWC);
    DmaOp F(CascadingBufferFormat::NHWC);
    DmaOp H(CascadingBufferFormat::NHWC);
    Buffer a(Location::Dram, CascadingBufferFormat::NHWC, TensorShape{ 1, 16, 16, 16 }, TensorShape{},
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer c(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 16, 16, 16 }, TensorShape{ 1, 16, 16, 16 },
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer e(Location::Dram, CascadingBufferFormat::NHWC, TensorShape{ 1, 16, 16, 16 }, TensorShape{},
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer g(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 16, 16, 16 }, TensorShape{ 1, 16, 16, 16 },
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer i(Location::Dram, CascadingBufferFormat::NHWC, TensorShape{ 1, 16, 16, 16 }, TensorShape{},
             TraversalOrder::Xyz, 0, QuantizationInfo());
    graph.AddOp(&B);
    graph.AddOp(&D);
    graph.AddOp(&F);
    graph.AddOp(&H);

    graph.AddBuffer(&a);
    graph.AddBuffer(&c);
    graph.AddBuffer(&e);
    graph.AddBuffer(&g);
    graph.AddBuffer(&i);

    graph.AddConsumer(&a, &B, 0);
    graph.AddProducer(&c, &B);
    graph.AddConsumer(&c, &D, 0);
    graph.AddProducer(&e, &D);
    graph.AddConsumer(&e, &F, 0);
    graph.AddProducer(&g, &F);
    graph.AddConsumer(&g, &H, 0);
    graph.AddProducer(&i, &H);

    bool debug = false;
    if (debug)
    {
        std::ofstream s("OpGraph RemoveRedundantCopiesSramToDram Invalid Buffers and Ops Pre.dot");
        SaveOpGraphToDot(graph, s, DetailLevel::Low);
    }

    SECTION("All good")
    {
        // Confirm that the optimisation is done when nothing is changed
        graph.RemoveRedundantCopies();

        // e, F, g, H removed
        //
        //  a (Dram)
        //     |
        //     B
        //     |
        //  c (Sram)
        //     |
        //     D
        //     |
        //  i (Dram)
        //
        CHECK(graph.GetBuffers().size() == 3);
        CHECK(graph.GetOps().size() == 2);
    }

    SECTION("Buffers not in DRAM/SRAM")
    {
        c.m_Location = Location::PleInputSram;
        g.m_Location = Location::VirtualSram;

        graph.RemoveRedundantCopies();

        // No optimisation possible, because buffers are in weird places
        CHECK(graph.GetBuffers().size() == 5);
        CHECK(graph.GetOps().size() == 4);
    }

    SECTION("Non-Dma Ops")
    {
        // Replace D with something that's not a DMA op
        MceOp newD;
        graph.RemoveConsumer(&c, &D, 0);
        graph.RemoveProducer(&e, &D);
        graph.RemoveAndPrune(&D);
        graph.AddOp(&newD);
        graph.AddConsumer(&c, &newD, 0);
        graph.AddProducer(&e, &newD);

        graph.RemoveRedundantCopies();

        // No optimisation possible, because there is no longer a long-enough chain of DmaOps
        CHECK(graph.GetBuffers().size() == 5);
        CHECK(graph.GetOps().size() == 4);
    }

    SECTION("Reintepreting DmaOp")
    {
        // Change the format of D so that it's doing a reinterpret (not a simple copy)
        D.m_TransferFormat = CascadingBufferFormat::NHWCB;

        graph.RemoveRedundantCopies();

        // No optimisation possible, because there is no longer a long-enough chain of valid DmaOps
        CHECK(graph.GetBuffers().size() == 5);
        CHECK(graph.GetOps().size() == 4);
    }

    SECTION("Subtensor and Rehape")
    {
        // Change c -> e to be a reshape, and e -> g to be a subtensor
        e.m_TensorShape = { 1, 256, 1, 16 };
        F.m_Offset      = { 0, 128, 0, 0 };
        g.m_TensorShape = { 1, 128, 1, 16 };
        g.m_StripeShape = { 1, 128, 8, 16 };
        i.m_TensorShape = { 1, 128, 1, 16 };

        graph.RemoveRedundantCopies();

        // No optimisation possible, because we can't combine a reshape and subtensor
        CHECK(graph.GetBuffers().size() == 5);
        CHECK(graph.GetOps().size() == 4);
    }
}

TEST_CASE("OpGraph RemoveRedundantCopiesSramToDram Multiple Concat")
{
    // Create test graph with a several nested concats
    // (capital letters are DmaOps, lowercase letters are Buffers)
    //
    //  a (Sram)  b (Sram)
    //     |          |
    //     C          D
    //      \        /
    //        e (Dram)       f (Sram)
    //           |              |
    //           G              |
    //           |              |
    //        j (Sram)          |
    //           |              |
    //           K              H
    //             \           /
    //                i (Dram)

    OpGraph graph;
    DmaOp C(CascadingBufferFormat::NHWCB);
    C.m_Offset = TensorShape{ 0, 0, 0, 0 };
    DmaOp D(CascadingBufferFormat::NHWCB);
    D.m_Offset = TensorShape{ 0, 0, 0, 16 };
    DmaOp G(CascadingBufferFormat::NHWCB);
    DmaOp H(CascadingBufferFormat::NHWCB);
    H.m_Offset = TensorShape{ 0, 0, 0, 0 };
    DmaOp K(CascadingBufferFormat::NHWCB);
    K.m_Offset = TensorShape{ 0, 0, 0, 16 };
    Buffer a(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 16, 16, 16 }, TensorShape{ 1, 16, 16, 16 },
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer b(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 16, 16, 16 }, TensorShape{ 1, 16, 16, 16 },
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer e(Location::Dram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 16, 16, 32 }, TensorShape{},
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer f(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 16, 16, 16 }, TensorShape{ 1, 16, 16, 16 },
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer i(Location::Dram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 16, 16, 48 }, TensorShape{},
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer j(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 16, 16, 32 }, TensorShape{ 1, 16, 16, 32 },
             TraversalOrder::Xyz, 0, QuantizationInfo());
    graph.AddOp(&C);
    graph.AddOp(&D);
    graph.AddOp(&G);
    graph.AddOp(&H);
    graph.AddOp(&K);

    graph.AddBuffer(&a);
    graph.AddBuffer(&b);
    graph.AddBuffer(&e);
    graph.AddBuffer(&f);
    graph.AddBuffer(&i);
    graph.AddBuffer(&j);

    graph.AddConsumer(&a, &C, 0);
    graph.AddConsumer(&b, &D, 0);
    graph.AddProducer(&e, &C);
    graph.AddProducer(&e, &D);
    graph.AddConsumer(&e, &G, 0);
    graph.AddProducer(&j, &G);
    graph.AddConsumer(&j, &K, 0);
    graph.AddConsumer(&f, &H, 0);
    graph.AddProducer(&i, &K);
    graph.AddProducer(&i, &H);

    bool debug = true;
    if (debug)
    {
        std::ofstream s("OpGraph RemoveRedundantCopiesSramToDram Multiple Concat Pre.dot");
        SaveOpGraphToDot(graph, s, DetailLevel::High);
    }

    SECTION("All good")
    {
        graph.RemoveRedundantCopies();

        if (debug)
        {
            std::ofstream s("OpGraph RemoveRedundantCopiesSramToDram Multiple Concat Post.dot");
            SaveOpGraphToDot(graph, s, DetailLevel::High);
        }

        // The nested concat is removed, leaving a single one-level concat with three inputs
        //
        //  a (Sram)  b (Sram)
        //     |          |
        //     C          D
        //      \         |
        //       \        |    f (Sram)
        //        \       |       |
        //         \      |       |
        //          \     |       |
        //           \    |       |
        //            \   |       |
        //             \  |       H
        //              \ |       /
        //                i (Dram)
        //
        CHECK(graph.GetBuffers() == std::vector<Buffer*>{ &a, &b, &f, &i });
        CHECK(graph.GetOps() == std::vector<Op*>{ &C, &D, &H });

        CHECK(graph.GetConsumers(&a) == OpGraph::ConsumersList{ { &C, 0 } });
        CHECK(graph.GetConsumers(&b) == OpGraph::ConsumersList{ { &D, 0 } });
        CHECK(graph.GetConsumers(&f) == OpGraph::ConsumersList{ { &H, 0 } });
        CHECK(graph.GetProducers(&i) == std::vector<Op*>{ &H, &C, &D });

        CHECK(C.m_Offset == TensorShape{ 0, 0, 0, 16 });
        CHECK(D.m_Offset == TensorShape{ 0, 0, 0, 32 });
        CHECK(H.m_Offset == TensorShape{ 0, 0, 0, 0 });
    }

    SECTION("Invalid subtensor")
    {
        // Change the graph so that not all of the input data makes it into the output buffer, making
        // the optimisation not possible
        G.m_Offset      = TensorShape{ 0, 0, 0, 16 };      // From 0,0,0,0
        j.m_TensorShape = TensorShape{ 1, 16, 16, 16 };    // Down from 32 depth
        j.m_StripeShape = TensorShape{ 1, 16, 16, 16 };    // Down from 32 depth
        i.m_TensorShape = TensorShape{ 1, 16, 16, 32 };    // Down from 48 depth

        graph.RemoveRedundantCopies();

        // No optimisation possible
        CHECK(graph.GetBuffers().size() == 6);
        CHECK(graph.GetOps().size() == 5);
    }

    SECTION("Invalid branch")
    {
        // Add a second consumer to e, which should prevent the optimisation
        DmaOp newConsumer(CascadingBufferFormat::NHWCB);
        graph.AddOp(&newConsumer);
        graph.AddConsumer(&e, &newConsumer, 0);

        graph.RemoveRedundantCopies();

        // No optimisation possible
        CHECK(graph.GetBuffers().size() == 6);
        CHECK(graph.GetOps().size() == 6);
    }
}

TEST_CASE("OpGraph RemoveRedundantCopiesDramToSram Multiple Split")
{
    // Create test graph with a several nested splits
    // (capital letters are DmaOps, lowercase letters are Buffers)
    //
    //                m (Sram)       Put a leading SRAM buffer on the start that won't be optimised,
    //                   |           to check that the chain search stops at the DRAM buffer below.
    //                   L
    //                   |
    //                i (Dram)
    //             /           \_
    //           K              H
    //           |              |
    //        j (Sram)          |
    //           |              |
    //           G              |
    //           |              |
    //        e (Dram)       f (Sram)
    //      /        \_
    //     C          D
    //     |          |
    //  a (Sram)   b (Sram)

    OpGraph graph;
    DmaOp C(CascadingBufferFormat::NHWCB);
    C.m_Offset = TensorShape{ 0, 0, 0, 0 };
    DmaOp D(CascadingBufferFormat::NHWCB);
    D.m_Offset = TensorShape{ 0, 0, 0, 16 };
    DmaOp G(CascadingBufferFormat::NHWCB);
    DmaOp H(CascadingBufferFormat::NHWCB);
    H.m_Offset = TensorShape{ 0, 0, 0, 0 };
    DmaOp K(CascadingBufferFormat::NHWCB);
    K.m_Offset = TensorShape{ 0, 0, 0, 16 };
    DmaOp L(CascadingBufferFormat::NHWCB);
    Buffer a(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 16, 16, 16 }, TensorShape{ 1, 16, 16, 16 },
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer b(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 16, 16, 16 }, TensorShape{ 1, 16, 16, 16 },
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer e(Location::Dram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 16, 16, 32 }, TensorShape{},
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer f(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 16, 16, 16 }, TensorShape{ 1, 16, 16, 16 },
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer i(Location::Dram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 16, 16, 48 }, TensorShape{},
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer j(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 16, 16, 32 }, TensorShape{ 1, 16, 16, 32 },
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer m(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 16, 16, 48 }, TensorShape{ 1, 16, 16, 48 },
             TraversalOrder::Xyz, 0, QuantizationInfo());
    graph.AddOp(&C);
    graph.AddOp(&D);
    graph.AddOp(&G);
    graph.AddOp(&H);
    graph.AddOp(&K);
    graph.AddOp(&L);

    graph.AddBuffer(&a);
    graph.AddBuffer(&b);
    graph.AddBuffer(&e);
    graph.AddBuffer(&f);
    graph.AddBuffer(&i);
    graph.AddBuffer(&j);
    graph.AddBuffer(&m);

    graph.AddProducer(&a, &C);
    graph.AddProducer(&b, &D);
    graph.AddConsumer(&e, &C, 0);
    graph.AddConsumer(&e, &D, 0);
    graph.AddProducer(&e, &G);
    graph.AddConsumer(&j, &G, 0);
    graph.AddProducer(&j, &K);
    graph.AddProducer(&f, &H);
    graph.AddConsumer(&i, &K, 0);
    graph.AddConsumer(&i, &H, 0);
    graph.AddProducer(&i, &L);
    graph.AddConsumer(&m, &L, 0);

    bool debug = true;
    if (debug)
    {
        std::ofstream s("OpGraph RemoveRedundantCopiesSramToDram Multiple Split Pre.dot");
        SaveOpGraphToDot(graph, s, DetailLevel::High);
    }

    SECTION("All good")
    {
        graph.RemoveRedundantCopies();

        if (debug)
        {
            std::ofstream s("OpGraph RemoveRedundantCopiesSramToDram Multiple Split Post.dot");
            SaveOpGraphToDot(graph, s, DetailLevel::High);
        }

        // The nested split is removed, leaving a single one-level split with three outputs
        //
        //                m (Sram)
        //                   |
        //                   L
        //                   |
        //                i (Dram)
        //             /  |        \_
        //            /   |        H
        //           /    |        |
        //          /     |        |
        //         /      |        |
        //        /       |        |
        //       /        |        |
        //      /         |     f (Sram)
        //     /          |
        //     C          D
        //     |          |
        //  a (Sram)   b (Sram)
        //
        CHECK(graph.GetBuffers() == std::vector<Buffer*>{ &a, &b, &f, &i, &m });
        CHECK(graph.GetOps() == std::vector<Op*>{ &C, &D, &H, &L });

        CHECK(graph.GetConsumers(&i) == OpGraph::ConsumersList{ { &H, 0 }, { &D, 0 }, { &C, 0 } });
        CHECK(graph.GetProducers(&a) == std::vector<Op*>{ &C });
        CHECK(graph.GetProducers(&b) == std::vector<Op*>{ &D });
        CHECK(graph.GetProducers(&f) == std::vector<Op*>{ &H });
        CHECK(graph.GetConsumers(&m) == OpGraph::ConsumersList{ { &L, 0 } });
        CHECK(graph.GetProducers(&i) == std::vector<Op*>{ &L });

        CHECK(C.m_Offset == TensorShape{ 0, 0, 0, 16 });
        CHECK(D.m_Offset == TensorShape{ 0, 0, 0, 32 });
        CHECK(H.m_Offset == TensorShape{ 0, 0, 0, 0 });
    }

    SECTION("Invalid subtensor")
    {
        // Change the graph so that some of an output buffer doesn't come from the input buffer, making
        // the optimisation not possible
        G.m_Offset      = TensorShape{ 0, 0, 0, 16 };      // From 0,0,0,0
        j.m_TensorShape = TensorShape{ 1, 16, 16, 16 };    // Down from 32 depth
        j.m_StripeShape = TensorShape{ 1, 16, 16, 16 };    // Down from 32 depth
        i.m_TensorShape = TensorShape{ 1, 16, 16, 32 };    // Down from 48 depth
        m.m_TensorShape = TensorShape{ 1, 16, 16, 32 };    // Down from 48 depth
        m.m_StripeShape = TensorShape{ 1, 16, 16, 32 };    // Down from 48 depth

        graph.RemoveRedundantCopies();

        // No optimisation possible
        CHECK(graph.GetBuffers().size() == 7);
        CHECK(graph.GetOps().size() == 6);
    }

    SECTION("Invalid branch")
    {
        // Add a second producer to e, which should prevent the optimisation
        DmaOp newProducer(CascadingBufferFormat::NHWCB);
        graph.AddOp(&newProducer);
        graph.AddProducer(&e, &newProducer);

        graph.RemoveRedundantCopies();

        // No optimisation possible
        CHECK(graph.GetBuffers().size() == 7);
        CHECK(graph.GetOps().size() == 7);
    }
}

TEST_CASE("OpGraph RemoveRedundantCopiesSramToDram Concat one branch invalid")
{
    // Create test graph with a concat and a conversion afterwards.
    // The first branch can be optimised, but the second can't, which means that none of it can
    // (capital letters are DmaOps, lowercase letters are Buffers)
    //
    //  a (Sram)  b (Sram)
    //     |          |
    //     C          D
    //      \        /
    //        e (Dram)
    //           |
    //           G
    //           |
    //        j (Sram)
    //           |
    //           K
    //           |
    //        i (Dram)

    OpGraph graph;
    DmaOp C(CascadingBufferFormat::NHWC);
    // This DMA can always be optimised straight into buffer i, no matter its format
    C.m_Offset = TensorShape{ 0, 0, 0, 0 };
    DmaOp D(CascadingBufferFormat::NHWC);
    // This DMA can't be optimised straight into buffer i, if it is NHWCB - it only works if it's NHWC
    D.m_Offset = TensorShape{ 0, 10, 0, 0 };
    DmaOp G(CascadingBufferFormat::NHWC);
    DmaOp K(CascadingBufferFormat::NHWC);
    Buffer a(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 10, 16, 16 }, TensorShape{ 1, 16, 16, 16 },
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer b(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 6, 16, 16 }, TensorShape{ 1, 16, 16, 16 },
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer e(Location::Dram, CascadingBufferFormat::NHWC, TensorShape{ 1, 16, 16, 16 }, TensorShape{},
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer j(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 16, 16, 16 }, TensorShape{ 1, 16, 16, 16 },
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer i(Location::Dram, CascadingBufferFormat::NHWC, TensorShape{ 1, 16, 16, 16 }, TensorShape{},
             TraversalOrder::Xyz, 0, QuantizationInfo());
    graph.AddOp(&C);
    graph.AddOp(&D);
    graph.AddOp(&G);
    graph.AddOp(&K);

    graph.AddBuffer(&a);
    graph.AddBuffer(&b);
    graph.AddBuffer(&e);
    graph.AddBuffer(&j);
    graph.AddBuffer(&i);

    graph.AddConsumer(&a, &C, 0);
    graph.AddConsumer(&b, &D, 0);
    graph.AddProducer(&e, &C);
    graph.AddProducer(&e, &D);
    graph.AddConsumer(&e, &G, 0);
    graph.AddProducer(&j, &G);
    graph.AddConsumer(&j, &K, 0);
    graph.AddProducer(&i, &K);

    bool debug = true;
    if (debug)
    {
        std::ofstream s("OpGraph RemoveRedundantCopiesSramToDram Concat one branch invalid Pre.dot");
        SaveOpGraphToDot(graph, s, DetailLevel::High);
    }

    SECTION("All good")
    {
        // Confirm that the optimisation is done, if we don't make any changes
        graph.RemoveRedundantCopies();

        if (debug)
        {
            std::ofstream s("OpGraph RemoveRedundantCopiesSramToDram Concat one branch invalid Post.dot");
            SaveOpGraphToDot(graph, s, DetailLevel::High);
        }

        //
        //  a (Sram)  b (Sram)
        //     |          |
        //     C          D
        //      \        /
        //        i (Dram)
        //
        CHECK(graph.GetBuffers().size() == 3);
        CHECK(graph.GetOps().size() == 2);
    }

    SECTION("Make one branch invalid, but the other is valid still")
    {
        // We can no longer DMA straight from b -> i, as you can't start at H offset 10 into NHWCB
        K.m_TransferFormat = CascadingBufferFormat::NHWCB;
        i.m_Format         = CascadingBufferFormat::NHWCB;

        graph.RemoveRedundantCopies();

        // This means that the optimisation can't be performed on one branch. This then prevents the
        // optimisation from occuring on the other branch as well, otherwise we'd be left with a "concat buffer"
        // with only one input.
        CHECK(graph.GetBuffers().size() == 5);
        CHECK(graph.GetOps().size() == 4);
    }
}

TEST_CASE("OpGraph RemoveRedundantCopiesDramToSram Split one branch invalid")
{
    // Create test graph with a split and a conversion beforehand
    // The first branch can be optimised, but the second can't. Unlike for Concat, we can still optimise just one branch here
    // (capital letters are DmaOps, lowercase letters are Buffers)
    //
    //        i (Dram)
    //           |
    //           K
    //           |
    //        j (Sram)
    //           |
    //           G
    //           |
    //        e (Dram)
    //      /        \_
    //     C          D
    //     |          |
    //  a (Sram)  b (Sram)
    //
    OpGraph graph;
    DmaOp C(CascadingBufferFormat::NHWC);
    // This DMA can always be optimised straight from buffer i, no matter its format
    C.m_Offset = TensorShape{ 0, 0, 0, 0 };
    DmaOp D(CascadingBufferFormat::NHWC);
    // This DMA can't be optimised straight from buffer i, if it is NHWCB - it only works if it's NHWC
    D.m_Offset = TensorShape{ 0, 10, 0, 0 };
    DmaOp G(CascadingBufferFormat::NHWC);
    DmaOp K(CascadingBufferFormat::NHWC);
    Buffer a(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 10, 16, 16 }, TensorShape{ 1, 16, 16, 16 },
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer b(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 6, 16, 16 }, TensorShape{ 1, 16, 16, 16 },
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer e(Location::Dram, CascadingBufferFormat::NHWC, TensorShape{ 1, 16, 16, 16 }, TensorShape{},
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer j(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape{ 1, 16, 16, 16 }, TensorShape{ 1, 16, 16, 16 },
             TraversalOrder::Xyz, 0, QuantizationInfo());
    Buffer i(Location::Dram, CascadingBufferFormat::NHWC, TensorShape{ 1, 16, 16, 16 }, TensorShape{},
             TraversalOrder::Xyz, 0, QuantizationInfo());
    graph.AddOp(&C);
    graph.AddOp(&D);
    graph.AddOp(&G);
    graph.AddOp(&K);

    graph.AddBuffer(&a);
    graph.AddBuffer(&b);
    graph.AddBuffer(&e);
    graph.AddBuffer(&j);
    graph.AddBuffer(&i);

    graph.AddProducer(&a, &C);
    graph.AddProducer(&b, &D);
    graph.AddConsumer(&e, &C, 0);
    graph.AddConsumer(&e, &D, 0);
    graph.AddProducer(&e, &G);
    graph.AddConsumer(&j, &G, 0);
    graph.AddProducer(&j, &K);
    graph.AddConsumer(&i, &K, 0);

    bool debug = true;
    if (debug)
    {
        std::ofstream s("OpGraph RemoveRedundantCopiesDramToSram Split one branch invalid Pre.dot");
        SaveOpGraphToDot(graph, s, DetailLevel::High);
    }

    SECTION("All good")
    {
        // Confirm that the optimisation is done, if we don't make any changes
        graph.RemoveRedundantCopies();

        if (debug)
        {
            std::ofstream s("OpGraph RemoveRedundantCopiesDramToSram Split one branch invalid Post.dot");
            SaveOpGraphToDot(graph, s, DetailLevel::High);
        }

        //
        //        i (Dram)
        //      /        \_
        //     C          D
        //     |          |
        //  a (Sram)  b (Sram)
        //
        CHECK(graph.GetBuffers().size() == 3);
        CHECK(graph.GetOps().size() == 2);
    }

    SECTION("Make one branch invalid, but the other is valid still")
    {
        // We can no longer DMA straight from i -> b, as you can't start at H offset 10 into NHWCB
        K.m_TransferFormat = CascadingBufferFormat::NHWCB;
        i.m_Format         = CascadingBufferFormat::NHWCB;

        graph.RemoveRedundantCopies();

        if (debug)
        {
            std::ofstream s("OpGraph RemoveRedundantCopiesDramToSram Split one branch invalid Post2.dot");
            SaveOpGraphToDot(graph, s, DetailLevel::High);
        }

        // This means that the optimisation can't be performed on one branch. The other branch
        // can still be optimised though (unlike for Concat!)
        //
        //        i (Dram)
        //      /    |
        //     |     K
        //     |     |
        //     |  j (Sram)
        //     |     |
        //     |     G
        //     |     |
        //     |  e (Dram)
        //     |         \_
        //     C          D
        //     |          |
        //  a (Sram)  b (Sram)
        //

        CHECK(graph.GetBuffers() == std::vector<Buffer*>{ &a, &b, &e, &j, &i });
        CHECK(graph.GetOps() == std::vector<Op*>{ &C, &D, &G, &K });

        CHECK(graph.GetConsumers(&i) == OpGraph::ConsumersList{ { &K, 0 }, { &C, 0 } });
        CHECK(graph.GetProducers(&a) == std::vector<Op*>{ &C });
        CHECK(graph.GetProducers(&j) == std::vector<Op*>{ &K });
        CHECK(graph.GetConsumers(&j) == OpGraph::ConsumersList{ { &G, 0 } });
        CHECK(graph.GetProducers(&e) == std::vector<Op*>{ &G });
        CHECK(graph.GetConsumers(&e) == OpGraph::ConsumersList{ { &D, 0 } });
        CHECK(graph.GetProducers(&b) == std::vector<Op*>{ &D });

        CHECK(C.m_Offset == TensorShape{ 0, 0, 0, 0 });
        CHECK(C.m_TransferFormat == CascadingBufferFormat::NHWCB);
        CHECK(K.m_Offset == TensorShape{ 0, 0, 0, 0 });
        CHECK(G.m_Offset == TensorShape{ 0, 0, 0, 0 });
        CHECK(D.m_Offset == TensorShape{ 0, 10, 0, 0 });
    }
}
