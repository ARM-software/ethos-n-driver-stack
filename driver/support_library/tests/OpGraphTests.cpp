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
