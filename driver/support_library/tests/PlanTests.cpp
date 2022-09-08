//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/cascading/Plan.hpp"
#include "../src/cascading/PleKernelDatabase.hpp"

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

TEST_CASE("Get size in bytes helpers")
{
    Buffer* buffer;
    Buffer tempBuffer;

    Plan planASram;
    tempBuffer = Buffer(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(), TensorShape{ 1, 8, 8, 16 },
                        TraversalOrder::Xyz, 4 * 8 * 8 * 16, QuantizationInfo());
    buffer     = planASram.m_OpGraph.AddBuffer(std::make_unique<Buffer>(std::move(tempBuffer)));
    planASram.m_OutputMappings[buffer] = PartOutputSlot{ 0, 0 };

    REQUIRE(GetTotSizeInBytes(planASram).m_Tot == 4 * 8 * 8 * 16);
    REQUIRE(GetInputsSizeInBytes(planASram).m_Tot == 0);

    Plan planBSram;
    tempBuffer = Buffer(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(), TensorShape{ 1, 8, 8, 8 },
                        TraversalOrder::Xyz, 4 * 8 * 8 * 8, QuantizationInfo());
    buffer     = planBSram.m_OpGraph.AddBuffer(std::make_unique<Buffer>(std::move(tempBuffer)));
    planBSram.m_InputMappings[buffer] = PartInputSlot{ 0, 0 };

    REQUIRE(GetTotSizeInBytes(planBSram).m_Tot == 4 * 8 * 8 * 8);
    REQUIRE(GetInputsSizeInBytes(planBSram).m_Tot == 4 * 8 * 8 * 8);
}

TEST_CASE("FindPleKernelIdFromDatabase")
{
    // ADDITION's block multiplier = 1 independent of input stripe width
    // It is also block size "agnostic"
    PleKernelId id0 = plelib::FindPleKernelIdFromDatabase(ethosn::command_stream::BlockConfig{ 8u, 16u }, 64,
                                                          ethosn::command_stream::DataType::U8,
                                                          ethosn::command_stream::PleOperation::ADDITION);
    REQUIRE(id0 == PleKernelId::ADDITION_16X16_1);

    PleKernelId id1 = plelib::FindPleKernelIdFromDatabase(ethosn::command_stream::BlockConfig{ 16u, 16u }, 8,
                                                          ethosn::command_stream::DataType::U8,
                                                          ethosn::command_stream::PleOperation::ADDITION);
    REQUIRE(id1 == PleKernelId::ADDITION_16X16_1);

    // signed
    PleKernelId id2 = plelib::FindPleKernelIdFromDatabase(ethosn::command_stream::BlockConfig{ 16u, 16u }, 8,
                                                          ethosn::command_stream::DataType::S8,
                                                          ethosn::command_stream::PleOperation::ADDITION);
    REQUIRE(id2 == PleKernelId::ADDITION_16X16_1_S);

    // PASSTHROUGH is SignAgnostic
    PleKernelId id3 = plelib::FindPleKernelIdFromDatabase(ethosn::command_stream::BlockConfig{ 16u, 16u }, 64,
                                                          ethosn::command_stream::DataType::S8,
                                                          ethosn::command_stream::PleOperation::PASSTHROUGH);
    REQUIRE(id3 == PleKernelId::PASSTHROUGH_16X16_1);

    PleKernelId id4 = plelib::FindPleKernelIdFromDatabase(ethosn::command_stream::BlockConfig{ 8u, 16u }, 64,
                                                          ethosn::command_stream::DataType::S8,
                                                          ethosn::command_stream::PleOperation::PASSTHROUGH);
    REQUIRE(id4 == PleKernelId::PASSTHROUGH_8X16_1);

    // Best block multiplier = 2 for (16, 8) stripeWidth / BlockWidth = 2
    PleKernelId id5 = plelib::FindPleKernelIdFromDatabase(ethosn::command_stream::BlockConfig{ 16u, 8u }, 64,
                                                          ethosn::command_stream::DataType::U8,
                                                          ethosn::command_stream::PleOperation::PASSTHROUGH);
    REQUIRE(id5 == PleKernelId::PASSTHROUGH_16X8_2);

    // Best block multiplier = 1 for (32, 8), although stripeWidth / blkWidth >= 2
    PleKernelId id6 = plelib::FindPleKernelIdFromDatabase(ethosn::command_stream::BlockConfig{ 32u, 8u }, 64,
                                                          ethosn::command_stream::DataType::U8,
                                                          ethosn::command_stream::PleOperation::DOWNSAMPLE_2X2);
    REQUIRE(id6 == PleKernelId::DOWNSAMPLE_2X2_32X8_1);

    // Best block multiplier = 2 for (8, 8), stripeWidth / blkWidth = 1
    // Downsample is also SignAgnostic
    PleKernelId id7 = plelib::FindPleKernelIdFromDatabase(ethosn::command_stream::BlockConfig{ 8u, 8u }, 8,
                                                          ethosn::command_stream::DataType::S8,
                                                          ethosn::command_stream::PleOperation::DOWNSAMPLE_2X2);
    REQUIRE(id7 == PleKernelId::DOWNSAMPLE_2X2_8X8_2);
}
