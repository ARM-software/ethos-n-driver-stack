//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/DebuggingContext.hpp"
#include "../src/nonCascading/BufferManager.hpp"

#include <catch.hpp>

using namespace ethosn::support_library;

TEST_CASE("BufferManager alignment")
{
    // Check that the BufferManager aligns buffers to 64-byte boundaries
    // Create several buffers of each different type for maximum coverage (each type of buffer is allocated in a
    // separate space). Each buffer is size 1.
    BufferManager m;
    m.AddDramConstant(BufferType::ConstantControlUnit, std::vector<uint8_t>{ 0 });
    m.AddDramConstant(BufferType::ConstantControlUnit, std::vector<uint8_t>{ 0 });
    m.AddDramConstant(BufferType::ConstantControlUnit, std::vector<uint8_t>{ 0 });
    m.AddDramConstant(BufferType::ConstantDma, std::vector<uint8_t>{ 0 });
    m.AddDramConstant(BufferType::ConstantDma, std::vector<uint8_t>{ 0 });
    m.AddDramConstant(BufferType::ConstantDma, std::vector<uint8_t>{ 0 });
    m.AddDram(BufferType::Input, 1);
    m.AddDram(BufferType::Input, 1);
    m.AddDram(BufferType::Input, 1);
    uint32_t intermediateId1 = m.AddDram(BufferType::Intermediate, 1);
    m.MarkBufferUsedAtTime(intermediateId1, 0);
    uint32_t intermediateId2 = m.AddDram(BufferType::Intermediate, 1);
    m.MarkBufferUsedAtTime(intermediateId2, 0);
    uint32_t intermediateId3 = m.AddDram(BufferType::Intermediate, 1);
    m.MarkBufferUsedAtTime(intermediateId3, 0);
    m.AddDram(BufferType::Output, 1);
    m.AddDram(BufferType::Output, 1);
    m.AddDram(BufferType::Output, 1);

    // Allocate the buffers
    CompilationOptions::DebugInfo debugInfo;
    m.Allocate(DebuggingContext(&debugInfo));

    // Check their alignment
    for (auto& bufferIt : m.GetBuffers())
    {
        CHECK(bufferIt.second.m_Offset % 64 == 0);
    }
}

using namespace first_fit_allocation;

TEST_CASE("FirstFitAllocation no overlap", "[implementation-unaware]")
{
    // These three buffers do not have overlapping lifetimes and so can all be allocated at address 0
    std::vector<Buffer> input = std::vector<Buffer>{
        { 0u, 1u, 10u },
        { 1u, 2u, 10u },
        { 2u, 3u, 10u },
    };
    std::vector<uint32_t> actual   = FirstFitAllocation(input, 1);
    std::vector<uint32_t> expected = { 0, 0, 0 };
    CHECK(actual == expected);
}

TEST_CASE("FirstFitAllocation alternate overlapping", "[implementation-unaware]")
{
    // Each buffer has a lifetime of length 2 and overlaps with both the buffer before and after.
    // This is the typical pattern of intermediate buffers for linear networks.
    std::vector<Buffer> input = std::vector<Buffer>{
        { 0u, 2u, 10u },
        { 1u, 3u, 10u },
        { 2u, 4u, 10u },
        { 3u, 5u, 10u },
    };
    std::vector<uint32_t> actual = FirstFitAllocation(input, 1);
    // We can re-use the space of the previous-but-one buffer, so we alternate between two locations
    std::vector<uint32_t> expected = { 0, 10, 0, 10 };
    CHECK(actual == expected);
}

TEST_CASE("FirstFitAllocation one long lived", "[implementation-unaware]")
{
    std::vector<Buffer> input = std::vector<Buffer>{
        { 0u, 10u, 10u },    // This buffer has a long lifetime and so nothing can re-use its space
        { 1u, 2u, 10u },
        { 2u, 3u, 10u },
    };
    std::vector<uint32_t> actual   = FirstFitAllocation(input, 1);
    std::vector<uint32_t> expected = { 0, 10, 10 };
    CHECK(actual == expected);
}

TEST_CASE("FirstFitAllocation order independent - reverse time", "[implementation-unaware]")
{
    // Lists the buffers in a non-obvious order - from largest creation time to smallest creation time
    std::vector<Buffer> input = std::vector<Buffer>{
        { 3u, 5u, 10u },
        { 2u, 4u, 10u },
        { 1u, 3u, 10u },
        { 0u, 2u, 10u },
    };
    std::vector<uint32_t> actual   = FirstFitAllocation(input, 1);
    std::vector<uint32_t> expected = { 10, 0, 10, 0 };
    CHECK(actual == expected);
}

TEST_CASE("FirstFitAllocation order independent - same time", "[implementation-unaware]")
{
    // Several buffers are created at the same instant - they should be allocated in the order that they are provided
    // to the function, so that the results are deterministic.
    std::vector<Buffer> input = std::vector<Buffer>{
        { 0u, 1u, 10u },
        { 0u, 1u, 10u },
        { 0u, 1u, 10u },
    };
    std::vector<uint32_t> actual   = FirstFitAllocation(input, 1);
    std::vector<uint32_t> expected = { 0, 10, 20 };
    CHECK(actual == expected);
}

TEST_CASE("FirstFitAllocation fragmented", "[implementation-unaware]")
{
    // Three buffers are allocated and then the middle one is freed. This leaves a hole that could be used for the final
    // buffer, but it is not big enough, so the final buffer must be placed at the end
    std::vector<Buffer> input = std::vector<Buffer>{
        { 0u, 5u, 10u },
        { 0u, 1u, 10u },
        { 0u, 5u, 10u },
        { 3u, 4u, 20u },
    };
    std::vector<uint32_t> actual   = FirstFitAllocation(input, 1);
    std::vector<uint32_t> expected = { 0, 10, 20, 30 };
    CHECK(actual == expected);
}

TEST_CASE("FirstFitAllocation alignment", "[implementation-unaware]")
{
    // Allocate three buffers then free the middle one, so that (if the algorithm ignored alignment) then there would
    // be a gap in the middle that would be big enough for the fourth allocation, but wouldn't be aligned correctly.
    // The implementation instead places every buffer on a multiple of 10, to avoid this.
    std::vector<Buffer> input = std::vector<Buffer>{
        { 0u, 2u, 9u },
        { 0u, 1u, 9u },
        { 0u, 2u, 9u },
        { 1u, 2u, 5u },
    };
    std::vector<uint32_t> actual   = FirstFitAllocation(input, 10);
    std::vector<uint32_t> expected = { 0, 10, 20, 10 };
    CHECK(actual == expected);
}

TEST_CASE("FirstFitAllocation free region created", "[implementation-aware]")
{
    // This test is targeted at the code in the implementation which creates a new free region.
    // Three buffers are allocated then the middle one is freed.
    // This should create a new free region that can be used for a fourth buffer
    std::vector<Buffer> input = std::vector<Buffer>{
        { 0u, 2u, 10u },
        { 0u, 1u, 10u },
        { 0u, 2u, 10u },
        { 1u, 2u, 10u },
    };
    std::vector<uint32_t> actual   = FirstFitAllocation(input, 1);
    std::vector<uint32_t> expected = { 0, 10, 20, 10 };
    CHECK(actual == expected);
}

TEST_CASE("FirstFitAllocation free regions merged", "[implementation-aware]")
{
    // This test is targeted at the code in the implementation which merges free regions together.
    // Three buffers are allocated then the outer ones are freed and then the middle one is freed.
    // This should leave a single big free region that can be used for a fourth buffer
    std::vector<Buffer> input = std::vector<Buffer>{
        { 0u, 1u, 10u },
        { 0u, 2u, 10u },
        { 0u, 1u, 10u },
        { 3u, 4u, 30u },
    };
    std::vector<uint32_t> actual   = FirstFitAllocation(input, 1);
    std::vector<uint32_t> expected = { 0, 10, 20, 0 };
    CHECK(actual == expected);
}

TEST_CASE("FirstFitAllocation free region before extended", "[implementation-aware]")
{
    // This test is targeted at the code in the implementation which extends free regions.
    // Three buffers are created and then the first one is freed, leaving a free region at the start of memory.
    // The second buffer is then freed which should extend the free region.
    // A fourth buffer is allocated that should take the space that the first two used.
    std::vector<Buffer> input = std::vector<Buffer>{
        { 0u, 1u, 10u },
        { 0u, 2u, 10u },
        { 0u, 10u, 10u },
        { 3u, 4u, 20u },
    };
    std::vector<uint32_t> actual   = FirstFitAllocation(input, 1);
    std::vector<uint32_t> expected = { 0, 10, 20, 0 };
    CHECK(actual == expected);
}

TEST_CASE("FirstFitAllocation free region after extended", "[implementation-aware]")
{
    // This test is targeted at the code in the implementation which extends free regions.
    // A buffer is created and then freed, which should extend the 'infinite' free region back to the start of
    // memory. A second buffer is then allocated which should re-use address 0.
    std::vector<Buffer> input = std::vector<Buffer>{
        { 0u, 1u, 10u },
        { 2u, 3u, 10u },
    };
    std::vector<uint32_t> actual   = FirstFitAllocation(input, 1);
    std::vector<uint32_t> expected = { 0, 0 };
    CHECK(actual == expected);
}
