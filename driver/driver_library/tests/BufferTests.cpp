//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_driver_library/Buffer.hpp"
#include "../include/ethosn_driver_library/Network.hpp"
#include "../include/ethosn_driver_library/ProcMemAllocator.hpp"
#include "../src/Utils.hpp"
#include <catch.hpp>

#include <cstring>
#include <iostream>

using namespace ethosn::driver_library;

TEST_CASE("SimpleBufferAllocation")
{
    uint32_t bufSize = 1000;

    ProcMemAllocator test_allocator;

    // Create Simple buffer
    Buffer test_buffer = test_allocator.CreateBuffer(bufSize);

    // Verify Buffer properties
    REQUIRE(test_buffer.GetSize() == bufSize);
}

TEST_CASE("BufferSource")
{
    uint8_t test_src[] = "This is a test source data";

    ProcMemAllocator test_allocator;

    // Create a buffer with test source data
    Buffer test_buffer = test_allocator.CreateBuffer(test_src, sizeof(test_src));

    // Verify Buffer properties
    REQUIRE(test_buffer.GetSize() == sizeof(test_src));
    REQUIRE(std::memcmp(test_buffer.Map(), test_src, sizeof(test_src)) == 0);
}

TEST_CASE("BufferMove")
{
    // Test that a move is possible and behaves correct.
    // The aim of this test is to try to verify unique properties of Buffer,
    // but as most of the unique "features" shows itself as compile time errors
    // we ended up with only a move test.

    uint8_t test_src[] = "This is a test to check that data and buffer properties are moved";

    ProcMemAllocator test_allocator;

    // Create the first buffer with test source data
    Buffer test_buffer = test_allocator.CreateBuffer(test_src, sizeof(test_src));

    // Create a new buffer with a move from first buffer
    Buffer test_buffer2 = std::move(test_buffer);

    // Verify that first buffer is not available anymore but new buffer is
    // cppcheck-suppress accessMoved
    REQUIRE(!test_buffer);
    REQUIRE(test_buffer2);

    // Verify that first buffer throws exception if used
    int bufferId;
    REQUIRE_THROWS(test_buffer.GetSize() == sizeof(test_src));
    REQUIRE_THROWS(bufferId = test_buffer.GetBufferHandle());
    REQUIRE_THROWS(std::memcmp(test_buffer.Map(), test_src, sizeof(test_src)) == 0);
    REQUIRE_THROWS(test_buffer.Unmap());

    // Verify that new buffer properties and content match what was set in the first buffer
    REQUIRE(test_buffer2.GetSize() == sizeof(test_src));
    REQUIRE(std::memcmp(test_buffer2.Map(), test_src, sizeof(test_src)) == 0);
    REQUIRE_NOTHROW(test_buffer2.Unmap());
}

TEST_CASE("BufferMap/Unmap")
{
    uint8_t test_src[] = "This is a test source data";

    ProcMemAllocator test_allocator;

    // Create a buffer with test source data
    Buffer test_buffer = test_allocator.CreateBuffer(test_src, sizeof(test_src));

    // Verify Buffer properties
    REQUIRE(test_buffer.GetSize() == sizeof(test_src));
    REQUIRE(std::memcmp(test_buffer.Map(), test_src, sizeof(test_src)) == 0);
    REQUIRE_NOTHROW(test_buffer.Unmap());
    // Check that it is not going to munmap twice
    REQUIRE_NOTHROW(test_buffer.Unmap());
}
