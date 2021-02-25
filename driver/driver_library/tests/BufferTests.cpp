//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_driver_library/Buffer.hpp"
#include "../include/ethosn_driver_library/Network.hpp"
#include "../src/Utils.hpp"
#include <catch.hpp>

#include <cstring>
#include <iostream>

using namespace ethosn::driver_library;

TEST_CASE("SimpleBufferAllocation")
{
    uint32_t bufSize = 1000;

    // Create Simple buffer
    Buffer test_buffer(bufSize, DataFormat::NHWC);

    // Verify Buffer properties
    REQUIRE(test_buffer.GetSize() == bufSize);
    REQUIRE(test_buffer.GetDataFormat() == DataFormat::NHWC);
}

TEST_CASE("BufferSource")
{
    uint8_t test_src[] = "This is a test source data";

    // Create a buffer with test source data
    Buffer test_buffer(test_src, sizeof(test_src), DataFormat::NHWC);

    // Verify Buffer properties
    REQUIRE(test_buffer.GetSize() == sizeof(test_src));
    REQUIRE(test_buffer.GetDataFormat() == DataFormat::NHWC);
    REQUIRE(std::memcmp(test_buffer.GetMappedBuffer(), test_src, sizeof(test_src)) == 0);
}

TEST_CASE("BufferDescriptor")
{
    uint8_t test_src[] = "This is a test source data";
    uint32_t buf_size  = sizeof(test_src);
    ethosn::support_library::BufferInfo info(0, 1000, buf_size);

    // Create a buffer with test source data
    Buffer test_buffer(test_src, info.m_Size, DataFormat::NHWC);

    // Verify Buffer properties
    REQUIRE(test_buffer.GetSize() == buf_size);
    REQUIRE(test_buffer.GetDataFormat() == DataFormat::NHWC);
    REQUIRE(std::memcmp(test_buffer.GetMappedBuffer(), test_src, buf_size) == 0);
}
