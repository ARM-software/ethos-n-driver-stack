//
// Copyright © 2018-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/SramAllocator.hpp"

#include <catch.hpp>

using namespace ethosn::support_library;

TEST_CASE("SramAllocator: Allocate")
{
    SramAllocator sram(10);

    std::pair<bool, uint32_t> res = sram.Allocate(5);
    REQUIRE(res.first == true);
    REQUIRE(res.second == 0);
}

TEST_CASE("SramAllocator: Allocate prefer end")
{
    SramAllocator sram(10);

    std::pair<bool, uint32_t> res = sram.Allocate(3, AllocationPreference::End);
    REQUIRE(res.first == true);
    REQUIRE(res.second == 7);
}

TEST_CASE("SramAllocator: Allocate prefer end full")
{
    SramAllocator sram(10);

    auto res0 = sram.Allocate(6, AllocationPreference::End);
    REQUIRE(res0.first == true);
    REQUIRE(res0.second == 4);
    auto res1 = sram.Allocate(4, AllocationPreference::End);
    REQUIRE(res1.first == true);
    REQUIRE(res1.second == 0);
}

TEST_CASE("SramAllocator: Allocate prefer end Fail")
{
    SramAllocator sram(10);

    std::pair<bool, uint32_t> res = sram.Allocate(3, AllocationPreference::End);
    REQUIRE(res.first == true);
    REQUIRE(res.second == 7);
    res = sram.Allocate(3, AllocationPreference::Start);
    REQUIRE(res.first == true);
    REQUIRE(res.second == 0);
    res = sram.Allocate(3, AllocationPreference::End);
    REQUIRE(res.first == true);
    REQUIRE(res.second == 4);
    res = sram.Allocate(1, AllocationPreference::End);
    REQUIRE(res.first == true);
    REQUIRE(res.second == 3);

    res = sram.Allocate(1, AllocationPreference::End);
    REQUIRE(res.first == false);
    REQUIRE(res.second == 0);
}

TEST_CASE("SramAllocator: Allocate prefer end free")
{
    SramAllocator sram(10);

    auto res0 = sram.Allocate(3, AllocationPreference::End);
    REQUIRE(res0.first == true);
    REQUIRE(res0.second == 7);
    auto res1 = sram.Allocate(3, AllocationPreference::Start);
    REQUIRE(res1.first == true);
    REQUIRE(res1.second == 0);
    auto res2 = sram.Allocate(3, AllocationPreference::End);
    REQUIRE(res2.first == true);
    REQUIRE(res2.second == 4);

    bool resFree = sram.TryFree(res0.second);
    REQUIRE(resFree == true);
}

TEST_CASE("SramAllocator: Fail Allocate")
{
    SramAllocator sram(10);

    std::pair<bool, uint32_t> res = sram.Allocate(5);
    REQUIRE(res.first == true);
    REQUIRE(res.second == 0);

    res = sram.Allocate(5);
    REQUIRE(res.first == true);
    REQUIRE(res.second == 5);

    res = sram.Allocate(1);
    REQUIRE(res.first == false);
}

TEST_CASE("SramAllocator: Free")
{
    SramAllocator sram(10);

    std::pair<bool, uint32_t> res = sram.Allocate(5);
    REQUIRE(res.first == true);
    REQUIRE(res.second == 0);

    res = sram.Allocate(5);
    REQUIRE(res.first == true);
    REQUIRE(res.second == 5);

    auto test = sram.TryFree(res.second);
    REQUIRE(test == true);
}

TEST_CASE("SramAllocator: Fail Free")
{
    SramAllocator sram(10);

    auto res = sram.TryFree(0);
    REQUIRE(res == false);
}

TEST_CASE("SramAllocator: Fail Double Free")
{
    SramAllocator sram(10);

    std::pair<bool, uint32_t> res = sram.Allocate(5);
    REQUIRE(res.first == true);
    REQUIRE(res.second == 0);

    res = sram.Allocate(5);
    REQUIRE(res.first == true);
    REQUIRE(res.second == 5);

    auto test = sram.TryFree(res.second);
    REQUIRE(test == true);

    test = sram.TryFree(res.second);
    REQUIRE(test == false);
}

TEST_CASE("SramAllocator: Allocate Free Allocate")
{
    SramAllocator sram(10);

    std::pair<bool, uint32_t> res = sram.Allocate(5);
    REQUIRE(res.first == true);
    REQUIRE(res.second == 0);

    res = sram.Allocate(5);
    REQUIRE(res.first == true);
    REQUIRE(res.second == 5);

    auto test = sram.TryFree(res.second);
    REQUIRE(test == true);

    res = sram.Allocate(5);
    REQUIRE(res.first == true);
    REQUIRE(res.second == 5);
}

TEST_CASE("SramAllocator: Allocate between blocks")
{
    SramAllocator sram(10);

    std::pair<bool, uint32_t> res0 = sram.Allocate(3);
    REQUIRE(res0.first == true);
    REQUIRE(res0.second == 0);

    std::pair<bool, uint32_t> res1 = sram.Allocate(3);
    REQUIRE(res1.first == true);
    REQUIRE(res1.second == 3);

    std::pair<bool, uint32_t> res2 = sram.Allocate(3);
    REQUIRE(res2.first == true);
    REQUIRE(res2.second == 6);

    auto test = sram.TryFree(res1.second);
    REQUIRE(test == true);

    std::pair<bool, uint32_t> res = sram.Allocate(3);
    REQUIRE(res.first == true);
    REQUIRE(res.second == 3);
}

TEST_CASE("SramAllocator: Reset")
{
    SramAllocator sram(10);

    std::pair<bool, uint32_t> res = sram.Allocate(3);
    REQUIRE(res.first == true);
    REQUIRE(res.second == 0);

    sram.Reset();
    auto freeResult = sram.TryFree(res.second);
    REQUIRE(freeResult == false);

    res = sram.Allocate(10);
    REQUIRE(res.first == true);
    REQUIRE(res.second == 0);
}
