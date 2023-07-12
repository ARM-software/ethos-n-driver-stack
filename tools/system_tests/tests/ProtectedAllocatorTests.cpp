//
// Copyright © 2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../ProtectedAllocator.hpp"

#include <catch.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>

namespace
{
constexpr size_t mebiByte = 1024U * 1024U;
}

namespace ethosn
{
namespace system_tests
{

TEST_CASE("ProtectedAllocator: Buffer allocate, populate and retrieve data", "[TZMP1-Test-Module]")
{
    std::array<uint8_t, mebiByte> testData;

    // Generate test data for the buffer testing
    std::mt19937 randomGenerator(std::random_device{}());
    std::uniform_int_distribution<uint8_t> distrib;
    std::generate(testData.begin(), testData.end(), [&]() { return distrib(randomGenerator); });

    ProtectedAllocator protAlloc;
    REQUIRE(protAlloc.GetMemorySourceType() == armnn::MemorySource::DmaBufProtected);

    void* dmaBufHandle = protAlloc.allocate(mebiByte, 0);
    protAlloc.PopulateData(dmaBufHandle, testData.data(), testData.size());

    std::array<uint8_t, mebiByte> readData{};
    protAlloc.RetrieveData(dmaBufHandle, readData.data(), readData.size());

    protAlloc.free(dmaBufHandle);

    REQUIRE(readData == testData);
}

TEST_CASE("ProtectedAllocator: Zero size allocation throws", "[TZMP1-Test-Module]")
{
    ProtectedAllocator protAlloc;
    REQUIRE_THROWS(protAlloc.allocate(0U, 0U));
}

TEST_CASE("ProtectedAllocator: Double free throws", "[TZMP1-Test-Module]")
{
    ProtectedAllocator protAlloc;

    void* dmaBufHandle = protAlloc.allocate(mebiByte, 0);
    protAlloc.free(dmaBufHandle);

    REQUIRE_THROWS(protAlloc.free(dmaBufHandle));
}

TEST_CASE("ProtectedAllocator: Invalid buffer handle ptr throws", "[TZMP1-Test-Module]")
{
    ProtectedAllocator protAlloc;
    std::array<uint8_t, mebiByte> testData;

    CHECK_THROWS(protAlloc.RetrieveData(nullptr, testData.data(), testData.size()));
    CHECK_THROWS(protAlloc.PopulateData(nullptr, testData.data(), testData.size()));
    CHECK_THROWS(protAlloc.free(nullptr));

    int invalidHandle = 5;
    CHECK_THROWS(protAlloc.PopulateData(&invalidHandle, testData.data(), testData.size()));
    CHECK_THROWS(protAlloc.RetrieveData(&invalidHandle, testData.data(), testData.size()));
    CHECK_THROWS(protAlloc.free(&invalidHandle));
}

TEST_CASE("ProtectedAllocator: Invalid data ptr or zero length data throws", "[TZMP1-Test-Module]")
{
    ProtectedAllocator protAlloc;
    std::array<uint8_t, mebiByte> testData{};

    void* dmaBufHandle = protAlloc.allocate(mebiByte, 0);

    CHECK_THROWS(protAlloc.PopulateData(dmaBufHandle, nullptr, testData.size()));
    CHECK_THROWS(protAlloc.RetrieveData(dmaBufHandle, nullptr, testData.size()));
    CHECK_THROWS(protAlloc.PopulateData(dmaBufHandle, testData.data(), 0U));
    CHECK_THROWS(protAlloc.RetrieveData(dmaBufHandle, testData.data(), 0U));

    protAlloc.free(dmaBufHandle);
}

}    // namespace system_tests
}    // namespace ethosn
