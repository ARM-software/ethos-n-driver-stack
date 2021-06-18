//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/cascading/EstimationUtils.hpp"

#include <catch.hpp>

using namespace ethosn::support_library;

/// Tests method ComparePerformanceData
TEST_CASE("Right more performant than left (less total dram bandwidth)", "[EstimationUtils]")
{
    PassPerformanceData pass             = {};
    NetworkPerformanceData perfDataLeft  = {};
    NetworkPerformanceData perfDataRight = {};

    pass.m_Stats.m_Input.m_MemoryStats.m_DramParallel      = 2UL;
    pass.m_Stats.m_Weights.m_MemoryStats.m_DramParallel    = 2UL;
    pass.m_Stats.m_Output.m_MemoryStats.m_DramParallel     = 2UL;
    pass.m_Stats.m_Input.m_MemoryStats.m_DramNonParallel   = 1UL;
    pass.m_Stats.m_Weights.m_MemoryStats.m_DramNonParallel = 2UL;
    pass.m_Stats.m_Output.m_MemoryStats.m_DramNonParallel  = 1UL;

    // Add (6,4)
    perfDataLeft.m_Stream.push_back(pass);

    pass.m_Stats.m_Input.m_MemoryStats.m_DramParallel      = 2UL;
    pass.m_Stats.m_Weights.m_MemoryStats.m_DramParallel    = 3UL;
    pass.m_Stats.m_Output.m_MemoryStats.m_DramParallel     = 2UL;
    pass.m_Stats.m_Input.m_MemoryStats.m_DramNonParallel   = 2UL;
    pass.m_Stats.m_Weights.m_MemoryStats.m_DramNonParallel = 1UL;
    pass.m_Stats.m_Output.m_MemoryStats.m_DramNonParallel  = 1UL;
    // Add (7,4)
    perfDataLeft.m_Stream.push_back(pass);

    // Add (7,4)
    perfDataRight.m_Stream.push_back(pass);
    pass.m_Stats.m_Input.m_MemoryStats.m_DramParallel      = 2UL;
    pass.m_Stats.m_Weights.m_MemoryStats.m_DramParallel    = 1UL;
    pass.m_Stats.m_Output.m_MemoryStats.m_DramParallel     = 1UL;
    pass.m_Stats.m_Input.m_MemoryStats.m_DramNonParallel   = 2UL;
    pass.m_Stats.m_Weights.m_MemoryStats.m_DramNonParallel = 1UL;
    pass.m_Stats.m_Output.m_MemoryStats.m_DramNonParallel  = 1UL;
    // Add (4,4)
    perfDataRight.m_Stream.push_back(pass);

    REQUIRE(perfDataLeft.m_Stream.size() == 2UL);
    REQUIRE(GetPerformanceTotalDataMetric(perfDataLeft) == 21UL);
    REQUIRE(GetPerformanceParallelDataMetric(perfDataLeft) == 13UL);
    REQUIRE(GetPerformanceNonParallelDataMetric(perfDataLeft) == 8UL);
    REQUIRE(GetPerformanceNumberOfPassesMetric(perfDataLeft) == 2UL);

    REQUIRE(GetPerformanceTotalDataMetric(perfDataRight) == 19UL);
    REQUIRE(GetPerformanceParallelDataMetric(perfDataRight) == 11UL);
    REQUIRE(GetPerformanceNonParallelDataMetric(perfDataRight) == 8UL);
    REQUIRE(GetPerformanceNumberOfPassesMetric(perfDataRight) == 2UL);

    // Right is more performant than left for total dram
    REQUIRE(ComparePerformanceData(perfDataLeft, perfDataRight) == PerformanceComparisonResult::RightBetter);
}

/// Tests method ComparePerformanceData
TEST_CASE("Left more performant than right (same total dram bandwidth but less non-parallel)", "[EstimationUtils]")
{
    PassPerformanceData pass             = {};
    NetworkPerformanceData perfDataLeft  = {};
    NetworkPerformanceData perfDataRight = {};

    pass.m_Stats.m_Input.m_MemoryStats.m_DramParallel      = 2UL;
    pass.m_Stats.m_Weights.m_MemoryStats.m_DramParallel    = 3UL;
    pass.m_Stats.m_Output.m_MemoryStats.m_DramParallel     = 4UL;
    pass.m_Stats.m_Input.m_MemoryStats.m_DramNonParallel   = 2UL;
    pass.m_Stats.m_Weights.m_MemoryStats.m_DramNonParallel = 1UL;
    pass.m_Stats.m_Output.m_MemoryStats.m_DramNonParallel  = 1UL;
    // Add (9,4)
    perfDataLeft.m_Stream.push_back(pass);

    pass.m_Stats.m_Input.m_MemoryStats.m_DramParallel      = 4UL;
    pass.m_Stats.m_Weights.m_MemoryStats.m_DramParallel    = 3UL;
    pass.m_Stats.m_Output.m_MemoryStats.m_DramParallel     = 1UL;
    pass.m_Stats.m_Input.m_MemoryStats.m_DramNonParallel   = 1UL;
    pass.m_Stats.m_Weights.m_MemoryStats.m_DramNonParallel = 2UL;
    pass.m_Stats.m_Output.m_MemoryStats.m_DramNonParallel  = 2UL;
    // Add (8,5)
    perfDataRight.m_Stream.push_back(pass);

    REQUIRE(GetPerformanceTotalDataMetric(perfDataLeft) == 13UL);
    REQUIRE(GetPerformanceParallelDataMetric(perfDataLeft) == 9UL);
    REQUIRE(GetPerformanceNonParallelDataMetric(perfDataLeft) == 4UL);
    REQUIRE(GetPerformanceNumberOfPassesMetric(perfDataLeft) == 1UL);

    REQUIRE(GetPerformanceTotalDataMetric(perfDataRight) == 13UL);
    REQUIRE(GetPerformanceParallelDataMetric(perfDataRight) == 8UL);
    REQUIRE(GetPerformanceNonParallelDataMetric(perfDataRight) == 5UL);
    REQUIRE(GetPerformanceNumberOfPassesMetric(perfDataRight) == 1UL);

    // Left is more performant than right for non parallel dram
    REQUIRE(ComparePerformanceData(perfDataLeft, perfDataRight) == PerformanceComparisonResult::LeftBetter);
}

TEST_CASE("Left more pass data performant than right")
/// Tests method ComparePerformanceData
{
    PassPerformanceData pass             = {};
    NetworkPerformanceData perfDataLeft  = {};
    NetworkPerformanceData perfDataRight = {};

    pass.m_Stats.m_Input.m_MemoryStats.m_DramParallel      = 2UL;
    pass.m_Stats.m_Weights.m_MemoryStats.m_DramParallel    = 3UL;
    pass.m_Stats.m_Output.m_MemoryStats.m_DramParallel     = 4UL;
    pass.m_Stats.m_Input.m_MemoryStats.m_DramNonParallel   = 2UL;
    pass.m_Stats.m_Weights.m_MemoryStats.m_DramNonParallel = 1UL;
    pass.m_Stats.m_Output.m_MemoryStats.m_DramNonParallel  = 1UL;
    // Add (9,4)
    perfDataLeft.m_Stream.push_back(pass);

    pass.m_Stats.m_Input.m_MemoryStats.m_DramParallel      = 2UL;
    pass.m_Stats.m_Weights.m_MemoryStats.m_DramParallel    = 2UL;
    pass.m_Stats.m_Output.m_MemoryStats.m_DramParallel     = 1UL;
    pass.m_Stats.m_Input.m_MemoryStats.m_DramNonParallel   = 0UL;
    pass.m_Stats.m_Weights.m_MemoryStats.m_DramNonParallel = 1UL;
    pass.m_Stats.m_Output.m_MemoryStats.m_DramNonParallel  = 1UL;
    // Add (5,2)
    perfDataRight.m_Stream.push_back(pass);

    pass.m_Stats.m_Input.m_MemoryStats.m_DramParallel      = 2UL;
    pass.m_Stats.m_Weights.m_MemoryStats.m_DramParallel    = 2UL;
    pass.m_Stats.m_Output.m_MemoryStats.m_DramParallel     = 0UL;
    pass.m_Stats.m_Input.m_MemoryStats.m_DramNonParallel   = 1UL;
    pass.m_Stats.m_Weights.m_MemoryStats.m_DramNonParallel = 0UL;
    pass.m_Stats.m_Output.m_MemoryStats.m_DramNonParallel  = 1UL;
    // Add (4,2)
    perfDataRight.m_Stream.push_back(pass);

    REQUIRE(GetPerformanceTotalDataMetric(perfDataLeft) == 13UL);
    REQUIRE(GetPerformanceParallelDataMetric(perfDataLeft) == 9UL);
    REQUIRE(GetPerformanceNonParallelDataMetric(perfDataLeft) == 4UL);
    REQUIRE(GetPerformanceNumberOfPassesMetric(perfDataLeft) == 1UL);

    REQUIRE(GetPerformanceTotalDataMetric(perfDataRight) == 13UL);
    REQUIRE(GetPerformanceParallelDataMetric(perfDataRight) == 9UL);
    REQUIRE(GetPerformanceNonParallelDataMetric(perfDataRight) == 4UL);
    REQUIRE(GetPerformanceNumberOfPassesMetric(perfDataRight) == 2UL);

    // Left is more performant than right for number of passes
    REQUIRE(ComparePerformanceData(perfDataLeft, perfDataRight) == PerformanceComparisonResult::LeftBetter);
}
