//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/cascading/EstimationUtils.hpp"
#include "../src/cascading/MceEstimationUtils.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>

using namespace ethosn;
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
    pass.m_Stats.m_Mce.m_CycleCount                        = 1UL;

    // Add (6,4,1)
    perfDataLeft.m_Stream.push_back(pass);

    pass.m_Stats.m_Input.m_MemoryStats.m_DramParallel      = 2UL;
    pass.m_Stats.m_Weights.m_MemoryStats.m_DramParallel    = 3UL;
    pass.m_Stats.m_Output.m_MemoryStats.m_DramParallel     = 2UL;
    pass.m_Stats.m_Input.m_MemoryStats.m_DramNonParallel   = 2UL;
    pass.m_Stats.m_Weights.m_MemoryStats.m_DramNonParallel = 1UL;
    pass.m_Stats.m_Output.m_MemoryStats.m_DramNonParallel  = 1UL;
    pass.m_Stats.m_Mce.m_CycleCount                        = 2UL;
    // Add (7,4,2)
    perfDataLeft.m_Stream.push_back(pass);

    // Add (7,4,2)
    perfDataRight.m_Stream.push_back(pass);
    pass.m_Stats.m_Input.m_MemoryStats.m_DramParallel      = 2UL;
    pass.m_Stats.m_Weights.m_MemoryStats.m_DramParallel    = 1UL;
    pass.m_Stats.m_Output.m_MemoryStats.m_DramParallel     = 1UL;
    pass.m_Stats.m_Input.m_MemoryStats.m_DramNonParallel   = 2UL;
    pass.m_Stats.m_Weights.m_MemoryStats.m_DramNonParallel = 1UL;
    pass.m_Stats.m_Output.m_MemoryStats.m_DramNonParallel  = 1UL;
    pass.m_Stats.m_Mce.m_CycleCount                        = 3UL;
    // Add (4,4,3)
    perfDataRight.m_Stream.push_back(pass);

    REQUIRE(perfDataLeft.m_Stream.size() == 2UL);
    REQUIRE(GetPerformanceTotalDataMetric(perfDataLeft) == 21UL);
    REQUIRE(GetPerformanceParallelDataMetric(perfDataLeft) == 13UL);
    REQUIRE(GetPerformanceNonParallelDataMetric(perfDataLeft) == 8UL);
    REQUIRE(GetPerformanceMceCycleCountMetric(perfDataLeft) == 3UL);

    REQUIRE(GetPerformanceTotalDataMetric(perfDataRight) == 19UL);
    REQUIRE(GetPerformanceParallelDataMetric(perfDataRight) == 11UL);
    REQUIRE(GetPerformanceNonParallelDataMetric(perfDataRight) == 8UL);
    REQUIRE(GetPerformanceMceCycleCountMetric(perfDataRight) == 5UL);

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
    pass.m_Stats.m_Mce.m_CycleCount                        = 1UL;
    // Add (9,4,1)
    perfDataLeft.m_Stream.push_back(pass);

    pass.m_Stats.m_Input.m_MemoryStats.m_DramParallel      = 4UL;
    pass.m_Stats.m_Weights.m_MemoryStats.m_DramParallel    = 3UL;
    pass.m_Stats.m_Output.m_MemoryStats.m_DramParallel     = 1UL;
    pass.m_Stats.m_Input.m_MemoryStats.m_DramNonParallel   = 1UL;
    pass.m_Stats.m_Weights.m_MemoryStats.m_DramNonParallel = 2UL;
    pass.m_Stats.m_Output.m_MemoryStats.m_DramNonParallel  = 2UL;
    pass.m_Stats.m_Mce.m_CycleCount                        = 2UL;
    // Add (8,5,2)
    perfDataRight.m_Stream.push_back(pass);

    REQUIRE(GetPerformanceTotalDataMetric(perfDataLeft) == 13UL);
    REQUIRE(GetPerformanceParallelDataMetric(perfDataLeft) == 9UL);
    REQUIRE(GetPerformanceNonParallelDataMetric(perfDataLeft) == 4UL);
    REQUIRE(GetPerformanceMceCycleCountMetric(perfDataLeft) == 1UL);

    REQUIRE(GetPerformanceTotalDataMetric(perfDataRight) == 13UL);
    REQUIRE(GetPerformanceParallelDataMetric(perfDataRight) == 8UL);
    REQUIRE(GetPerformanceNonParallelDataMetric(perfDataRight) == 5UL);
    REQUIRE(GetPerformanceMceCycleCountMetric(perfDataRight) == 2UL);

    // Left is more performant than right for non parallel dram
    REQUIRE(ComparePerformanceData(perfDataLeft, perfDataRight) == PerformanceComparisonResult::LeftBetter);
}

/// Tests method ComparePerformanceData
TEST_CASE("Left more performant than right (same dram bandwidths but less MCE cycles)", "[EstimationUtils]")
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
    pass.m_Stats.m_Mce.m_CycleCount                        = 1UL;
    // Add (9,4,1)
    perfDataLeft.m_Stream.push_back(pass);

    pass.m_Stats.m_Input.m_MemoryStats.m_DramParallel      = 2UL;
    pass.m_Stats.m_Weights.m_MemoryStats.m_DramParallel    = 2UL;
    pass.m_Stats.m_Output.m_MemoryStats.m_DramParallel     = 1UL;
    pass.m_Stats.m_Input.m_MemoryStats.m_DramNonParallel   = 0UL;
    pass.m_Stats.m_Weights.m_MemoryStats.m_DramNonParallel = 1UL;
    pass.m_Stats.m_Output.m_MemoryStats.m_DramNonParallel  = 1UL;
    pass.m_Stats.m_Mce.m_CycleCount                        = 2UL;
    // Add (5,2,2)
    perfDataRight.m_Stream.push_back(pass);

    pass.m_Stats.m_Input.m_MemoryStats.m_DramParallel      = 2UL;
    pass.m_Stats.m_Weights.m_MemoryStats.m_DramParallel    = 2UL;
    pass.m_Stats.m_Output.m_MemoryStats.m_DramParallel     = 0UL;
    pass.m_Stats.m_Input.m_MemoryStats.m_DramNonParallel   = 1UL;
    pass.m_Stats.m_Weights.m_MemoryStats.m_DramNonParallel = 0UL;
    pass.m_Stats.m_Output.m_MemoryStats.m_DramNonParallel  = 1UL;
    pass.m_Stats.m_Mce.m_CycleCount                        = 3UL;
    // Add (4,2,3)
    perfDataRight.m_Stream.push_back(pass);

    REQUIRE(GetPerformanceTotalDataMetric(perfDataLeft) == 13UL);
    REQUIRE(GetPerformanceParallelDataMetric(perfDataLeft) == 9UL);
    REQUIRE(GetPerformanceNonParallelDataMetric(perfDataLeft) == 4UL);
    REQUIRE(GetPerformanceMceCycleCountMetric(perfDataLeft) == 1UL);

    REQUIRE(GetPerformanceTotalDataMetric(perfDataRight) == 13UL);
    REQUIRE(GetPerformanceParallelDataMetric(perfDataRight) == 9UL);
    REQUIRE(GetPerformanceNonParallelDataMetric(perfDataRight) == 4UL);
    REQUIRE(GetPerformanceMceCycleCountMetric(perfDataRight) == 5UL);

    // Left is more performant than right for number of passes
    REQUIRE(ComparePerformanceData(perfDataLeft, perfDataRight) == PerformanceComparisonResult::LeftBetter);
}

TEST_CASE("GetMceStats upsampled", "[EstimationUtils]")
{
    HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
    MceStats stats =
        GetMceStats(caps, Stride(1, 1), command_stream::MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
                    TensorShape{ 1, 16, 16, 16 }, TensorShape{ 1, 32, 32, 16 }, TensorShape{ 1, 1, 16, 16 });

    // The upsampled IFM is 32x32, and there are 16 IFM channels and 16 OFM channels
    CHECK(stats.m_Operations == 2 * 32 * 32 * 16 * 16);
    // 4 TOPS can do 16 (num IGs) * 16 (num OGs) * 8 (num MACs) per cycle
    CHECK(stats.m_CycleCount == (32 * 32 * 16 * 16) / (16 * 16 * 8));
}

TEST_CASE("GetMceStats valid padding", "[EstimationUtils]")
{
    HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
    MceStats stats =
        GetMceStats(caps, Stride(1, 1), command_stream::MceOperation::CONVOLUTION, CompilerMceAlgorithm::Direct,
                    TensorShape{ 1, 10, 10, 16 }, TensorShape{ 1, 2, 2, 16 }, TensorShape{ 9, 9, 16, 16 });

    // The OFM is bigger than the IFM and there are only 2 x 2 XY elements to calculate
    // There are 16 IFM channels and 16 OFM channels, and 9x9 kernel elements
    CHECK(stats.m_Operations == 2 * 2 * 2 * 16 * 16 * 9 * 9);
}

TEST_CASE("GetMceStats fully connected", "[EstimationUtils]")
{
    // 1024 channels in, 16 channels out
    HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
    MceStats stats =
        GetMceStats(caps, Stride(1, 1), command_stream::MceOperation::FULLY_CONNECTED, CompilerMceAlgorithm::Direct,
                    TensorShape{ 1, 8, 8, 16 }, TensorShape{ 1, 1, 1, 16 }, TensorShape{ 1, 1, 1024, 16 });

    CHECK(stats.m_Operations == 2 * 1024 * 16);
}
