//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/cascading/EstimationUtils.hpp"
#include "../src/cascading/MceEstimationUtils.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>

using namespace ethosn;
using namespace ethosn::support_library;
using namespace ethosn::command_stream;

namespace
{

bool CompareDouble(double a, double b)
{
    return std::abs(a - b) < std::numeric_limits<double>::epsilon();
}

}    // namespace

TEST_CASE("CalculateMetric only parallel dram", "[EstimationUtils]")
{
    PassPerformanceData pass        = {};
    NetworkPerformanceData perfData = {};

    // Make numbers large enough so metric is simple to reason about
    pass.m_Stats.m_Input.m_MemoryStats.m_DramParallel      = 30UL;
    pass.m_Stats.m_Weights.m_MemoryStats.m_DramParallel    = 36UL;
    pass.m_Stats.m_Output.m_MemoryStats.m_DramParallel     = 30UL;
    pass.m_Stats.m_Input.m_MemoryStats.m_DramNonParallel   = 0UL;
    pass.m_Stats.m_Weights.m_MemoryStats.m_DramNonParallel = 0UL;
    pass.m_Stats.m_Output.m_MemoryStats.m_DramNonParallel  = 0UL;
    pass.m_Stats.m_Mce.m_CycleCount                        = 0UL;

    perfData.m_Stream.push_back(pass);

    const double metric = CalculateMetric(perfData);

    CHECK(CompareDouble(metric, 10.0));
}

/// Test to make sure CalculateMetric accounts for that fact that Dram and Mce cycles can be done in parallel
TEST_CASE("CalculateMetric mce cycles > parallel dram", "[EstimationUtils]")
{
    PassPerformanceData pass        = {};
    NetworkPerformanceData perfData = {};

    // Make numbers large enough so metric is simple to reason about
    pass.m_Stats.m_Input.m_MemoryStats.m_DramParallel      = 30UL;
    pass.m_Stats.m_Weights.m_MemoryStats.m_DramParallel    = 36UL;
    pass.m_Stats.m_Output.m_MemoryStats.m_DramParallel     = 30UL;
    pass.m_Stats.m_Input.m_MemoryStats.m_DramNonParallel   = 0UL;
    pass.m_Stats.m_Weights.m_MemoryStats.m_DramNonParallel = 0UL;
    pass.m_Stats.m_Output.m_MemoryStats.m_DramNonParallel  = 0UL;
    pass.m_Stats.m_Mce.m_CycleCount                        = 20UL;

    perfData.m_Stream.push_back(pass);

    const double metric = CalculateMetric(perfData);

    CHECK(CompareDouble(metric, 20.0));
}

/// Test to make sure CalculateMetric accounts for that fact that non parallel dram is a bottleneck
TEST_CASE("CalculateMetric non parallel", "[EstimationUtils]")
{
    PassPerformanceData pass        = {};
    NetworkPerformanceData perfData = {};

    // Make numbers large enough so metric is simple to reason about
    pass.m_Stats.m_Input.m_MemoryStats.m_DramParallel      = 30UL;
    pass.m_Stats.m_Weights.m_MemoryStats.m_DramParallel    = 36UL;
    pass.m_Stats.m_Output.m_MemoryStats.m_DramParallel     = 30UL;
    pass.m_Stats.m_Input.m_MemoryStats.m_DramNonParallel   = 120UL;
    pass.m_Stats.m_Weights.m_MemoryStats.m_DramNonParallel = 144L;
    pass.m_Stats.m_Output.m_MemoryStats.m_DramNonParallel  = 120UL;
    pass.m_Stats.m_Mce.m_CycleCount                        = 20UL;

    perfData.m_Stream.push_back(pass);

    const double metric = CalculateMetric(perfData);

    CHECK(CompareDouble(metric, 60.0));
}

TEST_CASE("GetMceStats upsampled", "[EstimationUtils]")
{
    HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
    MceStats stats            = GetMceStats(caps, Stride(1, 1), command_stream::MceOperation::CONVOLUTION,
                                 CompilerMceAlgorithm::Direct, TensorShape{ 1, 16, 16, 16 },
                                 TensorShape{ 1, 32, 32, 16 }, TensorShape{ 1, 1, 16, 16 }, BlockConfig{ 8u, 8u });

    // The upsampled IFM is 32x32, and there are 16 IFM channels and 16 OFM channels
    CHECK(stats.m_Operations == 2 * 32 * 32 * 16 * 16);
    // 4 TOPS can do 16 (num IGs) * 16 (num OGs) * 8 (num MACs) per cycle
    CHECK(stats.m_CycleCount == (32 * 32 * 16 * 16) / (16 * 16 * 8));
}

TEST_CASE("GetMceStats valid padding", "[EstimationUtils]")
{
    HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
    MceStats stats            = GetMceStats(caps, Stride(1, 1), command_stream::MceOperation::CONVOLUTION,
                                 CompilerMceAlgorithm::Direct, TensorShape{ 1, 10, 10, 16 }, TensorShape{ 1, 2, 2, 16 },
                                 TensorShape{ 9, 9, 16, 16 }, BlockConfig{ 8u, 8u });

    // The OFM is bigger than the IFM and there are only 2 x 2 XY elements to calculate
    // There are 16 IFM channels and 16 OFM channels, and 9x9 kernel elements
    CHECK(stats.m_Operations == 2 * 2 * 2 * 16 * 16 * 9 * 9);
}

TEST_CASE("GetMceStats fully connected", "[EstimationUtils]")
{
    // 1024 channels in, 16 channels out
    HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
    MceStats stats            = GetMceStats(caps, Stride(1, 1), command_stream::MceOperation::FULLY_CONNECTED,
                                 CompilerMceAlgorithm::Direct, TensorShape{ 1, 8, 8, 16 }, TensorShape{ 1, 1, 1, 16 },
                                 TensorShape{ 1, 1, 1024, 16 }, BlockConfig{ 8u, 8u });

    CHECK(stats.m_Operations == 2 * 1024 * 16);
}
