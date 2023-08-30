//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/EstimationUtils.hpp"
#include "../src/MceEstimationUtils.hpp"
#include "TestUtils.hpp"

#include <catch.hpp>

using namespace ethosn;
using namespace ethosn::support_library;
using namespace ethosn::command_stream;

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
