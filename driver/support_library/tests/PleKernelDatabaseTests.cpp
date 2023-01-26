//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/cascading/PleKernelDatabase.hpp"

#include <catch.hpp>

using namespace ethosn::support_library;
using namespace ethosn::command_stream::cascading;

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
