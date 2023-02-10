//
// Copyright © 2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "TestUtils.hpp"
#include "cascading/PartUtils.hpp"
#include <ethosn_command_stream/cascading/CommandStream.hpp>

#include <catch.hpp>

using namespace ethosn::support_library;
using namespace ethosn::support_library::utils;
using namespace ethosn::command_stream::cascading;

TEST_CASE("CalculateTileSize/TileClamping")
{
    const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);

    // Tile is clamped to the tensor size when not using FCAF
    impl::TileSizeCalculation tile =
        impl::CalculateTileSize(caps, TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 16, 16, 16 },
                                PackedBoundaryThickness{ 0, 0, 0, 0 }, 2, false);
    // Height is rounded up to 24 (multiple of brick group size)
    CHECK(tile.slotSizeInBytes == 16 * 16 * 16);
    CHECK(tile.sizeInBytes == 24 * 16 * 16);

    // Tile is clamped less aggressively when using FCAF
    tile = impl::CalculateTileSize(caps, TensorShape{ 1, 16, 17, 16 }, TensorShape{ 1, 64, 64, 16 },
                                   PackedBoundaryThickness{ 0, 0, 0, 0 }, 2, true);
    // Width is rounded up to 32 (multiple of FCAF cell size)
    CHECK(tile.slotSizeInBytes == 64 * 64 * 16);
    CHECK(tile.sizeInBytes == 16 * 32 * 16);

    // Tile is not clamped at all when using packed boundary data
    tile = impl::CalculateTileSize(caps, TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 16, 16, 16 },
                                   PackedBoundaryThickness{ 0, 8, 0, 8 }, 2, false);
    CHECK(tile.slotSizeInBytes == (16 + 8 + 8) * 16 * 16);
    CHECK(tile.sizeInBytes == 2 * (16 + 8 + 8) * 16 * 16);

    // Slot size is rounded up when FCAF_WIDE could be used
    tile = impl::CalculateTileSize(caps, TensorShape{ 1, 100, 88, 100 }, TensorShape{ 1, 16, 88, 16 },
                                   PackedBoundaryThickness{ 0, 0, 0, 0 }, 2, true);
    CHECK(tile.slotSizeInBytes == 16 * 96 * 16);
    CHECK(tile.sizeInBytes == (16 * 96 * 16) * 2);

    // Slot size is not rounded up when FCAF_WIDE can't be used
    tile = impl::CalculateTileSize(caps, TensorShape{ 1, 100, 88, 100 }, TensorShape{ 1, 16, 88, 16 },
                                   PackedBoundaryThickness{ 0, 0, 0, 0 }, 2, false);
    CHECK(tile.slotSizeInBytes == 16 * 88 * 16);
    CHECK(tile.sizeInBytes == (16 * 88 * 16) * 2);

    // Slot size is not rounded up when FCAF_WIDE could be used, but it would require too much extra SRAM.
    // Instead the "forbidFcafWide" flag is set. This also means that the total tile size can be clamped
    // more aggressively (as no need to make space for FCAF_WIDE).
    tile = impl::CalculateTileSize(caps, TensorShape{ 1, 16, 8, 16 }, TensorShape{ 1, 16, 8, 16 },
                                   PackedBoundaryThickness{ 0, 0, 0, 0 }, 2, true);
    CHECK(tile.slotSizeInBytes == 16 * 8 * 16);
    CHECK(tile.sizeInBytes == 16 * 8 * 16);
    CHECK(tile.forbidFcafWide == true);
}
