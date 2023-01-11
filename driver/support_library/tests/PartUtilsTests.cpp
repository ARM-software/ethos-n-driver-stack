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
    std::pair<uint32_t, uint32_t> slotSizeAndTotalSize =
        impl::CalculateTileSize(caps, TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 16, 16, 16 },
                                PackedBoundaryThickness{ 0, 0, 0, 0 }, 2, false);
    // Height is rounded up to 24 (multiple of brick group size)
    CHECK(slotSizeAndTotalSize == std::make_pair<uint32_t, uint32_t>(16 * 16 * 16, 24 * 16 * 16));

    // Tile is clamped less aggressively when using FCAF
    slotSizeAndTotalSize = impl::CalculateTileSize(caps, TensorShape{ 1, 16, 17, 16 }, TensorShape{ 1, 64, 64, 16 },
                                                   PackedBoundaryThickness{ 0, 0, 0, 0 }, 2, true);
    // Width is rounded up to 32 (multiple of FCAF cell size)
    CHECK(slotSizeAndTotalSize == std::make_pair<uint32_t, uint32_t>(64 * 64 * 16, 16 * 32 * 16));

    // Tile is not clamped at all when using packed boundary data
    slotSizeAndTotalSize = impl::CalculateTileSize(caps, TensorShape{ 1, 17, 16, 16 }, TensorShape{ 1, 16, 16, 16 },
                                                   PackedBoundaryThickness{ 0, 8, 0, 8 }, 2, false);
    CHECK(slotSizeAndTotalSize ==
          std::make_pair<uint32_t, uint32_t>((16 + 8 + 8) * 16 * 16, 2 * (16 + 8 + 8) * 16 * 16));
}
