//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/cascading/Plan.hpp"

#include <catch.hpp>

#include <fstream>

using namespace ethosn::support_library;
using namespace ethosn::command_stream::cascading;

TEST_CASE("Get size in bytes helpers")
{
    Buffer* buffer;
    Buffer tempBuffer;

    Plan planASram;
    tempBuffer = Buffer(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(), TensorShape{ 1, 8, 8, 16 },
                        TraversalOrder::Xyz, 4 * 8 * 8 * 16, QuantizationInfo());
    buffer     = planASram.m_OpGraph.AddBuffer(std::make_unique<Buffer>(std::move(tempBuffer)));
    planASram.m_OutputMappings[buffer] = PartOutputSlot{ 0, 0 };

    REQUIRE(GetTotSizeInBytes(planASram).m_Tot == 4 * 8 * 8 * 16);
    REQUIRE(GetInputsSizeInBytes(planASram).m_Tot == 0);

    Plan planBSram;
    tempBuffer = Buffer(Location::Sram, CascadingBufferFormat::NHWCB, TensorShape(), TensorShape{ 1, 8, 8, 8 },
                        TraversalOrder::Xyz, 4 * 8 * 8 * 8, QuantizationInfo());
    buffer     = planBSram.m_OpGraph.AddBuffer(std::make_unique<Buffer>(std::move(tempBuffer)));
    planBSram.m_InputMappings[buffer] = PartInputSlot{ 0, 0 };

    REQUIRE(GetTotSizeInBytes(planBSram).m_Tot == 4 * 8 * 8 * 8);
    REQUIRE(GetInputsSizeInBytes(planBSram).m_Tot == 4 * 8 * 8 * 8);
}
