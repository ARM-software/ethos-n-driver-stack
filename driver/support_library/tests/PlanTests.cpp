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
    {
        Plan planASram;
        SramBuffer* buffer    = planASram.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
        buffer->m_Format      = CascadingBufferFormat::NHWCB;
        buffer->m_TensorShape = TensorShape();
        buffer->m_StripeShape = TensorShape{ 1, 8, 8, 16 };
        buffer->m_Order       = TraversalOrder::Xyz;
        buffer->m_SizeInBytes = 4 * 8 * 8 * 16;

        planASram.m_OutputMappings[buffer] = PartOutputSlot{ 0, 0 };

        REQUIRE(GetTotSizeInBytes(planASram).m_Tot == 4 * 8 * 8 * 16);
        REQUIRE(GetInputsSizeInBytes(planASram).m_Tot == 0);
    }

    {
        Plan planBSram;
        SramBuffer* buffer                = planBSram.m_OpGraph.AddBuffer(std::make_unique<SramBuffer>());
        buffer->m_Format                  = CascadingBufferFormat::NHWCB;
        buffer->m_TensorShape             = TensorShape();
        buffer->m_StripeShape             = TensorShape{ 1, 8, 8, 8 };
        buffer->m_Order                   = TraversalOrder::Xyz;
        buffer->m_SizeInBytes             = 4 * 8 * 8 * 8;
        planBSram.m_InputMappings[buffer] = PartInputSlot{ 0, 0 };

        REQUIRE(GetTotSizeInBytes(planBSram).m_Tot == 4 * 8 * 8 * 8);
        REQUIRE(GetInputsSizeInBytes(planBSram).m_Tot == 4 * 8 * 8 * 8);
    }
}
