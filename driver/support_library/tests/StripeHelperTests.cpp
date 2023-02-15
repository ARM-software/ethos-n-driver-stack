//
// Copyright © 2022-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/cascading/StripeHelper.hpp"

#include <catch.hpp>

using namespace ethosn::support_library;

TEST_CASE("StripeShapeLoop")
{
    auto compare = [](const impl::StripeShapeLoop& l, std::vector<uint32_t> expected) {
        std::vector<uint32_t> actual;
        for (uint32_t x : l)
        {
            actual.push_back(x);
        }
        CHECK(actual == expected);
    };

    compare(impl::StripeShapeLoop::Inclusive(8, 8), { 8 });
    compare(impl::StripeShapeLoop::Inclusive(32, 8), { 8, 16, 32 });
    compare(impl::StripeShapeLoop::Inclusive(48, 8), { 8, 16, 32, 48 });
    compare(impl::StripeShapeLoop::Inclusive(49, 8), { 8, 16, 32, 56 });
    compare(impl::StripeShapeLoop::Inclusive(47, 8), { 8, 16, 32, 48 });
    compare(impl::StripeShapeLoop::Inclusive(1, 8), { 8 });

    compare(impl::StripeShapeLoop::Exclusive(32, 8), { 8, 16 });
    compare(impl::StripeShapeLoop::Exclusive(48, 8), { 8, 16, 32 });
    compare(impl::StripeShapeLoop::Exclusive(49, 8), { 8, 16, 32 });
    compare(impl::StripeShapeLoop::Exclusive(47, 8), { 8, 16, 32 });
    compare(impl::StripeShapeLoop::Exclusive(65, 8), { 8, 16, 32, 64 });
    compare(impl::StripeShapeLoop::Exclusive(1, 8), {});
    compare(impl::StripeShapeLoop::Exclusive(8, 8), {});
}

TEST_CASE("IsSramBufferCompatibleWithDramBuffer")
{
    SramBuffer sram;
    DramBuffer dram;

    SECTION("Reshape without NHWC is invalid")
    {
        sram.m_TensorShape     = { 1, 16, 32, 16 };
        sram.m_StripeShape     = { 1, 16, 32, 16 };
        dram.m_Format          = CascadingBufferFormat::NHWCB;
        dram.m_TensorShape     = { 1, 16, 16, 32 };    // Reshaped from sram shape
        TensorShape dramOffset = { 0, 0, 0, 0 };
        // The order of the elements would not be correct, because of the NHWCB layout.
        CHECK(impl::IsSramBufferCompatibleWithDramBuffer(sram, dram, dramOffset) == false);
    }

    SECTION("Reshape with NHWC is OK")
    {
        sram.m_TensorShape     = { 1, 16, 32, 16 };
        sram.m_StripeShape     = { 1, 16, 32, 16 };
        dram.m_Format          = CascadingBufferFormat::NHWC;
        dram.m_TensorShape     = { 1, 16, 16, 32 };    // Reshaped from sram shape
        TensorShape dramOffset = { 0, 0, 0, 0 };
        // Because NHWC is linear, the order of the elements will be correct.
        CHECK(impl::IsSramBufferCompatibleWithDramBuffer(sram, dram, dramOffset) == true);
    }

    SECTION("Reshape with depth split in SRAM, but no depth split in DRAM is invalid")
    {
        sram.m_TensorShape     = { 1, 16, 16, 32 };
        sram.m_StripeShape     = { 1, 16, 16, 16 };
        dram.m_Format          = CascadingBufferFormat::NHWC;
        dram.m_TensorShape     = { 1, 16, 32, 16 };    // Reshaped from sram shape
        TensorShape dramOffset = { 0, 0, 0, 0 };
        // This is splitting the tensor in depth, as we use the SRAM tensor shape in the command
        // we send to the firmware.
        CHECK(impl::IsSramBufferCompatibleWithDramBuffer(sram, dram, dramOffset) == false);
    }

    SECTION("Reshape with no depth split in SRAM, but depth split in DRAM is valid")
    {
        sram.m_TensorShape     = { 1, 16, 32, 16 };
        sram.m_StripeShape     = { 1, 16, 16, 16 };
        dram.m_Format          = CascadingBufferFormat::NHWC;
        dram.m_TensorShape     = { 1, 16, 16, 32 };    // Reshaped from sram shape
        TensorShape dramOffset = { 0, 0, 0, 0 };
        // This is not splitting the tensor in depth, as we use the SRAM tensor shape in the command
        // we send to the firmware.
        CHECK(impl::IsSramBufferCompatibleWithDramBuffer(sram, dram, dramOffset) == true);
    }

    SECTION("DRAM offset must be aligned to the format's block size (NHWC)")
    {
        sram.m_TensorShape     = { 1, 16, 16, 32 };
        sram.m_StripeShape     = { 1, 16, 16, 32 };
        dram.m_Format          = CascadingBufferFormat::NHWC;
        dram.m_TensorShape     = { 1, 32, 32, 32 };
        TensorShape dramOffset = { 0, 1, 2, 0 };
        // Any offset in W or H is fine for NHWC
        CHECK(impl::IsSramBufferCompatibleWithDramBuffer(sram, dram, dramOffset) == true);

        dramOffset = { 0, 1, 2, 16 };
        // But C can never be offset
        CHECK(impl::IsSramBufferCompatibleWithDramBuffer(sram, dram, dramOffset) == false);
    }

    SECTION("DRAM offset must be aligned to the format's block size (NHWCB)")
    {
        sram.m_TensorShape     = { 1, 16, 16, 16 };
        sram.m_StripeShape     = { 1, 16, 16, 16 };
        dram.m_Format          = CascadingBufferFormat::NHWCB;
        dram.m_TensorShape     = { 1, 32, 32, 32 };
        TensorShape dramOffset = { 0, 8, 8, 16 };
        // This offset is a multiple of the brick group shape, so is OK
        CHECK(impl::IsSramBufferCompatibleWithDramBuffer(sram, dram, dramOffset) == true);

        // These ones aren't
        dramOffset = { 0, 7, 8, 16 };
        CHECK(impl::IsSramBufferCompatibleWithDramBuffer(sram, dram, dramOffset) == false);

        dramOffset = { 0, 8, 9, 16 };
        CHECK(impl::IsSramBufferCompatibleWithDramBuffer(sram, dram, dramOffset) == false);

        dramOffset = { 0, 8, 8, 13 };
        CHECK(impl::IsSramBufferCompatibleWithDramBuffer(sram, dram, dramOffset) == false);
    }

    SECTION("DRAM offset must be aligned to the format's block size (FCAF_WIDE)")
    {
        sram.m_TensorShape     = { 1, 16, 16, 16 };
        sram.m_StripeShape     = { 1, 16, 16, 16 };
        dram.m_Format          = CascadingBufferFormat::FCAF_WIDE;
        dram.m_TensorShape     = { 1, 32, 32, 32 };
        TensorShape dramOffset = { 0, 8, 16, 16 };
        // This offset is a multiple of the cell shape, so is OK
        CHECK(impl::IsSramBufferCompatibleWithDramBuffer(sram, dram, dramOffset) == true);

        // These ones aren't
        dramOffset = { 0, 7, 16, 16 };
        CHECK(impl::IsSramBufferCompatibleWithDramBuffer(sram, dram, dramOffset) == false);

        dramOffset = { 0, 8, 8, 16 };
        CHECK(impl::IsSramBufferCompatibleWithDramBuffer(sram, dram, dramOffset) == false);

        dramOffset = { 0, 8, 16, 8 };
        CHECK(impl::IsSramBufferCompatibleWithDramBuffer(sram, dram, dramOffset) == false);
    }

    SECTION("DRAM offset must be aligned to the format's block size (FCAF_DEEP)")
    {
        sram.m_TensorShape     = { 1, 16, 16, 32 };
        sram.m_StripeShape     = { 1, 16, 16, 32 };
        dram.m_Format          = CascadingBufferFormat::FCAF_DEEP;
        dram.m_TensorShape     = { 1, 32, 32, 64 };
        TensorShape dramOffset = { 0, 8, 8, 32 };
        // This offset is a multiple of the cell shape, so is OK
        CHECK(impl::IsSramBufferCompatibleWithDramBuffer(sram, dram, dramOffset) == true);

        // These ones aren't
        dramOffset = { 0, 7, 8, 32 };
        CHECK(impl::IsSramBufferCompatibleWithDramBuffer(sram, dram, dramOffset) == false);

        dramOffset = { 0, 8, 9, 32 };
        CHECK(impl::IsSramBufferCompatibleWithDramBuffer(sram, dram, dramOffset) == false);

        dramOffset = { 0, 8, 8, 16 };
        CHECK(impl::IsSramBufferCompatibleWithDramBuffer(sram, dram, dramOffset) == false);
    }

    SECTION("Subtensor does not need to end on an aligned boundary")
    {
        sram.m_TensorShape     = { 1, 16, 16, 15 };
        sram.m_StripeShape     = { 1, 16, 16, 32 };
        dram.m_Format          = CascadingBufferFormat::FCAF_DEEP;
        dram.m_TensorShape     = { 1, 32, 32, 64 };
        TensorShape dramOffset = { 0, 8, 8, 0 };
        // The tensor will end at channel 15, which isn't aligned to 32 (cell depth), but this is fine
        CHECK(impl::IsSramBufferCompatibleWithDramBuffer(sram, dram, dramOffset) == true);
    }

    SECTION("NHWC depth split is not allowed")
    {
        sram.m_TensorShape     = { 1, 16, 16, 32 };
        sram.m_StripeShape     = { 1, 16, 16, 16 };
        dram.m_Format          = CascadingBufferFormat::NHWC;
        dram.m_TensorShape     = { 1, 16, 16, 32 };
        TensorShape dramOffset = { 0, 0, 0, 0 };
        CHECK(impl::IsSramBufferCompatibleWithDramBuffer(sram, dram, dramOffset) == false);

        sram.m_StripeShape = { 1, 16, 16, 32 };
        CHECK(impl::IsSramBufferCompatibleWithDramBuffer(sram, dram, dramOffset) == true);
    }

    SECTION("FCAF_WIDE needs compatible stripe shape")
    {
        sram.m_TensorShape     = { 1, 16, 8, 32 };
        sram.m_StripeShape     = { 1, 8, 8, 16 };
        dram.m_Format          = CascadingBufferFormat::FCAF_WIDE;
        dram.m_TensorShape     = { 1, 16, 8, 32 };
        TensorShape dramOffset = { 0, 0, 0, 0 };
        // Stripe shape is 8 wide, not a multiple of 16. However this is fine because there
        // is only one stripe in the DRAM buffer in the W direction
        CHECK(impl::IsSramBufferCompatibleWithDramBuffer(sram, dram, dramOffset) == true);

        // But with > 1 stripe in the W direction, can't work
        dram.m_TensorShape = { 1, 16, 32, 32 };
        CHECK(impl::IsSramBufferCompatibleWithDramBuffer(sram, dram, dramOffset) == false);

        // Fix the stripe shape to be a multiple of 16 in W, this works even though there are >1 stripes in W.
        sram.m_StripeShape = { 1, 8, 16, 16 };
        CHECK(impl::IsSramBufferCompatibleWithDramBuffer(sram, dram, dramOffset) == true);
    }

    SECTION("FCAF_DEEP needs compatible stripe shape")
    {
        sram.m_TensorShape     = { 1, 16, 16, 16 };
        sram.m_StripeShape     = { 1, 8, 8, 16 };
        dram.m_Format          = CascadingBufferFormat::FCAF_DEEP;
        dram.m_TensorShape     = { 1, 16, 16, 16 };
        TensorShape dramOffset = { 0, 0, 0, 0 };
        // Stripe shape is only 16 deep, not a multiple of 32. However this is fine because
        // there is only one stripe in the DRAM buffer in the C direction.
        CHECK(impl::IsSramBufferCompatibleWithDramBuffer(sram, dram, dramOffset) == true);

        // But with >1 in the C direction, can't work
        dram.m_TensorShape = { 1, 16, 16, 64 };
        CHECK(impl::IsSramBufferCompatibleWithDramBuffer(sram, dram, dramOffset) == false);

        // Fix the stripe shape to be a multiple of 32 in C, this works even though there are >1 stripes in C.
        sram.m_StripeShape = { 1, 8, 8, 32 };
        CHECK(impl::IsSramBufferCompatibleWithDramBuffer(sram, dram, dramOffset) == true);
    }

    SECTION("PackedBoundaryData only for NHWCB")
    {
        sram.m_PackedBoundaryThickness = { 8, 0, 8, 0 };
        sram.m_TensorShape             = { 1, 16, 16, 32 };
        sram.m_StripeShape             = { 1, 8, 8, 32 };
        dram.m_Format                  = CascadingBufferFormat::FCAF_DEEP;
        dram.m_TensorShape             = { 1, 16, 16, 32 };
        TensorShape dramOffset         = { 0, 0, 0, 0 };
        CHECK(impl::IsSramBufferCompatibleWithDramBuffer(sram, dram, dramOffset) == false);

        dram.m_Format = CascadingBufferFormat::NHWCB;
        CHECK(impl::IsSramBufferCompatibleWithDramBuffer(sram, dram, dramOffset) == true);
    }
}
