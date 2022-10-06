//
// Copyright © 2022 Arm Limited.
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
