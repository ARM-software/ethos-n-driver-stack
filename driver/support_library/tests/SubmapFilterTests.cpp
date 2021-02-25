//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/SubmapFilter.hpp"

#include <catch.hpp>

#include <vector>

using namespace ethosn::support_library;

TEST_CASE("Get subfilters for 1x1 conv stride 1")
{
    auto filters = GetSubmapFilters(1, 1, 1, 1, 0, 0);
    REQUIRE(filters.size() == 1);
    auto& a = filters[0];
    REQUIRE(a.GetFilterX() == 1);
    REQUIRE(a.GetFilterY() == 1);
}
TEST_CASE("Get subfilters for 1x3 conv stride 1")
{
    auto filters = GetSubmapFilters(3, 1, 1, 1, 0, 0);
    REQUIRE(filters.size() == 1);
    auto& a = filters[0];
    REQUIRE(a.GetFilterX() == 3);
    REQUIRE(a.GetFilterY() == 1);
}
TEST_CASE("Get subfilters for 3x3 conv stride 1")
{
    auto filters = GetSubmapFilters(3, 3, 1, 1, 0, 0);
    REQUIRE(filters.size() == 1);
    auto& a = filters[0];
    REQUIRE(a.GetFilterX() == 3);
    REQUIRE(a.GetFilterY() == 3);
}
TEST_CASE("Get subfilters for 1x1 conv stride 2")
{
    auto filters = GetSubmapFilters(1, 1, 2, 2, 0, 0);
    REQUIRE(filters.size() == 4);
    auto& a = filters[0];
    auto& b = filters[1];
    auto& c = filters[2];
    auto& d = filters[3];
    REQUIRE(a.GetFilterX() == 1);
    REQUIRE(a.GetFilterY() == 1);

    REQUIRE(b.GetFilterX() == 0);
    REQUIRE(b.GetFilterY() == 1);

    REQUIRE(c.GetFilterX() == 1);
    REQUIRE(c.GetFilterY() == 0);

    REQUIRE(d.GetFilterX() == 0);
    REQUIRE(d.GetFilterY() == 0);
}

TEST_CASE("Get subfilters for 3x3 conv stride 2")
{
    auto filters = GetSubmapFilters(3, 3, 2, 2, 0, 0);
    REQUIRE(filters.size() == 4);
    auto& a = filters[0];
    auto& b = filters[1];
    auto& c = filters[2];
    auto& d = filters[3];
    REQUIRE(a.GetFilterX() == 2);
    REQUIRE(a.GetFilterY() == 2);

    REQUIRE(b.GetFilterX() == 1);
    REQUIRE(b.GetFilterY() == 2);

    REQUIRE(c.GetFilterX() == 2);
    REQUIRE(c.GetFilterY() == 1);

    REQUIRE(d.GetFilterX() == 1);
    REQUIRE(d.GetFilterY() == 1);
}

TEST_CASE("Get subfilters for 3x3 conv stride 2 padding 1")
{
    auto filters = GetSubmapFilters(3, 3, 2, 2, 1, 1);
    REQUIRE(filters.size() == 4);
    auto& a = filters[0];
    auto& b = filters[1];
    auto& c = filters[2];
    auto& d = filters[3];
    REQUIRE(a.GetFilterX() == 1);
    REQUIRE(a.GetFilterY() == 1);

    REQUIRE(b.GetFilterX() == 2);
    REQUIRE(b.GetFilterY() == 1);

    REQUIRE(c.GetFilterX() == 1);
    REQUIRE(c.GetFilterY() == 2);

    REQUIRE(d.GetFilterX() == 2);
    REQUIRE(d.GetFilterY() == 2);

    const uint8_t weights[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    utils::ConstTensorData wd(weights, { 3, 3, 1, 1 });
    REQUIRE(a.GetWeightAt(wd, 0, 0, 0, 0) == 5);
}

TEST_CASE("Get subfilters for 1x3 conv stride 2x1")
{
    auto filters = GetSubmapFilters(3, 1, 1, 2, 0, 0);
    REQUIRE(filters.size() == 2);
    auto& a = filters[0];
    auto& b = filters[1];

    REQUIRE(a.GetFilterX() == 3);
    REQUIRE(a.GetFilterY() == 1);

    REQUIRE(b.GetFilterX() == 3);
    REQUIRE(b.GetFilterY() == 0);
}

TEST_CASE("Get subfilters for wide kernel 8x1 conv in Winograd mode")
{
    auto filters = GetSubmapFilters(8, 1, 3, 3);
    REQUIRE(filters.size() == 3);
    auto& a = filters[0];
    auto& b = filters[1];
    auto& c = filters[2];

    REQUIRE(a.GetFilterX() == 3);
    REQUIRE(a.GetFilterY() == 1);
    REQUIRE(a.GetOffsetX() == 0);
    REQUIRE(a.GetOffsetY() == 0);

    REQUIRE(b.GetFilterX() == 3);
    REQUIRE(b.GetFilterY() == 1);
    REQUIRE(b.GetOffsetX() == 3);
    REQUIRE(b.GetOffsetY() == 0);

    REQUIRE(c.GetFilterX() == 3);
    REQUIRE(c.GetFilterY() == 1);
    REQUIRE(c.GetOffsetX() == 6);
    REQUIRE(c.GetOffsetY() == 0);
}

TEST_CASE("Get subfilters for wide kernel 1x8 conv in Direct mode")
{
    auto filters = GetSubmapFilters(1, 8, 3, 7);
    REQUIRE(filters.size() == 3);
    auto& a = filters[0];
    auto& b = filters[1];
    auto& c = filters[2];

    REQUIRE(a.GetFilterX() == 1);
    REQUIRE(a.GetFilterY() == 3);
    REQUIRE(a.GetOffsetX() == 0);
    REQUIRE(a.GetOffsetY() == 0);

    REQUIRE(b.GetFilterX() == 1);
    REQUIRE(b.GetFilterY() == 3);
    REQUIRE(b.GetOffsetX() == 0);
    REQUIRE(b.GetOffsetY() == 3);

    REQUIRE(c.GetFilterX() == 1);
    REQUIRE(c.GetFilterY() == 3);
    REQUIRE(c.GetOffsetX() == 0);
    REQUIRE(c.GetOffsetY() == 6);
}
