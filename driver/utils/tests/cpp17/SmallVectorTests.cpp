//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../../include/ethosn_utils/SmallVector.hpp"

#include <catch.hpp>

struct Nhwc
{
    int n, h, w, c;

    ETHOSN_USE_AS_SV_VECTOR(Nhwc, int, 4)
};

struct Xyz
{
    unsigned x, y, z;

    ETHOSN_USE_AS_SV_VECTOR(Xyz, unsigned, 3)
};

TEST_CASE("SmallVector")
{
    using namespace ethosn::utils;

    sv::Vector v1{ 1, 2, 3, 4 };
    sv::Vector<int, 4> v2{ 2, 4, 6, 8 };

    SECTION("operator+")
    {
        CHECK(All((v1 + v2) == sv::Vector{ 3, 6, 9, 12 }));
        CHECK(All((v1 + 1) == sv::Vector{ 2, 3, 4, 5 }));
        CHECK(All((v1 + uint16_t{ 1 }) == sv::Vector{ 2, 3, 4, 5 }));
        CHECK(All((1 + v1) == sv::Vector{ 2, 3, 4, 5 }));
        CHECK(All((uint16_t{ 1 } + v1) == sv::Vector{ 2, 3, 4, 5 }));
    }

    SECTION("operator-")
    {
        CHECK(All((v2 - v1) == sv::Vector{ 1, 2, 3, 4 }));
        CHECK(All((v1 - 1) == sv::Vector{ 0, 1, 2, 3 }));
        CHECK(All((v1 - uint16_t{ 1 }) == sv::Vector{ 0, 1, 2, 3 }));
        CHECK(All((4 - v1) == sv::Vector{ 3, 2, 1, 0 }));
        CHECK(All((uint16_t{ 4 } - v1) == sv::Vector{ 3, 2, 1, 0 }));
    }

    SECTION("operator*")
    {
        CHECK(All((v1 * v2) == sv::Vector{ 2, 8, 18, 32 }));
    }

    SECTION("operator/")
    {
        CHECK(All((v2 / v1) == sv::Vector{ 2, 2, 2, 2 }));
    }

    SECTION("operator%")
    {
        CHECK(All((v1 % v2) == sv::Vector{ 1, 2, 3, 4 }));
    }

    SECTION("Nhwc")
    {
        Nhwc nhwc{};
        nhwc = (2 * v1) - (Nhwc{} + Nhwc{});

        CHECK(All(nhwc == sv::Vector{ 2, 4, 6, 8 }));
        CHECK(All((sv::Vector{ 1, 2, 3, 4 } * Nhwc{ 2, 2, 2, 2 }) == sv::Vector{ 2, 4, 6, 8 }));
        CHECK(All((Nhwc{ 1, 2, 3, 4 } * sv::Vector{ 2, 2, 2, 2 }.To<Nhwc>()) == sv::Vector{ 2, 4, 6, 8 }));
        CHECK(All(sv::Vector{ 1, 2, 3, 4 }.Resize<2>() == sv::Vector{ 1, 2 }));
        CHECK(All(sv::Vector{ 1, 2 }.Resize<4>(3) == sv::Vector{ 1, 2, 3, 3 }));
        CHECK(sv::Vector{ 1, 2, 3, 4 }.Slice<2>().AsArray() == sv::Vector{ 3, 4 }.AsArray());
        CHECK(sv::Vector{ 1, 2 }.Slice<1, 3>(3).AsArray() == sv::Vector{ 2, 3, 3 }.AsArray());
        CHECK(All(CSel((v1 + 1) < v2, v1, v2) == sv::Vector{ v2[0], v1[1], v1[2], v1[3] }));

        CHECK(Sum(sv::Vector{ 1, 2, 3, 4 }) == 10);
        CHECK(Reduce(sv::Vector{ 1, 2, 3, 4 }, std::minus<>{}) == -10);
        CHECK(Reduce(sv::Vector{ 1, 2, 3, 4 }, std::multiplies<>{}, 1) == 24);
        CHECK(Prod(sv::Vector{ 1, 2, 3, 4 }) == 24);
    }

    SECTION("all_operators")
    {
        Xyz v = Xyz::Dup(16U);

        CHECK(All(v == sv::Vector<unsigned, 3>::Dup(16U)));
        CHECK(!All(sv::Vector<unsigned, 3>::Dup(16U) != v));
        CHECK(!All(v == sv::Vector<unsigned, 3>{ 16U }));
        CHECK(Any(v == sv::Vector<unsigned, 3>{ 16U }));
        CHECK(!Any(v == sv::Vector<unsigned, 3>{ 1U }));
        CHECK(None(v == sv::Vector<unsigned, 3>::Dup(1U)));
        CHECK(!None(v == sv::Vector<unsigned, 3>{ 16U }));

        CHECK(All(CSel(v < Xyz{ 32, 16, 5 }, +Xyz{ 1, 1, 1 }, +Xyz{ 0, 0, 0 }) == Xyz{ 1, 0, 0 }));

        CHECK(All(+v == +16U));
        CHECK(All(-v == -16U));
        CHECK(All(!v == false));
        CHECK(All(~v == ~16U));
        CHECK(All((0U + v) == 16U));
        CHECK(All((v + 0U) == 16U));
        CHECK(All((v - 0U) == 16U));
        CHECK(All((v * 1U) == 16U));
        CHECK(All((v / 1U) == 16U));
        CHECK(All((v % 32U) == 16U));
        CHECK(All((v == Xyz::Dup(16U)) == true));
        CHECK(All((v != 0U) == true));
        CHECK(All((v > 0U) == true));
        CHECK(All((v < 32U) == true));
        CHECK(All((v >= 0U) == true));
        CHECK(All((v <= 32U) == true));
        CHECK(All((v && true) == true));
        CHECK(All((v || false) == true));
        CHECK(All((v & 0xFFU) == 16));
        CHECK(All((v | 0U) == 16));
        CHECK(All((v << 0) == 16));
        CHECK(All((v >> 0) == 16));
        CHECK(All((v ^ 0U) == 16));
    }
}
