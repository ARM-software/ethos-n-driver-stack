//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../../include/ethosn_utils/SmallVector.hpp"

#include <catch.hpp>

//The following macros are required to test ETHOSN_DECL_SV_VECTOR_STRUCT(...)

#define EXPAND(x) x

// Expands to the number of variadic arguments in the call. Add elements to the descending
// integer sequence to increase the maximum number of variadic arguments supported
#define N_ARGS(...) EXPAND(N_ARGS_IMPL(__VA_ARGS__, 4, 3, 2, 1))

// Helper macro for N_ARGS(). Add placeholder arguments to increase the maximum
// number of variadic arguments supported
#define N_ARGS_IMPL(_1, _2, _3, _4, n, ...) n

using namespace ethosn::utils;

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

ETHOSN_DECL_SV_VECTOR_STRUCT(TypeA, data1, data2, data3)
ETHOSN_DECL_SV_VECTOR_STRUCT(TypeB, data1, data2, data3, data4)

TEST_CASE("NamedStructureTests")
{
    TypeA<uint16_t> var1 = { .data1 = 1U, .data2 = 2U, .data3 = 5U };
    TypeA<uint16_t> var2 = { .data1 = 2U, .data2 = 5U, .data3 = 6U };

    TypeB<uint32_t> var3 = { .data1 = 1U, .data2 = 5U, .data3 = 2U, .data4 = 19U };
    TypeB<uint32_t> var4 = { .data1 = 4U, .data2 = 9U, .data3 = 6U, .data4 = 49U };

    SECTION("Operator+-%*/")
    {
        TypeA<uint16_t> expectedOutput1 = { .data1 = 3U, .data2 = 7U, .data3 = 11U };
        TypeA<uint16_t> output1{ var1 + var2 };

        TypeB<uint32_t> expectedOutput2 = { .data1 = 3U, .data2 = 4U, .data3 = 4U, .data4 = 30U };
        TypeB<uint32_t> output2{ var4 - var3 };

        TypeA<uint16_t> expectedOutput3 = { .data1 = 0U, .data2 = 1U, .data3 = 1U };
        TypeA<uint16_t> output3{ var2 % var1 };

        TypeB<uint32_t> expectedOutput4 = { .data1 = 3U, .data2 = 4U, .data3 = 4U, .data4 = 30U };
        TypeB<uint32_t> output4{ var4 - var3 };

        TypeA<uint16_t> expectedOutput5 = { .data1 = 2U, .data2 = 2U, .data3 = 1U };
        TypeA<uint16_t> output5{ var2 / var1 };

        CHECK(All(expectedOutput1 == output1));
        CHECK(All(expectedOutput2 == output2));
        CHECK(All(expectedOutput3 == output3));
        CHECK(All(expectedOutput4 == output4));
        CHECK(All(expectedOutput5 == output5));
    }

    SECTION("BoolOperator")
    {
        CHECK(All((var1 == var1) == true));
        CHECK(All((var1 != var1) == false));
        CHECK(All((var1 > (var1 + var2)) == false));
        CHECK(All((var1 < (var1 + var2)) == true));
        CHECK(All((var1 >= (var1 - TypeA<uint16_t>{ 0U, 0U, 1U })) == true));
        CHECK(All((var1 <= (var1 + TypeA<uint16_t>{ 0U, 0U, 1U })) == true));
    }
}

TEST_CASE("SmallVector")
{
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

TEST_CASE("SmallVector/NonNarrowingSigned")
{
    const sv::Vector v1{ int16_t{ 1 }, int16_t{ 2 }, int16_t{ 3 } };
    {    // same type
        sv::Vector<int16_t, 3> v2 = v1;
        CHECK(All(v1 == v2));
    }
    {    // larger type
        sv::Vector<int32_t, 3> v2 = v1;
        CHECK(All(v1 == v2));
    }
}

TEST_CASE("SmallVector/NonNarrowingUnsigned")
{
    const sv::Vector v1{ uint16_t{ 1 }, uint16_t{ 2 }, uint16_t{ 3 } };
    {    // same type
        sv::Vector<uint16_t, 3> v2 = v1;
        CHECK(All(v1 == v2));
    }
    {    // larger type
        sv::Vector<uint32_t, 3> v2 = v1;
        CHECK(All(v1 == v2));
    }
    {    // larger type, signed
        sv::Vector<int32_t, 3> v2 = v1;
        CHECK(All(v1 == v2));
    }
}

TEST_CASE("SmallVector/NonNarrowingFloat")
{
    const sv::Vector v1{ 1.f, 2.f, 3.f };
    {    // same type
        sv::Vector<float, 3> v2 = v1;
        CHECK(All(v1 == v2));
    }
    {    // larger type
        sv::Vector<double, 3> v2 = v1;
        CHECK(All(v1 == v2));
    }
}
