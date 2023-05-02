//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include <common/FixedString.hpp>

#include <catch.hpp>

using namespace ethosn;
using namespace control_unit;

TEST_CASE("FixedString Default Constructor")
{
    FixedString<10> s;
    REQUIRE(s.GetCapacity() == 10);
    REQUIRE(s.GetSize() == 0);
    REQUIRE(strcmp(s.GetCString(), "") == 0);
}

TEST_CASE("FixedString Raw String Constructor")
{
    FixedString<10> s("hello");
    REQUIRE(s.GetSize() == 5);
    REQUIRE(strcmp(s.GetCString(), "hello") == 0);
}

TEST_CASE("FixedString Raw String Constructor Overflow")
{
    FixedString<2> s("hello");
    REQUIRE(s.GetSize() == 2);
    REQUIRE(strcmp(s.GetCString(), "he") == 0);
}

TEST_CASE("FixedString Format Static Constructor")
{
    FixedString<10> s = FixedString<10>::Format("%d", 7);
    REQUIRE(s.GetSize() == 1);
    REQUIRE(strcmp(s.GetCString(), "7") == 0);
}

TEST_CASE("FixedString Format Static Constructor Overflow")
{
    FixedString<2> s = FixedString<2>::Format("%d %d", 7, 19);
    REQUIRE(s.GetSize() == 2);
    REQUIRE(strcmp(s.GetCString(), "7 ") == 0);
}

TEST_CASE("FixedString Append")
{
    FixedString<20> s;

    s += "hello";
    REQUIRE(s.GetSize() == 5);
    REQUIRE(strcmp(s.GetCString(), "hello") == 0);

    s += "goodbye";
    REQUIRE(s.GetSize() == 5 + 7);
    REQUIRE(strcmp(s.GetCString(), "hellogoodbye") == 0);
}

TEST_CASE("FixedString Append Overflow")
{
    FixedString<2> s;
    s += "0";
    REQUIRE(s.GetSize() == 1);
    REQUIRE(strcmp(s.GetCString(), "0") == 0);
    s += "1";
    REQUIRE(s.GetSize() == 2);
    REQUIRE(strcmp(s.GetCString(), "01") == 0);
    s += "2";
    REQUIRE(s.GetSize() == 2);
    REQUIRE(strcmp(s.GetCString(), "01") == 0);
}

TEST_CASE("FixedString AppendFormat")
{
    FixedString<20> s;

    s.AppendFormat("%d %d", 10, 20);
    REQUIRE(s.GetSize() == 5);
    REQUIRE(strcmp(s.GetCString(), "10 20") == 0);

    s.AppendFormat("%d %d", 30, 40);
    REQUIRE(s.GetSize() == 5 + 5);
    REQUIRE(strcmp(s.GetCString(), "10 2030 40") == 0);
}

TEST_CASE("FixedString AppendFormat Overflow")
{
    FixedString<7> s;

    s.AppendFormat("%d %d", 10, 20);
    REQUIRE(s.GetSize() == 5);
    REQUIRE(strcmp(s.GetCString(), "10 20") == 0);

    s.AppendFormat("%d %d", 30, 40);
    REQUIRE(s.GetSize() == 7);
    REQUIRE(strcmp(s.GetCString(), "10 2030") == 0);
}

TEST_CASE("FixedString Clear")
{
    FixedString<20> s;
    s += "hello";
    s.Clear();
    REQUIRE(s.GetSize() == 0);
    REQUIRE(strcmp(s.GetCString(), "") == 0);
}
