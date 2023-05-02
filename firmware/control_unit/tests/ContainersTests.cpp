//
// Copyright © 2018-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include <common/Containers.hpp>

#include <catch.hpp>

#include <algorithm>

using namespace ethosn;
using namespace control_unit;

namespace
{
template <typename T, uint32_t N>
void TestVectorConstructor()
{
    Vector<T, N> v;
    REQUIRE(v.Size() == 0);
}

template <typename T, uint32_t N>
void TestVectorPushBack()
{
    Vector<T, N> v;
    v.PushBack(10);
    v.PushBack(20);
    REQUIRE(v.Size() == 2);
    REQUIRE(v[0] == 10);
    REQUIRE(v[1] == 20);
}

template <typename T, uint32_t N>
void TestVectorRemoveFirst()
{
    Vector<T, N> v;
    v.PushBack(0);
    v.PushBack(1);
    v.PushBack(2);
    v.Remove(0);
    REQUIRE(v.Size() == 2);
    REQUIRE(v[0] == 1);
    REQUIRE(v[1] == 2);
}

template <typename T, uint32_t N>
void TestVectorRemoveLast()
{
    Vector<T, N> v;
    v.PushBack(0);
    v.PushBack(1);
    v.PushBack(2);
    v.Remove(2);
    REQUIRE(v.Size() == 2);
    REQUIRE(v[0] == 0);
    REQUIRE(v[1] == 1);
}

template <typename T, uint32_t N>
void TestVectorRemoveIndex()
{
    Vector<T, N> v;
    v.PushBack(10);
    v.PushBack(11);
    v.PushBack(12);

    auto it = std::find(v.begin(), v.end(), 11);
    v.Remove(it);

    auto it2 = std::find(v.begin(), v.end(), 11);
    REQUIRE(it2 == v.end());
}

template <typename T, uint32_t N>
void TestVectorFind()
{
    Vector<T, N> v;
    v.PushBack(0);
    v.PushBack(1);
    v.PushBack(2);

    auto it = std::find(v.begin(), v.end(), 1);
    REQUIRE(*it == 1);

    auto it2 = std::find(v.begin(), v.end(), 100);
    REQUIRE(it2 == v.end());
}

}    // namespace

TEST_CASE("Vector Constructor")
{
    TestVectorConstructor<uint32_t, 10>();
    TestVectorConstructor<const uint32_t, 10>();
}

TEST_CASE("Vector PushBack")
{
    TestVectorPushBack<uint32_t, 10>();
    TestVectorPushBack<const uint32_t, 10>();
}

TEST_CASE("Vector Remove First")
{
    TestVectorRemoveFirst<uint32_t, 10>();
    // Remove() not supported on const types
}

TEST_CASE("Vector Remove Last")
{
    TestVectorRemoveLast<uint32_t, 10>();
    // Remove() not supported on const types
}

TEST_CASE("Vector Remove Index")
{
    TestVectorRemoveIndex<uint32_t, 10>();
    // Remove() not supported on const types
}

TEST_CASE("Vector find")
{
    TestVectorFind<uint32_t, 10>();
    TestVectorFind<const uint32_t, 10>();
}
