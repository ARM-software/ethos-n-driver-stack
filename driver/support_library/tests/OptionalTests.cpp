//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Optional.hpp"
#include "../src/Utils.hpp"

#include <catch.hpp>

using namespace ethosn::support_library::utils;

namespace
{

void PassStringRef(Optional<std::string&> value)
{
    ETHOSN_UNUSED(value);
}

void PassStringRefWithDefault(Optional<std::string&> value = EmptyOptional())
{
    ETHOSN_UNUSED(value);
}

}    // namespace

TEST_CASE("SimpleStringTests")
{
    Optional<std::string> optionalString;
    REQUIRE(static_cast<bool>(optionalString) == false);
    REQUIRE(optionalString.has_value() == false);
    REQUIRE((optionalString == Optional<std::string>()));

    optionalString = std::string("Hello World");
    REQUIRE(static_cast<bool>(optionalString) == true);
    REQUIRE(optionalString.has_value() == true);
    REQUIRE(optionalString.value() == "Hello World");
    REQUIRE((optionalString == Optional<std::string>("Hello World")));

    Optional<std::string> otherString;
    otherString = optionalString;
    REQUIRE(static_cast<bool>(otherString) == true);
    REQUIRE(otherString.value() == "Hello World");

    optionalString.reset();
    REQUIRE(static_cast<bool>(optionalString) == false);
    REQUIRE(optionalString.has_value() == false);

    const std::string stringValue("Hello World");
    Optional<std::string> optionalString2(stringValue);
    REQUIRE(static_cast<bool>(optionalString2) == true);
    REQUIRE(optionalString2.has_value() == true);
    REQUIRE(optionalString2.value() == "Hello World");

    Optional<std::string> optionalString3(std::move(optionalString2));
    REQUIRE(static_cast<bool>(optionalString3) == true);
    REQUIRE(optionalString3.has_value() == true);
    REQUIRE(optionalString3.value() == "Hello World");
}

TEST_CASE("StringRefTests")
{
    Optional<std::string&> optionalStringRef{ EmptyOptional() };
    REQUIRE(optionalStringRef.has_value() == false);

    PassStringRef(optionalStringRef);
    PassStringRefWithDefault();

    Optional<std::string&> optionalStringRef2 = optionalStringRef;

    std::string helloWorld("Hello World");

    std::string& helloWorldRef              = helloWorld;
    Optional<std::string&> optionalHelloRef = helloWorldRef;
    REQUIRE(optionalHelloRef.has_value() == true);
    REQUIRE(optionalHelloRef.value() == "Hello World");

    Optional<std::string&> optionalHelloRef2 = helloWorld;
    REQUIRE(optionalHelloRef2.has_value() == true);
    REQUIRE(optionalHelloRef2.value() == "Hello World");

    Optional<std::string&> optionalHelloRef3{ helloWorldRef };
    REQUIRE(optionalHelloRef3.has_value() == true);
    REQUIRE(optionalHelloRef3.value() == "Hello World");

    Optional<std::string&> optionalHelloRef4{ helloWorld };
    REQUIRE(optionalHelloRef4.has_value() == true);
    REQUIRE(optionalHelloRef4.value() == "Hello World");

    // modify through the optional reference
    optionalHelloRef4.value().assign("Long Other String");
    REQUIRE(helloWorld == "Long Other String");
    REQUIRE(optionalHelloRef.value() == "Long Other String");
    REQUIRE(optionalHelloRef2.value() == "Long Other String");
    REQUIRE(optionalHelloRef3.value() == "Long Other String");
}

TEST_CASE("SimpleIntTests")
{
    const int intValue = 123;

    Optional<int> optionalInt;
    REQUIRE(static_cast<bool>(optionalInt) == false);
    REQUIRE(optionalInt.has_value() == false);
    REQUIRE((optionalInt == Optional<int>()));

    optionalInt = intValue;
    REQUIRE(static_cast<bool>(optionalInt) == true);
    REQUIRE(optionalInt.has_value() == true);
    REQUIRE(optionalInt.value() == intValue);
    REQUIRE((optionalInt == Optional<int>(intValue)));

    Optional<int> otherOptionalInt;
    otherOptionalInt = optionalInt;
    REQUIRE(static_cast<bool>(otherOptionalInt) == true);
    REQUIRE(otherOptionalInt.value() == intValue);
}

TEST_CASE("ObjectConstructedInPlaceTests")
{
    struct SimpleObject
    {
    public:
        SimpleObject(const std::string& name, int value)
            : m_Name(name)
            , m_Value(value)
        {}

        bool operator==(const SimpleObject& other)
        {
            return m_Name == other.m_Name && m_Value == other.m_Value;
        }

    private:
        std::string m_Name;
        int m_Value;
    };

    std::string objectName("SimpleObject");
    int objectValue = 1;
    SimpleObject referenceObject(objectName, objectValue);

    // Use MakeOptional
    Optional<SimpleObject> optionalObject1 = MakeOptional<SimpleObject>(objectName, objectValue);
    REQUIRE(static_cast<bool>(optionalObject1) == true);
    REQUIRE(optionalObject1.has_value() == true);
    REQUIRE((optionalObject1.value() == referenceObject));

    // Call in-place constructor directly
    Optional<SimpleObject> optionalObject2(ConstructInPlace{}, objectName, objectValue);
    REQUIRE(static_cast<bool>(optionalObject1) == true);
    REQUIRE(optionalObject1.has_value() == true);
    REQUIRE((optionalObject1.value() == referenceObject));
}
