//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/ethosn_support_library/Support.hpp"
#include "TestUtils.hpp"
#include "Utils.hpp"

#include <catch.hpp>

using namespace ethosn::support_library;

SCENARIO("QuantizationInfo: API - constructors")
{
    const float defaultScale       = 1.0f;
    const int32_t defaultZeroPoint = 0;
    const QuantizationInfo::QuantizationDim defaultDim;
    const QuantizationScales defaultScales(defaultScale);

    GIVEN("No parameters")
    {
        WHEN("Instantiating QuantizationInfo()")
        {
            QuantizationInfo info;

            THEN("All properties are set to default values")
            {
                REQUIRE(info.GetZeroPoint() == defaultZeroPoint);
                REQUIRE(info.GetScale() == defaultScale);
                REQUIRE(info.GetScales()[0] == defaultScale);
                REQUIRE(info.GetQuantizationDim() == defaultDim);
            }
        }
    }

    GIVEN("Some construction parameters")
    {
        const float argScale1      = 1.1f;
        const float argScale2      = 0.9f;
        const int32_t argZeroPoint = 5;
        const QuantizationScales argScales{ argScale1, argScale2 };

        WHEN("Instantiating QuantizationInfo(const float)")
        {
            QuantizationInfo info(0, argScale1);

            THEN("Properties are set to given args values")
            {
                REQUIRE(info.GetScale() == argScale1);
                REQUIRE(info.GetScales()[0] == argScale1);
            }
            AND_THEN("Poperties are set to default values")
            {
                REQUIRE(info.GetZeroPoint() == defaultZeroPoint);
                REQUIRE(info.GetQuantizationDim() == defaultDim);
            }
        }

        WHEN("Instantiating QuantizationInfo(const QuantizationScales&)")
        {
            QuantizationInfo info(0, argScales);

            THEN("Properties are set to given args values")
            {
                REQUIRE(info.GetScale(0) == argScale1);
                REQUIRE(info.GetScale(1) == argScale2);
                REQUIRE(info.GetScales() == argScales);
            }
            AND_THEN("Poperties are set to default values")
            {
                REQUIRE(info.GetZeroPoint() == defaultZeroPoint);
                REQUIRE(info.GetQuantizationDim() == defaultDim);
            }
        }

        WHEN("Instantiating QuantizationInfo(QuantizationScales&&)")
        {
            const QuantizationScales& argScalesRef = argScales;
            QuantizationInfo info(0, argScalesRef);

            THEN("Properties are set to given args values")
            {
                REQUIRE(info.GetScale(0) == argScale1);
                REQUIRE(info.GetScale(1) == argScale2);
                REQUIRE(info.GetScales() == argScales);
            }
            AND_THEN("Poperties are set to default values")
            {
                REQUIRE(info.GetZeroPoint() == defaultZeroPoint);
                REQUIRE(info.GetQuantizationDim() == defaultDim);
            }
        }

        WHEN("Instantiating QuantizationInfo(const int32_t, const float)")
        {
            QuantizationInfo info(argZeroPoint, argScale1);

            THEN("Properties are set to given args values")
            {
                REQUIRE(info.GetScale() == argScale1);
                REQUIRE(info.GetScales()[0] == argScale1);
                REQUIRE(info.GetZeroPoint() == argZeroPoint);
            }
            AND_THEN("Poperties are set to default values")
            {
                REQUIRE(info.GetQuantizationDim() == defaultDim);
            }
        }

        WHEN("Instantiating QuantizationInfo(const int32_t, const QuantizationScales&)")
        {
            QuantizationInfo info(argZeroPoint, argScales);

            THEN("Properties are set to given args values")
            {
                REQUIRE(info.GetScale(0) == argScale1);
                REQUIRE(info.GetScale(1) == argScale2);
                REQUIRE(info.GetScales() == argScales);
                REQUIRE(info.GetZeroPoint() == argZeroPoint);
            }
            AND_THEN("Poperties are set to default values")
            {
                REQUIRE(info.GetQuantizationDim() == defaultDim);
            }
        }

        WHEN("Instantiating QuantizationInfo(const int32_t, QuantizationScales&& scales)")
        {
            const QuantizationScales& argScalesRef = argScales;
            QuantizationInfo info(argZeroPoint, argScalesRef);

            THEN("Properties are set to given args values")
            {
                REQUIRE(info.GetScale(0) == argScale1);
                REQUIRE(info.GetScale(1) == argScale2);
                REQUIRE(info.GetScales() == argScales);
                REQUIRE(info.GetZeroPoint() == argZeroPoint);
            }
            AND_THEN("Poperties are set to default values")
            {
                REQUIRE(info.GetQuantizationDim() == defaultDim);
            }
        }

        WHEN("Instantiating QuantizationInfo(const QuantizationInfo&)")
        {
            const QuantizationInfo infoOrigin(argZeroPoint, argScales);
            QuantizationInfo info(infoOrigin);

            THEN("Properties are set to given args values from origin object")
            {
                REQUIRE(info.GetScale(0) == infoOrigin.GetScale(0));
                REQUIRE(info.GetScale(1) == infoOrigin.GetScale(1));
                REQUIRE(info.GetScales() == infoOrigin.GetScales());
                REQUIRE(info.GetZeroPoint() == infoOrigin.GetZeroPoint());
                REQUIRE(info.GetQuantizationDim() == infoOrigin.GetQuantizationDim());
            }
        }

        WHEN("Instantiating QuantizationInfo(QuantizationInfo&&)")
        {
            const QuantizationInfo infoOrigin(argZeroPoint, argScales);
            const QuantizationInfo& infoOriginRef = infoOrigin;
            QuantizationInfo info(infoOriginRef);

            THEN("Properties are set to given args values from origin object")
            {
                REQUIRE(info.GetScale(0) == infoOrigin.GetScale(0));
                REQUIRE(info.GetScale(1) == infoOrigin.GetScale(1));
                REQUIRE(info.GetScales() == infoOrigin.GetScales());
                REQUIRE(info.GetZeroPoint() == infoOrigin.GetZeroPoint());
                REQUIRE(info.GetQuantizationDim() == infoOrigin.GetQuantizationDim());
            }
        }
    }
}

SCENARIO("QuantizationInfo: API - operators")
{
    const float argScale1      = 1.1f;
    const float argScale2      = 0.9f;
    const int32_t argZeroPoint = 5;
    const QuantizationScales argScales{ argScale1, argScale2 };

    GIVEN("A const QuantizationInfo object")
    {
        const QuantizationInfo infoOrigin(argZeroPoint, argScales);

        WHEN("Using attribution operator=")
        {
            QuantizationInfo info = infoOrigin;

            THEN("Properties are set to given args values from origin object")
            {
                REQUIRE(info.GetScale(0) == infoOrigin.GetScale(0));
                REQUIRE(info.GetScale(1) == infoOrigin.GetScale(1));
                REQUIRE(info.GetScales() == infoOrigin.GetScales());
                REQUIRE(info.GetZeroPoint() == infoOrigin.GetZeroPoint());
                REQUIRE(info.GetQuantizationDim() == infoOrigin.GetQuantizationDim());
            }
            AND_THEN("Objects are compared equal")
            {
                REQUIRE(info == infoOrigin);
            }
        }
    }

    GIVEN("A QuantizationInfo object")
    {
        QuantizationInfo info;
        QuantizationInfo info1(argZeroPoint, argScales);

        WHEN("Using attribution operator=")
        {
            QuantizationInfo info = info1;

            THEN("Properties are set to given args values from origin object")
            {
                REQUIRE(info.GetScale(0) == info1.GetScale(0));
                REQUIRE(info.GetScale(1) == info1.GetScale(1));
                REQUIRE(info.GetScales() == info1.GetScales());
                REQUIRE(info.GetZeroPoint() == info1.GetZeroPoint());
                REQUIRE(info.GetQuantizationDim() == info1.GetQuantizationDim());
            }
            AND_THEN("Object is equal to origin")
            {
                REQUIRE(info == info1);
            }
        }
    }

    GIVEN("Two identical QuantizationInfo objects")
    {
        QuantizationInfo info1(argZeroPoint, argScales);
        QuantizationInfo info2(argZeroPoint, argScales);

        WHEN("Comparing objects")
        {
            THEN("Equality is true")
            {
                REQUIRE(info1 == info2);
            }
            AND_THEN("Non equality is false")
            {
                REQUIRE_FALSE(info1 != info2);
            }
        }
    }

    GIVEN("Two different QuantizationInfo objects")
    {
        QuantizationInfo info1;
        QuantizationInfo info2(argZeroPoint, argScales);

        WHEN("Comparing objects")
        {
            THEN("Equality is false")
            {
                REQUIRE_FALSE(info1 == info2);
            }
            AND_THEN("Non equality is true")
            {
                REQUIRE(info1 != info2);
            }
        }
    }
}

SCENARIO("QuantizationInfo: API - accessors")
{
    const float argScale1      = 1.1f;
    const float argScale2      = 0.9f;
    const int32_t argZeroPoint = 5;
    const QuantizationScales argScales{ argScale1, argScale2 };

    GIVEN("A QuantizationInfo object")
    {
        QuantizationInfo info;

        WHEN("Setting ZeroPoint property")
        {
            info.SetZeroPoint(argZeroPoint);

            THEN("Get the same value")
            {
                REQUIRE(info.GetZeroPoint() == argZeroPoint);
            }
        }

        WHEN("Setting Scale property")
        {
            info.SetScale(argScale1);

            THEN("Get the same value")
            {
                REQUIRE(info.GetScale(0) == argScale1);
                REQUIRE(info.GetScale() == argScale1);
                REQUIRE(info.GetScales()[0] == argScale1);
            }
        }

        WHEN("Setting QuantizationScales property")
        {
            info.SetScales(argScales);

            THEN("Get the same value")
            {
                REQUIRE(info.GetScale(0) == argScale1);
                REQUIRE(info.GetScale(1) == argScale2);
                REQUIRE(info.GetScales() == argScales);
            }
        }

        WHEN("Setting QuantizationScales property from reference")
        {
            const QuantizationScales& argScaleRef = argScales;
            info.SetScales(argScaleRef);

            THEN("Get the same value")
            {
                REQUIRE(info.GetScale(0) == argScale1);
                REQUIRE(info.GetScale(1) == argScale2);
                REQUIRE(info.GetScales() == argScales);
            }
        }

        WHEN("Setting QuantizationDim property")
        {
            info.SetQuantizationDim(5);

            THEN("Get the same value")
            {
                REQUIRE(info.GetQuantizationDim() == 5);
            }
        }
    }
}
