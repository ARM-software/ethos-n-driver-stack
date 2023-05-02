//
// Copyright © 2018-2020,2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#include "MeanXyCommon.hpp"

struct OutputToInput7x7
{
    constexpr Xyz operator()(const Xyz out, const EnumBitset<Flags>) const
    {
        return { 7, 7, out.z };
    }
};

extern "C" void __attribute__((noreturn)) main()
{
    MainWithStripeLoop<MceStripeLoop<MeanXy>, OutputToInput7x7>();
}
