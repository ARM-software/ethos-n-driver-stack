//
// Copyright © 2020-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#include "MaxPool_3x3_2_2_Common.hpp"

struct OutputToInputEven
{
    constexpr Xyz operator()(const Xyz out, const EnumBitset<Flags> flags) const
    {
        return { 2 * out.x, 2 * out.y, out.z };
    }
};

extern "C" void __attribute__((noreturn)) main()
{
    MainWithStripeLoop<MaxPool_3x3_2_2_StripeLoop, OutputToInputEven>();
}
