//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
// Defines the interface between the PLE and NCU MCU using the scratch registers.
// Note that this contains only the data sent back from the PLE to the NCU MCU,
// whereas the data sent from the NCU MCU to the PLE is encoded in the support
// library.
//
#pragma once

#include <cstdint>

namespace ncu_ple_interface
{

struct alignas(uint32_t) PleMsg
{
    enum class Type : uint32_t
    {
        FAULT_INFO,
        LOG_TXT,
        LOG_NUMS,
        STRIPE_DONE,
    };

    struct alignas(uint32_t) FaultInfo
    {
        static constexpr Type type = Type::FAULT_INFO;

        uint32_t cfsr;
        uint32_t pc;
        uint32_t shcsr;
    };

    struct alignas(uint32_t) LogTxt
    {
        static constexpr Type type = Type::LOG_TXT;

        // Remaining space up to the 32-byte limit is the message
        char txt[28];
    };

    struct alignas(uint32_t) LogNums
    {
        static constexpr Type type = Type::LOG_NUMS;

        enum class Fmt : uint8_t
        {
            NONE,
            I32,
            U32,
            HEX
        };

        uint32_t values[4];
        Fmt fmts[4];
        uint8_t widths[4];
    };

    struct alignas(uint32_t) StripeDone
    {
        static constexpr Type type = Type::STRIPE_DONE;
    };

    Type type;

    union
    {
        FaultInfo faultInfo;
        LogTxt logTxt;
        LogNums logNums;
        StripeDone stripeDone;
    };
};

static_assert(sizeof(PleMsg) <= 32, "PleMsg must fit in scratch registers");

}    // namespace ncu_ple_interface
