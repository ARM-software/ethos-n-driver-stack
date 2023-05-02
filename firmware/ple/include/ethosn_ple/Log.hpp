//
// Copyright © 2018-2019 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#define PLE_LOGGING_OFF 0
#define PLE_LOGGING_MODEL 1
#define PLE_LOGGING_NCU 2

#include "PleState.hpp"
#include "hw.h"
#include "lsu.h"
#include "xyz.h"

namespace    // Internal linkage
{
namespace log
{
struct Txt : public ncu_ple_interface::PleMsg::LogTxt
{
    constexpr Txt(const char* const str)
        : ncu_ple_interface::PleMsg::LogTxt{}
    {
        size_t i = 0;
#pragma unroll 1
        for (; (i < (sizeof(txt) - 1U)) && (str[i] != '\0'); ++i)
        {
            txt[i] = str[i];
        }
        txt[i] = '\0';
    }
};

using Numbers = ncu_ple_interface::PleMsg::LogNums;

struct Num
{
    uint32_t value;
    Numbers::Fmt fmt;
    uint8_t width;

    constexpr Num(const uint32_t value, const Numbers::Fmt fmt, const uint8_t width = 0)
        : value(value)
        , fmt(fmt)
        , width(width)
    {}

    constexpr Num(const int32_t value, const Numbers::Fmt fmt, const uint8_t width = 0)
        : Num(static_cast<uint32_t>(value), fmt, width)
    {}

    constexpr Num(const uint32_t value, const uint8_t width = 0)
        : Num(value, Numbers::Fmt::U32, width)
    {}

    constexpr Num(const int32_t value, const uint8_t width = 0)
        : Num(static_cast<uint32_t>(value), Numbers::Fmt::I32, width)
    {}

    constexpr Num()
        : Num(0, Numbers::Fmt::NONE)
    {}
};

struct Hex : public Num
{
    constexpr Hex(const uint32_t value, const uint8_t width = 0)
        : Num(value, Numbers::Fmt::HEX, width)
    {}
};

namespace common
{
template <typename LoggerT>
__attribute__((noinline)) void Log(const LoggerT& logger, const log::Txt& msg)
{
    logger.Log(msg);
}

template <typename LoggerT>
__attribute__((noinline)) void Log(const LoggerT& logger,
                                   const log::Num& num0 = {},
                                   const log::Num& num1 = {},
                                   const log::Num& num2 = {},
                                   const log::Num& num3 = {})
{
    logger.Log({ num0, num1, num2, num3 });
}

template <typename LoggerT>
void Log(const LoggerT& logger, const Xy& xy)
{
    common::Log(logger, xy.x, xy.y);
}

template <typename LoggerT>
void Log(const LoggerT& logger, const Xyz& xyz)
{
    common::Log(logger, xyz.x, xyz.y, xyz.z);
}

template <typename LoggerT, typename T>
void Log(const LoggerT& logger, const RfReg<T>& reg, const log::Numbers::Fmt fmt, const uint8_t width = 0)
{
    for (unsigned i = 0; i < 4; ++i)
    {
        common::Log(logger, log::Num(reg[i][0], fmt, width), log::Num(reg[i][1], fmt, width),
                    log::Num(reg[i][2], fmt, width), log::Num(reg[i][3], fmt, width));
    }
}

template <typename LoggerT, typename T>
void Log(const LoggerT& logger, const RfReg<T>& reg, const uint8_t width = 0)
{
    for (unsigned i = 0; i < 4; ++i)
    {
        common::Log(logger, log::Num(reg[i][0], width), log::Num(reg[i][1], width), log::Num(reg[i][2], width),
                    log::Num(reg[i][3], width));
    }
}

template <typename LoggerT>
void Log(const LoggerT&)
{}

template <typename LoggerT, typename... Ts>
void Log(const LoggerT& logger, const log::Txt& txt, const Ts&... ts)
{
    common::Log(logger, txt);
    common::Log(logger, ts...);
}
}    // namespace common

#if PLE_LOGGING == PLE_LOGGING_OFF
inline
#endif
    namespace off
{
class Logger
{
public:
    void Log(const Txt&) const
    {}

    void Log(const Num (&)[4]) const
    {}
};

inline void Log(...)
{}
}    // namespace off

#if PLE_LOGGING == PLE_LOGGING_MODEL
inline
#endif
    namespace model
{
class Logger
{
public:
    void Log(const Txt& msg) const
    {
        LogModel(msg.txt);
    }

    void Log(const Num (&nums)[4]) const
    {
        static char buf[32];

#pragma unroll 1
        for (unsigned i = 0; i < 4U; ++i)
        {
            buf[(4U * i) + 0] = '%';
            buf[(4U * i) + 1] = '4';

            switch (nums[i].fmt)
            {
                case log::Numbers::Fmt::NONE:
                case log::Numbers::Fmt::I32:
                {
                    buf[(4U * i) + 2] = 'd';
                    break;
                }
                case log::Numbers::Fmt::U32:
                {
                    buf[(4U * i) + 2] = 'u';
                    break;
                }
                case log::Numbers::Fmt::HEX:
                {
                    buf[(4U * i) + 2] = 'x';
                    break;
                }
            }

            buf[(4U * i) + 3] = ' ';

            reinterpret_cast<volatile uint32_t&>(buf[16 + (4U * i)]) = nums[i].value;
        }

        buf[15] = '\0';

        LogModel(buf);
    }

private:
    void LogModel(const char* const txt) const
    {
        write_reg(0xCCCC, reinterpret_cast<uint32_t>(txt));
    }
};

template <typename... Ts>
void Log(const Ts&... ts)
{
    common::Log(Logger{}, ts...);
}

template <typename... Ts>
void Log(PleState&, const Ts&... ts)
{
    Log(ts...);
}
}    // namespace model

#if PLE_LOGGING == PLE_LOGGING_NCU
inline
#endif
    namespace ncu
{
class Logger
{
public:
    Logger(PleState& pleState)
        : m_PleState(pleState)
    {}

    void Log(const Num (&nums)[4]) const
    {
        auto& pleMsg = *reinterpret_cast<volatile ncu_ple_interface::PleMsg*>(PLE_REG(CE_RP, CE_PLE_SCRATCH0));

        const Numbers::Fmt fmts[4] = { nums[0].fmt, nums[1].fmt, nums[2].fmt, nums[3].fmt };
        const uint8_t widths[4]    = { nums[0].width, nums[1].width, nums[2].width, nums[3].width };

        WriteToRegisters(&pleMsg.type, Numbers::type);

        for (unsigned i = 0; i < 4; ++i)
        {
            WriteToRegisters(&pleMsg.logNums.values[i], nums[i].value);
        }

        WriteToRegisters(&pleMsg.logNums.fmts, fmts);
        WriteToRegisters(&pleMsg.logNums.widths, widths);

        SignalNcu();
    }

    void Log(const Txt& msg) const
    {
        auto& pleMsg = *reinterpret_cast<volatile ncu_ple_interface::PleMsg*>(PLE_REG(CE_RP, CE_PLE_SCRATCH0));

        WriteToRegisters(&pleMsg.type, msg.type);
        WriteToRegisters(&pleMsg.logTxt, msg);

        SignalNcu();
    }

private:
    void SignalNcu() const
    {
        __SEV();
        m_PleState.WaitForEvent<Event::SETIRQ_EVENT>();
    }

    PleState& m_PleState;
};

template <typename... Ts>
void Log(PleState& pleState, const Ts&... ts)
{
    common::Log(Logger{ pleState }, ts...);
}
}    // namespace ncu

}    // namespace log
}    // namespace
