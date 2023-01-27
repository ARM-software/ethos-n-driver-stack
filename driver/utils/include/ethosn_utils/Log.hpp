//
// Copyright © 2020,2022-2023 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

// Header-only logging framework.
//
// This file declares the class `Logger`, which you can instantiate and use for logging in your library/executable.
// An example integration of this this would be:
//   1. Choose or add a .hpp and .cpp file and declare an instance of Logger (e.g. g_Logger) in the header so that it is
//      available across your codebase. Define the variable in the .cpp file.
//   2. When you want to emit a log message, #include your new header and then call g_Logger.Info("hello") or similar.
//
// In order to actually see log messages appearing somewhere, several things need to be set up properly:
//   1. There is some code actually logging a message.
//   2. The severity of the logged message passes the *compile-time* check (see below).
//   3. The severity of the logged message passes the *run-time* check (see below).
//   4. There is a log sink registered that does something appropriate with the message (see below).
//
// By default, the log messages will not go anywhere (i.e. there are no sinks hooked up by default). This is because
// if you're using this framework in a library that others will consume, you don't want to spam them with log messages
// unless you've been explicitly told to. You can add log sinks using Logger::AddSink(), specifying either
// one of the provided sinks from ethosn::utils::log::sinks:: or your own custom sink function.
//
// If you want to attach more than the default max number of sinks, you can set this as a template argument when
// declaring your Logger class.
//
// Log messages below a given severity can be removed at *compile-time* by setting the template argument when declaring
// your Logger class. If not set, it defaults to Info.
//
// Log messages below a given severity can be skipped at *run-time* by calling Logger::SetMaxSeverity().
//
// The max log message length can be overridden via the template argument of the Logger class.
//
// This header is *not* intended to be included as part of the public API of your module - it should be used internally
// only. You may wish to add some public APIs to your module to control logging, which can internally forward to this
// header.

#pragma once

#include <algorithm>
#include <array>
#include <cstdarg>
#include <cstdio>
#include <cstring>

#if defined(__GNUC__)
#define ETHOSN_PRINTF_LIKE(archetype, stringIndex, argsIndex) __attribute__((format(archetype, stringIndex, argsIndex)))
#else
#define ETHOSN_PRINTF_LIKE(...)
#endif

namespace ethosn
{
namespace utils
{
namespace log
{

enum class Severity
{
    Panic   = 0,
    Error   = 1,
    Warning = 2,
    Info    = 3,
    Debug   = 4,
    Verbose = 5,
};

constexpr const char* GetSeverityCode(Severity s)
{
    switch (s)
    {
        case Severity::Panic:
            return "P";
        case Severity::Error:
            return "E";
        case Severity::Warning:
            return "W";
        case Severity::Info:
            return "I";
        case Severity::Debug:
            return "D";
        case Severity::Verbose:
            return "V";
        default:
            return "?";
    }
}

using TLogSink = void (*)(Severity severity, const char* msg);

/// Logging API object with state (e.g. which sinks are attached, which logging level).
/// Can be customized at compile-time (with its template params) and at runtime (by calling its methods).
///
/// Implementation details:
///
/// We use a 'varargs' interface rather than C++ 11 variadic templates so that we can benefit from GCC's compile-time
/// checking of format strings and arguments at the call sites (see ETHOSN_PRINTF_LIKE macro).
template <Severity CompileTimeMaxSeverity = Severity::Info, size_t MaxSinks = 3, size_t MaxMessageLength = 1024>
struct Logger
{
public:
    explicit Logger(const std::array<TLogSink, MaxSinks>& sinks = {},
                    Severity runtimeMaxSeverity                 = CompileTimeMaxSeverity)
        : m_RuntimeMaxSeverity(runtimeMaxSeverity)
        , m_Sinks(sinks)
    {}

    // Log functions for runtime-determined severity
    void ETHOSN_PRINTF_LIKE(printf, 3, 0) Log(Severity severity, const char* formatStr, va_list formatArgs)
    {
        LogImpl(severity, formatStr, formatArgs);
    }

    void ETHOSN_PRINTF_LIKE(printf, 3, 4) Log(Severity severity, const char* formatStr, ...)
    {
        va_list formatArgs;
        va_start(formatArgs, formatStr);
        LogImpl(severity, formatStr, formatArgs);
        va_end(formatArgs);
    }

    // Log functions for compile-time-determined severity
    template <Severity severity>
    void ETHOSN_PRINTF_LIKE(printf, 2, 0) Log(const char* formatStr, va_list formatArgs)
    {
        if (severity <= CompileTimeMaxSeverity)
        {
            LogImpl(severity, formatStr, formatArgs);
        }
    }

    template <Severity severity>
    void ETHOSN_PRINTF_LIKE(printf, 2, 3) Log(const char* formatStr, ...)
    {
        if (severity <= CompileTimeMaxSeverity)
        {
            va_list formatArgs;
            va_start(formatArgs, formatStr);
            LogImpl(severity, formatStr, formatArgs);
            va_end(formatArgs);
        }
    }

    /// Shorthand log functions for compile-time severities.
    /// Note that it's not possible to use a template function to share code between these functions, as the
    /// va_start/va_end calls need to be in the outer function and then won't get optimised based on
    /// CompileTimeMaxSeverity.
    /// @{
    void ETHOSN_PRINTF_LIKE(printf, 2, 0) Panic(const char* formatStr, va_list formatArgs)
    {
        if (Severity::Panic <= CompileTimeMaxSeverity)
        {
            LogImpl(Severity::Panic, formatStr, formatArgs);
        }
    }

    void ETHOSN_PRINTF_LIKE(printf, 2, 3) Panic(const char* formatStr, ...)
    {
        if (Severity::Panic <= CompileTimeMaxSeverity)
        {
            va_list formatArgs;
            va_start(formatArgs, formatStr);
            LogImpl(Severity::Panic, formatStr, formatArgs);
            va_end(formatArgs);
        }
    }

    void ETHOSN_PRINTF_LIKE(printf, 2, 0) Error(const char* formatStr, va_list formatArgs)
    {
        if (Severity::Error <= CompileTimeMaxSeverity)
        {
            LogImpl(Severity::Error, formatStr, formatArgs);
        }
    }

    void ETHOSN_PRINTF_LIKE(printf, 2, 3) Error(const char* formatStr, ...)
    {
        if (Severity::Error <= CompileTimeMaxSeverity)
        {
            va_list formatArgs;
            va_start(formatArgs, formatStr);
            LogImpl(Severity::Error, formatStr, formatArgs);
            va_end(formatArgs);
        }
    }

    void ETHOSN_PRINTF_LIKE(printf, 2, 0) Warning(const char* formatStr, va_list formatArgs)
    {
        if (Severity::Warning <= CompileTimeMaxSeverity)
        {
            LogImpl(Severity::Warning, formatStr, formatArgs);
        }
    }

    void ETHOSN_PRINTF_LIKE(printf, 2, 3) Warning(const char* formatStr, ...)
    {
        if (Severity::Warning <= CompileTimeMaxSeverity)
        {
            va_list formatArgs;
            va_start(formatArgs, formatStr);
            LogImpl(Severity::Warning, formatStr, formatArgs);
            va_end(formatArgs);
        }
    }

    void ETHOSN_PRINTF_LIKE(printf, 2, 0) Info(const char* formatStr, va_list formatArgs)
    {
        if (Severity::Info <= CompileTimeMaxSeverity)
        {
            LogImpl(Severity::Info, formatStr, formatArgs);
        }
    }

    void ETHOSN_PRINTF_LIKE(printf, 2, 3) Info(const char* formatStr, ...)
    {
        if (Severity::Info <= CompileTimeMaxSeverity)
        {
            va_list formatArgs;
            va_start(formatArgs, formatStr);
            LogImpl(Severity::Info, formatStr, formatArgs);
            va_end(formatArgs);
        }
    }

    void ETHOSN_PRINTF_LIKE(printf, 2, 0) Debug(const char* formatStr, va_list formatArgs)
    {
        if (Severity::Debug <= CompileTimeMaxSeverity)
        {
            LogImpl(Severity::Debug, formatStr, formatArgs);
        }
    }

    void ETHOSN_PRINTF_LIKE(printf, 2, 3) Debug(const char* formatStr, ...)
    {
        if (Severity::Debug <= CompileTimeMaxSeverity)
        {
            va_list formatArgs;
            va_start(formatArgs, formatStr);
            LogImpl(Severity::Debug, formatStr, formatArgs);
            va_end(formatArgs);
        }
    }

    void ETHOSN_PRINTF_LIKE(printf, 2, 0) Verbose(const char* formatStr, va_list formatArgs)
    {
        if (Severity::Verbose <= CompileTimeMaxSeverity)
        {
            LogImpl(Severity::Verbose, formatStr, formatArgs);
        }
    }

    void ETHOSN_PRINTF_LIKE(printf, 2, 3) Verbose(const char* formatStr, ...)
    {
        if (Severity::Verbose <= CompileTimeMaxSeverity)
        {
            va_list formatArgs;
            va_start(formatArgs, formatStr);
            LogImpl(Severity::Verbose, formatStr, formatArgs);
            va_end(formatArgs);
        }
    }
    /// @}

    void SetMaxSeverity(Severity maxSeverity)
    {
        m_RuntimeMaxSeverity = maxSeverity;
    }

    bool AddSink(TLogSink sink)
    {
        auto freeSlotIt = std::find(std::begin(m_Sinks), std::end(m_Sinks), nullptr);
        if (freeSlotIt == std::end(m_Sinks))
        {
            return false;
        }
        *freeSlotIt = sink;
        return true;
    }

    bool RemoveSink(TLogSink sink)
    {
        auto it = std::find(std::begin(m_Sinks), std::end(m_Sinks), sink);
        if (it == std::end(m_Sinks))
        {
            return false;
        }
        *it = nullptr;
        return true;
    }

private:
    void ETHOSN_PRINTF_LIKE(printf, 3, 0) LogImpl(Severity severity, const char* formatStr, va_list formatArgs)
    {
        if (severity > m_RuntimeMaxSeverity)
        {
            return;
        }
        char formattedMsg[MaxMessageLength];
        bool formattedMsgInitialised = false;    // Lazy-initialise in case there are no sinks
        for (TLogSink sink : m_Sinks)
        {
            if (sink != nullptr)
            {
                if (!formattedMsgInitialised)
                {
                    vsnprintf(formattedMsg, sizeof(formattedMsg), formatStr, formatArgs);
                    formattedMsgInitialised = true;
                }
                sink(severity, formattedMsg);
            }
        }
    }

    Severity m_RuntimeMaxSeverity          = CompileTimeMaxSeverity;
    std::array<TLogSink, MaxSinks> m_Sinks = {};
};

/// Standard sink functions that the user can use (they can also provide their own).
namespace sinks
{

template <const char ModuleName[]>
void StdOut(Severity severity, const char* msg)
{
    fprintf(stdout, "[%s %s] %s\n", ModuleName, GetSeverityCode(severity), msg);
}

template <const char ModuleName[]>
void StdErr(Severity severity, const char* msg)
{
    fprintf(stderr, "[%s %s] %s\n", ModuleName, GetSeverityCode(severity), msg);
}

}    // namespace sinks

}    // namespace log
}    // namespace utils
}    // namespace ethosn
