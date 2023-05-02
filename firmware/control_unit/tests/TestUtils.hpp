//
// Copyright © 2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <functional>
#include <string>
#if defined(__GNUC__)
#include <cxxabi.h>
#endif

namespace ethosn
{
namespace control_unit
{
namespace tests
{

#ifdef CONTROL_UNIT_ASSERTS
/// Verifies that the testFunc causes a fatal call
/// The fatal call can only be overridden for testing in an assert enabled build
void RequireFatalCall(std::function<void()> testFunc);
#endif    // CONTROL_UNIT_ASSERTS

/// Given a type T, returns the human-readable type name without any namespace
/// prefix or cv-qualifiers.
template <typename T>
static std::string GetDemangledTypeName(const T&)
{
    const char* implementationDefinedName = typeid(T).name();
    // The format of the name returned from the RTTI is implementation-defined.
#if defined(__GNUC__)
    // GCC returns mangled typenames, and so we need to demangle it before
    // extracting the final part of the name.
    int status;
    char* demangledNameRaw    = abi::__cxa_demangle(implementationDefinedName, nullptr, nullptr, &status);
    std::string demangledName = "<ERROR>";
    if (demangledNameRaw != nullptr)
    {
        if (status == 0)
        {
            demangledName = demangledNameRaw;
        }
        free(demangledNameRaw);
    }
#else
    // MSVC returns human-readable names, so no demangling is necessary.
    // For other implementations - fallback to a default behaviour of assuming unmangled.
    const std::string demangledName = implementationDefinedName;
#endif
    // Extract the final part of the type name (after the ::)
    size_t pos = demangledName.rfind("::");
    if (pos == std::string::npos)
    {
        return demangledName;
    }
    else
    {
        return demangledName.substr(pos + 2);
    }
}

}    // namespace tests
}    // namespace control_unit
}    // namespace ethosn
