//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <cstring>

namespace ethosn
{
namespace command_stream
{

/// This list uses X macro technique.
/// See https://en.wikipedia.org/wiki/X_Macro for more info

// define actual enum
enum class PleKernelId : uint16_t
{
#define X(a) a,
#include "PleKernelIdsGenerated.hpp"
#undef X
};

namespace ple_id_detail
{
static constexpr const char* g_PleKernelNames[] = {
#define X(a) #a,
#include "PleKernelIdsGenerated.hpp"
#undef X
};
static constexpr uint16_t g_PleKernelNamesSize = sizeof(g_PleKernelNames) / sizeof(g_PleKernelNames[0]);
}    // namespace ple_id_detail

inline const char* PleKernelId2String(const PleKernelId id)
{
    const auto idU32 = static_cast<uint32_t>(id);

    if (idU32 >= ple_id_detail::g_PleKernelNamesSize)
    {
        return "NOT_FOUND";
    }

    return ple_id_detail::g_PleKernelNames[idU32];
}

}    // namespace command_stream
}    // namespace ethosn
