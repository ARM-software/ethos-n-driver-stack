//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "include/common/Log.hpp"
#include "include/common/Utils.hpp"

#include <scylla_addr_fields.h>
#include <scylla_regs_name_map.h>

#include <algorithm>
#include <cstdio>
#if !defined(CONTROL_UNIT_HARDWARE)
#include <sstream>
#endif

namespace ethosn
{
namespace control_unit
{
namespace utils
{

#if defined(CONTROL_UNIT_ASSERTS)
TAssertCallback g_AssertCallback = DefaultAssert;
#endif

#if defined(CONTROL_UNIT_HARDWARE)

void DefaultAssert(
    const char* rep, const char* file, unsigned line, const char* function, const char* const fmt, va_list args)
{
    LoggerType logger({ LogSink });

    logger.Panic("ASSERT \"%s\" at %s:%u in %s() failed: ", rep, file, line, function);
    if (fmt)
    {
        logger.Panic(fmt, args);
    }
    // This instruction will break into the debugger if one is configured (CONTROL_UNIT_DEBUG_MONITOR),
    // else it will go to the fault handler and send an interrupt to the kernel which will reset the firmware.
    // The breakpoint number must be 0 so that MRI handles it properly.
    __asm volatile("bkpt #0");
}

#if !defined(CONTROL_UNIT_ASSERTS)
void Fatal(const char* const fmt, va_list args)
{
    LoggerType logger({ LogSink });

    logger.Panic(fmt, args);

    __builtin_trap();
}
#endif

#else

void DefaultAssert(
    const char* rep, const char* file, unsigned line, const char* function, const char* const fmt, va_list args)
{
    fprintf(stderr, "ASSERT \"%s\" at %s:%u in %s() failed: ", rep, file, line, function);
    if (fmt)
    {
        vfprintf(stderr, fmt, args);
    }
    fprintf(stderr, "\n");
    abort();
}

#if !defined(CONTROL_UNIT_ASSERTS)
void Fatal(const char* const fmt, va_list args)
{
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    abort();
}
#endif

#endif

const char* LookupRegisterName(uint32_t registerAddress)
{
    scylla_top_addr registerAddressParts(registerAddress);
    const char* prefix;
    switch (registerAddressParts.get_reg_page())
    {
        case DMA_RP:
            prefix = "DMA.";
            break;
        case PMU_RP:
            prefix = "PMU.";
            break;
        case DL1_RP:
            prefix = "DL1.";
            break;
        case DL2_RP:
            prefix = "DL2.";
            break;
        case DL3_RP:
            prefix = "DL3.";
            break;
        case GLOBAL_RP:
            prefix = "GLOBAL.";
            break;
        case CE_RP:
            prefix = "CE.";
            break;
        case STRIPE_RP:
            prefix = "CE_STRIPE.";
            break;
        case BLOCK_RP:
            prefix = "CE_BLOCK.";
            break;
        case TSU_RP:
            prefix = "TSU.";
            break;
        default:
            ASSERT(false);
            break;
    }

    const auto begin = std::begin(scylla_regs_name_map);
    const auto end   = std::end(scylla_regs_name_map);

    const register_name_map_entry* match = std::find_if(begin, end, [&](const register_name_map_entry& entry) {
        return (entry.address == registerAddressParts.get_page_offset()) && (strstr(entry.name, prefix) != NULL);
    });

    return match == end ? nullptr : match->name;
}

#if !defined(CONTROL_UNIT_HARDWARE)

std::string GetRegisterName(uint32_t registerAddress)
{
    const char* friendlyName = LookupRegisterName(registerAddress);
    if (friendlyName)
    {
        return friendlyName;
    }
    else
    {
        std::ostringstream ss;
        ss.fill('0');
        ss.width(8);
        ss << std::hex << registerAddress;
        ASSERT(!ss.fail());
        return ss.str();
    }
}

#endif

}    // namespace utils
}    // namespace control_unit
}    // namespace ethosn
