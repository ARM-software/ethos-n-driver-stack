//
// Copyright © 2019-2020,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <string>

/// Functions which decode the metadata values. The layout of the metadata value may change.
/// It shouldn't be relied upon. The implementations have been provided in a public header
/// so it may be inlined.
namespace impl
{

inline uint64_t GetCounterValue(uint64_t metadataValue)
{
    // The counter value is stored verbatim in the metadata value
    return metadataValue;
}

inline std::string GetFirmwareLabel(uint64_t metadataValue)
{
    const char chars[4] = {
        static_cast<char>(metadataValue & 0xff),
        static_cast<char>(metadataValue >> 8 & 0xff),
        static_cast<char>(metadataValue >> 16 & 0xff),
        0,
    };
    return std::string(chars);
}

}    // namespace impl
