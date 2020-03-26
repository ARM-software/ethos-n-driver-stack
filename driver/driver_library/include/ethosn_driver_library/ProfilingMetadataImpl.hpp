//
// Copyright © 2019-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>

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

}    // namespace impl