//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "EthosNBackendId.hpp"

#include <Profiling.hpp>
#include <armnn/Types.hpp>

#define ARMNN_SCOPED_PROFILING_EVENT_ETHOSN(name) ARMNN_SCOPED_PROFILING_EVENT(armnn::EthosNBackendId(), name)

namespace armnn
{

namespace ethosnbackend
{

/// Returns the first argument rounded UP to the nearest multiple of the second argument
template <typename T, typename S>
constexpr T RoundUpToNearestMultiple(T num, S nearestMultiple)
{
    T remainder = num % nearestMultiple;

    if (remainder == 0)
    {
        return num;
    }

    return num + nearestMultiple - remainder;
}

}    // namespace ethosnbackend

}    // namespace armnn
