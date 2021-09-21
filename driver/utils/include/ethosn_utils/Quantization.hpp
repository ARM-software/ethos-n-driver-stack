//
// Copyright © 2020-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

namespace ethosn
{
namespace utils
{

template <typename T>
T Quantize(float value, float scale, int32_t offset)
{
    constexpr T max = std::numeric_limits<T>::max();
    constexpr T min = std::numeric_limits<T>::lowest();

    float clampedValue = std::min(
        std::max(static_cast<float>(round(value / scale) + static_cast<float>(offset)), static_cast<float>(min)),
        static_cast<float>(max));
    auto quantizedBits = static_cast<T>(clampedValue);
    return quantizedBits;
}

template <typename T>
float Dequantize(T value, float scale, int32_t offset)
{
    float dequantized = static_cast<float>(value - offset) * scale;
    return dequantized;
}

}    // namespace utils
}    // namespace ethosn
