//
// Copyright © 2022 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ethosn
{
namespace utils
{

template <typename T>
T NextEnumValue(T current)
{
    return static_cast<T>(static_cast<uint32_t>(current) + 1);
}

}    // namespace utils
}    // namespace ethosn
