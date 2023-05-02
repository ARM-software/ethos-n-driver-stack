//
// Copyright © 2018-2020 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "xyz.h"

namespace
{
namespace sizes
{
template <unsigned Width, unsigned Height, unsigned Depth>
struct Size
{
    static constexpr unsigned X = Width;
    static constexpr unsigned Y = Height;
    static constexpr unsigned Z = Depth;

    explicit constexpr operator Xyz() const
    {
        return { X, Y, Z };
    }

    explicit constexpr operator Xy() const
    {
        return { X, Y };
    }
};

template <unsigned Width, unsigned Height, unsigned Depth = 1>
struct BlockSize : public Size<Width, Height, Depth>
{};

template <unsigned Width, unsigned Height, unsigned Depth = 1>
struct GroupSize : public Size<Width, Height, Depth>
{};

template <typename T>
constexpr unsigned TotalSize(const T& t = {})
{
    return xyz::TotalSize(Xyz(t));
}
}    // namespace sizes
}    // namespace
