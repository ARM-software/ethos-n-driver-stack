//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "SubmapFilter.hpp"

#include <cstdint>
#include <vector>

namespace ethosn
{
namespace support_library
{

// For strided convolution, filter kernels and ifms needs to be subdivided.

SubmapFilter::SubmapFilter(uint32_t originalFilterX,
                           uint32_t originalFilterY,
                           uint32_t offsetX,
                           uint32_t offsetY,
                           uint32_t strideX,
                           uint32_t strideY)
    : m_StrideX(strideX)
    , m_StrideY(strideY)
    , m_OffsetX(offsetX)
    , m_OffsetY(offsetY)
    , m_SubFilterX(offsetX == strideX - 1 ? originalFilterX / strideX : utils::DivRoundUp(originalFilterX, strideX))
    , m_SubFilterY(offsetY == strideY - 1 ? originalFilterY / strideY : utils::DivRoundUp(originalFilterY, strideY))
{}

uint32_t SubmapFilter::GetFilterX() const
{
    return m_SubFilterX;
}

uint32_t SubmapFilter::GetFilterY() const
{
    return m_SubFilterY;
}

uint32_t SubmapFilter::GetOffsetX() const
{
    return m_OffsetX;
}

uint32_t SubmapFilter::GetOffsetY() const
{
    return m_OffsetY;
}

uint8_t SubmapFilter::GetWeightAt(
    utils::ConstTensorData& weightData, uint32_t y, uint32_t x, uint32_t ifmIdx, uint32_t ofmIdx) const
{
    assert(x < m_SubFilterX && y < m_SubFilterY);
    return weightData.GetElement(y * m_StrideY + m_OffsetY, x * m_StrideX + m_OffsetX, ifmIdx, ofmIdx);
}

std::vector<SubmapFilter> GetSubmapFilters(const uint32_t filterX,
                                           const uint32_t filterY,
                                           const uint32_t strideX,
                                           const uint32_t strideY,
                                           const uint32_t paddingLeft,
                                           const uint32_t paddingTop)
{
    // The order in which the submap filters are returned is very important and must be compatible with both the
    // PLE interleave operator and the firmware. This order has been chosen for the weight encoder because it allows
    // the PLE to have a fixed order (of which elements go where), independent of the IFM padding.
    std::vector<SubmapFilter> filters;
    for (uint32_t y = 0; y < strideY; ++y)
    {
        uint32_t shiftedY = (y + paddingTop) % strideY;
        for (uint32_t x = 0; x < strideX; ++x)
        {
            uint32_t shiftedX = (x + paddingLeft) % strideX;
            filters.emplace_back(filterX, filterY, shiftedX, shiftedY, strideX, strideY);
        }
    }
    return filters;
}

std::vector<SubmapFilter> GetSubmapFilters(const uint32_t filterX,
                                           const uint32_t filterY,
                                           const uint32_t wideKernelSize,
                                           const uint32_t maxFilterSize)
{
    // For wide kernels filter width and height need to be extended to
    // multiple of 3 as the HW only supports 3x3, 3x1 and 1x3 kernels.
    // For Winograd the filter height and width needs to be extended to
    // multiple of 3 in all cases.
    // Use wide kernels:
    // Winograd: filter size width/height greater than 3
    // Direct: filter size width/height greater than 7

    const uint32_t subKernelSizeX = filterX > maxFilterSize ? (filterX == 1 ? 1 : wideKernelSize) : filterX;
    const uint32_t subKernelSizeY = filterY > maxFilterSize ? (filterY == 1 ? 1 : wideKernelSize) : filterY;
    uint32_t wFilterW             = utils::DivRoundUp(filterX, subKernelSizeX);
    uint32_t wFilterH             = utils::DivRoundUp(filterY, subKernelSizeY);

    // The order in which the submap filters are returned must be row-major.
    std::vector<SubmapFilter> filters;
    for (uint32_t h = 0; h < wFilterH; ++h)
    {
        for (uint32_t w = 0; w < wFilterW; ++w)
        {
            // Stride must be 1 for wide kernels
            filters.emplace_back(subKernelSizeX, subKernelSizeY, w * subKernelSizeX, h * subKernelSizeY, 1, 1);
        }
    }
    return filters;
}

}    // namespace support_library
}    // namespace ethosn
