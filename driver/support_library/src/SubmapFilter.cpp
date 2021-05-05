//
// Copyright © 2018-2021 Arm Limited.
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
                           uint32_t strideY,
                           const TensorShape& tensorShape)
    : m_OffsetX(offsetX)
    , m_OffsetY(offsetY)
    , m_SubFilterX(offsetX == strideX - 1 ? originalFilterX / strideX : utils::DivRoundUp(originalFilterX, strideX))
    , m_SubFilterY(offsetY == strideY - 1 ? originalFilterY / strideY : utils::DivRoundUp(originalFilterY, strideY))
    // Pre-calculate constants used to calculate the index into the weight data given an HWIO location.
    // These are used to efficiently evaluate the following expression:
    //    (y * m_StrideY + m_OffsetY) * tensorShape[1] * tensorShape[2] * tensorShape[3] +
    //    (x * m_StrideX + m_OffsetX) * tensorShape[2] * tensorShape[3] +
    //    ifmIdx * tensorShape[3] +
    //    ofmIdx;
    , m_IdxCoeffY(strideY * tensorShape[1] * tensorShape[2] * tensorShape[3])
    , m_IdxCoeffX(strideX * tensorShape[2] * tensorShape[3])
    , m_IdxCoeffIfm(tensorShape[3])
    , m_IdxConstant(m_OffsetY * tensorShape[1] * tensorShape[2] * tensorShape[3] +
                    m_OffsetX * tensorShape[2] * tensorShape[3])
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

uint8_t
    SubmapFilter::GetWeightAt(const uint8_t* weightData, uint32_t y, uint32_t x, uint32_t ifmIdx, uint32_t ofmIdx) const
{
    assert(x < m_SubFilterX && y < m_SubFilterY);
    uint32_t index = y * m_IdxCoeffY + x * m_IdxCoeffX + ifmIdx * m_IdxCoeffIfm + ofmIdx + m_IdxConstant;
    return weightData[index];
}

std::vector<SubmapFilter> GetSubmapFilters(const uint32_t filterX,
                                           const uint32_t filterY,
                                           const uint32_t strideX,
                                           const uint32_t strideY,
                                           const uint32_t paddingLeft,
                                           const uint32_t paddingTop,
                                           const TensorShape& tensorShape)
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
            filters.emplace_back(filterX, filterY, shiftedX, shiftedY, strideX, strideY, tensorShape);
        }
    }
    return filters;
}

std::vector<SubmapFilter> GetSubmapFilters(const uint32_t filterX,
                                           const uint32_t filterY,
                                           const uint32_t wideKernelSize,
                                           const uint32_t maxFilterSize,
                                           const TensorShape& tensorShape)
{
    // For wide kernels filter width and height need to be extended to
    // multiple of 3 as the HW only supports 3x3, 3x1 and 1x3 kernels.
    // For Winograd the filter height and width needs to be extended to
    // multiple of 3 in all cases.
    // Use wide kernels:
    // Winograd: filter size width/height greater than 3
    // Direct: filter size width/height greater than 7
    //         wide kernel mode (H or W, both > 7)
    //         then both H,W are rounded up to multiple of 3
    //         unless H, W = 1

    const bool wideKernel         = filterX > maxFilterSize || filterY > maxFilterSize;
    const uint32_t subKernelSizeX = wideKernel ? (filterX == 1 ? 1 : wideKernelSize) : filterX;
    const uint32_t subKernelSizeY = wideKernel ? (filterY == 1 ? 1 : wideKernelSize) : filterY;
    uint32_t wFilterW             = utils::DivRoundUp(filterX, subKernelSizeX);
    uint32_t wFilterH             = utils::DivRoundUp(filterY, subKernelSizeY);

    // The order in which the submap filters are returned must be row-major.
    std::vector<SubmapFilter> filters;
    for (uint32_t h = 0; h < wFilterH; ++h)
    {
        for (uint32_t w = 0; w < wFilterW; ++w)
        {
            // Stride must be 1 for wide kernels
            filters.emplace_back(subKernelSizeX, subKernelSizeY, w * subKernelSizeX, h * subKernelSizeY, 1, 1,
                                 tensorShape);
        }
    }
    return filters;
}

}    // namespace support_library
}    // namespace ethosn
