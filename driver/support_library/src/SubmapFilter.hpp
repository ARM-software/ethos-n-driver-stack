//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "WeightEncoder.hpp"

#include <cstdint>
#include <vector>

namespace ethosn
{
namespace support_library
{

/// For strided convolution, filter kernels and ifms needs to be subdivided.
/// see the document "Strided and dilated convolutions" for reference.
/// This class provides a strided "view" on the original weights data.
class SubmapFilter
{
public:
    SubmapFilter(uint32_t subfilterIdxX,
                 uint32_t subfilterIdxY,
                 uint32_t originalFilterX,
                 uint32_t originalFilterY,
                 uint32_t offsetX,
                 uint32_t offsetY,
                 uint32_t strideX,
                 uint32_t strideY,
                 const TensorShape& tensorShape);

    /// Shape of the subfilter, which will be <= the shape of the original filter.
    /// @{
    uint32_t GetFilterX() const;
    uint32_t GetFilterY() const;
    /// @}
    /// For striding, GetOffsetX/Y is the coordinate of the start of the IFM submap within the *padded* original (full) IFM.
    /// @{
    uint32_t GetOffsetX() const;
    uint32_t GetOffsetY() const;
    /// @}
    /// For striding, gets the amount of padding on the top/left of the interleaved tensors, for this particular submap.
    /// @{
    uint32_t GetPadLeft(uint32_t origPadLeft) const;
    uint32_t GetPadTop(uint32_t origPadTop) const;
    /// @}

    uint8_t GetWeightAt(const uint8_t* weightData, uint32_t x, uint32_t y, uint32_t ifmIdx, uint32_t ofmIdx) const;

    /// For striding, this calculates the post-interleave input width/height for the specific submap index.
    TensorShape GetIfmSubmapShape(const TensorShape& origIfmShape) const;

private:
    uint32_t m_SubfilterIdxX;
    uint32_t m_SubfilterIdxY;

    uint32_t m_StrideX;
    uint32_t m_StrideY;

    uint32_t m_OffsetX;
    uint32_t m_OffsetY;

    uint32_t m_SubFilterWidth;
    uint32_t m_SubFilterHeight;

    /// Pre-calculated constants used to calculate the index into the weight data given an HWIO location.
    /// @{
    uint32_t m_IdxCoeffY;
    uint32_t m_IdxCoeffX;
    uint32_t m_IdxCoeffIfm;
    uint32_t m_IdxConstant;
    /// @}
};

std::vector<SubmapFilter> GetSubmapFilters(const uint32_t filterX,
                                           const uint32_t filterY,
                                           const uint32_t strideX,
                                           const uint32_t strideY,
                                           const uint32_t paddingX,
                                           const uint32_t paddingY,
                                           const TensorShape& tensorShape);
std::vector<SubmapFilter> GetSubmapFilters(const uint32_t filterX,
                                           const uint32_t filterY,
                                           const uint32_t wideKernelSize,
                                           const uint32_t maxFilterSize,
                                           const TensorShape& tensorShape);

}    // namespace support_library
}    // namespace ethosn
