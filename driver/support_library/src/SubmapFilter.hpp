//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Utils.hpp"
#include "WeightEncoder.hpp"

#include <cstdint>
#include <vector>

namespace ethosn
{
namespace support_library
{

// For strided convolution, filter kernels and ifms needs to be subdivided.
// see the document "Strided and dilated convolutions" for reference.

class SubmapFilter
{

public:
    SubmapFilter(uint32_t originalFilterX,
                 uint32_t originalFilterY,
                 uint32_t offsetX,
                 uint32_t offsetY,
                 uint32_t strideX,
                 uint32_t strideY);

    uint32_t GetFilterX() const;
    uint32_t GetFilterY() const;
    uint32_t GetOffsetX() const;
    uint32_t GetOffsetY() const;
    uint8_t
        GetWeightAt(utils::ConstTensorData& weightData, uint32_t x, uint32_t y, uint32_t ifmIdx, uint32_t ofmIdx) const;

private:
    uint32_t m_StrideX;
    uint32_t m_StrideY;
    uint32_t m_OffsetX;
    uint32_t m_OffsetY;
    uint32_t m_SubFilterX;
    uint32_t m_SubFilterY;
};

std::vector<SubmapFilter> GetSubmapFilters(const uint32_t filterX,
                                           const uint32_t filterY,
                                           const uint32_t strideX,
                                           const uint32_t strideY,
                                           const uint32_t paddingX,
                                           const uint32_t paddingY);
std::vector<SubmapFilter> GetSubmapFilters(const uint32_t filterX,
                                           const uint32_t filterY,
                                           const uint32_t wideKernelSize,
                                           const uint32_t maxFilterSize);

}    // namespace support_library
}    // namespace ethosn
