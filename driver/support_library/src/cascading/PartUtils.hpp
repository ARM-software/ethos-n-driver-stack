//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Part.hpp"
#include "Plan.hpp"

namespace ethosn
{
namespace support_library
{

using NumStripesType = uint32_t;

struct NumStripes
{
    NumStripesType m_Min;
    NumStripesType m_Max;
    bool operator<(const NumStripes& rhs) const;
};
struct MemoryStripeInfo
{
    NumStripes m_Range;
    TensorShape m_Shape;
    bool operator<(const MemoryStripeInfo& rhs) const;
};

struct MemoryStripesInfo
{
    MemoryStripeInfo m_Input;
    MemoryStripeInfo m_Output;
    MemoryStripeInfo m_Weight;
    MemoryStripeInfo m_PleInput;
    bool operator<(const MemoryStripesInfo& rhs) const;
};

struct NumMemoryStripes
{
    NumStripesType m_Input;
    NumStripesType m_Output;
    NumStripesType m_Weight;
    NumStripesType m_PleInput;
    bool operator<(const NumMemoryStripes& rhs) const;
};
// A representation of plans that only use DMA and thus only
// have information about memory
struct DmaOnlyInfo
{
    MemoryStripeInfo m_Input;
    MemoryStripeInfo m_Output;
    Lifetime m_Lifetime = Lifetime::Cascade;

    bool operator<(const DmaOnlyInfo& rhs) const;
};

class PartUtils
{
public:
    PartUtils() = delete;
    static CascadingBufferFormat GetFormat(Location location);
    static CascadingBufferFormat GetCascadingBufferFormatFromCompilerDataFormat(const CompilerDataFormat& format);
    static uint32_t CalculateBufferSize(const TensorShape& shape, CascadingBufferFormat f);
    static uint32_t CalculateSizeInBytes(const TensorShape& shape);
    static uint32_t CalculateTileSize(const HardwareCapabilities& caps,
                                      const TensorShape& tensorShape,
                                      const TensorShape& stripeShape,
                                      uint32_t numStripes);
    static uint32_t CalculateTileSize(Node* node,
                                      const HardwareCapabilities& caps,
                                      const TensorShape& inputTensorShape,
                                      const TensorShape& inputStripeShape,
                                      const TensorShape& outputStripeShape,
                                      uint32_t numStripes);
    static void AddOpToOpGraphWithInputOutputBuffers(const PartId partId,
                                                     const HardwareCapabilities& capabilities,
                                                     OwnedOpGraph& opGraph,
                                                     Node* node,
                                                     Node* outputNode,
                                                     TraversalOrder order,
                                                     DmaOnlyInfo& info,
                                                     NumMemoryStripes& numMemoryStripes,
                                                     Location inputBufferLocation,
                                                     Location outputBufferLocation,
                                                     PartInputMapping& inputMappings,
                                                     PartOutputMapping& outputMappings);
    virtual ~PartUtils() = delete;
};

}    // namespace support_library
}    // namespace ethosn
