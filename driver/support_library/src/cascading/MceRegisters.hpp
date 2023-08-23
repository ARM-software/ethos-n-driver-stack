//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../include/ethosn_support_library/Support.hpp"
#include <ethosn_command_stream/cascading/CommandStream.hpp>

#include "RegistersCommon.hpp"

namespace ethosn
{
namespace support_library
{

class HardwareCapabilities;

struct ReluActivation
{
    int16_t min;
    int16_t max;
};

struct StrideXy
{
    uint32_t x;
    uint32_t y;
};

/// Mce Scheduler work size
struct MceSWorkSize
{
    uint32_t ofmHeight;
    uint32_t ofmWidth;
    uint32_t ofmChannels;
    uint32_t ifmChannels;
};

struct FilterShape
{
    uint8_t width;
    uint8_t height;
};

struct McePadding
{
    uint8_t left;
    uint8_t top;
};

struct IfmDelta
{
    int8_t width;
    int8_t height;
};

struct IfmStripeShape
{
    uint32_t width;
    uint32_t height;
};

enum class MceAlgorithm : uint8_t
{
    DIRECT,
    WINOGRAD,
};

enum class MceUpsampleType : uint8_t
{
    OFF,
    BILINEAR,
    NEAREST_NEIGHBOUR,
    TRANSPOSE,
};

enum class MceUpsampleEdgeMode : uint8_t
{
    GENERATE,
    DROP,
};

struct MceUpsampleEdgeModeType
{
    MceUpsampleEdgeMode row;
    MceUpsampleEdgeMode col;
};

struct BlockSize
{
    uint8_t width;
    uint8_t height;
};

/// Mce Scheduler data
struct MceSDesc
{
    /// IFM SRAM tile info
    Tile ifmTile;
    /// Weight SRAM tile info
    Tile wgtTile;
    /// Mce block size
    BlockSize blockSize;
    /// Default stripe size in elements granularity
    MceSWorkSize defaultStripeSize;
    /// Last stripe size in each dimension in elements granularity
    MceSWorkSize edgeStripeSize;
    /// Number of stripes for each "work" dimension
    MceSWorkSize numStripes;
    /// Stride info for stripe ID (scalar) to stripe coord (ND) conversion
    MceSWorkSize stripeIdStrides;
    /// Conv stride
    StrideXy convStrideXy;
    /// Ifm zero point
    int16_t ifmZeroPoint;
    /// Is Ifm signed
    uint8_t isIfmSigned;
    /// Is Ofm signed
    uint8_t isOfmSigned;
    /// Upsample type
    MceUpsampleType upsampleType;
    /// Upsample edge mode
    MceUpsampleEdgeModeType upsampleEdgeMode;
    /// Mce Op mode can be: conv, depthwise, fully connected
    command_stream::cascading::MceOperation mceOpMode;
    MceAlgorithm algorithm;
    uint8_t isWideFilter;
    uint8_t isExtraIfmStripeAtRightEdge;
    uint8_t isExtraIfmStripeAtBottomEdge;
    /// Does the IFM tile contain boundary data packed in the X-direction.
    uint8_t isPackedBoundaryX;
    /// Does the IFM tile contain boundary data packed in the Y-direction.
    uint8_t isPackedBoundaryY;
    std::array<FilterShape, static_cast<uint8_t>(4U)> filterShape;
    std::array<McePadding, static_cast<uint8_t>(4U)> padding;
    /// The amount of extra IFM valid (not padding) data available to the right/bottom of the central OFM stripe.
    /// The values may differ across the OFM, so there are separate values for each possibility, based on how
    /// close the OFM stripe is to the edge of the tensor.
    /// @{
    std::array<IfmDelta, static_cast<uint8_t>(4U)> ifmDeltaDefault;
    std::array<IfmDelta, static_cast<uint8_t>(4U)> ifmDeltaOneFromEdge;
    std::array<IfmDelta, static_cast<uint8_t>(4U)> ifmDeltaEdge;
    /// @}
    /// The width/height (in elements) of IFM slots.
    /// This would typically be the same as defaultStripeSize, but may be different in cases of
    /// upsampling, VALID padding and/or packed boundary data.
    IfmStripeShape ifmStripeShapeDefault;
    IfmStripeShape ifmStripeShapeEdge;
    /// Relu activation values
    ReluActivation reluActiv;
    /// ID of the PLE kernel
    command_stream::PleKernelId pleKernelId;
};

MceUpsampleType ConvertResizeAlgorithmToCascadingCommand(const ResizeAlgorithm algorithm);

/// Generates the ProgramMceStripeCommand needed for the given stripe of the given MCE scheduler agent.
command_stream::cascading::ProgramMceStripeCommand GenerateProgramMceStripeCommand(const MceSDesc& mceS,
                                                                                   uint32_t agentId,
                                                                                   uint32_t stripeId,
                                                                                   const HardwareCapabilities& caps);
/// Generates the StartMceStripeCommand needed for the given stripe of the given MCE scheduler agent.
command_stream::cascading::StartMceStripeCommand GenerateStartMceStripeCommand(const MceSDesc& mceS,
                                                                               uint32_t agentId,
                                                                               uint32_t stripeId,
                                                                               const HardwareCapabilities& caps);

/// Creates an MceS agent for the command stream, by copying the relevant parts of the given MceSDesc
/// which do not vary between stripes of the agent.
command_stream::cascading::MceS CreateMceS(const MceSDesc& mceSDesc);

}    // namespace support_library
}    // namespace ethosn
