//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ethosn_command_stream/cascading/CommandStream.hpp>

#include "RegistersCommon.hpp"

#include <cstdint>

namespace ethosn
{
namespace support_library
{

struct WeightsMetadata;
class HardwareCapabilities;

struct SupertensorSize
{
    uint32_t width;
    uint32_t channels;
};

/// Ifm/Ofm Data type
enum class FmsDataType : uint8_t
{
    NHWC,
    FCAF_WIDE,
    FCAF_DEEP,
    NHWCB,
};

/// FCAF Compression Info
struct FcafInfo
{
    /// Zero point info needed for FCAF
    int16_t zeroPoint;
    /// Signed activation info needed for FCAF
    bool signedActivation;
};

struct PackedBoundaryThickness
{
    uint8_t left;
    uint8_t top;
    uint8_t right;
    uint8_t bottom;

    bool AnyNonZero() const
    {
        return left > 0 || top > 0 || right > 0 || bottom > 0;
    }
};

/// Ifm/Ofm Streamer common data
struct FmSDesc
{
    /// Buffer ID of the supertensor
    uint16_t bufferId;
    /// Starting offset of the tensor inside the supertensor
    uint32_t dramOffset;
    /// IFM/OFM data type
    FmsDataType dataType;
    /// FCAF Compression Info
    FcafInfo fcafInfo;
    /// IFM/OFM SRAM tile info
    Tile tile;
    /// Default stripe size. Actual stripe size could be smaller at the tensor edges
    TensorSize defaultStripeSize;
    /// Size of the stripes at the edge of each dimension
    TensorSize edgeStripeSize;
    /// Size of the supertensor in number of cells in the width and channels dimensions.
    /// Cells are 1x1x1 (NHWC/NCHW), 8x8x16 (NHWCB), 8x16x16 (FCAF_WIDE) or 8x8x32 (FCAF_DEEP)
    SupertensorSize supertensorSizeInCells;
    /// Number of unique stripes in each tensor dimension (numStripesTotal will be
    /// a larger multiple of the product of all dimensions if reloading is needed)
    TensorSize numStripes;
    /// Stride info for stripe ID (scalar) to stripe coord (ND) conversion
    TensorSize stripeIdStrides;
};

/// Ifm Streamer data
struct IfmSDesc
{
    FmSDesc fmData;
    /// How much (if any) boundary data on each side should be loaded and packed into the same slot as the
    /// central (non-boundary) data. This is expected to be used for streaming strategies that split
    /// the IFM in both width and height, and therefore need boundary data that cannot be re-used.
    PackedBoundaryThickness packedBoundaryThickness;
    /// For some valid padding cases when using packed boundary data, the IfmS will not need to load
    /// the final stripe of data on the right/bottom edge and so the numStripes will be one smaller,
    /// but this extra data will still need to be included the packed boundary data for the second-to-last
    /// row/column.
    /// @{
    uint8_t isExtraPackedBoundaryDataOnRightEdge;
    uint8_t isExtraPackedBoundaryDataOnBottomEdge;
    /// @}
};

struct WgtSWorkSize
{
    uint32_t ofmChannels;
    uint32_t ifmChannels;
};

/// Weight Streamer data
struct WgtSDesc
{
    uint16_t bufferId;
    const std::vector<WeightsMetadata>* metadata;
    /// Weight SRAM tile info
    Tile tile;
    /// Number of stripes for each "work" dimension
    WgtSWorkSize numStripes;
    /// Stride info for stripe ID (scalar) to stripe coord (ND) conversion
    WgtSWorkSize stripeIdStrides;
};

/// PLE Loader data
struct PleLDesc
{
    /// ID of the kernel used
    command_stream::cascading::PleKernelId pleKernelId;
    /// Destination SRAM address
    uint32_t sramAddr;
};

/// Ofm Streamer data
struct OfmSDesc
{
    FmSDesc fmData;
};

/// Calculates the total number of DMA chunks needed for a particular stripe in the given IFM or OFM agent.
/// This accounts for multiple regions due to packed boundary data, if applicable.
/// @{
uint32_t CalculateNumChunks(const IfmSDesc& ifmS, uint32_t stripeId);
uint32_t CalculateNumChunks(const OfmSDesc& ofmS, uint32_t stripeId);
/// @}

/// Generates the DmaExtraData needed for the given stripe and chunk of the given IFM streamer agent.
command_stream::cascading::DmaExtraData GenerateDmaExtraDataForLoadIfmStripe(
    const IfmSDesc& ifmS, uint32_t stripeId, uint32_t chunkId, const HardwareCapabilities& caps, uint32_t nextDmaCmdId);

/// Generates the DmaExtraData needed for the given stripe of the given weight streamer agent.
command_stream::cascading::DmaExtraData GenerateDmaExtraDataForLoadWgtStripe(const WgtSDesc& wgtS,
                                                                             uint32_t stripeId,
                                                                             const HardwareCapabilities& caps,
                                                                             uint32_t nextDmaCmdId);

/// Generates the DmaExtraData needed for the given PLE loader agent.
/// All stripes require the same DMA command, so no stripeId is needed.
command_stream::cascading::DmaExtraData
    GenerateDmaExtraDataForLoadPleCode(const PleLDesc& pleL, const HardwareCapabilities& caps, uint32_t nextDmaCmdId);

/// Generates the DmaExtraData needed for the given stripe and chunk of the given OFM streamer agent.
command_stream::cascading::DmaExtraData GenerateDmaExtraDataForStoreOfmStripe(
    const OfmSDesc& ofmS, uint32_t stripeId, uint32_t chunkId, const HardwareCapabilities& caps, uint32_t nextDmaCmdId);

/// Creates an IfmS/OfmS agent for the command stream, by copying the relevant parts of the given IfmSDesc/OfmSDesc
/// which do not vary between stripes of the agent.
/// @{
command_stream::cascading::IfmS CreateIfmS(const IfmSDesc& ifmSDesc);
command_stream::cascading::OfmS CreateOfmS(const OfmSDesc& ifmSDesc);
/// @}

}    // namespace support_library
}    // namespace ethosn
