//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "DmaRegisters.hpp"

#include "RegistersLayout.hpp"
#include "Utils.hpp"
#include "WeightEncoder.hpp"

#include <ethosn_utils/Enums.hpp>

using namespace ethosn::command_stream;
using namespace ethosn::support_library;
using namespace ethosn::support_library::utils;
using namespace ethosn::support_library::registers;

namespace ethosn
{
namespace support_library
{

namespace
{

constexpr uint32_t DEFAULT_SRAM_GROUP_STRIDE = 4;

bool IsFmsFcaf(const FmSDesc& fmsData)
{
    return (fmsData.dataType == FmsDataType::FCAF_DEEP) || (fmsData.dataType == FmsDataType::FCAF_WIDE);
}

bool IsFmsNhwcb(const FmSDesc& fmsData)
{
    return fmsData.dataType == FmsDataType::NHWCB;
}

/// Stores state for a DMA command that is split into multiple HW commands.
/// Some DMA commands need multiple HW commands for example if they are NHWCB
/// and partial in width or depth or packing boundary data.
struct DmaCmdState
{
    /// For some streaming strategies we need to load boundary data (data from neighbouring stripes)
    /// which we cannot re-use. This data is typically smaller than the regular (non-boundary) data
    /// as we only need a few elements from it. Therefore we 'pack' this data into the same slot as
    /// the regular data rather than using a separate slot for it, which would waste SRAM.
    /// When packing the boundary and non-boundary data into a slot, the data needs to be arranged in
    /// a way that the MCE can process, and this means (for example) that the top boundary data actually
    /// needs to be positioned at the bottom of the slot (see below for more details).
    /// This means that we can't load all the data in a single transaction, and so we need to split up
    /// the loading into several 'regions'.
    /// A region is spatially contiguous in both DRAM and SRAM.
    /// Note that in the case of not packing boundary data, then this is treated as a single Centre region.
    /// Regions are named so that the names make sense for all three cases (see below).
    /// They're named based on the spatial position in the SLOT, NOT the data that they contain
    /// (e.g. bottom-right region actually contains top-left boundary data!). See below for specifics.
    /// There are 3 different cases that we support with multiple regions. The following
    /// diagrams show the spatial layout of a single slot in the tile, with the words inside
    /// each region showing what data is loaded there, named based on where that data comes from
    /// in relation to the current stripe being processed (e.g. top left means data to the top left
    /// of the current stripe).
    ///  -  Horizontal and vertical streaming, with re-use of packed boundary data in the X direction ("strategy 6 XY").
    ///     We need packed boundary data above and below.
    ///     The bottom boundary data is in the same region as the mid data, but the top boundary data
    ///     is in a separate region because it is not spatially contiguous.
    ///
    ///        ------------------
    ///        |  mid centre    |
    ///        |                |   <-  Centre region
    ///        |  bottom centre |
    ///        |----------------|
    ///        |  top centre    |   <-  Bottom region
    ///        ------------------
    ///
    ///     The Right and BottomRight regions are not relevant in this case, because that boundary data will
    ///     be loaded into a separate slot.
    ///
    ///  -  Horizontal and vertical streaming, with re-use of packed boundary data in the Y direction ("strategy 6 YX").
    ///     We need packed boundary data to the left and right.
    ///     The right boundary data is in the same region as the centre data, but the left boundary data
    ///     is in a separate region because it is not spatially contiguous.
    ///
    ///        --------------------------------------------------
    ///        |  mid centre       mid right    |   mid left    |
    ///        --------------------------------------------------
    ///                      ^                         ^
    ///                 Centre region             Right region
    ///
    ///     The Bottom and BottomRight regions are not relevant in this case, because that boundary data will
    ///     be loaded into a separate slot.
    ///
    ///  -  Horizontal, vertical and IFM depth streaming ("strategy 7"). We need packed boundary data on all sides.
    ///     The bottom/right boundary data is in the same region as the centre data, but the left, top and top-left
    ///     boundary data are in separate regions because they are not spatially contiguous.
    ///
    ///                    Centre region              Right region
    ///                          v                         v
    ///        --------------------------------------------------------
    ///        |   mid centre        mid right     |   mid left       |
    ///        |                                   |                  |
    ///        |  bottom centre     bottom right   |   bottom left    |
    ///        --------------------------------------------------------
    ///        |   top centre       top right      |   top left       |    <-  BottomRight region
    ///        --------------------------------------------------------
    ///                         ^
    ///                    Bottom region
    ///
    /// See also the diagrams for the code setting the IFM slot registers (e.g. ifm_top_slots_r).
    ///
    /// We have some freedom for choosing how to deal with stripes around the edge of the tensor,
    /// where some of the boundary data is not required (for example at the left edge of the tensor,
    /// there is no left boundary data). This means that some regions are not needed for some stripes,
    /// (for example the Right region will not be needed at the left edge of the tensor).
    /// We choose to still layout the SRAM with a gap for these regions, but no data will be loaded
    /// into those gaps, and the MCE will not read any data from them. This simplifies the MCE configuration.
    /// We also choose to leave gaps for the bottom/right data within the Centre region even when it is
    /// not needed. This simplifies the MCE configuration.
    /// Partial stripes at the right/bottom edge are padded to the default stripe shape when packing
    /// boundary data in that dimension as this simplifies the MCE configuration.
    /// When not packing boundary data, we tightly pack the data so that it can be transferred using a
    /// smaller number of DMA transfers.

    /// This field stores which region we are currently working on.
    /// Note that in the case of not packing boundary data, then this is treated as a single region and
    /// this field will always be set to Centre.
    enum class Region
    {
        Centre,
        Right,
        Bottom,
        BottomRight,
    } region;

    /// The number of bytes into the SRAM slot that the first chunk should be transferred to.
    /// Later chunks will be transferred to later addresses.
    uint32_t sramSlotOffsetForFirstChunk;
    /// The number of bytes into the DRAM buffer that the first chunk should be transferred from.
    /// Later chunks will be transferred from later addresses.
    uint32_t dramBufferOffsetForFirstChunk;

    /// A region can be split into multiple chunks. This field stores which chunk we should transfer next.
    uint32_t chunkId;
    /// SRAM stride between adjacent groups (8x8) in the X-direction.
    uint32_t sramStridePerGroupCol;
    /// SRAM stride between adjacent groups (8x8) in the Y-direction.
    uint32_t sramStridePerGroupRow;
    uint8_t isSramChannelStrided;
    bool dramStride;
    TensorSize chunkSize;
    TensorSize numChunks;
};

struct FmsDmaRegParams
{
    uint32_t dramOffset;
    uint32_t stride0;
    uint32_t stride3;
    uint32_t sramAddr;
    uint32_t sramGroupStride;
    uint32_t sramRowStride;
    uint32_t totalBytes;
    uint32_t channels;
    uint32_t emcMask;
};

constexpr TensorShape GetCellSize(FmsDataType fmsDataType)
{
    switch (fmsDataType)
    {
        case FmsDataType::FCAF_DEEP:
        {
            return g_FcafDeepCellShape;
        }
        case FmsDataType::FCAF_WIDE:
        {
            return g_FcafWideCellShape;
        }
        case FmsDataType::NHWCB:
        {
            return g_BrickGroupShape;
        }
        case FmsDataType::NHWC:
        default:
        {
            return { 1, 1, 1 };
        }
    }
}

FmsDmaRegParams GetDmaParamsFcaf(const FmSDesc& fmData,
                                 uint32_t stripeId,
                                 const HardwareCapabilities& caps,
                                 const DmaCmdState& chunkState)
{
    // FCAF specific registers are programmed as required:
    // Stripe=(h,w,c), Tensor=(H,W,C)
    // Tensor dimensions rounded up to a multiple of the cell size
    // DMA_CHANNELS: c
    // DMA_EMCS: non-zero
    // DMA_TOTAL_BYTES: h*w*c, each dimension rounded up to a multiple of the cell size
    // DMA_STRIDE0: w
    // DMA_STRIDE1: C
    // DMA_STRIDE2: W*C
    // DMA_STRIDE3: h

    FmsDmaRegParams fmsDmaParams = {};

    const TensorShape fcafCellShape = GetCellSize(fmData.dataType);

    {
        fmsDmaParams.dramOffset = chunkState.dramBufferOffsetForFirstChunk;
    }
    {
        fmsDmaParams.stride0 = chunkState.chunkSize.width;
    }
    {
        fmsDmaParams.sramAddr = SramAddr(fmData.tile, stripeId) + chunkState.sramSlotOffsetForFirstChunk;

        // These strides are in terms of 128-bit (16-byte) words
        fmsDmaParams.sramGroupStride = chunkState.sramStridePerGroupCol / 16;
        fmsDmaParams.sramRowStride   = chunkState.sramStridePerGroupRow / 16;
    }
    {
        fmsDmaParams.stride3 = chunkState.chunkSize.height;
    }
    {
        // See FCAF Specification, section 3.4.1.1
        fmsDmaParams.totalBytes =
            (utils::RoundUpToNearestMultiple(chunkState.chunkSize.height, GetHeight(fcafCellShape)) *
             utils::RoundUpToNearestMultiple(chunkState.chunkSize.width, GetWidth(fcafCellShape)) *
             utils::RoundUpToNearestMultiple(chunkState.chunkSize.channels, GetChannels(fcafCellShape)));
    }

    fmsDmaParams.channels        = chunkState.chunkSize.channels;
    const uint32_t numActiveEmcs = std::min(fmsDmaParams.channels, +caps.GetNumberOfSrams());
    fmsDmaParams.emcMask         = (1U << numActiveEmcs) - 1U;

    return fmsDmaParams;
}

FmsDmaRegParams GetDmaParamsNhwcb(const FmSDesc& fmData,
                                  uint32_t stripeId,
                                  bool input,
                                  const HardwareCapabilities& caps,
                                  const DmaCmdState& chunkState)
{
    // NHWCB specific registers are programmed as required:
    // Stripe=(h,w,c), Tensor=(H,W,C)
    // DMA_CHANNELS: c
    // DMA_TOTAL_BYTES: h*w*c, each dimension rounded up to a multiple of the brickGroup size

    const SupertensorSize& supertensorSizeInCells = fmData.supertensorSizeInCells;

    FmsDmaRegParams fmsDmaParams = {};

    constexpr TensorShape nhwcbBrickGroupShape = GetCellSize(FmsDataType::NHWCB);

    {
        const TensorSize brickGroupStride{
            .height   = 1024U * supertensorSizeInCells.width * supertensorSizeInCells.channels,
            .width    = 1024U * supertensorSizeInCells.channels,
            .channels = 1024U,
        };

        TensorSize chunkCoords;

        uint32_t numActiveEmcs = std::min(chunkState.chunkSize.channels, caps.GetNumberOfSrams());

        if (chunkState.isSramChannelStrided)
        {
            chunkCoords.height   = chunkState.chunkId / (chunkState.numChunks.width * chunkState.numChunks.channels);
            chunkCoords.width    = (chunkState.chunkId / chunkState.numChunks.channels) % chunkState.numChunks.width;
            chunkCoords.channels = chunkState.chunkId % chunkState.numChunks.channels;

            // The following explanation is only suitable for current supported cases where chunkification along
            // channels can happen only in chunks of 8x8x8.
            //
            // When chunkfication is done accross channels, i.e. each chunk has depth of 8, the correct EMC must
            // be turned on. If the number of emc is 8, all the emcs are turned on as each channel data goes into
            // every SRAM. But this changes when the number of emcs is 16, in that case, for the chunk that starts
            // right after a multiple of full brickgroup depth, the first 8 emcs are turned on and, for the next
            // chunk, last 8 emcs are turned on. The above is done so that channel data is properly alligned in
            // the SRAM.
            uint32_t onlyLast8EmcsRequired = (chunkCoords.channels % 2U != 0U) && (numActiveEmcs == 16U);
            fmsDmaParams.emcMask = ((1U << std::min(chunkState.chunkSize.channels, caps.GetNumberOfSrams()))) - 1U;
            fmsDmaParams.emcMask = fmsDmaParams.emcMask << (onlyLast8EmcsRequired ? 8U : 0U);
        }
        else
        {
            // Offset of a chunk within a stripe, equal to 0 for no chunkification
            chunkCoords.height   = chunkState.chunkId / chunkState.numChunks.width;
            chunkCoords.width    = chunkState.chunkId % chunkState.numChunks.width;
            chunkCoords.channels = 0U;

            fmsDmaParams.emcMask = ((1U << numActiveEmcs) - 1U);
        }

        uint32_t dramOffset = chunkState.dramBufferOffsetForFirstChunk;

        if (!chunkState.isSramChannelStrided)
        {
            dramOffset += chunkCoords.width * brickGroupStride.width + chunkCoords.height * brickGroupStride.height +
                          chunkCoords.channels * brickGroupStride.channels;
        }
        else
        {
            // Consider the following example where the two stripes of size
            // 8x8x24 have to be DMAed. This means that the supertensor has
            // the dimensions: 8x8x48.
            //
            // The first stripe can be transferred without any chunkification
            // as the data to be DMAed is contiguous in memory. But the second
            // stripe isn't and therefore, this stripe is split in three chunks
            // each of size 8x8x8.
            //
            // Following diagram represents a 8x8x48 NHWCB tensor placed in DRAM
            //
            //                    __  ________________________________    ___  ___
            //  DRAM OFFSET:0     |   |                                |     |    |
            //                    |   |        1 - 4x4x16              |     |    |--> 16 channel information for the first 4x4 patch.
            //                    |   |                                |     |    |    Essentially it comprises of 16 rows of 4x4 patch.
            //                    |   |_ _ _ _ _ _ _ _ _ _ _ _ _ _ _  _|     | ___|    Each brick group of 8x8x16 comprises of 4 such areas
            //                    |   |                                |     |         in DRAM. Let's call this 4x4x16 area as a patch group.
            //                    |   |        2 - 4x4x16              |     |
            //                    |   |                                |     |---> Depicts a brick group in Dram, i.e. 8x8x16 tensor
            //  Brick Group 1 <---|   |_ _ _ _ _ _ _ _ _ _ _ _ _ _ _  _|     |
            //                    |   |                                |     |
            //                    |   |        3 - 4x4x16              |     |
            //                    |   |                                |     |
            //                    |   |_ _ _ _ _ _ _ _ _ _ _ _ _ _ _  _|     |
            //                    |   |                                |     |
            //                    |   |        4 - 4x4x16              |     |
            //                    |   |                                |     |
            //                    |__ |________________________________|   __|
            //  DRAM OFFSET:1024 |    |                                |
            //                   |    |                                | __
            //                   |    |                                |  |-> The first chunk for second stripe starts at DRAM position 1152
            //                   |    |_ _ _ _ _ _ _ _ _ _ _ _ _ _ _  _|      . This is so becasue 25th channel which is the starting of second
            //                   |    |                                |      output is located at 9th channel of 2nd Brick group. And since the
            //                   |    |                                |      9th channel for first patch group of 2nd brick group occurs at 9th
            //                   |    |                                |      row, the offest for first chunk becomes 1024 + 4x4x8 = 1152. This
            //  Brick Group 2 <--|    |_ _ _ _ _ _ _ _ _ _ _ _ _ _ _  _|      value is also described as dramBufferOffsetForFirstChunk.
            //                   |    |                                |
            //                   |    |                                |
            //                   |    |                                |
            //                   |    |_ _ _ _ _ _ _ _ _ _ _ _ _ _ _  _|
            //                   |    |                                |
            //                   |    |                                |
            //                   |    |                                |
            //                   |_ _ |________________________________| __
            // DRAM OFFSET: 2048   |  |                                |  |-> The second chunk for second stripe starts at DRAM position 2048.
            //                     |  |                                | __   This is so because 33rd channel which is 9th channel in second
            //                     |  |                                |  |   output is located at 1st channel of 3rd brick group. And since the
            //                     |  |_ _ _ _ _ _ _ _ _ _ _ _ _ _ _  _|  |   1st channel for first patch group of 3rd brick group occurs at 1st
            //                     |  |                                |  |   row, the offset for second chunk becomes 2048. This makes the offset
            //                     |  |                                |  |   for 2nd chunk 896 bytes away from dramBufferOffsetForFirstChunk.
            //  Brick Group 3 <----|  |_ _ _ _ _ _ _ _ _ _ _ _ _ _ _  _|  |
            //                     |  |                                |  |
            //                     |  |                                |  |
            //                     |  |                                |  |-> The third chunk for second stripe starts at DRAM position 2176.
            //                     |  |_ _ _ _ _ _ _ _ _ _ _ _ _ _ _  _|      This is so becasue 41st channel which is 17th channel in second
            //                     |  |                                |      output is located at 9th channel of 3rd brick group. And since the
            //                     |  |                                |      9th channel for first patch group in 3rd brick group occurs at 9th
            //                     |  |                                |      row, the offset for second channel becomes 2048 + 4x4x16 or in other
            //                     |_ |________________________________|      words 2176. This makes the offset for 3rd chunk 1024 bytes away from
            //                                                                dramBufferOffsetForFirstChunk.
            uint32_t depthOffset =
                chunkCoords.channels % 2U == 0U ? 0U : (1024U - 4U * 4U * chunkState.chunkSize.channels);
            uint32_t depthOffsetMultiplier = chunkCoords.channels / 2U;
            dramOffset += (depthOffset + 1024U * depthOffsetMultiplier);
        }

        fmsDmaParams.dramOffset = dramOffset;

        if (!input && utils::DivRoundUp(chunkState.chunkSize.channels, 16U) == 1)
        {
            // NHWCB_WEIGHT_STREAMING allows for a consistent dram stride between brick groups
            // which is only non-zero if the chunk is one brick group in depth
            const uint32_t brickGroupsToSkip = supertensorSizeInCells.channels - 1U;
            fmsDmaParams.stride0             = 1024U * brickGroupsToSkip;
        }
        else if (!input && chunkState.chunkSize.channels == 8U && chunkState.isSramChannelStrided)
        {
            fmsDmaParams.stride0 = 8U * 8U * 8U;
        }
        else
        {
            fmsDmaParams.stride0 = 0U;
        }

        // Offset within the tile slot for individual chunks
        fmsDmaParams.sramAddr = SramAddr(fmData.tile, stripeId) + chunkState.sramSlotOffsetForFirstChunk +
                                chunkState.sramStridePerGroupRow * chunkCoords.height +
                                chunkState.sramStridePerGroupCol * chunkCoords.width;

        if (chunkState.isSramChannelStrided)
        {
            if (std::min(chunkState.chunkSize.channels, +caps.GetNumberOfSrams()) == 8U)
            {
                // In case of 8 EMCs high byte address always remains the same as all EMCs stay active.
                //
                // Low byte address
                fmsDmaParams.sramAddr += (chunkCoords.channels * 64U);
            }
            else
            {
                // Low byte address
                fmsDmaParams.sramAddr += (chunkCoords.channels / 2U) * (64U);
                // High byte address
                uint32_t sramSizePerEmc = caps.GetTotalSramSize() / caps.GetNumberOfSrams();
                fmsDmaParams.sramAddr += (chunkCoords.channels % 2U) * sramSizePerEmc;
            }
        }
    }

    {
        fmsDmaParams.totalBytes =
            utils::RoundUpToNearestMultiple(chunkState.chunkSize.height, GetHeight(nhwcbBrickGroupShape)) *
            utils::RoundUpToNearestMultiple(chunkState.chunkSize.width, GetWidth(nhwcbBrickGroupShape));

        fmsDmaParams.totalBytes *= chunkState.chunkSize.channels;
    }
    {
        fmsDmaParams.channels = chunkState.chunkSize.channels;
    }
    return fmsDmaParams;
}

FmsDmaRegParams
    GetDmaParamsNhwc(TensorSize& stripeSize, const FmSDesc& fmData, uint32_t stripeId, const HardwareCapabilities& caps)
{
    // NHWC specific registers are programmed as required
    // Stripe=(h,w,c), Tensor=(H,W,C)
    // DMA_CHANNELS: c
    // DMA_EMCS: non-zero
    // DMA_TOTAL_BYTES: h*w*c
    // DMA_STRIDE0: w*c
    // DMA_STRIDE1: W*C
    // NHWC transfer cannot split channels so c must equal C

    if (fmData.supertensorSizeInCells.width != 1)
    {
        assert(stripeSize.channels == fmData.supertensorSizeInCells.channels &&
               "NHWC transfer cannot split channels unless width is 1");
    }

    FmsDmaRegParams fmsDmaParams = {};

    TensorSize stripeCoord;
    stripeCoord.width  = (static_cast<uint32_t>(stripeId) / fmData.stripeIdStrides.width) % fmData.numStripes.width;
    stripeCoord.height = (static_cast<uint32_t>(stripeId) / fmData.stripeIdStrides.height) % fmData.numStripes.height;
    stripeCoord.channels =
        (static_cast<uint32_t>(stripeId) / fmData.stripeIdStrides.channels) % fmData.numStripes.channels;

    {
        const TensorSize stripeDramStrides{
            .height = 1U * fmData.supertensorSizeInCells.width * fmData.supertensorSizeInCells.channels *
                      fmData.defaultStripeSize.height,
            .width    = 1U * fmData.supertensorSizeInCells.channels * fmData.defaultStripeSize.width,
            .channels = fmData.defaultStripeSize.channels,
        };

        fmsDmaParams.dramOffset = fmData.dramOffset + stripeCoord.width * stripeDramStrides.width +
                                  stripeCoord.height * stripeDramStrides.height +
                                  stripeCoord.channels * stripeDramStrides.channels;
    }
    {
        fmsDmaParams.stride0 = stripeSize.width * stripeSize.channels;
    }
    {
        fmsDmaParams.totalBytes = stripeSize.width * stripeSize.height * stripeSize.channels;
    }
    {
        fmsDmaParams.sramAddr = SramAddr(fmData.tile, stripeId);

        fmsDmaParams.sramGroupStride = DEFAULT_SRAM_GROUP_STRIDE;
    }

    fmsDmaParams.channels        = stripeSize.channels;
    const uint32_t numActiveEmcs = std::min(stripeSize.channels, caps.GetNumberOfSrams());
    fmsDmaParams.emcMask         = (1U << numActiveEmcs) - 1U;

    return fmsDmaParams;
}

FmsDmaRegParams GetDmaParams(TensorSize& stripeSize,
                             const FmSDesc& fmData,
                             uint32_t stripeId,
                             bool input,
                             const HardwareCapabilities& caps,
                             const DmaCmdState& chunkState)
{
    switch (fmData.dataType)
    {
        case FmsDataType::NHWC:
            return GetDmaParamsNhwc(stripeSize, fmData, stripeId, caps);
        case FmsDataType::FCAF_WIDE:
        case FmsDataType::FCAF_DEEP:
            return GetDmaParamsFcaf(fmData, stripeId, caps, chunkState);
        case FmsDataType::NHWCB:
            return GetDmaParamsNhwcb(fmData, stripeId, input, caps, chunkState);
        default:
            assert(false);
            return {};
    }
}

/// Common code for both IFM and OFM.
void GenerateDmaCommandCommon(const FmSDesc& fmData,
                              uint32_t stripeId,
                              bool input,
                              DmaCommand& cmd,
                              const HardwareCapabilities& caps,
                              const DmaCmdState& chunkState)
{
    TensorSize stripeCoord;
    stripeCoord.width  = (static_cast<uint32_t>(stripeId) / fmData.stripeIdStrides.width) % fmData.numStripes.width;
    stripeCoord.height = (static_cast<uint32_t>(stripeId) / fmData.stripeIdStrides.height) % fmData.numStripes.height;
    stripeCoord.channels =
        (static_cast<uint32_t>(stripeId) / fmData.stripeIdStrides.channels) % fmData.numStripes.channels;

    const bool isFcaf  = IsFmsFcaf(fmData);
    const bool isNhwcb = IsFmsNhwcb(fmData);

    TensorSize stripeSize;
    stripeSize.width = (stripeCoord.width == (fmData.numStripes.width - 1U)) ? fmData.edgeStripeSize.width
                                                                             : fmData.defaultStripeSize.width;
    stripeSize.height = (stripeCoord.height == (fmData.numStripes.height - 1U)) ? fmData.edgeStripeSize.height
                                                                                : fmData.defaultStripeSize.height;
    stripeSize.channels = (stripeCoord.channels == (fmData.numStripes.channels - 1U))
                              ? fmData.edgeStripeSize.channels
                              : fmData.defaultStripeSize.channels;

    // Get DMA parameters specific to each feature map format
    const FmsDmaRegParams fmsDmaParams = GetDmaParams(stripeSize, fmData, stripeId, input, caps, chunkState);

    {
        sram_addr_r sramAddr;
        sramAddr.set_address(fmsDmaParams.sramAddr);
        cmd.SRAM_ADDR = sramAddr.word;
    }
    {
        cmd.m_DramOffset = fmsDmaParams.dramOffset;
    }
    if (!isNhwcb)
    {
        dma_sram_stride_r sramStride;
        sramStride.set_sram_group_stride(fmsDmaParams.sramGroupStride);
        if (isFcaf)
        {
            sramStride.set_sram_row_stride(fmsDmaParams.sramRowStride);
        }
        cmd.DMA_SRAM_STRIDE = sramStride.word;
    }

    {
        dma_channels_r channels;
        channels.set_channels(fmsDmaParams.channels);
        cmd.DMA_CHANNELS = channels.word;
    }
    {
        cmd.DMA_EMCS = fmsDmaParams.emcMask;
    }

    if (!isNhwcb || (!input && fmsDmaParams.stride0 != 0U))
    {
        dma_stride0_r stride0;
        stride0.set_inner_stride(fmsDmaParams.stride0);
        cmd.DMA_STRIDE0 = stride0.word;
    }
    {
        dma_total_bytes_r tot;
        tot.set_total_bytes(fmsDmaParams.totalBytes);
        cmd.DMA_TOTAL_BYTES = tot.word;
    }

    if (isFcaf)
    {
        dma_stride3_r stride3;
        stride3.set_stride3(fmsDmaParams.stride3);
        cmd.DMA_STRIDE3 = stride3.word;
    }
}

/// Updates `state` with chunking information derived from the given parameters.
void ConfigureChunks(DmaCmdState& state,
                     FmsDataType format,
                     TensorSize& stripeSize,
                     const SupertensorSize& supertensorSizeInCells,
                     uint32_t dramOffset,
                     TensorSize& dramPosition,
                     uint32_t sramOffset,
                     uint32_t sramWidthSkipPerRow,
                     bool dramStridingAllowed,
                     uint32_t numEmcs,
                     bool isChunkingStartingMidBrick = false)
{
    TensorShape cellShape = GetCellSize(format);

    if ((dramPosition.height % GetHeight(cellShape)) != 0)
    {
        throw InternalErrorException("dramPosition must be a multiple of the brickgroup height");
    }
    if ((dramPosition.width % GetWidth(cellShape)) != 0)
    {
        throw InternalErrorException("dramPosition must be a multiple of the brickgroup width");
    }
    if (dramPosition.channels % GetChannels(cellShape) > 0)
    {
        if (format == FmsDataType::NHWCB)
        {
            if (!(stripeSize.channels <= (GetChannels(cellShape) - dramPosition.channels % GetChannels(cellShape))))
            {
                throw InternalErrorException("Can't go through boundary of 16 with NHWCB");
            }
        }
        else
        {
            throw InternalErrorException("For formats other than NHWCB, the DRAM offset must be aligned to a cell.");
        }
    }
    if ((sramWidthSkipPerRow % 8U) != 0)
    {
        throw InternalErrorException("sramWidthSkipPerRow must be a multiple of the brickgroup width");
    }

    state.chunkSize = stripeSize;
    state.numChunks = TensorSize{ .height = 1U, .width = 1U, .channels = 1U };

    if (format == FmsDataType::NHWCB)
    {
        // Consistent non-zero DRAM stride needed for output streaming to use DRAM striding
        const bool canDramStride = dramStridingAllowed && utils::DivRoundUp(stripeSize.channels, 16U) == 1U &&
                                   supertensorSizeInCells.channels > 1;

        state.dramStride = canDramStride;

        const bool partialDepth = utils::DivRoundUp(stripeSize.channels, 16U) < supertensorSizeInCells.channels;
        const bool partialWidth = utils::DivRoundUp(stripeSize.width, 8U) < supertensorSizeInCells.width;

        // Input NHWCB cannot DRAM stride, output NHWCB can only dram stride with stripes
        // one brick group in depth.

        // DRAM striding can be used for as much of the stripe that has a consistent stride
        // i.e. can cover the full stripe if it is full width, or each row if it is partial

        // Stride between X chunks if partial depth
        if (partialDepth && !canDramStride)
        {
            state.chunkSize.width = 8U;
            state.numChunks.width = utils::DivRoundUp(stripeSize.width, 8U);
        }

        // Stride between Y chunks if partial width or partial depth
        if ((partialDepth && !canDramStride) || partialWidth)
        {
            state.chunkSize.height = 8U;
            state.numChunks.height = utils::DivRoundUp(stripeSize.height, 8U);
        }

        if (partialDepth && stripeSize.channels % 8U == 0U && isChunkingStartingMidBrick)
        {
            state.numChunks.channels   = utils::DivRoundUp(stripeSize.channels, 8U);
            state.chunkSize.channels   = 8U;
            state.isSramChannelStrided = true;
        }
    }

    state.sramStridePerGroupCol = 8 * 8U * utils::DivRoundUp(state.chunkSize.channels, numEmcs);
    state.sramStridePerGroupRow = utils::DivRoundUp(stripeSize.width, 8U) * state.sramStridePerGroupCol +
                                  sramWidthSkipPerRow * 8U * utils::DivRoundUp(state.chunkSize.channels, numEmcs);

    uint32_t dramCellSize = format == FmsDataType::NHWCB ? 1024U : 2112U;
    const TensorSize cellStride{
        .height   = dramCellSize * supertensorSizeInCells.width * supertensorSizeInCells.channels,
        .width    = dramCellSize * supertensorSizeInCells.channels,
        .channels = dramCellSize,
    };

    state.dramBufferOffsetForFirstChunk = dramOffset;
    state.dramBufferOffsetForFirstChunk += dramPosition.width / GetWidth(cellShape) * cellStride.width +
                                           dramPosition.height / GetHeight(cellShape) * cellStride.height +
                                           dramPosition.channels / GetChannels(cellShape) * cellStride.channels;
    if (format == FmsDataType::NHWCB)
    {
        // NHWCB can have transfers partway through a brick group in DRAM
        state.dramBufferOffsetForFirstChunk += (dramPosition.channels % 16) * 16;
    }

    state.sramSlotOffsetForFirstChunk = sramOffset;
}

uint32_t CalculateNumChunksInShape(FmsDataType format,
                                   const TensorSize& stripeSize,
                                   const SupertensorSize& supertensorSizeInCells,
                                   bool dramStridingAllowed,
                                   bool isChunkingStartingMidBrick = false)
{
    if (format != FmsDataType::NHWCB)
    {
        // Chunking is only relevant for NHWCB
        return 1;
    }

    // Consistent non-zero DRAM stride needed for output streaming to use DRAM striding
    const bool canDramStride =
        dramStridingAllowed && utils::DivRoundUp(stripeSize.channels, 16U) == 1U && supertensorSizeInCells.channels > 1;

    TensorSize numChunks = TensorSize{ .height = 1U, .width = 1U, .channels = 1U };

    const bool partialDepth = utils::DivRoundUp(stripeSize.channels, 16U) < supertensorSizeInCells.channels;
    const bool partialWidth = utils::DivRoundUp(stripeSize.width, 8U) < supertensorSizeInCells.width;

    // Input NHWCB cannot DRAM stride, output NHWCB can only dram stride with stripes
    // one brick group in depth.

    // DRAM striding can be used for as much of the stripe that has a consistent stride
    // i.e. can cover the full stripe if it is full width, or each row if it is partial

    // Stride between X chunks if partial depth
    if (partialDepth && !canDramStride)
    {
        numChunks.width = utils::DivRoundUp(stripeSize.width, 8U);
    }

    // Stride between Y chunks if partial width or partial depth
    if ((partialDepth && !canDramStride) || partialWidth)
    {
        numChunks.height = utils::DivRoundUp(stripeSize.height, 8U);
    }

    if (partialDepth && stripeSize.channels % 8U == 0U && isChunkingStartingMidBrick)
    {
        numChunks.channels = utils::DivRoundUp(stripeSize.channels, 8U);
    }

    return numChunks.width * numChunks.height * numChunks.channels;
}

uint32_t CalculateNumChunksInRegion(DmaCmdState::Region region,
                                    const FmSDesc& fmData,
                                    const PackedBoundaryThickness& packedBoundaryThickness,
                                    bool isExtraPackedBoundaryDataOnRightEdge,
                                    bool isExtraPackedBoundaryDataOnBottomEdge,
                                    uint32_t stripeId,
                                    bool dramStridingAllowed)
{
    TensorSize stripeCoord;
    stripeCoord.width  = (static_cast<uint32_t>(stripeId) / fmData.stripeIdStrides.width) % fmData.numStripes.width;
    stripeCoord.height = (static_cast<uint32_t>(stripeId) / fmData.stripeIdStrides.height) % fmData.numStripes.height;
    stripeCoord.channels =
        (static_cast<uint32_t>(stripeId) / fmData.stripeIdStrides.channels) % fmData.numStripes.channels;

    TensorSize stripeSize;
    stripeSize.width = (stripeCoord.width == (fmData.numStripes.width - 1U)) ? fmData.edgeStripeSize.width
                                                                             : fmData.defaultStripeSize.width;
    stripeSize.height = (stripeCoord.height == (fmData.numStripes.height - 1U)) ? fmData.edgeStripeSize.height
                                                                                : fmData.defaultStripeSize.height;
    stripeSize.channels = (stripeCoord.channels == (fmData.numStripes.channels - 1U))
                              ? fmData.edgeStripeSize.channels
                              : fmData.defaultStripeSize.channels;

    const bool isChunkingStartingMidBrick =
        (fmData.dramOffset % (8U * 8U * 16U) != 0U &&
         fmData.supertensorSizeInCells.channels != (stripeSize.channels % 16U) * fmData.numStripes.channels);

    const bool isLeftEdge  = stripeCoord.width == 0;
    const bool isTopEdge   = stripeCoord.height == 0;
    const bool isRightEdge = !isExtraPackedBoundaryDataOnRightEdge && stripeCoord.width == fmData.numStripes.width - 1U;
    const bool isBottomEdge =
        !isExtraPackedBoundaryDataOnBottomEdge && stripeCoord.height == fmData.numStripes.height - 1U;

    // The following region calculations need to take into account:
    //  * We don't want to transfer boundary data when such data does not exist (at the edge of the tensor)
    //  * We leave gaps/padding for regions which aren't relevant for this particular stripe
    //  * When packing boundary data, we always pad the Centre regions to the full default stripe size
    //    to simplify the MCE config. If we're not packing boundary data in a dimension though, we
    //    keep it compact to simplify the DMA transfers.
    //
    // See also comments in DmaCmdState.
    const PackedBoundaryThickness& boundary = packedBoundaryThickness;

    const uint32_t rightRegionWidth   = boundary.left;
    const uint32_t bottomRegionHeight = boundary.top;

    switch (region)
    {
        case DmaCmdState::Region::Centre:
        {
            // Centre region contains centre data and right/bottom data (if there is any).
            // Note that this region can never be empty. Even if there is no boundary data to load,
            // we still need the regular (non-boundary) data.
            // Note that we don't necessarily fill centreRegionWidth/Height - we might be copying less data
            // but leaving padding.
            stripeSize.width += isRightEdge ? 0 : boundary.right;
            stripeSize.height += isBottomEdge ? 0 : boundary.bottom;
            break;
        }
        case DmaCmdState::Region::Right:
        {
            // Right region contains the mid left and bottom left boundary data (if there is any)
            if (!isLeftEdge && rightRegionWidth > 0)
            {
                stripeSize.width = rightRegionWidth;
                stripeSize.height += isBottomEdge ? 0 : boundary.bottom;
            }
            else
            {
                // This region is empty
                return 0;
            }
            break;
        }
        case DmaCmdState::Region::Bottom:
        {
            // Bottom region contains the top centre and top right boundary data (if there is any)
            if (!isTopEdge && bottomRegionHeight > 0)
            {
                stripeSize.height = bottomRegionHeight;
                stripeSize.width += isRightEdge ? 0 : boundary.right;
            }
            else
            {
                // This region is empty, advance to the next one
                return 0;
            }
            break;
        }
        case DmaCmdState::Region::BottomRight:
        {
            // BottomRight region contains the top left boundary data (if there is any)
            if (!isTopEdge && bottomRegionHeight > 0 && !isLeftEdge && rightRegionWidth > 0)
            {
                stripeSize.height = bottomRegionHeight;
                stripeSize.width  = rightRegionWidth;
            }
            else
            {
                // This region is empty
                return 0;
            }
            break;
        }
        default:
            assert(false);
    }

    return CalculateNumChunksInShape(fmData.dataType, stripeSize, fmData.supertensorSizeInCells, dramStridingAllowed,
                                     isChunkingStartingMidBrick);
}

/// Constructs a DmaCmdState for the given chunk
DmaCmdState GetStateForChunkIfm(uint32_t chunkId, uint32_t stripeId, const IfmSDesc& ifmS, uint32_t numEmcs)
{
    const FmSDesc& fmData = ifmS.fmData;

    // Figure out which region we are in, from the chunkId
    const bool dramStridingAllowed = false;    // No DRAM striding for DMA read commands
    DmaCmdState::Region region     = DmaCmdState::Region::Centre;
    while (true)
    {
        uint32_t numChunksInRegion = CalculateNumChunksInRegion(
            region, ifmS.fmData, ifmS.packedBoundaryThickness, ifmS.isExtraPackedBoundaryDataOnRightEdge,
            ifmS.isExtraPackedBoundaryDataOnBottomEdge, stripeId, dramStridingAllowed);

        if (chunkId < numChunksInRegion)
        {
            break;
        }

        region = ethosn::utils::NextEnumValue(region);
        assert(region <= DmaCmdState::Region::BottomRight);
        chunkId -= numChunksInRegion;
    }

    TensorSize stripeCoord;
    stripeCoord.width  = (static_cast<uint32_t>(stripeId) / fmData.stripeIdStrides.width) % fmData.numStripes.width;
    stripeCoord.height = (static_cast<uint32_t>(stripeId) / fmData.stripeIdStrides.height) % fmData.numStripes.height;
    stripeCoord.channels =
        (static_cast<uint32_t>(stripeId) / fmData.stripeIdStrides.channels) % fmData.numStripes.channels;

    TensorSize stripeSize;
    stripeSize.width = (stripeCoord.width == (fmData.numStripes.width - 1U)) ? fmData.edgeStripeSize.width
                                                                             : fmData.defaultStripeSize.width;
    stripeSize.height = (stripeCoord.height == (fmData.numStripes.height - 1U)) ? fmData.edgeStripeSize.height
                                                                                : fmData.defaultStripeSize.height;
    stripeSize.channels = (stripeCoord.channels == (fmData.numStripes.channels - 1U))
                              ? fmData.edgeStripeSize.channels
                              : fmData.defaultStripeSize.channels;

    TensorSize dramPosition;
    dramPosition.width    = stripeCoord.width * fmData.defaultStripeSize.width;
    dramPosition.height   = stripeCoord.height * fmData.defaultStripeSize.height;
    dramPosition.channels = stripeCoord.channels * fmData.defaultStripeSize.channels;

    const bool isChunkingStartingMidBrick =
        (ifmS.fmData.dramOffset % (8U * 8U * 16U) != 0U &&
         ifmS.fmData.supertensorSizeInCells.channels != (stripeSize.channels % 16U) * ifmS.fmData.numStripes.channels);

    const bool isLeftEdge = stripeCoord.width == 0;
    const bool isTopEdge  = stripeCoord.height == 0;
    const bool isRightEdge =
        !ifmS.isExtraPackedBoundaryDataOnRightEdge && stripeCoord.width == ifmS.fmData.numStripes.width - 1U;
    const bool isBottomEdge =
        !ifmS.isExtraPackedBoundaryDataOnBottomEdge && stripeCoord.height == ifmS.fmData.numStripes.height - 1U;

    // The following region calculations need to take into account:
    //  * We don't want to transfer boundary data when such data does not exist (at the edge of the tensor)
    //  * We leave gaps/padding for regions which aren't relevant for this particular stripe
    //  * When packing boundary data, we always pad the Centre regions to the full default stripe size
    //    to simplify the MCE config. If we're not packing boundary data in a dimension though, we
    //    keep it compact to simplify the DMA transfers.
    //
    // See also comments in DmaCmdState.
    const PackedBoundaryThickness& boundary = ifmS.packedBoundaryThickness;

    const uint32_t centreRegionWidth =
        (boundary.right + boundary.left == 0)
            ? stripeSize.width
            : static_cast<uint32_t>(ifmS.fmData.defaultStripeSize.width + boundary.right);
    const uint32_t centreRegionHeight =
        (boundary.bottom + boundary.top) == 0
            ? stripeSize.height
            : static_cast<uint32_t>(ifmS.fmData.defaultStripeSize.height + boundary.bottom);
    const uint32_t rightRegionWidth   = boundary.left;
    const uint32_t bottomRegionHeight = boundary.top;

    uint32_t sramOffset          = 0;
    uint32_t sramWidthSkipPerRow = 0;

    switch (region)
    {
        case DmaCmdState::Region::Centre:
        {
            // Centre region contains centre data and right/bottom data (if there is any).
            // Note that this region can never be empty. Even if there is no boundary data to load,
            // we still need the regular (non-boundary) data.
            // Note that we don't necessarily fill centreRegionWidth/Height - we might be copying less data
            // but leaving padding.
            stripeSize.width += isRightEdge ? 0 : boundary.right;
            stripeSize.height += isBottomEdge ? 0 : boundary.bottom;

            // Leave space for Right region (containing left boundary data) and any padding within the Centre region
            sramWidthSkipPerRow = (centreRegionWidth - stripeSize.width) + rightRegionWidth;
            break;
        }
        case DmaCmdState::Region::Right:
        {
            // Right region contains the mid left and bottom left boundary data (if there is any)
            if (!isLeftEdge && rightRegionWidth > 0)
            {
                sramOffset = centreRegionWidth * 8U * utils::DivRoundUp(stripeSize.channels, numEmcs);

                // Right region data is interleaved with Centre region data
                sramWidthSkipPerRow = centreRegionWidth;

                stripeSize.width = rightRegionWidth;
                stripeSize.height += isBottomEdge ? 0 : boundary.bottom;
                dramPosition.width -= boundary.left;
            }
            else
            {
                // This region is empty
                assert(false);
            }
            break;
        }
        case DmaCmdState::Region::Bottom:
        {
            // Bottom region contains the top centre and top right boundary data (if there is any)
            if (!isTopEdge && bottomRegionHeight > 0)
            {
                sramOffset = centreRegionHeight * (centreRegionWidth + rightRegionWidth) *
                             utils::DivRoundUp(stripeSize.channels, numEmcs);

                stripeSize.height = bottomRegionHeight;
                stripeSize.width += isRightEdge ? 0 : boundary.right;
                dramPosition.height -= boundary.top;
                // Leave space for BottomRight region (containing top-left boundary data) and any padding within the Bottom region
                // (this value is probably irrelevant in practical cases, because the Bottom region
                // will always be a single row of chunks)
                sramWidthSkipPerRow = (centreRegionWidth - stripeSize.width) + rightRegionWidth;
            }
            else
            {
                // This region is empty
                assert(false);
            }
            break;
        }
        case DmaCmdState::Region::BottomRight:
        {
            // BottomRight region contains the top left boundary data (if there is any)
            if (!isTopEdge && bottomRegionHeight > 0 && !isLeftEdge && rightRegionWidth > 0)
            {
                sramOffset = (centreRegionHeight * (centreRegionWidth + rightRegionWidth) + centreRegionWidth * 8U) *
                             utils::DivRoundUp(stripeSize.channels, numEmcs);

                stripeSize.height = bottomRegionHeight;
                stripeSize.width  = rightRegionWidth;
                dramPosition.width -= boundary.left;
                dramPosition.height -= boundary.top;

                // BottomRight region data is interleaved with Bottom region data
                // (this value is probably irrelevant in practical cases, because the BottomRight region
                // will always be a single row of chunks)
                sramWidthSkipPerRow = centreRegionWidth;
            }
            else
            {
                // This region is empty
                assert(false);
            }
            break;
        }
        default:
            assert(false);
    }

    DmaCmdState state = { .region                        = DmaCmdState::Region::Centre,
                          .sramSlotOffsetForFirstChunk   = 0,
                          .dramBufferOffsetForFirstChunk = 0,
                          .chunkId                       = chunkId,
                          .sramStridePerGroupCol         = 0,
                          .sramStridePerGroupRow         = 0,
                          .isSramChannelStrided          = 0,
                          .dramStride                    = 0,
                          .chunkSize                     = { .height = 0U, .width = 0U, .channels = 0U },
                          .numChunks                     = { .height = 0U, .width = 0U, .channels = 0U } };

    ConfigureChunks(state, ifmS.fmData.dataType, stripeSize, ifmS.fmData.supertensorSizeInCells, ifmS.fmData.dramOffset,
                    dramPosition, sramOffset, sramWidthSkipPerRow, dramStridingAllowed, numEmcs,
                    isChunkingStartingMidBrick);

    return state;
}

/// Constructs a DmaCmdState for the given chunk
DmaCmdState GetStateForChunkOfm(uint32_t chunkId, uint32_t stripeId, const OfmSDesc& ofmS, uint32_t numEmcs)
{
    const FmSDesc& fmData = ofmS.fmData;

    TensorSize stripeCoord;
    stripeCoord.width  = (static_cast<uint32_t>(stripeId) / fmData.stripeIdStrides.width) % fmData.numStripes.width;
    stripeCoord.height = (static_cast<uint32_t>(stripeId) / fmData.stripeIdStrides.height) % fmData.numStripes.height;
    stripeCoord.channels =
        (static_cast<uint32_t>(stripeId) / fmData.stripeIdStrides.channels) % fmData.numStripes.channels;

    TensorSize stripeSize;
    stripeSize.width = (stripeCoord.width == (fmData.numStripes.width - 1U)) ? fmData.edgeStripeSize.width
                                                                             : fmData.defaultStripeSize.width;
    stripeSize.height = (stripeCoord.height == (fmData.numStripes.height - 1U)) ? fmData.edgeStripeSize.height
                                                                                : fmData.defaultStripeSize.height;
    stripeSize.channels = (stripeCoord.channels == (fmData.numStripes.channels - 1U))
                              ? fmData.edgeStripeSize.channels
                              : fmData.defaultStripeSize.channels;

    TensorSize dramPosition;
    dramPosition.width    = stripeCoord.width * ofmS.fmData.defaultStripeSize.width;
    dramPosition.height   = stripeCoord.height * ofmS.fmData.defaultStripeSize.height;
    dramPosition.channels = stripeCoord.channels * ofmS.fmData.defaultStripeSize.channels;

    const bool dramStridingAllowed = true;    // DRAM striding is allowed for DMA write commands

    const bool isChunkingStartingMidBrick =
        (ofmS.fmData.dramOffset % (8U * 8U * 16U) != 0U &&
         ofmS.fmData.supertensorSizeInCells.channels != (stripeSize.channels % 16U) * ofmS.fmData.numStripes.channels);

    DmaCmdState result = { .region                        = DmaCmdState::Region::Centre,
                           .sramSlotOffsetForFirstChunk   = 0,
                           .dramBufferOffsetForFirstChunk = 0,
                           .chunkId                       = chunkId,
                           .sramStridePerGroupCol         = 0,
                           .sramStridePerGroupRow         = 0,
                           .isSramChannelStrided          = 0,
                           .dramStride                    = 0,
                           .chunkSize                     = { .height = 0U, .width = 0U, .channels = 0U },
                           .numChunks                     = { .height = 0U, .width = 0U, .channels = 0U } };

    ConfigureChunks(result, ofmS.fmData.dataType, stripeSize, ofmS.fmData.supertensorSizeInCells,
                    ofmS.fmData.dramOffset, dramPosition, 0, 0, dramStridingAllowed, numEmcs,
                    isChunkingStartingMidBrick);

    return result;
}

}    // namespace

uint32_t CalculateNumChunks(const IfmSDesc& ifmS, uint32_t stripeId)
{
    uint32_t numChunks             = 0;
    const bool dramStridingAllowed = false;    // No DRAM striding for DMA read commands
    for (DmaCmdState::Region region = DmaCmdState::Region::Centre; region <= DmaCmdState::Region::BottomRight;
         region                     = ethosn::utils::NextEnumValue(region))
    {
        numChunks += CalculateNumChunksInRegion(
            region, ifmS.fmData, ifmS.packedBoundaryThickness, ifmS.isExtraPackedBoundaryDataOnRightEdge,
            ifmS.isExtraPackedBoundaryDataOnBottomEdge, stripeId, dramStridingAllowed);
    }
    return numChunks;
}

uint32_t CalculateNumChunks(const OfmSDesc& ofmS, uint32_t stripeId)
{
    const FmSDesc& fmData = ofmS.fmData;

    const bool dramStridingAllowed = true;    // DRAM striding is allowed for DMA write commands
    // Only one region (Centre) for OfmS - no packed boundary data
    uint32_t numChunks = CalculateNumChunksInRegion(DmaCmdState::Region::Centre, fmData, { 0, 0, 0, 0 }, false, false,
                                                    stripeId, dramStridingAllowed);
    return numChunks;
}

command_stream::DmaCommand GenerateDmaCommandForLoadIfmStripe(const IfmSDesc& ifmS,
                                                              uint32_t agentId,
                                                              uint32_t stripeId,
                                                              uint32_t chunkId,
                                                              const HardwareCapabilities& caps,
                                                              uint32_t nextDmaCmdId)
{
    DmaCommand result = {};
    result.type       = CommandType::LoadIfmStripe;
    result.agentId    = agentId;

    if (ifmS.fmData.dataType != FmsDataType::NHWCB && ifmS.fmData.dataType != FmsDataType::FCAF_DEEP &&
        ifmS.fmData.dataType != FmsDataType::FCAF_WIDE)
    {
        if (ifmS.packedBoundaryThickness.AnyNonZero())
        {
            throw InternalErrorException("Packed boundary not supported for this format");
        }
    }

    DmaCmdState chunkState = {};
    chunkState.numChunks   = { 1, 1, 1 };
    if (ifmS.fmData.dataType == FmsDataType::NHWCB || ifmS.fmData.dataType == FmsDataType::FCAF_DEEP ||
        ifmS.fmData.dataType == FmsDataType::FCAF_WIDE)
    {
        chunkState = GetStateForChunkIfm(chunkId, stripeId, ifmS, caps.GetNumberOfSrams());
    }

    // Write dma registers using common method
    GenerateDmaCommandCommon(ifmS.fmData, stripeId, true, result, caps, chunkState);

    // Prepare read command
    dma_rd_cmd_r rdCmd;

    switch (ifmS.fmData.dataType)
    {
        case FmsDataType::NHWC:
        {
            rdCmd.set_format(dma_format_read_t::NHWC);
            break;
        }
        case FmsDataType::FCAF_DEEP:
        {
            rdCmd.set_format(dma_format_read_t::FCAF_DEEP);
            break;
        }
        case FmsDataType::FCAF_WIDE:
        {
            rdCmd.set_format(dma_format_read_t::FCAF_WIDE);
            break;
        }
        case FmsDataType::NHWCB:
        {
            rdCmd.set_format(dma_format_read_t::NHWCB);
            break;
        }
        default:
        {
            assert(false);
            break;
        }
    }
    rdCmd.set_rd_id(nextDmaCmdId);

    // The stream type field in the cmd register is set in the firmware, not here, as this controls
    // access to external memory (e.g. NSAIDs) and so is more of a security concern, so we don't want
    // the host system's userspace to be able to change this.

    result.DMA_CMD = rdCmd.word;

    return result;
}

command_stream::DmaCommand GenerateDmaCommandForLoadWgtStripe(
    const WgtSDesc& wgtS, uint32_t agentId, uint32_t stripeId, const HardwareCapabilities& caps, uint32_t nextDmaCmdId)
{
    DmaCommand result = {};
    result.type       = CommandType::LoadWgtStripe;
    result.agentId    = agentId;

    WgtSWorkSize stripeCoord;
    stripeCoord.ifmChannels =
        (static_cast<uint32_t>(stripeId) / wgtS.stripeIdStrides.ifmChannels) % wgtS.numStripes.ifmChannels;
    stripeCoord.ofmChannels =
        (static_cast<uint32_t>(stripeId) / wgtS.stripeIdStrides.ofmChannels) % wgtS.numStripes.ofmChannels;
    const uint32_t uniqueStripeId = (stripeCoord.ofmChannels * wgtS.numStripes.ifmChannels) + stripeCoord.ifmChannels;

    // DRAM addresss
    const WeightsMetadata& weightsMetadata = (*wgtS.metadata)[uniqueStripeId];
    assert(weightsMetadata.m_Size % caps.GetNumberOfSrams() == 0);
    assert((weightsMetadata.m_Size / caps.GetNumberOfSrams()) <= wgtS.tile.slotSize &&
           "Weight stripe will not fit in slot!");

    // write DMA registers
    {
        result.m_DramOffset = weightsMetadata.m_Offset;
    }
    {
        sram_addr_r sramAddr;
        sramAddr.set_address(SramAddr(wgtS.tile, stripeId));
        result.SRAM_ADDR = sramAddr.word;
    }
    // Note that even if this stripe has less OFM channels than the number of EMCs, we still use all of the EMCs,
    // in order to be consistent with the transfer size stored in the weights metadata.
    result.DMA_EMCS = (1U << caps.GetNumberOfSrams()) - 1U;
    // n/a for WEIGHTS format: DMA_DMA_CHANNELS, DMA_DMA_STRIDEx
    const uint32_t totalBytes = weightsMetadata.m_Size;
    {
        dma_total_bytes_r tot;
        tot.set_total_bytes(totalBytes);
        result.DMA_TOTAL_BYTES = tot.word;
    }

    {
        // Prepare read command
        dma_rd_cmd_r rdCmd;
        // weights format
        rdCmd.set_format(dma_format_read_t::WEIGHTS);
        // Set cmd id (not really needed but npu_model complains if pending cmds don't have a unique id)
        rdCmd.set_rd_id(nextDmaCmdId);

        // The stream type field in the cmd register is set in the firmware, not here, as this controls
        // access to external memory (e.g. NSAIDs) and so is more of a security concern, so we don't want
        // the host system's userspace to be able to change this.

        // fields int_transfer and nhwc16 - n/a and intialized to zero by constructor
        result.DMA_CMD = rdCmd.word;
    }
    return result;
}

command_stream::DmaCommand GenerateDmaCommandForLoadPleCode(const PleLDesc& pleL,
                                                            uint32_t agentId,
                                                            const HardwareCapabilities& caps,
                                                            uint32_t nextDmaCmdId)
{
    DmaCommand result = {};
    result.type       = CommandType::LoadPleCodeIntoSram;
    result.agentId    = agentId;

    {
        sram_addr_r sramAddrReg;
        sramAddrReg.set_address(pleL.sramAddr);
        result.SRAM_ADDR = sramAddrReg.word;
    }
    {
        result.DMA_EMCS = (1U << caps.GetNumberOfEngines()) - 1U;
    }
    // Prepare read command
    dma_rd_cmd_r rdCmd;
    // Weights format
    rdCmd.set_format(dma_format_read_t::BROADCAST);
    // Set cmd id (not really needed but npu_model complains if pending cmds don't have a unique id)
    rdCmd.set_rd_id(nextDmaCmdId);

    // The stream type field in the cmd register is set in the firmware, not here, as this controls
    // access to external memory (e.g. NSAIDs) and so is more of a security concern, so we don't want
    // the host system's userspace to be able to change this.

    // fields int_transfer and nhwc16 - n/a and intialized to zero by constructor
    result.DMA_CMD = rdCmd.word;

    return result;
}

command_stream::DmaCommand GenerateDmaCommandForStoreOfmStripe(const OfmSDesc& ofmS,
                                                               uint32_t agentId,
                                                               uint32_t stripeId,
                                                               uint32_t chunkId,
                                                               const HardwareCapabilities& caps,
                                                               uint32_t nextDmaCmdId)
{
    DmaCommand result = {};
    result.type       = CommandType::StoreOfmStripe;
    result.agentId    = agentId;

    DmaCmdState chunkState = {};
    chunkState.numChunks   = { 1, 1, 1 };
    if (ofmS.fmData.dataType == FmsDataType::NHWCB || ofmS.fmData.dataType == FmsDataType::FCAF_DEEP ||
        ofmS.fmData.dataType == FmsDataType::FCAF_WIDE)
    {
        chunkState = GetStateForChunkOfm(chunkId, stripeId, ofmS, caps.GetNumberOfSrams());
    }

    // Write DMA registers using common method
    GenerateDmaCommandCommon(ofmS.fmData, stripeId, false, result, caps, chunkState);

    // The last write should be to DMA_DMA_WR_CMD, which will push the command to the HW queue
    {
        // Prepare write command
        dma_wr_cmd_r wrCmd;

        switch (ofmS.fmData.dataType)
        {
            case FmsDataType::NHWC:
            {
                wrCmd.set_format(dma_format_write_t::NHWC);
                break;
            }
            case FmsDataType::FCAF_DEEP:
            {
                wrCmd.set_format(dma_format_write_t::FCAF_DEEP);
                break;
            }
            case FmsDataType::FCAF_WIDE:
            {
                wrCmd.set_format(dma_format_write_t::FCAF_WIDE);
                break;
            }
            case FmsDataType::NHWCB:
            {
                wrCmd.set_format(chunkState.dramStride ? dma_format_write_t::NHWCB_WEIGHT_STREAMING
                                                       : dma_format_write_t::NHWCB);
                break;
            }
            default:
            {
                assert(false);
                break;
            }
        }
        // Set cmd id (not really needed but npu_model complains if pending cmds don't have a unique id)
        wrCmd.set_wr_id(nextDmaCmdId);

        // The stream type field in the cmd register is set in the firmware, not here, as this controls
        // access to external memory (e.g. NSAIDs) and so is more of a security concern, so we don't want
        // the host system's userspace to be able to change this.

        result.DMA_CMD = wrCmd.word;
    }

    return result;
}

namespace
{

uint32_t GetDmaCompConfig0Reg(const FcafInfo& fcafInfo)
{
    dma_comp_config0_r compReg;
    compReg.set_signed_activations(fcafInfo.signedActivation);
    // Zero point value can be negative but arch header file stores the data in 8 bit unsigned format
    // and hence, zero point value has to be truncated.
    compReg.set_zero_point(static_cast<uint32_t>(fcafInfo.zeroPoint) & ((1u << 8) - 1));
    return compReg.word;
}

uint32_t GetDmaStride1(const FmSDesc& fmDesc)
{
    dma_stride1_r stride1;
    if (fmDesc.dataType == FmsDataType::FCAF_DEEP || fmDesc.dataType == FmsDataType::FCAF_WIDE)
    {
        const TensorShape fcafCellShape = GetCellSize(fmDesc.dataType);
        stride1.set_outer_stride(
            static_cast<uint32_t>(fmDesc.supertensorSizeInCells.channels * GetChannels(fcafCellShape)));
    }
    else if (fmDesc.dataType == FmsDataType::NHWC)
    {
        stride1.set_outer_stride(1U * fmDesc.supertensorSizeInCells.width * fmDesc.supertensorSizeInCells.channels);
    }
    return stride1.word;
}

uint32_t GetDmaStride2(const FmSDesc& fmDesc)
{
    dma_stride2_r stride2;
    if (fmDesc.dataType == FmsDataType::FCAF_DEEP || fmDesc.dataType == FmsDataType::FCAF_WIDE)
    {
        const TensorShape fcafCellShape = GetCellSize(fmDesc.dataType);
        stride2.set_extra_stride(static_cast<uint32_t>(fmDesc.supertensorSizeInCells.width * GetWidth(fcafCellShape)) *
                                 fmDesc.supertensorSizeInCells.channels * GetChannels(fcafCellShape));
    }
    return stride2.word;
}

}    // namespace

IfmS CreateIfmS(const IfmSDesc& ifmSDesc)
{
    IfmS ifmS             = {};
    ifmS.bufferId         = ifmSDesc.fmData.bufferId;
    ifmS.DMA_COMP_CONFIG0 = GetDmaCompConfig0Reg(ifmSDesc.fmData.fcafInfo);
    ifmS.DMA_STRIDE1      = GetDmaStride1(ifmSDesc.fmData);
    ifmS.DMA_STRIDE2      = GetDmaStride2(ifmSDesc.fmData);
    return ifmS;
}

OfmS CreateOfmS(const OfmSDesc& ofmSDesc)
{
    OfmS ofmS             = {};
    ofmS.bufferId         = ofmSDesc.fmData.bufferId;
    ofmS.DMA_COMP_CONFIG0 = GetDmaCompConfig0Reg(ofmSDesc.fmData.fcafInfo);
    ofmS.DMA_STRIDE1      = GetDmaStride1(ofmSDesc.fmData);
    ofmS.DMA_STRIDE2      = GetDmaStride2(ofmSDesc.fmData);
    return ofmS;
}

}    // namespace support_library
}    // namespace ethosn
