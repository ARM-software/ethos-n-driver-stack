//
// Copyright © 2018-2025 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "Utils.hpp"

#include "Compiler.hpp"
#include "Part.hpp"

#include <iomanip>

namespace ethosn
{
namespace support_library
{

#if defined(ETHOSN_LOGGING)
constexpr const char SupportLibraryName[] = "support_library";
LoggerType g_Logger({ &ethosn::utils::log::sinks::StdOut<SupportLibraryName> });
#else
LoggerType g_Logger;
#endif

HardwareCapabilities::HardwareCapabilities(const FirmwareAndHardwareCapabilities& fwAndHwCapabilities)
    : m_FirmwareAndHardwareCapabilities(fwAndHwCapabilities)
{}

uint32_t HardwareCapabilities::GetTotalSramSize() const
{
    return m_FirmwareAndHardwareCapabilities.m_TotalSramSize;
}

uint32_t HardwareCapabilities::GetNumberOfEngines() const
{
    return m_FirmwareAndHardwareCapabilities.m_NumberOfEngines;
}

uint32_t HardwareCapabilities::GetIgsPerEngine() const
{
    return m_FirmwareAndHardwareCapabilities.m_IgsPerEngine;
}

uint32_t HardwareCapabilities::GetOgsPerEngine() const
{
    return m_FirmwareAndHardwareCapabilities.m_OgsPerEngine;
}

uint32_t HardwareCapabilities::GetNumberOfOgs() const
{
    // Return the total number of OFMs that can be generated
    return m_FirmwareAndHardwareCapabilities.m_NumberOfEngines * m_FirmwareAndHardwareCapabilities.m_OgsPerEngine;
}

uint32_t HardwareCapabilities::GetNumberOfSrams() const
{
    return m_FirmwareAndHardwareCapabilities.m_NumberOfEngines * m_FirmwareAndHardwareCapabilities.m_EmcPerEngine;
}

uint32_t HardwareCapabilities::GetNumberofSramsPerEngine() const
{
    return m_FirmwareAndHardwareCapabilities.m_EmcPerEngine;
}

uint32_t HardwareCapabilities::GetMaxPleSize() const
{
    return m_FirmwareAndHardwareCapabilities.m_MaxPleSize;
}

uint32_t HardwareCapabilities::GetBoundaryStripeHeight() const
{
    return m_FirmwareAndHardwareCapabilities.m_BoundaryStripeHeight;
}

uint32_t HardwareCapabilities::GetNumBoundarySlots() const
{
    return m_FirmwareAndHardwareCapabilities.m_NumBoundarySlots;
}

uint32_t HardwareCapabilities::GetNumCentralSlots() const
{
    return m_FirmwareAndHardwareCapabilities.m_NumCentralSlots;
}

uint32_t HardwareCapabilities::GetMacUnitsPerOg() const
{
    return m_FirmwareAndHardwareCapabilities.m_MacUnitsPerOg;
}

uint32_t HardwareCapabilities::GetTotalAccumulatorsPerOg() const
{
    return m_FirmwareAndHardwareCapabilities.m_TotalAccumulatorsPerOg;
}

uint32_t HardwareCapabilities::GetNumberOfPleLanes() const
{
    return m_FirmwareAndHardwareCapabilities.m_NumPleLanes;
}

namespace utils
{

const ShapeMultiplier ShapeMultiplier::Identity = g_IdentityShapeMultiplier;

uint32_t RoundDownToPow2(uint32_t x)
{
    uint32_t candidate = 1;
    while (true)
    {
        if (candidate == (1U << 31))
        {
            // This is the largest representable power of two so must be correct once we reach here.
            // We can't continue anyway, as our calculations will overflow.
            return candidate;
        }
        if (candidate * 2U > x)
        {
            return candidate;
        }
        candidate *= 2U;
    }
}

uint32_t EstimateWeightSizeBytes(const TensorShape& shape, const HardwareCapabilities& capabilities, bool isHwim)
{
    // Suppose we have 32 OFMs, we will have to assign 2 per CE. They have to be aligned
    //      in groups of numOFMs / numCEs (in this case 2).
    // The start of each group of 2 must be 16 byte aligned
    //      but within the group there are no alignment requirements.
    //
    // In the diagram below, 4 OFMs are pictured (out of our example 32):

    // H = Header
    // . = Weight
    // x = Padding
    //  <-- 16 bytes -->
    // |HHHHHHHHHHHH....|
    // |................|  - OFM 0 ┐
    // |............HHHH|          ├──────> CE 0
    // |HHHHHHHH........|  - OFM 1 ┘
    // |................|
    // |........xxxxxxxx|  - Padding
    // |HHHHHHHHHHHH....|
    // |................|  - OFM 2 ┐
    // |............HHHH|          ├──────> CE 1
    // |HHHHHHHH........|  - OFM 3 ┘
    // |................|
    // |........xxxxxxxx|  - Padding

    // For HWIM format (Depthwise), compared to 'regular' HWIO weights, we only need to specify the weights for numCes
    // number of IFMs rather than all of the IFMs.
    // Mathematically we only need to supply 1 (as each OFM is dependant on only 1 IFM), but the HW
    // requires a full set of numCes number of weights so we just set the others to zero.
    // See MCE specification 6.13 Weight Decoder and WeightEncoder.cpp in support_library for more information.
    // HWIM always uses ZERO COMPRESSION: 1 byte weight + mask (1 bit for each IG)
    const uint32_t numIfmsProcessedInParallel = capabilities.GetIgsPerEngine() * capabilities.GetNumberOfEngines();
    const uint32_t numIfmsRounded = utils::RoundUpToNearestMultiple(std::get<2>(shape), numIfmsProcessedInParallel);
    uint32_t numIfmsPerCe         = isHwim ? 1 + (capabilities.GetNumberOfSrams() / 8) : numIfmsRounded;
    uint32_t numBytesPerOfm       = std::get<0>(shape) * std::get<1>(shape) * numIfmsPerCe;

    // Worst case scenario.
    // See Ethos-N78 MCE specification 6.8.6.3.2 & 6.8.6.3.3 for more information.
    numBytesPerOfm = (numBytesPerOfm * 9 + 7) / 8;
    numBytesPerOfm += ((17 + 1 + 3 + 3 + 1 + 1 + 5 + 5 + 3 + 32 * 9) + 7) / 8;

    // The weights tensor has a small header at the start of each output channel.
    numBytesPerOfm += 14;

    uint32_t numOutputChannels = std::get<3>(shape);
    if (isHwim)
    {
        numOutputChannels *= std::get<2>(shape);
    }
    const uint32_t numOfmsProducedInParallel = isHwim ? capabilities.GetNumberOfSrams() : capabilities.GetNumberOfOgs();
    uint32_t numOfmsPerIteration             = utils::DivRoundUp(numOutputChannels, numOfmsProducedInParallel);
    uint32_t numBytesPerIteration            = numBytesPerOfm * numOfmsPerIteration;
    numBytesPerIteration = static_cast<uint32_t>(RoundUpToNearestMultiple(numBytesPerIteration, 16));
    return numBytesPerIteration * numOfmsProducedInParallel;
}

uint32_t CalculateBufferSize(const TensorShape& shape, BufferFormat dataFormat)
{
    switch (dataFormat)
    {
        case BufferFormat::FCAF_DEEP:
            return TotalSizeBytesFCAFDeep(shape);
        case BufferFormat::FCAF_WIDE:
            return TotalSizeBytesFCAFWide(shape);
        case BufferFormat::NHWCB:
            return TotalSizeBytesNHWCB(shape);
        case BufferFormat::NHWC:    // intentional fallthrough
        case BufferFormat::NCHW:
            return TotalSizeBytes(shape);
        default:
            assert(false);
            return 0;
    }
}

uint32_t
    GetNumOrigChannels(uint32_t nChannels, uint32_t strideX, uint32_t strideY, const HardwareCapabilities& capabilities)
{
    uint32_t result;
    if (strideX == 1 && strideY == 1)
    {
        result = nChannels;
    }
    else
    {
        // For a single sram the number of channels after submap decomposition is equal to the
        // original number of channels multiplied by the stride in X and Y direction.
        // When looking at the whole set of srams things change slightly.
        //
        // The example below shows a case where original number of channels is 16 and stride 2x2.
        // x = active channel
        // - = non-active channel
        // RAM0 RAM1 RAM2 RAM3 RAM4 RAM5 RAM6 RAM7 RAM8 RAM9 RAM10 RAM11 RAM12 RAM13 RAM14 RAM15
        //  x(0) x    x    x    x    x    x    x    x    x    x     x     x     x     x     x
        //  x    x    x    x    x    x    x    x    x    x    x     x     x     x     x     x
        //  x    x    x    x    x    x    x    x    x    x    x     x     x     x     x     x
        //  x    x    x    x    x    x    x    x    x    x    x     x     x     x     x     x(63)
        // The global number of channels is 64 in the example above. So the number of original
        // channels is 64 / 2 * 2 = 16
        //
        // The example below shows a case where original number of channels is 3 and stride 2x2
        // RAM0 RAM1 RAM2 RAM3 RAM4 RAM5 RAM6 RAM7 RAM8 RAM9 RAM10 RAM11 RAM12 RAM13 RAM14 RAM15
        //  x(0) x    x    -    -    -    -    -    -    -    -     -     -     -     -     -
        //  x    x    x    -    -    -    -    -    -    -    -     -     -     -     -     -
        //  x    x    x    -    -    -    -    -    -    -    -     -     -     -     -     -
        //  x    x   x(50) -    -    -    -    -    -    -    -     -     -     -     -     -
        // The global number of channels is 51 (need to count non-active channels). The number of
        // original channels is equal to global number minus (strideX*strideY - 1)*16 divided by
        // strideX*strideY which is equal to 3.
        // The formula below generalises this concept.

        const uint32_t numSrams           = capabilities.GetNumberOfSrams();
        const uint32_t fullBlocks         = strideX * strideY * numSrams;
        const uint32_t nChannelsRemainder = nChannels % numSrams;

        // The result is the number of full channel blocks times numSrams plus the remainder
        result = ((nChannels / fullBlocks) * numSrams) + nChannelsRemainder;
    }
    return result;
}

uint32_t GetNumSubmapChannels(uint32_t nChannels,
                              uint32_t strideX,
                              uint32_t strideY,
                              const HardwareCapabilities& capabilities)
{
    // These formulas are described in "MCE specification" section "Usage of IFM parameters"
    uint32_t result;
    if (strideX == 1 && strideY == 1)
    {
        result = nChannels;
    }
    else
    {
        const uint32_t interleaveStride = capabilities.GetNumberOfSrams();
        if (nChannels % interleaveStride)
        {
            result = DivRoundUp(nChannels, interleaveStride) * interleaveStride * strideX * strideY -
                     (interleaveStride - (nChannels % interleaveStride));
        }
        else
        {
            result = nChannels * strideX * strideY;
        }
    }
    return result;
}

uint32_t CalculateDramOffset(const BufferFormat dataFormat, const TensorShape& tensorSize, const TensorShape& offset)
{
    switch (dataFormat)
    {
        case BufferFormat::NHWCB:
            return utils::CalculateDramOffsetNHWCB(tensorSize, offset[1], offset[2], offset[3]);
        case BufferFormat::NHWC:
            // Deliberate fallthrough
        case BufferFormat::NCHW:
            return utils::CalculateDramOffsetNHWC(tensorSize, offset[1], offset[2], offset[3]);
        case BufferFormat::FCAF_DEEP:
            return utils::CalculateDramOffsetFcafDeep(tensorSize, offset[1], offset[2], offset[3]);
        case BufferFormat::FCAF_WIDE:
            return utils::CalculateDramOffsetFcafWide(tensorSize, offset[1], offset[2], offset[3]);
        default:
        {
            assert(false);
            return 0;
        }
    };
}

uint32_t CalculateDramOffsetNHWCB(const TensorShape& tensorShape, uint32_t offsetY, uint32_t offsetX, uint32_t offsetC)
{
    const uint32_t brickGroupSize     = GetNumElements(g_BrickGroupShape);
    const uint32_t brickGroupHeight   = GetHeight(g_BrickGroupShape);
    const uint32_t brickGroupWidth    = GetWidth(g_BrickGroupShape);
    const uint32_t brickGroupChannels = GetChannels(g_BrickGroupShape);
    const uint32_t patchSize          = GetNumElements(g_PatchShape);
    const uint32_t patchHeight        = GetHeight(g_PatchShape);
    const uint32_t patchWidth         = GetWidth(g_PatchShape);

    const uint32_t numBrickGroupDepth = utils::DivRoundUp(GetChannels(tensorShape), brickGroupChannels);
    const uint32_t numBrickGroupWidth = utils::DivRoundUp(GetWidth(tensorShape), brickGroupWidth);

    const uint32_t offsetBrickGroupX = offsetX / brickGroupWidth;
    const uint32_t offsetBrickGroupY = offsetY / brickGroupHeight;
    const uint32_t offsetBrickGroupC = offsetC / brickGroupChannels;
    const uint32_t offsetChannels    = offsetC % brickGroupChannels;
    const uint32_t offsetBrickGroups = offsetBrickGroupC + offsetBrickGroupX * numBrickGroupDepth +
                                       offsetBrickGroupY * numBrickGroupDepth * numBrickGroupWidth;
    const uint32_t offsetWithinBrickGroupX   = offsetX % brickGroupWidth;
    const uint32_t offsetWithinBrickGroupY   = offsetY % brickGroupHeight;
    const uint32_t patchWithinBrickGroupX    = offsetWithinBrickGroupX / patchWidth;
    const uint32_t patchWithinBrickGroupY    = offsetWithinBrickGroupY / patchHeight;
    const uint32_t brickGroupHeightInPatches = brickGroupHeight / patchHeight;
    const uint32_t brickWithinBrickGroup  = patchWithinBrickGroupX * brickGroupHeightInPatches + patchWithinBrickGroupY;
    const uint32_t offsetWithinBrickGroup = (brickWithinBrickGroup * brickGroupChannels + offsetChannels) * patchSize;

    const uint32_t offsetBytes = brickGroupSize * offsetBrickGroups + offsetWithinBrickGroup;

    return offsetBytes;
}

namespace
{

uint32_t CalculateCellIdx(const TensorShape& tensorShape, const TensorShape& offset, const TensorShape& cellShape)
{
    // It's not possible to have an offset partway through a cell
    assert(GetWidth(offset) % GetWidth(cellShape) == 0);
    assert(GetHeight(offset) % GetHeight(cellShape) == 0);
    assert(GetChannels(offset) % GetChannels(cellShape) == 0);
    const uint32_t totalCellsX = utils::DivRoundUp(GetWidth(tensorShape), GetWidth(cellShape));
    const uint32_t totalCellsC = utils::DivRoundUp(GetChannels(tensorShape), GetChannels(cellShape));
    const uint32_t cellX       = GetWidth(offset) / GetWidth(cellShape);
    const uint32_t cellY       = GetHeight(offset) / GetHeight(cellShape);
    const uint32_t cellC       = GetChannels(offset) / GetChannels(cellShape);

    return cellC + cellX * totalCellsC + cellY * totalCellsC * totalCellsX;
}

}    // namespace

uint32_t CalculateDramOffsetNHWC(const TensorShape& tensorShape, uint32_t offsetY, uint32_t offsetX, uint32_t offsetC)
{
    return offsetC + offsetX * GetChannels(tensorShape) + offsetY * GetChannels(tensorShape) * GetWidth(tensorShape);
}

inline uint32_t
    CalculateDramOffsetFcafDeep(const TensorShape& tensorShape, uint32_t offsetY, uint32_t offsetX, uint32_t offsetC)
{
    return g_FcafCellSizeBytes *
           CalculateCellIdx(tensorShape, TensorShape{ 1, offsetY, offsetX, offsetC }, g_FcafDeepCellShape);
}

inline uint32_t
    CalculateDramOffsetFcafWide(const TensorShape& tensorShape, uint32_t offsetY, uint32_t offsetX, uint32_t offsetC)
{
    return g_FcafCellSizeBytes *
           CalculateCellIdx(tensorShape, TensorShape{ 1, offsetY, offsetX, offsetC }, g_FcafWideCellShape);
}

DataTypeRange GetRangeOfDataType(const DataType type)
{
    switch (type)
    {
        case DataType::UINT8_QUANTIZED:
            return GetTypeLimits<uint8_t>();
        case DataType::INT8_QUANTIZED:
            return GetTypeLimits<int8_t>();
        case DataType::INT32_QUANTIZED:
            return GetTypeLimits<int32_t>();
        default:
        {
            std::string errorMessage = "Error in " + std::string(__func__) + ": DataType " +
                                       std::to_string(static_cast<uint32_t>(type)) + " not supported";
            throw std::invalid_argument(errorMessage);
        }
    }
}

bool IsCompressionFormatCompatibleWithStripeShape(CompilerDataCompressedFormat compressionFormat,
                                                  const TensorShape& stripeShape,
                                                  const TensorShape& dramTensorShape)
{
    TensorShape cellShape;
    switch (compressionFormat)
    {
        case CompilerDataCompressedFormat::FCAF_DEEP:
            cellShape = g_FcafDeepCellShape;
            break;
        case CompilerDataCompressedFormat::FCAF_WIDE:
            cellShape = g_FcafWideCellShape;
            break;
        default:
            return false;
    }
    // The stripe shape must be a multiple of the cell shape for all dimensions in which there are multiple
    // stripes. If there is only a single stripe in that dimension, then it doesn't matter.
    for (uint32_t dim = 0; dim < 4; ++dim)
    {
        uint32_t numStripes = DivRoundUp(dramTensorShape[dim], stripeShape[dim]);
        if (numStripes > 1 && (stripeShape[dim] % cellShape[dim]) != 0)
        {
            return false;
        }
    }
    return true;
}

constexpr bool FilterToSize(const BlockConfig& blockConfig, uint32_t width, uint32_t height)
{
    return blockConfig == BlockConfig{ width, height };
}

bool FilterToSizes(const BlockConfig& blockConfig, const std::initializer_list<BlockConfig>& allowedConfigs)
{
    return std::find(allowedConfigs.begin(), allowedConfigs.end(), blockConfig) != allowedConfigs.end();
}

std::vector<BlockConfig> FilterPleBlockConfigs(const PleOperation pleOp,
                                               const std::vector<BlockConfig>& allowedBlockConfigs)
{
    std::vector<BlockConfig> res = allowedBlockConfigs;

    if (pleOp == PleOperation::DOWNSAMPLE_2X2)
    {
        auto filter = [](const auto& blockConfig) {
            return FilterToSizes(blockConfig, { { 16U, 8U }, { 32U, 8U }, { 16U, 16U }, { 8U, 8U } });
        };
        res = Filter(res, filter);
    }
    else if (pleOp == PleOperation::INTERLEAVE_2X2_2_2)
    {
        auto filter = [](const auto& blockConfig) { return FilterToSize(blockConfig, 16, 16); };
        res         = Filter(res, filter);
    }
    else if (pleOp == PleOperation::MAXPOOL_2X2_2_2)
    {
        // MaxPool 2x2 2,2 supports only 16x16, 32x8, 8x8, 16x8
        auto filter = [](const auto& blockConfig) {
            return FilterToSizes(blockConfig, { { 16U, 16U }, { 32U, 8U }, { 8U, 8U }, { 16U, 8U } });
        };
        res = Filter(res, filter);
    }
    else if ((pleOp == PleOperation::MEAN_XY_7X7) || (pleOp == PleOperation::MEAN_XY_8X8))
    {
        auto filter = [](const auto& blockConfig) { return FilterToSize(blockConfig, 8, 8); };
        res         = Filter(res, filter);
    }
    else if (pleOp == PleOperation::MAXPOOL_3X3_2_2_EVEN || pleOp == PleOperation::MAXPOOL_3X3_2_2_ODD)
    {
        // The maxpool 3x3_2_2 and avgpool 3x3_1_1 ple kernels only support 8x8, 32x8 blocks
        auto filter = [](const auto& blockConfig) { return FilterToSizes(blockConfig, { { 32U, 8U }, { 8U, 8U } }); };
        res         = Filter(res, filter);
    }
    else if (pleOp == PleOperation::TRANSPOSE_XY)
    {
        // The transpose_xy ple kernel only support 8x8 blocks
        auto filter = [&](const auto& blockConfig) { return FilterToSizes(blockConfig, { { 8U, 8U } }); };

        res = Filter(res, filter);
    }

    return res;
}

bool PleBlockConfigAllowed(const PleOperation pleOp, const BlockConfig& allowedBlockConfig)
{
    std::vector<BlockConfig> res;

    res = FilterPleBlockConfigs(pleOp, { allowedBlockConfig });

    return (!res.empty());
}

unsigned CalculateSpaceToDepthSramUsage(uint32_t blockSize, uint32_t s1, uint32_t s2)
{
    // Without optimizing the SRAM usage, the algorithm would need s1 * blockSize + s2 * blockSize bytes / EMC.
    // However, by overwriting data in SRAM from the first pass that's no longer needed in the second pass of the
    // algorithm, SRAM requirement can be reduced to s1 + max(s1, s2) * (blockSize - 1) + s2.
    // This is achieved by writing data to the start of SRAM in the first pass, but write data starting at the end
    // of the SRAM in the second pass. Eventually, data written in the second pass will overwrite data from the first
    // pass but when this happens, the data that's overwritten isn't needed anymore.
    return s1 + std::max(s1, s2) * (blockSize - 1) + s2;
}

std::pair<uint32_t, uint32_t>
    CalculateSpaceToDepthBlockSizes(const TensorShape tensor, uint32_t usedSrams, uint32_t blockSize)
{
    // Size of the subtensors produced in the first pass in bytes per EMC
    // Subtensor dimension: (ifmHeight / blockSize, ifmWidth * ifmChannels / usedSrams, usedSrams)
    // Note: The purpose of the divisions by 8 is to align the dimensions to 8x8.
    uint32_t s1 = utils::DivRoundUp(GetWidth(tensor) * GetChannels(tensor), usedSrams * 8) *
                  utils::DivRoundUp(GetHeight(tensor), blockSize * 8) * 64;

    // Size of the subtensors produced in the second pass in bytes per EMC
    // Subtensor dimension: (ifmWidth * ifmHeight / blockSize^2, blockSize * ifmChannels / usedSrams, usedSrams)
    uint32_t s2 = utils::DivRoundUp(blockSize * GetChannels(tensor), usedSrams * 8) *
                  utils::DivRoundUp(GetWidth(tensor) * GetHeight(tensor), blockSize * blockSize * 8) * 64;

    return std::pair<uint32_t, uint32_t>(s1, s2);
}

std::tuple<bool, bool, bool> IsSplitting(const TensorShape& tensorShape, const TensorShape& stripeShape)
{
    using namespace utils;
    const bool splitH = GetHeight(stripeShape) < GetHeight(tensorShape);
    const bool splitW = GetWidth(stripeShape) < GetWidth(tensorShape);
    const bool splitC = GetChannels(stripeShape) < GetChannels(tensorShape);
    return std::make_tuple(splitH, splitW, splitC);
}

bool IsFullTensor(const TensorShape& tensorShape, const TensorShape& stripeShape)
{
    return GetHeight(stripeShape) >= GetHeight(tensorShape) && GetWidth(stripeShape) >= GetWidth(tensorShape) &&
           GetChannels(stripeShape) >= GetChannels(tensorShape);
}

bool CheckOverlap(uint32_t startA, uint32_t sizeA, uint32_t startB, uint32_t sizeB)
{
    const uint32_t endA = startA + sizeA;
    const uint32_t endB = startB + sizeB;
    return (startA <= startB && endA > startB) || (startB <= startA && endB > startA);
}

}    // namespace utils

}    // namespace support_library
}    // namespace ethosn
