//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "Utils.hpp"

#include "Compiler.hpp"
#include "nonCascading/McePlePass.hpp"
#include "nonCascading/Strategies.hpp"

namespace ethosn
{
namespace support_library
{

HardwareCapabilities::HardwareCapabilities(const FirmwareAndHardwareCapabilities& fwAndHwCapabilities)
{
    m_FirmwareAndHardwareCapabilities = fwAndHwCapabilities;
}

uint32_t HardwareCapabilities::GetTotalSramSize() const
{
    return m_FirmwareAndHardwareCapabilities.m_TotalSramSize;
}

uint32_t HardwareCapabilities::GetNumberOfEngines() const
{
    return m_FirmwareAndHardwareCapabilities.m_NumberOfEngines;
}

uint32_t HardwareCapabilities::GetIfmPerEngine() const
{
    return m_FirmwareAndHardwareCapabilities.m_IfmPerEngine;
}

uint32_t HardwareCapabilities::GetOfmPerEngine() const
{
    return m_FirmwareAndHardwareCapabilities.m_OfmPerEngine;
}

uint32_t HardwareCapabilities::GetNumberOfOfm() const
{
    // Return the total number of OFMs that can be generated
    return m_FirmwareAndHardwareCapabilities.m_NumberOfEngines * m_FirmwareAndHardwareCapabilities.m_OfmPerEngine;
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

const TensorShape& HardwareCapabilities::GetBrickGroupShape() const
{
    return m_FirmwareAndHardwareCapabilities.m_BrickGroupShape;
}

const TensorShape& HardwareCapabilities::GetPatchShape() const
{
    return m_FirmwareAndHardwareCapabilities.m_PatchShape;
}

uint32_t HardwareCapabilities::GetMacUnitsPerEngine() const
{
    return m_FirmwareAndHardwareCapabilities.m_MacUnitsPerEngine;
}

uint32_t HardwareCapabilities::GetTotalAccumulatorsPerEngine() const
{
    return m_FirmwareAndHardwareCapabilities.m_TotalAccumulatorsPerEngine;
}

uint32_t HardwareCapabilities::GetWeightCompressionVersion() const
{
    return m_FirmwareAndHardwareCapabilities.m_WeightCompressionVersion;
}

uint32_t HardwareCapabilities::GetNumberOfPleLanes() const
{
    return m_FirmwareAndHardwareCapabilities.m_NumPleLanes;
}

uint32_t HardwareCapabilities::GetActivationCompressionVersion() const
{
    return m_FirmwareAndHardwareCapabilities.m_ActivationCompressionVersion;
}

uint32_t HardwareCapabilities::GetIsNchwSupported() const
{
    return m_FirmwareAndHardwareCapabilities.m_IsNchwSupported;
}

namespace utils
{

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
    const uint32_t numIfmsProcessedInParallel = capabilities.GetIfmPerEngine() * capabilities.GetNumberOfEngines();
    const uint32_t numIfmsRounded = utils::RoundUpToNearestMultiple(std::get<2>(shape), numIfmsProcessedInParallel);
    uint32_t numIfmsPerCe         = isHwim ? 1 + (capabilities.GetNumberOfSrams() / 8) : numIfmsRounded;
    uint32_t numBytesPerOfm       = std::get<0>(shape) * std::get<1>(shape) * numIfmsPerCe;

    if (capabilities.GetWeightCompressionVersion() > 0)
    {
        // Worst case scenario.
        // See Ethos-N78 MCE specification 6.8.6.3.2 & 6.8.6.3.3 for more information.
        numBytesPerOfm = (numBytesPerOfm * 9 + 7) / 8;
        numBytesPerOfm += ((17 + 1 + 3 + 3 + 1 + 1 + 5 + 5 + 3 + 32 * 9) + 7) / 8;
    }

    // The weights tensor has a small header at the start of each output channel.
    numBytesPerOfm += 14;

    uint32_t numOutputChannels = std::get<3>(shape);
    if (isHwim)
    {
        numOutputChannels *= std::get<2>(shape);
    }
    const uint32_t numOfmsProducedInParallel = isHwim ? capabilities.GetNumberOfSrams() : capabilities.GetNumberOfOfm();
    uint32_t numOfmsPerIteration             = utils::DivRoundUp(numOutputChannels, numOfmsProducedInParallel);
    uint32_t numBytesPerIteration            = numBytesPerOfm * numOfmsPerIteration;
    numBytesPerIteration = static_cast<uint32_t>(command_stream::impl::RoundUp<16>(numBytesPerIteration));
    return numBytesPerIteration * numOfmsProducedInParallel;
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
        const uint32_t interleaveStride = capabilities.GetNumberOfSrams();
        // The stride is nicely aligned when the number of channels is multiple of interleaveStride
        uint32_t strideAlignment = strideX * strideY * interleaveStride;
        // Input channels remainder of the strideAlignment
        uint32_t nChannelsRemainder = nChannels % strideAlignment;

        // For a single engine the number of channels after striding is equal to the
        // original number of channels multiplied by the stride in X and Y direction.
        // When looking at the whole set of engines things change slightly.
        //
        // The example below shows a case where orignal number of channels is 16 and stride 2x2.
        // interleaveStride is 16.
        // x = active channel
        // - = non-active channel
        // CE#0 CE#1 CE#2 CE#3 CE#4 CE#5 CE#6 CE#7 CE#8 CE#9 CE#10 CE#11 CE#12 CE#13 CE#14 CE#15
        //  x(0) x    x    x    x    x    x    x    x    x    x     x     x     x     x     x
        //  x    x    x    x    x    x    x    x    x    x    x     x     x     x     x     x
        //  x    x    x    x    x    x    x    x    x    x    x     x     x     x     x     x
        //  x    x    x    x    x    x    x    x    x    x    x     x     x     x     x     x(63)
        // The global number of channels is 64 in the example above. So the number of original
        // channels is 64 / 2 * 2 = 16
        //
        // The example below shows a case where original number of channels is 3 and stride 2x2
        // CE#0 CE#1 CE#2 CE#3 CE#4 CE#5 CE#6 CE#7 CE#8 CE#9 CE#10 CE#11 CE#12 CE#13 CE#14 CE#15
        //  x(0) x    x    -    -    -    -    -    -    -    -     -     -     -     -     -
        //  x    x    x    -    -    -    -    -    -    -    -     -     -     -     -     -
        //  x    x    x    -    -    -    -    -    -    -    -     -     -     -     -     -
        //  x    x   x(50) -    -    -    -    -    -    -    -     -     -     -     -     -
        // The global number of channels is 51 (need to count non-active channels). The number of
        // original channels is equal to global number minus (strideX*strideY - 1)*16 which is
        // equal to 3
        // The formulas below generalise this concept.
        if (nChannelsRemainder)
        {
            result = interleaveStride * (nChannels / strideAlignment) + (nChannelsRemainder) -
                     (strideX * strideY - 1) * interleaveStride;
        }
        else
        {
            result = nChannels / (strideX * strideY);
        }
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

uint64_t GetPerformanceDataMetric(const PassStats& passStat)
{
    return passStat.m_Input.m_MemoryStats.m_DramParallel + passStat.m_Input.m_MemoryStats.m_DramNonParallel +
           passStat.m_Output.m_MemoryStats.m_DramParallel + passStat.m_Output.m_MemoryStats.m_DramNonParallel +
           passStat.m_Weights.m_MemoryStats.m_DramParallel + passStat.m_Weights.m_MemoryStats.m_DramNonParallel;
}

uint64_t GetMetric(const NetworkPerformanceData& netPerfData)
{
    uint64_t performanceMetric = 0;
    for (PassPerformanceData passPerfData : netPerfData.m_Stream)
    {
        performanceMetric += GetPerformanceDataMetric(passPerfData.m_Stats);
    }
    return performanceMetric;
}

bool IsLeftMoreDataPerformantThanRight(const NetworkPerformanceData& left, const NetworkPerformanceData& right)
{
    return GetMetric(left) < GetMetric(right);
}

command_stream::DataType GetCommandDataType(const DataType supportLibraryDataType)
{
    switch (supportLibraryDataType)
    {
        case DataType::UINT8_QUANTIZED:
            return command_stream::DataType::U8;
        case DataType::INT8_QUANTIZED:
            return command_stream::DataType::S8;
        default:
        {
            std::string errorMessage = "Error in " + std::string(__func__) + ": type " +
                                       std::to_string(static_cast<uint32_t>(supportLibraryDataType)) +
                                       " is not yet supported";
            throw std::invalid_argument(errorMessage);
        }
    }
}

bool IsDataTypeSigned(const DataType type)
{
    switch (type)
    {
        case DataType::UINT8_QUANTIZED:
            return false;
        case DataType::INT8_QUANTIZED:
        case DataType::INT32_QUANTIZED:
            return true;
        default:
        {
            std::string errorMessage = "Error in " + std::string(__func__) + ": DataType " +
                                       std::to_string(static_cast<uint32_t>(type)) + " not supported";
            throw std::invalid_argument(errorMessage);
        }
    }
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

command_stream::UpsampleType ConvertResizeAlgorithmToCommand(const ResizeAlgorithm algorithm)
{
    if (algorithm == ResizeAlgorithm::BILINEAR)
    {
        return command_stream::UpsampleType::BILINEAR;
    }
    else if (algorithm == ResizeAlgorithm::NEAREST_NEIGHBOUR)
    {
        return command_stream::UpsampleType::NEAREST_NEIGHBOUR;
    }
    else
    {
        assert(false);
        return command_stream::UpsampleType::OFF;
    }
}

bool IsCompressionFormatCompatibleWithStripeAndShape(const CompilerDataCompressedFormat& compressionFormat,
                                                     const TensorShape& nodeShape,
                                                     const TensorShape& stripeShape)
{
    switch (compressionFormat)
    {
        case CompilerDataCompressedFormat::NHWCB_COMPRESSED:
            // The stripe must be the full width and depth of the node input/output shape
            return stripeShape[2] >= nodeShape[2] && stripeShape[3] >= nodeShape[3];
        case CompilerDataCompressedFormat::FCAF_DEEP:
            // The stripe shape must be a multiple of the cells height (8), width (8) and depth (32)
            return (((stripeShape[1] % 8) == 0) && ((stripeShape[2] % 8) == 0) && ((stripeShape[3] % 32) == 0));
        case CompilerDataCompressedFormat::FCAF_WIDE:
            // The stripe shape must be a multiple of the cells height (8), width (16) and depth (16)
            return (((stripeShape[1] % 8) == 0) && ((stripeShape[2] % 16) == 0) && ((stripeShape[3] % 16) == 0));
        default:
            return false;
    }
}

CompilerMceAlgorithm FindBestConvAlgorithm(const HardwareCapabilities& caps, uint32_t w, uint32_t h)
{
    uint32_t numMultsDirect;
    uint32_t numMultsWinograd;

    // Only chooses WINOGRAD if it reduces the number of
    // multiplications because it adds some additional overheads
    // See the 2x2 Winograd Support Specification for further details

    // Decompose kernels with width and height > 3 into multiple 3x3, 3x1 or 1x3 sub-kernels.
    const uint32_t wideKernelSize = caps.GetWideKernelSize();
    if (w == 1 || h == 1)
    {
        // 1D convolution kernel dim w x 1 or 1 x h
        // numOfMultiplications = 2 * w or 2 * h                   DIRECT
        //                      = 4 * CEIL(W/3) or 4 * CEIL(H/3)   WINOGRAD
        numMultsDirect   = w * h * caps.GetOutputSizePerWinograd2D() * caps.GetOutputSizePerWinograd1D();
        numMultsWinograd = caps.GetMacsPerWinograd1D() * utils::DivRoundUp(w * h, wideKernelSize);
    }
    else
    {
        // 2D convolution kernel dim w x h
        // numOfMultiplications = 4 * w * h                    DIRECT
        //                      = 16 * CEIL(W/3) * CEIL(H/3)   WINOGRAD
        numMultsDirect = w * h * caps.GetOutputSizePerWinograd2D() * caps.GetOutputSizePerWinograd2D();
        numMultsWinograd =
            caps.GetMacsPerWinograd2D() * utils::DivRoundUp(w, wideKernelSize) * utils::DivRoundUp(h, wideKernelSize);
    }

    if (numMultsWinograd < numMultsDirect)
    {
        return CompilerMceAlgorithm::Winograd;
    }
    else
    {
        return CompilerMceAlgorithm::Direct;
    }
}

TensorShape GetRoundedWeights(const TensorShape& originalShape, const CompilerMceAlgorithm algorithm)
{
    TensorShape newShape = originalShape;
    if (algorithm == CompilerMceAlgorithm::Winograd ||
        (algorithm == CompilerMceAlgorithm::Direct && ((originalShape[0] > 7) || (originalShape[1] > 7))))
    {
        // WINOGRAD: width and height are rounded up to multiple of 3
        // if it is not equal to 1
        // This needs to be taken into consideration in selecting
        // memory strategy.
        // DIRECT: wide kernel mode (H or W, both > 7)
        // then both H,W are rounded up to multiple of 3
        // unless H, W = 1
        if (originalShape[0] != 1)
        {
            newShape[0] = utils::RoundUpToNearestMultiple(originalShape[0], 3);
        }

        if (originalShape[1] != 1)
        {
            newShape[1] = utils::RoundUpToNearestMultiple(originalShape[1], 3);
        }
    }

    return newShape;
}

}    // namespace utils

}    // namespace support_library
}    // namespace ethosn
