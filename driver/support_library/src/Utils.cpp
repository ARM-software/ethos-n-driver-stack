//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "Utils.hpp"

#include "Compiler.hpp"

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

std::string ReplaceAll(std::string str, const std::string& from, const std::string& to)
{
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos)
    {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();    // Handles case where 'to' is a substring of 'from'
    }
    return str;
}

}    // namespace utils

}    // namespace support_library
}    // namespace ethosn
