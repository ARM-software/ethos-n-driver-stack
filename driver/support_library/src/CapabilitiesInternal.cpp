//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "CapabilitiesInternal.hpp"

#include <ethosn_command_stream/CommandStream.hpp>

#include <cassert>

namespace ethosn
{
namespace support_library
{

namespace
{
void SetCommonCapabilities(FirmwareAndHardwareCapabilities& fwHwCapabilities)
{
    fwHwCapabilities.m_CommandStreamBeginRangeMajor = ETHOSN_COMMAND_STREAM_VERSION_MAJOR;
    fwHwCapabilities.m_CommandStreamBeginRangeMinor = 0;
    fwHwCapabilities.m_CommandStreamEndRangeMajor   = ETHOSN_COMMAND_STREAM_VERSION_MAJOR;
    fwHwCapabilities.m_CommandStreamEndRangeMinor   = ETHOSN_COMMAND_STREAM_VERSION_MINOR;

    fwHwCapabilities.m_Header.m_Size    = sizeof(FirmwareAndHardwareCapabilities);
    fwHwCapabilities.m_Header.m_Version = FW_AND_HW_CAPABILITIES_VERSION;

    fwHwCapabilities.m_MaxPleSize           = 4096;
    fwHwCapabilities.m_BoundaryStripeHeight = 8;
    fwHwCapabilities.m_NumBoundarySlots     = 8;
    // There are 4 bits as slot ID, but these need to be used for central and
    // boundary slots (see above).
    fwHwCapabilities.m_NumCentralSlots = 8;
    fwHwCapabilities.m_BrickGroupShape = { 1, 8, 8, 16 };
    fwHwCapabilities.m_PatchShape      = { 1, 4, 4, 1 };
    // Total num of accumulators per engine is defined by "mce_num_acc x mce_num_macs"
    fwHwCapabilities.m_MacUnitsPerOg          = 8;
    fwHwCapabilities.m_AccumulatorsPerMacUnit = 64;
    fwHwCapabilities.m_TotalAccumulatorsPerOg =
        fwHwCapabilities.m_MacUnitsPerOg * fwHwCapabilities.m_AccumulatorsPerMacUnit;
}
}    // namespace

FirmwareAndHardwareCapabilities GetEthosN78FwHwCapabilities(EthosNVariant variant, uint32_t sramSize)
{
    FirmwareAndHardwareCapabilities fwHwCapabilities;

    switch (variant)
    {
        case EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO:
            // Fallthrough
        case EthosNVariant::ETHOS_N78_1TOPS_4PLE_RATIO:
            fwHwCapabilities.m_NumberOfEngines = 2;
            fwHwCapabilities.m_IgsPerEngine    = 4;
            fwHwCapabilities.m_OgsPerEngine    = 4;
            fwHwCapabilities.m_EmcPerEngine    = 4;
            fwHwCapabilities.m_TotalSramSize   = 448 * 1024;
            fwHwCapabilities.m_NumPleLanes     = (variant == EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO) ? 1 : 2;
            break;
        case EthosNVariant::ETHOS_N78_2TOPS_2PLE_RATIO:
            // Fallthrough
        case EthosNVariant::ETHOS_N78_2TOPS_4PLE_RATIO:
            fwHwCapabilities.m_NumberOfEngines = 4;
            fwHwCapabilities.m_IgsPerEngine    = 2;
            fwHwCapabilities.m_OgsPerEngine    = 4;
            fwHwCapabilities.m_EmcPerEngine    = 2;
            fwHwCapabilities.m_TotalSramSize   = 768 * 1024;
            fwHwCapabilities.m_NumPleLanes     = (variant == EthosNVariant::ETHOS_N78_2TOPS_2PLE_RATIO) ? 1 : 2;
            break;
        case EthosNVariant::ETHOS_N78_4TOPS_2PLE_RATIO:
            fwHwCapabilities.m_NumberOfEngines = 4;
            fwHwCapabilities.m_IgsPerEngine    = 4;
            fwHwCapabilities.m_OgsPerEngine    = 4;
            fwHwCapabilities.m_EmcPerEngine    = 4;
            fwHwCapabilities.m_TotalSramSize   = 1024 * 1024;
            fwHwCapabilities.m_NumPleLanes     = 2;
            break;
        case EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO:
            fwHwCapabilities.m_NumberOfEngines = 8;
            fwHwCapabilities.m_IgsPerEngine    = 2;
            fwHwCapabilities.m_OgsPerEngine    = 2;
            fwHwCapabilities.m_EmcPerEngine    = 2;
            fwHwCapabilities.m_TotalSramSize   = 1024 * 1024;
            fwHwCapabilities.m_NumPleLanes     = 2;
            break;
        case EthosNVariant::ETHOS_N78_8TOPS_2PLE_RATIO:
            fwHwCapabilities.m_NumberOfEngines = 8;
            fwHwCapabilities.m_IgsPerEngine    = 2;
            fwHwCapabilities.m_OgsPerEngine    = 4;
            fwHwCapabilities.m_EmcPerEngine    = 2;
            fwHwCapabilities.m_TotalSramSize   = 2048 * 1024;
            fwHwCapabilities.m_NumPleLanes     = 2;
            break;
        default:
            throw NotSupportedException("Unsupported variant");
            break;
    }

    if (sramSize != 0)
    {
        // EthosN78 only allows sram size per emc between 32kB and 128kB in steps of 16kB.
        // Additionally sram sizes of 56kB and 256kB are allowed
        constexpr uint32_t minSramSizePerEmcKb           = 32 * 1024;
        constexpr uint32_t maxSramSizePerEmcKb           = 128 * 1024;
        constexpr uint32_t additionalMinSramSizePerEmcKb = 56 * 1024;
        constexpr uint32_t additionalMaxSramSizePerEmcKb = 256 * 1024;
        constexpr uint32_t sramSizeIncrementPerEmcKb     = 16 * 1024;

        uint32_t numEmcs          = (fwHwCapabilities.m_NumberOfEngines * fwHwCapabilities.m_IgsPerEngine);
        uint32_t sramSizePerEmcKb = (sramSize / numEmcs);

        assert((sramSizePerEmcKb == additionalMinSramSizePerEmcKb) ||
               (sramSizePerEmcKb == additionalMaxSramSizePerEmcKb) ||
               ((sramSizePerEmcKb >= minSramSizePerEmcKb) && (sramSizePerEmcKb <= maxSramSizePerEmcKb) &&
                (sramSizePerEmcKb % sramSizeIncrementPerEmcKb == 0)));
        if (sramSizePerEmcKb < minSramSizePerEmcKb)
        {
            throw NotSupportedException(
                "User configured SRAM size is smaller than the minimum allowed for this variant");
        }
        else if ((sramSizePerEmcKb > maxSramSizePerEmcKb) && (sramSizePerEmcKb != additionalMaxSramSizePerEmcKb))
        {
            throw NotSupportedException(
                "User configured SRAM size is larger than the maximum allowed for this variant");
        }
        else if ((sramSizePerEmcKb % sramSizeIncrementPerEmcKb != 0) &&
                 (sramSizePerEmcKb != additionalMinSramSizePerEmcKb))
        {
            throw NotSupportedException("User configured SRAM size per Emc is not a multiple of 16");
        }
        fwHwCapabilities.m_TotalSramSize = sramSize;
    }

    fwHwCapabilities.m_WeightCompressionVersion     = 1;
    fwHwCapabilities.m_ActivationCompressionVersion = 1;
    fwHwCapabilities.m_IsNchwSupported              = 1;
    SetCommonCapabilities(fwHwCapabilities);
    return fwHwCapabilities;
}

FirmwareAndHardwareCapabilities GetEthosN77FwHwCapabilities()
{
    FirmwareAndHardwareCapabilities fwHwCapabilities;
    uint32_t sramPerEngine                          = 64 * 1024;
    fwHwCapabilities.m_NumberOfEngines              = 16;
    fwHwCapabilities.m_IgsPerEngine                 = 1;
    fwHwCapabilities.m_OgsPerEngine                 = 1;
    fwHwCapabilities.m_EmcPerEngine                 = 1;
    fwHwCapabilities.m_TotalSramSize                = fwHwCapabilities.m_NumberOfEngines * sramPerEngine;
    fwHwCapabilities.m_NumPleLanes                  = 1;
    fwHwCapabilities.m_WeightCompressionVersion     = 0;
    fwHwCapabilities.m_ActivationCompressionVersion = 0;
    fwHwCapabilities.m_IsNchwSupported              = 0;

    SetCommonCapabilities(fwHwCapabilities);

    return fwHwCapabilities;
}

void ValidateCapabilities(const std::vector<char>& rawCaps)
{
    // Decode the capabilities struct by looking first at the header
    if (rawCaps.size() < sizeof(FirmwareAndHardwareCapabilitiesHeader))
    {
        // Invalid size.
        throw VersionMismatchException("m_FwAndHwCapabilities is not valid");
    }

    const auto header = reinterpret_cast<const FirmwareAndHardwareCapabilitiesHeader*>(rawCaps.data());
    // For now we support only the current version.
    if (header->m_Size != sizeof(FirmwareAndHardwareCapabilities) ||
        header->m_Version != FW_AND_HW_CAPABILITIES_VERSION)
    {
        // Unsupported version.
        throw VersionMismatchException("m_FwAndHwCapabilities is not valid");
    }
}

FirmwareAndHardwareCapabilities GetValidCapabilities(const std::vector<char>& rawCaps)
{
    FirmwareAndHardwareCapabilities caps;
    ValidateCapabilities(rawCaps);

    // Now we can decode the full struct.
    memcpy(&caps, rawCaps.data(), sizeof(FirmwareAndHardwareCapabilities));

    return caps;
}

bool IsCommandStreamInRange(const FirmwareAndHardwareCapabilities& caps, const uint32_t& major, const uint32_t& minor)
{
    // If major version is start of range, check that minor version is above minimum
    // Check if major version is within range
    // If major version is end of range, check that minor version is below maximum
    if (((major == caps.m_CommandStreamBeginRangeMajor) && (minor >= caps.m_CommandStreamBeginRangeMinor)) ||
        ((major > caps.m_CommandStreamBeginRangeMajor) && (major < caps.m_CommandStreamEndRangeMajor)) ||
        ((major == caps.m_CommandStreamEndRangeMajor) && (minor <= caps.m_CommandStreamEndRangeMinor)))
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool VerifySupportedCommandStream(const FirmwareAndHardwareCapabilities& caps)
{
    return IsCommandStreamInRange(caps, ETHOSN_COMMAND_STREAM_VERSION_MAJOR, ETHOSN_COMMAND_STREAM_VERSION_MINOR);
};

}    // namespace support_library
}    // namespace ethosn
