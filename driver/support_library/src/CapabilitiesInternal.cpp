//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "CapabilitiesInternal.hpp"

#include <ethosn_command_stream/CommandStream.hpp>

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
    fwHwCapabilities.m_MacUnitsPerEngine      = 8;
    fwHwCapabilities.m_AccumulatorsPerMacUnit = 64;
    fwHwCapabilities.m_TotalAccumulatorsPerEngine =
        fwHwCapabilities.m_MacUnitsPerEngine * fwHwCapabilities.m_AccumulatorsPerMacUnit;
}
}    // namespace

FirmwareAndHardwareCapabilities GetEthosN77FwHwCapabilities()
{
    FirmwareAndHardwareCapabilities fwHwCapabilities;
    uint32_t sramPerEngine             = 64 * 1024;
    fwHwCapabilities.m_NumberOfEngines = 16;
    fwHwCapabilities.m_IfmPerEngine    = 1;
    fwHwCapabilities.m_OfmPerEngine    = 1;
    fwHwCapabilities.m_EmcPerEngine    = 1;
    fwHwCapabilities.m_TotalSramSize   = fwHwCapabilities.m_NumberOfEngines * sramPerEngine;

    SetCommonCapabilities(fwHwCapabilities);

    return fwHwCapabilities;
}

FirmwareAndHardwareCapabilities GetEthosN57FwHwCapabilities()
{
    FirmwareAndHardwareCapabilities fwHwCapabilities;
    uint32_t sramPerEngine             = 64 * 1024;
    fwHwCapabilities.m_NumberOfEngines = 8;
    fwHwCapabilities.m_IfmPerEngine    = 1;
    fwHwCapabilities.m_OfmPerEngine    = 2;
    fwHwCapabilities.m_EmcPerEngine    = 1;
    fwHwCapabilities.m_TotalSramSize   = fwHwCapabilities.m_NumberOfEngines * sramPerEngine;

    SetCommonCapabilities(fwHwCapabilities);

    return fwHwCapabilities;
}

FirmwareAndHardwareCapabilities GetEthosN37FwHwCapabilities()
{
    FirmwareAndHardwareCapabilities fwHwCapabilities;
    uint32_t sramPerEngine             = 128 * 1024;
    fwHwCapabilities.m_NumberOfEngines = 4;
    fwHwCapabilities.m_IfmPerEngine    = 2;
    fwHwCapabilities.m_OfmPerEngine    = 2;
    fwHwCapabilities.m_EmcPerEngine    = 2;
    fwHwCapabilities.m_TotalSramSize   = fwHwCapabilities.m_NumberOfEngines * sramPerEngine;

    SetCommonCapabilities(fwHwCapabilities);

    return fwHwCapabilities;
}
}    // namespace support_library
}    // namespace ethosn
