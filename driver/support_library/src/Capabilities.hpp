//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
// Contains the definition of the *current* version of the FirmwareAndHardwareCapabilities struct.
// This is needed by the Control Unit to construct the opaque capabilities data for the support library
// against which it is built.
//
// This header SHOULD NOT BE INCLUDED by any code other than the Control Unit or the internals of the Support Library.
// No other code should need to be aware of the layout of this struct, especially client code. These places should
// deal with the opaque 'array of bytes' instead. The types in this file should not appear in any public APIs.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#define FW_AND_HW_CAPABILITIES_VERSION 5

namespace ethosn
{
namespace support_library
{

using TensorShape = std::array<uint32_t, 4>;

/// This must always be at the start of any version of FirmwareAndHardwareCapabilities so that the Support Library
/// can decode the rest. It cannot change between versions.
struct FirmwareAndHardwareCapabilitiesHeader
{
    /// Version and size of the FirmwareAndHardwareCapabilities struct.
    /// These two fields must always be the first 8 bytes of the struct across all versions.
    /// This allows the Support Library to inspect these fields first before decoding the rest of the struct
    /// so it knows what other fields it should expect.
    /// @{
    uint32_t m_Version;
    uint32_t m_Size;
    /// @}
};

static_assert(offsetof(FirmwareAndHardwareCapabilitiesHeader, m_Version) == 0 &&
                  sizeof(FirmwareAndHardwareCapabilitiesHeader::m_Version) == 4,
              "FirmwareAndHardwareCapabilitiesHeader must start with 4 byte version.");
static_assert(offsetof(FirmwareAndHardwareCapabilitiesHeader, m_Size) == 4 &&
                  sizeof(FirmwareAndHardwareCapabilitiesHeader::m_Size) == 4,
              "FirmwareAndHardwareCapabilitiesHeader must follow with 4 byte size.");

/// The current version of the description of the firmware and hardware capabilities. This is provided to the
/// Support Library so it knows what features of the HW/FW it should compile for.
/// The Support Library may also support older versions of this struct (provided by older versions of the FW).
struct FirmwareAndHardwareCapabilities
{
    FirmwareAndHardwareCapabilitiesHeader m_Header;

    // Command stream version range
    uint32_t m_CommandStreamBeginRangeMajor;
    uint32_t m_CommandStreamBeginRangeMinor;
    uint32_t m_CommandStreamEndRangeMajor;
    uint32_t m_CommandStreamEndRangeMinor;

    // Hardware capabilities
    uint32_t m_TotalSramSize;
    uint32_t m_NumberOfEngines;
    uint32_t m_OgsPerEngine;
    uint32_t m_IgsPerEngine;
    uint32_t m_EmcPerEngine;
    uint32_t m_MaxPleSize;
    uint32_t m_BoundaryStripeHeight;
    uint32_t m_NumBoundarySlots;
    uint32_t m_NumCentralSlots;
    TensorShape m_BrickGroupShape;
    TensorShape m_PatchShape;
    uint32_t m_MacUnitsPerOg;
    uint32_t m_AccumulatorsPerMacUnit;
    uint32_t m_TotalAccumulatorsPerOg;
    uint32_t m_NumPleLanes;
    uint32_t m_WeightCompressionVersion;
    uint32_t m_ActivationCompressionVersion;
    uint32_t m_IsNchwSupported;

    // Firmware capabilities
    uint32_t m_AgentWindowSize;
    uint32_t m_MaxMceStripesPerPleStripe;
    uint32_t m_MaxIfmAndWgtStripesPerPleStripe;
};

// The FirmwareAndHardwareCapabilities struct is copied through the driver stack as a simple block of memory
// and therefore needs to remain valid when copied as such.
static_assert(std::is_trivially_copyable<FirmwareAndHardwareCapabilities>::value,
              "FirmwareAndHardwareCapabilities should always be trivially copyable.");
static_assert(offsetof(FirmwareAndHardwareCapabilities, m_Header) == 0,
              "FirmwareAndHardwareCapabilities must start with header.");
static_assert(
    std::is_same<decltype(FirmwareAndHardwareCapabilities::m_Header), FirmwareAndHardwareCapabilitiesHeader>::value,
    "FirmwareAndHardwareCapabilities must start with header.");

}    // namespace support_library
}    // namespace ethosn
