//
// Copyright © 2022-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "Cmsis.hpp"

#include <cstddef>
#include <cstdint>

// The MPU setup code below needs to know the location and sizes of the privileged & unprivileged stacks,
// which are defined in Boot.cpp.
extern char g_UnprivilegedStack[0x40000];
extern char g_PrivilegedStack[0x40000];

namespace ethosn
{
namespace control_unit
{

namespace
{

/// The MPU region sizes are defined with an enum, rather than the actual number of bytes.
/// This function converts from a number of bytes to the appropriate enum.
/// Not all values are valid, in which case this returns zero (which is an invalid MPU region enum).
constexpr uint8_t GetMpuRegionSize(size_t sizeBytes)
{
    switch (sizeBytes)
    {
        case 0x1000:
            return ARM_MPU_REGION_SIZE_4KB;
        case 0x2000:
            return ARM_MPU_REGION_SIZE_8KB;
        case 0x4000:
            return ARM_MPU_REGION_SIZE_16KB;
        case 0x8000:
            return ARM_MPU_REGION_SIZE_32KB;
        case 0x10000:
            return ARM_MPU_REGION_SIZE_64KB;
        case 0x20000:
            return ARM_MPU_REGION_SIZE_128KB;
        case 0x40000:
            return ARM_MPU_REGION_SIZE_256KB;
        case 0x80000:
            return ARM_MPU_REGION_SIZE_512KB;
        case 0x100000:
            return ARM_MPU_REGION_SIZE_1MB;
        case 0x200000:
            return ARM_MPU_REGION_SIZE_2MB;
        case 0x400000:
            return ARM_MPU_REGION_SIZE_4MB;
        case 0x800000:
            return ARM_MPU_REGION_SIZE_8MB;
        case 0x1000000:
            return ARM_MPU_REGION_SIZE_16MB;
        case 0x2000000:
            return ARM_MPU_REGION_SIZE_32MB;
        case 0x4000000:
            return ARM_MPU_REGION_SIZE_64MB;
        case 0x8000000:
            return ARM_MPU_REGION_SIZE_128MB;
        case 0x10000000:
            return ARM_MPU_REGION_SIZE_256MB;
        case 0x20000000:
            return ARM_MPU_REGION_SIZE_512MB;
        default:
            // Anything below 4KB or above 512MB or not power of two
            // is considered incorrect
            return 0;
    }
}

}    // namespace

// https://github.com/ARM-software/CMSIS_5/issues/532
#undef ARM_MPU_ACCESS_NORMAL
#define ARM_MPU_ACCESS_NORMAL(OuterCp, InnerCp, IsShareable)                                                           \
    ARM_MPU_ACCESS_((4U | (OuterCp)), IsShareable, ((InnerCp) >> 1), ((InnerCp)&1U))

/// Memory layout and MPU configuration.
///
/// This layout has been chosen based on various requirements, including:
///    * Data being in the appropriate NPU stream (stream 0 at 0x0, stream 1 at 0x6 and stream 2 at 0x8)
///    * MPU regions being compatible with the MPU limitations (e.g. offset and size being powers-of-two)
///
///  'S' denotes a disabled MPU subregion.
///
/// Note that this diagram is for the SMMU use case. There is a small different for the carveout use case, which
/// is that the Mailbox/profiling and Command stream assets will be offset into their respective streams, because
/// in the carveout case all NPU streams map to the same address, so they can't overlap. This doesn't affect the MPU
/// set up though, so isn't shown in the diagram.
///
/// There is also a major difference for dual core with carveout, because the vector table and stacks for the
/// second core will be after the privileged stack for the first core (note that all the code is shared between cores).
/// This means that they fall into region 1 which is Read-Write and thus not ideal.
/// There isn't an easy fix for this so we accept the reduced security as carveout is already very limited in
/// its security.
///
///      Address range    Data                                           MPU Regions          Effective MPU region and access rights
/// 0x0  =======================================================     |-----|-----|-------|    ======================================
///       0x00-0x0X       Privileged code                            |     |     |       |      2: Read-only + executable
///      -------------------------------------------------------     |     |     |       |
///       0x0X-0x0y       Unprivileged code                          |     |     |   2   |      2: Read-only + executable
///      -------------------------------------------------------     |     |     |       |
///       0x0y-0x0Y       PLE code                                   |     |     |       |      2: Read-only + executable
///      -------------------------------------------------------     |     |     |       |
///       0x0Y-0x0Z       Vector table                               |     |  1  |       |      2: Read-only + executable
///      -------------------------------------------------------     |     |     |-------|
///       0x0W-0x0V       Unprivileged stack                         |     |     |              1: Read-write
///      -------------------------------------------------------     |     |     |-------|
///       0x0V-0x0U       Privileged stack                           |     |     |   3   |      3: Read-write privileged only
///      -------------------------------------------------------     |     |     |-------|
///       0x0U-0x10       <Unused>                                   |     |     |              1: Read-write
/// 0x1  -------------------------------------------------------     |     | - - |
///       0x10-0x20       <Unused>                                   |     |     |              0: Deny
/// 0x2  =======================================================     |     |  S  |            ======================================
///       0x2-0x4         <Unused>                                   |     |     |              0: Deny
/// 0x4  -------------------------------------------------------     |     | - - |
///       0x4-0x5         Control registers                          |     |     |              1: Read-write
///      -------------------------------------------------------     |     |     |-------|
///       0x5-0x500X      Control registers (privileged only)        |  0  |  1  |   4   |      4: Read-write privileged only
///      -------------------------------------------------------     |     |     |-------|
///       0x500X-0x6      Control registers                          |     |     |              1: Read-write
/// 0x6  =======================================================     |     | - - |-------|    ======================================
///       0x60-0x6X       Mailbox                                    |     |     |   5   |      5: Read-write privileged only, or read-write if profiling enabled.
///      -------------------------------------------------------     |     | S/1 |-------|
///       0x6X-0x7        Profiling                                  |     |     |              0/1: Deny, or read-write if profiling enabled.
/// 0x7  -------------------------------------------------------     |     | - - |
///       0x7-0x8         <Unused>                                   |     |  S  |              0: Deny
/// 0x8  =======================================================     |     |-----|            ======================================
///       0x80-0x8X       Command stream                             |     |  6  |              6: Read-only
///      -------------------------------------------------------     |     |-----|
///       0x8X-0xA        <Unused>                                   |     |                    0: Deny
/// 0xA  =======================================================     |     |                  ======================================
///       0xA-0xE         <Unused>                                   |     |                    0: Deny
/// 0xE  -------------------------------------------------------     |     |-----|
///       0xE-END         System registers (PPB)                     |     |  7  |              7: Read-only + privileged writes
/// END ========================================================     |-----|-----|            ======================================

void __attribute__((used)) EnableMpu(size_t mailboxSize, size_t commandStreamSize)
{
#if defined(CONTROL_UNIT_PROFILING)
    // Profiling is in the same MPU region as the mailbox, and require non-privileged access.
    // This does reduce security, but this isn't important when profiling is enabled, as this is an
    // internal-only feature.
    constexpr uint8_t region5Access = ARM_MPU_AP_FULL;
    // Subregion mask - LSB is for lowest address (i.e. 0x0-0x1), MSB is for highest address (i.e. 0x7-0x8)
    // Mailbox size is being reported more precisely so profiling and debug monitor need to fall through
    // to a subregion if it falls outside of the mailbox region, which has to be rounded up to the nearest
    // power of two.
    constexpr uint8_t region1SubregionMask = 0b10001110;
#else
    constexpr uint8_t region5Access        = ARM_MPU_AP_PRIV;
    constexpr uint8_t region1SubregionMask = 0b11001110;
#endif

    const ARM_MPU_Region_t config[] = {
        // clang-format off
        // Region 0
        // Background memory region to avoid speculative accesses (No access)
        {
            ARM_MPU_RBAR(0, 0x00000000),
            ARM_MPU_RASR_EX(1, ARM_MPU_AP_NONE, ARM_MPU_ACCESS_ORDERED, 0, ARM_MPU_REGION_SIZE_4GB)
        },
        // Region 1
        // Big RW region that covers a lot of things.
        // Uses subregions so that some addresses fall through to the background region
        {
            ARM_MPU_RBAR(1, 0x00000000),
            ARM_MPU_RASR_EX(1, ARM_MPU_AP_FULL, ARM_MPU_ACCESS_NORMAL(ARM_MPU_CACHEP_WB_WRA, ARM_MPU_CACHEP_WB_WRA, 0),
                            region1SubregionMask, ARM_MPU_REGION_SIZE_2GB)
        },
        // Region 2
        // Firmware code and vector table. Read-only and executable access.
        {
            // This region goes up to but not including the unprivileged stack
            ARM_MPU_RBAR(2, 0x00000000),
            ARM_MPU_RASR_EX(0, ARM_MPU_AP_RO, ARM_MPU_ACCESS_NORMAL(ARM_MPU_CACHEP_WT_NWA, ARM_MPU_CACHEP_WT_NWA, 0),
                            0, GetMpuRegionSize(reinterpret_cast<size_t>(&g_UnprivilegedStack)))
        },
        // Region 3
        // Privileged stack
        {
            ARM_MPU_RBAR(3, reinterpret_cast<size_t>(&g_PrivilegedStack[0])),
            ARM_MPU_RASR_EX(1, ARM_MPU_AP_PRIV, ARM_MPU_ACCESS_NORMAL(ARM_MPU_CACHEP_WB_WRA, ARM_MPU_CACHEP_WB_WRA, 0),
                            0, GetMpuRegionSize(sizeof(g_PrivilegedStack)))
        },
        // Region 4
        // ACC interface to SEC & DL1 control registers (Privileged only read-write)
        {
            ARM_MPU_RBAR(4, 0x50000000),
            ARM_MPU_RASR_EX(1, ARM_MPU_AP_PRIV, ARM_MPU_ACCESS_DEVICE(0),
                            0, ARM_MPU_REGION_SIZE_128KB)
        },
        // Region 5
        // Mailbox (privileged read-write)
        {
            ARM_MPU_RBAR(5, 0x60000000),
            ARM_MPU_RASR_EX(1, region5Access, ARM_MPU_ACCESS_NORMAL(ARM_MPU_CACHEP_WB_WRA, ARM_MPU_CACHEP_WB_WRA, 0),
                            0, GetMpuRegionSize(mailboxSize))
        },
        // Region 6
        // Command stream (read-only)
        {
            ARM_MPU_RBAR(6, 0x80000000),
            ARM_MPU_RASR_EX(1, ARM_MPU_AP_RO, ARM_MPU_ACCESS_NORMAL(ARM_MPU_CACHEP_WT_NWA, ARM_MPU_CACHEP_WT_NWA, 0),
                            0, GetMpuRegionSize(commandStreamSize))
        },
        // Region 7
        // Private Peripheral bus - PPB, e.g. Sytem control block, MPU, etc. (Privileged read-write / Unprivileged read-only)
        {
            ARM_MPU_RBAR(7, 0xE0000000),
            ARM_MPU_RASR_EX(1, ARM_MPU_AP_URO, ARM_MPU_ACCESS_ORDERED,
                            0, ARM_MPU_REGION_SIZE_512MB)
        },
        // clang-format on
    };

    constexpr unsigned regionCnt = sizeof(config) / sizeof(ARM_MPU_Region_t);
    static_assert(regionCnt <= 8, "Number of regions must be less or equal to 8");
    ARM_MPU_Load(config, regionCnt);

    ARM_MPU_Enable(0);
}

}    // namespace control_unit
}    // namespace ethosn
