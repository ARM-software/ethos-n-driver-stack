//
// Copyright © 2020-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

/// This header provides an interface to the firmware that can be used without needing to include the private
/// control unit headers, architecture headers or bennto headers, which are all implementation details of the firmware
// (and in the case of the arch headers, vary based on the variant being targeted).
/// The Driver Library uses this file and so does not need to built differently based on the variant, and does not
/// need to have dependencies on arch or bennto.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

extern "C" {
#include <ethosn_firmware.h>
}

namespace ethosn
{
namespace control_unit
{

std::vector<char> GetFirmwareAndHardwareCapabilities(const char* modelOptions);

class IModelFirmwareInterface
{
public:
    virtual ~IModelFirmwareInterface()
    {}

    static std::unique_ptr<IModelFirmwareInterface> Create(const char* modelOptions,
                                                           const char* uscriptFile,
                                                           bool uscriptUseFriendlyRegNames,
                                                           uint64_t pleKernelDataAddr);

    virtual void RecordDramLoad(uint32_t dramAddress, std::string filename) = 0;

    virtual bool LoadDram(uint64_t destAddress, const uint8_t* data, uint64_t size)      = 0;
    virtual bool LoadSram(uint32_t ceIdx,
                          uint32_t sramIdxWithinCe,
                          uint64_t destAddressWithinSram,
                          const uint8_t* data,
                          uint64_t size)                                                 = 0;
    virtual void DumpSram(const char* prefix)                                            = 0;
    virtual void ResetAndEnableProfiling(ethosn_firmware_profiling_configuration config) = 0;

    virtual bool RunInference(const std::vector<uint32_t>& inferenceData) = 0;

    virtual bool DumpDram(uint8_t* dest, uint64_t srcAddress, uint64_t size) = 0;
    virtual bool
        DumpSram(uint8_t* dest, uint32_t ceIdx, uint32_t sramIdxWithinCe, uint64_t srcAddress, uint64_t size) = 0;

    virtual uint64_t GetNumDramBytesRead() = 0;
};

}    // namespace control_unit
}    // namespace ethosn
