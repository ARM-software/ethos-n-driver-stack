//
// Copyright © 2018-2021 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../include/ethosn_driver_library/Buffer.hpp"
#include "../include/ethosn_driver_library/Inference.hpp"

#include <ethosn_support_library/Support.hpp>

#include <cstdint>

namespace ethosn
{
namespace driver_library
{

/// Base class for all NetworkImpls.
/// This provides the functionality to dump a combined memory map.
class NetworkImpl
{
public:
    NetworkImpl(support_library::CompiledNetwork& compiledNetwork);

    virtual ~NetworkImpl()
    {}

    /// This simple base implementation only dumps the CMM file, rather than scheduling inferences.
    virtual Inference* ScheduleInference(Buffer* const inputBuffers[],
                                         uint32_t numInputBuffers,
                                         Buffer* const outputBuffers[],
                                         uint32_t numOutputBuffers) const;

    void SetDebugName(const char* name);

protected:
    enum CmmSection : uint8_t
    {
        Cmm_ConstantDma         = 0x1,
        Cmm_ConstantControlUnit = 0x2,
        Cmm_Inference           = 0x4,
        Cmm_Ifm                 = 0x8,
        Cmm_All                 = 0xFF,
    };

    void DumpCmmBasedOnEnvVar(Buffer* const inputBuffers[], uint32_t numInputBuffers) const;

    void DumpCmm(Buffer* const inputBuffers[],
                 uint32_t numInputBuffers,
                 const char* cmmFilename,
                 uint8_t sections) const;

    std::vector<uint32_t> BuildInferenceData(uint64_t constantControlUnitDataBaseAddress,
                                             uint64_t constantDmaDataBaseAddress,
                                             uint64_t inputBuffersBaseAddress,
                                             uint64_t outputBuffersBaseAddress,
                                             uint64_t intermediateDataBaseAddress) const;

    support_library::CompiledNetwork& m_CompiledNetwork;
    std::string m_DebugName;
};

}    // namespace driver_library
}    // namespace ethosn
