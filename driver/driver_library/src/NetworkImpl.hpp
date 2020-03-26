//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
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

protected:
    void DumpCmm(Buffer* const inputBuffers[], uint32_t numInputBuffers, const char* cmmFilename) const;

    std::vector<uint32_t> BuildInferenceData(uint64_t constantControlUnitDataBaseAddress,
                                             uint64_t constantDmaDataBaseAddress,
                                             uint64_t inputBuffersBaseAddress,
                                             uint64_t outputBuffersBaseAddress,
                                             uint64_t intermediateDataBaseAddress) const;

    support_library::CompiledNetwork& m_CompiledNetwork;
};

}    // namespace driver_library
}    // namespace ethosn
