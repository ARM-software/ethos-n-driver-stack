//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Pmu.hpp"
#include "Profiling.hpp"

#include <common/FirmwareApi.hpp>
#include <common/Inference.hpp>

#include <array>
#include <cstdint>

namespace ethosn
{

namespace control_unit
{

template <typename HAL>
class Firmware final
{
public:
    Firmware(HAL& hal, uint64_t pleKernelDataAddr);
    ~Firmware()                     = default;
    Firmware(Firmware const& other) = delete;
    Firmware& operator=(Firmware const& other) = delete;

    /// Note this returns an opaque block of memory which is owned by this Firmware object.
    /// It does not return a concrete type because code calling this should not assume the format of the data
    /// as it may change between versions.
    std::pair<const char*, size_t> GetCapabilities() const;

    struct InferenceResult
    {
        bool success        = false;
        uint64_t cycleCount = 0;
        profiling::ProfilingOnly<typename profiling::ProfilingDataImpl<HAL>::NumEntriesWritten> numProfilingEntries;
    };

    InferenceResult RunInference(const Inference& inference);

    /// Profiling interface
    /// @{
    void ResetAndEnableProfiling(const ethosn_firmware_profiling_configuration& config);
    void StopProfiling();
    /// @}

private:
    /// Fills the m_Capabilities data from information from the HW.
    void FillCapabilities();

    HAL& m_Hal;
    Pmu<HAL> m_Pmu;
    profiling::ProfilingData<HAL> m_ProfilingData;
    BufferTable m_BufferTable;
    const uint64_t m_PleKernelDataAddr;

    /// Opaque block of data storing capabilities, filled in during construction and accessed via GetFirmware().
    /// This is not stored as a concrete type in order to discourage assumptions about the format of this data,
    /// which may change between versions.
    Vector<char, 1024> m_Capabilities;
};

}    // namespace control_unit
}    // namespace ethosn
