//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common/Log.hpp>
#include <common/hals/HalBase.hpp>

#include <scylla_addr_fields.h>
#include <scylla_regs.h>
#include <src/veriflib/model_interface.h>

#include <chrono>

#if defined(__GNUC__)
#define ETHOSN_NO_SANITIZE_ADDRESS __attribute__((no_sanitize_address))
#else
#define ETHOSN_NO_SANITIZE_ADDRESS
#endif

namespace ethosn
{
namespace control_unit
{

// External hardware configuration parameters
struct HardwareCfgExternal
{
    uint32_t m_Tops;
    uint32_t m_PleRatio;
    uint32_t m_SramSizeKb;
};

// Internal hardware configuration parameters
struct HardwareCfgInternal
{
    uint32_t m_Ces;
    uint32_t m_Igs;
    uint32_t m_Ogs;
    uint32_t m_NumPleLanes;
    uint32_t m_SramSizeKb;
};

class ModelHal final : public HalBase<ModelHal>
{
public:
    static ModelHal CreateWithCmdLineOptions(const char* options);

    ModelHal(const char* apiTraceFilename        = nullptr,
             const char* debugLogFilename        = nullptr,
             uint64_t debugMask                  = 0,
             uint64_t debugInstMask              = 0,
             uint32_t suppressArchErrorMask      = 0,
             uint64_t debugVerbosity             = 0,
             const HardwareCfgInternal& hwCfgInt = { 2, 4, 4, 2, 448 }) ETHOSN_NO_SANITIZE_ADDRESS;

    ~ModelHal();
    ModelHal(const ModelHal& other) = delete;
    ModelHal& operator=(const ModelHal& other) = delete;
    ModelHal(ModelHal&& other)                 = default;

    void WriteReg(uint32_t regAddress, uint32_t value);

    uint32_t ReadReg(uint32_t regAddress);

    void WaitForEvents();

    /// Extension of WaitForEvents (which is not part of the HAL) with an optional timeout.
    /// A timeout of zero disables the timeout.
    void WaitForEventsWithTimeout(uint32_t timeoutMilliseconds = 0);

    void RaiseIRQ()
    {}

    void DumpDram(const char* filename, uint64_t dramAddress, uint32_t dramSize);
    void DumpSram(const char* prefix);

    bhandle_t GetBenntoHandle() const;

    void EnableDebug();
    void DisableDebug();
    void ConfigureDebug(const char* debugLogFilename,
                        uint64_t debugMask,
                        uint64_t debugInstMask,
                        uint32_t suppressArchErrorMask,
                        uint64_t debugVerbosity,
                        bool dumpPle);

    void Nop()
    {}

    LoggerType m_Logger;

private:
    bhandle_t m_BenntoHandle;
    std::chrono::high_resolution_clock::time_point m_PmuCyclesStartTime;
};

}    // namespace control_unit
}    // namespace ethosn
