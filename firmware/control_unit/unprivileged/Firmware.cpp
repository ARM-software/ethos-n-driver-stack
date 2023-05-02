//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "Firmware.hpp"

#include "HwAbstraction.hpp"
#include "Runner.hpp"

#include <common/Optimize.hpp>
#include <common/Utils.hpp>

#include <Capabilities.hpp>
#include <ethosn_command_stream/CommandStream.hpp>
#include <scylla_addr_fields.h>

#include <algorithm>
#include <cinttypes>
#include <type_traits>

// All multiple IG, IC, OG registers are at 4k offset
#define IOG_OFFSET 0x1000

#if defined(CE_STRIPE_MUL_ENABLE_OG1)
static_assert(TOP_REG(STRIPE_RP, CE_STRIPE_MUL_ENABLE_OG0 + IOG_OFFSET) == TOP_REG(STRIPE_RP, CE_STRIPE_MUL_ENABLE_OG1),
              "Applying offset does not result in expected register");
#endif

#if defined(CE_STRIPE_IFM_SLOT_BASE_ADDRESS_IG1)
static_assert(TOP_REG(STRIPE_RP, CE_STRIPE_IFM_SLOT_BASE_ADDRESS_IG0 + IOG_OFFSET) ==
                  TOP_REG(STRIPE_RP, CE_STRIPE_IFM_SLOT_BASE_ADDRESS_IG1),
              "Applying offset does not result in expected register");
#endif

namespace ethosn
{
namespace control_unit
{

template <typename HAL>
Firmware<HAL>::Firmware(HAL& hal, const uint64_t pleKernelDataAddr)
    : m_Hal(hal)
    , m_Pmu(hal)
    , m_ProfilingData{ m_Pmu }
    , m_BufferTable(nullptr, nullptr)
    , m_PleKernelDataAddr(pleKernelDataAddr)
{
    // Before doing anything else, describe the hardware we're running on
    const uint32_t engines  = m_Hal.NumCes().GetValue();
    const uint32_t totalIgs = engines * m_Hal.MceFeatures().get_ifm_generated_per_engine();
    const uint32_t totalOgs = engines * m_Hal.MceFeatures().get_ofm_generated_per_engine();
    // Calculate TOPS, assuming the standard frequency of 1GHz.
    const uint32_t tops     = (m_Hal.MceFeatures().get_mce_num_macs() * totalIgs * totalOgs * 2) / 1024;
    const uint32_t pleRatio = (m_Hal.NumPleLanes() * engines) / tops;
    const uint32_t sram     = m_Hal.SizeCeSram() / 1024;
    m_Hal.m_Logger.Debug("Hal configuration: %uTOPS_%uPLE_RATIO_%uKB: ces=%u, igs=%u, ogs=%u, ple lanes=%u", tops,
                         pleRatio, sram, engines, m_Hal.MceFeatures().get_ifm_generated_per_engine(),
                         m_Hal.MceFeatures().get_ofm_generated_per_engine(), m_Hal.NumPleLanes().GetValue());

    FillCapabilities();

    // Enable all events the firmware currently needs to wait for in the mask register
    // tsu event mask
    tsu_event_msk_r maskReg(0);
    maskReg.set_dma_done_mask(event_mask_t::ENABLED);
    maskReg.set_udma_or_clear_done_mask(event_mask_t::ENABLED);
    maskReg.set_ple_stripe_done_mask(event_mask_t::ENABLED);
    maskReg.set_ple_layer_done_mask(event_mask_t::ENABLED);
    maskReg.set_mce_stripe_done_mask(event_mask_t::ENABLED);
    m_Hal.WriteReg(TOP_REG(TSU_RP, TSU_TSU_EVENT_MSK), maskReg.word);

    utils::DisablePleMcuEvents(m_Hal);
}

template <typename HAL>
void Firmware<HAL>::FillCapabilities()
{
    support_library::FirmwareAndHardwareCapabilities caps;

    caps.m_Header.m_Version = FW_AND_HW_CAPABILITIES_VERSION;
    caps.m_Header.m_Size    = sizeof(caps);

    caps.m_CommandStreamBeginRangeMajor = ETHOSN_COMMAND_STREAM_VERSION_MAJOR;
    caps.m_CommandStreamBeginRangeMinor = 0;
    caps.m_CommandStreamEndRangeMajor   = ETHOSN_COMMAND_STREAM_VERSION_MAJOR;
    caps.m_CommandStreamEndRangeMinor   = ETHOSN_COMMAND_STREAM_VERSION_MINOR;

    // Hardware capabilities
    caps.m_MaxPleSize           = m_Hal.PleCodeSramSize();
    caps.m_BoundaryStripeHeight = 8;
    caps.m_NumBoundarySlots     = 8;
    // There are 4 bits as slot ID, but these need to be used for central and
    // boundary slots (see above).
    caps.m_NumCentralSlots = 8;
    caps.m_BrickGroupShape = { 1, 8, 8, 16 };
    caps.m_PatchShape      = { 1, 4, 4, 1 };
    // Total num of accumulators per engine is defined by "mce_num_acc x mce_num_macs"
    caps.m_MacUnitsPerOg          = 8;
    caps.m_AccumulatorsPerMacUnit = 64;
    caps.m_TotalAccumulatorsPerOg = caps.m_MacUnitsPerOg * caps.m_AccumulatorsPerMacUnit;

    caps.m_NumberOfEngines = m_Hal.NumCes();
    caps.m_IgsPerEngine    = m_Hal.IfmGeneratedPerCe();
    caps.m_OgsPerEngine    = m_Hal.OfmPerCe();
    caps.m_EmcPerEngine    = m_Hal.EmcPerCe();
    caps.m_TotalSramSize   = m_Hal.SizeCeSram();
    caps.m_NumPleLanes     = m_Hal.NumPleLanes();

    caps.m_WeightCompressionVersion     = m_Hal.WdFeatures().get_compression_version();
    caps.m_ActivationCompressionVersion = m_Hal.DfcFeatures().get_activation_compression();

    // Nchw at hardware level is only supported on EthosN78
    caps.m_IsNchwSupported = m_Hal.IsEthosN78();

    m_Capabilities.Resize(sizeof(caps));
    std::copy_n(reinterpret_cast<const char*>(&caps), sizeof(caps), &m_Capabilities[0]);
}

template <typename HAL>
std::pair<const char*, size_t> Firmware<HAL>::GetCapabilities() const
{
    return std::make_pair(&m_Capabilities[0], m_Capabilities.Size());
}

template <typename HAL>
typename Firmware<HAL>::InferenceResult Firmware<HAL>::RunInference(const Inference& inference)
{
    // Note this is stored even if not compiled with profiling - we always provide the inference cycle count.
    uint64_t inferenceStartTime = m_Pmu.GetCycleCount64();
    m_ProfilingData.BeginInference();    // Prevent profiling data from overwriting itself during this inference
    // There may have been a long gap between enabling profiling and running this inference. This means that
    // the PMU counter may have increased beyond what can be stored in the 32-bit timestamps that we store in
    // profiling entries, and so would overflow. This means that the driver library would be unable to reconstruct
    // the original timestamps and would have missing time. We therefore send the full timestamp at the start of the
    // inference to allow the driver library to catch up the missing time. Further entries can still be 32-bits because
    // the gap between them should be small.
    m_ProfilingData.RecordTimestampFull();
    uint8_t profilingEventId = m_ProfilingData.RecordStart(TimelineEventType::Inference);
    // A previous inference may have resulted in an error being observed by the event queue and thus failed that inference.
    // We assume that the error was transient and will not affect this inference so we clear it here.
    m_BufferTable = inference.GetBufferTable();

    const command_stream::CommandStreamParser parser = inference.GetCommandStream();
    if (!parser.IsValid())
    {
        m_Hal.m_Logger.Error("Invalid or unsupported command stream. Version reported as: %" PRIu32 ".%" PRIu32
                             ".%" PRIu32,
                             parser.GetVersionMajor(), parser.GetVersionMinor(), parser.GetVersionPatch());
        return { false, 0, {} };
    }
    const command_stream::CommandStream& cmdStream = *parser.GetData();

    // Set PWRCTLR Active for the CEs
    {
        dl2_pwrctlr_r pwrCtl;
        pwrCtl.set_active(1);
        m_Hal.WriteReg(TOP_REG(DL2_RP, DL2_PWRCTLR), pwrCtl.word);
    }

    HwAbstraction<HAL> hwAbstraction{ m_BufferTable, m_PleKernelDataAddr, m_Hal, m_ProfilingData };
    bool result = RunCommandStream(cmdStream, hwAbstraction);

    // Unset PWRCTLR Active for the CEs
    {
        dl2_pwrctlr_r pwrCtl;
        pwrCtl.set_active(0);
        m_Hal.WriteReg(TOP_REG(DL2_RP, DL2_PWRCTLR), pwrCtl.word);
    }

    m_ProfilingData.RecordEnd(profilingEventId);
    profiling::ProfilingOnly<typename profiling::ProfilingDataImpl<HAL>::NumEntriesWritten> profilingNumEntries =
        m_ProfilingData.EndInference();
    // Even when profiling disabled we still report some limited stats.
    const uint64_t inferenceCycleCount = m_Pmu.GetCycleCount64() - inferenceStartTime;
    m_ProfilingData.UpdateWritePointer();

    return { result, inferenceCycleCount, profilingNumEntries };
}

template <typename HAL>
void Firmware<HAL>::ResetAndEnableProfiling(const ethosn_firmware_profiling_configuration& config)
{
    m_Pmu.Reset(0, {});
    m_ProfilingData.Reset(config);
}

template <typename HAL>
void Firmware<HAL>::StopProfiling()
{
    m_ProfilingData.Reset();
}

}    // namespace control_unit
}    // namespace ethosn

// Because we are defining template methods in this cpp file we need to explicitly instantiate all versions
// that we intend to use.
#if defined(CONTROL_UNIT_MODEL)
#include <model/ModelHal.hpp>
template class ethosn::control_unit::Firmware<ethosn::control_unit::ModelHal>;

#include <model/UscriptHal.hpp>
template class ethosn::control_unit::Firmware<ethosn::control_unit::UscriptHal<ethosn::control_unit::ModelHal>>;

#include <model/LoggingHal.hpp>
template class ethosn::control_unit::Firmware<ethosn::control_unit::LoggingHal>;
#endif

#if defined(CONTROL_UNIT_HARDWARE)
#include <common/hals/HardwareHal.hpp>
template class ethosn::control_unit::Firmware<ethosn::control_unit::HardwareHal>;
#endif
