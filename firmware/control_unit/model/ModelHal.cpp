//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "include/model/ModelHal.hpp"

#include <common/Log.hpp>
#include <common/Utils.hpp>

#include <scylla_addr_fields.h>
#include <scylla_regs.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cinttypes>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

namespace ethosn
{
namespace control_unit
{

namespace
{

// clang-format off
std::map<std::string, uint64_t> g_BenntoDebugVerbosityLookup =
{
    { "NONE", BDEBUG_VERB_NONE },
    { "INFO", BDEBUG_VERB_INFO },
    { "IFACE", BDEBUG_VERB_IFACE },
    { "LOW", BDEBUG_VERB_LOW },
    { "MED", BDEBUG_VERB_MED },
    { "HIGH", BDEBUG_VERB_HIGH }
};

std::map<std::string, uint64_t> g_BenntoDebugInstMaskLookup =
{
    { "SINGLE", 1},
    { "ALL"   , BDEBUG_INST_ALL},
};

std::map<std::string, uint64_t> g_BenntoDebugMaskLookup =
{
    { "ALL", BDEBUG_ALL },
    { "CONFIG", BDEBUG_CONFIG },
    { "STATS", BDEBUG_STATS },
    { "DMA", BDEBUG_DMA },
    { "CMD_STREAM", BDEBUG_CMD_STREAM },
    { "TSU", BDEBUG_TSU },
    { "WIT", BDEBUG_WIT },
    { "WD", BDEBUG_WD },
    { "MAC", BDEBUG_MAC },
    { "WFT", BDEBUG_WFT },
    { "PLE", BDEBUG_PLE },
    { "CESRAM", BDEBUG_CESRAM },
    { "DATABLOCK", BDEBUG_DATABLOCK },
    { "PLE_CMD", BDEBUG_PLE_CMD },
    { "EVENTQ", BDEBUG_EVENTQ },
    { "FASTMODEL", BDEBUG_FASTMODEL },
    { "NCU", BDEBUG_NCU },
    { "MCU_DEBUG", BDEBUG_MCU_DEBUG }
};

std::map<uint8_t, bcesram_t> g_CeSramLookup =
{
    {0, BCESRAM_CE_SRAM0},
    {1, BCESRAM_CE_SRAM1},
    {2, BCESRAM_CE_SRAM2},
    {3, BCESRAM_CE_SRAM3}
};
// clang-format on

std::string ToUpper(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](const char c) { return std::toupper(c); });
    return s;
}

std::vector<std::string> Split(std::string s, char delim)
{
    std::stringstream ss(s);
    std::string token;
    std::vector<std::string> results;
    while (std::getline(ss, token, delim))
    {
        results.push_back(token);
    }
    return results;
}

HardwareCfgInternal ConvertAndValidateEthosN78ConfigurationOptions(const HardwareCfgExternal& hwCfgExt)
{
    HardwareCfgInternal hwCfgInt;

    // Perform range check on the configuration parameters. EthosN78 supports only certain combination
    // of configuration parameters
    ASSERT_MSG(((((hwCfgExt.m_Tops == 1) || (hwCfgExt.m_Tops == 2) || (hwCfgExt.m_Tops == 4)) &&
                 ((hwCfgExt.m_PleRatio == 2) || (hwCfgExt.m_PleRatio == 4))) ||
                ((hwCfgExt.m_Tops == 8) && (hwCfgExt.m_PleRatio == 2))),
               "Unsupported EthosN78 configuration");

    ASSERT_MSG(((hwCfgExt.m_SramSizeKb >= 384) && (hwCfgExt.m_SramSizeKb <= 4096)),
               "EthosN78 only supports sramSizeKb >=384 and <=4096");

    // Derive internal parameters m_Ces, m_Igs, m_Ogs, and m_PleLanes
    switch (hwCfgExt.m_Tops)
    {
        // 1Tops variants have 2 ces, 4 igs and 4 ogs
        case 1:
        {
            hwCfgInt.m_Ces = 2;
            hwCfgInt.m_Igs = 4;
            hwCfgInt.m_Ogs = 4;
        }
        break;
        // 2Tops variants have 4 ces, 2 igs and 4 ogs
        case 2:
        {
            hwCfgInt.m_Ces = 4;
            hwCfgInt.m_Igs = 2;
            hwCfgInt.m_Ogs = 4;
        }
        break;
        // 4Tops variants have 4 ces, 4 igs and 4 ogs or 8 ces, 2 igs and 2 ogs
        case 4:
        {
            hwCfgInt.m_Ces = (hwCfgExt.m_PleRatio == 2) ? 4 : 8;
            hwCfgInt.m_Igs = (hwCfgExt.m_PleRatio == 2) ? 4 : 2;
            hwCfgInt.m_Ogs = (hwCfgExt.m_PleRatio == 2) ? 4 : 2;
        }
        break;
        // 8Tops variants have 8 ces, 2 igs and 4 ogs
        case 8:
        {
            hwCfgInt.m_Ces = 8;
            hwCfgInt.m_Igs = 2;
            hwCfgInt.m_Ogs = 4;
        }
        break;
        default:
            ASSERT("Invalid EthosN78 hardware configuration");
            break;
    }

    // Assign sramSize
    hwCfgInt.m_SramSizeKb = hwCfgExt.m_SramSizeKb;

    // Ple ratio is defined as "The ratio between the number of PLE lanes in the NPU and MCE compute capacity in TOPs"
    // i,e Ple ratio = (Number of ple lanes per ce * number of ces)/tops.
    // Given Ple ratio, the above equation can be rearranged to get number of ple lanes per ce.
    hwCfgInt.m_NumPleLanes = ((hwCfgExt.m_PleRatio * hwCfgExt.m_Tops) / hwCfgInt.m_Ces);

    ASSERT_MSG(((hwCfgInt.m_NumPleLanes == 1) || (hwCfgInt.m_NumPleLanes == 2)),
               "EthosN78 only supports 1 or 2 ple lanes");

    uint32_t totalSramCnt = hwCfgInt.m_Ces * hwCfgInt.m_Igs;

    // Sram size per emc can be anywhere between 32kB to 128kB in steps if 16kB and
    // additional configurations of 56kB and 256kB are allowed
    ASSERT_MSG(
        ((hwCfgExt.m_SramSizeKb == (56 * totalSramCnt)) || (hwCfgExt.m_SramSizeKb == (256 * totalSramCnt)) ||
         ((hwCfgExt.m_SramSizeKb >= (32 * totalSramCnt)) && (hwCfgExt.m_SramSizeKb <= (128 * totalSramCnt)) &&
          (hwCfgExt.m_SramSizeKb % (16 * totalSramCnt) == 0))),
        "Invalid Sram size per emc, EthosN78 supports 56kB, 256kB, and anything between 32kB-128kB in steps of 16kB");

    return hwCfgInt;
}
}    // namespace

ModelHal ModelHal::CreateWithCmdLineOptions(const char* options)
{
    std::string apiTraceFilename;
    std::string debugLogFilename;
    uint64_t debugMask = BDEBUG_NONE;
    // Disable logging for all but the first CE, as this reduces log spam and is nearly always enough.
    uint64_t debugInstMask         = 1;
    uint32_t suppressArchErrorMask = 0;
    uint64_t debugVerbosity        = BDEBUG_VERB_NONE;
    // Default configuration
    HardwareCfgExternal hwCfgExt = { 1, 4, 448 };
    HardwareCfgInternal hwCfgInt = { 2, 4, 4, 2, 448 };

    // Extract ModelHal constructor arguments from command line
    for (auto option : Split(options, ' '))
    {
        auto optionPair         = Split(option, '=');
        std::string optionName  = optionPair[0];
        std::string optionValue = optionPair[1];

        if (optionName == "tops")
        {
            hwCfgExt.m_Tops = static_cast<uint32_t>(stoul(optionValue));
        }
        else if (optionName == "ple_ratio")
        {
            hwCfgExt.m_PleRatio = static_cast<uint32_t>(stoul(optionValue));
        }
        else if (optionName == "sram_size_kb")
        {
            hwCfgExt.m_SramSizeKb = static_cast<uint32_t>(stoul(optionValue));
        }
        else if (optionName == "trace")
        {
            apiTraceFilename = optionValue;
        }
        else if (optionName == "log")
        {
            debugLogFilename = optionValue;
        }
        else if (optionName == "inst_mask")
        {
            auto maskIt = g_BenntoDebugInstMaskLookup.find(ToUpper(optionValue));
            if (maskIt == g_BenntoDebugInstMaskLookup.end())
            {
                throw std::invalid_argument(std::string("Unknown debug Inst mask: ") + optionValue);
            }
            debugInstMask = maskIt->second;
        }
        else if (optionName == "mask")
        {
            // Parse the human-readable mask string into the bennto bit mask.
            // It is of the format "PLE|MAC|BLARG"

            // Reset in case it was set to a new default by the verbosity code block below.
            debugMask = BDEBUG_NONE;

            for (auto mask : Split(optionValue, '|'))
            {
                auto maskIt = g_BenntoDebugMaskLookup.find(ToUpper(mask));
                if (maskIt == g_BenntoDebugMaskLookup.end())
                {
                    throw std::invalid_argument(std::string("Unknown debug mask: ") + mask);
                }
                debugMask |= maskIt->second;
            }
            // If some debugging bits have been enabled then make sure the verbosity is high enough to show some messages.
            if (debugMask != BDEBUG_NONE && debugVerbosity == BDEBUG_VERB_NONE)
            {
                debugVerbosity = BDEBUG_VERB_IFACE | BDEBUG_VERB_INFO;
            }
        }
        else if (optionName == "suppress_arch_error_mask")
        {
            suppressArchErrorMask = static_cast<uint32_t>(strtoul(optionValue.c_str(), NULL, 0));
        }
        else if (optionName == "verbosity")
        {
            // Parse the human-readable verbosity string into the bennto constant value.

            // Reset in case it was set to a new default by the mask code block above.
            debugVerbosity = BDEBUG_VERB_NONE;

            auto verbIt = g_BenntoDebugVerbosityLookup.find(ToUpper(optionValue));
            if (verbIt == g_BenntoDebugVerbosityLookup.end())
            {
                throw std::invalid_argument(std::string("Unknown debug verbosity: ") + optionValue);
            }
            debugVerbosity = verbIt->second;
            // If the verbosity has been turned up then turn on some debugging bits so that some messages are shown
            if (debugVerbosity != BDEBUG_VERB_NONE && debugMask == BDEBUG_NONE)
            {
                debugMask = BDEBUG_ALL;
            }
        }
        else
        {
            throw std::invalid_argument(std::string("Unknown ModelHal option: ") + option);
        }
    }

    // Convert External to internal configuration and validate parameters
    hwCfgInt = ConvertAndValidateEthosN78ConfigurationOptions(hwCfgExt);

    return ModelHal(apiTraceFilename.empty() ? nullptr : apiTraceFilename.c_str(),
                    debugLogFilename.empty() ? nullptr : debugLogFilename.c_str(), debugMask, debugInstMask,
                    suppressArchErrorMask, debugVerbosity, hwCfgInt);
}

ModelHal::ModelHal(const char* apiTraceFilename,
                   const char* debugLogFilename,
                   uint64_t debugMask,
                   uint64_t debugInstMask,
                   uint32_t suppressArchErrorMask,
                   uint64_t debugVerbosity,
                   const HardwareCfgInternal& hwCfgInt)
    : HalBase(m_Logger)
    , m_Logger({ LogSink })
{
    berror res;

    bennto_init();

    if (apiTraceFilename)
    {
        res = bennto_init_api_trace(apiTraceFilename);
        ASSERT_MSG(res == BERROR_OK, "bennto_init_api_trace failed");
    }
    res = bennto_begin_boilerplate_instance(BCONFIG_TOP_LEVEL, "default model", &m_BenntoHandle);
    ASSERT_MSG(res == BERROR_OK, "bennto_begin_boilerplate_instance failed");

    // Set ces, igs, ogs, ples, sram_size_kb of N78 config
    res = bennto_set_n78_config(m_BenntoHandle, hwCfgInt.m_Ces, hwCfgInt.m_Igs, hwCfgInt.m_Ogs,
                                hwCfgInt.m_NumPleLanes - 1, hwCfgInt.m_SramSizeKb);
    ASSERT_MSG(res == BERROR_OK, "Unable to configure for N78");

    res = bennto_set_config(m_BenntoHandle, "dma.variant", BWD_N78);
    ASSERT_MSG(res == BERROR_OK, "Unable to configure DMA variant for N78");

    res = bennto_set_config(m_BenntoHandle, "wd.variant", BWD_N78);
    ASSERT_MSG(res == BERROR_OK, "Unable to configure Weight decoder variant for N78");

    res = bennto_set_config(m_BenntoHandle, "wft.variant", BWD_N78);
    ASSERT_MSG(res == BERROR_OK, "Unable to configure Weight encoder variant for N78");

    res = bennto_set_config(m_BenntoHandle, "dma.deferred_execute", 1);
    ASSERT_MSG(res == BERROR_OK, "bennto_set_config failed");

    res = bennto_set_config(m_BenntoHandle, "dma.strict_id_check", 1);
    ASSERT_MSG(res == BERROR_OK, "bennto_set_config failed");

    res = bennto_set_config(m_BenntoHandle, "dma.nhwcb_exact_channels", 1);
    ASSERT_MSG(res == BERROR_OK, "bennto_set_config failed");

    res = bennto_set_config(m_BenntoHandle, "ple.enable_fastmodel", 1);
    ASSERT_MSG(res == BERROR_OK, "bennto_set_config failed");

    res = bennto_set_config(m_BenntoHandle, "ple.timeout_cycles", 10000000U);
    ASSERT_MSG(res == BERROR_OK, "bennto_set_config failed");

    res = bennto_set_config(m_BenntoHandle, "tsu.requireAllPleStripeDones", 0);
    ASSERT_MSG(res == BERROR_OK, "bennto_set_config failed");

    // Make the model consume 1 stripe done event at a time.
    res = bennto_set_config(m_BenntoHandle, "verif.advance_single_events", bevent_mask_t::BEVENT_MASK_MCE_MAC_BATCH);
    ASSERT_MSG(res == BERROR_OK, "bennto_set_config failed");

    // Disable this check which is overly restrictive and makes it harder for us to calculate the IFM delta values.
    res = bennto_set_config(m_BenntoHandle, "verif.check_ifm_parameters", 0);
    ASSERT_MSG(res == BERROR_OK, "bennto_set_config failed");

    // Enable unbuffered output in case we crash.
    res = bennto_set_debug_output_unbuffered(m_BenntoHandle, true);
    ASSERT_MSG(res == BERROR_OK, "bennto_set_debug_output_unbuffered failed");

    // If at least some debugging has been enabled, also enable the PLE dumps.
    const bool someDebuggingEnabled = (debugMask != 0 && debugVerbosity != 0);
    ConfigureDebug(debugLogFilename, debugMask, debugInstMask, suppressArchErrorMask, debugVerbosity,
                   someDebuggingEnabled);

    res = bennto_create_instance(m_BenntoHandle);
    ASSERT_MSG(res == BERROR_OK, "bennto_create_instance failed");
}

ModelHal::~ModelHal()
{
    berror res = bennto_destroy_instance(m_BenntoHandle);
    ASSERT_MSG(res == BERROR_OK, "Bennto destroy failed");

    res = bennto_fini_api_trace();    // Note this is safe even if we didn't call bennto_init_api_trace
    ASSERT_MSG(res == BERROR_OK, "bennto_fini_api_trace failed");

    bennto_fini();
}

void ModelHal::WriteReg(uint32_t regAddress, uint32_t value)
{
    if (regAddress == TOP_REG(PMU_RP, PMU_PMCR))
    {
        // Because we model the PMU cycle counter ourselves (see below), record the time at which it was
        // reset so that we can simulate it properly.
        pmcr_r pmcr = value;
        if (pmcr.get_cycle_cnt_rst())
        {
            m_PmuCyclesStartTime = std::chrono::high_resolution_clock::now();
        }
    }

    berror res = bennto_write_config_reg(m_BenntoHandle, regAddress, value);
    ASSERT_MSG(res == BERROR_OK, "Bennto reports error when writing %08" PRIx32 " to 0x%08" PRIx32, value, regAddress);
}

uint32_t ModelHal::ReadReg(uint32_t regAddress)
{
    if (regAddress == TOP_REG(PMU_RP, PMU_PMCCNTR_LO))
    {
        // When running on the model, we do have the modelled PMU in bennto, but this doesn't produce results which
        // look as nice on the timeline graph (e.g. many entries are recorded at the same time the counter doesn't
        // advance very often). Instead we use the wall clock. This means the events won't be related as much to the
        // timings on the real hardware, but they give a better indication of what is going on inside the firmware,
        // which is arguably more useful in this case.
        // Offset the wall clock time from when the PMU cycle counter was reset, to better simulate the HW.
        uint32_t timestamp = static_cast<uint32_t>(
            static_cast<uint64_t>((std::chrono::high_resolution_clock::now() - m_PmuCyclesStartTime).count()));
        return timestamp;
    }

    uint32_t value;
    berror res = bennto_read_config_reg(m_BenntoHandle, regAddress, &value);
    ASSERT_MSG(res == BERROR_OK, "Error reading from register %08" PRIx32, regAddress);
    return value;
}

void ModelHal::WaitForEvents()
{
    WaitForEventsWithTimeout(0);
}

void ModelHal::WaitForEventsWithTimeout(uint32_t timeoutMilliseconds)
{
    // First check if there are any bennto events pending.
    // If there aren't then there's no point advancing the model as it won't do anything.
    // Therefore to match the real HW we must wait until a bennto event has been scheduled.
    // In practice this won't happen when running the Firmware as it is single-threaded so once we get stuck
    // in this loop we will never get out.
    // However this behaviour is more faithful to the real HW and we have a unit test that checks that
    // we hang until the timeout (ModelHal_WaitForEvent).
    uint64_t pendingBenntoEvents = 0;
    auto startTime               = std::chrono::high_resolution_clock::now();
    while (pendingBenntoEvents == 0)
    {
        berror res = bennto_advance_model(m_BenntoHandle, 0, &pendingBenntoEvents);

        ASSERT_MSG(res == BERROR_OK, "bennto_advance_model failed");

        // In debug builds it is more useful to assert than to hang indefinitely.
        ASSERT_MSG(pendingBenntoEvents != 0, "No pending bennto events - this is most likely a hang.");

        // If the timeout has been enabled and has been reached, then return immediately
        if (timeoutMilliseconds > 0 &&
            (std::chrono::high_resolution_clock::now() - startTime) > std::chrono::milliseconds(timeoutMilliseconds))
        {
            return;
        }
    }

    // Advance the model, allowing it to process any and all bennto events (0xFFFFFFFF).
    // Note these are *not* the same as the hardware events that this method is waiting for - they are internal
    // bennto events.
    berror res = bennto_advance_model(m_BenntoHandle, 0xFFFFFFFF, nullptr);
    ASSERT_MSG(res == BERROR_OK, "bennto_advance_model failed");

    // Advancing the model will most likely yield an event, but in some cases it may not
    // (for example the PLE code that ran didn't trigger any). This is fine though, as the real HW could also be woken
    // up spuriously.
}

void ModelHal::DumpDram(const char* filename, uint64_t dramAddress, uint32_t dramSize)
{
    berror res = bennto_dump_mem_file(m_BenntoHandle, filename, dramAddress, dramSize);
    ASSERT_MSG(res == BERROR_OK, "bennto_dump_mem_file failed");
}

void ModelHal::DumpSram(const char* prefix)
{
    dl1_dfc_features_r dfc  = ReadReg(TOP_REG(DL1_RP, DL1_DFC_FEATURES));
    dl1_unit_count_r uCount = ReadReg(TOP_REG(DL1_RP, DL1_UNIT_COUNT));
    const uint32_t sramSize = dfc.get_dfc_mem_size_per_emc();
    const uint32_t dfcPerCe = uCount.get_dfc_emc_per_engine();
    const uint32_t numCes   = NumCes();

    berror res;

    // Dump the whole CE_SRAM for each CE
    for (uint32_t ceId = 0; ceId < numCes; ++ceId)
    {
        std::ostringstream ss;
        ss << prefix << "_";
        ss.fill('0');
        ss.width(2);
        ss << ceId;
        for (uint32_t dfcId = 0; dfcId < dfcPerCe; ++dfcId)
        {
            std::ostringstream ss_emc;
            ss_emc << ss.str() << "_DFC" << dfcId << ".hex";
            va_t sramAddress;
            res = bennto_calc_sram_address(m_BenntoHandle, 0, ceId, g_CeSramLookup[0], &sramAddress);
            ASSERT_MSG(res == BERROR_OK, "Unable to calculate SRAM address");
            res = bennto_dump_sram_file(m_BenntoHandle, ceId, g_CeSramLookup[0], ss_emc.str().c_str(), sramAddress,
                                        sramSize);
            ASSERT_MSG(res == BERROR_OK, "bennto_dump_sram_file failed");
        }
    }
}

bhandle_t ModelHal::GetBenntoHandle() const
{
    return m_BenntoHandle;
}

void ModelHal::DisableDebug()
{
    ConfigureDebug(nullptr, 0, 0, 0, 0, false);
}

void ModelHal::EnableDebug()
{
    ConfigureDebug("bennto.log", BDEBUG_ALL, 1, 0, BDEBUG_VERB_HIGH, true);
}

void ModelHal::ConfigureDebug(const char* debugLogFilename,
                              uint64_t debugMask,
                              uint64_t debugInstMask,
                              uint32_t suppressArchErrorMask,
                              uint64_t debugVerbosity,
                              bool dumpPle)
{
    berror res;

    if (debugLogFilename)
    {
        m_Logger.Info("Bennto debug messages being logged to '%s'.", debugLogFilename);
        res = bennto_set_debug_file(m_BenntoHandle, debugLogFilename);
        ASSERT_MSG(res == BERROR_OK, "bennto_set_debug failed");
    }

    res = bennto_set_debug(m_BenntoHandle, debugMask, debugVerbosity);
    ASSERT_MSG(res == BERROR_OK, "bennto_set_debug failed");

    res = bennto_set_debug_instance_mask(m_BenntoHandle, debugInstMask);
    ASSERT_MSG(res == BERROR_OK, "bennto_set_debug failed");

    res = bennto_suppress_arch_error(m_BenntoHandle, suppressArchErrorMask);
    ASSERT_MSG(res == BERROR_OK, "bennto_suppress_arch_error failed");

    // Note that we don't assert the result of these two config settings, because they will fail in
    // the case that we are modifying debug options after the model has been initialized.
    bennto_set_config(m_BenntoHandle, "ple.dump_mcu_trace", dumpPle);
    bennto_set_config(m_BenntoHandle, "ple.dump_ple_uscript", dumpPle);
}

}    // namespace control_unit
}    // namespace ethosn
