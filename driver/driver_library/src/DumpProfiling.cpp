//
// Copyright © 2018-2020,2022 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "DumpProfiling.hpp"

#include "ProfilingInternal.hpp"
#include "Utils.hpp"

#include <ethosn_utils/Enums.hpp>

#include <ostream>
#include <string>
#include <vector>

namespace ethosn
{
namespace driver_library
{
namespace profiling
{

void DumpAllProfilingData(std::ostream& outStream)
{
    std::vector<ProfilingEntry> entries = profiling::g_ProfilingEntries;
    // As well as dumping the currently queued profiling events, include a sample of every pollable counter.
    for (PollCounterName counter = static_cast<PollCounterName>(PollCounterName::DriverLibraryNumLiveBuffers);
         counter < PollCounterName::NumValues; counter = ethosn::utils::NextEnumValue(counter))
    {
        ProfilingEntry entry;
        entry.m_Timestamp        = std::chrono::high_resolution_clock::now();
        entry.m_Type             = ProfilingEntry::Type::CounterSample;
        entry.m_Id               = static_cast<uint64_t>(counter);
        entry.m_MetadataCategory = ProfilingEntry::MetadataCategory::CounterValue;
        entry.m_MetadataValue    = metadata::CreateCounterValue(GetCounterValue(counter));

        entries.push_back(entry);
    }
    DumpProfilingData(entries, outStream);
}

void DumpProfilingData(const std::vector<ProfilingEntry>& profilingData, std::ostream& outStream)
{
    if (!outStream.good())
    {
        return;
    }
    outStream << "[\n";
    auto DumpEntry = [& o = outStream](const ProfilingEntry& entry) {
        o << "\t{\n";
        o << "\t\t"
          << R"("time_stamp": )" << std::to_string(entry.m_Timestamp.time_since_epoch().count()) << ",\n";
        o << "\t\t"
          << R"("type": )" << std::to_string(static_cast<uint64_t>(entry.m_Type)) << ",\n";
        o << "\t\t"
          << R"("id": )" << std::to_string(entry.m_Id) << ",\n";
        o << "\t\t"
          << R"("metadata_category": )" << std::to_string(static_cast<uint64_t>(entry.m_MetadataCategory)) << ",\n";
        o << "\t\t"
          << R"("metadata_value":)"
          << "\n";
        o << "\t\t{\n";
        switch (entry.m_MetadataCategory)
        {
            case ProfilingEntry::MetadataCategory::FirmwareWfe:
            {
                o << "\t\t\t"
                  << R"("firmware_wfe_value": )" << std::to_string(entry.m_MetadataValue) << "\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::FirmwareInference:
            {
                o << "\t\t\t"
                  << R"("firmware_inference_value": )" << std::to_string(entry.m_MetadataValue) << "\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::FirmwareCommand:
            {
                o << "\t\t\t"
                  << R"("firmware_command_value": )" << std::to_string(entry.m_MetadataValue) << "\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::FirmwareDma:
            {
                o << "\t\t\t"
                  << R"("firmware_dma_value": )" << std::to_string(entry.m_MetadataValue) << "\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::FirmwareTsu:
            {
                o << "\t\t\t"
                  << R"("firmware_tsu_value": )" << std::to_string(entry.m_MetadataValue) << "\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::FirmwareMceStripeSetup:
            {
                o << "\t\t\t"
                  << R"("firmware_mce_stripe_setup_value": )" << std::to_string(entry.m_MetadataValue) << "\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::FirmwarePleStripeSetup:
            {
                o << "\t\t\t"
                  << R"("firmware_ple_stripe_setup_value": )" << std::to_string(entry.m_MetadataValue) << "\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::FirmwareLabel:
            {
                o << "\t\t\t"
                  << R"("firmware_label_value": )" << std::to_string(entry.m_MetadataValue) << "\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::FirmwareDmaSetup:
            {
                o << "\t\t\t"
                  << R"("firmware_dma_setup_value": )" << std::to_string(entry.m_MetadataValue) << "\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::FirmwareGetCompleteCommand:
            {
                o << "\t\t\t"
                  << R"("firmware_get_complete_command_value": )" << std::to_string(entry.m_MetadataValue) << "\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::FirmwareScheduleNextCommand:
            {
                o << "\t\t\t"
                  << R"("firmware_schedule_next_command_value": )" << std::to_string(entry.m_MetadataValue) << "\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::FirmwareTimeSync:
            {
                o << "\t\t\t"
                  << R"("firmware_time_sync_value": )" << std::to_string(entry.m_MetadataValue) << "\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::FirmwareAgent:
            {
                o << "\t\t\t"
                  << R"("firmware_agent_value": )" << std::to_string(entry.m_MetadataValue) << "\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::FirmwareAgentStripe:
            {
                o << "\t\t\t"
                  << R"("firmware_agent_stripe_value": )" << std::to_string(entry.m_MetadataValue) << "\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::InferenceLifetime:
            {
                o << "\t\t\t"
                  << R"("inference_value": )" << std::to_string(entry.m_MetadataValue) << "\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::BufferLifetime:
            {
                o << "\t\t\t"
                  << R"("buffer_value": )" << std::to_string(entry.m_MetadataValue) << "\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::CounterValue:
            {
                o << "\t\t\t"
                  << R"("counter_value": )" << std::to_string(entry.GetCounterValue()) << "\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::FirmwarePle:
            {
                o << "\t\t\t"
                  << R"("firmware_ple_value": )" << std::to_string(entry.m_MetadataValue) << "\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::FirmwareUdma:
            {
                o << "\t\t\t"
                  << R"("firmware_udma_value": )" << std::to_string(entry.m_MetadataValue) << "\n";
                break;
            }
            default:
            {
                // Some Metadata categories don't have metadata
            }
        }
        o << "\t\t}\n";
        o << "\t}";
    };
    for (size_t i = 0; i < profilingData.size(); ++i)
    {
        const ProfilingEntry& entry = profilingData[i];
        DumpEntry(entry);
        if (i != profilingData.size() - 1)
        {
            outStream << ",\n";
        }
    }
    outStream << "\n";
    outStream << "]\n";
}

}    // namespace profiling
}    // namespace driver_library
}    // namespace ethosn
