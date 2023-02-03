//
// Copyright © 2018-2020,2022-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "DumpProfiling.hpp"

#include "ProfilingInternal.hpp"
#include "Utils.hpp"

#include <ethosn_utils/Enums.hpp>
#define ETHOSN_ASSERT_MSG(cond, msg) assert(cond)
#include <ethosn_utils/NumericCast.hpp>

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

    // Counter to keep track of the command indexes in profiling entries from the firmware. See fixCmdIdx().
    uint32_t highestCmdIdx = 0;

    auto DumpEntry = [& o = outStream, &highestCmdIdx](const ProfilingEntry& entry) {
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
          << R"("metadata_value": )" << std::to_string(entry.m_MetadataValue) << ",\n";
        o << "\t\t"
          << R"("metadata":)"
          << "\n";
        o << "\t\t{\n";

        // Because the firmware has a limited number of bits to store counters such as the command index,
        // command streams with a lot of commands can overflow this counter and so we attempt to detect
        // this and recover the original command index when processing the entries from the firmware.
        // We make the assumption that the cmd idxs that we observe generally increase over time, and so if
        // the value suddenly jumps backwards, it's likely wrapped around. We also account for some "jitter"
        // in the observed values, i.e. it won't monotonically increase.
        auto fixCmdIdx = [&](auto x) {
            constexpr uint32_t N = 1024;    // Value at which the cmd idx will wrap around
            const uint32_t diff  = (x - highestCmdIdx) % N;
            if (diff < N / 2)
            {
                // Assume the new cmd idx is ahead of the largest seen one
                x             = highestCmdIdx + diff;
                highestCmdIdx = x;
            }
            else
            {
                // Assume the new cmd idx is behind the largest seen one
                x = highestCmdIdx - (N - diff);
            }
            return x;
        };

        DataUnion kernelEntry = {};
        kernelEntry.m_Raw     = utils::NumericCast<EntryData>(entry.m_MetadataValue);

        switch (entry.m_MetadataCategory)
        {
            case ProfilingEntry::MetadataCategory::FirmwareWfe:
            {
                o << "\t\t\t"
                  << R"("category": "FirmwareWfe",)"
                  << "\n";
                o << "\t\t\t"
                  << R"("type": )" << std::to_string(kernelEntry.m_WfeFields.m_Type) << "\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::FirmwareInference:
            {
                if (entry.m_Type == ProfilingEntry::Type::TimelineEventStart)
                {
                    // Start of inference => reset the command counter as the next command we expect to see will be the zeroth.
                    highestCmdIdx = 0;
                }

                o << "\t\t\t"
                  << R"("category": "FirmwareInference")"
                  << "\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::FirmwareCommand:
            {
                o << "\t\t\t"
                  << R"("category": "FirmwareCommand",)"
                  << "\n";
                o << "\t\t\t"
                  << R"("command_idx": )" << std::to_string(fixCmdIdx(kernelEntry.m_CommandFields.m_CommandIdx))
                  << "\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::FirmwareDma:
            {
                o << "\t\t\t"
                  << R"("category": "FirmwareDma",)"
                  << "\n";
                o << "\t\t\t"
                  << R"("command_idx": )" << std::to_string(fixCmdIdx(kernelEntry.m_DmaFields.m_CommandIdx)) << ",\n";
                o << "\t\t\t"
                  << R"("dma_category": )" << std::to_string(kernelEntry.m_DmaFields.m_DmaCategory) << ",\n";
                o << "\t\t\t"
                  << R"("dma_hardware_id": )" << std::to_string(kernelEntry.m_DmaFields.m_DmaHardwareId) << ",\n";
                o << "\t\t\t"
                  << R"("stripe_idx": )" << std::to_string(kernelEntry.m_DmaFields.m_StripeIdx) << "\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::FirmwareTsu:
            {
                o << "\t\t\t"
                  << R"("category": "FirmwareTsu",)"
                  << "\n";
                o << "\t\t\t"
                  << R"("command_idx": )" << std::to_string(fixCmdIdx(kernelEntry.m_TsuFields.m_CommandIdx)) << ",\n";
                o << "\t\t\t"
                  << R"("stripe_idx": )" << std::to_string(kernelEntry.m_TsuFields.m_StripeIdx) << ",\n";
                o << "\t\t\t"
                  << R"("bank_id": )" << std::to_string(kernelEntry.m_TsuFields.m_BankId) << "\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::FirmwareMceStripeSetup:
            {
                o << "\t\t\t"
                  << R"("category": "FirmwareMceStripeSetup",)"
                  << "\n";
                o << "\t\t\t"
                  << R"("command_idx": )" << std::to_string(fixCmdIdx(kernelEntry.m_MceStripeSetupFields.m_CommandIdx))
                  << ",\n";
                o << "\t\t\t"
                  << R"("stripe_idx": )" << std::to_string(kernelEntry.m_MceStripeSetupFields.m_StripeIdx) << "\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::FirmwarePleStripeSetup:
            {
                o << "\t\t\t"
                  << R"("category": "FirmwarePleStripeSetup",)"
                  << "\n";
                o << "\t\t\t"
                  << R"("command_idx": )" << std::to_string(fixCmdIdx(kernelEntry.m_PleStripeSetupFields.m_CommandIdx))
                  << ",\n";
                o << "\t\t\t"
                  << R"("stripe_idx": )" << std::to_string(kernelEntry.m_PleStripeSetupFields.m_StripeIdx) << "\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::FirmwareLabel:
            {
                o << "\t\t\t"
                  << R"("category": "FirmwareLabel",)"
                  << "\n";
                const size_t size     = sizeof(kernelEntry.m_LabelFields.m_Chars);
                char buffer[size + 1] = { 0 };
                strncat(buffer, reinterpret_cast<const char*>(kernelEntry.m_LabelFields.m_Chars), size);
                o << "\t\t\t"
                  << R"("chars": ")" << buffer << "\"\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::FirmwareDmaSetup:
            {
                o << "\t\t\t"
                  << R"("category": "FirmwareDmaSetup",)"
                  << "\n";
                o << "\t\t\t"
                  << R"("command_idx": )" << std::to_string(fixCmdIdx(kernelEntry.m_DmaStripeSetupFields.m_CommandIdx))
                  << ",\n";
                o << "\t\t\t"
                  << R"("stripe_idx": )" << std::to_string(kernelEntry.m_DmaStripeSetupFields.m_StripeIdx) << ",\n";
                o << "\t\t\t"
                  << R"("dma_category": )" << std::to_string(kernelEntry.m_DmaStripeSetupFields.m_DmaCategory) << "\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::FirmwareGetCompleteCommand:
            {
                o << "\t\t\t"
                  << R"("category": "FirmwareGetCompleteCommand",)"
                  << "\n";
                o << "\t\t\t"
                  << R"("command_idx": )"
                  << std::to_string(fixCmdIdx(kernelEntry.m_CompleteCommandsFields.m_CommandIdx)) << "\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::FirmwareScheduleNextCommand:
            {
                o << "\t\t\t"
                  << R"("category": "FirmwareScheduleNextCommand",)"
                  << "\n";
                o << "\t\t\t"
                  << R"("command_idx": )"
                  << std::to_string(fixCmdIdx(kernelEntry.m_ScheduleCommandsFields.m_CommandIdx)) << "\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::FirmwareAgent:
            {
                o << "\t\t\t"
                  << R"("category": "FirmwareAgent",)"
                  << "\n";
                o << "\t\t\t"
                  << R"("type": )" << std::to_string(kernelEntry.m_AgentFields.m_Type) << ",\n";
                o << "\t\t\t"
                  << R"("idx": )" << std::to_string(kernelEntry.m_AgentFields.m_Idx) << ",\n";
                o << "\t\t\t"
                  << R"("command_idx": )" << std::to_string(fixCmdIdx(kernelEntry.m_AgentFields.m_CommandIdx)) << "\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::FirmwareAgentStripe:
            {
                o << "\t\t\t"
                  << R"("category": "FirmwareAgentStripe",)"
                  << "\n";
                o << "\t\t\t"
                  << R"("agent_stripe_type": )" << std::to_string(kernelEntry.m_AgentStripeFields.m_AgentStripeType)
                  << ",\n";
                o << "\t\t\t"
                  << R"("agent_stripe_idx": )" << std::to_string(kernelEntry.m_AgentStripeFields.m_AgentStripeIdx)
                  << ",\n";
                o << "\t\t\t"
                  << R"("command_idx": )" << std::to_string(fixCmdIdx(kernelEntry.m_AgentStripeFields.m_CommandIdx))
                  << ",\n";
                o << "\t\t\t"
                  << R"("stripe_idx": )" << std::to_string(kernelEntry.m_AgentStripeFields.m_StripeIdx) << "\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::InferenceLifetime:
            {
                o << "\t\t\t"
                  << R"("category": "InferenceLifetime")"
                  << "\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::BufferLifetime:
            {
                o << "\t\t\t"
                  << R"("category": "BufferLifetime")"
                  << "\n";
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
                  << R"("category": "FirmwarePle",)"
                  << "\n";
                o << "\t\t\t"
                  << R"("command_idx": )" << std::to_string(fixCmdIdx(kernelEntry.m_PleFields.m_CommandIdx)) << ",\n";
                o << "\t\t\t"
                  << R"("stripe_idx": )" << std::to_string(kernelEntry.m_PleFields.m_StripeIdx) << "\n";
                break;
            }
            case ProfilingEntry::MetadataCategory::FirmwareUdma:
            {
                o << "\t\t\t"
                  << R"("category": "FirmwareUdma",)"
                  << "\n";
                o << "\t\t\t"
                  << R"("command_idx": )" << std::to_string(fixCmdIdx(kernelEntry.m_UdmaFields.m_CommandIdx)) << ",\n";
                o << "\t\t\t"
                  << R"("stripe_idx": )" << std::to_string(kernelEntry.m_UdmaFields.m_StripeIdx) << "\n";
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
