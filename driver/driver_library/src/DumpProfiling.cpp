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

    auto DumpEntry = [& o = outStream](const ProfilingEntry& entry) {
        o << "\t{\n";
        o << "\t\t"
          << R"("timestamp": )" << std::to_string(entry.m_Timestamp.time_since_epoch().count()) << ",\n";
        o << "\t\t"
          << R"("type": ")" << EntryTypeToCString(entry.m_Type) << "\",\n";
        // If this is a counter sample entry, then the ID is a counter name, otherwise just a number
        if (entry.m_Type == ProfilingEntry::Type::CounterSample)
        {
            // The counter could be either a collated or a polled counter
            const char* counterName = entry.m_Id < static_cast<uint64_t>(CollatedCounterName::NumValues)
                                          ? CollatedCounterNameToCString(static_cast<CollatedCounterName>(entry.m_Id))
                                          : PollCounterNameToCString(static_cast<PollCounterName>(entry.m_Id));

            o << "\t\t"
              << R"("counter_name": ")" << counterName << "\",\n"
              << "\t\t"
              << R"("counter_value": )" << entry.GetCounterValue() << "\n";
        }
        else
        {
            o << "\t\t"
              << R"("id": )" << std::to_string(entry.m_Id) << ",\n";
            o << "\t\t"
              << R"("metadata_category": ")" << MetadataCategoryToCString(entry.m_MetadataCategory) << "\",\n";
            o << "\t\t"
              << R"("metadata": {)";

            switch (entry.m_MetadataCategory)
            {
                case ProfilingEntry::MetadataCategory::FirmwareLabel:
                {
                    o << "\n\t\t\t"
                      << R"("label": ")" << entry.GetFirmwareLabel() << "\"\n"
                      << "\t\t}\n";
                    break;
                }
                default:
                {
                    // Some Metadata categories don't have any metadata
                    o << "}\n";
                }
            }
        }

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
