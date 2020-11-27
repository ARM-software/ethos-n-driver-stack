//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "PerformanceData.hpp"

#include <ethosn_utils/Json.hpp>

using namespace ethosn::utils;

namespace ethosn
{
namespace support_library
{

namespace
{

std::ostream& Print(std::ostream& os, Indent indent, const MemoryStats& stats)
{
    os << indent << JsonField("DramParallelBytes") << ' ' << stats.m_DramParallel << ",\n";
    os << indent << JsonField("DramNonParallelBytes") << ' ' << stats.m_DramNonParallel << ",\n";
    os << indent << JsonField("SramBytes") << ' ' << stats.m_Sram;
    return os;
}

std::ostream& Print(std::ostream& os, Indent indent, const StripesStats& stats)
{
    os << indent << JsonField("NumCentralStripes") << ' ' << stats.m_NumCentralStripes << ",\n";
    os << indent << JsonField("NumBoundaryStripes") << ' ' << stats.m_NumBoundaryStripes << ",\n";
    os << indent << JsonField("NumReloads") << ' ' << stats.m_NumReloads;
    return os;
}

std::ostream& Print(std::ostream& os, Indent indent, const InputStats& stats)
{
    os << indent << "{\n";

    ++indent;

    Print(os, indent, stats.m_MemoryStats);
    os << ",\n";
    Print(os, indent, stats.m_StripesStats);
    os << "\n";

    --indent;

    os << indent << "}";

    return os;
}

std::ostream& Print(std::ostream& os, Indent indent, const WeightsStats& stats)
{
    os << indent << "{\n";

    ++indent;

    Print(os, indent, stats.m_MemoryStats);
    os << ",\n";
    Print(os, indent, stats.m_StripesStats);
    os << ",\n";
    os << indent << JsonField("CompressionSavings") << ' ' << stats.m_WeightCompressionSavings << "\n";

    --indent;

    os << indent << "}";

    return os;
}

std::ostream& Print(std::ostream& os, Indent indent, const MceStats& mceStats)
{
    os << indent << "{\n";

    ++indent;

    os << indent << JsonField("Operations") << ' ' << mceStats.m_Operations << ",\n";
    os << indent << JsonField("CycleCount") << ' ' << mceStats.m_CycleCount << "\n";

    --indent;

    os << indent << "}";

    return os;
}

std::ostream& Print(std::ostream& os, Indent indent, const PleStats& pleStats)
{
    os << indent << "{\n";

    ++indent;

    os << indent << JsonField("NumOfPatches") << ' ' << pleStats.m_NumOfPatches << ",\n";
    os << indent << JsonField("Operation") << ' ' << pleStats.m_Operation << "\n";

    --indent;

    os << indent << "}";

    return os;
}

}    // namespace

std::ostream& PrintPassPerformanceData(std::ostream& os, Indent indent, const PassPerformanceData& pass)
{
    os << indent << "{\n";

    ++indent;

    os << indent << JsonField("OperationIds") << ' ';
    Print(os, Indent(0), JsonArray(pass.m_OperationIds)) << ",\n";

    os << indent << JsonField("ParentIds") << ' ' << (pass.m_ParentIds.empty() ? "[]" : pass.m_ParentIds) << ",\n";

    os << indent << JsonField("Input") << '\n';
    Print(os, indent, pass.m_Stats.m_Input) << ",\n";

    os << indent << JsonField("Output") << '\n';
    Print(os, indent, pass.m_Stats.m_Output) << ",\n";

    os << indent << JsonField("Weights") << '\n';
    Print(os, indent, pass.m_Stats.m_Weights) << ",\n";

    os << indent << JsonField("Mce") << '\n';
    Print(os, indent, pass.m_Stats.m_Mce) << ",\n";

    os << indent << JsonField("Ple") << '\n';
    Print(os, indent, pass.m_Stats.m_Ple) << "\n";

    --indent;

    os << indent << "}";

    return os;
}

std::ostream&
    PrintFailureReasons(std::ostream& os, Indent indent, const std::map<uint32_t, std::string>& failureReasons)
{
    os << indent << "{\n";

    ++indent;

    for (auto it = failureReasons.begin(); it != failureReasons.end(); ++it)
    {
        os << indent << JsonField(it->first) << ' ' << Quoted(it->second);

        if (it != std::prev(failureReasons.end()))
        {
            os << ",\n";
        }
        else
        {
            os << "\n";
        }
    }

    --indent;

    os << indent << "}";

    return os;
}

}    // namespace support_library
}    // namespace ethosn
