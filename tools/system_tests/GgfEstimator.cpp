//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "EthosNParseRunner.hpp"
#include "GlobalParameters.hpp"
#include "SystemTestsUtils.hpp"

#include <ethosn_utils/Json.hpp>

#include <catch.hpp>

using namespace ethosn::support_library;
using namespace ethosn::utils;

namespace ethosn
{
namespace system_tests
{

TEST_CASE("GgfEstimator", "[!hide]")
{
    CAPTURE(g_GgfFilePath);

    std::ifstream ggfFile(g_GgfFilePath);

    if (ggfFile.fail())
    {
        const std::string error = std::string("Failed to open ggf file: ") + g_GgfFilePath + "\n";
        throw std::invalid_argument(error);
    }

    g_Logger.Debug("Estimating performance on Ethos-N...");

    LayerData layerData;

    EthosNParseRunner::CreationOptions creationOptions =
        EthosNParseRunner::CreationOptions::CreateWithGlobalOptions(ggfFile, layerData);
    creationOptions.m_EstimationMode = true;
    EthosNParseRunner ethosnParseRunner(creationOptions);

    ethosnParseRunner.SetStrategies(g_Strategies);
    ethosnParseRunner.SetBlockConfigs(g_BlockConfigs);

    const NetworkPerformanceData perfData   = ethosnParseRunner.EstimateNetwork();
    const EstimationOptions& estimationOpts = ethosnParseRunner.GetEstimationOptions();

    if (perfData.m_Stream.size() == 0)
    {
        throw std::runtime_error("Estimation failed");
    }

    constexpr char outPerfFile[] = "ethosn_perf.json";

    std::ofstream os(outPerfFile);

    Indent indent(0);
    os << indent << "{\n";
    ++indent;

    os << indent << JsonField("Config") << "\n";
    os << indent << "{\n";
    indent++;

    os << indent << JsonField("Variant") << " \"N/A\",\n";
    os << indent << JsonField("ActivationCompressionSavings") << ' ' << estimationOpts.m_ActivationCompressionSaving
       << ",\n";
    if (estimationOpts.m_UseWeightCompressionOverride)
    {
        os << indent << JsonField("WeightCompressionSavings") << ' ' << estimationOpts.m_WeightCompressionSaving
           << ",\n";
    }
    else
    {
        os << indent << JsonField("WeightCompressionSavings") << ' ' << Quoted("Not Specified") << ",\n";
    }
    os << indent << JsonField("Current") << ' ' << estimationOpts.m_Current << "\n";

    indent--;
    os << indent << "},\n";

    os << indent << JsonField("OperationNames") << " {},\n";

    os << indent << JsonField("Results") << '\n';
    PrintNetworkPerformanceDataJson(os, static_cast<uint32_t>(indent.m_Depth), perfData);

    --indent;
    os << indent << "}\n";

    g_Logger.Debug("Performance estimation results written to: %s", outPerfFile);
}

}    // namespace system_tests
}    // namespace ethosn
