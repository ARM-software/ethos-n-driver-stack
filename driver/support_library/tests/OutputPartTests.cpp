//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "CapabilitiesInternal.hpp"
#include "GlobalParameters.hpp"
#include "TestUtils.hpp"
#include "Utils.hpp"
#include "cascading/OutputPart.hpp"
#include "cascading/Part.hpp"
#include "cascading/Plan.hpp"
#include "cascading/Visualisation.hpp"

#include <catch.hpp>
#include <fstream>
#include <regex>

using namespace ethosn::support_library;

namespace command_stream = ethosn::command_stream;

namespace
{

struct CheckPlansParams
{
    PartId m_PartId;
    TensorShape m_InputShape;
    QuantizationInfo m_InputQuantInfo;
    std::set<uint32_t> m_OperationIds;
};

void CheckOutputDram(Buffer* outputBuffer, const CheckPlansParams& params)
{
    // Check properties of Output DRAM buffer
    if (outputBuffer)
    {
        CHECK(outputBuffer->m_Location == Location::Dram);
        CHECK(outputBuffer->m_Format == CascadingBufferFormat::NHWCB);
        CHECK(outputBuffer->m_QuantizationInfo == params.m_InputQuantInfo);
        CHECK(outputBuffer->m_TensorShape == params.m_InputShape);
        CHECK(outputBuffer->m_SizeInBytes == utils::TotalSizeBytesNHWCB(outputBuffer->m_TensorShape));
        CHECK(outputBuffer->Dram()->m_EncodedWeights == nullptr);
    }
}

void CheckMappings(const CheckPlansParams& params, const Plan& plan, Buffer* outputBuffer)
{
    // Check input/output mappings
    CHECK(plan.m_InputMappings.size() == 1);
    CHECK(plan.m_OutputMappings.size() == 0);

    if (outputBuffer)
    {
        CHECK(plan.m_InputMappings.begin()->first == outputBuffer);
    }

    CHECK(plan.m_InputMappings.begin()->second.m_PartId == params.m_PartId);
    CHECK(plan.m_InputMappings.begin()->second.m_InputIndex == 0);
}

/// Checks that the given list of Plans matches expectations, based on both generic requirements of all plans (e.g. all plans
/// must follow the expected OpGraph structure) and also specific requirements on plans which can be customized using the provided callbacks.
/// These are all configured by the CheckPlansParams struct.
void CheckPlans(const Plans& plans, const CheckPlansParams& params)
{
    CHECK(plans.size() > 0);

    for (auto&& plan : plans)
    {
        INFO("plan " << plan.m_DebugTag);

        const OpGraph::BufferList& buffers = plan.m_OpGraph.GetBuffers();
        Buffer* outputBuffer               = buffers.front();

        CheckOutputDram(outputBuffer, params);
        CheckMappings(params, plan, outputBuffer);
    }
}

void SavePlansToDot(const Plans& plans, const std::string& test)
{
    if (!g_AllowDotFileGenerationInTests)
    {
        return;
    }

    std::stringstream str;
    std::stringstream stripes;
    for (const auto& plan : plans)
    {
        SaveOpGraphToDot(plan.m_OpGraph, str, DetailLevel::High);

        SaveOpGraphToTxtFile(plan.m_OpGraph, stripes);
    }

    std::regex re("digraph");
    std::string s = std::regex_replace(str.str(), re, "subgraph");

    std::ofstream file(test + ".dot");
    std::ofstream stripesFile(test + "_stripes.txt");
    file << "digraph {" << std::endl << s << "}" << std::endl;
    stripesFile << stripes.str() << std::endl;
}
}    // namespace

TEST_CASE("OutputPart Plan Generation", "[OutputPartTests]")
{
    GIVEN("A simple OutputPart")
    {
        const PartId partId                   = 1;
        TensorShape inputTensorShape          = { 1, 32, 32, 3 };
        CompilerDataFormat compilerDataFormat = CompilerDataFormat::NHWCB;
        QuantizationInfo quantizationInfo(0, 1.0f);
        const std::set<uint32_t> operationIds = { 1 };
        const EstimationOptions estOpt;
        const CompilationOptions compOpt;
        HardwareCapabilities hwCapabilities(GetEthosN78FwHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

        OutputPart outputPart(partId, inputTensorShape, compilerDataFormat, quantizationInfo, DataType::UINT8_QUANTIZED,
                              operationIds, 0, estOpt, compOpt, hwCapabilities);

        CheckPlansParams params;
        params.m_PartId         = partId;
        params.m_InputShape     = inputTensorShape;
        params.m_InputQuantInfo = quantizationInfo;
        params.m_OperationIds   = operationIds;

        WHEN("Asked to generate Lonely plans")
        {
            Plans plans = outputPart.GetPlans(CascadeType::Lonely, command_stream::BlockConfig{}, { nullptr }, 0);
            SavePlansToDot(plans, "OutputPart GetPlans structure Lonely");

            THEN("The number of generated plans = 1")
            {
                CHECK(plans.size() == 1);
            }

            AND_THEN("The plan is valid and end in Dram")
            {
                CheckPlans(plans, params);
            }
        }

        WHEN("Asked to generate Beginning plans")
        {
            Plans plans = outputPart.GetPlans(CascadeType::Beginning, command_stream::BlockConfig{}, { nullptr }, 0);
            SavePlansToDot(plans, "OutputPart GetPlans structure Beginning");

            THEN("The number of generated plans = 0")
            {
                CHECK(plans.size() == 0);
            }
        }

        WHEN("Asked to generate Middle plans")
        {
            Plans plans = outputPart.GetPlans(CascadeType::Middle, command_stream::BlockConfig{}, { nullptr }, 0);
            SavePlansToDot(plans, "OutputPart GetPlans structure Middle");

            THEN("The number of generated plans = 0")
            {
                CHECK(plans.size() == 0);
            }
        }

        WHEN("Asked to generate End plans")
        {
            Plans plans = outputPart.GetPlans(CascadeType::End, command_stream::BlockConfig{}, { nullptr }, 0);
            SavePlansToDot(plans, "OutputPart GetPlans structure End");

            THEN("The number of generated plans = 0")
            {
                CHECK(plans.size() == 0);
            }
        }
    }
}
