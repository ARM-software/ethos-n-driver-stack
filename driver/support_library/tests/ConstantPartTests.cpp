//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "CapabilitiesInternal.hpp"
#include "GlobalParameters.hpp"
#include "TestUtils.hpp"
#include "Utils.hpp"
#include "cascading/ConstantPart.hpp"
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
    TensorShape m_OutputShape;
    QuantizationInfo m_OutputQuantInfo;
    std::set<uint32_t> m_OperationIds;
};

void CheckInputDram(Buffer* inputBuffer, const CheckPlansParams& params)
{
    // Check properties of Input DRAM buffer
    if (inputBuffer)
    {
        CHECK(inputBuffer->m_Location == Location::Dram);
        CHECK(inputBuffer->m_Format == CascadingBufferFormat::NHWCB);
        CHECK(inputBuffer->m_QuantizationInfo == params.m_OutputQuantInfo);
        CHECK(inputBuffer->m_TensorShape == params.m_OutputShape);
        CHECK(inputBuffer->m_SizeInBytes == utils::TotalSizeBytesNHWCB(inputBuffer->m_TensorShape));
        CHECK(inputBuffer->Dram()->m_EncodedWeights == nullptr);
    }
}

void CheckMappings(const CheckPlansParams& params, const Plan& plan, Buffer* inputBuffer)
{
    // Check input/output mappings
    CHECK(plan.m_InputMappings.size() == 0);
    CHECK(plan.m_OutputMappings.size() == 1);

    if (inputBuffer)
    {
        CHECK(plan.m_OutputMappings.begin()->first == inputBuffer);
    }

    CHECK(plan.m_OutputMappings.begin()->second.m_PartId == params.m_PartId);
    CHECK(plan.m_OutputMappings.begin()->second.m_OutputIndex == 0);
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
        Buffer* inputBuffer                = buffers.front();

        CheckInputDram(inputBuffer, params);
        CheckMappings(params, plan, inputBuffer);
    }
}

void SavePlansToDot(const Plans& plans, const std::string test)
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

TEST_CASE("ConstantPart Plan Generation", "[ConstantPartTests]")
{
    GIVEN("A simple ConstantPart")
    {
        const PartId partId                   = 1;
        TensorShape outputTensorShape         = { 1, 32, 32, 3 };
        CompilerDataFormat compilerDataFormat = CompilerDataFormat::NHWCB;
        QuantizationInfo quantizationInfo(0, 1.0f);
        const std::set<uint32_t> operationIds = {};
        const EstimationOptions estOpt;
        const CompilationOptions compOpt;
        HardwareCapabilities hwCapabilities(GetEthosN78FwHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

        ConstantPart constantPart(partId, outputTensorShape, compilerDataFormat, quantizationInfo,
                                  DataType::UINT8_QUANTIZED, operationIds, estOpt, compOpt, hwCapabilities);

        CheckPlansParams params;
        params.m_PartId          = partId;
        params.m_OutputShape     = outputTensorShape;
        params.m_OutputQuantInfo = quantizationInfo;
        params.m_OperationIds    = operationIds;

        WHEN("Asked to generate Lonely plans")
        {
            Plans plans = constantPart.GetPlans(CascadeType::Lonely, command_stream::BlockConfig{}, nullptr, 0);
            SavePlansToDot(plans, "ConstantPart GetPlans structure Lonely");

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
            Plans plans = constantPart.GetPlans(CascadeType::Beginning, command_stream::BlockConfig{}, nullptr, 0);
            SavePlansToDot(plans, "ConstantPart GetPlans structure Beginning");

            THEN("The number of generated plans = 1")
            {
                CHECK(plans.size() == 1);
            }

            AND_THEN("The plan is valid and end in Dram")
            {
                CheckPlans(plans, params);
            }
        }

        WHEN("Asked to generate Middle plans")
        {
            Plans plans = constantPart.GetPlans(CascadeType::Middle, command_stream::BlockConfig{}, nullptr, 0);
            SavePlansToDot(plans, "ConstantPart GetPlans structure Middle");

            THEN("The number of generated plans = 0")
            {
                CHECK(plans.size() == 0);
            }
        }

        WHEN("Asked to generate End plans")
        {
            Plans plans = constantPart.GetPlans(CascadeType::End, command_stream::BlockConfig{}, nullptr, 0);
            SavePlansToDot(plans, "ConstantPart GetPlans structure End");

            THEN("The number of generated plans = 0")
            {
                CHECK(plans.size() == 0);
            }
        }
    }
}
