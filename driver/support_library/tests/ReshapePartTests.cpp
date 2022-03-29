//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "CapabilitiesInternal.hpp"
#include "GlobalParameters.hpp"
#include "TestUtils.hpp"
#include "Utils.hpp"
#include "cascading/Part.hpp"
#include "cascading/PartUtils.hpp"
#include "cascading/Plan.hpp"
#include "cascading/ReshapePart.hpp"
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
    TensorShape m_OutputShape;
    QuantizationInfo m_InputQuantInfo;
    std::set<uint32_t> m_OperationIds;
};

void CheckReshapeDram(Buffer* reshapedBuffer, const CheckPlansParams& params)
{
    // Check properties of reshaped DRAM buffer
    if (reshapedBuffer)
    {
        CHECK(reshapedBuffer->m_Location == Location::Dram);
        CHECK(reshapedBuffer->m_Format == CascadingBufferFormat::NHWC);
        CHECK(reshapedBuffer->m_QuantizationInfo == params.m_InputQuantInfo);
        CHECK(reshapedBuffer->m_TensorShape == params.m_OutputShape);
        CHECK(reshapedBuffer->m_StripeShape == TensorShape{ 0, 0, 0, 0 });
        CHECK(reshapedBuffer->m_Order == TraversalOrder::Xyz);
        CHECK(reshapedBuffer->m_SizeInBytes == utils::TotalSizeBytes(reshapedBuffer->m_TensorShape));
        CHECK(reshapedBuffer->m_NumStripes == 0);
        CHECK(reshapedBuffer->m_EncodedWeights == nullptr);
    }
}

void CheckMappings(const CheckPlansParams& params, const Plan& plan, Buffer* reshapedBuffer)
{
    // Check input/output mappings
    CHECK(plan.m_InputMappings.size() == 1);
    CHECK(plan.m_OutputMappings.size() == 1);

    if (reshapedBuffer)
    {
        CHECK(plan.m_InputMappings.begin()->first == reshapedBuffer);
        CHECK(plan.m_OutputMappings.begin()->first == reshapedBuffer);
    }

    CHECK(plan.m_InputMappings.begin()->second.m_PartId == params.m_PartId);
    CHECK(plan.m_OutputMappings.begin()->second.m_PartId == params.m_PartId);
    CHECK(plan.m_InputMappings.begin()->second.m_InputIndex == 0);
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
        Buffer* reshapedBuffer             = buffers.front();

        CheckReshapeDram(reshapedBuffer, params);
        CheckMappings(params, plan, reshapedBuffer);
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

TEST_CASE("ReshapePart Plan Generation", "[ReshapePartTests]")
{
    GIVEN("A simple ReshapePart")
    {
        const PartId partId                   = 1;
        TensorShape inputTensorShape          = { 1, 32, 32, 3 };
        TensorShape outputTensorShape         = { 1, 16, 16, 12 };
        CompilerDataFormat compilerDataFormat = CompilerDataFormat::NHWC;
        QuantizationInfo quantizationInfo(0, 1.0f);
        const std::set<uint32_t> operationIds = { 1 };
        const EstimationOptions estOpt;
        const CompilationOptions compOpt;
        HardwareCapabilities hwCapabilities(GetEthosN78FwHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

        ReshapePart reshapePart(partId, inputTensorShape, outputTensorShape, compilerDataFormat, quantizationInfo,
                                operationIds, estOpt, compOpt, hwCapabilities);

        CheckPlansParams params;
        params.m_PartId         = partId;
        params.m_InputShape     = inputTensorShape;
        params.m_OutputShape    = outputTensorShape;
        params.m_InputQuantInfo = quantizationInfo;
        params.m_OperationIds   = operationIds;

        WHEN("Asked to generate Lonely plans")
        {
            Plans plans = reshapePart.GetPlans(CascadeType::Lonely, command_stream::BlockConfig{}, nullptr, 0);
            SavePlansToDot(plans, "ReshapePart GetPlans structure Lonely");

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
            Plans plans = reshapePart.GetPlans(CascadeType::Beginning, command_stream::BlockConfig{}, nullptr, 0);
            SavePlansToDot(plans, "ReshapePart GetPlans structure Beginning");

            THEN("The number of generated plans = 0")
            {
                CHECK(plans.size() == 0);
            }
        }

        WHEN("Asked to generate Middle plans")
        {
            Plans plans = reshapePart.GetPlans(CascadeType::Middle, command_stream::BlockConfig{}, nullptr, 0);
            SavePlansToDot(plans, "ReshapePart GetPlans structure Middle");

            THEN("The number of generated plans = 0")
            {
                CHECK(plans.size() == 0);
            }
        }

        WHEN("Asked to generate End plans")
        {
            Plans plans = reshapePart.GetPlans(CascadeType::End, command_stream::BlockConfig{}, nullptr, 0);
            SavePlansToDot(plans, "ReshapePart GetPlans structure End");

            THEN("The number of generated plans = 0")
            {
                CHECK(plans.size() == 0);
            }
        }
    }
}
