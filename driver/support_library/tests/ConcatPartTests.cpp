//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "CapabilitiesInternal.hpp"
#include "GlobalParameters.hpp"
#include "TestUtils.hpp"
#include "Utils.hpp"
#include "cascading/ConcatPart.hpp"
#include "cascading/Part.hpp"
#include "cascading/PartUtils.hpp"
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
    std::vector<TensorInfo> m_InputTensorsInfo;
    TensorInfo m_OutputTensorInfo;
    QuantizationInfo m_OutputQuantInfo;
    std::set<uint32_t> m_OperationIds;
    CascadingBufferFormat m_DataFormat;
};

void CheckConcatOperation(const Plan& plan)
{
    // Check operation, consumers, and producers
    CHECK(plan.m_OpGraph.GetOps().size() == 1);
    Op* op = plan.m_OpGraph.GetOp(0);

    const OpGraph::BufferList& buffers = plan.m_OpGraph.GetBuffers();

    for (uint32_t inputIndex = 0; inputIndex < buffers.size() - 1; inputIndex++)
    {
        CHECK(plan.m_OpGraph.GetConsumers(buffers[inputIndex]).size() == 1);
        CHECK(plan.m_OpGraph.GetConsumers(buffers[inputIndex])[0].first == op);
    }

    CHECK(plan.m_OpGraph.GetProducer(buffers.back()) != nullptr);
    CHECK(plan.m_OpGraph.GetProducer(buffers.back()) == op);
}

void CheckConcatDram(Buffer* concatBuffer, const CheckPlansParams& params)
{
    // Check properties of concat DRAM buffer
    if (concatBuffer)
    {
        CHECK(concatBuffer->m_Location == Location::Dram);
        CHECK(concatBuffer->m_Lifetime == Lifetime::Atomic);
        CHECK(concatBuffer->m_Format == params.m_DataFormat);
        CHECK(concatBuffer->m_TensorShape == params.m_OutputTensorInfo.m_Dimensions);
        CHECK(concatBuffer->m_Order == TraversalOrder::Xyz);
        CHECK(concatBuffer->m_SizeInBytes == utils::TotalSizeBytes(params.m_OutputTensorInfo.m_Dimensions));
        CHECK(concatBuffer->m_NumStripes == 0);
        CHECK(concatBuffer->m_EncodedWeights == nullptr);
    }
}

void CheckMappings(const CheckPlansParams& params, const Plan& plan, Buffer* concatBuffer)
{
    // Check input/output mappings
    const OpGraph::BufferList& buffers = plan.m_OpGraph.GetBuffers();

    CHECK(plan.m_InputMappings.size() == buffers.size() - 1);
    CHECK(plan.m_OutputMappings.size() == 1);

    for (uint32_t inputIndex = 0; inputIndex < buffers.size() - 1; inputIndex++)
    {
        CHECK(plan.m_InputMappings.at(buffers[inputIndex]).m_PartId == params.m_PartId);
        CHECK(plan.m_InputMappings.at(buffers[inputIndex]).m_InputIndex == inputIndex);
    }

    CHECK(plan.m_OutputMappings.begin()->first == concatBuffer);
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
        INFO("plan " << plan->m_DebugTag);

        const OpGraph::BufferList& buffers = plan->m_OpGraph.GetBuffers();
        Buffer* concatBuffer               = buffers.back();

        CheckConcatOperation(*plan);
        CheckConcatDram(concatBuffer, params);
        CheckMappings(params, *plan, concatBuffer);
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
        SaveOpGraphToDot(plan->m_OpGraph, str, DetailLevel::High);

        SaveOpGraphToTxtFile(plan->m_OpGraph, stripes);
    }

    std::regex re("digraph");
    std::string s = std::regex_replace(str.str(), re, "subgraph");

    std::ofstream file(test + ".dot");
    std::ofstream stripesFile(test + "_stripes.txt");
    file << "digraph {" << std::endl << s << "}" << std::endl;
    stripesFile << stripes.str() << std::endl;
}
}    // namespace

TEST_CASE("ConcatPart Plan Generation", "[ConcatPartTests]")
{
    GIVEN("A simple ConcatPart")
    {
        const PartId partId = 1;

        std::vector<TensorInfo> inputTensorsInfo;
        TensorInfo inputTensorInfo1;
        TensorInfo inputTensorInfo2;
        CompilerDataFormat compilerDataFormat;
        DataFormat dataFormat = GENERATE(DataFormat::NHWC, DataFormat::NHWCB);

        if (dataFormat == DataFormat::NHWC)
        {
            inputTensorInfo1.m_Dimensions = { 1, 16, 16, 16 };
            inputTensorInfo1.m_DataType   = DataType::INT8_QUANTIZED;
            inputTensorInfo1.m_DataFormat = DataFormat::NHWC;

            inputTensorInfo2.m_Dimensions = { 1, 16, 16, 16 };
            inputTensorInfo2.m_DataType   = DataType::INT8_QUANTIZED;
            inputTensorInfo2.m_DataFormat = DataFormat::NHWC;

            compilerDataFormat = CompilerDataFormat::NHWC;
        }
        else
        {
            inputTensorInfo1.m_Dimensions = { 1, 16, 16, 16 };
            inputTensorInfo1.m_DataType   = DataType::INT8_QUANTIZED;
            inputTensorInfo1.m_DataFormat = DataFormat::NHWCB;

            inputTensorInfo2.m_Dimensions = { 1, 16, 16, 16 };
            inputTensorInfo2.m_DataType   = DataType::INT8_QUANTIZED;
            inputTensorInfo2.m_DataFormat = DataFormat::NHWCB;

            compilerDataFormat = CompilerDataFormat::NHWCB;
        }

        inputTensorsInfo.push_back(inputTensorInfo1);
        inputTensorsInfo.push_back(inputTensorInfo2);

        QuantizationInfo quantizationInfo(0, 1.0f);
        ConcatenationInfo concatInfo(1, quantizationInfo);

        const std::set<uint32_t> operationIds = { 1 };
        const EstimationOptions estOpt;
        const CompilationOptions compOpt;
        HardwareCapabilities hwCapabilities(GetEthosN78FwHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

        ConcatPart concatPart(partId, inputTensorsInfo, concatInfo, compilerDataFormat, operationIds, estOpt, compOpt,
                              hwCapabilities);

        CheckPlansParams params;
        params.m_PartId           = partId;
        params.m_InputTensorsInfo = inputTensorsInfo;
        params.m_OutputTensorInfo = Concatenation::CalculateOutputTensorInfo(params.m_InputTensorsInfo, concatInfo);
        params.m_OperationIds     = operationIds;
        if (dataFormat == DataFormat::NHWC)
        {
            params.m_DataFormat = CascadingBufferFormat::NHWC;
        }
        else
        {
            params.m_DataFormat = CascadingBufferFormat::NHWCB;
        }

        WHEN("Asked to generate Lonely plans")
        {
            Plans plans = concatPart.GetPlans(CascadeType::Lonely, command_stream::BlockConfig{}, nullptr, 0);
            SavePlansToDot(plans, "ConcatPart GetPlans structure Lonely");

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
            Plans plans = concatPart.GetPlans(CascadeType::Beginning, command_stream::BlockConfig{}, nullptr, 0);
            SavePlansToDot(plans, "ConcatPart GetPlans structure Beginning");

            THEN("The number of generated plans = 0")
            {
                CHECK(plans.size() == 0);
            }
        }

        WHEN("Asked to generate Middle plans")
        {
            Plans plans = concatPart.GetPlans(CascadeType::Middle, command_stream::BlockConfig{}, nullptr, 0);
            SavePlansToDot(plans, "ConcatPart GetPlans structure Middle");

            THEN("The number of generated plans = 0")
            {
                CHECK(plans.size() == 0);
            }
        }

        WHEN("Asked to generate End plans")
        {
            Plans plans = concatPart.GetPlans(CascadeType::End, command_stream::BlockConfig{}, nullptr, 0);
            SavePlansToDot(plans, "ConcatPart GetPlans structure End");

            THEN("The number of generated plans = 0")
            {
                CHECK(plans.size() == 0);
            }
        }
    }
}