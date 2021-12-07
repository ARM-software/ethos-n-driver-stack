//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "CapabilitiesInternal.hpp"
#include "GlobalParameters.hpp"
#include "TestUtils.hpp"
#include "Utils.hpp"
#include "cascading/EstimateOnlyPart.hpp"
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
    std::vector<TensorInfo> m_InputTensorsInfo;
    std::vector<TensorInfo> m_OutputTensorsInfo;
    std::set<uint32_t> m_OperationIds;
    CascadingBufferFormat m_DataFormat;
};

/// Checks that the given list of Plans matches expectations, based on both generic requirements of all plans (e.g. all plans
/// must follow the expected OpGraph structure) and also specific requirements on plans which can be customized using the provided callbacks.
/// These are all configured by the CheckPlansParams struct.
void CheckPlans(const Plans& plans, const CheckPlansParams& params)
{
    // Expected to have only one plan for EstimateOnlyPart
    CHECK(plans.size() == 1);

    INFO("plan " << plans[0].m_DebugTag);

    const OpGraph::BufferList& buffers = plans[0].m_OpGraph.GetBuffers();

    // Check properties of output DRAM buffers
    // Output buffers indices should be 2 and 3 given that the test case below has 2 input buffers and 2 output buffers.
    for (uint32_t outputIndex = 2; outputIndex < buffers.size(); outputIndex++)
    {
        CHECK(buffers[outputIndex]->m_Location == Location::Dram);
        CHECK(buffers[outputIndex]->m_Lifetime == Lifetime::Atomic);
        CHECK(buffers[outputIndex]->m_Format == CascadingBufferFormat::NHWCB);
        CHECK(buffers[outputIndex]->m_QuantizationInfo ==
              params.m_OutputTensorsInfo[outputIndex - 2].m_QuantizationInfo);
        CHECK(buffers[outputIndex]->m_TensorShape == params.m_OutputTensorsInfo[outputIndex - 2].m_Dimensions);
        CHECK(buffers[outputIndex]->m_StripeShape == TensorShape{ 0, 0, 0, 0 });
        CHECK(buffers[outputIndex]->m_Order == TraversalOrder::Xyz);
        CHECK(buffers[outputIndex]->m_SizeInBytes ==
              utils::TotalSizeBytesNHWCB(params.m_OutputTensorsInfo[outputIndex - 2].m_Dimensions));
        CHECK(buffers[outputIndex]->m_NumStripes == 0);
        CHECK(buffers[outputIndex]->m_EncodedWeights == nullptr);
    }

    // Check the input and output mappings
    CHECK(plans[0].m_InputMappings.size() == 2);
    CHECK(plans[0].m_OutputMappings.size() == 2);

    for (uint32_t inputIndex = 0; inputIndex < 2; inputIndex++)
    {
        CHECK(plans[0].m_InputMappings.at(buffers[inputIndex]).m_PartId == params.m_PartId);
        CHECK(plans[0].m_InputMappings.at(buffers[inputIndex]).m_InputIndex == inputIndex);
    }

    for (uint32_t outputIndex = 2; outputIndex < buffers.size(); outputIndex++)
    {
        CHECK(plans[0].m_OutputMappings.at(buffers[outputIndex]).m_PartId == params.m_PartId);
        CHECK(plans[0].m_OutputMappings.at(buffers[outputIndex]).m_OutputIndex == outputIndex - 2);
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

TEST_CASE("EstimateOnlyPart Plan Generation", "[EstimateOnlyPartTests]")
{
    GIVEN("A simple EstimateOnlyPart")
    {
        const PartId partId = 1;
        std::vector<TensorInfo> inputTensorsInfo;
        std::vector<TensorInfo> outputTensorsInfo;
        TensorInfo inputTensorInfo1;
        TensorInfo inputTensorInfo2;
        TensorInfo outputTensorInfo1;
        TensorInfo outputTensorInfo2;

        inputTensorInfo1.m_Dimensions       = { 1, 16, 16, 16 };
        inputTensorInfo1.m_DataType         = DataType::INT8_QUANTIZED;
        inputTensorInfo1.m_DataFormat       = DataFormat::NHWCB;
        inputTensorInfo1.m_QuantizationInfo = QuantizationInfo(0, 1.0f);

        inputTensorInfo2.m_Dimensions       = { 1, 16, 16, 16 };
        inputTensorInfo2.m_DataType         = DataType::INT8_QUANTIZED;
        inputTensorInfo2.m_DataFormat       = DataFormat::NHWCB;
        inputTensorInfo2.m_QuantizationInfo = QuantizationInfo(0, 1.0f);

        outputTensorInfo1.m_Dimensions       = { 1, 16, 16, 16 };
        outputTensorInfo1.m_DataType         = DataType::INT8_QUANTIZED;
        outputTensorInfo1.m_DataFormat       = DataFormat::NHWCB;
        outputTensorInfo1.m_QuantizationInfo = QuantizationInfo(0, 1.0f);

        outputTensorInfo2.m_Dimensions       = { 1, 16, 16, 16 };
        outputTensorInfo2.m_DataType         = DataType::INT8_QUANTIZED;
        outputTensorInfo2.m_DataFormat       = DataFormat::NHWCB;
        outputTensorInfo2.m_QuantizationInfo = QuantizationInfo(0, 1.0f);

        CompilerDataFormat compilerDataFormat = CompilerDataFormat::NHWCB;

        inputTensorsInfo.push_back(inputTensorInfo1);
        inputTensorsInfo.push_back(inputTensorInfo2);
        outputTensorsInfo.push_back(outputTensorInfo1);
        outputTensorsInfo.push_back(outputTensorInfo2);

        const std::set<uint32_t> operationIds = { 1 };
        const EstimationOptions estOpt;
        const CompilationOptions compOpt;
        HardwareCapabilities hwCapabilities(GetEthosN78FwHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));
        std::string reasonForEstimateOnly =
            "EstimateOnly operation added for internal EstimateOnlyMultipleInputsOutputs test.";

        EstimateOnlyPart estimateOnlyPart(partId, reasonForEstimateOnly, inputTensorsInfo, outputTensorsInfo,
                                          compilerDataFormat, operationIds, estOpt, compOpt, hwCapabilities);

        CheckPlansParams params;
        params.m_PartId            = partId;
        params.m_InputTensorsInfo  = inputTensorsInfo;
        params.m_OutputTensorsInfo = outputTensorsInfo;
        params.m_OperationIds      = operationIds;
        params.m_DataFormat        = CascadingBufferFormat::NHWCB;

        WHEN("Asked to generate Lonely plans")
        {
            Plans plans = estimateOnlyPart.GetPlans(CascadeType::Lonely, command_stream::BlockConfig{}, nullptr, 0);
            SavePlansToDot(plans, "EstimateOnlyPart GetPlans structure Lonely");

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
            Plans plans = estimateOnlyPart.GetPlans(CascadeType::Beginning, command_stream::BlockConfig{}, nullptr, 0);
            SavePlansToDot(plans, "EstimateOnlyPart GetPlans structure Beginning");

            THEN("The number of generated plans = 0")
            {
                CHECK(plans.size() == 0);
            }
        }

        WHEN("Asked to generate Middle plans")
        {
            Plans plans = estimateOnlyPart.GetPlans(CascadeType::Middle, command_stream::BlockConfig{}, nullptr, 0);
            SavePlansToDot(plans, "EstimateOnlyPart GetPlans structure Middle");

            THEN("The number of generated plans = 0")
            {
                CHECK(plans.size() == 0);
            }
        }

        WHEN("Asked to generate End plans")
        {
            Plans plans = estimateOnlyPart.GetPlans(CascadeType::End, command_stream::BlockConfig{}, nullptr, 0);
            SavePlansToDot(plans, "EstimateOnlyPart GetPlans structure End");

            THEN("The number of generated plans = 0")
            {
                CHECK(plans.size() == 0);
            }
        }
    }
}
