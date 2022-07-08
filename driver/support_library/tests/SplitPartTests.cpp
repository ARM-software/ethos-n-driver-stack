//
// Copyright © 2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "CapabilitiesInternal.hpp"
#include "GlobalParameters.hpp"
#include "TestUtils.hpp"
#include "Utils.hpp"
#include "cascading/Part.hpp"
#include "cascading/PartUtils.hpp"
#include "cascading/Plan.hpp"
#include "cascading/SplitPart.hpp"
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
    TensorInfo m_InputTensorInfo;
    std::vector<TensorInfo> m_OutputTensorInfos;
    QuantizationInfo m_OutputQuantInfo;
    std::set<uint32_t> m_OperationIds;
    CascadingBufferFormat m_DataFormat;
};

void CheckSplitOperation(const Plan& plan)
{
    // Check operation, consumers, and producers
    CHECK(plan.m_OpGraph.GetOps().size() == 2);
    Op* op = plan.m_OpGraph.GetOp(0);

    const OpGraph::BufferList& buffers = plan.m_OpGraph.GetBuffers();

    CHECK(plan.m_OpGraph.GetConsumers(buffers.front()).front().first != nullptr);
    CHECK(plan.m_OpGraph.GetConsumers(buffers.front()).front().first == op);

    for (uint32_t outputIndex = 1; outputIndex < buffers.size() - 1; outputIndex++)
    {
        CHECK(plan.m_OpGraph.GetProducer(buffers[outputIndex]) != nullptr);
        CHECK(plan.m_OpGraph.GetProducer(buffers[outputIndex]) == op);
    }
}

void CheckSplitDram(std::vector<Buffer*> splitBuffers, const CheckPlansParams& params)
{
    // Check properties of split DRAM buffers
    for (uint32_t bufferIndex = 0; bufferIndex < splitBuffers.size(); ++bufferIndex)
        if (splitBuffers[bufferIndex])
        {
            CHECK(splitBuffers[bufferIndex]->m_Location == Location::Dram);
            CHECK(splitBuffers[bufferIndex]->m_Format == params.m_DataFormat);
            CHECK(splitBuffers[bufferIndex]->m_TensorShape == params.m_OutputTensorInfos[bufferIndex].m_Dimensions);
            CHECK(splitBuffers[bufferIndex]->m_Order == TraversalOrder::Xyz);
            CHECK(splitBuffers[bufferIndex]->m_SizeInBytes ==
                  utils::TotalSizeBytes(params.m_OutputTensorInfos[bufferIndex].m_Dimensions));
            CHECK(splitBuffers[bufferIndex]->m_NumStripes == 0);
            CHECK(splitBuffers[bufferIndex]->m_EncodedWeights == nullptr);
        }
}

void CheckMappings(const CheckPlansParams& params, const Plan& plan, std::vector<Buffer*> splitBuffers)
{
    // Check input/output mappings
    const OpGraph::BufferList& buffers = plan.m_OpGraph.GetBuffers();

    CHECK(plan.m_InputMappings.size() == 1);
    CHECK(plan.m_OutputMappings.size() == buffers.size() - 1);

    CHECK(plan.m_InputMappings.begin()->second.m_PartId == params.m_PartId);
    CHECK(plan.m_InputMappings.begin()->second.m_InputIndex == 0);

    for (uint32_t outputIndex = 1; outputIndex < buffers.size(); outputIndex++)
    {
        CHECK(plan.m_OutputMappings.at(buffers[outputIndex]).m_PartId == params.m_PartId);
        CHECK(plan.m_OutputMappings.at(buffers[outputIndex]).m_OutputIndex == outputIndex - 1);
    }

    ETHOSN_UNUSED(splitBuffers);
}

/// Checks that the given list of Plans matches expectations, based on both generic requirements of all plans (e.g. all plans
/// must follow the expected OpGraph structure) and also specific requirements on plans which can be customized using the provided callbacks.
/// These are all configured by the CheckPlansParams struct.
void CheckPlans(const Plans& plans, const CheckPlansParams& params, const SplitInfo& splitInfo)
{
    CHECK(plans.size() > 0);

    for (auto&& plan : plans)
    {
        INFO("plan " << plan.m_DebugTag);

        const OpGraph::BufferList& buffers = plan.m_OpGraph.GetBuffers();

        std::vector<Buffer*> splitBuffers;
        for (uint32_t bufferIndex = 1; bufferIndex < splitInfo.m_Sizes.size() + 1; ++bufferIndex)
        {
            splitBuffers.push_back(buffers[bufferIndex]);
        }

        CheckSplitOperation(plan);
        CheckSplitDram(splitBuffers, params);
        CheckMappings(params, plan, splitBuffers);
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

TEST_CASE("SplitPart Plan Generation", "[SplitPartTests]")
{
    GIVEN("A simple SplitPart")
    {
        const PartId partId = 1;

        TensorInfo inputTensorInfo;
        CompilerDataFormat compilerDataFormat;
        DataFormat dataFormat = GENERATE(DataFormat::NHWC, DataFormat::NHWCB);

        if (dataFormat == DataFormat::NHWC)
        {
            inputTensorInfo.m_Dimensions = { 1, 16, 16, 16 };
            inputTensorInfo.m_DataType   = DataType::INT8_QUANTIZED;
            inputTensorInfo.m_DataFormat = DataFormat::NHWC;

            compilerDataFormat = CompilerDataFormat::NHWC;
        }
        else
        {
            inputTensorInfo.m_Dimensions = { 1, 16, 16, 16 };
            inputTensorInfo.m_DataType   = DataType::INT8_QUANTIZED;
            inputTensorInfo.m_DataFormat = DataFormat::NHWCB;

            compilerDataFormat = CompilerDataFormat::NHWCB;
        }

        std::vector<uint32_t> outputTensorShapes{ 8, 8 };
        uint32_t splitAxis{ 1 };
        SplitInfo splitInfo(splitAxis, outputTensorShapes);

        const std::set<uint32_t> operationIds = { 1, 2 };
        const EstimationOptions estOpt;
        const CompilationOptions compOpt;
        HardwareCapabilities hwCapabilities(GetEthosN78FwHwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO));

        SplitPart splitPart(partId, inputTensorInfo, splitInfo, compilerDataFormat, operationIds, estOpt, compOpt,
                            hwCapabilities);

        CheckPlansParams params;
        params.m_PartId            = partId;
        params.m_InputTensorInfo   = inputTensorInfo;
        params.m_OutputTensorInfos = Split::CalculateOutputTensorInfos(params.m_InputTensorInfo, splitInfo);
        params.m_OperationIds      = operationIds;

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
            Plans plans = splitPart.GetPlans(CascadeType::Lonely, command_stream::BlockConfig{}, nullptr, 0);
            SavePlansToDot(plans, "SplitPart GetPlans structure Lonely");

            THEN("The number of generated plans = 1")
            {
                CHECK(plans.size() == 1);
            }

            AND_THEN("The plan is valid and end in Dram")
            {
                CheckPlans(plans, params, splitInfo);
            }
        }

        WHEN("Asked to generate Beginning plans")
        {
            Plans plans = splitPart.GetPlans(CascadeType::Beginning, command_stream::BlockConfig{}, nullptr, 0);
            SavePlansToDot(plans, "SplitPart GetPlans structure Beginning");

            THEN("The number of generated plans = 0")
            {
                CHECK(plans.size() == 0);
            }
        }

        WHEN("Asked to generate Middle plans")
        {
            Plans plans = splitPart.GetPlans(CascadeType::Middle, command_stream::BlockConfig{}, nullptr, 0);
            SavePlansToDot(plans, "SplitPart GetPlans structure Middle");

            THEN("The number of generated plans = 0")
            {
                CHECK(plans.size() == 0);
            }
        }

        WHEN("Asked to generate End plans")
        {
            Plans plans = splitPart.GetPlans(CascadeType::End, command_stream::BlockConfig{}, nullptr, 0);
            SavePlansToDot(plans, "SplitPart GetPlans structure End");

            THEN("The number of generated plans = 0")
            {
                CHECK(plans.size() == 0);
            }
        }
    }
}
