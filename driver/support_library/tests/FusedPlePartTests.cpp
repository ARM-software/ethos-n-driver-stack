//
// Copyright © 2021-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../src/Compiler.hpp"
#include "../src/FusedPlePart.hpp"
#include "../src/Visualisation.hpp"
#include "../src/WeightEncoderCache.hpp"
#include "CapabilitiesInternal.hpp"
#include "GlobalParameters.hpp"
#include "TestUtils.hpp"
#include "ThreadPool.hpp"
#include "Utils.hpp"
#include "ethosn_support_library/Support.hpp"

#include <catch.hpp>
#include <fstream>
#include <regex>
#include <sstream>

using namespace ethosn::support_library;
namespace command_stream = ethosn::command_stream;

namespace
{

/// Tests that the given object of Base type is of the given Derived type (using RTTI), and returns the casted object.
/// This isn't a good pattern in general, but is appropriate for unit testing the result of 'factory methods' like
/// our plan generation (which return Ops of different concrete types).
template <typename Derived, typename Base>
Derived RequireCast(Base b)
{
    Derived result = dynamic_cast<Derived>(b);
    REQUIRE(result != nullptr);
    return result;
}

FusedPlePart BuildPart(TensorShape inputShape,
                       TensorShape outputShape,
                       PleOperation op,
                       utils::ShapeMultiplier shapeMultiplier,
                       const CompilationOptions& compOpts,
                       const HardwareCapabilities& caps,
                       const EstimationOptions& estOpts,
                       DebuggingContext& debuggingContext,
                       ThreadPool& threadPool)
{
    const PartId partId = 0;
    const QuantizationInfo inputQuantInfo(0, 1.0f);
    const QuantizationInfo outputQuantInfo(0, 1.0f);
    const std::set<uint32_t> operationsIds = { 1 };
    FusedPlePart part(
        partId, inputShape, outputShape, inputQuantInfo, outputQuantInfo, op, shapeMultiplier, estOpts, compOpts, caps,
        operationsIds, DataType::UINT8_QUANTIZED, DataType::UINT8_QUANTIZED, debuggingContext, threadPool,
        op == PleOperation::LEAKY_RELU ? std::map<std::string, std::string>{ { "datatype", "u8" } }
                                       : std::map<std::string, std::string>{},
        std::map<std::string, int>{ { "block_width", 16 }, { "block_height", 16 }, { "block_multiplier", 1 } },
        std::map<std::string, int>{});

    part.SetOutputRequirements({ BoundaryRequirements{} }, { false });

    return part;
}

FusedPlePart BuildPart(TensorShape inputShape,
                       TensorShape outputShape,
                       PleOperation op,
                       const CompilationOptions& compOpts,
                       const HardwareCapabilities& caps,
                       const EstimationOptions& estOpts,
                       DebuggingContext& debuggingContext,
                       ThreadPool& threadPool)
{
    return BuildPart(inputShape, outputShape, op, utils::ShapeMultiplier{ 1, 1, 1 }, compOpts, caps, estOpts,
                     debuggingContext, threadPool);
}

struct PlanDesc
{
    Buffer* m_InputDram    = nullptr;
    Buffer* m_InputSram    = nullptr;
    Buffer* m_WeightsDram  = nullptr;
    Buffer* m_WeightsSram  = nullptr;
    Buffer* m_PleInputSram = nullptr;
    Buffer* m_OutputSram   = nullptr;
    Buffer* m_OutputDram   = nullptr;

    DmaOp* m_InputDma   = nullptr;
    DmaOp* m_WeightsDma = nullptr;
    MceOp* m_Mce        = nullptr;
    PleOp* m_Ple        = nullptr;
    DmaOp* m_OutputDma  = nullptr;

    Buffer* m_Input  = nullptr;
    Buffer* m_Output = nullptr;
};

enum class PlanInputLocation
{
    PleInputSram = 0x1,
    Sram         = 0x2,
    Dram         = 0x4,
};

enum class PlanOutputLocation : uint32_t
{
    Sram = 0x1,
    Dram = 0x2,
};

using PlanDescFunc      = std::function<void(const PlanDesc& planDesc)>;
using PlanDescPredicate = std::function<bool(const PlanDesc& planDesc)>;

struct CheckPlansParams
{
    /// The structure of the expected plans. If the OpGraph structure of any plans are not consistent with
    /// the input/output locations allowed here, then the test will fail.
    /// @{
    PlanInputLocation m_InputLocation   = PlanInputLocation::Sram;
    PlanOutputLocation m_OutputLocation = PlanOutputLocation::Sram;
    /// @}

    /// If provided, the properties of Ops and Buffers all plans must meet, otherwise the test will fail.
    /// @{
    utils::Optional<PartId> m_PartId;
    utils::Optional<TensorShape> m_InputShape;
    utils::Optional<QuantizationInfo> m_InputQuantInfo;
    utils::Optional<TensorShape> m_OutputShape;
    utils::Optional<QuantizationInfo> m_OutputQuantInfo;
    utils::Optional<PleOperation> m_PleOp;
    utils::Optional<std::set<uint32_t>> m_OperationIds;
    /// @}

    std::vector<PlanDescPredicate>
        m_Any;    ///< At least one plan must pass each of these predicates (though not necessarily the same plan for each).
    PlanDescFunc m_All =
        nullptr;    ///< If set, this function will be called once per plan, to perform additional checks on all plans.
};

// Get the buffers from the OpGraph
void ExtractBuffers(const Plan& plan, PlanDesc& desc, const CheckPlansParams& params)
{
    const OpGraph::BufferList& buffers = plan.m_OpGraph.GetBuffers();
    size_t i                           = 0;
    desc.m_Input                       = buffers.front();
    if (params.m_InputLocation == PlanInputLocation::Dram)
    {
        desc.m_InputDram = buffers.at(i++);
    }
    if (params.m_InputLocation != PlanInputLocation::PleInputSram)
    {
        desc.m_InputSram   = buffers.at(i++);
        desc.m_WeightsDram = buffers.at(i++);
        desc.m_WeightsSram = buffers.at(i++);
    }
    desc.m_PleInputSram = buffers.at(i++);
    if ((params.m_OutputLocation == PlanOutputLocation::Sram) && i + 1 == buffers.size())
    {
        desc.m_OutputSram = buffers.at(i++);
    }
    else if ((params.m_OutputLocation == PlanOutputLocation::Dram) && i + 2 == buffers.size())
    {
        desc.m_OutputSram = buffers.at(i++);
        desc.m_OutputDram = buffers.at(i++);
    }
    else
    {
        FAIL("Unexpected number of buffers");
    }
    desc.m_Output = buffers.back();
}

void CheckInputDram(const CheckPlansParams& params, PlanDesc& desc)
{
    // Check properties of Input DRAM buffer (if we have one)
    if (params.m_InputLocation == PlanInputLocation::Dram)
    {
        CHECK(desc.m_InputDram->m_Location == Location::Dram);
        CHECK(desc.m_InputDram->m_Format == BufferFormat::NHWCB);
        if (params.m_InputQuantInfo)
        {
            CHECK(desc.m_InputDram->m_QuantizationInfo == params.m_InputQuantInfo.value());
        }
        if (params.m_InputShape)
        {
            CHECK(desc.m_InputDram->m_TensorShape == params.m_InputShape.value());
        }
        CHECK(desc.m_InputDram->m_SizeInBytes == utils::TotalSizeBytesNHWCB(desc.m_InputDram->m_TensorShape));
        CHECK(desc.m_InputDram->Dram()->m_EncodedWeights == nullptr);
    }
}

void CheckInputSram(PlanDesc& desc, const CheckPlansParams& params)
{
    // Check properties of Input SRAM buffer
    if (params.m_InputLocation != PlanInputLocation::PleInputSram)
    {
        CHECK(desc.m_InputSram->m_Location == Location::Sram);
        CHECK(desc.m_InputSram->m_Format == BufferFormat::NHWCB);
        if (params.m_InputQuantInfo)
        {
            CHECK(desc.m_InputSram->m_QuantizationInfo == params.m_InputQuantInfo.value());
        }
        else if (
            desc.m_InputDram)    // If we weren't provided with an expected quant info, then at least check that it's consistent between the Dram and Sram buffers
        {
            CHECK(desc.m_InputSram->m_QuantizationInfo == desc.m_InputDram->m_QuantizationInfo);
        }
        if (params.m_InputShape)
        {
            CHECK(desc.m_InputSram->m_TensorShape == params.m_InputShape.value());
        }
        else if (
            desc.m_InputDram)    // If we weren't provided with an expected shape, then at least check that it's consistent between the Dram and Sram buffers
        {
            CHECK(desc.m_InputSram->m_TensorShape == desc.m_InputDram->m_TensorShape);
        }
        // m_StripeShape, m_Order, m_SizeInBytes and m_NumStripes will depend on the streaming strategy, and so cannot be checked generically
    }
}

void CheckWeightsDram(PlanDesc& desc, const CheckPlansParams& params)
{
    // Check properties of Weights DRAM buffer
    if (params.m_InputLocation != PlanInputLocation::PleInputSram)
    {
        CHECK(desc.m_WeightsDram->m_Location == Location::Dram);
        CHECK(desc.m_WeightsDram->m_Format == BufferFormat::WEIGHT);
        CHECK(desc.m_WeightsDram->m_QuantizationInfo == QuantizationInfo{ 0, 0.5f });
        CHECK(desc.m_WeightsDram->m_TensorShape == TensorShape{ 1, 1, desc.m_Input->m_TensorShape[3], 1 });
        REQUIRE(desc.m_WeightsDram->Dram()->m_EncodedWeights != nullptr);
        CHECK(desc.m_WeightsDram->Dram()->m_EncodedWeights->m_Data.size() > 0);
        CHECK(desc.m_WeightsDram->m_SizeInBytes == desc.m_WeightsDram->Dram()->m_EncodedWeights->m_Data.size());
    }
}

void CheckWeightsSram(PlanDesc& desc, const CheckPlansParams& params)
{
    // Check properties of Weights SRAM buffer
    if (params.m_InputLocation != PlanInputLocation::PleInputSram)
    {
        CHECK(desc.m_WeightsSram->m_Location == Location::Sram);
        CHECK(desc.m_WeightsSram->m_Format == BufferFormat::WEIGHT);
        CHECK(desc.m_WeightsSram->m_QuantizationInfo == QuantizationInfo{ 0, 0.5f });
        CHECK(desc.m_WeightsSram->m_TensorShape == TensorShape{ 1, 1, desc.m_Input->m_TensorShape[3], 1 });
        // m_StripeShape, m_Order, m_NumStripes will depend on the streaming strategy, and so cannot be checked generically
        CHECK(desc.m_WeightsSram->Sram()->m_SizeInBytes ==
              desc.m_WeightsDram->Dram()->m_EncodedWeights->m_MaxSize * desc.m_WeightsSram->Sram()->m_NumStripes);
    }
}

void CheckPleInputSram(PlanDesc& desc, const CheckPlansParams& params)
{
    // Check properties of Ple Input SRAM buffer
    CHECK(desc.m_PleInputSram->m_Location == Location::PleInputSram);
    CHECK(desc.m_PleInputSram->m_Format == BufferFormat::NHWCB);
    if (params.m_OutputQuantInfo)
    {
        // Note if this isn't provided, we can still check the quant info by comparing with the m_OutputSram buffer,
        // if that is present (see CheckOutputSram).
        CHECK(desc.m_PleInputSram->m_QuantizationInfo == params.m_OutputQuantInfo.value());
    }
    if (params.m_InputShape)
    {
        // Note if this isn't provided, we can still check the tensor shape by comparing with the m_OutputSram buffer,
        // if that is present (see CheckOutputSram).
        CHECK(desc.m_PleInputSram->m_TensorShape == params.m_InputShape.value());
    }
    // m_StripeShape, m_Order, m_SizeInBytes, m_NumStripes will depend on the streaming strategy, and so cannot be checked generically
}

void CheckOutputSram(PlanDesc& desc, const CheckPlansParams& params)
{
    // Check properties of Output SRAM buffer (if we have one)
    if (desc.m_OutputSram)
    {
        CHECK(desc.m_OutputSram->m_Location == Location::Sram);
        CHECK(desc.m_OutputSram->m_Format == BufferFormat::NHWCB);
        if (params.m_OutputQuantInfo)
        {
            CHECK(desc.m_OutputSram->m_QuantizationInfo == params.m_OutputQuantInfo.value());
        }
        else    // If we weren't provided with an expected output tensor info, then at least check that it's consistent
        {
            CHECK(desc.m_OutputSram->m_QuantizationInfo == desc.m_PleInputSram->m_QuantizationInfo);
        }
        if (params.m_OutputShape)
        {
            CHECK(desc.m_OutputSram->m_TensorShape == params.m_OutputShape.value());
        }
        // m_StripeShape, m_Order, m_SizeInBytes and m_NumStripes will depend on the streaming strategy, and so cannot be checked generically
    }
}

void CheckOutputDram(PlanDesc& desc, const CheckPlansParams& params)
{
    // Check properties of Output DRAM buffer (if we have one)
    if (desc.m_OutputDram)
    {
        CHECK(desc.m_OutputDram->m_Location == Location::Dram);
        CHECK(desc.m_OutputDram->m_Format == BufferFormat::NHWCB);
        if (params.m_OutputQuantInfo)
        {
            CHECK(desc.m_OutputDram->m_QuantizationInfo == params.m_OutputQuantInfo.value());
        }
        else    // If we weren't provided with an expected quant info, then at least check that it's consistent
        {
            CHECK(desc.m_OutputDram->m_QuantizationInfo == desc.m_OutputSram->m_QuantizationInfo);
        }
        if (params.m_OutputShape)
        {
            CHECK(desc.m_OutputDram->m_TensorShape == params.m_OutputShape.value());
        }
        else    // If we weren't provided with an expected shape, then at least check that it's consistent
        {
            CHECK(desc.m_OutputDram->m_TensorShape == desc.m_OutputSram->m_TensorShape);
        }
        CHECK(desc.m_OutputDram->m_SizeInBytes == utils::TotalSizeBytesNHWCB(desc.m_OutputDram->m_TensorShape));
        CHECK(desc.m_OutputDram->Dram()->m_EncodedWeights == nullptr);
    }
}

void ExtractOps(const Plan& plan, const CheckPlansParams& params, PlanDesc& desc)
{
    // Get the ops from the OpGraph
    {
        auto ops = plan.m_OpGraph.GetOps();
        size_t i = 0;
        if (params.m_InputLocation == PlanInputLocation::Dram)
        {
            desc.m_InputDma = RequireCast<DmaOp*>(ops.at(i++));
        }
        if (params.m_InputLocation != PlanInputLocation::PleInputSram)
        {
            desc.m_WeightsDma = RequireCast<DmaOp*>(ops.at(i++));
            desc.m_Mce        = RequireCast<MceOp*>(ops.at(i++));
        }
        if ((params.m_OutputLocation == PlanOutputLocation::Sram) && i + 1 == ops.size())
        {
            desc.m_Ple = RequireCast<PleOp*>(ops.at(i++));
        }
        else if ((params.m_OutputLocation == PlanOutputLocation::Dram) && i + 2 == ops.size())
        {
            desc.m_Ple       = RequireCast<PleOp*>(ops.at(i++));
            desc.m_OutputDma = RequireCast<DmaOp*>(ops.at(i++));
        }
        else
        {
            FAIL("Unexpected number of ops");
        }
    }
}

void CheckInputDma(const CheckPlansParams& params, PlanDesc& desc)
{
    // Check properties of Input DMA (if we have one)
    if (params.m_InputLocation == PlanInputLocation::Dram)
    {
        if (params.m_OperationIds)
        {
            CHECK(desc.m_InputDma->m_OperationIds == params.m_OperationIds.value());
        }
    }
}

void CheckWeightsDma(const CheckPlansParams& params, PlanDesc& desc)
{
    // Check properties of Weights DMA
    if (params.m_InputLocation != PlanInputLocation::PleInputSram)
    {
        if (params.m_OperationIds)
        {
            CHECK(desc.m_WeightsDma->m_OperationIds == params.m_OperationIds.value());
        }
    }
}

void CheckMce(const CheckPlansParams& params, PlanDesc& desc)
{
    // Check properties of Mce Op

    if (params.m_InputLocation != PlanInputLocation::PleInputSram)
    {
        if (params.m_OperationIds)
        {
            CHECK(desc.m_Mce->m_OperationIds == params.m_OperationIds.value());
        }
        CHECK(desc.m_Mce->m_Op == command_stream::MceOperation::DEPTHWISE_CONVOLUTION);
        // m_Algo, m_Block, m_InputStripeShape, m_OutputStripeShape, m_WeightsStripeShape, m_Order will depend on the streaming strategy, and so cannot be checked generically
        CHECK(desc.m_Mce->m_Stride == Stride{ 1, 1 });
        CHECK(desc.m_Mce->m_PadLeft == 0);
        CHECK(desc.m_Mce->m_PadTop == 0);
    }
}

void CheckPle(PlanDesc& desc, const CheckPlansParams& params)
{
    // Check properties of Ple Op (if we have one)
    if (desc.m_Ple)
    {
        if (params.m_OperationIds)
        {
            CHECK(desc.m_Ple->m_OperationIds == params.m_OperationIds.value());
        }
        if (params.m_PleOp)
        {
            CHECK(desc.m_Ple->m_Op == params.m_PleOp.value());
        }
        // m_BlockConfig will depend on the streaming strategy, and so cannot be checked generically
        CHECK(desc.m_Ple->m_NumInputs == 1);
        CHECK(
            desc.m_Ple->m_InputStripeShapes.size() ==
            1);    // The shapes themselves will depend on the streaming strategy, and so cannot be checked generically
    }
}

void CheckOutputDma(PlanDesc& desc, const CheckPlansParams& params)
{
    // Check properties of Output DMA (if we have one)
    if (desc.m_OutputDma)
    {
        if (params.m_OperationIds)
        {
            CHECK(desc.m_OutputDma->m_OperationIds == params.m_OperationIds.value());
        }
    }
}

void CheckConnections(const CheckPlansParams& params, const Plan& plan, PlanDesc& desc)
{
    // Check OpGraph connections
    if (params.m_InputLocation == PlanInputLocation::Dram)
    {
        CHECK(plan.m_OpGraph.GetSingleProducer(desc.m_InputDram) == nullptr);
        CHECK(plan.m_OpGraph.GetConsumers(desc.m_InputDram) ==
              std::vector<std::pair<Op*, uint32_t>>{ { desc.m_InputDma, 0 } });
    }
    if (params.m_InputLocation != PlanInputLocation::PleInputSram)
    {
        CHECK(plan.m_OpGraph.GetSingleProducer(desc.m_InputSram) ==
              (params.m_InputLocation == PlanInputLocation::Dram ? desc.m_InputDma : nullptr));
        CHECK(plan.m_OpGraph.GetConsumers(desc.m_InputSram) ==
              std::vector<std::pair<Op*, uint32_t>>{ { desc.m_Mce, 0 } });

        CHECK(plan.m_OpGraph.GetSingleProducer(desc.m_WeightsDram) == nullptr);
        CHECK(plan.m_OpGraph.GetConsumers(desc.m_WeightsDram) ==
              std::vector<std::pair<Op*, uint32_t>>{ { desc.m_WeightsDma, 0 } });

        CHECK(plan.m_OpGraph.GetSingleProducer(desc.m_WeightsSram) == desc.m_WeightsDma);
        CHECK(plan.m_OpGraph.GetConsumers(desc.m_WeightsSram) ==
              std::vector<std::pair<Op*, uint32_t>>{ { desc.m_Mce, 1 } });
    }

    CHECK(plan.m_OpGraph.GetSingleProducer(desc.m_PleInputSram) ==
          (params.m_InputLocation == PlanInputLocation::PleInputSram ? nullptr : desc.m_Mce));
    CHECK(plan.m_OpGraph.GetConsumers(desc.m_PleInputSram) ==
          std::vector<std::pair<Op*, uint32_t>>{ { desc.m_Ple, 0 } });

    CHECK(plan.m_OpGraph.GetSingleProducer(desc.m_OutputSram) == desc.m_Ple);
    CHECK(plan.m_OpGraph.GetConsumers(desc.m_OutputSram) ==
          (desc.m_OutputDma ? std::vector<std::pair<Op*, uint32_t>>{ { desc.m_OutputDma, 0 } }
                            : std::vector<std::pair<Op*, uint32_t>>{}));
    if (desc.m_OutputDram)
    {
        CHECK(plan.m_OpGraph.GetSingleProducer(desc.m_OutputDram) == desc.m_OutputDma);
        CHECK(plan.m_OpGraph.GetConsumers(desc.m_OutputDram) == std::vector<std::pair<Op*, uint32_t>>{});
    }
}

void CheckMappings(const CheckPlansParams& params, const Plan& plan, PlanDesc& desc)
{
    // Check input/output mappings
    CHECK(plan.m_InputMappings.size() == 1);
    if (desc.m_InputDram)
    {
        CHECK(plan.m_InputMappings.begin()->first == desc.m_InputDram);
    }
    else if (desc.m_InputSram)
    {
        CHECK(plan.m_InputMappings.begin()->first == desc.m_InputSram);
    }
    else
    {
        CHECK(plan.m_InputMappings.begin()->first == desc.m_PleInputSram);
    }
    CHECK(plan.m_OutputMappings.size() == 1);
    if (desc.m_OutputDram)
    {
        CHECK(plan.m_OutputMappings.begin()->first == desc.m_OutputDram);
    }
    else if (desc.m_OutputSram)
    {
        CHECK(plan.m_OutputMappings.begin()->first == desc.m_OutputSram);
    }
    else
    {
        CHECK(plan.m_OutputMappings.begin()->first == desc.m_PleInputSram);
    }
    if (params.m_PartId)
    {
        CHECK(plan.m_InputMappings.begin()->second.m_PartId == params.m_PartId.value());
        CHECK(plan.m_OutputMappings.begin()->second.m_PartId == params.m_PartId.value());
    }
    else    // If we don't know what the PartId should be, at least check that the two mappings refer to the same one
    {
        CHECK(plan.m_InputMappings.begin()->second.m_PartId == plan.m_OutputMappings.begin()->second.m_PartId);
    }
    CHECK(plan.m_InputMappings.begin()->second.m_InputIndex == 0);
    CHECK(plan.m_OutputMappings.begin()->second.m_OutputIndex == 0);
}

/// Checks that the given list of Plans matches expectations, based on both generic requirements of all plans (e.g. all plans
/// must follow the expected OpGraph structure) and also specific requirements on plans which can be customized using the provided callbacks.
/// These are all configured by the CheckPlansParams struct.
void CheckPlans(const Plans& plans, const CheckPlansParams& params)
{
    CHECK(plans.size() > 0);

    std::vector<bool> anyPredicatesMatched(params.m_Any.size(), false);
    for (auto&& plan : plans)
    {
        INFO("plan " << plan.m_DebugTag);
        PlanDesc desc;

        ExtractBuffers(plan, desc, params);
        CheckInputDram(params, desc);
        CheckInputSram(desc, params);
        CheckWeightsDram(desc, params);
        CheckWeightsSram(desc, params);
        CheckPleInputSram(desc, params);
        CheckOutputSram(desc, params);
        CheckOutputDram(desc, params);

        ExtractOps(plan, params, desc);
        CheckInputDma(params, desc);
        CheckWeightsDma(params, desc);
        CheckMce(params, desc);
        CheckPle(desc, params);
        CheckOutputDma(desc, params);
        CheckConnections(params, plan, desc);
        CheckMappings(params, plan, desc);

        // Check custom predicates/functions for this plan
        for (size_t i = 0; i < params.m_Any.size(); ++i)
        {
            if (!anyPredicatesMatched[i])
            {
                anyPredicatesMatched[i] = params.m_Any[i](desc);
            }
        }
        if (params.m_All)
        {
            params.m_All(desc);
        }
    }

    for (size_t i = 0; i < params.m_Any.size(); ++i)
    {
        INFO("No plans matched one of the given m_Any predicates " << i);
        CHECK(anyPredicatesMatched[i]);
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

/// Checks that FusedPlePart::GetPlans returns sensible plans for different cascade types.
/// Doesn't check anything specific to any streaming strategy, just checks that the Plans have the right
/// structure and the Buffers and Ops have the right properties.
TEST_CASE("FusedPlePart GetPlans structure")
{
    GIVEN("A simple FusedPlePart")
    {
        const CompilationOptions compOpt;
        EstimationOptions estOpts;
        const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
        DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
        ThreadPool threadPool(0);

        const PartId partId = 0;
        TensorShape tsIn    = { 1, 32, 32, 3 };
        TensorShape tsOut   = { 1, 64, 64, 1 };
        const QuantizationInfo inputQuantInfo(0, 1.0f);
        const QuantizationInfo outputQuantInfo(0, 1.0f);
        const std::set<uint32_t> operationIds  = { 1, 2, 3 };
        const PleOperation csOp                = PleOperation::PASSTHROUGH;
        const utils::ShapeMultiplier shapeMult = { 1, 1, 1 };
        FusedPlePart part(partId, tsIn, tsOut, inputQuantInfo, outputQuantInfo, csOp, shapeMult, estOpts, compOpt, caps,
                          operationIds, DataType::UINT8_QUANTIZED, DataType::UINT8_QUANTIZED, debuggingContext,
                          threadPool, {}, std::map<std::string, int>{ { "block_width", 16 }, { "block_height", 16 } },
                          std::map<std::string, int>{});
        part.SetOutputRequirements({ BoundaryRequirements{} }, { false });

        CheckPlansParams params;
        params.m_PartId          = partId;
        params.m_InputShape      = tsIn;
        params.m_InputQuantInfo  = inputQuantInfo;
        params.m_OutputShape     = tsOut;
        params.m_OutputQuantInfo = outputQuantInfo;
        params.m_PleOp           = csOp;
        params.m_OperationIds    = operationIds;

        WHEN("Asked to produce Lonely plans")
        {
            Plans plans = part.GetPlans(CascadeType::Lonely, BlockConfig{}, { nullptr }, 1);
            SavePlansToDot(plans, "FusedPlePart GetPlans structure Lonely");

            THEN("The plans are valid, start and end in Sram")
            {
                params.m_InputLocation  = PlanInputLocation::Sram;
                params.m_OutputLocation = PlanOutputLocation::Sram;
                params.m_Any.push_back([](const PlanDesc& plan) {
                    return (plan.m_InputSram->Sram()->m_NumStripes == 1) &&
                           (plan.m_OutputSram->Sram()->m_NumStripes == 1);
                });
                CheckPlans(plans, params);
            }
        }

        WHEN("Asked to produce Beginning plans")
        {
            Plans plans = part.GetPlans(CascadeType::Beginning, BlockConfig{}, { nullptr }, 1);
            SavePlansToDot(plans, "FusedPlePart GetPlans structure Beginning");

            THEN("The plans are valid and start in Sram and end in Sram")
            {
                params.m_InputLocation  = PlanInputLocation::Sram;
                params.m_OutputLocation = PlanOutputLocation::Sram;
                params.m_Any.push_back([](const PlanDesc& plan) {
                    return (plan.m_InputSram->Sram()->m_NumStripes == 1) &&
                           (plan.m_OutputSram->Sram()->m_NumStripes == 1);
                });
                CheckPlans(plans, params);
            }
        }

        WHEN("Asked to produce Middle plans with an input buffer in sram")
        {
            std::unique_ptr<SramBuffer> prevBuffer = SramBuffer::Build()
                                                         .AddFormat(BufferFormat::NHWCB)
                                                         .AddQuantization({ 0, 1.0f })
                                                         .AddTensorShape(tsIn)
                                                         .AddStripeShape(TensorShape{ 1, 8, 16, 16 })
                                                         .AddTraversalOrder(TraversalOrder::Xyz)
                                                         .AddSlotSize(8 * 16 * 16 * 1)
                                                         .AddNumStripes(1);

            Plans plans = part.GetPlans(CascadeType::Middle, BlockConfig{ 16u, 16u }, { prevBuffer.get() }, 1);
            SavePlansToDot(plans, "FusedPlePart GetPlans structure Middle sram input");

            THEN("The plans are valid and start in Sram and end in Sram")
            {
                params.m_InputLocation  = PlanInputLocation::Sram;
                params.m_OutputLocation = PlanOutputLocation::Sram;
                CheckPlans(plans, params);
            }
        }

        WHEN("Asked to produce Middle plans with an input buffer in Ple Input Sram")
        {
            std::unique_ptr<PleInputSramBuffer> prevBuffer = PleInputSramBuffer::Build()
                                                                 .AddFormat(BufferFormat::NHWCB)
                                                                 .AddQuantization({ 0, 1.0f })
                                                                 .AddTensorShape(tsIn)
                                                                 .AddStripeShape(TensorShape{ 1, 8, 16, 16 })
                                                                 .AddSizeInBytes(8 * 16 * 16 * 1)
                                                                 .AddNumStripes(1);

            Plans plans = part.GetPlans(CascadeType::Middle, BlockConfig{ 16u, 16u }, { prevBuffer.get() }, 1);
            SavePlansToDot(plans, "FusedPlePart GetPlans structure Middle sram input");

            THEN("The plans are valid and start in Sram and end in Sram")
            {
                params.m_InputLocation  = PlanInputLocation::PleInputSram;
                params.m_OutputLocation = PlanOutputLocation::Sram;
                CheckPlans(plans, params);
            }
        }

        WHEN("Asked to produce End plans with an input buffer in sram")
        {
            std::unique_ptr<SramBuffer> prevBuffer = SramBuffer::Build()
                                                         .AddFormat(BufferFormat::NHWCB)
                                                         .AddQuantization({ 0, 1.0f })
                                                         .AddTensorShape(tsIn)
                                                         .AddStripeShape(TensorShape{ 1, 8, 16, 16 })
                                                         .AddTraversalOrder(TraversalOrder::Xyz)
                                                         .AddSlotSize(8 * 16 * 16 * 1)
                                                         .AddNumStripes(1);

            Plans plans = part.GetPlans(CascadeType::End, BlockConfig{ 16u, 16u }, { prevBuffer.get() }, 1);
            SavePlansToDot(plans, "FusedPlePart GetPlans structure End sram input");

            THEN("The plans are valid and start in Sram and end in Sram")
            {
                params.m_InputLocation  = PlanInputLocation::Sram;
                params.m_OutputLocation = PlanOutputLocation::Sram;
                CheckPlans(plans, params);
            }
        }

        WHEN("Asked to produce End plans with an input buffer in Ple Input Sram")
        {
            std::unique_ptr<PleInputSramBuffer> prevBuffer = PleInputSramBuffer::Build()
                                                                 .AddFormat(BufferFormat::NHWCB)
                                                                 .AddQuantization({ 0, 1.0f })
                                                                 .AddTensorShape(tsIn)
                                                                 .AddStripeShape(TensorShape{ 1, 8, 16, 16 })
                                                                 .AddSizeInBytes(8 * 16 * 16 * 1)
                                                                 .AddNumStripes(1);

            Plans plans = part.GetPlans(CascadeType::End, BlockConfig{ 16u, 16u }, { prevBuffer.get() }, 1);
            SavePlansToDot(plans, "FusedPlePart GetPlans structure End sram input");

            THEN("The plans are valid and start in Sram and end in Sram")
            {
                params.m_InputLocation  = PlanInputLocation::PleInputSram;
                params.m_OutputLocation = PlanOutputLocation::Sram;
                CheckPlans(plans, params);
            }
        }
    }
}

/// Checks that FusedPlePart::GetPlans returns sensible plans for MAXPOOL_3X3_2_2 with different cascade types.
/// Specific checks were added in order to test whether Plans are generated with the correct Height, Width, Depth split strategy.
TEST_CASE("FusedPlePart GetPlans MaxPool")
{
    GIVEN("A simple FusedPlePart")
    {
        const CompilationOptions compOpt;
        EstimationOptions estOpts;
        const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_8TOPS_2PLE_RATIO);
        DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
        ThreadPool threadPool(0);

        const PartId partId  = 0;
        TensorShape tsInEven = { 1, 128, 128, 64 };
        TensorShape tsOut    = { 1, 64, 64, 64 };
        const QuantizationInfo inputQuantInfo(0, 1.0f);
        const QuantizationInfo outputQuantInfo(0, 1.0f);
        const std::set<uint32_t> operationIds  = { 1, 2, 3 };
        const PleOperation csOpEven            = PleOperation::MAXPOOL_3X3_2_2_EVEN;
        const utils::ShapeMultiplier shapeMult = { { 1, 2 }, { 1, 2 }, 1 };
        FusedPlePart partEven(partId, tsInEven, tsOut, inputQuantInfo, outputQuantInfo, csOpEven, shapeMult, estOpts,
                              compOpt, caps, operationIds, DataType::UINT8_QUANTIZED, DataType::UINT8_QUANTIZED,
                              debuggingContext, threadPool, { { "datatype", "u8" } },
                              std::map<std::string, int>{ { "block_width", 16 }, { "block_height", 16 } },
                              std::map<std::string, int>{});
        partEven.SetOutputRequirements({ BoundaryRequirements{} }, { false });

        TensorShape tsInOdd        = { 1, 129, 129, 64 };
        const PleOperation csOpOdd = PleOperation::MAXPOOL_3X3_2_2_ODD;
        FusedPlePart partOdd(partId, tsInOdd, tsOut, inputQuantInfo, outputQuantInfo, csOpOdd, shapeMult, estOpts,
                             compOpt, caps, operationIds, DataType::UINT8_QUANTIZED, DataType::UINT8_QUANTIZED,
                             debuggingContext, threadPool, { { "datatype", "u8" } },
                             std::map<std::string, int>{ { "block_width", 16 }, { "block_height", 16 } },
                             std::map<std::string, int>{});
        partOdd.SetOutputRequirements({ BoundaryRequirements{} }, { false });

        CheckPlansParams paramsEven;
        paramsEven.m_PartId          = partId;
        paramsEven.m_InputShape      = tsInEven;
        paramsEven.m_InputQuantInfo  = inputQuantInfo;
        paramsEven.m_OutputShape     = tsOut;
        paramsEven.m_OutputQuantInfo = outputQuantInfo;
        paramsEven.m_PleOp           = csOpEven;
        paramsEven.m_OperationIds    = operationIds;

        CheckPlansParams paramsOdd;
        paramsOdd.m_PartId          = partId;
        paramsOdd.m_InputShape      = tsInOdd;
        paramsOdd.m_InputQuantInfo  = inputQuantInfo;
        paramsOdd.m_OutputShape     = tsOut;
        paramsOdd.m_OutputQuantInfo = outputQuantInfo;
        paramsOdd.m_PleOp           = csOpOdd;
        paramsOdd.m_OperationIds    = operationIds;

        WHEN("Asked to produce Lonely plans")
        {
            Plans plansEven = partEven.GetPlans(CascadeType::Lonely, BlockConfig{}, { nullptr }, 1);
            SavePlansToDot(plansEven, "FusedPlePart GetPlans MaxPoolEven Lonely");

            Plans plansOdd = partOdd.GetPlans(CascadeType::Lonely, BlockConfig{}, { nullptr }, 1);
            SavePlansToDot(plansOdd, "FusedPlePart GetPlans MaxPoolOdd Lonely");

            THEN("The plans are valid, start and end in Sram")
            {
                // Lonely: MaxPoolEven checks
                paramsEven.m_InputLocation  = PlanInputLocation::Sram;
                paramsEven.m_OutputLocation = PlanOutputLocation::Sram;
                paramsEven.m_All            = [](const PlanDesc& desc) {
                    // InputWidth: No splits are performed, Ple's InputStripe should be larger or equal to InputTensor dimension.
                    TensorShape InputTensor      = { 1, 128, 128, 64 };
                    uint32_t PleInputStripeWidth = desc.m_Ple->m_InputStripeShapes[0][2];
                    CHECK(PleInputStripeWidth >= InputTensor[2]);

                    // OutputWidth: No splits are performed, Ple's OutputStripe should be equal to OutputTensor dimension.
                    TensorShape OutputTensor      = { 1, 64, 64, 64 };
                    uint32_t PleOutputStripeWidth = desc.m_Ple->m_OutputStripeShape[2];
                    CHECK(PleOutputStripeWidth == OutputTensor[2]);
                };
                paramsEven.m_Any.push_back([](const PlanDesc& plan) {
                    return (plan.m_InputSram->Sram()->m_NumStripes == 1) &&
                           (plan.m_OutputSram->Sram()->m_NumStripes == 1);
                });
                CheckPlans(plansEven, paramsEven);

                // Lonely: MaxPoolOdd checks
                paramsOdd.m_InputLocation  = PlanInputLocation::Sram;
                paramsOdd.m_OutputLocation = PlanOutputLocation::Sram;
                paramsOdd.m_All            = [](const PlanDesc& desc) {
                    // InputWidth: No splits are performed, Ple's InputStripe should be larger or equal to InputTensor dimension.
                    TensorShape InputTensor      = { 1, 129, 129, 64 };
                    uint32_t PleInputStripeWidth = desc.m_Ple->m_InputStripeShapes[0][2];
                    CHECK(PleInputStripeWidth >= InputTensor[2]);

                    // OutputWidth: No splits are performed, Ple's OutputStripe should be equal to OutputTensor dimension.
                    TensorShape OutputTensor      = { 1, 64, 64, 64 };
                    uint32_t PleOutputStripeWidth = desc.m_Ple->m_OutputStripeShape[2];
                    CHECK(PleOutputStripeWidth == OutputTensor[2]);
                };
                paramsOdd.m_Any.push_back([](const PlanDesc& plan) {
                    return (plan.m_InputSram->Sram()->m_NumStripes == 1) &&
                           (plan.m_OutputSram->Sram()->m_NumStripes == 1);
                });
                CheckPlans(plansOdd, paramsOdd);
            }
        }

        WHEN("Asked to produce Beginning plans")
        {
            Plans plansEven = partEven.GetPlans(CascadeType::Beginning, BlockConfig{}, { nullptr }, 1);
            SavePlansToDot(plansEven, "FusedPlePart GetPlans MaxPoolEven Beginning");

            Plans plansOdd = partOdd.GetPlans(CascadeType::Beginning, BlockConfig{}, { nullptr }, 1);
            SavePlansToDot(plansOdd, "FusedPlePart GetPlans MaxPoolOdd Beginning");

            THEN("The plans are valid and start in Sram and end in Sram")
            {
                // Beginning: MaxPoolEven checks
                paramsEven.m_InputLocation  = PlanInputLocation::Sram;
                paramsEven.m_OutputLocation = PlanOutputLocation::Sram;
                paramsEven.m_All            = [](const PlanDesc& desc) {
                    // InputWidth: No splits are performed, Ple's InputStripe should be larger or equal to InputTensor dimension.
                    TensorShape InputTensor      = { 1, 128, 128, 64 };
                    uint32_t PleInputStripeWidth = desc.m_Ple->m_InputStripeShapes[0][2];
                    CHECK(PleInputStripeWidth >= InputTensor[2]);

                    // OutputWidth: No splits are performed, Ple's OutputStripe should be equal to OutputTensor dimension.
                    TensorShape OutputTensor      = { 1, 64, 64, 64 };
                    uint32_t PleOutputStripeWidth = desc.m_Ple->m_OutputStripeShape[2];
                    CHECK(PleOutputStripeWidth == OutputTensor[2]);
                };
                paramsEven.m_Any.push_back([](const PlanDesc& plan) {
                    return (plan.m_InputSram->Sram()->m_NumStripes == 1) &&
                           (plan.m_OutputSram->Sram()->m_NumStripes == 1);
                });
                CheckPlans(plansEven, paramsEven);

                // Beginning: MaxPoolOdd checks
                paramsOdd.m_InputLocation  = PlanInputLocation::Sram;
                paramsOdd.m_OutputLocation = PlanOutputLocation::Sram;
                paramsOdd.m_All            = [](const PlanDesc& desc) {
                    // InputWidth: No splits are performed, Ple's InputStripe should be larger or equal to InputTensor dimension.
                    TensorShape InputTensor      = { 1, 129, 129, 64 };
                    uint32_t PleInputStripeWidth = desc.m_Ple->m_InputStripeShapes[0][2];
                    CHECK(PleInputStripeWidth >= InputTensor[2]);

                    // OutputWidth: No splits are performed, Ple's OutputStripe should be equal to OutputTensor dimension.
                    TensorShape OutputTensor      = { 1, 64, 64, 64 };
                    uint32_t PleOutputStripeWidth = desc.m_Ple->m_OutputStripeShape[2];
                    CHECK(PleOutputStripeWidth == OutputTensor[2]);
                };
                paramsOdd.m_Any.push_back([](const PlanDesc& plan) {
                    return (plan.m_InputSram->Sram()->m_NumStripes == 1) &&
                           (plan.m_OutputSram->Sram()->m_NumStripes == 1);
                });
                CheckPlans(plansOdd, paramsOdd);
            }
        }

        WHEN("Asked to produce Middle plans with split Height in Sram")
        {
            std::unique_ptr<SramBuffer> prevBufferEven = SramBuffer::Build()
                                                             .AddFormat(BufferFormat::NHWCB)
                                                             .AddQuantization({ 0, 1.0f })
                                                             .AddTensorShape(tsInEven)
                                                             .AddStripeShape(TensorShape{ 1, 8, 128, 64 })
                                                             .AddTraversalOrder(TraversalOrder::Xyz)
                                                             .AddSlotSize(1 * 8 * 128 * 64)
                                                             .AddNumStripes(1);

            Plans plansEven =
                partEven.GetPlans(CascadeType::Middle, BlockConfig{ 8U, 8U }, { prevBufferEven.get() }, 1);
            SavePlansToDot(plansEven, "FusedPlePart GetPlans MaxPoolEven Middle Sram NoFullTensorInput");

            THEN("The are no valid plans that start in Sram and end in Sram")
            {
                CHECK(plansEven.size() == 0);
            }
        }

        WHEN("Asked to produce Middle plans with an input buffer in Sram")
        {
            std::unique_ptr<SramBuffer> prevBufferEven = SramBuffer::Build()
                                                             .AddFormat(BufferFormat::NHWCB)
                                                             .AddQuantization({ 0, 1.0f })
                                                             .AddTensorShape(tsInEven)
                                                             .AddStripeShape(TensorShape{ 1, 128, 128, 64 })
                                                             .AddTraversalOrder(TraversalOrder::Xyz)
                                                             .AddSlotSize(1 * 128 * 128 * 64)
                                                             .AddNumStripes(1);

            Plans plansEven =
                partEven.GetPlans(CascadeType::Middle, BlockConfig{ 8U, 8U }, { prevBufferEven.get() }, 1);
            SavePlansToDot(plansEven, "FusedPlePart GetPlans MaxPoolEven Middle Sram Input");

            std::unique_ptr<SramBuffer> prevBufferOdd = SramBuffer::Build()
                                                            .AddFormat(BufferFormat::NHWCB)
                                                            .AddQuantization({ 0, 1.0f })
                                                            .AddTensorShape(tsInOdd)
                                                            .AddStripeShape(TensorShape{ 1, 136, 136, 64 })
                                                            .AddTraversalOrder(TraversalOrder::Xyz)
                                                            .AddSlotSize(1 * 136 * 136 * 64)
                                                            .AddNumStripes(1);

            Plans plansOdd = partOdd.GetPlans(CascadeType::Middle, BlockConfig{ 8U, 8U }, { prevBufferOdd.get() }, 1);
            SavePlansToDot(plansOdd, "FusedPlePart GetPlans MaxPoolOdd Middle Sram Input");

            THEN("The are valid plans that start in Sram and end in Sram")
            {
                // Middle Sram: MaxPoolEven checks
                paramsEven.m_InputLocation  = PlanInputLocation::Sram;
                paramsEven.m_OutputLocation = PlanOutputLocation::Sram;
                paramsEven.m_All            = [](const PlanDesc& desc) {
                    // No splits are performed, Ple's InputStripe should be equal to InputTensor dimension.
                    TensorShape InputTensor       = { 1, 128, 128, 64 };
                    uint32_t PleInputStripeHeight = desc.m_Ple->m_InputStripeShapes[0][1];
                    uint32_t PleInputStripeWidth  = desc.m_Ple->m_InputStripeShapes[0][2];
                    uint32_t PleInputStripeDepth  = desc.m_Ple->m_InputStripeShapes[0][3];
                    CHECK(PleInputStripeHeight == InputTensor[1]);
                    CHECK(PleInputStripeWidth == InputTensor[2]);
                    CHECK(PleInputStripeDepth == InputTensor[3]);

                    // No splits are performed, Ple's OutputStripe should be equal to OutputTensor dimension.
                    TensorShape OutputTensor       = { 1, 64, 64, 64 };
                    uint32_t PleOutputStripeHeight = desc.m_Ple->m_OutputStripeShape[1];
                    uint32_t PleOutputStripeWidth  = desc.m_Ple->m_OutputStripeShape[2];
                    uint32_t PleOutputStripeDepth  = desc.m_Ple->m_OutputStripeShape[3];
                    CHECK(PleOutputStripeHeight == OutputTensor[1]);
                    CHECK(PleOutputStripeWidth == OutputTensor[2]);
                    CHECK(PleOutputStripeDepth == OutputTensor[3]);
                };
                paramsEven.m_Any.push_back([](const PlanDesc& plan) {
                    return (plan.m_InputSram->Sram()->m_NumStripes == 1) &&
                           (plan.m_OutputSram->Sram()->m_NumStripes == 1);
                });
                CheckPlans(plansEven, paramsEven);

                // Middle Sram: MaxPoolOdd checks
                paramsOdd.m_InputLocation  = PlanInputLocation::Sram;
                paramsOdd.m_OutputLocation = PlanOutputLocation::Sram;
                paramsOdd.m_All            = [](const PlanDesc& desc) {
                    // No splits are performed, Ple's InputStripe should be equal to InputTensor dimension.
                    CHECK(desc.m_Ple->m_InputStripeShapes[0][1] == 136);
                    CHECK(desc.m_Ple->m_InputStripeShapes[0][2] == 136);
                    CHECK(desc.m_Ple->m_InputStripeShapes[0][3] == 64);

                    // No splits are performed, Ple's OutputStripe should be equal to OutputTensor dimension.
                    TensorShape OutputTensor       = { 1, 64, 64, 64 };
                    uint32_t PleOutputStripeHeight = desc.m_Ple->m_OutputStripeShape[1];
                    uint32_t PleOutputStripeWidth  = desc.m_Ple->m_OutputStripeShape[2];
                    uint32_t PleOutputStripeDepth  = desc.m_Ple->m_OutputStripeShape[3];
                    CHECK(PleOutputStripeHeight == OutputTensor[1]);
                    CHECK(PleOutputStripeWidth == OutputTensor[2]);
                    CHECK(PleOutputStripeDepth == OutputTensor[3]);
                };
                paramsOdd.m_Any.push_back([](const PlanDesc& plan) {
                    return (plan.m_InputSram->Sram()->m_NumStripes == 1) &&
                           (plan.m_OutputSram->Sram()->m_NumStripes == 1);
                });
                CheckPlans(plansOdd, paramsOdd);
            }
        }

        WHEN("Asked to produce Middle plans with an input buffer in Ple Input Sram")
        {
            std::unique_ptr<PleInputSramBuffer> prevBufferEven = PleInputSramBuffer::Build()
                                                                     .AddFormat(BufferFormat::NHWCB)
                                                                     .AddQuantization({ 0, 1.0f })
                                                                     .AddTensorShape(tsInEven)
                                                                     .AddStripeShape(TensorShape{ 1, 128, 128, 64 })
                                                                     .AddSizeInBytes(1 * 128 * 128 * 64)
                                                                     .AddNumStripes(1);

            Plans plansEven =
                partEven.GetPlans(CascadeType::Middle, BlockConfig{ 8U, 8U }, { prevBufferEven.get() }, 1);
            SavePlansToDot(plansEven, "FusedPlePart GetPlans MaxPoolEven Middle Ple Sram Input");

            std::unique_ptr<PleInputSramBuffer> prevBufferOdd = PleInputSramBuffer::Build()
                                                                    .AddFormat(BufferFormat::NHWCB)
                                                                    .AddQuantization({ 0, 1.0f })
                                                                    .AddTensorShape(tsInOdd)
                                                                    .AddStripeShape(TensorShape{ 1, 136, 136, 64 })
                                                                    .AddSizeInBytes(1 * 136 * 136 * 64)
                                                                    .AddNumStripes(1);

            Plans plansOdd = partOdd.GetPlans(CascadeType::Middle, BlockConfig{ 8U, 8U }, { prevBufferOdd.get() }, 1);
            SavePlansToDot(plansOdd, "FusedPlePart GetPlans MaxPoolOdd Middle Ple Sram Input");

            THEN("The plans are valid and start in PleInputSram and end in Sram")
            {
                // Middle PleSram: MaxPoolEven checks
                paramsEven.m_InputLocation  = PlanInputLocation::PleInputSram;
                paramsEven.m_OutputLocation = PlanOutputLocation::Sram;
                paramsEven.m_All            = [](const PlanDesc& desc) {
                    // No splits are performed, Ple's InputStripe should be equal to InputTensor dimension.
                    TensorShape InputTensor       = { 1, 128, 128, 64 };
                    uint32_t PleInputStripeHeight = desc.m_Ple->m_InputStripeShapes[0][1];
                    uint32_t PleInputStripeWidth  = desc.m_Ple->m_InputStripeShapes[0][2];
                    uint32_t PleInputStripeDepth  = desc.m_Ple->m_InputStripeShapes[0][3];
                    CHECK(PleInputStripeHeight == InputTensor[1]);
                    CHECK(PleInputStripeWidth == InputTensor[2]);
                    CHECK(PleInputStripeDepth == InputTensor[3]);

                    // No splits are performed, Ple's OutputStripe should be equal to OutputTensor dimension.
                    TensorShape OutputTensor       = { 1, 64, 64, 64 };
                    uint32_t PleOutputStripeHeight = desc.m_Ple->m_OutputStripeShape[1];
                    uint32_t PleOutputStripeWidth  = desc.m_Ple->m_OutputStripeShape[2];
                    uint32_t PleOutputStripeDepth  = desc.m_Ple->m_OutputStripeShape[3];
                    CHECK(PleOutputStripeHeight == OutputTensor[1]);
                    CHECK(PleOutputStripeWidth == OutputTensor[2]);
                    CHECK(PleOutputStripeDepth == OutputTensor[3]);
                };
                CheckPlans(plansEven, paramsEven);

                // Middle PleSram: MaxPoolOdd checks
                paramsOdd.m_InputLocation  = PlanInputLocation::PleInputSram;
                paramsOdd.m_OutputLocation = PlanOutputLocation::Sram;
                paramsOdd.m_All            = [](const PlanDesc& desc) {
                    // No splits are performed, Ple's InputStripe should be equal to InputTensor dimension.
                    CHECK(desc.m_Ple->m_InputStripeShapes[0][1] == 136);
                    CHECK(desc.m_Ple->m_InputStripeShapes[0][2] == 136);
                    CHECK(desc.m_Ple->m_InputStripeShapes[0][3] == 64);

                    // No splits are performed, Ple's OutputStripe should be equal to OutputTensor dimension.
                    TensorShape OutputTensor       = { 1, 64, 64, 64 };
                    uint32_t PleOutputStripeHeight = desc.m_Ple->m_OutputStripeShape[1];
                    uint32_t PleOutputStripeWidth  = desc.m_Ple->m_OutputStripeShape[2];
                    uint32_t PleOutputStripeDepth  = desc.m_Ple->m_OutputStripeShape[3];
                    CHECK(PleOutputStripeHeight == OutputTensor[1]);
                    CHECK(PleOutputStripeWidth == OutputTensor[2]);
                    CHECK(PleOutputStripeDepth == OutputTensor[3]);
                };
                CheckPlans(plansOdd, paramsOdd);
            }
        }

        WHEN("Asked to produce End plans with an input buffer in Sram")
        {
            std::unique_ptr<SramBuffer> prevBufferEven = SramBuffer::Build()
                                                             .AddFormat(BufferFormat::NHWCB)
                                                             .AddQuantization({ 0, 1.0f })
                                                             .AddTensorShape(tsInEven)
                                                             .AddStripeShape(TensorShape{ 1, 128, 128, 64 })
                                                             .AddTraversalOrder(TraversalOrder::Xyz)
                                                             .AddSlotSize(1 * 128 * 128 * 64)
                                                             .AddNumStripes(1);

            Plans plansEven = partEven.GetPlans(CascadeType::End, BlockConfig{ 8U, 8U }, { prevBufferEven.get() }, 1);
            SavePlansToDot(plansEven, "FusedPlePart GetPlans MaxPoolEven End Sram Input");

            std::unique_ptr<SramBuffer> prevBufferOdd = SramBuffer::Build()
                                                            .AddFormat(BufferFormat::NHWCB)
                                                            .AddQuantization({ 0, 1.0f })
                                                            .AddTensorShape(tsInOdd)
                                                            .AddStripeShape(TensorShape{ 1, 136, 136, 64 })
                                                            .AddTraversalOrder(TraversalOrder::Xyz)
                                                            .AddSlotSize(1 * 136 * 136 * 64)
                                                            .AddNumStripes(1);

            Plans plansOdd = partOdd.GetPlans(CascadeType::End, BlockConfig{ 8U, 8U }, { prevBufferOdd.get() }, 1);
            SavePlansToDot(plansOdd, "FusedPlePart GetPlans MaxPoolOdd End Sram Input");

            THEN("The plans are valid and start in Sram and end in Sram")
            {
                // End Sram: MaxPoolEven checks
                paramsEven.m_InputLocation  = PlanInputLocation::Sram;
                paramsEven.m_OutputLocation = PlanOutputLocation::Sram;
                paramsEven.m_All            = [](const PlanDesc& desc) {
                    // No splits are performed, Ple's InputStripe should be equal to InputTensor dimension.
                    TensorShape InputTensor       = { 1, 128, 128, 64 };
                    uint32_t PleInputStripeHeight = desc.m_Ple->m_InputStripeShapes[0][1];
                    uint32_t PleInputStripeWidth  = desc.m_Ple->m_InputStripeShapes[0][2];
                    uint32_t PleInputStripeDepth  = desc.m_Ple->m_InputStripeShapes[0][3];
                    CHECK(PleInputStripeHeight == InputTensor[1]);
                    CHECK(PleInputStripeWidth == InputTensor[2]);
                    CHECK(PleInputStripeDepth == InputTensor[3]);

                    // No splits are performed, Ple's OutputStripe should be equal to OutputTensor dimension.
                    TensorShape OutputTensor       = { 1, 64, 64, 64 };
                    uint32_t PleOutputStripeHeight = desc.m_Ple->m_OutputStripeShape[1];
                    uint32_t PleOutputStripeWidth  = desc.m_Ple->m_OutputStripeShape[2];
                    uint32_t PleOutputStripeDepth  = desc.m_Ple->m_OutputStripeShape[3];
                    CHECK(PleOutputStripeHeight == OutputTensor[1]);
                    CHECK(PleOutputStripeWidth == OutputTensor[2]);
                    CHECK(PleOutputStripeDepth == OutputTensor[3]);
                };
                paramsEven.m_Any.push_back([](const PlanDesc& plan) {
                    return (plan.m_InputSram->Sram()->m_NumStripes == 1) &&
                           (plan.m_OutputSram->Sram()->m_NumStripes == 1);
                });
                CheckPlans(plansEven, paramsEven);

                // End Sram: MaxPoolOdd checks
                paramsOdd.m_InputLocation  = PlanInputLocation::Sram;
                paramsOdd.m_OutputLocation = PlanOutputLocation::Sram;
                paramsOdd.m_All            = [](const PlanDesc& desc) {
                    // No splits are performed, Ple's InputStripe should be equal to InputTensor dimension.
                    CHECK(desc.m_Ple->m_InputStripeShapes[0][1] == 136);
                    CHECK(desc.m_Ple->m_InputStripeShapes[0][2] == 136);
                    CHECK(desc.m_Ple->m_InputStripeShapes[0][3] == 64);

                    // No splits are performed, Ple's OutputStripe should be equal to OutputTensor dimension.
                    TensorShape OutputTensor       = { 1, 64, 64, 64 };
                    uint32_t PleOutputStripeHeight = desc.m_Ple->m_OutputStripeShape[1];
                    uint32_t PleOutputStripeWidth  = desc.m_Ple->m_OutputStripeShape[2];
                    uint32_t PleOutputStripeDepth  = desc.m_Ple->m_OutputStripeShape[3];
                    CHECK(PleOutputStripeHeight == OutputTensor[1]);
                    CHECK(PleOutputStripeWidth == OutputTensor[2]);
                    CHECK(PleOutputStripeDepth == OutputTensor[3]);
                };
                paramsOdd.m_Any.push_back([](const PlanDesc& plan) {
                    return (plan.m_InputSram->Sram()->m_NumStripes == 1) &&
                           (plan.m_OutputSram->Sram()->m_NumStripes == 1);
                });
                CheckPlans(plansOdd, paramsOdd);
            }
        }

        WHEN("Asked to produce End plans with an input buffer in Ple Input Sram")
        {
            std::unique_ptr<PleInputSramBuffer> prevBufferEven = PleInputSramBuffer::Build()
                                                                     .AddFormat(BufferFormat::NHWCB)
                                                                     .AddQuantization({ 0, 1.0f })
                                                                     .AddTensorShape(tsInEven)
                                                                     .AddStripeShape(TensorShape{ 1, 128, 128, 64 })
                                                                     .AddSizeInBytes(1 * 128 * 128 * 64)
                                                                     .AddNumStripes(1);

            Plans plansEven = partEven.GetPlans(CascadeType::End, BlockConfig{ 8U, 8U }, { prevBufferEven.get() }, 1);
            SavePlansToDot(plansEven, "FusedPlePart GetPlans MaxPoolEven End Ple Sram Input");

            std::unique_ptr<PleInputSramBuffer> prevBufferOdd = PleInputSramBuffer::Build()
                                                                    .AddFormat(BufferFormat::NHWCB)
                                                                    .AddQuantization({ 0, 1.0f })
                                                                    .AddTensorShape(tsInOdd)
                                                                    .AddStripeShape(TensorShape{ 1, 136, 136, 64 })
                                                                    .AddSizeInBytes(1 * 136 * 136 * 64)
                                                                    .AddNumStripes(1);

            Plans plansOdd = partOdd.GetPlans(CascadeType::End, BlockConfig{ 8U, 8U }, { prevBufferOdd.get() }, 1);
            SavePlansToDot(plansOdd, "FusedPlePart GetPlans MaxPoolOdd End Ple Sram Input");

            THEN("The plans are valid and start in PleInputSram and end in Sram")
            {
                // End PleSram: MaxPoolEven checks
                paramsEven.m_InputLocation  = PlanInputLocation::PleInputSram;
                paramsEven.m_OutputLocation = PlanOutputLocation::Sram;
                paramsEven.m_All            = [](const PlanDesc& desc) {
                    // No splits are performed, Ple's InputStripe should be equal to InputTensor dimension.
                    TensorShape InputTensor       = { 1, 128, 128, 64 };
                    uint32_t PleInputStripeHeight = desc.m_Ple->m_InputStripeShapes[0][1];
                    uint32_t PleInputStripeWidth  = desc.m_Ple->m_InputStripeShapes[0][2];
                    uint32_t PleInputStripeDepth  = desc.m_Ple->m_InputStripeShapes[0][3];
                    CHECK(PleInputStripeHeight == InputTensor[1]);
                    CHECK(PleInputStripeWidth == InputTensor[2]);
                    CHECK(PleInputStripeDepth == InputTensor[3]);

                    // No splits are performed, Ple's OutputStripe should be equal to OutputTensor dimension.
                    TensorShape OutputTensor       = { 1, 64, 64, 64 };
                    uint32_t PleOutputStripeHeight = desc.m_Ple->m_OutputStripeShape[1];
                    uint32_t PleOutputStripeWidth  = desc.m_Ple->m_OutputStripeShape[2];
                    uint32_t PleOutputStripeDepth  = desc.m_Ple->m_OutputStripeShape[3];
                    CHECK(PleOutputStripeHeight == OutputTensor[1]);
                    CHECK(PleOutputStripeWidth == OutputTensor[2]);
                    CHECK(PleOutputStripeDepth == OutputTensor[3]);
                };
                CheckPlans(plansEven, paramsEven);

                // End PleSram: MaxPoolOdd checks
                paramsOdd.m_InputLocation  = PlanInputLocation::PleInputSram;
                paramsOdd.m_OutputLocation = PlanOutputLocation::Sram;
                paramsOdd.m_All            = [](const PlanDesc& desc) {
                    // No splits are performed, Ple's InputStripe should be equal to InputTensor dimension.
                    CHECK(desc.m_Ple->m_InputStripeShapes[0][1] == 136);
                    CHECK(desc.m_Ple->m_InputStripeShapes[0][2] == 136);
                    CHECK(desc.m_Ple->m_InputStripeShapes[0][3] == 64);

                    // No splits are performed, Ple's OutputStripe should be equal to OutputTensor dimension.
                    TensorShape OutputTensor       = { 1, 64, 64, 64 };
                    uint32_t PleOutputStripeHeight = desc.m_Ple->m_OutputStripeShape[1];
                    uint32_t PleOutputStripeWidth  = desc.m_Ple->m_OutputStripeShape[2];
                    uint32_t PleOutputStripeDepth  = desc.m_Ple->m_OutputStripeShape[3];
                    CHECK(PleOutputStripeHeight == OutputTensor[1]);
                    CHECK(PleOutputStripeWidth == OutputTensor[2]);
                    CHECK(PleOutputStripeDepth == OutputTensor[3]);
                };
                CheckPlans(plansOdd, paramsOdd);
            }
        }
    }
}

/// Checks that FusedPleParts::GetPlans returns a valid plan for strategy 0 with a non identity shape multiplier
TEST_CASE("FusedPlePart GetPlans strategy 0 shape multiplier")
{
    GIVEN("An FusedPlePart for a passthrough")
    {
        const CompilationOptions compOpts;
        const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
        const EstimationOptions estOpts;
        DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
        ThreadPool threadPool(0);

        TensorShape inputShape{ 1, 32, 16, 16 };
        TensorShape outputShape{ 1, 16, 8, 64 };
        PleOperation pleOp = PleOperation::INTERLEAVE_2X2_2_2;

        FusedPlePart part = BuildPart(inputShape, outputShape, pleOp, utils::ShapeMultiplier{ { 1, 2 }, { 1, 2 }, 4 },
                                      compOpts, caps, estOpts, debuggingContext, threadPool);
        part.SetOutputRequirements({ BoundaryRequirements{} }, { false });

        WHEN("Asked to generate plans at the beginning of a cascade")
        {
            Plans plans = part.GetPlans(CascadeType::Beginning, BlockConfig{}, { nullptr }, 1);

            SavePlansToDot(plans, "FusedPlePart GetPlans strategy 0 shape multiplier");

            THEN("There is a plan generated for strategy 0")
            {
                REQUIRE(plans.size() > 0);
                CheckPlansParams params;
                params.m_InputShape  = inputShape;
                params.m_OutputShape = outputShape;
                params.m_PleOp       = pleOp;
                params.m_Any.push_back([](const PlanDesc& plan) {
                    TensorShape inputStripe{ 1, 16, 16, 16 };
                    uint32_t numInputStripes = 1;
                    TensorShape outputStripe{ 1, 8, 8, 64 };
                    uint32_t numOutputStripes = 1;
                    TensorShape pleInputStripe{ 1, 16, 16, 16 };
                    TensorShape pleOutputComputeStripe{ 1, 8, 8, 64 };
                    return plan.m_InputSram->Sram()->m_StripeShape == inputStripe &&
                           plan.m_InputSram->Sram()->m_NumStripes == numInputStripes &&
                           plan.m_OutputSram->Sram()->m_StripeShape == outputStripe &&
                           plan.m_OutputSram->Sram()->m_NumStripes == numOutputStripes &&
                           plan.m_PleInputSram->PleInputSram()->m_StripeShape == pleInputStripe &&
                           plan.m_Ple->m_OutputStripeShape == pleOutputComputeStripe;
                });
                CheckPlans(plans, params);
            }
        }
    }
}

/// Checks that FusedPlePart::GetPlans returns 0 zero plans when called with a previous buffer
/// that is invalid
TEST_CASE("FusedPlePart GetPlans invalid previous buffer")
{
    GIVEN("An FusedPlePart for a Leaky Relu")
    {
        const CompilationOptions compOpts;
        const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);
        const EstimationOptions estOpts;
        DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
        ThreadPool threadPool(0);

        TensorShape inputShape{ 1, 32, 16, 16 };
        TensorShape outputShape{ 1, 32, 16, 16 };
        PleOperation pleOp = PleOperation::LEAKY_RELU;

        FusedPlePart part =
            BuildPart(inputShape, outputShape, pleOp, compOpts, caps, estOpts, debuggingContext, threadPool);
        part.SetOutputRequirements({ BoundaryRequirements{} }, { false });

        WHEN("Asked to generate plans with the number of input stripes > 1")
        {
            BlockConfig blockConfig                = { 8u, 8u };
            std::unique_ptr<SramBuffer> prevBuffer = SramBuffer::Build()
                                                         .AddFormat(BufferFormat::NHWCB)
                                                         .AddQuantization({ 0, 1.0f })
                                                         .AddTensorShape(inputShape)
                                                         .AddStripeShape(TensorShape{ 1, 8, 16, 16 })
                                                         .AddTraversalOrder(TraversalOrder::Xyz)
                                                         .AddSlotSize(1 * 8 * 16 * 16)
                                                         .AddNumStripes(2);

            Plans plans = part.GetPlans(CascadeType::Middle, blockConfig, { prevBuffer.get() }, 1);

            SavePlansToDot(plans, "FusedPlePart GetPlans Filters Sram buffer");

            THEN("There are zero plans generated")
            {
                REQUIRE(plans.size() == 0);
            }
        }
    }
}

/// Checks that FusedPlePart::GetPlans for lonely plans does generate height / width splitting plans
/// as other plans do not fit in sram
TEST_CASE("FusedPlePart GetPlans lonely height and width splits")
{
    GIVEN("An FusedPlePart for a Leaky Relu")
    {
        const CompilationOptions compOpts;
        const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_1TOPS_4PLE_RATIO);
        const EstimationOptions estOpts;
        DebuggingContext debuggingContext(CompilationOptions::DebugInfo{});
        ThreadPool threadPool(0);

        TensorShape inputShape{ 1, 500, 500, 100 };
        TensorShape outputShape{ 1, 500, 500, 100 };
        PleOperation pleOp = PleOperation::LEAKY_RELU;

        FusedPlePart part =
            BuildPart(inputShape, outputShape, pleOp, compOpts, caps, estOpts, debuggingContext, threadPool);
        part.SetOutputRequirements({ BoundaryRequirements{} }, { false });

        WHEN("Asked to generate plans")
        {
            BlockConfig blockConfig = { 8u, 8u };
            Plans plans             = part.GetPlans(CascadeType::Lonely, blockConfig, { nullptr }, 1);

            SavePlansToDot(plans, "FusedPlePart GetPlans Filters Sram buffer");

            THEN("There are plans with split height and width generated")
            {
                CheckPlansParams params;
                params.m_Any.push_back([&](const PlanDesc& desc) {
                    bool splitHeight = desc.m_Input->Sram()->m_StripeShape[1] < inputShape[1];
                    bool splitWidth  = desc.m_Input->Sram()->m_StripeShape[2] < inputShape[2];
                    bool splitBoth   = splitHeight && splitWidth;
                    return splitBoth;
                });

                CheckPlans(plans, params);
            }
        }
    }
}
