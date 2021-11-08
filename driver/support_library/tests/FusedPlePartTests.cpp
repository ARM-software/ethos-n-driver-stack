//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "CapabilitiesInternal.hpp"
#include "GlobalParameters.hpp"
#include "TestUtils.hpp"
#include "Utils.hpp"
#include "cascading/Cascading.hpp"
#include "cascading/FusedPlePart.hpp"
#include "cascading/Visualisation.hpp"
#include "cascading/WeightEncoderCache.hpp"
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
                       command_stream::PleOperation op,
                       utils::ShapeMultiplier shapeMultiplier,
                       const HardwareCapabilities& caps)
{
    const PartId partId = 0;
    const QuantizationInfo inputQuantInfo(0, 1.0f);
    const QuantizationInfo outputQuantInfo(0, 1.0f);
    const std::set<uint32_t> operationsIds = { 1 };
    CompilationOptions compOpts;
    EstimationOptions estOps;
    FusedPlePart part(partId, inputShape, outputShape, inputQuantInfo, outputQuantInfo, op, shapeMultiplier, estOps,
                      compOpts, caps, operationsIds);

    return part;
}

FusedPlePart BuildPart(TensorShape inputShape,
                       TensorShape outputShape,
                       command_stream::PleOperation op,
                       const HardwareCapabilities& caps)
{
    return BuildPart(inputShape, outputShape, op, utils::ShapeMultiplier{ 1, 1, 1 }, caps);
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
    utils::Optional<command_stream::PleOperation> m_PleOp;
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
        CHECK(desc.m_InputDram->m_Lifetime == Lifetime::Cascade);
        CHECK(desc.m_InputDram->m_Format == CascadingBufferFormat::NHWCB);
        if (params.m_InputQuantInfo)
        {
            CHECK(desc.m_InputDram->m_QuantizationInfo == params.m_InputQuantInfo.value());
        }
        if (params.m_InputShape)
        {
            CHECK(desc.m_InputDram->m_TensorShape == params.m_InputShape.value());
        }
        CHECK(desc.m_InputDram->m_StripeShape == TensorShape{ 0, 0, 0, 0 });
        CHECK(desc.m_InputDram->m_Order == TraversalOrder::Xyz);
        CHECK(desc.m_InputDram->m_SizeInBytes == utils::TotalSizeBytesNHWCB(desc.m_InputDram->m_TensorShape));
        CHECK(desc.m_InputDram->m_NumStripes == 0);
        CHECK(desc.m_InputDram->m_EncodedWeights == nullptr);
    }
}

void CheckInputSram(PlanDesc& desc, const CheckPlansParams& params)
{
    // Check properties of Input SRAM buffer
    if (params.m_InputLocation != PlanInputLocation::PleInputSram)
    {
        CHECK(desc.m_InputSram->m_Location == Location::Sram);
        CHECK(desc.m_InputSram->m_Lifetime == Lifetime::Cascade);
        CHECK(desc.m_InputSram->m_Format == CascadingBufferFormat::NHWCB);
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
        CHECK(desc.m_InputSram->m_EncodedWeights == nullptr);
    }
}

void CheckWeightsDram(PlanDesc& desc, const CheckPlansParams& params)
{
    // Check properties of Weights DRAM buffer
    if (params.m_InputLocation != PlanInputLocation::PleInputSram)
    {
        CHECK(desc.m_WeightsDram->m_Location == Location::Dram);
        CHECK(desc.m_WeightsDram->m_Lifetime == Lifetime::Cascade);
        CHECK(desc.m_WeightsDram->m_Format == CascadingBufferFormat::WEIGHT);
        CHECK(desc.m_WeightsDram->m_QuantizationInfo == QuantizationInfo{ 0, 0.5f });
        CHECK(desc.m_WeightsDram->m_TensorShape == TensorShape{ 1, 1, desc.m_Input->m_TensorShape[3], 1 });
        CHECK(desc.m_WeightsDram->m_StripeShape == TensorShape{ 0, 0, 0, 0 });
        CHECK(desc.m_WeightsDram->m_Order == TraversalOrder::Xyz);
        CHECK(desc.m_WeightsDram->m_NumStripes == 0);
        REQUIRE(desc.m_WeightsDram->m_EncodedWeights != nullptr);
        CHECK(desc.m_WeightsDram->m_EncodedWeights->m_Data.size() > 0);
        CHECK(desc.m_WeightsDram->m_SizeInBytes == desc.m_WeightsDram->m_EncodedWeights->m_Data.size());
    }
}

void CheckWeightsSram(PlanDesc& desc, const CheckPlansParams& params)
{
    // Check properties of Weights SRAM buffer
    if (params.m_InputLocation != PlanInputLocation::PleInputSram)
    {
        CHECK(desc.m_WeightsSram->m_Location == Location::Sram);
        CHECK(desc.m_WeightsSram->m_Lifetime == Lifetime::Cascade);
        CHECK(desc.m_WeightsSram->m_Format == CascadingBufferFormat::WEIGHT);
        CHECK(desc.m_WeightsSram->m_QuantizationInfo == QuantizationInfo{ 0, 0.5f });
        CHECK(desc.m_WeightsSram->m_TensorShape == TensorShape{ 1, 1, desc.m_Input->m_TensorShape[3], 1 });
        // m_StripeShape, m_Order, m_NumStripes will depend on the streaming strategy, and so cannot be checked generically
        CHECK(desc.m_WeightsSram->m_SizeInBytes ==
              desc.m_WeightsDram->m_EncodedWeights->m_MaxSize * desc.m_WeightsSram->m_NumStripes);
        CHECK(desc.m_WeightsSram->m_EncodedWeights == nullptr);
    }
}

void CheckPleInputSram(PlanDesc& desc, const CheckPlansParams& params)
{
    // Check properties of Ple Input SRAM buffer
    CHECK(desc.m_PleInputSram->m_Location == Location::PleInputSram);
    CHECK(desc.m_PleInputSram->m_Lifetime == Lifetime::Cascade);
    CHECK(desc.m_PleInputSram->m_Format == CascadingBufferFormat::NHWCB);
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
    CHECK(desc.m_PleInputSram->m_EncodedWeights == nullptr);
}

void CheckOutputSram(PlanDesc& desc, const CheckPlansParams& params)
{
    // Check properties of Output SRAM buffer (if we have one)
    if (desc.m_OutputSram)
    {
        CHECK(desc.m_OutputSram->m_Location == Location::Sram);
        CHECK(desc.m_OutputSram->m_Lifetime == Lifetime::Cascade);
        CHECK(desc.m_OutputSram->m_Format == CascadingBufferFormat::NHWCB);
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
        CHECK(desc.m_OutputSram->m_EncodedWeights == nullptr);
    }
}

void CheckOutputDram(PlanDesc& desc, const CheckPlansParams& params)
{
    // Check properties of Output DRAM buffer (if we have one)
    if (desc.m_OutputDram)
    {
        CHECK(desc.m_OutputDram->m_Location == Location::Dram);
        CHECK(desc.m_OutputDram->m_Lifetime == Lifetime::Cascade);
        CHECK(desc.m_OutputDram->m_Format == CascadingBufferFormat::NHWCB);
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
        CHECK(desc.m_OutputDram->m_StripeShape == TensorShape{ 0, 0, 0, 0 });
        CHECK(desc.m_OutputDram->m_Order == TraversalOrder::Xyz);
        CHECK(desc.m_OutputDram->m_SizeInBytes == utils::TotalSizeBytesNHWCB(desc.m_OutputDram->m_TensorShape));
        CHECK(desc.m_OutputDram->m_NumStripes == 0);
        CHECK(desc.m_OutputDram->m_EncodedWeights == nullptr);
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
        CHECK(desc.m_InputDma->m_Lifetime == Lifetime::Cascade);
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
        CHECK(desc.m_WeightsDma->m_Lifetime == Lifetime::Cascade);
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
        CHECK(desc.m_Mce->m_Lifetime == Lifetime::Cascade);
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
        CHECK(desc.m_Ple->m_Lifetime == Lifetime::Cascade);
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
        CHECK(desc.m_OutputDma->m_Lifetime == Lifetime::Cascade);
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
        CHECK(plan.m_OpGraph.GetProducer(desc.m_InputDram) == nullptr);
        CHECK(plan.m_OpGraph.GetConsumers(desc.m_InputDram) ==
              std::vector<std::pair<Op*, uint32_t>>{ { desc.m_InputDma, 0 } });
    }
    if (params.m_InputLocation != PlanInputLocation::PleInputSram)
    {
        CHECK(plan.m_OpGraph.GetProducer(desc.m_InputSram) ==
              (params.m_InputLocation == PlanInputLocation::Dram ? desc.m_InputDma : nullptr));
        CHECK(plan.m_OpGraph.GetConsumers(desc.m_InputSram) ==
              std::vector<std::pair<Op*, uint32_t>>{ { desc.m_Mce, 0 } });

        CHECK(plan.m_OpGraph.GetProducer(desc.m_WeightsDram) == nullptr);
        CHECK(plan.m_OpGraph.GetConsumers(desc.m_WeightsDram) ==
              std::vector<std::pair<Op*, uint32_t>>{ { desc.m_WeightsDma, 0 } });

        CHECK(plan.m_OpGraph.GetProducer(desc.m_WeightsSram) == desc.m_WeightsDma);
        CHECK(plan.m_OpGraph.GetConsumers(desc.m_WeightsSram) ==
              std::vector<std::pair<Op*, uint32_t>>{ { desc.m_Mce, 1 } });
    }

    CHECK(plan.m_OpGraph.GetProducer(desc.m_PleInputSram) ==
          (params.m_InputLocation == PlanInputLocation::PleInputSram ? nullptr : desc.m_Mce));
    CHECK(plan.m_OpGraph.GetConsumers(desc.m_PleInputSram) ==
          std::vector<std::pair<Op*, uint32_t>>{ { desc.m_Ple, 0 } });

    CHECK(plan.m_OpGraph.GetProducer(desc.m_OutputSram) == desc.m_Ple);
    CHECK(plan.m_OpGraph.GetConsumers(desc.m_OutputSram) ==
          (desc.m_OutputDma ? std::vector<std::pair<Op*, uint32_t>>{ { desc.m_OutputDma, 0 } }
                            : std::vector<std::pair<Op*, uint32_t>>{}));
    if (desc.m_OutputDram)
    {
        CHECK(plan.m_OpGraph.GetProducer(desc.m_OutputDram) == desc.m_OutputDma);
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
        INFO("plan " << plan->m_DebugTag);
        PlanDesc desc;

        ExtractBuffers(*plan, desc, params);
        CheckInputDram(params, desc);
        CheckInputSram(desc, params);
        CheckWeightsDram(desc, params);
        CheckWeightsSram(desc, params);
        CheckPleInputSram(desc, params);
        CheckOutputSram(desc, params);
        CheckOutputDram(desc, params);

        ExtractOps(*plan, params, desc);
        CheckInputDma(params, desc);
        CheckWeightsDma(params, desc);
        CheckMce(params, desc);
        CheckPle(desc, params);
        CheckOutputDma(desc, params);
        CheckConnections(params, *plan, desc);
        CheckMappings(params, *plan, desc);

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

        const PartId partId = 0;
        TensorShape tsIn    = { 1, 32, 32, 3 };
        TensorShape tsOut   = { 1, 64, 64, 1 };
        const QuantizationInfo inputQuantInfo(0, 1.0f);
        const QuantizationInfo outputQuantInfo(0, 1.0f);
        const std::set<uint32_t> operationIds   = { 1, 2, 3 };
        const command_stream::PleOperation csOp = command_stream::PleOperation::PASSTHROUGH;
        const utils::ShapeMultiplier shapeMult  = { 1, 1, 1 };
        FusedPlePart part(partId, tsIn, tsOut, inputQuantInfo, outputQuantInfo, csOp, shapeMult, estOpts, compOpt, caps,
                          operationIds);

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
            Plans plans = part.GetPlans(CascadeType::Lonely, command_stream::BlockConfig{}, nullptr, 0);
            SavePlansToDot(plans, "FusedPlePart GetPlans structure Lonely");

            THEN("The plans are valid, start and end in Sram")
            {
                params.m_InputLocation  = PlanInputLocation::Sram;
                params.m_OutputLocation = PlanOutputLocation::Sram;
                CheckPlans(plans, params);
            }
        }

        WHEN("Asked to produce Beginning plans")
        {
            Plans plans = part.GetPlans(CascadeType::Beginning, command_stream::BlockConfig{}, nullptr, 0);
            SavePlansToDot(plans, "FusedPlePart GetPlans structure Beginning");

            THEN("The plans are valid and start in Sram and end in Sram")
            {
                params.m_InputLocation  = PlanInputLocation::Sram;
                params.m_OutputLocation = PlanOutputLocation::Sram;
                CheckPlans(plans, params);
            }
        }

        WHEN("Asked to produce Middle plans with an input buffer in sram")
        {
            Buffer prevBuffer;
            prevBuffer.m_Lifetime         = Lifetime::Cascade;
            prevBuffer.m_Location         = Location::Sram;
            prevBuffer.m_Format           = CascadingBufferFormat::NHWCB;
            prevBuffer.m_QuantizationInfo = { 0, 1.0f };
            prevBuffer.m_TensorShape      = tsIn;
            prevBuffer.m_StripeShape      = TensorShape{ 1, 8, 16, 16 };
            prevBuffer.m_Order            = TraversalOrder::Xyz;
            prevBuffer.m_SizeInBytes      = 8 * 16 * 16 * 1;
            prevBuffer.m_NumStripes       = 1;

            Plans plans = part.GetPlans(CascadeType::Middle, command_stream::BlockConfig{ 16u, 16u }, &prevBuffer, 1);
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
            Buffer prevBuffer;
            prevBuffer.m_Lifetime         = Lifetime::Cascade;
            prevBuffer.m_Location         = Location::PleInputSram;
            prevBuffer.m_Format           = CascadingBufferFormat::NHWCB;
            prevBuffer.m_QuantizationInfo = { 0, 1.0f };
            prevBuffer.m_TensorShape      = tsIn;
            prevBuffer.m_StripeShape      = TensorShape{ 1, 8, 16, 16 };
            prevBuffer.m_Order            = TraversalOrder::Xyz;
            prevBuffer.m_SizeInBytes      = 8 * 16 * 16 * 1;
            prevBuffer.m_NumStripes       = 1;

            Plans plans = part.GetPlans(CascadeType::Middle, command_stream::BlockConfig{ 16u, 16u }, &prevBuffer, 1);
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
            Buffer prevBuffer;
            prevBuffer.m_Lifetime         = Lifetime::Cascade;
            prevBuffer.m_Location         = Location::Sram;
            prevBuffer.m_Format           = CascadingBufferFormat::NHWCB;
            prevBuffer.m_QuantizationInfo = { 0, 1.0f };
            prevBuffer.m_TensorShape      = tsIn;
            prevBuffer.m_StripeShape      = TensorShape{ 1, 8, 16, 16 };
            prevBuffer.m_Order            = TraversalOrder::Xyz;
            prevBuffer.m_SizeInBytes      = 8 * 16 * 16 * 1;
            prevBuffer.m_NumStripes       = 1;

            Plans plans = part.GetPlans(CascadeType::End, command_stream::BlockConfig{ 16u, 16u }, &prevBuffer, 1);
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
            Buffer prevBuffer;
            prevBuffer.m_Lifetime         = Lifetime::Cascade;
            prevBuffer.m_Location         = Location::PleInputSram;
            prevBuffer.m_Format           = CascadingBufferFormat::NHWCB;
            prevBuffer.m_QuantizationInfo = { 0, 1.0f };
            prevBuffer.m_TensorShape      = tsIn;
            prevBuffer.m_StripeShape      = TensorShape{ 1, 8, 16, 16 };
            prevBuffer.m_Order            = TraversalOrder::Xyz;
            prevBuffer.m_SizeInBytes      = 8 * 16 * 16 * 1;
            prevBuffer.m_NumStripes       = 1;

            Plans plans = part.GetPlans(CascadeType::End, command_stream::BlockConfig{ 16u, 16u }, &prevBuffer, 1);
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

/// Checks that FusedPleParts::GetPlans returns a valid plan for strategy 0 with a non identity shape multiplier
TEST_CASE("FusedPlePart GetPlans strategy 0 shape multiplier")
{
    GIVEN("An FusedPlePart for a passthrough")
    {
        const CompilationOptions compOpt;
        const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);

        TensorShape inputShape{ 1, 16, 16, 16 };
        TensorShape outputShape{ 1, 8, 8, 64 };
        command_stream::PleOperation pleOp = command_stream::PleOperation::INTERLEAVE_2X2_2_2;

        FusedPlePart part =
            BuildPart(inputShape, outputShape, pleOp, utils::ShapeMultiplier{ { 1, 2 }, { 1, 2 }, 4 }, caps);

        WHEN("Asked to generate plans at the beginning of a cascade")
        {
            Plans plans = part.GetPlans(CascadeType::Beginning, command_stream::BlockConfig{}, nullptr, 0);

            SavePlansToDot(plans, "FusedPlePart GetPlans strategy 0 shape multiplier");

            THEN("There is a plan generated for strategy 0")
            {
                REQUIRE(plans.size() > 0);
                CheckPlansParams params;
                params.m_InputShape  = inputShape;
                params.m_OutputShape = outputShape;
                params.m_PleOp       = pleOp;
                params.m_Any.push_back([](const PlanDesc& plan) {
                    bool blockConfig = plan.m_Ple->m_BlockConfig == command_stream::BlockConfig{ 16u, 8u };
                    TensorShape inputStripe{ 1, 8, 16, 16 };
                    uint32_t numInputStripes = 1;
                    TensorShape outputStripe{ 1, 8, 8, 64 };
                    uint32_t numOutputStripes = 1;
                    TensorShape pleInputStripe{ 1, 8, 16, 16 };
                    TensorShape pleOutputComputeStripe{ 1, 4, 8, 64 };
                    return blockConfig && plan.m_InputSram->m_StripeShape == inputStripe &&
                           plan.m_InputSram->m_NumStripes == numInputStripes &&
                           plan.m_OutputSram->m_StripeShape == outputStripe &&
                           plan.m_OutputSram->m_NumStripes == numOutputStripes &&
                           plan.m_PleInputSram->m_StripeShape == pleInputStripe &&
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
        const CompilationOptions compOpt;
        const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);

        TensorShape inputShape{ 1, 32, 16, 16 };
        TensorShape outputShape{ 1, 32, 16, 16 };
        command_stream::PleOperation pleOp = command_stream::PleOperation::LEAKY_RELU;

        FusedPlePart part = BuildPart(inputShape, outputShape, pleOp, caps);

        WHEN("Asked to generate plans with the number of input stripes > 1")
        {
            command_stream::BlockConfig blockConfig = { 8u, 8u };
            Buffer prevBuffer;
            prevBuffer.m_Lifetime         = Lifetime::Cascade;
            prevBuffer.m_Location         = Location::Sram;
            prevBuffer.m_Format           = CascadingBufferFormat::NHWCB;
            prevBuffer.m_QuantizationInfo = { 0, 1.0f };
            prevBuffer.m_TensorShape      = inputShape;
            prevBuffer.m_StripeShape      = TensorShape{ 1, 8, 16, 16 };
            prevBuffer.m_Order            = TraversalOrder::Xyz;
            prevBuffer.m_SizeInBytes      = 8 * 16 * 16 * 2;
            prevBuffer.m_NumStripes       = 2;
            Plans plans                   = part.GetPlans(CascadeType::Middle, blockConfig, &prevBuffer, 1);

            SavePlansToDot(plans, "FusedPlePart GetPlans Filters Sram buffer");

            THEN("There are zero plans generated")
            {
                REQUIRE(plans.size() == 0);
            }
        }
    }
}

/// Checks that FusedPlePart produces at least the plans that we need for cascading MobileNet V1 in the same way that
/// the prototype compiler does.
///
/// MobileNet V1 Parts are as follows:
///  0. InputPart 224,224,3
///  1. FusedPlePart INTERLEAVE 224,224,3 -> 112,112,3*(num srams) + 3
///  2. McePart CONVOLUTION 112,112,3*(num srams) + 3 -> 112,112,32. Stride 2x2. Padding 1,1. Weights 3,3,3,32.
///  3. McePart DEPTHWISE_CONVOLUTION 112,112,32 -> 112,112,32. Stride 1x1. Padding 1,1. Weights 3,3,32,1.
///  4. McePart CONVOLUTION 112,112,32 -> 112,112,64. Stride 1x1. Padding 0,0. Weights 1,1,32,64.
///  5. FusedPlePart INTERLEAVE 112,112,64 -> 56,56,256
///  6. McePart DEPTHWISE_CONVOLUTION 56,56,256 -> 56,56,64. Stride 2x2. Padding 1,1. Weights 3,3,64,1.
///  7. McePart CONVOLUTION 56,56,64 -> 56,56,128. Stride 1x1. Padding 0,0. Weights 1,1,64,128.
///  8. McePart DEPTHWISE_CONVOLUTION 56,56,128 -> 56,56,128. Stride 1x1. Padding 1,1. Weights 3,3,128,1.
///  9. McePart CONVOLUTION 56,56,128 -> 56,56,128. Stride 1x1. Padding 0,0. Weights 1,1,128,128.
///  10. FusedPlePart INTERLEAVE 56,56,128 -> 28,28,512
///  ...
///
/// The MceParts are skipped here, and covered by a corresponding test in McePartTests.cpp.
///
/// For each FusedPlePart in the above list, we create a FusedPlePart object with the corresponding properties and ask it to generate
/// plans, providing the context (prevBuffer etc.) that it would have assuming that everything beforehand was chosen in agreement
/// with the prototype compiler. We then check that at least one of the plans returned is consistent with the prototype
/// compiler result. This gives us confidence that FusedPlePart is capable of generating the right plans in order to give an overall
/// performance consistent with the prototype compiler.
///
/// We don't cover every Part in the whole Network as that would be a lot of test code and would also be a lot of duplication.
TEST_CASE("FusedPlePart GetPlans MobileNet V1")
{
    const CompilationOptions compOpt;
    SECTION("8TOPS_2PLE_RATIO")
    {
        // Choose the largest variant in order to have the most cascading. In this case, all Parts can be cascaded into a single 'strategy 1' section.
        const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_8TOPS_2PLE_RATIO);

        // Define common properties of the "prevBuffer", which will be the case for all Parts we're testing. This avoids
        // duplicating these lines for each Part being tested.
        Buffer prevBuffer;
        prevBuffer.m_Lifetime         = Lifetime::Cascade;
        prevBuffer.m_Format           = CascadingBufferFormat::NHWCB;
        prevBuffer.m_QuantizationInfo = { 0, 1.0f };
        prevBuffer.m_Order            = TraversalOrder::Xyz;

        // Notes:
        // - The output buffer in SRAM will always have the full stripe shape and a single stripe,
        //   because this is always the case for strategy 1 cascading. This may be different from the MCE output stripe shape,
        //   because the MCE still computes the data in multiple stripes, it's just stored in SRAM in a layout which is
        //   consistent with a single full stripe.
        // - For the configuration we have chosen (ETHOS_N78_8TOPS_2PLE_RATIO), there are 32 OGs and so the OFM stripe depths
        //   and weight stripe depths are generally going to be 32.

        ///  1. FusedPlePart INTERLEAVE 224,224,3 -> 112,112,51
        SECTION("Part 1")
        {
            // Even though this is strategy 1, the variant we are compiling for has 32 OGs and so there is no actual splitting
            // and this is equivalent to strategy 3.
            // This Part doesn't exist in the prototype compiler because it assumes that the DMA is used to do the interleaving
            // of the IFM, rather than a separate Part as we are doing.
            // This is the start of a cascade
            // It won't be fused with a preceding McePart, so the plan we expect here will startin SRAM and contain an identity MCE.
            TensorShape inputShape{ 1, 224, 224, 3 };
            TensorShape outputShape{ 1, 112, 112, 51 };
            FusedPlePart part = BuildPart(inputShape, outputShape, command_stream::PleOperation::INTERLEAVE_2X2_2_2,
                                          utils::ShapeMultiplier{ { 1, 2 }, { 1, 2 }, 4 }, caps);

            Plans plans = part.GetPlans(CascadeType::Beginning, command_stream::BlockConfig{}, nullptr, 0);
            CheckPlansParams params;

            params.m_Any.push_back([&](const PlanDesc& plan) {
                return plan.m_InputSram->m_StripeShape == TensorShape{ 1, 224, 224, 16 } &&
                       plan.m_InputSram->m_NumStripes == 1 &&
                       plan.m_WeightsSram->m_StripeShape == TensorShape{ 1, 1, 16, 1 } &&    // Identity depthwise
                       plan.m_WeightsSram->m_NumStripes == 1 &&
                       plan.m_Mce->m_InputStripeShape == TensorShape{ 1, 224, 224, 16 } &&
                       plan.m_Mce->m_WeightsStripeShape == TensorShape{ 1, 1, 16, 1 } &&
                       plan.m_Mce->m_OutputStripeShape == TensorShape{ 1, 224, 224, 32 } &&
                       plan.m_Ple->m_InputStripeShapes == std::vector<TensorShape>{ { 1, 224, 224, 32 } } &&
                       plan.m_Ple->m_OutputStripeShape == TensorShape{ 1, 112, 112, 64 } &&
                       plan.m_OutputSram->m_StripeShape == TensorShape{ 1, 112, 112, 64 } &&
                       plan.m_OutputSram->m_NumStripes == 1;
            });
            CheckPlans(plans, params);
        }

        ///  5. FusedPlePart INTERLEAVE 112,112,64 -> 56,56,256
        SECTION("Part 5")
        {
            TensorShape inputShape{ 1, 112, 112, 64 };
            TensorShape outputShape{ 1, 56, 56, 256 };
            FusedPlePart part     = BuildPart(inputShape, outputShape, command_stream::PleOperation::INTERLEAVE_2X2_2_2,
                                          utils::ShapeMultiplier{ { 1, 2 }, { 1, 2 }, 4 }, caps);
            prevBuffer.m_Location = Location::PleInputSram;    // The previous Part is an McePart, so we will be fused
            prevBuffer.m_TensorShape = inputShape;
            prevBuffer.m_StripeShape = TensorShape{ 1, 112, 112, 32 };    // Depth splitting
            prevBuffer.m_SizeInBytes = 0;    // PleInputSram buffers have no size or num stripes
            prevBuffer.m_NumStripes  = 0;

            Plans plans = part.GetPlans(CascadeType::Middle, command_stream::BlockConfig{ 16u, 16u }, &prevBuffer, 2);
            CheckPlansParams params;
            params.m_InputLocation = PlanInputLocation::PleInputSram;
            params.m_Any.push_back([&](const PlanDesc& plan) {
                return plan.m_Input->m_Location == Location::PleInputSram &&
                       plan.m_Input->m_StripeShape == TensorShape{ 1, 112, 112, 32 } &&
                       plan.m_Input->m_NumStripes == 0 &&    // PleInputSram buffers have no num stripes
                       plan.m_Ple->m_InputStripeShapes == std::vector<TensorShape>{ { 1, 112, 112, 32 } } &&
                       plan.m_Ple->m_OutputStripeShape == TensorShape{ 1, 56, 56, 128 } &&    // 32*4
                       plan.m_OutputSram->m_StripeShape == TensorShape{ 1, 56, 56, 256 } &&
                       plan.m_OutputSram->m_NumStripes == 1;
            });
            CheckPlans(plans, params);
        }

        ///  10. FusedPlePart INTERLEAVE 56,56,128 -> 28,28,512
        SECTION("Part 10")
        {
            TensorShape inputShape{ 1, 56, 56, 128 };
            TensorShape outputShape{ 1, 28, 28, 512 };
            FusedPlePart part     = BuildPart(inputShape, outputShape, command_stream::PleOperation::INTERLEAVE_2X2_2_2,
                                          utils::ShapeMultiplier{ { 1, 2 }, { 1, 2 }, 4 }, caps);
            prevBuffer.m_Location = Location::PleInputSram;    // The previous Part is an McePart, so we will be fused
            prevBuffer.m_TensorShape = inputShape;
            prevBuffer.m_StripeShape = TensorShape{ 1, 56, 56, 32 };    // Depth splitting
            prevBuffer.m_SizeInBytes = 0;    // PleInputSram buffers have no size or num stripes
            prevBuffer.m_NumStripes  = 0;

            Plans plans = part.GetPlans(CascadeType::Middle, command_stream::BlockConfig{ 16u, 16u }, &prevBuffer, 2);
            CHECK(plans.size() == 1);
            CheckPlansParams params;
            params.m_InputLocation = PlanInputLocation::PleInputSram;
            params.m_Any.push_back([&](const PlanDesc& plan) {
                return plan.m_Input->m_Location == Location::PleInputSram &&
                       plan.m_Input->m_StripeShape == TensorShape{ 1, 56, 56, 32 } &&
                       plan.m_Input->m_NumStripes == 0 &&    // PleInputSram buffers have no num stripes
                       plan.m_Ple->m_InputStripeShapes == std::vector<TensorShape>{ { 1, 56, 56, 32 } } &&
                       plan.m_Ple->m_OutputStripeShape == TensorShape{ 1, 28, 28, 128 } &&    // 32*4
                       plan.m_OutputSram->m_StripeShape == TensorShape{ 1, 32, 32, 512 } &&
                       plan.m_OutputSram->m_NumStripes == 1;
            });
            CheckPlans(plans, params);
        }
    }

    SECTION("1TOPS_2PLE_RATIO")
    {
        // Choose the smallest variant in order to have the most cascades. In this case there is a combination of single
        // layer cascades (Lonely parts) as well as some longer cascades.
        const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO);

        // Define common properties of the "prevBuffer", which will be the case for all Parts we're testing. This avoids
        // duplicating these lines for each Part being tested.
        Buffer prevBuffer;
        prevBuffer.m_Lifetime         = Lifetime::Cascade;
        prevBuffer.m_Format           = CascadingBufferFormat::NHWCB;
        prevBuffer.m_QuantizationInfo = { 0, 1.0f };
        prevBuffer.m_Order            = TraversalOrder::Xyz;

        ///  1. FusedPlePart INTERLEAVE 224,224,3 -> 112,112,27
        SECTION("Part 1")
        {
            // This Part doesn't exist in the prototype compiler because it assumes that the DMA is used to do the interleaving
            // of the IFM, rather than a separate Part as we are doing.
            // This is the start of a cascade
            // It won't be fused with a preceding McePart, so the plan we expect here will start in SRAM and contain an identity MCE.
            // This is part of a strategy 0 cascade.
            TensorShape inputShape{ 1, 224, 224, 3 };
            TensorShape outputShape{ 1, 112, 112, 27 };
            FusedPlePart part = BuildPart(inputShape, outputShape, command_stream::PleOperation::INTERLEAVE_2X2_2_2,
                                          utils::ShapeMultiplier{ { 1, 2 }, { 1, 2 }, 4 }, caps);

            Plans plans = part.GetPlans(CascadeType::Beginning, command_stream::BlockConfig{}, nullptr, 0);
            CheckPlansParams params;
            params.m_Any.push_back([&](const PlanDesc& plan) {
                return plan.m_InputSram->m_StripeShape == TensorShape{ 1, 8, 224, 16 } &&
                       // The 1x1 convolution doesn't need neighbouring data but we double buffer.
                       plan.m_InputSram->m_NumStripes == 2 &&
                       plan.m_WeightsSram->m_StripeShape == TensorShape{ 1, 1, 16, 1 } &&    // Identity depthwise
                       plan.m_WeightsSram->m_NumStripes == 1 &&
                       plan.m_Mce->m_InputStripeShape == TensorShape{ 1, 8, 224, 16 } &&
                       plan.m_Mce->m_WeightsStripeShape == TensorShape{ 1, 1, 16, 1 } &&
                       plan.m_Mce->m_OutputStripeShape == TensorShape{ 1, 8, 224, 8 } &&
                       plan.m_Ple->m_InputStripeShapes == std::vector<TensorShape>{ { 1, 8, 224, 16 } } &&
                       plan.m_Ple->m_OutputStripeShape == TensorShape{ 1, 4, 112, 32 } &&
                       // 2 stripes are accumulated in sram to make up a full brick-group
                       plan.m_OutputSram->m_StripeShape == TensorShape{ 1, 8, 112, 32 } &&
                       plan.m_OutputSram->m_NumStripes ==
                           3;    // Following McePart has kernel of height 3, so needs neighbouring data
            });
            CheckPlans(plans, params);
        }

        ///  5. FusedPlePart INTERLEAVE 112,112,64 -> 56,56,256
        SECTION("Part 5")
        {
            // This is the final Part in strategy 0 cascade

            TensorShape inputShape{ 1, 112, 112, 64 };
            TensorShape outputShape{ 1, 56, 56, 256 };
            FusedPlePart part     = BuildPart(inputShape, outputShape, command_stream::PleOperation::INTERLEAVE_2X2_2_2,
                                          utils::ShapeMultiplier{ { 1, 2 }, { 1, 2 }, 4 }, caps);
            prevBuffer.m_Location = Location::PleInputSram;    // The previous Part is an McePart, so we will be fused
            prevBuffer.m_TensorShape = inputShape;
            prevBuffer.m_StripeShape = TensorShape{ 1, 8, 112, 64 };    // Height splitting
            prevBuffer.m_SizeInBytes = 0;    // PleInputSram buffers have no size or num stripes
            prevBuffer.m_NumStripes  = 0;

            Plans plans = part.GetPlans(CascadeType::End, command_stream::BlockConfig{ 32u, 8u }, &prevBuffer, 1);
            CheckPlansParams params;
            params.m_InputLocation = PlanInputLocation::PleInputSram;
            params.m_Any.push_back([&](const PlanDesc& plan) {
                return plan.m_Input->m_Location == Location::PleInputSram &&
                       plan.m_Input->m_StripeShape == TensorShape{ 1, 8, 112, 64 } &&
                       plan.m_Input->m_NumStripes == 0 &&    // PleInputSram buffers have no num stripes
                       plan.m_Ple->m_InputStripeShapes == std::vector<TensorShape>{ { 1, 8, 112, 64 } } &&
                       plan.m_Ple->m_OutputStripeShape ==
                           TensorShape{
                               1, 4, 56, 256
                           } &&    // PLE accumulates 2 stripes before writing out, to make up a full brick-group
                       plan.m_OutputSram->m_StripeShape == TensorShape{ 1, 8, 56, 256 } &&
                       plan.m_OutputSram->m_NumStripes == 2;    // End of cascade, double buffered
            });
            CheckPlans(plans, params);
        }

        ///  10. FusedPlePart INTERLEAVE 56,56,128 -> 28,28,512
        SECTION("Part 10")
        {
            // Last part in a short strategy 1 cascade

            TensorShape inputShape{ 1, 56, 56, 128 };
            TensorShape outputShape{ 1, 28, 28, 512 };
            FusedPlePart part     = BuildPart(inputShape, outputShape, command_stream::PleOperation::INTERLEAVE_2X2_2_2,
                                          utils::ShapeMultiplier{ { 1, 2 }, { 1, 2 }, 4 }, caps);
            prevBuffer.m_Location = Location::PleInputSram;    // The previous Part is an McePart, so we will be fused
            prevBuffer.m_TensorShape = inputShape;
            prevBuffer.m_StripeShape = TensorShape{ 1, 56, 56, 16 };    // Depth splitting
            prevBuffer.m_SizeInBytes = 0;    // PleInputSram buffers have no size or num stripes
            prevBuffer.m_NumStripes  = 0;

            Plans plans = part.GetPlans(CascadeType::End, command_stream::BlockConfig{ 32u, 8u }, &prevBuffer, 2);
            CheckPlansParams params;
            params.m_InputLocation = PlanInputLocation::PleInputSram;
            params.m_Any.push_back([&](const PlanDesc& plan) {
                return plan.m_Input->m_Location == Location::PleInputSram &&
                       plan.m_Input->m_StripeShape == TensorShape{ 1, 56, 56, 16 } &&
                       plan.m_Input->m_NumStripes == 0 &&    // PleInputSram buffers have no num stripes
                       plan.m_Ple->m_InputStripeShapes == std::vector<TensorShape>{ { 1, 56, 56, 16 } } &&
                       plan.m_Ple->m_OutputStripeShape ==
                           TensorShape{ 1, 28, 28, 64 } &&    // Channels multiplied by 4 for interleave
                       plan.m_OutputSram->m_StripeShape ==
                           TensorShape{
                               1, 32, 32, 16
                           } &&    // Strategy 1 output is not the full tensor because we are at the end of a cascade
                       plan.m_OutputSram->m_NumStripes == 2;
            });
            CheckPlans(plans, params);
        }
    }
}

//TODO: similar tests to above but for VGG16?