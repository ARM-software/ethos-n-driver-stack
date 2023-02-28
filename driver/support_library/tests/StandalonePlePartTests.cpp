//
// Copyright © 2022-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "CapabilitiesInternal.hpp"
#include "GlobalParameters.hpp"
#include "TestUtils.hpp"
#include "Utils.hpp"
#include "cascading/Part.hpp"
#include "cascading/PartUtils.hpp"
#include "cascading/Plan.hpp"
#include "cascading/StandalonePlePart.hpp"
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

StandalonePlePart BuildPart(const std::vector<TensorShape>& inputShapes,
                            const std::vector<QuantizationInfo>& inputQuantizationInfos,
                            TensorShape outputShape,
                            command_stream::PleOperation op,
                            const HardwareCapabilities& caps,
                            const PartId partId,
                            EstimationOptions& estOpts,
                            CompilationOptions& compOpts)
{
    const QuantizationInfo outputQuantInfo(0, 1.0f);
    const std::set<uint32_t> operationsIds = { 1 };

    StandalonePlePart part(partId, inputShapes, outputShape, inputQuantizationInfos, outputQuantInfo, op, estOpts,
                           compOpts, caps, operationsIds, DataType::UINT8_QUANTIZED);

    return part;
}

void CheckPleOperation(const Plan& plan)
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

    CHECK(plan.m_OpGraph.GetSingleProducer(buffers.back()) != nullptr);
    CHECK(plan.m_OpGraph.GetSingleProducer(buffers.back()) == op);
}

void CheckMappings(const CheckPlansParams& params, const Plan& plan)
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

    CHECK(plan.m_OutputMappings.begin()->second.m_PartId == params.m_PartId);
    CHECK(plan.m_OutputMappings.begin()->second.m_OutputIndex == 0);
}

void CheckOutputBuffer(Buffer* buffer, const CheckPlansParams& params)
{
    if (buffer)
    {
        CHECK(buffer->m_Location == Location::Sram);
        CHECK(buffer->m_Format == params.m_DataFormat);
        CHECK(buffer->m_TensorShape == params.m_OutputTensorInfo.m_Dimensions);
        CHECK(buffer->Sram()->m_Order == TraversalOrder::Xyz);
        // Buffer size calculations are non-trivial so we can't check exactly here
        CHECK(buffer->m_SizeInBytes > 0);
        CHECK(buffer->Sram()->m_NumStripes > 0);
    }
}

void CheckInputBuffer(Buffer* buffer, const CheckPlansParams& params, size_t bufIdx)
{
    if (buffer)
    {
        CHECK(buffer->m_Location == Location::Sram);
        CHECK(buffer->m_Format == params.m_DataFormat);
        CHECK(buffer->m_TensorShape == params.m_InputTensorsInfo[bufIdx].m_Dimensions);
        CHECK(buffer->Sram()->m_Order == TraversalOrder::Xyz);
        // Buffer size calculations are non-trivial so we can't check exactly here
        CHECK(buffer->m_SizeInBytes > 0);
        CHECK(buffer->Sram()->m_NumStripes > 0);
    }
}

/// Checks that the given list of Plans matches expectations, based on both generic requirements of all plans (e.g. all plans
/// must follow the expected OpGraph structure) and also specific requirements on plans which can be customized using the provided callbacks.
/// These are all configured by the CheckPlansParams struct.
void CheckPlans(const Plans& plans, const CheckPlansParams& params)
{
    CHECK(plans.size() > 0);

    CHECK(params.m_PartId == 0);

    for (auto&& plan : plans)
    {
        INFO("plan " << plan.m_DebugTag);

        const OpGraph::BufferList& buffers = plan.m_OpGraph.GetBuffers();
        Buffer* outputBuffer               = buffers.back();

        CheckPleOperation(plan);
        CheckOutputBuffer(outputBuffer, params);
        CheckMappings(params, plan);

        for (size_t input = 0; input < (buffers.size() - 1); ++input)
        {
            Buffer* inputBuffer = buffers.at(input);
            CheckInputBuffer(inputBuffer, params, input);
        }
    }
}

}    // namespace

TEST_CASE("StandalonePlePart AVGPOOL_3X3_1_1_UDMA")
{
    SECTION("8TOPS_2PLE_RATIO")
    {
        const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_8TOPS_2PLE_RATIO);

        const PartId partId = 0;

        TensorShape inputShape{ 1, 32, 32, 192 };
        TensorShape outputShape{ 1, 32, 32, 192 };
        const QuantizationInfo inputQuantInfo(0, 1.0f);

        TensorInfo inputTensorsInfo(inputShape, DataType::UINT8_QUANTIZED, DataFormat::NHWCB, inputQuantInfo);
        TensorInfo outputTensorsInfo(outputShape, DataType::UINT8_QUANTIZED, DataFormat::NHWCB,
                                     QuantizationInfo(0, 1.0f));

        CheckPlansParams params;
        params.m_PartId           = partId;
        params.m_InputTensorsInfo = { inputTensorsInfo };
        params.m_OutputTensorInfo = outputTensorsInfo;
        params.m_DataFormat       = CascadingBufferFormat::NHWCB;

        EstimationOptions estOpts;
        CompilationOptions compOpts;
        StandalonePlePart part =
            BuildPart({ inputShape }, { inputQuantInfo }, outputShape,
                      command_stream::PleOperation::AVGPOOL_3X3_1_1_UDMA, caps, partId, estOpts, compOpts);

        // A plan is returned since both input and output tensors is fit into SRAM
        Plans plans0 = part.GetPlans(CascadeType::Beginning, command_stream::BlockConfig{}, nullptr, 1);
        CheckPlans(plans0, params);

        Plans plans1 = part.GetPlans(CascadeType::Lonely, command_stream::BlockConfig{}, nullptr, 1);
        CheckPlans(plans1, params);

        // A plan is returned since both input and output tensors is fit into SRAM
        SramBuffer prevBuffer;
        prevBuffer.m_StripeShape = inputShape;
        Plans plans2             = part.GetPlans(CascadeType::Middle, command_stream::BlockConfig{}, &prevBuffer, 1);
        CheckPlans(plans2, params);

        // No plan is returned since the input tensor and prev buffer's stripe shape does not match
        prevBuffer.m_StripeShape = TensorShape{ 1, 32, 16, 192 };
        Plans plans3             = part.GetPlans(CascadeType::Middle, command_stream::BlockConfig{}, &prevBuffer, 1);
        REQUIRE(plans3.size() == 0);

        // A plan is returned since both input and output tensors is fit into SRAM
        prevBuffer.m_StripeShape = inputShape;
        Plans plans4             = part.GetPlans(CascadeType::End, command_stream::BlockConfig{}, &prevBuffer, 1);
        CheckPlans(plans4, params);
    }

    SECTION("1TOPS_2PLE_RATIO")
    {
        const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO);

        const PartId partId = 0;

        TensorShape inputShape{ 1, 128, 32, 192 };
        TensorShape outputShape{ 1, 128, 32, 192 };
        const QuantizationInfo inputQuantInfo(0, 1.0f);

        TensorInfo inputTensorInfo(inputShape, DataType::UINT8_QUANTIZED, DataFormat::NHWCB, inputQuantInfo);
        TensorInfo outputTensorInfo(outputShape, DataType::UINT8_QUANTIZED, DataFormat::NHWCB,
                                    QuantizationInfo(0, 1.0f));

        EstimationOptions estOpts;
        CompilationOptions compOpts;

        StandalonePlePart part =
            BuildPart({ inputShape }, { inputQuantInfo }, outputShape,
                      command_stream::PleOperation::AVGPOOL_3X3_1_1_UDMA, caps, partId, estOpts, compOpts);

        // The input tensor will not be split to fit into SRAM
        // Therefore only the "lonely" type is expected to return a plan.

        Plans plans0 = part.GetPlans(CascadeType::Beginning, command_stream::BlockConfig{}, nullptr, 1);
        REQUIRE(plans0.size() == 0);

        Plans plans1 = part.GetPlans(CascadeType::Lonely, command_stream::BlockConfig{}, nullptr, 1);
        REQUIRE(plans1.size() == 1);

        CheckPlansParams params;
        params.m_PartId           = partId;
        params.m_InputTensorsInfo = { inputTensorInfo };
        params.m_OutputTensorInfo = outputTensorInfo;
        params.m_DataFormat       = CascadingBufferFormat::NHWCB;

        CheckPlans(plans1, params);

        SramBuffer prevBuffer;
        prevBuffer.m_StripeShape = inputShape;
        Plans plans2             = part.GetPlans(CascadeType::Middle, command_stream::BlockConfig{}, &prevBuffer, 1);
        REQUIRE(plans2.size() == 0);

        prevBuffer.m_StripeShape = inputShape;
        Plans plans4             = part.GetPlans(CascadeType::End, command_stream::BlockConfig{}, &prevBuffer, 1);
        REQUIRE(plans4.size() == 0);
    }
}

TEST_CASE("StandalonePlePart ADDITION")
{
    SECTION("4TOPS_2PLE_RATIO")
    {
        const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_2PLE_RATIO);

        const PartId partId = 0;

        TensorShape inputShape{ 1, 128, 32, 64 };
        TensorShape outputShape{ 1, 128, 32, 64 };
        const QuantizationInfo inputQuantInfo(0, 1.0f);

        TensorInfo inputTensorsInfo(inputShape, DataType::UINT8_QUANTIZED, DataFormat::NHWCB, inputQuantInfo);
        TensorInfo outputTensorsInfo(outputShape, DataType::UINT8_QUANTIZED, DataFormat::NHWCB,
                                     QuantizationInfo(0, 1.0f));

        CheckPlansParams params;
        params.m_PartId           = partId;
        params.m_InputTensorsInfo = { inputTensorsInfo, inputTensorsInfo };
        params.m_OutputTensorInfo = outputTensorsInfo;
        params.m_DataFormat       = CascadingBufferFormat::NHWCB;

        EstimationOptions estOpts;
        CompilationOptions compOpts;

        StandalonePlePart part = BuildPart({ inputShape, inputShape }, { inputQuantInfo, inputQuantInfo }, outputShape,
                                           command_stream::PleOperation::ADDITION, caps, partId, estOpts, compOpts);

        // Only the lonely part is expected to return a plan

        SramBuffer prevBuffer;

        Plans plans0 = part.GetPlans(CascadeType::Beginning, command_stream::BlockConfig{}, nullptr, 1);
        REQUIRE(plans0.size() == 0);

        Plans plans1 = part.GetPlans(CascadeType::Lonely, command_stream::BlockConfig{}, nullptr, 1);
        CheckPlans(plans1, params);

        Plans plans2 = part.GetPlans(CascadeType::Middle, command_stream::BlockConfig{}, &prevBuffer, 1);
        REQUIRE(plans2.size() == 0);

        Plans plans3 = part.GetPlans(CascadeType::Middle, command_stream::BlockConfig{}, &prevBuffer, 1);
        REQUIRE(plans3.size() == 0);

        Plans plans4 = part.GetPlans(CascadeType::End, command_stream::BlockConfig{}, &prevBuffer, 1);
        REQUIRE(plans4.size() == 0);
    }

    SECTION("1TOPS_4PLE_RATIO")
    {
        const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_1TOPS_4PLE_RATIO);

        const PartId partId = 0;

        TensorShape inputShape{ 1, 128, 128, 64 };
        TensorShape outputShape{ 1, 128, 128, 64 };
        const QuantizationInfo inputQuantInfo(0, 1.0f);

        TensorInfo inputTensorsInfo(inputShape, DataType::UINT8_QUANTIZED, DataFormat::NHWCB, inputQuantInfo);
        TensorInfo outputTensorsInfo(outputShape, DataType::UINT8_QUANTIZED, DataFormat::NHWCB,
                                     QuantizationInfo(0, 1.0f));

        CheckPlansParams params;
        params.m_PartId           = partId;
        params.m_InputTensorsInfo = { inputTensorsInfo, inputTensorsInfo };
        params.m_OutputTensorInfo = outputTensorsInfo;
        params.m_DataFormat       = CascadingBufferFormat::NHWCB;

        EstimationOptions estOpts;
        CompilationOptions compOpts;
        StandalonePlePart part = BuildPart({ inputShape, inputShape }, { inputQuantInfo, inputQuantInfo }, outputShape,
                                           command_stream::PleOperation::ADDITION, caps, partId, estOpts, compOpts);

        // Only the lonely part is expected to return a plan

        SramBuffer prevBuffer;

        Plans plans0 = part.GetPlans(CascadeType::Beginning, command_stream::BlockConfig{}, nullptr, 1);
        REQUIRE(plans0.size() == 0);

        Plans plans1 = part.GetPlans(CascadeType::Lonely, command_stream::BlockConfig{}, nullptr, 1);
        CheckPlans(plans1, params);

        Plans plans2 = part.GetPlans(CascadeType::Middle, command_stream::BlockConfig{}, &prevBuffer, 1);
        REQUIRE(plans2.size() == 0);

        Plans plans3 = part.GetPlans(CascadeType::Middle, command_stream::BlockConfig{}, &prevBuffer, 1);
        REQUIRE(plans3.size() == 0);

        Plans plans4 = part.GetPlans(CascadeType::End, command_stream::BlockConfig{}, &prevBuffer, 1);
        REQUIRE(plans4.size() == 0);
    }
}

TEST_CASE("StandalonePlePart ADDITION_RESCALE")
{
    SECTION("2TOPS_2PLE_RATIO")
    {
        const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_2TOPS_2PLE_RATIO);

        const PartId partId = 0;

        TensorShape inputShape{ 1, 128, 32, 64 };
        TensorShape outputShape{ 1, 128, 32, 64 };
        const QuantizationInfo inputQuantInfo(0, 1.0f);

        TensorInfo inputTensorsInfo(inputShape, DataType::UINT8_QUANTIZED, DataFormat::NHWCB, inputQuantInfo);
        TensorInfo outputTensorsInfo(outputShape, DataType::UINT8_QUANTIZED, DataFormat::NHWCB,
                                     QuantizationInfo(0, 1.0f));

        CheckPlansParams params;
        params.m_PartId           = partId;
        params.m_InputTensorsInfo = { inputTensorsInfo, inputTensorsInfo };
        params.m_OutputTensorInfo = outputTensorsInfo;
        params.m_DataFormat       = CascadingBufferFormat::NHWCB;

        EstimationOptions estOpts;
        CompilationOptions compOpts;
        StandalonePlePart part =
            BuildPart({ inputShape, inputShape }, { inputQuantInfo, inputQuantInfo }, outputShape,
                      command_stream::PleOperation::ADDITION_RESCALE, caps, partId, estOpts, compOpts);

        // Only the lonely part is expected to return a plan

        SramBuffer prevBuffer;

        Plans plans0 = part.GetPlans(CascadeType::Beginning, command_stream::BlockConfig{}, nullptr, 1);
        REQUIRE(plans0.size() == 0);

        Plans plans1 = part.GetPlans(CascadeType::Lonely, command_stream::BlockConfig{}, nullptr, 1);
        CheckPlans(plans1, params);

        Plans plans2 = part.GetPlans(CascadeType::Middle, command_stream::BlockConfig{}, &prevBuffer, 1);
        REQUIRE(plans2.size() == 0);

        Plans plans3 = part.GetPlans(CascadeType::Middle, command_stream::BlockConfig{}, &prevBuffer, 1);
        REQUIRE(plans3.size() == 0);

        Plans plans4 = part.GetPlans(CascadeType::End, command_stream::BlockConfig{}, &prevBuffer, 1);
        REQUIRE(plans4.size() == 0);
    }

    SECTION("2TOPS_4PLE_RATIO")
    {
        const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_2TOPS_4PLE_RATIO);

        const PartId partId = 0;

        TensorShape inputShape{ 1, 128, 256, 64 };
        TensorShape outputShape{ 1, 128, 256, 64 };
        const QuantizationInfo inputQuantInfo(0, 1.0f);

        TensorInfo inputTensorsInfo(inputShape, DataType::UINT8_QUANTIZED, DataFormat::NHWCB, inputQuantInfo);
        TensorInfo outputTensorsInfo(outputShape, DataType::UINT8_QUANTIZED, DataFormat::NHWCB,
                                     QuantizationInfo(0, 2.0f));

        CheckPlansParams params;
        params.m_PartId           = partId;
        params.m_InputTensorsInfo = { inputTensorsInfo, inputTensorsInfo };
        params.m_OutputTensorInfo = outputTensorsInfo;
        params.m_DataFormat       = CascadingBufferFormat::NHWCB;

        EstimationOptions estOpts;
        CompilationOptions compOpts;
        StandalonePlePart part =
            BuildPart({ inputShape, inputShape }, { inputQuantInfo, inputQuantInfo }, outputShape,
                      command_stream::PleOperation::ADDITION_RESCALE, caps, partId, estOpts, compOpts);

        // Only the lonely part is expected to return a plan

        SramBuffer prevBuffer;

        Plans plans0 = part.GetPlans(CascadeType::Beginning, command_stream::BlockConfig{}, nullptr, 1);
        REQUIRE(plans0.size() == 0);

        Plans plans1 = part.GetPlans(CascadeType::Lonely, command_stream::BlockConfig{}, nullptr, 1);
        CheckPlans(plans1, params);

        Plans plans2 = part.GetPlans(CascadeType::Middle, command_stream::BlockConfig{}, &prevBuffer, 1);
        REQUIRE(plans2.size() == 0);

        Plans plans3 = part.GetPlans(CascadeType::Middle, command_stream::BlockConfig{}, &prevBuffer, 1);
        REQUIRE(plans3.size() == 0);

        Plans plans4 = part.GetPlans(CascadeType::End, command_stream::BlockConfig{}, &prevBuffer, 1);
        REQUIRE(plans4.size() == 0);
    }
}
