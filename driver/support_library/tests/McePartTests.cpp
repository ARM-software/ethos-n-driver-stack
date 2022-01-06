//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "CapabilitiesInternal.hpp"
#include "GlobalParameters.hpp"
#include "TestUtils.hpp"
#include "Utils.hpp"
#include "cascading/Cascading.hpp"
#include "cascading/McePart.hpp"
#include "cascading/Visualisation.hpp"
#include "ethosn_support_library/Support.hpp"

#include <catch.hpp>
#include <fstream>
#include <regex>

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

McePart BuildPart(TensorShape inputShape,
                  TensorShape outputShape,
                  TensorShape weightShape,
                  command_stream::MceOperation op,
                  Stride stride,
                  uint32_t padTop,
                  uint32_t padLeft,
                  uint32_t upscaleFactor,
                  command_stream::UpsampleType upsampleType,
                  const CompilationOptions& compOpt,
                  const HardwareCapabilities& caps)
{
    EstimationOptions estOps;
    McePart::ConstructionParams params(estOps, compOpt, caps);
    params.m_Id                     = 0;
    params.m_InputTensorShape       = inputShape;
    params.m_OutputTensorShape      = outputShape;
    params.m_InputQuantizationInfo  = QuantizationInfo(0, 1.0f);
    params.m_OutputQuantizationInfo = QuantizationInfo(0, 1.0f);
    params.m_WeightsInfo            = weightShape;
    params.m_WeightsInfo.m_DataFormat =
        op == command_stream::MceOperation::DEPTHWISE_CONVOLUTION ? DataFormat::HWIM : DataFormat::HWIO;
    params.m_WeightsInfo.m_QuantizationInfo = { 0, 0.9f };
    params.m_WeightsData                    = std::vector<uint8_t>(utils::GetNumElements(weightShape), 1);
    params.m_BiasInfo                       = TensorShape{ 1, 1, 1, outputShape[3] };
    params.m_BiasData                       = std::vector<int32_t>(outputShape[3], 0);
    params.m_Stride                         = stride;
    params.m_PadTop                         = padTop;
    params.m_PadLeft                        = padLeft;
    params.m_Op                             = op;
    params.m_OperationIds                   = { 1 };
    params.m_DataType                       = ethosn::command_stream::DataType::U8;
    params.m_UpscaleFactor                  = upscaleFactor;
    params.m_UpsampleType                   = upsampleType;
    McePart part(std::move(params));

    return part;
}

McePart BuildPart(TensorShape inputShape,
                  TensorShape outputShape,
                  TensorShape weightShape,
                  command_stream::MceOperation op,
                  Stride stride,
                  uint32_t padTop,
                  uint32_t padLeft,
                  const CompilationOptions& compOpt,
                  const HardwareCapabilities& caps)
{
    return BuildPart(inputShape, outputShape, weightShape, op, stride, padTop, padLeft, 1,
                     command_stream::UpsampleType::OFF, compOpt, caps);
}

McePart BuildPart(TensorShape inputShape,
                  TensorShape outputShape,
                  TensorShape weightShape,
                  command_stream::MceOperation op,
                  const CompilationOptions& compOpt,
                  const HardwareCapabilities& caps)
{
    return BuildPart(inputShape, outputShape, weightShape, op, Stride(1, 1), 0, 0, compOpt, caps);
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
    Sram,
    Dram,
};

enum class PlanOutputLocation : uint32_t
{
    PleInputSram = 0x1,
    Sram         = 0x2,
    Dram         = 0x4,
};

PlanOutputLocation operator|(PlanOutputLocation l, PlanOutputLocation r) noexcept
{
    return static_cast<PlanOutputLocation>(static_cast<uint32_t>(l) | static_cast<uint32_t>(r));
}
bool operator&(PlanOutputLocation l, PlanOutputLocation r) noexcept
{
    return static_cast<bool>(static_cast<uint32_t>(l) & static_cast<uint32_t>(r));
}

using PlanDescFunc      = std::function<void(const PlanDesc& planDesc)>;
using PlanDescPredicate = std::function<bool(const PlanDesc& planDesc)>;

struct CheckPlansParams
{
    /// The structure of the expected plans. If the OpGraph structure of any plans are not consistent with
    /// the input/output locations allowed here, then the test will fail.
    /// @{
    PlanInputLocation m_InputLocation    = PlanInputLocation::Sram;
    PlanOutputLocation m_OutputLocations = PlanOutputLocation::Sram | PlanOutputLocation::PleInputSram;
    /// @}

    /// If provided, the properties of Ops and Buffers all plans must meet, otherwise the test will fail.
    /// @{
    utils::Optional<PartId> m_PartId;
    utils::Optional<TensorShape> m_InputShape;
    utils::Optional<QuantizationInfo> m_InputQuantInfo;
    utils::Optional<TensorShape> m_OutputShape;
    utils::Optional<QuantizationInfo> m_OutputQuantInfo;
    utils::Optional<TensorInfo> m_WeightsTensorInfo;
    utils::Optional<command_stream::MceOperation> m_MceOp;
    utils::Optional<Stride> m_Stride;
    utils::Optional<uint32_t> m_PadTop;
    utils::Optional<uint32_t> m_PadLeft;
    utils::Optional<uint32_t> m_UpscaleFactor;
    utils::Optional<command_stream::UpsampleType> m_UpsampleType;
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
    desc.m_InputSram    = buffers.at(i++);
    desc.m_WeightsDram  = buffers.at(i++);
    desc.m_WeightsSram  = buffers.at(i++);
    desc.m_PleInputSram = buffers.at(i++);
    if ((params.m_OutputLocations & PlanOutputLocation::PleInputSram) && i == buffers.size())
    {
        // Fine, no more buffers
    }
    else if ((params.m_OutputLocations & PlanOutputLocation::Sram) && i + 1 == buffers.size())
    {
        desc.m_OutputSram = buffers.at(i++);
    }
    else if ((params.m_OutputLocations & PlanOutputLocation::Dram) && i + 2 == buffers.size())
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

void CheckWeightsDram(PlanDesc& desc, const CheckPlansParams& params)
{
    // Check properties of Weights DRAM buffer
    CHECK(desc.m_WeightsDram->m_Location == Location::Dram);
    CHECK(desc.m_WeightsDram->m_Lifetime == Lifetime::Cascade);
    CHECK(desc.m_WeightsDram->m_Format == CascadingBufferFormat::WEIGHT);
    if (params.m_WeightsTensorInfo)
    {
        CHECK(desc.m_WeightsDram->m_QuantizationInfo == params.m_WeightsTensorInfo.value().m_QuantizationInfo);
        CHECK(desc.m_WeightsDram->m_TensorShape == params.m_WeightsTensorInfo.value().m_Dimensions);
    }
    CHECK(desc.m_WeightsDram->m_StripeShape == TensorShape{ 0, 0, 0, 0 });
    CHECK(desc.m_WeightsDram->m_Order == TraversalOrder::Xyz);
    CHECK(desc.m_WeightsDram->m_NumStripes == 0);
    REQUIRE(desc.m_WeightsDram->m_EncodedWeights != nullptr);
    CHECK(desc.m_WeightsDram->m_EncodedWeights->m_Data.size() > 0);
    CHECK(desc.m_WeightsDram->m_SizeInBytes == desc.m_WeightsDram->m_EncodedWeights->m_Data.size());
}

void CheckWeightsSram(PlanDesc& desc, const CheckPlansParams& params)
{
    // Check properties of Weights SRAM buffer
    CHECK(desc.m_WeightsSram->m_Location == Location::Sram);
    CHECK(desc.m_WeightsSram->m_Lifetime == Lifetime::Cascade);
    CHECK(desc.m_WeightsSram->m_Format == CascadingBufferFormat::WEIGHT);
    if (params.m_WeightsTensorInfo)
    {
        CHECK(desc.m_WeightsSram->m_QuantizationInfo == params.m_WeightsTensorInfo.value().m_QuantizationInfo);
        CHECK(desc.m_WeightsSram->m_TensorShape == params.m_WeightsTensorInfo.value().m_Dimensions);
    }
    else    // If we weren't provided with an expected tensor info, then at least check that it's consistent between the Dram and Sram buffers
    {
        CHECK(desc.m_WeightsSram->m_QuantizationInfo == desc.m_WeightsDram->m_QuantizationInfo);
        CHECK(desc.m_WeightsSram->m_TensorShape == desc.m_WeightsDram->m_TensorShape);
    }
    // m_StripeShape, m_Order, m_NumStripes will depend on the streaming strategy, and so cannot be checked generically
    CHECK(desc.m_WeightsSram->m_SizeInBytes ==
          desc.m_WeightsDram->m_EncodedWeights->m_MaxSize * desc.m_WeightsSram->m_NumStripes);
    CHECK(desc.m_WeightsSram->m_EncodedWeights == nullptr);
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
    if (params.m_OutputShape)
    {
        // Note if this isn't provided, we can still check the tensor shape by comparing with the m_OutputSram buffer,
        // if that is present (see CheckOutputSram).
        CHECK(desc.m_PleInputSram->m_TensorShape == params.m_OutputShape.value());
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
        else    // If we weren't provided with an expected output tensor info, then at least check that it's consistent
        {
            CHECK(desc.m_OutputSram->m_TensorShape == desc.m_PleInputSram->m_TensorShape);
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
        desc.m_WeightsDma = RequireCast<DmaOp*>(ops.at(i++));
        desc.m_Mce        = RequireCast<MceOp*>(ops.at(i++));
        if ((params.m_OutputLocations & PlanOutputLocation::PleInputSram) && i == ops.size())
        {
            // Fine, no more ops
        }
        else if ((params.m_OutputLocations & PlanOutputLocation::Sram) && i + 1 == ops.size())
        {
            desc.m_Ple = RequireCast<PleOp*>(ops.at(i++));
        }
        else if ((params.m_OutputLocations & PlanOutputLocation::Dram) && i + 2 == ops.size())
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
    CHECK(desc.m_WeightsDma->m_Lifetime == Lifetime::Cascade);
    if (params.m_OperationIds)
    {
        CHECK(desc.m_WeightsDma->m_OperationIds == params.m_OperationIds.value());
    }
}

void CheckMce(const CheckPlansParams& params, PlanDesc& desc)
{
    // Check properties of Mce Op
    CHECK(desc.m_Mce->m_Lifetime == Lifetime::Cascade);
    if (params.m_OperationIds)
    {
        CHECK(desc.m_Mce->m_OperationIds == params.m_OperationIds.value());
    }
    if (params.m_MceOp)
    {
        CHECK(desc.m_Mce->m_Op == params.m_MceOp.value());
    }
    // m_Algo, m_Block, m_InputStripeShape, m_OutputStripeShape, m_WeightsStripeShape, m_Order will depend on the streaming strategy, and so cannot be checked generically
    if (params.m_Stride)
    {
        CHECK(desc.m_Mce->m_Stride == params.m_Stride.value());
    }
    if (params.m_PadLeft)
    {
        CHECK(desc.m_Mce->m_PadLeft == params.m_PadLeft.value());
    }
    if (params.m_PadTop)
    {
        CHECK(desc.m_Mce->m_PadTop == params.m_PadTop.value());
    }
    if (params.m_UpscaleFactor)
    {
        CHECK(desc.m_Mce->m_UpscaleFactor == params.m_UpscaleFactor.value());
    }
    if (params.m_UpsampleType)
    {
        CHECK(desc.m_Mce->m_UpsampleType == params.m_UpsampleType.value());
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
        CHECK(desc.m_Ple->m_Op == command_stream::PleOperation::PASSTHROUGH);
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

    CHECK(plan.m_OpGraph.GetProducer(desc.m_InputSram) ==
          (params.m_InputLocation == PlanInputLocation::Dram ? desc.m_InputDma : nullptr));
    CHECK(plan.m_OpGraph.GetConsumers(desc.m_InputSram) == std::vector<std::pair<Op*, uint32_t>>{ { desc.m_Mce, 0 } });

    CHECK(plan.m_OpGraph.GetProducer(desc.m_WeightsDram) == nullptr);
    CHECK(plan.m_OpGraph.GetConsumers(desc.m_WeightsDram) ==
          std::vector<std::pair<Op*, uint32_t>>{ { desc.m_WeightsDma, 0 } });

    CHECK(plan.m_OpGraph.GetProducer(desc.m_WeightsSram) == desc.m_WeightsDma);
    CHECK(plan.m_OpGraph.GetConsumers(desc.m_WeightsSram) ==
          std::vector<std::pair<Op*, uint32_t>>{ { desc.m_Mce, 1 } });

    CHECK(plan.m_OpGraph.GetProducer(desc.m_PleInputSram) == desc.m_Mce);
    CHECK(plan.m_OpGraph.GetConsumers(desc.m_PleInputSram) ==
          (desc.m_Ple ? std::vector<std::pair<Op*, uint32_t>>{ { desc.m_Ple, 0 } }
                      : std::vector<std::pair<Op*, uint32_t>>{}));

    if (desc.m_OutputSram)
    {
        CHECK(plan.m_OpGraph.GetProducer(desc.m_OutputSram) == desc.m_Ple);
        CHECK(plan.m_OpGraph.GetConsumers(desc.m_OutputSram) ==
              (desc.m_OutputDma ? std::vector<std::pair<Op*, uint32_t>>{ { desc.m_OutputDma, 0 } }
                                : std::vector<std::pair<Op*, uint32_t>>{}));
    }
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
    CHECK(plan.m_InputMappings.begin()->first ==
          (params.m_InputLocation == PlanInputLocation::Dram ? desc.m_InputDram : desc.m_InputSram));
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

/// Checks that McePart::GetPlans returns sensible plans for different cascade types.
/// Doesn't check anything specific to any streaming strategy, just checks that the Plans have the right
/// structure (an MceOp with weights buffer etc.) and the Buffers and Ops have the right properties.
TEST_CASE("McePart GetPlans structure")
{
    GIVEN("A simple McePart")
    {
        const CompilationOptions compOpt;
        EstimationOptions estOps;
        const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);

        const PartId partId = 0;
        TensorShape tsIn    = { 1, 32, 32, 3 };
        TensorShape tsOut   = { 1, 64, 64, 1 };
        const std::vector<uint8_t> weights(1 * 1 * utils::GetChannels(tsIn) * utils::GetChannels(tsOut), 1);
        const std::vector<int32_t> bias(utils::GetChannels(tsOut), 0);
        const QuantizationInfo inputQuantInfo(0, 1.0f);
        const QuantizationInfo outputQuantInfo(0, 1.0f);
        const TensorInfo weightsTensorInfo{ TensorShape{ 1, 1, utils::GetChannels(tsIn), utils::GetChannels(tsOut) },
                                            DataType::UINT8_QUANTIZED, DataFormat::HWIO, QuantizationInfo(0, 0.9f) };
        const TensorInfo biasTensorInfo({ 1, 1, 1, utils::GetChannels(tsOut) });
        const std::set<uint32_t> operationIds   = { 1, 2, 3 };
        const command_stream::MceOperation csOp = command_stream::MceOperation::CONVOLUTION;
        const Stride stride                     = {};
        const uint32_t padTop                   = 0;
        const uint32_t padLeft                  = 0;
        McePart part(partId, tsIn, tsOut, inputQuantInfo, outputQuantInfo, weightsTensorInfo, weights, biasTensorInfo,
                     bias, stride, padTop, padLeft, csOp, estOps, compOpt, caps, operationIds,
                     ethosn::command_stream::DataType::U8);

        CheckPlansParams params;
        params.m_PartId            = partId;
        params.m_InputShape        = tsIn;
        params.m_InputQuantInfo    = inputQuantInfo;
        params.m_OutputShape       = tsOut;
        params.m_OutputQuantInfo   = outputQuantInfo;
        params.m_WeightsTensorInfo = weightsTensorInfo;
        params.m_MceOp             = csOp;
        params.m_Stride            = stride;
        params.m_PadTop            = padTop;
        params.m_PadLeft           = padLeft;
        params.m_OperationIds      = operationIds;

        WHEN("Asked to produce Lonely plans")
        {
            Plans plans = part.GetPlans(CascadeType::Lonely, command_stream::BlockConfig{}, nullptr, 0);
            SavePlansToDot(plans, "McePart GetPlans structure Lonely");

            THEN("The plans are valid, start and end in Sram")
            {
                params.m_InputLocation   = PlanInputLocation::Sram;
                params.m_OutputLocations = PlanOutputLocation::Sram;
                CheckPlans(plans, params);
            }
        }

        WHEN("Asked to produce Beginning plans")
        {
            Plans plans = part.GetPlans(CascadeType::Beginning, command_stream::BlockConfig{}, nullptr, 0);
            SavePlansToDot(plans, "McePart GetPlans structure Beginning");

            THEN("The plans are valid and start in Sram and end in either Sram or PleInputSram")
            {
                params.m_InputLocation   = PlanInputLocation::Sram;
                params.m_OutputLocations = PlanOutputLocation::Sram | PlanOutputLocation::PleInputSram;
                // Confirm that we have at least one plan that ends in Sram and at least one that ends in PleInputSram
                params.m_Any.push_back(
                    [](const PlanDesc& plan) { return plan.m_Output->m_Location == Location::Sram; });
                params.m_Any.push_back(
                    [](const PlanDesc& plan) { return plan.m_Output->m_Location == Location::PleInputSram; });
                CheckPlans(plans, params);
            }
        }

        WHEN("Asked to produce Middle plans")
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

            Plans plans = part.GetPlans(CascadeType::Middle, command_stream::BlockConfig{ 8U, 8U }, &prevBuffer, 1);
            SavePlansToDot(plans, "McePart GetPlans structure Middle");

            THEN("The plans are valid and start in Sram and end in either Sram or PleInputSram")
            {
                params.m_InputLocation   = PlanInputLocation::Sram;
                params.m_OutputLocations = PlanOutputLocation::Sram | PlanOutputLocation::PleInputSram;
                // Confirm that we have at least one plan that ends in Sram and at least one that ends in PleInputSram
                params.m_Any.push_back(
                    [](const PlanDesc& plan) { return plan.m_Output->m_Location == Location::Sram; });
                params.m_Any.push_back(
                    [](const PlanDesc& plan) { return plan.m_Output->m_Location == Location::PleInputSram; });
                CheckPlans(plans, params);
            }
        }

        WHEN("Asked to produce End plans")
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
            Plans plans = part.GetPlans(CascadeType::End, command_stream::BlockConfig{ 8U, 8U }, &prevBuffer, 1);
            SavePlansToDot(plans, "McePart GetPlans structure End");

            THEN("The plans are valid and start in Sram and end in Sram")
            {
                params.m_InputLocation   = PlanInputLocation::Sram;
                params.m_OutputLocations = PlanOutputLocation::Sram;
                CheckPlans(plans, params);
            }
        }
    }
}

/// Checks that McePart::GetPlans returns a sensible plan for strategy 3.
/// This covers the Buffer/Op properties which aren't covered by above 'structure' test as they are specific
/// to the strategy.
TEST_CASE("McePart GetPlans Strategy3", "[slow]")
{
    GIVEN("An McePart for a simple convolution layer")
    {
        const CompilationOptions compOpt;
        const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);

        TensorShape inputShape{ 1, 16, 16, 16 };
        TensorShape outputShape{ 1, 16, 16, 16 };
        TensorShape weightShape{ 1, 1, 16, 16 };

        McePart part =
            BuildPart(inputShape, outputShape, weightShape, command_stream::MceOperation::CONVOLUTION, compOpt, caps);

        WHEN("Asked to generate plans")
        {
            Plans plans = part.GetPlans(CascadeType::Lonely, command_stream::BlockConfig{}, nullptr, 1);
            SavePlansToDot(plans, "McePart GetPlans Strategy3");

            THEN("The plans are valid and contain at least one plan with Strategy3 stripe shapes and properties")
            {
                CheckPlansParams params;
                params.m_InputShape  = inputShape;
                params.m_OutputShape = outputShape;
                params.m_Any.push_back([](const PlanDesc& plan) {
                    if (plan.m_Output->m_Location != Location::Sram)
                    {
                        // Wait until we get a plan that includes a PleOp (some will end before the Ple), so we can test more things.
                        return false;
                    }

                    bool inputSramValid = plan.m_InputSram->m_StripeShape == TensorShape{ 1, 16, 16, 16 } &&
                                          plan.m_InputSram->m_Order == TraversalOrder::Xyz &&
                                          plan.m_InputSram->m_SizeInBytes == 16 * 16 * 16 &&
                                          plan.m_InputSram->m_NumStripes == 1;
                    bool weightsSramValid = plan.m_WeightsSram->m_StripeShape == TensorShape{ 1, 1, 16, 16 } &&
                                            plan.m_WeightsSram->m_Order == TraversalOrder::Xyz &&
                                            plan.m_WeightsSram->m_NumStripes == 1;
                    bool pleInputSramValid = plan.m_PleInputSram->m_StripeShape == TensorShape{ 1, 16, 16, 16 } &&
                                             plan.m_PleInputSram->m_Order == TraversalOrder::Xyz &&
                                             plan.m_PleInputSram->m_SizeInBytes == 16 * 16 * 16 &&
                                             plan.m_PleInputSram->m_NumStripes == 0;
                    bool outputSramValid = plan.m_OutputSram->m_StripeShape == TensorShape{ 1, 16, 16, 16 } &&
                                           plan.m_OutputSram->m_Order == TraversalOrder::Xyz &&
                                           plan.m_OutputSram->m_SizeInBytes == 16 * 16 * 16 &&
                                           plan.m_OutputSram->m_NumStripes == 1;
                    bool mceValid = plan.m_Mce->m_Algo == CompilerMceAlgorithm::Direct &&
                                    plan.m_Mce->m_BlockConfig == command_stream::BlockConfig{ 16u, 16u } &&
                                    plan.m_Mce->m_InputStripeShape == TensorShape{ 1, 16, 16, 16 } &&
                                    plan.m_Mce->m_OutputStripeShape == TensorShape{ 1, 16, 16, 16 } &&
                                    plan.m_Mce->m_WeightsStripeShape == TensorShape{ 1, 1, 16, 16 } &&
                                    plan.m_Mce->m_Order == TraversalOrder::Xyz;
                    bool pleValid =
                        plan.m_Ple->m_BlockConfig == command_stream::BlockConfig{ 16u, 16u } &&
                        plan.m_Ple->m_InputStripeShapes == std::vector<TensorShape>{ TensorShape{ 1, 16, 16, 16 } } &&
                        plan.m_Ple->m_OutputStripeShape == TensorShape{ 1, 16, 16, 16 };

                    return inputSramValid && weightsSramValid && pleInputSramValid && outputSramValid && mceValid &&
                           pleValid;
                });
                CheckPlans(plans, params);
            }
        }
    }
}

/// Checks that McePart::GetPlans returns a sensible plan for strategy 3.
/// This covers the Buffer/Op properties which aren't covered by above 'structure' test as they are specific
/// to the strategy.
TEST_CASE("McePart GetPlans Strategy0", "[slow]")
{
    GIVEN("An McePart for a simple convolution layer")
    {
        const CompilationOptions compOpt;
        const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);

        TensorShape inputShape{ 1, 32, 16, 16 };
        TensorShape outputShape{ 1, 32, 16, 16 };
        TensorShape weightShape{ 1, 1, 16, 16 };

        McePart part =
            BuildPart(inputShape, outputShape, weightShape, command_stream::MceOperation::CONVOLUTION, compOpt, caps);

        WHEN("Asked to generate plans")
        {
            Plans plans = part.GetPlans(CascadeType::Lonely, command_stream::BlockConfig{}, nullptr, 1);
            SavePlansToDot(plans, "McePart GetPlans Strategy0");

            THEN("The plans are valid and contain at least one plan with Strategy0 stripe shapes and properties")
            {
                CheckPlansParams params;
                params.m_InputShape  = inputShape;
                params.m_OutputShape = outputShape;
                params.m_Any.push_back([](const PlanDesc& plan) {
                    if (plan.m_Output->m_Location != Location::Sram)
                    {
                        // Wait until we get a plan that includes a PleOp (some will end before the Ple), so we can test more things.
                        return false;
                    }

                    bool inputSramValid = plan.m_InputSram->m_StripeShape == TensorShape{ 1, 8, 16, 16 } &&
                                          plan.m_InputSram->m_Order == TraversalOrder::Xyz &&
                                          plan.m_InputSram->m_SizeInBytes == 8 * 16 * 16 &&
                                          plan.m_InputSram->m_NumStripes == 1;
                    bool weightsSramValid = plan.m_WeightsSram->m_StripeShape == TensorShape{ 1, 1, 16, 16 } &&
                                            plan.m_WeightsSram->m_Order == TraversalOrder::Xyz &&
                                            plan.m_WeightsSram->m_NumStripes == 1;
                    bool pleInputSramValid = plan.m_PleInputSram->m_StripeShape == TensorShape{ 1, 8, 16, 16 } &&
                                             plan.m_PleInputSram->m_Order == TraversalOrder::Xyz &&
                                             plan.m_PleInputSram->m_SizeInBytes == 8 * 16 * 16 &&
                                             plan.m_PleInputSram->m_NumStripes == 0;
                    bool outputSramValid = plan.m_OutputSram->m_StripeShape == TensorShape{ 1, 8, 16, 16 } &&
                                           plan.m_OutputSram->m_Order == TraversalOrder::Xyz &&
                                           plan.m_OutputSram->m_SizeInBytes == 8 * 16 * 16 &&
                                           plan.m_OutputSram->m_NumStripes == 1;
                    bool mceValid = plan.m_Mce->m_Algo == CompilerMceAlgorithm::Direct &&
                                    plan.m_Mce->m_BlockConfig == command_stream::BlockConfig{ 16u, 8u } &&
                                    plan.m_Mce->m_InputStripeShape == TensorShape{ 1, 8, 16, 16 } &&
                                    plan.m_Mce->m_OutputStripeShape == TensorShape{ 1, 8, 16, 16 } &&
                                    plan.m_Mce->m_WeightsStripeShape == TensorShape{ 1, 1, 16, 16 } &&
                                    plan.m_Mce->m_Order == TraversalOrder::Xyz;
                    bool pleValid =
                        plan.m_Ple->m_BlockConfig == command_stream::BlockConfig{ 16u, 8u } &&
                        plan.m_Ple->m_InputStripeShapes == std::vector<TensorShape>{ TensorShape{ 1, 8, 16, 16 } } &&
                        plan.m_Ple->m_OutputStripeShape == TensorShape{ 1, 8, 16, 16 };

                    return inputSramValid && weightsSramValid && pleInputSramValid && outputSramValid && mceValid &&
                           pleValid;
                });
                CheckPlans(plans, params);
            }
        }
    }
}

/// Checks that McePart::GetPlans returns a correctly filtered set of plans when requesting a specific block config,
/// previous SRAM buffer or number of weight stripes.
TEST_CASE("McePart GetPlans Filters", "[slow]")
{
    GIVEN("An McePart for a simple convolution layer")
    {
        const CompilationOptions compOpt;
        const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);

        TensorShape inputShape{ 1, 16, 16, 16 };
        TensorShape outputShape{ 1, 16, 16, 16 };
        TensorShape weightShape{ 1, 1, 16, 16 };

        McePart part =
            BuildPart(inputShape, outputShape, weightShape, command_stream::MceOperation::CONVOLUTION, compOpt, caps);

        WHEN("Asked to generate plans with a specific block config, Sram Buffer, and the number of weight stripes")
        {
            command_stream::BlockConfig requestedBlockConfig = { 32u, 8u };

            Buffer prevBuffer;
            prevBuffer.m_Lifetime         = Lifetime::Cascade;
            prevBuffer.m_Location         = Location::Sram;
            prevBuffer.m_Format           = CascadingBufferFormat::NHWCB;
            prevBuffer.m_QuantizationInfo = { 0, 1.0f };
            prevBuffer.m_TensorShape      = inputShape;
            prevBuffer.m_StripeShape      = TensorShape{ 1, 8, 16, 16 };
            prevBuffer.m_Order            = TraversalOrder::Xyz;
            prevBuffer.m_SizeInBytes      = 8 * 16 * 16 * 1;
            prevBuffer.m_NumStripes       = 1;

            const uint32_t numWeightStripes = 1;

            Plans plans = part.GetPlans(CascadeType::Middle, requestedBlockConfig, &prevBuffer, numWeightStripes);

            SavePlansToDot(plans, "McePart GetPlans Filters Block Config");

            THEN("The plans all use the requested block config, Sram Buffer, and the number of weight stripes")
            {
                CheckPlansParams params;
                params.m_All = [&](const PlanDesc& plan) {
                    CHECK(plan.m_Mce->m_BlockConfig == requestedBlockConfig);
                    if (plan.m_Ple)
                    {
                        CHECK(plan.m_Ple->m_BlockConfig == requestedBlockConfig);
                    }

                    CHECK(plan.m_Input->m_Lifetime == prevBuffer.m_Lifetime);
                    CHECK(plan.m_Input->m_Location == prevBuffer.m_Location);
                    CHECK(plan.m_Input->m_Format == prevBuffer.m_Format);
                    CHECK(plan.m_Input->m_QuantizationInfo == prevBuffer.m_QuantizationInfo);
                    CHECK(plan.m_Input->m_TensorShape == prevBuffer.m_TensorShape);
                    CHECK(plan.m_Input->m_StripeShape == prevBuffer.m_StripeShape);
                    CHECK(plan.m_Input->m_Order == prevBuffer.m_Order);
                    CHECK(plan.m_Input->m_SizeInBytes == prevBuffer.m_SizeInBytes);
                    CHECK(plan.m_Input->m_NumStripes == prevBuffer.m_NumStripes);

                    CHECK(plan.m_WeightsSram->m_NumStripes == numWeightStripes);
                };
                CheckPlans(plans, params);
            }
        }
        WHEN("Asked to generate plans with a specific block config, Sram Buffer, and too many weight stripes")
        {
            command_stream::BlockConfig requestedBlockConfig = { 32u, 8u };

            Buffer prevBuffer;
            prevBuffer.m_Lifetime         = Lifetime::Cascade;
            prevBuffer.m_Location         = Location::Sram;
            prevBuffer.m_Format           = CascadingBufferFormat::NHWCB;
            prevBuffer.m_QuantizationInfo = { 0, 1.0f };
            prevBuffer.m_TensorShape      = inputShape;
            prevBuffer.m_StripeShape      = TensorShape{ 1, 8, 16, 16 };
            prevBuffer.m_Order            = TraversalOrder::Xyz;
            prevBuffer.m_SizeInBytes      = 8 * 16 * 16 * 1;
            prevBuffer.m_NumStripes       = 1;

            const uint32_t numWeightStripes = 2;

            Plans plans = part.GetPlans(CascadeType::Middle, requestedBlockConfig, &prevBuffer, numWeightStripes);

            THEN("There are 0 plans generated")
            {
                REQUIRE(plans.size() == 0);
            }
        }
        WHEN("Asked to generate plans with an sram buffer with too much data")
        {
            command_stream::BlockConfig requestedBlockConfig = { 32u, 8u };

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

            const uint32_t numWeightStripes = 2;

            Plans plans = part.GetPlans(CascadeType::Middle, requestedBlockConfig, &prevBuffer, numWeightStripes);

            THEN("There are 0 plans generated")
            {
                REQUIRE(plans.size() == 0);
            }
        }
    }
}

/// Checks that McePart::GetPlans returns a correctly filtered set of plans when requesting a specific block config,
/// previous SRAM buffer or number of weight stripes.
TEST_CASE("McePart GetPlans multiple", "[slow]")
{
    GIVEN("3 MceParts simple convolution layers")
    {
        const CompilationOptions compOpt;
        const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);

        TensorShape inputShape{ 1, 16, 16, 16 };
        TensorShape outputShape{ 1, 16, 16, 16 };
        TensorShape weightShape{ 1, 1, 16, 16 };

        McePart part0 =
            BuildPart(inputShape, outputShape, weightShape, command_stream::MceOperation::CONVOLUTION, compOpt, caps);
        Buffer part0OutputBuffer;
        McePart part1 =
            BuildPart(inputShape, outputShape, weightShape, command_stream::MceOperation::CONVOLUTION, compOpt, caps);
        Buffer part1OutputBuffer;
        McePart part2 =
            BuildPart(inputShape, outputShape, weightShape, command_stream::MceOperation::CONVOLUTION, compOpt, caps);

        WHEN("Asked to generate plans for the beginning, middle and end of a cascade")
        {

            const uint32_t numWeightStripes = 1;

            Plans plans =
                part0.GetPlans(CascadeType::Beginning, command_stream::BlockConfig{}, nullptr, numWeightStripes);

            SavePlansToDot(plans, "McePart GetPlans Filters Block Config");

            THEN("The plans are valid")
            {
                CheckPlansParams params;
                params.m_InputShape  = inputShape;
                params.m_OutputShape = outputShape;
                params.m_Any.push_back([&](const PlanDesc& plan) {
                    bool inputSramValid = plan.m_InputSram->m_StripeShape == TensorShape{ 1, 8, 16, 16 } &&
                                          plan.m_InputSram->m_Order == TraversalOrder::Xyz &&
                                          plan.m_InputSram->m_SizeInBytes == 8 * 16 * 16 * 2 &&
                                          plan.m_InputSram->m_NumStripes == 2;
                    bool weightsSramValid = plan.m_WeightsSram->m_StripeShape == TensorShape{ 1, 1, 16, 16 } &&
                                            plan.m_WeightsSram->m_Order == TraversalOrder::Xyz &&
                                            plan.m_WeightsSram->m_NumStripes == 1;
                    bool pleInputSramValid = plan.m_PleInputSram->m_StripeShape == TensorShape{ 1, 8, 16, 16 } &&
                                             plan.m_PleInputSram->m_Order == TraversalOrder::Xyz &&
                                             plan.m_PleInputSram->m_SizeInBytes == 8 * 16 * 16 &&
                                             plan.m_PleInputSram->m_NumStripes == 0;
                    bool outputSramValid = true;
                    if (plan.m_OutputSram)
                    {
                        outputSramValid = plan.m_OutputSram->m_StripeShape == TensorShape{ 1, 8, 16, 16 } &&
                                          plan.m_OutputSram->m_Order == TraversalOrder::Xyz &&
                                          plan.m_OutputSram->m_SizeInBytes == 8 * 16 * 16 &&
                                          plan.m_OutputSram->m_NumStripes == 1;
                    }
                    bool mceValid = plan.m_Mce->m_Algo == CompilerMceAlgorithm::Direct &&
                                    plan.m_Mce->m_BlockConfig == command_stream::BlockConfig{ 16u, 8u } &&
                                    plan.m_Mce->m_InputStripeShape == TensorShape{ 1, 8, 16, 16 } &&
                                    plan.m_Mce->m_OutputStripeShape == TensorShape{ 1, 8, 16, 16 } &&
                                    plan.m_Mce->m_WeightsStripeShape == TensorShape{ 1, 1, 16, 16 } &&
                                    plan.m_Mce->m_Order == TraversalOrder::Xyz;
                    bool pleValid = true;
                    if (plan.m_Ple)
                    {
                        pleValid = plan.m_Ple->m_BlockConfig == command_stream::BlockConfig{ 16u, 8u } &&
                                   plan.m_Ple->m_InputStripeShapes ==
                                       std::vector<TensorShape>{ TensorShape{ 1, 8, 16, 16 } } &&
                                   plan.m_Ple->m_OutputStripeShape == TensorShape{ 1, 8, 16, 16 };
                    }
                    bool pass = inputSramValid && weightsSramValid && pleInputSramValid && outputSramValid &&
                                mceValid && pleValid;
                    if (pass && plan.m_OutputSram)
                    {
                        part0OutputBuffer = *plan.m_OutputSram;
                    }
                    return pass;
                });
                CheckPlans(plans, params);

                command_stream::BlockConfig requestedBlockConfig = { 16u, 8u };

                Plans plans1 =
                    part1.GetPlans(CascadeType::Middle, requestedBlockConfig, &part0OutputBuffer, numWeightStripes);

                // There are 4 plans which are generated
                // 3 for mce + ple.
                //   1 output stripes
                //   2 output stripes
                //   3 output stripes
                // 1 for mce only
                REQUIRE(plans1.size() == 4);
                part1OutputBuffer = *plans[0].m_OpGraph.GetBuffers().back();

                Plans plans2 =
                    part2.GetPlans(CascadeType::End, requestedBlockConfig, &part1OutputBuffer, numWeightStripes);

                // There is only 1 plan as we don't consider double buffering as the output stripe's is already >= output tensor due to rounding
                REQUIRE(plans2.size() == 1);
            }
        }
    }
}

TEST_CASE("McePart GetPlans Winograd")
{
    GIVEN("An McePart for a simple convolution")
    {
        const CompilationOptions compOpt;
        const HardwareCapabilities caps = GetEthosN78HwCapabilities();

        const uint32_t numIfms = 128;
        const uint32_t numOfms = 256;
        TensorShape tsIn       = { 1, 32, 32, numIfms };
        TensorShape tsOut      = { 1, 64, 64, numOfms };
        McePart part = BuildPart(tsIn, tsOut, { 3, 3, numIfms, numOfms }, command_stream::MceOperation::CONVOLUTION,
                                 Stride{ 1, 1 }, 1, 1, compOpt, caps);

        WHEN("Asked to generate plans")
        {
            Plans plans = part.GetPlans(CascadeType::Lonely, command_stream::BlockConfig{}, nullptr, 0);
            SavePlansToDot(plans, "McePart GetPlans Winograd");

            THEN("The plans are valid and have winograd enabled where possible")
            {
                CheckPlansParams params;
                params.m_InputShape  = tsIn;
                params.m_OutputShape = tsOut;
                params.m_All         = [&](const PlanDesc& plan) {
                    if (plan.m_Mce->m_WeightsStripeShape[2] < numIfms)
                    {
                        CHECK(plan.m_Mce->m_Algo == CompilerMceAlgorithm::Direct);
                    }
                    else if ((plan.m_Mce->m_BlockConfig == command_stream::BlockConfig{ 8U, 8U }) ||
                             (plan.m_Mce->m_BlockConfig == command_stream::BlockConfig{ 8U, 16U }) ||
                             (plan.m_Mce->m_BlockConfig == command_stream::BlockConfig{ 16U, 8U }))
                    {
                        CHECK(plan.m_Mce->m_Algo == CompilerMceAlgorithm::Winograd);
                    }
                    else
                    {
                        CHECK(plan.m_Mce->m_Algo == CompilerMceAlgorithm::Direct);
                    }
                };
                CheckPlans(plans, params);
            }
        }
    }
}

TEST_CASE("McePart GetPlans Split input in depth")
{
    GIVEN("An McePart for a convolution")
    {
        const CompilationOptions compOpt;
        const HardwareCapabilities caps = GetEthosN78HwCapabilities();

        TensorShape tsIn  = { 1, 64, 64, 256 };
        TensorShape tsOut = { 1, 64, 64, 64 };
        McePart part      = BuildPart(tsIn, tsOut, { 1, 1, 256, 64 }, command_stream::MceOperation::CONVOLUTION,
                                 Stride{ 2U, 2U }, 0, 0, compOpt, caps);

        WHEN("Asked to generate plans")
        {
            Plans plans = part.GetPlans(CascadeType::Lonely, command_stream::BlockConfig{}, nullptr, 0);
            SavePlansToDot(plans, "McePart GetPlans Split input in depth");

            THEN("The plans are valid, do not have unexpected stripe configs but do have expected stripe configs")
            {
                CheckPlansParams params;
                params.m_InputShape  = tsIn;
                params.m_OutputShape = tsOut;
                params.m_All         = [&](const PlanDesc& plan) {
                    CHECK(!(plan.m_InputSram->m_StripeShape == TensorShape{ 1, 16, 16, caps.GetNumberOfOgs() } &&
                            plan.m_InputSram->m_NumStripes == 1));
                    CHECK(!(plan.m_InputSram->m_StripeShape == TensorShape{ 1, 16, 16, caps.GetNumberOfOgs() } &&
                            plan.m_InputSram->m_NumStripes == 2));
                };
                params.m_Any.push_back([&](const PlanDesc& plan) {
                    return plan.m_InputSram->m_StripeShape == TensorShape{ 1, 16, 16, caps.GetNumberOfOgs() * 4 } &&
                           (plan.m_InputSram->m_NumStripes == 1 || plan.m_InputSram->m_NumStripes == 2);
                });
                CheckPlans(plans, params);
            }
        }
    }
}

TEST_CASE("McePart GetPlans Split output in depth")
{
    GIVEN("An McePart for a convolution")
    {
        const CompilationOptions compOpt;
        const HardwareCapabilities caps = GetEthosN78HwCapabilities();

        TensorShape inputShape{ 1, 8, 8, 32 };
        TensorShape outputShape{ 1, 8, 8, 32 };
        TensorShape weightShape{ 3, 3, 32, 32 };
        McePart part = BuildPart(inputShape, outputShape, weightShape, command_stream::MceOperation::CONVOLUTION,
                                 Stride{ 1, 1 }, 1, 1, compOpt, caps);

        WHEN("Asked to generate plans")
        {
            Plans plans = part.GetPlans(CascadeType::Lonely, command_stream::BlockConfig{}, nullptr, 0);
            SavePlansToDot(plans, "McePart GetPlans Split output in depth");

            THEN("The plans are valid and contain at least one plan with the stripe config we expect")
            {
                CheckPlansParams params;
                params.m_InputShape  = inputShape;
                params.m_OutputShape = outputShape;
                params.m_Any.push_back([&](const PlanDesc& plan) {
                    TensorShape inputStripe{ 1, 8, 8, 32 };
                    uint32_t numInputStripes = 1;
                    TensorShape pleOutputStripe{ 1, 8, 8, 8 };
                    TensorShape outputStripe{ 1, 8, 8, 16 };
                    uint32_t numOutputStripes = 2;

                    return plan.m_InputSram->m_StripeShape == inputStripe &&
                           plan.m_InputSram->m_NumStripes == numInputStripes &&
                           plan.m_OutputSram->m_StripeShape == outputStripe &&
                           plan.m_OutputSram->m_NumStripes == numOutputStripes &&
                           plan.m_Ple->m_OutputStripeShape == pleOutputStripe &&
                           // Check also the algorithm, to make sure we include output-depth-split plans with Winograd enabled
                           // (these were previously missing)
                           plan.m_Mce->m_Algo == CompilerMceAlgorithm::Winograd;
                });
                CheckPlans(plans, params);
            }
        }
    }
}

/// Checks that McePart produces at least the plans that we need for cascading MobileNet V1 in the same way that
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
/// The FusedPleParts are skipped here, and covered by a corresponding test in FusedPlePartTests.cpp.
///
/// For each McePart in the above list, we create an McePart object with the corresponding properties and ask it to generate
/// plans, providing the context (prevBuffer etc.) that it would have assuming that everything beforehand was chosen in agreement
/// with the prototype compiler. We then check that at least one of the plans returned is consistent with the prototype
/// compiler result. This gives us confidence that McePart is capable of generating the right plans in order to give an overall
/// performance consistent with the prototype compiler.
///
/// We don't cover every Part in the whole Network as that would be a lot of test code and would also be a lot of duplication.
TEST_CASE("McePart GetPlans MobileNet V1")
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
        prevBuffer.m_Location         = Location::Sram;
        prevBuffer.m_Format           = CascadingBufferFormat::NHWCB;
        prevBuffer.m_QuantizationInfo = { 0, 1.0f };
        prevBuffer.m_Order            = TraversalOrder::Xyz;
        prevBuffer.m_NumStripes = 1;    // For strategy 1 cascading, the buffers in SRAM are always the full tensor.

        // Notes:
        // - When the output buffer of a plan is in SRAM, it will always have the full stripe shape and a single stripe,
        //   because this is always the case for strategy 1 cascading. This may be different from the MCE output stripe shape,
        //   because the MCE still computes the data in multiple stripes, it's just stored in SRAM in a layout which is
        //   consistent with a single full stripe.
        // - For the configuration we have chosen (ETHOS_N78_8TOPS_2PLE_RATIO), there are 32 OGs and so the OFM stripe depths
        //   and weight stripe depths are generally going to be 32.

        ///  2. McePart CONVOLUTION 112,112,51 -> 112,112,32. Stride 2x2. Padding 1,1. Weights 3,3,3,32.
        SECTION("Part 2")
        {
            // Even though this is strategy 1, the variant we are compiling for has 32 OGs and so there is no actual splitting
            // and this is equivalent to strategy 3.
            // There are some differences between our results and the prototype compiler because this is a strided convolution and
            // the prototype compiler assumes that the DMA is used to do the interleaving of the IFM, which leads to a different IFM
            // depth (12 vs 51).

            TensorShape inputShape{ 1, 112, 112, 51 };
            TensorShape outputShape{ 1, 112, 112, 32 };
            TensorShape weightShape{ 3, 3, 3, 32 };
            McePart part = BuildPart(inputShape, outputShape, weightShape, command_stream::MceOperation::CONVOLUTION,
                                     Stride{ 2u, 2u }, 1, 1, compOpt, caps);
            prevBuffer.m_TensorShape = inputShape;
            prevBuffer.m_StripeShape = TensorShape{ 1, 112, 112, 64 };
            prevBuffer.m_SizeInBytes = 112 * 112 * 64;

            Plans plans = part.GetPlans(CascadeType::Middle, command_stream::BlockConfig{ 16u, 16u }, &prevBuffer, 1);
            CHECK(plans.size() == 2);
            CheckPlansParams params;
            params.m_Any.push_back([&](const PlanDesc& plan) {
                return plan.m_InputSram->m_StripeShape == TensorShape{ 1, 112, 112, 64 } &&
                       plan.m_InputSram->m_NumStripes == 1 &&
                       plan.m_WeightsSram->m_StripeShape == TensorShape{ 3, 3, 64, 32 } &&    // Strided
                       plan.m_WeightsSram->m_NumStripes == 1 &&
                       plan.m_Mce->m_BlockConfig == command_stream::BlockConfig{ 16u, 16u } &&
                       plan.m_Mce->m_InputStripeShape == TensorShape{ 1, 112, 112, 64 } &&
                       plan.m_Mce->m_WeightsStripeShape == TensorShape{ 3, 3, 64, 32 } &&
                       plan.m_Mce->m_OutputStripeShape == TensorShape{ 1, 112, 112, 32 } &&
                       // The following Part is another McePart, so we'll use the plan which includes a Passthrough PLE.
                       plan.m_Output->m_Location == Location::Sram &&
                       plan.m_Output->m_StripeShape == TensorShape{ 1, 112, 112, 32 } &&
                       plan.m_Output->m_NumStripes == 1;
            });
            CheckPlans(plans, params);
        }

        ///  3. McePart DEPTHWISE_CONVOLUTION 112,112,32 -> 112,112,32. Stride 1x1. Padding 1,1. Weights 3,3,32,1.
        SECTION("Part 3")
        {
            // Even though this is strategy 1, the variant we are compiling for has 32 OGs and so there is no actual splitting
            // and this is equivalent to strategy 3.

            TensorShape inputShape{ 1, 112, 112, 32 };
            TensorShape outputShape{ 1, 112, 112, 32 };
            TensorShape weightShape{ 3, 3, 32, 1 };
            McePart part =
                BuildPart(inputShape, outputShape, weightShape, command_stream::MceOperation::DEPTHWISE_CONVOLUTION,
                          Stride{ 1, 1 }, 1, 1, compOpt, caps);
            prevBuffer.m_TensorShape = inputShape;
            prevBuffer.m_StripeShape = TensorShape{ 1, 112, 112, 32 };
            prevBuffer.m_SizeInBytes = 112 * 112 * 32;

            Plans plans = part.GetPlans(CascadeType::Middle, command_stream::BlockConfig{ 16u, 16u }, &prevBuffer, 1);
            CHECK(plans.size() == 2);
            CheckPlansParams params;
            params.m_Any.push_back([&](const PlanDesc& plan) {
                return plan.m_InputSram->m_StripeShape == TensorShape{ 1, 112, 112, 32 } &&
                       plan.m_InputSram->m_NumStripes == 1 &&
                       plan.m_WeightsSram->m_StripeShape == TensorShape{ 3, 3, 32, 1 } &&
                       plan.m_WeightsSram->m_NumStripes == 1 &&
                       plan.m_Mce->m_BlockConfig == command_stream::BlockConfig{ 16u, 16u } &&
                       plan.m_Mce->m_InputStripeShape == TensorShape{ 1, 112, 112, 32 } &&
                       plan.m_Mce->m_WeightsStripeShape == TensorShape{ 3, 3, 32, 1 } &&
                       plan.m_Mce->m_OutputStripeShape == TensorShape{ 1, 112, 112, 32 } &&
                       // The following Part is another McePart, so we'll use the plan which includes a Passthrough PLE.
                       plan.m_Output->m_Location == Location::Sram &&
                       plan.m_Output->m_StripeShape == TensorShape{ 1, 112, 112, 32 } &&
                       plan.m_Output->m_NumStripes == 1;
            });
            CheckPlans(plans, params);
        }

        ///  4. McePart CONVOLUTION 112,112,32 -> 112,112,64. Stride 1x1. Padding 0,0. Weights 1,1,32,64.
        SECTION("Part 4")
        {
            TensorShape inputShape{ 1, 112, 112, 32 };
            TensorShape outputShape{ 1, 112, 112, 64 };
            TensorShape weightShape{ 1, 1, 32, 64 };
            McePart part = BuildPart(inputShape, outputShape, weightShape, command_stream::MceOperation::CONVOLUTION,
                                     Stride{ 1, 1 }, 0, 0, compOpt, caps);
            prevBuffer.m_TensorShape = inputShape;
            prevBuffer.m_StripeShape = TensorShape{ 1, 112, 112, 32 };
            prevBuffer.m_SizeInBytes = 112 * 112 * 32;

            Plans plans = part.GetPlans(CascadeType::Middle, command_stream::BlockConfig{ 16u, 16u }, &prevBuffer, 2);
            CHECK(plans.size() == 2);
            CheckPlansParams params;
            params.m_Any.push_back([&](const PlanDesc& plan) {
                return plan.m_InputSram->m_StripeShape == TensorShape{ 1, 112, 112, 32 } &&
                       plan.m_InputSram->m_NumStripes == 1 &&
                       plan.m_WeightsSram->m_StripeShape == TensorShape{ 1, 1, 32, 32 } &&
                       plan.m_WeightsSram->m_NumStripes == 2 &&
                       plan.m_Mce->m_BlockConfig == command_stream::BlockConfig{ 16u, 16u } &&
                       plan.m_Mce->m_InputStripeShape == TensorShape{ 1, 112, 112, 32 } &&
                       plan.m_Mce->m_WeightsStripeShape == TensorShape{ 1, 1, 32, 32 } &&
                       plan.m_Mce->m_OutputStripeShape == TensorShape{ 1, 112, 112, 32 } &&
                       // The following Part is a FusedPlePart, so we'll use the plan which ends at PleInputSram (and doesn't include a Passthrough PLE)
                       plan.m_Output->m_Location == Location::PleInputSram &&
                       plan.m_Output->m_StripeShape == TensorShape{ 1, 112, 112, 32 } &&
                       plan.m_Output->m_NumStripes == 0;
            });
            ;
            CheckPlans(plans, params);
        }

        ///  6. McePart DEPTHWISE_CONVOLUTION 56,56,256 -> 56,56,64. Stride 2x2. Padding 1,1. Weights 3,3,64,1.
        SECTION("Part 6")
        {
            TensorShape inputShape{ 1, 56, 56, 256 };
            TensorShape outputShape{ 1, 56, 56, 64 };
            TensorShape weightShape{ 3, 3, 64, 1 };
            McePart part =
                BuildPart(inputShape, outputShape, weightShape, command_stream::MceOperation::DEPTHWISE_CONVOLUTION,
                          Stride{ 2, 2 }, 1, 1, compOpt, caps);
            prevBuffer.m_TensorShape = inputShape;
            prevBuffer.m_StripeShape = TensorShape{ 1, 56, 56, 256 };
            prevBuffer.m_SizeInBytes = 56 * 56 * 256;

            Plans plans = part.GetPlans(CascadeType::Middle, command_stream::BlockConfig{ 16u, 16u }, &prevBuffer, 2);
            CHECK(plans.size() == 2);
            CheckPlansParams params;
            params.m_Any.push_back([&](const PlanDesc& plan) {
                return plan.m_InputSram->m_StripeShape == TensorShape{ 1, 56, 56, 256 } &&
                       plan.m_InputSram->m_NumStripes == 1 &&
                       plan.m_WeightsSram->m_StripeShape ==
                           TensorShape{ 3, 3, 128, 1 } &&    // This is 32 but 4x because of strided
                       plan.m_WeightsSram->m_NumStripes == 2 &&
                       plan.m_Mce->m_BlockConfig == command_stream::BlockConfig{ 16u, 16u } &&
                       plan.m_Mce->m_InputStripeShape == TensorShape{ 1, 56, 56, 256 } &&
                       plan.m_Mce->m_WeightsStripeShape == TensorShape{ 3, 3, 128, 1 } &&
                       plan.m_Mce->m_OutputStripeShape == TensorShape{ 1, 56, 56, 32 } &&
                       // The following Part is another McePart, so we'll use the plan which includes a Passthrough PLE.
                       plan.m_Output->m_Location == Location::Sram &&
                       plan.m_Output->m_StripeShape == TensorShape{ 1, 56, 56, 64 } && plan.m_Output->m_NumStripes == 1;
            });
            CheckPlans(plans, params);
        }

        ///  7. McePart CONVOLUTION 56,56,64 -> 56,56,128. Stride 1x1. Padding 0,0. Weights 1,1,64,128.
        SECTION("Part 7")
        {
            TensorShape inputShape{ 1, 56, 56, 64 };
            TensorShape outputShape{ 1, 56, 56, 128 };
            TensorShape weightShape{ 1, 1, 64, 128 };
            McePart part = BuildPart(inputShape, outputShape, weightShape, command_stream::MceOperation::CONVOLUTION,
                                     Stride{ 1, 1 }, 0, 0, compOpt, caps);
            prevBuffer.m_TensorShape = inputShape;
            prevBuffer.m_StripeShape = TensorShape{ 1, 56, 56, 64 };
            prevBuffer.m_SizeInBytes = 56 * 56 * 64;

            Plans plans = part.GetPlans(CascadeType::Middle, command_stream::BlockConfig{ 16u, 16u }, &prevBuffer, 2);
            CHECK(plans.size() == 2);
            CheckPlansParams params;
            params.m_Any.push_back([&](const PlanDesc& plan) {
                return plan.m_InputSram->m_StripeShape == TensorShape{ 1, 56, 56, 64 } &&
                       plan.m_InputSram->m_NumStripes == 1 &&
                       plan.m_WeightsSram->m_StripeShape == TensorShape{ 1, 1, 64, 32 } &&
                       plan.m_WeightsSram->m_NumStripes == 2 &&
                       plan.m_Mce->m_BlockConfig == command_stream::BlockConfig{ 16u, 16u } &&
                       plan.m_Mce->m_InputStripeShape == TensorShape{ 1, 56, 56, 64 } &&
                       plan.m_Mce->m_WeightsStripeShape == TensorShape{ 1, 1, 64, 32 } &&
                       plan.m_Mce->m_OutputStripeShape == TensorShape{ 1, 56, 56, 32 } &&
                       // The following Part is another McePart, so we'll use the plan which includes a Passthrough PLE.
                       plan.m_Output->m_Location == Location::Sram &&
                       plan.m_Output->m_StripeShape == TensorShape{ 1, 56, 56, 128 } &&
                       plan.m_Output->m_NumStripes == 1;
            });
            CheckPlans(plans, params);
        }

        ///  8. McePart DEPTHWISE_CONVOLUTION 56,56,128 -> 56,56,128. Stride 1x1. Padding 1,1. Weights 3,3,128,1.
        SECTION("Part 8")
        {
            TensorShape inputShape{ 1, 56, 56, 128 };
            TensorShape outputShape{ 1, 56, 56, 128 };
            TensorShape weightShape{ 3, 3, 128, 1 };
            McePart part =
                BuildPart(inputShape, outputShape, weightShape, command_stream::MceOperation::DEPTHWISE_CONVOLUTION,
                          Stride{ 1, 1 }, 1, 1, compOpt, caps);
            prevBuffer.m_TensorShape = inputShape;
            prevBuffer.m_StripeShape = TensorShape{ 1, 56, 56, 128 };
            prevBuffer.m_SizeInBytes = 56 * 56 * 128;

            Plans plans = part.GetPlans(CascadeType::Middle, command_stream::BlockConfig{ 16u, 16u }, &prevBuffer, 2);
            CHECK(plans.size() == 2);
            CheckPlansParams params;
            params.m_Any.push_back([&](const PlanDesc& plan) {
                return plan.m_InputSram->m_StripeShape == TensorShape{ 1, 56, 56, 128 } &&
                       plan.m_InputSram->m_NumStripes == 1 &&
                       plan.m_WeightsSram->m_StripeShape == TensorShape{ 3, 3, 32, 1 } &&
                       plan.m_WeightsSram->m_NumStripes == 2 &&
                       plan.m_Mce->m_BlockConfig == command_stream::BlockConfig{ 16u, 16u } &&
                       plan.m_Mce->m_InputStripeShape == TensorShape{ 1, 56, 56, 128 } &&
                       plan.m_Mce->m_WeightsStripeShape == TensorShape{ 3, 3, 32, 1 } &&
                       plan.m_Mce->m_OutputStripeShape == TensorShape{ 1, 56, 56, 32 } &&
                       // The following Part is another McePart, so we'll use the plan which includes a Passthrough PLE.
                       plan.m_Output->m_Location == Location::Sram &&
                       plan.m_Output->m_StripeShape == TensorShape{ 1, 56, 56, 128 } &&
                       plan.m_Output->m_NumStripes == 1;
            });
            CheckPlans(plans, params);
        }

        ///  9. McePart CONVOLUTION 56,56,128 -> 56,56,128. Stride 1x1. Padding 0,0. Weights 1,1,128,128.
        SECTION("Part 9")
        {
            TensorShape inputShape{ 1, 56, 56, 128 };
            TensorShape outputShape{ 1, 56, 56, 128 };
            TensorShape weightShape{ 1, 1, 128, 128 };
            McePart part = BuildPart(inputShape, outputShape, weightShape, command_stream::MceOperation::CONVOLUTION,
                                     Stride{ 1, 1 }, 0, 0, compOpt, caps);
            prevBuffer.m_TensorShape = inputShape;
            prevBuffer.m_StripeShape = TensorShape{ 1, 56, 56, 128 };
            prevBuffer.m_SizeInBytes = 56 * 56 * 128;

            Plans plans = part.GetPlans(CascadeType::Middle, command_stream::BlockConfig{ 16u, 16u }, &prevBuffer, 2);
            CHECK(plans.size() == 2);
            CheckPlansParams params;
            params.m_Any.push_back([&](const PlanDesc& plan) {
                return plan.m_InputSram->m_StripeShape == TensorShape{ 1, 56, 56, 128 } &&
                       plan.m_InputSram->m_NumStripes == 1 &&
                       plan.m_WeightsSram->m_StripeShape == TensorShape{ 1, 1, 128, 32 } &&
                       plan.m_WeightsSram->m_NumStripes == 2 &&
                       plan.m_Mce->m_BlockConfig == command_stream::BlockConfig{ 16u, 16u } &&
                       plan.m_Mce->m_InputStripeShape == TensorShape{ 1, 56, 56, 128 } &&
                       plan.m_Mce->m_WeightsStripeShape == TensorShape{ 1, 1, 128, 32 } &&
                       plan.m_Mce->m_OutputStripeShape == TensorShape{ 1, 56, 56, 32 } &&
                       // The following Part is a FusedPlePart, so we'll use the plan which ends at PleInputSram (and doesn't include a Passthrough PLE)
                       plan.m_Output->m_Location == Location::PleInputSram &&
                       plan.m_Output->m_StripeShape == TensorShape{ 1, 56, 56, 32 } && plan.m_Output->m_NumStripes == 0;
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
        prevBuffer.m_Location         = Location::Sram;
        prevBuffer.m_Format           = CascadingBufferFormat::NHWCB;
        prevBuffer.m_QuantizationInfo = { 0, 1.0f };
        prevBuffer.m_Order            = TraversalOrder::Xyz;

        ///  2. McePart CONVOLUTION 112,112,27 -> 112,112,32. Stride 2x2. Padding 1,1. Weights 3,3,3,32.
        SECTION("Part 2")
        {
            // This is part of a strategy 0 cascade.
            // There are some differences between our results and the prototype compiler because this is a strided convolution and
            // the prototype compiler assumes that the DMA is used to do the interleaving of the IFM, which leads to a different IFM
            // depth (12 vs 27).

            TensorShape inputShape{ 1, 112, 112, 27 };
            TensorShape outputShape{ 1, 112, 112, 32 };
            TensorShape weightShape{ 3, 3, 3, 32 };
            McePart part = BuildPart(inputShape, outputShape, weightShape, command_stream::MceOperation::CONVOLUTION,
                                     Stride{ 2u, 2u }, 1, 1, compOpt, caps);
            prevBuffer.m_TensorShape = inputShape;
            prevBuffer.m_StripeShape = TensorShape{ 1, 8, 112, 32 };
            prevBuffer.m_NumStripes  = 3;    // 3 required for neighbouring data (kernel has height 3)
            prevBuffer.m_SizeInBytes = 8 * 112 * 32 * 3;

            Plans plans = part.GetPlans(CascadeType::Middle, command_stream::BlockConfig{ 32u, 8u }, &prevBuffer, 1);
            CheckPlansParams params;
            params.m_Any.push_back([&](const PlanDesc& plan) {
                return plan.m_InputSram->m_StripeShape == TensorShape{ 1, 8, 112, 32 } &&
                       plan.m_InputSram->m_NumStripes == 3 &&
                       plan.m_WeightsSram->m_StripeShape ==
                           TensorShape{ 3, 3, 32, 32 } &&    // 64 input channels due to striding
                       plan.m_WeightsSram->m_NumStripes == 1 &&
                       plan.m_Mce->m_BlockConfig == command_stream::BlockConfig{ 32u, 8u } &&
                       plan.m_Mce->m_InputStripeShape == TensorShape{ 1, 8, 112, 32 } &&
                       plan.m_Mce->m_WeightsStripeShape == TensorShape{ 3, 3, 32, 32 } &&
                       plan.m_Mce->m_OutputStripeShape == TensorShape{ 1, 8, 112, 32 } &&
                       // The following Part is another McePart, so we'll use the plan which includes a Passthrough PLE.
                       plan.m_Output->m_Location == Location::Sram &&
                       plan.m_Output->m_StripeShape == TensorShape{ 1, 8, 112, 32 } &&
                       plan.m_Output->m_NumStripes ==
                           3;    // Following McePart has a kernel with height 3 so needs neighbouring stripes
            });
            CheckPlans(plans, params);
        }

        ///  3. McePart DEPTHWISE_CONVOLUTION 112,112,32 -> 112,112,32. Stride 1x1. Padding 1,1. Weights 3,3,32,1.
        SECTION("Part 3")
        {
            // This is part of the same strategy 0 cascade

            TensorShape inputShape{ 1, 112, 112, 32 };
            TensorShape outputShape{ 1, 112, 112, 32 };
            TensorShape weightShape{ 3, 3, 32, 1 };
            McePart part =
                BuildPart(inputShape, outputShape, weightShape, command_stream::MceOperation::DEPTHWISE_CONVOLUTION,
                          Stride{ 1, 1 }, 1, 1, compOpt, caps);
            prevBuffer.m_TensorShape = inputShape;
            prevBuffer.m_StripeShape = TensorShape{ 1, 8, 112, 32 };
            prevBuffer.m_NumStripes  = 3;
            prevBuffer.m_SizeInBytes = 8 * 112 * 32 * 3;

            Plans plans = part.GetPlans(CascadeType::Middle, command_stream::BlockConfig{ 32u, 8u }, &prevBuffer, 1);
            CheckPlansParams params;
            params.m_Any.push_back([&](const PlanDesc& plan) {
                return plan.m_InputSram->m_StripeShape == TensorShape{ 1, 8, 112, 32 } &&
                       plan.m_InputSram->m_NumStripes == 3 &&
                       // The prototype compiler has M=8 here that appaears to be a bug on their end as this doesn't make sense.
                       plan.m_WeightsSram->m_StripeShape == TensorShape{ 3, 3, 32, 1 } &&
                       plan.m_WeightsSram->m_NumStripes == 1 &&
                       plan.m_Mce->m_BlockConfig == command_stream::BlockConfig{ 32u, 8u } &&
                       plan.m_Mce->m_InputStripeShape == TensorShape{ 1, 8, 112, 32 } &&
                       plan.m_Mce->m_WeightsStripeShape == TensorShape{ 3, 3, 32, 1 } &&
                       plan.m_Mce->m_OutputStripeShape == TensorShape{ 1, 8, 112, 32 } &&
                       // The following Part is another McePart, so we'll use the plan which includes a Passthrough PLE.
                       plan.m_Output->m_Location == Location::Sram &&
                       plan.m_Output->m_StripeShape == TensorShape{ 1, 8, 112, 32 } &&
                       plan.m_Output->m_NumStripes ==
                           1;    // The following McePart has a kernel with height 1 so no neighbouring stripes are needed
            });
            CheckPlans(plans, params);
        }

        ///  4. McePart CONVOLUTION 112,112,32 -> 112,112,64. Stride 1x1. Padding 0,0. Weights 1,1,32,64.
        SECTION("Part 4")
        {
            // Part of strategy 0 cascade

            TensorShape inputShape{ 1, 112, 112, 32 };
            TensorShape outputShape{ 1, 112, 112, 64 };
            TensorShape weightShape{ 1, 1, 32, 64 };
            McePart part = BuildPart(inputShape, outputShape, weightShape, command_stream::MceOperation::CONVOLUTION,
                                     Stride{ 1, 1 }, 0, 0, compOpt, caps);
            prevBuffer.m_TensorShape = inputShape;
            prevBuffer.m_StripeShape = TensorShape{ 1, 8, 112, 32 };
            prevBuffer.m_SizeInBytes = 8 * 112 * 32;
            prevBuffer.m_NumStripes  = 1;

            Plans plans = part.GetPlans(CascadeType::Middle, command_stream::BlockConfig{ 32u, 8u }, &prevBuffer, 1);
            CheckPlansParams params;
            params.m_Any.push_back([&](const PlanDesc& plan) {
                return plan.m_InputSram->m_StripeShape == TensorShape{ 1, 8, 112, 32 } &&
                       plan.m_InputSram->m_NumStripes == 1 &&
                       plan.m_WeightsSram->m_StripeShape == TensorShape{ 1, 1, 32, 64 } &&
                       plan.m_WeightsSram->m_NumStripes == 1 &&
                       plan.m_Mce->m_BlockConfig == command_stream::BlockConfig{ 32u, 8u } &&
                       plan.m_Mce->m_InputStripeShape == TensorShape{ 1, 8, 112, 32 } &&
                       plan.m_Mce->m_WeightsStripeShape == TensorShape{ 1, 1, 32, 64 } &&
                       // The prototype compiler disagrees - because it includes the following interleave kernel
                       plan.m_Mce->m_OutputStripeShape == TensorShape{ 1, 8, 112, 64 } &&
                       // The following Part is a FusedPlePart, so we'll use the plan which ends at PleInputSram (and doesn't include a Passthrough PLE)
                       plan.m_Output->m_Location == Location::PleInputSram &&
                       plan.m_Output->m_StripeShape == TensorShape{ 1, 8, 112, 64 } && plan.m_Output->m_NumStripes == 0;
            });
            CheckPlans(plans, params);
        }

        ///  6. McePart DEPTHWISE_CONVOLUTION 56,56,256 -> 56,56,64. Stride 2x2. Padding 1,1. Weights 3,3,64,1.
        SECTION("Part 6")
        {
            // This is a lonely strategy 6 part.
            // There are differences between our results and the prototype compiler because this is a strided convolution and
            // the prototype compiler assumes that the DMA is used to do the interleaving of the IFM, which leads to a different IFM
            // shape.

            TensorShape inputShape{ 1, 56, 56, 256 };
            TensorShape outputShape{ 1, 56, 56, 64 };
            TensorShape weightShape{ 3, 3, 64, 1 };
            McePart part =
                BuildPart(inputShape, outputShape, weightShape, command_stream::MceOperation::DEPTHWISE_CONVOLUTION,
                          Stride{ 2, 2 }, 1, 1, compOpt, caps);

            Plans plans = part.GetPlans(CascadeType::Lonely, command_stream::BlockConfig{}, nullptr, 2);
            CheckPlansParams params;
            params.m_Any.push_back([&](const PlanDesc& plan) {
                return plan.m_InputSram->m_StripeShape ==
                           TensorShape{
                               1, 16, 16, 32
                           } &&    //TODO: not sure about this shape coming from PC (had to halve due to interleave differnces
                       plan.m_InputSram->m_NumStripes ==
                           3 &&    //TODO: not sure about this coming from PC. How does this work for strategy 6 with boundary slots?
                       plan.m_WeightsSram->m_StripeShape ==
                           TensorShape{
                               3, 3, 32, 1
                           } &&    // This is 8 but 4x because of strided. Prototype compiler splits weights here as well.
                       plan.m_WeightsSram->m_NumStripes == 2 &&
                       // The prototype compiler has a block config of {8, 16}
                       // We only generate stripe shapes which match block configs so must choose 16x16
                       plan.m_Mce->m_BlockConfig == command_stream::BlockConfig{ 16u, 16u } &&
                       plan.m_Mce->m_InputStripeShape == TensorShape{ 1, 16, 16, 32 } &&
                       plan.m_Mce->m_WeightsStripeShape == TensorShape{ 3, 3, 32, 1 } &&
                       plan.m_Mce->m_OutputStripeShape == TensorShape{ 1, 16, 16, 8 } &&
                       // This is a lonely Part, so we'll use the plan which includes a Passthrough PLE.
                       plan.m_Output->m_Location == Location::Sram &&
                       plan.m_Output->m_StripeShape == TensorShape{ 1, 16, 16, 16 } && plan.m_Output->m_NumStripes == 2;
            });
            CheckPlans(plans, params);
        }

        ///  7. McePart CONVOLUTION 56,56,64 -> 56,56,128. Stride 1x1. Padding 0,0. Weights 1,1,64,128.
        SECTION("Part 7")
        {
            // This is the start of a new strategy 0 cascade

            TensorShape inputShape{ 1, 56, 56, 64 };
            TensorShape outputShape{ 1, 56, 56, 128 };
            TensorShape weightShape{ 1, 1, 64, 128 };
            McePart part = BuildPart(inputShape, outputShape, weightShape, command_stream::MceOperation::CONVOLUTION,
                                     Stride{ 1, 1 }, 0, 0, compOpt, caps);

            Plans plans = part.GetPlans(CascadeType::Beginning, command_stream::BlockConfig{}, nullptr, 1);
            CheckPlansParams params;
            params.m_Any.push_back([&](const PlanDesc& plan) {
                return plan.m_InputSram->m_StripeShape == TensorShape{ 1, 8, 56, 64 } &&
                       plan.m_InputSram->m_NumStripes == 2 &&
                       plan.m_WeightsSram->m_StripeShape == TensorShape{ 1, 1, 64, 128 } &&
                       plan.m_WeightsSram->m_NumStripes == 1 &&
                       plan.m_Mce->m_BlockConfig == command_stream::BlockConfig{ 32u, 8u } &&
                       plan.m_Mce->m_InputStripeShape == TensorShape{ 1, 8, 56, 64 } &&
                       plan.m_Mce->m_WeightsStripeShape == TensorShape{ 1, 1, 64, 128 } &&
                       plan.m_Mce->m_OutputStripeShape == TensorShape{ 1, 8, 56, 128 } &&
                       // The following Part is another McePart, so we'll use the plan which includes a Passthrough PLE.
                       plan.m_Output->m_Location == Location::Sram &&
                       plan.m_Output->m_StripeShape == TensorShape{ 1, 8, 56, 128 } &&
                       plan.m_Output->m_NumStripes ==
                           3;    // The following McePart has a kernel with height 3 so neighbouring stripes are needed
            });
            CheckPlans(plans, params);
        }

        ///  8. McePart DEPTHWISE_CONVOLUTION 56,56,128 -> 56,56,128. Stride 1x1. Padding 1,1. Weights 3,3,128,1.
        SECTION("Part 8")
        {
            // This is the end of a strategy 0 cascade

            TensorShape inputShape{ 1, 56, 56, 128 };
            TensorShape outputShape{ 1, 56, 56, 128 };
            TensorShape weightShape{ 3, 3, 128, 1 };
            McePart part =
                BuildPart(inputShape, outputShape, weightShape, command_stream::MceOperation::DEPTHWISE_CONVOLUTION,
                          Stride{ 1, 1 }, 1, 1, compOpt, caps);
            prevBuffer.m_TensorShape = inputShape;
            prevBuffer.m_StripeShape = TensorShape{ 1, 8, 56, 128 };
            prevBuffer.m_SizeInBytes = 8 * 56 * 128;
            prevBuffer.m_NumStripes  = 3;

            Plans plans = part.GetPlans(CascadeType::End, command_stream::BlockConfig{ 32u, 8u }, &prevBuffer, 1);
            CheckPlansParams params;
            params.m_Any.push_back([&](const PlanDesc& plan) {
                return plan.m_InputSram->m_StripeShape == TensorShape{ 1, 8, 56, 128 } &&
                       plan.m_InputSram->m_NumStripes == 3 &&
                       plan.m_WeightsSram->m_StripeShape == TensorShape{ 3, 3, 128, 1 } &&
                       plan.m_WeightsSram->m_NumStripes == 1 &&
                       plan.m_Mce->m_BlockConfig == command_stream::BlockConfig{ 32u, 8u } &&
                       plan.m_Mce->m_InputStripeShape == TensorShape{ 1, 8, 56, 128 } &&
                       plan.m_Mce->m_WeightsStripeShape == TensorShape{ 3, 3, 128, 1 } &&
                       plan.m_Mce->m_OutputStripeShape == TensorShape{ 1, 8, 56, 128 } &&
                       // This is the end of a cascade, so we'll use the plan which includes a Passthrough PLE.
                       plan.m_Output->m_Location == Location::Sram &&
                       plan.m_Output->m_StripeShape == TensorShape{ 1, 8, 56, 128 } &&
                       plan.m_Output->m_NumStripes == 2;    // End of cascade => double buffered
            });
            CheckPlans(plans, params);
        }

        ///  9. McePart CONVOLUTION 56,56,128 -> 56,56,128. Stride 1x1. Padding 0,0. Weights 1,1,128,128.
        SECTION("Part 9")
        {
            // Beginning of short strategy 1 cascade
            // This is a bit different to the prototype compiler's results, because they take the output back to SRAM
            // and don't include an interleave, whereas we leave the output in PleInputSram for a following interleave.

            TensorShape inputShape{ 1, 56, 56, 128 };
            TensorShape outputShape{ 1, 56, 56, 128 };
            TensorShape weightShape{ 1, 1, 128, 128 };
            McePart part = BuildPart(inputShape, outputShape, weightShape, command_stream::MceOperation::CONVOLUTION,
                                     Stride{ 1, 1 }, 0, 0, compOpt, caps);

            Plans plans = part.GetPlans(CascadeType::Beginning, command_stream::BlockConfig{}, nullptr, 2);
            CheckPlansParams params;
            params.m_Any.push_back([&](const PlanDesc& plan) {
                bool b = true;
                b &= plan.m_InputSram->m_StripeShape == TensorShape{ 1, 56, 56, 128 };
                b &= plan.m_InputSram->m_NumStripes == 1;
                b &= plan.m_WeightsSram->m_StripeShape == TensorShape{ 1, 1, 128, 8 };
                b &= plan.m_WeightsSram->m_NumStripes == 2;
                b &= plan.m_Mce->m_BlockConfig == command_stream::BlockConfig{ 32u, 8u };
                b &= plan.m_Mce->m_InputStripeShape == TensorShape{ 1, 56, 56, 128 };
                b &= plan.m_Mce->m_WeightsStripeShape == TensorShape{ 1, 1, 128, 8 };
                // The prototype compiler has 32 width and height as it includes in the following interleave kernel.
                b &= plan.m_Mce->m_OutputStripeShape == TensorShape{ 1, 56, 56, 8 };
                // The following Part is a FusedPlePart, so we'll use the plan which ends at PleInputSram (and doesn't include a Passthrough PLE)
                b &= plan.m_Output->m_Location == Location::PleInputSram;
                b &= plan.m_Output->m_StripeShape == TensorShape{ 1, 56, 56, 8 };
                b &= plan.m_Output->m_NumStripes == 0;
                return b;
            });
            CheckPlans(plans, params);
        }
    }
}

TEST_CASE("McePart GetPlans Upsampling")
{
    GIVEN("An McePart for a convolution with upsampling")
    {
        const CompilationOptions compOpt;
        const HardwareCapabilities caps = GetEthosN78HwCapabilities();

        TensorShape tsIn  = { 1, 64, 64, 16 };
        TensorShape tsOut = { 1, 128, 128, 16 };
        McePart part = BuildPart(tsIn, tsOut, { 1, 1, 16, 16 }, command_stream::MceOperation::CONVOLUTION, Stride{}, 0,
                                 0, 2, command_stream::UpsampleType::NEAREST_NEIGHBOUR, compOpt, caps);

        WHEN("Asked to generate Lonely plans")
        {
            Plans plans = part.GetPlans(CascadeType::Lonely, command_stream::BlockConfig{}, nullptr, 0);
            SavePlansToDot(plans, "McePart GetPlans Upsampling Lonely");

            THEN("The plans are all valid and have stripe configs that are consistent with upsampling and there is a "
                 "strategy 0 plan")
            {
                CheckPlansParams params;
                params.m_InputShape    = tsIn;
                params.m_OutputShape   = tsOut;
                params.m_UpscaleFactor = 2;
                params.m_UpsampleType  = command_stream::UpsampleType::NEAREST_NEIGHBOUR;
                params.m_All           = [&](const PlanDesc& plan) {
                    CHECK(plan.m_PleInputSram->m_StripeShape[1] == 2 * plan.m_InputSram->m_StripeShape[1]);
                    CHECK(plan.m_PleInputSram->m_StripeShape[2] == 2 * plan.m_InputSram->m_StripeShape[2]);
                };
                params.m_Any.push_back([&](const PlanDesc& plan) {
                    return plan.m_InputSram->m_StripeShape == TensorShape{ 1, 16, 64, 16 } &&
                           plan.m_PleInputSram->m_StripeShape == TensorShape{ 1, 32, 128, 16 };
                });
                CheckPlans(plans, params);
            }
        }

        WHEN("Asked to generate Middle plans")
        {
            Buffer prevBuffer;
            prevBuffer.m_Lifetime         = Lifetime::Cascade;
            prevBuffer.m_Location         = Location::Sram;
            prevBuffer.m_Format           = CascadingBufferFormat::NHWCB;
            prevBuffer.m_QuantizationInfo = { 0, 1.0f };
            prevBuffer.m_TensorShape      = tsIn;
            prevBuffer.m_StripeShape      = TensorShape{ 1, 8, 64, 16 };
            prevBuffer.m_Order            = TraversalOrder::Xyz;
            prevBuffer.m_SizeInBytes      = 8 * 64 * 16 * 1;
            prevBuffer.m_NumStripes       = 1;

            Plans plans = part.GetPlans(CascadeType::Middle, command_stream::BlockConfig{ 32u, 8u }, &prevBuffer, 1);
            SavePlansToDot(plans, "McePart GetPlans Upsampling Middle");

            THEN("The plans are all valid and have stripe configs that are consistent with upsampling and there is a "
                 "strategy 0 plan")
            {
                CheckPlansParams params;
                params.m_InputShape    = tsIn;
                params.m_OutputShape   = tsOut;
                params.m_UpscaleFactor = 2;
                params.m_UpsampleType  = command_stream::UpsampleType::NEAREST_NEIGHBOUR;
                params.m_All           = [&](const PlanDesc& plan) {
                    CHECK(plan.m_PleInputSram->m_StripeShape[1] == 2 * plan.m_InputSram->m_StripeShape[1]);
                    CHECK(plan.m_PleInputSram->m_StripeShape[2] == 2 * plan.m_InputSram->m_StripeShape[2]);
                };
                params.m_Any.push_back([&](const PlanDesc& plan) {
                    return plan.m_InputSram->m_StripeShape == TensorShape{ 1, 8, 64, 16 } &&
                           plan.m_PleInputSram->m_StripeShape == TensorShape{ 1, 16, 128, 16 };
                });
                CheckPlans(plans, params);
            }
        }
    }
}
