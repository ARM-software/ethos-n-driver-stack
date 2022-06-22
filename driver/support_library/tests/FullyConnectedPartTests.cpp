//
// Copyright © 2021-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "CapabilitiesInternal.hpp"
#include "GlobalParameters.hpp"
#include "TestUtils.hpp"
#include "Utils.hpp"
#include "cascading/Cascading.hpp"
#include "cascading/FullyConnectedPart.hpp"
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
        CHECK(desc.m_InputDram->m_Format == CascadingBufferFormat::NHWC);
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
        CHECK(desc.m_InputDram->m_SizeInBytes == utils::GetNumElements(desc.m_InputDram->m_TensorShape));
        CHECK(desc.m_InputDram->m_NumStripes == 0);
        CHECK(desc.m_InputDram->m_EncodedWeights == nullptr);
    }
}

void CheckInputSram(PlanDesc& desc, const CheckPlansParams& params)
{
    // However we interpret it as NHWCB so that it gets copied without conversion into SRAM.
    // We choose the smallest shape that will encompass all the data when it is interpreted in brick format.
    auto GetShapeContainingLinearElements = [](const TensorShape& brickGroupShape, uint32_t numElements) {
        const uint32_t brickGroupHeight           = brickGroupShape[1];
        const uint32_t brickGroupWidth            = brickGroupShape[2];
        const uint32_t brickGroupChannels         = brickGroupShape[3];
        const uint32_t patchHeight                = 4;
        const uint32_t patchWidth                 = 4;
        const uint32_t patchesPerBrickGroupHeight = brickGroupHeight / patchHeight;
        const uint32_t patchesPerBrickGroupWidth  = brickGroupWidth / patchWidth;
        const uint32_t patchesPerBrickGroup =
            patchesPerBrickGroupHeight * patchesPerBrickGroupWidth * brickGroupChannels;

        // If there are less than one bricks worth of elements then we can have a tensor with a single patch in XY
        // and up to 16 channels.
        // If there are between one and two bricks worth of elements then we can have a tensor with a column of two
        // patches in XY and 16 channels. Note we always need 16 channels in this case as the first brick is full.
        // If there are between two and four bricks worth of elements then we can have a tensor of a full brick group.
        // Again note we always need 16 channels in this case as the first two brick are full.
        // If we have more than four bricks of elements then we add brick groups behind the first one (i.e. stacking
        // along depth). The number of channels in the final brick group may be less than 16 if there is less
        // than a full bricks worth of elements in that final brick group.
        const uint32_t numPatches = utils::DivRoundUp(numElements, patchWidth * patchHeight);
        const uint32_t reinterpretedWidth =
            numPatches <= brickGroupChannels * patchesPerBrickGroupHeight ? patchWidth : brickGroupWidth;
        const uint32_t reinterpretedHeight = numPatches <= brickGroupChannels ? patchHeight : brickGroupHeight;
        const uint32_t numFullBrickGroups  = numPatches / patchesPerBrickGroup;
        const uint32_t reinterpretedChannels =
            brickGroupChannels * numFullBrickGroups + std::min(brickGroupChannels, numPatches % patchesPerBrickGroup);
        return TensorShape{ 1, reinterpretedHeight, reinterpretedWidth, reinterpretedChannels };
    };
    // Check properties of Input SRAM buffer
    CHECK(desc.m_InputSram->m_Location == Location::Sram);
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
        CHECK(desc.m_InputSram->m_TensorShape ==
              GetShapeContainingLinearElements({ 1, 8, 8, 16 }, utils::GetNumElements(params.m_InputShape.value())));
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
        if (params.m_OperationIds)
        {
            CHECK(desc.m_InputDma->m_OperationIds == params.m_OperationIds.value());
        }
    }
}

void CheckWeightsDma(const CheckPlansParams& params, PlanDesc& desc)
{
    // Check properties of Weights DMA
    if (params.m_OperationIds)
    {
        CHECK(desc.m_WeightsDma->m_OperationIds == params.m_OperationIds.value());
    }
}

void CheckMce(const CheckPlansParams& params, PlanDesc& desc)
{
    // Check properties of Mce Op
    if (params.m_OperationIds)
    {
        CHECK(desc.m_Mce->m_OperationIds == params.m_OperationIds.value());
    }
    if (params.m_MceOp)
    {
        CHECK(desc.m_Mce->m_Op == params.m_MceOp.value());
    }
    CHECK(desc.m_Mce->m_BlockConfig == command_stream::BlockConfig{ 8u, 8u });
    // m_Algo, m_InputStripeShape, m_OutputStripeShape, m_WeightsStripeShape, m_Order will depend on the streaming strategy, and so cannot be checked generically
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
        CHECK(desc.m_Ple->m_Op == command_stream::PleOperation::PASSTHROUGH);
        CHECK(desc.m_Ple->m_BlockConfig == command_stream::BlockConfig{ 8u, 8u });
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

/// Checks that FullyConnectedPart::GetPlans returns sensible plans
TEST_CASE("FullyConnectedPart GetPlans", "[slow]")
{
    GIVEN("An FullyConnectedPart")
    {
        const CompilationOptions compOpt;
        EstimationOptions estOps;
        const HardwareCapabilities caps = GetEthosN78HwCapabilities(EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO);

        const PartId partId  = 0;
        TensorShape tsInOrig = { 1, 1, 1, 2048 };
        TensorShape tsIn     = { 1, 8, 8, 32 };
        TensorShape tsOut    = { 1, 1, 1, 1024 };
        const std::vector<uint8_t> weights(1 * 1 * 2048 * 1024, 0);
        const std::vector<int32_t> bias(1024, 0);
        const QuantizationInfo inputQuantInfo(0, 1.0f);
        const QuantizationInfo outputQuantInfo(0, 1.0f);
        const TensorInfo weightsTensorInfo{ TensorShape{ 1, 1, 2048, 1024 }, DataType::UINT8_QUANTIZED,
                                            DataFormat::HWIO, QuantizationInfo(0, 0.9f) };
        const TensorInfo biasTensorInfo({ 1, 1, 1, 1024 });
        const std::set<uint32_t> operationIds = { 1, 2, 3 };
        FullyConnectedPart part(partId, tsInOrig, tsIn, tsOut, inputQuantInfo, outputQuantInfo, weightsTensorInfo,
                                weights, biasTensorInfo, bias, estOps, compOpt, caps, operationIds,
                                ethosn::command_stream::DataType::U8, ethosn::command_stream::DataType::U8);

        WHEN("Asked to generate plans")
        {
            Plans plans = part.GetPlans(CascadeType::Lonely, command_stream::BlockConfig{}, nullptr, 1);
            SavePlansToDot(plans, "FullyConnected GetPlans");

            THEN("The plans are valid and contain at least one plan with the full IFM and full OFM")
            {
                CheckPlansParams params;
                params.m_InputLocation = PlanInputLocation::Dram;
                params.m_InputShape    = tsInOrig;
                params.m_Any.push_back([](const PlanDesc& plan) {
                    bool inputSramValid = plan.m_InputSram->m_StripeShape == TensorShape{ 1, 8, 8, 32 } &&
                                          plan.m_InputSram->m_Order == TraversalOrder::Zxy &&
                                          plan.m_InputSram->m_SizeInBytes == 8 * 8 * 32 &&
                                          plan.m_InputSram->m_NumStripes == 1;
                    bool weightsSramValid = plan.m_WeightsSram->m_StripeShape == TensorShape{ 1, 1, 2048, 1024 } &&
                                            plan.m_WeightsSram->m_Order == TraversalOrder::Xyz &&
                                            plan.m_WeightsSram->m_NumStripes == 1;
                    bool pleInputSramValid = plan.m_PleInputSram->m_StripeShape == TensorShape{ 1, 8, 8, 1024 } &&
                                             plan.m_PleInputSram->m_Order == TraversalOrder::Xyz &&
                                             plan.m_PleInputSram->m_SizeInBytes == 8 * 8 * 1024 &&
                                             plan.m_PleInputSram->m_NumStripes == 0;
                    bool outputSramValid = plan.m_OutputSram->m_StripeShape == TensorShape{ 1, 8, 8, 1024 } &&
                                           plan.m_OutputSram->m_Order == TraversalOrder::Xyz &&
                                           plan.m_OutputSram->m_SizeInBytes == 8 * 8 * 1024 &&
                                           plan.m_OutputSram->m_NumStripes == 1;
                    bool mceValid = plan.m_Mce->m_Algo == CompilerMceAlgorithm::Direct &&
                                    plan.m_Mce->m_InputStripeShape == TensorShape{ 1, 8, 8, 32 } &&
                                    plan.m_Mce->m_OutputStripeShape == TensorShape{ 1, 8, 8, 1024 } &&
                                    plan.m_Mce->m_WeightsStripeShape == TensorShape{ 1, 1, 2048, 1024 } &&
                                    plan.m_Mce->m_Order == TraversalOrder::Xyz;
                    bool pleValid =
                        plan.m_Ple->m_InputStripeShapes == std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 1024 } } &&
                        plan.m_Ple->m_OutputStripeShape == TensorShape{ 1, 8, 8, 1024 };

                    return inputSramValid && weightsSramValid && pleInputSramValid && outputSramValid && mceValid &&
                           pleValid;
                });
                CheckPlans(plans, params);
            }
            THEN("The plans are valid and contain at least one plan with the full IFM and partial OFM")
            {
                CheckPlansParams params;
                params.m_InputLocation = PlanInputLocation::Dram;
                params.m_InputShape    = tsInOrig;
                params.m_Any.push_back([](const PlanDesc& plan) {
                    bool inputSramValid = plan.m_InputSram->m_StripeShape == TensorShape{ 1, 8, 8, 32 } &&
                                          plan.m_InputSram->m_Order == TraversalOrder::Zxy &&
                                          plan.m_InputSram->m_SizeInBytes == 8 * 8 * 32 &&
                                          plan.m_InputSram->m_NumStripes == 1;
                    bool weightsSramValid = plan.m_WeightsSram->m_StripeShape == TensorShape{ 1, 1, 2048, 16 } &&
                                            plan.m_WeightsSram->m_Order == TraversalOrder::Xyz &&
                                            plan.m_WeightsSram->m_NumStripes == 1;
                    bool pleInputSramValid = plan.m_PleInputSram->m_StripeShape == TensorShape{ 1, 8, 8, 16 } &&
                                             plan.m_PleInputSram->m_Order == TraversalOrder::Xyz &&
                                             plan.m_PleInputSram->m_SizeInBytes == 8 * 8 * 16 &&
                                             plan.m_PleInputSram->m_NumStripes == 0;
                    bool outputSramValid = plan.m_OutputSram->m_StripeShape == TensorShape{ 1, 8, 8, 16 } &&
                                           plan.m_OutputSram->m_Order == TraversalOrder::Xyz &&
                                           plan.m_OutputSram->m_SizeInBytes == 8 * 8 * 16 &&
                                           plan.m_OutputSram->m_NumStripes == 1;
                    bool mceValid = plan.m_Mce->m_Algo == CompilerMceAlgorithm::Direct &&
                                    plan.m_Mce->m_InputStripeShape == TensorShape{ 1, 8, 8, 32 } &&
                                    plan.m_Mce->m_OutputStripeShape == TensorShape{ 1, 8, 8, 16 } &&
                                    plan.m_Mce->m_WeightsStripeShape == TensorShape{ 1, 1, 2048, 16 } &&
                                    plan.m_Mce->m_Order == TraversalOrder::Xyz;
                    bool pleValid =
                        plan.m_Ple->m_InputStripeShapes == std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 16 } } &&
                        plan.m_Ple->m_OutputStripeShape == TensorShape{ 1, 8, 8, 16 };

                    return inputSramValid && weightsSramValid && pleInputSramValid && outputSramValid && mceValid &&
                           pleValid;
                });
                CheckPlans(plans, params);
            }
            THEN("The plans are valid and contain at least one plan with the partial IFM and partial OFM")
            {
                CheckPlansParams params;
                params.m_InputLocation = PlanInputLocation::Dram;
                params.m_InputShape    = tsInOrig;
                params.m_Any.push_back([](const PlanDesc& plan) {
                    bool inputSramValid = plan.m_InputSram->m_StripeShape == TensorShape{ 1, 8, 8, 16 } &&
                                          plan.m_InputSram->m_Order == TraversalOrder::Zxy &&
                                          plan.m_InputSram->m_SizeInBytes == 8 * 8 * 16 &&
                                          plan.m_InputSram->m_NumStripes == 1;
                    bool weightsSramValid = plan.m_WeightsSram->m_StripeShape == TensorShape{ 1, 1, 1024, 16 } &&
                                            plan.m_WeightsSram->m_Order == TraversalOrder::Xyz &&
                                            plan.m_WeightsSram->m_NumStripes == 1;
                    bool pleInputSramValid = plan.m_PleInputSram->m_StripeShape == TensorShape{ 1, 8, 8, 16 } &&
                                             plan.m_PleInputSram->m_Order == TraversalOrder::Xyz &&
                                             plan.m_PleInputSram->m_SizeInBytes == 8 * 8 * 16 &&
                                             plan.m_PleInputSram->m_NumStripes == 0;
                    bool outputSramValid = plan.m_OutputSram->m_StripeShape == TensorShape{ 1, 8, 8, 16 } &&
                                           plan.m_OutputSram->m_Order == TraversalOrder::Xyz &&
                                           plan.m_OutputSram->m_SizeInBytes == 8 * 8 * 16 &&
                                           plan.m_OutputSram->m_NumStripes == 1;
                    bool mceValid = plan.m_Mce->m_Algo == CompilerMceAlgorithm::Direct &&
                                    plan.m_Mce->m_InputStripeShape == TensorShape{ 1, 8, 8, 16 } &&
                                    plan.m_Mce->m_OutputStripeShape == TensorShape{ 1, 8, 8, 16 } &&
                                    plan.m_Mce->m_WeightsStripeShape == TensorShape{ 1, 1, 1024, 16 } &&
                                    plan.m_Mce->m_Order == TraversalOrder::Xyz;
                    bool pleValid =
                        plan.m_Ple->m_InputStripeShapes == std::vector<TensorShape>{ TensorShape{ 1, 8, 8, 16 } } &&
                        plan.m_Ple->m_OutputStripeShape == TensorShape{ 1, 8, 8, 16 };

                    return inputSramValid && weightsSramValid && pleInputSramValid && outputSramValid && mceValid &&
                           pleValid;
                });
                CheckPlans(plans, params);
            }
        }
    }
}
