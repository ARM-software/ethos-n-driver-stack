//
// Copyright © 2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../nonCascading/PlePass.hpp"
#include "PartUtils.hpp"
#include "Plan.hpp"
#include "StandalonePlePart.hpp"
#include "StripeHelper.hpp"

#include <ethosn_utils/Macros.hpp>

#include <memory>

using namespace ethosn::command_stream;

namespace ethosn
{
namespace support_library
{
using namespace impl;

StandalonePlePart::StandalonePlePart(PartId id,
                                     const std::vector<TensorShape>& inputTensorShapes,
                                     const TensorShape& outputTensorShape,
                                     const std::vector<QuantizationInfo>& inputQuantizationInfos,
                                     const QuantizationInfo& outputQuantizationInfo,
                                     command_stream::PleOperation op,
                                     const EstimationOptions& estOpt,
                                     const CompilationOptions& compOpt,
                                     const HardwareCapabilities& capabilities,
                                     std::set<uint32_t> correspondingOperationIds,
                                     command_stream::DataType dataType)
    : BasePart(
          id, "StandalonePlePart", CompilerDataFormat::NONE, correspondingOperationIds, estOpt, compOpt, capabilities)
    , m_InputTensorShapes(inputTensorShapes)
    , m_OutputTensorShape(outputTensorShape)
    , m_InputQuantizationInfos(inputQuantizationInfos)
    , m_OutputQuantizationInfo(outputQuantizationInfo)
    , m_KernelOperation(op)
    , m_DataType(dataType)
{}

Plans StandalonePlePart::GetPlans(CascadeType cascadeType,
                                  ethosn::command_stream::BlockConfig blockConfig,
                                  Buffer* prevBuffer,
                                  uint32_t numWeightStripes) const
{
    ETHOSN_UNUSED(numWeightStripes);
    ETHOSN_UNUSED(blockConfig);

    Plans plans;

    TensorShape splittableDims = {};
    switch (m_KernelOperation)
    {
        case (command_stream::PleOperation::ADDITION):
        case (command_stream::PleOperation::ADDITION_RESCALE):
        {
            // ADDITION, ADDITION_RESCALE both have two inputs
            // that makes them not cascadable in the current
            // design where only SISO part is allowed in
            // a section.
            if (cascadeType == CascadeType::Lonely)
            {
                splittableDims = { 1, 1, 1, 1 };
            }
            else
            {
                return Plans{};
            }
            break;
        }
        case (command_stream::PleOperation::AVGPOOL_3X3_1_1_UDMA):
        {
            // AVGPOOL_3X3_1_1_UDMA: only split in D is allowed.
            // This makes it cascadalbe only if the whole input, output
            // tensors are fit into SRAM (in other words no split)
            splittableDims = { 0, 0, 0, 0 };

            if (cascadeType == CascadeType::Lonely)
            {
                splittableDims = { 0, 0, 0, 1 };
            }
            else if (cascadeType == CascadeType::Middle || cascadeType == CascadeType::End)
            {
                assert(prevBuffer != nullptr);

                // A cascadable plan is not possible if the stripe shape of the previous buffer
                // is smaller than the input tensor (in other words a full tensor plan is NOT
                // compatible with its predecessors).
                if (prevBuffer->m_StripeShape[1] < m_InputTensorShapes[0][1] ||
                    prevBuffer->m_StripeShape[2] < m_InputTensorShapes[0][2] ||
                    prevBuffer->m_StripeShape[3] < m_InputTensorShapes[0][3])
                {
                    return Plans{};
                }
            }
            break;
        }
        default:
        {
            assert(false);
            break;
        }
    }

    SramAllocator::UserId userId = 0;
    SramAllocator alloc(m_Capabilities.GetTotalSramSize() / m_Capabilities.GetNumberOfSrams());

    using Allocated = std::pair<bool, uint32_t>;

    // PLE kernel SRAM usage is considered before input/output buffers.
    Allocated allocated = alloc.Allocate(userId, m_Capabilities.GetMaxPleSize() / m_Capabilities.GetNumberOfSrams(),
                                         AllocationPreference::Start);

    if (!allocated.first)
    {
        return Plans{};
    }

    assert(m_InputQuantizationInfos.size() == m_InputTensorShapes.size());

    std::vector<std::pair<bool, uint32_t>> inputsStaticAndOffset;
    inputsStaticAndOffset.reserve(m_InputTensorShapes.size());

    std::vector<SramTensorAllocation> inputSramAllocations;
    inputSramAllocations.resize(m_InputTensorShapes.size());

    for (size_t i = 0; i < m_InputTensorShapes.size(); ++i)
    {
        inputsStaticAndOffset.push_back({ false, 0 });
    }

    PleStrategySelectionParameter pleStrategySelectionParameter{ userId,
                                                                 m_Capabilities,
                                                                 alloc,
                                                                 inputSramAllocations,
                                                                 m_InputTensorShapes,
                                                                 m_OutputTensorShape,
                                                                 inputsStaticAndOffset,
                                                                 splittableDims };

    // Lonely part: only needs to choose one best strategy
    PleStrategySelectionReturnValue rv = PlePass::ChooseAndSetupStrategy(pleStrategySelectionParameter);

    if (!rv.success)
    {
        return Plans{};
    }

    // Uses block configure (16, 16) which will be ignored
    // by a standalone PLE kernel.
    // Standalone PLE Ops either have an Atomic Lifetime or
    // form Lonely plans. In the latter case, Lifetime is
    // irrelevant because SRAM eviction will not take place.
    // Therefore, Lifetime::Atomic is used in all cases below.
    command_stream::BlockConfig blkConfig = { 16u, 16u };
    auto op                               = std::make_unique<PleOp>(Lifetime::Atomic, m_KernelOperation, blkConfig,
                                      static_cast<uint32_t>(m_InputTensorShapes.size()), m_InputTensorShapes,
                                      m_OutputTensorShape, m_DataType, true);

    OwnedOpGraph opGraph;
    PartInputMapping inputMappings;
    PartOutputMapping outputMappings;

    // only m_Output is used by AddPleToOpGraph
    NumMemoryStripes numMemoryStripes;
    numMemoryStripes.m_Output = rv.outputSramAllocation.numStripesInTile;

    std::vector<Buffer*> pleInputBuffers;
    pleInputBuffers.resize(m_InputTensorShapes.size());

    // PLE input buffers
    for (size_t i = 0; i < m_InputTensorShapes.size(); ++i)
    {
        pleInputBuffers[i] = AddPleInBuffer(opGraph, rv.inputSramAllocations.at(i).numStripesInTile,
                                            m_InputTensorShapes.at(i), rv.inputSramAllocations.at(i).stripeShape,
                                            m_InputQuantizationInfos.at(i), TraversalOrder::Xyz, Location::Sram);
    }

    // Output buffer
    auto outBufferAndPleOp = AddPleToOpGraph(
        opGraph, Lifetime::Atomic, TraversalOrder::Xyz, rv.outputSramAllocation.stripeShape, numMemoryStripes,
        std::move(op), m_OutputTensorShape, m_OutputQuantizationInfo, m_CorrespondingOperationIds);

    for (size_t i = 0; i < m_InputTensorShapes.size(); ++i)
    {
        opGraph.AddConsumer(pleInputBuffers[i], outBufferAndPleOp.second, static_cast<uint32_t>(i));
        inputMappings[pleInputBuffers[i]] = PartInputSlot{ m_PartId, static_cast<uint32_t>(i) };
    }

    outputMappings[outBufferAndPleOp.first] = PartOutputSlot{ m_PartId, 0 };
    AddNewPlan(std::move(inputMappings), std::move(outputMappings), std::move(opGraph), plans);

    return plans;
}

ethosn::support_library::DotAttributes StandalonePlePart::GetDotAttributes(DetailLevel detail) const
{
    DotAttributes result = BasePart::GetDotAttributes(detail);
    if (detail >= DetailLevel::High)
    {
        result.m_Label += "InputTensorShape = " + ArrayToString(m_InputTensorShapes) + "\n";
        result.m_Label += "OutputTensorShape = " + ToString(m_OutputTensorShape) + "\n";
        result.m_Label += "InputQuantizationInfo = " + ArrayToString(m_InputQuantizationInfos) + "\n";
        result.m_Label += "OutputQuantizationInfo = " + ToString(m_OutputQuantizationInfo) + "\n";
    }
    return result;
}

}    // namespace support_library
}    // namespace ethosn
