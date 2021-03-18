//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Pass.hpp"
#include "SramAllocator.hpp"
#include "StrategiesCommon.hpp"

#include <ethosn_command_stream/PleOperation.hpp>

namespace ethosn
{
namespace support_library
{

class StandalonePleOperationNode;
class FormatConversionNode;

struct PleStrategySelectionParameter
{
    PleStrategySelectionParameter(SramAllocator::UserId userId,
                                  const HardwareCapabilities& capabilities,
                                  SramAllocator sramAllocator,
                                  const std::vector<SramTensorAllocation>& inputSramAllocations,
                                  const std::vector<TensorShape>& inputShapes,
                                  const TensorShape& outputShape,
                                  const std::vector<std::pair<bool, uint32_t>>& inputsStaticAndOffset,
                                  const TensorShape& splittableDims)
        : userId{ userId }
        , capabilities{ capabilities }
        , sramAllocator{ sramAllocator }
        , inputSramAllocations{ inputSramAllocations }
        , inputShapes{ inputShapes }
        , outputShape{ outputShape }
        , inputsStaticAndOffset{ inputsStaticAndOffset }
        , splittableDims{ splittableDims }

    {}
    // The sole purpose of this struct is to pack all the parameters given to ChooseAndSetupStrategy and
    // make sure all the arguments are read only. Hence the copy constructor and assignment operator are
    // deleted.
    PleStrategySelectionParameter(const PleStrategySelectionParameter&) = delete;
    PleStrategySelectionParameter& operator=(const PleStrategySelectionParameter&) = delete;

    SramAllocator::UserId userId;
    HardwareCapabilities capabilities;
    SramAllocator sramAllocator;
    const std::vector<SramTensorAllocation>& inputSramAllocations;
    std::vector<TensorShape> inputShapes;
    TensorShape outputShape;
    std::vector<std::pair<bool, uint32_t>> inputsStaticAndOffset;
    TensorShape splittableDims;
};

struct PleStrategySelectionReturnValue
{
    bool success{ false };
    SramAllocator sramAllocator;
    std::vector<SramTensorAllocation> inputSramAllocations;
    SramTensorAllocation pleSramAllocation;
    SramTensorAllocation outputSramAllocation;
};

/// A set of operations which are evaluated by Ethos-N in a single "pass" through the PLE only.
class PlePass : public Pass
{
public:
    static std::unique_ptr<PlePass> CreateGreedily(const HardwareCapabilities& capabilities,
                                                   size_t id,
                                                   Node* firstNode,
                                                   SramAllocator& SramAllocator);

    /// Creates a PlePass, consisting of just the given PLE operation.
    PlePass(const HardwareCapabilities& capabilities,
            size_t id,
            StandalonePleOperationNode* pleOperation,
            FormatConversionNode* postConversionNode,
            std::vector<SramTensorAllocation>& inputSramAllocations,
            SramTensorAllocation& pleSramAllocation,
            SramTensorAllocation& outputSramAllocation,
            BufferLocation outputLocation,
            uint32_t sramOffset);

    /// Generates this Pass by adding appropriate entries to the given command stream, memory map and buffer table.
    void Generate(command_stream::CommandStreamBuffer& cmdStream, BufferManager& bufferManager, bool dumpRam) override;

    DotAttributes GetDotAttributes() override;

    static PleStrategySelectionReturnValue
        ChooseAndSetupStrategy(const PleStrategySelectionParameter& pleStrategySelectionParameter);

private:
    PassStats GetStats(const EstimationOptions& estimationOptions) override;

    command_stream::PleOperation GetPleOperation() const;

    StandalonePleOperationNode* m_PleOperation;

    std::vector<SramTensorAllocation> m_InputSramAllocations;
    SramTensorAllocation m_PleSramAllocation;
    SramTensorAllocation m_OutputSramAllocation;
};

}    // namespace support_library
}    // namespace ethosn
