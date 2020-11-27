//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Pass.hpp"

#include <ethosn_command_stream/PleOperation.hpp>

namespace ethosn
{
namespace support_library
{

class StandalonePleOperationNode;
class FormatConversionNode;

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

    static bool ChooseAndSetupStrategy(const HardwareCapabilities& capabilities,
                                       SramAllocator& sramAllocator,
                                       std::vector<SramTensorAllocation>& inputOffsetsAndSizes,
                                       SramTensorAllocation& pleOffsetAndSize,
                                       SramTensorAllocation& outputOffsetAndSize,
                                       const std::vector<TensorShape>& inputShapes,
                                       const TensorShape& outputShape,
                                       const std::vector<std::pair<bool, uint32_t>>& inputsStaticAndOffset,
                                       const TensorShape& splittableDims);

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
