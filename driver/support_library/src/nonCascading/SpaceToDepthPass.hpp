//
// Copyright © 2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Pass.hpp"

namespace ethosn
{
namespace support_library
{

struct SpaceToDepthData
{
    uint32_t m_UsedEmcs;
    uint32_t m_Intermediate1Size;
    uint32_t m_Intermediate2Size;
};

class SpaceToDepthPass : public Pass
{
public:
    static std::pair<bool, uint32_t> ChooseAndAllocateSram(const NodeId& nodeId,
                                                           const HardwareCapabilities& capabilities,
                                                           const TensorShape& inputShape,
                                                           const TensorShape& outputShape,
                                                           SramAllocator& sramAllocator,
                                                           TensorShape& outIfmStripeShape,
                                                           SpaceToDepthData& outSpaceToDepthData);

    static std::unique_ptr<SpaceToDepthPass> CreateGreedily(const HardwareCapabilities& capabilities,
                                                            size_t id,
                                                            Node* firstNode,
                                                            SramAllocator& sramAllocator);

    SpaceToDepthPass(const HardwareCapabilities& capabilities,
                     size_t id,
                     Node* node,
                     uint32_t workBuffersSramOffset,
                     const TensorShape& ifmStripeShape,
                     const SpaceToDepthData& spaceToDepthData);

    /// Generates this Pass by adding appropriate entries to the given command stream, memory map and buffer table.
    void Generate(command_stream::CommandStreamBuffer& cmdStream, BufferManager& bufferManager, bool dumpRam) override;

    DotAttributes GetDotAttributes() override;

private:
    PassStats GetStats(const EstimationOptions& estimationOptions) override;

    Node* m_Node;

    uint32_t m_WorkBuffersSramOffset;
    TensorShape m_IfmStripeShape;
    SpaceToDepthData m_SpaceToDepthData;
};

}    // namespace support_library
}    // namespace ethosn
