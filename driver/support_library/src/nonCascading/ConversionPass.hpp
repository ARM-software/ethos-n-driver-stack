//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Pass.hpp"

namespace ethosn
{
namespace support_library
{

class ConversionPass : public Pass
{
public:
    static std::unique_ptr<ConversionPass> CreateGreedily(const HardwareCapabilities& capabilities,
                                                          size_t id,
                                                          Node* firstNode,
                                                          SramAllocator& sramAllocator);

    ConversionPass(const HardwareCapabilities& capabilities,
                   size_t id,
                   const std::vector<Node*>& nodes,
                   TensorShape stripeShape,
                   uint32_t sramOffset);

    /// Generates this Pass by adding appropriate entries to the given command stream, memory map and buffer table.
    void Generate(command_stream::CommandStreamBuffer& cmdStream, BufferManager& bufferManager, bool dumpRam) override;

    DotAttributes GetDotAttributes() override;

private:
    PassStats GetStats(const EstimationOptions& estimationOptions) override;

    TensorShape m_StripeShape;
};

}    // namespace support_library
}    // namespace ethosn
