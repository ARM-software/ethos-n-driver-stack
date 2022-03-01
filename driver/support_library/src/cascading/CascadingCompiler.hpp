//
// Copyright © 2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Plan.hpp"
#include <unordered_map>

using namespace std;
using namespace ethosn::command_stream::cascading;

namespace ethosn
{
namespace support_library
{

class CascadingCompiler
{
public:
    CascadingCompiler(const OpGraph& mergedOpGraph,
                      const std::set<uint32_t>& operationIds,
                      const HardwareCapabilities& capabilities,
                      const CompilationOptions& compilationOptions);
    CascadingCompiler(const CascadingCompiler&) = delete;
    CascadingCompiler& operator=(const CascadingCompiler&) = delete;
    ~CascadingCompiler();
    std::unique_ptr<CompiledNetwork> Compile();
    OpGraph GetMergedOpGraph() const;

private:
    // Private function to add the lifetime information of the intermediate DRAM buffers
    void AddLifetimeInfoForIntermediateDramBuffers();

    // Private functions for processing OpGraph Ops
    void ProcessDmaOp(Op* const ptrDmaOp);
    void ProcessMceOp(Op* const ptrMceOp);
    void ProcessPleOp(Op* const ptrPleOp);
    void ProcessConcatOp(Op* const ptrConcatOp);
    void ProcessSplitOp(Op* const ptrSplitOp);
    void ProcessSpaceToDepthOp(Op* const ptrSpaceToDepthOp);
    void ProcessTransposeOp(Op* const ptrTransposeOp);

    // Intermediate DRAM Buffer to Buffer Id mapping
    std::unordered_map<Buffer*, uint32_t> m_IntermdiateDramBufToBufIdMapping;

    // Merged OpGraph used to generate the command stream, set at creation time.
    const OpGraph m_MergedOpGraph;
    const std::set<uint32_t> m_OperationIds;

    // Compilation parameters, set at creation time.
    HardwareCapabilities m_Capabilities;
    const CompilationOptions m_CompilationOptions;

    // Data structure for mapping an Op to its Agents ID
    std::unordered_map<Op*, uint32_t> m_OpToAgentIdMapping;
    // Data structures for down the sequence dependencies (i.e WriteAfterRead and ScheduleTime)
    std::unordered_map<Op*, Dependency*> m_PendingWriteAfterReadDependencyForConsumerOp;
    std::unordered_map<Op*, Dependency*> m_PendingScheduleTimeDependencyForConsumerOp;

    // Command stream agents used to build the command stream that is stored in the BufferManager instance at BufferId = 0
    std::vector<Agent> m_CommandStreamAgents;

    // BufferManager instance which maintains and builds up the set of buffers required by the compiled network
    BufferManager m_BufferManager;
};

}    // namespace support_library
}    // namespace ethosn
