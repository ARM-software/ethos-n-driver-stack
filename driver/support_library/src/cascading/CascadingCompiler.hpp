//
// Copyright © 2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Plan.hpp"
#include "ethosn_command_stream/CommandStreamBuffer.hpp"
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

private:
    // Private functions for processing OpGraph Ops
    void ProcessDmaOp(Op* const ptrDmaOp);
    void ProcessMceOp(Op* const ptrMceOp);
    void ProcessPleOp(Op* const ptrPleOp);
    void ProcessConcatOp(Op* const ptrConcatOp);
    void ProcessSplitOp(Op* const ptrSplitOp);
    void ProcessSpaceToDepthOp(Op* const ptrSpaceToDepthOp);
    void ProcessTransposeOp(Op* const ptrTransposeOp);

    // Merged OpGraph used to generate the command stream, set at creation time.
    const OpGraph m_MergedOpGraph;
    const std::set<uint32_t> m_OperationIds;

    // Compilation parameters, set at creation time.
    HardwareCapabilities m_Capabilities;
    const CompilationOptions m_CompilationOptions;

    // Data structures for up the sequence dependencies (i.e ReadAfterWrite and SramOverlap)
    std::unordered_map<Op*, uint8_t> m_ReadAfterWriteDependencies;
    // Data structures for down the sequence dependencies (i.e WriteAfterRead and ScheduleTime)
    std::unordered_map<Op*, Dependency*> m_WriteAfterReadDependencies;
    std::unordered_map<Op*, Dependency*> m_ScheduleTimeDependencies;

    // Command Stream Buffer to be added to the BufferManager instance at BufferId = 0
    command_stream::CommandStreamBuffer m_CommandStream;

    // BufferManager instance which maintains and builds up the set of buffers required by the compiled network
    BufferManager m_BufferManager;
};

}    // namespace support_library
}    // namespace ethosn
