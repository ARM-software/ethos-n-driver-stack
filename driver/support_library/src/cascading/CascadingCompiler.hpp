//
// Copyright © 2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Plan.hpp"
#include "ethosn_command_stream/CommandStreamBuffer.hpp"
#include <unordered_map>

namespace ethosn
{
namespace support_library
{
namespace cascading_compiler
{

using AgentIdType         = std::vector<command_stream::cascading::Agent>::size_type;
using RelativeAgentIdType = uint8_t;

constexpr uint8_t g_MaxRelativeAgentPosition = 255;
constexpr uint8_t g_DmaInputBufferIndex      = 0;
constexpr uint8_t g_MceIfmBufferIndex        = 0;
constexpr uint8_t g_MceWeightBufferIndex     = 1;
constexpr uint8_t g_PleInputBuffer0Index     = 0;
constexpr uint8_t g_PleInputBuffer1Index     = 1;

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
    std::vector<command_stream::cascading::Agent> GetCommandStreamOfAgents();
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

    // Private function to add IFM_STREAMER to the command stream
    AgentIdType AddIfmStreamerToCommandStream(DmaOp* const ptrDmaOp);
    // Private function to add WGT_STREAMER to the command stream
    AgentIdType AddWeightStreamerToCommandStream(DmaOp* const ptrDmaOp);
    // Private function to add MCE_SCHEDULER to the command stream
    AgentIdType AddMceSchedulerToCommandStream(MceOp* const ptrMceOp,
                                               const command_stream::cascading::PleKernelId pleKernelId);
    // Private function to add PLE_LOADER to the command stream
    AgentIdType AddPleLoaderToCommandStream(PleOp* const ptrPleOp);
    // Private function to add PLE_SCHEDULER to the command stream
    AgentIdType AddPleSchedulerToCommandStream(PleOp* const ptrPleOp);
    // Private function to add OFM_STREAMER to the command stream
    AgentIdType AddOfmStreamerToCommandStream(DmaOp* const ptrDmaOp);

    // Private function to add ReadAfterWrite Dependency
    // Consumer agent creates and own the dependency
    inline void AddReadAfterWriteDependency(const command_stream::cascading::AgentType consumerAgentType,
                                            const AgentIdType consumerAgentId,
                                            const command_stream::cascading::AgentType producerAgentType,
                                            const AgentIdType producerAgentId);
    // Private function to add SRAM Overlap Dependency
    // Consumer agent creates and own the dependency
    inline void AddSramOverlapDependency(const command_stream::cascading::AgentType consumerAgentType,
                                         const AgentIdType consumerAgentId,
                                         const command_stream::cascading::AgentType producerAgentType,
                                         const AgentIdType producerAgentId);
    // Private function to add WriteAfterRead Dependency
    // Last consumer agent creates the dependency and assign it to the producer agent
    inline void AddWriteAfterReadDependency(const command_stream::cascading::AgentType consumerAgentType,
                                            const AgentIdType consumerAgentId,
                                            const command_stream::cascading::AgentType producerAgentType,
                                            const AgentIdType producerAgentId);
    // Private function to add ScheduleTime Dependency
    // First consumer agent creates the dependency and assign it to the producer agent
    inline void AddScheduleTimeDependency(const command_stream::cascading::AgentType consumerAgentType,
                                          const AgentIdType consumerAgentId,
                                          const command_stream::cascading::AgentType producerAgentType,
                                          const AgentIdType producerAgentId);
    // Private function to fill the dependency data for Read After Write or SRAM Overlap dependencies
    void FillConsumerAgentDependency(command_stream::cascading::Dependency& consumerAgentDependency,
                                     const command_stream::cascading::AgentType consumerAgentType,
                                     const AgentIdType consumerAgentId,
                                     const command_stream::cascading::AgentType producerAgentType,
                                     const AgentIdType producerAgentId);
    // Private function to fill the dependency data for Write After Read or Schedule Time dependencies
    void FillProducerAgentDependency(command_stream::cascading::Dependency& producerAgentDependency,
                                     const command_stream::cascading::AgentType consumerAgentType,
                                     const AgentIdType consumerAgentId,
                                     const command_stream::cascading::AgentType producerAgentType,
                                     const AgentIdType producerAgentId);

    // Private function to add the lifetime information of the intermediate DRAM buffers
    void AddLifetimeInfoForIntermediateDramBuffers();

    // Intermediate DRAM Buffer to Buffer Id mapping
    std::unordered_map<Buffer*, uint32_t> m_IntermdiateDramBufToBufIdMapping;

    // Merged OpGraph used to generate the command stream, set at creation time.
    const OpGraph m_MergedOpGraph;
    const std::set<uint32_t> m_OperationIds;

    // Compilation parameters, set at creation time.
    HardwareCapabilities m_Capabilities;
    const CompilationOptions m_CompilationOptions;

    // Data structure for mapping an Op to its Agent ID
    std::unordered_map<Op*, AgentIdType> m_OpToAgentIdMapping;

    // Define a Hash functor for enum classes to workaround an issue found when compiling with g++ 5.1
    struct EnumHasher
    {
        template <typename T>
        std::size_t operator()(T t) const noexcept
        {
            return static_cast<std::size_t>(t);
        }
    };
    // Data structure for mapping a Ple kernel to its loader Agent
    std::unordered_map<command_stream::cascading::PleKernelId, AgentIdType, EnumHasher>
        m_PleKernelToPleLoaderAgentIdMapping;

    // Command stream agents used to build the command stream and to be stored in the BufferManager instance at BufferId = 0
    std::vector<command_stream::cascading::Agent> m_CommandStreamAgents;
    command_stream::CommandStreamBuffer m_CommandStream;

    // BufferManager instance which maintains and builds up the set of buffers required by the compiled network
    BufferManager m_BufferManager;
};
}    // namespace cascading_compiler
}    // namespace support_library
}    // namespace ethosn
