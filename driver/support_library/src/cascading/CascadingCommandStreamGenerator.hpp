//
// Copyright © 2022-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Estimation.hpp"
#include "Plan.hpp"
#include "Scheduler.hpp"
#include "ethosn_command_stream/CommandStreamBuffer.hpp"
#include <unordered_map>

namespace ethosn
{
namespace support_library
{
class CompiledNetworkImpl;
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

struct CompiledOpGraph
{
    EstimatedOpGraph m_EstimatedOpGraph;
    std::unordered_map<Op*, AgentIdType> m_OpToAgentIdMapping;
    std::unordered_map<DramBuffer*, uint32_t> m_BufferIds;
    std::unique_ptr<CompiledNetworkImpl> m_CompiledNetwork;
};

class CascadingCommandStreamGenerator
{
public:
    CascadingCommandStreamGenerator(const OpGraph& mergedOpGraph,
                                    const std::set<uint32_t>& operationIds,
                                    const HardwareCapabilities& capabilities,
                                    const CompilationOptions& compilationOptions,
                                    const DebuggingContext& debuggingContext);
    CascadingCommandStreamGenerator(const CascadingCommandStreamGenerator&) = delete;
    CascadingCommandStreamGenerator& operator=(const CascadingCommandStreamGenerator&) = delete;
    ~CascadingCommandStreamGenerator();

    // Compile a given network and return the compiled network
    CompiledOpGraph Generate();

    // Functions used to retrieve private members
    const BufferManager& GetBufferManager() const;
    const OpGraph& GetMergedOpGraph() const;
    const std::unordered_map<DramBuffer*, uint32_t>& GetDramBufToBufIdMapping() const;

private:
    // Private functions for processing OpGraph Ops
    void ProcessDmaOp(DmaOp* const ptrDmaOp);
    void ProcessMceOp(Op* const ptrMceOp);
    void ProcessPleOp(Op* const ptrPleOp);
    void ProcessSpaceToDepthOp(Op* const ptrSpaceToDepthOp);
    void ProcessTransposeOp(Op* const ptrTransposeOp);

    // Private function to add IFM_STREAMER to the command stream
    AgentIdType AddIfmStreamerToCommandStream(DmaOp* const ptrOp,
                                              const uint16_t inputDramBufferId,
                                              const Buffer* const inputDramBuffer,
                                              const SramBuffer* const inputSramBuffer,
                                              const CascadingBufferFormat transferFormat,
                                              const uint32_t inputDramBufferOffset,
                                              bool isExtraIfmStripeAtRightEdge,
                                              bool isExtraIfmStripeAtBottomEdge);
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
    AgentIdType AddOfmStreamerToCommandStream(DmaOp* const ptrOp,
                                              const SramBuffer* const outputSramBuffer,
                                              const uint16_t outputDramBufferId,
                                              const Buffer* const outputDramBuffer,
                                              const uint32_t outputDramBufferOffset);

    // Private function to add ReadAfterWrite Dependency
    // Consumer agent creates and own the dependency
    inline void AddReadAfterWriteDependency(const command_stream::cascading::AgentType consumerAgentType,
                                            const AgentIdType consumerAgentId,
                                            const command_stream::cascading::AgentType producerAgentType,
                                            const AgentIdType producerAgentId,
                                            const Op* producerOp);
    // Private function to add SRAM Overlap Dependency
    // Consumer agent creates and own the dependency
    inline void AddSramOverlapDependency(const command_stream::cascading::AgentType consumerAgentType,
                                         const AgentIdType consumerAgentId,
                                         const command_stream::cascading::AgentType producerAgentType,
                                         const AgentIdType producerAgentId,
                                         const Op* producerOp);
    // Private function to add WriteAfterRead Dependency
    // Last consumer agent creates the dependency and assign it to the producer agent
    inline void AddWriteAfterReadDependency(const command_stream::cascading::AgentType consumerAgentType,
                                            const AgentIdType consumerAgentId,
                                            const command_stream::cascading::AgentType producerAgentType,
                                            const AgentIdType producerAgentId,
                                            const Op* producerOp);
    // Private function to fill the dependency data for Read After Write or SRAM Overlap dependencies
    void FillConsumerAgentDependency(Dependency& consumerAgentDependency,
                                     const command_stream::cascading::AgentType consumerAgentType,
                                     const AgentIdType consumerAgentId,
                                     const command_stream::cascading::AgentType producerAgentType,
                                     const AgentIdType producerAgentId,
                                     const Op* producerOp) const;
    // Private function to fill the dependency data for Write After Read dependencies
    void FillProducerAgentDependency(Dependency& producerAgentDependency,
                                     const command_stream::cascading::AgentType consumerAgentType,
                                     const AgentIdType consumerAgentId,
                                     const command_stream::cascading::AgentType producerAgentType,
                                     const AgentIdType producerAgentId,
                                     const Op* producerOp) const;

    // Private function to add the lifetime information of the intermediate DRAM buffers
    void AddLifetimeInfoForIntermediateDramBuffers();

    // DRAM Buffer to Buffer Id mapping
    std::unordered_map<DramBuffer*, uint32_t> m_DramBufToBufIdMapping;
    // Used for intermediate and input buffers as they can be duplicated
    uint16_t AddDramBufferAndCacheId(DramBuffer* inputBuffer, Op* const op);

    // Merged OpGraph used to generate the command stream, set at creation time.
    const OpGraph m_MergedOpGraph;
    const std::set<uint32_t> m_OperationIds;

    // Compilation parameters, set at creation time.
    HardwareCapabilities m_Capabilities;
    const CompilationOptions m_CompilationOptions;
    const DebuggingContext& m_DebuggingContext;

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

    /// Command stream agents used to build the command stream. This is an intermediate representation
    /// that is just for the support library and is converted into the command stream agents, commands and
    /// extra data by the scheduler and register generation code.
    std::vector<AgentDescAndDeps> m_CommandStreamAgents;

    command_stream::CommandStreamBuffer m_CommandStream;

    // BufferManager instance which maintains and builds up the set of buffers required by the compiled network
    BufferManager m_BufferManager;

    /// The most recent Op that could block later Ops from loading data into SRAM, to prevent corruption.
    /// The next occurence of an IfmS, PleL or WgtS will include a dependency on these to stall them until the
    /// fence Op has finished.
    Op* m_FenceOp;
};
}    // namespace cascading_compiler
}    // namespace support_library
}    // namespace ethosn
