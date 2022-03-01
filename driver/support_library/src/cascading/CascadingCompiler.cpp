//
// Copyright © 2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../Compiler.hpp"
#include "CascadingCompiler.hpp"

#include <memory>

namespace ethosn
{
namespace support_library
{

CascadingCompiler::CascadingCompiler(const OpGraph& mergedOpGraph,
                                     const std::set<uint32_t>& operationIds,
                                     const HardwareCapabilities& capabilities,
                                     const CompilationOptions& compilationOptions)
    : m_MergedOpGraph{ mergedOpGraph }
    , m_OperationIds{ operationIds }
    , m_Capabilities{ capabilities }
    , m_CompilationOptions{ compilationOptions }
{}

CascadingCompiler::~CascadingCompiler()
{}

std::unique_ptr<CompiledNetwork> CascadingCompiler::Compile()
{
    OpGraph::OpList opsInExecutionOrder = m_MergedOpGraph.GetOps();

    assert(opsInExecutionOrder.size() != 0 && m_CommandStreamAgents.size() == 0);

    try
    {
        for (auto currentOp : opsInExecutionOrder)
        {
            if (IsObjectOfType<DmaOp>(currentOp))
            {
                ProcessDmaOp(currentOp);
            }
            else if (IsObjectOfType<MceOp>(currentOp))
            {
                ProcessMceOp(currentOp);
            }
            else if (IsObjectOfType<PleOp>(currentOp))
            {
                ProcessPleOp(currentOp);
            }
            else if (IsObjectOfType<ConcatOp>(currentOp))
            {
                ProcessConcatOp(currentOp);
            }
            else
            {
                throw NotSupportedException("Op is not currently supported by the Cascading Compiler");
            }
        }
    }
    catch (const NotSupportedException& e)
    {
        g_Logger.Error("Error: %s", e.what());
        return std::unique_ptr<CompiledNetworkImpl>(nullptr);
    }

    // Add the lifetime information of the intermediate DRAM buffers so the memory required to store these
    // buffers is reduced
    AddLifetimeInfoForIntermediateDramBuffers();

    // Add the generated command stream to the buffer manager
    CommandStream commandStream{ m_CommandStreamAgents };
    //m_BufferManager.AddCommandStream(commandStream);

    // Create the compiled network using the updated BufferManager instance
    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = std::make_unique<CompiledNetworkImpl>(
        m_BufferManager.GetConstantDmaData(), m_BufferManager.GetConstantControlUnitData(),
        m_BufferManager.GetBuffers(), m_OperationIds);

    return compiledNetwork;
}

OpGraph CascadingCompiler::GetMergedOpGraph() const
{
    return m_MergedOpGraph;
}

// Private function to add the lifetime information of the intermediate DRAM buffers
void CascadingCompiler::AddLifetimeInfoForIntermediateDramBuffers()
{
    // Lifetime start of the buffer holds the producer agent Id
    uint32_t lifetimeStart;
    // Lifetime end of the buffer holds the last consumer agent Id
    uint32_t lifetimeEnd;

    // Add the lifetime information for each intermediate DRAM buffer
    for (Buffer* buffer : m_MergedOpGraph.GetBuffers())
    {
        if (buffer->m_Location == Location::Dram)
        {
            assert(buffer->m_BufferType.has_value());

            // Check that the buffer type is intermediate
            if (buffer->m_BufferType.value() == BufferType::Intermediate)
            {
                // Set the Lifetime start and end of the intermediate DRAM buffer
                Op* producer = m_MergedOpGraph.GetProducer(buffer);
                assert(producer != nullptr);

                lifetimeStart = m_OpToAgentIdMapping.at(producer);
                lifetimeEnd   = 0;

                OpGraph::ConsumersList consumers = m_MergedOpGraph.GetConsumers(buffer);
                assert(consumers.size() >= 1);
                for (auto consumer : consumers)
                {
                    uint32_t consumerAgentId = m_OpToAgentIdMapping.at(consumer.first);
                    if (consumerAgentId > lifetimeEnd)
                    {
                        lifetimeEnd = consumerAgentId;
                    }
                }

                // Add lifetime information of the corresponding buffer to the buffer manager
                m_BufferManager.MarkBufferUsedAtTime(m_IntermdiateDramBufToBufIdMapping.at(buffer), lifetimeStart,
                                                     lifetimeEnd + 1);
            }
        }
    }
}

void CascadingCompiler::ProcessDmaOp(Op* const ptrDmaOp)
{
    ETHOSN_UNUSED(ptrDmaOp);
}

void CascadingCompiler::ProcessMceOp(Op* const ptrMceOp)
{
    ETHOSN_UNUSED(ptrMceOp);
}

void CascadingCompiler::ProcessPleOp(Op* const ptrPleOp)
{
    ETHOSN_UNUSED(ptrPleOp);
}

void CascadingCompiler::ProcessConcatOp(Op* const ptrConcatOp)
{
    ETHOSN_UNUSED(ptrConcatOp);
}

void CascadingCompiler::ProcessSplitOp(Op* const ptrSplitOp)
{
    ETHOSN_UNUSED(ptrSplitOp);
}

void CascadingCompiler::ProcessSpaceToDepthOp(Op* const ptrSpaceToDepthOp)
{
    ETHOSN_UNUSED(ptrSpaceToDepthOp);
}

void CascadingCompiler::ProcessTransposeOp(Op* const ptrTransposeOp)
{
    ETHOSN_UNUSED(ptrTransposeOp);
}

}    // namespace support_library
}    // namespace ethosn
