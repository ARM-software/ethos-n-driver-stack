//
// Copyright © 2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "CascadingCommandStreamGenerator.hpp"

#include "CascadingCommandStreamGeneratorUtils.hpp"
#include "Compiler.hpp"
#include "Visualisation.hpp"

#include <memory>

namespace ethosn
{
namespace support_library
{
namespace cascading_compiler
{

CascadingCommandStreamGenerator::CascadingCommandStreamGenerator(const OpGraph& mergedOpGraph,
                                                                 const std::set<uint32_t>& operationIds,
                                                                 const HardwareCapabilities& capabilities,
                                                                 const CompilationOptions& compilationOptions,
                                                                 const DebuggingContext& debuggingContext)
    : m_MergedOpGraph{ mergedOpGraph }
    , m_OperationIds{ operationIds }
    , m_Capabilities{ capabilities }
    , m_CompilationOptions{ compilationOptions }
    , m_DebuggingContext(debuggingContext)
{

    m_CommandStreamAgents.reserve(m_MergedOpGraph.GetOps().size());
}

CascadingCommandStreamGenerator::~CascadingCommandStreamGenerator()
{}

// Compile a given network and return the compiled network
std::unique_ptr<CompiledNetworkImpl> CascadingCommandStreamGenerator::Generate()
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
    m_CommandStream.EmplaceBack(command_stream::Cascade{ static_cast<uint32_t>(m_CommandStreamAgents.size()) });
    for (auto& agent : m_CommandStreamAgents)
    {
        m_CommandStream.EmplaceBack<Agent>(agent);
    }

    // Add DUMP_DRAM commands to the command stream, if requested.
    if (m_DebuggingContext.m_DebugInfo.m_DumpRam)
    {
        for (std::pair<Buffer*, uint32_t> b : m_DramBufToBufIdMapping)
        {
            if (b.first->m_BufferType == BufferType::Intermediate)
            {
                const TensorShape& shape = b.first->m_TensorShape;

                std::string dumpName;
                {
                    std::stringstream ss;
                    ss << "EthosNIntermediateBuffer_" << b.second;
                    // Currently all buffers are assumed to be UINT8. This will need changing once we support INT8 too.
                    ss << "_" << ToString(DataType::UINT8_QUANTIZED);
                    ss << "_" << ToString(b.first->m_Format);
                    ss << "_" << shape[0] << "_" << shape[1] << "_" << shape[2] << "_" << shape[3];
                    ss << ".hex";

                    dumpName = ss.str();
                }

                ethosn::command_stream::DumpDram cmdStrDumpDram;
                cmdStrDumpDram.m_DramBufferId() = b.second;

                assert(dumpName.size() < sizeof(cmdStrDumpDram.m_Filename()));
                std::copy(dumpName.begin(), dumpName.end(), cmdStrDumpDram.m_Filename().begin());
                m_CommandStream.EmplaceBack(cmdStrDumpDram);
            }
        }
    }

    m_BufferManager.AddCommandStream(m_CommandStream);

    m_BufferManager.Allocate(m_DebuggingContext);

    // Create the compiled network using the updated BufferManager instance
    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = std::make_unique<CompiledNetworkImpl>(
        m_BufferManager.GetConstantDmaData(), m_BufferManager.GetConstantControlUnitData(),
        m_BufferManager.GetBuffers(), m_OperationIds);

    return compiledNetwork;
}

// Functions used to retrieve private members
const std::vector<Agent>& CascadingCommandStreamGenerator::GetCommandStreamOfAgents() const
{
    return m_CommandStreamAgents;
}

const BufferManager& CascadingCommandStreamGenerator::GetBufferManager() const
{
    return m_BufferManager;
}

const OpGraph& CascadingCommandStreamGenerator::GetMergedOpGraph() const
{
    return m_MergedOpGraph;
}

const std::unordered_map<Buffer*, uint32_t>& CascadingCommandStreamGenerator::GetDramBufToBufIdMapping() const
{
    return m_DramBufToBufIdMapping;
}

uint16_t CascadingCommandStreamGenerator::AddDramBufferAndCacheId(Buffer* inputBuffer, Op* const)
{
    uint16_t inputBufferId = std::numeric_limits<uint16_t>::max();
    if (m_DramBufToBufIdMapping.find(inputBuffer) != m_DramBufToBufIdMapping.end())
    {
        inputBufferId = ethosn::utils::NumericCast<uint16_t>(m_DramBufToBufIdMapping[inputBuffer]);
    }
    else
    {
        if (inputBuffer->m_BufferType.value() == BufferType::Input)
        {
            assert(inputBuffer->m_OperationId.has_value());
            inputBufferId = ethosn::utils::NumericCast<uint16_t>(
                m_BufferManager.AddDramInput(inputBuffer->m_SizeInBytes, inputBuffer->m_OperationId.value()));
            m_DramBufToBufIdMapping[inputBuffer] = inputBufferId;
        }
        else if (inputBuffer->m_BufferType.value() == BufferType::Intermediate)
        {
            inputBufferId = ethosn::utils::NumericCast<uint16_t>(
                m_BufferManager.AddDram(inputBuffer->m_BufferType.value(), inputBuffer->m_SizeInBytes));
            m_DramBufToBufIdMapping[inputBuffer] = inputBufferId;
        }
    }
    return inputBufferId;
}

// Private functions for processing OpGraph Ops
void CascadingCommandStreamGenerator::ProcessDmaOp(Op* const ptrDmaOp)
{
    // Get the input buffer to the Dma Op
    OpGraph::BufferList inputBuffers = m_MergedOpGraph.GetInputs(ptrDmaOp);
    Buffer* inputBuffer              = inputBuffers[g_DmaInputBufferIndex];
    assert(inputBuffers.size() == 1);

    // Get the output buffer from the Dma Op
    Buffer* outputBuffer = m_MergedOpGraph.GetOutput(ptrDmaOp);
    assert(outputBuffer != nullptr);

    // Construct and add the required agents to the command stream
    if (inputBuffer->m_Location == Location::Dram && outputBuffer->m_Location == Location::Sram)
    {
        assert(inputBuffer->m_BufferType.has_value());

        if (inputBuffer->m_Format != CascadingBufferFormat::WEIGHT)
        {
            assert(inputBuffer->m_BufferType.value() == BufferType::Intermediate ||
                   inputBuffer->m_BufferType.value() == BufferType::Input);

            DmaOp* const dmaOp = static_cast<DmaOp*>(ptrDmaOp);

            uint16_t inputBufferId         = AddDramBufferAndCacheId(inputBuffer, ptrDmaOp);
            AgentIdType ifmStreamerAgentId = AddIfmStreamerToCommandStream(ptrDmaOp, inputBufferId, inputBuffer,
                                                                           outputBuffer, dmaOp->m_TransferFormat);

            // Only intermediate input buffers need the dependencies not inputs to the network
            if (inputBuffer->m_BufferType == BufferType::Intermediate)
            {
                // Add 'Read After Write' dependency to the IfmStreamer agent
                // Read After Write Dependency for [IfmStreamer][OfmStreamer]
                AddReadAfterWriteDependency(AgentType::IFM_STREAMER, ifmStreamerAgentId, AgentType::OFM_STREAMER,
                                            m_OpToAgentIdMapping[m_MergedOpGraph.GetProducer(inputBuffer)]);

                // Add 'Schedule Time' dependency information to the OfmStreamer agent
                // Schedule Time Dependency for [OfmStreamer][IfmStreamer]
                AddScheduleTimeDependency(AgentType::IFM_STREAMER, ifmStreamerAgentId, AgentType::OFM_STREAMER,
                                          m_OpToAgentIdMapping[m_MergedOpGraph.GetProducer(inputBuffer)]);
            }
        }
        else
        {
            // Weight Streamer Agent
            AgentIdType weightStreamerAgentId = AddWeightStreamerToCommandStream(static_cast<DmaOp*>(ptrDmaOp));

            // Get the agent ID of the OFM Streamer
            // Get the Mce consumer of the weights buffer
            auto weightBufferConsumer = m_MergedOpGraph.GetConsumer(outputBuffer, 0);
            assert(weightBufferConsumer.first != nullptr && IsObjectOfType<MceOp>(weightBufferConsumer.first));

            // Look further up in the graph, to see if there is an OfmS that we need to wait for.
            // This is to prevent the weight streamer from the next section starting to load data before
            // the OfmS from the previous section is done (sections can't overlap!).
            Buffer* mceInput = m_MergedOpGraph.GetInputs(weightBufferConsumer.first)[g_MceIfmBufferIndex];
            assert(mceInput != nullptr);

            Op* ifmProducer = m_MergedOpGraph.GetProducer(mceInput);
            if (IsObjectOfType<DmaOp>(ifmProducer))
            {
                Buffer* ifmDmaInput = m_MergedOpGraph.GetInputs(ifmProducer)[0];
                assert(ifmDmaInput != nullptr);

                Op* intermediateProducer = m_MergedOpGraph.GetProducer(ifmDmaInput);
                if (intermediateProducer != nullptr &&
                    IsObjectOfType<DmaOp>(intermediateProducer))    // May be nullptr if ifmDmaInput is a network input
                {
                    AgentIdType ofmStreamerAgentId = m_OpToAgentIdMapping[intermediateProducer];

                    // Add 'Sram Overlap' dependency to the WeightStreamer agent
                    // Sram Overlap Dependency for [WeightStreamer][OfmStreamer]
                    AddSramOverlapDependency(AgentType::WGT_STREAMER, weightStreamerAgentId, AgentType::OFM_STREAMER,
                                             ofmStreamerAgentId);
                }
            }
            // We also need to check and wait for an MceS in the same section if this is a strategy 1 cascade.
            // Strategy 1 cascade means that each MCE finishes before the next starts, and our combiner does an
            // sram eviction for each new layer --- hence a potential sram overlap hazard.
            // Therefore we need to make sure that the firmware doesn't load the new weights until the weights
            // from the previous layer are finished with.
            else if (IsObjectOfType<PleOp>(ifmProducer))
            {
                Buffer* pleInput = m_MergedOpGraph.GetInputs(ifmProducer)[0];
                assert(pleInput != nullptr);

                Op* mce = m_MergedOpGraph.GetProducer(pleInput);
                if (IsObjectOfType<MceOp>(mce))
                {
                    // Check if this is an s1 cascade
                    if (mceInput->IsFullTensor())
                    {
                        AgentIdType mceStreamerAgentId = m_OpToAgentIdMapping[mce];

                        // Add 'Sram Overlap' dependency to the WeightStreamer agent
                        // Sram Overlap Dependency for [WeightStreamer][MceScheduler]
                        AddSramOverlapDependency(AgentType::WGT_STREAMER, weightStreamerAgentId,
                                                 AgentType::MCE_SCHEDULER, mceStreamerAgentId);
                    }
                }
            }
        }
    }
    else if (inputBuffer->m_Location == Location::Sram && outputBuffer->m_Location == Location::Dram)
    {
        assert(inputBuffer->m_Offset.has_value());
        assert(outputBuffer->m_BufferType.has_value());

        // Get the producer of the input buffer and the producing agent type
        Op* producerOp = m_MergedOpGraph.GetProducer(inputBuffer);
        assert(IsObjectOfType<PleOp>(producerOp) || IsObjectOfType<DmaOp>(producerOp));

        AgentType producerAgentType;
        if (IsObjectOfType<PleOp>(producerOp))
        {
            producerAgentType = AgentType::PLE_SCHEDULER;
        }
        else
        {
            producerAgentType = AgentType::IFM_STREAMER;
        }

        uint16_t outputBufferId = static_cast<uint16_t>(
            m_BufferManager.AddDram(outputBuffer->m_BufferType.value(), outputBuffer->m_SizeInBytes));

        // If this is an Intermediate Dram Buffer, add it to the DramBufToBufId map with the appropriate Id
        if (outputBuffer->m_BufferType.value() == BufferType::Intermediate)
        {
            assert(m_DramBufToBufIdMapping.find(outputBuffer) == m_DramBufToBufIdMapping.end());
            m_DramBufToBufIdMapping[outputBuffer] = outputBufferId;
        }
        else if (outputBuffer->m_BufferType.value() == BufferType::Output)
        {
            assert(outputBuffer->m_OperationId.has_value());
            assert(outputBuffer->m_ProducerOutputIndx);
            m_BufferManager.ChangeToOutput(outputBufferId, outputBuffer->m_OperationId.value(),
                                           outputBuffer->m_ProducerOutputIndx.value());
        }

        // Ofm Streamer Agent
        AgentIdType ofmStreamerAgentId =
            AddOfmStreamerToCommandStream(ptrDmaOp, inputBuffer, outputBufferId, outputBuffer);

        // Add 'Read After Write' dependency information to the IfmStreamer and PleScheduler agents
        // Read After Write Dependency for [OfmStreamer][IfmStreamer] or
        // Read After Write Dependency for [OfmStreamer][PleScheduler]
        AddReadAfterWriteDependency(AgentType::OFM_STREAMER, ofmStreamerAgentId, producerAgentType,
                                    m_OpToAgentIdMapping[producerOp]);

        // Add 'Write After Read' dependency information to the IfmStreamer and PleScheduler agents
        // Write After Read Dependency for [IfmStreamer][OfmStreamer] or
        // Write After Read Dependency for [PleScheduler][OfmStreamer]
        AddWriteAfterReadDependency(AgentType::OFM_STREAMER, ofmStreamerAgentId, producerAgentType,
                                    m_OpToAgentIdMapping[producerOp]);

        // Add 'Schedule Time' dependency information to the IfmStreamer and PleScheduler agents
        // Schedule Time Dependency for [IfmStreamer][OfmStreamer] or
        // Schedule Time Dependency for [PleScheduler][OfmStreamer]
        AddScheduleTimeDependency(AgentType::OFM_STREAMER, ofmStreamerAgentId, producerAgentType,
                                  m_OpToAgentIdMapping[producerOp]);
    }
    else
    {
        assert(false);
    }
}

void CascadingCommandStreamGenerator::ProcessMceOp(Op* const ptrMceOp)
{
    // Get the input buffers to the Mce Op
    OpGraph::BufferList inputBuffers = m_MergedOpGraph.GetInputs(ptrMceOp);
    assert(inputBuffers.size() == 2 && inputBuffers[g_MceIfmBufferIndex]->m_Offset.has_value() &&
           inputBuffers[g_MceWeightBufferIndex]->m_Offset.has_value());

    // Get the output buffer from the Mce Op
    Buffer* outputBuffer = m_MergedOpGraph.GetOutput(ptrMceOp);
    assert(outputBuffer != nullptr);

    auto producerOp = m_MergedOpGraph.GetProducer(inputBuffers[g_MceIfmBufferIndex]);
    AgentType producerAgentType;
    if (IsObjectOfType<PleOp>(producerOp))
    {
        // MceOp takes input 0 from pleS agent
        producerAgentType = AgentType::PLE_SCHEDULER;
    }
    else
    {
        // MceOp takes input 0 from ifmS agent
        producerAgentType = AgentType::IFM_STREAMER;
    }

    // Construct and add the required agents to the command stream
    // Ple Loader Agent
    auto mceOpConsumer = m_MergedOpGraph.GetConsumer(outputBuffer, 0);
    assert(mceOpConsumer.first != nullptr && IsObjectOfType<PleOp>(mceOpConsumer.first));

    AgentIdType pleLoaderAgentId = 0;
    PleOp* ptrPleOp              = static_cast<PleOp*>(mceOpConsumer.first);

    if (ptrPleOp->m_LoadKernel)
    {
        pleLoaderAgentId = AddPleLoaderToCommandStream(ptrPleOp);
    }

    // MCE Scheduler Agent
    AgentIdType mceSchedulerAgentId =
        AddMceSchedulerToCommandStream(static_cast<MceOp*>(ptrMceOp), ptrPleOp->m_PleKernelId);

    // Add 'Read After Write' dependency to the MceScheduler agent
    // Read After Write Dependency for [MceScheduler][IfmStreamer] or
    // Read After Write Dependency for [MceScheduler][PleScheduler]
    AddReadAfterWriteDependency(AgentType::MCE_SCHEDULER, mceSchedulerAgentId, producerAgentType,
                                m_OpToAgentIdMapping[producerOp]);
    // Read After Write Dependency for [MceScheduler][WeightStreamer]
    AddReadAfterWriteDependency(
        AgentType::MCE_SCHEDULER, mceSchedulerAgentId, AgentType::WGT_STREAMER,
        m_OpToAgentIdMapping[m_MergedOpGraph.GetProducer(inputBuffers[g_MceWeightBufferIndex])]);

    // Add 'Write After Read' dependency information to the IfmStreamer and WeightStreamer agents
    // Write After Read Dependency for [IfmStreamer][MceScheduler] or
    // Write After Read Dependency for [PleScheduler][MceScheduler]
    AddWriteAfterReadDependency(AgentType::MCE_SCHEDULER, mceSchedulerAgentId, producerAgentType,
                                m_OpToAgentIdMapping[producerOp]);
    // Write After Read Dependency for [WeightStreamer][MceScheduler]
    AddWriteAfterReadDependency(
        AgentType::MCE_SCHEDULER, mceSchedulerAgentId, AgentType::WGT_STREAMER,
        m_OpToAgentIdMapping[m_MergedOpGraph.GetProducer(inputBuffers[g_MceWeightBufferIndex])]);

    // Add 'Schedule Time' dependency information to the IfmStreamer and WeightStreamer agents
    // Schedule Time Dependency for [IfmStreamer][MceScheduler] or
    // Schedule Time Dependency for [PleScheduler][MceScheduler]
    AddScheduleTimeDependency(AgentType::MCE_SCHEDULER, mceSchedulerAgentId, producerAgentType,
                              m_OpToAgentIdMapping[producerOp]);
    // Schedule Time Dependency for [WeightStreamer][MceScheduler]
    AddScheduleTimeDependency(AgentType::MCE_SCHEDULER, mceSchedulerAgentId, AgentType::WGT_STREAMER,
                              m_OpToAgentIdMapping[m_MergedOpGraph.GetProducer(inputBuffers[g_MceWeightBufferIndex])]);
    // Add 'Schedule Time' dependency information to the PLE Loader agent
    // Schedule Time Dependency for [PLE Loader][MceScheduler]
    if (ptrPleOp->m_LoadKernel)
    {
        AddScheduleTimeDependency(AgentType::MCE_SCHEDULER, mceSchedulerAgentId, AgentType::PLE_LOADER,
                                  pleLoaderAgentId);
    }

    if (producerAgentType == AgentType::PLE_SCHEDULER)
    {
        Buffer* pleInputBuffer = m_MergedOpGraph.GetInputs(producerOp)[0];
        Op* pleInputProducer   = m_MergedOpGraph.GetProducer(pleInputBuffer);
        if (IsObjectOfType<MceOp>(pleInputProducer) && !inputBuffers[g_MceIfmBufferIndex]->IsFullTensor())
        {
            // Strategy 0 cascade - need schedule dependency from previous MceS
            AddScheduleTimeDependency(AgentType::MCE_SCHEDULER, mceSchedulerAgentId, AgentType::MCE_SCHEDULER,
                                      m_OpToAgentIdMapping[pleInputProducer]);
        }
    }
}

void CascadingCommandStreamGenerator::ProcessPleOp(Op* const ptrPleOp)
{
    // Get the input buffers to the Ple Op
    OpGraph::BufferList inputBuffers = m_MergedOpGraph.GetInputs(ptrPleOp);
    assert(inputBuffers.size() == 1 || inputBuffers.size() == 2);

    for (auto inputBuffer : inputBuffers)
    {
        if (inputBuffer->m_Location == Location::Sram)
        {
            assert(inputBuffer->m_Offset.has_value());
        }
        ETHOSN_UNUSED(inputBuffer);
    }

    // Get the output buffer from the Ple Op
    Buffer* outputBuffer = m_MergedOpGraph.GetOutput(ptrPleOp);
    assert(outputBuffer->m_Offset.has_value());

    // Determine whether ple op is standalone or fused
    bool isStandAlonePle = false;
    if (inputBuffers[g_PleInputBuffer0Index]->m_Location == Location::PleInputSram)
    {
        isStandAlonePle = false;
    }
    else if (inputBuffers[g_PleInputBuffer0Index]->m_Location == Location::Sram)
    {
        isStandAlonePle = true;
    }
    else
    {
        assert(false);
    }

    Op* input0Producer = m_MergedOpGraph.GetProducer(inputBuffers[g_PleInputBuffer0Index]);

    bool loadKernel = static_cast<PleOp*>(ptrPleOp)->m_LoadKernel;
    if (isStandAlonePle)
    {
        AgentIdType pleLoaderAgentId = {};

        if (loadKernel)
        {
            pleLoaderAgentId = AddPleLoaderToCommandStream(static_cast<PleOp*>(ptrPleOp));
        }

        AgentIdType pleSchedulerAgentId = AddPleSchedulerToCommandStream(static_cast<PleOp*>(ptrPleOp));

        // Read After Write Dependency for [PleScheduler][IfmStreamer]
        AddReadAfterWriteDependency(AgentType::PLE_SCHEDULER, pleSchedulerAgentId, AgentType::IFM_STREAMER,
                                    m_OpToAgentIdMapping[input0Producer]);

        if (loadKernel)
        {
            // Read After Write Dependency for [PleScheduler][PleLoader]
            AddReadAfterWriteDependency(
                AgentType::PLE_SCHEDULER, pleSchedulerAgentId, AgentType::PLE_LOADER,
                m_PleKernelToPleLoaderAgentIdMapping[static_cast<PleOp*>(ptrPleOp)->m_PleKernelId]);
        }

        // Write After Read Dependency for [IfmStreamer][PleScheduler]
        AddWriteAfterReadDependency(AgentType::PLE_SCHEDULER, pleSchedulerAgentId, AgentType::IFM_STREAMER,
                                    m_OpToAgentIdMapping[input0Producer]);

        // Schedule Time Dependency for [IfmStreamer][PleScheduler]
        AddScheduleTimeDependency(AgentType::PLE_SCHEDULER, pleSchedulerAgentId, AgentType::IFM_STREAMER,
                                  m_OpToAgentIdMapping[input0Producer]);

        if (loadKernel)
        {
            // Schedule Time Dependency for [PleLoader][PleScheduler]
            AddScheduleTimeDependency(AgentType::PLE_SCHEDULER, pleSchedulerAgentId, AgentType::PLE_LOADER,
                                      pleLoaderAgentId);
        }
    }
    else
    {
        AgentIdType pleSchedulerAgentId = AddPleSchedulerToCommandStream(static_cast<PleOp*>(ptrPleOp));

        // Read After Write Dependency for [PleScheduler][MceScheduler]
        AddReadAfterWriteDependency(AgentType::PLE_SCHEDULER, pleSchedulerAgentId, AgentType::MCE_SCHEDULER,
                                    m_OpToAgentIdMapping[input0Producer]);
        if (loadKernel)
        {
            // Read After Write Dependency for [PleScheduler][PleLoader]
            AddReadAfterWriteDependency(
                AgentType::PLE_SCHEDULER, pleSchedulerAgentId, AgentType::PLE_LOADER,
                m_PleKernelToPleLoaderAgentIdMapping[static_cast<PleOp*>(ptrPleOp)->m_PleKernelId]);
        }

        // Schedule Time Dependency for [MceScheduler][PleScheduler]
        AddScheduleTimeDependency(AgentType::PLE_SCHEDULER, pleSchedulerAgentId, AgentType::MCE_SCHEDULER,
                                  m_OpToAgentIdMapping[input0Producer]);

        if (IsObjectOfType<MceOp>(input0Producer))
        {
            Buffer* mceInputBuffer = m_MergedOpGraph.GetInputs(input0Producer)[g_MceIfmBufferIndex];
            Op* mceInputProducer   = m_MergedOpGraph.GetProducer(mceInputBuffer);
            if (IsObjectOfType<PleOp>(mceInputProducer) && !mceInputBuffer->IsFullTensor())
            {
                // Strategy 0 cascade - need schedule dependency from previous PleS
                AddScheduleTimeDependency(AgentType::PLE_SCHEDULER, pleSchedulerAgentId, AgentType::PLE_SCHEDULER,
                                          m_OpToAgentIdMapping[mceInputProducer]);
            }
        }
    }
    ETHOSN_UNUSED(outputBuffer);
}

void CascadingCommandStreamGenerator::ProcessConcatOp(Op* const ptrConcatOp)
{
    // Get the input buffers to the Concat Op
    OpGraph::BufferList inputBuffers = m_MergedOpGraph.GetInputs(ptrConcatOp);
    assert(inputBuffers.size() >= 1);

    // Get the output buffer from the Concat Op
    Buffer* outputBuffer = m_MergedOpGraph.GetOutput(ptrConcatOp);
    assert(outputBuffer != nullptr && outputBuffer->m_Location == Location::Dram &&
           outputBuffer->m_BufferType.has_value());

    uint16_t outputBufferId =
        static_cast<uint16_t>(m_BufferManager.AddDram(outputBuffer->m_BufferType.value(), outputBuffer->m_SizeInBytes));

    // If this is an Intermediate Dram Buffer, add it to the IntermdiateDramBufToBufId map with the appropriate Id
    if (outputBuffer->m_BufferType.value() == BufferType::Intermediate)
    {
        assert(m_DramBufToBufIdMapping.find(outputBuffer) == m_DramBufToBufIdMapping.end());
        m_DramBufToBufIdMapping[outputBuffer] = outputBufferId;
    }

    uint32_t sramBufferOffset = 0U;
    uint32_t dramBufferOffset = 0U;
    uint32_t sramBufferSlotSize;

    for (auto inputBuffer : inputBuffers)
    {
        assert(inputBuffer->m_Location == Location::Dram && inputBuffer->m_BufferType.has_value());
        assert(inputBuffer->m_Format == CascadingBufferFormat::NHWCB ||
               inputBuffer->m_Format == CascadingBufferFormat::NHWC);

        // Temporary SRAM buffer used between IFMStreamer and OFMStreamer
        TensorShape sramBufferShape = { 1, 8, 8, utils::GetChannels(inputBuffer->m_TensorShape) };

        Buffer sramBuffer(Location::Sram, CascadingBufferFormat::NHWCB, sramBufferShape, sramBufferShape,
                          TraversalOrder::Xyz, utils::TotalSizeBytesNHWCB(sramBufferShape),
                          inputBuffer->m_QuantizationInfo);
        sramBuffer.m_NumStripes       = 2;
        sramBuffer.m_QuantizationInfo = inputBuffer->m_QuantizationInfo;
        sramBuffer.m_Offset           = sramBufferOffset;
        sramBuffer.m_SlotSizeInBytes  = utils::CalculateBufferSize(sramBufferShape, CascadingBufferFormat::NHWCB);

        sramBufferSlotSize = sramBuffer.m_SlotSizeInBytes / m_Capabilities.GetNumberOfSrams();

        assert(utils::DivRoundUp(utils::TotalSizeBytesNHWCB(sramBufferShape), m_Capabilities.GetNumberOfSrams()) <
               m_Capabilities.GetTotalSramSize());
        assert(sramBuffer.m_Offset.value() + (sramBuffer.m_NumStripes * sramBufferSlotSize) <
               m_Capabilities.GetTotalSramSize());

        // Set output DRAM buffer offset
        outputBuffer->m_Offset = dramBufferOffset;

        uint16_t inputBufferId;
        // If this is an Intermediate or Input Dram Buffer, add it to the DramBufToBufId map with the appropriate Id
        if (inputBuffer->m_BufferType.value() == BufferType::Intermediate ||
            inputBuffer->m_BufferType.value() == BufferType::Input)
        {
            inputBufferId = AddDramBufferAndCacheId(inputBuffer, ptrConcatOp);

            ConcatOp* const concatOp = static_cast<ConcatOp*>(ptrConcatOp);

            // Ifm Streamer Agent
            AgentIdType ifmStreamerAgentId = AddIfmStreamerToCommandStream(ptrConcatOp, inputBufferId, inputBuffer,
                                                                           &sramBuffer, concatOp->m_TransferFormat);

            // Ofm Streamer Agent
            AgentIdType ofmStreamerAgentId =
                AddOfmStreamerToCommandStream(ptrConcatOp, &sramBuffer, outputBufferId, outputBuffer);

            // Add 'Read After Write' dependency to the OfmStreamer agent
            // Read After Write Dependency for [OfmStreamer][IfmStreamer]
            AddReadAfterWriteDependency(AgentType::OFM_STREAMER, ofmStreamerAgentId, AgentType::IFM_STREAMER,
                                        ifmStreamerAgentId);

            // Add 'Write After Read' dependency information to the IfmStreamer agent
            // Write After Read Dependency for [IfmStreamer][OfmStreamer]
            AddWriteAfterReadDependency(AgentType::OFM_STREAMER, ofmStreamerAgentId, AgentType::IFM_STREAMER,
                                        ifmStreamerAgentId);

            // Add 'Schedule Time' dependency information to the OfmStreamer agent
            // Schedule Time Dependency for [IfmStreamer][OfmStreamer]
            AddScheduleTimeDependency(AgentType::OFM_STREAMER, ofmStreamerAgentId, AgentType::IFM_STREAMER,
                                      ifmStreamerAgentId);

            // Update the SRAM offset value for the next IfmStreamer Agent
            sramBufferOffset = sramBufferOffset + (sramBuffer.m_NumStripes * sramBufferSlotSize);

            // Update the Output DRAM offset value for the next OfmStreamer Agent
            std::tuple<bool, bool, bool> isHWCSplit =
                utils::IsSplitting(outputBuffer->m_TensorShape, inputBuffer->m_TensorShape);

            // Concatenation is happening in the Height dimension
            if (std::get<0>(isHWCSplit))
            {
                if (outputBuffer->m_Format == CascadingBufferFormat::NHWCB)
                {
                    assert(utils::GetHeight(inputBuffer->m_TensorShape) %
                               utils::GetHeight(m_Capabilities.GetBrickGroupShape()) ==
                           0);
                }

                uint32_t heightOffset = utils::CalculateBufferSize(inputBuffer->m_TensorShape, inputBuffer->m_Format);
                dramBufferOffset      = dramBufferOffset + heightOffset;
            }
            // Concatenation is happening in the Width dimension
            else if (std::get<1>(isHWCSplit))
            {
                uint32_t widthOffset;

                if (outputBuffer->m_Format == CascadingBufferFormat::NHWC)
                {
                    widthOffset =
                        utils::GetChannels(inputBuffer->m_TensorShape) * utils::GetWidth(inputBuffer->m_TensorShape);
                }
                else
                {
                    assert(utils::GetWidth(inputBuffer->m_TensorShape) %
                               utils::GetWidth(m_Capabilities.GetBrickGroupShape()) ==
                           0);

                    uint32_t widthInBrickGroups =
                        utils::DivRoundUp(utils::GetWidth(inputBuffer->m_TensorShape),
                                          utils::GetWidth(m_Capabilities.GetBrickGroupShape()));
                    uint32_t channelsInBrickGroups =
                        utils::DivRoundUp(utils::GetChannels(inputBuffer->m_TensorShape),
                                          utils::GetChannels(m_Capabilities.GetBrickGroupShape()));
                    uint32_t numberOfBrickGroups = channelsInBrickGroups * widthInBrickGroups;
                    widthOffset = numberOfBrickGroups * utils::CalculateBufferSize(m_Capabilities.GetBrickGroupShape(),
                                                                                   CascadingBufferFormat::NHWCB);
                }

                dramBufferOffset = dramBufferOffset + widthOffset;
            }
            // Concatenation is happening in the Depth/Channels dimension
            else if (std::get<2>(isHWCSplit))
            {
                assert(outputBuffer->m_Format == CascadingBufferFormat::NHWCB);
                uint32_t channelsInBrickGroups =
                    utils::DivRoundUp(utils::GetChannels(inputBuffer->m_TensorShape),
                                      utils::GetChannels(m_Capabilities.GetBrickGroupShape()));
                uint32_t depthOffset =
                    channelsInBrickGroups *
                    utils::CalculateBufferSize(m_Capabilities.GetBrickGroupShape(), outputBuffer->m_Format);
                dramBufferOffset = dramBufferOffset + depthOffset;
            }
        }
        else
        {
            assert(false);
        }
    }
}

void CascadingCommandStreamGenerator::ProcessSplitOp(Op* const ptrSplitOp)
{
    ETHOSN_UNUSED(ptrSplitOp);
}

void CascadingCommandStreamGenerator::ProcessSpaceToDepthOp(Op* const ptrSpaceToDepthOp)
{
    ETHOSN_UNUSED(ptrSpaceToDepthOp);
}

void CascadingCommandStreamGenerator::ProcessTransposeOp(Op* const ptrTransposeOp)
{
    ETHOSN_UNUSED(ptrTransposeOp);
}

// Private function to add IFM_STREAMER to the command stream
AgentIdType CascadingCommandStreamGenerator::AddIfmStreamerToCommandStream(Op* const ptrOp,
                                                                           const uint16_t inputDramBufferId,
                                                                           const Buffer* const inputDramBuffer,
                                                                           const Buffer* const inputSramBuffer,
                                                                           const CascadingBufferFormat transferFormat)
{
    assert(IsObjectOfType<DmaOp>(ptrOp) || IsObjectOfType<ConcatOp>(ptrOp));
    assert(inputSramBuffer->m_Format == CascadingBufferFormat::NHWCB);

    IfmS ifmStreamerData = {};

    ifmStreamerData.fmData.bufferId = inputDramBufferId;

    StreamersUtils::SetBufferDataType(ifmStreamerData.fmData, transferFormat);
    ifmStreamerData.fmData.fcafInfo.signedActivation = false;
    ifmStreamerData.fmData.fcafInfo.zeroPoint =
        ethosn::utils::NumericCast<uint8_t>(inputDramBuffer->m_QuantizationInfo.GetZeroPoint());

    CommonUtils::SetTileInfoForBuffer(m_Capabilities, ifmStreamerData.fmData.tile, inputSramBuffer);

    StreamersUtils::SetStripeHeightInfo(m_Capabilities, ifmStreamerData.fmData, inputSramBuffer->m_TensorShape,
                                        inputSramBuffer->m_StripeShape);
    StreamersUtils::SetStripeWidthInfo(m_Capabilities, ifmStreamerData.fmData, inputSramBuffer->m_TensorShape,
                                       inputSramBuffer->m_StripeShape);
    StreamersUtils::SetStripeChannelsInfo(ifmStreamerData.fmData, inputSramBuffer->m_TensorShape,
                                          inputSramBuffer->m_StripeShape);

    // The supertensor size is taken from either the SRAM buffer or the DRAM buffer, because these might be
    // different if there was a reshape. In the case of reshape then we use the SRAM shape so that is consistent
    // with the stripe shape which always comes from the SRAM buffer. If this is a concat/split though
    // then we need to use the DRAM shape because it will be a supertensor.
    if (utils::GetNumElements(inputSramBuffer->m_TensorShape) == utils::GetNumElements(inputDramBuffer->m_TensorShape))
    {
        StreamersUtils::SetSuperTensorSizeInCells(ifmStreamerData.fmData, inputSramBuffer->m_TensorShape,
                                                  transferFormat);
    }
    else
    {
        StreamersUtils::SetSuperTensorSizeInCells(ifmStreamerData.fmData, inputDramBuffer->m_TensorShape,
                                                  transferFormat);
    }

    StreamersUtils::SetStripeIdStrides(ifmStreamerData.fmData, inputSramBuffer->m_Order);
    ifmStreamerData.packedBoundaryThickness = inputSramBuffer->m_PackedBoundaryThickness;

    AgentDependencyInfo dependencyInfo = {};
    dependencyInfo.numStripesTotal     = ethosn::utils::NumericCast<uint16_t>(
        utils::GetNumStripesTotal(inputSramBuffer->m_TensorShape, inputSramBuffer->m_StripeShape) *
        inputSramBuffer->m_NumLoads);

    Agent ifmStreamerAgent{ ifmStreamerData, dependencyInfo };

    // Push the Ifm Streamer agent to the command stream
    AgentIdType agentId         = m_CommandStreamAgents.size();
    m_OpToAgentIdMapping[ptrOp] = agentId;
    m_CommandStreamAgents.push_back(ifmStreamerAgent);

    return agentId;
}

// Private function to add WGT_STREAMER to the command stream
AgentIdType CascadingCommandStreamGenerator::AddWeightStreamerToCommandStream(DmaOp* const ptrDmaOp)
{
    // Get the input buffer to the Dma Op
    OpGraph::BufferList inputBuffers = m_MergedOpGraph.GetInputs(ptrDmaOp);
    Buffer* weightsDramBuffer        = inputBuffers[g_DmaInputBufferIndex];
    Buffer* weightsSramBuffer        = m_MergedOpGraph.GetOutput(ptrDmaOp);

    // Get the Mce consumer of the weights buffer
    auto weightBufferConsumer = m_MergedOpGraph.GetConsumer(weightsSramBuffer, 0);
    assert(weightBufferConsumer.first != nullptr && IsObjectOfType<MceOp>(weightBufferConsumer.first));

    Buffer* ifmBuffer = m_MergedOpGraph.GetInputs(weightBufferConsumer.first)[0];
    Buffer* ofmBuffer = m_MergedOpGraph.GetOutput(weightBufferConsumer.first);

    WgtS weightStreamerData = {};

    EncodedWeights* encodedWeights          = weightsDramBuffer->m_EncodedWeights.get();
    std::vector<uint8_t>& compressedWeights = encodedWeights->m_Data;
    std::vector<uint8_t> metadataBytes;
    metadataBytes.assign(
        reinterpret_cast<const uint8_t*>(encodedWeights->m_Metadata.data()),
        reinterpret_cast<const uint8_t*>(encodedWeights->m_Metadata.data() + encodedWeights->m_Metadata.size()));
    weightStreamerData.bufferId = ethosn::utils::NumericCast<uint16_t>(
        m_BufferManager.AddDramConstant(BufferType::ConstantDma, compressedWeights));
    weightStreamerData.metadataBufferId = ethosn::utils::NumericCast<uint16_t>(
        m_BufferManager.AddDramConstant(BufferType::ConstantControlUnit, metadataBytes));
    CommonUtils::SetTileInfoForBuffer(m_Capabilities, weightStreamerData.tile, weightsSramBuffer);

    weightStreamerData.numStripes.ifmChannels =
        ethosn::utils::NumericCast<uint16_t>(utils::GetNumStripesC(ifmBuffer->m_TensorShape, ifmBuffer->m_StripeShape));
    weightStreamerData.numStripes.ofmChannels =
        ethosn::utils::NumericCast<uint16_t>(utils::GetNumStripesC(ofmBuffer->m_TensorShape, ofmBuffer->m_StripeShape));
    weightStreamerData.stripeIdStrides.ifmChannels = 1;
    weightStreamerData.stripeIdStrides.ofmChannels =
        ethosn::utils::NumericCast<uint16_t>(weightStreamerData.numStripes.ifmChannels * weightsSramBuffer->m_NumLoads);

    AgentDependencyInfo dependencyInfo = {};

    dependencyInfo.numStripesTotal = ethosn::utils::NumericCast<uint16_t>(
        utils::GetNumStripesTotal(weightsSramBuffer->m_TensorShape, weightsSramBuffer->m_StripeShape) *
        weightsSramBuffer->m_NumLoads);
    Agent weightStreamerAgent{ weightStreamerData, dependencyInfo };

    // Push the Weight Streamer agent to the command stream
    AgentIdType agentId            = m_CommandStreamAgents.size();
    m_OpToAgentIdMapping[ptrDmaOp] = agentId;
    m_CommandStreamAgents.push_back(weightStreamerAgent);

    return agentId;
}

// Private function to add MCE_SCHEDULER to the command stream
AgentIdType CascadingCommandStreamGenerator::AddMceSchedulerToCommandStream(MceOp* const ptrMceOp,
                                                                            const PleKernelId pleKernelId)
{
    // Get the input buffers to the Mce Op
    OpGraph::BufferList inputBuffers = m_MergedOpGraph.GetInputs(ptrMceOp);
    Buffer* inputBuffer              = inputBuffers[g_MceIfmBufferIndex];
    Buffer* weightBuffer             = inputBuffers[g_MceWeightBufferIndex];

    // Get the output buffer from the Mce Op
    Buffer* outputBuffer = m_MergedOpGraph.GetOutput(ptrMceOp);

    MceS mceSchedulerData = {};

    CommonUtils::SetTileInfoForBuffer(m_Capabilities, mceSchedulerData.ifmTile, inputBuffer);

    CommonUtils::SetTileInfoForBuffer(m_Capabilities, mceSchedulerData.wgtTile, weightBuffer);

    mceSchedulerData.blockSize.width  = ethosn::utils::NumericCast<uint8_t>(ptrMceOp->m_BlockConfig.m_BlockWidth());
    mceSchedulerData.blockSize.height = ethosn::utils::NumericCast<uint8_t>(ptrMceOp->m_BlockConfig.m_BlockHeight());

    MceSUtils::setMcesOpMode(mceSchedulerData, ptrMceOp->m_Op);

    MceSUtils::SetMcesOfmHeightStripeInfo(mceSchedulerData, outputBuffer->m_TensorShape, ptrMceOp->m_OutputStripeShape);
    MceSUtils::SetMcesOfmWidthStripeInfo(mceSchedulerData, outputBuffer->m_TensorShape, ptrMceOp->m_OutputStripeShape);
    MceSUtils::SetMcesOfmChannelsStripeInfo(mceSchedulerData, outputBuffer->m_TensorShape,
                                            ptrMceOp->m_OutputStripeShape);
    MceSUtils::SetMcesIfmChannelsStripeInfo(mceSchedulerData, inputBuffer->m_TensorShape, inputBuffer->m_StripeShape);

    MceSUtils::SetStripeIdStrides(mceSchedulerData, outputBuffer->m_Order);

    mceSchedulerData.convStrideXy.x = ethosn::utils::NumericCast<uint8_t>(ptrMceOp->m_Stride.m_X);
    mceSchedulerData.convStrideXy.y = ethosn::utils::NumericCast<uint8_t>(ptrMceOp->m_Stride.m_Y);
    mceSchedulerData.ifmZeroPoint =
        ethosn::utils::NumericCast<uint16_t>(inputBuffer->m_QuantizationInfo.GetZeroPoint());

    MceSUtils::setMcesAlgorithm(mceSchedulerData, ptrMceOp->m_Algo);

    if (ptrMceOp->m_Stride.m_X == 1 && ptrMceOp->m_Stride.m_Y == 1)
    {
        for (int i = 0; i < 4; i++)
        {
            mceSchedulerData.filterShape[i].height =
                ethosn::utils::NumericCast<uint8_t>(weightBuffer->m_TensorShape[0]);
            mceSchedulerData.filterShape[i].width = ethosn::utils::NumericCast<uint8_t>(weightBuffer->m_TensorShape[1]);

            mceSchedulerData.ifmDeltaDefault[i].height = static_cast<int8_t>(
                mceSchedulerData.filterShape[i].height / 2 + inputBuffer->m_PackedBoundaryThickness.bottom);
            mceSchedulerData.ifmDeltaDefault[i].width = static_cast<int8_t>(
                mceSchedulerData.filterShape[i].width / 2 + inputBuffer->m_PackedBoundaryThickness.right);

            mceSchedulerData.ifmDeltaEdge[i].height =
                static_cast<int8_t>(inputBuffer->m_TensorShape[1] - outputBuffer->m_TensorShape[1]);
            mceSchedulerData.ifmDeltaEdge[i].width =
                static_cast<int8_t>(inputBuffer->m_TensorShape[2] - outputBuffer->m_TensorShape[2]);

            mceSchedulerData.padding[i].left = ethosn::utils::NumericCast<uint8_t>(ptrMceOp->m_PadLeft);
            mceSchedulerData.padding[i].top  = ethosn::utils::NumericCast<uint8_t>(ptrMceOp->m_PadTop);
        }
    }
    else if (ptrMceOp->m_Stride.m_X == 2 && ptrMceOp->m_Stride.m_Y == 2)
    {
        MceSUtils::setMcesStridedConvolutionData(mceSchedulerData, m_MergedOpGraph, ptrMceOp);
    }
    else
    {
        assert(false);
    }

    mceSchedulerData.ifmStripeShapeDefault.height =
        static_cast<uint16_t>(inputBuffer->m_StripeShape[1] + inputBuffer->m_PackedBoundaryThickness.top +
                              inputBuffer->m_PackedBoundaryThickness.bottom);
    mceSchedulerData.ifmStripeShapeDefault.width =
        static_cast<uint16_t>(inputBuffer->m_StripeShape[2] + inputBuffer->m_PackedBoundaryThickness.left +
                              inputBuffer->m_PackedBoundaryThickness.right);
    // Note that the IFM edge stripe shape is not used when packed boundary data is used, so we don't need to account
    // for that here.
    mceSchedulerData.ifmStripeShapeEdge.height = CommonUtils::CalculateEdgeSize(
        utils::GetHeight(inputBuffer->m_TensorShape), utils::GetHeight(inputBuffer->m_StripeShape));
    mceSchedulerData.ifmStripeShapeEdge.width = CommonUtils::CalculateEdgeSize(
        utils::GetWidth(inputBuffer->m_TensorShape), utils::GetWidth(inputBuffer->m_StripeShape));

    mceSchedulerData.reluActiv.min = ptrMceOp->m_LowerBound;
    mceSchedulerData.reluActiv.max = ptrMceOp->m_UpperBound;
    mceSchedulerData.pleKernelId   = pleKernelId;

    mceSchedulerData.isPackedBoundaryX =
        (inputBuffer->m_PackedBoundaryThickness.left + inputBuffer->m_PackedBoundaryThickness.right) > 0;
    mceSchedulerData.isPackedBoundaryY =
        (inputBuffer->m_PackedBoundaryThickness.top + inputBuffer->m_PackedBoundaryThickness.bottom) > 0;

    AgentDependencyInfo dependencyInfo = {};
    dependencyInfo.numStripesTotal     = ethosn::utils::NumericCast<uint16_t>(
        mceSchedulerData.numStripes.ifmChannels * mceSchedulerData.numStripes.ofmChannels *
        mceSchedulerData.numStripes.ofmWidth * mceSchedulerData.numStripes.ofmHeight);

    Agent mceSchedulerAgent{ mceSchedulerData, dependencyInfo };

    // Push the Mce Scheduler agent to the command stream
    AgentIdType agentId            = m_CommandStreamAgents.size();
    m_OpToAgentIdMapping[ptrMceOp] = agentId;
    m_CommandStreamAgents.push_back(mceSchedulerAgent);

    return agentId;
}

// Private function to add PLE_LOADER to the command stream
AgentIdType CascadingCommandStreamGenerator::AddPleLoaderToCommandStream(PleOp* const ptrPleOp)
{
    // Create a new Ple Loader agent
    PleL pleLoaderData        = {};
    pleLoaderData.pleKernelId = ptrPleOp->m_PleKernelId;
    pleLoaderData.sramAddr    = ethosn::utils::NumericCast<uint16_t>(ptrPleOp->m_Offset.value());

    AgentDependencyInfo dependencyInfo = {};
    dependencyInfo.numStripesTotal     = 1;

    Agent pleLoaderAgent{ pleLoaderData, dependencyInfo };

    // Push the Ple Loader agent to the command stream
    AgentIdType agentId                                           = m_CommandStreamAgents.size();
    m_PleKernelToPleLoaderAgentIdMapping[ptrPleOp->m_PleKernelId] = agentId;
    m_CommandStreamAgents.push_back(pleLoaderAgent);

    return agentId;
}

// Private function to add PLE_SCHEDULER to the command stream
AgentIdType CascadingCommandStreamGenerator::AddPleSchedulerToCommandStream(PleOp* const ptrPleOp)
{
    // Get the input buffers to the Ple Op
    OpGraph::BufferList inputBuffers = m_MergedOpGraph.GetInputs(ptrPleOp);
    assert(inputBuffers.size() == 1 || inputBuffers.size() == 2);

    Buffer* inputBuffer0 = inputBuffers[g_PleInputBuffer0Index];

    // Get the output buffer from the Ple Op
    Buffer* outputBuffer = m_MergedOpGraph.GetOutput(ptrPleOp);

    PleS pleS = {};

    pleS.ofmZeroPoint = ethosn::utils::NumericCast<int16_t>(outputBuffer->m_QuantizationInfo.GetZeroPoint());

    PleSUtils::SetPlesHeightStripeInfo(pleS, outputBuffer->m_TensorShape, ptrPleOp->m_OutputStripeShape);
    PleSUtils::SetPlesWidthStripeInfo(pleS, outputBuffer->m_TensorShape, ptrPleOp->m_OutputStripeShape);
    PleSUtils::SetPlesChannelsStripeInfo(pleS, outputBuffer->m_TensorShape, ptrPleOp->m_OutputStripeShape);

    PleSUtils::SetStripeIdStrides(pleS, outputBuffer);

    // Can't use CommonUtils::SetTileInfoForBuffer because PLE OFM tile might be different to OfmS tile
    // (strategies where OfmS does the full height but PLE does partial height)
    PleSUtils::SetPlesTileInfo(m_Capabilities, pleS, outputBuffer);

    // Calculate input mode of Ple OP dependent on input buffer producer.
    auto pleOpProducer = m_MergedOpGraph.GetProducer(inputBuffer0);
    if (inputBuffer0->m_Location == Location::Sram)
    {
        pleS.inputMode = PleInputMode::SRAM;
    }
    else if (inputBuffer0->m_Location == Location::PleInputSram)
    {
        PleSUtils::SetFusedPleSInputMode(pleS, static_cast<MceOp*>(pleOpProducer));
    }
    else
    {
        assert(false);
    }

    pleS.pleKernelSramAddr = ethosn::utils::NumericCast<uint16_t>(ptrPleOp->m_Offset.value());

    pleS.pleKernelId = ptrPleOp->m_PleKernelId;

    if (pleS.inputMode == PleInputMode::SRAM)
    {
        CommonUtils::SetTileInfoForBuffer(m_Capabilities, pleS.ifmTile0, inputBuffer0);

        const double outputScale = outputBuffer->m_QuantizationInfo.GetScale();
        const double inputScale0 = inputBuffer0->m_QuantizationInfo.GetScale();
        uint16_t multiplier0;
        uint16_t shift0;
        utils::CalculateRescaleMultiplierAndShift(inputScale0 / outputScale, multiplier0, shift0);

        pleS.ifmInfo0 = { ethosn::utils::NumericCast<int16_t>(inputBuffer0->m_QuantizationInfo.GetZeroPoint()),
                          multiplier0, shift0 };

        if (inputBuffers.size() == 2)
        {
            Buffer* inputBuffer1 = inputBuffers[g_PleInputBuffer1Index];

            const double inputScale1 = inputBuffer1->m_QuantizationInfo.GetScale();
            uint16_t multiplier1;
            uint16_t shift1;
            utils::CalculateRescaleMultiplierAndShift(inputScale1 / outputScale, multiplier1, shift1);

            CommonUtils::SetTileInfoForBuffer(m_Capabilities, pleS.ifmTile1, inputBuffer1);

            pleS.ifmInfo1 = { ethosn::utils::NumericCast<int16_t>(inputBuffer1->m_QuantizationInfo.GetZeroPoint()),
                              multiplier1, shift1 };
        }
    }

    AgentData agentData{ pleS };

    AgentDependencyInfo info = {};
    info.numStripesTotal     = ethosn::utils::NumericCast<uint16_t>(
        utils::GetNumStripesTotal(outputBuffer->m_TensorShape, ptrPleOp->m_OutputStripeShape));

    Agent pleSchedulerAgent{ agentData, info };

    // Push the Ple Scheduler agent to the command stream
    AgentIdType agentId            = m_CommandStreamAgents.size();
    m_OpToAgentIdMapping[ptrPleOp] = agentId;
    m_CommandStreamAgents.push_back(pleSchedulerAgent);

    return agentId;
}

// Private function to add OFM_STREAMER to the command stream
AgentIdType CascadingCommandStreamGenerator::AddOfmStreamerToCommandStream(Op* const ptrOp,
                                                                           const Buffer* const outputSramBuffer,
                                                                           const uint16_t outputDramBufferId,
                                                                           const Buffer* const outputDramBuffer)
{
    assert(IsObjectOfType<DmaOp>(ptrOp) || IsObjectOfType<ConcatOp>(ptrOp));
    assert(outputSramBuffer->m_Format == CascadingBufferFormat::NHWCB);

    OfmS ofmStreamerData = {};

    if (outputDramBuffer->m_Offset.has_value())
    {
        ofmStreamerData.fmData.dramOffset = outputDramBuffer->m_Offset.value();
    }

    ofmStreamerData.fmData.bufferId = outputDramBufferId;

    StreamersUtils::SetBufferDataType(ofmStreamerData.fmData, outputDramBuffer->m_Format);

    ofmStreamerData.fmData.fcafInfo.signedActivation = false;
    ofmStreamerData.fmData.fcafInfo.zeroPoint =
        ethosn::utils::NumericCast<uint8_t>(outputDramBuffer->m_QuantizationInfo.GetZeroPoint());

    CommonUtils::SetTileInfoForBuffer(m_Capabilities, ofmStreamerData.fmData.tile, outputSramBuffer);

    StreamersUtils::SetStripeHeightInfo(m_Capabilities, ofmStreamerData.fmData, outputSramBuffer->m_TensorShape,
                                        outputSramBuffer->m_StripeShape);
    StreamersUtils::SetStripeWidthInfo(m_Capabilities, ofmStreamerData.fmData, outputSramBuffer->m_TensorShape,
                                       outputSramBuffer->m_StripeShape);
    StreamersUtils::SetStripeChannelsInfo(ofmStreamerData.fmData, outputSramBuffer->m_TensorShape,
                                          outputSramBuffer->m_StripeShape);

    // The supertensor size is taken from either the SRAM buffer or the DRAM buffer, because these might be
    // different if there was a reshape. In the case of reshape then we use the SRAM shape so that is consistent
    // with the stripe shape which always comes from the SRAM buffer. If this is a concat/split though
    // then we need to use the DRAM shape because it will be a supertensor.
    if (utils::GetNumElements(outputSramBuffer->m_TensorShape) ==
        utils::GetNumElements(outputDramBuffer->m_TensorShape))
    {
        StreamersUtils::SetSuperTensorSizeInCells(ofmStreamerData.fmData, outputSramBuffer->m_TensorShape,
                                                  outputDramBuffer->m_Format);
    }
    else
    {
        StreamersUtils::SetSuperTensorSizeInCells(ofmStreamerData.fmData, outputDramBuffer->m_TensorShape,
                                                  outputDramBuffer->m_Format);
    }

    StreamersUtils::SetStripeIdStrides(ofmStreamerData.fmData, outputSramBuffer->m_Order);

    AgentDependencyInfo dependencyInfo = {};
    dependencyInfo.numStripesTotal     = ethosn::utils::NumericCast<uint16_t>(
        utils::GetNumStripesTotal(outputSramBuffer->m_TensorShape, outputSramBuffer->m_StripeShape));

    Agent ofmStreamerAgent{ ofmStreamerData, dependencyInfo };

    // Push the Ofm Streamer agent to the command stream
    AgentIdType agentId         = m_CommandStreamAgents.size();
    m_OpToAgentIdMapping[ptrOp] = agentId;
    m_CommandStreamAgents.push_back(ofmStreamerAgent);

    return agentId;
}

// Private function to add ReadAfterWrite Dependency
// Consumer agent creates and own the dependency
inline void CascadingCommandStreamGenerator::AddReadAfterWriteDependency(const AgentType consumerAgentType,
                                                                         const AgentIdType consumerAgentId,
                                                                         const AgentType producerAgentType,
                                                                         const AgentIdType producerAgentId)
{
    AgentIdType relativeAgentId = consumerAgentId - producerAgentId;
    assert(relativeAgentId <= g_MaxRelativeAgentPosition);

    Dependency newDependency      = {};
    newDependency.relativeAgentId = static_cast<RelativeAgentIdType>(relativeAgentId);
    FillConsumerAgentDependency(newDependency, consumerAgentType, consumerAgentId, producerAgentType, producerAgentId);
    DependencyUtils::AddDependency(m_CommandStreamAgents[consumerAgentId].info.readDependencies, newDependency);
}

// Private function to add SRAM Overlap Dependency
// Consumer agent creates and own the dependency
inline void CascadingCommandStreamGenerator::AddSramOverlapDependency(
    const command_stream::cascading::AgentType consumerAgentType,
    const AgentIdType consumerAgentId,
    const command_stream::cascading::AgentType producerAgentType,
    const AgentIdType producerAgentId)
{
    AgentIdType relativeAgentId = consumerAgentId - producerAgentId;
    assert(relativeAgentId <= g_MaxRelativeAgentPosition);

    Dependency newDependency      = {};
    newDependency.relativeAgentId = static_cast<RelativeAgentIdType>(relativeAgentId);
    FillConsumerAgentDependency(newDependency, consumerAgentType, consumerAgentId, producerAgentType, producerAgentId);
    DependencyUtils::AddDependency(m_CommandStreamAgents[consumerAgentId].info.readDependencies, newDependency);
}

// Private function to add WriteAfterRead Dependency
// Last consumer agent creates the dependency and assign it to the producer agent
inline void CascadingCommandStreamGenerator::AddWriteAfterReadDependency(const AgentType consumerAgentType,
                                                                         const AgentIdType consumerAgentId,
                                                                         const AgentType producerAgentType,
                                                                         const AgentIdType producerAgentId)
{
    AgentIdType relativeAgentId = consumerAgentId - producerAgentId;
    assert(relativeAgentId <= g_MaxRelativeAgentPosition);

    Dependency newDependency      = {};
    newDependency.relativeAgentId = static_cast<RelativeAgentIdType>(relativeAgentId);
    FillProducerAgentDependency(newDependency, consumerAgentType, consumerAgentId, producerAgentType, producerAgentId);
    DependencyUtils::AddDependency(m_CommandStreamAgents[producerAgentId].info.writeDependencies, newDependency);
}

// Private function to add ScheduleTime Dependency
// First consumer agent creates the dependency and assign it to the producer agent
inline void CascadingCommandStreamGenerator::AddScheduleTimeDependency(const AgentType consumerAgentType,
                                                                       const AgentIdType consumerAgentId,
                                                                       const AgentType producerAgentType,
                                                                       const AgentIdType producerAgentId)
{
    AgentIdType relativeAgentId = consumerAgentId - producerAgentId;
    assert(relativeAgentId <= g_MaxRelativeAgentPosition);

    Dependency newDependency      = {};
    newDependency.relativeAgentId = static_cast<RelativeAgentIdType>(relativeAgentId);
    FillProducerAgentDependency(newDependency, consumerAgentType, consumerAgentId, producerAgentType, producerAgentId);
    DependencyUtils::AddDependency(m_CommandStreamAgents[producerAgentId].info.scheduleDependencies, newDependency);
}

// Private function to fill the dependency data for Read After Write or SRAM Overlap dependencies
void CascadingCommandStreamGenerator::FillConsumerAgentDependency(
    command_stream::cascading::Dependency& consumerAgentDependency,
    const command_stream::cascading::AgentType consumerAgentType,
    const AgentIdType consumerAgentId,
    const command_stream::cascading::AgentType producerAgentType,
    const AgentIdType producerAgentId)
{
    Agent& consumerAgent = m_CommandStreamAgents[consumerAgentId];
    Agent& producerAgent = m_CommandStreamAgents[producerAgentId];

    // Add a new 'Read After Write' dependency
    switch (consumerAgentType)
    {
        case AgentType::IFM_STREAMER:
        {
            // Read After Write Dependency for [IfmStreamer][OfmStreamer]
            if (producerAgentType == AgentType::OFM_STREAMER)
            {
                // The IfmS should wait until the OfmS has completely finished.
                consumerAgentDependency.outerRatio.other = producerAgent.info.numStripesTotal;
                consumerAgentDependency.outerRatio.self  = consumerAgent.info.numStripesTotal;

                consumerAgentDependency.innerRatio.other = producerAgent.info.numStripesTotal;
                consumerAgentDependency.innerRatio.self  = 1;

                consumerAgentDependency.boundary = 0;
            }
            break;
        }

        case AgentType::WGT_STREAMER:
        {
            // Sram Overlap Dependency for [WeightStreamer][OfmStreamer]
            if (producerAgentType == AgentType::OFM_STREAMER)
            {
                // The WgtS should wait until the OfmS has completely finished.
                consumerAgentDependency.outerRatio.other = producerAgent.info.numStripesTotal;
                consumerAgentDependency.outerRatio.self  = consumerAgent.info.numStripesTotal;

                consumerAgentDependency.innerRatio.other = producerAgent.info.numStripesTotal;
                consumerAgentDependency.innerRatio.self  = 1;

                consumerAgentDependency.boundary = 0;
            }
            // Sram Overlap Dependency for [WeightStreamer][MceScheduler]
            else if (producerAgentType == AgentType::MCE_SCHEDULER)
            {
                // The WgtS needs to wait for an MceS in the same section if this is a strategy 1 cascade,
                // because these weights shouldn't be loaded until the weights from the previous layer are finished with.
                // The WgtS should wait until the MceS has completely finished.
                consumerAgentDependency.outerRatio.other = producerAgent.info.numStripesTotal;
                consumerAgentDependency.outerRatio.self  = consumerAgent.info.numStripesTotal;

                consumerAgentDependency.innerRatio.other = producerAgent.info.numStripesTotal;
                consumerAgentDependency.innerRatio.self  = 1;

                consumerAgentDependency.boundary = 0;
            }
            break;
        }

        case AgentType::MCE_SCHEDULER:
        {
            // Read After Write Dependency for [MceScheduler][IfmStreamer]
            if (producerAgentType == AgentType::IFM_STREAMER)
            {
                if (producerAgent.data.ifm.fmData.numStripes.height > 1 &&
                    producerAgent.data.ifm.fmData.numStripes.width > 1)
                {
                    // Splitting width and height => outer ratio is for each row
                    consumerAgentDependency.outerRatio.other = ethosn::utils::NumericCast<uint16_t>(
                        producerAgent.data.ifm.fmData.numStripes.width *
                        // Note we use the ifmChannels from the MceS, not the IfmS, so that this is correct for depthwise
                        // (where IfmS might have multiple IFM stripes but MceS won't)
                        consumerAgent.data.mce.numStripes.ifmChannels);
                    consumerAgentDependency.outerRatio.self = ethosn::utils::NumericCast<uint16_t>(
                        consumerAgent.data.mce.numStripes.ofmWidth * consumerAgent.data.mce.numStripes.ifmChannels);
                }
                else
                {
                    // Not splitting width and height => outer ratio is not needed (set to max)
                    consumerAgentDependency.outerRatio.other = producerAgent.info.numStripesTotal;
                    consumerAgentDependency.outerRatio.self  = consumerAgent.info.numStripesTotal;
                }

                // The MceS can process more data than is loaded by the IfmS (e.g. two stripes at a time)
                uint16_t widthRatio  = ethosn::utils::NumericCast<uint16_t>(utils::DivRoundUp(
                    consumerAgent.data.mce.numStripes.ofmWidth, producerAgent.data.ifm.fmData.numStripes.width));
                uint16_t heightRatio = ethosn::utils::NumericCast<uint16_t>(utils::DivRoundUp(
                    consumerAgent.data.mce.numStripes.ofmHeight, producerAgent.data.ifm.fmData.numStripes.height));

                if (consumerAgent.data.mce.mceOpMode == MceOperation::DEPTHWISE_CONVOLUTION)
                {
                    assert(consumerAgent.data.mce.numStripes.ifmChannels == 1);
                }
                else
                {
                    assert(consumerAgent.data.mce.numStripes.ifmChannels ==
                           producerAgent.data.ifm.fmData.numStripes.channels);
                }

                consumerAgentDependency.innerRatio.other =
                    ethosn::utils::NumericCast<uint16_t>(widthRatio * heightRatio);
                consumerAgentDependency.innerRatio.self = 1;

                // MceS needs to wait for two IfmS stripes at the start of each outer ratio if neighbouring data
                // is needed. This is not applicable if all the boundary data is packed though.
                if (!(consumerAgent.data.mce.isPackedBoundaryX && consumerAgent.data.mce.isPackedBoundaryY) &&
                    ((producerAgent.data.ifm.fmData.numStripes.height > 1 &&
                      consumerAgent.data.mce.filterShape[0].height > 1) ||
                     (producerAgent.data.ifm.fmData.numStripes.width > 1 &&
                      consumerAgent.data.mce.filterShape[0].width > 1)))
                {
                    consumerAgentDependency.boundary = 1;
                }
                else
                {
                    consumerAgentDependency.boundary = 0;
                }
            }
            // Read After Write Dependency for [MceScheduler][WeightStreamer]
            else if (producerAgentType == AgentType::WGT_STREAMER)
            {
                // MCE always traverses in IXYO order. Each MCE stripe needs a new weight stripe, unless a weight stripe
                // can be re-used which can only happen if we are not IFM splitting and we are moving in XY.

                // Outer ratio is not needed (set to max)
                consumerAgentDependency.outerRatio.other = producerAgent.info.numStripesTotal;
                consumerAgentDependency.outerRatio.self  = consumerAgent.info.numStripesTotal;

                if (consumerAgent.data.mce.numStripes.ifmChannels == 1)
                {
                    // Weight stripes can be re-used as we move in XY
                    consumerAgentDependency.innerRatio.self = ethosn::utils::NumericCast<uint16_t>(
                        consumerAgent.data.mce.numStripes.ofmHeight * consumerAgent.data.mce.numStripes.ofmWidth);
                    consumerAgentDependency.innerRatio.other = 1;
                }
                else
                {
                    // No re-use, 1:1 dependency
                    consumerAgentDependency.innerRatio.self  = 1;
                    consumerAgentDependency.innerRatio.other = 1;
                }

                consumerAgentDependency.boundary = 0;
            }
            // Read After Write Dependency for [MceScheduler][PleScheduler]
            else if (producerAgentType == AgentType::PLE_SCHEDULER)
            {
                // Calculate outer ratios using number of stripes
                consumerAgentDependency.outerRatio.other = ethosn::utils::NumericCast<uint16_t>(
                    producerAgent.data.pleS.numStripes.height * producerAgent.data.pleS.numStripes.width *
                    producerAgent.data.pleS.numStripes.channels);
                consumerAgentDependency.outerRatio.self = ethosn::utils::NumericCast<uint16_t>(
                    consumerAgent.data.mce.numStripes.ofmHeight * consumerAgent.data.mce.numStripes.ofmWidth *
                    consumerAgent.data.mce.numStripes.ofmChannels);

                // Calculate inner ratios using ratio of stripe size
                uint16_t widthRatio   = ethosn::utils::NumericCast<uint16_t>(utils::DivRoundUp(
                    producerAgent.data.pleS.numStripes.width, consumerAgent.data.mce.numStripes.ofmWidth));
                uint16_t heightRatio  = ethosn::utils::NumericCast<uint16_t>(utils::DivRoundUp(
                    producerAgent.data.pleS.numStripes.height, consumerAgent.data.mce.numStripes.ofmHeight));
                uint16_t channelRatio = ethosn::utils::NumericCast<uint16_t>(utils::DivRoundUp(
                    producerAgent.data.pleS.numStripes.channels, consumerAgent.data.mce.numStripes.ofmChannels));

                consumerAgentDependency.innerRatio.other =
                    ethosn::utils::NumericCast<uint16_t>(widthRatio * heightRatio * channelRatio);
                consumerAgentDependency.innerRatio.self = 1;

                if ((producerAgent.data.pleS.numStripes.height > 1 &&
                     consumerAgent.data.mce.filterShape[0].height > 1) ||
                    (producerAgent.data.pleS.numStripes.width > 1 && consumerAgent.data.mce.filterShape[0].width > 1))
                {
                    consumerAgentDependency.boundary = 1;
                }
                else
                {
                    consumerAgentDependency.boundary = 0;
                }
            }
            else
            {
                assert(false);
            }
            break;
        }

        case AgentType::PLE_LOADER:
        {
            assert(false);
            break;
        }

        case AgentType::PLE_SCHEDULER:
        {
            // Read After Write Dependency for [PleScheduler][IfmStreamer]
            if (producerAgentType == AgentType::IFM_STREAMER)
            {
                // Calculate outer ratios using number of stripes
                consumerAgentDependency.outerRatio.other = ethosn::utils::NumericCast<uint16_t>(
                    producerAgent.data.ifm.fmData.numStripes.width * producerAgent.data.ifm.fmData.numStripes.height *
                    producerAgent.data.ifm.fmData.numStripes.channels);
                consumerAgentDependency.outerRatio.self = ethosn::utils::NumericCast<uint16_t>(
                    consumerAgent.data.pleS.numStripes.height * consumerAgent.data.pleS.numStripes.width *
                    consumerAgent.data.pleS.numStripes.channels);
            }
            // Read After Write Dependency for [PleScheduler][MceScheduler]
            else if (producerAgentType == AgentType::MCE_SCHEDULER)
            {
                // Outer ratio not used (set to max)
                consumerAgentDependency.outerRatio.other = producerAgent.info.numStripesTotal;
                consumerAgentDependency.outerRatio.self  = consumerAgent.info.numStripesTotal;

                // Calculate inner ratios using ratio of stripe size
                uint16_t widthRatio   = ethosn::utils::NumericCast<uint16_t>(utils::DivRoundUp(
                    producerAgent.data.mce.numStripes.ofmWidth, consumerAgent.data.pleS.numStripes.width));
                uint16_t heightRatio  = ethosn::utils::NumericCast<uint16_t>(utils::DivRoundUp(
                    producerAgent.data.mce.numStripes.ofmHeight, consumerAgent.data.pleS.numStripes.height));
                uint16_t channelRatio = ethosn::utils::NumericCast<uint16_t>(utils::DivRoundUp(
                    producerAgent.data.mce.numStripes.ofmChannels, consumerAgent.data.pleS.numStripes.channels));

                consumerAgentDependency.innerRatio.other = ethosn::utils::NumericCast<uint16_t>(
                    widthRatio * heightRatio * channelRatio * producerAgent.data.mce.numStripes.ifmChannels);
                consumerAgentDependency.innerRatio.self = 1;

                // Set boundary to 1 if producer stripe count is not a factor of consumer stripe count
                uint16_t numberOfIfmStripesInXYDimProducer = ethosn::utils::NumericCast<uint16_t>(
                    producerAgent.data.mce.numStripes.ofmWidth * producerAgent.data.mce.numStripes.ofmHeight);
                uint16_t numberOfIfmStripesInXYDimConsumer = ethosn::utils::NumericCast<uint16_t>(
                    consumerAgent.data.pleS.numStripes.width * consumerAgent.data.pleS.numStripes.height);

                uint16_t ifmStripeRemainder = ethosn::utils::NumericCast<uint16_t>(numberOfIfmStripesInXYDimConsumer %
                                                                                   numberOfIfmStripesInXYDimProducer);
                if (ifmStripeRemainder == 0)
                {
                    consumerAgentDependency.boundary = 0;
                }
                else
                {
                    consumerAgentDependency.boundary = 1;
                }
            }
            // Read After Write Dependency for [PleScheduler][PleLoader]
            else if (producerAgentType == AgentType::PLE_LOADER)
            {
                consumerAgentDependency.outerRatio.other = 1U;
                consumerAgentDependency.outerRatio.self  = ethosn::utils::NumericCast<uint16_t>(
                    consumerAgent.data.pleS.numStripes.height * consumerAgent.data.pleS.numStripes.width *
                    consumerAgent.data.pleS.numStripes.channels);
            }
            else
            {
                assert(false);
            }
            break;
        }

        case AgentType::OFM_STREAMER:
        {
            // Read After Write Dependency for [OfmStreamer][IfmStreamer]
            if (producerAgentType == AgentType::IFM_STREAMER)
            {
                // Simple 1:1 dependency
                consumerAgentDependency.outerRatio.other = 1;
                consumerAgentDependency.outerRatio.self  = 1;

                consumerAgentDependency.innerRatio.other = 1;
                consumerAgentDependency.innerRatio.self  = 1;

                consumerAgentDependency.boundary = 0;
            }
            // Read After Write Dependency for [OfmStreamer][PleScheduler]
            else if (producerAgentType == AgentType::PLE_SCHEDULER)
            {
                // Normally this is a simple 1:1 dependency, but in some cases the PLE can have multiple stripes
                // for each OFM stripe (strategies where OfmS does the full height but PLE does partial height)

                // Outer ratio is not used (set to max)
                consumerAgentDependency.outerRatio.other = producerAgent.info.numStripesTotal;
                consumerAgentDependency.outerRatio.self  = consumerAgent.info.numStripesTotal;

                // Inner ratio based on the stripe heights
                consumerAgentDependency.innerRatio.other =
                    consumerAgent.data.ofm.fmData.dfltStripeSize.height / producerAgent.data.pleS.dfltStripeSize.height;
                consumerAgentDependency.innerRatio.self = 1;

                consumerAgentDependency.boundary = 0;
            }
            else
            {
                assert(false);
            }
            break;
        }

        default:
        {
            assert(false);
            break;
        }
    }

    // Calculate remaining agent dependencies
    if (consumerAgentDependency.relativeAgentId != 0)
    {
        ethosn::support_library::cascading_compiler::DependencyUtils::CalculateInnerRatio(consumerAgentDependency);

        ethosn::support_library::cascading_compiler::DependencyUtils::CalculateRemainingAgentDependencies(
            consumerAgentDependency);
    }
}

// Private function to fill the dependency data for Write After Read or Schedule Time dependencies
void CascadingCommandStreamGenerator::FillProducerAgentDependency(
    command_stream::cascading::Dependency& producerAgentDependency,
    const command_stream::cascading::AgentType consumerAgentType,
    const AgentIdType consumerAgentId,
    const command_stream::cascading::AgentType producerAgentType,
    const AgentIdType producerAgentId)
{
    Agent& consumerAgent = m_CommandStreamAgents[consumerAgentId];
    Agent& producerAgent = m_CommandStreamAgents[producerAgentId];

    // Add a new 'Write After Read' dependency or
    // Add a new 'Schedule Time' dependency
    switch (consumerAgentType)
    {
        case AgentType::IFM_STREAMER:
        {
            // Write After Read Dependency for [OfmStreamer][IfmStreamer] or
            // Schedule Time Dependency for [OfmStreamer][IfmStreamer]
            if (producerAgentType == AgentType::OFM_STREAMER)
            {
                // The last OFM stripe is needed by the first IFM stripe
                producerAgentDependency.outerRatio.other = consumerAgent.info.numStripesTotal;
                producerAgentDependency.outerRatio.self  = producerAgent.info.numStripesTotal;

                producerAgentDependency.innerRatio.other = 1;
                producerAgentDependency.innerRatio.self  = producerAgent.info.numStripesTotal;

                producerAgentDependency.boundary = 0;
            }
            break;
        }

        case AgentType::WGT_STREAMER:
        {
            assert(false);
            break;
        }

        case AgentType::MCE_SCHEDULER:
        {
            // Write After Read Dependency for [IfmStreamer][MceScheduler] or
            // Schedule Time Dependency for [IfmStreamer][MceScheduler]
            if (producerAgentType == AgentType::IFM_STREAMER)
            {
                if (producerAgent.data.ifm.fmData.numStripes.height > 1 &&
                    producerAgent.data.ifm.fmData.numStripes.width > 1)
                {
                    // Splitting width and height => outer ratio is for each row
                    producerAgentDependency.outerRatio.self = ethosn::utils::NumericCast<uint16_t>(
                        producerAgent.data.ifm.fmData.numStripes.width *
                        // Note we use the ifmChannels from the MceS, not the IfmS, so that this is correct for depthwise
                        // (where IfmS might have multiple IFM stripes but MceS won't)
                        consumerAgent.data.mce.numStripes.ifmChannels);
                    producerAgentDependency.outerRatio.other = ethosn::utils::NumericCast<uint16_t>(
                        consumerAgent.data.mce.numStripes.ofmWidth * consumerAgent.data.mce.numStripes.ifmChannels);
                }
                else
                {
                    // Not splitting width and height => outer ratio is not needed (set to max)
                    producerAgentDependency.outerRatio.self  = producerAgent.info.numStripesTotal;
                    producerAgentDependency.outerRatio.other = consumerAgent.info.numStripesTotal;
                }

                // The MceS can process more data than is loaded by the IfmS (e.g. two stripes at a time)
                uint16_t widthRatio  = ethosn::utils::NumericCast<uint16_t>(utils::DivRoundUp(
                    consumerAgent.data.mce.numStripes.ofmWidth, producerAgent.data.ifm.fmData.numStripes.width));
                uint16_t heightRatio = ethosn::utils::NumericCast<uint16_t>(utils::DivRoundUp(
                    consumerAgent.data.mce.numStripes.ofmHeight, producerAgent.data.ifm.fmData.numStripes.height));

                if (consumerAgent.data.mce.mceOpMode == MceOperation::DEPTHWISE_CONVOLUTION)
                {
                    assert(consumerAgent.data.mce.numStripes.ifmChannels == 1);
                }
                else
                {
                    assert(producerAgent.data.ifm.fmData.numStripes.channels ==
                           consumerAgent.data.mce.numStripes.ifmChannels);
                }

                producerAgentDependency.innerRatio.other = 1;
                producerAgentDependency.innerRatio.self =
                    ethosn::utils::NumericCast<uint16_t>(widthRatio * heightRatio);

                // MceS needs to wait for two IfmS stripes at the start of each outer ratio if neighbouring data
                // is needed. This is not applicable if all the boundary data is packed though.
                if (!(consumerAgent.data.mce.isPackedBoundaryX && consumerAgent.data.mce.isPackedBoundaryY) &&
                    ((producerAgent.data.ifm.fmData.numStripes.height > 1 &&
                      consumerAgent.data.mce.filterShape[0].height > 1) ||
                     (producerAgent.data.ifm.fmData.numStripes.width > 1 &&
                      consumerAgent.data.mce.filterShape[0].width > 1)))
                {
                    producerAgentDependency.boundary = 1;
                }
                else
                {
                    producerAgentDependency.boundary = 0;
                }
            }
            // Write After Read Dependency for [WeightStreamer][MceScheduler] or
            // Schedule Time Dependency for [WeightStreamer][MceScheduler]
            else if (producerAgentType == AgentType::WGT_STREAMER)
            {
                // MCE always traverses in IXYO order. Each MCE stripe needs a new weight stripe, unless a weight stripe
                // can be re-used which can only happen if we are not IFM splitting and we are moving in XY.

                // Outer ratio is not needed (set to max)
                producerAgentDependency.outerRatio.other = consumerAgent.info.numStripesTotal;
                producerAgentDependency.outerRatio.self  = producerAgent.info.numStripesTotal;

                if (consumerAgent.data.mce.numStripes.ifmChannels == 1)
                {
                    // Weight stripes can be re-used as we move in XY
                    producerAgentDependency.innerRatio.other = ethosn::utils::NumericCast<uint16_t>(
                        consumerAgent.data.mce.numStripes.ofmHeight * consumerAgent.data.mce.numStripes.ofmWidth);
                    producerAgentDependency.innerRatio.self = 1;
                }
                else
                {
                    // No re-use, 1:1 dependency
                    producerAgentDependency.innerRatio.other = 1;
                    producerAgentDependency.innerRatio.other = 1;
                }

                producerAgentDependency.boundary = 0;
            }
            // Schedule Time Dependency for [PleLoader][MceScheduler]
            else if (producerAgentType == AgentType::PLE_LOADER)
            {
                producerAgentDependency.outerRatio.other = ethosn::utils::NumericCast<uint16_t>(
                    consumerAgent.data.mce.numStripes.ofmHeight * consumerAgent.data.mce.numStripes.ofmWidth *
                    consumerAgent.data.mce.numStripes.ifmChannels);
                producerAgentDependency.outerRatio.self = 1;

                producerAgentDependency.innerRatio.other = ethosn::utils::NumericCast<uint16_t>(
                    consumerAgent.data.mce.numStripes.ofmHeight * consumerAgent.data.mce.numStripes.ofmWidth *
                    consumerAgent.data.mce.numStripes.ifmChannels);
                producerAgentDependency.innerRatio.self = 1;

                producerAgentDependency.boundary = 0;
            }
            // Schedule Time Dependency for [PleScheduler][MceScheduler]
            else if (producerAgentType == AgentType::PLE_SCHEDULER)
            {
                // Calculate outer ratios using number of stripes
                producerAgentDependency.outerRatio.other = ethosn::utils::NumericCast<uint16_t>(
                    consumerAgent.data.mce.numStripes.ofmHeight * consumerAgent.data.mce.numStripes.ofmWidth *
                    consumerAgent.data.mce.numStripes.ofmChannels);
                producerAgentDependency.outerRatio.self = ethosn::utils::NumericCast<uint16_t>(
                    producerAgent.data.pleS.numStripes.height * producerAgent.data.pleS.numStripes.width *
                    producerAgent.data.pleS.numStripes.channels);

                // Calculate inner ratios using ratio of stripe size
                uint16_t widthRatio   = ethosn::utils::NumericCast<uint16_t>(utils::DivRoundUp(
                    producerAgent.data.pleS.numStripes.width, consumerAgent.data.mce.numStripes.ofmWidth));
                uint16_t heightRatio  = ethosn::utils::NumericCast<uint16_t>(utils::DivRoundUp(
                    producerAgent.data.pleS.numStripes.height, consumerAgent.data.mce.numStripes.ofmHeight));
                uint16_t channelRatio = ethosn::utils::NumericCast<uint16_t>(utils::DivRoundUp(
                    producerAgent.data.pleS.numStripes.channels, consumerAgent.data.mce.numStripes.ofmChannels));

                producerAgentDependency.innerRatio.self =
                    ethosn::utils::NumericCast<uint16_t>(widthRatio * heightRatio * channelRatio);
                producerAgentDependency.innerRatio.other = 1;

                if ((producerAgent.data.pleS.numStripes.height > 1 &&
                     consumerAgent.data.mce.filterShape[0].height > 1) ||
                    (producerAgent.data.pleS.numStripes.width > 1 && consumerAgent.data.mce.filterShape[0].width > 1))
                {
                    producerAgentDependency.boundary = 1;
                }
                else
                {
                    producerAgentDependency.boundary = 0;
                }
            }
            // Schedule Time Dependency for [MceScheduler][MceScheduler]
            else if (producerAgentType == AgentType::MCE_SCHEDULER)
            {
                // We need to ensure that MCE stripes are scheduled in the same order as the PLE stripes, otherwise the firmware
                // will deadlock. This can happen in a strategy 0 cascade if an MCE stripe is scheduled but the following PLE stripe
                // is not, because it is not yet needed. An MCE and PLE stripe from the following layer can then get scheduled, and
                // this means that we missed the PLE stripe from the first layer.
                // To prevent this, we make sure that an MCE stripe is not scheduled unless the PLE stripe following it is needed,
                // so that it will be scheduled before any other PLE stripes. This is done by adding a schedule dependency on the following
                // Mce agent, so that the MCE stripe from the first layer will not be scheduled until the MCE stripe from the second layer
                // is scheduled.

                // Calculate outer ratios using number of stripes
                producerAgentDependency.outerRatio.other = ethosn::utils::NumericCast<uint16_t>(
                    consumerAgent.data.mce.numStripes.ofmHeight * consumerAgent.data.mce.numStripes.ofmWidth *
                    consumerAgent.data.mce.numStripes.ofmChannels);
                producerAgentDependency.outerRatio.self = ethosn::utils::NumericCast<uint16_t>(
                    producerAgent.data.mce.numStripes.ofmHeight * producerAgent.data.mce.numStripes.ofmWidth *
                    producerAgent.data.mce.numStripes.ofmChannels);

                // Calculate inner ratios using ratio of stripe size
                uint16_t widthRatio   = ethosn::utils::NumericCast<uint16_t>(utils::DivRoundUp(
                    producerAgent.data.mce.numStripes.ofmWidth, consumerAgent.data.mce.numStripes.ofmWidth));
                uint16_t heightRatio  = ethosn::utils::NumericCast<uint16_t>(utils::DivRoundUp(
                    producerAgent.data.mce.numStripes.ofmHeight, consumerAgent.data.mce.numStripes.ofmHeight));
                uint16_t channelRatio = ethosn::utils::NumericCast<uint16_t>(utils::DivRoundUp(
                    producerAgent.data.mce.numStripes.ofmChannels, consumerAgent.data.mce.numStripes.ofmChannels));

                producerAgentDependency.innerRatio.self =
                    ethosn::utils::NumericCast<uint16_t>(widthRatio * heightRatio * channelRatio);
                producerAgentDependency.innerRatio.other = 1;

                if ((producerAgent.data.mce.numStripes.ofmHeight > 1 &&
                     consumerAgent.data.mce.filterShape[0].height > 1) ||
                    (producerAgent.data.mce.numStripes.ofmWidth > 1 && consumerAgent.data.mce.filterShape[0].width > 1))
                {
                    producerAgentDependency.boundary = 1;
                }
                else
                {
                    producerAgentDependency.boundary = 0;
                }
            }
            else
            {
                assert(false);
            }

            break;
        }

        case AgentType::PLE_LOADER:
        {
            assert(false);
            break;
        }

        case AgentType::PLE_SCHEDULER:
        {
            // Write After Read Dependency for [IfmStreamer][PleScheduler] or
            // Schedule Time Dependency for [IfmStreamer][PleScheduler]
            if (producerAgentType == AgentType::IFM_STREAMER)
            {
                // Calculate outer ratios using number of stripes.
                producerAgentDependency.outerRatio.other = ethosn::utils::NumericCast<uint16_t>(
                    consumerAgent.data.pleS.numStripes.height * consumerAgent.data.pleS.numStripes.width *
                    consumerAgent.data.pleS.numStripes.channels);
                producerAgentDependency.outerRatio.self = ethosn::utils::NumericCast<uint16_t>(
                    producerAgent.data.ifm.fmData.numStripes.width * producerAgent.data.ifm.fmData.numStripes.height *
                    producerAgent.data.ifm.fmData.numStripes.channels);
            }
            // Schedule Time Dependency for [MceScheduler][PleScheduler]
            else if (producerAgentType == AgentType::MCE_SCHEDULER)
            {
                // Outer ratio not used (set to max)
                producerAgentDependency.outerRatio.other = consumerAgent.info.numStripesTotal;
                producerAgentDependency.outerRatio.self  = producerAgent.info.numStripesTotal;

                // Calculate inner ratios using ratio of stripe size
                uint16_t widthRatio   = ethosn::utils::NumericCast<uint16_t>(utils::DivRoundUp(
                    consumerAgent.data.pleS.numStripes.width, producerAgent.data.mce.numStripes.ofmWidth));
                uint16_t heightRatio  = ethosn::utils::NumericCast<uint16_t>(utils::DivRoundUp(
                    consumerAgent.data.pleS.numStripes.height, producerAgent.data.mce.numStripes.ofmHeight));
                uint16_t channelRatio = ethosn::utils::NumericCast<uint16_t>(utils::DivRoundUp(
                    consumerAgent.data.pleS.numStripes.channels, producerAgent.data.mce.numStripes.ofmChannels));

                producerAgentDependency.innerRatio.other = 1;
                producerAgentDependency.innerRatio.self  = ethosn::utils::NumericCast<uint16_t>(
                    widthRatio * heightRatio * channelRatio * producerAgent.data.mce.numStripes.ifmChannels);

                // Set boundary to 1 if producer stripe count is not a factor of consumer stripe count
                uint16_t numberOfIfmStripesInXYDimProducer = ethosn::utils::NumericCast<uint16_t>(
                    producerAgent.data.mce.numStripes.ofmWidth * producerAgent.data.mce.numStripes.ofmHeight *
                    producerAgent.data.mce.numStripes.ofmChannels);
                uint16_t numberOfIfmStripesInXYDimConsumer = ethosn::utils::NumericCast<uint16_t>(
                    consumerAgent.data.pleS.numStripes.width * consumerAgent.data.pleS.numStripes.height *
                    consumerAgent.data.pleS.numStripes.channels);

                uint16_t ifmStripeRemainder = ethosn::utils::NumericCast<uint16_t>(numberOfIfmStripesInXYDimConsumer %
                                                                                   numberOfIfmStripesInXYDimProducer);

                if (ifmStripeRemainder == 0)
                {
                    producerAgentDependency.boundary = 0;
                }
                else
                {
                    producerAgentDependency.boundary = 1;
                }
            }
            // Schedule Time Dependency for [PleLoader][PleScheduler]
            else if (producerAgentType == AgentType::PLE_LOADER)
            {
                producerAgentDependency.outerRatio.other = ethosn::utils::NumericCast<uint16_t>(
                    consumerAgent.data.pleS.numStripes.height * consumerAgent.data.pleS.numStripes.width *
                    consumerAgent.data.pleS.numStripes.channels);
                producerAgentDependency.outerRatio.self = 1U;
            }
            // Schedule Time Dependency for [PleScheduler][PleScheduler]
            else if (producerAgentType == AgentType::PLE_SCHEDULER)
            {
                // We need to ensure that PLE stripes are scheduled in the same order as the MCE stripes, otherwise the firmware
                // will deadlock. This can happen in a strategy 0 cascade if an MCE stripe is scheduled but the following PLE stripe
                // is not, because there is no space in the queue. An MCE and PLE stripe from the preceding layer can then get scheduled,
                // and this means that we missed the PLE stripe from the second layer.
                // To prevent this, we make sure that a PLE stripe is not scheduled until the next PLE stripe in the next layer is needed,
                // so that the order is correct. This is done by adding a schedule dependency on the following
                // PLE agent.

                // Calculate outer ratios using number of stripes
                producerAgentDependency.outerRatio.other = ethosn::utils::NumericCast<uint16_t>(
                    consumerAgent.data.pleS.numStripes.height * consumerAgent.data.pleS.numStripes.width *
                    consumerAgent.data.pleS.numStripes.channels);
                producerAgentDependency.outerRatio.self = ethosn::utils::NumericCast<uint16_t>(
                    producerAgent.data.pleS.numStripes.height * producerAgent.data.pleS.numStripes.width *
                    producerAgent.data.pleS.numStripes.channels);

                // Calculate inner ratios using ratio of stripe size
                uint16_t widthRatio   = ethosn::utils::NumericCast<uint16_t>(utils::DivRoundUp(
                    producerAgent.data.pleS.numStripes.width, consumerAgent.data.pleS.numStripes.width));
                uint16_t heightRatio  = ethosn::utils::NumericCast<uint16_t>(utils::DivRoundUp(
                    producerAgent.data.pleS.numStripes.height, consumerAgent.data.pleS.numStripes.height));
                uint16_t channelRatio = ethosn::utils::NumericCast<uint16_t>(utils::DivRoundUp(
                    producerAgent.data.pleS.numStripes.channels, consumerAgent.data.pleS.numStripes.channels));

                producerAgentDependency.innerRatio.self =
                    ethosn::utils::NumericCast<uint16_t>(widthRatio * heightRatio * channelRatio);
                producerAgentDependency.innerRatio.other = 1;

                const Agent& secondMce = m_CommandStreamAgents[consumerAgentId - 1];
                assert(secondMce.data.type == AgentType::MCE_SCHEDULER);
                if ((producerAgent.data.pleS.numStripes.height > 1 && secondMce.data.mce.filterShape[0].height > 1) ||
                    (producerAgent.data.pleS.numStripes.width > 1 && secondMce.data.mce.filterShape[0].width > 1))
                {
                    producerAgentDependency.boundary = 1;
                }
                else
                {
                    producerAgentDependency.boundary = 0;
                }
            }
            else
            {
                assert(false);
            }
            break;
        }

        case AgentType::OFM_STREAMER:
        {
            // Write After Read Dependency for [IfmStreamer][OfmStreamer] or
            // Schedule Time Dependency for [IfmStreamer][OfmStreamer]
            if (producerAgentType == AgentType::IFM_STREAMER)
            {
                // Simple 1:1 dependency
                producerAgentDependency.outerRatio.other = 1;
                producerAgentDependency.outerRatio.self  = 1;

                producerAgentDependency.innerRatio.other = 1;
                producerAgentDependency.innerRatio.self  = 1;

                producerAgentDependency.boundary = 0;
            }
            // Write After Read Dependency for [PleScheduler][OfmStreamer] or
            // Schedule Time Dependency for [PleScheduler][OfmStreamer]
            else if (producerAgentType == AgentType::PLE_SCHEDULER)
            {
                // Normally this is a simple 1:1 dependency, but in some cases the PLE can have multiple stripes
                // for each OFM stripe (strategies where OfmS does the full height but PLE does partial height)
                producerAgentDependency.outerRatio.other = consumerAgent.info.numStripesTotal;
                producerAgentDependency.outerRatio.self  = producerAgent.info.numStripesTotal;

                producerAgentDependency.innerRatio.other =
                    producerAgent.data.pleS.dfltStripeSize.height / consumerAgent.data.ofm.fmData.dfltStripeSize.height;
                producerAgentDependency.innerRatio.self = 1;

                producerAgentDependency.boundary = 0;
            }
            else
            {
                assert(false);
            }
            break;
        }

        default:
        {
            assert(false);
            break;
        }
    }

    // Calculate remaining agent dependencies
    if (producerAgentDependency.relativeAgentId != 0)
    {
        ethosn::support_library::cascading_compiler::DependencyUtils::CalculateInnerRatio(producerAgentDependency);

        ethosn::support_library::cascading_compiler::DependencyUtils::CalculateRemainingAgentDependencies(
            producerAgentDependency);
    }
}

namespace
{

/// Returns the index of the Op (in execution order) of the earliest Op
/// which could write to the given buffer.
size_t WalkGraphUp(const OpGraph& graph, Buffer* b)
{
    size_t result = std::numeric_limits<size_t>::max();

    Op* producer = graph.GetProducer(b);
    assert(producer != nullptr);

    for (Buffer* input : graph.GetInputs(producer))
    {
        if (input->m_Location != Location::Dram)
        {
            result = std::min(result, WalkGraphUp(graph, input));
        }
    }

    if (result == std::numeric_limits<size_t>::max())
    {
        // We didn't find any Ops with all inputs in DRAM further up the graph, so this one is the earliest
        result = utils::FindIndex(graph.GetOps(), producer).second;
    }
    return result;
}

/// Returns the index of the Op (in execution order) of the latest Op
/// which could read from the given buffer.
size_t WalkGraphDown(const OpGraph& graph, Buffer* b)
{
    size_t result = 0;
    for (std::pair<Op*, uint32_t> consumer : graph.GetConsumers(b))
    {
        Buffer* output = graph.GetOutput(consumer.first);
        assert(output != nullptr);

        size_t i;
        if (output->m_Location == Location::Dram)
        {
            i = utils::FindIndex(graph.GetOps(), consumer.first).second;
        }
        else
        {
            i = WalkGraphDown(graph, output);
        }
        result = std::max(result, i);
    }

    return result;
}

}    // namespace

// Private function to add the lifetime information of the intermediate DRAM buffers
/// Determines the start and end of the lifetime of the given (intermediate DRAM) buffer.
/// The approach is to walk the graph backwards from the buffer to find the previous time
/// there was a DRAM buffer, at which point we know the target buffer would not be needed,
/// and we do the same walking forwards, to know the point at which the target buffer
/// will be finished with. When there are branches, we go along each to find the
/// furthest away usage.
/// This is somewhat conservative because in a strategy 1 or 3 cascade, we could
/// shorten the lifetime, but we don't account for that here to keep it simple.
void CascadingCommandStreamGenerator::AddLifetimeInfoForIntermediateDramBuffers()
{
    for (Buffer* buffer : m_MergedOpGraph.GetBuffers())
    {
        if (buffer->m_Location == Location::Dram)
        {
            assert(buffer->m_BufferType.has_value());
            if (buffer->m_BufferType.value() == BufferType::Intermediate)
            {
                AgentIdType lifetimeStart = WalkGraphUp(m_MergedOpGraph, buffer);
                AgentIdType lifetimeEnd   = WalkGraphDown(m_MergedOpGraph, buffer);
                m_BufferManager.MarkBufferUsedAtTime(m_DramBufToBufIdMapping.at(buffer),
                                                     static_cast<uint32_t>(lifetimeStart),
                                                     static_cast<uint32_t>(lifetimeEnd + 1));
            }
        }
    }
}
}    // namespace cascading_compiler
}    // namespace support_library
}    // namespace ethosn
