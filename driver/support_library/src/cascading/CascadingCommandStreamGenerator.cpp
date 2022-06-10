//
// Copyright © 2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "CascadingCommandStreamGenerator.hpp"
#include "CascadingCommandStreamGeneratorUtils.hpp"
#include "Compiler.hpp"

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

uint16_t CascadingCommandStreamGenerator::AddDramBufferAndCacheId(Buffer* inputBuffer, Op* const op)
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
            inputBufferId = ethosn::utils::NumericCast<uint16_t>(
                m_BufferManager.AddDramInput(inputBuffer->m_SizeInBytes, *op->m_OperationIds.begin()));
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

            DmaOp* const dmaOp = static_cast<DmaOp* const>(ptrDmaOp);

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

                // Add 'Write After Read' dependency information to the OfmStreamer agent
                // Write After Read Dependency for [OfmStreamer][IfmStreamer]
                AddWriteAfterReadDependency(AgentType::IFM_STREAMER, ifmStreamerAgentId, AgentType::OFM_STREAMER,
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

            Op* ifmDma = m_MergedOpGraph.GetProducer(mceInput);
            if (IsObjectOfType<DmaOp>(ifmDma))
            {
                Buffer* ifmDmaInput = m_MergedOpGraph.GetInputs(ifmDma)[0];
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
            //TODO make this generic
            m_BufferManager.ChangeToOutput(outputBufferId, 3, 0);
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
                                m_OpToAgentIdMapping[m_MergedOpGraph.GetProducer(inputBuffers[g_MceIfmBufferIndex])]);
    // Read After Write Dependency for [MceScheduler][WeightStreamer]
    AddReadAfterWriteDependency(
        AgentType::MCE_SCHEDULER, mceSchedulerAgentId, AgentType::WGT_STREAMER,
        m_OpToAgentIdMapping[m_MergedOpGraph.GetProducer(inputBuffers[g_MceWeightBufferIndex])]);

    // Add 'Write After Read' dependency information to the IfmStreamer and WeightStreamer agents
    // Write After Read Dependency for [IfmStreamer][MceScheduler] or
    // Write After Read Dependency for [PleScheduler][MceScheduler]
    AddWriteAfterReadDependency(AgentType::MCE_SCHEDULER, mceSchedulerAgentId, producerAgentType,
                                m_OpToAgentIdMapping[m_MergedOpGraph.GetProducer(inputBuffers[g_MceIfmBufferIndex])]);
    // Write After Read Dependency for [WeightStreamer][MceScheduler]
    AddWriteAfterReadDependency(
        AgentType::MCE_SCHEDULER, mceSchedulerAgentId, AgentType::WGT_STREAMER,
        m_OpToAgentIdMapping[m_MergedOpGraph.GetProducer(inputBuffers[g_MceWeightBufferIndex])]);

    // Add 'Schedule Time' dependency information to the IfmStreamer and WeightStreamer agents
    // Schedule Time Dependency for [IfmStreamer][MceScheduler] or
    // Schedule Time Dependency for [PleScheduler][MceScheduler]
    AddScheduleTimeDependency(AgentType::MCE_SCHEDULER, mceSchedulerAgentId, producerAgentType,
                              m_OpToAgentIdMapping[m_MergedOpGraph.GetProducer(inputBuffers[g_MceIfmBufferIndex])]);
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
        AddReadAfterWriteDependency(
            AgentType::PLE_SCHEDULER, pleSchedulerAgentId, AgentType::IFM_STREAMER,
            m_OpToAgentIdMapping[m_MergedOpGraph.GetProducer(inputBuffers[g_PleInputBuffer0Index])]);

        if (loadKernel)
        {
            // Read After Write Dependency for [PleScheduler][PleLoader]
            AddReadAfterWriteDependency(
                AgentType::PLE_SCHEDULER, pleSchedulerAgentId, AgentType::PLE_LOADER,
                m_PleKernelToPleLoaderAgentIdMapping[static_cast<PleOp*>(ptrPleOp)->m_PleKernelId]);
        }

        // Write After Read Dependency for [IfmStreamer][PleScheduler]
        AddWriteAfterReadDependency(
            AgentType::PLE_SCHEDULER, pleSchedulerAgentId, AgentType::IFM_STREAMER,
            m_OpToAgentIdMapping[m_MergedOpGraph.GetProducer(inputBuffers[g_PleInputBuffer0Index])]);

        // Schedule Time Dependency for [IfmStreamer][PleScheduler]
        AddScheduleTimeDependency(
            AgentType::PLE_SCHEDULER, pleSchedulerAgentId, AgentType::IFM_STREAMER,
            m_OpToAgentIdMapping[m_MergedOpGraph.GetProducer(inputBuffers[g_PleInputBuffer0Index])]);

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
        AddReadAfterWriteDependency(
            AgentType::PLE_SCHEDULER, pleSchedulerAgentId, AgentType::MCE_SCHEDULER,
            m_OpToAgentIdMapping[m_MergedOpGraph.GetProducer(inputBuffers[g_PleInputBuffer0Index])]);
        if (loadKernel)
        {
            // Read After Write Dependency for [PleScheduler][PleLoader]
            AddReadAfterWriteDependency(
                AgentType::PLE_SCHEDULER, pleSchedulerAgentId, AgentType::PLE_LOADER,
                m_PleKernelToPleLoaderAgentIdMapping[static_cast<PleOp*>(ptrPleOp)->m_PleKernelId]);
        }

        // Schedule Time Dependency for [MceScheduler][PleScheduler]
        AddScheduleTimeDependency(
            AgentType::PLE_SCHEDULER, pleSchedulerAgentId, AgentType::MCE_SCHEDULER,
            m_OpToAgentIdMapping[m_MergedOpGraph.GetProducer(inputBuffers[g_PleInputBuffer0Index])]);
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

        sramBufferSlotSize = CommonUtils::CalculateBufferSize(sramBufferShape, CascadingBufferFormat::NHWCB) /
                             m_Capabilities.GetNumberOfSrams();

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

            ConcatOp* const concatOp = static_cast<ConcatOp* const>(ptrConcatOp);

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

                uint32_t heightOffset =
                    CommonUtils::CalculateBufferSize(inputBuffer->m_TensorShape, inputBuffer->m_Format);
                dramBufferOffset = dramBufferOffset + heightOffset;
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
                    widthOffset =
                        numberOfBrickGroups * CommonUtils::CalculateBufferSize(m_Capabilities.GetBrickGroupShape(),
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
                    CommonUtils::CalculateBufferSize(m_Capabilities.GetBrickGroupShape(), outputBuffer->m_Format);
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
    StreamersUtils::SetStripeChannelsInfo(m_Capabilities, ifmStreamerData.fmData, inputSramBuffer->m_TensorShape,
                                          inputSramBuffer->m_StripeShape);

    StreamersUtils::SetSuperTensorSizeInCells(ifmStreamerData.fmData, inputDramBuffer->m_TensorShape, transferFormat);

    StreamersUtils::SetStripeIdStrides(ifmStreamerData.fmData, inputDramBuffer->m_Order);

    AgentDependencyInfo dependencyInfo = {};
    dependencyInfo.numStripesTotal     = ethosn::utils::NumericCast<uint16_t>(
        utils::GetNumStripesTotal(inputSramBuffer->m_TensorShape, inputSramBuffer->m_StripeShape));

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
    weightStreamerData.stripeIdStrides.ofmChannels = weightStreamerData.numStripes.ifmChannels;

    AgentDependencyInfo dependencyInfo = {};

    dependencyInfo.numStripesTotal = ethosn::utils::NumericCast<uint16_t>(
        utils::GetNumStripesTotal(weightsSramBuffer->m_TensorShape, weightsSramBuffer->m_StripeShape));
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

    MceSUtils::setMcesOpMode(mceSchedulerData, ptrMceOp->m_Op);
    MceSUtils::setMcesAlgorithm(mceSchedulerData, ptrMceOp->m_Algo);

    if (ptrMceOp->m_Stride.m_X == 1 && ptrMceOp->m_Stride.m_Y == 1)
    {
        for (int i = 0; i < 4; i++)
        {
            mceSchedulerData.filterShape[i].height =
                ethosn::utils::NumericCast<uint8_t>(weightBuffer->m_TensorShape[0]);
            mceSchedulerData.filterShape[i].width = ethosn::utils::NumericCast<uint8_t>(weightBuffer->m_TensorShape[1]);
        }

        mceSchedulerData.padding[0].left = ethosn::utils::NumericCast<uint8_t>(ptrMceOp->m_PadLeft);
        mceSchedulerData.padding[0].top  = ethosn::utils::NumericCast<uint8_t>(ptrMceOp->m_PadTop);

        mceSchedulerData.ifmDeltaDefault[0].height =
            static_cast<int8_t>(inputBuffer->m_TensorShape[1] - outputBuffer->m_TensorShape[1]);
        mceSchedulerData.ifmDeltaDefault[0].width =
            static_cast<int8_t>(inputBuffer->m_TensorShape[2] - outputBuffer->m_TensorShape[2]);

        mceSchedulerData.ifmDeltaEdge[0].height =
            static_cast<int8_t>(inputBuffer->m_TensorShape[1] - outputBuffer->m_TensorShape[1]);
        mceSchedulerData.ifmDeltaEdge[0].width =
            static_cast<int8_t>(inputBuffer->m_TensorShape[2] - outputBuffer->m_TensorShape[2]);
    }
    else if (ptrMceOp->m_Stride.m_X == 2 && ptrMceOp->m_Stride.m_Y == 2)
    {
        MceSUtils::setMcesStridedConvolutionData(mceSchedulerData, m_MergedOpGraph, ptrMceOp);
    }
    else
    {
        assert(false);
    }

    mceSchedulerData.ifmStripeShapeDefault.height = ethosn::utils::NumericCast<uint16_t>(inputBuffer->m_StripeShape[1]);
    mceSchedulerData.ifmStripeShapeDefault.width  = ethosn::utils::NumericCast<uint16_t>(inputBuffer->m_StripeShape[2]);

    mceSchedulerData.reluActiv.min = ptrMceOp->m_LowerBound;
    mceSchedulerData.reluActiv.max = ptrMceOp->m_UpperBound;
    mceSchedulerData.pleKernelId   = pleKernelId;

    AgentDependencyInfo dependencyInfo = {};
    dependencyInfo.numStripesTotal     = ethosn::utils::NumericCast<uint16_t>(
        utils::GetNumStripesTotal(outputBuffer->m_TensorShape, outputBuffer->m_StripeShape));

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

    if (ptrPleOp->m_Lifetime == Lifetime::Atomic)
    {
        assert(ptrPleOp->m_OutputStripeShape == outputBuffer->m_StripeShape);
    }

    PleS pleS = {};

    CommonUtils::SetTileInfoForBuffer(m_Capabilities, pleS.ofmTile, outputBuffer);

    pleS.ofmZeroPoint = ethosn::utils::NumericCast<int16_t>(outputBuffer->m_QuantizationInfo.GetZeroPoint());

    PleSUtils::SetPlesHeightStripeInfo(pleS, outputBuffer->m_TensorShape, ptrPleOp->m_OutputStripeShape);
    PleSUtils::SetPlesWidthStripeInfo(pleS, outputBuffer->m_TensorShape, ptrPleOp->m_OutputStripeShape);
    PleSUtils::SetPlesChannelsStripeInfo(pleS, outputBuffer->m_TensorShape, ptrPleOp->m_OutputStripeShape);

    PleSUtils::SetStripeIdStrides(pleS, outputBuffer);

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
        utils::GetNumStripesTotal(outputBuffer->m_TensorShape, outputBuffer->m_StripeShape));

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
    StreamersUtils::SetStripeChannelsInfo(m_Capabilities, ofmStreamerData.fmData, outputSramBuffer->m_TensorShape,
                                          outputSramBuffer->m_StripeShape);

    StreamersUtils::SetSuperTensorSizeInCells(ofmStreamerData.fmData, outputDramBuffer->m_TensorShape,
                                              outputDramBuffer->m_Format);

    StreamersUtils::SetStripeIdStrides(ofmStreamerData.fmData, outputDramBuffer->m_Order);

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

    if (producerAgentType == AgentType::WGT_STREAMER || producerAgentType == AgentType::MCE_SCHEDULER ||
        (producerAgentType == AgentType::IFM_STREAMER && consumerAgentType == AgentType::PLE_SCHEDULER))
    {
        Dependency& consumerAgentReadDependency1Ref =
            m_CommandStreamAgents[consumerAgentId].info.readDependencies.at(1);
        consumerAgentReadDependency1Ref.relativeAgentId = static_cast<RelativeAgentIdType>(relativeAgentId);
        FillConsumerAgentDependency(consumerAgentReadDependency1Ref, consumerAgentType, consumerAgentId,
                                    producerAgentType, producerAgentId);
    }
    else
    {
        Dependency& consumerAgentReadDependency0Ref =
            m_CommandStreamAgents[consumerAgentId].info.readDependencies.at(0);
        consumerAgentReadDependency0Ref.relativeAgentId = static_cast<RelativeAgentIdType>(relativeAgentId);
        FillConsumerAgentDependency(consumerAgentReadDependency0Ref, consumerAgentType, consumerAgentId,
                                    producerAgentType, producerAgentId);
    }
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
    assert((producerAgentType != AgentType::MCE_SCHEDULER));

    if ((producerAgentType != AgentType::WGT_STREAMER))
    {
        Dependency& consumerAgentReadDependency0Ref =
            m_CommandStreamAgents[consumerAgentId].info.readDependencies.at(0);
        consumerAgentReadDependency0Ref.relativeAgentId = static_cast<RelativeAgentIdType>(relativeAgentId);
        FillConsumerAgentDependency(consumerAgentReadDependency0Ref, consumerAgentType, consumerAgentId,
                                    producerAgentType, producerAgentId);
    }
    else
    {
        Dependency& consumerAgentReadDependency1Ref =
            m_CommandStreamAgents[consumerAgentId].info.readDependencies.at(1);
        consumerAgentReadDependency1Ref.relativeAgentId = static_cast<RelativeAgentIdType>(relativeAgentId);
        FillConsumerAgentDependency(consumerAgentReadDependency1Ref, consumerAgentType, consumerAgentId,
                                    producerAgentType, producerAgentId);
    }
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

    Dependency& producerAgentWriteDependencyRef = m_CommandStreamAgents[producerAgentId].info.writeDependencies.at(0);
    producerAgentWriteDependencyRef.relativeAgentId = static_cast<RelativeAgentIdType>(relativeAgentId);
    FillProducerAgentDependency(producerAgentWriteDependencyRef, consumerAgentType, consumerAgentId, producerAgentType,
                                producerAgentId);
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

    Dependency& producerAgentScheduleDependencyRef =
        m_CommandStreamAgents[producerAgentId].info.scheduleDependencies.at(0);

    // Only the first consumer needs to update the relative agent id of the schedule dependency
    if (producerAgentScheduleDependencyRef.relativeAgentId == 0)
    {
        producerAgentScheduleDependencyRef.relativeAgentId = static_cast<RelativeAgentIdType>(relativeAgentId);
        FillProducerAgentDependency(producerAgentScheduleDependencyRef, consumerAgentType, consumerAgentId,
                                    producerAgentType, producerAgentId);
    }
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
                consumerAgentDependency.outerRatio.other = ethosn::utils::NumericCast<uint16_t>(
                    producerAgent.data.ofm.fmData.numStripes.height * producerAgent.data.ofm.fmData.numStripes.width *
                    producerAgent.data.ofm.fmData.numStripes.channels);
                consumerAgentDependency.outerRatio.self = ethosn::utils::NumericCast<uint16_t>(
                    consumerAgent.data.ifm.fmData.numStripes.height * consumerAgent.data.ifm.fmData.numStripes.width *
                    consumerAgent.data.ifm.fmData.numStripes.channels);
            }
            break;
        }

        case AgentType::WGT_STREAMER:
        {
            // Sram Overlap Dependency for [WeightStreamer][OfmStreamer]
            if (producerAgentType == AgentType::OFM_STREAMER)
            {
                consumerAgentDependency.outerRatio.other = ethosn::utils::NumericCast<uint16_t>(
                    producerAgent.data.ofm.fmData.numStripes.height * producerAgent.data.ofm.fmData.numStripes.width *
                    producerAgent.data.ofm.fmData.numStripes.channels);
                consumerAgentDependency.outerRatio.self = ethosn::utils::NumericCast<uint16_t>(
                    consumerAgent.data.wgt.numStripes.ifmChannels * consumerAgent.data.wgt.numStripes.ofmChannels);
            }
            break;
        }

        case AgentType::MCE_SCHEDULER:
        {
            // Read After Write Dependency for [MceScheduler][IfmStreamer]
            if (producerAgentType == AgentType::IFM_STREAMER)
            {
                consumerAgentDependency.outerRatio.other = ethosn::utils::NumericCast<uint16_t>(
                    producerAgent.data.ifm.fmData.numStripes.width * producerAgent.data.ifm.fmData.numStripes.height *
                    producerAgent.data.ifm.fmData.numStripes.channels);
                consumerAgentDependency.outerRatio.self = ethosn::utils::NumericCast<uint16_t>(
                    consumerAgent.data.mce.numStripes.ofmHeight * consumerAgent.data.mce.numStripes.ofmWidth *
                    consumerAgent.data.mce.numStripes.ifmChannels);

                assert(consumerAgent.data.mce.numStripes.ifmChannels ==
                       producerAgent.data.ifm.fmData.numStripes.channels);
                ;
            }
            // Read After Write Dependency for [MceScheduler][WeightStreamer]
            else if (producerAgentType == AgentType::WGT_STREAMER)
            {
                consumerAgentDependency.outerRatio.other = ethosn::utils::NumericCast<uint16_t>(
                    producerAgent.data.wgt.numStripes.ifmChannels * producerAgent.data.wgt.numStripes.ofmChannels);
                consumerAgentDependency.outerRatio.self = ethosn::utils::NumericCast<uint16_t>(
                    consumerAgent.data.mce.numStripes.ofmHeight * consumerAgent.data.mce.numStripes.ofmWidth);
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
                // Calculate outer ratios using number of stripes
                consumerAgentDependency.outerRatio.other = ethosn::utils::NumericCast<uint16_t>(
                    producerAgent.data.mce.numStripes.ofmWidth * producerAgent.data.mce.numStripes.ofmHeight *
                    producerAgent.data.mce.numStripes.ofmChannels);
                consumerAgentDependency.outerRatio.self = ethosn::utils::NumericCast<uint16_t>(
                    consumerAgent.data.pleS.numStripes.height * consumerAgent.data.pleS.numStripes.width *
                    consumerAgent.data.pleS.numStripes.channels);
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
                consumerAgentDependency.outerRatio.other = ethosn::utils::NumericCast<uint16_t>(
                    producerAgent.data.ifm.fmData.numStripes.height * producerAgent.data.ifm.fmData.numStripes.width *
                    producerAgent.data.ifm.fmData.numStripes.channels);
                consumerAgentDependency.outerRatio.self = ethosn::utils::NumericCast<uint16_t>(
                    consumerAgent.data.ofm.fmData.numStripes.height * consumerAgent.data.ofm.fmData.numStripes.width *
                    consumerAgent.data.ofm.fmData.numStripes.channels);
            }
            // Read After Write Dependency for [OfmStreamer][PleScheduler]
            else if (producerAgentType == AgentType::PLE_SCHEDULER)
            {
                consumerAgentDependency.outerRatio.other = 1;
                consumerAgentDependency.outerRatio.self  = 1;
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
    ethosn::support_library::cascading_compiler::DependencyUtils::CalculateInnerRatio(consumerAgentDependency);

    ethosn::support_library::cascading_compiler::DependencyUtils::CalculateRemainingAgentDependencies(
        consumerAgentDependency);
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
                producerAgentDependency.outerRatio.other = ethosn::utils::NumericCast<uint16_t>(
                    consumerAgent.data.ifm.fmData.numStripes.height * consumerAgent.data.ifm.fmData.numStripes.width *
                    consumerAgent.data.ifm.fmData.numStripes.channels);
                producerAgentDependency.outerRatio.self = ethosn::utils::NumericCast<uint16_t>(
                    producerAgent.data.ofm.fmData.numStripes.height * producerAgent.data.ofm.fmData.numStripes.width *
                    producerAgent.data.ofm.fmData.numStripes.channels);
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
                producerAgentDependency.outerRatio.other = ethosn::utils::NumericCast<uint16_t>(
                    consumerAgent.data.mce.numStripes.ofmHeight * consumerAgent.data.mce.numStripes.ofmWidth *
                    consumerAgent.data.mce.numStripes.ifmChannels);
                producerAgentDependency.outerRatio.self = ethosn::utils::NumericCast<uint16_t>(
                    producerAgent.data.ifm.fmData.numStripes.width * producerAgent.data.ifm.fmData.numStripes.height *
                    producerAgent.data.ifm.fmData.numStripes.channels);

                assert(producerAgent.data.ifm.fmData.numStripes.channels ==
                       consumerAgent.data.mce.numStripes.ifmChannels);
            }
            // Write After Read Dependency for [WeightStreamer][MceScheduler] or
            // Schedule Time Dependency for [WeightStreamer][MceScheduler]
            else if (producerAgentType == AgentType::WGT_STREAMER)
            {
                producerAgentDependency.outerRatio.other = ethosn::utils::NumericCast<uint16_t>(
                    consumerAgent.data.mce.numStripes.ofmHeight * consumerAgent.data.mce.numStripes.ofmWidth *
                    consumerAgent.data.mce.numStripes.ofmChannels);
                producerAgentDependency.outerRatio.self = ethosn::utils::NumericCast<uint16_t>(
                    producerAgent.data.wgt.numStripes.ifmChannels * producerAgent.data.wgt.numStripes.ofmChannels);
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
                // Calculate outer ratios using number of stripes
                producerAgentDependency.outerRatio.other = ethosn::utils::NumericCast<uint16_t>(
                    consumerAgent.data.pleS.numStripes.height * consumerAgent.data.pleS.numStripes.width *
                    consumerAgent.data.pleS.numStripes.channels);
                producerAgentDependency.outerRatio.self = ethosn::utils::NumericCast<uint16_t>(
                    producerAgent.data.mce.numStripes.ofmHeight * producerAgent.data.mce.numStripes.ofmWidth *
                    producerAgent.data.mce.numStripes.ofmChannels);
            }
            // Schedule Time Dependency for [PleLoader][PleScheduler]
            else if (producerAgentType == AgentType::PLE_LOADER)
            {
                producerAgentDependency.outerRatio.other = ethosn::utils::NumericCast<uint16_t>(
                    consumerAgent.data.pleS.numStripes.height * consumerAgent.data.pleS.numStripes.width *
                    consumerAgent.data.pleS.numStripes.channels);
                producerAgentDependency.outerRatio.self = 1U;
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
                producerAgentDependency.outerRatio.other = ethosn::utils::NumericCast<uint16_t>(
                    consumerAgent.data.ofm.fmData.numStripes.height * consumerAgent.data.ofm.fmData.numStripes.width *
                    consumerAgent.data.ofm.fmData.numStripes.channels);
                producerAgentDependency.outerRatio.self = ethosn::utils::NumericCast<uint16_t>(
                    producerAgent.data.ofm.fmData.numStripes.height * producerAgent.data.ofm.fmData.numStripes.width *
                    producerAgent.data.ofm.fmData.numStripes.channels);
            }
            // Write After Read Dependency for [PleScheduler][OfmStreamer] or
            // Schedule Time Dependency for [PleScheduler][OfmStreamer]
            else if (producerAgentType == AgentType::PLE_SCHEDULER)
            {
                producerAgentDependency.outerRatio.other = 1;
                producerAgentDependency.outerRatio.self  = 1;
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
    ethosn::support_library::cascading_compiler::DependencyUtils::CalculateInnerRatio(producerAgentDependency);

    ethosn::support_library::cascading_compiler::DependencyUtils::CalculateRemainingAgentDependencies(
        producerAgentDependency);
}

// Private function to add the lifetime information of the intermediate DRAM buffers
void CascadingCommandStreamGenerator::AddLifetimeInfoForIntermediateDramBuffers()
{
    // Lifetime start of the buffer holds the producer agent Id
    AgentIdType lifetimeStart;
    // Lifetime end of the buffer holds the last consumer agent Id
    AgentIdType lifetimeEnd;

    // Add the lifetime information for each intermediate DRAM buffer
    for (Buffer* buffer : m_MergedOpGraph.GetBuffers())
    {
        if (buffer->m_Location == Location::Dram)
        {
            assert(buffer->m_BufferType.has_value());

            // Check that the buffer type is intermediate
            if (buffer->m_BufferType.value() == BufferType::Intermediate)
            {
                // Set the Lifetime start and end of the DRAM buffer
                Op* producer = m_MergedOpGraph.GetProducer(buffer);
                assert(producer != nullptr);

                lifetimeStart = m_OpToAgentIdMapping.at(producer);
                lifetimeEnd   = 0;

                OpGraph::ConsumersList consumers = m_MergedOpGraph.GetConsumers(buffer);
                assert(consumers.size() >= 1);
                for (auto consumer : consumers)
                {
                    AgentIdType consumerAgentId = m_OpToAgentIdMapping.at(consumer.first);
                    if (consumerAgentId > lifetimeEnd)
                    {
                        lifetimeEnd = consumerAgentId;
                    }
                }

                // Add lifetime information of the corresponding buffer to the buffer manager
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