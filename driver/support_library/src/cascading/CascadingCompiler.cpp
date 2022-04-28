//
// Copyright © 2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "../Compiler.hpp"
#include "CascadingCompiler.hpp"
#include "CascadingCompilerUtils.hpp"

#include <memory>

namespace ethosn
{
namespace support_library
{
namespace cascading_compiler
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
    m_CommandStream.EmplaceBack(command_stream::Cascade{ static_cast<uint32_t>(m_CommandStreamAgents.size()) });
    for (auto& agent : m_CommandStreamAgents)
    {
        m_CommandStream.EmplaceBack<Agent>(agent);
    }
    m_BufferManager.AddCommandStream(m_CommandStream);

    // Create the compiled network using the updated BufferManager instance
    std::unique_ptr<CompiledNetworkImpl> compiledNetwork = std::make_unique<CompiledNetworkImpl>(
        m_BufferManager.GetConstantDmaData(), m_BufferManager.GetConstantControlUnitData(),
        m_BufferManager.GetBuffers(), m_OperationIds);

    return compiledNetwork;
}

std::vector<Agent> CascadingCompiler::GetCommandStreamOfAgents()
{
    return m_CommandStreamAgents;
}

void CascadingCompiler::ProcessDmaOp(Op* const ptrDmaOp)
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
            // Ifm Streamer Agent
            uint16_t inputBufferId = static_cast<uint16_t>(
                m_BufferManager.AddDram(inputBuffer->m_BufferType.value(), inputBuffer->m_SizeInBytes));

            // If this is an Intermediate Dram Buffer, add it to the IntermdiateDramBufToBufId map with the appropriate Id
            if (inputBuffer->m_BufferType.value() == BufferType::Intermediate)
            {
                m_IntermdiateDramBufToBufIdMapping[inputBuffer] = inputBufferId;
            }

            AgentIdType ifmStreamerAgentId =
                AddIfmStreamerToCommandStream(ptrDmaOp, inputBufferId, inputBuffer, outputBuffer);

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

            AgentIdType ofmStreamerAgentId = m_OpToAgentIdMapping[m_MergedOpGraph.GetProducer(
                m_MergedOpGraph.GetInputs(weightBufferConsumer.first)[g_MceIfmBufferIndex])];

            // Add 'Sram Overlap' dependency to the WeightStreamer agent
            // Sram Overlap Dependency for [WeightStreamer][OfmStreamer]
            AddSramOverlapDependency(AgentType::WGT_STREAMER, weightStreamerAgentId, AgentType::OFM_STREAMER,
                                     ofmStreamerAgentId);
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

        // If this is an Intermediate Dram Buffer, add it to the IntermdiateDramBufToBufId map with the appropriate Id
        if (outputBuffer->m_BufferType.value() == BufferType::Intermediate)
        {
            m_IntermdiateDramBufToBufIdMapping[outputBuffer] = outputBufferId;
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

void CascadingCompiler::ProcessMceOp(Op* const ptrMceOp)
{
    // Get the input buffers to the Mce Op
    OpGraph::BufferList inputBuffers = m_MergedOpGraph.GetInputs(ptrMceOp);
    assert(inputBuffers.size() == 2 && inputBuffers[g_MceIfmBufferIndex]->m_Offset.has_value() &&
           inputBuffers[g_MceWeightBufferIndex]->m_Offset.has_value());

    // Get the output buffer from the Mce Op
    Buffer* outputBuffer = m_MergedOpGraph.GetOutput(ptrMceOp);
    assert(outputBuffer != nullptr);

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
    // Read After Write Dependency for [MceScheduler][IfmStreamer]
    AddReadAfterWriteDependency(AgentType::MCE_SCHEDULER, mceSchedulerAgentId, AgentType::IFM_STREAMER,
                                m_OpToAgentIdMapping[m_MergedOpGraph.GetProducer(inputBuffers[g_MceIfmBufferIndex])]);
    // Read After Write Dependency for [MceScheduler][WeightStreamer]
    AddReadAfterWriteDependency(
        AgentType::MCE_SCHEDULER, mceSchedulerAgentId, AgentType::WGT_STREAMER,
        m_OpToAgentIdMapping[m_MergedOpGraph.GetProducer(inputBuffers[g_MceWeightBufferIndex])]);

    // Add 'Write After Read' dependency information to the IfmStreamer and WeightStreamer agents
    // Write After Read Dependency for [IfmStreamer][MceScheduler]
    AddWriteAfterReadDependency(AgentType::MCE_SCHEDULER, mceSchedulerAgentId, AgentType::IFM_STREAMER,
                                m_OpToAgentIdMapping[m_MergedOpGraph.GetProducer(inputBuffers[g_MceIfmBufferIndex])]);
    // Write After Read Dependency for [WeightStreamer][MceScheduler]
    AddWriteAfterReadDependency(
        AgentType::MCE_SCHEDULER, mceSchedulerAgentId, AgentType::WGT_STREAMER,
        m_OpToAgentIdMapping[m_MergedOpGraph.GetProducer(inputBuffers[g_MceWeightBufferIndex])]);

    // Add 'Schedule Time' dependency information to the IfmStreamer and WeightStreamer agents
    // Schedule Time Dependency for [IfmStreamer][MceScheduler]
    AddScheduleTimeDependency(AgentType::MCE_SCHEDULER, mceSchedulerAgentId, AgentType::IFM_STREAMER,
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

void CascadingCompiler::ProcessPleOp(Op* const ptrPleOp)
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

        // Write After Read Dependency for [MceScheduler][PleScheduler]
        AddWriteAfterReadDependency(
            AgentType::PLE_SCHEDULER, pleSchedulerAgentId, AgentType::MCE_SCHEDULER,
            m_OpToAgentIdMapping[m_MergedOpGraph.GetProducer(inputBuffers[g_PleInputBuffer0Index])]);

        // Schedule Time Dependency for [MceScheduler][PleScheduler]
        AddScheduleTimeDependency(
            AgentType::PLE_SCHEDULER, pleSchedulerAgentId, AgentType::MCE_SCHEDULER,
            m_OpToAgentIdMapping[m_MergedOpGraph.GetProducer(inputBuffers[g_PleInputBuffer0Index])]);
    }
    ETHOSN_UNUSED(outputBuffer);
}

void CascadingCompiler::ProcessConcatOp(Op* const ptrConcatOp)
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
        m_IntermdiateDramBufToBufIdMapping[outputBuffer] = outputBufferId;
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

        uint16_t inputBufferId = static_cast<uint16_t>(
            m_BufferManager.AddDram(inputBuffer->m_BufferType.value(), inputBuffer->m_SizeInBytes));

        // If this is an Intermediate Dram Buffer, add it to the IntermdiateDramBufToBufId map with the appropriate Id
        if (inputBuffer->m_BufferType.value() == BufferType::Intermediate)
        {
            m_IntermdiateDramBufToBufIdMapping[inputBuffer] = inputBufferId;
        }

        // Ifm Streamer Agent
        AgentIdType ifmStreamerAgentId =
            AddIfmStreamerToCommandStream(ptrConcatOp, inputBufferId, inputBuffer, &sramBuffer);

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

            uint32_t heightOffset = CommonUtils::CalculateBufferSize(inputBuffer->m_TensorShape, inputBuffer->m_Format);
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

                uint32_t widthInBrickGroups = utils::DivRoundUp(utils::GetWidth(inputBuffer->m_TensorShape),
                                                                utils::GetWidth(m_Capabilities.GetBrickGroupShape()));
                uint32_t channelsInBrickGroups =
                    utils::DivRoundUp(utils::GetChannels(inputBuffer->m_TensorShape),
                                      utils::GetChannels(m_Capabilities.GetBrickGroupShape()));
                uint32_t numberOfBrickGroups = channelsInBrickGroups * widthInBrickGroups;
                widthOffset =
                    numberOfBrickGroups *
                    CommonUtils::CalculateBufferSize(m_Capabilities.GetBrickGroupShape(), CascadingBufferFormat::NHWCB);
            }

            dramBufferOffset = dramBufferOffset + widthOffset;
        }
        // Concatenation is happening in the Depth/Channels dimension
        else if (std::get<2>(isHWCSplit))
        {
            assert(outputBuffer->m_Format == CascadingBufferFormat::NHWCB);
            uint32_t channelsInBrickGroups = utils::DivRoundUp(utils::GetChannels(inputBuffer->m_TensorShape),
                                                               utils::GetChannels(m_Capabilities.GetBrickGroupShape()));
            uint32_t depthOffset =
                channelsInBrickGroups *
                CommonUtils::CalculateBufferSize(m_Capabilities.GetBrickGroupShape(), outputBuffer->m_Format);
            dramBufferOffset = dramBufferOffset + depthOffset;
        }
    }
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

// Private function to add IFM_STREAMER to the command stream
AgentIdType CascadingCompiler::AddIfmStreamerToCommandStream(Op* const ptrOp,
                                                             const uint16_t inputDramBufferId,
                                                             const Buffer* const inputDramBuffer,
                                                             const Buffer* const inputSramBuffer)
{
    assert(IsObjectOfType<DmaOp>(ptrOp) || IsObjectOfType<ConcatOp>(ptrOp));

    IfmS ifmStreamerData = {};

    ifmStreamerData.fmData.bufferId = inputDramBufferId;

    StreamersUtils::SetBufferDataType(ifmStreamerData.fmData, inputDramBuffer->m_Format);
    ifmStreamerData.fmData.fcafInfo.signedActivation = false;
    ifmStreamerData.fmData.fcafInfo.zeroPoint =
        static_cast<uint8_t>(inputDramBuffer->m_QuantizationInfo.GetZeroPoint());

    CommonUtils::SetTileInfoForBuffer(m_Capabilities, ifmStreamerData.fmData.tile, inputSramBuffer);

    StreamersUtils::SetStripeHeightInfo(ifmStreamerData.fmData, inputSramBuffer->m_TensorShape,
                                        inputSramBuffer->m_StripeShape);
    StreamersUtils::SetStripeWidthInfo(ifmStreamerData.fmData, inputSramBuffer->m_TensorShape,
                                       inputSramBuffer->m_StripeShape);
    StreamersUtils::SetStripeChannelsInfo(ifmStreamerData.fmData, inputSramBuffer->m_TensorShape,
                                          inputSramBuffer->m_StripeShape);

    StreamersUtils::SetSuperTensorSizeInCells(ifmStreamerData.fmData, inputDramBuffer->m_TensorShape,
                                              inputDramBuffer->m_Format);

    StreamersUtils::SetStripeIdStrides(ifmStreamerData.fmData, inputDramBuffer->m_Order);

    AgentDependencyInfo dependencyInfo = {};
    dependencyInfo.numStripesTotal     = static_cast<uint16_t>(
        utils::GetNumStripesTotal(inputSramBuffer->m_TensorShape, inputSramBuffer->m_StripeShape));

    Agent ifmStreamerAgent{ ifmStreamerData, dependencyInfo };

    // Push the Ifm Streamer agent to the command stream
    AgentIdType agentId         = m_CommandStreamAgents.size();
    m_OpToAgentIdMapping[ptrOp] = agentId;
    m_CommandStreamAgents.push_back(ifmStreamerAgent);

    return agentId;
}

// Private function to add WGT_STREAMER to the command stream
AgentIdType CascadingCompiler::AddWeightStreamerToCommandStream(DmaOp* const ptrDmaOp)
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
    weightStreamerData.bufferId =
        static_cast<uint16_t>(m_BufferManager.AddDramConstant(BufferType::ConstantDma, compressedWeights));
    weightStreamerData.metadataBufferId =
        static_cast<uint16_t>(m_BufferManager.AddDramConstant(BufferType::ConstantControlUnit, metadataBytes));
    CommonUtils::SetTileInfoForBuffer(m_Capabilities, weightStreamerData.tile, weightsSramBuffer);

    weightStreamerData.numStripes.ifmChannels =
        static_cast<uint16_t>(utils::GetNumStripesC(ifmBuffer->m_TensorShape, ifmBuffer->m_StripeShape));
    weightStreamerData.numStripes.ofmChannels =
        static_cast<uint16_t>(utils::GetNumStripesC(ofmBuffer->m_TensorShape, ofmBuffer->m_StripeShape));
    weightStreamerData.stripeIdStrides.ifmChannels = 1;
    weightStreamerData.stripeIdStrides.ofmChannels = weightStreamerData.numStripes.ifmChannels;

    AgentDependencyInfo dependencyInfo = {};

    Agent weightStreamerAgent{ weightStreamerData, dependencyInfo };
    dependencyInfo.numStripesTotal = static_cast<uint16_t>(
        utils::GetNumStripesTotal(weightsSramBuffer->m_TensorShape, weightsSramBuffer->m_StripeShape));

    // Push the Weight Streamer agent to the command stream
    AgentIdType agentId            = m_CommandStreamAgents.size();
    m_OpToAgentIdMapping[ptrDmaOp] = agentId;
    m_CommandStreamAgents.push_back(weightStreamerAgent);

    return agentId;
}

// Private function to add MCE_SCHEDULER to the command stream
AgentIdType CascadingCompiler::AddMceSchedulerToCommandStream(MceOp* const ptrMceOp, const PleKernelId pleKernelId)
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

    mceSchedulerData.blockSize.width  = static_cast<uint8_t>(ptrMceOp->m_BlockConfig.m_BlockWidth());
    mceSchedulerData.blockSize.height = static_cast<uint8_t>(ptrMceOp->m_BlockConfig.m_BlockHeight());

    MceSUtils::SetMcesOfmHeightStripeInfo(mceSchedulerData, outputBuffer->m_TensorShape, ptrMceOp->m_OutputStripeShape);
    MceSUtils::SetMcesOfmWidthStripeInfo(mceSchedulerData, outputBuffer->m_TensorShape, ptrMceOp->m_OutputStripeShape);
    MceSUtils::SetMcesOfmChannelsStripeInfo(mceSchedulerData, outputBuffer->m_TensorShape,
                                            ptrMceOp->m_OutputStripeShape);
    MceSUtils::SetMcesIfmChannelsStripeInfo(mceSchedulerData, inputBuffer->m_TensorShape, inputBuffer->m_StripeShape);

    MceSUtils::SetStripeIdStrides(mceSchedulerData, outputBuffer->m_Order);

    mceSchedulerData.convStrideXy.x = static_cast<uint8_t>(ptrMceOp->m_Stride.m_X);
    mceSchedulerData.convStrideXy.y = static_cast<uint8_t>(ptrMceOp->m_Stride.m_Y);
    mceSchedulerData.ifmZeroPoint   = static_cast<int16_t>(inputBuffer->m_QuantizationInfo.GetZeroPoint());

    MceSUtils::setMcesOpMode(mceSchedulerData, ptrMceOp->m_Op);
    MceSUtils::setMcesAlgorithm(mceSchedulerData, ptrMceOp->m_Algo);

    mceSchedulerData.filterShape.height = static_cast<uint8_t>(weightBuffer->m_TensorShape[1]);
    mceSchedulerData.filterShape.width  = static_cast<uint8_t>(weightBuffer->m_TensorShape[2]);
    mceSchedulerData.padding.left       = static_cast<uint8_t>(ptrMceOp->m_PadLeft);
    mceSchedulerData.padding.top        = static_cast<uint8_t>(ptrMceOp->m_PadTop);
    mceSchedulerData.ifmDeltaDefault.height =
        static_cast<int8_t>(inputBuffer->m_TensorShape[1] - outputBuffer->m_TensorShape[1]);
    mceSchedulerData.ifmDeltaDefault.width =
        static_cast<int8_t>(inputBuffer->m_TensorShape[2] - outputBuffer->m_TensorShape[2]);
    mceSchedulerData.ifmDeltaEdge.height =
        static_cast<int8_t>(inputBuffer->m_TensorShape[1] - outputBuffer->m_TensorShape[1]);
    mceSchedulerData.ifmDeltaEdge.width =
        static_cast<int8_t>(inputBuffer->m_TensorShape[2] - outputBuffer->m_TensorShape[2]);
    mceSchedulerData.reluActiv.min = ptrMceOp->m_LowerBound;
    mceSchedulerData.reluActiv.max = ptrMceOp->m_UpperBound;
    mceSchedulerData.pleKernelId   = pleKernelId;

    AgentDependencyInfo dependencyInfo = {};
    dependencyInfo.numStripesTotal =
        static_cast<uint16_t>(utils::GetNumStripesTotal(outputBuffer->m_TensorShape, outputBuffer->m_StripeShape));

    Agent mceSchedulerAgent{ mceSchedulerData, dependencyInfo };

    // Push the Mce Scheduler agent to the command stream
    AgentIdType agentId            = m_CommandStreamAgents.size();
    m_OpToAgentIdMapping[ptrMceOp] = agentId;
    m_CommandStreamAgents.push_back(mceSchedulerAgent);

    return agentId;
}

// Private function to add PLE_LOADER to the command stream
AgentIdType CascadingCompiler::AddPleLoaderToCommandStream(PleOp* const ptrPleOp)
{
    // Create a new Ple Loader agent
    PleL pleLoaderData        = {};
    pleLoaderData.pleKernelId = ptrPleOp->m_PleKernelId;
    pleLoaderData.sramAddr    = static_cast<uint16_t>(ptrPleOp->m_Offset.value());

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
AgentIdType CascadingCompiler::AddPleSchedulerToCommandStream(PleOp* const ptrPleOp)
{
    // Get the input buffers to the Ple Op
    OpGraph::BufferList inputBuffers = m_MergedOpGraph.GetInputs(ptrPleOp);
    assert(inputBuffers.size() == 1 || inputBuffers.size() == 2);

    Buffer* inputBuffer0 = inputBuffers[g_PleInputBuffer0Index];

    // Get the output buffer from the Ple Op
    Buffer* outputBuffer = m_MergedOpGraph.GetOutput(ptrPleOp);
    assert(ptrPleOp->m_OutputStripeShape == outputBuffer->m_StripeShape);

    PleS pleS = {};

    CommonUtils::SetTileInfoForBuffer(m_Capabilities, pleS.ofmTile, outputBuffer);

    pleS.ofmZeroPoint = static_cast<int16_t>(outputBuffer->m_QuantizationInfo.GetZeroPoint());

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

    pleS.pleKernelSramAddr = static_cast<uint16_t>(ptrPleOp->m_Offset.value());

    pleS.pleKernelId = ptrPleOp->m_PleKernelId;

    if (pleS.inputMode == PleInputMode::SRAM)
    {
        CommonUtils::SetTileInfoForBuffer(m_Capabilities, pleS.ifmTile0, inputBuffer0);

        const double outputScale = outputBuffer->m_QuantizationInfo.GetScale();
        const double inputScale0 = inputBuffer0->m_QuantizationInfo.GetScale();
        uint16_t multiplier0;
        uint16_t shift0;
        utils::CalculateRescaleMultiplierAndShift(inputScale0 / outputScale, multiplier0, shift0);

        pleS.ifmInfo0 = { static_cast<int16_t>(inputBuffer0->m_QuantizationInfo.GetZeroPoint()), multiplier0, shift0 };

        if (inputBuffers.size() == 2)
        {
            Buffer* inputBuffer1 = inputBuffers[g_PleInputBuffer1Index];

            const double inputScale1 = inputBuffer1->m_QuantizationInfo.GetScale();
            uint16_t multiplier1;
            uint16_t shift1;
            utils::CalculateRescaleMultiplierAndShift(inputScale1 / outputScale, multiplier1, shift1);

            CommonUtils::SetTileInfoForBuffer(m_Capabilities, pleS.ifmTile1, inputBuffer1);

            pleS.ifmInfo1 = { static_cast<int16_t>(inputBuffer1->m_QuantizationInfo.GetZeroPoint()), multiplier1,
                              shift1 };
        }
    }

    AgentData agentData{ pleS };

    AgentDependencyInfo info = {};
    info.numStripesTotal =
        static_cast<uint16_t>(utils::GetNumStripesTotal(outputBuffer->m_TensorShape, outputBuffer->m_StripeShape));

    Agent pleSchedulerAgent{ agentData, info };

    // Push the Ple Scheduler agent to the command stream
    AgentIdType agentId            = m_CommandStreamAgents.size();
    m_OpToAgentIdMapping[ptrPleOp] = agentId;
    m_CommandStreamAgents.push_back(pleSchedulerAgent);

    return agentId;
}

// Private function to add OFM_STREAMER to the command stream
AgentIdType CascadingCompiler::AddOfmStreamerToCommandStream(Op* const ptrOp,
                                                             const Buffer* const outputSramBuffer,
                                                             const uint16_t outputDramBufferId,
                                                             const Buffer* const outputDramBuffer)
{
    assert(IsObjectOfType<DmaOp>(ptrOp) || IsObjectOfType<ConcatOp>(ptrOp));

    OfmS ofmStreamerData = {};

    if (outputDramBuffer->m_Offset.has_value())
    {
        ofmStreamerData.fmData.dramOffset = outputDramBuffer->m_Offset.value();
    }

    ofmStreamerData.fmData.bufferId = outputDramBufferId;

    StreamersUtils::SetBufferDataType(ofmStreamerData.fmData, outputDramBuffer->m_Format);

    ofmStreamerData.fmData.fcafInfo.signedActivation = false;
    ofmStreamerData.fmData.fcafInfo.zeroPoint =
        static_cast<uint8_t>(outputDramBuffer->m_QuantizationInfo.GetZeroPoint());

    CommonUtils::SetTileInfoForBuffer(m_Capabilities, ofmStreamerData.fmData.tile, outputSramBuffer);

    StreamersUtils::SetStripeHeightInfo(ofmStreamerData.fmData, outputSramBuffer->m_TensorShape,
                                        outputSramBuffer->m_StripeShape);
    StreamersUtils::SetStripeWidthInfo(ofmStreamerData.fmData, outputSramBuffer->m_TensorShape,
                                       outputSramBuffer->m_StripeShape);
    StreamersUtils::SetStripeChannelsInfo(ofmStreamerData.fmData, outputSramBuffer->m_TensorShape,
                                          outputSramBuffer->m_StripeShape);

    StreamersUtils::SetSuperTensorSizeInCells(ofmStreamerData.fmData, outputSramBuffer->m_TensorShape,
                                              outputSramBuffer->m_Format);

    StreamersUtils::SetStripeIdStrides(ofmStreamerData.fmData, outputDramBuffer->m_Order);

    AgentDependencyInfo dependencyInfo = {};
    dependencyInfo.numStripesTotal     = static_cast<uint16_t>(
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
inline void CascadingCompiler::AddReadAfterWriteDependency(const AgentType consumerAgentType,
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
inline void CascadingCompiler::AddSramOverlapDependency(const command_stream::cascading::AgentType consumerAgentType,
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
inline void CascadingCompiler::AddWriteAfterReadDependency(const AgentType consumerAgentType,
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
inline void CascadingCompiler::AddScheduleTimeDependency(const AgentType consumerAgentType,
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
void CascadingCompiler::FillConsumerAgentDependency(command_stream::cascading::Dependency& consumerAgentDependency,
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
                consumerAgentDependency.outerRatio.other = static_cast<uint16_t>(
                    producerAgent.data.ofm.fmData.numStripes.height * producerAgent.data.ofm.fmData.numStripes.width *
                    producerAgent.data.ofm.fmData.numStripes.channels);
                consumerAgentDependency.outerRatio.self = static_cast<uint16_t>(
                    consumerAgent.data.ifm.fmData.numStripes.height * consumerAgent.data.ifm.fmData.numStripes.width *
                    consumerAgent.data.ifm.fmData.numStripes.channels);

                consumerAgentDependency.innerRatio.other = static_cast<uint16_t>(
                    producerAgent.data.ofm.fmData.numStripes.height * producerAgent.data.ofm.fmData.numStripes.width *
                    producerAgent.data.ofm.fmData.numStripes.channels);
                consumerAgentDependency.innerRatio.self = 1;

                consumerAgentDependency.boundary = 0;
            }
            break;
        }

        case AgentType::WGT_STREAMER:
        {
            // Sram Overlap Dependency for [WeightStreamer][OfmStreamer]
            if (producerAgentType == AgentType::OFM_STREAMER)
            {
                consumerAgentDependency.outerRatio.other = static_cast<uint16_t>(
                    producerAgent.data.ofm.fmData.numStripes.height * producerAgent.data.ofm.fmData.numStripes.width *
                    producerAgent.data.ofm.fmData.numStripes.channels);
                consumerAgentDependency.outerRatio.self = static_cast<uint16_t>(
                    consumerAgent.data.wgt.numStripes.ifmChannels * consumerAgent.data.wgt.numStripes.ofmChannels);

                consumerAgentDependency.innerRatio.other = static_cast<uint16_t>(
                    producerAgent.data.ofm.fmData.numStripes.height * producerAgent.data.ofm.fmData.numStripes.width *
                    producerAgent.data.ofm.fmData.numStripes.channels);
                consumerAgentDependency.innerRatio.self = 1;

                consumerAgentDependency.boundary = 0;
            }
            break;
        }

        case AgentType::MCE_SCHEDULER:
        {
            // Read After Write Dependency for [MceScheduler][IfmStreamer]
            if (producerAgentType == AgentType::IFM_STREAMER)
            {
                consumerAgentDependency.outerRatio.other = static_cast<uint16_t>(
                    producerAgent.data.ifm.fmData.numStripes.width * producerAgent.data.ifm.fmData.numStripes.height *
                    producerAgent.data.ifm.fmData.numStripes.channels);
                consumerAgentDependency.outerRatio.self = static_cast<uint16_t>(
                    consumerAgent.data.mce.numStripes.ofmHeight * consumerAgent.data.mce.numStripes.ofmWidth *
                    consumerAgent.data.mce.numStripes.ifmChannels);

                uint8_t widthRatio  = static_cast<uint8_t>(utils::DivRoundUp(
                    consumerAgent.data.mce.numStripes.ofmWidth, producerAgent.data.ifm.fmData.numStripes.width));
                uint8_t heightRatio = static_cast<uint8_t>(utils::DivRoundUp(
                    consumerAgent.data.mce.numStripes.ofmHeight, producerAgent.data.ifm.fmData.numStripes.height));

                assert(consumerAgent.data.mce.numStripes.ifmChannels ==
                       producerAgent.data.ifm.fmData.numStripes.channels);

                consumerAgentDependency.innerRatio.other = static_cast<uint16_t>(widthRatio * heightRatio);
                consumerAgentDependency.innerRatio.self  = 1;

                if ((producerAgent.data.ifm.fmData.numStripes.height > 1 &&
                     consumerAgent.data.mce.filterShape.height > 1) ||
                    (producerAgent.data.ifm.fmData.numStripes.width > 1 &&
                     consumerAgent.data.mce.filterShape.width > 1))
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
                consumerAgentDependency.outerRatio.other = 1;
                consumerAgentDependency.innerRatio.other = 1;

                if (producerAgent.data.wgt.numStripes.ofmChannels > 1)
                {
                    consumerAgentDependency.outerRatio.self = 1;
                    consumerAgentDependency.innerRatio.self = 1;
                }
                else
                {
                    consumerAgentDependency.outerRatio.self = static_cast<uint16_t>(
                        consumerAgent.data.mce.numStripes.ofmHeight * consumerAgent.data.mce.numStripes.ofmWidth);
                    consumerAgentDependency.innerRatio.self = static_cast<uint16_t>(
                        consumerAgent.data.mce.numStripes.ofmHeight * consumerAgent.data.mce.numStripes.ofmWidth);
                }

                consumerAgentDependency.boundary = 0;
            }
            else
            {
                assert(false);
            }

            break;
        }

        case AgentType::PLE_LOADER:
        {
            break;
        }

        case AgentType::PLE_SCHEDULER:
        {
            // Read After Write Dependency for [PleScheduler][IfmStreamer]
            if (producerAgentType == AgentType::IFM_STREAMER)
            {
                // Calculate outer ratios using number of stripes
                consumerAgentDependency.outerRatio.other = static_cast<uint16_t>(
                    producerAgent.data.ifm.fmData.numStripes.width * producerAgent.data.ifm.fmData.numStripes.height *
                    producerAgent.data.ifm.fmData.numStripes.channels);
                consumerAgentDependency.outerRatio.self = static_cast<uint16_t>(
                    consumerAgent.data.pleS.numStripes.height * consumerAgent.data.pleS.numStripes.width *
                    consumerAgent.data.pleS.numStripes.channels);

                // Calculate inner ratios using ratio of stripe size
                uint8_t widthRatio   = static_cast<uint8_t>(utils::DivRoundUp(
                    consumerAgent.data.pleS.numStripes.width, producerAgent.data.ifm.fmData.numStripes.width));
                uint8_t heightRatio  = static_cast<uint8_t>(utils::DivRoundUp(
                    consumerAgent.data.pleS.numStripes.height, producerAgent.data.ifm.fmData.numStripes.height));
                uint8_t channelRatio = static_cast<uint8_t>(utils::DivRoundUp(
                    consumerAgent.data.pleS.numStripes.channels, producerAgent.data.ifm.fmData.numStripes.channels));

                consumerAgentDependency.innerRatio.other =
                    static_cast<uint16_t>(widthRatio * heightRatio * channelRatio);
                consumerAgentDependency.innerRatio.self = 1;

                // Set boundary to 1 if producer stripe count is not a factor of consumer stripe count
                uint8_t numberOfIfmStripesInXYDimProducer = static_cast<uint8_t>(
                    producerAgent.data.ifm.fmData.numStripes.width * producerAgent.data.ifm.fmData.numStripes.height);
                uint8_t numberOfIfmStripesInXYDimConsumer = static_cast<uint8_t>(
                    consumerAgent.data.pleS.numStripes.width * consumerAgent.data.pleS.numStripes.height);

                uint8_t ifmStripeRemainder =
                    static_cast<uint8_t>(numberOfIfmStripesInXYDimConsumer % numberOfIfmStripesInXYDimProducer);
                if (ifmStripeRemainder == 0)
                {
                    consumerAgentDependency.boundary = 0;
                }
                else
                {
                    consumerAgentDependency.boundary = 1;
                }
            }
            // Read After Write Dependency for [PleScheduler][MceScheduler]
            else if (producerAgentType == AgentType::MCE_SCHEDULER)
            {
                // Calculate outer ratios using number of stripes
                consumerAgentDependency.outerRatio.other = static_cast<uint16_t>(
                    producerAgent.data.mce.numStripes.ofmWidth * producerAgent.data.mce.numStripes.ofmHeight *
                    producerAgent.data.mce.numStripes.ofmChannels);
                consumerAgentDependency.outerRatio.self = static_cast<uint16_t>(
                    consumerAgent.data.pleS.numStripes.height * consumerAgent.data.pleS.numStripes.width *
                    consumerAgent.data.pleS.numStripes.channels);

                // Calculate inner ratios using ratio of stripe size
                uint8_t widthRatio   = static_cast<uint8_t>(utils::DivRoundUp(
                    consumerAgent.data.pleS.numStripes.width, producerAgent.data.mce.numStripes.ofmWidth));
                uint8_t heightRatio  = static_cast<uint8_t>(utils::DivRoundUp(
                    consumerAgent.data.pleS.numStripes.height, producerAgent.data.mce.numStripes.ofmHeight));
                uint8_t channelRatio = static_cast<uint8_t>(utils::DivRoundUp(
                    consumerAgent.data.pleS.numStripes.channels, producerAgent.data.mce.numStripes.ofmChannels));

                consumerAgentDependency.innerRatio.other =
                    static_cast<uint16_t>(widthRatio * heightRatio * channelRatio);
                consumerAgentDependency.innerRatio.self = 1;

                // Set boundary to 1 if producer stripe count is not a factor of consumer stripe count
                uint8_t numberOfIfmStripesInXYDimProducer = static_cast<uint8_t>(
                    producerAgent.data.mce.numStripes.ofmWidth * producerAgent.data.mce.numStripes.ofmHeight);
                uint8_t numberOfIfmStripesInXYDimConsumer = static_cast<uint8_t>(
                    consumerAgent.data.pleS.numStripes.width * consumerAgent.data.pleS.numStripes.height);

                uint8_t ifmStripeRemainder =
                    static_cast<uint8_t>(numberOfIfmStripesInXYDimConsumer % numberOfIfmStripesInXYDimProducer);
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
                consumerAgentDependency.outerRatio.self  = static_cast<uint16_t>(
                    consumerAgent.data.pleS.numStripes.height * consumerAgent.data.pleS.numStripes.width *
                    consumerAgent.data.pleS.numStripes.channels);

                uint8_t widthRatio   = static_cast<uint8_t>(consumerAgent.data.pleS.numStripes.width / 1U);
                uint8_t heightRatio  = static_cast<uint8_t>(consumerAgent.data.pleS.numStripes.height / 1U);
                uint8_t channelRatio = static_cast<uint8_t>(consumerAgent.data.pleS.numStripes.channels / 1U);

                consumerAgentDependency.innerRatio.other = 1U;
                consumerAgentDependency.innerRatio.self =
                    static_cast<uint16_t>(widthRatio * heightRatio * channelRatio);

                consumerAgentDependency.boundary = 0;
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
                consumerAgentDependency.outerRatio.other = static_cast<uint16_t>(
                    producerAgent.data.ofm.fmData.numStripes.height * producerAgent.data.ofm.fmData.numStripes.width *
                    producerAgent.data.ofm.fmData.numStripes.channels);
                consumerAgentDependency.outerRatio.self = static_cast<uint16_t>(
                    consumerAgent.data.ofm.fmData.numStripes.height * consumerAgent.data.ofm.fmData.numStripes.width *
                    consumerAgent.data.ofm.fmData.numStripes.channels);

                consumerAgentDependency.innerRatio.other = static_cast<uint16_t>(
                    producerAgent.data.ofm.fmData.numStripes.height * producerAgent.data.ofm.fmData.numStripes.width *
                    producerAgent.data.ofm.fmData.numStripes.channels);
                consumerAgentDependency.innerRatio.self = 1;

                consumerAgentDependency.boundary = 0;
            }
            // Read After Write Dependency for [OfmStreamer][PleScheduler]
            else if (producerAgentType == AgentType::PLE_SCHEDULER)
            {
                consumerAgentDependency.outerRatio.other = 1;
                consumerAgentDependency.outerRatio.self  = 1;

                consumerAgentDependency.innerRatio.other = 1;
                consumerAgentDependency.innerRatio.self  = 1;

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
            break;
        }
    }
}

// Private function to fill the dependency data for Write After Read or Schedule Time dependencies
void CascadingCompiler::FillProducerAgentDependency(command_stream::cascading::Dependency& producerAgentDependency,
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
                producerAgentDependency.outerRatio.other = static_cast<uint16_t>(
                    consumerAgent.data.ofm.fmData.numStripes.height * consumerAgent.data.ofm.fmData.numStripes.width *
                    consumerAgent.data.ofm.fmData.numStripes.channels);
                producerAgentDependency.outerRatio.self = static_cast<uint16_t>(
                    producerAgent.data.ofm.fmData.numStripes.height * producerAgent.data.ofm.fmData.numStripes.width *
                    producerAgent.data.ofm.fmData.numStripes.channels);

                producerAgentDependency.innerRatio.other = 1;
                producerAgentDependency.innerRatio.self  = static_cast<uint16_t>(
                    producerAgent.data.ofm.fmData.numStripes.height * producerAgent.data.ofm.fmData.numStripes.width *
                    producerAgent.data.ofm.fmData.numStripes.channels);

                producerAgentDependency.boundary = 0;
            }
            break;
        }

        case AgentType::WGT_STREAMER:
        {
            break;
        }

        case AgentType::MCE_SCHEDULER:
        {
            // Write After Read Dependency for [IfmStreamer][MceScheduler] or
            // Schedule Time Dependency for [IfmStreamer][MceScheduler]
            if (producerAgentType == AgentType::IFM_STREAMER)
            {
                producerAgentDependency.outerRatio.other = static_cast<uint16_t>(
                    consumerAgent.data.mce.numStripes.ofmHeight * consumerAgent.data.mce.numStripes.ofmWidth *
                    consumerAgent.data.mce.numStripes.ifmChannels);
                producerAgentDependency.outerRatio.self = static_cast<uint16_t>(
                    producerAgent.data.ifm.fmData.numStripes.width * producerAgent.data.ifm.fmData.numStripes.height *
                    producerAgent.data.ifm.fmData.numStripes.channels);

                uint8_t widthRatio  = static_cast<uint8_t>(utils::DivRoundUp(
                    consumerAgent.data.mce.numStripes.ofmWidth, producerAgent.data.ifm.fmData.numStripes.width));
                uint8_t heightRatio = static_cast<uint8_t>(utils::DivRoundUp(
                    consumerAgent.data.mce.numStripes.ofmHeight, producerAgent.data.ifm.fmData.numStripes.height));

                assert(producerAgent.data.ifm.fmData.numStripes.channels ==
                       consumerAgent.data.mce.numStripes.ifmChannels);

                producerAgentDependency.innerRatio.other = 1;
                producerAgentDependency.innerRatio.self  = static_cast<uint16_t>(widthRatio * heightRatio);

                if ((producerAgent.data.ifm.fmData.numStripes.height > 1 &&
                     consumerAgent.data.mce.filterShape.height > 1) ||
                    (producerAgent.data.ifm.fmData.numStripes.width > 1 &&
                     consumerAgent.data.mce.filterShape.width > 1))
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
                if (producerAgent.data.wgt.numStripes.ofmChannels > 1)
                {
                    producerAgentDependency.outerRatio.other = 1;
                    producerAgentDependency.innerRatio.other = 1;
                }
                else
                {
                    producerAgentDependency.outerRatio.other = static_cast<uint16_t>(
                        consumerAgent.data.mce.numStripes.ofmHeight * consumerAgent.data.mce.numStripes.ofmWidth);
                    producerAgentDependency.innerRatio.other = static_cast<uint16_t>(
                        consumerAgent.data.mce.numStripes.ofmHeight * consumerAgent.data.mce.numStripes.ofmWidth);
                }

                producerAgentDependency.outerRatio.self = 1;
                producerAgentDependency.innerRatio.self = 1;

                producerAgentDependency.boundary = 0;
            }
            // Schedule Time Dependency for [PleLoader][MceScheduler]
            else if (producerAgentType == AgentType::PLE_LOADER)
            {
                producerAgentDependency.outerRatio.other = static_cast<uint16_t>(
                    consumerAgent.data.mce.numStripes.ofmHeight * consumerAgent.data.mce.numStripes.ofmWidth *
                    consumerAgent.data.mce.numStripes.ifmChannels);
                producerAgentDependency.outerRatio.self = 1;

                producerAgentDependency.innerRatio.other = static_cast<uint16_t>(
                    consumerAgent.data.mce.numStripes.ofmHeight * consumerAgent.data.mce.numStripes.ofmWidth *
                    consumerAgent.data.mce.numStripes.ifmChannels);
                producerAgentDependency.innerRatio.self = 1;

                producerAgentDependency.boundary = 0;
            }
            else
            {
                assert(false);
            }

            break;
        }

        case AgentType::PLE_LOADER:
        {
            break;
        }

        case AgentType::PLE_SCHEDULER:
        {
            // Write After Read Dependency for [IfmStreamer][PleScheduler] or
            // Schedule Time Dependency for [IfmStreamer][PleScheduler]
            if (producerAgentType == AgentType::IFM_STREAMER)
            {
                // Calculate outer ratios using number of stripes.
                producerAgentDependency.outerRatio.other = static_cast<uint16_t>(
                    consumerAgent.data.pleS.numStripes.height * consumerAgent.data.pleS.numStripes.width *
                    consumerAgent.data.pleS.numStripes.channels);
                producerAgentDependency.outerRatio.self = static_cast<uint16_t>(
                    producerAgent.data.ifm.fmData.numStripes.width * producerAgent.data.ifm.fmData.numStripes.height *
                    producerAgent.data.ifm.fmData.numStripes.channels);

                // Calculate inner ratios using ratio of stripe size
                uint8_t widthRatio   = static_cast<uint8_t>(utils::DivRoundUp(
                    consumerAgent.data.pleS.numStripes.width, producerAgent.data.ifm.fmData.numStripes.width));
                uint8_t heightRatio  = static_cast<uint8_t>(utils::DivRoundUp(
                    consumerAgent.data.pleS.numStripes.height, producerAgent.data.ifm.fmData.numStripes.height));
                uint8_t channelRatio = static_cast<uint8_t>(utils::DivRoundUp(
                    consumerAgent.data.pleS.numStripes.channels, producerAgent.data.ifm.fmData.numStripes.channels));

                producerAgentDependency.innerRatio.other = 1U;
                producerAgentDependency.innerRatio.self =
                    static_cast<uint16_t>(widthRatio * heightRatio * channelRatio);

                // Set boundary to 1 if producer stripe count is not a factor of consumer stripe count
                uint16_t numberOfIfmStripesInXYDimProducer = static_cast<uint16_t>(
                    producerAgent.data.ifm.fmData.numStripes.width * producerAgent.data.ifm.fmData.numStripes.height *
                    producerAgent.data.ifm.fmData.numStripes.channels);
                uint16_t numberOfIfmStripesInXYDimConsumer = static_cast<uint16_t>(
                    consumerAgent.data.pleS.numStripes.width * consumerAgent.data.pleS.numStripes.height *
                    consumerAgent.data.pleS.numStripes.channels);

                uint8_t ifmStripeRemainder =
                    static_cast<uint8_t>(numberOfIfmStripesInXYDimConsumer % numberOfIfmStripesInXYDimProducer);

                if (ifmStripeRemainder == 0)
                {
                    producerAgentDependency.boundary = 0U;
                }
                else
                {
                    producerAgentDependency.boundary = 1U;
                }
            }
            else if (producerAgentType == AgentType::MCE_SCHEDULER)
            {
                // Calculate outer ratios using number of stripes
                producerAgentDependency.outerRatio.other = static_cast<uint16_t>(
                    consumerAgent.data.pleS.numStripes.height * consumerAgent.data.pleS.numStripes.width *
                    consumerAgent.data.pleS.numStripes.channels);
                producerAgentDependency.outerRatio.self = static_cast<uint16_t>(
                    producerAgent.data.mce.numStripes.ofmHeight * producerAgent.data.mce.numStripes.ofmWidth *
                    producerAgent.data.mce.numStripes.ofmChannels);

                // Calculate inner ratios using ratio of stripe size
                uint8_t widthRatio   = static_cast<uint8_t>(utils::DivRoundUp(
                    consumerAgent.data.pleS.numStripes.width, producerAgent.data.mce.numStripes.ofmWidth));
                uint8_t heightRatio  = static_cast<uint8_t>(utils::DivRoundUp(
                    consumerAgent.data.pleS.numStripes.height, producerAgent.data.mce.numStripes.ofmHeight));
                uint8_t channelRatio = static_cast<uint8_t>(utils::DivRoundUp(
                    consumerAgent.data.pleS.numStripes.channels, producerAgent.data.mce.numStripes.ofmChannels));

                producerAgentDependency.innerRatio.other = 1;
                producerAgentDependency.innerRatio.self =
                    static_cast<uint16_t>(widthRatio * heightRatio * channelRatio);

                // Set boundary to 1 if producer stripe count is not a factor of consumer stripe count
                uint16_t numberOfIfmStripesInXYDimProducer = static_cast<uint16_t>(
                    producerAgent.data.mce.numStripes.ofmWidth * producerAgent.data.mce.numStripes.ofmHeight *
                    producerAgent.data.mce.numStripes.ofmChannels);
                uint16_t numberOfIfmStripesInXYDimConsumer = static_cast<uint16_t>(
                    consumerAgent.data.pleS.numStripes.width * consumerAgent.data.pleS.numStripes.height *
                    consumerAgent.data.pleS.numStripes.channels);

                uint8_t ifmStripeRemainder =
                    static_cast<uint8_t>(numberOfIfmStripesInXYDimConsumer % numberOfIfmStripesInXYDimProducer);

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
                producerAgentDependency.outerRatio.other = static_cast<uint16_t>(
                    consumerAgent.data.pleS.numStripes.height * consumerAgent.data.pleS.numStripes.width *
                    consumerAgent.data.pleS.numStripes.channels);
                producerAgentDependency.outerRatio.self = 1U;

                uint8_t widthRatio   = static_cast<uint8_t>(consumerAgent.data.pleS.numStripes.width);
                uint8_t heightRatio  = static_cast<uint8_t>(consumerAgent.data.pleS.numStripes.height);
                uint8_t channelRatio = static_cast<uint8_t>(consumerAgent.data.pleS.numStripes.channels);

                producerAgentDependency.innerRatio.other =
                    static_cast<uint16_t>(widthRatio * heightRatio * channelRatio);
                producerAgentDependency.innerRatio.self = 1U;

                producerAgentDependency.boundary = 0;
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
                producerAgentDependency.outerRatio.other = static_cast<uint16_t>(
                    consumerAgent.data.ofm.fmData.numStripes.height * consumerAgent.data.ofm.fmData.numStripes.width *
                    consumerAgent.data.ofm.fmData.numStripes.channels);
                producerAgentDependency.outerRatio.self = static_cast<uint16_t>(
                    producerAgent.data.ofm.fmData.numStripes.height * producerAgent.data.ofm.fmData.numStripes.width *
                    producerAgent.data.ofm.fmData.numStripes.channels);

                producerAgentDependency.innerRatio.other = 1;
                producerAgentDependency.innerRatio.self  = static_cast<uint16_t>(
                    producerAgent.data.ofm.fmData.numStripes.height * producerAgent.data.ofm.fmData.numStripes.width *
                    producerAgent.data.ofm.fmData.numStripes.channels);

                producerAgentDependency.boundary = 0;
            }
            // Write After Read Dependency for [PleScheduler][OfmStreamer] or
            // Schedule Time Dependency for [PleScheduler][OfmStreamer]
            else if (producerAgentType == AgentType::PLE_SCHEDULER)
            {
                producerAgentDependency.outerRatio.other = 1;
                producerAgentDependency.outerRatio.self  = 1;

                producerAgentDependency.innerRatio.other = 1;
                producerAgentDependency.innerRatio.self  = 1;

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
            break;
        }
    }
}

const BufferManager& CascadingCompiler::GetBufferManager()
{
    return m_BufferManager;
}

const OpGraph& CascadingCompiler::GetMergedOpGraph()
{
    return m_MergedOpGraph;
}

const std::unordered_map<Buffer*, uint32_t>& CascadingCompiler::GetIntermdiateDramBufToBufIdMapping()
{
    return m_IntermdiateDramBufToBufIdMapping;
}

// Private function to add the lifetime information of the intermediate DRAM buffers
void CascadingCompiler::AddLifetimeInfoForIntermediateDramBuffers()
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
                // Set the Lifetime start and end of the intermediate DRAM buffer
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
                m_BufferManager.MarkBufferUsedAtTime(m_IntermdiateDramBufToBufIdMapping.at(buffer),
                                                     static_cast<uint32_t>(lifetimeStart),
                                                     static_cast<uint32_t>(lifetimeEnd + 1));
            }
        }
    }
}
}    // namespace cascading_compiler
}    // namespace support_library
}    // namespace ethosn
