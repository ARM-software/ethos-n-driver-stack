//
// Copyright © 2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "CascadingCommandStreamGenerator.hpp"

#include "CascadingCommandStreamGeneratorUtils.hpp"
#include "Compiler.hpp"
#include "Visualisation.hpp"

#include <iomanip>
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
    , m_FenceOpForIfmS(nullptr)
    , m_FenceOpForPleL(nullptr)
    , m_FenceOpForWgtS(nullptr)
{

    m_CommandStreamAgents.reserve(m_MergedOpGraph.GetOps().size());
}

CascadingCommandStreamGenerator::~CascadingCommandStreamGenerator()
{}

// Compile a given network and return the compiled network
CompiledOpGraph CascadingCommandStreamGenerator::Generate()
{
    assert(m_MergedOpGraph.GetOps().size() != 0 && m_CommandStreamAgents.size() == 0);

    try
    {
        for (auto currentOp : m_MergedOpGraph.GetOps())
        {
            if (IsObjectOfType<DmaOp>(currentOp))
            {
                ProcessDmaOp(static_cast<DmaOp*>(currentOp));
            }
            else if (IsObjectOfType<MceOp>(currentOp))
            {
                ProcessMceOp(currentOp);
            }
            else if (IsObjectOfType<PleOp>(currentOp))
            {
                ProcessPleOp(currentOp);
            }
            else
            {
                throw NotSupportedException("Op is not currently supported by the Cascading Compiler");
            }

            Buffer* producedBuffer = m_MergedOpGraph.GetOutput(currentOp);
            if (producedBuffer != nullptr && producedBuffer->IsFullTensor() &&
                !(IsObjectOfType<DmaOp>(currentOp) && producedBuffer->m_Location == Location::Sram))
            {
                m_FenceOpForIfmS = currentOp;
                m_FenceOpForPleL = currentOp;
                m_FenceOpForWgtS = currentOp;
            }
        }
    }
    catch (const NotSupportedException& e)
    {
        g_Logger.Error("Error: %s", e.what());
        return {};
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
                    // Pad the buffer ID for easy sorting of dumped file names
                    ss << "EthosNIntermediateBuffer_" << std::setfill('0') << std::setw(3) << b.second << std::setw(0);
                    ss << "_" << ToString(b.first->m_DataType);
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

    CompiledOpGraph result;
    result.m_EstimatedOpGraph = EstimateOpGraph(m_MergedOpGraph, m_Capabilities, EstimationOptions());

    // Create the compiled network using the updated BufferManager instance
    result.m_CompiledNetwork    = std::make_unique<CompiledNetworkImpl>(m_BufferManager.GetConstantDmaData(),
                                                                     m_BufferManager.GetConstantControlUnitData(),
                                                                     m_BufferManager.GetBuffers(), m_OperationIds);
    result.m_OpToAgentIdMapping = m_OpToAgentIdMapping;
    result.m_BufferIds          = m_DramBufToBufIdMapping;

    return result;
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
void CascadingCommandStreamGenerator::ProcessDmaOp(DmaOp* const ptrDmaOp)
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

            uint16_t inputBufferId = AddDramBufferAndCacheId(inputBuffer, ptrDmaOp);

            uint32_t inputDramBufferOffset = CommonUtils::GetDramOffset(
                inputBuffer->m_Format, inputBuffer->m_TensorShape, ptrDmaOp->m_Offset, m_Capabilities);

            AgentIdType ifmStreamerAgentId = AddIfmStreamerToCommandStream(
                ptrDmaOp, inputBufferId, inputBuffer, outputBuffer, dmaOp->m_TransferFormat, inputDramBufferOffset);

            if (m_FenceOpForIfmS != nullptr)
            {
                // Note that this is an overly pessimistic approach, as corruption would only happen in practice if the SRAM
                // addresses used overlap, which we do not bother checking. A future improvement would be to check this first.
                AddReadAfterWriteDependency(
                    AgentType::IFM_STREAMER, ifmStreamerAgentId,
                    m_CommandStreamAgents[this->m_OpToAgentIdMapping.at(m_FenceOpForIfmS)].data.type,
                    this->m_OpToAgentIdMapping.at(m_FenceOpForIfmS));
                m_FenceOpForIfmS = nullptr;
            }
        }
        else
        {
            // Weight Streamer Agent
            AgentIdType weightStreamerAgentId = AddWeightStreamerToCommandStream(static_cast<DmaOp*>(ptrDmaOp));

            if (m_FenceOpForWgtS != nullptr)
            {
                // Note that this is an overly pessimistic approach, as corruption would only happen in practice if the SRAM
                // addresses used overlap, which we do not bother checking. A future improvement would be to check this first.
                AddReadAfterWriteDependency(
                    AgentType::WGT_STREAMER, weightStreamerAgentId,
                    m_CommandStreamAgents[this->m_OpToAgentIdMapping.at(m_FenceOpForWgtS)].data.type,
                    this->m_OpToAgentIdMapping.at(m_FenceOpForWgtS));
                m_FenceOpForWgtS = nullptr;
            }
        }
    }
    else if (inputBuffer->m_Location == Location::Sram && outputBuffer->m_Location == Location::Dram)
    {
        assert(inputBuffer->m_Offset.has_value());
        assert(outputBuffer->m_BufferType.has_value());

        // Get the producer of the input buffer and the producing agent type
        Op* producerOp = m_MergedOpGraph.GetSingleProducer(inputBuffer);
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

        uint16_t outputBufferId = std::numeric_limits<uint16_t>::max();
        // Don't add buffers multiple times if they are used more than once
        if (m_DramBufToBufIdMapping.find(outputBuffer) == m_DramBufToBufIdMapping.end())
        {
            outputBufferId = static_cast<uint16_t>(
                m_BufferManager.AddDram(outputBuffer->m_BufferType.value(), outputBuffer->m_SizeInBytes));
            m_DramBufToBufIdMapping[outputBuffer] = outputBufferId;

            if (outputBuffer->m_BufferType.value() == BufferType::Output)
            {
                assert(outputBuffer->m_OperationId.has_value());
                assert(outputBuffer->m_ProducerOutputIndx);
                m_BufferManager.ChangeToOutput(outputBufferId, outputBuffer->m_OperationId.value(),
                                               outputBuffer->m_ProducerOutputIndx.value());
            }
        }
        else
        {
            outputBufferId = static_cast<uint16_t>(m_DramBufToBufIdMapping[outputBuffer]);
        }

        uint32_t outputDramBufferOffset = CommonUtils::GetDramOffset(
            outputBuffer->m_Format, outputBuffer->m_TensorShape, ptrDmaOp->m_Offset, m_Capabilities);

        // Ofm Streamer Agent
        AgentIdType ofmStreamerAgentId =
            AddOfmStreamerToCommandStream(ptrDmaOp, inputBuffer, outputBufferId, outputBuffer, outputDramBufferOffset);

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

    auto producerOp = m_MergedOpGraph.GetSingleProducer(inputBuffers[g_MceIfmBufferIndex]);
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

        if (m_FenceOpForPleL != nullptr)
        {
            // Note that this is an overly pessimistic approach, as corruption would only happen in practice if the SRAM
            // addresses used overlap, which we do not bother checking. A future improvement would be to check this first.
            AddReadAfterWriteDependency(
                AgentType::PLE_LOADER, pleLoaderAgentId,
                m_CommandStreamAgents[this->m_OpToAgentIdMapping.at(m_FenceOpForPleL)].data.type,
                this->m_OpToAgentIdMapping.at(m_FenceOpForPleL));
            m_FenceOpForPleL = nullptr;
        }
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
        m_OpToAgentIdMapping[m_MergedOpGraph.GetSingleProducer(inputBuffers[g_MceWeightBufferIndex])]);

    // Add 'Write After Read' dependency information to the IfmStreamer and WeightStreamer agents
    // Write After Read Dependency for [IfmStreamer][MceScheduler] or
    // Write After Read Dependency for [PleScheduler][MceScheduler]
    AddWriteAfterReadDependency(AgentType::MCE_SCHEDULER, mceSchedulerAgentId, producerAgentType,
                                m_OpToAgentIdMapping[producerOp]);
    // Write After Read Dependency for [WeightStreamer][MceScheduler]
    AddWriteAfterReadDependency(
        AgentType::MCE_SCHEDULER, mceSchedulerAgentId, AgentType::WGT_STREAMER,
        m_OpToAgentIdMapping[m_MergedOpGraph.GetSingleProducer(inputBuffers[g_MceWeightBufferIndex])]);

    // Add 'Schedule Time' dependency information to the IfmStreamer and WeightStreamer agents
    // Schedule Time Dependency for [IfmStreamer][MceScheduler] or
    // Schedule Time Dependency for [PleScheduler][MceScheduler]
    AddScheduleTimeDependency(AgentType::MCE_SCHEDULER, mceSchedulerAgentId, producerAgentType,
                              m_OpToAgentIdMapping[producerOp]);
    // Schedule Time Dependency for [WeightStreamer][MceScheduler]
    AddScheduleTimeDependency(
        AgentType::MCE_SCHEDULER, mceSchedulerAgentId, AgentType::WGT_STREAMER,
        m_OpToAgentIdMapping[m_MergedOpGraph.GetSingleProducer(inputBuffers[g_MceWeightBufferIndex])]);
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

    Op* input0Producer = m_MergedOpGraph.GetSingleProducer(inputBuffers[g_PleInputBuffer0Index]);
    Op* input1Producer = nullptr;
    if (inputBuffers.size() == 2)
    {
        input1Producer = m_MergedOpGraph.GetSingleProducer(inputBuffers[g_PleInputBuffer1Index]);
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
        AddReadAfterWriteDependency(AgentType::PLE_SCHEDULER, pleSchedulerAgentId, AgentType::IFM_STREAMER,
                                    m_OpToAgentIdMapping[input0Producer]);
        if (input1Producer != nullptr)
        {
            // Read After Write Dependency for [PleScheduler][IfmStreamer]
            AddReadAfterWriteDependency(AgentType::PLE_SCHEDULER, pleSchedulerAgentId, AgentType::IFM_STREAMER,
                                        m_OpToAgentIdMapping[input1Producer]);
        }

        if (loadKernel)
        {
            // Read After Write Dependency for [PleScheduler][PleLoader]
            AddReadAfterWriteDependency(
                AgentType::PLE_SCHEDULER, pleSchedulerAgentId, AgentType::PLE_LOADER,
                m_PleKernelToPleLoaderAgentIdMapping[static_cast<PleOp*>(ptrPleOp)->m_PleKernelId]);

            if (m_FenceOpForPleL != nullptr)
            {
                // Note that this is an overly pessimistic approach, as corruption would only happen in practice if the SRAM
                // addresses used overlap, which we do not bother checking. A future improvement would be to check this first.
                AddReadAfterWriteDependency(
                    AgentType::PLE_LOADER, pleLoaderAgentId,
                    m_CommandStreamAgents[this->m_OpToAgentIdMapping.at(m_FenceOpForPleL)].data.type,
                    this->m_OpToAgentIdMapping.at(m_FenceOpForPleL));
                m_FenceOpForPleL = nullptr;
            }
        }

        // Write After Read Dependency for [IfmStreamer][PleScheduler]
        AddWriteAfterReadDependency(AgentType::PLE_SCHEDULER, pleSchedulerAgentId, AgentType::IFM_STREAMER,
                                    m_OpToAgentIdMapping[input0Producer]);

        // Schedule Time Dependency for [IfmStreamer][PleScheduler]
        AddScheduleTimeDependency(AgentType::PLE_SCHEDULER, pleSchedulerAgentId, AgentType::IFM_STREAMER,
                                  m_OpToAgentIdMapping[input0Producer]);

        if (input1Producer != nullptr)
        {
            // Write After Read Dependency for [IfmStreamer][PleScheduler]
            AddWriteAfterReadDependency(AgentType::PLE_SCHEDULER, pleSchedulerAgentId, AgentType::IFM_STREAMER,
                                        m_OpToAgentIdMapping[input1Producer]);

            // Schedule Time Dependency for [IfmStreamer][PleScheduler]
            AddScheduleTimeDependency(AgentType::PLE_SCHEDULER, pleSchedulerAgentId, AgentType::IFM_STREAMER,
                                      m_OpToAgentIdMapping[input1Producer]);
        }

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
    }
    ETHOSN_UNUSED(outputBuffer);
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
                                                                           const CascadingBufferFormat transferFormat,
                                                                           const uint32_t inputDramBufferOffset)
{
    assert(IsObjectOfType<DmaOp>(ptrOp));
    assert(inputSramBuffer->m_Format == CascadingBufferFormat::NHWCB);

    IfmS ifmStreamerData = {};

    ifmStreamerData.fmData.dramOffset = inputDramBufferOffset;

    ifmStreamerData.fmData.bufferId = inputDramBufferId;

    StreamersUtils::SetBufferDataType(ifmStreamerData.fmData, transferFormat);
    ifmStreamerData.fmData.fcafInfo.signedActivation = (inputDramBuffer->m_DataType == DataType::INT8_QUANTIZED);
    ifmStreamerData.fmData.fcafInfo.zeroPoint =
        ethosn::utils::NumericCast<int16_t>(inputDramBuffer->m_QuantizationInfo.GetZeroPoint());

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
    if (ptrMceOp->m_Op == command_stream::MceOperation::FULLY_CONNECTED)
    {
        // Fully connected stripe shapes are always 8x8xC (for both default and edge stripes).
        // This is due to the reinterpretation that the hardware requires.
        const uint16_t w = ethosn::utils::NumericCast<uint16_t>(utils::GetWidth(m_Capabilities.GetBrickGroupShape()));
        const uint16_t h = ethosn::utils::NumericCast<uint16_t>(utils::GetHeight(m_Capabilities.GetBrickGroupShape()));
        mceSchedulerData.edgeStripeSize.ofmWidth  = w;
        mceSchedulerData.edgeStripeSize.ofmHeight = h;
        mceSchedulerData.dfltStripeSize.ofmWidth  = w;
        mceSchedulerData.dfltStripeSize.ofmHeight = h;
    }

    MceSUtils::SetMcesOfmChannelsStripeInfo(mceSchedulerData, outputBuffer->m_TensorShape,
                                            ptrMceOp->m_OutputStripeShape);
    MceSUtils::SetMcesIfmChannelsStripeInfo(mceSchedulerData, inputBuffer->m_TensorShape, inputBuffer->m_StripeShape);

    MceSUtils::SetStripeIdStrides(mceSchedulerData, outputBuffer->m_Order);

    mceSchedulerData.convStrideXy.x = ethosn::utils::NumericCast<uint8_t>(ptrMceOp->m_Stride.m_X);
    mceSchedulerData.convStrideXy.y = ethosn::utils::NumericCast<uint8_t>(ptrMceOp->m_Stride.m_Y);
    mceSchedulerData.ifmZeroPoint = ethosn::utils::NumericCast<int16_t>(inputBuffer->m_QuantizationInfo.GetZeroPoint());
    mceSchedulerData.isIfmSigned  = static_cast<uint8_t>(inputBuffer->m_DataType == DataType::INT8_QUANTIZED);
    mceSchedulerData.isOfmSigned  = static_cast<uint8_t>(outputBuffer->m_DataType == DataType::INT8_QUANTIZED);

    MceSUtils::setMcesAlgorithm(mceSchedulerData, ptrMceOp->m_Algo);

    mceSchedulerData.upsampleType = ptrMceOp->m_UpsampleType;

    const uint32_t outputBufferWidth  = utils::GetWidth(outputBuffer->m_TensorShape);
    const uint32_t outputBufferHeight = utils::GetHeight(outputBuffer->m_TensorShape);

    const bool isUpsample = mceSchedulerData.upsampleType != UpsampleType::OFF;
    if (isUpsample)
    {
        // As only 2x resize is supported, drop mode is only possible for odd output width/height.
        mceSchedulerData.upsampleEdgeMode.col =
            (outputBufferWidth & 1) ? UpsampleEdgeMode::DROP : UpsampleEdgeMode::GENERATE;
        mceSchedulerData.upsampleEdgeMode.row =
            (outputBufferHeight & 1) ? UpsampleEdgeMode::DROP : UpsampleEdgeMode::GENERATE;
    }

    // Calculate IFM Delta Edge
    auto Upscale = [isUpsample](uint32_t dim, UpsampleEdgeMode mode) {
        return isUpsample ? dim * 2 - (mode == UpsampleEdgeMode::DROP ? 1 : 0) : dim;
    };
    const uint32_t inputBufferWidth    = utils::GetWidth(inputBuffer->m_TensorShape);
    const uint32_t inputBufferHeight   = utils::GetHeight(inputBuffer->m_TensorShape);
    const uint32_t upscaledInputWidth  = Upscale(inputBufferWidth, mceSchedulerData.upsampleEdgeMode.col);
    const uint32_t upscaledInputHeight = Upscale(inputBufferHeight, mceSchedulerData.upsampleEdgeMode.row);
    const int8_t ifmDeltaEdgeWidth     = static_cast<int8_t>(upscaledInputWidth - outputBufferWidth);
    const int8_t ifmDeltaEdgeHeight    = static_cast<int8_t>(upscaledInputHeight - outputBufferHeight);

    if (ptrMceOp->m_Stride.m_X == 1 && ptrMceOp->m_Stride.m_Y == 1)
    {
        for (int i = 0; i < 4; i++)
        {
            mceSchedulerData.filterShape[i].height =
                ethosn::utils::NumericCast<uint8_t>(weightBuffer->m_TensorShape[0]);
            mceSchedulerData.filterShape[i].width = ethosn::utils::NumericCast<uint8_t>(weightBuffer->m_TensorShape[1]);

            if (mceSchedulerData.mceOpMode != MceOperation::FULLY_CONNECTED)
            {
                mceSchedulerData.ifmDeltaDefault[i].height = ethosn::utils::NumericCast<int8_t>(
                    (mceSchedulerData.filterShape[i].height / 2) + inputBuffer->m_PackedBoundaryThickness.bottom);
                mceSchedulerData.ifmDeltaDefault[i].width = ethosn::utils::NumericCast<int8_t>(
                    (mceSchedulerData.filterShape[i].width / 2) + inputBuffer->m_PackedBoundaryThickness.right);

                if (isUpsample)
                {
                    mceSchedulerData.ifmDeltaDefault[i].height =
                        std::max(static_cast<int8_t>(2), mceSchedulerData.ifmDeltaDefault[i].height);
                    mceSchedulerData.ifmDeltaDefault[i].width =
                        std::max(static_cast<int8_t>(2), mceSchedulerData.ifmDeltaDefault[i].width);
                }

                mceSchedulerData.ifmDeltaEdge[i].height = ifmDeltaEdgeHeight;
                mceSchedulerData.ifmDeltaEdge[i].width  = ifmDeltaEdgeWidth;

                mceSchedulerData.padding[i].left = ethosn::utils::NumericCast<uint8_t>(ptrMceOp->m_PadLeft);
                mceSchedulerData.padding[i].top  = ethosn::utils::NumericCast<uint8_t>(ptrMceOp->m_PadTop);
            }
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
    pleLoaderData.sramAddr    = ethosn::utils::NumericCast<uint32_t>(ptrPleOp->m_Offset.value());

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
    auto pleOpProducer = m_MergedOpGraph.GetSingleProducer(inputBuffer0);
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

    pleS.pleKernelSramAddr = ethosn::utils::NumericCast<uint32_t>(ptrPleOp->m_Offset.value());

    pleS.pleKernelId = ptrPleOp->m_PleKernelId;

    if (pleS.inputMode == PleInputMode::SRAM)
    {
        CommonUtils::SetTileInfoForBuffer(m_Capabilities, pleS.ifmTile0, inputBuffer0);
    }

    pleS.ifmInfo0.zeroPoint  = ethosn::utils::NumericCast<int16_t>(inputBuffer0->m_QuantizationInfo.GetZeroPoint());
    pleS.ifmInfo0.multiplier = ptrPleOp->m_Input0Multiplier;
    pleS.ifmInfo0.shift      = ptrPleOp->m_Input0Shift;

    // Note these are set even if there is only 1 input, because some PLE kernels (e.g. LeakyRelu)
    // use these to pass extra information
    pleS.ifmInfo1.multiplier = ptrPleOp->m_Input1Multiplier;
    pleS.ifmInfo1.shift      = ptrPleOp->m_Input1Shift;

    if (inputBuffers.size() == 2)
    {
        Buffer* inputBuffer1 = inputBuffers[g_PleInputBuffer1Index];
        CommonUtils::SetTileInfoForBuffer(m_Capabilities, pleS.ifmTile1, inputBuffer1);

        pleS.ifmInfo1.zeroPoint = ethosn::utils::NumericCast<int16_t>(inputBuffer1->m_QuantizationInfo.GetZeroPoint());
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
                                                                           const Buffer* const outputDramBuffer,
                                                                           const uint32_t outputDramBufferOffset)
{
    assert(IsObjectOfType<DmaOp>(ptrOp));
    assert(outputSramBuffer->m_Format == CascadingBufferFormat::NHWCB);

    OfmS ofmStreamerData = {};

    ofmStreamerData.fmData.dramOffset = outputDramBufferOffset;

    ofmStreamerData.fmData.bufferId = outputDramBufferId;

    StreamersUtils::SetBufferDataType(ofmStreamerData.fmData, outputDramBuffer->m_Format);

    ofmStreamerData.fmData.fcafInfo.signedActivation = (outputDramBuffer->m_DataType == DataType::INT8_QUANTIZED);
    ofmStreamerData.fmData.fcafInfo.zeroPoint =
        ethosn::utils::NumericCast<int16_t>(outputDramBuffer->m_QuantizationInfo.GetZeroPoint());

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
    if (newDependency.relativeAgentId != 0)
    {
        DependencyUtils::AddDependency(m_CommandStreamAgents[consumerAgentId].info.readDependencies, newDependency);
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

    Dependency newDependency      = {};
    newDependency.relativeAgentId = static_cast<RelativeAgentIdType>(relativeAgentId);
    FillProducerAgentDependency(newDependency, consumerAgentType, consumerAgentId, producerAgentType, producerAgentId,
                                DependencyType::Write);
    if (newDependency.relativeAgentId != 0)
    {
        DependencyUtils::AddDependency(m_CommandStreamAgents[producerAgentId].info.writeDependencies, newDependency);
    }
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
    FillProducerAgentDependency(newDependency, consumerAgentType, consumerAgentId, producerAgentType, producerAgentId,
                                DependencyType::Schedule);
    if (newDependency.relativeAgentId != 0)
    {
        DependencyUtils::AddDependency(m_CommandStreamAgents[producerAgentId].info.scheduleDependencies, newDependency);
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
            // Sram Overlap Dependency for [WeightStreamer][PleScheduler]
            else if (producerAgentType == AgentType::PLE_SCHEDULER)
            {
                // The WgtS needs to wait for the prior PleS in the same section, for example in a strategy 1 cascade,
                // because these weights shouldn't be loaded until the weights from the previous layer are finished with.
                // The WgtS should wait until the PleS has completely finished.
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
                if (!(consumerAgent.data.mce.isPackedBoundaryX && consumerAgent.data.mce.isPackedBoundaryY))
                {
                    bool needsBoundaryBeforeX = consumerAgent.data.mce.filterShape[0].width >= 2 ||
                                                consumerAgent.data.mce.upsampleType != UpsampleType::OFF;
                    bool needsBoundaryAfterX = consumerAgent.data.mce.filterShape[0].width >= 3 ||
                                               consumerAgent.data.mce.upsampleType != UpsampleType::OFF;
                    bool needsBoundaryBeforeY = consumerAgent.data.mce.filterShape[0].height >= 2 ||
                                                consumerAgent.data.mce.upsampleType != UpsampleType::OFF;
                    bool needsBoundaryAfterY = consumerAgent.data.mce.filterShape[0].height >= 3 ||
                                               consumerAgent.data.mce.upsampleType != UpsampleType::OFF;
                    consumerAgentDependency.boundary =
                        needsBoundaryBeforeX || needsBoundaryAfterX || needsBoundaryBeforeY || needsBoundaryAfterY;
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

                bool needsBoundaryBeforeX = producerAgent.data.pleS.numStripes.width > 1 &&
                                            (consumerAgent.data.mce.filterShape[0].width >= 2 ||
                                             consumerAgent.data.mce.upsampleType != UpsampleType::OFF);
                bool needsBoundaryAfterX = producerAgent.data.pleS.numStripes.width > 1 &&
                                           (consumerAgent.data.mce.filterShape[0].width >= 3 ||
                                            consumerAgent.data.mce.upsampleType != UpsampleType::OFF);
                bool needsBoundaryBeforeY = producerAgent.data.pleS.numStripes.height > 1 &&
                                            (consumerAgent.data.mce.filterShape[0].height >= 2 ||
                                             consumerAgent.data.mce.upsampleType != UpsampleType::OFF);
                bool needsBoundaryAfterY = producerAgent.data.pleS.numStripes.height > 1 &&
                                           (consumerAgent.data.mce.filterShape[0].height >= 3 ||
                                            consumerAgent.data.mce.upsampleType != UpsampleType::OFF);
                consumerAgentDependency.boundary =
                    needsBoundaryBeforeX || needsBoundaryAfterX || needsBoundaryBeforeY || needsBoundaryAfterY;
            }
            else
            {
                assert(false);
            }
            break;
        }

        case AgentType::PLE_LOADER:
        {
            // Sram Overlap Dependency for [PleLoader][PleScheduler]
            if (producerAgentType == AgentType::PLE_SCHEDULER)
            {
                consumerAgentDependency.outerRatio.other = producerAgent.info.numStripesTotal;
                consumerAgentDependency.outerRatio.self  = consumerAgent.info.numStripesTotal;

                consumerAgentDependency.innerRatio.other = producerAgent.info.numStripesTotal;
                consumerAgentDependency.innerRatio.self  = 1;

                consumerAgentDependency.boundary = 0;
            }
            // Sram Overlap Dependency for [PleLoader][OfmStreamer]
            else if (producerAgentType == AgentType::OFM_STREAMER)
            {
                consumerAgentDependency.outerRatio.other = producerAgent.info.numStripesTotal;
                consumerAgentDependency.outerRatio.self  = consumerAgent.info.numStripesTotal;

                consumerAgentDependency.innerRatio.other = producerAgent.info.numStripesTotal;
                consumerAgentDependency.innerRatio.self  = 1;

                consumerAgentDependency.boundary = 0;
            }
            else
            {
                assert(false);
            }
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
                    ethosn::utils::NumericCast<uint16_t>(consumerAgent.data.ofm.fmData.dfltStripeSize.height /
                                                         producerAgent.data.pleS.dfltStripeSize.height);
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
    const AgentIdType producerAgentId,
    DependencyType dependencyType)
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
                if (!(consumerAgent.data.mce.isPackedBoundaryX && consumerAgent.data.mce.isPackedBoundaryY))
                {
                    bool needsBoundaryBeforeX = consumerAgent.data.mce.filterShape[0].width >= 2 ||
                                                consumerAgent.data.mce.upsampleType != UpsampleType::OFF;
                    bool needsBoundaryAfterX = consumerAgent.data.mce.filterShape[0].width >= 3 ||
                                               consumerAgent.data.mce.upsampleType != UpsampleType::OFF;
                    bool needsBoundaryBeforeY = consumerAgent.data.mce.filterShape[0].height >= 2 ||
                                                consumerAgent.data.mce.upsampleType != UpsampleType::OFF;
                    bool needsBoundaryAfterY = consumerAgent.data.mce.filterShape[0].height >= 3 ||
                                               consumerAgent.data.mce.upsampleType != UpsampleType::OFF;
                    producerAgentDependency.boundary =
                        needsBoundaryBeforeX || needsBoundaryAfterX || needsBoundaryBeforeY || needsBoundaryAfterY;
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
                if (dependencyType == DependencyType::Write && consumerAgent.info.numStripesTotal == 1)
                {
                    // For the case where we have the PLE stripes split in height but being written into an output buffer
                    // which is the full tensor, we have only one stripe in the following MceS. We don't want a write dependency
                    // from the PleS onto this MceS, otherwise it will stall.
                    producerAgentDependency.relativeAgentId = 0;
                    break;
                }

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

                bool needsBoundaryBeforeX = producerAgent.data.pleS.numStripes.width > 1 &&
                                            (consumerAgent.data.mce.filterShape[0].width >= 2 ||
                                             consumerAgent.data.mce.upsampleType != UpsampleType::OFF);
                bool needsBoundaryAfterX = producerAgent.data.pleS.numStripes.width > 1 &&
                                           (consumerAgent.data.mce.filterShape[0].width >= 3 ||
                                            consumerAgent.data.mce.upsampleType != UpsampleType::OFF);
                bool needsBoundaryBeforeY = producerAgent.data.pleS.numStripes.height > 1 &&
                                            (consumerAgent.data.mce.filterShape[0].height >= 2 ||
                                             consumerAgent.data.mce.upsampleType != UpsampleType::OFF);
                bool needsBoundaryAfterY = producerAgent.data.pleS.numStripes.height > 1 &&
                                           (consumerAgent.data.mce.filterShape[0].height >= 3 ||
                                            consumerAgent.data.mce.upsampleType != UpsampleType::OFF);
                producerAgentDependency.boundary =
                    needsBoundaryBeforeX || needsBoundaryAfterX || needsBoundaryBeforeY || needsBoundaryAfterY;
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
                    ethosn::utils::NumericCast<uint16_t>(producerAgent.data.pleS.dfltStripeSize.height /
                                                         consumerAgent.data.ofm.fmData.dfltStripeSize.height);
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

    for (Op* producer : graph.GetProducers(b))
    {
        assert(producer != nullptr);
        size_t earliestOpIdxThisProducer = std::numeric_limits<size_t>::max();
        for (Buffer* input : graph.GetInputs(producer))
        {
            if (input->m_Location != Location::Dram)
            {
                earliestOpIdxThisProducer = std::min(earliestOpIdxThisProducer, WalkGraphUp(graph, input));
            }
        }

        if (earliestOpIdxThisProducer == std::numeric_limits<size_t>::max())
        {
            // This producer has all inputs in DRAM, so is the earliest along this branch
            earliestOpIdxThisProducer = utils::FindIndex(graph.GetOps(), producer).second;
        }

        result = std::min(result, earliestOpIdxThisProducer);
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

        size_t latestOpIdxThisConsumer;
        if (output->m_Location == Location::Dram)
        {
            latestOpIdxThisConsumer = utils::FindIndex(graph.GetOps(), consumer.first).second;
        }
        else
        {
            latestOpIdxThisConsumer = WalkGraphDown(graph, output);
        }
        result = std::max(result, latestOpIdxThisConsumer);
    }

    return result;
}

}    // namespace

/// Private function to add the lifetime information of the intermediate DRAM buffers
/// Determines the start and end of the lifetime of the given (intermediate DRAM) buffer.
/// The approach is to walk the graph backwards from the buffer to find the previous time
/// there was a DRAM buffer, at which point we know the target buffer would not be needed,
/// and we do the same walking forwards, to know the point at which the target buffer
/// will be finished with. When there are branches, we go along each to find the
/// furthest away usage. This can be thought of as a "flood fill" to find the set of Ops
/// in the section before/after the target buffer, and then finding the min/max agent ID
/// of those Ops.
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
