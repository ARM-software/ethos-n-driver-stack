//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "EthosNPreCompiledWorkload.hpp"

#include "EthosNBackend.hpp"
#include "EthosNTensorHandle.hpp"
#include "EthosNWorkloadUtils.hpp"
#include "LabelsAndEventClasses.hpp"

#include <armnn/ArmNN.hpp>
#include <armnn/utility/Assert.hpp>
#include <common/include/Threads.hpp>
#include <ethosn_driver_library/Network.hpp>
#include <ethosn_driver_library/ProcMemAllocator.hpp>
#include <ethosn_support_library/Support.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>
#if defined(__unix__)
#include <unistd.h>
#elif defined(_MSC_VER)
#include <io.h>
#define O_CLOEXEC 0
#endif

using namespace arm::pipe;

namespace armnn
{
namespace
{

void SendProfilingEvents()
{
    auto context        = EthosNBackendProfilingService::Instance().GetContext();
    auto timelineEvents = ethosn::driver_library::profiling::ReportNewProfilingData();
    auto sender         = context->GetSendTimelinePacket();
    auto& map           = context->GetIdToEntityGuids();
    auto& guidGenerator = context->GetGuidGenerator();

    // Currently Arm NN doesn't call EnableTimelineReporting so always report timeline events
    for (auto event : timelineEvents)
    {
        using namespace ethosn::driver_library::profiling;
        using namespace armnn::profiling;
        // Filter for timeline events.
        if (event.m_Type != ProfilingEntry::Type::TimelineEventStart &&
            event.m_Type != ProfilingEntry::Type::TimelineEventEnd &&
            event.m_Type != ProfilingEntry::Type::TimelineEventInstant)
        {
            continue;
        }
        auto guidIt = map.find({ event.m_Id });

        // If we don't find the guid in the map, then assume it is the first time we send one for this entity
        // An example of an entity is a single buffer.
        // An entity can have multiple events associated with it. e.g. buffer lifetime start and buffer lifetime end.
        if (guidIt == map.end())
        {
            auto entityGuid = guidGenerator.NextGuid();
            sender->SendTimelineEntityBinaryPacket(entityGuid);
            map.insert({ { event.m_Id }, entityGuid });
            // Register a label with with the category and id e.g. Buffer 0
            // Note: This Id is a global Id so "Buffer 2" may not be the third buffer.
            std::string label = "EthosN " + std::string(MetadataCategoryToCString(event.m_MetadataCategory)) + " " +
                                std::to_string(event.m_Id);
            auto labelGuid = guidGenerator.GenerateStaticId(label);
            sender->SendTimelineLabelBinaryPacket(labelGuid, label);
            auto relationshipGuid = guidGenerator.NextGuid();
            sender->SendTimelineRelationshipBinaryPacket(ProfilingRelationshipType::LabelLink, relationshipGuid,
                                                         entityGuid, labelGuid, LabelsAndEventClasses::NAME_GUID);
        }
        auto entityGuid = map.at({ event.m_Id });
        auto eventGuid  = guidGenerator.NextGuid();
        auto timeInNanoSeconds =
            std::chrono::duration_cast<std::chrono::nanoseconds>(event.m_Timestamp.time_since_epoch()).count();
        sender->SendTimelineEventBinaryPacket(static_cast<uint64_t>(timeInNanoSeconds), GetCurrentThreadId(),
                                              eventGuid);

        auto executionLinkId = IProfilingService::GetNextGuid();

        // If we are sending Start and End timeline events then we add a link to the Start/End of Life Event Classes.
        if (event.m_Type == ProfilingEntry::Type::TimelineEventStart)
        {
            sender->SendTimelineRelationshipBinaryPacket(ProfilingRelationshipType::ExecutionLink, executionLinkId,
                                                         entityGuid, eventGuid,
                                                         LabelsAndEventClasses::ARMNN_PROFILING_SOL_EVENT_CLASS);
        }
        if (event.m_Type == ProfilingEntry::Type::TimelineEventEnd)
        {
            sender->SendTimelineRelationshipBinaryPacket(ProfilingRelationshipType::ExecutionLink, executionLinkId,
                                                         entityGuid, eventGuid,
                                                         LabelsAndEventClasses::ARMNN_PROFILING_EOL_EVENT_CLASS);
        }
        sender->Commit();
    }
}

}    // anonymous namespace

void EthosNPreCompiledWorkload::Init(const EthosNPreCompiledObject::Network& network, const std::string& deviceId)
{
    const bool kernelVerified =
        deviceId.empty() ? ethosn::driver_library::VerifyKernel() : ethosn::driver_library::VerifyKernel(deviceId);
    if (!kernelVerified)
    {
        throw RuntimeException("Kernel version is not supported");
    }

    auto& procMemAllocator   = armnn::EthosNBackendAllocatorService::GetInstance().GetProcMemAllocator(deviceId);
    auto intermediateBufSize = m_PreCompiledObject->GetIntermediateBufferSize();
    ethosn::driver_library::IntermediateBufferReq req;

    if (intermediateBufSize > 0)
    {
        if (m_InternalAllocator != nullptr)
        {
            if (procMemAllocator.GetProtected())
            {
                if (m_InternalAllocator->GetMemorySourceType() != armnn::MemorySource::DmaBufProtected)
                {
                    throw RuntimeException(
                        "Backend configured for Protected requires a Custom Allocator supporting DmaBufProtected");
                }
            }
            else

            {
                if (m_InternalAllocator->GetMemorySourceType() != armnn::MemorySource::DmaBuf)
                {
                    throw RuntimeException(
                        "Backend configured for Non-protected requires a Custom Allocator supporting DmaBuf");
                }
            }

            void* mem_handle = m_InternalAllocator->allocate(intermediateBufSize, 0);
            if (mem_handle == nullptr)
            {
                throw NullPointerException("Failed to allocate memory for intermediate buffers");
            }
            req.type  = ethosn::driver_library::MemType::IMPORT;
            req.fd    = *static_cast<uint32_t*>(mem_handle);
            req.flags = O_RDWR | O_CLOEXEC;
        }
        else
        {
            if (procMemAllocator.GetProtected())
            {
                throw RuntimeException("Backend configured for Protected requires a Custom Allocator");
            }

            req.type = ethosn::driver_library::MemType::ALLOCATE;
        }
    }
    else
    {
        req.type = ethosn::driver_library::MemType::NONE;
    }

    m_Network = std::make_unique<ethosn::driver_library::Network>(procMemAllocator.CreateNetwork(
        network.m_SerializedCompiledNetwork.data(), network.m_SerializedCompiledNetwork.size(), req));

    m_Network->SetDebugName(("Subgraph" + std::to_string(m_PreCompiledObject->GetSubgraphIndex())).c_str());
}

EthosNPreCompiledWorkload::EthosNPreCompiledWorkload(const PreCompiledQueueDescriptor& descriptor,
                                                     const WorkloadInfo& info,
                                                     const std::string& deviceId,
                                                     std::shared_ptr<armnn::ICustomAllocator> customAllocator)
    : BaseWorkload<PreCompiledQueueDescriptor>(descriptor, info)
    , m_PreCompiledObject(static_cast<const EthosNPreCompiledObject*>(descriptor.m_PreCompiledObject))
    , m_InternalAllocator(std::move(customAllocator))
{
    // Check that the workload is holding a pointer to a valid pre-compiled object
    if (m_PreCompiledObject == nullptr)
    {
        throw InvalidArgumentException("EthosNPreCompiledWorkload requires a valid pre-compiled object");
    }

    if (!m_PreCompiledObject->IsSkipInference())
    {
        Init(m_PreCompiledObject->GetNetwork().value(), deviceId);
    }
}

void EthosNPreCompiledWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_ETHOSN("EthosNPreCompiledWorkload_Execute");

    if (m_PreCompiledObject->IsSkipInference())
    {
        return;
    }

    uint32_t numInputBuffers  = static_cast<uint32_t>(m_Data.m_Inputs.size());
    uint32_t numOutputBuffers = static_cast<uint32_t>(m_Data.m_Outputs.size());

    std::vector<ethosn::driver_library::Buffer*> inputBuffers(numInputBuffers);
    std::vector<ethosn::driver_library::Buffer*> outputBuffers(numOutputBuffers);

    // Fill inputBuffers from the input tensor handles, assuming that the order
    // is the same from the Arm NN inputs slots to the Ethos-N inputs slots.
    for (uint32_t inputSlotIdx = 0; inputSlotIdx < numInputBuffers; ++inputSlotIdx)
    {
        auto&& inputTensorHandle   = m_Data.m_Inputs[inputSlotIdx];
        inputBuffers[inputSlotIdx] = &(static_cast<EthosNBaseTensorHandle*>(inputTensorHandle)->GetBuffer());
    }
    // Fill outputBuffers from the output tensor handles, assuming that the order
    // is the same from the Arm NN output slots to the Ethos-N output slots.
    for (uint32_t outputSlotIdx = 0; outputSlotIdx < numOutputBuffers; ++outputSlotIdx)
    {
        auto&& outputTensorHandle    = m_Data.m_Outputs[outputSlotIdx];
        outputBuffers[outputSlotIdx] = &(static_cast<EthosNBaseTensorHandle*>(outputTensorHandle)->GetBuffer());
    }

    ARMNN_LOG(debug) << "Ethos-N ScheduleInference Subgraph " << m_PreCompiledObject->GetSubgraphIndex();
    std::unique_ptr<ethosn::driver_library::Inference> inference;
    {
        ARMNN_SCOPED_PROFILING_EVENT_ETHOSN("EthosNPreCompiledWorkload_ScheduleInference");

        inference = std::unique_ptr<ethosn::driver_library::Inference>(
            m_Network->ScheduleInference(inputBuffers.data(), numInputBuffers, outputBuffers.data(), numOutputBuffers));
    }

    ethosn::driver_library::InferenceResult result;
    {
        ARMNN_SCOPED_PROFILING_EVENT_ETHOSN("EthosNPreCompiledWorkload_Wait");
        uint32_t timeoutMs = m_PreCompiledObject->GetInferenceTimeout() < 0
                                 ? UINT32_MAX
                                 : static_cast<uint32_t>(m_PreCompiledObject->GetInferenceTimeout()) * 1000;
        result = inference->Wait(timeoutMs);
    }

    ARMNN_LOG(info) << "Ethos-N cycle count: " << inference->GetCycleCount();
    if (EthosNBackendProfilingService::Instance().IsProfilingEnabled())
    {
        SendProfilingEvents();
    }
    switch (result)
    {
        case ethosn::driver_library::InferenceResult::Scheduled:
            // Intentional fallthrough
        case ethosn::driver_library::InferenceResult::Running:
            throw RuntimeException("Ethos-N inference timed out after " +
                                   std::to_string(m_PreCompiledObject->GetInferenceTimeout()) + "s");
        case ethosn::driver_library::InferenceResult::Completed:
            // Yay!
            break;
        case ethosn::driver_library::InferenceResult::Error:
            throw RuntimeException("Ethos-N inference error");
    }
}

}    //namespace armnn
