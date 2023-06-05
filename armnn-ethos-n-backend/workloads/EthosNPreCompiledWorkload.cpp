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
#include <poll.h>
#include <unistd.h>
#elif defined(_MSC_VER)
#include <io.h>
#define O_CLOEXEC 0
#endif

using namespace arm::pipe;

// Error codes for the WaitStatus class
enum class WaitErrorCode
{
    Success = 0,
    Timeout = 1,
    Error   = 2
};

// Status class for WaitForInference
class WaitStatus
{
public:
    // Default constructor
    WaitStatus()
        : m_ErrorCode(WaitErrorCode::Success)
        , m_ErrorDescription("")
    {}

    // Standard constructor
    explicit WaitStatus(WaitErrorCode errorCode, std::string errorDescription = "")
        : m_ErrorCode(errorCode)
        , m_ErrorDescription(std::move(errorDescription))
    {}

    // Allow instances of this class to be copy constructed
    WaitStatus(const WaitStatus&) = default;

    // Allow instances of this class to be move constructed
    WaitStatus(WaitStatus&&) = default;

    // Allow instances of this class to be copy assigned
    WaitStatus& operator=(const WaitStatus&) = default;

    // Allow instances of this class to be move assigned
    WaitStatus& operator=(WaitStatus&&) = default;

    // Explicit bool conversion operator
    explicit operator bool() const noexcept
    {
        return m_ErrorCode == WaitErrorCode::Success;
    }

    // Gets the error code
    WaitErrorCode GetErrorCode() const
    {
        return m_ErrorCode;
    }

    // Gets the error description (if any)
    std::string GetErrorDescription() const
    {
        return m_ErrorDescription;
    }

private:
    WaitErrorCode m_ErrorCode;
    std::string m_ErrorDescription;
};

namespace armnn
{
namespace
{

// Wait for an inference to complete
WaitStatus WaitForInference(int fd, int timeout)
{
    // Default to success as for platforms other than Linux we assume we are running on the model and therefore
    // there is no need to wait.
    WaitStatus result;

#if defined(__unix__)
    // Wait for the inference to complete
    struct pollfd fds;
    memset(&fds, 0, sizeof(fds));
    fds.fd     = fd;
    fds.events = POLLIN;    // Wait for any available input

    const int msPerSeconds = 1000;
    int pollResult         = poll(&fds, 1, timeout * msPerSeconds);
    // Stash errno immediately after poll call
    int pollErrorCode = errno;
    if (pollResult > 0)
    {
        ethosn::driver_library::InferenceResult ethosNResult;
        if (read(fd, &ethosNResult, sizeof(ethosNResult)) != static_cast<ssize_t>(sizeof(ethosNResult)))
        {
            result = WaitStatus(WaitErrorCode::Error,
                                "Failed to read inference result status (" + std::string(strerror(errno)) + ")");
        }
        else if (ethosNResult == ethosn::driver_library::InferenceResult::Completed)
        {
            result = WaitStatus(WaitErrorCode::Success);
        }
        else
        {
            result = WaitStatus(WaitErrorCode::Error,
                                "Inference failed with status " + std::to_string(static_cast<uint32_t>(ethosNResult)));
        }
    }
    else if (pollResult == 0)
    {
        result = WaitStatus(WaitErrorCode::Timeout, "Timed out while waiting for the inference to complete");
    }
    else
    {
        // pollResult < 0
        result = WaitStatus(WaitErrorCode::Error, "Error while waiting for the inference to complete (" +
                                                      std::string(strerror(pollErrorCode)) + ")");
    }
#endif

    return result;
}

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
    const std::unique_ptr<ethosn::driver_library::Inference> inference(
        m_Network->ScheduleInference(inputBuffers.data(), numInputBuffers, outputBuffers.data(), numOutputBuffers));

    WaitStatus result = WaitForInference(inference->GetFileDescriptor(), m_PreCompiledObject->GetInferenceTimeout());

    if (EthosNBackendProfilingService::Instance().IsProfilingEnabled())
    {
        SendProfilingEvents();
    }
    switch (result.GetErrorCode())
    {
        case WaitErrorCode::Success:
            break;
        case WaitErrorCode::Timeout:
        case WaitErrorCode::Error:
        default:
            throw RuntimeException("An error has occurred waiting for the inference of a pre-compiled object: " +
                                   result.GetErrorDescription());
    }
}

}    //namespace armnn
