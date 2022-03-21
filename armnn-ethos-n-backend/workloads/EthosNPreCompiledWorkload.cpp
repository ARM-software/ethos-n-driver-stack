//
// Copyright © 2018-2022 Arm Limited.
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
        , m_ErrorDescription(errorDescription)
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

void EthosNPreCompiledWorkload::Init(const PreCompiledDescriptor& descriptor,
                                     const EthosNPreCompiledObject::Network& network,
                                     const std::string& deviceId)
{
    const bool kernelVerified =
        deviceId.empty() ? ethosn::driver_library::VerifyKernel() : ethosn::driver_library::VerifyKernel(deviceId);
    if (!kernelVerified)
    {
        throw RuntimeException("Kernel version is not supported");
    }

    // Set up the buffers in the PreCompiledLayer::CreateWorkload() method, pass them in PreCompiledQueueDescriptor
    unsigned int numInputBuffers = descriptor.m_NumInputSlots;
    m_InputBuffers.resize(numInputBuffers);

    // Fill m_InputBuffers from the input tensor handles, assuming that the order
    // is the same from the Arm NN inputs slots to the Ethos-N inputs slots.
    for (unsigned int inputSlotIdx = 0; inputSlotIdx < numInputBuffers; ++inputSlotIdx)
    {
        m_InputBuffers[inputSlotIdx] = &(static_cast<EthosNTensorHandle*>(m_Data.m_Inputs[inputSlotIdx])->GetBuffer());
    }

    // Set up the buffers in the PreCompiledLayer::CreateWorkload() method, pass them in PreCompiledQueueDescriptor
    unsigned int numOutputBuffers = descriptor.m_NumOutputSlots;
    m_OutputBuffers.resize(numOutputBuffers);

    // Fill m_OutputBuffers from the output tensor handles, assuming that the order
    // is the same from the Arm NN output slots to the Ethos-N output slots.
    for (unsigned int outputSlotIdx = 0; outputSlotIdx < numOutputBuffers; ++outputSlotIdx)
    {
        m_OutputBuffers[outputSlotIdx] =
            &(static_cast<EthosNTensorHandle*>(m_Data.m_Outputs[outputSlotIdx])->GetBuffer());
    }

    if (deviceId.empty())
    {
        m_Network = std::make_unique<ethosn::driver_library::Network>(network.m_SerializedCompiledNetwork.data(),
                                                                      network.m_SerializedCompiledNetwork.size());
    }
    else
    {
        m_Network = std::make_unique<ethosn::driver_library::Network>(
            network.m_SerializedCompiledNetwork.data(), network.m_SerializedCompiledNetwork.size(), deviceId);
    }

    m_Network->SetDebugName(std::to_string(m_Guid).c_str());
}

EthosNPreCompiledWorkload::EthosNPreCompiledWorkload(const PreCompiledQueueDescriptor& descriptor,
                                                     const WorkloadInfo& info,
                                                     const std::string& deviceId)
    : BaseWorkload<PreCompiledQueueDescriptor>(descriptor, info)
    , m_PreCompiledObject(static_cast<const EthosNPreCompiledObject*>(descriptor.m_PreCompiledObject))
{
    // Check that the workload is holding a pointer to a valid pre-compiled object
    if (m_PreCompiledObject == nullptr)
    {
        throw InvalidArgumentException("EthosNPreCompiledWorkload requires a valid pre-compiled object");
    }

    if (!m_PreCompiledObject->IsPerfEstimationOnly())
    {
        Init(descriptor.m_Parameters, *m_PreCompiledObject->GetNetwork(), deviceId);
    }
}

void EthosNPreCompiledWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_ETHOSN("EthosNPreCompiledWorkload_Execute");

    if (m_PreCompiledObject->IsPerfEstimationOnly())
    {
        SavePerformanceJson();
    }
    else
    {
        uint32_t numInputBuffers  = static_cast<uint32_t>(m_InputBuffers.size());
        uint32_t numOutputBuffers = static_cast<uint32_t>(m_OutputBuffers.size());

        const std::unique_ptr<ethosn::driver_library::Inference> inference(m_Network->ScheduleInference(
            m_InputBuffers.data(), numInputBuffers, m_OutputBuffers.data(), numOutputBuffers));

        WaitStatus result = WaitForInference(inference->GetFileDescriptor(), 60);

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
}

namespace
{

template <typename T>
struct QuotedT
{
    explicit constexpr QuotedT(const T& value)
        : m_Value(value)
    {}

    const T& m_Value;
};

template <typename T>
QuotedT<T> Quoted(const T& value)
{
    return QuotedT<T>(value);
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const QuotedT<T>& field)
{
    return os << '"' << field.m_Value << '"';
}

template <typename T>
struct JsonFieldT
{
    explicit constexpr JsonFieldT(const T& value)
        : m_Value(value)
    {}

    const T& m_Value;
};

template <typename T>
JsonFieldT<T> JsonField(const T& value)
{
    return JsonFieldT<T>(value);
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const JsonFieldT<T>& field)
{
    return os << Quoted(field.m_Value) << ':';
}

struct Indent
{
    explicit constexpr Indent(const size_t depth)
        : m_Depth(depth)
    {}

    constexpr operator size_t&()
    {
        return m_Depth;
    }

    constexpr operator size_t() const
    {
        return m_Depth;
    }

    size_t m_Depth;
};

std::ostream& operator<<(std::ostream& os, const Indent& indent)
{
    for (size_t i = 0; i < indent; ++i)
    {
        os << '\t';
    }

    return os;
}

std::ostream& operator<<(std::ostream& os, const ethosn::support_library::EthosNVariant variant)
{
    switch (variant)
    {
        case ethosn::support_library::EthosNVariant::ETHOS_N78_1TOPS_2PLE_RATIO:
            os << Quoted("Ethos-N78_1TOPS_2PLE_RATIO");
            break;
        case ethosn::support_library::EthosNVariant::ETHOS_N78_1TOPS_4PLE_RATIO:
            os << Quoted("Ethos-N78_1TOPS_4PLE_RATIO");
            break;
        case ethosn::support_library::EthosNVariant::ETHOS_N78_2TOPS_2PLE_RATIO:
            os << Quoted("Ethos-N78_2TOPS_2PLE_RATIO");
            break;
        case ethosn::support_library::EthosNVariant::ETHOS_N78_2TOPS_4PLE_RATIO:
            os << Quoted("Ethos-N78_2TOPS_4PLE_RATIO");
            break;
        case ethosn::support_library::EthosNVariant::ETHOS_N78_4TOPS_2PLE_RATIO:
            os << Quoted("Ethos-N78_4TOPS_2PLE_RATIO");
            break;
        case ethosn::support_library::EthosNVariant::ETHOS_N78_4TOPS_4PLE_RATIO:
            os << Quoted("Ethos-N78_4TOPS_4PLE_RATIO");
            break;
        case ethosn::support_library::EthosNVariant::ETHOS_N78_8TOPS_2PLE_RATIO:
            os << Quoted("Ethos-N78_8TOPS_2PLE_RATIO");
            break;
        default:
            ARMNN_ASSERT_MSG(false, "Unexpected variant");
    }
    return os;
}

std::ostream& Print(std::ostream& os, Indent indent, const std::map<uint32_t, std::string>& map)
{
    os << indent << "{\n";
    ++indent;

    for (auto it = map.begin(); it != map.end(); ++it)
    {
        os << indent << JsonField(it->first) << ' ' << Quoted(it->second);
        if (it != std::prev(map.end()))
        {
            os << ",";
        }
        os << '\n';
    }

    --indent;
    os << indent << "}";
    return os;
}

}    // namespace

void EthosNPreCompiledWorkload::SavePerformanceJson() const
{
    const EthosNPreCompiledObject::PerfData& perfData = *m_PreCompiledObject->GetPerfData();

    std::ofstream os(perfData.m_PerfOutFile);

    Indent indent(0);
    os << indent << "{\n";
    ++indent;

    os << indent << JsonField("Config") << "\n";
    os << indent << "{\n";
    indent++;

    os << indent << JsonField("Variant") << ' ' << perfData.m_PerfVariant << ",\n";
    os << indent << JsonField("SramSizeBytesOverride") << ' ' << perfData.m_PerfSramSizeBytesOverride << ",\n";
    os << indent << JsonField("ActivationCompressionSavings") << ' '
       << perfData.m_EstimationOptions.m_ActivationCompressionSaving << ",\n";

    if (perfData.m_EstimationOptions.m_UseWeightCompressionOverride)
    {
        os << indent << JsonField("WeightCompressionSavings") << ' '
           << perfData.m_EstimationOptions.m_WeightCompressionSaving << ",\n";
    }
    else
    {
        os << indent << JsonField("WeightCompressionSavings") << ' ' << Quoted("Not Specified") << ",\n";
    }

    os << indent << JsonField("Current") << ' ' << perfData.m_EstimationOptions.m_Current << "\n";

    indent--;
    os << indent << "},\n";

    os << indent << JsonField("OperationNames") << '\n';
    Print(os, indent, m_PreCompiledObject->GetEthosNOperationNameMapping()) << ",\n";

    os << indent << JsonField("Results") << '\n';
    ethosn::support_library::PrintNetworkPerformanceDataJson(os, static_cast<uint32_t>(indent.m_Depth),
                                                             perfData.m_Data);

    --indent;

    os << indent << "}\n";
}

bool EthosNPreCompiledWorkloadValidate(std::string*)
{
    return true;
}

}    //namespace armnn
