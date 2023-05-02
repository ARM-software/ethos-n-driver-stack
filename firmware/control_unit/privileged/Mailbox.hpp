//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common/FirmwareApi.hpp>
#include <common/Log.hpp>

#include <ethosn_command_stream/CommandStream.hpp>

#include <ethosn_utils/Macros.hpp>

#if defined CONTROL_UNIT_HARDWARE
#include "HardwareHelpers.hpp"
#endif

#include <array>
#include <cstdint>
#include <cstring>

namespace ethosn
{
namespace control_unit
{

template <typename HAL>
class Mailbox final
{
public:
    enum class Status
    {
        OK,
        ERROR
    };

    Mailbox(HAL& hal, ethosn_mailbox* mailbox)
        : m_Hal(hal)
        , m_Request(*reinterpret_cast<ethosn_queue*>(mailbox->request))
        , m_Response(*reinterpret_cast<ethosn_queue*>(mailbox->response))
    {}

    Status ReadMessage(ethosn_message_header& header, uint8_t* data, size_t length)
    {
        // Wait until a new message arrives
        // We assume that if there is any data in the queue at all
        // then the full message is available. Partial messages should not be
        // observable, as the kernel only updates its write pointer only
        // once the full message is written.
        InvalidateQueueHeaderWritePointer(m_Request);
        while (ethosn_queue_get_size(&m_Request) == 0)
        {
            m_Hal.WaitForEvents();
            InvalidateQueueHeaderWritePointer(m_Request);
        }

        // Read the header
        Status status = Read(header);
        if (status != Status::OK)
        {
            return status;
        }

        if (header.length > length)
        {
            return Status::ERROR;
        }

        // Read the payload
        status = Read(data, header.length);
        if (status != Status::OK)
        {
            return status;
        }

        return Status::OK;
    }

    Status SendFwAndHwCapabilitiesResponse(const std::pair<const char*, size_t>& fwHwCapabilities)
    {
        return WriteMessageRaw<1>(ETHOSN_MESSAGE_FW_HW_CAPS_RESPONSE,
                                  { { static_cast<const void*>(fwHwCapabilities.first), fwHwCapabilities.second } });
    }

    Status SendInferenceResponse(ethosn_inference_status status, uint64_t userArgument, uint64_t cycleCount)
    {
        ethosn_message_inference_response response;
        response.user_argument = userArgument;
        response.status        = status;
        response.cycle_count   = cycleCount;

        return WriteMessage(ETHOSN_MESSAGE_INFERENCE_RESPONSE, response);
    }

    Status SendPong()
    {
        return WriteMessage(ETHOSN_MESSAGE_PONG);
    }

    Status SendConfigureProfilingAck()
    {
        return WriteMessage(ETHOSN_MESSAGE_CONFIGURE_PROFILING_ACK);
    }

    Status SendErrorResponse(uint32_t message_type, ethosn_error_status status)
    {
        ethosn_message_error_response response;
        response.type   = message_type;
        response.status = static_cast<uint32_t>(status);
        return WriteMessage(ETHOSN_MESSAGE_ERROR_RESPONSE, response);
    }

    Status Log(ethosn_log_severity severity, const char* msg)
    {
        ethosn_message_text text;
        text.severity = severity;

        return WriteMessageRaw<2>(ETHOSN_MESSAGE_TEXT,
                                  { { { &text, sizeof(text) }, { static_cast<const void*>(msg), strlen(msg) + 1 } } });
    }

private:
    struct Iovec
    {
        const void* Base;
        size_t Length;
    };

    template <typename T>
    Status Read(T& dst)
    {
        return Read(reinterpret_cast<uint8_t*>(&dst), sizeof(dst));
    }

    Status Read(uint8_t* dst, size_t length)
    {
#if defined CONTROL_UNIT_HARDWARE
        // Invalidate the data we are about to read, taking care of wraparound
        uint32_t unwrappedSize = std::min(static_cast<uint32_t>(length), m_Request.capacity - m_Request.read);
        uint32_t wrappedSize   = static_cast<uint32_t>(length) - unwrappedSize;
        Cache::DInvalidate(static_cast<void*>(m_Request.data + m_Request.read), static_cast<ptrdiff_t>(unwrappedSize));
        if (wrappedSize)
        {
            Cache::DInvalidate(static_cast<void*>(&m_Request.data), static_cast<ptrdiff_t>(wrappedSize));
        }
#endif

        if (!ethosn_queue_read(&m_Request, dst, static_cast<uint32_t>(length)))
        {
            return Status::ERROR;
        }

        // Make sure the changes are visible from the host CPU.
        FlushQueueHeaderReadPointer(m_Request);

        return Status::OK;
    }

    template <size_t N, size_t Index>
    void BuildIovecArray(std::array<Iovec, N>&)
    {}

    template <size_t N, size_t Index = 0, typename TPayload, typename... TOthers>
    void BuildIovecArray(std::array<Iovec, N>& vec, const TPayload& payload, const TOthers&... others)
    {
        vec[Index] = Iovec{ &payload, sizeof(payload) };
        BuildIovecArray<N, Index + 1>(vec, others...);
    }

    /// Writes a message of the given type to the m_Response queue, to be picked up by the host CPU.
    /// The message is defined by its type and its payload, which consists of 0 or more objects to be
    /// copied bytewise into the message.
    /// This is a higher-level variant of WriteMessageRaw, useful for if you want to send a C++ object and avoids
    /// having to calculate pointers and sizes.
    /// @{
    template <typename... TPayloads>
    Status WriteMessage(ethosn_message_type type, const TPayloads&... payloads)
    {
        std::array<Iovec, sizeof...(payloads)> vec;
        BuildIovecArray(vec, payloads...);
        return WriteMessageRaw(type, vec);
    }

    Status WriteMessage(ethosn_message_type type)
    {
        return WriteMessageRaw<0>(type, {});
    }
    /// @}

    /// Writes a message of the given type to the m_Response queue, to be picked up by the host CPU.
    /// The message is defined by its type and its payload, which consists of 0 or more memory regions (Iovecs)
    /// which are copied bytewise into the message.
    /// This is a lower-level variant of WriteMessage, useful for if you want to directly specify a pointer and size
    /// rather than an object.
    template <size_t N>
    Status WriteMessageRaw(ethosn_message_type type, std::array<Iovec, N> payload)
    {
        ethosn_message_header header;
        header.type   = type;
        header.length = 0;    // We calculate this below

        constexpr uint32_t headerSize = static_cast<uint32_t>(sizeof(header));

        // Prepend a buffer for the header which we've constructed
        const uint8_t* buffers[N + 1];
        uint32_t sizes[N + 1];
        buffers[0] = reinterpret_cast<const uint8_t*>(&header);
        sizes[0]   = headerSize;
        for (size_t i = 0; i < N; ++i)
        {
            buffers[i + 1] = static_cast<const uint8_t*>(payload[i].Base);
            sizes[i + 1]   = static_cast<uint32_t>(payload[i].Length);
            header.length += static_cast<uint32_t>(payload[i].Length);
        }

        // Check if the message is too large to fit in the queue at all, even once the reading CPU has caught up.
        // In this case fail early rather than calling Write and getting stuck in an infinite loop waiting for the
        // space.
        const uint32_t totalSize = headerSize + header.length;
        if (!ethosn_queue_can_ever_fit(&m_Response, totalSize))
        {
            m_Hal.m_Logger.Error("Mailbox is not large enough to fit message of size: %u", totalSize);
            return Status::ERROR;
        }

        return Write(buffers, sizes, N + 1, totalSize);
    }

    /// Writes the given buffers into the m_Response queue and raises
    /// an interrupt to notify the host CPU.
    /// buffers and sizes are both arrays of length numBuffers.
    /// totalSize is the sum of the size of every buffer.
    Status Write(const uint8_t** buffers, const uint32_t* sizes, uint32_t numBuffers, uint32_t totalSize)
    {
        const uint32_t writeStart = m_Response.write;
        uint32_t writePending;
        // If it fails, keep trying while we wait for the kernel to read data and free space in the queue.
        while (!ethosn_queue_write(&m_Response, buffers, sizes, numBuffers, &writePending))
        {
            // Make sure to pick up the state that has been updated by the kernel.
            InvalidateQueueHeaderReadPointer(m_Response);
        }

#if defined CONTROL_UNIT_HARDWARE
        // Flush all the data written. Note we must do this *before* modifying the write pointer so that
        // the kernel doesn't get a chance to read invalid data.
        // Note we must account for potential wraparound
        const uint32_t unwrappedSize = std::min(totalSize, m_Response.capacity - writeStart);
        const uint32_t wrappedSize   = totalSize - unwrappedSize;
        Cache::DClean(static_cast<void*>(m_Response.data + writeStart), static_cast<ptrdiff_t>(unwrappedSize));
        if (wrappedSize > 0)
        {
            Cache::DClean(static_cast<void*>(m_Response.data), static_cast<ptrdiff_t>(wrappedSize));
        }
#else
        ETHOSN_UNUSED(totalSize);
        ETHOSN_UNUSED(writeStart);
#endif

        // Data flushed. Update the write pointer.
        m_Response.write = writePending;

        // Make sure the write pointer is visible from the host CPU.
        FlushQueueHeaderWritePointer(m_Response);

        // Signal the host CPU that a new message is available.
        m_Hal.RaiseIRQ();

        return Status::OK;
    }

    /// Commits to DRAM any changes made to the given queue's read pointer.
    void FlushQueueHeaderReadPointer(const ethosn_queue& queue)
    {
#if defined CONTROL_UNIT_HARDWARE
        Cache::DClean(static_cast<const void*>(&queue.read), static_cast<ptrdiff_t>(sizeof(queue.read)));
#else
        ETHOSN_UNUSED(queue);
#endif
    }

    /// Commits to DRAM any changes made to the given queue's write pointer.
    void FlushQueueHeaderWritePointer(const ethosn_queue& queue)
    {
#if defined CONTROL_UNIT_HARDWARE
        Cache::DClean(static_cast<const void*>(&queue.write), static_cast<ptrdiff_t>(sizeof(queue.write)));
#else
        ETHOSN_UNUSED(queue);
#endif
    }

    /// Ensures that any changes to the given queue's read pointer written by the
    /// kernel module are visible to us.
    void InvalidateQueueHeaderReadPointer(ethosn_queue& queue)
    {
#if defined CONTROL_UNIT_HARDWARE
        Cache::DInvalidate(static_cast<void*>(&queue.read), static_cast<ptrdiff_t>(sizeof(queue.read)));
#else
        ETHOSN_UNUSED(queue);
#endif
    }

    /// Ensures that any changes to the given queue's write pointer written by the
    /// kernel module are visible to us.
    void InvalidateQueueHeaderWritePointer(ethosn_queue& queue)
    {
#if defined CONTROL_UNIT_HARDWARE
        Cache::DInvalidate(static_cast<void*>(&queue.write), static_cast<ptrdiff_t>(sizeof(queue.write)));
#else
        ETHOSN_UNUSED(queue);
#endif
    }

    HAL& m_Hal;
    ethosn_queue& m_Request;
    ethosn_queue& m_Response;
};

}    // namespace control_unit
}    // namespace ethosn
