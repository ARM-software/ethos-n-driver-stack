//
// Copyright © 2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "Types.hpp"
#include <common/Utils.hpp>

namespace ethosn::control_unit
{

// Cache line is 32 bytes
constexpr uint32_t g_CacheLineSize = 32;

/// A view of an existing list of variable-length Commands.
/// This simply stores a pointer and size.
/// The view can be shrunk by removing an element from the front, which will move up the pointer
/// and reduce the size by one.
class CommandList
{
public:
    CommandList(const Command* data, uint32_t size, uint32_t prefetchSize, const char* endOfCmdStream)
        : m_Data(data)
        , m_Size(size)
        , m_NextAddressToPrefetch(reinterpret_cast<const char*>(data))
        , m_PrefetchSize(prefetchSize)
        , m_EndOfCmdStream(endOfCmdStream)
    {
        Prefetch();
    }

    uint32_t GetSize() const
    {
        return m_Size;
    }

    bool IsEmpty() const
    {
        return m_Size == 0;
    }

    const Command& GetFirst() const
    {
        ASSERT(!IsEmpty());
        const Command& result = *m_Data;
        return result;
    }

    const Command& GetSecond() const
    {
        ASSERT(m_Size >= 2);
        const Command& result =
            *reinterpret_cast<const Command*>(reinterpret_cast<const char*>(m_Data) + m_Data->GetSize());
        return result;
    }

    const Command& RemoveFirst()
    {
        ASSERT(!IsEmpty());
        const Command& result = m_Data[0];

        m_Data = reinterpret_cast<const Command*>(reinterpret_cast<const char*>(m_Data) + m_Data->GetSize());

        --m_Size;
        return result;
    }

    void Prefetch()
    {
        const char* prefetchTo = reinterpret_cast<const char*>(m_Data) + m_PrefetchSize;
        prefetchTo             = std::min(m_EndOfCmdStream, prefetchTo);    // Avoids warning from strict-overflow
        while (m_NextAddressToPrefetch < prefetchTo)
        {
            __builtin_prefetch(m_NextAddressToPrefetch);
            m_NextAddressToPrefetch += g_CacheLineSize;
        }
    }

private:
    const Command* m_Data;
    uint32_t m_Size;
    const char* m_NextAddressToPrefetch;    // char* to be able to increase address in byte increments
    uint32_t m_PrefetchSize;
    const char* m_EndOfCmdStream;
};

inline LoggingString CommandListToString(const CommandList& cmds, uint32_t origNumCommands)
{
    LoggingString result;
    result.AppendFormat("%u/%u", origNumCommands - cmds.GetSize(), origNumCommands);
    if (cmds.GetSize() > 0)
    {
        result.AppendFormat(" (%u: %s", origNumCommands - cmds.GetSize(), ToString(cmds.GetFirst()).GetCString());
        if (cmds.GetSize() > 1)
        {
            result.AppendFormat(", %u: %s", origNumCommands - cmds.GetSize() + 1,
                                ToString(cmds.GetSecond()).GetCString());
        }
        result += ", ...)";
    }
    return result;
}

template <typename Ctrl>
bool ResolveWaitForCounterCommand(const WaitForCounterCommand& c, Ctrl& ctrl)
{
    switch (c.counterName)
    {
        case CounterName::DmaRd:
            return ctrl.dmaRdCounter >= c.counterValue;
        case CounterName::DmaWr:
            return ctrl.dmaWrCounter >= c.counterValue;
        case CounterName::Mceif:
            return ctrl.mceifCounter >= c.counterValue;
        case CounterName::MceStripe:
            return ctrl.mceStripeCounter >= c.counterValue;
        case CounterName::PleCodeLoadedIntoPleSram:
            return ctrl.pleCodeLoadedIntoPleSramCounter >= c.counterValue;
        case CounterName::PleStripe:
            return ctrl.pleStripeCounter >= c.counterValue;
        default:
            return false;
    }
}

}    // namespace ethosn::control_unit
