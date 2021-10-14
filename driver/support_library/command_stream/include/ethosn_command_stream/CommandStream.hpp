//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Command.hpp"
#include "CommandData.hpp"
#include "cascading/CommandStream.hpp"

#include <cstddef>

#define ETHOSN_COMMAND_STREAM_VERSION_MAJOR 3
#define ETHOSN_COMMAND_STREAM_VERSION_MINOR 0
#define ETHOSN_COMMAND_STREAM_VERSION_PATCH 0

namespace ethosn
{
namespace command_stream
{

class CommandStreamConstIterator
{
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type        = CommandHeader;
    using difference_type   = int;
    using pointer           = CommandHeader*;
    using reference         = CommandHeader&;

    explicit CommandStreamConstIterator(const CommandHeader* head)
        : m_Head(head)
    {}

    const CommandHeader& operator*() const
    {
        return *m_Head;
    }

    const CommandHeader* operator->() const
    {
        return m_Head;
    }

    CommandStreamConstIterator& operator++()
    {
        m_Head = NextHeader();
        return *this;
    }

    bool operator==(const CommandStreamConstIterator& other) const
    {
        return m_Head == other.m_Head;
    }

    bool operator!=(const CommandStreamConstIterator& other) const
    {
        return !(*this == other);
    }

private:
    template <Opcode O>
    const CommandHeader* NextHeaderImpl() const
    {
        return &(m_Head->GetCommand<O>() + 1U)->m_Header();
    }

    const CommandHeader* NextHeader() const;

    const CommandHeader* m_Head;
};

template <>
inline const CommandHeader* CommandStreamConstIterator::NextHeaderImpl<Opcode::CASCADE>() const
{
    const auto cascadeHeader = m_Head->GetCommand<Opcode::CASCADE>();
    const size_t cascadeSize = cascadeHeader->m_Data().m_NumAgents() * sizeof(cascading::Agent);
    const auto nextHeader    = reinterpret_cast<const uint8_t*>(cascadeHeader + 1U) + cascadeSize;
    return reinterpret_cast<const CommandHeader*>(nextHeader);
}

inline const CommandHeader* CommandStreamConstIterator::NextHeader() const
{
    switch (m_Head->m_Opcode())
    {
        case Opcode::FENCE:
            return NextHeaderImpl<Opcode::FENCE>();
        case Opcode::OPERATION_MCE_PLE:
            return NextHeaderImpl<Opcode::OPERATION_MCE_PLE>();
        case Opcode::OPERATION_PLE_ONLY:
            return NextHeaderImpl<Opcode::OPERATION_PLE_ONLY>();
        case Opcode::OPERATION_SOFTMAX:
            return NextHeaderImpl<Opcode::OPERATION_SOFTMAX>();
        case Opcode::OPERATION_CONVERT:
            return NextHeaderImpl<Opcode::OPERATION_CONVERT>();
        case Opcode::OPERATION_SPACE_TO_DEPTH:
            return NextHeaderImpl<Opcode::OPERATION_SPACE_TO_DEPTH>();
        case Opcode::DUMP_DRAM:
            return NextHeaderImpl<Opcode::DUMP_DRAM>();
        case Opcode::DUMP_SRAM:
            return NextHeaderImpl<Opcode::DUMP_SRAM>();
        case Opcode::SECTION:
            return NextHeaderImpl<Opcode::SECTION>();
        case Opcode::DELAY:
            return NextHeaderImpl<Opcode::DELAY>();
        case Opcode::CASCADE:
            return NextHeaderImpl<Opcode::CASCADE>();
        default:
            return nullptr;
    }
}

class CommandStream
{
public:
    using ConstIterator = CommandStreamConstIterator;

    CommandStream(const CommandHeader* begin, const CommandHeader* end)
        : m_VersionMajor(ETHOSN_COMMAND_STREAM_VERSION_MAJOR)
        , m_VersionMinor(ETHOSN_COMMAND_STREAM_VERSION_MINOR)
        , m_VersionPatch(ETHOSN_COMMAND_STREAM_VERSION_PATCH)
        , m_Begin(begin)
        , m_End(end)
    {}

    CommandStream(const void* rawBegin, const void* rawEnd)
        : m_VersionMajor(0)
        , m_VersionMinor(0)
        , m_VersionPatch(0)
        , m_Begin(nullptr)
        , m_End(nullptr)
    {
        const uint32_t* rawBeginU32 = reinterpret_cast<const uint32_t*>(rawBegin);
        const uint32_t* rawEndU32   = reinterpret_cast<const uint32_t*>(rawEnd);

        constexpr ptrdiff_t versionHeaderSizeWords = 4;
        if (rawEndU32 - rawBeginU32 < versionHeaderSizeWords)
        {
            return;
        }

        const uint32_t fourcc             = rawBeginU32[0];
        constexpr uint32_t expectedFourcc = static_cast<uint32_t>('E') | (static_cast<uint32_t>('N') << 8) |
                                            (static_cast<uint32_t>('C') << 16) | (static_cast<uint32_t>('S') << 24);
        if (fourcc != expectedFourcc)
        {
            return;
        }

        m_VersionMajor = rawBeginU32[1];
        m_VersionMinor = rawBeginU32[2];
        m_VersionPatch = rawBeginU32[3];
        if (m_VersionMajor != ETHOSN_COMMAND_STREAM_VERSION_MAJOR ||
            m_VersionMinor != ETHOSN_COMMAND_STREAM_VERSION_MINOR ||
            m_VersionPatch != ETHOSN_COMMAND_STREAM_VERSION_PATCH)
        {
            return;
        }

        m_Begin = reinterpret_cast<const CommandHeader*>(rawBeginU32 + versionHeaderSizeWords);
        m_End   = reinterpret_cast<const CommandHeader*>(rawEnd);
    }

    ConstIterator begin() const
    {
        return ConstIterator(m_Begin);
    }

    ConstIterator end() const
    {
        return ConstIterator(m_End);
    }

    bool IsValid() const
    {
        return m_Begin != nullptr && m_End != nullptr;
    }

    uint32_t GetVersionMajor() const
    {
        return m_VersionMajor;
    }
    uint32_t GetVersionMinor() const
    {
        return m_VersionMinor;
    }
    uint32_t GetVersionPatch() const
    {
        return m_VersionPatch;
    }

private:
    uint32_t m_VersionMajor;
    uint32_t m_VersionMinor;
    uint32_t m_VersionPatch;
    const CommandHeader* m_Begin;
    const CommandHeader* m_End;
};

template <Opcode O>
constexpr bool AreCommandsEqual(const CommandHeader& lhs, const CommandHeader& rhs)
{
    const auto opLhs = lhs.GetCommand<O>();
    const auto opRhs = rhs.GetCommand<O>();
    bool equal       = *opLhs == *opRhs;
    return equal;
}

constexpr bool AreCommandsEqual(const CommandHeader& lhs, const CommandHeader& rhs)
{
    if (lhs.m_Opcode() != rhs.m_Opcode())
    {
        return false;
    }
    switch (lhs.m_Opcode())
    {
        case Opcode::OPERATION_MCE_PLE:
            return AreCommandsEqual<Opcode::OPERATION_MCE_PLE>(lhs, rhs);
        case Opcode::OPERATION_PLE_ONLY:
            return AreCommandsEqual<Opcode::OPERATION_PLE_ONLY>(lhs, rhs);
        case Opcode::OPERATION_CONVERT:
            return AreCommandsEqual<Opcode::OPERATION_CONVERT>(lhs, rhs);
        case Opcode::DELAY:
            return AreCommandsEqual<Opcode::DELAY>(lhs, rhs);
        case Opcode::DUMP_DRAM:
            return AreCommandsEqual<Opcode::DUMP_DRAM>(lhs, rhs);
        case Opcode::DUMP_SRAM:
            return AreCommandsEqual<Opcode::DUMP_SRAM>(lhs, rhs);
        case Opcode::FENCE:
            return AreCommandsEqual<Opcode::FENCE>(lhs, rhs);
        case Opcode::OPERATION_SOFTMAX:
            return AreCommandsEqual<Opcode::OPERATION_SOFTMAX>(lhs, rhs);
        case Opcode::OPERATION_SPACE_TO_DEPTH:
            return AreCommandsEqual<Opcode::OPERATION_SPACE_TO_DEPTH>(lhs, rhs);
        case Opcode::SECTION:
            return AreCommandsEqual<Opcode::SECTION>(lhs, rhs);
        case Opcode::CASCADE:
            return AreCommandsEqual<Opcode::CASCADE>(lhs, rhs);
        default:
            return false;
    }
}

}    // namespace command_stream
}    // namespace ethosn
