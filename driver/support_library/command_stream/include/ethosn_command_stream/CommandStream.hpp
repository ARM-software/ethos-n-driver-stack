//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Command.hpp"
#include "CommandData.hpp"

#define ETHOSN_COMMAND_STREAM_VERSION_MAJOR 1
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

    const CommandHeader* NextHeader() const
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
            default:
                return nullptr;
        }
    }

    const CommandHeader* m_Head;
};

class CommandStream
{
public:
    using ConstIterator = CommandStreamConstIterator;

    CommandStream(const CommandHeader* begin, const CommandHeader* end)
        : m_Begin(begin)
        , m_End(end)
    {}

    CommandStream(const void* rawBegin, const void* rawEnd)
        : CommandStream(reinterpret_cast<const CommandHeader*>(rawBegin),
                        reinterpret_cast<const CommandHeader*>(rawEnd))
    {}

    ConstIterator begin() const
    {
        return ConstIterator(m_Begin);
    }

    ConstIterator end() const
    {
        return ConstIterator(m_End);
    }

private:
    const CommandHeader* m_Begin;
    const CommandHeader* m_End;
};

}    // namespace command_stream
}    // namespace ethosn
