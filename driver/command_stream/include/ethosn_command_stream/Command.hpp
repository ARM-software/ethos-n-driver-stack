//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "BinaryTuple.hpp"
#include "Opcode.hpp"

namespace ethosn
{
namespace command_stream
{

template <Opcode O>
struct Command;

template <Opcode O>
struct CommandData;

// clang-format off
// cppcheck-suppress syntaxError symbolName=CommandHeaderBase
NAMED_BINARY_TUPLE(CommandHeaderBase,
                   const Opcode, Opcode);
// clang-format on

struct CommandHeader : public CommandHeaderBase
{
    constexpr CommandHeader(Opcode opcode)
        : CommandHeaderBase(opcode)
    {}

    template <Opcode O>
    constexpr const Command<O>* GetCommand() const
    {
        return (m_Opcode() == O) ? reinterpret_cast<const Command<O>*>(this) : nullptr;
    }
};

// clang-format off
template <Opcode O>
NAMED_ALIGNED_BINARY_TUPLE_4(CommandBase,
                             CommandHeader, Header,
                             CommandData<O>, Data);
// clang-format on

template <Opcode O>
struct Command : public CommandBase<O>
{
    constexpr Command(const CommandData<O>& data)
        : CommandBase<O>(O, data)
    {
        using Properties = typename CommandBase<O>::Properties;
        static_assert(Properties::Header == static_cast<Properties>(0), "Header must be the first property");
    }

    constexpr Command()
        : Command(CommandData<O>{})
    {}
};

}    // namespace command_stream
}    // namespace ethosn
