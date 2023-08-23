//
// Copyright © 2018-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "BinaryTuple.hpp"
#include "Opcode.hpp"
#include "cascading/CommandStream.hpp"

#include <array>

namespace ethosn
{
namespace command_stream
{

using Filename = std::array<char, 128>;

enum class DataType : uint8_t
{
    U8,
    S8
};

enum class DataFormat : uint8_t
{
    NHWCB_COMPRESSED,
    NHWCB,
    NHWC,
    NCHW,
    WEIGHT_STREAM,
    FCAF_DEEP,
    FCAF_WIDE
};

enum class MceOperation : uint8_t
{
    CONVOLUTION,
    DEPTHWISE_CONVOLUTION,
    FULLY_CONNECTED,
};

// clang-format off

NAMED_BINARY_TUPLE(BlockConfig,
                   uint32_t, BlockWidth,
                   uint32_t, BlockHeight);

template <Opcode O>
struct CommandData;

NAMED_BINARY_TUPLE_SPECIALIZATION(CommandData<Opcode::DUMP_DRAM>, CommandData,
                                  uint32_t, DramBufferId,
                                  Filename, Filename);

NAMED_BINARY_TUPLE_SPECIALIZATION(CommandData<Opcode::DUMP_SRAM>, CommandData,
                                  Filename, Filename);

template<> struct CommandData<Opcode::CASCADE> : public BinaryTuple<>
{
    /// Total size (in bytes) of all the data in this Cascade. This includes the size of this struct,
    /// plus the data which follows it (array of Agents and lists of mixed-type Commands).
    uint32_t TotalSize;

    /// Offset (in bytes) from the start of this struct to the array of agents.
    uint32_t AgentsOffset;
    uint32_t NumAgents;

    /// Offset (in bytes) from the start of this struct to the DMA read commands.
    uint32_t DmaRdCommandsOffset;
    uint32_t NumDmaRdCommands;

    /// Offset (in bytes) from the start of this struct to the DMA write commands.
    uint32_t DmaWrCommandsOffset;
    uint32_t NumDmaWrCommands;

    /// Offset (in bytes) from the start of this struct to the MCE commands.
    uint32_t MceCommandsOffset;
    uint32_t NumMceCommands;

    /// Offset (in bytes) from the start of this struct to the PLE commands.
    uint32_t PleCommandsOffset;
    uint32_t NumPleCommands;

    // Following this struct there will be an array of cascading::Agent then four
    // lists of mixed-type cascading::Commands.
    // The above fields describe this layout, and the below methods provide easy access to them.

    const cascading::Agent* GetAgentsArray() const
    {
        const char* basePtr = reinterpret_cast<const char*>(this);
        return reinterpret_cast<const cascading::Agent*>(basePtr + AgentsOffset);
    }
    const cascading::Command* GetDmaRdCommandsBegin() const
    {
        const char* basePtr = reinterpret_cast<const char*>(this);
        return reinterpret_cast<const cascading::Command*>(basePtr + DmaRdCommandsOffset);
    }
    const cascading::Command* GetDmaWrCommandsBegin() const
    {
        const char* basePtr = reinterpret_cast<const char*>(this);
        return reinterpret_cast<const cascading::Command*>(basePtr + DmaWrCommandsOffset);
    }
    const cascading::Command* GetMceCommandsBegin() const
    {
        const char* basePtr = reinterpret_cast<const char*>(this);
        return reinterpret_cast<const cascading::Command*>(basePtr + MceCommandsOffset);
    }
    const cascading::Command* GetPleCommandsBegin() const
    {
        const char* basePtr = reinterpret_cast<const char*>(this);
        return reinterpret_cast<const cascading::Command*>(basePtr + PleCommandsOffset);
    }
};

namespace impl {

// BinaryTypeTraits specialization for CommandData<Opcode::CASCADE>.
// This is needed because we are using a regular struct rather than a BinaryTuple.
template <>
struct BinaryTypeTraits<CommandData<Opcode::CASCADE>>
{
    constexpr static size_t Align = alignof(CommandData<Opcode::CASCADE>);
    constexpr static size_t Size = sizeof(CommandData<Opcode::CASCADE>);
};

}

// clang-format on

using DumpDram = CommandData<Opcode::DUMP_DRAM>;
using DumpSram = CommandData<Opcode::DUMP_SRAM>;
using Cascade  = CommandData<Opcode::CASCADE>;

}    // namespace command_stream
}    // namespace ethosn
