//
// Copyright © 2018-2021,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "BinaryTuple.hpp"
#include "Opcode.hpp"
#include "PleOperation.hpp"
#include "cascading/CommandStream.hpp"

#include <array>

namespace ethosn
{
namespace command_stream
{

using TensorShape = std::array<uint32_t, 4>;
using Filename    = std::array<char, 128>;

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

enum class SramAllocationStrategy : uint8_t
{
    STRATEGY_0,
    STRATEGY_1,
    STRATEGY_3,
    STRATEGY_4,
    STRATEGY_6,
    STRATEGY_7,
    STRATEGY_X
};

enum class UpsampleType : uint8_t
{
    OFF,
    BILINEAR,
    NEAREST_NEIGHBOUR,
    TRANSPOSE,
};

enum class UpsampleEdgeMode : uint8_t
{
    GENERATE,
    DROP,
};

enum class MceOperation : uint8_t
{
    CONVOLUTION,
    DEPTHWISE_CONVOLUTION,
    FULLY_CONNECTED,
};

enum class MceAlgorithm : uint8_t
{
    DIRECT,
    WINOGRAD,
};

enum class DataLocation : uint8_t
{
    DRAM,
    SRAM,
};

enum class SectionType : uint8_t
{
    SISO,
    SISO_CASCADED,
    SIMO,
    SIMO_CASCADED,
    SISO_BRANCHED_CASCADED,
    MISO,
};

// clang-format off
NAMED_BINARY_TUPLE(TensorInfo,
                   DataType, DataType,
                   DataFormat, DataFormat,
                   TensorShape, TensorShape,
                   TensorShape, SupertensorShape,
                   TensorShape, SupertensorOffset,
                   TensorShape, StripeShape,
                   uint32_t, TileSize,
                   uint32_t, DramBufferId,
                   uint32_t, SramOffset,
                   int16_t, ZeroPoint,
                   DataLocation, DataLocation);

NAMED_BINARY_TUPLE(SramConfig,
                   SramAllocationStrategy, AllocationStrategy);

NAMED_BINARY_TUPLE(BlockConfig,
                   uint32_t, BlockWidth,
                   uint32_t, BlockHeight);

NAMED_BINARY_TUPLE(MceStrideConfig,
                   uint32_t, X,
                   uint32_t, Y);

NAMED_BINARY_TUPLE(MceData,
                   MceStrideConfig, Stride,
                   uint32_t, PadTop,
                   uint32_t, PadLeft,
                   TensorShape, UninterleavedInputShape,
                   TensorShape, OutputShape,
                   TensorShape, OutputStripeShape,
                   int16_t, OutputZeroPoint,
                   UpsampleType, UpsampleType,
                   UpsampleEdgeMode, UpsampleEdgeModeRow,
                   UpsampleEdgeMode, UpsampleEdgeModeCol,
                   MceOperation, Operation,
                   MceAlgorithm, Algorithm,
                   int16_t, ActivationMin,
                   int16_t, ActivationMax);

NAMED_BINARY_TUPLE(PleData,
                   uint32_t, CeSram,
                   uint32_t, PleSram,
                   PleOperation, Operation,
                   uint16_t, RescaleMultiplier0,
                   uint16_t, RescaleShift0,
                   uint16_t, RescaleMultiplier1,
                   uint16_t, RescaleShift1);

template <Opcode O>
struct CommandData;

NAMED_BINARY_TUPLE_SPECIALIZATION(CommandData<Opcode::OPERATION_MCE_PLE>, CommandData,
                                  TensorInfo, InputInfo,
                                  TensorInfo, WeightInfo,
                                  uint32_t, WeightMetadataBufferId,
                                  TensorInfo, OutputInfo,
                                  SramConfig, SramConfig,
                                  BlockConfig, BlockConfig,
                                  MceData, MceData,
                                  PleData, PleData);

NAMED_BINARY_TUPLE_SPECIALIZATION(CommandData<Opcode::OPERATION_PLE_ONLY>, CommandData,
                                  int32_t, NumInputInfos,
                                  TensorInfo, InputInfo,
                                  TensorInfo, InputInfo2,
                                  TensorInfo, OutputInfo,
                                  SramConfig, SramConfig,
                                  PleData, PleData);

NAMED_BINARY_TUPLE_SPECIALIZATION(CommandData<Opcode::OPERATION_CONVERT>, CommandData,
    TensorInfo, InputInfo,
    TensorInfo, OutputInfo);

NAMED_BINARY_TUPLE_SPECIALIZATION(CommandData<Opcode::OPERATION_SPACE_TO_DEPTH>, CommandData,
    TensorInfo, InputInfo,
    TensorInfo, OutputInfo,
    uint32_t, UsedEmcs,
    uint32_t, Intermediate1Size,
    uint32_t, Intermediate2Size);

NAMED_BINARY_TUPLE_SPECIALIZATION(CommandData<Opcode::DUMP_DRAM>, CommandData,
                                  uint32_t, DramBufferId,
                                  Filename, Filename);

NAMED_BINARY_TUPLE_SPECIALIZATION(CommandData<Opcode::DUMP_SRAM>, CommandData,
                                  Filename, Filename);

template<> struct CommandData<Opcode::FENCE> : public BinaryTuple<> {};

NAMED_BINARY_TUPLE_SPECIALIZATION(CommandData<Opcode::SECTION>, CommandData,
                                  SectionType, Type);

NAMED_BINARY_TUPLE_SPECIALIZATION(CommandData<Opcode::DELAY>, CommandData,
                                  uint32_t, Value);

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

using McePle       = CommandData<Opcode::OPERATION_MCE_PLE>;
using PleOnly      = CommandData<Opcode::OPERATION_PLE_ONLY>;
using Convert      = CommandData<Opcode::OPERATION_CONVERT>;
using SpaceToDepth = CommandData<Opcode::OPERATION_SPACE_TO_DEPTH>;
using DumpDram     = CommandData<Opcode::DUMP_DRAM>;
using DumpSram     = CommandData<Opcode::DUMP_SRAM>;
using Fence        = CommandData<Opcode::FENCE>;
using Section      = CommandData<Opcode::SECTION>;
using Delay        = CommandData<Opcode::DELAY>;
using Cascade      = CommandData<Opcode::CASCADE>;

}    // namespace command_stream
}    // namespace ethosn
