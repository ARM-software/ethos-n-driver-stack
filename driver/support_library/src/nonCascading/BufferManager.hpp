//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <map>
#include <vector>

namespace ethosn
{

namespace command_stream
{
class CommandStreamBuffer;
}

namespace support_library
{

constexpr uint32_t g_NhwcbBufferAlignment = 1024;

enum class BufferType
{
    Input,
    Output,
    ConstantDma,
    ConstantControlUnit,
    Intermediate
};

enum class BufferLocation
{
    None,
    Dram,
    Sram,
};

struct CompilerBufferInfo
{
public:
    constexpr static const uint32_t ms_InvalidValue = 0xFFFFFFFF;

    CompilerBufferInfo(BufferType type, uint32_t offset, uint32_t size, BufferLocation location)
        : m_Type(type)
        , m_Offset(offset)
        , m_Size(size)
        , m_Location(location)
        , m_SourceOperationId(ms_InvalidValue)
        , m_SourceOperationOutputIndex(ms_InvalidValue)
    {}

    BufferType m_Type;
    uint32_t m_Offset;    ///< For DRAM buffers, this is not set to a proper value until Allocate().
    uint32_t m_Size;
    BufferLocation m_Location;
    std::vector<uint8_t> m_ConstantData;      ///< May be empty if this buffer is not constant.
    uint32_t m_SourceOperationId;             ///< Only relevant for input and output buffer infos.
    uint32_t m_SourceOperationOutputIndex;    ///< Only relevant for input and output buffer infos.
};

/// Maintains and builds up the set of buffers required by the compiled network.
class BufferManager
{
public:
    BufferManager();

    /// Adds a new buffer with the given properties. Returns the ID of the buffer.
    /// @{
    uint32_t AddDram(BufferType type, uint32_t size);
    uint32_t AddDramConstant(BufferType type, const std::vector<uint8_t>& constantData);
    uint32_t AddDramInput(uint32_t size, uint32_t sourceOperationId);
    uint32_t AddSram(uint32_t size, uint32_t offset);
    /// @}

    /// Adds the command stream buffer, which always has an ID of zero.
    void AddCommandStream(const ethosn::command_stream::CommandStreamBuffer& cmdStream);

    /// Changes the given buffer into an output.
    void ChangeToOutput(uint32_t bufferId, uint32_t sourceOperationId, uint32_t sourceOperationOutputIndex);

    void ChangeBufferAlignment(uint32_t bufferId, uint32_t alignment);

    /// If the given buffer is an SRAM buffer then returns the offset in SRAM of the given buffer,
    /// otherwise returns zero.
    uint32_t GetSramOffset(uint32_t bufferId);

    /// Sets of m_Offset field of all DRAM buffers such that all buffers of each type are laid out contiguously.
    /// Also fills in m_ConstantDmaData and m_ConstantControlUnitData with the concatenated data from all
    /// constant buffers of the corresponding type.
    /// Call this once all buffers have been added.
    void Allocate();

    const std::map<uint32_t, CompilerBufferInfo>& GetBuffers() const;
    const std::vector<uint8_t>& GetConstantDmaData() const;
    const std::vector<uint8_t>& GetConstantControlUnitData() const;

private:
    /// All the buffers we currently know about, looked up by ID.
    /// Note that the order of this map is unimportant but we still use an ordered map so that the
    /// order of iteration is consistent across implementations so that Allocate() will allocate
    /// buffers in the same order.
    std::map<uint32_t, CompilerBufferInfo> m_Buffers;
    uint32_t m_NextDramBufferId;
    uint32_t m_NextSramBufferId;

    std::vector<uint8_t> m_ConstantDmaData;
    std::vector<uint8_t> m_ConstantControlUnitData;
};

namespace first_fit_allocation
{

/// Minimal description of a buffer, to be used as input for FirstFitAllocation.
struct Buffer
{
    uint32_t m_LifetimeStart;
    uint32_t m_LifetimeEnd;
    uint32_t m_Size;
};

/// Decides where each of the given buffers should be placed, such that no buffers overlap in space and lifetime.
/// This is implemented with a 'first-fit' scheme - each buffer is allocated at the smallest memory address
/// that gives a valid allocation (not overlapping lifetime and memory with any other buffer).
/// This is not an optimal solution but it is quite fast and gives acceptable results for the use case of intermediate
/// DRAM buffer allocation.
/// The result is an array of allocated addresses, with each element containing the allocated address for the
/// corresponding input buffer.
/// All allocated addresses are guaranteed to be aligned to the given alignment.
std::vector<uint32_t> FirstFitAllocation(std::vector<Buffer> buffers, uint32_t alignment);

}    // namespace first_fit_allocation

}    // namespace support_library
}    // namespace ethosn
