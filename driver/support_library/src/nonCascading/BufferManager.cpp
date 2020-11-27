//
// Copyright © 2018-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "BufferManager.hpp"

#include "Utils.hpp"

#include <ethosn_command_stream/CommandStreamBuffer.hpp>

#include <cassert>

namespace ethosn
{
namespace support_library
{

BufferManager::BufferManager()
    // Reserve buffer ID 0 for command stream
    : m_NextDramBufferId(1)
    // Use a separate ID space for SRAM buffers because they are not needed at runtime.
    , m_NextSramBufferId(0x8000000)
{}

uint32_t BufferManager::AddDram(BufferType type, uint32_t size)
{
    assert(type == BufferType::Input || type == BufferType::Intermediate || type == BufferType::Output);
    CompilerBufferInfo buffer(type, 0, size, BufferLocation::Dram, std::vector<uint8_t>(), 0xFFFFFFFF, 0xFFFFFFFF);
    m_Buffers.insert({ m_NextDramBufferId, buffer });
    ++m_NextDramBufferId;
    return m_NextDramBufferId - 1;
}

uint32_t BufferManager::AddDramConstant(BufferType type, const std::vector<uint8_t>& constantData)
{
    assert(type == BufferType::ConstantDma || type == BufferType::ConstantControlUnit);
    CompilerBufferInfo buffer(type, 0, static_cast<uint32_t>(constantData.size()), BufferLocation::Dram, constantData,
                              0xFFFFFFFF, 0xFFFFFFFF);
    m_Buffers.insert({ m_NextDramBufferId, buffer });
    ++m_NextDramBufferId;
    return m_NextDramBufferId - 1;
}

uint32_t BufferManager::AddDramInput(uint32_t size, uint32_t sourceOperationId)
{
    // Input index will always be index 0 because it is the output of the Input layer
    //      and this layer cannot have more than one output. (CompilerBufferInfo last argument)
    CompilerBufferInfo buffer(BufferType::Input, 0, size, BufferLocation::Dram, std::vector<uint8_t>(),
                              sourceOperationId, 0);
    m_Buffers.insert({ m_NextDramBufferId, buffer });
    ++m_NextDramBufferId;
    return m_NextDramBufferId - 1;
}

uint32_t BufferManager::AddSram(uint32_t size, uint32_t offset)
{
    CompilerBufferInfo buffer(BufferType::Intermediate, offset, size, BufferLocation::Sram, std::vector<uint8_t>(),
                              0xFFFFFFFF, 0xFFFFFFFF);
    m_Buffers.insert({ m_NextSramBufferId, buffer });
    ++m_NextSramBufferId;
    return m_NextSramBufferId - 1;
}

void BufferManager::AddCommandStream(const ethosn::command_stream::CommandStreamBuffer& cmdStream)
{
    assert(m_Buffers.find(0) == m_Buffers.end());
    std::vector<uint8_t> cmdStreamData;
    cmdStreamData.assign(reinterpret_cast<const uint8_t*>(cmdStream.GetData().data()),
                         reinterpret_cast<const uint8_t*>(cmdStream.GetData().data() + cmdStream.GetData().size()));
    CompilerBufferInfo buffer(BufferType::ConstantControlUnit, 0, static_cast<uint32_t>(cmdStreamData.size()),
                              BufferLocation::Dram, cmdStreamData, 0xFFFFFFFF, 0xFFFFFFFF);
    m_Buffers.insert({ 0, buffer });    // Command stream is always buffer 0.
}

void BufferManager::ChangeToOutput(uint32_t bufferId, uint32_t sourceOperationId, uint32_t sourceOperationOutputIndex)
{
    m_Buffers.at(bufferId).m_Type                       = BufferType::Output;
    m_Buffers.at(bufferId).m_SourceOperationId          = sourceOperationId;
    m_Buffers.at(bufferId).m_SourceOperationOutputIndex = sourceOperationOutputIndex;
}

uint32_t BufferManager::GetSramOffset(uint32_t bufferId)
{
    const CompilerBufferInfo& buffer = m_Buffers.at(bufferId);
    return buffer.m_Location == BufferLocation::Sram ? buffer.m_Offset : 0;
}

namespace
{

uint32_t AppendBufferAligned(uint32_t& cumulativeOffset, uint32_t alignment, uint32_t size)
{
    cumulativeOffset = utils::RoundUpToNearestMultiple(cumulativeOffset, alignment);
    uint32_t offset  = cumulativeOffset;
    cumulativeOffset += size;
    return offset;
}

uint32_t AppendBufferAligned(std::vector<uint8_t>& dest, uint32_t alignment, const std::vector<uint8_t>& src)
{
    // Pad to the required alignment
    dest.resize(utils::RoundUpToNearestMultiple(dest.size(), alignment));

    // Remember the location where we are about to append the data.
    uint32_t offset = static_cast<uint32_t>(dest.size());

    // Append the data
    std::copy(src.begin(), src.end(), std::back_inserter(dest));

    return offset;
}

}    // namespace

void BufferManager::Allocate()
{
    // There is a restriction on the alignment of DRAM accesses for NHWCB and NHWCB_COMPRESSED formats.
    // NHWCB needs to be 16 byte aligned.
    // NHWCB_COMPRESSED needs to be 64 byte aligned.
    constexpr uint32_t alignment = 64;
    uint32_t intermediatesOffset = 0;
    uint32_t inputsOffset        = 0;
    uint32_t outputsOffset       = 0;
    for (auto& internalBufferIt : m_Buffers)
    {
        CompilerBufferInfo& buffer = internalBufferIt.second;
        if (buffer.m_Location != BufferLocation::Dram)
        {
            // Sram buffers already have their offsets set when they are added, so there is nothing to do here.
            continue;
        }

        switch (buffer.m_Type)
        {
            case BufferType::Intermediate:
                buffer.m_Offset = AppendBufferAligned(intermediatesOffset, alignment, buffer.m_Size);
                break;
            case BufferType::ConstantControlUnit:
                buffer.m_Offset = AppendBufferAligned(m_ConstantControlUnitData, alignment, buffer.m_ConstantData);
                break;
            case BufferType::ConstantDma:
                buffer.m_Offset = AppendBufferAligned(m_ConstantDmaData, alignment, buffer.m_ConstantData);
                break;
            case BufferType::Input:
                buffer.m_Offset = AppendBufferAligned(inputsOffset, alignment, buffer.m_Size);
                break;
            case BufferType::Output:
                buffer.m_Offset = AppendBufferAligned(outputsOffset, alignment, buffer.m_Size);
                break;
            default:
                assert(false);
        }
    }
}

const std::map<uint32_t, CompilerBufferInfo>& BufferManager::GetBuffers() const
{
    return m_Buffers;
}

const std::vector<uint8_t>& BufferManager::GetConstantDmaData() const
{
    return m_ConstantDmaData;
}

const std::vector<uint8_t>& BufferManager::GetConstantControlUnitData() const
{
    return m_ConstantControlUnitData;
}

}    // namespace support_library
}    // namespace ethosn
