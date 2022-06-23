//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "BufferManager.hpp"

#include "DebuggingContext.hpp"
#include "Utils.hpp"

#include <ethosn_command_stream/CommandStreamBuffer.hpp>

#include <cassert>
#include <fstream>
#include <list>

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
    CompilerBufferInfo buffer(type, 0, size, BufferLocation::Dram);
    m_Buffers.insert({ m_NextDramBufferId, buffer });
    ++m_NextDramBufferId;
    return m_NextDramBufferId - 1;
}

uint32_t BufferManager::AddDramConstant(BufferType type, const std::vector<uint8_t>& constantData)
{
    assert(type == BufferType::ConstantDma || type == BufferType::ConstantControlUnit);
    CompilerBufferInfo buffer(type, 0, static_cast<uint32_t>(constantData.size()), BufferLocation::Dram);
    buffer.m_ConstantData = constantData;
    m_Buffers.insert({ m_NextDramBufferId, buffer });
    ++m_NextDramBufferId;
    return m_NextDramBufferId - 1;
}

uint32_t BufferManager::AddDramInput(uint32_t size, uint32_t sourceOperationId)
{
    CompilerBufferInfo buffer(BufferType::Input, 0, size, BufferLocation::Dram);
    buffer.m_SourceOperationId = sourceOperationId;
    // Input index will always be index 0 because it is the output of the Input layer
    //      and this layer cannot have more than one output. (CompilerBufferInfo last argument)
    buffer.m_SourceOperationOutputIndex = 0;
    m_Buffers.insert({ m_NextDramBufferId, buffer });
    ++m_NextDramBufferId;
    return m_NextDramBufferId - 1;
}

uint32_t BufferManager::AddSram(uint32_t size, uint32_t offset)
{
    CompilerBufferInfo buffer(BufferType::Intermediate, offset, size, BufferLocation::Sram);
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
                              BufferLocation::Dram);
    buffer.m_ConstantData = std::move(cmdStreamData);
    m_Buffers.insert({ 0, buffer });    // Command stream is always buffer 0.
}

void BufferManager::ChangeToOutput(uint32_t bufferId, uint32_t sourceOperationId, uint32_t sourceOperationOutputIndex)
{
    m_Buffers.at(bufferId).m_Type                       = BufferType::Output;
    m_Buffers.at(bufferId).m_SourceOperationId          = sourceOperationId;
    m_Buffers.at(bufferId).m_SourceOperationOutputIndex = sourceOperationOutputIndex;
}

void BufferManager::ChangeBufferAlignment(uint32_t bufferId, uint32_t alignment)
{
    const uint32_t bufferSize = m_Buffers.at(bufferId).m_Size;

    m_Buffers.at(bufferId).m_Size = utils::RoundUpToNearestMultiple(bufferSize, alignment);
}

void BufferManager::MarkBufferUsedAtTime(uint32_t bufferId, uint32_t startTime, uint32_t endTime)
{
    CompilerBufferInfo& buffer = m_Buffers.at(bufferId);

    buffer.m_LifetimeStart = startTime;
    buffer.m_LifetimeEnd   = endTime;
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

void BufferManager::Allocate(const DebuggingContext& debuggingContext)
{
    // There is a restriction on the alignment of DRAM accesses for the NHWCB and FCAF formats.
    // NHWCB needs to be 16 byte aligned.
    // FCAF needs to be 64 byte aligned.
    constexpr uint32_t alignment = 64;
    uint32_t inputsOffset        = 0;
    uint32_t outputsOffset       = 0;
    std::vector<uint32_t> intermediateBufferIds;
    std::vector<first_fit_allocation::Buffer> intermediateFirstFitBuffers;
    for (auto& internalBufferIt : m_Buffers)
    {
        uint32_t bufferId          = internalBufferIt.first;
        CompilerBufferInfo& buffer = internalBufferIt.second;
        if (buffer.m_Location != BufferLocation::Dram)
        {
            // Sram buffers already have their offsets set when they are added, so there is nothing to do here.
            continue;
        }

        switch (buffer.m_Type)
        {
            case BufferType::Intermediate:
                // Intermediate buffers are allocated using a more complicated algorithm and are handled afterwards.
                // We just build up an array of them here
                intermediateBufferIds.push_back(bufferId);
                first_fit_allocation::Buffer firstFitBuffer;
                firstFitBuffer.m_LifetimeStart = buffer.m_LifetimeStart;
                firstFitBuffer.m_LifetimeEnd   = buffer.m_LifetimeEnd;
                firstFitBuffer.m_Size          = buffer.m_Size;
                intermediateFirstFitBuffers.push_back(firstFitBuffer);
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

    bool debugDisableBufferReuse = false;
    // Enable this debugging flag in order to prevent intermediate buffers from re-using the same memory as other
    // intermediate buffers. This can be useful when using the Driver Library's debug option to dump intermediate
    // buffers after an inference completes, as otherwise some intermediate buffers may be corrupted (overwritten by
    // other buffers re-using the same space).
    if (!debugDisableBufferReuse)
    {
        // Allocate intermediate buffers using first-fit algorithm and store the results
        std::vector<uint32_t> intermediateAllocations =
            first_fit_allocation::FirstFitAllocation(std::move(intermediateFirstFitBuffers), alignment);
        for (uint32_t i = 0; i < intermediateBufferIds.size(); ++i)
        {
            uint32_t bufferId               = intermediateBufferIds[i];
            m_Buffers.at(bufferId).m_Offset = intermediateAllocations[i];
        }
    }
    else
    {
        uint32_t intermediatesOffset = 0;
        for (uint32_t bufferId : intermediateBufferIds)
        {
            CompilerBufferInfo& buffer = m_Buffers.at(bufferId);
            buffer.m_Offset            = AppendBufferAligned(intermediatesOffset, alignment, buffer.m_Size);
        }
    }

    // Dump intermediate buffer allocations for debugging/analysis
    if (debuggingContext.m_DebugInfo.m_DumpDebugFiles >= CompilationOptions::DebugLevel::Medium)
    {
        std::ofstream f(debuggingContext.GetAbsolutePathOutputFileName("IntermediateDramBuffers.txt"));
        for (uint32_t bufferId : intermediateBufferIds)
        {
            const CompilerBufferInfo& buffer = m_Buffers.at(bufferId);
            if (buffer.m_Location == BufferLocation::Dram && buffer.m_Type == BufferType::Intermediate)
            {
                f << "Buffer " << bufferId << ", " << buffer.m_Size << " bytes, lifetime " << buffer.m_LifetimeStart
                  << "-" << buffer.m_LifetimeEnd << ", "
                  << "allocated at " << buffer.m_Offset << std::endl;
            }
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

namespace first_fit_allocation
{

std::vector<uint32_t> FirstFitAllocation(std::vector<Buffer> buffers, uint32_t alignment)
{
    // Round up all the buffer sizes as a simple way to ensure that all the allocations will be aligned
    for (Buffer& b : buffers)
    {
        b.m_Size = utils::RoundUpToNearestMultiple(b.m_Size, alignment);
    }

    // Build up a list of when buffers need to be allocated or destroyed, sorted by time
    struct Event
    {
        uint32_t timestamp;
        uint32_t buffer;
        enum class Type
        {
            Free     = 0,    // We sort by the numerical values, so these are important
            Allocate = 1,
        } type;
    };
    std::vector<Event> events;
    events.reserve(buffers.size() * 2);
    for (uint32_t i = 0; i < buffers.size(); ++i)
    {
        assert(buffers[i].m_LifetimeEnd > buffers[i].m_LifetimeStart);
        events.push_back({ buffers[i].m_LifetimeStart, i, Event::Type::Allocate });
        events.push_back({ buffers[i].m_LifetimeEnd, i, Event::Type::Free });
    }
    std::sort(events.begin(), events.end(), [](const Event& a, const Event& b) {
        // Sort by timestamp first, then by event type (so that we free before allocate if there are multiple event types
        // on the same timestamp)
        // Finally sort by buffer ID so that we get deterministic results.
        if (a.timestamp != b.timestamp)
        {
            return a.timestamp < b.timestamp;
        }
        else if (a.type != b.type)
        {
            return a.type < b.type;
        }
        else
        {
            return a.buffer < b.buffer;
        }
    });

    // Go through the sorted events and allocate/free as required.
    // Maintain a list of free regions which we shrink/expand/merge as we go.
    // This is always kept sorted and never has overlapping or adjacent regions.
    // It always has at least one entry.
    constexpr uint32_t MAX = 0xFFFFFFFF;
    std::vector<uint32_t> allocations(buffers.size(), MAX);
    struct Region
    {
        uint32_t start;
        uint32_t end;
    };
    // Note we use a std::list for constant-time insert() and erase() at any point in the list. We do not need random-access.
    std::list<Region> freeRegions{ { 0, MAX } };    // Initially, all memory is free
    for (const Event& e : events)
    {
        if (e.type == Event::Type::Allocate)
        {
            const uint32_t size = buffers[e.buffer].m_Size;
            // Find the first free region that is big enough
            for (auto regionIt = freeRegions.begin(); regionIt != freeRegions.end(); ++regionIt)
            {
                if (size <= regionIt->end - regionIt->start)
                {
                    // Allocate this buffer at the start of the free region, and shrink the free region accordingly.
                    allocations[e.buffer] = regionIt->start;
                    regionIt->start += size;
                    // If the region is now empty, remove it
                    if (regionIt->start == regionIt->end)
                    {
                        freeRegions.erase(regionIt);
                    }
                    break;
                }
            }
        }
        else if (e.type == Event::Type::Free)
        {
            const uint32_t freedStart = allocations[e.buffer];
            const uint32_t freedSize  = buffers[e.buffer].m_Size;
            const uint32_t freedEnd   = freedStart + freedSize;

            // Check if there is a free region immediately beforehand
            auto freeRegionIt                  = freeRegions.begin();
            auto freeRegionImmediatelyBeforeIt = freeRegions.end();
            for (; freeRegionIt->end <= freedStart; ++freeRegionIt)
            {
                if (freeRegionIt->end == freedStart)
                {
                    freeRegionImmediatelyBeforeIt = freeRegionIt;
                    // Note that the loop will now finish after incrementing the iterator
                    // (because the next free region will be > freedStart)
                }
            }

            // Check if there is a free region immediately afterwards
            // Note that after the above loop finishes, freeRegionIt will now be pointing to the free region
            // after the freed buffer, but we still need to check if it follows on immediately
            auto freeRegionImmediatelyAfterIt = freeRegionIt->start == freedEnd ? freeRegionIt : freeRegions.end();

            // Now we either merge, extend or create a new free region, depending on whether there was already a free
            // region before or after
            if (freeRegionImmediatelyBeforeIt == freeRegions.end() && freeRegionImmediatelyAfterIt == freeRegions.end())
            {
                // No free region before or after -> create a new free region
                // Note that freeRegionIt will be pointing to the next free region after the freed buffer, which is where
                // we want to insert the new free region
                freeRegions.insert(freeRegionIt, { freedStart, freedEnd });
            }
            else if (freeRegionImmediatelyBeforeIt == freeRegions.end() &&
                     freeRegionImmediatelyAfterIt != freeRegions.end())
            {
                // Free region after but not before -> extend the region after
                freeRegionImmediatelyAfterIt->start = freedStart;
            }
            else if (freeRegionImmediatelyBeforeIt != freeRegions.end() &&
                     freeRegionImmediatelyAfterIt == freeRegions.end())
            {
                // Free region before but not after -> extend the region before
                freeRegionImmediatelyBeforeIt->end = freedEnd;
            }
            else
            {
                // Freed region both before and after -> merge them
                freeRegionImmediatelyBeforeIt->end = freeRegionImmediatelyAfterIt->end;
                freeRegions.erase(freeRegionImmediatelyAfterIt);
            }
        }
    }

    return allocations;
}

}    // namespace first_fit_allocation

}    // namespace support_library
}    // namespace ethosn
