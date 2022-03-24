//
// Copyright © 2018-2022 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "SramAllocator.hpp"

#include <algorithm>
#include <cassert>

namespace ethosn
{
namespace support_library
{

SramAllocator& SramAllocator::operator=(const SramAllocator& s)
{
    this->m_Capacity   = s.m_Capacity;
    this->m_FreeMemory = s.m_FreeMemory;
    this->m_UsedMemory = s.m_UsedMemory;
    return *this;
}

std::pair<bool, uint32_t>
    SramAllocator::Allocate(UserId userId, uint32_t size, AllocationPreference pref, std::string debugName)
{
    if (pref == AllocationPreference::Start)
    {
        for (auto range = m_FreeMemory.begin(); range != m_FreeMemory.end(); ++range)
        {
            if (size <= range->m_End - range->m_Begin)
            {
                MemoryChunk chunk = { range->m_Begin, range->m_Begin + size, { userId }, debugName };
                m_UsedMemory.emplace_back(chunk);
                range->m_Begin += size;
                if (range->m_Begin == range->m_End)
                {
                    m_FreeMemory.erase(range);
                }
                return { true, chunk.m_Begin };
            }
        }
    }
    else if (pref == AllocationPreference::End)
    {
        // If we prefer starting at the end, we iterate through the free regions backwards
        // and allocate at the end of the region
        for (auto range = m_FreeMemory.rbegin(); range != m_FreeMemory.rend(); ++range)
        {
            if (size <= range->m_End - range->m_Begin)
            {
                MemoryChunk chunk = { range->m_End - size, range->m_End, { userId }, debugName };
                m_UsedMemory.emplace_back(chunk);
                range->m_End -= size;
                if (range->m_Begin == range->m_End)
                {
                    m_FreeMemory.erase(std::next(range).base());
                }
                return { true, chunk.m_Begin };
            }
        }
    }

    return { false, 0 };
}

void SramAllocator::IncrementReferenceCount(UserId userId, uint32_t offset)
{
    auto MatchChunk    = [offset](const auto& chunk) { return (offset == chunk.m_Begin); };
    auto memoryChunkIt = std::find_if(m_UsedMemory.begin(), m_UsedMemory.end(), MatchChunk);
    assert(memoryChunkIt != m_UsedMemory.end());
    memoryChunkIt->m_ListOfUsers.push_back(userId);
}

bool SramAllocator::TryFree(UserId userId, uint32_t offset)
{
    auto MatchChunk = [offset](const auto& chunk) { return (offset == chunk.m_Begin); };
    // Remove the chunk from used memory and add it to the free memory.
    auto memoryChunkIt = std::find_if(m_UsedMemory.begin(), m_UsedMemory.end(), MatchChunk);

    if (memoryChunkIt == m_UsedMemory.end())
    {
        return false;
    }

    std::vector<SramAllocator::UserId>& listOfUser = memoryChunkIt->m_ListOfUsers;
    auto userIt                                    = std::find(listOfUser.begin(), listOfUser.end(), userId);
    assert(userIt != listOfUser.end());
    listOfUser.erase(userIt);
    if (listOfUser.size() == 0)
    {
        MemoryChunk memoryChunk = *memoryChunkIt;
        m_UsedMemory.erase(memoryChunkIt);
        m_FreeMemory.push_back(memoryChunk);
        auto SortMemoryChunks = [](const auto& lhs, const auto& rhs) { return lhs.m_Begin < rhs.m_Begin; };
        std::sort(m_FreeMemory.begin(), m_FreeMemory.end(), SortMemoryChunks);
        CollapseRegions();
    }
    return true;
}

void SramAllocator::Free(UserId userId, uint32_t offset)
{
    bool success = TryFree(userId, offset);
    assert(success);
    ETHOSN_UNUSED(success);
}

std::string SramAllocator::DumpUsage() const
{
    std::string ret;
    ret += std::string("Sram Used Memory: \n");
    for (const auto& x : m_UsedMemory)
    {
        ret += std::string("range=") + std::to_string(x.m_Begin) + std::string("---") + std::to_string(x.m_End) + " " +
               x.m_Debug + std::string("\n");
    }
    ret += std::string("Sram Free Memory: \n");
    for (const auto& x : m_FreeMemory)
    {
        ret += std::string("range=") + std::to_string(x.m_Begin) + std::string("---") + std::to_string(x.m_End) +
               std::string("\n");
    }
    return ret;
}

size_t SramAllocator::GetAllocationSize() const
{
    return m_UsedMemory.size();
}

void SramAllocator::CollapseRegions()
{
    for (size_t i = m_FreeMemory.size() - 1; i >= 1; --i)
    {
        // Regions should never overlap otherwise something has gone horribly wrong
        assert(m_FreeMemory[i - 1].m_End <= m_FreeMemory[i].m_Begin);
        if (m_FreeMemory[i - 1].m_End == m_FreeMemory[i].m_Begin)
        {
            m_FreeMemory[i - 1].m_End = m_FreeMemory[i].m_End;
            m_FreeMemory.erase(m_FreeMemory.begin() + static_cast<ptrdiff_t>(i));
        }
    }
}

void SramAllocator::Reset()
{
    m_FreeMemory = { { 0, m_Capacity, {}, "" } };
    m_UsedMemory = {};
}

bool SramAllocator::IsFull()
{
    return m_FreeMemory.empty();
}

bool SramAllocator::IsEmpty()
{
    return m_UsedMemory.empty();
}

}    // namespace support_library
}    // namespace ethosn
