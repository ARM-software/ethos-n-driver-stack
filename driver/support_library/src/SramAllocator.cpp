//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#include "SramAllocator.hpp"
#include "Utils.hpp"

#include <ethosn_utils/Macros.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>

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

std::pair<bool, uint32_t> SramAllocator::Allocate(uint32_t size, AllocationPreference pref, std::string debugName)
{
    if (pref == AllocationPreference::Start)
    {
        for (auto range = m_FreeMemory.begin(); range != m_FreeMemory.end(); ++range)
        {
            if (size <= range->m_End - range->m_Begin)
            {
                MemoryChunk chunk = { range->m_Begin, range->m_Begin + size, debugName };
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
                MemoryChunk chunk = { range->m_End - size, range->m_End, debugName };
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

bool SramAllocator::TryFree(uint32_t offset)
{
    auto MatchChunk = [offset](const auto& chunk) { return (offset == chunk.m_Begin); };
    // Remove the chunk from used memory and add it to the free memory.
    auto memoryChunkIt = std::find_if(m_UsedMemory.begin(), m_UsedMemory.end(), MatchChunk);

    if (memoryChunkIt == m_UsedMemory.end())
    {
        return false;
    }

    MemoryChunk memoryChunk = *memoryChunkIt;
    m_UsedMemory.erase(memoryChunkIt);
    m_FreeMemory.push_back(memoryChunk);
    auto SortMemoryChunks = [](const auto& lhs, const auto& rhs) { return lhs.m_Begin < rhs.m_Begin; };
    std::sort(m_FreeMemory.begin(), m_FreeMemory.end(), SortMemoryChunks);
    CollapseRegions();

    return true;
}

void SramAllocator::Free(uint32_t offset)
{
    bool success = TryFree(offset);
    assert(success);
    ETHOSN_UNUSED(success);
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
    m_FreeMemory = { { 0, m_Capacity, "" } };
    m_UsedMemory = {};
}

bool SramAllocator::IsEmpty()
{
    return m_UsedMemory.empty();
}

}    // namespace support_library
}    // namespace ethosn
