//
// Copyright © 2018-2021 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include "Graph.hpp"

namespace ethosn
{
namespace support_library
{

enum class AllocationPreference
{
    Start,
    End,
};

// A simple allocator to be used to allocate data in SRAM.
// Assumes a small number of chunks allocated at once,
// thus iterating over the internal vectors is fast, and minimal fragmentation
class SramAllocator
{
public:
    using UserId = size_t;
    SramAllocator()
        : m_Capacity(0)
        , m_FreeMemory()
        , m_UsedMemory()
    {
        Reset();
    }

    SramAllocator(uint32_t capacity)
        : m_Capacity(capacity)
        , m_FreeMemory()
        , m_UsedMemory()
    {
        Reset();
    }

    SramAllocator(const SramAllocator& s) = default;

    SramAllocator& operator=(const SramAllocator& s);

    // Return whether allocating was successful and the offset of the requested size
    std::pair<bool, uint32_t> Allocate(UserId nodeId,
                                       uint32_t size,
                                       AllocationPreference pref = AllocationPreference::Start,
                                       std::string debugName     = "");

    bool Free(UserId userId, uint32_t offset);

    void IncrementReferenceCount(UserId userId, uint32_t offset);

    void Reset();

    std::string DumpUsage() const;

    bool IsFull();

    bool IsEmpty();

private:
    struct MemoryChunk
    {
        uint32_t m_Begin;
        uint32_t m_End;
        std::unordered_set<UserId> m_ListOfUsers;
        std::string m_Debug;
    };

    // Collapse regions of contiguous free memory into one chunk
    void CollapseRegions();

    uint32_t m_Capacity;

    // Pairs of numbers to represent the range of free contiguous memory left to allocate and the memory in use
    std::vector<MemoryChunk> m_FreeMemory;
    std::vector<MemoryChunk> m_UsedMemory;
};

}    // namespace support_library
}    // namespace ethosn
