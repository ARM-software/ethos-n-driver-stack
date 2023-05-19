//
// Copyright © 2018-2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

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

    /// Attempts to allocate the given size.
    /// Returns whether allocating was successful and the offset of the requested size
    std::pair<bool, uint32_t>
        Allocate(uint32_t size, AllocationPreference pref = AllocationPreference::Start, std::string debugName = "");

    /// Frees the allocation at the given offset.
    /// If there is no allocation at the given offset then the behaviour is undefined (assert).
    void Free(uint32_t offset);

    /// Attempts to free the allocation at the given offset.
    /// If there is no allocation at the given offset then returns false, otherwise returns true.
    bool TryFree(uint32_t offset);

    /// Adds the given userId to the list of users of the given allocation.
    /// If there is no allocation at the given offset then the behaviour is undefined (assert).
    void IncrementReferenceCount(uint32_t offset);

    void Reset();

    bool IsEmpty();

    size_t GetAllocationSize() const;

private:
    struct MemoryChunk
    {
        uint32_t m_Begin;
        uint32_t m_End;
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
