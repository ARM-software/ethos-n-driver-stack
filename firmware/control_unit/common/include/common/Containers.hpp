//
// Copyright © 2018-2020,2023 Arm Limited.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Utils.hpp"

#include <algorithm>
#include <type_traits>

namespace ethosn
{
namespace control_unit
{

template <typename T, uint32_t N>
class UninitializedArray
{
public:
    UninitializedArray()
    {
        // Leave m_Data uninitialized for performance
    }

    T& operator[](const uint32_t i)
    {
        // Don't assert for i = N so this can be used to get a pointer to the end
        ASSERT(i <= N);
        return reinterpret_cast<T&>(m_Data[i]);
    }

    const T& operator[](const uint32_t i) const
    {
        return const_cast<UninitializedArray&>(*this)[i];
    }

private:
    std::aligned_storage_t<sizeof(T), alignof(T)> m_Data[N];
};

template <typename T, uint32_t N>
class Vector
{
public:
    static constexpr uint32_t GetCapacity()
    {
        return N;
    }

    Vector()
        : m_Buffer()
        , m_Size(0)
    {}

    /// Lowercase for compatibility with C++ range-based for-loops.
    const T* begin() const
    {
        return &m_Buffer[0];
    }

    /// Lowercase for compatibility with C++ range-based for-loops.
    const T* end() const
    {
        return &m_Buffer[m_Size];
    }

    /// Lowercase for compatibility with C++ range-based for-loops.
    T* begin()
    {
        return const_cast<T*>(const_cast<const Vector*>(this)->begin());
    }

    /// Lowercase for compatibility with C++ range-based for-loops.
    T* end()
    {
        return const_cast<T*>(const_cast<const Vector*>(this)->end());
    }

    const T& operator[](const uint32_t idx) const
    {
        ASSERT_MSG(idx < m_Size, "Index out of range");
        return m_Buffer[idx];
    }

    T& operator[](const uint32_t idx)
    {
        return const_cast<T&>(const_cast<const Vector&>(*this)[idx]);
    }

    const T& Back() const
    {
        return (*this)[m_Size - 1U];
    }

    T& Back()
    {
        return const_cast<T&>(const_cast<const Vector*>(this)->Back());
    }

    void PushBack(std::remove_const_t<T> value)
    {
        ASSERT_MSG(!IsFull(), "Vector is full");
        ++m_Size;
        new (const_cast<std::remove_const_t<T>*>(&Back())) T(std::move(value));
    }

    void Remove(T* const it)
    {
        ASSERT_MSG(it < end(), "Index out of range");
        std::move(it + 1, end(), it);
        Back().~T();
        --m_Size;
    }

    void Remove(const uint32_t idx)
    {
        Remove(&m_Buffer[idx]);
    }

    // Unfortunately, this overload is necessary so idx = 0 is not ambiguously interpreted as nullptr
    void Remove(const int idx)
    {
        Remove(static_cast<uint32_t>(idx));
    }

    uint32_t Size() const
    {
        return m_Size;
    }

    bool IsFull() const
    {
        return m_Size >= N;
    }

    void Resize(const uint32_t newSize, const T& value = T{})
    {
        ASSERT_MSG(newSize < N, "Too large for capacity");
        // Remove elements outside new size
        for (T* it = &m_Buffer[newSize]; it < end(); ++it)
        {
            it->~T();
        }
        // Init new elements
        for (T* it = end(); it < &m_Buffer[newSize]; ++it)
        {
            new (it) T(value);
        }
        m_Size = newSize;
    }

private:
    UninitializedArray<T, N> m_Buffer;
    uint32_t m_Size;
};

}    // namespace control_unit
}    // namespace ethosn
